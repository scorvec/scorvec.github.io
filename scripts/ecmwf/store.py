#!/usr/bin/env python3
"""Robust, shared ECMWF open-data download manager.

ONE cache of AIFS-ENS / IFS-ENS GRIB, fetched once per (cycle, spec) and shared by every
pipeline (MJO AAM / torque / Hovmöller / SOI / RMM, the ensembles page, …). Replaces the
per-script bespoke downloaders.

Guarantees:
  • source fallback        AWS → Azure → ECMWF portal, rotated per retry
  • per-step chunked + resume   a stall re-fetches one step (~200 MB), not the whole pull;
                                completed step-parts persist → a failed run resumes
  • atomic canonical writes     stage → verify message count → os.replace; a partial GRIB
                                NEVER appears where consumers read (kills truncated stubs)
  • fetch-once locking          a per-spec flock so two processes don't double-download
  • completeness verification   members × steps × levels message count + a .json sidecar
  • prune                       old cycles dropped to bound disk

API (what other programs call):
    from store import ensure, open_ds, Spec, Cycle, registry
    p  = ensure(Cycle("20260604","00"), Spec("aifs-ens","pf","z","pl",(500,), STEPS))
    ds = open_ds(cycle, spec)          # xr.Dataset (cfgrib)

CLI:
    python store.py --date 20260604 --time 00            # fetch the whole registry
    python store.py --date 20260604 --time 00 --prune 4  # …and keep 4 newest cycles
"""
from __future__ import annotations
import argparse, contextlib, fcntl, glob, json, os, shutil, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path

from ecmwf.opendata import Client

# ── config ──────────────────────────────────────────────────────────────────────
CACHE = Path(os.environ.get("ECMWF_CACHE",
                            str(Path(__file__).resolve().parent / "cache")))
SOURCES = os.environ.get("ECMWF_SOURCES", "aws,azure,ecmwf").split(",")
# 2 streams (not 4): fewer concurrent byte-ranges → far less likely to trip S3 SlowDown
# in the first place. Per-step chunking already bounds the rest.
WORKERS = int(os.environ.get("ECMWF_DL_WORKERS", "2"))      # parallel member streams (pf)
PF_MEMBERS = 50
MIN_RATE = float(os.environ.get("ECMWF_DL_MIN_RATE", "40000"))   # B/s; below ⇒ stalled
# 240 s: multiurl recovers from a 503 with a ~120 s backoff, so the watchdog must be
# more patient than that — otherwise it aborts mid-recovery and re-triggers the throttle.
STALL_SECS = int(os.environ.get("ECMWF_DL_STALL_SECS", "240"))
TRIES = int(os.environ.get("ECMWF_DL_TRIES", "4"))

# Daily steps. STEPS = analysis + Day 1..15 (used by RMM/AAM/MMSF, which want the 0-h
# analysis). STEPS_FC = forecast days only (used by the SOI/Hovmöller, which don't).
STEPS = [0] + list(range(24, 361, 24))
STEPS_FC = list(range(24, 361, 24))
LEVELS_AAM = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


@dataclass(frozen=True)
class Cycle:
    date: str            # "YYYYMMDD"
    time: str            # "00" | "12"

    @property
    def tag(self) -> str:
        return f"{self.date}{self.time}z"

    @property
    def iso(self) -> str:
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:8]}"


@dataclass(frozen=True)
class Spec:
    model: str                       # "aifs-ens" | "ifs"
    type: str                        # "pf" | "cf" | "em"
    param: str                       # "u" | "z" | "2t" | "10u" | "10v" | "msl" | "sp"
    levtype: str                     # "pl" | "sfc"
    levelist: tuple = ()             # () for sfc
    steps: tuple = tuple(STEPS)

    @property
    def filename(self) -> str:
        lv = "-".join(str(x) for x in self.levelist) if self.levelist else self.levtype
        # step signature so two consumers wanting the same param but DIFFERENT step sets
        # (e.g. SOI's 24-360 vs AAM's 0-360) don't collide on one file
        s = self.steps
        sig = f"s{int(s[0])}-{int(s[-1])}x{len(s)}"
        return f"{self.type}_{self.param}_{lv}_{sig}.grib2"

    def members(self):               # member set for the message-count expectation
        return list(range(1, PF_MEMBERS + 1)) if self.type == "pf" else None

    def n_expected(self) -> int:
        n = len(self.members()) if self.type == "pf" else 1
        n *= len(self.steps)
        n *= len(self.levelist) if self.levtype == "pl" else 1
        return n


def path(cycle: Cycle, spec: Spec) -> Path:
    return CACHE / cycle.tag / spec.model / spec.filename


# ── GRIB helpers ─────────────────────────────────────────────────────────────────
def count_msgs(p: str | Path) -> int:
    import eccodes as ec
    n = 0
    with open(p, "rb") as f:
        while True:
            g = ec.codes_grib_new_from_file(f)
            if g is None:
                break
            n += 1; ec.codes_release(g)
    return n


def _complete(p: Path, expected: int) -> bool:
    """Trust the sidecar if it records ≥expected for this exact spec; else count once."""
    if not p.exists() or p.stat().st_size == 0:
        return False
    side = p.with_suffix(p.suffix + ".json")
    if side.exists():
        try:
            meta = json.loads(side.read_text())
            if int(meta.get("messages", -1)) >= expected:
                return True
        except Exception:                                  # noqa: BLE001
            pass
    try:
        ok = count_msgs(p) >= expected
    except Exception:                                      # noqa: BLE001
        return False
    if ok:                                                 # backfill a sidecar
        _sidecar(p, expected, "unknown")
    return ok


def _sidecar(p: Path, messages: int, src: str) -> None:
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(
        {"messages": messages, "source": src, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


# ── low-level retrieval (ported from download_aifs, staging-aware) ───────────────
def _dl_bytes(target: str) -> int:
    return sum(os.path.getsize(p) for p in [target, *glob.glob(target + ".part*")] if os.path.exists(p))


def _clean(target: str) -> None:
    for p in [target, *glob.glob(target + ".part*")]:
        try:
            os.remove(p)
        except OSError:
            pass


def _single(req: dict, target: str, src: str) -> None:
    Client(source=src).retrieve(target=target, **req)


def _parallel_once(req: dict, target: str, src: str, members, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor
    groups = [g for g in (members[i::workers] for i in range(workers)) if g]
    parts = [f"{target}.part{i}" for i in range(len(groups))]
    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        list(ex.map(lambda gp: _single({**req, "number": gp[0]}, gp[1], src), zip(groups, parts)))
    with open(target, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(p)


def _watch(do_fn, target: str):
    done = threading.Event(); box = {"e": None}

    def _run():
        try:
            do_fn()
        except Exception as e:                             # noqa: BLE001
            box["e"] = e
        finally:
            done.set()
    threading.Thread(target=_run, daemon=True).start()
    last = 0; slow = 0
    while not done.wait(15):
        cur = _dl_bytes(target); rate = (cur - last) / 15.0; last = cur
        slow = slow + 15 if rate < MIN_RATE else 0
        if slow >= STALL_SECS:
            return False, TimeoutError(f"stalled <{MIN_RATE/1000:.0f} kB/s for {STALL_SECS}s")
    return box["e"] is None, box["e"]


def _robust(req: dict, target: str, parallel: bool, members, expected: int) -> str:
    label = os.path.basename(target)
    use_parallel = parallel and WORKERS >= 2 and members and len(members) >= 2
    err = None
    for attempt in range(1, TRIES + 1):
        src = SOURCES[(attempt - 1) % len(SOURCES)]
        _clean(target)
        if use_parallel:
            do = lambda s=src: _parallel_once(req, target, s, members, WORKERS)
        elif parallel:
            do = lambda s=src: _single({**req, "number": members}, target, s)
        else:
            do = lambda s=src: _single(req, target, s)
        ok, err = _watch(do, target)
        if ok:
            try:
                got = count_msgs(target)
            except Exception:                              # noqa: BLE001
                got = -1
            if got >= expected:
                return src
            err = RuntimeError(f"incomplete GRIB: {got}/{expected} msgs")
        print(f"    {label}: {repr(err)[:55]} — retry {attempt}/{TRIES} (mirror)", flush=True)
    _clean(target)
    raise err if err is not None else RuntimeError(f"{label}: failed after {TRIES} tries")


def _robust_chunked(req: dict, target: str, parallel: bool, members) -> str:
    """Per-step download with resume: each step → its own part file (kept if complete),
    stall re-fetches only that step, parts assembled in order at the end. Only the heavy
    member (pf) pulls are chunked — cf/em are small enough that one retrieve beats the
    per-step connection overhead."""
    steps = req.get("step")
    if not parallel or not isinstance(steps, (list, tuple)) or len(steps) <= 1:
        exp = _msgs_for(req, members)
        return _robust(req, target, parallel, members, exp)
    cdir = Path(f"{target}.parts"); cdir.mkdir(parents=True, exist_ok=True)
    parts, src = [], None
    for st in steps:
        pf = cdir / f"s{int(st):03d}.grib2"
        exp = _msgs_for({**req, "step": [st]}, members)
        if pf.exists():
            try:
                if count_msgs(str(pf)) >= exp:
                    parts.append(pf); continue
            except Exception:                              # noqa: BLE001
                pass
        src = _robust({**req, "step": st}, str(pf), parallel, members, exp)
        parts.append(pf)
    with open(target, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    shutil.rmtree(cdir, ignore_errors=True)
    return src or "chunked"


def _msgs_for(req: dict, members) -> int:
    n = len(members) if members else 1
    for k in ("step", "levelist"):
        v = req.get(k)
        if isinstance(v, (list, tuple)):
            n *= len(v)
    return n


# ── the public store ─────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _lock(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    lf = open(p.parent / (p.name + ".lock"), "w")
    fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN); lf.close()


def _to_req(cycle: Cycle, spec: Spec) -> dict:
    r = dict(model=spec.model, date=cycle.iso, time=int(cycle.time), stream="enfo",
             type=spec.type, param=spec.param, levtype=spec.levtype, step=list(spec.steps))
    if spec.levtype == "pl":
        r["levelist"] = [int(x) for x in spec.levelist]
    return r


def ensure(cycle: Cycle, spec: Spec) -> Path:
    """Fetch (if needed), verify, and return the canonical path for (cycle, spec).
    Idempotent + concurrency-safe; never returns a partial file."""
    p = path(cycle, spec); expected = spec.n_expected()
    if _complete(p, expected):
        return p
    with _lock(p):
        if _complete(p, expected):                         # someone else just finished
            return p
        stage = p.parent / (".stage_" + p.name)
        _clean(str(stage)); shutil.rmtree(f"{stage}.parts", ignore_errors=True)
        print(f"  ECMWF {cycle.tag} {spec.model}/{spec.filename}: fetching "
              f"({expected} msgs) …", flush=True)
        src = _robust_chunked(_to_req(cycle, spec), str(stage), spec.type == "pf", spec.members())
        got = count_msgs(str(stage))
        if got < expected:
            _clean(str(stage))
            raise RuntimeError(f"{spec.filename}: incomplete after fetch ({got}/{expected})")
        _sidecar(stage, got, src)
        os.replace(stage.with_suffix(stage.suffix + ".json"), p.with_suffix(p.suffix + ".json"))
        os.replace(stage, p)                               # atomic publish
        print(f"  ECMWF {cycle.tag} {spec.model}/{spec.filename}: ✓ {got} msgs via {src}", flush=True)
    return p


def open_ds(cycle: Cycle, spec: Spec, **backend_kwargs):
    import xarray as xr
    bk = {"indexpath": ""}; bk.update(backend_kwargs)
    return xr.open_dataset(ensure(cycle, spec), engine="cfgrib", backend_kwargs=bk)


def open_members(cycle: Cycle, spec: Spec):
    """Member-aware open (handles cf+pf split): returns the DataArray with a 'number' dim."""
    import cfgrib
    dss = cfgrib.open_datasets(ensure(cycle, spec), backend_kwargs={"indexpath": ""})
    cand = [d for d in dss if "number" in d.dims] or dss
    return cand[0][list(cand[0].data_vars)[0]]


# ── registry: everything a 00Z/12Z cycle needs (for the pre-fetcher) ─────────────
def registry() -> list[Spec]:
    S = tuple(STEPS)        # 0..360 (with analysis) — RMM, AAM, MMSF, ens
    F = tuple(STEPS_FC)     # 24..360 (forecast only) — SOI, Hovmöller
    return [
        # MJO RMM — AIFS u @ 200/850
        Spec("aifs-ens", "pf", "u", "pl", (200, 850), S),
        Spec("aifs-ens", "cf", "u", "pl", (200, 850), S),
        # AAM / torque / zonal — AIFS u @ 13 levels + surface pressure
        Spec("aifs-ens", "pf", "u", "pl", LEVELS_AAM, S),
        Spec("aifs-ens", "cf", "u", "pl", LEVELS_AAM, S),
        Spec("aifs-ens", "pf", "sp", "sfc", (), S),
        Spec("aifs-ens", "cf", "sp", "sfc", (), S),
        # Hovmöller — AIFS + IFS 10u (forecast days only)
        Spec("aifs-ens", "pf", "10u", "sfc", (), F),
        Spec("aifs-ens", "cf", "10u", "sfc", (), F),
        Spec("ifs", "pf", "10u", "sfc", (), F),
        # SOI — AIFS + IFS msl (forecast days only)
        Spec("aifs-ens", "pf", "msl", "sfc", (), F),
        Spec("aifs-ens", "cf", "msl", "sfc", (), F),
        Spec("ifs", "pf", "msl", "sfc", (), F),
        # torque map — AIFS 10v (forecast days; 10u/msl/sp above are reused)
        Spec("aifs-ens", "pf", "10v", "sfc", (), F),
        Spec("aifs-ens", "cf", "10v", "sfc", (), F),
        # MMSF — AIFS analysis (step 0) meridional wind @ 13 levels
        Spec("aifs-ens", "cf", "v", "pl", LEVELS_AAM, (0,)),
        # ensembles page + general reuse — AIFS z @ 500 + 2 m temperature, full
        # ensemble (cf control + 50 pf members), with the analysis Day 0.
        Spec("aifs-ens", "pf", "z", "pl", (500,), S),
        Spec("aifs-ens", "cf", "z", "pl", (500,), S),
        Spec("aifs-ens", "pf", "2t", "sfc", (), S),
        Spec("aifs-ens", "cf", "2t", "sfc", (), S),
    ]


def _cycle_init(tag: str):
    """datetime of a cache cycle dir name like '2026060400z' (UTC, tz-naive)."""
    from datetime import datetime
    return datetime.strptime(tag[:10], "%Y%m%d%H")


def prune(keep: int = 0, days: float = 0) -> None:
    """Drop old cycle dirs. keep>0 keeps the N newest; days>0 drops cycles whose
    init time is more than `days` old (the default retention policy)."""
    from datetime import datetime, timedelta
    cycles = sorted([d for d in CACHE.glob("*z") if d.is_dir()])
    drop = set()
    if keep > 0:
        drop |= set(cycles[:-keep])
    if days > 0:
        cutoff = datetime.utcnow() - timedelta(days=days)
        for d in cycles:
            try:
                if _cycle_init(d.name) < cutoff:
                    drop.add(d)
            except ValueError:
                pass                                       # not a cycle dir — leave it
    for d in sorted(drop):
        shutil.rmtree(d, ignore_errors=True)
        print(f"  pruned {d.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--prune", type=int, default=0, help="keep N newest cycles (0 = no prune)")
    ap.add_argument("--prune-days", type=float, default=0, help="drop cycles older than D days")
    a = ap.parse_args()
    cyc = Cycle(a.date, a.time)
    print(f"== ECMWF store: stocking {cyc.tag} → {CACHE} ==", flush=True)
    ok = 0
    for spec in registry():
        try:
            ensure(cyc, spec); ok += 1
        except Exception as e:                             # noqa: BLE001
            print(f"  {spec.model}/{spec.filename}: FAILED ({repr(e)[:70]})", flush=True)
    print(f"stocked {ok}/{len(registry())} specs for {cyc.tag}", flush=True)
    if a.prune or a.prune_days:
        prune(a.prune, a.prune_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
