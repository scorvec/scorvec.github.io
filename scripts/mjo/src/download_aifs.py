"""
Download AIFS-ENS fields from ECMWF open data.

Variables retrieved:
  - u : zonal wind at 850 and 200 hPa

Note: AIFS-ENS does not output OLR/TTR, so only wind-based RMM is computed.

The AIFS ensemble ('aifs-ens') runs twice daily (00Z / 12Z) with
50 perturbed members + 1 control, stream 'enfo'.

Usage:
    python src/download_aifs.py                        # latest available run
    python src/download_aifs.py --date 20240601 --time 00
"""

import argparse
import glob
import os
import shutil
import sys
import threading
from pathlib import Path

from ecmwf.opendata import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                    # shared ECMWF download manager


DATA_DIR = Path(__file__).parent.parent / "data" / "aifs"

# Steps to day 15. Default DAILY (24-hourly): the RMM only needs daily values
# (it builds daily means via step//24 groupby, so one sample/day is enough), and
# daily steps cut the ~1 MB/s-bandwidth-bound download ~4x (≈4 GB → ≈1 GB).
# Set AIFS_STEP_HOURS=6 for the old 4-samples/day behavior if ever needed.
# Step 0 (the analysis) is included so lead_day 0 exists — the zero-lag "truth"
# point archived to obs_history (daily steps otherwise start at lead_day 1).
_STEP_HOURS = int(os.environ.get("AIFS_STEP_HOURS", "24"))
STEPS = [0] + list(range(_STEP_HOURS, 361, _STEP_HOURS))


def rmm_steps(init_time) -> list:
    """Lead steps that land on 00Z VALID times for any init hour, so the RMM samples one
    consistent 00Z point per day across BOTH 00Z and 12Z runs (run-to-run comparable —
    same forecast valid points). 00Z init → 0,24,…,360; 12Z init → 0,12,36,…,348 (forecast
    leads offset +12 h to reach the next 00Z). Step 0 (the init analysis) is always kept —
    it's the zero-lag 'truth' archived to obs_history. Same daily resolution / bandwidth."""
    first = (24 - int(init_time) % 24) % 24                   # 0 for 00Z, 12 for 12Z
    anchored = list(range(first, 361, 24))                    # forecast leads valid at 00Z
    return anchored if first == 0 else [0] + anchored         # 12Z: prepend the 12Z analysis

# aws/azure blobs + the ECMWF portal (which enforces a connection limit / HTTP 429).
# google is EXCLUDED: it only partially/laggingly mirrors the latest cycle (400s on the
# perturbed sp/2t, z500 and pl data it hasn't synced yet) → just retry noise on daily
# latest-cycle pulls. Override with the AIFS_SOURCES env var.
SOURCES = [s.strip() for s in
           os.environ.get("AIFS_SOURCES", "aws,azure,ecmwf").split(",") if s.strip()]


def _archive_grib(target: str, req: dict) -> None:
    """If MJO_GRIB_ARCHIVE is set (laptop only), keep a copy of each downloaded
    GRIB under <archive>/<date>_<time>z/ for easy access. Skips concat chunks."""
    arch = os.environ.get("MJO_GRIB_ARCHIVE")
    if not arch or ".part" in os.path.basename(target) or not os.path.exists(target):
        return
    try:
        d = str(req.get("date", "unknown")); t = f"{int(req.get('time', 0)):02d}"
        dest_dir = Path(arch) / f"{d}_{t}z"; dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / os.path.basename(target)
        if dest.exists():
            return
        try:
            os.link(target, dest)                  # hard link — no extra disk on same volume
        except OSError:
            import shutil; shutil.copy2(target, dest)   # cross-device fallback
    except Exception as e:  # noqa: BLE001 — archiving must never break a build
        print(f"  (grib archive skipped: {e})", flush=True)


# ── robust retrieval: per-attempt mirror rotation + a throughput watchdog ──
# Open-data per-connection bandwidth is the bottleneck (~0.85 MB/s) but the link
# has headroom, so the perturbed ensemble is split across N concurrent streams
# (~3x faster). A connection can also HANG at a trickle without ever erroring
# (seen: 1.6 kB/s for hours); the watchdog aborts any download that sustains
# < DL_MIN_RATE for DL_STALL_SECS and retries with fresh streams on the next
# mirror, so one stalled stream can no longer wedge the whole pipeline.
DL_WORKERS = int(os.environ.get("AIFS_DL_WORKERS", "4"))
PF_MEMBERS = 50                                  # AIFS-ENS / IFS-ENS enfo pf count
DL_MIN_RATE = float(os.environ.get("AIFS_DL_MIN_RATE", "40000"))   # bytes/s; sustained-below ⇒ stalled
DL_STALL_SECS = int(os.environ.get("AIFS_DL_STALL_SECS", "75"))    # tolerate slow this long, then retry
DL_TRIES = int(os.environ.get("AIFS_DL_TRIES", "4"))


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
    """Fetch a perturbed request over `workers` concurrent member-group streams
    (one mirror), then concatenate the self-contained GRIB messages."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    groups = [g for g in (members[i::workers] for i in range(workers)) if g]
    parts = [f"{target}.part{i}" for i in range(len(groups))]
    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        list(ex.map(lambda gp: _single({**req, "number": gp[0]}, gp[1], src), zip(groups, parts)))
    with open(target, "wb") as out:                              # concatenate parts
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(p)


def _watch(do_fn, target: str):
    """Run do_fn() (writes target [+ .partN]) in a daemon thread; return (ok, err).
    ok=False if throughput stays < DL_MIN_RATE for DL_STALL_SECS — the stalled
    connection is abandoned; its now-unlinked partials write to discarded inodes
    (unix), so the retry's fresh streams never collide."""
    done = threading.Event(); box = {"e": None}

    def _run():
        try:
            do_fn()
        except Exception as e:                              # noqa: BLE001
            box["e"] = e
        finally:
            done.set()
    threading.Thread(target=_run, daemon=True).start()
    last = 0; slow = 0
    while not done.wait(15):
        cur = _dl_bytes(target); rate = (cur - last) / 15.0; last = cur
        slow = slow + 15 if rate < DL_MIN_RATE else 0
        if slow >= DL_STALL_SECS:
            return False, TimeoutError(f"stalled <{DL_MIN_RATE / 1000:.0f} kB/s for {DL_STALL_SECS}s")
    return box["e"] is None, box["e"]


def _count_msgs(target: str) -> int:
    """Number of GRIB messages in a file (fast, no decode)."""
    import eccodes as ec
    n = 0
    with open(target, "rb") as f:
        while True:
            g = ec.codes_grib_new_from_file(f)
            if g is None:
                break
            n += 1; ec.codes_release(g)
    return n


def _expected_msgs(req: dict, members) -> int:
    """Messages a complete retrieve must contain = members × steps × params × levels.
    A throttled parallel stream can silently drop messages (e.g. one worker's early
    steps), leaving all-NaN fields downstream — this lets _robust detect that."""
    exp = len(members) if members else 1
    for k in ("step", "param", "levelist"):
        v = req.get(k)
        exp *= len(v) if isinstance(v, (list, tuple)) else (1 if v is not None else 1)
    return exp


def _robust(req: dict, target: str, parallel: bool, members=None, workers: int = None) -> str:
    label = os.path.basename(target)
    workers = workers or DL_WORKERS
    members = list(members) if members is not None else list(range(1, PF_MEMBERS + 1))
    use_parallel = parallel and workers >= 2 and len(members) >= 2
    # expected message count (members carry through for the parallel/tiny-pf paths;
    # a non-member single retrieve falls back to whatever req["number"] implies).
    exp_members = members if parallel else ([req["number"]] if "number" in req
                                            and not isinstance(req["number"], (list, tuple))
                                            else req.get("number"))
    expected = _expected_msgs(req, exp_members)
    err = None
    for attempt in range(1, DL_TRIES + 1):
        src = SOURCES[(attempt - 1) % len(SOURCES)]          # rotate mirror each retry
        _clean(target)
        if use_parallel:
            do = lambda s=src: _parallel_once(req, target, s, members, workers)
        elif parallel:                                       # tiny pf request → single stream
            do = lambda s=src: _single({**req, "number": members}, target, s)
        else:
            do = lambda s=src: _single(req, target, s)
        ok, err = _watch(do, target)
        if ok:                                               # verify completeness, not just no-error
            try:
                got = _count_msgs(target)
            except Exception:                                # noqa: BLE001
                got = -1
            if got >= expected:
                _archive_grib(target, req)
                return f"{src}×{workers}" if use_parallel else src
            err = RuntimeError(f"incomplete GRIB: {got}/{expected} messages")
        print(f"  {label}: {repr(err)[:60]} — retry {attempt}/{DL_TRIES} (mirror rotate)", flush=True)
    _clean(target)
    raise err if err is not None else RuntimeError(f"{label}: failed after {DL_TRIES} tries")


def _robust_chunked(req: dict, target: str, parallel: bool, members=None, workers=None) -> str:
    """Download a multi-step request ONE STEP AT A TIME, keeping each completed step's
    GRIB on disk. A stalled/throttled stream then only re-fetches the step it died on
    (~200 MB) instead of restarting the whole ~3 GB pull — and, because the per-step
    parts persist (the chunk dir is removed only on full success), a failed cycle
    RESUMES from where it left off on the next attempt/run. Single-step (or scalar-step)
    requests fall straight through to _robust."""
    steps = req.get("step")
    if not isinstance(steps, (list, tuple)) or len(steps) <= 1:
        return _robust(req, target, parallel, members, workers)
    mem = (list(members) if members is not None else list(range(1, PF_MEMBERS + 1))) if parallel else \
          ([req["number"]] if "number" in req and not isinstance(req["number"], (list, tuple))
           else req.get("number"))
    cdir = Path(f"{target}.parts"); cdir.mkdir(parents=True, exist_ok=True)
    parts, src_used = [], None
    for st in steps:
        pf = cdir / f"s{int(st):03d}.grib2"
        exp = _expected_msgs({**req, "step": [st]}, mem)
        if pf.exists():
            try:
                if _count_msgs(str(pf)) >= exp:
                    parts.append(pf); continue                 # already have this step
            except Exception:                                  # noqa: BLE001
                pass
        src_used = _robust({**req, "step": st}, str(pf), parallel, members, workers)
        parts.append(pf)
    with open(target, "wb") as out:                            # assemble in step order
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    if _count_msgs(target) >= _expected_msgs(req, mem):
        shutil.rmtree(cdir, ignore_errors=True)                # success → drop the parts
        return src_used or "chunked"
    raise RuntimeError(f"{os.path.basename(target)}: chunked assembly incomplete")


def _retrieve(req: dict, target: str) -> str:
    """Robust single-stream retrieve (watchdog + mirror-rotating retry; per-step resume)."""
    return _robust_chunked(req, target, parallel=False)


def retrieve_parallel(req: dict, target: str, members=None, workers: int = None) -> str:
    """Robust parallel (multi-member-stream) retrieve; per-step resume on stall."""
    return _robust_chunked(req, target, parallel=True, members=members, workers=workers)


def _retrieve_probe(req: dict, target: str) -> None:
    """Single-shot probe (no watchdog/retry), used by latest_run."""
    last = None
    for src in SOURCES:
        try:
            _single(req, target, src); return
        except Exception as e:                              # noqa: BLE001
            last = e
    raise last if last is not None else RuntimeError("probe failed")


def latest_run() -> tuple[str, str]:
    """Return (date, time) for the most recent available AIFS-ENS run.

    The ecmwf-opendata client prints an attribution banner / progress to stdout
    on each retrieve; we suppress stdout during the probe so callers that parse
    this function's output get only what *they* print.
    """
    import contextlib
    import datetime
    import tempfile
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    for offset in range(0, 4):
        for run_time in ("12", "00"):
            date = today_utc - datetime.timedelta(days=offset)
            date_str = date.strftime("%Y%m%d")
            try:
                with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmp, \
                        open(os.devnull, "w") as _dn, \
                        contextlib.redirect_stdout(_dn):
                    # Probe a PERTURBED member (pf #50), not the control: ECMWF publishes the
                    # control before the 50 perturbed members, so a cf-only probe can call a
                    # cycle "available" while the ensemble the RMM actually needs is still
                    # landing — and the full 50-member download then fails. Requiring a pf
                    # member to be present makes the runner wait (next hourly poll) for a
                    # delayed/incrementally-published cycle instead of failing on partial data.
                    _retrieve_probe(dict(model="aifs-ens", date=date_str, time=int(run_time),
                                         stream="enfo", type="pf", number=50, levtype="pl",
                                         levelist=[850], param="u", step=6), tmp.name)
                return date_str, run_time
            except Exception:
                continue
    raise RuntimeError("Could not find a recent available AIFS-ENS run (checked last 4 days)")


def download(date: str, time: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"aifs_{date}_{time}z"

    print(f"Downloading AIFS-ENS {date} {time}Z via shared store …")

    # u on the 200/850 levels the RMM needs — a light ~1 GB pull, kept separate from the
    # heavy 13-level AAM download so the MJO critical path is never blocked behind it. The
    # AAM builder fetches the other 11 levels and concatenates 200/850 back in (no dup).
    # The RMM reader expects data/aifs/<stem>.{pf,cf}.u.grib2, so hardlink the cache file there.
    cyc = ecmwf.Cycle(date, time); steps = tuple(rmm_steps(time))   # 00Z-anchored valid times
    for typ in ("pf", "cf"):
        src = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl", ecmwf.LEVELS_RMM, steps))
        dst = out_dir / f"{stem}.{typ}.u.grib2"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)                       # free same-filesystem pointer
        except OSError:
            shutil.copy2(src, dst)
        print(f"  {typ} u-wind: {dst.name} -> {Path(src).name}")

    # tp for the precip pseudo-OLR channel of the RMM (accumulated from init;
    # step 0 is skipped — the accumulation is zero there). Non-fatal: on any
    # failure the RMM falls back to wind-only, so the critical path survives
    # a tp outage on the mirrors.
    tp_steps = tuple(s for s in steps if s > 0)
    for typ in ("pf", "cf"):
        try:
            src = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "tp", "sfc", (), tp_steps))
            dst = out_dir / f"{stem}.{typ}.tp.grib2"
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            print(f"  {typ} tp: {dst.name} -> {Path(src).name}")
        except Exception as e:                      # noqa: BLE001
            print(f"  {typ} tp unavailable ({repr(e)[:60]}) — RMM will be wind-only")

    print("Download complete.")


def download_ifs(date: str, time: str, out_dir: Path) -> bool:
    """Fetch IFS-ENS u@850/200 + tp for the RMM comparison overlay.

    IFS-ENS is disseminated ~1-2 h AFTER AIFS-ENS, so this is best-effort:
    returns True only when both wind files landed (tp stays optional — the
    RMM falls back to wind-only per model). The caller re-tries on later
    polls via the plots/<stem>.png.missing sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ifs_{date}_{time}z"
    cyc = ecmwf.Cycle(date, time)
    steps = tuple(rmm_steps(time))
    # pf ONLY: IFS 0.25-deg enfo open data serves no separate cf through this
    # path (the -enfo-cf index 404s and the client finds no cf entries;
    # verified 2026-08-16) — 50 perturbed members are ample for the overlay.
    ok = True
    try:
        src = ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "u", "pl",
                                           ecmwf.LEVELS_RMM, steps))
        dst = out_dir / f"{stem}.pf.u.grib2"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        print(f"  IFS pf u-wind: {dst.name}")
    except Exception as e:                          # noqa: BLE001
        print(f"  IFS pf u unavailable ({repr(e)[:60]})")
        ok = False
    if ok:
        try:
            tp_steps = tuple(s for s in steps if s > 0)
            src = ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "tp", "sfc", (), tp_steps))
            dst = out_dir / f"{stem}.pf.tp.grib2"
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            print(f"  IFS pf tp: {dst.name}")
        except Exception as e:                      # noqa: BLE001
            print(f"  IFS pf tp unavailable ({repr(e)[:60]}) — wind-only")
    return ok



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: latest available)")
    parser.add_argument("--time", default=None, help="00 or 12 (default: latest available)")
    parser.add_argument("--out-dir", default="data/aifs")
    args = parser.parse_args()

    if args.date is None or args.time is None:
        print("No date/time specified — finding latest available AIFS-ENS run …")
        date, time = latest_run()
        print(f"  Latest run: {date} {time}Z")
    else:
        date, time = args.date, args.time

    download(date, time, Path(args.out_dir))


if __name__ == "__main__":
    main()
