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
try:
    import budget as _budget
except ImportError:
    _budget = None

# ── config ──────────────────────────────────────────────────────────────────────
CACHE = Path(os.environ.get("ECMWF_CACHE",
                            str(Path(__file__).resolve().parent / "cache")))
# Mirror pool: google/aws/azure/ecmwf. google RE-ADDED 2026-07-26: benchmarks put it at
# 30-55 MB/s vs aws ~10 / ecmwf ~3.5 — by far the fastest when it has the file. It still
# syncs the newest cycle with some lag (a 404 there is normal early in a cycle), but the
# per-range mirror rotation makes that a ~0.1 s fall-through, not a failure.
#
# ORDERING: next_mirror_order() prefers measured speed (an EMA of MB/s per mirror,
# accumulated from every completed fetch — see _note_speed) with unmeasured mirrors
# probed first; with no measurements yet it falls back to the old round-robin. Any
# mirror that threw 503s last run is demoted to the end. The runner calls
# next_mirror_order() and exports ECMWF_SOURCES; an explicit ECMWF_SOURCES env always
# wins (manual override). Module import is read-only (uses the last order) so
# non-download importers (dashboard, consumers) don't advance the rotation.
_BASE_MIRRORS = ["google", "aws", "azure", "ecmwf"]
_ROT_STATE = Path(__file__).resolve().parent / ".mirror_rotation.json"
_ROT_LOCK = threading.Lock()


def _read_rot() -> tuple[list, set]:
    try:
        st = json.loads(_ROT_STATE.read_text())
        order = [m for m in st.get("order", []) if m in _BASE_MIRRORS]
        order += [m for m in _BASE_MIRRORS if m not in order]   # pick up newly-added mirrors
        return order, set(st.get("throttled", []))
    except Exception:                                           # noqa: BLE001
        return list(_BASE_MIRRORS), set()


def _read_speed() -> dict:
    try:
        return {m: float(v) for m, v in
                json.loads(_ROT_STATE.read_text()).get("speed", {}).items()}
    except Exception:                                           # noqa: BLE001
        return {}


def _note_speed(src: str, mbps: float) -> None:
    """Fold a completed fetch's throughput into the per-mirror speed EMA that
    next_mirror_order() sorts by. `src` is the sidecar label ("v2:google",
    "aws+azure", …) — attribution to the first/named mirror(s) is approximate
    but converges over many fetches."""
    names = [s for s in src.replace("v2:", "").split("+") if s in _BASE_MIRRORS]
    if not names or mbps <= 0:
        return
    try:
        with _ROT_LOCK:
            st = json.loads(_ROT_STATE.read_text()) if _ROT_STATE.exists() else {}
            sp = st.setdefault("speed", {})
            for n in names:
                sp[n] = round(0.7 * float(sp.get(n, mbps)) + 0.3 * mbps, 2)
            _ROT_STATE.write_text(json.dumps(st))
    except Exception:                                           # noqa: BLE001
        pass


def next_mirror_order() -> str:
    """Order mirrors for this run: measured-fastest first (speed EMA from completed
    fetches), unmeasured mirrors ahead of measured ones so a new mirror gets probed;
    round-robin fallback while no speeds exist. Mirrors that threw 503s last run are
    demoted to the end. Resets the throttle flags. Returns the comma-joined order for
    ECMWF_SOURCES. The RUNNER calls this once per pipeline run."""
    order, bad = _read_rot()
    speed = _read_speed()
    if speed:
        # unmeasured mirrors get an optimistic prior (90% of the best) so they're
        # probed ahead of known-slow mirrors but never ahead of the known-fastest
        prior = 0.9 * max(speed.values())
        order = sorted(order, key=lambda m: -speed.get(m, prior))
    else:
        order = order[1:] + order[:1]                           # round-robin
    # google is PINNED first (user policy 2026-07-26: always favor the google
    # mirror — it benchmarks 3-15x faster); the speed order applies to the rest.
    # A throttled google still demotes below for the run (next clause).
    if "google" in order:
        order = ["google"] + [m for m in order if m != "google"]
    order = [m for m in order if m not in bad] + [m for m in order if m in bad]  # demote throttled
    try:
        with _ROT_LOCK:
            st = json.loads(_ROT_STATE.read_text()) if _ROT_STATE.exists() else {}
            st.update(order=order, throttled=[])
            _ROT_STATE.write_text(json.dumps(st))
    except Exception:                                           # noqa: BLE001
        pass
    return ",".join(order)


def _note_throttle_src(src: str) -> None:
    """Record that `src` threw a 503 this run → next run's rotation demotes it."""
    if not src or src not in _BASE_MIRRORS:
        return
    try:
        with _ROT_LOCK:
            st = json.loads(_ROT_STATE.read_text()) if _ROT_STATE.exists() else \
                {"order": list(_BASE_MIRRORS), "throttled": []}
            t = st.setdefault("throttled", [])
            if src not in t:
                t.append(src); _ROT_STATE.write_text(json.dumps(st))
    except Exception:                                           # noqa: BLE001
        pass


if os.environ.get("ECMWF_SOURCES"):
    SOURCES = os.environ["ECMWF_SOURCES"].split(",")           # explicit override wins
else:
    SOURCES, _ = _read_rot()                                   # read-only: last order (no advance)
MULTISOURCE = os.environ.get("ECMWF_MULTISOURCE", "1") != "0"   # spread steps across mirrors
WORKERS = int(os.environ.get("ECMWF_DL_WORKERS", "1"))      # parallel member streams (pf)
                                                            # 1 by default: fewer concurrent
                                                            # range-requests → fewer S3 503s
PER_SRC = int(os.environ.get("ECMWF_DL_PER_SRC", "2"))      # hard cap: in-flight retrieves / mirror
# The open-data portal caps a client at 500 SIMULTANEOUS connections, and ecmwf-opendata
# opens ~1 connection per GRIB message — so a single big retrieve (e.g. a 1600- or 3200-
# message pf request) blows past it on ANY mirror. Cap both: ≤ CHUNK_MSGS messages per
# retrieve, and ≤ MAX_CONN total connections in flight at once (≈ MAX_CONN/CHUNK_MSGS
# concurrent retrieves). Kept well under 500 to avoid throttling / IP bans.
MAX_CONN = int(os.environ.get("ECMWF_MAX_CONN", "400"))
CHUNK_MSGS = int(os.environ.get("ECMWF_CHUNK_MSGS", "200"))
PF_MEMBERS = 50
# The heavy 12-level AAM u-pull only ever feeds ensemble MEAN + SPREAD (AAM, torque,
# zonal, jets), so half the members carry it fine: 25 of 50 halves the ~6 GB file with
# a √2 bump in mean noise on an already-smooth global integral. Set 0 (or the env) to
# restore the full ensemble. Encoded in the filename ("…m25"), so a subset cache file
# never masquerades as a full one.
AAM_PF_MEMBERS = int(os.environ.get("ECMWF_AAM_MEMBERS", "25"))
MIN_RATE = float(os.environ.get("ECMWF_DL_MIN_RATE", "40000"))   # B/s; below ⇒ stalled
# With fail-fast retries (below) a throttled mirror raises in ~30 s and we rotate, so the
# watchdog no longer has to out-wait multiurl's old 120 s backoff.
STALL_SECS = int(os.environ.get("ECMWF_DL_STALL_SECS", "120"))
TRIES = int(os.environ.get("ECMWF_DL_TRIES", "6"))          # store-level mirror rotations
# Speed-based mirror switching: a per-step transfer steadily below SLOW_RATE for SLOW_SECS
# (past a SLOW_GRACE ramp-up) is abandoned so _robust rotates to a hopefully-faster mirror.
# Only honoured while we still have an untried mirror this spec — after sampling them all we
# accept the best available rather than churn (or fail) when every mirror is just busy.
SLOW_RATE = float(os.environ.get("ECMWF_DL_SLOW_RATE", "1000000"))   # B/s — under 1 MB/s ⇒ try another mirror
SLOW_SECS = int(os.environ.get("ECMWF_DL_SLOW_SECS", "35"))
SLOW_GRACE = int(os.environ.get("ECMWF_DL_SLOW_GRACE", "18"))
WATCH_TICK = int(os.environ.get("ECMWF_DL_WATCH_TICK", "9"))

# ── fail-fast the multiurl inner retry ────────────────────────────────────────
# ecmwf-opendata → multiurl.download wraps every byte-range GET in robust(maximum_tries=
# 500, retry_after=120): on a 503/429 it sleeps 120 s and retries the SAME mirror up to
# 500× (~16 h), so our own mirror-rotation never gets a turn. Clamp it hard so a throttled
# mirror is abandoned in seconds and _robust() rotates to a fresh one immediately.
FAILFAST_TRIES = int(os.environ.get("ECMWF_FAILFAST_TRIES", "2"))
FAILFAST_WAIT = int(os.environ.get("ECMWF_FAILFAST_WAIT", "12"))
try:
    import multiurl, multiurl.http, multiurl.downloader
    import ecmwf.opendata.client as _eoc
    _ORIG_ROBUST = multiurl.http.robust
    # *args/**kwargs: newer ecmwf-opendata/multiurl pass extra options through robust()
    # (e.g. use_server_retry_after, added ~0.3.33) — a fixed signature made every call
    # raise TypeError, which latest_run() swallowed per-cycle (Actions dead Jul 30→Aug 14).
    def _failfast_robust(call, maximum_tries=500, retry_after=120, *args, **kwargs):
        def _metered(*a, **k):                     # PREVENTIVE cross-process budget:
            if _budget is not None:                # every underlying GET pays a token
                _budget.acquire("*")
            return call(*a, **k)
        return _ORIG_ROBUST(_metered, min(maximum_tries, FAILFAST_TRIES),
                            min(retry_after, FAILFAST_WAIT), *args, **kwargs)
    # `robust` is bound in FOUR places, all pointing at the same original. ecmwf-opendata
    # fetches the .index via its OWN `multiurl.robust` binding (the real 500×/120s culprit),
    # and the byte-range data via the HTTP downloader. Replace the name everywhere it's bound.
    for _m in (multiurl, multiurl.http, multiurl.downloader, _eoc):
        if getattr(_m, "robust", None) is _ORIG_ROBUST:
            _m.robust = _failfast_robust
    # And clamp every downloader INSTANCE too (data layer), independent of binding.
    _ORIG_INIT = multiurl.http.HTTPDownloaderBase.__init__
    def _failfast_init(self, *a, **k):
        _ORIG_INIT(self, *a, **k)
        self.maximum_retries = min(getattr(self, "maximum_retries", 500), FAILFAST_TRIES)
        self.retry_after = min(getattr(self, "retry_after", 120), FAILFAST_WAIT)
    multiurl.http.HTTPDownloaderBase.__init__ = _failfast_init
except Exception:                                           # noqa: BLE001 — patch best-effort
    pass

# ── adaptive global back-off on S3 throttling (503 / SlowDown) ────────────────────
# A 503 is a per-IP request-RATE limit, not bandwidth — retrying FAST just keeps it
# tripped. So every download thread shares ONE penalty that grows (×2) on each throttle
# (capped) and decays on success; each retry sleeps the current penalty, collectively
# cutting the request rate until S3's limit resets. FETCH_TRIES re-pulls a whole spec
# (per-step resume) if a throttled mirror dropped messages, rather than aborting.
BACKOFF_BASE = float(os.environ.get("ECMWF_BACKOFF_BASE", "8"))    # s — first throttle wait
BACKOFF_MAX = float(os.environ.get("ECMWF_BACKOFF_MAX", "240"))   # s — cap
FETCH_TRIES = int(os.environ.get("ECMWF_FETCH_TRIES", "5"))   # whole-spec resume passes for stragglers
_THROTTLE = {"pen": 0.0}
_THROTTLE_LOCK = threading.Lock()
# Mirrors that 503'd recently → excluded only for a COOLDOWN window, then re-probed. A 503 is a
# transient per-IP rate limit, so a demoted mirror usually recovers in minutes; permanently
# excluding it for the whole run funnels every retrieve onto the one survivor (slow). Map of
# mirror → epoch it may be retried again; expired entries are dropped (so the mirror is re-probed).
THROTTLE_COOLDOWN = float(os.environ.get("ECMWF_THROTTLE_COOLDOWN", "240"))   # s — base re-probe wait
THROTTLE_COOLDOWN_MAX = float(os.environ.get("ECMWF_THROTTLE_COOLDOWN_MAX", "1800"))   # s — cap
_THROTTLED_UNTIL: dict = {}
_THROTTLE_STREAK: dict = {}       # mirror → consecutive-503 count (drives the exponential cooldown)


def _cooldown_for(src: str) -> float:
    """Re-probe wait for a mirror that just 503'd, growing exponentially with its consecutive-503
    streak (240 s → 480 → 960 → … capped). A transiently-throttled mirror recovers on its next
    success and resets to the base wait; a persistently rate-limited one is re-probed less and less
    often, so we stop hammering it (and stop logging a 503 every 240 s)."""
    n = _THROTTLE_STREAK.get(src, 0) + 1
    _THROTTLE_STREAK[src] = n
    return min(THROTTLE_COOLDOWN * (2 ** (n - 1)), THROTTLE_COOLDOWN_MAX)


def _throttled_set() -> set:
    """Mirrors still inside their post-503 cooldown; expired entries are evicted (→ re-probed)."""
    now = time.time()
    with _THROTTLE_LOCK:
        for s in [s for s, t in _THROTTLED_UNTIL.items() if now >= t]:
            _THROTTLED_UNTIL.pop(s, None)
        return set(_THROTTLED_UNTIL)


def _is_throttle(err) -> bool:
    s = repr(err).lower()
    return any(k in s for k in ("503", "slow down", "slowdown", "reduce your request", "429"))


def _throttle_pen() -> float:
    with _THROTTLE_LOCK:
        return _THROTTLE["pen"]


def _throttle_bump() -> float:
    with _THROTTLE_LOCK:
        _THROTTLE["pen"] = min(BACKOFF_MAX, max(BACKOFF_BASE, _THROTTLE["pen"] * 2))
        return _THROTTLE["pen"]


def _throttle_ease() -> None:
    # Decay GENTLY (×0.85), not a snap-to-zero — otherwise an intermittent success
    # between 503s keeps resetting the penalty and it never escalates to give S3 a
    # real break under sustained throttling.
    with _THROTTLE_LOCK:
        _THROTTLE["pen"] = 0.0 if _THROTTLE["pen"] < 1.0 else _THROTTLE["pen"] * 0.85


# One semaphore per mirror → never more than PER_SRC concurrent retrieves on a source,
# however many steps/mirrors the multi-source fan-out is juggling.
_SEM = {s: threading.Semaphore(PER_SRC) for s in SOURCES}
# Global cap on simultaneous retrieves (hence connections): each retrieve is ≤ CHUNK_MSGS
# connections and at most MAX_CONN//CHUNK_MSGS run at once, shared across ALL specs/threads
# in this process → total open connections stay under the portal's 500 limit.
_CONN_SLOTS = threading.Semaphore(max(1, MAX_CONN // CHUNK_MSGS))

# Daily steps. STEPS = analysis + Day 1..15 (used by RMM/AAM/MMSF, which want the 0-h
# analysis). STEPS_FC = forecast days only (used by the SOI/Hovmöller, which don't).
STEPS = [0] + list(range(24, 361, 24))
STEPS_FC = list(range(24, 361, 24))
#   10 hPa included so the AAM integral covers the full column (top layer 0–30 hPa —
#   the winter polar-night jet, worth up to ±1×10²⁵ kg m² s⁻¹; open data has no 20/30 hPa).
LEVELS_AAM = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
LEVELS_RMM = (200, 850)                                        # the MJO/RMM task downloads these
LEVELS_AAM_REST = tuple(l for l in LEVELS_AAM if l not in LEVELS_RMM)   # AAM downloads the other 12
#   The RMM (200/850) and AAM (the other 12) u-downloads are kept SEPARATE so the light RMM
#   critical-path isn't blocked behind the ~6 GB AAM pull, with NO duplicate level: 200/850 is
#   fetched once (by RMM) and the AAM builder concatenates the two files back to 14 levels.


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
    model: str                       # "aifs-ens" | "ifs" | "aifs-single" (stream oper, type fc)
    type: str                        # "pf" | "cf" | "em"
    param: object                    # one param str, or a tuple of params batched in 1 retrieve
    levtype: str                     # "pl" | "sfc"
    levelist: tuple = ()             # () for sfc
    steps: tuple = tuple(STEPS)
    nmembers: int = 0                # pf only: fetch just members 1..N (0 = all 50)

    @property
    def params(self) -> tuple:       # normalise to a tuple
        return self.param if isinstance(self.param, tuple) else (self.param,)

    @property
    def filename(self) -> str:
        lv = "-".join(str(x) for x in self.levelist) if self.levelist else self.levtype
        # step signature so two consumers wanting the same param but DIFFERENT step sets
        # (e.g. SOI's 24-360 vs AAM's 0-360) don't collide on one file
        s = self.steps
        sig = f"s{int(s[0])}-{int(s[-1])}x{len(s)}"
        if self.type == "pf" and self.nmembers:
            sig += f"m{self.nmembers}"
        return f"{self.type}_{'-'.join(self.params)}_{lv}_{sig}.grib2"

    def members(self):               # member set for the message-count expectation
        if self.type != "pf":
            return None
        return list(range(1, (self.nmembers or PF_MEMBERS) + 1))

    def n_expected(self) -> int:
        n = len(self.members()) if self.type == "pf" else 1
        n *= len(self.steps)
        n *= len(self.levelist) if self.levtype == "pl" else 1
        n *= len(self.params)
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


def _sidecar(p: Path, messages: int, src: str, secs: float | None = None) -> None:
    meta = {"messages": messages, "source": src,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if secs and secs > 0:
        try:
            size = os.path.getsize(p)
            meta.update(bytes=int(size), secs=round(secs, 1),
                        MBps=round(size / 1048576 / secs, 2))
        except OSError:
            pass
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta))


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
    with _SEM.get(src, contextlib.nullcontext()):          # cap in-flight retrieves / mirror
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


def _watch(do_fn, target: str, slow_ok: bool = False):
    done = threading.Event(); box = {"e": None}

    def _run():
        try:
            do_fn()
        except Exception as e:                             # noqa: BLE001
            box["e"] = e
        finally:
            done.set()
    threading.Thread(target=_run, daemon=True).start()
    last = 0; stalled = 0; slow = 0; elapsed = 0
    while not done.wait(WATCH_TICK):
        cur = _dl_bytes(target); rate = (cur - last) / WATCH_TICK; last = cur; elapsed += WATCH_TICK
        stalled = stalled + WATCH_TICK if rate < MIN_RATE else 0
        slow = slow + WATCH_TICK if rate < SLOW_RATE else 0
        if stalled >= STALL_SECS:
            return False, TimeoutError(f"stalled <{MIN_RATE/1000:.0f} kB/s for {STALL_SECS}s")
        if slow_ok and elapsed > SLOW_GRACE and slow >= SLOW_SECS:
            return False, TimeoutError(f"slow <{SLOW_RATE/1000:.0f} kB/s for {SLOW_SECS}s — switching mirror")
    return box["e"] is None, box["e"]


def _mark_src(target: str, src: str, attempt: int) -> None:
    """Breadcrumb the mirror currently being tried, so the status dashboard can show it."""
    try:
        with open(f"{target}.src", "w") as fh:
            fh.write(f"{src} {attempt}/{TRIES}")
    except OSError:
        pass


def _next_src(start: int, tried: list) -> str:
    """Mirror for the next attempt: a healthy (not-throttled-this-run) mirror not yet
    tried for this step; else round-robin the healthy pool; only fall back to a throttled
    mirror if EVERY mirror is currently throttled. This is what makes us switch AWAY from
    a 503'ing mirror mid-run instead of round-robining straight back onto it."""
    order = [SOURCES[(i + start) % len(SOURCES)] for i in range(len(SOURCES))]
    bad = _throttled_set()                                 # mirrors still in their 503 cooldown
    pool = [s for s in order if s not in bad] or order
    for s in pool:                                         # prefer a healthy mirror we haven't tried yet
        if s not in tried:
            return s
    return pool[len(tried) % len(pool)]                    # else round-robin the healthy pool


def _robust(req: dict, target: str, parallel: bool, members, expected: int, start: int = 0) -> str:
    label = os.path.basename(target)
    use_parallel = parallel and WORKERS >= 2 and members and len(members) >= 2
    err = None
    tried: list = []
    for attempt in range(1, TRIES + 1):
        src = _next_src(start, tried)                      # skip mirrors throttled this run
        tried.append(src)
        _mark_src(target, src, attempt)
        _clean(target)
        if use_parallel:
            do = lambda s=src: _parallel_once(req, target, s, members, WORKERS)
        elif parallel:
            do = lambda s=src: _single({**req, "number": members}, target, s)
        else:
            do = lambda s=src: _single(req, target, s)
        # abandon a < SLOW_RATE mirror only while a fresh, healthy mirror remains to switch to —
        # once every mirror has been sampled (all just slow), accept rather than churn or fail.
        fresh = [s for s in SOURCES if s not in _throttled_set() and s not in tried]
        with _CONN_SLOTS:                                  # global cap → stay under portal's 500 conns
            ok, err = _watch(do, target, slow_ok=bool(fresh))
        if ok:
            try:
                got = count_msgs(target)
            except Exception:                              # noqa: BLE001
                got = -1
            if got >= expected:
                _throttle_ease()
                with _THROTTLE_LOCK:
                    _THROTTLE_STREAK[src] = 0              # mirror healthy again → reset its backoff
                return src
            err = RuntimeError(f"incomplete GRIB: {got}/{expected} msgs")
        if _is_throttle(err):                              # 503/SlowDown
            if _budget is not None:
                _budget.penalize(src)                      # slow every process down together
            _note_throttle_src(src)                        # demote this mirror NEXT run too
            with _THROTTLE_LOCK:                           # …and cool it down (exponential per consecutive 503)
                wait = _cooldown_for(src)
                _THROTTLED_UNTIL[src] = time.time() + wait
            healthy_left = [s for s in SOURCES if s not in _throttled_set()]
            if healthy_left:                               # switch immediately — never sit on a throttled mirror
                print(f"    {label}: 503 throttle ({src}, cooldown {wait:.0f}s) → switching mirror "
                      f"(retry {attempt}/{TRIES}; healthy: {','.join(healthy_left)})", flush=True)
            else:                                          # everything cooling down → one global back-off, then re-probe all
                gwait = _throttle_bump()
                print(f"    {label}: 503 throttle — ALL mirrors cooling down, backing off "
                      f"{gwait:.0f}s (retry {attempt}/{TRIES})", flush=True)
                time.sleep(gwait)
                with _THROTTLE_LOCK:
                    _THROTTLED_UNTIL.clear()               # reset so the next attempt re-probes every mirror
        else:
            print(f"    {label}: {repr(err)[:55]} — retry {attempt}/{TRIES} (mirror)", flush=True)
    _clean(target)
    raise err if err is not None else RuntimeError(f"{label}: failed after {TRIES} tries")


def _plan_chunks(req: dict, members) -> list[tuple]:
    """Split a request into sub-retrieves each ≤ CHUNK_MSGS messages, so no single
    retrieve exceeds the portal's per-client connection budget. Split by step, then (if a
    single step still exceeds CHUNK_MSGS — e.g. 13-level pf) by member sub-groups.
    Returns [(step, member_subgroup_or_None, label), …]."""
    steps = req.get("step")
    steps = list(steps) if isinstance(steps, (list, tuple)) else [steps]
    factor = 1                                              # msgs per (step, member): levels × params
    for k in ("levelist", "param"):
        v = req.get(k)
        factor *= len(v) if isinstance(v, (list, tuple)) else 1
    out = []
    if members:
        mper = max(1, CHUNK_MSGS // max(1, factor))         # members per sub-chunk
        groups = [members[i:i + mper] for i in range(0, len(members), mper)]
        multi = len(groups) > 1
        for st in steps:
            for gi, g in enumerate(groups):
                out.append((st, g, f"s{int(st):03d}" + (f"_m{gi}" if multi else "")))
    else:
        for st in steps:
            out.append((st, None, f"s{int(st):03d}"))
    return out


def _robust_chunked(req: dict, target: str, parallel: bool, members) -> str:
    """Resumable chunked download: split into ≤ CHUNK_MSGS sub-retrieves (by step, then
    members), each → its own part file (kept if complete, so resume re-fetches only the
    short ones). The global _CONN_SLOTS semaphore (inside _robust) caps how many run at
    once, so total open connections stay under the portal's 500 limit. Small requests
    (cf/em, ≤ CHUNK_MSGS) are a single retrieve."""
    total = _msgs_for(req, members)
    if total <= CHUNK_MSGS or not isinstance(req.get("step"), (list, tuple)):
        return _robust(req, target, parallel, members, total)
    cdir = Path(f"{target}.parts"); cdir.mkdir(parents=True, exist_ok=True)
    plan = _plan_chunks(req, members)
    nsrc = len(SOURCES)

    def _fetch(idx_item):
        idx, (st, mg, lab) = idx_item
        pf = cdir / f"{lab}.grib2"
        sub = {**req, "step": [st]}
        if mg is not None:
            sub["number"] = mg
        exp = _msgs_for(sub, mg if mg is not None else members)
        if pf.exists():
            try:
                if count_msgs(str(pf)) >= exp:
                    return SOURCES[idx % nsrc]             # already complete (resume)
            except Exception:                              # noqa: BLE001
                pass
        return _robust(sub, str(pf), False, mg, exp, start=idx)   # parallel=False: already chunked

    if MULTISOURCE and len(plan) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, MAX_CONN // CHUNK_MSGS)) as ex:
            used = list(ex.map(_fetch, enumerate(plan)))   # _CONN_SLOTS is the real concurrency cap
    else:
        used = [_fetch(it) for it in enumerate(plan)]

    with open(target, "wb") as out:                        # assemble in plan order
        for st, mg, lab in plan:
            with open(cdir / f"{lab}.grib2", "rb") as f:
                shutil.copyfileobj(f, out)
    srcs = sorted({u for u in used if u})
    return "+".join(srcs) if len(srcs) > 1 else (srcs[0] if srcs else "chunked")


def _short_steps(parts_dir: str, req: dict, members) -> list[tuple]:
    """Sub-chunks whose stored part file is short of its expected msg count →
    (label, got, exp). Lets ensure() report what's missing and resume only those chunks."""
    out = []
    d = Path(parts_dir)
    if not d.is_dir():
        return out
    for st, mg, lab in _plan_chunks(req, members):
        exp = _msgs_for({**req, "step": [st]}, mg if mg is not None else members)
        pf = d / f"{lab}.grib2"
        try:
            got = count_msgs(str(pf)) if pf.exists() else 0
        except Exception:                                  # noqa: BLE001
            got = 0
        if got < exp:
            out.append((lab, got, exp))
    return out


def _msgs_for(req: dict, members) -> int:
    n = len(members) if members else 1
    for k in ("step", "levelist", "param"):
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
    pm = list(spec.params)
    stream = "oper" if spec.model == "aifs-single" else "enfo"
    r = dict(model=spec.model, date=cycle.iso, time=int(cycle.time), stream=stream,
             type=spec.type, param=pm if len(pm) > 1 else pm[0],
             levtype=spec.levtype, step=list(spec.steps))
    if spec.levtype == "pl":
        r["levelist"] = [int(x) for x in spec.levelist]
    return r


def _fetch_v2(cycle: Cycle, spec: Spec, target: str) -> str | None:
    """Coalesced-range fetch of the whole spec into `target` (all steps,
    sequential; each step ~24-way parallel inside rangefetch). Returns the
    source label, or None to signal fallback to the legacy client path."""
    if os.environ.get("ECMWF_FETCH", "v2") != "v2":
        return None
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import rangefetch as rf
    except Exception:                                      # noqa: BLE001
        return None
    srcs = [s for s in (os.environ.get("ECMWF_SOURCES") or ",".join(_BASE_MIRRORS)).split(",")
            if s in rf.MIRRORS] or list(_BASE_MIRRORS)
    levl = list(spec.levelist) if spec.levtype == "pl" else None
    nums = spec.members()
    stats: dict = {}
    try:
        with open(target, "wb") as out:
            stream = "oper" if spec.model == "aifs-single" else "enfo"
            for step in spec.steps:
                idx = rf.fetch_index(cycle.date, cycle.time, spec.model,
                                     int(step), spec.type, stream=stream,
                                     sources=srcs)
                want = rf.select(idx, param=list(spec.params), levelist=levl,
                                 numbers=nums)
                if not want:
                    raise RuntimeError(f"step {step}: 0 msgs matched")
                ranges = rf.coalesce(want)
                blob = rf.fetch_ranges(
                    rf.path_for(cycle.date, cycle.time, spec.model,
                                int(step), spec.type, stream=stream) + ".grib2",
                    ranges, sources=srcs, stats=stats)
                out.write(blob)
        top = max(stats, key=stats.get) if stats else srcs[0]   # mirror that served the most bytes
        return f"v2:{top}"
    except Exception as e:                                 # noqa: BLE001
        print(f"  fetch-v2 failed ({str(e)[:90]}) — falling back to client path",
              flush=True)
        Path(target).unlink(missing_ok=True)
        return None


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
        Path(f"{stage}.src").unlink(missing_ok=True)       # stale source breadcrumb
        print(f"  ECMWF {cycle.tag} {spec.model}/{spec.filename}: fetching "
              f"({expected} msgs) …", flush=True)
        req = _to_req(cycle, spec); members = spec.members()
        got, src = 0, None
        t0 = time.time()
        v2src = _fetch_v2(cycle, spec, str(stage))
        if v2src:
            got = count_msgs(str(stage))
            if got >= expected:
                src = v2src
            else:
                print(f"  fetch-v2 short ({got}/{expected}) — client fallback",
                      flush=True)
                _clean(str(stage))
        for ftry in range(1, FETCH_TRIES + 1):
            if got >= expected:
                break
            src = _robust_chunked(req, str(stage), spec.type == "pf", members)
            got = count_msgs(str(stage))
            if got >= expected:
                break
            # incomplete → say exactly which steps are short, then resume just those
            short = _short_steps(f"{stage}.parts", req, members)
            Path(str(stage)).unlink(missing_ok=True)       # drop the bad assembly; KEEP .parts to resume
            if not short:                                  # per-step counts OK but total short → clean slate next
                shutil.rmtree(f"{stage}.parts", ignore_errors=True)
            miss = ", ".join(f"{h} {g}/{e}" for h, g, e in short) or "mid-file corruption"
            # back off harder each pass so S3's per-prefix rate limit can actually reset before we
            # come back for the stragglers (the .parts are kept, so each pass only re-fetches the
            # short chunks — never the whole file).
            wait = min(BACKOFF_MAX, max(BACKOFF_BASE, _throttle_pen()) + 20 * (ftry - 1))
            print(f"  {spec.filename}: incomplete {got}/{expected} — missing [{miss}] — "
                  f"re-fetch {ftry}/{FETCH_TRIES} after {wait:.0f}s", flush=True)
            time.sleep(wait)
        if got < expected:
            _clean(str(stage)); shutil.rmtree(f"{stage}.parts", ignore_errors=True)
            raise RuntimeError(f"{spec.filename}: still incomplete after {FETCH_TRIES} fetches ({got}/{expected})")
        shutil.rmtree(f"{stage}.parts", ignore_errors=True)   # verified complete → drop the parts
        secs = time.time() - t0
        _sidecar(stage, got, src, secs=secs)
        mbps = os.path.getsize(stage) / 1048576 / secs if secs > 0 else 0.0
        _note_speed(src or "", mbps)
        os.replace(stage.with_suffix(stage.suffix + ".json"), p.with_suffix(p.suffix + ".json"))
        os.replace(stage, p)                               # atomic publish
        print(f"  ECMWF {cycle.tag} {spec.model}/{spec.filename}: ✓ {got} msgs via {src} "
              f"({mbps:.1f} MB/s)", flush=True)
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


# ── surface-field batches ─────────────────────────────────────────────────────
# One retrieve per (model, type) per step-class, instead of a separate pull per field.
# A single retrieve carries one step set, so split by step-class: forecast-step fields
# (10u/10v/msl — Hovmöller/SOI/torque) vs analysis-included fields (sp/2t — AAM/ens).
# Consumers open the batch with backend_kwargs={"filter_by_keys": {"shortName": <field>}}.
# ALL AIFS surface fields in ONE batch (steps 0..360 — sp needs the 0-h analysis).
# 2t is dropped (the ensembles t2m page is paused; re-add "2t" here to fetch it again).
SFC = ("sp", "10u", "10v", "msl")
SFC_IFS = ("10u", "10v", "msl")       # IFS feeds Hovmöller + SOI + super-ensemble MSLP/wind map
# Legacy pre-consolidation batches — kept ONLY for the sfc_path fallback, so a cycle that
# already fetched the old fc/an batches before the merge isn't needlessly re-downloaded.
_SFC_FC_LEGACY = ("10u", "10v", "msl")
_SFC_AN_LEGACY = ("sp", "2t")


def sfc_spec(model: str, typ: str) -> Spec:
    """The single batched surface Spec for a model (all fields, steps 0..360)."""
    if model == "ifs":
        return Spec("ifs", typ, SFC_IFS, "sfc", (), tuple(STEPS_FC))
    return Spec(model, typ, SFC, "sfc", (), tuple(STEPS))


def sfc_path(cycle: Cycle, model: str, typ: str, short: str) -> Path:
    """Ensure the surface batch carrying `short` (sp/10u/10v/msl) and return its path.
    Open it filtered, e.g. backend_kwargs={'filter_by_keys': {'shortName': short}}."""
    spec = sfc_spec(model, typ)
    if model != "ifs" and not _complete(path(cycle, spec), spec.n_expected()):
        # transition fallback: reuse a legacy fc/an batch already complete in this cycle
        for legacy, steps in ((_SFC_FC_LEGACY, STEPS_FC), (_SFC_AN_LEGACY, STEPS)):
            if short in legacy:
                lspec = Spec(model, typ, legacy, "sfc", (), tuple(steps))
                if _complete(path(cycle, lspec), lspec.n_expected()):
                    return path(cycle, lspec)
    return ensure(cycle, spec)


# ── registry: everything a 00Z/12Z cycle needs (for the pre-fetcher) ─────────────
def registry() -> list[Spec]:
    S = tuple(STEPS)        # 0..360 (with analysis) — RMM, AAM, MMSF, ens
    L13 = tuple(l for l in LEVELS_AAM if l != 10)   # walker/mmsf analysis pulls
    return [
        # PRIORITY ORDER IS THE RENDER DAG: renders self-fetch via ensure() under
        # per-file flocks, so each product group starts the moment ITS files land.
        # Small high-value files must never queue behind the multi-GB AAM pull.
        # 1) RMM critical path (normally already fetched by Stage 1)
        Spec("aifs-ens", "pf", "u", "pl", LEVELS_RMM, S),
        Spec("aifs-ens", "cf", "u", "pl", LEVELS_RMM, S),
        # 2) surface batches (sp/10u/10v/msl, one file per type) — TC tracker,
        #    MSLP/wind, Hovmöller, SOI. IFS carries just 10u/msl.
        sfc_spec("aifs-ens", "pf"), sfc_spec("aifs-ens", "cf"),
        sfc_spec("ifs", "pf"),
        # 3) analysis-step (0) multi-level winds — small; Walker + MMSF unblock here
        Spec("aifs-ens", "cf", "u", "pl", L13, (0,)),
        Spec("aifs-ens", "cf", "v", "pl", L13, (0,)),
        Spec("aifs-ens", "cf", "v", "pl", LEVELS_AAM, (0,)),
        # 4) v@200 all leads (~0.6 GB) — velocity potential + WAF unblock here
        Spec("aifs-ens", "pf", "v", "pl", (200,), S),
        # 5) the heavy AAM/jets multi-level pull (~3 GB at 25 members) goes LAST
        Spec("aifs-ens", "pf", "u", "pl", LEVELS_AAM_REST, S, AAM_PF_MEMBERS),
        Spec("aifs-ens", "cf", "u", "pl", LEVELS_AAM_REST, S),
        # z@500 dropped 2026-07-24: ensembles page is retired (ens_cycle gates its
        # copy behind ENS_FETCH_Z500=1) — it was ~0.5 GB/cycle of dead weight.
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
