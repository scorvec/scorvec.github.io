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
import threading
from pathlib import Path

from ecmwf.opendata import Client


DATA_DIR = Path(__file__).parent.parent / "data" / "aifs"

# Steps to day 15. Default DAILY (24-hourly): the RMM only needs daily values
# (it builds daily means via step//24 groupby, so one sample/day is enough), and
# daily steps cut the ~1 MB/s-bandwidth-bound download ~4x (≈4 GB → ≈1 GB).
# Set AIFS_STEP_HOURS=6 for the old 4-samples/day behavior if ever needed.
# Step 0 (the analysis) is included so lead_day 0 exists — the zero-lag "truth"
# point archived to obs_history (daily steps otherwise start at lead_day 1).
_STEP_HOURS = int(os.environ.get("AIFS_STEP_HOURS", "24"))
STEPS = [0] + list(range(_STEP_HOURS, 361, _STEP_HOURS))

# Prefer the cloud mirrors over the main ECMWF portal (which enforces a
# connection limit / HTTP 429 that throttles the 4 GB ensemble download).
# NOTE: the "google" source 400s for aifs-ens pressure-level data (it doesn't
# host this product), so it's excluded — AWS/Azure first, portal last.
# Override with the AIFS_SOURCES env var.
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


def _robust(req: dict, target: str, parallel: bool, members=None, workers: int = None) -> str:
    label = os.path.basename(target)
    workers = workers or DL_WORKERS
    members = list(members) if members is not None else list(range(1, PF_MEMBERS + 1))
    use_parallel = parallel and workers >= 2 and len(members) >= 2
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
        if ok:
            _archive_grib(target, req)
            return f"{src}×{workers}" if use_parallel else src
        print(f"  {label}: {repr(err)[:60]} — retry {attempt}/{DL_TRIES} (mirror rotate)", flush=True)
    _clean(target)
    raise err if err is not None else RuntimeError(f"{label}: failed after {DL_TRIES} tries")


def _retrieve(req: dict, target: str) -> str:
    """Robust single-stream retrieve (watchdog + mirror-rotating retry)."""
    return _robust(req, target, parallel=False)


def retrieve_parallel(req: dict, target: str, members=None, workers: int = None) -> str:
    """Robust parallel (multi-member-stream) retrieve."""
    return _robust(req, target, parallel=True, members=members, workers=workers)


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
                    _retrieve_probe(dict(model="aifs-ens", date=date_str, time=int(run_time),
                                         stream="enfo", type="cf", levtype="pl",
                                         levelist=[850], param="u", step=6), tmp.name)
                return date_str, run_time
            except Exception:
                continue
    raise RuntimeError("Could not find a recent available AIFS-ENS run (checked last 4 days)")


def download(date: str, time: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"aifs_{date}_{time}z"

    print(f"Downloading AIFS-ENS {date} {time}Z (sources: {','.join(SOURCES)}) …")

    common = dict(
        model="aifs-ens",
        date=date,
        time=int(time),
        stream="enfo",
        levtype="pl",
        levelist=[200, 850],
        param="u",
        step=STEPS,
    )

    # Perturbed forecasts (parallel member streams — ~3x faster)
    pf_path = out_dir / f"{stem}.pf.u.grib2"
    if not pf_path.exists():
        src = retrieve_parallel({**common, "type": "pf"}, str(pf_path))
        print(f"  pf u-wind saved via {src}: {pf_path.name}")
    else:
        print(f"  {pf_path.name}: already exists, skipping")

    # Control forecast
    cf_path = out_dir / f"{stem}.cf.u.grib2"
    if not cf_path.exists():
        src = _retrieve({**common, "type": "cf"}, str(cf_path))
        print(f"  cf u-wind saved via {src}: {cf_path.name}")
    else:
        print(f"  {cf_path.name}: already exists, skipping")

    print("Download complete.")


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
