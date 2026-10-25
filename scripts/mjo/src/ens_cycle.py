#!/usr/bin/env python3
"""One-shot fetch of every AIFS/IFS-ENS field our products need for a cycle.

A single "download the cycle" task for both the MJO and El Niño (SST-page)
products. It delegates to each product's idempotent downloader (skip-if-exists),
populating the per-product caches so the RMM, the equatorial wind Hovmöller and
the SOI forecast can all then run straight from disk with no re-download.

Manifest (all DAILY-step now → ~3 GB/cycle, ~1 MB/s-bandwidth-bound):
  AIFS-ENS (cf+pf): u@200/850 (RMM) · 10u (Hovmöller) · msl (SOI)
  IFS-ENS  (pf):    10u (Hovmöller) · msl (SOI)

Run once per cycle, then build everything from cache:
    python src/ens_cycle.py --date 20260601 --time 00
    python run_rmm.py --skip-download --date 20260601 --time 00
    python src/eq_hovmoller.py --date 20260601 --time 00 --out …
    python src/soi_forecast.py --date 20260601 --time 00 --out …
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store
import download_aifs
import eq_hovmoller
import soi_forecast
from download_aifs import latest_run

RETAIN_DAYS = 7  # store cache retention — raw gribs dropped after this many days


def fetch_strat(date: str, time: str) -> None:
    """Only what the stratosphere page reads from the store cache.

    nh_vortex globs aifs-ens pf_u_10-*/cf_u_10-* (the 12-level AAM u pull) and
    qbo_duct gates on cf_u_10-*x16 + cf_u_200-850_s0-*x16 (it fetches its own v
    and t). Nothing on that page reads the Hovmöller/SOI surface batches or
    IFS-ENS, so this subset drops the IFS-ENS dependency of fetch(): at the
    07:35Z dispatch IFS 00Z is not yet on open data, and waiting on it (once
    for the Hovmöller, once for the SOI) can consume the whole 60-min step
    budget. ~5 min instead of ~19."""
    print("== stratosphere subset: AIFS-ENS u @ 200/850 + the 12 AAM levels ==", flush=True)
    cyc = store.Cycle(date, time)
    S = tuple(store.STEPS)
    specs = [
        store.Spec("aifs-ens", "cf", "u", "pl", store.LEVELS_RMM, S),
        store.Spec("aifs-ens", "pf", "u", "pl", store.LEVELS_RMM, S),
        store.Spec("aifs-ens", "cf", "u", "pl", store.LEVELS_AAM_REST, S),
        store.Spec("aifs-ens", "pf", "u", "pl", store.LEVELS_AAM_REST, S),
    ]
    for sp in specs:  # the core u files: let a failure propagate
        store.ensure(cyc, sp)


def fetch(date: str, time: str, data_root: Path) -> None:
    # RMM is the core product — let a failure here propagate (fail the cycle).
    print("== AIFS-ENS u@200/850 (RMM) ==", flush=True)
    download_aifs.download(date, time, data_root / "aifs")

    # Hovmöller / SOI fields are best-effort: a single model missing for this
    # cycle shouldn't abort the others (each build has its own per-model fallback).
    print("== 10u — AIFS-ENS + IFS-ENS (Hovmöller) ==", flush=True)
    for k in eq_hovmoller.MODELS:
        try:
            eq_hovmoller.download(k, date, time, data_root / "u10")
        except Exception as e:  # noqa: BLE001
            print(f"  10u/{k}: skipped ({repr(e)[:70]})", flush=True)

    print("== msl — AIFS-ENS + IFS-ENS (SOI) ==", flush=True)
    for cfg in soi_forecast.MODELS:
        try:
            soi_forecast.download_msl(cfg, date, time, data_root / "msl")
        except Exception as e:  # noqa: BLE001
            print(f"  msl/{cfg['model']}: skipped ({repr(e)[:70]})", flush=True)

    cyc_dir = store.CACHE / store.Cycle(date, time).tag
    total = sum(f.stat().st_size for f in cyc_dir.rglob("*.grib2")) / 1e9 if cyc_dir.exists() else 0.0
    print(f"Cycle cache ready ({total:.2f} GB in {cyc_dir}).")

    # Retention: drop store cycles older than 7 days (the raw gribs aren't needed
    # once each product's processed history/archive is written).
    store.prune(days=RETAIN_DAYS)
    cache_gb = sum(f.stat().st_size for f in store.CACHE.rglob("*.grib2")) / 1e9
    print(
        f"Store cache now {cache_gb:.1f} GB across "
        f"{len(list(store.CACHE.glob('*z')))} cycle(s) (≤{RETAIN_DAYS}-day retention)."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: latest available)")
    ap.add_argument("--time", help="00 or 12 (default: latest available)")
    ap.add_argument("--data-root", default="data", help="cache root (per-product subdirs)")
    ap.add_argument(
        "--only",
        choices=("all", "strat"),
        default="all",
        help="'strat': just the AIFS-ENS u files the stratosphere page reads "
        "(no IFS-ENS, no surface batches) - see fetch_strat()",
    )
    args = ap.parse_args()

    if args.date and args.time:
        date, time = args.date, args.time
    else:
        print("Finding latest available AIFS-ENS cycle …", flush=True)
        date, time = latest_run()
    print(f"Fetching ENS cycle {date} {time}Z", flush=True)

    t0 = _time.time()
    if args.only == "strat":
        try:
            fetch_strat(date, time)
        except store.NotPublished as e:
            # Not an error in the fetcher: the cycle is not on the mirrors yet.
            # Exit non-zero so the caller's `|| warning` fires, without a traceback.
            print(f"ens_cycle: {e}", flush=True)
            return 2
        store.prune(days=RETAIN_DAYS)
    else:
        fetch(date, time, Path(args.data_root))
    print(f"Done in {(_time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
