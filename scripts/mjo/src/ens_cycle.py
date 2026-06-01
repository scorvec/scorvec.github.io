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
import download_aifs
import eq_hovmoller
import soi_forecast
from download_aifs import latest_run


def fetch(date: str, time: str, data_root: Path) -> None:
    def _size(d: Path) -> float:
        return sum(f.stat().st_size for f in d.glob("*.grib2")) / 1e9 if d.exists() else 0.0

    # RMM is the core product — let a failure here propagate (fail the cycle).
    print("== AIFS-ENS u@200/850 (RMM) ==", flush=True)
    download_aifs.download(date, time, data_root / "aifs")

    # Hovmöller / SOI fields are best-effort: a single model missing for this
    # cycle shouldn't abort the others (each build has its own per-model fallback).
    print("== 10u — AIFS-ENS + IFS-ENS (Hovmöller) ==", flush=True)
    for k in eq_hovmoller.MODELS:
        try:
            eq_hovmoller.download(k, date, time, data_root / "u10")
        except Exception as e:                              # noqa: BLE001
            print(f"  10u/{k}: skipped ({repr(e)[:70]})", flush=True)

    print("== msl — AIFS-ENS + IFS-ENS (SOI) ==", flush=True)
    for cfg in soi_forecast.MODELS:
        try:
            soi_forecast.download_msl(cfg, date, time, data_root / "msl")
        except Exception as e:                              # noqa: BLE001
            print(f"  msl/{cfg['model']}: skipped ({repr(e)[:70]})", flush=True)

    total = sum(_size(data_root / d) for d in ("aifs", "u10", "msl"))
    print(f"Cycle cache ready ({total:.2f} GB across aifs/u10/msl).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: latest available)")
    ap.add_argument("--time", help="00 or 12 (default: latest available)")
    ap.add_argument("--data-root", default="data", help="cache root (per-product subdirs)")
    args = ap.parse_args()

    if args.date and args.time:
        date, time = args.date, args.time
    else:
        print("Finding latest available AIFS-ENS cycle …", flush=True)
        date, time = latest_run()
    print(f"Fetching ENS cycle {date} {time}Z", flush=True)

    t0 = _time.time()
    fetch(date, time, Path(args.data_root))
    print(f"Done in {(_time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
