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

RETAIN_DAYS = 7        # store cache retention — raw gribs dropped after this many days


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
        except Exception as e:                              # noqa: BLE001
            print(f"  10u/{k}: skipped ({repr(e)[:70]})", flush=True)

    print("== msl — AIFS-ENS + IFS-ENS (SOI) ==", flush=True)
    for cfg in soi_forecast.MODELS:
        try:
            soi_forecast.download_msl(cfg, date, time, data_root / "msl")
        except Exception as e:                              # noqa: BLE001
            print(f"  msl/{cfg['model']}: skipped ({repr(e)[:70]})", flush=True)

    # z500 + 2 m temperature (full cf+pf ensemble) — kept in the store for the
    # ensembles page and general later reuse. Best-effort.
    print("== AIFS-ENS z@500 + 2t (ensemble store) ==", flush=True)
    cyc = store.Cycle(date, time); S = tuple(store.STEPS)
    for param, levtype, levels in (("z", "pl", (500,)), ("2t", "sfc", ())):
        for typ in ("cf", "pf"):
            try:
                store.ensure(cyc, store.Spec("aifs-ens", typ, param, levtype, levels, S))
            except Exception as e:                          # noqa: BLE001
                print(f"  {param}/{typ}: skipped ({repr(e)[:70]})", flush=True)

    # Heavy SST-page atmospheric fields (AAM / torque / MMSF). These builders are
    # laptop-only (not in the public Action checkout), so guard the imports: where
    # they're absent the downloads are simply skipped. Best-effort — a miss here
    # must not abort the cycle (each builder also degrades gracefully on its own).
    dirs = ["aifs", "u10", "msl"]
    try:
        import aam
        print("== sp + u@13lev (AAM) ==", flush=True)
        aam.download(date, time, data_root / "aam"); dirs.append("aam")
    except ImportError:
        pass
    except Exception as e:                                  # noqa: BLE001
        print(f"  AAM fields: skipped ({repr(e)[:70]})", flush=True)
    try:
        import torque_map_anim as _tma                      # gate on the private builder
        print("== 10v (surface torque) ==", flush=True)
        cyc = store.Cycle(date, time); steps = tuple(_tma.DAILY_STEPS)
        for typ in ("cf", "pf"):
            store.ensure(cyc, store.Spec("aifs-ens", typ, "10v", "sfc", (), steps))
        dirs.append("torque")
    except ImportError:
        pass
    except Exception as e:                                  # noqa: BLE001
        print(f"  10v: skipped ({repr(e)[:70]})", flush=True)
    try:
        import mmsf
        print("== v@13lev step0 (MMSF) ==", flush=True)
        mmsf.download_v(date, time, data_root / "mmsf"); dirs.append("mmsf")
    except ImportError:
        pass
    except Exception as e:                                  # noqa: BLE001
        print(f"  v field: skipped ({repr(e)[:70]})", flush=True)

    cyc_dir = store.CACHE / store.Cycle(date, time).tag
    total = (sum(f.stat().st_size for f in cyc_dir.rglob("*.grib2")) / 1e9
             if cyc_dir.exists() else 0.0)
    print(f"Cycle cache ready ({total:.2f} GB in {cyc_dir}).")

    # Retention: drop store cycles older than 7 days (the raw gribs aren't needed
    # once each product's processed history/archive is written).
    store.prune(days=RETAIN_DAYS)
    cache_gb = sum(f.stat().st_size for f in store.CACHE.rglob("*.grib2")) / 1e9
    print(f"Store cache now {cache_gb:.1f} GB across "
          f"{len(list(store.CACHE.glob('*z')))} cycle(s) (≤{RETAIN_DAYS}-day retention).")


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
