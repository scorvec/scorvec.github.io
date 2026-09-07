#!/usr/bin/env python3
"""One-off (laptop): seed the χ history from the AIFS-ENS 0-h analyses still on the Google mirror
(~45 days). Writes scripts/mjo/data/reference/walker_chi_history_seed.nc, which walker_chi merges
with the frames-branch history on every runner.

    python src/walker_chi_backfill.py --days 44
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).parent))
import store as ecmwf                                        # noqa: E402
import walker_chi as wc                                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=44)
    a = ap.parse_args()
    t0 = time.time()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    done = 0
    for d in range(a.days, -1, -1):
        day = now - timedelta(days=d)
        for hh in ("00", "12"):
            valid = datetime(day.year, day.month, day.day, int(hh))
            if valid > now - timedelta(hours=8):
                continue
            cyc = ecmwf.Cycle(valid.strftime("%Y%m%d"), hh)
            try:
                up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", wc.LEVELS, (0,)))
                vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", wc.LEVELS, (0,)))
            except Exception as ex:                                  # noqa: BLE001
                print(f"  {cyc.tag}: not available ({str(ex)[:60]})", flush=True); continue
            u = xr.open_dataset(up, **ku)["u"]; v = xr.open_dataset(vp, **ku)["v"]
            chis = {}
            for lev in wc.LEVELS:
                chi, lat, lon = wc.chi_strip(u.sel(isobaricInhPa=lev), v.sel(isobaricInhPa=lev))
                chis[lev] = chi
            wc.update_history(wc.SEED, wc.record(chis, lat, lon, valid))
            done += 1
            print(f"  {cyc.tag} ok ({done}, {time.time() - t0:.0f}s)", flush=True)
    h = wc.load_history(wc.SEED)
    print(f"seed: {h.time.size} analyses {str(h.time.values[0])[:13]} … {str(h.time.values[-1])[:13]}, {wc.SEED.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
