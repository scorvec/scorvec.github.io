#!/usr/bin/env python3
"""One-time reconciliation after the 13→14-level (add 10 hPa) AAM upgrade.

1. aam_history.nc — recompute every existing observed date with the 14-level
   integral (exact, streamed from ARCO ERA5; the old values were 13-level).
2. aam_forecast_archive.nc — archived ensemble-mean forecasts were computed
   from 13-level AIFS fields and can't be recomputed (the 10 hPa data was
   never fetched). Patch each pre-upgrade init additively with
   Δclim(region, doy of valid) = clim14 − clim13, so plotted anomalies
   (fc − clim14) match what the 13-level pair produced. Idempotent via MARK.

    python src/reconcile_aam_14lev.py --old-clim <backup aam_clim.nc>
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from aam import ARCHIVE_PATH, CLIM_PATH, HIST_PATH, LEVELS, _aam_era5, eval_clim

MARK = "level14_upgrade_2026_07_18"
CUTOVER = pd.Timestamp("2026-07-18T06:00")     # inits before this were 13-level


def rebuild_history() -> None:
    hist = xr.open_dataset(HIST_PATH).load()
    if hist.attrs.get("levels_mark") == MARK:
        print("history already 14-level — skipping"); return
    times = pd.to_datetime(hist.time.values)
    print(f"recomputing {len(times)} observed days at {len(LEVELS)} levels …", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda t: _aam_era5(t.strftime("%Y-%m-%dT12:00")), times))
    arr = np.array(rows)                                        # (time, 3) global/nh/sh
    out = xr.Dataset({k: ("time", arr[:, i]) for i, k in enumerate(("global", "nh", "sh"))},
                     coords={"time": times}, attrs={"levels_mark": MARK})
    tmp = HIST_PATH.with_suffix(".tmp.nc"); out.to_netcdf(tmp); os.replace(tmp, HIST_PATH)
    d = arr[:, 0] - np.stack([hist["global"].values, hist["nh"].values, hist["sh"].values]).T[:, 0]
    print(f"  history rebuilt: global Δ mean {d.mean():+.3f}, range {d.min():+.3f}…{d.max():+.3f} ×10²⁵")


def patch_archive(old_clim_path: Path) -> None:
    arch = xr.open_dataset(ARCHIVE_PATH).load()
    if arch.attrs.get("levels_mark") == MARK:
        print("forecast archive already patched — skipping"); return
    new_c, old_c = xr.open_dataset(CLIM_PATH), xr.open_dataset(old_clim_path)
    fc = arch["fc_mean"].values                                  # (init, region, lead)
    inits = pd.to_datetime(arch.init.values)
    regions = [str(r) for r in arch.region.values]
    leads = arch.lead.values
    n_patched = 0
    for i, it in enumerate(inits):
        if it >= CUTOVER:
            continue
        doys = np.array([(it + pd.Timedelta(days=int(l))).dayofyear for l in leads], float)
        for j, reg in enumerate(regions):
            delta = (eval_clim(new_c["coeffs"].sel(region=reg).values, doys)
                     - eval_clim(old_c["coeffs"].sel(region=reg).values, doys))
            fc[i, j, :] += delta
        n_patched += 1
    arch["fc_mean"].values = fc
    arch.attrs["levels_mark"] = MARK
    tmp = ARCHIVE_PATH.with_suffix(".tmp.nc"); arch.to_netcdf(tmp); os.replace(tmp, ARCHIVE_PATH)
    print(f"  archive: {n_patched}/{len(inits)} pre-upgrade inits shifted by Δclim (14lev−13lev)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-clim", required=True, help="backup of the 13-level aam_clim.nc")
    args = ap.parse_args()
    rebuild_history()
    patch_archive(Path(args.old_clim))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
