#!/usr/bin/env python3
"""ERA5 weekly TRUTH for the SFS subseasonal page's skill masks (2026-10-12). LAPTOP ONLY: it reads ~/era5_store, the
local-only ERA5 store, and writes a small table the Actions build verifies the SFS reforecast against.

For one SFS start month MM and every year 1991-2025: the weekly means of days 1-7, 8-14, ..., 36-42 counted from the
1st of the month (day 1 = the start day, the SFS lead 0), as anomalies against the 1991-2020 mean of the same calendar
week. 2 m temperature is global (wb2_1p5_daily_global, 1991-2026); precipitation is Northern Hemisphere only and ends
in 2023 (wb2_1p5_daily: the store holds no Southern Hemisphere precipitation), NaN elsewhere. The store's attributes
lie (prcp is mm/day; t2m's dimension order changes in 2023) - read by name, values checked.

Grid: the GEFS page's 1.5 deg grid (gefs_reforecast.LAT 90..-90, LON 0..358.5), so the masks drop straight onto the maps.

    python scripts/sfs/sfs_truth.py --months 09 10 --out /tmp/sfs_truth
-> sfs_truth_MM.npz {years, t2m (35, 6, 121, 240) float16 K, pr (35, 6, 121, 240) float16 mm/day}
   uploaded to the release sfs-ref-v1 (gh release upload), read by sfs_sub.py in Actions.
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

STORE = Path("~/era5_store").expanduser()
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]
YEARS = list(range(1991, 2026))
LAT = np.arange(90.0, -90.0 - 1e-6, -1.5)
LON = np.arange(0.0, 360.0 - 1e-6, 1.5)


def daily(var: str, year: int, months: tuple[int, int]):
    """(days, lat 90..-90, lon) for the months [m0, m1] of `year` (m1 may run into the next year)."""
    sub = "wb2_1p5_daily_global" if var == "t2m" else "wb2_1p5_daily"
    out = []
    for y in sorted({year, year + (1 if months[1] < months[0] else 0)}):
        f = STORE / sub / var / f"{var}_{y}.nc"
        if not f.exists():
            return None
        out.append(xr.open_dataset(f, engine="netcdf4")[var])
    da = xr.concat(out, "time").transpose("time", "latitude", "longitude")
    return da.reindex(latitude=LAT, longitude=LON, method="nearest", tolerance=0.01)   # the store's coordinates carry float noise


def build(mm: int) -> dict:
    res = {}
    for var in ("t2m", "pr"):
        src = "t2m" if var == "t2m" else "prcp"
        W = np.full((len(YEARS), len(WEEKS), len(LAT), len(LON)), np.nan, np.float32)
        for yi, y in enumerate(YEARS):
            d0 = dt.date(y, mm, 1)
            d1 = d0 + dt.timedelta(days=41)
            da = daily(src, y, (mm, d1.month))
            if da is None:
                continue
            da = da.sel(time=slice(str(d0), str(d1)))
            if da.sizes["time"] < 42:
                continue
            v = da.values.astype(np.float64)
            for k, (a, b) in enumerate(WEEKS):
                W[yi, k] = v[a - 1:b].mean(0)
            if yi % 10 == 0:
                print(f"  {var} {mm:02d}: {y}", flush=True)
        ok = np.array([1991 <= y <= 2020 for y in YEARS])
        clim = np.nanmean(W[ok], axis=0)
        A = W - clim[None]
        if var == "t2m":                                  # value check: K, plausible weekly anomaly spread
            assert 200 < np.nanmean(clim) < 320, np.nanmean(clim)
        else:                                             # mm/day despite the "m" attribute
            assert 0.5 < np.nanmean(clim) < 10, np.nanmean(clim)
        print(f"  {var} {mm:02d}: years with data {int(np.isfinite(A[:, 0, 60, 0]).sum() if var == 't2m' else np.isfinite(A[:, 0, 20, 0]).sum())}, "
              f"anomaly sd week 1 {np.nanstd(A[:, 0]):.2f}", flush=True)
        res[var] = A.astype(np.float16)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["09", "10"])
    ap.add_argument("--out", default="/tmp/sfs_truth")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for m in a.months:
        r = build(int(m))
        p = out / f"sfs_truth_{int(m):02d}.npz"
        np.savez_compressed(p, years=np.array(YEARS), weeks=np.array(WEEKS), lat=LAT, lon=LON, **r)
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
