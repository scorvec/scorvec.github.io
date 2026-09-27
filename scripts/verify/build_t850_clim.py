#!/usr/bin/env python3
"""850 hPa temperature climatology for the AIFS verification's ERA5 anomaly scores (t850 had RMSE and bias only).

WeatherBench2 ERA5 6-hourly, 1991-2020, 00Z and 12Z (the verified valid times) averaged, day-of-year mean then a
+/-7-day circular window, on the verifier's 1.5 deg cell-centre grid, Northern Hemisphere, float16 - the same recipe as
build_t2m_clim_6h.py. The store's temperature chunks hold 8 time steps x all 13 levels, so one level costs the full
read; a 6-day stride (every third chunk) keeps it to ~1,800 chunks and still leaves ~150 samples in every +/-7-day
window (fetch_strat_clim.py used a 5-day stride for the same reason).

Output: scripts/verify/data/clim/clim_1p5_t850.npz  keys t850 (366, 60, 240) float16 KELVIN, lat (60,), lon (240,)
Run once on the laptop; published as the `verify-clim-t850-v1` release asset, which aifs-verify-backfill.yml downloads
into the clim directory and merges onto the frames branch, where aifs-compare.yml seeds it from.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aifs_station_verify as V  # noqa: E402

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
OUT = V.DATA / "clim" / "clim_1p5_t850.npz"
STRIDE_CHUNKS = 3                                   # chunks are 8 six-hourly steps = 2 days


def main() -> int:
    t0 = time.time()
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    da = ds["temperature"].sel(level=850, time=slice("1991-01-01", "2020-12-31"))
    n = da.sizes["time"]
    keep = np.zeros(n, bool)
    for c0 in range(0, n, 8 * STRIDE_CHUNKS):
        keep[c0:c0 + 8] = True
    hrs = da.time.dt.hour.values
    keep &= (hrs == 0) | (hrs == 12)
    sub = da.isel(time=np.flatnonzero(keep))
    print(f"  {int(keep.sum())} of {n} steps (00/12Z, every {STRIDE_CHUNKS}rd chunk)", flush=True)
    vals = sub.load()
    print(f"  loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    doy = vals.time.dt.dayofyear.values
    arr = vals.transpose("time", "latitude", "longitude").values.astype(np.float64)
    lat_w, lon_w = vals.latitude.values, vals.longitude.values
    clim = np.full((366, len(lat_w), len(lon_w)), np.nan)
    for d in range(1, 367):
        sel = np.abs(((doy - d + 183) % 366) - 183) <= 7
        clim[d - 1] = arr[sel].mean(axis=0)
    cda = xr.DataArray(clim, coords=dict(dayofyear=np.arange(1, 367), latitude=lat_w, longitude=lon_w),
                       dims=("dayofyear", "latitude", "longitude"))
    cda = xr.concat([cda, cda.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
    lat_t, lon_t = V.grid_1p5()
    nh = lat_t > 0
    interp = cda.interp(latitude=lat_t[nh], longitude=lon_t, method="linear").values
    print(f"  range {np.nanmin(interp):.1f}..{np.nanmax(interp):.1f} K, nan {np.isnan(interp).mean():.4f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, t850=interp.astype(np.float16), lat=lat_t[nh].astype(np.float32),
                        lon=lon_t.astype(np.float32))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
