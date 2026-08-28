#!/usr/bin/env python3
"""Daily 1.5° NH ERA5 layer of the local store (~/era5_store/wb2_1p5_daily).

Sits beside arco_0p25/ in the same store: streamed once, kept forever,
deleting files is safe (they re-stream on the next populate run).
Per-variable yearly files:

    ~/era5_store/wb2_1p5_daily/<var>/<var>_<YYYY>.nc   (time, latitude 0-90N, longitude)

  z500  geopotential height at 500 hPa, daily mean of the 4 synoptic hours (m)
  t2m   2 m temperature, daily mean (K)
  prcp  total precipitation, daily sum (mm)
  u200 / v200  daily-mean 200 hPa wind (m/s)
  zplev geopotential height at all 13 WB2 levels (time, level, lat, lon; m)
  slp   mean sea-level pressure, daily mean (hPa)

Sources: WB2 ERA5 1.5° 6-hourly (1959-2023), ARCO 0.25° subsampled ::6 for the
tail (2023->present). Resumable: complete (var, year) files are skipped, so
re-running is always safe and only missing pieces stream. Var-major loop order
— the first --vars entry finishes across all years before the next starts.

    python wb2_daily_store.py --vars z500 --start 1974 --end 2023
    python wb2_daily_store.py --start 1974 --end 2023            # all vars
    python wb2_daily_store.py --source arco --start 2023 --end 2026
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

STORE = Path(os.environ.get("ERA5_STORE", "~/era5_store")).expanduser()
LAYER = STORE / "wb2_1p5_daily"

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665

VARS = ("z500", "t2m", "prcp", "u200", "v200", "zplev", "slp")
LEVS13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


GLOBAL = False                                          # --global: skip the NH clip


def open_fields(source):
    """{var: 6-hourly DataArray} on the 1.5° grid, 0-90N (or global with
    --global), oriented S->N."""
    if source == "wb2":
        ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
        sel = {}
    else:
        # dask-backed (zarr-native chunks): each ARCO chunk spans all 37 levels
        # at 0.25 deg, so chunks must stream+release — chunks=None accumulates
        # them and OOMs. Worker count capped to bound decode memory.
        import dask
        dask.config.set(scheduler="threads", num_workers=4)
        ds = xr.open_zarr(ARCO, storage_options={"token": "anon"})
        sel = dict(latitude=slice(None, None, 6), longitude=slice(None, None, 6))
    pv = "total_precipitation_6hr" if "total_precipitation_6hr" in ds else "total_precipitation"
    out = {
        "z500": ds["geopotential"].sel(level=500) / G,
        "t2m": ds["2m_temperature"],
        "prcp": ds[pv],
        "u200": ds["u_component_of_wind"].sel(level=200),
        "v200": ds["v_component_of_wind"].sel(level=200),
        "zplev": ds["geopotential"].sel(level=LEVS13) / G,
        "slp": ds["mean_sea_level_pressure"] / 100.0,
    }
    out = {k: (v if k == "zplev" else v.drop_vars("level", errors="ignore"))
           for k, v in out.items()}
    if sel:
        out = {k: v.isel(**sel) for k, v in out.items()}
    if GLOBAL:
        return {k: v.sortby("latitude") for k, v in out.items()}
    return {k: v.sortby("latitude").sel(latitude=slice(0, 90)) for k, v in out.items()}


def year_complete(fp, source, y, end):
    if not fp.exists():
        return False
    try:
        with xr.open_dataset(fp) as ds:
            n = ds.sizes["time"]
        return n >= 360 or (source == "arco" and y < end)   # tail year always re-pulled
    except Exception:                                       # noqa: BLE001
        return False                                        # corrupt -> re-stream


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wb2", "arco"], default="wb2")
    ap.add_argument("--vars", nargs="+", default=list(VARS), choices=VARS)
    ap.add_argument("--start", type=int, default=1974)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--global", dest="glob", action="store_true",
                    help="all latitudes -> sibling layer wb2_1p5_daily_global/")
    a = ap.parse_args()
    global GLOBAL, LAYER
    if a.glob:
        GLOBAL = True
        LAYER = STORE / "wb2_1p5_daily_global"

    fields = open_fields(a.source)
    for var in a.vars:                                      # var-major: finish z500 first
        src = fields[var]
        for y in range(a.start, a.end + 1):
            fp = LAYER / var / f"{var}_{y}.nc"
            if year_complete(fp, a.source, y, a.end):
                print(f"  {var} {y}: exists — skipped", flush=True)
                continue
            t0 = time.time()
            if a.source == "arco":                          # hourly; month-wise to bound memory
                months = []
                for m in range(1, 13):
                    vm = src.sel(time=slice(f"{y}-{m:02d}", f"{y}-{m:02d}"))
                    if vm.sizes["time"] == 0:
                        continue
                    vm = vm.sel(time=vm.time.dt.hour.isin([0, 6, 12, 18]))
                    r = vm.resample(time="1D")
                    months.append((r.sum() * 1000.0 if var == "prcp" else r.mean()).compute())
                if not months:
                    print(f"  {var} {y}: no data — not cached", flush=True)
                    continue
                day = xr.concat(months, dim="time")
            else:
                vy = src.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
                r = vy.resample(time="1D")
                day = (r.sum() * 1000.0 if var == "prcp" else r.mean()).compute()
            if bool(np.isnan(day.values).all()):            # ARCO NaN padding — never cache
                print(f"  {var} {y}: all-NaN (unpublished) — not cached", flush=True)
                continue
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(f".tmp{os.getpid()}.nc")
            enc = {var: dict(zlib=True, complevel=1)} if var == "zplev" else None
            day.astype("float32").rename(var).to_netcdf(tmp, encoding=enc)
            os.replace(tmp, fp)
            print(f"  {var} {y}: {day.sizes['time']} days in {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
