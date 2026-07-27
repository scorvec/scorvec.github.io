#!/usr/bin/env python3
"""Daily NH outcome fields for the teleconnection compositor.

Per day (from WB2 ERA5 1.5deg 6-hourly, then the ARCO tail):
  z500  geopotential height at 500 hPa, daily mean of the 4 synoptic hours (m)
  t2m   2 m temperature, daily mean (K)
  prcp  total precipitation, daily sum (mm)
  u200 / v200  daily-mean 200 hPa wind (m/s) — WAF is computed in the
               compositor from these (perturbation streamfunction form)

Domain: 0-90N (WAF needs some subtropics; composites display 20-90N).
Output: one file per year, scripts/telecon/data/outcomes/outcomes_YYYY.nc
(float32, ~150 MB/yr — local only, gitignored). Resumable: existing complete
years are skipped.

    python build_outcome_fields.py --source wb2  --start 1959 --end 2023
    python build_outcome_fields.py --source arco --start 2023 --end 2026
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
OUTD = HERE / "data" / "outcomes"

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665


def open_source(source):
    if source == "wb2":
        ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
        sel = {}
    else:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        sel = dict(latitude=slice(None, None, 6), longitude=slice(None, None, 6))
    out = {
        "z500": ds["geopotential"].sel(level=500) / G,
        "t2m": ds["2m_temperature"],
        "u200": ds["u_component_of_wind"].sel(level=200),
        "v200": ds["v_component_of_wind"].sel(level=200),
    }
    # each pl field keeps its own scalar `level` coord -> Dataset merge conflict
    out = {k: v.drop_vars("level", errors="ignore") for k, v in out.items()}
    pv = "total_precipitation_6hr" if "total_precipitation_6hr" in ds else "total_precipitation"
    out["prcp"] = ds[pv]
    if sel:
        out = {k: v.isel(**sel) for k, v in out.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wb2", "arco"], default="wb2")
    ap.add_argument("--start", type=int, default=1959)
    ap.add_argument("--end", type=int, default=2023)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)

    fields = open_source(a.source)
    fields = {k: v.sortby("latitude").sel(latitude=slice(0, 90)) for k, v in fields.items()}

    for y in range(a.start, a.end + 1):
        fp = OUTD / f"outcomes_{y}.nc"
        if fp.exists():
            try:
                n = xr.open_dataset(fp).sizes["time"]
                if n >= 360 or (a.source == "arco" and y < a.end):
                    print(f"  {y}: exists ({n} d) — skipped", flush=True); continue
            except Exception:                              # noqa: BLE001
                pass
        t0 = time.time()
        sel = dict(time=slice(f"{y}-01-01", f"{y}-12-31"))
        day = {}
        for k, v in fields.items():
            vy = v.sel(**sel)
            if a.source == "arco":                         # hourly -> 6-hourly first
                vy = vy.sel(time=vy.time.dt.hour.isin([0, 6, 12, 18]))
            r = vy.resample(time="1D")
            day[k] = (r.sum() * 1000.0 if k == "prcp" else r.mean()).compute()
        enc = {k: {"dtype": "float32"} for k in day}
        xr.Dataset(day).to_netcdf(fp, encoding=enc)
        print(f"  {y}: {day['z500'].sizes['time']} days in {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
