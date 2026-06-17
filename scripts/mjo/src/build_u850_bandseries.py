#!/usr/bin/env python3
"""Extract the full 1959–2023 ERA5 850 mb zonal-wind 5°S–5°N band-mean Hovmöller
series from WeatherBench2 (1.5° conservative regrid — 70× faster than the 37-level
ARCO store, and band-averaging makes the resolution difference <0.1 m/s).

Output: data/reference/eq_u850_bandseries.nc — a small (time, longitude) daily array
on the 1° longitude grid, the raw input for the analog-event Hovmöllers. Caching the
whole record once means clim / detrend / per-event windows are all instant downstream.

    python src/build_u850_bandseries.py                 # full 1959–2023 (~4–5 min)
    python src/build_u850_bandseries.py --start 1980    # shorter test
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# ARCO over-reads all 37 levels per chunk, so fewer time-steps ≈ proportionally faster.
# Default 4×/day (matches WB2's daily-mean); set ARCO_HOURS=12 for a ~4× quicker 1×/day
# snapshot (the diurnal cycle of band-mean u850 over ocean is <0.3 m/s — invisible here).
ARCO_HOURS = [int(h) for h in os.environ.get("ARCO_HOURS", "0,6,12,18").split(",")]
LAT_BAND = 5.0
LON_GRID = np.arange(0.0, 360.0, 1.0)
OUT = Path(__file__).resolve().parent.parent / "data" / "reference" / "eq_u850_bandseries.nc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wb2", "arco"], default="wb2",
                    help="wb2 = fast 1.5° (1959–2023); arco = native 0.25° (post-2023 tail)")
    ap.add_argument("--start", type=int, default=1959)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    url = WB2 if args.source == "wb2" else ARCO
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=850).sortby("latitude").sel(latitude=slice(-6, 6))
    if args.source == "arco":
        u = u.sel(time=u.time.dt.hour.isin(ARCO_HOURS))    # match WB2's 4×/day cadence
    lat = u.latitude
    w = np.cos(np.deg2rad(lat)).where(np.abs(lat) <= LAT_BAND, 0.0)
    print(f"opened {args.source} in {time.time()-t0:.1f}s; extracting {args.start}–{args.end} …", flush=True)

    parts = []
    for y in range(args.start, args.end + 1):
        ty = time.time()
        uy = u.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        band = uy.weighted(w).mean("latitude").resample(time="1D").mean().compute()
        parts.append(band)
        print(f"  {y}: {band.sizes['time']} days in {time.time()-ty:.1f}s", flush=True)

    series = xr.concat(parts, dim="time")
    if float(series.longitude.min()) < 0:
        series = series.assign_coords(longitude=series.longitude % 360).sortby("longitude")
    series = series.interp(longitude=LON_GRID)              # 1° grid (matches eq_hovmoller)
    series.name = "u850"
    series.attrs.update(source="WeatherBench2 ERA5 1.5° conservative", band="5S-5N cos-weighted",
                        level="850 hPa", note="daily-mean zonal wind")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    series.to_netcdf(args.out)
    print(f"saved {args.out}  shape={tuple(series.shape)}  "
          f"({series.nbytes/1e6:.1f} MB) in {(time.time()-t0)/60:.1f} min total", flush=True)
    print(f"  full-record band mean: {float(series.mean()):+.2f} m/s "
          f"[{float(series.min()):+.1f}, {float(series.max()):+.1f}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
