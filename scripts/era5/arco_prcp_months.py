#!/usr/bin/env python3
"""Daily precipitation for specific months from ARCO ERA5 hourly, onto the store's
1.5 deg 0-90N grid, as monthly tail pieces: wb2_1p5_daily/prcp/tail_pieces/prcp_YYYY-MM.nc

WHY. The WB2 daily zarr the store was built from ends 2023-01-10 for precipitation
(prcp_2023.nc holds ten days), and wb2_daily_store's ARCO path samples four hours
a day and sums them, which for an HOURLY accumulation field is a quarter of the
rain. This sums all 24 hourly totals (m -> mm) and block-means the 0.25 deg field
to the 1.5 deg nodes (6 x 6 cells centred on each node), which is what the
conservative WB2 grid approximates. First use: DJF 2023/24 for the SEAS5 case study.

    python scripts/era5/arco_prcp_months.py 2023-12 2024-01 2024-02
"""
from __future__ import annotations
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

STORE = Path(os.environ.get("ERA5_STORE", "~/era5_store")).expanduser()
OUT = STORE / "wb2_1p5_daily" / "prcp" / "tail_pieces"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LAT = np.arange(0.0, 90.01, 1.5); LON = np.arange(0.0, 359.9, 1.5)


def coarsen(field_025: np.ndarray, lat025: np.ndarray, lon025: np.ndarray) -> np.ndarray:
    """0.25 deg (lat 90..-90, lon 0..359.75) -> 1.5 deg nodes by a 6x6 block mean centred on the node."""
    out = np.empty((LAT.size, LON.size), dtype=np.float32)
    lat_idx = {v: i for i, v in enumerate(np.round(lat025, 2))}; lon_n = lon025.size
    for j, la in enumerate(LAT):
        i0 = lat_idx[round(la, 2)]; rows = slice(max(i0 - 3, 0), min(i0 + 3, lat025.size))          # +-0.75 deg
        for k, lo in enumerate(LON):
            c = int(round(lo / 0.25)); cols = [(c + d) % lon_n for d in range(-3, 3)]
            out[j, k] = field_025[rows][:, cols].mean()
    return out


def one_day(ds, day: pd.Timestamp):
    v = ds["total_precipitation"].sel(time=slice(day, day + pd.Timedelta(hours=23)))
    arr = v.values.sum(axis=0) * 1000.0                                # 24 hourly accumulations, m -> mm
    return coarsen(arr, ds.latitude.values, ds.longitude.values)


def main():
    months = sys.argv[1:] or ["2023-12", "2024-01", "2024-02"]
    OUT.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    for ym in months:
        fp = OUT / f"prcp_{ym}.nc"
        if fp.exists():
            print(f"{ym}: exists", flush=True); continue
        days = pd.date_range(f"{ym}-01", periods=pd.Period(ym).days_in_month, freq="D")
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex: fields = list(ex.map(lambda d: one_day(ds, d), days))
        da = xr.DataArray(np.stack(fields), dims=("time", "latitude", "longitude"), coords={"time": days, "latitude": LAT, "longitude": LON}, name="prcp",
                          attrs={"long_name": "Total precipitation", "units": "mm/day", "source": "ARCO ERA5 hourly, 24-h sum, 6x6 block mean to 1.5 deg"})
        da.to_netcdf(fp); print(f"{ym}: {len(days)} days, mean {float(da.mean()):.2f} mm/day, {time.time() - t0:.0f} s", flush=True)
    print("PRCP_DONE")


if __name__ == "__main__":
    main()
