#!/usr/bin/env python3
"""
Build the equatorial 10 m zonal-wind climatology used to anomalize the AIFS/IFS
ensemble Hovmöller. Averages 5°S–5°N and fits a smooth day-of-year harmonic
(mean + annual + semiannual) per longitude. Output: data/reference/eq_u10_clim.nc
(coeffs: coef × longitude), committed; the daily workflow only reads this.

Modes:
  --mode daily   (default) true ERA5 daily means, 1991-2020, fetched per-year
                 (standard hourly single-levels, 4×/day, averaged to daily).
                 Resumable: per-year files cached; safe to run overnight.
  --mode monthly ERA5 monthly means (one small request) — fast fallback; the
                 seasonal cycle is ~identical after the harmonic fit.

    python src/build_eq_wind_clim.py            # daily (overnight)
    python src/build_eq_wind_clim.py --mode monthly
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path("data/reference")
DAILY_DIR = REF / "era5_u10_daily"
MONTHLY_RAW = REF / "era5_u10_monthly.nc"
OUT = REF / "eq_u10_clim.nc"
LON_GRID = np.arange(0.0, 360.0, 1.0)
LAT_BAND = 5.0
YEARS = range(1991, 2021)
AREA = [6, -180, -6, 179.75]
GRID = [1.0, 1.0]


# ── CDS helpers ───────────────────────────────────────────────────────────────
def _is_throttle(msg: str) -> bool:
    m = msg.lower()
    return ("rejected" in m or "temporarily" in m or "429" in m or
            "too many" in m or "queue" in m)


def _retrieve(c, dataset, req, target, attempts=40, wait=120):
    for i in range(1, attempts + 1):
        try:
            c.retrieve(dataset, req, target)
            return True
        except Exception as e:
            if _is_throttle(str(e)) and i < attempts:
                print(f"    throttled (try {i}); waiting {wait}s …", flush=True)
                time.sleep(wait)
                continue
            print(f"    retrieve failed: {str(e)[:120]}", flush=True)
            return False
    return False


ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_HOURS = [0, 6, 12, 18]            # 4×/day -> daily mean


def build_arco():
    """True ERA5 daily-mean climatology streamed from ARCO-ERA5 (Google Cloud,
    no CDS queue). Per-year so progress is visible and memory bounded."""
    import time
    ds = xr.open_zarr(ARCO_URL, storage_options={"token": "anon"})
    u = ds["10m_u_component_of_wind"].sel(latitude=slice(6, -6))
    lat = u.latitude
    w = np.cos(np.deg2rad(lat)).where(np.abs(lat) <= LAT_BAND, 0.0)
    lon = u.longitude.values
    ssum = np.zeros((367, lon.size))
    scnt = np.zeros(367)
    for y in YEARS:
        uy = u.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        uy = uy.sel(time=uy.time.dt.hour.isin(ARCO_HOURS))
        daily = uy.weighted(w).mean("latitude").resample(time="1D").mean()
        t0 = time.time()
        dv = daily.compute()
        doy = pd.to_datetime(dv.time.values).dayofyear.values
        for i, dd in enumerate(doy):
            ssum[dd] += dv.values[i]
            scnt[dd] += 1
        print(f"  {y}: {dv.sizes['time']} days in {time.time()-t0:.0f}s", flush=True)
    clim = ssum[1:367] / np.maximum(scnt[1:367], 1)[:, None]
    clim_da = xr.DataArray(clim, dims=("doy", "longitude"),
                           coords={"doy": np.arange(1, 367), "longitude": lon})
    clim_da = (clim_da.assign_coords(longitude=clim_da.longitude % 360)
               .sortby("longitude").interp(longitude=LON_GRID))
    coeffs = _fit_harmonics(clim_da.values, clim_da.doy.values)
    _save(coeffs, "ERA5 1991-2020 daily means (ARCO-ERA5)")


def _band_series(u: xr.DataArray) -> xr.DataArray:
    """5°S–5°N cosine-weighted mean on the 1° longitude grid -> (time, lon)."""
    latn = "latitude" if "latitude" in u.coords else "lat"
    lonn = "longitude" if "longitude" in u.coords else "lon"
    lat = u[latn]
    w = np.cos(np.deg2rad(lat)).where(np.abs(lat) <= LAT_BAND, 0.0)
    band = u.weighted(w).mean(dim=latn)
    band = band.assign_coords({lonn: (band[lonn] % 360)}).sortby(lonn)
    return band.interp({lonn: LON_GRID})


def _fit_harmonics(clim_by_doy: np.ndarray, doys: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * doys / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w),
                         np.cos(2 * w), np.sin(2 * w)])
    return np.linalg.lstsq(X, clim_by_doy, rcond=None)[0]        # (5, nlon)


def build_daily():
    files = sorted(DAILY_DIR.glob("u10_*.nc"))
    print(f"building daily climatology from {len(files)} years …", flush=True)
    ds = xr.open_mfdataset(files, combine="by_coords")
    uname = "u10" if "u10" in ds else list(ds.data_vars)[0]
    band = _band_series(ds[uname])
    tname = "valid_time" if "valid_time" in band.coords else "time"
    daily = band.resample({tname: "1D"}).mean().compute()       # daily means
    t = pd.to_datetime(daily[tname].values)
    clim = daily.groupby(xr.DataArray(t.dayofyear, dims=tname, name="doy")).mean()
    coeffs = _fit_harmonics(clim.transpose("doy", "longitude").values,
                            clim["doy"].values)
    _save(coeffs, "ERA5 1991-2020 daily means")


def build_monthly():
    import cdsapi
    if not MONTHLY_RAW.exists():
        REF.mkdir(parents=True, exist_ok=True)
        print("requesting ERA5 monthly 10u …", flush=True)
        cdsapi.Client(quiet=True).retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {"product_type": "monthly_averaged_reanalysis",
             "variable": "10m_u_component_of_wind",
             "year": [str(y) for y in YEARS], "month": [f"{m:02d}" for m in range(1, 13)],
             "time": "00:00", "area": AREA, "grid": GRID, "format": "netcdf"},
            str(MONTHLY_RAW))
    ds = xr.open_dataset(MONTHLY_RAW)
    uname = "u10" if "u10" in ds else list(ds.data_vars)[0]
    band = _band_series(ds[uname])
    tname = "valid_time" if "valid_time" in band.coords else "time"
    t = pd.to_datetime(band[tname].values)
    clim = band.groupby(xr.DataArray(t.month, dims=tname, name="month")).mean()
    mid = pd.to_datetime([f"2001-{m:02d}-15" for m in range(1, 13)]).dayofyear.values
    coeffs = _fit_harmonics(clim.transpose("month", "longitude").values, mid)
    _save(coeffs, "ERA5 1991-2020 monthly means")


def _save(coeffs, base):
    xr.DataArray(coeffs, dims=("coef", "longitude"),
                 coords={"coef": np.arange(5), "longitude": LON_GRID},
                 attrs={"base": base, "var": "10u",
                        "lat_band": f"|lat|<= {LAT_BAND}"}).to_netcdf(OUT)
    print(f"  saved {OUT} ({base})", flush=True)


def eval_clim(coeffs: np.ndarray, doy) -> np.ndarray:
    w = 2 * np.pi * np.asarray(doy, float) / 365.25
    X = np.stack([np.ones_like(w), np.cos(w), np.sin(w),
                  np.cos(2 * w), np.sin(2 * w)], axis=-1)
    return X @ coeffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "monthly"], default="daily")
    args = ap.parse_args()
    if args.mode == "daily":
        build_arco()         # ERA5 daily means streamed from ARCO (no CDS)
    else:
        build_monthly()      # CDS monthly-means fallback
