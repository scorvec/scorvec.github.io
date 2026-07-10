#!/usr/bin/env python3
"""1991–2020 ERA5 daily climatologies (t2m, mslp, z500) for the climate monitor.

Source: WeatherBench2's conservatively regridded 1.5° ERA5 zarr (6-hourly,
1959–2023) — the same public store build_aam_density_clim uses, so no CDS
queue and no auth. Per variable: accumulate per-(month, day) daily means over
1991–2020 (Feb 29 folds into Feb 28), then fit mean + NHARM annual harmonics
per gridpoint. Output: era5_clim_<var>.nc (~1 MB each, committed); the daily
monitor evaluates clim(doy) from the coefficients.

Resumable per variable (skips vars whose output exists).

    python scripts/climate/build_era5_clim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
YEARS = range(1991, 2021)
NHARM = 4

VARS = {
    # name: source var, pressure level (or None), scale, offset  → display units
    "t2m":  dict(src="2m_temperature",          level=None, scale=1.0,        offset=-273.15),  # °C
    "mslp": dict(src="mean_sea_level_pressure", level=None, scale=0.01,       offset=0.0),      # hPa
    "z500": dict(src="geopotential",            level=500,  scale=1 / 9.80665, offset=0.0),     # m
}

# (month, day) → 0..364 bin, Feb 29 folded onto Feb 28
_DOY = {}
for _m in range(1, 13):
    for _d in range(1, 32):
        try:
            _DOY[(_m, _d)] = pd.Timestamp(2001, _m, _d).dayofyear - 1
        except ValueError:
            pass
_DOY[(2, 29)] = _DOY[(2, 28)]


def harmonic_fit(clim365: np.ndarray) -> np.ndarray:
    """(365, ny, nx) daily normals → (1+2·NHARM, ny, nx) harmonic coefficients."""
    x = 2 * np.pi * np.arange(365) / 365.0
    cols = [np.ones(365)]
    for h in range(1, NHARM + 1):
        cols += [np.cos(h * x), np.sin(h * x)]
    A = np.column_stack(cols)                                  # (365, 9)
    flat = clim365.reshape(365, -1)
    beta, *_ = np.linalg.lstsq(A, flat, rcond=None)
    return beta.reshape((A.shape[1],) + clim365.shape[1:]).astype("float32")


def build_var(ds: xr.Dataset, name: str, spec: dict) -> Path:
    out = HERE / f"era5_clim_{name}.nc"
    if out.exists():
        print(f"  {out.name} exists — skipping")
        return out
    da = ds[spec["src"]]
    if spec["level"] is not None:
        da = da.sel(level=spec["level"])
    ny, nx = da.sizes["latitude"], da.sizes["longitude"]
    ssum = np.zeros((365, ny, nx), "float64")
    scnt = np.zeros(365, "int32")
    for yr in YEARS:
        sel = da.sel(time=str(yr))
        daily = (sel.resample(time="1D").mean()
                 .transpose("time", "latitude", "longitude").compute())
        dates = pd.to_datetime(daily.time.values)
        vals = daily.values
        for i, d in enumerate(dates):
            k = _DOY[(d.month, d.day)]
            ssum[k] += vals[i]
            scnt[k] += 1
        print(f"  {name}: {yr} accumulated", flush=True)
    clim = (ssum / scnt[:, None, None]) * spec["scale"] + spec["offset"]
    coef = harmonic_fit(clim.astype("float32"))
    xr.Dataset(
        {"coef": (("coef_idx", "latitude", "longitude"), coef)},
        coords={"latitude": ds.latitude.values, "longitude": ds.longitude.values},
        attrs={"source": "ERA5 via WeatherBench2 1.5deg conservative", "base": "1991-2020",
               "nharm": NHARM, "var": spec["src"], "level": spec["level"] or 0,
               "units": {"t2m": "degC", "mslp": "hPa", "z500": "m"}[name]},
    ).to_netcdf(out)
    print(f"  wrote {out.name} ({out.stat().st_size // 1024} KB)", flush=True)
    return out


def main() -> int:
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(WB2), chunks={"time": 1464})
    for name, spec in VARS.items():
        build_var(ds, name, spec)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
