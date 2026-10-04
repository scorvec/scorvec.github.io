#!/usr/bin/env python3
"""ERA5 1991-2020 climatology of total column water (precipitable water, kg m-2 = mm) for the AR page's PWAT anomaly
loops: day-of-year harmonics (mean + 3 annual harmonics), fitted separately at 00 and 12 UTC so the anomaly at a
00Z/12Z valid time carries no diurnal bias. Source: WeatherBench2 ERA5 6-hourly, 1.5 degrees, `total_column_water`
(the same quantity as ECMWF open data's `tcw`, cloud liquid and ice included). Domain 0-80N, 100E-70W.

    python scripts/ar/build_pwat_clim.py        # laptop or runner, ~5 min; writes scripts/ar/data/pwat_clim.nc
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
OUT = Path(__file__).resolve().parent / "data" / "pwat_clim.nc"
NH = 3


def design(doy):
    w = 2 * np.pi * np.asarray(doy, float) / 365.25
    cols = [np.ones_like(w)]
    for k in range(1, NH + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.column_stack(cols)


def main():
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    v = ds["total_column_water"].sel(time=slice("1991-01-01", "2020-12-31T18"), latitude=slice(0, 80),
                                     longitude=slice(100, 290))
    print("units", v.attrs.get("units"), dict(v.sizes))
    coefs = []
    for hh in (0, 12):
        parts = []
        for y in range(1991, 2021):
            x = v.sel(time=str(y))
            x = x.sel(time=x.time.dt.hour == hh).load().transpose("time", "latitude", "longitude")
            parts.append(x)
        da = xr.concat(parts, "time")
        print(f"{hh:02d}Z: {da.sizes['time']} days, mean {float(da.mean()):.1f} mm, range {float(da.min()):.1f}..{float(da.max()):.1f}")
        X = design(pd.DatetimeIndex(da.time.values).dayofyear)
        Y = da.values.reshape(da.sizes["time"], -1)
        c, *_ = np.linalg.lstsq(X, Y, rcond=None)
        coefs.append(c.reshape(X.shape[1], da.sizes["latitude"], da.sizes["longitude"]))
    out = xr.Dataset({"coef": (("hour", "k", "latitude", "longitude"), np.stack(coefs).astype(np.float32))},
                     coords={"hour": [0, 12], "latitude": da.latitude.values, "longitude": da.longitude.values},
                     attrs={"source": "WeatherBench2 ERA5 total_column_water, 1.5 deg", "base": "1991-2020",
                            "units": "kg m-2", "harmonics": NH, "note": "evaluate: coef . [1, cos w, sin w, cos 2w, ...], "
                            "w = 2 pi doy / 365.25; one set per valid hour"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
