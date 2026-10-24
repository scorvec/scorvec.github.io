#!/usr/bin/env python3
"""The normals the polar-cap sea-level pressure and 500 hPa height forecasts (polarcap.py) are measured against.

ERA5 1991-2020 on WeatherBench 2's 1.5-deg cells (conservative cell means of ERA5, 240 x 121 with the poles), the
Northern Hemisphere from 19.5 N. Per cell a smooth curve through the twelve monthly means, every day of a month weighted
alike (scripts/lib/harmonic_clim.py: the mean and NHARM annual harmonics whose month averages match them):

  z500  the ERA5 store's daily means (the mean of 00, 06, 12 and 18 UTC, metres; offline: $ERA5_STORE)
        -> reference/z500_clim.nc, coef (term, latitude, longitude) in m. The 500 hPa height's daily cycle is a few
        metres, so the one curve serves forecasts valid at 00 and at 12 UTC.
  msl   the 00 and 12 UTC fields separately, from WeatherBench 2's 6-hourly zarr (one pass over 1991-2020, ~3.5 GB; in
        Actions, polarcap-clim.yml): sea-level pressure has a daily tide of about 1 hPa at 20 N, which a daily-mean
        normal would leave in the anomaly of a forecast valid at 00 or 12 UTC.
        -> reference/msl_clim.nc, coef (hour, term, latitude, longitude) in hPa, with each cell's surface height orog
        (m, geopotential_at_surface / g: the polar cap's sea-level pressure leaves out the high cells).

    python scripts/strat/build_polarcap_clim.py z500
    python scripts/strat/build_polarcap_clim.py msl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "lib"))
import harmonic_clim as HC

REF = HERE / "reference"
YEARS = (1991, 2020)
LAT = np.round(np.arange(13, 61) * 1.5, 2)  # 19.5..90 N
LON = np.round(np.arange(240) * 1.5, 2)  # 0..358.5 E
HOURS = (0, 12)
G = 9.80665
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def on_grid(da: xr.DataArray) -> xr.DataArray:
    """The cells LAT x LON of a (..., latitude, longitude) field, whatever its dimension order."""
    da = da.sel(latitude=LAT, longitude=LON, method="nearest", tolerance=1e-3)
    return da.transpose(*[d for d in da.dims if d not in ("latitude", "longitude")], "latitude", "longitude")


def fit(sums: np.ndarray, days: np.ndarray, what: str) -> np.ndarray:
    """The curve's coefficients through the monthly means sums / days (12, ...), and the fit's month residual."""
    means = sums / days.reshape(12, *([1] * (sums.ndim - 1)))
    coef = HC.fit(means)
    resid = np.tensordot(HC.design(), coef, axes=(1, 0)) - means
    q = np.percentile(np.sqrt((resid**2).mean(0)), [50, 99, 100]).round(3)
    print(f"  {what}: month-mean residual of the fit, rms per cell (median, 99 %, max): {q.tolist()}", flush=True)
    return coef


def save(path: Path, ds: xr.Dataset) -> None:
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def z500() -> None:
    from localdirs import env_dir

    s, n = np.zeros((12, LAT.size, LON.size)), np.zeros(12)
    for y in range(YEARS[0], YEARS[1] + 1):
        with xr.open_dataset(env_dir("ERA5_STORE", "wb2_1p5_daily", "z500", f"z500_{y}.nc")) as ds:
            z = on_grid(ds["z500"]).load()  # metres, whatever the attributes say
        if not 4500 < float(z.mean()) < 6000:
            raise SystemExit(f"z500_{y}.nc: not the 500 hPa height in metres")
        month = z.time.dt.month.values
        for m in range(12):
            s[m] += z.values[month == m + 1].sum(0)
            n[m] += (month == m + 1).sum()
    coef = fit(s, n, "z500")
    save(
        REF / "z500_clim.nc",
        xr.Dataset(
            {"coef": (("term", "latitude", "longitude"), coef.astype("float32"))},
            coords={"term": np.arange(coef.shape[0]), "latitude": LAT, "longitude": LON},
            attrs={
                "source": f"ERA5 500 hPa height, daily means {YEARS[0]}-{YEARS[1]} (WeatherBench 2, 1.5 deg)",
                "method": f"mean + {HC.NHARM} annual harmonics fitted to the month means (scripts/lib/harmonic_clim.py)",
                "units": "m",
            },
        ),
    )


def msl() -> None:
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"}, chunks=None)
    p = ds["mean_sea_level_pressure"]
    s, n = np.zeros((12, len(HOURS), LAT.size, LON.size)), np.zeros(12)
    for y in range(YEARS[0], YEARS[1] + 1):
        t = pd.DatetimeIndex(p.time.sel(time=str(y)).values)
        keep = t[t.hour.isin(HOURS)]
        v = on_grid(p.sel(time=keep)).load() / 100.0  # Pa -> hPa
        hours = v.time.dt.hour.values
        month = v.time.dt.month.values
        for m in range(12):
            for k, h in enumerate(HOURS):
                s[m, k] += v.values[(month == m + 1) & (hours == h)].sum(0)
            n[m] += ((month == m + 1) & (hours == HOURS[0])).sum()
        print(f"  {y}: {len(keep)} fields, mean {float(v.mean()):.1f} hPa", flush=True)
    coef = fit(s, n, "msl")  # (term, hour, lat, lon)
    orog = on_grid(ds["geopotential_at_surface"]).load().values / G
    tide = np.abs(coef[0, 1] - coef[0, 0]).mean(1)  # the annual means' 12 - 00 UTC difference, by latitude
    print(
        "  |12 - 00 UTC| (hPa) at "
        + ", ".join(f"{la} N {tide[np.searchsorted(LAT, la)]:.2f}" for la in (19.5, 45, 70.5))
    )
    save(
        REF / "msl_clim.nc",
        xr.Dataset(
            {
                "coef": (("hour", "term", "latitude", "longitude"), coef.transpose(1, 0, 2, 3).astype("float32")),
                "orog": (("latitude", "longitude"), orog.astype("float32")),
            },
            coords={"hour": list(HOURS), "term": np.arange(coef.shape[0]), "latitude": LAT, "longitude": LON},
            attrs={
                "source": f"ERA5 mean sea-level pressure at 00 and 12 UTC, {YEARS[0]}-{YEARS[1]} (WeatherBench 2, 1.5 deg)",
                "method": f"mean + {HC.NHARM} annual harmonics fitted to the month means (scripts/lib/harmonic_clim.py)",
                "units": "hPa; orog m",
            },
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("var", choices=("z500", "msl"))
    a = ap.parse_args()
    {"z500": z500, "msl": msl}[a.var]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
