#!/usr/bin/env python3
"""Read the LOCAL ERA5 store (~/era5_store) for the SEAS5 page — before any CDS pull.

The laptop keeps 100+ GB of ERA5 already streamed from WeatherBench-2 and ARCO
(scripts/era5/*.py manage it; layout in ~/era5_store/README.md). Anything the
SEAS5 page needs from ERA5 that the store holds comes from here; the CDS is only
for what the store lacks (evaporation, 10 m wind, solar radiation, Southern
Hemisphere precipitation, stratospheric winds).

What this module serves, and where it comes from:

  monthly(var)           monthly means [time, lat, lon] on the 1.5° grid
      t2m   wb2_1p5_daily_global/t2m   1991–2026, K                (global)
      z500  wb2_1p5_daily_global/z500  1991–2020, m² s⁻² → m       (global)
      slp   wb2_1p5_daily_global/slp   1991–2020, Pa → hPa         (global)
      tp    wb2_1p5_daily/prcp         1959–2023, m/day → mm/day   (0–90°N only)
      z<L>  wb2_1p5_daily/zplev        1959–2026, m, L in 50…1000  (0–90°N only)
      Cached after the first aggregation in data/seas5/era5/local_monthly_<var>.nc
      (a few minutes per variable the first time, seconds after).

  daily_t2m(years)       daily 2 m temperature [time, lat, lon] on the 1.5° global grid,
                         for population-weighted daily records.

Grids: latitude −90…90 (global) or 0…90 (NH), longitude 0…358.5 step 1.5.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STORE = Path(os.environ.get("ERA5_STORE", str(Path.home() / "era5_store")))
CACHE = Path(__file__).resolve().parent / "data" / "seas5" / "era5"

SOURCES = {
    "t2m": ("wb2_1p5_daily_global/t2m", "t2m", 1.0, 0.0),
    "z500": ("wb2_1p5_daily_global/z500", "z500", 1.0 / 9.80665, 0.0),
    "slp": ("wb2_1p5_daily_global/slp", "slp", 0.01, 0.0),
    "tp": ("wb2_1p5_daily/prcp", "prcp", 1000.0, 0.0),
}


def available() -> bool:
    return STORE.exists()


def _files(sub: str, stem: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(STORE / sub / f"{stem}_*.nc")))


def _year_of(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def _align(da: xr.DataArray) -> xr.DataArray:
    """The 2023+ tail files carry float32 coordinates (1.5000001…), so concatenating them with
    the WB2 years does an outer join and pads the grid with NaN rows. Snap every file to the
    exact 1.5° values before joining."""
    return da.assign_coords(latitude=np.round(da.latitude.values.astype("float64"), 3),
                            longitude=np.round(da.longitude.values.astype("float64"), 3))


def monthly(var: str, level: int | None = None) -> xr.DataArray | None:
    """Monthly means for the whole record the store holds, cached. `var` is one of
    SOURCES or "z" with a `level` from the 13-level stack. Returns None if absent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{var}{level}" if level else var
    cpath = CACHE / f"local_monthly_{key}.nc"
    if var == "z" and level:
        files = _files("wb2_1p5_daily/zplev", "zplev")
    else:
        if var not in SOURCES:
            return None
        files = _files(*SOURCES[var][:2])
    files = [f for f in files if "tail" not in f.stem]
    if not files:
        return None
    latest = max(f.stat().st_mtime for f in files)
    if cpath.exists() and cpath.stat().st_mtime >= latest:
        return xr.open_dataarray(cpath).load()
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        da = ds[list(ds.data_vars)[0]]
        if var == "z" and level:
            da = da.sel(level=level)
        da = _align(da.transpose("time", "latitude", "longitude"))
        parts.append(da.resample(time="1MS").mean().load())
        ds.close()
    out = xr.concat(parts, dim="time", join="exact").sortby("time").sortby("latitude")
    if var != "z":
        fac, off = SOURCES[var][2], SOURCES[var][3]
        out = out * fac + off
    out.name = key
    out.attrs["source"] = "local ERA5 store (WeatherBench-2 / ARCO), monthly mean of daily values"
    out.to_netcdf(cpath)
    return out


def daily_t2m(y0: int, y1: int, lat_slice=None, lon_slice=None, nh: bool = False) -> xr.DataArray | None:
    """Daily-mean 2 m temperature (K) on the 1.5° grid for y0..y1, optionally subset.
    nh=True reads the 0–90°N set, which goes back to 1959 (the global set starts in 1991)."""
    files = [f for f in _files("wb2_1p5_daily/t2m" if nh else "wb2_1p5_daily_global/t2m", "t2m") if y0 <= _year_of(f) <= y1]
    if not files:
        return None
    parts = []
    for f in files:
        da = _align(xr.open_dataset(f)["t2m"].transpose("time", "latitude", "longitude"))
        if lat_slice is not None:
            da = da.sel(latitude=lat_slice)
        if lon_slice is not None:
            da = da.sel(longitude=lon_slice)
        parts.append(da.load())
    return xr.concat(parts, dim="time", join="exact").sortby("time")


def to_lon180(da: xr.DataArray) -> xr.DataArray:
    """0…358.5 → −180…178.5 ordering, for matching model grids in −180…180."""
    lon = da.longitude.values
    return da.assign_coords(longitude=np.where(lon > 180, lon - 360, lon)).sortby("longitude")


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:] or ["t2m", "z500", "slp", "tp"]:
        m = monthly(v)
        print(v, None if m is None else (dict(m.sizes), str(m.time.values[0])[:7], str(m.time.values[-1])[:7]))
