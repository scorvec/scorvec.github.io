#!/usr/bin/env python3
"""Shared pieces for the tropical velocity-potential history behind the weekly Walker departure
product (walker_chi_weekly.py): the χ strip on the χ-solver's own grid and the rolling history.

The history is float32 χ at 200 and 850 hPa, |lat| ≤ 32°, one record per AIFS-ENS 0-h analysis;
~40 KB per analysis, so it lives on the frames branch (assets/sst/anim/walker/) with a one-off seed
in scripts/mjo/data/reference/ for a cold runner — never in main's history.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LMAX = 42
LEVELS = (200, 850)
STRIP = 32.0
MAXN = 260                                                   # ~130 days of two analyses a day
HIST_NAME = "walker_chi_history.nc"
SEED = Path(__file__).resolve().parent.parent / "data" / "reference" / "walker_chi_history_seed.nc"


def chi_strip(u2d: xr.DataArray, v2d: xr.DataArray):
    """χ (m² s⁻¹) on the DH2 grid restricted to |lat| ≤ STRIP. Returns (chi, lat, lon)."""
    from wind200_vpot import velocity_potential                       # pyshtools: only the producers need it, not the sst.yml renderer
    chi, dlat, dlon = velocity_potential(u2d, v2d, lmax=LMAX)
    m = np.abs(dlat) <= STRIP
    return chi[m].astype("float32"), dlat[m], dlon


def record(chis: dict, lat, lon, valid) -> xr.DataArray:
    arr = np.stack([chis[l] for l in LEVELS])[None]
    return xr.DataArray(arr, dims=("time", "level", "latitude", "longitude"),
                        coords={"time": [pd.Timestamp(valid)], "level": list(LEVELS), "latitude": lat, "longitude": lon}, name="chi",
                        attrs={"units": "m2 s-1", "note": f"velocity potential from AIFS-ENS 0-h analyses, spherical harmonics lmax {LMAX}"})


def load_history(path: Path) -> xr.DataArray | None:
    """The rolling history merged with the seed (union of analysis times), or None."""
    parts = []
    for p in (path, SEED):
        if p.exists():
            try:
                parts.append(xr.open_dataarray(p).load())
            except Exception as ex:                                  # noqa: BLE001
                print(f"  walker_chi: cannot read {p.name} ({str(ex)[:60]})", flush=True)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    a, b = parts
    b = b.sel(time=~b.time.isin(a.time))
    return xr.concat([a, b], dim="time").sortby("time") if b.time.size else a


def update_history(path: Path, cur: xr.DataArray) -> xr.DataArray:
    old = load_history(path)
    if old is not None:
        old = old.sel(time=old.time != cur.time.values[0])
        da = xr.concat([old, cur], dim="time").sortby("time") if old.time.size else cur
    else:
        da = cur
    da = da.isel(time=slice(-MAXN, None))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    da.to_netcdf(tmp, encoding={"chi": {"zlib": True, "complevel": 4, "dtype": "float32"}}); tmp.replace(path)
    return da
