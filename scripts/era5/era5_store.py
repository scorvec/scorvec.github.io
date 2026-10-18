#!/usr/bin/env python3
"""Persistent on-disk store for raw ERA5 fields streamed from ARCO.

Its readers (get_u, get_v, get_sp, get_t2m_conus) stream a field from ARCO once on
a miss and save it to disk; afterwards it loads locally (~50 ms vs ~20 s). Level
requests are merged per timestamp — if a file holds levels {10,50,...} and a
later call wants {20,30} too, only the missing levels stream and the file is
rewritten with the union. So clim rebuilds, history refreshes and validation
runs never re-download a byte they've already seen.

Layout (root set by ERA5_STORE):
    $ERA5_STORE/
      README.md
      arco_0p25/
        u/1991/u_19910102T12.nc     one file per timestamp; `level` dim grows (v alike)
        sp/1991/sp_19910102T12.nc   surface pressure (float32)

Packing: the winds u, v are int16 with scale_factor 0.01 m/s (±327 m/s range,
0.01 precision) — 2 bytes/point, ~30 MB per 14-level timestamp; a packing for
winds only. sp and the CONUS t2m stay float32.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from localdirs import env_dir

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

_ds = None
_lock = threading.Lock()
_io = threading.Lock()  # HDF5/netCDF-C is not thread-safe: serialize disk I/O

_README = """# ERA5 store

Raw ERA5 0.25° fields streamed once from ARCO (Google Analysis-Ready ERA5)
and kept forever. Managed by scripts/era5/era5_store.py — do not hand-edit
file contents; deleting files is safe (they re-stream on next use).

Layout: arco_0p25/<var>/<year>/<var>_<YYYYMMDDTHH>.nc
  u  — zonal wind, int16-packed (scale 0.01 m/s), `level` dim holds whatever
       pressure levels have been needed so far for that timestamp (files are
       rewritten with the union when new levels are requested).
  sp — surface pressure, float32, Pa.
"""


def _arco():
    global _ds  # noqa: PLW0603  lazy store handle guarded by _lock (functools.cache could open it twice)
    with _lock:
        if _ds is None:
            _ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
            store = env_dir("ERA5_STORE")
            store.mkdir(parents=True, exist_ok=True)
            rd = store / "README.md"
            if not rd.exists():
                rd.write_text(_README)
    return _ds


def _path(var: str, t: pd.Timestamp) -> Path:
    return env_dir("ERA5_STORE", "arco_0p25", var, f"{t.year}", f"{var}_{t:%Y%m%dT%H}.nc")


def _write_atomic(ds: xr.Dataset, path: Path, encoding: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}-{threading.get_ident()}.nc")
    with _io:
        ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


def _get_pl(var: str, arco_name: str, time, levels) -> xr.DataArray:
    """Pressure-level field (level, latitude, longitude) float32 at the requested
    levels (hPa, any order → returned sorted ascending). Local-first, ARCO on miss."""
    t = pd.Timestamp(time)
    want = sorted(int(l) for l in levels)
    p = _path(var, t)
    have = None
    if p.exists():
        try:
            with _io, xr.open_dataarray(p) as f:
                have = f.load()
        except Exception:  # noqa: BLE001
            have = None  # corrupt → re-stream
    if have is not None:
        got = [int(l) for l in have.level.values]
        if set(want) <= set(got):
            return have.sel(level=want).astype("float32")
    missing = want if have is None else sorted(set(want) - {int(l) for l in have.level.values})
    fresh = _arco()[arco_name].sel(time=t, level=missing).load().astype("float32")
    if fresh.dims[0] != "level":  # single level squeezed
        fresh = fresh.expand_dims("level")
    if bool(np.isnan(fresh.values).all()):
        # ARCO pads not-yet-published dates with NaNs — never cache those, or
        # the date would stay empty forever once ERA5 backfills
        merged = fresh if have is None else xr.concat([have, fresh], dim="level")
        return merged.sortby("level").sel(level=want).astype("float32")
    merged = fresh if have is None else xr.concat([have, fresh], dim="level")
    merged = merged.sortby("level").rename(var)
    _write_atomic(
        merged.to_dataset(name=var),
        p,
        {var: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": False}},
    )
    return merged.sel(level=want).astype("float32")


def get_u(time, levels) -> xr.DataArray:
    """Zonal wind — see _get_pl."""
    return _get_pl("u", "u_component_of_wind", time, levels)


def get_v(time, levels) -> xr.DataArray:
    """Meridional wind — see _get_pl."""
    return _get_pl("v", "v_component_of_wind", time, levels)


CONUS = dict(latitude=slice(55, 20), longitude=slice(-130 % 360, -60 % 360))


def get_t2m_conus(time) -> xr.DataArray:
    """2 m temperature over CONUS (K, float32) — subset storage: full-field
    t2m would be ~30x the bytes for products that only sample points."""
    t = pd.Timestamp(time)
    p = _path("t2m_conus", t)
    if p.exists():
        try:
            with _io, xr.open_dataarray(p) as f:
                return f.load().astype("float32")
        except Exception:  # noqa: BLE001
            pass
    da = _arco()["2m_temperature"].sel(time=t).sel(**CONUS).load().astype("float32").rename("t2m")
    if not bool(np.isnan(da.values).all()):  # never cache NaN padding
        _write_atomic(da.to_dataset(name="t2m"), p, {"t2m": {"zlib": False}})
    return da


def get_sp(time) -> xr.DataArray:
    """Surface pressure (latitude, longitude) float32 Pa."""
    t = pd.Timestamp(time)
    p = _path("sp", t)
    if p.exists():
        try:
            with _io, xr.open_dataarray(p) as f:
                return f.load().astype("float32")
        except Exception:  # noqa: BLE001
            pass
    sp = _arco()["surface_pressure"].sel(time=t).load().astype("float32").rename("sp")
    if not bool(np.isnan(sp.values).all()):  # never cache ARCO's NaN padding
        _write_atomic(sp.to_dataset(name="sp"), p, {"sp": {"zlib": False}})
    return sp
