"""
The real-time RMM's low-frequency filter map: the trailing 120-day-mean U850/U200 anomaly on the EOF grid that is
subtracted from the forecast, following Wheeler & Hendon (2004), held fixed across lead time.

refresh_map120.py builds it from ARCO-ERA5 in mjo.yml and writes it with save_map120; run_rmm.py reads it with
load_map120. _to_eof_grid is the shared regridding onto the 0-357.5 degree 2.5-degree grid, 15S-15N band mean.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from setup_reference import REF_DIR, lat_mean

GRID_RES = 2.5

MAP120_PATH = REF_DIR / "wind_map120.nc"  # forecast 120-day filter map (committed)


def save_map120(mean120: dict, path: Path = MAP120_PATH) -> None:
    """Persist the 120-day-mean U850/U200 anomaly maps.

    An optional "window_end" (YYYY-MM-DD of the last analysis day in the
    120-day window) is stored as an attribute so consumers can tell
    how stale the low-frequency filter is — the file's mtime can't (git
    checkouts rewrite it).
    """
    lon = np.arange(0, 360, GRID_RES)
    ds = xr.Dataset(
        {"u850": ("longitude", mean120["u850"]), "u200": ("longitude", mean120["u200"])},
        coords={"longitude": lon},
    )
    if mean120.get("window_end"):
        ds.attrs["window_end"] = str(mean120["window_end"])
    ds.to_netcdf(path)


def load_map120(path: Path = MAP120_PATH) -> dict:
    d = xr.open_dataset(path)
    out = {"u850": d["u850"].values, "u200": d["u200"].values}
    if "window_end" in d.attrs:
        out["window_end"] = str(d.attrs["window_end"])
    return out


# ── coordinate-name helpers (ERA5 naming drifts between products) ─────────────
def _coord(da, *names):
    for n in names:
        if n in da.coords or n in da.dims:
            return n
    raise KeyError(f"none of {names} in {list(da.coords)}")


def _to_eof_grid(da: xr.DataArray) -> xr.DataArray:
    """Return (time, lon) on the 0–357.5° 2.5° grid, latitude-band averaged."""
    lonn = _coord(da, "longitude", "lon")
    if float(da[lonn].min()) < 0:
        da = da.assign_coords({lonn: da[lonn] % 360}).sortby(lonn)
    lon_new = np.arange(0, 360, GRID_RES)
    da = da.interp({lonn: lon_new})
    return lat_mean(da)  # weighted 15°S–15°N mean, retains longitude
