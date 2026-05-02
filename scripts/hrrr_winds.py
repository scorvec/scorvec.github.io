"""
HRRR 80m wind ingestion via Herbie.

HRRR `wrfsfcf` (surface) GRIB2 files contain UGRD and VGRD on the 80 m
above-ground level directly — no extrapolation from pressure levels needed.
For density correction we also pull TMP/PRES at the surface and a 2-m
temperature; the standard ISO 61400-12 method scales air density with
station-pressure-corrected temperature at hub height.

The HRRR GRIB messages we want, by GRIB2 search regex:

    :UGRD:80 m above ground:
    :VGRD:80 m above ground:
    :TMP:2 m above ground:
    :PRES:surface:
    :HGT:surface:           # for terrain elevation, used in P→ρ

Operational vs archive
----------------------
Herbie auto-routes:
  - Recent cycles → NOMADS / Google / Azure mirrors
  - Older cycles  → AWS s3://noaa-hrrr-bdp-pds/ public archive

Both are handled by `Herbie(date, model="hrrr", product="sfc", fxx=...)`.

Returns
-------
For each call, an xarray Dataset with variables:
    u80, v80         (m/s)
    t2m              (K)
    psfc             (Pa)
    hgt              (m)
on the native HRRR Lambert Conformal grid. The function `interp_to_points`
samples the dataset to turbine lat/lon using nearest-neighbor (HRRR is 3
km — interpolation between cells doesn't add much for fleet-level forecasts
and nearest-neighbor matches what most operational shops do).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fetch_hrrr_winds(cycle_dt: datetime, fxx: int):
    """Pull 80m winds + density variables for a single (cycle, fxx) pair.

    Parameters
    ----------
    cycle_dt : datetime
        HRRR run initialization time (UTC). Hour must be 0..23.
    fxx : int
        Forecast hour (0..48 for the extended HRRR runs at 00/06/12/18 UTC,
        0..18 otherwise).

    Returns
    -------
    xarray.Dataset with u80, v80, t2m, psfc, hgt.
    """
    try:
        from herbie import Herbie
    except ImportError as e:
        raise RuntimeError(
            "Herbie is required for HRRR ingestion. "
            "Install with: pip install herbie-data"
        ) from e
    import xarray as xr  # noqa: WPS433

    H = Herbie(cycle_dt, model="hrrr", product="sfc", fxx=fxx)

    # Pull each field separately — Herbie's xarray() supports a search regex.
    # Note: Herbie returns a list of datasets when multiple GRIB messages
    # match different `typeOfLevel` values; we pluck what we need from each.
    ds_uv = H.xarray(r":[UV]GRD:80 m above ground:")
    ds_t2 = H.xarray(r":TMP:2 m above ground:")
    ds_ps = H.xarray(r":PRES:surface:")
    ds_hg = H.xarray(r":HGT:surface:")

    out = xr.Dataset(
        data_vars=dict(
            u80=ds_uv["u"].rename("u80") if "u" in ds_uv else ds_uv["u80"],
            v80=ds_uv["v"].rename("v80") if "v" in ds_uv else ds_uv["v80"],
            t2m=ds_t2["t2m"] if "t2m" in ds_t2 else list(ds_t2.data_vars.values())[0],
            psfc=ds_ps["sp"] if "sp" in ds_ps else list(ds_ps.data_vars.values())[0],
            hgt=ds_hg["orog"] if "orog" in ds_hg else list(ds_hg.data_vars.values())[0],
        ),
        attrs=dict(
            cycle=cycle_dt.isoformat(),
            fxx=fxx,
            model="hrrr",
            product="sfc",
        ),
    )
    return out


def fetch_hrrr_forecast(cycle_dt: datetime,
                        fxx_range: Iterable[int]):
    """Concat multiple forecast hours into a single dataset along a `fxx` dim."""
    import xarray as xr
    pieces = []
    for fxx in fxx_range:
        ds = fetch_hrrr_winds(cycle_dt, fxx)
        ds = ds.expand_dims(valid_time=[pd.Timestamp(cycle_dt) +
                                        pd.Timedelta(hours=fxx)])
        pieces.append(ds)
    return xr.concat(pieces, dim="valid_time")


# ---------------------------------------------------------------------------
# Spatial sampling
# ---------------------------------------------------------------------------

def interp_to_points(ds, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbor sample of HRRR grid at turbine lat/lon points.

    Returns a DataFrame indexed (point_id, valid_time) with columns
    u80, v80, ws80, wd80, t2m, psfc, hgt.
    """
    import xarray as xr  # noqa: WPS433

    # HRRR has 2-D `latitude` and `longitude` coordinates on its
    # Lambert Conformal grid. We cannot use ds.sel(lat=..., lon=...);
    # we need a KD-tree style nearest lookup.
    grid_lat = ds["latitude"].values
    grid_lon = ds["longitude"].values
    if grid_lon.max() > 180:
        grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Build flat KD-tree
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([grid_lat.ravel(), grid_lon.ravel()]))
    _, idx = tree.query(np.column_stack([lats, lons]), k=1)
    iy, ix = np.unravel_index(idx, grid_lat.shape)

    rows = []
    for i, (yi, xi) in enumerate(zip(iy, ix)):
        sub = ds.isel(y=yi, x=xi) if "y" in ds.dims else \
              ds.isel(latitude=yi, longitude=xi)
        df = sub.to_dataframe().reset_index()
        df["point_id"] = i
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)

    # Derive wind speed / direction
    out["ws80"] = np.hypot(out["u80"], out["v80"])
    out["wd80"] = (np.degrees(np.arctan2(-out["u80"], -out["v80"])) + 360.0) % 360.0
    return out
