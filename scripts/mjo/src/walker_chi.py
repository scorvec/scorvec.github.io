#!/usr/bin/env python3
"""Shared pieces for the tropical velocity-potential history behind the weekly Walker departure
product (walker_chi_weekly.py) and the MJO time-longitude chart (mjo_chi.py): the χ strip on the
χ-solver's own grid, the rolling history, and where the reference's SST regression is significant.

The history is float32 χ at 200 and 850 hPa, |lat| ≤ 32°, one record per AIFS-ENS 0-h analysis;
~40 KB per analysis, so it lives on the frames branch (assets/sst/anim/walker/) with a seed in
scripts/mjo/data/reference/ (walker_chi_backfill.py) for a cold runner — never in main's history.
The seed is rebuilt whenever the solver changes, so its records replace the rolling file's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LMAX = 42
FINE = 4 * (LMAX + 1) - 1  # the divergence's grid: its DH2 grid holds LMAX's as every 4th point
LEVELS = (200, 850)
STRIP = 32.0
MAXN = 260  # ~130 days of two analyses a day
HIST_NAME = "walker_chi_history.nc"
SEED = Path(__file__).resolve().parent.parent / "data" / "reference" / "walker_chi_history_seed.nc"
# Benjamini-Hochberg false discovery rate: over the strip's grid points, and over the weekly board's matches
FDR_Q = 0.10
N_EFF = 62  # independent samples among the 90 training months, for the R²-based F-test (r2_pvalues)


def chi_strip(u2d: xr.DataArray, v2d: xr.DataArray):
    """χ (m² s⁻¹) of degrees ≤ LMAX on LMAX's DH2 grid restricted to |lat| ≤ STRIP. Returns (chi, lat, lon).

    The divergence is taken on the FINE grid: winds sampled straight onto LMAX's own 4° grid alias their
    smaller scales into χ, ~0.4e6 m² s⁻¹ of noise in a 15°S-15°N mean from 0.25° analyses."""
    from wind200_vpot import velocity_potential  # pyshtools: only the producers need it, not the sst.yml renderer

    chi, dlat, dlon = velocity_potential(u2d, v2d, lmax=FINE, keep=LMAX)
    chi, dlat, dlon = chi[::4, ::4], dlat[::4], dlon[::4]
    m = np.abs(dlat) <= STRIP
    return chi[m].astype("float32"), dlat[m], dlon


def record(chis: dict, lat, lon, valid) -> xr.DataArray:
    arr = np.stack([chis[l] for l in LEVELS])[None]
    return xr.DataArray(
        arr,
        dims=("time", "level", "latitude", "longitude"),
        coords={"time": [pd.Timestamp(valid)], "level": list(LEVELS), "latitude": lat, "longitude": lon},
        name="chi",
        attrs={
            "units": "m2 s-1",
            "note": f"velocity potential from AIFS-ENS 0-h analyses, spherical harmonics lmax {LMAX} (divergence on the lmax {FINE} grid)",
        },
    )


def load_history(path: Path) -> xr.DataArray | None:
    """The rolling history merged with the seed (union of analysis times; the seed's record where both hold one),
    or None."""
    parts = []
    for p in (path, SEED):
        if p.exists():
            try:
                parts.append(xr.open_dataarray(p).load())
            except Exception as ex:  # noqa: BLE001
                print(f"  walker_chi: cannot read {p.name} ({str(ex)[:60]})", flush=True)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    a, b = parts
    a = a.sel(time=~a.time.isin(b.time))
    return xr.concat([a, b], dim="time").sortby("time") if a.time.size else b


def write_history(path: Path, da: xr.DataArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    da.to_netcdf(tmp, encoding={"chi": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    tmp.replace(path)


def update_history(path: Path, cur: xr.DataArray) -> xr.DataArray:
    old = load_history(path)
    if old is not None:
        old = old.sel(time=old.time != cur.time.values[0])
        da = xr.concat([old, cur], dim="time").sortby("time") if old.time.size else cur
    else:
        da = cur
    da = da.isel(time=slice(-MAXN, None))
    write_history(path, da)
    return da


def r2_pvalues(clim: xr.Dataset, month: int):
    """Masks (ENSO part, Indian+Atlantic part) of the grid points where those predictors are significant in the
    month's regression (walker_chi_clim.nc): nested F-tests from the reference's in-sample R² (ENSO only, full),
    Benjamini-Hochberg over both maps. The 90 training months (±1 month x 30 years) are not independent - the months
    of a year share their anomalies - so the residual degrees of freedom are N_EFF - 7: N_EFF = 62 is the smallest
    effective sample, over months and predictor groups, for which this test reproduces the whole-year permutation
    test of the band-mean W (walker_basins.perm_pvalues), with which it then agrees at 89-99 % of longitudes. The χ
    reference keeps no training anomalies that could be permuted directly."""
    from quantmet.stats import benjamini_hochberg
    from scipy import stats

    r2e = clim.r2.sel(month=month, stage="enso").values.astype(float)
    r2f = clim.r2.sel(month=month, stage="full").values.astype(float)
    df2 = N_EFF - len(clim.pred) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        den = (1 - r2f) / df2
        p = np.stack([stats.f.sf(r2e / 2 / den, 2, df2), stats.f.sf((r2f - r2e) / 4 / den, 4, df2)])
    sig = benjamini_hochberg(p, FDR_Q)
    return sig[0], sig[1]
