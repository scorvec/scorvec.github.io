"""
Observed-RMM history (the growing 'truth' record).

Seeded from recent ERA5 analysis, then extended each run with the AIFS-ENS
analysis (control member, lead-day 0) — a zero-lag, self-consistent observed
value (same model as the forecast, so no ERA5→AIFS handoff jump).  Over time
the AIFS-grown portion replaces the ~5-day-lagged ERA5 seed.

Store: data/reference/obs_history.nc  (time, rmm1, rmm2)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ARCHIVE = Path("data/reference/obs_history.nc")


def append_truth(date, rmm1: float, rmm2: float, path: Path = ARCHIVE) -> xr.Dataset:
    """Append (or replace) one observed RMM point at its ACTUAL valid time.

    Truth used to be keyed to the normalized calendar day: the 12Z run's
    lead-0 (a 12Z analysis) overwrote the 00Z run's point and displayed half
    a day early, so the observed track's tail hopped back and forth between
    the 00Z and 12Z builds. Keeping the real epoch gives a clean 12-hourly
    track; a re-run of the same cycle still replaces its own point."""
    t = np.datetime64(pd.Timestamp(date), "ns")
    new = xr.Dataset(
        {"rmm1": ("time", [float(rmm1)]), "rmm2": ("time", [float(rmm2)])},
        coords={"time": [t]},
    )
    if path.exists():
        old = xr.open_dataset(path).load()
        old = old.sel(time=old.time != t)          # de-dup same date
        out = xr.concat([old, new], dim="time").sortby("time")
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    return out


def seed(obs: xr.Dataset, path: Path = ARCHIVE) -> None:
    """Initialise the history from an ERA5 observed-RMM dataset (time/rmm1/rmm2),
    merging with anything already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base = obs[["rmm1", "rmm2"]]
    if path.exists():
        old = xr.open_dataset(path).load()
        keep = old.sel(time=~old.time.isin(base.time))
        base = xr.concat([keep, base], dim="time")
    base.sortby("time").to_netcdf(path)


def load_truth(path: Path = ARCHIVE, days: int | None = None) -> xr.Dataset | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    if days is not None:
        ds = ds.isel(time=slice(-days, None))
    ds.attrs.setdefault("source", "ERA5/AIFS")
    return ds
