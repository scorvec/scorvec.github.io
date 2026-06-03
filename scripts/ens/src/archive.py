#!/usr/bin/env python3
"""Rolling per-run median archive → the CHANGE products (change-vs-prev, 48h-trend).

Each run's ensemble-median field (all Day 0–15 leads) is stored keyed by its init
time, per (ensemble, variable). Differences are taken at the SAME VALID TIME: for a
current run at init I0, the field valid at V = I0 + lead is compared with an earlier
run's field that is ALSO valid at V. Because the leads are daily (0,24,…,360 h), the
valid times only line up when the earlier run is an integer number of DAYS back — so
change-vs-prev uses the run 24 h earlier and 48h-trend the run 48 h earlier.

Archive lives in data/archive/ (local run-to-run state, not a committed artifact);
only the rendered change frames get committed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from common import REF, grid, LEADS

ARCH = REF.parent / "archive"                       # scripts/ens/data/archive/
_VAR = "__xarray_dataarray_variable__"


def _path(ens: str, var: str) -> Path:
    return ARCH / f"ens_arch_{ens}_{var}.nc"


def archive_run(ens: str, var: str, init, med: np.ndarray, keep: int = 8) -> None:
    """Store this run's median `med` (nlead, lat, lon) under its init; keep the newest
    `keep` inits (≥3 days of daily runs → covers the 48 h-trend lookback)."""
    lat, lon = grid(var)
    init = pd.Timestamp(init)
    da = xr.DataArray(med[None].astype("float32"),
                      dims=("init", "lead", "latitude", "longitude"),
                      coords={"init": [init], "lead": list(LEADS),
                              "latitude": lat, "longitude": lon})
    p = _path(ens, var); ARCH.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = xr.open_dataarray(p).load()
        old = old.where(old.init != np.datetime64(init), drop=True)   # replace same init
        if old.sizes.get("init", 0):
            da = xr.concat([old, da], dim="init")
    da = da.sortby("init").isel(init=slice(-keep, None))
    da.to_netcdf(p, encoding={_VAR: {"zlib": True, "complevel": 4}})


def load_median(ens: str, var: str, init):
    """The archived median (nlead, lat, lon) for this exact init, or None."""
    p = _path(ens, var)
    if not p.exists():
        return None
    arch = xr.open_dataarray(p).load()
    init = np.datetime64(pd.Timestamp(init))
    if init not in arch.init.values:
        return None
    return arch.sel(init=init).values


def change_fields(ens: str, var: str, init, med: np.ndarray, hours_back: int):
    """current − (run `hours_back` h earlier) at the SAME VALID TIME, as (nlead,lat,lon).
    Returns None if that earlier run isn't archived; NaN at leads whose matching valid
    time isn't present in the earlier run (e.g. its longest lead)."""
    p = _path(ens, var)
    if not p.exists():
        return None
    arch = xr.open_dataarray(p).load()
    target = pd.Timestamp(init) - pd.Timedelta(hours=hours_back)
    inits = {pd.Timestamp(t) for t in arch.init.values}
    if target not in inits:
        return None
    prev = arch.sel(init=np.datetime64(target))           # (lead, lat, lon), valid=target+lead
    prev_leads = set(int(x) for x in arch.lead.values)
    out = np.full_like(med, np.nan, dtype="float32")
    for i, ld in enumerate(LEADS):
        pl = ld + hours_back                              # earlier run's lead at same valid time
        if pl in prev_leads:
            out[i] = med[i] - prev.sel(lead=pl).values
    return out
