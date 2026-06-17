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


def trend_fields(ens: str, var: str, init, med: np.ndarray,
                 hours_back: int = 48, step_h: int = 24):
    """Per-VALID-TIME least-squares slope (units/day) of the forecast across the runs
    from `init` back to init−hours_back (spaced step_h). For each valid time V = init+
    lead, the forecast for V from each run is gathered (that run's matching lead) and a
    line is fit vs run-init time, so the slope is how the forecast for that fixed valid
    time is TRENDING run-to-run (warmer +, colder −). Returns (nlead,lat,lon) or None if
    the needed earlier runs aren't all archived; NaN at leads whose lookback exceeds the
    archived lead range."""
    p = _path(ens, var)
    if not p.exists():
        return None
    arch = xr.open_dataarray(p).load()
    inits_have = {pd.Timestamp(t) for t in arch.init.values}
    backs = list(range(0, hours_back + 1, step_h))                 # 0, 24, 48
    runs = [pd.Timestamp(init) - pd.Timedelta(hours=h) for h in backs]
    if not all(r in inits_have for r in runs):
        return None
    arch_leads = set(int(x) for x in arch.lead.values)
    x = -np.array(backs, float) / 24.0                            # run time (days): 0,-1,-2
    xx = x - x.mean()
    out = np.full_like(med, np.nan, dtype="float32")
    for i, ld in enumerate(LEADS):
        ys, ok = [], True
        for h, r in zip(backs, runs):
            rl = ld + h                                           # that run's lead to reach V
            if rl in arch_leads:
                ys.append(arch.sel(init=np.datetime64(r), lead=rl).values)
            else:
                ok = False; break
        if ok:
            Y = np.stack(ys)                                      # (nrun, lat, lon)
            out[i] = np.tensordot(xx, Y - Y.mean(0), axes=([0], [0])) / (xx ** 2).sum()
    return out


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
