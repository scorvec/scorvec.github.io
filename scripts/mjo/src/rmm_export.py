"""The multi-model RMM phase diagram's data (assets/mjo/rmm_models.json): the wind-only RMM of the AIFS-ENS and
IFS-ENS forecasts (ensemble mean and every member), the observed track, and the ENSO correction (enso_rmm.py), which
mjo.html applies in the browser so the same diagram can show the raw or the ENSO-removed index for every model.

The GEPS and GEFS pages write the same track layout (track()) for their ensembles to assets/geps/data and
assets/gefs/data; the page reads all three files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

INDEX_NOTE = "wind-only RMM (U850 + U200 projected onto the WH04 wind EOFs), previous 120-day mean removed"


def _r(a, nd=3):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float).ravel()]


def track(r1, r2, leads, valid, label: str, init: str, anchor=None, note: str = "") -> dict:
    """r1/r2 (member, lead) -> one model's track: the ensemble mean (its first point replaced by ``anchor``, the
    analysis, when given), the members, lead days and valid times."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    m1, m2 = r1.mean(0), r2.mean(0)
    if anchor is not None:
        m1[0], m2[0] = anchor
    return {
        "label": label,
        "init": init,
        "n": int(r1.shape[0]),
        "lead": _r(leads, 2),
        "t": [pd.Timestamp(v).strftime("%Y-%m-%dT%H") for v in valid],
        "r1": _r(m1),
        "r2": _r(m2),
        "m1": [_r(x, 2) for x in r1],
        "m2": [_r(x, 2) for x in r2],
        "note": note,
    }


def ecmwf_track(ds: xr.Dataset, init: pd.Timestamp, label: str) -> dict:
    leads = ds["lead_day"].values.astype(float)
    valid = [init + pd.Timedelta(days=float(x)) for x in leads]
    members = [str(x) for x in ds["member"].values]
    cf = members.index("cf") if "cf" in members else 0
    anchor = (float(ds["rmm1_wind"][cf, 0]), float(ds["rmm2_wind"][cf, 0]))  # day 0 is the control's analysis
    return track(
        ds["rmm1_wind"].values, ds["rmm2_wind"].values, leads, valid, label, init.strftime("%Y-%m-%d %HZ"), anchor
    )


def write(
    path: Path,
    init: pd.Timestamp,
    rmm: xr.Dataset,
    rmm_ifs: xr.Dataset | None,
    obs: xr.Dataset | None,
    dn: float | None = None,
    c: np.ndarray | None = None,
    n34: pd.Series | None = None,
) -> None:
    models = {"aifs": ecmwf_track(rmm, init, "AIFS-ENS")}
    if rmm_ifs is not None:
        models["ifs"] = ecmwf_track(rmm_ifs, init, "IFS-ENS")
    o = None
    if obs is not None and obs.sizes.get("time", 0):
        t = pd.to_datetime(obs["time"].values)
        o = {"t": [x.strftime("%Y-%m-%dT%H") for x in t], "r1": _r(obs["rmm1"].values), "r2": _r(obs["rmm2"].values)}
        if n34 is not None and c is not None:
            from enso_rmm import delta_n34

            d = np.array([delta_n34(n34, x) for x in t])
            o["dn"] = _r(np.where(np.isfinite(d), d, 0.0))  # the correction per observed day (ENSO moves over 60 days)
    out = {
        "updated": datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ"),
        "init": init.strftime("%Y-%m-%d %HZ"),
        "index": INDEX_NOTE,
        "dn34": None if dn is None or not np.isfinite(dn) else round(float(dn), 3),
        "c": None if c is None else _r(c, 4),
        "obs": o,
        "models": models,
    }
    path.write_text(json.dumps(out, separators=(",", ":")))
