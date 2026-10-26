"""An update of the month-ahead outlook (forecast.py --update), for when the stratosphere changes course after the
issue on the 15th: the target month is re-issued from the medium range and the predicted stratosphere.

  days 1-7      the IFS-ENS and AIFS-ENS ensemble-mean anomaly of the target month's first week (medium_week.py)
  days 8 to end this method (mo_method) trained for those days (the archive's block 1, part=1), with the lower-vortex
                predictor taken over days 1-7 (vortex="week"): every year of the library its ERA5 value, the year
                being forecast the value of each member of IFS-ENS, AIFS-ENS, GEFS and GEPS (vortex_window.py; the
                ensembles a quarter each); its forecast is the mean of the forecasts for VORTEX_Q quantiles of those
                values
  the month     the days-weighted sum of the two (7 days and the rest). Its tercile probabilities average those of
                the VORTEX_Q forecasts, each from the month's 1991-2020 climatology shifted by the forecast and
                narrowed by the error of the two parts: the days-from-the-8th hindcast's cross-validated error and the
                first week's ensemble spread, each weighted by its share of the days. The skill mask is the
                days-from-the-8th hindcast's (the first week's forecast has no hindcast here).

No GEFS blend: the first week comes from the two ECMWF ensembles, and GEFS's weeks 3-4 are the part its reforecast
gives least weight.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np

import blend as B
import forecast as F
import mo_grid as G
import mo_method as M

VORTEX_Q = 21  # the forecasts for the days from the 8th: one per quantile of the vortex members' distribution


def vortex_quantiles(vx: dict) -> np.ndarray:
    """VORTEX_Q values of the vortex predictor: the quantiles at the bin midpoints of the members' distribution with
    each ensemble weighing a quarter (each of its members 1 / (4 n))."""
    vals, wts = [], []
    for e in vx["ensembles"].values():
        vals += e["window"]
        wts += [1.0 / (len(vx["ensembles"]) * len(e["window"]))] * len(e["window"])
    order = np.argsort(vals)
    cw = np.cumsum(np.array(wts)[order])
    q = (np.arange(VORTEX_Q) + 0.5) / VORTEX_Q
    return np.array(vals)[order][np.minimum(np.searchsorted(cw, q), len(vals) - 1)]


def vortex_summary(vx: dict, res: dict) -> dict:
    """The update's vortex for the page: its days, run, value and anomaly (against the 1991-2020 mean of the same days
    in the archive), each ensemble's anomaly and the members' 10-90 % range."""
    clim = vx["mean"] - res["indices"]["u50v"]  # the predictor is the anomaly
    return {
        "window": vx["window"],
        "cycle": vx["cycle"],
        "era5_days": vx["era5"]["days"],
        "analysis_days": len(vx["analyses"]),
        "value": round(vx["mean"], 2),
        "anomaly": round(res["indices"]["u50v"], 2),
        "normal": round(clim, 2),
        "ensembles": {m: round(e["mean"] - clim, 2) for m, e in vx["ensembles"].items()},
        "q10_q90": [round(float(np.percentile(vortex_quantiles(vx), p)) - clim, 2) for p in (10, 90)],
    }


def issue(ty: int, tm: int, A: M.Archive, vortex: Path, medium: Path, original: Path) -> dict:
    """The update of target (ty, tm): forecast.run's result for the days from the 8th, with each variable's issued
    forecast switched to the month (the days from the 8th kept under stat_*, the first week under mr_fc)."""
    if not original.exists():
        raise SystemExit(f"--update: the {ty}-{tm:02d} outlook has not been issued")
    vx = json.loads(vortex.read_text())
    if vx["target"] != f"{ty}-{tm:02d}":
        raise SystemExit(f"--update: {vortex} is the vortex of {vx['target']}")
    res = F.run(ty, tm, A, vortex="week", vortex_live=vx["mean"], vortex_members=vortex_quantiles(vx), part=1)
    if "analogs" not in res:
        raise SystemExit(f"the predictor window for {ty}-{tm:02d} is not complete in the archive")
    n0 = G.SPLIT_DAY - 1
    w0 = n0 / calendar.monthrange(ty, tm)[1]
    w1 = 1.0 - w0
    with np.load(medium) as mw:
        mr = {v: (mw[f"{v}_anom"].astype(np.float64), mw[f"{v}_sd"].astype(np.float64)) for v in M.TARGETS}
        med = {"cycle": str(mw["cycle"]), "days": [str(mw["first"]), str(mw["last"])],
               "members": {k[:-2].split("_", 1)[1]: int(mw[k]) for k in mw.files if k.startswith("z500_") and k.endswith("_n")}}  # fmt: skip
    for v, o in res["vars"].items():
        mean0, sd0 = mr[v]
        for key in ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw"):
            o[f"stat_{key}"] = o[key]
        o["mr_fc"] = mean0
        fcs = w0 * mean0 + w1 * o["fc_members"]
        o["fc"] = fcs.mean(0)
        clim, lo, hi = B.obs_clim(A, v, tm)
        err = o["hind_obs"] - o["hind_cv"]
        mse = (err**2).mean(0)
        scale = w1 * np.sqrt(1.0 + (w0 * sd0) ** 2 / np.maximum(w1 * w1 * mse, 1e-12))  # both parts' errors
        b_, a_ = F.mixed_terciles(fcs, err * scale, lo, hi, clim)
        o["p_below_raw"], o["p_above_raw"] = b_, a_
        o["p_below"], o["p_above"] = np.where(o["sig"], b_, 1 / 3), np.where(o["sig"], a_, 1 / 3)
        o["tlo"], o["thi"] = lo, hi
    res["update"] = {
        "of": json.loads(original.read_text())["issued"],
        "vortex": vortex_summary(vx, res),
        "medium": med,
        "weights": {"first_week": round(w0, 3), "rest": round(w1, 3)},
    }
    return res
