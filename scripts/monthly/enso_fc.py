#!/usr/bin/env python3
"""Claude's Niño-3.4 forecast: the next twelve months of Niño-3.4 and of the relative index (RONI), from the tropical
SST and the equatorial Pacific heat content, combined with the C3S seasonal models where they reach.

    python scripts/monthly/enso_fc.py --ersst sst.mnmean.nc            # hindcast table + the issue
    python scripts/monthly/enso_fc.py --ersst sst.mnmean.nc --hindcast-only

Statistical model. The state at the end of a month is the leading 8 EOFs of the monthly SST anomaly over 25S-25N
(ERSSTv6, every longitude), the leading 3 of the 0-300 m temperature anomaly at the 8 equatorial TAO/TRITON
moorings (165E to 95W), and Niño-3.4 and relative Niño-3.4 themselves; SST and heat content are linearly detrended
over the training years. For each lead (1-12 months) and calendar month of the start, Niño-3.4 (detrended; the
trend is added back) and relative Niño-3.4 (no trend: the tropical mean removes it) are regressed on the state,
ridge-damped (RIDGE x the sample size on the diagonal), with the starts of the two calendar months either side
pooled in (POOL). Training runs from 1980 (the heat-content record) to the last complete month.

The MJO: two more predictors, the mean RMM1 and RMM2 (three-field) over the 35 days from the 15th of the month after the
start, the issue date. They are trained on the observed RMM (ERA5, scripts/mjo/data/reference/rmm_era5.nc: perfect
prog) and forecast with the 35-day GEPS and GEFS runs (each model's newest within MJO_SLACK days of the 15th, the
three-field tracks of scripts/geps/mjo_geps.py and scripts/gefs/gefs_mjo.py; in the hindcast their reforecasts, GEPS
2001-2020 and GEFS 2000-2019, data/*_mjo_hc.npz), half each, GEFS damped (MJO_DAMPED), either alone when only one has
a run, nothing where neither has. Over 2001-2020 GEPS alone cut the mean-square error by 11 % at two months and 14 %
at three (95 % year-block bootstrap intervals excluding zero), most of what a perfect 35-day MJO forecast gives
(15 %, 16 %); GEFS alone, its 35-day mean a third too strong, 5 % (not significant), damped 7 %; the two together as
much as GEPS (11 %, 12 %) with intervals half as wide.

Heat content: ECMWF ORAS5 at the moorings (scripts/sst/reference/reanalysis_hc_monthly.csv, reanalysis_hc.py) for
training, then the TAO moorings themselves (assets/sst/data/subsurface_hc.json, sst_subsurface_events.py: each
mooring's monthly mean of the screened daily index), less each mooring's mean TAO-minus-ORAS5 difference over the
months both hold: ORAS5 lags real time by about six weeks, TAO by days.

Hindcast. Every start from 1980 is forecast with that year and its neighbours left out of everything fitted (the
EOFs, the trends and the regressions), so the scores and the error spreads are out of sample. Over 1993-2016 it
matched the C3S multi-system mean at every lead from 1 to 6 months (RMSE 0.48 against 0.49 °C at 6 months), and the
two averaged beat either (0.44).

Claude's forecast. The statistical forecast averaged with the C3S multi-system mean (assets/sst/data/
enso_forecast.json: the systems the ENSO page keeps, each system's member mean weighing the same) where the latest
C3S issue reaches, half each (C3S_WEIGHT); past the models' last month, the statistical forecast plus half the
models' last-month difference from it, decaying by DECAY a month (over 1993-2016 that beat the statistical forecast
alone at leads 7-9 and was no worse at 10-12). Its spread is the hindcast's: the statistical errors by start month
and lead, scaled where the models enter by the consensus-to-statistical RMSE ratio of the 1993-2016 hindcast
(c3s_hc_n34.npz, the C3S hindcast Niño-3.4 of each system), and the probabilities are normal about the forecast.

Outputs (the MJO run used, if any, under inputs.mjo): scripts/monthly/data/enso_hc.npz (the hindcast table, read by mo_method's forecast-ENSO component through
EnsoTable) and assets/monthly/data/enso_<YYYYMM>.json (the issue of month YYYYMM; latest.json names it).
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sst"))
from enso_models import EXCLUDE_SYSTEMS

DATA = HERE / "data"
OUT = REPO / "assets" / "monthly" / "data"
ERSST_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc"
HC_CSV = REPO / "scripts" / "sst" / "reference" / "reanalysis_hc_monthly.csv"
TAO_JSON = REPO / "assets" / "sst" / "data" / "subsurface_hc.json"
C3S_JSON = REPO / "assets" / "sst" / "data" / "enso_forecast.json"
C3S_HC = DATA / "c3s_hc_n34.npz"
RMM_NC = REPO / "scripts" / "mjo" / "data" / "reference" / "rmm_era5.nc"  # observed RMM (ERA5), daily
# the MJO forecasts: per model, its reforecast's 35-day mean RMM per start (GEPS 2001-2020, GEFS 2000-2019) and the
# newest run's three-field track (scripts/geps/mjo_geps.py, scripts/gefs/gefs_mjo.py)
MJO_MODELS = {
    "GEPS": (DATA / "geps_mjo_hc.npz", REPO / "assets" / "geps" / "data" / "geps_rmm.json"),
    "GEFS": (DATA / "gefs_mjo_hc.npz", REPO / "assets" / "gefs" / "data" / "gefs_rmm.json"),
}
MJO_DAMPED = ("GEFS",)  # its 35-day mean runs ~35 % too strong: scaled by the slope of the observed on it
RONI_SCALE = REPO / "scripts" / "sst" / "roni_sigma.json"
HC_TABLE = DATA / "enso_hc.npz"

T0 = "1980-01"  # the heat-content record's first month
LEADS = 12
K_SST, K_HC = 8, 3
SST_LAT = 25.0
POOL = 2
RIDGE = 0.05
CLIM = (1991, 2020)
SCORE_FROM = 1982  # hindcast scores and error spreads: starts from this year
MJO_DAYS = 35  # the MJO predictor: the mean RMM over the 35 days from the 15th of the month after the start
MJO_FROM = 15
MJO_SLACK = 3  # a run (reforecast or live) counts when it starts within this many days of that 15th
C3S_WEIGHT = 0.5
DECAY = 0.8
INDICES = ("n34", "rel")
CATS = (  # (key, label, lower, upper) on the RONI or Niño-3.4 scale of a 3-month mean
    ("lanina", "La Niña", -np.inf, -0.5),
    ("neutral", "Neutral", -0.5, 0.5),
    ("elnino", "El Niño", 0.5, np.inf),
)
STRONG = ((1.5, "strong"), (2.0, "very strong"))


# --------------------------------------------------------------------------------------------------- the inputs
def months_range(a: str, b: str) -> list[str]:
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def roni_scale() -> np.ndarray:
    """(12,) CPC's σ(ONI)/σ(relative) by calendar month (1991-2020)."""
    tab = json.loads(RONI_SCALE.read_text())["scale_by_month"]
    return np.array([float(tab[str(m)]) for m in range(1, 13)])


def ersst(path: Path, start: str = "1959-01") -> dict:
    """Monthly ERSSTv6 anomalies against 1991-2020: the 25S-25N field (T, cells) from T0, and Niño-3.4 and relative
    Niño-3.4 (Niño-3.4 minus the 20S-20N mean) from `start`, keyed by month."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        d = ds["sst"].load()
    if d.lat[0] > d.lat[-1]:
        d = d.isel(lat=slice(None, None, -1))
    months = [str(pd.Period(t, "M")) for t in d.time.values]
    x = d.values.astype(np.float64)
    mo = np.array([int(m[5:]) for m in months])
    yr = np.array([int(m[:4]) for m in months])
    for k in range(1, 13):
        s = mo == k
        x[s] -= np.nanmean(x[s & (yr >= CLIM[0]) & (yr <= CLIM[1])], 0)
    lat, lon = d.lat.values, d.lon.values
    la = (lat >= -5) & (lat <= 5)
    lo = (lon >= 190) & (lon <= 240)
    w = np.cos(np.deg2rad(lat[la]))
    n34 = np.nansum(x[:, la][:, :, lo] * w[None, :, None], (1, 2)) / (w.sum() * lo.sum())
    tr = (lat >= -20) & (lat <= 20)
    wt = np.cos(np.deg2rad(lat[tr]))[:, None] * np.isfinite(x[0, tr])
    trop = np.nansum(np.nan_to_num(x[:, tr]) * wt, (1, 2)) / wt.sum()
    band = (lat >= -SST_LAT) & (lat <= SST_LAT)
    field = x[:, band].reshape(len(months), -1)
    ok = np.isfinite(field[months.index(T0) :]).all(0)
    cw = np.sqrt(np.cos(np.deg2rad(np.repeat(lat[band], len(lon)))))[ok]
    i0 = months.index(start)
    return {
        "months": months,
        "field": field[:, ok] * cw,  # sqrt(cos)-weighted for the EOFs
        "n34": dict(zip(months[i0:], n34[i0:], strict=True)),
        "rel": dict(zip(months[i0:], (n34 - trop)[i0:], strict=True)),
    }


def heat_content(months: list[str]) -> tuple[np.ndarray, dict]:
    """(T, 8) 0-300 m anomaly at the 8 moorings for `months`: ORAS5 while it lasts, then TAO less each mooring's mean
    TAO-minus-ORAS5 difference over the months both hold; NaN where neither has the month."""
    csv = pd.read_csv(HC_CSV, index_col=0, parse_dates=True)
    cols = [c for c in csv.columns if c.startswith("oras5_")]
    ora = csv[cols]
    ora.index = ora.index.to_period("M").astype(str)
    out = ora.reindex(months).to_numpy(np.float64)
    meta = {"oras5_through": str(ora.dropna().index[-1])}
    if TAO_JSON.exists():
        tj = json.loads(TAO_JSON.read_text()).get("monthly")
        if tj:
            tao = pd.DataFrame(tj["anom"], index=tj["months"], columns=[f"oras5_{m}" for m in tj["moorings"]])[cols]
            both = tao.index.intersection(ora.dropna().index)
            if len(both) >= 6:  # the offset needs an overlap to be measured on
                off = (tao.loc[both] - ora.loc[both]).mean()
                for i, m in enumerate(months):
                    if np.isnan(out[i]).any() and m in tao.index:
                        out[i] = (tao.loc[m] - off).to_numpy()
                meta.update(tao_through=str(tao.index[-1]), tao_offset=dict(zip(tj["moorings"], off.round(3))),
                            tao_overlap=len(both))  # fmt: skip
    return out, meta


# ---------------------------------------------------------------------------------------------------- the model
def _detrend(x: np.ndarray, tt: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([np.ones(train.sum()), tt[train]], 1)
    coef = np.linalg.lstsq(X, x[train], rcond=None)[0]
    return x - (coef[0] + np.outer(tt, coef[1]) if x.ndim > 1 else coef[0] + tt * coef[1]), coef


def _pcs(x: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    mu = x[train].mean(0)
    _, s, vt = np.linalg.svd(x[train] - mu, full_matrices=False)
    return (x - mu) @ vt[:k].T / (s[:k] / np.sqrt(train.sum()))


class Model:
    """The state and the regressions for one training set (`train`: a mask over the months). The last two columns of
    the state are the MJO predictor: trained on the observed RMM (perfect prog), forecast with GEPS's."""

    def __init__(self, D: dict, train: np.ndarray):
        tt = np.arange(len(D["n34"])) / 120.0
        sst, _ = _detrend(D["field"], tt, train)
        hcd, _ = _detrend(D["hc"], tt, train)
        self.n34d, c = _detrend(D["n34"], tt, train)
        self.trend = lambda i: c[0] + c[1] * i / 120.0  # Niño-3.4's linear trend at month index i
        mjo = D["mjo_obs"]
        self.mjo_ok = np.isfinite(mjo).all(1)
        self.mjo_sd = mjo[train & self.mjo_ok].std(0)
        base = [_pcs(sst, train, K_SST), _pcs(hcd, train, K_HC), self.n34d[:, None], D["rel"][:, None]]
        self.X = np.concatenate([*base, np.nan_to_num(mjo) / self.mjo_sd], 1)
        self.rel = D["rel"]
        self.train = train

    def betas(self, mo: np.ndarray, start_month: int) -> dict:
        """{index: (LEADS, k) coefficients} for starts in calendar month `start_month`."""
        T = len(self.rel)
        dm = np.minimum((mo - start_month) % 12, (start_month - mo) % 12)
        out = {k: np.zeros((LEADS, self.X.shape[1])) for k in INDICES}
        for L in range(1, LEADS + 1):
            rows = np.where(self.train[: T - L] & self.train[L:] & self.mjo_ok[: T - L] & (dm[: T - L] <= POOL))[0]
            A = self.X[rows]
            AtA = A.T @ A + RIDGE * len(rows) * np.eye(A.shape[1])
            for k, y in (("n34", self.n34d), ("rel", self.rel)):
                out[k][L - 1] = np.linalg.solve(AtA, A.T @ y[rows + L])
        return out

    def forecast(self, i: int, b: dict, mjo: np.ndarray) -> dict:
        """{index: (LEADS,)} from the state at month index i with the MJO forecast `mjo` (RMM1, RMM2; zeros: none),
        Niño-3.4 with its trend added back."""
        x = np.r_[self.X[i, :-2], mjo / self.mjo_sd]
        f = {k: b[k] @ x for k in INDICES}
        f["n34"] = f["n34"] + np.array([self.trend(i + L) for L in range(1, LEADS + 1)])
        return f


def mjo_window(start: str) -> pd.Timestamp:
    """The first day of a start month's MJO predictor window: the 15th of the month after it (the issue date)."""
    return (pd.Period(start, "M") + 1).to_timestamp() + pd.Timedelta(days=MJO_FROM - 1)


def mjo_observed(months: list[str]) -> np.ndarray:
    """(T, 2) observed RMM1/RMM2 (ERA5, three-field) averaged over each start's window; NaN where under 80 % of it is
    in the record."""
    import xarray as xr

    with xr.open_dataset(RMM_NC) as ds:
        r = pd.DataFrame({"r1": ds["rmm1"].values, "r2": ds["rmm2"].values}, index=pd.DatetimeIndex(ds["time"].values))
    out = np.full((len(months), 2), np.nan)
    for i, m in enumerate(months):
        d0 = mjo_window(m)
        w = r.loc[d0 : d0 + pd.Timedelta(days=MJO_DAYS - 1)].dropna()
        if len(w) >= 0.8 * MJO_DAYS:
            out[i] = w.mean().to_numpy()
    return out


def mjo_matched(months: list[str], model: str) -> np.ndarray:
    """(T, 2) a model's reforecast 35-day mean RMM from the start nearest each window's first day (within MJO_SLACK
    days), NaN where it has none."""
    out = np.full((len(months), 2), np.nan)
    path = MJO_MODELS[model][0]
    if not path.exists():
        return out
    with np.load(path) as z:
        S, r = pd.DatetimeIndex(z["starts"]), np.stack([z["r1"], z["r2"]], 1)
    for i, m in enumerate(months):
        k = int(np.argmin(np.abs((S - mjo_window(m)).days)))
        if abs((S[k] - mjo_window(m)).days) <= MJO_SLACK:
            out[i] = r[k]
    return out


def mjo_combine(values: dict, damp: dict) -> np.ndarray:
    """The MJO forecast from the models that have one: the mean of each (damped where MJO_DAMPED); zeros (no MJO
    information) where none has. values: {model: (..., 2)} with NaN where missing."""
    x = np.stack([v * damp[m] for m, v in values.items()])  # (models, ..., 2)
    have = np.isfinite(x).all(-1, keepdims=True)
    n = have.sum(0)
    return np.where(n > 0, np.where(have, x, 0.0).sum(0) / np.maximum(n, 1), 0.0)


def mjo_hindcast(months: list[str], obs: np.ndarray) -> tuple[np.ndarray, dict]:
    """(T, 2) the combined reforecast MJO forecast of every start, and each model's damping (2,): the slope through
    the origin of the observed window means on its reforecast's, for MJO_DAMPED models (1 otherwise). Over 2001-2019
    GEPS + damped GEFS cut the error variance as much as GEPS alone (11 % at two months, 12 % at three) with intervals
    half as wide; damped GEFS alone 7 %, so it also stands in when no GEPS run is near the 15th."""
    vals = {m: mjo_matched(months, m) for m in MJO_MODELS}
    damp = {}
    for m, v in vals.items():
        ok = np.isfinite(v).all(1) & np.isfinite(obs).all(1)
        damp[m] = (np.array([(v[ok, k] * obs[ok, k]).sum() / (v[ok, k] ** 2).sum() for k in range(2)])
                   if m in MJO_DAMPED and ok.sum() > 50 else np.ones(2))  # fmt: skip
    return mjo_combine(vals, damp), damp


def mjo_live(start: str, damp: dict) -> dict | None:
    """The combined MJO forecast over the window of `start` from each model's newest run that starts within
    MJO_SLACK days of the window's first day (the days of its 35 from that day on): {"runs": {model: {init, days,
    rmm}}, "rmm": combined}, or None when no model has such a run."""
    d0, runs, vals = mjo_window(start), {}, {}
    for m, (_, track) in MJO_MODELS.items():
        if not track.exists():
            continue
        g = json.loads(track.read_text())
        init = pd.Timestamp(g["init"][:10])
        if abs((init - d0).days) > MJO_SLACK:
            continue
        t = pd.DatetimeIndex([x[:10] for x in g["t"]])
        sel = (t >= d0) & (t < d0 + pd.Timedelta(days=MJO_DAYS))
        r = np.array([np.array(g["r1"], float)[sel].mean(), np.array(g["r2"], float)[sel].mean()])
        runs[m] = {"init": g["init"], "days": int(sel.sum()), "rmm": [round(float(x), 3) for x in r]}
        vals[m] = r
    if not runs:
        return None
    rmm = mjo_combine({m: vals.get(m, np.full(2, np.nan)) for m in MJO_MODELS}, damp)
    return {"runs": runs, "rmm": [round(float(x), 3) for x in rmm]}


def inputs(ersst_path: Path) -> dict:
    E = ersst(ersst_path)
    months = E["months"][E["months"].index(T0) :]
    hc, meta = heat_content(months)
    field = E["field"][E["months"].index(T0) :]
    ok = np.isfinite(hc).all(1) & np.isfinite(field).all(1)
    last = int(np.where(ok)[0][-1])  # the newest month with both SST and heat content
    months, field, hc = months[: last + 1], field[: last + 1], hc[: last + 1]
    meta.update(sst_through=E["months"][-1], init=months[-1])
    mjo_obs = mjo_observed(months)
    mjo_fc, mjo_damp = mjo_hindcast(months, mjo_obs)
    return {
        "months": months,
        "mo": np.array([int(m[5:]) for m in months]),
        "yr": np.array([int(m[:4]) for m in months]),
        "field": field,
        "hc": hc,
        "n34": np.array([E["n34"][m] for m in months]),
        "rel": np.array([E["rel"][m] for m in months]),
        "mjo_obs": mjo_obs,
        "mjo_fc": mjo_fc,
        "mjo_damp": mjo_damp,
        "obs_n34": E["n34"],
        "obs_rel": E["rel"],
        "meta": meta,
    }


def hindcast(D: dict) -> dict:
    """Cross-validated forecasts {index: (T, LEADS)} of every start, each year's from a model fitted without that
    year and its neighbours."""
    T = len(D["months"])
    F = {k: np.full((T, LEADS), np.nan) for k in INDICES}
    for y in np.unique(D["yr"]):
        train = np.abs(D["yr"] - y) > 1
        m = Model(D, train)
        for sm in np.unique(D["mo"][D["yr"] == y]):
            b = m.betas(D["mo"], int(sm))
            for i in np.where((D["yr"] == y) & (D["mo"] == sm))[0]:
                f = m.forecast(int(i), b, D["mjo_fc"][i])
                for k in INDICES:
                    F[k][i] = f[k]
    return F


def truth(D: dict, k: str) -> np.ndarray:
    """(T, LEADS) the observed index at every start's leads (NaN beyond the record)."""
    obs = D[f"obs_{k}"]
    T = len(D["months"])
    out = np.full((T, LEADS), np.nan)
    for i, m in enumerate(D["months"]):
        p = pd.Period(m, "M")
        for L in range(1, LEADS + 1):
            out[i, L - 1] = obs.get(str(p + L), np.nan)
    return out


def error_sd(F: np.ndarray, O: np.ndarray, mo: np.ndarray, yr: np.ndarray, span: int = 1) -> np.ndarray:
    """(12, LEADS - span + 1) rms error of the `span`-month mean from lead L, by calendar month of the start, the
    starts of the month either side pooled in, scored from SCORE_FROM."""
    nL = LEADS - span + 1
    f = np.stack([F[:, L : L + span].mean(1) for L in range(nL)], 1)
    o = np.stack([O[:, L : L + span].mean(1) for L in range(nL)], 1)
    e2 = (f - o) ** 2
    out = np.zeros((12, nL))
    for m in range(1, 13):
        dm = np.minimum((mo - m) % 12, (m - mo) % 12)
        s = (dm <= 1) & (yr >= SCORE_FROM)
        out[m - 1] = np.sqrt(np.nanmean(e2[s], 0))
    return out


# --------------------------------------------------------------------------------- the C3S models and consensus
def c3s_issue() -> dict | None:
    """The newest C3S Niño-3.4 issue the ENSO page holds: {"issue", "months", "n34", "rel", "systems"} with each kept
    system's member mean per valid month (relative Niño-3.4 unscaled from the RONI scale)."""
    if not C3S_JSON.exists():
        return None
    e = json.loads(C3S_JSON.read_text())
    sc = roni_scale()
    months = e["valid_months"]
    out = {"issue": e["issue"], "months": months, "systems": {}}
    for name, m in e["models"].items():
        if name in EXCLUDE_SYSTEMS:
            continue
        n34 = [float(np.mean(x)) for x in m["n34"]]
        rel = [float(np.mean(x)) / sc[int(v[5:]) - 1] for x, v in zip(m["rnino"], months, strict=True)]
        out["systems"][name] = {"n34": n34, "rel": rel}
    if not out["systems"]:
        return None
    for k in INDICES:
        out[k] = np.mean([s[k] for s in out["systems"].values()], 0)
    return out


def consensus(stat: np.ndarray, c3s: np.ndarray | None) -> np.ndarray:
    """Claude's forecast from the statistical one (LEADS,) and the models' mean on the same months (NaN where they
    stop): the two averaged where the models reach, then the models' last difference decaying."""
    if c3s is None or not np.isfinite(c3s).any():
        return stat.copy()
    out = stat.copy()
    have = np.isfinite(c3s)
    out[have] = (1 - C3S_WEIGHT) * stat[have] + C3S_WEIGHT * c3s[have]
    last = int(np.where(have)[0][-1])
    d = C3S_WEIGHT * (c3s[last] - stat[last])
    for L in range(last + 1, LEADS):
        out[L] = stat[L] + d * DECAY ** (L - last)
    return out


def base_shift(obs: dict, month: int) -> float:
    """Observed 1993-2016 minus 1991-2020 mean of calendar month `month`: moves a C3S anomaly (against each system's
    1993-2016 hindcast) onto the 1991-2020 base."""
    v = lambda y0, y1: np.mean([obs[f"{y}-{month:02d}"] for y in range(y0, y1 + 1)])  # noqa: E731
    return float(v(1993, 2016) - v(*CLIM))


def c3s_hindcast_ratio(D: dict, F: dict) -> dict:
    """{index: (LEADS,)} RMSE of Claude's forecast over that of the statistical one in the 1993-2016 hindcast (a
    start's C3S partner is the C3S start of the month after it); 1 where untestable (no C3S table)."""
    if not C3S_HC.exists():
        return {k: np.ones(LEADS) for k in INDICES}
    z = np.load(C3S_HC)
    systems = [str(s) for s in z["systems"]]
    out = {}
    for k in INDICES:
        O = truth(D, k)
        e_s, e_c = np.zeros(LEADS), np.zeros(LEADS)
        for i, m in enumerate(D["months"]):
            p = pd.Period(m, "M") + 1  # the C3S start
            keys = [f"{s}_{p.month:02d}_{k}" for s in systems if f"{s}_{p.month:02d}_{k}" in z]
            if not 1993 <= p.year <= 2016 or len(keys) < 4:
                continue
            c = np.full(LEADS, np.nan)
            c[:6] = np.mean([z[key][p.year - 1993] for key in keys], 0)
            c[:6] += [base_shift(D[f"obs_{k}"], (p + L).month) for L in range(6)]
            f_s = F[k][i]
            f_c = consensus(f_s, c)
            ok = np.isfinite(O[i])
            e_s[ok] += (f_s[ok] - O[i][ok]) ** 2
            e_c[ok] += (f_c[ok] - O[i][ok]) ** 2
        out[k] = np.where(e_s > 0, np.sqrt(e_c / np.maximum(e_s, 1e-12)), 1.0)
    return out


def build_gefs_mjo_hc(src: Path) -> None:
    """data/gefs_mjo_hc.npz from the GEFS reforecast's RMM (gefs_mjo_hindcast.npz of release gefs-ref-v1, written by
    scripts/gefs/build_gefs_mjo.py: every member of the 2000-2019 Wednesday starts, projected as gefs_mjo.py does the
    live run): per start the members' mean over days 1-35."""
    with np.load(src) as z:
        starts = z["starts"]
        r1 = z["fm1"][:, :, :MJO_DAYS].astype(np.float64).mean((1, 2))
        r2 = z["fm2"][:, :, :MJO_DAYS].astype(np.float64).mean((1, 2))
    ok = np.isfinite(r1) & np.isfinite(r2)
    np.savez_compressed(
        MJO_MODELS["GEFS"][0], starts=starts[ok], r1=r1[ok].astype(np.float32), r2=r2[ok].astype(np.float32)
    )
    print(f"{MJO_MODELS['GEFS'][0].name}: {ok.sum()} starts {starts[ok][0]}..{starts[ok][-1]}")


def build_geps_mjo_hc(src: Path) -> None:
    """data/geps_mjo_hc.npz from the GEPS reforecast's RMM (scripts/geps/mjo_hindcast.py: mjo_hindcast.nc, the
    ensemble-mean three-field RMM of every 2001-2020 start, projected as mjo_geps.py does the live run): per start
    the mean over days 1-35."""
    import xarray as xr

    with xr.open_dataset(src) as ds:
        lead = ds["L"].values
        sel = (lead >= 0.5) & (lead <= MJO_DAYS - 0.5)  # day d is lead d - 0.5
        r1, r2 = ds["f1"].values[:, sel].mean(1), ds["f2"].values[:, sel].mean(1)
        starts = pd.DatetimeIndex(ds["S"].values)
    ok = np.isfinite(r1) & np.isfinite(r2)
    out = MJO_MODELS["GEPS"][0]
    np.savez_compressed(out, starts=np.array([f"{d:%Y-%m-%d}" for d in starts[ok]]),
                        r1=r1[ok].astype(np.float32), r2=r2[ok].astype(np.float32))  # fmt: skip
    print(f"{out.name}: {ok.sum()} starts {starts[ok][0]:%Y-%m-%d}..{starts[ok][-1]:%Y-%m-%d}")


def build_c3s_hc(src: Path) -> None:
    """data/c3s_hc_n34.npz from the C3S hindcast box files c3s_nino34.py caches (src = its data/hindcast): per system
    and start month, the (24, 6) ensemble-mean Niño-3.4 and relative Niño-3.4 of 1993-2016, leads 1-6 (lead 1 = the
    start month), as anomalies against the system's own 1993-2016 mean."""
    import c3s_nino34 as C

    arrs, systems = {}, []
    for centre, system, label, _ in C.MODELS:
        sid = f"{centre}_{system}"
        for mm in range(1, 13):
            p1, p2 = src / f"{sid}_{mm:02d}_n34.grib", src / f"{sid}_{mm:02d}.grib"
            if not (p1.exists() and p2.exists()):
                continue
            n34 = C._box_series(p1, C.N34_LAT, C.N34_LON)
            trop = C._box_series(p2, C.TROP_LAT, None)
            acc: dict = {}
            for (num, dd, dt, lead), v in n34.items():
                t = trop.get((num, dd, dt, lead), np.nan)
                y, m0 = int(str(dd)[:4]), int(str(dd)[4:6])
                y += int(mm == 1 and m0 == 12)  # a lagged member started in December belongs to the January start
                if np.isfinite(v) and np.isfinite(t) and 1993 <= y <= 2016 and 1 <= lead <= 6:
                    acc.setdefault((y, int(lead)), []).append((v, v - t))
            tab = np.full((2, 24, 6), np.nan)
            for (y, lead), vals in acc.items():
                tab[:, y - 1993, lead - 1] = np.mean(vals, 0)
            tab -= np.nanmean(tab, 1, keepdims=True)
            label_sid = "eccc" if centre == "eccc" else sid  # CanSIPS: its two components pooled below
            for j, k in enumerate(INDICES):
                arrs.setdefault(f"{label_sid}_{mm:02d}_{k}", []).append(tab[j])
            if label in EXCLUDE_SYSTEMS or label_sid in systems:
                continue
            systems.append(label_sid)
    keep = [s for s in systems if not any(s == f"{c}_{y}" for c, y, lab, _ in C.MODELS if lab in EXCLUDE_SYSTEMS)]
    out = {k: np.mean(v, 0).astype(np.float32) for k, v in arrs.items() if k.rsplit("_", 2)[0] in keep}
    np.savez_compressed(C3S_HC, systems=np.array(keep), **out)
    print(f"{C3S_HC.name}: {len(keep)} systems, {len(out) // 2} system-start tables")


# ------------------------------------------------------------------------------------------------- the products
class EnsoTable:
    """The forecast-ENSO component's input (mo_method EN): observed relative Niño-3.4 by month, the hindcast's
    forecasts by start month and lead, and for the year being forecast the issue's own forecast."""

    def __init__(self, path: Path | None = None, live: dict | None = None):
        with np.load(path or HC_TABLE) as z:
            self.obs = dict(zip(z["obs_months"].tolist(), z["obs_rel"].tolist(), strict=True))
            self.starts = {m: i for i, m in enumerate(z["months"].tolist())}
            self.F = z["rel"]
        self.live = live  # {"year": target year, "init": start month, "rel": {month: value}}

    @staticmethod
    def ym(y: int, m: int) -> str:
        return f"{y + (m - 1) // 12}-{(m - 1) % 12 + 1:02d}"

    def target(self, years: np.ndarray, month: int, span: int, lead: int) -> tuple[np.ndarray, np.ndarray]:
        """(observed, forecast) relative Niño-3.4 over the `span` months from `month`, per year; the forecast is the
        one from the month before the issue (an issue `lead` months before `month`, on the 15th)."""
        o, f = np.full(len(years), np.nan), np.full(len(years), np.nan)
        for k, y in enumerate(years):
            ms = [self.ym(int(y), month + d) for d in range(span)]
            o[k] = np.mean([self.obs.get(m, np.nan) for m in ms])
            start = self.ym(int(y), month - lead - 1)
            if self.live is not None and int(y) == self.live["year"]:
                f[k] = np.mean([self.live["rel"].get(m, np.nan) for m in ms])
            elif start in self.starts:
                f[k] = np.mean(self.F[self.starts[start], lead : lead + span])
        return o, f


def probabilities(mu: float, sd: float, scale: float = 1.0) -> dict:
    """Normal probabilities of the ENSO categories for a 3-month mean (thresholds on the index's own scale)."""
    from scipy.stats import norm

    p = {k: float(norm.cdf((hi - mu) / sd) - norm.cdf((lo - mu) / sd)) for k, _, lo, hi in CATS}
    for thr, lab in STRONG:
        p[lab.replace(" ", "_")] = float(norm.sf((thr - mu) / sd))
        p[f"{lab.replace(' ', '_')}_lanina"] = float(norm.cdf((-thr - mu) / sd))
    return {k: round(v, 3) for k, v in p.items()}


def issue(D: dict, F: dict, issued: str) -> dict:
    """The forecast from the newest start (D['months'][-1]) and its record for the page."""
    T = len(D["months"])
    init = D["months"][-1]
    m = Model(D, np.ones(T, bool))
    mjo = mjo_live(init, D["mjo_damp"])
    stat = m.forecast(T - 1, m.betas(D["mo"], int(D["mo"][-1])), np.array(mjo["rmm"]) if mjo else np.zeros(2))
    targets = [str(pd.Period(init, "M") + L) for L in range(1, LEADS + 1)]
    sc = roni_scale()
    tsc = np.array([sc[int(t[5:]) - 1] for t in targets])
    c3 = c3s_issue()
    ratio = c3s_hindcast_ratio(D, F)
    meta = {**D["meta"], "mjo": mjo and {**mjo, "window": f"{mjo_window(init):%Y-%m-%d}"}}
    out = {"issued": issued, "init": init, "targets": targets, "inputs": meta, "series": {}, "skill": {}}
    sm = int(D["mo"][-1]) - 1
    for k in INDICES:
        O = truth(D, k)
        cm = np.full(LEADS, np.nan)
        if c3 is not None:
            for j, v in enumerate(c3["months"]):
                if v in targets:
                    cm[targets.index(v)] = c3[k][j] + base_shift(D[f"obs_{k}"], int(v[5:]))
        claude = consensus(stat[k], cm)
        rk = ratio[k] if np.isfinite(cm).any() else np.ones(LEADS)  # the consensus's error where the models enter
        sd = error_sd(F[k], O, D["mo"], D["yr"])[sm] * rk
        sd3 = error_sd(F[k], O, D["mo"], D["yr"], span=3)[sm]
        r3 = np.array([rk[L : L + 3].mean() for L in range(LEADS - 2)])
        f = 1.0 if k == "n34" else tsc  # RONI: the relative index on Niño-3.4's scale, per calendar month
        rec = {
            "stat": np.round(stat[k] * f, 3).tolist(),
            "c3s": [None if not np.isfinite(x) else round(float(x), 3) for x in cm * f],
            "claude": np.round(claude * f, 3).tolist(),
            "sd": np.round(sd * f, 3).tolist(),
        }
        if c3 is not None:
            rec["systems"] = {}  # each system's member mean on the target months (None where it does not reach)
            for n, sy in c3["systems"].items():
                row = [None] * LEADS
                for j, v in enumerate(c3["months"]):
                    if v in targets:
                        x = (sy[k][j] + base_shift(D[f"obs_{k}"], int(v[5:]))) * (
                            1.0 if k == "n34" else sc[int(v[5:]) - 1]
                        )
                        row[targets.index(v)] = round(float(x), 3)
                rec["systems"][n] = row
        seasons = []
        for L in range(LEADS - 2):
            ms = targets[L : L + 3]
            centre = int(ms[1][5:])
            g = 1.0 if k == "n34" else sc[centre - 1]
            mu = float(np.mean(claude[L : L + 3])) * g
            s3 = float(sd3[L] * r3[L]) * g
            seasons.append({"months": ms, "label": "".join(calendar.month_abbr[int(x[5:])][0] for x in ms),
                            "mean": round(float(mu), 3), "sd": round(float(s3), 3), "p": probabilities(mu, s3)})  # fmt: skip
        rec["seasons"] = seasons
        hist = [str(pd.Period(init, "M") - j) for j in range(23, -1, -1)]
        rec["obs"] = {"months": hist, "values": [round(float(D[f"obs_{k}"][h]) * (1.0 if k == "n34" else sc[int(h[5:]) - 1]), 3) for h in hist]}  # fmt: skip
        out["series"]["n34" if k == "n34" else "roni"] = rec
        # hindcast skill by lead (all starts from SCORE_FROM; the consensus over 1993-2016 as the ratio)
        sel = D["yr"] >= SCORE_FROM
        r = [float(np.corrcoef(F[k][sel & np.isfinite(O[:, L]), L], O[sel & np.isfinite(O[:, L]), L])[0, 1])
             for L in range(LEADS)]  # fmt: skip
        e = [float(np.sqrt(np.nanmean((F[k][sel, L] - O[sel, L]) ** 2))) for L in range(LEADS)]
        out["skill"]["n34" if k == "n34" else "roni"] = {
            "r": np.round(r, 3).tolist(),
            "rmse": np.round(e, 3).tolist(),
            "ratio_claude": np.round(ratio[k], 3).tolist(),
            "years": [SCORE_FROM, int(D["yr"][-1])],
        }
    if c3 is not None:
        out["c3s"] = {"issue": c3["issue"], "systems": list(c3["systems"])}
    return out


def save_table(D: dict, F: dict) -> None:
    obs_m = sorted(D["obs_rel"])
    np.savez_compressed(
        HC_TABLE,
        months=np.array(D["months"]),
        rel=F["rel"].astype(np.float32),
        n34=F["n34"].astype(np.float32),
        obs_months=np.array(obs_m),
        obs_rel=np.array([D["obs_rel"][m] for m in obs_m], np.float32),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ersst", type=Path, required=False, help="ERSSTv6 sst.mnmean.nc (ERSST_URL)")
    ap.add_argument("--hindcast-only", action="store_true", help="write the hindcast table, no issue")
    ap.add_argument("--force", action="store_true", help="re-issue this month's forecast if it exists")
    ap.add_argument("--build-c3s-hc", type=Path, help="c3s_nino34.py's hindcast directory: write data/c3s_hc_n34.npz")
    ap.add_argument("--build-geps-mjo", type=Path, help="mjo_hindcast.py's mjo_hindcast.nc: write data/geps_mjo_hc.npz")
    ap.add_argument(
        "--build-gefs-mjo", type=Path, help="gefs_mjo_hindcast.npz (gefs-ref-v1): write data/gefs_mjo_hc.npz"
    )
    a = ap.parse_args()
    if a.build_c3s_hc:
        build_c3s_hc(a.build_c3s_hc)
        return 0
    if a.build_geps_mjo:
        build_geps_mjo_hc(a.build_geps_mjo)
        return 0
    if a.build_gefs_mjo:
        build_gefs_mjo_hc(a.build_gefs_mjo)
        return 0
    D = inputs(a.ersst)
    F = hindcast(D)
    save_table(D, F)
    print(f"hindcast {D['months'][0]}..{D['months'][-1]} -> {HC_TABLE.name}; inputs {D['meta']}", flush=True)
    now = datetime.now(UTC)
    tag = f"{now:%Y%m}"
    if a.hindcast_only:
        return 0
    if (OUT / f"enso_{tag}.json").exists() and not a.force:  # an issue stands as issued
        print(f"enso_{tag}.json was already issued; --force re-issues it")
        return 0
    res = issue(D, F, now.strftime("%Y-%m-%d"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"enso_{tag}.json").write_text(json.dumps(res, separators=(",", ":")))
    s = res["series"]["n34"]
    print(
        f"enso_{tag}.json: from {res['init']}: "
        + " ".join(f"{t[2:]} {v:+.2f}" for t, v in zip(res["targets"], s["claude"], strict=True))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
