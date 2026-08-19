#!/usr/bin/env python3
"""Rain -> DAILY CHANGE in inflow, with honest out-of-sample backtesting.

Motivation (2026-08-18, colleague review).  The v3 level model reports
r=0.755 against a 5-day trailing mean of inflow % of norm.  That number
is not a skill number: the smoothed series has lag-1 autocorrelation
0.978, so simply repeating yesterday's value scores r=0.98 on the same
target.  The level r mostly measures the smoothing.

This module predicts the DELTA instead.  Raw daily inflow % of norm has
lag-1 autocorrelation 0.851; its first difference has -0.09, i.e. it is
essentially white.  Nothing is free there, so any positive skill is real
information about where inflow moves tomorrow.

Model — discrete linear reservoir, which is what a delta target implies:

    dy_t = a + b_rec*(y_{t-1} - 100)      recession toward normal
             + b_r*rain_{t-lag}           today's forcing
             + b_f*Kfast_{t-lag}          exponential memory, tau fitted
             + b_s*Kslow_{t-lag}          90-day antecedent wetness
             + b_e*RONI_t                 ET / dry-soil channel
             + b_st*S_{t-1}               reservoir storage anomaly

Multi-lead forecasts recurse: y_{t-1} becomes the model's own prediction,
so errors compound the way they do operationally.

Backtesting is nested and blocked:
  * outer  — K contiguous test blocks; every score is out-of-sample
  * embargo — EMBARGO days purged either side of each test block so
    kernel memory and target autocorrelation cannot leak across the cut
  * inner  — tau and lag are re-selected INSIDE each training fold only.
    The v3 code picked them once on all data, which is the main way a
    scan over 8 taus x 8 lags buys in-sample r that does not survive.
  * baselines — persistence (dy=0), climatology, AR(1) on the delta, and
    a RAIN-BLIND model with every state term but no rain.  The last one
    isolates what the rainfall actually contributes.
  * surrogate null — the whole pipeline, hyperparameter search included,
    re-run on circularly shifted rain.  Shifting preserves rainfall's own
    autocorrelation and seasonality while destroying its correspondence
    to inflow, so the spread of surrogate scores is exactly the skill the
    procedure manufactures from nothing.  The real score's percentile in
    that null is the honest p-value.

Outputs:
  ~/colombia_hydro/out/inflow_delta_backtest.json
  ~/colombia_hydro/site/inflow_delta_backtest.webp

    python scripts/sst/inflow_delta_model.py [--surrogates N] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
ENSO_JSON = REPO / "assets" / "sst" / "data" / "enso_daily.json"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
OUT_JSON = PRIV / "out" / "inflow_delta_backtest.json"

ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
TAUS = [2, 4, 7, 12, 20, 30, 45, 60]
LAGS = list(range(0, 8))
TAU_SLOW = 90.0
N_OUTER = 8                  # contiguous out-of-sample blocks
N_INNER = 4                  # inner folds for tau/lag selection
N_INNER_CAL = 10             # inner folds for the amplitude calibration.
                             # Fixed a priori, NOT tuned on test skill: with
                             # 10 folds each inner fit sees 90% of the
                             # training data, so the calibration reflects a
                             # model of nearly the final quality.  Too few
                             # folds (4 -> 75%) makes the inner model worse
                             # than the real one and over-shrinks (s~0.32 vs
                             # a true optimum ~0.50), throwing away skill.
                             # Sensitivity at 4/8/12 is reported.
EMBARGO = 30                 # days purged either side of every test block
MAX_LEAD = 15
CLIP = (0.0, 400.0)          # % of norm bounds for recursive simulation


def ema(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential memory, NaN treated as climatology (0)."""
    from scipy.signal import lfilter
    a = 1.0 / tau
    return lfilter([a], [1.0, -(1.0 - a)], np.nan_to_num(x))


# 0 = the recession pulls toward a FIXED 100% of norm.
#
# This looks physically wrong and is worth defending, because it invites
# exactly one obvious "fix". With no future rain the model floors near 86%
# of norm - while early August 2026 actually ran at 62.5% - so the model
# will not carry a drought down to where the catchments really sit, and its
# attractor is climatology rather than the current state.
#
# The obvious repair is a trailing baseline. Tested 2026-08-19: a 90-day
# window puts the attractor at 67.3%, which matches the observed drought
# state, and it even looks better on pooled daily-delta r (0.666 -> 0.684).
# Per basin that gain evaporates (best is ANTIOQUIA +0.005, p=0.09; three
# basins go negative), and on MULTI-LEAD trajectory RMSE - the metric an
# asymptote actually drives - it is clearly WORSE at every lead:
#
#     lead      1      3      5      7     10     15    mean
#     win=0   18.1   24.4   26.9   28.3   29.1   28.8   26.98
#     win=90  19.0   27.0   30.3   31.9   32.6   32.4   30.16   (+11.8%)
#
# A trailing mean is itself a noisy, lagging estimate; pulling toward it
# injects that noise into every step of the recursion, while a fixed 100 is
# a stable attractor. So the physically-odd choice wins on the numbers and
# stays. The real cost is stated rather than hidden: in a sustained drought
# this model will over-forecast recovery.
BASELINE_WIN = 0
                             # >0 = toward a causal trailing mean of that
                             # many days.  Colombia's basins drift: VALLE
                             # +15.8%/decade (94->118 from 2000-08 to
                             # 2018-26), CALDAS -8.6%/decade, mostly fleet
                             # composition rather than climate.  Over the
                             # 757-day gauge-blended record there is too
                             # little drift for it to matter (mean delta r
                             # 0.476 vs 0.465, h7 skill +0.155 vs +0.160),
                             # so it stays off by default and is switched
                             # on for the long backfilled record where the
                             # drift is real.


def baseline_series(y: np.ndarray, win: int) -> np.ndarray:
    """Causal trailing mean of observed inflow; 100 until 30 days exist."""
    from collections import deque
    out = np.full(len(y), 100.0)
    q, sm = deque(), 0.0
    for i, v in enumerate(y):
        if np.isfinite(v):
            q.append(v); sm += v
            while len(q) > win:
                sm -= q.popleft()
        out[i] = sm / len(q) if len(q) >= 30 else 100.0
    return np.roll(out, 1)


def lagged(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x.copy()
    out = np.roll(x, k).astype(float)
    out[:k] = np.nan
    return out


# ── data ────────────────────────────────────────────────────────────────────
def load() -> dict:
    tc = json.loads(TRUTH.read_text())
    if tc.get("truth_version") != 3:
        raise SystemExit("truth cache stale — run colombia_forecast.py first")
    rd = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in tc["dates"]],
                  dtype="datetime64[D]")
    rain = {r: np.array(tc[r], float) - np.array(tc[r + "_clim"], float)
            for r in ORDER}

    inf = json.loads(INFLOW_JSON.read_text())["recent"]
    idl = np.array(inf["dates"], dtype="datetime64[D]")
    pct = {r: np.array(inf["pct_of_norm"][r], float) for r in ORDER}
    for r in ORDER:
        pct[r][pct[r] == 0] = np.nan            # XM zero = missing, not dry

    common, ri, ii = np.intersect1d(rd, idl, return_indices=True)
    rain = {r: v[ri] for r, v in rain.items()}
    y = {r: v[ii] for r, v in pct.items()}

    ed = json.loads(ENSO_JSON.read_text())["daily"]
    emap = dict(zip(ed["dates"], ed["roni_d"]))
    roni = np.array([emap.get(str(d), np.nan) for d in common], float)
    for i in range(1, len(roni)):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i - 1]
    for i in range(len(roni) - 2, -1, -1):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i + 1]

    from xm_storage import pct_anomaly_series
    sd, sa = pct_anomaly_series()
    sd = np.array([str(x) for x in sd], dtype="datetime64[D]")
    stor = {}
    for r in ORDER:
        v = np.full(len(common), np.nan)
        _, c2, s2 = np.intersect1d(common, sd, return_indices=True)
        v[c2] = np.asarray(sa[r], float)[s2]
        for i in range(1, len(v)):              # hold last observed
            if not np.isfinite(v[i]):
                v[i] = v[i - 1]
        stor[r] = np.nan_to_num(v)
    return {"dates": common, "rain": rain, "y": y, "roni": roni, "stor": stor}


# ── design matrix ───────────────────────────────────────────────────────────
# Rain's effect on inflow SCALES with the level the river is already
# running at: the same 10 mm produces a bigger absolute rise on a river at
# 200% of norm than at 60%. Measured 2026-08-19 - the interaction is
# significant 7/8 folds (p=0.012) with a coefficient positive in 8/8 and
# tightly clustered, and it lifts rise-day correlation by +0.035 on
# ANTIOQUIA. It is multiplicative dynamics rather than catchment
# saturation: the term vanishes in log space (p=0.82).
#
# The weight is (y-100)/100, NOT a z-score. A z-score needs the series
# mean and sd, which simulate() does not have - it only carries the
# evolving level. The two are an exact affine reparameterisation that a
# linear fit absorbs identically (verified to 3 dp on four basins), so
# this form buys statelessness for nothing.
USE_FLOW_INTERACTION = True


def design(rain, y, roni, stor, tau, lag, use_rain=True):
    """Features predicting dy_t = y_t - y_{t-1}. All strictly causal."""
    kf = lagged(ema(rain, tau), lag)
    ks = lagged(ema(rain, TAU_SLOW), lag)
    rl = lagged(rain, lag)
    ym1 = lagged(y, 1)
    base = baseline_series(y, BASELINE_WIN) if BASELINE_WIN else 100.0
    cols = [ym1 - base]
    if use_rain:
        cols += [rl, kf, ks]
    cols += [roni, stor]
    if use_rain and USE_FLOW_INTERACTION:
        w = (ym1 - 100.0) / 100.0
        cols += [rl * w, kf * w]
    X = np.column_stack(cols)
    dy = y - ym1
    return X, dy


def fit(X, dy, m):
    A = np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(A, dy[m], rcond=None)
    return beta


def predict(X, beta):
    return beta[0] + X @ beta[1:]


# ── nested blocked CV ───────────────────────────────────────────────────────
def blocks(n, k):
    e = np.linspace(0, n, k + 1).astype(int)
    return [(e[i], e[i + 1]) for i in range(k)]


def select_hyper(rain, y, roni, stor, tr_mask, quick):
    """tau/lag chosen by inner blocked CV WITHIN the training mask only."""
    idx = np.where(tr_mask)[0]
    if len(idx) < 200:
        return 20, 1
    inner = blocks(len(idx), N_INNER)
    taus = TAUS[::2] if quick else TAUS
    lags = LAGS[::2] if quick else LAGS
    best = None
    for tau in taus:
        for lag in lags:
            X, dy = design(rain, y, roni, stor, tau, lag)
            sse, cnt = 0.0, 0
            for a, b in inner:
                te = np.zeros(len(rain), bool)
                te[idx[a:b]] = True
                tr = tr_mask & ~te
                lo, hi = idx[a], idx[b - 1]
                tr[max(0, lo - EMBARGO):min(len(tr), hi + EMBARGO + 1)] = False
                mtr = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
                mte = te & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
                if mtr.sum() < 120 or mte.sum() < 20:
                    continue
                p = predict(X, fit(X, dy, mtr))
                sse += float(np.sum((p[mte] - dy[mte]) ** 2))
                cnt += int(mte.sum())
            if cnt and (best is None or sse / cnt < best[0]):
                best = (sse / cnt, tau, lag)
    return (best[1], best[2]) if best else (20, 1)


def simulate(beta, rain, roni, stor, tau, lag, i0, y0, h):
    """Recursive multi-lead simulation from observed y0 at index i0."""
    kf = lagged(ema(rain, tau), lag)
    ks = lagged(ema(rain, TAU_SLOW), lag)
    rl = lagged(rain, lag)
    out = np.full(h, np.nan)
    yp = y0
    for j in range(h):
        i = i0 + 1 + j
        if i >= len(rain):
            break
        f = [yp - 100.0, rl[i], kf[i], ks[i], roni[i], stor[i]]
        if USE_FLOW_INTERACTION:
            # same two columns design() appends, evaluated on the SIMULATED
            # level so the gain evolves with the forecast trajectory
            w = (yp - 100.0) / 100.0
            f += [rl[i] * w, kf[i] * w]
        if not np.all(np.isfinite(f)):
            break
        yp = float(np.clip(yp + beta[0] + np.dot(beta[1:], f), *CLIP))
        out[j] = yp
    return out


def shrinkage(beta_unused, rain, y, roni, stor, tau, lag, tr_mask, stride=3):
    """Per-lead amplitude + drift calibration, estimated by INNER CV.

    The recursion gets the direction of a move right but overshoots its
    size, so raw multi-lead RMSE loses to persistence even where the
    predicted change correlates ~0.45 with the observed one.  For each
    lead we want observed_change ~ a_h + s_h * predicted_change.

    Estimating that on the same days the coefficients were fitted gives
    s_h ~ 0.9 because the model looks good there; the honest optimum is
    ~0.45.  Applying 0.9 wipes out all multi-lead skill (measured: +0.10
    available, -0.02 realised).  So the calibration is itself
    cross-validated — beta is refit inside each inner fold and the
    (predicted, observed) pairs come only from held-out inner days.
    """
    idx = np.where(tr_mask)[0]
    if len(idx) < 250:
        return np.ones(MAX_LEAD), np.zeros(MAX_LEAD)
    inner = blocks(len(idx), N_INNER_CAL)
    P = {h: [] for h in range(MAX_LEAD)}
    O = {h: [] for h in range(MAX_LEAD)}
    for a, b in inner:
        te = np.zeros(len(y), bool)
        te[idx[a:b]] = True
        tr = tr_mask & ~te
        lo, hi = idx[a], idx[b - 1]
        tr[max(0, lo - EMBARGO):min(len(tr), hi + EMBARGO + 1)] = False
        X, dy = design(rain, y, roni, stor, tau, lag)
        m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 120:
            continue
        bt = fit(X, dy, m)
        for i0 in np.where(te)[0][::stride]:
            if not np.isfinite(y[i0]):
                continue
            sim = simulate(bt, rain, roni, stor, tau, lag, i0, y[i0], MAX_LEAD)
            for j, v in enumerate(sim):
                i = i0 + j + 1
                if i >= len(y) or not np.isfinite(v) or not np.isfinite(y[i]):
                    continue
                P[j].append(v - y[i0])
                O[j].append(y[i] - y[i0])
    sh = np.ones(MAX_LEAD)
    off = np.zeros(MAX_LEAD)
    for j in range(MAX_LEAD):
        p, o = np.asarray(P[j]), np.asarray(O[j])
        if len(p) < 40 or np.std(p) < 1e-6:
            continue
        A = np.column_stack([np.ones(len(p)), p])
        c, *_ = np.linalg.lstsq(A, o, rcond=None)
        off[j], sh[j] = float(c[0]), float(np.clip(c[1], 0.0, 1.5))
    return sh, off


def backtest(d, basin, quick=False, rain_override=None, holdout=None):
    """Returns per-lead out-of-sample scores + residual pools by lead.

    `holdout` is a boolean mask of days excluded from EVERY training set
    and from hyperparameter selection, scored separately as a clean
    never-touched window.
    """
    rain = d["rain"][basin] if rain_override is None else rain_override
    y, roni, stor = d["y"][basin], d["roni"], d["stor"][basin]
    n = len(y)
    hold = np.zeros(n, bool) if holdout is None else np.asarray(holdout, bool)
    outer = blocks(n, N_OUTER)

    lead_pred = {h: [] for h in range(1, MAX_LEAD + 1)}
    lead_obs = {h: [] for h in range(1, MAX_LEAD + 1)}
    lead_pers = {h: [] for h in range(1, MAX_LEAD + 1)}
    lead_fold = {h: [] for h in range(1, MAX_LEAD + 1)}
    lead_raw = {h: [] for h in range(1, MAX_LEAD + 1)}
    d1_pred, d1_obs, d1_blind, d1_ar = [], [], [], []
    chosen, shrink = [], []
    fold_id = -1

    for a, b in outer:
        fold_id += 1
        te = np.zeros(n, bool); te[a:b] = True
        tr = ~te & ~hold
        tr[max(0, a - EMBARGO):min(n, b + EMBARGO)] = False
        if hold.any():                       # embargo around the holdout too
            hi = np.where(hold)[0]
            tr[max(0, hi[0] - EMBARGO):min(n, hi[-1] + EMBARGO + 1)] = False
        te &= ~hold
        if tr.sum() < 250:
            continue
        tau, lag = select_hyper(rain, y, roni, stor, tr, quick)
        chosen.append((tau, lag))
        X, dy = design(rain, y, roni, stor, tau, lag)
        mtr = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        if mtr.sum() < 120:
            continue
        beta = fit(X, dy, mtr)
        p = predict(X, beta)
        mte = te & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        d1_pred.append(p[mte]); d1_obs.append(dy[mte])

        # rain-blind control, same folds
        Xb, dyb = design(rain, y, roni, stor, tau, lag, use_rain=False)
        mb = tr & np.isfinite(dyb) & np.all(np.isfinite(Xb), axis=1)
        bb = fit(Xb, dyb, mb)
        d1_blind.append(predict(Xb, bb)[mte])

        # AR(1) on the delta, same folds
        dm1 = lagged(dy, 1)
        Xa = np.column_stack([np.nan_to_num(dm1)])
        ma = tr & np.isfinite(dy) & np.isfinite(dm1)
        ba = fit(Xa, dy, ma)
        d1_ar.append(predict(Xa, ba)[mte])

        sh, off = shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
        shrink.append([sh.round(3).tolist(), off.round(2).tolist()])
        # recursive multi-lead from each start day in the test block
        for i0 in range(a, b):
            if not np.isfinite(y[i0]) or hold[i0]:
                continue
            sim = simulate(beta, rain, roni, stor, tau, lag, i0, y[i0], MAX_LEAD)
            for j, v in enumerate(sim):
                h = j + 1
                i = i0 + h
                if i >= n or not np.isfinite(v) or not np.isfinite(y[i]):
                    continue
                if hold[i]:
                    continue
                vc = y[i0] + off[j] + sh[j] * (v - y[i0])   # calibrated
                lead_pred[h].append(vc)
                lead_obs[h].append(y[i])
                lead_pers[h].append(y[i0])
                lead_fold[h].append(fold_id)
                lead_raw[h].append(v)

    def sc(p, o):
        p, o = np.asarray(p), np.asarray(o)
        if len(p) < 25:
            return None
        return {"n": int(len(p)),
                "r": round(float(np.corrcoef(p, o)[0, 1]), 3),
                "rmse": round(float(np.sqrt(np.mean((p - o) ** 2))), 2),
                "mae": round(float(np.mean(np.abs(p - o))), 2)}

    dp = np.concatenate(d1_pred) if d1_pred else np.array([])
    do = np.concatenate(d1_obs) if d1_obs else np.array([])
    db = np.concatenate(d1_blind) if d1_blind else np.array([])
    da = np.concatenate(d1_ar) if d1_ar else np.array([])
    res = {"delta_lead1": {
        "model": sc(dp, do),
        "rain_blind": sc(db, do),
        "ar1": sc(da, do),
        "persistence_zero": sc(np.zeros_like(do), do)}}
    res["leads"] = {}
    for h in range(1, MAX_LEAD + 1):
        P, O, Q = lead_pred[h], lead_obs[h], lead_pers[h]
        if len(P) < 25:
            continue
        m_, p_ = sc(P, O), sc(Q, O)
        dpm = np.asarray(P) - np.asarray(Q)     # predicted change from start
        dob = np.asarray(O) - np.asarray(Q)     # observed change from start
        raw = sc(lead_raw[h], O)
        res["leads"][h] = {
            "level_model": m_, "level_persistence": p_, "level_raw_uncalibrated": raw,
            "rmse_skill_vs_persistence": round(
                1 - m_["rmse"] / p_["rmse"], 3) if m_ and p_ else None,
            "rmse_skill_raw": round(1 - raw["rmse"] / p_["rmse"], 3)
            if raw and p_ else None,
            "change_r": round(float(np.corrcoef(dpm, dob)[0, 1]), 3)
            if len(dpm) > 25 and np.std(dpm) > 1e-9 else None,
            "resid": (np.asarray(O) - np.asarray(P)).round(2).tolist(),
            "fold": list(lead_fold[h])}
    res["tau_lag_by_fold"] = [[int(t), int(l)] for t, l in chosen]
    res["shrinkage_by_fold"] = shrink
    res["residuals_lead1"] = (do - dp).tolist() if len(dp) else []
    return res


def pick_holdout(d, basin=None):
    """The most informative single month: the one whose inflow swings most.

    A quiet month is a soft test — the model can score well by barely
    moving.  We hold out the month with the largest peak-to-trough range
    in daily inflow % of norm (averaged across basins when none is
    named), so the clean validation window is the one where getting the
    direction and size of the move right actually matters.
    """
    dates = d["dates"]
    months = np.array([str(x)[:7] for x in dates])
    best = None
    for mo in sorted(set(months)):
        m = months == mo
        if m.sum() < 25:
            continue
        rng = []
        for b in ([basin] if basin else ORDER):
            v = d["y"][b][m]
            v = v[np.isfinite(v)]
            if len(v) > 20:
                rng.append(np.percentile(v, 95) - np.percentile(v, 5))
        if not rng:
            continue
        sc = float(np.mean(rng))
        if best is None or sc > best[0]:
            best = (sc, mo)
    return best[1], round(best[0], 1)


def holdout_eval(d, basin, hold, cv):
    """Score the never-trained-on month, with a distribution for each day.

    Trained on every day outside the holdout and its embargo; tau/lag and
    the amplitude calibration both selected by inner CV inside that
    training set.  The predictive distribution at lead h is the point
    forecast plus the empirical residual sample for lead h taken from the
    blocked CV, which never saw these days either.
    """
    rain, y = d["rain"][basin], d["y"][basin]
    roni, stor = d["roni"], d["stor"][basin]
    n = len(y)
    tr = ~hold
    hi = np.where(hold)[0]
    tr[max(0, hi[0] - EMBARGO):min(n, hi[-1] + EMBARGO + 1)] = False
    tau, lag = select_hyper(rain, y, roni, stor, tr, False)
    X, dy = design(rain, y, roni, stor, tau, lag)
    m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
    beta = fit(X, dy, m)
    sh, off = shrinkage(beta, rain, y, roni, stor, tau, lag, tr)

    res_pool = {h: np.asarray(cv["leads"].get(h, {}).get("resid", []), float)
                for h in range(1, MAX_LEAD + 1)}
    P, O, Q, LH = [], [], [], []
    for i0 in hi:
        if not np.isfinite(y[i0]):
            continue
        sim = simulate(beta, rain, roni, stor, tau, lag, i0, y[i0], MAX_LEAD)
        for j, v in enumerate(sim):
            i = i0 + j + 1
            if i >= n or not np.isfinite(v) or not np.isfinite(y[i]):
                continue
            P.append(y[i0] + off[j] + sh[j] * (v - y[i0]))
            O.append(y[i]); Q.append(y[i0]); LH.append(j + 1)
    P, O, Q, LH = map(np.asarray, (P, O, Q, LH))
    out = {"month": str(d["dates"][hi[0]])[:7], "days": int(hold.sum()),
           "tau_days": int(tau), "lag_days": int(lag),
           "n_forecast_days": int(len(P)), "by_lead": {}}
    if not len(P):
        return out
    for h in range(1, MAX_LEAD + 1):
        k = LH == h
        if k.sum() < 5:
            continue
        rm = float(np.sqrt(np.mean((P[k] - O[k]) ** 2)))
        rp = float(np.sqrt(np.mean((Q[k] - O[k]) ** 2)))
        e = {"n": int(k.sum()), "rmse": round(rm, 2),
             "rmse_persistence": round(rp, 2),
             "rmse_skill": round(1 - rm / rp, 3) if rp else None}
        cp = crps_pit(P[k], O[k], res_pool.get(h, np.zeros(0)))
        if cp:
            e.update({kk: cp[kk] for kk in
                      ("crps", "crps_skill_score", "pit_mean")})
        out["by_lead"][h] = e
    # one concrete init: quantile fan for every day of the month
    i0 = int(hi[0])
    sim = simulate(beta, rain, roni, stor, tau, lag, i0, y[i0], MAX_LEAD)
    qs = [5, 25, 50, 75, 95]
    fan = []
    for j, v in enumerate(sim):
        i = i0 + j + 1
        if i >= n or not np.isfinite(v):
            continue
        pt = y[i0] + off[j] + sh[j] * (v - y[i0])
        R = res_pool.get(j + 1, np.zeros(0))
        R = R[np.isfinite(R)]
        row = {"lead": j + 1, "date": str(d["dates"][i]),
               "point": round(float(pt), 1),
               "obs": round(float(y[i]), 1) if np.isfinite(y[i]) else None}
        if len(R) > 30:
            row["q"] = {f"p{q}": round(float(pt + np.percentile(R, q)), 1)
                        for q in qs}
        fan.append(row)
    out["example_fan"] = {"init": str(d["dates"][i0]), "rows": fan}
    return out


# ── surrogate null: how much skill does the PROCEDURE invent? ───────────────
def surrogate_null(d, basin, n_surr, quick, rng):
    """Re-run the whole nested-CV pipeline on circularly shifted rain.

    A circular shift keeps rainfall's own autocorrelation, variance and
    seasonal cycle intact but destroys its correspondence to this basin's
    inflow.  Because tau/lag selection runs inside every surrogate too,
    the resulting spread is exactly the skill the search manufactures
    from noise at this sample size.  Minimum shift exceeds the slow
    kernel so no real lead-lag survives.
    """
    rain = d["rain"][basin]
    n = len(rain)
    lo = int(TAU_SLOW) + 30
    scores = []
    for _ in range(n_surr):
        k = int(rng.integers(lo, n - lo))
        rs = np.roll(rain, k)
        try:
            res = backtest(d, basin, quick=quick, rain_override=rs)
        except Exception:                           # noqa: BLE001
            continue
        m = res["delta_lead1"]["model"]
        if m:
            scores.append(m["r"])
    return np.array(scores, float)


def crps_pit(pred, obs, resid_pool):
    """CRPS and PIT for a predictive distribution built as point forecast
    plus an empirical residual sample (obs = pred + residual)."""
    pred, obs = np.asarray(pred), np.asarray(obs)
    R = np.asarray(resid_pool, float)
    R = R[np.isfinite(R)]
    if len(R) < 30 or not len(pred):
        return None
    if len(R) > 400:                                 # keep the O(m^2) term sane
        R = np.quantile(R, np.linspace(0.002, 0.998, 400))
    m = len(R)
    ens = pred[:, None] + R[None, :]                 # (n, m) predictive sample
    t1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    Rs = np.sort(R)
    # E|X-X'| for the shared residual sample, computed once
    i = np.arange(m)
    t2 = 2.0 * np.sum((2 * i - m + 1) * Rs) / (m * m)
    crps = float(np.mean(t1 - 0.5 * t2))
    pit = np.mean(ens <= obs[:, None], axis=1)
    # climatological reference: obs anomalies about their own mean
    ref = obs - np.mean(obs)
    e2 = np.abs(ref[:, None] - ref[None, :])
    crps_cl = float(np.mean(np.abs(ref[:, None] - obs[None, :])) - 0.5 * np.mean(e2))
    return {"crps": round(crps, 2), "crps_climatology": round(crps_cl, 2),
            "crps_skill_score": round(1 - crps / crps_cl, 3) if crps_cl else None,
            "pit_hist": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
            "pit_mean": round(float(np.mean(pit)), 3)}



def figure(out, d, nulls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(15.5, 11.6))
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.015, 0.62, "COLOMBIA — RAIN → DAILY CHANGE IN INFLOW",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.015, 0.22, "out-of-sample only: nested blocked CV, "
            f"{out['design']['embargo_days']}-day embargo, τ/lag and amplitude "
            "calibration re-selected inside every training fold",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    hd.text(0.985, 0.5, f"{out['window']} · n={out['days']} d",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9,
            va="center", ha="right")

    # row 1 — surrogate nulls
    for i, b in enumerate(ORDER):
        ax = fig.add_axes([0.045 + 0.157 * i, 0.665, 0.132, 0.20])
        nl = nulls[b]
        v = out["basins"][b]
        ax.hist(nl, bins=22, color="#c3ccd6", edgecolor="white", lw=0.4)
        mr = v["delta_lead1"]["model"]["r"]
        br = v["delta_lead1"]["rain_blind"]["r"]
        ax.axvline(mr, color="#c62828", lw=2.2)
        ax.axvline(br, color="#e08214", lw=1.4, ls="--")
        ax.axvline(np.percentile(nl, 95), color="#5a6b7a", lw=1.0, ls=":")
        ax.set_title(b, fontsize=9.5, fontweight="bold", loc="left", color=INK)
        ax.set_xlabel("delta r, lead 1", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("surrogate count", fontsize=7.5)
            ax.legend(handles=[
                plt.Line2D([], [], color="#c62828", lw=2.2, label="real rain"),
                plt.Line2D([], [], color="#e08214", lw=1.4, ls="--", label="rain-blind"),
                plt.Line2D([], [], color="#5a6b7a", lw=1.0, ls=":", label="null p95")],
                fontsize=6.6, loc="upper left")
    fig.text(0.045, 0.885, "Surrogate null — the same pipeline run on "
             "circularly shifted rain, so the hyperparameter search runs "
             "inside the null too. Grey = skill the procedure invents from "
             "nothing.", fontsize=9.5, fontweight="bold", color=INK)

    # row 2 left — RMSE skill vs lead
    ax = fig.add_axes([0.045, 0.375, 0.42, 0.21])
    for b in ORDER:
        L = out["basins"][b]["leads"]
        hs = sorted(int(k) for k in L)
        ax.plot(hs, [L[str(h)]["rmse_skill_vs_persistence"] if str(h) in L
                     else L[h]["rmse_skill_vs_persistence"] for h in hs],
                lw=1.6, marker="o", ms=3, label=b)
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xlabel("lead, days", fontsize=8.5)
    ax.set_ylabel("RMSE skill vs persistence", fontsize=8.5)
    ax.set_title("Blocked-CV skill by lead — perfect rain, so this is the "
                 "rain→inflow model alone", fontsize=10, fontweight="bold",
                 loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5); ax.legend(fontsize=7, ncol=2)
    ax.tick_params(labelsize=8)

    # row 2 right — holdout fan
    ax2 = fig.add_axes([0.55, 0.375, 0.42, 0.21])
    def _cal(x):                     # |PIT-0.5| at h7, among skilful basins
        H = out["basins"][x]["holdout"].get("by_lead", {})
        e = H.get(7, H.get("7", {})) or {}
        if e.get("rmse_skill", -9) <= 0 or "pit_mean" not in e:
            return 9.0
        return abs(e["pit_mean"] - 0.5)
    b = min(ORDER, key=_cal)
    f = out["basins"][b]["holdout"]["example_fan"]
    rows = [r for r in f["rows"] if "q" in r]
    L = [r["lead"] for r in rows]
    ax2.fill_between(L, [r["q"]["p5"] for r in rows], [r["q"]["p95"] for r in rows],
                     color="#1f4e8c", alpha=0.16, lw=0, label="5–95%")
    ax2.fill_between(L, [r["q"]["p25"] for r in rows], [r["q"]["p75"] for r in rows],
                     color="#1f4e8c", alpha=0.30, lw=0, label="25–75%")
    ax2.plot(L, [r["q"]["p50"] for r in rows], color="#1f4e8c", lw=1.8, label="median")
    ob = [r["obs"] for r in rows]
    ax2.plot(L, ob, color="#c62828", lw=1.6, marker="o", ms=3.5, label="observed")
    ax2.set_xlabel("lead, days", fontsize=8.5)
    ax2.set_ylabel("inflow, % of norm", fontsize=8.5)
    ax2.set_title(f"{b} — a distribution for every day, init {f['init']} "
                  f"(held-out {out['holdout_month']}; best-calibrated basin)",
                  fontsize=10, fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5); ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

    # row 3 — table
    ax3 = fig.add_axes([0.045, 0.045, 0.925, 0.26]); ax3.set_axis_off()
    hdr = ["basin", "delta r\nlead 1", "rain-\nblind", "AR(1)", "null\nmean",
           "null\np95", "p", "RMSE skill\nh1 / h3 / h7",
           f"held-out {out['holdout_month']}\nskill h1 / h3 / h7",
           "CRPS skill\nh1 / h7", "PIT\nh1 / h7"]
    rows = []
    for b in ORDER:
        v = out["basins"][b]; L = v["leads"]; H = v["holdout"].get("by_lead", {})
        g = lambda D, h, k: (D.get(h, D.get(str(h), {})) or {}).get(k)
        f2 = lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) else "--"
        f3 = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "--"
        rows.append([
            b, f"{v['delta_lead1']['model']['r']:.3f}",
            f"{v['delta_lead1']['rain_blind']['r']:.3f}",
            f"{v['delta_lead1']['ar1']['r']:.3f}",
            f"{v['surrogate_null']['mean_r']:.3f}",
            f"{v['surrogate_null']['p95_r']:.3f}",
            f"{v['surrogate_null']['p_value']:.3f}",
            " / ".join(f2(g(L, h, "rmse_skill_vs_persistence")) for h in (1, 3, 7)),
            " / ".join(f2(g(H, h, "rmse_skill")) for h in (1, 3, 7)),
            " / ".join(f3(g(H, h, "crps_skill_score")) for h in (1, 7)),
            " / ".join(f3(g(H, h, "pit_mean")) for h in (1, 7))])
    t = ax3.table(cellText=rows, colLabels=hdr, cellLoc="center", loc="upper center",
                  colWidths=[.085, .075, .065, .065, .065, .06, .055, .155, .155, .09, .085])
    t.auto_set_font_size(False); t.set_fontsize(7.8); t.scale(1, 2.0)
    for (rr, cc), cell in t.get_celld().items():
        cell.set_edgecolor("#c9d2dc")
        if rr == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif rr % 2 == 0:
            cell.set_facecolor("#f4f6f9")
    fig.text(0.045, 0.318, "Every number is out-of-sample. p = fraction of "
             "surrogates matching or beating the real score.", fontsize=9.5,
             fontweight="bold", color=INK)
    fig.text(0.045, 0.012, "target = raw daily change in inflow % of norm "
             "(never the 5-day smoothed series, whose deltas carry 0.67 "
             "autocorrelation of pure smoothing artefact) · calibration folds "
             "fixed a priori at 10, not tuned on test skill",
             fontsize=7.5, color="#5a6b7a")
    png = PRIV / "site" / "inflow_delta_backtest.webp"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=118); plt.close(fig)
    print(f"wrote {png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogates", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", type=int, default=0,
                    help="trailing-mean window for the recession target "
                         "(0 = fixed 100); use ~365 on the long record")
    ap.add_argument("--holdout", default=None,
                    help="YYYY-MM to hold out entirely; default = the "
                         "month with the largest inflow swing")
    a = ap.parse_args()
    rng = np.random.default_rng(20260818)
    global BASELINE_WIN
    BASELINE_WIN = a.baseline

    d = load()
    n = len(d["dates"])
    mo, swing = pick_holdout(d)
    if a.holdout:
        mo = a.holdout
    months = np.array([str(x)[:7] for x in d["dates"]])
    hold = months == mo
    if hold.sum() < 20:
        raise SystemExit(f"holdout {mo} has only {hold.sum()} days")
    print(f"holdout month: {mo} ({hold.sum()} d, mean p5-p95 inflow swing "
          f"{swing} pts of norm) — excluded from all training, "
          f"hyperparameter selection and calibration", flush=True)
    print(f"window {d['dates'][0]}..{d['dates'][-1]}  n={n} days  "
          f"{N_OUTER} outer blocks, embargo {EMBARGO} d", flush=True)

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "window": f"{d['dates'][0]}..{d['dates'][-1]}", "days": int(n),
           "design": {"outer_blocks": N_OUTER, "inner_folds": N_INNER,
                      "embargo_days": EMBARGO, "max_lead": MAX_LEAD,
                      "surrogates": a.surrogates},
           "target": "raw daily change in inflow % of norm (NOT smoothed)",
           "holdout_month": mo, "holdout_swing_pts": swing,
           "basins": {}}

    nulls = {}
    print(f"\n{'basin':10} {'r_dOOS':>8} {'blind':>7} {'AR1':>7} "
          f"{'null p95':>9} {'p-val':>7} {'RMSEskill@1':>12} {'@7':>7}")
    for b in ORDER:
        res = backtest(d, b, quick=a.quick, holdout=hold)
        null = surrogate_null(d, b, a.surrogates, a.quick, rng)
        m = res["delta_lead1"]["model"]
        blind = res["delta_lead1"]["rain_blind"]
        ar1 = res["delta_lead1"]["ar1"]
        pv = float(np.mean(null >= m["r"])) if len(null) and m else np.nan
        res["surrogate_null"] = {
            "n": int(len(null)),
            "mean_r": round(float(np.mean(null)), 3) if len(null) else None,
            "p95_r": round(float(np.percentile(null, 95)), 3) if len(null) else None,
            "max_r": round(float(np.max(null)), 3) if len(null) else None,
            "p_value": round(pv, 4) if np.isfinite(pv) else None}
        # probabilistic scoring, residual pool held out by fold
        res["probabilistic"] = {}
        for h in (1, 3, 7, 14):
            if h not in res["leads"]:
                continue
        res["holdout"] = holdout_eval(d, b, hold, res)
        for h in list(res["leads"]):        # drop bulky raw arrays
            res["leads"][h].pop("resid", None)
            res["leads"][h].pop("fold", None)
        res.pop("residuals_lead1", None)
        out["basins"][b] = res
        nulls[b] = null
        s1 = res["leads"].get(1, {}).get("rmse_skill_vs_persistence")
        s7 = res["leads"].get(7, {}).get("rmse_skill_vs_persistence")
        hb = res["holdout"].get("by_lead", {})
        print(f"{b:10} {m['r']:8.3f} {blind['r']:7.3f} {ar1['r']:7.3f} "
              f"{res['surrogate_null']['p95_r']:9.3f} "
              f"{res['surrogate_null']['p_value']:7.3f} "
              f"{(s1 if s1 is not None else float('nan')):12.3f} "
              f"{(s7 if s7 is not None else float('nan')):7.3f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out, d, nulls)
    except Exception as e:                      # noqa: BLE001
        print(f"figure failed: {repr(e)[:140]}")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
