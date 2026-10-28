"""Blend Claude's statistical forecast with the dynamical models: the C3S seasonal systems at every lead, then, for
next month, the whole 35-day GEFS run.

Stage 1, C3S (apply_c3s). The multi-system mean of the C3S issue published two days before the outlook (c3s_em.py:
seven systems from six centres, CanSIPS's two components pooled, each group weighing the same), over the target's
months, is a second predictor beside the statistical forecast:

    forecast = a(x) * statistical + b(x) * C3S,     a, b >= 0

with the weights at each point from the non-negative regression of the statistical consolidation (normal equations
smoothed over ~2500 km, ridge-damped toward zero), fitted on the 24 hindcast years 1993-2016 against the statistical
forecast's cross-validated hindcast. C3S anomalies are against each system's own 1993-2016 hindcast; the observed
1993-2016 mean anomaly moves them to 1991-2020. In that hindcast (September and October issues, every target to
lead 6) the C3S mean beat the statistical forecast for 2 m temperature (e.g. December-February ACC 0.39 against
0.19), precipitation and 500 hPa height, and the free blend matched or beat C3S alone; an anchored blend (the
statistical weight pulled toward 1, as for GEFS) scored worse, so the weights here are free.

Stage 2, GEFS (apply_gefs, next month only). The GEFS run of the 15th (gefs_guide.py) reaches the 18th or 19th of
the target month. It gives two predictors: its mean over the target month's days it reaches, and its mean over its
days before the month (the lead-in: the state the month starts from). The blend is

    forecast = a(x) * stage-1 forecast + b(x) * GEFS in the month + c(x) * GEFS lead-in,     a, b, c >= 0

fitted on the years both exist (the GEFS reforecast 2000-2019 against the stage-1 forecast's cross-validated
hindcast: 2000-2016 when C3S entered, 2000-2019 otherwise). The stage-1 forecast is already calibrated on its own
hindcast, so a is pulled toward 1 (a ridge penalty as strong as the years' own data, PRIOR) rather than toward 0:
refitted freely on 20 years the statistical weight fell to ~0.3, damping the whole forecast for a sampling accident
of those years. b and c are damped toward 0. The lead-in enters for temperature only (gefs_guide.LEAD), and over
2000-2016 GEFS added nothing to precipitation once C3S was in (it still enters; the regression gives it what it
earns).

Each stage is cross-validated (each year's weights fitted without that year) and issued for a variable only where it
beats the forecast before it in that comparison (hemispheric mean-square error); otherwise that forecast stands, and
the page says so. When issued, its probabilities take the statistical forecast's 1981-onward error spread scaled by
the stage's mean-square-error ratio on the shared years, and its skill mask is the statistical forecast's, plus the
points where the stage's own hindcast is significant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3s_em as CE
import forecast as F
import gefs_guide as GG
import mo_method as M

PRIOR = 1.0  # ridge pulling the stage-1 weight toward 1, in units of its own smoothed sum of squares
MIN_YEARS = 15


def anchored_weights(X: np.ndarray, Y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """X (n, k, P) [base forecast, model predictors...], Y (n, P) -> (P, k) >= 0 from the smoothed normal equations,
    the base forecast's weight ridge-pulled toward 1 (PRIOR) and the others toward 0 (the consolidation ridge)."""
    A = np.einsum("qp,pij->qij", K, np.einsum("nip,njp->pij", X, X))
    b = K @ np.einsum("nip,np->pi", X, Y)
    lam = np.zeros_like(A)
    for i in range(A.shape[1]):
        lam[:, i, i] = (PRIOR if i == 0 else M.CONS_RIDGE) * A[:, i, i]
    b[:, 0] += PRIOR * A[:, 0, 0]
    return M.nnls_batched(A + lam, b)


def obs_clim(A: M.Archive, v: str, month: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target month's anomalies over 1991-2020 (m, P) and its tercile bounds (an update's whole month)."""
    x = M.month_series(A.cal[v], A.years, month, int(A.years[0]))[:, M.nh_rows(v)].reshape(len(A.years), -1)
    a = M.clim_anom(x, A.years)
    c = (A.years >= M.CLIM[0]) & (A.years <= M.CLIM[1]) & np.isfinite(a).all(1)
    lo, hi = np.percentile(a[c], [100 / 3, 200 / 3], axis=0)
    return a[c], lo, hi


def scores(cv: np.ndarray, Y: np.ndarray, clim: np.ndarray, lo, hi, w) -> dict:
    """Cross-validated scores of a hindcast cv (n, P) against Y: mean hemispheric ACC, mean point correlation, MSE
    and RPSS of tercile probabilities (equal chances where the leave-one-out skill is not significant)."""
    n = len(cv)
    err = Y - cv
    pb, pa = np.zeros_like(cv), np.zeros_like(cv)
    for j in range(n):
        keep = np.arange(n) != j
        rj = F.corr(cv[keep], Y[keep])
        tj = rj * np.sqrt((n - 3) / np.maximum(1 - rj * rj, 1e-9))
        sk = F.bh_mask(stats.t.sf(tj, n - 3), 0.10)
        b_, a_ = F.terciles(cv[j], err[keep], lo, hi, clim)
        pb[j], pa[j] = np.where(sk, b_, 1 / 3), np.where(sk, a_, 1 / 3)
    cat = np.greater_equal(Y, lo).astype(int) + np.greater(Y, hi)
    rps = ((pb - (cat == 0)) ** 2 + ((1 - pa) - (cat <= 1)) ** 2).sum(0)
    ref = ((1 / 3 - (cat == 0)) ** 2 + (2 / 3 - (cat <= 1)) ** 2).sum(0)
    r = F.corr(cv, Y)
    return {
        "acc": float(np.mean([F.acc(cv[j], Y[j], w) for j in range(n)])),
        "r": float((w * r).sum() / w.sum()),
        "mse": float((w * (err**2).mean(0)).sum() / w.sum()),
        "rpss": float(1 - (w * rps).sum() / (w * ref).sum()),
        "r_map": r,
    }


def base(res: dict, o: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(years, cross-validated hindcast, observed, error scale) of the forecast a stage builds on: the statistical
    forecast's, or an earlier stage's when that was issued."""
    if "base_cv" in o:
        return o["base_years"], o["base_cv"], o["base_obs"], o["err_scale"]
    return np.array(res["hindcast_years"]), o["hind_cv"], o["hind_obs"], 1.0


def stage(o: dict, years: np.ndarray, Xh: np.ndarray, Y: np.ndarray, live: list | None, names: list[str],
          free: bool, scale0: float) -> dict:  # fmt: skip
    """Fit, cross-validate and (if it wins) issue one stage. Xh (n, k, P): [base hindcast, predictors...] on `years`,
    live: the k live inputs (or None for a skill-only run). Switches o's issued forecast when the stage beats the
    base; returns the stage's record."""
    K = M.sphere_kernel()
    w = F.hemi_weights()
    fit = M.consolidation_weights if free else anchored_weights
    n = len(years)
    cv = np.zeros_like(Y)
    for j in range(n):
        keep = np.arange(n) != j
        cv[j] = np.einsum("pk,kp->p", fit(Xh[keep], Y[keep], K), Xh[j])
    beta = fit(Xh, Y, K)
    clim, lo, hi = o["clim_anoms"], o["tlo"], o["thi"]
    sc = {nm: scores(Xh[:, i], Y, clim, lo, hi, w) for i, nm in enumerate(names)}
    sc["blend"] = scores(cv, Y, clim, lo, hi, w)
    win = sc["blend"]["mse"] < sc[names[0]]["mse"]
    rec = {
        "years": [int(years[0]), int(years[-1])],
        "n": n,
        "scores": {k: {kk: round(vv, 3) for kk, vv in d.items() if kk != "r_map"} for k, d in sc.items()},
        "weight_nh": {nm: round(float((w * beta[:, i]).sum() / w.sum()), 3) for i, nm in enumerate(names)},
        "issued": False,
    }
    if live is None or "fc" not in o:
        return rec
    fc_b = np.einsum("pk,kp->p", beta, np.stack(live))
    o[f"{names[1]}_beta"] = beta.astype(np.float32)
    o[f"{names[1]}_blend_fc"] = fc_b.astype(np.float32)
    if win:
        # the issued forecast becomes the blend: the statistical forecast's long-hindcast error spread, scaled by
        # the stage's mean-square-error ratio on the shared years, and its skill mask widened by the points where
        # the stage's own hindcast is significant
        ratio = sc["blend"]["mse"] / sc[names[0]]["mse"]
        scale = scale0 * np.sqrt(ratio)
        err = (o["hind_obs"] - o["hind_cv"]) * scale
        r = sc["blend"]["r_map"]
        t = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-9))
        sig = o["sig"] | F.bh_mask(stats.t.sf(t, n - 2), 0.10)
        b_, a_ = F.terciles(fc_b, err, lo, hi, clim)
        if "stat_fc" not in o:  # the statistical forecast, kept for the page and the record
            for key in ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw", "sig"):
                o[f"stat_{key}"] = o[key]
        o["fc"] = fc_b
        o["p_below_raw"], o["p_above_raw"] = b_, a_
        o["p_below"], o["p_above"] = np.where(sig, b_, 1 / 3), np.where(sig, a_, 1 / 3)
        o["sig"] = sig
        o["issued_r"] = r
        o["base_years"], o["base_cv"], o["base_obs"], o["err_scale"] = years, cv, Y, scale
        rec["issued"] = True
        rec["mse_ratio"] = round(float(ratio), 3)
    return rec


def apply_c3s(res: dict, issue: str, month: int, span: int, realtime: bool = True) -> dict:
    """Stage 1: res['c3s'][v] holds the comparison; where the blend wins, res['vars'][v] is switched to it (the C3S
    multi-system mean kept under c3s_fc, on the 1991-2020 base)."""
    res["c3s"] = {}
    for v, o in res["vars"].items():
        t = CE.target(issue, v, month, span, live=realtime)
        if t is None:
            continue
        hy, hcv, hobs, scale0 = base(res, o)
        use = [y for y in t["years"] if y in hy]
        if len(use) < MIN_YEARS:
            continue
        ih = [int(np.where(hy == y)[0][0]) for y in use]
        ic = [int(np.where(t["years"] == y)[0][0]) for y in use]
        Y = hobs[ih]
        shift = Y.mean(0)  # observed 1993-2016 mean anomaly: the C3S base (1993-2016) onto 1991-2020
        Xh = np.stack([hcv[ih], t["hind"][ic] + shift], 1)
        live = None
        if realtime and t.get("live") is not None:
            o["c3s_fc"] = (t["live"] + shift).astype(np.float32)
            live = [o["fc"], o["c3s_fc"]]
        rec = stage(o, np.array(use), Xh, Y, live, ["stat", "c3s"], free=True, scale0=scale0)
        rec["systems"] = t["systems"]
        rec["issue"] = issue
        res["c3s"][v] = rec
    return res


def apply_gefs(res: dict, target_year: int, month: int, realtime: bool = True) -> dict:
    """Stage 2 (next month): res['blend'][v] holds the comparison; where the blend wins, res['vars'][v] is switched
    to it (GEFS's two predictors kept under models_fc and lead_fc)."""
    res["blend"] = {}
    g = GG.live(target_year, month) if realtime else None
    if realtime and g is None:
        print("no GEFS run for this issue: the forecast stands without it", flush=True)
        return res
    for v, o in res["vars"].items():
        hc = GG.hindcast(month, v)
        if hc is None:
            continue
        hc_years, table, lead = hc
        hy, hcv, hobs, scale0 = base(res, o)
        use = [y for y in hc_years if y in hy]
        if len(use) < MIN_YEARS:
            continue
        ih = [int(np.where(hy == y)[0][0]) for y in use]
        im = [int(np.where(hc_years == y)[0][0]) for y in use]
        uses_lead = v in GG.LEAD and lead is not None
        Xh = np.stack([hcv[ih], table[im], lead[im]] if uses_lead else [hcv[ih], table[im]], 1)  # (n, 2 or 3, P)
        names = ["base", "models", "lead"] if uses_lead else ["base", "models"]
        live = None
        if realtime and "fc" in o:
            o["models_fc"] = g[v].astype(np.float32)
            live = [o["fc"], g[v].astype(np.float64)]
            if uses_lead:
                o["lead_fc"] = g[f"{v}_lead"].astype(np.float32)
                live.append(g[f"{v}_lead"].astype(np.float64))
        rec = stage(o, np.array(use), Xh, hobs[ih], live, names, free=False, scale0=scale0)
        rec["models"] = [GG.NAME]
        if live is not None:
            rec["init"] = g["init"]
            rec["days"] = [1, int(g["ndays"])]
        res["blend"][v] = rec
    return res
