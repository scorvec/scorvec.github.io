"""Blend Claude's statistical month-ahead forecast with the whole 35-day GEFS run.

The GEFS run of the 15th (gefs_guide.py) reaches the 18th or 19th of the target month. It gives two predictors: its
mean over the target month's days it reaches, and its mean over its days before the month (the lead-in: the state
the month starts from). Per variable the blend is

    forecast = a(x) * statistical + b(x) * GEFS in the month + c(x) * GEFS lead-in,     a, b, c >= 0

with the weights from the same spatially smoothed non-negative regression as the statistical consolidation, fitted
on the years both exist (the GEFS reforecast, 2000-2019, against the statistical forecast's cross-validated
hindcast) and against the whole target month: b and c say how much of each part of the run carries into the
monthly mean. The statistical forecast is already calibrated on its own 1981-onward hindcast, so a is pulled toward
1 (a ridge penalty as strong as the 20 years' own data, PRIOR) rather than toward 0: refitted freely on 20 years it
fell to ~0.3, damping the whole forecast for a sampling accident of those years, and the anchored fit scored the
same (and better for precipitation) in cross-validation. b and c are damped toward 0 like any consolidation weight.
The lead-in enters for temperature only (LEAD): in cross-validation over all twelve months it earned weight there
(c ~0.09, the correlation 0.290 -> 0.299), and for height and precipitation it earned little (~0.06, ~0.02) and made
the blend slightly worse. The month's days scored the same whether cut at day 15 or run to the end of the forecast.

Its skill is cross-validated the same way (each year's weights fitted without that year), on the same years as the
statistical forecast it is compared with. The blend is issued only where it beats the statistical forecast in that
comparison (hemispheric mean-square error); otherwise the statistical forecast stands, and the page says so. When
issued, its probabilities take the statistical forecast's 1981-onward error spread scaled by the blend's
mean-square-error ratio on the shared years, and its skill mask is the statistical forecast's, plus the points
where the blend's own 20-year hindcast is significant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import forecast as F
import gefs_guide as GG
import mo_method as M

PRIOR = 1.0  # ridge pulling the statistical weight toward 1, in units of its own smoothed sum of squares
LEAD = ("t2m",)  # variables that also take the run's lead-in (the days before the target month)


def anchored_weights(X: np.ndarray, Y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """X (n, k, P) [statistical, GEFS predictors...], Y (n, P) -> (P, k) >= 0 from the smoothed normal equations,
    the statistical weight ridge-pulled toward 1 (PRIOR) and the others toward 0 (the consolidation ridge)."""
    A = np.einsum("qp,pij->qij", K, np.einsum("nip,njp->pij", X, X))
    b = K @ np.einsum("nip,np->pi", X, Y)
    lam = np.zeros_like(A)
    for i in range(A.shape[1]):
        lam[:, i, i] = (PRIOR if i == 0 else M.CONS_RIDGE) * A[:, i, i]
    b[:, 0] += PRIOR * A[:, 0, 0]
    return M.nnls_batched(A + lam, b)


def obs_clim(A: M.Archive, v: str, month: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target month's anomalies over 1991-2020 (m, P) and its tercile bounds."""
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


def apply(res: dict, A: M.Archive, target_year: int, month: int, realtime: bool = True) -> dict:
    """Add the blend to a forecast.run() result: res['blend'][v] holds the comparison; where the blend wins (and,
    for a live forecast, the GEFS run is in), res['vars'][v] is switched to the blend and the statistical forecast
    is kept under stat_* keys (GEFS's two predictors under models_fc and lead_fc)."""
    K = M.sphere_kernel()
    w = F.hemi_weights()
    hy = np.array(res["hindcast_years"])
    res["blend"] = {}
    g = GG.live(target_year, month) if realtime else None
    if realtime and g is None:
        print("no GEFS run for this issue: the statistical forecast stands", flush=True)
        return res
    for v, o in res["vars"].items():
        hc = GG.hindcast(month, v)
        if hc is None:
            continue
        hc_years, table, lead = hc
        use = [y for y in hc_years if y in hy]
        if len(use) < 15:
            continue
        ih = [int(np.where(hy == y)[0][0]) for y in use]
        im = [int(np.where(hc_years == y)[0][0]) for y in use]
        gm = table[im]
        stat = o["hind_cv"][ih]
        Y = o["hind_obs"][ih]
        X = np.stack([stat, gm, lead[im]] if v in LEAD else [stat, gm], 1)  # (n, 2 or 3, P)
        n = len(use)
        cv = np.zeros_like(Y)
        for j in range(n):
            keep = np.arange(n) != j
            b_j = anchored_weights(X[keep], Y[keep], K)
            cv[j] = np.einsum("pk,kp->p", b_j, X[j])
        beta = anchored_weights(X, Y, K)
        clim, lo, hi = obs_clim(A, v, month)
        sc = {"stat": scores(stat, Y, clim, lo, hi, w), "models": scores(gm, Y, clim, lo, hi, w)}
        sc["blend"] = scores(cv, Y, clim, lo, hi, w)
        use_blend = sc["blend"]["mse"] < sc["stat"]["mse"]
        rec = {
            "models": [GG.NAME],
            "years": [int(use[0]), int(use[-1])],
            "n": n,
            "scores": {k: {kk: round(vv, 3) for kk, vv in d.items() if kk != "r_map"} for k, d in sc.items()},
            "weight_nh": {
                "stat": round(float((w * beta[:, 0]).sum() / w.sum()), 3),
                "models": round(float((w * beta[:, 1]).sum() / w.sum()), 3),
            },
            "issued": False,
        }
        if v in LEAD:
            rec["weight_nh"]["lead"] = round(float((w * beta[:, 2]).sum() / w.sum()), 3)
        if realtime and "fc" in o:
            live, live_lead = g[v].astype(np.float64), g[f"{v}_lead"].astype(np.float64)
            fc_b = beta[:, 0] * o["fc"] + beta[:, 1] * live + (beta[:, 2] * live_lead if v in LEAD else 0.0)
            rec["init"] = g["init"]
            rec["days"] = [1, int(g["ndays"])]
            o["models_fc"] = live.astype(np.float32)
            if v in LEAD:
                o["lead_fc"] = live_lead.astype(np.float32)
            o["blend_fc"] = fc_b.astype(np.float32)
            o["blend_beta"] = beta.astype(np.float32)
            if use_blend:
                # the issued forecast becomes the blend: the statistical forecast's long-hindcast error spread,
                # scaled by the blend's mean-square-error ratio on the shared years, and its skill mask widened by
                # the points where the blend's own 20-year hindcast is significant
                ratio = sc["blend"]["mse"] / sc["stat"]["mse"]
                err = (o["hind_obs"] - o["hind_cv"]) * np.sqrt(ratio)
                r = sc["blend"]["r_map"]
                t = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-9))
                sig = o["sig"] | F.bh_mask(stats.t.sf(t, n - 2), 0.10)
                b_, a_ = F.terciles(fc_b, err, lo, hi, clim)
                for key in ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw", "sig"):
                    o[f"stat_{key}"] = o[key]
                o["fc"] = fc_b
                o["p_below_raw"], o["p_above_raw"] = b_, a_
                o["p_below"], o["p_above"] = np.where(sig, b_, 1 / 3), np.where(sig, a_, 1 / 3)
                o["sig"] = sig
                rec["issued"] = True
                rec["mse_ratio"] = round(float(ratio), 3)
        res["blend"][v] = rec
    return res
