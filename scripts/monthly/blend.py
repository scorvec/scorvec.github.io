"""Blend Claude's statistical month-ahead forecast with the dynamical guidance (SEAS5, UKMO GloSea, CanSIPS; models.py).

Per variable, the models available for this issue (a hindcast table for the start month and, for a live forecast,
the live issue) are averaged with equal weight into a model mean, moved from their 1993-2016 base to 1991-2020 with
ERA5. The blend is then

    forecast = a(x) * statistical + b(x) * model mean,     a, b >= 0

with a and b from the same spatially smoothed non-negative regression as the statistical consolidation, fitted on
the years both exist (the C3S hindcast, 1993-2016, against the statistical forecast's cross-validated hindcast).
Its skill is cross-validated the same way (each year's weights, error spread and skill mask fitted without that
year), on the same years as the statistical forecast it is compared with. The blend is issued only where it beats
the statistical forecast in that comparison (hemispheric mean-square error); otherwise the statistical forecast
stands, and the page says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import forecast as F
import mo_method as M
import models as MD


def base_shift(A: M.Archive, v: str, month: int) -> np.ndarray:
    """ERA5 1993-2016 mean minus 1991-2020 mean of the target month: moves a C3S anomaly (against the 1993-2016
    hindcast) onto the outlook's 1991-2020 base."""
    x = M.month_series(A.cal[v], A.years, month, int(A.years[0]))[:, M.nh_rows(v)].reshape(len(A.years), -1)
    in_c3s = (A.years >= MD.HC_YEARS[0]) & (A.years <= MD.HC_YEARS[-1])
    in_clim = (A.years >= M.CLIM[0]) & (A.years <= M.CLIM[1])
    return np.nanmean(x[in_c3s], 0) - np.nanmean(x[in_clim], 0)


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
    for a live forecast, the live issue is in), res['vars'][v] is switched to the blend and the statistical forecast
    is kept under stat_* keys."""
    iy, mm = (target_year, month - 1) if month > 1 else (target_year - 1, 12)
    issue = f"{iy}{mm:02d}"
    K = M.sphere_kernel()
    w = F.hemi_weights()
    hy = np.array(res["hindcast_years"])
    res["blend"] = {}
    for v, o in res["vars"].items():
        shift = base_shift(A, v, month)
        models = [m for m in MD.MODELS if MD.hindcast(m, mm, v) is not None]
        if realtime:
            models = [m for m in models if MD.live(m, issue, v) is not None]
        if not models:
            continue
        hc_years = np.array(MD.HC_YEARS)
        use = [y for y in hc_years if y in hy]
        if len(use) < 15:
            continue
        ih = [int(np.where(hy == y)[0][0]) for y in use]
        im = [int(np.where(hc_years == y)[0][0]) for y in use]
        mme = np.mean([MD.hindcast(m, mm, v)[im] for m in models], 0) + shift
        stat = o["hind_cv"][ih]
        Y = o["hind_obs"][ih]
        X = np.stack([stat, mme], 1)  # (n, 2, P)
        n = len(use)
        cv = np.zeros_like(Y)
        for j in range(n):
            keep = np.arange(n) != j
            b_j = M.consolidation_weights(X[keep], Y[keep], K)
            cv[j] = np.einsum("pk,kp->p", b_j, X[j])
        beta = M.consolidation_weights(X, Y, K)
        clim, lo, hi = obs_clim(A, v, month)
        sc = {"stat": scores(stat, Y, clim, lo, hi, w), "models": scores(mme, Y, clim, lo, hi, w)}
        sc["blend"] = scores(cv, Y, clim, lo, hi, w)
        use_blend = sc["blend"]["mse"] < sc["stat"]["mse"]
        rec = {
            "models": models,
            "years": [int(use[0]), int(use[-1])],
            "n": n,
            "scores": {k: {kk: round(vv, 3) for kk, vv in d.items() if kk != "r_map"} for k, d in sc.items()},
            "weight_nh": {
                "stat": round(float((w * beta[:, 0]).sum() / w.sum()), 3),
                "models": round(float((w * beta[:, 1]).sum() / w.sum()), 3),
            },
            "issued": False,
        }
        if realtime and "fc" in o:
            live = np.mean([MD.live(m, issue, v) for m in models], 0) + shift
            fc_b = beta[:, 0] * o["fc"] + beta[:, 1] * live
            rec["issue"] = issue
            o["models_fc"] = live.astype(np.float32)
            o["blend_fc"] = fc_b.astype(np.float32)
            o["blend_beta"] = beta.astype(np.float32)
            if use_blend:
                # the issued forecast becomes the blend: its probabilities, error spread and skill mask from the
                # blend's own cross-validated hindcast on the shared years
                err = Y - cv
                r = sc["blend"]["r_map"]
                t = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-9))
                sig = F.bh_mask(stats.t.sf(t, n - 2), 0.10)
                b_, a_ = F.terciles(fc_b, err, lo, hi, clim)
                for key in ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw", "sig", "r"):
                    o[f"stat_{key}"] = o[key]
                o["fc"] = fc_b
                o["p_below_raw"], o["p_above_raw"] = b_, a_
                o["p_below"], o["p_above"] = np.where(sig, b_, 1 / 3), np.where(sig, a_, 1 / 3)
                o["sig"], o["r"] = sig, r
                rec["issued"] = True
        res["blend"][v] = rec
    return res
