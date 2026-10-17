#!/usr/bin/env python3
"""Issue the month-ahead outlook (the target month's mean 500 hPa height and 2 m temperature anomalies, Northern
Hemisphere) from the ERA5 archive, with its cross-validated hindcast. Method in mo_method.py.

    python scripts/monthly/forecast.py                       # target = the month after today's
    python scripts/monthly/forecast.py --target 2026-11
    python scripts/monthly/forecast.py --skill-all           # hindcast skill of every target month (no output)

Writes assets/monthly/data/outlook_<YYYYMM>.npz (fields) and .json (summary, analogues, weights, skill), and
assets/monthly/data/latest.json pointing at the newest outlook. Rendering is render.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_grid as G
import mo_method as M

REPO = HERE.parents[1]
OUT = REPO / "assets" / "monthly" / "data"
HIND0 = 1981
ACC_LAT = 20.0  # hemispheric scores over 20-90N
CV_GAP = 1.5  # a hindcast leaves out every library sample within 1.5 years of its target month


def hemi_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w[G.LAT_NH < ACC_LAT] = 0.0
    return w.ravel()


def acc(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> float:
    """Uncentred anomaly correlation (anomalies vs 1991-2020), area weighted."""
    return float((w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum() + 1e-30))


def run(target_year: int, month: int, A: M.Archive, realtime: bool = True) -> dict:
    P = M.Problem(A, month)
    years = P.years
    hind = [int(y) for y in years if y >= HIND0 and y != target_year and P.available(y)]
    k = len(M.COMPONENTS)
    nP = P.obs["t2m"].shape[1]
    F = {v: np.zeros((len(hind), k, nP), np.float32) for v in M.TARGETS}
    analog_log = {}
    for j, y in enumerate(hind):
        comp = P.forecast(y, CV_GAP, target_year)
        analog_log[y] = P.last_analogs
        for v in M.TARGETS:
            F[v][j] = np.stack([comp[c][v] for c in M.COMPONENTS])
    hi = np.array([y - years[0] for y in hind])
    K = M.sphere_kernel()
    w = hemi_weights()
    res = {"target": f"{target_year}-{month:02d}", "month": month, "hindcast_years": hind, "vars": {}}
    zc = M.month_series(A.cal["z500"], years, month, int(years[0])).reshape(len(years), -1)
    res["z500_clim"] = np.nanmean(zc[(years >= M.CLIM[0]) & (years <= M.CLIM[1])], 0)
    rt_ok = realtime and P.available(target_year, need_target=False)
    if rt_ok:
        comp_rt = P.forecast(target_year, 0.0, target_year)
        res["analogs"] = P.last_analogs
        ca_t, alpha = P.last_alpha
        top = np.argsort(-np.abs(alpha))[:10]
        res["ca_top"] = [(f"{int(ca_t[t]):d}-{round(ca_t[t] % 1 * 12) + 1:02d}", float(alpha[t])) for t in top]
        it = target_year - years[0]
        im, yi = (month - 1, target_year) if month > 1 else (12, target_year - 1)
        raw = {k: float(A.win[k][A.t(yi, im)]) for k in ("u50eq", "u50v")}
        res["indices"] = {
            "n34r": float(P.n34r[it]),
            "roni": float(P.roni[it]),
            "n34": float(P.n34[it]),
            "n34r_pct": float((P.n34r[np.isfinite(P.n34r)] < P.n34r[it]).mean()),
            "u50eq": float(P.idx[it, 2]),
            "u50v": float(P.idx[it, 3]),
            "u50eq_raw": raw["u50eq"],
            "u50v_raw": raw["u50v"],
        }
    for v in M.TARGETS:
        Y = P.obs[v][hi]
        beta = M.consolidation_weights(F[v], Y, K)
        # leave-one-year-out weights for the cross-validated consolidated hindcast
        cv = np.zeros_like(Y)
        for j in range(len(hind)):
            keep = np.arange(len(hind)) != j
            b_j = M.consolidation_weights(F[v][keep], Y[keep], K)
            cv[j] = np.einsum("pk,kp->p", b_j, F[v][j])

        # skill: per point correlation, MSSS, and hemispheric ACC by year
        def corr(a, b):
            a = a - a.mean(0)
            b = b - b.mean(0)
            return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-30)

        r = corr(cv, Y)
        msss = 1 - ((cv - Y) ** 2).sum(0) / ((Y**2).sum(0) + 1e-30)
        n = len(hind)
        tstat = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-9))
        p = stats.t.sf(tstat, n - 2)  # one-sided: skill > 0
        sig = bh_mask(p, 0.10)
        comp_r = {c: corr(F[v][:, i], Y) for i, c in enumerate(M.COMPONENTS)}
        acc_y = [acc(cv[j], Y[j], w) for j in range(n)]
        rmse = np.sqrt(((cv - Y) ** 2).mean(0))
        out = {
            "beta": beta,
            "r": r,
            "msss": msss,
            "sig": sig,
            "rmse": rmse,
            "comp_r": comp_r,
            "acc_by_year": acc_y,
            "hind_cv": cv,
            "hind_obs": Y,
        }
        if rt_ok:
            Frt = np.stack([comp_rt[c][v] for c in M.COMPONENTS])  # (k, P)
            out["comp"] = Frt
            out["fc"] = np.einsum("pk,kp->p", beta, Frt)
            out["contrib"] = beta.T * Frt  # (k, P)
            # probability of an above-normal month (above the 1991-2020 mean), Gaussian CV errors
            out["p_above"] = stats.norm.cdf(out["fc"] / np.maximum(rmse, 1e-6))
        res["vars"][v] = out
    return res


def bh_mask(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate control at level q over the 20-90N points."""
    w = hemi_weights() > 0
    ps = p[w]
    order = np.argsort(ps)
    m = len(ps)
    thr = q * np.arange(1, m + 1) / m
    passed = ps[order] <= thr
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    cut = ps[order][kmax - 1] if kmax else -1.0
    out = np.zeros_like(p, bool)
    out[w] = ps <= cut
    return out


def summarise(res: dict, issued: str) -> dict:
    w = hemi_weights()
    s = {
        "target": res["target"],
        "target_label": M.target_label(int(res["target"][:4]), res["month"]),
        "issued": issued,
        "window": window_label(int(res["target"][:4]), res["month"]),
        "hindcast": [res["hindcast_years"][0], res["hindcast_years"][-1]],
        "components": {c: M.NAMES[c] for c in M.COMPONENTS},
        "vars": {},
    }
    for key in ("analogs", "ca_top", "indices"):
        if key in res:
            s[key] = res[key]
    for v, o in res["vars"].items():
        sv = {
            "nh_acc_mean": round(float(np.mean(o["acc_by_year"])), 3),
            "nh_acc_by_year": dict(zip(map(str, res["hindcast_years"]), np.round(o["acc_by_year"], 3).tolist())),
            "skill_r_nh": round(float((w * o["r"]).sum() / w.sum()), 3),
            "skill_area_sig": round(float((w * o["sig"]).sum() / w.sum()), 3),
            "comp_r_nh": {c: round(float((w * r).sum() / w.sum()), 3) for c, r in o["comp_r"].items()},
            "weight_nh": {
                c: round(float((w * o["beta"][:, i]).sum() / w.sum()), 3) for i, c in enumerate(M.COMPONENTS)
            },
        }
        if "fc" in o:
            fc = o["fc"]
            sv["fc_nh_mean"] = round(float((w * fc).sum() / w.sum()), 3)
            sv["contrib_rms"] = {
                c: round(float(np.sqrt((w * o["contrib"][i] ** 2).sum() / w.sum())), 3)
                for i, c in enumerate(M.COMPONENTS)
            }
        s["vars"][v] = sv
    return s


def window_label(y: int, m: int) -> str:
    import calendar

    im, yi = (m - 1, y) if m > 1 else (12, y - 1)
    pm = im - 1 if im > 1 else 12
    return f"{G.SPLIT_DAY} {calendar.month_abbr[pm]} – {G.SPLIT_DAY - 1} {calendar.month_abbr[im]} {yi}"


def save(res: dict, summary: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = res["target"].replace("-", "")
    arrs = {}
    for v, o in res["vars"].items():
        for key in ("fc", "p_above", "r", "sig", "rmse", "comp", "contrib", "beta"):
            if key in o:
                arrs[f"{v}_{key}"] = np.asarray(o[key]).astype(np.float32 if key != "sig" else np.bool_)
    arrs["z500_clim"] = res["z500_clim"].astype(np.float32)
    np.savez_compressed(OUT / f"outlook_{tag}.npz", **arrs)
    (OUT / f"outlook_{tag}.json").write_text(json.dumps(summary, indent=1))
    (OUT / "latest.json").write_text(json.dumps({"target": res["target"], "file": f"outlook_{tag}.json"}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="YYYY-MM (default: the month after today's)")
    ap.add_argument("--skill-all", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-issue an outlook that already exists")
    a = ap.parse_args()
    A = M.load_archive()
    if a.skill_all:
        for m in range(1, 13):
            res = run(9999, m, A, realtime=False)
            line = [f"{m:2d}"]
            for v in M.TARGETS:
                s = summarise(res, "")["vars"][v]
                line.append(
                    f"{v}: acc {s['nh_acc_mean']:+.2f} r {s['skill_r_nh']:+.2f} sig {s['skill_area_sig']:.2f} "
                    + " ".join(f"{c}{s['comp_r_nh'][c]:+.2f}/{s['weight_nh'][c]:.2f}" for c in M.COMPONENTS)
                )
            print(" | ".join(line), flush=True)
        return 0
    now = datetime.now(UTC)
    if a.target:
        ty, tm = map(int, a.target.split("-"))
    else:
        ty, tm = (now.year, now.month + 1) if now.month < 12 else (now.year + 1, 1)
    if (OUT / f"outlook_{ty}{tm:02d}.json").exists() and not a.force:
        print(f"the {ty}-{tm:02d} outlook was already issued; --force re-issues it")
        return 0
    res = run(ty, tm, A)
    if "analogs" not in res:
        raise SystemExit(f"the predictor window for {ty}-{tm:02d} is not complete in the archive")
    summary = summarise(res, now.strftime("%Y-%m-%d"))
    save(res, summary)
    print(json.dumps(summary, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
