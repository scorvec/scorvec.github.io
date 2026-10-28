#!/usr/bin/env python3
"""Issue the long-range outlook: the month after the issue and the three seasons from it (500 hPa height, 2 m
temperature and precipitation anomalies and tercile probabilities, Northern Hemisphere), each the statistical
forecast (mo_method.py) with its cross-validated hindcast, blended with the C3S multi-system mean and, for the month,
the GEFS run of the 15th (blend.py).

    python scripts/monthly/forecast.py                       # the issue for the month after today's
    python scripts/monthly/forecast.py --target 2026-11
    python scripts/monthly/forecast.py --seasons-only        # the seasons alone, the month standing as issued
    python scripts/monthly/forecast.py --skill-all           # hindcast skill of every target month (no output)
    python scripts/monthly/forecast.py --target 2026-11 --update vortex_202611.json   # an update (below)

Writes assets/monthly/data/outlook_<YYYYMM>.npz (fields) and .json (summary, analogues, weights, skill) for the month
and season_<YYYYMM>_l<lead>.* for the seasons (YYYYMM their first month), and latest.json naming the newest issue's
files (and enso_<YYYYMM>.json, enso_fc.py's Niño-3.4 forecast of the issue month, which runs first and also feeds the
forecast-ENSO component). Rendering is render.py.

An update (--update, after the issue on the 15th, when the stratosphere changes course) re-issues the target month
from the medium range and the predicted stratosphere (update.py): its days 1-7 from IFS-ENS and AIFS-ENS, its days
from the 8th from this method trained for those days, from the window ending on the 7th of the target month (ERA5,
analyses and the two ensembles: update_data.py) and the vortex predicted over days 1-7 (vortex_window.py, every member
of four ensembles). It is written as outlook_<YYYYMM>u<DD>.* beside the issue it updates, which stays on record and is
scored as issued; the seasons stand.
"""

from __future__ import annotations

import argparse
import copy
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


def run(
    target_year: int,
    month: int,
    A: M.Archive,
    realtime: bool = True,
    vortex: str = "issue",
    vortex_live: float | None = None,
    vortex_members: np.ndarray | None = None,
    part: int | None = None,
    window: str = "issue",
    window_live: dict | None = None,
    lead: int = 1,
    span: int = 1,
    enso=None,
) -> dict:
    """The outlook for (target_year, month) with its cross-validated hindcast. late_vortex: the update's vortex
    predictor (the window ending on the 7th of the target month), vortex_live its value this year and vortex_members
    the values whose forecasts are averaged (forecast() runs once per value; the hindcast does not depend on them)."""
    P = M.Problem(A, month, vortex, (target_year, vortex_live) if vortex_live is not None else None, part, window,
                  window_live, lead, span, enso)  # fmt: skip
    years = P.years
    hind = [int(y) for y in years if y >= HIND0 and y != target_year and P.available(y)]
    k = len(M.COMPONENTS)
    F = {v: np.zeros((len(hind), k, P.obs[v].shape[1]), np.float32) for v in M.TARGETS}
    analog_log = {}
    for j, y in enumerate(hind):
        comp = P.forecast(y, CV_GAP, target_year)
        analog_log[y] = P.last_analogs
        for v in M.TARGETS:
            F[v][j] = np.stack([comp[c][v] for c in M.COMPONENTS])
    hi = np.array([y - years[0] for y in hind])
    K = M.sphere_kernel()
    w = hemi_weights()
    res = {"target": f"{target_year}-{month:02d}", "month": month, "lead": lead, "span": span, "hindcast_years": hind,
           "vars": {}}  # fmt: skip
    zc = sum(
        M.month_series(A.cal["z500"], years, month + d, int(years[0])).reshape(len(years), -1) for d in range(span)
    )
    res["z500_clim"] = np.nanmean(zc[(years >= M.CLIM[0]) & (years <= M.CLIM[1])], 0) / span
    rt_ok = realtime and P.available(target_year, need_target=False)
    if rt_ok:
        comp_rt = P.forecast(target_year, 0.0, target_year)
        res["analogs"] = P.last_analogs
        ca_t, alpha = P.last_alpha
        top = np.argsort(-np.abs(alpha))[:10]
        res["ca_top"] = [(f"{int(ca_t[t]):d}-{round(ca_t[t] % 1 * 12) + 1:02d}", float(alpha[t])) for t in top]
        it = target_year - years[0]
        im, yi = (month - lead - 1) % 12 + 1, target_year + (month - lead - 1) // 12
        raw = {k: float(A.win[k][A.t(yi, im)]) for k in ("u50eq", "u50v")}
        if window_live is not None:  # an update's window
            raw["u50eq"] = float(window_live["u50eq"])
        if vortex != "issue":
            raw["u50v"] = vortex_live
        comp_m = [comp_rt]
        if vortex_members is not None:  # the forecast for each value of the vortex predictor
            a0 = float(P.c.idx[it, 3])
            comp_m = []
            for x in vortex_members:
                P.c.idx[it, 3] = P.c.win["u50v"][it, 0] = a0 + (x - vortex_live)
                comp_m.append(P.forecast(target_year, 0.0, target_year))
            P.c.idx[it, 3] = P.c.win["u50v"][it, 0] = a0
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

        r = corr(cv, Y)
        msss = 1 - ((cv - Y) ** 2).sum(0) / ((Y**2).sum(0) + 1e-30)
        n = len(hind)
        tstat = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-9))
        sig = bh_mask(stats.t.sf(tstat, n - 2), 0.10)  # one-sided: skill > 0
        comp_r = {c: corr(F[v][:, i], Y) for i, c in enumerate(M.COMPONENTS)}
        acc_y = [acc(cv[j], Y[j], w) for j in range(n)]
        rmse = np.sqrt(((cv - Y) ** 2).mean(0))
        # skill beyond the trend: the trend component alone (unweighted) as the reference forecast
        TRf = F[v][:, M.COMPONENTS.index("TR")]
        msss_tr = 1 - ((cv - Y) ** 2).sum(0) / (((TRf - Y) ** 2).sum(0) + 1e-30)
        r_beyond = corr(cv - TRf, Y - TRf)
        acc_tr = [acc(TRf[j], Y[j], w) for j in range(n)]
        # tercile probabilities against the 1991-2020 terciles of each point (terciles()), set to equal chances
        # where the hindcast has no significant skill (CPC's "EC"); each hindcast year's probabilities leave that
        # year out of the error spread, the climatology and the skill test
        clim = (years >= M.CLIM[0]) & (years <= M.CLIM[1]) & np.isfinite(P.obs[v]).all(1)
        lo, hi_t = np.percentile(P.obs[v][clim], [100 / 3, 200 / 3], axis=0)
        e2 = (cv - Y) ** 2
        err = Y - cv
        cy = np.where(clim)[0]
        pb, pa = np.zeros_like(cv), np.zeros_like(cv)
        for j, yj in enumerate(hi):
            keep = np.arange(n) != j
            rj = corr(cv[keep], Y[keep])
            tj = rj * np.sqrt((n - 3) / np.maximum(1 - rj * rj, 1e-9))
            skj = bh_mask(stats.t.sf(tj, n - 3), 0.10)
            b_, a_ = terciles(cv[j], err[keep], lo, hi_t, P.obs[v][cy[cy != yj]])
            pb[j], pa[j] = np.where(skj, b_, 1 / 3), np.where(skj, a_, 1 / 3)
        cat = np.greater_equal(Y, lo).astype(int) + np.greater(Y, hi_t)  # 0 below, 1 near, 2 above normal
        rps_fc = (pb - (cat == 0)) ** 2 + ((1 - pa) - (cat <= 1)) ** 2
        rps_cl = (1 / 3 - (cat == 0)) ** 2 + (2 / 3 - (cat <= 1)) ** 2
        rpss = 1 - rps_fc.sum(0) / rps_cl.sum(0)
        out = {
            "beta": beta,
            "r": r,
            "msss": msss,
            "sig": sig,
            "rmse": rmse,
            "comp_r": comp_r,
            "acc_by_year": acc_y,
            "acc_trend_by_year": acc_tr,
            "msss_trend": msss_tr,
            "r_beyond": r_beyond,
            "rpss": rpss,
            "rpss_nh": float(1 - (w * rps_fc.sum(0)).sum() / (w * rps_cl.sum(0)).sum()),
            "msss_trend_nh": float(1 - (w * e2.sum(0)).sum() / (w * ((TRf - Y) ** 2).sum(0)).sum()),
            "hind_cv": cv,
            "hind_obs": Y,
            "clim_anoms": P.obs[v][clim],
            "tlo": lo,
            "thi": hi_t,
        }
        if rt_ok:
            Fm = np.stack([np.stack([cm[c][v] for c in M.COMPONENTS]) for cm in comp_m])  # (members, k, P)
            Frt = Fm.mean(0)  # (k, P)
            out["comp"] = Frt
            out["fc"] = np.einsum("pk,kp->p", beta, Frt)
            out["contrib"] = beta.T * Frt  # (k, P)
            out["fc_members"] = np.einsum("pk,mkp->mp", beta, Fm)
            b_, a_ = mixed_terciles(out["fc_members"], err, lo, hi_t, P.obs[v][clim])
            out["p_below_raw"], out["p_above_raw"] = b_, a_
            out["p_below"], out["p_above"] = np.where(sig, b_, 1 / 3), np.where(sig, a_, 1 / 3)
        res["vars"][v] = out
    return res


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation along axis 0, per point."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-30)


def terciles(
    mu: np.ndarray, err: np.ndarray, lo: np.ndarray, hi: np.ndarray, clim: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """P(below the lower tercile) and P(above the upper) for the forecast mu (P,), given the hindcast's errors
    err (n, P) (observed minus forecast) and the 1991-2020 anomalies clim (m, P): the point's own climatology,
    shifted by the forecast and narrowed by the hindcast's error relative to the climatological spread
    (x_k = mu + s c_k, s = rms error / climatological sd, at most 1.5), kernel-smoothed. A forecast of zero from a
    skill-less hindcast is then exactly climatology (1/3 each), and a skewed variable such as precipitation keeps
    its skew (a normal distribution around the forecast scored the same for temperature and height)."""
    sd = np.maximum(np.sqrt((err**2).mean(0)), 1e-6)
    c = clim - clim.mean(0)
    sc = np.maximum(c.std(0), 1e-6)
    s = np.minimum(sd / sc, 1.5)
    x = mu + s * c  # (m, P)
    h = np.maximum(0.25 * s * sc, 1e-6)
    return stats.norm.cdf((lo - x) / h).mean(0), stats.norm.sf((hi - x) / h).mean(0)


def mixed_terciles(
    fcs: np.ndarray, err: np.ndarray, lo: np.ndarray, hi: np.ndarray, clim: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """terciles() of each forecast in fcs (n, P), averaged: one forecast gives terciles() itself; an update's
    forecasts for the vortex members spread the probabilities by the vortex forecast's own uncertainty."""
    bs, as_ = zip(*(terciles(f, err, lo, hi, clim) for f in fcs), strict=True)
    return np.mean(bs, 0), np.mean(as_, 0)


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
    y, span, lead = int(res["target"][:4]), res.get("span", 1), res.get("lead", 1)
    s = {
        "target": res["target"],
        "span": span,
        "lead": lead,
        "target_label": M.target_label(y, res["month"], span),
        "target_short": M.target_short(y, res["month"], span),
        "issued": issued,
        "window": window_label(y, res["month"], res.get("update") is not None, lead),
        "hindcast": [res["hindcast_years"][0], res["hindcast_years"][-1]],
        "components": {c: M.NAMES[c] for c in M.COMPONENTS},
        "vars": {},
    }
    for key in ("analogs", "ca_top", "indices", "c3s", "blend", "update", "enso"):
        if key in res:
            s[key] = res[key]
    for v, o in res["vars"].items():
        sv = {
            "nh_acc_mean": round(float(np.mean(o["acc_by_year"])), 3),
            "nh_acc_by_year": dict(zip(map(str, res["hindcast_years"]), np.round(o["acc_by_year"], 3).tolist())),
            "skill_r_nh": round(float((w * o["r"]).sum() / w.sum()), 3),
            "acc_trend_mean": round(float(np.mean(o["acc_trend_by_year"])), 3),
            "acc_trend_by_year": dict(
                zip(map(str, res["hindcast_years"]), np.round(o["acc_trend_by_year"], 3).tolist())
            ),
            "msss_trend_nh": round(o["msss_trend_nh"], 3),
            "r_beyond_nh": round(float((w * o["r_beyond"]).sum() / w.sum()), 3),
            "rpss_nh": round(o["rpss_nh"], 3),
            "skill_area_sig": round(float((w * o["sig"]).sum() / w.sum()), 3),
            "comp_r_nh": {c: round(float((w * r).sum() / w.sum()), 3) for c, r in o["comp_r"].items()},
            "weight_nh": {
                c: round(float((w * o["beta"][:, i]).sum() / w.sum()), 3) for i, c in enumerate(M.COMPONENTS)
            },
        }
        # the issued forecast's own hindcast: the statistical forecast's, or the last blend stage issued
        sv["issued"] = {"inputs": ["statistical"], "acc": sv["nh_acc_mean"], "rpss": sv["rpss_nh"],
                        "years": s["hindcast"]}  # fmt: skip
        for key, name in (("c3s", "C3S"), ("blend", "GEFS")):
            st = res.get(key, {}).get(v)
            if st and st["issued"]:
                sv["issued"] = {"inputs": [*sv["issued"]["inputs"], name], "acc": st["scores"]["blend"]["acc"],
                                "rpss": st["scores"]["blend"]["rpss"], "years": st["years"]}  # fmt: skip
        if "fc" in o:
            fc = o["fc"]
            sv["fc_nh_mean"] = round(float((w * fc).sum() / w.sum()), 3)
            sv["contrib_rms"] = {
                c: round(float(np.sqrt((w * o["contrib"][i] ** 2).sum() / w.sum())), 3)
                for i, c in enumerate(M.COMPONENTS)
            }
        s["vars"][v] = sv
    return s


def window_label(y: int, m: int, update: bool = False, lead: int = 1) -> str:
    """The predictor window: the ~30 days ending on the 7th of the issue month, `lead` months before the target's
    first month (an update's: of the target month)."""
    import calendar

    im, yi = (m - lead - 1) % 12 + 1, y + (m - lead - 1) // 12
    if update:
        im, yi = m, y
    pm = im - 1 if im > 1 else 12
    return f"{G.SPLIT_DAY} {calendar.month_abbr[pm]} – {G.SPLIT_DAY - 1} {calendar.month_abbr[im]} {yi}"


def save(res: dict, summary: dict, name: str) -> None:
    """assets/monthly/data/<name>.npz (the fields) and .json (the summary)."""
    OUT.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for v, o in res["vars"].items():
        keys = ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw", "tlo", "thi", "r", "sig", "rmse", "rpss")
        keys = (*keys, "msss_trend", "comp", "contrib", "models_fc", "lead_fc", "mr_fc", "c3s_fc")
        keys = (*keys, "c3s_beta", "c3s_blend_fc", "models_beta", "models_blend_fc", "issued_r")
        keys = (*keys, *(f"stat_{k}" for k in ("fc", "p_below", "p_above", "p_below_raw", "p_above_raw", "sig", "r")))
        for key in (*keys, "beta"):
            if key in o:
                arrs[f"{v}_{key}"] = np.asarray(o[key]).astype(np.bool_ if key.endswith("sig") else np.float32)
    arrs["z500_clim"] = res["z500_clim"].astype(np.float32)
    np.savez_compressed(OUT / f"{name}.npz", **arrs)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1))


def season_name(ty: int, tm: int, lead: int) -> str:
    """The file stem of a season outlook: its first month and its lead (months from the issue)."""
    return f"season_{ty}{tm:02d}_l{lead}"


def write_latest(**kw) -> None:
    """latest.json: the newest issue's files (an update replaces the month's file and keeps the rest)."""
    p = OUT / "latest.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    p.write_text(json.dumps(cur, indent=1))


def enso_live(issue: str, target_year: int):
    """The forecast-ENSO component's input with this issue's Niño-3.4 forecast (enso_fc.py writes it first), or the
    hindcast table alone when there is no issue (the component is then zero for the year being forecast)."""
    import enso_fc as EF

    if not EF.HC_TABLE.exists():
        return None
    live = None
    p = OUT / f"enso_{issue.replace('-', '')}.json"
    if p.exists():
        e = json.loads(p.read_text())
        sc = EF.roni_scale()
        rel = {t: x / sc[int(t[5:]) - 1] for t, x in zip(e["targets"], e["series"]["roni"]["claude"], strict=True)}
        live = {"year": target_year, "rel": rel}
    return EF.EnsoTable(live=live)


def issue_all(A: M.Archive, ty: int, tm: int, issued: str, force: bool, seasons_only: bool = False) -> None:
    """The issue for target month (ty, tm) on the 15th of the month before: the month itself and the three seasons
    from it (lead 1-3), each the statistical forecast blended with C3S where that beats it, the month also with
    GEFS. seasons_only: the seasons alone, the month's outlook standing as issued."""
    import blend as B  # blend imports this module

    iy, im = (ty, tm - 1) if tm > 1 else (ty - 1, 12)
    issue = f"{iy}{im:02d}"
    done = (OUT / season_name(ty, tm, 1)).with_suffix(".json") if seasons_only else OUT / f"outlook_{ty}{tm:02d}.json"
    if done.exists() and not force:
        print(f"the {issue} issue was already made; --force re-issues it")
        return
    E = enso_live(f"{iy}-{im:02d}", ty)
    seasons = []
    for span, lead in ((3, 1), (3, 2), (3, 3)) if seasons_only else ((1, 1), (3, 1), (3, 2), (3, 3)):
        m0 = (tm - 1 + max(lead - 1, 0)) % 12 + 1
        y0 = ty + (tm - 1 + max(lead - 1, 0)) // 12
        E_k = E
        if E is not None and E.live is not None:  # the season's year: the forecast fills that year's row
            E_k = copy.copy(E)
            E_k.live = {**E.live, "year": y0}
        res = run(y0, m0, A, lead=lead, span=span, enso=E_k)
        if "analogs" not in res:
            raise SystemExit(f"the predictor window for the {issue} issue is not complete in the archive")
        res = B.apply_c3s(res, issue, m0, span)
        if span == 1:
            res = B.apply_gefs(res, y0, m0)
        name = f"outlook_{y0}{m0:02d}" if span == 1 else season_name(y0, m0, lead)
        res["enso"] = {"used": E_k is not None and E_k.live is not None}
        summary = summarise(res, issued)
        save(res, summary, name)
        if span == 3:
            seasons.append(f"{name}.json")
        print(f"{name}: " + ", ".join(f"{v} ACC {summary['vars'][v]['nh_acc_mean']:+.3f}"
                                      + (" +C3S" if res["c3s"].get(v, {}).get("issued") else "")
                                      + (" +GEFS" if res.get("blend", {}).get(v, {}).get("issued") else "")
                                      for v in M.TARGETS), flush=True)  # fmt: skip
    enso = f"enso_{issue}.json"
    month = {} if seasons_only else {"file": f"outlook_{ty}{tm:02d}.json"}
    write_latest(issue=f"{iy}-{im:02d}", target=f"{ty}-{tm:02d}", seasons=seasons,
                 enso=enso if (OUT / enso).exists() else None, **month)  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="YYYY-MM (default: the month after today's)")
    ap.add_argument("--skill-all", nargs="?", const="1,2,3,4,5,6,7,8,9,10,11,12", help="hindcast skill of months")
    ap.add_argument("--force", action="store_true", help="re-issue an outlook that already exists")
    ap.add_argument("--seasons-only", action="store_true", help="issue the seasons alone (the month stands as issued)")
    ap.add_argument("--update", type=Path, help="an update: the vortex over the target's days 1-7 (vortex_window.py)")
    ap.add_argument("--data", type=Path, help="an update: its first week and window (update_data.py)")
    a = ap.parse_args()
    A = M.load_archive()
    if a.skill_all:
        import blend as B  # blend imports this module

        for m in map(int, a.skill_all.split(",")):
            res = B.apply_gefs(run(9999, m, A, realtime=False), 9999, m, realtime=False)
            line = [f"{m:2d}"]
            for v in M.TARGETS:
                s = summarise(res, "")["vars"][v]
                line.append(
                    f"{v}: acc {s['nh_acc_mean']:+.3f} (trend {s['acc_trend_mean']:+.3f}) r {s['skill_r_nh']:+.3f} "
                    f"sig {s['skill_area_sig']:.2f} rpss {s['rpss_nh']:+.3f} msss_tr {s['msss_trend_nh']:+.3f} "
                    + " ".join(f"{c}{s['comp_r_nh'][c]:+.2f}" for c in M.COMPONENTS)
                )
            for v, b in res.get("blend", {}).items():
                sc = b["scores"]
                line.append(
                    f"blend {v} {b['years'][0]}-{b['years'][1]}: "
                    + " ".join(
                        f"{k} acc {d['acc']:+.3f} rpss {d['rpss']:+.3f} mse {d['mse']:.3g}" for k, d in sc.items()
                    )
                )
            print(" | ".join(line), flush=True)
        return 0
    now = datetime.now(UTC)
    if a.target:
        ty, tm = map(int, a.target.split("-"))
    else:
        ty, tm = (now.year, now.month + 1) if now.month < 12 else (now.year + 1, 1)
    if not a.update:
        issue_all(A, ty, tm, now.strftime("%Y-%m-%d"), a.force, a.seasons_only)
        return 0
    tag = f"{ty}{tm:02d}u{now:%d}"
    if (OUT / f"outlook_{tag}.json").exists() and not a.force:
        print(f"the {tag} outlook was already issued; --force re-issues it")
        return 0
    import update as U  # update imports this module

    res = U.issue(ty, tm, A, a.update, a.data, OUT / f"outlook_{ty}{tm:02d}.json")
    summary = summarise(res, now.strftime("%Y-%m-%d"))
    save(res, summary, f"outlook_{tag}")
    write_latest(file=f"outlook_{tag}.json")
    print(json.dumps(summary, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
