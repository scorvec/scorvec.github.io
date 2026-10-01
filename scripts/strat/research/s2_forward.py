#!/usr/bin/env python3
"""Q3/Q4 of the 2026-09-30 SSW precursor study: forward prediction on ALL eligible winter days.

Sample: every Nov 1 - Mar 31 day of the MERRA-2 winters 1980/81-2025/26 on which u(60N,10hPa) > 0 and the vortex is
not inside a CP07 refractory period (from a reversal until 20 consecutive westerly days; marginal reversals included,
so a day is never "eligible" right after a reversal the catalogue lists).
Outcome: y_w = 1 if one of the 28 confirmed central dates falls in (t + a, t + b] for windows
  1-14, 15-28, 21-42 (the 3-6 week window), 43-63 days.
Predictors (all daily, standardised): seasonal harmonics (2), QBO = CPC 50 hPa index, ENSO = RONI of the season centred
one month earlier (no look-ahead), U = u(60N,10hPa) anomaly, T = ERA5 100-50 hPa 0-15N layer temperature anomaly,
dT = its TRAILING 10-day change, HF = 100 hPa 45-75N heat flux anomaly, HF40 = its trailing 40-day mean; and the
structure metrics of s0_metrics.py.
Models: logistic, fitted leave-one-winter-out (LOWO); skill on the pooled out-of-sample predictions: Brier score,
BSS against the seasonal climatology and against the seasonal + QBO + ENSO baseline, ROC AUC. Significance: winter-
block bootstrap (2,000 resamples of whole winters) of the Brier-score difference against the reference; two-sided p;
Benjamini-Hochberg FDR 10 % across every model x window test of the family.
Output: ~/research/ssw_structure/data/s2_forward.json, s2_preds.pkl
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sswr_common as C

WINDOWS = {"1-14": (1, 14), "15-28": (15, 28), "21-42": (21, 42), "43-63": (43, 63)}
NBOOT = 2000
STRUCT = ["jet_lat", "jet_width", "depth_ratio", "top_shear", "u60_100", "qy_edge", "qy_pole", "bt_neg_upper",
          "qg_neg_upper", "bt_neg_flank", "n2k1", "n2k2", "n2k2_pos", "cen_lat", "aspect", "area"]


def build_frame():
    u = C.u10()
    E = C.events()
    rev = C.all_reversals()
    refr = C.refractory(u, rev)
    tt = C.ttr()
    M = pd.read_pickle(C.DATA / "metrics_daily.pkl")
    d = pd.DataFrame(index=u.index)
    d["u"] = u
    d["U"] = C.std_anom(u)
    d["T"] = C.std_anom(tt).reindex(d.index)
    d["dT"] = C.std_anom(tt.diff(10)).reindex(d.index)
    hf = C.std_anom(M.vT100)
    d["HF"] = hf
    h40 = hf.rolling(40, min_periods=30).mean()
    d["HF40"] = h40 / h40[h40.index.month.isin(C.WIN_MONTHS)].std()
    q = C.qbo50()
    d["QBO"] = ((q - q.mean()) / q.std()).reindex(d.index)
    d["ENSO"] = C.roni().reindex(d.index)
    for k in STRUCT:
        s = M[k].copy()
        if k == "aspect":
            s = np.log(s.where((s >= 1) & (s < 20)))
        d[k] = C.std_anom(s).reindex(d.index)
    x = 2 * np.pi * (d.index.dayofyear.values - 1) / 365.25
    for h in (1, 2):
        d[f"c{h}"], d[f"s{h}"] = np.cos(h * x), np.sin(h * x)
    d["winter"] = [C.winter_of(t) for t in d.index]
    ev = pd.DatetimeIndex(E.date)
    t = d.index.values
    pos = np.searchsorted(ev.values, t, side="right")
    nxt = np.where(pos < len(ev), ev.values[np.minimum(pos, len(ev) - 1)], np.datetime64("NaT"))
    dn = (nxt - t) / np.timedelta64(1, "D")
    for w, (a, b) in WINDOWS.items():
        d[f"y{w}"] = ((dn >= a) & (dn <= b)).astype(float)
    d["dnext"] = dn
    keep = d.index.month.isin(C.WIN_MONTHS) & (d.u > 0) & ~refr.values & (d.winter >= 1980) & (d.winter <= 2025)
    d = d[keep]
    # the last winter's outcome windows must be observable (u10 to 2026-06-26; events only Nov-Mar anyway)
    return d, E


SEAS = ["c1", "s1", "c2", "s2"]
MODELS = {
    "CLIM": SEAS,
    "QE": SEAS + ["QBO", "ENSO"],
    "QE+U": SEAS + ["QBO", "ENSO", "U"],
    "QE+T": SEAS + ["QBO", "ENSO", "T"],
    "QE+dT": SEAS + ["QBO", "ENSO", "dT"],
    "QE+HF": SEAS + ["QBO", "ENSO", "HF"],
    "QE+HF40": SEAS + ["QBO", "ENSO", "HF40"],
    "QE+U+T": SEAS + ["QBO", "ENSO", "U", "T"],
    "QE+U+dT": SEAS + ["QBO", "ENSO", "U", "dT"],
    "QE+U+HF": SEAS + ["QBO", "ENSO", "U", "HF"],
    "QE+U+HF40": SEAS + ["QBO", "ENSO", "U", "HF40"],
    "QE+U+T+dT+HF+HF40": SEAS + ["QBO", "ENSO", "U", "T", "dT", "HF", "HF40"],
}


def design(d, cols, inter=()):
    X = [np.ones(len(d))] + [d[c].values for c in cols]
    for a, b in inter:
        X.append(d[a].values * d[b].values)
    return np.column_stack(X)


def lowo(d, cols, ycol, inter=()):
    X = design(d, cols, inter)
    y = d[ycol].values
    w = d.winter.values
    p = np.full(len(d), np.nan)
    for wy in np.unique(w):
        tr = w != wy
        b = C.logit_fit(X[tr], y[tr], ridge=1e-3)
        p[~tr] = C.logit_pred(X[~tr], b)
    return p


def boot_delta(y, p_ref, p_mod, winters, rng):
    """Block bootstrap over winters: BS_ref - BS_mod (positive = model better), and AUC_mod - AUC_ref."""
    uw = np.unique(winters)
    idx = {k: np.where(winters == k)[0] for k in uw}
    se_r, se_m = (y - p_ref) ** 2, (y - p_mod) ** 2
    obs = se_r.mean() - se_m.mean()
    bs, ba = np.empty(NBOOT), np.empty(NBOOT)
    for i in range(NBOOT):
        pick = np.concatenate([idx[k] for k in rng.choice(uw, len(uw))])
        bs[i] = se_r[pick].mean() - se_m[pick].mean()
        ba[i] = C.auc(y[pick], p_mod[pick]) - C.auc(y[pick], p_ref[pick])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()); p = max(p, 1 / NBOOT)
    pa = 2 * min((ba <= 0).mean(), (ba >= 0).mean()); pa = max(pa, 1 / NBOOT)
    return float(obs), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float(p), float(pa)


def score(d, preds, ycol, ref, mod, rng):
    m = np.isfinite(preds[ref]) & np.isfinite(preds[mod])
    y = d[ycol].values[m]
    pr, pm = preds[ref][m], preds[mod][m]
    bsr, bsm = np.mean((y - pr) ** 2), np.mean((y - pm) ** 2)
    dlt, ci, p, pa = boot_delta(y, pr, pm, d.winter.values[m], rng)
    return {"bss": float(1 - bsm / bsr), "dBS": dlt, "ci": ci, "p": p, "auc": float(C.auc(y, pm)), "auc_ref": float(C.auc(y, pr)),
            "p_auc": pa, "n": int(m.sum())}


def main():
    rng = np.random.default_rng(20260930)
    d, E = build_frame()
    print(f"eligible days {len(d)}, winters {d.winter.nunique()}; "
          + ", ".join(f"{w}: {int(d[f'y{w}'].sum())} positive days" for w in WINDOWS), flush=True)
    out = {"n_days": int(len(d)), "n_winters": int(d.winter.nunique()), "windows": list(WINDOWS), "skill": {}, "coef": {},
           "struct": {}, "strong_subset": {}, "curves": {}}
    preds_all = {}
    # ---------------- Q3: physical predictors
    for w in WINDOWS:
        yc = f"y{w}"
        base_cols = sorted(set(sum(MODELS.values(), [])))
        dd = d.dropna(subset=base_cols)
        preds = {m: lowo(dd, cols, yc) for m, cols in MODELS.items()}
        for m in preds:
            preds_all[(w, m)] = pd.Series(preds[m], index=dd.index)
        out["skill"][w] = {}
        for m in MODELS:
            if m == "CLIM":
                continue
            rows = {"vsCLIM": score(dd, preds, yc, "CLIM", m, rng)}
            if m != "QE":
                rows["vsQE"] = score(dd, preds, yc, "QE", m, rng)
            if m.startswith("QE+U+"):
                rows["vsQEU"] = score(dd, preds, yc, "QE+U", m, rng)
            out["skill"][w][m] = rows
        # in-sample coefficients of the full model (per sd), for interpretation
        X = design(dd, MODELS["QE+U+T+dT+HF+HF40"])
        b = C.logit_fit(X, dd[yc].values, ridge=1e-3)
        out["coef"][w] = dict(zip(["const"] + MODELS["QE+U+T+dT+HF+HF40"], np.round(b, 3).tolist()))
        print(w, {m: (round(r["vsCLIM"]["bss"], 3), round(r["vsCLIM"]["auc"], 3), round(r["vsCLIM"]["p"], 3))
                  for m, r in out["skill"][w].items()}, flush=True)
    # FDR across the Q3 family: every (window, model, reference) test
    fam = [(w, m, ref) for w in WINDOWS for m in out["skill"][w] for ref in out["skill"][w][m]]
    sig = C.fdr_bh([out["skill"][w][m][ref]["p"] for w, m, ref in fam], 0.10)
    for (w, m, ref), s in zip(fam, sig):
        out["skill"][w][m][ref]["fdr10"] = bool(s)
    # ---------------- Q4: structure beyond strength
    for w in ("21-42", "1-14", "15-28", "43-63"):
        yc = f"y{w}"
        out["struct"][w] = {}
        for k in STRUCT:
            cols0 = SEAS + ["QBO", "ENSO", "U"]
            dd = d.dropna(subset=cols0 + [k])
            p0 = lowo(dd, cols0, yc)
            p1 = lowo(dd, cols0 + [k], yc)
            p2 = lowo(dd, cols0 + [k], yc, inter=[("U", k)])
            ps = lowo(dd, SEAS + ["QBO", "ENSO", k], yc)
            pc = lowo(dd, SEAS + ["QBO", "ENSO"], yc)
            preds = {"QE+U": p0, "QE+U+S": p1, "QE+UxS": p2, "QE+S": ps, "QE": pc}
            X = design(dd, cols0 + [k], inter=[("U", k)])
            b = C.logit_fit(X, dd[yc].values, ridge=1e-3)
            out["struct"][w][k] = {"S_vs_U": score(dd, preds, yc, "QE+U", "QE+U+S", rng),
                                   "UxS_vs_U": score(dd, preds, yc, "QE+U", "QE+UxS", rng),
                                   "S_alone_vs_QE": score(dd, preds, yc, "QE", "QE+S", rng),
                                   "coef_S": float(b[-2]), "coef_UxS": float(b[-1]), "coef_U": float(b[-3])}
        print(w, "structure:", {k: (round(v["S_vs_U"]["bss"], 3), round(v["S_vs_U"]["p"], 3), round(v["UxS_vs_U"]["p"], 3))
                                for k, v in out["struct"][w].items()}, flush=True)
    fam = [(w, k, t) for w in out["struct"] for k in out["struct"][w] for t in ("S_vs_U", "UxS_vs_U", "S_alone_vs_QE")]
    sig = C.fdr_bh([out["struct"][w][k][t]["p"] for w, k, t in fam], 0.10)
    for (w, k, t), s in zip(fam, sig):
        out["struct"][w][k][t]["fdr10"] = bool(s)
    # ---------------- strong-vortex subset: does structure separate fragile strong vortices? (21-42 d)
    sv = d[d.U > 1.0]
    out["strong_subset"]["n_days"] = int(len(sv)); out["strong_subset"]["n_winters"] = int(sv.winter.nunique())
    out["strong_subset"]["n_pos"] = int(sv["y21-42"].sum())
    out["strong_subset"]["pos_winters"] = sorted(int(x) for x in sv[sv["y21-42"] == 1].winter.unique())
    res = {}
    for k in STRUCT + ["U", "T", "dT", "HF", "HF40", "QBO", "ENSO"]:
        s = sv.dropna(subset=[k])
        y = s["y21-42"].values; x = s[k].values
        a = C.auc(y, x)
        uw = s.winter.unique(); idx = {q: np.where(s.winter.values == q)[0] for q in uw}
        bs = []
        for _ in range(NBOOT):
            pick = np.concatenate([idx[q] for q in rng.choice(uw, len(uw))])
            bs.append(C.auc(y[pick], x[pick]))
        bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
        p = 2 * min((bs <= 0.5).mean(), (bs >= 0.5).mean()); p = max(p, 1 / NBOOT)
        res[k] = {"auc": float(a), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(p), "n": int(len(s))}
    sig = C.fdr_bh([r["p"] for r in res.values()], 0.10)
    for (k, r), s in zip(res.items(), sig):
        r["fdr10"] = bool(s)
    out["strong_subset"]["auc"] = res
    print("strong-vortex subset AUC:", {k: (round(v["auc"], 2), round(v["p"], 3), v["fdr10"]) for k, v in res.items()})
    # ---------------- forward-probability curves: empirical, binned, winter-bootstrap CI
    for k in ("U", "T", "dT", "HF", "HF40", "QBO", "ENSO", "bt_neg_upper", "qy_edge", "aspect", "cen_lat", "depth_ratio", "jet_lat"):
        bins = np.array([-9, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 9]) if k != "ENSO" else np.array([-9, -1, -0.5, 0.5, 1, 1.5, 9])
        cv = {}
        for w in WINDOWS:
            s = d.dropna(subset=[k])
            cat = np.digitize(s[k].values, bins) - 1
            rows = []
            for j in range(len(bins) - 1):
                m = cat == j
                if m.sum() < 30:
                    rows.append(None); continue
                y = s[f"y{w}"].values[m]; wv = s.winter.values[m]
                uw = np.unique(wv); idx = {q: np.where(wv == q)[0] for q in uw}
                bs = [y[np.concatenate([idx[q] for q in rng.choice(uw, len(uw))])].mean() for _ in range(1000)]
                rows.append({"p": float(y.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                             "n": int(m.sum()), "nw": int(len(uw))})
            cv[w] = rows
        out["curves"][k] = {"bins": bins.tolist(), "rows": cv, "base": {w: float(d[f"y{w}"].mean()) for w in WINDOWS}}
    (C.DATA / "s2_forward.json").write_text(json.dumps(out, indent=1))
    pd.to_pickle({"frame": d, "preds": preds_all}, C.DATA / "s2_preds.pkl")


if __name__ == "__main__":
    main()
