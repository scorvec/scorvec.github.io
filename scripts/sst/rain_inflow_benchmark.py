#!/usr/bin/env python3
"""How well can we reproduce ONS ENA from rainfall? Controlled benchmark.

Same inputs for every candidate: gauge-corrected IMERG basin rain (daily),
Thornthwaite PET, EAR anomaly (measured state, lagged 1 d). Target: ONS
daily ENA (MWmed) per SIN basin. Two out-of-sample folds:
  F1 = last 120 days (current dry season 2026)
  F2 = the 2025/26 wet season (Nov 15 - Mar 15), trained on the rest
Metrics: NSE (daily), NSE of 7-day means, r, percent bias.

Candidates:
  kernel_v3   fast EMA(tau scan)+slow EMA180 (+lag) + EAR anom  [regression]
  cascade     NNLS on EMA(3,7,15,30,60,120,180) of RAW rain + season + EAR
  smap_nse    daily SMAP, DE on NSE, closed-form gain
  smap_kge    daily SMAP, DE on KGE (Gupta) with log-flow term (low-flow aware)
  hybrid      NNLS on [smap_kge runoff, kernel features, EAR] (stacked)
  persistence 7-day-mean-of-last-week benchmark (what "no rain info" gives)

Outputs: ~/brazil_hydro/out/rain_inflow_benchmark.json + site/benchmark.webp
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_model import MAJORS                                     # noqa: E402
from smap_ons import smap_run, thornthwaite_pet, basin_centroid, basin_tmean_monthly  # noqa: E402
from rain_inflow_model import ema                                   # noqa: E402

PRIV = Path.home() / "brazil_hydro"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
ENA = PRIV / "raw" / "ena_bacia_daily.json.gz"
EAR = PRIV / "raw" / "ear_bacia_daily.json.gz"
OUT_JSON = PRIV / "out" / "rain_inflow_benchmark.json"
OUT_PNG = PRIV / "site" / "benchmark.webp"
TAUS = [4, 7, 12, 20, 30, 45, 60, 90, 120]
CAS = [3, 7, 15, 30, 60, 120, 180]


def nse(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    if m.sum() < 10:
        return np.nan
    return 1 - np.sum((s[m] - o[m]) ** 2) / max(np.sum((o[m] - o[m].mean()) ** 2), 1e-9)


def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    r = np.corrcoef(s[m], o[m])[0, 1]
    a = s[m].std() / max(o[m].std(), 1e-9)
    b = s[m].mean() / max(o[m].mean(), 1e-9)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def week_mean(x):
    k = np.ones(7) / 7
    y = np.convolve(np.where(np.isfinite(x), x, np.nan), k, "same")
    return y


def metrics(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    return {"nse": round(float(nse(s, o)), 3),
            "nse7": round(float(nse(week_mean(s), week_mean(o))), 3),
            "r": round(float(np.corrcoef(s[m], o[m])[0, 1]), 3),
            "pbias": round(float(100 * (s[m].mean() / o[m].mean() - 1)), 1)}


def nnls_fit(X, y, m):
    from scipy.optimize import nnls
    A = np.column_stack([np.ones(m.sum()), X[m]])
    # allow negative intercept/season by splitting sign columns for those
    coef, _ = nnls(A, y[m])
    return coef


def ear_anom(b, dates):
    with gzip.open(EAR, "rt") as f:
        ear = json.load(f)
    d = ear.get(b, {})
    sd = sorted(d)
    sv = np.array([d[k] for k in sd], float)
    sdoy = np.array([min(datetime.strptime(k, "%Y-%m-%d").timetuple().tm_yday, 365) for k in sd])
    norm = np.full(365, np.nan)
    for dd in range(1, 366):
        dist = np.minimum(np.abs(sdoy - dd), 365 - np.abs(sdoy - dd))
        mm = dist <= 10
        if mm.sum() > 40:
            norm[dd - 1] = np.median(sv[mm])
    an = dict(zip(sd, sv - norm[sdoy - 1]))
    out = np.array([an.get(dt.strftime("%Y-%m-%d"), np.nan) for dt in dates])
    out = np.roll(out, 1); out[0] = np.nan
    for i in range(1, len(out)):
        if not np.isfinite(out[i]): out[i] = out[i - 1]
    return np.nan_to_num(out)


def smap_calibrate(P, Ep, y, train, objective):
    from scipy.optimize import differential_evolution
    warm = 200
    def gain_for(Qmm):
        s = Qmm[train]; o = y[train]; mm = np.isfinite(o) & (np.arange(len(y))[train] >= warm)
        return float(np.dot(s[mm], o[mm]) / max(np.dot(s[mm], s[mm]), 1e-9))
    def loss(p6):
        Qmm, _ = smap_run(P, Ep, tuple(p6) + (1.0,))
        g = gain_for(Qmm)
        s = Qmm * g
        sel = train & (np.arange(len(y)) >= warm)
        if objective == "nse":
            return -nse(s[sel], y[sel])
        # KGE on flows + KGE on log flows (low-flow aware)
        return -(0.5 * kge(s[sel], y[sel]) + 0.5 * kge(np.log1p(s[sel]), np.log1p(y[sel])))
    bounds = [(200, 4000), (0.5, 15), (0.1, 60), (0.0, 20), (10, 70), (30, 400)]
    res = differential_evolution(loss, bounds, seed=2, maxiter=60, popsize=12, tol=1e-5,
                                 polish=True, workers=1)
    Qmm, _ = smap_run(P, Ep, tuple(res.x) + (1.0,))
    g = gain_for(Qmm)
    return Qmm * g, np.concatenate([res.x, [g]])


def main() -> int:
    a = sys.argv[1:]
    basins = a[a.index("--basins") + 1:] if "--basins" in a else MAJORS
    tc = json.loads(TRUTH.read_text())
    dates = [datetime.strptime(d, "%Y%m%d") for d in tc["dates"]]
    n = len(dates)
    with gzip.open(ENA, "rt") as f:
        ena_all = json.load(f)
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
    th = 2 * np.pi * doy / 365
    season = np.column_stack([np.sin(th), np.cos(th)])
    idx = np.arange(n)
    f1 = idx >= n - 120
    w0 = next(i for i, d in enumerate(dates) if d >= datetime(2025, 11, 15))
    w1 = next(i for i, d in enumerate(dates) if d >= datetime(2026, 3, 15))
    f2 = (idx >= w0) & (idx < w1)
    folds = {"F1_dry2026": f1, "F2_wet2025_26": f2}
    out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"basins": {}}
    for b in basins:
        if b not in tc or b not in ena_all:
            continue
        P = np.nan_to_num(np.array(tc[b], float))
        y = np.array([ena_all[b].get(d.strftime("%Y-%m-%d"), [np.nan])[0] for d in dates], float)
        lat, lon = basin_centroid(b)
        Ep = thornthwaite_pet(dates, basin_tmean_monthly(lat, lon), lat)
        S = ear_anom(b, dates)
        res_b = {}
        for fname, test in folds.items():
            train = ~test & (idx >= 60)
            R = {}
            # persistence: last-week mean at test start held flat (naive)
            per = np.full(n, np.nan)
            t0 = np.where(test)[0][0]
            per[test] = np.nanmean(y[max(0, t0 - 7):t0])
            R["persistence"] = metrics(per[test], y[test])
            # kernel_v3
            best = None
            for tau in TAUS:
                k = ema(P - np.nanmean(P), tau)
                ks = ema(P - np.nanmean(P), 180)
                for lag in range(0, 6):
                    kl = np.roll(k, lag); kl[:lag] = np.nan
                    ksl = np.roll(ks, lag); ksl[:lag] = np.nan
                    X = np.column_stack([kl, ksl, S, season])
                    m = train & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                    A = np.column_stack([np.ones(m.sum()), X[m]])
                    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                    fit_tr = A @ beta
                    sc = nse(fit_tr, y[m])
                    if best is None or sc > best[0]:
                        best = (sc, tau, lag, beta, X)
            sc, tau, lag, beta, X = best
            pred = beta[0] + X @ beta[1:]
            R["kernel_v3"] = {**metrics(pred[test], y[test]), "tau": tau, "lag": lag}
            # cascade (NNLS on EMAs of raw rain) + season + EAR (signed via +/- cols)
            E = np.column_stack([ema(P, t) for t in CAS])
            Xc = np.column_stack([E, season, -season, S, -S])
            m = train & np.isfinite(y)
            coef = nnls_fit(Xc, y, m)
            predc = coef[0] + Xc @ coef[1:]
            R["cascade"] = metrics(predc[test], y[test])
            # SMAP nse / kge
            smap_out = {}
            for obj in ("nse", "kge"):
                q, prm = smap_calibrate(P, Ep, y, train, obj)
                smap_out[obj] = q
                R[f"smap_{obj}"] = {**metrics(q[test], y[test]),
                                    "params": [round(float(v), 2) for v in prm]}
            # hybrid: NNLS on smap_kge runoff + cascade features
            Xh = np.column_stack([smap_out["kge"], E, season, -season, S, -S])
            coef = nnls_fit(Xh, y, m)
            predh = coef[0] + Xh @ coef[1:]
            R["hybrid"] = metrics(predh[test], y[test])
            res_b[fname] = R
            print(f"  {b:16} {fname:14} " + " ".join(
                f"{k}:{v['nse7']:+.2f}" for k, v in R.items()), flush=True)
        out["basins"][b] = res_b
        OUT_JSON.write_text(json.dumps(out, indent=1))
    out["generated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out["note"] = ("target ONS daily ENA MWmed; inputs corrected IMERG basin rain, "
                   "Thornthwaite PET, EAR anom(-1d); folds out-of-sample; nse7 = NSE of "
                   "7-day means (the weekly-deck-relevant score)")
    OUT_JSON.write_text(json.dumps(out, indent=1))

    # ── figure: nse7 by model, per basin, both folds
    models = ["persistence", "kernel_v3", "cascade", "smap_nse", "smap_kge", "hybrid"]
    cols = ["#9e9e9e", "#1f4e8c", "#5b87c0", "#e08214", "#c62828", "#2e9e4f"]
    bs = [b for b in MAJORS if b in out["basins"]]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, fname in zip(axes, folds):
        x = np.arange(len(bs)); wd = 0.13
        for k, (mname, col) in enumerate(zip(models, cols)):
            vals = [out["basins"][b][fname][mname]["nse7"] for b in bs]
            ax.bar(x + (k - 2.5) * wd, np.clip(vals, -0.5, 1), wd, color=col, label=mname)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylim(-0.5, 1.02)
        ax.set_ylabel("NSE of 7-day mean ENA (out-of-sample)", fontsize=8.5)
        ax.set_title(f"held-out {fname}", fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(lw=0.25, alpha=0.5, axis="y"); ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, ncol=6, loc="lower left")
    axes[1].set_xticks(np.arange(len(bs))); axes[1].set_xticklabels(bs, rotation=25, fontsize=8.5)
    fig.suptitle("Reproducing ONS ENA from rainfall — candidate rain→inflow models, out-of-sample "
                 "(bars clipped at −0.5)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=115); plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
