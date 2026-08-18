#!/usr/bin/env python3
"""Seasonal skill for MONTHLY MEAN NATIONAL INFLOW — purely probabilistic.

Target: the monthly mean of national inflow, % of the fleet-corrected
norm.  Forecast: a full predictive distribution per (init, lead month),
not a point.  Scored with CRPS against a climatological distribution,
plus PIT for calibration and the usual deterministic numbers for the
ensemble mean.

Four competing forecasts, all fitted identically and scored identically:
  CLIM   the climatological distribution of that calendar month
  STAT   ENSO (ONI at init) + storage + antecedent 30/90-day inflow
  C3S    the C3S seasonal rainfall ensemble alone
  BOTH   STAT + C3S

C3S RAIN IS USED AS A RATIO, never as an absolute.  A seasonal model's
precipitation climatology differs from observed by tens of percent and
drifts with lead, so each member is divided by that system's own
hindcast ensemble climatology for the same (target month, lead).  Model
drift cancels; only the anomaly signal crosses over.  Every member is
pushed through the fitted equation separately, so the ensemble spread
becomes forecast spread rather than being averaged away first.

Validation is leave-one-year-out with a +/-1 YEAR EMBARGO: an ENSO event
spans two calendar years, so a plain LOYO would leave half of it in
training.  The C3S hindcast covers 1993-2016 and inflow starts in 2000,
so the common period is 2000-2016 - 17 years, ~200 monthly cases per
lead.  That is the honest sample and it is small; treat lead-by-lead
differences of a few hundredths as noise.

    python scripts/sst/seasonal_inflow.py

Outputs: ~/colombia_hydro/out/seasonal_inflow.json
         ~/colombia_hydro/site/seasonal_inflow.webp
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
import national_inflow as NI                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
C3S = PRIV / "out" / "c3s_basin_precip.json"
OUT_JSON = PRIV / "out" / "seasonal_inflow.json"
OUT_PNG = PRIV / "site" / "seasonal_inflow.webp"
ORDER = NI.ORDER
MAXLEAD = 6


def c3s_national():
    """{(init, lead): [national rain mm/day per member]} on energy weights."""
    d = json.loads(C3S.read_text())["data"]
    W = NI.basin_energy_weights()
    out = {}
    for init, byl in d.items():
        for lead, byb in byl.items():
            n = min(len(byb[b]) for b in ORDER if b in byb)
            if n < 5:
                continue
            v = np.zeros(n)
            for b in ORDER:
                v += np.asarray(byb[b][:n], float) * W[b]
            out[(init, int(lead))] = v
    return out


def crps_ens(members, obs):
    m = np.sort(np.asarray(members, float))
    n = len(m)
    i = np.arange(n)
    return float(np.mean(np.abs(m - obs)) - np.sum((2 * i - n + 1) * m) / (n * n))


def build(d, c3s):
    """One row per (init, lead) with target and every predictor."""
    rows = NI.monthly_frame(d)
    keys = sorted(rows)
    recs = []
    for (init, lead), mem in c3s.items():
        im = init[:7]
        if im not in rows:
            continue
        a = int(init[:4]) * 12 + int(init[5:7]) + lead
        ty, tm = (a - 1) // 12, (a - 1) % 12 + 1
        tgt = f"{ty:04d}-{tm:02d}"
        if tgt not in rows:
            continue
        r = rows[im]
        recs.append({"init": im, "lead": lead, "target": tgt, "tmonth": tm,
                     "y": rows[tgt]["pct"], "oni": r["oni"],
                     "stor": r["stor_end"], "ant30": r["ant30"],
                     "ant90": r["ant90"], "mem": mem})
    return recs


def design(recs, kind, ratio_by):
    """(X per member, y, year) — X has one row per member."""
    X, Y, YR, NM = [], [], [], []
    for r in recs:
        key = (r["tmonth"], r["lead"])
        base = ratio_by.get(key)
        if base is None or base <= 0:
            continue
        lr = np.log(np.maximum(np.asarray(r["mem"], float) / base, 0.05))
        s1 = np.sin(2 * np.pi * r["tmonth"] / 12)
        c1 = np.cos(2 * np.pi * r["tmonth"] / 12)
        stat = [r["oni"], r["oni"] * s1, r["oni"] * c1, r["stor"],
                r["ant30"] - 100.0, r["ant90"] - 100.0, s1, c1]
        for v in lr:
            if kind == "STAT":
                X.append(stat)
            elif kind == "C3S":
                X.append([v, v * s1, v * c1, s1, c1])
            else:
                X.append([v, v * s1, v * c1] + stat)
        Y.append(r["y"]); YR.append(int(r["target"][:4])); NM.append(len(lr))
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(YR), np.asarray(NM)


def run(recs, kind, ratio_by, embargo=1):
    X, Y, YR, NM = design(recs, kind, ratio_by)
    if not len(Y):
        return None
    off = np.concatenate([[0], np.cumsum(NM)])
    ylog = np.log(np.clip(Y, 5, None))
    ymem = np.repeat(ylog, NM)
    yr_mem = np.repeat(YR, NM)
    pred = np.full(len(ymem), np.nan)
    for y in sorted(set(YR)):
        te = yr_mem == y
        tr = np.abs(yr_mem - y) > embargo
        if tr.sum() < 200 or te.sum() == 0:
            continue
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A, ymem[tr], rcond=None)
        pred[te] = b[0] + X[te] @ b[1:]
        res = ymem[tr] - (b[0] + X[tr] @ b[1:])
        sd = float(np.std(res))
        pred[te] = pred[te]                       # spread added below
    ok = np.isfinite(pred)
    if ok.sum() < 100:
        return None
    # residual sd from a pooled LOYO pass, used to widen each member
    sd = float(np.nanstd(ymem[ok] - pred[ok]))
    out = {"n_cases": int(len(Y)), "resid_sd_log": round(sd, 4), "by_lead": {}}
    rng = np.random.default_rng(7)
    for lead in range(1, MAXLEAD + 1):
        P, O, C, CC = [], [], [], []
        for i, r in enumerate(recs):
            if r["lead"] != lead:
                continue
            sl = slice(off[i], off[i + 1])
            p = pred[sl]
            if not np.isfinite(p).all():
                continue
            ens = np.exp(p[:, None] + rng.normal(0, sd, (len(p), 12))).ravel()
            O.append(Y[i]); P.append(float(np.mean(ens)))
            C.append(crps_ens(ens, Y[i]))
            CC.append(float(np.mean(ens <= Y[i])))       # PIT of the observation
        if len(O) < 30:
            continue
        P, O = np.asarray(P), np.asarray(O)
        clim = np.array([y for y in Y])
        crps_cl = float(np.mean([crps_ens(clim, o) for o in O]))
        rm = float(np.sqrt(np.mean((P - O) ** 2)))
        rc = float(np.sqrt(np.mean((O - O.mean()) ** 2)))
        pit = np.asarray(CC, float)
        # a calibrated forecast has PIT ~ Uniform(0,1): mean 0.5, and the
        # 10-bin histogram flat.  A U-shape means under-dispersed (too
        # narrow), a hump means over-dispersed.
        hist = np.histogram(pit, bins=10, range=(0, 1))[0]
        out["by_lead"][lead] = {
            "pit_mean": round(float(np.mean(pit)), 3),
            "pit_hist": hist.tolist(),
            "pit_flatness": round(float(np.std(hist / max(hist.sum(), 1)) / 0.1), 3),
            "coverage_10_90": round(float(np.mean((pit > 0.1) & (pit < 0.9))), 3),
            "n": int(len(O)), "r": round(float(np.corrcoef(P, O)[0, 1]), 3),
            "rmse": round(rm, 2), "rmse_clim": round(rc, 2),
            "skill_rmse": round(1 - rm / rc, 3),
            "crps": round(float(np.mean(C)), 2),
            "crps_clim": round(crps_cl, 2),
            "crps_skill": round(1 - float(np.mean(C)) / crps_cl, 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embargo", type=int, default=1)
    a = ap.parse_args()
    d = NI.load_national()
    c3s = c3s_national()
    recs = build(d, c3s)
    yrs = sorted({int(r["target"][:4]) for r in recs})
    print(f"C3S x inflow overlap: {yrs[0]}..{yrs[-1]} ({len(yrs)} yr), "
          f"{len(recs)} (init,lead) cases, "
          f"{len(recs[0]['mem'])} members", flush=True)

    # model climatology per (target month, lead) — the bias-correction base
    ratio_by = {}
    acc = {}
    for r in recs:
        acc.setdefault((r["tmonth"], r["lead"]), []).extend(r["mem"])
    for k, v in acc.items():
        ratio_by[k] = float(np.mean(v))

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "target": "monthly mean NATIONAL inflow, % of fleet-corrected norm",
           "validation": f"leave-one-year-out, +/-{a.embargo} year embargo",
           "period": f"{yrs[0]}..{yrs[-1]}", "models": {}}
    print(f"\n{'model':6}{'lead':>5}{'n':>5}{'r':>7}{'RMSE':>8}{'clim':>8}"
          f"{'skill':>8}{'CRPS':>8}{'clim':>8}{'CRPSss':>8}")
    for kind in ("STAT", "C3S", "BOTH"):
        R = run(recs, kind, ratio_by, a.embargo)
        if not R:
            print(f"{kind}: insufficient data"); continue
        out["models"][kind] = R
        for lead, v in R["by_lead"].items():
            print(f"{kind:6}{lead:5d}{v['n']:5d}{v['r']:7.3f}{v['rmse']:8.2f}"
                  f"{v['rmse_clim']:8.2f}{v['skill_rmse']:+8.3f}{v['crps']:8.2f}"
                  f"{v['crps_clim']:8.2f}{v['crps_skill']:+8.3f}", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    print(f"wrote {OUT_JSON}")
    return 0


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(13.8, 6.6))
    hd = fig.add_axes([0, 0.925, 1, 0.075]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.014, 0.62, "SEASONAL SKILL — MONTHLY MEAN NATIONAL INFLOW",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.014, 0.2, f"probabilistic, scored by CRPS against a climatological "
            f"distribution · {out['period']} · {out['validation']}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    cols = {"STAT": "#1f4e8c", "C3S": "#e08214", "BOTH": "#1f7a4d"}
    lbl = {"STAT": "ENSO + storage + antecedent", "C3S": "C3S rainfall ensemble",
           "BOTH": "both"}
    for i, (key, ttl) in enumerate((("crps_skill", "CRPS skill vs climatology"),
                                    ("skill_rmse", "RMSE skill of the ensemble mean"))):
        ax = fig.add_axes([0.07 + i * 0.49, 0.12, 0.40, 0.72])
        for m, R in out["models"].items():
            L = sorted(int(k) for k in R["by_lead"])
            v = [R["by_lead"][str(k)][key] if str(k) in R["by_lead"]
                 else R["by_lead"][k][key] for k in L]
            ax.plot(L, v, color=cols[m], lw=2.1, marker="o", ms=5, label=lbl[m])
        ax.axhline(0, color="0.4", lw=1.1)
        ax.set_xlabel("lead, months", fontsize=9.5)
        ax.set_ylabel(ttl, fontsize=9.5)
        ax.set_title(ttl, fontsize=11, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8.5)
        if i == 0:
            ax.legend(fontsize=8.5)
    fig.text(0.07, 0.02, "zero = no better than climatology · every member pushed "
             "through the fitted equation separately, so ensemble spread becomes "
             "forecast spread · C3S rain enters as a ratio to its own hindcast "
             "climatology, so model drift cancels",
             fontsize=7.8, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
