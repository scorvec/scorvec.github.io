#!/usr/bin/env python3
"""Does GEFS rain at days 10-35 beat climatology over the Colombia basins?

The fusion is only worth building if the subseasonal rain carries
information.  Tropical precipitation skill collapses somewhere past two
weeks, and if GEFS days 16-35 cannot beat climatology then the honest
fusion is exactly what the current handover already does: relax toward
climatology and let the monthly state model take over.

This pairs archived GEFS ensemble-mean basin rain against gauge-corrected
IMERG for the same day, by lead band, and reports:
  raw and bias-corrected MAE and correlation
  skill against climatology (predicting the seasonal norm every day)
  the fitted multiplicative bias factor per band

Bias factors are regularised ratios, the same form the AIFS/IFS chain
uses: F = (sum obs + P*r0) / (sum fcst + P), so a band with few pairs
stays near its prior instead of chasing noise.

    python scripts/sst/gefs_verify.py

Output: ~/colombia_hydro/out/gefs_verify.json
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import national_inflow as NI                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "gefs_rain"
OUT = PRIV / "out" / "gefs_verify.json"
ORDER = NI.ORDER
BANDS = [(10, 15), (16, 22), (23, 35)]
K_PRIOR = 8.0          # days of climatology in the prior
PRIOR_R0 = 0.6         # GEFS runs wet over these basins; start there


def _anom_r(sub):
    """Anomaly correlation, computed WITHIN each basin.

    Pooling first and subtracting a single mean is wrong and gives a
    spuriously NEGATIVE answer: basins differ in mean rainfall by a factor
    of two, so a pooled "forecast anomaly" is mostly basin identity while
    the observed anomaly is against a basin- and day-specific climatology.
    The first version of this function made exactly that error and
    reported -0.16 to -0.21.
    """
    fa, oa = [], []
    for b in sorted({r["basin"] for r in sub}):
        ss = [r for r in sub if r["basin"] == b]
        if len(ss) < 20:
            continue
        f_ = np.array([r["fcst"] for r in ss])
        fa += list(f_ - f_.mean())
        oa += [r["obs"] - r["clim"] for r in ss]
    if len(fa) < 40:
        return None
    return round(float(np.corrcoef(fa, oa)[0, 1]), 3)


def calibrate(rows, bi, cycles):
    """How much weight does GEFS deserve at this lead band?

    Rather than eyeballing an anomaly correlation, fit the weight and let
    it decide:  obs_anom = a + b * fcst_anom,  leave-one-CYCLE-out (not
    leave-one-day-out - days inside a cycle share a forecast and are not
    independent).  b is the shrinkage the data supports: b -> 0 means the
    honest use of this band is to ignore it and fall back on climatology.
    Reported with the out-of-sample MSE skill the calibrated forecast
    achieves against climatology, which is the number that matters.
    """
    sub = [r for r in rows if r["band"] == bi]
    if len(sub) < 60:
        return None
    # per-basin anomalies, forecast against its own basin mean
    fmean = {}
    for b in {r["basin"] for r in sub}:
        v = [r["fcst"] for r in sub if r["basin"] == b]
        fmean[b] = float(np.mean(v))
    fa = np.array([r["fcst"] - fmean[r["basin"]] for r in sub])
    oa = np.array([r["obs"] - r["clim"] for r in sub])
    cy = np.array([r["cycle"] for r in sub])
    pred = np.full(len(oa), np.nan)
    bs = []
    for c in sorted(set(cy)):
        te, tr = cy == c, cy != c
        if tr.sum() < 40:
            continue
        A = np.column_stack([np.ones(tr.sum()), fa[tr]])
        beta, *_ = np.linalg.lstsq(A, oa[tr], rcond=None)
        pred[te] = beta[0] + beta[1] * fa[te]
        bs.append(float(beta[1]))
    # INTERCEPT-ONLY control, same folds.  Without it the slope gets credit
    # for a plain climatology offset: if the verification window ran drier
    # than the harmonic climatology, fitting a constant beats zero-anomaly
    # and looks like forecast skill.  The number that matters is the
    # INCREMENTAL skill the slope adds over that control.
    pred0 = np.full(len(oa), np.nan)
    for c in sorted(set(cy)):
        te, tr = cy == c, cy != c
        if tr.sum() < 40:
            continue
        pred0[te] = float(np.mean(oa[tr]))
    ok = np.isfinite(pred) & np.isfinite(pred0)
    if ok.sum() < 40 or not bs:
        return None
    mse = float(np.mean((pred[ok] - oa[ok]) ** 2))
    mse0 = float(np.mean((pred0[ok] - oa[ok]) ** 2))
    mse_cl = float(np.mean(oa[ok] ** 2))          # climatology = zero anomaly
    inc = 1 - mse / mse0
    return {"n": int(ok.sum()), "cycles": len(set(cy)),
            "b_mean": round(float(np.mean(bs)), 3),
            "b_sd_across_cycles": round(float(np.std(bs)), 3),
            "mse_skill_vs_clim": round(1 - mse / mse_cl, 4),
            "mse_skill_intercept_only": round(1 - mse0 / mse_cl, 4),
            "incremental_skill_from_gefs": round(inc, 4),
            "verdict": ("usable" if inc > 0.01 and np.mean(bs) > 0
                        else "no usable signal beyond a constant offset")}


def main() -> int:
    d = PR.load_all()
    dates = np.array([str(x) for x in d["dates"]])
    obs = {b: dict(zip(dates, d["rain_abs"][b])) for b in ORDER}
    clim = {b: dict(zip(dates, d["clim"][b])) for b in ORDER}

    rows = []
    for f in sorted(glob.glob(str(ARCH / "*_mean.json.gz"))):
        rec = json.load(gzip.open(f, "rt"))
        init = datetime.strptime(rec["init_date"], "%Y%m%d")
        for day, bym in rec["days"].items():
            if "geavg" not in bym:
                continue
            lead = (datetime.strptime(day, "%Y-%m-%d") - init).days
            band = next((i for i, (a, b) in enumerate(BANDS) if a <= lead <= b), None)
            if band is None:
                continue
            for b in ORDER:
                o = obs[b].get(day)
                if o is None or not np.isfinite(o):
                    continue
                rows.append({"cycle": rec["init_date"],
                             "basin": b, "lead": lead, "band": band,
                             "fcst": float(bym["geavg"][b]), "obs": float(o),
                             "clim": float(clim[b].get(day, np.nan))})
    if not rows:
        print("no matured GEFS pairs yet"); return 1
    print(f"{len(rows)} matured pairs from "
          f"{len(set(r['basin'] for r in rows))} basins", flush=True)

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "pairs": len(rows), "bands": [f"d{a}-{b}" for a, b in BANDS],
           "by_band": {}, "by_basin_band": {}}
    for bi, (a, b) in enumerate(BANDS):
        sub = [r for r in rows if r["band"] == bi]
        if len(sub) < 30:
            continue
        F = ((sum(r["obs"] for r in sub) + K_PRIOR * PRIOR_R0
              * np.mean([r["clim"] for r in sub]) * len(ORDER))
             / (sum(r["fcst"] for r in sub) + K_PRIOR
                * np.mean([r["clim"] for r in sub]) * len(ORDER)))
        f_ = np.array([r["fcst"] for r in sub])
        o_ = np.array([r["obs"] for r in sub])
        c_ = np.array([r["clim"] for r in sub])
        cor = f_ * F
        sk = lambda p: 1 - float(np.mean(np.abs(p - o_))) / float(
            np.mean(np.abs(c_ - o_)))
        out["by_band"][f"d{a}-{b}"] = {
            "n": len(sub), "bias_factor": round(float(F), 3),
            "r_raw": round(float(np.corrcoef(f_, o_)[0, 1]), 3),
            "mae_raw": round(float(np.mean(np.abs(f_ - o_))), 2),
            "mae_corrected": round(float(np.mean(np.abs(cor - o_))), 2),
            "mae_climatology": round(float(np.mean(np.abs(c_ - o_))), 2),
            "skill_vs_clim_corrected": round(sk(cor), 3),
            "anomaly_r": _anom_r(sub)}
    print(f"\n{'band':9}{'n':>6}{'F':>7}{'r':>7}{'anom r':>8}{'MAE raw':>9}"
          f"{'MAE corr':>10}{'MAE clim':>10}{'skill':>8}")
    for k, v in out["by_band"].items():
        print(f"{k:9}{v['n']:6d}{v['bias_factor']:7.2f}{v['r_raw']:7.3f}"
              f"{v['anomaly_r']:8.3f}{v['mae_raw']:9.2f}{v['mae_corrected']:10.2f}"
              f"{v['mae_climatology']:10.2f}{v['skill_vs_clim_corrected']:+8.3f}")
    for bb in ORDER:
        for bi, (a, b) in enumerate(BANDS):
            sub = [r for r in rows if r["band"] == bi and r["basin"] == bb]
            if len(sub) < 20:
                continue
            f_ = np.array([r["fcst"] for r in sub])
            o_ = np.array([r["obs"] for r in sub])
            c_ = np.array([r["clim"] for r in sub])
            out["by_basin_band"].setdefault(bb, {})[f"d{a}-{b}"] = {
                "n": len(sub),
                "anomaly_r": _anom_r(sub)}
    out["calibration"] = {}
    print(f"\n{'band':9}{'cyc':>5}{'n':>7}{'b':>8}{'sd(b)':>8}"
          f"{'skill':>9}{'intercept':>11}{'GEFS adds':>11}   verdict")
    for bi, (a, b) in enumerate(BANDS):
        c = calibrate(rows, bi, None)
        if not c:
            continue
        out["calibration"][f"d{a}-{b}"] = c
        print(f"d{a}-{b:<7}{c['cycles']:5d}{c['n']:7d}{c['b_mean']:8.3f}"
              f"{c['b_sd_across_cycles']:8.3f}{c['mse_skill_vs_clim']:+9.4f}"
              f"{c['mse_skill_intercept_only']:+11.4f}"
              f"{c['incremental_skill_from_gefs']:+11.4f}   {c['verdict']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
