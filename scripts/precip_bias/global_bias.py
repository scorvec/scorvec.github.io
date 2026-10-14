#!/usr/bin/env python3
"""Rain-bias correction for the tropical rainfall forecast (2026-10-14, user: "global bias-adjusted rainfall/rainfall anomaly
plots from the aifs/ifs ensembles", tropics section, "use your best judgement").

For each ensemble and lead week, a multiplicative correction per 2° box:  R = sum(w * truth) / sum(w * forecast)
  forecast  ensemble mean (members 1-6) of the week's rain from the global archive (ecmwf_tp.py --domain global), runs of
            the CURRENT model cycle only (build_bias.VERSIONS: user rule, never pool across cycles);
  truth     IMERG Late over the same 7 days, scaled to IMERG Final by its seasonal ratio (build_bias.truth_terms, from the
            days Late and Final share) - so the corrected forecast speaks the same language as the GPCP climatology;
  w         a season kernel: Gaussian in day-of-year distance between each run's verifying week and the target week
            (sd 30 days), so a correction for late October leans on August-October and not on May;
  uncertainty  block bootstrap over runs (21-day blocks, as build_bias), plus the Late-to-Final ratio's own uncertainty.
A ratio is applied only where it is significantly different from 1 (BH-FDR q 0.10 over the map), so the published
"correction applied" map is exactly the significant bias; and there only as far as it is distinguishable from 1: log R is
shrunk by max(0, 1 - (se / |log R|)^2) (at 2 se applied at 3/4 strength, at 4 se at 15/16), clipped to [1/3, 3]. Boxes
where the weighted forecast or truth averages under 0.2 mm/day get no correction.

    python scripts/precip_bias/global_bias.py --data DIR --out DIR [--target YYYY-MM-DD]
-> DIR/bias_ratio_global.npz: per model and week: ratio (applied, 2°), raw, se_log, mean_fc, mean_obs, sig (BH-FDR q 0.10),
   and runs / effective n / first-last runs / cycle; plus the tropical-band (30°S-30°N) ratio with its 90 % interval.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G                                          # noqa: E402
import build_bias as BB                                   # noqa: E402

STEPS = (168, 336)                                        # = ecmwf_tp.STEPS_GLOBAL
WEEKS = {"wk1": (0, 7), "wk2": (7, 14)}
K, SD_DAYS, DRY, CLIP, NB = 2, 30.0, 0.2, (1 / 3, 3.0), 1000


def load_global_model(data: Path, model: str, k: int):
    inits, acc = [], []
    for f in sorted(data.glob(f"{model}_[0-9]*.npz")):
        z = np.load(f)
        if tuple(z["steps"]) != STEPS or z["acc"].shape[-1] != 360:
            raise ValueError(f"{f.name}: not a global week-1/2 file (steps {tuple(z['steps'])}, shape {z['acc'].shape})")
        inits += [BB._d(s) for s in z["inits"]]
        acc.append(G.coarsen(z["acc"].mean(axis=2).astype("float32"), k))
    if not inits:
        return [], None
    acc = np.concatenate(acc); o = np.argsort(np.array(inits, dtype="datetime64[D]"))
    return [inits[i] for i in o], acc[o]


def doy_dist(a: date, b: date) -> int:
    d = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(d, 365 - d)


def one(model, inits, acc, li, lp, D, target, rng):
    cyc = BB.VERSIONS[model][-1][0]
    keep = [i for i, d in enumerate(inits) if BB.cycle_of(model, d) == cyc]
    out = {}
    for wk, (a, b) in WEEKS.items():
        ini = [inits[i] for i in keep]
        F = (acc[keep, 0] / 7.0) if wk == "wk1" else ((acc[keep, 1] - acc[keep, 0]) / 7.0)
        L = BB.obs_rate(li, lp, ini, a, b)
        full = np.isfinite(L).any(axis=(1, 2))
        ini = [d for d, ok in zip(ini, full) if ok]; F, L = F[full], L[full]
        if len(ini) < 30:
            raise SystemExit(f"{model} {wk}: only {len(ini)} verified runs of cycle {cyc}")
        mids = [(d + timedelta(days=(a + b) / 2)).month for d in ini]
        T, se_adj = BB.adjust_late(L, mids, D)
        if T is None:
            raise SystemExit(f"{model} {wk}: no Late-to-Final ratio for some season")
        tgt = target + timedelta(days=(a + b) / 2)
        w = np.array([np.exp(-0.5 * (doy_dist(d + timedelta(days=(a + b) / 2), tgt) / SD_DAYS) ** 2) for d in ini])
        ok = np.isfinite(T) & np.isfinite(F)
        Tw = np.where(ok, T, 0.0) * w[:, None, None]; Fw = np.where(ok, F, 0.0) * w[:, None, None]; Cw = ok * w[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            mT, mF = Tw.sum(0) / Cw.sum(0), Fw.sum(0) / Cw.sum(0)
            raw = mT / mF
        # block bootstrap of the weighted ratio (21-day blocks of runs)
        blocks = np.array([(d - ini[0]).days // BB.BLOCK_DAYS for d in ini]); ub, inv = np.unique(blocks, return_inverse=True)
        cells = G.NY * G.NX
        ST = np.zeros((len(ub), cells)); SF = np.zeros_like(ST)
        np.add.at(ST, inv, Tw.reshape(len(ini), cells)); np.add.at(SF, inv, Fw.reshape(len(ini), cells))
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NB).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            se_boot = np.nanstd(np.log((W @ ST) / (W @ SF)), axis=0).reshape(G.NY, G.NX)
            rel_truth = np.where(mT > 0, np.nan_to_num(se_adj) / mT, 0.0)       # the Late-to-Final ratio's uncertainty, relative
            lr = np.log(raw)
        se = np.sqrt(se_boot ** 2 + rel_truth ** 2)
        usable = np.isfinite(lr) & np.isfinite(se) & (mF >= DRY) & (mT >= DRY)
        with np.errstate(invalid="ignore", divide="ignore"):
            shrink = np.where(usable, np.clip(1 - (se / np.abs(lr)) ** 2, 0, 1), 0.0)
        z = np.where(usable, lr / se, np.nan)
        sig = BB.bh_fdr(BB.pvals(z), q=0.10)
        applied = np.exp(np.clip(np.where(usable & sig, lr * shrink, 0.0), np.log(CLIP[0]), np.log(CLIP[1])))
        # tropical band, all boxes 30°S-30°N: weighted ratio of sums with a 90 % block-bootstrap interval
        band = (np.abs(G.LAT) <= 30)[:, None] & np.ones((1, G.NX), bool)
        aw = (np.cos(np.deg2rad(G.LAT))[:, None] * np.ones((1, G.NX)))[band]
        st = (ST.reshape(len(ub), G.NY, G.NX)[:, band] * aw); sf = (SF.reshape(len(ub), G.NY, G.NX)[:, band] * aw)
        with np.errstate(invalid="ignore"):
            draws = (W @ np.nansum(sf, axis=1)) / (W @ np.nansum(st, axis=1))     # forecast / truth: > 1 = too wet
        bandr = float(np.nansum(sf) / np.nansum(st)); lo, hi = np.nanpercentile(draws, [5, 95])
        out[wk] = {"ratio": applied.astype("float32"), "raw": raw.astype("float32"), "se_log": se.astype("float32"),
                   "mean_fc": mF.astype("float32"), "mean_obs": mT.astype("float32"), "sig": sig,
                   "meta": {"runs": len(ini), "eff_n": round(float(w.sum() ** 2 / (w ** 2).sum()), 1), "first": f"{ini[0]}",
                            "last": f"{ini[-1]}", "cycle": cyc, "blocks": int(len(ub)), "sig_frac": round(float(sig[usable].mean()), 3),
                            "tropics_fc_over_obs": [round(bandr, 3), round(float(lo), 3), round(float(hi), 3)]}}
        print(f"{model} {wk} ({cyc}): {len(ini)} runs {ini[0]}..{ini[-1]}, eff n {out[wk]['meta']['eff_n']}, "
              f"{100 * out[wk]['meta']['sig_frac']:.0f} % of boxes sig.; 30S-30N forecast/obs {bandr:.2f} [{lo:.2f}, {hi:.2f}]", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--target", help="first day of the forecasts the correction is for (default: today)")
    a = ap.parse_args()
    G.use_domain("global")
    data = Path(a.data); target = date.fromisoformat(a.target) if a.target else date.today()
    rng = np.random.default_rng(20261014)
    # everything is loaded on the 1° grid and coarsened to K-degree boxes as it loads; then the grid becomes the box grid
    li, lp = BB.load_obs(data, "late", K); fi, fp = BB.load_obs(data, "final", K)
    models = {m: load_global_model(data, m, K) for m in ("ifs", "aifs-ens")}
    G.set_coarse(K)
    D = BB.truth_terms(li, lp, fi, fp, rng)
    print("Late-to-Final ratio seasons:", {s: v["days"] for s, v in D.items()}, flush=True)
    res, meta = {}, {"target": f"{target}", "kernel_sd_days": SD_DAYS, "coarse_deg": K, "clip": CLIP, "dry_mm_day": DRY}
    for model, (inits, acc) in models.items():
        r = one(model, inits, acc, li, lp, D, target, rng)
        for wk, v in r.items():
            for k2, arr in v.items():
                if k2 != "meta":
                    res[f"{model}_{wk}_{k2}"] = arr
            meta[f"{model}_{wk}"] = v["meta"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / "bias_ratio_global.npz", lat=G.LAT, lon=G.LON, meta=json.dumps(meta), **res)
    (Path(a.out) / "bias_ratio_global.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
