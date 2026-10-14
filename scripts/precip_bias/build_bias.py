#!/usr/bin/env python3
"""Americas model-rain bias field: IFS-ENS and AIFS-ENS against IMERG, by lead and season (2026-10-13).

Inputs: the monthly archive of release precip-bias-arch-v1 (ecmwf_tp.py, imerg_americas.py), unpacked in one directory.

Method, per model, lead band and season (season of the band's middle valid day):
  F  ensemble-mean rain rate of the 6 archived members over the band (mm/day, 1° box means)
  L  IMERG V07 Late over the same UTC days (every day present, or the cell is skipped for that run)
  bias vs IMERG Late          mean(F - L) over the runs
  truth (since 2026-10-13)     Late x r_s, r_s = sum(Final) / sum(Late) over every day of season s in the Final record
                               (multiplicative - the additive version below made dry-season truths negative). Formerly:
  truth term                   D_s = mean(Late - Final) over every day of season s in the Final record (to 2025-09-30):
                               IMERG Late is satellite-only with a climatological gauge calibration; Final carries the
                               monthly GPCC gauge adjustment, so D_s is Late's own error, assumed steady by season. A
                               property of IMERG, not of the runs: a sample takes sum_s (n_s / n) D_s over its runs' seasons
  bias vs gauge-adjusted IMERG mean(F - L) + mean(L - Final)          <- the headline field
  relative bias                mean F / (mean L - mean(L - Final)) - 1, only where that observed mean is >= 0.5 mm/day

Significance (USER RULE: nothing insignificant on the site): consecutive runs share weather, so the standard error of
each mean comes from a block bootstrap over 21-day blocks of start dates (1000 draws, multinomial block weights), the
two terms' variances added; two-sided normal p; Benjamini-Hochberg FDR at q = Q_FDR (0.20) across the cells of each map
(Wilks 2016). Cells that fail are left blank on the maps and counted in the JSON.

Two comparisons: IFS-ENS over its whole record (2024-03 on, the most precise field), and IFS-ENS vs AIFS-ENS over the
SAME runs (AIFS-ENS starts 2025-07-02), so the two models are judged on the same weather.

    python scripts/precip_bias/build_bias.py --data DIR --out scripts/precip_bias/data/bias_fields.npz --json assets/precip_bias/bias_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G                                          # noqa: E402

STEPS = (24, 72, 168, 240, 336, 360)                      # = ecmwf_tp.STEPS (not imported: that module needs eccodes)
LEADS = {"day1": (0, 1, "day 1"), "d2_3": (1, 3, "days 2-3"), "d4_7": (3, 7, "days 4-7"),
         "d8_10": (7, 10, "days 8-10"), "d11_14": (10, 14, "days 11-14"), "wk1": (0, 7, "week 1 (days 1-7)"),
         "wk2": (7, 14, "week 2 (days 8-14)")}
SEASONS = {"all": tuple(range(1, 13)), "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MODELS = {"ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS"}
BLOCK_DAYS, NBOOT, REL_MIN = 21, 1000, 0.5
# Significance levels (2026-10-13, user: "coarse graining for the bias maps is a good idea. Maybe relax the significance criteria
# as well"): maps at FDR q = 0.20 (was 0.10) - on average at most one shaded cell in five is a false positive, about a 10 %
# field-wide test for fields this spatially correlated (Wilks 2016: alpha_FDR = 2 alpha_global); intervals 90 % (was 95 %);
# statistics on 2-degree boxes (COARSE, --coarse).
Q_FDR, Z_CI, CI_LEVEL, COARSE = 0.20, 1.645, 0.90, 2
# Model cycles (2026-10-13, user: "There have been a few ifs upgrades since then, use with care"). IFS 48r1 -> 49r1 on
# 2024-11-12 and 49r1 -> 50r1 with AIFS-ENS v1 -> v2 on 2026-05-12, both at the 06 UTC run, so a cycle starts with the next
# day's 00Z run. Runs are NEVER pooled across cycles: every field below belongs to one cycle of each model.
VERSIONS = {
    "ifs": [("48r1", date(2024, 3, 1), date(2024, 11, 12)), ("49r1", date(2024, 11, 13), date(2026, 5, 12)),
            ("50r1", date(2026, 5, 13), None)],
    "aifs-ens": [("v1", date(2025, 7, 2), date(2026, 5, 12)), ("v2", date(2026, 5, 13), None)],
}
COMPARISONS = {                                           # key: (label, [(model, cycle)], the same runs for both)
    "current": ("IFS-ENS 50r1 and AIFS-ENS v2, the systems running now", [("ifs", "50r1"), ("aifs-ens", "v2")]),
    "previous": ("IFS-ENS 49r1 and AIFS-ENS v1", [("ifs", "49r1"), ("aifs-ens", "v1")]),
    "ifs49": ("IFS-ENS 49r1, its whole operational life", [("ifs", "49r1")]),
}
UPGRADES = [("ifs", "48r1", "49r1"), ("ifs", "49r1", "50r1"), ("aifs-ens", "v1", "v2")]


def cycle_of(model: str, d: date) -> str | None:
    for name, a, b in VERSIONS[model]:
        if d >= a and (b is None or d <= b):
            return name
    return None
REGIONS = {                                               # (lat0, lat1, lon0, lon1), land or all cells
    "Eastern US": (30, 45, -95, -75), "Western US": (32, 49, -124, -105), "Mexico": (15, 30, -110, -95),
    "Amazon": (-12, 3, -72, -50), "Southeast South America": (-35, -22, -62, -47),
    "Northeast Brazil": (-12, -3, -45, -35), "Andes": (-20, 0, -80, -68), "East Pacific ITCZ": (4, 12, -130, -95),
    "Caribbean": (12, 22, -85, -62),
}


def _d(s): return datetime.strptime(str(s), "%Y%m%d").date()


def load_obs(data: Path, product: str, k: int = 1, prefix: str = "imerg_"):
    """{date: row}, (days, NY, NX) mm/day on k-degree boxes - or (None, None). prefix "" for the blend (blend_YYYYMM.npz)."""
    files = sorted(data.glob(f"{prefix}{product}_[0-9]*.npz"))
    if not files:
        return None, None
    dates, arr = [], []
    for f in files:
        z = np.load(f)
        dates += [_d(s) for s in z["dates"]]; arr.append(G.coarsen(z["pr"].astype("float32"), k))
    pr = np.concatenate(arr)
    order = np.argsort(np.array(dates, dtype="datetime64[D]"))
    dates = [dates[i] for i in order]; pr = pr[order]
    return {d: i for i, d in enumerate(dates)}, pr


def load_model(data: Path, model: str, k: int = 1):
    """inits (sorted dates), ensemble-mean accumulations (run, step, NY, NX) mm on k-degree boxes."""
    inits, acc = [], []
    for f in sorted(data.glob(f"{model}_*.npz")):
        z = np.load(f)
        if tuple(z["steps"]) != STEPS:
            raise ValueError(f"{f.name}: steps {tuple(z['steps'])} != {STEPS}")
        inits += [_d(s) for s in z["inits"]]
        acc.append(G.coarsen(z["acc"].mean(axis=2).astype("float32"), k))   # mean over the members, then k-degree boxes
    if not inits:
        return [], None
    acc = np.concatenate(acc)
    order = np.argsort(np.array(inits, dtype="datetime64[D]"))
    return [inits[i] for i in order], acc[order]


def lead_rate(acc: np.ndarray, a: int, b: int) -> np.ndarray:
    """Mean rain rate over lead days [a, b) from accumulations at STEPS (accumulation at step 0 is zero)."""
    hi = acc[:, STEPS.index(24 * b)]
    lo = acc[:, STEPS.index(24 * a)] if a else 0.0
    return (hi - lo) / (b - a)


def obs_rate(idx: dict, pr: np.ndarray, inits: list, a: int, b: int) -> np.ndarray:
    """(run, NY, NX) observed mean over the runs' lead days [a, b); NaN where a day is missing or the cell invalid."""
    out = np.full((len(inits), G.NY, G.NX), np.nan, dtype="float32")
    for r, d in enumerate(inits):
        rows = [idx.get(d + timedelta(days=k)) for k in range(a, b)]
        if any(x is None for x in rows):
            continue
        out[r] = pr[rows].mean(axis=0)              # NaN in any day -> NaN mean: one invalid day voids the cell
    return out


def _boot_mean_se(x: np.ndarray, blocks: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, bootstrap SE and count per cell of x (run, cells) with NaNs, resampling whole blocks of runs."""
    ok = np.isfinite(x); x0 = np.where(ok, x, 0.0)
    ub, inv = np.unique(blocks, return_inverse=True)
    S = np.zeros((len(ub), x.shape[1])); C = np.zeros_like(S)
    np.add.at(S, inv, x0); np.add.at(C, inv, ok)
    n = C.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S.sum(0) / n
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NBOOT).astype("float64")
        draws = (W @ S) / (W @ C)
    se = np.nanstd(draws, axis=0)
    return mean, se, n


def bh_fdr(p: np.ndarray, q: float = Q_FDR) -> np.ndarray:
    """Benjamini-Hochberg: boolean 'significant' for the finite p-values of one map."""
    sig = np.zeros(p.shape, bool); f = np.isfinite(p)
    if not f.any():
        return sig
    ps = p[f]; o = np.argsort(ps); m = len(ps)
    thr = q * np.arange(1, m + 1) / m
    passed = ps[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    s = np.zeros(m, bool); s[o[:k]] = True
    sig[f] = s
    return sig


def pvals(z):
    from math import erfc, sqrt
    return np.vectorize(lambda v: erfc(abs(v) / sqrt(2)) if np.isfinite(v) else np.nan)(z)


SEASON_OF = {m: k for k, v in SEASONS.items() if k != "all" for m in v}


RATIO_MIN_LATE, RATIO_CLIP = 0.1, (0.25, 4.0)


def truth_terms(li, lp, fi, fp, rng):
    """{season: {ratio, se_ratio, diff, se_diff, days}} from the days IMERG Late and Final share; {} without Final.
    ratio = sum(Final) / sum(Late) per box: the truth is Late SCALED by its season's ratio (2026-10-13: the earlier ADDITIVE
    shift, Late minus the seasonal mean of Late - Final, made the "observed" mean NEGATIVE over the dry-season high Andes and
    inflated the bias there). Ratio 1 where Late averages under 0.1 mm/day, clipped to [0.25, 4]; its se from the 21-day block
    bootstrap of both sums. diff = mean(Late - Final), kept for the truth maps."""
    if not fi:
        return {}
    days = sorted(set(li) & set(fi))
    out, d0, cells = {}, days[0], G.NY * G.NX
    for sk in ("DJF", "MAM", "JJA", "SON"):
        dd = [d for d in days if d.month in SEASONS[sk]]
        if len(dd) < 30:
            continue
        Lx = lp[[li[d] for d in dd]].reshape(len(dd), cells); Fx = fp[[fi[d] for d in dd]].reshape(len(dd), cells)
        ok = np.isfinite(Lx) & np.isfinite(Fx)
        blocks = np.array([(d - d0).days // BLOCK_DAYS for d in dd]); ub, inv = np.unique(blocks, return_inverse=True)
        SL = np.zeros((len(ub), cells)); SF = np.zeros_like(SL); C = np.zeros_like(SL)
        np.add.at(SL, inv, np.where(ok, Lx, 0.0)); np.add.at(SF, inv, np.where(ok, Fx, 0.0)); np.add.at(C, inv, ok)
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NBOOT).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = SF.sum(0) / SL.sum(0); se_r = np.nanstd((W @ SF) / (W @ SL), axis=0); mL = SL.sum(0) / C.sum(0)
        wet = mL >= RATIO_MIN_LATE
        r = np.where(wet, np.clip(np.nan_to_num(r, nan=1.0), *RATIO_CLIP), 1.0); se_r = np.where(wet, np.nan_to_num(se_r), 0.0)
        m, se, _ = _boot_mean_se(Lx - Fx, blocks, rng)
        out[sk] = {"ratio": r.reshape(G.NY, G.NX), "se_ratio": se_r.reshape(G.NY, G.NX), "diff": m.reshape(G.NY, G.NX),
                   "se_diff": se.reshape(G.NY, G.NX), "days": len(dd)}
    return out


TRUTH = {"kind": "imerg", "spread": None, "w_imerg": None}     # set in main(): "blend" uses blend.py's daily truth and its seasonal spread


def truth_of(L, mids, D):
    """The truth for runs whose bands sit in months `mids`: (field, se of its sample mean) or (None, None).
    blend: L already IS the blend; its systematic uncertainty = the products' seasonal spread (sd of their seasonal means),
    combined over the sample's seasons as fully correlated (sum of weighted sds - the conservative choice).
    imerg: IMERG Late scaled by its seasonal ratio to IMERG Final (adjust_late)."""
    if TRUTH["kind"] != "blend":
        return adjust_late(L, mids, D)
    seas = [SEASON_OF[int(m)] for m in mids]
    sp = TRUTH["spread"] or {}
    se = np.zeros((G.NY, G.NX)); v = np.zeros((G.NY, G.NX))
    w = TRUTH.get("w_imerg"); w = np.ones((G.NY, G.NX)) if w is None else w
    for x in set(seas):
        if x in sp:
            se += seas.count(x) / len(seas) * sp[x]
        # the adjusted IMERG inside the blend (all of it over the sea, where the spread is zero) carries the uncertainty of its
        # Late-to-Final ratio, as in adjust_late; independent of the spread, so added in quadrature
        if D and x in D:
            k = [i for i, y in enumerate(seas) if y == x]
            with np.errstate(invalid="ignore", divide="ignore"):
                mL = np.nanmean(L[k], axis=0)
                rel = np.where(D[x]["ratio"] > 0, D[x]["se_ratio"] / D[x]["ratio"], 0.0)
            v += (len(k) / len(seas) * np.nan_to_num(mL) * np.nan_to_num(rel) * w) ** 2
    return L, np.sqrt(se ** 2 + v)


def adjust_late(L, mids, D):
    """(Late scaled by each run's season ratio, se of the sample-mean adjustment from the ratios' own uncertainty) or (None, None)
    when a season has no ratio. L: (run, NY, NX); mids: the month of each run's band."""
    if not D:
        return None, None
    seas = [SEASON_OF[int(m)] for m in mids]
    if any(x not in D for x in set(seas)):
        return None, None
    La = L * np.stack([D[x]["ratio"] for x in seas])
    v = np.zeros((G.NY, G.NX))
    for x in set(seas):
        k = [i for i, y in enumerate(seas) if y == x]
        with np.errstate(invalid="ignore"):
            mL = np.nanmean(L[k], axis=0)
        v += (len(k) / len(seas) * np.nan_to_num(mL) * D[x]["se_ratio"]) ** 2
    return La, np.sqrt(v)


def field(inits, F, L, Fin, rng, d_start: date, mids=None, D=None):
    """Bias statistics for one (model, lead, season) sample. F, L: (run, NY, NX); Fin unused (the truth comes from D)."""
    nr = len(inits)
    if nr < 20:
        return None
    blocks = np.array([(d - d_start).days // BLOCK_DAYS for d in inits])
    cells = G.NY * G.NX
    La, se_tf = truth_of(L, mids, D)
    adjusted = La is not None
    if not adjusted:
        La, se_tf = L, np.zeros((G.NY, G.NX))
    m_e, se_e, _ = _boot_mean_se((F - La).reshape(nr, cells), blocks, rng)       # bias vs the gauge-adjusted truth
    m_late, _, _ = _boot_mean_se((F - L).reshape(nr, cells), blocks, rng)         # vs raw IMERG Late, for reference
    m_f, _, _ = _boot_mean_se(F.reshape(nr, cells), blocks, rng)
    obs, _, _ = _boot_mean_se(La.reshape(nr, cells), blocks, rng)                  # gauge-adjusted observed mean
    se = np.sqrt(se_e ** 2 + se_tf.ravel() ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = m_e / se
    p = pvals(z); sig = bh_fdr(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(obs >= REL_MIN, (m_f - obs) / obs, np.nan)
    shp = (G.NY, G.NX)
    return {"bias": m_e.reshape(shp), "se": se.reshape(shp), "sig": sig.reshape(shp), "rel": rel.reshape(shp),
            "bias_late": m_late.reshape(shp), "truth": (m_e - m_late).reshape(shp), "obs": obs.reshape(shp), "fc": m_f.reshape(shp),
            "n_runs": int(nr), "truth_adjusted": bool(adjusted),
            "frac_sig": float(sig[np.isfinite(p)].mean()) if np.isfinite(p).any() else 0.0}


def region_stats(inits, F, L, Fin, rng, d_start, mids=None, D=None):
    """Area-mean bias per region with a block-bootstrap CI_LEVEL interval (same truth adjustment)."""
    w = G.area_weights(); out = {}
    blocks = np.array([(d - d_start).days // BLOCK_DAYS for d in inits])
    La, se_tf = truth_of(L, mids, D)
    adjusted = La is not None
    if not adjusted:
        La, se_tf = L, np.zeros((G.NY, G.NX))
    for name, (y0, y1, x0, x1) in REGIONS.items():
        my = (G.LAT >= y0) & (G.LAT <= y1); mx = (G.LON >= x0) & (G.LON <= x1)
        ww = w[np.ix_(my, mx)]
        def amean(a):
            s_ = a[:, my][:, :, mx]
            ok = np.isfinite(s_).all(axis=(1, 2))
            return np.where(ok, (np.nan_to_num(s_) * ww).sum(axis=(1, 2)) / ww.sum(), np.nan)
        m_e, se_e, n_e = _boot_mean_se(amean(F - La)[:, None], blocks, rng)
        m_o, _, _ = _boot_mean_se(amean(La)[:, None], blocks, rng)
        se_t = float((se_tf[np.ix_(my, mx)] * ww).sum() / ww.sum())          # boxes' ratio errors taken as fully correlated
        b = float(m_e[0]); s_ = float(np.sqrt(se_e[0] ** 2 + se_t ** 2)); ob = float(m_o[0])
        z = b / s_ if s_ > 0 else np.nan
        out[name] = {"bias": round(b, 3), "lo": round(b - Z_CI * s_, 3), "hi": round(b + Z_CI * s_, 3), "obs": round(ob, 3),
                     "rel": round(b / ob, 3) if ob >= REL_MIN else None, "sig": bool(np.isfinite(z) and abs(z) > Z_CI),
                     "truth_adjusted": bool(adjusted), "n_runs": int(n_e[0])}
    return out


def run_errors(ini, acc, idx, la, lb, li, lp, D=None):
    """Area-mean (F - gauge-adjusted IMERG) per run over the cells valid in that run, mm/day (NaN if the run has no truth)."""
    w = G.area_weights()
    inis = [ini[i] for i in idx]
    F = lead_rate(acc[idx], la, lb); L = obs_rate(li, lp, inis, la, lb)
    mids = np.array([(d + timedelta(days=(la + lb) / 2)).month for d in inis])
    La, _ = truth_of(L, mids, D)
    e = F - (L if La is None else La); ok = np.isfinite(e)
    with np.errstate(invalid="ignore"):
        return np.where(ok.any(axis=(1, 2)), np.nansum(np.where(ok, e, 0) * w, axis=(1, 2)) / (w * ok).sum(axis=(1, 2)), np.nan)


def paired_change(d_new, d_old, dates, rng):
    """Mean of the paired difference with a 21-day block bootstrap CI_LEVEL interval (blocks of matched days)."""
    x = d_new - d_old; ok = np.isfinite(x)
    x = x[ok]; dd = [d for d, k in zip(dates, ok) if k]
    if len(x) < 40:
        return None
    blocks = np.array([(d - dd[0]).days // BLOCK_DAYS for d in dd])
    m, se, _ = _boot_mean_se(x[:, None], blocks, rng)
    lo, hi = float(m[0] - Z_CI * se[0]), float(m[0] + Z_CI * se[0])
    return {"mean": round(float(m[0]), 3), "lo": round(lo, 3), "hi": round(hi, 3), "sig": bool(lo > 0 or hi < 0), "n": int(len(x))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="npz of the fields (input of render_bias.py)")
    ap.add_argument("--json", required=True, help="summary JSON for the page")
    ap.add_argument("--coarse", type=int, default=COARSE, help="box size in degrees (1, 2, 4, 5)")
    ap.add_argument("--truth", choices=["auto", "blend", "imerg"], default="auto",
                    help="blend = blend.py's truth (blend_YYYYMM.npz + blend_spread.npz in --data); auto = blend if present")
    a = ap.parse_args()
    data = Path(a.data); rng = np.random.default_rng(20261013)
    li, lp = load_obs(data, "late", a.coarse); fi, fp = load_obs(data, "final", a.coarse)
    use_blend = a.truth == "blend" or (a.truth == "auto" and (data / "blend_spread.npz").exists())
    li0, lp0 = li, lp                                         # raw IMERG Late: its ratio to Final and the Late-minus-Final maps
    if use_blend:                                             # the truth fields replace IMERG Late everywhere below
        li, lp = load_obs(data, "blend", a.coarse, prefix="")
        zs = np.load(data / "blend_spread.npz")
        TRUTH["kind"] = "blend"; TRUTH["spread"] = {k: G.coarsen(zs[k].astype("float32"), a.coarse) for k in ("DJF", "MAM", "JJA", "SON") if k in zs}
        if "w_imerg" in zs and "land" in zs:                    # IMERG's share of the blend: its weight on land, all of it at sea
            TRUTH["w_imerg"] = G.coarsen(np.where(zs["land"], zs["w_imerg"], 1.0).astype("float32"), a.coarse)
    if li is None:
        print("no IMERG Late in", data); return 1
    print(f"IMERG Late {min(li)}..{max(li)} ({len(li)} days); Final {min(fi) if fi else '-'}..{max(fi) if fi else '-'} ({len(fi or {})} days)", flush=True)
    M = {m: load_model(data, m, a.coarse) for m in MODELS}
    G.set_coarse(a.coarse)                                       # every statistic below is on the k-degree boxes
    D = truth_terms(li0, lp0, fi, fp, rng)
    print("truth ratio (Final / Late) seasons:", {k: v["days"] for k, v in D.items()}, flush=True)
    for m, (ini, _) in M.items():
        print(f"{MODELS[m]}: {len(ini)} runs {ini[0] if ini else '-'}..{ini[-1] if ini else '-'}", flush=True)
    fields, summary = {}, {"generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "leads": {k: v[2] for k, v in LEADS.items()},
                           "models": MODELS, "imerg_late": [str(min(li)), str(max(li))],
                           "imerg_final": [str(min(fi)), str(max(fi))] if fi else None, "q_fdr": Q_FDR, "ci_level": CI_LEVEL, "grid_deg": a.coarse, "block_days": BLOCK_DAYS,
                           "comparisons": {}}
    def sample(model, idx, comp, lk_list=LEADS, seasons=SEASONS, store=True):
        """Bias fields for runs idx of one model (one cycle) - per lead and season; returns {(lk, sk): field}."""
        ini, acc = M[model]
        ini_k = [ini[i] for i in idx]; acc_k = acc[idx]
        out = {}
        for lk in lk_list:
            la, lb, _ = LEADS[lk]
            F = lead_rate(acc_k, la, lb)
            L = obs_rate(li, lp, ini_k, la, lb)
            mid = np.array([(d + timedelta(days=(la + lb) / 2)).month for d in ini_k])
            ok_run = np.isfinite(L).any(axis=(1, 2))
            for sk in seasons:
                sel = np.where(np.isin(mid, SEASONS[sk]) & ok_run)[0]
                if len(sel) < 20:
                    continue
                ins = [ini_k[i] for i in sel]
                r = field(ins, F[sel], L[sel], None, rng, ini_k[0], mid[sel], D)
                if r is None:
                    continue
                out[(lk, sk)] = r
                if store:
                    fields[f"{comp}|{model}|{lk}|{sk}"] = r
                    reg = region_stats(ins, F[sel], L[sel], None, rng, ini_k[0], mid[sel], D)
                    summary["comparisons"][comp][model]["leads"].setdefault(lk, {})[sk] = {
                        "runs": r["n_runs"], "truth_adjusted": r["truth_adjusted"], "frac_sig": round(r["frac_sig"], 3),
                        "domain_bias": round(float(np.nansum(r["bias"] * G.area_weights()) / np.sum(G.area_weights() * np.isfinite(r["bias"]))), 3),
                        "first": str(ins[0]), "last": str(ins[-1]), "regions": reg}
                    print(f"  {comp:8s} {model:8s} {lk:6s} {sk:3s} runs {r['n_runs']:4d}  significant {100 * r['frac_sig']:5.1f} %", flush=True)
        return out

    seg = {m: {v: [i for i, d in enumerate(M[m][0]) if cycle_of(m, d) == v] for v, _, _ in VERSIONS[m]} for m in MODELS}
    summary["cycles"] = {m: [{"cycle": v, "from": str(a), "to": str(b) if b else None, "runs": len(seg[m][v])} for v, a, b in VERSIONS[m]]
                         for m in MODELS}
    for comp, (label, members) in COMPARISONS.items():
        dates = None
        for m, v in members:                                  # the same runs for every member of the comparison
            ds = {M[m][0][i] for i in seg[m][v]}
            dates = ds if dates is None else dates & ds
        if not dates or len(dates) < 20:
            print(f"  {comp}: {len(dates or [])} common runs - skipped", flush=True); continue
        summary["comparisons"][comp] = {"label": label}
        for m, v in members:
            idx = [i for i in seg[m][v] if M[m][0][i] in dates]
            summary["comparisons"][comp][m] = {"cycle": v, "runs": len(idx), "first": str(M[m][0][idx[0]]), "last": str(M[m][0][idx[-1]]), "leads": {}}
            sample(m, idx, comp)

    # Did an upgrade change the bias? The new cycle's runs against the old cycle's runs on the SAME calendar days one year
    # earlier (so the season mix matches; the weather year does not, so part of any difference is the weather).
    summary["upgrades"] = {}
    pair_err = {}
    for m, old, new in UPGRADES:
        ini = M[m][0]; pos = {d: i for i, d in enumerate(ini)}
        pairs = []
        for i in seg[m][new]:
            d = ini[i]
            try:
                d0 = d.replace(year=d.year - 1)
            except ValueError:                                # 29 February
                continue
            j = pos.get(d0)
            if j is not None and cycle_of(m, d0) == old:
                pairs.append((j, i))
        key = f"{m}|{old}->{new}"
        if len(pairs) < 40:
            summary["upgrades"][key] = {"runs": len(pairs), "note": "too few matched runs"}; continue
        f_old = sample(m, [j for j, _ in pairs], None, seasons=("all",), store=False)
        f_new = sample(m, [i for _, i in pairs], None, seasons=("all",), store=False)
        res = {"runs": len(pairs), "old": [str(ini[pairs[0][0]]), str(ini[pairs[-1][0]])], "new": [str(ini[pairs[0][1]]), str(ini[pairs[-1][1]])], "leads": {}}
        pdates = [ini[i] for _, i in pairs]
        pair_err[key] = {}
        for lk in LEADS:
            if (lk, "all") not in f_old or (lk, "all") not in f_new:
                continue
            fa, fb = f_old[(lk, "all")], f_new[(lk, "all")]
            dlt = fb["bias"] - fa["bias"]; se = np.sqrt(fa["se"] ** 2 + fb["se"] ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = pvals((dlt / se).ravel())
            sg = bh_fdr(p).reshape(dlt.shape)
            ok = np.isfinite(fa["bias"]) & np.isfinite(fb["bias"])
            r = float(np.corrcoef(fa["bias"][ok], fb["bias"][ok])[0, 1])
            w = G.area_weights()
            la, lb, _ = LEADS[lk]
            e_old = run_errors(ini, M[m][1], [j for j, _ in pairs], la, lb, li, lp, D)
            e_new = run_errors(ini, M[m][1], [i for _, i in pairs], la, lb, li, lp, D)
            pair_err[key][lk] = dict(zip(pdates, e_new - e_old))
            chg = paired_change(e_new, e_old, pdates, rng)
            fields[f"change|{key}|{lk}"] = {"bias": dlt, "sig": sg, "rel": np.full_like(dlt, np.nan), "truth": np.full_like(dlt, np.nan),
                                            "obs": np.full_like(dlt, np.nan), "bias_late": dlt}
            res["leads"][lk] = {"frac_sig": round(float(sg[np.isfinite(p.reshape(dlt.shape))].mean()), 3), "pattern_r": round(r, 3),
                                "domain_change": round(float(np.nansum(dlt * w) / np.sum(w * np.isfinite(dlt))), 3),
                                "domain_old": round(float(np.nansum(fa["bias"] * w) / np.sum(w * np.isfinite(fa["bias"]))), 3),
                                "domain_new": round(float(np.nansum(fb["bias"] * w) / np.sum(w * np.isfinite(fb["bias"]))), 3),
                                "domain_change_test": chg}
            print(f"  upgrade {key:16s} {lk:6s} runs {len(pairs):4d}  bias pattern r {r:.2f}  cells changed (FDR) {100 * res['leads'][lk]['frac_sig']:5.1f} %", flush=True)
        summary["upgrades"][key] = res
    did = {}
    a_k, b_k = "ifs|49r1->50r1", "aifs-ens|v1->v2"
    if a_k in pair_err and b_k in pair_err:
        for lk in LEADS:
            A, B = pair_err[a_k].get(lk, {}), pair_err[b_k].get(lk, {})
            common = sorted(set(A) & set(B))
            if len(common) >= 40:
                did[lk] = paired_change(np.array([A[d] for d in common]), np.array([B[d] for d in common]), common, rng)
        summary["upgrade_did"] = {"label": "IFS-ENS 49r1->50r1 change minus AIFS-ENS v1->v2 change, same days (the weather year cancels)",
                                  "leads": did}
        if "wk1" in did and did["wk1"]:
            print(f"  difference-in-differences wk1: {did['wk1']}", flush=True)
    for sk, t in D.items():
        m, se = t["diff"], t["se_diff"]
        with np.errstate(invalid="ignore", divide="ignore"):
            sgt = bh_fdr(pvals((m / se).ravel())).reshape(m.shape)
        fields[f"truth|{sk}"] = {"bias": m, "sig": sgt, "rel": np.full_like(m, np.nan), "truth": m, "obs": np.full_like(m, np.nan), "bias_late": m}
    summary["truth_days"] = {k: v["days"] for k, v in D.items()}
    for sk, sp in (TRUTH["spread"] or {}).items():
        fields[f"spread|{sk}"] = {"bias": sp, "sig": np.isfinite(sp), "rel": np.full_like(sp, np.nan), "truth": sp,
                                  "obs": np.full_like(sp, np.nan), "bias_late": sp}
    summary["truth_method"] = ("blend of IMERG (gauge-adjusted), MERGE, CPC, GPCC and CHIRPS weighted by their skill against IDEAM gauges "
                               "(scripts/precip_bias/blend.py); its products' spread is part of every interval" if TRUTH["kind"] == "blend"
                               else "IMERG Late scaled by its seasonal ratio to IMERG Final (per box)")
    summary["truth_kind"] = TRUTH["kind"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, keys=np.array(list(fields)), lat=G.LAT, lon=G.LON,
                        **{f"{k}#{v}": fields[k][v] for k in fields for v in ("bias", "sig", "rel", "truth", "obs", "bias_late")})
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(summary, indent=1))
    print(f"wrote {len(fields)} fields -> {a.out}; summary -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
