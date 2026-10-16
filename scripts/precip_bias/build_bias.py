"""Shared estimation tools for the model-rain bias products (scripts/precip_bias/global_bias.py, tropical_rain.py).

Model cycles (VERSIONS, cycle_of: runs are never pooled across model upgrades), the IMERG archive readers (load_obs,
obs_rate), the scaling of IMERG Late to the gauge-adjusted Final by season (truth_terms, adjust_late), and the
significance tools: a block bootstrap over BLOCK_DAYS-long blocks of start dates (consecutive runs share weather),
two-sided normal p-values and Benjamini-Hochberg FDR control at Q_FDR (Wilks 2016).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid as G

SEASONS = {"all": tuple(range(1, 13)), "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}

BLOCK_DAYS, NBOOT, REL_MIN = 21, 1000, 0.5

# Significance levels: maps at FDR q = 0.20 - on average at most one shaded cell in five is a false positive, about a 10 %
# field-wide test for fields this spatially correlated (Wilks 2016: alpha_FDR = 2 alpha_global); intervals 90 %;
# statistics on 2-degree boxes (COARSE, --coarse).
Q_FDR, Z_CI, CI_LEVEL, COARSE = 0.20, 1.645, 0.90, 2

# Model cycles. IFS 48r1 -> 49r1 on
# 2024-11-12 and 49r1 -> 50r1 with AIFS-ENS v1 -> v2 on 2026-05-12, both at the 06 UTC run, so a cycle starts with the next
# day's 00Z run. Runs are NEVER pooled across cycles: every field below belongs to one cycle of each model.
VERSIONS = {
    "ifs": [
        ("48r1", date(2024, 3, 1), date(2024, 11, 12)),
        ("49r1", date(2024, 11, 13), date(2026, 5, 12)),
        ("50r1", date(2026, 5, 13), None),
    ],
    "aifs-ens": [("v1", date(2025, 7, 2), date(2026, 5, 12)), ("v2", date(2026, 5, 13), None)],
}


def cycle_of(model: str, d: date) -> str | None:
    for name, a, b in VERSIONS[model]:
        if d >= a and (b is None or d <= b):
            return name
    return None


def _d(s):
    return datetime.strptime(str(s), "%Y%m%d").date()


def load_obs(data: Path, product: str, k: int = 1, prefix: str = "imerg_"):
    """{date: row}, (days, NY, NX) mm/day on k-degree boxes - or (None, None). prefix "" for the blend (blend_YYYYMM.npz)."""
    files = sorted(data.glob(f"{prefix}{product}_[0-9]*.npz"))
    if not files:
        return None, None
    dates, arr = [], []
    for f in files:
        z = np.load(f)
        dates += [_d(s) for s in z["dates"]]
        arr.append(G.coarsen(z["pr"].astype("float32"), k))
    pr = np.concatenate(arr)
    order = np.argsort(np.array(dates, dtype="datetime64[D]"))
    dates = [dates[i] for i in order]
    pr = pr[order]
    return {d: i for i, d in enumerate(dates)}, pr


def obs_rate(idx: dict, pr: np.ndarray, inits: list, a: int, b: int) -> np.ndarray:
    """(run, NY, NX) observed mean over the runs' lead days [a, b); NaN where a day is missing or the cell invalid."""
    out = np.full((len(inits), G.NY, G.NX), np.nan, dtype="float32")
    for r, d in enumerate(inits):
        rows = [idx.get(d + timedelta(days=k)) for k in range(a, b)]
        if any(x is None for x in rows):
            continue
        out[r] = pr[rows].mean(axis=0)  # NaN in any day -> NaN mean: one invalid day voids the cell
    return out


def _boot_mean_se(x: np.ndarray, blocks: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, bootstrap SE and count per cell of x (run, cells) with NaNs, resampling whole blocks of runs."""
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    ub, inv = np.unique(blocks, return_inverse=True)
    S = np.zeros((len(ub), x.shape[1]))
    C = np.zeros_like(S)
    np.add.at(S, inv, x0)
    np.add.at(C, inv, ok)
    n = C.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S.sum(0) / n
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NBOOT).astype("float64")
        draws = (W @ S) / (W @ C)
    se = np.nanstd(draws, axis=0)
    return mean, se, n


def bh_fdr(p: np.ndarray, q: float = Q_FDR) -> np.ndarray:
    """Benjamini-Hochberg: boolean 'significant' for the finite p-values of one map."""
    sig = np.zeros(p.shape, bool)
    f = np.isfinite(p)
    if not f.any():
        return sig
    ps = p[f]
    o = np.argsort(ps)
    m = len(ps)
    thr = q * np.arange(1, m + 1) / m
    passed = ps[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    s = np.zeros(m, bool)
    s[o[:k]] = True
    sig[f] = s
    return sig


def pvals(z):
    from math import erfc, sqrt

    return np.vectorize(lambda v: erfc(abs(v) / sqrt(2)) if np.isfinite(v) else np.nan)(z)


SEASON_OF = {m: k for k, v in SEASONS.items() if k != "all" for m in v}

RATIO_MIN_LATE, RATIO_CLIP = 0.1, (0.25, 4.0)


def truth_terms(li, lp, fi, fp, rng):
    """{season: {ratio, se_ratio, diff, se_diff, days}} from the days IMERG Late and Final share; {} without Final.
    ratio = sum(Final) / sum(Late) per box: the truth is Late SCALED by its season's ratio (an ADDITIVE
    shift, Late minus the seasonal mean of Late - Final, makes the "observed" mean NEGATIVE over the dry-season high Andes and
    inflates the bias there). Ratio 1 where Late averages under 0.1 mm/day, clipped to [0.25, 4]; its se from the 21-day block
    bootstrap of both sums. diff = mean(Late - Final), kept for the truth maps."""
    if not fi:
        return {}
    days = sorted(set(li) & set(fi))
    out, d0, cells = {}, days[0], G.NY * G.NX
    for sk in ("DJF", "MAM", "JJA", "SON"):
        dd = [d for d in days if d.month in SEASONS[sk]]
        if len(dd) < 30:
            continue
        Lx = lp[[li[d] for d in dd]].reshape(len(dd), cells)
        Fx = fp[[fi[d] for d in dd]].reshape(len(dd), cells)
        ok = np.isfinite(Lx) & np.isfinite(Fx)
        blocks = np.array([(d - d0).days // BLOCK_DAYS for d in dd])
        ub, inv = np.unique(blocks, return_inverse=True)
        SL = np.zeros((len(ub), cells))
        SF = np.zeros_like(SL)
        C = np.zeros_like(SL)
        np.add.at(SL, inv, np.where(ok, Lx, 0.0))
        np.add.at(SF, inv, np.where(ok, Fx, 0.0))
        np.add.at(C, inv, ok)
        W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NBOOT).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = SF.sum(0) / SL.sum(0)
            se_r = np.nanstd((W @ SF) / (W @ SL), axis=0)
            mL = SL.sum(0) / C.sum(0)
        wet = mL >= RATIO_MIN_LATE
        r = np.where(wet, np.clip(np.nan_to_num(r, nan=1.0), *RATIO_CLIP), 1.0)
        se_r = np.where(wet, np.nan_to_num(se_r), 0.0)
        m, se, _ = _boot_mean_se(Lx - Fx, blocks, rng)
        out[sk] = {
            "ratio": r.reshape(G.NY, G.NX),
            "se_ratio": se_r.reshape(G.NY, G.NX),
            "diff": m.reshape(G.NY, G.NX),
            "se_diff": se.reshape(G.NY, G.NX),
            "days": len(dd),
        }
    return out


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
