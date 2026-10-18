"""Shared estimation tools for the model-rain bias products (scripts/precip_bias/global_bias.py, blend.py,
tropical_rain.py).

Model cycles (VERSIONS, cycle_of: runs are never pooled across model upgrades), the IMERG archive readers (load_obs,
obs_rate), the scaling of IMERG Late to the gauge-adjusted Final by season (truth_terms, adjust_late), the GPCP normal
week (clim_at), and the significance tools: BLOCK_DAYS-long blocks of start dates for the block bootstraps (consecutive
runs share weather), two-sided normal p-values and Benjamini-Hochberg FDR control (Wilks 2016).
"""

from __future__ import annotations

import calendar
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid as G

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}

BLOCK_DAYS, NBOOT = 21, 1000

# Model cycles (first and last 00Z run, inclusive). IFS 48r1 -> 49r1 on 2024-11-12 and 49r1 -> 50r1 with AIFS-ENS v1 -> v2
# on 2026-05-12, both at the 06 UTC run, so a cycle starts with the next day's 00Z run. GEFS v12 from the 12 UTC run of
# 2020-09-23 (NCEP's v12.x releases since then changed the aerosol member, the wave products and GRIB packing, not the
# atmosphere). GEPS 8.0.0 from the 12 UTC run of 2024-06-11; 8.1.0, ECCC's move to its new supercomputer, completed on
# 2026-04-14 at a run ECCC does not name, so that day belongs to neither cycle. Runs are NEVER pooled across cycles: every
# field below belongs to one cycle of each model.
VERSIONS = {
    "ifs": [
        ("48r1", date(2024, 3, 1), date(2024, 11, 12)),
        ("49r1", date(2024, 11, 13), date(2026, 5, 12)),
        ("50r1", date(2026, 5, 13), None),
    ],
    "aifs-ens": [("v1", date(2025, 7, 2), date(2026, 5, 12)), ("v2", date(2026, 5, 13), None)],
    "gefs": [("v12", date(2020, 9, 24), None)],
    "geps": [("8.0.0", date(2024, 6, 12), date(2026, 4, 13)), ("8.1.0", date(2026, 4, 15), None)],
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
    """(run, lat, lon) observed mean over the runs' lead days [a, b); NaN where a day is missing or the cell invalid."""
    out = np.full((len(inits), *pr.shape[1:]), np.nan, dtype="float32")
    for r, d in enumerate(inits):
        rows = [idx.get(d + timedelta(days=k)) for k in range(a, b)]
        if any(x is None for x in rows):
            continue
        out[r] = pr[rows].mean(axis=0)  # NaN in any day -> NaN mean: one invalid day voids the cell
    return out


def bh_fdr(p: np.ndarray, q: float) -> np.ndarray:
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


SEASON_OF = {m: k for k, v in SEASONS.items() for m in v}

RATIO_MIN_LATE, RATIO_CLIP = 0.1, (0.25, 4.0)


def truth_terms(li, lp, fi, fp, rng):
    """{season: {ratio, se_ratio, days}} from the days IMERG Late and Final share; {} without Final.
    ratio = sum(Final) / sum(Late) per box: the truth is Late SCALED by its season's ratio (an ADDITIVE
    shift, Late minus the seasonal mean of Late - Final, makes the "observed" mean NEGATIVE over the dry-season high Andes and
    inflates the bias there). Ratio 1 where Late averages under 0.1 mm/day, clipped to [0.25, 4]; its se from the 21-day block
    bootstrap of both sums."""
    if not fi:
        return {}
    days = sorted(set(li) & set(fi))
    shape = lp.shape[1:]  # the boxes the archive was loaded on (1° or 2°)
    out, d0, cells = {}, days[0], shape[0] * shape[1]
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
        out[sk] = {"ratio": r.reshape(shape), "se_ratio": se_r.reshape(shape), "days": len(dd)}
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
    v = np.zeros(L.shape[1:])
    for x in set(seas):
        k = [i for i, y in enumerate(seas) if y == x]
        with np.errstate(invalid="ignore"):
            mL = np.nanmean(L[k], axis=0)
        v += (len(k) / len(seas) * np.nan_to_num(mL) * D[x]["se_ratio"]) ** 2
    return La, np.sqrt(v)


# A cell whose normal week's upper tercile is under 2 mm (dry season, desert) has no testable tercile: most normal weeks
# there are (nearly) dry, so "below normal" cannot happen and one wet member makes "above normal".
DRY_Q67 = 2.0  # mm per week


def clim_at(C, start: date) -> dict:
    """The GPCP normal 7-day totals (gpcp_clim.py: mean, q33, q50, q67) for the week starting on `start`, linear between
    the pentads."""
    doy = start.timetuple().tm_yday
    if calendar.isleap(start.year) and doy >= 60:  # gpcp_clim.py's 365-day calendar (1 March = 60, 29 Feb dropped):
        doy = 59.5 if doy == 60 else doy - 1  # 29 Feb sits halfway between 28 Feb and 1 March
    doy = min(doy, 365)
    s = C["start_doy"]
    p0 = int((doy - 1) // 5)
    p1 = (p0 + 1) % len(s)
    span = (s[p1] - s[p0]) % 365 or 5
    fr = ((doy - s[p0]) % 365) / span
    return {k: (1 - fr) * C[k][p0] + fr * C[k][p1] for k in ("mean", "q33", "q50", "q67")}
