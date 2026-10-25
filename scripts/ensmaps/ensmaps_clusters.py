#!/usr/bin/env python3
"""Four-ensemble clusters of 500 hPa height and sea-level pressure for ens-anomalies.html: the members of the newest
IFS-ENS, AIFS-ENS, GEFS and GEPS runs pooled on the same valid dates and grouped into the few scenarios the spread of the
grand ensemble supports - or one, when it supports none. No download: each ensmaps job leaves its members' 2.5-deg
anomalies in the Actions cache (save_fields, from ensmaps.py), and `run` clusters the newest run of each model there.

  fields    each member's daily-mean 500 hPa height and sea-level pressure anomaly against the ensmaps 1991-2020 normals,
            on the 2.5-deg NCEP grid 20-90N (regime_core.box25); the pressure anomaly is 0 where the model's ground is
            above 1500 m. A model's lead-dependent drift is removed by the regimes' rule (regimes_ecmwf), on these
            fields: the archive of its 00Z runs' ensemble-mean anomalies (DRIFT_KEEP runs, the Actions cache), lead-L
            forecast less the day-1 forecast of the run started on its valid day, shrunk n / (n + 10), at the leads with
            >= 10 verified runs
  windows   by VALID date from the newest run's first full day: days 3-5, 6-10 and 11-15, each the mean over its days,
            cut to the days every model in it holds (a model holding fewer than 3 of them is left out of the window)
  domains   regime_core's two sector boxes (Euro-Atlantic, North America and Pacific) and the hemisphere 20-90N
  vector    each field divided by its domain-mean climatological standard deviation for the window's length and time of
            year (SD_FILE: ERA5 1991-2020 from the store, +-15 days), weighted by sqrt(cos lat), the two concatenated
  weights   each model's members share 1/M of the weight (M models in the window), so 51 members count as much as 21
  EOFs      of the weighted deviations from the grand-ensemble mean, as many as explain VAR_KEEP of that spread
  k-means   weighted, k = 1..K_MAX, N_START starts (k-means++), in the EOFs' space
  choice    k > 1 only when (1) its within-cluster variance reduction beats the 95th percentile of N_NULL synthetic Gaussian
            ensembles with the same covariance and weights, (2) every cluster holds >= MIN_W of the weight, (3) the same
            k is chosen in >= STABLE of N_BOOT bootstrap resamples of the members (within each model); among the passing
            k the smallest whose gap (null less observed log within-variance) is at least the next k's less one null
            standard deviation (the gap statistic's rule). Otherwise one scenario.
  outputs   per window and domain: each cluster's probability (its weight) and each model's share of its own members in
            it; the nearest regime of each sector (pattern correlation of the cluster's mean 500 hPa anomaly with
            regime_core's centroids of the season, "no regime" below 0.25); how members move from window to window; a
            permutation chi-square test of model x cluster independence, naming the models a cluster over- or
            under-represents (adjusted residual beyond 2.58) when it fails at 5 %; polar stereographic Northern
            Hemisphere maps per cluster, 500 hPa height and sea-level pressure, the anomaly (or the standardized anomaly:
            ensmaps_ref.Spread of the window's length) shaded where at least sign_k(n) of its n members agree on its
            sign, dots where its mean differs from the grand ensemble's (N_PERM random member sets of its size,
            Benjamini-Hochberg at FDR_Q)

    python scripts/ensmaps/ensmaps_clusters.py sd                    # once: SD_FILE from $ERA5_STORE
    python scripts/ensmaps/ensmaps_clusters.py run --fields f/*.npz --site . --lists /tmp/lists
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "lib"), str(ROOT / "scripts" / "regimes")]
import ensmaps_ref as R

SD_FILE = HERE / "reference" / "ensmaps_cluster_sd.npz"
LAT = np.arange(90.0, 19.99, -2.5)  # 29 rows, 90N..20N (the NCEP grid's)
LON = np.arange(0.0, 360.0, 2.5)  # 144, from 0E (regime_core's order)
NROW = LAT.size
WINDOW_DAYS = (3, 4, 5)  # the windows' lengths the SD table holds
HALF = 15  # +- days of the time of year
WINDOWS = (("d3_5", 3, 5), ("d6_10", 6, 10), ("d11_15", 11, 15))
K_MAX, N_START, N_NULL, N_BOOT, STABLE, MIN_W, VAR_KEEP = 5, 100, 100, 50, 0.6, 0.05, 0.80
N_START_FAST = 10  # starts for the null and bootstrap ensembles
N_PERM, FDR_Q = 500, 0.10
DRIFT_MIN, DRIFT_K, DRIFT_KEEP, DRIFT_DAYS = 10, 10.0, 60, 15
MODEL_ORDER = ("ifs", "aifs", "gefs", "geps")
LABEL = {"ifs": "IFS-ENS", "aifs": "AIFS-ENS", "gefs": "GEFS", "geps": "GEPS"}
DATA = "assets/models/data/ensmaps/clusters"
SEED = 31


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def domains() -> dict:
    """{key: (label, row mask over LAT, column mask over LON)}: the regimes' sector boxes and the hemisphere."""
    import regime_core as RC

    lat, lon180 = LAT, (LON + 180) % 360 - 180
    out = {"nh": ("Northern Hemisphere", np.ones(LAT.size, bool), np.ones(LON.size, bool))}
    for sk, spec in RC.SECTORS.items():
        la = (lat <= spec["lat"][0] + 1e-6) & (lat >= spec["lat"][1] - 1e-6)
        lo = (lon180 >= spec["lon"][0] - 1e-6) & (lon180 <= spec["lon"][1] + 1e-6)
        out[sk] = (spec["label"], la, lo)
    return out


def bilinear25(coef: np.ndarray) -> np.ndarray:
    """(k, LATU, LONU) curves on the store's 1.5-deg grid (columns from 180W) -> (k, NROW, 144) on this grid."""
    from scipy.ndimage import map_coordinates

    rows = np.broadcast_to(((R.LATU[0] - LAT) / 1.5)[:, None], (NROW, LON.size)).ravel()
    cols = np.broadcast_to((((LON - R.LONU[0]) % 360) / 1.5)[None], (NROW, LON.size)).ravel()
    out = np.empty((coef.shape[0], NROW, LON.size), np.float32)
    for k, c in enumerate(coef):
        w = np.concatenate([c, c[:, :1]], axis=1).astype(np.float64)
        out[k] = map_coordinates(w, [rows, cols], order=1, mode="nearest").reshape(NROW, LON.size)
    return out


def normals25(field: str) -> R.Normals:
    N = R.Normals.upper(field)
    return R.Normals(bilinear25(N.coef), LAT, LON, N.meta)


def spread25(field: str) -> R.Spread:
    S = R.Spread.upper(field)
    return R.Spread({n: bilinear25(c) for n, c in S.coef.items()}, LAT, LON)


# ── the members' fields, written by each ensmaps job ──────────────────────────────────────────────────────────────────
def update_drift(path: Path | None, first: dt.date, cycle: str, mean: np.ndarray) -> tuple[np.ndarray, list]:
    """The model's archive of 00Z ensemble-mean anomalies (days, 2, NROW, 144): this run's added when it is a 00Z run;
    -> (the drift to remove per day, the corrected days): regimes_ecmwf's rule on these fields."""
    runs: dict = {}
    if path is not None and path.exists():
        with np.load(path) as z:
            runs = {str(d): z["f"][i].astype(np.float32) for i, d in enumerate(z["dates"])}
    if cycle.endswith("00"):
        runs[f"{first:%Y%m%d}"] = mean[:DRIFT_DAYS].astype(np.float32)
        keep = sorted(runs)[-DRIFT_KEEP:]
        runs = {k: runs[k] for k in keep}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, dates=np.array(keep), f=np.stack([runs[k] for k in keep]).astype(np.float16))
    acc = np.zeros((DRIFT_DAYS, *mean.shape[1:]))
    n = np.zeros(DRIFT_DAYS)
    for d0, f in runs.items():
        t0 = dt.datetime.strptime(d0, "%Y%m%d").date()
        for L in range(1, f.shape[0]):
            v = f"{t0 + dt.timedelta(days=L):%Y%m%d}"
            if v >= f"{first:%Y%m%d}" or v not in runs:
                continue
            acc[L] += f[L] - runs[v][0]
            n[L] += 1
    drift = np.zeros(mean.shape, np.float32)
    use = [L for L in range(min(DRIFT_DAYS, mean.shape[0])) if n[L] >= DRIFT_MIN]
    for L in use:
        drift[L] = n[L] / (n[L] + DRIFT_K) * acc[L] / n[L]
    return drift, [L + 1 for L in use]


def save_fields(out: Path, mod, cycle: str, init, days: list, Z25, M25, ok, high25, drift_path: Path | None) -> None:
    """The members' 500 hPa height and sea-level pressure anomalies on the 2.5-deg grid 20-90N for the clusters, drift
    removed where the rule allows; float16."""
    dates = [d for d, _ in days]
    t = [d + pd.Timedelta(hours=12) for d in dates]
    Nz, Nm = normals25("z500"), normals25("msl")
    z = Z25[:, :, :NROW] - np.stack([Nz.at([x])[0] for x in t])[None]
    m = M25[:, :, :NROW] - np.stack([Nm.at([x])[0] for x in t])[None]
    m[:, :, high25[:NROW]] = 0.0
    full = ok.all(1)
    with np.errstate(invalid="ignore"):
        mean = np.stack([np.nanmean(z[full], 0), np.nanmean(m[full], 0)], axis=1)  # (days, 2, rows, lon)
    drift, used = update_drift(drift_path, dates[0].date(), cycle, mean)
    z -= drift[None, :, 0]
    m -= drift[None, :, 1]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        model=mod.key,
        cycle=cycle,
        init=f"{init:%Y-%m-%dT%H}:00Z",
        valid=np.array([f"{d:%Y-%m-%d}" for d in dates]),
        z=z.astype(np.float16),
        m=m.astype(np.float16),
        ok=ok,
        high=high25[:NROW],
        drift_days=np.array(used, int),
    )
    log(f"  cluster fields: {out} ({out.stat().st_size / 1e6:.1f} MB); drift removed on days {used or 'none yet'}")


# ── clustering ────────────────────────────────────────────────────────────────────────────────────────────────────────
def load(paths: list[Path]) -> list[dict]:
    runs = {}
    for p in paths:
        with np.load(p) as z:
            r = {k: z[k] for k in z.files}
        key = str(r["model"])
        r["key"], r["cycle"], r["init"] = key, str(r["cycle"]), str(r["init"])
        r["valid"] = [str(v) for v in r["valid"]]
        if key not in runs or r["cycle"] > runs[key]["cycle"]:
            runs[key] = r
    return [runs[k] for k in MODEL_ORDER if k in runs]


def window_dates(models: list[dict], a: int, b: int) -> tuple[list[str], list[dict]]:
    """The window's dates (days a..b from the newest first day, cut to what every model in it holds) and its models."""
    first = max(r["valid"][0] for r in models)
    t0 = dt.date.fromisoformat(first)
    want = [f"{t0 + dt.timedelta(days=d - 1):%Y-%m-%d}" for d in range(a, b + 1)]
    ins = [r for r in models if sum(d in r["valid"] for d in want) >= 3]
    dates = [d for d in want if all(d in r["valid"] for r in ins)]
    if len(dates) < 3:
        return [], []
    return dates, ins


def members(ins: list[dict], dates: list[str]):
    """(model index, member index, z (N, NROW, 144), m, weights (N,)) of the members complete on every date."""
    mi, ii, zs, ms, ws = [], [], [], [], []
    for k, r in enumerate(ins):
        cols = [r["valid"].index(d) for d in dates]
        full = np.flatnonzero(r["ok"][:, cols].all(1))
        mi += [k] * full.size
        ii += list(full)
        zs.append(r["z"][full][:, cols].astype(np.float32).mean(1))
        ms.append(r["m"][full][:, cols].astype(np.float32).mean(1))
        ws.append(np.full(full.size, 1.0 / (len(ins) * max(full.size, 1))))
    return np.array(mi), np.array(ii), np.concatenate(zs), np.concatenate(ms), np.concatenate(ws)


def sd_of(field: str, n: int, dk: str, dates: list[str]) -> float:
    with np.load(SD_FILE) as z:
        s = z[f"{field}_{min(max(n, 3), 5)}_{dk}"]
    doys = [
        (pd.Timestamp(d).dayofyear - (1 if pd.Timestamp(d).is_leap_year and pd.Timestamp(d).month > 2 else 0))
        for d in dates
    ]
    return float(np.mean([s[min(d, 365) - 1] for d in doys]))


def vectors(z, m, dk: str, dates: list[str]) -> np.ndarray:
    """(N, P): the domain's two fields over their climatological sd, weighted by sqrt(cos lat), concatenated."""
    _, la, lo = domains()[dk]
    w = np.sqrt(np.cos(np.deg2rad(LAT[la])))[:, None]
    n = len(dates)
    parts = [
        (f[:, la][:, :, lo] * w / sd_of(name, n, dk, dates)).reshape(f.shape[0], -1)
        for name, f in (("z500", z), ("msl", m))
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def pca(X: np.ndarray, w: np.ndarray):
    """(scores (N, r), variances (r,), explained share): EOFs of the weighted deviations, VAR_KEEP of the spread."""
    W = w / w.sum()
    D = X - (W[:, None] * X).sum(0)
    _, S, Vt = np.linalg.svd(np.sqrt(W)[:, None] * D, full_matrices=False)
    var = S**2
    share = np.cumsum(var) / var.sum()
    r = int(np.searchsorted(share, VAR_KEEP) + 1)
    return D @ Vt[:r].T, var[:r], float(share[r - 1])


def kmeans(Y: np.ndarray, w: np.ndarray, k: int, starts: int, rng) -> tuple[np.ndarray, float]:
    """Weighted k-means (k-means++ starts): (labels, within-cluster weighted sum of squares) of the best start."""
    N = Y.shape[0]
    p0 = w / w.sum()
    if k == 1:
        c = (p0[:, None] * Y).sum(0)
        return np.zeros(N, int), float((w * ((Y - c) ** 2).sum(1)).sum())
    best = (np.inf, None)
    for _ in range(starts):
        C = [Y[rng.choice(N, p=p0)]]
        for _ in range(1, k):
            d2 = np.min([((Y - c) ** 2).sum(1) for c in C], axis=0)
            q = w * d2
            C.append(Y[rng.choice(N, p=q / q.sum())] if q.sum() > 0 else Y[rng.choice(N)])
        C = np.array(C)
        lab = None
        for _ in range(100):
            d = ((Y[:, None, :] - C[None]) ** 2).sum(2)
            new = d.argmin(1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            for j in range(k):
                s = lab == j
                if s.any():
                    C[j] = (w[s, None] * Y[s]).sum(0) / w[s].sum()
        within = float((w * ((Y - C[lab]) ** 2).sum(1)).sum())
        if within < best[0]:
            best = (within, lab.copy())
    return best[1], best[0]


def reductions(Y, w, starts, rng):
    """(1 - W_k / W_1 for k = 1..K_MAX, labels per k, the smallest cluster's weight share per k)."""
    W1 = kmeans(Y, w, 1, 1, rng)[1]
    red, labs, minw = [0.0], [np.zeros(Y.shape[0], int)], [1.0]
    for k in range(2, K_MAX + 1):
        lab, W = kmeans(Y, w, k, starts, rng)
        red.append(1 - W / W1)
        labs.append(lab)
        minw.append(min(w[lab == j].sum() for j in range(k)) / w.sum())
    return np.array(red), labs, np.array(minw)


def pick(red: np.ndarray, minw: np.ndarray, null: np.ndarray) -> int:
    """The k of the gates (1) and (2) and the gap rule (see the module docstring), 1 if none."""
    p95 = np.percentile(null, 95, axis=0)
    lw = np.log(np.clip(1 - null, 1e-9, None))
    gap = lw.mean(0) - np.log(np.clip(1 - red, 1e-9, None))
    s = lw.std(0) * np.sqrt(1 + 1 / null.shape[0])
    for k in range(2, K_MAX + 1):
        i = k - 1
        if red[i] > p95[i] and minw[i] >= MIN_W and (k == K_MAX or gap[i] >= gap[i + 1] - s[i + 1]):
            return k
    return 1


def choose(Y, w, var, mi, rng) -> tuple[int, list, dict]:
    """(k, labels for each k, the evidence): the observed reductions, the synthetic null, the bootstrap."""
    red, labs, minw = reductions(Y, w, N_START, rng)
    null = np.empty((N_NULL, K_MAX))
    for b in range(N_NULL):  # Gaussian ensembles with the same covariance and weights
        Ys = rng.standard_normal(Y.shape) * np.sqrt(var)[None]
        null[b] = reductions(Ys, w, N_START_FAST, rng)[0]
    k = pick(red, minw, null)
    stab = None
    if k > 1:
        hits = 0
        for _ in range(N_BOOT):  # members resampled within each model
            idx = np.concatenate([rng.choice(np.flatnonzero(mi == j), (mi == j).sum()) for j in np.unique(mi)])
            rb, _, mb = reductions(Y[idx], w[idx], N_START_FAST, rng)
            hits += pick(rb, mb, null) == k
        stab = hits / N_BOOT
        if stab < STABLE:
            k = 1
    ev = {
        "reduction": np.round(red, 3).tolist(),
        "null_p95": np.round(np.percentile(null, 95, axis=0), 3).tolist(),
        "min_weight": np.round(minw, 3).tolist(),
        "stability": None if stab is None else round(stab, 2),
    }
    return k, labs, ev


def order(lab: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Labels renumbered by weight, the largest cluster 0."""
    wt = [w[lab == j].sum() for j in range(k)]
    rank = np.argsort(wt)[::-1]
    return np.argsort(rank)[lab]


def independence(mi: np.ndarray, lab: np.ndarray, k: int, nm: int, rng) -> dict:
    """Permutation chi-square test of model x cluster; the models each cluster over- or under-represents (adjusted
    residual beyond 2.58) when p < 0.05."""
    if k < 2 or nm < 2:
        return {"p": None, "flags": [[] for _ in range(k)]}

    def chi2(lb):
        O = np.array([[np.sum((mi == i) & (lb == j)) for j in range(k)] for i in range(nm)], float)
        E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
        return ((O - E) ** 2 / np.where(E > 0, E, 1)).sum(), O, E

    x0, O, E = chi2(lab)
    perm = np.array([chi2(rng.permutation(lab))[0] for _ in range(2000)])
    p = float((1 + (perm >= x0).sum()) / (1 + perm.size))
    flags = [[] for _ in range(k)]
    if p < 0.05:
        N = O.sum()
        ri, cj = O.sum(1, keepdims=True) / N, O.sum(0, keepdims=True) / N
        res = (O - E) / np.sqrt(np.where(E > 0, E, 1) * (1 - ri) * (1 - cj))
        for i in range(nm):
            for j in range(k):
                if abs(res[i, j]) > 2.58:
                    flags[j].append({"i": i, "over": bool(res[i, j] > 0)})
    return {"p": round(p, 4), "flags": flags}


def nearest_regimes(zc: np.ndarray, season: str, sks: list[str]) -> dict:
    """{sector: (name, r)}: the regime_core centroid nearest the cluster's mean 500 hPa anomaly; "no regime" below 0.25."""
    import regime_core as RC

    ref = RC.Ref()
    g = np.full((73, 144), np.nan)
    g[:NROW] = zc
    out = {}
    for sk in sks:
        sec, _, _ = RC.sector_cut(g, RC.SECTORS[sk])
        lab, corr = ref.classify(sec[None], sk, season)
        out[sk] = {"name": ref.name(sk, season, int(lab[0])), "r": round(float(corr[0]), 2)}
    return out


def differs(x: np.ndarray, w: np.ndarray, sel: np.ndarray, rng) -> np.ndarray:
    """Where the weighted mean of the members `sel` differs from the grand ensemble's: N_PERM random member sets of the
    same size, two-sided, Benjamini-Hochberg at FDR_Q over the points. x (N, P)."""
    from quantmet.stats import benjamini_hochberg

    N, n = x.shape[0], int(sel.sum())
    g = (w[:, None] * x).sum(0) / w.sum()
    obs = (w[sel, None] * x[sel]).sum(0) / w[sel].sum() - g
    S = np.zeros((N_PERM, N))
    for p in range(N_PERM):
        idx = rng.choice(N, n, replace=False)
        S[p, idx] = w[idx] / w[idx].sum()
    null = S @ x - g[None]
    pv = (1 + (np.abs(null) >= np.abs(obs)[None]).sum(0)) / (1 + N_PERM)
    return benjamini_hochberg(pv, alpha=FDR_Q)


def run(paths: list[Path], site: Path, lists: Path | None, manifest_in: Path | None) -> int:
    import regime_core as RC

    import ensmaps as E

    models = load(paths)
    if not models:
        log("::notice::no ensemble's fields in the cache yet: no clusters")
        return 0
    rng = np.random.default_rng(SEED)
    stamp = max(r["cycle"] for r in models)
    outdir = site / DATA / stamp
    if outdir.exists():
        import shutil

        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    log("clusters of " + ", ".join(f"{LABEL[r['key']]} {r['cycle']}" for r in models))
    N25 = {f: normals25(f) for f in ("z500", "msl")}
    S25 = {f: spread25(f) for f in ("z500", "msl")}
    doms = domains()
    figs = Figures(outdir)
    result, prev_lab = {}, {}
    windows = []
    for wk, a, b in WINDOWS:
        dates, ins = window_dates(models, a, b)
        if not dates:
            continue
        mi, ii, z, m, w = members(ins, dates)
        t = [pd.Timestamp(d) + pd.Timedelta(hours=12) for d in dates]
        season = RC.season_of_date(dt.date.fromisoformat(dates[len(dates) // 2]))
        nz = np.mean([N25["z500"].at([x])[0] for x in t], axis=0)
        nmsl = np.mean([N25["msl"].at([x])[0] for x in t], axis=0)
        sdz, sdm = S25["z500"].sd(t, min(len(t), 5)), S25["msl"].sd(t, min(len(t), 5))
        high = np.any([r["high"] for r in ins], axis=0)
        windows.append({"key": wk, "days": [a, b], "dates": dates, "models": [r["key"] for r in ins]})
        result[wk] = {}
        for dk, (dlabel, _, _) in doms.items():
            X = vectors(z, m, dk, dates)
            Y, var, share = pca(X, w)
            k, labs, ev = choose(Y, w, var, mi, rng)
            lab = order(labs[k - 1], w, k)
            ind = independence(mi, lab, k, len(ins), rng)
            cl = []
            for j in range(k):
                sel = lab == j
                cz = (w[sel, None, None] * z[sel]).sum(0) / w[sel].sum()
                cm = (w[sel, None, None] * m[sel]).sum(0) / w[sel].sum()
                sig_z = (
                    differs(z.reshape(len(w), -1), w, sel, rng).reshape(cz.shape) if k > 1 else np.zeros(cz.shape, bool)
                )
                sig_m = (
                    differs(m.reshape(len(w), -1), w, sel, rng).reshape(cm.shape) if k > 1 else np.zeros(cm.shape, bool)
                )
                n = int(sel.sum())
                kk = E.sign_k(n) if n >= 6 else n + 1
                agz = (np.sign(z[sel]) == np.sign(cz)[None]).sum(0) >= kk
                agm = (np.sign(m[sel]) == np.sign(cm)[None]).sum(0) >= kk
                shares = {
                    r["key"]: round(float((sel & (mi == i)).sum() / max((mi == i).sum(), 1)), 3)
                    for i, r in enumerate(ins)
                }
                info = {
                    "p": round(float(w[sel].sum() / w.sum()), 3),
                    "n": n,
                    "k": kk,
                    "shares": shares,
                    "regimes": nearest_regimes(cz, season, ["ea", "na"] if dk == "nh" else [dk]),
                    "flags": [{"model": ins[f["i"]]["key"], "over": f["over"]} for f in ind["flags"][j]],
                }
                title = figure_title(info, j, k, dlabel, dates, ins, wk)
                info["img"] = {
                    v: figs.draw(f"{wk}_{dk}_{j}_{v}", dk, title, v, cz, cm, nz + cz, nmsl + cm, agz, agm, sig_z, sig_m, high,
                                 sdz, sdm)
                    for v in ("phys", "sigma")
                }  # fmt: skip
                cl.append(info)
            res = {"k": k, "eofs": int(Y.shape[1]), "explained": round(share, 3), "evidence": ev, "independence": ind["p"],
                   "clusters": cl, "members": len(w)}  # fmt: skip
            key = {(int(a_), int(b_)): j for a_, b_, j in zip(mi, ii, lab)}
            p = prev_lab.get(dk)
            if p is not None:  # members' moves from the previous window's clusters (weight shares)
                kp = max(p["lab"].values()) + 1
                T = np.zeros((kp, k))
                for (mm, i), j in key.items():
                    if (mm_key := (ins[mm]["key"], i)) in p["lab_by_model"]:
                        T[p["lab_by_model"][mm_key], j] += w[(mi == mm) & (ii == i)].sum()
                res["from_previous"] = {"window": p["window"], "flow": np.round(T / max(T.sum(), 1e-12), 3).tolist()}
            prev_lab[dk] = {
                "window": wk,
                "lab": key,
                "lab_by_model": {(ins[mm]["key"], i): j for (mm, i), j in key.items()},
            }
            result[wk][dk] = res
            log(f"  {wk} {dk}: {len(w)} members, {Y.shape[1]} EOFs ({share:.0%}), k = {k} "
                f"(reductions {ev['reduction'][1:]}, null p95 {ev['null_p95'][1:]}, stability {ev['stability']})")  # fmt: skip
    entry = {
        "stamp": stamp,
        "made": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "runs": {
            r["key"]: {"cycle": r["cycle"], "init": r["init"], "drift_days": [int(x) for x in r["drift_days"]]}
            for r in models
        },
        "windows": windows,
        "domains": {k: v[0] for k, v in doms.items()},
        "results": result,
        "files": f"{DATA}/{stamp}",
        "rules": {"k_max": K_MAX, "min_weight": MIN_W, "stable": STABLE, "var_keep": VAR_KEEP, "fdr_q": FDR_Q},
    }
    mpath = site / "assets/models/ensmaps_clusters.json"
    try:
        old = json.loads((manifest_in or mpath).read_text())
    except Exception:  # noqa: BLE001
        old = {}
    drop = [old["stamp"]] if old.get("stamp") and old["stamp"] != stamp else []
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(entry, separators=(",", ":")))
    if lists:
        lists.mkdir(parents=True, exist_ok=True)
        (lists / "publish.txt").write_text(f"{DATA}/{stamp}\n")
        (lists / "prune.txt").write_text("".join(f"{DATA}/{s}\n" for s in drop))
        (lists / "floor.txt").write_text(f"{DATA}:{stamp}\n")
    log(f"  {figs.n} figures, {figs.bytes / 1e6:.1f} MB -> {outdir}")
    return 0


def figure_title(info: dict, j: int, k: int, dlabel: str, dates: list[str], ins: list[dict], wk: str) -> list[str]:
    d0, d1 = (pd.Timestamp(x) for x in (dates[0], dates[-1]))
    span = f"{d0:%a %b %-d} – {d1:%a %b %-d}"
    head = (
        f"Scenario {j + 1} of {k}: {info['p']:.0%} of the four ensembles"
        if k > 1
        else "One scenario: the four ensembles' mean"
    )
    return [
        f"{head} · {dlabel} · {span}",
        "Members in it: " + " · ".join(f"{LABEL[m]} {s:.0%}" for m, s in info["shares"].items()),
        "Runs: " + " · ".join(f"{LABEL[r['key']]} {pd.Timestamp(r['init'][:13]):%b %-d %H}Z" for r in ins),
    ]


# ── the figures: polar stereographic Northern Hemisphere, 500 hPa height and sea-level pressure ─────────────────────
class Figures:
    DIV = (  # the page's diverging scale (ens-anomalies.html DIV)
        "#2b1a5e", "#23449c", "#2f70b6", "#5f9ed2", "#9cc6e4", "#d3e5f1", "#f3f4f1", "#f6f2ea", "#f8dcc4", "#f3ad85",
        "#e37953", "#c84432", "#9c1c2a", "#5c0a26",
    )  # fmt: skip

    def __init__(self, outdir: Path):
        import matplotlib

        matplotlib.use("Agg")
        self.outdir, self.n, self.bytes = outdir, 0, 0
        geo = json.loads((ROOT / "assets" / "models" / "geo_polar_coarse.json").read_text())
        self.coast = []
        for name in ("coast", "lakes"):
            L = geo["layers"][name]
            for ln in L["lines"]:
                x, y, pts = ln[0], ln[1], []
                for i in range(len(ln) // 2):
                    if i:
                        x, y = x + ln[2 * i], y + ln[2 * i + 1]
                    pts.append((x / L["q"], y / L["q"]))
                self.coast.append(np.array(pts))

    @staticmethod
    def xy(lon, lat):
        rho = np.tan(np.pi / 4 - np.deg2rad(lat) / 2) * 2
        lam = np.deg2rad(lon)
        return rho * np.sin(lam), -rho * np.cos(lam)

    def panel(self, ax, field, shade, agree, absf, sig, high, lim, step, ci, box):
        from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

        lon = np.r_[LON, 360.0]
        L, A = np.meshgrid(lon, LAT)
        X, Y = self.xy(L, A)
        cyc = lambda a: np.concatenate([a, a[:, :1]], axis=1)  # noqa: E731
        r20 = self.xy(0, 20)[1] * -1
        th = np.linspace(0, 2 * np.pi, 361)
        ax.fill(r20 * np.cos(th), r20 * np.sin(th), color="#e4e4df", zorder=0)
        g = np.where(agree & ~(high if field == "msl" else np.zeros_like(high)), shade, np.nan)
        levels = np.arange(-lim, lim + step / 2, step)
        cmap = LinearSegmentedColormap.from_list("div", self.DIV, N=len(levels) - 1)
        cs = ax.contourf(X, Y, cyc(np.clip(g, -lim + 1e-6, lim - 1e-6)), levels=levels, cmap=cmap,
                         norm=BoundaryNorm(levels, cmap.N), zorder=1)  # fmt: skip
        if field == "msl":
            ax.contourf(X, Y, cyc(high.astype(float)), levels=[0.5, 1.5], colors=["#cdcfc9"], zorder=1)
        lv = np.arange(np.floor(np.nanmin(absf) / ci) * ci, np.nanmax(absf) + ci, ci)
        c = ax.contour(X, Y, cyc(absf), levels=lv, colors="#1f261c", linewidths=0.6, zorder=3)
        ax.clabel(c, fontsize=6.5, fmt=(lambda v: f"{v / 10:.0f}") if field == "z500" else "%d", inline_spacing=1)
        if sig.any():
            sx, sy = self.xy(*np.meshgrid(LON, LAT))
            ax.scatter(sx[sig], sy[sig], s=1.2, c="#1f261c", linewidths=0, zorder=4)
        for pts in self.coast:
            keep = pts[:, 1] >= 18
            if keep.sum() >= 2:
                px, py = self.xy(pts[:, 0], pts[:, 1])
                ax.plot(np.where(keep, px, np.nan), np.where(keep, py, np.nan), color="#2d312d", lw=0.45, zorder=2)
        for la in (40, 60, 80):
            ax.plot(*self.xy(np.linspace(0, 360, 181), la), color="#6f6b64", lw=0.35, ls=(0, (3, 3)), zorder=2)
        if box is not None:
            lo = np.linspace(box[0], box[1], 61)
            bl = np.r_[lo, np.full(20, box[1]), lo[::-1], np.full(20, box[0])]
            ba = np.r_[
                np.full(61, box[3]),
                np.linspace(box[3], box[2], 20),
                np.full(61, box[2]),
                np.linspace(box[2], box[3], 20),
            ]
            ax.plot(*self.xy(bl, ba), color="#1f261c", lw=1.4, zorder=5)
        ax.set_xlim(-r20 * 1.02, r20 * 1.02)
        ax.set_ylim(-r20 * 1.02, r20 * 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
        return cs

    def draw(self, name, dk, title, variant, cz, cm, absz, absm, agz, agm, sig_z, sig_m, high, sdz, sdm) -> str:
        import matplotlib.pyplot as plt
        from PIL import Image

        import regime_core as RC

        box = None
        if dk in RC.SECTORS:
            sp = RC.SECTORS[dk]
            box = (sp["lon"][0], sp["lon"][1], sp["lat"][1], sp["lat"][0])
        fig = plt.figure(figsize=(9.6, 5.6), dpi=100)
        fig.text(0.012, 0.975, title[0], fontsize=10.5, fontweight="bold", va="top", color="#1f261c")
        fig.text(0.012, 0.935, title[1], fontsize=9, va="top", color="#3d4540")
        fig.text(0.012, 0.9, title[2], fontsize=9, va="top", color="#6f6b64")
        sig = variant == "sigma"
        specs = (
            ("z500", cz / sdz if sig else cz, agz, absz, sig_z, 3 if sig else 300, 0.25 if sig else 25, 60,
             "500 hPa height: contours every 6 dam", "σ" if sig else "m"),
            ("msl", cm / sdm if sig else cm, agm, absm, sig_m, 3 if sig else 24, 0.25 if sig else 2, 4,
             "sea-level pressure: isobars every 4 hPa", "σ" if sig else "hPa"),
        )  # fmt: skip
        for i, (f, shade, ag, absf, sg, lim, step, ci, cap, unit) in enumerate(specs):
            ax = fig.add_axes([0.005 + 0.5 * i, 0.1, 0.49, 0.76])
            cs = self.panel(ax, f, shade, ag, absf, sg, high, lim, step, ci, box)
            cax = fig.add_axes([0.07 + 0.5 * i, 0.075, 0.36, 0.022])
            cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(-lim, lim, 7))
            cb.ax.tick_params(labelsize=7.5, length=2)
            what = "standardized anomaly" if sig else "anomaly"
            cb.set_label(f"{cap}; {what}, {unit}, shaded where the members agree", fontsize=7.5, color="#3d4540")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        out = io.BytesIO()
        im.save(out, "WEBP", quality=86, method=4)
        b = out.getvalue()
        fn = f"{name}_{hashlib.sha1(b).hexdigest()[:8]}.webp"
        (self.outdir / fn).write_bytes(b)
        self.n += 1
        self.bytes += len(b)
        return fn


# ── the climatological spread (offline, from the ERA5 store) ─────────────────────────────────────────────────────────
def build_sd(out: Path) -> None:
    """SD_FILE: per field, window length (WINDOW_DAYS), domain and day of the common year (1..365), the domain-mean
    standard deviation of the window-mean anomaly (against the ensmaps normals' curve), ERA5 1991-2020, all years' days
    within +-HALF days, area weighted (cos lat), on the store's 1.5-deg grid."""
    from scipy.ndimage import uniform_filter1d

    north = R.LATU >= 19.9
    lat15, lon15_0 = R.LATU[north], (R.LONU + 360) % 360
    dom15 = {}
    for k, (_, la, lo) in domains().items():  # the same boxes on the 1.5-deg grid
        rows = (lat15 <= LAT[la].max() + 1e-6) & (lat15 >= LAT[la].min() - 1e-6)
        cl, lsel = (lon15_0 + 180) % 360 - 180, (LON[lo] + 180) % 360 - 180
        cols = (cl >= lsel.min() - 1e-6) & (cl <= lsel.max() + 1e-6) if k != "nh" else np.ones(R.LONU.size, bool)
        dom15[k] = (rows, cols)
    res = {}
    for field, (var, _, scale) in R.UPPER.items():
        N = R.Normals.upper(field)
        parts, doys = [], []
        for a, t in R.store_years(var, scale):
            parts.append((a[:, north] - N.at(t + pd.Timedelta(hours=12))[:, north]).astype(np.float32))
            doys.append(np.array([d.dayofyear - (1 if d.is_leap_year and d.month > 2 else 0) for d in t]))
        an, doy = np.concatenate(parts), np.concatenate(doys)
        cw = np.cos(np.deg2rad(lat15))
        for n in WINDOW_DAYS:
            wm = uniform_filter1d(an, n, axis=0, mode="nearest")  # centred n-day means
            for dk, (rows, cols) in dom15.items():
                sub = wm[:, rows][:, :, cols]
                w = cw[rows][:, None] * np.ones(cols.sum())
                sd = np.empty(365)
                for d in range(1, 366):
                    m = np.abs((doy - d + 182) % 365 - 182) <= HALF
                    sd[d - 1] = np.sqrt((sub[m].var(0) * w).sum() / w.sum())
                res[f"{field}_{n}_{dk}"] = sd.astype(np.float32)
                log(f"  {field} {n}-day {dk}: sd {sd.min():.2f}..{sd.max():.2f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    log(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sd")
    s.add_argument("--out", type=Path, default=SD_FILE)
    r = sub.add_parser("run")
    r.add_argument("--fields", type=Path, nargs="*", default=[])
    r.add_argument("--site", type=Path, default=ROOT)
    r.add_argument("--lists", type=Path)
    r.add_argument("--manifest-in", type=Path)
    a = ap.parse_args()
    if a.cmd == "sd":
        build_sd(a.out)
        return 0
    return run([p for p in a.fields if p.exists()], a.site, a.lists, a.manifest_in)


if __name__ == "__main__":
    raise SystemExit(main())
