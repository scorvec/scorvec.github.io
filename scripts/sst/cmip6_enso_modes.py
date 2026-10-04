#!/usr/bin/env python3
"""East-based vs central-Pacific El Nino, and the PDO with ENSO removed, over the Americas (2026-09-27; user: "For the El
Nino CMIP6 study, it would also be useful to look at the impact of east-based events and of the PDO index as well (trying
to separate out the ENSO parts as much as possible)"). LAPTOP analysis on the same 429 CMIP6 historical members (16
models, 1950-2014) and the same seasonal Americas tas/pr fields as cmip6_enso_impacts.py, plus the monthly Pacific ts from
cmip6_pacific_ts_extract.py.

E and C (Takahashi et al. 2011, GRL 38, L10704)
  per member: monthly ts anomalies over 10S-10N, 140E-80W (own monthly climatology and own quadratic trend removed),
  EOF1/EOF2 (sqrt-cos weighted), PCs standardised, signs fixed by pattern correlation with the observed ERSSTv6 EOFs
  (EOF1 warm over Nino-3.4; EOF2 warm in Nino-4 and cold in Nino-1+2, as Takahashi); E = (PC1 - PC2)/sqrt2 (east),
  C = (PC1 + PC2)/sqrt2 (central). Seasonal means, in monthly standard deviations.
  Cross-check (Kug et al. 2009, J. Climate 22, 1499): standardised Nino-3 > standardised Nino-4 -> EP, else CP.
PDO (Mantua et al. 1997; Zhang et al. 1997), ENSO removed (Newman et al. 2016, J. Climate 29, 4399)
  per member: monthly ts anomalies over 20-70N, 110E-100W minus that month's 60S-60N ocean-mean anomaly; ice months at
  the freezing point (271.35 K), boxes iced > 50 % of months dropped; EOF1, PC standardised, sign fixed against the
  observed PDO pattern (reference/pdo_pattern.nc); own quadratic trend removed. Reddened ENSO: PDO_t = a PDO_{t-1} + b
  N34_t fitted by OLS; the ENSO part is that recursion driven by N34 alone, P_E(t) = a P_E(t-1) + b N34_t, and the
  ENSO-free PDO = PDO - P_E (standardised). Robustness: PDO regressed on N34 at lags 0-12 months, residual.
Tests (the house rule): CMIP6 across-model one-sample t-test (each model counts once), Benjamini-Hochberg FDR 10 % over
  the map AND >= 80 % of models agreeing on the sign. Observed: OLS t-test per grid point, FDR 10 %; observed classes
  with < 8 events are not tested.

    python scripts/sst/cmip6_enso_modes.py            -> data/cmip6_enso/results/enso_modes_state.pkl + summary
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_enso_impacts as E                                               # noqa: E402

PAC = E.SRC / "pacific_ts"
STATE = E.OUT / "enso_modes_state.pkl"
ERSST = HERE / "data" / "ersst_v6_mnmean.nc"
PDO_DAT = HERE / "data" / "ersst.v5.pdo.dat"
PDO_PAT = HERE / "reference" / "pdo_pattern.nc"
LAT = np.arange(-30.0, 70.01, 2.5)
LON = np.arange(100.0, 290.01, 2.5)
TROP = ((-10, 10), (140, 280))
NPAC = ((20, 70), (110, 260))
BOX = {"n34": ((-5, 5), (190, 240)), "n3": ((-5, 5), (210, 270)), "n4": ((-5, 5), (160, 210)), "n12": ((-10, 0), (270, 280))}
FREEZE = 271.35
BINS = [0.5, 1.0, 1.5, 2.0, np.inf]                                            # matched Nino-3.4 strength bins (|index|)
PHASE = 0.5                                                                    # |ENSO-free PDO| >= 0.5 sd for a phase
SEAS = E.SEASONS


# ------------------------------------------------------------------------------------------------ helpers
def sel(lat, lon, box):
    (a, b), (c, d) = box
    return ((lat >= a) & (lat <= b))[:, None] & ((lon >= c) & (lon <= d))[None, :]


def boxmean(x, lat, lon, box):
    m = sel(lat, lon, box)
    w = np.cos(np.deg2rad(lat))[:, None] * m
    return np.nansum(x * w, axis=(-2, -1)) / np.sum(w * np.isfinite(x), axis=(-2, -1))


def mon_anom(x, nyr):
    """(T, ...) monthly -> anomalies about each calendar month's mean."""
    s = x.reshape(nyr, 12, *x.shape[1:])
    return (s - np.nanmean(s, 0, keepdims=True)).reshape(x.shape)


def eofs(X, w, k=2):
    """X (T, P) anomalies, w (P,) sqrt-cos weights -> PCs standardised (T, k), patterns (k, P) in X units per sd."""
    Xw = X * w[None]
    U, S, _ = np.linalg.svd(Xw, full_matrices=False)
    pcs = U[:, :k] * S[:k]
    pcs = (pcs - pcs.mean(0)) / pcs.std(0)
    pats = (X.T @ pcs / len(X)).T
    var = S[:k] ** 2 / np.sum(S ** 2)
    return pcs, pats, var


def pattern_r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def seasonal(x, months):
    """monthly (T, ...) with 'YYYY-MM' labels -> {season: {year: mean}} over complete seasons (DJF by its Jan year)."""
    acc = defaultdict(list)
    for i, m in enumerate(months):
        y, mm = int(m[:4]), int(m[5:7])
        for s, ms in E.SMON.items():
            if mm in ms:
                acc[(s, y + 1 if (s == "DJF" and mm == 12) else y)].append(x[i])
    out = {s: {} for s in SEAS}
    for (s, y), v in acc.items():
        if len(v) == 3 and np.all(np.isfinite(v)):
            out[s][y] = float(np.mean(v))
    return out


def reddened(pdo, n34):
    """Fit PDO_t = a PDO_{t-1} + b N34_t; return (a, b, ENSO part by recursion, R2 of that part)."""
    X = np.column_stack([pdo[:-1], n34[1:]]); a, b = np.linalg.lstsq(X, pdo[1:], rcond=None)[0]
    pe = np.zeros_like(pdo)
    for t in range(len(pdo)):
        pe[t] = (a * pe[t - 1] if t else 0.0) + b * n34[t]
    r2 = 1 - np.var(pdo - pe) / np.var(pdo)
    return float(a), float(b), pe, float(r2)


def lag_residual(pdo, n34, L=12):
    X = np.column_stack([np.r_[np.full(k, np.nan), n34[:len(n34) - k]] for k in range(L + 1)])
    ok = np.isfinite(X).all(1)
    c = np.linalg.lstsq(X[ok], pdo[ok], rcond=None)[0]
    r = np.full_like(pdo, np.nan); r[ok] = pdo[ok] - X[ok] @ c
    return r


def ols(X, A):
    """X (n, k) predictors (centred inside), A (n, lat, lon) -> coef (k, lat, lon), t (k, lat, lon), dof."""
    Xc = X - X.mean(0); Y = (A - A.mean(0)).reshape(len(A), -1)
    XtX = Xc.T @ Xc; B = np.linalg.solve(XtX, Xc.T @ Y)
    res = Y - Xc @ B; dof = len(A) - X.shape[1] - 1
    s2 = (res ** 2).sum(0) / dof
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(XtX)), s2))
    k = X.shape[1]
    return B.reshape(k, *A.shape[1:]), (B / se).reshape(k, *A.shape[1:]), dof


def bin_of(x):
    ax = np.abs(x)
    return np.digitize(ax, BINS) - 1                                           # -1 below 0.5, 0..3 otherwise


class Sums:
    """Per (model, key) event sums: field sum, count, index sum."""
    def __init__(self):
        self.d = {}

    def add(self, model, key, fields, idx):
        if len(fields) == 0:
            return
        e = self.d.setdefault((model, key), [np.zeros(fields.shape[1:]), 0, 0.0])
        e[0] += fields.sum(0); e[1] += len(fields); e[2] += float(np.sum(idx))

    def get(self, model, key):
        return self.d.get((model, key))


def matched(S, model, v, s, a, b, bins=range(4)):
    """Bin-matched difference a - b (class keys formatted with {k} for the bin): weights min(n_a, n_b) per bin."""
    num, wsum, xa, xb, na, nb = 0.0, 0.0, 0.0, 0.0, 0, 0
    for k in bins:
        A, B = S.get(model, (v, s, a.format(k=k))), S.get(model, (v, s, b.format(k=k)))
        if not A or not B or A[1] == 0 or B[1] == 0:
            continue
        w = min(A[1], B[1])
        num = num + w * (A[0] / A[1] - B[0] / B[1]); wsum += w
        xa += w * A[2] / A[1]; xb += w * B[2] / B[1]; na += A[1]; nb += B[1]
    if wsum == 0:
        return None
    return dict(diff=num / wsum, xa=xa / wsum, xb=xb / wsum, na=na, nb=nb)


# ------------------------------------------------------------------------------------------------ observations (indices)
def ersst_box():
    """ERSSTv6 monthly 1950-01 .. latest on the 2.5-degree box grid, NaN over land / under ice."""
    d = xr.open_dataset(ERSST)["sst"].sel(time=slice("1950-01-01", None)).sortby("lat")
    d = d.interp(lat=LAT, lon=LON)
    months = [f"{t.year:04d}-{t.month:02d}" for t in d.indexes["time"]]
    return d.values.astype("float64"), months


def obs_indices():
    """Observed monthly N34, E, C, PDO (NCEI) and the ENSO-free PDO; the EOF patterns that fix the model signs."""
    x, months = ersst_box()
    T = len(months); full = T - T % 12
    ny_fit = sum(1 for m in months if int(m[:4]) <= 2025) // 12
    # anomalies: monthly climatology over the fit years, then a LINEAR trend per grid point fitted 1950-2025 and
    # extrapolated (the only honest way to place 2026; the model side, fully in sample, removes a quadratic)
    fit = x[: ny_fit * 12]
    clim = np.nanmean(fit.reshape(ny_fit, 12, *x.shape[1:]), 0)
    a = x - np.tile(clim, (int(np.ceil(T / 12)), 1, 1))[:T]
    t = np.arange(T) / 12.0
    tf = t[: ny_fit * 12]
    af = a[: ny_fit * 12].reshape(ny_fit * 12, -1)
    ok = np.isfinite(af).all(0)
    V = np.column_stack([np.ones_like(tf), tf])
    coef = np.full((2, af.shape[1]), np.nan); coef[:, ok] = np.linalg.lstsq(V, af[:, ok], rcond=None)[0]
    a = a - (np.column_stack([np.ones_like(t), t]) @ coef).reshape(a.shape)
    idx = {k: boxmean(a, LAT, LON, b) for k, b in BOX.items()}
    # tropical EOFs on the fit years, 2026 projected
    m = sel(LAT, LON, TROP) & np.isfinite(a[: ny_fit * 12]).all(0)
    w = np.sqrt(np.cos(np.deg2rad(np.broadcast_to(LAT[:, None], m.shape)[m])))
    Xf = a[: ny_fit * 12][:, m]
    Xw = Xf * w[None]
    U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
    pcs_all = (a[:, m] * w[None]) @ Vt[:2].T
    mu, sd = pcs_all[: ny_fit * 12].mean(0), pcs_all[: ny_fit * 12].std(0)
    pcs = (pcs_all - mu) / sd
    pats = (Xf.T @ pcs[: ny_fit * 12] / (ny_fit * 12)).T
    P1, P2 = _put(m, pats[0]), _put(m, pats[1])
    if np.nanmean(P1[sel(LAT, LON, BOX["n34"])]) < 0:
        pcs[:, 0] *= -1; P1 = -P1
    if np.nanmean(P2[sel(LAT, LON, BOX["n4"])]) - np.nanmean(P2[sel(LAT, LON, BOX["n12"])]) < 0:
        pcs[:, 1] *= -1; P2 = -P2
    Ei = (pcs[:, 0] - pcs[:, 1]) / np.sqrt(2); Ci = (pcs[:, 0] + pcs[:, 1]) / np.sqrt(2)
    # PDO: NCEI's index, standardised 1950-2025
    pdo = {}
    for ln in PDO_DAT.read_text().splitlines()[2:]:
        p = ln.split()
        for k, v in enumerate(p[1:13]):
            if float(v) < 90:
                pdo[f"{int(p[0]):04d}-{k + 1:02d}"] = float(v)
    P = np.array([pdo.get(mm, np.nan) for mm in months])
    fitm = np.array([int(mm[:4]) <= 2025 for mm in months])
    P = (P - np.nanmean(P[fitm])) / np.nanstd(P[fitm])
    n34 = idx["n34"]
    okp = np.isfinite(P) & fitm
    last = np.flatnonzero(np.isfinite(P))[-1] + 1
    a_, b_, pe, r2 = reddened(P[:last], n34[:last])
    free = np.full_like(P, np.nan); free[:last] = P[:last] - pe
    free = (free - np.nanmean(free[okp])) / np.nanstd(free[okp])
    lagr = np.full_like(P, np.nan); lagr[:last] = lag_residual(P[:last], n34[:last])
    # the observed PDO pattern on the box grid (for the model sign)
    pp = xr.open_dataset(PDO_PAT)["eof"]
    pp = pp.assign_coords(lon=pp.lon % 360).sortby("lat").sortby("lon").interp(lat=LAT, lon=LON).values
    return dict(months=months, E=Ei, C=Ci, n34=n34, n3=idx["n3"], n4=idx["n4"], pdo=P, pdo_free=free, pdo_lag=lagr,
                a=a_, b=b_, r2=r2, P1=P1, P2=P2, var=(S[:2] ** 2 / np.sum(S ** 2)).tolist(), pdo_pat=pp,
                lat=LAT, lon=LON)


def _put(m, v):
    out = np.full(m.shape, np.nan); out[m] = v
    return out


# ------------------------------------------------------------------------------------------------ models
def member_indices(npz, ref):
    z = np.load(npz)
    q = z["ts"].astype("float64"); ts = np.where(q == -32768, np.nan, q / 100.0 + 273.15)
    months = [str(m) for m in z["months"]]; ny = len(months) // 12
    yrs = np.arange(ny) + int(months[0][:4])
    tt = np.repeat(yrs, 12) + (np.tile(np.arange(12), ny) + 0.5) / 12
    ocean = np.isfinite(ts).any(0)
    iced = np.isnan(ts) & ocean[None]
    icefrac = iced.mean(0)
    ts = np.where(iced, FREEZE, ts)
    a = mon_anom(ts, ny)
    # own quadratic trend per box (ocean boxes kept whole)
    flat = a.reshape(len(a), -1); okb = np.isfinite(flat).all(0)
    flat[:, okb] = E.detrend(flat[:, okb], tt).astype("float64")
    a = flat.reshape(a.shape)
    out = {"months": months}
    for k, b in BOX.items():
        out[k] = boxmean(a, LAT, LON, b)
    # E, C
    m = sel(LAT, LON, TROP) & np.isfinite(a).all(0)
    w = np.sqrt(np.cos(np.deg2rad(np.broadcast_to(LAT[:, None], m.shape)[m])))
    pcs, pats, var = eofs(a[:, m], w)
    P1, P2 = _put(m, pats[0]), _put(m, pats[1])
    r1, r2 = pattern_r(P1, ref["P1"]), pattern_r(P2, ref["P2"])
    if r1 < 0:
        pcs[:, 0] *= -1; r1 = -r1
    if r2 < 0:
        pcs[:, 1] *= -1; r2 = -r2
    out["E"] = (pcs[:, 0] - pcs[:, 1]) / np.sqrt(2); out["C"] = (pcs[:, 0] + pcs[:, 1]) / np.sqrt(2)
    out["eof_r"] = (r1, r2); out["eof_var"] = var.tolist()
    # PDO: raw ts anomalies (global-mean anomaly removed), before the per-box detrend
    ga = z["gm"].astype("float64"); ga = ga - np.tile(ga.reshape(ny, 12).mean(0), ny)
    araw = mon_anom(ts, ny) - ga[:, None, None]
    mp = sel(LAT, LON, NPAC) & ocean & (icefrac <= 0.5)
    wp = np.sqrt(np.cos(np.deg2rad(np.broadcast_to(LAT[:, None], mp.shape)[mp])))
    pc, pat, pvar = eofs(araw[:, mp], wp, k=1)
    rp = pattern_r(_put(mp, pat[0]), ref["pdo_pat"])
    pdo = pc[:, 0] * (1 if rp >= 0 else -1)
    pdo = E.detrend(pdo, tt).astype("float64"); pdo = (pdo - pdo.mean()) / pdo.std()
    n34 = out["n34"]
    ar, br, pe, r2p = reddened(pdo, n34)
    free = pdo - pe; free = (free - free.mean()) / free.std()
    lag = lag_residual(pdo, n34); lag = (lag - np.nanmean(lag)) / np.nanstd(lag)
    out.update(pdo=pdo, pdo_free=free, pdo_lag=lag, pdo_r=abs(rp), pdo_var=float(pvar[0]), a=ar, b=br, r2=r2p)
    return out


def process(name, ref, R, Sm, info):
    ds = xr.open_dataset(E.SRC / f"{name}.nc")
    model = ds.attrs["source_id"]
    years = ds["year"].values.astype(int)
    F = {v: ds[v].values.astype("float32") for v in E.VARS}
    idx34, _ = E.season_index(ds); ds.close()
    mi = member_indices(PAC / f"{name}.npz", ref)
    sea = {k: seasonal(mi[k], mi["months"]) for k in ("E", "C", "n3", "n4", "pdo", "pdo_free", "pdo_lag")}
    info["members"].append(dict(model=model, member=name, eof_r=mi["eof_r"], eof_var=mi["eof_var"], pdo_r=mi["pdo_r"],
                                pdo_var=mi["pdo_var"], a=mi["a"], b=mi["b"], r2=mi["r2"],
                                r_free_lag=float(np.corrcoef(mi["pdo_free"][13:], mi["pdo_lag"][13:])[0, 1])))
    for si, s in enumerate(SEAS):
        ok = np.array([y in idx34[s] and all(y in sea[k][s] for k in sea) for y in years])
        for v in E.VARS:
            ok &= np.isfinite(F[v][si]).all(axis=(1, 2))
        yrs = years[ok]
        if len(yrs) < 30:
            continue
        x = E.detrend(np.array([idx34[s][y] for y in yrs]), yrs.astype(float)).astype(float)
        I = {k: np.array([sea[k][s][y] for y in yrs]) for k in sea}
        n3 = (I["n3"] - I["n3"].mean()) / I["n3"].std(); n4 = (I["n4"] - I["n4"].mean()) / I["n4"].std()
        info["idx"].append(dict(model=model, member=name, season=s, years=yrs.tolist(), n34=x.tolist(),
                                E=I["E"].tolist(), C=I["C"].tolist(), pdo=I["pdo"].tolist(), pdo_free=I["pdo_free"].tolist()))
        neu = np.abs(x) < E.NEUTRAL
        en, ln = x >= 0.5, x <= -0.5
        ep, cp = en & (I["E"] > I["C"]), en & (I["E"] <= I["C"])
        kep, kcp = en & (n3 > n4), en & (n3 <= n4)
        pp, pm = I["pdo_free"] >= PHASE, I["pdo_free"] <= -PHASE
        b = bin_of(x)
        for v in E.VARS:
            A = E.detrend(F[v][si][ok], yrs.astype(float)).astype("float64")
            # partial / simple regressions
            for key, X, one in (("EC", np.column_stack([I["E"], I["C"]]), False), ("NP", np.column_stack([x, I["pdo_free"]]), False),
                                ("NPlag", np.column_stack([x, I["pdo_lag"]]), False), ("Praw", I["pdo"][:, None], True),
                                ("Pfree", I["pdo_free"][:, None], True), ("N34", x[:, None], True)):
                B = ols(X, A)[0]
                acc = R[(v, s, key)].setdefault(model, [0.0, 0])
                acc[0] = acc[0] + (B[0] if one else B); acc[1] += 1
            # composites (relative to the member's neutral mean)
            if neu.sum() < E.MIN_NEU:
                continue
            An = A - A[neu].mean(0)
            for k in range(4):
                bk = b == k
                for nm, m in (("ep", ep), ("cp", cp), ("kep", kep), ("kcp", kcp),
                              ("en+", en & pp), ("en-", en & pm), ("ln+", ln & pp), ("ln-", ln & pm)):
                    Sm.add(model, (v, s, f"{nm}{k}"), An[m & bk], x[m & bk])
            Sm.add(model, (v, s, "neu+0"), An[neu & pp], x[neu & pp]); Sm.add(model, (v, s, "neu-0"), An[neu & pm], x[neu & pm])
    return model


def obs_part(ref, W):
    """Observed regressions (ERA5 tas 1960-, GPCP pr 1980-) on [E, C], [ONI, ENSO-free PDO], raw and ENSO-free PDO."""
    oni = E.oni_table()
    lat, lon = E.STATE_LAT, E.STATE_LON
    OS = {"tas": E.obs_seasons(E.era5_monthly(lat, lon), oni), "pr": E.obs_seasons(E.gpcp_monthly(lat, lon), oni)}
    sea = {k: seasonal(np.asarray(ref[k], float), ref["months"]) for k in ("E", "C", "pdo", "pdo_free", "n3", "n4")}
    out = {}
    for v in E.VARS:
        for s in SEAS:
            yrs, x, A, clim = OS[v][s]
            ok = np.array([all(y in sea[k][s] for k in sea) for y in yrs])
            yy, xx, AA = yrs[ok], x[ok], A[ok]
            I = {k: np.array([sea[k][s][y] for y in yy]) for k in sea}
            ent = dict(n=int(ok.sum()), span=[int(yy[0]), int(yy[-1])], clim=clim)
            for key, X in (("EC", np.column_stack([I["E"], I["C"]])), ("NP", np.column_stack([xx, I["pdo_free"]])),
                           ("Praw", I["pdo"][:, None]), ("Pfree", I["pdo_free"][:, None])):
                B, t, dof = ols(X, AA)
                p = 2 * stats.t.sf(np.abs(t), dof)
                ent[key] = dict(coef=B, sig=np.stack([E.fdr(p[i]) for i in range(len(B))]), p=p)
            # observed EP / CP counts (tested only with >= 8 of each)
            en = xx >= 0.5
            ent["n_ep"] = int((en & (I["E"] > I["C"])).sum()); ent["n_cp"] = int((en & (I["E"] <= I["C"])).sum())
            ent["n_en_pos"] = int((en & (I["pdo_free"] >= PHASE)).sum()); ent["n_en_neg"] = int((en & (I["pdo_free"] <= -PHASE)).sum())
            ln = xx <= -0.5
            ent["n_ln_pos"] = int((ln & (I["pdo_free"] >= PHASE)).sum()); ent["n_ln_neg"] = int((ln & (I["pdo_free"] <= -PHASE)).sum())
            if ent["n_ep"] >= E.OBS_TEST_N and ent["n_cp"] >= E.OBS_TEST_N:
                _, p = stats.ttest_ind(AA[en & (I["E"] > I["C"])], AA[en & (I["E"] <= I["C"])], axis=0, equal_var=False)
                ent["epcp"] = dict(diff=AA[en & (I["E"] > I["C"])].mean(0) - AA[en & (I["E"] <= I["C"])].mean(0), sig=E.fdr(p))
            out[(v, s)] = ent
    return out, sea


# ------------------------------------------------------------------------------------------------ PDO robustness
VARIANTS = {"V0": ("n34", False, False), "V1": ("n34", False, True), "V2": ("n34", True, True), "V3": ("ecq", True, True)}
VLABEL = {"V0": "a, b constant, N34 (published)", "V1": "b by calendar month", "V2": "a and b by calendar month",
          "V3": "monthly a; monthly b on E, C and N34*|N34|"}


def enso_part(pdo, X, mon, monthly_a, monthly_b):
    """Reddened-ENSO part of the PDO with optionally month-dependent persistence a(m) and forcing b_k(m) on several
    predictors X (T, k): fit PDO_t = a(m_t) PDO_{t-1} + sum_k b_k(m_t) X_k(t) by OLS, then run the recursion driven by
    the X alone. Returns (P_E, variance share of the PDO it carries)."""
    T, K = X.shape
    oh = np.eye(12)[mon]                                                   # (T, 12)
    cols = [(pdo[:-1, None] * oh[1:]) if monthly_a else pdo[:-1, None]]
    for k in range(K):
        cols.append(X[1:, k:k + 1] * oh[1:] if monthly_b else X[1:, k:k + 1])
    D = np.hstack(cols); ok = np.isfinite(D).all(1) & np.isfinite(pdo[1:])
    c = np.linalg.lstsq(D[ok], pdo[1:][ok], rcond=None)[0]
    na = 12 if monthly_a else 1; nb = 12 if monthly_b else 1
    a = c[:na]; B = c[na:].reshape(K, nb)
    pe = np.zeros(T)
    for t in range(T):
        m = mon[t]
        f = sum(B[k, m if monthly_b else 0] * (X[t, k] if np.isfinite(X[t, k]) else 0.0) for k in range(K))
        pe[t] = (a[m if monthly_a else 0] * pe[t - 1] if t else 0.0) + f
    ok = np.isfinite(pdo)
    return pe, float(1 - np.var(pdo[ok] - pe[ok]) / np.var(pdo[ok]))


def variant_free(pdo, n34, E_, C_, months):
    mon = np.array([int(m[5:7]) - 1 for m in months])
    out = {}
    for v, (pred, ma, mb) in VARIANTS.items():
        X = n34[:, None] if pred == "n34" else np.column_stack([E_, C_, n34 * np.abs(n34)])
        pe, share = enso_part(pdo, X, mon, ma, mb)
        f = pdo - pe; ok = np.isfinite(f)
        f = (f - np.nanmean(f[ok])) / np.nanstd(f[ok])
        out[v] = (f, share)
    return out


def robust_pdo() -> int:
    """Does a stronger ENSO removal change the ENSO-free PDO results? Per member, the four VARIANTS; per season the
    partial regression on [same-season N34, ENSO-free PDO] (tas, pr), the robust land share, the pattern r against V0,
    regional values, and the El Nino x PDO-phase interaction. Observed variance shares from NCEI PDO + ERSST v6."""
    t0 = time.time()
    st = pickle.load(open(E.STATE, "rb"))
    W, land = st["W"], st["lf"] >= 0.5
    ref = obs_indices()
    names = sorted(Path(f).stem for f in glob.glob(str(E.SRC / "*.nc")) if ".part" not in f)
    names = [n for n in names if (PAC / f"{n}.npz").exists()]
    R = defaultdict(dict); Sm = Sums(); info = defaultdict(list)
    for i, name in enumerate(names):
        ds = xr.open_dataset(E.SRC / f"{name}.nc"); model = ds.attrs["source_id"]
        years = ds["year"].values.astype(int); F = {v: ds[v].values.astype("float32") for v in E.VARS}
        idx34, _ = E.season_index(ds); ds.close()
        mi = member_indices(PAC / f"{name}.npz", ref)
        vf = variant_free(mi["pdo"], mi["n34"], mi["E"], mi["C"], mi["months"])
        for v_, (f, share) in vf.items():
            info[v_].append(dict(model=model, share=share, r_v0=float(np.corrcoef(f, vf["V0"][0])[0, 1])))
        sea = {v_: seasonal(f, mi["months"]) for v_, (f, _) in vf.items()}
        for si, s in enumerate(SEAS):
            ok = np.array([y in idx34[s] and all(y in sea[v_][s] for v_ in sea) for y in years])
            for v in E.VARS:
                ok &= np.isfinite(F[v][si]).all(axis=(1, 2))
            yrs = years[ok]
            if len(yrs) < 30:
                continue
            x = E.detrend(np.array([idx34[s][y] for y in yrs]), yrs.astype(float)).astype(float)
            neu, en = np.abs(x) < E.NEUTRAL, x >= 0.5
            b = bin_of(x)
            for v in E.VARS:
                A = E.detrend(F[v][si][ok], yrs.astype(float)).astype("float64")
                An = A - A[neu].mean(0)
                for v_ in sea:
                    P = np.array([sea[v_][s][y] for y in yrs])
                    B = ols(np.column_stack([x, P]), A)[0][1]
                    acc = R[(v, s, v_)].setdefault(model, [0.0, 0]); acc[0] = acc[0] + B; acc[1] += 1
                    pp, pm = P >= PHASE, P <= -PHASE
                    for k in range(4):
                        Sm.add(model, (v, s, f"{v_}en+{k}"), An[en & pp & (b == k)], x[en & pp & (b == k)])
                        Sm.add(model, (v, s, f"{v_}en-{k}"), An[en & pm & (b == k)], x[en & pm & (b == k)])
                    Sm.add(model, (v, s, f"{v_}neu+0"), An[neu & pp], x[neu & pp]); Sm.add(model, (v, s, f"{v_}neu-0"), An[neu & pm], x[neu & pm])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)} members, {time.time() - t0:.0f} s", flush=True)
    # observed variance shares
    omon = [m for m, p in zip(ref["months"], ref["pdo"]) if np.isfinite(p)]
    nobs = len(omon)
    ovf = variant_free(ref["pdo"][:nobs], ref["n34"][:nobs], ref["E"][:nobs], ref["C"][:nobs], omon)
    res = {"variants": VLABEL, "models": {}, "observed": {}, "maps": {}}
    for v_, rows in info.items():
        sh = np.array([r["share"] for r in rows]); rr = np.array([r["r_v0"] for r in rows])
        res["models"][v_] = dict(share_p10_50_90=np.round(np.percentile(sh, [10, 50, 90]), 3).tolist(),
                                 r_with_V0_p10_50_90=np.round(np.percentile(rr, [10, 50, 90]), 3).tolist())
        f, share = ovf[v_]
        res["observed"][v_] = dict(share=round(share, 3), r_with_V0=round(float(np.corrcoef(f, ovf["V0"][0])[0, 1]), 3))
    wl = np.cos(np.deg2rad(st["lat"]))[:, None] * land
    REG = {r: k for k, r in enumerate(E.RNAMES)}
    for v in E.VARS:
        for s in SEAS:
            base = None
            for v_ in VARIANTS:
                d = R[(v, s, v_)]; mn = sorted(d)
                S = np.stack([d[m][0] / d[m][1] for m in mn])
                mm, sg, _, _ = E.robust(S)
                if v_ == "V0":
                    base = mm
                r_land = float(np.sum(wl * mm * base) / np.sqrt(np.sum(wl * mm ** 2) * np.sum(wl * base ** 2)))
                Rg = E.regional(S, W)
                if v == "pr":
                    cl = E.regional(st["clim"][s][1].mean(0), W); Rg = 100 * Rg / cl
                _, p = stats.ttest_1samp(Rg, 0, axis=0); ag = np.maximum((Rg > 0).mean(0), (Rg < 0).mean(0))
                okr = E.fdr(p) & (ag >= 0.8)
                regs = {r: (round(float(Rg[:, REG[r]].mean()), 3) if okr[REG[r]] else "n.s.")
                        for r in ("Alaska", "Pacific Northwest", "Southeast US", "Gulf Coast", "N Plains / Prairies", "Ohio Valley")}
                # interaction (El Nino, +PDO - -PDO) - (neutral, +PDO - -PDO), matched
                ints = []
                for m in mn:
                    e_ = matched(Sm, m, v, s, f"{v_}en+{{k}}", f"{v_}en-{{k}}")
                    n_ = matched(Sm, m, v, s, f"{v_}neu+0", f"{v_}neu-0", bins=(0,))
                    if e_ and n_:
                        ints.append(e_["diff"] - n_["diff"])
                imm, isg, _, _ = E.robust(np.stack(ints))
                res["maps"][f"{v}|{s}|{v_}"] = dict(land_robust=round(float(sg[land].mean()), 3), pattern_r_vs_V0=round(r_land, 3),
                                                    regions=regs, interaction_land_robust=round(float(isg[land].mean()), 3))
    out = E.OUT / "pdo_robustness.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("models", "observed")}, indent=1))
    for k, e in res["maps"].items():
        print(k, e)
    print(f"wrote {out} ({time.time() - t0:.0f} s)")
    return 0


def site() -> int:
    """Compact reference for the site renderer: reference/enso_modes_site.{npz,json}. CMIP6 maps are tested in the
    study's units (mm/day for rain) and shown as % of the multi-model normal (cells under 0.3 mm/day unscored)."""
    import cmip6_enso_site as CS
    st = pickle.load(open(STATE, "rb"))
    lat, lon = st["lat"], st["lon"]
    land = st["lf"] >= 0.5
    npz = {"lat": lat.astype("float32"), "lon": lon.astype("float32")}
    meta = {"models": len(st["members"]), "members": int(sum(st["members"].values())), "maps": {}}

    def put(key, f, sg, info, v, s):
        if v == "pr":
            f, sg = CS.to_pct(f, sg, st["clim"][s][1].mean(0))
        npz[key + "|f"] = np.asarray(f, "float32"); npz[key + "|s"] = np.asarray(sg, bool)
        ok = land & np.isfinite(f)
        info = dict(info, land_sig=round(float(sg[ok].mean()), 3)); meta["maps"][key] = info
        return f

    share = {}
    for v in E.VARS:
        for s in SEAS:
            e = st["maps"][(v, s, "EC")]
            for j, nm in enumerate(("E", "C")):
                put(f"ec|{v}|{s}|{nm}|cmip6", e["mm"][j], e["sig"][j], dict(models=len(e["models"]), members=e["n_members"]), v, s)
            e = st["maps"][(v, s, "NP")]
            put(f"pdo|{v}|{s}|free|cmip6", e["mm"][1], e["sig"][1], dict(models=len(e["models"]), members=e["n_members"]), v, s)
            e = st["maps"][(v, s, "Praw")]
            put(f"pdo|{v}|{s}|raw|cmip6", e["mm"], e["sig"], dict(models=len(e["models"]), members=e["n_members"]), v, s)
            # how much of the raw PDO's impact is ENSO: raw vs ENSO-free simple regressions (per sd), over land
            raw, fr = st["maps"][(v, s, "Praw")], st["maps"][(v, s, "Pfree")]
            wl = (np.cos(np.deg2rad(lat))[:, None] * land)
            rr = np.sum(wl * raw["mm"] * fr["mm"]) / np.sqrt(np.sum(wl * raw["mm"] ** 2) * np.sum(wl * fr["mm"] ** 2))
            vs = float(np.sum(wl * fr["mm"] ** 2) / np.sum(wl * raw["mm"] ** 2))
            per = [(float(np.sum(wl * a * b) / np.sqrt(np.sum(wl * a * a) * np.sum(wl * b * b))), float(np.sum(wl * b * b) / np.sum(wl * a * a)))
                   for a, b in zip(raw["per_model"], fr["per_model"])]
            lag = st["maps"][(v, s, "NPlag")]
            rl = np.sum(wl * lag["mm"][1] * st["maps"][(v, s, "NP")]["mm"][1]) / np.sqrt(
                np.sum(wl * lag["mm"][1] ** 2) * np.sum(wl * st["maps"][(v, s, "NP")]["mm"][1] ** 2))
            share[f"{v}|{s}"] = dict(pattern_r=round(float(rr), 3), var_share=round(vs, 3),
                                     raw_land_robust=round(float(raw["sig"][land].mean()), 3),
                                     free_land_robust=round(float(fr["sig"][land].mean()), 3),
                                     partial_land_robust=round(float(st["maps"][(v, s, "NP")]["sig"][1][land].mean()), 3),
                                     model_pattern_r=[round(float(np.median([p[0] for p in per])), 3), round(float(min(p[0] for p in per)), 3), round(float(max(p[0] for p in per)), 3)],
                                     model_var_share=[round(float(np.median([p[1] for p in per])), 3), round(float(min(p[1] for p in per)), 3), round(float(max(p[1] for p in per)), 3)],
                                     lag_vs_reddened_pattern_r=round(float(rl), 3))
            for key, nm in (("epcp", "diff"), ("kepcp", "kug"), ("en_pdo", "en"), ("ln_pdo", "ln"), ("en_pdo_int", "int")):
                c = st["comps"].get((v, s, key))
                if not c or not c.get("tested"):
                    meta["maps"][f"cmp|{v}|{s}|{nm}"] = dict(tested=False, models=len(c["models"]) if c else 0); continue
                inf = dict(models=len(c["models"]), tested=True)
                for k2 in ("n_a", "n_b", "x_a", "x_b"):
                    if k2 in c:
                        inf[k2] = round(c[k2], 3) if isinstance(c[k2], float) else c[k2]
                put(f"cmp|{v}|{s}|{nm}", c["mm"], c["sig"], inf, v, s)
            # Kug vs Takahashi composite agreement
            a_, b_ = st["comps"].get((v, s, "epcp")), st["comps"].get((v, s, "kepcp"))
            if a_ and b_ and a_.get("tested") and b_.get("tested"):
                share[f"{v}|{s}"]["epcp_takahashi_vs_kug_r"] = round(pattern_r(a_["mm"], b_["mm"]), 3)
            # observed
            o = st["obs"][(v, s)]
            for key, j, nm in (("EC", 0, "E"), ("EC", 1, "C"), ("NP", 1, "free"), ("Praw", 0, "raw")):
                f = o[key]["coef"][j]; sg = o[key]["sig"][j]
                if v == "pr":
                    cl = o["clim"]; f = np.where(cl >= CS.DRY, 100.0 * f / np.maximum(cl, 1e-6), np.nan); sg = sg & np.isfinite(f)
                pref = "ec" if key == "EC" else "pdo"
                npz[f"{pref}|{v}|{s}|{nm}|obs|f"] = f.astype("float32"); npz[f"{pref}|{v}|{s}|{nm}|obs|s"] = sg
                meta["maps"][f"{pref}|{v}|{s}|{nm}|obs"] = dict(n=o["n"], span=o["span"], land_sig=round(float(sg[land & np.isfinite(f)].mean()), 3))
            meta.setdefault("obs_counts", {})[f"{v}|{s}"] = {k: o[k] for k in ("n_ep", "n_cp", "n_en_pos", "n_en_neg", "n_ln_pos", "n_ln_neg")}
    meta["pdo_share"] = share
    # indices: EOF quality, reddened ENSO, and the E-C placement (models vs observed)
    mem = st["info"]["members"]
    q = lambda k, i=None: [round(float(np.percentile([m[k] if i is None else m[k][i] for m in mem], p)), 3) for p in (10, 50, 90)]
    meta["quality"] = dict(eof1_r=q("eof_r", 0), eof2_r=q("eof_r", 1), eof1_var=q("eof_var", 0), eof2_var=q("eof_var", 1),
                           pdo_r=q("pdo_r"), a=q("a"), b=q("b"), r2=q("r2"), r_free_lag=q("r_free_lag"))
    ref = st["ref"]
    meta["obs_ref"] = dict(a=round(ref["a"], 3), b=round(ref["b"], 3), r2=round(ref["r2"], 3), eof_var=[round(x, 3) for x in ref["var"]])
    osea = {k: seasonal(np.asarray(ref[k], float), ref["months"]) for k in ("E", "C", "n34", "pdo", "pdo_free")}
    meta["obs_djf"] = {str(y): dict(E=round(osea["E"]["DJF"][y], 2), C=round(osea["C"]["DJF"][y], 2), n34=round(osea["n34"]["DJF"][y], 2),
                                    pdo=round(osea["pdo"]["DJF"].get(y, np.nan), 2), pdo_free=round(osea["pdo_free"]["DJF"].get(y, np.nan), 2))
                       for y in sorted(osea["E"]["DJF"]) if y in osea["n34"]["DJF"]}
    meta["obs_jja"] = {str(y): dict(E=round(osea["E"]["JJA"][y], 2), C=round(osea["C"]["JJA"][y], 2), n34=round(osea["n34"]["JJA"][y], 2))
                       for y in sorted(osea["E"]["JJA"])}
    meta["obs_months"] = {m: dict(E=round(float(ref["E"][i]), 2), C=round(float(ref["C"][i]), 2), n34=round(float(ref["n34"][i]), 2),
                                  pdo=None if not np.isfinite(ref["pdo"][i]) else round(float(ref["pdo"][i]), 2),
                                  pdo_free=None if not np.isfinite(ref["pdo_free"][i]) else round(float(ref["pdo_free"][i]), 2))
                          for i, m in enumerate(ref["months"]) if m >= "2025-06"}
    # model DJF El Nino events and the JJA before them, pooled over members: (E, C, N34), and the super subset
    djf, jja = [], []
    by = defaultdict(dict)
    for d in st["info"]["idx"]:
        by[d["member"]][d["season"]] = d
    for mname, dd in by.items():
        if "DJF" not in dd or "JJA" not in dd:
            continue
        D, J = dd["DJF"], dd["JJA"]
        jrow = {y: i for i, y in enumerate(J["years"])}
        for i, y in enumerate(D["years"]):
            if D["n34"][i] >= 0.5:
                djf.append((D["E"][i], D["C"][i], D["n34"][i], D["pdo_free"][i]))
                if (y - 1) in jrow:
                    k = jrow[y - 1]; jja.append((J["E"][k], J["C"][k], J["n34"][k], D["n34"][i]))
    djf, jja = np.array(djf, "float32"), np.array(jja, "float32")
    npz["djf_events"] = djf; npz["jja_before"] = jja
    sup = djf[:, 2] >= 2.0; supj = jja[:, 3] >= 2.0
    e26 = meta["obs_jja"].get("2026", {})
    meta["placement"] = dict(
        n_djf=int(len(djf)), n_super=int(sup.sum()), ep_share_all=round(float(np.mean(djf[:, 0] > djf[:, 1])), 3),
        ep_share_super=round(float(np.mean(djf[sup, 0] > djf[sup, 1])), 3),
        ep_share_strong=round(float(np.mean(djf[djf[:, 2] >= 1.5, 0] > djf[djf[:, 2] >= 1.5, 1])), 3),
        n_jja_super=int(supj.sum()),
        jja2026_E_pct_among_super_jja=round(float(100 * np.mean(jja[supj, 0] < e26.get("E", np.nan))), 1) if e26 else None,
        jja2026_C_pct_among_super_jja=round(float(100 * np.mean(jja[supj, 1] < e26.get("C", np.nan))), 1) if e26 else None,
        super_given_jja_like_2026=None)
    if e26:
        near = (np.abs(jja[:, 0] - e26["E"]) < 0.5) & (np.abs(jja[:, 1] - e26["C"]) < 0.5)
        meta["placement"]["n_jja_near_2026"] = int(near.sum())
        if near.sum() >= 20:
            dn = jja[near, 3]
            meta["placement"]["djf_after_jja_near_2026"] = dict(p10_50_90=[round(float(x), 2) for x in np.percentile(dn, [10, 50, 90])],
                                                               p_super=round(float(np.mean(dn >= 2.0)), 3), p_strong=round(float(np.mean(dn >= 1.5)), 3))
    out = HERE / "reference"
    np.savez_compressed(out / "enso_modes_site.npz", **npz)
    (out / "enso_modes_site.json").write_text(json.dumps(meta, separators=(",", ":"), default=float))
    print(f"wrote enso_modes_site.npz ({(out / 'enso_modes_site.npz').stat().st_size / 1e6:.1f} MB, {len(meta['maps'])} maps)")
    return 0


def main() -> int:
    if "--site" in sys.argv:
        return site()
    if "--pdo-robust" in sys.argv:
        return robust_pdo()
    t0 = time.time()
    st = pickle.load(open(E.STATE, "rb"))
    E.STATE_LAT, E.STATE_LON = st["lat"], st["lon"]
    ref = obs_indices()
    print(f"observed indices: EOF var {np.round(ref['var'], 3)}, reddened ENSO a {ref['a']:.2f} b {ref['b']:.2f} "
          f"R2 {ref['r2']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    names = sorted(Path(f).stem for f in glob.glob(str(E.SRC / "*.nc")) if ".part" not in f)
    names = [n for n in names if (PAC / f"{n}.npz").exists()]
    R = defaultdict(dict); Sm = Sums(); info = {"members": [], "idx": []}      # R[key][model] = [sum of member maps, n]
    members = defaultdict(int)
    for i, n in enumerate(names):
        members[process(n, ref, R, Sm, info)] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(names)} members, {time.time() - t0:.0f} s", flush=True)
    print(f"{len(names)} members, {len(members)} models", flush=True)
    maps = {}
    for key, d in R.items():
        mnames = sorted(d)
        S = np.stack([d[m][0] / d[m][1] for m in mnames])                # (models, [k,] lat, lon); members equal within a model
        e = dict(models=mnames, n_members=int(sum(d[m][1] for m in mnames)), per_model=S.astype("float32"))
        if S.ndim == 4:
            e["mm"], e["sig"] = [], []
            for j in range(S.shape[1]):
                mm, sg, _, _ = E.robust(S[:, j]); e["mm"].append(mm); e["sig"].append(sg)
        else:
            e["mm"], e["sig"], _, _ = E.robust(S)
        maps[key] = e
    comps = {}
    for v in E.VARS:
        for s in SEAS:
            for key, a, b, bins in (("epcp", "ep{k}", "cp{k}", range(4)), ("kepcp", "kep{k}", "kcp{k}", range(4)),
                                    ("en_pdo", "en+{k}", "en-{k}", range(4)), ("ln_pdo", "ln+{k}", "ln-{k}", range(4)),
                                    ("neu_pdo", "neu+0", "neu-0", (0,))):
                rows = {m: matched(Sm, m, v, s, a, b, bins) for m in members}
                rows = {m: r for m, r in rows.items() if r is not None}
                if len(rows) < E.MIN_MODELS:
                    comps[(v, s, key)] = dict(models=sorted(rows), tested=False); continue
                S = np.stack([rows[m]["diff"] for m in sorted(rows)])
                mm, sg, p, agree = E.robust(S)
                comps[(v, s, key)] = dict(models=sorted(rows), tested=True, mm=mm, sig=sg, per_model=S.astype("float32"),
                                          n_a=int(sum(r["na"] for r in rows.values())), n_b=int(sum(r["nb"] for r in rows.values())),
                                          x_a=float(np.mean([r["xa"] for r in rows.values()])),
                                          x_b=float(np.mean([r["xb"] for r in rows.values()])))
            # interaction: (El Nino | +PDO - El Nino | -PDO) - (neutral | +PDO - neutral | -PDO), per model
            en, ne = comps[(v, s, "en_pdo")], comps[(v, s, "neu_pdo")]
            if en.get("tested") and ne.get("tested"):
                common = [m for m in en["models"] if m in ne["models"]]
                S = np.stack([en["per_model"][en["models"].index(m)] - ne["per_model"][ne["models"].index(m)] for m in common])
                mm, sg, _, _ = E.robust(S)
                comps[(v, s, "en_pdo_int")] = dict(models=common, tested=len(common) >= E.MIN_MODELS, mm=mm, sig=sg)
    print(f"model maps done {time.time() - t0:.0f} s; observations ...", flush=True)
    obs, obs_sea = obs_part(ref, st["W"])
    state = dict(lat=st["lat"], lon=st["lon"], lf=st["lf"], W=st["W"], clim=st["clim"], maps=maps, comps=comps, obs=obs,
                 ref={k: v for k, v in ref.items()}, obs_sea=obs_sea, info=info, members=dict(members))
    with open(STATE, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {STATE} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
