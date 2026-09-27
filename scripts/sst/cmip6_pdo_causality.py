#!/usr/bin/env python3
"""Does the North Pacific (ENSO-free PDO) SST force the winter atmosphere, or only record it? (2026-09-27; user: "It's
very implausible that this wave response could just be from the PDO alone" / "Is there a way you could test the
causality?"). The same-season regression mostly shows the atmosphere driving the SST (Frankignoul & Hasselmann 1977;
Newman et al. 2016). Three estimates of the FORCED response, and the reverse direction:

  lead-lag   coupled CMIP6 historical (147 members with z500, 16 models) and ERA5: DJF z500 regressed on the PRECEDING SON
             (and JAS) ENSO-free PDO with the preceding atmosphere (SON Aleutian-low index) and Nino-3.4 (SON and DJF) held
             fixed; Granger: nested regressions, F-test, and the out-of-sample gain (5-fold by blocks of years) of adding
             the SON PDO to predict the DJF Aleutian low - and the reverse, the SON Aleutian low predicting the DJF PDO.
  Frankignoul  lagged-covariance response  B = cov(Z_t, P_{t-tau}) / cov(P_t, P_{t-tau}),  tau = 1-3 months, Z_t in DJF
             months, Z with the same month's Nino-3.4 regressed out (Frankignoul et al. 1998).
  AMIP       prescribed observed SST (the atmosphere cannot drive it), 1979-2014: DJF z500 on the OBSERVED ENSO-free PDO with
             observed Nino-3.4 and tropical Indo-Pacific indices alongside.
  DCPP-C     dcppC-ipv-NexTrop-pos minus -neg: the response to the imposed extratropical North Pacific pattern, scaled to
             the PDO index the imposed SST projects to (NCEI PDO pattern), and the full-pattern ipv-pos minus -neg.
Tests as the site's: CMIP6 across-model t-test, BH-FDR 10 % AND >= 80 % sign agreement (maps; regional indices plain
across-model t-test); ERA5 OLS t-tests / F-tests. Nothing here is published; results go to
data/cmip6_enso/results/pdo_causality.json and figures in data/cmip6_enso/results/causality_*.png.

    python scripts/sst/cmip6_pdo_causality.py [coupled|amip|dcpp|all]
"""
from __future__ import annotations

import glob
import json
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
import cmip6_enso_modes as MO                                                # noqa: E402
import cmip6_z500_impacts as ZI                                              # noqa: E402

LAT, LON = ZI.LAT, ZI.LON
AL_BOX = ((30, 65), (160, 220))                         # Aleutian-low index: z500 area mean, 30-65N 160E-140W (low = negative)
NPC_BOX = ((35, 45), (170, 210))                        # PDO centre of action (SST, K per sd of the PDO)
RES = E.OUT
FORCED = E.SRC / "forced"


def boxmean(f, box, lat=LAT, lon=LON):
    (a, b), (c, d) = box
    m = ((lat >= a) & (lat <= b))[:, None] & ((lon >= c) & (lon <= d))[None, :]
    w = np.cos(np.deg2rad(lat))[:, None] * m
    return np.nansum(f * w, axis=(-2, -1)) / np.sum(w * np.isfinite(f), axis=(-2, -1))


def mon_anom(x, months):
    mm = np.array([int(m[5:7]) for m in months])
    out = np.array(x, dtype="float64", copy=True)
    for k in range(1, 13):
        out[mm == k] -= np.nanmean(out[mm == k], axis=0)
    return out


def season_mean(x, months, s, year_shift=0):
    """{year: mean} of monthly x (T, ...) over season s; DJF labelled by its Jan year; year_shift moves the label."""
    return {y + year_shift: v for y, v in MO.seasonal_fields_any(x, months, s).items()}


def _seasonal_fields_any(x, months, s):
    acc = defaultdict(list)
    ms = E.SMON[s] if s in E.SMON else {"JAS": (7, 8, 9)}[s]
    for i, m in enumerate(months):
        y, mm = int(m[:4]), int(m[5:7])
        if mm in ms:
            acc[y + 1 if (s == "DJF" and mm == 12) else y].append(i)
    return {y: x[ii].mean(0) for y, ii in acc.items() if len(ii) == 3}


MO.seasonal_fields_any = _seasonal_fields_any


def ols_full(X, y):
    """OLS with intercept: coef (k,), t (k,), rss, dof. X (n, k), y (n, ...) (fields allowed)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    Y = y.reshape(len(y), -1)
    B, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    res = Y - Xc @ B; dof = len(y) - Xc.shape[1]
    s2 = (res ** 2).sum(0) / dof
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(Xc.T @ Xc)), s2))
    return B[1:].reshape(X.shape[1], *y.shape[1:]), (B / se)[1:].reshape(X.shape[1], *y.shape[1:]), (res ** 2).sum(0), dof


def cv_gain(Xr, Xf, y, folds=5):
    """Out-of-sample MSE of the restricted and full models (contiguous blocks), -> 1 - MSE_full / MSE_restricted."""
    n = len(y); idx = np.array_split(np.arange(n), folds)
    er, ef = [], []
    for te in idx:
        tr = np.setdiff1d(np.arange(n), te)
        for X, e in ((Xr, er), (Xf, ef)):
            A = np.column_stack([np.ones(len(tr)), X[tr]]); b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            e.append((y[te] - np.column_stack([np.ones(len(te)), X[te]]) @ b) ** 2)
    return float(1.0 - np.mean(np.concatenate(ef)) / np.mean(np.concatenate(er)))


def f_test(Xr, Xf, y):
    _, _, rr, _ = ols_full(Xr, y); _, _, rf, df = ols_full(Xf, y)
    q = Xf.shape[1] - Xr.shape[1]
    F = float(np.squeeze(((rr - rf) / q) / (rf / df)))
    return F, float(stats.f.sf(F, q, df))


# ------------------------------------------------------------------------------------------------ coupled runs
def member_series(name, ref):
    """Monthly series for one coupled member: z500 anomalies (NH grid), AL index, ENSO-free PDO, N34, NP-centre SST."""
    z = np.load(ZI.ZD / f"{name}.npz")
    zm = [str(m) for m in z["months"]]
    Z = z["z"].astype("float64") + 5000.0
    mi = MO.member_indices(MO.PAC / f"{name}.npz", ref)
    pm = mi["months"]
    # align on the pacific months (1950-01 .. 2014-12)
    k0 = zm.index(pm[0]); Z = Z[k0:k0 + len(pm)]
    Za = mon_anom(Z, pm)
    ny = len(pm) // 12
    Za = MO.E.detrend(Za.reshape(len(pm), -1), np.repeat(np.arange(ny), 12) + np.tile(np.arange(12), ny) / 12.0).reshape(Za.shape).astype("float64")
    pz = np.load(MO.PAC / f"{name}.npz")
    q = pz["ts"].astype("float64"); ts = np.where(q == -32768, np.nan, q / 100.0 + 273.15)
    npc = boxmean(mon_anom(ts, pm), NPC_BOX, MO.LAT, MO.LON)
    return dict(months=pm, Z=Za, AL=boxmean(Za, AL_BOX), P=mi["pdo_free"], N=mi["n34"], NPC=npc)


def coupled(ref):
    t0 = time.time()
    names = sorted(Path(f).stem for f in glob.glob(str(ZI.ZD / "*.npz")) if ".part" not in f)
    names = [n for n in names if (MO.PAC / f"{n}.npz").exists()]
    acc = defaultdict(lambda: defaultdict(list))
    scal = defaultdict(lambda: defaultdict(list))
    for i, name in enumerate(names):
        model = name.rsplit("_", 1)[0]
        S = member_series(name, ref)
        pm = S["months"]
        seas = {}
        for s in ("JAS", "SON", "DJF"):
            seas[s] = {k: _seasonal_fields_any(S[k], pm, s) for k in ("Z", "AL", "P", "N", "NPC")}
        yrs = sorted(y for y in seas["DJF"]["Z"] if (y - 1) in seas["SON"]["P"] and (y - 1) in seas["JAS"]["P"])
        g = lambda s, k, lag=0: np.array([seas[s][k][y - lag] for y in yrs])
        zD, alD, pD, nD = g("DJF", "Z"), g("DJF", "AL"), g("DJF", "P"), g("DJF", "N")
        pS, alS, nS, npcS = g("SON", "P", 1), g("SON", "AL", 1), g("SON", "N", 1), g("SON", "NPC", 1)
        pJ, alJ, nJ = g("JAS", "P", 1), g("JAS", "AL", 1), g("JAS", "N", 1)
        # same-season co-variability (for contrast): DJF z on DJF PDO with DJF N34
        B, _, _, _ = ols_full(np.column_stack([nD, pD]), zD); acc["same_DJF"][model].append(B[1])
        # lagged: DJF z on SON PDO | SON AL, SON N34, DJF N34 (primary)
        B, _, _, _ = ols_full(np.column_stack([pS, alS, nS, nD]), zD); acc["lag_SON"][model].append(B[0])
        B, _, _, _ = ols_full(np.column_stack([pS, alS, nS]), zD); acc["lag_SON_noDJFn34"][model].append(B[0])
        B, _, _, _ = ols_full(np.column_stack([pS, nS, nD]), zD); acc["lag_SON_noAL"][model].append(B[0])
        B, _, _, _ = ols_full(np.column_stack([pJ, alJ, nJ, nD]), zD); acc["lag_JAS"][model].append(B[0])
        # tas / pr over the Americas: lagged (controlled) and same-season co-variability
        ds = xr.open_dataset(E.SRC / f"{name}.nc"); yrA = ds["year"].values.astype(int)
        for var in ("tas", "pr"):
            FA = ds[var].sel(season="DJF").values.astype("float64")
            row = {y: i for i, y in enumerate(yrA)}
            okA = [y for y in yrs if y in row and np.isfinite(FA[row[y]]).all()]
            if len(okA) < 30:
                continue
            sel_ = np.array([yrs.index(y) for y in okA])
            A = E.detrend(np.stack([FA[row[y]] for y in okA]), np.array(okA, float)).astype("float64")
            acc[f"{var}_lag_SON"][model].append(ols_full(np.column_stack([pS, alS, nS, nD])[sel_], A)[0][0])
            acc[f"{var}_same_DJF"][model].append(ols_full(np.column_stack([nD, pD])[sel_], A)[0][1])
        ds.close()
        # Granger on the DJF Aleutian-low index
        Xr = np.column_stack([alS, nS, nD]); Xf = np.column_stack([alS, nS, nD, pS])
        F1, p1 = f_test(Xr, Xf, alD)
        scal["granger_P_to_AL_F_p"][model].append(p1); scal["granger_P_to_AL_cv"][model].append(cv_gain(Xr, Xf, alD))
        scal["granger_P_to_AL_coef"][model].append(float(ols_full(Xf, alD)[0][3]))
        # reverse: SON Aleutian low -> DJF PDO beyond SON PDO and N34
        Xr = np.column_stack([pS, nS, nD]); Xf = np.column_stack([pS, nS, nD, alS])
        F2, p2 = f_test(Xr, Xf, pD)
        scal["granger_AL_to_P_F_p"][model].append(p2); scal["granger_AL_to_P_cv"][model].append(cv_gain(Xr, Xf, pD))
        scal["granger_AL_to_P_coef"][model].append(float(ols_full(Xf, pD)[0][3]))
        # same-month contemporaneous: DJF AL on DJF PDO (co-variability) for the index
        scal["same_AL_on_P"][model].append(float(ols_full(np.column_stack([nD, pD]), alD)[0][1]))
        # scale: K of NP-centre SST per sd of the SON ENSO-free PDO
        scal["K_per_sd"][model].append(float(ols_full(pS[:, None], npcS)[0][0]))
        # Frankignoul lagged covariance, monthly, Z_t in DJF months, Z with same-month N34 regressed out
        Z, P, N = S["Z"], S["P"], S["N"]
        bN = (N @ Z.reshape(len(N), -1)) / (N @ N)
        Zr = Z - (N[:, None] * bN[None]).reshape(Z.shape)
        mm = np.array([int(m[5:7]) for m in pm])
        for tau in (1, 2, 3):
            t = np.flatnonzero(np.isin(mm, (12, 1, 2)) & (np.arange(len(mm)) >= tau))
            num = np.tensordot(P[t - tau] - P[t - tau].mean(), Zr[t] - Zr[t].mean(0), axes=(0, 0)) / len(t)
            den = np.mean((P[t] - P[t].mean()) * (P[t - tau] - P[t - tau].mean()))
            acc[f"frank_tau{tau}"][model].append(num / den)
            scal[f"frank_AL_tau{tau}"][model].append(float(boxmean(num / den, AL_BOX)))
        if (i + 1) % 25 == 0:
            print(f"  coupled {i + 1}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    maps, idx = {}, {}
    for key, d in acc.items():
        mn = sorted(d); Sx = np.stack([np.mean(d[m], axis=0) for m in mn])
        mm_, sg, p, agree = E.robust(Sx)
        maps[key] = dict(mm=mm_, sig=sg, per_model=Sx, models=mn)
    for key, d in scal.items():
        mn = sorted(d); v = np.array([np.mean(d[m]) for m in mn])
        if key.endswith("_F_p"):
            # share of members whose own F-test passes at 5 %, per model, then the median over models
            share = np.array([np.mean(np.array(d[m]) < 0.05) for m in mn])
            idx[key] = dict(share_members_p05_median_model=round(float(np.median(share)), 3),
                            share_members_p05_range=[round(float(share.min()), 3), round(float(share.max()), 3)])
        else:
            t, p = stats.ttest_1samp(v, 0.0)
            idx[key] = dict(mean=round(float(v.mean()), 4), median=round(float(np.median(v)), 4),
                            range=[round(float(v.min()), 4), round(float(v.max()), 4)], same_sign=int(max((v > 0).sum(), (v < 0).sum())),
                            models=len(v), p=float(p))
    print(f"coupled done {time.time() - t0:.0f} s", flush=True)
    return maps, idx


_ST = {}


def reg_summary(per_model, var, regions=("Alaska", "Pacific Northwest", "N Plains / Prairies", "Southeast US", "Gulf Coast", "Ohio Valley", "California")):
    """Regional means per model (land-weighted, the impacts study's boxes); robust = across-model t-test FDR 10 % over the
    20 regions AND >= 80 % sign agreement. pr in % of the multi-model DJF normal."""
    import pickle
    if not _ST:
        _ST.update(pickle.load(open(E.STATE, "rb")))
    R = E.regional(per_model, _ST["W"])
    if var == "pr":
        R = 100 * R / E.regional(_ST["clim"]["DJF"][1].mean(0), _ST["W"])
    _, p = stats.ttest_1samp(R, 0, axis=0); ag = np.maximum((R > 0).mean(0), (R < 0).mean(0))
    ok = E.fdr(p) & (ag >= 0.8)
    land = _ST["lf"] >= 0.5
    return {r: (round(float(R[:, E.RNAMES.index(r)].mean()), 2) if ok[E.RNAMES.index(r)] else "n.s.") for r in regions}, land


# ------------------------------------------------------------------------------------------------ observed (ERA5)
def observed(ref):
    em = ZI.era5_monthly()
    keys = sorted(em); months = [f"{y:04d}-{m:02d}" for y, m in keys]
    Z = np.stack([em[k] for k in keys]); Za = mon_anom(Z, months)
    tt = np.array([int(m[:4]) + (int(m[5:7]) - 0.5) / 12 for m in months])
    V = np.column_stack([np.ones_like(tt), tt - tt.mean()])
    Y = Za.reshape(len(Za), -1); Y = Y - V @ np.linalg.lstsq(V, Y, rcond=None)[0]; Za = Y.reshape(Za.shape)
    rm = ref["months"]; ri = {m: i for i, m in enumerate(rm)}
    P = np.array([ref["pdo_free"][ri[m]] if m in ri else np.nan for m in months])
    N = np.array([ref["n34"][ri[m]] if m in ri else np.nan for m in months])
    AL = boxmean(Za, AL_BOX)
    seas = {s: {k: _seasonal_fields_any(v, months, s) for k, v in (("Z", Za), ("AL", AL), ("P", P), ("N", N))} for s in ("SON", "DJF", "JAS")}
    yrs = sorted(y for y in seas["DJF"]["Z"] if (y - 1) in seas["SON"]["P"] and np.isfinite(seas["SON"]["P"][y - 1])
                 and y in seas["DJF"]["P"] and np.isfinite(seas["DJF"]["P"][y]) and y <= 2026)
    g = lambda s, k, lag=0: np.array([seas[s][k][y - lag] for y in yrs])
    zD, alD, pD, nD = g("DJF", "Z"), g("DJF", "AL"), g("DJF", "P"), g("DJF", "N")
    pS, alS, nS = g("SON", "P", 1), g("SON", "AL", 1), g("SON", "N", 1)
    out = {"n": len(yrs), "span": [yrs[0], yrs[-1]]}
    maps = {}
    for key, X, j in (("same_DJF", np.column_stack([nD, pD]), 1), ("lag_SON", np.column_stack([pS, alS, nS, nD]), 0)):
        B, t, _, dof = ols_full(X, zD); p = 2 * stats.t.sf(np.abs(t[j]), dof)
        maps[key] = dict(mm=B[j], sig=E.fdr(p))
    Xr = np.column_stack([alS, nS, nD]); Xf = np.column_stack([alS, nS, nD, pS])
    F, p = f_test(Xr, Xf, alD); B, t, _, _ = ols_full(Xf, alD)
    out["granger_P_to_AL"] = dict(F=round(F, 3), p=round(p, 4), cv_gain=round(cv_gain(Xr, Xf, alD), 3), coef=round(float(B[3]), 2))
    Xr = np.column_stack([pS, nS, nD]); Xf = np.column_stack([pS, nS, nD, alS])
    F, p = f_test(Xr, Xf, pD); B, t, _, _ = ols_full(Xf, pD)
    out["granger_AL_to_P"] = dict(F=round(F, 3), p=round(p, 4), cv_gain=round(cv_gain(Xr, Xf, pD), 3), coef=round(float(B[3]), 4))
    out["same_AL_on_P"] = round(float(ols_full(np.column_stack([nD, pD]), alD)[0][1]), 2)
    out["lag_AL_on_SON_P"] = round(float(ols_full(np.column_stack([pS, alS, nS, nD]), alD)[0][0]), 2)
    return maps, out


# ------------------------------------------------------------------------------------------------ AMIP
def obs_tropics():
    """Observed monthly indices from ERSST v6 (linear trend 1950-2025 removed, 1950-2025 monthly climatology):
    IOB (20S-20N, 40-100E), warm pool WP (10S-10N, 120-160E), North Pacific centre NPC (35-45N, 170E-150W)."""
    d = xr.open_dataset(MO.ERSST)["sst"].sel(time=slice("1950-01-01", None)).sortby("lat")
    months = [f"{t.year:04d}-{t.month:02d}" for t in d.indexes["time"]]
    lat, lon = d.lat.values, d.lon.values
    x = d.values.astype("float64")
    out = {}
    for k, box in (("IOB", ((-20, 20), (40, 100))), ("WP", ((-10, 10), (120, 160))), ("NPC", NPC_BOX)):
        s = boxmean(x, box, lat, lon)
        a = mon_anom(s, months)
        fit = np.array([int(m[:4]) <= 2025 for m in months])
        t = np.arange(len(a)) / 12.0
        c = np.polyfit(t[fit], a[fit], 1); out[k] = a - np.polyval(c, t)
    out["months"] = months
    return out


def amip(ref):
    t0 = time.time()
    tro = obs_tropics()
    rm = {m: i for i, m in enumerate(ref["months"])}; tm = {m: i for i, m in enumerate(tro["months"])}
    files = sorted(glob.glob(str(FORCED / "amip" / "*.npz")))
    acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        z = np.load(f); model = str(z["source_id"]); months = [str(m) for m in z["months"]]
        obsm = lambda k: np.array([ref[k][rm[m]] for m in months])
        trom = lambda k: np.array([tro[k][tm[m]] for m in months])
        idx = {"N": obsm("n34"), "P": obsm("pdo_free"), "E": obsm("E"), "C": obsm("C"),
               "IOB": trom("IOB"), "WP": trom("WP"), "NPC": trom("NPC")}
        for var in ("zg", "tas", "pr", "psl"):
            if var not in z.files:
                continue
            X = mon_anom(z[var].astype("float64"), months)
            D = _seasonal_fields_any(X, months, "DJF")
            I = {k: _seasonal_fields_any(v, months, "DJF") for k, v in idx.items()}
            yrs = sorted(y for y in D if all(y in I[k] for k in I))
            A = np.stack([D[y] for y in yrs]); A = E.detrend(A, np.array(yrs, float)).astype("float64")
            g = lambda k: np.array([I[k][y] for y in yrs])
            for key, cols in (("main", ["N", "IOB", "WP", "P"]), ("n34only", ["N", "P"]), ("withEC", ["N", "E", "C", "IOB", "WP", "P"]),
                              ("npc", ["N", "IOB", "WP", "NPC"])):
                B, _, _, _ = ols_full(np.column_stack([g(c) for c in cols]), A)
                acc[(var, key)][model].append(B[-1])
            acc[(var, "nyears")][model].append(np.full(1, len(yrs)))
    maps = {}
    for (var, key), d in acc.items():
        if key == "nyears":
            continue
        mn = sorted(d); Sx = np.stack([np.mean(d[m], axis=0) for m in mn])
        mm_, sg, _, _ = E.robust(Sx)
        maps[(var, key)] = dict(mm=mm_, sig=sg, per_model=Sx, models=mn, members=int(sum(len(d[m]) for m in mn)))
    npc_sd = float(np.std(np.array(list(_seasonal_fields_any(tro["NPC"], tro["months"], "DJF").values()))))
    print(f"amip done {time.time() - t0:.0f} s ({len(files)} members)", flush=True)
    return maps, npc_sd


# ------------------------------------------------------------------------------------------------ DCPP-C
def pdo_of_delta(dts, dgm):
    """NCEI-scale PDO index change of an SST anomaly field on the Pacific box grid (MO.LAT, MO.LON) with global-mean
    anomaly dgm: (proj(a) - g * proj_one) * calib_slope, the published pattern's own conversion (sst-roni.py)."""
    pat = xr.open_dataset(MO.PDO_PAT)
    eof = pat["eof"]; w = np.cos(np.deg2rad(eof["lat"]))
    a = xr.DataArray(dts, dims=("lat", "lon"), coords={"lat": MO.LAT, "lon": MO.LON}).interp(lat=eof.lat, lon=eof.lon)
    num = float((a * eof * w).sum(skipna=True)); den = float(((eof ** 2) * w).where(a.notnull()).sum())
    return ((num / den) - dgm * float(pat.attrs["proj_one"])) * float(pat.attrs["calib_slope"])


def dcpp_arm(model, exp):
    fs = sorted(glob.glob(str(FORCED / "dcpp" / f"{model}_{exp}_*.npz")))
    out = defaultdict(list); ts, gm = [], []
    for f in fs:
        z = np.load(f)
        for var in ("zg", "psl", "tas", "pr"):
            if var in z.files:
                D = _seasonal_fields_any(z[var].astype("float64"), [str(m) for m in z[f"{var}_months"]], "DJF")
                out[var] += [D[y] for y in sorted(D)]
        if "ts" in z.files:
            q = z["ts"].astype("float64")
            D = _seasonal_fields_any(q, [str(m) for m in z["ts_months"]], "DJF")
            G = _seasonal_fields_any(z["ts_gm"].astype("float64"), [str(m) for m in z["ts_months"]], "DJF")
            ts.append(np.mean([D[y] for y in D], axis=0)); gm.append(np.mean([G[y] for y in G]))
    return {k: np.stack(v) for k, v in out.items()}, (np.mean(ts, axis=0) if ts else None), (float(np.mean(gm)) if gm else None), len(fs)


def dcpp():
    res, maps = {}, {}
    for model in ("IPSL-CM6A-LR", "HadGEM3-GC31-MM", "CNRM-CM6-1"):
        for tag, (pos, neg) in (("NexTrop", ("dcppC-ipv-NexTrop-pos", "dcppC-ipv-NexTrop-neg")), ("ipv", ("dcppC-ipv-pos", "dcppC-ipv-neg"))):
            P, tsP, gP, nP = dcpp_arm(model, pos); N, tsN, gN, nN = dcpp_arm(model, neg)
            if nP < 3 or nN < 3 or "zg" not in P or "zg" not in N:
                continue
            ent = dict(members=[nP, nN], winters=[len(P["zg"]), len(N["zg"])])
            dpdo = None
            if tsP is not None and tsN is not None:
                dts = 0.5 * (tsP - tsN); dg = 0.5 * (gP - gN)
                dpdo = pdo_of_delta(dts, dg)
                ent.update(imposed_pdo_index=round(dpdo, 3), imposed_npc_K=round(float(boxmean(dts, NPC_BOX, MO.LAT, MO.LON)), 3),
                           imposed_global_K=round(dg, 3))
                maps[f"dcpp|{model}|{tag}|dts"] = dts
            for var in ("zg", "psl", "tas", "pr"):
                if var not in P or var not in N:
                    continue
                d = 0.5 * (P[var].mean(0) - N[var].mean(0))
                _, p = stats.ttest_ind(P[var], N[var], axis=0, equal_var=False)
                sg = E.fdr(p)
                maps[f"dcpp|{model}|{tag}|{var}|mm"] = d; maps[f"dcpp|{model}|{tag}|{var}|sig"] = sg
                e = dict(sig_frac=round(float(sg[LAT >= 20].mean() if var in ("zg", "psl") else sg.mean()), 3))
                if var in ("zg", "psl"):
                    al = float(boxmean(d, AL_BOX))
                    e.update(AL_box=round(al, 2), min=round(float(d[LAT >= 20].min()), 1), max=round(float(d[LAT >= 20].max()), 1))
                    if dpdo:
                        e["AL_box_per_pdo_sd"] = round(al / dpdo, 2)
                        e["AL_box_per_K_npc"] = round(al / ent["imposed_npc_K"], 2)
                ent[var] = e
            res[f"{model}|{tag}"] = ent
            print(f"  {model} {tag}: {ent}", flush=True)
    return res, maps


# ------------------------------------------------------------------------------------------------ phase from the PRECEDING season
def phase_prev(ref, lead="SON"):
    """El Nino / neutral DJF winters split by the PRECEDING season's ENSO-free PDO (>= +0.5 vs <= -0.5 sd), matched in
    DJF Nino-3.4 bins; El Nino difference, neutral difference, interaction. tas/pr from all 429 members, z500 from the 147."""
    t0 = time.time()
    names = sorted(Path(f).stem for f in glob.glob(str(E.SRC / "*.nc")) if ".part" not in f)
    names = [n for n in names if (MO.PAC / f"{n}.npz").exists()]
    Sm = MO.Sums(); models = set()
    for i, name in enumerate(names):
        ds = xr.open_dataset(E.SRC / f"{name}.nc"); model = ds.attrs["source_id"]; models.add(model)
        years = ds["year"].values.astype(int); idx34, _ = E.season_index(ds)
        F = {v: ds[v].sel(season="DJF").values.astype("float64") for v in ("tas", "pr")}; ds.close()
        mi = MO.member_indices(MO.PAC / f"{name}.npz", ref)
        Pl = _seasonal_fields_any(mi["pdo_free"], mi["months"], lead)
        zfile = ZI.ZD / f"{name}.npz"
        Zd = None
        if zfile.exists():
            z = np.load(zfile); Zd = ZI.seasonal_fields(z["z"].astype("float64") + 5000.0, [str(m) for m in z["months"]])["DJF"]
        fields = {"tas": {y: F["tas"][k] for k, y in enumerate(years)}, "pr": {y: F["pr"][k] for k, y in enumerate(years)}}
        if Zd is not None:
            fields["z500"] = Zd
        for var, FD in fields.items():
            yrs = np.array([y for y in years if y in idx34["DJF"] and (y - 1) in Pl and y in FD and np.isfinite(FD[y]).all()])
            if len(yrs) < 30:
                continue
            x = E.detrend(np.array([idx34["DJF"][y] for y in yrs]), yrs.astype(float)).astype(float)
            A = E.detrend(np.stack([FD[y] for y in yrs]), yrs.astype(float)).astype("float64")
            P = np.array([Pl[y - 1] for y in yrs])
            neu, en, ln = np.abs(x) < E.NEUTRAL, x >= 0.5, x <= -0.5
            if neu.sum() < E.MIN_NEU:
                continue
            An = A - A[neu].mean(0); b = MO.bin_of(x); pp, pm = P >= MO.PHASE, P <= -MO.PHASE
            for k in range(4):
                for nm, m in (("en+", en & pp), ("en-", en & pm), ("ln+", ln & pp), ("ln-", ln & pm)):
                    Sm.add(model, (var, "DJF", f"{nm}{k}"), An[m & (b == k)], x[m & (b == k)])
            Sm.add(model, (var, "DJF", "neu+0"), An[neu & pp], x[neu & pp]); Sm.add(model, (var, "DJF", "neu-0"), An[neu & pm], x[neu & pm])
        if (i + 1) % 100 == 0:
            print(f"  phase_prev {i + 1}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    out, maps = {}, {}
    for var in ("tas", "pr", "z500"):
        rows = {}
        for key, a, b, bins in (("en", "en+{k}", "en-{k}", range(4)), ("ln", "ln+{k}", "ln-{k}", range(4)), ("neu", "neu+0", "neu-0", (0,))):
            r = {m: MO.matched(Sm, m, var, "DJF", a, b, bins) for m in sorted(models)}
            rows[key] = {m: v for m, v in r.items() if v is not None}
        common = sorted(set(rows["en"]) & set(rows["neu"]))
        res_v = {}
        for key, Sx in (("en", np.stack([rows["en"][m]["diff"] for m in common])),
                        ("neu", np.stack([rows["neu"][m]["diff"] for m in common])),
                        ("int", np.stack([rows["en"][m]["diff"] - rows["neu"][m]["diff"] for m in common]))):
            mm_, sg, _, _ = E.robust(Sx)
            maps[f"phase_{lead}|{var}|{key}|mm"] = mm_; maps[f"phase_{lead}|{var}|{key}|sig"] = sg
            if var == "z500":
                res_v[key] = dict(sig_frac=round(float(sg[LAT >= 20].mean()), 3), AL_box=round(float(boxmean(mm_, AL_BOX)), 1))
            else:
                regs, land = reg_summary(Sx, var)
                res_v[key] = dict(land_robust=round(float(sg[land].mean()), 3), regions=regs)
        res_v["models"] = len(common)
        res_v["n_en_pos"] = int(sum(rows["en"][m]["na"] for m in common)); res_v["n_en_neg"] = int(sum(rows["en"][m]["nb"] for m in common))
        out[var] = res_v
    print(f"phase_prev done {time.time() - t0:.0f} s", flush=True)
    return out, maps


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "coupled"
    ref = MO.obs_indices()
    res_f = RES / "pdo_causality.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    maps_f = RES / "pdo_causality_maps.npz"
    mp = dict(np.load(maps_f)) if maps_f.exists() else {}
    if mode in ("coupled", "all"):
        maps, idx = coupled(ref)
        res["coupled"] = idx
        for k, e in maps.items():
            mp[f"coupled|{k}|mm"] = e["mm"]; mp[f"coupled|{k}|sig"] = e["sig"]
            if k.startswith(("tas", "pr")):
                continue
            res.setdefault("coupled_maps", {})[k] = dict(
                sig_frac=round(float(e["sig"][LAT >= 20].mean()), 3),
                AL_box_gpm=round(float(boxmean(e["mm"], AL_BOX)), 2),
                min=round(float(np.min(e["mm"][LAT >= 20])), 1), max=round(float(np.max(e["mm"][LAT >= 20])), 1),
                pattern_r_vs_same=None)
        for k in ("tas_lag_SON", "tas_same_DJF", "pr_lag_SON", "pr_same_DJF"):
            if k in maps:
                regs, land = reg_summary(maps[k]["per_model"], k[:3] if k.startswith("tas") else "pr")
                res["coupled_maps"][k] = dict(regions=regs, land_robust=round(float(maps[k]["sig"][land].mean()), 3))
        base = maps["same_DJF"]["mm"]
        for k, e in maps.items():
            if k.startswith(("tas", "pr")):
                continue
            res["coupled_maps"][k]["pattern_r_vs_same"] = round(ZI.pattern_r(e["mm"], base), 3)
        om, oo = observed(ref)
        res["observed"] = oo
        for k, e in om.items():
            mp[f"obs|{k}|mm"] = e["mm"]; mp[f"obs|{k}|sig"] = e["sig"]
            oo.setdefault("maps", {})[k] = dict(sig_frac=round(float(e["sig"][LAT >= 20].mean()), 3),
                                                AL_box_gpm=round(float(boxmean(e["mm"], AL_BOX)), 2))
    if mode in ("phase", "all"):
        for lead in ("SON", "JAS"):
            r, m = phase_prev(ref, lead)
            res[f"phase_prev_{lead}"] = r; mp.update(m)
            print(lead, json.dumps(r, indent=1)[:3000])
    if mode in ("dcpp", "all"):
        r, m = dcpp()
        res["dcpp"] = r
        mp.update(m)
        if "coupled|same_DJF|mm" in mp:
            for k in list(r):
                key = f"dcpp|{k.split('|')[0]}|{k.split('|')[1]}|zg|mm"
                if key in mp:
                    r[k]["zg"]["pattern_r_vs_coupled_same"] = round(ZI.pattern_r(mp[key], mp["coupled|same_DJF|mm"]), 3)
    if mode in ("amip", "all"):
        maps, npc_sd = amip(ref)
        res["amip"] = {"npc_sd_K": round(npc_sd, 3)}
        for (var, key), e in maps.items():
            mp[f"amip|{var}|{key}|mm"] = e["mm"]; mp[f"amip|{var}|{key}|sig"] = e["sig"]
            ent = dict(models=len(e["models"]), members=e["members"])
            if var in ("zg", "psl"):
                ent.update(sig_frac=round(float(e["sig"][LAT >= 20].mean()), 3), AL_box=round(float(boxmean(e["mm"], AL_BOX)), 2),
                           min=round(float(e["mm"][LAT >= 20].min()), 1), max=round(float(e["mm"][LAT >= 20].max()), 1))
                if "coupled|same_DJF|mm" in mp and var == "zg":
                    ent["pattern_r_vs_coupled_same"] = round(ZI.pattern_r(e["mm"], mp["coupled|same_DJF|mm"]), 3)
                al = np.array([boxmean(x, AL_BOX) for x in e["per_model"]])
                t_, p_ = stats.ttest_1samp(al, 0.0)
                ent["AL_box_models"] = dict(mean=round(float(al.mean()), 2), range=[round(float(al.min()), 2), round(float(al.max()), 2)],
                                            same_sign=int(max((al > 0).sum(), (al < 0).sum())), p=float(p_))
            else:
                regs, land = reg_summary(e["per_model"], var)
                ent.update(land_robust=round(float(e["sig"][land].mean()), 3), regions=regs)
            res["amip"][f"{var}|{key}"] = ent
    np.savez_compressed(maps_f, **mp)
    res_f.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
