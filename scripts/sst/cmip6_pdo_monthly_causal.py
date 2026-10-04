#!/usr/bin/env python3
"""The PDO's FORCED effect on North American temperature and precipitation (and NH 500 hPa height), month by month
(2026-09-27; user: "I'm also interested in PDO's impact on months in the rest of the year and also impacts on north
america temp and precip"). Causal designs only (a same-month regression shows the atmosphere driving the ocean; see
cmip6_pdo_causality.py); the same-month co-variability is computed only as a labelled contrast.

  lagged   coupled CMIP6 historical, 147 members / 16 models (z500 + monthly NA tas/pr): month-m field on the ENSO-free PDO
           of month m-1 with the month-(m-1) atmosphere (the member's own two leading North Pacific z500 PCs, 20-70N
           150E-120W) and Nino-3.4 (m-1 and m) held fixed. Check: PDO of m-2 with the atmosphere of m-2 and m-1 fixed.
           Coastal variant: the member's own coastal NE Pacific SST (30-55N, 135-120W ocean boxes) in place of the PDO.
  amip     26 members / 14 models, observed SST: month-m field on the observed ENSO-free PDO of m-1 with Nino-3.4 (m, m-1),
           Indian Ocean and warm-pool (m) alongside, 1979-2014.
  dcpp     DCPP-C NexTrop pos minus neg by season (IPSL-CM6A-LR, HadGEM3-GC31-MM), per sd of the imposed PDO.
  observed ERA5 t2m and z500 (1959-2026), GPCP (1979-2026): the lagged design with ERA5's own North Pacific z500 PCs.
Tests: CMIP6 / AMIP across-model t-test, BH-FDR 10 % over the map (North American land for tas/pr, 20-90N for z500) AND
>= 80 % sign agreement; DCPP Welch t-test of pos vs neg seasons, FDR 10 %; observed OLS t-test, FDR 10 %. Regional values
(land-weighted boxes, coastal boxes included) with the across-model test, FDR over the regions of a row.

    python scripts/sst/cmip6_pdo_monthly_causal.py [coupled|amip|dcpp|obs|all]
    -> data/cmip6_enso/results/pdo_monthly_causal.{json,npz}
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
import cmip6_pdo_causality as C                                              # noqa: E402

NAD = E.SRC / "na_monthly"
LATN = np.arange(15.0, 75.01, 2.5); LONN = np.arange(190.0, 310.01, 2.5)
LATZ, LONZ = ZI.LAT, ZI.LON
NP_EOF = ((20, 70), (150, 240))
COAST_SST = ((30, 55), (225, 240))
REGIONS = {"Alaska south coast": (55, 61, 200, 225), "Alaska interior": (61, 70, 195, 220),
           "Pacific NW coast": (42, 49, 234.5, 239), "Pacific NW interior": (42, 49, 239, 244),
           "California coast": (32, 42, 235.5, 240), "California interior / Southwest": (32, 42, 240, 250),
           "Northern Plains / Prairies": (45, 57, 245, 265), "Great Lakes": (41, 49, 267, 284), "Northeast US": (38, 46, 280, 293),
           "Southeast US": (25, 36.5, 276, 284.5), "Gulf Coast": (28, 33, 262, 276), "Mexico": (16, 31, 250, 266)}
RN = list(REGIONS)
OUTJ = E.OUT / "pdo_monthly_causal.json"; OUTN = E.OUT / "pdo_monthly_causal.npz"
_W = {}


def weights():
    if not _W:
        lf = E.land_fraction(LATN, LONN)
        W = np.zeros((len(REGIONS), len(LATN), len(LONN)))
        for k, (a, b, c, d) in enumerate(REGIONS.values()):
            m = ((LATN >= a) & (LATN <= b))[:, None] & ((LONN >= c) & (LONN <= d))[None, :]
            W[k] = m * lf * np.cos(np.deg2rad(LATN))[:, None]
            W[k] /= max(W[k].sum(), 1e-12)
        _W.update(lf=lf, W=W, land=lf >= 0.5)
    return _W


def anom_detrend(x, months):
    """Monthly (T, ...) -> anomalies about each calendar month, quadratic trend removed per calendar month."""
    mm = np.array([int(m[5:7]) for m in months]); yy = np.array([int(m[:4]) for m in months], float)
    out = np.full(x.shape, np.nan)
    for k in range(1, 13):
        i = mm == k
        out[i] = E.detrend(x[i] - np.nanmean(x[i], 0), yy[i]).astype("float64")
    return out


def np_pcs(Za, months, lat=LATZ, lon=LONZ, k=2, box=None):
    (a, b), (c, d) = box or NP_EOF
    m = ((lat >= a) & (lat <= b))[:, None] & ((lon >= c) & (lon <= d))[None, :]
    w = np.sqrt(np.cos(np.deg2rad(np.broadcast_to(lat[:, None], m.shape)[m])))
    X = Za[:, m] * w[None]; X = np.where(np.isfinite(X), X, 0.0)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs = U[:, :k] * S[:k]
    return (pcs - pcs.mean(0)) / pcs.std(0)


PNA_EOF = ((20, 70), (150, 300))


def partial_cell(y, x1, X, x2):
    """Coefficient of x1 on y (n, cells) with the common predictors X (n, k) AND a cell-specific predictor x2 (n, cells)
    held fixed (Frisch-Waugh, vectorised over cells). NaN where y or x2 is not finite."""
    ok = np.isfinite(y).all(0) & np.isfinite(x2).all(0)
    Y = np.where(ok, y, 0.0); Z = np.where(ok, x2, 0.0)
    A = np.column_stack([np.ones(len(X)), X]); Q, _ = np.linalg.qr(A)
    res = lambda v: v - Q @ (Q.T @ v)
    r1 = res(x1[:, None])[:, 0]; ry = res(Y); r2 = res(Z)
    S11 = r1 @ r1; S22 = (r2 ** 2).sum(0); S12 = r1 @ r2; S1y = r1 @ ry; S2y = (r2 * ry).sum(0)
    b = (S1y * S22 - S12 * S2y) / np.maximum(S11 * S22 - S12 ** 2, 1e-12)
    return np.where(ok, b, np.nan)


def lagged_design(months, ser, lag):
    """Rows (target month index t) with every needed predictor finite; returns {month: (t_idx, X)} for the lag design."""
    T = len(months); mm = np.array([int(m[5:7]) for m in months])
    out = {}
    for m in range(1, 13):
        t = np.flatnonzero((mm == m) & (np.arange(T) >= lag))
        if lag == 1:
            X = np.column_stack([ser["P"][t - 1], ser["PC1"][t - 1], ser["PC2"][t - 1], ser["N"][t - 1], ser["N"][t]])
        else:
            X = np.column_stack([ser["P"][t - 2], ser["PC1"][t - 2], ser["PC2"][t - 2], ser["PC1"][t - 1], ser["PC2"][t - 1],
                                 ser["N"][t - 2], ser["N"][t]])
        ok = np.isfinite(X).all(1)
        out[m] = (t[ok], X[ok])
    return out


def coef0(X, Y):
    """First-predictor OLS coefficient (with intercept) for fields Y (n, ...), NaN-safe per column."""
    B = C.ols_full(X, np.where(np.isfinite(Y), Y, 0.0))[0][0]
    return np.where(np.isfinite(Y).all(0), B, np.nan)


def reg_summary(S, var):
    """S (models, lat, lon) on the NA grid -> {region: value or 'n.s.'} (across-model t-test FDR 10 % over the regions and
    >= 80 % sign agreement). pr in mm/day (per sd); tas in K."""
    W = weights()["W"]
    R = np.einsum("mij,kij->mk", np.nan_to_num(S), W)
    _, p = stats.ttest_1samp(R, 0, axis=0); ag = np.maximum((R > 0).mean(0), (R < 0).mean(0))
    ok = E.fdr(p) & (ag >= 0.8)
    return {r: (round(float(R[:, k].mean()), 3) if ok[k] else "n.s.") for k, r in enumerate(RN)}, R


def robust_na(S):
    land = weights()["land"]
    mm, sg, p, agree = E.robust(np.where(land[None], np.nan_to_num(S), 0.0))
    sg = sg & land
    # FDR over land cells only: redo on the land subset
    _, pv = stats.ttest_1samp(S[:, land], 0, axis=0)
    ag = np.maximum((S[:, land] > 0).mean(0), (S[:, land] < 0).mean(0))
    s2 = np.zeros(land.shape, bool); s2[land] = E.fdr(pv) & (ag >= 0.8)
    return S.mean(0), s2


def robust_nh(S):
    mm, sg, p, agree = E.robust(S)
    m20 = LATZ >= 20
    _, pv = stats.ttest_1samp(S[:, m20], 0, axis=0)
    ag = np.maximum((S[:, m20] > 0).mean(0), (S[:, m20] < 0).mean(0))
    s2 = np.zeros(S.shape[1:], bool); s2[m20] = E.fdr(pv) & (ag >= 0.8)
    return S.mean(0), s2


# ------------------------------------------------------------------------------------------------ coupled
def coupled(ref):
    t0 = time.time()
    names = sorted(Path(f).stem for f in glob.glob(str(NAD / "*.npz")) if ".part" not in f)
    names = [n for n in names if (ZI.ZD / f"{n}.npz").exists() and (MO.PAC / f"{n}.npz").exists()]
    acc = defaultdict(lambda: defaultdict(list))
    for i, name in enumerate(names):
        model = name.rsplit("_", 1)[0]
        mi = MO.member_indices(MO.PAC / f"{name}.npz", ref)
        pm = mi["months"]
        z = np.load(ZI.ZD / f"{name}.npz"); zm = [str(m) for m in z["months"]]; k0 = zm.index(pm[0])
        Za = anom_detrend(z["z"][k0:k0 + len(pm)].astype("float64") + 5000.0, pm)
        pcs = np_pcs(Za, pm)
        n = np.load(NAD / f"{name}.npz")
        F = {}
        for var in ("tas", "pr"):
            q = n[var].astype("float64"); F[var] = anom_detrend(np.where(q == -32768, np.nan, q / 100.0), pm)
        F["z500"] = Za
        pz = np.load(MO.PAC / f"{name}.npz"); q = pz["ts"].astype("float64")
        ts = np.where(q == -32768, np.nan, q / 100.0 + 273.15)
        cst = C.boxmean(anom_detrend(ts, pm), COAST_SST, MO.LAT, MO.LON); cst = (cst - np.nanmean(cst)) / np.nanstd(cst)
        ser = dict(P=mi["pdo_free"], N=mi["n34"], PC1=pcs[:, 0], PC2=pcs[:, 1])
        ser_c = dict(ser, P=cst)
        # stronger control: three Pacific-North America z500 PCs AND the cell's own previous month
        px = np_pcs(Za, pm, k=3, box=PNA_EOF)
        T = len(pm); mm_s = np.array([int(x[5:7]) for x in pm])
        for Pser, tag in ((ser["P"], "lag1s"), (cst, "coast1s")):
            for m in range(1, 13):
                t = np.flatnonzero((mm_s == m) & (np.arange(T) >= 1))
                X = np.column_stack([px[t - 1], ser["N"][t - 1], ser["N"][t]])
                okr = np.isfinite(X).all(1) & np.isfinite(Pser[t - 1]); t = t[okr]; X = X[okr]
                for var in ("tas", "pr", "z500"):
                    if tag == "coast1s" and var == "z500":
                        continue
                    Y = F[var][t].reshape(len(t), -1); Y1 = F[var][t - 1].reshape(len(t), -1)
                    acc[(tag, var, m)][model].append(partial_cell(Y, Pser[t - 1], X, Y1).reshape(F[var].shape[1:]))
        for lag, sr, tag in ((1, ser, "lag1"), (2, ser, "lag2"), (1, ser_c, "coast1")):
            D = lagged_design(pm, sr, lag)
            for m, (t, X) in D.items():
                for var in ("tas", "pr", "z500"):
                    if tag == "coast1" and var == "z500":
                        continue
                    acc[(tag, var, m)][model].append(coef0(X, F[var][t]))
        mm_ = np.array([int(x[5:7]) for x in pm])
        for m in range(1, 13):                                               # same-month co-variability (contrast only)
            t = np.flatnonzero(mm_ == m)
            X = np.column_stack([ser["P"][t], ser["N"][t]])
            for var in ("tas", "pr", "z500"):
                acc[("covar", var, m)][model].append(coef0(X, F[var][t]))
        if (i + 1) % 25 == 0:
            print(f"  coupled {i + 1}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    return summarise(acc, "coupled")


def summarise(acc, src):
    res, maps = {}, {}
    for (tag, var, m), d in acc.items():
        mn = sorted(d)
        S = np.stack([np.nanmean(d[x], axis=0) for x in mn])
        if len(mn) < E.MIN_MODELS:
            continue
        mm_, sg = (robust_nh(S) if var == "z500" else robust_na(S))
        key = f"{src}|{tag}|{var}|{m:02d}"
        maps[key + "|mm"] = mm_.astype("float32"); maps[key + "|sig"] = sg
        e = dict(models=len(mn))
        if var == "z500":
            e.update(sig_frac=round(float(sg[LATZ >= 20].mean()), 3), AL_box=round(float(C.boxmean(mm_, C.AL_BOX)), 2))
        else:
            land = weights()["land"]
            regs, _ = reg_summary(S, var)
            e.update(land_sig=round(float(sg[land].mean()), 3), regions=regs)
        res[key] = e
    return res, maps


# ------------------------------------------------------------------------------------------------ AMIP
def amip(ref):
    tro = C.obs_tropics()
    d = xr.open_dataset(MO.ERSST)["sst"].sel(time=slice("1950-01-01", None)).sortby("lat")
    cm = [f"{t.year:04d}-{t.month:02d}" for t in d.indexes["time"]]
    cs = C.mon_anom(C.boxmean(d.values.astype("float64"), COAST_SST, d.lat.values, d.lon.values), cm)
    fit = np.array([int(x[:4]) <= 2025 for x in cm]); tt = np.arange(len(cs)) / 12.0
    cs = cs - np.polyval(np.polyfit(tt[fit], cs[fit], 1), tt); cs = (cs - cs[fit].mean()) / cs[fit].std()
    assert cm == tro["months"]
    tro["COAST"] = cs
    rm = {m: i for i, m in enumerate(ref["months"])}; tm = {m: i for i, m in enumerate(tro["months"])}
    acc = defaultdict(lambda: defaultdict(list))
    ia = (np.arange(-60, 75.01, 2.5) >= 15); ja = (np.arange(190, 330.01, 2.5) <= 310)
    for f in sorted(glob.glob(str(C.FORCED / "amip" / "*.npz"))):
        z = np.load(f); model = str(z["source_id"]); months = [str(m) for m in z["months"]]
        g = lambda k: np.array([ref[k][rm[m]] for m in months]); h = lambda k: np.array([tro[k][tm[m]] for m in months])
        P, N, IOB, WP = g("pdo_free"), g("n34"), h("IOB"), h("WP")
        mm_ = np.array([int(x[5:7]) for x in months])
        F = {"z500": anom_detrend(z["zg"].astype("float64"), months)}
        for var in ("tas", "pr"):
            F[var] = anom_detrend(z[var].astype("float64")[:, ia][:, :, ja], months)
        CS = h("COAST")
        for m in range(1, 13):
            t = np.flatnonzero((mm_ == m) & (np.arange(len(months)) >= 1))
            X = np.column_stack([P[t - 1], N[t], N[t - 1], IOB[t], WP[t]])
            X0 = np.column_stack([P[t], N[t], IOB[t], WP[t]])
            Xc = np.column_stack([CS[t], N[t], IOB[t], WP[t]])
            for var in F:
                acc[("amip1", var, m)][model].append(coef0(X, F[var][t]))
                acc[("amip0", var, m)][model].append(coef0(X0, F[var][t]))
                if var != "z500":
                    acc[("amipcoast", var, m)][model].append(coef0(Xc, F[var][t]))
    return summarise(acc, "amip")


# ------------------------------------------------------------------------------------------------ DCPP-C by season
def dcpp(ref):
    res, maps = {}, {}
    ia = (np.arange(-60, 75.01, 2.5) >= 15); ja = (np.arange(190, 330.01, 2.5) <= 310)
    land = weights()["land"]
    for model in ("IPSL-CM6A-LR", "HadGEM3-GC31-MM"):
        arms = {}
        for arm, exp in (("pos", "dcppC-ipv-NexTrop-pos"), ("neg", "dcppC-ipv-NexTrop-neg")):
            fs = sorted(glob.glob(str(C.FORCED / "dcpp" / f"{model}_{exp}_*.npz")))
            per = defaultdict(lambda: defaultdict(list)); ts_ = defaultdict(list); gm_ = defaultdict(list)
            for f in fs:
                z = np.load(f)
                for var, key in (("zg", "z500"), ("tas", "tas"), ("pr", "pr")):
                    x = z[var].astype("float64")
                    if var != "zg":
                        x = x[:, ia][:, :, ja]
                    for s in E.SEASONS:
                        D = C._seasonal_fields_any(x, [str(m) for m in z[f"{var}_months"]], s)
                        per[key][s] += [D[y] for y in sorted(D)]
                if "ts" in z.files:
                    for s in E.SEASONS:
                        D = C._seasonal_fields_any(z["ts"].astype("float64"), [str(m) for m in z["ts_months"]], s)
                        G = C._seasonal_fields_any(z["ts_gm"].astype("float64"), [str(m) for m in z["ts_months"]], s)
                        ts_[s].append(np.mean([D[y] for y in D], 0)); gm_[s].append(np.mean([G[y] for y in G]))
            arms[arm] = (per, ts_, gm_, len(fs))
        (P, tsP, gP, nP), (N, tsN, gN, nN) = arms["pos"], arms["neg"]
        for s in E.SEASONS:
            dpdo = C.pdo_of_delta(0.5 * (np.mean(tsP[s], 0) - np.mean(tsN[s], 0)), 0.5 * (np.mean(gP[s]) - np.mean(gN[s])))
            for var in ("tas", "pr", "z500"):
                a, b = np.stack(P[var][s]), np.stack(N[var][s])
                d = 0.5 * (a.mean(0) - b.mean(0)) / dpdo
                _, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
                if var == "z500":
                    m20 = LATZ >= 20; sg = np.zeros(d.shape, bool); sg[m20] = E.fdr(p[m20])
                    e = dict(sig_frac=round(float(sg[m20].mean()), 3), AL_box=round(float(C.boxmean(d, C.AL_BOX)), 2))
                else:
                    sg = np.zeros(d.shape, bool); sg[land] = E.fdr(p[land])
                    W = weights()["W"]
                    ra, rb = np.einsum("nij,kij->nk", a, W), np.einsum("nij,kij->nk", b, W)
                    _, pr_ = stats.ttest_ind(ra, rb, axis=0, equal_var=False)
                    okr = E.fdr(pr_)
                    val = 0.5 * (ra.mean(0) - rb.mean(0)) / dpdo
                    e = dict(land_sig=round(float(sg[land].mean()), 3),
                             regions={r: (round(float(val[k]), 3) if okr[k] else "n.s.") for k, r in enumerate(RN)})
                e.update(imposed_pdo=round(float(dpdo), 3), members=[nP, nN], seasons=[len(a), len(b)])
                key = f"dcpp|{model}|{var}|{s}"
                res[key] = e; maps[key + "|mm"] = d.astype("float32"); maps[key + "|sig"] = sg
    return res, maps


# ------------------------------------------------------------------------------------------------ observed
def observed(ref):
    res, maps = {}, {}
    em = ZI.era5_monthly(); keys = sorted(em); zm = [f"{y:04d}-{m:02d}" for y, m in keys]
    Za = anom_detrend(np.stack([em[k] for k in keys]), zm)
    pcs = np_pcs(Za, zm)
    rm = {m: i for i, m in enumerate(ref["months"])}
    P = np.array([ref["pdo_free"][rm[m]] if m in rm else np.nan for m in zm]); N = np.array([ref["n34"][rm[m]] if m in rm else np.nan for m in zm])
    ser = dict(P=P, N=N, PC1=pcs[:, 0], PC2=pcs[:, 1])
    # ERA5 t2m on the NA grid
    store = Path.home() / "era5_store" / "wb2_1p5_daily" / "t2m"
    t2m = {}
    for f in sorted(store.glob("t2m_*.nc")):
        d = xr.open_dataset(f); v = d[list(d.data_vars)[0]].transpose("time", "latitude", "longitude").sortby("latitude")
        v = v.sel(latitude=slice(10, 80), longitude=slice(185, 315)).load()
        for m in range(1, 13):
            vm = v.sel(time=v.time.dt.month == m)
            if vm.sizes["time"] >= 28 and np.isfinite(vm.values).all():
                t2m[(int(vm.time.dt.year.values[0]), m)] = vm.mean("time").interp(latitude=LATN, longitude=LONN).values
        d.close()
    gp = E.gpcp_monthly(LATN, LONN)
    fields = {"z500": (zm, Za)}
    for var, dct in (("tas", t2m), ("pr", gp)):
        ks = [k for k in sorted(dct) if f"{k[0]:04d}-{k[1]:02d}" in set(zm)]
        mo = [f"{y:04d}-{m:02d}" for y, m in ks]
        fields[var] = (mo, anom_detrend(np.stack([dct[k] for k in ks]), mo))
    land = weights()["land"]
    for var, (mo, Fv) in fields.items():
        idx = np.array([zm.index(m) for m in mo])
        s = {k: v[idx] for k, v in ser.items()}
        for m, (t, X) in lagged_design(mo, s, 1).items():
            B, tt, _, dof = C.ols_full(X, np.where(np.isfinite(Fv[t]), Fv[t], 0.0))
            p = 2 * stats.t.sf(np.abs(tt[0]), dof)
            dom = (LATZ >= 20)[:, None] * np.ones((1, len(LONZ)), bool) if var == "z500" else land
            sg = np.zeros(B[0].shape, bool); sg[dom] = E.fdr(p[dom])
            key = f"obs|lag1|{var}|{m:02d}"
            maps[key + "|mm"] = B[0].astype("float32"); maps[key + "|sig"] = sg
            e = dict(n=int(len(t)), sig_frac=round(float(sg[dom].mean()), 3))
            if var != "z500":
                W = weights()["W"]
                R = np.einsum("nij,kij->nk", np.nan_to_num(Fv[t]), W)
                Bk, tk, _, dk = C.ols_full(X, R); pk = 2 * stats.t.sf(np.abs(tk[0]), dk); okr = E.fdr(pk)
                e["regions"] = {r: (round(float(Bk[0][k]), 3) if okr[k] else "n.s.") for k, r in enumerate(RN)}
            res[key] = e
    return res, maps


SITE_KEYS = {"cpl_coast": ("coupled|coast1s", ("tas", "pr")), "cpl_pdo": ("coupled|lag1s", ("tas", "pr", "z500")),
             "amip_coast": ("amip|amipcoast", ("tas", "pr")), "amip_pdo": ("amip|amip0", ("tas", "pr", "z500")),
             "obs": ("obs|lag1", ("tas", "pr", "z500")), "covar": ("coupled|covar", ("tas", "pr", "z500"))}


def site() -> int:
    """Compact reference for enso.html "PDO: forced effects by month": reference/enso_pdo_monthly_site.{npz,json}."""
    res = json.loads(OUTJ.read_text()); mp = dict(np.load(OUTN))
    npz = {"latn": LATN.astype("float32"), "lonn": LONN.astype("float32"), "latz": LATZ.astype("float32"), "lonz": LONZ.astype("float32")}
    meta = {"regions": RN, "maps": {}}
    for tag, (src, vars_) in SITE_KEYS.items():
        for var in vars_:
            for m in range(1, 13):
                k = f"{src}|{var}|{m:02d}"
                if k + "|mm" not in mp:
                    continue
                key = f"{tag}|{var}|{m:02d}"
                npz[key + "|f"] = mp[k + "|mm"].astype("float32"); npz[key + "|s"] = mp[k + "|sig"].astype(bool)
                meta["maps"][key] = res[k]
    for model, tag in (("IPSL-CM6A-LR", "ipsl"), ("HadGEM3-GC31-MM", "hadgem")):
        for var in ("tas", "pr", "z500"):
            for sea in E.SEASONS:
                k = f"dcpp|{model}|{var}|{sea}"
                key = f"{tag}|{var}|{sea}"
                npz[key + "|f"] = mp[k + "|mm"].astype("float32"); npz[key + "|s"] = mp[k + "|sig"].astype(bool)
                meta["maps"][key] = res[k]
    out = HERE / "reference"
    np.savez_compressed(out / "enso_pdo_monthly_site.npz", **npz)
    (out / "enso_pdo_monthly_site.json").write_text(json.dumps(meta, separators=(",", ":"), default=float))
    print(f"wrote enso_pdo_monthly_site.npz ({(out / 'enso_pdo_monthly_site.npz').stat().st_size / 1e6:.1f} MB, {len(meta['maps'])} maps)")
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "site":
        return site()
    ref = MO.obs_indices()
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    mp = dict(np.load(OUTN)) if OUTN.exists() else {}
    for name, fn in (("coupled", coupled), ("amip", amip), ("dcpp", dcpp), ("obs", observed)):
        if mode in (name, "all"):
            t0 = time.time(); r, m = fn(ref); res.update(r); mp.update(m)
            print(f"{name}: {len(r)} entries, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUTN, **mp)
    OUTJ.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
