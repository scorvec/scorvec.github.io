#!/usr/bin/env python3
"""QBO zero-wind line: MERRA-2 reference, validation and the Holton-Tan test. LAPTOP, run once (and whenever the
MERRA-2 pull is extended); renders nothing - qbo_zeroline.py draws in Actions.

Inputs
  data/m2_zmu/zmu_<YYYY>.npz     MERRA-2 daily zonal-mean u, 50S-50N, 100-10 hPa (fetch_m2_zmu.py, Earthdata OPeNDAP)
  data/bdc_dc/fp_tail/*.npz      GEOS FP 00Z zonal means (bdc_dc.py's analysis cache) - the GEOS FP vs MERRA-2 check
  reference/strat_history.nc     gap-filled MERRA-2 u(60N, 10 hPa) daily (build_strat_history.py)
  reference/strat_history_events.json   winters table: major SSWs per winter (Charlton-Polvani, Nov-Mar)
  assets/sst/data/cpc_official.json     CPC RONI (official ENSO index since Feb 2026)
  KIT / FU Berlin qbo.dat (Singapore monthly winds, fetched here)
Outputs
  reference/qbo_zeroline_ref.nc   daily index 1980->, climatology by day of year, E/W-QBO composites, recent [u]
  reference/qbo_zeroline_test.json   validation numbers + the test (what the page states)

    python scripts/strat/build_qbo_zeroline.py            # everything
"""
from __future__ import annotations

import glob
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import qbo_zeroline as Z                                                       # noqa: E402

M2 = HERE / "data" / "m2_zmu"
FP = HERE / "data" / "bdc_dc" / "fp_tail"
QBO_URL = "https://www.atmohub.kit.edu/data/qbo.dat"
RNG = np.random.default_rng(20260928)
VARIANTS = {"umin0": dict(umin=0.0), "umin2": dict(umin=2.0), "umin4": dict(umin=4.0), "sigma0": dict(sigma=0.0),
            "sigma4": dict(sigma=4.0)}
NB = 10000


# ------------------------------------------------------------------------------------------------------------ data ---
def load_m2():
    fs = sorted(glob.glob(str(M2 / "zmu_*.npz")))
    t, u = [], []
    for f in fs:
        d = np.load(f)
        t.append(pd.DatetimeIndex(d["time"])); u.append(d["u"]); lev, lat = d["lev"], d["lat"]
    t = t[0].append(t[1:]) if len(t) > 1 else t[0]
    u = np.concatenate(u)
    full = pd.date_range(t[0], t[-1])
    miss = full.difference(t)
    s = pd.Series(np.arange(len(t)), t)
    U = np.full((len(full),) + u.shape[1:], np.nan, np.float32)
    U[full.get_indexer(t)] = u
    return full, list(lev.astype(int)), lat, U, miss


def kit_singapore():
    raw = urllib.request.urlopen(QBO_URL, timeout=60).read().decode("latin-1").splitlines()
    rows = {}
    for ln in raw:
        # fixed width: station (0-4), YYMM (6-9), then 7-character fields 70 50 40 30 20 15 10 hPa - the value in
        # columns 11+7k..16+7k and an optional quality digit at 17+7k (present on the older rows only)
        if len(ln) < 20 or not ln[:5].isdigit() or not ln[6:10].isdigit():
            continue
        yy, mm = int(ln[6:8]), int(ln[8:10])
        y = 1900 + yy if yy >= 50 else 2000 + yy
        v = {}
        for k, L in enumerate((70, 50, 40, 30, 20, 15, 10)):
            f = ln[11 + 7 * k:17 + 7 * k].strip()
            v[L] = int(f) / 10.0 if f.lstrip("-").isdigit() else np.nan
        rows[pd.Timestamp(y, mm, 1)] = v
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------------------------------------- stats ---
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def resid(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X]) if len(X) else np.ones((len(y), 1))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def corr_test(x, y):
    """r, bootstrap 95 % CI (winters resampled), two-sided permutation p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = pearson(x, y)
    n = len(x)
    bs = []
    for _ in range(NB):
        i = RNG.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            bs.append(pearson(x[i], y[i]))
    perm = np.array([pearson(RNG.permutation(x), y) for _ in range(NB)])
    p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (NB + 1)
    return {"r": round(r, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "p": round(float(p), 4), "n": n}


def partial_test(x, y, covs):
    """Partial correlation of x with y given covs; Freedman-Lane permutation (x residuals permuted); bootstrap CI;
    and the nested-OLS F-test p for adding x to covs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = resid(x, covs), resid(y, covs)
    r = pearson(rx, ry)
    perm = np.array([pearson(RNG.permutation(rx), ry) for _ in range(NB)])
    p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (NB + 1)
    n = len(x); bs = []
    for _ in range(NB):
        i = RNG.integers(0, n, n)
        cv = [np.asarray(c, float)[i] for c in covs]
        a, b = resid(x[i], cv), resid(y[i], cv)
        if np.std(a) > 0 and np.std(b) > 0:
            bs.append(pearson(a, b))
    k = len(covs)
    rss0 = (ry ** 2).sum(); rss1 = (resid(y, list(covs) + [x]) ** 2).sum()
    from scipy import stats
    F = (rss0 - rss1) / (rss1 / (n - k - 2))
    return {"partial_r": round(r, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "p_perm": round(float(p), 4), "p_F": round(float(stats.f.sf(F, 1, n - k - 2)), 4),
            "dR2": round(float((rss0 - rss1) / ((y - y.mean()) ** 2).sum()), 3), "n": n}


def compare_r(x1, x2, y):
    """|r(x1,y)| - |r(x2,y)|, bootstrap CI and the share of resamples below zero (one-sided p for x1 better)."""
    x1, x2, y = (np.asarray(a, float) for a in (x1, x2, y))
    d0 = abs(pearson(x1, y)) - abs(pearson(x2, y))
    n = len(y); ds = []
    for _ in range(NB):
        i = RNG.integers(0, n, n)
        ds.append(abs(pearson(x1[i], y[i])) - abs(pearson(x2[i], y[i])))
    ds = np.array(ds)
    return {"diff_abs_r": round(float(d0), 3), "ci": [round(float(np.percentile(ds, 2.5)), 3), round(float(np.percentile(ds, 97.5)), 3)],
            "p_zl_not_better": round(float(np.mean(ds <= 0)), 4)}


def logit_fit(X, y, iters=50):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-9, 1 - 1e-9)
    return b, float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def auc(x, y):
    from scipy.stats import rankdata
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ssw_test(x, y, covs, sign):
    """SSW (0/1 per winter) on x: AUC with permutation p; likelihood-ratio for adding x to covs, permutation p
    (x residualised on covs and permuted)."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    a0 = auc(sign * x, y)
    perm = np.array([auc(sign * RNG.permutation(x), y) for _ in range(NB)])
    p_auc = (np.sum(np.abs(perm - 0.5) >= abs(a0 - 0.5)) + 1) / (NB + 1)
    out = {"auc": round(a0, 3), "p_auc": round(float(p_auc), 4), "n_ssw_winters": int(y.sum()), "n": len(y)}
    if covs:
        _, ll0 = logit_fit([np.asarray(c, float) for c in covs], y)
        _, ll1 = logit_fit([np.asarray(c, float) for c in covs] + [x], y)
        lr = 2 * (ll1 - ll0)
        rx = resid(x, covs); fit = x - rx
        lrs = np.array([2 * (logit_fit([np.asarray(c, float) for c in covs] + [fit + RNG.permutation(rx)], y)[1] - ll0)
                        for _ in range(2000)])
        out.update({"lr_added": round(float(lr), 3), "p_added_perm": round(float((np.sum(lrs >= lr) + 1) / 2001), 4)})
    return out


def clean(o):
    """NaN -> null, numpy scalars -> Python, recursively (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def fdr(ps, q=0.10):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    thr = q * (np.arange(1, m + 1)) / m
    ok = ps[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    passed = np.zeros(m, bool); passed[o[:k]] = True
    return passed


# ----------------------------------------------------------------------------------------------------------- main ---
def main():
    t, lev, lat, U, miss = load_m2()
    print(f"MERRA-2 {t[0].date()} -> {t[-1].date()}, {len(t)} days, {len(miss)} missing", flush=True)
    LV = list(Z.LEVELS)
    li = [lev.index(L) for L in LV]
    U4 = U[:, li]                                                            # (time, 4, lat)
    good = np.isfinite(U4).all(axis=(1, 2))

    # --- daily index, primary rule, both hemispheres
    zl = np.full((len(t), 2, 4), np.nan, np.float32); fl = np.ones((len(t), 2, 4), np.int8)
    for h, hemi in enumerate(("nh", "sh")):
        for k in range(4):
            z, f = Z.zero_line(U4[good, k], lat, hemi)
            zl[good, h, k] = np.where(f == 1, np.nan, z); fl[good, h, k] = f
    print("index done", flush=True)

    val = {}
    # --- validation 1: behaviour of the daily NH index
    months = t.month
    for k, L in enumerate(LV):
        z = zl[:, 0, k]; f = fl[:, 0, k]
        dz = np.abs(np.diff(z)); dz = dz[np.isfinite(dz)]
        on = (months == 10) | (months == 11)
        djf = np.isin(months, (12, 1, 2))
        v = {"undefined_share_by_month": [round(float((f[months == m] == 1).mean()), 3) for m in range(1, 13)],
             "censored_share_ON": round(float((f[on & good] == 2).mean()), 3),
             "censored_share_DJF": round(float((f[djf & good] == 2).mean()), 3),
             "daily_jump_median": round(float(np.median(dz)), 2), "daily_jump_p99": round(float(np.percentile(dz, 99)), 1),
             "share_days_jump_gt10": round(float((dz > 10).mean()), 4)}
        # rule sensitivity on Oct-Nov days: correlation of the primary with variants, and mean difference
        sel = np.where(on & good)[0]
        for tag, kw in (("umin0", dict(umin=0.0)), ("umin2", dict(umin=2.0)), ("sigma0", dict(sigma=0.0)), ("sigma4", dict(sigma=4.0))):
            z2, f2 = Z.zero_line(U4[sel, k], lat, "nh", **kw)
            a = zl[sel, 0, k]; b = np.where(f2 == 1, np.nan, z2); m = np.isfinite(a) & np.isfinite(b)
            v[f"vs_{tag}"] = {"r": round(pearson(a[m], b[m]), 3), "same_within_1deg": round(float((np.abs(a[m] - b[m]) <= 1).mean()), 3)}
        val[str(L)] = v
    print(json.dumps(val["50"]), flush=True)

    # --- validation 2: GEOS FP 00Z vs MERRA-2 daily mean on common days
    fpv = {}
    if FP.exists():
        tdays, tail = Z.load_tail(FP)
        common = [d for d in tdays if d in t and good[t.get_loc(d)]]
        for k, L in enumerate(LV):
            if not common:
                break
            a = np.array([Z.to_grid(U4[t.get_loc(d), k], lat) for d in common])
            b = np.array([tail[L][list(tdays).index(d)] for d in common])
            za, fa = Z.zero_line(a, Z.LAT, "nh"); zb, fb = Z.zero_line(b, Z.LAT, "nh")
            m = (fa != 1) & (fb != 1)
            trop = np.abs(Z.LAT) <= 30
            fpv[str(L)] = {"days": len(common), "u_rms_30S_30N": round(float(np.sqrt(np.mean((a - b)[:, trop] ** 2))), 2),
                           "u_bias_30S_30N": round(float(np.mean((b - a)[:, trop])), 2),
                           "index_mean_diff_fp_minus_m2": round(float(np.mean(zb[m] - za[m])), 2) if m.any() else None,
                           "index_mad": round(float(np.median(np.abs(zb[m] - za[m]))), 2) if m.any() else None,
                           "index_days_defined": int(m.sum())}
        fpv["period"] = f"{common[0].date()}..{common[-1].date()}" if common else None
    val["geosfp_vs_merra2"] = fpv
    print("GEOS FP check", json.dumps(fpv), flush=True)

    # --- climatology of the NH/SH index by day of year (1980-2025, +-7 day window)
    doy = np.clip(t.dayofyear.values, 1, 365)
    inyr = (t.year >= 1980) & (t.year <= 2025)
    zc = np.full((365, 2, 4, 3), np.nan, np.float32); zd = np.zeros((365, 2, 4), np.float32)
    for dd in range(1, 366):
        w = inyr & (np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd)) <= 7) & good
        for h in range(2):
            for k in range(4):
                x = zl[w, h, k]
                zd[dd - 1, h, k] = np.isfinite(x).mean()
                if np.isfinite(x).sum() >= 50:
                    zc[dd - 1, h, k] = np.nanpercentile(x, [10, 50, 90])

    # --- QBO phase per winter from Singapore 50 hPa in Oct-Nov (KIT / FU Berlin)
    kit = kit_singapore()
    years = list(range(1980, 2026))
    sing = {}
    for y in years + [2026]:
        v = [kit.loc[pd.Timestamp(y, m, 1), 50] for m in (10, 11) if pd.Timestamp(y, m, 1) in kit.index]
        sing[y] = float(np.mean(v)) if len(v) == 2 else np.nan
    # composites on a nominal Jul 1 -> Mar 31 calendar (2001/02, no leap day)
    cday = pd.date_range("2001-07-01", "2002-03-31")
    Ug = np.full((len(t), 4, len(Z.LAT)), np.nan, np.float32)
    Ug[good] = Z.to_grid(U4[good], lat)
    comp_u = np.full((2, len(cday), 4, len(Z.LAT)), np.nan, np.float32)
    comp_zl = np.full((2, len(cday), 4, 3), np.nan, np.float32)
    comp_n = [0, 0]
    for p, ph in enumerate(("E", "W")):
        ys = [y for y in years if np.isfinite(sing[y]) and ((sing[y] < 0) if ph == "E" else (sing[y] > 0))]
        comp_n[p] = len(ys)
        stackU, stackZ = [], []
        for y in ys:
            days = [pd.Timestamp(y + (d.year - 2001), d.month, d.day) for d in cday]
            ix = t.get_indexer(days)
            if (ix < 0).any():
                continue
            stackU.append(Ug[ix]); stackZ.append(zl[ix, 0])
        comp_u[p] = np.nanmean(stackU, axis=0)
        comp_zl[p] = np.moveaxis(np.nanpercentile(np.array(stackZ), [25, 50, 75], axis=0), 0, -1)
    print("composites: E", comp_n[0], "W", comp_n[1], flush=True)

    # --- recent [u] on the common grid for the live section (the last 450 days of MERRA-2)
    rt = t[-450:]
    rec = Ug[-450:]

    ds = xr.Dataset(
        {"zl": (("time", "hemi", "lev"), zl), "flag": (("time", "hemi", "lev"), fl),
         "zl_clim": (("doy", "hemi", "lev", "q"), np.moveaxis(zc, 0, 0)),
         "zl_defined": (("doy", "hemi", "lev"), zd),
         "comp_u": (("phase", "cday", "lev", "lat"), comp_u), "comp_zl": (("phase", "cday", "lev", "q"), comp_zl),
         "comp_n": (("phase",), np.array(comp_n)),
         "recent_u": (("recent_time", "lev", "lat"), rec)},
        coords={"time": t, "hemi": ["nh", "sh"], "lev": LV, "doy": np.arange(1, 366), "q": ["p10", "p50", "p90"],
                "phase": ["E", "W"], "cday": cday, "lat": Z.LAT, "recent_time": rt},
        attrs={"title": "QBO zero-wind line: MERRA-2 daily index, climatology and QBO composites",
               "rule": Z.rule_text(), "source": "MERRA-2 M2I3NPASM daily mean of 00/06/12/18Z, zonal mean of 144 "
               "longitudes, Earthdata Cloud OPeNDAP (fetch_m2_zmu.py)", "clim_period": "1980-2025, +-7 day window",
               "comp_q": "p25, p50, p75 (stored in the p10/p50/p90 slots)",
               "qbo_phase": "KIT/FU Berlin Singapore 50 hPa, Oct-Nov mean sign", "missing_days": len(miss),
               "built": pd.Timestamp.today().strftime("%Y-%m-%d")})
    enc = {v: {"zlib": True, "complevel": 5} for v in ds.data_vars}
    ds.to_netcdf(Z.REF, encoding=enc)
    print("wrote", Z.REF, f"{Z.REF.stat().st_size / 1e6:.1f} MB", flush=True)

    # ============================================================================================= the test ===
    sh = xr.open_dataset(HERE / "reference" / "strat_history.nc")
    u10 = sh["u10"].to_series()
    wint = {w["y"]: w for w in json.loads((HERE / "reference" / "strat_history_events.json").read_text())["winters"]}
    roni = json.loads((REPO / "assets" / "sst" / "data" / "cpc_official.json").read_text())["roni"]
    eq = (np.abs(lat) <= 5)
    cw = np.cos(np.deg2rad(lat[eq]))
    rows = []
    for y in years:
        on = (t >= pd.Timestamp(y, 10, 1)) & (t <= pd.Timestamp(y, 11, 30)) & good
        prof = U4[on].mean(axis=0)                                            # (4, lat) ON-mean profile
        r = {"y": y, "sing50": sing[y], "roni_djf": roni.get(f"{y + 1}-01"),
             "u10_djf": float(u10[f"{y}-12-01":f"{y + 1}-02-28"].mean()),
             "u10_jf": float(u10[f"{y + 1}-01-01":f"{y + 1}-02-28"].mean()),
             "ssw": int(wint[y]["n_ssw"] > 0) if y in wint else None}
        for k, L in enumerate(LV):
            z, f = Z.zero_line(prof[k], lat, "nh")
            r[f"zl{L}"] = float(z) if f != 1 else np.nan
            r[f"zl{L}_flag"] = int(f)
            r[f"zl{L}_daily"] = float(np.nanmean(zl[on, 0, k]))
            r[f"m2eq{L}"] = float((prof[k][eq] * cw).sum() / cw.sum())
        # rule sensitivity for the primary (50 hPa, Oct-Nov profile)
        for tag, kw in VARIANTS.items():
            z, f = Z.zero_line(prof[LV.index(50)], lat, "nh", **kw)
            r[f"zl50_{tag}"] = float(z) if f != 1 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).set_index("y")
    df = df.dropna(subset=["sing50", "roni_djf", "u10_djf", "ssw"])
    print(df[["sing50", "m2eq50", "zl50", "zl50_daily", "zl70", "zl30", "zl10", "u10_djf", "ssw", "roni_djf"]].round(1).to_string(), flush=True)

    res = {"n_winters": len(df), "winters": f"{df.index[0]}/{str(df.index[0] + 1)[2:]}..{df.index[-1]}/{str(df.index[-1] + 1)[2:]}"}
    Y = df["u10_djf"].values; YJ = df["u10_jf"].values; S = df["ssw"].values.astype(int)
    SG = df["sing50"].values; EN = df["roni_djf"].values.astype(float)
    res["rival"] = {"sing50_vs_u10_djf": corr_test(SG, Y), "sing50_vs_u10_jf": corr_test(SG, YJ),
                    "m2eq50_vs_u10_djf": corr_test(df["m2eq50"].values, Y),
                    "sing50_ssw": ssw_test(SG, S, [], -1),
                    "sing50_partial_enso_djf": partial_test(SG, Y, [EN])}
    res["zl"] = {}
    for L in LV:
        for kind in ("", "_daily"):
            x = df[f"zl{L}{kind}"].values
            if not np.isfinite(x).all():
                continue
            key = f"{L}{kind or '_profile'}"
            res["zl"][key] = {
                "r_djf": corr_test(x, Y), "r_jf": corr_test(x, YJ),
                "r_with_sing50": round(pearson(x, SG), 3),
                "added_djf_given_sing50_enso": partial_test(x, Y, [SG, EN]),
                "added_jf_given_sing50_enso": partial_test(x, YJ, [SG, EN]),
                "partial_djf_given_enso": partial_test(x, Y, [EN]),
                "better_than_sing50_djf": compare_r(x, SG, Y),
                "ssw": ssw_test(x, S, [SG, EN], +1),
                "censored_winters": int((df[f"zl{L}_flag"] == 2).sum()) if not kind else None}
    res["zl50_rule_sensitivity"] = {}
    for tag in VARIANTS:
        x = df[f"zl50_{tag}"].values
        if np.isfinite(x).all():
            res["zl50_rule_sensitivity"][tag] = {"r_djf": corr_test(x, Y),
                                                 "added_djf_given_sing50_enso": partial_test(x, Y, [SG, EN])}
    # multiple testing over the 8 (level x form) added-value tests, DJF: FDR 10 %
    keys = list(res["zl"])
    ps = [res["zl"][k]["added_djf_given_sing50_enso"]["p_perm"] for k in keys]
    passed = fdr(ps)
    res["fdr_added_djf"] = {k: bool(p) for k, p in zip(keys, passed)}
    prim = res["zl"]["50_profile"]
    sig_alone = prim["r_djf"]["p"] < 0.05
    sig_added = prim["added_djf_given_sing50_enso"]["p_perm"] < 0.05
    sig_ssw = prim["ssw"]["p_auc"] < 0.05
    res["primary"] = "50_profile: zero-line latitude of the Oct-Nov mean [u] profile at 50 hPa vs DJF u(60N,10 hPa)"
    res["significance"] = {"zl_alone": bool(sig_alone), "zl_added_beyond_sing50_enso": bool(sig_added), "zl_ssw_auc": bool(sig_ssw),
                           "sing50_alone": bool(res["rival"]["sing50_vs_u10_djf"]["p"] < 0.05)}
    res["verdict_short"] = ("Tested 1980–2025: the Oct–Nov zero line adds no significant skill for the winter vortex beyond "
                            "Singapore 50 hPa and ENSO, so it is shown as a diagnostic." if not sig_added else
                            "Tested 1980–2025: the Oct–Nov zero line adds significant skill beyond Singapore 50 hPa and ENSO.")
    out = {"built": pd.Timestamp.today().strftime("%Y-%m-%d"), "rule": Z.rule_text(), "validation": val, "test": res,
           "table": df.round(2).reset_index().to_dict(orient="records"), "sing50_on_2026": sing.get(2026)}
    Z.TEST.write_text(json.dumps(clean(out), indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
