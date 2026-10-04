"""Age of stratospheric air from the MERRA-2 GMI replay: climatology, seasonal cycle, trend, ENSO and QBO (laptop, once).

Input: scripts/strat/data/aoa/aoa_YYYY.npz from fetch_m2gmi_aoa.py (monthly zonal-mean GMI age-of-air tracer `aoadays`,
1980-2019, 250-0.1 hPa, 2-deg latitude). Output: scripts/strat/reference/age_of_air.nc (small, committed), drawn in
Actions by age_of_air.py. Printed diagnostics are the numbers quoted on the page.

WHAT THE TRACER IS. GMI carries a clock tracer: a surface source that grows linearly in time, so the mixing ratio deficit
at a point is the mean time since the air there was last at the surface. It is the model's MEAN AGE (the first moment of
the age spectrum), referenced to the surface, not to the tropical tropopause as most balloon estimates are; the
difference is the ~0.2-0.3 years the air spends getting to the tropopause.

RECORD CHECKS (see main() output): spin-up at the start, and steps at the four MERRA-2 production streams (1980, 1992,
2000, 2010 starts), where a replay run from a new initial state would show up as a jump in an integrating tracer.

STATISTICS. Monthly anomalies from the monthly climatology; regression on [1, t, ENSO(t - L), QBO u50(t), QBO u30(t)] with
AR(1) errors (Prais-Winsten, two iterations: age anomalies are persistent, lag-1 autocorrelation 0.8-0.95, so ordinary
least squares would overstate significance several-fold). Volcanic windows are excluded (El Chichon Apr 1982-Mar 1984,
Pinatubo Jun 1991-Dec 1993): the volcanic warming coincides with El Nino events in both and would load onto ENSO.
ENSO = CPC relative Nino-3.4 (RONI, the site's official index) from nino_history.json; QBO = CPC 30 and 50 hPa equatorial
zonal-wind indices. Maps are tested cell by cell (two-sided t, n - k dof of the transformed regression) and the
false-discovery rate is controlled at 10 % over every cell drawn (Benjamini-Hochberg; Wilks 2016).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AOA = HERE / "data" / "aoa"
TD = REPO / "scripts" / "telecon" / "data"
OUT = HERE / "reference" / "age_of_air.nc"
SUB0 = 2004                                # start of the sub-period trend
VOLCANIC = [("1982-04", "1984-03"), ("1991-06", "1993-12")]
STREAMS = ["1992-01", "2000-01", "2010-01"]
P_TOP_MAP, P_BOT_MAP = 1.0, 200.0          # the cells drawn (and FDR-tested) on the maps
ENSO_LAGS = range(0, 9)


def load_aoa():
    fs = sorted(AOA.glob("aoa_*.npz"))
    yrs = [int(f.stem[4:]) for f in fs]
    miss = sorted(set(range(1980, 2020)) - set(yrs))
    if miss:
        print("MISSING YEARS:", miss)
    arrs, times = [], []
    for f in fs:
        z = np.load(f)
        arrs.append(z["aoa_days"].astype(float) / 365.25)
        times.append(pd.to_datetime(z["time"]))
        lev, lat = z["lev"].astype(float), z["lat"].astype(float)
    A = np.concatenate(arrs, 0)
    t = pd.DatetimeIndex(np.concatenate(times)).to_period("M").to_timestamp()
    return A, t, lev, lat


def load_qbo(level):
    rows = {}
    for line in open(TD / f"cpc_qbo_u{level}.txt"):
        if "STANDARDIZED" in line.upper():
            break
        m = re.match(r"^\s*(\d{4})(.*)$", line)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(2))]
        for k, x in enumerate(v[:12]):
            if x > -900:
                rows[pd.Timestamp(int(m.group(1)), k + 1, 1)] = x
    return pd.Series(rows).sort_index()


def load_roni():
    j = json.load(open(REPO / "assets" / "sst" / "data" / "nino_history.json"))
    return pd.Series(j["series"]["roni"]["anom"], index=pd.to_datetime([m + "-01" for m in j["months"]]), dtype=float)


def coslat_mean(A, lat, lo, hi):
    w = np.cos(np.deg2rad(lat)) * ((lat >= lo) & (lat <= hi))
    return (A * w).sum(-1) / w.sum()


def prais_winsten(y, X, start=None, iters=2):
    """GLS with AR(1) errors. y (n, m) for m series sharing the design X (n, k); `start` marks rows that begin a new
    contiguous segment (after an excluded window), which are scaled like the first row instead of differenced.
    Returns beta (k, m), se (k, m), rho (m,), dof."""
    n, k = X.shape
    st = np.zeros(n, bool) if start is None else np.asarray(start, bool).copy()
    st[0] = True
    pair = ~st[1:]                                   # rows i>0 whose predecessor is adjacent in time
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = np.zeros(y.shape[1]); se = np.zeros_like(beta)
    for _ in range(iters):
        r = y - X @ beta
        rho = np.clip((r[1:][pair] * r[:-1][pair]).sum(0) / (r[:-1][pair] ** 2).sum(0), -0.97, 0.97)
        beta = np.empty((k, y.shape[1])); se = np.empty((k, y.shape[1]))
        for j in range(y.shape[1]):
            c = np.sqrt(1 - rho[j] ** 2)
            ys = np.empty(n); Xs = np.empty_like(X, dtype=float)
            ys[st] = c * y[st, j]; Xs[st] = c * X[st]
            nx = np.where(~st)[0]
            ys[nx] = y[nx, j] - rho[j] * y[nx - 1, j]; Xs[nx] = X[nx] - rho[j] * X[nx - 1]
            b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            e = ys - Xs @ b
            s2 = (e @ e) / (n - k)
            cov = s2 * np.linalg.inv(Xs.T @ Xs)
            beta[:, j] = b; se[:, j] = np.sqrt(np.diag(cov))
    return beta, se, rho, n - k


def segstart(ok):
    """For the retained rows of a monthly boolean mask: True where the previous month was not retained."""
    idx = np.where(ok)[0]
    return np.r_[True, np.diff(idx) > 1]


def tp(t, dof):
    from scipy import stats
    return 2 * stats.t.sf(np.abs(t), dof)


def bh(p, alpha=0.10):
    """Benjamini-Hochberg: boolean mask of discoveries among the finite p."""
    flat = p[np.isfinite(p)]
    out = np.zeros(p.shape, bool)
    if not flat.size:
        return out
    s = np.sort(flat)
    ok = s <= alpha * np.arange(1, len(s) + 1) / len(s)
    if ok.any():
        out = np.isfinite(p) & (p <= s[np.where(ok)[0].max()])
    return out


def main():
    A, t, lev, lat = load_aoa()
    nt, nk, nj = A.shape
    print(f"record {t[0]:%Y-%m}..{t[-1]:%Y-%m}, {nt} months, levels {lev.max():g}-{lev.min():g} hPa, {nj} latitudes")
    kmap = (lev <= P_BOT_MAP) & (lev >= P_TOP_MAP)

    # ---------------------------------------------------------------- record checks: spin-up, drift, stream steps
    clim_m = np.stack([A[t.month == m].mean(0) for m in range(1, 13)])
    anom = A - clim_m[t.month.values - 1]
    glob_w = np.cos(np.deg2rad(lat))
    checks = {}
    for L in (100, 50, 20, 10, 5, 1):
        k = int(np.argmin(np.abs(lev - L)))
        g = (anom[:, k] * glob_w).sum(-1) / glob_w.sum()
        ann = pd.Series(g, index=t).groupby(t.year).mean()
        jumps = {}
        for s in STREAMS:
            s0 = pd.Timestamp(s)
            a = pd.Series(g, index=t)[s0 - pd.DateOffset(months=12):s0 - pd.DateOffset(months=1)]
            b = pd.Series(g, index=t)[s0:s0 + pd.DateOffset(months=11)]
            jumps[s] = round(float(b.mean() - a.mean()), 3)
        checks[L] = {"annual": {int(y): round(float(v), 3) for y, v in ann.items()},
                     "step_12mo_across_stream": jumps,
                     "first_3y_slope_yr_per_yr": round(float(np.polyfit(np.arange(36) / 12, g[:36], 1)[0]), 3),
                     "later_slope_yr_per_decade": round(float(np.polyfit(np.arange(nt - 36) / 120, g[36:], 1)[0]), 3)}
        print(f"{L:>5} hPa global-mean age anomaly by year:",
              " ".join(f"{y % 100:02d}:{v:+.2f}" for y, v in ann.items()))
        print(f"       steps across stream starts (12 mo after - before): {jumps}; "
              f"slope 1980-82 {checks[L]['first_3y_slope_yr_per_yr']:+.3f} yr/yr, later {checks[L]['later_slope_yr_per_decade']:+.3f} yr/dec")
    START = int(sys.argv[1]) if len(sys.argv) > 1 else 1980   # set after reading the checks
    use = t.year >= START
    A, t, anom = A[use], t[use], None
    nt = len(t)
    clim_m = np.stack([A[t.month == m].mean(0) for m in range(1, 13)])
    iav_m = np.stack([A[t.month == m].std(0, ddof=1) for m in range(1, 13)])
    anom = A - clim_m[t.month.values - 1]
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    clim_s = np.stack([clim_m[[m - 1 for m in ms]].mean(0) for ms in seasons.values()] + [clim_m.mean(0)])

    # ---------------------------------------------------------------- validation numbers
    def box(Aann, p, lo, hi):
        k = int(np.argmin(np.abs(lev - p)))
        return float(coslat_mean(Aann[k], lat, lo, hi))
    ann = clim_m.mean(0)
    val = {f"{p}hPa_{nm}": round(box(ann, p, lo, hi), 2) for p in (70, 50, 30, 20, 10)
           for nm, (lo, hi) in {"tropics_10S_10N": (-10, 10), "NH_35_55N": (35, 55), "SH_35_55S": (-55, -35),
                                "NH_polar_65_90N": (65, 90), "SH_polar_65_90S": (-90, -65)}.items()}
    print("annual-mean age (years):", json.dumps(val))

    # ---------------------------------------------------------------- regression: trend, ENSO, QBO
    roni, u50, u30 = load_roni(), load_qbo(50), load_qbo(30)
    keep = np.ones(nt, bool)
    for a, b in VOLCANIC:
        keep &= ~((t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b) + pd.offsets.MonthEnd(0)))
    tt = (t - pd.Timestamp("2000-01-01")).days.values / 3652.5         # decades
    q50 = u50.reindex(t).values; q30 = u30.reindex(t).values
    q50s, q30s = (q50 - np.nanmean(q50)) / np.nanstd(q50), (q30 - np.nanmean(q30)) / np.nanstd(q30)
    Y = anom[:, kmap].reshape(nt, -1)
    trop70 = int(np.argmin(np.abs(lev[kmap] - 70)))
    jtrop = (lat >= -10) & (lat <= 10)
    lagtab = {}
    fits = {}
    for L in ENSO_LAGS:
        e = roni.reindex(t - pd.DateOffset(months=L)).values
        ok = keep & np.isfinite(e) & np.isfinite(q50s) & np.isfinite(q30s)
        # contiguous-ish sample: Prais-Winsten on the retained months in order (the gaps are short relative to the record)
        X = np.column_stack([np.ones(ok.sum()), tt[ok], e[ok], q50s[ok], q30s[ok]])
        yb = coslat_mean(anom[ok][:, kmap][:, trop70], lat, -10, 10)[:, None]
        b, s, rho, dof = prais_winsten(yb, X, segstart(ok))
        lagtab[L] = {"coef_months_per_K": round(float(b[2, 0] * 12), 3), "t": round(float(b[2, 0] / s[2, 0]), 2),
                     "rho": round(float(rho[0]), 3)}
    print("ENSO lag scan, tropical 70 hPa (months of age per K RONI):", lagtab)
    L_best = max(lagtab, key=lambda L: abs(lagtab[L]["t"]))
    e = roni.reindex(t - pd.DateOffset(months=L_best)).values
    ok = keep & np.isfinite(e) & np.isfinite(q50s) & np.isfinite(q30s)
    X = np.column_stack([np.ones(ok.sum()), tt[ok], e[ok], q50s[ok], q30s[ok]])
    b, s, rho, dof = prais_winsten(Y[ok], X, segstart(ok))
    shp = (int(kmap.sum()), nj)
    names = ["const", "trend", "enso", "qbo50", "qbo30"]
    res = {}
    for i, nm in enumerate(names[1:], start=1):
        coef = b[i].reshape(shp); se = s[i].reshape(shp)
        p = tp(coef / se, dof)
        res[nm] = (coef, se, p, bh(p))
        print(f"{nm}: {int(res[nm][3].sum())} of {p.size} cells pass FDR 10 %")
    print(f"ENSO lag used {L_best} months; n = {ok.sum()} months; median AR(1) rho {np.median(rho):.2f}")

    # the same model on 2004-2019 only (one MERRA-2 stream apart from 2010, Aura MLS temperatures assimilated, no
    # volcanic window): is there a trend inside a steadier observing system?
    ok04 = ok & (t >= pd.Timestamp(f"{SUB0}-01-01"))
    X04 = np.column_stack([np.ones(ok04.sum()), tt[ok04], e[ok04], q50s[ok04], q30s[ok04]])
    b4, s4, rho4, dof4 = prais_winsten(Y[ok04], X04, segstart(ok04))
    c4, se4 = b4[1].reshape(shp), s4[1].reshape(shp)
    p4 = tp(c4 / se4, dof4)
    res["trend04"] = (c4, se4, p4, bh(p4))
    print(f"trend {SUB0}-{t[-1].year}: {int(res['trend04'][3].sum())} of {p4.size} cells pass FDR 10 %")

    # box series + box trends (with the same model)
    boxes = {"engel_30_50N_30_5hPa": ((30, 50), (5, 30)), "tropics_70hPa": ((-10, 10), (70, 70)),
             "tropics_50hPa": ((-10, 10), (50, 50)), "nh_mid_50hPa": ((35, 55), (50, 50)),
             "sh_mid_50hPa": ((-55, -35), (50, 50)), "nh_mid_10hPa": ((35, 55), (10, 10)),
             "sh_mid_10hPa": ((-55, -35), (10, 10))}
    box_out = {}
    for nm, ((lo, hi), (pt, pb)) in boxes.items():
        kk = (lev >= pt - 1e-6) & (lev <= pb + 1e-6)
        raw = coslat_mean(A[:, kk].mean(1), lat, lo, hi)
        an = coslat_mean(anom[:, kk].mean(1), lat, lo, hi)
        bb, ss, rr, dd = prais_winsten(an[ok][:, None], X, segstart(ok))
        coefs = {n: {"coef": float(bb[i, 0]), "se": float(ss[i, 0]), "p": float(tp(bb[i, 0] / ss[i, 0], dd))}
                 for i, n in enumerate(names) if i}
        b4b, s4b, _, d4b = prais_winsten(an[ok04][:, None], X04, segstart(ok04))
        coefs["trend04"] = {"coef": float(b4b[1, 0]), "se": float(s4b[1, 0]), "p": float(tp(b4b[1, 0] / s4b[1, 0], d4b))}
        early = raw[(t.year >= 1980) & (t.year <= 1991)].mean(); late = raw[t.year >= SUB0].mean()
        coefs["late_minus_early"] = float(late - early)
        box_out[nm] = {"raw": raw, "anom": an, "fit": coefs, "rho": float(rr[0]), "mean": float(raw.mean())}
        print(f"{nm}: {SUB0}+ trend {coefs['trend04']['coef']:+.3f} +- {coefs['trend04']['se']:.3f} (p {coefs['trend04']['p']:.3f}); "
              f"{SUB0}-2019 minus 1980-1991 {coefs['late_minus_early']:+.2f} yr")
        print(f"{nm}: mean {raw.mean():.2f} yr; trend {coefs['trend']['coef']:+.3f} +- {coefs['trend']['se']:.3f} yr/dec "
              f"(p {coefs['trend']['p']:.3f}); ENSO {coefs['enso']['coef'] * 12:+.2f} mo/K (p {coefs['enso']['p']:.3f}); "
              f"QBO50 {coefs['qbo50']['coef'] * 12:+.2f} mo/sd (p {coefs['qbo50']['p']:.3f}); "
              f"QBO30 {coefs['qbo30']['coef'] * 12:+.2f} mo/sd (p {coefs['qbo30']['p']:.3f})")
    # Engel et al. (2017) period for the like-for-like comparison: to 2016
    kk = (lev >= 5 - 1e-6) & (lev <= 30 + 1e-6)
    eng = coslat_mean(A[:, kk].mean(1), lat, 30, 50)
    ann_e = pd.Series(eng, index=t).groupby(t.year).mean()
    yy = ann_e.index.values.astype(float)
    sel = yy <= 2016
    bb, ss, rr, dd = prais_winsten(ann_e.values[sel][:, None], np.column_stack([np.ones(sel.sum()), (yy[sel] - 2000) / 10]))
    engel = {"period": f"{int(yy[sel][0])}-2016", "trend": float(bb[1, 0]), "se": float(ss[1, 0]),
             "p": float(tp(bb[1, 0] / ss[1, 0], dd)), "rho": float(rr[0])}
    print(f"Engel box annual means {engel['period']}: {engel['trend']:+.3f} +- {engel['se']:.3f} yr/dec, p {engel['p']:.3f}")

    # ---------------------------------------------------------------- write
    import xarray as xr
    levm = lev[kmap]
    ds = xr.Dataset(
        {"clim_month": (("month", "lev", "lat"), clim_m.astype(np.float32)),
         "iav_month": (("month", "lev", "lat"), iav_m.astype(np.float32)),
         "clim_season": (("season", "lev", "lat"), clim_s.astype(np.float32)),
         **{f"{nm}_coef": (("levm", "lat"), res[nm][0].astype(np.float32)) for nm in res},
         **{f"{nm}_se": (("levm", "lat"), res[nm][1].astype(np.float32)) for nm in res},
         **{f"{nm}_p": (("levm", "lat"), res[nm][2].astype(np.float32)) for nm in res},
         **{f"{nm}_sig": (("levm", "lat"), res[nm][3].astype(np.int8)) for nm in res},
         **{f"box_{nm}": (("time",), v["raw"].astype(np.float32)) for nm, v in box_out.items()},
         **{f"boxanom_{nm}": (("time",), v["anom"].astype(np.float32)) for nm, v in box_out.items()},
         "roni_lagged": (("time",), roni.reindex(t - pd.DateOffset(months=L_best)).values.astype(np.float32)),
         "qbo_u50": (("time",), q50.astype(np.float32)), "qbo_u30": (("time",), q30.astype(np.float32)),
         "used_in_regression": (("time",), (keep & np.isfinite(e) & np.isfinite(q50s) & np.isfinite(q30s)).astype(np.int8))},
        coords={"month": np.arange(1, 13), "lev": lev, "levm": levm, "lat": lat, "season": list(seasons) + ["ANN"],
                "time": t.values})
    ds.attrs.update({
        "source": "MERRA-2 GMI replay (NASA GMAO), tavgM_3d_dac_Np aoadays, NCCS OPeNDAP; zonal means of 72 longitudes",
        "period": f"{t[0]:%Y-%m}..{t[-1]:%Y-%m}", "start_year": START,
        "regression": "anomaly ~ 1 + t + RONI(t-L) + u50s + u30s, AR(1) errors (Prais-Winsten), volcanic windows excluded",
        "enso_lag_months": L_best, "enso_lag_scan": json.dumps(lagtab), "n_months_regression": int(ok.sum()),
        "dof": int(dof), "qbo_sd_ms": json.dumps({"u50": float(np.nanstd(q50)), "u30": float(np.nanstd(q30))}),
        "fdr": "Benjamini-Hochberg 10 % over 200-1 hPa, all latitudes", "volcanic_excluded": json.dumps(VOLCANIC),
        "validation": json.dumps(val), "record_checks": json.dumps(checks),
        "box_fits": json.dumps({k: {"fit": v["fit"], "rho": v["rho"], "mean": v["mean"]} for k, v in box_out.items()}),
        "engel_comparison": json.dumps(engel),
        "n_sig": json.dumps({nm: int(res[nm][3].sum()) for nm in res}), "n_cells": int(kmap.sum() * nj),
        "sub_period_start": SUB0, "n_months_sub": int(ok04.sum())})
    enc = {v: {"zlib": True, "complevel": 5} for v in ds.data_vars}
    OUT.parent.mkdir(exist_ok=True)
    ds.to_netcdf(OUT, encoding=enc)
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
