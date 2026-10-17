#!/usr/bin/env python3
"""Zonal-mean structure of ERA5's angular momentum analysis increments, the in-window residual, and the AIFS-ENS
forecast AAM drift - the increments section of the El Nino mountain-torque study page.

ERA5's open archive has no increment fields, so the increment is estimated by the jump method at the two 4D-Var
window starts (the hourly budget shows the jumps in the 09->10 and 21->22 UTC intervals). For any quantity X with a
budget X(h) - X(h-1) = S(h) dt + R(h):

    R(h)  = [X(h) - X(h-1)] - S(h) dt                     (S = the band's surface torques; transport is unknown)
    J10   = R(10) - 1/2 [R(09) + R(11)]                   J22 = R(22) - 1/2 [R(21) + R(23)]

i.e. the residual in the window-start hour minus the mean of the hours either side. The smooth part of everything
that is not an increment (transport between bands, tides, the local trend) cancels to first order; what remains is
the step between two analysis trajectories. Inputs (committed by torque-history.yml):
  torque_bands5_jumps_2025_2026.csv.gz  5-degree band motion/mass AAM and torques at 08-11 and 20-23 UTC, daily
  torque_u37_jumps_2025_2026.nc         37-level relative-AAM density jumps per layer and band (no torque term)
  torque_1h_2026.csv.gz                 the hourly 2026 global budget (in-window residual vs range torques)
  aam_forecast_archive.nc               AIFS-ENS ensemble-mean AAM forecasts to 2026-10-16 (forecast drift)

    python scripts/mjo/src/torque_increments.py --out <dir>
    python scripts/mjo/src/torque_increments.py --out <dir> --wind-sign     # only the strengthen / weaken test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import torque_enso_study as S

REF = S.REF
B5 = REF / "torque_bands5_jumps_2025_2026.csv.gz"
U37 = REF / "torque_u37_jumps_2025_2026.nc"
H1 = REF / "torque_1h_2026.csv.gz"
FC = REF / "aam_forecast_archive.nc"
HS = 3600 * 1e18 / 1e25  # 1 Hadley over one hour, in 1e25 kg m2/s
LATC = -87.5 + 5 * np.arange(36)
RNG = np.random.default_rng(20261013)


def effn_ttest(X):
    """Two-sided p of mean != 0 per column of X (time, ...), with N_eff = N (1 - r1) / (1 + r1) from the lag-1
    autocorrelation of each column (r1 clipped to [0, 0.9])."""
    from scipy import stats

    X = np.asarray(X, float)
    n = np.isfinite(X).sum(0)
    m = np.nanmean(X, 0)
    v = np.nanvar(X, 0, ddof=1)
    a = X - m
    r1 = np.clip(np.nan_to_num(np.nanmean(a[:-1] * a[1:], 0) / np.where(v > 0, v, np.nan)), 0, 0.9)
    neff = np.maximum(n * (1 - r1) / (1 + r1), 2)
    t = m / np.sqrt(np.where(v > 0, v, np.nan) / neff)
    return 2 * stats.t.sf(np.abs(t), neff - 1)


def block_boot_mean(X, block=10, n=4000, rng=RNG):
    """X (days, ...): mean, 95% CI and two-sided p of the mean, moving blocks of `block` days."""
    X = np.asarray(X, float)
    N = X.shape[0]
    m = np.nanmean(X, 0)
    nb = int(np.ceil(N / block))
    st = rng.integers(0, N - block + 1, (n, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :N]
    bm = np.stack([np.nanmean(X[i], 0) for i in idx])
    lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
    c = bm - bm.mean(0)  # null: centred bootstrap distribution
    p = (np.sum(np.abs(c) >= np.abs(m)[None], 0) + 1) / (n + 1)
    return m, lo, hi, p


def band_jumps():
    d = pd.read_csv(B5, parse_dates=["time"]).set_index("time").sort_index()
    days = sorted(set(d.index.normalize()))
    rows = []
    for day in days:
        hrs = {h: day + pd.Timedelta(hours=h) for h in (8, 9, 10, 11, 20, 21, 22, 23)}
        if not all(t in d.index for t in hrs.values()):
            continue
        g = {h: d.loc[t] for h, t in hrs.items()}
        rec = {"day": day}
        for k in range(36):
            kk = f"{k:02d}"

            def M(h, part, g=g, kk=kk):
                if part == "r":
                    return g[h][f"aam_{kk}"]
                if part == "m":
                    return g[h][f"aamm_{kk}"]
                return g[h][f"aam_{kk}"] + g[h][f"aamm_{kk}"]

            def T(h, g=g, kk=kk):  # torque over (h-1, h], Hadley
                return 0.5 * (g[h - 1][f"mtn_{kk}"] + g[h][f"mtn_{kk}"]) + g[h][f"fric_{kk}"] + g[h][f"gwd_{kk}"]

            for w, (a, b, c) in {"10": (9, 10, 11), "22": (21, 22, 23)}.items():
                for part in ("t", "r", "m"):
                    R = {h: M(h, part) - M(h - 1, part) for h in (a, b, c)}
                    rec[f"J{w}{part}_{kk}"] = R[b] - 0.5 * (R[a] + R[c])
                tq = {h: T(h) * HS for h in (a, b, c)}
                rec[f"J{w}T_{kk}"] = tq[b] - 0.5 * (tq[a] + tq[c])  # torque part across the jump
                rec[f"J{w}_{kk}"] = rec[f"J{w}t_{kk}"] - rec[f"J{w}T_{kk}"]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("day")


MAPS = REF / "torque_u37maps_2025_2026.nc"
MAPS_REG = REF / "torque_u37maps_2025_2026_regions.csv"


def maps_analysis(od: Path):
    """Low-level increment maps: FDR significance per cell and the regional split of the >=850 hPa AM increment."""
    import xarray as xr
    from scipy import stats

    if not MAPS.exists():
        return None
    z = xr.open_dataset(MAPS)
    out = {"days": z.attrs.get("days"), "area": {k[5:]: float(v) for k, v in z.attrs.items() if k.startswith("area_")}}
    sig = {}
    for var in ("u1000", "u925", "u850", "u700", "u10", "amL", "amC"):
        for combo in ("10_ALL", "22_ALL", "both_ALL", "both_DJF", "both_JJA"):
            k = f"{var}_{combo}"
            if f"{k}_mean" not in z:
                continue
            t = z[f"{k}_t"].values
            neff = z[f"{k}_neff"].values
            p = 2 * stats.t.sf(np.abs(t), np.maximum(neff - 1, 1))
            s = S.bh_fdr(np.where(np.isfinite(p), p, np.nan).ravel()).reshape(p.shape)
            sig[f"{k}_sig"] = (("lat", "lon"), s)
            out[f"frac_sig_{k}"] = float(np.nanmean(s[np.isfinite(t)]))
    xr.Dataset(sig, coords={"lat": z.lat, "lon": z.lon}).to_netcdf(od / "inc_maps_sig.nc")
    tab = pd.read_csv(MAPS_REG, parse_dates=["day"]).set_index("day").sort_index()
    regs = ("ORO", "OCX", "OCO", "LND", "HGT", "SDO", "GLB")
    split = {}
    for var in ("amL", "amC"):
        X = {r: (tab[f"{var}_10_{r}"] + tab[f"{var}_22_{r}"]).values for r in regs}
        m = tab.index.month.values
        for sname, sel in (
            ("ALL", np.ones(len(tab), bool)),
            ("DJF", np.isin(m, (12, 1, 2))),
            ("JJA", np.isin(m, (6, 7, 8))),
        ):
            N = int(sel.sum())
            g = X["GLB"][sel]
            blk = 3  # 3 sampled days = 9 calendar days
            nb = int(np.ceil(N / blk))
            st = RNG.integers(0, max(N - blk + 1, 1), (4000, nb))
            idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(4000, -1)[:, :N]
            for r in regs:
                x = X[r][sel]
                share = x.mean() / g.mean()
                bs = x[idx].mean(1) / g[idx].mean(1)
                split[f"{var}_{sname}_{r}"] = dict(
                    mean=float(x.mean()),
                    share=float(share),
                    lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)),
                    n=N,
                    area=out["area"].get(r),
                )
    out["split"] = split
    # ocean vs land 10 m wind increment (cell means, all days, both windows), area-weighted
    u10 = z["u10_both_ALL_mean"].values
    lsm = z["lsm"].values
    w = np.cos(np.deg2rad(z.lat.values))[:, None] * np.ones_like(u10)
    for nm, mk in (("ocean", lsm < 0.5), ("land", lsm >= 0.5)):
        ok = mk & np.isfinite(u10)
        out[f"u10_{nm}"] = float((u10 * w)[ok].sum() / w[ok].sum())
    (od / "increments_maps.json").write_text(json.dumps(out, indent=1, default=float))
    return out


ERA5_U = Path.home() / "era5_store" / "arco_0p25" / "u"  # ERA5 u at 14 levels, 12 UTC, sampled days
SIGN_LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700)


def wind_sign_test(od: Path):
    """Does an increment strengthen or weaken the wind it acts on? Tests the hypothesis of too much boundary-layer drag
    (too little momentum below 850 hPa). Angular momentum is signed westerly-positive, so in the easterly trades an
    increment that adds angular momentum slows the wind; too much drag predicts increments that strengthen the wind.
    Per pressure level and region, the FDR-significant increments are split by whether they share the sign of the
    background zonal wind (|u| > 0.5 m/s zonal mean, > 1 m/s per cell), weighted by
    |increment|: (a) the 5-degree zonal-mean section (inc_section.npz, both window starts), the background zonal-mean u
    log-p interpolated from 1000/925/850/700 hPa; (b) the 2.5-degree maps at 1000/925/850/700 hPa (MAPS means, inc_maps_sig.nc).
    Background: ERA5 u (ARCO 0.25 deg), every sampled 2026 day that has the low levels."""
    import glob
    import xarray as xr

    lv4 = [1000, 925, 850, 700]
    acc, n, months = None, 0, set()
    for fn in sorted(glob.glob(str(ERA5_U / "2026" / "u_*.nc"))):
        d = xr.open_dataset(fn)
        if not all(l in d.level.values for l in lv4):
            continue
        u = d["u"].sel(level=lv4).load()
        acc = u if acc is None else acc + u.values
        n += 1
        months.add(str(d.time.values)[:7])
    if acc is None:
        print("no background wind in", ERA5_U)
        return None
    ub = acc / n  # (4, lat 90..-90, lon)
    latq = ub.latitude.values
    out = {
        "background": {"source": "ERA5 u, 12 UTC", "days": n, "months": [min(months), max(months)]},
        "zonal": {},
        "grid": {},
    }
    # (a) zonal-mean section
    z = np.load(od / "inc_section.npz")
    lev = z["level"]
    lat5 = z["lat"]
    zm = ub.mean("longitude").values
    zm5 = np.array([[zm[k][(latq >= la - 2.5) & (latq < la + 2.5)].mean() for la in lat5] for k in range(4)])
    lp = np.log(np.array(lv4, float))[::-1]
    for p in SIGN_LEVELS:
        k = int(np.where(lev == p)[0][0])
        ubar = np.array([np.interp(np.log(p), lp, zm5[::-1, j]) for j in range(len(lat5))])
        row = {}
        for reg, band in (("trop", np.abs(lat5) < 30), ("extra", np.abs(lat5) >= 30)):
            st = wk = 0.0
            cells = 0
            for w in ("10", "22"):
                j = z[f"J{w}_mean"][k]
                sg = z[f"J{w}_sig"][k] & band & (np.abs(ubar) > 0.5) & np.isfinite(j)
                same = np.sign(j) == np.sign(ubar)
                st += float(np.abs(j[sg & same]).sum())
                wk += float(np.abs(j[sg & ~same]).sum())
                cells += int(sg.sum())
            row[reg] = {"strengthen": round(st / (st + wk), 3) if st + wk else None, "cells": cells}
        out["zonal"][str(p)] = row
    # (c) layer-integrated: sum over each layer and region of the daily increment (both window starts) with the sign of the
    # background wind, + = strengthens; 95 % intervals from the 10-day block bootstrap of block_boot_mean. Too little drag
    # leaves the layer-integrated wind too fast; extra mixing only moves momentum between layers.
    if U37.exists():
        U = xr.open_dataset(U37)
        lat_b = U.band_lat.values
        levs = [float(p) for p in U.level.values if 700 <= p <= 1000]
        zmb = np.array([[zm[k][(latq >= la - 2.5) & (latq < la + 2.5)].mean() for la in lat_b] for k in range(4)])
        sg = np.stack(
            [
                np.where(
                    np.abs(ub_ := np.array([np.interp(np.log(p), lp, zmb[::-1, j]) for j in range(len(lat_b))])) > 0.5,
                    np.sign(ub_),
                    0.0,
                )
                for p in levs
            ]
        )  # (level, band)
        X = (U["J10"] + U["J22"]).sel(level=levs).values * sg[None]  # (day, level, band)
        lev_a = np.array(levs)
        out["layers"] = {
            "days": int(X.shape[0]),
            "units": "1e25 kg m2 s-1 per day, signed + = strengthens the wind",
            "regions": {},
        }
        for reg, rm in (("trop", np.abs(lat_b) < 30), ("extra", (np.abs(lat_b) >= 30) & (np.abs(lat_b) <= 60))):
            rr = {}
            for name, (a, b) in (
                ("1000-950", (950, 1000)),
                ("925-875", (875, 925)),
                ("1000-875", (875, 1000)),
                ("850-700", (700, 850)),
                ("1000-700", (700, 1000)),
            ):
                lm = (lev_a >= a) & (lev_a <= b)
                m_, lo_, hi_, _p = block_boot_mean(X[:, lm][:, :, rm].sum(axis=(1, 2))[:, None])
                rr[name] = {
                    "mean": float(m_[0]),
                    "lo": float(lo_[0]),
                    "hi": float(hi_[0]),
                    "sig": bool(lo_[0] > 0 or hi_[0] < 0),
                }
            out["layers"]["regions"][reg] = rr
    # (b) 2.5-degree maps
    if MAPS.exists() and (od / "inc_maps_sig.nc").exists():
        M = xr.open_dataset(MAPS)
        G = xr.open_dataset(od / "inc_maps_sig.nc")
        lat, lon = M.lat.values, M.lon.values
        cb = ub.coarsen(latitude=10, longitude=10, boundary="trim").mean()
        cb = cb.interp(
            latitude=xr.DataArray(lat, dims="lat"), longitude=xr.DataArray(lon, dims="lon"), method="nearest"
        )
        wgt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        for p in lv4:
            inc = M[f"u{p}_both_ALL_mean"].values
            sg = G[f"u{p}_both_ALL_sig"].values.astype(bool)
            u0 = cb.sel(level=p).values
            same = np.sign(inc) == np.sign(u0)
            row = {}
            for reg, m in (("trop", np.abs(lat) < 30), ("extra", np.abs(lat) >= 30)):
                sel = sg & m[:, None] & (np.abs(u0) > 1.0) & np.isfinite(inc)
                a = np.abs(inc[sel]) * wgt[sel]
                row[reg] = {
                    "strengthen": round(float(a[same[sel]].sum() / a.sum()), 3) if a.sum() else None,
                    "cells": int(sel.sum()),
                }
            out["grid"][str(p)] = row
    (od / "inc_wind_sign.json").write_text(json.dumps(out, indent=1))
    for reg, rr in out.get("layers", {}).get("regions", {}).items():
        print("  layers", reg, {k: (round(v["mean"], 4), v["sig"]) for k, v in rr.items()})
    for p, r in out["zonal"].items():
        print(
            f"  {p:>4s} hPa  zonal mean: tropics {r['trop']['strengthen']}  extratropics {r['extra']['strengthen']}"
            + (
                f"   | maps: {out['grid'][p]['trop']['strengthen']} / {out['grid'][p]['extra']['strengthen']}"
                if p in out["grid"]
                else ""
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps-only", action="store_true")
    ap.add_argument("--wind-sign", action="store_true", help="only the strengthen / weaken test (inc_wind_sign.json)")
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    if a.wind_sign:
        return 0 if wind_sign_test(od) else 1
    if a.maps_only:
        r = maps_analysis(od)
        print(json.dumps({k: v for k, v in (r or {}).items() if k != "split"}, indent=1, default=float))
        if r:
            for k, v in r["split"].items():
                print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
        return 0
    res = {}
    maps_analysis(od)
    # ---------------- (a) latitude structure from the 5-degree bands
    J = band_jumps()
    J.to_csv(od / "inc_band_jumps.csv.gz")
    res["days"] = [str(J.index[0].date()), str(J.index[-1].date()), len(J)]
    prof = {}
    for w in ("10", "22"):
        for part in ("", "r", "m", "T"):
            cols = [f"J{w}{part}_{k:02d}" if part else f"J{w}_{k:02d}" for k in range(36)]
            X = J[cols].values
            m, lo, hi, p = block_boot_mean(X)
            sig = S.bh_fdr(p)
            prof[f"{w}{part}"] = dict(mean=m.tolist(), lo=lo.tolist(), hi=hi.tolist(), p=p.tolist(), sig=sig.tolist())
            g = X.sum(1)
            gm, glo, ghi, gp = block_boot_mean(g[:, None])
            prof[f"{w}{part}"]["global"] = dict(mean=float(gm[0]), lo=float(glo[0]), hi=float(ghi[0]), p=float(gp[0]))
    for part in ("", "r", "m"):  # both window starts together (per day)
        X = sum(J[[f"J{w}{part}_{k:02d}" if part else f"J{w}_{k:02d}" for k in range(36)]].values for w in ("10", "22"))
        m, lo, hi, p = block_boot_mean(X)
        prof[f"both{part}"] = dict(
            mean=m.tolist(), lo=lo.tolist(), hi=hi.tolist(), p=p.tolist(), sig=S.bh_fdr(p).tolist()
        )
    res["profile"] = prof
    # hemispheric and belt sums of the total increment per window
    belts = {
        "90-60S": (0, 6),
        "60-30S": (6, 12),
        "30S-0": (12, 18),
        "0-30N": (18, 24),
        "30-60N": (24, 30),
        "60-90N": (30, 36),
    }
    bt = {}
    for w in ("10", "22"):
        for nm, (k0, k1) in belts.items():
            X = J[[f"J{w}_{k:02d}" for k in range(k0, k1)]].values.sum(1)
            m, lo, hi, p = block_boot_mean(X[:, None])
            bt[f"{w}_{nm}"] = dict(mean=float(m[0]), lo=float(lo[0]), hi=float(hi[0]), p=float(p[0]))
    sigb = S.bh_fdr([v["p"] for v in bt.values()])
    for (_k, v), s_ in zip(bt.items(), sigb):
        v["fdr"] = bool(s_)
    res["belts"] = bt
    # seasonal cycle: month x band of the daily total (J10 + J22), t-test per cell, FDR over cells
    tot = pd.DataFrame({k: J[f"J10_{k:02d}"] + J[f"J22_{k:02d}"] for k in range(36)}, index=J.index)
    full = tot.groupby(tot.index.to_period("M")).size()
    tot = tot[
        tot.index.to_period("M").isin(full[full >= 20].index)
    ]  # complete months only (drops the 3-day October 2026)
    mon = tot.groupby(tot.index.to_period("M"))
    Mm = mon.mean()
    P = np.full(Mm.shape, np.nan)
    for i, (_per, g) in enumerate(mon):
        if len(g) >= 15:
            P[i] = effn_ttest(g.values)
    nodata = ~np.isfinite(Mm.values) | (np.abs(Mm.values) < 1e-12)
    P[nodata] = np.nan  # no-data cells are not tested
    SG = S.bh_fdr(P.ravel()).reshape(P.shape)
    pd.DataFrame(Mm.values, index=Mm.index.astype(str), columns=[f"{c:.1f}" for c in LATC]).to_csv(
        od / "inc_season_mean.csv"
    )
    pd.DataFrame(SG, index=Mm.index.astype(str), columns=[f"{c:.1f}" for c in LATC]).to_csv(od / "inc_season_sig.csv")
    gmon = tot.sum(1).groupby(tot.index.to_period("M")).agg(["mean", "std", "count"])
    res["global_by_month"] = {
        str(k): dict(mean=float(v["mean"]), sd=float(v["std"]), n=int(v["count"])) for k, v in gmon.iterrows()
    }
    # ---------------- (b) vertical structure from 37-level u
    if U37.exists():
        import xarray as xr

        u = xr.open_dataset(U37)
        lev = u.level.values
        sec = {}
        for w in ("J10", "J22"):
            X = u[w].values  # (day, level, band)
            m, lo, hi, p = block_boot_mean(X, block=10, n=4000)  # 10-day blocks respect the day-to-day memory
            nodata = np.all(np.abs(X) < 1e-12, axis=0)  # layers below the ground (no mass)
            p = np.where(nodata, np.nan, p)
            sig = S.bh_fdr(p.ravel()).reshape(p.shape)
            sec[w] = dict(mean=m, sig=sig, nodata=nodata)
            res.setdefault("u37_frac_sig_with_data", {})[w] = float(sig[~nodata].mean())
            layers = {
                "stratosphere (<100 hPa)": lev < 100,
                "upper troposphere (100-400 hPa)": (lev >= 100) & (lev < 400),
                "mid troposphere (400-850 hPa)": (lev >= 400) & (lev < 850),
                "lowest layers (>=850 hPa)": lev >= 850,
            }
            lay = {}
            for nm, mk in layers.items():
                s = X[:, mk, :].sum((1, 2))
                mm, lo, hi, pp = block_boot_mean(s[:, None], block=10)
                lay[nm] = dict(mean=float(mm[0]), lo=float(lo[0]), hi=float(hi[0]), p=float(pp[0]))
            col = X.sum(1)  # column per band: compare with the 2-D motion jump
            r2d = J[[f"{w}r_{k:02d}" for k in range(36)]].reindex(pd.DatetimeIndex(u.day.values).normalize()).values
            ok = np.isfinite(r2d).all(1)
            res[f"u37_{w}"] = dict(
                layers=lay,
                n_days=int(X.shape[0]),
                frac_sig=float(sig.mean()),
                global_mean=float(X.sum((1, 2)).mean()),
                corr_column_vs_2d=float(np.corrcoef(col[ok].ravel(), r2d[ok].ravel())[0, 1]) if ok.any() else None,
            )
            # the COORDE test: lower stratosphere (50-150 hPa) over the NH ranges' latitudes (25-50N) vs the troposphere below
            ks = (lev >= 50) & (lev <= 150)
            kt = lev >= 400
            bs = (LATC >= 25) & (LATC <= 50)
            for nm, mk in (("ls_25_50N", ks), ("trop_25_50N", kt)):
                s = X[:, mk, :][:, :, bs].sum((1, 2))
                mm, lo, hi, pp = block_boot_mean(s[:, None], block=10)
                res[f"u37_{w}"][nm] = dict(mean=float(mm[0]), lo=float(lo[0]), hi=float(hi[0]), p=float(pp[0]))
        np.savez_compressed(
            od / "inc_section.npz",
            level=lev,
            lat=LATC,
            **{f"{w}_{k}": v for w, d_ in sec.items() for k, v in d_.items()},
        )
    # ---------------- (c) the steady in-window residual against each range's torque (hourly 2026)
    h = pd.read_csv(H1, parse_dates=["time"]).set_index("time").sort_index()
    M = h["aam_glob"] + h["aamm_glob"]
    dM = M.diff() / HS
    tq = {}
    for c in ["mtn_AND", "mtn_HIM", "mtn_ROC", "mtn_GRL", "mtn_ANT", "mtn_ALP", "mtn_glob", "fric_glob", "gwd_glob"]:
        tq[c] = 0.5 * (h[c] + h[c].shift(1)) if c.startswith("mtn") else h[c]
    T = tq["mtn_glob"] + tq["fric_glob"] + tq["gwd_glob"]
    R = dM - T
    inside = ~h.index.hour.isin([10, 22])
    X = pd.DataFrame(
        {
            "AND": tq["mtn_AND"],
            "HIM": tq["mtn_HIM"],
            "ROC": tq["mtn_ROC"],
            "GRL": tq["mtn_GRL"],
            "ANT": tq["mtn_ANT"],
            "ALP": tq["mtn_ALP"],
            "other_mtn": tq["mtn_glob"] - sum(tq[f"mtn_{r}"] for r in ("AND", "HIM", "ROC", "GRL", "ANT", "ALP")),
            "fric": tq["fric_glob"],
            "gwd": tq["gwd_glob"],
        }
    )
    ok = inside & X.notna().all(1) & R.notna()
    Xv = np.column_stack([np.ones(ok.sum()), X[ok].values])
    yv = R[ok].values
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    day = h.index[ok].normalize()
    ud = np.unique(day)
    bs = []
    for _ in range(1000):  # bootstrap over whole days
        pick = RNG.choice(len(ud), len(ud))
        sel = np.concatenate([np.where(day == ud[i])[0] for i in pick])
        bs.append(np.linalg.lstsq(Xv[sel], yv[sel], rcond=None)[0])
    bs = np.array(bs)
    names = ["intercept", *list(X.columns)]
    reg = {}
    for i, nm in enumerate(names):
        lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
        p = float(min(1, 2 * min((bs[:, i] <= 0).mean(), (bs[:, i] >= 0).mean()) + 1e-3))
        mean_t = float(X[nm][ok].mean()) if nm != "intercept" else 1.0
        reg[nm] = dict(
            coef=float(beta[i]),
            lo=float(lo),
            hi=float(hi),
            p=p,
            mean_torque=mean_t,
            share_of_mean_residual=float(beta[i] * mean_t),
        )
    sig = S.bh_fdr([reg[n]["p"] for n in names[1:]])
    for n, s_ in zip(names[1:], sig):
        reg[n]["fdr"] = bool(s_)
    pred = Xv @ beta
    res["inwindow_regression"] = dict(
        terms=reg, mean_residual=float(yv.mean()), r2=float(1 - np.var(yv - pred) / np.var(yv)), n_hours=int(ok.sum())
    )
    # ---------------- (d) AIFS-ENS forecast AAM drift
    if FC.exists():
        import xarray as xr

        f = xr.open_dataset(FC)["fc_mean"]
        # 14-level AAM since 2026-07-18; the first two days are left out so that every verifying analysis is 14-level too
        f = f.sel(init=f.init >= np.datetime64("2026-07-20"))
        inits = pd.DatetimeIndex(f.init.values)
        dr = {}
        for reg_ in ("global", "nh", "sh"):
            rows = []
            for L in range(16):
                vals = []
                for i0 in inits:
                    iv = i0 + pd.Timedelta(days=L)
                    if iv in inits:
                        vals.append(
                            (
                                i0,
                                float(f.sel(init=i0, region=reg_, lead=L)) - float(f.sel(init=iv, region=reg_, lead=0)),
                            )
                        )
                if not vals:
                    continue
                s = pd.Series(dict(vals)).sort_index()
                wk = s.index.to_period("W")
                weeks = np.unique(wk)
                bm = []
                for _ in range(4000):
                    pick = RNG.choice(len(weeks), len(weeks))
                    bm.append(np.concatenate([s.values[wk == weeks[j]] for j in pick]).mean())
                lo, hi = np.percentile(bm, [2.5, 97.5])
                rows.append(
                    dict(lead=L, mean=float(s.mean()), lo=float(lo), hi=float(hi), n=len(s), sig=bool(lo > 0 or hi < 0))
                )
            dr[reg_] = rows
        res["fc_drift"] = dict(inits=[str(inits[0]), str(inits[-1]), len(inits)], by_region=dr)
    (od / "increments.json").write_text(json.dumps(res, indent=1, default=float))
    wind_sign_test(od)
    print(
        json.dumps({k: v for k, v in res.items() if k not in ("profile", "global_by_month")}, indent=1, default=float)[
            :6000
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
