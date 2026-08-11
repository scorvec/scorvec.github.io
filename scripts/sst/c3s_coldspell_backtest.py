#!/usr/bin/env python3
"""Backtest of the cold-spell product: 24 hindcast winters, leave-one-out.

The daily hindcasts (1993-2016, August inits) that build the drift
climatologies are also 24 archived real-time forecasts of winters ERA5 knows.
For each winter Y and model:

  * climatology: per-lead-day mean/sigma pooled from the OTHER 23 years
    (leave-one-out, exact via per-year windowed sum/sum-of-squares) — year Y's
    own members never touch the reference they are scored against;
  * forecast: fraction of year-Y members with >= SPELL_DAYS consecutive days
    below (own-model normal) - 1 sigma, Dec 1 - Feb 28 — identical to the
    operational product;
  * truth: the identical spell test on ERA5 vs its 1991-2020 daily normal;
  * skill: Brier skill score vs the leave-one-out climatological base rate,
    pooled reliability, and the Permian absolute-threshold hit table at
    Midland (the 2010/11 Texas freeze winter is in sample; Uri itself
    post-dates the hindcast era).

Hindcast ensembles are smaller than real time (SEAS5 25 vs 51), so measured
skill slightly understates the operational product.

    python c3s_coldspell_backtest.py [--prep-only]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c3s_coldspell import (DATA, ASSETS, WB2_T2M, SPELL_DAYS,
                           era5_daily_climo, spell_prob)

BT = DATA / "backtest"
YEARS = list(range(1993, 2017))
MODELS = [("ecmwf", "51", "ECMWF SEAS5"),
          ("eccc", "4", "ECCC GEM5-NEMO"),
          ("eccc", "5", "ECCC CanESM5")]
HALFWIN = 7
ND = 90                                  # Dec 1 .. Feb 28, leap day trimmed
TX = (25.8, 36.6, -106.7, -93.5)         # Texas box (lat0, lat1, lon0, lon1)


def land_mask(tlat, tlon):
    """NE 110m land on the scoring grid, cached."""
    cache = BT / "land_mask.npz"
    if cache.exists():
        return np.load(cache)["land"]
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    geo = prep(unary_union(list(shpreader.Reader(shpreader.natural_earth(
        resolution="110m", category="physical", name="land")).geometries())))
    l180 = np.where(tlon > 180, tlon - 360, tlon)
    land = np.array([[geo.contains(Point(lo_, la_)) for lo_ in l180]
                     for la_ in tlat])
    BT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, land=land)
    return land


def tx_weights(tlat, tlon, land):
    """cos-weighted Texas land-box weights on the scoring grid."""
    l180 = np.where(tlon > 180, tlon - 360, tlon)
    box = ((tlat >= TX[0]) & (tlat <= TX[1]))[:, None] \
        & ((l180 >= TX[2]) & (l180 <= TX[3]))[None, :]
    w = np.cos(np.deg2rad(tlat))[:, None] * (box & land)
    return w / w.sum()


def longest_runs(series, thresh=0.0):
    """Longest consecutive sub-`thresh` run per row of (n, days)."""
    streak = np.zeros(series.shape[0], np.int16)
    longest = np.zeros_like(streak)
    for d in range(series.shape[1]):
        streak = np.where(series[:, d] < thresh, streak + 1, 0)
        longest = np.maximum(longest, streak)
    return longest


def _grid(clim_mean):
    glat = clim_mean.latitude.values
    glon = clim_mean.longitude.values
    la = (glat >= 16) & (glat <= 72)
    lo = (glon >= 190) & (glon <= 310)
    return glat[la], glon[lo], la, lo


def prep_year(centre, system, y, tlat, tlon):
    """Member daily means for hindcast winter y/y+1 on the 1.5 deg scoring
    grid, cached. Returns (mem, ND, lat, lon) float32 or None."""
    cache = BT / f"bt_{centre}_{system}_{y}.npz"
    if cache.exists():
        return np.load(cache)["daily"]
    src = DATA / f"dailyhc_{centre}_{system}_{y}08.grib"
    if not src.exists():
        return None
    try:
        ds = xr.open_dataset(src, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception as e:                                # noqa: BLE001
        print(f"  {src.name}: unreadable ({str(e)[:50]}) — skipped", flush=True)
        return None
    da = ds[[v for v in ds.data_vars][0]] - 273.15
    vt = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds["valid_time"].values)))
    norm = vt.normalize()
    udays = pd.DatetimeIndex(sorted(set(norm)))
    udays = udays[(udays >= pd.Timestamp(f"{y}-12-01"))
                  & (udays < pd.Timestamp(f"{y + 1}-03-01"))][:ND]
    if len(udays) < ND:
        print(f"  {src.name}: only {len(udays)} days — skipped", flush=True)
        return None
    lonm = da.longitude.values
    da = da.assign_coords(longitude=np.where(lonm < 0, lonm + 360, lonm)) \
           .sortby("longitude").sortby("latitude")
    arr = da.transpose("number", "step", "latitude", "longitude")
    days = [arr.isel(step=np.where(norm == d)[0]).mean("step") for d in udays]
    dx = (xr.concat(days, dim="day")
          .transpose("number", "day", "latitude", "longitude")
          .interp(latitude=tlat, longitude=tlon))
    daily = dx.values.astype(np.float32)
    ds.close()
    BT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, daily=daily)
    print(f"  cached {centre}/{system} {y}: {daily.shape[0]} members", flush=True)
    return daily


def loo_stats(all_daily):
    """Per-year windowed (sum, sumsq, count) per lead day -> exact
    leave-one-out mean/sigma. all_daily: {year: (mem, ND, lat, lon)}."""
    years = sorted(all_daily)
    shp = all_daily[years[0]].shape[2:]
    S = {}; Q = {}; N = {}
    for y in years:
        a = all_daily[y].astype(np.float64)
        s = np.empty((ND,) + shp); q = np.empty_like(s); n = np.empty(ND)
        for d in range(ND):
            lo, hi = max(0, d - HALFWIN), min(ND, d + HALFWIN + 1)
            win = a[:, lo:hi]
            s[d] = win.sum(axis=(0, 1))
            q[d] = (win ** 2).sum(axis=(0, 1))
            n[d] = win.shape[0] * win.shape[1]
        S[y], Q[y], N[y] = s, q, n
    St = sum(S.values()); Qt = sum(Q.values()); Nt = sum(N.values())
    out = {}
    for y in years:
        n = (Nt - N[y])[:, None, None]
        m = (St - S[y]) / n
        v = (Qt - Q[y]) / n - m ** 2
        out[y] = (m.astype(np.float32), np.sqrt(np.maximum(v, 0)).astype(np.float32))
    return out


def era5_truth(clim_mean, clim_std, tlat, tlon, la, lo, txw):
    """Per-winter observed spell occurrence (24, lat, lon) uint8 + Texas
    land-box-mean daily series (24, ND) in deg C, cached."""
    cache = BT / "era5_truth_v2.npz"
    if cache.exists():
        z = np.load(cache)
        return z["occ"], z["tx"]
    cm = clim_mean.values[:, la][:, :, lo]
    cs = clim_std.values[:, la][:, :, lo]
    occ, txs = [], []
    for y in YEARS:
        a = xr.open_dataset(WB2_T2M / f"t2m_{y}.nc")["t2m"]
        b = xr.open_dataset(WB2_T2M / f"t2m_{y+1}.nc")["t2m"]
        da = xr.concat([a.sel(time=slice(f"{y}-12-01", None)),
                        b.sel(time=slice(None, f"{y+1}-02-28"))], dim="time")
        da = da.transpose("time", "latitude", "longitude") \
               .isel(latitude=la, longitude=lo)
        v = da.values
        if v.max() > 150:
            v = v - 273.15
        doy = da.time.dt.dayofyear.values
        z = ((v + 273.15) - cm[doy - 1]) / np.maximum(cs[doy - 1], 0.5) \
            if clim_mean.values.max() > 150 else \
            (v - cm[doy - 1]) / np.maximum(cs[doy - 1], 0.5)
        occ.append(spell_prob(z[None], -1.0, SPELL_DAYS)[0])
        txs.append((v[:ND] * txw[None]).sum(axis=(1, 2)))
        print(f"  ERA5 winter {y}/{y+1} ✓", flush=True)
    BT.mkdir(parents=True, exist_ok=True)
    occ = np.array(occ, np.uint8)
    txs = np.array(txs, np.float32)
    np.savez_compressed(cache, occ=occ, tx=txs)
    return occ, txs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()

    print("ERA5 daily climatology …", flush=True)
    clim_mean, clim_std = era5_daily_climo()
    tlat, tlon, la, lo = _grid(clim_mean)
    land = land_mask(tlat, tlon)
    txw = tx_weights(tlat, tlon, land)

    print("ERA5 truth winters …", flush=True)
    occ, era5_tx = era5_truth(clim_mean, clim_std, tlat, tlon, la, lo, txw)

    fprob = {}                        # label -> (24, lat, lon) forecast P
    mid_abs = {}                      # label -> per-year Midland member stats
    finite_all = None                 # model NaN (fetch-box edge) => unscoreable
    for centre, system, label in MODELS:
        print(f"{label}: preparing hindcast winters …", flush=True)
        all_daily = {}
        for y in YEARS:
            d = prep_year(centre, system, y, tlat, tlon)
            if d is not None:
                all_daily[y] = d
        if len(all_daily) < 20:
            print(f"  {label}: only {len(all_daily)} usable years — skipped")
            continue
        if args.prep_only:
            continue
        loo = loo_stats(all_daily)
        for y in all_daily:
            f = np.isfinite(all_daily[y]).all(axis=(0, 1))
            finite_all = f if finite_all is None else (finite_all & f)
        pj, mj = [], []
        cmC = clim_mean.values[:, la][:, :, lo]
        cmC = cmC - 273.15 if cmC.max() > 150 else cmC
        for y in sorted(all_daily):
            m, s = loo[y]
            z = (all_daily[y] - m[None]) / np.maximum(s[None], 0.5)
            pj.append(spell_prob(z, -1.0, SPELL_DAYS).mean(axis=0))
            # Texas land-box-mean member series in absolute deg C
            # (LOO drift removal + ERA5 daily-normal reference)
            doy = pd.date_range(f"{y}-12-01", periods=ND).dayofyear.values
            # subset to the box BEFORE weighting: the domain edge is NaN in
            # model space and NaN * 0-weight is still NaN
            jj = np.where(txw.sum(axis=1) > 0)[0]
            kk = np.where(txw.sum(axis=0) > 0)[0]
            wsub = txw[np.ix_(jj, kk)]
            absf = (all_daily[y][:, :, jj][:, :, :, kk]
                    - m[None][:, :, jj][:, :, :, kk]
                    + cmC[doy - 1][None][:, :, jj][:, :, :, kk])
            mj.append((absf * wsub).sum(axis=(2, 3)) / wsub.sum())
        fprob[label] = np.array(pj)
        mid_abs[label] = mj
        print(f"  {label}: {len(pj)} winters scored", flush=True)

    if args.prep_only or not fprob:
        return

    # multi-model mean (years common to all = all 24 here)
    mm = np.mean([fprob[k] for k in fprob], axis=0)
    fprob["Multi-model"] = mm

    # --- Brier skill vs leave-one-out base rate ------------------------------
    occf = occ.astype(np.float32)
    n = len(YEARS)
    base_loo = (occf.sum(axis=0)[None] - occf) / (n - 1)
    bs_ref = ((base_loo - occf) ** 2).sum(axis=0)
    nev = occf.sum(axis=0)
    mask = (nev >= 2) & (nev <= n - 2)          # skill undefined off the tails
    if finite_all is not None:
        mask &= finite_all            # fetch-box edge: model data is NaN there
    mask &= land                      # decision-relevant sample: land only
    bss = {}
    for k, p in fprob.items():
        bs = ((p - occf) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = 1.0 - bs / bs_ref
        bss[k] = np.where(mask, b, np.nan)

    # --- pooled reliability (multi-model) ------------------------------------
    edges = np.array([0, .02, .05, .10, .20, .30, .45, .65, 1.0001])
    pk, ok_, ck = [], [], []
    pf = mm[:, mask]; of = occf[:, mask]
    for i in range(len(edges) - 1):
        m2 = (pf >= edges[i]) & (pf < edges[i + 1])
        ck.append(int(m2.sum()))
        pk.append(float(pf[m2].mean()) if m2.any() else np.nan)
        ok_.append(float(of[m2].mean()) if m2.any() else np.nan)

    # --- Midland Uri-class hit table -----------------------------------------
    # --- Texas statewide freeze spells: box-mean < 0 C for >=2 / >=3 days ---
    tx_p = {t: np.mean([[(longest_runs(a) >= t).mean() for a in mid_abs[k]]
                        for k in mid_abs], axis=0) for t in (2, 3)}
    obs_run = longest_runs(era5_tx)
    tx_o = {t: (obs_run >= t).astype(float) for t in (2, 3)}
    tx_bss = {}
    for t in (2, 3):
        ref = ((tx_o[t].mean() - tx_o[t]) ** 2).mean()
        tx_bss[t] = (1 - ((tx_p[t] - tx_o[t]) ** 2).mean() / ref
                     if ref > 0 else np.nan)
    uri_p, uri_o = tx_p[3], tx_o[3]
    uri_base = uri_o.mean()

    # --- figure --------------------------------------------------------------
    order = ["Multi-model"] + [k for k in fprob if k != "Multi-model"]
    lon_plot = np.where(tlon > 180, tlon - 360, tlon)
    proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=45)
    fig = plt.figure(figsize=(14.6, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    levels = [-0.4, -0.3, -0.2, -0.1, -0.02, 0.02, 0.1, 0.2, 0.3, 0.4]
    map_axes = []
    for i, k in enumerate(order):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        map_axes.append(ax)
        cf = ax.contourf(lon_plot, tlat, bss[k], levels=levels, cmap="RdBu",
                         extend="both", transform=ccrs.PlateCarree())
        ax.set_extent([-128, -60, 22, 62], ccrs.PlateCarree())
        ax.coastlines(lw=0.5, color="0.25")
        ax.add_feature(cfeature.BORDERS, lw=0.35, edgecolor="0.4",
                       facecolor="none")
        ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="0.55",
                       facecolor="none")
        med = np.nanmedian(bss[k])
        ax.set_title(f"{k} · med {med:+.2f}", fontsize=10, fontweight="bold",
                     loc="left")
    cb = fig.colorbar(cf, ax=map_axes, orientation="horizontal", pad=0.01,
                      fraction=0.03, aspect=60)
    cb.set_label("Brier skill score vs leave-one-out climatology · "
                 "≥5 d < normal −1σ · blue = beats climatology", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    axr = fig.add_subplot(gs[1, 0:2])
    axr.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    axr.axhline(np.nanmean(of), color="0.75", lw=0.8, ls=":")
    axr.plot(pk, ok_, marker="o", ms=5, color="#1565c0", lw=1.5)
    for x, yv, c in zip(pk, ok_, ck):
        if np.isfinite(x):
            axr.annotate(f"{c//1000}k" if c >= 1000 else str(c), (x, yv),
                         textcoords="offset points", xytext=(6, -4),
                         fontsize=7, color="0.4")
    axr.set_xlabel("forecast probability (multi-model)", fontsize=9)
    axr.set_ylabel("observed frequency", fontsize=9)
    axr.set_title("Reliability · land gridpoints × 24 winters",
                  fontsize=10.5, fontweight="bold", loc="left")
    axr.tick_params(labelsize=8)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1)

    axm = fig.add_subplot(gs[1, 2:4])
    xs = np.arange(n)
    cols = ["#c62828" if o else "#90a4ae" for o in uri_o]
    axm.bar(xs, 100 * uri_p, color=cols, width=0.75)
    axm.set_xticks(xs)
    axm.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS], fontsize=7)
    axm.axhline(100 * uri_base, color="0.4", lw=0.9, ls=":")
    axm.set_ylabel("forecast P(TX box-mean < 0 °C ≥3 d), %", fontsize=9)
    axm.set_title(f"Texas statewide freeze spell (≥3 d) · red = happened "
                  f"({int(uri_o.sum())}/{n}, base {100*uri_base:.0f}%) · "
                  f"BSS {tx_bss[3]:+.2f}", fontsize=10.5, fontweight="bold",
                  loc="left")
    axm.set_ylim(0, 100 * max(uri_p.max(), uri_base) * 1.45)
    axm.text(0.02, 0.97, "box-mean of TX land gridpoints · ≥2-day tier: base "
             f"{100*tx_o[2].mean():.0f}%, BSS {tx_bss[2]:+.2f} · Feb-2011 "
             "deepest in sample; Uri (2021) out of sample",
             transform=axm.transAxes, fontsize=7.6, color="0.35", va="top")
    axm.tick_params(labelsize=8)

    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.95))
    fig.suptitle("Cold-spell backtest — 24 winters (1993/94–2016/17) · "
                 "Aug issues → Dec–Feb (lead 4–6 mo) · leave-one-out",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.004,
             "Each winter scored with a climatology from the other 23 years · truth: "
             "identical spell test on ERA5 vs its 1991–2020 daily normal · BSS masked "
             "where <2 or >22 event winters · land gridpoints only · hindcast "
             "ensembles smaller than real time, so skill is understated",
             fontsize=7.6, ha="center", color="0.35")
    out = ASSETS / "c3s_coldspell_backtest.webp"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    for k in order:
        b = bss[k]
        frac = float((b[np.isfinite(b)] > 0).mean())
        print(f"{k}: median BSS {np.nanmedian(b):+.3f} · "
              f"area-frac BSS>0 {frac:.2f}")
    for t in (2, 3):
        o, pr = tx_o[t], tx_p[t]
        print(f"TX box ≥{t}d: BSS {tx_bss[t]:+.2f} · base {o.mean():.2f} · "
              f"event-winter mean P {100*pr[o == 1].mean():.0f}% vs "
              f"non-event {100*pr[o == 0].mean():.0f}%")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
