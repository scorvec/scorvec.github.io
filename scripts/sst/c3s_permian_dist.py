#!/usr/bin/env python3
"""Permian winter distributions — this winter's members vs model climatology.

The threshold products (freeze-off card) reduce to single probabilities; this
shows the distributions they are sampled from, for the Permian box
(30-34N, 104-100W, land-weighted box mean):

  * daily-mean T2m, Dec 1 - Feb 28: ERA5 observed (30 winters), each model's
    own 24 hindcast winters (leave-one-out reconstruction, member-pooled) and
    the 2026/27 forecast members — density + cold-tail exceedance with the
    operational thresholds marked;
  * monthly snowfall (Dec + Jan totals): each snow-publishing system's own
    hindcast distribution vs its forecast members, as exceedance curves.

Models are equal-weighted (per-model curves averaged), never raw-pooled — a
51-member ensemble must not out-vote a 20-member one.

    python c3s_permian_dist.py --issue 202608
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3s_coldspell_backtest as bt
from c3s_coldspell import (DATA, ASSETS, WB2_T2M, era5_daily_climo,
                           model_daily_climo, T20F, DAILY_MODELS)
from c3s_snow_winter import DATA as SNOW_DATA, open_snow
from c3s_t2m_winter import month_samples
import c3s_nino34 as c3s

PBOX = (30.0, 34.0, 256.0, 260.0)          # Permian: 30-34N, 104-100W (degE)
ND = bt.ND


def _box_weights(tlat, tlon, land):
    box = ((tlat >= PBOX[0]) & (tlat <= PBOX[1]))[:, None] \
        & ((tlon >= PBOX[2]) & (tlon <= PBOX[3]))[None, :]
    w = np.cos(np.deg2rad(tlat))[:, None] * (box & land)
    return w / w.sum()


def _boxmean(a, w):
    """(…, lat, lon) -> (…) — subset the box first: edge NaNs x 0-weight."""
    jj = np.where(w.sum(axis=1) > 0)[0]
    kk = np.where(w.sum(axis=0) > 0)[0]
    ws = w[np.ix_(jj, kk)]
    sub = a[..., jj, :][..., :, kk]
    return (sub * ws).sum(axis=(-2, -1)) / ws.sum()


def era5_box_daily(w, la, lo, y0=1991, y1=2020):
    cache = bt.BT / "era5_permian_daily.npz"
    if cache.exists():
        return np.load(cache)["v"]
    out = []
    for y in range(y0, y1):
        a = xr.open_dataset(WB2_T2M / f"t2m_{y}.nc")["t2m"]
        b = xr.open_dataset(WB2_T2M / f"t2m_{y+1}.nc")["t2m"]
        da = xr.concat([a.sel(time=slice(f"{y}-12-01", None)),
                        b.sel(time=slice(None, f"{y+1}-02-28"))], dim="time")
        v = da.transpose("time", "latitude", "longitude").values[:ND]
        v = v[:, la][:, :, lo]
        if v.max() > 150:
            v = v - 273.15
        out.append(_boxmean(v, w))
    v = np.concatenate(out)
    np.savez_compressed(cache, v=v)
    return v


def hindcast_box_daily(centre, system, tlat, tlon, cmC, w):
    """All hindcast winters' members, absolute deg C box means, pooled."""
    ad = {y: bt.prep_year(centre, system, y, tlat, tlon) for y in bt.YEARS}
    ad = {y: d for y, d in ad.items() if d is not None}
    if len(ad) < 20:
        return None
    loo = bt.loo_stats(ad)
    out = []
    for y in ad:
        doy = pd.date_range(f"{y}-12-01", periods=ND).dayofyear.values
        absf = ad[y] - loo[y][0][None] + cmC[doy - 1][None]
        out.append(_boxmean(absf, w).ravel())
    return np.concatenate(out)


def forecast_box_daily(centre, system, issue, tlat, tlon, cmC, w):
    """This winter's members, absolute deg C box means (same reconstruction
    as the operational build: own daily-hindcast drift + ERA5 normal)."""
    dest = DATA / f"daily_{centre}_{system}_{issue}.grib"
    if not dest.exists():
        return None
    ds = xr.open_dataset(dest, engine="cfgrib", backend_kwargs={"indexpath": ""})
    if "number" in ds.dims and ds.sizes["number"] < 10:
        ds.close()
        return None
    da = ds[[v for v in ds.data_vars][0]] - 273.15
    vt = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds["valid_time"].values)))
    norm = vt.normalize()
    udays = pd.DatetimeIndex(sorted(set(norm)))
    dec1 = pd.Timestamp(f"{issue[:4]}-12-01")
    udays = udays[udays >= dec1]
    lonm = da.longitude.values
    da = da.assign_coords(longitude=np.where(lonm < 0, lonm + 360, lonm)) \
           .sortby("longitude").sortby("latitude")
    arr = da.transpose("number", "step", "latitude", "longitude")
    days = [arr.isel(step=np.where(norm == d)[0]).mean("step") for d in udays]
    dx = (xr.concat(days, dim="day")
          .transpose("number", "day", "latitude", "longitude")
          .interp(latitude=tlat, longitude=tlon))
    daily_v = dx.values
    ds.close()
    mclim_d, _mstd, nd = model_daily_climo(centre, system, issue, tlat, tlon)
    if mclim_d is None:
        return None
    nkeep = min(nd, daily_v.shape[1], ND)
    doy = udays[:nkeep].dayofyear.values
    absf = daily_v[:, :nkeep] - mclim_d[None, :nkeep] + cmC[doy - 1][None]
    return _boxmean(absf, w).ravel()


def snow_box(issue):
    """Per snow system: (hindcast monthly totals, forecast monthly totals) for
    Dec + Jan, Permian box mean, mm w.e./month."""
    out = {}
    yr = int(issue[:4])
    for centre, system, label, _c in c3s.MODELS:
        hcp = SNOW_DATA / f"hc_{centre}_{system}_{issue[4:]}.grib"
        fcp = SNOW_DATA / f"fc_{centre}_{system}_{issue}.grib"
        if not (hcp.exists() and fcp.exists()):
            continue
        hc, hvt = open_snow(hcp)
        fc, fvt = open_snow(fcp)
        lat = hc.latitude.values
        lon = hc.longitude.values
        lon = np.where(lon < 0, lon + 360, lon)
        jj = np.where((lat >= PBOX[0]) & (lat <= PBOX[1]))[0]
        kk = np.where((lon >= PBOX[2]) & (lon <= PBOX[3]))[0]
        wlat = np.cos(np.deg2rad(lat[jj]))[:, None] * np.ones((len(jj), len(kk)))
        wlat = wlat / wlat.sum()
        h, f = [], []
        for m, ndays in ((12, 31), (1, 31)):
            hs = month_samples(hc, hvt, m)
            fs = month_samples(fc, fvt, m, year=yr + 1 if m == 1 else yr)
            if hs.size:
                h.append((hs[:, jj][:, :, kk] * wlat).sum(axis=(1, 2)) * ndays)
            if fs.size:
                f.append((fs[:, jj][:, :, kk] * wlat).sum(axis=(1, 2)) * ndays)
        if h and f:
            out[label] = (np.concatenate(h), np.concatenate(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=pd.Timestamp.utcnow().strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue

    print("grids + ERA5 …", flush=True)
    clim_mean, _ = era5_daily_climo()
    tlat, tlon, la, lo = bt._grid(clim_mean)
    land = bt.land_mask(tlat, tlon)
    w = _box_weights(tlat, tlon, land)
    cmC = clim_mean.values[:, la][:, :, lo]
    if cmC.max() > 150:
        cmC = cmC - 273.15
    era5 = era5_box_daily(w, la, lo)

    hind, fcst = {}, {}
    for centre, system, label in DAILY_MODELS:
        f = forecast_box_daily(centre, system, issue, tlat, tlon, cmC, w)
        if f is None:
            continue
        h = hindcast_box_daily(centre, system, tlat, tlon, cmC, w)
        if h is None:
            continue
        hind[label], fcst[label] = h, f
        print(f"  {label}: hindcast {h.size} · forecast {f.size} samples",
              flush=True)
    snow = snow_box(issue)
    print(f"  snow systems: {sorted(snow)}", flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9),
                             constrained_layout=True)
    axd, axt, axs = axes

    bins = np.arange(-22, 32.1, 1.0)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    from scipy.ndimage import gaussian_filter1d

    def dens(v):
        h, _ = np.histogram(v, bins=bins, density=True)
        return gaussian_filter1d(h, 1.2)

    axd.fill_between(ctr, dens(era5), color="0.75", alpha=0.55,
                     label="ERA5 observed (30 winters)")
    hpool = np.mean([dens(hind[k]) for k in hind], axis=0)
    fpool = np.mean([dens(fcst[k]) for k in fcst], axis=0)
    axd.plot(ctr, hpool, color="#1565c0", lw=2.2,
             label=f"model climatology ({len(hind)} models, own hindcasts)")
    axd.plot(ctr, fpool, color="#c62828", lw=2.2,
             label="forecast 2026/27 members")
    for k in fcst:
        axd.axvline(np.mean(fcst[k]), color="#c62828", lw=0.7, alpha=0.45)
    axd.axvline(np.mean(era5), color="0.4", lw=0.9, ls=":")
    axd.set_xlim(-15, 30)
    axd.set_xlabel("daily-mean T2m, Permian box (°C)", fontsize=9)
    axd.set_ylabel("density", fontsize=9)
    axd.set_title("Daily temperature distribution · Dec–Feb", fontsize=10.5,
                  fontweight="bold", loc="left")
    axd.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
    axd.tick_params(labelsize=8)

    xs = np.arange(-18, 10.01, 0.25)

    def exceed(v):
        v = np.sort(v)
        return np.searchsorted(v, xs, side="right") / v.size

    axt.semilogy(xs, np.maximum(exceed(era5), 1e-5), color="0.45", lw=2.0,
                 label="ERA5")
    axt.semilogy(xs, np.maximum(np.mean([exceed(hind[k]) for k in hind],
                                        axis=0), 1e-5),
                 color="#1565c0", lw=2.2, label="model climatology")
    axt.semilogy(xs, np.maximum(np.mean([exceed(fcst[k]) for k in fcst],
                                        axis=0), 1e-5),
                 color="#c62828", lw=2.2, label="forecast 2026/27")
    for x0, lab in ((0.0, "0 °C"), (-5.0, "−5 °C"), (T20F, "20 °F")):
        axt.axvline(x0, color="0.7", lw=0.8, ls="--")
        axt.text(x0, 1.3, lab, fontsize=7.5, ha="center", color="0.35")
    axt.set_ylim(1e-4, 2)
    axt.set_xlim(-18, 10)
    axt.set_xlabel("threshold x (°C)", fontsize=9)
    axt.set_ylabel("P(daily mean < x)  · per day", fontsize=9)
    axt.set_title("Cold tail · per-day exceedance", fontsize=10.5,
                  fontweight="bold", loc="left")
    axt.legend(fontsize=7.6, loc="lower right", framealpha=0.9)
    axt.tick_params(labelsize=8)
    axt.grid(alpha=0.25, lw=0.4)

    sx = np.concatenate([[0.0], np.geomspace(0.2, 80, 120)])

    def snow_exceed(v):
        v = np.sort(v)
        return 1.0 - np.searchsorted(v, sx, side="right") / v.size

    if snow:
        hcur = np.mean([snow_exceed(snow[k][0]) for k in snow], axis=0)
        fcur = np.mean([snow_exceed(snow[k][1]) for k in snow], axis=0)
        axs.plot(sx, 100 * hcur, color="#1565c0", lw=2.2,
                 label=f"model climatology ({len(snow)} systems)")
        axs.plot(sx, 100 * fcur, color="#c62828", lw=2.2,
                 label="forecast Dec+Jan members")
        axs.set_xscale("symlog", linthresh=1.0)
        axs.set_xlim(0, 80)
        top = 105 * max(hcur[sx >= 0.2].max(), fcur[sx >= 0.2].max())
        axs.set_ylim(0, min(90, top))
        axs.set_xlabel("monthly snowfall ≥ x (mm w.e.)", fontsize=9)
        axs.set_ylabel("P(exceed), %  · per month", fontsize=9)
        axs.set_title("Monthly snowfall exceedance · Dec + Jan", fontsize=10.5,
                      fontweight="bold", loc="left")
        axs.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
        axs.tick_params(labelsize=8)
        axs.grid(alpha=0.25, lw=0.4)

    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.93))
    fig.suptitle(f"Permian winter distributions — forecast members vs model "
                 f"climatology · issue {issue[:4]}-{issue[4:]}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.008,
             "Permian box 30–34°N 104–100°W, land-weighted · temps: daily members, own-model "
             "drift + ERA5 normal; climatology = same reconstruction of the 24 hindcast "
             "winters · snow: monthly totals, per-system curves averaged",
             fontsize=7.4, ha="center", color="0.35")
    out = ASSETS / "c3s_permian_dist.webp"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    for k in fcst:
        print(f"  {k}: fc mean {fcst[k].mean():+.1f} °C · "
              f"climo {hind[k].mean():+.1f} °C · "
              f"fc P(<0) {(fcst[k] < 0).mean():.3f} vs climo "
              f"{(hind[k] < 0).mean():.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
