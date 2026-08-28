#!/usr/bin/env python3
"""Hemispheric AAM phase diagnostics: latitude-time heatmap + GWO-style
phase orbits, with an auto-generated plain-language phase headline
("NH: rising AAM — subtropical westerlies increasing").

Top    — band-integrated relative-AAM anomaly by latitude × time: ~90 days
         observed (ERA5 via the local store, 3-day steps, 13 tropospheric
         levels to match the density climatology) + the 15-day AIFS-ENS
         ensemble-mean forecast, "today" line at the seam.
Bottom — NH & SH phase orbits: x = hemispheric AAM anomaly (14-level, from
         aam_history/aam_clim), y = centred tendency; observed trail + the
         latest forecast from aam_forecast_archive.nc (no refetch).

    python src/aam_phase.py --date 20260718 --time 00 --out ../../assets/sst/aam_phase.webp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import era5_store
from aam import (A, G, SCALE, CLIM_PATH, HIST_PATH, ARCHIVE_PATH,
                 _vert_weights, eval_clim)

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
DENS_CLIM = REF / "aam_density_clim.nc"          # (doy, lev13, lat1.5°) per-band
LATBAND_HIST = REF / "aam_latband_history.nc"    # incrementally cached observed rows
LEVELS13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
OBS_DAYS = 60                                    # observed window (3-day steps)
DSCALE = 1e23                                    # heatmap units


def _latband_day(t, latc):
    """Band-integrated AAM (kg m² s⁻¹) on the 1.5° clim latitude grid for one
    12Z ERA5 timestamp (13 levels, matching the density climatology)."""
    u = era5_store.get_u(t, LEVELS13)
    sp = era5_store.get_sp(t)
    lat = u.latitude.values
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
    dp = _vert_weights(u.level.values * 100.0, sp.values)
    dens = ((A ** 3 / G) * (u.values * dp).sum(axis=0).sum(axis=1)
            * np.cos(np.deg2rad(lat)) ** 2 * dlon * dlat)      # per 0.25° row
    # aggregate 0.25° rows into the 1.5° clim bands
    out = np.zeros(len(latc))
    half = 0.75
    for i, lc in enumerate(latc):
        sel = (lat >= lc - half) & (lat < lc + half)
        out[i] = dens[sel].sum()
    return out


def observed_latbands(end_day, latc):
    """(times, bands) for the trailing OBS_DAYS window, cached incrementally."""
    want = pd.date_range(end_day - pd.Timedelta(days=OBS_DAYS), end_day, freq="3D")
    have = None
    if LATBAND_HIST.exists():
        have = xr.open_dataarray(LATBAND_HIST).load()
    rows, times = [], []
    n_new = 0
    for t in want:
        if have is not None and t in pd.to_datetime(have.time.values):
            rows.append(have.sel(time=t).values); times.append(t)
            continue
        try:
            r = _latband_day(t.strftime("%Y-%m-%dT12:00"), latc)
        except Exception:                                      # noqa: BLE001
            continue
        if not np.isfinite(r).all() or not r.any():
            continue
        rows.append(r); times.append(t); n_new += 1
    if not rows:
        return None, None
    da = xr.DataArray(np.array(rows), dims=("time", "latitude"),
                      coords={"time": times, "latitude": latc}, name="aam_band")
    if n_new:
        merged = da if have is None else xr.concat(
            [have.drop_sel(time=[t for t in have.time.values
                                 if pd.Timestamp(t) in set(times)], errors="ignore"), da],
            dim="time").sortby("time").drop_duplicates("time")
        tmp = LATBAND_HIST.with_suffix(".tmp.nc")
        merged.to_netcdf(tmp); os.replace(tmp, LATBAND_HIST)
        print(f"  latband history: +{n_new} day(s)")
    return pd.DatetimeIndex(times), np.array(rows)


def forecast_latbands(date, time, latc):
    """AIFS-ENS ens-mean band AAM per forecast day (13 levels), from the
    cached u/sp files (cache-hits when the AAM products ran this cycle)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
    import store as ecmwf
    from aam import DAILY_STEPS
    cyc = ecmwf.Cycle(date, time)
    steps = tuple(DAILY_STEPS)
    acc, spacc, n = None, None, 0
    for typ in ("cf", "pf"):
        up_rest = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl",
                                               ecmwf.LEVELS_AAM_REST, steps,
                                               ecmwf.AAM_PF_MEMBERS))
        up_rmm = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl",
                                              ecmwf.LEVELS_RMM, steps))
        sp_path = ecmwf.sfc_path(cyc, "aifs-ens", typ, "sp")
        kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
        ur = xr.open_dataset(up_rest, **kw)["u"]
        um_ = xr.open_dataset(up_rmm, **kw)["u"]
        if "number" in ur.dims and "number" in um_.dims:   # rest file may be a member subset
            um_ = um_.sel(number=ur.number)
        u = xr.concat([ur, um_], dim="isobaricInhPa").sortby("isobaricInhPa")
        u = u.sel(isobaricInhPa=LEVELS13)
        sp = xr.open_dataset(sp_path, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"shortName": "sp"},
                                             "indexpath": ""}, chunks={"number": 1})
        spv = sp[[v for v in sp.data_vars][0]]
        um = (u.mean("number") if "number" in u.dims else u).transpose(
            "step", "isobaricInhPa", "latitude", "longitude").values
        spm = (spv.mean("number") if "number" in spv.dims else spv).transpose(
            "step", "latitude", "longitude").values
        w = 50 if typ == "pf" else 1
        acc = um * w if acc is None else acc + um * w
        spacc = spm * w if spacc is None else spacc + spm * w
        n += w
        lat = u.latitude.values
        steps_h = (u.step / np.timedelta64(1, "h")).values.astype(int)
    um, spm = acc / n, spacc / n
    dlon = np.deg2rad(0.25); dlat = np.deg2rad(0.25)
    cos2 = np.cos(np.deg2rad(lat)) ** 2
    out = np.zeros((um.shape[0], len(latc)))
    for k in range(um.shape[0]):
        dp = _vert_weights(np.asarray(LEVELS13, float) * 100.0, spm[k])
        dens = (A ** 3 / G) * (um[k] * dp).sum(axis=0).sum(axis=1) * cos2 * dlon * dlat
        for i, lc in enumerate(latc):
            sel = (lat >= lc - 0.75) & (lat < lc + 0.75)
            out[k, i] = dens[sel].sum()
    return steps_h, out


def band_clim(latc, doys):
    """Climatological band AAM (doy, lat) from the density climatology."""
    cl = xr.open_dataarray(DENS_CLIM)                       # (doy, lev, lat) per band
    tot = cl.sum("level")                                   # already per 1.5° band
    return np.stack([tot.sel(dayofyear=int(d)).values for d in doys])


def _phase_words(hemi, m_anom, dm, band_change, bands):
    updown = "rising" if dm > 0 else "falling"
    hilo = "above-normal" if m_anom > 0 else "below-normal"
    i = int(np.argmax(np.abs(band_change)))
    sign = "increasing" if band_change[i] > 0 else "weakening"
    return f"{hemi}: {updown} {hilo} AAM — {bands[i]} westerlies {sign}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out", default="../../assets/sst/aam_phase.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    print("== AAM phase diagnostics ==", flush=True)

    cl = xr.open_dataarray(DENS_CLIM)
    latc = cl.latitude.values
    band = np.abs(latc) <= 78
    latc_show = latc[band]

    # observed lat-time anomaly
    times, obs = observed_latbands(init - pd.Timedelta(days=5), latc)
    doys = np.array([t.dayofyear for t in times])
    obs_anom = (obs - band_clim(latc, doys))[:, band] / DSCALE

    # forecast lat-time anomaly (ens mean, daily)
    steps_h, fc = forecast_latbands(args.date, args.time, latc)
    fvalid = [init + pd.Timedelta(hours=int(h)) for h in steps_h]
    fdoys = np.array([v.dayofyear for v in fvalid])
    fc_anom = (fc - band_clim(latc, fdoys))[:, band] / DSCALE

    # hemispheric phase series (14-level history + clim + forecast archive)
    hist = xr.open_dataset(HIST_PATH)
    clim = xr.open_dataset(CLIM_PATH)
    ht = pd.to_datetime(hist.time.values)
    sel = ht >= init - pd.Timedelta(days=75)
    arch = xr.open_dataset(ARCHIVE_PATH)
    a_inits = pd.to_datetime(arch.init.values)
    a_init = a_inits[-1]
    phase = {}
    for key in ("nh", "sh"):
        coef = clim["coeffs"].sel(region=key).values
        obs_m = hist[key].values[sel] - eval_clim(coef, np.array(
            [t.dayofyear for t in ht[sel]]))
        # ERA5 lags ~5-6 days — bridge to today with the archived AIFS
        # lead-0 (analysis) values so the trail meets the forecast seamlessly
        bmask = (a_inits > ht[sel][-1]) & (a_inits <= a_init)
        b_t = a_inits[bmask]
        b_m = (arch["fc_mean"].sel(region=key, lead=0).values[bmask]
               - eval_clim(coef, np.array([v.dayofyear for v in b_t])))
        t_o = ht[sel].append(pd.DatetimeIndex(b_t))
        obs_m = np.concatenate([obs_m, b_m])
        fcm = arch["fc_mean"].sel(region=key, init=a_init).values
        fl = [a_init + pd.Timedelta(days=int(l)) for l in arch.lead.values]
        fc_m = fcm - eval_clim(coef, np.array([v.dayofyear for v in fl]))
        phase[key] = (t_o, obs_m, fl, fc_m)

    # plain-language headline per hemisphere (band change day 12-15 vs day 0-2)
    bands_def = [("deep-tropical", 0, 15), ("subtropical", 15, 35), ("mid-latitude", 35, 60)]
    head = []
    for hemi, key, sgn in (("NH", "nh", 1), ("SH", "sh", -1)):
        _, obs_m, _, fc_m = phase[key]
        chg = []
        for _, b0, b1 in bands_def:
            bs = (latc_show * sgn >= b0) & (latc_show * sgn < b1)
            chg.append(fc_anom[-4:, bs].mean() - fc_anom[:3, bs].mean())
        dm = fc_m[-1] - fc_m[0]                  # net change over the 15 days
        head.append(_phase_words(hemi, fc_m[-1], dm, np.array(chg),
                                 [b[0] for b in bands_def]))

    # ── figure ──
    fig = plt.figure(figsize=(13.6, 10.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1], hspace=0.32,
                  wspace=0.18, left=0.06, right=0.985, top=0.90, bottom=0.07)
    ax = fig.add_subplot(gs[0, :])
    tall = list(times) + fvalid
    z = np.vstack([obs_anom, fc_anom])
    lim = max(float(np.nanpercentile(np.abs(z), 99)), 1.0)
    pm = ax.pcolormesh(tall, latc_show, z.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="nearest")
    # contours over the forecast segment: key anomaly values for readability
    cs = ax.contour(fvalid, latc_show, fc_anom.T, levels=[-10, -5, 5, 10],
                    colors="k", linewidths=[0.9, 0.5, 0.5, 0.9],
                    negative_linestyles="dashed")
    ax.clabel(cs, levels=[-10, -5, 5, 10], fmt="%d", fontsize=6, inline=True)
    ax.axvline(init, color="k", lw=1.4, ls="--")
    ax.text(init, latc_show[-1] + 1.5, "forecast →", fontsize=8, va="bottom", ha="left")
    ax.set_yticks(range(-75, 76, 15))
    for yl in range(-75, 76, 15):
        ax.axhline(yl, color="0.4", lw=0.7 if yl == 0 else 0.3,
                   ls="-" if yl == 0 else ":", alpha=0.7 if yl == 0 else 0.45)
    ax.set_ylabel("latitude")
    ax.set_title(f"Relative-AAM anomaly by latitude — observed (ERA5, 3-day) + "
                 f"AIFS-ENS mean forecast · init {init:%Y-%m-%d %HZ}",
                 fontsize=11, fontweight="bold")
    cb = fig.colorbar(pm, ax=ax, pad=0.012)
    cb.set_label(f"band AAM anomaly (10²³ kg m² s⁻¹ per 1.5° lat)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for lab in ax.get_xticklabels():
        lab.set_fontsize(8)

    for j, (hemi, key) in enumerate((("Northern Hemisphere", "nh"),
                                     ("Southern Hemisphere", "sh"))):
        axp = fig.add_subplot(gs[1, j])
        t_o, m_o, t_f, m_f = phase[key]
        # ONE combined obs+forecast series (the first forecast point IS the
        # last bridge analysis — drop the duplicate), smoothed and
        # differentiated once on the true time axis: continuous by construction
        t_all = list(t_o) + list(t_f[1:])
        m_all = np.concatenate([m_o, m_f[1:]])
        ms = pd.Series(m_all).rolling(3, center=True, min_periods=1).mean().values
        tdays = np.array([(x - t_all[0]).total_seconds() / 86400.0 for x in t_all])
        dm = np.gradient(ms, tdays)
        n_o = len(m_o)
        axp.axhline(0, color="0.75", lw=0.8); axp.axvline(0, color="0.75", lw=0.8)
        axp.plot(ms[:n_o], dm[:n_o], color="0.45", lw=1.1, alpha=0.9, zorder=3)
        axp.scatter(ms[:n_o], dm[:n_o], c=np.arange(n_o), cmap="Greys", s=14,
                    zorder=4, vmin=-n_o * 0.4)
        axp.plot(ms[n_o - 1:], dm[n_o - 1:], color="#b71c1c", lw=2.4, zorder=5)
        axp.plot(ms[-1], dm[-1], marker="*", ms=13, color="#b71c1c",
                 mec="k", mew=0.5, zorder=6)
        axp.plot(ms[n_o - 1], dm[n_o - 1], marker="o", ms=8, color="k", zorder=6)
        for xf, yf, s in ((0.97, 0.97, "high & rising"), (0.03, 0.97, "low & rising"),
                          (0.97, 0.03, "high & falling"), (0.03, 0.03, "low & falling")):
            axp.text(xf, yf, s, transform=axp.transAxes, fontsize=7.5, color="0.55",
                     ha="right" if xf > 0.5 else "left",
                     va="top" if yf > 0.5 else "bottom", style="italic")
        axp.set_title(hemi, fontsize=10.5, fontweight="bold")
        axp.set_xlabel("AAM anomaly (10²⁵ kg m² s⁻¹)", fontsize=8.5)
        if j == 0:
            axp.set_ylabel("tendency (10²⁵ per day)", fontsize=8.5)
        axp.tick_params(labelsize=7.5)
        axp.grid(True, alpha=0.2)

    fig.suptitle("Hemispheric AAM phase — " + "   ·   ".join(head),
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.008,
             "orbit: grey = observed trail (75 days, darker = recent, ● = today) · red = AIFS-ENS mean forecast (★ = day 15) · "
             "headline bands from the forecast day-12–15 minus day-0–2 band-AAM change",
             ha="center", fontsize=7.5, color="0.4")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=118, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print("  " + " | ".join(head))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
