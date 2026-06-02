#!/usr/bin/env python3
"""AAM surface-torque budget products (forecast days), AIFS-ENS control.

Two outputs:
  1. Frame-based applet: per-day 2-panel maps (friction + mountain torque-density
     ANOMALY vs the forecast-period mean, ~2.5°, ±60° zonal inset with a FIXED
     x-axis, smoothed MSLP contours) → frames + a standalone manifest the shared
     sst_anim.html player reads via ?manifest=.
  2. Budget time-series: Global | NH | SH. Each shows friction + mountain + their
     net surface torque. On the GLOBAL panel the net torque ≈ d(AAM)/dt, so we
     overlay the actual d(AAM)/dt computed from the AIFS-ENS wind field (--aam-dir);
     the residual is gravity-wave / form drag (not in the open-data feed). The
     hemispheric panels show surface torque only — a hemisphere's AAM also changes
     via cross-equatorial momentum transport, so net torque ≠ d(M_hemi)/dt there.

  friction: τ_λ = -ρ C_d |V₁₀| u₁₀ ;  density = τ_λ · a cosφ
  mountain: T_mtn = -a²∬ p_s (∂h_s/∂λ) cosφ dλ dφ

    python src/torque_map_anim.py --date 20260602 --time 00 \
        --anim-dir assets/sst/anim/torque --manifest assets/sst/anim/torque_manifest.json \
        --ts-out assets/sst/torque_timeseries.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve
import aam

OROG = Path(__file__).resolve().parent.parent / "data" / "reference" / "era5_orography.nc"
STEPS_6H = list(range(0, 361, 6))
COARSEN = 10                                    # ~2.5°
A = 6.371e6; RHO = 1.225; CD = 1.3e-3; HU = 1e18   # Hadley unit


def _daily(dd, date, time, param):
    p = dd / f"{param}6h_{date}_{time}z_cf.grib2"
    if not p.exists():
        print(f"  downloading 6-hourly control {param} …", flush=True)
        _retrieve(dict(model="aifs-ens", date=date, time=int(time), stream="enfo",
                       type="cf", levtype="sfc", param=param, step=STEPS_6H), str(p))
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[[v for v in ds.data_vars][0]].sortby("latitude")
    if float(da.longitude.min()) < 0:                       # AIFS −180…180 → 0…360
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return da.assign_coords(day=("step", hrs // 24)).groupby("day").mean()


def _aam_dmdt(aam_dir, date, time):
    """Observed d(relative AAM)/dt (Hadley) per daily step, from the cf u@13lev
    field aam.py already downloaded. Returns (valid_days, {G,NH,SH: dM/dt}) or None."""
    up = Path(aam_dir) / f"u_{date}_{time}z_cf.grib2"
    spp = Path(aam_dir) / f"sp_{date}_{time}z_cf.grib2"
    if not (up.exists() and spp.exists()):
        print(f"  (no cf AAM data in {aam_dir}; skipping observed dM/dt overlay)")
        return None
    du = xr.open_dataset(up, engine="cfgrib", backend_kwargs={"indexpath": ""})
    ds = xr.open_dataset(spp, engine="cfgrib", backend_kwargs={"indexpath": ""})
    u = du["u"].sortby("isobaricInhPa"); p_pa = u.isobaricInhPa.values * 100.0
    lat = u.latitude.values
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
    spv = ds[[v for v in ds.data_vars][0]]
    hrs = (u.step / np.timedelta64(1, "h")).values.astype(int)
    M = {"G": [], "NH": [], "SH": []}
    for k in range(u.shape[0]):
        g, n, s = aam.aam_of(u.isel(step=k).values, p_pa, spv.isel(step=k).values, lat, dlon, dlat)
        M["G"].append(g); M["NH"].append(n); M["SH"].append(s)
    sec = hrs.astype(float) * 3600.0
    dmdt = {k: np.gradient(np.array(v), sec) / HU for k, v in M.items()}   # Hadley
    return hrs // 24, dmdt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/torque")
    ap.add_argument("--aam-dir", default="data/aam")
    ap.add_argument("--anim-dir", default="assets/sst/anim/torque")
    ap.add_argument("--manifest", default="assets/sst/anim/torque_manifest.json")
    ap.add_argument("--ts-out", default="assets/sst/torque_timeseries.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    dd = Path(args.data_dir); dd.mkdir(parents=True, exist_ok=True)

    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    sp = _daily(dd, args.date, args.time, "sp").reindex(latitude=lat, longitude=lon, method="nearest")
    u = _daily(dd, args.date, args.time, "10u").reindex(latitude=lat, longitude=lon, method="nearest")
    v = _daily(dd, args.date, args.time, "10v").reindex(latitude=lat, longitude=lon, method="nearest")
    try:                                                        # MSLP for context contours
        msl = _daily(dd, args.date, args.time, "msl").reindex(latitude=lat, longitude=lon, method="nearest")
    except Exception as e:
        print(f"  (msl unavailable: {e}; skipping MSLP contours)"); msl = None
    cosphi = np.cos(np.deg2rad(lat))[:, None]
    dhdlam = np.gradient(o.values, np.deg2rad(lon), axis=1)
    dlam = np.deg2rad(abs(lon[1] - lon[0])); dphi = np.deg2rad(abs(lat[1] - lat[0]))

    fric = -RHO * CD * np.hypot(u.values, v.values) * u.values * (A * cosphi)   # (day,lat,lon) dens
    mtn = -sp.values * dhdlam[None, :, :]
    days = [int(d) for d in sp.day.values if d <= 14]
    def DA(a): return xr.DataArray(a, dims=("day", "latitude", "longitude"),
                                   coords={"day": sp.day.values, "latitude": lat, "longitude": lon}).sel(day=days)
    fricD, mtnD = DA(fric), DA(mtn)

    # --- torque totals (Hadley): integrate density × dA over Global / each hemi ---
    dA = (A**2) * cosphi * dlam * dphi                          # (lat,1) area weight per cell
    def total(dens, mask):                                      # dens (day,lat,lon)
        return (dens * dA[None])[:, mask, :].sum((1, 2)) / HU   # per day
    masks = {"G": np.ones_like(lat, bool), "NH": lat > 0, "SH": lat < 0}
    rows = {}
    for nm, mask in masks.items():
        rows[nm] = dict(fric=total(fricD.values, mask), mtn=total(mtnD.values, mask))
        rows[nm]["sum"] = rows[nm]["fric"] + rows[nm]["mtn"]
    obs = _aam_dmdt(args.aam_dir, args.date, args.time)         # observed d(AAM)/dt overlay

    # --- map frames (anomaly vs period mean, coarsened) ---
    fa = (fricD - fricD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    ma = (mtnD - mtnD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    clon = fa.longitude.values; clat = fa.latitude.values
    fl = float(np.nanpercentile(np.abs(fa.values), 99)); ml = float(np.nanpercentile(np.abs(ma.values), 99))
    mm = (clat >= -60) & (clat <= 60)
    cosw = np.cos(np.deg2rad(clat[mm]))                         # cosφ area weight (budget impact)
    # fixed zonal-mean inset x-axis (consistent across frames), per term
    zf = float(np.nanpercentile(np.abs(np.nanmean(fa.values, axis=2)[:, mm] * cosw), 98)) or fl
    zm = float(np.nanpercentile(np.abs(np.nanmean(ma.values, axis=2)[:, mm] * cosw), 98)) or ml
    if msl is not None:                                         # smoothed MSLP (hPa) for contours
        import scipy.ndimage as ndi
        mslS = np.stack([ndi.gaussian_filter(s, sigma=6, mode="wrap") for s in (msl.sel(day=days).values / 100.0)])
        mlevs = np.arange(940, 1052, 8)

    def inset(zi, prof, zlim):
        zi.plot(prof, clat[mm], color="#444", lw=1.3)
        zi.axvline(0, color="0.6", lw=0.7); zi.set_ylim(-60, 60)
        zi.set_yticks([-60, -30, 0, 30, 60]); zi.tick_params(labelsize=7)
        zi.set_xlim(-zlim, zlim)
        e = int(np.floor(np.log10(zlim))) if zlim > 0 else 0; s = 10.0 ** e
        zi.set_xticks([-zlim, 0, zlim])
        zi.set_xticklabels([f"{-zlim / s:.0f}", "0", f"{zlim / s:.0f}"], fontsize=6.5)
        zi.set_title(f"zonal mean·cosφ\n(×10$^{{{e}}}$)", fontsize=7.5); zi.grid(True, alpha=0.25)

    def hl_centers(F, mode, size=64, n=9, latcap=72):
        """Synoptic high/low pressure centres: local extrema of the smoothed MSLP,
        thinned so picks aren't clustered. -> list of (lat, lon) value pairs."""
        import scipy.ndimage as ndi
        flt = (ndi.maximum_filter if mode == "H" else ndi.minimum_filter)(F, size=size, mode="wrap")
        ys, xs = np.where(F == flt)
        order = np.argsort(F[ys, xs])[::-1] if mode == "H" else np.argsort(F[ys, xs])
        picks = []
        for i in order:
            y, x = ys[i], xs[i]
            if abs(lat[y]) > latcap:
                continue
            if all(abs(lat[y] - py) > 12 or abs(((lon[x] - px + 180) % 360) - 180) > 18 for py, px, _ in picks):
                picks.append((lat[y], lon[x], F[y, x]))
            if len(picks) >= n:
                break
        return picks

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, d in enumerate(days):
        hl = {"H": hl_centers(mslS[k], "H"), "L": hl_centers(mslS[k], "L")} if msl is not None else None
        fig = plt.figure(figsize=(11, 9.4))
        gs = GridSpec(2, 2, width_ratios=[7, 1], wspace=0.03, hspace=0.09,
                      left=0.015, right=0.985, top=0.935, bottom=0.05)
        for row, (arr, lim, zlim, ttl) in enumerate(
                [(fa.sel(day=d).values, fl, zf, "Friction-torque density anomaly  (−ρC$_d$|V|u·a cosφ)"),
                 (ma.sel(day=d).values, ml, zm, "Mountain-torque density anomaly  (−p$_s$ ∂h/∂λ)")]):
            ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=180))
            ax.pcolormesh(clon, clat, arr, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
            if msl is not None:
                cyc, cl = add_cyclic_point(mslS[k], coord=lon)
                ax.contour(cl, lat, cyc, levels=mlevs, colors="0.2", linewidths=0.35,
                           alpha=0.55, transform=ccrs.PlateCarree())
                for sym, col in (("H", "#b2182b"), ("L", "#2166ac")):
                    for plat, plon, pval in hl[sym]:
                        ax.text(plon, plat, sym, color=col, fontsize=11, fontweight="bold",
                                ha="center", va="center", transform=ccrs.PlateCarree(), clip_on=True,
                                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
                        ax.text(plon, plat - 3.5, f"{pval:.0f}", color=col, fontsize=5.5,
                                ha="center", va="top", transform=ccrs.PlateCarree(), clip_on=True,
                                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
            ax.coastlines(resolution="110m", lw=0.4, color="0.35"); ax.set_global()
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            inset(fig.add_subplot(gs[row, 1]), np.nanmean(arr, axis=1)[mm] * cosw, zlim)
        valid = init + pd.Timedelta(days=d)
        fig.suptitle(f"AIFS-ENS · AAM surface-torque anomalies — {valid:%Y-%m-%d} (forecast day {d})",
                     fontsize=12.5, fontweight="bold", y=0.99)
        if msl is not None:
            fig.text(0.5, 0.013, "grey = smoothed MSLP contours · H / L = pressure centres (hPa)",
                     ha="center", va="bottom", fontsize=8.5, color="0.4")
        fp = anim / f"F{k:02d}.webp"
        fig.savefig(fp, dpi=104, bbox_inches="tight"); plt.close(fig)
        entries.append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"),
                        "label": f"day {d} · {valid:%b %d}"})
    mani = {"regions": {"torque": {"label": "Surface torque (friction + mountain)", "frames": entries}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries)} frames + {args.manifest}")

    # --- budget time-series (Global | NH | SH) ---
    valid = [init + pd.Timedelta(days=d) for d in days]
    titles = {"G": "Global budget", "NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}
    obs_by_day = {}
    if obs is not None:
        odays, dmdt = obs
        obs_by_day = {h: {int(dd_): dmdt[h][i] for i, dd_ in enumerate(odays)} for h in ("G", "NH", "SH")}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for ax, hemi in zip(axes, ("G", "NH", "SH")):
        ax.plot(valid, rows[hemi]["fric"], color="#2166ac", lw=1.7, label="friction torque")
        ax.plot(valid, rows[hemi]["mtn"], color="#b2182b", lw=1.7, label="mountain torque")
        ax.plot(valid, rows[hemi]["sum"], color="k", lw=2.4, label="friction + mountain")
        if obs_by_day:
            odm = np.array([obs_by_day[hemi].get(d, np.nan) for d in days])      # observed dM/dt
            if hemi == "G":
                ax.plot(valid, odm - rows["G"]["sum"], color="#1b7837", lw=1.7, label="implied GWD (residual)")
            ax.plot(valid, odm, color="0.35", lw=2.2, ls="--", label="observed d(AAM)/dt")
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.25)
        ax.set_title(titles[hemi], fontsize=11, fontweight="bold"); ax.legend(fontsize=7.8, loc="best")
        for lb in ax.get_xticklabels(): lb.set_rotation(45); lb.set_ha("right"); lb.set_fontsize(7.5)
    axes[0].set_ylabel("torque (Hadley = 10¹⁸ N m)")
    fig.suptitle(f"AAM torque budget — AIFS-ENS init {init:%Y-%m-%d %HZ}", fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.02,
             "Only the GLOBAL budget closes from surface torques: friction + mountain + implied gravity-wave/form drag (residual) = d(AAM)/dt.  "
             "Each hemisphere's gap between net surface torque (black) and observed d(AAM)/dt (dashed) is cross-equatorial momentum transport (+ GWD) — an internal flux, not a torque.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    Path(args.ts_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.ts_out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.ts_out} (Global net {rows['G']['sum'][0]:+.0f}→{rows['G']['sum'][-1]:+.0f} Hadley; "
          f"obs dM/dt {'overlaid' if obs is not None else 'n/a'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
