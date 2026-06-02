#!/usr/bin/env python3
"""AAM surface-torque budget products (forecast days), AIFS-ENS control.

Two outputs:
  1. Frame-based applet: per-day 2-panel maps (friction + mountain torque-density
     ANOMALY vs the forecast-period mean, ~2.5°, ±60° zonal inset) → frames +
     a standalone manifest the shared sst_anim.html player reads via ?manifest=.
  2. Budget time-series: hemispheric totals of each term + their sum (= dM/dt)
     vs forecast day, NH and SH — what's driving the AAM anomaly each hemisphere.

  friction: τ_λ = -ρ C_d |V₁₀| u₁₀ ;  T_fric = a³∬ (force·armarm)…  (per-area dens)
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
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/torque")
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
    cosphi = np.cos(np.deg2rad(lat))[:, None]
    dhdlam = np.gradient(o.values, np.deg2rad(lon), axis=1)
    dlam = np.deg2rad(abs(lon[1] - lon[0])); dphi = np.deg2rad(abs(lat[1] - lat[0]))

    fric = -RHO * CD * np.hypot(u.values, v.values) * u.values * (A * cosphi)   # (day,lat,lon) dens
    mtn = -sp.values * dhdlam[None, :, :]
    days = [int(d) for d in sp.day.values if d <= 14]
    def DA(a): return xr.DataArray(a, dims=("day", "latitude", "longitude"),
                                   coords={"day": sp.day.values, "latitude": lat, "longitude": lon}).sel(day=days)
    fricD, mtnD = DA(fric), DA(mtn)

    # --- hemispheric torque totals (Hadley): integrate density × dA over each hemi ---
    dA = (A**2) * cosphi * dlam * dphi                          # (lat,1) area weight per cell
    def total(dens, mask):                                      # dens (day,lat,lon)
        return (dens * dA[None])[:, mask, :].sum((1, 2)) / HU   # per day
    nh = lat > 0; sh = lat < 0
    rows = {}
    for nm, mask in (("NH", nh), ("SH", sh)):
        rows[nm] = dict(fric=total(fricD.values, mask), mtn=total(mtnD.values, mask))
        rows[nm]["sum"] = rows[nm]["fric"] + rows[nm]["mtn"]

    # --- map frames (anomaly vs period mean, coarsened) ---
    fa = (fricD - fricD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    ma = (mtnD - mtnD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    clon = fa.longitude.values; clat = fa.latitude.values
    fl = float(np.nanpercentile(np.abs(fa.values), 99)); ml = float(np.nanpercentile(np.abs(ma.values), 99))
    mm = (clat >= -60) & (clat <= 60)
    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, d in enumerate(days):
        fig = plt.figure(figsize=(11, 8))
        gs = GridSpec(2, 2, width_ratios=[6, 1], wspace=0.04, hspace=0.18)
        for row, (arr, lim, ttl) in enumerate(
                [(fa.sel(day=d).values, fl, "Friction-torque density anomaly  (−ρC$_d$|V|u·a cosφ)"),
                 (ma.sel(day=d).values, ml, "Mountain-torque density anomaly  (−p$_s$ ∂h/∂λ)")]):
            ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=180))
            ax.pcolormesh(clon, clat, arr, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
            ax.coastlines(resolution="110m", lw=0.4, color="0.35"); ax.set_global()
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            zi = fig.add_subplot(gs[row, 1])
            zi.plot(np.nanmean(arr, axis=1)[mm], clat[mm], color="#444", lw=1.3)
            zi.axvline(0, color="0.6", lw=0.7); zi.set_ylim(-60, 60); zi.set_xticks([])
            zi.set_yticks([-60, -30, 0, 30, 60]); zi.tick_params(labelsize=7)
            zi.set_title("zonal mean", fontsize=8); zi.grid(True, alpha=0.25)
        valid = init + pd.Timedelta(days=d)
        fig.suptitle(f"AAM surface-torque anomalies — {valid:%Y-%m-%d} (day {d})", fontsize=12, fontweight="bold")
        fp = anim / f"F{k:02d}.webp"
        fig.savefig(fp, dpi=104, bbox_inches="tight"); plt.close(fig)
        entries.append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"),
                        "label": f"day {d} · {valid:%b %d}"})
    mani = {"regions": {"torque": {"label": "Surface torque (friction + mountain)", "frames": entries}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries)} frames + {args.manifest}")

    # --- budget time-series (NH | SH) ---
    valid = [init + pd.Timedelta(days=d) for d in days]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, hemi in zip(axes, ("NH", "SH")):
        ax.plot(valid, rows[hemi]["fric"], color="#2166ac", lw=1.8, label="friction")
        ax.plot(valid, rows[hemi]["mtn"], color="#b2182b", lw=1.8, label="mountain")
        ax.plot(valid, rows[hemi]["sum"], color="k", lw=2.6, label="sum (= dM/dt)")
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.25)
        ax.set_title(f"{hemi} torque budget", fontsize=11, fontweight="bold")
        for lb in ax.get_xticklabels(): lb.set_rotation(45); lb.set_ha("right"); lb.set_fontsize(7.5)
    axes[0].set_ylabel("torque (Hadley = 10¹⁸ N m)"); axes[0].legend(fontsize=8.5, loc="best")
    fig.suptitle(f"Hemispheric AAM torque budget — AIFS-ENS init {init:%Y-%m-%d %HZ}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    Path(args.ts_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.ts_out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.ts_out} (NH sum {rows['NH']['sum'][0]:+.0f}→{rows['NH']['sum'][-1]:+.0f}, "
          f"SH sum {rows['SH']['sum'][0]:+.0f}→{rows['SH']['sum'][-1]:+.0f} Hadley)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
