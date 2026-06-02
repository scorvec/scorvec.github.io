#!/usr/bin/env python3
"""Animated AAM torque-density maps (forecast days): friction + mountain.

Two stacked panels, each the DAILY-MEAN control torque density as an ANOMALY from
the forecast-period mean (the static topography-locked mean is removed, so the
moving synoptic signal is what animates), coarsened to ~2.5°, with a ±60° zonal
inset. Frames over forecast days → one looping animated WebP.

  friction: -ρ·C_d·|V₁₀|·u₁₀ · a cosφ   (surface wind stress; trades/jets)
  mountain: -p_s · ∂h_s/∂λ              (surface pressure on topography)

    python src/torque_map_anim.py --date 20260602 --time 00 --out assets/sst/torque_anim.webp
"""
from __future__ import annotations

import argparse
import io
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve

OROG = Path(__file__).resolve().parent.parent / "data" / "reference" / "era5_orography.nc"
STEPS_6H = list(range(0, 361, 6))
COARSEN = 10                                    # ~2.5°
A = 6.371e6; RHO = 1.225; CD = 1.3e-3


def _daily(path, data_dir, date, time, param):
    p = data_dir / f"{param}6h_{date}_{time}z_cf.grib2"
    if not p.exists():
        print(f"  downloading 6-hourly control {param} …", flush=True)
        _retrieve(dict(model="aifs-ens", date=date, time=int(time), stream="enfo",
                       type="cf", levtype="sfc", param=param, step=STEPS_6H), str(p))
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[[v for v in ds.data_vars][0]].sortby("latitude")
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return da.assign_coords(day=("step", hrs // 24)).groupby("day").mean()


def _coarsen(da):
    return da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/torque")
    ap.add_argument("--out", default="assets/sst/torque_anim.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    dd = Path(args.data_dir); dd.mkdir(parents=True, exist_ok=True)

    o = xr.open_dataarray(OROG); lat0 = o.latitude.values; lon0 = o.longitude.values
    sp = _daily(None, dd, args.date, args.time, "sp").reindex(latitude=lat0, longitude=lon0, method="nearest")
    u = _daily(None, dd, args.date, args.time, "10u").reindex(latitude=lat0, longitude=lon0, method="nearest")
    v = _daily(None, dd, args.date, args.time, "10v").reindex(latitude=lat0, longitude=lon0, method="nearest")
    cosphi = np.cos(np.deg2rad(lat0))[:, None]
    dhdlam = np.gradient(o.values, np.deg2rad(lon0), axis=1)

    spd = np.hypot(u.values, v.values)
    fric = -RHO * CD * spd * u.values * (A * cosphi)            # (day,lat,lon)  friction torque dens
    mtn = -sp.values * dhdlam[None, :, :]                       # mountain torque density
    def da(arr): return xr.DataArray(arr, dims=("day", "latitude", "longitude"),
                                     coords={"day": sp.day.values, "latitude": lat0, "longitude": lon0})
    fric = _coarsen(da(fric)); mtn = _coarsen(da(mtn))
    days = [d for d in fric.day.values if d <= 14]
    fric = fric.sel(day=days); mtn = mtn.sel(day=days)
    fric = fric - fric.mean("day"); mtn = mtn - mtn.mean("day")  # anomaly vs period mean
    lon = fric.longitude.values; lat = fric.latitude.values
    fl = float(np.nanpercentile(np.abs(fric.values), 99)); ml = float(np.nanpercentile(np.abs(mtn.values), 99))
    mlat = (lat >= -60) & (lat <= 60)

    frames = []
    for d in days:
        fig = plt.figure(figsize=(11, 8))
        gs = GridSpec(2, 2, width_ratios=[6, 1], wspace=0.04, hspace=0.18)
        for row, (fld, lim, ttl) in enumerate(
                [(fric.sel(day=d).values, fl, "Friction-torque density anomaly  (−ρC$_d$|V|u · a cosφ)"),
                 (mtn.sel(day=d).values, ml, "Mountain-torque density anomaly  (−p$_s$ ∂h/∂λ)")]):
            ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=180))
            ax.pcolormesh(lon, lat, fld, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
            ax.coastlines(resolution="110m", lw=0.4, color="0.35"); ax.set_global()
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            zi = fig.add_subplot(gs[row, 1])
            zi.plot(np.nanmean(fld, axis=1)[mlat], lat[mlat], color="#444", lw=1.3)
            zi.axvline(0, color="0.6", lw=0.7); zi.set_ylim(-60, 60); zi.set_xticks([])
            zi.set_yticks([-60, -30, 0, 30, 60]); zi.tick_params(labelsize=7)
            zi.set_title("zonal mean", fontsize=8); zi.grid(True, alpha=0.25)
        fig.suptitle(f"AAM surface-torque anomalies — {(init+pd.Timedelta(days=int(d))):%Y-%m-%d} (day {int(d)})",
                     fontsize=12, fontweight="bold")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=104, bbox_inches="tight")
        plt.close(fig); buf.seek(0); frames.append(Image.open(buf).convert("RGB"))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"saved {out} ({len(frames)} frames; fric±{fl:.1e}, mtn±{ml:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
