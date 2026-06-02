#!/usr/bin/env python3
"""Animated mountain-torque-density map (forecast days), AIFS-ENS control.

Mountain (pressure) torque density  −p_s ∂h_s/∂λ  — where surface pressure pushes
on topography. Uses DAILY-MEAN control surface pressure (6-hourly → daily mean),
coarsened to ~1°, scaled so only the strong ranges show, with a zonal-mean inset.
Frames over forecast days → one looping animated WebP (like the other site anims).

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
STEPS_6H = list(range(0, 361, 6))               # 6-hourly → daily means
COARSEN = 10                                    # ~2.5° blocks
A = 6.371e6


def daily_mean_sp(date, time, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / f"sp6h_{date}_{time}z_cf.grib2"
    if not p.exists():
        print("  downloading 6-hourly control sp …", flush=True)
        _retrieve(dict(model="aifs-ens", date=date, time=int(time), stream="enfo",
                       type="cf", levtype="sfc", param="sp", step=STEPS_6H), str(p))
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    sp = ds[[v for v in ds.data_vars][0]].sortby("latitude")
    hrs = (sp.step / np.timedelta64(1, "h")).values.astype(int)
    day = (hrs // 24).astype(int)
    sp = sp.assign_coords(day=("step", day))
    return sp.groupby("day").mean()                              # (day, lat, lon)


def torque_density(spv, hs, lon):
    """Coarsened mountain-torque density −p_s ∂h/∂λ (per day) as xr.DataArray."""
    dhdlam = np.gradient(hs, np.deg2rad(lon), axis=-1)
    dens = -spv.values * dhdlam[None, :, :]                      # (day, lat, lon)
    da = xr.DataArray(dens, dims=("day", "latitude", "longitude"),
                      coords={"day": spv.day.values, "latitude": spv.latitude.values,
                              "longitude": lon})
    return da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/torque")
    ap.add_argument("--out", default="assets/sst/torque_anim.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")

    o = xr.open_dataarray(OROG)
    sp = daily_mean_sp(args.date, args.time, Path(args.data_dir))
    sp = sp.reindex(latitude=o.latitude.values, longitude=o.longitude.values, method="nearest")
    dens = torque_density(sp, o.values, o.longitude.values)
    lon = dens.longitude.values; lat = dens.latitude.values
    days = [d for d in dens.day.values if d <= 14]               # drop partial last day
    lim = float(np.nanpercentile(np.abs(dens.sel(day=days).values), 99.0))  # scale: only big show

    frames = []
    for d in days:
        f = dens.sel(day=d).values
        fig = plt.figure(figsize=(12, 4.8))
        gs = GridSpec(1, 2, width_ratios=[6, 1], wspace=0.04)
        ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
        ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-lim, vmax=lim,
                      transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
        ax.coastlines(resolution="110m", lw=0.4, color="0.35"); ax.set_global()
        ax.set_title(f"Mountain-torque density  −p$_s$ ∂h/∂λ  ·  {(init+pd.Timedelta(days=int(d))):%Y-%m-%d} "
                     f"(day {int(d)})", fontsize=11, fontweight="bold")
        zi = fig.add_subplot(gs[1])                               # zonal-mean inset (±60°)
        m = (lat >= -60) & (lat <= 60)
        zi.plot(np.nanmean(f, axis=1)[m], lat[m], color="#444", lw=1.4)
        zi.axvline(0, color="0.6", lw=0.7); zi.set_ylim(-60, 60)
        zi.set_xticks([]); zi.set_yticks([-60, -30, 0, 30, 60])
        zi.tick_params(labelsize=7); zi.set_title("zonal mean", fontsize=8)
        zi.grid(True, alpha=0.25)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig); buf.seek(0); frames.append(Image.open(buf).convert("RGB"))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"saved {out} ({len(frames)} frames, scale ±{lim:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
