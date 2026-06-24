#!/usr/bin/env python3
"""AIFS-ENS 200 hPa wind and upper-level divergence (45°S–45°N, global, Pacific-centred).

200 hPa wind speed (shaded) with barbs, plus smoothed positive divergence (the upper-level
outflow over deep convection) as filled areas. Control member: frame 0 is the 0-h analysis
(also written as the static thumbnail), frames 1–15 are the forecast days, stacked into a
rolling animation. Divergence is computed on the sphere from u, v and Gaussian-smoothed so
broad outflow areas read clearly rather than grid-scale noise.

    python src/wind200_div.py --date 20260624 --time 12 \
        --anim-dir assets/sst/anim/wind200 --manifest assets/sst/anim/wind200_manifest.json \
        --out assets/sst/wind200.webp
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf

A = 6.371e6                                          # Earth radius (m)
SPD_LEV = np.arange(20, 80, 5)                       # wind-speed shading (m/s)
DIV_LEV = [0.6, 1, 1.5, 2, 3, 4]                     # divergence areas (10^-5 s^-1)
DIV_COLORS = ["#fff0a8", "#fbcf66", "#f6a13a", "#ec6a26", "#cf2c1a", "#9c0f2e"]
LATBAND = 45


def divergence(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Smoothed horizontal divergence (10^-5 s^-1) on the sphere from (lat, lon) u, v."""
    latr = np.deg2rad(lat); lonr = np.deg2rad(lon); cosp = np.cos(latr)[:, None]
    d = (np.gradient(u, lonr, axis=1) + np.gradient(v * cosp, latr, axis=0)) / (A * cosp)
    return gaussian_filter(d, sigma=8, mode="wrap") * 1e5


def plot(u, v, lat, lon, title: str, out: Path):
    div = divergence(u, v, lat, lon)
    m = (lat >= -LATBAND) & (lat <= LATBAND)
    lat = lat[m]; u = u[m]; v = v[m]; div = div[m]; spd = np.hypot(u, v)
    fig = plt.figure(figsize=(14, 4.9))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([-180, 180, -LATBAND, LATBAND], crs=ccrs.PlateCarree())
    PC = ccrs.PlateCarree()
    cf = ax.contourf(lon, lat, spd, levels=SPD_LEV, cmap="Blues", extend="max",
                     alpha=0.9, transform=PC)
    ax.contourf(lon, lat, np.ma.masked_less(div, DIV_LEV[0]), levels=DIV_LEV,
                colors=DIV_COLORS, alpha=0.42, extend="max", transform=PC)
    ax.contour(lon, lat, div, levels=[1, 2, 3], colors="#b81414", linewidths=0.5,
               alpha=0.5, transform=PC)
    s = 30
    ax.barbs(lon[::s], lat[::s], u[::s, ::s], v[::s, ::s], length=4.2, linewidth=0.45,
             color="0.25", transform=PC)
    ax.coastlines(linewidth=0.5, color="0.45")
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=55, shrink=0.7)
    cb.set_label("200 hPa wind speed (m s⁻¹)", fontsize=8); cb.ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(DIV_COLORS),
                               norm=matplotlib.colors.BoundaryNorm(DIV_LEV, len(DIV_COLORS)))
    cb2 = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.10, aspect=55, shrink=0.5, extend="max")
    cb2.set_label("upper-level divergence (10⁻⁵ s⁻¹)", fontsize=8); cb2.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/wind200")
    ap.add_argument("--manifest", default="assets/sst/anim/wind200_manifest.json")
    ap.add_argument("--out", default="assets/sst/wind200.webp")
    args = ap.parse_args()

    cyc = ecmwf.Cycle(args.date, args.time)
    path = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", ("u", "v"), "pl", (200,), tuple(ecmwf.STEPS)))
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    u = ds["u"].squeeze(drop=True).sortby("latitude")     # drop the scalar 200-hPa level coord
    v = ds["v"].squeeze(drop=True).sortby("latitude")
    lat = u.latitude.values; lon = u.longitude.values
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    steps_h = (u.step / np.timedelta64(1, "h")).round().astype(int).values

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, sh in enumerate(steps_h):
        valid = init + pd.Timedelta(hours=int(sh))
        lead = int(round(sh / 24))
        tag = "analysis" if sh == 0 else f"forecast day {lead}"
        title = (f"AIFS-ENS 200 hPa wind & upper-level divergence — {tag}"
                 f"  ·  valid {valid:%Y-%m-%d %HZ}  (init {init:%m-%d %HZ})")
        fp = anim / f"F{i:02d}.webp"
        plot(u.isel(step=i).values, v.isel(step=i).values, lat, lon, title, fp)
        if sh == 0:
            plot(u.isel(step=i).values, v.isel(step=i).values, lat, lon, title, Path(args.out))
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": "analysis" if sh == 0 else f"+{lead} d  ({valid:%a %b %d})"})
    mani = {"ver": int(_time.time()), "regions": {"wind200": {
        "label": "200 hPa wind & upper-level divergence (AIFS-ENS)",
        "n_frames": len(frames), "frames": frames}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"  wrote {len(frames)} frames + manifest + static {Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
