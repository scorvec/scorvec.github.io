#!/usr/bin/env python3
"""AIFS single vs AIFS-ENS control — side-by-side forecast animator.

Two-panel TropicalTidbits-style frames (MSLP + 1000-500 thickness + 6-h
precipitation) over North America, every 6 h through the full run, for each
00Z/12Z cycle — the physical-space companion to the spectral-fidelity study:
watch the single's precipitation smear into washes at range while the control
keeps convective texture.

Output: assets/sst/anim/aifs_compare/F##.webp + manifest (site anim player).
Data (~400 MB/cycle) flows through the shared ecmwf store (scripts/ecmwf) —
per-file locks, GRIB message-count integrity, mirror fallbacks, cycle-dir
pruning — instead of an ad-hoc cache.

    python aifs_compare_anim.py [--date YYYYMMDD --time 00]
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf
ANIM = REPO / "assets" / "sst" / "anim" / "aifs_compare"
MANIFEST = REPO / "assets" / "sst" / "anim" / "aifs_compare_manifest.json"
ANIM_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500"
MANIFEST_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500_manifest.json"

STEPS = list(range(0, 361, 6))
G = 9.80665
EXTENT = [-128, -63, 22, 55]                  # North America zoom
P_LEVELS = [0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 50]
P_COLORS = ["#d7ecd7", "#a8d8a8", "#6cbf6c", "#2d8f2d", "#ffe57d",
            "#ffbb4d", "#ff8c33", "#e85c29", "#c62828"]


def fetch(date: str, hh: str) -> dict:
    """All four files through the shared store; returns {(model,kind): path}."""
    cyc = ecmwf.Cycle(date, hh)
    S = tuple(STEPS)
    out = {}
    for mkey, model, typ in (("single", "aifs-single", "fc"),
                             ("ens", "aifs-ens", "cf")):
        out[(mkey, "sfc")] = ecmwf.ensure(
            cyc, ecmwf.Spec(model, typ, ("msl", "tp"), "sfc", (), S))
        out[(mkey, "z")] = ecmwf.ensure(
            cyc, ecmwf.Spec(model, typ, "z", "pl", (1000, 500), S))
    return out


def load_all(paths):
    out = {}
    for mkey in ("single", "ens"):
        kw = dict(engine="cfgrib")
        msl = xr.open_dataset(paths[(mkey, "sfc")], backend_kwargs=dict(
            filter_by_keys={"shortName": "msl"}, indexpath=""), **kw)
        tp = xr.open_dataset(paths[(mkey, "sfc")], backend_kwargs=dict(
            filter_by_keys={"shortName": "tp"}, indexpath=""), **kw)
        z = xr.open_dataset(paths[(mkey, "z")],
                            backend_kwargs={"indexpath": ""}, **kw)
        out[mkey] = dict(msl=msl["msl"], tp=tp["tp"], z=z["z"])
    return out


def render(date: str, hh: str, paths):
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all(paths)
    ANIM.mkdir(parents=True, exist_ok=True)
    for old in ANIM.glob("F*.webp"):
        old.unlink()
    frames = []
    lat = F["single"]["msl"].latitude.values
    lon = F["single"]["msl"].longitude.values
    proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1),
                                 constrained_layout=True,
                                 subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"),
                                  ("AIFS single", "AIFS-ENS control")):
            d = F[mkey]
            sd = pd.Timedelta(hours=s)
            msl = d["msl"].sel(step=sd).values / 100.0
            thk = ((d["z"].sel(step=sd, isobaricInhPa=500)
                    - d["z"].sel(step=sd, isobaricInhPa=1000)).values / G / 10.0)
            if s == 0:
                pr = np.zeros_like(msl)
            else:
                pr = (d["tp"].sel(step=sd).values
                      - d["tp"].sel(step=sd - pd.Timedelta(hours=6)).values)
            cf = ax.contourf(lon, lat, np.clip(pr, 0, None), levels=P_LEVELS,
                             colors=P_COLORS, extend="max",
                             transform=ccrs.PlateCarree())
            for rng, col in (((410, 540), "#1565c0"), ((546, 620), "#c62828")):
                ct = ax.contour(lon, lat, thk, levels=np.arange(rng[0], rng[1], 6),
                                colors=col, linewidths=0.6, linestyles="--",
                                transform=ccrs.PlateCarree())
                if len(ct.levels):
                    ax.clabel(ct, levels=list(ct.levels)[::2],
                              fmt="%d", fontsize=5.5, inline_spacing=2)
            c540 = ax.contour(lon, lat, thk, levels=[540], colors="#1565c0",
                              linewidths=1.6, linestyles="--", transform=ccrs.PlateCarree())
            ax.clabel(c540, fmt="%d", fontsize=6, inline_spacing=2)
            cs = ax.contour(lon, lat, msl, levels=np.arange(940, 1061, 4),
                            colors="k", linewidths=0.75, transform=ccrs.PlateCarree())
            ax.clabel(cs, levels=np.arange(940, 1061, 8), fmt="%d", fontsize=6)
            ax.set_extent(EXTENT, ccrs.PlateCarree())
            ax.coastlines(lw=1.1, color="0.05")
            ax.add_feature(cfeature.BORDERS, lw=0.8, edgecolor="0.15", facecolor="none")
            ax.add_feature(cfeature.STATES, lw=0.55, edgecolor="0.3", facecolor="none")
            ax.add_feature(cfeature.LAKES, lw=0.55, edgecolor="0.3", facecolor="none")
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal",
                          pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("6-h precipitation (mm)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"MSLP (hPa) · 1000–500 thickness (dam, 540 bold) · 6-h precip — "
                     f"hour {s} · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold")
        out = ANIM / f"F{i:02d}.webp"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        frames.append({"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"h{s:03d} · {valid:%b %d %HZ}"})
        if i % 12 == 0:
            print(f"  frame {i}/{len(STEPS)}", flush=True)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"aifs_compare": {
               "label": "AIFS single vs AIFS-ENS control — MSLP/thickness/precip",
               "n_frames": len(frames), "frames": frames}}}
    MANIFEST.write_text(json.dumps(man))
    print(f"wrote {len(frames)} frames + manifest")
    render_z500(F, base)


def render_z500(F, base):
    """NH 500 hPa height, contour-filled, single vs control — the planetary
    pattern view of the same comparison."""
    ANIM_Z.mkdir(parents=True, exist_ok=True)
    for old in ANIM_Z.glob("F*.webp"):
        old.unlink()
    frames = []
    lat = F["single"]["z"].latitude.values
    lon = F["single"]["z"].longitude.values
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    levels = np.arange(486, 601, 6)
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6),
                                 constrained_layout=True,
                                 subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"),
                                  ("AIFS single", "AIFS-ENS control")):
            z5 = (F[mkey]["z"].sel(step=pd.Timedelta(hours=s),
                                   isobaricInhPa=500).values / G / 10.0)
            cf = ax.contourf(lon, lat, z5, levels=levels, cmap="turbo",
                             extend="both", transform=ccrs.PlateCarree())
            cl = ax.contour(lon, lat, z5, levels=levels[::2], colors="k",
                            linewidths=0.5, transform=ccrs.PlateCarree())
            ax.clabel(cl, levels=levels[::4], fmt="%d", fontsize=6,
                      inline_spacing=2)
            ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
            ax.coastlines(lw=0.8, color="0.15")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.3",
                           facecolor="none")
            ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 50, 70],
                         xlocs=range(-180, 181, 60))
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal",
                          pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("500 hPa height (dam)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"500 hPa geopotential height · NH — hour {s} · "
                     f"valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold")
        out = ANIM_Z / f"F{i:02d}.webp"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        frames.append({"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"h{s:03d} · {valid:%b %d %HZ}"})
        if i % 12 == 0:
            print(f"  z500 frame {i}/{len(STEPS)}", flush=True)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"aifs_z500": {
               "label": "AIFS single vs AIFS-ENS control — 500 hPa height (NH)",
               "n_frames": len(frames), "frames": frames}}}
    MANIFEST_Z.write_text(json.dumps(man))
    print(f"wrote {len(frames)} z500 frames + manifest")


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default="00")
    args = ap.parse_args()
    paths = fetch(args.date, args.time)
    render(args.date, args.time, paths)


if __name__ == "__main__":
    main()
