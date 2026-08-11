#!/usr/bin/env python3
"""AIFS single vs AIFS-ENS control — side-by-side forecast animator.

Two-panel TropicalTidbits-style frames (MSLP + 1000-500 thickness + 6-h
precipitation) over North America, every 6 h through the full run, for each
00Z/12Z cycle — the physical-space companion to the spectral-fidelity study:
watch the single's precipitation smear into washes at range while the control
keeps convective texture.

Output: assets/sst/anim/aifs_compare/F##.webp + manifest (site anim player).
~400 MB of open data per cycle, cached under scripts/verify/data/anim_cache
and deleted after rendering.

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
CACHE = HERE / "data" / "anim_cache"
ANIM = REPO / "assets" / "sst" / "anim" / "aifs_compare"
MANIFEST = REPO / "assets" / "sst" / "anim" / "aifs_compare_manifest.json"

STEPS = list(range(0, 361, 6))
G = 9.80665
EXTENT = [-133, -60, 20, 58]                  # North America zoom
P_LEVELS = [0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 50]
P_COLORS = ["#d7ecd7", "#a8d8a8", "#6cbf6c", "#2d8f2d", "#ffe57d",
            "#ffbb4d", "#ff8c33", "#e85c29", "#c62828"]


def fetch(date_iso: str, hh: int) -> bool:
    from ecmwf.opendata import Client
    CACHE.mkdir(parents=True, exist_ok=True)
    for mkey, (model, typ, stream) in (("single", ("aifs-single", "fc", "oper")),
                                       ("ens", ("aifs-ens", "cf", "enfo"))):
        c = Client(source="ecmwf", model=model)
        for kind, kw in (("sfc", dict(levtype="sfc", param=["msl", "tp"])),
                         ("z", dict(levtype="pl", param="z", levelist=[1000, 500]))):
            dest = CACHE / f"{mkey}_{kind}.grib2"
            if dest.exists() and dest.stat().st_size > 1e6:
                continue
            for attempt in range(3):
                try:
                    c.retrieve(date=date_iso, time=hh, type=typ, stream=stream,
                               step=STEPS, target=str(dest), **kw)
                    break
                except Exception as e:                    # noqa: BLE001
                    print(f"{mkey} {kind}: {str(e)[:70]}", file=sys.stderr)
                    time.sleep(10)
            else:
                return False
            print(f"{mkey} {kind}: ✓ {dest.stat().st_size/1e6:.0f} MB", flush=True)
    return True


def load_all():
    out = {}
    for mkey in ("single", "ens"):
        sfc = xr.open_dataset(CACHE / f"{mkey}_sfc.grib2", engine="cfgrib",
                              backend_kwargs=dict(filter_by_keys={"shortName": "msl"},
                                                  indexpath=""))
        tp = xr.open_dataset(CACHE / f"{mkey}_sfc.grib2", engine="cfgrib",
                             backend_kwargs=dict(filter_by_keys={"shortName": "tp"},
                                                 indexpath=""))
        z = xr.open_dataset(CACHE / f"{mkey}_z.grib2", engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
        out[mkey] = dict(msl=sfc["msl"], tp=tp["tp"], z=z["z"])
    return out


def render(date: str, hh: str):
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all()
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
                ax.contour(lon, lat, thk, levels=np.arange(rng[0], rng[1], 6),
                           colors=col, linewidths=0.6, linestyles="--",
                           transform=ccrs.PlateCarree())
            ax.contour(lon, lat, thk, levels=[540], colors="#1565c0",
                       linewidths=1.6, linestyles="--", transform=ccrs.PlateCarree())
            cs = ax.contour(lon, lat, msl, levels=np.arange(940, 1061, 4),
                            colors="k", linewidths=0.75, transform=ccrs.PlateCarree())
            ax.clabel(cs, levels=np.arange(940, 1061, 8), fmt="%d", fontsize=6)
            ax.set_extent(EXTENT, ccrs.PlateCarree())
            ax.coastlines(lw=0.6, color="0.2")
            ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor="0.3", facecolor="none")
            ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.5", facecolor="none")
            ax.add_feature(cfeature.LAKES, lw=0.3, edgecolor="0.5", facecolor="none")
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
    shutil.rmtree(CACHE, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default="00")
    args = ap.parse_args()
    iso = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}"
    if not fetch(iso, int(args.time)):
        raise SystemExit("open data not fully available yet")
    render(args.date, args.time)


if __name__ == "__main__":
    main()
