#!/usr/bin/env python3
"""AIFS-ENS stratosphere forecast animations, both hemispheres.

Per cycle, from the tiny ensemble-mean (em) open-data fields:
  strat10:  10 hPa wind speed (fill) + heights (contours), NH | SH per frame
  (control member: AIFS-ENS open data has no em/gh for these levels; z-param
  cf fields are ~0.8 MB each — the ensemble-mean upgrade would need ~1 GB+
  of pf per cycle)
  strat100: 100 hPa height ANOMALY vs NCEP 1991-2020 daily climatology (fill)
            + 100 hPa wind speed (contours), NH | SH per frame
16 daily frames (day 0..15) -> assets/sst/anim/<name>/F##.webp + manifest,
played by the site's standard sst_anim.html iframe player.

    python scripts/strat/build_strat_anim.py [--date YYYYMMDD --time HH]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from ecmwf.opendata import Client
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ANIM = REPO / "assets" / "sst" / "anim"
DATA = HERE / "data"
STEPS = list(range(0, 361, 24))


def fetch_em(date, hh):
    """Per-step requests (bundled multi-step em queries fail on partially
    indexed cycles); skip-and-continue on missing steps."""
    cdir = DATA / f"em_{date}{hh}z"
    cdir.mkdir(parents=True, exist_ok=True)
    c = Client(source="ecmwf", model="aifs-ens")
    fields = {}
    steps_ok = None
    for p, l in (("u", 10), ("v", 10), ("z", 10),
                 ("u", 100), ("v", 100), ("z", 100)):
        parts = []
        got = []
        for s in STEPS:
            f = cdir / f"cf_{p}{l}_s{s:03d}.grib2"
            if not f.exists() or f.stat().st_size < 1e4:
                for attempt in range(3):
                    try:
                        c.retrieve(date=date, time=int(hh), stream="enfo",
                                   type="cf", levtype="pl", param=p,
                                   levelist=l, step=s, target=str(f))
                        break
                    except Exception:
                        time.sleep(5)
                else:
                    continue
            try:
                ds = xr.open_dataset(f, engine="cfgrib",
                                     backend_kwargs=dict(indexpath=""))
                parts.append(ds[list(ds.data_vars)[0]])
                got.append(s)
            except Exception:
                f.unlink(missing_ok=True)
        fields[(p, l)] = xr.concat(parts, dim="step")
        steps_ok = got if steps_ok is None else [s for s in steps_ok if s in got]
    return fields, steps_ok


def clim_on(zltm, lev, doy, lat, lon):
    c = zltm.z_ltm.sel(level=lev, doy=min(doy, 365)).sortby("lat")
    # cyclic longitude point + wrap targets into 0-360
    import xarray as _xr
    last = c.isel(lon=0).assign_coords(lon=360.0)
    c = _xr.concat([c, last], dim="lon")
    return c.interp(lat=("pt", lat.repeat(len(lon))),
                    lon=("pt", np.tile(np.mod(lon, 360.0), len(lat))),
                    method="linear").values.reshape(len(lat), len(lon))


def render_frames(fields, base, name, mode, steps_ok=None):
    outdir = ANIM / name
    outdir.mkdir(parents=True, exist_ok=True)
    zltm = xr.open_dataset(DATA / "z_ltm.nc") if mode == "anom100" else None
    frames = []
    lev = 10 if mode == "spd10" else 100
    u = fields[("u", lev)]; v = fields[("v", lev)]
    g = fields[("z", lev)] / 9.80665
    lat = u.latitude.values; lon = u.longitude.values
    use = steps_ok if steps_ok else STEPS
    for i, s in enumerate(use):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), subplot_kw=None)
        fig.clf()
        axN = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
        axS = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo())
        for ax, hemi in ((axN, "nh"), (axS, "sh")):
            ext = [-180, 180, 38, 90] if hemi == "nh" else [-180, 180, -90, -38]
            ax.set_extent(ext, ccrs.PlateCarree())
            ax.coastlines(lw=0.5, color="0.55")
            ax.gridlines(color="0.7", lw=0.3, alpha=0.6,
                         ylocs=[-80, -70, -60, -50, 50, 60, 70, 80],
                         xlocs=range(-180, 181, 60))
            ui = u.isel(step=i).values; vi = v.isel(step=i).values
            gi = g.isel(step=i).values
            if mode == "spd10":
                spd = np.hypot(ui, vi)
                cf = ax.contourf(lon, lat, spd, levels=np.arange(20, 101, 10),
                                 cmap="plasma", extend="max",
                                 transform=ccrs.PlateCarree())
                ax.contour(lon, lat, gi/1000, levels=np.arange(27, 32.6, 0.3),
                           colors="0.4", linewidths=0.55,
                           transform=ccrs.PlateCarree())
            else:
                anom = gi - clim_on(zltm, 100, valid.dayofyear, lat, lon)
                cf = ax.contourf(lon, lat, anom, levels=np.linspace(-600, 600, 21),
                                 cmap="RdBu_r", extend="both",
                                 transform=ccrs.PlateCarree())
                spd = np.hypot(ui, vi)
                ax.contour(lon, lat, spd, levels=[30, 40, 50],
                           colors="#1b5e20", linewidths=[0.7, 1.0, 1.4],
                           alpha=0.85, transform=ccrs.PlateCarree())
            ax.set_title("Northern Hemisphere" if hemi == "nh"
                         else "Southern Hemisphere", fontsize=10.5)
        ttl = ("10 hPa wind speed + heights" if mode == "spd10"
               else "100 hPa height anomaly (fill) + wind speed (green)")
        fig.suptitle(f"AIFS-ENS control · {ttl} · day {s//24} · "
                     f"valid {valid:%a %b %d}", fontsize=12, fontweight="bold",
                     y=0.98)
        cb = fig.colorbar(cf, ax=[axN, axS], shrink=0.75, pad=0.02)
        cb.set_label("m/s" if mode == "spd10" else "m")
        png = outdir / f"F{i:02d}.png"
        fig.savefig(png, dpi=105, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        Image.open(png).convert("RGB").save(outdir / f"F{i:02d}.webp",
                                            quality=84, method=6)
        png.unlink()
        frames.append({"idx": i, "file": f"F{i:02d}.webp",
                       "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s//24} · {valid:%b %d}"})
    label = ("AIFS-ENS 10 hPa winds & heights" if mode == "spd10" else
             "AIFS-ENS 100 hPa height anomaly & winds")
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {name: {"label": label, "n_frames": len(frames),
                              "frames": frames}}}
    (ANIM / f"{name}_manifest.json").write_text(json.dumps(man))
    print(f"  wrote {len(frames)} frames + manifest for {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--time", default="12")
    a = ap.parse_args()
    if a.date:
        date, hh = a.date, a.time
    else:
        c = Client(source="ecmwf", model="aifs-ens")
        r = c.latest(stream="enfo", type="cf", levtype="pl", param="gh",
                     levelist=10, step=24)
        date, hh = f"{r:%Y%m%d}", f"{r:%H}"
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    print(f"cycle {date} {hh}z", flush=True)
    fields, steps_ok = fetch_em(date, hh)
    print(f"  steps available: {len(steps_ok)}", flush=True)
    render_frames(fields, base, "strat10", "spd10", steps_ok)
    render_frames(fields, base, "strat100", "anom100", steps_ok)
    # prune old cycle caches (keep 2)
    for d in sorted(DATA.glob("*m_*z"))[:-2]:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


if __name__ == "__main__":
    main()