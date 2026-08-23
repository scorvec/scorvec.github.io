#!/usr/bin/env python3
"""
Equatorial Pacific SST at kilometre scale — NASA JPL MUR v4.1 (GHRSST L4).

Two daily images for the El Niño monitor:
  assets/sst/mur_eqpac.webp      — 10°S–10°N, 160°E–80°W at 0.05° (strided):
                                   the whole cold tongue / warm pool stage
  assets/sst/mur_coldtongue.webp — 5°S–5°N, 145°W–85°W at native 0.01°:
                                   tropical instability wave cusps, front
                                   filaments, and the Galápagos wake

Source: NOAA CoastWatch ERDDAP (jplMURSST41) — credential-free HTTPS subsets,
~2-day latency. The AWS Open Data zarr mirror is a frozen archive (ends 2020),
so ERDDAP is the near-real-time path. MUR is heavily interpolated where IR is
cloud-blocked; treat fine structure as analysis, not observation.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parent.parent)
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "mur"

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41"

WIDE = dict(lat=(-10, 10), stride=5)          # 0.05° effective
ZOOM = dict(lat=(-8, 8), lon=(-155, -79))     # native 0.01°, out to the Peru coast
# Colour range adapts to each day's field (1st–99th percentile, rounded to
# 0.5 °C): a fixed range wastes the palette when the cold tongue runs warm
# (El Niño) and hides the TIW / front detail this product exists to show.
def _auto_range(field):
    lo = float(np.floor(np.nanpercentile(field, 1) * 2) / 2)
    hi = float(np.ceil(np.nanpercentile(field, 99) * 2) / 2)
    return lo, max(hi, lo + 2.0)


def _fetch(url: str, dest: Path, tries: int = 3) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"  fetch attempt {attempt}/{tries} failed ({repr(e)[:70]})",
                  flush=True)
            tmp.unlink(missing_ok=True)
            time.sleep(20 * attempt)
    raise last


def latest_time(tries: int = 4) -> pd.Timestamp:
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(f"{ERDDAP}.das", timeout=60) as r:
                das = r.read().decode("utf-8", errors="replace")
            m = re.search(r'time \{[^}]*actual_range ([0-9.e+]+), ([0-9.e+]+)',
                          das, re.S)
            return pd.Timestamp(float(m.group(2)), unit="s")
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"  .das attempt {attempt}/{tries} failed ({repr(e)[:60]})",
                  flush=True)
            time.sleep(30 * attempt)
    raise last


def grab(t: pd.Timestamp, lat, lon, stride: int = 1, tag: str = "x") -> xr.DataArray:
    ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    s = f":{stride}:" if stride > 1 else ":"
    url = (f"{ERDDAP}.nc?analysed_sst%5B({ts})%5D"
           f"%5B({lat[0]}){s}({lat[1]})%5D%5B({lon[0]}){s}({lon[1]})%5D")
    p = _fetch(url, DATA / f"mur_{tag}.nc")
    with xr.open_dataset(p) as ds:
        da = ds["analysed_sst"].squeeze("time", drop=True).load()
        if ds["analysed_sst"].attrs.get("units", "").lower().startswith("k"):
            da = da - 273.15                     # PO.DAAC serves Kelvin; CoastWatch, Celsius
    return da


def _style(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d4c8", zorder=3)
    ax.coastlines(resolution="10m", lw=0.5, color="#444", zorder=4)
    gl = ax.gridlines(draw_labels=True, lw=0.25, color="#999", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}


def render(field: xr.DataArray, extent, title: str, out: Path,
           figsize, note: str):
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=figsize, dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    _style(ax, extent)
    lon = field["longitude"].values
    vmin, vmax = _auto_range(field.values)
    mesh = ax.pcolormesh(lon, field["latitude"].values, field.values,
                         transform=ccrs.PlateCarree(), cmap="turbo",
                         vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.012,
                      fraction=0.025, aspect=28)
    cb.set_label("SST (°C)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    fig.text(0.005, 0.005, note, fontsize=6.5, color="#666")
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.name} ({out.stat().st_size/1e3:.0f} kB)", flush=True)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    t = latest_time()
    print(f"MUR latest analysis: {t:%Y-%m-%d}", flush=True)
    note = ("NASA JPL MUR SST v4.1 (GHRSST L4, ~1 km analysis) via NOAA CoastWatch "
            "ERDDAP · MUR interpolates under cloud — fine structure is analysis, "
            "not direct observation.")

    # Wide view: two lon chunks (ERDDAP ranges cannot cross the antimeridian).
    west = grab(t, WIDE["lat"], (160, 179.99), stride=WIDE["stride"], tag="wide_w")
    east = grab(t, WIDE["lat"], (-179.99, -80), stride=WIDE["stride"], tag="wide_e")
    east = east.assign_coords(longitude=east["longitude"] % 360)
    west = west.assign_coords(longitude=west["longitude"] % 360)
    wide = xr.concat([west, east], dim="longitude").sortby("longitude")
    render(wide, (160, 280, -10, 10),
           f"Equatorial Pacific SST — MUR 1 km analysis (0.05° view) — {t:%Y-%m-%d}",
           ASSETS / "mur_eqpac.webp", figsize=(16, 3.6), note=note)

    # Cold-tongue zoom at native resolution.
    zoom = grab(t, ZOOM["lat"], ZOOM["lon"], tag="zoom")
    zoom = zoom.assign_coords(longitude=zoom["longitude"] % 360)
    render(zoom, (205, 281, -8, 8),
           f"Cold tongue & tropical instability waves — MUR native 0.01° — {t:%Y-%m-%d}",
           ASSETS / "mur_coldtongue.webp", figsize=(20, 4.0), note=note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
