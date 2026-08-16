#!/usr/bin/env python3
"""
NOAA OISST sea-surface-temperature anomaly maps + RONI.

Produces three static images for the website:
  1. Global SST anomaly map           -> assets/sst/global_sst_anom.webp
  2. Tropical Pacific SST anomaly map  -> assets/sst/tropical_sst_anom.webp
  3. RONI time series (bar chart)      -> assets/sst/roni.webp
Plus a small manifest               -> assets/sst/manifest.json

Data (NOAA PSL, OISST v2.1 high-res, 1/4 deg). Anomalies are computed HERE as
sst.day.mean minus PSL's 1991-2020 daily climatology (shared helper
oisst9120.py). NCEI's published sst.day.anom files use a 1971-2000 base (per
their FAQ) and are no longer used anywhere in this repo.

RONI (Relative Oceanic Nino Index), Tier-1 fidelity:
  RONI = Nino-3.4 SST anomaly  -  tropical-mean (20S-20N) SST anomaly,
  with a 3-month running mean. This is the *relative* ENSO index that
  removes the background tropical-mean warming trend. We plot it in degC
  and label it as computed from OISST (not CPC's standardized values).

  Nino-3.4 box: 5S-5N, 170W-120W  (190-240 degE)
  Tropical belt: 20S-20N, all longitudes

This script has no live-data access inside the build sandbox, but the
PSL endpoints are openly reachable from the user's machine and CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

import oisst9120

HERE = Path(__file__).resolve().parent          # wherever this script lives
# Self-contained by default (assets/html under HERE). When embedded in the
# website repo, set SST_SITE_ROOT to the repo root so the page and images are
# written there; the data cache always stays beside the script (and is
# git-ignored). The template lives next to the script either way.
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
HTML_OUT = SITE_ROOT / "sst.html"

PC = ccrs.PlateCarree()

# Nino-3.4 and tropical-belt boxes (lat S->N, lon in degE 0-360)
NINO34 = dict(lat=(-5, 5), lon=(190, 240))
TROPICAL = dict(lat=(-20, 20), lon=(0, 360))

# The four canonical CPC ENSO monitoring regions (lon in 0–360°E). Niño-3.4 overlaps
# Niño-4 and Niño-3 by definition. Each colour is shared between the map box outline
# and the region time-series lines so the two read together. Insertion order = west→east.
# Palette rule: NO reds/blues/oranges — those impersonate the anomaly colormap on the
# maps (a red box over a warm blob disappears or reads as data). Niño-3.4 is near-black
# ink as the primary ONI region; the others are hues absent from both colormaps.
NINO_REGIONS = {
    "nino4":  dict(lat=(-5, 5),  lon=(160, 210), label="Niño-4",   color="#7c4fd0"),  # 160°E–150°W, violet
    "nino34": dict(lat=(-5, 5),  lon=(190, 240), label="Niño-3.4", color="#141414"),  # 170°W–120°W, ink
    "nino3":  dict(lat=(-5, 5),  lon=(210, 270), label="Niño-3",   color="#0e8a80"),  # 150°W–90°W, teal
    "nino12": dict(lat=(-10, 0), lon=(270, 280), label="Niño-1+2", color="#bf3d8d"),  # 90°W–80°W, magenta
}

# Map extents and projection centering. The OISST grid is 0-360 in lon;
# we center both maps on the dateline (central_longitude=180) so the
# Pacific (and the ENSO cold tongue) sits in the middle, uninterrupted.
GLOBAL_CENTRAL_LON = 180.0
GLOBAL_EXTENT = (-180, 180, -65, 65)        # near-global; in PC(clon=180) frame
TROPICAL_CENTRAL_LON = 180.0
# Tropical Pacific basin: 120E to 70W (i.e. 120..290 degE), 30S-30N.
TROPICAL_EXTENT = (120, 290, -30, 30)

# Region definitions reused by the static maps AND the animation viewer.
REGIONS = {
    "global": dict(label="Global", extent=GLOBAL_EXTENT,
                   central_lon=GLOBAL_CENTRAL_LON, figsize=(14, 7)),
    "tropical": dict(label="Tropical Pacific", extent=TROPICAL_EXTENT,
                     central_lon=TROPICAL_CENTRAL_LON, figsize=(14, 5.5)),
}
# Which regions get an animation (the others still get a static map).
ANIM_REGIONS = ["tropical"]
ANIM_DAYS = 90
# Animation frames are independent cartopy renders → render them in a process pool.
RENDER_WORKERS = int(os.environ.get("SST_RENDER_WORKERS", str(min(os.cpu_count() or 4, 8))))


# (download helpers live in oisst9120, shared by every OISST consumer)


def _find_var(ds: xr.Dataset, candidates=("anom", "sst", "anomaly")) -> str:
    """Pick the data variable (PSL anomaly files use 'anom')."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    # fall back to the first non-coordinate data var
    for v in ds.data_vars:
        if ds[v].ndim >= 2:
            return v
    raise RuntimeError(f"No usable data variable in {list(ds.data_vars)}")


def _lat_name(ds) -> str:
    for c in ("lat", "latitude", "Y"):
        if c in ds.coords or c in ds.dims:
            return c
    raise RuntimeError("No latitude coordinate found")


def _lon_name(ds) -> str:
    for c in ("lon", "longitude", "X"):
        if c in ds.coords or c in ds.dims:
            return c
    raise RuntimeError("No longitude coordinate found")


# ----------------------------------------------------------------------
# Colormap: diverging blue-white-red for SST anomaly
# ----------------------------------------------------------------------
def sst_anom_cmap():
    # Perceptually balanced cold->warm with a clean white center.
    stops = [
        (0.00, "#1a3a8f"),
        (0.15, "#2b6fd6"),
        (0.32, "#7db8e8"),
        (0.46, "#cfe6f5"),
        (0.50, "#ffffff"),
        (0.54, "#fbe3d0"),
        (0.68, "#f3a072"),
        (0.85, "#d9402a"),
        (1.00, "#7a0d18"),
    ]
    return LinearSegmentedColormap.from_list("sst_anom", stops)


# ----------------------------------------------------------------------
# Map rendering
# ----------------------------------------------------------------------
def _draw_box(ax, lat_rng, lon_rng, color="black", lw=1.1):
    """Outline a lat/lon box (plain, no white casing — 2026-07-17). Edges are
    dense point sequences along the parallels and meridians so the rectangle
    follows the projection (doesn't bow) regardless of the map's central
    longitude."""
    lon0, lon1 = lon_rng
    lat0, lat1 = lat_rng
    lons = np.linspace(lon0, lon1, 100)
    lats = np.linspace(lat0, lat1, 100)
    edge_lon = np.concatenate([lons, np.full(100, lon1), lons[::-1], np.full(100, lon0)])
    edge_lat = np.concatenate([np.full(100, lat0), lats, np.full(100, lat1), lats[::-1]])
    ax.plot(edge_lon, edge_lat, transform=PC, color=color, lw=lw,
            zorder=5.05, solid_capstyle="round")


# Region extent rulers: dimension-style |\u2014\u2014| lines under the boxes, staggered
# in latitude so the three overlapping equatorial regions read separately.
# (lat of the ruler, lat of its label, label above the line?)
_RULER_ROWS = {"nino4": (-6.6, -7.4, False), "nino3": (-6.6, -7.4, False),
               "nino34": (6.6, 7.4, True)}


def _draw_ruler(ax, lon0, lon1, lat, text, lab_lat, lab_above, fontsize=7.5):
    cap = 0.9                                  # end-cap half-height, deg lat
    ax.plot([lon0, lon1], [lat, lat], transform=PC, color="black", lw=1.0,
            zorder=6)
    for x in (lon0, lon1):
        ax.plot([x, x], [lat - cap, lat + cap], transform=PC, color="black",
                lw=1.0, zorder=6)
    ax.text((lon0 + lon1) / 2.0, lab_lat, text, transform=PC,
            fontsize=fontsize, ha="center", va="bottom" if lab_above else "top",
            color="black", zorder=6, fontweight="bold")


def _draw_nino_boxes(ax, label=True):
    """Outline all four CPC ENSO regions in plain black (coloured, overlapping
    outlines were unreadable), with capped extent rulers under the equator
    identifying each region: Ni\u00f1o-4 and Ni\u00f1o-3 share the shallow row (they
    abut at 150\u00b0W), Ni\u00f1o-3.4 \u2014 which overlaps both \u2014 gets its own deeper row,
    and Ni\u00f1o-1+2's small coastal box is labelled beneath itself."""
    for key, r in NINO_REGIONS.items():
        _draw_box(ax, r["lat"], r["lon"], lw=1.4 if key == "nino34" else 1.0)
    if not label:
        return
    for key, (row, lab_lat, above) in _RULER_ROWS.items():
        r = NINO_REGIONS[key]
        pad = 0.8                              # abutting rulers don't touch
        _draw_ruler(ax, r["lon"][0] + pad, r["lon"][1] - pad, row,
                    r["label"], lab_lat, above)
    r12 = NINO_REGIONS["nino12"]
    ax.text((r12["lon"][0] + r12["lon"][1]) / 2.0, r12["lat"][0] - 1.2,
            r12["label"], transform=PC, fontsize=7.5, ha="center", va="top",
            color="black", zorder=6, fontweight="bold")


def render_sst_map(anom2d, lat_name, lon_name, extent, title, out_path,
                   figsize, central_lon=0.0, vmin=-5.0, vmax=5.0,
                   annotation=None, nino_box=True,
                   png_path=None, png_dpi=150):
    cmap = sst_anom_cmap()
    proj = ccrs.PlateCarree(central_longitude=central_lon)
    fig, ax = plt.subplots(figsize=figsize, dpi=100,
                           subplot_kw=dict(projection=proj))
    # set_extent takes geographic (PlateCarree) coords, so the same
    # -180..180 span works regardless of the projection's central
    # longitude; the dateline-centering just rolls the Pacific to the
    # middle. For the near-global map we trim the polar caps for aspect.
    ax.set_extent(extent, crs=PC)

    lons = anom2d[lon_name].values
    lats = anom2d[lat_name].values
    vals = anom2d.values

    im = ax.pcolormesh(lons, lats, vals, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=PC, shading="auto", rasterized=True,
                       zorder=1)
    ax.add_feature(cfeature.LAND.with_scale("110m"),
                   facecolor="#d9d6cf", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   edgecolor="#555", linewidth=0.4, zorder=3)

    if nino_box:
        _draw_nino_boxes(ax)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.015, shrink=0.85, fraction=0.030,
                        extend="both")
    cbar.set_label("SST anomaly (\u00b0C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=12, loc="left", pad=8)

    if annotation:
        ax.text(0.015, 0.04, annotation, transform=ax.transAxes,
                fontsize=10, va="bottom", ha="left", zorder=6,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.85))

    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 82, "method": 6})
    if png_path is not None:                    # high-res PNG for the overview hero
        fig.savefig(png_path, dpi=png_dpi, facecolor="white", edgecolor="none",
                    bbox_inches="tight", pad_inches=0.08)
        print(f"  wrote {Path(png_path).name} (dpi {png_dpi})")
    plt.close(fig)
    print(f"  wrote {Path(out_path).name}")


def render_anim_frames(full_anom, la, lo, region_id, extent, central_lon,
                       figsize, n_days=60):
    """Render the last n_days daily anomaly frames for one region.

    Writes F00.webp ... F<n-1>.webp into assets/sst/anim/<region_id>/ and
    returns a list of per-frame dicts for the manifest. Frame 0 is the
    oldest, last frame is the most recent (natural play direction).
    """
    out_dir = ASSETS / "anim" / region_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale frames so a shorter run doesn't leave old ones behind.
    for old in out_dir.glob("F*.webp"):
        old.unlink()

    times = pd.to_datetime(full_anom["time"].values)
    n = min(n_days, len(times))
    sel = full_anom.isel(time=slice(len(times) - n, len(times)))
    sel_times = pd.to_datetime(sel["time"].values)

    frames = []
    for i in range(n):
        day = sel.isel(time=i)
        valid = sel_times[i]
        out_path = out_dir / f"F{i:02d}.webp"
        render_sst_map(
            day, la, lo, extent,
            f"SST Anomaly \u2014 {valid:%Y-%m-%d}",
            out_path, figsize=figsize, central_lon=central_lon,
            vmin=-5.0, vmax=5.0, nino_box=True, annotation=None,
        )
        frames.append({
            "idx": i,
            "file": f"F{i:02d}.webp",
            "date": f"{valid:%Y-%m-%d}",
            "label": f"{valid:%a %b %d, %Y}",
        })
    print(f"  region '{region_id}': {n} frames -> {out_dir}")
    return frames


# ----------------------------------------------------------------------
# Two-panel (global + tropical Pacific) animated products
# ----------------------------------------------------------------------
PRODUCTS = {
    "anomaly":  dict(kind="anom", label="SST Anomaly — Global & Tropical Pacific"),
    "absolute": dict(kind="abs",  label="Absolute SST — Global & Tropical Pacific"),
    "relative": dict(kind="rel",  label="SST Anomaly (global-mean removed)"),
}
_KIND_TITLE = {"anom": "SST Anomaly", "abs": "Absolute SST",
               "rel": "SST Anomaly (global-mean removed)"}


def _kind_style(kind):
    if kind == "abs":
        return dict(cmap=plt.get_cmap("turbo"), vmin=-2.0, vmax=32.0,
                    cbar="SST (°C)")
    return dict(cmap=sst_anom_cmap(), vmin=-5.0, vmax=5.0,
                cbar="SST anomaly (°C)")


def _draw_map_ax(ax, field, la, lo, extent, style, nino_box,
                 isotherms=None, annotation=None):
    ax.set_extent(extent, crs=PC)
    im = ax.pcolormesh(field[lo].values, field[la].values, field.values,
                       cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"],
                       transform=PC, shading="auto", rasterized=True, zorder=1)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555",
                   linewidth=0.4, zorder=3)
    if isotherms is not None:
        cs = ax.contour(field[lo].values, field[la].values, field.values,
                        levels=isotherms, colors="k", linewidths=0.6,
                        transform=PC, zorder=4)
        ax.clabel(cs, fmt="%d°C", fontsize=7, inline=True, inline_spacing=2)
    if nino_box:
        # label here: in the 2-panel products only the tropical panel passes
        # nino_box=True (the global panel is nino_box=False), so labels land on the
        # roomy tropical map and never clutter the global one.
        _draw_nino_boxes(ax, label=True)
    if annotation:
        ax.text(0.012, 0.05, annotation, transform=ax.transAxes, fontsize=8.5,
                va="bottom", ha="left", family="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#bbb", alpha=0.85))
    return im


def render_2panel_frame(field, la, lo, kind, title, out_path, annotation=None):
    """Global (top) + tropical Pacific (bottom) map of one field in one figure.
    Equal-height panels; absolute SST gets 26/28/30 °C isotherm contours; the
    anomaly product shows the ONI/tropical-mean/RONI readout on the tropical panel."""
    style = _kind_style(kind)
    isos = [26, 28, 30] if kind == "abs" else None
    fig = plt.figure(figsize=(10.5, 8.2), dpi=125)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.10,
                          left=0.02, right=0.9, top=0.92, bottom=0.03)
    ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=GLOBAL_CENTRAL_LON))
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=TROPICAL_CENTRAL_LON))
    im = _draw_map_ax(ax1, field, la, lo, GLOBAL_EXTENT, style,
                      nino_box=False, isotherms=isos)
    _draw_map_ax(ax2, field, la, lo, TROPICAL_EXTENT, style,
                 nino_box=True, isotherms=isos, annotation=annotation)
    ax1.set_title("Global", fontsize=10, loc="left")
    ax2.set_title("Tropical Pacific", fontsize=10, loc="left")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    cax = fig.add_axes([0.915, 0.12, 0.016, 0.74])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(style["cbar"], fontsize=10)
    cb.ax.tick_params(labelsize=9)
    # defensive: a concurrent pipeline's `git reset --hard` can briefly remove the anim
    # dir mid-render — recreate it so the worker never dies with FileNotFoundError.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor="white",
                pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def _frame_task(args):
    """Process-pool worker: rebuild a light DataArray from the passed arrays and render
    one 2-panel frame. (Passing raw numpy + coords avoids pickling lazy/dask DataArrays.)"""
    vals, lat_vals, lon_vals, la, lo, kind, title, out, ann = args
    import xarray as xr
    field = xr.DataArray(vals, dims=(la, lo), coords={la: lat_vals, lo: lon_vals})
    render_2panel_frame(field, la, lo, kind, title, out, annotation=ann)


def render_product_anim(field_full, la, lo, product_id, n_days=ANIM_DAYS):
    cfg = PRODUCTS[product_id]
    out_dir = ASSETS / "anim" / product_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("F*.webp"):
        old.unlink()
    times = pd.to_datetime(field_full["time"].values)
    n = min(n_days, len(times))
    sel = field_full.isel(time=slice(len(times) - n, len(times))).load()   # eager (pool-safe)
    st = pd.to_datetime(sel["time"].values)
    pfx = _KIND_TITLE[cfg["kind"]]
    lat_vals, lon_vals = sel[la].values, sel[lo].values
    tasks, frames = [], []
    sc = load_roni_scale()          # so the frames' "Daily RONI" matches the maps (scaled)
    for i in range(n):
        ann = None
        if cfg["kind"] == "anom":
            n34, trop, rel = daily_nino_readout(sel.isel(time=i), la, lo)
            ann = (f"Daily ONI:  {n34:+.2f} °C\n"
                   f"Trop-mean:  {trop:+.2f} °C\n"
                   f"Daily RONI: {rel * sc.get(int(st[i].month), 1.0):+.2f} °C")
        tasks.append((sel.isel(time=i).values, lat_vals, lon_vals, la, lo, cfg["kind"],
                      f"{pfx} — {st[i]:%Y-%m-%d}", str(out_dir / f"F{i:02d}.webp"), ann))
        frames.append({"idx": i, "file": f"F{i:02d}.webp",
                       "date": f"{st[i]:%Y-%m-%d}", "label": f"{st[i]:%a %b %d, %Y}"})
    with ProcessPoolExecutor(max_workers=RENDER_WORKERS) as ex:
        list(ex.map(_frame_task, tasks))
    print(f"  product '{product_id}': {n} frames ({RENDER_WORKERS} workers)", flush=True)
    return frames


def global_mean_removed(full, la, lo):
    """Per-time field minus its cosine-weighted ocean-area global mean."""
    w = np.cos(np.deg2rad(full[la]))
    valid = full.notnull()
    gmean = (full * w).sum(dim=(la, lo), skipna=True) / (w * valid).sum(dim=(la, lo))
    return full - gmean


# ----------------------------------------------------------------------
# ONI / RONI
# ----------------------------------------------------------------------
def _box_anomaly_series(anom, la, lo, lat_rng, lon_rng):
    """Cosine-lat-weighted mean anomaly over an ocean box, as a time series.

    Land/missing cells are NaN in OISST anomalies. xarray's weighted mean
    skips NaN in the data AND renormalizes the weights over only the valid
    (ocean) cells, so land is correctly excluded from the average.
    """
    lat = anom[la]
    latsel = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    lonsel = (anom[lo] >= lon_rng[0]) & (anom[lo] <= lon_rng[1])
    box = anom.where(latsel & lonsel, drop=True)
    w = np.cos(np.deg2rad(box[la]))
    # weighted() renormalizes by the sum of weights over non-NaN cells,
    # so missing/land points drop out of both numerator and denominator.
    return box.weighted(w).mean(dim=(la, lo), skipna=True)


def load_roni_scale():
    """Per-calendar-month RONI scale s = σ(ONI)/σ(relative) (CPC/ECMWF), from roni_sigma.json.

    The published RONI = s·(Niño-3.4 − tropical-mean) rescales the relative anomaly back to ONI's
    amplitude (s>1, since removing the tropical mean removes variance), so it stays in °C and shares
    ONI's ±0.5/1.0/1.5 °C thresholds. Returns {1..12: s} or {} (→ no scaling, raw relative shown).
    """
    import json
    p = HERE / "roni_sigma.json"
    if not p.exists():
        return {}
    tab = json.loads(p.read_text()).get("scale_by_month")
    return {int(k): float(v) for k, v in tab.items()} if tab else {}


def scale_roni(rel_monthly):
    """Relative anomaly (°C) × per-month s → the CPC/ECMWF RONI in °C. Identity if no table built."""
    s = load_roni_scale()
    if not s:
        return rel_monthly
    fac = pd.Series([s.get(int(m), 1.0) for m in pd.DatetimeIndex(rel_monthly.index).month],
                    index=rel_monthly.index)
    return rel_monthly * fac


def compute_oni_roni(mean_fields, la, lo):
    """ONI and RONI monthly series from DAILY-MEAN fields (one per year).

    Box-average each daily field and subtract the 1991-2020 box climatology
    (oisst9120; reductions commute, so this equals the box mean of the gridded
    anomaly), then resample monthly and take the centered 3-month mean.
    min_periods=3 keeps only true 3-month seasons: the season centered on the
    final data month (whose trailing month has no data yet) and one centered
    on the first month drop out naturally.

    ONI  = Nino-3.4 monthly anomaly, 3-month running mean.
    RONI = (Nino-3.4 - tropical-mean[20S-20N]) monthly anomaly, 3-mo mean.
    Returns DataFrame: month, oni, roni (degC).
    """
    n34_parts, trop_parts = [], []
    for f in mean_fields:
        n34_parts.append(oisst9120.box_anom_series(f, la, lo,
                                                   NINO34["lat"], NINO34["lon"]))
        trop_parts.append(oisst9120.box_anom_series(f, la, lo,
                                                    TROPICAL["lat"], TROPICAL["lon"]))
    nino34_d = pd.concat(n34_parts).sort_index()
    tropical_d = pd.concat(trop_parts).sort_index()

    oni_raw = nino34_d.resample("MS").mean()
    roni_raw = oni_raw - tropical_d.resample("MS").mean()

    # 3-month centered running mean (ONI/RONI convention), full seasons only.
    oni = oni_raw.rolling(3, center=True, min_periods=3).mean()
    rel = roni_raw.rolling(3, center=True, min_periods=3).mean()   # raw relative (Niño-3.4 − TROP)
    roni = scale_roni(rel)                            # ×s(month) → CPC/ECMWF RONI, still in °C

    df = pd.DataFrame({
        "month": pd.to_datetime(oni.index),
        "oni": oni.values,
        "roni": roni.values,
    }).dropna(subset=["oni", "roni"]).reset_index(drop=True)
    return df


def daily_nino_readout(daily_anom, la, lo):
    """Latest-day Nino-3.4, tropical-mean, and relative values from the
    daily anomaly field. All single-day (noisier than the 3-mo indices).

    Returns (nino34_degC, tropical_mean_degC, relative_degC) where
    relative = Nino-3.4 minus tropical-mean.
    """
    n34 = _box_anomaly_series(daily_anom.expand_dims("t"), la, lo,
                              NINO34["lat"], NINO34["lon"])
    trop = _box_anomaly_series(daily_anom.expand_dims("t"), la, lo,
                               TROPICAL["lat"], TROPICAL["lon"])
    n34v = float(np.asarray(n34).ravel()[0])
    tropv = float(np.asarray(trop).ravel()[0])
    relv = n34v - tropv
    return n34v, tropv, relv


def render_daily_three_metrics(n34, trop, out_path):
    """Daily time series of all three metrics on one chart: Nino-3.4 anomaly,
    tropical-mean anomaly, and relative (Nino-3.4 minus tropical-mean), from
    precomputed daily box anomaly series (vs 1991-2020)."""
    rel = n34 - trop
    t = pd.to_datetime(n34.index)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=100)
    ax.plot(t, n34.values, color="#d9402a", lw=1.5,
            label="Ni\u00f1o-3.4 anomaly (ONI-like)")
    ax.plot(t, trop.values, color="#2a8a4a", lw=1.5,
            label="Tropical-mean (20\u00b0S\u201320\u00b0N)")
    ax.plot(t, rel.values, color="#2b6fd6", lw=1.8,
            label="Relative = Ni\u00f1o-3.4 \u2212 tropical")

    for y, c in [(0.5, "#d9402a"), (-0.5, "#2b6fd6")]:
        ax.axhline(y, color=c, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="#333", lw=0.8)

    ax.set_ylabel("SST anomaly (\u00b0C)", fontsize=11)
    span = f"{t[0]:%b %d} \u2013 {t[-1]:%b %d, %Y}"
    ax.set_title(f"Daily Equatorial Pacific SST Indices \u2014 {span}",
                 fontsize=11, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.2)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncols=3)
    fig.autofmt_xdate()
    fig.text(0.005, 0.005,
             "Daily values from NOAA OISST v2.1 daily means, anomalies vs "
             "1991\u20132020. Daily \u2014 noisier than the 3-month running indices.",
             fontsize=7, color="#888")

    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def render_nino_region_series(a_ser, b_ser, out_path):
    """Recent daily series for the four CPC Niño regions from precomputed box
    series: a_ser = anomaly (vs 1991-2020), b_ser = absolute SST (or None).
    Two stacked panels sharing a date axis; colours match the map boxes."""
    ndays = len(a_ser["nino34"])

    npanel = 2 if b_ser is not None else 1
    fig, axes = plt.subplots(npanel, 1, figsize=(12, 3.1 * npanel + 1.0), dpi=100,
                             sharex=True, squeeze=False)
    axes = list(axes[:, 0])

    if b_ser is not None:
        axA = axes.pop(0)
        for k, r in NINO_REGIONS.items():
            s = b_ser[k]
            axA.plot(s.index, s.values, color=r["color"], lw=1.6, label=r["label"])
        axA.set_ylabel("Absolute SST (°C)", fontsize=11)
        axA.grid(axis="y", alpha=0.2)
        axA.margins(x=0.01)
        axA.legend(loc="lower left", fontsize=8, framealpha=0.85, ncols=4)
        axA.set_title(f"Niño-region sea-surface temperature — last {ndays} days",
                      fontsize=12, loc="left", pad=8)

    axN = axes[0]
    for k, r in NINO_REGIONS.items():
        s = a_ser[k]
        axN.plot(s.index, s.values, color=r["color"], lw=1.6, label=r["label"])
    for y in (0.5, -0.5):
        axN.axhline(y, color="#888", lw=0.8, ls="--", alpha=0.6)
    axN.axhline(0, color="#333", lw=0.8)
    axN.set_ylabel("SST anomaly (°C)", fontsize=11)
    axN.grid(axis="y", alpha=0.2)
    axN.margins(x=0.01)
    if b_ser is None:
        axN.legend(loc="upper left", fontsize=8, framealpha=0.85, ncols=4)
        axN.set_title(f"Niño-region SST anomaly — last {ndays} days",
                      fontsize=12, loc="left", pad=8)

    readout = "   ".join(f"{NINO_REGIONS[k]['label']} {a_ser[k].iloc[-1]:+.1f}"
                         for k in NINO_REGIONS)
    axN.text(0.005, 0.04, f"latest anomaly:  {readout}  °C",
             transform=axN.transAxes, fontsize=8, va="bottom", ha="left", zorder=6,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#bbb", alpha=0.85))

    fig.autofmt_xdate()
    fig.text(0.005, 0.005,
             "Daily cosine-weighted box means from NOAA OISST v2.1 (anomalies vs 1991–2020). "
             "Niño-4 160°E–150°W · 3.4 170–120°W · "
             "3 150–90°W · 1+2 90–80°W, 0–10°S · "
             "daily values (noisier than 3-month indices).",
             fontsize=7, color="#888")
    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# CPC overlapping-season labels for the centered 3-month mean, indexed by centre month (1=Jan).
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def render_roni(df: pd.DataFrame, out_path: Path,
                latest_oni=None, latest_roni=None, latest_month=None,
                last_partial=True):
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=100)
    roni_vals = df["roni"].values
    months = pd.to_datetime(df["month"])
    n = len(roni_vals)
    x = np.arange(n)   # categorical positions, one per month

    # Threshold colors: red if > +0.5, blue if < -0.5, grey otherwise.
    def bar_color(v):
        if v > 0.5:
            return "#d9402a"
        if v < -0.5:
            return "#2b6fd6"
        return "#9a9a96"
    colors = [bar_color(v) for v in roni_vals]

    ax.bar(x, roni_vals, width=0.8, color=colors, edgecolor="#fff",
           linewidth=0.5, zorder=2)
    # Every bar is a true 3-month season (compute_oni_roni uses min_periods=3, with the
    # prior year concatenated), so only the newest bar can be unsettled — hatch it while
    # the current month is still partial (e.g. MJJ drawn mid-July).
    prov = [last_partial and i == n - 1 for i in range(n)]
    for i in range(n):
        if prov[i]:
            ax.bar(x[i], roni_vals[i], width=0.8, color=colors[i], edgecolor="#222",
                   linewidth=0.9, hatch="////", zorder=3)

    # ENSO threshold guides and zero line.
    for y, c in [(0.5, "#d9402a"), (-0.5, "#2b6fd6")]:
        ax.axhline(y, color=c, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="#333", lw=0.8)

    # One labeled tick per month (e.g. "Jan", with year on January).
    ax.set_xticks(x)
    labels = []
    for i, m in enumerate(months):
        head = m.strftime("%b %Y") if (m.month == 1 or i == 0) else m.strftime("%b")
        labels.append(f"{head}\n{SEASONS[m.month - 1]}")     # month over its centered 3-mo season, e.g. "May\nAMJ"
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.6, n - 0.4)

    # Y-limits: include the data AND the +/-0.5 guides, with headroom, so
    # small bars stay visible instead of being dwarfed by the guide lines.
    vmax = float(np.nanmax(roni_vals))
    vmin = float(np.nanmin(roni_vals))
    hi = max(vmax, 0.6) + 0.15
    lo = min(vmin, -0.6) - 0.15
    ax.set_ylim(lo, hi)

    ax.set_ylabel("RONI (\u00b0C)", fontsize=11)
    ax.set_title("Relative Oceanic Ni\u00f1o Index (RONI, CPC/ECMWF) \u2014 (Ni\u00f1o-3.4 \u2212 tropical-mean) SST "
                 "anomaly,\nvariance-rescaled to ONI per calendar month \u00b7 3-month running mean (\u00b0C)",
                 fontsize=10.5, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.2)

    # Current-value readout box.
    if latest_roni is not None:
        oni_line = (f"ONI  {latest_oni:+.2f} \u00b0C\n"
                    if latest_oni is not None else "")
        _tag = ", provisional" if last_partial else ""
        txt = (f"latest: {latest_month:%b %Y} ({SEASONS[latest_month.month - 1]}{_tag})\n"
               f"{oni_line}"
               f"RONI {latest_roni:+.2f} \u00b0C")
        ax.text(0.985, 0.05, txt, transform=ax.transAxes, fontsize=9,
                va="bottom", ha="right", family="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))

    # Two lines: keep each within the plot width so bbox_inches="tight" doesn't stretch the canvas
    # (a single long line ballooned the image to ~2000 px wide and shrank the chart in the card).
    fig.text(0.005, 0.012,
             "RONI = (Ni\u00f1o-3.4 \u2212 tropical-mean 20\u00b0S\u201320\u00b0N) anomaly rescaled by \u03c3(ONI)/\u03c3(relative) per calendar "
             "month (CPC/ECMWF), in \u00b0C, comparable to ONI (red >+0.5, blue <\u22120.5, grey neutral).\n"
             "Each bar is the centered 3-month season (e.g. May = AMJ); a hatched bar is provisional (its "
             "season includes the incomplete current month). NOAA OISST v2.1, anomalies vs 1991\u20132020.",
             fontsize=7, color="#888")

    fig.subplots_adjust(bottom=0.20, top=0.86)   # room for the 2-line season ticks + footnote
    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def publish_enso_daily_json(mean_fields, la, lo, idx, valid, out_path):
    """One JSON feed with every daily ENSO index series the interactive
    overview charts need — the page draws numbers, not pixels.

    daily   : ~2 years of daily box series — anomaly AND absolute SST for the
              four Niño regions, the 20°S–20°N tropical mean, the relative
              index (Niño-3.4 − tropical), the daily RONI (relative × the
              CPC/ECMWF per-month σ-scale) and a 90-day trailing mean of
              Niño-3.4 as the running daily ONI estimate.
    monthly : the official-convention ONI/RONI (3-month centered) table.
    latest  : current values with 7- and 30-day changes, for the stat chips.
    """
    def _cat(fn, region):
        parts = [fn(f, la, lo, region["lat"], region["lon"]) for f in mean_fields]
        return pd.concat(parts).sort_index()

    a = {k: _cat(oisst9120.box_anom_series, r) for k, r in NINO_REGIONS.items()}
    b = {k: _cat(oisst9120.box_mean_series, r) for k, r in NINO_REGIONS.items()}
    trop = _cat(oisst9120.box_anom_series, TROPICAL)
    rel = a["nino34"] - trop
    sc = load_roni_scale()
    months = pd.DatetimeIndex(rel.index).month
    roni_d = rel * pd.Series([sc.get(int(m), 1.0) for m in months], index=rel.index)
    oni90 = a["nino34"].rolling(90, min_periods=90).mean()

    dates = pd.DatetimeIndex(a["nino34"].index)
    rnd = lambda s: [None if not np.isfinite(v) else round(float(v), 3)  # noqa: E731
                     for v in np.asarray(s)]
    daily = {"dates": [d.strftime("%Y-%m-%d") for d in dates],
             "trop": rnd(trop), "rel": rnd(rel), "roni_d": rnd(roni_d),
             "oni90": rnd(oni90)}
    for k in NINO_REGIONS:
        daily[k] = rnd(a[k])
        daily[k + "_abs"] = rnd(b[k])

    def _delta(s, days):
        s = s.dropna()
        if len(s) < 2:
            return None
        past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
        return None if past.empty else round(float(s.iloc[-1] - past.iloc[-1]), 3)

    latest = {"date": f"{valid:%Y-%m-%d}",
              "oni_month": f"{idx['month'].iloc[-1]:%Y-%m}",
              "oni": round(float(idx["oni"].iloc[-1]), 2),
              "roni": round(float(idx["roni"].iloc[-1]), 2)}
    for key, ser in [("nino34", a["nino34"]), ("nino34_abs", b["nino34"]),
                     ("nino12", a["nino12"]), ("nino3", a["nino3"]),
                     ("nino4", a["nino4"]), ("trop", trop),
                     ("roni_d", roni_d), ("oni90", oni90)]:
        sv = ser.dropna()
        if sv.empty:
            continue
        latest[key] = round(float(sv.iloc[-1]), 2)
        latest[key + "_d7"] = _delta(ser, 7)
        latest[key + "_d30"] = _delta(ser, 30)

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "NOAA OISST v2.1 daily means; anomalies vs 1991-2020 (oisst9120)",
        "regions": {k: {"label": r["label"], "color": r["color"]}
                    for k, r in NINO_REGIONS.items()},
        "daily": daily,
        "monthly": {"months": [f"{m:%Y-%m}" for m in idx["month"]],
                    "oni": [round(float(v), 2) for v in idx["oni"]],
                    "roni": [round(float(v), 2) for v in idx["roni"]]},
        "latest": latest,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"  wrote {out_path.relative_to(SITE_ROOT)} "
          f"({len(dates)} days, {out_path.stat().st_size // 1024} KB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-download data even if cached")
    args = ap.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    year = datetime.now(timezone.utc).year

    # --- Daily mean SST: the single source field (maps, animations, indices).
    #     Anomalies are computed against the 1991-2020 climatology (oisst9120). ---
    print("Daily SST mean (anomalies computed vs 1991-2020):")
    try:
        mean_path = oisst9120.ensure_mean(year, force=args.force)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # Only when this year's file is genuinely absent (early-January edge case)
        # do we use last year's — never for a transient failure (download retries).
        print(f"  {year} daily file not found (404); using {year-1}", file=sys.stderr)
        year -= 1
        mean_path = oisst9120.ensure_mean(year, force=args.force)

    dsm = xr.open_dataset(mean_path)
    var = _find_var(dsm, candidates=("sst",))
    la, lo = _lat_name(dsm), _lon_name(dsm)
    full_abs = dsm[var]
    if float(full_abs[lo].min()) < 0:                  # OISST is 0-360 already; defensive
        full_abs = full_abs.assign_coords({lo: (full_abs[lo] % 360)}).sortby(lo)
    valid = pd.to_datetime(dsm["time"].values[-1])
    print(f"  latest day: {valid:%Y-%m-%d}")

    # Gridded anomalies only where a map needs them: the last ANIM_DAYS window
    # (whose newest day also feeds the two static maps). Anything longer runs
    # through the box-level helpers instead — reduce first, then subtract.
    anom_win = oisst9120.anom(full_abs.isel(time=slice(-ANIM_DAYS, None))).load()
    latest = anom_win.isel(time=-1)

    # Daily 'current conditions' readout (single-day values).
    day_n34, day_trop, day_rel = daily_nino_readout(latest, la, lo)
    print(f"  daily Nino-3.4 (ONI-like) {day_n34:+.2f} degC, "
          f"tropical-mean {day_trop:+.2f} degC, relative "
          f"{day_rel:+.2f} degC")

    # --- ONI + RONI monthly indices from box-level anomaly series. The prior
    #     year is included so early seasons are true 3-month means and the
    #     chart shows a rolling ~18-month window. ---
    print("ONI/RONI (box anomalies vs 1991-2020, rolling window):")
    mean_fields = [full_abs]
    try:
        dsp = xr.open_dataset(oisst9120.ensure_mean(year - 1))
        prev = dsp[_find_var(dsp, candidates=("sst",))]
        if float(prev[lo].min()) < 0:
            prev = prev.assign_coords({lo: (prev[lo] % 360)}).sortby(lo)
        mean_fields.insert(0, prev)
    except Exception as e:                                   # noqa: BLE001
        print(f"  prior-year daily file unavailable ({repr(e)[:70]}); "
              "indices are year-to-date only", file=sys.stderr)
    idx = compute_oni_roni(mean_fields, la, lo)
    idx = idx.tail(18).reset_index(drop=True)                # rolling display window
    latest_oni = float(idx["oni"].iloc[-1])
    latest_roni = float(idx["roni"].iloc[-1])
    latest_month = idx["month"].iloc[-1]
    print(f"  latest monthly ONI {latest_oni:+.2f} degC, "
          f"RONI {latest_roni:+.2f} degC ({latest_month:%Y-%m})")

    # Daily RONI for the map label: single-day relative \u00d7 this month's CPC/ECMWF scale s.
    _sc = load_roni_scale()
    day_roni = day_rel * _sc.get(int(valid.month), 1.0)

    # Daily current-conditions label for the maps (single-day values).
    annotation = (
        f"Current Daily Ocean Ni\u00f1o Index: {day_n34:+.2f} \u00b0C\n"
        f"Current Daily Tropical Mean SST: {day_trop:+.2f} \u00b0C\n"
        f"Current Daily Relative Oceanic Ni\u00f1o Index: {day_roni:+.2f} \u00b0C"
    )

    # --- Maps (dateline-centered, +/-5 degC scale) ---
    render_sst_map(latest, la, lo, GLOBAL_EXTENT,
                   f"Global SST Anomaly \u2014 {valid:%Y-%m-%d} "
                   f"(OISST v2.1, anomalies vs {oisst9120.BASE_LABEL})",
                   ASSETS / "global_sst_anom.webp",
                   figsize=(14, 7), central_lon=GLOBAL_CENTRAL_LON,
                   vmin=-5.0, vmax=5.0,
                   annotation=annotation)

    render_sst_map(latest, la, lo, TROPICAL_EXTENT,
                   f"Tropical Pacific SST Anomaly \u2014 {valid:%Y-%m-%d}",
                   ASSETS / "tropical_sst_anom.webp",
                   figsize=(14, 5.5), central_lon=TROPICAL_CENTRAL_LON,
                   vmin=-5.0, vmax=5.0,
                   annotation=annotation)

    # --- RONI time series chart ---
    # Is the newest bar's season still accumulating days? True unless the daily
    # file already covers through the end of the latest data month.
    last_partial = ((valid + pd.Timedelta(days=1)).month == valid.month)
    render_roni(idx, ASSETS / "roni.webp",
                latest_oni=latest_oni, latest_roni=latest_roni,
                latest_month=latest_month, last_partial=last_partial)
    latest_roni_month = latest_month

    # --- Daily index JSON feed (drives the interactive overview charts) ---
    print("Daily ENSO index JSON:")
    publish_enso_daily_json(mean_fields, la, lo, idx, valid,
                            ASSETS / "data" / "enso_daily.json")

    # --- Daily 3-metric chart (year-to-date box anomaly series) ---
    print("Daily 3-metric time series:")
    n34_ytd = oisst9120.box_anom_series(full_abs, la, lo,
                                        NINO34["lat"], NINO34["lon"])
    trop_ytd = oisst9120.box_anom_series(full_abs, la, lo,
                                         TROPICAL["lat"], TROPICAL["lon"])
    render_daily_three_metrics(n34_ytd, trop_ytd, ASSETS / "daily_indices.webp")

    # --- Ni\u00f1o-region recent series: absolute SST + anomaly, all four boxes ---
    print("Ni\u00f1o-region recent series (absolute + anomaly):")
    reg_win = full_abs.isel(time=slice(-120, None))
    a_ser = {k: oisst9120.box_anom_series(reg_win, la, lo, r["lat"], r["lon"])
             for k, r in NINO_REGIONS.items()}
    b_ser = {k: oisst9120.box_mean_series(reg_win, la, lo, r["lat"], r["lon"])
             for k, r in NINO_REGIONS.items()}
    render_nino_region_series(a_ser, b_ser, ASSETS / "nino_regions.webp")

    # --- Two-panel animated products: anomaly, absolute, relative ---
    print(f"Two-panel animations (last {ANIM_DAYS} days):")
    full_rel = global_mean_removed(anom_win, la, lo)
    fields = {"anomaly": anom_win, "relative": full_rel, "absolute": full_abs}
    anim_manifest = {"days": ANIM_DAYS, "regions": {}}
    for pid in ("anomaly", "absolute", "relative"):
        frames = render_product_anim(fields[pid], la, lo, pid, n_days=ANIM_DAYS)
        anim_manifest["regions"][pid] = {
            "label": PRODUCTS[pid]["label"], "n_frames": len(frames),
            "frames": frames,
        }
    # Drop the retired single-panel 'tropical' region dir if present.
    old_trop = ASSETS / "anim" / "tropical"
    if old_trop.exists():
        for f in old_trop.glob("F*.webp"):
            f.unlink()
        try:
            old_trop.rmdir()
        except OSError:
            pass
    # Overwrite manifest with our products; sst_subsurface.py merges
    # 'equatorial' afterwards (the workflow runs it after this script).
    (ASSETS / "anim" / "manifest.json").write_text(
        json.dumps(anim_manifest, indent=2))
    print("  wrote anim/manifest.json (anomaly, absolute, relative)")

    # --- Manifest ---
    manifest = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sst_valid_day": f"{valid:%Y-%m-%d}",
        "roni_latest_month": f"{latest_roni_month:%Y-%m}",
        "roni_latest_value": round(latest_roni, 3),
        "oni_latest_value": round(latest_oni, 3),
        "daily_nino34_anom": round(day_n34, 3),
        "daily_tropical_anom": round(day_trop, 3),
        "daily_relative": round(day_rel, 3),      # raw Niño-3.4 − tropical-mean (unscaled)
        "daily_roni": round(day_roni, 3),         # × σ-scale — matches the map/frame labels
        "anim_days": ANIM_DAYS,
        "files": {
            "global": "global_sst_anom.webp",
            "tropical": "tropical_sst_anom.webp",
            "roni": "roni.webp",
            "daily_indices": "daily_indices.webp",
            "nino_regions": "nino_regions.webp",
        },
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest.json")

    # --- Stamp the HTML page ---
    # The template lives next to this script as sst.html.template and is
    # written to the repo root as sst.html with cache-buster + dates filled
    # in. Keeping the template separate means the {placeholders} never
    # erode across runs.
    stamp_html(valid, latest_roni_month)

    print(f"\nDone. Latest SST day {valid:%Y-%m-%d}, "
          f"RONI {latest_roni:+.2f}\u00b0C ({latest_roni_month:%Y-%m})")
    return 0


def stamp_html(sst_valid, roni_month):
    """Render the El Niño Monitor pages (Overview + 4 subpages) from the shared
    partials/fragments via enso_site, stamping the data tokens (__TAO_DAY__ is
    filled later by sst_subsurface, once the TAO date is known)."""
    sys.path.insert(0, str(HERE))
    import enso_site
    import hashlib
    # Cache-buster = content hash of the static page images, NOT a wall-clock stamp, so
    # the pages are byte-identical between runs when nothing changed (else every poll
    # rewrote ?v= → a spurious commit). It also flips whenever an MJO-owned image on these
    # pages changes, so those get cache-busted too. (anim/ frames self-bust via manifest "ver".)
    # images updated out-of-band by their own hourly Actions cache-bust client-side
    # (see footer.html), so exclude them or they'd churn the page hash every hour.
    HOURLY = {"soi_hourly.webp", "kiribati_wind.webp", "kiribati_history.webp", "olr_hovmoller.webp", "olr_waves.webp", "wave_tracker.webp", "eq_current_hov.webp", "eq_uwind_hov.webp"}
    h = hashlib.md5()
    for p in sorted(ASSETS.rglob("*.webp")):
        if "/anim/" in p.as_posix() or p.name in HOURLY:
            continue
        h.update(p.name.encode()); h.update(p.read_bytes())
    cache = h.hexdigest()[:12]
    tokens = {"cache": cache,
              "sst_day": f"OISST {sst_valid:%Y-%m-%d}",
              "roni_month": f"RONI thru {roni_month:%Y-%m}"}
    written = enso_site.render_all(tokens, SITE_ROOT)
    print(f"  wrote {len(written)} pages (cache={cache}): "
          + ", ".join(p.name for p in written))


if __name__ == "__main__":
    sys.exit(main())
