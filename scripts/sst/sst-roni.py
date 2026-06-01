#!/usr/bin/env python3
"""
NOAA OISST sea-surface-temperature anomaly maps + RONI.

Produces three static images for the website:
  1. Global SST anomaly map           -> assets/sst/global_sst_anom.webp
  2. Tropical Pacific SST anomaly map  -> assets/sst/tropical_sst_anom.webp
  3. RONI time series (bar chart)      -> assets/sst/roni.webp
Plus a small manifest               -> assets/sst/manifest.json

Data (NOAA PSL, OISST v2.1 high-res, 1/4 deg, anomalies vs 1991-2020):
  - Daily anomaly (latest day -> the maps):
      https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.<YEAR>.nc
  - Monthly anomaly (full record -> RONI time series):
      https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.anom.nc
    (falls back to the low-res monthly mean + self-computed anomaly if needed)

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
import sys
import urllib.request
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

import os

HERE = Path(__file__).resolve().parent          # wherever this script lives
# Self-contained by default (assets/html under HERE). When embedded in the
# website repo, set SST_SITE_ROOT to the repo root so the page and images are
# written there; the data cache always stays beside the script (and is
# git-ignored). The template lives next to the script either way.
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
HTML_OUT = SITE_ROOT / "sst.html"

PSL_BASE = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
DAILY_ANOM_URL = PSL_BASE + "/sst.day.anom.{year}.nc"
DAILY_MEAN_URL = PSL_BASE + "/sst.day.mean.{year}.nc"

PC = ccrs.PlateCarree()

# Nino-3.4 and tropical-belt boxes (lat S->N, lon in degE 0-360)
NINO34 = dict(lat=(-5, 5), lon=(190, 240))
TROPICAL = dict(lat=(-20, 20), lon=(0, 360))

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
ANIM_DAYS = 60


# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
def _download(url: str, dest: Path, force: bool = False) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  cached: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"  downloading {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)
    print(f"  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


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
def _draw_nino34_box(ax, label=True):
    """Outline the Nino-3.4 region (5S-5N, 170W-120W = 190-240 degE).

    Edges are drawn as dense point sequences along the parallels and
    meridians so the rectangle follows the projection correctly (and
    doesn't bow) regardless of the map's central longitude.
    """
    lon0, lon1 = 190, 240
    lat0, lat1 = -5, 5
    lons = np.linspace(lon0, lon1, 100)
    lats = np.linspace(lat0, lat1, 100)
    edge_lon = np.concatenate([lons, np.full(100, lon1),
                               lons[::-1], np.full(100, lon0)])
    edge_lat = np.concatenate([np.full(100, lat0), lats,
                               np.full(100, lat1), lats[::-1]])
    ax.plot(edge_lon, edge_lat, transform=PC, color="#111", lw=1.4,
            zorder=5, solid_capstyle="round")
    if label:
        ax.text(215, lat1 + 1.5, "Ni\u00f1o-3.4", transform=PC,
                fontsize=8.5, ha="center", va="bottom", color="#111",
                zorder=6, fontweight="bold")


def render_sst_map(anom2d, lat_name, lon_name, extent, title, out_path,
                   figsize, central_lon=0.0, vmin=-5.0, vmax=5.0,
                   annotation=None, nino_box=True):
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
        _draw_nino34_box(ax)

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
    plt.close(fig)
    print(f"  wrote {out_path.name}")


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
        _draw_nino34_box(ax)
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
    fig = plt.figure(figsize=(10.5, 8.2), dpi=88)
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
    fig.savefig(out_path, dpi=88, facecolor="white",
                pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def render_product_anim(field_full, la, lo, product_id, n_days=ANIM_DAYS):
    cfg = PRODUCTS[product_id]
    out_dir = ASSETS / "anim" / product_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("F*.webp"):
        old.unlink()
    times = pd.to_datetime(field_full["time"].values)
    n = min(n_days, len(times))
    sel = field_full.isel(time=slice(len(times) - n, len(times)))
    st = pd.to_datetime(sel["time"].values)
    pfx = _KIND_TITLE[cfg["kind"]]
    frames = []
    for i in range(n):
        out = out_dir / f"F{i:02d}.webp"
        ann = None
        if cfg["kind"] == "anom":
            n34, trop, rel = daily_nino_readout(sel.isel(time=i), la, lo)
            ann = (f"Daily ONI:  {n34:+.2f} °C\n"
                   f"Trop-mean:  {trop:+.2f} °C\n"
                   f"Daily RONI: {rel:+.2f} °C")
        render_2panel_frame(sel.isel(time=i), la, lo, cfg["kind"],
                            f"{pfx} — {st[i]:%Y-%m-%d}", out, annotation=ann)
        frames.append({"idx": i, "file": f"F{i:02d}.webp",
                       "date": f"{st[i]:%Y-%m-%d}", "label": f"{st[i]:%a %b %d, %Y}"})
    print(f"  product '{product_id}': {n} frames", flush=True)
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


def compute_oni_roni(daily_anom, la, lo):
    """Compute ONI and RONI monthly series from the DAILY anomaly field.

    The PSL daily file covers only the current year, so this yields a
    short (year-to-date) series. We resample the gridded daily anomalies
    to monthly means first, then take the Nino-3.4 and tropical-mean box
    averages and a 3-month running mean.

    ONI  = Nino-3.4 monthly anomaly, 3-month running mean.
    RONI = (Nino-3.4 - tropical-mean[20S-20N]) monthly anomaly, 3-mo mean.
    Tropical mean excludes land/missing (NaN) cells via weighted mean.
    Returns DataFrame: month, oni, roni (degC).
    """
    anom = daily_anom
    if float(anom[lo].min()) < 0:
        anom = anom.assign_coords({lo: (anom[lo] % 360)}).sortby(lo)

    # Daily -> monthly-mean gridded anomaly (skip NaN days/cells).
    monthly = anom.resample(time="1MS").mean(skipna=True)

    nino34 = _box_anomaly_series(monthly, la, lo,
                                 NINO34["lat"], NINO34["lon"])
    tropical = _box_anomaly_series(monthly, la, lo,
                                   TROPICAL["lat"], TROPICAL["lon"])

    oni_raw = nino34.to_series()
    roni_raw = (nino34 - tropical).to_series()

    # 3-month centered running mean (ONI/RONI convention). With a short
    # year-to-date series min_periods=1 keeps the early/most-recent months.
    oni = oni_raw.rolling(3, center=True, min_periods=1).mean()
    roni = roni_raw.rolling(3, center=True, min_periods=1).mean()

    df = pd.DataFrame({
        "month": pd.to_datetime(oni.index),
        "oni": oni.values,
        "roni": roni.values,
    }).dropna().reset_index(drop=True)
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


def render_daily_three_metrics(full_anom, la, lo, out_path, days=None):
    """Daily time series of all three metrics on one chart:
    Nino-3.4 anomaly, tropical-mean anomaly, and relative (Nino-3.4 minus
    tropical-mean). Computed from the daily anomaly grid; optionally
    limited to the most recent `days`.
    """
    anom = full_anom
    if float(anom[lo].min()) < 0:
        anom = anom.assign_coords({lo: (anom[lo] % 360)}).sortby(lo)
    if days:
        anom = anom.isel(time=slice(-days, None))

    n34 = _box_anomaly_series(anom, la, lo,
                              NINO34["lat"], NINO34["lon"]).to_series()
    trop = _box_anomaly_series(anom, la, lo,
                               TROPICAL["lat"], TROPICAL["lon"]).to_series()
    rel = n34 - trop
    t = pd.to_datetime(n34.index)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=100)
    ax.plot(t, n34.values, color="#d9402a", lw=1.5,
            label="Ni\u00f1o-3.4 anomaly (ONI-like)")
    ax.plot(t, trop.values, color="#2a8a4a", lw=1.5,
            label="Tropical-mean (20\u00b0S\u201320\u00b0N)")
    ax.plot(t, rel.values, color="#2b6fd6", lw=1.8,
            label="Relative = Ni\u00f1o-3.4 \u2212 tropical (RONI)")

    for y, c in [(0.5, "#d9402a"), (-0.5, "#2b6fd6")]:
        ax.axhline(y, color=c, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="#333", lw=0.8)

    ax.set_ylabel("SST anomaly (\u00b0C)", fontsize=11)
    span = f"last {days} days" if days else f"{t[0]:%b %d} \u2013 {t[-1]:%b %d, %Y}"
    ax.set_title(f"Daily Equatorial Pacific SST Indices \u2014 {span}",
                 fontsize=11, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.2)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncols=3)
    fig.autofmt_xdate()
    fig.text(0.005, 0.005,
             "Daily values computed from NOAA OISST v2.1 (1991\u20132020 "
             "base). Daily \u2014 noisier than the 3-month running indices.",
             fontsize=7, color="#888")

    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def render_roni(df: pd.DataFrame, out_path: Path,
                latest_oni=None, latest_roni=None, latest_month=None):
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=100)
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

    # ENSO threshold guides and zero line.
    for y, c in [(0.5, "#d9402a"), (-0.5, "#2b6fd6")]:
        ax.axhline(y, color=c, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="#333", lw=0.8)

    # One labeled tick per month (e.g. "Jan", with year on January).
    ax.set_xticks(x)
    labels = [m.strftime("%b\n%Y") if m.month == 1 or i == 0
              else m.strftime("%b") for i, m in enumerate(months)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, n - 0.4)

    # Y-limits: include the data AND the +/-0.5 guides, with headroom, so
    # small bars stay visible instead of being dwarfed by the guide lines.
    vmax = float(np.nanmax(roni_vals))
    vmin = float(np.nanmin(roni_vals))
    hi = max(vmax, 0.6) + 0.15
    lo = min(vmin, -0.6) - 0.15
    ax.set_ylim(lo, hi)

    ax.set_ylabel("RONI (\u00b0C)", fontsize=11)
    ax.set_title("Relative Oceanic Ni\u00f1o Index (RONI) \u2014 "
                 "Ni\u00f1o-3.4 minus tropical-mean SST anomaly, "
                 "3-month running mean",
                 fontsize=11, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.2)

    # Current-value readout box.
    if latest_roni is not None:
        oni_line = (f"ONI  {latest_oni:+.2f} \u00b0C\n"
                    if latest_oni is not None else "")
        txt = (f"latest ({latest_month:%b %Y})\n"
               f"{oni_line}"
               f"RONI {latest_roni:+.2f} \u00b0C")
        ax.text(0.985, 0.05, txt, transform=ax.transAxes, fontsize=9,
                va="bottom", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))

    fig.text(0.005, 0.005,
             "Red >+0.5\u00b0C, blue <\u22120.5\u00b0C, grey neutral. "
             "Computed from NOAA OISST v2.1 (1991\u20132020 base); "
             "relative index \u2014 not CPC-standardized.",
             fontsize=7, color="#888")

    fig.savefig(out_path, dpi=100, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-download data even if cached")
    args = ap.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    year = datetime.now(timezone.utc).year

    # --- Daily anomaly for the maps ---
    print("Daily SST anomaly (maps):")
    daily_path = DATA / f"sst.day.anom.{year}.nc"
    try:
        _download(DAILY_ANOM_URL.format(year=year), daily_path,
                  force=args.force)
    except Exception as e:
        # Early-January edge case: this year's file may not exist yet.
        print(f"  {year} daily file failed ({e}); trying {year-1}",
              file=sys.stderr)
        year -= 1
        daily_path = DATA / f"sst.day.anom.{year}.nc"
        _download(DAILY_ANOM_URL.format(year=year), daily_path,
                  force=args.force)

    dsd = xr.open_dataset(daily_path)
    var = _find_var(dsd)
    la, lo = _lat_name(dsd), _lon_name(dsd)
    full = dsd[var]
    # Normalize daily lon to 0-360 for consistent box selection.
    if float(full[lo].min()) < 0:
        full = full.assign_coords({lo: (full[lo] % 360)}).sortby(lo)
    latest = full.isel(time=-1)
    valid = pd.to_datetime(dsd["time"].values[-1])
    print(f"  latest day: {valid:%Y-%m-%d}")

    # --- Absolute SST (sst.day.mean) for the absolute-SST animation ---
    print("Daily absolute SST (sst.day.mean):")
    full_abs = None
    try:
        mean_path = DATA / f"sst.day.mean.{year}.nc"
        _download(DAILY_MEAN_URL.format(year=year), mean_path, force=args.force)
        dsm = xr.open_dataset(mean_path)
        full_abs = dsm[_find_var(dsm, candidates=("sst", "anom"))]
        if float(full_abs[lo].min()) < 0:
            full_abs = full_abs.assign_coords({lo: (full_abs[lo] % 360)}).sortby(lo)
    except Exception as e:
        print(f"  absolute SST unavailable ({e}); skipping that product", file=sys.stderr)

    # Daily 'current conditions' readout (single-day values).
    day_n34, day_trop, day_rel = daily_nino_readout(latest, la, lo)
    print(f"  daily Nino-3.4 (ONI-like) {day_n34:+.2f} degC, "
          f"tropical-mean {day_trop:+.2f} degC, relative (RONI) "
          f"{day_rel:+.2f} degC")

    # --- ONI + RONI from the DAILY field (resampled to monthly), so we
    #     need no separate monthly download. Year-to-date series. ---
    print("ONI/RONI (from daily anomalies, year-to-date):")
    idx = compute_oni_roni(full, la, lo)
    latest_oni = float(idx["oni"].iloc[-1])
    latest_roni = float(idx["roni"].iloc[-1])
    latest_month = idx["month"].iloc[-1]
    print(f"  latest monthly ONI {latest_oni:+.2f} degC, "
          f"RONI {latest_roni:+.2f} degC ({latest_month:%Y-%m})")

    # Daily current-conditions label for the maps (the three requested
    # lines, using single-day values).
    annotation = (
        f"Current Daily Ocean Ni\u00f1o Index: {day_n34:+.2f} \u00b0C\n"
        f"Current Daily Tropical Mean SST: {day_trop:+.2f} \u00b0C\n"
        f"Current Daily Relative Oceanic Ni\u00f1o Index: {day_rel:+.2f} \u00b0C"
    )

    # --- Maps (dateline-centered, +/-5 degC scale) ---
    render_sst_map(latest, la, lo, GLOBAL_EXTENT,
                   f"Global SST Anomaly \u2014 {valid:%Y-%m-%d} "
                   f"(OISST v2.1, base 1991\u20132020)",
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
    dsd.close()

    # --- RONI time series chart (monthly, year-to-date) ---
    render_roni(idx, ASSETS / "roni.webp",
                latest_oni=latest_oni, latest_roni=latest_roni,
                latest_month=latest_month)
    latest_roni_month = latest_month

    # --- Daily 3-metric chart (Nino-3.4, tropical-mean, relative) ---
    print("Daily 3-metric time series:")
    render_daily_three_metrics(full, la, lo,
                               ASSETS / "daily_indices.webp")

    # --- Two-panel animated products: anomaly, absolute, relative ---
    print(f"Two-panel animations (last {ANIM_DAYS} days):")
    full_rel = global_mean_removed(full, la, lo)
    fields = {"anomaly": full, "relative": full_rel}
    if full_abs is not None:
        fields["absolute"] = full_abs
    anim_manifest = {"days": ANIM_DAYS, "regions": {}}
    for pid in ("anomaly", "absolute", "relative"):
        if pid not in fields:
            continue
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
        "daily_relative": round(day_rel, 3),
        "anim_days": ANIM_DAYS,
        "files": {
            "global": "global_sst_anom.webp",
            "tropical": "tropical_sst_anom.webp",
            "roni": "roni.webp",
            "daily_indices": "daily_indices.webp",
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
    """Write repo-root sst.html from the template with values stamped."""
    template = HERE / "sst.html.template"
    out = HTML_OUT
    if not template.exists():
        print(f"  WARN: {template} not found; leaving sst.html untouched",
              file=sys.stderr)
        return
    cache = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    html = template.read_text()
    html = (html
            .replace("__CACHE__", cache)
            .replace("__SST_DAY__", f"OISST {sst_valid:%Y-%m-%d}")
            .replace("__RONI_MONTH__", f"RONI thru {roni_month:%Y-%m}"))
    out.write_text(html)
    print(f"  wrote {out.name} (cache={cache})")


if __name__ == "__main__":
    sys.exit(main())
