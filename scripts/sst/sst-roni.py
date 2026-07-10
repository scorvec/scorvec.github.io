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

# The four canonical CPC ENSO monitoring regions (lon in 0–360°E). Niño-3.4 overlaps
# Niño-4 and Niño-3 by definition. Each colour is shared between the map box outline
# and the region time-series lines so the two read together. Insertion order = west→east.
NINO_REGIONS = {
    "nino4":  dict(lat=(-5, 5),  lon=(160, 210), label="Niño-4",   color="#6a3d9a"),  # 160°E–150°W
    "nino34": dict(lat=(-5, 5),  lon=(190, 240), label="Niño-3.4", color="#e31a1c"),  # 170°W–120°W
    "nino3":  dict(lat=(-5, 5),  lon=(210, 270), label="Niño-3",   color="#33a02c"),  # 150°W–90°W
    "nino12": dict(lat=(-10, 0), lon=(270, 280), label="Niño-1+2", color="#1f78b4"),  # 90°W–80°W
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
ANIM_DAYS = 60
# Animation frames are independent cartopy renders → render them in a process pool.
RENDER_WORKERS = int(os.environ.get("SST_RENDER_WORKERS", str(min(os.cpu_count() or 4, 8))))


# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
def _remote_is_newer(url: str, dest: Path) -> bool:
    """HEAD the URL: True only if its Last-Modified is newer than the cached file
    (so we should refresh). Any failure → False = keep the cache. This is what makes
    the daily/monthly PSL files download once and then reuse until PSL appends new
    data, instead of re-pulling ~240 MB on every (4-hourly) run."""
    try:
        from email.utils import parsedate_to_datetime
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as h:
            lm = h.headers.get("Last-Modified")
        if not lm:
            return False
        return parsedate_to_datetime(lm).timestamp() > dest.stat().st_mtime + 60
    except Exception as e:                                   # noqa: BLE001
        # make a persistently failing HEAD visible instead of silently serving
        # the cache forever
        print(f"  HEAD {url.rsplit('/', 1)[-1]} failed ({repr(e)[:60]}); keeping cache", flush=True)
        return False


def _download(url: str, dest: Path, force: bool = False, tries: int = 4) -> Path:
    """Download with streaming + retry. A genuine 404 raises immediately (so the
    caller's year fallback only fires when the file is truly absent); transient
    failures (IncompleteRead, timeouts, dropped connections) retry — they must NOT
    fall through to a stale prior-year file."""
    import shutil, time, http.client, urllib.error
    DATA.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if not _remote_is_newer(url, dest):
            print(f"  cached: {dest.name} ({dest.stat().st_size/1e6:.1f} MB; PSL not newer)")
            return dest
        print(f"  {dest.name}: PSL published newer data → refreshing", flush=True)
    print(f"  downloading {url} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)            # stream in 1 MB chunks
            if tmp.stat().st_size < 1_000_000:               # guard against a truncated/empty body
                raise IOError(f"suspiciously small download ({tmp.stat().st_size} B)")
            tmp.replace(dest)
            print(f"  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise                                        # truly absent → let caller try year-1
            last = e
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, IOError, OSError) as e:
            last = e
        print(f"  download attempt {attempt}/{tries} failed ({repr(last)[:80]}); retrying in 5s…", flush=True)
        try: tmp.unlink()
        except OSError: pass
        time.sleep(5)
    raise last if last is not None else RuntimeError(f"download failed: {url}")


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
def _draw_box(ax, lat_rng, lon_rng, color, lw=1.2):
    """Outline a lat/lon box. Edges are dense point sequences along the parallels
    and meridians so the rectangle follows the projection (doesn't bow) regardless
    of the map's central longitude."""
    lon0, lon1 = lon_rng
    lat0, lat1 = lat_rng
    lons = np.linspace(lon0, lon1, 100)
    lats = np.linspace(lat0, lat1, 100)
    edge_lon = np.concatenate([lons, np.full(100, lon1), lons[::-1], np.full(100, lon0)])
    edge_lat = np.concatenate([np.full(100, lat0), lats, np.full(100, lat1), lats[::-1]])
    ax.plot(edge_lon, edge_lat, transform=PC, color=color, lw=lw,
            zorder=5, solid_capstyle="round")


# label vertical offsets (deg lat above each box top) so the three equatorial
# boxes' labels stagger instead of colliding; Ni\u00f1o-1+2 sits to the south/east.
_NINO_LAB_DY = {"nino4": 1.3, "nino34": 6.0, "nino3": 1.3, "nino12": 1.3}


def _draw_nino_boxes(ax, label=True):
    """Outline all four CPC ENSO regions (Ni\u00f1o-4/3.4/3/1+2), each in its region
    colour (matching the time-series chart). Ni\u00f1o-3.4 is drawn slightly heavier as
    the primary ONI region. 3.4 overlaps 4 and 3 by definition."""
    for key, r in NINO_REGIONS.items():
        _draw_box(ax, r["lat"], r["lon"], r["color"],
                  lw=1.6 if key == "nino34" else 1.1)
        if label:
            import matplotlib.patheffects as mpe
            lon_c = (r["lon"][0] + r["lon"][1]) / 2.0
            # Black text with a thin white halo so it reads on BOTH the warm absolute-SST
            # map and the lighter anomaly map (the region colour stays on the box outline).
            ax.text(lon_c, r["lat"][1] + _NINO_LAB_DY[key], r["label"], transform=PC,
                    fontsize=7.5, ha="center", va="bottom", color="black",
                    zorder=6, fontweight="bold",
                    path_effects=[mpe.withStroke(linewidth=1.6, foreground="white")])


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
    # defensive: a concurrent pipeline's `git reset --hard` can briefly remove the anim
    # dir mid-render — recreate it so the worker never dies with FileNotFoundError.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=88, facecolor="white",
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


def compute_oni_roni(daily_anom, la, lo):
    """Compute ONI and RONI monthly series from the DAILY anomaly field.

    Box-average the daily grid first (two cheap 1-D series), THEN resample to
    monthly — box means and time means commute, and this avoids materializing
    a monthly copy of the full 1/4-deg grid. The caller may pass a multi-year
    concatenation (main appends the prior-year file) so early-year seasons are
    real and the chart can show a rolling window.

    ONI  = Nino-3.4 monthly anomaly, 3-month running mean.
    RONI = (Nino-3.4 - tropical-mean[20S-20N]) monthly anomaly, 3-mo mean.
    min_periods=3 keeps only true 3-month seasons: the season centered on the
    final data month (whose trailing month has no data yet — the "JJA in July"
    bar) and one centered on the first month drop out naturally.
    Tropical mean excludes land/missing (NaN) cells via weighted mean.
    Returns DataFrame: month, oni, roni (degC).
    """
    anom = daily_anom
    if float(anom[lo].min()) < 0:
        anom = anom.assign_coords({lo: (anom[lo] % 360)}).sortby(lo)

    nino34_d = _box_anomaly_series(anom, la, lo,
                                   NINO34["lat"], NINO34["lon"]).to_series()
    tropical_d = _box_anomaly_series(anom, la, lo,
                                     TROPICAL["lat"], TROPICAL["lon"]).to_series()

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
            label="Relative = Ni\u00f1o-3.4 \u2212 tropical")

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


def render_nino_region_series(full_anom, full_abs, la, lo, out_path, days=120):
    """Recent daily time series for the four CPC Niño regions, two stacked panels
    sharing a date axis: top = absolute SST (°C), bottom = SST anomaly (°C). One
    region-coloured line each (colours match the map boxes); the anomaly panel
    carries the ±0.5 °C El Niño / La Niña reference lines. If the absolute field is
    unavailable, only the anomaly panel is drawn."""
    def _prep(da):
        if float(da[lo].min()) < 0:
            da = da.assign_coords({lo: (da[lo] % 360)}).sortby(lo)
        return da.isel(time=slice(-days, None))

    anom = _prep(full_anom)
    abso = _prep(full_abs) if full_abs is not None else None
    a_ser = {k: _box_anomaly_series(anom, la, lo, r["lat"], r["lon"]).to_series()
             for k, r in NINO_REGIONS.items()}
    b_ser = ({k: _box_anomaly_series(abso, la, lo, r["lat"], r["lon"]).to_series()
              for k, r in NINO_REGIONS.items()} if abso is not None else None)
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
             "Daily cosine-weighted box means from NOAA OISST v2.1 (1991–2020 base). "
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
             "season includes the incomplete current month). NOAA OISST v2.1, 1991\u20132020 base.",
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
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # Only when this year's file is genuinely absent (early-January edge case)
        # do we use last year's — NEVER as a fallback for a transient network drop,
        # which would silently serve stale data. _download already retries those.
        print(f"  {year} daily file not found (404); using {year-1}", file=sys.stderr)
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
          f"tropical-mean {day_trop:+.2f} degC, relative "
          f"{day_rel:+.2f} degC")

    # --- ONI + RONI from the DAILY field (box means resampled to monthly), so
    #     we need no separate monthly download. The prior-year file (cached
    #     once; PSL never rewrites closed years) is concatenated so early-year
    #     seasons like DJF are true 3-month means and the chart shows a rolling
    #     ~18-month window. Maps/animations keep using the current-year field.
    print("ONI/RONI (from daily anomalies, rolling window):")
    full_idx = full
    try:
        prev_path = DATA / f"sst.day.anom.{year - 1}.nc"
        _download(DAILY_ANOM_URL.format(year=year - 1), prev_path)
        dsp = xr.open_dataset(prev_path)
        prev = dsp[_find_var(dsp)]
        if float(prev[lo].min()) < 0:
            prev = prev.assign_coords({lo: (prev[lo] % 360)}).sortby(lo)
        full_idx = xr.concat([prev, full], dim="time")
    except Exception as e:                                   # noqa: BLE001
        print(f"  prior-year daily file unavailable ({repr(e)[:70]}); "
              "indices are year-to-date only", file=sys.stderr)
    idx = compute_oni_roni(full_idx, la, lo)
    idx = idx.tail(18).reset_index(drop=True)                # rolling display window
    latest_oni = float(idx["oni"].iloc[-1])
    latest_roni = float(idx["roni"].iloc[-1])
    latest_month = idx["month"].iloc[-1]
    print(f"  latest monthly ONI {latest_oni:+.2f} degC, "
          f"RONI {latest_roni:+.2f} degC ({latest_month:%Y-%m})")

    # Daily RONI for the map label: single-day relative \u00d7 this month's CPC/ECMWF scale s.
    _sc = load_roni_scale()
    day_roni = day_rel * _sc.get(int(valid.month), 1.0)

    # Daily current-conditions label for the maps (the three requested
    # lines, using single-day values).
    annotation = (
        f"Current Daily Ocean Ni\u00f1o Index: {day_n34:+.2f} \u00b0C\n"
        f"Current Daily Tropical Mean SST: {day_trop:+.2f} \u00b0C\n"
        f"Current Daily Relative Oceanic Ni\u00f1o Index: {day_roni:+.2f} \u00b0C"
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
    # Is the newest bar's season still accumulating days? True unless the daily
    # file already covers through the end of the latest data month.
    last_partial = ((valid + pd.Timedelta(days=1)).month == valid.month)
    render_roni(idx, ASSETS / "roni.webp",
                latest_oni=latest_oni, latest_roni=latest_roni,
                latest_month=latest_month, last_partial=last_partial)
    latest_roni_month = latest_month

    # --- Daily 3-metric chart (Nino-3.4, tropical-mean, relative) ---
    print("Daily 3-metric time series:")
    render_daily_three_metrics(full, la, lo,
                               ASSETS / "daily_indices.webp")

    # --- Niño-region recent series: absolute SST + anomaly, all four boxes ---
    print("Niño-region recent series (absolute + anomaly):")
    render_nino_region_series(full, full_abs, la, lo,
                              ASSETS / "nino_regions.webp")

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
    HOURLY = {"soi_hourly.webp", "kiribati_wind.webp", "kiribati_history.webp", "olr_hovmoller.webp", "olr_waves.webp", "eq_current_hov.webp", "eq_uwind_hov.webp"}
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
