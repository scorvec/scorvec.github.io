#!/usr/bin/env python3
"""
Equatorial Pacific subsurface temperature & anomaly cross-sections (depth ×
longitude) from TAO/TRITON, with a 120-day animation.

Inputs (from tao_subsurface.py):
  data/tao_eq_recent.nc      recent ~120 days, daily T(time,depth,longitude)
  data/tao_eq_clim_base.nc   1991-2020 daily T (for the climatology)

Pipeline:
  1. Interpolate every profile onto a regular depth grid (the raw files use
     ragged, deployment-dependent sensor depths).
  2. Build a smooth day-of-year climatology per (depth, longitude) by harmonic
     fit (mean + annual + semiannual) over 1991-2020.
  3. Anomaly = recent - climatology(day-of-year), 5-day smoothed.
  4. For each day: interpolate across the moorings onto a fine longitude grid
     and render a 2-panel cross-section (temperature + anomaly).
  5. Compile the frames into an animation.

Outputs:
  assets/sst/equatorial_xsection.webp   latest 2-panel frame
  assets/sst/anim/equatorial/F##.webp   animation frames + manifest.json
  assets/sst/equatorial_xsection.gif    animated GIF (quick view)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
COEFFS_PATH = HERE / "tao_eq_clim_coeffs.nc"   # committed; built once from 1991–2020

# Core equatorial moorings (degE) and the regular grids for analysis/plotting.
STD_LONS = [165.0, 180.0, 190.0, 205.0, 220.0, 235.0, 250.0, 265.0]
DEPTH_GRID = np.arange(0, 301, 5.0)          # 0–300 m, 5 m
LON_GRID = np.arange(165.0, 265.01, 1.0)     # 165°E … 95°W, 1°
SMOOTH_DAYS = 5
ANIM_DAYS = 120


def _lon_label(lon: float) -> str:
    return f"{int(round(lon))}°E" if lon <= 180 else f"{int(round(360 - lon))}°W"


# ── regridding ────────────────────────────────────────────────────────────────
def _vinterp(depths: np.ndarray, temps: np.ndarray) -> np.ndarray:
    m = np.isfinite(temps)
    if m.sum() < 3:
        return np.full(DEPTH_GRID.shape, np.nan)
    d, t = depths[m], temps[m]
    o = np.argsort(d); d, t = d[o], t[o]
    out = np.interp(DEPTH_GRID, d, t)
    out[DEPTH_GRID > d.max()] = np.nan       # don't extrapolate below deepest sensor
    return out


def to_depth_grid(ds: xr.Dataset) -> xr.DataArray:
    """Return T(time, DEPTH_GRID, STD_LONS), vertically interpolated."""
    depths = ds.depth.values.astype(float)
    lons = [l for l in STD_LONS if float(l) in set(ds.longitude.values.astype(float))]
    out = np.full((ds.sizes["time"], len(DEPTH_GRID), len(lons)), np.nan)
    for j, lo in enumerate(lons):
        col = ds["temp"].sel(longitude=lo).values          # (time, depth)
        for i in range(col.shape[0]):
            out[i, :, j] = _vinterp(depths, col[i])
    return xr.DataArray(out, dims=("time", "depth", "longitude"),
                        coords={"time": ds.time.values, "depth": DEPTH_GRID,
                                "longitude": lons})


# ── climatology ───────────────────────────────────────────────────────────────
def harmonic_climatology(base: xr.DataArray) -> xr.DataArray:
    """Fit mean + annual + semiannual per (depth, longitude) over the base
    period. Returns coeffs DataArray (coef[5], depth, longitude)."""
    doy = pd.to_datetime(base.time.values).dayofyear.values
    w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w),
                         np.cos(2 * w), np.sin(2 * w)])
    nd, nl = base.sizes["depth"], base.sizes["longitude"]
    coeffs = np.full((5, nd, nl), np.nan)
    V = base.values                                        # (time, depth, lon)
    for k in range(nd):
        for j in range(nl):
            y = V[:, k, j]
            m = np.isfinite(y)
            if m.sum() > 200:
                coeffs[:, k, j], *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    return xr.DataArray(coeffs, dims=("coef", "depth", "longitude"),
                        coords={"coef": np.arange(5), "depth": base.depth.values,
                                "longitude": base.longitude.values})


def load_or_build_coeffs(lons) -> np.ndarray:
    """Climatology coefficients for the given longitudes. Loads the committed
    cache if present (CI path), else builds from data/tao_eq_clim_base.nc and
    saves it (run this once locally to create the cache)."""
    if COEFFS_PATH.exists():
        cf = xr.open_dataarray(COEFFS_PATH)
    else:
        print("building 1991–2020 harmonic climatology (one-time) …")
        base = to_depth_grid(xr.open_dataset(DATA / "tao_eq_clim_base.nc"))
        cf = harmonic_climatology(base)
        cf.to_netcdf(COEFFS_PATH)
        print(f"  saved {COEFFS_PATH.name}")
    return cf.sel(longitude=list(lons)).values


def eval_climatology(coeffs: np.ndarray, doy: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * np.asarray(doy) / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w),
                         np.cos(2 * w), np.sin(2 * w)])      # (ntime, 5)
    return np.einsum("tp,pkj->tkj", X, coeffs)              # (ntime, depth, lon)


# ── longitude interpolation for smooth contours ──────────────────────────────
def interp_lon(field2d: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """field2d (depth, lon) on mooring lons -> (depth, LON_GRID)."""
    out = np.full((field2d.shape[0], LON_GRID.size), np.nan)
    for k in range(field2d.shape[0]):
        row = field2d[k]
        m = np.isfinite(row)
        if m.sum() >= 2:
            out[k] = np.interp(LON_GRID, lons[m], row[m],
                               left=np.nan, right=np.nan)
    return out


# ── plotting ──────────────────────────────────────────────────────────────────
TEMP_LEVELS = np.arange(8, 31.001, 1.0)
TEMP_ISOTHERMS = [26, 28, 30]
ANOM_LEVELS = np.arange(-8, 8.001, 0.5)
ANOM_LIM = 8.0


def plot_frame(temp2d, anom2d, lons, date, out_path):
    Tg = interp_lon(temp2d, lons)
    Ag = interp_lon(anom2d, lons)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    fig.suptitle(f"Equatorial Pacific (0°N) ocean temperature — {date:%d %b %Y}",
                 fontsize=12, fontweight="bold")

    cf1 = ax1.contourf(LON_GRID, DEPTH_GRID, Tg, levels=TEMP_LEVELS,
                       cmap="turbo", extend="both")
    ci = ax1.contour(LON_GRID, DEPTH_GRID, Tg, levels=TEMP_ISOTHERMS,
                     colors="k", linewidths=1.3)
    ax1.clabel(ci, fmt="%d°C", fontsize=8)
    ax1.set_title("Temperature", fontsize=10, loc="left")
    fig.colorbar(cf1, ax=ax1, label="°C", pad=0.02, fraction=0.046)

    cf2 = ax2.contourf(LON_GRID, DEPTH_GRID, Ag, levels=ANOM_LEVELS,
                       cmap="RdBu_r", extend="both",
                       norm=mcolors.TwoSlopeNorm(0, -ANOM_LIM, ANOM_LIM))
    ax2.contour(LON_GRID, DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=1.5)
    c5 = ax2.contour(LON_GRID, DEPTH_GRID, Ag, levels=[-5, 5], colors="k", linewidths=0.8)
    ax2.clabel(c5, fmt="%+d", fontsize=7)
    ax2.set_title("Anomaly (vs 1991–2020)", fontsize=10, loc="left")
    fig.colorbar(cf2, ax=ax2, label="°C", pad=0.02, fraction=0.046)

    for ax in (ax1, ax2):
        ax.set_ylim(300, 0)
        ax.set_ylabel("Depth (m)")
        ax.set_xticks(lons)
        ax.set_xticklabels([_lon_label(l) for l in lons], fontsize=8)
        ax.scatter(lons, np.full_like(lons, 4), marker="v", s=18,
                   color="k", clip_on=False, zorder=5)
    ax2.set_xlabel("Longitude (mooring sites marked ▾)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── driver ────────────────────────────────────────────────────────────────────
def merge_region(frames, dates, label="Equatorial Pacific T(z) cross-section"):
    """Add/replace the 'equatorial' region in the shared anim manifest.json,
    preserving any other regions (e.g. 'tropical' written by sst-roni.py)."""
    mpath = ASSETS / "anim" / "manifest.json"
    manifest = {"regions": {}}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except Exception:
            pass
    manifest.setdefault("regions", {})
    manifest["regions"]["equatorial"] = {
        "label": label, "n_frames": len(frames),
        "frames": [{"idx": n, "file": f.name, "date": f"{d:%Y-%m-%d}",
                    "label": f"{d:%a %b %d, %Y}"} for n, (f, d) in enumerate(zip(frames, dates))],
    }
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))


def main() -> int:
    recent = to_depth_grid(xr.open_dataset(DATA / "tao_eq_recent.nc"))
    lons = recent.longitude.values
    coeffs = load_or_build_coeffs(lons)

    # 5-day smoothed recent fields + anomalies vs the harmonic climatology
    recent_s = recent.rolling(time=SMOOTH_DAYS, center=True, min_periods=1).mean()
    times = pd.to_datetime(recent_s.time.values)
    anom = recent_s.values - eval_climatology(coeffs, times.dayofyear.values)

    sel = np.arange(max(0, len(times) - ANIM_DAYS), len(times))
    anim_dir = ASSETS / "anim" / "equatorial"
    if anim_dir.exists():
        for f in anim_dir.glob("F*.webp"):
            f.unlink()

    print(f"rendering {len(sel)} frames …")
    frames, dates = [], []
    for n, i in enumerate(sel):
        fp = anim_dir / f"F{n:02d}.webp"
        plot_frame(recent_s.values[i], anom[i], lons, times[i], fp)
        frames.append(fp); dates.append(times[i])

    ASSETS.mkdir(parents=True, exist_ok=True)
    Image.open(frames[-1]).save(ASSETS / "equatorial_xsection.webp")
    merge_region(frames, dates)

    # stamp the TAO date into the page (sst-roni.py fills __CACHE__/__SST_DAY__)
    html = SITE_ROOT / "sst.html"
    if html.exists():
        txt = html.read_text()
        if "__TAO_DAY__" in txt:
            html.write_text(txt.replace("__TAO_DAY__", f"TAO {dates[-1]:%Y-%m-%d}"))

    print(f"\nDone. {len(frames)} frames, latest {dates[-1]:%Y-%m-%d}.")
    print(f"  static : {ASSETS/'equatorial_xsection.webp'}")
    print(f"  region : equatorial added to {ASSETS/'anim'/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
