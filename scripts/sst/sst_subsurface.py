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


# ── secular (climate) trend ───────────────────────────────────────────────────
# A per-(depth, longitude) linear trend in the deseasonalized anomaly, fit over
# the same 1991–2020 TAO base record the climatology comes from. Removing it lets
# us compare events across decades (1997 vs 2023 vs now) with the background ocean
# warming/cooling taken out, isolating the ENSO signal. Referenced to the base-
# period midpoint so de-trending only tilts the series — it leaves the 1991–2020
# mean (hence the anomaly zero-line) unchanged.
TREND_PATH = HERE / "tao_eq_trend_coeffs.nc"   # committed; built once from the base
TREND_REF = 2005.5                             # midpoint of 1991–2020 (trend pivot)
TREND_PERIOD = "1991–2020"
_TREND_MIN_PTS = 365                           # ~1 yr of valid days required to fit
_TREND_MIN_SPAN = 15.0                         # …spanning ≥15 yr (else leave NaN→0)
_TREND_DSMOOTH = 5                             # vertical smoothing of the slope (cells)


def decimal_year(times) -> np.ndarray:
    t = pd.to_datetime(times)
    return (t.year + (t.dayofyear - 1) / 365.25).values.astype(float)


def build_trend(base: xr.DataArray, coeffs: np.ndarray) -> xr.DataArray:
    """Linear slope (°C/yr) per (depth, longitude) of the base-period anomaly."""
    t = pd.to_datetime(base.time.values)
    anom = base.values - eval_climatology(coeffs, t.dayofyear.values)   # (time,depth,lon)
    dy = decimal_year(t) - TREND_REF
    nd, nl = base.sizes["depth"], base.sizes["longitude"]
    slope = np.full((nd, nl), np.nan)
    for k in range(nd):
        for j in range(nl):
            y = anom[:, k, j]; m = np.isfinite(y)
            if m.sum() >= _TREND_MIN_PTS and (dy[m].max() - dy[m].min()) >= _TREND_MIN_SPAN:
                X = np.column_stack([np.ones(m.sum()), dy[m]])
                slope[k, j] = np.linalg.lstsq(X, y[m], rcond=None)[0][1]
    sl = xr.DataArray(slope, dims=("depth", "longitude"),
                      coords={"depth": base.depth.values, "longitude": base.longitude.values})
    # light vertical smoothing isolates the climate signal from per-cell TAO noise
    sl = sl.rolling(depth=_TREND_DSMOOTH, center=True, min_periods=1).mean()
    sl.attrs.update(ref_year=TREND_REF, period=TREND_PERIOD, units="degC/yr")
    return sl


def load_or_build_trend() -> xr.DataArray:
    """Trend-slope DataArray(depth, longitude). Loads the committed cache if present,
    else builds it from data/tao_eq_clim_base.nc (run once locally to create it)."""
    if TREND_PATH.exists():
        return xr.open_dataarray(TREND_PATH)
    print("building 1991–2020 subsurface trend (one-time) …")
    base = to_depth_grid(xr.open_dataset(DATA / "tao_eq_clim_base.nc"))
    coeffs = load_or_build_coeffs(base.longitude.values)
    sl = build_trend(base, coeffs)
    sl.to_netcdf(TREND_PATH)
    print(f"  saved {TREND_PATH.name}")
    return sl


def detrend(anom: xr.DataArray) -> xr.DataArray:
    """Remove the secular climate trend from an anomaly grid (time, depth, longitude).
    Subtracts slope(depth, lon) × (decimal_year − {ref}); cells without a fitted
    slope are left unchanged (NaN slope → 0)."""
    sl = load_or_build_trend().reindex(longitude=anom.longitude, depth=anom.depth).fillna(0.0)
    dy = xr.DataArray(decimal_year(anom.time.values) - TREND_REF,
                      dims="time", coords={"time": anom.time})
    return anom - sl * dy


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
ANOM_LEVELS = np.arange(-12, 12.001, 0.5)   # ±8→±10 (Jul 2026) → ±12 (Aug 2026: +10.65 at the thermocline)
ANOM_LIM = 10.0


def plot_frame(temp2d, anom2d, lons, date, out_path):
    Tg = interp_lon(temp2d, lons)
    Ag = interp_lon(anom2d, lons)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
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
    c5 = ax2.contour(LON_GRID, DEPTH_GRID, Ag, levels=[-7, -5, 5, 7], colors="k", linewidths=0.8)
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


def plot_anom_pair(araw2d, adt2d, lons, date, out_path):
    """Companion frame: raw anomaly (top) vs the same with the 1991–2020 climate
    trend removed (bottom), so the secular signal's footprint is visible directly."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    fig.suptitle(f"Equatorial Pacific (0°N) temperature anomaly — {date:%d %b %Y}",
                 fontsize=11.5, fontweight="bold")
    panels = [(ax1, interp_lon(araw2d, lons), "Anomaly (vs 1991–2020)"),
              (ax2, interp_lon(adt2d, lons), "Anomaly — detrended with data from 1991–2020")]
    for ax, Ag, title in panels:
        cf = ax.contourf(LON_GRID, DEPTH_GRID, Ag, levels=ANOM_LEVELS, cmap="RdBu_r",
                         extend="both", norm=mcolors.TwoSlopeNorm(0, -ANOM_LIM, ANOM_LIM))
        ax.contour(LON_GRID, DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=1.5)
        c5 = ax.contour(LON_GRID, DEPTH_GRID, Ag, levels=[-7, -5, 5, 7], colors="k", linewidths=0.8)
        ax.clabel(c5, fmt="%+d", fontsize=7)
        ax.set_title(title, fontsize=10, loc="left")
        fig.colorbar(cf, ax=ax, label="°C", pad=0.02, fraction=0.046)
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


# ── interactive JSON feed ─────────────────────────────────────────────────────
# Powers the Plotly cross-section on enso-subsurface.html: snapshots every
# JSON_STEP days (plus the latest day) of temperature + anomaly on a coarser
# (depth x longitude) grid, and the TAO mooring metadata — actual buoy
# longitudes, their sensor depths, and whether each reported recently.
JSON_STEP = 5
JSON_DEPTHS = np.arange(0, 301, 10.0)
JSON_LONS = np.arange(165.0, 265.01, 2.5)


def _grid_json(field2d: np.ndarray, lons: np.ndarray) -> list:
    """(depth, mooring-lon) field -> rounded nested lists on the JSON grid."""
    g = interp_lon(field2d, lons)                        # (DEPTH_GRID, LON_GRID)
    ki = [int(np.argmin(np.abs(DEPTH_GRID - d))) for d in JSON_DEPTHS]
    ji = [int(np.argmin(np.abs(LON_GRID - l))) for l in JSON_LONS]
    sub = g[np.ix_(ki, ji)]
    return [[None if not np.isfinite(v) else round(float(v), 2) for v in row]
            for row in sub]


def publish_json(recent_s, anom, lons, times) -> None:
    raw = xr.open_dataset(DATA / "tao_eq_recent.nc")
    last7 = raw["temp"].isel(time=slice(-7, None))
    buoys = []
    for lo in lons:
        col = raw["temp"].sel(longitude=lo)
        depths = [float(d) for d in raw.depth.values
                  if d <= 300 and np.isfinite(col.sel(depth=d).values).any()]
        live = float(np.isfinite(last7.sel(longitude=lo).values).mean()) > 0.2
        buoys.append({"lon": float(lo), "label": f"0°N {_lon_label(lo)}",
                      "depths": depths, "live": live})
    raw.close()

    idx = sorted(set(range(len(times) - 1, -1, -JSON_STEP)))
    snaps = []
    for i in idx:
        snaps.append({
            "date": f"{times[i]:%Y-%m-%d}",
            "label": f"{times[i]:%b %d}",
            "temp": _grid_json(recent_s.values[i], lons),
            "anom": _grid_json(anom[i], lons),
        })
    out = {
        "depths": [float(d) for d in JSON_DEPTHS],
        "lons": [float(l) for l in JSON_LONS],
        "lon_labels": [_lon_label(l) for l in JSON_LONS],
        "buoys": buoys,
        "snapshots": snaps,
        "base": "1991–2020",
        "smooth_days": SMOOTH_DAYS,
    }
    path = ASSETS / "data" / "tao_section.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path.name} ({path.stat().st_size/1e3:.0f} KB, "
          f"{len(snaps)} snapshots, {len(buoys)} buoys)")


# ── driver ────────────────────────────────────────────────────────────────────
def merge_region(frames, dates, label="Equatorial Pacific T(z) cross-section",
                 region="equatorial"):
    """Add/replace the named region in the shared anim manifest.json, preserving
    any other regions (e.g. 'tropical' written by sst-roni.py)."""
    mpath = ASSETS / "anim" / "manifest.json"
    manifest = {"regions": {}}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except Exception:
            pass
    manifest.setdefault("regions", {})
    manifest["regions"][region] = {
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
    # de-trended companion anomaly (1991–2020 secular trend removed)
    anom_da = xr.DataArray(anom, dims=recent_s.dims, coords=recent_s.coords)
    anom_dt = detrend(anom_da).values

    publish_json(recent_s, anom, lons, times)

    sel = np.arange(max(0, len(times) - ANIM_DAYS), len(times))
    anim_dir = ASSETS / "anim" / "equatorial"
    anim_dt_dir = ASSETS / "anim" / "equatorial_dt"
    for d in (anim_dir, anim_dt_dir):
        if d.exists():
            for f in d.glob("F*.webp"):
                f.unlink()

    print(f"rendering {len(sel)} frames (+ de-trended companion) …")
    frames, dates = [], []
    frames_dt = []
    for n, i in enumerate(sel):
        fp = anim_dir / f"F{n:02d}.webp"
        plot_frame(recent_s.values[i], anom[i], lons, times[i], fp)
        frames.append(fp); dates.append(times[i])
        fpd = anim_dt_dir / f"F{n:02d}.webp"
        plot_anom_pair(anom[i], anom_dt[i], lons, times[i], fpd)
        frames_dt.append(fpd)

    ASSETS.mkdir(parents=True, exist_ok=True)
    Image.open(frames[-1]).save(ASSETS / "equatorial_xsection.webp")
    Image.open(frames_dt[-1]).save(ASSETS / "equatorial_xsection_detrended.webp")
    merge_region(frames, dates)
    merge_region(frames_dt, dates, label="Equatorial Pacific anomaly — raw vs de-trended",
                 region="equatorial_dt")

    # stamp the TAO date into every monitor page that shows it (sst-roni.py rendered
    # the pages with __CACHE__/__SST_DAY__ already filled, leaving __TAO_DAY__ for here)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import enso_site
    enso_site.stamp_tao(SITE_ROOT, f"TAO {dates[-1]:%Y-%m-%d}")

    print(f"\nDone. {len(frames)} frames, latest {dates[-1]:%Y-%m-%d}.")
    print(f"  static : {ASSETS/'equatorial_xsection.webp'}")
    print(f"  region : equatorial added to {ASSETS/'anim'/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
