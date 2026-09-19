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
def missing_spans(lons: np.ndarray, missing) -> list[tuple[float, float]]:
    """Longitude spans owned by the moorings flagged missing: out to halfway to
    each neighbour (to the grid edge for the end moorings)."""
    if missing is None:
        return []
    spans = []
    for j in np.flatnonzero(missing):
        lo = LON_GRID[0] if j == 0 else 0.5 * (lons[j - 1] + lons[j])
        hi = LON_GRID[-1] if j == len(lons) - 1 else 0.5 * (lons[j] + lons[j + 1])
        spans.append((float(lo), float(hi)))
    return spans


def interp_lon(field2d: np.ndarray, lons: np.ndarray, missing=None) -> np.ndarray:
    """field2d (depth, lon) on mooring lons -> (depth, LON_GRID). Spans of
    moorings flagged `missing` are left NaN rather than bridged by interpolation
    between the reporting neighbours."""
    out = np.full((field2d.shape[0], LON_GRID.size), np.nan)
    for k in range(field2d.shape[0]):
        row = field2d[k]
        m = np.isfinite(row)
        if m.sum() >= 2:
            out[k] = np.interp(LON_GRID, lons[m], row[m],
                               left=np.nan, right=np.nan)
    for lo, hi in missing_spans(lons, missing):
        out[:, (LON_GRID >= lo) & (LON_GRID <= hi)] = np.nan
    return out


def mark_moorings(ax, lons, missing, grid=None) -> None:
    """Mooring triangles along the top; missing ones hollow over a hatched
    'no data yet' span, so a partial day reads as partial, not as a blank chart.
    The hatch covers every empty column of the plotted `grid` (also the edge
    past the outermost reporting mooring, which interpolation cannot reach)."""
    miss = np.zeros(len(lons), bool) if missing is None else np.asarray(missing, bool)
    if miss.any() and grid is not None:
        # contourf leaves the cell next to an empty column unfilled, so each
        # span reaches one grid step into the data (drawn beneath it).
        empty = ~np.isfinite(grid).any(0)
        j = 0
        while j < empty.size:
            if empty[j]:
                k = j
                while k + 1 < empty.size and empty[k + 1]:
                    k += 1
                ax.axvspan(LON_GRID[max(j - 1, 0)], LON_GRID[min(k + 1, empty.size - 1)],
                           facecolor="#e6e4df",
                           edgecolor="#8f8b84", hatch="//", linewidth=0, zorder=0.5)
                j = k + 1
            else:
                j += 1
        for lo in lons[miss]:
            ax.text(np.clip(lo, LON_GRID[0] + 5, LON_GRID[-1] - 5), 150, "no data\nyet", ha="center", va="center", fontsize=8,
                    color="#4a4741", zorder=6,
                    bbox=dict(facecolor="#e6e4df", edgecolor="none", pad=1.5))
    ax.scatter(lons[~miss], np.full((~miss).sum(), 4), marker="v", s=18,
               color="k", clip_on=False, zorder=5)
    ax.set_xlim(LON_GRID[0], LON_GRID[-1])
    if miss.any():
        ax.scatter(lons[miss], np.full(miss.sum(), 4), marker="v", s=18,
                   facecolors="none", edgecolors="#6f6b64", clip_on=False, zorder=5)


def _coverage_note(lons, missing) -> str:
    if missing is None or not np.any(missing):
        return ""
    return f" — {len(lons) - int(np.sum(missing))} of {len(lons)} moorings reporting"


# ── plotting ──────────────────────────────────────────────────────────────────
TEMP_LEVELS = np.arange(8, 31.001, 1.0)
TEMP_ISOTHERMS = [26, 28, 30]
# Z20 is the conventional thermocline proxy, so it gets a heavier line than the
# warm-pool isotherms above. Z10 sits below 300 m in normal conditions — the
# source profiles reach 500 m and 7.6 degC, but DEPTH_GRID stops at 300, so it
# draws only if water that cold reaches into the plotted range. Both are guarded:
# contour() on a level outside the data warns and draws nothing.
THERMOCLINE_ISOTHERM = 20.0
DEEP_ISOTHERM = 10.0
# Anomaly scale, SET FROM THE DATA once per run rather than hand-bumped.
#
# It had been raised three times as the event grew (±8 → ±10 in Jul 2026 → ±12
# in Aug 2026) and was wrong again by 2026-08-30. Worse, the two constants had
# drifted apart: the contour bands ran to ±12 while TwoSlopeNorm still clipped
# the COLOUR mapping at ±10, so everything above +10 was painted the same red -
# genuinely off the scale - while the colourbar implied headroom that did not
# exist. One number now drives both.
#
# Chosen ONCE from the whole series, not per frame: the animation is only
# readable if a colour means the same anomaly in every frame.
ANOM_LIM = 12.0                       # replaced by set_anom_scale() at runtime
ANOM_LEVELS = np.arange(-ANOM_LIM, ANOM_LIM + 0.001, 0.5)
# Labelled contours on the anomaly panel. +10 was missing entirely while the
# field was reaching +11.6, so the strongest core on the plot had no line on it.
ANOM_CONTOURS = [-10, -7, -5, 5, 7, 10, 12, 14]


def set_anom_scale(anom, floor=12.0, step=2.0):
    """Fix the anomaly scale from the data, rounded up, with a floor.

    The floor keeps the scale stable through ordinary conditions so frames stay
    comparable month to month; the round-up means a record event widens it
    automatically instead of saturating and waiting for someone to notice.
    """
    global ANOM_LIM, ANOM_LEVELS
    peak = float(np.nanmax(np.abs(np.asarray(anom, float))))
    lim = max(floor, step * np.ceil(peak / step))
    ANOM_LIM = lim
    ANOM_LEVELS = np.arange(-lim, lim + 0.001, 0.5)
    print(f"  anomaly scale ±{lim:.0f} °C (peak |anomaly| {peak:.2f})", flush=True)
    return lim


def plot_frame(temp2d, anom2d, lons, date, out_path, missing=None):
    Tg = interp_lon(temp2d, lons, missing)
    Ag = interp_lon(anom2d, lons, missing)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    fig.suptitle(f"Equatorial Pacific (0°N) ocean temperature — {date:%d %b %Y}"
                 f"{_coverage_note(lons, missing)}",
                 fontsize=12, fontweight="bold")

    cf1 = ax1.contourf(LON_GRID, DEPTH_GRID, Tg, levels=TEMP_LEVELS,
                       cmap="turbo", extend="both")
    ci = ax1.contour(LON_GRID, DEPTH_GRID, Tg, levels=TEMP_ISOTHERMS,
                     colors="k", linewidths=1.3)
    ax1.clabel(ci, fmt="%d°C", fontsize=8)
    tmin, tmax = np.nanmin(Tg), np.nanmax(Tg)
    if tmin <= THERMOCLINE_ISOTHERM <= tmax:
        c20 = ax1.contour(LON_GRID, DEPTH_GRID, Tg, levels=[THERMOCLINE_ISOTHERM],
                          colors="k", linewidths=2.1)
        ax1.clabel(c20, fmt="%d°C", fontsize=8)
    if tmin <= DEEP_ISOTHERM <= tmax:
        c10 = ax1.contour(LON_GRID, DEPTH_GRID, Tg, levels=[DEEP_ISOTHERM],
                          colors="#1f5fbf", linewidths=1.5, linestyles="--")
        ax1.clabel(c10, fmt="%d°C", fontsize=8)
    ax1.set_title("Temperature", fontsize=10, loc="left")
    fig.colorbar(cf1, ax=ax1, label="°C", pad=0.02, fraction=0.046)

    cf2 = ax2.contourf(LON_GRID, DEPTH_GRID, Ag, levels=ANOM_LEVELS,
                       cmap="RdBu_r", extend="both",
                       norm=mcolors.TwoSlopeNorm(0, -ANOM_LIM, ANOM_LIM))
    ax2.contour(LON_GRID, DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=1.5)
    lv = [v for v in ANOM_CONTOURS if np.nanmin(Ag) <= v <= np.nanmax(Ag)]
    if lv:
        c5 = ax2.contour(LON_GRID, DEPTH_GRID, Ag, levels=lv, colors="k", linewidths=0.8)
        ax2.clabel(c5, fmt="%+d", fontsize=7)
    ax2.set_title("Anomaly (vs 1991–2020)", fontsize=10, loc="left")
    fig.colorbar(cf2, ax=ax2, label="°C", pad=0.02, fraction=0.046)

    for ax, g in ((ax1, Tg), (ax2, Ag)):
        ax.set_ylim(300, 0)
        ax.set_ylabel("Depth (m)")
        ax.set_xticks(lons)
        ax.set_xticklabels([_lon_label(l) for l in lons], fontsize=8)
        mark_moorings(ax, lons, missing, g)
    ax2.set_xlabel("Longitude (mooring sites marked ▾)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_anom_pair(araw2d, adt2d, lons, date, out_path, missing=None):
    """Companion frame: raw anomaly (top) vs the same with the 1991–2020 climate
    trend removed (bottom), so the secular signal's footprint is visible directly."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    fig.suptitle(f"Equatorial Pacific (0°N) temperature anomaly — {date:%d %b %Y}"
                 f"{_coverage_note(lons, missing)}",
                 fontsize=11.5, fontweight="bold")
    panels = [(ax1, interp_lon(araw2d, lons, missing), "Anomaly (vs 1991–2020)"),
              (ax2, interp_lon(adt2d, lons, missing), "Anomaly — detrended with data from 1991–2020")]
    for ax, Ag, title in panels:
        cf = ax.contourf(LON_GRID, DEPTH_GRID, Ag, levels=ANOM_LEVELS, cmap="RdBu_r",
                         extend="both", norm=mcolors.TwoSlopeNorm(0, -ANOM_LIM, ANOM_LIM))
        ax.contour(LON_GRID, DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=1.5)
        # The detrended companion had only the zero line: the +10 core it shares
        # with the raw panel was unlabelled, so switching between them looked
        # like the contour had been lost.
        lvc = [v for v in ANOM_CONTOURS if np.nanmin(Ag) <= v <= np.nanmax(Ag)]
        if lvc:
            cc = ax.contour(LON_GRID, DEPTH_GRID, Ag, levels=lvc, colors="k",
                            linewidths=0.8)
            ax.clabel(cc, fmt="%+d", fontsize=7)
        c5 = ax.contour(LON_GRID, DEPTH_GRID, Ag, levels=[-7, -5, 5, 7], colors="k", linewidths=0.8)
        ax.clabel(c5, fmt="%+d", fontsize=7)
        ax.set_title(title, fontsize=10, loc="left")
        fig.colorbar(cf, ax=ax, label="°C", pad=0.02, fraction=0.046)
        ax.set_ylim(300, 0)
        ax.set_ylabel("Depth (m)")
        ax.set_xticks(lons)
        ax.set_xticklabels([_lon_label(l) for l in lons], fontsize=8)
        mark_moorings(ax, lons, missing, Ag)
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


def _grid_json(field2d: np.ndarray, lons: np.ndarray, missing=None) -> list:
    """(depth, mooring-lon) field -> rounded nested lists on the JSON grid."""
    g = interp_lon(field2d, lons, missing)                        # (DEPTH_GRID, LON_GRID)
    ki = [int(np.argmin(np.abs(DEPTH_GRID - d))) for d in JSON_DEPTHS]
    ji = [int(np.argmin(np.abs(LON_GRID - l))) for l in JSON_LONS]
    sub = g[np.ix_(ki, ji)]
    return [[None if not np.isfinite(v) else round(float(v), 2) for v in row]
            for row in sub]


def publish_json(recent_s, anom, lons, times, missing) -> None:
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
            "temp": _grid_json(recent_s.values[i], lons, missing[i]),
            "anom": _grid_json(anom[i], lons, missing[i]),
            "moorings_in": int(len(lons) - missing[i].sum()),
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
    # PMEL posts a day's moorings over a day or two, so the newest days arrive
    # PARTIAL. Trimming them (up to 3 days) froze the page for 5 days when four
    # moorings went quiet on 2026-09-14, so the trailing partial days are now
    # SHOWN: a mooring with no data on such a day is flagged missing, its span
    # hatched "no data yet" in the frames and nulled in the JSON. The flag is
    # per raw day - without it the centred 5-day mean would carry a silent
    # mooring's older values forward for two more days.
    rep_ = np.isfinite(recent).any([d for d in recent.dims if d not in ("time", "longitude")])
    have = rep_.sum("longitude").values
    usual = int(np.median(have[-17:-3])) if len(have) > 17 else int(have.max())
    missing = np.zeros(rep_.shape, bool)                # (time, longitude)
    i = len(have) - 1
    while i > 0 and have[i] < usual:
        missing[i] = ~rep_.values[i]
        i -= 1
    if missing.any():
        n = int(missing.any(1).sum())
        print(f"  last {n} day(s) partial ({[int(h) for h in have[-n:]]} of the usual {usual} "
              f"moorings) - drawn with the missing spans hatched")
    lons = recent.longitude.values
    coeffs = load_or_build_coeffs(lons)

    # 5-day smoothed recent fields + anomalies vs the harmonic climatology
    recent_s = recent.rolling(time=SMOOTH_DAYS, center=True, min_periods=1).mean()
    recent_s = recent_s.where(~xr.DataArray(missing, dims=("time", "longitude"),
                                            coords={"time": recent.time,
                                                    "longitude": recent.longitude}))
    times = pd.to_datetime(recent_s.time.values)
    anom = recent_s.values - eval_climatology(coeffs, times.dayofyear.values)
    # de-trended companion anomaly (1991–2020 secular trend removed)
    anom_da = xr.DataArray(anom, dims=recent_s.dims, coords=recent_s.coords)
    anom_dt = detrend(anom_da).values

    publish_json(recent_s, anom, lons, times, missing)

    sel = np.arange(max(0, len(times) - ANIM_DAYS), len(times))
    anim_dir = ASSETS / "anim" / "equatorial"
    anim_dt_dir = ASSETS / "anim" / "equatorial_dt"
    for d in (anim_dir, anim_dt_dir):
        if d.exists():
            for f in d.glob("F*.webp"):
                f.unlink()

    # One scale for every frame, from the whole series - see set_anom_scale().
    set_anom_scale(np.concatenate([np.asarray(anom, float).ravel(),
                                   np.asarray(anom_dt, float).ravel()]))
    print(f"rendering {len(sel)} frames (+ de-trended companion) …")
    frames, dates = [], []
    frames_dt = []
    for n, i in enumerate(sel):
        fp = anim_dir / f"F{n:02d}.webp"
        plot_frame(recent_s.values[i], anom[i], lons, times[i], fp, missing[i])
        frames.append(fp); dates.append(times[i])
        fpd = anim_dt_dir / f"F{n:02d}.webp"
        plot_anom_pair(anom[i], anom_dt[i], lons, times[i], fpd, missing[i])
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
