#!/usr/bin/env python3
"""GOES-East GLM lightning flash-density loop (trailing 3-hour accumulation).

Companion to the GMGSI IR loop: where the IR shows cold cloud tops, this shows where
lightning has actually been striking. NOAA's Geostationary Lightning Mapper (GLM) on
GOES-East reports every flash as a point; we bin the flashes into a 0.1° grid and, for
each hourly frame, accumulate the trailing 3 hours, then render a log-scaled density
heatmap over the same South & Central America region (same coastlines/borders/Brazil
states for orientation).

Efficiency: GLM L2 is ~180 tiny granules/hour, so we cache a per-clock-hour binned grid
(data/glm_hourly/, gitignored) and compose each trailing-3 h frame from 3 cached grids —
an hourly run only fetches the newest hour. Committed webp frames persist the rolling
window across fresh CI checkouts (same pattern as the IR loops).

    python scripts/sst/glm_density.py --hours 24    # render last N hourly frames + manifest
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import json
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacific_satellite import _offset, _state_geoms   # shared geometry helpers

HERE = Path(__file__).resolve().parent
ANIM_ROOT = HERE.parent.parent / "assets" / "sst" / "anim"
HOURLY_CACHE = HERE / "data" / "glm_hourly"           # per-clock-hour binned grids (gitignored)
GLM_S3 = "https://noaa-goes19.s3.amazonaws.com"       # GOES-East GLM L2 (anonymous S3)
BIN = 0.1                                             # density grid spacing (°)
KEEP_H = 72                                           # rolling loop length (hours)
CACHE_PREFIX = "glmden"                               # hourly-grid cache filename prefix
# Accumulation windows offered in the animator dropdown: (region id, hours, colorbar vmax,
# label). Each composes from the shared hourly grids; vmax is per-window (Catatumbo is
# nocturnal, so the per-cell peak grows only modestly with the window).
WINDOWS = [("glm_1h", 1, 500, "1 hour"),
           ("glm_3h", 3, 1000, "3 hours"),
           ("glm_24h", 24, 2000, "24 hours")]
DEFAULT_WINDOW = "glm_3h"
MAX_WIN = max(w[1] for w in WINDOWS)

# GLM region — tighter than the IR loop: focused on Colombia + Brazil (excludes most of
# Central America to the NW and the southern half of Argentina). extent = (lon0, lon1 °E,
# lat0, lat1); clon = region midpoint (°E); dlon/dlat = gridline spacing.
CFG = dict(extent=(277, 328, -37, 14), clon=302.5, figsize=(8.6, 8.9),
           dlon=10, dlat=10, scale="50m", borders=True, states="Brazil")


def _grid_axes(cfg):
    lo0, lo1, la0, la1 = cfg["extent"]
    lon_e = np.arange(lo0, lo1 + BIN, BIN)            # bin EDGES (°E)
    lat_e = np.arange(la0, la1 + BIN, BIN)
    return lon_e, lat_e


# ── GLM L2 fetch (reused pattern from colombia_satellite) ──────────────────────
def _hour_keys(hour: datetime) -> list[str]:
    """All GLM-L2-LCFA granule keys for one clock hour [hour, hour+1h)."""
    pre = f"GLM-L2-LCFA/{hour:%Y/%j/%H}/"
    try:
        xml = urllib.request.urlopen(
            f"{GLM_S3}/?list-type=2&prefix={pre}&max-keys=1000", timeout=30).read().decode()
    except Exception:                                 # noqa: BLE001
        return []
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def _download(key: str):
    try:
        return urllib.request.urlopen(f"{GLM_S3}/{key}", timeout=30).read()
    except Exception:                                 # noqa: BLE001
        return None


# Time budget: the Action job has timeout-minutes; a deep backfill (cache lost or
# schedule gap > KEEP_H) can't finish in one run. Fetching past the deadline would
# get the JOB killed — which skips the actions/cache post-step, loses every fetched
# hour, and deadlocks recovery. Instead we stop fetching in time, keep what's
# banked, and let successive hourly runs walk the backfill down.
_DEADLINE = time.monotonic() + float(os.environ.get("GLM_BUDGET_MIN", "11")) * 60


class _OutOfTime(Exception):
    pass


def _bin_hour(hour: datetime, cfg) -> np.ndarray:
    """Binned flash-count grid for one clock hour. Cached to data/glm_hourly/ (gitignored).
    HDF5 is not thread-safe, so download threaded but open the granules sequentially."""
    lon_e, lat_e = _grid_axes(cfg)
    cache = HOURLY_CACHE / f"{CACHE_PREFIX}_{hour:%Y%m%d%H}.npy"
    if cache.exists():
        return np.load(cache)
    if time.monotonic() > _DEADLINE:                  # cached hours above still flow
        raise _OutOfTime()
    grid = np.zeros((lat_e.size - 1, lon_e.size - 1), dtype=np.int32)
    keys = _hour_keys(hour)
    if keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            blobs = list(ex.map(_download, keys))
        lo0, lo1, la0, la1 = cfg["extent"]
        for b in blobs:
            if not b:
                continue
            try:
                with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
                    tf.write(b); tf.flush()
                    d = xr.open_dataset(tf.name)
                    la = d["flash_lat"].values; lo = d["flash_lon"].values % 360
                    d.close()
                m = (lo >= lo0) & (lo <= lo1) & (la >= la0) & (la <= la1)
                if m.any():
                    h, _, _ = np.histogram2d(la[m], lo[m], bins=[lat_e, lon_e])
                    grid += h.astype(np.int32)
            except Exception:                         # noqa: BLE001
                continue
    HOURLY_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, grid)                              # cache even an empty hour (no re-fetch)
    return grid


def trailing_density(frame_hour: datetime, cfg, win_h: int) -> np.ndarray:
    """Flash-count grid accumulated over the win_h complete hours before frame_hour."""
    lon_e, lat_e = _grid_axes(cfg)
    total = np.zeros((lat_e.size - 1, lon_e.size - 1), dtype=np.int32)
    for k in range(1, win_h + 1):
        total += _bin_hour(frame_hour - timedelta(hours=k), cfg)
    return total


# ── render ─────────────────────────────────────────────────────────────────────
def render_frame(frame_hour: datetime, out: Path, cfg, win_h: int, vmax: int) -> bool:
    dens = trailing_density(frame_hour, cfg, win_h)
    lon_e, lat_e = _grid_axes(cfg)
    lo0, lo1, la0, la1 = cfg["extent"]
    clon = cfg["clon"]; scale = cfg.get("scale", "50m")
    off_e = _offset(lon_e, clon)
    proj = ccrs.PlateCarree(central_longitude=clon)
    norm = LogNorm(vmin=1, vmax=vmax)                 # FIXED per window → frames comparable

    fig = plt.figure(figsize=cfg["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([_offset(lo0, clon), _offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#000000")
    total = int(dens.sum())
    if total:
        masked = np.ma.masked_where(dens == 0, dens)
        ax.pcolormesh(off_e, lat_e, masked, transform=proj, cmap="inferno",
                      norm=norm, shading="flat", zorder=2)
    # always draw the colour scale (even on a quiet hour) so the legend is stable
    sm = ScalarMappable(norm=norm, cmap="inferno"); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.04, aspect=42,
                      shrink=0.92, extend="max")
    cb.set_label("flashes per 0.1° cell", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.coastlines(linewidth=0.7, color="#7a7a7a", resolution=scale)
    if cfg.get("borders"):
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.5, edgecolor="#6f6f6f")
    if cfg.get("states"):
        geoms = _state_geoms(cfg["states"], scale)
        if geoms:
            ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="none",
                              edgecolor="#4a4a4a", linewidth=0.3, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.35, linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    lon_ticks = [((t + 180) % 360) - 180
                 for t in range(int(lo0), int(lo1) + 1) if t % cfg["dlon"] == 0]
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(range(int(la0), int(la1) + 1, cfg["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(f"GOES-East GLM lightning — flash density, trailing {win_h} h  ·  "
                 f"{frame_hour:%Y-%m-%d %HZ}  ·  {total:,} flashes", fontsize=9, loc="left")
    fig.savefig(out, dpi=92, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=KEEP_H, help="ensure frames for the last N hours")
    args = ap.parse_args(argv)
    cfg = CFG
    manifest_path = ANIM_ROOT / "glmden_manifest.json"
    # Frames sit at the top of each complete hour (newest = last full hour). All windows
    # share the same frame hours and the same hourly-grid cache.
    now_h = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cutoff = now_h - timedelta(hours=KEEP_H)
    regions = {}
    for region_id, win_h, vmax, label in WINDOWS:
        anim = ANIM_ROOT / region_id
        anim.mkdir(parents=True, exist_ok=True)
        # newest frame first: after an outage the current frames publish on the
        # first run and the tail backfills across later runs within the budget
        for h in range(0, min(args.hours, KEEP_H)):
            fh = now_h - timedelta(hours=h)
            fp = anim / f"{fh:%Y%m%d%H}.webp"
            if time.monotonic() > _DEADLINE:          # rendering costs wall time too —
                print(f"  {region_id}: budget reached before {fh:%Y-%m-%d %HZ}; "
                      "next run continues", flush=True)
                break                                 # a deep backfill is 100s of maps
            try:
                if not fp.exists() and render_frame(fh, fp, cfg, win_h, vmax):
                    print(f"  {region_id}: rendered {fh:%Y-%m-%d %HZ}", flush=True)
            except _OutOfTime:
                print(f"  {region_id}: time budget reached at {fh:%Y-%m-%d %HZ} — "
                      "banking fetched hours; next run continues", flush=True)
                break
        entries = []
        for fp in sorted(anim.glob("*.webp")):
            try:
                dt = datetime.strptime(fp.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if dt < cutoff:
                fp.unlink(); continue
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%-d %b %HZ")})
        regions[region_id] = {"label": f"{label} accumulation", "frames": entries}
    # prune hourly cache beyond the rolling window + the longest accumulation window
    if HOURLY_CACHE.exists():
        for cf in HOURLY_CACHE.glob(f"{CACHE_PREFIX}_*.npy"):
            try:
                ch = datetime.strptime(cf.stem.split("_")[-1], "%Y%m%d%H").replace(tzinfo=timezone.utc)
                if ch < cutoff - timedelta(hours=MAX_WIN):
                    cf.unlink()
            except ValueError:
                continue
    manifest_path.write_text(json.dumps({"ver": now_h.strftime("%Y%m%d%H"),
                                         "selectorLabel": "Accumulation",
                                         "default": DEFAULT_WINDOW, "regions": regions}))
    n = len(regions[DEFAULT_WINDOW]["frames"])
    print(f"wrote {n} frames × {len(WINDOWS)} windows + {manifest_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
