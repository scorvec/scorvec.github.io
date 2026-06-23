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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacific_satellite import _offset, _state_geoms   # shared geometry helpers

HERE = Path(__file__).resolve().parent
ANIM_ROOT = HERE.parent.parent / "assets" / "sst" / "anim"
HOURLY_CACHE = HERE / "data" / "glm_hourly"           # per-clock-hour binned grids (gitignored)
GLM_S3 = "https://noaa-goes19.s3.amazonaws.com"       # GOES-East GLM L2 (anonymous S3)
BIN = 0.1                                             # density grid spacing (°)
WIN_H = 3                                             # trailing accumulation window (hours)
KEEP_H = 72                                           # rolling loop length (hours)
CACHE_PREFIX = "glmden"                               # hourly-grid cache filename prefix

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


def _bin_hour(hour: datetime, cfg) -> np.ndarray:
    """Binned flash-count grid for one clock hour. Cached to data/glm_hourly/ (gitignored).
    HDF5 is not thread-safe, so download threaded but open the granules sequentially."""
    lon_e, lat_e = _grid_axes(cfg)
    cache = HOURLY_CACHE / f"{CACHE_PREFIX}_{hour:%Y%m%d%H}.npy"
    if cache.exists():
        return np.load(cache)
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


def trailing_density(frame_hour: datetime, cfg) -> np.ndarray:
    """Flash-count grid accumulated over the WIN_H complete hours before frame_hour."""
    lon_e, lat_e = _grid_axes(cfg)
    total = np.zeros((lat_e.size - 1, lon_e.size - 1), dtype=np.int32)
    for k in range(1, WIN_H + 1):
        total += _bin_hour(frame_hour - timedelta(hours=k), cfg)
    return total


# ── render ─────────────────────────────────────────────────────────────────────
def render_frame(frame_hour: datetime, out: Path, cfg) -> bool:
    dens = trailing_density(frame_hour, cfg)
    lon_e, lat_e = _grid_axes(cfg)
    lo0, lo1, la0, la1 = cfg["extent"]
    clon = cfg["clon"]; scale = cfg.get("scale", "50m")
    off_e = _offset(lon_e, clon)
    proj = ccrs.PlateCarree(central_longitude=clon)

    fig = plt.figure(figsize=cfg["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([_offset(lo0, clon), _offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#000000")
    total = int(dens.sum())
    if total:
        masked = np.ma.masked_where(dens == 0, dens)
        ax.pcolormesh(off_e, lat_e, masked, transform=proj, cmap="inferno",
                      norm=LogNorm(vmin=1, vmax=max(10, dens.max())),
                      shading="flat", zorder=2)
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
    ax.set_title(f"GOES-East GLM lightning — flash density, trailing {WIN_H} h  ·  "
                 f"{frame_hour:%Y-%m-%d %HZ}  ·  {total:,} flashes", fontsize=9, loc="left")
    fig.savefig(out, dpi=92, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=KEEP_H, help="ensure frames for the last N hours")
    args = ap.parse_args(argv)
    cfg = CFG
    anim = ANIM_ROOT / "glmden"
    manifest = ANIM_ROOT / "glmden_manifest.json"
    anim.mkdir(parents=True, exist_ok=True)
    # Frames sit at the top of each complete hour (newest = last full hour).
    now_h = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for h in range(min(args.hours, KEEP_H) - 1, -1, -1):
        fh = now_h - timedelta(hours=h)
        fp = anim / f"{fh:%Y%m%d%H}.webp"
        if not fp.exists() and render_frame(fh, fp, cfg):
            print(f"  rendered {fh:%Y-%m-%d %HZ}", flush=True)
    # prune old frames + hourly cache, rebuild manifest
    cutoff = now_h - timedelta(hours=KEEP_H)
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
    if HOURLY_CACHE.exists():
        for cf in HOURLY_CACHE.glob(f"{CACHE_PREFIX}_*.npy"):
            try:
                ch = datetime.strptime(cf.stem.split("_")[-1], "%Y%m%d%H").replace(tzinfo=timezone.utc)
                if ch < cutoff - timedelta(hours=WIN_H):
                    cf.unlink()
            except ValueError:
                continue
    manifest.write_text(json.dumps({"ver": now_h.strftime("%Y%m%d%H"),
                                    "regions": {"glmden": {
                                        "label": "Lightning flash density (GLM, trailing 3 h)",
                                        "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {manifest.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
