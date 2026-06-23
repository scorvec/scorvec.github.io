#!/usr/bin/env python3
"""Seamless georeferenced IR satellite loops from the GMGSI global mosaic.

NOAA already blends GOES-E/W + Himawari + Meteosat into the GMGSI global mosaic on a
regular lat/lon grid, so there is no stitching, no seam and no satellite-switch parallax
to fix — we just crop a region and render it georeferenced (cartopy, with coastlines for
orientation). The McIDAS byte → brightness-temperature calibration is shown with an
enhanced-IR colortable (cold cloud tops = colour/white = deep convection). Each hourly
frame is named by timestamp; the rolling last KEEP_H hours feed the sst_anim.html iframe.

Region presets (--region): "pacsat" tropical Pacific (default), "samsat" South & Central
America. Each writes its own anim/<region>/ frames + anim/<region>_manifest.json.

    python scripts/sst/pacific_satellite.py                 # tropical Pacific (default)
    python scripts/sst/pacific_satellite.py --region samsat # South & Central America
    python scripts/sst/pacific_satellite.py --hours 24      # render last N hours + manifest
"""
from __future__ import annotations

import argparse
import json
import re
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# Enhanced-IR colortable in the style of tropicaltidbits.com: warm scene (ocean → low/mid
# cloud) is plain grayscale up to ~ -33 °C, then deep-convective cold tops switch into a
# blue→green→yellow→orange→red→magenta rainbow, coldest overshooting tops back to white.
# Nodes are (position, color) with position = (Tb − 180 K) / 120 K, so 0 = 180 K, 1 = 300 K.
IR_NODES = [
    (0.000, "#ffffff"),   # ~180 K  coldest overshoot
    (0.085, "#7a0030"),   # ~190 K  magenta/maroon
    (0.150, "#d11500"),   # ~198 K  red
    (0.217, "#f08a00"),   # ~206 K  orange
    (0.283, "#e6e60a"),   # ~214 K  yellow
    (0.350, "#19b34d"),   # ~222 K  green
    (0.417, "#1d6fd6"),   # ~230 K  blue
    (0.499, "#39c6ff"),   # ~240 K  light blue  (cold edge of the enhancement)
    (0.500, "#ffffff"),   # ~240 K  abrupt step to white = top of the grayscale ramp
    (0.608, "#9a9a9a"),   # ~253 K  mid grayscale
    (1.000, "#000000"),   # ~300 K  warm ocean = black
]
IR_CMAP = LinearSegmentedColormap.from_list("ir_enh", IR_NODES, N=256)
IR_NORM = Normalize(vmin=180, vmax=300)

HERE = Path(__file__).resolve().parent
ANIM_ROOT = HERE.parent.parent / "assets" / "sst" / "anim"
S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
KEEP_H = 72                            # rolling loop length (hours) — 3-day progression

# Region presets. extent = (lon0, lon1 in °E, lat0, lat1); clon = projection central
# longitude (°E) — pick the region midpoint so the crop stays contiguous (no dateline
# smear); dlon/dlat = gridline spacing; figsize ≈ the region aspect (+ title room);
# scale = Natural Earth resolution for coast/borders; borders = draw country boundaries;
# states = country name whose admin-1 (state/province) outlines to draw (or None).
REGIONS = {
    "pacsat": dict(extent=(100, 290, -40, 40), clon=180.0, figsize=(12.4, 5.6),
                   dlon=20, dlat=20, where="tropical Pacific",
                   label="Tropical Pacific IR (GMGSI)",
                   scale="110m", borders=False, states=None, dpi=92),
    "samsat": dict(extent=(245, 330, -58, 32), clon=287.5, figsize=(8.6, 9.2),
                   dlon=15, dlat=15, where="South & Central America",
                   label="South & Central America IR (GMGSI)",
                   scale="50m", borders=True, states="Brazil", dpi=140),
}


def _offset(lon, clon):
    """Longitude (°E) → signed offset (−180,180] from the projection's central meridian,
    so a region straddling the dateline stays contiguous in plot coordinates."""
    return ((np.asarray(lon) - clon + 180) % 360) - 180


_STATE_GEOMS: dict = {}      # (country, scale) -> list of admin-1 geometries (cached across frames)


def _state_geoms(country: str, scale: str):
    """Admin-1 (state/province) geometries for one country from Natural Earth, cached.
    Degrades to [] if the dataset can't be fetched, so a frame still renders."""
    key = (country, scale)
    if key not in _STATE_GEOMS:
        try:
            shp = shpreader.natural_earth(resolution=scale, category="cultural",
                                          name="admin_1_states_provinces")
            _STATE_GEOMS[key] = [rec.geometry for rec in shpreader.Reader(shp).records()
                                 if rec.attributes.get("admin") == country]
        except Exception as e:                       # noqa: BLE001 — NE download hiccup
            print(f"  (admin-1 outlines for {country} unavailable: {e})", flush=True)
            _STATE_GEOMS[key] = []
    return _STATE_GEOMS[key]


def gmgsi_tb(dt: datetime):
    """GMGSI LW for the given hour → (brightness-temperature K, lat1d, lon1d 0..360)."""
    pre = f"GMGSI_LW/{dt:%Y/%m/%d/%H}/"
    try:
        xml = urllib.request.urlopen(f"{S3}/?list-type=2&prefix={pre}&max-keys=5", timeout=30).read().decode()
    except Exception:
        return None
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    if not keys:
        return None
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(urllib.request.urlopen(f"{S3}/{keys[0]}", timeout=90).read()); tf.flush()
        d = xr.open_dataset(tf.name)
        b = d["data"].isel(time=0).values.astype("float32")
        lat = d["lat"].isel(xc=0).values
        lon = d["lon"].isel(yc=d.sizes["yc"] // 2).values % 360
    Tb = np.where(b > 176, 418.0 - b, (660.0 - b) / 2.0)     # McIDAS IR byte → K
    return Tb, lat, lon


def render_frame(dt: datetime, out: Path, cfg: dict) -> bool:
    r = gmgsi_tb(dt)
    if r is None:
        return False
    Tb, lat, lon = r
    lo0, lo1, la0, la1 = cfg["extent"]
    clon = cfg["clon"]
    ry = np.where((lat >= la0) & (lat <= la1))[0]
    # Work in offset-from-central-meridian space so any region (incl. dateline-straddling
    # tropical Pacific) is a single contiguous strip; o0<0<o1 for a region centred on clon.
    o0, o1 = _offset(lo0, clon), _offset(lo1, clon)
    off = _offset(lon, clon)
    order = np.argsort(off); off_s = off[order]; m = (off_s >= o0) & (off_s <= o1); cx = order[m]
    sub = Tb[np.ix_(ry, cx)]; latv = lat[ry]; offv = off_s[m]
    if latv[0] > latv[-1]:
        latv = latv[::-1]; sub = sub[::-1]
    proj = ccrs.PlateCarree(central_longitude=clon)
    fig = plt.figure(figsize=cfg["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([o0, o1, la0, la1], crs=proj)
    ax.pcolormesh(offv, latv, sub, transform=proj, cmap=IR_CMAP, norm=IR_NORM,
                  shading="auto", rasterized=True)
    scale = cfg.get("scale", "110m")
    ax.coastlines(linewidth=0.7, color="#cfcfcf", resolution=scale)
    if cfg.get("borders"):     # international boundaries
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.5,
                       edgecolor="#cfcfcf")
    if cfg.get("states"):      # admin-1 (state/province) outlines for one country, thinner
        geoms = _state_geoms(cfg["states"], scale)
        if geoms:
            ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="none",
                              edgecolor="#9a9a9a", linewidth=0.3, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.4, linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    # Gridline locators take TRUE longitudes (−180..180); build them across the region in °E.
    lon_ticks = [((t + 180) % 360) - 180
                 for t in range(int(lo0), int(lo1) + 1) if t % cfg["dlon"] == 0]
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(range(int(la0), int(la1) + 1, cfg["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(f"GMGSI enhanced IR — {cfg['where']}  ·  {dt:%Y-%m-%d %HZ}  ·  "
                 "colour = deep convection (cold tops)", fontsize=9, loc="left")
    fig.savefig(out, dpi=cfg.get("dpi", 92), bbox_inches="tight",
                pil_kwargs={"quality": 78, "method": 6}); plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=sorted(REGIONS), default="pacsat",
                    help="region preset (default: pacsat = tropical Pacific)")
    ap.add_argument("--hours", type=int, default=KEEP_H, help="ensure frames for the last N hours")
    args = ap.parse_args(argv)
    cfg = REGIONS[args.region]
    anim = ANIM_ROOT / args.region
    manifest = ANIM_ROOT / f"{args.region}_manifest.json"
    anim.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for h in range(min(args.hours, KEEP_H) - 1, -1, -1):       # render only missing frames
        dt = now - timedelta(hours=h)
        fp = anim / f"{dt:%Y%m%d%H}.webp"
        if not fp.exists() and render_frame(dt, fp, cfg):
            print(f"  rendered {dt:%Y-%m-%d %HZ}", flush=True)
    # trim frames older than the rolling window + build the manifest from what remains
    cutoff = now - timedelta(hours=KEEP_H)
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
    manifest.write_text(json.dumps({"ver": now.strftime("%Y%m%d%H"),
                                    "regions": {args.region: {"label": cfg["label"],
                                                              "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {manifest.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
