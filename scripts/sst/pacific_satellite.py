#!/usr/bin/env python3
"""Seamless tropical-Pacific IR satellite loop from the GMGSI global mosaic.

NOAA already blends GOES-E/W + Himawari + Meteosat into the GMGSI global mosaic on a
regular lat/lon grid, so there is no stitching, no seam and no satellite-switch parallax
to fix — we just crop the tropical Pacific and render it georeferenced (cartopy, with
coastlines for orientation). The McIDAS byte → brightness-temperature calibration is
shown as a grayscale IR (cold cloud tops = white = deep convection). Each hourly frame is
named by timestamp; the rolling last KEEP_H hours feed the sst_anim.html iframe.

    python scripts/sst/pacific_satellite.py --hours 24    # render last N hours + manifest
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
ANIM = HERE.parent.parent / "assets" / "sst" / "anim" / "pacsat"
MANIFEST = HERE.parent.parent / "assets" / "sst" / "anim" / "pacsat_manifest.json"
S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
EXTENT = (100, 290, -40, 40)          # lon0, lon1 (°E), lat0, lat1
KEEP_H = 72                            # rolling loop length (hours) — 3-day progression


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


def render_frame(dt: datetime, out: Path) -> bool:
    r = gmgsi_tb(dt)
    if r is None:
        return False
    Tb, lat, lon = r
    lo0, lo1, la0, la1 = EXTENT
    ry = np.where((lat >= la0) & (lat <= la1))[0]
    order = np.argsort(lon); lon_s = lon[order]; m = (lon_s >= lo0) & (lon_s <= lo1); cx = order[m]
    sub = Tb[np.ix_(ry, cx)]; latv = lat[ry]; lonv = lon_s[m]
    if latv[0] > latv[-1]:
        latv = latv[::-1]; sub = sub[::-1]
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(12.4, 5.6))
    ax = plt.axes(projection=proj)
    ax.set_extent([lo0 - 180, lo1 - 180, la0, la1], crs=proj)
    ax.pcolormesh(lonv - 180, latv, sub, transform=proj, cmap=IR_CMAP, norm=IR_NORM,
                  shading="auto", rasterized=True)
    ax.coastlines(linewidth=0.7, color="#cfcfcf")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.4, linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(range(-40, 41, 20))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(f"GMGSI enhanced IR — tropical Pacific  ·  {dt:%Y-%m-%d %HZ}  ·  "
                 "colour = deep convection (cold tops)", fontsize=9, loc="left")
    fig.savefig(out, dpi=92, bbox_inches="tight"); plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=KEEP_H, help="ensure frames for the last N hours")
    args = ap.parse_args(argv)
    ANIM.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for h in range(min(args.hours, KEEP_H) - 1, -1, -1):       # render only missing frames
        dt = now - timedelta(hours=h)
        fp = ANIM / f"{dt:%Y%m%d%H}.webp"
        if not fp.exists() and render_frame(dt, fp):
            print(f"  rendered {dt:%Y-%m-%d %HZ}", flush=True)
    # trim frames older than the rolling window + build the manifest from what remains
    cutoff = now - timedelta(hours=KEEP_H)
    entries = []
    for fp in sorted(ANIM.glob("*.webp")):
        try:
            dt = datetime.strptime(fp.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if dt < cutoff:
            fp.unlink(); continue
        entries.append({"idx": len(entries), "file": fp.name,
                        "date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%-d %b %HZ")})
    MANIFEST.write_text(json.dumps({"ver": now.strftime("%Y%m%d%H"),
                                    "regions": {"pacsat": {"label": "Tropical Pacific IR (GMGSI)",
                                                           "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
