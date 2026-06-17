#!/usr/bin/env python3
"""High-res true-color (visible) satellite loop over Colombia / Panama — GOES-East via NASA GIBS.

NASA's GIBS serves the GOES-East ABI GeoColor product as georeferenced imagery you can request for
an exact bounding box and size; in daylight GeoColor *is* true-colour visible. This crops the
Colombia/Panama region (near GOES-East's sub-satellite point, so about as sharp as geostationary
gets), overlays crisp country borders + coastlines, and keeps a rolling 2-day loop of the *sunlit*
frames (visible is dark at night, so night steps are skipped).

    python scripts/sst/colombia_satellite.py --hours 48        # backfill/refresh the 2-day loop
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
BASIN_GEOJSON = HERE / "metar" / "magdalena_basin.geojson"   # committed (so the Action has it too)

# major cities (lat, lon) + Lake Gatún, labelled for orientation
CITIES = {
    "Bogotá": (4.61, -74.08), "Medellín": (6.25, -75.57), "Cali": (3.44, -76.52),
    "Barranquilla": (10.96, -74.80), "Cartagena": (10.42, -75.51), "Bucaramanga": (7.12, -73.12),
    "Panama City": (8.98, -79.52),
}
LAKES = {"Lake Gatún": (9.18, -79.92)}

GLM_S3 = "https://noaa-goes19.s3.amazonaws.com"   # GOES-East GLM lightning (anonymous S3)
GLM_WINDOW_MIN = 15                                # accumulate flashes over this trailing window


def _glm_keys(dt: datetime) -> list[str]:
    """GLM-L2-LCFA granule keys whose 20-s window falls in [dt − GLM_WINDOW_MIN, dt]."""
    keys, lo = [], dt - timedelta(minutes=GLM_WINDOW_MIN)
    hours = {lo.replace(minute=0, second=0, microsecond=0), dt.replace(minute=0, second=0, microsecond=0)}
    for h in hours:
        pre = f"GLM-L2-LCFA/{h:%Y/%j/%H}/"
        try:
            xml = urllib.request.urlopen(f"{GLM_S3}/?list-type=2&prefix={pre}&max-keys=400", timeout=30).read().decode()
        except Exception:                                       # noqa: BLE001
            continue
        for k in re.findall(r"<Key>([^<]+)</Key>", xml):
            m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", k)
            if not m:
                continue
            yr, jd, hh, mm, ss = map(int, m.groups())
            t = datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=jd - 1, hours=hh, minutes=mm, seconds=ss)
            if lo <= t <= dt:
                keys.append(k)
    return keys


def _glm_download(key: str):
    try:
        return urllib.request.urlopen(f"{GLM_S3}/{key}", timeout=30).read()
    except Exception:                                           # noqa: BLE001
        return None


def fetch_glm_flashes(dt: datetime):
    """All GLM flash (lat, lon) inside the box over the trailing GLM_WINDOW_MIN. Fails soft → empty.
    Downloads run threaded (I/O), but the NetCDF granules are opened *sequentially* — HDF5 is not
    thread-safe and opening it from multiple threads segfaults."""
    keys = _glm_keys(dt)
    if not keys:
        return np.array([]), np.array([])
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        blobs = list(ex.map(_glm_download, keys))               # threaded download only
    las, los = [], []
    for b in blobs:
        if not b:
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
                tf.write(b); tf.flush()
                d = xr.open_dataset(tf.name)                    # opened one at a time, main thread
                la, lo = d["flash_lat"].values, d["flash_lon"].values
                d.close()
            m = (lo >= W) & (lo <= E) & (la >= S) & (la <= N)
            las.append(la[m]); los.append(lo[m])
        except Exception:                                       # noqa: BLE001
            continue
    return (np.concatenate(las) if las else np.array([]),
            np.concatenate(los) if los else np.array([]))
ANIM = HERE.parent.parent / "assets" / "sst" / "anim" / "colombia"
MANIFEST = HERE.parent.parent / "assets" / "sst" / "anim" / "colombia_manifest.json"
GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
LAYER = "GOES-East_ABI_GeoColor"
W, E, S, N = -83.0, -69.0, -2.0, 13.0           # Colombia/Panama bounding box (°)
WPX = 1680                                       # ~1 km native at this latitude (lon span 14°)
HPX = int(round(WPX * (N - S) / (E - W)))        # keep square degrees
KEEP_H = 48                                      # rolling 2-day window
STEP_MIN = 60                                    # frame cadence (minutes)
SUN_MIN_ELEV = 6.0                               # only render frames with the sun this far up
BORDER = "#ffd400"                               # crisp yellow borders/coastlines


def solar_elev(lat: float, lon: float, dt: datetime) -> float:
    """Approximate solar elevation (deg) at (lat, lon) for a UTC datetime."""
    doy = dt.timetuple().tm_yday
    decl = -23.44 * math.cos(math.radians(360.0 / 365.0 * (doy + 10)))
    hour = dt.hour + dt.minute / 60.0
    H = 15.0 * (hour + lon / 15.0 - 12.0)        # solar hour angle (lon west = negative)
    la, de, Ha = map(math.radians, (lat, decl, H))
    return math.degrees(math.asin(math.sin(la) * math.sin(de) + math.cos(la) * math.cos(de) * math.cos(Ha)))


def fetch_gibs(dt: datetime) -> np.ndarray | None:
    """GOES-East GeoColor over the box for `dt` → RGB array, or None if blank/unavailable."""
    q = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": LAYER,
         "CRS": "EPSG:4326", "BBOX": f"{S},{W},{N},{E}", "WIDTH": WPX, "HEIGHT": HPX,
         "FORMAT": "image/png", "TIME": dt.strftime("%Y-%m-%dT%H:%M:00Z")}
    try:
        data = urllib.request.urlopen(f"{GIBS}?{urllib.parse.urlencode(q)}", timeout=120).read()
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:                                       # noqa: BLE001
        return None
    return arr if arr.mean() > 6 else None                  # all-black → no imagery for this step


def render_frame(dt: datetime, arr: np.ndarray, out: Path) -> None:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(WPX / 150, HPX / 150))
    ax = plt.axes([0, 0, 1, 1], projection=proj)
    ax.set_extent([W, E, S, N], crs=proj)
    ax.imshow(arr, origin="upper", extent=[W, E, S, N], transform=proj, interpolation="nearest")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor=BORDER, linewidth=1.1)
    ax.coastlines("10m", color=BORDER, linewidth=0.8)
    # Magdalena River Basin outline (cyan dashed) if a polygon has been sourced
    if BASIN_GEOJSON.exists():
        try:
            from shapely.geometry import shape
            gj = json.loads(BASIN_GEOJSON.read_text())
            feats = gj["features"] if "features" in gj else [gj]
            geoms = [shape(f.get("geometry", f)) for f in feats]
            ax.add_geometries(geoms, crs=proj, facecolor="none", edgecolor="#19e0ff",
                              linewidth=1.5, linestyle=(0, (6, 3)), zorder=6)
            ax.text(-74.3, 8.6, "Magdalena\nRiver Basin", transform=proj, fontsize=8.5, color="#aef0ff",
                    va="center", ha="center", zorder=8, fontstyle="italic",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        except Exception:                                       # noqa: BLE001
            pass
    stroke = [pe.withStroke(linewidth=2.4, foreground="black")]
    for nm, (la, lo) in CITIES.items():
        ax.plot(lo, la, marker="o", ms=4, mfc="white", mec="black", mew=0.7, transform=proj, zorder=8)
        ax.text(lo + 0.1, la, nm, transform=proj, fontsize=9, color="white", va="center", ha="left",
                zorder=8, path_effects=stroke)
    for nm, (la, lo) in LAKES.items():
        ax.plot(lo, la, marker="s", ms=4, mfc="#8fd3ff", mec="black", mew=0.6, transform=proj, zorder=8)
        ax.text(lo, la - 0.2, nm, transform=proj, fontsize=8.5, color="#bfe9ff", va="top", ha="center",
                zorder=8, fontstyle="italic", path_effects=stroke)
    # GLM lightning flashes over the trailing window (bright cyan +)
    fla, flo = fetch_glm_flashes(dt)
    if len(fla):
        ax.scatter(flo, fla, s=13, marker="+", c="#41ffe6", linewidths=0.6, alpha=0.8,
                   transform=proj, zorder=9)
    glm = f"  ·  GLM lightning (cyan, last {GLM_WINDOW_MIN} min)" if len(fla) else ""
    ax.text(0.008, 0.992, f"GOES-East true colour  ·  {dt:%Y-%m-%d %H:%MZ}{glm}", transform=ax.transAxes,
            va="top", ha="left", color="white", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.45))
    fig.savefig(out, dpi=150); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=KEEP_H, help="render sunlit frames over the last N hours")
    args = ap.parse_args(argv)
    ANIM.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    latc, lonc = (S + N) / 2, (W + E) / 2
    n = min(args.hours, KEEP_H)
    steps = int(n * 60 / STEP_MIN)
    for k in range(steps, -1, -1):                          # oldest → newest
        dt = now - timedelta(minutes=k * STEP_MIN)
        if solar_elev(latc, lonc, dt) < SUN_MIN_ELEV:       # night → skip (visible is dark)
            continue
        fp = ANIM / f"{dt:%Y%m%d%H%M}.webp"
        if fp.exists():
            continue
        arr = fetch_gibs(dt)
        if arr is not None:
            render_frame(dt, arr, fp)
            print(f"  rendered {dt:%Y-%m-%d %H:%MZ}", flush=True)
    # trim to the rolling window + rebuild manifest
    cutoff = now - timedelta(hours=KEEP_H)
    entries = []
    for fp in sorted(ANIM.glob("*.webp")):
        try:
            dt = datetime.strptime(fp.stem, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if dt < cutoff:
            fp.unlink(); continue
        entries.append({"idx": len(entries), "file": fp.name,
                        "date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%-d %b %H:%MZ")})
    MANIFEST.write_text(json.dumps({"ver": now.strftime("%Y%m%d%H%M"),
                                    "regions": {"colombia": {"label": "Colombia / Panama true colour",
                                                             "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
