#!/usr/bin/env python3
"""
Equatorial Pacific SST at kilometre scale — NASA JPL MUR v4.1 (GHRSST L4).

Two daily images for the El Niño monitor:
  assets/sst/mur_eqpac.webp      — 10°S–10°N, 160°E–80°W at 0.05° (strided):
                                   the whole cold tongue / warm pool stage
  assets/sst/mur_coldtongue.webp — 5°S–5°N, 145°W–85°W at native 0.01°:
                                   tropical instability wave cusps, front
                                   filaments, and the Galápagos wake

Source: NOAA CoastWatch ERDDAP (jplMURSST41) — credential-free HTTPS subsets,
~2-day latency. The AWS Open Data zarr mirror is a frozen archive (ends 2020),
so ERDDAP is the near-real-time path. MUR is heavily interpolated where IR is
cloud-blocked; treat fine structure as analysis, not observation.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parent.parent)
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "mur"

# CoastWatch intermittently 403s cloud-provider IPs (GitHub runners); a real
# User-Agent and a mirror that also serves jplMURSST41 cover it.
ERDDAP_HOSTS = ["https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
                "https://upwell.pfeg.noaa.gov/erddap/griddap/jplMURSST41"]
ERDDAP = ERDDAP_HOSTS[0]
UA = {"User-Agent": "scorvec.com El Nino monitor (xarray/urllib; contact: site owner)"}


def _open(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=timeout)

WIDE = dict(lat=(-10, 10), stride=5)          # 0.05° effective
# Zoom crosses the antimeridian (an ERDDAP range cannot), so it fetches in two
# lon chunks: 170°E → the Peru coast.
ZOOM = dict(lat=(-8, 8), lon_chunks=[(170, 179.99), (-179.99, -79)])
ZOOM_EXTENT = (170, 281, -8, 8)
ANIM_DIR = "mur_ct"                           # assets/sst/anim/<region>/
ANIM_STRIDE = 4                               # 0.04° — plenty at animation size
ANIM_START = "2026-04-01"                     # pre-onset context; RONI ≥ +0.5 from 2026-05-21
# Colour range adapts to each day's field (0.5–99.5th percentile ± 0.5 °C,
# rounded to 0.5): a fixed range wastes the palette when the cold tongue runs
# warm (El Niño) and hides the TIW / front detail this product exists to show.
def _auto_range(field):
    lo = float(np.floor((np.nanpercentile(field, 0.5) - 0.5) * 2) / 2)
    hi = float(np.ceil((np.nanpercentile(field, 99.5) + 0.5) * 2) / 2)
    return lo, max(hi, lo + 2.0)


def _fetch(suffix: str, dest: Path, tries: int = 4) -> Path:
    """Download an ERDDAP query, alternating mirrors between attempts."""
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last = None
    for attempt in range(1, tries + 1):
        url = ERDDAP_HOSTS[(attempt - 1) % len(ERDDAP_HOSTS)] + suffix
        try:
            with _open(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"  fetch attempt {attempt}/{tries} failed ({repr(e)[:70]})",
                  flush=True)
            tmp.unlink(missing_ok=True)
            time.sleep(20 * attempt)
    raise last


def latest_time(tries: int = 4) -> pd.Timestamp:
    last = None
    for attempt in range(1, tries + 1):
        host = ERDDAP_HOSTS[(attempt - 1) % len(ERDDAP_HOSTS)]
        try:
            with _open(f"{host}.das", timeout=60) as r:
                das = r.read().decode("utf-8", errors="replace")
            m = re.search(r'time \{[^}]*actual_range ([0-9.e+]+), ([0-9.e+]+)',
                          das, re.S)
            return pd.Timestamp(float(m.group(2)), unit="s")
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"  .das attempt {attempt}/{tries} failed ({repr(e)[:60]})",
                  flush=True)
            time.sleep(30 * attempt)
    raise last


def grab(t: pd.Timestamp, lat, lon, stride: int = 1, tag: str = "x") -> xr.DataArray:
    ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    s = f":{stride}:" if stride > 1 else ":"
    suffix = (f".nc?analysed_sst%5B({ts})%5D"
              f"%5B({lat[0]}){s}({lat[1]})%5D%5B({lon[0]}){s}({lon[1]})%5D")
    p = _fetch(suffix, DATA / f"mur_{tag}.nc")
    with xr.open_dataset(p) as ds:
        da = ds["analysed_sst"].squeeze("time", drop=True).load()
        if ds["analysed_sst"].attrs.get("units", "").lower().startswith("k"):
            da = da - 273.15                     # PO.DAAC serves Kelvin; CoastWatch, Celsius
    return da


def _style(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d4c8", zorder=3)
    ax.coastlines(resolution="10m", lw=0.5, color="#444", zorder=4)
    gl = ax.gridlines(draw_labels=True, lw=0.25, color="#999", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}


def fetch_zoom(t: pd.Timestamp, stride: int = 1, tag: str = "zoom") -> xr.DataArray:
    """The dateline-crossing zoom domain, pasted onto 0–360 longitudes."""
    parts = []
    for i, chunk in enumerate(ZOOM["lon_chunks"]):
        da = grab(t, ZOOM["lat"], chunk, stride=stride, tag=f"{tag}_{i}")
        parts.append(da.assign_coords(longitude=da["longitude"] % 360))
    return xr.concat(parts, dim="longitude").sortby("longitude")


def render(field: xr.DataArray, extent, title: str, out: Path,
           figsize, note: str, vrange=None, dpi=200):
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    _style(ax, extent)
    lon = field["longitude"].values
    vmin, vmax = vrange if vrange else _auto_range(field.values)
    mesh = ax.pcolormesh(lon, field["latitude"].values, field.values,
                         transform=ccrs.PlateCarree(), cmap="turbo",
                         vmin=vmin, vmax=vmax, rasterized=True)
    # 26/28/30 °C isotherms, contoured on a ~0.06°-coarsened field so the
    # lines follow the fronts instead of kilometre-scale pixel noise.
    dlat = abs(float(field["latitude"][1] - field["latitude"][0]))
    c = max(1, round(0.06 / dlat))
    sm = (field.coarsen(latitude=c, longitude=c, boundary="trim").mean()
          if c > 1 else field)
    cs = ax.contour(sm["longitude"].values, sm["latitude"].values, sm.values,
                    levels=[26, 28, 30], colors="#111111", linewidths=0.6,
                    alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
    ax.clabel(cs, fmt=lambda v: f"{v:.0f}°", fontsize=6, inline=True)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.012,
                      fraction=0.025, aspect=28)
    cb.set_label("SST (°C)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, loc="left", pad=4)
    # Anchor the source note just under the axes (not the figure bottom) so
    # bbox_inches="tight" can close up the whitespace band beneath the map.
    ax.text(0.0, -0.16, note, transform=ax.transAxes, fontsize=6.5,
            color="#666", va="top", ha="left")
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.05,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.name} ({out.stat().st_size/1e3:.0f} kB)", flush=True)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    t = latest_time()
    print(f"MUR latest analysis: {t:%Y-%m-%d}", flush=True)
    note = ("NASA JPL MUR SST v4.1 (GHRSST L4, ~1 km analysis) via NOAA CoastWatch "
            "ERDDAP · MUR interpolates under cloud — fine structure is analysis, "
            "not direct observation.")

    # Wide view: two lon chunks (ERDDAP ranges cannot cross the antimeridian).
    west = grab(t, WIDE["lat"], (160, 179.99), stride=WIDE["stride"], tag="wide_w")
    east = grab(t, WIDE["lat"], (-179.99, -80), stride=WIDE["stride"], tag="wide_e")
    east = east.assign_coords(longitude=east["longitude"] % 360)
    west = west.assign_coords(longitude=west["longitude"] % 360)
    wide = xr.concat([west, east], dim="longitude").sortby("longitude")
    render(wide, (160, 280, -10, 10),
           f"NASA JPL MUR SST v4.1 — 0.05° — {t:%Y-%m-%d}",
           ASSETS / "mur_eqpac.webp", figsize=(16, 3.6), note=note)

    # Cold-tongue zoom at native resolution, 170°E → Peru coast.
    zoom = fetch_zoom(t)
    render(zoom, ZOOM_EXTENT,
           f"NASA JPL MUR SST v4.1 — native 0.01° — {t:%Y-%m-%d}",
           ASSETS / "mur_coldtongue.webp", figsize=(22, 3.9), note=note)
    return 0


# ── Event animation: one frame per day since ANIM_START ─────────────────────
def anim(argv_start: str | None = None) -> int:
    """Backfill/append daily frames of the zoom domain and write the manifest
    the site's animation player reads. Idempotent: existing frames are kept;
    only missing days are fetched (throttled — ERDDAP 503s under rapid fire).
    The colour range is fixed across the whole animation (stored in the
    manifest) so frames don't flicker as the field warms."""
    import json
    frames_dir = ASSETS / "anim" / ANIM_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    man_path = ASSETS / "anim" / "mur_manifest.json"
    note = ("NASA JPL MUR SST v4.1 (~1 km analysis, 0.04° frames) via NOAA "
            "CoastWatch ERDDAP · fixed colour range across the event")

    latest = latest_time()
    start = pd.Timestamp(argv_start or ANIM_START)
    days = pd.date_range(start, latest.normalize(), freq="D")

    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    vrange = man.get("vrange")
    if not vrange:
        # Anchor the fixed range on both ends of the event: coldest early
        # field and the current warm field.
        a = fetch_zoom(days[0] + pd.Timedelta(hours=9), stride=ANIM_STRIDE, tag="vr0")
        b = fetch_zoom(latest, stride=ANIM_STRIDE, tag="vr1")
        lo = min(_auto_range(a.values)[0], _auto_range(b.values)[0])
        hi = max(_auto_range(a.values)[1], _auto_range(b.values)[1])
        vrange = [lo, hi]
        print(f"  fixed animation range: {lo}–{hi} °C", flush=True)

    made = 0
    for d in days:
        out = frames_dir / f"{d:%Y%m%d}.webp"
        if out.exists():
            continue
        try:
            f = fetch_zoom(d + pd.Timedelta(hours=9), stride=ANIM_STRIDE,
                           tag="animday")
        except Exception as e:                                  # noqa: BLE001
            print(f"  {d:%Y-%m-%d}: fetch failed ({repr(e)[:60]}); skipping",
                  flush=True)
            continue
        render(f, ZOOM_EXTENT, f"NASA JPL MUR SST v4.1 — 0.04° — {d:%Y-%m-%d}", out,
               figsize=(22, 3.9), note=note, vrange=tuple(vrange), dpi=110)
        made += 1
        time.sleep(2)                       # be a polite ERDDAP citizen

    frames = sorted(frames_dir.glob("*.webp"))
    entries = [{"idx": i, "file": p.name,
                "date": f"{p.stem[:4]}-{p.stem[4:6]}-{p.stem[6:8]}",
                "label": pd.Timestamp(p.stem).strftime("%d %b")}
               for i, p in enumerate(frames)]
    man = {"ver": pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H"),
           "vrange": vrange,
           "regions": {ANIM_DIR: {
               "label": "MUR 1 km — cold tongue since event onset",
               "frames": entries}}}
    man_path.write_text(json.dumps(man))
    print(f"  animation: {len(entries)} frames ({made} new) → {man_path.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    if "--anim" in sys.argv:
        s = None
        if "--start" in sys.argv:
            s = sys.argv[sys.argv.index("--start") + 1]
        sys.exit(anim(s))
    sys.exit(main())
