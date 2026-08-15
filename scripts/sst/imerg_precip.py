#!/usr/bin/env python3
"""IMERG precipitation accumulation loops for Colombia & Brazil (1h / 24h / 7d / 14d).

NASA GPM IMERG (Early run, V07) via earthaccess (Earthdata login through ~/.netrc, or
EARTHDATA_USERNAME/PASSWORD in CI). Each accumulation window is a trailing total binned to
the IMERG 0.1° grid and looped over recent time — the rainfall companion to the IR cloud
loop and the GLM lightning loop, same Colombia+Brazil framing.

Data: short windows (1h, 24h) come from the HALF-HOURLY product (GPM_3IMERGHHE,
/Grid/precipitation in mm/hr → ×0.5 h per granule); long windows (7d, 14d) come from the
DAILY product (GPM_3IMERGDE, /precipitation already in mm/day) — far fewer files than
summing hundreds of half-hourly granules. Per-step binned grids are cached (gitignored) so
a run only fetches what's new; committed webp frames carry the rolling window.

    python scripts/sst/imerg_precip.py            # ensure recent frames for all windows
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import h5py

import map_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacific_satellite import _offset, _state_geoms      # shared geometry helpers

warnings.filterwarnings("ignore")                          # quiet earthaccess/h5py chatter

# earthaccess.download() issues requests with NO timeout; one stalled Earthdata
# connection blocked imerg_gatun for 18 h (2026-08-14/15) and its lock starved
# every SST poll behind it. requests/urllib3 fall back to the global socket
# default when unset, so this makes any silent stall raise instead of hanging.
import socket
socket.setdefaulttimeout(120)

HERE = Path(__file__).resolve().parent
ANIM_ROOT = HERE.parent.parent / "assets" / "sst" / "anim"
CACHE = HERE / "data" / "imerg"                            # binned grids + raw granules (gitignored)
HOURLY_CACHE = CACHE / "hourly"
DAILY_CACHE = CACHE / "daily"
GRANULES = CACHE / "granules"

# Colombia + Brazil — same framing as the GLM lightning loop.
CFG = dict(extent=(277, 328, -37, 14), clon=302.5, figsize=(8.0, 8.4),
           dlon=10, dlat=10, scale="50m", borders=True, states="Brazil")

# Accumulation windows: (region id, source kind, # units, frame step h, span h, label, levels mm)
WINDOWS = [
    dict(id="precip_1h",  kind="hourly", units=1,  step_h=1,  span_h=48,
         label="1 hour",   levels=[0.5, 1, 2, 4, 7, 10, 15, 20, 30, 45, 65]),
    dict(id="precip_24h", kind="hourly", units=24, step_h=3,  span_h=72,
         label="24 hours", levels=[1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200]),
    dict(id="precip_7d",  kind="daily",  units=7,  step_h=24, span_h=336,
         label="7 days",   levels=[2, 5, 10, 20, 40, 70, 100, 150, 250, 350, 500]),
    dict(id="precip_14d", kind="daily",  units=14, step_h=24, span_h=504,
         label="14 days",  levels=[5, 10, 20, 40, 70, 100, 175, 275, 400, 550, 750]),
    # month-to-date: variable length (1st of month → each frame day), daily-based, short loop
    dict(id="precip_mtd", kind="mtd",    units=31, step_h=24, span_h=240,
         label="month-to-date", levels=[2, 5, 10, 25, 50, 100, 175, 275, 400, 550, 750]),
]
DEFAULT_WINDOW = "precip_24h"


def _units(w: dict, ft: datetime, step: timedelta) -> int:
    """Trailing step count for a frame. Fixed for normal windows; for month-to-date it's the
    day-of-month of the last included day (ft − step), i.e. how many days since the 1st."""
    return (ft - step).day if w["kind"] == "mtd" else w["units"]
# Per-cadence cache retention = each cadence's deepest reach (span + window) + a small buffer,
# so hourly grids don't pile up to the 14-day window's depth.
HOURLY_KEEP_H = max(w["span_h"] + w["units"] for w in WINDOWS if w["kind"] == "hourly") + 6
# daily cache must cover the deepest reach of ANY consumer: the anomaly product
# (30-day window + ~10-day loop), the Gatun tracker's 90-day window + loop
# (imerg_gatun.py), and the Colombia hydro validation's multi-year record
# (imerg_backfill.py / validate_region_rain.py) — hence 800 days (~1 MB/day).
DAILY_KEEP_H = max((max(w["span_h"] // 24 + w["units"] for w in WINDOWS if w["kind"] == "daily") + 2) * 24,
                   800 * 24)

# precip palette: deep blue → blue → teal → green → yellow → orange → red → magenta → white
_STOPS = ["#2b3a6b", "#3b76c4", "#3fb0b0", "#52c452", "#c8d63f",
          "#f0a800", "#e2502a", "#b3247a", "#ffffff"]
PRECIP_CMAP = LinearSegmentedColormap.from_list("precip", _STOPS)

# IMERG V07 is a fixed 0.1° global grid (lon 3600, lat 1800) — define the axes analytically
# so the region subset works even on cached re-runs that read no granule.
_LON = -179.95 + 0.1 * np.arange(3600)
_LAT = -89.95 + 0.1 * np.arange(1800)
_AUTH = False
_BASIN = None
BASIN_GEOJSON = HERE / "metar" / "magdalena_basin.geojson"   # Colombia, committed (used by colombia_satellite)


def _magdalena_geom():
    """Magdalena River basin outline (Colombia) from the committed geojson, cached."""
    global _BASIN
    if _BASIN is None:
        from shapely.geometry import shape
        if BASIN_GEOJSON.exists():
            gj = json.loads(BASIN_GEOJSON.read_text())
            feats = gj.get("features", [gj])
            _BASIN = [shape(f.get("geometry", f)) for f in feats]
        else:
            _BASIN = []
    return _BASIN


def _login():
    global _AUTH
    if not _AUTH:
        import earthaccess
        strategy = "environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"
        earthaccess.login(strategy=strategy)
        _AUTH = True


def _search(short_name: str, **kw):
    """earthaccess CMR search with retries — CMR returns transient 500s under load."""
    import earthaccess
    import time
    _login()
    for attempt in range(5):
        try:
            return earthaccess.search_data(short_name=short_name, version="07", **kw)
        except Exception as e:                             # noqa: BLE001 — CMR 500 / network
            if attempt == 4:
                print(f"  search {short_name} failed after retries: {repr(e)[:80]}", flush=True)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def _grid_axes():
    lo0, lo1, la0, la1 = CFG["extent"]                     # extent in 0–360°E; IMERG lon is −180..180
    ml = (_LON % 360 >= lo0) & (_LON % 360 <= lo1)
    mt = (_LAT >= la0) & (_LAT <= la1)
    return ml, mt


def _read_subset(path: Path, var: str) -> np.ndarray:
    """Region-subset precip grid (lat, lon) from an IMERG HDF5/nc4 granule (fill→0)."""
    with h5py.File(path, "r") as f:
        arr = f[var][0].astype("float32")                  # (lon, lat)
    ml, mt = _grid_axes()
    return np.where(arr < 0, 0.0, arr)[np.ix_(ml, mt)].T   # (lat, lon)


# ── fetch + cache per-step grids ───────────────────────────────────────────────
def _granule_time(url: str) -> datetime:
    m = re.search(r"\.3IMERG\.(\d{8})-S(\d{6})", url)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def ensure_hourly(hours: set[datetime]):
    """Bin half-hourly granules → mm grids for each needed clock hour (cache to HOURLY_CACHE)."""
    need = sorted(h for h in hours if not (HOURLY_CACHE / f"{h:%Y%m%d%H}.npy").exists())
    if not need:
        return
    import earthaccess
    _login()
    GRANULES.mkdir(parents=True, exist_ok=True); HOURLY_CACHE.mkdir(parents=True, exist_ok=True)
    res = _search("GPM_3IMERGHHE",
            temporal=(f"{need[0]:%Y-%m-%d %H:%M:%S}", f"{need[-1] + timedelta(hours=1):%Y-%m-%d %H:%M:%S}"))
    by_hr: dict[datetime, list] = {}
    for r in res:
        t = _granule_time(r.data_links()[0]).replace(minute=0, second=0)
        by_hr.setdefault(t, []).append(r)
    for h in need:
        gs = by_hr.get(h, [])
        if len(gs) < 2:                                    # need both 30-min granules for a full hour
            continue
        files = earthaccess.download(gs, str(GRANULES))
        acc = None
        for fp in files:
            g = _read_subset(Path(fp), "/Grid/precipitation") * 0.5   # mm/hr × 0.5 h
            acc = g if acc is None else acc + g
        np.save(HOURLY_CACHE / f"{h:%Y%m%d%H}.npy", acc.astype("float32"))
        for fp in files:                                   # raw global granule (~10 MB) is transient
            Path(fp).unlink(missing_ok=True)


def ensure_daily(days: set[datetime]):
    """Save daily-total mm grids for each needed UTC day (cache to DAILY_CACHE)."""
    need = sorted(d for d in days if not (DAILY_CACHE / f"{d:%Y%m%d}.npy").exists())
    if not need:
        return
    import earthaccess
    _login()
    GRANULES.mkdir(parents=True, exist_ok=True); DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    res = _search("GPM_3IMERGDE",
            temporal=(f"{need[0]:%Y-%m-%d}", f"{need[-1]:%Y-%m-%d}"))
    by_day = {_granule_time(r.data_links()[0]).date(): r for r in res}
    for d in need:
        r = by_day.get(d.date())
        if r is None:
            continue
        fp = earthaccess.download([r], str(GRANULES))[0]
        g = _read_subset(Path(fp), "/precipitation")        # already mm/day
        np.save(DAILY_CACHE / f"{d:%Y%m%d}.npy", g.astype("float32"))
        Path(fp).unlink(missing_ok=True)                    # raw global granule is transient


def _load(cache: Path, key: str):
    p = cache / f"{key}.npy"
    return np.load(p) if p.exists() else None


def trailing_sum(cache: Path, end: datetime, n: int, step: timedelta, fmt: str) -> np.ndarray | None:
    """Sum the n cached grids in (end-n·step, end]; None if too many are missing."""
    grids = [_load(cache, (end - k * step).strftime(fmt)) for k in range(1, n + 1)]
    have = [g for g in grids if g is not None]
    if len(have) < max(1, int(0.6 * n)):                   # tolerate a few gaps, not most
        return None
    return np.sum(have, axis=0)


# ── render ─────────────────────────────────────────────────────────────────────
def render_frame(field: np.ndarray, valid_label: str, out: Path, win: dict):
    lo0, lo1, la0, la1 = CFG["extent"]; clon = CFG["clon"]; scale = CFG["scale"]
    ml, mt = _grid_axes()
    LON, LAT = _LON[ml], _LAT[mt]
    off = _offset(LON, clon)
    levels = win["levels"]; norm = BoundaryNorm(levels, PRECIP_CMAP.N, extend="max")
    proj = ccrs.PlateCarree(central_longitude=clon)

    fig = plt.figure(figsize=CFG["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([_offset(lo0, clon), _offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#0d0d16")
    wet = np.ma.masked_less(field, levels[0])
    cf = ax.contourf(off, LAT, wet, levels=levels, cmap=PRECIP_CMAP, norm=norm,
                     extend="max", transform=proj)
    ax.coastlines(linewidth=0.7, color="#b6b6b6", resolution=scale)
    if CFG["borders"]:                                      # international boundaries
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.55, edgecolor="#c2c2c2")
    if CFG["states"]:                                       # Brazil state outlines (lighter than borders)
        geoms = _state_geoms(CFG["states"], scale)
        if geoms:
            ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="none",
                              edgecolor="#8f8f8f", linewidth=0.4, zorder=3)
    basin = _magdalena_geom()                              # Magdalena River basin (Colombia)
    if basin:                                              # white reads cleanly over the dark blues
        ax.add_geometries(basin, crs=ccrs.PlateCarree(), facecolor="none",
                          edgecolor="#ffffff", linewidth=1.1, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.35, linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(map_grid.lon_ticks(lo0, lo1, CFG["dlon"]))
    gl.ylocator = mticker.FixedLocator(map_grid.lat_ticks(la0, la1, CFG["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    map_grid.add_ref_lines(ax, (lo0, lo1, la0, la1))
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=42, extend="max")
    cb.set_label("accumulated precip (mm)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(f"IMERG Early — {win['label']} accumulated precipitation  ·  {valid_label}",
                 fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def _latest_anchors() -> tuple[datetime, datetime]:
    """The frame anchor for each cadence = the boundary just AFTER the latest COMPLETE unit
    (IMERG Early lags ~6 h, daily ~1 day), so the newest frame holds real data, not a gap."""
    import earthaccess
    _login()
    now = datetime.now(timezone.utc)
    res = _search("GPM_3IMERGHHE",
            temporal=(f"{now - timedelta(hours=18):%Y-%m-%d %H:%M:%S}", f"{now:%Y-%m-%d %H:%M:%S}"))
    if res:
        last = max(_granule_time(r.data_links()[0]) for r in res)        # latest 30-min start
        anchor_h = last.replace(minute=0) + (timedelta(hours=1) if last.minute == 30 else timedelta())
    else:
        anchor_h = now.replace(minute=0, second=0, microsecond=0)
    dres = _search("GPM_3IMERGDE",
            temporal=(f"{now - timedelta(days=4):%Y-%m-%d}", f"{now:%Y-%m-%d}"))
    if dres:
        ld = max(_granule_time(r.data_links()[0]).date() for r in dres)  # latest complete day
        anchor_day = datetime(ld.year, ld.month, ld.day, tzinfo=timezone.utc) + timedelta(days=1)
    else:
        anchor_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return anchor_h, anchor_day


def _frame_times(win: dict, anchor_h: datetime, anchor_day: datetime) -> list[datetime]:
    step = timedelta(hours=win["step_h"]); n = win["span_h"] // win["step_h"]
    base = anchor_h if win["kind"] == "hourly" else anchor_day   # daily + mtd sit on day boundaries
    return [base - k * step for k in range(n)][::-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-scale", type=float, default=1.0,
                    help="multiply each window's span (e.g. 0.25 to seed a short window)")
    args = ap.parse_args(argv)
    now_h = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    anchor_h, anchor_day = _latest_anchors()
    print(f"  anchors: hourly→{anchor_h:%Y-%m-%d %HZ}, daily→{anchor_day:%Y-%m-%d}", flush=True)
    regions = {}
    for win in WINDOWS:
        w = dict(win); w["span_h"] = max(w["step_h"], int(w["span_h"] * args.span_scale))
        cache = HOURLY_CACHE if w["kind"] == "hourly" else DAILY_CACHE
        step = timedelta(hours=1) if w["kind"] == "hourly" else timedelta(hours=24)
        fmt = "%Y%m%d%H" if w["kind"] == "hourly" else "%Y%m%d"
        ftimes = _frame_times(w, anchor_h, anchor_day)
        # ensure the grids every frame needs (its trailing unit count)
        need = set()
        for ft in ftimes:
            for k in range(1, _units(w, ft, step) + 1):
                need.add((ft - k * step).replace(minute=0, second=0, microsecond=0))
        (ensure_hourly if w["kind"] == "hourly" else ensure_daily)(need)

        anim = ANIM_ROOT / w["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            u = _units(w, ft, step)
            if not fp.exists():
                field = trailing_sum(cache, ft, u, step, fmt)
                if field is None:
                    continue
                if w["kind"] == "mtd":
                    e = ft - step                          # last included day
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                elif w["kind"] == "daily":
                    vlabel = (f"{(ft - timedelta(days=u)):%b %d} – {(ft - step):%b %d, %Y}"
                              f"  ({u}-day total)")
                else:
                    vlabel = f"ending {ft:%Y-%m-%d %HZ}"
                render_frame(field, vlabel, fp, w)
                print(f"  {w['id']}: rendered {tag}", flush=True)
            d = datetime.strptime(ft.strftime("%Y%m%d%H"), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            label = (f"MTD → {(d - step):%b %-d}" if w["kind"] == "mtd"
                     else f"{(d - timedelta(days=u)):%b %d} – {(d - step):%b %d}" if w["kind"] == "daily"
                     else f"{d:%-d %b %HZ}")
            entries.append({"idx": len(entries), "file": fp.name, "date": f"{d:%Y-%m-%d}", "label": label})
        # prune frames outside this window's span
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[w["id"]] = {"label": f"{w['label']} accumulation", "frames": entries}

    # prune stale grid caches (per cadence) + raw granules (by product)
    hourly_cut = now_h - timedelta(hours=HOURLY_KEEP_H)
    daily_cut = now_h - timedelta(hours=DAILY_KEEP_H)
    for cdir, fmt, cut in ((HOURLY_CACHE, "%Y%m%d%H", hourly_cut), (DAILY_CACHE, "%Y%m%d", daily_cut)):
        if cdir.exists():
            for cf in cdir.glob("*.npy"):
                try:
                    if datetime.strptime(cf.stem, fmt).replace(tzinfo=timezone.utc) < cut:
                        cf.unlink()
                except ValueError:
                    pass
    if GRANULES.exists():
        for gf in GRANULES.glob("*"):
            m = re.search(r"\.3IMERG\.(\d{8})-S(\d{6})", gf.name)
            if not m:
                continue
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if t < (daily_cut if "-DAY-" in gf.name else hourly_cut):
                gf.unlink()
    manifest = ANIM_ROOT / "precip_manifest.json"
    manifest.write_text(json.dumps({"ver": now_h.strftime("%Y%m%d%H"),
                                    "selectorLabel": "Accumulation",
                                    "default": DEFAULT_WINDOW, "regions": regions}))
    print(f"wrote {len(regions)} windows + {manifest.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
