#!/usr/bin/env python3
"""Detailed Colombia IMERG hydro chart set: country-scale rainfall loops with
hydro regions and hydroelectric plants overlaid.

Rides the imerg_precip data layer (same binned caches — the region grid covers
all of Colombia). Accumulation + anomaly loops at hydro-relevant windows, with:
  • hydro region polygons (scripts/sst/colombia_hydro_regions.geojson if
    present — e.g. IDEAM zonificación; falls back to the Magdalena outline)
  • hydro plants from colombia_hydro_plants.json, marker AREA ∝ capacity (MW)

Products (assets/sst/anim/):
  colombia_precip_{24h,5d,7d,14d,30d,mtd,90d}/ + colombia_precip_manifest.json
  colombia_anom_{1d,5d,7d,14d,30d,mtd,90d}/    + colombia_anom_manifest.json

    python scripts/sst/imerg_colombia.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg_precip as IP
import imerg_precip_anom as IPA
from build_imerg_clim import OUT as CLIM_NC

HERE = Path(__file__).resolve().parent
ANIM_ROOT = IP.ANIM_ROOT
PC = ccrs.PlateCarree()

# Full-country framing: 78.2°W–66.8°W, 4.6°S–12.8°N (°E 0–360)
CCFG = dict(extent=(281.8, 293.2, -4.6, 12.8), clon=287.5, figsize=(7.2, 9.6),
            dlon=2, dlat=2)
PLANTS = HERE / "colombia_hydro_plants.json"
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"

WINDOWS = [
    dict(id="colombia_precip_24h", kind="hourly", units=24, step_h=3, span_h=72,
         label="24 hours", levels=[1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200]),
    dict(id="colombia_precip_5d",  kind="daily", units=5,  step_h=24, span_h=240,
         label="5 days",  levels=[2, 5, 10, 20, 40, 70, 100, 150, 225, 325, 450]),
    dict(id="colombia_precip_7d",  kind="daily", units=7,  step_h=24, span_h=240,
         label="7 days",  levels=[2, 5, 10, 20, 40, 70, 100, 150, 250, 350, 500]),
    dict(id="colombia_precip_14d", kind="daily", units=14, step_h=24, span_h=240,
         label="14 days", levels=[5, 10, 20, 40, 70, 100, 175, 275, 400, 550, 750]),
    dict(id="colombia_precip_30d", kind="daily", units=30, step_h=24, span_h=240,
         label="30 days", levels=[10, 25, 50, 100, 175, 275, 400, 550, 750, 1000, 1300]),
    dict(id="colombia_precip_mtd", kind="mtd",   units=31, step_h=24, span_h=240,
         label="month-to-date", levels=[2, 5, 10, 25, 50, 100, 175, 275, 400, 550, 750]),
    dict(id="colombia_precip_90d", kind="daily", units=90, step_h=24, span_h=168,
         label="90 days", levels=[50, 100, 200, 350, 550, 800, 1100, 1450, 1850, 2300, 2800]),
]
DEFAULT_WINDOW = "colombia_precip_24h"

ANOM_WINDOWS = [
    dict(id="colombia_anom_1d",  days=1,  span=10, label="1-day",
         levels=[3, 7, 12, 20, 35, 60]),
    dict(id="colombia_anom_5d",  days=5,  span=10, label="5-day",
         levels=[5, 15, 30, 50, 85, 140]),
    dict(id="colombia_anom_7d",  days=7,  span=10, label="7-day",
         levels=[5, 15, 30, 60, 100, 160]),
    dict(id="colombia_anom_14d", days=14, span=10, label="14-day",
         levels=[10, 20, 40, 70, 120, 200]),
    dict(id="colombia_anom_30d", days=30, span=10, label="30-day",
         levels=[15, 30, 60, 110, 180, 300]),
    dict(id="colombia_anom_mtd", days=31, mtd=True, span=10, label="month-to-date",
         levels=[15, 30, 60, 110, 180, 300]),
    dict(id="colombia_anom_90d", days=90, span=7, label="90-day",
         levels=[30, 60, 120, 220, 350, 550]),
]
DEFAULT_ANOM = "colombia_anom_30d"


# ── sub-grid ──────────────────────────────────────────────────────────────────
def _sub_axes():
    ml, mt = IP._grid_axes()
    LONb, LATb = IP._LON[ml], IP._LAT[mt]
    lo0, lo1, la0, la1 = CCFG["extent"]
    sl = (LONb % 360 >= lo0) & (LONb % 360 <= lo1)
    st = (LATb >= la0) & (LATb <= la1)
    return sl, st, LONb[sl], LATb[st]


def _sub(field: np.ndarray) -> np.ndarray:
    sl, st, _, _ = _sub_axes()
    return field[np.ix_(st, sl)]


# ── overlays ─────────────────────────────────────────────────────────────────
_REGIONS = None


def _region_geoms():
    """Hydro region polygons: colombia_hydro_regions.geojson when available,
    else the committed Magdalena basin outline."""
    global _REGIONS
    if _REGIONS is None:
        from shapely.geometry import shape
        if REGIONS_GJ.exists():
            gj = json.loads(REGIONS_GJ.read_text())
            _REGIONS = [(f.get("properties", {}).get("name", ""), shape(f["geometry"]))
                        for f in gj.get("features", [])]
        else:
            _REGIONS = [("", g) for g in IP._magdalena_geom()]
    return _REGIONS


def _plants():
    return json.loads(PLANTS.read_text())["plants"]


def _geo(ax, dark: bool):
    coast, border, river = (("#b6b6b6", "#c2c2c2", "#4f7fa8") if dark
                            else ("#555", "#666", "#7fa8c9"))
    outline = "#ffffff" if dark else "#111111"
    ax.coastlines(linewidth=0.7, color=coast, resolution="10m")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.55, edgecolor=border)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "10m", facecolor="none"),
        edgecolor=river, linewidth=0.45, alpha=0.8, zorder=3)
    for name, geom in _region_geoms():
        ax.add_geometries([geom], crs=PC, facecolor="none",
                          edgecolor=outline, linewidth=1.0, zorder=4)
    # hydro plants: marker AREA ∝ capacity, cased for legibility on any field
    ps = _plants()
    lons = [p["lon"] for p in ps]; lats = [p["lat"] for p in ps]
    sizes = [max(10.0, p["mw"] * 0.085) for p in ps]           # pts² — Ituango ≈ 204
    ax.scatter(lons, lats, s=[s * 1.9 for s in sizes], c="white", transform=PC,
               zorder=5, lw=0)
    ax.scatter(lons, lats, s=sizes, c="#e8408a", edgecolors="#111", linewidths=0.5,
               transform=PC, zorder=5.1)
    for p in sorted(ps, key=lambda q: -q["mw"])[:3]:           # label the giants only
        ax.text(p["lon"] + 0.25, p["lat"], p["name"], transform=PC, fontsize=6.5,
                color=outline, va="center", zorder=6, fontweight="bold",
                path_effects=[mpe.withStroke(linewidth=1.6,
                                             foreground="#0d0d16" if dark else "white")])
    # capacity legend (three reference sizes)
    for mw, y in ((2400, 0.055), (800, 0.028), (200, 0.008)):
        ax.scatter([0.03], [y + 0.012], s=max(10.0, mw * 0.085), c="#e8408a",
                   edgecolors="white", linewidths=1.2, transform=ax.transAxes, zorder=7)
        ax.text(0.055, y + 0.012, f"{mw} MW", transform=ax.transAxes, fontsize=6,
                color=outline, va="center", zorder=7,
                path_effects=[mpe.withStroke(linewidth=1.5,
                                             foreground="#0d0d16" if dark else "white")])


def _frame_axes(dark: bool):
    lo0, lo1, la0, la1 = CCFG["extent"]; clon = CCFG["clon"]
    proj = ccrs.PlateCarree(central_longitude=clon)
    fig = plt.figure(figsize=CCFG["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([IP._offset(lo0, clon), IP._offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#0d0d16" if dark else "#f3f1ec")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.35,
                      linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([((t + 180) % 360) - 180
                  for t in range(int(lo0), int(lo1) + 1) if t % CCFG["dlon"] == 0])
    gl.ylocator = mticker.FixedLocator(range(int(la0) - 1, int(la1) + 1, CCFG["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return fig, ax


def render_total(field, vlabel, out: Path, win: dict):
    _, _, LON, LAT = _sub_axes()
    off = IP._offset(LON, CCFG["clon"])
    levels = win["levels"]
    norm = BoundaryNorm(levels, IP.PRECIP_CMAP.N, extend="max")
    fig, ax = _frame_axes(dark=True)
    wet = np.ma.masked_less(field, levels[0])
    cf = ax.contourf(off, LAT, wet, levels=levels, cmap=IP.PRECIP_CMAP, norm=norm,
                     extend="max",
                     transform=ccrs.PlateCarree(central_longitude=CCFG["clon"]))
    _geo(ax, dark=True)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=42, extend="max")
    cb.set_label("accumulated precip (mm) · circles = hydro plants (area ∝ MW)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Colombia hydro — IMERG {win['label']} precipitation  ·  {vlabel}",
                 fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def render_anom(field, vlabel, out: Path, win: dict):
    _, _, LON, LAT = _sub_axes()
    off = IP._offset(LON, CCFG["clon"])
    lev = [-x for x in reversed(win["levels"])] + list(win["levels"])
    norm = BoundaryNorm(lev, IPA.ANOM_CMAP.N, extend="both")
    fig, ax = _frame_axes(dark=False)
    cf = ax.contourf(off, LAT, field, levels=lev, cmap=IPA.ANOM_CMAP, norm=norm,
                     extend="both",
                     transform=ccrs.PlateCarree(central_longitude=CCFG["clon"]))
    _geo(ax, dark=False)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=42, extend="both")
    cb.set_label("precip anomaly vs 2001–2025 climatology (mm) · circles = hydro plants (area ∝ MW)",
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Colombia hydro — IMERG {win['label']} precip anomaly  ·  {vlabel}",
                 fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


# ── loops (mirror imerg_gatun's structure) ───────────────────────────────────
def _units(w, ft, step):
    return (ft - step).day if w["kind"] == "mtd" else w["units"]


def build_totals(anchor_h, anchor_day):
    regions = {}
    for w in WINDOWS:
        cache = IP.HOURLY_CACHE if w["kind"] == "hourly" else IP.DAILY_CACHE
        step = timedelta(hours=1) if w["kind"] == "hourly" else timedelta(hours=24)
        fmt = "%Y%m%d%H" if w["kind"] == "hourly" else "%Y%m%d"
        base = anchor_h if w["kind"] == "hourly" else anchor_day
        n = w["span_h"] // w["step_h"]
        ftimes = [base - k * timedelta(hours=w["step_h"]) for k in range(n)][::-1]
        need = set()
        for ft in ftimes:
            for k in range(1, _units(w, ft, step) + 1):
                need.add((ft - k * step).replace(minute=0, second=0, microsecond=0))
        (IP.ensure_hourly if w["kind"] == "hourly" else IP.ensure_daily)(need)
        anim = ANIM_ROOT / w["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            u = _units(w, ft, step)
            if not fp.exists():
                field = IP.trailing_sum(cache, ft, u, step, fmt)
                if field is None:
                    continue
                if w["kind"] == "mtd":
                    e = ft - step
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                elif w["kind"] == "daily":
                    vlabel = f"{(ft - timedelta(days=u)):%b %d} – {(ft - step):%b %d, %Y}"
                else:
                    vlabel = f"ending {ft:%Y-%m-%d %HZ}"
                render_total(_sub(field), vlabel, fp, w)
                print(f"  {w['id']}: rendered {tag}", flush=True)
            label = (f"MTD → {(ft - step):%b %-d}" if w["kind"] == "mtd"
                     else f"{(ft - timedelta(days=u)):%b %d} – {(ft - step):%b %d}"
                     if w["kind"] == "daily" else f"{ft:%-d %b %HZ}")
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": f"{ft:%Y-%m-%d}", "label": label})
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[w["id"]] = {"label": f"{w['label']} accumulation", "frames": entries}
    return regions


def build_anoms(coef, anchor_day):
    regions = {}
    step = timedelta(days=1)
    for w in ANOM_WINDOWS:
        ftimes = [anchor_day - k * step for k in range(w["span"])][::-1]
        need = set()
        for ft in ftimes:
            nd = (ft - step).day if w.get("mtd") else w["days"]
            for k in range(1, nd + 1):
                need.add((ft - k * step).replace(hour=0, minute=0, second=0, microsecond=0))
        IP.ensure_daily(need)
        anim = ANIM_ROOT / w["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            nd = (ft - step).day if w.get("mtd") else w["days"]
            if not fp.exists():
                recent = IP.trailing_sum(IP.DAILY_CACHE, ft, nd, step, "%Y%m%d")
                if recent is None:
                    continue
                anom = recent - IPA.clim_accum(coef, ft, nd)
                if w.get("mtd"):
                    e = ft - step
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                else:
                    vlabel = f"{(ft - timedelta(days=nd)):%b %d} – {(ft - step):%b %d, %Y}"
                render_anom(_sub(anom), vlabel, fp, w)
                print(f"  {w['id']}: rendered {tag}", flush=True)
            label = (f"MTD → {(ft - step):%b %-d}" if w.get("mtd")
                     else f"{(ft - timedelta(days=nd)):%b %d} – {(ft - step):%b %d}")
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": f"{ft:%Y-%m-%d}", "label": label})
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[w["id"]] = {"label": f"{w['label']} anomaly", "frames": entries}
    return regions


def main() -> int:
    import xarray as xr
    anchor_h, anchor_day = IP._latest_anchors()
    print(f"  anchors: hourly→{anchor_h:%Y-%m-%d %HZ}, daily→{anchor_day:%Y-%m-%d}", flush=True)
    regions = build_totals(anchor_h, anchor_day)
    (ANIM_ROOT / "colombia_precip_manifest.json").write_text(json.dumps(
        {"ver": anchor_h.strftime("%Y%m%d%H"), "selectorLabel": "Accumulation",
         "default": DEFAULT_WINDOW, "regions": regions}))
    print(f"wrote {len(regions)} total windows + colombia_precip_manifest.json", flush=True)
    if CLIM_NC.exists():
        coef = xr.open_dataset(CLIM_NC)["coef"].values
        aregions = build_anoms(coef, anchor_day)
        (ANIM_ROOT / "colombia_anom_manifest.json").write_text(json.dumps(
            {"ver": anchor_day.strftime("%Y%m%d"), "selectorLabel": "Window",
             "default": DEFAULT_ANOM, "regions": aregions}))
        print(f"wrote {len(aregions)} anomaly windows + colombia_anom_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
