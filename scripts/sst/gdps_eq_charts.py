#!/usr/bin/env python3
"""Tropical Pacific forecast products from the ECCC GDPS — wind maps & simulated IR.

The Canadian Global Deterministic Prediction System (15 km, open Datamart)
as an independent physics model beside the AIFS-ENS / IFS-ENS products on
the Atmosphere page:

  assets/sst/anim/gdps_wind/     — daily MSLP + 10 m wind maps, forecast
  + gdps_wind_manifest.json        days 1..10, same domain, shading and
                                   grammar as the super-ensemble animator
                                   (scripts/mjo/src/mslp_wind_anim.py)
  assets/sst/anim/gdps_ir/       — SIMULATED IR satellite loop: forecast
  + gdps_ir_manifest.json          top-of-atmosphere OLR converted to IR
                                   brightness temperature (inverse of the
                                   Ohring–Gruber relation the synthetic-OLR
                                   products use forward) and rendered
                                   exactly like the live GMGSI tropical
                                   Pacific loop (pacific_satellite.py):
                                   same domain, colortable and styling,
                                   3-hourly frames to day 10

GDPS is deterministic (one run, no spread) and Datamart serves whole-globe
GRIBs only, but the files are small (~1–2 MB) and the 00Z cycle is up by
~06 UTC. Model OLR is smoother than real pixel IR — 15 km fields resolve
the convective envelopes, not individual overshooting tops.

    python scripts/sst/gdps_eq_charts.py             # latest complete 00Z
    python scripts/sst/gdps_eq_charts.py --date 20260824
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parent.parent)
ASSETS = SITE_ROOT / "assets" / "sst"
sys.path.insert(0, str(HERE))
# The GMGSI loops' enhanced-IR colortable + shared graticule helpers.
from pacific_satellite import IR_CMAP, IR_NORM                  # noqa: E402
import map_grid                                                 # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap      # noqa: E402

# Map domain + grammar copied from the super-ensemble animator
# (scripts/mjo/src/mslp_wind_anim.py) so the two products read identically.
EXTENT = (100, 280, -30, 45)                    # lon0, lon1 (0..360), lat0, lat1
STATIONS = {"Darwin (YPDN)": (130.9, -12.4), "Tarawa (NGTA)": (173.0, 1.4),
            "Christmas I. (PLCH)": (202.5, 2.0), "Tahiti (NTAA)": (210.4, -17.5)}
MS2KT = 1.94384
PLEVS = np.arange(900, 1064, 4)                 # MSLP contour levels (hPa)
WLEV = [5, 8, 11, 14, 17, 20, 23, 27, 31, 36, 41, 47, 53, 60]
WCOLS = ["#dcefff", "#bfe0f5", "#9ccde9", "#73aedb", "#4a86c5", "#3559a8",
         "#5a3f9c", "#8036a0", "#a82f9c", "#cf2592", "#e8408a", "#f57247", "#f59f00"]
WCMAP = ListedColormap(WCOLS); WCMAP.set_under("#ffffff00"); WCMAP.set_over("#d97706")
WNORM = BoundaryNorm(WLEV, WCMAP.N)

# Simulated-IR domain/styling: the "pacsat" preset of the live GMGSI loop
# (pacific_satellite.REGIONS) — same crop the main page shows. Rendered at a
# higher dpi than the live loop: the page column runs up to ~2000 px and the
# lightbox serves the raw frame, so ~1550 px keeps the 15 km field at native
# detail. Slightly taller figure than pacsat to make room for the colorbar.
IR_EXTENT = (100, 290, -40, 40)
IR_CLON = 180.0
IR_FIGSIZE = (12.4, 6.1)
IR_DPI = 150
IR_DLON, IR_DLAT = 20, 20

BASE = "https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{cyc}/{lead:03d}"
FILE = "{date}T{cyc}Z_MSC_GDPS_{var}_LatLon0.15_PT{lead:03d}H.grib2"
VARS = {"u": "WindU_AGL-10m", "v": "WindV_AGL-10m", "p": "Pressure_MSL",
        "u150": "WindU_IsbL-0150", "v150": "WindV_IsbL-0150",
        "olr": "UpwardLongwaveRadiationFlux_NTAtm"}
LEADS = list(range(24, 241, 24))                 # wind maps: forecast days 1..10
IR_LEADS = list(range(3, 241, 3))                # simulated IR: 3-hourly loop
MIN_LEADS = 8                                    # fewer → cycle not ready, fall back

# OLR → IR brightness temperature: inverse of the Ohring–Gruber relation the
# synthetic-OLR products use forward (Tf = Tb·(1.228 − 1.106e-3·Tb); OLR = σTf⁴).
SIGMA = 5.670374419e-8
_OG_A, _OG_B = 1.106e-3, 1.228


def olr_to_tb(olr: np.ndarray) -> np.ndarray:
    tf = (np.asarray(olr, float) / SIGMA) ** 0.25
    disc = np.clip(_OG_B ** 2 - 4.0 * _OG_A * tf, 0.0, None)
    return (_OG_B - np.sqrt(disc)) / (2.0 * _OG_A)


# ── Datamart fetch ───────────────────────────────────────────────────────────
def fetch(date: str, cyc: str, lead: int, var: str) -> Path | None:
    url = (BASE.format(date=date, cyc=cyc, lead=lead) + "/"
           + FILE.format(date=date, cyc=cyc, var=VARS[var], lead=lead))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scorvec-enso/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    tmp = Path(tempfile.mkstemp(suffix=".grib2")[1])
    tmp.write_bytes(data)
    return tmp


def _grab_map(date: str, cyc: str, lead: int, var: str, extent) -> xr.DataArray | None:
    """One field subset to a map domain, loaded. GDPS grids are lat-ascending,
    lon 0–360 — no reorientation needed."""
    p = fetch(date, cyc, lead, var)
    if p is None:
        return None
    try:
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
        da = ds[list(ds.data_vars)[0]].sel(
            latitude=slice(extent[2], extent[3]),
            longitude=slice(extent[0], extent[1])).load()
    except Exception:                                   # noqa: BLE001
        da = None
    p.unlink(missing_ok=True)
    return da


def collect_maps(date: str, cyc: str) -> dict | None:
    """Per-forecast-day map fields (surface u/v/mslp + 150 hPa u/v), or None
    if the cycle is incomplete."""
    maps, leads_ok = [], []
    for lead in LEADS:
        f = {k: _grab_map(date, cyc, lead, k, EXTENT)
             for k in ("u", "v", "p", "u150", "v150")}
        if any(v is None for v in f.values()):
            print(f"  {date} {cyc}Z +{lead:03d}h: map fields missing — skipped",
                  flush=True)
            continue
        maps.append(f)
        leads_ok.append(lead)
    if len(leads_ok) < MIN_LEADS:
        print(f"  {date} {cyc}Z: only {len(leads_ok)} days — cycle incomplete",
              flush=True)
        return None
    print(f"  {date} {cyc}Z: {len(leads_ok)} wind-map days", flush=True)
    return {"maps": maps, "leads": leads_ok}


# ── MSLP + 10 m wind map frames (mirrors mslp_wind_anim.py) ─────────────────
# GDPS is 0.15° vs the super-ensemble's 0.25°: smoothing / neighbourhood sizes
# scale by 0.25/0.15 so H/L detection and contour smoothness match visually.
_GS = 0.25 / 0.15


def _hl(p2d, lat, lon, ax, proj):
    s = gaussian_filter(p2d, 4 * _GS, mode=("nearest", "wrap"))
    for filt, op, col, sym in ((minimum_filter, np.less_equal, "#c0152f", "L"),
                               (maximum_filter, np.greater_equal, "#1f4fb0", "H")):
        ext = filt(s, size=int(28 * _GS), mode=("nearest", "wrap"))
        ys, xs = np.where(op(s, ext))
        seen = []
        for y, x in zip(ys, xs):
            if any(abs(y - yy) < 18 * _GS and abs(x - xx) < 18 * _GS
                   for yy, xx in seen):
                continue
            seen.append((y, x))
            ax.text(lon[x], lat[y], sym, color=col, fontsize=12, fontweight="bold",
                    ha="center", va="center", transform=proj, clip_on=True)
            ax.text(lon[x], lat[y] - 2.4, f"{s[y, x]:.0f}", color=col, fontsize=6.5,
                    ha="center", va="top", transform=proj, clip_on=True)


def render_wind_maps(maps, leads, init: pd.Timestamp) -> None:
    anim = ASSETS / "anim" / "gdps_wind"
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    proj = ccrs.PlateCarree(central_longitude=180)
    pc = ccrs.PlateCarree()
    entries = []
    for k, (f, h) in enumerate(zip(maps, leads)):
        u, v, pr = f["u"], f["v"], f["p"]
        valid = init + pd.Timedelta(hours=int(h))
        lat = u.latitude.values; lon = u.longitude.values
        spd = np.hypot(u.values, v.values) * MS2KT
        p_hpa = pr.values / 100.0
        bstride = max(1, int(round(3.5 / abs(lat[1] - lat[0]))))
        fig = plt.figure(figsize=(12.6, 6.2))
        ax = plt.axes(projection=proj)
        ax.set_extent([EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]], crs=pc)
        cf = ax.contourf(lon, lat, spd, levels=WLEV, cmap=WCMAP, norm=WNORM,
                         extend="both", transform=pc)
        cs = ax.contour(lon, lat, gaussian_filter(p_hpa, 1.2 * _GS,
                                                  mode=("nearest", "wrap")),
                        levels=PLEVS, colors="#333", linewidths=0.6, transform=pc)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
        ax.barbs(lon[::bstride], lat[::bstride],
                 u.values[::bstride, ::bstride] * MS2KT,
                 v.values[::bstride, ::bstride] * MS2KT,
                 length=4.2, linewidth=0.4, color="#222", transform=pc)
        _hl(p_hpa, lat, lon, ax, pc)
        ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="0.05",
                       linewidth=1.1, zorder=4)
        ax.coastlines(linewidth=1.1, color="0.05", zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.45", alpha=0.5,
                          linestyle=(0, (3, 3)), zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(list(range(-180, 181, 20)))
        gl.ylocator = mticker.FixedLocator(list(range(-30, 46, 15)))
        gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": "0.3"}
        for name, (slon, slat) in STATIONS.items():
            ax.plot(slon, slat, marker="o", ms=4.5, mfc="#ffd400", mec="k",
                    mew=0.7, transform=pc, zorder=7)
            ax.text(slon, slat + 1.6, name, fontsize=6.2, fontweight="bold",
                    ha="center", va="bottom", color="k", transform=pc, zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        cax = fig.add_axes([0.13, 0.06, 0.74, 0.02])
        fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both").set_label(
            "10 m wind speed (kt)", fontsize=8)
        cax.tick_params(labelsize=7)
        ax.set_title(f"GDPS MSLP (mb) + 10 m wind  ·  ECCC 15 km deterministic\n"
                     f"init {init:%Y-%m-%d %H}Z  ·  F{int(h):03d} valid "
                     f"{valid:%Y-%m-%d %H}Z", fontsize=10, loc="left")
        fp = anim / f"F{k:02d}.webp"
        fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.10)
        fig.savefig(fp, dpi=104); plt.close(fig)
        entries.append({"idx": k, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                        "label": f"F{int(h):03d} · {valid:%Y-%m-%d %H}Z"})
    mani = {"ver": f"{init:%Y%m%d%H}",
            "regions": {"gdps_wind": {"label": "GDPS MSLP + 10 m wind",
                                      "frames": entries}}}
    (ASSETS / "anim" / "gdps_wind_manifest.json").write_text(json.dumps(mani))
    print(f"  wrote {len(entries)} wind-map frames + gdps_wind_manifest.json",
          flush=True)


# ── 150 hPa outflow maps: divergence shaded + streamlines + jet isotachs ────
# Colour language matches the site's 200 hPa velocity-potential product
# (wind200_vpot.py): GREEN = upper-level divergence (the outflow above deep
# convection), ORANGE = convergence/subsidence, white = neutral.
from matplotlib.colors import LinearSegmentedColormap                # noqa: E402
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div150", ["#a8330f", "#df6a1e", "#f0a64b", "#fbe2bd", "#ffffff",
               "#cfe8cf", "#86c98a", "#43a047", "#1b5e20"])
# Heavier smoothing + wider bins than a first guess: at 2°/±1.5 the map
# saturated into mesoscale blobs — 3° and ±3 leaves white background with
# only the planetary-scale outflow/convergence centres coloured.
DIV_LEVELS = [-20, -15, -10, -6, -3, 3, 6, 10, 15, 20]    # ×1e-6 s⁻¹, smoothed
JET_LEVELS = [30, 50, 70]                                 # 150 hPa isotachs (m/s)
DIV_SMOOTH_DEG = 3.0                                      # Gaussian sigma (degrees)


def _divergence(u: xr.DataArray, v: xr.DataArray) -> np.ndarray:
    """Spherical horizontal divergence (s⁻¹) of one (lat, lon) level."""
    a = 6.371e6
    lat_r = np.deg2rad(u.latitude.values)
    lon_r = np.deg2rad(u.longitude.values)
    coslat = np.cos(lat_r)[:, None]
    dudx = np.gradient(u.values, lon_r, axis=1) / (a * coslat)
    dvdy = np.gradient(v.values * coslat, lat_r, axis=0) / (a * coslat)
    return dudx + dvdy


def render_outflow_maps(maps, leads, init: pd.Timestamp) -> None:
    anim = ASSETS / "anim" / "gdps_outflow"
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    proj = ccrs.PlateCarree(central_longitude=180)
    pc = ccrs.PlateCarree()
    entries = []
    for k, (f, h) in enumerate(zip(maps, leads)):
        u, v = f["u150"], f["v150"]
        valid = init + pd.Timedelta(hours=int(h))
        lat = u.latitude.values; lon = u.longitude.values
        dx = abs(lat[1] - lat[0])
        div = gaussian_filter(_divergence(u, v), DIV_SMOOTH_DEG / dx,
                              mode=("nearest", "wrap")) * 1e6
        spd = np.hypot(u.values, v.values)
        fig = plt.figure(figsize=(12.6, 6.2))
        ax = plt.axes(projection=proj)
        ax.set_extent([EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]], crs=pc)
        cf = ax.contourf(lon, lat, div, levels=DIV_LEVELS, cmap=DIV_CMAP,
                         extend="both", transform=pc)
        cj = ax.contour(lon, lat, gaussian_filter(spd, 1.0 / dx,
                                                  mode=("nearest", "wrap")),
                        levels=JET_LEVELS, colors="#111111", linewidths=0.7,
                        transform=pc)
        ax.clabel(cj, inline=True, fontsize=6, fmt="%d")
        # streamlines on a ~0.6° subsample — full 0.15° makes streamplot crawl
        s = 4
        lw = 0.35 + 1.1 * np.clip(spd[::s, ::s] / 60.0, 0, 1)
        ax.streamplot(lon[::s], lat[::s], u.values[::s, ::s], v.values[::s, ::s],
                      transform=pc, density=2.4, color="#37474f",
                      linewidth=lw, arrowsize=0.7)
        ax.coastlines(linewidth=1.0, color="0.05", zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.45", alpha=0.5,
                          linestyle=(0, (3, 3)), zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(list(range(-180, 181, 20)))
        gl.ylocator = mticker.FixedLocator(list(range(-30, 46, 15)))
        gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": "0.3"}
        cax = fig.add_axes([0.13, 0.06, 0.74, 0.02])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
        cb.set_label("150 hPa divergence (10⁻⁶ s⁻¹) · green = outflow above deep "
                     "convection · orange = convergence · black contours = "
                     "isotachs (m/s)", fontsize=8)
        cax.tick_params(labelsize=7)
        ax.set_title(f"GDPS 150 hPa outflow — divergence + streamlines  ·  "
                     f"ECCC 15 km deterministic\ninit {init:%Y-%m-%d %H}Z  ·  "
                     f"F{int(h):03d} valid {valid:%Y-%m-%d %H}Z",
                     fontsize=10, loc="left")
        fp = anim / f"F{k:02d}.webp"
        fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.10)
        fig.savefig(fp, dpi=104); plt.close(fig)
        entries.append({"idx": k, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                        "label": f"F{int(h):03d} · {valid:%Y-%m-%d %H}Z"})
    mani = {"ver": f"{init:%Y%m%d%H}",
            "regions": {"gdps_outflow": {"label": "GDPS 150 hPa outflow",
                                         "frames": entries}}}
    (ASSETS / "anim" / "gdps_outflow_manifest.json").write_text(json.dumps(mani))
    print(f"  wrote {len(entries)} outflow frames + gdps_outflow_manifest.json",
          flush=True)


# ── Simulated IR loop (mirrors pacific_satellite.render_frame, pacsat) ──────
def render_ir_frame(tb2d, lat, lon, init: pd.Timestamp, lead: int,
                    out: Path) -> None:
    valid = init + pd.Timedelta(hours=int(lead))
    lo0, lo1, la0, la1 = IR_EXTENT
    proj = ccrs.PlateCarree(central_longitude=IR_CLON)
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=IR_FIGSIZE)
    ax = fig.add_axes([0.045, 0.14, 0.93, 0.78], projection=proj)
    ax.set_extent([lo0, lo1, la0, la1], crs=pc)
    mesh = ax.pcolormesh(lon, lat, tb2d, transform=pc, cmap=IR_CMAP, norm=IR_NORM,
                         shading="auto", rasterized=True)
    ax.coastlines(linewidth=0.7, color="#cfcfcf", resolution="110m")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="w", alpha=0.55,
                      linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    lon_ticks = [((t + 180) % 360) - 180
                 for t in range(int(lo0), int(lo1) + 1) if t % IR_DLON == 0]
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(map_grid.lat_ticks(la0, la1, IR_DLAT))
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    map_grid.add_ref_lines(ax, IR_EXTENT, color="w", lw=1.0)
    ax.set_title(f"GDPS simulated enhanced IR — tropical Pacific  ·  "
                 f"init {init:%Y-%m-%d %HZ}  ·  F{lead:03d} valid "
                 f"{valid:%Y-%m-%d %HZ}  ·  colour = deep convection",
                 fontsize=9, loc="left")
    # legend: the enhanced-IR ramp with its brightness-temperature scale
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.020])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks(list(range(180, 301, 20)))
    cb.set_label("IR brightness temperature (K) · colour = cold tops (deep "
                 "convection), grayscale = warm / clear", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(out, dpi=IR_DPI, bbox_inches="tight",
                pil_kwargs={"quality": 74, "method": 6})
    plt.close(fig)


def build_ir_loop(date: str, cyc: str, init: pd.Timestamp) -> None:
    """Stream the 3-hourly OLR steps → simulated-IR frames (one at a time, so
    ~80 domain fields never sit in memory together)."""
    anim = ASSETS / "anim" / "gdps_ir"
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    entries, made = [], 0
    for lead in IR_LEADS:
        da = _grab_map(date, cyc, lead, "olr", IR_EXTENT)
        if da is None:
            print(f"  IR +{lead:03d}h: missing — skipped", flush=True)
            continue
        fp = anim / f"F{lead:03d}.webp"
        render_ir_frame(olr_to_tb(da.values), da.latitude.values,
                        da.longitude.values, init, lead, fp)
        valid = init + pd.Timedelta(hours=lead)
        entries.append({"idx": made, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                        "label": f"F{lead:03d} · {valid:%Y-%m-%d %H}Z"})
        made += 1
    mani = {"ver": f"{init:%Y%m%d%H}",
            "regions": {"gdps_ir": {"label": "GDPS simulated IR — forecast loop",
                                    "frames": entries}}}
    (ASSETS / "anim" / "gdps_ir_manifest.json").write_text(json.dumps(mani))
    print(f"  wrote {made} simulated-IR frames + gdps_ir_manifest.json",
          flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="init date YYYYMMDD (default: latest complete)")
    ap.add_argument("--cycle", default="00")
    args = ap.parse_args(argv)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    tries = ([args.date] if args.date
             else [today, (datetime.now(timezone.utc)
                           - timedelta(days=1)).strftime("%Y%m%d")])
    data, init, date_ok = None, None, None
    for date in tries:
        print(f"GDPS {date} {args.cycle}Z:", flush=True)
        data = collect_maps(date, args.cycle)
        if data is not None:
            init = pd.Timestamp(f"{date}T{args.cycle}:00")
            date_ok = date
            break
    if data is None:
        print("no complete GDPS cycle available", flush=True)
        return 1

    render_wind_maps(data["maps"], data["leads"], init)
    render_outflow_maps(data["maps"], data["leads"], init)
    build_ir_loop(date_ok, args.cycle, init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
