#!/usr/bin/env python3
"""Tropical Pacific forecast products from the ECCC GDPS — 10 m wind maps & OLR.

The Canadian Global Deterministic Prediction System (15 km, open Datamart)
as an independent physics model beside the AIFS-ENS / IFS-ENS products on
the Atmosphere page:

  assets/sst/anim/gdps_wind/     — daily MSLP + 10 m wind maps, forecast
  + gdps_wind_manifest.json        days 1..10, same domain, shading and
                                   grammar as the super-ensemble animator
                                   (scripts/mjo/src/mslp_wind_anim.py)
  assets/sst/gdps_olr.webp       — tropical OLR Hovmöller (5°S–5°N,
                                   longitude × forecast day), absolute
                                   values on the SAME enhanced-IR ramp as
                                   the GMGSI satellite loops (OLR → Tb via
                                   the inverted Ohring–Gruber relation the
                                   synthetic-OLR products use forward)

GDPS is deterministic (one run, no spread) and Datamart serves whole-globe
GRIBs only, but the files are small (~1–2 MB) and the 00Z cycle is up by
~06 UTC. OLR rows are TRUE DAILY MEANS of the 3-hourly instantaneous
fields — sampling one fixed UTC hour imprints the diurnal cycle as a
standing stripe over the Indian Ocean (seen on the first cut of this
chart).

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
# The GMGSI loops' enhanced-IR colortable.
from pacific_satellite import IR_CMAP                           # noqa: E402
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

BASE = "https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{cyc}/{lead:03d}"
FILE = "{date}T{cyc}Z_MSC_GDPS_{var}_LatLon0.15_PT{lead:03d}H.grib2"
VARS = {"u": "WindU_AGL-10m", "v": "WindV_AGL-10m", "p": "Pressure_MSL",
        "olr": "UpwardLongwaveRadiationFlux_NTAtm"}
LEADS = list(range(24, 241, 24))                 # forecast days 1..10
OLR_STEP = 3                                     # GDPS output cadence (hours)
MIN_LEADS = 8                                    # fewer → cycle not ready, fall back
MIN_OLR_SAMPLES = 6                              # of 8 three-hourly steps per day
LAT_BAND = 5.0
LON_GRID = np.arange(0.0, 360.0, 1.0)
LON_VIEW = (40.0, 290.0)                         # Indian Ocean → eastern Pacific

# The GMGSI ramp is built for instantaneous pixel brightness temperatures; a
# 5°S–5°N DAILY-MEAN band average never gets that cold (everything landed in
# the grayscale half on the first cut). Same colours, renormalized in
# OLR-space: the ramp's grayscale→enhancement break (240 K on the loops) is
# pinned to the 220 W m⁻² deep-convection threshold the site's OLR products
# contour, so colour = convectively active, grayscale = suppressed/clear.
OLR_VMIN, OLR_VMAX = 140.0, 300.0        # break sits at (220−140)/160 = 0.5


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


def _open_field(path: Path) -> xr.DataArray | None:
    """GDPS grids are lat-ascending, lon 0–360 — no reorientation needed."""
    try:
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
    except Exception:                                   # noqa: BLE001
        return None
    return ds[list(ds.data_vars)[0]]


def _grab_map(date: str, cyc: str, lead: int, var: str) -> xr.DataArray | None:
    """One field subset to the map domain, loaded."""
    p = fetch(date, cyc, lead, var)
    if p is None:
        return None
    da = _open_field(p)
    if da is not None:
        da = da.sel(latitude=slice(EXTENT[2], EXTENT[3]),
                    longitude=slice(EXTENT[0], EXTENT[1])).load()
    p.unlink(missing_ok=True)
    return da


def _grab_band(date: str, cyc: str, lead: int, var: str) -> np.ndarray | None:
    """Cosine-weighted 5°S–5°N mean on LON_GRID."""
    p = fetch(date, cyc, lead, var)
    if p is None:
        return None
    da = _open_field(p)
    if da is None:
        p.unlink(missing_ok=True)
        return None
    da = da.sel(latitude=slice(-LAT_BAND, LAT_BAND))
    w = np.cos(np.deg2rad(da.latitude))
    out = da.weighted(w).mean("latitude").interp(longitude=LON_GRID).values
    p.unlink(missing_ok=True)               # only after the lazy read resolves
    return out


def collect(date: str, cyc: str) -> dict | None:
    """Per-forecast-day map fields (u, v, mslp) + daily-mean OLR band rows,
    or None if the cycle is incomplete."""
    maps, olr_rows, leads_ok = [], [], []
    for lead in LEADS:
        u = _grab_map(date, cyc, lead, "u")
        v = _grab_map(date, cyc, lead, "v")
        pr = _grab_map(date, cyc, lead, "p")
        if u is None or v is None or pr is None:
            print(f"  {date} {cyc}Z +{lead:03d}h: map fields missing — skipped",
                  flush=True)
            continue
        samples = [b for h in range(lead - 24 + OLR_STEP, lead + 1, OLR_STEP)
                   if (b := _grab_band(date, cyc, h, "olr")) is not None]
        if len(samples) < MIN_OLR_SAMPLES:
            print(f"  {date} {cyc}Z day {lead // 24}: only {len(samples)} OLR "
                  f"steps — skipped", flush=True)
            continue
        maps.append((u, v, pr))
        olr_rows.append(np.mean(samples, axis=0))
        leads_ok.append(lead)
        print(f"  {date} {cyc}Z day {lead // 24}: maps + {len(samples)}-step "
              f"OLR mean", flush=True)
    if len(leads_ok) < MIN_LEADS:
        print(f"  {date} {cyc}Z: only {len(leads_ok)} days — cycle incomplete",
              flush=True)
        return None
    return {"maps": maps, "olr": np.array(olr_rows), "leads": leads_ok}


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
    for k, ((u, v, pr), h) in enumerate(zip(maps, leads)):
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


# ── OLR Hovmöller, GMGSI-IR colours ─────────────────────────────────────────
def plot_olr(absf, lead_days, init: pd.Timestamp, out: Path):
    m = (LON_GRID >= LON_VIEW[0]) & (LON_GRID <= LON_VIEW[1])
    lons = LON_GRID[m]
    fig = plt.figure(figsize=(5.4, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 6.5], hspace=0.05,
                          left=0.10, right=0.86, top=0.87, bottom=0.07)
    fig.suptitle(f"Tropical OLR forecast — init {init:%Y-%m-%d %HZ}\n"
                 f"ECCC GDPS top-of-atmosphere OLR · 5°S–5°N daily mean\n"
                 f"GMGSI enhanced-IR colours · colour = deep convection",
                 fontsize=11, fontweight="bold")
    axm = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
    axm.set_extent([LON_VIEW[0], LON_VIEW[1], -15, 15], crs=ccrs.PlateCarree())
    axm.set_aspect("auto")
    axm.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf", zorder=2)
    axm.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555",
                    linewidth=0.3, zorder=3)
    axm.add_patch(plt.Rectangle((LON_VIEW[0], -LAT_BAND),
                  LON_VIEW[1] - LON_VIEW[0], 2 * LAT_BAND,
                  transform=ccrs.PlateCarree(), facecolor="none",
                  edgecolor="k", lw=0.6, ls="--", zorder=4))
    ax = fig.add_subplot(gs[1])
    im = ax.contourf(lons, lead_days, absf[:, m],
                     levels=np.linspace(OLR_VMIN, OLR_VMAX, 81),
                     cmap=IR_CMAP, extend="both")
    ax.contour(lons, lead_days, absf[:, m], levels=[180, 220], colors="k",
               linewidths=0.5, alpha=0.5)
    ax.set_ylim(max(lead_days), min(lead_days))
    ticks = [60, 120, 180, 240]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}°E" if t <= 180 else f"{360 - t}°W" for t in ticks])
    ax.tick_params(labelsize=8)
    ax.axvline(180, color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Forecast lead (days)", fontsize=9)
    cax = fig.add_axes([0.88, 0.09, 0.022, 0.72])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks(list(range(140, 301, 20)))
    cb.set_label("OLR (W m⁻²) · colour below the 220 convective threshold",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="init date YYYYMMDD (default: latest complete)")
    ap.add_argument("--cycle", default="00")
    args = ap.parse_args(argv)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    tries = ([args.date] if args.date
             else [today, (datetime.now(timezone.utc)
                           - timedelta(days=1)).strftime("%Y%m%d")])
    data, init = None, None
    for date in tries:
        print(f"GDPS {date} {args.cycle}Z:", flush=True)
        data = collect(date, args.cycle)
        if data is not None:
            init = pd.Timestamp(f"{date}T{args.cycle}:00")
            break
    if data is None:
        print("no complete GDPS cycle available", flush=True)
        return 1

    render_wind_maps(data["maps"], data["leads"], init)
    plot_olr(data["olr"], np.array(data["leads"]) / 24.0, init,
             ASSETS / "gdps_olr.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
