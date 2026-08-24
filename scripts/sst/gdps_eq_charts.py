#!/usr/bin/env python3
"""Equatorial forecast Hovmöllers from the ECCC GDPS — 10 m zonal wind & OLR.

The Canadian Global Deterministic Prediction System (15 km, open Datamart),
as an independent cross-check on the AIFS-ENS / IFS-ENS equatorial wind
Hovmöller (scripts/mjo/src/eq_hovmoller.py) — same layout, same ERA5
climatology, so the columns read identically:

  assets/sst/gdps_eq_wind.webp — 10 m zonal wind, 5°S–5°N, longitude ×
                                 forecast day 1..10; anomaly vs the ERA5
                                 1991–2020 climatology + absolute field
  assets/sst/gdps_olr.webp     — top-of-atmosphere OLR over the same band:
                                 anomaly vs the NOAA interpolated-OLR
                                 1979–2022 daily climatology + absolute

GDPS is deterministic (one run, no spread) and Datamart serves whole-globe
GRIBs only, but the files are small (~1–2 MB) and the 00Z cycle is up by
~06 UTC. Wind rows are instantaneous daily steps (same as the AIFS chart);
OLR rows are TRUE DAILY MEANS of the 3-hourly instantaneous fields —
sampling one fixed UTC hour against a daily-mean climatology imprints the
diurnal cycle as a standing +30 W m⁻² brown stripe over the Indian Ocean
(seen on the first cut of this chart). The OLR anomaly is still
model-minus-observed-climatology, so a constant model bias can tint the
field — read the pattern, not the absolute numbers.

    python scripts/sst/gdps_eq_charts.py             # latest complete 00Z
    python scripts/sst/gdps_eq_charts.py --date 20260824
"""
from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parent.parent)
ASSETS = SITE_ROOT / "assets" / "sst"
sys.path.insert(0, str(SITE_ROOT / "scripts" / "mjo" / "src"))
from build_eq_wind_clim import eval_clim                      # noqa: E402

BASE = "https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{cyc}/{lead:03d}"
FILE = "{date}T{cyc}Z_MSC_GDPS_{var}_LatLon0.15_PT{lead:03d}H.grib2"
VARS = {"u": "WindU_AGL-10m", "olr": "UpwardLongwaveRadiationFlux_NTAtm"}
LEADS = list(range(24, 241, 24))                 # forecast days 1..10
OLR_STEP = 3                                     # GDPS output cadence (hours)
MIN_LEADS = 8                                    # fewer → cycle not ready, fall back
MIN_OLR_SAMPLES = 6                              # of 8 three-hourly steps per day
LAT_BAND = 5.0
LON_GRID = np.arange(0.0, 360.0, 1.0)
LON_VIEW = (40.0, 290.0)                         # Indian Ocean → eastern Pacific
U_CLIM = SITE_ROOT / "scripts" / "mjo" / "data" / "reference" / "eq_u10_clim.nc"
OLR_CLIM = HERE / "metar" / "olr_clim.nc"        # NOAA interp-OLR 1979–2022 bands

# Same display grammar as the AIFS/IFS chart (eq_hovmoller.py) …
U_ANOM_LIM, U_ABS_LIM = 12.0, 10.0
# … and as the site's OLR products (synthetic_olr.py / olr_waves.py).
OLR_LEV = list(range(100, 301, 10))
OLR_ANOM_LEV = np.arange(-60, 61, 10)


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


def band_mean(path: Path) -> np.ndarray | None:
    """Cosine-weighted 5°S–5°N mean on LON_GRID; GDPS grids are lat-ascending,
    lon 0–360."""
    try:
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
    except Exception:                                   # noqa: BLE001
        return None
    da = ds[list(ds.data_vars)[0]].sel(latitude=slice(-LAT_BAND, LAT_BAND))
    w = np.cos(np.deg2rad(da.latitude))
    band = da.weighted(w).mean("latitude")
    return band.interp(longitude=LON_GRID).values


def _grab(date: str, cyc: str, lead: int, var: str) -> np.ndarray | None:
    p = fetch(date, cyc, lead, var)
    if p is None:
        return None
    band = band_mean(p)
    p.unlink(missing_ok=True)
    return band


def collect(date: str, cyc: str) -> dict | None:
    """Band-mean forecast rows, or None if the cycle is incomplete.

    u:   instantaneous daily steps (24..240 h)              -> (nday, nlon)
    olr: daily means of every 3-hourly step in each forecast
         day ((d-1)*24, d*24]                                -> (nday, nlon)
    """
    u_rows, olr_rows, leads_ok = [], [], []
    for lead in LEADS:
        u = _grab(date, cyc, lead, "u")
        if u is None:
            print(f"  {date} {cyc}Z +{lead:03d}h: u10 missing — skipped",
                  flush=True)
            continue
        samples = [b for h in range(lead - 24 + OLR_STEP, lead + 1, OLR_STEP)
                   if (b := _grab(date, cyc, h, "olr")) is not None]
        if len(samples) < MIN_OLR_SAMPLES:
            print(f"  {date} {cyc}Z day {lead // 24}: only {len(samples)} OLR "
                  f"steps — skipped", flush=True)
            continue
        u_rows.append(u)
        olr_rows.append(np.mean(samples, axis=0))
        leads_ok.append(lead)
        print(f"  {date} {cyc}Z day {lead // 24}: u10 + {len(samples)}-step "
              f"OLR mean", flush=True)
    if len(leads_ok) < MIN_LEADS:
        print(f"  {date} {cyc}Z: only {len(leads_ok)} days — cycle incomplete",
              flush=True)
        return None
    return {"u": np.array(u_rows), "olr": np.array(olr_rows), "leads": leads_ok}


def _lon_ticks():
    ticks = [60, 120, 180, 240]
    labs = [f"{t}°E" if t <= 180 else f"{360 - t}°W" for t in ticks]
    return ticks, labs


def _ref_map(ax, title):
    """Tropical-belt basemap atop one column, same longitude span as the
    Hovmöller below (mirrors eq_hovmoller.py)."""
    ax.set_extent([LON_VIEW[0], LON_VIEW[1], -15, 15], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555",
                   linewidth=0.3, zorder=3)
    ax.add_patch(plt.Rectangle((LON_VIEW[0], -LAT_BAND),
                 LON_VIEW[1] - LON_VIEW[0], 2 * LAT_BAND,
                 transform=ccrs.PlateCarree(), facecolor="none",
                 edgecolor="k", lw=0.6, ls="--", zorder=4))
    ax.set_title(title, fontsize=9)


def _hov_axes(fig, ncol, suptitle):
    gs = fig.add_gridspec(2, ncol, height_ratios=[0.85, 6.5], hspace=0.05,
                          wspace=0.10, left=0.07, right=0.89, top=0.88,
                          bottom=0.07)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    return gs


def _style_hov(ax, lead_days, j):
    ax.set_ylim(max(lead_days), min(lead_days))
    ax.set_xticks(*_lon_ticks())
    ax.tick_params(labelsize=7.5)
    ax.axvline(180, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Forecast lead (days)" if j == 0 else "")
    if j > 0:
        ax.set_yticklabels([])


def plot_wind(anom, absu, lead_days, init, out: Path):
    m = (LON_GRID >= LON_VIEW[0]) & (LON_GRID <= LON_VIEW[1])
    lons = LON_GRID[m]
    fig = plt.figure(figsize=(6.7, 8.8))
    gs = _hov_axes(fig, 2,
                   f"Equatorial Pacific 10 m zonal wind forecast — init {init:%Y-%m-%d %HZ}\n"
                   f"ECCC GDPS (15 km deterministic) · 5°S–5°N · anomaly vs ERA5 1991–2020")
    for j, (fld, kind) in enumerate([(anom[:, m], "anomaly"),
                                     (absu[:, m], "absolute u10")]):
        _ref_map(fig.add_subplot(gs[0, j],
                 projection=ccrs.PlateCarree(central_longitude=180)),
                 f"GDPS\n{kind}")
        ax = fig.add_subplot(gs[1, j])
        if kind == "anomaly":
            im = ax.contourf(lons, lead_days, fld,
                             levels=np.arange(-U_ANOM_LIM, U_ANOM_LIM + .01, 1.5),
                             cmap="RdBu_r", extend="both",
                             norm=mcolors.TwoSlopeNorm(0, -U_ANOM_LIM, U_ANOM_LIM))
            cax = fig.add_axes([0.905, 0.52, 0.015, 0.34])
            lab = "u anomaly (m s⁻¹)"
        else:
            im = ax.contourf(lons, lead_days, fld,
                             levels=np.arange(-U_ABS_LIM, U_ABS_LIM + .01, 1),
                             cmap="RdBu_r", extend="both",
                             norm=mcolors.TwoSlopeNorm(0, -U_ABS_LIM, U_ABS_LIM))
            cax = fig.add_axes([0.905, 0.09, 0.015, 0.34])
            lab = "u10 (m s⁻¹)"
        ax.contour(lons, lead_days, fld, levels=[0], colors="k",
                   linewidths=0.5, alpha=0.5)
        _style_hov(ax, lead_days, j)
        c = fig.colorbar(im, cax=cax, extend="both")
        c.set_label(lab, fontsize=8); c.ax.tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def plot_olr(anom, absf, lead_days, init, out: Path):
    m = (LON_GRID >= LON_VIEW[0]) & (LON_GRID <= LON_VIEW[1])
    lons = LON_GRID[m]
    fig = plt.figure(figsize=(6.7, 8.8))
    gs = _hov_axes(fig, 2,
                   f"Tropical OLR forecast — init {init:%Y-%m-%d %HZ}\n"
                   f"ECCC GDPS top-of-atmosphere OLR · 5°S–5°N · anomaly vs NOAA OLR 1979–2022")
    for j, (fld, kind) in enumerate([(anom[:, m], "anomaly"),
                                     (absf[:, m], "absolute OLR")]):
        _ref_map(fig.add_subplot(gs[0, j],
                 projection=ccrs.PlateCarree(central_longitude=180)),
                 f"GDPS\n{kind}")
        ax = fig.add_subplot(gs[1, j])
        if kind == "anomaly":
            # site OLR-anomaly convention: green = enhanced convection (low
            # OLR), brown = suppressed (olr_waves.py)
            im = ax.contourf(lons, lead_days, fld, levels=OLR_ANOM_LEV,
                             cmap="BrBG_r", extend="both",
                             norm=mcolors.BoundaryNorm(OLR_ANOM_LEV, 256))
            cax = fig.add_axes([0.905, 0.52, 0.015, 0.34])
            lab = "OLR anomaly (W m⁻²)"
        else:
            # site absolute-OLR convention: Spectral, convection warm-coloured
            # (synthetic_olr.py), 180/220 W m⁻² convective contours
            cmap = plt.get_cmap("Spectral")
            im = ax.contourf(lons, lead_days, fld, levels=OLR_LEV, cmap=cmap,
                             norm=mcolors.BoundaryNorm(OLR_LEV, cmap.N),
                             extend="both")
            ax.contour(lons, lead_days, fld, levels=[180, 220], colors="k",
                       linewidths=0.5, alpha=0.5)
            cax = fig.add_axes([0.905, 0.09, 0.015, 0.34])
            lab = "OLR (W m⁻²)"
        _style_hov(ax, lead_days, j)
        c = fig.colorbar(im, cax=cax, extend="both")
        c.set_label(lab, fontsize=8); c.ax.tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def olr_clim_on_grid(doys: np.ndarray) -> np.ndarray:
    """clim_05 (2.5° lons, 0–357.5) → (nlead, LON_GRID), wrapping at 360."""
    ds = xr.open_dataset(OLR_CLIM)
    c = ds["clim_05"].sel(dayofyear=xr.DataArray(doys, dims="lead"))
    lon = np.append(c["lon"].values, 360.0)
    vals = np.concatenate([c.values, c.values[:, :1]], axis=1)   # wrap 0° → 360°
    out = np.empty((len(doys), LON_GRID.size))
    for i in range(len(doys)):
        out[i] = np.interp(LON_GRID, lon, vals[i])
    return out


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

    lead_days = np.array(data["leads"]) / 24.0
    valid = [init + pd.Timedelta(hours=h) for h in data["leads"]]
    doys_u = np.array([v.dayofyear for v in valid])
    # each OLR row is the mean over (lead-24, lead], centered 12 h earlier
    doys_olr = np.array([(v - pd.Timedelta(hours=12)).dayofyear for v in valid])

    u_clim = eval_clim(xr.open_dataarray(U_CLIM).values, doys_u)   # (nlead, 360)
    plot_wind(data["u"] - u_clim, data["u"], lead_days, init,
              ASSETS / "gdps_eq_wind.webp")
    plot_olr(data["olr"] - olr_clim_on_grid(doys_olr), data["olr"], lead_days,
             init, ASSETS / "gdps_olr.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
