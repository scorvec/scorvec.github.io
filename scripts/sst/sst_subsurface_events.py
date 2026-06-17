#!/usr/bin/env python3
"""Subsurface El Niño analog comparison (TAO/TRITON equatorial Pacific).

Compares the current event's equatorial subsurface temperature against the
1997-98, 2015-16 and 2023-24 El Niños, reusing the TAO DISDEL fetch
(tao_subsurface) and the 1991-2020 harmonic climatology (sst_subsurface).

Products (assets/sst/):
  subsurface_events_xsec.webp  matching-phase depth×lon temperature-anomaly
                               cross-sections (2×2: current vs analogs)
  subsurface_events_hc.webp    equatorial 0–300 m temperature-anomaly index
                               through each event's development year + next

TAO mooring coverage varies by year (1997 and 2023 are gappier), so some
longitudes / periods are missing — shown as gaps.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tao_subsurface as tao
import sst_subsurface as ss

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
EVDIR = DATA / "subsurface_events"

ANALOGS = [1997, 2015, 2023]
ANALOG_COLORS = {1997: "#1f77b4", 2015: "#2ca02c", 2023: "#9467bd"}
XSEC_WIN = 14            # days averaged for the matching-phase cross-section
HC_SMOOTH = 15           # days, heat-content running mean


def _anomaly(grid: xr.DataArray) -> xr.DataArray:
    coeffs = ss.load_or_build_coeffs(grid.longitude.values)
    doy = pd.to_datetime(grid.time.values).dayofyear.values
    return xr.DataArray(grid.values - ss.eval_climatology(coeffs, doy),
                        dims=grid.dims, coords=grid.coords)


def current_ds() -> xr.Dataset:
    """Current-year equatorial TAO temps — reuse the daily cache if present
    (produced by the daily sst job), else fetch year-to-date from DISDEL."""
    p = DATA / "tao_eq_temp.nc"
    if p.exists():
        return xr.open_dataset(p)
    import re
    from datetime import timezone
    now = datetime.now(timezone.utc)
    EVDIR.mkdir(parents=True, exist_ok=True)
    ap = EVDIR / f"tao_{now.year}_cur.ascii"
    # Re-fetch the current-year TAO whenever the cache lags the main subsurface file
    # (tao_eq_recent.nc, which run_local_sst re-downloads every run). The old "fetched
    # today" check pinned the analog's "current" panel to a stale date when the first
    # daily run fetched before DISDEL had posted the new day (TAO lags ~2 days). The
    # analog-year ascii are fixed history and stay cached.
    def _last(path: Path):
        try:
            ms = re.findall(r"^\s*(\d{8})\s", path.read_text(errors="replace"), re.M)
            return pd.to_datetime(ms[-1], format="%Y%m%d").date() if ms else None
        except Exception:
            return None
    target = None
    recent = DATA / "tao_eq_recent.nc"
    if recent.exists():
        try:
            target = pd.to_datetime(xr.open_dataset(recent).time.values[-1]).date()
        except Exception:
            target = None
    cached = _last(ap) if (ap.exists() and ap.stat().st_size > 0) else None
    fresh = cached is not None and target is not None and cached >= target
    if not fresh:
        tao.deliver(datetime(now.year, 1, 1), now, ap)
    return tao.parse(ap)


def event_anom(year: int, cur_year: int, cur_ds: xr.Dataset) -> xr.DataArray:
    """Regridded temperature ANOMALY (time, depth, lon) for one event.
    Current year reuses cur_ds; analogs fetch Yr0+Yr1 from DISDEL."""
    if year == cur_year:
        ds = cur_ds
    else:
        EVDIR.mkdir(parents=True, exist_ok=True)
        p = EVDIR / f"tao_{year}_{year + 1}.ascii"
        if not (p.exists() and p.stat().st_size > 0):
            tao.deliver(datetime(year, 1, 1), datetime(year + 1, 12, 31), p)
        ds = tao.parse(p)
    return _anomaly(ss.to_depth_grid(ds))


# ── 1. matching-phase cross-sections ──────────────────────────────────────────
def xsec(events: dict, cur_year: int, match: pd.Timestamp, out: Path, detr: bool = False):
    years = [cur_year] + ANALOGS
    labels = [f"{cur_year} (current)"] + [f"{y}–{str(y + 1)[2:]}" for y in ANALOGS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    im = None
    for ax, yr, lab in zip(axes.ravel(), years, labels):
        a = events[yr]
        end = pd.Timestamp(yr, match.month, match.day)
        win = a.sel(time=slice(end - pd.Timedelta(days=XSEC_WIN - 1), end)).mean("time")
        lons = win.longitude.values
        if len(lons) >= 2:
            Ag = ss.interp_lon(win.values, lons)
            im = ax.contourf(ss.LON_GRID, ss.DEPTH_GRID, Ag, levels=ss.ANOM_LEVELS,
                             cmap="RdBu_r", extend="both")
            ax.contour(ss.LON_GRID, ss.DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=0.6)
        ax.scatter(lons, np.full(len(lons), 6), marker="v", s=18, color="k", zorder=5)
        ax.set_ylim(300, 0)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([165, 180, 200, 220, 240, 260])
        ax.set_xticklabels(["165°E", "180°", "160°W", "140°W", "120°W", "100°W"], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    extra = ("\ndetrended with data from 1991–2020" if detr else "")
    fig.suptitle(f"Equatorial Pacific subsurface temperature anomaly — week of ~{match:%b %d} of each event"
                 f"{extra}\n(triangles = moorings reporting; TAO coverage varies by year)",
                 fontsize=11.5, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, extend="both")
    cb.set_label("Temperature anomaly (°C)", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (matching ~{match:%b %d})")


# ── 2. heat-content (0–300 m T anomaly) evolution overlay ─────────────────────
def heat_content(events: dict, cur_year: int, out: Path, detr: bool = False):
    def hc(a: xr.DataArray) -> pd.Series:
        col = a.sel(depth=slice(0, 300)).mean("depth", skipna=True)   # (time, lon)
        s = pd.Series(col.mean("longitude", skipna=True).values,
                      index=pd.to_datetime(a.time.values)).dropna()
        return s.rolling(HC_SMOOTH, center=True, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    def devdays(idx, yr):
        return (idx - pd.Timestamp(yr, 1, 1)).days
    for yr in ANALOGS:
        s = hc(events[yr])
        ax.plot(devdays(s.index, yr), s.values, color=ANALOG_COLORS[yr], lw=1.8,
                label=f"{yr}–{str(yr + 1)[2:]}")
    cur = hc(events[cur_year])
    ax.plot(devdays(cur.index, cur_year), cur.values, color="#d62728", lw=3.0,
            label=f"{cur_year} (current)", zorder=5)
    ax.axhline(0, color="0.6", lw=0.8)
    ticks = [pd.Timestamp(2001, m, 1).dayofyear - 1 for m in (1, 4, 7, 10)]
    ticks = ticks + [t + 365 for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{m}\nYr0" for m in ("Jan", "Apr", "Jul", "Oct")] +
                       [f"{m}\nYr1" for m in ("Jan", "Apr", "Jul", "Oct")], fontsize=8)
    ax.set_xlim(0, 730)
    ax.set_ylabel("Equatorial 0–300 m temperature anomaly (°C)")
    if detr:
        title = ("Subsurface heat content (eq. Pacific 0–300 m T anomaly)\n"
                 "detrended with data from 1991–2020")
        fs = 10.5
    else:
        title = "Subsurface heat content (eq. Pacific 0–300 m T anomaly): current vs. 1997, 2015, 2023"
        fs = 11.5
    ax.set_title(title, fontsize=fs, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (current latest {cur.dropna().index[-1]:%b %d} {cur.dropna().iloc[-1]:+.2f}°C)")


def main() -> int:
    cur_ds = current_ds()
    match = pd.to_datetime(cur_ds.time.values[-1])
    cur_year = int(match.year)
    events = {y: event_anom(y, cur_year, cur_ds) for y in [cur_year] + ANALOGS}
    xsec(events, cur_year, match, ASSETS / "subsurface_events_xsec.webp")
    heat_content(events, cur_year, ASSETS / "subsurface_events_hc.webp")

    # de-trended companions: remove the 1991–2020 secular trend so the analog
    # comparison isolates the ENSO signal from the background ocean warming/cooling.
    events_dt = {y: ss.detrend(a) for y, a in events.items()}
    xsec(events_dt, cur_year, match, ASSETS / "subsurface_events_xsec_detrended.webp", detr=True)
    heat_content(events_dt, cur_year, ASSETS / "subsurface_events_hc_detrended.webp", detr=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
