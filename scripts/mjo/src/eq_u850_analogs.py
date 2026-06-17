#!/usr/bin/env python3
"""Analog El Niño 850 mb zonal-wind Hovmöllers (5°S–5°N), for the El Niño monitor.

For each selected El Niño, a daily longitude × (months-from-peak) Hovmöller of the
equatorial 850 hPa u-wind, in two flavours:
  • anomaly — vs the ERA5 day-of-year climatology, with the per-longitude secular
    linear trend removed ("detrended band mean"), so 1982 and 2023 compare on a
    warming-free basis;
  • absolute — the raw daily band-mean wind (trades vs westerly bursts).

Source: data/reference/eq_u850_bandseries.nc (WeatherBench2 ERA5 1.5°, 1959–2023,
built by build_u850_bandseries.py); the 2023–24 event, past WB2's end, is read from
the native-0.25° ARCO tail (eq_u850_2024_arco.nc). Band-averaging makes the two
sources agree to <0.1 m/s, so they're directly comparable.

    python src/eq_u850_analogs.py --out-anom plots/u850_analogs_anom.webp \
                                  --out-abs  plots/u850_analogs_abs.webp
"""
from __future__ import annotations

import argparse
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

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
WB2_NC = REF / "eq_u850_bandseries.nc"
CUR_YEAR = pd.Timestamp.now(tz="UTC").year        # the "current" event = this calendar year
SRC_FILES = {                                     # post-WB2 tail read from native ARCO
    "wb2":     WB2_NC,
    "arco_cur": REF / f"eq_u850_{CUR_YEAR}_arco.nc",
}

# label -> (developing year "year 0", source). Each event is aligned from Jan 1 of its
# developing year; the DJF peak then falls ~12 months in. The current event is in year 0.
EVENTS = {
    "1982-83": (1982, "wb2"),
    "1997-98": (1997, "wb2"),
    "2015-16": (2015, "wb2"),
    f"current ({CUR_YEAR})": (CUR_YEAR, "arco_cur"),  # year 0 = this year (developing); ARCO daily
}
WIN_LO, WIN_HI = 0, 334               # Jan 1 → end of Nov of year 0 (the developing year)
SMOOTH_DAYS = 5                        # centered running-mean window (tame daily noise)
LON_VIEW = (40.0, 290.0)              # Indian Ocean → eastern Pacific
ANOM_LIM, ABS_LIM = 15.0, 18.0        # wide range so only the strongest bursts saturate
DEADBAND = 1.0                        # |anomaly| < this is white (suppress weak noise)


# ── climatology + trend (built from the long WB2 record) ──────────────────────
def _harm_clim(series: xr.DataArray, nharm: int = 3) -> np.ndarray:
    """Smooth day-of-year climatology (mean + nharm annual harmonics) -> (366, nlon)."""
    doy = series.time.dt.dayofyear.values
    raw = series.values                                   # (time, lon)
    nlon = raw.shape[1]
    clim = np.full((366, nlon), np.nan)
    dm = np.array([np.nanmean(raw[doy == d], axis=0) for d in range(1, 367)])  # (366,nlon)
    t = np.arange(366)
    X = [np.ones_like(t, float)]
    for k in range(1, nharm + 1):
        X += [np.cos(2 * np.pi * k * t / 365.25), np.sin(2 * np.pi * k * t / 365.25)]
    X = np.vstack(X).T
    good = np.isfinite(dm).all(1)
    coef, *_ = np.linalg.lstsq(X[good], dm[good], rcond=None)
    return X @ coef                                       # (366, nlon)


def _trend(series: xr.DataArray, clim: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-longitude linear trend of the anomaly (raw − clim) vs decimal year.
    Returns (slope per yr, intercept, ref_year)."""
    doy = series.time.dt.dayofyear.values.clip(1, 366)
    anom = series.values - clim[doy - 1]                  # (time, lon)
    yr = (series.time.dt.year + (series.time.dt.dayofyear - 1) / 365.25).values
    ref = float(yr.mean())
    A = np.vstack([yr - ref, np.ones_like(yr)]).T
    coef, *_ = np.linalg.lstsq(A, np.nan_to_num(anom), rcond=None)   # (2, nlon)
    return coef[0], coef[1], ref


def _decimal_year(dates: pd.DatetimeIndex) -> np.ndarray:
    return (dates.year + (dates.dayofyear - 1) / 365.25).values


# ── per-event extraction on the peak-relative axis ────────────────────────────
def _smooth(a: np.ndarray) -> np.ndarray:
    """Centered SMOOTH_DAYS running mean along time (axis 0), per longitude; edge- and
    NaN-tolerant (partial windows via min_periods)."""
    return pd.DataFrame(a).rolling(SMOOTH_DAYS, center=True, min_periods=1).mean().values


def _event(series: xr.DataArray, peak_year: int, clim: np.ndarray,
           slope: np.ndarray, ref: float) -> dict:
    anchor = pd.Timestamp(f"{peak_year}-01-01")
    lo, hi = anchor + pd.Timedelta(days=WIN_LO), anchor + pd.Timedelta(days=WIN_HI)
    sub = series.sel(time=slice(lo, hi))
    dates = pd.to_datetime(sub.time.values)
    rel = (dates - anchor).days.values
    raw = sub.values                                      # (time, lon)
    doy = dates.dayofyear.values.clip(1, 366)
    detrend = slope * (_decimal_year(dates) - ref)[:, None]
    anom = raw - clim[doy - 1] - detrend                  # detrended anomaly
    raw = _smooth(raw); anom = _smooth(anom)              # 5-day running mean (per longitude)
    grid = np.arange(WIN_LO, WIN_HI + 1)
    def _ongrid(a):
        out = np.full((grid.size, raw.shape[1]), np.nan)
        idx = np.searchsorted(grid, rel)
        ok = (idx >= 0) & (idx < grid.size)
        out[idx[ok]] = a[ok]
        return out
    return {"abs": _ongrid(raw), "anom": _ongrid(anom), "rel": grid}


# ── plot ──────────────────────────────────────────────────────────────────────
def _lon_ticks():
    ticks = [60, 120, 180, 240, 300]
    return ticks, [f"{t}°E" if t <= 180 else f"{360 - t}°W" for t in ticks]


# day-of-year (calendar) y-axis: quarter-month ticks anchored on Jan 1 of the peak
# year (rel-day 0). Negative = developing year, positive = peak/decay year.
# ticks every 2 months across the developing year (Jan → Nov of year 0).
_CAL_POS = np.array([0, 61, 122, 182, 244, 305]) / 30.44      # months from Jan 1 of yr 0
_CAL_LAB = ["Jan⁰", "Mar", "May", "Jul", "Sep", "Nov"]


def _deadband(lim: float, dead: float = DEADBAND, step: float = 0.5):
    """Diverging RdBu_r with a white |x|<dead deadband; values beyond ±lim saturate."""
    pos = np.arange(dead, lim + 1e-9, step)
    bounds = np.concatenate([-pos[::-1], pos])      # gap [-dead, dead] is the centre bin
    n = len(bounds) - 1
    cols = plt.cm.RdBu_r(np.linspace(0, 1, n))
    cols[len(pos) - 1] = [1, 1, 1, 1]               # central bin -> white
    cmap = mcolors.ListedColormap(cols)
    cmap.set_under(plt.cm.RdBu_r(0.0)); cmap.set_over(plt.cm.RdBu_r(1.0))
    return cmap, mcolors.BoundaryNorm(bounds, n)


def _ref_map(ax):
    ax.set_extent([LON_VIEW[0], LON_VIEW[1], -15, 15], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555", linewidth=0.3, zorder=3)
    ax.add_patch(plt.Rectangle((LON_VIEW[0], -5), LON_VIEW[1] - LON_VIEW[0], 10,
                 transform=ccrs.PlateCarree(), facecolor="none", edgecolor="k",
                 lw=0.6, ls="--", zorder=4))


def plot(events: dict, lon: np.ndarray, kind: str, out: Path):
    if kind == "anom":
        cmap, norm = _deadband(ANOM_LIM)
        label = f"u850 anomaly (m s⁻¹) · detrended · |x|<{DEADBAND:g} white"
        extend = "both"
    else:
        cmap, norm = "RdBu_r", mcolors.TwoSlopeNorm(0, -ABS_LIM, ABS_LIM)
        label = "u850 (m s⁻¹) · easterly ↔ westerly"
        extend = "both"
    m = (lon >= LON_VIEW[0]) & (lon <= LON_VIEW[1])
    lons = lon[m]
    names = list(events)
    ncol = len(names)

    fig = plt.figure(figsize=(3.15 * ncol, 9.0))
    gs = fig.add_gridspec(2, ncol, height_ratios=[0.7, 7.5], hspace=0.04,
                          wspace=0.10, left=0.06, right=0.9, top=0.91, bottom=0.06)
    fig.suptitle(f"Equatorial 850 hPa zonal wind — El Niño analogs ({'detrended anomaly' if kind=='anom' else 'absolute'})\n"
                 f"5°S–5°N · developing year (Jan→Nov of year 0) · {SMOOTH_DAYS}-day smoothed · ERA5",
                 fontsize=12, fontweight="bold")

    im = None
    for j, nm in enumerate(names):
        _ref_map(fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree(central_longitude=180)))
        fig.axes[-1].set_title(nm, fontsize=10, fontweight="bold")
        ax = fig.add_subplot(gs[1, j])
        fld = events[nm][kind][:, m]
        rel_mo = events[nm]["rel"] / 30.44
        im = ax.pcolormesh(lons, rel_mo, fld, cmap=cmap, norm=norm, shading="nearest")
        ax.axvline(180, color="0.45", lw=0.5, ls=":")         # dateline
        ax.set_ylim(WIN_HI / 30.44, 0)                        # Jan yr0 at top, time downward
        ax.set_xticks(*_lon_ticks()); ax.tick_params(labelsize=7.5)
        ax.set_yticks(_CAL_POS); ax.set_yticklabels(_CAL_LAB if j == 0 else [], fontsize=8)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Months from Jan 1 of year 0  (developing year)" if j == 0 else "")
    c = fig.colorbar(im, cax=fig.add_axes([0.915, 0.30, 0.014, 0.40]), extend=extend)
    c.set_label(label, fontsize=8); c.ax.tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-anom", default="plots/u850_analogs_anom.webp")
    ap.add_argument("--out-abs", default="plots/u850_analogs_abs.webp")
    args = ap.parse_args()

    wb2 = xr.open_dataarray(WB2_NC)
    lon = wb2.longitude.values
    clim = _harm_clim(wb2)                              # climatology + trend from the long record
    slope, intc, ref = _trend(wb2, clim)
    print(f"clim+trend from WB2 {pd.to_datetime(wb2.time.values[0]):%Y}–"
          f"{pd.to_datetime(wb2.time.values[-1]):%Y}; "
          f"mean |trend| {np.abs(slope).mean()*10:.2f} m/s per decade", flush=True)

    sources = {k: (wb2 if k == "wb2" else xr.open_dataarray(p))
               for k, p in SRC_FILES.items() if k == "wb2" or p.exists()}
    events = {}
    for nm, (yr, src) in EVENTS.items():
        if src not in sources:
            print(f"  {nm}: source '{src}' not built yet — skipped", flush=True); continue
        events[nm] = _event(sources[src], yr, clim, slope, ref)
        a = events[nm]["anom"]
        print(f"  {nm}: peak-yr {yr} ({src})  anom range [{np.nanmin(a):+.1f},{np.nanmax(a):+.1f}]")

    plot(events, lon, "anom", Path(args.out_anom))
    plot(events, lon, "abs", Path(args.out_abs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
