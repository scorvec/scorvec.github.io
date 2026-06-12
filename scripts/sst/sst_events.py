#!/usr/bin/env python3
"""
El Niño event-comparison products: track the current ENSO evolution against the
1997-98, 2015-16, and 2023-24 events at WEEKLY resolution.

Data: OISST daily anomaly files (already anomalies vs 1991-2020) for just the
event years — small, reliable (no OPeNDAP), 7-day-smoothed.

The warming background is removed so events decades apart are comparable:
indices subtract the 20°S–20°N tropical-mean anomaly (RONI-style "relative"
index); the global maps subtract the area-weighted global-mean anomaly.

Products (assets/sst/):
  events_nino34.webp   Niño-3.4 weekly overlay — RONI (bold) + raw ONI (faint)
  events_bars.webp     relative Niño-1+2/3/3.4/4 same-phase flavor bars
  events_maps.webp     matching-phase global anomaly maps (global-mean removed)
"""
from __future__ import annotations

import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
PSL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"

ANALOGS = [1997, 2015, 2023]
ANALOG_COLORS = {1997: "#1f77b4", 2015: "#2ca02c", 2023: "#9467bd"}
SMOOTH = 7                                       # 7-day running mean (weekly)
NINO = {
    "1+2": dict(lat=(-10, 0), lon=(270, 280)),
    "3":   dict(lat=(-5, 5),  lon=(210, 270)),
    "3.4": dict(lat=(-5, 5),  lon=(190, 240)),
    "4":   dict(lat=(-5, 5),  lon=(160, 210)),
}


def _ensure_daily(year: int) -> Path:
    p = DATA / f"sst.day.anom.{year}.nc"
    if not (p.exists() and p.stat().st_size > 0):
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"  downloading sst.day.anom.{year}.nc …", flush=True)
        urllib.request.urlretrieve(f"{PSL}/sst.day.anom.{year}.nc", p)
    return p


def current_year() -> int:
    y = datetime.now(timezone.utc).year
    try:
        _ensure_daily(y)
        return y
    except Exception:
        return y - 1


def _open(years):
    ds = xr.open_mfdataset([_ensure_daily(y) for y in sorted(set(years))],
                           combine="by_coords")
    la = "lat" if "lat" in ds.coords else "latitude"
    lo = "lon" if "lon" in ds.coords else "longitude"
    var = "anom" if "anom" in ds.data_vars else list(ds.data_vars)[0]
    return ds, var, la, lo


def _latslice(ds, la, lo_lat):
    """lat slice honoring the file's latitude ordering (OISST is N→S)."""
    return (slice(lo_lat[1], lo_lat[0]) if float(ds[la][0]) > float(ds[la][-1])
            else slice(lo_lat[0], lo_lat[1]))


_TM_CACHE: dict = {}


def tropical_mean_series(ds, var, la, lo) -> pd.Series:
    """Area-weighted 20°S–20°N mean anomaly (7-day smoothed). Subtracting this
    from a Niño box gives the RELATIVE index (RONI-style): it removes the slowly
    rising tropical background so events decades apart are comparable."""
    key = id(ds)
    if key not in _TM_CACHE:
        sub = ds[var].sel({la: _latslice(ds, la, (-20, 20))}).isel({lo: slice(None, None, 2)})
        w = np.cos(np.deg2rad(sub[la]))
        tm = sub.weighted(w).mean((la, lo)).compute()
        s = pd.Series(tm.values, index=pd.to_datetime(tm.time.values)).dropna()
        _TM_CACHE[key] = s.rolling(SMOOTH, center=True, min_periods=3).mean()
    return _TM_CACHE[key]


def region_daily(ds, var, la, lo, region, relative=False) -> pd.Series:
    b = NINO[region]
    box = ds[var].sel({la: _latslice(ds, la, b["lat"]), lo: slice(*b["lon"])}).mean((la, lo)).compute()
    s = pd.Series(box.values, index=pd.to_datetime(box.time.values)).dropna()
    s = s.rolling(SMOOTH, center=True, min_periods=3).mean()         # weekly-smoothed
    if relative:                                                      # minus 20°S–20°N mean
        s = (s - tropical_mean_series(ds, var, la, lo).reindex(s.index)).dropna()
    return s


def _sigma_doy(key):
    """Smooth σ(day-of-year 1..366) from the 12 monthly σ in roni_sigma.json (periodic interp).

    key = 'sigma_by_month' (RONI) or 'sigma_oni_by_month' (ONI). Returns a length-366 array
    (indexed by doy−1) or None if the table is absent → the chart falls back to raw °C.
    """
    import json
    p = HERE / "roni_sigma.json"
    if not p.exists():
        return None
    tab = json.loads(p.read_text()).get(key)
    if not tab:
        return None
    months = np.arange(1, 13)
    mid = np.array([pd.Timestamp(2001, m, 15).dayofyear for m in months], float)
    vals = np.array([tab[str(m)] for m in months], float)
    x = np.concatenate(([mid[-1] - 365], mid, [mid[0] + 365]))      # periodic wrap (Dec↔Jan)
    y = np.concatenate(([vals[-1]], vals, [vals[0]]))
    return np.interp(np.arange(1, 367), x, y)


def _standardize(series, sigma_doy):
    """Daily series ÷ σ(day-of-year) → standardized (σ units); pass-through if σ unavailable."""
    if sigma_doy is None:
        return series
    doy = pd.DatetimeIndex(series.index).dayofyear.values
    return series / pd.Series(sigma_doy[doy - 1], index=series.index)


# ── 1. Niño-3.4 weekly evolution overlay ──────────────────────────────────────
def overlay_nino34(out: Path):
    y0 = current_year()
    years = [y for yr in ANALOGS for y in (yr, yr + 1)] + [y0]
    ds, var, la, lo = _open(years)
    roni = region_daily(ds, var, la, lo, "3.4", relative=True)   # RONI (primary)
    oni = region_daily(ds, var, la, lo, "3.4")                   # raw ONI (reference)

    # Standardize to unitless σ by the per-calendar-month climatological SD (WMO/CPC method),
    # interpolated to a smooth day-of-year curve so there are no month-boundary steps.
    sig_r, sig_o = _sigma_doy("sigma_by_month"), _sigma_doy("sigma_oni_by_month")
    std = sig_r is not None
    roni, oni = _standardize(roni, sig_r), _standardize(oni, sig_o)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    def devdays(idx, yr):
        return (idx - pd.Timestamp(yr, 1, 1)).days
    def draw(s, yr, col, lw, **kw):
        seg = s[f"{yr}":f"{yr+1}"] if yr in ANALOGS else s[f"{yr}":]
        ax.plot(devdays(seg.index, yr), seg.values, color=col, lw=lw, **kw)
    ODASH = (0, (2, 1.3))                                          # tighter dots → ONI reads better
    for yr in ANALOGS:
        draw(oni, yr, ANALOG_COLORS[yr], 1.4, ls=ODASH, alpha=0.85)          # ONI (now clearly visible)
        draw(roni, yr, ANALOG_COLORS[yr], 1.8, label=f"{yr}–{str(yr+1)[2:]}")  # RONI bold
    draw(oni, y0, "#d62728", 2.0, ls=ODASH, alpha=0.9, zorder=5)
    draw(roni, y0, "#d62728", 3.0, label=f"{y0} (current)", zorder=6)
    ax.axhline(0, color="0.6", lw=0.8)
    for g in (0.5, -0.5):
        ax.axhline(g, color="0.7", lw=0.7, ls="--")
    # month ticks across Yr0 and Yr1
    ticks = [pd.Timestamp(2001, m, 1).dayofyear - 1 for m in (1, 4, 7, 10)]
    ticks = ticks + [t + 365 for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{m}\nYr0" for m in ("Jan", "Apr", "Jul", "Oct")] +
                       [f"{m}\nYr1" for m in ("Jan", "Apr", "Jul", "Oct")], fontsize=8)
    ax.set_xlim(0, 730)
    ax.set_ylabel("standardized index (σ)" if std else "Niño-3.4 index (°C)")
    # style legend: solid = RONI, dotted = ONI
    style_h = [plt.Line2D([], [], color="0.35", lw=2.2, label="RONI (relative)"),
               plt.Line2D([], [], color="0.35", lw=1.6, ls=ODASH, label="ONI (raw)")]
    leg2 = ax.legend(handles=style_h, fontsize=8, loc="lower right", framealpha=0.9)
    ax.add_artist(leg2)
    title = ("Niño-3.4 evolution (7-day mean): standardized RONI vs ONI — current vs. 1997, 2015, 2023"
             if std else
             "Niño-3.4 evolution (7-day mean): RONI vs. ONI — current vs. 1997, 2015, 2023")
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    if std:
        fig.text(0.005, 0.002,
                 "Each index standardized by its per-calendar-month standard deviation "
                 "(WMO/CPC method, 1991–2020 OISST) → unitless σ, comparable across seasons.",
                 fontsize=7, color="#888")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    unit = "σ" if std else "°C"
    print(f"wrote {out} (latest {roni.index[-1]:%Y-%m-%d} RONI {roni.iloc[-1]:+.2f} / "
          f"ONI {oni.reindex([roni.index[-1]]).iloc[0]:+.2f}{unit})")


# ── 2. Multi-region flavor bars ───────────────────────────────────────────────
def bars_multiregion(out: Path):
    y0 = current_year()
    ds, var, la, lo = _open([y0] + ANALOGS)
    regions = list(NINO)
    series = {r: region_daily(ds, var, la, lo, r, relative=True) for r in regions}
    latest = series["3.4"].dropna().index[-1]
    doy = latest.dayofyear

    def at_phase(s, yr):
        target = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(doy) - 1)
        i = s.index.get_indexer([target], method="nearest")[0]
        return float(s.iloc[i])
    events = [(y0, f"current ({latest:%b %d})", "#d62728")] + \
             [(y, f"{y}", ANALOG_COLORS[y]) for y in ANALOGS]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(regions)); w = 0.2
    for k, (yr, lab, col) in enumerate(events):
        ax.bar(x + (k - 1.5) * w, [at_phase(series[r], yr) for r in regions], w,
               color=col, label=lab)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Niño-{r}" for r in regions])
    ax.set_ylabel("Relative SST anomaly (°C)\n(minus 20°S–20°N mean)")
    ax.set_title(f"Relative Niño-region anomalies at the same phase (~{latest:%b %d}) — current vs. analogs",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, ncol=2); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out} (phase ~{latest:%b %d})")


# ── 3. Matching-phase global anomaly maps ─────────────────────────────────────
def matching_phase_maps(out: Path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    y0 = current_year()
    ds, var, la, lo = _open([y0] + ANALOGS)
    latest = pd.to_datetime(ds[var].dropna("time", how="all").time.values[-1])
    doy = latest.dayofyear
    years = [y0] + ANALOGS
    labels = [f"{y0} (current)"] + [str(y) for y in ANALOGS]

    def week_map(yr):
        end = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(doy) - 1)
        sl = ds[var].sel(time=slice(end - pd.Timedelta(days=SMOOTH - 1), end)).mean("time").compute()
        gm = float(sl.weighted(np.cos(np.deg2rad(sl[la]))).mean((la, lo)))  # area-wtd global mean
        return sl - gm                                                       # detrended (relative)

    PC = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), subplot_kw=dict(projection=proj))
    fig.suptitle(f"Relative SST anomaly (global-mean removed) — week ending ~{latest:%b %d} of each event",
                 fontsize=13, fontweight="bold")
    im = None
    for ax, yr, lab in zip(axes.ravel(), years, labels):
        f = week_map(yr)
        ax.set_extent((-180, 180, -60, 60), crs=PC)
        im = ax.pcolormesh(f[lo].values, f[la].values, f.values, cmap="RdBu_r",
                           vmin=-3, vmax=3, transform=PC, shading="auto", rasterized=True)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555", linewidth=0.3)
        ax.set_title(lab, fontsize=10)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.014, 0.7]),
                 extend="both").set_label("SST anomaly − global mean (°C)", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.04, wspace=0.05, hspace=0.15)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out} (week ending ~{latest:%b %d})")


if __name__ == "__main__":
    overlay_nino34(ASSETS / "events_nino34.webp")
    bars_multiregion(ASSETS / "events_bars.webp")
    matching_phase_maps(ASSETS / "events_maps.webp")
