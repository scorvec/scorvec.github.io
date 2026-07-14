#!/usr/bin/env python3
"""
Atmospheric fingerprints of the "super" El Niño analogs.

For each calendar month of the ENSO cycle (Jun of the development year through
May of the following year) render a multi-panel map comparing the monthly
anomaly pattern of the current event against the 1982-83, 1997-98, 2015-16 and
2023-24 El Niños, for three variables:

    z500   500 hPa geopotential height (m)
    mslp   sea-level pressure (hPa)
    t2m    2 m air temperature (degC)

Data: ERA5 monthly means via CDS (1950-present; ERA5T gives last month within
~5 days). NCEP R1 was retired in March 2026, so PSL's monthly files froze —
ERA5 is the maintained (and better) record. Anomalies are TREND-ADJUSTED: at
each gridpoint a linear fit per calendar month over 1950-present is removed,
so each event is measured against its own era's expected climate — the same
philosophy as the site's detrended city-temperature and u850 analogs
(otherwise 1982 panels are mostly "colder era", not ENSO).

Outputs:
    assets/sst/analogs/atmos_{var}_{mm:02d}.webp   one per variable x month
    assets/sst/analogs/atmos_manifest.json         months available per event

Cache: scripts/sst/data/era5_{var}_mon.nc — global 2 deg monthly means
(~15-60 MB each); a CDS re-request happens only when a newer month should
exist than the cache holds.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parents[1])
OUT = SITE_ROOT / "assets" / "sst" / "analogs"
DATA = HERE / "data"

VARS = {
    "z500": dict(cds_var="geopotential", pressure="500",
                 label="500 hPa height anomaly", units="m",
                 levels=np.arange(-160, 160.1, 20.0), cmap="RdBu_r"),
    "mslp": dict(cds_var="mean_sea_level_pressure", pressure=None,
                 label="Sea-level pressure anomaly", units="hPa",
                 levels=np.arange(-12, 12.01, 1.5), cmap="RdBu_r"),
    "t2m": dict(cds_var="2m_temperature", pressure=None,
                 label="2 m temperature anomaly", units="°C",
                 levels=np.arange(-6, 6.01, 0.75), cmap="RdBu_r"),
}

# ENSO development years of the comparable "super" events + the current event.
EVENTS = [
    dict(y0=1982, label="1982–83", peak="+2.2"),
    dict(y0=1997, label="1997–98", peak="+2.4"),
    dict(y0=2015, label="2015–16", peak="+2.6"),
    dict(y0=2023, label="2023–24", peak="+2.0"),
    dict(y0=2026, label="2026–27 (current)", peak=None, current=True),
]
# Jun(yr0) ... May(yr1): the ENSO-cycle month order shown in the dropdown.
CYCLE_MONTHS = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]

TREND_Y0 = 1950            # fit window start for the trend-adjusted normal
EXTENT = (100, 350, -15, 80)   # Pacific + North America


def _expected_last_month() -> pd.Timestamp:
    """Newest month ERA5T should have (~5-day publication lag)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=6)
    return pd.Timestamp(now.year, now.month, 1) - pd.offsets.MonthBegin(1)


def fetch(vid: str) -> xr.DataArray:
    """Global 2-degree ERA5 monthly means 1950-present, cached; a CDS
    request runs only when a newer month should exist than the cache has."""
    spec = VARS[vid]
    cache = DATA / f"era5_{vid}_mon.nc"
    want = _expected_last_month()
    if cache.exists():
        da = xr.open_dataarray(cache).load()
        have = pd.Timestamp(da.time.values[-1]).normalize().replace(day=1)
        if have >= want:
            print(f"  {vid}: cache current thru {have:%Y-%m}")
            return da
        da.close()
    print(f"  {vid}: CDS request thru {want:%Y-%m} …")
    import cdsapi
    c = cdsapi.Client(quiet=True)
    dataset = ("reanalysis-era5-pressure-levels-monthly-means"
               if spec["pressure"] else
               "reanalysis-era5-single-levels-monthly-means")
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [spec["cds_var"]],
        "year": [str(y) for y in range(TREND_Y0, want.year + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "grid": ["2.0/2.0"],
        "data_format": "netcdf",
    }
    if spec["pressure"]:
        req["pressure_level"] = [spec["pressure"]]
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".dl.nc")
    c.retrieve(dataset, req, str(tmp))
    ds = xr.open_dataset(tmp)
    name = [v for v in ds.data_vars if ds[v].ndim >= 3][0]
    da = ds[name].squeeze(drop=True).load()
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    da = da.rename({tdim: "time"} if tdim != "time" else {})
    for old, new in (("latitude", "lat"), ("longitude", "lon")):
        if old in da.dims:
            da = da.rename({old: new})
    if vid == "z500":
        da = da / 9.80665                      # geopotential -> meters
    elif vid == "mslp":
        da = da / 100.0                        # Pa -> hPa
    elif vid == "t2m":
        da = da - 273.15                       # K -> degC
    da.name = vid
    ds.close()
    tmp2 = cache.with_suffix(".tmp.nc")
    da.to_netcdf(tmp2)
    tmp2.replace(cache)
    tmp.unlink(missing_ok=True)
    print(f"  {vid}: cached {len(da.time)} months")
    return da


def trend_adjusted_anom(da: xr.DataArray) -> xr.DataArray:
    """Anomaly vs a per-gridpoint, per-calendar-month linear trend fit
    (TREND_Y0-present): each map is the departure from what that era's
    climate would predict for that month."""
    t = pd.to_datetime(da.time.values)
    out = np.full(da.shape, np.nan, dtype=np.float32)
    V = da.values.reshape(len(t), -1)
    for m in range(1, 13):
        rows = np.where((t.month == m) & (t.year >= TREND_Y0))[0]
        yrs = t.year.values[rows].astype(float)
        Y = V[rows]
        good = np.isfinite(Y).all(axis=0)
        # polyfit over years, all gridpoints at once
        A = np.column_stack([yrs, np.ones_like(yrs)])
        coef, *_ = np.linalg.lstsq(A, Y[:, good], rcond=None)
        fit = A @ coef
        res = np.full_like(Y, np.nan)
        res[:, good] = Y[:, good] - fit
        out.reshape(len(t), -1)[rows] = res
    return xr.DataArray(out, dims=da.dims, coords=da.coords)


def month_field(anom: xr.DataArray, year: int, month: int):
    sel = anom.sel(time=f"{year}-{month:02d}")
    return sel.isel(time=0) if "time" in sel.dims else sel


def render(vid: str, anom: xr.DataArray, month: int, avail: set) -> None:
    spec = VARS[vid]
    proj = ccrs.PlateCarree(central_longitude=200)
    # panels are wide (250 deg x 95 deg): size the figure so rows pack tight
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.6),
                             subplot_kw=dict(projection=proj))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.10,
                        hspace=0.28, wspace=0.05)
    axes = axes.ravel()
    mname = datetime(2000, month, 1).strftime("%B")
    cf = None
    for i, ev in enumerate(EVENTS):
        ax = axes[i]
        year = ev["y0"] if month >= 6 else ev["y0"] + 1
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        yr_tag = f"{mname} {year}"
        if (year, month) not in avail:
            ax.set_facecolor("#f2f2f2")
            ax.text(0.5, 0.5, f"{yr_tag}\nnot yet available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="#999")
            ax.coastlines(lw=0.4, color="#bbb")
            ax.set_title(f"{ev['label']} · {yr_tag}", fontsize=11, loc="left")
            continue
        fld = month_field(anom, year, month)
        cf = ax.contourf(fld.lon, fld.lat, fld.values, levels=spec["levels"],
                         cmap=spec["cmap"], extend="both",
                         transform=ccrs.PlateCarree())
        ax.contour(fld.lon, fld.lat, fld.values, levels=[0], colors="#444",
                   linewidths=0.5, transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="#222")
        ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#555")
        peak = f"  (peak ONI {ev['peak']})" if ev["peak"] else ""
        ax.set_title(f"{ev['label']} · {yr_tag}{peak}",
                     fontsize=11, loc="left",
                     fontweight="bold" if ev.get("current") else "normal")

    # info panel in the spare slot
    ax = axes[5]
    ax.set_axis_off()
    ax.text(0.02, 0.95,
            f"{spec['label']}\n{mname} of the ENSO cycle",
            fontsize=15, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.62,
            "ERA5 monthly means (Copernicus CDS).\n"
            "Anomalies are departures from a per-gridpoint,\n"
            f"per-month linear trend fit ({TREND_Y0}–present),\n"
            "so events decades apart are measured against\n"
            "their own era's climate — what remains is the\n"
            "circulation signal, not background warming.",
            fontsize=10.5, color="#444", va="top", transform=ax.transAxes)

    if cf is not None:
        cax = fig.add_axes([0.18, 0.045, 0.64, 0.018])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
        cb.set_label(f"{spec['label']} ({spec['units']})", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"atmos_{vid}_{month:02d}.webp", dpi=105,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    manifest = {"events": [], "updated":
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    avail_all = None
    for vid in VARS:
        da = fetch(vid)
        t = pd.to_datetime(da.time.values)
        avail = {(y, m) for y, m in zip(t.year, t.month)}
        avail_all = avail if avail_all is None else (avail_all & avail)
        print(f"  {vid}: trend-adjusting …")
        anom = trend_adjusted_anom(da)
        for month in CYCLE_MONTHS:
            render(vid, anom, month, avail)
        print(f"  {vid}: rendered {len(CYCLE_MONTHS)} months")

    for ev in EVENTS:
        months = []
        for month in CYCLE_MONTHS:
            year = ev["y0"] if month >= 6 else ev["y0"] + 1
            if (year, month) in avail_all:
                months.append(month)
        manifest["events"].append({"label": ev["label"], "y0": ev["y0"],
                                   "months": months,
                                   "current": bool(ev.get("current"))})
    manifest["vars"] = {k: v["label"] for k, v in VARS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "atmos_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT / 'atmos_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
