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
    dict(y0=1991, label="1991–92", peak="+1.7"),
    dict(y0=1997, label="1997–98", peak="+2.4"),
    dict(y0=2015, label="2015–16", peak="+2.6"),
    dict(y0=2023, label="2023–24", peak="+2.0"),
    dict(y0=2026, label="2026–27 (current)", peak=None, current=True),
]
# Jun(yr0) ... May(yr1): the ENSO-cycle month order shown in the dropdown.
CYCLE_MONTHS = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]

TREND_Y0 = 1950            # fit window start for the trend-adjusted normal
EXTENT = (100, 350, -15, 80)   # Pacific + North America

# CPC official monthly teleconnection indices (standardized), 1950-present.
CPC_TELE = {
    "AO": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
          "daily_ao_index/monthly.ao.index.b50.current.ascii",
    "NAO": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
           "norm.nao.monthly.b5001.current.ascii",
    "PNA": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
           "norm.pna.monthly.b5001.current.ascii",
}
# EPO has no official CPC monthly product; computed from our ERA5 z500
# trend-adjusted anomaly with the Riddle et al. (2013) box definition:
# std. anomaly (20-35N, 160-125W) minus (55-65N, 160-125W), standardized
# per calendar month. Positive EPO = trough near Alaska / progressive
# Pacific flow; negative = Alaska ridge (cold delivery into central US).
EPO_S = dict(lat=slice(20, 35), lon=slice(200, 235))
EPO_N = dict(lat=slice(55, 65), lon=slice(200, 235))


def fetch_teleconnections() -> dict:
    """{(year, month): {'AO': v, 'NAO': v, 'PNA': v}} from CPC ASCII."""
    import urllib.request
    out = {}
    for name, url in CPC_TELE.items():
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
        except Exception as e:                               # noqa: BLE001
            print(f"  WARN: CPC {name} fetch failed ({e}); values omitted")
            continue
        for line in text.splitlines():
            p = line.split()
            if len(p) >= 3:
                try:
                    y, m, v = int(p[0]), int(p[1]), float(p[2])
                except ValueError:
                    continue
                out.setdefault((y, m), {})[name] = v
    return out


def compute_epo(anom: xr.DataArray) -> dict:
    """{(year, month): epo} from the z500 trend-adjusted anomaly."""
    lat = anom.lat.values
    a = anom.sortby("lat") if lat[0] > lat[-1] else anom
    s = a.sel(**EPO_S).mean(("lat", "lon"))
    n = a.sel(**EPO_N).mean(("lat", "lon"))
    raw = (s - n)
    t = pd.to_datetime(raw.time.values)
    vals = raw.values
    out = {}
    for m in range(1, 13):
        rows = t.month == m
        sd = np.nanstd(vals[rows])
        for i in np.where(rows)[0]:
            if np.isfinite(vals[i]) and sd > 0:
                out[(t[i].year, t[i].month)] = float(vals[i] / sd)
    return out


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


def _tele_text(tele: dict, year: int, month: int) -> str | None:
    v = tele.get((year, month))
    if not v:
        return None
    parts = []
    for k in ("PNA", "NAO", "EPO", "AO"):
        parts.append(f"{k} {v[k]:+.1f}" if k in v and v[k] is not None
                     else f"{k} —")
    return "  ·  ".join(parts)


def render(vid: str, anom: xr.DataArray, month: int, avail: set,
           tele: dict) -> None:
    spec = VARS[vid]
    proj = ccrs.PlateCarree(central_longitude=200)
    # panels are wide (250 deg x 95 deg): size the figure so rows pack tight
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.9),
                             subplot_kw=dict(projection=proj))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10,
                        hspace=0.28, wspace=0.05)
    axes = axes.ravel()
    mname = datetime(2000, month, 1).strftime("%B")
    fig.suptitle(f"{spec['label']} ({spec['units']}) — {mname} of the ENSO cycle",
                 fontsize=14, fontweight="bold", y=0.985)
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
        tt = _tele_text(tele, year, month)
        if tt:
            ax.text(0.012, 0.035, tt, transform=ax.transAxes, fontsize=7.8,
                    va="bottom", ha="left", color="#111", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.82,
                              edgecolor="#999", lw=0.4,
                              boxstyle="round,pad=0.25"))

    if cf is not None:
        cax = fig.add_axes([0.18, 0.055, 0.64, 0.018])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=9)
    fig.text(0.5, 0.008,
             "ERA5 monthly means · trend-adjusted anomalies (per-gridpoint "
             f"{TREND_Y0}–present fit per month) · PNA/NAO/AO: CPC monthly "
             "indices · EPO: computed from ERA5 z500 (Riddle et al. 2013 boxes)",
             ha="center", fontsize=8.5, color="#666")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"atmos_{vid}_{month:02d}.webp", dpi=105,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    manifest = {"events": [], "updated":
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    tele = fetch_teleconnections()
    avail_all = None
    for vid in VARS:                        # z500 first: it also feeds EPO
        da = fetch(vid)
        t = pd.to_datetime(da.time.values)
        avail = {(y, m) for y, m in zip(t.year, t.month)}
        avail_all = avail if avail_all is None else (avail_all & avail)
        print(f"  {vid}: trend-adjusting …")
        anom = trend_adjusted_anom(da)
        if vid == "z500":
            for ym, epo in compute_epo(anom).items():
                tele.setdefault(ym, {})["EPO"] = epo
        for month in CYCLE_MONTHS:
            render(vid, anom, month, avail, tele)
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
