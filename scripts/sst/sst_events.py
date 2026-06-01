#!/usr/bin/env python3
"""
El Niño event-comparison products for the monitor: track the current ENSO
evolution against the 1997-98, 2015-16, and 2023-24 events.

Data: OISST monthly mean via PSL OPeNDAP (subset to the Niño boxes / specific
months — no bulk download), anomalies vs the 1991-2020 monthly climatology.

Products (assets/sst/):
  events_nino34.webp   Niño-3.4 monthly evolution overlay (current vs analogs)
  events_bars.webp     Niño-1+2/3/3.4/4 latest-month bars (current vs analogs)
  events_maps.webp     matching-phase global anomaly maps (current vs analogs)
"""
from __future__ import annotations

import os
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
MON_LOCAL = DATA / "sst.mon.mean.nc"
LTM_LOCAL = DATA / "sst.mon.ltm.1991-2020.nc"


def _ensure(local: Path, name: str) -> Path:
    """Download the monthly file once if absent (OPeNDAP proved unreliable)."""
    if local.exists() and local.stat().st_size > 0:
        return local
    import urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {name} …", flush=True)
    urllib.request.urlretrieve(f"{PSL}/{name}", local)
    return local

ANALOGS = [1997, 2015, 2023]
ANALOG_COLORS = {1997: "#1f77b4", 2015: "#2ca02c", 2023: "#9467bd"}
# Niño regions: lon in degE, lat S->N
NINO = {
    "1+2": dict(lat=(-10, 0), lon=(270, 280)),
    "3":   dict(lat=(-5, 5),  lon=(210, 270)),
    "3.4": dict(lat=(-5, 5),  lon=(190, 240)),
    "4":   dict(lat=(-5, 5),  lon=(160, 210)),
}


def _open():
    ds = xr.open_dataset(_ensure(MON_LOCAL, "sst.mon.mean.nc"))
    la = "lat" if "lat" in ds.coords else "latitude"
    lo = "lon" if "lon" in ds.coords else "longitude"
    return ds, la, lo


def region_anom(ds, la, lo, region) -> pd.Series:
    """Monthly anomaly (°C, vs 1991-2020) for a Niño box, full record."""
    b = NINO[region]
    latsel = (slice(b["lat"][1], b["lat"][0]) if float(ds[la][0]) > float(ds[la][-1])
              else slice(b["lat"][0], b["lat"][1]))
    box = ds["sst"].sel({la: latsel, lo: slice(*b["lon"])}).mean((la, lo)).compute()
    s = pd.Series(box.values, index=pd.to_datetime(box.time.values)).dropna()
    s = s[s.abs() > 1.0]                       # drop unfilled (zero) months
    clim = s["1991":"2020"].groupby(lambda d: d.month).mean()
    return s - s.index.month.map(lambda m: clim[m])


def current_year0(s: pd.Series) -> int:
    """Developing-year anchor for the ongoing event = year of the latest month
    (events ramp up through the calendar year toward a DJF peak)."""
    return int(s.index[-1].year)


def event_window(anom: pd.Series, year0: int) -> pd.Series:
    """24 monthly values indexed 1..24 = Jan(year0) .. Dec(year0+1)."""
    idx = pd.date_range(f"{year0}-01-01", f"{year0+1}-12-01", freq="MS")
    vals = anom.reindex(idx)
    return pd.Series(vals.values, index=np.arange(1, 25))


def overlay_nino34(out: Path):
    ds, la, lo = _open()
    a = region_anom(ds, la, lo, "3.4")
    y0 = current_year0(a)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for yr in ANALOGS:
        w = event_window(a, yr)
        ax.plot(w.index, w.values, color=ANALOG_COLORS[yr], lw=1.8,
                label=f"{yr}–{str(yr+1)[2:]}")
    cur = event_window(a, y0).dropna()
    ax.plot(cur.index, cur.values, color="#d62728", lw=3.0, marker="o", ms=4,
            label=f"{y0} (current)", zorder=5)
    ax.axhline(0, color="0.6", lw=0.8)
    for g in (0.5, -0.5):
        ax.axhline(g, color="0.7", lw=0.7, ls="--")
    months = ["Jan", "Apr", "Jul", "Oct"]
    ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
    ax.set_xticklabels([f"{m}\nYr0" if i < 4 else f"{m}\nYr1"
                        for i, m in enumerate(months * 2)], fontsize=8)
    ax.set_xlim(1, 24)
    ax.set_ylabel("Niño-3.4 SST anomaly (°C)")
    ax.set_title("Niño-3.4 evolution: current event vs. 1997, 2015, 2023",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (current year0={y0}, latest {a.index[-1]:%Y-%m} {a.iloc[-1]:+.2f}°C)")


def bars_multiregion(out: Path):
    """Niño-1+2/3/3.4/4 anomalies at the current phase: current vs analogs at
    the same calendar month (shows EP-vs-CP flavor)."""
    ds, la, lo = _open()
    regions = ["1+2", "3", "3.4", "4"]
    series = {r: region_anom(ds, la, lo, r) for r in regions}
    latest = series["3.4"].index[-1]
    y0, m = latest.year, latest.month
    events = [(y0, "current", "#d62728")] + [(y, str(y), ANALOG_COLORS[y]) for y in ANALOGS]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(regions))
    w = 0.2
    for k, (yr, lab, col) in enumerate(events):
        vals = [float(series[r].get(pd.Timestamp(yr, m, 1), np.nan)) for r in regions]
        ax.bar(x + (k - 1.5) * w, vals, w, color=col,
               label=(f"{lab} ({latest:%b %Y})" if lab == "current" else f"{yr} ({pd.Timestamp(yr,m,1):%b})"))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Niño-{r}" for r in regions])
    ax.set_ylabel("SST anomaly (°C)")
    ax.set_title(f"Niño-region anomalies at the same phase ({latest:%B}) — current vs. analogs",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (phase {latest:%Y-%m})")


def matching_phase_maps(out: Path):
    """Global SST-anomaly maps for the current month vs the same calendar month
    in each analog year (2x2)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ds, la, lo = _open()
    ltm = xr.open_dataset(_ensure(LTM_LOCAL, "sst.mon.ltm.1991-2020.nc"))
    lav = "lat" if "lat" in ltm.coords else "latitude"
    latest = pd.to_datetime(ds["sst"].dropna("time", how="all").time.values[-1])
    # use the latest fully-valid month
    m = latest.month
    clim = ltm["sst"].isel(time=m - 1)                       # April climatology map
    years = [latest.year] + ANALOGS
    labels = [f"{latest.year} (current)"] + [f"{y}" for y in ANALOGS]

    PC = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4),
                             subplot_kw=dict(projection=proj))
    fig.suptitle(f"Global SST anomaly — {latest:%B} of each event "
                 f"(current vs 1997/2015/2023)", fontsize=13, fontweight="bold")
    im = None
    for ax, yr, lab in zip(axes.ravel(), years, labels):
        fld = ds["sst"].sel(time=f"{yr}-{m:02d}").squeeze() - clim.values
        ax.set_extent((-180, 180, -60, 60), crs=PC)
        im = ax.pcolormesh(fld[lo].values, fld[la].values, fld.values,
                           cmap="RdBu_r", vmin=-3, vmax=3, transform=PC,
                           shading="auto", rasterized=True)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555", linewidth=0.3)
        ax.set_title(lab, fontsize=10)
    cax = fig.add_axes([0.92, 0.15, 0.014, 0.7])
    fig.colorbar(im, cax=cax, extend="both").set_label("SST anomaly (°C)", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.04, wspace=0.05, hspace=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (matching month {latest:%B})")


if __name__ == "__main__":
    overlay_nino34(ASSETS / "events_nino34.webp")
    bars_multiregion(ASSETS / "events_bars.webp")
    matching_phase_maps(ASSETS / "events_maps.webp")

