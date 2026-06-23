#!/usr/bin/env python3
"""Synthetic equatorial OLR Hovmöller from GMGSI longwave-IR — real-time convection.

NOAA's interpolated-OLR product ended in 2022, so this derives an OLR proxy in real time
from the GMGSI global longwave-IR brightness-temperature mosaic (AWS, hourly, seamless
lat/lon). The 0-255 byte is the McIDAS IR calibration → brightness temperature, then the
Ohring-Gruber flux-temperature relation → OLR. The 5°S-5°N zonal mean is appended to a
rolling store and drawn as a longitude × time Hovmöller; low OLR (cold cloud tops) = deep
convection, so its eastward march tracks the MJO and the warm-pool/dateline convection
shift during El Niño.

    python scripts/sst/synthetic_olr.py                 # append latest hours + render
    python scripts/sst/synthetic_olr.py --bootstrap 21  # backfill N days (2/day) first
"""
from __future__ import annotations

import argparse
import re
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm

HERE = Path(__file__).resolve().parent
STORE = HERE / "metar" / "olr_hov.nc"
S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
LON0, LON1 = 40, 300                 # Indo-Pacific Hovmöller window (°E)
KEEP_DAYS = 45
HOURS = (0, 6, 12, 18)               # representative hours averaged into each daily mean
RECENT = 1                           # recompute today + yesterday each run (catch late data)
SIGMA = 5.67e-8
OLR_LEV = list(range(100, 301, 10))  # W/m²


def _olr_lon(dt: datetime) -> pd.Series | None:
    """GMGSI LW for the hour nearest dt → 5°S-5°N-mean synthetic OLR per 1° longitude."""
    pre = f"GMGSI_LW/{dt:%Y/%m/%d/%H}/"
    try:
        xml = urllib.request.urlopen(f"{S3}/?list-type=2&prefix={pre}&max-keys=5", timeout=30).read().decode()
    except Exception:
        return None
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    if not keys:
        return None
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(urllib.request.urlopen(f"{S3}/{keys[0]}", timeout=90).read()); tf.flush()
        d = xr.open_dataset(tf.name)
        b = d["data"].isel(time=0).astype("float32")
        lat = d["lat"]; lon = d["lon"]
        Tb = xr.where(b > 176, 418.0 - b, (660.0 - b) / 2.0)       # McIDAS IR byte → K
        band = Tb.where((lat >= -5) & (lat <= 5)).mean("yc")        # (xc,) 5S-5N mean
        Tf = band * (1.228 - 1.106e-3 * band)
        olr = (SIGMA * Tf ** 4).values
        lon1d = lon.isel(yc=lon.sizes["yc"] // 2).values % 360
    s = pd.Series(olr, index=lon1d).groupby(np.round(pd.Index(lon1d)).astype(int)).mean()
    s = s[(s.index >= LON0) & (s.index <= LON1)]
    s.index.name = "lon"
    return s


def load_store() -> xr.DataArray | None:
    if STORE.exists():
        return xr.open_dataarray(STORE)
    return None


def build_days(da, dates):
    """(Re)compute the daily-mean synthetic OLR(lon) for each date (mean over HOURS) and
    upsert it into the rolling daily store. Keeps the committed store tiny (~50 KB)."""
    new_t, new_v = [], []
    for dd in dates:
        vals = []
        for h in HOURS:
            s = _olr_lon(datetime(dd.year, dd.month, dd.day, h, tzinfo=timezone.utc))
            if s is not None:
                vals.append(s.reindex(range(LON0, LON1 + 1)).interpolate().values)
        if not vals:
            continue
        new_t.append(np.datetime64(dd, "D")); new_v.append(np.nanmean(vals, axis=0))
        print(f"  {dd}: {len(vals)} hr · OLR min {np.nanmin(new_v[-1]):.0f} W/m²", flush=True)
    if not new_t:
        return da
    new = xr.DataArray(new_v, dims=("time", "lon"),
                       coords={"time": new_t, "lon": list(range(LON0, LON1 + 1))})
    if da is not None:
        da = da.sel(time=~da.time.isin(new.time))          # drop the days we just recomputed
        da = xr.concat([da, new], "time")
    else:
        da = new
    da = da.assign_coords(time=da["time"].dt.floor("D"))           # snap to whole days
    da = da.sel(time=~da.get_index("time").duplicated(keep="last"))
    da = da.sortby("time")
    cutoff = np.datetime64(datetime.utcnow().date()) - np.timedelta64(KEEP_DAYS, "D")
    return da.sel(time=slice(cutoff, None))


def render(da, out: Path):
    daily = da.dropna("time", how="all")
    t = pd.to_datetime(daily.time.values); lon = daily.lon.values
    cmap = plt.get_cmap("Spectral")   # low OLR (deep convection) → yellow/orange/red, high OLR → blue
    fig, ax = plt.subplots(figsize=(8.6, 9))
    pm = ax.contourf(lon, t, daily.values, levels=OLR_LEV, cmap=cmap,
                     norm=BoundaryNorm(OLR_LEV, cmap.N), extend="both")
    ax.contour(lon, t, daily.values, levels=[180, 220], colors="k", linewidths=0.5, alpha=0.5)
    for L, nm in ((100, "Mar. Cont."), (180, "Dateline"), (240, "E. Pac.")):
        ax.axvline(L, color="0.5", lw=0.5, ls=":")
    ax.set_xlim(LON0, LON1); ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("date"); ax.invert_yaxis()
    ax.set_xticks([60, 90, 120, 150, 180, 210, 240, 270])
    ax.set_xticklabels(["60E", "90E", "120E", "150E", "180", "150W", "120W", "90W"])
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.set_title("Synthetic equatorial OLR (5°S–5°N) — GMGSI longwave IR\n"
                 "low OLR (yellow/red) = deep convection · eastward tilt = MJO", fontsize=10)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06, aspect=40, extend="both")
    cb.set_label("OLR (W m⁻²)"); cb.ax.invert_xaxis()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out} ({daily.sizes['time']} days)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/olr_hovmoller.webp")
    ap.add_argument("--bootstrap", type=int, default=0, help="backfill N days (2/day) before appending")
    args = ap.parse_args(argv)
    da = load_store()
    today = datetime.utcnow().date()
    days = [today - timedelta(days=k) for k in range(max(args.bootstrap, RECENT), -1, -1)]
    da = build_days(da, days)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(STORE)
    render(da, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
