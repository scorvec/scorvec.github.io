#!/usr/bin/env python3
"""Append the newest days to the ERA5 850 hPa equatorial u-wind TAIL (ARCO).

build_u850_bandseries.py --source arco rebuilds a whole calendar year every
time it runs: one ~150 MB ARCO chunk per day (all 37 levels ride along), so a
daily refresh in September costs ~38 GB. This reads ONLY the days after the
last one in the existing tail file (typically 1-3 chunks), and appends. A cold
start (no tail file) begins at Jan 1 of the target year, so the first run of a
year is the expensive one — seed the file instead.

The tail feeds build_mei_nowcast.py (the eq-u850 predictor of the daily MEI
nowcast); the historical band series (1959-2023, WB2) is a separate file.

    python src/refresh_u850_tail.py [--year 2026] [--hours 12] [--max-days 10]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
LAT_BAND = 5.0
LON_GRID = np.arange(0.0, 360.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=pd.Timestamp.utcnow().year)
    ap.add_argument("--hours", default="12", help="UTC hours per day to read (12 = one chunk/day)")
    ap.add_argument("--max-days", type=int, default=10, help="cap per run; the rest next time")
    ap.add_argument("--lag-days", type=int, default=5, help="ARCO trails real time by ~5-7 d")
    a = ap.parse_args()
    out = REF / f"eq_u850_{a.year}_arco.nc"
    hours = [int(h) for h in a.hours.split(",")]

    have = None
    if out.exists():
        have = xr.open_dataset(out)["u850"].load()
        # the laptop-built tail spans the whole year with NaN beyond the last
        # real day: keep only complete days and continue from the last one
        have = have.isel(time=np.isfinite(have.values).all(axis=1))
        if have.sizes["time"] == 0:
            have = None
    if have is not None:
        last = pd.Timestamp(have.time.values[-1]).normalize()
        start = last + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(a.year, 1, 1)
    end = min(pd.Timestamp(a.year, 12, 31),
              pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=a.lag_days))
    if start > end:
        print(f"tail {out.name}: up to date ({start - pd.Timedelta(days=1):%Y-%m-%d})")
        return 0
    end = min(end, start + pd.Timedelta(days=a.max_days - 1))
    t0 = time.time()
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=850).sortby("latitude").sel(latitude=slice(-6, 6))
    u = u.sel(time=slice(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d} 23:00"))
    u = u.sel(time=u.time.dt.hour.isin(hours))
    if u.sizes["time"] == 0:
        print(f"tail: ARCO has nothing yet for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
        return 0
    lat = u.latitude
    w = np.cos(np.deg2rad(lat)).where(np.abs(lat) <= LAT_BAND, 0.0)
    band = u.weighted(w).mean("latitude").resample(time="1D").mean().compute()
    if not np.isfinite(band.values).all():             # a day ARCO has not finished yet
        ok = np.isfinite(band.values).all(axis=1)
        band = band.isel(time=ok)
        if band.sizes["time"] == 0:
            print("tail: newest days not complete in ARCO yet"); return 0
    if float(band.longitude.min()) < 0:
        band = band.assign_coords(longitude=band.longitude % 360).sortby("longitude")
    band = band.interp(longitude=LON_GRID)
    band.name = "u850"
    series = band if have is None else xr.concat([have, band], dim="time")
    series = series.sortby("time")
    series = series.isel(time=~pd.Index(series.time.values).duplicated(keep="last"))
    series.attrs.update(source="ARCO ERA5 0.25°, one snapshot/day", band="5S-5N cos-weighted",
                        level="850 hPa")
    tmp = out.with_suffix(".nc.tmp")
    series.to_netcdf(tmp); tmp.replace(out)
    print(f"tail {out.name}: +{band.sizes['time']} day(s) → {series.sizes['time']} "
          f"({pd.Timestamp(series.time.values[0]):%Y-%m-%d}→{pd.Timestamp(series.time.values[-1]):%Y-%m-%d}) "
          f"in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
