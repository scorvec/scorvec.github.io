#!/usr/bin/env python3
"""Daily EPO and WPO from ERA5: the observed reference the GEPS and GEFS teleconnection plumes draw on their tails.

PSL's daily EPO/WPO files were built on NCEP/NCAR R1 and stopped with it on 17 March 2026, so the two indices are
computed here from ERA5 with PSL's definition: the 500 hPa height anomaly in area-weighted (cos lat) box means,
EPO = (20-35N, 160-125W) minus (55-65N, 160-125W), WPO = (25-40N, 140E-150W) minus (50-70N, 140E-150W), in metres.

  heights  ARCO-ERA5 on Google Cloud (anonymous), the 12 UTC analysis of each day: one chunk per day, ERA5T about
           five days behind real time. The daily values are kept in DIR/era5_epo_wpo.csv, so a run reads only the
           days it does not have yet.
  normal   the WeatherBench 2 ERA5 climatology (1990-2019, smoothed by day of year, 12 UTC, 1.5 deg), reduced once
           to the same box differences in DIR/era5_epo_wpo_normal.csv. WeatherBench 2 keys it by the raw day of the
           year (1 March = 61 in a leap year), so it is read the same way.

The callers put the metres into seasonal standard deviations with the projector's own sigma, as they did PSL's.

    python era5_epo_wpo.py --cache DIR      # refresh DIR's series for the last 60 days and print the newest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2_NORMAL = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)
BOXES = {  # (south box, north box), each (lat0, lat1, lon0, lon1) in degrees east
    "epo": ((20, 35, 200, 235), (55, 65, 200, 235)),
    "wpo": ((25, 40, 140, 210), (50, 70, 140, 210)),
}
HOUR = 12
G = 9.80665
NEWEST_LAG = 4  # days before today: nothing newer is ever in ARCO
KEEP_DAYS = 90  # rolling length of the cached series


def box_diff(z, lat, lon) -> dict:
    """{sid: south box mean - north box mean} of a (lat, lon) field, cos-lat weighted."""
    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64") % 360
    out = {}
    for sid, boxes in BOXES.items():
        means = []
        for la0, la1, lo0, lo1 in boxes:
            m = (lat[:, None] >= la0) & (lat[:, None] <= la1) & (lon[None, :] >= lo0) & (lon[None, :] <= lo1)
            w = np.cos(np.deg2rad(lat))[:, None] * m
            means.append(float((np.asarray(z) * w).sum() / w.sum()))
        out[sid] = means[0] - means[1]
    return out


def _open(url):
    import xarray as xr

    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})


def _subset(da):
    """The 15-75N, 135-240E window that holds every box, as (latitude, longitude)."""
    lat = da["latitude"]
    da = da.sel(latitude=slice(75, 15) if float(lat[0]) > float(lat[-1]) else slice(15, 75))
    return da.sel(longitude=slice(135, 240)).transpose(..., "latitude", "longitude")


def normal(cache: Path) -> pd.DataFrame:
    """Box differences of the WeatherBench 2 normal by raw day of year (1..366), metres; read once, then cached."""
    p = cache / "era5_epo_wpo_normal.csv"
    if p.exists():
        return pd.read_csv(p, index_col="doy")
    z = _subset(_open(WB2_NORMAL)["geopotential"].sel(hour=HOUR, level=500)).transpose("dayofyear", ...).load()
    rows = {int(d): box_diff(z.sel(dayofyear=d).values / G, z.latitude, z.longitude) for d in z.dayofyear.values}
    df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("doy").sort_index()
    if list(df.index) != list(range(1, 367)) or not np.isfinite(df.values).all():
        raise RuntimeError(f"WeatherBench 2 normal looks wrong ({len(df)} days)")
    cache.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(p)
    return df


def refresh(cache: Path, first: pd.Timestamp) -> dict:
    """{sid: daily Series of the ERA5 index in metres} from `first` to the newest ERA5 day, the cache extended with
    the days it lacks. Fail-soft: a read error keeps what the cache holds and says so."""
    cache = Path(cache)
    p = cache / "era5_epo_wpo.csv"
    have = pd.read_csv(p, index_col="date", parse_dates=["date"]) if p.exists() else pd.DataFrame()
    first = pd.Timestamp(first).normalize()
    newest = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=NEWEST_LAG)
    need = [d for d in pd.date_range(first, newest) if d not in have.index]
    if need:
        try:
            clim = normal(cache)
            z500 = _subset(_open(ARCO)["geopotential"].sel(level=500))
            on_axis = set(pd.DatetimeIndex(z500.time.values))
            rows = {}
            for d in need:  # the axis runs ahead of ERA5T: those days are NaN-filled and cost no download
                t = d + pd.Timedelta(hours=HOUR)
                if t not in on_axis:
                    continue
                f = z500.sel(time=t).values
                if not np.isfinite(f).all():
                    continue
                raw = box_diff(f / G, z500.latitude, z500.longitude)
                rows[d] = {sid: raw[sid] - clim.loc[d.dayofyear, sid] for sid in BOXES}
            if rows:
                new = pd.DataFrame.from_dict(rows, orient="index").rename_axis("date")
                have = pd.concat([have, new]).sort_index()
                have = have[have.index >= newest - pd.Timedelta(days=KEEP_DAYS)]
                cache.mkdir(parents=True, exist_ok=True)
                have.round(2).to_csv(p)
            print(f"  ERA5 EPO/WPO: {len(rows)} new day(s), {len(have)} held", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  (ERA5 EPO/WPO refresh failed: {str(e)[:120]}; {len(have)} cached day(s) used)", flush=True)
    have = have[have.index >= first]
    return {sid: have[sid].dropna() for sid in BOXES if sid in have}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    out = refresh(a.cache, pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=60))
    for sid, s in out.items():
        print(f"  {sid}: {len(s)} days to {s.index.max():%Y-%m-%d}, last {s.iloc[-1]:+.0f} m" if len(s) else sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
