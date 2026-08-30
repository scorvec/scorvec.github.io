#!/usr/bin/env python3
"""ERA5 (ARCO) daily equatorial-band OLR — the third RMM channel, in house.

The site's RMM has been WIND-ONLY because the OLR channel needs a trailing
120-day-mean map to strip interannual variability, and no public OLR feed is
current: PSL's interpolated OLR stops in 2022 and the uninterpolated one in
2023. ERA5 closes that gap. `mean_top_net_long_wave_radiation_flux` is the
top-of-atmosphere net longwave flux, i.e. -OLR, and the ARCO store runs to
within about a week of real time (2026-08-23 as of 2026-08-29), which is well
inside the tolerance of a 120-day filter.

Same access pattern as fetch_arco_eq_u.py: anonymous GCS zarr, cos-weighted
15S-15N band mean, stored per longitude. Two differences that matter:

  4 snapshots/day, not 1.  OLR has a large diurnal cycle over land, and at a
  FIXED UTC hour the local time varies with longitude, so a single daily
  snapshot puts a longitude-dependent diurnal bias straight into the channel
  the EOFs project. 00/06/12/18Z samples four phases at every longitude, so the
  daily mean is unbiased.

  2.5 deg longitude, not 1.  This feeds the Wheeler & Hendon EOFs, which live on
  the 144-point 2.5 deg grid; there is nothing to gain from finer.

Sign: stored as OLR POSITIVE UP (W m-2), the convention of clim_olr in
climatology.nc, so the two difference directly.

    python src/fetch_arco_olr_band.py                 # last 200 days, resumable
    python src/fetch_arco_olr_band.py --start 2026-01-01 --workers 8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
OUT = REF / "era5_olr_band.nc"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "mean_top_net_long_wave_radiation_flux"
HOURS = (0, 6, 12, 18)
LON = np.arange(0, 360, 2.5)


def open_olr():
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    a = ds[VAR]
    lat = a.latitude.values
    sl = slice(15.05, -15.05) if lat[0] > lat[-1] else slice(-15.05, 15.05)
    return a.sel(latitude=sl)


def band(a: xr.DataArray) -> np.ndarray:
    """cos-weighted 15S-15N mean, then averaged onto the 2.5 deg EOF longitudes."""
    w = np.cos(np.deg2rad(a.latitude))
    b = a.weighted(w).mean(dim="latitude")
    # 0.25 -> 2.5 deg is an exact 10:1 block mean, no interpolation needed
    v = b.values
    v = v.reshape(v.shape[:-1] + (len(LON), 10)).mean(-1)
    return -v                                    # net TOA longwave is -OLR


def fetch_day(a, d: pd.Timestamp) -> np.ndarray | None:
    for attempt in range(3):
        try:
            sub = a.sel(time=[d + pd.Timedelta(hours=h) for h in HOURS]).compute()
            v = band(sub).mean(0)                # daily mean of the four snapshots
            if not np.isfinite(v).all() or not (80 < np.nanmean(v) < 340):
                raise ValueError(f"sanity check failed (mean {np.nanmean(v):.0f})")
            return v.astype("float32")
        except Exception as e:                                   # noqa: BLE001
            if attempt == 2:
                print(f"    {d:%Y-%m-%d}: {str(e)[:70]}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    have = {}
    if OUT.exists():
        d0 = xr.open_dataset(OUT)
        have = {pd.Timestamp(t).normalize(): v
                for t, v in zip(d0.time.values, d0["olr"].values)}
        d0.close()
        print(f"held {len(have)} days ({min(have):%Y-%m-%d} -> {max(have):%Y-%m-%d})")

    olr = open_olr()
    last = pd.Timestamp(olr.time.values[-1])
    # the store's time axis is padded far into the future; find the real end
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    while end > pd.Timestamp("2020-01-01"):
        if fetch_day(olr, end) is not None:
            break
        end -= pd.Timedelta(days=1)
    start = (pd.Timestamp(a.start) if a.start
             else end - pd.Timedelta(days=a.days - 1))
    want = [d for d in pd.date_range(start, end, freq="D") if d not in have]
    print(f"ERA5 data ends {end:%Y-%m-%d}; {len(want)} days to fetch", flush=True)

    if want:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, (d, v) in enumerate(zip(want, ex.map(lambda x: fetch_day(olr, x), want))):
                if v is not None:
                    have[d] = v
                if (i + 1) % 25 == 0:
                    el = time.time() - t0
                    print(f"  {i+1}/{len(want)}  {el/60:.1f} min  "
                          f"~{(len(want)-i-1)*el/(i+1)/60:.1f} min left", flush=True)

    ts = sorted(have)
    xr.Dataset(
        {"olr": (("time", "longitude"), np.stack([have[t] for t in ts]))},
        coords={"time": pd.DatetimeIndex(ts), "longitude": LON},
        attrs={"title": "ERA5 15S-15N band-mean OLR (positive up)",
               "source": f"{ARCO} :: {VAR} (negated)",
               "sampling": "daily mean of 00/06/12/18Z hourly-mean fluxes",
               "note": ("Positive-up OLR on the 2.5 deg Wheeler & Hendon longitude grid, "
                        "so it differences directly against clim_olr in climatology.nc."),
               "units": "W m-2"},
    ).to_netcdf(OUT)
    print(f"wrote {OUT.name}: {len(ts)} days, {ts[0]:%Y-%m-%d} -> {ts[-1]:%Y-%m-%d}, "
          f"mean {np.mean([have[t].mean() for t in ts]):.1f} W/m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
