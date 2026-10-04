#!/usr/bin/env python3
"""ERA5 (ARCO) recent equatorial-band U850/U200 — the observed RMM wind channels.

fetch_arco_eq_u.py builds the long 2003-2024 record used to calibrate pc_wind_std.
This is its short operational sibling: the last N days only, two levels, for the
observed RMM track the phase diagram draws.

Why not NCEP: _load_recent_ncep in recent_analysis.py pulls PSL's NCEP/NCAR R1
daily uwnd, and that stream is running about five months behind (2026-03-17 when
this was written) — fine for the historical calibration it was added for, useless
for a live track. ERA5 via ARCO is six days behind instead.

Cost note: the ARCO chunk is (1 hour, 37 levels, 721, 1440) = 154 MB, so
selecting two levels saves decode but nothing on the wire. One 00Z snapshot per
day is therefore the right knob, and it is enough: the band-mean zonal wind's
diurnal cycle is under 0.3 m/s against a 1.8 m/s standardisation.

    python src/fetch_arco_eq_uv.py --days 60
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
OUT = REF / "era5_eq_uv_recent.nc"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [850, 200]
LON = np.arange(0, 360, 2.5)


def open_u():
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    a = ds["u_component_of_wind"]
    lat = a.latitude.values
    sl = slice(15.05, -15.05) if lat[0] > lat[-1] else slice(-15.05, 15.05)
    return a.sel(latitude=sl, level=LEVELS)


def band(a: xr.DataArray) -> np.ndarray:
    w = np.cos(np.deg2rad(a.latitude))
    v = a.weighted(w).mean(dim="latitude").values          # (level, lon)
    return v.reshape(v.shape[:-1] + (len(LON), 10)).mean(-1)


def fetch_day(a, d):
    for k in range(3):
        try:
            v = band(a.sel(time=d).compute())
            if not np.isfinite(v).all() or np.abs(v).max() > 120:
                raise ValueError("sanity check failed")
            return v.astype("float32")
        except Exception as e:                                   # noqa: BLE001
            if k == 2:
                return None
            time.sleep(3 * (k + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    have = {}
    if OUT.exists():
        d0 = xr.open_dataset(OUT)
        have = {pd.Timestamp(t).normalize(): v
                for t, v in zip(d0.time.values, d0["u"].values)}
        d0.close()

    u = open_u()
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    while fetch_day(u, end) is None and end > pd.Timestamp("2020-01-01"):
        end -= pd.Timedelta(days=1)
    want = [d for d in pd.date_range(end - pd.Timedelta(days=a.days - 1), end, freq="D")
            if d not in have]
    print(f"ERA5 winds end {end:%Y-%m-%d}; {len(want)} days to fetch", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (d, v) in enumerate(zip(want, ex.map(lambda x: fetch_day(u, x), want))):
            if v is not None:
                have[d] = v
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(want)}  {el/60:.1f} min  "
                      f"~{(len(want)-i-1)*el/(i+1)/60:.1f} min left", flush=True)

    ts = sorted(have)
    xr.Dataset({"u": (("time", "level", "longitude"), np.stack([have[t] for t in ts]))},
               coords={"time": pd.DatetimeIndex(ts), "level": LEVELS, "longitude": LON},
               attrs={"title": "ERA5 15S-15N band-mean zonal wind, recent days",
                      "source": f"{ARCO} :: u_component_of_wind, 00Z daily",
                      "units": "m s-1"}).to_netcdf(OUT)
    print(f"wrote {OUT.name}: {len(ts)} days, {ts[0]:%Y-%m-%d} -> {ts[-1]:%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
