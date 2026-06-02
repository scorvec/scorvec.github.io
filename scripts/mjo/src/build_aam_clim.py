#!/usr/bin/env python3
"""Build the ERA5 relative-AAM climatology (global / NH / SH): seasonal mean,
±1σ spread, and the 1991-2020 min/max record envelope, for the AAM card.

Streams ARCO-ERA5 u@13-levels + surface pressure every 5 days across 1991-2020,
computes AAM with the SAME integral as aam.py, and per day-of-year sample stores
mean / std / min / max over the 30 years (plus a harmonic-fit smooth mean).

    python src/build_aam_clim.py            # writes data/reference/aam_clim.nc
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from aam import aam_of, LEVELS, SCALE

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
YEARS = range(1991, 2021)
DOY_SAMPLES = list(range(2, 366, 5))                  # every 5 days → 73 samples/yr
OUT = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_clim.nc"
REGIONS = ("global", "nh", "sh")


def _one(args):
    ds, t = args
    u = ds["u_component_of_wind"].sel(time=t, level=LEVELS).sortby("level")
    sp = ds["surface_pressure"].sel(time=t)
    lat = u.latitude.values
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
    g, n, s = aam_of(u.values, u.level.values * 100.0, sp.values, lat, dlon, dlat)
    return g / SCALE, n / SCALE, s / SCALE


def _harmonic(doy, y):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return coef


def main() -> int:
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    times, doys = [], []
    for y in YEARS:
        base = pd.Timestamp(y, 1, 1)
        for d in DOY_SAMPLES:
            times.append((base + pd.Timedelta(days=d - 1)).strftime("%Y-%m-%dT12:00"))
            doys.append(d)
    print(f"computing ERA5 AAM at {len(times)} samples (5-day × 1991-2020) …", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(_one, [(ds, t) for t in times]))
    df = pd.DataFrame(rows, columns=list(REGIONS)); df["doy"] = doys

    coords_doy = DOY_SAMPLES
    stat = {s: {} for s in ("mean", "std", "min", "max")}
    coeffs = []
    for k in REGIONS:
        gb = df.groupby("doy")[k]
        stat["mean"][k] = gb.mean().reindex(coords_doy).values
        stat["std"][k] = gb.std().reindex(coords_doy).values
        stat["min"][k] = gb.min().reindex(coords_doy).values
        stat["max"][k] = gb.max().reindex(coords_doy).values
        coeffs.append(_harmonic(df["doy"].values, df[k].values))
    out = xr.Dataset(
        {"coeffs": (("region", "coef"), np.stack(coeffs)),
         "mean": (("region", "doy"), np.stack([stat["mean"][k] for k in REGIONS])),
         "std":  (("region", "doy"), np.stack([stat["std"][k] for k in REGIONS])),
         "min":  (("region", "doy"), np.stack([stat["min"][k] for k in REGIONS])),
         "max":  (("region", "doy"), np.stack([stat["max"][k] for k in REGIONS]))},
        coords={"region": list(REGIONS), "coef": np.arange(5), "doy": coords_doy})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    print(f"saved {OUT}")
    for k in REGIONS:
        print(f"  {k}: mean {df[k].mean():.1f}  typical σ {np.nanmean(stat['std'][k]):.2f}  "
              f"record {np.nanmin(stat['min'][k]):.1f}…{np.nanmax(stat['max'][k]):.1f} ×10²⁵")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
