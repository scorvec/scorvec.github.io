#!/usr/bin/env python3
"""
WeatherNext 2 ensemble RMM forecast — LOCAL ONLY for now.

Not wired into the site or any pipeline: publication of real-time-derived
products awaits the go-ahead on the GDM Real-Time Experimental Data terms
(non-retrievable value-added services appear publishable with attribution —
see the terms review in the session notes — but the owner decides).

Reads the WeatherNext 2 ensemble (64 members, 0.25°, 15 days) straight from
the access-gated GCS zarr (needs `gcloud auth application-default login` with
the approved account), reduces to the tropical band at the daily leads, and
projects onto the SAME reference EOFs / climatology / 120-day ENSO filter as
the AIFS product (src/rmm.py compute_rmm via its `fields` input) — so the two
models' plumes are directly comparable. Wind-only channels for now: pulling
the 6-hourly precip stack would triple the ~6 GB daily transfer.

Usage:
    python run_rmm_wnx.py                    # latest available init
    python run_rmm_wnx.py --date 20260824 --time 00
    python run_rmm_wnx.py --members 32       # subsample for a lighter pull

Output: plots/wnx/wnx_rmm_<init>.png + rmm_wnx_<init>.nc (git-ignored dirs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rmm import compute_rmm, GRID_RES, LAT_PAD
from plot import plot_rmm
import recent_analysis

HERE = Path(__file__).resolve().parent
CLIM_PATH = HERE / "data" / "reference" / "climatology.nc"
EOFS_PATH = HERE / "data" / "reference" / "eofs.nc"
OBS_PATH = HERE / "data" / "reference" / "obs_history.nc"
OUT_DIR = HERE / "plots" / "wnx"

BUCKET = "weathernext/weathernext_2_0_0/zarr"


def open_store(date: str | None, run_time: str | None):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="google_default")
    # yearly groups (2025_to_present, …); the root also holds a stray docs PDF
    years = sorted(p for p in fs.ls(BUCKET) if "_to_" in p.rsplit("/", 1)[-1])
    if date is None:
        latest = sorted(fs.ls(years[-1]))[-1].rsplit("/", 1)[-1]  # YYYYMMDD_HHhr_01_preds
        date, hr = latest.split("_")[0], latest.split("_")[1][:2]
    else:
        hr = run_time or "00"
    store = None
    for y in years:
        cand = f"{y}/{date}_{hr}hr_01_preds/predictions.zarr"
        if fs.exists(cand + "/.zmetadata"):
            store = cand
            break
    if store is None:
        raise FileNotFoundError(f"no WeatherNext store for {date} {hr}z")
    ds = xr.open_zarr(gcsfs.GCSMap(store, gcs=fs), consolidated=True)
    return ds, date, hr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="Init date YYYYMMDD (default: latest)")
    ap.add_argument("--time", default=None, help="Init hour 00/06/12/18")
    ap.add_argument("--members", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds, date, hr = open_store(args.date, args.time)
    print(f"WeatherNext 2 init {date} {hr}z — {ds.sizes['sample']} members, "
          f"{ds.sizes['time']} leads")

    u = ds["u_component_of_wind"].sel(level=[850, 200])
    # daily-anchored leads only (24 h multiples) — 15 of 60, ~4x lighter
    hours = (u["time"] / np.timedelta64(1, "h")).astype(int)
    u = u.sel(time=(hours % 24 == 0))
    if args.members < ds.sizes["sample"]:
        u = u.isel(sample=slice(0, args.members))

    u = u.sortby("lat").sel(lat=slice(-LAT_PAD, LAT_PAD))
    lon_new = np.arange(0, 360, GRID_RES)
    lat_new = np.arange(-20, 20.01, GRID_RES)
    if float(u["lon"].min()) < 0:
        u = u.assign_coords(lon=u["lon"] % 360).sortby("lon")
    u = u.interp(lat=lat_new, lon=lon_new, method="linear")
    u = u.rename({"sample": "number", "time": "step",
                  "lat": "latitude", "lon": "longitude"})

    print("  pulling + reducing (dask, daily leads, tropical band)…", flush=True)
    u = u.load()
    print(f"  loaded in {time.time()-t0:.0f}s")

    u850 = u.sel(level=850, drop=True)
    u200 = u.sel(level=200, drop=True)

    clim = xr.open_dataset(CLIM_PATH)
    eofs = xr.open_dataset(EOFS_PATH)
    mean120 = recent_analysis.load_map120()
    rmm = compute_rmm(HERE / "data" / "aifs", date, hr, clim, eofs,
                      mean120=mean120, prcp_clim=None, model="weathernext2",
                      fields={"ens": (u850, u200, None)})
    print(f"  RMM computed: {rmm.sizes['member']} members × "
          f"{rmm.sizes['lead_day']} days ({rmm.attrs['channels']})")

    rmm.attrs["model_label"] = "WeatherNext 2"
    obs = xr.open_dataset(OBS_PATH) if OBS_PATH.exists() else None
    png = OUT_DIR / f"wnx_rmm_{date}_{hr}z.png"
    plot_rmm(rmm, obs=obs, out_path=png)
    rmm.to_netcdf(OUT_DIR / f"rmm_wnx_{date}_{hr}z.nc")
    print(f"  wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
