#!/usr/bin/env python3
"""Build the per-calendar-month standardization factor (σ) for WMO/CPC-style RONI.

The official Relative Oceanic Niño Index standardizes the relative Niño-3.4 anomaly
(Niño-3.4 SSTA − tropical-mean 20°S–20°N SSTA, 3-month running mean) by its standard
deviation — computed *separately for each calendar (center) month*, because the
relative-anomaly variance has a strong seasonal cycle (large in boreal winter, small
in spring). This derives that 12-value σ table from the OISST v2.1 monthly record over
the 1991–2020 base period (the same base as our daily anomalies) and writes
data/roni_sigma.json. One-time / occasional — the table is committed and loaded by
sst-roni.py to convert RONI from °C to standardized (σ) units.

    python scripts/sst/build_roni_sigma.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"                                 # gitignored cache for the big source files
OUT = HERE / "roni_sigma.json"                       # committed σ table (loaded by sst-roni.py)
PSL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
MEAN_URL = PSL + "/sst.mon.mean.nc"                  # 2.2 GB monthly mean (OPeNDAP is unreliable on it)
LTM_URL = PSL + "/sst.mon.ltm.1991-2020.nc"          # 46 MB monthly climatology
NINO34 = dict(lat=(-5, 5), lon=(190, 240))
TROPICAL = dict(lat=(-20, 20), lon=(0, 360))
BASE = (1991, 2020)


def _fetch(url: str) -> Path:
    dest = DATA / url.rsplit("/", 1)[1]
    if not dest.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {dest.name} …", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def _box(da: xr.DataArray, box: dict) -> xr.DataArray:
    """Cosine-lat-weighted, NaN-aware (land-excluded) mean over a lat/lon box → time series."""
    sub = da.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(("lat", "lon"), skipna=True)


def main() -> int:
    print("loading OISST v2.1 monthly (PSL) …", flush=True)
    mean = xr.open_dataset(_fetch(MEAN_URL))["sst"].sel(lat=slice(-25, 25))    # band covers both boxes
    clim = xr.open_dataset(_fetch(LTM_URL))["sst"].sel(lat=slice(-25, 25)).groupby("time.month").mean()
    anom = (mean.groupby("time.month") - clim).load()            # monthly anomaly vs 1991-2020
    print(f"  loaded anomaly band: {dict(anom.sizes)}", flush=True)

    nino = _box(anom, NINO34)
    trop = _box(anom, TROPICAL)
    rel = (nino - trop).to_series().sort_index()
    rel3 = rel.rolling(3, center=True, min_periods=3).mean()     # the RONI 3-month running mean
    base = rel3[(rel3.index.year >= BASE[0]) & (rel3.index.year <= BASE[1])]

    sigma = {int(m): float(base[base.index.month == m].std(ddof=1)) for m in range(1, 13)}
    counts = {int(m): int((base.index.month == m).sum()) for m in range(1, 13)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "sigma_by_month": sigma,
        "base_period": f"{BASE[0]}-{BASE[1]}",
        "n_years_by_month": counts,
        "source": "NOAA OISST v2.1 monthly (PSL)",
        "definition": ("standard deviation (ddof=1) of the 3-month-running-mean relative "
                       "Niño-3.4 index (Niño-3.4 − tropical[20S-20N] anomaly), per center month"),
    }, indent=2))
    print("σ by month (°C):", {m: round(s, 3) for m, s in sigma.items()})
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
