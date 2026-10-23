#!/usr/bin/env python3
"""The 10 hPa temperature normal the polar temperature maps (t10_maps.py) are drawn against.

ERA5 monthly-mean temperature at 10 hPa, 1991-2020, on a global 1 deg grid (CDS reanalysis-era5-pressure-levels-
monthly-means; run in Actions by t10-clim.yml), each calendar month averaged over the 30 years, then a smooth daily
curve per grid point through those twelve means (scripts/lib/harmonic_clim.py: the mean and three annual harmonics
whose month averages match them). -> reference/t10_clim.nc, the coefficients coef (term, latitude, longitude) in K.

    python scripts/strat/build_t10_clim.py --work /tmp/t10      # needs a CDS key (~/.cdsapirc)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "lib"))
import harmonic_clim as HC

OUT = HERE / "reference" / "t10_clim.nc"
YEARS = range(1991, 2021)


def fetch(work: Path) -> Path:
    import cdsapi

    f = work / "era5_t10_monthly_1991-2020.grib"
    if f.exists():
        return f
    work.mkdir(parents=True, exist_ok=True)
    print("requesting ERA5 monthly-mean temperature at 10 hPa, 1991-2020 ...", flush=True)
    cdsapi.Client(timeout=900, quiet=True, progress=False, wait_until_complete=True, retry_max=1).retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["10"],
            "year": [str(y) for y in YEARS],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": [90, -180, -90, 180],
            "grid": [1.0, 1.0],
            "data_format": "grib",
        },
        str(f),
    )
    return f


def build(grib: Path) -> xr.Dataset:
    t = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""})["t"]
    tname = "valid_time" if "valid_time" in t.dims else "time"
    t = t.transpose(tname, "latitude", "longitude")
    month = pd.DatetimeIndex(t[tname].values).month
    if sorted(set(month)) != list(range(1, 13)) or t.sizes[tname] != 12 * len(YEARS):
        raise SystemExit(f"expected {12 * len(YEARS)} monthly fields, got {t.sizes[tname]}")
    means = np.stack([t.values[month == m].mean(axis=0) for m in range(1, 13)])  # (12, lat, lon)
    coef = HC.fit(means)
    fitted = np.stack([HC.evaluate(coef, np.linspace(a, b, 200)).mean(axis=0) for a, b in zip(HC.EDGES, HC.EDGES[1:])])
    print(f"  month-mean residual of the fit: rms {np.sqrt(np.nanmean((fitted - means) ** 2)):.3f} K", flush=True)
    return xr.Dataset(
        {"coef": (("term", "latitude", "longitude"), coef.astype("float32"))},
        coords={"term": np.arange(coef.shape[0]), "latitude": t.latitude.values, "longitude": t.longitude.values},
        attrs={
            "source": "ERA5 monthly-mean temperature at 10 hPa, 1991-2020 (Copernicus Climate Change Service)",
            "method": f"mean + {HC.NHARM} annual harmonics fitted to the month means (scripts/lib/harmonic_clim.py)",
            "units": "K",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="download directory")
    a = ap.parse_args()
    ds = build(fetch(a.work))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT, encoding={"coef": {"zlib": True, "complevel": 4}})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
