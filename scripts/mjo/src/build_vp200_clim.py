#!/usr/bin/env python3
"""ERA5 1991-2020 day-of-year harmonic climatology of 200 hPa velocity potential χ —
the anomaly baseline for wind200_vpot.py.

WeatherBench2 1.5° ERA5 (light, conservative regrid), sampled on a ~10-day stride; χ is
computed per sample with wind200_vpot.velocity_potential (pyshtools spherical-harmonic Poisson
inversion), then fit per grid cell to mean + annual + semiannual (5 coeffs) →
data/reference/vp200_clim_coeffs.nc (committed, like the MMSF/AAM clim coeffs).

    python src/build_vp200_clim.py                              # full 1991-2020
    python src/build_vp200_clim.py --y0 2019 --y1 2020 --stride 30   # quick test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from wind200_vpot import velocity_potential, CLIM, LMAX

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=10, help="day stride between samples")
    a = ap.parse_args(argv)

    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=200)
    v = ds["v_component_of_wind"].sel(level=200)
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    want = want[np.isin(want.values, ds.time.values)]
    print(f"sampling {len(want)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1}, WB2 1.5°)", flush=True)

    chis, doys = [], []
    dlat = dlon = None
    t0 = time.time()
    for y in range(a.y0, a.y1 + 1):
        ty = want[want.year == y]
        if len(ty) == 0:
            continue
        uy = u.sel(time=ty).load(); vy = v.sel(time=ty).load()
        for t in ty:
            chi, dlat, dlon = velocity_potential(uy.sel(time=t), vy.sel(time=t))
            chis.append(chi.astype("float32")); doys.append(int(pd.Timestamp(t).dayofyear))
        print(f"  {y}: {len(ty)} samples  ({time.time()-t0:.0f}s)", flush=True)

    chis = np.stack(chis); doys = np.array(doys)
    w = 2 * np.pi * doys / 365.25
    X = np.stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)], axis=1)
    n, nlat, nlon = chis.shape
    coef, *_ = np.linalg.lstsq(X, chis.reshape(n, -1), rcond=None)
    coef = coef.reshape(5, nlat, nlon).astype("float32")

    CLIM.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"coef": (("ncoef", "lat", "lon"), coef)},
        coords={"lat": dlat, "lon": dlon},
        attrs={"note": f"ERA5 {a.y0}-{a.y1} 200 hPa velocity-potential harmonic clim (m^2/s); "
                       "eval with wind200_vpot.eval_vp_clim",
               "source": "WeatherBench2 ERA5 1.5deg conservative", "lmax": LMAX},
    ).to_netcdf(CLIM, encoding={"coef": {"zlib": True, "complevel": 5}})
    print(f"wrote {CLIM}  ({CLIM.stat().st_size/1e6:.1f} MB, 5 coeffs, {n} samples, {a.y0}-{a.y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
