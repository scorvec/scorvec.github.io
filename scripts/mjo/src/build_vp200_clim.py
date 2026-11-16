#!/usr/bin/env python3
"""ERA5 1991-2020 day-of-year harmonic climatology of 200 hPa velocity potential χ —
the anomaly baseline for wind200_vpot.py.

WeatherBench2 1.5° ERA5 (light, conservative regrid), sampled on a ~10-day stride at 00 and at
12 UTC; χ is computed per sample with wind200_vpot.velocity_potential (pyshtools spherical-harmonic
Poisson inversion), then fit per grid cell and hour to mean + annual + semiannual (5 coeffs) →
data/reference/vp200_clim_coeffs.nc (committed, like the MMSF/AAM clim coeffs). One fit per hour
because the divergent wind carries the atmospheric tide: the 00 UTC normal differs from the
12 UTC one by about 2e6 m² s⁻¹ over the east Pacific and the Indian Ocean, more than the maps'
first shading level, so a frame is compared with the normal of its own hour. Built in Actions
(.github/workflows/vp200-clim.yml), which commits the file.

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
from doycal import noleap_doy

HOURS = (0, 12)  # the hours of the frames wind200_vpot.py draws
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=10, help="day stride between samples")
    a = ap.parse_args(argv)

    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=200)
    v = ds["v_component_of_wind"].sel(level=200)
    days = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D")
    dlat = dlon = None
    t0 = time.time()
    coefs, n = [], 0
    for hour in HOURS:
        want = days + pd.Timedelta(hours=hour)
        want = want[np.isin(want.values, ds.time.values)]
        print(f"{hour:02d} UTC: {len(want)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1}, WB2 1.5°)", flush=True)
        chis, doys = [], []
        for y in range(a.y0, a.y1 + 1):
            ty = want[want.year == y]
            if len(ty) == 0:
                continue
            uy = u.sel(time=ty).load()
            vy = v.sel(time=ty).load()
            for t in ty:
                chi, dlat, dlon = velocity_potential(uy.sel(time=t), vy.sel(time=t))
                chis.append(chi.astype("float32"))
                doys.append(float(noleap_doy(t)))  # common-year day
            print(f"  {y}: {len(ty)} samples  ({time.time() - t0:.0f}s)", flush=True)
        chis = np.stack(chis)
        w = 2 * np.pi * np.array(doys) / 365.25
        X = np.stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)], axis=1)
        n, nlat, nlon = chis.shape
        coefs.append(np.linalg.lstsq(X, chis.reshape(n, -1), rcond=None)[0].reshape(5, nlat, nlon).astype("float32"))
    coef = np.stack(coefs)
    tide = np.abs(coef[0, 0] - coef[1, 0])
    print(f"annual-mean {HOURS[0]:02d} minus {HOURS[1]:02d} UTC: up to {tide.max() / 1e6:.2f}e6 m2/s", flush=True)

    CLIM.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"coef": (("hour", "ncoef", "lat", "lon"), coef)},
        coords={"hour": list(HOURS), "lat": dlat, "lon": dlon},
        attrs={
            "note": f"ERA5 {a.y0}-{a.y1} 200 hPa velocity-potential harmonic clim (m^2/s) at each hour (UTC); "
            "eval with wind200_vpot.eval_vp_clim",
            "source": "WeatherBench2 ERA5 1.5deg conservative",
            "lmax": LMAX,
        },
    ).to_netcdf(CLIM, encoding={"coef": {"zlib": True, "complevel": 5}})
    print(f"wrote {CLIM}  ({CLIM.stat().st_size / 1e6:.1f} MB, 5 coeffs, {n} samples an hour, {a.y0}-{a.y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
