#!/usr/bin/env python3
"""ERA5 per-gridcell climatology of the friction + mountain torque DENSITY on the
~5° torque-map grid — harmonic (mean+annual+semiannual) day-of-year coeffs.

Computes the SAME quantities the maps plot — friction = −ρC_d|V|u·a cosφ (bulk, from
ERA5 10-m winds) and mountain = h·∂p_s/∂λ (ERA5 surface pressure + cached orography) —
coarsened to the map grid, then fits the seasonal cycle per cell → torque_map_clim_coeffs.nc.
The torque maps subtract clim(day-of-year) instead of the forecast-period mean.

    python src/build_torque_map_clim.py --stride 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
OUT = REF / "torque_map_clim_coeffs.nc"
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6; RHO = 1.225; CD = 1.3e-3; COARSEN = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    cosphi = np.cos(np.deg2rad(lat))[:, None]; dlam = np.deg2rad(abs(lon[1] - lon[0]))
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)

    def coarse(field2d):
        da = xr.DataArray(field2d, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        return da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()

    clat = clon = None; XtX = np.zeros((5, 5)); Xtf = Xtm = None
    for n, t in enumerate(times):
        try:
            u = ds["10m_u_component_of_wind"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
            v = ds["10m_v_component_of_wind"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
            sp = ds["surface_pressure"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {e}"); continue
        fric = -RHO * CD * np.hypot(u, v) * u * (A * cosphi)
        dpd = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * dlam)   # periodic ∂p_s/∂λ
        mtn = o.values * dpd
        fa = coarse(fric); ma = coarse(mtn)
        if clat is None:
            clat = fa.latitude.values; clon = fa.longitude.values
            Xtf = np.zeros((5, len(clat), len(clon))); Xtm = np.zeros_like(Xtf)
        w = 2 * np.pi * t.dayofyear / 365.25
        x = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        XtX += np.outer(x, x)
        Xtf += x[:, None, None] * fa.values[None]; Xtm += x[:, None, None] * ma.values[None]
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)

    cf = np.linalg.solve(XtX, Xtf.reshape(5, -1)).reshape(5, len(clat), len(clon))
    cm = np.linalg.solve(XtX, Xtm.reshape(5, -1)).reshape(5, len(clat), len(clon))
    xr.Dataset({"friction": (("coef", "latitude", "longitude"), cf),
                "mountain": (("coef", "latitude", "longitude"), cm)},
               coords={"coef": np.arange(5), "latitude": clat, "longitude": clon},
               attrs={"note": f"ERA5 {a.y0}-{a.y1} torque-density harmonic clim, ~5° map grid"}
               ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
