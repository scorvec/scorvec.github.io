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
CACHE = Path.home() / "mjo" / "era5_cache"          # reduced ERA5 samples, kept for re-use
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6; RHO = 1.225; CD = 1.3e-3; COARSEN = 20


def _samples(a):
    """Coarsened (5°) friction + mountain torque-density samples — from the local
    cache if present, else streamed from ARCO-ERA5 and SAVED so we never re-download."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # _h0012: samples at BOTH 00Z and 12Z. The friction density has a strong land
    # diurnal cycle, and a 12Z-only clim put a spurious standing solar-phase
    # pattern into every 00Z run's anomalies (the pipeline runs both cycles).
    cpath = CACHE / f"torque_density_{a.y0}-{a.y1}_s{a.stride}_h0012.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        return c["fric"].values, c["mtn"].values, c.latitude.values, c.longitude.values, pd.to_datetime(c.time.values)
    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    cosphi = np.cos(np.deg2rad(lat))[:, None]; dlam = np.deg2rad(abs(lon[1] - lon[0]))
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    days_ = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D")
    want = days_.append(days_ + pd.Timedelta(hours=12)).sort_values()
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)

    def coarse(field2d):
        da = xr.DataArray(field2d, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        return da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()

    flist, mlist, tlist, clat, clon = [], [], [], None, None
    for n, t in enumerate(times):
        try:
            u = ds["10m_u_component_of_wind"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
            v = ds["10m_v_component_of_wind"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
            sp = ds["surface_pressure"].sel(time=t).reindex(latitude=lat, longitude=lon, method="nearest").values
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {e}"); continue
        fric = -RHO * CD * np.hypot(u, v) * u * (A * cosphi)
        dpd = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * dlam)   # periodic ∂p_s/∂λ
        fa = coarse(fric); ma = coarse(o.values * dpd)
        if clat is None:
            clat = fa.latitude.values; clon = fa.longitude.values
        flist.append(fa.values); mlist.append(ma.values); tlist.append(t)
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)
    fr = np.stack(flist); mt = np.stack(mlist)
    xr.Dataset({"fric": (("time", "latitude", "longitude"), fr), "mtn": (("time", "latitude", "longitude"), mt)},
               coords={"time": pd.DatetimeIndex(tlist), "latitude": clat, "longitude": clon}
               ).to_netcdf(cpath)
    print(f"saved ERA5 samples → {cpath}  ({(fr.nbytes+mt.nbytes)/1e6:.0f} MB)", flush=True)
    return fr, mt, clat, clon, pd.DatetimeIndex(tlist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    fr, mt, clat, clon, times = _samples(a)
    # separate harmonic fit per sampling hour (00Z / 12Z): the live product then
    # evaluates the clim at ITS valid hour, so the friction diurnal cycle cancels.
    hours = np.array(sorted(set(times.hour)))
    cf = np.empty((len(hours), 5, len(clat), len(clon)))
    cm = np.empty_like(cf)
    for j, h in enumerate(hours):
        sel = times.hour == h
        doy = times.dayofyear.values[sel]; w = 2 * np.pi * doy / 365.25
        X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        cf[j] = np.linalg.lstsq(X, fr[sel].reshape(sel.sum(), -1), rcond=None)[0].reshape(5, len(clat), len(clon))
        cm[j] = np.linalg.lstsq(X, mt[sel].reshape(sel.sum(), -1), rcond=None)[0].reshape(5, len(clat), len(clon))
        print(f"  fitted hour {h:02d}Z on {int(sel.sum())} samples")
    xr.Dataset({"friction": (("hour", "coef", "latitude", "longitude"), cf),
                "mountain": (("hour", "coef", "latitude", "longitude"), cm)},
               coords={"hour": hours, "coef": np.arange(5), "latitude": clat, "longitude": clon},
               attrs={"note": f"ERA5 {a.y0}-{a.y1} torque-density harmonic clim per sampling "
                              "hour (00Z/12Z), ~5° map grid"}
               ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
