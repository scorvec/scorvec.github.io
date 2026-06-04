#!/usr/bin/env python3
"""Zonal-mean relative-AAM DENSITY climatology m_clim(dayofyear, level, latitude) for
the AAM-anatomy plot's anomaly panels. Same mass-weighted integrand as aam.py, but kept
resolved in (level, latitude) instead of integrated:

    m(φ,p) = (a³/g) · cos²φ · dλ·dφ · Σ_λ (u·Δp)      (per (lev,lat) band)

Read EVERY day (12Z) of 1991–2020 from the WeatherBench2 1.5° ERA5 store (u on the 13
AAM levels + surface pressure — small per field, unlike the 37-level ARCO store), take
the per-day-of-year mean. Stored at 1.5°; the plot interpolates it up to the forecast
grid. No smoothing (matches the z500-normal convention).

    python src/build_aam_density_clim.py --workers 8
"""
from __future__ import annotations
import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

sys.path.insert(0, str(Path(__file__).parent))
from aam import A, G, LEVELS, _vert_weights

STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
OUT = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_density_clim.nc"
CHUNK = 60


def _accumulate(times):
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None)
    u = ds["u_component_of_wind"].sel(level=LEVELS).sortby("level")
    sp = ds["surface_pressure"]
    lat = u.latitude.values
    p_pa = u.level.values * 100.0
    cos2 = np.cos(np.deg2rad(lat)) ** 2
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
    pre = (A ** 3 / G) * dlon * dlat
    nlev, nlat = len(p_pa), len(lat)
    ssum = np.zeros((366, nlev, nlat), "float64"); scnt = np.zeros(366, "int32")
    for t in times:
        try:
            ut = u.sel(time=t).transpose("level", "latitude", "longitude").values
            spt = sp.sel(time=t).transpose("latitude", "longitude").values
        except Exception:                                    # noqa: BLE001
            continue
        dp = _vert_weights(p_pa, spt)                        # (lev,lat,lon)
        band = pre * cos2[None, :] * (ut * dp).sum(axis=2)   # (lev,lat)
        d = int(pd.Timestamp(t).dayofyear)
        ssum[d - 1] += band; scnt[d - 1] += 1
    return ssum, scnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None)
    lat = ds.latitude.values
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq="1D") + pd.Timedelta(hours=12)
    era5_times = pd.to_datetime(ds.time.values)
    times = list(want[want.isin(era5_times)])
    ntot = len(times)
    print(f"AAM density clim: reading EVERY day 12Z, {ntot} fields ({a.y0}-{a.y1}), "
          f"{a.workers} processes …", flush=True)

    chunks = [times[i:i + CHUNK] for i in range(0, ntot, CHUNK)]
    ssum = np.zeros((366, len(LEVELS), len(lat)), "float64"); scnt = np.zeros(366, "int32")
    done = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_accumulate, ch) for ch in chunks]
        for fut in as_completed(futs):
            s, c = fut.result(); ssum += s; scnt += c; done += int(c.sum())
            el = time.time() - t0; rate = done / el if el else 0
            eta = (ntot - done) / rate / 60 if rate else 0
            print(f"  {done}/{ntot} ({100*done/ntot:.0f}%)  {el/60:.1f} min · ETA {eta:.1f} min", flush=True)

    clim = (ssum / np.maximum(scnt, 1)[:, None, None]).astype("float32")   # (doy,lev,lat)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    xr.DataArray(clim, dims=("dayofyear", "level", "latitude"),
                 coords={"dayofyear": np.arange(1, 367), "level": list(LEVELS), "latitude": lat},
                 attrs={"units": "kg m^2 s^-1 per band", "period": f"{a.y0}-{a.y1}",
                        "note": "zonal-sum (a^3/g) cos^2(phi) dlam dphi u*dp, per-doy mean (WB2 1.5)"}
                 ).to_netcdf(OUT, encoding={"__xarray_dataarray_variable__": {"zlib": True, "complevel": 4}})
    print(f"wrote {OUT}  ({clim.shape}, {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
