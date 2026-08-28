#!/usr/bin/env python3
"""ERA5 climatology of the Walker (equatorial zonal) streamfunction Ψ_W(level, lon).

Harmonic (mean + annual + semiannual) day-of-year fit per (level, longitude) over
1991–2020 → data/reference/walker_clim_coeffs.nc, used by walker.py for the Ψ_W
anomaly. Samples stream from the WeatherBench2 1.5° ERA5 store (u+v on the 13
levels per sample is ~3 MB there — the 0.25° ARCO store would be ~100× that) and
are reduced through the SAME χ→u_D→Ψ_W path as the live product (walker.py's
solver and constants), then cached locally so a re-fit never re-downloads.

    python src/build_walker_clim.py --stride 5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

sys.path.insert(0, str(Path(__file__).parent))
from walker import LEVELS, LMAX, BAND, streamfunction
from wind200_vpot import velocity_potential, irrotational_wind

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "walker_clim_coeffs.nc"
CACHE = Path.home() / "mjo" / "era5_cache"
STORE = ("gs://weatherbench2/datasets/era5/"
         "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")


def _ud_of(u_da: xr.DataArray, v_da: xr.DataArray):
    """(lev, lon) equatorial-band divergent zonal wind — same reduction as walker.py."""
    rows, lon_out = [], None
    for lev in u_da.level.values:
        chi, dlat, dlon = velocity_potential(u_da.sel(level=lev), v_da.sel(level=lev),
                                             lmax=LMAX)
        uchi, _ = irrotational_wind(chi, dlat, dlon)
        band = np.abs(dlat) <= BAND
        w = np.cos(np.deg2rad(dlat[band]))
        rows.append((uchi[band] * w[:, None]).sum(0) / w.sum())
        lon_out = dlon
    return np.stack(rows), lon_out


def _samples(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"walker_ud_{a.y0}-{a.y1}_s{a.stride}.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        return c["ud"].values, c.longitude.values, pd.to_datetime(c.time.values)
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None)
    u = ds["u_component_of_wind"].sel(level=LEVELS).sortby("level")
    v = ds["v_component_of_wind"].sel(level=LEVELS).sortby("level")
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)
    ulist, tlist, lon = [], [], None
    for n, t in enumerate(times):
        try:
            ut = u.sel(time=t).rename({"lat": "latitude", "lon": "longitude"}
                                      if "lat" in u.dims else {}).load()
            vt = v.sel(time=t).rename({"lat": "latitude", "lon": "longitude"}
                                      if "lat" in v.dims else {}).load()
            ud, lon = _ud_of(ut, vt)
            ulist.append(ud); tlist.append(t)
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {repr(e)[:70]}"); continue
        if n % 100 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)
    arr = np.stack(ulist)
    xr.Dataset({"ud": (("time", "level", "longitude"), arr)},
               coords={"time": pd.DatetimeIndex(tlist), "level": LEVELS, "longitude": lon}
               ).to_netcdf(cpath)
    print(f"saved ERA5 samples → {cpath}  ({arr.nbytes/1e6:.0f} MB)", flush=True)
    return arr, lon, pd.DatetimeIndex(tlist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    ud, lon, times = _samples(a)
    p_pa = np.asarray(LEVELS, float) * 100.0
    psi = np.stack([streamfunction(ud[i], p_pa) for i in range(len(times))])
    doy = times.dayofyear.values; w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coef = np.linalg.lstsq(X, psi.reshape(len(times), -1), rcond=None)[0]
    coef = coef.reshape(5, len(LEVELS), len(lon))
    da = xr.DataArray(coef, dims=("coef", "level", "longitude"),
                      coords={"coef": np.arange(5), "level": LEVELS, "longitude": lon},
                      name="psi_coeffs",
                      attrs={"note": f"ERA5 {a.y0}-{a.y1} Walker Ψ_W harmonic clim "
                                     f"(WB2 1.5°, 12Z, {a.stride}-day stride); same "
                                     "χ→u_D→Ψ_W path and constants as walker.py"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.nc"); da.to_netcdf(tmp); tmp.replace(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
