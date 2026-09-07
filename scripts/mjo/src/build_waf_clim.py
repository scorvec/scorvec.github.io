#!/usr/bin/env python3
"""ERA5 basic state for the wave-activity-flux product: U, V and ψ at 200 hPa.

Harmonic (mean + annual + semiannual) day-of-year fits over 1991–2020 from the
WeatherBench2 1.5° ERA5 store, all reduced onto the SAME pole-free→DH2 grid as
the live product (ψ through the identical spherical-harmonic inversion in
waf.py, U/V interpolated to that grid) → data/reference/waf_clim_coeffs.nc.

    python src/build_waf_clim.py --stride 3
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

import sys
sys.path.insert(0, str(Path(__file__).parent))
from waf import streamfunction_psi, LMAX
from wind200_vpot import _to_0360

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT_TMPL = "waf_clim_coeffs{suffix}.nc"            # 200 hPa (legacy, no suffix) or _250
CACHE = Path.home() / "mjo" / "era5_cache"
STORE = ("gs://weatherbench2/datasets/era5/"
         "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")


def _samples(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"waf_uvpsi{a.level_tag}_{a.y0}-{a.y1}_s{a.stride}.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        return (c["U"].values, c["V"].values, c["psi"].values,
                c.latitude.values, c.longitude.values, pd.to_datetime(c.time.values))
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 64})      # dask: chunks fetched in parallel
    u = ds["u_component_of_wind"].sel(level=a.level)
    v = ds["v_component_of_wind"].sel(level=a.level)
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1}), loading a year at a time", flush=True)
    Ul, Vl, Pl, tl, glat, glon = [], [], [], [], None, None
    for yr in range(a.y0, a.y1 + 1):
        ty = times[times.year == yr]
        uy = u.sel(time=ty).load(); vy = v.sel(time=ty).load()        # one parallel fetch per year
        for t in ty:
            try:
                ut = uy.sel(time=t); vt = vy.sel(time=t)
                psi, glat, glon = streamfunction_psi(ut, vt, lmax=LMAX)
                Ug = (_to_0360(ut).interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values)
                Vg = (_to_0360(vt).interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values)
                Ul.append(Ug); Vl.append(Vg); Pl.append(psi); tl.append(t)
            except Exception as e:                                     # noqa: BLE001
                print(f"  skip {t:%Y-%m-%d}: {repr(e)[:60]}"); continue
        print(f"  {yr}: {len(tl)} samples so far", flush=True)
    U, V, P = np.stack(Ul), np.stack(Vl), np.stack(Pl)
    xr.Dataset({"U": (("time", "latitude", "longitude"), U),
                "V": (("time", "latitude", "longitude"), V),
                "psi": (("time", "latitude", "longitude"), P)},
               coords={"time": pd.DatetimeIndex(tl), "latitude": glat, "longitude": glon}
               ).to_netcdf(cpath)
    print(f"saved ERA5 samples → {cpath}  ({(U.nbytes*3)/1e6:.0f} MB)", flush=True)
    return U, V, P, glat, glon, pd.DatetimeIndex(tl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--level", type=int, default=200, help="pressure level (hPa); 250 is the operational WAF level since 2026-09-07")
    a = ap.parse_args()
    a.level_tag = "" if a.level == 200 else f"_{a.level}"
    OUT = REF / OUT_TMPL.format(suffix=a.level_tag)
    U, V, P, glat, glon, times = _samples(a)
    doy = times.dayofyear.values; w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])

    def fit(y):
        c = np.linalg.lstsq(X, y.reshape(len(times), -1), rcond=None)[0]
        return c.reshape(X.shape[1], len(glat), len(glon))

    out = xr.Dataset({"U": (("coef", "latitude", "longitude"), fit(U)),
                      "V": (("coef", "latitude", "longitude"), fit(V)),
                      "psi": (("coef", "latitude", "longitude"), fit(P))},
                     coords={"coef": np.arange(5), "latitude": glat, "longitude": glon},
                     attrs={"level_hPa": a.level, "note": f"ERA5 {a.y0}-{a.y1} {a.level} hPa U/V/ψ harmonic clim on the "
                                    f"lmax={LMAX} DH2 grid (WB2 1.5°, 12Z, {a.stride}-day "
                                    "stride); ψ via waf.streamfunction_psi"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.nc"); out.to_netcdf(tmp); tmp.replace(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
