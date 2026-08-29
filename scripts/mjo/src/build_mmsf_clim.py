#!/usr/bin/env python3
"""ERA5 climatology of the meridional mass streamfunction Ψ(level, latitude).

Harmonic (mean + annual + semiannual) day-of-year fit per (level, lat) over
1991–2020 → data/reference/mmsf_clim_coeffs.nc, used by mmsf.py for the Ψ anomaly
(how the Hadley/Ferrel cells differ from the climatological seasonal cycle).

    python src/build_mmsf_clim.py --stride 5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

sys.path.insert(0, str(Path(__file__).parent))
from mmsf import LEVELS, streamfunction

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "mmsf_clim_coeffs.nc"
CACHE = Path.home() / "mjo" / "era5_cache"          # reduced ERA5 samples, kept for re-use
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def _samples(a):
    """Zonal-mean v(time, level, lat) samples — from the local cache if present,
    else streamed from ARCO-ERA5 and SAVED so we never re-download."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"mmsf_vbar_{a.y0}-{a.y1}_s{a.stride}.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        return c["vbar"].values, c.latitude.values, pd.to_datetime(c.time.values)
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)
    vlist, tlist, lat = [], [], None
    for n, t in enumerate(times):
        try:
            v = ds["v_component_of_wind"].sel(time=t, level=LEVELS).sortby("level")
            vlist.append(v.mean("longitude").values); tlist.append(t); lat = v.latitude.values
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {e}"); continue
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)
    vbar = np.stack(vlist)                                          # (time, lev, lat)
    xr.Dataset({"vbar": (("time", "level", "latitude"), vbar)},
               coords={"time": pd.DatetimeIndex(tlist), "level": LEVELS, "latitude": lat}
               ).to_netcdf(cpath)
    print(f"saved ERA5 samples → {cpath}  ({vbar.nbytes/1e6:.0f} MB)", flush=True)
    return vbar, lat, pd.DatetimeIndex(tlist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    vbar, lat, times = _samples(a)
    p_pa = np.array(LEVELS, float) * 100.0
    psi = np.stack([streamfunction(vbar[i], p_pa, lat) for i in range(len(vbar))])   # (time,lev,lat)
    doy = times.dayofyear.values; w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coeffs = np.linalg.lstsq(X, psi.reshape(len(psi), -1), rcond=None)[0].reshape(5, len(LEVELS), len(lat))
    xr.DataArray(coeffs, dims=("coef", "level", "latitude"),
                 coords={"coef": np.arange(5), "level": LEVELS, "latitude": lat},
                 attrs={"units": "10^10 kg/s", "note": f"ERA5 {a.y0}-{a.y1} Ψ harmonic clim"}
                 ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
