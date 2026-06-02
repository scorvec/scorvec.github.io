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
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)

    p_pa = np.array(LEVELS, float) * 100.0
    lat = None; XtX = np.zeros((5, 5)); Xty = None
    for n, t in enumerate(times):
        try:
            v = ds["v_component_of_wind"].sel(time=t, level=LEVELS).sortby("level")
            vbar = v.mean("longitude").values                      # (lev, lat)
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {e}"); continue
        if lat is None:
            lat = v.latitude.values; Xty = np.zeros((5, len(LEVELS), len(lat)))
        psi = streamfunction(vbar, p_pa, lat)                      # (lev, lat) ×10^10 kg/s
        w = 2 * np.pi * t.dayofyear / 365.25
        x = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        XtX += np.outer(x, x); Xty += x[:, None, None] * psi[None]
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)

    coeffs = np.linalg.solve(XtX, Xty.reshape(5, -1)).reshape(5, len(LEVELS), len(lat))
    xr.DataArray(coeffs, dims=("coef", "level", "latitude"),
                 coords={"coef": np.arange(5), "level": LEVELS, "latitude": lat},
                 attrs={"units": "10^10 kg/s", "note": f"ERA5 {a.y0}-{a.y1} Ψ harmonic clim"}
                 ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
