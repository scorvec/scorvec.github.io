#!/usr/bin/env python3
"""Hour-of-day 2 m temperature climatology (00/06/12/18Z) from WeatherBench2
ERA5 6-hourly, 1991-2020, ±7-day day-of-year window, on the verifier's 1.5°
cell-centre grid, Northern Hemisphere, float16.

Why: the daily-mean climatology (clim_1p5.npz: t2m) made every 12Z anomaly
read cold and every 00Z anomaly warm — the diurnal cycle, not weather. The
loop hid it behind a 24-h mean; this removes it properly: 18Z temperatures
against an 18Z normal (user question, 2026-09-03).

Output: scripts/verify/data/clim/clim_1p5_t2m6h.npz with keys
    t2m_h00, t2m_h06, t2m_h12, t2m_h18   (366, 60, 240) float16, KELVIN
    lat (60,)  lon (240,)                (the NH half of grid_1p5())
    slot366                              rows are doy366 slots: 29 Feb = 59 (leap years only),
                                         1 March = 60 in every year
Run once, offline (~5 GB of reads from GCS); the file rides the frames
branch under scripts/verify/data/clim/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from quantmet.degdays import doy366

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aifs_station_verify as V

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
OUT = V.DATA / "clim" / "clim_1p5_t2m6h.npz"


def main() -> int:
    t0 = time.time()
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    da = ds["2m_temperature"].sel(time=slice("1991-01-01", "2020-12-31"))
    lat_t, lon_t = V.grid_1p5()
    nh = lat_t > 0
    out = {"lat": lat_t[nh].astype(np.float32), "lon": lon_t.astype(np.float32)}
    for h in (0, 6, 12, 18):
        th = time.time()
        sub = da.sel(time=da.time.dt.hour == h)
        # day-of-year mean on the 366-slot calendar (a calendar date is the same
        # row in leap and common years; 29 Feb from the leap years alone), then
        # a ±7-day circular window
        slot = xr.DataArray(doy366(sub.time.values), coords={"time": sub.time}, dims="time", name="slot")
        doy = sub.groupby(slot).mean("time").compute()  # (366, lat, lon)
        arr = doy.transpose("slot", "latitude", "longitude").values.astype(np.float64)
        n = arr.shape[0]
        sm = np.empty_like(arr)
        for d in range(n):
            idx = [(d + k) % n for k in range(-7, 8)]
            sm[d] = arr[idx].mean(axis=0)
        cda = xr.DataArray(
            sm,
            coords=dict(dayofyear=doy.slot.values + 1, latitude=doy.latitude.values, longitude=doy.longitude.values),
            dims=("dayofyear", "latitude", "longitude"),
        )
        # wrap longitude so 359.x interpolates against 0 (the daily clim's
        # missing last column came from not doing this)
        cda = xr.concat([cda, cda.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
        interp = cda.interp(latitude=lat_t[nh], longitude=lon_t, method="linear").values
        out[f"t2m_h{h:02d}"] = interp.astype(np.float16)
        print(
            f"  {h:02d}Z: {sub.sizes['time']} steps → {interp.shape} in {time.time() - th:.0f} s "
            f"(nan {np.isnan(interp).mean():.3f})",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, slot366=np.array(1), **out)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
