#!/usr/bin/env python3
"""ERA5 reference for the intercomparison, through the SAME reducer as every model (common.Reducer), from ARCO-ERA5
(Google public bucket, anonymous; no CDS). One file per day, data/era5/<YYYYMMDD>.npz, laid out like a forecast
"init" with steps 0, 6, 12, 18, 24 h (torque fields) and 0 h (level winds), so consecutive days make the series:

  p_s, 10 m wind, ERA5's own turbulent surface stress (mean eastward turbulent surface stress: includes turbulent
  orographic form drag) and gravity-wave surface stress (orographic GWD + low-level blocking), total column water
  vapour, at 00/06/12/18/24 UTC (the mean-rate fields are the hour ending at each time, used as samples exactly as the
  site's torque product does); u and v on all 37 levels at 00 UTC (and the common subsets C11..C14 from them).
Orography: scripts/mjo/data/reference/era5_orography.nc (metres), the ERA5 model orography.

    python era5_collect.py --start 2026-09-14 --end 2026-10-08
Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                     # noqa: E402

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SFC = {"sp": "surface_pressure", "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
       "tau_turb": "mean_eastward_turbulent_surface_stress", "tau_gwd": "mean_eastward_gravity_wave_surface_stress",
       "tcwv": "total_column_water_vapour"}
_ds = None
_R = {}


def _open():
    global _ds
    if _ds is None:
        import gcsfs
        import xarray as xr
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None, decode_timedelta=False)
    return _ds


def _reducer():
    if "red" not in _R:
        import xarray as xr
        o = xr.open_dataarray(C.REF / "era5_orography.nc")
        lat = o.latitude.values.astype(np.float64)
        lon = o.longitude.values.astype(np.float64)
        h = o.values.astype(np.float64)
        assert lat[0] < lat[-1] and 4000 < h.max() < 7000
        _R["red"] = C.Reducer(C.Grid(lat, lon), h)
        ds = _open()
        _R["t0"] = np.datetime64(str(ds.time.values[0])[:19])
        _R["lev"] = ds.level.values.astype(float)
    return _R["red"]


def read(var, t, level=None):
    ds = _open()
    i = int((np.datetime64(t) - _R["t0"]) / np.timedelta64(1, "h"))
    v = ds[var].isel(time=i)
    a = np.asarray(v.values, dtype=np.float64)
    return a[..., ::-1, :]                                           # ARCO latitude descending -> ascending


def one_day(day: str):
    t0 = time.time()
    red = _reducer()
    D = dt.datetime.strptime(day, "%Y%m%d")
    T, L = {}, {}
    try:
        for s in (0, 6, 12, 18, 24):
            t = D + dt.timedelta(hours=s)
            with ThreadPoolExecutor(6) as ex:
                f = dict(zip(SFC, ex.map(lambda v: read(v, t), SFC.values())))
            if any(not np.isfinite(f[k]).all() for k in ("sp", "u10", "v10")):
                raise RuntimeError(f"NaN in ERA5 surface fields at {t}")
            for k in ("tau_turb", "tau_gwd", "tcwv"):
                f[k] = np.nan_to_num(f[k])
            if s == 0:
                sp0 = f["sp"]
            T[s] = red.torque_step(t.month, f["sp"], f["u10"], f["v10"], tau_turb=f["tau_turb"], tau_gwd=f["tau_gwd"],
                                   tcwv=f["tcwv"], maps=s == 0)
        with ThreadPoolExecutor(2) as ex:
            u, v = ex.map(lambda nm: read(nm, D), ("u_component_of_wind", "v_component_of_wind"))
        lev = list(_R["lev"])
        vv = {int(l): v[i] for i, l in enumerate(lev)}
        L[0] = red.level_step(lev, u.astype(np.float32), vv, sp0,
                              lb=True, maps=True)
    except Exception:                                                  # noqa: BLE001
        return day, None, traceback.format_exc()[-600:], time.time() - t0
    pk = C.pack_run("era5", T, L)
    return day, pk, None, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d0, d1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = []
    while d0 <= d1:
        if not (C.DATA / "era5" / f"{d0:%Y%m%d}.npz").exists():
            days.append(f"{d0:%Y%m%d}")
        d0 += dt.timedelta(days=1)
    print(f"{len(days)} ERA5 days to reduce", flush=True)
    (C.DATA / "era5").mkdir(parents=True, exist_ok=True)
    bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for day, pk, err, sec in ex.map(one_day, days):
            if pk is None:
                bad += 1
                print(f"{day}: FAILED {err}", flush=True)
                continue
            np.savez_compressed(C.DATA / "era5" / f"{day}.npz", meta=json.dumps({"model": "era5", "date": day,
                                                                                    "seconds": round(sec, 1)}), **pk)
            print(f"{day}: ok {sec:.0f} s", flush=True)
    return 1 if bad and bad == len(days) else 0


if __name__ == "__main__":
    sys.exit(main())
