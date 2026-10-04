#!/usr/bin/env python3
"""
Why do North American FALLS vary so much across strong/super El Niños?
Mine the CMIP6 exam-passers for El Niño events and store each event's
SON and DJF North America tas anomaly (detrended), with its RONI-z and
east-lean-z, so the fall-vs-winter coherence question can be answered
with hundreds of events instead of five.

Saves SCRATCH/cmip6_na_fall_<model>.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brazil_flavor_cmip6 as h                              # noqa: E402

NA_LAT = (14, 74)
NA_LON = (188, 312)          # -172..-48 in 0..360


def run(model: str) -> None:
    inst, grid, nmax = h.MODELS[model]
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    base = f"cmip6/CMIP6/CMIP/{inst}/{model}/historical"
    mems = sorted({p.rsplit("/", 1)[-1] for p in fs.ls(base)})
    mems = [m for m in mems if m.endswith(("i1p1f1", "i1p2f1"))][:nmax]
    print(f"{model}: {len(mems)} members", flush=True)

    def fields(mem):
        vdirs = fs.ls(f"{base}/{mem}/Amon/tas/{grid}")
        ds = xr.open_zarr(gcsfs.GCSMap(vdirs[-1], gcs=fs), consolidated=True)
        da = ds["tas"]
        tv = da["time"].values
        if np.issubdtype(np.asarray(tv).dtype, np.datetime64):
            t = pd.DatetimeIndex(tv).to_period("M").to_timestamp()
        else:
            t = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, 1)
                                  for x in tv])
        return da.assign_coords(time=t).sortby("lat")

    ev_son, ev_djf, ev_z, ev_el = [], [], [], []
    lat = lon = None
    for k, mem in enumerate(mems):
        try:
            tas = fields(mem)
            trop_band = tas.sel(lat=slice(-22, 22)).load()
            na = tas.sel(lat=slice(*NA_LAT), lon=slice(*NA_LON)).load()
        except Exception as e:                               # noqa: BLE001
            print(f"  {mem}: load failed ({repr(e)[:40]})", flush=True)
            continue
        clim = trop_band.groupby("time.month").mean("time")
        anom = trop_band.groupby("time.month") - clim
        trop = h.wmean(anom.sel(lat=slice(-20, 20)))
        n34 = h.detrend_z(h.ndjfm_series(h.rel_index(anom, -5, 5, 190, 240, trop)))
        n12 = h.detrend_z(h.ndjfm_series(h.rel_index(anom, -10, 0, 270, 280, trop)))
        n4 = h.detrend_z(h.ndjfm_series(h.rel_index(anom, -5, 5, 160, 210, trop)))
        el = h.detrend_z(n12 - n4)

        nclim = na.groupby("time.month").mean("time")
        nan_ = na.groupby("time.month") - nclim
        t = pd.DatetimeIndex(nan_["time"].values)
        lat, lon = na["lat"].values, na["lon"].values

        def season(y, months):
            stamps = [pd.Timestamp(y if m >= 8 else y + 1, m, 1)
                      for m in months]
            idx = t.get_indexer(stamps)
            if (idx < 0).any():
                return None
            return nan_.isel(time=idx).mean("time").values

        def cube(months):
            out, yy = [], []
            for y in range(h.Y0M, h.Y1M + 1):
                f = season(y, months)
                if f is not None:
                    out.append(f); yy.append(y)
            v = np.array(out); yr = np.array(yy, float)
            yc = yr - yr.mean()
            sl = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
            return v - v.mean(0)[None] - yc[:, None, None] * sl[None], yy

        cs, ys = cube([9, 10, 11])
        cd, yd = cube([12, 1, 2])
        for y in n34[n34 >= h.ZEV].index:
            y = int(y)
            if y not in el.index or y not in ys or y not in yd:
                continue
            ev_son.append(cs[ys.index(y)])
            ev_djf.append(cd[yd.index(y)])
            ev_z.append(float(n34[y]))
            ev_el.append(float(el[y]))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(mems)}; events {len(ev_z)}", flush=True)

    np.savez(h.SCRATCH / f"cmip6_na_fall_{model}.npz",
             lat=lat, lon=lon, son=np.array(ev_son), djf=np.array(ev_djf),
             z=np.array(ev_z), elean=np.array(ev_el))
    print(f"{model}: saved {len(ev_z)} events "
          f"({(np.array(ev_z) >= 2.0).sum()} super z>=2)", flush=True)


if __name__ == "__main__":
    for mdl in ("MPI-ESM1-2-LR", "EC-Earth3", "MIROC6"):
        run(mdl)
