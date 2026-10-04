#!/usr/bin/env python3
"""
Monthly EP-event temperature fingerprints from the CMIP6 exam passers.

Same member-standardized event mining as brazil_flavor_cmip6.py, but
accumulates per-calendar-month (Nov, Dec, Jan, Feb, Mar) detrended tas
anomaly composites over EP events only, plus the sum of raw (detrended,
°C, non-z) NDJFM relative-Niño-3.4 amplitudes so the composite can be
scaled to the current event.

Saves SCRATCH/cmip6_ep_monthly_<model>.npz per model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brazil_flavor_composites as bfc                       # noqa: E402
import brazil_flavor_cmip6 as h                              # noqa: E402

MONTHS = [11, 12, 1, 2, 3]
EXT = bfc.EXTENT


def detrend_raw(s: pd.Series) -> pd.Series:
    x = s.index.to_numpy(float)
    A = np.column_stack([np.ones_like(x), x - 1950, np.clip(x - 1970, 0, None)])
    c, *_ = np.linalg.lstsq(A, s.values, rcond=None)
    return pd.Series(s.values - A @ c, index=s.index)


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

    acc = {m: None for m in MONTHS}
    n_ep, amp_sum = 0, 0.0
    lat = lon = None
    for k, mem in enumerate(mems):
        try:
            tas = fields(mem)
            trop_band = tas.sel(lat=slice(-22, 22)).load()
            sa = tas.sel(lat=slice(EXT[2] - 3, EXT[3] + 3),
                         lon=slice(EXT[0] % 360, EXT[1] % 360)).load()
        except Exception as e:                               # noqa: BLE001
            print(f"  {mem}: load failed ({repr(e)[:50]})", flush=True)
            continue
        clim = trop_band.groupby("time.month").mean("time")
        anom = trop_band.groupby("time.month") - clim
        trop = h.wmean(anom.sel(lat=slice(-20, 20)))
        n34r = h.ndjfm_series(h.rel_index(anom, -5, 5, 190, 240, trop))
        n34 = h.detrend_z(n34r)
        n34_c = detrend_raw(n34r)
        n12 = h.detrend_z(h.ndjfm_series(h.rel_index(anom, -10, 0, 270, 280, trop)))
        n4 = h.detrend_z(h.ndjfm_series(h.rel_index(anom, -5, 5, 160, 210, trop)))
        el = h.detrend_z(n12 - n4)
        ep_years = [int(y) for y in n34[n34 >= h.ZEV].index
                    if y in el.index and el[y] >= h.ZEP]

        sclim = sa.groupby("time.month").mean("time")
        sanom = sa.groupby("time.month") - sclim
        t = pd.DatetimeIndex(sanom["time"].values)
        lat, lon = sa["lat"].values, sa["lon"].values
        for m in MONTHS:
            sel = sanom.isel(time=(t.month == m))
            yrs = pd.DatetimeIndex(sel["time"].values).year - (1 if m <= 3 else 0)
            yc = yrs.to_numpy(float) - yrs.to_numpy(float).mean()
            v = sel.values
            sl = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
            dv = v - v.mean(0)[None] - yc[:, None, None] * sl[None]
            for y in ep_years:
                w = np.where(yrs == y)[0]
                if not w.size:
                    continue
                f = dv[w[0]]
                acc[m] = f.copy() if acc[m] is None else acc[m] + f
        for y in ep_years:
            if y in n34_c.index:
                n_ep += 1
                amp_sum += float(n34_c[y])
        # month counts can differ by ±1 at record edges; n_ep tracks NDJFM
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(mems)}; EP n={n_ep}", flush=True)

    lons180 = np.where(lon > 180, lon - 360, lon)
    np.savez(h.SCRATCH / f"cmip6_ep_monthly_{model}.npz",
             lat=lat, lon=lons180, n_ep=n_ep, amp_sum=amp_sum,
             **{f"m{m}": acc[m] for m in MONTHS})
    print(f"{model}: EP n={n_ep}, mean raw amp={amp_sum/max(n_ep,1):+.2f} °C",
          flush=True)


if __name__ == "__main__":
    for mdl in ("MPI-ESM1-2-LR", "EC-Earth3", "MIROC6"):
        run(mdl)
