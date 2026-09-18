#!/usr/bin/env python3
"""Climatology of the 100 hPa eddy heat flux v'T', 45-75 deg, both hemispheres.

NCEP/NCAR R1 daily v and T at 100 hPa (PSL OPeNDAP, read SEQUENTIALLY: parallel
reads of that server return zeros without raising), 1991-2020. Per day and
hemisphere: the zonal-mean flux [v'T'] averaged 45-75 deg with cos(lat) weights,
for all waves and for zonal wavenumbers 1 and 2 (the k-th contribution is
2 Re(V_k conj(T_k)) / N^2, which sums to the total over k >= 1). The SH series is
sign-flipped so POSITIVE IS POLEWARD in both hemispheres.

Day-of-year statistics use a +/-15-day window: mean, sd, p10, p90 of the daily
values, and the mean and sd of the trailing 40-day mean (the quantity the vortex
responds to). Output scripts/strat/reference/vt100_clim.nc; the yearly series
are cached in scripts/strat/data/vt100_r1/ so a rerun only fetches what is missing.

    python scripts/strat/build_vt100_clim.py [--y0 1991 --y1 2020]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference" / "vt100_clim.nc"
CACHE = HERE / "data" / "vt100_r1"
PSL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Dailies/pressure"
KEYS = ("tot", "k1", "k2")


def band_flux(v: np.ndarray, T: np.ndarray, lat: np.ndarray) -> dict:
    """v, T (time, lat, lon) -> {tot, k1, k2} (time,), cos-weighted over the latitudes given."""
    n = v.shape[-1]
    V = np.fft.rfft(v, axis=-1); Tt = np.fft.rfft(T, axis=-1)
    per_k = 2.0 * np.real(V * np.conj(Tt)) / n ** 2           # (time, lat, k); k=0 is the mean part
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    return {"tot": (per_k[..., 1:].sum(-1) * w).sum(-1),
            "k1": (per_k[..., 1] * w).sum(-1), "k2": (per_k[..., 2] * w).sum(-1)}


def year_series(y: int) -> xr.Dataset:
    p = CACHE / f"vt100_{y}.nc"
    if p.exists():
        return xr.open_dataset(p).load()
    out = {}
    for hemi, sl, sign in (("nh", slice(75, 45), 1.0), ("sh", slice(-45, -75), -1.0)):
        v = xr.open_dataset(f"{PSL}/vwnd.{y}.nc")["vwnd"].sel(level=100, lat=sl).compute()
        T = xr.open_dataset(f"{PSL}/air.{y}.nc")["air"].sel(level=100, lat=sl).compute()
        a, b = np.asarray(v.values, float), np.asarray(T.values, float)
        # the server's failure mode is silent zeros / partial masks: refuse them
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or (np.abs(a).max(axis=(1, 2)) == 0).any() or b.min() < 150:
            raise RuntimeError(f"{y} {hemi}: corrupt read (zeros or masked values)")
        f = band_flux(a, b, v.lat.values)
        for k in KEYS:
            out[f"{hemi}_{k}"] = ("time", sign * f[k])
        t = pd.DatetimeIndex(v.time.values).normalize()
    ds = xr.Dataset(out, coords={"time": t})
    CACHE.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(p)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    parts = []
    for y in range(a.y0, a.y1 + 1):
        t0 = time.time()
        for attempt in range(3):
            try:
                parts.append(year_series(y)); break
            except Exception as e:                                   # noqa: BLE001
                print(f"  {y}: {str(e).splitlines()[0][:90]} (attempt {attempt + 1})", flush=True); time.sleep(20)
        else:
            raise SystemExit(f"{y} could not be read")
        print(f"  {y}: NH mean {float(parts[-1]['nh_tot'].mean()):+.1f}, SH {float(parts[-1]['sh_tot'].mean()):+.1f} K m/s  ({time.time() - t0:.0f} s)", flush=True)
    ds = xr.concat(parts, dim="time").sortby("time")
    doy = pd.DatetimeIndex(ds.time.values).dayofyear.values
    out = {}
    for name in ds.data_vars:
        x = ds[name].values
        run40 = pd.Series(x).rolling(40, min_periods=40).mean().values
        st = {k: np.full(366, np.nan) for k in ("mean", "sd", "p10", "p90", "m40", "sd40")}
        for d in range(1, 367):
            dist = np.abs(((doy - d) + 183) % 366 - 183)
            m = dist <= 15
            st["mean"][d - 1] = x[m].mean(); st["sd"][d - 1] = x[m].std()
            st["p10"][d - 1], st["p90"][d - 1] = np.percentile(x[m], [10, 90])
            r = run40[m & np.isfinite(run40)]
            st["m40"][d - 1] = r.mean(); st["sd40"][d - 1] = r.std()
        for k, v in st.items():
            out[f"{name}_{k}"] = ("doy", v.astype("float32"))
    res = xr.Dataset(out, coords={"doy": np.arange(1, 367)})
    res.attrs.update(source="NCEP/NCAR Reanalysis 1 daily v and T at 100 hPa (NOAA PSL)", period=f"{a.y0}-{a.y1}",
                     band="45-75 deg, cos(lat) weighted", units="K m s-1", sign="positive = poleward in both hemispheres",
                     window="+/-15 days; m40/sd40 are of the trailing 40-day mean")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
