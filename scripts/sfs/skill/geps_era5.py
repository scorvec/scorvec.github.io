"""GEPS8 hindcast (IRI SubX archive, 4-member MEAN) weekly means for the Sep 1 / Oct 1 starts, and the ERA5 truth:
weekly anomalies vs the 1991-2020 daily climatology (31-day running mean) over each system's own valid days."""

import pyproj  # noqa: F401  before xarray/eccodes: eccodes preloads its own libproj
import os
import sys
import datetime as dt
import numpy as np
import xarray as xr

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import YEARS, MONTHS, OUT, weekly, regrid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from localdirs import env_dir

GD = str(env_dir("DATA_ARCHIVE", "geps_subx", "geps8_hindcast")) + "/geps8_{t}_{ym}.nc"
GV = {"t2m": ("t2m", "tas", 1.0), "z500": ("zg500", "zg", 1.0), "pr": ("pr", "pr", 86400.0)}
ED = str(env_dir("ERA5_STORE", "wb2_1p5_daily")) + "/{v}/{v}_{y}.nc"
EV = {"t2m": "t2m", "z500": "z500", "pr": "prcp"}


def geps():
    res = {t: np.full((2, len(YEARS), 4, 61, 240), np.nan, np.float32) for t in GV}
    used = {}
    for t, (tag, var, fac) in GV.items():
        for m in MONTHS:
            for y in YEARS:
                ds = xr.open_dataset(GD.format(t=tag, ym=f"{y}{m:02d}"))
                S = ds["S"].values.astype("datetime64[D]")
                want = np.datetime64(f"{y}-{m:02d}-01")
                k = int(np.argmin(np.abs(S - want)))
                if abs((S[k] - want).astype(int)) > 3:
                    continue
                da = ds[var].isel(S=k)
                if "P" in da.dims:
                    da = da.isel(P=0)
                a = da.transpose("L", "Y", "X").values[:35].astype(np.float32) * fac  # L 0.5 .. 34.5 = days 1..35
                a = np.where(np.abs(a) > 1e19, np.nan, a)
                res[t][MONTHS.index(m), YEARS.index(y)] = regrid(weekly(a), ds["Y"].values, ds["X"].values)
                used[(m, y)] = str(S[k])
    np.savez_compressed(
        OUT + "geps_weekly.npz", **res, starts=np.array([[used.get((m, y), "") for y in YEARS] for m in MONTHS])
    )
    print("GEPS done", sum(1 for v in used.values() if v), "starts")


def era5():
    gef = np.load(OUT + "gefs_weekly.npz") if __import__("os").path.exists(OUT + "gefs_weekly.npz") else None
    starts = {"sfs": [[f"{y}{m:02d}01" for y in YEARS] for m in MONTHS]}
    starts["geps"] = starts["sfs"]  # GEPS uses the exact Sep 1 / Oct 1 starts (checked in geps())
    if gef is not None:
        starts["gefs"] = [[str(s) for s in row] for row in gef["starts"]]

    def key(d):
        return (d.month, d.day)

    out = {}
    for t, v in EV.items():
        # day-of-year climatology 1991-2020 (calendar key), then 31-day circular running mean; 29 Feb has no key of
        # its own: left out of the daily means and read as the mean of 28 Feb and 1 March (norm)
        days = [dt.date(2001, 1, 1) + dt.timedelta(days=i) for i in range(365)]
        kidx = {key(d): i for i, d in enumerate(days)}
        acc = np.zeros((365, 61, 240))
        n = np.zeros(365)
        for y in range(1991, 2021):
            ds = xr.open_dataset(ED.format(v=v, y=y))
            da = ds[v].transpose("time", "latitude", "longitude")
            assert np.allclose(da.latitude.values, np.arange(0, 90.01, 1.5)), "ERA5 lat order changed"
            tt = [dt.date.fromisoformat(str(x)[:10]) for x in da.time.values]
            a = da.values
            for i, d in enumerate(tt):
                if key(d) == (2, 29):
                    continue
                j = kidx[key(d)]
                acc[j] += a[i]
                n[j] += 1
        clim = acc / n[:, None, None]
        pad = np.concatenate([clim[-15:], clim, clim[:15]])
        cs = np.cumsum(np.concatenate([np.zeros((1, 61, 240)), pad]), axis=0)
        sm = (cs[31:] - cs[:-31]) / 31.0

        def norm(d, sm=sm, kidx=kidx):
            if key(d) == (2, 29):
                return 0.5 * (sm[kidx[(2, 28)]] + sm[kidx[(3, 1)]])
            return sm[kidx[key(d)]]

        cache = {}

        def daily(d, cache=cache, v=v):
            if d.year not in cache:
                ds = xr.open_dataset(ED.format(v=v, y=d.year))
                da = ds[v].transpose("time", "latitude", "longitude")
                cache[d.year] = (
                    {dt.date.fromisoformat(str(x)[:10]): i for i, x in enumerate(da.time.values)},
                    da.values,
                )
            ix, a = cache[d.year]
            return a[ix[d]] - norm(d)

        for sysn, st in starts.items():
            arr = np.full((2, len(YEARS), 4, 61, 240), np.nan, np.float32)
            clw = np.full((2, len(YEARS), 4, 61, 240), np.nan, np.float32)
            for mi, row in enumerate(st):
                for yi, s in enumerate(row):
                    if not s:
                        continue
                    d0 = dt.datetime.strptime(s[:10].replace("-", ""), "%Y%m%d").date()
                    ds_ = [d0 + dt.timedelta(days=k) for k in range(35)]
                    arr[mi, yi] = weekly(np.stack([daily(d) for d in ds_]))
                    clw[mi, yi] = weekly(np.stack([norm(d) for d in ds_]))  # ERA5 1991-2020 normal, same valid days
            out[f"{sysn}_{t}"] = arr
            out[f"clim_{sysn}_{t}"] = clw
        print("ERA5", t, "done", flush=True)
    np.savez_compressed(OUT + "era5_truth.npz", **out)


if __name__ == "__main__":
    {"geps": geps, "era5": era5}[sys.argv[1]]()
