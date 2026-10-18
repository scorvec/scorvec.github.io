"""ERA5 truth (ERA5 daily store, NH 1.5 deg) for the CFS comparison: weekly anomalies vs the 1991-2020 daily normal
(31-day running mean) over days 1-35 counted from the 1st; prcp ends 2023 (later years NaN)."""

import pyproj  # noqa: F401  before xarray/eccodes: eccodes preloads its own libproj
import sys
import os
import datetime as dt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import YEARS, MONTHS, OUT, weekly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lib"))
from localdirs import env_dir

ED = str(env_dir("ERA5_STORE", "wb2_1p5_daily")) + "/{v}/{v}_{y}.nc"
EV = {"t2m": "t2m", "z500": "z500", "pr": "prcp"}
LAT = np.arange(0.0, 90.0 + 1e-6, 1.5)
LON = np.arange(0.0, 360.0 - 1e-6, 1.5)


def load(v, y):
    f = ED.format(v=v, y=y)
    if not os.path.exists(f):
        return None
    da = xr.open_dataset(f)[v].transpose("time", "latitude", "longitude")
    da = da.reindex(latitude=LAT, longitude=LON, method="nearest", tolerance=0.01)  # float noise in the coordinates
    return {dt.date.fromisoformat(str(x)[:10]): i for i, x in enumerate(da.time.values)}, da.values.astype(np.float64)


def main():
    def key(d):
        return (d.month, d.day)  # 29 Feb: no key of its own, read as the mean of 28 Feb and 1 March

    days = [dt.date(2001, 1, 1) + dt.timedelta(days=i) for i in range(365)]
    kidx = {key(d): i for i, d in enumerate(days)}
    out = {}
    for t, v in EV.items():
        acc = np.zeros((365, 61, 240))
        n = np.zeros(365)
        for y in range(1991, 2021):
            ix, a = load(v, y)
            for d, i in ix.items():
                if key(d) == (2, 29):
                    continue
                j = kidx[key(d)]
                acc[j] += a[i]
                n[j] += 1
        clim = acc / n[:, None, None]
        pad = np.concatenate([clim[-15:], clim, clim[:15]])
        cs = np.cumsum(np.concatenate([np.zeros((1, 61, 240)), pad]), axis=0)
        sm = (cs[31:] - cs[:-31]) / 31.0
        if t == "t2m":
            assert 250 < sm[kidx[(10, 1)]].mean() < 300
        if t == "z500":
            assert 5000 < sm[kidx[(10, 1)]].mean() < 6000
        if t == "pr":
            assert 0.5 < sm[kidx[(10, 1)]].mean() < 6  # mm/day despite the attribute

        def norm(d, sm=sm):
            if key(d) == (2, 29):
                return 0.5 * (sm[kidx[(2, 28)]] + sm[kidx[(3, 1)]])
            return sm[kidx[key(d)]]

        cache = {}
        arr = np.full((2, len(YEARS), 4, 61, 240), np.nan, np.float32)
        for mi, m in enumerate(MONTHS):
            for yi, y in enumerate(YEARS):
                d0 = dt.date(y, m, 1)
                ds_ = [d0 + dt.timedelta(days=k) for k in range(35)]
                rows = []
                for d in ds_:
                    if d.year not in cache:
                        cache[d.year] = load(v, d.year)
                    c = cache[d.year]
                    if c is None or d not in c[0]:
                        rows = None
                        break
                    rows.append(c[1][c[0][d]] - norm(d))
                if rows is None:
                    continue
                arr[mi, yi] = weekly(np.stack(rows))
        out[t] = arr
        print(
            "ERA5",
            t,
            "cases",
            int(np.isfinite(arr[:, :, 0, 30, 0]).sum()),
            "anom sd wk1",
            float(np.nanstd(arr[:, :, 0])),
            flush=True,
        )
    np.savez_compressed(OUT + "era5_truth2.npz", years=np.array(YEARS), **out)


if __name__ == "__main__":
    main()
