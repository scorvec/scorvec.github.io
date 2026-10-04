#!/usr/bin/env python3
"""Recent-decade normal and terciles for the probability maps: the median and
the 33rd/67th percentiles of the 7-day-mean anomaly (vs the 1991-2020
climatology) over the last ten full years, by day of year, plus the absolute
1991-2020 weekly climatology (for precipitation's dry-season mask). Against the 1991-2020 median itself, P(above normal) in 2026 reads
>90% over every ocean and most land before the forecast says anything — the
warming since the base period, not weather. A ten-year normal is CPC's
"optimal climate normal" idea applied to the threshold.

    python recent_median.py --tags t2m,pr
    -> telecon/recent_median_{tag}.nc   median_recent[doy, lat, lon]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS                             # noqa: E402  laptop-only inputs/outputs
from hindcast_maps import truth_daily, TC, LAT25, LON25, CY0, CY1     # noqa: E402
from build_telecon_patterns import doy_clim                             # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="t2m,pr")
    ap.add_argument("--years", default="2016-2025"); a = ap.parse_args()
    y0, y1 = [int(x) for x in a.years.split("-")]
    for tag in a.tags.split(","):
        t0 = time.time()
        tr = truth_daily(tag, list(range(CY0, CY1 + 1)) + list(range(max(CY1 + 1, y0), y1 + 1)))
        clim = doy_clim(tr.sel(time=slice(f"{CY0}-01-01", f"{CY1}-12-31")))
        rec = tr.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        an = rec.values - clim.values[rec.time.dt.dayofyear.values - 1]
        k = np.ones(7) / 7.0
        wk = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 0, np.nan_to_num(an))
        cnt = np.apply_along_axis(lambda x: np.convolve(x, np.ones(7), mode="valid"), 0, np.isfinite(an).astype(float))
        wk = np.where(cnt >= 5, wk * 7.0 / np.maximum(cnt, 1), np.nan)
        t = pd.DatetimeIndex(rec.time.values)[:wk.shape[0]]
        doy = t.dayofyear.values + 3
        med = np.full((366, LAT25.size, LON25.size), np.nan, "float32")
        q33 = med.copy(); q67 = med.copy(); mean = med.copy()
        for d in range(1, 367):
            dist = np.abs(((doy - d + 183) % 366) - 183)
            sel = wk[dist <= 15]
            med[d - 1], q33[d - 1], q67[d - 1] = np.nanpercentile(sel, [50, 100 / 3, 200 / 3], axis=0)
            mean[d - 1] = np.nanmean(sel, axis=0)                  # the neutral point of prob_maps' calibration
        # absolute weekly climatology (1991-2020), for the dry-season mask
        cw = np.stack([np.nanmean(clim.values[[(d - 1 + k) % 366 for k in range(-3, 4)]], axis=0)
                       for d in range(1, 367)]).astype("float32")
        ds = xr.Dataset({"median_recent": (("doy", "lat", "lon"), med),
                         "q33_recent": (("doy", "lat", "lon"), q33),
                         "q67_recent": (("doy", "lat", "lon"), q67),
                         "mean_recent": (("doy", "lat", "lon"), mean),
                         "clim_weekly": (("doy", "lat", "lon"), cw)},
                        coords={"doy": np.arange(1, 367), "lat": LAT25, "lon": LON25})
        ds.attrs.update(tag=tag, years=a.years, note="median of 7-day-mean anomaly vs 1991-2020 clim, +-15 d window")
        ds.to_netcdf(TC / f"recent_median_{tag}.nc")
        print(f"  {tag}: recent median {a.years}, global mean offset {np.nanmean(med):+.2f}, {time.time()-t0:.0f} s")

if __name__ == "__main__":
    raise SystemExit(main())
