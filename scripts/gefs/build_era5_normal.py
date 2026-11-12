#!/usr/bin/env python3
"""ERA5 reference of a pressure-level field that joins the GEFS page (gefs_daily.D2) after its first build: the
re-basing shift and the zonal-mean normal, from WeatherBench2's 1.5 deg ERA5 (streamed; one year in memory at a time).

The daily mean of the four synoptic hours is pooled within +-7 days of each 5-day centre on the common-year calendar
(gefs_reforecast.noleap_doy), the recipe of the other fields' normals (scripts/geps/build/build_era5_clim.py), on the
GEFS 1.5 deg grid (gefs_reforecast.LAT 90 -> -90, LON 0E first):

  s15_<tag>  the 1991-2020 normal minus the 2001-2020 normal (2001-2020 stands in for the reforecast's 2000-2019),
             added to era5_shift.npz of release gefs-ref-v1: gefs_live re-bases the anomalies to 1991-2020 with it
  z_<tag>    the zonal mean of the 1991-2020 normal, added to data/era5_zonal_normals.npz: the observed seasonal march
             that gefs_drift sets the reforecast's drift against

Both files keep every array they hold.

    python build_era5_normal.py t850 --shift ref/era5_shift.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gefs_drift as GD  # (the zonal normals' file)
import gefs_reforecast as R  # (the grid, the day-of-year key, the pools)

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
FIELDS = {"t850": ("temperature", 850)}  # tag of gefs_daily.D2 -> (WeatherBench2 variable, level in hPa); GEFS units
NORMAL, REFORECAST = (1991, 2020), (2001, 2020)
HALF = 7  # days either side of a centre


def year_days(da, y: int):
    """(day-of-year keys, daily means (day, 121, 240) float64) of one year, on the GEFS grid."""
    day = da.sel(time=slice(f"{y}-01-01", f"{y}-12-31")).resample(time="1D").mean()
    day = day.transpose("time", "latitude", "longitude").sortby("latitude", ascending=False).compute()
    assert np.allclose(day.latitude, R.LAT) and np.allclose(day.longitude, R.LON), "not the GEFS 1.5 deg grid"
    v = day.values.astype("float64")
    assert np.isfinite(v).all(), f"{y}: missing values"
    return R.noleap_doy(day.time.values), v


def normals(tag: str):
    """The 5-day centres and the (73, 121, 240) normals of NORMAL and of REFORECAST."""
    var, level = FIELDS[tag]
    da = xr.open_zarr(WB2, storage_options={"token": "anon"})[var].sel(level=level)
    centres = np.arange(1, 366, 5)
    periods = (NORMAL, REFORECAST)
    tot = np.zeros((len(periods), len(centres), R.LAT.size, R.LON.size))
    cnt = np.zeros((len(periods), len(centres)), dtype=int)
    for y in range(min(p[0] for p in periods), max(p[1] for p in periods) + 1):
        t0 = time.time()
        doy, v = year_days(da, y)
        for k, c in enumerate(centres):
            m = R.pool(doy, c, HALF)
            s = v[m].sum(0)
            for i, (y0, y1) in enumerate(periods):
                if y0 <= y <= y1:
                    tot[i, k] += s
                    cnt[i, k] += m.sum()
        print(f"  {tag} {y}: {len(doy)} days, mean {v.mean():.2f} in {time.time() - t0:.0f} s", flush=True)
    return centres, *(tot / cnt[:, :, None, None])


def add(path: Path, doy, **arrays) -> None:
    """Add float32 arrays on the 5-day centres `doy` to an npz of such arrays, keeping the ones it holds."""
    z = dict(np.load(path))
    assert np.array_equal(z["doy"], doy), f"{path.name}: other centres"
    z.update({k: v.astype("float32") for k, v in arrays.items()})
    np.savez_compressed(path, **z)
    print(f"{path.name}: {', '.join(arrays)} added, {len(z)} arrays")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=sorted(FIELDS))
    ap.add_argument("--shift", required=True, help="era5_shift.npz of release gefs-ref-v1, updated in place")
    a = ap.parse_args()
    doy, full, late = normals(a.tag)
    shift = full - late
    lat = np.cos(np.deg2rad(R.LAT))[None, :, None]
    print(
        f"{a.tag}: {NORMAL[0]}-{NORMAL[1]} normal, global mean {(full * lat).sum() / (lat.sum() * full.shape[0] * R.LON.size):.2f}; "
        f"minus {REFORECAST[0]}-{REFORECAST[1]}: mean {shift.mean():+.3f}, range {shift.min():+.2f} to {shift.max():+.2f}"
    )
    add(Path(a.shift), doy, **{f"s15_{a.tag}": shift})
    add(GD.ZN, doy, **{f"z_{a.tag}": full.mean(2)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
