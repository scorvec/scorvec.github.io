#!/usr/bin/env python3
"""ERA5 climatology for the subtropical-jet monitor: zonal-mean zonal wind.

Two things, both from WeatherBench2 1.5° ERA5 (1991–2020, 12Z, strided):
  1. [u](level, latitude) harmonic (mean+annual+semiannual) day-of-year coeffs —
     for the cross-section anomaly shading in jets.py.
  2. Per-hemisphere subtropical jet-core metrics (strength = max [u]@200 hPa in
     15–45°|lat|, and its latitude) per sample → harmonic day-of-year mean AND a
     harmonically-smoothed σ(doy) of the residuals — the "vs normal" envelope.

    python src/build_jet_clim.py --stride 3
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

import sys
sys.path.insert(0, str(Path(__file__).parent))

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "jet_clim.nc"
CACHE = Path.home() / "mjo" / "era5_cache"
STORE = ("gs://weatherbench2/datasets/era5/"
         "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
JET_BAND = (15.0, 45.0)                        # |lat| window of the subtropical core
JET_LEV = 200.0


def jet_metrics(ubar_200: np.ndarray, lat: np.ndarray):
    """(nh_speed, nh_lat, sh_speed, sh_lat) from [u] at 200 hPa. The core must be
    an INTERIOR local maximum of the band — when the in-band max sits on the band
    edge (NH midsummer: the zonal wind keeps rising poleward of 45° into the
    merged eddy-driven jet) there is no closed subtropical core and the metrics
    are NaN rather than a value pinned to the boundary. The core latitude is
    refined with a parabolic fit through the max and its neighbours, so the
    1.5°/0.25° grids agree to a fraction of a degree."""
    out = []
    for lo, hi in ((JET_BAND[0], JET_BAND[1]), (-JET_BAND[1], -JET_BAND[0])):
        m = (lat >= lo) & (lat <= hi)
        seg, sl = ubar_200[m], lat[m]
        peaks = [i for i in range(1, len(seg) - 1)
                 if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
        if not peaks or np.nanmax(seg[peaks]) < np.nanmax(seg[[0, -1]]):
            out += [float("nan"), float("nan")]     # STJ indistinct this day
            continue
        i = int(max(peaks, key=lambda j: seg[j]))
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        denom = (y0 - 2 * y1 + y2)
        off = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        off = float(np.clip(off, -1, 1))
        speed = y1 - 0.25 * (y0 - y2) * off
        latc = sl[i] + off * (sl[1] - sl[0])
        out += [float(speed), float(latc)]
    return out


def _samples(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"jet_ubar_{a.y0}-{a.y1}_s{a.stride}.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        return c["ubar"].values, c.latitude.values, pd.to_datetime(c.time.values)
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None)
    u = ds["u_component_of_wind"].sel(level=LEVELS).sortby("level")
    want = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D") + pd.Timedelta(hours=12)
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, {a.y0}-{a.y1})", flush=True)
    ulist, tlist, lat = [], [], None
    for n, t in enumerate(times):
        try:
            ut = u.sel(time=t)
            ulist.append(ut.mean("longitude").values); tlist.append(t)
            lat = ut.latitude.values
        except Exception as e:                                     # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {repr(e)[:60]}"); continue
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)
    arr = np.stack(ulist)
    xr.Dataset({"ubar": (("time", "level", "latitude"), arr)},
               coords={"time": pd.DatetimeIndex(tlist), "level": LEVELS, "latitude": lat}
               ).to_netcdf(cpath)
    print(f"saved ERA5 samples → {cpath}  ({arr.nbytes/1e6:.0f} MB)", flush=True)
    return arr, lat, pd.DatetimeIndex(tlist)


def _harmfit(X, y):
    """lstsq coefficient fit, y (ntime, ...) flattened on trailing dims.
    NaN samples (indistinct-jet days in the core metrics) are dropped per column."""
    shp = y.shape
    yf = y.reshape(shp[0], -1)
    if np.isnan(yf).any():
        c = np.empty((X.shape[1], yf.shape[1]))
        for j in range(yf.shape[1]):
            ok = np.isfinite(yf[:, j])
            c[:, j] = np.linalg.lstsq(X[ok], yf[ok, j], rcond=None)[0]
    else:
        c = np.linalg.lstsq(X, yf, rcond=None)[0]
    return c.reshape((X.shape[1],) + shp[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    ubar, lat, times = _samples(a)
    doy = times.dayofyear.values; w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])

    cu = _harmfit(X, ubar)                                        # (5, lev, lat)

    k200 = LEVELS.index(int(JET_LEV))
    mets = np.array([jet_metrics(ubar[i, k200], lat) for i in range(len(times))])
    cm = _harmfit(X, mets)                                        # (5, 4)
    resid = mets - X @ cm.reshape(X.shape[1], -1)
    # σ(doy): harmonic fit of the squared residuals (mean+annual+semiannual is
    # plenty for the smooth seasonal march of jet variability), floored at 10%
    # of the overall σ so the envelope can never pinch to zero.
    cv = _harmfit(X, resid ** 2)                                  # (5, 4)
    ds = xr.Dataset(
        {"u_coeffs": (("coef", "level", "latitude"), cu),
         "met_coeffs": (("coef", "metric"), cm),
         "var_coeffs": (("coef", "metric"), cv)},
        coords={"coef": np.arange(5), "level": LEVELS, "latitude": lat,
                "metric": ["nh_speed", "nh_lat", "sh_speed", "sh_lat"]},
        attrs={"note": f"ERA5 {a.y0}-{a.y1} zonal-mean-u + subtropical-jet-core "
                       f"harmonic clim (WB2 1.5°, 12Z, {a.stride}-day stride); "
                       f"metrics = max [u]@200 in 15-45°|lat| (parabolic refine)",
               "floor_frac": 0.1})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.nc"); ds.to_netcdf(tmp); tmp.replace(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
