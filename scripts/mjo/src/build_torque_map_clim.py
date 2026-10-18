#!/usr/bin/env python3
"""ERA5 1991-2020 torque-density map climatology of the live torque maps (torque_map_anim.py), from ARCO-ERA5.

Both terms in the forecast's own form, on its 5-degree grid (the 0.25-degree density block-averaged, coarsen 20,
boundary trim):
  friction  -a cos(phi) tau, tau = ERA5's own eastward turbulent surface stress (mean_eastward_turbulent_surface_stress,
            the quantity IFS-ENS publishes accumulated as ewss, turbulent orographic form drag included), the day's
            mean of the hourly means ending 00, 06, 12 and 18 UTC. The forecast's friction is the 24 h mean of its
            ewss, and four times a day leave neither the diurnal nor the semidiurnal cycle in the mean (four against
            24 times a day differ by under 0.2 Hadley globally, torque-ENSO study);
  mountain  h dp_s/dlambda on era5_orography.nc at 00 and 12 UTC, one coefficient set per hour: the forecast day is
            the trapezoid of its 12-hourly values, so it takes the mean of the two, and the semidiurnal tide's alias at
            00/12 UTC cancels in the anomaly.
Each is a mean and two annual harmonics of the common-year day (doycal.noleap_doy, period 365), fitted by least squares
per cell on every day of 1991-2020.

The friction climatology used to be a bulk stress, C_d rho |V10| u10 with C_d fitted per cell and month to ERA5's
stress at 00/12 UTC. A fit through the origin does not keep the mean: the bulk climatology's global friction torque
sat 8 Hadley above ERA5's own (11 in January), all of it over the rough land of 30-60N, a standing spurious spin-down
in every live anomaly. The forecast's stress is now compared with ERA5's stress itself.

One matrix job per chunk of days streams ARCO-ERA5 anonymously and writes its normal equations (torque-history.yml,
mode mapclim); --combine sums them, checks that every day was seen once, and solves.

    python src/build_torque_map_clim.py --start 1991-01-01 --end 1992-12-31 --out part.npz
    python src/build_torque_map_clim.py --combine parts/ --out data/reference/torque_map_clim_coeffs.nc
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from doycal import noleap_doy

Y0, Y1 = 1991, 2020
FRIC_HOURS = (0, 6, 12, 18)
MTN_HOURS = (0, 12)
NCOEF = 5
MAX_MISSING = 0.005  # a harmonic fit on ~11,000 days barely notices a few; more means the stream is broken
STRESS = "mean_eastward_turbulent_surface_stress"
SP = "surface_pressure"


def harmonics(dates) -> np.ndarray:
    """(n, 5) design matrix: mean and two annual harmonics of the common-year day."""
    w = 2 * np.pi * noleap_doy(dates) / 365.0
    return np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def evaluate(coef: np.ndarray, dates) -> np.ndarray:
    """The harmonic climatology `coef` (5, ...) at `dates` -> (n, ...)."""
    return np.tensordot(harmonics(dates), coef, axes=(1, 0))


def block5(x: np.ndarray) -> np.ndarray:
    """0.25-degree (721, 1440) ascending-latitude field -> its 5-degree block means (36, 72), the torque maps'
    coarsen(20, boundary='trim'), which drops the 90N row."""
    return x[:720].reshape(36, 20, 72, 20).mean((1, 3))


def run(a) -> int:
    from torque_history_era5 import A, Grid, _open  # streams ARCO (gcsfs): the chunk jobs only

    G = Grid()
    cos = np.cos(np.deg2rad(G.lat))[:, None]
    ds = _open()
    t0 = pd.Timestamp(ds.time.values[0])
    assert pd.Timestamp(ds.time.values[1]) - t0 == pd.Timedelta(hours=1)
    assert np.array_equal(ds.latitude.values[::-1].astype(np.float64), G.lat), "ARCO grid changed"
    assert np.array_equal(ds.longitude.values.astype(np.float64), G.lon), "ARCO grid changed"

    def field(name, t):
        i = int((t - t0) / pd.Timedelta(hours=1))
        x = np.asarray(ds[name].isel(time=i).values, dtype=np.float64)[::-1, :]  # ARCO latitude descending
        bad = ~np.isfinite(x)
        if bad.any():  # a stray NaN cell takes its latitude row's mean, as torque_history_era5; more rejects the day
            if bad.mean() > 5e-4:
                raise ValueError(f"{name} {t:%Y-%m-%d %HZ}: {bad.sum()} non-finite cells")
            x[bad] = np.broadcast_to(np.nanmean(np.where(bad, np.nan, x), axis=1)[:, None], x.shape)[bad]
        return x

    def one(d):
        """(day, friction density (36, 72), mountain densities (2, 36, 72)) or (day, None, reason)."""
        try:
            tau = np.mean([field(STRESS, d + pd.Timedelta(hours=h)) for h in FRIC_HOURS], axis=0)
            mtn = []
            for h in MTN_HOURS:
                sp = field(SP, d + pd.Timedelta(hours=h))
                if not (40000 < sp.min() and sp.max() < 110000):  # noqa: SIM300  interval test, low < x and x < high
                    raise ValueError(f"sp {d:%Y-%m-%d} {h:02d}Z out of range {sp.min():.0f}..{sp.max():.0f}")
                mtn.append(block5(G.h * (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * G.dl)))
        except Exception as e:  # noqa: BLE001
            return d, None, f"{type(e).__name__}: {e}"[:160]
        return d, (block5(-(A * cos) * tau), np.stack(mtn)), None

    days = pd.date_range(a.start, a.end, freq="D")
    print(f"{len(days)} days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}, {a.workers} workers", flush=True)
    got, fric, mtn, bad, tstart = [], [], [], [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (d, out, err) in enumerate(ex.map(one, days)):
            if out is None:
                bad.append(d)
                print(f"  skip {d:%Y-%m-%d}: {err}", flush=True)
            else:
                got.append(d)
                fric.append(out[0])
                mtn.append(out[1])
            if n % 100 == 0:
                print(f"  {n}/{len(days)} {d:%Y-%m-%d}  {(time.time() - tstart) / 60:.1f} min", flush=True)
    X = harmonics(pd.DatetimeIndex(got))
    fric, mtn = np.stack(fric), np.stack(mtn)  # (n, 36, 72), (n, 2, 36, 72)
    np.savez(
        a.out,
        days=pd.DatetimeIndex(got).values.astype("datetime64[D]").astype(np.int64),
        xtx=X.T @ X,
        xty_fric=np.tensordot(X, fric, axes=(0, 0)),
        xty_mtn=np.tensordot(X, mtn, axes=(0, 0)),
        clat=G.clat,
        clon=G.clon,
    )
    print(f"wrote {a.out}: {len(got)} of {len(days)} days, {(time.time() - tstart) / 60:.1f} min", flush=True)
    return 0 if len(bad) <= MAX_MISSING * len(days) else 1


def combine(a) -> int:
    fs = sorted(glob.glob(str(Path(a.combine) / "*.npz")))
    parts = [np.load(f) for f in fs]
    days = np.concatenate([p["days"] for p in parts])
    want = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D").values.astype("datetime64[D]").astype(np.int64)
    if len(np.unique(days)) != len(days) or np.setdiff1d(days, want).size:
        raise SystemExit(f"a day appears in two parts or lies outside {Y0}-{Y1}")
    miss = np.setdiff1d(want, days).astype("datetime64[D]")
    if miss.size > MAX_MISSING * want.size:
        raise SystemExit(f"{miss.size} of {want.size} days missing (first {miss[:5]})")
    if miss.size:
        print(f"::warning::{miss.size} days missing: {', '.join(map(str, miss[:20]))}")
    clat, clon = parts[0]["clat"], parts[0]["clon"]
    xtx = sum(p["xtx"] for p in parts)
    fric = np.linalg.solve(xtx, sum(p["xty_fric"] for p in parts).reshape(NCOEF, -1)).reshape(NCOEF, *clat.shape, -1)
    xty_m = sum(p["xty_mtn"] for p in parts)  # (5, hour, lat, lon)
    mtn = np.stack([np.linalg.solve(xtx, xty_m[:, j].reshape(NCOEF, -1)).reshape(fric.shape) for j in range(2)])
    xr.Dataset(
        {
            "friction": (("coef", "latitude", "longitude"), fric),
            "mountain": (("hour", "coef", "latitude", "longitude"), mtn),
        },
        coords={"hour": list(MTN_HOURS), "coef": np.arange(NCOEF), "latitude": clat, "longitude": clon},
        attrs={
            "note": (
                f"ERA5 {Y0}-{Y1} torque-density harmonic climatology on the 5-degree map grid (N m per m^2): mean and "
                "two annual harmonics of the common-year day, period 365. friction = -a cos(phi) x ERA5 "
                "mean_eastward_turbulent_surface_stress, the daily mean of 00/06/12/18 UTC; mountain = h dp_s/dlambda "
                "per hour, 00 and 12 UTC. Every day, from ARCO-ERA5 (build_torque_map_clim.py)."
            ),
            "days": int(days.size),
        },
    ).to_netcdf(a.out)
    print(f"combined {len(fs)} parts, {days.size} days -> {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default=None, help="directory of part files to combine")
    a = ap.parse_args()
    return combine(a) if a.combine else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
