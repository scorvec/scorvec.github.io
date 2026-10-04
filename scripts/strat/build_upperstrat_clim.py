#!/usr/bin/env python3
"""Day-of-year climatology of the upper-stratosphere diagnostics (2026-09-27; user: "let's add GDPS, GEFS, GFS and
GEOS-FP (allow the user to toggle between and compare the upper stratosphere)").

Two quantities per hemisphere and level, 100 -> 1 hPa: the zonal-mean zonal wind at 60 deg (the sudden-warming
criterion latitude) and the polar-cap temperature, the cos-weighted mean of the zonal-mean temperature over 60-90 deg.
For each day of the year: the mean (a six-harmonic fit to every sample, so a steep season such as the southern final
warming keeps its shape) and the 10th/90th percentiles (the mean plus the quantiles of the anomalies within a window
around that day, smoothed over 15 days).

Sources (both MERRA-2 meteorology, both already on this laptop, nothing is fetched):
  NH  the daily-mean MERRA-2 zonal means in scripts/telecon/data/m2_strat (40-90N, 42 levels to 0.1 hPa)
  SH  the MERRA-2 GMI replay zonal means in scripts/strat/data/m2gmi_tem (all latitudes, 44 levels, 00Z every 10th day,
      1991-2019; GMI replays MERRA-2's meteorology, and the NH check below measures how closely)

BASE PERIOD. Not 1991-2020: MERRA-2 starts assimilating Aura MLS temperatures above 5 hPa in August 2004, and its
polar-cap temperature steps by about +4 K at 1 hPa and -4 K at 2-3 hPa across that date (annual-mean anomalies of the
NH cap, printed by --check-step). A 1991-2020 mean would put a 2 K artefact into every anomaly near the stratopause.
The reference is therefore the MLS era only: NH 2005-2024 (daily), SH 2005-2019 (the replay record ends in 2019).

    python scripts/strat/build_upperstrat_clim.py            # -> scripts/strat/reference/upperstrat_clim.nc (laptop, once)
    python scripts/strat/build_upperstrat_clim.py --check-step
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
M2 = REPO / "scripts" / "telecon" / "data" / "m2_strat"
GMI = HERE / "data" / "m2gmi_tem"
OUT = HERE / "reference" / "upperstrat_clim.nc"
LEVELS = [100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0]
PERIOD = {"nh": (2005, 2024), "sh": (2005, 2019)}
WINDOW = {"nh": 10, "sh": 20}            # +/- days for the percentile anomalies (daily vs every-10th-day sampling)
NHARM = 6


def doy365(t: pd.DatetimeIndex) -> np.ndarray:
    """0..364; 29 Feb shares 28 Feb's slot."""
    d = t.dayofyear.values - 1
    leap = t.is_leap_year & (t.month > 2)
    return np.where(leap, d - 1, d)


def cap_mean(tbar: np.ndarray, lat: np.ndarray, north: bool) -> np.ndarray:
    m = lat >= 60.0 if north else lat <= -60.0
    w = np.cos(np.deg2rad(lat[m]))
    return (tbar[..., m] * w).sum(-1) / w.sum()


def nh_series(y0: int, y1: int):
    fs = [f for f in sorted(glob.glob(str(M2 / "m2_strat_*.nc"))) if y0 <= int(f[-7:-3]) <= y1]
    d = xr.open_mfdataset(fs, combine="by_coords")[["ubar", "Tbar"]].sel(lev=LEVELS, method="nearest").load()
    u = d.ubar.sel(lat=60.0).values
    T = cap_mean(d.Tbar.values, d.lat.values, True)
    return pd.DatetimeIndex(d.time.values), u, T


def gmi_series(y0: int, y1: int, north: bool):
    ts, us, Ts = [], [], []
    for y in range(y0, y1 + 1):
        z = np.load(GMI / f"tem_{y}.npz")
        lev, lat = z["lev"], z["lat"]
        li = [int(np.argmin(np.abs(lev - L))) for L in LEVELS]
        j60 = int(np.argmin(np.abs(lat - (60.0 if north else -60.0))))
        ts.append(pd.DatetimeIndex(z["time"]))
        us.append(z["ubar"][:, li, j60])
        Ts.append(cap_mean(z["tbar"][:, li, :], lat, north))
    return ts[0].append(ts[1:]), np.concatenate(us), np.concatenate(Ts)


def screen(t, u, T):
    ok = np.isfinite(u).all(1) & np.isfinite(T).all(1) & (np.abs(u) < 200).all(1) & (T > 150).all(1) & (T < 330).all(1) \
         & ~(u == 0).all(1)
    if (~ok).sum():
        print(f"  screened out {(~ok).sum()} of {len(ok)} samples")
    return t[ok], u[ok], T[ok]


def climatology(t: pd.DatetimeIndex, x: np.ndarray, window: int):
    """x (n, level) -> mean, p10, p90 each (365, level)."""
    d = doy365(t)
    ang = 2 * np.pi * d / 365.0
    X = np.column_stack([np.ones(len(d))] + [f(k * ang) for k in range(1, NHARM + 1) for f in (np.cos, np.sin)])
    a = 2 * np.pi * np.arange(365) / 365.0
    Xd = np.column_stack([np.ones(365)] + [f(k * a) for k in range(1, NHARM + 1) for f in (np.cos, np.sin)])
    mean = np.empty((365, x.shape[1])); p10 = mean.copy(); p90 = mean.copy()
    for j in range(x.shape[1]):
        b = np.linalg.lstsq(X, x[:, j], rcond=None)[0]
        mean[:, j] = Xd @ b
        an = x[:, j] - X @ b
        q10 = np.empty(365); q90 = np.empty(365)
        for k in range(365):
            dd = np.abs(((d - k) + 182) % 365 - 182)
            s = an[dd <= window]
            q10[k], q90[k] = np.percentile(s, [10, 90])
        ker = np.ones(15) / 15.0
        q10 = np.convolve(np.r_[q10[-7:], q10, q10[:7]], ker, "valid")
        q90 = np.convolve(np.r_[q90[-7:], q90, q90[:7]], ker, "valid")
        p10[:, j] = mean[:, j] + q10; p90[:, j] = mean[:, j] + q90
    return mean, p10, p90


def check_step() -> int:
    t, u, T = nh_series(1991, 2024)
    df = pd.DataFrame(T, index=t, columns=LEVELS)
    an = (df - df.groupby(doy365(t)).transform("mean")).resample("YS").mean()
    print("NH polar-cap T, annual-mean anomaly (K) - MERRA-2 assimilates MLS from Aug 2004:")
    print(an[[10.0, 5.0, 3.0, 2.0, 1.0]].round(2).to_string())
    return 0


def build() -> int:
    out = {}
    counts = {}
    for hemi in ("nh", "sh"):
        y0, y1 = PERIOD[hemi]
        t, u, T = nh_series(y0, y1) if hemi == "nh" else gmi_series(y0, y1, north=False)
        t, u, T = screen(t, u, T)
        counts[hemi] = len(t)
        print(f"{hemi}: {len(t)} samples {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d}")
        for var, x in (("u60", u), ("capT", T)):
            out[(hemi, var)] = climatology(t, x, WINDOW[hemi])
    # how closely the replay (used for the SH) tracks MERRA-2 itself: the NH on the same days
    tg, ug, Tg = gmi_series(2005, 2019, north=True)
    tm, um, Tm = nh_series(2005, 2019)
    common = tg.intersection(tm)
    ig, im = tg.get_indexer(common), tm.get_indexer(common)
    for var, a, b in (("u60", ug, um), ("capT", Tg, Tm)):
        dlt = a[ig] - b[im]
        print(f"NH replay (00Z) minus MERRA-2 daily mean, {var}, {len(common)} days: " + ", ".join(
            f"{L:g} hPa {np.mean(dlt[:, LEVELS.index(L)]):+.2f} (sd {np.std(dlt[:, LEVELS.index(L)]):.2f})" for L in (10.0, 5.0, 1.0)))
    hemis, vars_ = ["nh", "sh"], ["u60", "capT"]
    data = {}
    for k, st in enumerate(("mean", "p10", "p90")):
        arr = np.full((2, 2, 365, len(LEVELS)), np.nan, "float32")
        for i, h in enumerate(hemis):
            for j, v in enumerate(vars_):
                arr[i, j] = out[(h, v)][k]
        data[st] = (("hemi", "var", "doy", "lev"), arr)
    ds = xr.Dataset(data, coords={"hemi": hemis, "var": vars_, "doy": np.arange(1, 366), "lev": LEVELS})
    ds.attrs.update(
        title="Upper-stratosphere day-of-year climatology: zonal-mean u at 60 deg and 60-90 deg polar-cap T, 100-1 hPa",
        nh=f"MERRA-2 daily-mean zonal means {PERIOD['nh'][0]}-{PERIOD['nh'][1]} ({counts['nh']} days), +/-{WINDOW['nh']} d anomaly window",
        sh=f"MERRA-2 GMI replay, 00Z every 10th day, {PERIOD['sh'][0]}-{PERIOD['sh'][1]} ({counts['sh']} samples), +/-{WINDOW['sh']} d anomaly window",
        method=f"mean = {NHARM}-harmonic fit; p10/p90 = mean + anomaly quantiles within the window, 15-day smoothed",
        base_period_reason="MLS era only: MERRA-2 cap T steps ~4 K at 1-3 hPa across Aug 2004 when Aura MLS enters",
        units="u60 m s-1; capT K", doy="1..365, 29 Feb uses 28 Feb")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT, encoding={k: {"zlib": True, "complevel": 5} for k in ("mean", "p10", "p90")})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
    for h in hemis:
        d0 = pd.Timestamp("2026-09-27").dayofyear - 1
        print(f"  {h} 27 Sep: " + "; ".join(f"{L:g} hPa u60 {ds['mean'].sel(hemi=h, var='u60', lev=L).values[d0]:+.1f} "
                                             f"[{ds['p10'].sel(hemi=h, var='u60', lev=L).values[d0]:+.1f}, {ds['p90'].sel(hemi=h, var='u60', lev=L).values[d0]:+.1f}] "
                                             f"capT {ds['mean'].sel(hemi=h, var='capT', lev=L).values[d0]:.1f}" for L in (10.0, 5.0, 1.0)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-step", action="store_true")
    a = ap.parse_args()
    return check_step() if a.check_step else build()


if __name__ == "__main__":
    sys.exit(main())
