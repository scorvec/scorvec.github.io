#!/usr/bin/env python3
"""ERA5 1991-2020 normals of the daily-mean sea-level pressure over the tropics and subtropics (ROWS, 40.5S-40.5N), the
footing of the Hadley cells' normalised sea-level pressure maps (hadley_slp.py, hadley_slp_fc.py).

ERA5 assimilates the surface pressure that stations, ships and buoys report, and its sea-level pressure is the closest
of the reanalyses' to them. It is read from WeatherBench2's 6-hourly ERA5 on the 1.5 deg grid (the Hadley tracker's,
build_branch_clim.py). A day is the mean of its 00, 06, 12 and 18 UTC fields: four equally spaced samples cancel the
diurnal (S1) and semidiurnal (S2) tides exactly, and in the tropics those reach 1-1.5 hPa, as large as a day's anomaly.

Per grid point, by common-year day (doycal.noleap_doy: 29 Feb takes its neighbours' mean):
  normal   mean + annual + semiannual harmonics + a linear trend, fitted together on the daily means; the trend is
           read at EPOCH (Normals): the tropical daily sd is near 1 hPa, so a few tenths of a hPa of change since the
           middle of 1991-2020 would read as a standing anomaly. The trend of a point is shrunk by its signal-to-noise
           first, b t^2 / (1 + t^2) with t its t-value on the 30 annual means (applied_trend): in 1991-2020 it is
           |t| >= 2 over only ~15 % of the area, and a trend that is mostly interannual noise, carried 21 years from
           the middle of the period, would put that noise into every day's anomaly
  sd       of the daily anomalies about that normal: mean + annual + semiannual harmonics fitted to their squares (the
           variance by day of year), held above VAR_FLOOR of its annual mean, square-rooted
  zs       the ERA5 terrain height on the grid (era5_orography.nc, conservatively averaged): over high ground the
           sea-level pressure is a reduction, not a measurement, and the maps leave it out above TERRAIN_M
The build prints its checks: the sd by latitude band, the harmonic variance against each day's +-15-day pooled sample
variance, and the trend's shift to EPOCH in sd units, as fitted and as applied.

    python src/build_slp_clim.py fetch --years 1991 1993 --out part.npz
    python src/build_slp_clim.py build part*.npz        # -> data/reference/hslp_clim.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import build_hadley_clim as B
import hadley_branches as HB
from doycal import noleap_doy

OUT = B.REF / "hslp_clim.nc"
OROG = B.REF / "era5_orography.nc"  # metres, 0.25 deg
LAT_MAX = 40.5
ROWS = np.flatnonzero(np.abs(HB.LAT) <= LAT_MAX)  # the grid's rows 40.5N..40.5S
LAT, LON = HB.LAT[ROWS], HB.LON
EPOCH = 2026.75  # the level of the trend the normals are read at (decimal year)
T_REF = 2005.5  # the trend term's origin, the middle of 1991-2020
VAR_FLOOR = 0.2  # the variance by day of year is held at or above this share of its annual mean
TERRAIN_M = 1500.0  # above this grid-mean terrain height (m) the maps and the scores leave the point out
POOL = 15  # +- days of year in the build's check of the harmonic variance
HARMONICS = ["mean", "cos1", "sin1", "cos2", "sin2"]


def decimal_year(t: pd.DatetimeIndex) -> np.ndarray:
    return (t.year + (t.dayofyear - 0.5) / np.where(t.is_leap_year, 366, 365)).values


def harm(doy) -> np.ndarray:
    """(n, 5) mean, annual and semiannual terms at common-year days."""
    w = 2 * np.pi * np.atleast_1d(np.asarray(doy, float)) / 365.25
    return np.column_stack([np.ones(w.size), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def applied_trend(b: np.ndarray, se: np.ndarray) -> np.ndarray:
    """The trend shrunk towards zero by its signal-to-noise: b t^2 / (1 + t^2), t = b / se."""
    t2 = (b / se) ** 2
    return b * t2 / (1.0 + t2)


def terrain() -> np.ndarray:
    """The ERA5 terrain height (m) on the domain's 1.5 deg cells."""
    with xr.open_dataarray(OROG) as o:
        return HB.coarse15(o.values, o.latitude.values, o.longitude.values)[ROWS]


class Normals:
    """The normal (hPa) and sd (hPa) of the daily-mean sea-level pressure on (LAT, LON) at a day, the normal at the
    EPOCH level of its trend; `z` turns a daily mean into the normalised anomaly, NaN over high ground."""

    def __init__(self, path: Path = OUT):
        with xr.open_dataset(path) as ds:
            self.coef = ds["coef"].values.astype(np.float64)
            self.coef[0] += applied_trend(ds["trend"].values, ds["trend_se"].values) * (EPOCH - T_REF)
            self.var = ds["var"].values.astype(np.float64)
            self.high = ds["zs"].values > TERRAIN_M

    @staticmethod
    def _doy(day) -> float:
        return float(noleap_doy(pd.DatetimeIndex([pd.Timestamp(day)]))[0])

    def normal(self, day) -> np.ndarray:
        return np.tensordot(harm(self._doy(day))[0], self.coef, axes=(0, 0))

    def sd(self, day) -> np.ndarray:
        v = np.tensordot(harm(self._doy(day))[0], self.var, axes=(0, 0))
        return np.sqrt(np.maximum(v, VAR_FLOOR * self.var[0]))

    def z(self, msl: np.ndarray, day) -> np.ndarray:
        """(..., LAT, LON) daily-mean sea-level pressure (hPa) -> its anomaly in units of the day's sd."""
        return np.where(self.high, np.nan, (msl - self.normal(day)) / self.sd(day))


# ── fetch ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fetch(y0: int, y1: int) -> dict:
    """The daily means (hPa) of the years y0..y1 on the domain; a day without all four times is left out."""
    ds = xr.open_zarr(B.WB2, storage_options={"token": "anon"})
    da = ds["mean_sea_level_pressure"].transpose("time", "latitude", "longitude").sortby("latitude", ascending=False)
    if not (np.allclose(da.latitude.values, HB.LAT) and np.allclose(da.longitude.values, HB.LON)):
        raise SystemExit("WeatherBench2's grid is not the 1.5 deg grid of the normals")
    da = da.isel(latitude=ROWS)
    times = pd.DatetimeIndex(da.time.values)
    days, rows = [], []
    for m0 in pd.date_range(f"{y0}-01-01", f"{y1}-12-01", freq="MS"):
        sel = np.flatnonzero((times >= m0) & (times < m0 + pd.offsets.MonthBegin(1)))
        blk, bt = da.isel(time=sel).values / 100.0, times[sel]  # Pa -> hPa
        for day in pd.DatetimeIndex(sorted(set(bt.normalize()))):
            k = np.flatnonzero(bt.normalize() == day)
            if len(k) == 4:
                days.append(day)
                rows.append(blk[k].mean(0).astype(np.float32))
        print(f"  {m0:%Y-%m}: {len(sel)} times", flush=True)
    msl = np.stack(rows)
    if not 990.0 < float(msl.mean()) < 1030.0:
        raise SystemExit(f"domain mean {msl.mean():.1f}: not a sea-level pressure in hPa")
    return {"date": np.array(days, "datetime64[D]"), "msl": msl}


# ── build ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _pooled_var(r2: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """(365, k) mean of the squared anomalies within +-POOL common-year days of each day (29 Feb counts as 28 Feb)."""
    d = np.floor(doy).astype(int) - 1
    S = np.zeros((365, r2.shape[1]))
    np.add.at(S, d, r2)
    N = np.bincount(d, minlength=365).astype(float)
    k = np.arange(-POOL, POOL + 1)
    Ss = sum(np.roll(S, -s, axis=0) for s in k)
    Ns = sum(np.roll(N, -s) for s in k)
    return Ss / Ns[:, None]


def build(paths, chunk: int = 2000) -> xr.Dataset:
    parts = [np.load(p) for p in paths]
    date = pd.DatetimeIndex(np.concatenate([p["date"] for p in parts]))
    Y = np.concatenate([p["msl"] for p in parts]).reshape(date.size, -1)
    order = np.argsort(date.values)
    date, Y = date[order], Y[order]
    if date.has_duplicates:
        raise SystemExit("a day is in two parts")
    doy, yr = noleap_doy(date), decimal_year(date)
    H = harm(doy)
    X = np.column_stack([H, yr - T_REF])
    years = date.year.values
    uy = np.unique(years)
    Sxx = float(((uy - uy.mean()) ** 2).sum())
    Hd = harm(np.arange(1, 366))
    npt = Y.shape[1]
    coef, var = np.zeros((5, npt)), np.zeros((5, npt))
    trend, se = np.zeros(npt), np.zeros(npt)
    fit_err = np.zeros(npt)
    for a in range(0, npt, chunk):
        y = Y[:, a : a + chunk].astype(np.float64)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
        coef[:, a : a + chunk], trend[a : a + chunk] = b[:5], b[5]
        v = np.linalg.lstsq(H, r**2, rcond=None)[0]
        var[:, a : a + chunk] = v
        e = np.stack([r[years == u].mean(0) for u in uy])  # the annual means about the fit
        se[a : a + chunk] = np.sqrt((e**2).sum(0) / (uy.size - 2) / Sxx)
        emp, harmv = _pooled_var(r**2, doy), Hd @ v
        fit_err[a : a + chunk] = np.sqrt(((harmv - emp) ** 2).mean(0)) / emp.mean(0)
    shape = (LAT.size, LON.size)
    zs = terrain()
    ds = xr.Dataset(
        {
            "coef": (("harmonic", "latitude", "longitude"), coef.reshape(5, *shape).astype(np.float32)),
            "trend": (("latitude", "longitude"), trend.reshape(shape).astype(np.float32)),
            "trend_se": (("latitude", "longitude"), se.reshape(shape).astype(np.float32)),
            "var": (("harmonic", "latitude", "longitude"), var.reshape(5, *shape).astype(np.float32)),
            "zs": (("latitude", "longitude"), zs.astype(np.float32)),
        },
        coords={"harmonic": HARMONICS, "latitude": LAT, "longitude": LON},
        attrs={
            "source": "ERA5 1991-2020, WeatherBench2 6-hourly 1.5 deg; daily means of 00, 06, 12 and 18 UTC",
            "coef": "hPa: the normal by common-year day at T_REF (its trend term apart)",
            "trend": "hPa per year, fitted with the harmonics; trend_se from the 30 annual means of the residuals",
            "var": "hPa^2: harmonics fitted to the squared daily anomalies about the normal",
            "zs": "ERA5 terrain height (m), conservatively averaged onto the grid",
            "t_ref": T_REF,
            "days": int(date.size),
        },
    )
    check(ds, fit_err.reshape(shape))
    return ds


def check(ds: xr.Dataset, fit_err: np.ndarray):
    """The build's sanity checks (printed): sd by band, the harmonic variance's fit, the trend in sd units."""
    sd_mean = np.sqrt(ds["var"].values[0])
    vd = harm(np.arange(1, 366)) @ ds["var"].values.reshape(5, -1)
    floor = (vd < VAR_FLOOR * ds["var"].values[0].reshape(1, -1)).mean()
    low = ds["zs"].values <= TERRAIN_M
    lat = np.abs(LAT)[:, None] * np.ones(LON.size)
    print(f"sd (hPa), annual mean of the variance, points below {TERRAIN_M:.0f} m:")
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 41)):
        m = low & (lat >= lo) & (lat < hi)
        q = np.percentile(sd_mean[m], [5, 50, 95])
        print(f"  {lo:2d}-{hi:2d} deg: p5 {q[0]:.2f}  median {q[1]:.2f}  p95 {q[2]:.2f}")
    vmin, vmax = (np.sqrt(np.clip(f(vd, 0), 0, None)).reshape(sd_mean.shape) for f in (np.min, np.max))
    m = low & (lat < 10)
    print(f"  deep tropics, smallest to largest day: median {np.median(vmin[m]):.2f} .. {np.median(vmax[m]):.2f}")
    print(f"  the variance floor binds on {100 * floor:.2f} % of point-days")
    q = np.percentile(fit_err[low], [50, 90, 99])
    print(
        f"harmonic variance against the +-{POOL}-day pooled variance, rms / mean: p50 {q[0]:.3f} p90 {q[1]:.3f} p99 {q[2]:.3f}"
    )
    b, se = ds["trend"].values, ds["trend_se"].values
    w = np.cos(np.deg2rad(lat))
    print(f"the trend's shift {T_REF} -> {EPOCH} in sd (of the annual-mean variance), below {TERRAIN_M:.0f} m:")
    print(
        f"  area with |t| >= 2 on the annual means: {100 * np.average(np.abs(b / se)[low] >= 2, weights=w[low]):.1f} %"
    )
    shifts = {k: x * (EPOCH - T_REF) / sd_mean for k, x in (("fitted", b), ("applied", applied_trend(b, se)))}
    for name, shift in shifts.items():
        a = np.abs(shift[low])
        print(
            f"  {name}: |shift| area mean {np.average(a, weights=w[low]):.3f}, p95 {np.percentile(a, 95):.3f}, "
            f"max {a.max():.3f}; area >= 0.1 sd {100 * np.average(a >= 0.1, weights=w[low]):.1f} %, "
            f">= 0.25 sd {100 * np.average(a >= 0.25, weights=w[low]):.1f} %"
        )
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 41)):
        m = low & (lat >= lo) & (lat < hi)
        print(f"  {lo:2d}-{hi:2d} deg: mean applied shift {np.average(shifts['applied'][m], weights=w[m]):+.3f} sd")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs=2, required=True)
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fetch":
        np.savez_compressed(a.out, **fetch(*a.years))
        return 0
    ds = build(a.parts)
    ds.to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
