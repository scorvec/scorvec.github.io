#!/usr/bin/env python3
"""ERA5 1991-2020 for the Hadley outflow and subtropical descent tracker (hadley_branches.py, hadley_traj.py).

From WeatherBench2's 6-hourly ERA5 (1.5 deg, the tracker's grid): u, v, omega and temperature on the 13 levels, the
500 hPa height and the sea-level pressure. Each day's four times are averaged and become the fields
(hadley_branches.fields); each day's 00 UTC also releases the trajectories (hadley_traj.trace, through the 6-hourly
winds of the DAYS_BACK days before it). Two references come out:
  maps     per grid point, mean + annual + semiannual harmonics by common-year day (doycal.noleap_doy), fitted on the
           daily fields; z500 also carries a linear trend, evaluated at EPOCH: the 500 hPa surface rises with the
           warming troposphere, and against a 1991-2020 level every recent day would read as a stronger ridge
  indices  the tracker's numbers of every day's CENTRED 7-day window (the statistic of the analysed weeks): the field
           indices of the 7-day-mean fields, and the share and median transit time of the 7 days' releases; their
           mean, sd and percentiles by day of year pooled over the 30 years and +-POOL days. The ridge numbers are
           read at EPOCH's level: each sector's trend in its band-mean z500 raises a day's peak height and lowers the
           threshold its 588-dam area is read at (interpolated on the THR ladder)

    python src/build_branch_clim.py fetch --years 1991 1993 --out part.npz
    python src/build_branch_clim.py build part*.npz          # -> data/reference/branch_clim.nc
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
import hadley_traj as T
from doycal import noleap_doy

OUT = B.REF / "branch_clim.nc"
G = 9.80665
HALF, POOL = 3, 3
PCTS = (10, 25, 50, 75, 90)
EPOCH = 2026.75  # the ridge numbers' reference level (decimal year); the trend in z500 is ~0.5-0.7 m a year
T_REF = 2005.5  # the trend term's origin
THR = np.arange(5800.0, 5961.0, 10.0)  # z500 thresholds of the 588-dam area ladder (m)
VARS = {"u": "u_component_of_wind", "v": "v_component_of_wind", "w": "vertical_velocity", "t": "temperature"}


def _year(t: pd.DatetimeIndex) -> np.ndarray:
    return (t.year + (t.dayofyear - 0.5) / np.where(t.is_leap_year, 366, 365)).values


def _design(t: pd.DatetimeIndex) -> np.ndarray:
    w = 2 * np.pi * noleap_doy(t) / 365.25
    return np.column_stack([np.ones(len(t)), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w), _year(t) - T_REF])


def traj_keys() -> list[str]:
    return [f"{q}_{s}_{int(p)}" for s in HB.SUBS for p in T.RELEASE_P for q in ("share", "transit")]


def traj_row(h: np.ndarray) -> dict:
    """The trajectory numbers of a window's summed (sector, level, bin) histograms."""
    share, med = T.stats(h)
    out = {}
    for i, s in enumerate(HB.SUBS):
        for j, p in enumerate(T.RELEASE_P):
            out[f"share_{s}_{int(p)}"], out[f"transit_{s}_{int(p)}"] = 100.0 * share[i, j], med[i, j]
    return out


def winds(block: dict) -> tuple:
    """The trajectory arrays of a 6-hourly block: u, v, w on P_ASC (time, level, lat, lon) and ptrop (time, lat, lon)."""
    asc = [list(HB.LEVELS).index(int(p)) for p in T.P_ASC]
    ptrop = np.stack([HB.tropopause_surface(*(block[k][i] for k in "uvt"))["p"] for i in range(len(block["t"]))])
    ptrop = ptrop.astype(np.float32)
    return (*(block[k][:, asc].astype(np.float32) for k in ("u", "v", "w")), ptrop)


def fetch(y0: int, y1: int) -> dict:
    """Years y0..y1 with the margins the centred windows and the back-trajectories reach into."""
    first = pd.Timestamp(f"{y0}-01-01") - pd.Timedelta(days=HALF)
    return reduce_days(
        first - pd.Timedelta(days=T.DAYS_BACK), first, pd.Timestamp(f"{y1}-12-31") + pd.Timedelta(days=HALF), y0, y1
    )


def reduce_days(t_load: pd.Timestamp, t_rel: pd.Timestamp, t1: pd.Timestamp, y0: int, y1: int) -> dict:
    """fetch() for data from t_load, releases and days from t_rel to t1, of which the years y0..y1 are the core."""
    ds = xr.open_zarr(B.WB2, storage_options={"token": "anon"})
    lev = list(HB.LEVELS)

    def prep(da):
        dims = [d for d in ("time", "level", "latitude", "longitude") if d in da.dims]
        return da.transpose(*dims).sortby("latitude", ascending=False)

    var = {k: prep(ds[n].sel(level=lev)) for k, n in VARS.items()}
    var["z"] = prep(ds["geopotential"].sel(level=500))
    var["msl"] = prep(ds["mean_sea_level_pressure"])
    times = pd.DatetimeIndex(var["u"].time.values)
    days, D = [], {f: [] for f in HB.FIELDS}
    roll: list[tuple] = []  # (time, u, v, w, ptrop) from DAYS_BACK + 1 days before the block in hand
    hist = {}
    for m0 in pd.date_range(t_load.normalize().replace(day=1), t1, freq="MS"):
        sel = np.flatnonzero(
            (times >= max(m0, t_load)) & (times < min(m0 + pd.offsets.MonthBegin(1), t1 + pd.Timedelta(days=1)))
        )
        if not sel.size:
            continue
        blk = {k: v.isel(time=sel).values for k, v in var.items()}
        bt = times[sel]
        u, v, w, pt = winds(blk)
        keep_from = bt[0] - pd.Timedelta(days=T.DAYS_BACK + 1)  # the block's first release reaches back this far
        roll = [r for r in roll if r[0] >= keep_from] + [(bt[i], u[i], v[i], w[i], pt[i]) for i in range(len(bt))]
        for day in pd.DatetimeIndex(sorted(set(bt.normalize()))):
            k = np.flatnonzero(bt.normalize() == day)
            if day < t_rel or len(k) != 4:
                continue
            mean = {n: blk[n][k].mean(0) for n in blk}
            F = HB.fields(mean["u"], mean["v"], mean["w"], mean["t"], mean["z"] / G, mean["msl"])
            for f in HB.FIELDS:
                D[f].append(F[f].astype(np.float32))
            days.append(day)
        rel = [d for d in bt[bt.hour == 0] if t_rel <= d <= t1]
        if rel and roll[0][0] <= rel[0] - pd.Timedelta(days=T.DAYS_BACK):
            W = T.Winds(
                roll[0][0].timestamp(),
                *(np.stack([r[i] for r in roll]) for i in range(1, 5)),
            )
            hist.update(zip(rel, T.trace_many(W, [d.timestamp() for d in rel]), strict=True))
        print(f"  {m0:%Y-%m}: {len(bt)} times, {len(rel)} releases", flush=True)
    days = pd.DatetimeIndex(days)
    D = {f: np.stack(v) for f, v in D.items()}
    core = np.flatnonzero((days.year >= y0) & (days.year <= y1))
    X = _design(days[core])
    out = {"XtX": X.T @ X, "n": len(core)}
    for f in HB.FIELDS:
        out[f"XtY_{f}"] = np.tensordot(X.T, D[f][core].astype(np.float64), axes=(1, 0))  # (6, lat, lon)
    rows, lad, zbox, idx_days = [], [], [], []
    for i in core:
        if i - HALF < 0 or i + HALF >= len(days) or (days[i + HALF] - days[i - HALF]).days != 2 * HALF:
            continue
        win = [days[i] + pd.Timedelta(days=d) for d in range(-HALF, HALF + 1)]
        if any(d not in hist for d in win):
            continue
        W = {f: D[f][i - HALF : i + HALF + 1].mean(0) for f in HB.FIELDS}
        rows.append(HB.indices(W) | traj_row(sum(hist[d] for d in win)))
        L = HB.ridge_ladder(W["z500"], THR)
        lad.append([L[s][0] for s in HB.RIDGE])
        zbox.append([L[s][1] for s in HB.RIDGE])
        idx_days.append(days[i])
    keys = list(rows[0])
    out.update(
        date=np.array(idx_days, "datetime64[D]"),
        keys=np.array(keys),
        idx=np.array([[r[k] for k in keys] for r in rows]),
        ladder=np.array(lad),  # (day, sector, threshold)
        zbox=np.array(zbox),  # (day, sector)
    )
    return out


def build(paths) -> xr.Dataset:
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    XtX = sum(p["XtX"] for p in parts)
    maps = {}
    for f in HB.FIELDS:
        XtY = sum(p[f"XtY_{f}"] for p in parts).reshape(6, -1)
        if f == "z500":  # with the trend term, folded into the mean at EPOCH's level
            c = np.linalg.solve(XtX, XtY)
            c = c[:5] + np.r_[EPOCH - T_REF, 0, 0, 0, 0][:, None] * c[5]
        else:
            c = np.linalg.solve(XtX[:5, :5], XtY[:5])
        maps[f] = c.reshape(5, HB.LAT.size, HB.LON.size)
    date = pd.DatetimeIndex(np.concatenate([p["date"] for p in parts]))
    keys = [str(k) for k in parts[0]["keys"]]
    order = np.argsort(date.values)
    df = pd.DataFrame(np.concatenate([p["idx"] for p in parts])[order], index=date[order], columns=keys)
    ladder = np.concatenate([p["ladder"] for p in parts])[order]
    zbox = np.concatenate([p["zbox"] for p in parts])[order]
    yr = _year(df.index)
    trends = {}
    Xh = _design(df.index)
    for j, s in enumerate(HB.RIDGE):  # the sector's trend in band-mean z500 (m/yr), seasonal cycle removed
        b = np.linalg.lstsq(Xh, zbox[:, j], rcond=None)[0][5]
        trends[s] = float(b)
        lift = b * (EPOCH - yr)
        df[f"zmax_{s}"] = df[f"zmax_{s}"] + lift
        df[f"r588_{s}"] = [np.interp(HB.Z588 - d, THR, ladder[i, j]) for i, d in enumerate(lift)]
    doy = noleap_doy(df.index)
    stats = np.full((len(keys), len(PCTS) + 2, 366), np.nan)
    for d in range(1, 366):
        band = np.abs(((doy - d + 182.5) % 365) - 182.5) <= POOL
        if band.sum() < 15 * (2 * POOL + 1):  # fewer than ~15 years' worth: too thin to trust
            continue
        v = df.values[band]
        stats[:, 0, d - 1], stats[:, 1, d - 1] = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)
        stats[:, 2:, d - 1] = np.nanpercentile(v, PCTS, axis=0).T
    stats[:, :, 365] = stats[:, :, 364]
    return xr.Dataset(
        {
            "coef": (("field", "harmonic", "latitude", "longitude"), np.stack([maps[f] for f in HB.FIELDS])),
            "index": (("key", "stat", "doy"), stats),
        },
        coords={
            "field": list(HB.FIELDS),
            "harmonic": ["mean", "cos1", "sin1", "cos2", "sin2"],
            "latitude": HB.LAT,
            "longitude": HB.LON,
            "key": keys,
            "stat": ["mean", "sd"] + [f"p{q}" for q in PCTS],
            "doy": np.arange(1, 367),
        },
        attrs={
            "source": "ERA5 1991-2020, 6-hourly, WeatherBench2 1.5 deg",
            "maps": "harmonic normal by common-year day of the daily-mean fields; z500 at the EPOCH level (linear trend)",
            "indices": f"of each day's centred 7-day window, pooled +-{POOL} days of year; ridge numbers at the EPOCH level",
            "epoch": EPOCH,
            "z500_trend_m_per_yr": str({k: round(v, 3) for k, v in trends.items()}),
        },
    )


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
    print(f"wrote {OUT}; z500 trends {ds.attrs['z500_trend_m_per_yr']}")
    keys = ("out_nh", "desc_nh", "r588_npac", "out_sh", "desc_sh", "share_nh_500", "transit_nh_500", "transit_sh_500")
    for k in keys:
        r = ds["index"].sel(key=k)
        print(f"  {k}: 1 Feb p50 {float(r.sel(stat='p50', doy=32)):.2f}  1 Aug {float(r.sel(stat='p50', doy=213)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
