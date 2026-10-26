#!/usr/bin/env python3
"""The data of an update of the month-ahead outlook (forecast.py --update, update.py), from the newest 00/12Z run that
all four ensembles have on ens-anomalies.html:

  week 1   the target month's days 1-7: the IFS-ENS and AIFS-ENS ensemble-mean anomalies (a half each) on the
           outlook's 3° cells against the ERA5 1991-2020 mean of the same days (the archive's block 0), with their
           spread (the mean of the days' member standard deviations: wider than the spread of the 7-day mean)
  window   the predictors' window, the 31 days ending on the 7th of the target month (the archive's block 1 of the
           month before and block 0 of the target month), for this year: ERA5 as far as the archive holds it, then
           the IFS control's analyses (00 and 12 UTC; precipitation from its 00 UTC runs' first day), then the two
           ensembles' means; sea ice, SST, snow, soil moisture and the QBO wind keep ERA5's anomaly of the days it holds

  z500, t2m   each day's members' mean on the 0.5° cells, from the run statistics the ensemble maps keep
              (scripts/ensmaps/ensmaps.py save_stats; the workflow restores them from the Actions cache)
  prcp        the members' total precipitation between two steps, from the rows of each message north of 30S
Everything ECMWF from the Google Cloud mirror only.

    python scripts/monthly/update_data.py --target 2026-11 --cycle 2026102512 --stats-dir in --out upd.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ensmaps")]
import ensmaps as E  # (imports pyproj before eccodes; puts scripts/ecmwf and lib on the path)
import mo_grid as G
import mo_method as M

ENSEMBLES = {"ifs": ("ifs", ("ef",)), "aifs": ("aifs-ens", ("cf", "pf"))}  # open-data model name, files
ROWS = tuple(range(481))  # 90N to 30S at 0.25°
PERSIST = ("sic", "sst", "snow", "swvl", "u50eq", "u50v")  # the window takes ERA5's anomaly of the days it holds
GRID = {"z500": "nh", "t2m": "nh", "prcp": "all"}  # the archive's grids (mo_grid.FIELDS)
_CO: dict = {}


def coarse(v: np.ndarray, lat: np.ndarray, lon: np.ndarray, grid: str) -> np.ndarray:
    key = (float(lat[0]), float(lat[-1]), float(lon[0]), v.shape)
    if key not in _CO:
        _CO[key] = G.Coarsener(lat, lon)
    return _CO[key](v.astype(np.float64), grid)


def step_of(init: dt.datetime, t: dt.date) -> int:
    return int((dt.datetime.combine(t, dt.time()) - init).total_seconds() // 3600)


def stats_days(path: Path, init: str, days: list[dt.date], f: str) -> tuple[np.ndarray, np.ndarray]:
    """(daily means (days, lat, lon), daily member sds) on the archive's grid from one run's statistics; t2m in deg C
    like the archive."""
    co = G.Coarsener(E.G50.lat, E.G50.lon)
    with np.load(path) as z:
        if str(z["init"]) != init:
            raise SystemExit(f"{path}: holds run {z['init']}, not {init}")
        row = {str(k): i for i, k in enumerate(z["keys"])}
        idx = [row[f"{d}/{d}"] for d in days]
        mean = z[f"mean_{f}"][idx].astype(np.float64)
        sd = z[f"sd_{f}"][idx].astype(np.float64)
    if f == "t2m":
        mean = mean - 273.15
    return co(mean, GRID[f]), co(sd, GRID[f])


def precip(model: str, init: dt.datetime, t0: dt.date, t1: dt.date) -> np.ndarray:
    """(members, lat, lon) total precipitation (mm) between 00 UTC on t0 and 00 UTC on t1 on the archive's grid."""
    import rangefetch as rf

    name, kinds = ENSEMBLES[model]
    date, hh = f"{init:%Y%m%d}", f"{init:%H}"
    acc = []
    for t in (t0, t1):
        step = step_of(init, t)
        per = {}
        if step > 0:
            for kind in kinds:
                idx = rf.fetch_index(date, hh, name, step, kind)
                want = rf.select(idx, param="tp")
                rel = rf.path_for(date, hh, name, step, kind) + ".grib2"
                each = lambda v, lat, lon: coarse(v, lat, lon, "all")  # noqa: E731
                per |= {
                    int(e.get("number", 0)): x
                    for e, x in zip(want, rf.fetch_rows(rel, want, ROWS, each=each), strict=True)
                }
        acc.append(per)
    members = sorted(acc[1]) if not acc[0] else sorted(set(acc[0]) & set(acc[1]))
    tot = np.stack([acc[1][m] - (acc[0][m] if acc[0] else 0.0) for m in members])
    return tot * 1000.0 if np.nanpercentile(tot, 99) < 2.0 else tot  # metres (IFS-ENS), not kg m-2


def analysis(day: dt.date) -> dict:
    """The IFS control's day: z500 and t2m (the mean of its 00 and 12 UTC analyses), prcp (its 00 UTC run's first 24 h),
    on the archive's grids."""
    import rangefetch as rf

    out = {"z500": [], "t2m": []}
    for h in ("00", "12"):
        idx = rf.fetch_index(f"{day:%Y%m%d}", h, "ifs", 0, "fc", stream="oper")
        rel = rf.path_for(f"{day:%Y%m%d}", h, "ifs", 0, "fc", stream="oper") + ".grib2"
        want = rf.select(idx, param="gh", levelist=[500]) + rf.select(idx, param="2t")
        vals = rf.fetch_rows(rel, want, ROWS, each=lambda v, lat, lon: coarse(v, lat, lon, "nh"))
        for e, x in zip(want, vals, strict=True):
            out["z500" if e["param"] == "gh" else "t2m"].append(x if e["param"] == "gh" else x - 273.15)
    idx = rf.fetch_index(f"{day:%Y%m%d}", "00", "ifs", 24, "fc", stream="oper")
    want = rf.select(idx, param="tp")
    rel = rf.path_for(f"{day:%Y%m%d}", "00", "ifs", 24, "fc", stream="oper") + ".grib2"
    tp = rf.fetch_rows(rel, want, ROWS, each=lambda v, lat, lon: coarse(v, lat, lon, "all"))[0]
    return {
        "z500": np.mean(out["z500"], 0),
        "t2m": np.mean(out["t2m"], 0),
        "prcp": tp * 1000.0 if np.nanpercentile(tp, 99) < 2.0 else tp,
    }


def block_normal(A: M.Archive, v: str, tm: int) -> np.ndarray:
    """The ERA5 1991-2020 mean of the target month's days 1-7 on the outlook's 0-90N cells (P,)."""
    x = M.month_series(A.blk[v][:, 0], A.years, tm, int(A.years[0]))[:, M.nh_rows(v)].reshape(len(A.years), -1)
    return np.nanmean(x[(A.years >= M.CLIM[0]) & (A.years <= M.CLIM[1])], 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="YYYY-MM")
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH, the runs the update comes from")
    ap.add_argument("--stats-dir", type=Path, required=True, help="<model>/<cycle>.npz run statistics (ensmaps)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ty, tm = map(int, a.target.split("-"))
    iy, im = (ty, tm - 1) if tm > 1 else (ty - 1, 12)
    init = dt.datetime.strptime(a.cycle, "%Y%m%d%H")
    first, last = dt.date(iy, im, G.SPLIT_DAY), dt.date(ty, tm, G.SPLIT_DAY - 1)  # the window: the 8th .. the 7th
    week = [dt.date(ty, tm, d) for d in range(1, G.SPLIT_DAY)]
    A = M.load_archive()
    arc = G.load_year(iy)
    n_obs = int(arc["n"][im - 1, 1]) // 4  # four 6-hourly samples a day
    fc0 = init.date() if init.hour == 0 else init.date() + dt.timedelta(days=1)  # the first day the runs forecast
    gap = [first + dt.timedelta(days=k) for k in range(n_obs, (fc0 - first).days)]
    fc_days = [fc0 + dt.timedelta(days=k) for k in range((last - fc0).days + 1)]
    ndays = (last - first).days + 1
    print(f"window {first} .. {last}: ERA5 {n_obs} days, IFS analyses {len(gap)} ({gap[0] if gap else '-'} ..), "
          f"the {a.cycle} runs {len(fc_days)} ({fc_days[0]} .. {fc_days[-1]})", flush=True)  # fmt: skip
    out = {"cycle": np.array(a.cycle), "window": np.array([first.isoformat(), last.isoformat()]),
           "parts": np.array([n_obs, len(gap), len(fc_days)]), "week": np.array([week[0].isoformat(), week[-1].isoformat()])}  # fmt: skip
    an = [analysis(d) for d in gap]
    for v in ("z500", "t2m", "prcp"):
        if v == "prcp":
            runs = [precip(m, init, fc0, last + dt.timedelta(days=1)).mean(0) for m in ENSEMBLES]
            wk = [precip(m, init, week[0], week[-1] + dt.timedelta(days=1)) / len(week) for m in ENSEMBLES]
            fc_sum = np.mean(runs, 0)
            wk_mean, wk_sd = (
                np.mean([w.mean(0) for w in wk], 0),
                np.sqrt(np.mean([w.std(0, ddof=1) ** 2 for w in wk], 0)),
            )
        else:
            per = [stats_days(a.stats_dir / m / f"{a.cycle}.npz", a.cycle, fc_days, v) for m in ENSEMBLES]
            daily = np.mean([p[0] for p in per], 0)  # (days, lat, lon), a half each
            sds = np.mean([p[1] for p in per], 0)
            k = [fc_days.index(d) for d in week]
            fc_sum = daily.sum(0)
            wk_mean, wk_sd = daily[k].mean(0), sds[k].mean(0)
        obs = n_obs * arc[v][im - 1, 1].astype(np.float64) if n_obs else 0.0
        out[f"win_{v}"] = ((obs + sum(x[v] for x in an) + fc_sum) / ndays).astype(np.float32)
        r = M.nh_rows(v)
        out[f"{v}_anom"] = (wk_mean[r].ravel() - block_normal(A, v, tm)).astype(np.float32)
        out[f"{v}_sd"] = wk_sd[r].ravel().astype(np.float32)
        print(f"  {v}: week-1 20-90N anomaly {np.nanmean(out[f'{v}_anom']):+.2f}", flush=True)
    for v in PERSIST:
        out[f"win_{v}"] = np.asarray(arc[v][im - 1, 1], np.float32)
    out["members"] = np.array([int(np.load(a.stats_dir / m / f"{a.cycle}.npz")["n_z500"].min()) for m in ENSEMBLES])
    np.savez_compressed(a.out, **out)
    print(f"update data for {a.target} from the {a.cycle} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
