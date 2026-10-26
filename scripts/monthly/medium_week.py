#!/usr/bin/env python3
"""The medium-range forecast of the target month's days 1-7 for an update of the month-ahead outlook (forecast.py
--update): IFS-ENS and AIFS-ENS, a half each, their ensemble-mean anomalies on the outlook's 3° cells against the ERA5
1991-2020 mean of the same days (the archive's block 0).

  z500, t2m  each day's members' mean on the 0.5° cells, from the run statistics the ensemble maps keep
             (scripts/ensmaps/ensmaps.py save_stats; the workflow restores them from the Actions cache), averaged over
             the days; the spread is the mean of the days' member standard deviations (wider than the spread of the
             7-day mean, so the probabilities stay on the safe side)
  prcp       total precipitation of the members between 00 UTC on the 1st and 00 UTC on the 8th, from the rows of
             each message north of the equator (ECMWF open data, Google Cloud mirror only)

    python scripts/monthly/medium_week.py --target 2026-11 --cycle 2026102512 --stats-dir in --out mw.npz
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

MODELS = {"ifs": ("ifs", ("ef",)), "aifs": ("aifs-ens", ("cf", "pf"))}  # open-data model name, files
ROWS = tuple(range(361))  # 90N to the equator at 0.25°


def days_of(ty: int, tm: int) -> list[dt.date]:
    return [dt.date(ty, tm, d) for d in range(1, G.SPLIT_DAY)]


def normal(A: M.Archive, v: str, tm: int) -> np.ndarray:
    """The ERA5 1991-2020 mean of the month's days 1-7 on the outlook's 0-90N cells (P,)."""
    y0 = int(A.years[0])
    x = M.month_series(A.blk[v][:, 0], A.years, tm, y0)[:, M.nh_rows(v)].reshape(len(A.years), -1)
    sel = (A.years >= M.CLIM[0]) & (A.years <= M.CLIM[1])
    return np.nanmean(x[sel], 0)


def from_stats(path: Path, init: str, days: list[dt.date], f: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(mean, spread) over the days on the outlook's cells, and the fewest members behind a day, from one run's
    statistics; t2m in deg C like the archive."""
    co = G.Coarsener(E.G50.lat, E.G50.lon)
    with np.load(path) as z:
        if str(z["init"]) != init:
            raise SystemExit(f"{path}: holds run {z['init']}, not {init}")
        row = {str(k): i for i, k in enumerate(z["keys"])}
        idx = [row[f"{d}/{d}"] for d in days]
        mean = z[f"mean_{f}"][idx].astype(np.float64).mean(0)
        sd = z[f"sd_{f}"][idx].astype(np.float64).mean(0)
        n = int(z[f"n_{f}"][idx].min())
    if f == "t2m":
        mean = mean - 273.15
    return co(mean, "nh").ravel(), co(sd, "nh").ravel(), n


def precip(model: str, init: dt.datetime, days: list[dt.date]) -> tuple[np.ndarray, np.ndarray, int]:
    """(mean, spread) of the members' daily precipitation (mm/day) over the days on the outlook's cells."""
    import rangefetch as rf

    name, kinds = MODELS[model]
    date, hh = f"{init:%Y%m%d}", f"{init:%H}"
    cos = {}

    def coarse(v, lat, lon):
        key = (float(lat[0]), float(lon[0]), v.shape)
        if key not in cos:
            cos[key] = G.Coarsener(lat, lon)
        return cos[key](v.astype(np.float64), "nh").ravel()

    acc = []
    for t in (days[0], days[-1] + dt.timedelta(days=1)):  # 00 UTC on the 1st and on the 8th
        step = int((dt.datetime.combine(t, dt.time()) - init).total_seconds() // 3600)
        per = {}
        for kind in kinds:
            idx = rf.fetch_index(date, hh, name, step, kind)
            want = rf.select(idx, param="tp")
            rel = rf.path_for(date, hh, name, step, kind) + ".grib2"
            for e, x in zip(want, rf.fetch_rows(rel, want, ROWS, each=coarse), strict=True):
                per[int(e.get("number", 0))] = x
        acc.append(per)
    members = sorted(set(acc[0]) & set(acc[1]))
    tot = np.stack([acc[1][m] - acc[0][m] for m in members])
    if np.nanpercentile(tot, 99) < 2.0:  # metres (IFS-ENS), not kg m-2
        tot = tot * 1000.0
    daily = tot / len(days)
    return daily.mean(0), daily.std(0, ddof=1), len(members)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="YYYY-MM")
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH, the runs the update's vortex comes from")
    ap.add_argument("--stats-dir", type=Path, required=True, help="<model>/<cycle>.npz run statistics (ensmaps)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ty, tm = map(int, a.target.split("-"))
    days = days_of(ty, tm)
    init = dt.datetime.strptime(a.cycle, "%Y%m%d%H")
    A = M.load_archive()
    out = {"cycle": np.array(a.cycle), "first": np.array(days[0].isoformat()), "last": np.array(days[-1].isoformat())}
    for v in M.TARGETS:
        parts = []
        for m in MODELS:
            if v == "prcp":
                mean, sd, n = precip(m, init, days)
            else:
                mean, sd, n = from_stats(a.stats_dir / m / f"{a.cycle}.npz", a.cycle, days, v)
            parts.append((mean, sd))
            out[f"{v}_{m}_n"] = np.array(n)
            print(f"  {v} {m}: {n} members, 20-90N mean {np.nanmean(mean):.2f}", flush=True)
        clim = normal(A, v, tm)
        out[f"{v}_anom"] = (np.mean([p[0] for p in parts], 0) - clim).astype(np.float32)  # a half each
        out[f"{v}_sd"] = np.sqrt(np.mean([p[1] ** 2 for p in parts], 0)).astype(np.float32)
    np.savez_compressed(a.out, **out)
    print(f"medium range {days[0]} .. {days[-1]} from the {a.cycle} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
