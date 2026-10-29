#!/usr/bin/env python3
"""GEPS guidance: the month's 35-day GEPS run, the second model beside GEFS in the month's blend (blend.py).

GEPS runs to day 39 on Mondays and Thursdays, so its newest run on or before the 15th reaches the 15th-19th of the
target month. As for GEFS (gefs_guide.py) the run gives two predictors: its mean anomaly over the target month's days
it reaches (target) and over its days before the month (lead-in, temperature only). Over 2001-2019, beside the
statistical forecast, GEPS alone scored better than GEFS alone for all three variables (twelve-month mean ACC: 500 hPa
0.17 against 0.14, 2 m temperature 0.30 against 0.28, precipitation 0.13 against 0.11), and the two averaged did
best (blend ACC 0.18 / 0.34 / 0.18 against GEFS's 0.15 / 0.30 / 0.15).

Anomalies: the ensemble mean minus the GEPS8 hindcast climatology at the same lead, interpolated to the start's
common-year day (scripts/geps/forecast.anomaly, which writes the live run's 1° anomalies to the frames store,
scripts/geps/archive.py); 500 hPa height less its global mean per lead (the GEFS page's rule: the offset between the
2001-2020 hindcast climate and today). The tables hold anomalies about their own 2001-2020 mean (and that mean, which
the live run loses too); blend.py puts both on 1991-2020 with the observed mean anomaly of the years.

Hindcast: the GEPS8 reforecast, 2001-2020 (ECCC's on-the-fly reforecast of GEPS 8.0.0, implemented 11 June 2024 and
still the science of 8.1.0, the April 2026 port to the new supercomputer: 4 members, Mondays and Thursdays of the
real-time calendar, kept in $DATA_ARCHIVE/geps_subx), the start nearest the 15th of the issue month on or before it
(within SLACK days), box-averaged onto the outlook's 3° cells, as data/geps_hc_<target MM>.npz.

    python scripts/monthly/geps_guide.py --build [MM,...]    # the reforecast tables (laptop: the reforecast is local)
    python scripts/monthly/geps_guide.py --live YYYY-MM      # print the live guidance for a target month
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import mo_grid as G

DATA = HERE / "data"
RUN_URL = "https://scorvec-store.b-cdn.net/scripts/geps/state/runs/geps_anom1_{tag}_{cycle}.nc"  # the frames store
TAGS = {"z500": ("zg500", "zg", 1.0), "t2m": ("t2m", "tas", 1.0), "prcp": ("pr", "pr", 86400.0)}  # tag, var, to units
DEMEAN = ("z500",)
LEAD = ("t2m",)
DAYS = 35
ISSUE_DAY = 15
SLACK = 4  # days: GEPS runs Mondays and Thursdays
MIN_DAYS = 14
NAME = "GEPS"


def windows(init: dt.date, first: dt.date) -> tuple[np.ndarray, np.ndarray] | None:
    """Masks over a run's days 1-35 (day d = init + d - 1): the target month's days and the days before it; None when
    the run reaches fewer than MIN_DAYS of the month."""
    valid = np.array([init + dt.timedelta(days=k) for k in range(DAYS)])
    tw = np.array([v >= first and v.month == first.month for v in valid])
    lw = valid < first
    return (tw, lw) if tw.sum() >= MIN_DAYS else None


def reduce(anom: np.ndarray, lat: np.ndarray, lon: np.ndarray, v: str, tw, lw) -> tuple[np.ndarray, np.ndarray]:
    """(target, lead-in) (P,) on the outlook's 0-90N cells from (35, lat, lon) daily anomalies in the variable's units;
    500 hPa height less its global mean per day."""
    if v in DEMEAN:
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))
        anom = anom - np.einsum("dij,ij->d", anom, w)[:, None, None] / w.sum()
    co = G.Coarsener(lat, lon)
    return co(anom[tw].mean(0), "nh").ravel(), co(anom[lw].mean(0), "nh").ravel()


def issue_start(starts, year: int, month: int):
    """The newest start on or before the 15th of month (year, month), within SLACK days, or None."""
    issue = dt.date(year, month, ISSUE_DAY)
    ok = [s for s in starts if issue - dt.timedelta(days=SLACK) <= s <= issue]
    return max(ok) if ok else None


# ------------------------------------------------------------------------------------------------- the hindcast
def build(month: int, store: Path) -> Path:
    """data/geps_hc_<MM>.npz for target month `month`: years (target years), init, ndays, and per variable v the
    (n, P) target and v_lead lead-in anomalies about their 2001-2020 mean (float16), with those means (mean_v,
    mean_v_lead)."""
    import pandas as pd
    import xarray as xr

    im = 12 if month == 1 else month - 1
    clim = {
        v: xr.open_dataset(store / "climatology" / f"geps8_clim_{tag}.nc")[var] for v, (tag, var, _) in TAGS.items()
    }
    refs: dict = {}  # (variable, start day) -> the climatology there: the reforecast repeats calendar dates every year
    tab = {v: ([], []) for v in TAGS}
    years, inits, nd = [], [], []
    for ty in range(2001, 2021):
        iy = ty - 1 if month == 1 else ty
        files = {v: store / "geps8_hindcast" / f"geps8_{tag}_{iy}{im:02d}.nc" for v, (tag, _, _) in TAGS.items()}
        if not all(f.exists() for f in files.values()):
            continue
        with xr.open_dataset(files["t2m"]) as ds:
            starts = [pd.Timestamp(s).date() for s in ds["S"].values]
        s0 = issue_start(starts, iy, im)
        win = windows(s0, dt.date(ty, month, 1)) if s0 else None
        if win is None:
            continue
        doy = float(s0.timetuple().tm_yday - int(s0.month > 2 and pd.Timestamp(s0).is_leap_year))  # common-year day
        for v, (_, var, scale) in TAGS.items():
            with xr.open_dataset(files[v]) as ds:
                x = ds[var].isel(S=starts.index(s0)).squeeze().values[:DAYS].astype(np.float64)
                lat, lon = ds["Y"].values.astype(float), ds["X"].values.astype(float)
            if (v, doy) not in refs:  # the live run's rule (forecast.anomaly): interpolated to the start's day
                c = clim[v]
                cw = xr.concat([c.isel(doy=-1).assign_coords(doy=c.doy[-1] - 366), c,
                                c.isel(doy=0).assign_coords(doy=c.doy[0] + 366)], dim="doy")  # fmt: skip
                refs[(v, doy)] = cw.isel(L=slice(0, DAYS)).interp(doy=doy).squeeze().values
            t, lead = reduce((x - refs[(v, doy)]) * scale, lat, lon, v, *win)
            tab[v][0].append(t)
            tab[v][1].append(lead)
        years.append(ty)
        inits.append(f"{s0:%Y-%m-%d}")
        nd.append(int(win[0].sum()))
    out = {"years": np.array(years), "init": np.array(inits), "ndays": np.array(nd)}
    for v, (t, lead) in tab.items():
        for key, a in ((v, np.array(t)), (f"{v}_lead", np.array(lead))):
            out[f"mean_{key}"] = a.mean(0).astype(np.float32)
            out[key] = (a - a.mean(0)).astype(np.float16)
    p = DATA / f"geps_hc_{month:02d}.npz"
    np.savez_compressed(p, **out)
    print(f"{p.name}: {len(years)} years ({years[0]}-{years[-1]}), target days {min(nd)}-{max(nd)}", flush=True)
    return p


def hindcast(month: int, v: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """(target years, (n, P) target, (n, P) lead-in or None) of the reforecast table, anomalies about their mean."""
    p = DATA / f"geps_hc_{month:02d}.npz"
    if not p.exists():
        return None
    with np.load(p) as f:
        lead = f[f"{v}_lead"].astype(np.float64) if v in LEAD else None
        return f["years"], f[v].astype(np.float64), lead


# ------------------------------------------------------------------------------------------------- the live run
def live(year: int, month: int, issue: dt.date | None = None) -> dict | None:
    """The newest GEPS run on or before the issue date (default the 15th of the month before) within SLACK days,
    reduced like the hindcast and less the table's means: {"init", "ndays", v: (P,), v_lead: (P,)}; None without one."""
    import xarray as xr

    first = dt.date(year, month, 1)
    if issue is None:
        issue = dt.date(year - (month == 1), 12 if month == 1 else month - 1, ISSUE_DAY)
    p = DATA / f"geps_hc_{month:02d}.npz"
    if not p.exists():
        return None
    means = dict(np.load(p))
    for back in range(SLACK + 1):
        d = issue - dt.timedelta(days=back)
        win = windows(d, first)
        if win is None:
            continue
        out = {"init": d.isoformat(), "ndays": int(win[0].sum())}
        try:
            for v, (tag, _, scale) in TAGS.items():
                with urllib.request.urlopen(RUN_URL.format(tag=tag, cycle=f"{d:%Y%m%d}"), timeout=120) as r:
                    ds = xr.open_dataset(io.BytesIO(r.read()))
                a = ds["anom"].values[:DAYS].astype(np.float64) * scale  # day L = init + L - 1, as the hindcast
                lat, lon = ds["latitude"].values.astype(float), ds["longitude"].values.astype(float)
                t, lead = reduce(a, lat, lon, v, *win)
                out[v], out[f"{v}_lead"] = t - means[f"mean_{v}"], lead - means[f"mean_{v}_lead"]
        except Exception:  # noqa: BLE001  no run that day (GEPS runs Mondays and Thursdays) or not yet archived
            continue
        return out
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="?", const="1,2,3,4,5,6,7,8,9,10,11,12", help="target months to tabulate")
    ap.add_argument("--live", help="YYYY-MM: print the live guidance")
    a = ap.parse_args()
    if a.build:
        from localdirs import env_dir

        store = env_dir("DATA_ARCHIVE", "geps_subx")
        for m in map(int, a.build.split(",")):
            build(m, store)
    if a.live:
        y, m = map(int, a.live.split("-"))
        g = live(y, m)
        if g is None:
            print("no GEPS run in the archive for this issue")
            return 1
        w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
        w[G.LAT_NH < 20] = 0
        print(f"GEPS {g['init']} 00Z: target days {g['ndays']} of {y}-{m:02d}")
        for v in TAGS:
            f = g[v].reshape(w.shape)
            print(f"  {v}: 20-90N mean {(w * f).sum() / w.sum():+.2f}, range {f.min():+.1f} .. {f.max():+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
