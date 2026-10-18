#!/usr/bin/env python3
"""The Walker circulation's two metrics in ERA5 at 00 UTC: their day-of-year climatology and every El Niño since 1963.

The Pacific cell strength (walker.pac_index: mean Ψ_W over 140°E-160°W, 300-700 hPa) and the Indo-Pacific ascent
longitude (walker.ascent_lon) of every day's CENTRED 7-day-mean equatorial divergent zonal wind - the statistic the
analysed and forecast series use (overturn_hist.week, overturn_fc.windows), at 00 UTC as all of them
(build_hadley_clim.py: the overturning has a diurnal cycle). Each day's u_D comes through walker.band_ud from ERA5's
u and v on the 13 levels: WeatherBench2 (1.5 deg) to January 2023, ARCO-ERA5 (0.25 deg, read straight from the store)
after. A day of year pools 1991-2020 and the days within +-POOL of it, as the Hadley climatology does; the events are
hadley_enso.events() (July-March RONI peak >= +1.0).

    python src/build_walker_metrics.py plan                                   # the fetch parts (JSON, for the workflow)
    python src/build_walker_metrics.py fetch --start 1990-12-29 --end 1994-01-03 --out part.npz
    python src/build_walker_metrics.py build part*.npz       # -> walker_metric_clim.nc + walker_events.nc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import build_hadley_clim as B
import hadley_enso as E
import walker
from doycal import noleap_doy

CLIM_OUT = B.REF / "walker_metric_clim.nc"
EVENTS_OUT = B.REF / "walker_events.nc"
P_HPA = np.asarray(walker.LEVELS, float)
LON = np.linspace(0.0, 360.0, 173)  # the chi solver's DH2 longitudes (walker.HIST's axis)
KEYS = ("pac", "asc_lon")
HALF, POOL = B.WIN // 2, B.POOL
PCTS = (10, 25, 50, 75, 90)


def metrics(ud: np.ndarray) -> dict:
    """The two metrics of one (lev, lon) divergent-wind field."""
    psi = walker.streamfunction(ud, P_HPA * 100.0)
    return {"pac": walker.pac_index(psi, P_HPA, LON), "asc_lon": walker.ascent_lon(psi, P_HPA, LON)}


def _ud(u, v) -> np.ndarray:
    """(lev, lon) u_D from one time's (level, ...) u and v DataArrays (any lat/lon order: the solver transposes by name)."""
    return np.stack([walker.band_ud(u.sel(level=lev), v.sel(level=lev))[0] for lev in walker.LEVELS])


def fetch(t0: pd.Timestamp, t1: pd.Timestamp, source: str) -> dict:
    """Daily 00 UTC u_D for the days t0..t1."""
    days, rows = [], []
    if source == "wb2":
        ds = xr.open_zarr(B.WB2, storage_options={"token": "anon"})
        u, v = (ds[f"{c}_component_of_wind"].sel(time=slice(t0, t1), level=walker.LEVELS) for c in ("u", "v"))
        t = pd.DatetimeIndex(u.time.values)
        sel = np.flatnonzero(t.hour == 0)
        for m0 in pd.date_range(t0, t1, freq="MS").union([t0]):
            idx = [i for i in sel if m0 <= t[i] < m0 + pd.offsets.MonthBegin(1)]
            if not idx:
                continue
            ub, vb = u.isel(time=idx).load(), v.isel(time=idx).load()
            for k in range(len(idx)):
                rows.append(_ud(ub.isel(time=k), vb.isel(time=k)))
                days.append(pd.Timestamp(ub.time.values[k]).normalize())
            print(f"  {m0:%Y-%m}: {len(idx)} days", flush=True)
    else:
        ds = xr.open_zarr(B.ARCO, chunks=None, storage_options={"token": "anon"})
        for d in pd.date_range(t0, t1):
            u = ds["u_component_of_wind"].sel(time=d, level=walker.LEVELS).load()
            v = ds["v_component_of_wind"].sel(time=d, level=walker.LEVELS).load()
            if bool(np.isnan(v.values).all()):  # ARCO pads days it has not published
                continue
            rows.append(_ud(u, v))
            days.append(d)
            if d.day == 1:
                print(f"  {d:%Y-%m}", flush=True)
    return {"date": np.array(days, "datetime64[D]"), "ud": np.asarray(rows, np.float32)}


def plan() -> list[dict]:
    """Ten parts for 1991-2020 (with the windows' margins), then each event window outside them."""
    half = pd.Timedelta(days=HALF)
    parts = [
        {
            "name": f"clim{y0}",
            "start": f"{pd.Timestamp(f'{y0}-01-01') - half:%Y-%m-%d}",
            "end": f"{pd.Timestamp(f'{y0 + 2}-12-31') + half:%Y-%m-%d}",
            "source": "wb2",
        }
        for y0 in range(1991, 2021, 3)
    ]
    lo, hi = pd.Timestamp("1991-01-01") - half, pd.Timestamp("2020-12-31") + half
    for ev in E.events():
        if pd.Timestamp(ev["start"]) >= lo and pd.Timestamp(ev["end"]) <= hi:
            continue  # inside the climatology's parts
        parts.append({"name": f"event{ev['year']}", "start": ev["start"], "end": ev["end"], "source": ev["source"]})
    return parts


def build(paths):
    acc_d, acc_u = [], []
    for p in paths:
        with np.load(p) as f:
            acc_d.append(f["date"])
            acc_u.append(f["ud"])
    date, first = np.unique(np.concatenate(acc_d), return_index=True)
    ud = np.concatenate(acc_u)[first]
    days = pd.DatetimeIndex(date)
    pos = {d: i for i, d in enumerate(days)}

    def week(d):
        w = [pos.get(d + pd.Timedelta(days=k)) for k in range(-HALF, HALF + 1)]
        return None if any(i is None for i in w) else metrics(ud[w].mean(0))

    rows, idx = [], []
    for d in days[(days.year >= 1991) & (days.year <= 2020)]:
        m = week(d)
        if m is not None:
            rows.append(m)
            idx.append(d)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    doy = noleap_doy(df.index)
    stats = {k: np.full((len(PCTS) + 2, 366), np.nan) for k in KEYS}
    counts = np.zeros(366, int)
    for d in range(1, 366):
        band = np.abs(((doy - d + 182.5) % 365) - 182.5) <= POOL
        for k in KEYS:
            v = df[k].values[band]
            v = v[np.isfinite(v)]
            if k == KEYS[0]:
                counts[d - 1] = v.size
            if v.size < 15 * (2 * POOL + 1):
                continue
            stats[k][0, d - 1], stats[k][1, d - 1] = v.mean(), v.std(ddof=1)
            for j, q in enumerate(PCTS):
                stats[k][2 + j, d - 1] = np.percentile(v, q)
    for k in KEYS:
        stats[k][:, 365] = stats[k][:, 364]
    counts[365] = counts[364]
    clim = xr.Dataset(
        {k: (("stat", "doy"), stats[k]) for k in KEYS},
        coords={"stat": ["mean", "sd"] + [f"p{q}" for q in PCTS], "doy": np.arange(1, 367)},
        attrs={
            "source": "ERA5 1991-2020 at 00 UTC (WeatherBench2 1.5 deg)",
            "basis": f"metrics of each day's centred 7-day-mean u_D, pooled over the 30 years and +-{POOL} days of year",
            "units": "pac: 10^10 kg/s; asc_lon: degrees east",
        },
    )
    clim["n_pooled"] = ("doy", counts)
    evs = E.events()
    data = {k: np.full((len(evs), 274), np.nan) for k in KEYS}
    for e, ev in enumerate(evs):
        for j, d in enumerate(E.season_days(ev["year"])):
            m = week(d)
            if m is not None:
                for k in KEYS:
                    data[k][e, j] = m[k]
        print(f"  {ev['label']} (RONI {ev['peak']:+.1f}): {np.isfinite(data['pac'][e]).sum()} days", flush=True)
    events = xr.Dataset(
        {k: (("event", "day"), v) for k, v in data.items()},
        coords={
            "event": [ev["label"] for ev in evs],
            "peak": ("event", [ev["peak"] for ev in evs]),
            "day": np.arange(274),
        },
        attrs={
            "basis": "metrics of the centred 7-day-mean ERA5 u_D at 00 UTC, day 0 = 1 July",
            "events": f"July-March RONI peak >= {E.EVENT_MIN} (CPC official)",
        },
    )
    return clim, events


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--source", choices=("wb2", "arco"), default="wb2")
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan()))
    elif a.cmd == "fetch":
        np.savez_compressed(a.out, **fetch(pd.Timestamp(a.start), pd.Timestamp(a.end), a.source))
    else:
        clim, events = build(a.parts)
        clim.to_netcdf(CLIM_OUT)
        events.to_netcdf(EVENTS_OUT)
        print(f"wrote {CLIM_OUT} (pooled n/doy {int(clim.n_pooled.min())}-{int(clim.n_pooled.max())}) and {EVENTS_OUT}")
        for lab, d in (("1 Feb", 32), ("1 May", 121), ("1 Aug", 213), ("1 Nov", 305)):
            r = clim.sel(doy=d)
            print(
                f"  {lab}: Pacific {float(r.pac.sel(stat='p50')):+.2f} [{float(r.pac.sel(stat='p10')):+.2f},"
                f"{float(r.pac.sel(stat='p90')):+.2f}]  ascent {float(r.asc_lon.sel(stat='p50')):.1f}E "
                f"[{float(r.asc_lon.sel(stat='p10')):.1f},{float(r.asc_lon.sel(stat='p90')):.1f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
