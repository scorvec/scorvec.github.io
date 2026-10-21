#!/usr/bin/env python3
"""Reference for the regional reflection index (reflection_index.py): Messori et al.'s (2022) events in ERA5 and what
the eastern Pacific and North America did after them.

    python build_reflection_index.py      # -> reference/reflection_index.json (inputs: ERA5_STORE, committed refs)

INDEX. RI = z(Siberia) - z(Canada): v*T* at 100 hPa over Siberia (45-75N 140-200E) and Canada (45-75N 230-280E)
standardised by the ERA5 1991-2020 day-of-year mean and sd (reference/vt100_era5.nc and vt100_clim.nc, the daily mean of
the 00 and 12 UTC fields, build_vt100_clim.py). An EVENT is RI above 1 for 10 days or more, December-March, with the
Siberian flux upward and the Canadian downward on average over it (their first two criteria; the third, a reflective
upper stratosphere, needs winds above 10 hPa the series does not carry).

AFTER THE EVENTS (ERA5, the local WeatherBench2 1.5 deg daily store):
  epo   PSL's East Pacific Oscillation: 500 hPa height over 20-35N minus 55-65N, 160-125W, standardised by its 1991-2020
        day-of-year mean and sd (366 slots, +/-15 days); negative = a ridge over Alaska and the eastern Pacific
  tna   2 m temperature over North America (30-60N, 240-290E), K from the same kind of normal
each over days -5..0, 1..5, 6..12 and 13..24 after onset, against random onsets in the same calendar months
(N_NULL draws, one-sided in the direction the windows lean).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import heatflux100 as HF
from localdirs import env_dir
from quantmet.degdays import doy366

OUT = HERE / "reference" / "reflection_index.json"
SERIES = HERE / "reference" / "vt100_era5.nc"
DJFM = (12, 1, 2, 3)
WINDOWS = ((-5, 0), (1, 5), (6, 12), (13, 24))
N_NULL = 2000
BASE = (1991, 2020)


def ri_series() -> tuple[pd.Series, pd.Series, pd.Series]:
    """(RI, Siberia, Canada) daily, ERA5."""
    E = xr.open_dataset(SERIES)
    C = xr.open_dataset(HF.CLIM)
    t = pd.DatetimeIndex(E.time.values)
    raw = {k: E[f"nh_{k}"].mean("hour").to_series() for k in HF.REGIONS}
    z = {k: (s - HF.clim_at(C, "nh", k, "mean", t)) / HF.clim_at(C, "nh", k, "sd", t) for k, s in raw.items()}
    return (z["sib"] - z["can"]).dropna(), raw["sib"], raw["can"]


def events(ri: pd.Series, sib: pd.Series, can: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    out, i, x = [], 0, ri.values
    while i < len(x):
        if x[i] <= HF.RI_THRESHOLD:
            i += 1
            continue
        j = i
        while j < len(x) and x[j] > HF.RI_THRESHOLD:
            j += 1
        d0, d1 = ri.index[i], ri.index[j - 1]
        if j - i >= HF.RI_DAYS and d0.month in DJFM and d1.month in DJFM and sib[d0:d1].mean() > 0 > can[d0:d1].mean():
            out.append((d0, j - i))
        i = j
    return out


def anomaly(s: pd.Series, scale: bool) -> pd.Series:
    """Anomaly from the 1991-2020 day-of-year mean (366 slots, +/-15 days); over its sd with `scale`."""
    slot = np.asarray(doy366(s.index)) + 1
    base = (s.index.year >= BASE[0]) & (s.index.year <= BASE[1])
    m, sd = np.full(367, np.nan), np.full(367, np.nan)
    for d in range(1, 367):
        sel = base & (np.abs(((slot - d) + 183) % 366 - 183) <= 15)
        m[d], sd[d] = s[sel].mean(), s[sel].std()
    return (s - m[slot]) / (sd[slot] if scale else 1.0)


def box(var: str, la0, la1, lo0, lo1, since: int) -> pd.Series:
    parts = []
    for f in sorted(env_dir("ERA5_STORE", "wb2_1p5_daily", var).glob(f"{var}_*.nc")):
        if int(f.stem.split("_")[-1]) < since:
            continue
        a = xr.open_dataset(f)[var].transpose("time", "latitude", "longitude")
        b = a.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
        w = np.cos(np.deg2rad(b.latitude))
        parts.append(((b.mean("longitude") * w).sum("latitude") / w.sum()).to_series())
    s = pd.concat(parts)
    s.index = pd.DatetimeIndex(s.index)
    return s


def after(series: pd.Series, onsets, rng) -> list[dict]:
    """The series over each window after the onsets, against random onsets in the same calendar months."""
    last = series.index[-1] - pd.Timedelta(days=30)
    pool = {m: series[(series.index.month == m) & (series.index <= last)].index for m in {d.month for d in onsets}}

    def mean_after(d, a0, a1):
        return series[d + pd.Timedelta(days=a0) : d + pd.Timedelta(days=a1)].mean()

    out = []
    for a0, a1 in WINDOWS:
        x = np.array([mean_after(d, a0, a1) for d in onsets])
        null = np.array(
            [
                np.nanmean([mean_after(pool[d.month][rng.integers(len(pool[d.month]))], a0, a1) for d in onsets])
                for _ in range(N_NULL)
            ]
        )
        obs = float(np.nanmean(x))
        side = -1 if obs < null.mean() else 1
        p = (np.sum(side * null >= side * obs) + 1) / (N_NULL + 1)
        out.append(
            {
                "days": [a0, a1],
                "mean": round(obs, 2),
                "share_negative": round(float(np.mean(x < 0)), 2),
                "null_mean": round(float(null.mean()), 2),
                "p": float(f"{p:.3g}"),
            }
        )
    return out


def main() -> int:
    ri, sib, can = ri_series()
    ev = events(ri, sib, can)
    onsets = [d for d, _ in ev]
    winters = len({d.year - (d.month < 7) for d in ri.index if d.month in DJFM})
    print(f"{len(ev)} events in {winters} winters, {ri.index[0]:%Y-%m-%d} .. {ri.index[-1]:%Y-%m-%d}", flush=True)
    since = ri.index[0].year
    epo = anomaly(box("z500", 20, 35, 200, 235, since) - box("z500", 55, 65, 200, 235, since), scale=True)
    tna = anomaly(box("t2m", 30, 60, 240, 290, since), scale=False)
    rng = np.random.default_rng(2022)
    doc = {
        "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "builder": "scripts/strat/build_reflection_index.py",
        "period": [f"{ri.index[0]:%Y-%m-%d}", f"{ri.index[-1]:%Y-%m-%d}"],
        "winters": winters,
        "events": [{"onset": f"{d:%Y-%m-%d}", "days": n} for d, n in ev],
        "epo_after": after(epo, onsets, rng),
        "tna_after": after(tna, onsets, rng),
        "definition": "RI = z(Siberia 45-75N 140-200E) - z(Canada 45-75N 230-280E), v*T* at 100 hPa, ERA5 1991-2020 "
        "day-of-year standardisation; event = RI > 1 for 10+ days, December-March, Siberia upward and Canada downward",
    }
    for k in ("epo_after", "tna_after"):
        print(f"  {k}: " + "; ".join(f"d{w['days'][0]}..{w['days'][1]} {w['mean']:+.2f} (p {w['p']})" for w in doc[k]))
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
