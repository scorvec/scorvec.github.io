"""Runtime helpers shared by the GEPS teleconnection, regime and probability products.

  readers       CPC / PSL / Long Paddock observed index files. The daily ones are refreshed each run into STATE
                (telecon_geps.refresh); CPC's monthly tele_index.nh is only read for 1991-2020 and ships in REF.
  period_shift  the ERA5 1991-2020 minus 2001-2020 re-basing on the 2.5 deg grid by day of year (was
                telecon_hindcast.period_shift; that module is now a laptop-only build script).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xarray as xr

import paths

TC = paths.TC_STATE                 # refreshed observed indices + run summaries
CPC = TC / "cpc"
LAT25 = np.arange(90.0, -90.1, -2.5)
LON25 = np.arange(0.0, 360.0, 2.5)


def _find(rel: str):
    """A file under STATE (refreshed), else REF (static), else the laptop's original store."""
    for root in (paths.TC_STATE, paths.TC_REF, paths.LOCAL_STORE / "telecon"):
        p = root / rel
        if p.exists():
            return p
    return paths.TC_STATE / rel


def period_shift(tag):
    """ERA5 1991-2020 minus 2001-2020 re-basing on the 2.5 deg grid, by day of
    year (index doy-1), the same term render_maps.reref adds to every map."""
    import render_maps as rm
    dummy = xr.DataArray(np.zeros((366, LAT25.size, LON25.size)), dims=("L", "latitude", "longitude"),
                         coords={"L": np.arange(366), "latitude": LAT25, "longitude": LON25})
    r = rm.reref(tag, dummy, pd.Timestamp("2000-01-01"))     # leap year: L = doy - 1
    return None if r is None else r.values.astype("float32")


def cpc_monthly():
    """CPC tele_index.nh -> DataFrame indexed by month-start, columns = patterns."""
    lines = _find("cpc/tele_index.nh").read_text().splitlines()
    cols = None
    rows = []
    for ln in lines:
        if cols is None and "NAO" in ln and "PNA" in ln:
            cols = [c for c in re.split(r"\s+", ln.strip()) if c][2:]
            continue
        m = re.match(r"^\s*(\d{4})\s+(\d{1,2})\s+(.*)$", ln)
        if not m or cols is None:
            continue
        # fixed-width columns: "-0.71-99.90" is two numbers, so split on the
        # number pattern rather than on whitespace
        vals = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(3))]
        rows.append([int(m.group(1)), int(m.group(2))] + vals[:len(cols)])
    df = pd.DataFrame(rows, columns=["y", "m"] + cols[:len(rows[0]) - 2])
    df.index = pd.to_datetime(dict(year=df.y, month=df.m, day=1))
    df = df.drop(columns=["y", "m"]).replace(-99.9, np.nan)
    return df


def cpc_daily(name):
    p = _find(f"cpc/{name}.daily.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    t = pd.to_datetime(dict(year=df.iloc[:, 0], month=df.iloc[:, 1], day=df.iloc[:, 2]))
    s = pd.Series(pd.to_numeric(df.iloc[:, 3], errors="coerce").values, index=t)
    return s.replace(-999.0, np.nan).dropna()


def psl_daily(name):
    """PSL daily EPO/WPO: 'yyyy mm dd value' in metres."""
    p = _find(f"psl/{name}.daily.txt")
    if not p.exists():
        return None
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["y", "m", "d", "v"])
    t = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    return pd.Series(df.v.values, index=t).dropna()


def cpc_soi_monthly():
    """CPC 'soi' file, the second (STANDARDIZED DATA) table -> monthly Series;
    -999.9 = missing. Fixed width: '-1.8-999.9' is two numbers."""
    lines = _find("cpc/soi.txt").read_text().splitlines()
    k = [i for i, ln in enumerate(lines) if "STANDARDIZED" in ln]
    idx, val = [], []
    for ln in lines[k[0]:] if k else []:
        m = re.match(r"^\s*(\d{4})(.*)$", ln)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(2))]
        if len(v) != 12:
            continue
        for mo, x in enumerate(v, 1):
            idx.append(pd.Timestamp(int(m.group(1)), mo, 1)); val.append(np.nan if x <= -999 else x)
    return pd.Series(val, index=idx).sort_index().dropna()


def cpc_eqsoi_monthly():
    """CPC reqsoi.for (standardized equatorial SOI) -> monthly Series; 999.9 = missing."""
    idx, val = [], []
    for ln in _find("cpc/reqsoi.for").read_text().splitlines():
        parts = ln.split()
        if len(parts) == 13 and parts[0].isdigit():
            for mo, x in enumerate(parts[1:], 1):
                x = float(x)
                idx.append(pd.Timestamp(int(parts[0]), mo, 1)); val.append(np.nan if x > 999 else x)
    return pd.Series(val, index=idx).sort_index().dropna()


def longpaddock_daily_soi():
    """Long Paddock (Queensland) daily SOI: 'Year Day Tahiti Darwin SOI', Troup
    x10 on the 1887-1989 base, 1991-06 -> yesterday."""
    p = _find("longpaddock/DailySOI1887-1989Base.txt")
    if not p.exists():
        return None
    df = pd.read_csv(p, sep=r"\s+")
    df.columns = [c.strip().lower() for c in df.columns]
    t = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(df["day"] - 1, unit="D")
    return pd.Series(pd.to_numeric(df["soi"], errors="coerce").values, index=t).dropna()
