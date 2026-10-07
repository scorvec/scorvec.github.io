#!/usr/bin/env python3
"""Equatorial Pacific 0-300 m temperature-anomaly index from the 8 TAO/TRITON moorings
(165E, 180, 170W, 155W, 140W, 125W, 110W, 95W at 0N) - the ONE implementation.

Used by sst_subsurface_events.py (the chart) and tao_hc_audit.py (the validation).

Definition
  * daily DISDEL profiles, parsed WITH their quality and source codes;
  * a sensor value is used only if its quality code is 1-3 (highest / default / adjusted)
    and its source code is a measurement (1, 2, 5, 6) - PMEL's temporally or vertically
    INTERPOLATED fills (source 3, 7, 8) and lower-quality / failed values (Q 4, 5) are dropped;
  * a mooring-day counts as REPORTING only if the screened profile spans 0-300 m: shallowest
    sensor <= 25 m, a sensor at >= 300 m, no vertical gap > PROFILE_MAX_GAP m. A profile that
    stops at 150 m used to enter the 0-300 m mean as if it were complete (the mean of whatever
    depths existed);
  * profile interpolated to 0-300 m every 5 m, anomaly against the committed 1991-2020
    harmonic climatology (sst_subsurface.tao_eq_clim_coeffs.nc, same 8 moorings), mean over depth;
  * index = FIXED 1/8 weight per mooring (never a nanmean over whoever reports - see the memory
    note tao-hc-index-coverage). A silent mooring is carried at its own last REPORTED anomaly for
    at most PERSIST_DAYS days (age counted from the last real report, so nothing can chain); with
    fewer than MIN_MOORINGS reporting-or-carried the TAO index is NaN for that day;
  * optional reanalysis fill (fill_df): a mooring silent for longer than PERSIST_DAYS takes the
    reanalysis anomaly at its own location (offset-adjusted, see glorys_fill) - flagged per day.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sst_subsurface as ss

LONS = [165.0, 180.0, 190.0, 205.0, 220.0, 235.0, 250.0, 265.0]
LON_LABEL = {165.0: "165E", 180.0: "180", 190.0: "170W", 205.0: "155W", 220.0: "140W",
             235.0: "125W", 250.0: "110W", 265.0: "95W"}
PERSIST_DAYS = 10
MIN_MOORINGS = 6
PROFILE_TOP = 25.0          # m: shallowest screened sensor must be at or above this
PROFILE_BOTTOM = 300.0      # m: a screened sensor must exist at or below this
PROFILE_MAX_GAP = 150.0     # m: largest allowed vertical gap between screened sensors (0-300 m + next deeper)
GOOD_Q = set("123")
GOOD_S = set("1256")
MISSING = -9.99


# ── parsing ───────────────────────────────────────────────────────────────────
def parse_qc(path: Path) -> xr.Dataset:
    """DISDEL t_xyzt_dy ascii -> Dataset temp/q/s (time, depth, longitude).

    q and s are the per-value quality and source digits (0 where the site has no sensor)."""
    text = Path(path).read_text(errors="replace")
    lons = [float(x) for x in re.search(r"^\s*Lon\s*=\s*(.+)$", text, re.M).group(1).split()]
    depths = [float(x) for x in re.search(r"^\s*Depth\s*=\s*(.+)$", text, re.M).group(1).split()]
    nd = len(depths)
    rec = {}
    cur_lon, didx = None, None
    for line in text.splitlines():
        m = re.match(r"\s*Location:\s+\S+\s+\S+\s+\(\s*(\d+),", line)
        if m:
            cur_lon = lons[int(m.group(1)) - 1]
            continue
        if line.lstrip().startswith("Index:"):
            didx = [int(x) - 1 for x in line.split()[1:]]
            continue
        m = re.match(r"\s*(\d{8})\s+\d{4}\s+(.*)$", line)
        if not (m and cur_lon is not None and didx):
            continue
        parts = m.group(2).split()
        n = len(didx)
        if len(parts) < n:
            continue
        try:
            v = np.array([float(x) for x in parts[:n]], float)
        except ValueError:
            continue
        qs = parts[n] if len(parts) > n else ""
        sc = parts[n + 1] if len(parts) > n + 1 else ""
        q = np.array([int(c) if c.isdigit() else 0 for c in qs.ljust(n, "0")[:n]], np.int8)
        s = np.array([int(c) if c.isdigit() else 0 for c in sc.ljust(n, "0")[:n]], np.int8)
        t = pd.Timestamp(m.group(1))
        key = (t, cur_lon)
        if key not in rec:
            rec[key] = [np.full(nd, np.nan), np.zeros(nd, np.int8), np.zeros(nd, np.int8)]
        T, Q, S = rec[key]
        T[didx] = np.where(np.isclose(v, MISSING), np.nan, v)
        Q[didx] = q
        S[didx] = s
    times = sorted({k[0] for k in rec})
    ulons = sorted({k[1] for k in rec})
    ti = {t: i for i, t in enumerate(times)}
    li = {l: i for i, l in enumerate(ulons)}
    T = np.full((len(times), nd, len(ulons)), np.nan)
    Q = np.zeros(T.shape, np.int8)
    S = np.zeros(T.shape, np.int8)
    for (t, l), (a, b, c) in rec.items():
        T[ti[t], :, li[l]] = a
        Q[ti[t], :, li[l]] = b
        S[ti[t], :, li[l]] = c
    return xr.Dataset({"temp": (("time", "depth", "longitude"), T),
                       "q": (("time", "depth", "longitude"), Q),
                       "s": (("time", "depth", "longitude"), S)},
                      coords={"time": times, "depth": depths, "longitude": ulons})


def concat(dss: list[xr.Dataset]) -> xr.Dataset:
    """Join several parsed deliveries (different depth/lon sets), later files win on overlap."""
    depths = sorted(set().union(*[set(d.depth.values.tolist()) for d in dss]))
    lons = sorted(set().union(*[set(d.longitude.values.tolist()) for d in dss]))
    out = []
    for d in dss:
        d = d.reindex(depth=depths, longitude=lons)
        d["q"] = d["q"].fillna(0).astype(np.int8)
        d["s"] = d["s"].fillna(0).astype(np.int8)
        out.append(d)
    ds = xr.concat(out, dim="time")
    idx = pd.Index(ds.time.values)
    return ds.isel(time=~idx.duplicated(keep="last")).sortby("time")


# ── screening ─────────────────────────────────────────────────────────────────
def screened_temp(ds: xr.DataArray | xr.Dataset, strict: bool = True) -> xr.DataArray:
    """Temperature with failed / lower-quality / interpolated values removed (strict=True).
    strict=False returns the raw temperatures (the pre-2026-10-07 behaviour)."""
    t = ds["temp"]
    if not strict:
        return t
    qok = np.isin(ds["q"].values, [1, 2, 3])
    sok = np.isin(ds["s"].values, [1, 2, 5, 6])
    return t.where(qok & sok)


def profile_ok(temp: np.ndarray, depths: np.ndarray) -> bool:
    m = np.isfinite(temp)
    if m.sum() < 3:
        return False
    d = np.sort(depths[m])
    if d.min() > PROFILE_TOP or d.max() < PROFILE_BOTTOM:
        return False
    d = d[d <= d[np.searchsorted(d, PROFILE_BOTTOM)]] if (d >= PROFILE_BOTTOM).any() else d
    return bool(np.diff(d).max(initial=0) <= PROFILE_MAX_GAP)


def mooring_anomaly(ds: xr.Dataset, strict: bool = True, detrend: bool = False):
    """Per-mooring daily 0-300 m mean anomaly.

    Returns (anom DataFrame time x LONS, info dict of DataFrames: nsens, deepest, ok)."""
    t = screened_temp(ds, strict)
    depths = ds.depth.values.astype(float)
    times = pd.to_datetime(ds.time.values)
    lons = [l for l in LONS if l in set(ds.longitude.values.astype(float))]
    grid = np.full((len(times), len(ss.DEPTH_GRID), len(lons)), np.nan)
    ok = np.zeros((len(times), len(lons)), bool)
    nsens = np.zeros((len(times), len(lons)), int)
    deepest = np.full((len(times), len(lons)), np.nan)
    for j, lo in enumerate(lons):
        col = t.sel(longitude=lo).values
        for i in range(len(times)):
            p = col[i]
            fin = np.isfinite(p)
            nsens[i, j] = int((fin & (depths <= 300)).sum())
            if fin.any():
                deepest[i, j] = depths[fin].max()
            if strict and not profile_ok(p, depths):
                continue
            g = ss._vinterp(depths, p)
            if np.isfinite(g).any():
                grid[i, :, j] = g
                ok[i, j] = True if strict else np.isfinite(g).any()
    da = xr.DataArray(grid, dims=("time", "depth", "longitude"),
                      coords={"time": times, "depth": ss.DEPTH_GRID, "longitude": lons})
    coeffs = ss.load_or_build_coeffs(lons)
    an = da - ss.eval_climatology(coeffs, times.dayofyear.values)
    if detrend:
        an = ss.detrend(an)
    if strict:
        # a complete profile must give a complete anomaly column; a NaN climatology cell would
        # silently shorten the depth mean, so require every level
        col = an.mean("depth", skipna=False)
    else:
        col = an.mean("depth", skipna=True)
    df = pd.DataFrame(col.values, index=times, columns=lons).reindex(columns=LONS)
    info = {"nsens": pd.DataFrame(nsens, index=times, columns=lons).reindex(columns=LONS),
            "deepest": pd.DataFrame(deepest, index=times, columns=lons).reindex(columns=LONS),
            "ok": df.notna()}
    return df, info


# ── index ─────────────────────────────────────────────────────────────────────
def carry(df: pd.DataFrame, persist: int = PERSIST_DAYS):
    """Carry each mooring's last REPORTED anomaly for <= persist calendar days.
    Returns (filled, age) - age = days since the value's report (0 = reported that day)."""
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    last_t = pd.DataFrame(np.where(df.notna(), idx.values[:, None], np.datetime64("NaT")),
                          index=idx, columns=df.columns).ffill()
    age = (pd.DataFrame(np.repeat(idx.values[:, None], df.shape[1], 1), index=idx,
                        columns=df.columns) - last_t).apply(lambda c: c.dt.days)
    filled = df.ffill()
    filled = filled.where(age <= persist)
    age = age.where(age <= persist)
    return filled, age


def index(df: pd.DataFrame, persist: int = PERSIST_DAYS, min_moorings: int = MIN_MOORINGS,
          fill_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily index frame: hc (TAO only, NaN below min_moorings), hc_filled (TAO + reanalysis fill
    for moorings silent > persist days; NaN only if the fill is missing too), n_live, n_carried,
    n_fill, and max_age (oldest carried value used)."""
    filled, age = carry(df, persist)
    live = (age == 0)
    used = filled.notna()
    n_used = used.sum(axis=1)
    hc = filled.mean(axis=1).where(n_used >= min_moorings)
    out = pd.DataFrame({"hc": hc, "n_live": live.sum(axis=1), "n_carried": (used & ~live).sum(axis=1),
                        "max_age": age.max(axis=1)})
    if fill_df is not None:
        f = fill_df.reindex(filled.index)
        comb = filled.where(used, f)
        n_comb = comb.notna().sum(axis=1)
        out["hc_filled"] = comb.mean(axis=1).where(n_comb == len(LONS))
        out["n_fill"] = (~used & f.notna()).sum(axis=1)
    return out


def smooth(s: pd.Series, n: int = 15) -> pd.Series:
    """Centred n-day mean where the full window exists; at the TAIL only, the trailing mean
    (>= n//2+1 days). Never drawn on a day whose own raw value is NaN - the old version filled
    every interior gap and ran the line up to 7 days past the last valid day."""
    c = s.rolling(n, center=True, min_periods=n).mean()
    tr = s.rolling(n, min_periods=n // 2 + 1).mean()
    last = s.last_valid_index()
    tail = s.index > (last - pd.Timedelta(days=n // 2)) if last is not None else np.zeros(len(s), bool)
    out = c.where(c.notna() | ~tail, tr)
    return out.where(s.notna())
