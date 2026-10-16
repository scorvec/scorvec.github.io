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
    at most PERSIST_DAYS days (age counted from the last real report, so nothing can chain); past
    that, with >= MIN_MOORINGS reporting-or-carried, it is interpolated in longitude from its
    neighbours; with fewer the TAO index is NaN for that day;
  * reanalysis fill (fill_df, see fill_from): a silent mooring takes the GLORYS12 anomaly at its own
    location, offset-corrected to its own TAO record; this outranks carry-forward (index() docstring).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sst_subsurface as ss

LONS = [165.0, 180.0, 190.0, 205.0, 220.0, 235.0, 250.0, 265.0]
LON_LABEL = {
    165.0: "165E",
    180.0: "180",
    190.0: "170W",
    205.0: "155W",
    220.0: "140W",
    235.0: "125W",
    250.0: "110W",
    265.0: "95W",
}
PERSIST_DAYS = 10
MIN_MOORINGS = 6
PROFILE_TOP = 25.0  # m: shallowest screened sensor must be at or above this
PROFILE_BOTTOM = 300.0  # m: a screened sensor must exist at or below this
PROFILE_MAX_GAP = 150.0  # m: largest allowed vertical gap between screened sensors (0-300 m + next deeper)
QSET = (1, 2, 3)  # quality codes kept (see audit: Q4 "lower quality" decision)
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
    return xr.Dataset(
        {
            "temp": (("time", "depth", "longitude"), T),
            "q": (("time", "depth", "longitude"), Q),
            "s": (("time", "depth", "longitude"), S),
        },
        coords={"time": times, "depth": depths, "longitude": ulons},
    )


def concat(dss: list[xr.Dataset]) -> xr.Dataset:
    """Join several parsed deliveries (different depth/lon sets), later files win on overlap."""
    depths = sorted(set().union(*[set(d.depth.values.tolist()) for d in dss]))
    lons = sorted(set().union(*[set(d.longitude.values.tolist()) for d in dss]))
    out = []
    for d0 in dss:
        d = d0.reindex(depth=depths, longitude=lons)
        d["q"] = d["q"].fillna(0).astype(np.int8)
        d["s"] = d["s"].fillna(0).astype(np.int8)
        out.append(d)
    ds = xr.concat(out, dim="time")
    idx = pd.Index(ds.time.values)
    return ds.isel(time=~idx.duplicated(keep="last")).sortby("time")


# ── screening ─────────────────────────────────────────────────────────────────
def screened_temp(ds: xr.DataArray | xr.Dataset, strict: bool = True, qset=QSET) -> xr.DataArray:
    """Temperature with failed / interpolated values (and quality codes outside qset) removed.
    strict=False returns the raw temperatures (the pre-2026-10-07 behaviour)."""
    t = ds["temp"]
    if not strict:
        return t
    qok = np.isin(ds["q"].values, list(qset))
    sok = np.isin(ds["s"].values, [1, 2, 5, 6])
    return t.where(qok & sok)


def profile_reason(temp: np.ndarray, depths: np.ndarray) -> str:
    """'ok' or why a screened profile cannot give a 0-300 m mean: none / few / bottom / top / gap."""
    m = np.isfinite(temp)
    if not m.any():
        return "none"
    if m.sum() < 3:
        return "few"
    d = np.sort(depths[m])
    if d.max() < PROFILE_BOTTOM:
        return "bottom"
    if d.min() > PROFILE_TOP:
        return "top"
    d = d[d <= d[np.searchsorted(d, PROFILE_BOTTOM)]]
    return "ok" if np.diff(d).max(initial=0) <= PROFILE_MAX_GAP else "gap"


def profile_ok(temp: np.ndarray, depths: np.ndarray) -> bool:
    return profile_reason(temp, depths) == "ok"


def mooring_anomaly(ds: xr.Dataset, strict: bool = True, detrend: bool = False, qset=QSET):
    """Per-mooring daily 0-300 m mean anomaly.

    Returns (anom DataFrame time x LONS, info dict of DataFrames: nsens, deepest, ok)."""
    t = screened_temp(ds, strict, qset)
    depths = ds.depth.values.astype(float)
    times = pd.to_datetime(ds.time.values)
    lons = [l for l in LONS if l in set(ds.longitude.values.astype(float))]
    grid = np.full((len(times), len(ss.DEPTH_GRID), len(lons)), np.nan)
    ok = np.zeros((len(times), len(lons)), bool)
    nsens = np.zeros((len(times), len(lons)), int)
    deepest = np.full((len(times), len(lons)), np.nan)
    reason = np.full((len(times), len(lons)), "none", dtype=object)
    for j, lo in enumerate(lons):
        col = t.sel(longitude=lo).values
        for i in range(len(times)):
            p = col[i]
            fin = np.isfinite(p)
            nsens[i, j] = int((fin & (depths <= 300)).sum())
            if fin.any():
                deepest[i, j] = depths[fin].max()
            reason[i, j] = profile_reason(p, depths)
            if strict and reason[i, j] != "ok":
                continue
            g = ss._vinterp(depths, p)
            if np.isfinite(g).any():
                grid[i, :, j] = g
                ok[i, j] = True if strict else np.isfinite(g).any()
    da = xr.DataArray(
        grid, dims=("time", "depth", "longitude"), coords={"time": times, "depth": ss.DEPTH_GRID, "longitude": lons}
    )
    coeffs = ss.load_or_build_coeffs(lons)
    an = da - ss.eval_climatology(coeffs, ss.harmonic_day(times))
    if detrend:
        an = ss.detrend(an)
    # strict: a complete profile must give a complete anomaly column; a NaN climatology cell would
    # silently shorten the depth mean, so require every level
    col = an.mean("depth", skipna=not strict)
    df = pd.DataFrame(col.values, index=times, columns=lons).reindex(columns=LONS)
    info = {
        "nsens": pd.DataFrame(nsens, index=times, columns=lons).reindex(columns=LONS),
        "deepest": pd.DataFrame(deepest, index=times, columns=lons).reindex(columns=LONS),
        "ok": df.notna(),
        "reason": pd.DataFrame(reason, index=times, columns=lons).reindex(columns=LONS),
    }
    return df, info


# ── index ─────────────────────────────────────────────────────────────────────
def carry(df: pd.DataFrame, persist: int = PERSIST_DAYS):
    """Carry each mooring's last REPORTED anomaly for <= persist calendar days.
    Returns (filled, age) - age = days since the value's report (0 = reported that day)."""
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    last_t = pd.DataFrame(
        np.where(df.notna(), idx.values[:, None], np.datetime64("NaT")), index=idx, columns=df.columns
    ).ffill()
    age = (pd.DataFrame(np.repeat(idx.values[:, None], df.shape[1], 1), index=idx, columns=df.columns) - last_t).apply(
        lambda c: c.dt.days
    )
    filled = df.ffill()
    filled = filled.where(age <= persist)
    age = age.where(age <= persist)
    return filled, age


def lon_interp(filled: pd.DataFrame, min_moorings: int = MIN_MOORINGS) -> pd.DataFrame:
    """Silent moorings estimated by linear interpolation in longitude between the nearest reporting
    (or carried) neighbours - the end moorings take their neighbour's value - on days with at least
    min_moorings present. The audit (dropping moorings on fully covered days, 1993-2026) measured
    RMSE 0.04-0.07 C for any one mooring this way against 0.08-0.17 C for averaging the rest."""
    lon = np.array(LONS)
    out = filled.copy()
    n = filled.notna().sum(axis=1)
    rows = n.index[(n >= min_moorings) & (n < len(LONS))]
    for t in rows:
        r = filled.loc[t].values.astype(float)
        m = np.isfinite(r)
        out.loc[t] = np.where(m, r, np.interp(lon, lon[m], r[m]))
    return out


def fill_from(df: pd.DataFrame, g: pd.DataFrame | None) -> pd.DataFrame | None:
    """Reanalysis anomaly at each mooring + that mooring's TAO-minus-reanalysis offset (365-day
    window, >= 120 paired days; see below). The offset absorbs the different climatologies
    (TAO 1991-2020 vs GLORYS 1993-2020) and slow model drift (GLORYS ran ~0.2 C warm of TAO in
    2024-25)."""
    if g is None:
        return None
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    g = g.reindex(index=idx, columns=df.columns)
    diff = df - g
    # centred 365-d window where it has >= 120 pairs (history: brackets a gap from both sides, so
    # a drifting model is tied to TAO before AND after); else the last such offset carried forward
    # (real time: only the past exists); else the first one carried back; else the record mean
    cen = diff.rolling(365, center=True, min_periods=120).mean()
    off = cen.ffill().fillna(cen.bfill())
    return g + off.fillna(diff.mean())


def index(
    df: pd.DataFrame, persist: int = PERSIST_DAYS, min_moorings: int = MIN_MOORINGS, fill_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Daily index frame - mean over the 8 FIXED longitudes, each mooring taken, in order, from

      1. its own screened TAO profile that day (live),
      2. the offset-corrected reanalysis at its location (fill_df, when given),
      3. its own last report, carried <= persist days,
      4. longitude interpolation from its neighbours, only when >= min_moorings have 1-3;
    NaN when fewer than min_moorings have 1-3. Audit 2026-10-07 (withholding moorings on fully
    covered days): reanalysis RMSE 0.03 C for one mooring, carry 0.03 (1 d) - 0.07 C (10 d),
    longitude interpolation 0.04-0.09 C.

    cat: 0 all 8 live; 1 >= min_moorings live, or (no reanalysis used) live + carried >=
    min_moorings; 2 complete only through the reanalysis (fewer than min_moorings live); 3 none."""
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    filled, age = carry(df, persist)
    live = df.notna()
    if fill_df is not None:
        f = fill_df.reindex(index=idx, columns=df.columns)
        src_fill = ~live & f.notna()
        comp = df.where(live, f)
    else:
        src_fill = pd.DataFrame(False, index=idx, columns=df.columns)
        comp = df.copy()
    src_carry = comp.isna() & filled.notna()
    comp = comp.where(comp.notna(), filled)
    n_have = comp.notna().sum(axis=1)
    est = lon_interp(comp, min_moorings)
    hc = est.mean(axis=1).where(n_have >= min_moorings)
    n_live, n_fill, n_carr = live.sum(axis=1), src_fill.sum(axis=1), src_carry.sum(axis=1)
    ok = hc.notna()
    cat = np.select(
        [
            ok & (n_live == len(LONS)),
            ok & ((n_live >= min_moorings) | ((n_fill == 0) & (n_live + n_carr >= min_moorings))),
            ok,
        ],
        [0, 1, 2],
        3,
    )
    return pd.DataFrame(
        {
            "hc": hc,
            "n_live": n_live,
            "n_fill": n_fill,
            "n_carried": n_carr,
            "n_interp": (len(LONS) - n_have).where(ok, 0),
            "cat": cat,
            "max_age": age.where(src_carry).max(axis=1),
        }
    )


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
