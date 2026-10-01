"""Shared loaders and statistics for the SSW precursor / vortex-structure study (2026-09-30).

Research only: nothing here is drawn on the site. Inputs are local (MERRA-2 zonal means in scripts/telecon/data/m2_strat
and scripts/strat/data/m2_zmu, NCEP R1 10 hPa heights in scripts/telecon/data/z10_r1_nh, the site's gap-filled
u(60N,10hPa) in scripts/strat/reference/strat_history.nc, ERA5 100-50 hPa tropical layer temperature cached by the
2026-09-27 tropical-precursor study, CPC QBO 50 hPa and RONI). Large intermediates go to ~/research/ssw_structure/data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HOME = Path.home()
SITE = HOME / "scorvec.github.io"
TD = SITE / "scripts" / "telecon" / "data"
SD = SITE / "scripts" / "strat"
WORK = HOME / "research" / "ssw_structure"
DATA = WORK / "data"
FIGS = WORK / "figs"
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
TTR_CACHE = HOME / "research" / "ssw_tropical_precursors" / "data" / "era5_ttr_layer.pkl"
RNG = np.random.default_rng(20260930)
WIN_MONTHS = (11, 12, 1, 2, 3)


# ------------------------------------------------------------------------------------------------ catalogue
def events() -> pd.DataFrame:
    """The site's 28 confirmed MERRA-2 major SSWs (CP07 on u(60N,10hPa); 2002-02-17 and 2025-11-28 are marginal and
    excluded, as on the site)."""
    j = json.loads((SD / "reference" / "strat_history_events.json").read_text())
    rows = [{"date": pd.Timestamp(e["date"]), "type": e["type"], "depth": e["depth"]} for e in j["ssw"] if not e["marginal"]]
    E = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    E["winter"] = [winter_of(d) for d in E.date]
    return E


def all_reversals() -> list[pd.Timestamp]:
    """Every CP07 central date in the catalogue, marginal ones included (used only to define the post-event
    refractory period, so a marginal reversal does not make a day eligible)."""
    j = json.loads((SD / "reference" / "strat_history_events.json").read_text())
    return sorted(pd.Timestamp(e["date"]) for e in j["ssw"])


def winter_of(d) -> int:
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


# ------------------------------------------------------------------------------------------------ series
def u10() -> pd.Series:
    """u(60N, 10 hPa), MERRA-2 daily means gap-filled from NCEP R1 + local offset (site's strat_history.nc)."""
    d = xr.open_dataset(SD / "reference" / "strat_history.nc")
    s = d.u10.to_series().astype(float)
    return s.asfreq("D")


def ttr() -> pd.Series:
    """ERA5 hypsometric 100-50 hPa layer temperature, 0-15N (K), 1959-2026-06."""
    s = pd.read_pickle(TTR_CACHE).astype(float)
    s = s[~s.index.duplicated()].sort_index().asfreq("D")
    return s.interpolate(limit=5)


def qbo50() -> pd.Series:
    """CPC 50 hPa QBO index (CDAS zonal wind, m/s, ORIGINAL data block), monthly at mid-month, interpolated to days."""
    rows = {}
    txt = (TD / "cpc_qbo_u50.txt").read_text().splitlines()
    for ln in txt:
        if "ANOMALY" in ln.upper() or "STANDARDIZED" in ln.upper():
            break                                   # only the first (ORIGINAL) block
        m = re.match(r"^\s*(\d{4})\s*(.*)$", ln)
        if not m:
            continue
        vals = re.findall(r"-?\d+\.\d+", m.group(2))
        if len(vals) != 12:
            continue
        for k, v in enumerate(vals):
            v = float(v)
            if v > -900:
                rows[pd.Timestamp(int(m.group(1)), k + 1, 15)] = v
    s = pd.Series(rows).sort_index()
    s = s[~s.index.duplicated()]
    return s.resample("D").interpolate("linear")


def roni(lag_months=1) -> pd.Series:
    """CPC RONI (official since Feb 2026), keyed by the centred month of its 3-month season. To avoid a look-ahead
    (the season centred on month m contains m+1), the value used on a day in month m is the season centred on
    m - lag_months, held for the month."""
    j = json.loads((SITE / "assets" / "sst" / "data" / "cpc_official.json").read_text())
    s = pd.Series({pd.Timestamp(k + "-01"): v for k, v in j["roni"].items() if v is not None}, dtype=float).sort_index()
    s.index = s.index + pd.DateOffset(months=lag_months)
    return s.resample("D").ffill()


def load_m2_fields(levels=(100., 70., 50., 30., 20., 10., 7., 5., 3., 2., 1.)):
    """MERRA-2 zonal means 40-90N (1 deg): ubar, Tbar on `levels`; vT at 100 hPa. Daily, gaps (<= 9 days) filled
    by linear interpolation in time and flagged. Cached."""
    cache = DATA / "m2_fields.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import glob
    fs = sorted(glob.glob(str(TD / "m2_strat" / "m2_strat_*.nc")))
    U, T, V, tt = [], [], [], []
    for f in fs:
        d = xr.open_dataset(f)
        d = d.sel(lev=list(levels))
        U.append(d.ubar.values.astype("float32")); T.append(d.Tbar.values.astype("float32"))
        V.append(xr.open_dataset(f).vT.sel(lev=100.0).values.astype("float32"))
        tt.append(d.time.values)
        lat = d.lat.values
    t = pd.DatetimeIndex(np.concatenate(tt))
    U, T, V = np.concatenate(U), np.concatenate(T), np.concatenate(V)
    keep = ~t.duplicated()
    t, U, T, V = t[keep], U[keep], T[keep], V[keep]
    # QC: the known all-zero day and physically impossible values
    bad = (np.abs(U).max((1, 2)) > 250) | (T.min((1, 2)) < 150) | (T.max((1, 2)) > 340) | (t == "1991-06-30") \
          | (np.abs(V).max(1) > 500)
    print(f"m2 QC: {int(bad.sum())} bad day(s): {[str(x.date()) for x in t[bad]]}", flush=True)
    full = pd.date_range(t[0], t[-1], freq="D")
    pos = full.get_indexer(t[~bad])
    out = {}
    for name, a in (("u", U[~bad]), ("T", T[~bad]), ("vT100", V[~bad])):
        b = np.full((len(full),) + a.shape[1:], np.nan, "float32")
        b[pos] = a
        flat = b.reshape(len(full), -1)
        miss = np.isnan(flat).any(1)
        x = np.arange(len(full))
        for j in range(flat.shape[1]):
            flat[miss, j] = np.interp(x[miss], x[~miss], flat[~miss, j])
        out[name] = flat.reshape(b.shape)
    filled = np.ones(len(full), bool); filled[pos] = False
    out.update(time=full.values.astype("datetime64[D]").astype(str), lat=lat, lev=np.array(levels), filled=filled)
    np.savez_compressed(cache, **out)
    return out


def load_zmu_low():
    """MERRA-2 zonal-mean u, 0.5 deg, -50..50N at 100..10 hPa (scripts/strat/data/m2_zmu), daily. Used only to extend
    the 10 hPa jet profile equatorward of 40N for the jet-width metric."""
    cache = DATA / "zmu_low.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import glob
    U, tt = [], []
    for f in sorted(glob.glob(str(SD / "data" / "m2_zmu" / "zmu_*.npz"))):
        z = np.load(f)
        lev = list(z["lev"]); i10 = lev.index(10.0)
        lat = z["lat"]; m = (lat >= 20) & (lat <= 40)
        U.append(z["u"][:, i10][:, m]); tt.append(z["time"])
    t = pd.DatetimeIndex(np.concatenate(tt))
    out = dict(time=t.values.astype("datetime64[D]").astype(str), lat=lat[m], u10=np.concatenate(U).astype("float32"))
    np.savez_compressed(cache, **out)
    return out


# ------------------------------------------------------------------------------------------------ anomalies
def harm_clim(s: pd.Series, nh=4, years=(1981, 2025)) -> pd.Series:
    t = s.index
    x = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    X = np.column_stack([np.ones(len(t))] + [f(k * x) for k in range(1, nh + 1) for f in (np.cos, np.sin)])
    ok = np.isfinite(s.values) & (t.year >= years[0]) & (t.year <= years[1])
    b = np.linalg.lstsq(X[ok], s.values[ok], rcond=None)[0]
    return pd.Series(X @ b, index=t)


def std_anom(s: pd.Series, nh=4) -> pd.Series:
    """Standardised daily anomaly: 4-harmonic climatology, sd by day of year smoothed over 31 days (the same recipe
    as the site figure's study)."""
    s = s.asfreq("D")
    a = s - harm_clim(s.interpolate(limit=5), nh)
    sd = a.groupby(a.index.dayofyear).transform("std").rolling(31, center=True, min_periods=5).mean()
    return a / sd.bfill().ffill()


# ------------------------------------------------------------------------------------------------ eligibility
def westerly_run(u: pd.Series) -> pd.Series:
    """Number of consecutive days BEFORE each day with u > 0 (the day itself not counted)."""
    x = (u.values > 0).astype(int)
    run = np.zeros(len(x), int)
    r = 0
    for i in range(len(x)):
        run[i] = r
        r = r + 1 if x[i] else 0
    return pd.Series(run, index=u.index)


def refractory(u: pd.Series, onsets) -> pd.Series:
    """True from each central date until the westerlies have returned for 20 consecutive days (CP07: no new event
    may start inside this period)."""
    out = pd.Series(False, index=u.index)
    x = u.values
    for c in onsets:
        i = u.index.get_indexer([pd.Timestamp(c)])[0]
        if i < 0:
            continue
        run, j = 0, i
        while j < len(x):
            run = run + 1 if x[j] > 0 else 0
            if run >= 20:
                break
            j += 1
        out.iloc[i:j + 1] = True
    return out


# ------------------------------------------------------------------------------------------------ statistics
def fdr_bh(p, q=0.10):
    p = np.asarray(p, float)
    flat = p[np.isfinite(p)]
    if not flat.size:
        return np.zeros(p.shape, bool)
    o = np.sort(flat); n = len(o)
    passed = o[o <= q * np.arange(1, n + 1) / n]
    thr = passed.max() if passed.size else -1
    return np.isfinite(p) & (p <= thr)


def lag_matrix(s: pd.Series, dates, lags):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    out = np.full((len(idx), len(lags)), np.nan)
    for k, L in enumerate(lags):
        out[:, k] = s.reindex(idx + pd.Timedelta(days=int(L))).values
    return out


def runs_of(mask, lags, minlen=1):
    """Contiguous runs (first, last lag) where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            if end - start + 1 >= minlen:
                out.append((int(lags[start]), int(lags[end])))
            start = None
    return out


def logit_fit(X, y, ridge=1e-6, iters=60):
    n, k = X.shape
    b = np.zeros(k)
    P = ridge * np.eye(k); P[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - P @ b
        H = (X.T * W) @ X + P
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def logit_pred(X, b):
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def auc(y, p):
    """ROC AUC by the rank-sum formula (ties averaged)."""
    y = np.asarray(y).astype(bool); p = np.asarray(p, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(p)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
