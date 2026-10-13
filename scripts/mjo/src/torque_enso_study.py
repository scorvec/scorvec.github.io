#!/usr/bin/env python3
"""Does El Nino raise global AAM through more positive mountain-torque EVENTS, or through a shift
of the standing cross-barrier pressure offset?  (study for the user, 2026-10-12)

Inputs (all committed by the one-off Actions workflows, nothing downloaded here):
  scripts/mjo/data/reference/torque_daily_era5.csv.gz    torque-history.yml (torque_history_era5.py)
  scripts/mjo/data/reference/torque_cmip6_monthly.csv.gz torque-cmip6.yml   (torque_history_cmip6.py)
  <cmip6 daily dir>/*_daily.csv                          torque-cmip6.yml artifact cmip6_daily_all
  assets/sst/data/cpc_official.json                      CPC official ONI (ENSO phase per season)
  scripts/mjo/data/reference/nina34.anom.data            PSL Nino-3.4 anomaly, ERSSTv6 (regressions)
  ~/era5_store wb2_1p5_daily(_global) z500 / u200        Q3 composites and jet lead-lag

Q1  events vs mean shift, per range x season, El Nino vs neutral (and La Nina), 1979-2025
Q2  the El Nino AAM rise split into mountain / friction / GWD torques (ERA5 + CMIP6)
Q3  lead-lag of range torque vs AAM / subtropical jet; z500 precursor composites and frequency

Every comparison carries a test: permutation of season-years (5000 draws) + bootstrap CI, BH-FDR
at 10% across the ranges x seasons family; regressions with moving-block bootstrap CIs.

    python scripts/mjo/src/torque_enso_study.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "scripts/mjo/data/reference"
ERA5_CSV = REF / "torque_daily_era5.csv.gz"
CMIP_CSV = REF / "torque_cmip6_monthly.csv.gz"
ONI_JSON = ROOT / "assets/sst/data/cpc_official.json"
N34_FILE = REF / "nina34.anom.data"
STORE = Path.home() / "era5_store"

RANGES = ["HIM", "ROC", "AND", "GRL", "ANT", "ALP"]
RNAME = {"HIM": "Himalaya/Tibet", "ROC": "Rockies/W. N.America", "AND": "Andes", "GRL": "Greenland",
         "ANT": "Antarctica", "ALP": "Alps/Europe"}
BANDS = ["glob", "b9060s", "b6030s", "b3000s", "b0030n", "b3060n", "b6090n"]
BNAME = {"glob": "global", "b9060s": "90-60S", "b6030s": "60-30S", "b3000s": "30S-0", "b0030n": "0-30N",
         "b3060n": "30-60N", "b6090n": "60-90N"}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SKEY = {"DJF": "01", "MAM": "04", "JJA": "07", "SON": "10"}       # ONI centred month of each season
HADLEY_DAY_TO_AAM = 86400 * 1e18 / 1e25                           # Hadley x 1 day -> 1e25 kg m2 s-1
RNG = np.random.default_rng(20261012)
NPERM = 20000


# ----------------------------------------------------------------------------- stats helpers
def bh_fdr(p, q=0.10):
    """Benjamini-Hochberg: boolean 'significant' array for p-values (NaN = not tested)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, bool)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    thr = q * (np.arange(1, m + 1) / m)
    passed = pv[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[o[:k]] = True
    out[ok] = sig
    return out


def perm_diff(a, b, n=NPERM, rng=RNG):
    """mean(a)-mean(b), two-sided permutation p over the pooled units, bootstrap 95% CI."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(diff=np.nan, p=np.nan, lo=np.nan, hi=np.nan, na=len(a), nb=len(b))
    d0 = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    idx = np.argsort(rng.random((n, len(pool))), axis=1)
    pa = pool[idx[:, :len(a)]].mean(1); pb = pool[idx[:, len(a):]].mean(1)
    p = (np.sum(np.abs(pa - pb) >= abs(d0) - 1e-12) + 1) / (n + 1)
    ba = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    bb = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return dict(diff=float(d0), p=float(p), lo=float(lo), hi=float(hi), na=len(a), nb=len(b),
                ma=float(a.mean()), mb=float(b.mean()))


def block_boot_slope(x, y, block=3, n=2000, rng=RNG):
    """OLS slope of y on x with a moving-block bootstrap 95% CI (blocks of consecutive units)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    N = len(x)
    if N < 8:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, r=np.nan, p=np.nan, n=N)
    xc = x - x.mean()
    s0 = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    r0 = float(np.corrcoef(x, y)[0, 1])
    nb = int(np.ceil(N / block))
    starts = rng.integers(0, N - block + 1, (n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :N]
    X = x[idx]; Y = y[idx]
    Xc = X - X.mean(1, keepdims=True)
    sl = (Xc * (Y - Y.mean(1, keepdims=True))).sum(1) / (Xc ** 2).sum(1)
    lo, hi = np.percentile(sl, [2.5, 97.5])
    # two-sided p from the bootstrap distribution centred on zero
    p = float(min(1.0, 2 * min((sl <= 0).mean(), (sl >= 0).mean()) + 1 / n))
    return dict(slope=s0, lo=float(lo), hi=float(hi), r=r0, p=p, n=N)


# ----------------------------------------------------------------------------- ENSO indices
def load_oni():
    d = json.loads(ONI_JSON.read_text())["oni"]
    s = pd.Series({pd.Timestamp(k + "-01"): v for k, v in d.items()}).sort_index()
    return s


def season_oni(oni, season, sy):
    k = pd.Timestamp(f"{sy}-{SKEY[season]}-01")
    return float(oni.get(k, np.nan))


def phase_of(v):
    if not np.isfinite(v):
        return "NA"
    return "EN" if v >= 0.5 else ("LN" if v <= -0.5 else "NEU")


def load_n34():
    rows = []
    for line in N34_FILE.read_text().splitlines()[1:]:
        c = line.split()
        if len(c) == 13 and c[0].isdigit():
            for m, v in enumerate(c[1:], 1):
                v = float(v)
                rows.append((pd.Timestamp(f"{c[0]}-{m:02d}-01"), np.nan if v < -99 else v))
    return pd.Series(dict(rows)).sort_index()


# ----------------------------------------------------------------------------- daily anomalies
def harmonic_anom(s: pd.Series, base=(1991, 2020), nh=4):
    """Anomaly vs a 1991-2020 day-of-year climatology (mean + nh harmonics)."""
    t = s.index
    yl = np.where(t.is_leap_year, 366.0, 365.0)
    ph = 2 * np.pi * (t.dayofyear.values - 0.5) / yl
    X = [np.ones(len(t))]
    for k in range(1, nh + 1):
        X += [np.cos(k * ph), np.sin(k * ph)]
    X = np.stack(X, 1)
    m = (t.year >= base[0]) & (t.year <= base[1]) & np.isfinite(s.values)
    beta, *_ = np.linalg.lstsq(X[m], s.values[m], rcond=None)
    return s - X @ beta, pd.Series(X @ beta, index=t)


def detrend(s: pd.Series, span):
    m = (s.index.year >= span[0]) & (s.index.year <= span[1]) & np.isfinite(s.values)
    tt = (s.index - s.index[0]).days.values.astype(float)
    c = np.polyfit(tt[m], s.values[m], 1)
    return s - np.polyval(c, tt), c[0] * 365.25


def lanczos_lowpass(x, cutoff_days=30.0, half=60):
    """Lanczos low-pass (periods >= cutoff), window 2*half+1 days; NaN within `half` of an edge/gap."""
    fc = 1.0 / cutoff_days
    k = np.arange(1, half + 1)
    sig = np.sin(np.pi * k / (half + 1)) / (np.pi * k / (half + 1))
    wk = np.sin(2 * np.pi * fc * k) / (np.pi * k) * sig
    w = np.concatenate([wk[::-1], [2 * fc], wk])
    w /= w.sum()
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    y = np.convolve(np.where(ok, x, 0.0), w, mode="same")
    bad = np.convolve((~ok).astype(float), np.ones(len(w)), mode="same") > 0
    y[bad] = np.nan
    y[:half] = np.nan; y[-half:] = np.nan
    return y


def season_of(idx: pd.DatetimeIndex):
    m = idx.month.values
    s = np.where(np.isin(m, (12, 1, 2)), "DJF", np.where(np.isin(m, (3, 4, 5)), "MAM",
                 np.where(np.isin(m, (6, 7, 8)), "JJA", "SON")))
    sy = idx.year.values + (m == 12)
    return s, sy


def load_era5(span=(1979, 2025)):
    df = pd.read_csv(ERA5_CSV, parse_dates=["date"]).set_index("date").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="D")
    df = df.reindex(full)
    num = df.select_dtypes("number").columns
    nbad = int(df[num[2]].isna().sum())
    # isolated incomplete days (an hour lost to NaN cells in the first streaming pass): linear in time
    df[num] = df[num].interpolate(limit=2, limit_area="inside")
    df.attrs["n_interpolated_days"] = nbad
    return df


def anomalies(df, cols, span):
    """Deseasonalised (1991-2020 harmonics) and linearly detrended (over `span`) daily anomalies."""
    out, trends = {}, {}
    for c in cols:
        a, _ = harmonic_anom(df[c])
        a, tr = detrend(a, span)
        out[c] = a; trends[c] = tr
    return pd.DataFrame(out), trends


# ----------------------------------------------------------------------------- events
def find_peaks(x, thr, sep=5):
    """Declustered local maxima of x above thr (thr per day), accepted peaks >= sep days apart."""
    x = np.asarray(x, float)
    xm = np.where(np.isfinite(x), x, -np.inf)
    cand = np.where((xm > thr) & (xm >= np.roll(xm, 1)) & (xm >= np.roll(xm, -1)))[0]
    order = cand[np.argsort(-xm[cand])]
    blocked = np.zeros(len(x), bool)
    keep = []
    for i in order:
        if not blocked[i]:
            keep.append(i)
            blocked[max(0, i - sep + 1):i + sep] = True
    return np.sort(np.array(keep, int))


def season_of_my(month, year):
    month = np.asarray(month); year = np.asarray(year)
    s = np.where(np.isin(month, (12, 1, 2)), "DJF", np.where(np.isin(month, (3, 4, 5)), "MAM",
                 np.where(np.isin(month, (6, 7, 8)), "JJA", "SON")))
    return s, year + (month == 12)


class _Arr:
    """Minimal stand-in for a daily Series on a model calendar: .values + month/year arrays."""
    def __init__(self, values, month, year):
        self.values = np.asarray(values, float); self.month = np.asarray(month); self.year = np.asarray(year)

    def __len__(self):
        return len(self.values)

    def __sub__(self, other):
        return _Arr(self.values - np.asarray(getattr(other, "values", other)), self.month, self.year)


def season_table(x, oni, sy_span, key, phase_fn=None):
    """Per season-year event statistics for one daily anomaly series (Q1 core). `x` is a daily
    pd.Series (DatetimeIndex) or an _Arr on a model calendar; `oni(season, sy)` -> ONI-like value
    (a pd.Series of official ONI is accepted too)."""
    if isinstance(x, pd.Series):
        xs = _Arr(x.values, x.index.month.values, x.index.year.values)
    else:
        xs = x
    oni_of = (lambda se, y: season_oni(oni, se, y)) if isinstance(oni, pd.Series) else oni
    lpv = lanczos_lowpass(xs.values)
    lp = _Arr(lpv, xs.month, xs.year)
    hp = xs - lp
    x = xs
    s, sy = season_of_my(xs.month, xs.year)
    insp = (sy >= sy_span[0]) & (sy <= sy_span[1]) & np.isfinite(lp.values)
    rows = []
    for se in SEASONS:
        m = insp & (s == se)
        sig_h = float(np.nanstd(hp.values[m]))
        sig_x = float(np.nanstd(x.values[m]))
        thr_h = np.where(s == se, sig_h, np.inf)
        thr_x = np.where(s == se, sig_x, np.inf)
        hv = np.where(insp, hp.values, np.nan)
        xv = np.where(insp, x.values, np.nan)
        pk_p = find_peaks(hv, thr_h); pk_n = find_peaks(-hv, thr_h)
        px_p = find_peaks(xv, thr_x); px_n = find_peaks(-xv, thr_x)
        win_p = np.zeros(len(x), bool); win_n = np.zeros(len(x), bool)
        for i in pk_p:
            win_p[max(0, i - 2):i + 3] = True
        for i in pk_n:
            win_n[max(0, i - 2):i + 3] = True
        for y in range(sy_span[0], sy_span[1] + 1):
            d = m & (sy == y)
            nd = int(d.sum())
            if nd < 80:
                continue
            o = oni_of(se, y)
            ip = pk_p[d[pk_p]]; ineg = pk_n[d[pk_n]]
            xp = px_p[d[px_p]]; xn = px_n[d[px_n]]
            rows.append(dict(
                key=key, season=se, sy=y, oni=o, phase=phase_of(o), super=bool(o >= 2.0), ndays=nd,
                sig_hp=sig_h, sig_x=sig_x,
                n_pos=len(ip), n_neg=len(ineg),
                amp_pos=float(hp.values[ip].mean()) if len(ip) else np.nan,
                amp_neg=float(hp.values[ineg].mean()) if len(ineg) else np.nan,
                x_mean=float(np.nanmean(x.values[d])), lp_mean=float(np.nanmean(lp.values[d])),
                hp_mean=float(np.nanmean(hp.values[d])),
                evc_pos=float(np.nansum(hp.values[d & win_p]) / nd),
                evc_neg=float(np.nansum(hp.values[d & win_n]) / nd),
                evx_pos=float(np.nansum(x.values[d & win_p]) / nd),
                frac_win_pos=float((d & win_p).sum() / nd),
                nx_pos=len(xp), nx_neg=len(xn),
                hp_sd=float(np.nanstd(hp.values[d])),
            ))
    return pd.DataFrame(rows), lp, hp


def shift_test(x: pd.Series, tab: pd.DataFrame, n=1000, rng=RNG):
    """Is the El Nino excess of TOTAL-anomaly events (x > +1 sd) what a pure shift of the neutral
    distribution by the El Nino-minus-neutral low-pass mean would give?  Per season."""
    s, sy = season_of(x.index)
    out = {}
    for se in SEASONS:
        t = tab[tab.season == se]
        en = t[t.phase == "EN"]; ne = t[t.phase == "NEU"]
        if len(en) < 3 or len(ne) < 3:
            continue
        sig_x = float(t.sig_x.iloc[0])
        cache = {}

        def counts(y, dshift):
            d = (s == se) & (sy == y)
            return len(find_peaks(np.where(d, x.values + dshift, np.nan), sig_x))

        def shifted_mean(years, dshift):
            return float(np.mean([counts(y, dshift) for y in years]))
        dlp = en.lp_mean.mean() - ne.lp_mean.mean()
        exp = shifted_mean(ne.sy.values, dlp)
        obs = en.nx_pos.mean()
        boots = []
        env, nev = en.nx_pos.values, ne.sy.values
        enlp, nelp = en.lp_mean.values, ne.lp_mean.values
        for _ in range(n):
            ie = rng.integers(0, len(en), len(en)); iu = rng.integers(0, len(ne), len(ne))
            dl = enlp[ie].mean() - nelp[iu].mean()
            key = round(float(dl), 2)
            if key not in cache:
                cache[key] = {y: counts(y, key) for y in nev}
            boots.append(env[ie].mean() - np.mean([cache[key][y] for y in nev[iu]]))
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)) + 1 / n))
        out[se] = dict(obs_en=float(obs), obs_neu=float(ne.nx_pos.mean()), expected_shift=float(exp),
                       excess=float(obs - exp), lo=float(lo), hi=float(hi), p=p, dlp=float(dlp))
    return out


def q1(an, oni, sy_span, prefix="mtn5", do_shift=True):
    tabs, comps, shifts = [], [], {}
    lps, hps = {}, {}
    for r in RANGES:
        t, lp, hp = season_table(an[f"{prefix}_{r}"], oni, sy_span, r)
        tabs.append(t)
        lps[r] = pd.Series(lp.values, index=an.index); hps[r] = pd.Series(hp.values, index=an.index)
        if do_shift:
            shifts[r] = shift_test(an[f"{prefix}_{r}"], t)
    tab = pd.concat(tabs, ignore_index=True)
    comp = compare_table(tab, RANGES)
    return tab, comp, shifts, lps, hps


METRICS = ["n_pos", "n_neg", "amp_pos", "amp_neg", "x_mean", "lp_mean", "hp_mean", "evc_pos", "evc_neg",
           "evx_pos", "nx_pos", "nx_neg", "hp_sd"]


def compare_table(tab, keys, metrics=METRICS, slopes=("n_pos", "n_neg", "amp_pos", "lp_mean", "x_mean", "nx_pos")):
    comps = []
    for r in keys:
        for se in SEASONS:
            t = tab[(tab.key == r) & (tab.season == se)]
            if t.empty:
                continue
            for cmp_, (ga, gb) in {"EN-NEU": ("EN", "NEU"), "LN-NEU": ("LN", "NEU")}.items():
                A = t[t.phase == ga]; B = t[t.phase == gb]
                for mt in metrics:
                    comps.append(dict(range=r, season=se, cmp=cmp_, metric=mt, **perm_diff(A[mt].values, B[mt].values)))
            A = t[t.super]; B = t[t.phase == "NEU"]
            for mt in metrics:
                comps.append(dict(range=r, season=se, cmp="SUPER-NEU", metric=mt,
                                  diff=float(A[mt].mean() - B[mt].mean()) if len(A) else np.nan,
                                  p=np.nan, lo=np.nan, hi=np.nan, na=len(A), nb=len(B),
                                  ma=float(A[mt].mean()) if len(A) else np.nan, mb=float(B[mt].mean())))
            for mt in slopes:
                b = block_boot_slope(t.oni.values, t[mt].values)
                comps.append(dict(range=r, season=se, cmp="ONI-slope", metric=mt, diff=b["slope"], p=b["p"],
                                  lo=b["lo"], hi=b["hi"], na=b["n"], nb=np.nan, r=b["r"]))
    comp = pd.DataFrame(comps)
    comp["sig"] = False
    for (cmp_, mt), g in comp.groupby(["cmp", "metric"]):
        comp.loc[g.index, "sig"] = bh_fdr(g.p.values)
    return comp


# ----------------------------------------------------------------------------- Q2 (ERA5 monthly)
def monthly_from_daily(an: pd.DataFrame, cols):
    """Monthly means of daily anomalies + dM/dt (Hadley) of each AAM column from the month-edge values."""
    mon = an[cols].resample("MS").mean()
    for b in [c for c in an.columns if c.startswith("aam_")]:
        Mb = an[b]
        edge = 0.5 * (Mb + Mb.shift(-1))              # AAM at the end of each day
        ends = edge.resample("MS").last()
        starts = edge.shift(1).resample("MS").first()
        nd = Mb.resample("MS").count()
        mon["dmdt_" + b[4:]] = (ends - starts) / (nd * HADLEY_DAY_TO_AAM)
    return mon


def lag_regress(mon: pd.DataFrame, n34: pd.Series, cols, lags=range(-6, 13), block=24, n=2000):
    """Regression (per K of Nino-3.4) of each monthly series at month t+lag on Nino-3.4 at month t."""
    out = []
    x0 = n34.reindex(mon.index)
    for c in cols:
        for L in lags:
            y = mon[c].shift(-L)
            b = block_boot_slope(x0.values, y.values, block=block, n=n)
            out.append(dict(series=c, lag=L, **b))
    df = pd.DataFrame(out)
    df["sig"] = False
    for c, g in df.groupby("series"):
        df.loc[g.index, "sig"] = bh_fdr(g.p.values)
    return df


def onset_years(oni, span):
    """Developing El Nino years Y: DJF(Y/Y+1) ONI >= 0.5 and DJF(Y-1/Y) ONI < 0.5 (fresh onsets).
    Super: peak ONI (OND..DJF) >= 2.0. Neutral: both DJFs inside (-0.5, 0.5)."""
    en, sup, neu = [], [], []
    for y in range(span[0], span[1] + 1):
        o1 = season_oni(oni, "DJF", y + 1); o0 = season_oni(oni, "DJF", y)
        if not np.isfinite(o1):
            continue
        peak = np.nanmax([float(oni.get(pd.Timestamp(f"{y}-{m:02d}-01"), np.nan)) for m in (10, 11, 12)] + [o1])
        if o1 >= 0.5 and o0 < 0.5:
            en.append(y)
            if peak >= 2.0:
                sup.append(y)
        elif -0.5 < o1 < 0.5 and -0.5 < o0 < 0.5:
            neu.append(y)
    return en, sup, neu


def budget_composite(an: pd.DataFrame, years, terms, d0="04-01", d1="12-31", pad=3, mcol="aam_glob"):
    """Per year: integral of each torque anomaly over [d0, d1] (1e25 kg m2/s) and the AAM anomaly change
    between 7-day means centred on the window edges."""
    rows = []
    for y in years:
        a = pd.Timestamp(f"{y}-{d0}"); b = pd.Timestamp(f"{y}-{d1}")
        r = {"year": y}
        for t in terms:
            r[t] = float(an[t].loc[a:b].sum() * HADLEY_DAY_TO_AAM)
        Ms = an[mcol].loc[a - pd.Timedelta(days=pad + 1):a + pd.Timedelta(days=pad - 1)].mean()
        Me = an[mcol].loc[b - pd.Timedelta(days=pad):b + pd.Timedelta(days=pad)].mean()
        r["dM"] = float(Me - Ms)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CMIP6
def model_oni(n34: pd.Series):
    """ONI-like index from a model's monthly Nino-3.4 (K): monthly-clim anomaly, linearly detrended,
    3-month running mean. Index = first of the centre month."""
    a = n34 - n34.groupby(n34.index.month).transform("mean")
    tt = np.arange(len(a))
    a = a - np.polyval(np.polyfit(tt, a.values, 1), tt)
    return a, a.rolling(3, center=True).mean()


def cmip6_monthly_member(g: pd.DataFrame):
    """One model member: monthly anomalies (clim + trend removed), dM/dt (Hadley) from the centred
    difference of monthly-mean AAM, Nino-3.4 anomaly and ONI-like index."""
    g = g.sort_values(["year", "month"])
    idx = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    g = g.set_index(idx)
    num = [c for c in g.columns if c not in ("model", "member", "year", "month")]
    out = {}
    tt = np.arange(len(g))
    for c in num:
        x = g[c].astype(float)
        if not np.isfinite(x).any() or np.nanstd(x.values) == 0:
            continue
        a = x - x.groupby(x.index.month).transform("mean")
        ok = np.isfinite(a.values)
        a = a - np.polyval(np.polyfit(tt[ok], a.values[ok], 1), tt)
        out[c] = a
    an = pd.DataFrame(out)
    days = 365.25 / 12
    M = g["aam_glob"].astype(float)
    an["dmdt_glob"] = (M.shift(-1) - M.shift(1)) / (2 * days * HADLEY_DAY_TO_AAM)
    an["dmdt_glob"] -= an["dmdt_glob"].groupby(an.index.month).transform("mean")
    an["T_glob"] = an["mtn_glob"] + an["fric_glob"] + (an["gwd_glob"] if "gwd_glob" in an else 0.0)
    an["T_mf_glob"] = an["mtn_glob"] + an["fric_glob"]
    n34a, oni = model_oni(g["nino34"].astype(float))
    an["n34"] = n34a; an["oni"] = oni
    # absolute closure: mean torque vs mean tendency (the latter ~0)
    absT = float((g["mtn_glob"] + g["fric_glob"]).mean())
    absG = float(g["gwd_glob"].mean()) if "gwd_glob" in g and g["gwd_glob"].notna().any() else np.nan
    return an, dict(abs_mtn=float(g["mtn_glob"].mean()), abs_fric=float(g["fric_glob"].mean()),
                    abs_gwd=absG, abs_mf=absT,
                    abs_dmdt=float(((M.iloc[-1] - M.iloc[0]) / (len(M) * days * HADLEY_DAY_TO_AAM))))


def lag_slopes(an: pd.DataFrame, cols, lags=range(-6, 13)):
    """OLS slopes (per K) of each column at t+lag on n34 at t; plain numbers (models pooled later)."""
    x = an["n34"].values
    out = {}
    for c in cols:
        if c not in an:
            continue
        for L in lags:
            y = an[c].shift(-L).values
            ok = np.isfinite(x) & np.isfinite(y)
            xc = x[ok] - x[ok].mean()
            out[(c, L)] = float((xc * (y[ok] - y[ok].mean())).sum() / (xc ** 2).sum())
    return out


def cmip6_onset_years(oni: pd.Series):
    """Model El Nino onsets: DJF(Y/Y+1) ONI >= 0.5 and DJF(Y-1/Y) < 0.5; neutral: both in (-0.5, 0.5)."""
    en, neu = [], []
    for y in range(oni.index.year.min() + 1, oni.index.year.max()):
        k1 = pd.Timestamp(f"{y + 1}-01-01"); k0 = pd.Timestamp(f"{y}-01-01")
        if k1 not in oni.index or k0 not in oni.index:
            continue
        o1, o0 = oni[k1], oni[k0]
        if not (np.isfinite(o1) and np.isfinite(o0)):
            continue
        if o1 >= 0.5 and o0 < 0.5:
            en.append(y)
        elif -0.5 < o1 < 0.5 and -0.5 < o0 < 0.5:
            neu.append(y)
    return en, neu


def cmip6_composite(an: pd.DataFrame, raw_aam: pd.Series, years, terms):
    """Apr-Dec integrated torque anomalies (1e25) and the AAM-anomaly change Mar/Apr -> Dec/Jan."""
    rows = []
    days = 365.25 / 12
    for y in years:
        r = {"year": y}
        sl = an.loc[f"{y}-04-01":f"{y}-12-01"]
        if len(sl) < 9:
            continue
        for t in terms:
            if t in an:
                r[t] = float(sl[t].sum() * days * HADLEY_DAY_TO_AAM)
        M = an["aam_glob"]
        try:
            r["dM"] = float(0.5 * (M[pd.Timestamp(f"{y}-12-01")] + M[pd.Timestamp(f"{y + 1}-01-01")])
                            - 0.5 * (M[pd.Timestamp(f"{y}-03-01")] + M[pd.Timestamp(f"{y}-04-01")]))
        except KeyError:
            continue
        rows.append(r)
    return pd.DataFrame(rows)


def across_models(per_model: pd.DataFrame, value="slope", group=("series",)):
    """Median, IQR, sign agreement and a one-sample t-test across models for each group."""
    from scipy import stats
    rows = []
    for k, g in per_model.groupby(list(group)):
        v = g[value].dropna().values
        if len(v) < 4:
            continue
        t, p = stats.ttest_1samp(v, 0.0)
        med = float(np.median(v))
        agree = float(np.mean(np.sign(v) == np.sign(med))) if med != 0 else 0.0
        rows.append(dict(zip(group, k if isinstance(k, tuple) else (k,))) |
                     dict(n=len(v), median=med, q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                          mean=float(v.mean()), p=float(p), agree=agree))
    df = pd.DataFrame(rows)
    if len(df):
        df["fdr"] = bh_fdr(df.p.values)
        df["robust"] = df.fdr & (df.agree >= 0.8)
    return df


def cmip6_daily_events(dd: pd.DataFrame, n34m: pd.Series, key_prefix="mtn"):
    """Event statistics for one model's daily range torques (model calendar), ENSO phase from the
    model's own ONI-like index."""
    d = dd.copy()
    ymd = d["date"].str.split("-", expand=True).astype(int)
    d["y"], d["m"], d["dd"] = ymd[0].values, ymd[1].values, ymd[2].values
    ylen = d.groupby("y").size().median()
    doy = d.groupby("y").cumcount().values + 1
    ph = 2 * np.pi * (doy - 0.5) / ylen
    X = np.stack([np.ones(len(d))] + [f(k * ph) for k in range(1, 5) for f in (np.cos, np.sin)], 1)
    _, oni = model_oni(n34m)

    def oni_of(se, sy):
        k = pd.Timestamp(f"{sy}-{SKEY[se]}-01")
        return float(oni.get(k, np.nan))
    tabs = []
    tt = np.arange(len(d), dtype=float)
    for r in RANGES:
        x = d[f"{key_prefix}_{r}"].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        a = x - X @ beta
        a = a - np.polyval(np.polyfit(tt, a, 1), tt)
        t, _, _ = season_table(_Arr(a, d["m"].values, d["y"].values), oni_of,
                               (int(d.y.min()) + 1, int(d.y.max())), r)
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)


# ----------------------------------------------------------------------------- Q3
def lagcorr(x, y, mask, lags=range(-15, 16)):
    """corr(x(t), y(t+L)) over days t in mask; N_eff (Bretherton) two-sided p per lag."""
    from scipy import stats
    out = []
    n = len(x)
    for L in lags:
        idx = np.where(mask)[0]
        idx = idx[(idx + L >= 0) & (idx + L < n)]
        a = x[idx]; b = y[idx + L]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r = float(np.corrcoef(a, b)[0, 1])
        r1 = pd.Series(x).autocorr(1); r2 = pd.Series(y).autocorr(1)
        neff = len(a) * (1 - r1 * r2) / (1 + r1 * r2)
        tstat = r * np.sqrt(max(neff - 2, 1) / max(1 - r * r, 1e-9))
        p = float(2 * stats.t.sf(abs(tstat), max(neff - 2, 1)))
        out.append(dict(lag=L, r=r, p=p, neff=float(neff)))
    df = pd.DataFrame(out)
    df["sig"] = bh_fdr(df.p.values)
    return df


def store_field(var, years, region, glob=True):
    """Daily 1.5-degree field from ~/era5_store (wb2_1p5_daily_global for 1991-2020 global; wb2_1p5_daily
    for NH 1959-2026) over a lat/lon box; returns (time, lat, lon) DataArray, transposed by name."""
    import xarray as xr
    base = STORE / ("wb2_1p5_daily_global" if glob else "wb2_1p5_daily") / var
    la0, la1, lo0, lo1 = region
    parts = []
    for y in years:
        f = base / f"{var}_{y}.nc"
        if not f.exists():
            continue
        d = xr.open_dataset(f)
        v = d[list(d.data_vars)[0]].transpose("time", "latitude", "longitude")
        v = v.sortby("latitude").sel(latitude=slice(la0, la1))
        if lo0 < lo1:
            v = v.sel(longitude=slice(lo0, lo1))
        else:
            v = xr.concat([v.sel(longitude=slice(lo0, 360)), v.sel(longitude=slice(0, lo1))], "longitude")
        v = v.astype("float32").load()
        if parts:
            # the store's year files carry float-jittered coordinates (an outer-join concat turned
            # 58% of the NH z500 box into NaN): pin every year to the first year's grid
            p0 = parts[0]
            if v.shape[1:] != p0.shape[1:] or not (np.allclose(v.latitude, p0.latitude, atol=0.02)
                                                  and np.allclose(v.longitude % 360, p0.longitude % 360, atol=0.02)):
                v = v.reindex(latitude=p0.latitude, longitude=p0.longitude, method="nearest", tolerance=0.05)
            else:
                v = v.assign_coords(latitude=p0.latitude, longitude=p0.longitude)
        parts.append(v)
    out = xr.concat(parts, "time", join="exact")
    bad = float(np.isnan(out.values).mean())
    if bad > 1e-4:
        raise ValueError(f"{var} store field has {bad:.4f} NaN after the grid pinning")
    return out


def field_anom_hp(da):
    """Harmonic (1991-2020 or available) anomaly per gridpoint, then the <30-day part (Lanczos)."""
    t = pd.DatetimeIndex(da.time.values)
    yl = np.where(t.is_leap_year, 366.0, 365.0)
    ph = 2 * np.pi * (t.dayofyear.values - 0.5) / yl
    X = np.stack([np.ones(len(t))] + [f(k * ph) for k in range(1, 5) for f in (np.cos, np.sin)], 1)
    Y = da.values.reshape(len(t), -1).astype(np.float64)
    m = (t.year >= 1991) & (t.year <= 2020)
    if m.sum() < 3000:
        m = np.ones(len(t), bool)
    beta, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
    A = Y - X @ beta
    # Lanczos low-pass along time for every column (FFT convolution)
    from scipy.signal import fftconvolve
    half = 60; fc = 1 / 30.0
    k = np.arange(1, half + 1)
    sig = np.sin(np.pi * k / (half + 1)) / (np.pi * k / (half + 1))
    wk = np.sin(2 * np.pi * fc * k) / (np.pi * k) * sig
    w = np.concatenate([wk[::-1], [2 * fc], wk]); w /= w.sum()
    L = fftconvolve(A, w[:, None], mode="same", axes=0)
    L[:half] = np.nan; L[-half:] = np.nan
    H = A - L
    return (da.copy(data=H.reshape(da.shape).astype("float32")),
            da.copy(data=A.reshape(da.shape).astype("float32")))


def composite_test(F, idx_a, idx_b=None):
    """Gridpoint t-test of mean(F[idx_a]) vs 0 (or Welch vs F[idx_b]); returns mean, p."""
    from scipy import stats
    A = F[idx_a]
    if idx_b is None:
        t, p = stats.ttest_1samp(A, 0.0, axis=0, nan_policy="omit")
        return np.nanmean(A, 0), p
    B = F[idx_b]
    t, p = stats.ttest_ind(A, B, axis=0, equal_var=False, nan_policy="omit")
    return np.nanmean(A, 0) - np.nanmean(B, 0), p


# ============================================================================= drivers
def joint_boot_contrib(mon, n34, cols, lags=range(-6, 0), block=24, n=2000, rng=RNG):
    """Sum over `lags` of the regression of each monthly torque/tendency on Nino-3.4, converted to the
    AAM it implies (1e25 kg m2/s per K): the part of the pre-peak AAM rise each term supplies.
    Moving-block bootstrap (24-month blocks) of the whole set jointly -> CIs that respect covariance."""
    x = n34.reindex(mon.index).values
    Y = {c: np.stack([mon[c].shift(-L).values for L in lags], 1) for c in cols}
    days = 365.25 / 12

    def est(ix):
        xs = x[ix]
        out = {}
        for c in cols:
            ys = Y[c][ix]
            ok = np.isfinite(xs)[:, None] & np.isfinite(ys)
            s = 0.0
            for j in range(ys.shape[1]):
                m = ok[:, j]
                xc = xs[m] - xs[m].mean()
                s += (xc * (ys[m, j] - ys[m, j].mean())).sum() / (xc ** 2).sum()
            out[c] = s * days * HADLEY_DAY_TO_AAM
        return out
    N = len(x)
    base = est(np.arange(N))
    nb = int(np.ceil(N / block))
    boots = {c: [] for c in cols}
    for _ in range(n):
        st = rng.integers(0, N - block + 1, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:N]
        e = est(ix)
        for c in cols:
            boots[c].append(e[c])
    res = {}
    for c in cols:
        b = np.array(boots[c])
        p = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / n))
        res[c] = dict(est=float(base[c]), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), p=p)
    return res


def era5_monthly(df, span):
    terms = ["mtn", "fric", "gwd"]
    cols = [f"{t}_{b}" for t in terms for b in BANDS] + [f"mtn5_{r}" for r in RANGES] + \
        [f"fric_{r}" for r in RANGES] + [f"gwd_{r}" for r in RANGES]
    acols = [f"aam_{b}" for b in BANDS] + [c for c in df.columns if c.startswith("aamm_") and "_h" not in c]
    an, tr = anomalies(df, cols + acols, span)
    if "aamm_glob" in an:
        an["aamt_glob"] = an["aam_glob"] + an["aamm_glob"]
    an = an.loc[f"{span[0]}-01-01":f"{span[1]}-12-31"]
    mon = an[cols].resample("MS").mean()
    for b in [c for c in an.columns if c.startswith(("aam_", "aamm_", "aamt_"))]:
        Mb = an[b]
        edge = 0.5 * (Mb + Mb.shift(-1))
        ends = edge.resample("MS").last()
        starts = edge.shift(1).resample("MS").first()
        nd = Mb.resample("MS").count()
        pre = {"aam_": "dmdt_", "aamm_": "dmmdt_", "aamt_": "dmtdt_"}
        for k, v in pre.items():
            if b.startswith(k):
                mon[v + b[len(k):]] = (ends - starts) / (nd * HADLEY_DAY_TO_AAM)
        mon[b] = Mb.resample("MS").mean()
    mon["T_glob"] = mon.mtn_glob + mon.fric_glob + mon.gwd_glob
    tend = "dmtdt_glob" if "dmtdt_glob" in mon else "dmdt_glob"
    mon["R_glob"] = mon[tend] - mon["T_glob"]
    return an, mon, tr, tend


def run_q1(df, oni, out, res, span=(1979, 2025)):
    cols = [f"{p}_{r}" for p in ("mtn5", "mtn", "mtn1", "mtn25") for r in RANGES]
    an, tr = anomalies(df, cols, span)
    tab, comp, shifts, lps, hps = q1(an, oni, span)
    tab.to_csv(out / "q1_season_table.csv", index=False)
    comp.to_csv(out / "q1_compare.csv", index=False)
    cells = [(r, se) for r in shifts for se in shifts[r]]
    sig = bh_fdr([shifts[r][se]["p"] for r, se in cells])
    for (r, se), s_ in zip(cells, sig):
        shifts[r][se]["fdr"] = bool(s_)
    (out / "q1_shift_test.json").write_text(json.dumps(shifts, indent=1))
    # decomposition of the El Nino - neutral seasonal-mean difference
    dec = []
    for r in RANGES:
        for se in SEASONS:
            c = comp[(comp.range == r) & (comp.season == se) & (comp.cmp == "EN-NEU")].set_index("metric")
            dec.append(dict(range=r, season=se, d_total=c.loc["x_mean", "diff"], p_total=c.loc["x_mean", "p"],
                            sig_total=bool(c.loc["x_mean", "sig"]),
                            d_lowpass=c.loc["lp_mean", "diff"], d_highpass=c.loc["hp_mean", "diff"],
                            d_event_pos=c.loc["evc_pos", "diff"], d_event_neg=c.loc["evc_neg", "diff"],
                            p_event_pos=c.loc["evc_pos", "p"], sig_event_pos=bool(c.loc["evc_pos", "sig"])))
    dec = pd.DataFrame(dec)
    dec.to_csv(out / "q1_decomposition.csv", index=False)
    # sensitivities: 1959-2025 and the 2.5-degree (h, p_s coarsened) torque
    an59, _ = anomalies(df, [f"mtn5_{r}" for r in RANGES], (1959, 2025))
    _, comp59, _, _, _ = q1(an59, oni, (1959, 2025), do_shift=False)
    comp59.to_csv(out / "q1_compare_1959_2025.csv", index=False)
    _, comp25, _, _, _ = q1(an, oni, span, prefix="mtn25", do_shift=False)
    comp25.to_csv(out / "q1_compare_mtn25.csv", index=False)
    res["q1"] = dict(n_season_years=tab.groupby(["season", "phase"]).sy.nunique().unstack().to_dict(),
                     trends_hadley_per_yr={k: v for k, v in tr.items() if k.startswith("mtn5")},
                     sig_cells={m: comp[(comp.cmp == "EN-NEU") & (comp.metric == m) & comp.sig][["range", "season", "diff", "p"]]
                                .to_dict(orient="records") for m in METRICS},
                     sig_cells_LN={m: comp[(comp.cmp == "LN-NEU") & (comp.metric == m) & comp.sig][["range", "season", "diff", "p"]]
                                   .to_dict(orient="records") for m in ("n_pos", "n_neg", "lp_mean", "x_mean", "nx_pos")},
                     sig_slopes={m: comp[(comp.cmp == "ONI-slope") & (comp.metric == m) & comp.sig][["range", "season", "diff", "lo", "hi", "p"]]
                                 .to_dict(orient="records") for m in ("n_pos", "n_neg", "amp_pos", "lp_mean", "x_mean", "nx_pos")})
    return an, tab, comp, lps, hps


def run_q2_era5(df, out, res, span=(1979, 2025)):
    an, mon, tr, tend = era5_monthly(df, span)
    n34 = load_n34().loc[f"{span[0]}":f"{span[1]}"]
    tt = np.arange(len(n34))
    n34 = n34 - np.polyval(np.polyfit(tt, n34.values, 1), tt)
    mon.to_csv(out / "q2_era5_monthly.csv")
    ser = (["aam_glob", tend, "T_glob", "mtn_glob", "fric_glob", "gwd_glob", "R_glob"]
           + (["dmdt_glob"] if tend != "dmdt_glob" else [])
           + (["aamm_glob", "dmmdt_glob"] if "aamm_glob" in mon else [])
           + [f"{t}_{b}" for t in ("mtn", "fric", "gwd") for b in BANDS[1:]] + [f"mtn5_{r}" for r in RANGES]
           + [f"fric_{r}" for r in RANGES])
    lr = lag_regress(mon, n34, ser, n=2000)
    lr.to_csv(out / "q2_era5_lagreg.csv", index=False)
    cc = [tend, "T_glob", "mtn_glob", "fric_glob", "gwd_glob", "R_glob"] + \
        [f"{t}_{b}" for t in ("mtn", "fric", "gwd") for b in BANDS[1:]] + [f"mtn5_{r}" for r in RANGES]
    contrib = joint_boot_contrib(mon, n34, cc)
    sig = bh_fdr([contrib[c]["p"] for c in cc])
    for c, s in zip(cc, sig):
        contrib[c]["fdr"] = bool(s)
    # closure diagnostics (anomalies) and absolute monthly means
    mclos = dict(corr=float(mon[[tend, "T_glob"]].corr().iloc[0, 1]),
                 slope=float(np.polyfit(mon["T_glob"], mon[tend], 1)[0]),
                 sd_tend=float(mon[tend].std()), sd_T=float(mon["T_glob"].std()), sd_R=float(mon["R_glob"].std()),
                 r_lag1_R=float(mon["R_glob"].autocorr(1)))
    raw = df.loc[f"{span[0]}":f"{span[1]}"]
    absT = {k: float(raw[k].mean()) for k in ("mtn_glob", "fric_glob", "gwd_glob")}
    mcol = "aam_glob"
    Mabs = raw[mcol] + (raw["aamm_glob"] if "aamm_glob" in raw else 0)
    absT["dMdt_mean"] = float((Mabs.iloc[-1] - Mabs.iloc[0]) / (len(Mabs) * HADLEY_DAY_TO_AAM))
    absT["sum"] = absT["mtn_glob"] + absT["fric_glob"] + absT["gwd_glob"]
    # El Nino onset composites: Apr-Dec integrated torque anomalies vs the AAM change
    en, sup, neu_strict = onset_years(load_oni(), (span[0], span[1] - 1))
    oni = load_oni()
    neu = [y for y in range(span[0], span[1]) if -0.5 < season_oni(oni, "DJF", y + 1) < 0.5]
    supers = [y for y in range(span[0], span[1]) if np.nanmax(
        [float(oni.get(pd.Timestamp(f"{y}-{m:02d}-01"), np.nan)) for m in (10, 11, 12)]
        + [season_oni(oni, "DJF", y + 1)]) >= 2.0]
    terms = ["mtn_glob", "fric_glob", "gwd_glob"] + [f"mtn5_{r}" for r in RANGES] + \
        [f"{t}_{b}" for t in ("mtn", "fric") for b in BANDS[1:]]
    mc = "aamt_glob" if "aamt_glob" in an else "aam_glob"
    ce = budget_composite(an, en, terms, mcol=mc)
    cn = budget_composite(an, neu, terms, mcol=mc)
    cs = budget_composite(an, supers, terms, mcol=mc)
    for c_ in (ce, cn, cs):
        c_["T"] = c_.mtn_glob + c_.fric_glob + c_.gwd_glob
        c_["resid"] = c_["dM"] - c_["T"]
        c_["other_mtn"] = c_.mtn_glob - sum(c_[f"mtn5_{r}"] for r in RANGES)
    pd.concat([ce.assign(group="EN"), cn.assign(group="NEU"), cs.assign(group="SUPER")]).to_csv(
        out / "q2_era5_composites.csv", index=False)
    comp = {}
    for c in ["dM", "T", "resid", "mtn_glob", "fric_glob", "gwd_glob", "other_mtn"] + [f"mtn5_{r}" for r in RANGES] + \
            [f"{t}_{b}" for t in ("mtn", "fric") for b in BANDS[1:]]:
        comp[c] = perm_diff(ce[c].values, cn[c].values) | dict(super_mean=float(cs[c].mean()) if len(cs) else np.nan)
    sigc = bh_fdr([comp[c]["p"] for c in comp])
    for c, s in zip(comp, sigc):
        comp[c]["fdr"] = bool(s)
    res["q2_era5"] = dict(tendency_column=tend, monthly_closure=mclos, absolute_means_hadley=absT,
                          contributions_lags_m6_m1=contrib,
                          aam_regression_change_lag0_minus_lagm6=float(
                              lr[(lr.series == "aam_glob") & (lr.lag == 0)].slope.iloc[0]
                              - lr[(lr.series == "aam_glob") & (lr.lag == -6)].slope.iloc[0]),
                          onset_years=en, super_years=supers, neutral_years=neu, composite_EN_minus_NEU=comp,
                          trends=tr)
    return an, mon, lr, contrib


def run_q2_cmip6(out, res, daily_dir=None):
    if not CMIP_CSV.exists():
        res["q2_cmip6"] = "missing"; return None
    cm = pd.read_csv(CMIP_CSV)
    ser = ["aam_glob", "dmdt_glob", "T_glob", "T_mf_glob", "mtn_glob", "fric_glob", "gwd_glob"] + \
        [f"{t}_{b}" for t in ("mtn", "fric", "gwd") for b in BANDS[1:]] + [f"mtn_{r}" for r in RANGES] + \
        [f"fric_{r}" for r in RANGES]
    rows, comps, clos = [], [], []
    for (mod, mem), g in cm.groupby(["model", "member"]):
        an, ab = cmip6_monthly_member(g)
        if an["aam_glob"].isna().all() or an["n34"].isna().all():
            continue
        has_gwd = "gwd_glob" in an and an["gwd_glob"].notna().sum() > 100 and an["gwd_glob"].std() > 0
        sl = lag_slopes(an, ser if has_gwd else [s for s in ser if not s.startswith("gwd") and s != "T_glob"])
        for (c, L), v in sl.items():
            rows.append(dict(model=mod, member=mem, series=c, lag=L, slope=v))
        an2 = an.copy()
        an2["R_mf"] = an2["dmdt_glob"] - an2["T_mf_glob"]
        if has_gwd:
            an2["R_all"] = an2["dmdt_glob"] - an2["T_glob"]
        ok = an2[["dmdt_glob", "T_mf_glob"]].dropna()
        clos.append(dict(model=mod, member=mem, has_gwd=has_gwd, corr_mf=float(ok.corr().iloc[0, 1]),
                         slope_mf=float(np.polyfit(ok.T_mf_glob, ok.dmdt_glob, 1)[0]),
                         sd_dmdt=float(ok.dmdt_glob.std()), sd_Tmf=float(ok.T_mf_glob.std()),
                         sd_Rmf=float((ok.dmdt_glob - ok.T_mf_glob).std()),
                         corr_all=float(an2[["dmdt_glob", "T_glob"]].dropna().corr().iloc[0, 1]) if has_gwd else np.nan,
                         sd_Rall=float(an2["R_all"].std()) if has_gwd else np.nan, **ab))
        en, neu = cmip6_onset_years(an["oni"])
        terms = [c for c in ["mtn_glob", "fric_glob", "gwd_glob", "T_mf_glob", "T_glob"] + [f"mtn_{r}" for r in RANGES]
                 + [f"fric_{b}" for b in BANDS[1:]] + [f"mtn_{b}" for b in BANDS[1:]] if c in an and (has_gwd or "gwd" not in c and c != "T_glob")]
        ce = cmip6_composite(an, None, en, terms)
        cn = cmip6_composite(an, None, neu, terms)
        if len(ce) >= 3 and len(cn) >= 3:
            for c in terms + ["dM"]:
                comps.append(dict(model=mod, member=mem, term=c, en=float(ce[c].mean()), neu=float(cn[c].mean()),
                                  diff=float(ce[c].mean() - cn[c].mean()), n_en=len(ce), n_neu=len(cn)))
    per = pd.DataFrame(rows)
    per.to_csv(out / "q2_cmip6_slopes_members.csv", index=False)
    pm = per.groupby(["model", "series", "lag"], as_index=False).slope.mean()       # members -> model
    pm.to_csv(out / "q2_cmip6_slopes_models.csv", index=False)
    summ = across_models(pm, group=("series", "lag"))
    summ.to_csv(out / "q2_cmip6_slopes_summary.csv", index=False)
    cl = pd.DataFrame(clos)
    cl.to_csv(out / "q2_cmip6_closure.csv", index=False)
    cp = pd.DataFrame(comps)
    cp.to_csv(out / "q2_cmip6_composites_members.csv", index=False)
    cpm = cp.groupby(["model", "term"], as_index=False)[["en", "neu", "diff"]].mean()
    csum = across_models(cpm, value="diff", group=("term",))
    csum.to_csv(out / "q2_cmip6_composites_summary.csv", index=False)
    # pre-peak integrated contributions (sum of lags -6..-1) per model
    days = 365.25 / 12
    pre = pm[(pm.lag >= -6) & (pm.lag <= -1)].groupby(["model", "series"], as_index=False).slope.sum()
    pre["contrib"] = pre.slope * days * HADLEY_DAY_TO_AAM
    pre.to_csv(out / "q2_cmip6_contrib_models.csv", index=False)
    psum = across_models(pre, value="contrib", group=("series",))
    psum.to_csv(out / "q2_cmip6_contrib_summary.csv", index=False)
    res["q2_cmip6"] = dict(n_models=int(pm.model.nunique()), models=sorted(pm.model.unique().tolist()),
                           n_members=int(per.groupby("model").member.nunique().sum()),
                           n_models_with_gwd=int(cl[cl.has_gwd].model.nunique()),
                           closure_median=cl.drop(columns=["model", "member", "has_gwd"]).median().to_dict(),
                           contrib_summary=psum.to_dict(orient="records"),
                           composite_summary=csum.to_dict(orient="records"))
    return pm, summ, cl, psum, csum


def run_q1_cmip6(out, res, daily_dir):
    if daily_dir is None or not Path(daily_dir).exists():
        res["q1_cmip6"] = "no daily directory"; return None
    cm = pd.read_csv(CMIP_CSV) if CMIP_CSV.exists() else None
    tabs = []
    for f in sorted(Path(daily_dir).glob("*_daily.csv")):
        dd = pd.read_csv(f)
        mod = dd.model.iloc[0]; mem = dd.member.iloc[0]
        nf = Path(daily_dir) / f"{mod}_daily_nino34.csv"
        if cm is not None and ((cm.model == mod) & (cm.member == mem)).any():
            g = cm[(cm.model == mod) & (cm.member == mem)]
        elif nf.exists():
            g = pd.read_csv(nf)
        else:
            continue
        n34 = pd.Series(g.nino34.values, index=pd.to_datetime(dict(year=g.year, month=g.month, day=1))).sort_index()
        t = cmip6_daily_events(dd, n34)
        t["model"] = mod
        tabs.append(t)
    if not tabs:
        res["q1_cmip6"] = "no daily files"; return None
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(out / "q1_cmip6_season_table.csv", index=False)
    rows = []
    for (mod, r, se), g in tab.groupby(["model", "key", "season"]):
        A = g[g.phase == "EN"]; B = g[g.phase == "NEU"]
        if len(A) < 3 or len(B) < 3:
            continue
        for mt in ("n_pos", "n_neg", "amp_pos", "lp_mean", "x_mean", "evc_pos", "nx_pos", "hp_sd"):
            rows.append(dict(model=mod, range=r, season=se, metric=mt, diff=float(A[mt].mean() - B[mt].mean()),
                             rel=float((A[mt].mean() - B[mt].mean()) / B[mt].mean()) if mt.startswith("n") else np.nan))
    pm = pd.DataFrame(rows)
    pm.to_csv(out / "q1_cmip6_diffs_models.csv", index=False)
    summ = across_models(pm, value="diff", group=("metric", "range", "season"))
    summ.to_csv(out / "q1_cmip6_summary.csv", index=False)
    res["q1_cmip6"] = dict(models=sorted(tab.model.unique().tolist()),
                           robust=summ[summ.robust].to_dict(orient="records") if len(summ) else [])
    return tab, summ


Q3_CASES = {  # range: (seasons, z500 box (lat0, lat1, lon0, lon1), global store?, jet latitude band)
    "HIM": (("DJF", "MAM"), (10, 72, 20, 160), False, (25, 40)),
    "ROC": (("DJF",), (12, 75, 170, 320), False, (25, 40)),
    "AND": (("DJF", "JJA"), (-75, 10, 220, 340), True, (-40, -25)),
}


def u200_zonal(glob):
    """Daily zonal-mean u200 (time, lat) from the local store: NH 1959-2023 or global 1991-2020."""
    import xarray as xr
    base = STORE / ("wb2_1p5_daily_global" if glob else "wb2_1p5_daily") / "u200"
    yrs = range(1991, 2021) if glob else range(1979, 2024)
    parts = []
    for y in yrs:
        f = base / f"u200_{y}.nc"
        if f.exists():
            d = xr.open_dataset(f)
            v = d[list(d.data_vars)[0]].transpose("time", "latitude", "longitude").mean("longitude")
            parts.append(v.load())
    return xr.concat(parts, "time").sortby("latitude")


def run_q3(df, an_r, hps, oni, out, res, span=(1979, 2025)):
    """Lead-lag of each range's high-pass torque with AAM / its tendency / the subtropical jet, and
    z500 precursor composites + precursor frequency by ENSO phase."""
    acols = ["aam_glob", "aam_b0030n", "aam_b3060n", "aam_b3000s", "aam_b6030s"] + \
        (["aamm_glob"] if "aamm_glob" in df else [])
    aa, _ = anomalies(df, acols, span)
    M = aa["aam_glob"] + (aa["aamm_glob"] if "aamm_glob" in aa else 0.0)
    dMdt = (M.shift(-1) - M.shift(1)) / (2 * HADLEY_DAY_TO_AAM)
    hpz = lambda s: s - pd.Series(lanczos_lowpass(s.values), index=s.index)       # noqa: E731
    Y = {"dMdt": hpz(dMdt), "M": hpz(M)}
    for b in ("b0030n", "b3060n", "b3000s", "b6030s"):
        Y[f"M_{b}"] = hpz(aa[f"aam_{b}"])
    jets = {}
    for glob in (False, True):
        u = u200_zonal(glob)
        jets[glob] = u
    ll = []
    s_all, sy_all = season_of(an_r.index)
    for r, (seas, box, glob, jb) in Q3_CASES.items():
        x = hps[r]
        u = jets[glob]
        uj = u.sel(latitude=slice(min(jb), max(jb))).mean("latitude").to_series()
        uj.index = pd.DatetimeIndex(uj.index).normalize()
        uja, _ = harmonic_anom(uj.reindex(x.index))
        Yr = dict(Y) | {"u200jet": hpz(uja.interpolate(limit=3))}
        for se in seas:
            mask = (s_all == se) & (sy_all >= span[0]) & (sy_all <= span[1]) & np.isfinite(x.values)
            for k, y in Yr.items():
                lc = lagcorr(x.values, y.reindex(x.index).values, mask)
                lc["range"] = r; lc["season"] = se; lc["series"] = k
                ll.append(lc)
    ll = pd.concat(ll, ignore_index=True)
    ll.to_csv(out / "q3_leadlag.csv", index=False)
    peaks = ll.loc[ll.groupby(["range", "season", "series"]).r.apply(lambda s: s.abs().idxmax())]
    res["q3_leadlag_peaks"] = peaks[["range", "season", "series", "lag", "r", "p", "sig"]].to_dict(orient="records")
    # ---- z500 precursor composites and frequency
    comp_out, freq = {}, []
    for r, (seas, box, glob, jb) in Q3_CASES.items():
        yrs = range(1991, 2021) if glob else range(span[0], span[1] + 1)
        z = store_field("z500", yrs, box, glob=glob)
        zh, za = field_anom_hp(z)
        zt = pd.DatetimeIndex(z.time.values).normalize()
        Z = zh.values
        lat = z.latitude.values
        wlat = np.sqrt(np.cos(np.deg2rad(lat)))[:, None]
        x = hps[r].reindex(zt)
        s, sy = season_of(zt)
        for se in seas:
            msk = (s == se) & np.isfinite(x.values)
            sig = float(np.nanstd(x.values[msk]))
            pk = find_peaks(np.where(msk, x.values, np.nan), sig)
            pk = pk[(pk >= 3)]
            ph = np.array([phase_of(season_oni(oni, se, sy[i])) for i in pk])
            pre = np.stack([np.nanmean(Z[i - 3:i], 0) for i in pk])            # lags -3..-1 mean
            mean_all, p_all = composite_test(pre, np.arange(len(pk)))
            sig_all = bh_fdr(p_all.ravel()).reshape(p_all.shape)
            ie = np.where(ph == "EN")[0]; inn = np.where(ph == "NEU")[0]; il = np.where(ph == "LN")[0]
            m_en, p_en = composite_test(pre, ie); m_ne, p_ne = composite_test(pre, inn)
            d_en, p_d = composite_test(pre, ie, inn)
            sd_en = bh_fdr(p_en.ravel()).reshape(p_en.shape)
            sd_ne = bh_fdr(p_ne.ravel()).reshape(p_ne.shape)
            sd_d = bh_fdr(p_d.ravel()).reshape(p_d.shape)
            comp_out[f"{r}_{se}"] = dict(lat=lat, lon=z.longitude.values, all=mean_all, sig_all=sig_all,
                                         en=m_en, sig_en=sd_en, neu=m_ne, sig_neu=sd_ne, diff=d_en, sig_diff=sd_d,
                                         n_all=len(pk), n_en=len(ie), n_neu=len(inn), n_ln=len(il))
            # precursor index: projection of daily hp z500 on the all-event lag -3..-1 pattern
            P = np.where(sig_all, mean_all, 0.0) * wlat
            if not sig_all.any():
                P = mean_all * wlat
            I = np.tensordot(Z * wlat[None], P, axes=([1, 2], [0, 1])) / (P ** 2).sum()
            I = pd.Series(I, index=zt)
            sI = float(I[msk].std())
            ipk = find_peaks(np.where(msk, I.values, np.nan), sI)
            evset = set(pk.tolist())
            rows = []
            for y in sorted(set(sy[msk])):
                dmask = msk & (sy == y)
                nd = int(dmask.sum())
                if nd < 80:
                    continue
                ip = ipk[dmask[ipk]]
                hits = sum(any((j + k) in evset for k in range(0, 5)) for j in ip)
                rows.append(dict(range=r, season=se, sy=y, phase=phase_of(season_oni(oni, se, y)),
                                 n_prec=len(ip), hits=hits, hit_rate=hits / len(ip) if len(ip) else np.nan,
                                 n_events=int(np.isin(pk, np.where(dmask)[0]).sum())))
            fr = pd.DataFrame(rows)
            freq.append(fr)
    freq = pd.concat(freq, ignore_index=True)
    freq.to_csv(out / "q3_precursor_freq.csv", index=False)
    ftests = []
    for (r, se), g in freq.groupby(["range", "season"]):
        for mt in ("n_prec", "hit_rate", "n_events"):
            ftests.append(dict(range=r, season=se, metric=mt, **perm_diff(g[g.phase == "EN"][mt].values,
                                                                           g[g.phase == "NEU"][mt].values)))
    ftests = pd.DataFrame(ftests)
    ftests["sig"] = False
    for mt, g in ftests.groupby("metric"):
        ftests.loc[g.index, "sig"] = bh_fdr(g.p.values)
    ftests.to_csv(out / "q3_precursor_tests.csv", index=False)
    np.savez_compressed(out / "q3_z500_composites.npz",
                        **{f"{k}__{kk}": vv for k, v in comp_out.items() for kk, vv in v.items()})
    res["q3_precursor_tests"] = ftests.to_dict(orient="records")
    res["q3_composite_counts"] = {k: dict(n_all=v["n_all"], n_en=v["n_en"], n_neu=v["n_neu"], n_ln=v["n_ln"],
                                          frac_sig_all=float(v["sig_all"].mean()), frac_sig_diff=float(v["sig_diff"].mean()))
                                  for k, v in comp_out.items()}
    return ll, comp_out, ftests


def run_slp_maps(oni, out, res, seasons=("DJF", "MAM", "JJA", "SON")):
    """Seasonal-mean SLP, El Nino minus neutral season-years (ERA5 1991-2020, 1.5 deg, local store):
    the standing pressure pattern straddling the ranges. Welch t-test per cell, BH-FDR 10%."""
    import xarray as xr
    z = store_field("slp", range(1991, 2021), (-90, 90, 0, 359.9), glob=True)
    t = pd.DatetimeIndex(z.time.values)
    s, sy = season_of(t)
    maps = {}
    for se in seasons:
        yrs = [y for y in range(1991, 2021) if ((s == se) & (sy == y)).sum() >= 80]
        F = np.stack([z.values[(s == se) & (sy == y)].mean(0) for y in yrs])
        ph = np.array([phase_of(season_oni(oni, se, y)) for y in yrs])
        o = np.array([season_oni(oni, se, y) for y in yrs])
        d, p = composite_test(F, np.where(ph == "EN")[0], np.where(ph == "NEU")[0])
        sig = bh_fdr(p.ravel()).reshape(p.shape)
        # regression on the ONI over all season-years (more power than 11 vs 8 composites)
        from scipy import stats
        oc = o - o.mean()
        Fc = F - F.mean(0)
        b = np.tensordot(oc, Fc, axes=(0, 0)) / (oc ** 2).sum()
        r = np.tensordot(oc, Fc, axes=(0, 0)) / np.sqrt((oc ** 2).sum() * (Fc ** 2).sum(0))
        n = len(o)
        tt = r * np.sqrt((n - 2) / np.clip(1 - r * r, 1e-9, None))
        pr = 2 * stats.t.sf(np.abs(tt), n - 2)
        sr = bh_fdr(pr.ravel()).reshape(pr.shape)
        maps[se] = dict(diff=d, sig=sig, slope=b, sig_slope=sr, n_en=int((ph == "EN").sum()),
                        n_neu=int((ph == "NEU").sum()), n_years=n)
    np.savez_compressed(out / "slp_enso_maps.npz", lat=z.latitude.values, lon=z.longitude.values,
                        **{f"{se}__{k}": v for se, m in maps.items() for k, v in m.items()})
    res["slp_maps"] = {se: dict(n_en=m["n_en"], n_neu=m["n_neu"], n_years=m["n_years"], frac_sig=float(m["sig"].mean()),
                                frac_sig_slope=float(m["sig_slope"].mean())) for se, m in maps.items()}
    return maps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmip6-daily", default=None, help="dir with the cmip6_daily_all artifact")
    ap.add_argument("--only", default="q1,q2,q2c,q1c,q3")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    res = {}
    df = load_era5()
    res["era5_rows"] = [str(df.index[0].date()), str(df.index[-1].date()), int(len(df)),
                        int(df.attrs.get("n_interpolated_days", 0))]
    oni = load_oni()
    an_r = hps = None
    if "q1" in only or "q3" in only:
        an_r, tab, comp, lps, hps = run_q1(df, oni, out, res)
    if "q2" in only:
        run_q2_era5(df, out, res)
    if "q2c" in only:
        run_q2_cmip6(out, res)
    if "q1c" in only:
        run_q1_cmip6(out, res, a.cmip6_daily)
    if "q3" in only:
        run_q3(df, an_r, hps, oni, out, res)
    if "slp" in only:
        run_slp_maps(oni, out, res)
    prev = out / "results.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old.update(res)
    prev.write_text(json.dumps(old, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
