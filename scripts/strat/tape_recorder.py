"""Tropical water-vapour "tape recorder": Aura MLS record + GEOS FP for the newest weeks (stratosphere page).

Air enters the stratosphere through the cold tropical tropopause, which freeze-dries it to the saturation mixing ratio of
the cold point. The cold point is coldest in boreal winter and warmest in boreal summer, so the air that enters carries an
annual cycle of water vapour (about 3 ppmv in February, 4.5-5 in August) up with it, like a signal written on a slowly
rising tape (Mote et al. 1996). The slope of the moist and dry bands in a height-time section is the ascent speed of
the Brewer-Dobson circulation's tropical branch; the value at 100-70 hPa is the entry value the cold point has just set.

SOURCES
- Aura MLS v6 (ML2H2O 006, GES DISC), 10S-10N daily mean profiles 2004-08 on, reduced by fetch_mls_h2o.py (Earthdata
  login; laptop). The gold standard: MLS measures the water directly. Since May 2024 its 190-GHz radiometer is duty-cycled
  (about a week a month, about four days a month from early 2026), so the recent MLS record is a few days per month.
  Committed as reference/tape_mls.npz by `build`. If EARTHDATA_USERNAME/_PASSWORD are set, `live` also appends new MLS
  days (not the case in Actions today: the repository holds no Earthdata secret).
- GEOS FP (NASA GMAO forward processing, NCCS OPeNDAP inst3_3d_asm_Np, 12Z analysis each day, 2017-12 on): qv on its
  pressure levels 150-1 hPa and T at 150-50 hPa, 10S-10N (1 x 2.5 deg subsample), cos-weighted. GEOS FP does NOT
  assimilate stratospheric water vapour: its stratospheric qv is the model's own, freeze-dried by the ANALYSED temperatures
  (so the entry value is constrained, the air above is advected by analysed winds, and anything injected from outside -
  the January 2022 Hunga eruption put ~150 Tg of water into the stratosphere - is absent). `build` measures this against MLS
  and the page uses GEOS FP only after the last MLS day, adjusted to MLS by the per-level offset of the last 365 days.

    python scripts/strat/tape_recorder.py fetch-fp 2017-12-01 2026-09-27 [step]  # laptop seed, ~5-8 s a day, resumable
    python scripts/strat/tape_recorder.py build                          # laptop: reference files + validation numbers
    python scripts/strat/tape_recorder.py live                           # Actions: new GEOS FP days + figures + json
"""
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data" / "tape"
REF = HERE / "reference"
FPDAYS = DATA / "fp_days"
MLSDAYS = DATA / "mls_days"
NCCS = "https://opendap.nccs.nasa.gov/dods"
ASSIM_NP = f"{NCCS}/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Np"
FP_LEVS = np.array([150, 100, 70, 50, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1], float)
T_LEVS = np.array([150, 100, 70, 50], float)
PPMV = 28.9644 / 18.01528 * 1e6          # kg/kg -> ppmv (mixing ratio; qv/(1-qv) ~ qv at stratospheric values)
H_SCALE = 7.0                            # km: log-pressure height z = -H ln(p/1000), the coordinate of the TEM w*
CLIM_YEARS = (2005, 2021)                # MLS climatology: whole years before the Hunga eruption (15 Jan 2022)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def zlp(p):
    return -H_SCALE * np.log(np.asarray(p, float) / 1000.0)


# ----------------------------------------------------------------------------------------------------------- GEOS FP
def open_fp(recent=False):
    import xarray as xr
    for i in range(4):
        try:
            ds = xr.open_dataset(ASSIM_NP)
            idx = ds.indexes["time"]
            if not isinstance(idx, pd.DatetimeIndex):          # the GrADS server's "days since 1-1-1" can decode as cftime
                ds = ds.assign_coords(time=idx.to_datetimeindex(unsafe=True))
            tt = pd.DatetimeIndex(ds.time.values)
            if tt[0].year != 2017 or tt[-1].year > 2035 or not tt.is_monotonic_increasing:
                raise RuntimeError(f"garbled time axis {tt[0]}..{tt[-1]}")  # seen while the server rewrites the axis
            if recent:
                last = pd.Timestamp(ds.time.values[-1])
                if last < pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=3):
                    raise RuntimeError(f"time axis ends {last}")
            return ds
        except Exception as e:                                                        # noqa: BLE001
            log(f"open failed ({str(e)[:80]}), retry {i + 1}")
            time.sleep(30 * (i + 1))
    raise RuntimeError("GEOS FP not reachable")


def fp_day(ds, d, _cache={}):
    """One GEOS FP 12Z analysis -> (qv ppmv on FP_LEVS, T K on T_LEVS), 10S-10N cos-weighted zonal-band means."""
    if "idx" not in _cache:
        lat, lev = ds.lat.values, ds.lev.values
        i0, i1 = np.searchsorted(lat, -10.0), np.searchsorted(lat, 10.0, side="right")
        kq = [int(np.argmin(np.abs(lev - L))) for L in FP_LEVS]
        kt = [int(np.argmin(np.abs(lev - L))) for L in T_LEVS]
        assert np.allclose(lev[kq], FP_LEVS) and np.allclose(lev[kt], T_LEVS), "unexpected GEOS FP levels"
        _cache["idx"] = (i0, i1, min(kq), max(kq), min(kt), max(kt), lev)
    i0, i1, q0, q1, t0, t1, lev = _cache["idx"]
    t = pd.Timestamp(d) + pd.Timedelta(hours=12)
    last = None
    for attempt in range(4):
        try:
            q = ds["qv"].sel(time=t).isel(lev=slice(q0, q1 + 1), lat=slice(i0, i1, 4), lon=slice(None, None, 8)).load()
            T = ds["t"].sel(time=t).isel(lev=slice(t0, t1 + 1), lat=slice(i0, i1, 4), lon=slice(None, None, 8)).load()
            w = np.cos(np.deg2rad(q.lat.values))
            qa = np.where(np.abs(q.values) > 1e10, np.nan, q.values)
            ta = np.where(np.abs(T.values) > 1e10, np.nan, T.values)
            qz = (np.nanmean(qa, axis=-1) * w).sum(-1) / w.sum() * PPMV
            tz = (np.nanmean(ta, axis=-1) * w).sum(-1) / w.sum()
            lq, lt = q.lev.values, T.lev.values
            qz = np.array([qz[int(np.argmin(np.abs(lq - L)))] for L in FP_LEVS])
            tz = np.array([tz[int(np.argmin(np.abs(lt - L)))] for L in T_LEVS])
            # physical screen: stratospheric water 1-12 ppmv above 100 hPa, 150 hPa below 400; T 180-240 K
            if not (np.isfinite(qz).all() and np.isfinite(tz).all() and (qz[1:] > 1).all() and (qz[1:] < 12).all()
                    and qz[0] < 400 and (tz > 180).all() and (tz < 240).all()):
                raise ValueError(f"implausible profile q={np.round(qz, 2)} T={np.round(tz, 1)}")
            return qz, tz
        except Exception as e:                                                        # noqa: BLE001
            last = e
            log(f"  {d} retry {attempt + 1}: {str(e)[:100]}")
            time.sleep(20 * (attempt + 1))
    raise last


def fetch_fp(d0, d1, budget_min=None, step=1):
    FPDAYS.mkdir(parents=True, exist_ok=True)
    ds = open_fp()
    newest = pd.Timestamp(ds.time.values[-1])
    t_start, n = time.time(), 0
    d = d0
    while d <= d1:
        f = FPDAYS / f"{d:%Y%m%d}.npz"
        if not f.exists() and pd.Timestamp(d) + pd.Timedelta(hours=12) <= newest:
            if budget_min and (time.time() - t_start) / 60 > budget_min:
                log("time budget reached; the rest on the next run"); break
            try:
                q, T = fp_day(ds, d)
                np.savez(f, q=q.astype(np.float32), t=T.astype(np.float32))
                n += 1
                if n % 50 == 0:
                    log(f"  {d} ok ({n} new)")
            except Exception as e:                                                    # noqa: BLE001
                log(f"  {d} FAILED: {str(e)[:100]}")
        d += timedelta(days=step)
    log(f"GEOS FP: {n} new days")
    return n


# ------------------------------------------------------------------------------------------------ records on disk
MLS_REF = REF / "tape_mls.npz"
FP_SEED = REF / "tape_geosfp_seed.npz"
P_MLS_KEEP = 150.0                        # MLS levels kept in the reference: 147 .. 1 hPa


def build_refs():
    """Laptop: daily MLS and GEOS FP files -> the two committed reference files."""
    rows, n, dates = [], [], []
    p = None
    for f in sorted(MLSDAYS.glob("*.npz")):
        z = np.load(f)
        if p is None:
            p = z["p"].astype(float)
        dates.append(int(f.stem)); rows.append(z["h2o"]); n.append(int(z["n"]))
    rows, n, dates = np.array(rows), np.array(n), np.array(dates)
    k = p <= P_MLS_KEEP
    good = n > 0
    np.savez_compressed(MLS_REF, date=dates[good].astype(np.int32), p=p[k].astype(np.float32),
                        h2o=rows[good][:, k].astype(np.float32), n=n[good].astype(np.int16))
    log(f"MLS reference: {good.sum()} good days of {len(n)} files ({dates[good][0]}..{dates[good][-1]}), "
        f"{k.sum()} levels -> {MLS_REF.stat().st_size / 1e3:.0f} kB")
    fd, fq, ft = [], [], []
    for f in sorted(FPDAYS.glob("*.npz")):
        z = np.load(f)
        fd.append(int(f.stem)); fq.append(z["q"]); ft.append(z["t"])
    np.savez_compressed(FP_SEED, date=np.array(fd, np.int32), q=np.array(fq, np.float32), t=np.array(ft, np.float32),
                        p_q=FP_LEVS.astype(np.float32), p_t=T_LEVS.astype(np.float32))
    log(f"GEOS FP seed: {len(fd)} days ({fd[0]}..{fd[-1]}) -> {FP_SEED.stat().st_size / 1e3:.0f} kB")


def _dates(ints):
    return pd.to_datetime(pd.Series(ints).astype(str), format="%Y%m%d").values


def load_mls(tail=None):
    z = np.load(MLS_REF)
    df = pd.DataFrame(z["h2o"].astype(float), index=pd.DatetimeIndex(_dates(z["date"])), columns=z["p"].astype(float))
    if tail:
        extra = pd.DataFrame({pd.Timestamp(d): v for d, v in tail.get("days", {}).items()}).T
        if len(extra):
            extra.columns = np.array(tail["p"], float)
            df = pd.concat([df, extra[~extra.index.isin(df.index)]]).sort_index()
    return df


def load_fp(tail=None):
    z = np.load(FP_SEED)
    idx = pd.DatetimeIndex(_dates(z["date"]))
    q = pd.DataFrame(z["q"].astype(float), index=idx, columns=z["p_q"].astype(float))
    t = pd.DataFrame(z["t"].astype(float), index=idx, columns=z["p_t"].astype(float))
    if tail and tail.get("days"):
        d = pd.DatetimeIndex([pd.Timestamp(x) for x in tail["days"]])
        tq = pd.DataFrame([v["q"] for v in tail["days"].values()], index=d, columns=q.columns)
        tt = pd.DataFrame([v["t"] for v in tail["days"].values()], index=d, columns=t.columns)
        q = pd.concat([q, tq[~tq.index.isin(q.index)]]).sort_index()
        t = pd.concat([t, tt[~tt.index.isin(t.index)]]).sort_index()
    return q, t


# --------------------------------------------------------------------------------------------------------- analysis
W1 = 2 * np.pi / 365.25


def harm_design(tdays, nh=3):
    cols = [np.ones_like(tdays, dtype=float)]
    for h in range(1, nh + 1):
        cols += [np.cos(h * W1 * tdays), np.sin(h * W1 * tdays)]
    return np.column_stack(cols)


def clim_fit(df, y0, y1, nh=3):
    """Day-of-year climatology per column: harmonic fit on the daily values of years y0..y1 -> function(dates)."""
    sub = df[(df.index.year >= y0) & (df.index.year <= y1)].dropna(how="all")
    doy = sub.index.dayofyear.values.astype(float)
    coefs = {}
    for c in df.columns:
        ok = np.isfinite(sub[c].values)
        coefs[c] = np.linalg.lstsq(harm_design(doy[ok], nh), sub[c].values[ok], rcond=None)[0]

    def at(dates):
        d = pd.DatetimeIndex(dates).dayofyear.values.astype(float)
        X = harm_design(d, nh)
        return pd.DataFrame({c: X @ coefs[c] for c in df.columns}, index=pd.DatetimeIndex(dates))
    at.mean = {c: float(coefs[c][0]) for c in df.columns}
    return at


def interp_levels(df, p_to):
    """Interpolate rows of df (columns = pressure) to p_to, linear in log p and log mixing ratio (the MLS basis)."""
    p = np.array(df.columns, float)
    o = np.argsort(np.log(p))
    lp, lpt = np.log(p)[o], np.log(np.asarray(p_to, float))
    X = np.log(np.clip(df.values[:, o], 1e-3, None))
    out = np.array([np.interp(lpt, lp, r, left=np.nan, right=np.nan) if np.isfinite(r).all() else np.full(len(lpt), np.nan)
                    for r in X])
    return pd.DataFrame(np.exp(out), index=df.index, columns=np.asarray(p_to, float))


def phase_times(df, levels, min_days):
    """Annual-harmonic phase: the day (within the window, unwrapped upward) of the moist maximum at each level, and its
    amplitude and standard error, from y = a + b cos wt + c sin wt (+ trend). Returns (tmax_days, amp, amp_se, n)."""
    if len(df) < max(min_days, 8):
        return None
    t = (df.index - df.index[0]).days.values.astype(float)
    tm, amp, ase = [], [], []
    for L in levels:
        y = df[L].values; ok = np.isfinite(y)
        if ok.sum() < min_days:
            return None
        X = np.column_stack([np.ones(ok.sum()), t[ok] / 365.25, np.cos(W1 * t[ok]), np.sin(W1 * t[ok])])
        b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        e = y[ok] - X @ b
        cov = (e @ e) / max(ok.sum() - 4, 1) * np.linalg.inv(X.T @ X)
        A = float(np.hypot(b[2], b[3]))
        tm.append(np.arctan2(b[3], b[2]) / W1)
        amp.append(A); ase.append(float(np.sqrt((cov[2, 2] + cov[3, 3]) / 2)))
    tm = np.array(tm)
    for i in range(1, len(tm)):                      # unwrap upward: each level lags the one below by < half a year
        while tm[i] < tm[i - 1] - 30:
            tm[i] += 365.25
        while tm[i] > tm[i - 1] + 365.25 - 30:
            tm[i] -= 365.25
    return tm, np.array(amp), np.array(ase), int(np.isfinite(df[levels[0]].values).sum())


def ascent(tm, levels, ok=None):
    """Ascent speed (mm/s) at each interior level from a local linear fit of phase time against log-pressure height over
    the level and its two neighbours; the lowest and highest levels use one-sided pairs. Windows touching a level whose
    annual signal is not resolved (ok False) are left out."""
    z = zlp(levels) * 1e6                                  # mm
    w = np.full(len(levels), np.nan)
    ok = np.ones(len(levels), bool) if ok is None else np.asarray(ok, bool)
    for i in range(len(levels)):
        j = slice(max(i - 1, 0), min(i + 2, len(levels)))
        if (j.stop - j.start) < 2 or not ok[j].all():
            continue
        s = np.polyfit(z[j], tm[j] * 86400.0, 1)[0]         # s per mm
        w[i] = 1.0 / s if s > 0 else np.nan
    return w


LAYERS = {"low": (82.54, 68.13, 56.23, 46.42), "high": (46.42, 38.31, 31.62, 26.1)}


def speeds(tm, levels, ok):
    """Profile (centred 3-level fits) and layer speeds (a straight-line fit of phase time against height over the
    layer's levels), mm/s; NaN where the annual signal is not resolved."""
    z = zlp(levels) * 1e6
    out = {"prof": ascent(tm, levels, ok)}
    for k, lv in LAYERS.items():
        if not all(L in levels for L in lv):
            out[k] = np.nan; continue
        idx = [levels.index(L) for L in lv]
        if not np.asarray(ok)[idx].all():
            out[k] = np.nan; continue
        sl = np.polyfit(z[idx], np.asarray(tm)[idx] * 86400.0, 1)[0]
        out[k] = 1.0 / sl if sl > 0 else np.nan
    return out


def yearly_ascent(df, levels, years, min_days=60):
    out = {}
    for y in years:
        sub = df[(df.index >= f"{y}-01-01") & (df.index <= f"{y}-12-31")].dropna(how="any", subset=levels)
        if sub.index.month.nunique() < 10:               # the annual phase needs the whole year sampled
            continue
        r = phase_times(sub, levels, min_days)
        if r is None:
            continue
        tm, amp, ase, n = r
        ok = np.cumprod(amp >= 3 * ase).astype(bool)     # resolved from the bottom up; nothing above a failure
        if not ok[:4].all():
            continue
        out[y] = speeds(tm, levels, ok)
    return out


def window_ascent(df, levels, t_end, min_days=40, boot=300, seed=1):
    """Speeds over the 365 days to t_end, with standard errors from a bootstrap over calendar months (days within a
    month are not independent: since 2024 each month is one burst of 4-8 MLS days)."""
    sub = df[(df.index > t_end - pd.Timedelta(days=365)) & (df.index <= t_end)].dropna(how="any", subset=levels)
    if sub.index.month.nunique() < 10:
        return None, None, 0
    r = phase_times(sub, levels, min_days)
    if r is None:
        return None, None, 0
    okl = np.cumprod(r[1] >= 3 * r[2]).astype(bool)
    w = speeds(r[0], levels, okl)
    rng = np.random.default_rng(seed)
    bs = []
    per = sub.index.to_period("M")
    months = per.unique()
    for _ in range(boot):
        pick = months[rng.integers(0, len(months), len(months))]
        s = pd.concat([sub[per == m] for m in pick]).sort_index()
        rr = phase_times(s, levels, min(min_days, len(s)))
        if rr is not None:
            bs.append(speeds(rr[0], levels, okl))
    def rsd(x, axis=None):                     # robust spread: half the 16-84 % range (a few resamples lose a month
        q = np.nanpercentile(np.asarray(x, float), [16, 84], axis=axis)   # and unwrap badly; they would dominate a sd)
        return (q[1] - q[0]) / 2
    se = {"prof": rsd(np.array([b["prof"] for b in bs]), axis=0)}
    for k in LAYERS:
        se[k] = float(rsd([b[k] for b in bs]))
    return w, se, len(sub)


def bins10(start, end):
    """Dekads: bins starting on the 1st, 11th and 21st of each month."""
    s = []
    for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        for d in (1, 11, 21):
            b0 = m.replace(day=d)
            b1 = (m.replace(day=d + 10) if d < 21 else m + pd.offsets.MonthBegin(1))
            if b1 > pd.Timestamp(start) and b0 <= pd.Timestamp(end):
                s.append((b0, b1))
    return s


def pfmt(p):
    return "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")


def norm_p(z):
    from scipy import stats
    return float(2 * stats.norm.sf(abs(z)))


SEC_LEVS = [100.0, 82.54, 68.13, 56.23, 46.42, 38.31, 31.62, 26.1, 21.54, 17.78, 14.68, 12.12, 10.0]
REC_LEVS = SEC_LEVS + [8.25, 6.81, 5.62, 4.64, 3.83, 3.16]
ASC_LEVS = [100.0, 82.54, 68.13, 56.23, 46.42, 38.31, 31.62, 26.1, 21.54]   # annual signal resolved to ~25 hPa
PROF_SHOW = slice(2, 8)          # profile drawn 68-26 hPa: MLS's ~3 km resolution smears the 100 hPa phase (and so the
                                 # centred fit at 83 hPa); 22 hPa is the edge of a resolved annual signal
FP_ASC = [100.0, 70.0, 50.0, 40.0, 30.0, 20.0]


def snap(df, levs):
    """Columns nearest to the requested MLS levels (float keys differ in the last digits)."""
    cols = np.array(df.columns, float)
    return df[[cols[int(np.argmin(np.abs(np.log(cols / L))))] for L in levs]].set_axis(levs, axis=1)


OFFSET_DAYS = 60


def fp_adjustment(mls, fpq):
    """GEOS FP -> MLS: per GEOS FP level, the mean MLS - GEOS FP difference on the common days of the last OFFSET_DAYS
    before the last MLS day (widened to 120 and 365 days if fewer than 3). Chosen by hindcast (every month with >= 3 MLS
    days, 2019 on, offset from the days before it): a 60-day offset beat a 120- or 365-day one and a 365-day level plus a
    seasonal shape at 100, 70, 50 and 30 hPa. The hindcast RMSE of a monthly mean is returned per level and is the error
    carried into every test that uses GEOS FP. Returns (offset function of dates, validation dict, rmse dict)."""
    levs = [L for L in fpq.columns if L <= 100]
    m_on_fp = interp_levels(mls, levs)
    common = m_on_fp.index.intersection(fpq.index)
    d = (m_on_fp.loc[common] - fpq.loc[common, levs]).dropna()

    def offset_before(t0):
        for nd in (OFFSET_DAYS, 120, 365):
            w = d[(d.index <= t0) & (d.index > t0 - pd.Timedelta(days=nd))]
            if len(w) >= 3:
                return w.mean(), nd, len(w)
        return d[d.index <= t0].tail(30).mean(), 0, 0
    cur, nd_used, n_used = offset_before(d.index.max())

    def off(dates):
        return pd.DataFrame({L: np.full(len(dates), float(cur[L])) for L in levs}, index=pd.DatetimeIndex(dates))
    # hindcast of monthly means
    err = {L: [] for L in levs}
    mon = m_on_fp.dropna().index.to_period("M")
    for mo in pd.period_range("2019-01", d.index.max().to_period("M"), freq="M"):
        days = m_on_fp.dropna().index[mon == mo].intersection(fpq.index)
        if len(days) < 3:
            continue
        o, _, n = offset_before(mo.start_time - pd.Timedelta(days=1))
        if n < 3:
            continue
        for L in levs:
            err[L].append(float(fpq.loc[days, L].mean() + o[L] - m_on_fp.loc[days, L].mean()))
    rmse = {L: float(np.sqrt(np.mean(np.square(e)))) if e else np.nan for L, e in err.items()}
    # validation: bias by era, correlation of deseasonalised monthly anomalies by era
    cm = clim_fit(m_on_fp.loc[m_on_fp.index.year.isin(range(2018, 2022))], 2018, 2021, 2)
    cf = clim_fit(fpq[levs].loc[fpq.index.year.isin(range(2018, 2022))], 2018, 2021, 2)
    am = (m_on_fp.loc[common] - cm(common)).resample("MS").mean()
    af = (fpq.loc[common, levs] - cf(common)).resample("MS").mean()
    val = {}
    for L in levs:
        v = {}
        for nm, (a, b) in {"2018-2021": (2018, 2021), "2022-2023": (2022, 2023), "2024-now": (2024, 2100)}.items():
            dd = d[(d.index.year >= a) & (d.index.year <= b)][L]
            mm = m_on_fp.loc[dd.index, L]
            e = (am.index.year >= a) & (am.index.year <= b)
            ok = e & np.isfinite(am[L].values) & np.isfinite(af[L].values)
            r = float(np.corrcoef(am[L].values[ok], af[L].values[ok])[0, 1]) if ok.sum() > 6 else None
            v[nm] = {"n_days": int(len(dd)), "fp_minus_mls_ppmv": round(float(-dd.mean()), 3) if len(dd) else None,
                     "fp_minus_mls_pct": round(float(-100 * dd.mean() / mm.mean()), 1) if len(dd) else None,
                     "r_monthly_anom": None if r is None else round(r, 2),
                     "mls_anom_mean": round(float(np.nanmean(am[L].values[ok])), 3) if ok.any() else None,
                     "fp_anom_mean": round(float(np.nanmean(af[L].values[ok])), 3) if ok.any() else None}
        val[f"{L:g}"] = v
    val["offset_ppmv"] = {f"{L:g}": round(float(cur[L]), 3) for L in levs}
    val["offset_window_days"] = nd_used; val["offset_n_days"] = n_used
    val["hindcast_rmse_monthly_ppmv"] = {f"{L:g}": round(rmse[L], 3) for L in levs}
    val["hindcast_n_months"] = len(err[levs[0]])
    val["n_common_days"] = int(len(d)); val["last_common_day"] = f"{d.index.max():%Y-%m-%d}"
    return off, val, rmse


def w_star_refs():
    """w* (mm/s) to compare with: MERRA-2 downward-control climatology (bdc_dc release files, annual mean of daily w*
    between the turnaround latitudes) and GEOS FP's last 30 days (mbudget.json: direct omega* and downward control; bdc_dc.json
    downward control on the MERRA-2 scale)."""
    A_E, G = 6.371e6, 9.80665
    out = {}
    cz = HERE / "data" / "bdc_dc" / "clim_merra2.npz"
    if cz.exists():
        z = np.load(cz, allow_pickle=True)
        turn = z["turn"]                                 # (366, (100, 70, 50), (S, N))
        for i, L in enumerate((100, 70, 50)):
            F = z[f"F{L}_mean"] * 1e9
            s, n = turn[:, i, 0], turn[:, i, 1]
            w = F * G * H_SCALE * 1e3 / (L * 100.0 * 2 * np.pi * A_E ** 2 * (np.sin(np.deg2rad(n)) - np.sin(np.deg2rad(s)))) * 1e3
            out[f"merra2_dc_{L}"] = {"annual": round(float(w.mean()), 3), "djf": round(float(w[np.r_[0:59, 334:366]].mean()), 3),
                                     "jja": round(float(w[151:243].mean()), 3),
                                     "years": f"{int(z['years'][0])}-{int(z['years'][-1])} ({len(z['years'])} years)"}
        bj = REPO / "assets" / "sst" / "data" / "bdc_dc.json"
        if bj.exists():                                  # the BDC product's last-30-day flux (on the MERRA-2 scale)
            j = json.loads(bj.read_text())
            for i, L in enumerate((100, 70)):
                r = j.get("last30", {}).get(f"F{L}")
                if not r:
                    continue
                d0, d1 = pd.Timestamp(r["window"][0]), pd.Timestamp(r["window"][1])
                doy = int((d0 + (d1 - d0) / 2).dayofyear)
                ii = (100, 70, 50).index(L)
                sN, nN = turn[doy - 1, ii, 0], turn[doy - 1, ii, 1]
                k = G * H_SCALE * 1e3 / (L * 100.0 * 2 * np.pi * A_E ** 2 * (np.sin(np.deg2rad(nN)) - np.sin(np.deg2rad(sN)))) * 1e3 * 1e9
                out[f"bdc_dc_{L}"] = {"geosfp_30d": round(float(r["geosfp_adjusted"] * k), 3),
                                      "merra2_normal_same_days": round(float(r["merra2_normal"] * k), 3),
                                      "p": r.get("p"), "window": r["window"]}
    mb = REPO / "assets" / "sst" / "data" / "mbudget.json"
    if mb.exists():
        j = json.loads(mb.read_text())
        for L in ("70", "100"):
            u = j.get("upwelling", {}).get(L, {}).get("last30", {})
            if u:
                out[f"geosfp_direct_{L}"] = round(float(u["dir"]), 3)
                out[f"geosfp_dc_{L}"] = round(float(u["sum"]), 3)
        out["geosfp_window"] = j.get("analysis")
    return out


def sections(mls, fpq_adj, t0, t_end, levs, max_gap_bins=5):
    """Dekad-mean section on MLS levels: MLS where measured (>= 2 days in the dekad), linear-in-time fill across MLS gaps
    of up to max_gap_bins dekads (the duty-cycled months), GEOS FP (on the MLS scale) after the last MLS day.
    Returns (bin centres, values (bins x levels), source codes 2 = MLS, 1 = filled, 3 = GEOS FP, 0 = none)."""
    B = bins10(t0, t_end)
    m = snap(mls, levs)
    last_mls = m.dropna(how="all").index.max()
    f = interp_levels(fpq_adj, levs) if fpq_adj is not None and len(fpq_adj) else None
    cen, V, S = [], [], []
    for b0, b1 in B:
        cen.append(b0 + (b1 - b0) / 2)
        sub = m[(m.index >= b0) & (m.index < b1)].dropna(how="any")
        if len(sub) >= 2:
            V.append(sub.mean().values); S.append(2)
        elif b0 > last_mls and f is not None:
            fs = f[(f.index >= b0) & (f.index < b1)].dropna(how="any")
            if len(fs):
                V.append(fs.mean().values); S.append(3)
            else:
                V.append(np.full(len(levs), np.nan)); S.append(0)
        else:
            V.append(np.full(len(levs), np.nan)); S.append(0)
    V, S = np.array(V), np.array(S)
    good = np.where(S == 2)[0]
    for i in np.where(S == 0)[0]:
        lo, hi = good[good < i], good[good > i]
        if len(lo) and len(hi) and hi[0] - lo[-1] <= max_gap_bins + 1:
            a, b = lo[-1], hi[0]
            V[i] = V[a] + (V[b] - V[a]) * (i - a) / (b - a); S[i] = 1
    return pd.DatetimeIndex(cen), V, S, last_mls


# ---------------------------------------------------------------------------------------------------------- figures
INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
NAVY, SIENNA, BAND, NORMAL = "#24466e", "#b4532a", "#e2e5e9", "#59636f"
DIV = ["#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#ffffff", "#c7eae5", "#80cdc1", "#35978f", "#01665e"]


def _style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa1a9")
    ax.tick_params(colors=INK, labelsize=9.2, length=3)


def _pax(ax, pb, pt, km=True):
    ax.set_yscale("log"); ax.set_ylim(pb, pt)
    yt = [p for p in (100, 70, 50, 30, 20, 10, 7, 5, 3) if pt <= p <= pb]
    ax.set_yticks(yt); ax.set_yticklabels([f"{p:g}" for p in yt]); ax.minorticks_off()
    ax.set_ylabel("hPa", fontsize=9.2, color=INK)
    if km:
        from matplotlib.ticker import FixedLocator, FuncFormatter
        sec = ax.secondary_yaxis("right", functions=(lambda p: -H_SCALE * np.log(np.clip(p, 1e-3, None) / 1000.0),
                                                     lambda z: 1000.0 * np.exp(-z / H_SCALE)))
        sec.yaxis.set_major_locator(FixedLocator(list(range(16, 42, 2))))
        sec.yaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))
        sec.yaxis.set_minor_locator(FixedLocator([]))
        sec.set_ylabel("log-pressure height, km", fontsize=8.6, color=MUTED); sec.tick_params(colors=MUTED, labelsize=8.2)


def _wrap(fig, text, y, x=0.055, size=8.6, width=None):
    import textwrap
    W = fig.get_figwidth()
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=width or int((1 - 2 * x) * W * 14.2))
    fig.text(x, y, "\n".join(lines), ha="left", va="bottom", fontsize=size, color=MUTED, linespacing=1.4)


def _head(fig, title, sub, x=0.055):
    import textwrap
    W, Hh = fig.get_figwidth(), fig.get_figheight()
    y = 1 - 0.28 / Hh
    fig.text(x, y, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    lines = textwrap.wrap(sub, width=int((1 - 2 * x) * W * 12.6))
    fig.text(x, y - 0.36 / Hh, "\n".join(lines), ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)
    return y - (0.36 + 0.19 * len(lines)) / Hh


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 90})
    import matplotlib.pyplot as plt
    plt.close(fig)


def _edges_time(cen):
    c = pd.DatetimeIndex(cen)
    mid = c[:-1] + (c[1:] - c[:-1]) / 2
    return pd.DatetimeIndex([c[0] - (mid[0] - c[0])] + list(mid) + [c[-1] + (c[-1] - mid[-1])])


def _pedges(levs):
    lp = np.log(np.asarray(levs, float))
    mid = (lp[1:] + lp[:-1]) / 2
    return np.exp(np.r_[lp[0] + (lp[0] - mid[0]), mid, lp[-1] + (lp[-1] - mid[-1])])


def fig_section(cen, V, S, levs, last_mls, fp_end, kind, clim_mean, clim_at, traj, path, subtitle_extra=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm, ListedColormap
    if kind == "dev":                      # the tape-recorder convention: minus the mean of the period shown, per level
        Z = V - np.nanmean(V, axis=0, keepdims=True)
        lv = np.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9, 1.2])
    else:
        Z = V - clim_at(cen)[levs].values
        lv = np.array([-0.8, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.8])
    cols = ["#6b3a07"] + DIV + ["#003c30"]
    cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1])
    norm = BoundaryNorm(lv, cmap.N)
    fig = plt.figure(figsize=(12.6, 7.4), dpi=130)
    if kind == "dev":
        title = "Aura MLS + GEOS FP · Tropical water-vapour tape recorder · 10°S–10°N"
        sub = ("Water vapour minus its mean over the period shown at each level, 10-day means: the dry (brown) and moist (green) "
               "bands are written at the tropopause by the annual cycle of cold-point temperature and carried up by the "
               "Brewer–Dobson circulation. Dashed: where a band would be if it rose at the MLS climatological ascent speed.")
    else:
        title = "Aura MLS + GEOS FP · Tropical water-vapour anomaly · 10°S–10°N"
        sub = ("Water vapour minus the MLS 2005–2021 seasonal climatology (before the January 2022 Hunga eruption), 10-day "
               "means: what is different from a normal year, with the seasonal tape-recorder bands removed.")
    top = _head(fig, title, sub + subtitle_extra)
    ax = fig.add_axes([0.065, 0.265, 0.855, top - 0.3])
    te = _edges_time(cen)
    Zm = np.ma.masked_invalid(Z)
    cf = ax.pcolormesh(te, _pedges(levs), Zm.T, cmap=cmap, norm=norm, shading="flat")
    if traj:
        for tr in traj:
            ax.plot(tr[0], tr[1], color=INK, lw=0.9, ls=(0, (4, 3)), alpha=0.7)
    _pax(ax, 100, levs[-1]); _style(ax); ax.grid(False)
    ax.set_xlim(te[0], te[-1])
    loc = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    # source strip under the section
    sx = fig.add_axes([0.065, 0.228, 0.855, 0.014]); sx.set_xlim(te[0], te[-1]); sx.axis("off")
    colors = {2: NAVY, 1: "#9fb3c8", 3: SIENNA}
    for i, s in enumerate(S):
        if s in colors:
            sx.axvspan(te[i], te[i + 1], color=colors[s], lw=0)
    ax.axvline(last_mls, color=INK, lw=0.8, ls=":")
    ax.text(last_mls, 1.005, "last MLS day ", transform=ax.get_xaxis_transform(), fontsize=8.2, color=MUTED, va="bottom",
            ha="right")
    cax = fig.add_axes([0.35, 0.112, 0.3, 0.016])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lv, extend="both")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.2, colors=INK)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("drier ← ppmv → moister", fontsize=9, color=INK, labelpad=3)
    fig.text(0.065, 0.185, "strip: ", fontsize=8.2, color=MUTED, va="center")
    for x, s, lab in ((0.098, 2, "Aura MLS measured"), (0.215, 1, "MLS, filled across the radiometer's off weeks"),
                      (0.47, 3, f"GEOS FP on the MLS scale, to {fp_end:%b %-d}")):
        fig.patches.append(plt.Rectangle((x, 0.179), 0.012, 0.012, transform=fig.transFigure, color=colors[s]))
        fig.text(x + 0.016, 0.185, lab, fontsize=8.2, color=INK, va="center")
    _wrap(fig, "Aura MLS v6 (NASA JPL / GES DISC), 10°S–10°N daily means of screened profiles; its 190-GHz radiometer has "
               "run about one week a month since May 2024 and about four days a month since early 2026, so the recent MLS "
               "record is monthly snapshots. GEOS FP (NASA GMAO) does not assimilate stratospheric water vapour; it is used "
               "only after the last MLS day, shifted onto the MLS scale by their mean difference over the 60 days before it.",
          0.012)
    _save(fig, path)


def fig_record(cen, V, S, levs, clim_at, path, last_mls):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm, ListedColormap
    Z = V - clim_at(cen)[levs].values
    lv = np.array([-0.8, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.8])
    cols = ["#6b3a07"] + DIV + ["#003c30"]
    cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1])
    norm = BoundaryNorm(lv, cmap.N)
    fig = plt.figure(figsize=(12.6, 6.6), dpi=130)
    top = _head(fig, f"Aura MLS · Tropical water-vapour anomaly, {cen[0]:%Y}–{last_mls:%Y} · 10°S–10°N",
                "Monthly means minus the 2005–2021 seasonal climatology, 100 to 3 hPa. Dry and moist anomalies enter at the "
                "tropopause (set by cold-point temperature, ENSO and the QBO) and rise; the Hunga eruption (15 January 2022) "
                "put about 150 Tg of water directly into the stratosphere.")
    ax = fig.add_axes([0.065, 0.2, 0.855, top - 0.235])
    te = _edges_time(cen)
    cf = ax.pcolormesh(te, _pedges(levs), np.ma.masked_invalid(Z).T, cmap=cmap, norm=norm, shading="flat")
    _pax(ax, 100, levs[-1]); _style(ax); ax.grid(False)
    ax.axvline(pd.Timestamp("2022-01-15"), color=INK, lw=0.8, ls=":")
    ax.text(pd.Timestamp("2022-01-15"), 1.005, " Hunga", transform=ax.get_xaxis_transform(), fontsize=8.2, color=MUTED)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.set_xlim(te[0], te[-1])
    cax = fig.add_axes([0.35, 0.10, 0.3, 0.018])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lv, extend="both")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.2, colors=INK)
    cb.ax.xaxis.set_label_position("top"); cb.set_label("drier ← ppmv → moister", fontsize=9, color=INK, labelpad=3)
    _wrap(fig, "Aura MLS v6 (NASA JPL / GES DISC), screened per the v6 data-quality document; months with fewer than 2 "
               "measured days are left blank (since May 2024 the radiometer runs only a few days a month).", 0.015)
    _save(fig, path)


def fig_ascent(levs, clim_w, clim_se, clim_sd, n_years, cur, cur_se, cur_end, fpw, fp_years, yr, wref, stats, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.6, 7.4), dpi=130)
    top = _head(fig, "Aura MLS · Ascent speed of the tape recorder vs the Brewer–Dobson w* · tropics",
                "The ascent speed is the slope of the bands: the height gained per day by the phase of the annual cycle "
                "(time of the moist maximum at each level, from an annual harmonic fitted over 12 months). It is compared "
                "with the residual vertical velocity w* of the site's Brewer–Dobson product.")
    ax = fig.add_axes([0.065, 0.32, 0.36, top - 0.36]); _style(ax)
    p = np.array(levs)
    ax.fill_betweenx(p, clim_w - clim_sd, clim_w + clim_sd, color=BAND, lw=0, label="MLS single years, ±1 sd")
    ax.errorbar(clim_w, p, xerr=1.96 * clim_se, color=NAVY, lw=2.2, capsize=2, elinewidth=0.9,
                label=f"MLS 2005–2021 mean ({n_years} years, ±95 %)")
    if cur is not None:
        ax.errorbar(cur, p, xerr=1.96 * cur_se, color=SIENNA, lw=2, capsize=2, elinewidth=0.9, marker="o", ms=3,
                    label=f"MLS, 12 months to {cur_end:%b %-d %Y}")
    if fpw is not None:
        ax.plot(fpw[2:5], FP_ASC[2:5], color="#7a8793", lw=1.6, ls=(0, (4, 2)), marker="s", ms=3,
                label=f"GEOS FP's own tape recorder, {fp_years}")
    mk = [("merra2_dc_100", 100, "D", "#1c2430", "MERRA-2 w* by downward control, annual mean"),
          ("merra2_dc_70", 70, "D", "#1c2430", None), ("merra2_dc_50", 50, "D", "#1c2430", None)]
    for key, L, m, c, lab in mk:
        if key in wref:
            ax.plot(wref[key]["annual"], L, m, color=c, ms=6, mfc="white", mew=1.5, label=lab, zorder=5)
    for key, L, lab in (("bdc_dc_70", 70, "GEOS FP w* by downward control, last 30 days"), ("bdc_dc_100", 100, None)):
        if key in wref:
            ax.plot(wref[key]["geosfp_30d"], L, "^", color=SIENNA, ms=7, label=lab, zorder=6)
    for key, L, lab in (("geosfp_direct_70", 70, "GEOS FP w* from analysed ω*, last 30 days"), ("geosfp_direct_100", 100, None)):
        if key in wref:
            ax.plot(wref[key], L, "*", color="#7a3a1a", ms=10, label=lab, zorder=6)
    ax.set_yscale("log"); ax.set_ylim(100, 25); ax.minorticks_off()
    ax.set_yticks([100, 70, 50, 40, 30]); ax.set_yticklabels(["100", "70", "50", "40", "30"])
    ax.set_xlim(0, 0.85); ax.set_xlabel("ascent speed, mm/s", fontsize=9.2, color=INK); ax.set_ylabel("hPa", fontsize=9.2, color=INK)
    ax.set_title("profile", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, 0.6)
    ax.legend(fontsize=8.0, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.1), ncol=2, handlelength=2.0,
              columnspacing=1.2)
    ax2 = fig.add_axes([0.52, 0.32, 0.45, top - 0.36]); _style(ax2)
    for (lab, key, col) in (("83–46 hPa", "low", NAVY), ("46–26 hPa", "high", "#5a78b4")):
        ys = sorted(yr[key]); vals = [yr[key][y] for y in ys]
        ax2.plot(ys, vals, "o-", color=col, lw=1.6, ms=4, label=f"MLS, calendar years, {lab}")
        mu, sd = stats[key]["clim_mean"], stats[key]["clim_sd"]
        ax2.axhspan(mu - sd, mu + sd, color=col, alpha=0.08, lw=0)
        ax2.axhline(mu, color=col, lw=0.8, ls="--")
        if stats[key].get("current") is not None:
            ax2.errorbar([cur_end.year + (cur_end.dayofyear - 182) / 365.25], [stats[key]["current"]],
                         yerr=[1.96 * stats[key]["current_se"]], fmt="*", color=SIENNA, ms=11, capsize=3,
                         label="12 months to the last MLS day" if key == "low" else None)
    ax2.set_ylabel("ascent speed, mm/s", fontsize=9.2, color=INK)
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("layer means by year (shading: 2005–2021 mean ± 1 sd)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax2.legend(fontsize=8.2, frameon=False, loc="upper left", ncol=1)
    s, h = stats["low"], stats["high"]
    hi_txt = (f"46–26 hPa: {h['clim_mean']:.2f} (±{h['clim_sd']:.2f}); last 12 months {h['current']:.2f} ± "
              f"{1.96 * h['current_se']:.2f}, {'significantly ' + ('faster' if h['z'] > 0 else 'slower') if h['p'] < 0.05 else 'not significantly different'} "
              f"({pfmt(h['p'])}). " if h.get("current") is not None else "")
    txt = (f"83–46 hPa (17.5–21.5 km): 2005–2021 mean {s['clim_mean']:.2f} mm/s (single years ±{s['clim_sd']:.2f}); "
           + (f"last 12 months {s['current']:.2f} ± {1.96 * s['current_se']:.2f}, "
              f"{'significantly ' + ('faster' if s['z'] > 0 else 'slower') if s['p'] < 0.05 else 'not significantly different'} "
              f"(z = {s['z']:+.1f}, {pfmt(s['p'])}). " if s.get("current") is not None else "") + hi_txt
           + (f"MERRA-2 w* at 70 hPa {wref['merra2_dc_70']['annual']:.2f} mm/s (annual mean {wref['merra2_dc_70']['years']}, "
              f"mean over the turnaround latitudes, ~±25–35°)" if "merra2_dc_70" in wref else "")
           + (f"; GEOS FP over the last 30 days {wref['bdc_dc_70']['geosfp_30d']:.2f} by downward control and "
              f"{wref.get('geosfp_direct_70', float('nan')):.2f} from the analysed ω*. " if "bdc_dc_70" in wref else ". ")
           + "The tape recorder moves with w* plus vertical mixing into the tropical pipe, and published comparisons find it "
             "a little faster than reanalysis w* (Brehon et al. 2026: 0.21–0.33 mm/s, slowest near 50 hPa).")
    _wrap(fig, txt, 0.012)
    _save(fig, path)


def fig_entry(mls, fpq_adj, fpt, clim_at, t0, last_mls, fp_end, stats, tstats, path, sd_doy):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(12.6, 7.4), dpi=130)
    top = _head(fig, "Aura MLS + GEOS FP · Water vapour entering the stratosphere · 10°S–10°N",
                "The entry value is set by the coldest point of the tropical tropopause: colder air holds less water. "
                "MLS measured days (dots) and their 30-day means, GEOS FP on the MLS scale after the last MLS day (and on "
                "the days without MLS inside the tested 30 days), against the MLS 2005–2021 normal and its 10–90 % range of "
                "30-day means; bottom, GEOS FP's 100 hPa temperature.")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.75], hspace=0.5, left=0.065, right=0.975, top=top - 0.075, bottom=0.14)
    days = pd.date_range(t0, fp_end)
    for i, L in enumerate((100.0, 70.0)):
        ax = fig.add_subplot(gs[i]); _style(ax)
        m = mls[L]; m = m[m.index >= t0]
        c = clim_at(days)[L]
        st = stats[f"{L:g}"]
        sdd = harm_design(days.dayofyear.values.astype(float), 2) @ sd_doy[L]
        if c is not None:
            ax.fill_between(days, c - 1.2816 * sdd, c + 1.2816 * sdd, color=BAND, lw=0,
                            label="MLS normal, 10–90 % range of 30-day means")
            ax.plot(days, c, color=NORMAL, lw=1.3, ls=(0, (5, 3)), label="MLS 2005–2021 normal")
        ax.plot(m.index, m.values, ".", color=NAVY, ms=2.2, alpha=0.6)
        r = m.rolling("30D", min_periods=3).mean()
        ax.plot(r.index, r.values, color=NAVY, lw=2.0, label="Aura MLS, 30-day mean")
        if fpq_adj is not None:
            f = fpq_adj[L][fpq_adj.index >= last_mls - pd.Timedelta(days=45)]
            fr = f.rolling("30D", min_periods=5).mean()
            fr = fr[fr.index >= last_mls]
            ax.plot(fr.index, fr.values, color=SIENNA, lw=2.0, label="GEOS FP on the MLS scale, 30-day mean")
        ax.axvline(last_mls, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(days[0], days[-1]); ax.set_ylabel("ppmv", fontsize=9, color=INK)
        sig = (f"significant at 5 %, {pfmt(st['p'])}" if st["p"] < 0.05 else f"not significant, {pfmt(st['p'])}")
        ax.set_title(f"{L:g} hPa", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_title(f"30 days to {st['window'][1]} ({st['source']}): {st['value']:.2f} ppmv vs normal {st['normal']:.2f} "
                     f"({st['anom']:+.2f}, {sig})", loc="right", fontsize=9.6, color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        if i == 0:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=8.4, frameon=False, ncol=4, loc="upper left",
                       bbox_to_anchor=(0.06, top - 0.005))
    ax = fig.add_subplot(gs[2]); _style(ax)
    ta = tstats["series"]
    ta = ta.reindex(pd.date_range(ta.index.min(), ta.index.max())).interpolate(limit=4).rolling(15, center=True, min_periods=3).mean()
    ax.fill_between(ta.index, 0, ta.values, where=ta.values >= 0, color="#cc6a3a", lw=0, interpolate=True)
    ax.fill_between(ta.index, 0, ta.values, where=ta.values < 0, color="#5a78b4", lw=0, interpolate=True)
    ax.axhline(0, color="#9aa1a9", lw=0.8)
    ax.set_xlim(days[0], days[-1]); ax.set_ylabel("K", fontsize=9, color=INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("GEOS FP 100 hPa temperature anomaly, 15-day running mean (its own 2018–2025 normal)", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    _wrap(fig, tstats["text"] + " Significance: z-test of the 30-day mean against the MLS year-to-year spread of 30-day means "
               "for the same dates (2005–2021) plus, for the GEOS FP days, its hindcast error on the MLS scale "
               f"({tstats['rmse100']:.2f} ppmv at 100 hPa, {tstats['rmse70']:.2f} at 70 hPa, rms of monthly means).", 0.012)
    _save(fig, path)


# --------------------------------------------------------------------------------------------------------- live
TAIL_JSON = REPO / "assets" / "sst" / "data" / "tape_geosfp_tail.json"
OUT_JSON = REPO / "assets" / "sst" / "data" / "tape_recorder.json"


def live(budget_min=8.0, out=None):
    out = Path(out) if out else REPO / "assets" / "sst"
    tail = json.loads(TAIL_JSON.read_text()) if TAIL_JSON.exists() else {"p_q": list(FP_LEVS), "p_t": list(T_LEVS), "days": {}}
    fpq, fpt = load_fp(tail)
    # new GEOS FP days since the seed / tail
    try:
        start = (fpq.index.max() + pd.Timedelta(days=1)).date()
        fetch_fp(start, date.today(), budget_min=budget_min)
        for f in sorted(FPDAYS.glob("*.npz")):
            d = pd.Timestamp(f.stem)
            if d > pd.Timestamp(str(np.load(FP_SEED)["date"].max())):
                z = np.load(f)
                tail["days"][f"{d:%Y-%m-%d}"] = {"q": [round(float(x), 4) for x in z["q"]], "t": [round(float(x), 3) for x in z["t"]]}
        TAIL_JSON.parent.mkdir(parents=True, exist_ok=True)
        TAIL_JSON.write_text(json.dumps(tail, separators=(",", ":")))
    except Exception as e:                                                            # noqa: BLE001
        log(f"GEOS FP update failed ({str(e)[:120]}); drawing from what is cached")
    fpq, fpt = load_fp(tail)
    mls_tail = None
    mls = load_mls(mls_tail)
    return render(mls, fpq, fpt, out)


def render(mls, fpq, fpt, out):
    t_end = max(mls.index.max(), fpq.index.max())
    last_mls = mls.index.max()
    clim = clim_fit(snap(mls, REC_LEVS), *CLIM_YEARS, nh=3)
    clim_mean = clim.mean
    # GEOS FP -> MLS scale
    off, val, sd30 = fp_adjustment(mls, fpq)
    levs_fp = [L for L in fpq.columns if L <= 100]
    fadj = fpq[levs_fp] + off(fpq.index)
    # ---- sections
    t0 = pd.Timestamp(f"{t_end.year - 3}-{t_end.month:02d}-01")
    cen, V, S, lm = sections(mls, fadj[fadj.index > last_mls], t0, t_end, SEC_LEVS)
    # ascent, climatology and years
    ma = snap(mls, ASC_LEVS)
    yrs = yearly_ascent(ma, ASC_LEVS, range(CLIM_YEARS[0], CLIM_YEARS[1] + 1))
    W = np.array([v["prof"] for v in yrs.values()])
    clim_w, clim_sd = np.nanmean(W, 0), np.nanstd(W, 0, ddof=1)
    clim_se = clim_sd / np.sqrt(np.isfinite(W).sum(0))
    all_years = yearly_ascent(ma, ASC_LEVS, range(2005, t_end.year + 1), min_days=40)
    cur, cur_se, ncur = window_ascent(ma, ASC_LEVS, last_mls)
    yr = {k: {y: v[k] for y, v in all_years.items() if np.isfinite(v[k])} for k in LAYERS}
    stats = {}
    for k in LAYERS:
        base = np.array([yr[k][y] for y in yr[k] if CLIM_YEARS[0] <= y <= CLIM_YEARS[1]])
        s_ = {"levels_hPa": list(LAYERS[k]), "clim_mean": float(base.mean()), "clim_sd": float(base.std(ddof=1)),
              "n_years": int(len(base))}
        if cur is not None and np.isfinite(cur[k]):
            z = (cur[k] - s_["clim_mean"]) / np.sqrt(s_["clim_sd"] ** 2 + cur_se[k] ** 2)
            s_.update({"current": float(cur[k]), "current_se": float(cur_se[k]), "z": float(z), "p": norm_p(z),
                       "n_days": ncur})
        stats[k] = s_
    fpw = None
    fp_years = "2018–2025"
    fyr = yearly_ascent(fpq[FP_ASC], FP_ASC, range(2018, 2026), min_days=60)
    if fyr:
        fpw = np.nanmean(np.array([v["prof"] for v in fyr.values()]), 0)
        fp_years = f"{min(fyr)}–{max(fyr)} ({len(fyr)} years)"
    wref = w_star_refs()
    # climatological trajectories of the dry band: each year's moist-max time at 100 hPa + integral of dz/w
    traj = []
    zz = zlp(SEC_LEVS[:8])
    cw = pd.Series(clim_w).ffill().bfill().values                          # 100 hPa takes the 83 hPa speed
    wz = np.interp(zz, zlp(ASC_LEVS), cw)
    tt = np.r_[0, np.cumsum(np.diff(zz) * 1e6 / wz[:-1] / 86400.0)]        # days above 100 hPa
    for y in range(t0.year, t_end.year + 1):
        sub = snap(mls, [100.0])[(mls.index >= f"{y}-01-01") & (mls.index <= f"{y}-12-31")]
        r = phase_times(sub, [100.0], 20)
        if r is None:
            continue
        for shift in (0.0, 182.6):                                         # moist maximum and dry minimum
            tstart = sub.index[0] + pd.Timedelta(days=float(r[0][0] % 365.25) + shift)
            traj.append(([tstart + pd.Timedelta(days=float(x)) for x in tt], SEC_LEVS[:8]))
    fp_end = fpq.index.max()
    fig_section(cen, V, S, SEC_LEVS, last_mls, fp_end, "dev", clim_mean, clim, traj, out / "tape_section.webp")
    fig_section(cen, V, S, SEC_LEVS, last_mls, fp_end, "anom", clim_mean, clim, None, out / "tape_anom.webp")
    # record: monthly MLS (>= 2 days)
    mm = snap(mls, REC_LEVS)
    cnt = mm[100.0].resample("MS").count()
    mon = mm.resample("MS").mean()[cnt >= 2]
    mon = mon.reindex(pd.date_range(mon.index.min(), mon.index.max(), freq="MS"))
    fig_record(mon.index + pd.Timedelta(days=14), mon.values, None, REC_LEVS, clim, out / "tape_record.webp", last_mls)
    fig_ascent(ASC_LEVS[PROF_SHOW], clim_w[PROF_SHOW], clim_se[PROF_SHOW], clim_sd[PROF_SHOW], len(yrs),
               None if cur is None else cur["prof"][PROF_SHOW], None if cur is None else cur_se["prof"][PROF_SHOW],
               last_mls, fpw, fp_years, yr, wref, stats, out / "tape_ascent.webp")
    # ---- entry value: last 30 days at 100 / 70 hPa; each day MLS where measured, else GEOS FP on the MLS scale
    estats = {}
    m2 = interp_levels(snap(mls, [121.15, 100.0, 82.54, 68.13, 56.23]), [100.0, 70.0])
    clim2 = clim_fit(m2, *CLIM_YEARS, nh=3)
    # year-to-year sd of 30-day means by day of year (for the band): trailing 30-day means of the anomaly in the
    # climatology years, sd across years per day of year, smoothed with two harmonics
    an2 = (m2 - clim2(m2.index))
    an2 = an2[(an2.index.year >= CLIM_YEARS[0]) & (an2.index.year <= CLIM_YEARS[1])]
    r30 = an2.rolling("30D", min_periods=10).mean()
    sd_doy = {}
    for L in (100.0, 70.0):
        g = r30[L].groupby(r30.index.dayofyear).std(ddof=1).dropna()
        cf = np.linalg.lstsq(harm_design(g.index.values.astype(float), 2), g.values, rcond=None)[0]
        sd_doy[L] = cf
    w1 = t_end; w0 = w1 - pd.Timedelta(days=29)
    days_w = pd.date_range(w0, w1)
    for L in (100.0, 70.0):
        ms = m2[L].reindex(days_w)
        fs = fadj[L].reindex(days_w)
        merged = ms.where(ms.notna(), fs)             # GEOS FP (on the MLS scale) on the days MLS did not measure
        n_m, n_f = int(ms.notna().sum()), int((ms.isna() & merged.notna()).sum())
        dd = merged.dropna().index
        val_ = float(merged.mean())
        src = " + ".join(x for x in ([f"MLS {n_m} d"] if n_m else []) + ([f"GEOS FP {n_f} d"] if n_f else []))
        extra = sd30.get(L, 0.0) * n_f / max(n_m + n_f, 1)
        normal = float(clim2(dd)[L].mean())
        yy = []                        # year-to-year sd of 30-day mean anomalies for these calendar dates, 2005-2021
        for y in range(CLIM_YEARS[0], CLIM_YEARS[1] + 1):
            a = pd.Timestamp(year=y, month=w0.month, day=min(w0.day, 28 if w0.month == 2 else w0.day))
            b = a + pd.Timedelta(days=29)
            s_ = m2[L][(m2.index >= a) & (m2.index <= b)].dropna()
            if len(s_) >= 10:
                yy.append(float((s_ - clim2(s_.index)[L]).mean()))
        sd_iav = float(np.std(yy, ddof=1))
        z = (val_ - normal) / np.sqrt(sd_iav ** 2 + extra ** 2)
        estats[f"{L:g}"] = {"window": [f"{w0:%b %-d}", f"{w1:%b %-d}"], "source": src, "value": val_, "normal": normal,
                            "anom": val_ - normal, "anom_pct": 100 * (val_ - normal) / normal, "sd_iav_30d": sd_iav,
                            "sd_offset": extra, "z": float(z), "p": norm_p(z), "sd30": sd_iav, "n_years": len(yy)}
    # GEOS FP T100 anomaly vs its own 2018-2025 normal, and its link to the MLS 100 hPa water
    tcl = clim_fit(fpt[[100.0]], 2018, 2025, 2)
    ta = (fpt[100.0] - tcl(fpt.index)[100.0])
    ta_d = ta[ta.index >= t0]
    mon_t = ta.resample("MS").mean()
    h100 = snap(mls, [100.0])[100.0]
    ha = (h100 - clim(h100.index)[100.0]).resample("MS").mean()
    j = mon_t.index.intersection(ha.index)
    ok = np.isfinite(mon_t[j].values) & np.isfinite(ha[j].values)
    x, y_ = mon_t[j].values[ok], ha[j].values[ok]
    r_ = float(np.corrcoef(x, y_)[0, 1]); slope = float(np.polyfit(x, y_, 1)[0])
    # AR(1)-adjusted significance of r
    def r1(v):
        v = v - v.mean(); return float((v[1:] * v[:-1]).sum() / (v * v).sum())
    neff = len(x) * (1 - r1(x) * r1(y_)) / (1 + r1(x) * r1(y_))
    from scipy import stats as SS
    tv = r_ * np.sqrt((neff - 2) / (1 - r_ ** 2)); p_r = float(2 * SS.t.sf(abs(tv), neff - 2))
    t30 = float(ta[ta.index > fpt.index.max() - pd.Timedelta(days=30)].mean())
    ttext = (f"Monthly 100 hPa water (MLS) against GEOS FP's 100 hPa temperature, {j[ok][0]:%Y-%m}–{j[ok][-1]:%Y-%m}: "
             f"r = {r_:.2f}, {slope:+.2f} ppmv per K ({'significant' if p_r < 0.05 else 'not significant'}, "
             f"{'p < 0.001' if p_r < 0.001 else f'p = {p_r:.3f}'}, "
             f"autocorrelation-adjusted). Last 30 days of GEOS FP: {t30:+.2f} K.")
    fig_entry(m2, fadj, fpt, clim2, t0, last_mls, fp_end, estats, {"series": ta_d, "text": ttext, "rmse100": sd30[100.0], "rmse70": sd30[70.0]}, out / "tape_entry.webp",
              sd_doy)
    js = {"made": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"), "last_mls_day": f"{last_mls:%Y-%m-%d}",
          "last_geosfp_day": f"{fp_end:%Y-%m-%d}", "band": "10S-10N", "clim_years": list(CLIM_YEARS),
          "mls_version": "ML2H2O v006 (v6.0x)", "geosfp_vs_mls": val,
          "ascent_levels_hPa": ASC_LEVS, "ascent_clim_mm_s": [round(float(x), 3) for x in clim_w],
          "ascent_clim_sd": [round(float(x), 3) for x in clim_sd], "ascent_years": sorted(int(y) for y in yrs),
          "ascent_current_mm_s": None if cur is None else [round(float(x), 3) for x in cur["prof"]],
          "ascent_current_se": None if cur is None else [round(float(x), 3) for x in cur_se["prof"]],
          "ascent_current_n_days": ncur, "ascent_layers": stats,
          "ascent_geosfp_mm_s": None if fpw is None else dict(zip([f"{L:g}" for L in FP_ASC], [round(float(x), 3) for x in fpw])),
          "w_star": wref, "entry": estats,
          "t100": {"r_monthly": round(r_, 3), "slope_ppmv_per_K": round(slope, 3), "p": round(p_r, 4), "n_eff": round(neff, 1),
                   "last30_K": round(t30, 2)}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    (out / "data" / "tape_recorder.json").write_text(json.dumps(js, indent=1, default=float)) \
        if (out / "data").exists() else OUT_JSON.write_text(json.dumps(js, indent=1, default=float))
    log(json.dumps({k: js[k] for k in ("last_mls_day", "last_geosfp_day", "ascent_layers", "entry", "t100")}, default=float)[:1500])
    return js


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "live"
    if cmd == "fetch-fp":
        fetch_fp(date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3]),
                 step=int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "build":
        build_refs()
    elif cmd == "render":                     # laptop test: from the references only, into a given directory
        fpq, fpt = load_fp(json.loads(TAIL_JSON.read_text()) if TAIL_JSON.exists() else None)
        o = Path(sys.argv[2]); (o / "data").mkdir(parents=True, exist_ok=True)
        render(load_mls(), fpq, fpt, o)
    elif cmd == "live":
        live()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
