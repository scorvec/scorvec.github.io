#!/usr/bin/env python3
"""Teleconnection indices from every GEPS member, days 1-35 — plumes,
weekly probabilities, change vs the previous run, and an observed tail.

The projectors come from build_telecon_patterns.py (CPC-style RPCA of NCEP/NCAR
R1 500 hPa height for the NH patterns, SLP EOF1 for the AO and AAO). Here they
are applied to three things on the same 2.5 deg grid and in the same units:

  members    every GEPS member's de-drifted anomaly (forecast.py: forecast minus
             the lead-matched GEPS8 hindcast climatology, re-based onto
             1991-2020 with the ERA5 period shift, exactly as the maps are),
             so the forecast index carries no model drift;
  observed   GEPS's own 00Z analyses (day 0, ensemble mean) for the last
             weeks, against the NCEP 1991-2020 day-of-year climatology — the
             tail the forecast should join. (NCEP R1's daily files on PSL lag
             by months, so the reanalysis cannot supply a current tail; the
             Datamart keeps ~25 days of analyses and telecon/analysis/ keeps
             them from there on.) For sea-level pressure the tail is the DAILY
             MEAN of the 00Z run's 0/6/12/18 h steps, not the instantaneous
             00Z analysis: a fixed-UTC pressure difference between Tahiti and
             Darwin (or the east Pacific and Indonesia) carries the S1/S2
             atmospheric tides as a constant ~1-2 hPa offset, 0.6-1 seasonal sd
             of the SOI/EQSOI, and the daily-mean climatology it is compared
             with has none. Four 6-hourly samples cancel both tides exactly,
             which is also how forecast.py builds every member day. Days that
             have left the Datamart fall back to the 00Z field plus the mean
             (daily - 00Z) offset measured on the days that have both. The
             500 hPa tail stays instantaneous (the tide there is metres at
             20-90N against index sigmas of 50-100 m).
  CPC        CPC's own published daily AO/NAO/PNA/AAO, drawn as markers on the
             tail, so the reader can see how closely this reproduction tracks
             the operational index (r in the validation file). For the two
             SLP ENSO indices (SOI, EQSOI — slp box differences, negative in
             El Nino) CPC publishes monthly values only: those are drawn as
             a bar across each month, put in daily-index units with the
             pattern file's monthly_sd (the std of monthly means of the
             annually standardized daily index) and the seasonal factor; the
             SOI also gets the Long Paddock (Queensland) daily SOI as markers.

When telecon/hindcast_indices.nc exists (telecon_hindcast.py), every weekly
probability is also CALIBRATED: the hindcast starts within +-45 days of the
init day-of-year give, per index and week, the anomaly correlation r (skill),
the regression obs = a*fcst + b and its residual spread s, and the live
ensemble-mean weekly index m becomes P(week >= +0.5) = 1 - Phi((0.5-a*m-b)/s).
With no skill that collapses to climatology, which is the honest week-5
answer. The hindcast mean is 4 members against 21 live, so a is a touch
conservative.

Everything is in standard deviations of the DAILY index over 1991-2020, so
"+1" means a day at the 84th percentile of daily variability, and a WEEKLY
mean of +1 is a strong signal. Probabilities are member fractions of a 21-
member ensemble, so they resolve to about 5% and 0%/100% mean "no member",
not certainty.

Outputs (figs/ and the site's assets/geps/)
  geps_telecon_series.webp   one panel per index: observed tail, CPC markers,
                             p10-p90 and p25-p75 bands, ensemble mean, and
                             the previous run's mean on the common days
  geps_telecon_probs.webp    weeks 1-5 x indices: P(<= -0.5), neutral,
                             P(>= +0.5) as stacked bars, weekly mean and its
                             change vs the previous run
  telecon.json               all of the above as numbers, for the page table

    python telecon_geps.py --cycle 20260903
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyproj  # noqa: F401,E402  before xarray/eccodes: eccodes preloads its own libproj (CLAUDE.md gotcha)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "scripts" / "lib")]
import paths                                                    # noqa: E402
import archive                                                  # noqa: E402
import refpack                                                  # noqa: E402
import render_maps as rm                                        # noqa: E402  (reref)
import forecast as fcmod                                        # noqa: E402  (_grab, coarsen25)
from math import erf, sqrt                                      # noqa: E402
from telecon_io import (cpc_daily, cpc_monthly, psl_daily, cpc_soi_monthly,     # noqa: E402
                        cpc_eqsoi_monthly, longpaddock_daily_soi, CPC, TC, period_shift)
sys.path.insert(0, str(paths.REPO / "scripts" / "strat"))
import strat_maps as sm                                          # noqa: E402  (100 hPa gribs)
import strat_series as ss                                        # noqa: E402  (members())

LIVE = paths.LIVE
FIGS = paths.FIGS
SITE = paths.SITE_ASSETS
TCR = paths.TC_REF              # static reference: patterns, validation, hindcast indices
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
ORDER = ["ao", "nao", "pna", "epo", "wpo", "soi", "eqsoi", "ea", "wp", "epnp", "eawr", "sca", "tnh", "pol",
         "nam100", "aao"]
SKILL_MIN = 0.25                # hindcast anomaly correlation below this = "no skill", shaded
INK, MUTED = "#2f2c28", "#5a5650"   # annotation greys: the lighter ones were hard to read
TAIL_DAYS = 45
ANA = paths.ANA                 # GEPS day-0 ensemble-mean fields at 2.5 deg, one npz per day
WINDOW_DOY = 45                 # hindcast starts pooled for skill/calibration
THR = 0.5                       # +-0.5 sd of the daily index = "positive"/"negative" phase
PSL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis"
CPC_URL = {"ao": "cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv",
           "nao": "cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv",
           "pna": "cwlinks/norm.daily.pna.cdas.z500.19500101_current.csv",
           "aao": "cwlinks/norm.daily.aao.cdas.z700.19790101_current.csv"}
CPC_MONTHLY_URL = {"soi": ("https://www.cpc.ncep.noaa.gov/data/indices/soi", "soi.txt"),
                   "eqsoi": ("https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for", "reqsoi.for")}
LONGPADDOCK_URL = ("https://data.longpaddock.qld.gov.au/SeasonalClimateOutlook/SouthernOscillationIndex/"
                   "SOIDataFiles/DailySOI1887-1989Base.txt")


STAMPS = TC / "refreshed.json"   # when each file was last fetched; mtimes do not survive the frames-branch round trip


def refresh(url, dest, max_age_h=12):
    """Re-download a file the source updates daily, unless this pipeline fetched it within max_age_h. A failed
    refresh keeps the copy already held (the overlay is then a few days short, not missing)."""
    if os.environ.get("GEPS_OFFLINE") == "1":
        return
    try:
        st = json.loads(STAMPS.read_text()) if STAMPS.exists() else {}
    except ValueError:
        st = {}
    key = str(dest.relative_to(TC)) if dest.is_relative_to(TC) else str(dest)
    if dest.exists() and key in st and \
            pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(st[key]) < pd.Timedelta(hours=max_age_h):
        return
    import webget
    try:
        b = webget.get(url, timeout=300)
    except Exception as e:                                       # noqa: BLE001
        print(f"  (refresh failed {dest.name}: {str(e)[:60]})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b)
    st[key] = pd.Timestamp.utcnow().tz_localize(None).isoformat()
    STAMPS.parent.mkdir(parents=True, exist_ok=True)
    STAMPS.write_text(json.dumps(st, indent=0))


class Patterns:
    def __init__(self):
        self.ds = xr.open_dataset(TCR / "patterns.nc")
        self.meta = json.loads((TCR / "validation.json").read_text())
        self.ids = [i for i in ORDER if i in self.meta] + \
                   [i for i in self.meta if i not in ORDER]
        # CPC does not define every pattern in every month (EP/NP is not a
        # leading mode in Aug-Sep, TNH only Dec-Feb): an index is shown only
        # for valid months in which CPC itself publishes it, judged by at
        # least half of the base years having a value in that calendar month.
        cm = cpc_monthly().loc["1991":"2020"]
        self.defined = {}
        for sid, m in self.meta.items():
            name = m.get("cpc")
            if m["field"] == "slp" or name not in cm:
                self.defined[sid] = set(range(1, 13))
            else:
                ok = cm[name].notna().groupby(cm.index.month).sum()
                self.defined[sid] = {int(k) for k, v in ok.items() if v >= 15}
        lat = self.ds.lat.values
        self.w = np.sqrt(np.clip(np.cos(np.deg2rad(lat.astype("float64"))), 0, None))[:, None] * np.ones((1, self.ds.lon.size))

    def index(self, sid, anom, months, doys):
        """anom: array (..., lat, lon) on the NCEP grid (z500 m / slp hPa);
        months, doys: calendar month and day of year of each sample.
        Returns the daily index in SEASONAL standard deviations (the std of
        the 1991-2020 daily projection within +-15 days of that day of year)."""
        m = self.meta[sid]
        mo = np.asarray(months) - 1
        sig = self.ds.sigma_doy.sel(index=sid).values[np.asarray(doys) - 1]
        P = self.ds.projector.sel(index=sid).values[mo]                    # (n, lat, lon), valid month
        a = np.asarray(anom, dtype="float64")
        if m["standardize"]:
            sd = self.ds.std_month_z500.values[mo]                        # (n, lat, lon)
            sd = np.where(sd > 0, sd, np.nan)
            a = a / sd
        if m.get("demean"):
            lat = self.ds.lat.values
            cosw = (np.cos(np.deg2rad(lat)) * (lat >= 20))[:, None] * np.ones((1, self.ds.lon.size))
            a = a - (np.nan_to_num(a) * cosw).sum(axis=(-2, -1), keepdims=True) / cosw.sum()
        z = np.nan_to_num(a * self.w)
        raw = np.einsum("nij,nij->n", z, P)
        out = raw / sig
        undefined = ~np.isin(np.asarray(months), sorted(self.defined[sid]))
        out[undefined] = np.nan
        return out

    def to_seasonal(self, sid, doys):
        """Factor that converts an ANNUALLY standardized value (CPC/PSL
        convention, and the hindcast file when it says so) to seasonal sd."""
        return self.meta[sid]["sigma_daily"] / self.ds.sigma_doy.sel(index=sid).values[np.asarray(doys) - 1]


def members_anom(tag, cycle, base):
    p = archive.members_path(tag, cycle)          # this run's live file, or the previous cycle's archive
    if p is None:
        return None
    d = xr.open_dataset(p)
    a = d[f"{tag}_anom"].load()                    # (L, number, latitude, longitude)
    # same 1991-2020 re-basing as the maps: period shift on the ensemble-mean
    # shape, then broadcast — it is a function of valid date only
    r = rm.reref(tag, a.mean("number"), base)
    if r is not None:
        a = a + r.values[:, None]
    if tag == "mslp":
        a = a * 0.01                                # Pa -> hPa, NCEP slp units
    return a


def analysis_field(tag, day):
    """GEPS 00Z day-0 ensemble mean on the 2.5 deg grid, cached; None if the
    Datamart no longer has it."""
    ANA.mkdir(parents=True, exist_ok=True)
    f = ANA / f"{tag}_{day:%Y%m%d}.npz"
    if f.exists():
        return np.load(f)["v"]
    var, _ = fcmod.VARS[tag]
    m = fcmod._grab(day.strftime("%Y%m%d"), var, 0)
    if m is None:
        return None
    v = fcmod.coarsen25(m).values.astype("float32")
    np.savez_compressed(f, v=v)
    return v


DAILY_MEAN_TAGS = ("mslp",)     # tail fields taken as the 00Z run's 0-18 h daily mean
TIDE_MIN_PAIRS = 10


def analysis_daily(tag, day):
    """GEPS 00Z-run daily mean (steps 0, 6, 12, 18 h, ensemble mean) on the
    2.5 deg grid, cached; None if the Datamart no longer has the steps."""
    f = ANA / f"{tag}_dm_{day:%Y%m%d}.npz"
    if f.exists():
        return np.load(f)["v"]
    v0 = analysis_field(tag, day)
    if v0 is None:
        return None
    var, _ = fcmod.VARS[tag]
    parts = [v0]
    for L in (6, 12, 18):
        q = ANA / f"{tag}_P{L:03d}_{day:%Y%m%d}.npz"        # an extended cycle's own first day: forecast.py kept it
        if q.exists():
            parts.append(np.load(q)["v"]); continue
        m = fcmod._grab(day.strftime("%Y%m%d"), var, L)
        if m is None:
            return None
        parts.append(fcmod.coarsen25(m).values.astype("float32"))
    v = np.mean(parts, axis=0).astype("float32")
    np.savez_compressed(f, v=v)
    return v


_CW = {}


def model_clim25(tag):
    """GEPS8 climatology at lead 0.5 (the day-1 mean, i.e. the analysis) on the
    2.5 deg grid, day-of-year axis wrapped."""
    if tag not in _CW:
        cv = refpack.clim(tag).sel(L=0.5, method="nearest").astype("float64")
        cv = xr.concat([cv, cv.isel(X=0).assign_coords(X=360.0)], dim="X")
        cv = cv.interp(Y=fcmod.LAT25, X=fcmod.LON25)
        _CW[tag] = xr.concat([cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 366), cv,
                              cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 366)], dim="doy")
    return _CW[tag]


def observed(pat, end, days=TAIL_DAYS):
    """Standardized indices from GEPS day-0 analyses for the last `days`.

    The analysis is anomalised exactly like the forecast — against the GEPS8
    climatology (here at lead 0.5, the day-1 mean) and re-based to 1991-2020
    with the ERA5 period shift — not against the NCEP climatology. Against
    NCEP's it carried the systematic GEPS-vs-R1 height offset, which
    projected onto the monopole-ish patterns (EA, POL) as a spurious +1 sd
    step at the forecast seam."""
    fields = {}
    # 100 hPa: lead-0 ensemble mean, NCEP climatology (no model climatology exists)
    ts, vs = [], []
    for k in range(days, -1, -1):
        d = end - pd.Timedelta(days=k)
        f = ANA / f"z100_{d:%Y%m%d}.npz"
        if f.exists():
            vs.append(np.load(f)["v"]); ts.append(d); continue
        g = sm.fetch(d.strftime("%Y%m%d"), "00", "HGT", 100, 0)
        m = sm.ens_mean(g) if g else None
        if m is None:
            continue
        m = m.rename({dd: n for dd, n in (("lat", "latitude"), ("lon", "longitude")) if dd in m.dims})
        v = fcmod.coarsen25(m).values.astype("float32")
        np.savez_compressed(f, v=v); vs.append(v); ts.append(d)
    if ts:
        t = pd.DatetimeIndex(ts)
        an = np.stack(vs) - pat.ds["clim_doy_z100"].values[t.dayofyear.values - 1]
        fields["z100"] = (t, an, t.month.values, t.dayofyear.values)
    for tag, fld in (("zg500", "z500"), ("mslp", "slp")):
        cw = model_clim25(tag)
        shift = period_shift(tag)
        ts, vs = [], []
        inst, pairs = [], []            # 00Z-only days; (daily mean - 00Z) where both exist
        for k in range(days, -1, -1):
            d = end - pd.Timedelta(days=k)
            v = analysis_field(tag, d)
            if v is None:
                continue
            if tag in DAILY_MEAN_TAGS:
                vd = analysis_daily(tag, d)
                if vd is not None:
                    pairs.append(vd - v); v = vd
                else:
                    inst.append(len(ts))
            an = v - cw.interp(doy=d.dayofyear).values
            if shift is not None:
                an = an + shift[d.dayofyear - 1]
            ts.append(d); vs.append(an)
        if not ts:
            continue
        t = pd.DatetimeIndex(ts)
        an = np.stack(vs)
        if inst:
            if len(pairs) >= TIDE_MIN_PAIRS:
                an[inst] += np.mean(pairs, axis=0)
                print(f"  {tag} tail: {len(ts) - len(inst)} daily-mean days, {len(inst)} on the 00Z field "
                      f"+ the mean (daily - 00Z) tide offset of {len(pairs)} days")
            else:
                print(f"  {tag} tail: {len(inst)} days on the instantaneous 00Z field, uncorrected "
                      f"(only {len(pairs)} daily-mean days to measure the tide offset)")
        if tag == "mslp":
            an = an * 0.01                                  # Pa -> hPa
        fields[fld] = (t, an, t.month.values, t.dayofyear.values)
    out = {}
    for sid in pat.ids:
        fld = pat.meta[sid]["field"]
        if fld not in fields:
            continue
        t, an, mo, dy = fields[fld]
        out[sid] = pd.Series(pat.index(sid, an, mo, dy), index=t)
    return out


def phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def calibration(pat, base):
    """Per index and week: skill and a regression calibration from the
    hindcast starts within +-WINDOW_DOY days of the init day-of-year."""
    q = TCR / "hindcast_indices.nc"
    if not q.exists():
        return {}
    ds = xr.open_dataset(q)
    S = pd.DatetimeIndex(ds.S.values)
    d = ((S.dayofyear.values - base.dayofyear + 183) % 366) - 183
    win = np.abs(d) <= WINDOW_DOY
    day = ds.L.values + 0.5
    annual = ds.attrs.get("units", "annual") == "annual"     # older file: annual sd
    vdoy = (S[win].values[:, None] + pd.to_timedelta(np.ceil(ds.L.values), unit="D").values[None]) \
        .astype("datetime64[D]")
    vdoy = pd.DatetimeIndex(vdoy.ravel()).dayofyear.values.reshape(vdoy.shape)
    out = {}
    for sid in ds.index.values:
        F = ds.fcst.sel(index=sid).values[win]
        O = ds.obs.sel(index=sid).values[win]
        if annual:
            k = pat.to_seasonal(str(sid), vdoy)
            F, O = F * k, O * k
        rows = []
        for w0, w1 in WEEKS:
            sel = (day >= w0) & (day <= w1)
            f = np.nanmean(F[:, sel], 1); o = np.nanmean(O[:, sel], 1)
            ok = np.isfinite(f) & np.isfinite(o)
            if ok.sum() < 30:
                rows.append(None); continue
            f, o = f[ok], o[ok]
            r = float(np.corrcoef(f, o)[0, 1])
            a, b = np.polyfit(f, o, 1)
            s_ = float(np.std(o - (a * f + b), ddof=2))
            rows.append(dict(n=int(ok.sum()), r=round(r, 2), a=round(float(a), 2), b=round(float(b), 2),
                             s=round(s_, 2), obs_sd=round(float(o.std()), 2),
                             fc_sd=round(float(f.std()), 2)))
        out[str(sid)] = rows
    return out


def members_z100(pat, cycle, base):
    """All members' 100 hPa height at 00Z, days 1-35, 2.5 deg, anomalised
    against the NCEP 1991-2020 day-of-year climatology. There is no GEPS8
    hindcast at 100 hPa, so this index is NOT de-drifted the way the others
    are — a plain anomaly against the reanalysis, with the hemispheric mean
    removed in the projection, which is what makes an annular mode."""
    cache = LIVE / f"geps_members_z100_{cycle}.nc"
    held = archive.members_path("z100", cycle)       # this run's cache, or the previous cycle's archive
    if held is not None:
        return xr.open_dataset(held)["z100_anom"].load()
    if cycle != os.environ.get("GEPS_CYCLE", cycle):
        return None                                  # never re-fetch an older cycle's 35 steps
    clim = pat.ds["clim_doy_z100"].values
    fields, leads = [], []
    for L in range(24, 841, 24):
        f = sm.fetch(cycle, "00", "HGT", 100, L)
        mm = ss.members(f) if f else None
        if mm is None:
            continue
        mm = mm.rename({d: n for d, n in (("lat", "latitude"), ("lon", "longitude")) if d in mm.dims})
        c = fcmod.coarsen25(mm).values.astype("float32")               # (number, 73, 144)
        valid = base + pd.Timedelta(hours=L)
        fields.append(c - clim[valid.dayofyear - 1][None]); leads.append(L // 24)
    if not fields:
        return None
    a = xr.DataArray(np.stack(fields), dims=("L", "number", "latitude", "longitude"),
                     coords={"L": leads, "number": np.arange(fields[0].shape[0]),
                             "latitude": fcmod.LAT25, "longitude": fcmod.LON25})
    xr.Dataset({"z100_anom": a}).to_netcdf(cache, encoding={"z100_anom": {"zlib": True, "complevel": 4}})
    return a


def forecast(pat, cycle):
    base = pd.Timestamp(cycle)
    fields = {"z500": members_anom("zg500", cycle, base),
              "slp": members_anom("mslp", cycle, base)}
    if any(pat.meta[i]["field"] == "z100" for i in pat.ids):
        try:
            fields["z100"] = members_z100(pat, cycle, base)
        except Exception as e:                                      # noqa: BLE001
            print(f"  (100 hPa members unavailable: {str(e)[:60]})"); fields["z100"] = None
    if all(v is None for v in fields.values()):
        return None
    out = {}
    for sid in pat.ids:
        a = fields.get(pat.meta[sid]["field"])
        if a is None:
            continue
        # day L of a daily-mean field is the UTC day base + L - 1 (the 24 h ending at step 24L); the 100 hPa height is
        # the instantaneous step 24L itself, dated 12 h before it so it falls inside the forecast day it closes
        off = pd.Timedelta(hours=12) if pat.meta[sid]["field"] == "z100" else pd.Timedelta(days=1)
        valid = base + pd.to_timedelta(a.L.values, unit="D") - off
        mo = np.repeat(valid.month.values, a.sizes["number"])
        dy = np.repeat(valid.dayofyear.values, a.sizes["number"])
        flat = a.values.reshape(-1, a.sizes["latitude"], a.sizes["longitude"])
        idx = pat.index(sid, flat, mo, dy).reshape(a.sizes["L"], a.sizes["number"])
        if not np.isfinite(idx).any():
            print(f"  {sid}: not a CPC-defined pattern in {valid[0]:%b}–{valid[-1]:%b} — left out")
            continue
        out[sid] = (pd.DatetimeIndex(valid), idx)
    return out


def weekly_stats(t, M, base, prev=None, cal=None):
    rows = []
    for k, (w0, w1) in enumerate(WEEKS):
        sel = (t >= base + pd.Timedelta(days=w0 - 1)) & (t < base + pd.Timedelta(days=w1)) \
              & np.isfinite(M).all(1)
        if not sel.any():
            continue
        wm = M[sel].mean(0)                       # per-member weekly mean
        row = dict(week=k + 1, d0=(base + pd.Timedelta(days=w0 - 1)).strftime("%Y-%m-%d"),
                   d1=(base + pd.Timedelta(days=w1 - 1)).strftime("%Y-%m-%d"),
                   ndays=int(sel.sum()),
                   mean=round(float(wm.mean()), 2),
                   p10=round(float(np.percentile(wm, 10)), 2),
                   p25=round(float(np.percentile(wm, 25)), 2),
                   p50=round(float(np.percentile(wm, 50)), 2),
                   p75=round(float(np.percentile(wm, 75)), 2),
                   p90=round(float(np.percentile(wm, 90)), 2),
                   p_neg=round(float((wm <= -THR).mean()), 3),
                   p_pos=round(float((wm >= THR).mean()), 3),
                   p_strong_neg=round(float((wm <= -1.0).mean()), 3),
                   p_strong_pos=round(float((wm >= 1.0).mean()), 3))
        if prev is not None:
            pt, pM = prev
            psel = (pt >= base + pd.Timedelta(days=w0 - 1)) & (pt < base + pd.Timedelta(days=w1)) \
                   & np.isfinite(pM).all(1)
            if psel.any():
                pw = pM[psel].mean(0)
                row["prev_mean"] = round(float(pw.mean()), 2)
                row["prev_ndays"] = int(psel.sum())
                row["prev_p_neg"] = round(float((pw <= -THR).mean()), 3)
                row["prev_p_pos"] = round(float((pw >= THR).mean()), 3)
        if cal and k < len(cal) and cal[k]:
            c = cal[k]
            mu = c["a"] * row["mean"] + c["b"]
            row.update(skill_r=c["r"], no_skill=bool(c["r"] < SKILL_MIN),
                       cal_n=c["n"], cal_a=c["a"], cal_b=c["b"], cal_s=c["s"],
                       cal_mean=round(float(mu), 2),
                       cal_p_pos=round(1.0 - phi((THR - mu) / c["s"]), 3),
                       cal_p_neg=round(phi((-THR - mu) / c["s"]), 3),
                       cal_p_strong_pos=round(1.0 - phi((1.0 - mu) / c["s"]), 3),
                       cal_p_strong_neg=round(phi((-1.0 - mu) / c["s"]), 3))
        rows.append(row)
    return rows


# ── figures ─────────────────────────────────────────────────────────────────
def _plume_panel(ax, pat, sid, base, prev_cycle, fc, pfc, obs, cpc, big=False):
    t, M = fc[sid]
    m = pat.meta[sid]
    fs = (1.25 if big else 1.0)
    ax.axhspan(-THR, THR, color="#000", alpha=0.035, lw=0)
    ax.fill_between(t, np.nanpercentile(M, 10, 1), np.nanpercentile(M, 90, 1),
                    color="#2c4a72", alpha=0.16, lw=0, label="p10–p90")
    ax.fill_between(t, np.nanpercentile(M, 25, 1), np.nanpercentile(M, 75, 1),
                    color="#2c4a72", alpha=0.28, lw=0, label="p25–p75")
    ax.plot(t, np.nanmean(M, 1), color="#16335c", lw=2.2 * fs, label="ensemble mean")
    if pfc and sid in pfc:
        pt, pM = pfc[sid]
        keep = (pt >= t[0]) & (pt <= t[-1])
        if keep.any():
            ax.plot(pt[keep], np.nanmean(pM, 1)[keep], ls=(0, (4, 2)), color="#c8781e", lw=1.7 * fs,
                    label=f"previous run mean, init {pd.Timestamp(prev_cycle):%b %d}")
    if sid in obs and len(obs[sid]):
        o = obs[sid]
        ax.plot(o.index, o.values, color="#3f3a33", lw=1.5 * fs, label="GEPS 00Z analysis, same projection")
        ax.plot(o.index, o.values, "o", color="#3f3a33", ms=2.4 * fs)
    if sid in cpc and cpc[sid] is not None:
        c = cpc[sid].loc[base - pd.Timedelta(days=TAIL_DAYS):base + pd.Timedelta(days=2)]
        if len(c):
            ax.plot(c.index, c.values, "x", color="#8a1d1d", ms=3.6 * fs, mew=1.0,
                    label=f"published daily index ({m.get('daily_reference', 'CPC; PSL for EPO/WPO')})")
    if f"{sid}_monthly" in cpc and cpc[f"{sid}_monthly"] is not None:
        c = cpc[f"{sid}_monthly"]
        first = True
        for t0, v in c.items():
            t1 = min(t0 + pd.offsets.MonthEnd(0), base)
            if t1 < base - pd.Timedelta(days=TAIL_DAYS) or t0 > base:
                continue
            ax.plot([t0, t1], [v, v], color="#8a1d1d", lw=1.8 * fs, solid_capstyle="butt",
                    label="published monthly index (CPC), in daily-index sd" if first else None)
            first = False
    ax.axhline(0, color="#555", lw=0.9)
    ax.axvline(base, color="#8a8680", lw=0.8, ls=":")
    ax.set_xlim(base - pd.Timedelta(days=TAIL_DAYS), t[-1] + pd.Timedelta(days=1))
    ylim = max(2.0, float(np.nanmax(np.abs(np.r_[np.nanpercentile(M, [5, 95], 1).ravel(),
                                                   obs[sid].values if sid in obs else 0]))) * 1.1)
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("seasonal sd", fontsize=8.5 * fs)
    ref = m.get("reference", "CPC")
    rt = (f"r vs {ref} daily {m['r_daily']:+.2f}" if "r_daily" in m else
          f"r vs CPC monthly {m['r_monthly']:+.2f}"
          + (f" · vs {m['daily_reference']} daily {m['r_daily_ref']:+.2f}" if "r_daily_ref" in m else "")
          if "r_monthly" in m else
          "no external reference")
    ax.set_title(f"{sid.upper()} — {m['name']}", fontsize=9.6 * fs, fontweight="bold",
                 loc="left", pad=4)
    ax.text(0.985, 0.955, rt, transform=ax.transAxes, ha="right", va="top", fontsize=6.9 * fs,
            color=MUTED, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.tick_params(labelsize=8 * fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10 if not big else 5))
    ax.grid(alpha=0.25, lw=0.5)


def plot_plumes(pat, cycle, prev_cycle, fc, pfc, obs, cpc):
    """One large plume per index, for the page's index dropdown."""
    base = pd.Timestamp(cycle)
    for sid in [i for i in pat.ids if i in fc]:
        fig = plt.figure(figsize=(12.4, 5.6))
        ax = fig.add_axes([0.06, 0.11, 0.925, 0.79])
        _plume_panel(ax, pat, sid, base, prev_cycle, fc, pfc, obs, cpc, big=True)
        ax.legend(fontsize=8, ncol=3, loc="lower left", framealpha=0.92)
        fig.suptitle(f"GEPS extended ensemble, 21 members · init {base:%Y-%m-%d} 00Z"
                     + (f" · dashed: previous run {pd.Timestamp(prev_cycle):%b %d}" if prev_cycle else ""),
                     fontsize=10.5, y=0.985, va="top", color=INK)
        fig.text(0.5, 0.012, "seasonal standard deviations of the daily index (std within ±15 d of the date, 1991–2020); "
                 "shaded band ±0.5 sd = neutral"
                 + ("; anomaly against the NCEP climatology (no 100 hPa hindcast to de-drift against)"
                    if pat.meta[sid]["field"] == "z100" else
                    "; forecast de-drifted against the lead-matched GEPS hindcast climatology"),
                 ha="center", va="bottom", fontsize=8, color=MUTED)
        for d in (FIGS, SITE):
            fig.savefig(d / f"geps_telecon_plume_{sid}.webp", dpi=105, facecolor="white",
                        pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
    print(f"  geps_telecon_plume_*.webp ({sum(1 for i in pat.ids if i in fc)})")


def plot_series(pat, cycle, prev_cycle, fc, pfc, obs, cpc):
    base = pd.Timestamp(cycle)
    ids = [i for i in pat.ids if i in fc]
    ncol = 4
    nrow = (len(ids) + ncol - 1) // ncol
    fig = plt.figure(figsize=(17.0, 3.15 * nrow + 0.9))
    L0, R0, T0, B0 = 0.038, 0.995, 0.905, 0.06
    PW = (R0 - L0) / ncol
    PH = (T0 - B0) / nrow
    for i, sid in enumerate(ids):
        col, row = i % ncol, i // ncol
        ax = fig.add_axes([L0 + col * PW + 0.028, T0 - (row + 1) * PH + 0.055,
                           PW * 0.86, PH * 0.74])
        _plume_panel(ax, pat, sid, base, prev_cycle, fc, pfc, obs, cpc)
        if i == 0:
            ax.legend(fontsize=6.4, ncol=2, loc="lower left", framealpha=0.92)
    fig.suptitle(f"Teleconnection indices — GEPS extended ensemble, 21 members · "
                 f"init {base:%Y-%m-%d} 00Z"
                 f"{'  ·  dashed: previous run ' + pd.Timestamp(prev_cycle).strftime('%b %d') if prev_cycle else ''}",
                 fontsize=13.5, fontweight="bold", y=0.985, va="top")
    fig.text(0.5, 0.012, "seasonal standard deviations of the daily index (std within ±15 d of the date, 1991–2020; NCEP/NCAR R1, CPC-style "
             "patterns); shaded band ±0.5 sd = neutral; forecast anomalies are de-drifted against the "
             "lead-matched GEPS hindcast climatology (NAM 100: plain anomaly vs NCEP, no 100 hPa hindcast)",
             ha="center", va="bottom", fontsize=8.2, color=MUTED)
    for d in (FIGS, SITE):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "geps_telecon_series.webp", dpi=105, facecolor="white",
                    pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print("  geps_telecon_series.webp")


def has_hindcast(row):
    return "skill_r" in row


def plot_probs(pat, cycle, prev_cycle, weeks):
    from matplotlib.patches import Rectangle
    base = pd.Timestamp(cycle)
    ids = [i for i in pat.ids if i in weeks]
    n = len(ids)
    has_cal = any("skill_r" in r for sid in ids for r in weeks[sid])
    fig = plt.figure(figsize=(13.6, 0.80 * n + 2.6))
    ax = fig.add_axes([0.205, 0.075, 0.765, 0.79])
    W = len(WEEKS)
    bw = 0.78

    def bar(y, h, pn, pp, x0, txt=True):
        pz = max(0.0, 1.0 - pn - pp)
        ax.barh(y, pn * bw, left=x0, height=h, color="#2c5fa8", lw=0)
        ax.barh(y, pz * bw, left=x0 + pn * bw, height=h, color="#d9d6d0", lw=0)
        ax.barh(y, pp * bw, left=x0 + (pn + pz) * bw, height=h, color="#b4453c", lw=0)
        if txt:
            for frac, off in ((pn, pn / 2), (pp, pn + pz + pp / 2)):
                if frac >= 0.12:
                    ax.text(x0 + off * bw, y, f"{frac*100:.0f}%", ha="center", va="center",
                            fontsize=7.4, color="white", fontweight="bold")

    for r_, sid in enumerate(ids):
        y = n - 1 - r_
        for row in weeks[sid]:
            x = row["week"] - 1
            x0 = x - bw / 2
            bar(y + 0.10, 0.44, row["p_neg"], row["p_pos"], x0, txt=not row.get("no_skill"))
            dm = f"  Δ{row['mean'] - row['prev_mean']:+.1f}" if "prev_mean" in row else ""
            line = f"mean {row['mean']:+.1f}{dm}"
            if "skill_r" in row:
                bar(y - 0.24, 0.16, row["cal_p_neg"], row["cal_p_pos"], x0, txt=False)
                ax.text(x0 + bw + 0.02, y - 0.24, f"{row['cal_p_neg']*100:.0f}/{row['cal_p_pos']*100:.0f}",
                        ha="left", va="center", fontsize=7.0, color=INK)
                line += f"  ·  skill r {row['skill_r']:+.2f}"
                if row.get("no_skill"):
                    # the raw bar still shows what the members say; the hatch says not to trust it
                    ax.add_patch(Rectangle((x0 - 0.02, y - 0.36), bw + 0.04, 0.72, facecolor="white",
                                           alpha=0.55, edgecolor="#8a8680", hatch="////", lw=0.6, zorder=5))
                    ax.text(x, y + 0.10, f"no skill  ({row['p_neg']*100:.0f}% / {row['p_pos']*100:.0f}%)",
                            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK, zorder=6,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
            elif not has_hindcast(row):
                ax.text(x0 + bw / 2, y - 0.24, "no hindcast — uncalibrated", ha="center", va="center",
                        fontsize=6.8, color=MUTED, style="italic")
            ax.text(x, y - 0.43, line, ha="center", va="center", fontsize=7.3, color=INK)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.7, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{sid.upper()}  {pat.meta[sid]['name'].replace('Northern Annular Mode, 100 hPa', 'N. Annular Mode 100 hPa')}"
                        for sid in ids][::-1], fontsize=8.8)
    ax.set_xticks(range(W))
    ax.set_xticklabels([f"Week {k+1}\n{base + pd.Timedelta(days=w0 - 1):%b %d} – "
                        f"{base + pd.Timedelta(days=w1 - 1):%b %d}" for k, (w0, w1) in enumerate(WEEKS)],
                       fontsize=8.8)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    from matplotlib.patches import Patch
    handles = [Patch(color="#2c5fa8", label=f"P(weekly mean ≤ −{THR} sd)"),
               Patch(color="#d9d6d0", label="neutral"),
               Patch(color="#b4453c", label=f"P(weekly mean ≥ +{THR} sd)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.59, 0.945), ncol=3,
               fontsize=8.4, frameon=False)
    fig.suptitle(f"Teleconnection phase probabilities by week — GEPS 21 members · init {base:%Y-%m-%d}"
                 + (f"  ·  Δ = change in weekly mean vs the {pd.Timestamp(prev_cycle):%b %d} run"
                    if prev_cycle else ""),
                 fontsize=12.5, fontweight="bold", y=0.985, va="top")
    note = ("upper bar: raw member fractions (21 members → 5% steps; 0% = no member, not certainty)"
            + ("; index in seasonal sd for the time of year\nthin lower bar: hindcast-calibrated probabilities — observed weekly index regressed on the forecast one over\n"
               f"2001–2020 GEPS8 starts within ±45 d of this date; skill r = anomaly correlation of the hindcast mean; "
               f"hatched = r below {SKILL_MIN}, no usable skill" if has_cal else ""))
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7.8, color=MUTED)
    for d in (FIGS, SITE):
        fig.savefig(d / "geps_telecon_probs.webp", dpi=105, facecolor="white",
                    pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print("  geps_telecon_probs.webp")


def previous_cycle(cycle):
    have = sorted({q.name.split("_")[-1][:8] for q in LIVE.glob("geps_members_zg500_*.nc")}
                  | set(archive.cycles("geps_members_zg500")))
    older = [c for c in have if c < cycle]
    return older[-1] if older else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--prev")
    a = ap.parse_args()
    os.environ.setdefault("GEPS_CYCLE", a.cycle)
    paths.ensure()
    pat = Patterns()
    base = pd.Timestamp(a.cycle)
    fc = forecast(pat, a.cycle)
    if not fc:
        print("no member files for this cycle — run forecast.py first"); return 1
    prev_cycle = a.prev or previous_cycle(a.cycle)
    pfc = forecast(pat, prev_cycle) if prev_cycle else None
    print(f"cycle {a.cycle}, previous {prev_cycle or 'none'}, {len(fc)} indices", flush=True)
    obs = observed(pat, base)
    cpc = {}
    for sid, u in CPC_URL.items():
        refresh(f"https://ftp.cpc.ncep.noaa.gov/{u}", CPC / f"{sid}.daily.csv")
        c = cpc_daily(sid)
        cpc[sid] = None if c is None else c * pat.to_seasonal(sid, c.index.dayofyear.values)
    for sid in ("epo", "wpo"):
        refresh(f"https://downloads.psl.noaa.gov/Public/map/teleconnections/{sid}.reanalysis.t10trunc.1948-present.txt",
                TC / "psl" / f"{sid}.daily.txt")
        ps = psl_daily(sid)
        if ps is not None:                       # PSL publishes metres: annual sd, then seasonal
            ps = ps / ps.loc["1991":"2020"].std()
            cpc[sid] = ps * pat.to_seasonal(sid, ps.index.dayofyear.values)
    for sid, (url, fname) in CPC_MONTHLY_URL.items():     # SOI / EQSOI: CPC monthly, standardized
        if sid not in pat.meta:
            continue
        refresh(url, CPC / fname)
        try:
            cm = cpc_soi_monthly() if sid == "soi" else cpc_eqsoi_monthly()
        except Exception as e:                                       # noqa: BLE001
            print(f"  ({sid} monthly overlay failed: {str(e)[:60]})"); continue
        cm = cm.loc[base - pd.Timedelta(days=TAIL_DAYS + 31):]
        mid = (cm.index + pd.Timedelta(days=14)).dayofyear.values
        cpc[f"{sid}_monthly"] = cm * pat.meta[sid]["monthly_sd"] * pat.to_seasonal(sid, mid)
    if "soi" in pat.meta:                                            # Long Paddock daily SOI
        refresh(LONGPADDOCK_URL, TC / "longpaddock" / "DailySOI1887-1989Base.txt")
        lp = longpaddock_daily_soi()
        if lp is not None:
            lp = lp / lp.loc["1991":"2020"].std()
            cpc["soi"] = lp * pat.to_seasonal("soi", lp.index.dayofyear.values)
    cal = calibration(pat, base)
    print(f"  hindcast calibration: {'yes, ' + str(len(cal)) + ' indices' if cal else 'none (run telecon_hindcast.py)'}")
    weeks = {sid: weekly_stats(t, M, base, pfc.get(sid) if pfc else None, cal.get(sid))
             for sid, (t, M) in fc.items()}
    for sid in pat.ids:
        if sid not in weeks:
            continue
        w = weeks[sid]
        print(f"  {sid:5s} " + "  ".join(f"wk{r['week']} {r['mean']:+.1f} "
                                          f"[{r['p_neg']*100:.0f}/{r['p_pos']*100:.0f}]"
                                          + (f" Δ{r['mean']-r['prev_mean']:+.1f}" if "prev_mean" in r else "")
                                          + (f" r{r['skill_r']:+.2f} cal[{r['cal_p_neg']*100:.0f}/{r['cal_p_pos']*100:.0f}]"
                                             if "skill_r" in r else "")
                                          for r in w), flush=True)
    plot_series(pat, a.cycle, prev_cycle, fc, pfc, obs, cpc)
    plot_plumes(pat, a.cycle, prev_cycle, fc, pfc, obs, cpc)
    plot_probs(pat, a.cycle, prev_cycle, weeks)
    # numbers for the page
    js = {"cycle": a.cycle, "prev_cycle": prev_cycle,
          "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
          "threshold": THR, "skill_min": SKILL_MIN,
          "units": "seasonal sd: std of the 1991-2020 daily index within +-15 d of the day of year",
          "n_members": int(next(iter(fc.values()))[1].shape[1]),
          "indices": {}}
    for sid in [i for i in pat.ids if i in fc]:
        t, M = fc[sid]
        m = pat.meta[sid]
        rec = {"name": m["name"], "cpc": m.get("cpc"), "field": m["field"],
               "reference": m.get("reference", "CPC" if m.get("cpc") else None),
               "has_hindcast": any("skill_r" in r for r in weeks[sid]),
               "defined_months": sorted(pat.defined[sid]),
               "r_daily": m.get("r_daily"), "r_monthly": m.get("r_monthly"),
               "r_daily_ref": m.get("r_daily_ref"), "daily_reference": m.get("daily_reference"),
               "weeks": weeks[sid],
               "daily": {"t": [x.strftime("%Y-%m-%d") for x in t],
                         "mean": np.round(np.nanmean(M, 1), 2).tolist(),
                         "p10": np.round(np.nanpercentile(M, 10, 1), 2).tolist(),
                         "p90": np.round(np.nanpercentile(M, 90, 1), 2).tolist()}}
        if sid in obs and len(obs[sid]):
            o = obs[sid].dropna()
            rec["obs"] = {"t": [x.strftime("%Y-%m-%d") for x in o.index],
                          "v": np.round(o.values, 2).tolist()}
        js["indices"][sid] = rec
    for d in (TC, SITE):
        d.mkdir(parents=True, exist_ok=True)
        (d / "telecon.json").write_text(json.dumps(js))
    print("  telecon.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
