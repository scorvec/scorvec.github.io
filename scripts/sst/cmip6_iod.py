#!/usr/bin/env python3
"""Positive Indian Ocean Dipole and super El Nino in CMIP6 (large-sample test, report only).

How often a super El Nino comes with a positive IOD, whether the IOD holds the raw RMM in phases 8-1-2 (as in the autumns
of 1982, 1997 and 2026), whether the models make the IOD term of the MJO page's cleaned index (enso_rmm.py), and what the
IOD adds to the El Nino teleconnections. 441 historical runs of 22 models (the ENSO-impact study's 429 and the MJO
screen's 63, reference/cmip6_iod_members.csv), 1950-2014, from the Pangeo CMIP6 archive on Google Cloud (anonymous;
Eyring et al. 2016, CC BY 4.0):

  ts       Amon surface temperature of the ocean (native cells whose centre is not on Natural Earth land and that are
           not sea ice; cmip6_enso_flavour's rule), area-overlap means on the 2.5-degree strip 20S-20N: Nino-3.4
           (5S-5N, 170-120W) and the DMI (Saji et al. 1999: 10S-10N 50-70E minus 10S-0 90-110E) - the models' sea
           surface temperature, as the site's other CMIP6 ENSO work reads it.
  fields   Amon pr (mm/day), tas (K) and psl (hPa; two runs have none), September to February, area-overlap means on
           the 2.5-degree grid 60S-75N.
  RMM      the daily band reductions of the 63 MJO-screened runs (release mjo-impacts-ref-v1, cmip6_mjo_extract.py),
           1979-2014: the raw RMM (previous 120-day mean removed, ENSO kept; mjo_impacts.rmm_from_band), three-field
           and wind-only.
Anomalies: every monthly value against the run's own mean of that calendar month in the other years within +-10 years
(cmip6_enso_flavour.local_anom), so the forced warming (fastest in the western Indian Ocean) never enters the DMI.
IOD year Y: the indices of September-November Y, the fields from September Y to February Y+1 (Y = 1950-2013), each
season day-weighted on the run's own calendar. ENSO-free IOD z: a run's SON DMI minus its regression on SON Nino-3.4,
in that run's standard deviations. Super El Nino: DJF Nino-3.4 >= 2.0 K; positive IOD: SON DMI >= 1 sd of the run.

Tests (each model counts once; across-model one-sample t-test, Benjamini-Hochberg FDR 10 % over the map or the regions
AND >= 80 % of the models agreeing on the sign: cmip6_enso_flavour.robust; fewer than 5 models: not tested)
  frequency  per model, the share of super El Ninos with a positive IOD against the share in neutral years
             (|DJF Nino-3.4| < 0.5 K), and the ENSO-free IOD of its super El Ninos.
  impacts    per model, members pooled: OLS of every grid point on [1, N34, N34^2, z, z * max(N34_DJF, 0)] (N34 of the
             same months): `iod` = z's coefficient (per sd), `interaction` = how much more it does per K of El Nino,
             `iod_in_super` = iod + S x interaction with S the observed super El Ninos' mean DJF Nino-3.4. All models,
             and the credible ones (SON DMI sd within 0.6-1.5 of ERSST v6's, r with SON Nino-3.4 >= 0.35).
             Regions: land-weighted means (pr, tas) or cos-weighted (psl) of each model's coefficients.
  RMM        per model, members pooled: the September 15-November 30 mean RMM regressed on SON N34 and z; the share of
             autumns whose RMM stands (mean vector >= 0.9 and |mean| / mean |RMM| >= 0.7) in super El Ninos with and
             without a positive IOD and in neutral years (shares pooled, a 2000-draw bootstrap over models); and the
             MJO page's correction rebuilt in the model on the wind-only index: c = slope on dN34 over every day,
             d = slope of what is left on dDMI over June-December (0 in January-May), dN34 and dDMI the daily-
             interpolated monthly indices minus their previous 120-day mean, against the site's applied c and d.
Observed values (ERSST v6 DMI and Nino-3.4: scripts/mjo/data/reference/dmi_ersst6.csv, nina34.anom.data) are computed
the same way.

    python scripts/sst/cmip6_iod.py members                     # reference/cmip6_iod_members.csv (Pangeo catalogue)
    python scripts/sst/cmip6_iod.py extract --model M --out DIR [--max-members N] [--workers 2]
    python scripts/sst/cmip6_iod.py rmm --model M --src DIR --band DIR --out DIR
    python scripts/sst/cmip6_iod.py build --src DIR --maps FILE [--require-all]   # -> reference/cmip6_iod_tests.json
All but `members` run in .github/workflows/cmip6-iod.yml.
"""

from __future__ import annotations

import pyproj  # noqa: F401  before xarray: the land mask goes through cartopy (libproj load order)
import argparse
import datetime as dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
MJO = HERE.parent / "mjo"
sys.path.insert(0, str(MJO / "src"))

import cmip6_enso_flavour as EF
import cmip6_pacific_ts_extract as PT

CAT = HERE.parent / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"  # the Pangeo catalogue (members only)
REF = HERE / "reference"
MEMBERS = REF / "cmip6_iod_members.csv"
TESTS = REF / "cmip6_iod_tests.json"
MJO_REF = MJO / "data" / "reference"
STRIP = (np.arange(-20.0, 20.01, 2.5), np.arange(0.0, 357.51, 2.5))
GLOBE = (np.arange(-60.0, 75.01, 2.5), np.arange(0.0, 357.51, 2.5))
MONTHS = tuple(range(1, 13))
FIELD_MONTHS = (9, 10, 11, 12, 1, 2)
CAL_YEARS = np.arange(1949, 2015)  # the calendar years of the record, 1949-12 .. 2014-12
YEARS = np.arange(1950, 2014)  # IOD years
RMM_YEARS = np.arange(1979, 2014)
BOXES = {"n34": (-5, 5, 190, 240), "west": (-10, 10, 50, 70), "east": (-10, 0, 90, 110)}  # lat0, lat1, lon0, lon1
PERIODS = {
    "SON": (9, 10, 11),
    "OND": (10, 11, 12),
    "DJF": (12, 1, 2),
    **{m: (k,) for m, k in (("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2))},
}
VARS = {"pr": ("pr", 86400.0), "tas": ("tas", 1.0), "psl": ("psl", 0.01)}  # store variable, scale to mm/day, K, hPa
PACK = {
    "ts": (273.15, 100.0),
    "pr": (0.0, 100.0),
    "tas": (273.15, 100.0),
    "psl": (1000.0, 100.0),
}  # int16 offset, scale
RANGE = {"pr": (-0.01, 150.0), "tas": (180.0, 330.0), "psl": (850.0, 1150.0)}  # plausible monthly means
SUPER, NEUTRAL, POS_IOD = 2.0, 0.5, 1.0
OBS_SUPER = (1982, 1997, 2015, 2023)
CREDIBLE_SD, CREDIBLE_R = (0.6, 1.5), 0.35
MIN_EVENTS = 8
STAND = (0.9, 0.7)
TERMS = ("iod", "interaction", "iod_in_super")
N_BOOT = 2000
LAND_REGIONS = {  # lat0, lat1, lon0, lon1 (degrees E, 0-360): the Indian Ocean rim, the ENSO-impact study's 20, Brazil
    "East Africa": (-5, 5, 35, 50),
    "South India / Sri Lanka": (5, 15, 75, 82),
    "Sumatra / Java": (-10, 0, 95, 115),
    "Southeast Australia": (-40, -30, 140, 152),
    **{k: (a, b, c % 360, e % 360) for k, (a, b, c, e) in EF.REGIONS.items()},
    "Centre-West Brazil": (-20, -10, 300, 315),
    "Southeast Brazil": (-25, -18, 310, 320),
}
SEA_REGIONS = {"South of Australia": (-50, -35, 110, 160), "Aleutian Low": (40, 60, 160, 220)}  # psl
DOMAINS = {
    "Indian Ocean rim": (-35, 30, 30, 120),
    "Australia": (-45, -10, 110, 155),
    "South America": (-57, 13, 275, 330),
    "North America": (15, 75, 190, 310),
}
_LAND_LOCK = threading.Lock()  # the prepared Natural Earth geometry is shared by the member threads


# ─────────────────────────────────────────────────────────────────────────── shared helpers
def box_mean(f: np.ndarray, lat: np.ndarray, lon: np.ndarray, b) -> np.ndarray:
    """(..., lat, lon) -> (...) cos-weighted mean of the finite cells whose centres lie in the box."""
    j, i = (lat >= b[0]) & (lat <= b[1]), (lon >= b[2]) & (lon <= b[3])
    x = f[..., j, :][..., i]
    w = np.broadcast_to(np.cos(np.deg2rad(lat[j]))[:, None], x.shape[-2:])
    ok = np.isfinite(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(ok, x, 0.0) * w).sum((-2, -1)) / (ok * w).sum((-2, -1))


def by_month(x: np.ndarray, labels, months=MONTHS, years=CAL_YEARS) -> np.ndarray:
    """(M, T, ...) monthly values labelled 'YYYY-MM' -> (M, years, months, ...); NaN where a month is absent."""
    lab = pd.Series(labels)
    y = lab.str.slice(0, 4).astype(int).values - years[0]
    m = lab.str.slice(5, 7).astype(int).values
    keep = np.isin(m, months) & (y >= 0) & (y < len(years))
    slot = np.array([months.index(v) for v in m[keep]])
    out = np.full((x.shape[0], len(years), len(months), *x.shape[2:]), np.nan, "float32")
    out[:, y[keep], slot] = x[:, keep]
    return out


def local_monthly(G: np.ndarray) -> np.ndarray:
    """(M, Y, S, ...) -> anomalies from the same calendar month of the other years within +-10 (EF.local_anom)."""
    return np.stack([EF.local_anom(G[:, :, s].astype("float64")).astype("float32") for s in range(G.shape[2])], 2)


def period_mean(A: np.ndarray, ND: np.ndarray, months, slots, years=YEARS, base=CAL_YEARS[0]) -> np.ndarray:
    """(M, Ycal, S, ...) anomalies and day counts -> (M, len(years), ...) day-weighted mean over the period's months of
    each IOD year (January and February from the next calendar year); NaN with a month missing."""
    total = days = 0.0
    for mm in months:
        yi = years - base + (mm < 9)
        a, w = A[:, yi, slots[mm]], ND[:, yi, slots[mm]]
        w = w.reshape(w.shape + (1,) * (a.ndim - w.ndim))
        total, days = total + a * w, days + w
    return total / days


def enso_free(dmi: np.ndarray, n34: np.ndarray) -> np.ndarray:
    """(M, Y) SON DMI and Nino-3.4 -> each member's DMI minus its OLS on Nino-3.4, in the member's own sd."""
    z = np.full_like(dmi, np.nan)
    for k in range(len(dmi)):
        ok = np.isfinite(dmi[k]) & np.isfinite(n34[k])
        r = dmi[k] - np.polyval(np.polyfit(n34[k, ok], dmi[k, ok], 1), n34[k])
        z[k] = r / np.nanstd(r[ok])
    return z


def unpack(q: np.ndarray, var: str) -> np.ndarray:
    off, sc = PACK[var]
    return np.where(q == -32768, np.nan, q / sc + off).astype("float32")


def num(x, nd: int = 3):
    """A JSON number (None for NaN)."""
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def across(v: np.ndarray) -> dict:
    """One value per model -> mean, across-model t-test p, share agreeing with the mean's sign."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < EF.MIN_MODELS:
        return {"n_models": len(v), "tested": False}
    p = stats.ttest_1samp(v, 0.0).pvalue
    agree = max((v > 0).mean(), (v < 0).mean())
    return {
        "n_models": len(v),
        "mean": num(v.mean()),
        "p": num(p, 4),
        "agree": num(agree, 2),
        "significant": bool(p < EF.ALPHA and agree >= EF.AGREE),
    }


# ─────────────────────────────────────────────────────────────────────────── members (offline, once)
def cmd_members(a) -> int:
    c = pd.read_csv(CAT)
    c = c[(c.experiment_id == "historical") & (c.table_id == "Amon") & c.variable_id.isin(["ts", *VARS])]
    st = c.sort_values("version").groupby(["source_id", "member_id", "variable_id"]).zstore.last().unstack()
    enso = pd.read_csv(REF / "cmip6_enso_members.csv")
    mjo = pd.read_csv(MJO_REF / "cmip6_mjo_members.csv")
    mj = set(zip(mjo.model, mjo.member))
    want = sorted(set(zip(enso.source_id, enso.member_id)) | mj)
    st = st.reindex(pd.MultiIndex.from_tuples(want, names=["source_id", "member_id"]))
    out = st.dropna(subset=["ts", "tas", "pr"])[["ts", "tas", "pr", "psl"]].fillna("").reset_index()
    out["mjo"] = [k in mj for k in zip(out.source_id, out.member_id)]
    out.to_csv(MEMBERS, index=False)
    print(
        f"wrote {MEMBERS.name}: {len(out)} of {len(want)} runs, {out.source_id.nunique()} models, "
        f"{int((out.psl != '').sum())} with psl, {int(out.mjo.sum())} MJO-screened"
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────── extract (one model, in Actions)
def ts_strip(store: str) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """Monthly ocean surface temperature (K) on the strip: area-overlap mean of the native cells whose centre is not on
    Natural Earth land and that are not sea ice."""
    import shapely

    v = EF._open(store, "ts").sel(lat=slice(-24, 24))
    months, nd = EF._calendar(v)
    lat, lon = v.lat.values.astype(float), v.lon.values.astype(float)
    x = v.values.astype("float64")
    X, Y = np.meshgrid(np.where(lon > 180, lon - 360, lon), lat)
    with _LAND_LOCK:
        ocean = ~shapely.contains_xy(PT.land_geom(), X.ravel(), Y.ravel()).reshape(X.shape)
    f = EF.regrid(x, *EF.overlap_ops(lat, lon, *STRIP), np.isfinite(x) & (x >= PT.ICE) & ocean[None])
    for k, b in BOXES.items():
        s = box_mean(f, *STRIP, b)
        if not np.isfinite(s).all() or s.min() < 285 or s.max() > 310:
            raise ValueError(f"{k} box ts outside 285..310 K or missing: {np.nanmin(s):.1f}..{np.nanmax(s):.1f}")
    return months, nd, f, str(v.time.dt.calendar)


def field_months(store: str, var: str, scale: float) -> np.ndarray:
    """One Amon field in September-February on the 60S-75N grid (area-overlap means of every native cell)."""
    v = EF._open(store, VARS[var][0])
    months, _ = EF._calendar(v)
    keep = np.flatnonzero(np.isin([int(m[5:7]) for m in months], FIELD_MONTHS))
    v = v.isel(time=keep).sel(lat=slice(GLOBE[0][0] - 6, GLOBE[0][-1] + 6))
    f = EF.regrid(v.values.astype("float64") * scale, *EF.overlap_ops(v.lat.values, v.lon.values, *GLOBE))
    lo, hi = RANGE[var]
    if not np.isfinite(f).all() or f.min() < lo or f.max() > hi:
        raise ValueError(f"{var} outside {lo}..{hi} or missing: {np.nanmin(f):.2f}..{np.nanmax(f):.2f}")
    return f


def extract_member(row, tries: int = 3) -> tuple[str, dict | str]:
    name = f"{row.source_id}_{row.member_id}"
    err = ""
    for attempt in range(tries):
        try:
            months, nd, ts, cal = ts_strip(row.ts)
            out = {"months": months, "ndays": nd, "calendar": cal, "ts": EF._i16(ts, *PACK["ts"])}
            for var, (_, scale) in VARS.items():
                store = getattr(row, var)
                if isinstance(store, str) and store:
                    out[var] = EF._i16(field_months(store, var, scale), *PACK[var])
            return name, out
        except Exception as e:  # noqa: BLE001  one bad store must not stop the model
            err = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  {name} attempt {attempt + 1}: {err}", flush=True)
            time.sleep(15)
    return name, err


def cmd_extract(a) -> int:
    rows = pd.read_csv(MEMBERS)
    rows = rows[rows.source_id == a.model]
    if rows.empty:
        raise SystemExit(f"{a.model}: not in {MEMBERS.name}")
    if a.max_members:
        rows = rows.head(a.max_members)
    t0 = time.time()
    PT.land_geom()  # the Natural Earth download once, before the member threads
    print(f"{a.model}: {len(rows)} members ({a.workers} workers)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(extract_member, rows.itertuples()))
    good = [(n, r) for n, r in res if isinstance(r, dict)]
    bad = {n: r for n, r in res if isinstance(r, str)}
    for n, r in bad.items():
        print(f"::warning::{n} skipped: {r}", flush=True)
    if not good:
        raise SystemExit(f"{a.model}: no member extracted")
    months = good[0][1]["months"]
    if any(r["months"] != months for _, r in good):
        raise SystemExit(f"{a.model}: members on different time axes")
    fm = [m for m in months if int(m[5:7]) in FIELD_MONTHS]
    psl = [(n, r) for n, r in good if "psl" in r]
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"fields_{a.model}.npz",
        source_id=a.model,
        calendar=",".join(sorted({r["calendar"] for _, r in good})),
        members=np.array([n.rsplit("_", 1)[1] for n, _ in good]),
        months=np.array(months),
        ndays=np.stack([r["ndays"] for _, r in good]).astype("float32"),
        ts=np.stack([r["ts"] for _, r in good]),
        field_months=np.array(fm),
        pr=np.stack([r["pr"] for _, r in good]),
        tas=np.stack([r["tas"] for _, r in good]),
        psl_members=np.array([n.rsplit("_", 1)[1] for n, _ in psl]),
        psl=np.stack([r["psl"] for _, r in psl]) if psl else np.zeros((0, len(fm), 0, 0), "int16"),
        skipped=np.array(sorted(bad)),
    )
    print(
        f"{a.model}: {len(good)} members ({len(psl)} with psl), {len(bad)} skipped, {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────── indices (models and observed)
def model_indices(z) -> dict:
    """One model's monthly box anomalies and IOD-year indices: n34/dmi monthly (M, T) on the record's labels, N34 of
    every period, SON DMI, z (ENSO-free) and zraw (DMI in the member's sd), (M, len(YEARS))."""
    labels = list(z["months"])
    ND = by_month(z["ndays"], labels)
    ts = unpack(z["ts"], "ts")
    yi = np.array([int(s[:4]) for s in labels]) - CAL_YEARS[0]
    mi = np.array([int(s[5:7]) for s in labels]) - 1
    A = {k: local_monthly(by_month(box_mean(ts, *STRIP, b), labels)) for k, b in BOXES.items()}
    dmi = A["west"] - A["east"]
    slots = {m: m - 1 for m in MONTHS}
    out = {
        "members": [str(m) for m in z["members"]],
        "labels": labels,
        "n34m": A["n34"][:, yi, mi],
        "dmim": dmi[:, yi, mi],
        "n34": {p: period_mean(A["n34"], ND, mm, slots) for p, mm in PERIODS.items()},
        "dmi": period_mean(dmi, ND, PERIODS["SON"], slots),
    }
    out["z"] = enso_free(out["dmi"], out["n34"]["SON"])
    out["zraw"] = out["dmi"] / np.nanstd(out["dmi"], axis=1, keepdims=True)
    return out


def _psl_monthly(path: Path) -> pd.Series:
    """PSL's monthly table format (year + 12 values, -99.99 missing) -> a monthly series."""
    rows = {}
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) == 13 and p[0].isdigit():
            rows.update({pd.Period(f"{p[0]}-{m:02d}", "M"): float(v) for m, v in enumerate(p[1:], 1) if float(v) > -99})
    return pd.Series(rows).sort_index()


def observed() -> dict:
    """ERSST v6 SON DMI and Nino-3.4 the models' way (local anomalies over 1950-2026): the DMI sd, its r and slope on
    Nino-3.4 over the models' years, the observed super El Ninos, and S (their mean DJF Nino-3.4)."""
    d = pd.read_csv(MJO_REF / "dmi_ersst6.csv")
    dmi = pd.Series(d.dmi.values, index=pd.PeriodIndex(d.month, freq="M"))
    n34 = _psl_monthly(MJO_REF / "nina34.anom.data")
    idx = pd.period_range("1950-01", min(dmi.index[-1], n34.index[-1]), freq="M")
    years = np.arange(1950, idx[-1].year + 1)
    labels = [str(p) for p in idx]
    ND = by_month(np.array([[p.days_in_month for p in idx]], float), labels, years=years)
    slots = {m: m - 1 for m in MONTHS}
    A = {
        k: local_monthly(by_month(s.reindex(idx).values[None], labels, years=years))
        for k, s in (("n34", n34), ("dmi", dmi))
    }
    iy = np.arange(1950, idx[-1].year)  # IOD years with their February
    son = {k: period_mean(A[k], ND, PERIODS["SON"], slots, iy, 1950)[0] for k in A}
    djf = period_mean(A["n34"], ND, PERIODS["DJF"], slots, iy, 1950)[0]
    fit = (iy >= YEARS[0]) & (iy <= YEARS[-1])
    b = np.polyfit(son["n34"][fit], son["dmi"][fit], 1)
    res = son["dmi"] - np.polyval(b, son["n34"])
    sd, rsd = son["dmi"][fit].std(), res[fit].std()
    sup = {
        str(y): {
            "n34_djf": num(djf[iy == y][0], 2),
            "dmi_son": num(son["dmi"][iy == y][0], 2),
            "zraw": num(son["dmi"][iy == y][0] / sd, 2),
            "z": num(res[iy == y][0] / rsd, 2),
        }
        for y in OBS_SUPER
    }
    return {
        "dmi_son_sd": num(sd),
        "r_n34": num(np.corrcoef(son["n34"][fit], son["dmi"][fit])[0, 1]),
        "slope_on_n34": num(b[0]),
        "super_el_ninos": sup,
        "S": num(np.mean([djf[iy == y][0] for y in OBS_SUPER]), 2),
    }


# ─────────────────────────────────────────────────────────────────────────── rmm (one MJO-screened model, in Actions)
def to_daily(labels, vals: np.ndarray, dates) -> np.ndarray:
    """Monthly values -> the days of the run, linear between mid-months on a calendar-free month axis (360-day and
    noleap dates included; mjo_impacts.daily_n34's)."""
    m = pd.Series(labels)
    mid = m.str.slice(0, 4).astype(int).values * 12 + m.str.slice(5, 7).astype(int).values - 0.5
    d = pd.Series(dates)
    x = (
        d.str.slice(0, 4).astype(int).values * 12
        + d.str.slice(5, 7).astype(int).values
        - 1
        + (d.str.slice(8, 10).astype(int).values - 0.5) / 31.0
    )
    ok = np.isfinite(vals)
    return np.interp(x, mid[ok], vals[ok])


def slope(e: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of e (k, n) on x (n)."""
    xc = x - x.mean()
    return (e - e.mean(1, keepdims=True)) @ xc / (xc @ xc)


def autumn(r1: np.ndarray, r2: np.ndarray, y, mo, day) -> np.ndarray:
    """September 15-November 30 of each RMM year: mean RMM1, RMM2, the mean vector's length and |mean| / mean |RMM|."""
    z = r1 + 1j * r2
    out = np.full((len(RMM_YEARS), 4), np.nan)
    for k, yy in enumerate(RMM_YEARS):
        sel = (y == yy) & (((mo == 9) & (day >= 15)) | (mo == 10) | (mo == 11)) & np.isfinite(z)
        if sel.sum() >= 60:
            mv = z[sel].mean()
            out[k] = mv.real, mv.imag, abs(mv), abs(mv) / np.abs(z[sel]).mean()
    return out


def cmd_rmm(a) -> int:
    import enso_rmm
    import mjo_impacts as mi

    z = np.load(a.src / f"fields_{a.model}.npz")
    idx = model_indices(z)
    rows = pd.read_csv(MEMBERS)
    mjo = set(rows[(rows.source_id == a.model) & rows.mjo].member_id)
    members = [m for m in idx["members"] if m in mjo]
    if not members:
        raise SystemExit(f"{a.model}: no MJO-screened member in fields_{a.model}.npz")
    eofs = mi.load_eofs()
    out = {k: [] for k in ("r3", "rw", "c", "d")}
    for m in members:
        k = idx["members"].index(m)
        r = np.load(a.band / f"{a.model}_{m}_rlut.npz")
        u = np.load(a.band / f"{a.model}_{m}_ua.npz")
        dates, lat = r["dates"], r["lat_band"]
        if not (u["dates"] == dates).all():
            raise SystemExit(f"{a.model} {m}: rlut and ua dates differ")
        y, mo, frac = mi.year_frac(dates, str(r["calendar"]))
        day = pd.Series(dates).str.slice(8, 10).astype(int).values
        t = np.arange(len(dates), dtype=float)
        olr = mi.band_mean(r["rlut_band"].astype(np.float32), lat)
        u8 = mi.band_mean(u["u850_band"].astype(np.float32), lat)
        u2 = mi.band_mean(u["u250_band"].astype(np.float32), lat)
        r3 = np.array(mi.rmm_from_band(olr, u8, u2, frac, t, eofs))
        rw = np.array(mi.rmm_from_band(None, u8, u2, frac, t, eofs))
        n34, dmi = (to_daily(idx["labels"], idx[s][k], dates) for s in ("n34m", "dmim"))
        dn = n34 - mi._prev120(n34)
        dd = np.where(np.isin(mo, enso_rmm.IOD_MONTHS), dmi - mi._prev120(dmi), 0.0)
        ok = np.isfinite(rw).all(0) & np.isfinite(dn) & np.isfinite(dd)
        c = slope(rw[:, ok], dn[ok])
        out["c"].append(c)
        out["d"].append(slope(rw[:, ok] - dn[ok] * c[:, None], dd[ok]))
        out["r3"].append(autumn(*r3, y, mo, day))
        out["rw"].append(autumn(*rw, y, mo, day))
        print(f"  {a.model} {m}: c {out['c'][-1].round(3)}, d {out['d'][-1].round(3)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"rmm_{a.model}.npz",
        members=np.array(members),
        years=RMM_YEARS,
        **{k: np.array(v) for k, v in out.items()},
    )
    print(f"{a.model}: {len(members)} MJO-screened members", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────── build (all models)
def model_impacts(z, idx: dict, S: float) -> tuple[dict, np.ndarray]:
    """One model's coefficient maps {var: {period: {term: (lat, lon)}}} (members pooled) and its SON rain dipole
    (west minus east box, mm/day; (M, len(YEARS)))."""
    labels, fm = list(z["months"]), list(z["field_months"])
    nd = z["ndays"][:, [labels.index(m) for m in fm]]
    slots = {m: k for k, m in enumerate(FIELD_MONTHS)}
    out, dipole = {}, None
    for var in VARS:
        rows = list(range(len(idx["members"])))
        if var == "psl":
            rows = [idx["members"].index(str(m)) for m in z["psl_members"]]
            if not rows:
                continue
        G = local_monthly(by_month(unpack(z[var], var), fm, FIELD_MONTHS))
        ND = by_month(nd[rows], fm, FIELD_MONTHS)
        out[var] = {}
        for p, mm in PERIODS.items():
            Y = period_mean(G, ND, mm, slots)
            if var == "pr" and p == "SON":
                dipole = box_mean(Y, *GLOBE, BOXES["west"]) - box_mean(Y, *GLOBE, BOXES["east"])
            n34p, zf, n34d = (idx["n34"][p][rows], idx["z"][rows], idx["n34"]["DJF"][rows])
            X = np.stack([np.ones_like(zf), n34p, n34p**2, zf, zf * np.maximum(n34d, 0.0)], -1).reshape(-1, 5)
            Yf = Y.reshape(len(X), -1)
            ok = np.isfinite(X).all(1) & np.isfinite(Yf).all(1)
            B = np.linalg.lstsq(X[ok], Yf[ok], rcond=None)[0].reshape(5, *Y.shape[2:])
            out[var][p] = {"iod": B[3], "interaction": B[4], "iod_in_super": B[3] + S * B[4]}
    return out, dipole


def model_summary(idx: dict, dipole: np.ndarray, obs: dict) -> dict:
    """The model's IOD realism and its super El Ninos."""
    n34s, dmi, n34d, zf, zr = idx["n34"]["SON"], idx["dmi"], idx["n34"]["DJF"], idx["z"], idx["zraw"]
    ok = np.isfinite(n34s) & np.isfinite(dmi)
    sd = float(np.nanmean(np.nanstd(dmi, axis=1)))
    r = float(np.corrcoef(n34s[ok], dmi[ok])[0, 1])
    okd = ok & np.isfinite(dipole)
    sup, neu, pos = n34d >= SUPER, np.abs(n34d) < NEUTRAL, zr >= POS_IOD
    return {
        "members": len(idx["members"]),
        "run_years": int(np.isfinite(n34d).sum()),
        "dmi_son_sd": num(sd),
        "sd_ratio": num(sd / obs["dmi_son_sd"], 2),
        "r_n34": num(r),
        "slope_on_n34": num(np.polyfit(n34s[ok], dmi[ok], 1)[0]),
        "r_rain_dipole": num(np.corrcoef(dipole[okd], dmi[okd])[0, 1]),
        "credible": bool(CREDIBLE_SD[0] <= sd / obs["dmi_son_sd"] <= CREDIBLE_SD[1] and r >= CREDIBLE_R),
        "super": int(sup.sum()),
        "super_with_pos_iod": int((sup & pos).sum()),
        "share_pos_iod_super": num((sup & pos).sum() / sup.sum()) if sup.any() else None,
        "share_pos_iod_neutral": num((neu & pos).sum() / neu.sum()) if neu.any() else None,
        "z_mean_super": num(np.nanmean(zf[sup])) if sup.any() else None,
        "zraw_mean_super": num(np.nanmean(zr[sup])) if sup.any() else None,
    }


def frequency(models: dict) -> dict:
    """Across models (those with MIN_EVENTS super El Ninos): positive-IOD share in super El Ninos minus in neutral
    years, and the ENSO-free IOD of the super El Ninos; pooled counts alongside."""
    m = {k: v for k, v in models.items() if v["super"] >= MIN_EVENTS}
    diff = [v["share_pos_iod_super"] - v["share_pos_iod_neutral"] for v in m.values()]
    return {
        "models_tested": sorted(m),
        "runs": sum(v["members"] for v in models.values()),
        "run_years": sum(v["run_years"] for v in models.values()),
        "super_el_ninos": sum(v["super"] for v in models.values()),
        "pooled_share_pos_iod_super": num(
            sum(v["super_with_pos_iod"] for v in models.values()) / max(sum(v["super"] for v in models.values()), 1)
        ),
        "median_share_pos_iod_super": num(np.median([v["share_pos_iod_super"] for v in m.values()])) if m else None,
        "median_share_pos_iod_neutral": num(np.median([v["share_pos_iod_neutral"] for v in m.values()])) if m else None,
        "super_minus_neutral": across(diff),
        "z_in_super": across([v["z_mean_super"] for v in m.values()]),
    }


def impacts(maps: dict, names: list[str], lf: np.ndarray) -> dict:
    """Multi-model tests of the coefficient maps of the `names` models: robust share of each domain (land for pr and
    tas, every cell for psl) and the regional means."""
    lat, lon = GLOBE
    cosw = np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))
    land = lf >= 0.5
    W = {"land": _weights(LAND_REGIONS, lf), "sea": _weights(SEA_REGIONS, np.ones_like(lf))}
    out, stacks = {}, {}
    for var in VARS:
        have = [n for n in names if var in maps[n]]
        if len(have) < EF.MIN_MODELS:
            continue
        regs = SEA_REGIONS if var == "psl" else LAND_REGIONS
        Wr = W["sea"] if var == "psl" else W["land"]
        out[var] = {}
        for p in PERIODS:
            out[var][p] = {}
            for t in TERMS:
                st = np.stack([maps[n][var][p][t] for n in have])
                mean, mask = EF.robust(st)
                stacks[(var, p, t)] = (mean, mask)
                dom = {}
                for d, (a, b, c, e) in DOMAINS.items():
                    inside = ((lat >= a) & (lat <= b))[:, None] & ((lon >= c) & (lon <= e))[None, :]
                    cells = inside & (land if var != "psl" else True)
                    dom[d] = num((mask * cosw * cells).sum() / (cosw * cells).sum(), 3)
                rv = np.einsum("myx,ryx->mr", st, Wr)
                rmean, rmask = EF.robust(rv)
                agree = np.maximum((rv > 0).mean(0), (rv < 0).mean(0))
                out[var][p][t] = {
                    "robust_share": dom,
                    "regions": {
                        k: {"mean": num(rmean[i]), "agree": num(agree[i], 2), "robust": bool(rmask[i])}
                        for i, k in enumerate(regs)
                    },
                }
    return {"models": sorted(names), "tests": out}, stacks


def _weights(regions: dict, lf: np.ndarray) -> np.ndarray:
    """(regions, lat, lon) lf x cos(lat) weights over the GLOBE cells of each box, each summing to 1."""
    lat, lon = GLOBE
    W = np.zeros((len(regions), len(lat), len(lon)))
    for k, (a, b, c, e) in enumerate(regions.values()):
        W[k] = ((lat >= a) & (lat <= b))[:, None] * ((lon >= c) & (lon <= e))[None, :] * lf
        W[k] *= np.cos(np.deg2rad(lat))[:, None]
        if W[k].sum() == 0:
            raise SystemExit(f"region {list(regions)[k]} holds no cell")
        W[k] /= W[k].sum()
    return W


def rmm_tests(rmm: dict, idxs: dict, site: dict) -> dict:
    """The RMM tests over the MJO-screened models."""
    y0 = int(RMM_YEARS[0] - YEARS[0])
    per, bz, bn, cs, ds, counts = {}, {}, {}, {}, {}, {}
    for model, r in rmm.items():
        idx = idxs[model]
        k = [idx["members"].index(str(m)) for m in r["members"]]
        sl = slice(y0, y0 + len(RMM_YEARS))
        n34s, n34d = idx["n34"]["SON"][k, sl], idx["n34"]["DJF"][k, sl]
        zf, zr = idx["z"][k, sl], idx["zraw"][k, sl]
        per[model] = {"members": len(k), "c": [num(v) for v in r["c"].mean(0)], "d": [num(v) for v in r["d"].mean(0)]}
        cs[model], ds[model] = r["c"].mean(0), r["d"].mean(0)
        for name in ("rw", "r3"):
            A = r[name]
            X = np.stack([np.ones_like(zf), n34s, zf], -1).reshape(-1, 3)
            Y = A[..., :2].reshape(-1, 2)
            ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
            B = np.linalg.lstsq(X[ok], Y[ok], rcond=None)[0]
            bz.setdefault(name, {})[model], bn.setdefault(name, {})[model] = B[2], B[1]
            stand = (A[..., 2] >= STAND[0]) & (A[..., 3] >= STAND[1])
            fin = np.isfinite(A[..., 2]) & np.isfinite(n34d)
            cls = {
                "super_pos_iod": fin & (n34d >= SUPER) & (zr >= POS_IOD),
                "super_no_pos_iod": fin & (n34d >= SUPER) & (zr < POS_IOD),
                "neutral": fin & (np.abs(n34d) < NEUTRAL),
            }
            counts.setdefault(name, {})[model] = {c: (int(m.sum()), int((m & stand).sum())) for c, m in cls.items()}
            per[model][name] = {
                "b_z": [num(v) for v in B[2]],
                "b_n34": [num(v) for v in B[1]],
                "standing": {c: list(v) for c, v in counts[name][model].items()},
            }
    out = {
        "per_model": per,
        "site_applied": {"c": [num(v, 4) for v in site["c"]], "d": [num(v, 4) for v in site["d"]]},
        "c": {f"rmm{i + 1}": across([v[i] for v in cs.values()]) for i in range(2)},
        "d": {f"rmm{i + 1}": across([v[i] for v in ds.values()]) for i in range(2)},
    }
    rng = np.random.default_rng(1)
    names = sorted(rmm)
    for name in ("rw", "r3"):
        sec = {
            "autumn_rmm_on_z": {f"rmm{i + 1}": across([v[i] for v in bz[name].values()]) for i in range(2)},
            "autumn_rmm_on_n34": {f"rmm{i + 1}": across([v[i] for v in bn[name].values()]) for i in range(2)},
        }
        C = np.array([[counts[name][m][c] for c in ("super_pos_iod", "super_no_pos_iod", "neutral")] for m in names])

        def shares(Ck):
            n, s = Ck[..., 0].sum(0), Ck[..., 1].sum(0)
            return s / np.maximum(n, 1)

        sh = shares(C)
        boot = np.array([shares(C[rng.integers(0, len(names), len(names))]) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot[:, 0] - boot[:, 1], [2.5, 97.5])
        sec["standing_share"] = {
            "n": dict(zip(("super_pos_iod", "super_no_pos_iod", "neutral"), C[..., 0].sum(0).tolist())),
            "share": dict(zip(("super_pos_iod", "super_no_pos_iod", "neutral"), [num(v) for v in sh])),
            "pos_minus_no_pos_ci95": [num(lo), num(hi)],
            "significant": bool(lo > 0 or hi < 0),
        }
        out["wind_only" if name == "rw" else "three_field"] = sec
    return out


def cmd_build(a) -> int:
    import enso_rmm

    want = sorted(pd.read_csv(MEMBERS).source_id.unique())
    files = {p.name[len("fields_") : -4]: p for p in sorted(a.src.glob("fields_*.npz"))}
    rfiles = {p.name[len("rmm_") : -4]: p for p in sorted(a.src.glob("rmm_*.npz"))}
    mjo_models = sorted(pd.read_csv(MEMBERS).query("mjo").source_id.unique())
    if a.require_all and (set(want) - set(files) or set(mjo_models) - set(rfiles)):
        raise SystemExit(
            f"missing: fields {sorted(set(want) - set(files))}, rmm {sorted(set(mjo_models) - set(rfiles))}"
        )
    obs = observed()
    lf = EF.land_fraction(*GLOBE)
    maps, models, idxs = {}, {}, {}
    for name, p in files.items():
        t0 = time.time()
        z = np.load(p)
        idxs[name] = model_indices(z)
        maps[name], dipole = model_impacts(z, idxs[name], obs["S"])
        models[name] = model_summary(idxs[name], dipole, obs)
        print(
            f"{name}: {models[name]['members']} members, {models[name]['super']} super El Ninos, {time.time() - t0:.0f} s"
        )
    credible = [n for n, v in models.items() if v["credible"]]
    imp_all, st_all = impacts(maps, sorted(models), lf)
    imp_cred, st_cred = impacts(maps, credible, lf)
    rmm = {n: dict(np.load(p)) for n, p in rfiles.items() if n in idxs}
    site = {"c": enso_rmm.load_c(), "d": enso_rmm.load_d()}
    out = {
        "title": "Positive IOD and super El Nino in CMIP6: frequency, the standing RMM and the teleconnections",
        "made": f"{dt.date.today():%Y-%m-%d}",
        "builder": "scripts/sst/cmip6_iod.py",
        "years": f"{YEARS[0]}-{YEARS[-1]} (IOD years; RMM {RMM_YEARS[0]}-{RMM_YEARS[-1]})",
        "units": "impacts per sd of the ENSO-free IOD: pr mm/day, tas K, psl hPa; RMM per sd or per K",
        "observed": obs,
        "models": models,
        "frequency": frequency(models),
        "impacts": {"all": imp_all, "credible": imp_cred},
        "rmm": rmm_tests(rmm, idxs, site) if rmm else None,
    }
    TESTS.write_text(json.dumps(out, indent=1))
    lat, lon = GLOBE
    arrs = {"lat": lat, "lon": lon, "models": np.array(sorted(models)), "credible": np.array(credible)}
    for tag, st in (("all", st_all), ("credible", st_cred)):
        for (var, p, t), (mean, mask) in st.items():
            arrs[f"{tag}_{var}_{p}_{t}_mean"] = mean.astype("float32")
            arrs[f"{tag}_{var}_{p}_{t}_robust"] = mask
    a.maps.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.maps, **arrs)
    print(f"wrote {TESTS} and {a.maps}: {len(models)} models ({len(credible)} credible), {len(rmm)} with the RMM")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("members")
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-members", type=int, default=0)
    e.add_argument("--workers", type=int, default=2)
    r = sub.add_parser("rmm")
    r.add_argument("--model", required=True)
    r.add_argument("--src", type=Path, required=True, help="the folder of fields_<model>.npz")
    r.add_argument("--band", type=Path, required=True, help="the model's rlut and ua band reductions")
    r.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", type=Path, required=True, help="the folder of fields_*.npz and rmm_*.npz")
    b.add_argument("--maps", type=Path, required=True, help="where the multi-model maps go (npz)")
    b.add_argument("--require-all", action="store_true")
    a = ap.parse_args(argv)
    return {"members": cmd_members, "extract": cmd_extract, "rmm": cmd_rmm, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
