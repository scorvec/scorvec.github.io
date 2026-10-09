#!/usr/bin/env python3
"""SEAS5 polar vortex and sudden-stratospheric-warming (SSW) risk, against the model's own normal.

User 2026-10-07: "Is there data available for 10/100mb winds and can we make some new charts comparing these to model
'normal' and come up with an SSW risk index?"

DATA
  * 10 hPa, DAILY. An SSW is a daily event, so the vortex is read from daily zonal-mean zonal wind at 10 hPa:
      forecast  CDS seasonal-original-pressure-levels, u at 10 hPa, 12-hourly steps 12…5160 h, 2° grid 90–30°N,
                51 members (one request per issue); daily mean = the 12 UTC and next 00 UTC steps of each day — the
                same right-closed rule that reproduces the CDS monthly means (seas5_lead7 lead-6 test).
      hindcast  the AWS SEAS5 daily copy (seas5_aws), u at 10 hPa, 25 members × 1981–2016 (36 winters: the full SEAS5
                hindcast, as the regime product uses), zonal mean of every cell row 89.5–29.5°N, interpolated to the
                2° latitudes (60°N = mean of the 60.5 and 59.5 rows). The lead-6 test found its monthly 60°N zonal
                mean identical to the CDS monthly hindcast (rms 5e-5 of the member spread).
  * 100 hPa, MONTHLY only: the AWS copy holds u at 10/200/500/850 hPa, so daily u100 hindcasts would take ~260k CDS
    fields per start month. The 100 hPa product is the CDS monthly mean (forecast and 1993–2016 hindcast, leads 1–6).
  * Observed: MERRA-2 daily u60N at 10 hPa 1980–2026-06 (scripts/strat/reference/strat_history.nc, gap-filled; the
    site's SSW catalogue strat_history_events.json is the cross-check) and the GEOS FP analysis tail for this season
    (reference/upperstrat_tail_seed.json + assets/sst/data/upperstrat_tail.json, u60 at 10 and 100 hPa).

SSW DEFINITION (Charlton & Polvani 2007): a major SSW is a day between 1 November and 31 March on which the daily
zonal-mean zonal wind at 60°N and 10 hPa turns easterly; a second event needs 20 consecutive westerly days after the
previous one; an event whose wind does not return to westerly for at least 10 consecutive days before 30 April is a
final warming, not an SSW. The final warming is the last westerly→easterly transition with no such return.

RISK INDEX: the share of members with at least one major SSW in Nov–Mar (and with an SSW onset in each month), set
against (1) the model's own base rate (hindcast, same start month, same detection) and (2) the observed rate
(MERRA-2, the same 36 winters and 1980–2025). SKILL: the hindcast's yearly member fraction is verified against the
observed occurrence in MERRA-2: Brier skill score against the leave-one-out climatology, reliability, ROC area, all
with moving-block bootstrap intervals over years (blocks of 3 winters, 5000 resamples). The page states the index
carries no skill beyond climatology unless the BSS interval excludes zero. ENSO context: SSW rates in El Niño
winters (observed DJF ONI ≥ 0.5) vs the rest, hindcast and observed, permutation tests over years.

    python seas5_ssw.py hindcast --month 10      # AWS → data/seas5/ssw/aws_u10_{MM}.npz (Actions, Python 3.12)
    python seas5_ssw.py fetch --issue 202610     # CDS forecast u10 daily + u100 monthly (Actions)
    python seas5_ssw.py build --issue 202610     # figures + data/seas5_ssw.json
"""
from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray/cartopy load another libproj
import argparse
import calendar
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import ASSETS, CENTRE, CLIM_YEARS, DATA, SYSTEM  # noqa: E402
import seas5_ref as R  # noqa: E402

DIR = DATA / "ssw"
REPO = HERE.parents[1]
STRAT_REF = REPO / "scripts" / "strat" / "reference"
OUT_JSON = ASSETS / "data" / "seas5_ssw.json"
HC_YEARS = list(range(1981, 2017))           # every SEAS5 hindcast start on the AWS copy (36 winters)
NDAYS = 215
SEC_LATS = np.arange(90.0, 29.0, -2.0)       # 90 … 30 °N, the forecast request's 2° rows
HOURS = [str(h) for h in range(12, 5161, 12)]
NBOOT = 5000
BLOCK = 3


# ── SSW detection ────────────────────────────────────────────────────────────
def detect(u: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """CP07 on one daily u60N10 series → {"ssw": [dates], "fw": date or None}. NaN days count as missing:
    a westerly or easterly run is broken by them, so a series with gaps is judged on what it has."""
    u = np.asarray(u, float)
    west = u >= 0
    n = len(u)
    events = []
    last = None
    for i in range(1, n):
        d = dates[i]
        if not (np.isfinite(u[i]) and np.isfinite(u[i - 1])) or not (west[i - 1] and not west[i]):
            continue
        if not ((d.month >= 11) or (d.month <= 3)):
            continue
        if last is not None:
            run = 0; j = i - 1
            while j > last and west[j] and np.isfinite(u[j]):
                run += 1; j -= 1
            if run < 20:
                continue
        # final-warming exclusion: 10 consecutive westerly days before 30 April
        apr30 = pd.Timestamp(year=d.year if d.month <= 4 else d.year + 1, month=4, day=30)
        run, ok = 0, False
        for j in range(i + 1, n):
            if dates[j] > apr30:
                break
            run = run + 1 if (np.isfinite(u[j]) and west[j]) else 0
            if run >= 10:
                ok = True; break
        if ok:
            events.append(d); last = i
    # final warming: the last westerly→easterly transition with no 10-day westerly return afterwards
    fw = None
    for i in range(n - 1, 0, -1):
        if np.isfinite(u[i]) and np.isfinite(u[i - 1]) and west[i - 1] and not west[i]:
            run, back = 0, False
            for j in range(i + 1, n):
                run = run + 1 if (np.isfinite(u[j]) and west[j]) else 0
                if run >= 10:
                    back = True; break
            if not back:
                fw = dates[i]
            break
    return {"ssw": events, "fw": fw}


# ── hindcast from AWS ────────────────────────────────────────────────────────
def aws_path(month: str) -> Path:
    return DIR / f"aws_u10_{int(month):02d}.npz"


def hindcast(month: str) -> bool:
    """Daily zonal-mean u at 10 hPa, 30–90°N, every member and start year → aws_u10_{MM}.npz [year, member, day, lat]."""
    import seas5_aws
    dest = aws_path(month)
    if dest.exists():
        return True
    t0 = time.time()
    root, _ = seas5_aws.open_root("hindcast")
    g = root["pressure/1platx1plon"]
    a = g["u"]
    it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
    il = [float(x) for x in g["pressure_level"][:]].index(10.0)
    lat = g["lat"][:]
    rows = np.where(lat >= 29.0)[0]                                       # 89.5 … 29.5
    nm = a.shape[0]
    out = np.full((len(HC_YEARS), nm, NDAYS, len(rows)), np.nan, np.float32)

    def one(job):
        yi, y, m0, m1 = job
        k = int(np.where((it.year == y) & (it.month == int(month)) & (it.day == 1))[0][0])
        blk = np.asarray(a[m0:m1, k, :, il, rows[0]:rows[-1] + 1, :], dtype=np.float32)   # (m, day, lat, lon)
        return yi, m0, m1, np.nanmean(blk, axis=-1)

    jobs = [(yi, y, m0, min(m0 + 10, nm)) for yi, y in enumerate(HC_YEARS) for m0 in range(0, nm, 10)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for yi, m0, m1, zm in ex.map(one, jobs):
            out[yi, m0:m1] = zm
    clat = lat[rows]
    # cell rows → the 2° latitudes: a whole-degree latitude is the mean of the two rows around it, 90°N the top row
    sec = np.stack([out[..., 0] if L >= 89.9 else out[..., [int(np.argmin(np.abs(clat - (L + 0.5)))), int(np.argmin(np.abs(clat - (L - 0.5))))]].mean(-1)
                    for L in SEC_LATS], axis=-1)
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, sec=sec.astype(np.float32), lats=SEC_LATS, years=np.array(HC_YEARS), month=int(month))
    print(f"  AWS u10 hindcast start {month}: {sec.shape} in {(time.time() - t0) / 60:.1f} min, finite {np.isfinite(sec).mean():.3f}", flush=True)
    return np.isfinite(sec).mean() > 0.99


def derive_hc(month: str) -> None:
    """hc_ssw_{MM}: the AWS daily section (float32) + every hindcast member-winter's SSW dates and final warming."""
    z = np.load(aws_path(month))
    sec, lats, years = z["sec"], z["lats"], z["years"]
    i60 = int(np.argmin(np.abs(lats - 60)))
    ev = {}
    for yi, y in enumerate(years):
        dates = pd.date_range(f"{y}-{int(month):02d}-01", periods=NDAYS, freq="D")
        for m in range(sec.shape[1]):
            r = detect(sec[yi, m, :, i60], dates)
            ev[f"{y}_{m}"] = {"ssw": [str(d.date()) for d in r["ssw"]], "fw": str(r["fw"].date()) if r["fw"] is not None else None}
    R.save(R.hc_ref("ssw", month), meta=dict(month=int(month), years=[int(years[0]), int(years[-1])], members=int(sec.shape[1]),
                                             source="AWS Planette SEAS5 daily u 10 hPa, zonal mean", events=ev),
           sec=sec.astype(np.float32), lats=lats, years=years)


def u100_hc_path(month: str) -> Path:
    return DIR / f"hc_u100_{int(month):02d}.grib"


def _monthly_u100(years, month: str, dest: Path) -> bool:
    import seas5_lead7 as L7
    req = {"originating_centre": CENTRE, "system": SYSTEM, "product_type": ["monthly_mean"], "variable": ["u_component_of_wind"],
           "pressure_level": ["100"], "year": [str(y) for y in years], "month": [month], "leadtime_month": ["1", "2", "3", "4", "5", "6"],
           "area": [60, -180, 60, 180], "grid": [2.0, 2.0], "data_format": "grib"}
    return L7._cds("seasonal-monthly-pressure-levels", req, dest, f"u100 60N monthly {month} {years[0]}..{years[-1]}")


def derive_u100(month: str) -> None:
    """hc_ssw_u100_{MM}: u60N 100 hPa monthly zonal mean of every hindcast sample [600, 6] (CDS 1993–2016)."""
    from seas5_build import load_field
    v, lat, lon = load_field(u100_hc_path(month), "u", lead7=False)
    R.save(R.hc_ref("ssw_u100", month), meta=dict(month=int(month), years="1993-2016"), u=np.nanmean(v[:, :, 0, :], axis=-1))


REF_UNITS = {"ssw": dict(raw=["ssw:aws"], fn=derive_hc), "ssw_u100": dict(raw=["ssw:u100"], fn=derive_u100)}
RAW_HC = {"ssw:aws": ((lambda month: [aws_path(month)]), (lambda month: hindcast(f"{int(month):02d}"))),
          "ssw:u100": ((lambda month: [u100_hc_path(month)]),
                       (lambda month: _monthly_u100(CLIM_YEARS, f"{int(month):02d}", u100_hc_path(month))))}


# ── forecast ─────────────────────────────────────────────────────────────────
def fc_path(ym: str) -> Path:
    return DIR / f"fc_u10_{ym}.grib"


def fc_u100_path(ym: str) -> Path:
    return DIR / f"fc_u100_{ym}.grib"


def fetch(ym: str, only: str | None = None) -> bool:
    """daily u10 (the SSW product) and/or the monthly u100 forecast; `only` = "u10" | "u100"."""
    import seas5_lead7 as L7
    if only == "u100":
        return _monthly_u100([int(ym[:4])], ym[4:], fc_u100_path(ym))
    req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": ["u_component_of_wind"], "pressure_level": ["10"],
           "year": [ym[:4]], "month": [ym[4:]], "day": ["01"], "leadtime_hour": HOURS,
           "area": [90, -180, 30, 180], "grid": [2.0, 2.0], "data_format": "grib"}
    ok = L7._cds("seasonal-original-pressure-levels", req, fc_path(ym), f"u10 daily 30-90N {ym}")
    if only != "u10":
        ok = _monthly_u100([int(ym[:4])], ym[4:], fc_u100_path(ym)) and ok
    return ok


def forecast_daily(ym: str):
    """→ (sec [member, day, lat] daily-mean zonal-mean u10 on SEC_LATS, dates)."""
    import xarray as xr
    ds = xr.open_dataset(fc_path(ym), engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = ds["u"].transpose("number", "step", "latitude", "longitude").sortby("step")
    hrs = (v.step.values / np.timedelta64(1, "h")).astype(int)
    zm = v.values.astype(np.float32).mean(-1)                           # (member, step, lat)
    lat = ds.latitude.values; ds.close()
    day = (hrs - 1) // 24                                              # 12 h and 24 h → day 0 (right-closed)
    out = np.stack([zm[:, day == d].mean(1) for d in range(NDAYS)], axis=1)
    order = [int(np.argmin(np.abs(lat - L))) for L in SEC_LATS]
    assert np.allclose(lat[order], SEC_LATS)
    return out[..., order], pd.date_range(f"{ym[:4]}-{ym[4:]}-01", periods=NDAYS, freq="D")


# ── observed ─────────────────────────────────────────────────────────────────
def merra2_u60() -> pd.Series:
    import xarray as xr
    ds = xr.open_dataset(STRAT_REF / "strat_history.nc")
    s = pd.Series(ds["u10"].values.astype(float), index=pd.to_datetime(ds["time"].values)); ds.close()
    return s


def geosfp_tail() -> dict:
    """{level: pd.Series} of GEOS FP u60N (seed + rolling tail)."""
    days = {}
    for p in (STRAT_REF / "upperstrat_tail_seed.json", ASSETS / "data" / "upperstrat_tail.json"):
        if p.exists():
            d = json.loads(p.read_text()); levs = d["levels"]
            for k, v in d["days"].items():
                days[k] = dict(zip(levs, v["nh"]["u60"]))
    if not days:
        return {}
    idx = pd.to_datetime(sorted(days))
    return {lev: pd.Series([days[k].get(lev, np.nan) for k in sorted(days)], index=idx) for lev in (10.0, 100.0)}


def oni_djf() -> dict:
    """{winter start year: DJF ONI} from the site's nino_history.json (CPC ERSST-based ONI)."""
    d = json.loads((ASSETS / "data" / "nino_history.json").read_text())
    months, s = d["months"], d["series"]["oni"]
    s = s["anom"] if isinstance(s, dict) else s
    m = dict(zip(months, s))
    out = {}
    for y in range(1970, 2030):
        k = f"{y + 1}-01"                                                # ONI for Jan = the DJF mean
        if m.get(k) is not None:
            out[y] = float(m[k])
    return out


# ── statistics ───────────────────────────────────────────────────────────────
def block_boot_idx(n: int, rng) -> np.ndarray:
    k = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=k)
    return np.concatenate([(np.arange(s, s + BLOCK) % n) for s in starts])[:n]


def verify(p: np.ndarray, o: np.ndarray, rng) -> dict:
    """Brier skill of yearly probabilities p against occurrences o (0/1): BSS vs the leave-one-out climatology, ROC area,
    reliability in three bins, block-bootstrap 90 % intervals."""
    n = len(o)

    def stats(ii):
        pp, oo = p[ii], o[ii]
        clim = (oo.sum() - oo) / (len(oo) - 1)                          # leave-one-out base rate
        bs, bsc = np.mean((pp - oo) ** 2), np.mean((clim - oo) ** 2)
        bss = 1 - bs / bsc if bsc > 0 else np.nan
        pos, neg = pp[oo == 1], pp[oo == 0]
        auc = np.nan if not len(pos) or not len(neg) else (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
        return bss, auc, bs, bsc

    bss, auc, bs, bsc = stats(np.arange(n))
    boot = np.array([stats(block_boot_idx(n, rng)) for _ in range(NBOOT)])
    lo, hi = np.nanpercentile(boot[:, 0], [5, 95]); alo, ahi = np.nanpercentile(boot[:, 1], [5, 95])
    rel = []
    for b0, b1 in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0001)):
        sel = (p >= b0) & (p < b1)
        rel.append(dict(bin=[round(b0, 2), round(min(b1, 1), 2)], n=int(sel.sum()), p_mean=(round(float(p[sel].mean()), 3) if sel.any() else None),
                        o_freq=(round(float(o[sel].mean()), 3) if sel.any() else None)))
    return dict(n=int(n), bss=round(float(bss), 3), bss_ci90=[round(float(lo), 3), round(float(hi), 3)], auc=round(float(auc), 3),
                auc_ci90=[round(float(alo), 3), round(float(ahi), 3)], brier=round(float(bs), 4), brier_clim=round(float(bsc), 4),
                skilful=bool(np.isfinite(lo) and lo > 0), reliability=rel)


def perm_diff(x: np.ndarray, grp: np.ndarray, rng, n=10000) -> tuple[float, float]:
    """mean(x[grp]) − mean(x[~grp]) and its two-sided permutation p-value (over years)."""
    d0 = x[grp].mean() - x[~grp].mean()
    cnt = 0
    for _ in range(n):
        g = rng.permutation(grp)
        if abs(x[g].mean() - x[~g].mean()) >= abs(d0) - 1e-12:
            cnt += 1
    return float(d0), (cnt + 1) / (n + 1)


def binom_two_sided(k: int, n: int, p0: float) -> float:
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0).pvalue)


# ── build ────────────────────────────────────────────────────────────────────
WINDOWS = {"Nov–Mar": (11, 12, 1, 2, 3), "Nov": (11,), "Dec": (12,), "Jan": (1,), "Feb": (2,), "Mar": (3,)}


def member_flags(events: list, months) -> bool:
    return any(pd.Timestamp(d).month in months for d in events)


def build(ym: str) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(20261007)
    month = ym[4:]
    H = R.need(R.hc_ref("ssw", month), "SEAS5 daily u10 hindcast (seas5_ssw hindcast)")
    hsec, lats, hyears = H["sec"], H["lats"], H["years"]
    i60 = int(np.argmin(np.abs(lats - 60)))
    hev = H["meta"]["events"]
    fsec, dates = forecast_daily(ym)
    nmem = fsec.shape[0]
    f60 = fsec[:, :, i60]
    # ── normal: the hindcast's mean and spread at each lead day (all 36 × 25 member-winters)
    h60 = hsec[:, :, :, i60].reshape(-1, NDAYS)
    clim_mean = np.nanmean(h60, 0); q10, q90 = np.nanpercentile(h60, [10, 90], axis=0)
    # σ of the hindcast at each lead, smoothed ±7 days (a daily σ from 900 values is noisy in the tails)
    sd = pd.Series(np.nanstd(h60, 0)).rolling(15, center=True, min_periods=5).mean().values
    z = (f60 - clim_mean[None]) / sd[None]
    # ── SSW detection
    fev = [detect(f60[m], dates) for m in range(nmem)]
    hyr = {}
    for key, e in hev.items():
        y, m = key.split("_"); hyr.setdefault(int(y), []).append(e)
    m2 = merra2_u60()
    obs = {}
    for y in range(1980, 2026):
        s = m2[f"{y}-{int(month):02d}-01":f"{y + 1}-05-03"]
        if len(s) < 200 or s.index[-1] < pd.Timestamp(f"{y + 1}-04-30"):
            continue
        r = detect(s.values, s.index)
        obs[y] = {"ssw": [str(d.date()) for d in r["ssw"]], "fw": str(r["fw"].date()) if r["fw"] is not None else None}
    # cross-check against the site's MERRA-2 SSW catalogue (same CP07 rule, built independently)
    cat = json.loads((STRAT_REF / "strat_history_events.json").read_text()).get("ssw", [])
    cat_dates = sorted(e["date"] for e in cat)
    mine = sorted(d for y in obs for d in obs[y]["ssw"])
    agree = sum(1 for d in mine if any(abs((pd.Timestamp(d) - pd.Timestamp(c)).days) <= 3 for c in cat_dates))
    print(f"  MERRA-2 CP07 here: {len(mine)} SSWs in {len(obs)} winters; {agree} match the site catalogue (±3 d), catalogue {len(cat_dates)}", flush=True)

    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "members": nmem,
           "hindcast": f"{hyears[0]}–{hyears[-1]}, 25 members (AWS SEAS5 daily copy)", "definition": "Charlton & Polvani (2007), u60N 10 hPa daily zonal mean",
           "observed_check": dict(here=len(mine), catalogue=len(cat_dates), matched=agree), "windows": {}}
    yrs = sorted(hyr)
    obs_years = [y for y in yrs if y in obs]
    for w, months in WINDOWS.items():
        fprob = np.mean([member_flags(e["ssw"], months) for e in fev])
        k = int(sum(member_flags(e["ssw"], months) for e in fev))
        pyear = np.array([np.mean([member_flags(e["ssw"], months) for e in hyr[y]]) for y in yrs])
        model_base = float(pyear.mean())
        o36 = np.array([member_flags(obs[y]["ssw"], months) for y in obs_years], float)
        olong = np.array([member_flags(obs[y]["ssw"], months) for y in sorted(obs)], float)
        pv = np.array([pyear[yrs.index(y)] for y in obs_years])
        # model base-rate interval: block bootstrap over years
        bb = np.array([pyear[block_boot_idx(len(pyear), rng)].mean() for _ in range(NBOOT)])
        ob = np.array([olong[block_boot_idx(len(olong), rng)].mean() for _ in range(NBOOT)])
        ent = dict(forecast=round(float(fprob), 3), forecast_k=k,
                   p_vs_model_base=round(binom_two_sided(k, nmem, min(max(model_base, 1e-6), 1 - 1e-6)), 4),
                   model_base=round(model_base, 3), model_base_ci90=[round(float(x), 3) for x in np.percentile(bb, [5, 95])],
                   obs_base_hc_years=round(float(o36.mean()), 3), obs_base_long=round(float(olong.mean()), 3),
                   obs_base_long_ci90=[round(float(x), 3) for x in np.percentile(ob, [5, 95])], obs_long_years=f"{min(obs)}/{(min(obs) + 1) % 100:02d}–{max(obs)}/{(max(obs) + 1) % 100:02d}",
                   n_obs_long=len(olong), skill=verify(pv, o36, rng))
        doc["windows"][w] = ent
        print(f"  {w}: forecast {fprob:.2f} ({k}/{nmem}); model base {model_base:.2f}; observed {o36.mean():.2f} (36 winters), {olong.mean():.2f} "
              f"({len(olong)}); BSS {ent['skill']['bss']:+.2f} [{ent['skill']['bss_ci90'][0]:+.2f}, {ent['skill']['bss_ci90'][1]:+.2f}] "
              f"AUC {ent['skill']['auc']:.2f}", flush=True)
    # ── ENSO context (observed DJF ONI ≥ 0.5 = El Niño)
    oni = oni_djf()
    ens = {}
    pyear = np.array([np.mean([member_flags(e["ssw"], WINDOWS["Nov–Mar"]) for e in hyr[y]]) for y in yrs])
    nino = np.array([oni.get(y, np.nan) >= 0.5 for y in yrs])
    dh, ph = perm_diff(pyear, nino, rng)
    ol = sorted(y for y in obs if y in oni)
    oo = np.array([member_flags(obs[y]["ssw"], WINDOWS["Nov–Mar"]) for y in ol], float); on = np.array([oni[y] >= 0.5 for y in ol])
    do, po = perm_diff(oo, on, rng)
    ens = dict(hindcast=dict(n_nino=int(nino.sum()), rate_nino=round(float(pyear[nino].mean()), 3), rate_other=round(float(pyear[~nino].mean()), 3),
                             diff=round(dh, 3), p=round(ph, 4)),
               observed=dict(years=f"{ol[0]}/{(ol[0] + 1) % 100:02d}–{ol[-1]}/{(ol[-1] + 1) % 100:02d}", n_nino=int(on.sum()),
                             rate_nino=round(float(oo[on].mean()), 3), rate_other=round(float(oo[~on].mean()), 3), diff=round(do, 3), p=round(po, 4)))
    # the model in strong El Niño winters (DJF ONI ≥ 1.5) — this winter's class — descriptive, with its year count
    strong = np.array([oni.get(y, np.nan) >= 1.5 for y in yrs])
    ens["hindcast"]["rate_strong_nino"] = round(float(pyear[strong].mean()), 3) if strong.any() else None
    ens["hindcast"]["strong_years"] = [int(y) for y, s in zip(yrs, strong) if s]
    doc["enso"] = ens
    print(f"  ENSO: hindcast Niño {ens['hindcast']['rate_nino']:.2f} vs {ens['hindcast']['rate_other']:.2f} (p {ph:.3f}); observed "
          f"{ens['observed']['rate_nino']:.2f} vs {ens['observed']['rate_other']:.2f} (p {po:.3f})", flush=True)
    # ── timing: first SSW date of each member; final warming before 1 March
    def doy(d):                                                           # days since 1 Nov
        t = pd.Timestamp(d); y0 = t.year if t.month >= 10 else t.year - 1
        return (t - pd.Timestamp(year=y0, month=11, day=1)).days
    f_first = [doy(e["ssw"][0]) for e in fev if e["ssw"]]
    h_first = [doy(e["ssw"][0]) for y in yrs for e in hyr[y] if e["ssw"]]
    o_all = [doy(d) for y in obs for d in obs[y]["ssw"]]
    fw_before_mar = lambda evs: float(np.mean([e["fw"] is not None and pd.Timestamp(e["fw"]).month in (10, 11, 12, 1, 2) for e in evs]))
    doc["timing"] = dict(forecast_first_doy=f_first, hindcast_first_doy=h_first, observed_doy=o_all,
                         fw_before_march=dict(forecast=round(fw_before_mar([{"fw": (str(e["fw"].date()) if e["fw"] is not None else None)} for e in fev]), 3),
                                              hindcast=round(fw_before_mar([e for y in yrs for e in hyr[y]]), 3),
                                              observed=round(fw_before_mar(list(obs.values())), 3)))
    # ── monthly table: u60N10 and u100 vs the model normal, in σ
    vm = [f"{d.year}-{d.month:02d}" for d in pd.date_range(dates[0], periods=7, freq="MS")]
    mon = []
    for v in vm:
        sel = np.array([f"{d.year}-{d.month:02d}" == v for d in dates])
        if sel.sum() < 28:
            continue
        fm = f60[:, sel].mean(1); hm = h60[:, sel].mean(1)
        mon.append(dict(month=v, mean=round(float(fm.mean()), 1), p10=round(float(np.percentile(fm, 10)), 1), p90=round(float(np.percentile(fm, 90)), 1),
                        normal=round(float(hm.mean()), 1), normal_p10=round(float(np.percentile(hm, 10)), 1), normal_p90=round(float(np.percentile(hm, 90)), 1),
                        z=round(float((fm.mean() - hm.mean()) / hm.std()), 2), p_easterly=round(float(np.mean(fm < 0)), 3),
                        p_easterly_normal=round(float(np.mean(hm < 0)), 3)))
    doc["monthly_u10"] = mon
    doc["u100"] = u100_monthly(ym)
    tail = geosfp_tail()
    doc["observed_tail"] = {str(int(l)): [[str(d.date()), round(float(x), 2)] for d, x in s.items() if np.isfinite(x)] for l, s in tail.items()}
    figs = render(ym, dates, f60, clim_mean, q10, q90, z, fsec, hsec, lats, tail, doc)
    doc["figures"] = figs
    def clean(o):                                                       # NaN is not JSON (a window with no observed event)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    OUT_JSON.write_text(json.dumps(clean(doc), separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return doc


def u100_monthly(ym: str) -> list:
    """Monthly u60N 100 hPa: forecast members (CDS monthly) against the 1993–2016 hindcast normal, leads 1–6."""
    from seas5_build import load_field, valid_months
    t = R.load(R.hc_ref("ssw_u100", ym[4:]))
    if t is None or not fc_u100_path(ym).exists():
        return []
    v, lat, lon = load_field(fc_u100_path(ym), "u", lead7=False)
    f = np.nanmean(v[:, :, 0, :], axis=-1)                              # (member, lead)
    h = t["u"]
    out = []
    for L, m in enumerate(valid_months(ym, 6)):
        fm, hm = f[:, L], h[:, L]
        out.append(dict(month=m, mean=round(float(fm.mean()), 2), p10=round(float(np.percentile(fm, 10)), 2), p90=round(float(np.percentile(fm, 90)), 2),
                        normal=round(float(hm.mean()), 2), normal_p10=round(float(np.percentile(hm, 10)), 2), normal_p90=round(float(np.percentile(hm, 90)), 2),
                        z=round(float((fm.mean() - hm.mean()) / hm.std()), 2), members=[round(float(x), 2) for x in fm]))
    return out


# ── figures ──────────────────────────────────────────────────────────────────
INK, MUTED, FAN, MEAN, CLIM, OBS = "#222", "#6f6b64", "#3f5f8f", "#8a4b2a", "#8a8680", "#111"


def _foot(fig, y, text, size=7.8):
    """A footnote wrapped to the figure's width (2026-10-08 sweep: one long line under bbox_inches="tight" widened the
    saved image to the text, leaving the u100 chart in the left two-thirds of a mostly empty picture)."""
    import textwrap
    n = max(60, int(fig.get_figwidth() * 0.97 * 72 / (size * 0.53)))
    fig.text(0.01, y, textwrap.fill(text, n), fontsize=size, color=MUTED, va="top", linespacing=1.35)


def render(ym, dates, f60, cm, q10, q90, z, fsec, hsec, lats, tail, doc) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    issue = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    figs = {}
    # 1. u10 60N plume + standardised anomaly
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
    x = dates
    ax.fill_between(x, q10, q90, color=CLIM, alpha=0.18, lw=0, label="model normal, 10–90 % (hindcast 1981–2016)")
    ax.plot(x, cm, color=CLIM, lw=1.6, ls="--", label="model normal, mean")
    for m in range(f60.shape[0]):
        ax.plot(x, f60[m], color=FAN, lw=0.45, alpha=0.35, zorder=2)
    p10, p90 = np.percentile(f60, [10, 90], axis=0)
    ax.fill_between(x, p10, p90, color=FAN, alpha=0.22, lw=0, zorder=1, label="51 members, 10–90 %")
    ax.plot(x, f60.mean(0), color=MEAN, lw=2.4, zorder=4, label="ensemble mean")
    if 10.0 in tail:
        s = tail[10.0]; s = s[s.index >= dates[0] - pd.Timedelta(days=45)]
        ax.plot(s.index, s.values, color=OBS, lw=1.8, zorder=5, label="observed (GEOS FP analysis)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvspan(pd.Timestamp(f"{dates[0].year}-11-01"), pd.Timestamp(f"{dates[0].year + 1}-03-31"), color="#000", alpha=0.03, lw=0)
    ax.set_ylabel("u at 60°N, 10 hPa (m/s)"); ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    ax.set_title(f"SEAS5 polar vortex: zonal-mean wind at 60°N, 10 hPa · {issue}", loc="left", fontsize=13)
    zm = z.mean(0); zp10, zp90 = np.percentile(z, [10, 90], axis=0)
    bx.fill_between(x, zp10, zp90, color=FAN, alpha=0.22, lw=0)
    bx.plot(x, zm, color=MEAN, lw=2.0)
    for v in (-1, 1):
        bx.axhline(v, color=CLIM, lw=0.7, ls=":")
    bx.axhline(0, color=INK, lw=0.8)
    bx.set_ylabel("anomaly (σ of the\nhindcast spread)")
    bx.xaxis.set_major_locator(mdates.MonthLocator()); bx.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e3dc", lw=0.6)
    _foot(fig, -0.02, "Daily means. Normal = SEAS5's own hindcast for the same start date and lead (25 members × 36 winters). Shaded: Nov–Mar, "
             "the SSW window. σ = the hindcast's member spread at each lead (15-day smoothed). Contains modified Copernicus Climate Change Service information 2026.")
    out = ASSETS / "seas5_ssw_u10.webp"; fig.savefig(out, dpi=105, bbox_inches="tight", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    figs["u10"] = out.name
    # 2. latitude–time section of the ensemble-mean anomaly vs the model normal
    hm = np.nanmean(hsec.reshape(-1, NDAYS, len(lats)), 0); hs = np.nanstd(hsec.reshape(-1, NDAYS, len(lats)), 0)
    an = fsec.mean(0) - hm
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    lev = [-20, -15, -10, -7.5, -5, -2.5, 2.5, 5, 7.5, 10, 15, 20]
    cs = ax.contourf(dates, lats, an.T, levels=lev, cmap="PuOr_r", extend="both")
    zz = an / np.where(hs > 0, hs, np.nan) * np.sqrt(fsec.shape[0])     # anomaly of the mean in standard errors
    ax.contour(dates, lats, np.abs(zz).T, levels=[2.0], colors=INK, linewidths=0.6)
    ax.set_ylabel("latitude (°N)"); ax.set_ylim(30, 90)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cb = fig.colorbar(cs, ax=ax, pad=0.01); cb.set_label("ensemble-mean anomaly (m/s)")
    ax.set_title(f"SEAS5 zonal-mean wind at 10 hPa, anomaly against the model normal · {issue}", loc="left", fontsize=12.5)
    _foot(fig, -0.02, "Black contour: the ensemble mean lies 2 standard errors from the normal (51 members, hindcast spread) — outside it the anomaly "
             "is within sampling noise. Purple = weaker westerlies (easterly anomaly), orange = stronger.")
    out = ASSETS / "seas5_ssw_section.webp"; fig.savefig(out, dpi=105, bbox_inches="tight", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    figs["section"] = out.name
    # 3. u100 monthly
    if doc.get("u100"):
        U = doc["u100"]
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        xs = np.arange(len(U))
        for i, e in enumerate(U):
            ax.fill_between([i - 0.32, i + 0.32], e["normal_p10"], e["normal_p90"], color=CLIM, alpha=0.2, lw=0)
            ax.plot([i - 0.32, i + 0.32], [e["normal"]] * 2, color=CLIM, lw=1.6, ls="--")
            ax.scatter(i + (np.random.default_rng(i).random(len(e["members"])) - 0.5) * 0.36, e["members"], s=9, color=FAN, alpha=0.6, zorder=3, lw=0)
            ax.plot([i - 0.25, i + 0.25], [e["mean"]] * 2, color=MEAN, lw=2.4, zorder=4)
            ax.text(i, ax.get_ylim()[0] if False else max(e["p90"], e["normal_p90"]) + 0.6, f"{e['z']:+.1f}σ", ha="center", fontsize=8.5, color=INK)
        if 100.0 in tail:
            s = tail[100.0].resample("MS").mean()
            for d, v in s.items():
                if d.strftime("%Y-%m") in [e["month"] for e in U]:
                    j = [e["month"] for e in U].index(d.strftime("%Y-%m")); ax.scatter([j], [v], marker="D", color=OBS, s=30, zorder=5)
        ax.set_xticks(xs); ax.set_xticklabels([f"{calendar.month_abbr[int(e['month'][5:])]}\n{e['month'][:4]}" for e in U])
        ax.set_ylabel("u at 60°N, 100 hPa (m/s)"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e3dc", lw=0.6)
        ax.set_title(f"SEAS5 lower-stratospheric vortex: monthly u at 60°N, 100 hPa · {issue}", loc="left", fontsize=12.5)
        _foot(fig, -0.03, "Dots: the 51 members' monthly means; brown bar: ensemble mean; grey: model normal (1993–2016 hindcast mean, dashed, and "
                 "10–90 % range); label: ensemble-mean anomaly in σ of the hindcast spread. Monthly only (no daily 100 hPa hindcast), leads 1–6.")
        out = ASSETS / "seas5_ssw_u100.webp"; fig.savefig(out, dpi=105, bbox_inches="tight", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
        figs["u100"] = out.name
    # 4. risk: forecast vs model base vs observed, timing histogram, reliability
    W = doc["windows"]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1.3, 1]))
    keys = list(W); xs = np.arange(len(keys)); w = 0.26
    for j, (lab, key, col) in enumerate((("SEAS5 this issue", "forecast", MEAN), ("SEAS5 hindcast base rate", "model_base", CLIM),
                                         (f"observed (MERRA-2 {W[keys[0]]['obs_long_years']})", "obs_base_long", OBS))):
        vals = [W[k][key] for k in keys]
        ax.bar(xs + (j - 1) * w, vals, w * 0.92, color=col, alpha=0.85 if j != 1 else 0.55, label=lab)
        if key in ("model_base", "obs_base_long"):
            ci = [W[k][key + "_ci90"] for k in keys]
            ax.errorbar(xs + (j - 1) * w, vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_xticks(xs); ax.set_xticklabels(keys); ax.set_ylabel("chance of ≥ 1 major SSW"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("SSW risk: share of members with a major SSW", loc="left", fontsize=11)
    T = doc["timing"]; bins = np.arange(0, 152, 10)
    for vals, col, lab in ((T["hindcast_first_doy"], CLIM, "hindcast members"), (T["observed_doy"], OBS, "observed (MERRA-2)"), (T["forecast_first_doy"], MEAN, "this issue's members")):
        if vals:
            bx.hist(vals, bins=bins, density=True, histtype="step" if col != CLIM else "stepfilled", color=col, alpha=0.5 if col == CLIM else 1, lw=1.6, label=f"{lab} (n={len(vals)})")
    bx.set_xticks([0, 30, 61, 92, 120, 151]); bx.set_xticklabels(["1 Nov", "1 Dec", "1 Jan", "1 Feb", "1 Mar", "31 Mar"])
    bx.set_ylabel("density"); bx.legend(fontsize=7.6, frameon=False); bx.spines[["top", "right"]].set_visible(False)
    bx.set_title("When the SSWs fall (first event per member)", loc="left", fontsize=11)
    S = W["Nov–Mar"]["skill"]
    rel = [r for r in S["reliability"] if r["n"]]
    cx.plot([0, 1], [0, 1], color=CLIM, lw=0.8, ls=":")
    cx.scatter([r["p_mean"] for r in rel], [r["o_freq"] for r in rel], s=[30 + 12 * r["n"] for r in rel], color=MEAN, zorder=3)
    for r in rel:
        cx.annotate(f"n={r['n']}", (r["p_mean"], r["o_freq"]), textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=MUTED)
    cx.set_xlim(0, 1); cx.set_ylim(-0.02, 1.02); cx.set_xlabel("hindcast probability"); cx.set_ylabel("observed frequency")
    cx.spines[["top", "right"]].set_visible(False)
    verdict = ("skill beyond climatology" if S["skilful"] else
               "WORSE than climatology" if (S["bss_ci90"][1] is not None and S["bss_ci90"][1] < 0) else "NO skill beyond climatology")
    cx.set_title(f"Hindcast reliability, Nov–Mar ({verdict})", loc="left", fontsize=10.5)
    cx.text(0.02, 0.98, f"BSS {S['bss']:+.2f} (90 % CI {S['bss_ci90'][0]:+.2f} to {S['bss_ci90'][1]:+.2f})\nROC area {S['auc']:.2f} "
            f"({S['auc_ci90'][0]:.2f}–{S['auc_ci90'][1]:.2f}), {S['n']} winters", transform=cx.transAxes, va="top", fontsize=8, color=INK)
    fig.suptitle(f"SEAS5 sudden-warming risk · {issue}" + ("" if S["skilful"] else "  —  no hindcast skill: the model's view, not a forecast"), x=0.01, ha="left", fontsize=13)
    _foot(fig, -0.04, "Major SSW: Charlton & Polvani (2007) on daily u60N 10 hPa, Nov–Mar, 20-day separation, final warmings excluded. Error bars: "
             "90 % block-bootstrap intervals over winters. Reliability: the hindcast's yearly member fraction against MERRA-2 occurrence, 1981/82–2016/17.")
    fig.subplots_adjust(wspace=0.32)
    out = ASSETS / "seas5_ssw_risk.webp"; fig.savefig(out, dpi=105, bbox_inches="tight", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    figs["risk"] = out.name
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hindcast", "fetch", "build"])
    ap.add_argument("--issue"); ap.add_argument("--month"); ap.add_argument("--only", choices=["u10", "u100"])
    a = ap.parse_args(argv)
    if a.cmd == "hindcast":
        return 0 if hindcast(a.month) else 1
    if a.cmd == "fetch":
        return 0 if fetch(a.issue, a.only) else 1
    build(a.issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
