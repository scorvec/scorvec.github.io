#!/usr/bin/env python3
"""Reference build (offline, once) for the GEFS page's weekly skill mask and tercile maps - the GEFS port of the GEPS
page's hindcast_maps.py and recent_median.py, so the GEFS page carries the same maps and charts as the GEPS page.

TRUTH - the GEPS page's observations, put on the GEFS 1.5 deg grid (gefs_reforecast.LAT 90 -> -90, LON 0E first):
  t2m    NCEP/NCAR R1 2 m air temperature (daily, T62 Gaussian): bilinear, clamped to the outermost Gaussian row
         beyond +-88.5
  zg500  NCEP/NCAR R1 500 hPa height (2.5 deg): bilinear
  mslp   NCEP/NCAR R1 sea-level pressure (2.5 deg, Pa): bilinear
  pr     CPC unified gauge analysis (0.5 deg, LAND ONLY): a CONSERVATIVE area-weighted box mean over the 1.5 deg box
         centred on each GEFS point - the same box the GEFS fields are averaged over - kept where land cells cover
         at least a quarter of the box. (GEPS box-meaned to 2.5 deg and took the nearest cell.)
Each against its own 1991-2020 day-of-year climatology (daily means by day of year, 31-day circular running mean:
GEPS's doy_clim). Observed week k of reforecast start S = the mean of the daily anomalies over S + 7k-7 ... S + 7k-1
(>= 5 valid days), GEFS's day convention (day d = the UTC calendar day init + d - 1).

FORECAST - every GEFSv12 reforecast start 2000-2019 (Wednesdays, 4 members, weekly means: the per-member weekly
fields of release gefs-clim-v3, gefs_hindw_MM.npz):
  anomaly = [F - M(start doy, week)] - shift(valid dates)
  M is the reforecast climatology read as gefs_live.clim_at reads it (5-day centres pooling the starts within +-15
  days of any year on the common-year calendar, linear in between) but LEAVE-SEASON-OUT: every start within 40 days
  of the verified start - its own season, including the neighbouring calendar year across New Year - is removed from
  the pool, so the forecast being verified is never inside its own climatology, as a live forecast never is. (GEPS
  de-drifted its hindcasts against the full 2001-2020 climatology, the verified year included; in-sample by ~5 % of
  the pool.) Pass 1 rebuilds the full-pool climatology from the same starts and checks it against the release.
  shift = the 7-day mean over the week's valid dates of era5_shift.npz (ERA5 1991-2020 minus 2001-2020), the maps'
  re-basing - applied to precipitation too (GEPS left its precipitation un-shifted; here all fields share one rule) -
  with its 2001-2020 normal LEAVE-SEASON-OUT too: the days of that normal within 40 days of each valid date are taken
  out of it (ERA5 NORMAL below). Left in, the normal's day-to-day shape holds a twentieth of each scored year's
  observed weather, which a correlation pooled over +-45 days of starts credits to the forecast. The 1991-2020 normal
  is the anomaly's definition, as the observed side's own 1991-2020 normal is, and stays whole (the teleconnection
  calibration, build_gefs_patterns_hindcast.py, and the GEPS page's telecon_hindcast.py treat the normals alike).

ERA5 NORMAL (era5-normal, era5-days, era5-loso) - the 2001-2020 normal inside era5_shift rebuilt per 5-day centre
(+-7 days) from the days it was built from, and for every centre and every season of its pool (the run of pool days
in one year, or across New Year) the normal without that season minus the normal, D, on the 1.5 deg grid:
  zg500          the 1.5 deg daily store (WeatherBench2 ERA5, daily mean of the four synoptic hours), every day, pooled
                 on the common-year calendar (build_era5_clim.py) - offline, from the local store
  t2m mslp pr    the GEPS page's 1 deg ARCO surface normal: the 00/06/12/18Z mean of every fifth day from 1 January,
                 0.25 deg box-filtered onto whole degrees ([1, 2, 2, 2, 1] / 8), pooled on the raw day of year, then
                 interpolated to 1.5 deg as era5_shift was (bilinear) - streamed from ARCO in Actions
  The rebuilt normal must reproduce the normal in use, else the run stops (era5-normal writes the surface one at 1.5
  deg, offline). A valid date takes D of the season within 40 days of it at the two centres around its day-of-year
  key, linear between them, as the re-basing reads era5_shift; dates outside the normal's years take none.

STATISTICS - GEPS's recipe: pairs pooled by calendar month (starts within +-45 days of the 15th), reduced at every
gridpoint and week to the anomaly correlation r, the regression obs = a*fcst + b, its residual sd s (n-2
denominator) and the observed sd; NaN where fewer than 30 pairs. Diagnostics kept alongside (work dir, not published):
the within-ensemble variance of the four members and the same statistics for a 2-member mean, which test how much a
4-member hindcast understates the skill of the 31-member live mean (reported under Other checks, not applied).

RECENT-DECADE TERCILES (t2m, pr; recent_median.py) - the 33rd/67th percentiles and the mean of the 7-day-mean
anomaly (vs 1991-2020) over 2016-2025, windows whose centre is within +-15 days of each 5-day centre, plus (pr) the
absolute 1991-2020 weekly climatology for the dry-week mask. Computed on the 1.5 deg truth itself, so thresholds,
skill and members are all on one grid and one box size (GEPS computed them at 2.5 deg and interpolated to its 1 deg
members; for t2m the two agree to r 0.98-0.997 and within 0.01 K in the mean).

CALIBRATION TEST (assess) - every start scored with its own init month's table refitted without its season, RPSS of
the calibrated terciles vs one third each (the 2026-09-26 build: 1043 starts, the ERA5 normal still in sample and
the climatology pooled on the raw day of year; area-weighted, cells with truth):
    t2m  weeks 1-5  +0.46 +0.24 +0.13 +0.10 +0.08  (cells with r >= 0.25 +0.46 ... +0.14, the rest +0.02 ... +0.04);
         reliable: P >= 0.4 bins forecast 0.71/0.60/0.54/0.53/0.52 vs observed 0.72/0.61/0.55/0.54/0.53
    pr   weeks 1-5  +0.18 +0.04 +0.01 +0.00 +0.00  (cells with r < 0.25 ~0); the split was the mask of the time, r >= 0.25,
         not the significance test (gefs_skill.noskill) the maps hatch with now - a rerun of assess splits by that
         GEPS's prob_maps form on the same pairs: +0.14 -0.00 -0.04 -0.04 -0.05 (worse than climatology from week 2 -
         the skewed-terciles artefact gefs_probs.py explains). t2m: the two forms agree to 0.003.
Other checks (2026-09-26): pass 1 reproduced its release exactly (n and means, max |d| 0.004 Pa); the re-based
hindcast anomalies are unbiased against the observed ones (2000-2019 mean forecast minus observed: t2m +0.01 K, pr
-0.03 mm/day, zg500 -0.5 m, mslp +3 Pa); the 2-member -> 4-member skill predicted from the member spread matches the
measured 4-member skill to 0.01 (t2m week 3: 0.400 vs 0.406), so a 31-member mean would score higher still
(t2m week 3 r ~0.47 vs 0.41) - not applied, the per-cell estimate amplifies sampling noise where skill is small.

TERCILE STEP TEST (step) - the live maps do not use the fit years' terciles that `assess` scores against: their
categories are the 2016-2025 terciles and their calibration is centred on the 2016-2025 mean anomaly. The four
reforecast years inside that decade (starts 2016-2019, STEP_Y0-STEP_Y1) are scored exactly as the live maps score a
forecast - gefs_probs' standardised form with the published r and forecast sd of the init month and week, the
2016-2025 terciles and recent mean at each week's centre, the dry-week mask - from a leave-season-out climatology,
against the observed category of each week (step-truth, offline: the `truth` weekly anomalies against the same
terciles, committed as data/gefs_tercile_obs.npz). Reported per field and week, area-weighted over the cells with
truth: the observed frequency of each 2016-2025 tercile, the mean calibrated probability of each, their difference
(the step's bias; per-start area means, t on the effective number of weekly starts from their lag-1
autocorrelation, Benjamini-Hochberg at FDR 10 % over every field, week and category), the RPSS against one third each
with its one-sided p, and the reliability of P(below) and P(above). The ERA5 2001-2020 re-basing normal leaves each
valid date's season out, as in `skill`; the published r and forecast sd were fitted on all 20 reforecast years, the
four scored ones included, so the sharpness and RPSS are not fully out of sample, while the bias test of the step
(categories and centring) does not depend on them to first order. The observed frequencies are also tested against a
third (same t, their own Benjamini-Hochberg family). Result (2026-10-17, 208 starts 2016-01-06 .. 2019-12-25, the
ERA5 normal still in sample and the climatology pooled on the raw day of year):
    t2m  observed below/near/above 0.325/0.331/0.345 (none off a third); calibrated 0.319-0.324/0.333-0.336/
         0.342-0.345; bias within 0.005 in every week, none significant; RPSS +0.42/+0.20/+0.10/+0.08/+0.06
    pr   the four years were drier than the decade: observed 0.351/0.328/0.321, all three off a third (FDR 10 %), so
         the calibrated maps, a third each from week 3, give "below" 0.011-0.019 too little (weeks 1-5) and "above"
         0.011-0.014 too much (weeks 2-5), all significant; RPSS +0.20/+0.04/+0.01/0.00/-0.00

Offline, from the local stores (obsweek_<tag>, era5_loso_zg500 and era5_normal_0120 in WORK are published as
release gefs-truth-v1):
    python build_gefs_mapskill.py truth  --tags t2m,pr,zg500,mslp   # ~1-2 min a tag; WORK/obsclim_, obsweek_{tag}
    python build_gefs_mapskill.py recent --tags t2m,pr              # -> REF/gefs_terciles_{tag}.npz
    python build_gefs_mapskill.py step-truth                        # after truth -> data/gefs_tercile_obs.npz
    python build_gefs_mapskill.py era5-loso --tags zg500 --store STORE --normals NORMALS  # -> WORK/era5_loso_zg500
    python build_gefs_mapskill.py era5-normal --normals NORMALS     # -> WORK/era5_normal_0120.npz
In Actions (gefs-mapskill.yml; the reforecast from release gefs-clim-v3):
    python build_gefs_mapskill.py era5-days --years 2001,2005 --out DAYS            # ARCO -> DAYS/era5_days_2001.npz
    python build_gefs_mapskill.py era5-loso --tags t2m,mslp,pr --days DAYS --work W # -> W/era5_loso_{tag}.npz
    python build_gefs_mapskill.py skill  --rf HINDW --clim CLIM --shift S --ref OUT --work W   # OUT/gefs_mapskill_*
    python build_gefs_mapskill.py assess --ref OUT --work W                         # the test into their meta
    python build_gefs_mapskill.py step   --rf HINDW --clim CLIM --shift S --ref OUT --work W --out step.json
Memory: one year of CPC (0.4 GB) or the recent decade of one field (~1.5 GB); the skill pass ~4 GB (four fields).
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R  # (grid, unpack, weeks, the climatology release)
import gefs_skill as SK  # (dates, centres, shift)

sys.path.insert(0, str(HERE.parent / "lib"))
from localdirs import env_dir, env_dir_if_set

# the offline defaults, under $DATA_ARCHIVE; a runner, where it is unset, passes every path on the command line
REF = env_dir_if_set("DATA_ARCHIVE", "gefs_ref", "probs")
WORK = env_dir_if_set("DATA_ARCHIVE", "gefs_port_test", "probs", "work")
SHIFT = env_dir_if_set("DATA_ARCHIVE", "gefs_ref", "era5_shift.npz")
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EY0, EY1 = 2001, 2020  # the normal era5_shift re-bases from
ERA5_HALF = 7  # its pool: days within +-7 of each 5-day centre
SFC = {"t2m": "2m_temperature", "mslp": "mean_sea_level_pressure", "pr": "mean_total_precipitation_rate"}
SFC_SCALE = {"t2m": 1.0, "mslp": 1.0, "pr": 86400.0}  # -> GEFS units (pr kg m-2 s-1 -> mm/day)
SFC_HOURS = (0, 6, 12, 18)
SFC_STRIDE = 5  # every fifth day from 1 January
# rebuilt normal vs the normal in use, largest tolerated |difference| (float32 rounding is ~1e-4 of these)
NORMAL_TOL = {"t2m": 0.01, "mslp": 1.0, "pr": 0.01, "zg500": 0.01}
CY0, CY1 = 1991, 2020  # observed normal
HY0, HY1 = 2000, 2019  # reforecast years
RY0, RY1 = 2016, 2025  # recent decade for the terciles
WIN = 45  # +-days around the 15th: the calendar-month pool
LOSO = 40  # leave-season-out: drop pool starts within this many days of the verified start
MIN_PAIRS = 30
LAND_MIN = 0.25  # CPC land cover needed in a 1.5 deg box
B0 = dt.date(2000, 1, 5)  # first reforecast start = first observed week block
Z3 = 0.4307272992954576  # standard normal upper tercile (gefs_probs.Z3)
STEP_Y0, STEP_Y1 = 2016, 2019  # the reforecast years inside the 2016-2025 tercile decade
STEP_OBS = HERE / "data" / "gefs_tercile_obs.npz"
CATS = ("below", "near", "above")
NY, NX = len(R.LAT), len(R.LON)
UNITS = {"t2m": "K", "pr": "mm/day", "zg500": "m", "mslp": "Pa"}
TRUTH = {
    "t2m": "NCEP/NCAR R1 2 m air",
    "pr": "CPC unified gauge (land only), conservative 1.5 deg box mean",
    "zg500": "NCEP/NCAR R1 500 hPa height",
    "mslp": "NCEP/NCAR R1 sea-level pressure",
}


# ── regridding weights ───────────────────────────────────────────────────────────────────────────────────────────
def lin_w(src, tgt, periodic=False):
    """(len(tgt), len(src)) linear interpolation weights; src monotone (either direction). Periodic: 360 deg wrap.
    Non-periodic targets outside the source range take the nearest end value."""
    src = np.asarray(src, float)
    W = np.zeros((len(tgt), len(src)))
    order = np.argsort(src)
    s = src[order]
    for i, t in enumerate(tgt):
        if periodic:
            ext = np.r_[s, s[0] + 360.0]
            tp = t % 360.0
            if tp < s[0]:
                tp += 360.0
            j = int(np.searchsorted(ext, tp, side="right")) - 1
            j = min(j, len(s) - 1)
            f = (tp - ext[j]) / (ext[j + 1] - ext[j])
            W[i, order[j]] += 1 - f
            W[i, order[(j + 1) % len(s)]] += f
        elif t <= s[0]:
            W[i, order[0]] = 1.0
        elif t >= s[-1]:
            W[i, order[-1]] = 1.0
        else:
            j = int(np.searchsorted(s, t, side="right")) - 1
            f = (t - s[j]) / (s[j + 1] - s[j])
            W[i, order[j]] = 1 - f
            W[i, order[j + 1]] = f
    return W


def box_w(src, half_src, tgt, half_tgt, periodic=False, coslat=False):
    """(len(tgt), len(src)) overlap weights of source cells [c-half_src, c+half_src] with target boxes
    [t-half_tgt, t+half_tgt] (latitude clipped to +-90); coslat multiplies by the source cell's cos(lat)."""
    src = np.asarray(src, float)
    W = np.zeros((len(tgt), len(src)))
    for i, t in enumerate(tgt):
        if periodic:
            d = ((src - t + 180.0) % 360.0) - 180.0
            lo, hi = np.maximum(d - half_src, -half_tgt), np.minimum(d + half_src, half_tgt)
        else:
            lo = np.maximum(src - half_src, max(t - half_tgt, -90.0))
            hi = np.minimum(src + half_src, min(t + half_tgt, 90.0))
        W[i] = np.clip(hi - lo, 0, None)
    if coslat:
        W = W * np.cos(np.deg2rad(src))[None, :]
    return W


_W = {}


def weights(kind, lat, lon):
    key = (kind, len(lat), len(lon))
    if key not in _W:
        if kind == "box":
            _W[key] = (box_w(lat, 0.25, R.LAT, 0.75, coslat=True), box_w(lon, 0.25, R.LON, 0.75, periodic=True))
        else:
            _W[key] = (lin_w(lat, R.LAT), lin_w(lon, R.LON, periodic=True))
    return _W[key]


def regrid(a, lat, lon, kind, chunk=31):
    """(T, lat, lon) -> (T, 121, 240) float32, a month of days at a time. 'box' is NaN-aware (land-only data) and
    returns NaN where the valid cells cover under LAND_MIN of the box."""
    if a.shape[0] > chunk:
        return np.concatenate([regrid(a[i : i + chunk], lat, lon, kind, chunk) for i in range(0, a.shape[0], chunk)])
    wy, wx = weights(kind, lat, lon)
    if kind != "box":
        return np.einsum("ij,tjk,lk->til", wy, a.astype("float64"), wx, optimize=True).astype("float32")
    ok = np.isfinite(a)
    num = np.einsum("ij,tjk,lk->til", wy, np.where(ok, a, 0.0).astype("float64"), wx, optimize=True)
    den = np.einsum("ij,tjk,lk->til", wy, ok.astype("float64"), wx, optimize=True)
    tot = (wy.sum(1)[:, None] * wx.sum(1)[None, :])[None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < LAND_MIN * tot] = np.nan
    return out.astype("float32")


# ── truth, a year at a time ──────────────────────────────────────────────────────────────────────────────────────
def daily_year(tag, y):
    """(dates, (T, 121, 240) float32) of year y in GEFS units, or None if the file is missing."""
    import xarray as xr

    if tag == "pr":
        p = env_dir("DATA_ARCHIVE", "geps_subx", "telecon", "cpc_precip", f"precip.{y}.nc")
        var, kind = "precip", "box"
    elif tag == "t2m":
        p = env_dir("DATA_ARCHIVE", "geps_subx", "telecon", "ncep", f"air.2m.gauss.{y}.nc")
        var, kind = "air", "lin"
    else:
        p = env_dir("DATA_ARCHIVE", "geps_subx", "telecon", "ncep", f"hgt.{y}.nc" if tag == "zg500" else f"slp.{y}.nc")
        var, kind = ("hgt" if tag == "zg500" else "slp"), "lin"
    if not p.exists():
        return None
    d = xr.open_dataset(p)
    a = d[var]
    if tag == "zg500":
        a = a.sel(level=500.0)
    v = a.values.astype("float32")
    lat, lon = d["lat"].values.astype(float), d["lon"].values.astype(float)
    t = [dt.date.fromisoformat(str(x)[:10]) for x in d["time"].values]
    d.close()
    if tag == "mslp" and np.nanmax(v) < 2000:  # hPa file -> Pa (R1 slp is Pa already)
        v = v * 100.0
    _, ix = np.unique(np.array([x.toordinal() for x in t]), return_index=True)
    return [t[i] for i in ix], regrid(v[ix], lat, lon, kind)


def doy_clim(sums, cnt):
    """GEPS build_telecon_patterns.doy_clim: day-of-year mean, gaps filled, 31-day circular running mean. sums, cnt:
    (365, lat, lon) by day-of-year key (SK.noleap_doy, 29 Feb left out); returns (366, lat, lon), slot 366 = 1 January
    (the wrap) - read it with R.at_key, which takes 29 Feb as the mean of 28 Feb and 1 March."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sums / cnt
    ok = np.isfinite(c).astype("float64")
    ext_v = np.concatenate([np.nan_to_num(c[-15:]), np.nan_to_num(c), np.nan_to_num(c[:15])])
    ext_o = np.concatenate([ok[-15:], ok, ok[:15]])
    cs_v = np.cumsum(np.concatenate([np.zeros((1, *c.shape[1:])), ext_v]), 0)
    cs_o = np.cumsum(np.concatenate([np.zeros((1, *c.shape[1:])), ext_o]), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = (cs_v[31:] - cs_v[:-31]) / (cs_o[31:] - cs_o[:-31])
    sm[(cs_o[31:] - cs_o[:-31]) < 16] = np.nan  # at least half the 31-day window observed
    return np.concatenate([sm, sm[:1]]).astype("float32")  # (366, lat, lon)


def cmd_truth(a) -> int:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    last = dt.date(HY1, 12, 31)
    while last.weekday() != 2:
        last -= dt.timedelta(days=1)
    nblk = (last - B0).days // 7 + len(SK.WEEKS)
    for tag in a.tags.split(","):
        t0 = time.time()
        sums = np.zeros((365, NY, NX))
        cnt = np.zeros((365, NY, NX))
        for y in range(CY0, CY1 + 1):
            r = daily_year(tag, y)
            if r is None:
                raise SystemExit(f"{tag}: truth year {y} missing")
            t, v = r
            for i, d in enumerate(t):
                x = SK.noleap_doy(d)
                if x != int(x):
                    continue  # 29 Feb: no key of its own
                j = int(x) - 1
                ok = np.isfinite(v[i])
                sums[j] += np.where(ok, v[i], 0.0)
                cnt[j] += ok
        clim = doy_clim(sums, cnt)
        np.save(work / f"obsclim_{tag}.npy", clim)
        del sums, cnt
        print(f"  {tag}: 1991-2020 clim ({time.time() - t0:.0f} s)", flush=True)
        bs = np.zeros((nblk, NY, NX), "float32")
        bc = np.zeros((nblk, NY, NX), "int8")
        for y in range(HY0, HY1 + 2):
            t, v = daily_year(tag, y)
            for i, d in enumerate(t):
                b = (d - B0).days
                if b < 0 or b // 7 >= nblk:
                    continue
                an = v[i] - R.at_key(clim, SK.noleap_doy(d))
                ok = np.isfinite(an)
                bs[b // 7] += np.where(ok, an, 0.0)
                bc[b // 7] += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            wk = np.where(bc >= 5, bs / np.maximum(bc, 1), np.nan).astype("float16")  # 3 digits: the skill pass's input
        np.savez_compressed(work / f"obsweek_{tag}.npz", wk=wk, b0=np.array(B0.isoformat()), nblk=nblk)
        fin = np.isfinite(wk)
        print(
            f"  {tag}: {nblk} observed weeks from {B0}, {fin.mean() * 100:.0f}% finite, "
            f"sd {np.nanstd(wk.astype('float32')):.2f} {UNITS[tag]}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
    return 0


def cmd_recent(a) -> int:
    ref = Path(a.ref)
    ref.mkdir(parents=True, exist_ok=True)
    for tag in a.tags.split(","):
        t0 = time.time()
        clim = np.load(Path(a.work) / f"obsclim_{tag}.npy")
        days, parts = [], []
        for y in range(RY0, RY1 + 1):
            r = daily_year(tag, y)
            if r is None:
                raise SystemExit(f"{tag}: truth year {y} missing")
            t, v = r
            parts.append(v - R.at_key(clim, SK.noleap_doy(t)))
            days += t
        an = np.concatenate(parts)
        del parts
        # continuity check: the 7-day windows must not jump a missing day
        gaps = np.diff([d.toordinal() for d in days])
        if (gaps != 1).any():
            raise SystemExit(f"{tag}: gaps in the recent daily truth")
        ok = np.isfinite(an)
        s7 = np.zeros((an.shape[0] - 6, NY, NX), "float32")
        c7 = np.zeros_like(s7)
        for k in range(7):
            s7 += np.where(ok[k : k + s7.shape[0]], an[k : k + s7.shape[0]], 0.0)
            c7 += ok[k : k + s7.shape[0]]
        del an, ok
        with np.errstate(invalid="ignore", divide="ignore"):
            wk = np.where(c7 >= 5, s7 / np.maximum(c7, 1), np.nan).astype("float32")
        del s7, c7
        wdoy = SK.noleap_doy([d + dt.timedelta(days=3) for d in days[: wk.shape[0]]])  # window centre, day key
        flat = wk.reshape(wk.shape[0], -1)
        allfin = np.isfinite(flat).all(0)
        anyfin = np.isfinite(flat).any(0)
        part = anyfin & ~allfin
        med = np.full((len(SK.CENTRES), NY * NX), np.nan, "float32")
        q33, q67, mean = med.copy(), med.copy(), med.copy()
        for ci, c in enumerate(SK.CENTRES):
            sel = np.abs(((wdoy - c + 182.5) % 365) - 182.5) <= 15
            x = flat[sel]
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean[ci] = np.nanmean(x, axis=0)
            p = np.percentile(x[:, allfin], [50, 100 / 3, 200 / 3], axis=0)
            med[ci, allfin], q33[ci, allfin], q67[ci, allfin] = p
            if part.any():
                with np.errstate(all="ignore"):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p = np.nanpercentile(x[:, part], [50, 100 / 3, 200 / 3], axis=0)
                med[ci, part], q33[ci, part], q67[ci, part] = p
        sh = (len(SK.CENTRES), NY, NX)
        out = {
            "q33": q33.reshape(sh).astype("float16"),  # (the median is not used, so it is not kept)
            "q67": q67.reshape(sh).astype("float16"),
            "mean": mean.reshape(sh).astype("float16"),
            "doy": SK.CENTRES.astype("int16"),
        }
        if tag == "pr":
            # absolute 1991-2020 weekly climatology (7 days centred on each centre), for the dry-week mask
            cw = np.stack([np.nanmean(clim[[(c - 1 + k) % 365 for k in range(-3, 4)]], axis=0) for c in SK.CENTRES])
            out["clim_weekly"] = cw.astype("float16")
        meta = dict(
            tag=tag,
            units=UNITS[tag],
            years=f"{RY0}-{RY1}",
            base=f"{CY0}-{CY1}",
            truth=TRUTH[tag],
            note="q33 / q67 / mean of the 7-day-mean anomaly vs the 1991-2020 day-of-year climatology over "
            "the recent decade, windows centred within +-15 d of each 5-day centre (doy); GEFS 1.5 deg "
            "grid, lat 90 -> -90; read linearly between centres (gefs_skill.at_doy)",
        )
        np.savez_compressed(ref / f"gefs_terciles_{tag}.npz", meta=np.array(json.dumps(meta)), **out)
        print(
            f"  {tag}: terciles {RY0}-{RY1}, median of median {np.nanmedian(med):+.3f}, "
            f"mean q67-q33 {np.nanmean(q67 - q33):.2f} {UNITS[tag]} "
            f"({(ref / f'gefs_terciles_{tag}.npz').stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)",
            flush=True,
        )
    return 0


# ── the reforecast ───────────────────────────────────────────────────────────────────────────────────────────────
def starts(rf_dir):
    """(date, (bundle, key prefix)) of every reforecast start in the weekly hindcast bundles gefs_hindw_MM.npz under
    rf_dir (release gefs-clim-v3), chronological."""
    out = []
    for f in sorted(Path(rf_dir).glob("gefs_hindw_*.npz")):
        with np.load(f) as z:
            sts = sorted({k.split("_")[0] for k in z.files if k.endswith("_members")})
        out += [(dt.date(int(s[:4]), int(s[4:6]), int(s[6:])), (str(f), f"{s}_")) for s in sts]
    return sorted(out)


@functools.cache
def _bundle(path: str):
    return np.load(path)  # kept open: every start of a month is read from it


def member_sums(src, tags):
    """(members, {tag: (m, 5, lat, lon) float32 weekly fields}) of one start, src = (bundle, key prefix)."""
    z, pre = _bundle(src[0]), src[1]
    return len(z[f"{pre}members"]), {t: R.unpack(t, z[f"{pre}w_{t}"]).astype("float32") for t in tags}


def pools_of(dates):
    """{date: [indices of the 5-day centres whose pool holds that start]} - gefs_reforecast.cmd_combine's pools."""
    return {d: [ci for ci, c in enumerate(SK.CENTRES) if R.pool(SK.noleap_doy(d), c)] for d in dates}


def clim_week(c: int, clim_dir: Path | None, work: Path) -> Path:
    """gefs_clim_DDD.npz of the release: from clim_dir when it holds it, else downloaded into the work cache."""
    p = Path(clim_dir) / f"gefs_clim_{c:03d}.npz" if clim_dir else None
    return p if p is not None and p.exists() else R.clim_file(f"gefs_clim_{c:03d}.npz", work / "clim")


# ── the ERA5 2001-2020 normal, leave-season-out ──────────────────────────────────────────────────────────────────
def normal_days(tag: str) -> list[dt.date]:
    """The days the 2001-2020 normal of `tag` averages: every day (zg500), every fifth day from 1 January (surface)."""
    if tag == "zg500":
        return [
            dt.date(EY0, 1, 1) + dt.timedelta(days=i)
            for i in range((dt.date(EY1, 12, 31) - dt.date(EY0, 1, 1)).days + 1)
        ]
    out = []
    for y in range(EY0, EY1 + 1):
        d = dt.date(y, 1, 1)
        while d.year == y:
            out.append(d)
            d += dt.timedelta(days=SFC_STRIDE)
    return out


def normal_keys(tag: str, days) -> np.ndarray:
    """The pool key the normal was built on: the common-year day (zg500, build_era5_clim.py) or the raw day of year
    (the surface normal) - on every fifth day from 1 January the two give the same pools."""
    if tag == "zg500":
        return SK.noleap_doy(days)
    return np.array([d.timetuple().tm_yday for d in days], dtype="float64")


class Loso:
    """One field's 2001-2020 normal per 5-day centre (the mean of its pool days) and, for every season of each pool -
    a run of pool days at most a few days apart, one per year (across New Year for the first and last centres) - the
    normal without that season minus the normal: D = (S - S_run) / (n - n_run) - S / n."""

    def __init__(self, days, keys):
        self.days = list(days)
        ords = np.array([d.toordinal() for d in self.days])
        self.member = [[] for _ in self.days]  # per day: (centre index, run index) of every pool holding it
        self.first, self.last, self.n_run, self.n = [], [], [], np.zeros(len(SK.CENTRES))
        for ci, c in enumerate(SK.CENTRES):
            idx = np.where(R.pool(keys, c, ERA5_HALF))[0]
            idx = idx[np.argsort(ords[idx])]
            cut = np.where(np.diff(ords[idx]) > LOSO)[0] + 1  # seasons are a year apart, their days within 15
            runs = np.split(idx, cut)
            for r, run in enumerate(runs):
                for i in run:
                    self.member[i].append((ci, r))
            self.first.append([int(ords[run[0]]) for run in runs])
            self.last.append([int(ords[run[-1]]) for run in runs])
            self.n_run.append([len(run) for run in runs])
            self.n[ci] = len(idx)
        self.R = max(len(f) for f in self.first)
        self.S = self.S_run = None

    def add(self, i: int, x) -> None:
        """Day i (index into days) of the field, (lat, lon) on the 1.5 deg grid."""
        if self.S is None:
            self.S = np.zeros((len(SK.CENTRES), *x.shape))
            self.S_run = np.zeros((len(SK.CENTRES), self.R, *x.shape))
        for ci, r in self.member[i]:
            self.S[ci] += x
            self.S_run[ci, r] += x

    def finish(self) -> dict:
        """-> normal E (73, lat, lon), D (73, R, lat, lon) float16 (0 in padding), first / last ordinal day and
        size of every season (73, R; 0 in padding), pool size n (73)."""
        E = self.S / self.n[:, None, None]

        def pad(v, fill):
            return np.array([x + [fill] * (self.R - len(x)) for x in v])

        n_run = pad(self.n_run, 0)
        D = np.zeros_like(self.S_run)
        for ci in range(len(SK.CENTRES)):
            for r in range(len(self.first[ci])):
                D[ci, r] = (self.S[ci] - self.S_run[ci, r]) / (self.n[ci] - n_run[ci, r]) - E[ci]
        return dict(E=E, D=D.astype("float16"), first=pad(self.first, 0), last=pad(self.last, 0), n_run=n_run, n=self.n)


class EraLeaveOut:
    """era5_loso_<tag>.npz: the weekly leave-season-out term added to a re-based hindcast anomaly."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.D, self.first, self.last, self.n_run = z["D"], z["first"], z["last"], z["n_run"]

    def day(self, d: dt.date) -> np.ndarray:
        """(normal without d's season - normal) at d, linear between the two centres around its day-of-year key."""
        v = d.toordinal()
        i_lo, w_lo, i_hi, w_hi = SK.bracket(SK.noleap_doy(d))
        out = np.zeros(self.D.shape[2:])
        for ci, w in ((i_lo, w_lo), (i_hi, w_hi)):
            near = np.where((self.n_run[ci] > 0) & (self.last[ci] >= v - LOSO) & (self.first[ci] <= v + LOSO))[0]
            if w == 0 or not near.size:
                continue
            r = near[0]
            if near.size > 1 or self.first[ci, r] < v - LOSO or self.last[ci, r] > v + LOSO:
                raise ValueError(f"{d}: centre {SK.CENTRES[ci]} has no single season inside the {LOSO}-day window")
            out += w * self.D[ci, r].astype("float64")
        return out

    def weekly(self, init: dt.date) -> np.ndarray:
        """(5, lat, lon): the 7-day mean over each week's valid dates (as gefs_skill.weekly_shift reads the shift)."""
        return np.stack([np.mean([self.day(d) for d in SK.week_dates(init, k)], axis=0) for k in range(1, 6)])


def check_normal(tag: str, E: np.ndarray, ref: np.ndarray) -> None:
    """The rebuilt normal must reproduce the normal in use (NORMAL_TOL), else stop."""
    d = np.abs(E - ref)
    print(
        f"  {tag}: rebuilt 2001-2020 normal vs the one in use: max |d| {d.max():.3g}, rms {np.sqrt((d**2).mean()):.3g}"
    )
    if not d.max() <= NORMAL_TOL[tag]:
        raise SystemExit(f"{tag}: the rebuilt normal does not reproduce the normal in use (max |d| {d.max():.3g})")


def to_1deg(v):
    """0.25 deg (lat 90 -> -90, lon from 0E) -> whole degrees, as the GEPS page's ARCO surface normal: a [1, 2, 2, 2, 1]
    / 8 box in each direction (edge-padded in latitude, wrapped in longitude), then every fourth point."""
    k = np.array([1.0, 2.0, 2.0, 2.0, 1.0]) / 8.0
    a = v
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (2, 2)
        b = np.pad(a, pad, mode=("wrap" if ax == 1 else "edge"))
        a = sum(k[i] * np.take(b, range(i, i + a.shape[ax]), axis=ax) for i in range(5))
    return a[::4, ::4]


LAT1, LON1 = np.arange(90.0, -90.5, -1.0), np.arange(0.0, 360.0, 1.0)


def cmd_era5_normal(a) -> int:
    """The surface normal in use (GEPS climatology era5_clim_sfc_2001-2020.nc, 1 deg) on the 1.5 deg grid, bilinear
    as era5_shift.npz was made, in GEFS units -> WORK/era5_normal_0120.npz (the check of era5-loso's ARCO rebuild)."""
    import xarray as xr

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    nrm = xr.open_dataset(Path(a.normals) / "era5_clim_sfc_2001-2020.nc")
    if not (np.allclose(nrm.lat.values, LAT1) and np.allclose(nrm.lon.values, LON1)):
        raise SystemExit("the surface normal is not on the 1 deg grid 90 -> -90, 0 -> 359")
    out = {t: regrid(nrm[t].values.astype("float64") * SFC_SCALE[t], LAT1, LON1, "lin") for t in SFC}
    nrm91 = xr.open_dataset(Path(a.normals) / "era5_clim_sfc_1991-2020.nc")
    sh = np.load(Path(a.shift))
    for t in SFC:  # the same interpolation must give era5_shift's difference back
        d = regrid(nrm91[t].values.astype("float64") * SFC_SCALE[t], LAT1, LON1, "lin") - out[t]
        err = np.abs(d - sh[f"s15_{t}"]).max()
        print(f"  {t}: interpolated 1991-2020 minus 2001-2020 vs era5_shift: max |d| {err:.2g}")
    p = work / "era5_normal_0120.npz"
    np.savez_compressed(p, **out, centres=SK.CENTRES)
    print(f"{p}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


def cmd_era5_days(a) -> int:
    """ERA5 from ARCO on the surface normal's days of years a.years (first,last): per day the mean of the four hours,
    to_1deg, bilinear to the 1.5 deg grid, GEFS units -> OUT/era5_days_<first>.npz (Actions; anonymous GCS)."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import xarray as xr

    y0, y1 = (int(x) for x in a.years.split(","))
    days = [d for d in normal_days("t2m") if y0 <= d.year <= y1]
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    lat, lon = ds.latitude.values, ds.longitude.values
    if not (np.allclose(lat, np.arange(90.0, -90.1, -0.25)) and np.allclose(lon, np.arange(0.0, 360.0, 0.25))):
        raise SystemExit("unexpected ARCO grid")
    # ERA5's precipitation rate carries packing noise just below zero (to -4e-5 mm/day): tolerated, not clipped, as in the
    # normal it must reproduce
    lim = {"t2m": (180.0, 340.0), "mslp": (85000.0, 110000.0), "pr": (-1e-3, 3000.0)}

    def day(d):
        t = [pd.Timestamp(d) + pd.Timedelta(hours=h) for h in SFC_HOURS]
        for attempt in range(4):
            try:
                out = {}
                for tag, var in SFC.items():
                    x = to_1deg(ds[var].sel(time=t).values.astype("float64").mean(0)) * SFC_SCALE[tag]
                    lo, hi = lim[tag]
                    if not (np.isfinite(x).all() and x.min() >= lo - 1e-6 and x.max() <= hi):
                        raise ValueError(
                            f"{d} {tag}: missing or out of range ({np.nanmin(x):.4g} .. {np.nanmax(x):.4g})"
                        )
                    out[tag] = regrid(x[None], LAT1, LON1, "lin")[0]
                return out
            except ValueError:
                raise
            except Exception:  # noqa: BLE001  (transient storage errors)
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(day, days))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"era5_days_{y0}.npz"
    np.savez(p, days=np.array([d.toordinal() for d in days]), **{t: np.stack([g[t] for g in got]) for t in SFC})
    print(f"{p}: {len(days)} days {days[0]} .. {days[-1]} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_era5_loso(a) -> int:
    """era5_loso_<tag>.npz for every tag: zg500 from the local daily store, the surface fields from the era5-days
    files in a.days; each rebuilt normal checked against the normal in use."""
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for tag in a.tags.split(","):
        t0 = time.time()
        days = normal_days(tag)
        L = Loso(days, normal_keys(tag, days))
        if tag == "zg500":
            import xarray as xr

            if not (a.store and a.normals):
                raise SystemExit("zg500 is built offline: give --store and --normals")
            i = 0
            for y in range(EY0, EY1 + 1):
                with xr.open_dataset(Path(a.store) / "z500" / f"z500_{y}.nc") as d:
                    v = d["z500"].transpose("time", "latitude", "longitude")  # the store's order varies: by name
                    if not np.allclose(v.latitude.values, R.LAT[::-1]):
                        raise SystemExit("the z500 store is not on the 1.5 deg grid -90 -> 90")
                    t = [dt.date.fromisoformat(str(x)[:10]) for x in v.time.values]
                    if t != days[i : i + len(t)]:
                        raise SystemExit(f"z500 {y}: days do not follow on")
                    x = v.values.astype("float64")[:, ::-1]  # -> 90 -> -90
                for k in range(len(t)):
                    L.add(i + k, x[k])
                i += len(t)
            ref = xr.open_dataset(Path(a.normals) / "era5_clim_2001-2020.nc")["zg500"].values[:, ::-1]
        else:
            got = {}
            for f in sorted(Path(a.days).glob("era5_days_*.npz")):
                with np.load(f) as z:
                    for o, x in zip(z["days"], z[tag]):
                        got[int(o)] = x
            missing = [d for d in days if d.toordinal() not in got]
            if missing:
                raise SystemExit(f"{tag}: {len(missing)} of the normal's days missing from {a.days}")
            for i, d in enumerate(days):
                L.add(i, got[d.toordinal()].astype("float64"))
            ref = np.load(work / "era5_normal_0120.npz")[tag]
        res = L.finish()
        check_normal(tag, res.pop("E"), ref)
        key = "common-year" if tag == "zg500" else "raw"
        meta = dict(
            tag=tag,
            normal=f"ERA5 {EY0}-{EY1}, {len(days)} days, +-{ERA5_HALF} d pools on the {key} day of year",
            season_days=LOSO,
            D="normal without the season minus the normal, GEFS units, 1.5 deg 90 -> -90",
        )
        p = work / f"era5_loso_{tag}.npz"
        np.savez_compressed(p, meta=np.array(json.dumps(meta)), centres=SK.CENTRES, **res)
        rms = np.sqrt(np.mean(res["D"][res["n_run"] > 0].astype("float64") ** 2))
        print(f"{p}: {L.R} seasons a centre, rms D {rms:.3g} {UNITS[tag]} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_skill(a) -> int:
    tags = a.tags.split(",")
    work = Path(a.work)
    out_dir = Path(a.ref) if not a.partial else work / "partial"  # a partial run never lands in the reference
    out_dir.mkdir(parents=True, exist_ok=True)
    S = starts(a.rf)
    want = {dt.date.fromisoformat(f"{s[:4]}-{s[4:6]}-{s[6:]}") for y in range(HY0, HY1 + 1) for s in R.wednesdays(y)}
    have = {s for s, _ in S}
    print(f"{len(S)} reforecast starts {min(have)} .. {max(have)}", flush=True)
    if have != want and not a.partial:
        raise SystemExit(f"{len(want - have)} of the {len(want)} reforecast starts missing; --partial to run anyway")
    t0 = time.time()
    # pass 1: the full-pool climatology sums per 5-day centre
    Ssum = {t: np.zeros((len(SK.CENTRES), 5, NY, NX)) for t in tags}
    Nn = np.zeros(len(SK.CENTRES))
    pool = pools_of(have)
    pools = [pool[s] for s, _ in S]
    for (_s, f), cs in zip(S, pools):
        n, F = member_sums(f, tags)
        for t in tags:
            fs = F[t].sum(0, dtype="float64")
            for ci in cs:
                Ssum[t][ci] += fs
        Nn[cs] += n
    print(f"  pass 1: climatology sums ({time.time() - t0:.0f} s)", flush=True)
    if not a.partial:  # check against the published climatology
        for c in (1, 61, 91, 181, 266, 361):
            with np.load(clim_week(c, a.clim, work)) as z:
                ci = (c - 1) // 5
                msg = [f"n {int(z['n'])}/{int(Nn[ci])}"]
                msg.extend(f"{t} max|d| {np.abs(Ssum[t][ci] / Nn[ci] - z[f'w_{t}']).max():.4f}" for t in tags)
            print(f"  check vs {R.CLIM_TAG} doy {c}: " + ", ".join(msg), flush=True)
    shift = np.load(a.shift)
    shifts = {t: shift[f"s15_{t}"].astype("float64") for t in tags}
    loso = {t: EraLeaveOut(work / f"era5_loso_{t}.npz") for t in tags}  # the ERA5 normal without the season
    wk_obs = {t: np.load(work / f"obsweek_{t}.npz")["wk"].astype("float32") for t in tags}  # float16 on disk
    mid = np.array([dt.date(2001, m, 15).timetuple().tm_yday for m in range(1, 13)])
    shp = (12, 5, NY, NX)
    keys = ("n", "sf", "so", "sff", "soo", "sfo", "sw", "sinv", "f2", "ff2", "fo2")
    acc = {t: {k: np.zeros(shp) for k in keys} for t in tags}
    cache = {}
    dates = [s for s, _ in S]
    # the paired weekly anomalies of the tercile fields, for the cross-validated test of the calibration (assess)
    pdir = work if not a.partial else work / "partial"
    PAIR = {
        t: {
            w: np.lib.format.open_memmap(
                pdir / f"pairs_{w}_{t}.npy", mode="w+", dtype="float16", shape=(len(S), 5, NY, NX)
            )
            for w in ("f", "o")
        }
        for t in tags
        if t in ("t2m", "pr")
    }
    np.save(pdir / "pairs_starts.npy", np.array([d.isoformat() for d in dates]))
    for si, (s, _f) in enumerate(S):
        # leave-season-out climatology at the two bracketing centres
        for j in [j for j in list(cache) if abs((dates[j] - s).days) > LOSO]:
            cache.pop(j)
        near = [j for j in range(max(0, si - 8), min(len(S), si + 9)) if abs((dates[j] - s).days) <= LOSO]
        for j in near:
            if j not in cache:
                n_j, F_j = member_sums(S[j][1], tags)
                cache[j] = (n_j, {t: F_j[t].sum(0, dtype="float64") for t in tags}, F_j)
        doy = SK.noleap_doy(s)
        i_lo, w_lo, i_hi, w_hi = SK.bracket(doy)
        n_m, F = cache[si][0], cache[si][2]
        months = np.where(np.abs(((doy - mid + 182.5) % 365) - 182.5) <= WIN)[0]
        vb = (s - B0).days // 7
        for t in tags:
            C = 0.0
            for ci, wgt in ((i_lo, w_lo), (i_hi, w_hi)):
                if wgt == 0:
                    continue
                num, den = Ssum[t][ci].copy(), Nn[ci]
                for j in near:
                    if ci in pools[j]:
                        num -= cache[j][1][t]
                        den -= cache[j][0]
                C = C + wgt * num / den
            an = F[t] - C[None] - (SK.weekly_shift(shifts[t], s) - loso[t].weekly(s))[None]  # (m, 5, lat, lon)
            fm = an.mean(0)
            w = an.var(0, ddof=1) if n_m > 1 else np.zeros_like(fm)
            f2 = an[1:3].mean(0) if n_m >= 3 else an[:2].mean(0)
            for k in range(5):
                if vb + k >= wk_obs[t].shape[0]:
                    continue
                ow = wk_obs[t][vb + k]
                if t in PAIR:
                    PAIR[t]["f"][si, k] = fm[k]
                    PAIR[t]["o"][si, k] = ow
                good = np.isfinite(fm[k]) & np.isfinite(ow)
                fw, o, ww, f2w = (np.where(good, x, 0.0) for x in (fm[k], ow, w[k], f2[k]))
                for m in months:
                    A = acc[t]
                    A["n"][m, k] += good
                    A["sf"][m, k] += fw
                    A["so"][m, k] += o
                    A["sff"][m, k] += fw * fw
                    A["soo"][m, k] += o * o
                    A["sfo"][m, k] += fw * o
                    A["sw"][m, k] += ww
                    A["sinv"][m, k] += good / n_m
                    A["f2"][m, k] += f2w
                    A["ff2"][m, k] += f2w * f2w
                    A["fo2"][m, k] += f2w * o
        if (si + 1) % 100 == 0 or si + 1 == len(S):
            print(f"  pass 2: {si + 1}/{len(S)} starts ({time.time() - t0:.0f} s)", flush=True)
    for t in PAIR:
        for w in PAIR[t]:
            PAIR[t][w].flush()
    del PAIR
    for t in tags:
        A = acc[t]
        n = A["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mf, mo = A["sf"] / n, A["so"] / n
            vf, vo = A["sff"] / n - mf**2, A["soo"] / n - mo**2
            cov = A["sfo"] / n - mf * mo
            r = cov / np.sqrt(vf * vo)
            slope = cov / vf
            icpt = mo - slope * mf
            resid = np.sqrt(np.clip(vo - slope * cov, 0, None) * n / np.clip(n - 2, 1, None))
            osd = np.sqrt(vo)
            noise = A["sw"] / n  # member noise variance
            inv = A["sinv"] / n  # mean 1/(members) of the pairs
            sig = vf - noise * inv  # signal variance of the 4-member mean
            mf2 = A["f2"] / n
            vf2 = A["ff2"] / n - mf2**2
            cov2 = A["fo2"] / n - mf2 * mo
            r2 = cov2 / np.sqrt(vf2 * vo)
            v4p = np.clip(vf2 - noise * (0.5 - inv), noise * inv, None)
            r4_from2 = cov2 / np.sqrt(v4p * vo)  # the 2 -> 4 member test
            v31 = np.clip(sig + noise / 31.0, noise / 31.0, None)
            r31 = cov / np.sqrt(v31 * vo)
        bad = n < MIN_PAIRS
        for x in (r, slope, icpt, resid, osd, r2, r4_from2, r31, sig, noise):
            x[bad] = np.nan
        meta = dict(
            tag=t,
            units=UNITS[t],
            truth=TRUTH[t],
            hindcast=f"GEFSv12 reforecast {HY0}-{HY1}, Wednesday starts, 4 members (c00 p01-p03), {len(S)} starts "
            f"(release {R.CLIM_TAG})",
            pool=f"init month: starts within +-{WIN} d of the 15th",
            min_pairs=MIN_PAIRS,
            model_clim=f"leave-season-out (starts within {LOSO} d of the verified start removed), common-year pools",
            rebase="minus the 7-day valid-date mean of era5_shift.npz (ERA5 1991-2020 - 2001-2020), its 2001-2020 "
            f"normal leave-season-out (its days within {LOSO} d of the valid date removed)",
            grid="GEFS 1.5 deg, lat 90 -> -90, lon 0 -> 358.5",
            skill_test=SK.test_note(),
            weeks="week k = init + 7k-7 ... init + 7k-1",
        )
        pub = {"r": r.astype("float16"), "n": np.clip(n, 0, 32767).astype("int16")}
        if t in ("t2m", "pr"):  # the calibration: r and the forecast sd
            fsd = np.sqrt(vf)
            fsd[bad] = np.nan
            pub.update(fsd=fsd.astype("float16"))
        np.savez_compressed(out_dir / f"gefs_mapskill_{t}.npz", meta=np.array(json.dumps(meta)), **pub)
        np.savez_compressed(
            work / f"diag_mapskill_{t}.npz",
            r=r,
            a=slope,
            b=icpt,
            s=resid,
            obs_sd=osd,
            n=n,
            mf=mf,
            mo=mo,
            fsd=np.sqrt(vf),
            r2=r2,
            r4_from2=r4_from2,
            r31=r31,
            sig=sig,
            noise=noise,
            inv=inv,
        )
        rows = (R.LAT >= 30) & (R.LAT <= 70)

        def share(x, rows=rows):  # x one week's full map: the test runs over the map, the share over the rows
            sk, ok = ~SK.noskill(x)[rows], np.isfinite(x[rows])
            return np.mean(sk[ok]) * 100

        print(
            f"  {t}: gefs_mapskill_{t}.npz ({(out_dir / f'gefs_mapskill_{t}.npz').stat().st_size / 1e6:.1f} MB); "
            f"median r (Sep, 30-70N) by week: "
            + " ".join(f"{np.nanmedian(r[8, k][rows]):.2f}" for k in range(5))
            + "; share significant: "
            + " ".join(f"{share(r[8, k]):.0f}%" for k in range(5)),
            flush=True,
        )
    return 0


def _probs(x, rr):
    """Tercile probabilities (below, above) of N(rr x, 1 - rr^2) against +-Z3 (gefs_probs' calibration)."""
    from scipy.special import ndtr

    den = np.sqrt(1.0 - rr**2)
    return ndtr((-Z3 - rr * x) / den), 1.0 - ndtr((Z3 - rr * x) / den)


def cmd_assess(a) -> int:
    """Cross-validated test of the calibrated tercile probabilities on the reforecast pairs saved by `skill`.

    Every start is scored with the table of its own calendar month (the live usage), fitted on the +-45 d pool WITHOUT
    the starts within 60 days of any start being scored (its season, across New Year too); the observed categories
    are the terciles of the fit set's observed anomalies, the forecast is centred on the fit set's forecast mean. Two
    calibrations: gefs_probs' standardised form, and GEPS's prob_maps form (median + a (m - median) against q33/q67,
    residual sd s). Score: RPSS against the one-third climatology, per cell (month, week) and pooled; reliability of
    P(below) and P(above) in 10 bins. The hindcast has 4 members; the live mean has 31, so live skill is at least this.
    -> WORK/assess_{tag}.npz, printed summary."""
    work = Path(a.work)
    pdir = work if not a.partial else work / "partial"
    dates = [dt.date.fromisoformat(str(x)) for x in np.load(pdir / "pairs_starts.npy")]
    ords = np.array([d.toordinal() for d in dates])
    doy = SK.noleap_doy(dates)
    mon = np.array([d.month for d in dates])
    yrs = np.array([d.year for d in dates])
    mid = np.array([dt.date(2001, m, 15).timetuple().tm_yday for m in range(1, 13)])
    wt = np.repeat(np.cos(np.deg2rad(R.LAT)), NX)
    for t in [x for x in a.tags.split(",") if x in ("t2m", "pr")]:
        t0 = time.time()
        PF = np.load(pdir / f"pairs_f_{t}.npy", mmap_mode="r")
        PO = np.load(pdir / f"pairs_o_{t}.npy", mmap_mode="r")
        shp = (12, 5, NY * NX)
        acc = {k: np.zeros(shp) for k in ("rps_new", "rps_geps", "rps_clim", "n")}
        # the same sums split by the FIT set's r (the mask as the live product would have drawn it, out of sample)
        spl = {k: np.zeros((5, 2)) for k in ("rps_new", "rps_geps", "rps_clim", "n")}
        rel = {k: np.zeros((5, 2, 10)) for k in ("new_p", "new_o", "new_w", "geps_p", "geps_o", "geps_w")}
        for m in range(12):
            pool = np.where(np.abs(((doy - mid[m] + 182.5) % 365) - 182.5) <= WIN)[0]
            for k in range(5):
                F = PF[pool, k].reshape(len(pool), -1).astype("float64")
                O = PO[pool, k].reshape(len(pool), -1).astype("float64")
                okc = np.isfinite(O).all(0) & np.isfinite(F).all(0)  # cells with every pair (land for pr)
                F, O = F[:, okc], O[:, okc]
                for y in sorted(set(yrs[mon == m + 1])):
                    tgt = np.where((mon == m + 1) & (yrs == y))[0]
                    lo, hi = ords[tgt].min() - 60, ords[tgt].max() + 60
                    fit = (ords[pool] < lo) | (ords[pool] > hi)
                    sc = np.searchsorted(pool, tgt)  # scored starts inside the pool
                    if fit.sum() < MIN_PAIRS or not np.array_equal(pool[sc], tgt):
                        continue
                    f, o = F[fit], O[fit]
                    mf, mo = f.mean(0), o.mean(0)
                    vf, vo = f.var(0), o.var(0)
                    cov = ((f - mf) * (o - mo)).mean(0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = cov / np.sqrt(vf * vo)
                        slope = cov / vf
                        s = np.sqrt(np.clip(vo - slope * cov, 0, None) * len(f) / (len(f) - 2))
                    q33, med, q67 = np.percentile(o, [100 / 3, 50, 200 / 3], axis=0)
                    fs, os_ = F[sc], O[sc]
                    I1 = (os_ < q33).astype(float)
                    I3 = (os_ > q67).astype(float)
                    # this port's standardised form
                    rr = np.clip(np.nan_to_num(r), 0.0, 0.99)
                    pb, pa = _probs((fs - mf) / np.sqrt(vf), rr)
                    # GEPS prob_maps form
                    from scipy.special import ndtr

                    mu = med + slope * (fs - med)
                    gb, ga = ndtr((q33 - mu) / s), 1.0 - ndtr((q67 - mu) / s)

                    def rps(b, a_, I1=I1, I3=I3):
                        return (b - I1) ** 2 + ((1 - a_) - (1 - I3)) ** 2

                    cols = np.where(okc)[0]
                    R_new, R_geps = rps(pb, pa), rps(gb, ga)
                    R_clim = rps(np.full_like(pb, 1 / 3), np.full_like(pa, 1 / 3))
                    acc["rps_new"][m, k, cols] += R_new.sum(0)
                    acc["rps_geps"][m, k, cols] += R_geps.sum(0)
                    acc["rps_clim"][m, k, cols] += R_clim.sum(0)
                    acc["n"][m, k, cols] += len(sc)
                    w = np.broadcast_to(wt[cols], pb.shape)
                    skl = np.isfinite(r) & ~SK.noskill(r)
                    for j, msk in enumerate((~skl, skl)):
                        for name, X in (("rps_new", R_new), ("rps_geps", R_geps), ("rps_clim", R_clim)):
                            spl[name][k, j] += (X[:, msk] * wt[cols][msk]).sum()
                        spl["n"][k, j] += len(sc) * wt[cols][msk].sum()
                    for name, P3 in (("new", (pb, pa)), ("geps", (gb, ga))):
                        for ci, (P, I) in enumerate(zip(P3, (I1, I3))):
                            b = np.clip((P * 10).astype(int), 0, 9)
                            rel[f"{name}_p"][k, ci] += np.bincount(b.ravel(), (P * w).ravel(), 10)
                            rel[f"{name}_o"][k, ci] += np.bincount(b.ravel(), (I * w).ravel(), 10)
                            rel[f"{name}_w"][k, ci] += np.bincount(b.ravel(), w.ravel(), 10)
            print(f"  {t}: month {m + 1} ({time.time() - t0:.0f} s)", flush=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rpss_new = 1 - acc["rps_new"] / acc["rps_clim"]
            rpss_geps = 1 - acc["rps_geps"] / acc["rps_clim"]
        np.savez_compressed(
            work / f"assess_{t}.npz",
            rpss_new=rpss_new.astype("float32"),
            rpss_geps=rpss_geps.astype("float32"),
            n=acc["n"].astype("int16"),
            **{f"split_{k}": v for k, v in spl.items()},
            **rel,
        )
        print(
            f"  {t}: cross-validated RPSS vs climatology (area-weighted, pooled over months, cells with truth; "
            f"unhatched/hatched = out-of-sample r significant or not: {SK.test_note()}):"
        )
        test = {
            "test": "RPSS of the calibrated tercile probabilities vs one third each, every 2000-2019 reforecast start "
            "scored with its init month's table refitted without its season (starts within 60 d), "
            "categories = the fit years' observed terciles; area-weighted, cells with truth; 4-member "
            "hindcast means (the 31-member live mean is at least as skilful)",
            "weeks": [],
        }
        for k in range(5):

            def ss(name, j, spl=spl, k=k):
                return float(1 - spl[name][k, j] / spl["rps_clim"][k, j])

            hi = rel["new_w"][k][:, 4:].sum()
            test["weeks"].append(
                {
                    "week": k + 1,
                    "rpss": round(float(1 - spl["rps_new"][k].sum() / spl["rps_clim"][k].sum()), 3),
                    "rpss_unhatched": round(ss("rps_new", 1), 3),
                    "rpss_hatched": round(ss("rps_new", 0), 3),
                    "share_unhatched": round(float(spl["n"][k, 1] / spl["n"][k].sum()), 3),
                    "rpss_geps_form": round(float(1 - spl["rps_geps"][k].sum() / spl["rps_clim"][k].sum()), 3),
                    "reliability_p_ge_0.4": {
                        "forecast": round(float(rel["new_p"][k][:, 4:].sum() / hi), 3),
                        "observed": round(float(rel["new_o"][k][:, 4:].sum() / hi), 3),
                    },
                }
            )
        if not a.partial:  # the result travels with the published skill file
            q = Path(a.ref) / f"gefs_mapskill_{t}.npz"
            z = dict(np.load(q))
            meta = json.loads(str(z.pop("meta")))
            meta["calibration_test"] = test
            np.savez_compressed(q, meta=np.array(json.dumps(meta)), **z)
        for k in range(5):
            line = []
            for name in ("rps_new", "rps_geps"):

                def ss(j, spl=spl, name=name, k=k):
                    return 1 - spl[name][k, j] / spl["rps_clim"][k, j]

                tot = 1 - spl[name][k].sum() / spl["rps_clim"][k].sum()
                line.append(f"{name[4:]} all {tot:+.3f} unhatched {ss(1):+.3f} hatched {ss(0):+.3f}")
            frac = spl["n"][k, 1] / spl["n"][k].sum()
            print(
                f"    week {k + 1} ({frac * 100:.0f}% unhatched): "
                + " | ".join(line)
                + f" | reliability (new, P>=0.4 bins) fcst {np.nansum(rel['new_p'][k][:, 4:]) / np.nansum(rel['new_w'][k][:, 4:]):.3f}"
                + f" obs {np.nansum(rel['new_o'][k][:, 4:]) / np.nansum(rel['new_w'][k][:, 4:]):.3f}",
                flush=True,
            )
    return 0


def step_blocks():
    """(first, end) observed week blocks the starts of STEP_Y0..STEP_Y1 verify on (weeks 1-5 of every start)."""
    first = dt.date(STEP_Y0, 1, 1)
    while first.weekday() != 2:
        first += dt.timedelta(days=1)
    last = dt.date(STEP_Y1, 12, 31)
    while last.weekday() != 2:
        last -= dt.timedelta(days=1)
    return (first - B0).days // 7, (last - B0).days // 7 + len(SK.WEEKS)


def cmd_step_truth(a) -> int:
    """The observed tercile of every week the overlap's starts verify on, against the live maps' 2016-2025 terciles at
    the week's centre (0 below, 1 near, 2 above; -1 no truth, or a dry precipitation week) -> STEP_OBS."""
    from gefs_probs import DRY_MM

    b0, b1 = step_blocks()
    out = {}
    for tag in a.tags.split(","):
        wk = np.load(Path(a.work) / f"obsweek_{tag}.npz")["wk"][b0:b1].astype("float32")
        terc = np.load(Path(a.ref) / f"gefs_terciles_{tag}.npz")
        cat = np.full(wk.shape, -1, "int8")
        for i in range(len(wk)):
            doy = SK.noleap_doy(B0 + dt.timedelta(days=7 * (b0 + i) + 3))  # the week's centre, as gefs_probs reads
            q33, q67 = (SK.at_doy(terc[n].astype("float64"), doy) for n in ("q33", "q67"))
            ok = np.isfinite(wk[i]) & np.isfinite(q33) & np.isfinite(q67)
            if tag == "pr":
                ok &= SK.at_doy(terc["clim_weekly"].astype("float64"), doy) >= DRY_MM
            cat[i] = np.where(ok, np.where(wk[i] < q33, 0, np.where(wk[i] > q67, 2, 1)), -1)
        out[f"cat_{tag}"] = cat
        f = [np.mean(cat[cat >= 0] == k) for k in range(3)]
        print(f"  {tag}: {len(wk)} weeks, observed below/near/above {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}", flush=True)
    meta = dict(
        first_week=(B0 + dt.timedelta(days=7 * b0)).isoformat(),
        categories="0 below, 1 near, 2 above the 2016-2025 terciles of the 7-day-mean anomaly (gefs_terciles_<tag>) at "
        "the week's centre; -1 no truth or a dry week (precipitation)",
        truth={t: TRUTH[t] for t in a.tags.split(",")},
    )
    np.savez_compressed(STEP_OBS, meta=np.array(json.dumps(meta)), **out)
    print(f"{STEP_OBS}  {STEP_OBS.stat().st_size / 1e6:.2f} MB")
    return 0


def mean_test(x):
    """Mean of a series of consecutive weekly starts with its 95% interval and two-sided p: t on the effective sample
    size n (1 - r1) / (1 + r1), r1 the lag-1 autocorrelation (>= 0)."""
    from scipy.stats import t as tdist

    x = np.asarray(x, dtype="float64")
    n = len(x)
    r1 = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], 0.0, 0.99)) if x.std() > 0 else 0.0
    neff = max(n * (1 - r1) / (1 + r1), 3.0)
    se = x.std(ddof=1) / np.sqrt(neff)
    m = float(x.mean())
    half = float(tdist.ppf(0.975, neff - 1) * se)
    p = float(2 * tdist.sf(abs(m) / se, neff - 1)) if se > 0 else 1.0
    return dict(mean=round(m, 4), ci=[round(m - half, 4), round(m + half, 4)], p=round(p, 4), n_eff=round(neff, 1))


def cmd_step(a) -> int:
    """The tercile step test (docstring) on the reforecast starts of STEP_Y0..STEP_Y1 -> a.out (json)."""
    from quantmet.stats import benjamini_hochberg

    tags = a.tags.split(",")
    work, ref = Path(a.work), Path(a.ref)
    files = dict(starts(a.rf))
    dates = sorted(files)
    pools = pools_of(dates)
    scored = [d for d in dates if STEP_Y0 <= d.year <= STEP_Y1]
    print(f"{len(scored)} scored starts {scored[0]} .. {scored[-1]}, {len(dates)} in the bundles", flush=True)
    shift = {t: np.load(a.shift)[f"s15_{t}"].astype("float64") for t in tags}
    loso = {t: EraLeaveOut(work / f"era5_loso_{t}.npz") for t in tags}
    terc = {t: np.load(ref / f"gefs_terciles_{t}.npz") for t in tags}
    sk = {t: SK.load(t, ref) for t in tags}
    obs = np.load(a.obs)
    b_first = dt.date.fromisoformat(json.loads(str(obs["meta"]))["first_week"])
    clim, sums = {}, {}

    def centre(ci):  # the release's member-runs and member-run mean weekly fields of a 5-day centre
        if ci not in clim:
            with np.load(clim_week(int(SK.CENTRES[ci]), a.clim, work)) as z:
                clim[ci] = (float(z["n"]), {t: z[f"w_{t}"].astype("float64") for t in tags})
        return clim[ci]

    def member_sum(d):
        if d not in sums:
            n, F = member_sums(files[d], tags)
            sums[d] = (n, {t: F[t].sum(0, dtype="float64") for t in tags})
        return sums[d]

    wt = np.repeat(np.cos(np.deg2rad(R.LAT))[:, None], NX, 1)
    keys = ("w", "p0", "p1", "p2", "i0", "i1", "i2", "rps", "rpsc")
    per = {(t, k): [] for t in tags for k in range(5)}  # per start: area sums of keys (all cells, then unhatched)
    rel = {(t, k, c): np.zeros((3, 10)) for t in tags for k in range(5) for c in (0, 2)}  # w, w p, w I per bin
    mask = {}
    for s in scored:
        for old in [d for d in sums if abs((d - s).days) > LOSO]:
            sums.pop(old)
        mates = [d for d in dates if abs((d - s).days) <= LOSO]
        i_lo, w_lo, i_hi, w_hi = SK.bracket(SK.noleap_doy(s))
        _, F = member_sums(files[s], tags)
        mi = s.month - 1
        for t in tags:
            C = 0.0
            for ci, wgt in ((i_lo, w_lo), (i_hi, w_hi)):
                if wgt == 0:
                    continue
                n_c, w_c = centre(ci)
                own = [member_sum(d) for d in mates if ci in pools[d]]
                C = C + wgt * (n_c * w_c[t] - sum(v[1][t] for v in own)) / (n_c - sum(v[0] for v in own))
            an = F[t].mean(0) - C - SK.weekly_shift(shift[t], s) + loso[t].weekly(s)  # (5, lat, lon) vs 1991-2020
            for k in range(5):
                doy = SK.noleap_doy(s + dt.timedelta(days=7 * k + 3))  # the week's centre (gefs_probs)
                r, fsd = (sk[t][n][mi, k].astype("float64") for n in ("r", "fsd"))
                if (t, mi, k) not in mask:
                    mask[(t, mi, k)] = ~SK.noskill(r)
                rr = np.clip(r, 0.0, 0.99)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pb, pa = _probs((an[k] - SK.at_doy(terc[t]["mean"].astype("float64"), doy)) / fsd, rr)
                cat = obs[f"cat_{t}"][(s - b_first).days // 7 + k]
                ok = (cat >= 0) & np.isfinite(pb) & np.isfinite(pa)
                P = (pb, 1.0 - pb - pa, pa)
                I = [(cat == c).astype("float64") for c in range(3)]
                rps = (pb - I[0]) ** 2 + (pa - I[2]) ** 2
                rpsc = (1 / 3 - I[0]) ** 2 + (1 / 3 - I[2]) ** 2
                row = []
                for sel in (ok, ok & mask[(t, mi, k)]):
                    w = np.where(sel, wt, 0.0)
                    row += [w.sum(), *[(w * np.nan_to_num(p)).sum() for p in P], *[(w * i).sum() for i in I]]
                    row += [(w * np.nan_to_num(rps)).sum(), (w * rpsc).sum()]
                per[(t, k)].append(row)
                for c in (0, 2):
                    b = np.clip((np.nan_to_num(P[c]) * 10).astype(int), 0, 9)[ok]
                    w = wt[ok]
                    rel[(t, k, c)] += [np.bincount(b, x, 10) for x in (w, w * P[c][ok], w * I[c][ok])]
    out = {
        "test": "the live tercile maps' calibrated probabilities (gefs_probs: published r and forecast sd, 2016-2025 "
        f"terciles and recent mean) on every reforecast start of {STEP_Y0}-{STEP_Y1} from a leave-season-out "
        "climatology, against the observed tercile of each week (scripts/gefs/data/gefs_tercile_obs.npz); "
        "area-weighted over cells with truth; per-start area means tested with t on the effective number of weekly "
        "starts (lag-1 autocorrelation), bias flagged by Benjamini-Hochberg at FDR 10 % over fields, weeks and "
        "categories; the ERA5 re-basing normal leaves the season out, r and sd include the scored years",
        "starts": len(scored),
        "fields": {},
    }
    flags = []
    for t in tags:
        rows = []
        for k in range(5):
            A = np.array(per[(t, k)])
            res = {"week": k + 1}
            for part, j in (("all", 0), ("unhatched", len(keys))):
                X = dict(zip(keys, A[:, j : j + len(keys)].T))
                g = X["w"] > 0
                tot = {n: float(v[g].sum()) for n, v in X.items()}
                cell = {"share": round(tot["w"] / A[:, 0].sum(), 3)}
                res[part] = cell
                if not tot["w"]:  # no unhatched cell in this month and week
                    continue
                cell.update(
                    observed={c: round(tot[f"i{i}"] / tot["w"], 4) for i, c in enumerate(CATS)},
                    forecast={c: round(tot[f"p{i}"] / tot["w"], 4) for i, c in enumerate(CATS)},
                    rpss=round(1.0 - tot["rps"] / tot["rpsc"], 4),
                )
                if part == "all":
                    for i, c in enumerate(CATS):
                        cell[f"bias_{c}"] = mean_test((X[f"p{i}"][g] - X[f"i{i}"][g]) / X["w"][g])
                        cell[f"freq_{c}_minus_third"] = mean_test(X[f"i{i}"][g] / X["w"][g] - 1 / 3)
                    gain = mean_test((X["rpsc"][g] - X["rps"][g]) / X["w"][g])  # > 0: better than a third each
                    gain["p_one_sided"] = round(gain["p"] / 2 if gain["mean"] > 0 else 1 - gain["p"] / 2, 4)
                    cell["rps_gain"] = gain
                    cell["reliability"] = {
                        c: {
                            "w": rel[(t, k, i)][0].round(1).tolist(),
                            "p": [round(float(x / w), 3) if w else None for x, w in zip(*rel[(t, k, i)][[1, 0]])],
                            "o": [round(float(x / w), 3) if w else None for x, w in zip(*rel[(t, k, i)][[2, 0]])],
                        }
                        for i, c in ((0, "below"), (2, "above"))
                    }
            rows.append(res)
            flags += [(t, k, f"{test}_{c}") for c in CATS for test in ("bias", "freq")]
        out["fields"][t] = rows
    for test in ("bias", "freq"):  # one family per question: is the step biased, is the overlap a typical third
        fam = [f for f in flags if f[2].startswith(test)]
        cells = [out["fields"][t][k]["all"][name + ("_minus_third" if test == "freq" else "")] for t, k, name in fam]
        for c, s_ in zip(cells, benjamini_hochberg(np.array([c["p"] for c in cells]), 0.10)):
            c["significant"] = bool(s_)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for t in tags:
        print(
            f"  {t}: week  observed b/n/a (*: off a third, FDR)  forecast b/n/a      bias b / a (95% CI, FDR)       "
            "RPSS  (unhatched)"
        )
        for res in out["fields"][t]:
            x = res["all"]
            o, f = x["observed"], x["forecast"]

            def b(c, x=x):
                y = x[f"bias_{c}"]
                return f"{y['mean']:+.3f} [{y['ci'][0]:+.3f},{y['ci'][1]:+.3f}]{'*' if y['significant'] else ''}"

            def q(c, x=x, o=o):
                return f"{o[c]:.3f}{'*' if x[f'freq_{c}_minus_third']['significant'] else ' '}"

            u = res["unhatched"]
            print(
                f"     {res['week']}  {q('below')}/{q('near')}/{q('above')}               "
                f"{f['below']:.3f}/{f['near']:.3f}/{f['above']:.3f}  {b('below')} / {b('above')}  {x['rpss']:+.3f} "
                + (f"({u['rpss']:+.3f}, {u['share'] * 100:.0f}% cells)" if "rpss" in u else "(no unhatched cell)"),
                flush=True,
            )
    print(f"{a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    cmds = {
        "truth": cmd_truth,
        "recent": cmd_recent,
        "step-truth": cmd_step_truth,
        "era5-normal": cmd_era5_normal,
        "era5-days": cmd_era5_days,
        "era5-loso": cmd_era5_loso,
        "skill": cmd_skill,
        "assess": cmd_assess,
        "step": cmd_step,
    }
    tags = {"truth": "t2m,pr,zg500,mslp", "skill": "t2m,pr,zg500,mslp", "era5-loso": "zg500"}
    for name in cmds:
        p = sp.add_parser(name)
        if name not in ("era5-normal", "era5-days"):  # those two cover the surface fields of SFC
            p.add_argument("--tags", default=tags.get(name, "t2m,pr"))
        p.add_argument(
            "--work", default=WORK and str(WORK), help="observed weeks, ERA5 leave-out files, pairs, diagnostics"
        )
        p.add_argument("--ref", default=REF and str(REF), help="gefs_terciles_* (read) and gefs_mapskill_* (written)")
        if name in ("skill", "assess"):
            p.add_argument("--partial", action="store_true", help="run on the reforecast starts present")
        if name in ("skill", "step"):
            p.add_argument("--rf", required=True, help="the weekly hindcast bundles gefs_hindw_MM.npz")
            p.add_argument("--clim", help="gefs_clim_DDD.npz of the release (default: downloaded into WORK/clim)")
        if name in ("skill", "step", "era5-normal"):
            p.add_argument("--shift", default=SHIFT and str(SHIFT), help="era5_shift.npz (release gefs-ref-v1)")
        if name == "step":
            p.add_argument("--obs", default=str(STEP_OBS))
            p.add_argument("--out", required=True)
        if name == "era5-days":
            p.add_argument("--years", required=True, help="first,last")
            p.add_argument("--out", required=True)
            p.add_argument("--workers", type=int, default=8)
        if name in ("era5-normal", "era5-loso"):
            p.add_argument(
                "--normals",
                required=name == "era5-normal",
                help="the GEPS page's ERA5 normals era5_clim{,_sfc}_{1991,2001}-2020.nc (zg500, era5-normal)",
            )
        if name == "era5-loso":
            p.add_argument("--store", help="the 1.5 deg global daily ERA5 store, z500/z500_YYYY.nc (zg500)")
            p.add_argument("--days", help="era5_days_*.npz (the surface fields)")
    a = ap.parse_args()
    if a.cmd != "era5-days" and None in (a.work, a.ref, getattr(a, "shift", "")):
        raise SystemExit("pass --work, --ref and --shift, or set DATA_ARCHIVE for their defaults")
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
