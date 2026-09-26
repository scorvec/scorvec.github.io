#!/usr/bin/env python3
"""Reference build (laptop, once) for the GEFS page's weekly skill mask and tercile maps - the GEFS port of the GEPS
page's hindcast_maps.py and recent_median.py (~/data_archive/geps_subx; 2026-09-26, user: "the GEFS pages should
have all the same maps/charts the GEPS one does").

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

FORECAST - every GEFSv12 reforecast start 2000-2019 (Wednesdays, 4 members, weekly means, ~/data_archive/gefs_rf):
  anomaly = [F - M(start doy, week)] - shift(valid dates)
  M is the reforecast climatology read as gefs_live.clim_at reads it (5-day centres pooling the starts within +-15
  days of any year, linear in between) but LEAVE-SEASON-OUT: every start within 40 days of the verified start - its
  own season, including the neighbouring calendar year across New Year - is removed from the pool, so the
  forecast being verified is never inside its own climatology, as a live forecast never is. (GEPS de-drifted its
  hindcasts against the full 2001-2020 climatology, the verified year included; in-sample by ~5 % of the pool.) Pass 1
  rebuilds the full-pool climatology from the same files and checks it against the release gefs-clim-v1.
  shift = the 7-day mean over the week's valid dates of era5_shift.npz (ERA5 1991-2020 minus 2001-2020), the maps'
  re-basing - applied to precipitation too (GEPS left its precipitation un-shifted; the brief is one convention).

STATISTICS - GEPS's recipe: pairs pooled by calendar month (starts within +-45 days of the 15th), reduced at every
gridpoint and week to the anomaly correlation r, the regression obs = a*fcst + b, its residual sd s (n-2
denominator) and the observed sd; NaN where fewer than 30 pairs. Diagnostics kept alongside (work dir, not published):
the within-ensemble variance of the four members and the same statistics for a 2-member mean, which test how much a
4-member hindcast understates the skill of the 31-member live mean (reported, not applied - see the report).

RECENT-DECADE TERCILES (t2m, pr; recent_median.py) - the 33rd/67th percentiles and the mean of the 7-day-mean
anomaly (vs 1991-2020) over 2016-2025, windows whose centre is within +-15 days of each 5-day centre, plus (pr) the
absolute 1991-2020 weekly climatology for the dry-week mask. Computed on the 1.5 deg truth itself, so thresholds,
skill and members are all on one grid and one box size (GEPS computed them at 2.5 deg and interpolated to its 1 deg
members; for t2m the two agree to r 0.98-0.997 and within 0.01 K in the mean).

CALIBRATION TEST (assess) - every start scored with its own init month's table refitted without its season, RPSS of
the calibrated terciles vs one third each (2026-09-26, 1043 starts; area-weighted, cells with truth):
    t2m  weeks 1-5  +0.46 +0.24 +0.13 +0.10 +0.08  (unhatched cells +0.46 ... +0.14, hatched +0.02 ... +0.04);
         reliable: P >= 0.4 bins forecast 0.71/0.60/0.54/0.53/0.52 vs observed 0.72/0.61/0.55/0.54/0.53
    pr   weeks 1-5  +0.18 +0.04 +0.01 +0.00 +0.00  (hatched cells ~0: the mask marks where there is nothing)
         GEPS's prob_maps form on the same pairs: +0.14 -0.00 -0.04 -0.04 -0.05 (worse than climatology from week 2 -
         the skewed-terciles artefact gefs_probs.py explains). t2m: the two forms agree to 0.003.
Other checks: pass 1 reproduces gefs-clim-v1 exactly (n and means, max |d| 0.004 Pa); the re-based hindcast
anomalies are unbiased against the observed ones (2000-2019 mean forecast minus observed: t2m +0.01 K, pr -0.03
mm/day, zg500 -0.5 m, mslp +3 Pa); the 2-member -> 4-member skill predicted from the member spread matches the
measured 4-member skill to 0.01 (t2m week 3: 0.400 vs 0.406), so a 31-member mean would score higher still
(t2m week 3 r ~0.47 vs 0.41) - not applied, the per-cell estimate amplifies sampling noise where skill is small.

    python build_gefs_mapskill.py truth  --tags t2m,pr,zg500,mslp     # ~1-2 min a tag; work/obs*_{tag}.np*
    python build_gefs_mapskill.py recent --tags t2m,pr                # -> REF/gefs_terciles_{tag}.npz
    python build_gefs_mapskill.py skill  --tags t2m,pr,zg500,mslp     # ~1.5 min -> REF/gefs_mapskill_{tag}.npz
    python build_gefs_mapskill.py assess --tags t2m,pr                # ~4 min; adds the test to the skill files' meta
Memory: one year of CPC (0.4 GB) or the recent decade of one field (~1.5 GB); the skill pass ~3.8 GB (four fields).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R                                            # noqa: E402  (grid, unpack, weeks)
import gefs_skill as SK                                                # noqa: E402  (dates, centres, shift)

TC = Path.home() / "data_archive" / "geps_subx" / "telecon"
NCEP, CPCP = TC / "ncep", TC / "cpc_precip"
RF = (Path.home() / "data_archive" / "gefs_rf",                           # rf-YYYY/rf_YYYYMMDD.npz, first hit wins
      Path.home() / "data_archive" / "gefs_port_test" / "probs" / "rf")
REF = Path.home() / "data_archive" / "gefs_ref" / "probs"
WORK = Path.home() / "data_archive" / "gefs_port_test" / "probs" / "work"
SHIFT = Path.home() / "data_archive" / "gefs_ref" / "era5_shift.npz"
CLIM_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v1/gefs_clim_{c:03d}.npz"
CY0, CY1 = 1991, 2020                 # observed normal
HY0, HY1 = 2000, 2019                 # reforecast years
RY0, RY1 = 2016, 2025                 # recent decade for the terciles
WIN = 45                              # +-days around the 15th: the calendar-month pool
LOSO = 40                             # leave-season-out: drop pool starts within this many days of the verified start
MIN_PAIRS = 30
LAND_MIN = 0.25                       # CPC land cover needed in a 1.5 deg box
B0 = dt.date(2000, 1, 5)              # first reforecast start = first observed week block
Z3 = 0.4307272992954576               # standard normal upper tercile (gefs_probs.Z3)
NY, NX = len(R.LAT), len(R.LON)
UNITS = {"t2m": "K", "pr": "mm/day", "zg500": "m", "mslp": "Pa"}
TRUTH = {"t2m": "NCEP/NCAR R1 2 m air", "pr": "CPC unified gauge (land only), conservative 1.5 deg box mean",
         "zg500": "NCEP/NCAR R1 500 hPa height", "mslp": "NCEP/NCAR R1 sea-level pressure"}


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
            t = t % 360.0
            if t < s[0]:
                t += 360.0
            j = int(np.searchsorted(ext, t, side="right")) - 1
            j = min(j, len(s) - 1)
            f = (t - ext[j]) / (ext[j + 1] - ext[j])
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
        return np.concatenate([regrid(a[i:i + chunk], lat, lon, kind, chunk) for i in range(0, a.shape[0], chunk)])
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
        p = CPCP / f"precip.{y}.nc"
        var, kind = "precip", "box"
    elif tag == "t2m":
        p = NCEP / f"air.2m.gauss.{y}.nc"
        var, kind = "air", "lin"
    else:
        p = NCEP / (f"hgt.{y}.nc" if tag == "zg500" else f"slp.{y}.nc")
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
    if tag == "mslp" and np.nanmax(v) < 2000:                       # hPa file -> Pa (R1 slp is Pa already)
        v = v * 100.0
    _, ix = np.unique(np.array([x.toordinal() for x in t]), return_index=True)
    return [t[i] for i in ix], regrid(v[ix], lat, lon, kind)


def doy_clim(sums, cnt):
    """GEPS build_telecon_patterns.doy_clim: day-of-year mean, gaps filled, 31-day circular running mean."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sums / cnt
    ok = np.isfinite(c).astype("float64")
    ext_v = np.concatenate([np.nan_to_num(c[-15:]), np.nan_to_num(c), np.nan_to_num(c[:15])])
    ext_o = np.concatenate([ok[-15:], ok, ok[:15]])
    cs_v = np.cumsum(np.concatenate([np.zeros((1,) + c.shape[1:]), ext_v]), 0)
    cs_o = np.cumsum(np.concatenate([np.zeros((1,) + c.shape[1:]), ext_o]), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = (cs_v[31:] - cs_v[:-31]) / (cs_o[31:] - cs_o[:-31])
    sm[(cs_o[31:] - cs_o[:-31]) < 16] = np.nan                     # at least half the 31-day window observed
    return sm.astype("float32")                                      # (366, lat, lon)


def cmd_truth(a) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    last = dt.date(HY1, 12, 31)
    while last.weekday() != 2:
        last -= dt.timedelta(days=1)
    nblk = (last - B0).days // 7 + len(SK.WEEKS)
    for tag in a.tags.split(","):
        t0 = time.time()
        sums = np.zeros((366, NY, NX)); cnt = np.zeros((366, NY, NX))
        for y in range(CY0, CY1 + 1):
            r = daily_year(tag, y)
            if r is None:
                raise SystemExit(f"{tag}: truth year {y} missing")
            t, v = r
            for i, d in enumerate(t):
                j = d.timetuple().tm_yday - 1
                ok = np.isfinite(v[i])
                sums[j] += np.where(ok, v[i], 0.0); cnt[j] += ok
        clim = doy_clim(sums, cnt)
        np.save(WORK / f"obsclim_{tag}.npy", clim)
        del sums, cnt
        print(f"  {tag}: 1991-2020 clim ({time.time() - t0:.0f} s)", flush=True)
        bs = np.zeros((nblk, NY, NX), "float32"); bc = np.zeros((nblk, NY, NX), "int8")
        for y in range(HY0, HY1 + 2):
            t, v = daily_year(tag, y)
            for i, d in enumerate(t):
                b = (d - B0).days
                if b < 0 or b // 7 >= nblk:
                    continue
                an = v[i] - clim[d.timetuple().tm_yday - 1]
                ok = np.isfinite(an)
                bs[b // 7] += np.where(ok, an, 0.0); bc[b // 7] += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            wk = np.where(bc >= 5, bs / np.maximum(bc, 1), np.nan).astype("float32")
        np.savez(WORK / f"obsweek_{tag}.npz", wk=wk, b0=np.array(B0.isoformat()), nblk=nblk)
        fin = np.isfinite(wk)
        print(f"  {tag}: {nblk} observed weeks from {B0}, {fin.mean() * 100:.0f}% finite, "
              f"sd {np.nanstd(wk):.2f} {UNITS[tag]}  ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_recent(a) -> int:
    REF.mkdir(parents=True, exist_ok=True)
    for tag in a.tags.split(","):
        t0 = time.time()
        clim = np.load(WORK / f"obsclim_{tag}.npy")
        days, parts = [], []
        for y in range(RY0, RY1 + 1):
            r = daily_year(tag, y)
            if r is None:
                raise SystemExit(f"{tag}: truth year {y} missing")
            t, v = r
            parts.append(v - clim[[d.timetuple().tm_yday - 1 for d in t]])
            days += t
        an = np.concatenate(parts); del parts
        # continuity check: the 7-day windows must not jump a missing day
        gaps = np.diff([d.toordinal() for d in days])
        if (gaps != 1).any():
            raise SystemExit(f"{tag}: gaps in the recent daily truth")
        ok = np.isfinite(an)
        s7 = np.zeros((an.shape[0] - 6, NY, NX), "float32"); c7 = np.zeros_like(s7)
        for k in range(7):
            s7 += np.where(ok[k:k + s7.shape[0]], an[k:k + s7.shape[0]], 0.0)
            c7 += ok[k:k + s7.shape[0]]
        del an, ok
        with np.errstate(invalid="ignore", divide="ignore"):
            wk = np.where(c7 >= 5, s7 / np.maximum(c7, 1), np.nan).astype("float32")
        del s7, c7
        wdoy = np.array([d.timetuple().tm_yday for d in days[:wk.shape[0]]]) + 3          # window centre
        flat = wk.reshape(wk.shape[0], -1)
        allfin = np.isfinite(flat).all(0)
        anyfin = np.isfinite(flat).any(0)
        part = anyfin & ~allfin
        med = np.full((len(SK.CENTRES), NY * NX), np.nan, "float32")
        q33, q67, mean = med.copy(), med.copy(), med.copy()
        for ci, c in enumerate(SK.CENTRES):
            sel = np.abs(((wdoy - c + 183) % 366) - 183) <= 15
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
        out = {"q33": q33.reshape(sh).astype("float16"),                 # (the median is not used: no longer kept)
               "q67": q67.reshape(sh).astype("float16"), "mean": mean.reshape(sh).astype("float16"),
               "doy": SK.CENTRES.astype("int16")}
        if tag == "pr":
            # absolute 1991-2020 weekly climatology (7 days centred on each centre), for the dry-week mask
            cw = np.stack([np.nanmean(clim[[(c - 1 + k) % 366 for k in range(-3, 4)]], axis=0) for c in SK.CENTRES])
            out["clim_weekly"] = cw.astype("float16")
        meta = dict(tag=tag, units=UNITS[tag], years=f"{RY0}-{RY1}", base=f"{CY0}-{CY1}", truth=TRUTH[tag],
                    note="q33 / q67 / mean of the 7-day-mean anomaly vs the 1991-2020 day-of-year climatology over "
                         "the recent decade, windows centred within +-15 d of each 5-day centre (doy); GEFS 1.5 deg "
                         "grid, lat 90 -> -90; read linearly between centres (gefs_skill.at_doy)")
        np.savez_compressed(REF / f"gefs_terciles_{tag}.npz", meta=np.array(json.dumps(meta)), **out)
        print(f"  {tag}: terciles {RY0}-{RY1}, median of median {np.nanmedian(med):+.3f}, "
              f"mean q67-q33 {np.nanmean(q67 - q33):.2f} {UNITS[tag]} "
              f"({(REF / f'gefs_terciles_{tag}.npz').stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)
    return 0


# ── the reforecast ───────────────────────────────────────────────────────────────────────────────────────────────
def starts():
    """(date, path) of every reforecast start on disk, 2000-2019, chronological (duplicates: the first root wins)."""
    seen = {}
    for root in RF:
        for y in range(HY0, HY1 + 1):
            for f in sorted((root / f"rf-{y}").glob("rf_*.npz")):
                d = dt.date(int(f.stem[3:7]), int(f.stem[7:9]), int(f.stem[9:11]))
                seen.setdefault(d, f)
    return sorted(seen.items())


def circ(doy, c):
    """gefs_reforecast.cmd_combine's pool distance: |doy - c| folded at 365."""
    d = abs(doy - c)
    return min(d, 365 - d)


def member_sums(f, tags):
    z = np.load(f)
    n = len(z["members"])
    return n, {t: R.unpack(t, z[f"w_{t}"]).astype("float32") for t in tags}


def cmd_skill(a) -> int:
    tags = a.tags.split(",")
    out_dir = REF if not a.partial else WORK / "partial"               # a partial run never lands in the reference
    out_dir.mkdir(parents=True, exist_ok=True)
    S = starts()
    years = sorted({s.year for s, _ in S})
    print(f"{len(S)} reforecast starts, {len(years)} years {years[0]}-{years[-1]}", flush=True)
    if len(years) < HY1 - HY0 + 1 and not a.partial:
        raise SystemExit(f"only {len(years)} reforecast years on disk ({years}); --partial to run anyway")
    t0 = time.time()
    # pass 1: the full-pool climatology sums per 5-day centre
    Ssum = {t: np.zeros((len(SK.CENTRES), 5, NY, NX)) for t in tags}
    Nn = np.zeros(len(SK.CENTRES))
    pools = []
    for s, f in S:
        n, F = member_sums(f, tags)
        doy = s.timetuple().tm_yday
        cs = [ci for ci, c in enumerate(SK.CENTRES) if circ(doy, c) <= 15]
        pools.append(cs)
        for t in tags:
            fs = F[t].sum(0, dtype="float64")
            for ci in cs:
                Ssum[t][ci] += fs
        Nn[cs] += n
    print(f"  pass 1: climatology sums ({time.time() - t0:.0f} s)", flush=True)
    if not a.partial:                                                  # check against the published climatology
        cache = WORK / "clim_v1"; cache.mkdir(parents=True, exist_ok=True)
        for c in (1, 91, 181, 266, 361):
            p = cache / f"gefs_clim_{c:03d}.npz"
            if not p.exists():
                p.write_bytes(R.get(CLIM_URL.format(c=c)))
            z = np.load(p); ci = (c - 1) // 5
            msg = [f"n {int(z['n'])}/{int(Nn[ci])}"]
            for t in tags:
                msg.append(f"{t} max|d| {np.abs(Ssum[t][ci] / Nn[ci] - z[f'w_{t}']).max():.4f}")
            print(f"  check vs gefs-clim-v1 doy {c}: " + ", ".join(msg), flush=True)
    shift = np.load(SHIFT)
    shifts = {t: shift[f"s15_{t}"].astype("float64") for t in tags}
    wk_obs = {t: np.load(WORK / f"obsweek_{t}.npz")["wk"] for t in tags}
    mid = np.array([dt.date(2001, m, 15).timetuple().tm_yday for m in range(1, 13)])
    shp = (12, 5, NY, NX)
    keys = ("n", "sf", "so", "sff", "soo", "sfo", "sw", "sinv", "f2", "ff2", "fo2")
    acc = {t: {k: np.zeros(shp) for k in keys} for t in tags}
    cache = {}
    dates = [s for s, _ in S]
    # the paired weekly anomalies of the tercile fields, for the cross-validated test of the calibration (assess)
    pdir = (WORK if not a.partial else WORK / "partial")
    PAIR = {t: {w: np.lib.format.open_memmap(pdir / f"pairs_{w}_{t}.npy", mode="w+", dtype="float16",
                                             shape=(len(S), 5, NY, NX)) for w in ("f", "o")}
            for t in tags if t in ("t2m", "pr")}
    np.save(pdir / "pairs_starts.npy", np.array([d.isoformat() for d in dates]))
    for si, (s, f) in enumerate(S):
        # leave-season-out climatology at the two bracketing centres
        for j in [j for j in list(cache) if abs((dates[j] - s).days) > LOSO]:
            cache.pop(j)
        near = [j for j in range(max(0, si - 8), min(len(S), si + 9)) if abs((dates[j] - s).days) <= LOSO]
        for j in near:
            if j not in cache:
                n_j, F_j = member_sums(S[j][1], tags)
                cache[j] = (n_j, {t: F_j[t].sum(0, dtype="float64") for t in tags}, F_j)
        doy = s.timetuple().tm_yday
        i_lo, w_lo, i_hi, w_hi = SK.bracket(doy)
        n_m, F = cache[si][0], cache[si][2]
        months = np.where(np.abs(((doy - mid + 183) % 366) - 183) <= WIN)[0]
        vb = (s - B0).days // 7
        for t in tags:
            C = 0.0
            for ci, wgt in ((i_lo, w_lo), (i_hi, w_hi)):
                if wgt == 0:
                    continue
                num, den = Ssum[t][ci].copy(), Nn[ci]
                for j in near:
                    if ci in pools[j]:
                        num -= cache[j][1][t]; den -= cache[j][0]
                C = C + wgt * num / den
            an = F[t] - C[None] - SK.weekly_shift(shifts[t], s)[None]       # (m, 5, lat, lon)
            fm = an.mean(0)
            w = an.var(0, ddof=1) if n_m > 1 else np.zeros_like(fm)
            f2 = an[1:3].mean(0) if n_m >= 3 else an[:2].mean(0)
            for k in range(5):
                if vb + k >= wk_obs[t].shape[0]:
                    continue
                ow = wk_obs[t][vb + k]
                if t in PAIR:
                    PAIR[t]["f"][si, k] = fm[k]; PAIR[t]["o"][si, k] = ow
                good = np.isfinite(fm[k]) & np.isfinite(ow)
                fw, o, ww, f2w = (np.where(good, x, 0.0) for x in (fm[k], ow, w[k], f2[k]))
                for m in months:
                    A = acc[t]
                    A["n"][m, k] += good; A["sf"][m, k] += fw; A["so"][m, k] += o
                    A["sff"][m, k] += fw * fw; A["soo"][m, k] += o * o; A["sfo"][m, k] += fw * o
                    A["sw"][m, k] += ww; A["sinv"][m, k] += good / n_m
                    A["f2"][m, k] += f2w; A["ff2"][m, k] += f2w * f2w; A["fo2"][m, k] += f2w * o
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
            vf, vo = A["sff"] / n - mf ** 2, A["soo"] / n - mo ** 2
            cov = A["sfo"] / n - mf * mo
            r = cov / np.sqrt(vf * vo)
            slope = cov / vf
            icpt = mo - slope * mf
            resid = np.sqrt(np.clip(vo - slope * cov, 0, None) * n / np.clip(n - 2, 1, None))
            osd = np.sqrt(vo)
            noise = A["sw"] / n                                       # member noise variance
            inv = A["sinv"] / n                                       # mean 1/(members) of the pairs
            sig = vf - noise * inv                                    # signal variance of the 4-member mean
            mf2 = A["f2"] / n
            vf2 = A["ff2"] / n - mf2 ** 2
            cov2 = A["fo2"] / n - mf2 * mo
            r2 = cov2 / np.sqrt(vf2 * vo)
            v4p = np.clip(vf2 - noise * (0.5 - inv), noise * inv, None)
            r4_from2 = cov2 / np.sqrt(v4p * vo)                        # the 2 -> 4 member test
            v31 = np.clip(sig + noise / 31.0, noise / 31.0, None)
            r31 = cov / np.sqrt(v31 * vo)
        bad = n < MIN_PAIRS
        for x in (r, slope, icpt, resid, osd, r2, r4_from2, r31, sig, noise):
            x[bad] = np.nan
        meta = dict(tag=t, units=UNITS[t], truth=TRUTH[t], hindcast=f"GEFSv12 reforecast {HY0}-{HY1}, "
                    f"Wednesday starts, 4 members (c00 p01-p03), {len(S)} starts",
                    pool=f"init month: starts within +-{WIN} d of the 15th", min_pairs=MIN_PAIRS,
                    model_clim=f"leave-season-out (starts within {LOSO} d of the verified start removed)",
                    rebase="minus the 7-day valid-date mean of era5_shift.npz (ERA5 1991-2020 - 2001-2020)",
                    grid="GEFS 1.5 deg, lat 90 -> -90, lon 0 -> 358.5", skill_min=SK.SKILL_MIN,
                    weeks="week k = init + 7k-7 ... init + 7k-1")
        pub = {"r": r.astype("float16"), "n": np.clip(n, 0, 32767).astype("int16")}
        if t in ("t2m", "pr"):                                        # the calibration: r and the forecast sd
            fsd = np.sqrt(vf); fsd[bad] = np.nan
            pub.update(fsd=fsd.astype("float16"))
        np.savez_compressed(out_dir / f"gefs_mapskill_{t}.npz", meta=np.array(json.dumps(meta)), **pub)
        np.savez_compressed(WORK / f"diag_mapskill_{t}.npz", r=r, a=slope, b=icpt, s=resid, obs_sd=osd, n=n, mf=mf, mo=mo,
                            fsd=np.sqrt(vf),
                            r2=r2, r4_from2=r4_from2, r31=r31, sig=sig, noise=noise, inv=inv)
        rows = (R.LAT >= 30) & (R.LAT <= 70)
        share = lambda x: np.mean(x[np.isfinite(x)] >= SK.SKILL_MIN) * 100
        print(f"  {t}: gefs_mapskill_{t}.npz ({(out_dir / f'gefs_mapskill_{t}.npz').stat().st_size / 1e6:.1f} MB); "
              f"median r (Sep, 30-70N) by week: "
              + " ".join(f"{np.nanmedian(r[8, k][rows]):.2f}" for k in range(5))
              + f"; share >= {SK.SKILL_MIN}: " + " ".join(f"{share(r[8, k][rows]):.0f}%" for k in range(5)), flush=True)
    return 0


def _probs(x, rr):
    """Tercile probabilities (below, above) of N(rr x, 1 - rr^2) against +-Z3 (gefs_probs' calibration)."""
    from scipy.special import ndtr
    den = np.sqrt(1.0 - rr ** 2)
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
    pdir = WORK if not a.partial else WORK / "partial"
    dates = [dt.date.fromisoformat(str(x)) for x in np.load(pdir / "pairs_starts.npy")]
    ords = np.array([d.toordinal() for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates])
    mon = np.array([d.month for d in dates])
    yrs = np.array([d.year for d in dates])
    mid = np.array([dt.date(2001, m, 15).timetuple().tm_yday for m in range(1, 13)])
    wt = np.repeat(np.cos(np.deg2rad(R.LAT)), NX)
    sk_all = {}
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
            pool = np.where(np.abs(((doy - mid[m] + 183) % 366) - 183) <= WIN)[0]
            for k in range(5):
                F = PF[pool, k].reshape(len(pool), -1).astype("float64")
                O = PO[pool, k].reshape(len(pool), -1).astype("float64")
                okc = np.isfinite(O).all(0) & np.isfinite(F).all(0)          # cells with every pair (land for pr)
                F, O = F[:, okc], O[:, okc]
                for y in sorted(set(yrs[mon == m + 1])):
                    tgt = np.where((mon == m + 1) & (yrs == y))[0]
                    lo, hi = ords[tgt].min() - 60, ords[tgt].max() + 60
                    fit = (ords[pool] < lo) | (ords[pool] > hi)
                    sc = np.searchsorted(pool, tgt)                          # scored starts inside the pool
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
                    I1 = (os_ < q33).astype(float); I3 = (os_ > q67).astype(float)
                    # this port's standardised form
                    rr = np.clip(np.nan_to_num(r), 0.0, 0.99)
                    pb, pa = _probs((fs - mf) / np.sqrt(vf), rr)
                    # GEPS prob_maps form
                    from scipy.special import ndtr
                    mu = med + slope * (fs - med)
                    gb, ga = ndtr((q33 - mu) / s), 1.0 - ndtr((q67 - mu) / s)
                    rps = lambda b, a_: (b - I1) ** 2 + ((1 - a_) - (1 - I3)) ** 2
                    cols = np.where(okc)[0]
                    R_new, R_geps = rps(pb, pa), rps(gb, ga)
                    R_clim = rps(np.full_like(pb, 1 / 3), np.full_like(pa, 1 / 3))
                    acc["rps_new"][m, k, cols] += R_new.sum(0)
                    acc["rps_geps"][m, k, cols] += R_geps.sum(0)
                    acc["rps_clim"][m, k, cols] += R_clim.sum(0)
                    acc["n"][m, k, cols] += len(sc)
                    w = np.broadcast_to(wt[cols], pb.shape)
                    skl = np.nan_to_num(r) >= SK.SKILL_MIN
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
        np.savez_compressed(WORK / f"assess_{t}.npz", rpss_new=rpss_new.astype("float32"),
                            rpss_geps=rpss_geps.astype("float32"), n=acc["n"].astype("int16"),
                            **{f"split_{k}": v for k, v in spl.items()}, **rel)
        print(f"  {t}: cross-validated RPSS vs climatology (area-weighted, pooled over months, cells with truth; "
              f"unhatched/hatched = out-of-sample r >= / < {SK.SKILL_MIN}):")
        test = {"test": "RPSS of the calibrated tercile probabilities vs one third each, every 2000-2019 reforecast start "
                        "scored with its init month's table refitted without its season (starts within 60 d), "
                        "categories = the fit years' observed terciles; area-weighted, cells with truth; 4-member "
                        "hindcast means (the 31-member live mean is at least as skilful)", "weeks": []}
        for k in range(5):
            ss = lambda name, j: float(1 - spl[name][k, j] / spl["rps_clim"][k, j])
            hi = rel["new_w"][k][:, 4:].sum()
            test["weeks"].append({"week": k + 1,
                                  "rpss": round(float(1 - spl["rps_new"][k].sum() / spl["rps_clim"][k].sum()), 3),
                                  "rpss_unhatched": round(ss("rps_new", 1), 3), "rpss_hatched": round(ss("rps_new", 0), 3),
                                  "share_unhatched": round(float(spl["n"][k, 1] / spl["n"][k].sum()), 3),
                                  "rpss_geps_form": round(float(1 - spl["rps_geps"][k].sum() / spl["rps_clim"][k].sum()), 3),
                                  "reliability_p_ge_0.4": {"forecast": round(float(rel["new_p"][k][:, 4:].sum() / hi), 3),
                                                           "observed": round(float(rel["new_o"][k][:, 4:].sum() / hi), 3)}})
        if not a.partial:                                   # the result travels with the published skill file
            q = REF / f"gefs_mapskill_{t}.npz"
            z = dict(np.load(q))
            meta = json.loads(str(z.pop("meta")))
            meta["calibration_test"] = test
            np.savez_compressed(q, meta=np.array(json.dumps(meta)), **z)
        for k in range(5):
            line = []
            for name in ("rps_new", "rps_geps"):
                ss = lambda j: 1 - spl[name][k, j] / spl["rps_clim"][k, j]
                tot = 1 - spl[name][k].sum() / spl["rps_clim"][k].sum()
                line.append(f"{name[4:]} all {tot:+.3f} unhatched {ss(1):+.3f} hatched {ss(0):+.3f}")
            frac = spl["n"][k, 1] / spl["n"][k].sum()
            print(f"    week {k + 1} ({frac * 100:.0f}% unhatched): " + " | ".join(line)
                  + f" | reliability (new, P>=0.4 bins) fcst {np.nansum(rel['new_p'][k][:, 4:]) / np.nansum(rel['new_w'][k][:, 4:]):.3f}"
                  + f" obs {np.nansum(rel['new_o'][k][:, 4:]) / np.nansum(rel['new_w'][k][:, 4:]):.3f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "recent", "skill", "assess"):
        p = sp.add_parser(name)
        p.add_argument("--tags", default="t2m,pr,zg500,mslp" if name in ("truth", "skill") else "t2m,pr")
        p.add_argument("--partial", action="store_true", help="skill: run on the reforecast years present")
    a = ap.parse_args()
    return {"truth": cmd_truth, "recent": cmd_recent, "skill": cmd_skill, "assess": cmd_assess}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
