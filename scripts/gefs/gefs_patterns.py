#!/usr/bin/env python3
"""The shared part of the GEFS pattern products: teleconnection indices (gefs_telecon.py), weather regimes
(gefs_regimes.py) and 500 hPa ensemble scenarios (gefs_clusters.py) - ported from the GEPS page (2026-09-26; user:
"the GEFS pages should have all the same maps/charts the GEPS one does").

Everything here is numpy + the standard library (the daily Actions job has no pandas or xarray).

Member anomalies (the forecast side)
  Every GEFS member's DAILY 2.5 deg field (m_<tag> of gefs2_<date>.npz: 500 hPa height, sea-level pressure, 100 hPa
  height; day d = the mean of the 24d-12 h and 24d h samples, the reforecast's own sampling) minus the lead-matched
  GEFSv12 reforecast climatology c25_<tag> read at the INIT day of year (release gefs-clim-v2, 2000-2019, 4 members,
  +-15 d pooling), then re-based to 1991-2020 with the ERA5 period shift (1991-2020 minus 2001-2020 normal) at the
  VALID date - the GEPS page's anomaly, [F - M(init doy, lead)] - shift(valid doy). 2001-2020 stands in for the
  reforecast's 2000-2019. As on the GEPS page, the per-member anomalies keep their global mean (only the maps remove
  it): the projections and the regime assignment are made against 1991-2020 fields, like the observed tail and CPC.
  The 100 hPa height is de-drifted the same way - the GEPS8 hindcast had no 100 hPa and GEPS's NAM100 was a plain
  anomaly against NCEP; GEFS's reforecast has it, so here NAM100 is treated exactly like the other indices, with a
  100 hPa ERA5 shift built by the same method (build_gefs_patterns_ref.py).

Valid dates
  Day d (1..35) is valid on init + (d - 1): the 12 h and 24 h samples, mostly that UTC day. Weeks are days 1-7 ...
  29-35. The same valid date is used for the ERA5 shift, the CPC month of each projector and the pairing with
  observations in the hindcast.

Observed tail (GEFS's own analyses)
  Day D of the tail = the mean of the GEFS control analyses (gec00 f000, pgrb2a 0.5 deg, byte-ranged through the .idx
  on NOAA's S3) at 12Z D and 00Z D+1 - the SAME two synoptic hours as a forecast day and as every reforecast day. The
  12-hourly pair cancels the diurnal tide S1 but not the semidiurnal S2; the reforecast climatology carries exactly
  the same S2 (it is sampled at 00Z and 12Z too), so S2 cancels in the anomaly. A tide-free four-sample daily mean
  (the GEPS page's f000/f006/f012/f018) would NOT cancel against this climatology: measured on 30 days of GEFS
  analyses (Aug 20 - Sep 18 2026) the four-sample mean sits +1.03 seasonal sd above the 00Z/12Z pair on the SOI and
  +1.26 on the EQSOI (day-to-day sd 0.2), against -0.03 / +0.01 on the AO / AAO - a spurious step at every seam.
  The anomaly is taken against the reforecast climatology at LEAD DAY 1 of a start on day D (whose day 1 is exactly
  those two samples; calib c1_<tag>, all 73 centres) - an analysis against a 12/24 h forecast climate, the one lead
  mismatch, and the smallest available (the reforecast has no lead 0). The 2.5 deg fields are cached one small file
  per day (gefs_telecon_ana_<date>.npz, int16) in the run archive, so each run fetches only the new day.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R                                           # noqa: E402  (get, index, decode)
import gefs_daily as G                                               # noqa: E402  (to_half, box, pack/unpack, grids)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
CLIM2_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v2/gefs_clim2_{c:03d}.npz"
REF_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-ref-v1/{name}"
REF_NAME = "gefs_patterns_ref.npz"          # projectors, sigmas, regime centroids, ERA5 shifts (v2-independent)
CALIB_NAME = "gefs_patterns_calib.npz"      # hindcast fits, regime hindcast labels, day-1 climatology (from v2)
CENTRES = list(range(1, 366, 5))
TAGS = ("zg500", "mslp", "zg100")
FIELD = {"zg500": "z500", "mslp": "slp", "zg100": "z100"}             # the pattern file's field names
UNITS = {"zg500": 1.0, "mslp": 0.01, "zg100": 1.0}                    # -> m, hPa, m (the patterns' units)
LAT25, LON25 = G.LAT25, G.LON25
WEEKS = G.WEEKS
DAYS = 35
TAIL_DAYS = 45
ANA_KEEP = 60                               # analysis-tail files kept in the run archive (days)
INK, MUTED = "#2f2c28", "#5a5650"           # the GEPS page's annotation greys (both pass AA on white)
ORDER = ["ao", "nao", "pna", "epo", "wpo", "soi", "eqsoi", "ea", "wp", "epnp", "eawr", "sca", "tnh", "pol",
         "nam100", "aao"]
# weather regimes (build_regimes.py on the GEPS side: k-means, k = 4, NCEP/NCAR R1 1991-2020, 5-day means)
SECTORS = {"ea": dict(label="Euro-Atlantic", lat=(80, 20), lon=(-90, 30)),
           "na": dict(label="North America / Pacific", lat=(80, 20), lon=(-170, -30))}
SEASONS = {"cold": (10, 11, 12, 1, 2, 3), "warm": (4, 5, 6, 7, 8, 9)}
K, R_MIN, SMOOTH = 4, 0.25, 5
TEST_NOTE = ""                              # set by --test-clim: stamped on every figure


# ── dates ─────────────────────────────────────────────────────────────────────────────────────────────────────
def parse_date(s: str) -> dt.date:
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def valid_days(base: dt.date, n: int = DAYS):
    """Day d (1..n) is valid on init + (d - 1)."""
    return [base + dt.timedelta(days=d) for d in range(n)]


def doy(d: dt.date) -> int:
    return d.timetuple().tm_yday


def season_of(month: int) -> str:
    return next(s for s, months in SEASONS.items() if month in months)


def bracket(day_of_year: int):
    """The two 5-day day-of-year centres around a day of year and the weight of the upper one (gefs_live's rule)."""
    lo = max(c for c in CENTRES if c <= day_of_year)
    hi = lo + 5 if lo + 5 <= 361 else 1
    return lo, hi, (day_of_year - lo) / 5.0


def centre_index(c: int) -> int:
    return (c - 1) // 5


# ── downloads ─────────────────────────────────────────────────────────────────────────────────────────────────
def get(url, rng=None):
    return R.get(url, rng)


def cached(url: str, p: Path) -> Path:
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part"); tmp.write_bytes(get(url)); tmp.replace(p)
    return p


def ref_path(name: str, cache: Path, local: Path | None = None) -> Path:
    """A reference file: the local directory if it has it (--ref), else release gefs-ref-v1 into the cache."""
    if local is not None and (Path(local) / name).exists():
        return Path(local) / name
    return cached(REF_URL.format(name=name), Path(cache) / "ref" / name)


def refresh(url: str, dest: Path, max_age_h: float = 12.0) -> Path:
    """A file its source updates daily (CPC, PSL, Long Paddock): re-downloaded unless fresh; failures keep the old."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and time.time() - dest.stat().st_mtime < max_age_h * 3600:
        return dest
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "scorvec/1.0"}), timeout=60) as r:
            dest.write_bytes(r.read())
    except Exception as e:                                           # noqa: BLE001
        print(f"  (refresh failed {dest.name}: {str(e)[:70]})", flush=True)
    return dest


# ── climatology ───────────────────────────────────────────────────────────────────────────────────────────────
def clim25_at(init: dt.date, cache: Path, test: Path | None = None):
    """c25_<tag> (35, 73, 144) of the v2 daily reforecast climatology at the init day of year, linear between the two
    5-day centres around it, in GEFS units. None when the release is not there yet. `test` = a stand-in file."""
    if test is not None:
        z = np.load(test)
        return {t: z[f"c25_{t}"].astype("float64") for t in TAGS if f"c25_{t}" in z.files}
    lo, hi, w = bracket(doy(init))
    out = {}
    try:
        for c, wt in ((lo, 1 - w), (hi, w)):
            z = np.load(cached(CLIM2_URL.format(c=c), Path(cache) / f"gefs_clim2_{c:03d}.npz"))
            for t in TAGS:
                if f"c25_{t}" in z.files:
                    out[t] = out.get(t, 0) + wt * G.unpack(t, z[f"c25_{t}"])
    except Exception as e:                                           # noqa: BLE001
        print(f"  daily climatology v2 unavailable ({str(e)[:80]})", flush=True)
        return None
    return out or None


# ── reference (projectors, sigmas, centroids, shifts) ─────────────────────────────────────────────────────────
class Ref:
    def __init__(self, path: Path, calib: Path | None = None):
        z = np.load(path)
        self.meta = json.loads(str(z["meta_json"]))
        self.pat = self.meta["patterns"]                             # validation.json of the GEPS side, per index
        self.ids = [i for i in ORDER if i in self.pat] + [i for i in self.pat if i not in ORDER]
        self.defined = {k: set(v) for k, v in self.meta["defined_months"].items()}
        self.P = {sid: z[f"proj_{sid}"].astype("float64") for sid in self.ids}          # (12, 73, 144)
        self.std = z["std_month_z500"].astype("float64")                                  # (12, 73, 144)
        self.sigma_doy = {sid: z[f"sigma_doy_{sid}"].astype("float64") for sid in self.ids}   # (366,)
        self.w = np.sqrt(np.clip(np.cos(np.deg2rad(LAT25)), 0, None))[:, None] * np.ones((1, LON25.size))
        self.shift_doy = z["shift_doy"].astype("float64")
        self.s25 = {}
        for t in TAGS:                                               # height shifts are stored north of the equator
            if f"s25_{t}" in z.files:
                s = z[f"s25_{t}"].astype("float64")
                self.s25[t] = np.concatenate([s, np.zeros((s.shape[0], LAT25.size - s.shape[1], LON25.size))], 1)
        self.cent = {k[4:]: z[k].astype("float64") for k in z.files if k.startswith("reg_")}   # (4, lat, lon)
        self.names = self.meta["regime_names"]
        self.regmeta = self.meta["regime_meta"]
        self.calib = np.load(calib) if calib is not None and Path(calib).exists() else None
        self._c1 = {}
        self.calib_meta = json.loads(str(self.calib["meta_json"])) if self.calib is not None else {}

    # teleconnection projection (telecon_geps.Patterns.index, verbatim in arithmetic)
    def index(self, sid, anom, months, doys):
        """anom (n, 73, 144) in the pattern's units (z m / slp hPa); months, doys of each sample's VALID date.
        -> daily index in SEASONAL sd (std of the 1991-2020 daily projection within +-15 d of that day of year);
        NaN in calendar months where CPC does not define the pattern."""
        m = self.pat[sid]
        mo = np.asarray(months) - 1
        sig = self.sigma_doy[sid][np.asarray(doys) - 1]
        P = self.P[sid][mo]
        a = np.asarray(anom, dtype="float64")
        if m["standardize"]:
            sd = self.std[mo]
            a = a / np.where(sd > 0, sd, np.nan)
        if m.get("demean"):
            cosw = (np.cos(np.deg2rad(LAT25)) * (LAT25 >= 20))[:, None] * np.ones((1, LON25.size))
            a = a - (np.nan_to_num(a) * cosw).sum(axis=(-2, -1), keepdims=True) / cosw.sum()
        z = np.nan_to_num(a * self.w)
        with np.errstate(invalid="ignore", divide="ignore"):         # sigma is NaN where CPC defines no pattern
            out = np.einsum("nij,nij->n", z, P) / sig
        out[~np.isin(np.asarray(months), sorted(self.defined[sid]))] = np.nan
        return out

    def to_seasonal(self, sid, doys):
        """Annually standardized value (CPC/PSL convention) -> seasonal sd."""
        return self.pat[sid]["sigma_daily"] / self.sigma_doy[sid][np.asarray(doys) - 1]

    def shift(self, tag, days):
        """ERA5 1991-2020 minus 2001-2020 normal (GEFS units) at each date, linear in day of year with the year
        wrapped (gefs_live.shift_series); None if the reference has no shift for the tag."""
        if tag not in self.s25:
            return None
        x = np.r_[self.shift_doy - 365.0, self.shift_doy, self.shift_doy + 365.0]
        f = np.concatenate([self.s25[tag]] * 3)
        out = []
        for d in days:
            t = float(doy(d))
            j = int(np.searchsorted(x, t)); w = (t - x[j - 1]) / (x[j] - x[j - 1])
            out.append((1 - w) * f[j - 1] + w * f[j])
        return np.stack(out)

    def clim_day1(self, tag, day: dt.date):
        """The reforecast climatology at lead day 1 for a start on `day` (calib c1_<tag>, 73 centres), GEFS units."""
        if self.calib is None or f"c1_{tag}" not in self.calib.files:
            return None
        lo, hi, w = bracket(doy(day))
        c1 = self._c1.setdefault(tag, self.calib[f"c1_{tag}"])
        v = (1 - w) * G.unpack(tag, c1[centre_index(lo)]) + w * G.unpack(tag, c1[centre_index(hi)])
        if v.shape[0] < LAT25.size:                                  # heights are stored north of the equator only
            v = np.concatenate([v, np.full((LAT25.size - v.shape[0], LON25.size), np.nan)], 0)
        return v

    # weather regimes
    def centroids(self, sk, season):
        return self.cent[f"{sk}_{season}"], self.cent[f"lat_{sk}"], self.cent[f"lon_{sk}"]

    def regime_name(self, sk, season, k):
        return self.names.get(f"{sk}_{season}_{k}", f"R{k + 1}")

    def classify(self, field, sk, season):
        """field (n, lat, lon) smoothed sector anomalies -> labels (K = no regime) and pattern correlations."""
        cent, lat, lon = self.centroids(sk, season)
        w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        X = (field * w).reshape(field.shape[0], -1)
        Cw = (cent * w).reshape(K, -1)
        lab = ((X[:, None, :] - Cw[None]) ** 2).sum(-1).argmin(1)
        xm = X - X.mean(1, keepdims=True); cm = Cw - Cw.mean(1, keepdims=True)
        corr = (xm @ cm.T) / (np.linalg.norm(xm, axis=1)[:, None] * np.linalg.norm(cm, axis=1)[None] + 1e-9)
        corr = corr[np.arange(len(lab)), lab]
        return np.where(corr < R_MIN, K, lab), corr


def sector_cut(a, spec):
    """(..., 73, 144) global 2.5 deg (lat 90 -> -90, lon 0E first) -> the sector, lat descending, lon in -180..180
    ascending (regimes_geps.sector_cut on xarray). -> (cut, lat, lon)."""
    lon = ((LON25 + 180) % 360) - 180
    o = np.argsort(lon, kind="stable")
    lon_s = lon[o]
    la = (LAT25 <= spec["lat"][0] + 1e-6) & (LAT25 >= spec["lat"][1] - 1e-6)
    lo = (lon_s >= spec["lon"][0] - 1e-6) & (lon_s <= spec["lon"][1] + 1e-6)
    b = np.asarray(a)[..., la, :][..., o][..., lo]
    return b, LAT25[la], lon_s[lo]


def running_partial(a, n, axis=0):
    """n-point running mean along axis with partial windows at the ends (regimes_geps.running_partial)."""
    a = np.moveaxis(np.asarray(a, dtype="float64"), axis, 0)
    cs = np.cumsum(np.nan_to_num(a), axis=0)
    cnt = np.cumsum(np.isfinite(a), axis=0)
    out = np.empty_like(a)
    h = n // 2
    for i in range(a.shape[0]):
        i0, i1 = max(0, i - h), min(a.shape[0], i + h + 1)
        top = cs[i1 - 1] - (cs[i0 - 1] if i0 > 0 else 0)
        c = cnt[i1 - 1] - (cnt[i0 - 1] if i0 > 0 else 0)
        out[i] = top / np.maximum(c, 1)
    return np.moveaxis(out, 0, axis)


# ── forecast side ─────────────────────────────────────────────────────────────────────────────────────────────
def member_anoms(z, base: dt.date, ref: Ref, clim):
    """{tag: (35, M, 73, 144)} daily member anomalies against 1991-2020 in the patterns' units (m, hPa)."""
    days = valid_days(base)
    out = {}
    import gefs_drift
    dr = gefs_drift.from_env()                                        # operational drift the reforecast lacks
    for t in TAGS:
        if clim is None or t not in clim or f"m_{t}" not in z.files:
            continue
        a = z[f"m_{t}"].astype("float64") - clim[t][None]              # (M, 35, 73, 144)
        s = ref.shift(t, days)
        if s is not None:
            a -= s[None]
        if dr is not None and t in dr.tags:
            a -= dr.on(t, LAT25)[None, :, :, None]
        a *= UNITS[t]
        if not np.isfinite(a).all():
            raise SystemExit(f"{t}: non-finite member anomaly")
        out[t] = np.ascontiguousarray(np.moveaxis(a, 0, 1)).astype("float32")
    return out


def indices_from(ref: Ref, fields, days, quiet=False):
    """{sid: (35, M) index} for every index whose field is available; indices undefined over the whole forecast
    (CPC does not publish them in these months) are left out."""
    mo = np.array([d.month for d in days]); dy = np.array([doy(d) for d in days])
    out = {}
    for sid in ref.ids:
        tag = next(t for t in TAGS if FIELD[t] == ref.pat[sid]["field"])
        if tag not in fields:
            continue
        a = fields[tag]                                              # (L, M, 73, 144)
        L, M = a.shape[:2]
        v = ref.index(sid, a.reshape(L * M, *a.shape[2:]), np.repeat(mo, M), np.repeat(dy, M)).reshape(L, M)
        if not np.isfinite(v).any():
            if not quiet:
                print(f"  {sid}: not a CPC-defined pattern in {days[0]:%b}-{days[-1]:%b} - left out", flush=True)
            continue
        out[sid] = v
    return out


# ── observed tail: GEFS control analyses ──────────────────────────────────────────────────────────────────────
ANA = {"zg500": ("HGT", "500 mb"), "mslp": ("PRMSL", "mean sea level"), "zg100": ("HGT", "100 mb")}


def ana_url(t: dt.datetime) -> str:
    return f"{S3}/gefs.{t:%Y%m%d}/{t:%H}/atmos/pgrb2ap5/gec00.t{t:%H}z.pgrb2a.0p50.f000"


def analysis25(t: dt.datetime):
    """{tag: (73, 144)} GEFS control analysis at one synoptic time, 2.5 deg by the members' own box mean; None if
    the cycle is not on S3 (or lacks a field)."""
    url = ana_url(t)
    try:
        idx = R.index(url)
    except Exception:                                                # noqa: BLE001
        return None
    jobs = []
    for tag, (name, lev) in ANA.items():
        hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == "anl"]
        if not hit:
            return None
        jobs.append((tag, hit[0][0], hit[0][1]))
    with ThreadPoolExecutor(3) as ex:
        bufs = list(ex.map(lambda j: get(url, (j[1], j[2])), jobs))
    return {tag: G.box(G.to_half(R.decode(b).astype("float64")), 5) for (tag, *_), b in zip(jobs, bufs)}


def tail_day(day: dt.date, runs: Path):
    """{tag: (73, 144)} of tail day `day`: the mean of the 12Z `day` and 00Z `day`+1 analyses (the forecast day's own
    two samples), cached as runs/gefs_telecon_ana_<day>.npz (int16, gefs_daily packing). None if unavailable."""
    f = Path(runs) / f"gefs_telecon_ana_{day:%Y%m%d}.npz"
    if f.exists():
        z = np.load(f)
        return {t: G.unpack(t, z[t]) for t in TAGS}
    t12 = dt.datetime(day.year, day.month, day.day, 12)
    a, b = analysis25(t12), analysis25(t12 + dt.timedelta(hours=12))
    if a is None or b is None:
        return None
    v = {t: 0.5 * (a[t] + b[t]) for t in TAGS}
    Path(runs).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **{t: G.pack(t, v[t]) for t in TAGS})
    return {t: G.unpack(t, G.pack(t, v[t])) for t in TAGS}


def prune_tail(runs: Path, base: dt.date):
    cut = (base - dt.timedelta(days=ANA_KEEP)).strftime("%Y%m%d")
    for q in Path(runs).glob("gefs_telecon_ana_*.npz"):
        if q.stem.split("_")[-1] < cut:
            q.unlink(missing_ok=True)


def observed_anoms(base: dt.date, ref: Ref, runs: Path, days: int = TAIL_DAYS):
    """({tag: (n, 73, 144)} tail anomalies against 1991-2020 in the patterns' units, [dates]) for the tail days
    base - days .. base - 1 that exist. Anomaly = analysis pair - reforecast climatology at lead day 1 of a start on
    that day - ERA5 shift at that day."""
    want = [base - dt.timedelta(days=k) for k in range(days, 0, -1)]
    with ThreadPoolExecutor(6) as ex:
        got = list(ex.map(lambda d: tail_day(d, runs), want))
    ts, vals = [], {t: [] for t in TAGS}
    for d, v in zip(want, got):
        if v is None:
            continue
        ok = True
        row = {}
        for t in TAGS:
            c = ref.clim_day1(t, d)
            if c is None:
                ok = False; break
            s = ref.shift(t, [d])
            row[t] = (v[t] - c - (s[0] if s is not None else 0.0)) * UNITS[t]
        if ok:
            ts.append(d)
            for t in TAGS:
                vals[t].append(row[t])
    if not ts:
        return {}, []
    missing = len(want) - len(ts)
    print(f"  observed tail: {len(ts)} days {ts[0]} .. {ts[-1]}" + (f" ({missing} missing)" if missing else ""), flush=True)
    return {t: np.stack(v) for t, v in vals.items()}, ts


# ── run archive (previous-run overlays) ───────────────────────────────────────────────────────────────────────
def previous_run(runs: Path, product: str, date: str, max_gap: int = 7):
    """The most recent earlier archived run of a product within max_gap days -> (date string, npz) or (None, None)."""
    base = parse_date(date)
    for k in range(1, max_gap + 1):
        d = (base - dt.timedelta(days=k)).strftime("%Y%m%d")
        p = Path(runs) / f"gefs_{product}_{d}.npz"
        if p.exists():
            return d, np.load(p, allow_pickle=False)
    return None, None


def prune_runs(runs: Path, product: str, base: dt.date, keep_days: int = 14):
    cut = (base - dt.timedelta(days=keep_days)).strftime("%Y%m%d")
    for q in Path(runs).glob(f"gefs_{product}_*.npz"):
        s = q.stem.split("_")[-1]
        if q.stem == f"gefs_{product}_{s}" and len(s) == 8 and s.isdigit() and s < cut:   # not the _ana_ tail files
            q.unlink(missing_ok=True)


def save_webp(fig, path: Path, dpi=105):
    fig.savefig(path, dpi=dpi, facecolor="white", pil_kwargs={"quality": 88, "method": 6})


def common_args(ap):
    ap.add_argument("--npz", required=True, help="gefs2_<date>.npz from gefs_live.fetch")
    ap.add_argument("--date", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--cache", required=True, help="download cache (climatology, reference, CPC files)")
    ap.add_argument("--ref", help="local directory holding the reference files (else release gefs-ref-v1)")
    ap.add_argument("--runs", help="run archive (default SITE/assets/gefs/anim/runs, the frames-branch archive)")
    ap.add_argument("--test-clim", help="TEST ONLY: a stand-in c25 climatology npz; stamps every figure")
    return ap


def setup(a):
    """Common start of the three products: paths, reference, climatology, member file."""
    global TEST_NOTE
    base = parse_date(a.date)
    site = Path(a.site); out = site / "assets" / "gefs"; data = out / "data"
    out.mkdir(parents=True, exist_ok=True); data.mkdir(exist_ok=True)
    runs = Path(a.runs) if a.runs else out / "anim" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    local = Path(a.ref) if a.ref else None
    refp = ref_path(REF_NAME, cache, local)
    try:
        calp = ref_path(CALIB_NAME, cache, local)
    except Exception as e:                                           # noqa: BLE001
        print(f"  no calibration file ({str(e)[:60]}): no observed tail, no hindcast skill", flush=True)
        calp = None
    if a.test_clim:
        TEST_NOTE = "  ·  TEST climatology"
    ref = Ref(refp, calp)
    clim = clim25_at(base, cache, Path(a.test_clim) if a.test_clim else None)
    z = np.load(a.npz)
    return base, out, data, runs, cache, ref, clim, z
