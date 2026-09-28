#!/usr/bin/env python3
"""Weather regimes: the shared core of every producer (GEPS on the laptop, GEFS / AIFS-ENS / IFS-ENS / the multi-model
view in Actions) - numpy and the standard library only, so the light Actions jobs need nothing else.

THE REGIMES (build_regimes.py in ~/data_archive/geps_subx, unchanged): k-means, k = 4, on 5-day running-mean 500 hPa
height anomalies of NCEP/NCAR R1 1991-2020, 14 EOFs, cos-lat weights, one set for the cold season (Oct-Mar) and one for
the warm season (Apr-Sep), in two sectors - Euro-Atlantic (20-80N, 90W-30E) and North America / Pacific (20-80N,
170W-30W). A day whose pattern correlation with its nearest centroid is below 0.25 is "no regime". The centroids,
names and the R1 base-period statistics ship in data/regimes_ref.npz (build_regimes_ref.py).

SEASON-AWARE CLASSIFICATION (2026-09-28). Every valid day - forecast, observed tail and hindcast alike - is classified
with the set of ITS OWN season: Oct 1-Mar 31 cold, Apr 1-Sep 30 warm, the boundaries the sets were built on. Until
then the init month's set was held for the whole forecast and days past the boundary were hatched. A soft blend
across the boundary was considered and rejected: every training day belonged to exactly one set, so a blend has no
counterpart in the regimes' own definition, and it would split one member-day between two sets' names. What carries
across the boundary instead is COLOUR: cold and warm regimes whose centroids correlate >= 0.6 are one "family" and
keep one colour and one stacking position, so a band that continues across Oct 1 is the same pattern continuing
(Euro-Atlantic: all four pair up, r 0.66-0.89; North America: Pacific trough 0.87, Greenland high 0.69, Pacific ridge
0.65; the cold Alaskan ridge and the warm Hudson Bay ridge have no counterpart - r -0.68 with each other - and are
families of their own).

LABELS. A label is the regime index WITHIN the set of that day (0..3, frequency-ordered as in regimes.nc) or 4 = no
regime; -1 = missing. `family_probs` turns labels into probabilities per family, "no regime" last.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REF_FILE = DATA / "regimes_ref.npz"

K, R_MIN, SMOOTH = 4, 0.25, 5
NONE = K
SECTORS = {"ea": dict(label="Euro-Atlantic", lat=(80, 20), lon=(-90, 30)),
           "na": dict(label="North America / Pacific", lat=(80, 20), lon=(-170, -30))}
SEASONS = {"cold": (10, 11, 12, 1, 2, 3), "warm": (4, 5, 6, 7, 8, 9)}
LAT25 = np.arange(90.0, -90.0 - 1e-6, -2.5)          # the NCEP R1 grid, 73
LON25 = np.arange(0.0, 360.0, 2.5)                   # 144
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
NONE_COLOR = "#cfcbc4"
INK, MUTED = "#2f2c28", "#5a5650"                    # both pass AA on white (contrast rule 2026-09-12)
FAMILY_MIN_R = 0.6

# Families: one colour and one stacking slot per pattern, across the two seasonal sets. `cold`/`warm` = the regime
# index in that set (None = the family does not exist in that set). Verified by build_regimes_ref.py against the
# centroid correlations; the build refuses to write a file whose pairs fall below FAMILY_MIN_R.
FAMILIES = {
    "ea": [dict(key="naop", cold=0, warm=0, color="#2c6fbb"),
           dict(key="scbl", cold=1, warm=2, color="#3f9b6a"),
           dict(key="ridge", cold=2, warm=3, color="#d08a2a"),
           dict(key="naom", cold=3, warm=1, color="#b4453c")],
    "na": [dict(key="pactr", cold=0, warm=0, color="#2c6fbb"),
           dict(key="pacr", cold=2, warm=2, color="#d08a2a"),
           dict(key="grnh", cold=3, warm=3, color="#b4453c"),
           dict(key="akr", cold=1, warm=None, color="#7b4fa0"),
           dict(key="hbr", cold=None, warm=1, color="#3f9b6a")],
}


# ── dates and seasons ─────────────────────────────────────────────────────────────────────────────────────────────
def parse_date(s) -> dt.date:
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    s = str(s).replace("-", "")
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def season_of(month: int) -> str:
    return "cold" if month in SEASONS["cold"] else "warm"


def season_of_date(d) -> str:
    return season_of(parse_date(d).month)


def boundary_in(days):
    """The first valid day whose season differs from the first day's, or None."""
    if not days:
        return None
    s0 = season_of_date(days[0])
    return next((parse_date(d) for d in days if season_of_date(d) != s0), None)


# ── reference ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Ref:
    """Centroids, names and base-period statistics (data/regimes_ref.npz)."""

    def __init__(self, path: Path = REF_FILE):
        z = np.load(path)
        self.meta = json.loads(str(z["meta_json"]))
        self.cent = {k[5:]: z[k].astype("float64") for k in z.files if k.startswith("cent_")}
        self.lat = {sk: z[f"lat_{sk}"].astype("float64") for sk in SECTORS}
        self.lon = {sk: z[f"lon_{sk}"].astype("float64") for sk in SECTORS}
        self.names = self.meta["names"]

    def centroids(self, sk, season):
        return self.cent[f"{sk}_{season}"], self.lat[sk], self.lon[sk]

    def name(self, sk, season, k):
        if k == NONE:
            return "no regime"
        return self.names.get(f"{sk}_{season}_{k}", f"R{k + 1}")

    def classify(self, field, sk, season):
        """field (n, lat, lon) smoothed sector anomalies (m) -> (labels, pattern correlation with the nearest
        centroid); label NONE when that correlation is below R_MIN. Nearest = weighted Euclidean, as the build."""
        cent, lat, lon = self.centroids(sk, season)
        w = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0, None))[:, None] * np.ones((1, lon.size))
        X = (np.asarray(field, dtype="float64") * w).reshape(len(field), -1)
        Cw = (cent * w).reshape(K, -1)
        lab = ((X[:, None, :] - Cw[None]) ** 2).sum(-1).argmin(1)
        xm = X - X.mean(1, keepdims=True)
        cm = Cw - Cw.mean(1, keepdims=True)
        corr = (xm @ cm.T) / (np.linalg.norm(xm, axis=1)[:, None] * np.linalg.norm(cm, axis=1)[None] + 1e-9)
        corr = corr[np.arange(len(lab)), lab]
        return np.where(corr < R_MIN, NONE, lab).astype("int8"), corr

    def classify_dated(self, field, dates, sk):
        """Each row classified with the set of its own date's season. field (n, lat, lon), dates (n,)."""
        seasons = np.array([season_of_date(d) for d in dates])
        lab = np.full(len(dates), -1, "int8")
        corr = np.full(len(dates), np.nan)
        for s in SEASONS:
            m = seasons == s
            if m.any():
                lab[m], corr[m] = self.classify(np.asarray(field)[m], sk, s)
        return lab, corr

    # families
    def families(self, sk):
        return FAMILIES[sk]

    def family_index(self, sk, season, k):
        """Family slot of regime k of `season`'s set (NONE -> len(families))."""
        fam = FAMILIES[sk]
        if k == NONE:
            return len(fam)
        return next(i for i, f in enumerate(fam) if f[season] == k)

    def family_name(self, sk, fi, seasons=("cold", "warm")):
        """Display name of a family over the seasons present, e.g. 'Greenland high → NAO− / Greenland blocking'."""
        fam = FAMILIES[sk]
        if fi == len(fam):
            return "no regime"
        names = []
        for s in seasons:
            k = fam[fi][s]
            if k is not None:
                n = self.name(sk, s, k)
                if n not in names:
                    names.append(n)
        return " → ".join(names) if names else "—"

    def family_colors(self, sk):
        return [f["color"] for f in FAMILIES[sk]] + [NONE_COLOR]


def family_of_labels(ref: Ref, sk, labels, dates):
    """labels (days, ...) within each day's set -> family slots (days, ...); -1 stays -1."""
    fam = FAMILIES[sk]
    out = np.full(np.shape(labels), -1, "int16")
    lab = np.asarray(labels)
    for i, d in enumerate(dates):
        s = season_of_date(d)
        lut = np.full(K + 1, -1, "int16")
        for k in range(K):
            lut[k] = ref.family_index(sk, s, k)
        lut[NONE] = len(fam)
        row = lab[i]
        ok = row >= 0
        o = np.full(np.shape(row), -1, "int16")
        o[ok] = lut[row[ok]]
        out[i] = o
    return out


def family_probs(ref: Ref, sk, labels, dates):
    """labels (days, members) -> probabilities (F + 1, days), 'no regime' last; a day with no members is NaN."""
    fl = family_of_labels(ref, sk, labels, dates)
    F = len(FAMILIES[sk])
    n = (fl >= 0).sum(1)
    P = np.stack([(fl == f).sum(1) for f in range(F + 1)]).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        P = P / n[None]
    P[:, n == 0] = np.nan
    return P


def season_probs_to_family(ref: Ref, sk, probs_by_season, dates):
    """{season: (K+1, days) probabilities with that set held} -> (F+1, days) family probabilities, each day taking
    the set of its own season. For run archives written before the season-aware switch (both sets were kept)."""
    F = len(FAMILIES[sk])
    out = np.full((F + 1, len(dates)), np.nan)
    for i, d in enumerate(dates):
        s = season_of_date(d)
        p = probs_by_season.get(s)
        if p is None or i >= p.shape[1] or not np.isfinite(p[:, i]).all():
            continue
        out[:, i] = 0.0
        for k in range(K + 1):
            out[ref.family_index(sk, s, k), i] += p[k, i]
    return out


# ── grids and smoothing ───────────────────────────────────────────────────────────────────────────────────────────
def sector_cut(a, spec):
    """(..., 73, 144) global 2.5 deg (lat 90 -> -90, lon 0E first) -> the sector, lat descending, lon in -180..180
    ascending. -> (cut, lat, lon)."""
    lon = ((LON25 + 180) % 360) - 180
    o = np.argsort(lon, kind="stable")
    lon_s = lon[o]
    la = (LAT25 <= spec["lat"][0] + 1e-6) & (LAT25 >= spec["lat"][1] - 1e-6)
    lo = (lon_s >= spec["lon"][0] - 1e-6) & (lon_s <= spec["lon"][1] + 1e-6)
    b = np.asarray(a)[..., la, :][..., o][..., lo]
    return b, LAT25[la], lon_s[lo]


def running_partial(a, n=SMOOTH, axis=0):
    """n-point running mean along axis with partial windows at the ends (the regimes' 5-day mean)."""
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


def box25(field, res):
    """A regular global field (lat 90 -> -90, lon 0 -> 360-res) at `res` deg -> the 2.5 deg NCEP grid by a centred
    box mean of half-width 1.25 deg (edge rows/columns half-weighted), latitudes clipped at the poles. field (..., ny,
    nx). The same reduction as the GEFS members' 5x5 box on 0.5 deg, generalised."""
    f = np.asarray(field, dtype="float64")
    ny, nx = f.shape[-2:]
    lat = 90.0 - res * np.arange(ny)
    lon = res * np.arange(nx)
    h = 1.25
    wy = np.clip(h + res / 2 - np.abs(lat[None, :] - LAT25[:, None]), 0, res) / res       # (73, ny) overlap weights
    dl = (lon[None, :] - LON25[:, None] + 180.0) % 360.0 - 180.0
    wx = np.clip(h + res / 2 - np.abs(dl), 0, res) / res                                   # (144, nx)
    wy = wy / wy.sum(1, keepdims=True)
    wx = wx / wx.sum(1, keepdims=True)
    return np.matmul(np.matmul(wy, f), wx.T)


# ── summaries ─────────────────────────────────────────────────────────────────────────────────────────────────────
def weekly(P, days=None):
    """(F+1, N) daily probabilities -> list of {w0, w1, ndays, p[F+1]} over the days of each week that exist."""
    rows = []
    for w0, w1 in WEEKS:
        v = P[:, w0 - 1:w1]
        if v.shape[1] == 0:
            continue
        ok = np.isfinite(v).all(0)
        if not ok.any():
            continue
        rows.append(dict(w0=w0, w1=w1, ndays=int(ok.sum()), p=[float(v[k, ok].mean()) for k in range(P.shape[0])]))
    return rows


# ── hindcast skill ────────────────────────────────────────────────────────────────────────────────────────────────
HINDCAST = {"geps": DATA / "regime_hindcast_geps.npz", "gefs": DATA / "regime_hindcast_gefs.npz"}
HINDCAST_LABEL = {"geps": "GEPS8 hindcast 2001–2020 (4-member mean)",
                  "gefs": "GEFSv12 reforecast 2000–2019 (4-member mean)"}


def doy(d: dt.date) -> int:
    return d.timetuple().tm_yday


def hindcast_skill(model: str, init, window: int = 45, n_boot: int = 1000, seed: int = 11, ndays: int = 35):
    """Hit rate of the hindcast ensemble-mean regime by lead day, for the starts within +-window days of init's day of
    year, every valid day classified with its own season's set (forecast AND observation), against two baselines on
    the same starts: persistence (the pattern observed over the three days before day 1, classified with the valid
    day's set) and climatology (always the regime most often observed at that lead in the window).

    Uncertainty: the starts cluster in years and consecutive starts share weather, so the 95% interval is a
    block bootstrap over YEARS (resample the hindcast years with replacement, n_boot times). A week is 'skilful' when
    the bootstrap 2.5th percentile of [hit - max(persistence, climatology)], averaged over the week's leads, is above 0.
    Returns None when the model's hindcast file is absent."""
    p = HINDCAST.get(model)
    if p is None or not p.exists():
        return None
    z = np.load(p)
    base = parse_date(init)
    S = [parse_date(str(x)) for x in z["S"]]
    dd = np.array([((doy(s) - doy(base) + 183) % 366) - 183 for s in S])
    win = np.abs(dd) <= window
    years = np.array([s.year for s in S])[win]
    out = {"model": model, "label": HINDCAST_LABEL.get(model, model), "n_starts": int(win.sum()),
           "years": [int(years.min()), int(years.max())] if win.any() else None, "sectors": {}}
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    boots = [np.concatenate([np.flatnonzero(years == y) for y in rng.choice(uy, uy.size)]) for _ in range(n_boot)]
    L = min(ndays, z["fc_ea"].shape[1])
    for sk in SECTORS:
        fc, ob, ps = z[f"fc_{sk}"][win][:, :L], z[f"ob_{sk}"][win][:, :L], z[f"pers_{sk}"][win][:, :L]
        ok = (fc >= 0) & (ob >= 0)
        okp = ok & (ps >= 0)
        H = np.where(ok, fc == ob, np.nan).astype("float64")
        Pz = np.where(okp, ps == ob, np.nan).astype("float64")
        # climatology: per lead, the class most often observed (5 classes incl. none) - a hit whenever it occurs
        C = np.full(ob.shape, np.nan)
        for j in range(L):
            o = ob[:, j][ob[:, j] >= 0]
            if o.size:
                best = np.bincount(o, minlength=K + 1).argmax()
                C[:, j] = np.where(ob[:, j] >= 0, ob[:, j] == best, np.nan)

        def mean(a, idx=None):
            a = a if idx is None else a[idx]
            with np.errstate(invalid="ignore"):
                return np.nanmean(a, 0)
        hit, pers, clim = mean(H), mean(Pz), mean(C)
        bh = np.array([mean(H, b) for b in boots])
        bp = np.array([mean(Pz, b) for b in boots])
        bc = np.array([mean(C, b) for b in boots])
        weeks = []
        for w0, w1 in WEEKS:
            if w0 > L:
                continue
            sl = slice(w0 - 1, min(w1, L))
            d = bh[:, sl].mean(1) - np.maximum(bp[:, sl].mean(1), bc[:, sl].mean(1))
            lo = float(np.nanpercentile(d, 2.5))
            weeks.append(dict(w0=w0, w1=w1, hit=float(np.nanmean(hit[sl])),
                              base=float(max(np.nanmean(pers[sl]), np.nanmean(clim[sl]))),
                              diff_lo=lo, diff_hi=float(np.nanpercentile(d, 97.5)), skilful=bool(lo > 0)))
        out["sectors"][sk] = dict(
            day=list(range(1, L + 1)), hit=hit.tolist(), hit_lo=np.nanpercentile(bh, 2.5, 0).tolist(),
            hit_hi=np.nanpercentile(bh, 97.5, 0).tolist(), persistence=pers.tolist(),
            pers_lo=np.nanpercentile(bp, 2.5, 0).tolist(), pers_hi=np.nanpercentile(bp, 97.5, 0).tolist(),
            clim=clim.tolist(), n_lead=ok.sum(0).tolist(), weeks=weeks,
            skilful_days=[int(j + 1) for j in range(L)
                          if np.nanpercentile(bh[:, j] - np.maximum(bp[:, j], bc[:, j]), 2.5) > 0])
    return out


# ── the per-model member file (the contract every producer writes) ────────────────────────────────────────────────
SCHEMA = "regimes-members-v1"


def members_record(model, label, init, n_members, days, labels_by_sector, observed_by_sector=None, notes=None,
                   max_age_days=None):
    """The per-model JSON the multi-model view reads. days: valid dates of forecast day 1..N (day d = init + d - 1);
    labels_by_sector: {sk: (N, members) int labels within each day's set}; observed: {sk: (dates, labels)}."""
    days = [parse_date(d) for d in days]
    rec = {"schema": SCHEMA, "model": model, "label": label, "init": parse_date(init).isoformat(),
           "n_members": int(n_members),
           "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "valid_convention": "day d valid on the UTC day init + (d - 1)",
           "valid": [d.isoformat() for d in days], "set": [season_of_date(d) for d in days],
           "label_note": "per day: regime index 0-3 within that day's seasonal set (data/regimes_ref.npz order), "
                         "4 = no regime, -1 = missing", "sectors": {}}
    if max_age_days is not None:
        rec["max_age_days"] = max_age_days
    if notes:
        rec["notes"] = notes
    for sk, lab in labels_by_sector.items():
        s = {"lab": np.asarray(lab).astype(int).tolist()}
        if observed_by_sector and sk in observed_by_sector and observed_by_sector[sk] is not None:
            od, ol = observed_by_sector[sk]
            s["observed"] = {"t": [parse_date(d).isoformat() for d in od], "lab": [int(x) for x in ol]}
        rec["sectors"][sk] = s
    return rec
