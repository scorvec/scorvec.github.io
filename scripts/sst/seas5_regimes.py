#!/usr/bin/env python3
"""Weather-regime frequencies in the SEAS5 seasonal forecast, against SEAS5's own climate
(user 2026-09-28: "extend this to seasonal models (maybe see how the regime distribution differs
from model climate)").

Regimes: the SAME sets the subseasonal pages use (GEPS ~/data_archive/geps_subx/build_regimes.py,
GEFS gefs_patterns_ref.npz, identical centroids): k-means k = 4 on 5-day-mean 500 hPa anomalies,
NCEP/NCAR R1 1991-2020, Euro-Atlantic and North America / Pacific sectors, a cold set (Oct-Mar)
and a warm set (Apr-Sep), 2.5 deg; a day whose pattern correlation with its nearest centroid is
below 0.25 is "no regime". This script only READS those centroids.

Model-relative frequencies. Every SEAS5 member-day (the 51-member forecast and the 25-member
hindcast of the same start month, the FULL SEAS5 hindcast 1981-2016, n = 36 years) becomes an anomaly against SEAS5's OWN lead-dependent
hindcast climatology (drift and bias out, as on the GEPS page), is smoothed with the regimes'
5-day running mean along the lead, and is assigned to the nearest centroid of the set for its
valid month. The forecast frequency of each regime in a month or season is then read against the
hindcast frequency at the same start and lead -- the model's own regime climate -- so SEAS5's
regime-frequency biases cancel.

Warming. Geopotential height rises with the warming trend, and a uniform rise pushes a
nearest-centroid (Euclidean) classifier toward the ridge regimes. Both the model and the
observations are therefore read against a linear trend of height by latitude (ERA5 1979-2025,
row means over the whole 170W-30E box, annual means): the hindcast year y and the forecast are
offset by s(lat) x (y - 1998.5), the observed days by the fitted line. What is left is circulation.

Skill (the site's significance rule). In the hindcast, the predicted frequency anomaly of each
regime, month and sector (forecast year minus the other 35 years, with the model climatology
itself recomputed without the held-out year) is correlated with the OBSERVED frequency anomaly
(ERA5 daily z500 from the local store, regridded to 2.5 deg, same anomaly, trend, smoothing and
classification). r with a 90 % bootstrap interval over years, a one-sided permutation p, and the
Benjamini-Hochberg false-discovery rate at q = 0.10 across every regime x target x sector cell of
the start (Wilks 2016). The ratio of predictable components (RPC = r / (sd of the ensemble-mean
frequency / sd of member frequencies); Eade et al. 2014) is reported with every correlation:
RPC > 1 is the seasonal "signal-to-noise paradox", a skilful forecast whose shifts are too small. A forecast shift is drawn as meaningful only where the hindcast skill
passes the FDR test AND the 90 % member-bootstrap range of the forecast excludes the model climate.

ENSO. (1) Conditional hindcast verification: the held-out predicted and the observed frequency
anomalies are composited over the hindcast's El Nino (RONI >= 0.5 in NDJ or DJF), strong El Nino
(>= 1.5) and La Nina (<= -0.5) winters, each tested by permutation against random same-size sets of
hindcast years, with bootstrap intervals of both and of their difference (own FDR family).
(2) The observed regime frequencies in strong El Nino winters (ERA5 1959-2025: 1965, 1972, 1986,
1997, 2009, 2015; 1982 and 1991 screened for El Chichon and Pinatubo and reported separately)
against all winters, permutation p (own FDR family); SEAS5's hindcast of the same winters inside
1981-2016 is the open circle on the figure.

Outputs: assets/sst/seas5_regimes_{na,ea}.webp, assets/sst/data/seas5_regimes.json.
Data: seas5_regimes_fetch.py (CDS, cached in scripts/sst/data/seas5/regimes/).

    python seas5_regimes.py --issue 202609
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import ASSETS                                                 # noqa: E402
from seas5_regimes_fetch import DIR, HC_YEARS, fc_file, hc_files                 # noqa: E402

G0 = 9.80665
K = 4
R_MIN = 0.25
SMOOTH = 5
CLIM_SMOOTH = 15                    # lead-day smoothing of the model climatology
SECTORS = {"na": dict(label="North America / Pacific", lat=(80, 20), lon=(-170, -30)),
           "ea": dict(label="Euro-Atlantic", lat=(80, 20), lon=(-90, 30))}
SEASONS = {"cold": (10, 11, 12, 1, 2, 3), "warm": (4, 5, 6, 7, 8, 9)}
BOX_LAT = np.arange(80.0, 19.99, -2.5)                  # 25
BOX_LON = np.arange(-170.0, 30.01, 2.5)                 # 81
REG_NC = Path.home() / "data_archive" / "geps_subx" / "telecon" / "regimes.nc"
REG_NAMES = Path.home() / "data_archive" / "geps_subx" / "telecon" / "regime_names.json"
REG_NPZ = Path.home() / "data_archive" / "gefs_ref" / "gefs_patterns_ref.npz"
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily" / "z500"
ERA5_Y0 = 1959
TREND_YEARS = (1979, 2025)
STRONG = [1965, 1972, 1982, 1986, 1991, 1997, 2009, 2015]     # RONI >= 1.5 in NDJ or DJF (cpc_official.json)
VOLCANIC = {1982: "El Chichón", 1991: "Pinatubo"}
Q_FDR = 0.10
NBOOT = 5000
NPERM = 10000
OUT_JSON = ASSETS / "data" / "seas5_regimes.json"
# one colour per regime FAMILY, as on the subseasonal pages (scripts/regimes/regime_core.FAMILIES): cold and warm
# regimes whose centroids correlate >= 0.6 share a colour; index = regime number within the set
COLORS = {("na", "cold"): ["#2c6fbb", "#7b4fa0", "#d08a2a", "#b4453c"],
          ("na", "warm"): ["#2c6fbb", "#3f9b6a", "#d08a2a", "#b4453c"],
          ("ea", "cold"): ["#2c6fbb", "#3f9b6a", "#d08a2a", "#b4453c"],
          ("ea", "warm"): ["#2c6fbb", "#b4453c", "#3f9b6a", "#d08a2a"]}
INK, MUTED = "#1f1d1a", "#6f6b64"
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def season_of(month: int) -> str:
    return "cold" if month in SEASONS["cold"] else "warm"


# ── regimes (read only) ────────────────────────────────────────────────────────
def load_regimes():
    """{(sector, set): (centroids (K, lat, lon), lat, lon)}, names {(sector, set): [K names]}."""
    cent = {}
    if REG_NC.exists():
        import xarray as xr
        ds = xr.open_dataset(REG_NC)
        for sk in SECTORS:
            for s in SEASONS:
                c = ds[f"{sk}_{s}"]
                cent[(sk, s)] = (c.values.astype("float64"), c[f"lat_{sk}"].values, c[f"lon_{sk}"].values)
    else:
        d = np.load(REG_NPZ)
        for sk in SECTORS:
            for s in SEASONS:
                cent[(sk, s)] = (d[f"reg_{sk}_{s}"].astype("float64"), d[f"reg_lat_{sk}"], d[f"reg_lon_{sk}"])
    raw = json.loads(REG_NAMES.read_text()) if REG_NAMES.exists() else {}
    names = {(sk, s): [raw.get(f"{sk}_{s}_{k}", f"R{k + 1}") for k in range(K)] for sk in SECTORS for s in SEASONS}
    for (sk, s), (c, lat, lon) in cent.items():            # the box must contain the sector grid exactly
        assert np.allclose(lat, BOX_LAT) and np.isin(np.round(lon, 3), np.round(BOX_LON, 3)).all(), (sk, s)
    return cent, names


def sector_idx(sk):
    lo = SECTORS[sk]["lon"]
    return np.flatnonzero((BOX_LON >= lo[0] - 1e-6) & (BOX_LON <= lo[1] + 1e-6))


class Classifier:
    """Nearest centroid in the weighted Euclidean sense of build_regimes.assign, "no regime" (= K)
    below R_MIN pattern correlation."""

    def __init__(self, cent):
        self.cent = cent
        self.w = {}
        self.Cw = {}
        for (sk, s), (c, lat, lon) in cent.items():
            w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
            self.w[sk] = w.ravel().astype("float32")
            Cw = (c * w).reshape(K, -1).astype("float32")
            cm = Cw - Cw.mean(1, keepdims=True)
            self.Cw[(sk, s)] = (Cw, (Cw ** 2).sum(1), cm / np.linalg.norm(cm, axis=1, keepdims=True))

    def __call__(self, X, sk, s):
        """X: (n, lat*lon_sector) anomalies (m) -> labels 0..K."""
        Cw, c2, cmn = self.Cw[(sk, s)]
        Xw = X * self.w[sk]
        d2 = (Xw ** 2).sum(1)[:, None] - 2.0 * Xw @ Cw.T + c2[None]
        lab = d2.argmin(1)
        xm = Xw - Xw.mean(1, keepdims=True)
        corr = (xm @ cmn.T) / (np.linalg.norm(xm, axis=1)[:, None] + 1e-9)
        r = corr[np.arange(len(lab)), lab]
        return np.where(r < R_MIN, K, lab)


def running_valid(a, n):
    """Centred n-day running mean along axis 0, NaN where the window is incomplete."""
    cs = np.cumsum(np.concatenate([np.zeros_like(a[:1]), a], axis=0), axis=0)
    out = np.full_like(a, np.nan)
    h = n // 2
    out[h: a.shape[0] - h] = (cs[n:] - cs[:-n]) / n
    return out


def running_partial(a, n, axis):
    """Centred n-point running mean along axis with partial windows at the ends (GEPS regimes_geps)."""
    a = np.moveaxis(a, axis, 0)
    cs = np.cumsum(np.concatenate([np.zeros_like(a[:1]), a], axis=0), axis=0)
    L, h = a.shape[0], n // 2
    i0 = np.clip(np.arange(L) - h, 0, L)
    i1 = np.clip(np.arange(L) + h + 1, 0, L)
    shp = (L,) + (1,) * (a.ndim - 1)
    out = (cs[i1] - cs[i0]) / (i1 - i0).reshape(shp)
    return np.moveaxis(out.astype(a.dtype), 0, axis)


# ── observations: ERA5 daily z500 from the local store ─────────────────────────
def era5_box():
    """(dates, z (T, 25, 81) m) on the 2.5 deg box, cached."""
    import pandas as pd
    import xarray as xr
    files = sorted(ERA5.glob("z500_*.nc"))
    cache = DIR / "era5_z500_box25.npz"
    stamp = f"{len(files)}:{int(files[-1].stat().st_mtime)}"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if str(d["stamp"]) == stamp:
            return pd.DatetimeIndex(d["dates"]), d["z"]
    zs, ts = [], []
    for f in files:
        y = int(f.stem.split("_")[1])
        if y < ERA5_Y0:
            continue
        ds = xr.open_dataset(f)
        v = ds[list(ds.data_vars)[0]].transpose("time", "latitude", "longitude")
        v = v.assign_coords(latitude=np.round(v.latitude.values.astype("float64"), 3),
                            longitude=((np.round(v.longitude.values.astype("float64"), 3) + 180) % 360) - 180).sortby("longitude")
        v = v.sel(latitude=slice(15, 85)).load()
        v = v.interp(latitude=BOX_LAT, longitude=BOX_LON)
        zs.append(v.values.astype("float32")); ts.append(pd.DatetimeIndex(v.time.values).normalize())
        ds.close()
    z = np.concatenate(zs); t = ts[0].append(ts[1:])
    if float(np.nanmean(z)) > 20000:                    # store attrs say m2/s2; check the values, not the attrs
        z = z / G0
    assert 5000 < float(np.nanmean(z)) < 6000, float(np.nanmean(z))
    _, idx = np.unique(t.values, return_index=True)
    t, z = t[idx], z[idx]
    np.savez(cache, dates=t.values, z=z, stamp=stamp)
    return t, z


def doy_clim(t, z, y0=1991, y1=2020):
    """Day-of-year mean over the base years, 31-day circular running mean (build_telecon_patterns.doy_clim)."""
    sel = (t.year >= y0) & (t.year <= y1)
    doy = t.dayofyear.values
    c = np.full((366,) + z.shape[1:], np.nan, dtype="float64")
    for d in range(1, 367):
        m = sel & (doy == d)
        if m.any():
            c[d - 1] = z[m].mean(0)
    if np.isnan(c[365]).all():
        c[365] = c[364]
    ext = np.concatenate([c[-15:], c, c[:15]], axis=0)
    cs = np.cumsum(np.concatenate([np.zeros_like(ext[:1]), ext]), axis=0)
    return ((cs[31:] - cs[:-31]) / 31.0).astype("float32")


def year_frac(t):
    return t.year.values + (t.dayofyear.values - 0.5) / np.where(t.is_leap_year, 366.0, 365.0)


def trend_by_lat(t, anom):
    """Linear trend of the annual-mean row-mean anomaly, per latitude, over TREND_YEARS.
    Returns (intercept, slope) arrays (25,), anomaly = a + s * year."""
    rows = anom.mean(2)                                   # (T, lat)
    yrs = np.arange(TREND_YEARS[0], TREND_YEARS[1] + 1)
    am = np.array([rows[t.year == y].mean(0) for y in yrs])
    X = np.column_stack([np.ones(yrs.size), yrs + 0.5])
    coef = np.linalg.lstsq(X, am, rcond=None)[0]
    return coef[0], coef[1]


def observed(clf):
    """Observed regime labels per sector (ERA5, anomaly vs 1991-2020 day of year, trend by latitude out,
    5-day mean) -> dates, {sector: labels (T,)}, trend coefficients."""
    t, z = era5_box()
    clim = doy_clim(t, z)
    an = z - clim[t.dayofyear.values - 1]
    a0, s = trend_by_lat(t, an)
    an = an - (a0[None, :, None] + s[None, :, None] * year_frac(t)[:, None, None]).astype("float32")
    sm = running_valid(an, SMOOTH)
    ok = np.isfinite(sm).all(axis=(1, 2))
    labs = {}
    for sk in SECTORS:
        ii = sector_idx(sk)
        X = sm[:, :, ii].reshape(len(t), -1)
        lab = np.full(len(t), -1, dtype=int)
        for sset in SEASONS:
            m = ok & np.isin(t.month, SEASONS[sset])
            lab[m] = clf(X[m], sk, sset)
        labs[sk] = lab
    return t, labs, (a0, s)


# ── SEAS5 ─────────────────────────────────────────────────────────────────────
def _daily_from_grib(path, hindcast):
    """-> (years or None, daily z (…, member, day, lat, lon) in m), day d = UTC day init + d - 1."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = ds["z"]
    hrs = (v.step.values / np.timedelta64(1, "h")).astype(int)
    assert np.allclose(v.latitude.values, BOX_LAT) and np.allclose(v.longitude.values, BOX_LON)
    ndays = 215
    day = (hrs // 24)                                     # 0-based day index: 12 h -> 0, 24 h -> 1, ...
    keep = day < ndays
    if hindcast:
        v = v.transpose("time", "number", "step", "latitude", "longitude")
        years = [int(str(x)[:4]) for x in v.time.values]
    else:
        v = v.transpose("number", "step", "latitude", "longitude")
        years = None
    a = v.values[..., keep, :, :].astype("float32") / G0
    d = day[keep]
    out = np.zeros(a.shape[:-3] + (ndays,) + a.shape[-2:], dtype="float32")
    cnt = np.zeros(ndays)
    for j, dd in enumerate(d):
        out[..., dd, :, :] += a[..., j, :, :]; cnt[dd] += 1
    out /= cnt.reshape((ndays, 1, 1))
    return years, out


def hindcast_daily(month):
    cache = DIR / f"hc_z500_daily_{month}_{HC_YEARS[0]}_{HC_YEARS[-1]}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode=None)
    parts = []
    for f in hc_files(month):
        yrs, a = _daily_from_grib(f, True)
        parts.append(a); print(f"    {f.name}: years {yrs[0]}-{yrs[-1]}", flush=True)
    H = np.concatenate(parts)                             # (24, 25, 215, lat, lon)
    assert H.shape[0] == len(HC_YEARS), H.shape
    np.save(cache, H)
    return H


def forecast_daily(ym):
    return _daily_from_grib(fc_file(ym), False)[1]          # (51, 215, lat, lon)


def lead_dates(ym):
    import pandas as pd
    return pd.date_range(f"{ym[:4]}-{ym[4:]}-01", periods=215, freq="D")


def targets_for(ym):
    """Months fully inside the 215 lead days, plus 3-month seasons of one regime set."""
    dates = lead_dates(ym)
    months = []
    for m0 in range(8):
        y, m = divmod(int(ym[4:]) - 1 + m0, 12)
        y += int(ym[:4]); m += 1
        sel = (dates.year == y) & (dates.month == m)
        if sel.sum() == calendar.monthrange(y, m)[1]:
            months.append((y, m))
    out = [dict(key=MON[m - 1], label=MON[m - 1], kind="month", months=[m], set=season_of(m), cal=[(y, m)])
           for y, m in months]
    for i in range(len(months) - 2):
        three = months[i:i + 3]
        sets = {season_of(m) for _, m in three}
        if len(sets) == 1:
            lab = "".join(MON[m - 1][0] for _, m in three)
            out.append(dict(key=lab, label=lab, kind="season", months=[m for _, m in three], set=sets.pop(), cal=three))
    return out


def model_freqs(labels, dates, tg):
    """labels (..., 215) -> frequencies (..., K+1) over the target's days."""
    cal = set(map(tuple, tg["cal"]))
    sel = np.array([(d.year, d.month) in cal for d in dates])
    lab = labels[..., sel]
    return np.stack([(lab == k).mean(-1) for k in range(K + 1)], axis=-1)


def classify_model(clf, X, dates, sk):
    """X: (..., 215, lat, lon_sector) smoothed anomalies -> labels (..., 215) with the valid month's set."""
    lead_shape = X.shape[:-3]
    lab = np.empty(lead_shape + (215,), dtype=np.int8)
    for sset in SEASONS:
        m = np.isin(dates.month, SEASONS[sset])
        if not m.any():
            continue
        Xs = X[..., m, :, :]
        flat = Xs.reshape(-1, Xs.shape[-2] * Xs.shape[-1])
        out = np.empty(flat.shape[0], dtype=np.int8)
        for i0 in range(0, flat.shape[0], 200000):
            out[i0:i0 + 200000] = clf(np.ascontiguousarray(flat[i0:i0 + 200000]), sk, sset)
        lab[..., m] = out.reshape(Xs.shape[:-2])
    return lab


# ── statistics ────────────────────────────────────────────────────────────────
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def skill_stats(p, o, rng):
    r = corr(p, o)
    n = len(p)
    idx = rng.integers(0, n, size=(NBOOT, n))
    pb, ob = p[idx], o[idx]
    pb = pb - pb.mean(1, keepdims=True); ob = ob - ob.mean(1, keepdims=True)
    den = np.sqrt((pb * pb).sum(1) * (ob * ob).sum(1))
    rb = np.where(den > 0, (pb * ob).sum(1) / np.where(den > 0, den, 1), np.nan)
    lo, hi = np.nanpercentile(rb, [5, 95])
    po = o[np.argsort(rng.random((NPERM, n)), axis=1)]
    pc, poc = p - p.mean(), po - po.mean(1, keepdims=True)
    den = np.sqrt((pc * pc).sum() * (poc * poc).sum(1))
    perm = np.where(den > 0, (poc @ pc) / np.where(den > 0, den, 1), -np.inf)
    pval = (1 + np.sum(perm >= r - 1e-9)) / (NPERM + 1) if np.isfinite(r) else 1.0
    sp = p.std()
    slope = float(((p - p.mean()) * (o - o.mean())).sum() / ((p - p.mean()) ** 2).sum()) if sp > 0 else np.nan
    return dict(r=round(r, 3), r_lo=round(float(lo), 3), r_hi=round(float(hi), 3), p=round(float(pval), 4),
                slope=round(slope, 2), sd_pred=round(float(p.std(ddof=1)) * 100, 2), sd_obs=round(float(o.std(ddof=1)) * 100, 2))


def bh(pvals, q):
    p = np.asarray(pvals, float)
    n = p.size
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok).max() + 1]] = True
    return sig


# ── main computation ──────────────────────────────────────────────────────────
def enso_table():
    """{start year: (max RONI of NDJ/DJF, min of the two)} for the winter a start's lead window covers."""
    d = json.loads((ASSETS / "data" / "cpc_official.json").read_text())["roni"]
    return d


def enso_of(roni, Y, start_month):
    w = Y if start_month >= 7 else Y - 1              # the winter Dec w / Jan w+1
    v = [roni.get(f"{w}-12"), roni.get(f"{w + 1}-01")]
    v = [x for x in v if x is not None]
    return (max(v), min(v)) if v else (np.nan, np.nan)


def enso_class(mx, mn):
    if mx >= 1.5:
        return "strong_el_nino"
    if mx >= 0.5:
        return "el_nino"
    if mn <= -0.5:
        return "la_nina"
    return "neutral"


def seas5_rnino34_djf(ym):
    """SEAS5's own ensemble-mean relative Nino-3.4 for DJF of this issue (seas5_outlook.json), for the text."""
    try:
        d = json.loads((ASSETS / "data" / "seas5_outlook.json").read_text())
        e = d["indices"][ym]
        v = [e["rnino34"]["mean"][e["valid"].index(m)] for m in e["valid"] if m[5:] in ("12", "01", "02")]
        return dict(seas5_relative_nino34_djf=round(float(np.mean(v)), 2), months=len(v)) if v else None
    except Exception:                                                            # noqa: BLE001
        return None


def composite_test(pred, obs, idx, rng):
    """Composite of the held-out predicted and the observed frequency anomaly over the years idx, each tested by
    permutation against random same-size sets of hindcast years (two-sided), plus bootstrap intervals."""
    idx = np.asarray(idx)
    n, N = len(idx), len(pred)
    pc, oc = float(pred[idx].mean()), float(obs[idx].mean())
    ridx = np.argsort(rng.random((NPERM, N)), axis=1)[:, :n]
    pn, on = pred[ridx].mean(1), obs[ridx].mean(1)
    p_pred = (1 + np.sum(np.abs(pn - pred.mean()) >= abs(pc - pred.mean()) - 1e-12)) / (NPERM + 1)
    p_obs = (1 + np.sum(np.abs(on - obs.mean()) >= abs(oc - obs.mean()) - 1e-12)) / (NPERM + 1)
    b = idx[rng.integers(0, n, size=(NBOOT, n))]
    pb, ob = pred[b].mean(1), obs[b].mean(1)
    q = lambda a: [round(float(x) * 100, 1) for x in np.percentile(a, [5, 95])]
    return dict(n=n, pred=round(pc * 100, 1), pred_ci=q(pb), p_pred=round(float(p_pred), 4),
                obs=round(oc * 100, 1), obs_ci=q(ob), p_obs=round(float(p_obs), 4), diff_ci=q(ob - pb))


def compute(ym):
    t0 = time.time()
    rng = np.random.default_rng(20260928)
    cent, names = load_regimes()
    clf = Classifier(cent)
    month = ym[4:]
    dates = lead_dates(ym)
    tgs = targets_for(ym)
    main_set = max(SEASONS, key=lambda s: sum(tg["kind"] == "month" and tg["set"] == s for tg in tgs))
    roni = enso_table()

    print("  observations (ERA5 local store) …", flush=True)
    ot, olab, (a0, s_lat) = observed(clf)
    print(f"    {ot[0]:%Y-%m-%d} .. {ot[-1]:%Y-%m-%d}, {time.time() - t0:.0f} s", flush=True)

    print("  SEAS5 hindcast …", flush=True)
    H = hindcast_daily(month)                            # (years, 25, 215, lat, lon)
    years = np.array([int(y) for y in HC_YEARS])
    ny = len(years)
    raw_mean = H.mean(axis=(0, 1))                       # (215, lat, lon)
    M = running_partial(raw_mean, CLIM_SMOOTH, axis=0)
    ybar = years.mean()                                   # the hindcast's mean start year
    trend = (s_lat[:, None] * np.ones((1, BOX_LON.size))).astype("float32")      # m per year, (lat, lon)
    A = H - M[None, None]
    A -= (years - ybar).reshape(-1, 1, 1, 1, 1).astype("float32") * trend[None, None, None]
    A = running_partial(A, SMOOTH, axis=2)
    # held-out-year shifts of the model climatology: delta_y = smooth(Fbar_y - mean) / (n - 1)
    ens = H.mean(1)
    delta = running_partial(running_partial(ens - raw_mean[None], CLIM_SMOOTH, axis=1), SMOOTH, axis=1) / (ny - 1)
    del H, ens

    print("  SEAS5 forecast …", flush=True)
    F = forecast_daily(ym)                               # (51, 215, lat, lon)
    yf = int(ym[:4])
    AF = F - M[None] - np.float32(yf - ybar) * trend[None, None]
    AF_raw = running_partial(F - M[None], SMOOTH, axis=1)  # sensitivity: no trend adjustment
    AF = running_partial(AF, SMOOTH, axis=1)
    nmem = F.shape[0]

    ev = {int(Y): enso_of(roni, int(Y), int(month)) for Y in range(ERA5_Y0, yf + 1)}
    ecls = {Y: enso_class(*v) for Y, v in ev.items() if np.isfinite(v[0])}
    groups = {"el_nino": [i for i, Y in enumerate(years) if ecls.get(int(Y)) in ("el_nino", "strong_el_nino")],
              "strong_el_nino": [i for i, Y in enumerate(years) if ecls.get(int(Y)) == "strong_el_nino"],
              "la_nina": [i for i, Y in enumerate(years) if ecls.get(int(Y)) == "la_nina"]}
    strong_all = sorted(Y for Y, c in ecls.items() if c == "strong_el_nino")
    out = dict(issue=ym, issue_label=f"{calendar.month_name[int(month)]} {ym[:4]}", start_month=month,
               generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), members=nmem,
               hindcast=dict(years=[int(years[0]), int(years[-1])], n_years=int(ny), members=25), main_set=main_set,
               enso=dict(rule="CPC official RONI of the winter the lead window covers: strong El Nino if NDJ or DJF >= 1.5, "
                              "El Nino if >= 0.5, La Nina if NDJ or DJF <= -0.5",
                         this_winter=seas5_rnino34_djf(ym),
                         hindcast_years={g: [int(years[i]) for i in ix] for g, ix in groups.items()},
                         strong_observed=[Y for Y in strong_all if Y not in VOLCANIC],
                         screened={str(y): v for y, v in VOLCANIC.items()}),
               trend_m_per_decade={f"{BOX_LAT[i]:.1f}": round(float(s_lat[i]) * 10, 1) for i in range(0, 25, 4)},
               q_fdr=Q_FDR, sectors={})
    fam_skill, fam_enso, fam_obs = [], [], []          # FDR families: (row, key path, p)
    for sk, spec in SECTORS.items():
        print(f"  {sk}: classifying …", flush=True)
        ii = sector_idx(sk)
        As = np.ascontiguousarray(A[..., ii])
        lab_hc = classify_model(clf, As, dates, sk)       # (years, 25, 215)
        lab_fc = classify_model(clf, AF[..., ii], dates, sk)   # (51, 215)
        lab_fr = classify_model(clf, AF_raw[..., ii], dates, sk)
        loyo = [classify_model(clf, As + delta[iy][None, None][..., ii], dates, sk) for iy in range(ny)]
        lab_o = olab[sk]
        sec = dict(label=spec["label"], names={s: names[(sk, s)] for s in SEASONS}, targets=[])
        for ti, tg in enumerate(tgs):
            fh = model_freqs(lab_hc, dates, tg)            # (years, 25, K+1)
            ff = model_freqs(lab_fc, dates, tg)            # (51, K+1)
            fr = model_freqs(lab_fr, dates, tg)
            clim = fh.mean(axis=(0, 1))
            pred = np.zeros((ny, K + 1))
            for iy in range(ny):
                fl = model_freqs(loyo[iy], dates, tg).mean(1)
                pred[iy] = fl[iy] - np.delete(fl, iy, axis=0).mean(0)
            var_mem = (fh - clim).reshape(-1, K + 1).var(0)          # member-level (total) variance
            var_sig = fh.mean(1).var(0)                               # ensemble-mean (signal) variance

            def obs_freq(Y):
                dy = Y - yf
                sel = np.zeros(len(ot), bool)
                for (cy, cm) in tg["cal"]:
                    sel |= (ot.year == cy + dy) & (ot.month == cm)
                lb = lab_o[sel]
                if sel.sum() == 0 or (lb < 0).any():
                    return None
                return np.array([(lb == k).mean() for k in range(K + 1)])
            ob = np.array([obs_freq(int(Y)) for Y in years])
            obs_anom = ob - ob.mean(0)
            # observed long-record strong El Nino composite (ERA5, volcano-screened)
            allY = [Y for Y in range(ERA5_Y0, yf + 1) if obs_freq(Y) is not None]
            fo = {Y: obs_freq(Y) for Y in allY}
            allF = np.array([fo[Y] for Y in allY])
            base = allF.mean(0)
            nino = [Y for Y in strong_all if Y not in VOLCANIC and Y in fo]
            volc = [Y for Y in strong_all if Y in VOLCANIC and Y in fo]
            comp = np.array([fo[Y] for Y in nino]).mean(0) - base
            compv = np.array([fo[Y] for Y in strong_all if Y in fo]).mean(0) - base
            ridx = np.argsort(rng.random((NPERM, len(allY))), axis=1)[:, :len(nino)]
            null = allF[ridx].mean(1) - base
            p_comp = (1 + (np.abs(null) >= np.abs(comp)[None] - 1e-12).sum(0)) / (NPERM + 1)
            bidx = rng.integers(0, len(nino), size=(NBOOT, len(nino)))
            nb = np.array([fo[Y] for Y in nino])[bidx].mean(1) - base
            c_lo, c_hi = np.percentile(nb, [5, 95], axis=0)
            hn = [i for i, Y in enumerate(years) if int(Y) in nino]
            bi = rng.integers(0, nmem, size=(NBOOT, nmem))
            fb = ff[bi].mean(1)
            f_lo, f_hi = np.percentile(fb, [5, 95], axis=0)
            fm = ff.mean(0)
            rows = []
            for k in range(K + 1):
                st = skill_stats(pred[:, k], obs_anom[:, k], rng)
                ratio = np.sqrt(var_sig[k] / var_mem[k]) if var_mem[k] > 0 else np.nan
                st["rpc"] = round(float(st["r"] / ratio), 2) if (np.isfinite(ratio) and ratio > 0 and st["r"] is not None) else None
                st["signal_to_total"] = round(float(ratio), 3)
                st["n"] = int(ny)
                row = dict(k=k, name=(names[(sk, tg["set"])][k] if k < K else "no regime"),
                           fc=round(float(fm[k]) * 100, 1), fc_lo=round(float(f_lo[k]) * 100, 1), fc_hi=round(float(f_hi[k]) * 100, 1),
                           clim=round(float(clim[k]) * 100, 1),
                           anom=round(float(fm[k] - clim[k]) * 100, 1),
                           anom_lo=round(float(f_lo[k] - clim[k]) * 100, 1), anom_hi=round(float(f_hi[k] - clim[k]) * 100, 1),
                           anom_no_trend_adj=round(float(fr.mean(0)[k] - clim[k]) * 100, 1),
                           obs_clim_hindcast_years=round(float(ob.mean(0)[k]) * 100, 1),
                           skill=st,
                           enso_hindcast={g: composite_test(pred[:, k], obs_anom[:, k], ix, rng) for g, ix in groups.items() if len(ix) >= 2},
                           obs_nino=dict(anom=round(float(comp[k]) * 100, 1), lo=round(float(c_lo[k]) * 100, 1),
                                         hi=round(float(c_hi[k]) * 100, 1), p=round(float(p_comp[k]), 4),
                                         n=len(nino), winters=nino, with_volcanic_anom=round(float(compv[k]) * 100, 1),
                                         volcanic_winters={str(Y): round(float(fo[Y][k] - base[k]) * 100, 1) for Y in volc},
                                         base_winters=len(allY)),
                           hc_nino=dict(years=[int(years[i]) for i in hn],
                                        pred=(round(float(pred[hn, k].mean()) * 100, 1) if hn else None),
                                        obs=(round(float(obs_anom[hn, k].mean()) * 100, 1) if hn else None)))
                rows.append(row)
                if k < K:
                    fam_skill.append((row, ("skill",), st["p"]))
                    for g, c in row["enso_hindcast"].items():
                        fam_enso.append((c, ("pred",), c["p_pred"])); fam_enso.append((c, ("obs",), c["p_obs"]))
                    fam_obs.append((row["obs_nino"], (), row["obs_nino"]["p"]))
            sec["targets"].append(dict(key=tg["key"], label=tg["label"], kind=tg["kind"], set=tg["set"],
                                       valid=[f"{y}-{m:02d}" for y, m in tg["cal"]], regimes=rows))
        out["sectors"][sk] = sec
        del loyo, As
    # Benjamini-Hochberg, q = 0.10, within each family of tests (all regimes x targets x sectors of this start)
    sig = bh([f[2] for f in fam_skill], Q_FDR)
    for (row, _, _), s_ in zip(fam_skill, sig):
        row["skill"]["fdr"] = bool(s_)
        excl = (row["anom_lo"] > 0) or (row["anom_hi"] < 0)
        row["shift_outside_members"] = bool(excl)
        row["meaningful"] = bool(s_ and excl)
    sig_e = bh([f[2] for f in fam_enso], Q_FDR)
    for (c, key, _), s_ in zip(fam_enso, sig_e):
        c[f"fdr_{key[0]}"] = bool(s_)
    sig_o = bh([f[2] for f in fam_obs], Q_FDR)
    for (c, _, _), s_ in zip(fam_obs, sig_o):
        c["fdr"] = bool(s_)
    for sk in SECTORS:
        for tg in out["sectors"][sk]["targets"]:
            nr = tg["regimes"][K]
            nr["skill"]["fdr"] = None; nr["meaningful"] = False
            nr["shift_outside_members"] = bool(nr["anom_lo"] > 0 or nr["anom_hi"] < 0)
    out["tests"] = dict(skill=dict(n=len(fam_skill), fdr_significant=int(sig.sum())),
                        enso_hindcast=dict(n=len(fam_enso), fdr_significant=int(sig_e.sum())),
                        observed_strong_el_nino=dict(n=len(fam_obs), fdr_significant=int(sig_o.sum())))
    out["summary"] = summarise(out)
    print(f"  computed in {time.time() - t0:.0f} s; tests {out['tests']}", flush=True)
    return out


def summarise(doc):
    """Plain-language lines per sector."""
    lines = {}
    for sk, sec in doc["sectors"].items():
        L = []
        ms = [(tg, r) for tg in sec["targets"] if tg["set"] == doc["main_set"] for r in tg["regimes"][:K] if r["meaningful"]]
        ms.sort(key=lambda x: -abs(x[1]["anom"]))
        seen = set()
        for tg, r in ms:
            if (r["name"], tg["kind"]) in seen and len(L) >= 2:
                continue
            seen.add((r["name"], tg["kind"]))
            word = "more" if r["anom"] > 0 else "less"
            when = tg["label"] if tg["kind"] == "season" else calendar.month_name[MON.index(tg["label"]) + 1]
            L.append(f"SEAS5 expects the {r['name']} regime on {r['fc']:.0f}% of {when} days against {r['clim']:.0f}% in its own "
                     f"climate ({word} often; members {r['fc_lo']:.0f}–{r['fc_hi']:.0f}%), and its hindcast has significant skill for "
                     f"that regime there (r {r['skill']['r']:.2f}, 90% interval {r['skill']['r_lo']:.2f} to {r['skill']['r_hi']:.2f}, "
                     f"p {r['skill']['p']:.3f}, passes the false-discovery test).")
            if len(L) >= 4:
                break
        if not L:
            nsk = sum(r["skill"]["fdr"] for tg in sec["targets"] if tg["set"] == doc["main_set"] for r in tg["regimes"][:K])
            L.append("No regime shift passes both tests: " + (f"the hindcast has significant skill in {nsk} regime-month cells, but the forecast "
                     "does not depart from SEAS5's climate there beyond member sampling." if nsk else
                     "the hindcast has no regime-frequency skill that survives the false-discovery test in this sector, so the forecast "
                     "shifts are shown greyed and should be read as unconfirmed."))
        # signal-to-noise where the hindcast is skilful
        sk_cells = [r for tg in sec["targets"] if tg["set"] == doc["main_set"] for r in tg["regimes"][:K]
                    if r["skill"]["fdr"] and r["skill"].get("rpc") is not None]
        if sk_cells:
            rpc = float(np.median([r["skill"]["rpc"] for r in sk_cells]))
            if rpc > 1.2:
                L.append(f"Where the hindcast is skilful its shifts are too small for that skill: median ratio of predictable "
                         f"components {rpc:.1f} over those {len(sk_cells)} cells (1 = a well-sized signal), so the observed shift is "
                         "typically larger than the forecast one — read the sign, not the size.")
        # ENSO-conditional check, El Nino hindcast winters, the main set's seasons
        for tg in sec["targets"]:
            if tg["kind"] != "season" or tg["set"] != doc["main_set"] or tg["key"] not in ("DJF", "JFM"):
                continue
            for r in tg["regimes"][:K]:
                c = r["enso_hindcast"].get("el_nino")
                if c and c.get("fdr_obs"):
                    agree = (c["pred"] * c["obs"] > 0) and c.get("fdr_pred")
                    L.append(f"In the hindcast's {c['n']} El Niño winters the {r['name']} regime was observed {c['obs']:+.0f} points "
                             f"in {tg['label']} (significant), and SEAS5 predicted {c['pred']:+.0f}"
                             + (" — same sign, also significant." if agree else " — not a significant match."))
        lines[sk] = L
    return lines


# ── figure ────────────────────────────────────────────────────────────────────
def render(doc, sk, cent, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    sec = doc["sectors"][sk]
    sset = doc["main_set"]
    COLS = COLORS[(sk, sset)]
    tgs = [tg for tg in sec["targets"] if tg["set"] == sset]
    names = sec["names"][sset]
    c, lat, lon = cent[(sk, sset)]
    spec = SECTORS[sk]
    n = len(tgs)
    W = 13.6
    row_h = 2.05
    top, bot = 1.18, 0.78
    H = top + K * row_h + bot
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    vals = [v for tg in tgs for r in tg["regimes"][:K] for v in (r["anom_lo"], r["anom_hi"], r["obs_nino"]["anom"], r["hc_nino"]["pred"]) if v is not None]
    ylim = max(10.0, np.ceil((max(abs(v) for v in vals) + 6) / 5) * 5)   # room for the labels under the dots
    proj = ccrs.LambertConformal(central_longitude=float(np.mean(spec["lon"])), standard_parallels=(35, 55))
    lim = float(np.nanpercentile(np.abs(c), 99))
    xmap0, xmap1 = 0.018, 0.158
    x0, x1 = 0.205, 0.992
    xs = []
    xi = 0
    for i, tg in enumerate(tgs):
        if i > 0 and tg["kind"] == "season" and tgs[i - 1]["kind"] == "month":
            xi += 0.6
        xs.append(xi); xi += 1
    xs = np.array(xs)
    for k in range(K):
        yb = 1 - (top + (k + 1) * row_h) / H
        hgt = (row_h - 0.30) / H
        axm = fig.add_axes([xmap0, yb + 0.02 / H, xmap1 - xmap0, hgt], projection=proj)
        axm.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
        axm.contourf(lon, lat, c[k], levels=np.linspace(-lim, lim, 17), cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
        axm.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="#333")
        for sp in axm.spines.values():
            sp.set_edgecolor(COLS[k]); sp.set_linewidth(2.0)
        axm.set_title(f"{k + 1}. {names[k]}", fontsize=(10.5 if len(names[k]) < 19 else 9.0), fontweight="bold", color=COLS[k], pad=3)
        ax = fig.add_axes([x0, yb + 0.02 / H, x1 - x0, hgt])
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="#8f8a82", lw=0.9, zorder=1)
        for j, tg in enumerate(tgs):
            r = tg["regimes"][k]
            st = r["skill"]
            if st.get("fdr"):
                ax.axvspan(xs[j] - 0.47, xs[j] + 0.47, facecolor="#e3efe0", edgecolor="none", zorder=0)
                ax.text(xs[j], ylim * 0.97, f"skill r {st['r']:.2f}", ha="center", va="top", fontsize=7.6, color="#2e6b3a", fontweight="bold")
            else:
                ax.axvspan(xs[j] - 0.47, xs[j] + 0.47, facecolor="#f3f2ef", edgecolor="#d9d6d0", hatch="////", lw=0, zorder=0)
                ax.text(xs[j], ylim * 0.97, "skill n.s.", ha="center", va="top", fontsize=7.4, color=MUTED)
            col = COLS[k] if r["meaningful"] else "#a7a29a"
            ax.plot([xs[j] - 0.12] * 2, [r["anom_lo"], r["anom_hi"]], color=col, lw=2.6, solid_capstyle="butt", zorder=4)
            ax.plot(xs[j] - 0.12, r["anom"], "o", ms=8.5, mfc=col, mec="white", mew=1.0, zorder=5)
            on = r["obs_nino"]
            ax.plot(xs[j] + 0.13, on["anom"], "D", ms=6.6, mfc=(INK if on.get("fdr") else "white"), mec=INK, mew=1.1, zorder=5)
            hn = r["hc_nino"]
            if hn["pred"] is not None: ax.plot(xs[j] + 0.30, hn["pred"], "o", ms=5.6, mfc="none", mec="#5b5750", mew=1.1, zorder=5)
            ax.text(xs[j] - 0.12, -ylim * 0.96, f"{r['fc']:.0f}% vs {r['clim']:.0f}%", ha="center", va="bottom", fontsize=7.8,
                    color=(INK if r["meaningful"] else MUTED), fontweight=("bold" if r["meaningful"] else "normal"))
        ax.set_xticks(xs)
        ax.set_xticklabels([tg["label"] for tg in tgs] if k == K - 1 else [], fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.2)
        ax.set_ylabel("pp vs model climate", fontsize=8.2, color=MUTED)
        ax.grid(axis="y", color="#ece9e2", lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    iss = doc["issue_label"]
    hy = doc["hindcast"]["years"]
    hcy = "/".join(str(y) for y in tgs[0]["regimes"][0]["hc_nino"]["years"])
    fig.text(0.012, 1 - 0.12 / H, f"SEAS5 weather regimes · {spec['label']} · {iss} start",
             fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 1 - 0.45 / H,
             f"How often each regime occurs: percentage points of days above or below SEAS5's own climate ({doc['members']} members vs the 25-member "
             f"{hy[0]}–{hy[1]} hindcast, same start and lead; bar = 90% member bootstrap).\n"
             f"{sset.capitalize()}-season regimes: k-means on 5-day-mean 500 hPa anomalies, NCEP R1 1991–2020; height trend removed.\n"
             "Coloured only where the hindcast skill passes the false-discovery test (green column) and the bar clears zero.",
             fontsize=8.8, color="#444", va="top", linespacing=1.45)
    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=COLS[0], mec="white", label="SEAS5 forecast, meaningful (regime colour)"),
               Line2D([], [], marker="o", ls="", ms=8, mfc="#a7a29a", mec="white", label="SEAS5 forecast, not confirmed by the hindcast"),
               Line2D([], [], marker="D", ls="", ms=6.5, mfc=INK, mec=INK, label=f"observed, strong El Niño winters (ERA5 1959–2025, n={tgs[0]['regimes'][0]['obs_nino']['n']}; hollow = not significant)"),
               Line2D([], [], marker="o", ls="", ms=5.6, mfc="none", mec="#5b5750", label=f"SEAS5 hindcast of those winters ({hcy} starts)"),
               Patch(facecolor="#e3efe0", label="hindcast skill significant (FDR q = 0.10)"),
               Patch(facecolor="#f3f2ef", edgecolor="#d9d6d0", hatch="////", label="no significant skill")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.4, frameon=False, bbox_to_anchor=(0.5, 0.005),
               columnspacing=1.6, handletextpad=0.5)
    fig.savefig(out_path, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def clean(o):
    """NaN/inf -> None, so the JSON parses in a browser."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--render-only", action="store_true", help="redraw from the cached JSON")
    a = ap.parse_args()
    cache = DIR / f"regimes_{a.issue}.json"
    if a.render_only and cache.exists():
        doc = json.loads(cache.read_text())
    else:
        if not fc_file(a.issue).exists() or not all(f.exists() for f in hc_files(a.issue[4:])):
            print(f"  regimes: SEAS5 z500 for {a.issue} not fetched yet (seas5_regimes_fetch.py all --issue {a.issue})")
            return 1
        doc = clean(compute(a.issue))
        cache.write_text(json.dumps(doc))
    cent, _ = load_regimes()
    for sk in SECTORS:
        p = ASSETS / f"seas5_regimes_{sk}.webp"
        render(doc, sk, cent, p)
        print(f"  {p.name}")
    OUT_JSON.write_text(json.dumps(clean(doc), separators=(",", ":"), allow_nan=False))
    print(f"  {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
