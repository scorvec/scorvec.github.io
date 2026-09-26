#!/usr/bin/env python3
"""Northern stratosphere history, 1980-2026: the reference behind the "Stratosphere history" items on
circulation.html (built 2026-09-25; user: "the best type of stuff for this site are things that don't need to be
updated every day, so those historical charts using MERRA2 are a great idea").

LAPTOP-ONLY and run once: it reads the raw MERRA-2 zonal-mean archive (scripts/telecon/data/m2_strat, 690 MB,
gitignored), the local ERA5 store and a few small index files, and reduces them to two committed reference files
that the renderer (strat_history.py, GitHub Actions) plots:

    scripts/strat/reference/strat_history.nc          daily series, composites, event maps
    scripts/strat/reference/strat_history_events.json  SSW / strong-vortex catalogue and the winter table

Rerun only if a definition below changes. The MERRA-2 archive ends 2026-06-26 (GES DISC retired its OPeNDAP
service, the fetcher now gets 410 Gone) - the history needs nothing later.

DEFINITIONS
  SSW          Charlton & Polvani (2007) on MERRA-2 daily-mean u at 60N, 10 hPa: central date = first easterly day,
               Nov-Mar; a new event needs 20 consecutive westerly days after the last; final warmings excluded (the
               wind must return westerly for 10 consecutive days before 30 April).
  strong vortex  Baldwin & Dunkerton (2001) style: 10 hPa annular index (minus the standardised polar-cap height
               anomaly) first crossing +1.5, Nov-Mar, events at least 60 days apart.
  polar cap    65-90N, cos-weighted (the fetcher's definition); anomalies against a 4-harmonic seasonal cycle and a
               linear trend fitted per level over the whole record, with a step at Aug 2004 above 5 hPa (MERRA-2
               starts assimilating Aura MLS there - a data step, not climate), standardised by a 3-harmonic sd.
  deep/shallow  mean standardised cap-height anomaly at 100 and 150 hPa on days +1..+30, split at the median
               (telecon DESIGN.md 2026-08-17: the best predictor of the surface response, r -0.47 with the AO).
  split/displacement  NCEP R1 10 hPa height (MERRA-2 here is zonal-mean only): split if, on any day from -3 to +7,
               two separate low centres lie below the vortex edge (the DJFM climatological 10 hPa height at 60N) minus
               150 m, the smaller at least a quarter of the larger's area; else displacement. Calibrated on 20 events
               with an established type in the literature (CP07 1981-2002 plus the well-studied later ones): 20/20,
               but it was tuned on them. Moment diagnostics (aspect ratio, centroid latitude; Seviour et al. 2013)
               are kept as columns - the 2.4 aspect-ratio threshold got 17/20 on this 2.5-degree grid.

UNCONFIRMED REVERSALS: a CP07 event whose MERRA-2 wind stays above -0.5 m/s and that NCEP R1 never reverses is
listed but kept out of the composites and the odds (2002-02-17, 2025-11-28).

GAPS: 298 of the 16,979 days (1980-01-01..2026-06-26) are missing from our pull, scattered, the longest 9 days.
u(60N,10hPa) is filled from NCEP R1 (r 0.998, winter bias +0.75 m/s) plus the MERRA-2-minus-R1 offset over the
surrounding 30 days; everything else is interpolated in time. The filled days are flagged in the file.

    python scripts/strat/build_strat_history.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TD = REPO / "scripts" / "telecon" / "data"
M2 = TD / "m2_strat"
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily"
OUT_NC = HERE / "reference" / "strat_history.nc"
OUT_JS = HERE / "reference" / "strat_history_events.json"

LAGS = np.arange(-40, 91)
SPLIT_OFFSET = 150.0   # m below the edge: two low centres at this contour, the smaller >= 1/4 of the larger, days -3..+7
SURF_WIN = {"d1_30": (1, 30), "d31_60": (31, 60)}
SURF_LAT = (20.0, 90.0)
REGIONS = {   # name: (lat0, lat1, lon0, lon1) in degrees east; land points only
    "Eastern US": (32, 45, 265, 290), "Central Canada": (48, 60, 255, 275), "NW Europe": (47, 60, 350, 15),
    "Scandinavia": (58, 70, 5, 30), "Siberia": (50, 65, 70, 120), "East Asia": (30, 45, 105, 135)}


# ------------------------------------------------------------------ inputs
def load_m2() -> xr.Dataset:
    fs = sorted(glob.glob(str(M2 / "m2_strat_*.nc")))
    d = xr.open_mfdataset(fs, combine="by_coords")[["u60", "cap_z", "cap_T"]].load()
    full = pd.date_range(pd.Timestamp(d.time.values[0]), pd.Timestamp(d.time.values[-1]), freq="D")
    d = d.reindex(time=full)
    # QC: a day whose 100 hPa cap height sits >1 km from its 31-day median is a bad record (1991-06-30 is all
    # zeros); one such day in the sd fit made the seasonal standard deviation collapse and the anomalies explode
    z100 = d.cap_z.sel(lev=100.0).to_series()
    bad = ((z100 - z100.rolling(31, center=True, min_periods=5).median()).abs() > 1000).values
    if bad.any():
        print(f"  QC: {int(bad.sum())} bad day(s) set missing: {[f'{x:%Y-%m-%d}' for x in full[bad]]}")
        for v in ("u60", "cap_z", "cap_T"):
            d[v][dict(time=np.where(bad)[0])] = np.nan
    miss = np.isnan(d.u60.sel(lev=10.0).values)
    print(f"MERRA-2 {full[0]:%Y-%m-%d}..{full[-1]:%Y-%m-%d}: {miss.sum()} of {len(full)} days missing")
    # u10 from R1 + local offset
    r1 = xr.open_dataset(TD / "u60n10_r1.nc")["uwnd"].to_series()
    r1.index = pd.DatetimeIndex(r1.index).normalize()
    u10 = d.u60.sel(lev=10.0).to_series()
    off = (u10 - r1.reindex(u10.index)).rolling(31, center=True, min_periods=10).mean()
    fill = (r1.reindex(u10.index) + off.interpolate(limit_direction="both"))
    u10f = u10.where(u10.notna(), fill)
    x = np.arange(len(full))
    for v in ("u60", "cap_z", "cap_T"):                  # time interpolation, every level (gaps are <= 9 days)
        a = d[v].values.reshape(len(full), -1).astype("float64")
        bad = np.isnan(a).any(1)
        for j in range(a.shape[1]):
            a[bad, j] = np.interp(x[bad], x[~bad], a[~bad, j])
        d[v] = (d[v].dims, a.reshape(d[v].shape).astype("float32"))
    d["u10"] = ("time", u10f.values.astype("float32"))
    d["filled"] = ("time", miss.astype("int8"))
    left = int(np.isnan(d.u10.values).sum()) + int(np.isnan(d.cap_z.values).any(1).sum())
    print(f"  after filling: {left} NaN days")
    return d


def seasonal_design(t: pd.DatetimeIndex, nh: int, trend=True, step=False):
    x = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    cols = [np.ones(len(t))]
    for k in range(1, nh + 1):
        cols += [np.cos(k * x), np.sin(k * x)]
    if trend:
        cols.append((t.year.values + t.dayofyear.values / 365.25 - 2000) / 10)
    if step:
        cols.append((t >= "2004-08-01").astype(float))
    return np.column_stack(cols)


def standardise(y: np.ndarray, t: pd.DatetimeIndex, step=False):
    """Anomaly against harmonics + trend (+ MLS step), and the anomaly over a 3-harmonic sd."""
    X = seasonal_design(t, 4, True, step)
    ok = np.isfinite(y)
    b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    r = y - X @ b
    Xs = seasonal_design(t, 3, False, False)
    r2 = np.minimum(r ** 2, np.nanpercentile(r ** 2, 99.9))           # one wild day must not bend the sd curve
    bs = np.linalg.lstsq(Xs[ok], r2[ok], rcond=None)[0]
    sd = np.sqrt(np.clip(Xs @ bs, 1e-6 * np.nanvar(r), None))
    # floor at a third of the level's winter-peak sd: late-spring and summer spreads are tiny aloft, and dividing a
    # March event's day +70 by them drew +/-5 sigma noise at 1-10 hPa
    sd = np.maximum(sd, sd.max() / 3)
    return r, r / sd


def load_ao() -> pd.Series:
    rows = {}                                   # missing days are fused to the day field ("30-99.000"): regex, not split
    for line in open(TD / "cpc_daily_ao.txt"):
        m = re.match(r"\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s*(-?\d+\.\d+)", line)
        if m and float(m.group(4)) > -90:
            rows[pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    return pd.Series(rows).sort_index()


def load_qbo50() -> pd.Series:
    """CPC qbo.u50.index, ORIGINAL (m/s) block only - the file carries a standardised copy below it."""
    rows = {}
    for line in open(TD / "cpc_qbo_u50.txt"):
        if "STANDARDIZED" in line.upper():
            break
        m = re.match(r"^\s*(\d{4})(.*)$", line)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(2))]
        for k, x in enumerate(v[:12]):
            if x > -900:
                rows[pd.Timestamp(int(m.group(1)), k + 1, 1)] = x
    return pd.Series(rows).sort_index()


def load_roni() -> pd.Series:
    j = json.load(open(REPO / "assets" / "sst" / "data" / "nino_history.json"))
    return pd.Series(j["series"]["roni"]["anom"], index=pd.to_datetime([m + "-01" for m in j["months"]]))


def load_ssn() -> pd.Series:
    """Monthly sunspot number (SILSO v2 via SWPC). The same file's f10.7 column is -1 before Oct 2004."""
    s = pd.DataFrame(json.load(open(TD / "swpc_solar_cycle.json")))
    return pd.Series(s["ssn"].values, index=pd.to_datetime(s["time-tag"])).where(lambda x: x >= 0)


# ------------------------------------------------------------------ events
def cp07(u: pd.Series) -> list[pd.Timestamp]:
    ev, next_ok = [], None
    for y in range(u.index[0].year, u.index[-1].year):
        w = u[f"{y}-11-01":f"{y + 1}-03-31"]
        spring = u[f"{y}-11-01":f"{y + 1}-04-30"]
        if len(w) < 100:
            continue
        for day, val in w.items():
            if val >= 0 or (next_ok is not None and day < next_ok):
                continue
            after = spring[day:]
            west = (after > 0).astype(int).values
            run, ok = 0, False
            for x in west:                      # 10 consecutive westerly days completed by 30 April
                run = run + 1 if x else 0
                if run >= 10:
                    ok = True; break
            if not ok:
                break                           # a final warming: nothing later this winter can qualify either
            ev.append(day)
            # the next event needs 20 consecutive westerly days after this one
            tail = u[day:]
            run = 0
            for dd, x in tail.items():
                run = run + 1 if x > 0 else 0
                if run >= 20:
                    next_ok = dd; break
            else:
                next_ok = u.index[-1]
    return ev


def strong_vortex(nam10: pd.Series) -> list[pd.Timestamp]:
    ev, last = [], None
    prev = nam10.shift(1)
    for day in nam10.index[(nam10 >= 1.5) & (prev < 1.5)]:
        if day.month not in (11, 12, 1, 2, 3):
            continue
        if last is not None and (day - last).days < 60:
            continue
        ev.append(day); last = day
    return ev


# ------------------------------------------------------------------ vortex geometry (R1 z10)
def load_z10():
    fs = sorted(glob.glob(str(TD / "z10_r1_nh" / "z10_*.nc")))
    z = xr.open_mfdataset(fs, combine="by_coords").z10.load()
    return z


def geometry(z: xr.DataArray, edge: float):
    """Centroid latitude and aspect ratio of the region below `edge`, moments on an equal-area polar grid."""
    lat, lon = np.deg2rad(z.lat.values), np.deg2rad(z.lon.values)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    r = 2 * np.sin((np.pi / 2 - LA) / 2)
    X, Y = r * np.cos(LO), r * np.sin(LO)
    dA = np.cos(LA)
    keep = LA >= np.deg2rad(30)
    out = []
    for f in z.values:
        q = np.where(keep, np.clip(edge - f, 0, None), 0) * dA
        m0 = q.sum()
        if m0 <= 0:
            out.append((np.nan, np.nan)); continue
        xc, yc = (q * X).sum() / m0, (q * Y).sum() / m0
        J11 = (q * (X - xc) ** 2).sum() / m0
        J22 = (q * (Y - yc) ** 2).sum() / m0
        J12 = (q * (X - xc) * (Y - yc)).sum() / m0
        s = np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2)
        ar = np.sqrt((J11 + J22 + s) / max(J11 + J22 - s, 1e-12))
        rc = np.hypot(xc, yc)
        clat = 90 - np.rad2deg(2 * np.arcsin(min(rc / 2, 1)))
        out.append((clat, ar))
    return np.array(out)


def vortex_count(f: np.ndarray, lat: np.ndarray, level: float, frac: float = 0.25):
    """Separate low centres below `level` poleward of 30N (longitude wraps, the pole row is one point):
    the number whose area is at least `frac` of the largest's, and the second-to-first area ratio."""
    from scipy import ndimage
    lab, n = ndimage.label((f < level) & (lat[:, None] >= 30))
    par = list(range(n + 1))
    def find(a):
        while par[a] != a:
            a = par[a]
        return a
    for i in range(f.shape[0]):
        if lab[i, 0] and lab[i, -1]:
            par[find(lab[i, 0])] = find(lab[i, -1])
    pole = [x for x in np.unique(lab[np.argmax(lat)]) if x]
    for x in pole[1:]:
        par[find(x)] = find(pole[0])
    root = np.vectorize(lambda x: find(x) if x else 0)(lab)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones(f.shape[1])[None, :]
    A = sorted([area[root == r].sum() for r in np.unique(root) if r], reverse=True)
    if not A:
        return 0, 0.0
    return sum(1 for a in A if a >= frac * A[0]), (A[1] / A[0] if len(A) > 1 else 0.0)


# ------------------------------------------------------------------ ERA5 surface
def era5_anoms(var: str, lat0: float):
    """Daily anomalies 1979-2026 against per-gridpoint harmonics(3) + linear trend, lat >= lat0."""
    fs = sorted(ERA5.glob(f"{var}/{var}_*.nc"))
    fs = [f for f in fs if int(f.stem.split("_")[-1]) >= 1979]
    pieces = []
    for f in fs:
        d = xr.open_dataset(f)[var].transpose("time", "latitude", "longitude")
        d = d.sel(latitude=d.latitude >= lat0 - 0.01).astype("float32").load()
        if pieces:                                  # year files differ in the 5th decimal of their coordinates:
            ref = pieces[0]                         # a plain concat outer-joins them into a ragged 82 x 314 grid
            assert d.shape[1:] == ref.shape[1:] and np.allclose(d.latitude, ref.latitude, atol=0.01) \
                and np.allclose(d.longitude, ref.longitude, atol=0.01), f"{f.name}: grid differs"
            d = d.assign_coords(latitude=ref.latitude, longitude=ref.longitude)
        pieces.append(d)
    a = xr.concat(pieces, "time").sortby("time")
    a = a.isel(time=~pd.Index(a.time.values).duplicated())
    t = pd.DatetimeIndex(a.time.values)
    Y = a.values.reshape(len(t), -1)
    X = seasonal_design(t, 3, True, False)
    b = np.linalg.solve(X.T @ X, X.T @ Y)
    A = (Y - (X @ b).astype("float32")).reshape(a.shape)
    print(f"  ERA5 {var}: {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d} {a.shape}")
    return xr.DataArray(A, coords=a.coords, dims=a.dims)


def land_mask(lat, lon):
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    geoms = [g for g in shp.Reader(shp.natural_earth("110m", "physical", "land")).geometries()]
    land = prep(unary_union(geoms))
    m = np.zeros((len(lat), len(lon)), bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            m[i, j] = land.contains(Point(((lo + 180) % 360) - 180, la))
    return m


# ------------------------------------------------------------------ main
def main():
    d = load_m2()
    t = pd.DatetimeIndex(d.time.values)
    lev = d.lev.values
    capz = d.cap_z.values.astype(float)
    zs = np.full_like(capz, np.nan)
    for k, L in enumerate(lev):
        zs[:, k] = standardise(capz[:, k], t, step=L <= 5.0)[1]
    tanom10 = standardise(d.cap_T.sel(lev=10.0).values.astype(float), t)[0]
    u10 = pd.Series(d.u10.values.astype(float), index=t)
    nam10 = pd.Series(-zs[:, list(lev).index(10.0)], index=t)
    ssw = cp07(u10)
    sv = strong_vortex(nam10)
    print(f"SSWs: {len(ssw)}   strong-vortex events: {len(sv)}")

    old = {e["date"] for e in json.load(open(TD / "ssw_depth_merra2.json"))["events"]}
    new = {f"{x:%Y-%m-%d}" for x in ssw}
    print("  vs ssw_depth_merra2.json: only new", sorted(new - old), " only old", sorted(old - new))

    ao, qbo, roni, ssn = load_ao(), load_qbo50(), load_roni(), load_ssn()
    zi = {L: list(lev).index(L) for L in (10.0, 100.0, 150.0)}
    zsdf = pd.DataFrame(zs, index=t)

    def win_mean(s, c, a, b):
        return float(s[c + pd.Timedelta(days=a):c + pd.Timedelta(days=b)].mean())

    # vortex geometry from R1
    z10 = load_z10()
    zt = pd.DatetimeIndex(z10.time.values)
    djfm = z10.sel(time=zt.month.isin([12, 1, 2, 3]))
    edge = float(djfm.sel(lat=60.0).mean())
    print(f"  vortex edge (DJFM mean 10 hPa height at 60N, R1): {edge:.0f} m")

    events = []
    maps, map_dates = [], []
    for c in ssw:
        e = {"date": f"{c:%Y-%m-%d}", "winter": f"{c.year if c.month >= 7 else c.year - 1}/{str((c.year if c.month >= 7 else c.year - 1) + 1)[2:]}"}
        seg = u10[c:c + pd.Timedelta(days=30)]
        e["u_min"] = round(float(seg.min()), 1)
        e["easterly_days"] = int((u10[c:c + pd.Timedelta(days=60)] < 0).sum())
        spell = 0                                        # the reversal itself: consecutive easterly days from day 0
        for x in u10[c:c + pd.Timedelta(days=90)].values:
            if x >= 0:
                break
            spell += 1
        e["spell"] = spell
        e["T10_peak"] = round(float(pd.Series(tanom10, index=t)[c - pd.Timedelta(days=10):c + pd.Timedelta(days=10)].max()), 1)
        e["lsz"] = round(float(zsdf[[zi[100.0], zi[150.0]]][c + pd.Timedelta(days=1):c + pd.Timedelta(days=30)].mean().mean()), 2)
        e["ao_pre"] = round(win_mean(ao, c, -20, -1), 2)
        e["ao_post"] = round(win_mean(ao, c, 1, 45), 2)
        e["ao_d15_60"] = round(win_mean(ao, c, 15, 60), 2)
        e["filled_days_pm10"] = int(d.filled.sel(time=slice(c - pd.Timedelta(days=10), c + pd.Timedelta(days=10))).sum())
        w = z10.sel(time=slice(c - pd.Timedelta(days=3), c + pd.Timedelta(days=7)))
        g = geometry(w, edge)
        nv = [vortex_count(f, z10.lat.values, edge - SPLIT_OFFSET) for f in w.values]
        e["ar_max"] = round(float(np.nanmax(g[:, 1])), 2)
        e["clat_min"] = round(float(np.nanmin(g[:, 0])), 1)
        e["split_days"] = int(sum(1 for x in nv if x[0] >= 2))
        e["type"] = "split" if e["split_days"] >= 1 else "displacement"
        # the map: the most even split, or the most displaced day
        k = int(np.argmax([x[1] if x[0] >= 2 else -1 for x in nv])) if e["type"] == "split" else int(np.nanargmin(g[:, 0]))
        maps.append(w.values[k]); map_dates.append(pd.Timestamp(w.time.values[k]))
        e["map_date"] = f"{map_dates[-1]:%Y-%m-%d}"
        events.append(e)
    r1 = xr.open_dataset(TD / "u60n10_r1.nc")["uwnd"].to_series()
    r1.index = pd.DatetimeIndex(r1.index).normalize()
    for e in events:
        # a reversal under 0.5 m/s that a second reanalysis (NCEP R1) never shows is listed, not used:
        # 2002-02-17 (MERRA-2 -0.1, R1 +1.8) and 2025-11-28 (-0.04, R1 +1.3). R1 confirms the other 28.
        c = pd.Timestamp(e["date"])
        w = r1[c - pd.Timedelta(days=5):c + pd.Timedelta(days=10)]
        e["r1_min"] = round(float(w.min()), 1) if len(w) else None
        e["marginal"] = bool(e["u_min"] > -0.5 and len(w) and w.min() >= 0)
    med = float(np.median([e["lsz"] for e in events if not e["marginal"]]))
    for e in events:
        e["depth"] = "deep" if e["lsz"] >= med else "shallow"
    known = {"1985-01-01": "S", "1987-01-23": "D", "1987-12-07": "S", "1988-03-14": "S", "1989-02-21": "S",
             "1998-12-15": "D", "1999-02-26": "S", "2001-02-11": "D", "2002-02-17": "D", "1984-02-24": "D"}
    for e in events:
        tag = known.get(e["date"], "")
        print(f"  {e['date']} {e['type'][:5]:5s} (CP07 {tag or '-'})  AR {e['ar_max']:.2f} clat {e['clat_min']:.1f}  "
              f"u_min {e['u_min']:6.1f}  LSZ {e['lsz']:+.2f} {e['depth']:7s}  AO {e['ao_pre']:+.2f} -> {e['ao_post']:+.2f}")

    sv_events = []
    for c in sv:
        sv_events.append({"date": f"{c:%Y-%m-%d}", "nam10_peak": round(float(nam10[c:c + pd.Timedelta(days=30)].max()), 2),
                          "u_max": round(float(u10[c:c + pd.Timedelta(days=30)].max()), 1),
                          "lsz": round(float(zsdf[[zi[100.0], zi[150.0]]][c + pd.Timedelta(days=1):c + pd.Timedelta(days=30)].mean().mean()), 2),
                          "ao_pre": round(win_mean(ao, c, -20, -1), 2), "ao_post": round(win_mean(ao, c, 1, 45), 2)})

    # winters
    winters = []
    for y in range(1980, 2026):
        w0, w1 = pd.Timestamp(y, 11, 1), pd.Timestamp(y + 1, 3, 31)
        ev = [e for e in events if w0 <= pd.Timestamp(e["date"]) <= w1 and not e["marginal"]]
        djf = u10[f"{y}-12-01":f"{y + 1}-02-28"]
        r = roni.get(pd.Timestamp(y + 1, 1, 1), np.nan)
        q = float(qbo[f"{y}-10-01":f"{y}-11-01"].mean())
        f = float(ssn[f"{y}-10-01":f"{y + 1}-02-01"].mean())
        winters.append({"winter": f"{y}/{str(y + 1)[2:]}", "y": y, "roni_djf": round(float(r), 2),
                        "enso": "El Niño" if r >= 0.5 else ("La Niña" if r <= -0.5 else "neutral"),
                        "qbo50_on": round(q, 1), "qbo": "W" if q >= 0 else "E", "ssn": round(f, 0),
                        "n_ssw": len(ev), "ssw": [e["date"] for e in ev],
                        "n_sv": sum(1 for s in sv if w0 <= s <= w1),
                        "u10_djf": round(float(djf.mean()), 1), "u10_min": round(float(u10[w0:w1].min()), 1)})
    fmed = float(np.median([w["ssn"] for w in winters]))
    for w in winters:
        w["sun"] = "high" if w["ssn"] >= fmed else "low"

    # where 2026/27 sits, from the newest value of each index (a static note on the base-rate chart)
    rr, qq, ff = roni.dropna(), qbo.dropna(), ssn.dropna()
    rv, qv, fv = float(rr.iloc[-1]), float(qq.iloc[-1]), float(ff.iloc[-1])
    enso_now = "El Niño" if rv >= 0.5 else ("La Niña" if rv <= -0.5 else "neutral")
    current = {"as_of": f"{max(rr.index[-1], qq.index[-1]):%B %Y}", "enso": enso_now, "qbo": "W" if qv >= 0 else "E",
               "sun": "high" if fv >= fmed else "low",
               "text": (f"{enso_now} (RONI {rv:+.1f}, {rr.index[-1]:%b}), QBO {'westerly' if qv >= 0 else 'easterly'} at 50 hPa "
                        f"({qv:+.1f} m/s, {qq.index[-1]:%b}), sunspot number {fv:.0f} ({ff.index[-1]:%b}) against the winter median of {fmed:.0f}.")}
    print("  current:", current["text"])

    # ---------------------------------------------------------- composites
    used = [e for e in events if not e["marginal"]]
    sets = {"ssw_all": [pd.Timestamp(e["date"]) for e in used],
            "ssw_deep": [pd.Timestamp(e["date"]) for e in used if e["depth"] == "deep"],
            "ssw_shallow": [pd.Timestamp(e["date"]) for e in used if e["depth"] == "shallow"],
            "ssw_split": [pd.Timestamp(e["date"]) for e in used if e["type"] == "split"],
            "ssw_disp": [pd.Timestamp(e["date"]) for e in used if e["type"] == "displacement"],
            "sv": list(sv)}
    SETS = list(sets)

    def stack(series_2d: np.ndarray, idx: pd.DatetimeIndex, dates):
        out = []
        pos = {x: i for i, x in enumerate(idx)}
        for c in dates:
            i = pos.get(c)
            rows = []
            for L in LAGS:
                j = None if i is None else i + L
                rows.append(series_2d[j] if (j is not None and 0 <= j < len(idx)) else np.full(series_2d.shape[1:], np.nan))
            out.append(np.stack(rows))
        return np.stack(out)

    def mean_t(a):
        n = np.isfinite(a).sum(0)
        m = np.nanmean(a, 0)
        sd = np.nanstd(a, 0, ddof=1)
        return m, m / (sd / np.sqrt(np.maximum(n, 1)))

    drip_m = np.zeros((len(SETS), len(LAGS), len(lev)), "float32"); drip_t = np.zeros_like(drip_m)
    for k, s in enumerate(SETS):
        drip_m[k], drip_t[k] = mean_t(stack(zs, t, sets[s]))

    ao_all = ao.reindex(pd.date_range("1979-01-01", "2026-06-30"))
    ao_ev = {s: stack(ao_all.values[:, None], pd.DatetimeIndex(ao_all.index), sets[s])[..., 0] for s in ("ssw_all", "sv")}
    # baseline: the same calendar days in every OTHER year, averaged
    base = {}
    for s in ("ssw_all", "sv"):
        rows = []
        for c in sets[s]:
            ys = [y for y in range(1980, 2026) if y != c.year]
            rows.append(np.nanmean([ao_all.reindex(pd.DatetimeIndex([c.replace(year=y) + pd.Timedelta(days=int(L)) for L in LAGS])).values
                                    for y in ys if not (c.month == 2 and c.day == 29)], 0))
        base[s] = np.nanmean(rows, 0)

    # surface
    print("ERA5 anomalies ...")
    T = era5_anoms("t2m", SURF_LAT[0])
    Z = era5_anoms("z500", SURF_LAT[0])
    lat, lon = T.latitude.values, T.longitude.values
    tt = pd.DatetimeIndex(T.time.values)
    lm = land_mask(lat, lon)
    surf = {}
    reg = {}
    for s in SETS:
        for wk, (a, b) in SURF_WIN.items():
            tw = np.stack([T.sel(time=slice(c + pd.Timedelta(days=a), c + pd.Timedelta(days=b))).mean("time").values for c in sets[s]])
            zw = np.stack([Z.sel(time=slice(c + pd.Timedelta(days=a), c + pd.Timedelta(days=b))).mean("time").values for c in sets[s]])
            m, tstat = mean_t(tw)
            surf[(s, wk)] = (m, tstat, np.nanmean(zw, 0))
            for rn, (la0, la1, lo0, lo1) in REGIONS.items():
                msk = lm & (lat[:, None] >= la0) & (lat[:, None] <= la1) & (
                    ((lon[None, :] >= lo0) & (lon[None, :] <= lo1)) if lo0 < lo1 else ((lon[None, :] >= lo0) | (lon[None, :] <= lo1)))
                wgt = np.cos(np.deg2rad(lat))[:, None] * msk
                vals = [float((x * wgt).sum() / wgt.sum()) for x in tw]
                reg[(s, wk, rn)] = vals
    # every calendar-matched random window, for the "how often colder than normal anyway" reference
    base_reg = {}
    for rn, (la0, la1, lo0, lo1) in REGIONS.items():
        msk = lm & (lat[:, None] >= la0) & (lat[:, None] <= la1) & (
            ((lon[None, :] >= lo0) & (lon[None, :] <= lo1)) if lo0 < lo1 else ((lon[None, :] >= lo0) | (lon[None, :] <= lo1)))
        wgt = (np.cos(np.deg2rad(lat))[:, None] * msk).astype("float32")
        daily = pd.Series(np.einsum("tij,ij->t", T.values, wgt) / wgt.sum(), index=tt)
        for wk, (a, b) in SURF_WIN.items():
            # mean over days c+a..c+b for every start day c in Dec-Mar 1980-2026
            ser = daily.rolling(b - a + 1).mean().shift(-b)
            pool = ser[(ser.index.month.isin([12, 1, 2, 3])) & (ser.index.year >= 1980)].dropna()
            base_reg[(wk, rn)] = float((pool < 0).mean())

    # ---------------------------------------------------------- write
    win = xr.Dataset(
        {"u10": ("time", d.u10.values.astype("float32")),
         "t10": ("time", d.cap_T.sel(lev=10.0).values.astype("float32")),
         "z100s": ("time", zs[:, zi[100.0]].astype("float32")),
         "nam10": ("time", nam10.values.astype("float32")),
         "filled": ("time", d.filled.values),
         "drip_mean": (("set", "lag", "lev"), drip_m), "drip_t": (("set", "lag", "lev"), drip_t),
         "ao_ssw": (("ssw_used", "lag"), ao_ev["ssw_all"].astype("float32")), "ao_sv": (("sv", "lag"), ao_ev["sv"].astype("float32")),
         "ao_base_ssw": ("lag", base["ssw_all"].astype("float32")), "ao_base_sv": ("lag", base["sv"].astype("float32")),
         "t2m_mean": (("set", "window", "lat", "lon"), np.stack([[surf[(s, w)][0] for w in SURF_WIN] for s in SETS]).astype("float32")),
         "t2m_t": (("set", "window", "lat", "lon"), np.stack([[surf[(s, w)][1] for w in SURF_WIN] for s in SETS]).astype("float32")),
         "z500_mean": (("set", "window", "lat", "lon"), np.stack([[surf[(s, w)][2] for w in SURF_WIN] for s in SETS]).astype("float32")),
         "z10_map": (("ssw", "mlat", "mlon"), np.stack(maps).astype("float32"))},
        coords={"time": t, "set": SETS, "lag": LAGS, "lev": lev, "window": list(SURF_WIN), "lat": lat, "lon": lon,
                "ssw": [e["date"] for e in events], "ssw_used": ("ssw_used", [e["date"] for e in used]), "sv": [s["date"] for s in sv_events],
                "mlat": z10.lat.values, "mlon": z10.lon.values})
    win.attrs.update(source="MERRA-2 zonal means (u60, 65-90N cap T and z), NCEP R1 10 hPa height maps, ERA5 t2m/z500 "
                            "(WeatherBench2 1.5 deg), CPC daily AO; built by scripts/strat/build_strat_history.py",
                     vortex_edge_m=edge, lsz_median=med, ssn_median=fmed)
    enc = {v: {"zlib": True, "complevel": 5} for v in win.data_vars}
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    win.to_netcdf(OUT_NC, encoding=enc)
    js = {"built": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "period": f"{t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}",
          "sets": {s: len(v) for s, v in sets.items()}, "vortex_edge_m": round(edge), "lsz_median": round(med, 2),
          "ssn_median": fmed, "current": current, "ssw": events, "strong_vortex": sv_events, "winters": winters,
          "regions": {rn: list(v) for rn, v in REGIONS.items()},
          "region_values": {f"{s}|{w}|{rn}": [round(x, 2) for x in v] for (s, w, rn), v in reg.items()},
          "region_base_pcold": {f"{w}|{rn}": round(v, 3) for (w, rn), v in base_reg.items()}}
    json.dump(js, open(OUT_JS, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT_NC.relative_to(REPO)} ({OUT_NC.stat().st_size / 1e6:.2f} MB) and {OUT_JS.relative_to(REPO)} "
          f"({OUT_JS.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
