#!/usr/bin/env python3
"""Shared pieces of the atmospheric-river products (ar_det.py, ar_ens.py, ar_ext.py, ar_validate.py).

IVT, the AR variable (Ralph et al. 2004; Rutz et al. 2014; the AR scale of Ralph et al. 2019):

    IVT = (1/g) | ∫ q V dp |,   1000 hPa → 300 hPa, trapezoidal in p over the levels each model publishes

Surface masking (`integrate`): a layer between two levels counts only above the ground. A level below the surface
(p > surface pressure) is dropped; the layer that straddles the ground is integrated from the surface pressure to the
level above it, using that upper level's q·V (the below-ground value in the GRIB is an extrapolation, not air). Over
the ocean this changes nothing (ps ≥ 1000 hPa there, and the integral is capped at 1000 hPa for every model so that
the vertical range is identical); over the Sierra Nevada, Cascades and Rockies it removes the fictitious below-ground
transport that otherwise inflates IVT by 20-60 %.

Sources (hard rules): ECMWF open data from the Google Cloud mirror ONLY (scripts/ecmwf/rangefetch.py refuses any other
host); NOAA from the AWS S3 open-data buckets, byte-ranged through each file's .idx; ECCC from the MSC Datamart.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS / "ecmwf"))
sys.path.insert(0, str(SCRIPTS / "lib"))

G0 = 9.80665
UA = "scorvec.com data pipeline (+https://scorvec.com)"
THR = (250, 500, 750, 1000, 1250)  # Ralph et al. (2019) intensity steps, kg m-1 s-1
# the analysis box every model is cropped to: the North Pacific, North America, the Gulf and the western Atlantic
CROP = dict(lat=(5.0, 72.0), lon=(120.0, 335.0))
DOMAINS = {
    "pac": dict(label="North Pacific and the West Coast", extent=(130.0, 250.0, 15.0, 65.0), central=190.0, sk=2.5),
    "na": dict(label="North America and the Gulf", extent=(225.0, 300.0, 15.0, 55.0), central=262.5, sk=1.75),
}
ENS_DOMAIN = dict(label="North Pacific and North America", extent=(130.0, 300.0, 12.0, 66.0), central=215.0, sk=3.0)
COAST = json.loads((HERE / "data" / "coast_transect.json").read_text())
# the AR-scale points: every 2.5 deg of latitude from 30 to 55 N on the coastal transect, labelled with the nearest place
POINT_NAMES = {
    30.0: "Baja California (El Rosario)",
    32.5: "San Diego",
    35.0: "Point Conception / San Luis Obispo",
    37.5: "San Francisco Bay / Half Moon Bay",
    40.0: "Cape Mendocino",
    42.5: "Oregon south coast (Port Orford)",
    45.0: "Oregon central coast (Lincoln City)",
    47.5: "Olympic Peninsula",
    50.0: "Vancouver Island (north)",
    52.5: "Haida Gwaii / central BC coast",
    55.0: "Southeast Alaska (Ketchikan)",
}


def coast_points():
    """[(lat, point_lon_east, name)] for the 2.5-deg AR-scale points, read off the transect (first ocean point west of
    the coast on each 0.25-deg latitude)."""
    tr = {round(r["lat"], 2): r for r in COAST["transect"]}
    out = []
    for la, nm in POINT_NAMES.items():
        r = tr[round(la, 2)]
        out.append((la, r["point_lon"] % 360.0, nm))
    return out


def transect():
    """(lat array, point_lon east array) of the whole 0.25-deg coastal transect (24-58 N)."""
    tr = COAST["transect"]
    return np.array([r["lat"] for r in tr]), np.array([r["point_lon"] % 360.0 for r in tr])


# ── GRIB decoding ─────────────────────────────────────────────────────────────────────────────────────────────────
def split_grib(blob: bytes) -> list[bytes]:
    """Concatenated GRIB2 (or GRIB1) messages -> list of single messages."""
    out, o, n = [], 0, len(blob)
    while o < n:
        k = blob.find(b"GRIB", o)
        if k < 0:
            break
        ed = blob[k + 7]
        ln = int.from_bytes(blob[k + 8 : k + 16], "big") if ed == 2 else int.from_bytes(blob[k + 4 : k + 7], "big")
        out.append(blob[k : k + ln])
        o = k + ln
    return out


_GRID_CACHE: dict = {}
_GRID_LOCK = threading.Lock()


def _grid(ni, nj, la0, la1, lo0, di, crop):
    key = (ni, nj, round(la0, 4), round(la1, 4), round(lo0, 4), round(di, 5), tuple(crop["lat"]), tuple(crop["lon"]))
    with _GRID_LOCK:
        if key in _GRID_CACHE:
            return _GRID_CACHE[key]
    lat = np.linspace(la0, la1, nj)
    lon = (lo0 + di * np.arange(ni)) % 360.0
    jo = np.argsort(lat)
    lat_s = lat[jo]
    io = np.argsort(lon)
    lon_s = lon[io]
    jj = jo[(lat_s >= crop["lat"][0] - 1e-6) & (lat_s <= crop["lat"][1] + 1e-6)]
    ii = io[(lon_s >= crop["lon"][0] - 1e-6) & (lon_s <= crop["lon"][1] + 1e-6)]
    g = (jj, ii, lat[jj], lon[ii])
    with _GRID_LOCK:
        _GRID_CACHE[key] = g
    return g


def decode(msg: bytes, crop=CROP):
    """One GRIB message -> (meta dict, values (lat asc, lon asc 0-360) cropped, lat, lon)."""
    import eccodes as ec

    g = ec.codes_new_from_message(msg)
    try:
        meta = {k: ec.codes_get(g, k) for k in ("shortName", "typeOfLevel", "level", "Ni", "Nj")}
        for k in ("perturbationNumber", "forecastTime", "stepRange"):
            try:
                meta[k] = ec.codes_get(g, k)
            except Exception:  # noqa: BLE001
                meta[k] = None
        la0 = ec.codes_get(g, "latitudeOfFirstGridPointInDegrees")
        la1 = ec.codes_get(g, "latitudeOfLastGridPointInDegrees")
        lo0 = ec.codes_get(g, "longitudeOfFirstGridPointInDegrees")
        di = ec.codes_get(g, "iDirectionIncrementInDegrees")
        ni, nj = meta["Ni"], meta["Nj"]
        v = ec.codes_get_values(g).reshape(nj, ni)
        try:
            if ec.codes_get(g, "bitmapPresent"):
                miss = ec.codes_get(g, "missingValue")
                v = np.where(v == miss, np.nan, v)
        except Exception:  # noqa: BLE001
            pass
    finally:
        ec.codes_release(g)
    jj, ii, lat, lon = _grid(ni, nj, la0, la1, lo0, di, crop)
    return meta, v[np.ix_(jj, ii)].astype("float32"), lat, lon


# ── physics ───────────────────────────────────────────────────────────────────────────────────────────────────────
def integrate(levels_hpa, q, u, v, ps=None, top=300.0, bottom=1000.0):
    """IVT east/north components (kg m-1 s-1) from q (kg/kg), u, v (m/s) on pressure levels.

    levels_hpa: 1-D, any order; q/u/v: (lev, ...) matching it. ps: surface pressure (Pa) or None (no masking).
    Layers are trapezoids between adjacent levels within [top, bottom] hPa; see the module docstring for the
    below-ground rule."""
    lv = np.asarray(levels_hpa, float)
    o = np.argsort(-lv)
    lv = lv[o]
    keep = (lv <= bottom + 1e-6) & (lv >= top - 1e-6)
    lv = lv[keep]
    q = np.asarray(q)[o][keep]
    u = np.asarray(u)[o][keep]
    v = np.asarray(v)[o][keep]
    p = lv * 100.0
    fu = q * u
    fv = q * v
    ivte = np.zeros(q.shape[1:], "float64")
    ivtn = np.zeros(q.shape[1:], "float64")
    psb = None if ps is None else np.minimum(np.asarray(ps, float), bottom * 100.0)
    for k in range(len(p) - 1):
        pb, pt = p[k], p[k + 1]
        if psb is None:
            w = pb - pt
            ivte += 0.5 * (fu[k] + fu[k + 1]) * w
            ivtn += 0.5 * (fv[k] + fv[k + 1]) * w
            continue
        w = np.clip(np.minimum(pb, psb) - pt, 0.0, None)
        below = psb < pb  # the lower level is under the ground here
        fe = np.where(below, fu[k + 1], fu[k])
        fn = np.where(below, fv[k + 1], fv[k])
        ivte += 0.5 * (fe + fu[k + 1]) * w
        ivtn += 0.5 * (fn + fv[k + 1]) * w
    return (ivte / G0).astype("float32"), (ivtn / G0).astype("float32")


def q_from_rh(rh, t, p_hpa):
    """Specific humidity (kg/kg) from relative humidity (%), temperature (K) and pressure (hPa). Saturation vapour
    pressure over water (Bolton 1980) at every temperature: the NCEP/ECCC RH fields' convention differs below 0 C, which
    matters only in the cold upper levels that carry little vapour."""
    tc = np.asarray(t, float) - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    e = np.clip(np.asarray(rh, float), 0, 100) / 100.0 * es
    p = np.asarray(p_hpa, float)
    return (0.622 * e / np.maximum(p - 0.378 * e, 1.0)).astype("float32")


def ar_rank(series, dt_h: float = 6.0):
    """Ralph et al. (2019) AR category from an IVT time series at a point: preliminary rank from the spell's peak IVT
    (250/500/750/1000/1250 -> 1..5), one lower if the spell (IVT >= 250) lasts under 24 h, one higher if 48 h or longer,
    capped at 5. Returns (rank, peak, duration h, first index) of the strongest spell; rank 0 = no AR.
    Duration = number of 6-hourly samples at/above 250 times 6 h."""
    s = np.asarray(series, float)
    best = (0, 0.0, 0.0, None)
    i, n = 0, len(s)
    while i < n:
        if s[i] >= 250:
            j = i
            while j + 1 < n and s[j + 1] >= 250:
                j += 1
            peak = float(s[i : j + 1].max())
            dur = dt_h * (j - i + 1)
            r = int(np.searchsorted(THR, peak, side="right"))
            if dur < 24:
                r -= 1
            elif dur >= 48:
                r += 1
            r = int(np.clip(r, 0, 5))
            if (r, peak) > (best[0], best[1]):
                best = (r, peak, dur, i)
            i = j + 1
        else:
            i += 1
    return best


def daily_ranks(series, valid, dt_h=6.0):
    """Per UTC day: the Ralph rank of the strongest AR spell that is under way during that day (a spell is rated over its
    whole length within the forecast, then credited to every day it touches). Returns {YYYY-MM-DD: rank}."""
    import pandas as pd

    s = np.asarray(series, float)
    days = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in valid]
    out = dict.fromkeys(days, 0)
    i, n = 0, len(s)
    while i < n:
        if s[i] >= 250:
            j = i
            while j + 1 < n and s[j + 1] >= 250:
                j += 1
            peak = float(s[i : j + 1].max())
            dur = dt_h * (j - i + 1)
            r = int(np.searchsorted(THR, peak, side="right"))
            r = int(np.clip(r - 1 if dur < 24 else (r + 1 if dur >= 48 else r), 0, 5))
            for k in range(i, j + 1):
                out[days[k]] = max(out[days[k]], r)
            i = j + 1
        else:
            i += 1
    return out


# ── HTTP ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class Ledger:
    """Bytes and requests per source, for the cost report."""

    def __init__(self):
        self.b = {}
        self.n = {}
        self.lock = threading.Lock()

    def add(self, src, nbytes):
        with self.lock:
            self.b[src] = self.b.get(src, 0) + nbytes
            self.n[src] = self.n.get(src, 0) + 1

    def report(self):
        return {k: {"MB": round(self.b[k] / 1e6, 1), "requests": self.n[k]} for k in sorted(self.b)}


LEDGER = Ledger()


def http_get(url, rng=None, tries=4, timeout=90, src="http", backoff=3.0):
    import requests

    h = {"User-Agent": UA}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{'' if rng[1] is None or rng[1] < 0 else rng[1]}"
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=(20, timeout))
            if r.status_code in (200, 206):
                LEDGER.add(src, len(r.content))
                return r.content
            if r.status_code == 404:
                return None
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 + backoff * k)
    raise RuntimeError(f"{url}: {last}")


def idx_records(url):
    """NOAA .idx -> [(start, end_or_None, var, level, step text)]."""
    txt = http_get(url + ".idx", src="aws-s3")
    if txt is None:
        return None
    lines = [l.split(":") for l in txt.decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        out.append((int(p[1]), end, p[3], p[4], p[5]))
    return out


# ── deterministic model readers: each returns, per step, {"q","u","v": (lev, lat, lon), "ps", "msl"} on its grid ──
ECMWF_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300)
GFS_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300)
GDPS_LEVELS = (1000, 985, 970, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300)


def _assemble(recs, levels, qkey="q", ukey="u", vkey="v"):
    lat = lon = None
    f = {}
    for (name, lev), (vals, la, lo) in recs.items():
        f[(name, lev)] = vals
        lat, lon = la, lo
    out = {"lat": lat, "lon": lon, "levels": list(levels)}
    for key, nm in (("q", qkey), ("u", ukey), ("v", vkey)):
        out[key] = np.stack([f[(nm, L)] for L in levels])
    out["ps"] = f.get(("ps", 0))
    out["msl"] = f.get(("msl", 0))
    return out


def ecmwf_step(date, hh, model, step, levels=ECMWF_LEVELS, crop=CROP):
    """IFS HRES ("ifs", stream oper) or AIFS single ("aifs-single") from the Google mirror: q/u/v on `levels`, surface
    pressure and MSLP for one step. None if the step is not on the mirror (the cycle stays partial)."""
    import rangefetch as rf

    try:
        idx = rf.fetch_index(date, hh, model, int(step), "fc", stream="oper")
    except rf.IndexUnavailable as e:
        if e.not_found:
            return None
        raise
    want = rf.select(idx, param=["q", "u", "v"], levelist=list(levels)) + rf.select(idx, param=["sp", "msl"])
    want = [
        e for e in want if (e.get("levtype") == "pl") or (e.get("param") in ("sp", "msl") and e.get("levtype") == "sfc")
    ]
    ranges = rf.coalesce(want, max_gap=256 * 1024)
    stats = {}
    blob = rf.fetch_ranges(rf.path_for(date, hh, model, int(step), "fc", stream="oper") + ".grib2", ranges, stats=stats)
    LEDGER.add("ecmwf-google", len(blob))
    recs = {}
    for m in split_grib(blob):
        meta, v, la, lo = decode(m, crop)
        sn = meta["shortName"]
        if meta["typeOfLevel"] == "isobaricInhPa":
            recs[(sn, int(meta["level"]))] = (v, la, lo)
        elif sn in ("sp", "msl"):
            recs[("ps" if sn == "sp" else "msl", 0)] = (v, la, lo)
    d = _assemble(recs, levels)
    d["msl"] = None if d["msl"] is None else d["msl"] / 100.0
    return d


GFS_S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def gfs_step(date, hh, step, levels=GFS_LEVELS, crop=CROP, workers=12):
    """GFS 0.25 deg pgrb2 from AWS S3 (noaa-gfs-bdp-pds), byte-ranged per record through the .idx."""
    url = f"{GFS_S3}/gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{int(step):03d}"
    recs_idx = idx_records(url)
    if recs_idx is None:
        return None
    lev_txt = {f"{L} mb": L for L in levels}
    want = []
    for s, e, var, lev, _ in recs_idx:
        if var in ("SPFH", "UGRD", "VGRD") and lev in lev_txt:
            want.append((s, e, {"SPFH": "q", "UGRD": "u", "VGRD": "v"}[var], lev_txt[lev]))
        elif var == "PRES" and lev == "surface":
            want.append((s, e, "ps", 0))
        elif var == "PRMSL" and lev == "mean sea level":
            want.append((s, e, "msl", 0))
    need = 3 * len(levels) + 2
    if len(want) < need:
        raise RuntimeError(f"GFS f{step:03d}: {len(want)}/{need} records in the idx")

    def one(w):
        s, e, nm, L = w
        b = http_get(url, (s, e), src="aws-s3")
        if b is None or not b.startswith(b"GRIB"):
            raise RuntimeError(f"GFS f{step:03d} {nm}{L}: bad range")
        _meta, v, la, lo = decode(b, crop)
        return (nm, L), (v, la, lo)

    with ThreadPoolExecutor(workers) as ex:
        recs = dict(ex.map(one, want))
    d = _assemble(recs, levels)
    d["msl"] = d["msl"] / 100.0
    return d


DATAMART = "https://dd.weather.gc.ca"
GDPS_VARS = {
    "q": "SpecificHumidity_IsbL-{L:04d}",
    "u": "WindU_IsbL-{L:04d}",
    "v": "WindV_IsbL-{L:04d}",
    "ps": "Pressure_Sfc",
    "msl": "Pressure_MSL",
}


def gdps_url(date, hh, step, var):
    return (
        f"{DATAMART}/{date}/WXO-DD/model_gdps/15km/{hh}/{int(step):03d}/"
        f"{date}T{hh}Z_MSC_GDPS_{var}_LatLon0.15_PT{int(step):03d}H.grib2"
    )


def gdps_step(date, hh, step, levels=GDPS_LEVELS, crop=CROP, workers=8):
    """ECCC GDPS 15 km lat-lon from the MSC Datamart: one file per variable, level and step."""
    jobs = [(k, L, GDPS_VARS[k].format(L=L)) for k in ("q", "u", "v") for L in levels] + [
        ("ps", 0, GDPS_VARS["ps"]),
        ("msl", 0, GDPS_VARS["msl"]),
    ]

    def one(j):
        k, L, var = j
        # the Datamart drops connections under load (connect timeouts seen 2026-10-07): more and longer retries
        b = http_get(gdps_url(date, hh, step, var), src="datamart", timeout=120, tries=7, backoff=10.0)
        if b is None:
            return (k, L), None
        _meta, v, la, lo = decode(b, crop)
        return (k, L), (v, la, lo)

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, jobs))
    if any(v is None for v in res.values()):
        miss = [f"{k}{L}" for (k, L), v in res.items() if v is None]
        print(f"  GDPS {date} {hh}Z +{step}h: missing {len(miss)} files ({', '.join(miss[:6])})", flush=True)
        return None
    d = _assemble(res, levels)
    if np.nanmax(d["q"]) > 0.5:  # sanity: kg/kg (a g/kg field would read ~20)
        raise RuntimeError("GDPS specific humidity is not in kg/kg")
    # Datamart GDPS surface pressure and MSLP are in Pa? check the magnitude rather than trusting it
    if d["ps"] is not None and np.nanmedian(d["ps"]) < 2000:
        d["ps"] = d["ps"] * 100.0
    d["msl"] = d["msl"] / 100.0 if np.nanmedian(d["msl"]) > 2000 else d["msl"]
    return d


MODELS = {
    "aifs": dict(
        label="ECMWF AIFS single",
        short="AIFS",
        levels=ECMWF_LEVELS,
        res="0.25°",
        src="ECMWF open data (Google Cloud mirror), aifs-single oper",
    ),
    "ifs": dict(
        label="ECMWF IFS HRES",
        short="IFS",
        levels=ECMWF_LEVELS,
        res="0.25°",
        src="ECMWF open data (Google Cloud mirror), ifs oper",
    ),
    "gfs": dict(label="NCEP GFS", short="GFS", levels=GFS_LEVELS, res="0.25°", src="NOAA GFS pgrb2 0.25° (AWS S3)"),
    "gdps": dict(
        label="ECCC GDPS", short="GDPS", levels=GDPS_LEVELS, res="0.15°", src="ECCC GDPS 15 km lat-lon (MSC Datamart)"
    ),
}


def read_step(model, date, hh, step):
    if model == "aifs":
        return ecmwf_step(date, hh, "aifs-single", step)
    if model == "ifs":
        return ecmwf_step(date, hh, "ifs", step)
    if model == "gfs":
        return gfs_step(date, hh, step)
    if model == "gdps":
        return gdps_step(date, hh, step)
    raise ValueError(model)


def ivt_of(d, levels=None):
    """(ivt, ivte, ivtn) from a read_step dict (optionally on a subset of its levels)."""
    lv = d["levels"]
    sel = list(range(len(lv))) if levels is None else [lv.index(L) for L in levels]
    e, n = integrate([lv[i] for i in sel], d["q"][sel], d["u"][sel], d["v"][sel], d["ps"])
    return np.hypot(e, n), e, n


def nearest_idx(axis, vals):
    axis = np.asarray(axis)
    return np.array([int(np.argmin(np.abs(axis - v))) for v in np.atleast_1d(vals)])


# ── map drawing ───────────────────────────────────────────────────────────────────────────────────────────────────
# IVT palette: two greys for sub-AR transport (100-250), then the AR range in bands that split each Ralph-scale step
# (250/500/750/1000/1250) in two: yellow -> orange -> red -> magenta -> purple.
IVT_LEV = [100, 175, 250, 375, 500, 625, 750, 875, 1000, 1250, 1500]
IVT_COL = [
    "#e4e4df",
    "#c9cbc4",
    "#fde68a",
    "#fbbf24",
    "#fb923c",
    "#ea580c",
    "#dc2626",
    "#a51c30",
    "#c0267e",
    "#7e22ce",
    "#3b0a5e",
]
INK, MUTED = "#1a1a1a", "#6f6b64"


def ivt_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cm = ListedColormap(IVT_COL[:-1])
    cm.set_over(IVT_COL[-1])
    return cm, BoundaryNorm(IVT_LEV, cm.N)


def map_axes(dom, width=12.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    w, e, s, n = dom["extent"]
    map_h = width * 0.94 * (n - s) / (e - w)
    top, bot = 0.62, 0.78
    H = map_h + top + bot
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=ccrs.PlateCarree(central_longitude=dom["central"]))
    ax.set_extent([w, e, s, n], crs=ccrs.PlateCarree())
    return fig, ax, H


def map_features(ax):
    import cartopy.feature as cfeature

    ax.coastlines(resolution="50m", linewidth=0.6, color="#111", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="#333", zorder=6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#555", zorder=6)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="#6f6b64",
        alpha=0.45,
        xlocs=range(-180, 181, 10),
        ylocs=range(0, 91, 10),
        zorder=5,
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#444"}


def heading(fig, H, title, sub):
    fig.text(0.03, 1 - 0.12 / H, title, fontsize=12.5, fontweight="bold", va="top", color=INK)
    fig.text(0.03, 1 - 0.40 / H, sub, fontsize=8.6, color="#444", va="top")


def ivt_frame(out, lat, lon, ivt, ivte, ivtn, dom, title, sub, msl=None, cb_label=None, prob=None):
    """One IVT map: shaded magnitude, vectors where IVT >= 250, optional MSLP contours (every 4 hPa)."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    fig, ax, H = map_axes(dom)
    cm, nm = ivt_cmap()
    w, e, s, n = dom["extent"]
    jj = (lat >= s - 1) & (lat <= n + 1)
    ii = (lon >= w - 1) & (lon <= e + 1)
    la, lo = lat[jj], lon[ii]

    def sub_(a):
        return a[np.ix_(jj, ii)]

    mag = sub_(ivt)
    pm = ax.pcolormesh(
        lo,
        la,
        np.where(mag >= IVT_LEV[0], mag, np.nan),
        cmap=cm,
        norm=nm,
        transform=pc,
        shading="nearest",
        rasterized=True,
        zorder=2,
    )
    if msl is not None:
        m = sub_(msl)
        stride = max(1, int(np.rint(0.5 / abs(lo[1] - lo[0]))))
        cs = ax.contour(
            lo[::stride],
            la[::stride],
            m[::stride, ::stride],
            levels=np.arange(940, 1061, 4),
            colors="#2b2b2b",
            linewidths=0.55,
            alpha=0.75,
            transform=pc,
            zorder=4,
        )
        ax.clabel(cs, fmt="%d", fontsize=6.5, inline=True)
    sk = dom["sk"]
    d = abs(lo[1] - lo[0])
    st = max(1, int(np.rint(sk / d)))
    ue, un, mg = sub_(ivte)[::st, ::st], sub_(ivtn)[::st, ::st], mag[::st, ::st]
    ok = mg >= 250
    # arrow length capped at 1000 (direction kept) so tropical cyclones, 2000-4000, do not cover the map
    cap = np.where(mg > 1000, 1000.0 / np.maximum(mg, 1e-6), 1.0)
    ue, un = ue * cap, un * cap
    if ok.any():
        X, Y = np.meshgrid(lo[::st], la[::st])
        ax.quiver(
            X[ok],
            Y[ok],
            ue[ok],
            un[ok],
            transform=pc,
            scale=26000,
            width=0.0018,
            headwidth=3.6,
            color="#111",
            alpha=0.9,
            zorder=5,
        )
    if prob is not None:
        ax.contour(lo, la, sub_(prob), levels=[0.5], colors=["#1b365d"], linewidths=1.0, transform=pc, zorder=5)
    map_features(ax)
    heading(fig, H, title, sub)
    cax = fig.add_axes([0.2, 0.36 / H, 0.6, 0.13 / H])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="max", ticks=IVT_LEV)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(
        cb_label
        or "IVT (kg m⁻¹ s⁻¹): AR threshold 250 · Ralph-scale steps 250 / 500 / 750 / 1000 / 1250 · arrows: IVT vector where ≥ 250 (length capped at 1000)",
        fontsize=8,
    )
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
    plt.close(fig)


def prob_frame(out, lat, lon, prob, dom, title, sub, contour=None, contour_label=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    fig, ax, H = map_axes(dom)
    lev = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cols = [
        "#eef4fa",
        "#d4e4f2",
        "#a9cbe6",
        "#7fb0d8",
        "#5793c4",
        "#3a76ad",
        "#f2c14e",
        "#ec8f2f",
        "#d9542b",
        "#a51c30",
    ]
    cm = ListedColormap(cols)
    nm = BoundaryNorm(lev, cm.N)
    w, e, s, n = dom["extent"]
    jj = (lat >= s - 1) & (lat <= n + 1)
    ii = (lon >= w - 1) & (lon <= e + 1)
    la, lo = lat[jj], lon[ii]
    p = prob[np.ix_(jj, ii)]
    pm = ax.pcolormesh(
        lo,
        la,
        np.where(p >= 0.05, p, np.nan),
        cmap=cm,
        norm=nm,
        transform=pc,
        shading="nearest",
        rasterized=True,
        zorder=2,
    )
    if contour is not None:
        ax.contour(
            lo,
            la,
            contour[np.ix_(jj, ii)],
            levels=[0.5],
            colors=["#111"],
            linewidths=1.1,
            linestyles="--",
            transform=pc,
            zorder=5,
        )
    map_features(ax)
    heading(fig, H, title, sub)
    cax = fig.add_axes([0.2, 0.36 / H, 0.6, 0.13 / H])
    cb = fig.colorbar(
        pm, cax=cax, orientation="horizontal", ticks=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    cb.ax.set_xticklabels(["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("probability (% of members)" + (f" · {contour_label}" if contour_label else ""), fontsize=8)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
    plt.close(fig)


def warm_features():
    """Draw the map features once in the parent process: forked render workers would otherwise download the Natural
    Earth shapefiles concurrently and read each other's half-written files (struct.error in shapefile.py, 2026-10-07)."""
    import io
    import matplotlib.pyplot as plt

    for dom in [*list(DOMAINS.values()), ENS_DOMAIN]:
        fig, ax, _H = map_axes(dom)
        map_features(ax)
        fig.savefig(io.BytesIO(), format="png", dpi=20)
        plt.close(fig)


def dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
