#!/usr/bin/env python3
"""Data access for the TC shear product. Only the levels and variables it needs, byte-ranged:

  ECMWF open data (AIFS-ENS control + 50, AIFS single, IFS HRES): Google Cloud mirror ONLY (user rule), through the
      repo's rangefetch (.index -> coalesced Range GETs). u, v, t at LEVELS, q at Q_LEVELS, msl.
  GFS 0.25 deg and HRRR (3 km, CONUS): AWS S3 open-data buckets (noaa-gfs-bdp-pds, noaa-hrrr-bdp-pds), one Range GET per
      record through the .idx (never NOMADS).
  SST: NOAA OISST v2.1 daily (NCEI, the newest preliminary day not after the cycle).

Every reader returns, per requested centre, a lat-lon BOX around it (shearlib.box) holding
  {"lat","lon" (box axes), "u","v","t","q" [lev, lat, lon] on LEVELS (q NaN-free, 0 above Q_LEVELS), "msl" hPa}
so that nothing global is kept: an AIFS-ENS step is ~2,300 global messages and only ~1 % of each is used.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecmwf"))
import shearlib as S                                                  # noqa: E402

LEVELS = S.LEVELS
Q_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300)
UA = "scorvec.com TC shear product (+https://scorvec.com)"
GFS_S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HRRR_S3 = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


class Ledger:
    def __init__(self):
        self.b = {}; self.lock = threading.Lock()

    def add(self, src, n):
        with self.lock:
            self.b[src] = self.b.get(src, 0) + n

    def report(self):
        return {k: round(v / 1e9, 3) for k, v in self.b.items()}


LEDGER = Ledger()


def http_get(url, rng=None, tries=4, timeout=90, src="http"):
    import requests
    h = {"User-Agent": UA}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{'' if rng[1] is None else rng[1]}"
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=(10, timeout))
            if r.status_code in (200, 206):
                LEDGER.add(src, len(r.content))
                return r.content
            if r.status_code == 404:
                return None
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:                                         # noqa: BLE001
            last = e
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"{url}: {last}")


def split_grib(blob: bytes) -> list[bytes]:
    out, o, n = [], 0, len(blob)
    while o < n:
        k = blob.find(b"GRIB", o)
        if k < 0:
            break
        ln = int.from_bytes(blob[k + 8:k + 16], "big") if blob[k + 7] == 2 else int.from_bytes(blob[k + 4:k + 7], "big")
        out.append(blob[k:k + ln]); o = k + ln
    return out


# ── decoding a global lat-lon message into boxes ──────────────────────────────────────────────────────────────────
def _decode_boxes(msg: bytes, centres_by_member: dict, half_lat: float, extra: dict | None = None):
    """-> (member, shortName, level, {key: (box values, box_lat, box_lon)}) for one regular lat-lon GRIB2 message.
    centres_by_member: {member: [(key, lat, lon), ...]}. extra: {key: (lat0, lat1, lon0, lon1)} fixed map boxes
    (member 0 / deterministic only)."""
    import eccodes as ec
    g = ec.codes_new_from_message(msg)
    try:
        sn = ec.codes_get(g, "shortName")
        tl = ec.codes_get(g, "typeOfLevel")
        lev = int(ec.codes_get(g, "level")) if tl == "isobaricInhPa" else 0
        try:
            mem = int(ec.codes_get(g, "perturbationNumber"))
        except Exception:                                              # noqa: BLE001
            mem = 0
        ni, nj = ec.codes_get(g, "Ni"), ec.codes_get(g, "Nj")
        la0 = ec.codes_get(g, "latitudeOfFirstGridPointInDegrees"); la1 = ec.codes_get(g, "latitudeOfLastGridPointInDegrees")
        lo0 = ec.codes_get(g, "longitudeOfFirstGridPointInDegrees"); di = ec.codes_get(g, "iDirectionIncrementInDegrees")
        vals = ec.codes_get_values(g).reshape(nj, ni)
    finally:
        ec.codes_release(g)
    lat = np.linspace(la0, la1, nj)
    lon = (lo0 + di * np.arange(ni)) % 360.0
    out = {}
    for key, clat, clon in centres_by_member.get(mem, []):
        jj, ii, bl, bo = S.box(lat, lon, clat, clon, half_lat=half_lat)
        out[key] = (vals[np.ix_(jj, ii)].astype(np.float32), bl, bo)
    if extra and mem == 0:
        for key, (a0, a1, b0, b1) in extra.items():
            jj = np.where((lat >= a0) & (lat <= a1))[0]; jj = jj[np.argsort(lat[jj])]
            dl = (lon - b0) % 360.0
            ii = np.where(dl <= (b1 - b0))[0]; ii = ii[np.argsort(dl[ii])]
            out[key] = (vals[np.ix_(jj, ii)].astype(np.float32), lat[jj], b0 + dl[ii])
    return mem, sn, lev, out


_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        _POOL = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2)), mp_context=mp.get_context("spawn"))
    return _POOL


def _work(args):
    msg, cbm, half, extra = args
    return _decode_boxes(msg, cbm, half, extra)


def _assemble(parts, nlev=len(LEVELS)):
    """parts: iterable of (member, shortName, level, {key: (vals, bl, bo)}) -> {key: box dict}."""
    boxes = {}
    li = {L: i for i, L in enumerate(LEVELS)}
    for mem, sn, lev, d in parts:
        for key, (vals, bl, bo) in d.items():
            b = boxes.get(key)
            if b is None:
                b = boxes[key] = {"lat": bl, "lon": bo, "_n": 0}
            nm = {"msl": "msl", "prmsl": "msl", "mslma": "msl", "u": "u", "v": "v", "t": "t", "q": "q"}.get(sn.lower())
            if nm is None:
                continue
            if nm == "msl":
                b["msl"] = vals / 100.0 if np.nanmean(vals) > 2000 else vals
            elif lev in li:
                if nm not in b:
                    b[nm] = np.full((nlev,) + vals.shape, np.nan, np.float32)
                b[nm][li[lev]] = vals
            b["_n"] += 1
    for b in boxes.values():
        if "q" in b:
            for i, L in enumerate(LEVELS):
                if L not in Q_LEVELS:
                    b["q"][i] = 0.0
    return boxes


def complete(b) -> bool:
    return all(k in b for k in ("u", "v", "t", "q", "msl")) and all(np.isfinite(b[k][i]).any() for k in ("u", "v", "t")
                                                                       for i in range(len(LEVELS))) \
        and all(np.isfinite(b["q"][LEVELS.index(L)]).any() for L in Q_LEVELS)


# ── ECMWF open data (Google mirror only) ──────────────────────────────────────────────────────────────────────────
def ecmwf_boxes(model, date, hh, step, kind, stream, centres_by_member, half_lat=16.0, extra=None, chunk=240):
    """Boxes for one step of an ECMWF open-data file. model 'aifs-ens' (kind cf/pf, stream enfo), 'aifs-single' or
    'ifs' (kind fc, stream oper). Returns {key: box} or None when the step is not on the Google mirror."""
    import rangefetch as rf
    try:
        idx = rf.fetch_index(date, hh, model, int(step), kind, stream=stream)
    except rf.IndexUnavailable as e:
        if e.not_found:
            return None
        raise
    mems = {str(m) for m in centres_by_member}
    def keep(e):
        if kind == "pf" and str(e.get("number")) not in mems:
            return False
        p, lt = e.get("param"), e.get("levtype")
        if p == "msl" and lt == "sfc":
            return True
        if lt != "pl":
            return False
        L = int(e.get("levelist", -1))
        return (p in ("u", "v", "t") and L in LEVELS) or (p == "q" and L in Q_LEVELS)
    want = sorted([e for e in idx if keep(e)], key=lambda e: e["_offset"])
    rel = rf.path_for(date, hh, model, int(step), kind, stream=stream) + ".grib2"
    parts = []
    futs = []
    ex = pool()
    for c0 in range(0, len(want), chunk):
        sub = want[c0:c0 + chunk]
        blob = rf.fetch_ranges(rel, rf.coalesce(sub, max_gap=128 * 1024))
        LEDGER.add("ecmwf-google", len(blob))
        for m in split_grib(blob):
            futs.append(ex.submit(_work, (m, centres_by_member, half_lat, extra)))
        if len(futs) > 3 * chunk:                                       # bound memory: drain the oldest
            parts += [f.result() for f in futs[:chunk]]; futs = futs[chunk:]
    parts += [f.result() for f in futs]
    return _assemble(parts)


# ── NOAA (AWS S3, .idx byte ranges) ───────────────────────────────────────────────────────────────────────────────
def idx_records(url):
    txt = http_get(url + ".idx", src="aws-s3")
    if txt is None:
        return None
    lines = [l.split(":") for l in txt.decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        out.append((int(p[1]), end, p[3], p[4]))
    return out


def _noaa_want(recs, msl_name):
    lev_txt = {f"{L} mb": L for L in LEVELS}
    want = []
    for s, e, var, lev in recs:
        if var in ("UGRD", "VGRD", "TMP") and lev in lev_txt:
            want.append((s, e, var, lev_txt[lev]))
        elif var == "SPFH" and lev in lev_txt and lev_txt[lev] in Q_LEVELS:
            want.append((s, e, var, lev_txt[lev]))
        elif var == msl_name and lev == "mean sea level":
            want.append((s, e, var, 0))
    return want


def gfs_boxes(date, hh, step, centres, half_lat=16.0, extra=None, workers=16):
    """GFS 0.25 deg from S3. centres: [(key, lat, lon)]. None if the step is not posted."""
    url = f"{GFS_S3}/gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{int(step):03d}"
    recs = idx_records(url)
    if recs is None:
        return None
    want = _noaa_want(recs, "PRMSL")
    need = 3 * len(LEVELS) + len(Q_LEVELS) + 1
    if len(want) < need:
        raise RuntimeError(f"GFS f{step:03d}: {len(want)}/{need} records in the idx")
    def one(w):
        s, e, var, L = w
        b = http_get(url, (s, e), src="aws-s3")
        if b is None or not b.startswith(b"GRIB"):
            raise RuntimeError(f"GFS f{step:03d} {var} {L}: bad range")
        return b
    with ThreadPoolExecutor(workers) as tp:
        msgs = list(tp.map(one, want))
    ex = pool()
    cbm = {0: centres}
    parts = list(ex.map(_work, [(m, cbm, half_lat, extra) for m in msgs]))
    return _assemble(parts)


# HRRR: Lambert conformal 3 km. Winds in wrfprs are GRID-relative: rotated to earth-relative here.
HRRR_LCC = dict(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, a=6371229, b=6371229)
HRRR_COARSEN = 4                                                      # 3 km -> 12 km block means before anything else


def _hrrr_decode(msg):
    import eccodes as ec
    g = ec.codes_new_from_message(msg)
    try:
        sn = ec.codes_get(g, "shortName"); tl = ec.codes_get(g, "typeOfLevel")
        lev = int(ec.codes_get(g, "level")) if tl == "isobaricInhPa" else 0
        nx, ny = ec.codes_get(g, "Nx"), ec.codes_get(g, "Ny")
        la1, lo1 = ec.codes_get(g, "latitudeOfFirstGridPointInDegrees"), ec.codes_get(g, "longitudeOfFirstGridPointInDegrees")
        dx = ec.codes_get(g, "DxInMetres")
        vals = ec.codes_get_values(g).reshape(ny, nx)
    finally:
        ec.codes_release(g)
    return sn, lev, vals.astype(np.float32), (nx, ny, la1, lo1, dx)


def _coarsen(a, f=HRRR_COARSEN):
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


class HRRRGrid:
    def __init__(self, nx, ny, la1, lo1, dx):
        import pyproj
        self.P = pyproj.Proj(HRRR_LCC)
        x0, y0 = self.P(lo1, la1)
        f = HRRR_COARSEN
        self.x = x0 + dx * (np.arange((nx // f)) * f + (f - 1) / 2)
        self.y = y0 + dx * (np.arange((ny // f)) * f + (f - 1) / 2)
        self.dx = dx * f
        X, Y = np.meshgrid(self.x, self.y)
        self.lon, self.lat = self.P(X, Y, inverse=True)
        # grid-relative -> earth-relative rotation angle (LCC, tangent cone): alpha = n (lon - lon0)
        n = math.sin(math.radians(38.5))
        self.alpha = np.radians(n * (((self.lon - (-97.5)) + 180) % 360 - 180))

    def to_earth(self, ug, vg):
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        return ca * ug + sa * vg, -sa * ug + ca * vg


def hrrr_fields(date, hh, step, workers=16):
    """HRRR wrfprs fields on the 12 km coarsened Lambert grid: {"grid", "u","v" (earth-relative), "ug","vg" (grid),
    "t","q" [lev, y, x], "msl"}. None if not posted."""
    url = f"{HRRR_S3}/hrrr.{date}/conus/hrrr.t{hh}z.wrfprsf{int(step):02d}.grib2"
    recs = idx_records(url)
    if recs is None:
        return None
    want = _noaa_want(recs, "MSLMA")
    def one(w):
        s, e, var, L = w
        b = http_get(url, (s, e), src="aws-s3")
        if b is None or not b.startswith(b"GRIB"):
            raise RuntimeError(f"HRRR f{step:02d} {var} {L}: bad range")
        return b
    with ThreadPoolExecutor(workers) as tp:
        msgs = list(tp.map(one, want))
    grid = None
    f = {}
    for m in msgs:
        sn, lev, vals, gi = _hrrr_decode(m)
        if grid is None:
            grid = HRRRGrid(*gi)
        f[(sn, lev)] = _coarsen(vals)
    li = {L: i for i, L in enumerate(LEVELS)}
    shape = grid.lat.shape
    out = {"grid": grid}
    for nm in ("u", "v", "t", "q"):
        out[nm] = np.full((len(LEVELS),) + shape, np.nan, np.float32)
    for (sn, lev), a in f.items():
        nm = {"u": "u", "v": "v", "t": "t", "q": "q"}.get(sn)
        if nm and lev in li:
            out[nm][li[lev]] = a[:shape[0], :shape[1]]
        elif sn in ("mslma", "prmsl", "msl"):
            out["msl"] = a[:shape[0], :shape[1]] / 100.0
    for i, L in enumerate(LEVELS):
        if L not in Q_LEVELS:
            out["q"][i] = 0.0
    out["ug"], out["vg"] = out["u"], out["v"]
    out["u"], out["v"] = grid.to_earth(out["ug"], out["vg"])
    return out


# ── SST ───────────────────────────────────────────────────────────────────────────────────────────────────────────
OISST = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"


def oisst(init: dt.datetime, cache: Path):
    """(lat, lon 0..360, sst degC [lat, lon] with NaN over land/ice, date used). The newest day <= init - 1 day,
    final or preliminary; looks back up to 10 days."""
    import netCDF4
    cache.mkdir(parents=True, exist_ok=True)
    for back in range(1, 11):
        d = init - dt.timedelta(days=back)
        for suf in ("", "_preliminary"):
            name = f"oisst-avhrr-v02r01.{d:%Y%m%d}{suf}.nc"
            fp = cache / name
            if not fp.exists():
                b = http_get(f"{OISST}/{d:%Y%m}/{name}", src="ncei", tries=2)
                if b is None:
                    continue
                fp.write_bytes(b)
            with netCDF4.Dataset(fp) as nc:
                sst = np.array(nc["sst"][0, 0], dtype=float)
                sst = np.where(sst < -50, np.nan, sst)
                ice = np.array(nc["ice"][0, 0], dtype=float) if "ice" in nc.variables else None
                lat = np.array(nc["lat"][:]); lon = np.array(nc["lon"][:])
            if ice is not None:
                sst = np.where(np.isfinite(ice) & (ice > 0.15), np.nan, sst)
            return lat, lon, sst, f"{d:%Y-%m-%d}" + (" (preliminary)" if suf else "")
    return None


def sst_at(oi, clat, clon, r_km=100.0):
    """Mean SST (degC) within r_km of the centre; NaN if fewer than half of those cells are ocean (landfall)."""
    lat, lon, sst, _ = oi
    j = np.where(np.abs(lat - clat) <= 1.5)[0]
    dl = (lon - clon + 180) % 360 - 180
    i = np.where(np.abs(dl) <= 1.5 / max(0.3, math.cos(math.radians(clat))))[0]
    LA, LO = np.meshgrid(lat[j], lon[i], indexing="ij")
    d = S.gc_km(clat, clon, LA, LO)
    v = sst[np.ix_(j, i)]
    m = d <= r_km
    if not m.any() or np.isfinite(v[m]).mean() < 0.5:
        return np.nan
    return float(np.nanmean(v[m]))


def hrrr_box(F, clat, clon, half_lat=16.0, res=0.125):
    """Resample the coarsened HRRR fields onto a regular lat-lon box around a centre (NaN outside the HRRR domain),
    so that the HRRR goes through exactly the same analysis as the global models."""
    from scipy.ndimage import map_coordinates
    g = F["grid"]
    half_lon = min(40.0, half_lat / max(0.3, math.cos(math.radians(min(abs(clat) + half_lat, 75.0)))))
    bl = np.arange(clat - half_lat, clat + half_lat + 1e-6, res)
    bo = np.arange(clon - half_lon, clon + half_lon + 1e-6, res)
    LO, LA = np.meshgrid(bo, bl)
    X, Y = g.P(LO, LA)
    fi = (X - g.x[0]) / (g.x[1] - g.x[0]); fj = (Y - g.y[0]) / (g.y[1] - g.y[0])
    out_of = (fi < 0) | (fi > len(g.x) - 1) | (fj < 0) | (fj > len(g.y) - 1)
    def rs(a):
        r = map_coordinates(np.nan_to_num(np.asarray(a, np.float64), nan=0.0), [fj, fi], order=1, mode="nearest")
        bad = map_coordinates(np.isnan(np.asarray(a, np.float64)).astype(float), [fj, fi], order=1, mode="nearest") > 0
        r[out_of | bad] = np.nan
        return r.astype(np.float32)
    b = {"lat": bl, "lon": bo}
    for k in ("u", "v", "t", "q"):
        b[k] = np.stack([rs(F[k][i]) for i in range(len(LEVELS))])
    b["msl"] = rs(F["msl"])
    return b
