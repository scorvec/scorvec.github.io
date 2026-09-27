#!/usr/bin/env python3
"""Upper stratosphere, 100 -> 1 hPa, from four models that carry it: data side (2026-09-27; user: "let's add GDPS,
GEFS, GFS and GEOS-FP (allow the user to toggle between and compare the upper stratosphere)").

Every model is reduced to the SAME diagnostics on its own levels, so the renderer (upperstrat.py) compares like with
like:
  u60   zonal-mean zonal wind at exactly 60 deg (the sudden-warming criterion latitude), per hemisphere
  capT  polar-cap temperature: the cos-weighted mean over 60-90 deg of the zonal-mean temperature
  maps  geopotential height and temperature at 5 and 1 hPa, 20-90 deg, on a common 1 x 1 deg grid, days 0-10
One instant per day: the 00Z field at every 24-hour step (migrating tides cancel in a zonal mean).

  gefs  NOAA GEFS 00Z, gec00 + gep01-30, 0.5 deg, days 0-35. 1-7 and 20/30/70 hPa are in the pgrb2b files, 10/50/100
        in pgrb2a. Byte-ranged from NOAA's S3 bucket through the .idx files (the site rule for NOAA data: S3, never
        NOMADS). Maps from the control.
  gfs   NOAA GFS 00Z, pgrb2.1p00 (the same inventory as the 0.25 deg file, a sixteenth of the bytes), days 0-16.
  gdps  ECCC GDPS 00Z, 15 km lat-lon GRIB2 from the dated MSC Datamart tree (dd.weather.gc.ca/YYYYMMDD/WXO-DD/...),
        days 0-10. Pressure levels 1, 5, 10, 20, 30, 50, 100 hPa only (no 2, 3, 7 or 70).
  geos  NASA GMAO GEOS FP 00Z forecast (NCCS OPeNDAP fcast/inst3_3d_asm_Np), days 0-10, read SEQUENTIALLY and
        subset (only the rows and levels used): parallel OPeNDAP reads have silently returned corrupt data here before.

    python upperstrat_data.py fetch --model gfs --date 20260927 --out scripts/strat/data/upperstrat
    python upperstrat_data.py tail --days 4 --out assets/sst/data/upperstrat_tail.json     # GEOS FP analyses
    python upperstrat_data.py latest                                                       # newest 00Z date with GFS f384
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Ambiguous reference date.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEV_ALL = [100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0]
MAP_LEVELS = [5.0, 1.0]
MAP_DAYS = 11                                   # days 0..10
LAT_N = np.arange(20.0, 90.0 + 1e-6, 1.0)        # map grid, ascending; SH uses -LAT_N
LON_1 = np.arange(0.0, 360.0 - 1e-6, 1.0)
HEMIS = ("nh", "sh")

MODELS = {
    "gefs": dict(name="GEFS", levels=[100.0, 70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0], days=35),
    "gfs": dict(name="GFS", levels=[100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0], days=16),
    "gdps": dict(name="GDPS", levels=[100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 1.0], days=10),
    "geos": dict(name="GEOS FP", levels=list(LEV_ALL), days=10),
}
S3_GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
S3_GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
DATAMART = "https://dd.weather.gc.ca"
GEOS_FC = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/inst3_3d_asm_Np"
GEOS_AS = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Np"
UA = {"User-Agent": "scorvec-upperstrat"}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ reductions ---
def reduce_zonal(field: np.ndarray, lat: np.ndarray):
    """(nlat, nlon) global field -> {nh, sh: (value at 60 deg, cap 60-90 mean)} of its zonal mean."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zm = np.nanmean(field, axis=1)
    out = {}
    for h, s in (("nh", 1.0), ("sh", -1.0)):
        j60 = int(np.argmin(np.abs(lat - 60.0 * s)))
        m = (lat * s) >= 60.0 - 1e-6
        w = np.cos(np.deg2rad(lat[m]))
        out[h] = (float(zm[j60]), float((zm[m] * w).sum() / w.sum()))
    return out


def to_maps(field: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """(nlat, nlon) -> {nh: (71, 360), sh: (71, 360)} on LAT_N (SH: -LAT_N, ordered 20 -> 90 S) x LON_1, bilinear."""
    from scipy.interpolate import RegularGridInterpolator
    o = np.argsort(lat)
    la = lat[o]
    lo = np.mod(lon, 360.0)
    ol = np.argsort(lo)
    lo = lo[ol]
    g = field[o][:, ol]
    g = np.concatenate([g, g[:, :1]], axis=1)                    # periodic in longitude
    lo = np.append(lo, lo[0] + 360.0)
    f = RegularGridInterpolator((la, lo), g, bounds_error=False, fill_value=np.nan)
    out = {}
    for h, s in (("nh", 1.0), ("sh", -1.0)):
        L, M = np.meshgrid(LAT_N * s, LON_1, indexing="ij")
        out[h] = f(np.column_stack([L.ravel(), M.ravel()])).reshape(L.shape).astype("float32")
    return out


# ------------------------------------------------------------------------------------------------ GRIB access ---
def http_get(url: str, rng: tuple[int, int] | None = None, tries: int = 5, timeout: int = 120) -> bytes:
    """GET with retries; a 404 is final at once (a file not uploaded yet must not cost 20 s of backoff)."""
    import urllib.error
    for k in range(tries):
        try:
            h = dict(UA)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
        except Exception:                                                    # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def idx_of(url: str):
    """[(start, end, var, level)] from a NOAA .idx (end = -1 for the last message)."""
    lines = [l.split(":") for l in http_get(url + ".idx").decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
        out.append((int(p[1]), end, p[3], p[4]))
    return out


def decode(buf: bytes):
    """One GRIB2 message -> (values (nlat, nlon), lat (nlat,), lon (nlon,)) for a regular lat-lon grid."""
    import eccodes
    if buf[:4] != b"GRIB":
        raise ValueError("not a GRIB message")
    g = eccodes.codes_new_from_message(buf)
    try:
        nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
        la0, la1 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees")
        lo0, dlo = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees"), eccodes.codes_get(g, "iDirectionIncrementInDegrees")
        v = eccodes.codes_get_values(g).reshape(nj, ni).astype("float64")
        miss = eccodes.codes_get(g, "missingValue")
    finally:
        eccodes.codes_release(g)
    v[v == miss] = np.nan
    lat = np.linspace(la0, la1, nj)
    lon = lo0 + dlo * np.arange(ni)
    return v, lat, lon


def grib_messages(url: str, want: list[tuple[str, str]], gap: int = 150_000):
    """{(var, level): (values, lat, lon)} for the wanted (VAR, 'N mb') messages of one file, byte-ranged; messages
    closer than `gap` bytes share one request. Raises KeyError when the file lacks one."""
    idx = idx_of(url)
    pos = {}
    for k, (s, e, var, lev) in enumerate(idx):
        if (var, lev) in want and (var, lev) not in pos:
            pos[(var, lev)] = (s, e, k)
    miss = [w for w in want if w not in pos]
    if miss:
        raise KeyError(f"{url.rsplit('/', 1)[-1]} lacks {miss}")
    items = sorted(pos.items(), key=lambda kv: kv[1][0])
    groups, cur = [], [items[0]]
    for it in items[1:]:
        if it[1][0] - cur[-1][1][1] <= gap and cur[-1][1][1] >= 0:
            cur.append(it)
        else:
            groups.append(cur); cur = [it]
    groups.append(cur)
    out = {}
    for grp in groups:
        s0, e1 = grp[0][1][0], grp[-1][1][1]
        buf = http_get(url, (s0, e1))
        for key, (s, e, _) in grp:
            out[key] = decode(buf[s - s0:(e - s0 + 1) if e >= 0 else None])
    return out


def mb(L: float) -> str:
    return f"{L:g} mb"


# ---------------------------------------------------------------------------------------------------- models ---
def _pack(zon: dict, levels, maps: dict | None):
    """zonal {level: {'u': reduce_zonal, 't': reduce_zonal}} -> u60/capT arrays (hemi, level)."""
    u = np.full((2, len(levels)), np.nan); t = u.copy()
    for j, L in enumerate(levels):
        for i, h in enumerate(HEMIS):
            u[i, j] = zon[L]["u"][h][0]
            t[i, j] = zon[L]["t"][h][1]
    return u, t, maps


def gfs_step(date: str, fh: int):
    levels = MODELS["gfs"]["levels"]
    url = f"{S3_GFS}/gfs.{date}/00/atmos/gfs.t00z.pgrb2.1p00.f{fh:03d}"
    want = [(v, mb(L)) for L in levels for v in ("TMP", "UGRD")]
    domap = fh <= 24 * (MAP_DAYS - 1)
    if domap:
        want += [("HGT", mb(L)) for L in MAP_LEVELS]
    msg = grib_messages(url, want)
    zon = {L: {"u": reduce_zonal(msg[("UGRD", mb(L))][0], msg[("UGRD", mb(L))][1]),
               "t": reduce_zonal(msg[("TMP", mb(L))][0], msg[("TMP", mb(L))][1])} for L in levels}
    maps = None
    if domap:
        maps = {(var, L): to_maps(*msg[(gv, mb(L))]) for L in MAP_LEVELS for var, gv in (("z", "HGT"), ("t", "TMP"))}
    return _pack(zon, levels, maps)


GEFS_A = {10.0, 50.0, 100.0}


def gefs_step(args):
    date, mem, fh, want_maps = args
    levels = MODELS["gefs"]["levels"]
    base = f"{S3_GEFS}/gefs.{date}/00/atmos"
    ua = f"{base}/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{fh:03d}"
    ub = f"{base}/pgrb2bp5/{mem}.t00z.pgrb2b.0p50.f{fh:03d}"
    wa = [(v, mb(L)) for L in levels if L in GEFS_A for v in ("TMP", "UGRD")]
    wb = [(v, mb(L)) for L in levels if L not in GEFS_A for v in ("TMP", "UGRD")]
    if want_maps:
        wb += [("HGT", mb(L)) for L in MAP_LEVELS]
    try:
        with ThreadPoolExecutor(2) as ex:
            fa, fb = ex.submit(grib_messages, ua, wa), ex.submit(grib_messages, ub, wb)
            msg = {**fa.result(), **fb.result()}
    except Exception as e:                                                  # noqa: BLE001
        return mem, fh, None, f"{type(e).__name__}: {str(e)[:120]}"
    zon = {L: {"u": reduce_zonal(msg[("UGRD", mb(L))][0], msg[("UGRD", mb(L))][1]),
               "t": reduce_zonal(msg[("TMP", mb(L))][0], msg[("TMP", mb(L))][1])} for L in levels}
    maps = None
    if want_maps:
        maps = {(var, L): to_maps(*msg[(gv, mb(L))]) for L in MAP_LEVELS for var, gv in (("z", "HGT"), ("t", "TMP"))}
    return mem, fh, _pack(zon, levels, maps), None


GDPS_VAR = {"t": "AirTemp", "u": "WindU", "z": "GeopotentialHeight"}


def gdps_file(date: str, fh: int, var: str, L: float):
    url = (f"{DATAMART}/{date}/WXO-DD/model_gdps/15km/00/{fh:03d}/"
           f"{date}T00Z_MSC_GDPS_{GDPS_VAR[var]}_IsbL-{int(L):04d}_LatLon0.15_PT{fh:03d}H.grib2")
    return decode(http_get(url))


def gdps_step(date: str, fh: int):
    levels = MODELS["gdps"]["levels"]
    jobs = [(v, L) for L in levels for v in ("t", "u")] + [("z", L) for L in MAP_LEVELS]
    with ThreadPoolExecutor(6) as ex:
        res = dict(zip(jobs, ex.map(lambda j: gdps_file(date, fh, *j), jobs)))
    zon = {L: {"u": reduce_zonal(res[("u", L)][0], res[("u", L)][1]), "t": reduce_zonal(res[("t", L)][0], res[("t", L)][1])}
           for L in levels}
    maps = {(var, L): to_maps(*res[(var, L)]) for L in MAP_LEVELS for var in ("z", "t")}
    return _pack(zon, levels, maps)


class Geos:
    """GEOS FP on NCCS OPeNDAP: one handle, sequential subset reads with retries (504s under load)."""

    def __init__(self, url):
        import xarray as xr
        self.d = None
        for k in range(5):
            try:
                d = xr.open_dataset(url, engine="netcdf4")
                if not np.issubdtype(d.time.dtype, np.datetime64):
                    raise RuntimeError("time axis not decoded")
                self.d = d
                break
            except Exception as e:                                          # noqa: BLE001
                log(f"  open {url.rsplit('/', 1)[-1]}: {str(e)[:90]}; retry")
                time.sleep(20 * (k + 1))
        if self.d is None:
            raise RuntimeError(f"GEOS FP unreachable: {url}")
        self.lev = self.d.lev.values
        self.lat = self.d.lat.values
        self.times = pd.DatetimeIndex(self.d.time.values)
        self.li = [int(np.argmin(np.abs(self.lev - L))) for L in LEV_ALL]
        assert np.allclose(self.lev[self.li], LEV_ALL), self.lev[self.li]
        self.l0, self.l1 = min(self.li), max(self.li)
        j60n, j60s = int(np.argmin(np.abs(self.lat - 60))), int(np.argmin(np.abs(self.lat + 60)))
        self.cap = {"nh": slice(j60n, len(self.lat), 2), "sh": slice(0, j60s + 1, 2)}
        j20n, j20s = int(np.argmin(np.abs(self.lat - 20))), int(np.argmin(np.abs(self.lat + 20)))
        self.mapsl = {"nh": slice(j20n, len(self.lat), 2), "sh": slice(0, j20s + 1, 2)}

    def read(self, vars_, it, lev: slice, lat: slice, lon: slice, tries: int = 7):
        """Server errors (504s under load) are retried up to `tries` times; a slab that comes back mostly fill values
        (a step not posted yet) is retried only twice, so an unposted cycle fails in a minute, not ten."""
        empty = 0
        for k in range(tries):
            try:
                sub = self.d[list(vars_)].isel(time=it, lev=lev, lat=lat, lon=lon).load()
                out = {v: np.where(np.abs(sub[v].values) < 1e10, sub[v].values, np.nan).astype("float64") for v in vars_}
                if not all(np.isfinite(out[v]).mean() > 0.99 for v in vars_):
                    empty += 1
                    raise RuntimeError("incomplete slab (not posted yet, or a partial read)")
                return out, sub.lat.values, sub.lon.values, sub.lev.values
            except Exception as e:                                          # noqa: BLE001
                if k == tries - 1 or empty >= 2:
                    raise
                log(f"    read t={it}: {str(e)[:90]}; retry")
                time.sleep(15 * (k + 1))

    def zonal(self, it):
        """{hemi: (u60 (13,), capT (13,))} at time index it, LEV_ALL order."""
        res = {}
        for h in HEMIS:
            f, lat, _, lev = self.read(("u", "t"), it, slice(self.l0, self.l1 + 1), self.cap[h], slice(None, None, 4))
            li = [int(np.argmin(np.abs(lev - L))) for L in LEV_ALL]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                zu, zt = np.nanmean(f["u"], -1)[li], np.nanmean(f["t"], -1)[li]          # (level, lat)
            s = 1.0 if h == "nh" else -1.0
            j60 = int(np.argmin(np.abs(lat - 60.0 * s)))
            m = (lat * s) >= 60.0 - 1e-6
            w = np.cos(np.deg2rad(lat[m]))
            res[h] = (zu[:, j60], (zt[:, m] * w).sum(-1) / w.sum())
        return res

    def maps(self, it):
        """{(var, level): {hemi: (71, 360)}} for z/t at MAP_LEVELS."""
        i5, i1 = int(np.argmin(np.abs(self.lev - 5.0))), int(np.argmin(np.abs(self.lev - 1.0)))
        out = {}
        for h in HEMIS:
            f, lat, lon, lev = self.read(("h", "t"), it, slice(min(i5, i1), max(i5, i1) + 1), self.mapsl[h], slice(None, None, 2))
            for L in MAP_LEVELS:
                k = int(np.argmin(np.abs(lev - L)))
                for var, gv in (("z", "h"), ("t", "t")):
                    out.setdefault((var, L), {})[h] = to_maps(f[gv][k], lat, lon)[h]      # the slab covers 20-90 deg
        return out


def geos_forecast(date: str):
    g = Geos(f"{GEOS_FC}/inst3_3d_asm_Np.{date}_00")
    init = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
    steps = [int(np.argmin(np.abs(g.times - (init + pd.Timedelta(days=d))))) for d in range(MODELS["geos"]["days"] + 1)]
    if g.times[steps[0]] != init or len(set(steps)) != len(steps):
        raise RuntimeError(f"GEOS FP {date}: time axis {g.times[0]} .. {g.times[-1]}")
    # the last step first: NCCS lists a cycle before it has filled it, and a half-posted run must fail fast
    g.read(("t",), steps[-1], slice(g.l1, g.l1 + 1), g.cap["nh"], slice(None, None, 8))
    U = np.full((2, 1, len(steps), len(LEV_ALL)), np.nan); T = U.copy()
    maps = []
    for k, it in enumerate(steps):
        t0 = time.time()
        z = g.zonal(it)
        for i, h in enumerate(HEMIS):
            U[i, 0, k], T[i, 0, k] = z[h]
        if k < MAP_DAYS:
            maps.append(g.maps(it))
        log(f"  geos day {k}: NH u60 1 hPa {U[0, 0, k, -1]:+.1f}  SH {U[1, 0, k, -1]:+.1f}  ({time.time() - t0:.0f}s)")
    return U, T, maps


# ----------------------------------------------------------------------------------------------------- fetch ---
def save(out_dir: Path, model: str, date: str, U, T, maps, members: list[str], leads_h, note=""):
    """U/T (hemi, member, lead, level); maps: list over map days of {(var, L): {hemi: arr}}."""
    nd = len(maps)
    M = np.full((2, nd, 2, len(MAP_LEVELS), len(LAT_N), len(LON_1)), np.nan, "float32")   # hemi, day, var(z,t), level
    for d, mp in enumerate(maps):
        if mp is None:
            continue
        for i, h in enumerate(HEMIS):
            for vi, var in enumerate(("z", "t")):
                for li, L in enumerate(MAP_LEVELS):
                    M[i, d, vi, li] = mp[(var, L)][h]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model}.npz"
    np.savez_compressed(p, init=f"{date[:4]}-{date[4:6]}-{date[6:8]}T00:00", model=model, name=MODELS[model]["name"],
                        levels=np.array(MODELS[model]["levels"]), leads_h=np.array(leads_h), members=np.array(members),
                        u60=U.astype("float32"), capT=T.astype("float32"), maps=M, map_levels=np.array(MAP_LEVELS),
                        map_lat=LAT_N, map_lon=LON_1, note=note)
    log(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB): {U.shape[1]} member(s) x {U.shape[2]} days x {U.shape[3]} levels, "
        f"{sum(m is not None for m in maps)} map days")


def fetch(a) -> int:
    model, date = a.model, a.date
    out = Path(a.out); out = out if out.is_absolute() else REPO / out
    cfg = MODELS[model]
    leads = [24 * d for d in range(cfg["days"] + 1)]
    t0 = time.time()
    if model == "geos":
        U, T, maps = geos_forecast(date)
        save(out, model, date, U, T, maps, ["det"], leads)
    elif model in ("gfs", "gdps"):
        step = gfs_step if model == "gfs" else gdps_step
        U = np.full((2, 1, len(leads), len(cfg["levels"])), np.nan); T = U.copy(); maps = []
        with ThreadPoolExecutor(4 if model == "gfs" else 2) as ex:
            for k, (u, t, mp) in enumerate(ex.map(lambda fh: step(date, fh), leads)):
                U[:, 0, k], T[:, 0, k] = u, t
                if k < MAP_DAYS:
                    maps.append(mp)
                log(f"  {model} day {k}: NH u60 1 hPa {u[0, -1]:+.1f}  SH {u[1, -1]:+.1f}")
        save(out, model, date, U, T, maps, ["det"], leads)
    else:
        if a.require_full:
            last = f"{S3_GEFS}/gefs.{date}/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f{leads[-1]:03d}.idx"
            try:
                http_get(last, tries=2)
            except Exception:                                               # noqa: BLE001
                raise SystemExit(f"GEFS {date}: the control's day-{cfg['days']} file is not on S3 (--require-full)")
        mems = (["gec00"] + [f"gep{i:02d}" for i in range(1, 31)])[:a.max_members]
        leads = leads[:a.max_days + 1]
        jobs = [(date, m, fh, m == "gec00" and fh <= 24 * (MAP_DAYS - 1)) for m in mems for fh in leads]
        U = np.full((2, len(mems), len(leads), len(cfg["levels"])), np.nan); T = U.copy()
        maps = [None] * MAP_DAYS
        errs = []
        with ProcessPoolExecutor(a.procs) as ex:
            for n, (mem, fh, res, err) in enumerate(ex.map(gefs_step, jobs, chunksize=2)):
                mi, li = mems.index(mem), leads.index(fh)
                if res is None:
                    errs.append(f"{mem} f{fh:03d} {err}")
                    continue
                U[:, mi, li], T[:, mi, li], mp = res
                if mp is not None:
                    maps[li] = mp
                if n % 100 == 0:
                    log(f"  gefs {n + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
        for e in errs[:8]:
            log("  missing:", e)
        # the longest run of leads over which the control and at least 16 members are complete (a cycle whose
        # extended range is only partly uploaded is cut there, not dropped); members complete through it are kept
        step_ok = np.isfinite(U).all(axis=(0, 3)) & np.isfinite(T).all(axis=(0, 3))          # (member, lead)
        thru = np.cumprod(step_ok, axis=1).astype(bool)
        need = min(16, len(mems))
        good = thru[0] & (thru.sum(0) >= need)
        nl = int(np.argmin(good)) if not good.all() else len(leads)
        log(f"  gefs: {len(errs)} member-steps failed; complete to day {nl - 1} with {int(thru[:, max(nl - 1, 0)].sum())} members")
        if nl < min(11, len(leads)):
            raise SystemExit(f"GEFS {date}: control + {need} members complete only to day {nl - 1}")
        keep = np.flatnonzero(thru[:, nl - 1])
        notes = []
        if len(keep) < len(mems):
            notes.append(f"{len(mems) - len(keep)} incomplete member(s) dropped")
        if nl < len(leads):
            notes.append(f"cut at day {nl - 1}: the rest of the cycle was not on S3")
        save(out, model, date, U[:, keep][:, :, :nl], T[:, keep][:, :, :nl], maps, [mems[i] for i in keep], leads[:nl],
             note="; ".join(notes))
    log(f"{model} {date}: {time.time() - t0:.0f}s")
    return 0


# ------------------------------------------------------------------------------------------------------ tail ---
TAIL_SEED = HERE / "reference" / "upperstrat_tail_seed.json"


def load_tail(extra: Path | None = None) -> dict:
    h = {}
    for p in (TAIL_SEED, extra):
        if p is not None and Path(p).exists():
            h.update(json.loads(Path(p).read_text()).get("days", {}))
    return dict(sorted(h.items()))


def tail(a) -> int:
    """GEOS FP analyses: the daily mean of the 00/06/12/18Z states of u60 and capT (both hemispheres, LEV_ALL)."""
    out = Path(a.out); out = out if out.is_absolute() else REPO / out
    have = json.loads(out.read_text()).get("days", {}) if out.exists() else {}
    seed = load_tail(None) if a.seed_too else {}
    g = Geos(GEOS_AS)
    tn = g.times[-1]
    end = tn.normalize() if tn.hour >= 18 else tn.normalize() - pd.Timedelta(days=1)
    days = [d for d in pd.date_range(end - pd.Timedelta(days=a.days - 1), end, freq="D")
            if a.force or (f"{d:%Y-%m-%d}" not in have and f"{d:%Y-%m-%d}" not in seed)]
    n = 0
    for d in days:
        t0 = time.time()
        acc = {h: [[], []] for h in HEMIS}
        try:
            for hh in (0, 6, 12, 18):
                it = int(np.argmin(np.abs(g.times - (d + pd.Timedelta(hours=hh)))))
                if g.times[it] != d + pd.Timedelta(hours=hh):
                    raise RuntimeError(f"{d:%Y-%m-%d} {hh:02d}Z not in the archive")
                z = g.zonal(it)
                for h in HEMIS:
                    acc[h][0].append(z[h][0]); acc[h][1].append(z[h][1])
        except Exception as e:                                               # noqa: BLE001
            log(f"  {d:%Y-%m-%d}: {str(e)[:100]}")
            continue
        have[f"{d:%Y-%m-%d}"] = {h: {"u60": np.round(np.mean(acc[h][0], 0), 2).tolist(),
                                     "capT": np.round(np.mean(acc[h][1], 0), 2).tolist()} for h in HEMIS}
        n += 1
        v = have[f"{d:%Y-%m-%d}"]
        log(f"  {d:%Y-%m-%d}: NH u60 10/5/1 hPa {v['nh']['u60'][6]:+.1f} {v['nh']['u60'][8]:+.1f} {v['nh']['u60'][12]:+.1f}  "
            f"SH {v['sh']['u60'][6]:+.1f} {v['sh']['u60'][8]:+.1f} {v['sh']['u60'][12]:+.1f}  ({time.time() - t0:.0f}s)")
        if n % 5 == 0:
            _write_tail(out, have, a.keep)
    _write_tail(out, have, a.keep)
    log(f"GEOS FP tail: {n} new day(s), {min(len(have), a.keep)} kept -> {out}")
    return 0


def _write_tail(out: Path, have: dict, keep: int):
    ks = sorted(have)[-keep:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "GEOS FP inst3_3d_asm_Np analyses (NASA GMAO, NCCS OPeNDAP): daily mean of the "
                                         "00/06/12/18Z states; u60 = zonal-mean u at 60 deg (m/s), capT = cos-weighted "
                                         "60-90 deg mean of the zonal-mean T (K)", "levels": LEV_ALL,
                               "days": {k: have[k] for k in ks}}, separators=(",", ":")))


def latest(a) -> int:
    """Newest date whose GFS 00Z f384 (1 deg) is on S3 - printed as YYYYMMDD."""
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for back in range(0, 3):
        d = now - pd.Timedelta(days=back)
        url = f"{S3_GFS}/gfs.{d:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.1p00.f384.idx"
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=30)
            print(f"{d:%Y%m%d}")
            return 0
        except Exception:                                                   # noqa: BLE001
            continue
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("--model", required=True, choices=list(MODELS)); f.add_argument("--date", required=True)
    f.add_argument("--out", default=str(HERE / "data" / "upperstrat")); f.add_argument("--procs", type=int, default=6)
    f.add_argument("--max-members", type=int, default=31, help="GEFS test subset"); f.add_argument("--max-days", type=int, default=35)
    f.add_argument("--require-full", action="store_true", help="GEFS: fail at once unless the control's last step is on S3")
    t = sp.add_parser("tail"); t.add_argument("--days", type=int, default=4); t.add_argument("--keep", type=int, default=120)
    t.add_argument("--out", default=str(REPO / "assets" / "sst" / "data" / "upperstrat_tail.json"))
    t.add_argument("--force", action="store_true"); t.add_argument("--seed-too", action="store_true",
                                                                   help="skip days already in the committed seed")
    sp.add_parser("latest")
    a = ap.parse_args()
    return {"fetch": fetch, "tail": tail, "latest": latest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
