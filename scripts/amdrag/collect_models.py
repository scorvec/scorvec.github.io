#!/usr/bin/env python3
"""Collector for the non-GEM models of the conservation-and-drag intercomparison (GEM: gem_collect.py). Each 00Z
init is fetched (byte ranges only), reduced IN THE RUNNER by common.Reducer and saved as one small .npz per
(model, init) under data/<model>/<YYYYMMDD>.npz. No GRIB is kept.

  gfs          NOAA GFS (S3 noaa-gfs-bdp-pds). Torque fields on the model's own Gaussian physics grid (sfluxgrbf:
               p_s, HGT, 10 m wind, UFLX = turbulent surface stress and U-GWD = orographic gravity-wave + blocking
               surface stress, both 6-h means; PWAT); level winds 0.25 deg pgrb2 (41 levels) every 24 h.
  gefs         NOAA GEFS v12 (S3 noaa-gefs-pds), control + 4, 0.5 deg pgrb2a+b: p_s, HGT, 10 m wind, UFLX, U-GWD,
               PWAT; 31 levels every 24 h.
  aigfs        NOAA AIGFS (S3 noaa-nws-graphcastgfs-pds, an AI model on GDAS): p_s, 10 m wind, 13 levels; its own
               static orography from the init dataset ('orog'). No stress (an AI model has none).
  ifs-ens      ECMWF IFS-ENS control + 4 (Google Cloud mirror ONLY): sp, 10u/10v, ewss (accumulated turbulent
               stress, includes turbulent orographic form drag), tcw; 14 levels; orography = HRES 'z' at step 0
               (IFS-ENS has run at the HRES resolution since cycle 48r1; the open-data control IS HRES).
  aifs-ens     ECMWF AIFS-ENS control + 4: sp, 10u/10v, tcw; 14 levels; its own 'z' at step 0. No stress.
  aifs-single  ECMWF AIFS deterministic: as aifs-ens.
All 00Z, 0-360 h, torque fields every 6 h, level winds every 24 h.

    python collect_models.py --model gfs --dates 20260914 20260915
Data: NOAA/NCEP GFS, GEFS and AIGFS via the NOAA Open Data Dissemination programme on AWS; ECMWF open data,
CC BY 4.0 (Google Cloud mirror).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "ecmwf"))
import common as C                                                     # noqa: E402

os.environ.setdefault("ECMWF_SOURCES", "google")
LAST = int(os.environ.get("AMDRAG_LAST", "360"))   # 24 = the day-1 (WGNE-comparable) mode
LEVELS_ON = os.environ.get("AMDRAG_LEVELS", "1") == "1"
DT = 6
WORKERS = int(os.environ.get("AMDRAG_WORKERS", "4"))
S3 = {"gfs": "https://noaa-gfs-bdp-pds.s3.amazonaws.com", "gefs": "https://noaa-gefs-pds.s3.amazonaws.com",
      "aigfs": "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"}
GFS_LEVELS = [0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250,
              300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
GEFS_A = [10, 50, 100, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
GEFS_B = [1, 2, 3, 5, 7, 20, 30, 70, 150, 350, 450, 550, 600, 650, 750, 800, 900, 950, 975]
EC_LEVELS = C.C14
AIGFS_LEVELS = C.C13
ENS_MEMBERS = [int(x) for x in os.environ.get("AMDRAG_MEMBERS", "0,1,2,3,4").split(",")]
SPEC = {
    "gfs": dict(members=[0], last=LAST),
    "gefs": dict(members=ENS_MEMBERS, last=LAST),
    "aigfs": dict(members=[0], last=LAST),
    "ifs-ens": dict(members=ENS_MEMBERS, last=LAST),
    "aifs-ens": dict(members=ENS_MEMBERS, last=LAST),
    "aifs-single": dict(members=[0], last=LAST),
}


# ---- GRIB ---------------------------------------------------------------------------------------------------------
def split_messages(blob: bytes) -> list[bytes]:
    out, pos = [], 0
    while True:
        i = blob.find(b"GRIB", pos)
        if i < 0:
            return out
        ed = blob[i + 7]
        ln = int.from_bytes(blob[i + 8:i + 16], "big") if ed == 2 else int.from_bytes(blob[i + 4:i + 7], "big")
        out.append(blob[i:i + ln])
        pos = i + ln


def decode_one(msg: bytes):
    """-> (values 2-D float64 with latitude ascending, lat, lon, keys)."""
    import eccodes as ec
    h = ec.codes_new_from_message(msg)
    try:
        gt = ec.codes_get(h, "gridType")
        Ni, Nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
        v = ec.codes_get_values(h).astype(np.float64)
        if ec.codes_get(h, "bitmapPresent"):
            v[v == ec.codes_get(h, "missingValue")] = np.nan
        v = v.reshape(Nj, Ni)
        north_first = ec.codes_get(h, "jScansPositively") == 0
        if gt == "regular_gg":
            la = np.sort(np.array(ec.codes_get_array(h, "distinctLatitudes"), float))
        else:
            la = np.sort(np.linspace(ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                                     ec.codes_get(h, "latitudeOfLastGridPointInDegrees"), Nj))
        lo0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        di = 360.0 / Ni if gt == "regular_gg" else ec.codes_get(h, "iDirectionIncrementInDegrees")
        keys = {}
        for k in ("shortName", "number", "level", "typeOfLevel", "stepRange", "units"):
            try:
                keys[k] = ec.codes_get(h, k)
            except Exception:                                          # noqa: BLE001
                pass
    finally:
        ec.codes_release(h)
    if north_first:                                                 # rows north -> south: flip to ascending
        v = v[::-1]
    lo = lo0 + di * np.arange(Ni)
    if abs(di * Ni - 360.0 - di) < 1e-3:
        lo, v = lo[:-1], v[:, :-1]
    return v, la, lo, keys


# ---- NOAA: idx byte ranges ---------------------------------------------------------------------------------------
def _get(url, rng=None, tries=5):
    import requests
    hd = {"User-Agent": "scorvec.com data pipeline (+https://scorvec.com)"}
    if rng:
        hd["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] is not None else ''}"
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=hd, timeout=(10, 120))
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}"
        except Exception as e:                                         # noqa: BLE001
            last = str(e)[:80]
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def idx_ranges(url: str, want) -> dict:
    """Parse a NOAA .idx and return {(var, level): (start, end_or_None)} for the wanted (var, level) pairs.
    `want` is a set of (var, level-string) - e.g. ("UGRD", "500 mb")."""
    txt = _get(url + ".idx")
    if txt is None:
        return {}
    lines = txt.decode().splitlines()
    rows = []
    for ln in lines:
        p = ln.split(":")
        rows.append((int(p[1]), p[3], p[4], p[5]))
    out = {}
    for i, (off, var, lev, fc) in enumerate(rows):
        if (var, lev) in want and (var, lev) not in out:
            end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
            out[(var, lev)] = (off, end)
    return out


def noaa_fetch(url: str, want, stats) -> dict:
    """{(var, level): decoded (v, lat, lon, keys)} for every wanted message present in the file."""
    rr = idx_ranges(url, set(want))
    res = {}

    def one(item):
        k, r = item
        b = _get(url, r)
        return k, b
    with ThreadPoolExecutor(8) as ex:
        for k, b in ex.map(one, rr.items()):
            if b is None:
                continue
            stats["bytes"] += len(b)
            res[k] = decode_one(b)
    return res


# ---- per-model step fetchers (run inside worker processes) ----------------------------------------------------------
def gfs_step(d, s, levels, stats):
    base = f"{S3['gfs']}/gfs.{d}/00/atmos/"
    want = [("PRES", "surface"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"), ("UFLX", "surface"),
            ("U-GWD", "surface"), ("PWAT", "entire atmosphere (considered as a single layer)")]
    if s == 0:
        want.append(("HGT", "surface"))
    f = noaa_fetch(base + f"gfs.t00z.sfluxgrbf{s:03d}.grib2", want, stats)
    T = None
    if all(k in f for k in want[:3]):
        T = {0: dict(grid=(f[want[0]][1], f[want[0]][2]), sp=f[want[0]][0], u10=f[want[1]][0], v10=f[want[2]][0],
                     tau_turb=f.get(("UFLX", "surface"), (None,))[0], tau_gwd=f.get(("U-GWD", "surface"), (None,))[0],
                     tcwv=f.get(want[5], (None,))[0])}
        if ("HGT", "surface") in f:
            T[0]["h"] = f[("HGT", "surface")][0]
    L = None
    if levels:
        lv = [("UGRD", f"{_fmt(l)} mb") for l in GFS_LEVELS] + [("VGRD", f"{_fmt(l)} mb") for l in GFS_LEVELS
                                                                 if l >= 10] + [("PRES", "surface")]
        g = noaa_fetch(base + f"gfs.t00z.pgrb2.0p25.f{s:03d}", lv, stats)
        L = _levels_from(g, GFS_LEVELS, 0)
    return T, L


def _fmt(l):
    return str(int(l)) if float(l).is_integer() else str(l)


def _levels_from(g, levels, m):
    us = [(l, g[("UGRD", f"{_fmt(l)} mb")]) for l in levels if ("UGRD", f"{_fmt(l)} mb") in g]
    if ("PRES", "surface") not in g or len(us) < len(levels):
        return None
    vs = {int(l) if float(l).is_integer() else l: g[("VGRD", f"{_fmt(l)} mb")][0]
          for l in levels if ("VGRD", f"{_fmt(l)} mb") in g}
    sp = g[("PRES", "surface")]
    return {m: dict(grid=(sp[1], sp[2]), sp=sp[0], levels=[float(l) for l, _ in us],
                    u=np.stack([x[0] for _, x in us]).astype(np.float32), v=vs)}


def gefs_step(d, s, levels, stats):
    T, L = {}, {}
    for m in ENS_MEMBERS:
        mem = "gec00" if m == 0 else f"gep{m:02d}"
        a = f"{S3['gefs']}/gefs.{d}/00/atmos/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{s:03d}"
        b = f"{S3['gefs']}/gefs.{d}/00/atmos/pgrb2bp5/{mem}.t00z.pgrb2b.0p50.f{s:03d}"
        wa = [("PRES", "surface"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
              ("PWAT", "entire atmosphere (considered as a single layer)")] + ([("HGT", "surface")] if s == 0 else [])
        wb = [("UFLX", "surface"), ("U-GWD", "surface")]                 # HGT is in pgrb2a at f000 only
        if levels:
            wa += [(v, f"{l} mb") for l in GEFS_A for v in ("UGRD", "VGRD")]
            wb += [(v, f"{l} mb") for l in GEFS_B for v in ("UGRD", "VGRD")]
        fa = noaa_fetch(a, wa, stats)
        fb = noaa_fetch(b, wb, stats)
        f = {**fa, **fb}
        if all(k in f for k in wa[:3]):
            T[m] = dict(grid=(f[wa[0]][1], f[wa[0]][2]), sp=f[wa[0]][0], u10=f[wa[1]][0], v10=f[wa[2]][0],
                        tau_turb=f.get(("UFLX", "surface"), (None,))[0], tau_gwd=f.get(("U-GWD", "surface"), (None,))[0],
                        tcwv=f.get(wa[3], (None,))[0])
            if ("HGT", "surface") in f:
                T[m]["h"] = f[("HGT", "surface")][0]
        if levels:
            lv = sorted(GEFS_A + GEFS_B)
            x = _levels_from(f, lv, m)
            if x:
                L.update(x)
    return T or None, (L or None) if levels else None


def aigfs_step(d, s, levels, stats):
    base = f"{S3['aigfs']}/aigfs.{d}/00/model/atmos/grib2/"
    w = [("PRES", "surface"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]
    f = noaa_fetch(base + f"aigfs.t00z.sfc.f{s:03d}.grib2", w, stats)
    T = None
    if all(k in f for k in w):
        T = {0: dict(grid=(f[w[0]][1], f[w[0]][2]), sp=f[w[0]][0], u10=f[w[1]][0], v10=f[w[2]][0])}
    L = None
    if levels and T:
        lv = [(v, f"{l} mb") for l in AIGFS_LEVELS for v in ("UGRD", "VGRD")]
        g = noaa_fetch(base + f"aigfs.t00z.pres.f{s:03d}.grib2", lv, stats)
        g[("PRES", "surface")] = f[w[0]]
        L = _levels_from(g, AIGFS_LEVELS, 0)
    return T, L


# (model path, stream, file kind) per source file. The open-data IFS ensemble file (-enfo-ef) holds the 50 perturbed
# members only; the control is published as the deterministic run (-oper-fc, 9 km, identical configuration since cycle
# 48r1), which is therefore member 0 of "ifs-ens" here (to 240 h only: the 00Z oper run stops at day 10).
EC = {"ifs-ens": [("ifs", "oper", "fc"), ("ifs", "enfo", "ef")], "aifs-ens": [("aifs-ens", "enfo", "cf"),
      ("aifs-ens", "enfo", "pf")], "aifs-single": [("aifs-single", "oper", "fc")]}


def ec_step(model, d, s, levels, stats):
    import rangefetch as rf
    sfc = ["sp", "10u", "10v", "tcw"] + (["ewss"] if model == "ifs-ens" else [])
    T, L = {}, {}
    for mdl, stream, kd in EC[model]:
        try:
            idx = rf.fetch_index(d, "00", mdl, s, kd, stream=stream)
        except Exception as ex:                                        # noqa: BLE001
            if getattr(ex, "not_found", False) and kd == "fc" and model == "ifs-ens":
                continue                                               # oper run beyond 240 h
            raise

        def mem_of(e):
            if kd in ("cf", "fc") or e.get("type") == "cf":
                return 0
            try:
                return int(e.get("number"))
            except (TypeError, ValueError):
                return -1
        want = []
        for e in idx:
            m = mem_of(e)
            if m not in SPEC[model]["members"]:
                continue
            lt, p = e.get("levtype"), e["param"]
            if lt == "sfc" and p in sfc:
                want.append(e)
            elif lt == "pl" and levels and p in ("u", "v") and int(e["levelist"]) in EC_LEVELS:
                want.append(e)
        if not want:
            continue
        want.sort(key=lambda e: e["_offset"])
        rng = rf.coalesce(want, max_gap=512 * 1024)
        blob = rf.fetch_ranges(rf.path_for(d, "00", mdl, s, kd, stream=stream) + ".grib2", rng, workers=8)
        stats["bytes"] += sum(r[1] - r[0] for r in rng)
        pos = 0
        for e in want:
            msg = blob[pos:pos + e["_length"]]
            pos += e["_length"]
            v, la, lo, _ = decode_one(msg)
            m = mem_of(e)
            if e.get("levtype") == "sfc":
                t = T.setdefault(m, {"grid": (la, lo)})
                t[{"sp": "sp", "10u": "u10", "10v": "v10", "tcw": "tcwv", "ewss": "tau_turb"}[e["param"]]] = v
            else:
                ll = L.setdefault(m, {"grid": (la, lo), "u": {}, "v": {}})
                ll[e["param"]][int(e["levelist"])] = v
    for m, ll in list(L.items()):
        if m not in T or "sp" not in T[m] or len(ll["u"]) < len(EC_LEVELS):
            del L[m]
            continue
        lev = sorted(ll["u"])
        L[m] = dict(grid=ll["grid"], sp=T[m]["sp"], levels=lev, u=np.stack([ll["u"][k] for k in lev]).astype(np.float32),
                    v=ll["v"])
    T = {m: t for m, t in T.items() if all(k in t for k in ("sp", "u10", "v10"))}
    return T or None, (L or None) if levels else None


def fetch_step(model, d, s, levels, stats):
    if model == "gfs":
        return gfs_step(d, s, levels, stats)
    if model == "gefs":
        return gefs_step(d, s, levels, stats)
    if model == "aigfs":
        return aigfs_step(d, s, levels, stats)
    return ec_step(model, d, s, levels, stats)


# ---- static orography ----------------------------------------------------------------------------------------------
def orography(model, d, stats):
    """The model's OWN orography (m) on its torque grid -> (h, lat, lon)."""
    if model in ("gfs", "gefs"):
        T, _ = fetch_step(model, d, 0, False, stats)
        t = T[0]
        return t["h"], t["grid"][0], t["grid"][1]
    if model == "aigfs":
        return aigfs_orog(d, stats)
    import rangefetch as rf
    mdl, stream, kind = {"ifs-ens": ("ifs", "oper", "fc"), "aifs-ens": ("aifs-ens", "enfo", "cf"),
                         "aifs-single": ("aifs-single", "oper", "fc")}[model]
    idx = rf.fetch_index(d, "00", mdl, 0, kind, stream=stream)
    e = [x for x in idx if x["param"] == "z" and x.get("levtype") == "sfc"]
    if not e and model == "aifs-single":                                # fall back to AIFS-ENS's static z
        idx = rf.fetch_index(d, "00", "aifs-ens", 0, "cf", stream="enfo")
        e = [x for x in idx if x["param"] == "z" and x.get("levtype") == "sfc"]
        mdl, stream, kind = "aifs-ens", "enfo", "cf"
    blob = rf.fetch_ranges(rf.path_for(d, "00", mdl, 0, kind, stream=stream) + ".grib2", rf.coalesce(e[:1]), workers=2)
    stats["bytes"] += len(blob)
    v, la, lo, _ = decode_one(blob)
    return v / C.G, la, lo


def aigfs_orog(d, stats):
    """'orog' (m) from the AIGFS init dataset (anemoi zarr, one chunk holds every variable of one time)."""
    import requests
    import re
    # the orography is static: when this init's dataset is not on the bucket (seen 2026-09-21), take a neighbour's
    zp = None
    d0 = dt.datetime.strptime(d, "%Y%m%d")
    for off in (0, -1, 1, -2, 2, -3, 3, -5, 5, -8, 8):
        dd = (d0 + dt.timedelta(days=off)).strftime("%Y%m%d")
        x = requests.get(f"{S3['aigfs']}/?list-type=2&prefix=aigfs.{dd}/00/model/atmos/init/&delimiter=/", timeout=60).text
        found = re.findall(r"<Prefix>(aigfs[^<]+\.zarr/)</Prefix>", x)
        if found:
            zp = found[0]
            break
    if zp is None:
        raise RuntimeError(f"no AIGFS init dataset near {d}")
    meta = requests.get(f"{S3['aigfs']}/{zp}.zmetadata", timeout=60).json()["metadata"]
    names = meta[".zattrs"]["variables"]
    za = meta["data/.zarray"]
    import numcodecs

    def chunk(name, key):
        za_ = meta[f"{name}/.zarray"]
        raw_ = requests.get(f"{S3['aigfs']}/{zp}{name}/{key}", timeout=600).content
        stats["bytes"] += len(raw_)
        dec = numcodecs.get_codec(za_["compressor"]).decode(raw_) if za_.get("compressor") else raw_
        return np.frombuffer(dec, dtype=za_["dtype"])
    arr = chunk("data", "1.0.0.0").reshape(za["chunks"])
    o = arr[0, names.index("orog"), 0].astype(np.float64)
    la = chunk("latitudes", "0").astype(np.float64)
    lo = chunk("longitudes", "0").astype(np.float64)
    fs = meta[".zattrs"]["field_shape"]
    o, la, lo = o.reshape(fs), la.reshape(fs)[:, 0], lo.reshape(fs)[0]
    if la[0] > la[-1]:
        o, la = o[::-1], la[::-1]
    if np.nanmax(o) > 20000.0:                                         # anemoi 'orog' here is geopotential (m2 s-2)
        o = o / C.G
    return o, la, lo


# ---- worker -------------------------------------------------------------------------------------------------------
_W = {}


def _winit(model, d, h, hlat, hlon):
    _W.update(model=model, d=d, h=h, hlat=hlat, hlon=hlon, red={})


def _red(grid, with_h):
    lat, lon = grid
    key = (len(lat), len(lon), float(lat[0]), float(lon[0]))
    if key not in _W["red"]:
        h = _W["h"] if with_h else np.zeros((len(lat), len(lon)))
        if with_h:
            assert len(lat) == len(_W["hlat"]) and np.allclose(lat, _W["hlat"]) and np.allclose(lon, _W["hlon"]), \
                "orography grid differs from the torque-field grid"
        _W["red"][key] = C.Reducer(C.Grid(lat, lon), h)
    return _W["red"][key]


def work(s):
    model, d = _W["model"], _W["d"]
    t0 = time.time()
    stats = {"bytes": 0}
    res = {"step": s, "T": {}, "L": {}, "chk": {}, "err": None}
    try:
        T, L = fetch_step(model, d, s, LEVELS_ON and s % 24 == 0, stats)
        mon = (dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=s)).month
        for m, t in (T or {}).items():
            red = _red(t["grid"], True)
            g = red.g
            sp = t["sp"] * (100.0 if np.nanmean(t["sp"]) < 2000 else 1.0)
            if np.isnan(sp).any() or np.isnan(t["u10"]).any():
                res["chk"][f"{m}:nan"] = 1
                continue
            tt, tg = t.get("tau_turb"), t.get("tau_gwd")
            if tt is not None and not np.isfinite(tt).all():
                tt = np.nan_to_num(tt)
            if tg is not None and not np.isfinite(tg).all():
                tg = np.nan_to_num(tg)
            first = m == min(T)
            res["T"][m] = red.torque_step(mon, sp, t["u10"], t["v10"], tau_turb=tt, tau_gwd=tg,
                                          tcwv=t.get("tcwv"), maps=s in C.MAP_STEPS and first, xmaps=first)
            msk = g.masks()
            wa = g.area[:, None]
            if tt is not None:                       # sign checks: stress ON the surface has the sign of the wind
                res["chk"][f"{m}:tt"] = float((tt * t["u10"] * wa * ~msk["land"]).sum())
            if tg is not None:
                res["chk"][f"{m}:tg"] = float((tg * t["u10"] * wa * msk["land"]).sum())
        for m, ll in (L or {}).items():
            red = _red(ll["grid"], False)
            sp = ll["sp"] * (100.0 if np.nanmean(ll["sp"]) < 2000 else 1.0)
            u = np.nan_to_num(ll["u"])
            v = {int(k) if float(k).is_integer() else k: np.nan_to_num(x) for k, x in ll["v"].items()}
            levs = [int(l) if float(l).is_integer() else l for l in ll["levels"]]
            res["L"][m] = red.level_step(levs, u, v, sp, lb=s in C.LB_STEPS, maps=s in C.MAP_STEPS and m == min(L))
    except Exception:                                                  # noqa: BLE001
        res["err"] = traceback.format_exc()[-800:]
    res["bytes"] = stats["bytes"]
    res["sec"] = time.time() - t0
    return res


# ---- one init -----------------------------------------------------------------------------------------------------
def process(model, d, out: Path):
    t0 = time.time()
    stats = {"bytes": 0}
    h, hlat, hlon = orography(model, d, stats)
    g = C.Grid(hlat, hlon)
    hm = float((h * g.area[:, None]).sum() / (g.area.sum() * len(hlon)))
    meta = {"model": model, "date": d, "bytes": stats["bytes"], "errors": [], "missing": [], "chk": {},
            "oro": {"area_mean_m": hm, "max_m": float(np.nanmax(h)), "nlat": len(hlat), "nlon": len(hlon),
                    "scales": C.orography_stats(g, h)}}
    print(f"  {model} {d}: orography {len(hlat)}x{len(hlon)} area-mean {hm:.1f} m max {np.nanmax(h):.0f} m", flush=True)
    steps = list(range(0, SPEC[model]["last"] + 1, DT))
    T, L = {}, {}
    with ProcessPoolExecutor(WORKERS, initializer=_winit, initargs=(model, d, h, hlat, hlon)) as ex:
        for r in ex.map(work, steps):
            meta["bytes"] += r["bytes"]
            if r["err"]:
                meta["errors"].append([r["step"], r["err"]])
                print(f"    step {r['step']} ERROR {r['err'][-400:]}", flush=True)
            for k, v in r["chk"].items():
                meta["chk"][k] = meta["chk"].get(k, 0.0) + v
            for m, o in r["T"].items():
                T.setdefault(m, {})[r["step"]] = o
            for m, o in r["L"].items():
                L.setdefault(m, {})[r["step"]] = o
            if not r["T"]:
                meta["missing"].append(r["step"])
            if r["step"] % 60 == 0:
                print(f"    step {r['step']}: {meta['bytes'] / 1e9:.2f} GB {time.time() - t0:.0f} s", flush=True)
    # own stresses: fix the sign convention (stress ON the surface, same sign as the low-level wind) and turn IFS
    # accumulations into 6-h interval means. Both are linear operations on the reduced torques.
    signs = {}
    for m in T:
        for key, pre in (("tt", "Tft"), ("tg", "Tg")):
            c = meta["chk"].get(f"{m}:{key}")
            if c is None:
                continue
            sg = 1.0 if c > 0 else -1.0
            if model == "ifs-ens" and key == "tt":
                sg = 1.0                             # ewss is accumulated, sign checked on the differences below
            signs[f"{m}:{key}"] = sg
            for s, o in T[m].items():
                for k in list(o):
                    if k.startswith(pre):
                        o[k] = np.asarray(o[k]) * sg
        if model == "ifs-ens":
            ss = sorted(T[m])
            acc = {s: {k: np.asarray(v) for k, v in T[m][s].items() if k.startswith("Tft")} for s in ss}
            for s in ss:
                for k in acc[s]:
                    if s == 0 or (s - DT) not in acc or k not in acc[s - DT]:
                        T[m][s].pop(k, None)
                    else:
                        T[m][s][k] = (acc[s][k] - acc[s - DT][k]) / (DT * 3600.0)
    meta["signs"] = signs
    if len(meta["errors"]) > 0.1 * len(steps) or not T:
        raise RuntimeError(f"{model} {d}: {len(meta['errors'])} of {len(steps)} steps failed - nothing saved")
    pack = {"grid|area_b": g.bands(g.area * len(hlon))}
    runs = []
    for m in sorted(T):
        run = model if SPEC[model]["members"] == [0] else f"{model}:{m}"
        runs.append(run)
        pack.update(C.pack_run(run, T[m], L.get(m, {}), interval_keys=("Tft", "Tg")))
    meta["runs"] = runs
    meta["n_T"] = {str(m): len(T[m]) for m in T}
    meta["n_L"] = {str(m): len(L.get(m, {})) for m in T}
    meta["seconds"] = round(time.time() - t0, 1)
    meta["complete"] = (not meta["errors"] and not meta["missing"] and len(runs) == len(SPEC[model]["members"])
                        and all(len(T[m]) == len(steps) and (not LEVELS_ON or len(L.get(m, {})) == len(steps[::4]))
                                for m in T))
    meta["last"] = LAST
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), **pack)
    print(f"  saved {out.relative_to(HERE)} {out.stat().st_size / 1e3:.0f} kB, {meta['bytes'] / 1e9:.2f} GB, "
          f"{meta['seconds']:.0f} s, complete={meta['complete']} signs={signs}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPEC))
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="output subdirectory suffix, e.g. d1 -> data/<model>_d1/")
    a = ap.parse_args()
    bad = 0
    for d in a.dates:
        out = C.DATA / (a.model + (f"_{a.tag}" if a.tag else "")) / f"{d}.npz"
        if out.exists() and not a.force:
            print(f"{a.model} {d}: exists")
            continue
        try:
            process(a.model, d, out)
        except Exception:                                              # noqa: BLE001
            traceback.print_exc()
            bad += 1
    return 1 if bad == len(a.dates) else 0


if __name__ == "__main__":
    sys.exit(main())
