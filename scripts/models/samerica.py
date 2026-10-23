#!/usr/bin/env python3
"""South America model viewer data for samerica-models.html: the IFS control and the GDPS to 10 days.

The IFS control is ECMWF's unperturbed 9 km forecast (open-data stream "oper"), read at 0.1 deg; the GDPS is ECCC's
15 km global model on its 0.15 deg latitude-longitude grid. Both are cropped to South America, 60 S - 15 N and
95 W - 30 W, and every field and forecast step becomes ONE lossless WebP on that native grid - the format of
models.html (scripts/models/models.py), drawn by the page with WebGL. Nothing is rendered here.

Fields (FIELDS below has the encodings; the page holds the same table):
  surface   2 m temperature and dewpoint (deg C), sea-level pressure, 10 m wind, 100 m wind (IFS), gust, precipitation
            rate with its type, accumulated precipitation and 10:1 snow, precipitable water, 925 and 850 hPa wind,
            850 hPa theta-e, and simulated IR (olr_tb: neither model publishes a simulated satellite band, so it is a
            window brightness temperature made from the outgoing longwave radiation)
  convection ECAPE and CAPE for the most-unstable and the 100 hPa mixed-layer parcels, the 0-6 km bulk shear vector
            and the storm-relative helicity 0-1 and 0-3 km, all computed here on every column by the site's ECAPE
            kernel (scripts/ecape/ecape_grid.cpp: SHARPlib, the sounding explorer's code), from the surface and the
            model's pressure levels above the ground; and the IFS's own most-unstable CAPE, for comparison.

Southern Hemisphere storms. A cyclonic supercell turns LEFT of the mean wind south of the equator. The kernel follows
the right-moving (Bunkers) storm, so a column south of the equator is handed its wind profile mirrored (v -> -v): the
right mover of the mirror image is the left mover of the real profile, the storm-relative wind speeds (the ECAPE inflow
term) are the same, and the helicity comes out with its sign flipped. ECAPE and the helicity are therefore those of
the cyclonic storm in both hemispheres, and the helicity is CYCLONIC-POSITIVE (the conventional Southern Hemisphere
value is its negative). The shear vector is turned back to the real frame before it is stored.

Meteograms and soundings come out of the same pass, while each step is in memory (nothing more is downloaded):
  points.json, points.bin   every airport of data/models_airports_sa.json (build_airports.py --region sa) on every step,
                            scripts/models/points.py's format and sampling (bilinear, CAPE and helicity nearest)
  snd/{ty}_{tx}.bin         the columns the kernel reads (surface + pressure levels, the winds NOT mirrored) every
                            MODELS[...]["snd_d"]-th grid point, in scripts/ecape/to_soundings.py's tiles (its
                            pressure-level mode), for the sounding explorer; the manifest's "snd" is their index

Column heights are hypsometric (virtual temperature, trapezoidal in ln p) up from the model's own terrain height and
surface pressure: the geopotential of every level would add ~15 % to the download, and each run checks the result
against the model's own 500 and 850 hPa heights on a few steps (manifest "checks", with the ECAPE kernel's MU CAPE
against the model's own and, for the IFS, the column precipitable water against its tcwv).

Sources (hard rules):
  ifs    ECMWF open data from the Google Cloud mirror ONLY, through scripts/ecmwf/rangefetch.py (byte ranges of only the
         wanted messages; it refuses any other host). 00/12Z, 3-hourly to +144 h then 6-hourly to +240 h; the 0.1 deg
         files are complete on the mirror ~9 h 40 min after the start (cycles.IFS_CONTROL_0P1).
  gdps   ECCC MSC Datamart (dd.weather.gc.ca): one global GRIB2 file per variable, level and hour, JPEG2000-packed, so a
         file cannot be cut to the domain and is read whole. 00/12Z to +240 h: surface fields every 3 h (the
         precipitation type, and so the rate map it colours, 6-hourly after +168 h: GDPS_TYPE_6H), the column (20
         pressure levels) on the IFS's steps. Readiness is a HEAD on the last hour's files, never a listing. A file
         missing from a cycle the plan found complete is not retried: the step fails and the cycle is not published.

    python scripts/models/samerica.py plan --model ifs --manifests assets/models
    python scripts/models/samerica.py run --model gdps --cycle 2026102300 --site . --procs 2 --lists /tmp/sa
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS / "ecmwf"))
sys.path.insert(0, str(SCRIPTS / "ecape"))
import models as MV  # the viewer's quantisers and WebP writer (models.html decodes the same codes)
import points as PTS  # meteogram point series
import to_models as EC  # the ECAPE grids' square-root code
import to_soundings as SND  # the forecast-sounding tiles

UA = {"User-Agent": "scorvec-models/1.0 (+https://scorvec.com/samerica-models.html)"}
DATA = "assets/models/data/sa"
DOMAIN = dict(lat=(-60.0, 15.0), lon=(-95.0, -30.0))
G0, RD, SIGMA = 9.80665, 287.04, 5.670374419e-8
STEPS = tuple(range(0, 144, 3)) + tuple(range(144, 241, 6))  # the IFS control's open-data steps to day 10
MISSING = -9999.0  # ecape_grid's missing value

# ── encoding (codes 0-254 data, 255 = missing); samerica-models.html holds the same table ─────────────────────────────
#   ir       K, 170 + 0.625 * code: simulated IR, the window brightness temperature from the OLR (olr_tb)
#   t2m, td2m  deg C, per-frame offset (manifest "q"), 0.5 C steps (coarser if the frame spans more than 127 C)
#   mslp     R = 940 + 0.5 hPa, G = 1 where the surface pressure is below 850 hPa (no highs or lows placed there)
#   wind10, wind100, wind925, wind850  R, G = earth-relative u, v, -50.8 + 0.4 m/s; 255 where the level is less
#            than 5 hPa above the ground (925, 850 hPa: the columns' rule)
#   gust     m/s, 0.25 m/s steps
#   rate     R = precipitation rate, log 0.1-100 mm/h (0 = below 0.1 or no type), G = type 1 rain, 2 snow, 3 freezing
#            rain or drizzle, 4 ice pellets, 5 rain-snow mix (the models.html codes)
#   apcp     mm since the start of the run, 16-bit (R high byte, G low), 0.1 mm
#   asnow    cm of snow at 10:1 from the snow water equivalent since the start, 16-bit, 0.1 cm
#   pwat     mm, 0.5 mm steps
#   the850   K, 270 + 0.5 * code; 255 where 850 hPa is within 10 hPa of the ground or below it
#   ecape_mu, ecape_ml, mucape, mlcape, capem   J/kg, code = round(2.8 sqrt(J/kg)) (scripts/ecape/to_models.py)
#   shr6     R, G = 0-6 km bulk shear vector u, v, earth-relative, -63.5 + 0.5 m/s
#   srh1, srh3  m2/s2, cyclonic-positive (see above), -250 + 5 * code
FIELDS = {
    "ir": dict(enc="lin", off=170.0, step=0.625),
    "t2m": dict(enc="linf"),
    "td2m": dict(enc="linf"),
    "mslp": dict(enc="mslp", off=940.0, step=0.5),
    "wind10": dict(enc="lin2", off=-50.8, step=0.4),
    "wind100": dict(enc="lin2", off=-50.8, step=0.4),
    "wind925": dict(enc="lin2", off=-50.8, step=0.4),
    "wind850": dict(enc="lin2", off=-50.8, step=0.4),
    "gust": dict(enc="lin", off=0.0, step=0.25),
    "rate": dict(enc="rate2"),
    "apcp": dict(enc="acc16", unit=0.1),
    "asnow": dict(enc="acc16", unit=0.1),
    "pwat": dict(enc="lin", off=0.0, step=0.5),
    "the850": dict(enc="lin", off=270.0, step=0.5),
    **{f: dict(enc="sqrt", k=EC.K) for f in ("ecape_mu", "ecape_ml", "mucape", "mlcape", "capem")},
    "shr6": dict(enc="lin2", off=-63.5, step=0.5),
    "srh1": dict(enc="lin", off=-250.0, step=5.0),
    "srh3": dict(enc="lin", off=-250.0, step=5.0),
}
RATE_LO, RATE_HI = 0.1, 100.0
# IFS ptype (ECMWF parameter 260015) -> the viewer's codes; wet snow counts as snow, freezing drizzle as freezing rain
IFS_PTYPE = {1: 1, 3: 3, 5: 2, 6: 2, 7: 5, 8: 4, 12: 3}

MODELS = {
    "ifs": dict(
        label="IFS control",
        source="ECMWF IFS control (the unperturbed forecast, 9 km) at 0.1 deg: ECMWF open data, Google Cloud mirror",
        levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50),
        sfc_hours=STEPS,
        conv_hours=STEPS,
        snd_d=4,  # soundings every 4th column: 0.4 deg
    ),
    "gdps": dict(
        label="GDPS",
        source="ECCC GDPS 15 km on its 0.15 deg grid: MSC Datamart",
        # dense below 850 hPa (the boundary layer, the mixed-layer parcel, the 0-1 km helicity), 100 hPa apart aloft
        levels=(1015, 1000, 985, 970, 950, 925, 900, 850, 800, 750, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50),
        sfc_hours=tuple(range(0, 241, 3)),
        conv_hours=STEPS,
        snd_d=3,  # 0.45 deg
    ),
}
QC_HOURS = (0, 120, 240)  # steps with the model's own 500/850 hPa heights (and the GDPS's own CAPE) fetched as a check
METHOD = {
    "ifs": dict(
        ir="simulated IR: the window brightness temperature from the outgoing longwave radiation (top thermal radiation, ttr, its mean since the previous step: 3 h to +144 h, 6 h after), by the Ohring-Gruber relation",
        gust="the largest gust since the previous step: 10fg3 (3 h) on the 3-hourly steps, 10fg (6 h) on the 6-hourly ones",
        apcp="tp, the total since the start of the run",
        asnow="10 x sf, the snowfall water equivalent since the start of the run (10:1)",
        pwat="tcwv, the model's total column water vapour",
        rate="tprate, instantaneous, coloured by ptype",
        capem="mucape, the most-unstable CAPE the IFS computes on its own model levels",
    ),
    "gdps": dict(
        ir="simulated IR: the window brightness temperature from the instantaneous outgoing longwave radiation at the top of the atmosphere, by the Ohring-Gruber relation",
        gust="WindGust, instantaneous",
        apcp="Precip-Accum, the total since the start of the run",
        asnow="10 x Snow-Accum, the snow water equivalent since the start of the run (10:1)",
        pwat="computed here from the surface and 20 pressure levels (SHARPlib)",
        rate="PrecipRate, instantaneous, coloured by PrecipType",
    ),
}


# ── grids ───────────────────────────────────────────────────────────────────────────────────────────────────────────
_CROP: dict = {}
_CROP_LOCK = threading.Lock()


def crop_index(ni, nj, la0, la1, lo0, di):
    """(rows, cols, lat, lon) selecting DOMAIN from a global lat-lon grid, rows south -> north and columns west -> east,
    lon in -180..180. Cached per grid."""
    key = (ni, nj, round(la0, 4), round(la1, 4), round(lo0, 4), round(di, 5))
    with _CROP_LOCK:
        if key in _CROP:
            return _CROP[key]
    lat = np.linspace(la0, la1, nj)
    lon = ((lo0 + di * np.arange(ni)) + 180.0) % 360.0 - 180.0
    eps = di * 1e-3
    jj = np.where((lat >= DOMAIN["lat"][0] - eps) & (lat <= DOMAIN["lat"][1] + eps))[0]
    jj = jj[np.argsort(lat[jj])]
    ii = np.where((lon >= DOMAIN["lon"][0] - eps) & (lon <= DOMAIN["lon"][1] + eps))[0]
    ii = ii[np.argsort(lon[ii])]
    if jj.size < 2 or ii.size < 2 or np.any(np.diff(lon[ii]) <= 0):
        raise ValueError(f"grid {key} does not cover the domain in one piece")
    out = (jj, ii, lat[jj], lon[ii])
    with _CROP_LOCK:
        _CROP[key] = out
    return out


def decode(msg):
    """One GRIB message -> (values cropped to DOMAIN, rows south -> north, float32; units; grid key tuple)."""
    import eccodes as ec

    g = ec.codes_new_from_message(msg)
    try:

        def gv(k):
            return ec.codes_get(g, k)

        if gv("gridType") != "regular_ll":
            raise ValueError(f"unexpected grid {gv('gridType')}")
        ni, nj = gv("Ni"), gv("Nj")
        la0, la1 = gv("latitudeOfFirstGridPointInDegrees"), gv("latitudeOfLastGridPointInDegrees")
        lo0, di = gv("longitudeOfFirstGridPointInDegrees"), gv("iDirectionIncrementInDegrees")
        if gv("iScansNegatively") or gv("jPointsAreConsecutive"):
            raise ValueError("unexpected scanning mode")
        try:
            units = gv("units")
        except Exception:  # noqa: BLE001
            units = ""
        v = ec.codes_get_values(g).reshape(nj, ni)
        if gv("bitmapPresent"):
            v = np.where(v == gv("missingValue"), np.nan, v)
    finally:
        ec.codes_release(g)
    jj, ii, lat, lon = crop_index(ni, nj, la0, la1, lo0, di)
    return v[np.ix_(jj, ii)].astype(np.float32), units, (lat, lon, float(di))


def grid_desc(lat, lon, di):
    """The page's grid: a regular lat-lon window, row j = 0 at the SOUTH (la1), column i = 0 at the WEST (lo1)."""
    return dict(
        type="ll",
        nx=int(lon.size),
        ny=int(lat.size),
        lo1=round(float(lon[0]), 6),
        la1=round(float(lat[0]), 6),
        di=round(float(di), 6),
        dj=round(float(lat[1] - lat[0]), 6),
    )


def split_grib(blob):
    """Concatenated GRIB2 messages -> list of messages (rangefetch returns the wanted messages back to back)."""
    out, o = [], 0
    while o < len(blob):
        k = blob.find(b"GRIB", o)
        if k < 0:
            break
        n = int.from_bytes(blob[k + 8 : k + 16], "big")
        out.append(blob[k : k + n])
        o = k + n
    return out


# ── transport ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.n = self.b = 0
        self.secs = 0.0  # the step's fetch time (the rest is decoding, the kernel and encoding)

    def add(self, nbytes, requests=1):
        with self.lock:
            self.n += requests
            self.b += nbytes


def ifs_select(idx, sfc_file, want_sfc, want_pl, levels):
    """The .index entries of the wanted messages of one 0.1 deg file, in file order. want_pl holds parameters wanted on
    every level in `levels` and (parameter, level) pairs wanted on that level alone."""
    if sfc_file:
        sel = [e for e in idx if e.get("levtype") == "sfc" and e.get("param") in want_sfc]
    else:
        sel = [
            e
            for e in idx
            if e.get("levtype") == "pl"
            and (
                (e.get("param") in want_pl and int(e.get("levelist", 0)) in levels)
                or (e.get("param"), int(e.get("levelist", 0))) in want_pl
            )
        ]
    return sorted(sel, key=lambda e: e["_offset"])


# ranges merged across gaps under 256 KB: the 0.1 deg messages are 1-5 MB each, so rangefetch's default 3 MB gap would
# bring whole unwanted ones along
IFS_GAP = 256 * 1024


def ifs_fetch(date, hh, step, want_sfc, want_pl, levels, stats=None):
    """{(param, level or None): message} for one step of the IFS control at 0.1 deg, from the Google mirror only.
    Raises rangefetch.IndexUnavailable when the step is not on the mirror."""
    import rangefetch as rf

    out = {}
    for rel, idx in rf.oper_files(date, hh, "ifs", step, "0p1"):
        sel = ifs_select(idx, rel.endswith("-sfc.grib2"), want_sfc, want_pl, levels)
        if not sel:
            continue
        ranges = rf.coalesce(sel, max_gap=IFS_GAP)
        blob = rf.fetch_ranges(rel, ranges, workers=8)
        msgs = split_grib(blob)
        if [len(m) for m in msgs] != [e["_length"] for e in sel]:
            raise RuntimeError(f"{rel}: {len(msgs)} messages for {len(sel)} index entries")
        if stats:
            stats.add(sum(b - a for a, b, _ in ranges), len(ranges))
        for e, m in zip(sel, msgs):
            lev = int(e["levelist"]) if e.get("levtype") == "pl" else None
            out[(e["param"], lev)] = m
    return out


def gdps_url(date, hh, hour, var):
    return (
        f"https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{hh}/{hour:03d}/"
        f"{date}T{hh}Z_MSC_GDPS_{var}_LatLon0.15_PT{hour:03d}H.grib2"
    )


class Missing(RuntimeError):
    """A file that is not (yet) published."""


def http_get(url, stats=None, tries=5):
    """A whole GRIB2 file, validated (GRIB ... 7777); Missing on 404. The Datamart drops connections under load: retried
    with a growing pause."""
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                body = r.read()
            if body[:4] == b"GRIB" and body[-4:] == b"7777":
                if stats:
                    stats.add(len(body))
                return body
            last = RuntimeError(f"not a whole GRIB message ({len(body)} B)")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise Missing(url) from None
            last = e
        except Exception as e:  # noqa: BLE001  the network: back off, then try again
            last = e
        time.sleep(5 + 10 * k)
    raise RuntimeError(f"{url}: {last}")


def head_ok(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


GDPS_SFC = {
    "t2m": "AirTemp_AGL-2m",
    "td2m": "DewPoint_AGL-2m",
    "u10": "WindU_AGL-10m",
    "v10": "WindV_AGL-10m",
    "gust": "WindGust_AGL-10m",
    "psfc": "Pressure_Sfc",
    "mslp": "Pressure_MSL",
    "apcp": "Precip-Accum_Sfc",
    "swe": "Snow-Accum_Sfc",
    "rate": "PrecipRate_Sfc",
    "ptype": "PrecipType-Instant_Sfc",
    "olr": "UpwardLongwaveRadiationFlux_NTAtm",
}
GDPS_COL = {
    "t": "AirTemp_IsbL-{:04d}",
    "q": "SpecificHumidity_IsbL-{:04d}",
    "u": "WindU_IsbL-{:04d}",
    "v": "WindV_IsbL-{:04d}",
}


GDPS_NOT_AT_0 = ("olr", "apcp", "swe", "rate", "ptype")  # no accumulation, rate or type (or OLR) at the analysis time
# PrecipType-Instant is published every 6 h after +168 h (+171, +177, ... are absent; HEAD probes of every GDPS_SFC file
# and the 1000 hPa temperature at +165 to +240 h, 2026-10-23: only the type, and CAPE_Sfc, thin out). The rate map is
# coloured by the type, so it follows the type's steps there; every other field stays 3-hourly.
GDPS_TYPE_6H = 168


def gdps_jobs(date, hh, hour, conv, qc):
    """{key: url} for one GDPS hour: the surface set, the column on convective hours, the checks on QC hours."""
    typed = hour <= GDPS_TYPE_6H or hour % 6 == 0
    jobs = {
        k: gdps_url(date, hh, hour, v)
        for k, v in GDPS_SFC.items()
        if (hour or k not in GDPS_NOT_AT_0) and (typed or k not in ("rate", "ptype"))
    }
    if hour == 0:
        jobs["zsfc"] = gdps_url(date, hh, 0, "GeopotentialHeight_Sfc")
    if conv:
        for p in MODELS["gdps"]["levels"]:
            for k, v in GDPS_COL.items():
                jobs[(k, p)] = gdps_url(date, hh, hour, v.format(p))
    if qc:
        for p in (500, 850):
            jobs[("gh", p)] = gdps_url(date, hh, hour, f"GeopotentialHeight_IsbL-{p:04d}")
        jobs["capem"] = gdps_url(date, hh, hour, "CAPE_Sfc")
    return jobs


def gdps_fetch(date, hh, hour, conv, qc, stats=None, workers=6):
    """{key: message} for one GDPS hour; a file that is missing raises Missing (the step fails)."""
    jobs = gdps_jobs(date, hh, hour, conv, qc)
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(jobs, ex.map(lambda u: http_get(u, stats), jobs.values())))


# ── physics ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def kelvin(t):
    return t + 273.15 if np.nanmean(t) < 150 else t


def metres(z, units, expect):
    """Geopotential height in metres whether the file holds gpm or decametres (ECCC writes some in dam): its units, or
    failing that whichever reading puts the domain's `expect` statistic (median or maximum, m) nearer the truth."""
    if units.lower() in ("dam", "decametre", "decametres"):
        return z * 10.0
    ref, want = expect
    got = float(np.nanmax(z) if ref == "max" else np.nanmedian(z))
    return z * 10.0 if abs(got * 10 - want) < abs(got - want) else z


def pa(p):
    return p * 100.0 if np.nanmedian(p) < 2000 else p


OG_A, OG_B = 1.106e-3, 1.228  # Ohring, Gruber and Ellingson (1984): flux temperature Tf = Tb (OG_B - OG_A Tb)


def olr_tb(olr):
    """Simulated IR: the window brightness temperature (K) from the outgoing longwave radiation (W m-2), the inverse of the
    Ohring-Gruber relation (as the site's GDPS simulated-IR loop, scripts/sst/gdps_eq_charts.py). The flux
    temperature (OLR / sigma)^(1/4) alone runs ~20 K cold over clear ground (water vapour absorbs outside the window);
    over deep cloud the two agree."""
    with np.errstate(invalid="ignore"):
        tf = (np.maximum(olr, 1.0) / SIGMA) ** 0.25
    return (OG_B - np.sqrt(np.clip(OG_B**2 - 4.0 * OG_A * tf, 0.0, None))) / (2.0 * OG_A)


def q_from_td(td_k, p):
    """Specific humidity from the dewpoint (Bolton 1980), the inverse of ecape_grid.cpp's dewpoint_from_q."""
    e = 611.2 * np.exp(17.67 * (td_k - 273.15) / (td_k - 29.65))
    return 0.621981 * e / (p - 0.378 * e)


def theta_e(t, q, p):
    """Equivalent potential temperature (Bolton 1980, eqs. 15, 39 and 43); t K, q kg/kg, p Pa."""
    r = q / (1.0 - q)
    e = np.maximum(p * r / (0.621981 + r), 1e-3)
    lg = np.log(e / 611.2)
    td = 243.5 * lg / (17.67 - lg) + 273.15
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0
    rg = 1000.0 * r
    th = t * (100000.0 / p) ** (0.2854 * (1.0 - 0.28e-3 * rg))
    return th * np.exp((3.376 / tl - 0.00254) * rg * (1.0 + 0.81e-3 * rg))


def column(sfc, levs, plevs_hpa):
    """The kernel's scratch (6, 1 + L, ny, nx): level 0 = the surface (pressure, terrain height, 2 m T and humidity, 10 m
    wind), then the pressure levels in decreasing pressure, NaN where a level is less than 5 hPa above the ground.
    Heights hypsometric from the surface. sfc: dict p (Pa), z (m), t (K), q, u, v; levs: {p_hpa: dict t, q, u, v}."""
    ny, nx = sfc["p"].shape
    L = len(plevs_hpa)
    out = np.full((6, 1 + L, ny, nx), np.nan, np.float32)
    out[:, 0] = [sfc[k] for k in ("p", "z", "t", "q", "u", "v")]
    pp, tv, zz = sfc["p"].astype(np.float64), sfc["t"] * (1 + 0.608 * sfc["q"]), sfc["z"].astype(np.float64)
    for n, ph in enumerate(sorted(plevs_hpa, reverse=True)):
        lv, p = levs[ph], ph * 100.0
        up = p < sfc["p"] - 500.0
        tvk = lv["t"] * (1 + 0.608 * lv["q"])
        zk = zz + RD / G0 * 0.5 * (tv + tvk) * np.log(pp / p)
        k = 1 + n
        out[0, k] = np.where(up, p, np.nan)
        out[1, k] = np.where(up, zk, np.nan)
        for v, key in ((2, "t"), (3, "q"), (4, "u"), (5, "v")):
            out[v, k] = np.where(up, lv[key], np.nan)
        pp, tv, zz = np.where(up, p, pp), np.where(up, tvk, tv), np.where(up, zk, zz)
    return out


def run_kernel(kernel, scratch, workdir):
    """ecape_grid on the scratch (compact pressure-level columns, with the kinematics) -> ({name: (ny, nx)}, meta)."""
    _, nl, ny, nx = scratch.shape
    stem = Path(workdir) / "col"
    scratch.astype(np.float32).tofile(f"{stem}.f32")
    Path(f"{stem}.json").write_text(json.dumps(dict(nlev=nl, ny=ny, nx=nx, compact=1, kin=1)))
    r = subprocess.run([str(kernel), str(stem)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"ecape_grid failed ({r.returncode}): {r.stderr[-400:]}")
    em = json.loads(Path(f"{stem}_ecape.json").read_text())
    arr = np.fromfile(f"{stem}_ecape.f32", np.float32).reshape(em["shape"])
    for suf in (".f32", ".json", "_ecape.f32", "_ecape.json"):
        Path(f"{stem}{suf}").unlink(missing_ok=True)
    out = {n: np.where(arr[i] <= MISSING + 1, np.nan, arr[i]) for i, n in enumerate(em["fields"])}
    return out, dict(ok=em["columns_ok"], skipped=em["columns_skipped"])


def cape_check(ours, theirs):
    """Our MU CAPE against the model's own, where either is 100 J/kg or more."""
    m = np.isfinite(ours) & np.isfinite(theirs) & ((ours >= 100) | (theirs >= 100))
    if m.sum() < 50:
        return dict(n=int(m.sum()))
    a, b = ours[m].astype(np.float64), theirs[m].astype(np.float64)
    both = (a >= 100) & (b >= 100)
    rat = a[both] / b[both]
    return dict(
        n=int(m.sum()),
        r=round(float(np.corrcoef(a, b)[0, 1]), 3),
        median_ratio=round(float(np.median(rat)), 3) if rat.size else None,
        within_x1_5=round(float(np.mean((rat >= 1 / 1.5) & (rat <= 1.5))), 3) if rat.size else None,
        p99_ours=round(float(np.percentile(a, 99))),
        p99_model=round(float(np.percentile(b, 99))),
    )


def diff_check(ours, theirs):
    m = np.isfinite(ours) & np.isfinite(theirs)
    d = (ours[m] - theirs[m]).astype(np.float64)
    if not d.size:
        return dict(n=0)
    return dict(
        n=int(d.size),
        bias=round(float(d.mean()), 2),
        mae=round(float(np.abs(d).mean()), 2),
        p99_abs=round(float(np.percentile(np.abs(d), 99)), 2),
    )


# ── encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────
def encode(name, x):
    """-> (uint8 image, row 0 at the NORTH; per-frame params or None). x arrays have row 0 at the south."""
    f = FIELDS[name]
    e = f["enc"]
    if e == "sqrt":
        return EC.encode_cape(x), None  # NaN (a skipped column) -> 255
    if e == "lin":
        return MV.q_lin(x, f["off"], f["step"])[::-1], None
    if e == "linf":
        c, q = MV.q_linf(x)
        return c[::-1], q
    if e == "acc16":
        v = np.clip(np.rint(np.nan_to_num(np.maximum(x, 0.0)) / f["unit"]), 0, 65535).astype(np.uint16)
        return np.dstack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), np.zeros(v.shape, np.uint8)])[
            ::-1
        ], None
    z = np.zeros(np.shape(x[0]), np.uint8)  # the empty third channel of a two-channel frame (x = a pair of grids)
    if e == "lin2":
        u, v = x
        bad = ~(np.isfinite(u) & np.isfinite(v))
        cu, cv = (
            MV.q_lin(np.where(bad, np.nan, u), f["off"], f["step"]),
            MV.q_lin(np.where(bad, np.nan, v), f["off"], f["step"]),
        )
        return np.dstack([cu, cv, z])[::-1], None
    if e == "mslp":
        hpa, high = x
        return np.dstack([MV.q_lin(hpa, f["off"], f["step"]), high.astype(np.uint8), z])[::-1], None
    if e == "rate2":
        mmh, t = x
        r = MV.q_log(mmh, RATE_LO, RATE_HI)
        r = np.where((r == 255) | (t == 0) | ~(np.nan_to_num(mmh) >= RATE_LO), 0, r).astype(np.uint8)
        return np.dstack([r, np.where(r > 0, t, 0).astype(np.uint8), z])[::-1], None
    raise ValueError(e)


def write_field(outdir, name, step, x):
    arr, q = encode(name, x)
    b = MV.webp(arr)
    d = Path(outdir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{step:03d}.webp").write_bytes(b)
    return q, len(b)


# ── one step (runs in a worker process: fetch, decode, compute, encode) ──────────────────────────────────────────────
_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)
    os.environ["OMP_NUM_THREADS"] = str(ctx["omp"])


def ifs_want(step):
    """(surface parameters, pressure-level parameters, levels) one IFS step reads: the surface set (the accumulations,
    the gust and the top thermal radiation from +3 h; the terrain at step 0), t, q, u and v on every level of the
    columns, and the 500 / 850 hPa heights on the check steps."""
    sfc = {"2t", "2d", "sp", "msl", "10u", "10v", "100u", "100v", "tcwv", "ptype", "tprate", "mucape"}
    # the gust is 10fg3 (the last 3 h) on the 3-hourly steps and 10fg (since the previous step, 6 h) on the 6-hourly ones
    sfc |= {"10fg3", "10fg", "tp", "sf", "ttr"} if step else {"z"}
    conv = step in MODELS["ifs"]["conv_hours"]
    pl = {"t", "q", "u", "v"} if conv else {("t", 850), ("q", 850)}
    if step in QC_HOURS:
        pl |= {("gh", 500), ("gh", 850)}
    return sfc, pl, MODELS["ifs"]["levels"] if conv else ()


def step_ifs(date, hh, step, outdir):
    conv, qc = step in MODELS["ifs"]["conv_hours"], step in QC_HOURS
    levels = MODELS["ifs"]["levels"]
    st, t0 = Stats(), time.time()
    blobs = ifs_fetch(date, hh, step, *ifs_want(step), st)
    st.secs = time.time() - t0
    dec, units, grid = {}, {}, None
    for key, m in blobs.items():
        dec[key], units[key], grid = decode(m)
    del blobs

    def S(p):
        return dec[(p, None)]

    lat = grid[0]
    zsfc = S("z") / G0 if step == 0 else _CTX["zsfc"]
    ps = S("sp")
    t2, td2 = S("2t"), S("2d")
    phys = {
        "t2m": t2 - 273.15,
        "td2m": td2 - 273.15,
        "mslp": (S("msl") / 100.0, ps < 85000.0),
        "wind10": (S("10u"), S("10v")),
        "wind100": (S("100u"), S("100v")),
        "pwat": S("tcwv"),
        "capem": S("mucape"),
        "apcp": np.maximum(S("tp"), 0) * 1000.0 if step else np.zeros_like(t2),
        "asnow": np.maximum(S("sf"), 0) * 1000.0 if step else np.zeros_like(t2),  # mm of water = cm of snow at 10:1
    }
    gust = dec.get(("10fg3", None), dec.get(("10fg", None)))
    if step and gust is not None:
        phys["gust"] = gust
    code = np.rint(np.nan_to_num(S("ptype"))).astype(int)
    pt = np.zeros(code.shape, np.uint8)
    for k, v in IFS_PTYPE.items():
        pt[code == k] = v
    phys["rate"] = (np.maximum(S("tprate"), 0) * 3600.0, pt)
    t850, q850 = dec[("t", 850)], dec[("q", 850)]
    phys["the850"] = np.where(ps >= 86000.0, theta_e(t850, q850, 85000.0), np.nan)
    if step:
        np.save(Path(outdir) / f"_ttr_{step:03d}.npy", S("ttr"))
    checks = {}
    if conv:
        q2 = q_from_td(td2, ps)
        sfc_col = dict(p=ps, z=zsfc, t=t2, q=q2, u=S("10u"), v=S("10v"))
        levs = {p: {k: dec[(k, p)] for k in ("t", "q", "u", "v")} for p in levels}
        phys.update(winds_aloft(ps, levs))
        k_out, kmeta = convective(sfc_col, levs, levels, lat, outdir)
        phys.update(k_out)
        checks["kernel"] = kmeta
        checks["mucape_vs_ifs"] = cape_check(phys["mucape"], S("mucape"))
        checks["pwat_vs_tcwv"] = diff_check(k_out["_pwat"], S("tcwv"))
        checks.update(soundings("ifs", step, dict(sfc_col, td=td2), levs, outdir, qc, dec, units))
    return phys, checks, st, grid, (zsfc if step == 0 else None)


def step_gdps(date, hh, hour, outdir):
    conv, qc = hour in MODELS["gdps"]["conv_hours"], hour in QC_HOURS
    levels = MODELS["gdps"]["levels"]
    st, t0 = Stats(), time.time()
    blobs = gdps_fetch(date, hh, hour, conv, qc, st)
    st.secs = time.time() - t0
    dec, units, grid = {}, {}, None
    for key, m in blobs.items():
        dec[key], units[key], grid = decode(m)
    del blobs
    lat = grid[0]
    # the terrain: the Andes reach ~6,000 m inside the domain
    zsfc = metres(dec["zsfc"], units["zsfc"], ("max", 6000.0)) if hour == 0 else _CTX["zsfc"]
    ps = pa(dec["psfc"])
    t2, td2 = kelvin(dec["t2m"]), kelvin(dec["td2m"])
    zero = np.zeros_like(t2)
    swe = np.maximum(dec["swe"], 0) * (1000.0 if units["swe"].lower() == "m" else 1.0) if hour else zero
    phys = {
        "t2m": t2 - 273.15,
        "td2m": td2 - 273.15,
        "mslp": (pa(dec["mslp"]) / 100.0, ps < 85000.0),
        "wind10": (dec["u10"], dec["v10"]),
        "gust": dec["gust"],
        "apcp": np.maximum(dec["apcp"], 0) if hour else zero,
        "asnow": swe,  # mm of snow water = cm of snow at 10:1
    }
    checks = {}
    if hour:
        phys["ir"] = olr_tb(dec["olr"])
    if "ptype" in dec:  # not at the analysis time, 6-hourly after +168 h (GDPS_TYPE_6H)
        code = np.rint(np.nan_to_num(dec["ptype"], nan=6)).astype(int)
        pt = np.zeros(code.shape, np.uint8)
        for k, v in MV.RDPS_PTYPE.items():  # the GEM codes as measured for the RDPS (models.py)
            pt[code == k] = v
        phys["rate"] = (np.maximum(dec["rate"], 0) * 3600.0, pt)
        # the code table is the RDPS's as measured, not GRIB 4.201: the 2 m temperature under each code says if it holds
        checks["ptype_t2m_median_c"] = {
            str(k): round(float(np.median(t2[code == k] - 273.15)), 1) for k in range(1, 7) if (code == k).sum() >= 50
        }
    if conv:
        levs = {p: dict(t=kelvin(dec[("t", p)]), q=dec[("q", p)], u=dec[("u", p)], v=dec[("v", p)]) for p in levels}
        if np.nanmax(levs[850]["q"]) > 0.5:
            raise RuntimeError("GDPS specific humidity is not in kg/kg")
        phys["the850"] = np.where(ps >= 86000.0, theta_e(levs[850]["t"], levs[850]["q"], 85000.0), np.nan)
        phys.update(winds_aloft(ps, levs))
        sfc_col = dict(p=ps, z=zsfc, t=t2, q=q_from_td(td2, ps), u=dec["u10"], v=dec["v10"])
        k_out, kmeta = convective(sfc_col, levs, levels, lat, outdir)
        phys.update(k_out)
        phys["pwat"] = k_out["_pwat"]
        checks["kernel"] = kmeta
        checks.update(soundings("gdps", hour, dict(sfc_col, td=td2), levs, outdir, qc, dec, units))
        if qc:
            checks["mucape_vs_gdps_cape"] = cape_check(phys["mucape"], dec["capem"])
    return phys, checks, st, grid, (zsfc if hour == 0 else None)


def convective(sfc, levs, levels, lat, outdir):
    """The kernel on every column, the Southern Hemisphere profiles mirrored (module docstring)."""
    sign = np.where(lat < 0, -1.0, 1.0).astype(np.float32)[:, None]
    mirrored = dict(sfc, v=sfc["v"] * sign)
    mlev = {p: dict(lv, v=lv["v"] * sign) for p, lv in levs.items()}
    scratch = column(mirrored, mlev, levels)
    work = Path(outdir) / f"_k{os.getpid()}"
    work.mkdir(exist_ok=True)
    k, meta = run_kernel(_CTX["kernel"], scratch, work)
    out = {f: k[f] for f in ("ecape_mu", "ecape_ml", "mucape", "mlcape", "srh1", "srh3")}
    out["shr6"] = (k["shr6_u"], k["shr6_v"] * sign)
    out["_pwat"] = k["pwat"]
    return out, meta


def winds_aloft(ps, levs):
    """The 925 and 850 hPa wind maps, masked where the level is less than 5 hPa above the ground (the columns' rule)."""
    out = {}
    for p in (925, 850):
        up = p * 100.0 < ps - 500.0
        out[f"wind{p}"] = (np.where(up, levs[p]["u"], np.nan), np.where(up, levs[p]["v"], np.nan))
    return out


def heights_check(col, levels, dec, units):
    """Hypsometric heights (column()) against the model's own at 500 and 850 hPa (above the ground only), with the 99th
    percentile of the difference apart over ground below and above 1,500 m (where a tail sits)."""
    typical = {500: 5700.0, 850: 1500.0}
    high = col[1, 0] >= 1500.0
    out = {}
    for p in (500, 850):
        if ("gh", p) not in dec:
            continue
        ours, theirs = (
            col[1, 1 + levels.index(p)],
            metres(dec[("gh", p)], units.get(("gh", p), ""), ("median", typical[p])),
        )
        c = diff_check(ours, theirs)
        err = np.abs(ours.astype(np.float64) - theirs)
        for name, m in (("low", ~high), ("high", high)):
            e = err[m & np.isfinite(err)]
            if e.size:
                c[f"p99_abs_{name}"] = round(float(np.percentile(e, 99)), 2)
        out[str(p)] = c
    return out


def soundings(model, step, sfc, levs, outdir, qc, dec, units):
    """One step's sounding tiles (snd/_parts, packed by cmd_run) from the real, unmirrored columns; on the check steps
    the height check and the tiles decoded back as the page decodes them (to_soundings.check_plev). -> checks"""
    levels, d = MODELS[model]["levels"], MODELS[model]["snd_d"]
    out = {}
    col = column(sfc, levs, levels) if qc else None
    if qc:
        out["heights"] = heights_check(col, levels, dec, units)
    try:
        codes = SND.quantise_plev(sfc, levs, levels, d)
        SND.write_parts(codes, Path(outdir) / "snd" / "_parts", step)
        if qc:
            out["snd"] = SND.check_plev(codes, levs, levels, d, col[1, 1:])
    except Exception as e:  # noqa: BLE001  the soundings never fail the maps (cmd_run leaves them out)
        print(f"::warning::{model} +{step:03d} h: sounding tiles not written ({e!r})", flush=True)
    return out


# ── meteogram points ────────────────────────────────────────────────────────────────────────────────────────────────
PT_SIMPLE = {"t2c": "t2m", "td2c": "td2m", "gust": "gust", "apcp_mm": "apcp", "asnow_cm": "asnow", "pwat": "pwat"}
PT_SIMPLE |= {"the850": "the850", "tb": "ir"}
PT_NEAR = ("ecape_mu", "ecape_ml", "mucape", "mlcape", "capem", "srh1", "srh3")


def sampler(grid):
    """This process's points.Sampler for the grid (built on first use), or None without an airport list."""
    if not _CTX.get("pts"):
        return None
    g = grid_desc(*grid)
    if _CTX.get("ps_grid") != g:
        _ids, la, lo = _CTX["pts"]
        _CTX["ps"], _CTX["ps_grid"] = PTS.Sampler(g, la, lo), g
    return _CTX["ps"]


def sample_points(phys, smp):
    """One step at every airport (points.py SPEC, its metric names): bilinear for the continuous fields, the nearest
    cell for the rate and its type and for CAPE, ECAPE, shear and helicity. -> {name: float32 (npoints,)}, NaN missing"""
    out = {}

    def near(a):
        v = smp.nearest(a).astype(np.float32)
        v[~smp.inside] = np.nan
        return v

    for n, k in PT_SIMPLE.items():
        if k in phys:
            out[n] = smp.bilinear(phys[k])
    for lev in ("10", "100", "925", "850"):
        if f"wind{lev}" in phys:
            u, v = (smp.bilinear(c) for c in phys[f"wind{lev}"])
            out[f"ws{lev}"] = np.hypot(u, v)
            out[f"wd{lev}"] = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    if "mslp" in phys:
        out["mslp"] = smp.bilinear(phys["mslp"][0])
    if "rate" in phys:
        r, t = (near(c) for c in phys["rate"])
        out["prate"] = np.where(r < 0.05, 0.0, r)
        out["ptype"] = np.where(r >= RATE_LO, t, np.where(np.isfinite(r), 0.0, np.nan))  # the map's rule
    for n in PT_NEAR:
        if n in phys:
            out[n] = near(phys[n])
    if "shr6" in phys:
        out["shr6"] = np.hypot(*(near(c) for c in phys["shr6"]))
    return out


def write_points(model, key, cyc_dt, hours, samples, outdir):
    """points.json and points.bin from the per-step samples (models.py's write_points). -> manifest entries"""
    ids = _CTX["pts"][0]
    names = sorted({n for s in samples.values() for n in s})
    series = {n: np.full((len(ids), len(hours)), np.nan, np.float32) for n in names}
    for j, h in enumerate(hours):
        for n, v in samples.get(h, {}).items():
            series[n][:, j] = v
    codes = {"0": "none", "1": "rain", "2": "snow", "3": "freezing rain", "4": "ice pellets", "5": "rain-snow mix"}
    doc = PTS.build(
        model, key, hours, ids, series, extra=dict(init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"), ptype_codes=codes)
    )
    b = PTS.dumps(doc).encode()
    (Path(outdir) / "points.json").write_bytes(b)
    bb, head = PTS.bin_bytes(doc)
    (Path(outdir) / "points.bin").write_bytes(bb)
    print(
        f"  points: {len(doc['ids'])} airports x {len(hours)} steps x {len(doc['fields'])} fields, points.json "
        f"{len(b) / 1e3:.0f} KB, points.bin {len(bb) / 1e3:.0f} KB",
        flush=True,
    )
    n = len(doc["ids"])
    return dict(
        points=dict(file="points.json", bytes=len(b), n=n, fields=list(doc["fields"])),
        points_bin=dict(file="points.bin", bytes=len(bb), head=head, n=n),
    )


def process_step(model, date, hh, step, outdir):
    """Fetch and encode one step. -> (step, {field: params}, bytes, checks, fetch stats, grid, zsfc or None, point
    samples)."""
    t0 = time.time()
    fn = step_ifs if model == "ifs" else step_gdps
    phys, checks, st, grid, zsfc = fn(date, hh, step, outdir)
    pts = {}
    try:  # the meteogram points never fail the maps
        smp = sampler(grid)
        if smp is not None:
            pts = sample_points(phys, smp)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::{model} +{step:03d} h: point sampling failed ({e!r})", flush=True)
    meta, nbytes = {}, 0
    for name, x in phys.items():
        if name.startswith("_") or name not in FIELDS:
            continue
        q, b = write_field(outdir, name, step, x)
        meta[name] = q
        nbytes += b
    checks["secs"], checks["fetch_secs"] = round(time.time() - t0, 1), round(st.secs, 1)
    return step, meta, nbytes, checks, (st.n, st.b), grid, zsfc, pts


def process_step_retry(model, date, hh, step, outdir, tries=3):
    """process_step, tried again after a pause (a dropped connection, a mirror still replicating); the step number alone
    when every try failed. A GDPS file the Datamart does not have (Missing) fails at once: the plan only takes complete
    cycles, so it will not appear, and another try would fetch the step's other files again for nothing."""
    for k in range(tries):
        try:
            return process_step(model, date, hh, step, outdir)
        except Missing as e:
            print(f"  +{step:03d} h: {e!r}"[:400], flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  +{step:03d} h attempt {k + 1}: {e!r}"[:400], flush=True)
            if k < tries - 1:
                time.sleep(30)
    return step


# ── plan / run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def complete(model, cyc_dt):
    """Is the whole cycle published? IFS: every step of both 0.1 deg files on the Google mirror (cycles.py). GDPS: the
    last hour's surface and top-level files answer a HEAD (they land together, ~5 h after the start)."""
    date, hh = cyc_dt.strftime("%Y%m%d"), cyc_dt.strftime("%H")
    if model == "ifs":
        import cycles

        return cycles.is_ready(date, hh, cycles.IFS_CONTROL_0P1)[0]
    last = MODELS["gdps"]["sfc_hours"][-1]
    return all(head_ok(gdps_url(date, hh, last, v)) for v in ("Precip-Accum_Sfc", "AirTemp_IsbL-0050"))


def load_manifest(path, model):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return dict(model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=[])


def cmd_plan(a):
    """Print "<model> <cycle>" for the newest complete 00/12Z cycle not yet in the manifest (nothing if none). With
    --cycle: that cycle, unless the manifest has it or a newer one (--force plans it anyway)."""
    man = load_manifest(Path(a.manifests) / f"sa_{a.model}.json", a.model)
    have = {c["cycle"] for c in man.get("cycles", [])}
    if a.cycle:
        if have and max(have) >= a.cycle and not a.force:
            print(f"{a.model} {a.cycle}: the page already has {max(have)}", file=sys.stderr)
        else:
            print(f"{a.model} {a.cycle}")
        return
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    c = now.replace(hour=12 if now.hour >= 12 else 0)
    for _ in range(4):
        key = c.strftime("%Y%m%d%H")
        if key in have:
            return  # the newest published: nothing newer is complete
        if complete(a.model, c):
            print(f"{a.model} {key}")
            return
        print(f"  {a.model} {key}: not complete yet", file=sys.stderr)
        c -= dt.timedelta(hours=12)


def ir_from_ttr(outdir, per_field, samples, grid):
    """IFS simulated IR from the mean outgoing longwave over each step's interval (the accumulated top thermal
    radiation differenced between consecutive steps), sampled at the airports too. -> bytes written"""
    nbytes, prev, tprev = 0, None, 0
    smp = sampler(grid)
    for f in sorted(Path(outdir).glob("_ttr_*.npy")):
        step = int(f.stem[5:])
        cur = np.load(f).astype(np.float64)
        f.unlink()
        if prev is not None or tprev == 0:
            base = prev if prev is not None else 0.0  # ttr is 0 at the start of the run
            tb = olr_tb(-(cur - base) / ((step - tprev) * 3600.0))
            _q, b = write_field(outdir, "ir", step, tb)
            per_field["ir"][step] = None
            nbytes += b
            if smp is not None:
                samples.setdefault(step, {})["tb"] = smp.bilinear(tb)
        prev, tprev = cur, step
    return nbytes


def cmd_run(a):
    model, key = a.model, a.cycle
    cyc_dt = dt.datetime.strptime(key, "%Y%m%d%H")
    date, hh = key[:8], key[8:]
    M = MODELS[model]
    hours = sorted(set(M["sfc_hours"]) | set(M["conv_hours"]))
    if os.environ.get("SA_STEPS"):  # testing only: a subset of the steps
        hours = [int(x) for x in os.environ["SA_STEPS"].split(",")]
    site = Path(a.site)
    outdir = site / DATA / model / key
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t_start = time.time()
    ncpu = os.cpu_count() or 2
    ctx = dict(kernel=str(Path(a.kernel).resolve()), omp=max(1, ncpu // a.procs))
    if PTS.AIRPORTS_SA.exists():  # the meteogram airports; each process locates them on the grid (sampler)
        ctx["pts"] = PTS.load_points((PTS.AIRPORTS_SA,))
    else:
        print(f"::warning::no {PTS.AIRPORTS_SA.name}: no meteograms for this run", flush=True)

    # step 0 first, alone: the terrain height every other step's columns start from
    _init(ctx)
    if hours[0] != 0:
        raise SystemExit("the first step must be step 0 (the terrain height)")
    first = process_step_retry(model, date, hh, 0, str(outdir))
    if isinstance(first, int):
        raise SystemExit(f"::error::{model} {key}: step 0 failed - nothing published")
    ctx["zsfc"] = first[6]
    per_field = {k: {} for k in FIELDS}
    checks, failed, nbytes, nreq, nb, samples = {}, [], 0, 0, 0, {}
    grid = first[5]

    def take(res):
        nonlocal nbytes, nreq, nb
        step, meta, b, ck, (n, bb), _g, _z, pts = res
        samples[step] = pts
        for name, q in meta.items():
            per_field[name][step] = q
        nbytes += b
        nreq += n
        nb += bb
        checks[str(step)] = ck
        print(
            f"  {model} +{step:03d} h: {b / 1e6:.2f} MB, fetched {bb / 1e6:.0f} MB in {n} requests in "
            f"{ck['fetch_secs']} s, {ck['secs']} s in all",
            flush=True,
        )

    take(first)
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(ctx,)) as pool:
        futs = [pool.submit(process_step_retry, model, date, hh, s, str(outdir)) for s in hours[1:]]
        for fu in as_completed(futs):
            r = fu.result()
            if isinstance(r, int):
                failed.append(r)
            else:
                take(r)
    if model == "ifs":
        nbytes += ir_from_ttr(outdir, per_field, samples, grid)
    for p in outdir.glob("_k*"):
        shutil.rmtree(p, ignore_errors=True)
    extra = {}
    try:  # never fail the maps for the meteograms or the soundings
        if _CTX.get("pts"):
            extra.update(write_points(model, key, cyc_dt, hours, samples, outdir))
    except Exception as e:  # noqa: BLE001
        print(f"::warning::{model} {key}: no points files ({e!r})", flush=True)
    sh = [h for h in hours if h in M["conv_hours"] and h not in failed]
    try:
        d, (lat, lon, _di) = M["snd_d"], grid
        ny_d, nx_d = -(-lat.size // d), -(-lon.size // d)
        total, nt = SND.pack_tiles(outdir / "snd", sh, ny_d, nx_d)
        extra["snd"] = SND.plev_index(
            model,
            d,
            (ny_d, nx_d),
            M["levels"],
            sh,
            total,
            nt,
            f"one grid column in {d} each way ({d * _di:.2f} deg); level 0 = the surface (2 m T and Td, 10 m wind); pressure "
            "levels less than 5 hPa above the ground left out; winds earth-relative",
        )
        nbytes += total
        print(
            f"  soundings: {nt} tiles x {len(sh)} steps, {total / 1e6:.1f} MB ({d * _di:.2f} deg columns)", flush=True
        )
    except (Exception, SystemExit) as e:  # noqa: BLE001  pack_tiles exits on a missing part: the maps still go out
        shutil.rmtree(outdir / "snd", ignore_errors=True)
        print(f"::warning::{model} {key}: no soundings ({e!r})", flush=True)

    method = METHOD[model]
    fields = {}
    for name, hs in per_field.items():
        if not hs:
            continue
        f = FIELDS[name]
        e = dict(hours=sorted(hs), grid="main", enc=f["enc"])
        if f["enc"] == "linf":
            e["q"] = [hs[h] for h in e["hours"]]
        for k in ("off", "step", "unit", "k"):
            if k in f:
                e[k] = f[k]
        if name in method:
            e["method"] = method[name]
        fields[name] = e
    wall = time.time() - t_start
    summary = summarise(checks)
    entry = dict(
        cycle=key,
        init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"),
        length=max(hours),
        fields=fields,
        grids={"main": grid_desc(*grid)},
        hours=sorted({h for e in fields.values() for h in e["hours"]}),
        levels=list(M["levels"]),
        missing_hours=sorted(failed),
        bytes=nbytes,
        fetch=dict(requests=nreq, mb=round(nb / 1e6, 1)),
        checks=summary,
        wall_s=round(wall, 1),
        made=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        **extra,
    )
    print(
        f"{model} {key}: {nbytes / 1e6:.1f} MB in {sum(len(e['hours']) for e in fields.values())} files, fetched "
        f"{nb / 1e6:.0f} MB in {nreq} requests, wall {wall:.0f} s, missing {sorted(failed)}",
        flush=True,
    )
    for name, e in fields.items():
        sz = sum((outdir / name / f"{h:03d}.webp").stat().st_size for h in e["hours"])
        print(f"  {name:9s} {len(e['hours']):3d} steps {sz / 1e6:6.1f} MB ({sz / len(e['hours']) / 1e3:.0f} KB/step)")
    print("checks: " + json.dumps(summary), flush=True)
    if failed:
        print(f"::error::{model} {key}: steps {sorted(failed)} failed - not publishing a partial cycle")
        sys.exit(1)

    mpath = site / "assets/models" / f"sa_{model}.json"
    man = load_manifest(Path(a.manifest_in) if a.manifest_in else mpath, model)
    cycles = sorted(
        [c for c in man.get("cycles", []) if c["cycle"] != key] + [entry], key=lambda c: c["cycle"], reverse=True
    )
    keep, drop = cycles[:1], cycles[1:]  # the newest run only, as models.html
    man.update(model=model, label=M["label"], source=M["source"], cycles=keep, updated=entry["made"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, separators=(",", ":")))
    if a.lists:
        L = Path(a.lists)
        L.mkdir(parents=True, exist_ok=True)
        stale = any(c["cycle"] == key for c in drop)  # a re-run of a cycle older than the one shown: not published
        (L / "publish.txt").write_text("" if stale else f"{DATA}/{model}/{key}\n")
        (L / "prune.txt").write_text("".join(f"{DATA}/{model}/{c['cycle']}\n" for c in drop))
        (L / "floor.txt").write_text(f"{DATA}/{model}:{keep[0]['cycle']}\n")
    print(f"manifest: keep {[c['cycle'] for c in keep]}, drop {[c['cycle'] for c in drop]}")


def summarise(checks):
    """The per-step checks -> the manifest's summary: medians over the steps, plus the worst."""
    out = {}
    steps = sorted(checks, key=int)

    def med(path):
        vals = []
        for s in steps:
            x = checks[s]
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
            if isinstance(x, (int, float)):
                vals.append(x)
        return (
            (round(float(np.median(vals)), 3), round(float(min(vals)), 3), round(float(max(vals)), 3)) if vals else None
        )

    for path in (
        ("mucape_vs_ifs", "r"),
        ("mucape_vs_ifs", "median_ratio"),
        ("mucape_vs_ifs", "within_x1_5"),
        ("mucape_vs_gdps_cape", "r"),
        ("mucape_vs_gdps_cape", "median_ratio"),
        ("pwat_vs_tcwv", "bias"),
        ("pwat_vs_tcwv", "mae"),
        ("heights", "500", "bias"),
        ("heights", "500", "mae"),
        ("heights", "500", "p99_abs"),
        ("heights", "500", "p99_abs_low"),
        ("heights", "500", "p99_abs_high"),
        ("heights", "850", "bias"),
        ("heights", "850", "mae"),
        ("kernel", "skipped"),
        ("snd", "t_err_k"),
        ("snd", "td_err_k"),
        ("snd", "u_err_ms"),
        ("snd", "z_err_m"),
    ):
        v = med(path)
        if v:
            out["/".join(path)] = dict(median=v[0], min=v[1], max=v[2])
    pt = [checks[s].get("ptype_t2m_median_c") for s in steps if checks[s].get("ptype_t2m_median_c")]
    if pt:
        out["ptype_t2m_median_c"] = pt[len(pt) // 2]
    snd = [checks[s]["snd"] for s in steps if "snd" in checks[s]]
    if snd:  # the sounding decode: lossless and the same levels as the columns on every check step
        out["snd/exact"] = all(c["lossless"] and c["levels_as_source"] for c in snd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--manifests", default="assets/models")
    p.add_argument("--cycle", default="", help="YYYYMMDDHH: plan this cycle (skipped when already published)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--cycle", required=True)
    p.add_argument("--site", default=".")
    p.add_argument("--procs", type=int, default=2, help="steps processed at once (each fetches its own)")
    p.add_argument("--kernel", default=str(SCRIPTS / "ecape" / "ecape_grid"), help="the built ecape_grid")
    p.add_argument("--lists", default="")
    p.add_argument("--manifest-in", default="", help="the manifest to extend (main's CURRENT copy)")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
