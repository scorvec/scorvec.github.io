#!/usr/bin/env python3
"""Short-range model viewer data (models.html; 2026-09-30, user: "add some hrrr/rrfs charts back to my site, but do it
intelligently this time with speed and quality being the most important factors ... simulated IR satellite, cloud
ceiling, visibility, temp, dewpoint and vertically integrated smoke, along with 80m winds, and incoming shortwave ...
similar charts for the rdps as well").

No figures are rendered here. Each field and forecast hour becomes ONE 8-bit lossless WebP on the model's NATIVE grid
(HRRR/RRFS 3 km Lambert 1799 x 1059, RDPS 10 km rotated lat-lon, cropped to North America), and the page draws them
with WebGL: the colour table, zoom, pan, hover values and the vector boundaries are all client-side, so switching
field or hour costs one small file and zooming costs nothing.

Encoding (codes 0-254 data, 255 = missing / "none"); the page holds the same table (models.html, FIELDS):
  ir      K,  linear, 170 K + 0.625 K * code
  ceil    ft above ground, log from 50 to 50,000 ft; 255 = no ceiling; 0 = below 50 ft
  vis     statute miles, log from 0.03 to 10 mi; 254 = 10 mi or more (the METAR ceiling: above 10 mi is noise that
          tripled the file)
  t2m     deg F, linear, PER-FRAME offset (floor of the frame minimum), 0.5 F steps (1 F if the frame spans > 127 F)
  td2m    as t2m
  smoke   mg m-2 column smoke, log from 0.1 to 3,000; 0 = below 0.1
  wind80  m/s, EARTH-relative u in R and v in G, -50.8 + 0.4 * code (the page draws speed, arrows and hover values)
  sw      W m-2, linear 5 W m-2 steps
Every file is written with its image row 0 at the NORTH (grid row j = ny - 1 - image row).

Sources (site rules: NOAA from the AWS open-data buckets byte-ranged via the .idx; RRFS v1 from NOMADS by the user's
decision 2026-09-27 - there is no real-time S3 copy - gently: one .idx + ONE multi-range request per file; ECCC
from the MSC Datamart):
  hrrr   noaa-hrrr-bdp-pds  hrrr.tHHz.wrfsfcfFF.grib2. Hourly cycles to f18, 00/06/12/18Z to f48.
         IR = SBT124 (the GOES-East 10.3 um window band as UPP labels it; SBT114 is the GOES-West copy - verified by
         the limb darkening: SBT124 - SBT114 is +11 K at 70W and -1.4 K at 120W). Ceiling = HGT at "cloud ceiling"
         (geopotential metres ABOVE SEA LEVEL - it never falls below the terrain) minus HGT at the surface.
         Smoke = COLMD (column mass density of smoke). DSWRF is instantaneous.
  rrfs   nomads .../rrfs/v1.0/rrfs.YYYYMMDD/HH/rrfs.tHHz.2dfld.3km.fFFF.conus.grib2 (the HRRR grid, identically).
         00/06/12/18Z to f84; the other hours only write sub-hourly 2-D files to f15-18 (not used).
         IR = SBTA1613 (GOES-16 ABI band 13, 10.3 um). Ceiling as HRRR. Smoke = COLMD of "particulate organic matter
         dry" (RRFS-SD's smoke tracer). DSWRF instantaneous (the file also has a 1-h mean).
  rdps   dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km/HH/FFF/, 00/06/12/18Z to f84.
         No simulated brightness temperature, ceiling or visibility exist in the RDPS output. The IR slot shows the
         OLR-derived brightness temperature Tb = (OLR / sigma)^(1/4) from UpwardLongwaveRadiationFlux_NTAtm (an
         instantaneous field, PDT 4.0; user 2026-09-30: "rdps has outgoing long wave as a proxy for IR") - a
         broadband quantity, so clear skies read ~20-30 K colder than a 10.3 um window band. Shortwave = the
         difference of consecutive DownwardShortwaveRadiationFlux-Accum_Sfc steps / 3600 s = the MEAN over the
         previous hour. Smoke from RAQDPS-FireWork (model_raqdps, same Datamart, same 0.09 deg rotated grid family
         but a smaller 729 x 599 window): PM2.5-WildfireSmokePlume_EAtm, column mass density, 00/12Z to f72; a 06/18Z
         RDPS cycle takes the RAQDPS run 6 h older, matched on valid time.

    python scripts/models/models.py plan --models hrrr,rrfs,rdps --manifests assets/models
    python scripts/models/models.py run --model hrrr --cycle 2026093012 --site . --procs 4 --lists /tmp/mv
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "snowband"))
import snowband as sb                                   # noqa: E402  the shared, validated transport (S3 / NOMADS / Datamart)

UA = {"User-Agent": "scorvec-models/1.0 (+https://scorvec.com/models.html)"}
SIGMA = 5.670374419e-8
DATA = "assets/models/data"

# ── encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────
FIELDS = {
    "ir": dict(enc="lin", off=170.0, step=0.625),
    "ceil": dict(enc="log", lo=50.0, hi=50000.0),
    "vis": dict(enc="log", lo=0.03, hi=10.0),        # 254 = 10 mi or more, as a METAR reports it
    "t2m": dict(enc="linf"),
    "td2m": dict(enc="linf"),
    "smoke": dict(enc="log", lo=0.1, hi=3000.0),
    "wind80": dict(enc="lin2", off=-50.8, step=0.4),
    "sw": dict(enc="lin", off=0.0, step=5.0),
}
ORDER = ["ir", "ceil", "vis", "t2m", "td2m", "smoke", "wind80", "sw"]


def q_lin(x, off, step):
    c = np.clip(np.rint((x - off) / step), 0, 254)
    c = np.where(np.isfinite(x), c, 255)
    return c.astype(np.uint8)


def q_log(x, lo, hi):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = 1 + np.rint((np.log(np.maximum(x, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 253)
    c = np.clip(c, 1, 254)
    c = np.where(x < lo, 0, c)
    c = np.where(np.isfinite(x), c, 255)
    return c.astype(np.uint8)


def q_linf(x):
    """Per-frame offset: 0.5 F steps from the floor of the frame minimum, 1 F (or coarser) if the frame is too wide."""
    fin = np.isfinite(x)
    if not fin.any():
        return np.full(x.shape, 255, np.uint8), [0.0, 0.5]
    lo, hi = float(np.floor(np.nanmin(x))), float(np.nanmax(x))
    step = 0.5
    while (hi - lo) / step > 254:
        step = {0.5: 1.0, 1.0: 1.5, 1.5: 2.0}.get(step, step * 2)
    return q_lin(x, lo, step), [lo, step]


def encode(name, x):
    """-> (uint8 image array, row 0 north; per-frame params or None)"""
    f = FIELDS[name]
    if f["enc"] == "lin":
        c, p = q_lin(x, f["off"], f["step"]), None
    elif f["enc"] == "log":
        c, p = q_log(x, f["lo"], f["hi"]), None
    elif f["enc"] == "linf":
        c, p = q_linf(x)
    else:                                                           # lin2: (u, v)
        u, v = x
        c = np.dstack([q_lin(u, f["off"], f["step"]), q_lin(v, f["off"], f["step"]), np.zeros(u.shape, np.uint8)])
        p = None
    return c[::-1], p


def webp(arr):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, "WEBP", lossless=True, quality=100, method=4)
    return b.getvalue()


# ── models ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def hrrr_len(cyc):
    return 48 if cyc % 6 == 0 else 18


MODELS = {
    "hrrr": dict(label="HRRR", every=1, length=hrrr_len, lag_h=(1, 30),
                 source="NOAA HRRR v4 (AWS open data, noaa-hrrr-bdp-pds)"),
    "rrfs": dict(label="RRFS", every=6, length=lambda c: 84, lag_h=(3, 40),
                 source="NOAA RRFS v1.0 (NOMADS; no real-time S3 copy exists)"),
    "rdps": dict(label="RDPS", every=6, length=lambda c: 84, lag_h=(3, 40),
                 source="ECCC RDPS 10 km and RAQDPS-FireWork (MSC Datamart)"),
}
IDX_WANT = {   # field -> regex on "VAR:LEVEL:TIME[:extra]" of a wgrib2 .idx line
    "hrrr": {"ir": r"^SBT124:top of atmosphere:", "ceilh": r"^HGT:cloud ceiling:", "zsfc": r"^HGT:surface:",
             "vis": r"^VIS:surface:", "t2m": r"^TMP:2 m above ground:", "td2m": r"^DPT:2 m above ground:",
             "smoke": r"^COLMD:entire atmosphere", "u80": r"^UGRD:80 m above ground:",
             "v80": r"^VGRD:80 m above ground:", "sw": r"^DSWRF:surface:(anl|\d+ hour fcst):?$"},
    "rrfs": {"ir": r"^SBTA1613:top of atmosphere:", "ceilh": r"^HGT:cloud ceiling:", "zsfc": r"^HGT:surface:",
             "vis": r"^VIS:surface:", "t2m": r"^TMP:2 m above ground:", "td2m": r"^DPT:2 m above ground:",
             "smoke": r"^COLMD:entire atmosphere.*Particulate organic matter dry", "u80": r"^UGRD:80 m above ground:",
             "v80": r"^VGRD:80 m above ground:", "sw": r"^DSWRF:surface:(anl|\d+ hour fcst):?$"},
}
RDPS_VARS = {"ir": "UpwardLongwaveRadiationFlux_NTAtm", "t2m": "AirTemp_AGL-2m", "td2m": "DewPoint_AGL-2m",
             "u80": "WindU_AGL-80m", "v80": "WindV_AGL-80m", "swacc": "DownwardShortwaveRadiationFlux-Accum_Sfc"}
RDPS_BOX = (-172.0, -45.0, 17.0, 80.0)      # crop of the RDPS grid (lon0, lon1, lat0, lat1): North America


def file_url(model, date, cyc, lead):
    if model == "hrrr":
        return f"{sb.hrrr_base(date, cyc)}.wrfsfcf{lead:02d}.grib2"
    if model == "rrfs":
        return f"{sb.rrfs_base(date, cyc)}.2dfld.3km.f{lead:03d}.conus.grib2"
    raise ValueError(model)


def rdps_url(date, cyc, lead, var):
    return (f"https://dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km/{cyc:02d}/{lead:03d}/"
            f"{date}T{cyc:02d}Z_MSC_RDPS_{var}_RLatLon0.09_PT{lead:03d}H.grib2")


def raq_url(date, cyc, lead):
    return (f"https://dd.weather.gc.ca/{date}/WXO-DD/model_raqdps/10km/grib2/{cyc:02d}/{lead:03d}/"
            f"{date}T{cyc:02d}Z_MSC_RAQDPS_PM2.5-WildfireSmokePlume_EAtm_RLatLon0.09_PT{lead:03d}H.grib2")


def raq_for(cycle_dt):
    """RAQDPS runs 00/12Z only: the newest run at or before the RDPS cycle, and the lead offset to add."""
    r = cycle_dt.replace(hour=0 if cycle_dt.hour < 12 else 12)
    return r, int((cycle_dt - r).total_seconds() // 3600)


# ── transport ───────────────────────────────────────────────────────────────────────────────────────────────────────
_FETCH = {}


def fetcher(model):
    if model not in _FETCH:
        _FETCH[model] = {"hrrr": lambda: sb.Fetcher("S3 HRRR", 16),
                         "rrfs": lambda: sb.Fetcher("NOMADS RRFS", 6, interval=1.1, throttle_host=True),
                         "rdps": lambda: sb.Fetcher("MSC Datamart", 12, interval=0.02)}[model]()
    return _FETCH[model]


def head_ok(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:                                           # noqa: BLE001
        return False


def idx_lines(f, url):
    body = f.get(url + ".idx", kind="idx").decode()
    rows = [l.split(":") for l in body.splitlines() if l.strip()]
    out = []
    for i, p in enumerate(rows):
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else -1
        out.append((int(p[1]), end, ":".join(p[3:])))
    return out


def fetch_idx_fields(model, url, want):
    """{key: GRIB message bytes} for the first .idx line matching each regex. NOMADS: all of them in ONE multi-range
    request (plus the .idx) - the budget the user agreed to; S3: one range per message, in parallel."""
    f = fetcher(model)
    lines = idx_lines(f, url)
    sel = {}
    for key, rx in want.items():
        for s, e, desc in lines:
            if re.search(rx, desc):
                sel[key] = (s, e)
                break
    missing = set(want) - set(sel)
    if missing:
        raise sb.Missing(f"{url}: no {sorted(missing)}")
    keys = sorted(sel, key=lambda k: sel[k][0])
    if f.throttle_host and f.multi_ok and len(keys) > 1:
        try:
            bufs = f.get_multi(url, [sel[k] for k in keys])
            return dict(zip(keys, bufs))
        except sb.MultiRangeRefused:
            f.multi_ok = False
    if f.throttle_host:
        return {k: f.get(url, sel[k]) for k in keys}
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(keys, ex.map(lambda k: f.get(url, sel[k]), keys)))


# ── grids ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def grid_desc(buf, crop=None):
    """The grid as the page needs it to map screen -> grid index, from a GRIB message."""
    import eccodes
    g = eccodes.codes_new_from_message(buf)
    try:
        t = eccodes.codes_get(g, "gridType")
        gv = lambda k: eccodes.codes_get(g, k)                   # noqa: E731
        if t == "lambert":
            d = dict(type="lcc", nx=gv("Nx"), ny=gv("Ny"), lat1=gv("Latin1InDegrees"), lat2=gv("Latin2InDegrees"),
                     lon0=((gv("LoVInDegrees") + 180) % 360) - 180, la1=gv("latitudeOfFirstGridPointInDegrees"),
                     lo1=((gv("longitudeOfFirstGridPointInDegrees") + 180) % 360) - 180,
                     dx=gv("DxInMetres"), dy=gv("DyInMetres"), R=6371229.0)
        elif t == "rotated_ll":
            d = dict(type="rotll", nx=gv("Ni"), ny=gv("Nj"), splat=gv("latitudeOfSouthernPoleInDegrees"),
                     splon=gv("longitudeOfSouthernPoleInDegrees"), la1=gv("latitudeOfFirstGridPointInDegrees"),
                     lo1=gv("longitudeOfFirstGridPointInDegrees"), di=gv("iDirectionIncrementInDegrees"),
                     dj=gv("jDirectionIncrementInDegrees"))
            if gv("iScansNegatively") or not gv("jScansPositively"):
                raise ValueError("unexpected scan order")
        else:
            raise ValueError(t)
        if crop:
            (j0, j1), (i0, i1) = crop
            d.update(i0=int(i0), j0=int(j0), nx=int(i1 - i0), ny=int(j1 - j0))
        return d
    finally:
        eccodes.codes_release(g)


def grid_latlon(buf):
    v, g = sb.decode(buf, want_grid=True)
    return g["lat"], g["lon"], g["grid_rel"]


def alpha_of(lat, lon):
    """Angle of the grid's +i axis counter-clockwise from east, per point (grid-relative -> earth-relative winds)."""
    dlon = (np.gradient(lon, axis=1) + 180) % 360 - 180
    return np.arctan2(np.gradient(lat, axis=1), dlon * np.cos(np.deg2rad(lat))).astype("float32")


def to_earth(u, v, alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    return u * ca - v * sa, u * sa + v * ca


# ── per-hour work (runs in a worker process) ───────────────────────────────────────────────────────────────────────
_CTX = {}


def _init(ctx):
    _CTX.update(ctx)


def dec(buf, crop=None):
    v = sb.decode(buf)
    if crop is not None:
        (j0, j1), (i0, i1) = crop
        v = v[j0:j1, i0:i1]
    return v


def f_of_k(k):
    return (k - 273.15) * 1.8 + 32.0 if np.nanmean(k) > 150 else k * 1.8 + 32.0


def process_hour(model, lead, blobs, outdir):
    """Decode, convert, quantise and write every field of one forecast hour. -> {field: params-or-None}, bytes"""
    t0 = time.time()
    crop = _CTX.get("crop")
    alpha = _CTX.get("alpha")
    phys = {}
    if model in ("hrrr", "rrfs"):
        if "ir" in blobs:
            phys["ir"] = dec(blobs["ir"])
        if "ceilh" in blobs:
            z = _CTX["zsfc"]
            c = dec(blobs["ceilh"])
            phys["ceil"] = np.maximum(c - z, 0.0) * 3.28084                     # m ASL -> ft above ground; NaN = none
        if "vis" in blobs:
            phys["vis"] = dec(blobs["vis"]) / 1609.344
        for k in ("t2m", "td2m"):
            if k in blobs:
                phys[k] = f_of_k(dec(blobs[k]))
        if "smoke" in blobs:
            phys["smoke"] = dec(blobs["smoke"]) * 1e6                          # kg m-2 -> mg m-2
        if "u80" in blobs and "v80" in blobs:
            u, v = dec(blobs["u80"]), dec(blobs["v80"])
            phys["wind80"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "sw" in blobs:
            phys["sw"] = dec(blobs["sw"])
    else:
        if "ir" in blobs:
            olr = dec(blobs["ir"], crop)
            with np.errstate(invalid="ignore"):
                phys["ir"] = (np.maximum(olr, 1.0) / SIGMA) ** 0.25
        for k in ("t2m", "td2m"):
            if k in blobs:
                phys[k] = f_of_k(dec(blobs[k], crop))
        if "u80" in blobs and "v80" in blobs:
            u, v = dec(blobs["u80"], crop), dec(blobs["v80"], crop)
            phys["wind80"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "swacc" in blobs and "swacc_prev" in blobs:
            a1, a0 = dec(blobs["swacc"], crop), dec(blobs["swacc_prev"], crop)
            phys["sw"] = np.maximum(a1 - a0, 0.0) / 3600.0
        if "smoke" in blobs:
            phys["smoke"] = dec(blobs["smoke"]) * 1e6
    meta, nbytes = {}, 0
    for name, x in phys.items():
        arr, p = encode(name, x)
        b = webp(arr)
        d = Path(outdir) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{lead:03d}.webp").write_bytes(b)
        nbytes += len(b)
        meta[name] = p
    return lead, meta, nbytes, time.time() - t0


# ── fetch plans ─────────────────────────────────────────────────────────────────────────────────────────────────────
def fetch_hour(model, date, cyc, lead, cyc_dt):
    """-> {key: bytes} for one forecast hour (keys as in IDX_WANT / RDPS_VARS)."""
    if model in ("hrrr", "rrfs"):
        want = {k: v for k, v in IDX_WANT[model].items() if k != "zsfc"}
        return fetch_idx_fields(model, file_url(model, date, cyc, lead), want)
    f = fetcher("rdps")
    names = dict(RDPS_VARS)
    if lead == 0:
        names.pop("ir", None)                    # no OLR at the analysis time
    jobs = {k: rdps_url(date, cyc, lead, v) for k, v in names.items()}
    if lead >= 1:
        jobs["swacc_prev"] = rdps_url(date, cyc, lead - 1, RDPS_VARS["swacc"])
    r_dt, off = raq_for(cyc_dt)
    if lead + off <= 72:
        jobs["smoke"] = raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, lead + off)
    out = {}

    def one(kv):
        k, url = kv
        try:
            return k, f.get(url)
        except sb.Missing:
            return k, None
    with ThreadPoolExecutor(8) as ex:
        for k, b in ex.map(one, jobs.items()):
            if b is not None:
                out[k] = b
    if lead == 0:
        out.pop("swacc", None)
    return out


def complete(model, cyc_dt):
    """Is the cycle's LAST forecast hour published (for RDPS, and the matching RAQDPS run to f72)?"""
    date, cyc = cyc_dt.strftime("%Y%m%d"), cyc_dt.hour
    n = MODELS[model]["length"](cyc)
    if model == "hrrr":
        return head_ok(file_url("hrrr", date, cyc, n) + ".idx")
    if model == "rrfs":
        # one directory listing (NOMADS is asked for as little as possible; listings need --compressed there too)
        try:
            req = urllib.request.Request(f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/v1.0/rrfs.{date}/{cyc:02d}/",
                                         headers={**UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception:                                       # noqa: BLE001
            return False
        return f"rrfs.t{cyc:02d}z.2dfld.3km.f{n:03d}.conus.grib2.idx" in body
    ok = head_ok(rdps_url(date, cyc, n, RDPS_VARS["swacc"]))
    if not ok:
        return False
    r_dt, off = raq_for(cyc_dt)
    if head_ok(raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, 72)):
        return True
    # RAQDPS late: wait for it up to 3 h after the RDPS run is complete, then go without smoke (the page says so)
    return (dt.datetime.utcnow() - cyc_dt).total_seconds() > (MODELS[model]["lag_h"][0] + 5) * 3600


# ── manifest ────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_manifest(path, model):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                           # noqa: BLE001
        return dict(model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=[])


def retain(cycles, model):
    """Newest first. Keep the two newest cycles; for HRRR also the newest 48-h run (an hourly cycle only reaches 18 h),
    so days 1-2 are always on the page. KEEP_<MODEL> overrides the count."""
    keep_n = int(os.environ.get(f"KEEP_{model.upper()}", "2"))
    cycles = sorted(cycles, key=lambda c: c["cycle"], reverse=True)
    keep = cycles[:keep_n]
    if model == "hrrr":
        ext = [c for c in cycles if int(c["cycle"][8:10]) % 6 == 0]
        if ext and ext[0] not in keep:
            keep.append(ext[0])
    return keep, [c for c in cycles if c not in keep]


# ── commands ────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan(a):
    """Print one line per model cycle to process: the newest COMPLETE cycle not already in that model's manifest, and
    for HRRR also the newest complete 48-h cycle if it is newer than the published one."""
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    todo = []
    for model in a.models.split(","):
        m = MODELS[model]
        man = load_manifest(Path(a.manifests) / f"{model}.json", model)
        have = {c["cycle"] for c in man.get("cycles", [])}
        cands = []
        for h in range(m["lag_h"][0], m["lag_h"][1] + 1):
            c = now - dt.timedelta(hours=h)
            if c.hour % m["every"] == 0:
                cands.append(c)
        picked = None
        for c in cands:
            key = c.strftime("%Y%m%d%H")
            if key in have:
                break                                           # newest published: nothing newer is complete
            if complete(model, c):
                picked = c
                break
        if picked:
            todo.append((model, picked.strftime("%Y%m%d%H")))
        if model == "hrrr":
            newest_ext = max([c["cycle"] for c in man.get("cycles", []) if int(c["cycle"][8:10]) % 6 == 0], default="")
            for c in cands:
                if c.hour % 6:
                    continue
                key = c.strftime("%Y%m%d%H")
                if key <= newest_ext or (picked and key == picked.strftime("%Y%m%d%H")):
                    break
                if complete(model, c):
                    todo.append((model, key))
                    break
    for model, key in todo:
        print(f"{model} {key}")


def cmd_run(a):
    model, key = a.model, a.cycle
    cyc_dt = dt.datetime.strptime(key, "%Y%m%d%H")
    date, cyc = key[:8], cyc_dt.hour
    n = MODELS[model]["length"](cyc)
    leads = list(range(0, n + 1))
    site = Path(a.site)
    outdir = site / DATA / model / key
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t_start = time.time()
    if model == "rrfs":
        fetcher("rrfs").budget = 2 * len(leads) + 30         # NOMADS: .idx + one multi-range request per file, a few retries

    # static context: grid description, terrain (ceiling AGL), wind rotation, the RDPS crop
    ctx, grids = {}, {}
    if model in ("hrrr", "rrfs"):
        z = fetch_idx_fields(model, file_url(model, date, cyc, 0), {"zsfc": IDX_WANT[model]["zsfc"]})["zsfc"]
        lat, lon, grel = grid_latlon(z)
        ctx.update(zsfc=sb.decode(z), grid_rel=grel, alpha=alpha_of(lat, lon) if grel else None)
        grids["main"] = grid_desc(z)
        del lat, lon
    else:
        f = fetcher("rdps")
        b = f.get(rdps_url(date, cyc, 1, RDPS_VARS["u80"]))
        lat, lon, grel = grid_latlon(b)
        m = (lat >= RDPS_BOX[2]) & (lat <= RDPS_BOX[3]) & (lon >= RDPS_BOX[0]) & (lon <= RDPS_BOX[1])
        jj, ii = np.where(m)
        crop = ((int(jj.min()), int(jj.max()) + 1), (int(ii.min()), int(ii.max()) + 1))
        sl = (slice(*crop[0]), slice(*crop[1]))
        ctx.update(crop=crop, grid_rel=grel, alpha=alpha_of(lat, lon)[sl] if grel else None)
        grids["main"] = grid_desc(b, crop)
        r_dt, off = raq_for(cyc_dt)
        try:
            rb = f.get(raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, off + 1))
            grids["raq"] = grid_desc(rb)
        except Exception as e:                                  # noqa: BLE001
            print(f"  RAQDPS {r_dt:%Y%m%d%H}: {e} - no smoke layer for this cycle")
        del lat, lon
    print(f"{model} {key}: {len(leads)} hours, grid {grids['main']['nx']} x {grids['main']['ny']}, "
          f"setup {time.time() - t_start:.1f}s", flush=True)

    # fetch (threads; NOMADS paced inside the Fetcher) -> decode/quantise/encode (processes)
    per_field = {k: {} for k in ORDER}
    total_bytes, failed = 0, []
    t_fetch = 0.0
    hour_workers = 1 if model == "rrfs" else (4 if model == "hrrr" else 3)
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(ctx,)) as pool, \
            ThreadPoolExecutor(hour_workers) as fx:
        def fetch_one(lead):
            t = time.time()
            for k in range(3):
                try:
                    return lead, fetch_hour(model, date, cyc, lead, cyc_dt), time.time() - t
                except (sb.Missing, sb.ShortRead) as e:
                    if k == 2:
                        print(f"  f{lead:03d}: {e}", flush=True)
                        return lead, None, time.time() - t
                    time.sleep(20)
        futs = []
        for lead, blobs, secs in fx.map(fetch_one, leads):
            t_fetch += secs
            if not blobs:
                failed.append(lead)
                continue
            futs.append(pool.submit(process_hour, model, lead, blobs, str(outdir)))
        for fu in as_completed(futs):
            lead, meta, nb, secs = fu.result()
            total_bytes += nb
            for name, p in meta.items():
                per_field[name][lead] = p
    fetch_stats = dict(requests=sum(f.nreq for f in _FETCH.values()), mb=round(sum(f.nbytes for f in _FETCH.values()) / 1e6, 1))
    fields = {}
    for name in ORDER:
        hrs = sorted(per_field[name])
        if not hrs:
            continue
        e = dict(hours=hrs, grid="raq" if (model == "rdps" and name == "smoke") else "main")
        if FIELDS[name]["enc"] == "linf":
            e["q"] = [per_field[name][h] for h in hrs]
        fields[name] = e
    if model == "rdps" and "smoke" in fields:
        r_dt, off = raq_for(cyc_dt)
        fields["smoke"]["from"] = f"RAQDPS-FireWork {r_dt:%Y-%m-%d %H}Z"
    wall = time.time() - t_start
    entry = dict(cycle=key, init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"), length=n, fields=fields, grids=grids,
                 missing_hours=failed, bytes=total_bytes, fetch=fetch_stats, wall_s=round(wall, 1),
                 made=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"))
    # the run may not publish a cycle that lost hours in the middle (a partial run would be kept for two cycles)
    ok = not failed or failed == [0]
    print(f"{model} {key}: {total_bytes / 1e6:.1f} MB in {sum(len(f['hours']) for f in fields.values())} files, "
          f"fetched {fetch_stats['mb']} MB in {fetch_stats['requests']} requests, wall {wall:.0f}s, "
          f"missing hours {failed}", flush=True)
    for name, e in fields.items():
        sz = sum((outdir / name / f"{h:03d}.webp").stat().st_size for h in e["hours"])
        print(f"  {name:7s} {len(e['hours']):3d} h  {sz / 1e6:7.1f} MB  ({sz / len(e['hours']) / 1e3:.0f} KB/h)")
    if not ok:
        print(f"::error::{model} {key}: hours {failed} failed - not publishing a partial cycle")
        sys.exit(1)

    # manifest + retention
    mpath = site / "assets/models" / f"{model}.json"
    man = load_manifest(Path(a.manifest_in) if a.manifest_in else mpath, model)
    cycles = [c for c in man.get("cycles", []) if c["cycle"] != key] + [entry]
    keep, drop = retain(cycles, model)
    man.update(model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=keep,
               updated=entry["made"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, separators=(",", ":")))
    if a.lists:
        L = Path(a.lists)
        L.mkdir(parents=True, exist_ok=True)
        (L / "publish.txt").write_text(f"{DATA}/{model}/{key}\n")
        (L / "prune.txt").write_text("".join(f"{DATA}/{model}/{c['cycle']}\n" for c in drop))
        (L / "cycle.json").write_text(json.dumps(entry))
    print(f"manifest: keep {[c['cycle'] for c in keep]}, drop {[c['cycle'] for c in drop]}")


def cmd_stamp(a):
    """After the frames push: record the frames commit that carries the cycle (the jsDelivr fallback pins it)."""
    p = Path(a.manifest)
    man = json.loads(p.read_text())
    for c in man["cycles"]:
        if c["cycle"] == a.cycle:
            c["sha"] = a.sha
    p.write_text(json.dumps(man, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--models", default="hrrr,rrfs,rdps")
    p.add_argument("--manifests", default="assets/models")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--cycle", required=True)
    p.add_argument("--site", default=".")
    p.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    p.add_argument("--lists", default="")
    p.add_argument("--manifest-in", default="", help="the manifest to extend (main's CURRENT copy); default the site's")
    p = sub.add_parser("stamp")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cycle", required=True)
    p.add_argument("--sha", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "stamp": cmd_stamp}[a.cmd](a)


if __name__ == "__main__":
    main()
