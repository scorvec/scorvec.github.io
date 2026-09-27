#!/usr/bin/env python3
"""Snow-band potential diagnostics from HRRR, RRFS and RDPS (snowbands.html; 2026-09-27, user: "adding some new
diagnostics for winter-time snow band potential. We could look at things like deformation zones, frontogenesis, CSI
(using EPV), etc. we could use the hrrr, rrfs and rdps data" and "For the snowbands, build the page now. And use
NOMADS if you need to").

Products, one frame per lead per region (`assets/snowband/anim/{model}_{product}_{region}/Fnn.webp`):

  ov    overview: the four diagnostics below side by side
  fg    700 hPa 2-D kinematic (Petterssen) frontogenesis, K per 100 km per 3 h, with 700 hPa theta and height and the
        axes of dilatation where the total deformation is large. Mesoscale snow bands sit on the warm side of a
        mid-level frontogenesis maximum, under the ascent branch of the frontal circulation it forces (Novak et al.
        2004, Wea. Forecasting 19:993).
  epv   saturated equivalent potential vorticity EPV* (PVU), 750-500 hPa layer mean, only where the layer is near
        saturation (RH >= 80 %, over ice below 0 C), 800-600 hPa frontogenesis contoured over it, hatched where theta_es
        falls with height (conditional instability) so CSI - EPV* < 0 with an upright-stable column - can be told apart
        from upright convection. CAVEAT (Schultz & Schumacher 1999, MWR 127:2709): EPV* < 0 is necessary, not
        sufficient - it also flags CI and inertial instability, it means something only in saturated air, and negative
        EPV* is potential that a circulation must release. "The atmosphere would not resist a band here", never "a band
        is here".
  dgz   dendritic-growth-zone lift: the strongest upward motion (-omega, microbar/s) in the -12 to -18 C layer where it
        is saturated, DGZ depth contoured - the "crosshair" of Waldstreicher (2001): strong lift through a deep
        saturated DGZ is what makes 2-4 in/h snowfall rates.
  bands the model's own precipitation: composite reflectivity (HRRR, RRFS) or 1-h precipitation (RDPS has no
        reflectivity) coloured by the model's precipitation type.
  ing   band ingredients: where frontogenesis >= 4, EPV* <= 0.25 PVU (saturated) and DGZ lift >= 10 microbar/s overlap
        (two of three, all three), with the model's snow reflectivity outlined. An OVERLAP of ingredients, not a
        probability, and the figure says so.
  radar (case studies only) NEXRAD base-reflectivity mosaic (Iowa Environmental Mesonet n0q) beside the model's.

Every field is Gaussian-smoothed (sigma SIGMA_KM = 18 km) on the native grid before any derivative, then evaluated on a
12 km grid (3 km models subsampled every 4th point; RDPS 10 km as is): frontogenesis and EPV are second order in the
wind and temperature and on a 3 km grid are dominated by convective-scale noise. Levels below or within 25 hPa of the
ground are masked (700 hPa is underground over the Rockies).

Sources (site rules: NOAA models from the AWS open-data buckets byte-ranged via the .idx; NOMADS for RRFS by the
user's decision 2026-09-27, as there is no S3 copy; RDPS from the MSC Datamart):
  hrrr  noaa-hrrr-bdp-pds  hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfprsfFF.grib2 + wrfsfcfFF. 00/06/12/18Z, f01-f24 every 3 h
        (f00 carries no usable omega). Winds grid-relative on the Lambert grid.
  rrfs  nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/v1.0/rrfs.YYYYMMDD/HH/rrfs.tHHz.prslev.3km.fFFF.conus.grib2 +
        2dfld.3km. 00/12Z, f03-f48 every 3 h. Vertical motion is DZDT (m/s) -> omega hydrostatically. NOMADS
        etiquette: <= 1 request/s, <= 6 at once; NOMADS throttles with an HTTP 200 "Over Rate Limit" HTML page, so
        every body is checked for the GRIB magic (or the .idx shape) and an HTML body means back off and give up on
        the cycle - never written as data.
  rdps  dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_rdps/10km/{HH}/{FFF}/..._MSC_RDPS_{Var}_{Level}_RLatLon0.09_PT{FFF}H.grib2
        00/12Z, f03-f48 every 3 h. VerticalVelocity only at 850/700/500 (and 250) hPa, so the DGZ lift is interpolated
        from three levels; PrecipType-Instant code 5 is snow (code 6 is "none" despite table 4.201 - measured).

Gates (the page states which one is in force):
  season   Nov 1 - Apr 15; outside it nothing is fetched, the loops are pruned and the page says "idle until Nov 1".
  snow     a cheap pre-check per model cycle (REFC + categorical snow; RDPS precip type + 1-h precipitation): a region is
           rendered only if >= 2 % of its area has model snow with >= 20 dBZ (RDPS >= 1 mm/h) at some lead; if no region
           qualifies the pressure levels are never fetched.
  --force  bypasses both (testing).

    python snowband.py run [--models hrrr,rrfs,rdps] [--force] [--site .]
    python snowband.py case --spec hrrr01:2020121613-2020121706     # HRRR f01 of successive cycles (verification)
    python snowband.py case --spec hrrr:2020121612                  # one HRRR run, f01-f18 hourly
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UA = {"User-Agent": "scorvec-snowband/1.0 (+https://scorvec.com/snowbands.html)"}
G, RD, OMEGA_E, R_E = 9.80665, 287.04, 7.2921e-5, 6371229.0
SIGMA_KM = 18.0
TARGET_KM = 12.0
MUTED = "#6f6b64"
SEASON = ((11, 1), (4, 15))                      # inclusive
GATE_FRAC, GATE_DBZ, GATE_MMH = 0.02, 20.0, 1.0
ING = dict(fg=4.0, epv=0.25, lift=10.0)          # band-ingredient thresholds

LEVS = [950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650, 625, 600, 575, 550, 525, 500, 475, 450]
LEVS_RDPS = [950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500, 450]
MODELS = {
    "hrrr": dict(label="HRRR", leads=[1, 3, 6, 9, 12, 15, 18, 21, 24], cycles=(0, 6, 12, 18), dx=3.0),
    "rrfs": dict(label="RRFS", leads=list(range(3, 49, 3)), cycles=(0, 12), dx=3.0),
    "rdps": dict(label="RDPS", leads=list(range(3, 49, 3)), cycles=(0, 12), dx=10.0),
}
# (label, [lon0, lon1, lat0, lat1]); a region is offered for a model only where its grid covers >= 97 % of the box
REGIONS = {
    "ne": ("Interior Northeast", [-81.0, -71.0, 39.8, 45.6]),
    "nec": ("New England and the coast", [-75.5, -66.5, 40.2, 47.5]),
    "ma": ("Mid-Atlantic", [-84.0, -72.5, 35.5, 41.8]),
    "gl": ("Great Lakes", [-93.0, -76.0, 40.0, 48.5]),
    "mw": ("Upper Midwest", [-101.0, -86.0, 40.0, 49.5]),
    "pl": ("Central Plains", [-104.0, -90.0, 34.0, 43.5]),
    "fr": ("Front Range and High Plains", [-109.5, -100.0, 36.5, 43.5]),
    "on": ("Southern Ontario and Quebec", [-84.5, -69.0, 42.0, 49.5]),
}
MASTER = [-113.0, -63.0, 31.0, 53.0]             # union of the regions + margin: everything is cropped to this first
PRODUCTS = {"ov": "Overview", "fg": "Frontogenesis and deformation", "epv": "EPV* and CSI", "dgz": "Lift in the DGZ",
            "bands": "The model's own bands", "ing": "Band ingredients", "radar": "Against NEXRAD"}
LIVE_PRODUCTS = ["ov", "fg", "epv", "dgz", "bands", "ing"]
CASES = {
    "hrrr01:2020121613-2020121706": dict(region="ne", mark=(42.21, -75.98, "Binghamton"),
                                         label="Binghamton band, 16–17 December 2020 (40.2 in), HRRR f01 each hour"),
    "hrrr:2020121612": dict(region="ne", mark=(42.21, -75.98, "Binghamton"),
                            label="Binghamton band, 16–17 December 2020 (40.2 in), the HRRR 12Z run of the 16th"),
    "hrrr01:2022012906-2022012923": dict(region="nec", mark=(42.36, -71.06, "Boston"),
                                         label="New England blizzard, 29 January 2022, HRRR f01 each hour"),
}


ONLY_LEADS: list[int] = []                       # --leads / --regions: testing overrides
ONLY_REGIONS: list[str] = []


class Throttled(RuntimeError):
    """NOMADS said stop (HTML rate-limit page, 403 or 429): give up on this cycle."""


class Missing(RuntimeError):
    """The file or message does not exist (yet)."""


# ── transport ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Fetcher:
    """HTTP GET (optionally ranged) with content validation.

    kind="grib": the body must start with b"GRIB", end with b"7777" and, for a closed range, have exactly the requested
    length; kind="idx": the first line must look like a wgrib2 inventory. Anything else - notably NOMADS's HTTP-200
    "Over Rate Limit" HTML page - is throttling on a throttled host and a corrupt read elsewhere; it is never returned.
    `interval` paces request STARTS across all threads (NOMADS: 1 s = 60/min) and `workers` caps concurrency."""

    def __init__(self, name, workers, interval=0.0, throttle_host=False):
        self.name, self.workers, self.interval, self.throttle_host = name, workers, interval, throttle_host
        self._lock = threading.Lock()
        self._next = 0.0
        self._paused_until = 0.0
        self._strikes = 0
        self.nreq = 0
        self.nbytes = 0

    def _pace(self):
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next, self._paused_until)
            self._next = t + self.interval
        if t > now:
            time.sleep(t - now)

    def _throttled(self, why):
        with self._lock:
            self._strikes += 1
            strikes = self._strikes
            self._paused_until = time.monotonic() + 90.0
        print(f"  {self.name}: throttled ({why}), strike {strikes}", flush=True)
        if strikes >= 2:
            raise Throttled(f"{self.name}: {why}")
        time.sleep(90.0)

    def get(self, url, rng=None, kind="grib", tries=3):
        for k in range(tries):
            self._pace()
            h = dict(UA)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                    body = r.read()
                self.nreq += 1
                self.nbytes += len(body)
            except urllib.error.HTTPError as e:
                self.nreq += 1
                if e.code == 404:
                    raise Missing(url) from None
                if e.code in (403, 429) and self.throttle_host:
                    self._throttled(f"HTTP {e.code}")
                    continue
                if k == tries - 1:
                    raise
                time.sleep(10 * (k + 1))
                continue
            except Exception:                                         # noqa: BLE001  network: back off, then give up
                if k == tries - 1:
                    raise
                time.sleep(10 * (k + 1))
                continue
            head = body[:512].lstrip().lower()
            if kind == "idx":
                if re.match(rb"^1:0:d=\d{10}", body[:40]):
                    return body
            else:
                ok = body[:4] == b"GRIB" and body[-4:] == b"7777"
                if ok and rng and rng[1] >= 0:
                    ok = len(body) == rng[1] - rng[0] + 1
                if ok:
                    return body
            if head.startswith(b"<") or b"rate limit" in head or b"<html" in head:
                if self.throttle_host:
                    self._throttled("HTML body instead of data")
                    continue
                raise Missing(f"HTML instead of data: {url}")
            if k == tries - 1:
                raise RuntimeError(f"{self.name}: invalid {kind} body ({len(body)} B) from {url}")
            time.sleep(5 * (k + 1))
        raise RuntimeError(f"{self.name}: gave up on {url}")


def idx_entries(fetch, url):
    """[(start, end, var, level, step)] from a wgrib2 .idx; end = -1 for the last message."""
    lines = [l.split(":") for l in fetch.get(url + ".idx", kind="idx").decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
        out.append((int(p[1]), end, p[3], p[4], p[5]))
    return out


def spans(entries, wanted):
    """Contiguous runs of WANTED messages -> [(start, end, [(start, end, key)])]. Only adjacent wanted messages are
    merged, so nothing unwanted is downloaded (single-message ranges, fewer requests where the file allows)."""
    sel = [(s, e, (v, l)) for s, e, v, l, _ in entries if (v, l) in wanted]
    sel.sort()
    runs = []
    for s, e, key in sel:
        if runs and runs[-1][1] >= 0 and runs[-1][1] + 1 == s:
            runs[-1][1] = e
            runs[-1][2].append((s, e, key))
        else:
            runs.append([s, e, [(s, e, key)]])
    return runs


def decode(buf, want_grid=False):
    import eccodes
    g = eccodes.codes_new_from_message(buf)
    try:
        nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
        v = eccodes.codes_get_values(g).reshape(nj, ni).astype("float32")
        try:
            miss = eccodes.codes_get(g, "missingValue")
            v[v == miss] = np.nan
        except Exception:                                         # noqa: BLE001
            pass
        grid = None
        if want_grid:
            lat = eccodes.codes_get_array(g, "latitudes").reshape(nj, ni)
            lon = eccodes.codes_get_array(g, "longitudes").reshape(nj, ni)
            flag = eccodes.codes_get(g, "resolutionAndComponentFlags")
            grid = dict(lat=lat, lon=((lon + 180.0) % 360.0) - 180.0, grid_rel=bool(flag & 8))
    finally:
        eccodes.codes_release(g)
    return (v, grid) if want_grid else v


def fetch_messages(fetch, url, wanted):
    """{(var, level): 2-D array} + grid from one .idx-indexed file."""
    runs = spans(idx_entries(fetch, url), set(wanted))
    missing = set(wanted) - {k for r in runs for _, _, k in r[2]}
    if missing:
        raise Missing(f"{url}: no {sorted(missing)[:4]}")

    def one(r):
        s, e, msgs = r
        buf = fetch.get(url, (s, e))
        return [(key, buf[a - s: (b - s + 1) if b >= 0 else None]) for a, b, key in msgs]
    out, grid = {}, None
    with ThreadPoolExecutor(fetch.workers) as ex:
        for pieces in ex.map(one, runs):
            for key, piece in pieces:
                if grid is None:
                    out[key], grid = decode(piece, want_grid=True)
                else:
                    out[key] = decode(piece)
    return out, grid


# ── model readers ───────────────────────────────────────────────────────────────────────────────────────────────────
FETCHERS = {}


def fetcher(model):
    if model not in FETCHERS:
        FETCHERS[model] = {"hrrr": lambda: Fetcher("S3 HRRR", 16),
                           "rrfs": lambda: Fetcher("NOMADS RRFS", 6, interval=1.0, throttle_host=True),
                           "rdps": lambda: Fetcher("MSC Datamart", 8, interval=0.05)}[model]()
    return FETCHERS[model]


def hrrr_base(date, cyc):
    return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{cyc:02d}z"


def rrfs_base(date, cyc):
    return f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/v1.0/rrfs.{date}/{cyc:02d}/rrfs.t{cyc:02d}z"


def rdps_url(date, cyc, lead, var, lev):
    return (f"https://dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km/{cyc:02d}/{lead:03d}/"
            f"{date}T{cyc:02d}Z_MSC_RDPS_{var}_{lev}_RLatLon0.09_PT{lead:03d}H.grib2")


RL_RRFS = "entire atmosphere (considered as a single layer)"


def read_surface(model, date, cyc, lead):
    """The cheap part: (reflectivity or 1-h precip, snow mask 0/1, surface pressure Pa) + grid."""
    f = fetcher(model)
    if model == "hrrr":
        raw, grid = fetch_messages(f, f"{hrrr_base(date, cyc)}.wrfsfcf{lead:02d}.grib2",
                                   [("REFC", "entire atmosphere"), ("CSNOW", "surface"), ("PRES", "surface")])
        return raw[("REFC", "entire atmosphere")], raw[("CSNOW", "surface")], raw[("PRES", "surface")], grid
    if model == "rrfs":
        raw, grid = fetch_messages(f, f"{rrfs_base(date, cyc)}.2dfld.3km.f{lead:03d}.conus.grib2",
                                   [("REFC", RL_RRFS), ("CSNOW", "surface"), ("PRES", "surface")])
        return raw[("REFC", RL_RRFS)], raw[("CSNOW", "surface")], raw[("PRES", "surface")], grid
    out, grid = {}, None
    for var, lev in (("PrecipType-Instant", "Sfc"), ("Precip-Accum1h", "Sfc"), ("Pressure", "Sfc")):
        buf = f.get(rdps_url(date, cyc, lead, var, lev))
        if grid is None:
            out[var], grid = decode(buf, want_grid=True)
        else:
            out[var] = decode(buf)
    return out["Precip-Accum1h"], (np.round(out["PrecipType-Instant"]) == 5).astype("float32"), out["Pressure"], grid


def read_levels(model, date, cyc, lead):
    """T, u, v, omega, q on the pressure levels + z700 -> {name: [nlev, nj, ni]} and the level list (hPa)."""
    f = fetcher(model)
    if model in ("hrrr", "rrfs"):
        wv = "VVEL" if model == "hrrr" else "DZDT"
        wanted = [(v, f"{p} mb") for p in LEVS for v in ("TMP", "UGRD", "VGRD", wv, "SPFH")] + [("HGT", "700 mb")]
        url = (f"{hrrr_base(date, cyc)}.wrfprsf{lead:02d}.grib2" if model == "hrrr"
               else f"{rrfs_base(date, cyc)}.prslev.3km.f{lead:03d}.conus.grib2")
        raw, grid = fetch_messages(f, url, wanted)
        st = lambda v: np.stack([raw[(v, f"{p} mb")] for p in LEVS])            # noqa: E731
        col = dict(T=st("TMP"), u=st("UGRD"), v=st("VGRD"), q=st("SPFH"), z700=raw[("HGT", "700 mb")])
        w = st(wv)
        if model == "rrfs":                                                    # DZDT (m/s) -> omega (Pa/s)
            p = np.array(LEVS, float)[:, None, None] * 100.0
            w = -p / (RD * col["T"] * (1 + 0.608 * col["q"])) * G * w
        col["w"] = w
        return col, list(LEVS), grid
    names = [(v, p) for v in ("AirTemp", "WindU", "WindV", "SpecificHumidity") for p in LEVS_RDPS]
    names += [("VerticalVelocity", p) for p in (850, 700, 500)] + [("GeopotentialHeight", 700)]

    def one(vp):
        v, p = vp
        return vp, f.get(rdps_url(date, cyc, lead, v, f"IsbL-{p:04d}"))
    raw, grid = {}, None
    with ThreadPoolExecutor(f.workers) as ex:
        for vp, buf in ex.map(one, names):
            if grid is None:
                raw[vp], grid = decode(buf, want_grid=True)
            else:
                raw[vp] = decode(buf)
    st = lambda v: np.stack([raw[(v, p)] for p in LEVS_RDPS])                  # noqa: E731
    T = st("AirTemp")
    T = T + 273.15 if np.nanmean(T) < 150 else T                               # the Datamart serves deg C
    wl = np.array([850.0, 700.0, 500.0])
    W3 = np.stack([raw[("VerticalVelocity", int(p))] for p in wl])
    w = np.empty_like(T)
    for k, p in enumerate(LEVS_RDPS):                                          # linear in p between 850/700/500
        if p >= 850:
            w[k] = W3[0]
        elif p <= 500:
            w[k] = W3[2]
        else:
            i = 0 if p > 700 else 1
            fr = (wl[i] - p) / (wl[i] - wl[i + 1])
            w[k] = (1 - fr) * W3[i] + fr * W3[i + 1]
    z7 = raw[("GeopotentialHeight", 700)]
    return dict(T=T, u=st("WindU"), v=st("WindV"), q=st("SpecificHumidity"), w=w,
                z700=z7 * 10.0 if np.nanmean(z7) < 1000 else z7), list(LEVS_RDPS), grid


# ── grid handling ───────────────────────────────────────────────────────────────────────────────────────────────────
class Grid:
    """The model grid cropped to MASTER, made right-handed (i east-ish, j north-ish), with the 12 km evaluation grid.
    Grid-relative winds stay grid-relative (consistent with differencing along i and j); earth-relative winds are
    rotated into the grid frame."""

    def __init__(self, g, dx_km):
        lat, lon = g["lat"], g["lon"]
        self.flip_j = lat[-1].mean() < lat[0].mean()
        self.flip_i = np.nanmedian(np.diff(lon, axis=1)) < 0
        lat, lon = self._orient(lat), self._orient(lon)
        m = (lon >= MASTER[0]) & (lon <= MASTER[1]) & (lat >= MASTER[2]) & (lat <= MASTER[3])
        jj, ii = np.where(m)
        self.sl = (slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1))
        self.lat, self.lon = lat[self.sl], lon[self.sl]
        self.dx_km = dx_km
        self.sig = SIGMA_KM / dx_km
        self.step = max(1, int(round(TARGET_KM / dx_km)))
        self.lat12, self.lon12 = self.lat[::self.step, ::self.step], self.lon[::self.step, ::self.step]
        self.grid_rel = g["grid_rel"]
        # angle of the grid's +i axis, counter-clockwise from east (earth-relative -> grid-relative winds)
        dlon = np.gradient(self.lon, axis=1) * np.cos(np.deg2rad(self.lat))
        self.alpha = np.arctan2(np.gradient(self.lat, axis=1), dlon)

    def _orient(self, a):
        if self.flip_j:
            a = a[..., ::-1, :]
        if self.flip_i:
            a = a[..., :, ::-1]
        return a

    def crop(self, a):
        return self._orient(a)[..., self.sl[0], self.sl[1]]

    def smooth12(self, a):
        """Crop, Gaussian-smooth by SIGMA_KM on the native grid, subsample to ~12 km."""
        from scipy.ndimage import gaussian_filter
        a = self.crop(a).astype("float32")
        bad = ~np.isfinite(a)
        if bad.any():
            a = np.where(bad, np.nanmean(a), a)
        if a.ndim == 3:
            out = np.stack([gaussian_filter(x, self.sig, mode="nearest") for x in a])
        else:
            out = gaussian_filter(a, self.sig, mode="nearest")
        return out[..., ::self.step, ::self.step]

    def winds(self, u, v):
        """Oriented grid-relative winds: a j-flip reverses v, an i-flip reverses u."""
        u, v = self.crop(u), self.crop(v)
        if not self.grid_rel:
            ca, sa = np.cos(self.alpha), np.sin(self.alpha)
            u, v = u * ca + v * sa, -u * sa + v * ca
        else:
            if self.flip_j:
                v = -v
            if self.flip_i:
                u = -u
        return u, v

    def coverage(self, box):
        """Fraction of a region box covered by the model grid (grid points inside vs expected by area)."""
        lo0, lo1, la0, la1 = box
        inside = ((self.lon >= lo0) & (self.lon <= lo1) & (self.lat >= la0) & (self.lat <= la1)).sum()
        area = (np.deg2rad(lo1 - lo0) * R_E * (np.sin(np.deg2rad(la1)) - np.sin(np.deg2rad(la0))) * R_E) / 1e6
        return inside * self.dx_km ** 2 / area


def spacing(lat, lon):
    def hav(la1, lo1, la2, lo2):
        la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2))
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 2 * R_E * np.arcsin(np.sqrt(a))
    dxi, dyj = np.empty_like(lat), np.empty_like(lat)
    dxi[:, 1:-1] = hav(lat[:, :-2], lon[:, :-2], lat[:, 2:], lon[:, 2:]) / 2
    dxi[:, 0], dxi[:, -1] = dxi[:, 1], dxi[:, -2]
    dyj[1:-1] = hav(lat[:-2], lon[:-2], lat[2:], lon[2:]) / 2
    dyj[0], dyj[-1] = dyj[1], dyj[-2]
    return dxi, dyj


# ── thermodynamics and diagnostics ──────────────────────────────────────────────────────────────────────────────────
def es_water(T):
    Tc = T - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def es_ice(T):
    return 611.15 * np.exp(22.452 * (T - 273.15) / (T - 0.61))


def theta(T, p):
    return T * (1e5 / p) ** 0.2854


def theta_es(T, p):
    """Saturated equivalent potential temperature, Bolton (1980) eq. 43 with r = r_s(T, p)."""
    es = es_water(T)
    r = 0.622 * es / (p - es) * 1000.0
    return T * (1e5 / p) ** (0.2854 * (1 - 0.28e-3 * r)) * np.exp((3.376 / T - 0.00254) * r * (1 + 0.81e-3 * r))


def petterssen(th, u, v, dx, dy):
    tx, ty = np.gradient(th, axis=-1) / dx, np.gradient(th, axis=-2) / dy
    ux, uy = np.gradient(u, axis=-1) / dx, np.gradient(u, axis=-2) / dy
    vx, vy = np.gradient(v, axis=-1) / dx, np.gradient(v, axis=-2) / dy
    f = -(tx * (ux * tx + vx * ty) + ty * (uy * tx + vy * ty)) / (np.hypot(tx, ty) + 1e-12)
    return f * 1e5 * 10800.0, ux, uy, vx, vy


def diagnose(c, levs, psfc, gr: Grid):
    """All diagnostics on the 12 km grid. c: {T,u,v,w,q,z700} already cropped+smoothed+subsampled, levs in hPa."""
    p = np.array(levs, float) * 100.0
    P = p[:, None, None]
    dx, dy = spacing(gr.lat12, gr.lon12)
    T, u, v, w, q = c["T"], c["u"], c["v"], c["w"], c["q"]
    above = P <= (psfc[None] - 2500.0)                                      # levels >= 25 hPa above the ground
    k7 = levs.index(700)
    fg7, ux, uy, vx, vy = petterssen(theta(T[k7], p[k7]), u[k7], v[k7], dx, dy)
    fg7 = np.where(above[k7], fg7, np.nan)
    E, F = ux - vy, vx + uy
    D = np.hypot(E, F)
    alpha = 0.5 * np.arctan2(F, E)
    lay = [k for k, x in enumerate(levs) if 600 <= x <= 800]
    fgl = np.stack([np.where(above[k], petterssen(theta(T[k], p[k]), u[k], v[k], dx, dy)[0], np.nan) for k in lay])
    fg_layer = nanmean_min(fgl, 0.5)
    tes = theta_es(T, P)
    dtes = np.gradient(tes, p, axis=0)
    dudp, dvdp = np.gradient(u, p, axis=0), np.gradient(v, p, axis=0)
    fcor = 2 * OMEGA_E * np.sin(np.deg2rad(gr.lat12))
    zeta = np.gradient(v, axis=-1) / dx - np.gradient(u, axis=-2) / dy
    epv = -G * ((zeta + fcor) * dtes - dvdp * np.gradient(tes, axis=-1) / dx + dudp * np.gradient(tes, axis=-2) / dy) * 1e6
    e = q * P / (0.622 + 0.378 * q)
    rh = 100.0 * e / np.where(T >= 273.15, es_water(T), es_ice(T))
    L = [k for k, x in enumerate(levs) if 500 <= x <= 750]
    epv_l = nanmean_min(np.where(above[L], epv[L], np.nan), 0.5)
    rh_l = nanmean_min(np.where(above[L], rh[L], np.nan), 0.5)
    kb, kt = L[0], L[-1]
    ci = (tes[kt] - tes[kb]) < 0
    # DGZ on a 5 hPa sub-grid of the column, only above ground and where saturated
    pf = np.arange(p.max(), p.min() - 1, -500.0)
    Tf = np.stack([interp_p(T, p, x) for x in pf])
    wf = np.stack([interp_p(w, p, x) for x in pf])
    rf = np.stack([interp_p(rh, p, x) for x in pf])
    ok = pf[:, None, None] <= (psfc[None] - 1000.0)
    dgz = (Tf <= 273.15 - 12) & (Tf >= 273.15 - 18) & ok
    lift = np.where(dgz & (rf >= 80), -wf * 10.0, -np.inf)                  # Pa/s -> microbar/s, up positive
    dgz_lift = lift.max(0)
    dgz_lift[~np.isfinite(dgz_lift)] = np.nan
    return dict(fg=fg7, fg_layer=fg_layer, th7=theta(T[k7], p[k7]), z7=c["z700"], D=D, alpha=alpha, epv=epv_l,
                rh_l=rh_l, sat=rh_l >= 80, ci=ci, dgz_lift=dgz_lift, dgz_depth=dgz.sum(0) * 5.0,
                ground700=~above[k7])


def nanmean_min(a, frac):
    """Mean over axis 0 where at least `frac` of the values are finite, else NaN."""
    n = np.isfinite(a).sum(0)
    with np.errstate(invalid="ignore"):
        m = np.nansum(a, 0) / np.maximum(n, 1)
    return np.where(n >= frac * a.shape[0], m, np.nan)


def interp_p(a, p, x):
    lp, lx = np.log(p), np.log(x)
    k = int(np.clip(np.searchsorted(-lp, -lx), 1, len(p) - 1))
    f = (lx - lp[k - 1]) / (lp[k] - lp[k - 1])
    return (1 - f) * a[k - 1] + f * a[k]


# ── one lead: fetch, smooth, diagnose ───────────────────────────────────────────────────────────────────────────────
def lead_fields(model, date, cyc, lead, sfc=None):
    """Everything the renderers need for one lead: 12 km diagnostics + native reflectivity/snow on MASTER."""
    if sfc is None:
        sfc = read_surface(model, date, cyc, lead)
    R, S, PS, g0 = sfc
    col, levs, g = read_levels(model, date, cyc, lead)
    gr = Grid(g, MODELS[model]["dx"])
    u, v = gr.winds(col["u"], col["v"])
    sm = dict(T=gr.smooth12(col["T"]), q=gr.smooth12(col["q"]), w=gr.smooth12(col["w"]), z700=gr.smooth12(col["z700"]),
              u=_smooth_cropped(gr, u), v=_smooth_cropped(gr, v))
    psfc = gr.smooth12(PS)
    d = diagnose(sm, levs, psfc, gr)
    return dict(d=d, lat=gr.lat12, lon=gr.lon12, latf=gr.lat, lonf=gr.lon, refc=gr.crop(R), snow=gr.crop(S),
                dx_eval=gr.dx_km * gr.step)


def _smooth_cropped(gr, a):
    """smooth12 for an array that is ALREADY oriented and cropped (the rotated winds)."""
    from scipy.ndimage import gaussian_filter
    a = a.astype("float32")
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    out = np.stack([gaussian_filter(x, gr.sig, mode="nearest") for x in a]) if a.ndim == 3 else gaussian_filter(a, gr.sig, mode="nearest")
    return out[..., ::gr.step, ::gr.step]


# ── cycles, season, gate ────────────────────────────────────────────────────────────────────────────────────────────
def in_season(day: dt.date):
    (m0, d0), (m1, d1) = SEASON
    return (day.month, day.day) >= (m0, d0) or (day.month, day.day) <= (m1, d1)


def next_season_start(day: dt.date):
    y = day.year if (day.month, day.day) < SEASON[0] else day.year + 1
    return dt.date(y, *SEASON[0])


def newest_cycle(model, now=None):
    """Newest cycle whose LAST lead is published (so the loop is complete), walking back up to 30 h."""
    now = now or dt.datetime.utcnow()
    f = fetcher(model)
    last = MODELS[model]["leads"][-1]
    for back in range(0, 31):
        t = now - dt.timedelta(hours=back)
        if t.hour not in MODELS[model]["cycles"]:
            continue
        date, cyc = t.strftime("%Y%m%d"), t.hour
        try:
            if model == "hrrr":
                f.get(f"{hrrr_base(date, cyc)}.wrfprsf{last:02d}.grib2.idx", kind="idx")
                f.get(f"{hrrr_base(date, cyc)}.wrfsfcf{last:02d}.grib2.idx", kind="idx")
            elif model == "rrfs":
                f.get(f"{rrfs_base(date, cyc)}.prslev.3km.f{last:03d}.conus.grib2.idx", kind="idx")
                f.get(f"{rrfs_base(date, cyc)}.2dfld.3km.f{last:03d}.conus.grib2.idx", kind="idx")
            else:
                f.get(rdps_url(date, cyc, last, "AirTemp", "IsbL-0450"))
            return date, cyc
        except Missing:
            continue
    return None


def gate(model, date, cyc, regions_all):
    """The cheap pre-check: max over leads of the fraction of each region with model snow and >= 20 dBZ (RDPS: >= 1 mm/h).
    Returns ({region: fraction}, {lead: surface tuple}, covered regions)."""
    sfc, frac, covered = {}, {}, None
    thr = GATE_MMH if model == "rdps" else GATE_DBZ
    for lead in (ONLY_LEADS or MODELS[model]["leads"]):
        R, S, PS, g = read_surface(model, date, cyc, lead)
        sfc[lead] = (R, S, PS, g)
        gr = Grid(g, MODELS[model]["dx"])
        if covered is None:
            covered = [r for r in regions_all if gr.coverage(REGIONS[r][1]) >= 0.97]
        Rc, Sc = gr.crop(R), gr.crop(S)
        hit = (Sc >= 0.5) & (Rc >= thr)
        for r in covered:
            lo0, lo1, la0, la1 = REGIONS[r][1]
            m = (gr.lon >= lo0) & (gr.lon <= lo1) & (gr.lat >= la0) & (gr.lat <= la1)
            frac[r] = max(frac.get(r, 0.0), float(hit[m].mean()))
    return frac, sfc, covered


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────────────
_FEAT = {}


def _features():
    if not _FEAT:
        import cartopy.feature as cf
        _FEAT["states"] = cf.STATES.with_scale("50m")
        _FEAT["coast"] = cf.COASTLINE.with_scale("50m")
        _FEAT["borders"] = cf.BORDERS.with_scale("50m")
        _FEAT["lakes"] = cf.LAKES.with_scale("50m")
    return _FEAT


def region_payload(L, region, extra=None):
    """Crop one lead's fields to a region (+ margin) - small enough to ship to a render worker."""
    lo0, lo1, la0, la1 = REGIONS[region][1]
    mg = 1.2

    def box(lat, lon):
        m = (lon >= lo0 - mg) & (lon <= lo1 + mg) & (lat >= la0 - mg) & (lat <= la1 + mg)
        jj, ii = np.where(m)
        return slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1)
    s12, sf = box(L["lat"], L["lon"]), box(L["latf"], L["lonf"])
    out = dict(lat=L["lat"][s12], lon=L["lon"][s12], latf=L["latf"][sf], lonf=L["lonf"][sf],
               refc=L["refc"][sf], snow=L["snow"][sf], dx_eval=L["dx_eval"],
               d={k: (v[s12] if isinstance(v, np.ndarray) and v.shape == L["lat"].shape else v) for k, v in L["d"].items()})
    if extra:
        out.update(extra)
    return out


def _proj(ext):
    import cartopy.crs as ccrs
    return ccrs.LambertConformal(central_longitude=0.5 * (ext[0] + ext[1]), central_latitude=0.5 * (ext[2] + ext[3]),
                                 standard_parallels=(ext[2] + 2, ext[3] - 2))


def _aspect(ext, proj):
    import cartopy.crs as ccrs
    lo = np.array([ext[0], ext[1], ext[0], ext[1], 0.5 * (ext[0] + ext[1]), 0.5 * (ext[0] + ext[1])])
    la = np.array([ext[2], ext[2], ext[3], ext[3], ext[3], ext[2]])
    xy = proj.transform_points(ccrs.PlateCarree(), lo, la)
    return (xy[:, 0].max() - xy[:, 0].min()) / (xy[:, 1].max() - xy[:, 1].min())


class Ctx:
    """Everything a panel needs: projected coordinates for both grids, the region, labels."""

    def __init__(self, P, ext, marks, model_label):
        import cartopy.crs as ccrs
        self.P, self.ext, self.marks, self.model = P, ext, marks, model_label
        self.proj = _proj(ext)
        pc = ccrs.PlateCarree()
        xy = self.proj.transform_points(pc, P["lon"], P["lat"])
        self.X, self.Y = xy[..., 0], xy[..., 1]
        xyf = self.proj.transform_points(pc, P["lonf"], P["latf"])
        self.XF, self.YF = xyf[..., 0], xyf[..., 1]
        self.d = P["d"]


def _base(ax, c):
    import cartopy.crs as ccrs
    F = _features()
    ax.set_extent(c.ext, crs=ccrs.PlateCarree())
    ax.add_feature(F["lakes"], facecolor="none", edgecolor="#222", lw=0.45)
    ax.add_feature(F["states"], lw=0.35, edgecolor="#444")
    ax.add_feature(F["coast"], lw=0.5, edgecolor="#222")
    ax.add_feature(F["borders"], lw=0.55, edgecolor="#222")
    for la, lo, lab in c.marks:
        ax.plot(lo, la, marker="*", ms=11, mfc="white", mec="k", mew=0.9, transform=ccrs.PlateCarree(), zorder=9)
        ax.text(lo + 0.15, la + 0.1, lab, fontsize=8, fontweight="bold", transform=ccrs.PlateCarree(), zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))


def panel_fg(ax, c):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    d = c.d
    lv = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
    cm = plt.get_cmap("YlOrRd")
    m = ax.contourf(c.X, c.Y, d["fg"], levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), extend="max")
    ax.contourf(c.X, c.Y, np.where(d["ground700"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#d9d6cf"])
    cs = ax.contour(c.X, c.Y, d["th7"], levels=np.arange(250, 330, 2), colors="#b04a1f", linewidths=0.55, linestyles="--")
    ax.clabel(cs, fmt="%d", fontsize=6.5, inline=True)
    cz = ax.contour(c.X, c.Y, d["z7"], levels=np.arange(2400, 3400, 30), colors="k", linewidths=0.9)
    ax.clabel(cz, fmt=lambda v: f"{v / 10:.0f}", fontsize=6.5, inline=True)
    ex = np.stack([np.gradient(c.X, axis=1), np.gradient(c.Y, axis=1)])
    ey = np.stack([np.gradient(c.X, axis=0), np.gradient(c.Y, axis=0)])
    ex /= np.hypot(*ex)
    ey /= np.hypot(*ey)
    dirx = np.cos(d["alpha"]) * ex[0] + np.sin(d["alpha"]) * ey[0]
    diry = np.cos(d["alpha"]) * ex[1] + np.sin(d["alpha"]) * ey[1]
    s = 3
    Dm = np.where(d["ground700"], np.nan, d["D"])
    sel = (Dm[::s, ::s] >= max(np.nanpercentile(Dm, 85) if np.isfinite(Dm).any() else 1, 6e-5))
    ax.quiver(c.X[::s, ::s][sel], c.Y[::s, ::s][sel], dirx[::s, ::s][sel], diry[::s, ::s][sel], headwidth=0,
              headlength=0, headaxislength=0, pivot="middle", scale=38, width=0.0022, color="#1f3b73", alpha=0.85)
    return m, lv, "700 hPa frontogenesis (K per 100 km per 3 h); θ dashed, height (dam), dilatation axes; grey: 700 hPa near the ground"


def panel_epv(ax, c):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    d = c.d
    lv = [-1.0, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1.0, 1.5]
    cm = ListedColormap(["#5b2a86", "#8a4fb8", "#b889d6", "#e3d0f0", "#f4f1e8", "#d9e8d2", "#a8cfa0", "#6aa36a", "#3c7040"])
    sat = d["sat"]
    m = ax.contourf(c.X, c.Y, np.where(sat, d["epv"], np.nan), levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), extend="both")
    ax.contourf(c.X, c.Y, np.where(sat & d["ci"], 1.0, np.nan), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(c.X, c.Y, np.where(sat, 1.0, 0.0), levels=[0.5], colors="#555", linewidths=0.6)
    cf_ = ax.contour(c.X, c.Y, d["fg_layer"], levels=[2, 4, 8, 16], colors="#d4145a", linewidths=[0.8, 1.1, 1.5, 1.9])
    ax.clabel(cf_, fmt="%d", fontsize=6.5, inline=True)
    return m, lv, "EPV* 750–500 hPa (PVU) where RH ≥ 80 %; magenta: 800–600 hPa frontogenesis; hatched: θes falls with height (upright CI, not CSI)"


def panel_dgz(ax, c):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    d = c.d
    lv = [2, 5, 10, 15, 20, 30, 40, 60]
    cm = plt.get_cmap("PuBu")
    m = ax.contourf(c.X, c.Y, d["dgz_lift"], levels=lv, cmap=cm, norm=BoundaryNorm(lv, cm.N), extend="max")
    cd = ax.contour(c.X, c.Y, d["dgz_depth"], levels=[50, 100, 150], colors="#7a5c00", linewidths=[0.6, 1.0, 1.4], linestyles="--")
    ax.clabel(cd, fmt="%d hPa", fontsize=6.5, inline=True)
    lab = "strongest upward motion in the saturated −12 to −18 °C layer (−ω, µb/s); dashed: DGZ depth"
    if c.model == "RDPS":
        lab += " (RDPS ω from 850/700/500 hPa only)"
    return m, lv, lab


def panel_bands(ax, c):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    P = c.P
    if c.model == "RDPS":
        lv, lab = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6], "1-h precipitation (mm) where the model's type is snow; other types olive"
    else:
        lv, lab = [5, 10, 15, 20, 25, 30, 35, 40, 45], "composite reflectivity (dBZ) where the model's type is snow; other types olive"
    snow_cm = ListedColormap(["#d7e7f7", "#a6c8ee", "#6fa2de", "#3f78c7", "#2352a8", "#4b2f9a", "#7b2aa0", "#b1209a", "#e0118b"])
    other_cm = ListedColormap(["#e9ecdf", "#d5dcc4", "#bdc8a6", "#a3b389", "#8a9d6d", "#728652", "#5b6f3b", "#445827", "#304215"])
    R, S = P["refc"], P["snow"]
    m = ax.contourf(c.XF, c.YF, np.where(S >= 0.5, R, np.nan), levels=lv, cmap=snow_cm, norm=BoundaryNorm(lv, snow_cm.N), extend="max")
    ax.contourf(c.XF, c.YF, np.where(S < 0.5, R, np.nan), levels=lv, cmap=other_cm, norm=BoundaryNorm(lv, other_cm.N), extend="max")
    return m, lv, lab


def ingredients(d):
    f = np.nan_to_num(d["fg"]) >= ING["fg"]
    e = d["sat"] & (np.nan_to_num(d["epv"], nan=9.0) <= ING["epv"])
    l = np.nan_to_num(d["dgz_lift"]) >= ING["lift"]
    return f.astype(int) + e.astype(int) + l.astype(int), f, e, l


def panel_ing(ax, c):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = c.d
    n, f, e, l = ingredients(d)
    cm = ListedColormap(["#f4d9a6", "#c2410c"])
    m = ax.contourf(c.X, c.Y, np.where(n >= 2, n, np.nan), levels=[1.5, 2.5, 3.5], cmap=cm, norm=BoundaryNorm([1.5, 2.5, 3.5], 2))
    ax.contour(c.X, c.Y, d["z7"], levels=np.arange(2400, 3400, 30), colors="#8a8680", linewidths=0.6)
    thr = 1.0 if c.model == "RDPS" else 25.0
    ax.contour(c.XF, c.YF, ((c.P["snow"] >= 0.5) & (np.nan_to_num(c.P["refc"]) >= thr)).astype(float), levels=[0.5],
               colors="#1d3f8f", linewidths=1.1)
    ax.legend(handles=[Patch(fc="#f4d9a6", ec="none", label="two of the three"), Patch(fc="#c2410c", ec="none", label="all three"),
                       Line2D([], [], color="#1d3f8f", lw=1.2, label=("model snow ≥ 1 mm/h" if c.model == "RDPS" else "model snow ≥ 25 dBZ"))],
              loc="lower left", fontsize=7.5, framealpha=0.85)
    return None, None, (f"Ingredients: 700 hPa frontogenesis ≥ {ING['fg']:g}, EPV* ≤ {ING['epv']:g} PVU in saturated air, "
                        f"DGZ lift ≥ {ING['lift']:g} µb/s. An overlap of ingredients, not a probability of a band.")


def panel_radar(ax, c):
    import cartopy.crs as ccrs
    P = c.P
    if P.get("nexrad") is None:
        ax.text(0.5, 0.5, "NEXRAD mosaic unavailable for this time", transform=ax.transAxes, ha="center", color=MUTED)
        return None, None, "NEXRAD base reflectivity (Iowa Environmental Mesonet n0q mosaic)"
    lo, la, z = P["nexrad"]
    m = ax.pcolormesh(lo, la, np.where(z >= 5, z, np.nan), vmin=5, vmax=50, cmap=_radar_cmap(), transform=ccrs.PlateCarree(),
                      shading="auto", rasterized=True)
    return m, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], "NEXRAD base reflectivity (dBZ), mosaic at the valid time"


def panel_model_refl(ax, c):
    import cartopy.crs as ccrs
    P = c.P
    m = ax.pcolormesh(P["lonf"], P["latf"], np.where(P["refc"] >= 5, P["refc"], np.nan), vmin=5, vmax=50, cmap=_radar_cmap(),
                      transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    n = ingredients(c.d)[0]
    ax.contour(c.X, c.Y, (n >= 3).astype(float), levels=[0.5], colors="k", linewidths=1.3)
    return m, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], f"{c.model} composite reflectivity (dBZ); black: all three band ingredients"


def _radar_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#c9e6f5", "#8fc7ea", "#4c9fd8", "#2f78c0", "#35a852", "#1e7a31", "#f4d03f", "#f39c12", "#e0521b", "#b3122c"])


SHORT = {"fg": "700 hPa frontogenesis (K per 100 km per 3 h)", "epv": "EPV* (PVU) where saturated; hatched: upright CI",
         "dgz": "DGZ lift (−ω, µb/s); dashed: DGZ depth", "bands": "reflectivity (dBZ): snow blue–magenta, other olive",
         "bands_rdps": "1-h precipitation (mm): snow blue–magenta, other olive"}
PANELS = {"fg": panel_fg, "epv": panel_epv, "dgz": panel_dgz, "bands": panel_bands, "ing": panel_ing}
TITLES = {"fg": "700 hPa frontogenesis, θ, height and dilatation axes", "epv": "EPV* 750–500 hPa where saturated",
          "dgz": "Lift in the dendritic growth zone", "bands": "The model's own precipitation", "ing": "Band ingredients"}


def figure(prod, P, meta, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = REGIONS[meta["region"]][1]
    c = Ctx(P, ext, meta.get("marks", []), meta["model_label"])
    r = _aspect(ext, c.proj)
    init, lead = meta["init"], meta["lead"]
    valid = init + dt.timedelta(hours=lead)
    LONG = {"ov": "snow-band diagnostics", "fg": "700 hPa frontogenesis and deformation", "epv": "EPV* and CSI, 750–500 hPa",
            "dgz": "lift in the dendritic growth zone", "bands": "precipitation and type", "ing": "band ingredients",
            "radar": "against the NEXRAD mosaic"}
    sub = (f"Fields Gaussian-smoothed σ = {SIGMA_KM:.0f} km, evaluated on a {P['dx_eval']:.0f} km grid; RH over ice below 0 °C. "
           "EPV* < 0 is necessary, not sufficient, for slantwise release (Schultz and Schumacher 1999).")
    if prod in ("ov", "radar"):
        keys = ["fg", "epv", "dgz", "bands"] if prod == "ov" else ["radar", "model"]
        ncol, nrow = 2, (2 if prod == "ov" else 1)
        pw = 6.3
    else:
        keys, ncol, nrow, pw = [prod], 1, 1, 9.4
    ph = pw / r
    multi = ncol > 1
    top, cbh, gap, side = (0.70 if multi else 1.12), (0.62 if prod != "ing" else 0.44), (0.34 if multi else 0.06), 0.12
    figw = ncol * pw + (ncol + 1) * side
    figh = top + nrow * (gap + ph + cbh)
    fig = plt.figure(figsize=(figw, figh))
    fx, fy = (lambda v: v / figw), (lambda v: v / figh)
    for i, key in enumerate(keys):
        row, col = divmod(i, ncol)
        bx = side + col * (pw + side)
        by = figh - top - (row + 1) * (gap + ph) - row * cbh
        ax = fig.add_axes([fx(bx), fy(by), fx(pw), fy(ph)], projection=c.proj)
        _base(ax, c)
        fn = {"radar": panel_radar, "model": panel_model_refl}.get(key, PANELS.get(key))
        m, ticks, lab = fn(ax, c)
        if multi and key in SHORT:
            lab = SHORT[key] if not (key == "bands" and c.model == "RDPS") else SHORT["bands_rdps"]
        if prod in ("ov", "radar"):
            ttl = TITLES.get(key, {"radar": "NEXRAD (observed)", "model": f"{meta['model_label']} (forecast)"}.get(key))
            ax.set_title(ttl, fontsize=10.5, fontweight="bold", pad=3)
        if m is not None:
            pos = ax.get_position()
            cax = fig.add_axes([pos.x0 + pos.width * 0.1, pos.y0 - fy(0.30), pos.width * 0.8, fy(0.12)])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=ticks, extend="both" if key != "ing" else "neither")
            cb.set_label(lab, fontsize=8.3, labelpad=1)
            cb.ax.tick_params(labelsize=7.5, pad=1)
        else:
            pos = ax.get_position()
            fig.text(pos.x0 + pos.width / 2, pos.y0 - fy(0.2), lab, ha="center", va="center", fontsize=8.6, color="#333")
    when = f"init {init:%Y-%m-%d %HZ} · f{lead:02d} valid {valid:%a %d %b %Y %HZ}"
    if multi:
        fig.suptitle(f"{meta['model_label']} {LONG[prod]} · {when} · {REGIONS[meta['region']][0]}", fontsize=14,
                     fontweight="bold", y=1 - fy(0.07), va="top")
    else:
        fig.suptitle(f"{meta['model_label']} {LONG[prod]} · {REGIONS[meta['region']][0]}", fontsize=13.5, fontweight="bold",
                     y=1 - fy(0.06), va="top")
        fig.text(0.5, 1 - fy(0.40), when, ha="center", va="top", fontsize=11, fontweight="bold", color="#222")
    import textwrap
    note = sub if prod != "ing" else ("An overlap of ingredients, not a probability: it marks where the model's own atmosphere "
                                      "is set up for banding, not where a band will form.")
    fig.text(0.5, 1 - fy(0.43 if multi else 0.70), "\n".join(textwrap.wrap(note, int(figw / 0.066))), ha="center", va="top",
             fontsize=8.4, color=MUTED, linespacing=1.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100 if ncol == 1 else 90, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def render_task(task):
    """Worker: all products of one (lead, region)."""
    P, meta, prods, dirs = task
    for prod in prods:
        figure(prod, P, meta, Path(dirs[prod]) / meta["file"])
    return meta["region"], meta["lead"]


# ── NEXRAD for case studies ─────────────────────────────────────────────────────────────────────────────────────────
def nexrad(valid: dt.datetime, ext):
    """IEM n0q base-reflectivity mosaic (0.005 deg, 5-min) at the valid time, cropped to the region + margin."""
    from io import BytesIO
    from PIL import Image
    u = f"https://mesonet.agron.iastate.edu/archive/data/{valid:%Y/%m/%d}/GIS/uscomp/n0q_{valid:%Y%m%d%H%M}.png"
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120) as r:
            im = np.array(Image.open(BytesIO(r.read())))
    except Exception as e:                                             # noqa: BLE001
        print(f"  NEXRAD {valid:%Y-%m-%d %HZ}: {e}", flush=True)
        return None
    lon = -126.0 + 0.005 * np.arange(im.shape[1])
    lat = 50.0 - 0.005 * np.arange(im.shape[0])
    i0, i1 = np.searchsorted(lon, ext[0] - 1), np.searchsorted(lon, ext[1] + 1)
    j0, j1 = np.searchsorted(-lat, -(ext[3] + 1)), np.searchsorted(-lat, -(ext[2] - 1))
    z = im[j0:j1:2, i0:i1:2].astype("float32") * 0.5 - 32.5
    z[im[j0:j1:2, i0:i1:2] == 0] = np.nan
    return lon[i0:i1:2], lat[j0:j1:2], z


# ── drivers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                                  # noqa: BLE001
        return default


def render_all(tasks, procs):
    if procs <= 1:
        for t in tasks:
            render_task(t)
        return
    with ProcessPoolExecutor(procs) as ex:
        list(ex.map(render_task, tasks))


def run_model(model, site: Path, status, force, procs, publish, prune):
    anim = site / "assets" / "snowband" / "anim"
    mf = anim / f"snowband_{model}_manifest.json"
    old = load_json(mf, {"regions": {}})
    st = status["models"].setdefault(model, {})
    now = dt.datetime.utcnow()
    nc = newest_cycle(model, now)
    if nc is None:
        st.update(state="error", note="no complete cycle found in the last 30 h", checked=now.strftime("%Y-%m-%dT%H:%MZ"))
        return
    date, cyc = nc
    cycle = f"{date}{cyc:02d}"
    if not force and not st.get("forced") and st.get("cycle") == cycle and st.get("state") in ("rendered", "nosnow"):
        print(f"{model}: {cycle} already done ({st.get('state')})", flush=True)
        return
    t0 = time.time()
    f = fetcher(model)
    r0, b0 = f.nreq, f.nbytes
    frac, sfc, covered = gate(model, date, cyc, list(REGIONS))
    todo = covered if force else [r for r in covered if frac.get(r, 0.0) >= GATE_FRAC]
    if ONLY_REGIONS:
        todo = [r for r in todo if r in ONLY_REGIONS]
    st.update(cycle=cycle, checked=now.strftime("%Y-%m-%dT%H:%MZ"), regions=covered,
              gate={r: round(frac.get(r, 0.0), 4) for r in covered}, forced=bool(force))
    print(f"{model} {cycle}: gate {st['gate']} -> {todo or 'nothing'}", flush=True)
    old_dirs = set(old.get("regions", {}))
    if not todo:
        st.update(state="nosnow", rendered=[], note="", seconds=round(time.time() - t0),
                  mb=round((f.nbytes - b0) / 1e6), requests=f.nreq - r0)
        prune.extend(f"assets/snowband/anim/{d}" for d in sorted(old_dirs))
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text(json.dumps({"ver": int(time.time()), "selectorLabel": "Region", "regions": {}}))
        return
    init = dt.datetime.strptime(cycle, "%Y%m%d%H")
    leads = [x for x in MODELS[model]["leads"] if x in sfc]
    tmpdirs = {}
    for r in todo:
        for p in LIVE_PRODUCTS:
            dname = f"{model}_{p}_{r}"
            dd = anim / dname
            if dd.exists():
                shutil.rmtree(dd)
            tmpdirs[(p, r)] = dd
    frames = {k: [] for k in tmpdirs}
    for i, lead in enumerate(leads):
        tl = time.time()
        L = lead_fields(model, date, cyc, lead, sfc=sfc.pop(lead))
        valid = init + dt.timedelta(hours=lead)
        tasks = []
        for r in todo:
            meta = dict(model_label=MODELS[model]["label"], init=init, lead=lead, region=r, file=f"F{i:02d}.webp")
            tasks.append((region_payload(L, r), meta, LIVE_PRODUCTS, {p: str(tmpdirs[(p, r)]) for p in LIVE_PRODUCTS}))
            for p in LIVE_PRODUCTS:
                frames[(p, r)].append({"idx": i, "file": f"F{i:02d}.webp", "date": valid.strftime("%Y-%m-%d"),
                                       "label": f"f{lead:02d} · {valid:%a %d %b %HZ}"})
        render_all(tasks, procs)
        print(f"  {model} f{lead:02d}: {time.time() - tl:.0f} s ({len(tasks) * len(LIVE_PRODUCTS)} frames)", flush=True)
    regions = {}
    for (p, r), fr in frames.items():
        regions[f"{model}_{p}_{r}"] = {"label": f"{MODELS[model]['label']} {PRODUCTS[p]} · {REGIONS[r][0]}", "frames": fr}
    mf.write_text(json.dumps({"ver": int(time.time()), "selectorLabel": "Region", "regions": regions}))
    publish.extend(f"assets/snowband/anim/{d}" for d in sorted(regions))
    prune.extend(f"assets/snowband/anim/{d}" for d in sorted(old_dirs - set(regions)))
    st.update(state="rendered", rendered=todo, note="", seconds=round(time.time() - t0),
              mb=round((f.nbytes - b0) / 1e6), requests=f.nreq - r0,
              leads=[f"f{x:02d}" for x in leads])
    print(f"{model} {cycle}: {len(regions)} loops, {st['mb']} MB in {st['requests']} requests, {st['seconds']} s", flush=True)


def cmd_run(a):
    site = Path(a.site).resolve()
    data = site / "assets" / "snowband" / "data"
    data.mkdir(parents=True, exist_ok=True)
    sp = data / "snowband_status.json"
    status = load_json(sp, {})
    status.setdefault("models", {})
    now = dt.datetime.utcnow()
    today = now.date()
    publish, prune = [], []
    active = in_season(today)
    status["season"] = dict(active=active, window="1 November – 15 April", next_start=next_season_start(today).isoformat())
    status["forced"] = bool(a.force)
    if not active and not a.force:
        # idle: prune whatever an earlier (forced or in-season) run left, write the status at most once a day
        changed = False
        for m in MODELS:
            mf = site / "assets" / "snowband" / "anim" / f"snowband_{m}_manifest.json"
            old = load_json(mf, {"regions": {}})
            if old.get("regions"):
                prune.extend(f"assets/snowband/anim/{d}" for d in sorted(old["regions"]))
                mf.write_text(json.dumps({"ver": int(time.time()), "selectorLabel": "Region", "regions": {}}))
                changed = True
            st = status["models"].setdefault(m, {})
            if st.get("state") != "idle":
                st.update(state="idle", rendered=[], note="")
                changed = True
        if changed or status.get("updated", "")[:10] != now.strftime("%Y-%m-%d"):
            status["updated"] = now.strftime("%Y-%m-%dT%H:%MZ")
            sp.write_text(json.dumps(status, indent=1))
        print(f"out of season - idle until {status['season']['next_start']}; pruning {len(prune)} loop(s)")
    else:
        for m in a.models.split(","):
            try:
                run_model(m, site, status, a.force, a.procs, publish, prune)
            except Throttled as e:
                status["models"].setdefault(m, {}).update(state="error", note=f"NOMADS rate limit; skipped this cycle ({e})",
                                                          checked=now.strftime("%Y-%m-%dT%H:%MZ"))
                print(f"{m}: {e}", flush=True)
            except Exception as e:                                     # noqa: BLE001
                status["models"].setdefault(m, {}).update(state="error", note=f"{type(e).__name__}: {str(e)[:200]}",
                                                          checked=now.strftime("%Y-%m-%dT%H:%MZ"))
                print(f"{m}: FAILED {type(e).__name__}: {e}", flush=True)
        status["updated"] = now.strftime("%Y-%m-%dT%H:%MZ")
        sp.write_text(json.dumps(status, indent=1))
    write_lists(a, publish, prune)


def write_lists(a, publish, prune):
    if a.lists:
        Path(a.lists).mkdir(parents=True, exist_ok=True)
        (Path(a.lists) / "publish.txt").write_text(" ".join(publish))
        (Path(a.lists) / "prune.txt").write_text(" ".join(prune))
    print(f"publish {len(publish)} dir(s), prune {len(prune)}")


def cmd_case(a):
    site = Path(a.site).resolve()
    spec = a.spec
    info = CASES.get(spec, {})
    region = a.region or info.get("region", "ne")
    m = re.match(r"^(hrrr01|hrrr):(\d{10})(?:-(\d{10}))?$", spec)
    if not m:
        raise SystemExit("spec: hrrr:YYYYMMDDHH (one run, f01-f18) or hrrr01:YYYYMMDDHH-YYYYMMDDHH (f01 of each cycle)")
    kind, t0 = m.group(1), dt.datetime.strptime(m.group(2), "%Y%m%d%H")
    if kind == "hrrr":
        steps = [(t0, L) for L in range(1, 19)]
    else:
        t1 = dt.datetime.strptime(m.group(3), "%Y%m%d%H") if m.group(3) else t0 + dt.timedelta(hours=17)
        steps, t = [], t0 - dt.timedelta(hours=1)
        while t <= t1 - dt.timedelta(hours=1):
            steps.append((t, 1))
            t += dt.timedelta(hours=1)
    anim = site / "assets" / "snowband" / "anim"
    prods = LIVE_PRODUCTS + ["radar"]
    dirs = {p: anim / f"case_{p}" for p in prods}
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
    marks = [info["mark"]] if info.get("mark") else []
    frames = {p: [] for p in prods}
    t_start = time.time()
    f = fetcher("hrrr")
    tasks = []
    for i, (init, lead) in enumerate(steps):
        L = lead_fields("hrrr", init.strftime("%Y%m%d"), init.hour, lead)
        valid = init + dt.timedelta(hours=lead)
        nx = nexrad(valid, REGIONS[region][1])
        meta = dict(model_label="HRRR", init=init, lead=lead, region=region, file=f"F{i:02d}.webp", marks=marks)
        tasks.append((region_payload(L, region, {"nexrad": nx}), meta, prods, {p: str(dirs[p]) for p in prods}))
        for p in prods:
            frames[p].append({"idx": i, "file": f"F{i:02d}.webp", "date": valid.strftime("%Y-%m-%d"),
                              "label": f"{valid:%d %b %HZ} · init {init:%HZ} f{lead:02d}"})
        print(f"  case step {i + 1}/{len(steps)}: {init:%Y-%m-%d %HZ} f{lead:02d}", flush=True)
    render_all(tasks, a.procs)
    regions = {f"case_{p}": {"label": PRODUCTS[p], "frames": frames[p]} for p in prods}
    (anim / "snowband_case_manifest.json").write_text(json.dumps({"ver": int(time.time()), "selectorLabel": "View",
                                                                  "regions": regions}))
    sp = site / "assets" / "snowband" / "data" / "snowband_status.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    status = load_json(sp, {"models": {}})
    status["case"] = dict(spec=spec, region=region, label=info.get("label", spec),
                          rendered=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), frames=len(steps),
                          mb=round(f.nbytes / 1e6), seconds=round(time.time() - t_start))
    sp.write_text(json.dumps(status, indent=1))
    print(f"case {spec}: {len(steps)} steps x {len(prods)} products, {f.nbytes / 1e6:.0f} MB in {f.nreq} requests, "
          f"{time.time() - t_start:.0f} s", flush=True)
    write_lists(a, [f"assets/snowband/anim/case_{p}" for p in prods], [])


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", default="hrrr,rrfs,rdps")
    r.add_argument("--force", action="store_true")
    r.add_argument("--leads", default="", help="testing: only these leads")
    r.add_argument("--regions", default="", help="testing: only these regions")
    c = sub.add_parser("case")
    c.add_argument("--spec", required=True)
    c.add_argument("--region")
    for p in (r, c):
        p.add_argument("--site", default=str(REPO))
        p.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
        p.add_argument("--lists", help="directory for publish.txt / prune.txt (the workflow's frames step reads them)")
    sub.add_parser("season", help="exit 0 inside the season (1 Nov - 15 Apr), 1 outside")
    a = ap.parse_args()
    if a.cmd == "season":
        return 0 if in_season(dt.datetime.utcnow().date()) else 1
    if a.cmd == "run":
        ONLY_LEADS[:] = [int(x) for x in a.leads.split(",") if x]
        ONLY_REGIONS[:] = [x for x in a.regions.split(",") if x]
    {"run": cmd_run, "case": cmd_case}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
