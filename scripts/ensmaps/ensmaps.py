#!/usr/bin/env python3
"""Global ensemble maps for ens-anomalies.html: 2 m temperature, 500 hPa height and sea-level pressure from four
ensembles (IFS-ENS, AIFS-ENS, GEFS, GEPS), each member's daily means against the 1991-2020 normals and how the run moved
the ensemble mean from the run 12 hours earlier, to day 15. Nothing is rendered here: every map is a lossless WebP of
the model's own numbers on its own grid, which the page colours, contours and reads.

Ensembles and grids (one job per model, each as its run lands; ensmaps.yml):
  ifs   IFS-ENS, the 50 perturbed members: 2 m temperature on the 0.1-deg grid (its surface files carry the perturbed
        members only), 500 hPa height and sea-level pressure on the 0.25-deg grid; Google Cloud mirror only, byte
        ranges of the wanted messages (rangefetch)
  aifs  AIFS-ENS, the control and 50 perturbed members, 0.25 deg; Google Cloud mirror only
  gefs  GEFS, the control and 30 perturbed members, 0.5 deg (pgrb2a); NOAA's S3 bucket noaa-gefs-pds only, byte
        ranges through each file's .idx
  geps  GEPS, the control and 20 perturbed members, 0.5 deg; the MSC Datamart through scripts/geps/datamart.py
Runs: 00, 06, 12 and 18Z (GEPS 00 and 12Z), each to +360 h (GEFS and GEPS +384 h), except IFS-ENS 06/18Z, which
runs to +144 h (Model.runs). A run can be followed as it lands (--follow, planned with --started once its first day is
complete): each step is fetched as soon as its source has it, and at the end of days 5 and 10 what is done is published
(--checkpoint: the days and the windows they close, "through" in the manifest entry); the change maps, the regimes,
the clusters' fields and the statistics come with the whole run. IFS-ENS comes in two stages (--fields): its 0.25-deg files land hours before the 0.1-deg
surface files, so 500 hPa height and sea-level pressure are published as soon as they are complete (with the regimes
and the clusters' fields; the run statistics kept as <cycle>-partial), and 2 m temperature is added to the same run's
entry when its files land (--fields t2m --merge: the maps added, the statistics completed, nothing else redone). A day is a calendar UTC day, the mean of 06, 12, 18 and 24 UTC: a 00Z run has 15 full
days, a 06, 12 or 18Z run 14 (its first and last partial days are not read), GEFS and GEPS one more, an IFS-ENS 06/18Z
run 5. Spans: days 1-5, 6-10, 11-15 (the last cut at the run's last day) and each day.

Anomaly: each member's daily mean less the day's normal (ensmaps_ref.Normals: the curve averaged over the day's four
times). 2 m temperature: ERA5-Land/ERA5 at 0.1 deg; on a coarser grid it is averaged over each cell (cellmeans) and
moved by LAPSE from the normal's orography (the IFS's, which ERA5-Land shares: the IFS control's step-0 surface
geopotential, 0.25 deg, averaged over the cell) to the model's own surface height, so a cell is compared with a normal
at its own height. 500 hPa height and sea-level pressure: ERA5 at 1.5 deg, interpolated bilinearly. The sea-level
pressure anomaly is not drawn where the model's surface is above HIGH_M (the reduction to sea level differs between
models and the reanalysis there). The page colours an anomaly where at least sign_k(n) of the n members share the sign
of their mean: a two-sided sign test against an even split at 5 % (33 of 50 or 51, 22 of 31, 16 of 21).

Change: this run's members' mean less the run 12 h earlier's over the same calendar days (the days both runs hold), on
a 0.5-deg grid (each member's field averaged over the cells first: the statistics carried from run to run stay small):
Welch's t-test on the two runs' members, Benjamini-Hochberg over the map's points at FDR_Q; the page shades the
change everywhere and dots the significant points. The same against the run 6 h earlier (c6/) and the run 48 h earlier
(c48/, the trend over two days) where those runs' statistics are there. A run's statistics come from its own --save
(the Actions cache, which keeps every run of the last 48 h).

Files, per field and span (assets/models/data/ensmaps/<model>/<cycle>/<field>/), rows 90N first, columns from 180W:
  a/<span>.webp   R anomaly of the members' mean, (code - 128) x enc[field]["anom"], 0 = not drawn
                  G the members whose anomaly has the sign of the mean, 255 = not drawn
                  B the members' mean, enc lo + (code - 1) x step (2 m temperature in deg C), 0 = missing
                  On the 0.1-deg grid: a/<span>/o.webp, every 5th point (the 0.5-deg points, exact values), and
                  a/<span>/<row><col>.webp, 6 x 6 tiles of 300 x 600 points each with one more row and column shared
                  with the next tile (the last column wraps to 180W).
  s/<span>.webp   (500 hPa height, sea-level pressure) as a/, R the standardized anomaly: the members' mean anomaly
                  over the sd of an anomaly of the span's own averaging period (a day or its n-day mean; ensmaps_ref
                  Spread), (code - 128) x enc[field]["sigma"]
  c/<span>.webp   R change of the members' mean from the run 12 h earlier, (code - 128) x enc[field]["change"], 0 =
                  none; G 1 = significant. c6/, c48/<span>.webp: the same from the run 6 h, 48 h earlier
  r/<sector><regime>/<span>.webp   the members of one weather regime (ensmaps_regimes.py), as a/ on the 0.5-deg grid
The manifest (assets/models/ensmaps_<model>.json: the KEEP_RUNS newest runs, newest first, each in the frames store)
lists the spans, members, gates, grids, encodings and the regimes; this run's statistics for the next runs go to --save,
its regime labels to --regimes-out.

    python scripts/ensmaps/ensmaps.py plan --model ifs --manifests assets/models
    python scripts/ensmaps/ensmaps.py run --model ifs --cycle 2026102500 --site . --normals ref/ensmaps_t2m_normals.npz \
        --prev stats/2026102412.npz --prev6 stats/2026102418.npz --prev48 stats/2026102300.npz --save stats/2026102500.npz \
        --lists /tmp/lists
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before eccodes)

import argparse
import datetime as dt
import functools
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "scripts" / "ecmwf"),
    str(ROOT / "scripts" / "lib"),
    str(ROOT / "scripts" / "geps"),
    str(ROOT / "scripts" / "regimes"),
]
import ensmaps_ref as R

FIELDS = ("t2m", "z500", "msl")
SAMPLES = (6, 12, 18, 24)  # hours after a day's 00 UTC: the four fields of its mean
LAST_STEP = 360  # the ECMWF ensembles' last step; GEFS and GEPS run to 384, IFS-ENS 06/18Z to 144 (Model.runs)
KEEP_RUNS = 4  # runs of each model on the page (the frames store and the manifest)
RETAIN_MIN = 60  # minutes the store keeps what a manifest stopped naming: the deploy and the browsers' caches lag it
PERIOD = 5  # days per window
FDR_Q = 0.10
LAPSE = 6.5e-3  # K/m: a coarser grid's 2 m normal is moved to the model's surface height at this rate
HIGH_M = 1500.0  # m: no sea-level pressure anomaly above this surface height
MIN_SHARE = 0.8  # a span is drawn from at least this share of the model's members (complete on each of its days)
G = 9.80665
AHEAD = 2  # steps fetched ahead of the decoding
TILE = (300, 600)  # rows, columns of a 0.1-deg tile
DATA = "assets/models/data/ensmaps"
# per field: anomaly step, the mean's lowest value and step, change step (K or deg C, m, hPa)
ENC = {
    "t2m": {"anom": 0.25, "lo": -78.0, "mean": 0.5, "change": 0.1},
    "z500": {"anom": 4.0, "lo": 4600.0, "mean": 6.0, "change": 2.0, "sigma": 0.025},
    "msl": {"anom": 0.4, "lo": 920.0, "mean": 0.6, "change": 0.2, "sigma": 0.025},
}
UNIT = {"t2m": "K", "z500": "m", "msl": "hPa"}
S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_IDX = {"t2m": ("TMP", "2 m above ground"), "z500": ("HGT", "500 mb"), "msl": ("PRMSL", "mean sea level")}
GEPS_VAR = {"t2m": "TMP_TGL_2m", "z500": "HGT_ISBL_0500", "msl": "PRMSL_MSL_0"}
ECMWF_PARAM = {"ifs": {"t2m": "2t", "z500": "gh", "msl": "msl"}, "aifs": {"t2m": "2t", "z500": "z", "msl": "msl"}}


def log(*a):
    print(*a, file=sys.stderr, flush=True)  # stdout carries only what plan prints


# ── grids ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, eq=False)
class Grid:
    """A global regular grid, 90N first, columns from 180W."""

    res: float

    @property
    def lat(self) -> np.ndarray:
        return np.round(90.0 - self.res * np.arange(round(180 / self.res) + 1), 4)

    @property
    def lon(self) -> np.ndarray:
        return np.round(-180.0 + self.res * np.arange(round(360 / self.res)), 4)

    @property
    def shape(self) -> tuple[int, int]:
        return round(180 / self.res) + 1, round(360 / self.res)

    @property
    def tiled(self) -> bool:
        return self.res < 0.25

    def desc(self) -> dict:
        ny, nx = self.shape
        return {"res": self.res, "ny": ny, "nx": nx, "lat0": 90.0, "lon0": -180.0}


G10, G25, G50 = Grid(0.1), Grid(0.25), Grid(0.5)


def to_half(a: np.ndarray, res: float) -> np.ndarray:
    """(..., lat, lon) on a global grid -> the 0.5-deg cells' means (..., 361, 720), cos-latitude weighted. The 0.1 and
    0.25-deg points nest in the cells: 5 x 5 points each (0.1 deg), or 3 x 3 with half weights on the edge points
    (0.25 deg); a pole cell is the half inside the grid."""
    if res == 0.5:
        return a
    k = round(0.5 / res)
    lat = Grid(res).lat
    w1 = np.ones(k) if k % 2 else np.r_[0.5, np.ones(k - 1), 0.5]  # the points across a cell, centred on its own
    h = len(w1) // 2
    cw = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)  # the pole points weigh nothing: their rings carry the cells
    ny, nx = a.shape[-2:]
    pad = np.concatenate([a[..., -h:], a, a[..., :h]], axis=-1)  # periodic in longitude
    xs = sum(w * pad[..., j : j + nx] for j, w in enumerate(w1))[..., ::k] / w1.sum()
    rows = np.arange(0, ny, k)
    out = np.empty((*a.shape[:-2], rows.size, nx // k), np.float32)
    for r, j in enumerate(rows):
        lo, hi = max(0, j - h), min(ny, j + h + 1)
        wr = w1[h - (j - lo) : h + (hi - j)] * cw[lo:hi]
        out[..., r, :] = np.tensordot(wr, xs[..., lo:hi, :], axes=(0, -2)) / wr.sum()
    return out


def to25(a: np.ndarray, res: float) -> np.ndarray:
    """A global field (90N first, columns from 180W) -> the regimes' 2.5-deg NCEP grid (regime_core.box25, which takes
    columns from 0E)."""
    import regime_core as RC

    return RC.box25(np.roll(a, a.shape[-1] // 2, axis=-1), res).astype(np.float32)


def roll_rows(v: np.ndarray, lat0: float, lon0: float, inc: float, up: bool) -> np.ndarray:
    """A global message's values (nj, ni) -> 90N first, columns from 180W."""
    if up:
        v = v[::-1]
    lon = np.round(-180.0 + inc * np.arange(v.shape[1]), 4)
    return R.roll_to(v, lon0, inc, lon)


# ── normals on a model's grids ────────────────────────────────────────────────────────────────────────────────────────
def bilinear(coef: np.ndarray, lat: np.ndarray, lon: np.ndarray, g: Grid) -> np.ndarray:
    """(k, lat, lon) curves on a regular grid (90N first, columns from 180W) -> (k, g's points), periodic in longitude."""
    from scipy.ndimage import map_coordinates

    res = float(lat[0] - lat[1])
    rows = np.broadcast_to(((lat[0] - g.lat) / res)[:, None], g.shape).ravel()
    cols = np.broadcast_to((((g.lon - lon[0]) % 360) / res)[None], g.shape).ravel()
    out = np.empty((coef.shape[0], *g.shape), np.float32)
    for k, c in enumerate(coef):
        wrapped = np.concatenate([c, c[:, :1]], axis=1)
        out[k] = map_coordinates(wrapped.astype(np.float64), [rows, cols], order=1, mode="nearest").reshape(g.shape)
    return out


def t2m_on(N: R.Normals, g: Grid, shift: np.ndarray | None) -> R.Normals:
    """The 0.1-deg 2 m normal averaged over g's cells, plus the constant height shift (K) when there is one."""
    import cellmeans as CM

    if g.res == 0.1:
        return N
    coef = CM.coarsen(N.coef.astype(np.float64), N.lat, N.lon, 0.1, g.lat, g.lon, g.res, 360.0).astype(np.float32)
    if shift is not None:
        coef[0] += shift
    return R.Normals(coef, g.lat, g.lon, N.meta)


def upper_on(field: str, g: Grid) -> R.Normals:
    N = R.Normals.upper(field)
    return R.Normals(bilinear(N.coef, N.lat, N.lon, g), g.lat, g.lon, N.meta)


def spread_on(field: str, g: Grid) -> R.Spread:
    """The standardized anomalies' sd curves (ensmaps_ref.Spread) interpolated bilinearly to g, as the normals are."""
    S = R.Spread.upper(field)
    return R.Spread({n: bilinear(c, S.lat, S.lon, g) for n, c in S.coef.items()}, g.lat, g.lon)


def day_normal(N: R.Normals, day: pd.Timestamp) -> np.ndarray:
    """The normal curve averaged over the day's four forecast times, as the day's mean is."""
    return N.at([day + pd.Timedelta(hours=h) for h in SAMPLES]).mean(0)


# ── the models ────────────────────────────────────────────────────────────────────────────────────────────────────────
def split_grib(blob: bytes) -> list[bytes]:
    """The GRIB2 messages of a file's bytes, by their own length fields (no eccodes: this runs in the fetch threads)."""
    out, i = [], 0
    while (i := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[i + 8 : i + 16], "big")
        out.append(blob[i : i + n])
        i += n
    return out


def ecmwf_messages(date, hh, model, step, kind, params: dict, resol="0p25", levtype=None) -> tuple[list, int]:
    """[(field, None, message)] of the wanted parameters (all members in the file) and the bytes read."""
    import rangefetch as rf

    idx = rf.fetch_index(date, hh, model, step, kind, resol=resol, levtype=levtype)
    want, of = [], {}
    for f, (param, lev) in params.items():
        for e in idx:
            if e.get("param") == param and (lev is None or str(e.get("levelist")) == lev):
                if kind == "ef" and e.get("type") != "pf":
                    continue
                want.append(e)
                of[e["_offset"]] = f
    if not want:
        raise RuntimeError(f"{model} {date} {hh}Z +{step} h: none of {params} in the index")
    want.sort(key=lambda e: e["_offset"])
    stats: dict = {}
    path = rf.path_for(date, hh, model, step, kind, resol=resol, levtype=levtype) + ".grib2"
    blob = rf.fetch_ranges(path, rf.coalesce(want), workers=12, stats=stats)
    lengths = [e["_length"] for e in want]
    msgs = [blob[e - n : e] for e, n in zip(np.cumsum(lengths), lengths)]
    return [(of[e["_offset"]], None, m) for e, m in zip(want, msgs)], sum(stats.values())


def fetch_ifs(date: str, hh: str, step: int, fields=FIELDS) -> tuple[list, int]:
    out, nb = [], 0
    if "t2m" in fields:
        a, na = ecmwf_messages(date, hh, "ifs", step, "ef", {"t2m": ("2t", None)}, resol="0p1", levtype="sfc")
        out, nb = out + a, nb + na
    upper = {f: p for f, p in (("z500", ("gh", "500")), ("msl", ("msl", None))) if f in fields}
    if upper:
        b, nu = ecmwf_messages(date, hh, "ifs", step, "ef", upper)
        out, nb = out + b, nb + nu
    return out, nb


def fetch_aifs(date: str, hh: str, step: int) -> tuple[list, int]:
    params = {"t2m": ("2t", None), "z500": ("z", "500"), "msl": ("msl", None)}
    a, na = ecmwf_messages(date, hh, "aifs-ens", step, "pf", params)
    b, nb = ecmwf_messages(date, hh, "aifs-ens", step, "cf", params)
    return a + b, na + nb


GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]


def gefs_url(date: str, hh: str, mem: str, step: int) -> str:
    return f"{S3}/gefs.{date}/{hh}/atmos/pgrb2ap5/{mem}.t{hh}z.pgrb2a.0p50.f{step:03d}"


def s3_get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    """One GET from the GEFS bucket (a byte range of a GRIB file, open-ended for the file's last message, or a whole
    .idx), checked: a GRIB range must be one whole message, as long as its own header says."""
    from webget import get

    hdr = {"Range": f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"} if rng else None
    for k in range(4):
        b = get(url, headers=hdr, timeout=120)
        if rng is None or (b[:4] == b"GRIB" and b[-4:] == b"7777" and len(b) == int.from_bytes(b[8:16], "big")):
            return b
        log(f"  {url}: a short or broken range ({len(b)} bytes), try {k + 1}")
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"{url} {rng}: no whole message after 4 tries")


def gefs_ranges(url: str, step: int) -> list[tuple[str, tuple[int, int]]]:
    """(field, byte range) of the three messages in one member's step file, from its .idx."""
    lines = [ln.split(":") for ln in s3_get(url + ".idx").decode().splitlines() if ln.strip()]
    out = []
    for f, (var, lev) in GEFS_IDX.items():
        hit = [i for i, p in enumerate(lines) if p[3] == var and p[4] == lev]
        if not hit:
            raise RuntimeError(f"{url}: no {var} {lev} in the .idx")
        i = hit[0]
        out.append((f, (int(lines[i][1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1)))
    return out


def fetch_gefs(date: str, hh: str, step: int) -> tuple[list, int]:
    jobs = [(m, gefs_url(date, hh, mem, step)) for m, mem in enumerate(GEFS_MEMBERS)]
    with ThreadPoolExecutor(16) as ex:
        ranges = list(ex.map(lambda j: gefs_ranges(j[1], step), jobs))
        flat = [(m, url, f, r) for (m, url), rs in zip(jobs, ranges) for f, r in rs]
        blobs = list(ex.map(lambda j: s3_get(j[1], j[3]), flat))
    return [(f, m, b) for (m, _, f, _), b in zip(flat, blobs)], sum(map(len, blobs))


def geps_file(date: str, hh: str, var: str, step: int) -> bytes:
    import datamart

    blob = datamart.get(datamart.url(date, var, step, hh), timeout=300)
    if blob is None:
        raise RuntimeError(f"GEPS {date} {hh}Z +{step} h: no {var} on the Datamart")
    return blob


def fetch_geps(date: str, hh: str, step: int) -> tuple[list, int]:
    out, n = [], 0
    for f, var in GEPS_VAR.items():
        blob = geps_file(date, hh, var, step)
        n += len(blob)
        out += [(f, None, m) for m in split_grib(blob)]
    return out, n


def decode(msg: bytes, member_key: str) -> tuple[int, np.ndarray, float]:
    """(member number (0 without member_key), the field on its grid (90N first, columns from 180W) float32, grid step)
    of one global message; eccodes runs here, in the calling thread only (decoding in the fetch threads has crashed the interpreter here)."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
        inc = eccodes.codes_get_double(gid, "iDirectionIncrementInDegrees")
        up = bool(eccodes.codes_get(gid, "jScansPositively"))
        try:
            num = int(eccodes.codes_get(gid, member_key)) if member_key else 0
        except eccodes.KeyValueNotFoundError:
            num = 0  # the GEPS control carries no perturbation number
        v = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(gid)
    if abs(ni * inc - 360) > 1e-3 or abs((nj - 1) * inc - 180) > 1e-3 or abs(abs(lat0) - 90) > 1e-3:
        raise RuntimeError(f"not a global grid: {ni} x {nj} at {inc} deg from {lat0}, {lon0}")
    return num, roll_rows(v, lat0, lon0, inc, up), inc


def to_units(f: str, v: np.ndarray) -> tuple[np.ndarray, str]:
    """A field in K / m / hPa, whatever the file holds (read from its magnitude: ECCC files have carried decametres
    under a metres label, AIFS publishes geopotential), and what it was."""
    med = float(np.median(v[::7, ::7]))
    if f == "t2m":
        return v, "K"
    if f == "msl":
        return (v / 100.0, "Pa") if med > 2000 else (v, "hPa")
    if med > 20000:
        return v / G, "m2 s-2"
    if med < 1000:
        return v * 10.0, "dam"
    return v, "m"


@dataclass(frozen=True, eq=False)
class Model:
    key: str
    label: str
    members: int
    first: int  # the GRIB number of the first member (IFS-ENS pf 1..50; the others' control is 0)
    grids: dict
    member_key: str  # the GRIB key that numbers the members (GEFS: one file per member, numbered by the fetch)
    fetch: Callable
    source: str
    ready: Callable  # (date, hh, last step, fields) -> the run is complete to its last step
    runs: dict  # start hour "HH" -> (last step, hours after the start by which the run is usually complete)

    def last(self, hh: str) -> int:
        return self.runs[hh][0]


def ready_ecmwf(model: str) -> Callable:
    def ready(date: str, hh: str, last: int, fields=FIELDS) -> bool:
        import cycles

        steps = tuple(range(6, last + 1, 6))
        if model == "ifs":  # the 0.1-deg surface files for 2 m temperature, the 0.25-deg ones for the rest
            needs = (cycles.Need("ifs", "ef", step=last, every=steps, resol="0p1", levtype="sfc"),) * ("t2m" in fields)
            needs += (cycles.Need("ifs", "ef", step=last, every=steps),) * bool({"z500", "msl"} & set(fields))
        else:
            needs = tuple(cycles.Need("aifs-ens", k, step=last, every=steps) for k in ("pf", "cf"))
        ok, why = cycles.is_ready(date, hh, needs)
        log(f"  {model} {date} {hh}Z: {why}")
        return ok

    return ready


def ready_gefs(date: str, hh: str, last: int, fields=FIELDS) -> bool:
    from webget import get

    try:
        for mem in (GEFS_MEMBERS[0], GEFS_MEMBERS[-1]):
            get(gefs_url(date, hh, mem, last) + ".idx", timeout=30, tries=2)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  gefs {date} {hh}Z: +{last} h not there ({str(e)[:80]})")
        return False


def ready_geps(date: str, hh: str, last: int, fields=FIELDS) -> bool:
    import datamart

    ok = all(datamart.exists(datamart.url(date, v, last, hh)) for v in GEPS_VAR.values())
    if not ok:
        log(f"  geps {date} {hh}Z: +{last} h not there")
    return ok


MODELS = {
    "ifs": Model("ifs", "IFS-ENS", 50, 1, {"t2m": G10, "z500": G25, "msl": G25}, "number", fetch_ifs,
                 "IFS-ENS perturbed members: 2 m temperature 0.1°, 500 hPa height and sea-level pressure 0.25° "
                 "(ECMWF open data, Google Cloud mirror)", ready_ecmwf("ifs"),
                 {"00": (LAST_STEP, 13.3), "06": (144, 11.0), "12": (LAST_STEP, 13.3), "18": (144, 11.0)}),
    "aifs": Model("aifs", "AIFS-ENS", 51, 0, dict.fromkeys(FIELDS, G25), "number", fetch_aifs,
                  "AIFS-ENS control and 50 perturbed members, 0.25° (ECMWF open data, Google Cloud mirror)",
                  ready_ecmwf("aifs"), dict.fromkeys(("00", "06", "12", "18"), (LAST_STEP, 7.5))),
    "gefs": Model("gefs", "GEFS", 31, 0, dict.fromkeys(FIELDS, G50), "", fetch_gefs,
                  "GEFS control and 30 perturbed members, 0.5° (NOAA open data, AWS)", ready_gefs,
                  dict.fromkeys(("00", "06", "12", "18"), (384, 6.0))),
    "geps": Model("geps", "GEPS", 21, 0, dict.fromkeys(FIELDS, G50), "perturbationNumber", fetch_geps,
                  "GEPS control and 20 perturbed members, 0.5° (MSC Datamart)", ready_geps,
                  dict.fromkeys(("00", "12"), (384, 5.3))),
}  # fmt: skip


# ── orography ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def ifs_orography(date: str, hh: str) -> np.ndarray:
    """(721, 1440) m: the IFS's orography, its control's surface geopotential at step 0 of the newest 00/12Z run (the
    oper stream's) on the Google Cloud mirror up to this one (it does not change), 0.25 deg."""
    import rangefetch as rf

    t0 = pd.Timestamp(f"{date} {hh}:00").floor("12h")
    for back in range(5):
        t = t0 - pd.Timedelta(hours=12 * back)
        d, h = f"{t:%Y%m%d}", f"{t:%H}"
        try:
            idx = rf.fetch_index(d, h, "ifs", 0, "fc", stream="oper")
            break
        except rf.IndexUnavailable:
            continue
    else:
        raise SystemExit(f"no IFS control run on the mirror in the 48 h before {t0:%Y-%m-%d %H}Z")
    want = [e for e in idx if e.get("param") == "z" and e.get("levtype") == "sfc"]
    if not want:
        raise SystemExit(f"IFS control {d} {h}Z: no surface geopotential in the step-0 index")
    blob = rf.fetch_ranges(rf.path_for(d, h, "ifs", 0, "fc", stream="oper") + ".grib2", rf.coalesce(want))
    return decode(split_grib(blob)[0], "number")[1] / G


def model_orography(mod: Model, date: str, hh: str) -> np.ndarray:
    """The model's own surface height (m) on its 2 m grid: AIFS-ENS its control's step-0 surface geopotential, GEFS its
    control's HGT at +0 h, GEPS the control's HGT at +0 h (the Datamart's in DECAMETRES under a metres label, 24 Oct
    2026: read from the field's maximum, so a corrected encoding is taken as it comes)."""
    if mod.key == "aifs":
        import rangefetch as rf

        idx = rf.fetch_index(date, hh, "aifs-ens", 0, "cf")
        want = [e for e in idx if e.get("param") == "z" and e.get("levtype") == "sfc"]
        blob = rf.fetch_ranges(rf.path_for(date, hh, "aifs-ens", 0, "cf") + ".grib2", rf.coalesce(want))
        return decode(split_grib(blob)[0], "number")[1] / G
    if mod.key == "gefs":
        url = gefs_url(date, hh, GEFS_MEMBERS[0], 0)
        lines = [ln.split(":") for ln in s3_get(url + ".idx").decode().splitlines() if ln.strip()]
        i = next(i for i, p in enumerate(lines) if p[3] == "HGT" and p[4] == "surface")
        return decode(s3_get(url, (int(lines[i][1]), int(lines[i + 1][1]) - 1)), "")[1]
    zg = dict(decode(m, "perturbationNumber")[:2] for m in split_grib(geps_file(date, hh, "HGT_SFC_0", 0)))[0]
    return zg * 10.0 if zg.max() < 1000.0 else zg


# ── statistics ────────────────────────────────────────────────────────────────────────────────────────────────────────
def sign_k(n: int) -> int:
    """The fewest of n members that must share a sign for a two-sided sign test against an even split to pass at 5 %."""
    from scipy.stats import binomtest

    return next(k for k in range(n // 2 + 1, n + 1) if binomtest(k, n).pvalue < 0.05)


def agreeing(x: np.ndarray, mem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the mean of the members mem of x (member, ...), how many of them share its sign); a member at a time, so a
    0.1-deg global field needs no member-sized temporaries."""
    m = np.zeros(x.shape[1:], np.float64)
    for i in mem:
        m += x[i]
    m = (m / mem.size).astype(np.float32)
    s = np.sign(m)
    n = np.zeros(m.shape, np.uint8)
    for i in mem:
        n += (np.sign(x[i]) == s) & (s != 0)
    return m, n


def welch_p(m1, s1, n1, m0, s0, n0) -> np.ndarray:
    """Two-sided p of Welch's t-test for the difference of two runs' member means (means m, standard deviations s,
    sizes n), Satterthwaite's degrees of freedom; NaN where neither run's members vary."""
    from scipy.stats import t as student

    v1, v0 = s1.astype(np.float64) ** 2 / n1, s0.astype(np.float64) ** 2 / n0
    v = v1 + v0
    tested = np.isfinite(m1) & np.isfinite(m0) & (v > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(tested, (m1 - m0) / np.sqrt(v), 0.0)
        df = np.where(tested, v * v / (v1 * v1 / (n1 - 1) + v0 * v0 / (n0 - 1)), 1.0)
    return np.where(tested, 2 * student.sf(np.abs(t), df), np.nan)


def significant(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg at FDR_Q over the map's tested points."""
    from quantmet.stats import benjamini_hochberg

    out = np.zeros(p.shape, bool)
    fam = np.isfinite(p)
    out[fam] = benjamini_hochberg(p[fam], alpha=FDR_Q)
    return out


# ── days and spans ────────────────────────────────────────────────────────────────────────────────────────────────────
def full_days(init: pd.Timestamp, last: int = LAST_STEP) -> list[tuple[pd.Timestamp, list[int]]]:
    """[(day, steps of its four fields)] for the run's calendar UTC days whose four fields are all forecast steps up to
    `last`."""
    out = []
    for day in pd.date_range(init.normalize(), (init + pd.Timedelta(hours=last)).normalize()):
        steps = [int((day + pd.Timedelta(hours=h) - init) / pd.Timedelta(hours=1)) for h in SAMPLES]
        if steps[0] >= 6 and steps[-1] <= last:
            out.append((day, steps))
    return out


def windows(ndays: int, anchor: int = 0) -> list[tuple[int, int]]:
    """(first, last) day indices of the PERIOD-day windows from `anchor`, the last one cut at the run's last day."""
    return [(k, min(k + PERIOD - 1, ndays - 1)) for k in range(anchor, ndays, PERIOD)][:3]


def spans(ndays: int) -> list[tuple[str, int, int]]:
    """(name, first day, last day) of the maps: the windows from day 1, then each day."""
    return [(f"P{i + 1}", k0, k1) for i, (k0, k1) in enumerate(windows(ndays))] + [
        (f"D{k + 1:02d}", k, k) for k in range(ndays)
    ]


def key(days, k0: int, k1: int) -> str:
    return f"{days[k0][0]:%Y-%m-%d}/{days[k1][0]:%Y-%m-%d}"


def span_text(days, k0: int, k1: int) -> tuple[str, str]:
    """(e.g. 'days 1–5', 'Fri Oct 23 – Tue Oct 27') or ('day 3', 'Sun Oct 25')."""
    d0, d1 = days[k0][0], days[k1][0]
    if k0 == k1:
        return f"day {k0 + 1}", f"{d0:%a %b %-d}"
    return f"days {k0 + 1}–{k1 + 1}", f"{d0:%a %b %-d} – {d1:%a %b %-d}"


def prev_row(days, k0: int, k1: int, prev: dict) -> tuple[int, int] | None:
    """(last day, row of the previous run's statistics) for the span k0..k1 cut to the days the previous run holds;
    None when it does not hold day k0 or has no statistics for the cut span."""
    last = max(k.split("/")[1] for k in prev["keys"])
    while k1 >= k0 and f"{days[k1][0]:%Y-%m-%d}" > last:
        k1 -= 1
    row = prev["keys"].get(key(days, k0, k1)) if k1 >= k0 else None
    return None if row is None else (k1, row)


def stat_spans(ndays: int) -> list[tuple[int, int]]:
    """The spans whose statistics the runs 6, 12 and 48 h later compare with: each day and the windows from day 1, 2 and
    3 (their days 1-5 start on this run's day 1 or 2, and two days later on its day 3)."""
    return [(k, k) for k in range(ndays)] + sorted(set(windows(ndays, 0) + windows(ndays, 1) + windows(ndays, 2)))


def save_stats(path: Path, init: pd.Timestamp, days: list, S: dict, done: dict) -> None:
    """Per field and span: the mean (float32) and standard deviation (float16) of the members' 0.5-deg cell means and
    how many members (complete on each of its days) they are from. The other fields' statistics of the same run already
    in the file (an IFS-ENS first stage) are kept."""
    sp = stat_spans(len(days))
    out = {"init": f"{init:%Y%m%d%H}", "keys": np.array([key(days, k0, k1) for k0, k1 in sp])}
    if path.exists():
        with np.load(path) as z:
            if str(z["init"]) == out["init"] and list(z["keys"]) == list(out["keys"]):
                out |= {
                    k: z[k]
                    for k in z.files
                    if k.split("_", 1)[0] in ("mean", "sd", "n") and k.split("_", 1)[1] not in S
                }
    for f in S:
        mean, sd, n = [], [], []
        for k0, k1 in sp:
            mem = np.flatnonzero(done[f][:, k0 : k1 + 1].all(1))
            x = S[f][mem, k0 : k1 + 1].mean(1)
            n.append(mem.size)
            mean.append(x.mean(0) if mem.size > 1 else np.full(x.shape[1:], np.nan, np.float32))
            sd.append((x.std(0, ddof=1) if mem.size > 1 else np.full(x.shape[1:], np.nan)).astype(np.float16))
        out |= {f"mean_{f}": np.stack(mean).astype(np.float32), f"sd_{f}": np.stack(sd), f"n_{f}": np.array(n)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    log(f"  statistics for the next run: {path} ({path.stat().st_size / 1e6:.0f} MB, {len(sp)} spans)")


def load_stats(path: Path | None, want_init: pd.Timestamp) -> dict | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as z:
        if str(z["init"]) != f"{want_init:%Y%m%d%H}":
            log(f"  {path}: holds run {z['init']}, not {want_init:%Y%m%d%H}; no change maps")
            return None
        out = {"keys": {str(k): i for i, k in enumerate(z["keys"])}}
        for f in FIELDS:
            if f"mean_{f}" in z:
                out[f] = {"mean": z[f"mean_{f}"], "sd": z[f"sd_{f}"], "n": z[f"n_{f}"]}
        return out


# ── encoding ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def code_signed(v: np.ndarray, step: float) -> np.ndarray:
    """128 + v / step, 1..255; 0 where v is not finite."""
    with np.errstate(invalid="ignore"):
        c = np.clip(np.rint(v / step) + 128, 1, 255)
    return np.where(np.isfinite(v), c, 0).astype(np.uint8)


def code_mean(v: np.ndarray, lo: float, step: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        c = np.clip(np.rint((v - lo) / step) + 1, 1, 255)
    return np.where(np.isfinite(v), c, 0).astype(np.uint8)


def webp(arr: np.ndarray) -> bytes:
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, "WEBP", lossless=True, quality=100, method=4)
    return b.getvalue()


def tiles(img: np.ndarray) -> dict:
    """{name: image} of a 0.1-deg frame: the overview (every 5th point) and the 6 x 6 tiles, each with one more row and
    column shared with its neighbour (the last column wraps)."""
    ny, nx = img.shape[:2]
    th, tw = TILE
    out = {"o": img[::5, ::5]}
    wrap = np.concatenate([img, img[:, :1]], axis=1)
    for r in range((ny - 1) // th):
        for c in range(nx // tw):
            out[f"{r}{c}"] = wrap[r * th : (r + 1) * th + 1, c * tw : (c + 1) * tw + 1]
    return out


class Writer:
    """Encodes and writes the frames in worker threads (Pillow releases the GIL while it encodes)."""

    def __init__(self, outdir: Path, threads: int):
        self.outdir, self.ex, self.futs, self.bytes = outdir, ThreadPoolExecutor(threads), [], 0

    def put(self, rel: str, img: np.ndarray, tiled: bool) -> None:
        parts = tiles(img) if tiled else {"": img}
        for name, a in parts.items():
            path = self.outdir / (f"{rel}/{name}.webp" if tiled else f"{rel}.webp")
            self.futs.append(self.ex.submit(self._write, path, np.ascontiguousarray(a)))

    def _write(self, path: Path, a: np.ndarray) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        b = webp(a)
        path.write_bytes(b)
        return len(b)

    def drain(self) -> int:
        """Wait for the frames put so far; the bytes written so far."""
        self.bytes += sum(f.result() for f in self.futs)
        self.futs = []
        return self.bytes

    def close(self) -> int:
        self.drain()
        self.ex.shutdown()
        return self.bytes


def following(fetch: Callable, minutes: float) -> Callable:
    """fetch(date, hh, step), waiting for a step its source does not have yet (a run followed as it lands), for up to
    `minutes` from now in all."""
    end = time.time() + 60 * minutes

    def get(date: str, hh: str, step: int):
        while True:
            try:
                return fetch(date, hh, step)
            except Exception as e:  # noqa: BLE001
                if time.time() > end:
                    raise
                log(f"  +{step} h not complete on the source yet ({str(e)[:80]}); again in a minute")
                time.sleep(60)

    return get


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class Field:
    """One field's state through a run: the step sums of the day, the window being summed, the 0.5-deg member means of
    every day, the normals."""

    def __init__(self, name: str, g: Grid, n: int, ndays: int, normals: R.Normals | None, high: np.ndarray | None):
        self.name, self.g, self.n, self.normals, self.high = name, g, n, normals, high
        self.acc = np.zeros((n, *g.shape), np.float32)
        self.cnt = np.zeros(n, int)
        self.win = np.zeros((n, *g.shape), np.float32)  # member anomalies (or values, without normals) summed
        self.win_ok = np.ones(n, bool)
        self.win_norm = np.zeros(g.shape, np.float32)
        self.S = np.full((n, ndays, *G50.shape), np.nan, np.float32)  # member daily means on the 0.5-deg cells
        self.done = np.zeros((n, ndays), bool)
        self.N50 = np.full((ndays, *G50.shape), np.nan, np.float32) if normals is not None else None  # the day normals


def frame(name: str, anom: np.ndarray | None, agree: np.ndarray | None, mean: np.ndarray, high=None) -> np.ndarray:
    """An anomaly frame (a/, r/): R anomaly, G members agreeing, B mean; `high` masks the anomaly."""
    e = ENC[name]
    m = mean - 273.15 if name == "t2m" else mean
    if anom is None:
        r = np.zeros(mean.shape, np.uint8)
        g = np.full(mean.shape, 255, np.uint8)
    else:
        r = code_signed(anom, e["anom"])
        g = agree.copy()
        if high is not None:
            r[high], g[high] = 0, 255
    return np.dstack([r, g, code_mean(m, e["lo"], e["mean"])])


def run(a) -> int:
    mod = MODELS[a.model]
    fields = tuple(f for f in FIELDS if f in a.fields.split(",")) if a.fields else FIELDS
    if fields != FIELDS and mod.key != "ifs":
        raise SystemExit("--fields: only IFS-ENS publishes in stages")
    if a.merge and "z500" in fields:
        raise SystemExit("--merge adds 2 m temperature to a run published without it")
    key_c = a.cycle
    init = pd.Timestamp(f"{key_c[:8]} {key_c[8:]}:00")
    date, hh = key_c[:8], key_c[8:]
    prev_init = init - pd.Timedelta(hours=12)
    days = full_days(init, mod.last(hh))
    if os.environ.get("ENSMAPS_DAYS"):  # testing only: the first n days
        days = days[: int(os.environ["ENSMAPS_DAYS"])]
    site = Path(a.site)
    outdir = site / DATA / mod.key / key_c
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t_start = time.time()
    log(
        f"{mod.label} {init:%Y-%m-%d %H}Z: {len(days)} full days {days[0][0]:%Y-%m-%d}..{days[-1][0]:%Y-%m-%d}, "
        f"{', '.join(fields)}{' (added to the published run)' if a.merge else ''}"
    )

    # normals on the model's grids
    t2m_normals = R.Normals.t2m(a.normals) if "t2m" in fields and a.normals and a.normals.exists() else None
    if t2m_normals is None and "t2m" in fields:
        log(f"::notice::no 2 m temperature normals ({a.normals}): its anomaly maps are left out")
    z_ifs = ifs_orography(date, hh)
    z_t2m = None
    if mod.key != "ifs":
        z_t2m = model_orography(mod, date, hh)
        z_ref = to_half(z_ifs, 0.25) if mod.grids["t2m"].res == 0.5 else z_ifs
        dz = z_t2m - z_ref
        log(
            f"  surface height less the IFS's: mean |dz| {np.mean(np.abs(dz)):.0f} m, largest {np.max(np.abs(dz)):.0f} m"
        )
    normals, high = {}, {}
    spread = {f: spread_on(f, mod.grids[f]) for f in ("z500", "msl") if f in fields}
    for f in fields:
        g = mod.grids[f]
        if f == "t2m":
            shift = None if z_t2m is None else (-LAPSE * dz).astype(np.float32)
            normals[f] = t2m_on(t2m_normals, g, shift) if t2m_normals else None
        else:
            normals[f] = upper_on(f, g)
        if f == "msl":  # the model's surface height on the pressure grid
            zs = z_ifs if mod.key == "ifs" else z_t2m
            high[f] = zs > HIGH_M
    n = mod.members
    F = {f: Field(f, mod.grids[f], n, len(days), normals[f], high.get(f)) for f in fields}
    W = Writer(outdir, a.threads)
    Z25 = np.full((n, len(days), 73, 144), np.nan, np.float32)  # the regimes' field: 500 hPa height on 2.5 deg
    M25 = np.full((n, len(days), 73, 144), np.nan, np.float32)  # and the clusters': sea-level pressure
    spans_meta: dict = {}
    min_n = int(np.ceil(MIN_SHARE * n))

    def draw(fd: Field, name: str, k0: int, k1: int, x: np.ndarray, ok: np.ndarray, norm: np.ndarray | None):
        """One span's anomaly frame from the members' anomalies x (member, ...) (values when norm is None)."""
        mem = np.flatnonzero(ok)
        entry = spans_meta.setdefault(name, {"key": name, "first": f"{days[k0][0]:%Y-%m-%d}",
                                             "last": f"{days[k1][0]:%Y-%m-%d}", "n": {}, "k": {}})  # fmt: skip
        if mem.size < min_n:
            log(f"  {fd.name} {name}: only {mem.size} complete members - not drawn")
            return
        m, agree = agreeing(x, mem)
        if norm is None:
            img = frame(fd.name, None, None, m)
        else:
            img = frame(fd.name, m, agree, m + norm, fd.high)
            entry["k"][fd.name] = sign_k(int(mem.size))
            if fd.name in spread:  # the same frame standardized by the sd of the span's own averaging period
                sd = spread[fd.name].sd([days[k][0] + pd.Timedelta(hours=12) for k in range(k0, k1 + 1)], k1 - k0 + 1)
                si = img.copy()
                si[..., 0] = code_signed(m / sd, ENC[fd.name]["sigma"])
                if fd.high is not None:
                    si[..., 0][fd.high] = 0
                W.put(f"{fd.name}/s/{name}", si, False)
        entry["n"][fd.name] = int(mem.size)
        W.put(f"{fd.name}/a/{name}", img, fd.g.tiled)

    def entry_of(through: int | None, **extra) -> dict:
        """The run's manifest entry: every span drawn, or (a checkpoint) those to day index `through`, which it names."""
        out_spans = []
        for name, k0, k1 in spans(len(days)):
            e = spans_meta.get(name)
            if e is None or not e["n"] or (through is not None and k1 > through):
                continue
            what, when = span_text(days, k0, k1)
            e["label"] = f"{what[0].upper()}{what[1:]} · {when}"
            for c in [v for k in ("change", "change6", "change48") for v in e.get(k, {}).values()]:
                c1 = next(k for k, (d, _) in enumerate(days) if f"{d:%Y-%m-%d}" == c["last"])
                cw, cwhen = span_text(days, k0, c1)
                c["label"] = f"{cw[0].upper()}{cw[1:]} · {cwhen}"
            out_spans.append(e)
        out = dict(
            cycle=key_c,
            init=f"{init:%Y-%m-%dT%H}:00Z",
            last_step=mod.last(hh),
            days=[f"{d:%Y-%m-%d}" for d, _ in days],
            spans=out_spans,
            fields={
                f: {
                    "unit": UNIT[f],
                    "grid": mod.grids[f].desc(),
                    "tiles": {"rows": TILE[0], "cols": TILE[1], "overview": 5} if mod.grids[f].tiled else None,
                    "change_grid": G50.desc(),
                    "anomaly": F[f].normals is not None,
                    "sigma": f in spread,
                    "units_read": sorted(units.get(f, [])),
                }
                for f in F
            },
            enc=ENC,
            high_m=HIGH_M,
            fdr_q=FDR_Q,
            normals={
                "t2m": t2m_normals.meta.get("label") if t2m_normals else None,
                "upper": R.UPPER_META["label"],
            },
            prev=None,
            prev6=None,
            prev48=None,
            regimes=None,
            made=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        )
        if through is not None:
            out["through"] = f"{days[through][0]:%Y-%m-%d}"
        return out | extra

    def checkpoint(k: int) -> None:
        """Publish the days to index k and the windows they close (a run followed as it lands): the manifest written,
        then --checkpoint run to put the frames in the store and the manifest on main."""
        seconds = round(time.time() - t0)
        entry = entry_of(k, fetch={"bytes": nbytes, "messages": nmsg, "seconds": seconds}, bytes=W.drain(),
                         wall_s=round(time.time() - t_start))  # fmt: skip
        write_manifest(site, mod, entry, a.manifest_in, a.lists)
        env = dict(os.environ, ENSMAPS_LISTS=str(a.lists), ENSMAPS_THROUGH=entry["through"])
        r = subprocess.run(a.checkpoint, shell=True, env=env, check=False)
        log(
            f"  through {entry['through']}: {'published' if r.returncode == 0 else f'NOT published (exit {r.returncode})'}"
        )

    last_of = {st[-1]: k for k, (_, st) in enumerate(days)}
    wins = {k1: (i, k0) for i, (k0, k1) in enumerate(windows(len(days)))}
    todo = deque(s for _, st in days for s in st)
    nbytes = nmsg = 0
    units: dict = {}
    t0 = time.time()
    fetch = functools.partial(mod.fetch, fields=fields) if mod.key == "ifs" else mod.fetch  # IFS-ENS: this stage's
    if a.follow:
        fetch = following(fetch, a.follow)
    with ThreadPoolExecutor(AHEAD) as ex:
        live: deque = deque()
        while todo or live:
            while todo and len(live) < AHEAD:
                s = todo.popleft()
                live.append((s, ex.submit(fetch, date, hh, s)))
            s, fut = live.popleft()
            msgs, nb = fut.result()
            nbytes += nb
            nmsg += len(msgs)
            seen = set()
            for f, m_idx, msg in msgs:
                if f not in F:
                    continue
                num, v, _ = decode(msg, mod.member_key)
                i = (m_idx if m_idx is not None else num) - (0 if m_idx is not None else mod.first)
                fd = F[f]
                if not (0 <= i < n) or (f, i) in seen:
                    continue
                if v.shape != fd.g.shape:
                    raise RuntimeError(f"{f} member {i} +{s} h: grid {v.shape}, not {fd.g.shape}")
                v, u = to_units(f, v)
                units.setdefault(f, set()).add(u)
                seen.add((f, i))
                fd.acc[i] += v
                fd.cnt[i] += 1
            if s not in last_of:
                continue
            k = last_of[s]
            day = days[k][0]
            for fd in F.values():
                ok = fd.cnt == len(SAMPLES)
                fd.acc *= 1.0 / len(SAMPLES)  # the members' daily means
                fd.acc[~ok] = np.nan
                for i in np.flatnonzero(ok):
                    fd.S[i, k] = to_half(fd.acc[i], fd.g.res)
                    if fd.name in ("z500", "msl"):
                        (Z25 if fd.name == "z500" else M25)[i, k] = to25(fd.acc[i], fd.g.res)
                fd.done[:, k] = ok
                norm = day_normal(fd.normals, day) if fd.normals is not None else None
                if norm is not None:
                    fd.N50[k] = to_half(norm, fd.g.res)
                    fd.acc -= norm[None]
                draw(fd, f"D{k + 1:02d}", k, k, fd.acc, ok, norm)
                if k in [k0 for k0, _ in windows(len(days))]:
                    fd.win[:] = 0.0
                    fd.win_ok[:] = True
                    fd.win_norm[:] = 0.0
                fd.win_ok &= ok
                for i in np.flatnonzero(ok):
                    fd.win[i] += fd.acc[i]
                if norm is not None:
                    fd.win_norm += norm
                if k in wins:
                    i, k0 = wins[k]
                    nd = k - k0 + 1
                    fd.win *= 1.0 / nd
                    draw(fd, f"P{i + 1}", k0, k, fd.win, fd.win_ok, fd.win_norm / nd if norm is not None else None)
                fd.acc[:] = 0.0
                fd.cnt[:] = 0
            gb, sec = nbytes / 1e9, time.time() - t0
            log(
                f"  day {k + 1} {day:%Y-%m-%d}: {int(next(iter(F.values())).done[:, k].sum())} members, {gb:.2f} GB, {sec:.0f} s"
            )
            if a.checkpoint and not a.merge and (k + 1) % PERIOD == 0 and k < len(days) - 1:
                checkpoint(k)
    fetched = {"bytes": nbytes, "messages": nmsg, "seconds": round(time.time() - t0)}
    for fd in F.values():  # the 0.1 / 0.25-deg sums are done with
        fd.acc = fd.win = None

    def change_maps(prev: dict | None, sub: str, meta: str) -> int:
        """The change from a run's statistics (load_stats) on the 0.5-deg cells: frames <field>/<sub>/<span>, the
        spans' `meta` entries."""
        n_maps = 0
        for name, k0, k1 in spans(len(days)) if prev is not None else ():
            cut = prev_row(days, k0, k1, prev)
            if cut is None:
                continue
            c1, row = cut
            for f, fd in F.items():
                if f not in prev or prev[f]["n"][row] < min_n:
                    continue
                mem = np.flatnonzero(fd.done[:, k0 : c1 + 1].all(1))
                if mem.size < min_n:
                    continue
                y = fd.S[mem, k0 : c1 + 1].mean(1)
                my, sy = y.mean(0), y.std(0, ddof=1)
                m0, s0, n0 = prev[f]["mean"][row], prev[f]["sd"][row].astype(np.float32), int(prev[f]["n"][row])
                sig = significant(welch_p(my, sy, mem.size, m0, s0, n0))
                d = my - m0
                img = np.dstack([code_signed(d, ENC[f]["change"]), sig.astype(np.uint8), np.zeros(d.shape, np.uint8)])
                W.put(f"{f}/{sub}/{name}", img, False)
                spans_meta[name].setdefault(meta, {})[f] = {
                    "last": f"{days[c1][0]:%Y-%m-%d}",
                    "n": int(mem.size),
                    "n0": n0,
                    "sig": round(float(sig.mean()), 4),
                }
                n_maps += 1
        return n_maps

    # change from the runs 12 h and 6 h earlier
    prev = load_stats(a.prev, prev_init)
    if prev is None:
        log(f"::notice::no statistics of the run 12 h earlier ({prev_init:%Y-%m-%d %H}Z): no change maps this run")
    prev6_init = init - pd.Timedelta(hours=6)
    prev6 = load_stats(a.prev6, prev6_init) if a.prev6 is not None else None
    prev48_init = init - pd.Timedelta(hours=48)
    prev48 = load_stats(a.prev48, prev48_init) if a.prev48 is not None else None
    changes = (
        change_maps(prev, "c", "change") + change_maps(prev6, "c6", "change6") + change_maps(prev48, "c48", "change48")
    )
    if a.save:
        save_stats(a.save, init, days, {f: fd.S for f, fd in F.items()}, {f: fd.done for f, fd in F.items()})

    # weather regimes, from the 500 hPa members held (ensmaps_regimes.py); not again when 2 m temperature is added
    import ensmaps_regimes as ER

    man_in = load_manifest(a.manifest_in, mod)
    before = next((c for c in man_in.get("cycles", []) if c["cycle"] <= f"{prev_init:%Y%m%d%H}"), None)  # 12 h or more
    regimes = None
    if "z500" in F and "msl" in F:
        high50 = to_half(F["msl"].high.astype(np.float32), mod.grids["msl"].res) > 0.5
        regimes = ER.run(mod, key_c, init, days, Z25, F, high50, W, a.regimes_out, before)
    clusters_ok = "z500" in F and "msl" in F and mod.last(hh) == max(r[0] for r in mod.runs.values())
    if a.fields_out and clusters_ok:  # the clusters' fields: full runs only
        # (the members' 2.5-deg anomalies for the four-ensemble clusters, ensmaps_clusters.py; an IFS-ENS 06/18Z run
        # would cut its windows to day 5)
        import ensmaps_clusters as EC

        ok = F["z500"].done & F["msl"].done
        high25 = to25(F["msl"].high.astype(np.float32), mod.grids["msl"].res) > 0.5
        EC.save_fields(a.fields_out, mod, key_c, init, days, Z25, M25, ok, high25, a.cluster_drift)
    del Z25, M25
    nbytes_out = W.close()
    wall = time.time() - t_start
    log(f"{mod.label} {key_c}: {nbytes_out / 1e6:.0f} MB of frames ({changes} change maps), fetched "
        f"{nbytes / 1e9:.2f} GB in {nmsg} messages, wall {wall:.0f} s")  # fmt: skip
    for f in F:
        sz = sum(p.stat().st_size for p in (outdir / f).rglob("*.webp"))
        log(f"  {f:5s} {sz / 1e6:7.1f} MB")

    entry = entry_of(
        None,
        prev=f"{prev_init:%Y-%m-%dT%H}:00Z" if prev else None,
        prev6=f"{prev6_init:%Y-%m-%dT%H}:00Z" if prev6 else None,
        prev48=f"{prev48_init:%Y-%m-%dT%H}:00Z" if prev48 else None,
        regimes=regimes,
        fetch=fetched,
        bytes=nbytes_out,
        wall_s=round(wall),
    )
    if a.merge:
        old = next((c for c in man_in.get("cycles", []) if c["cycle"] == key_c), None)
        if old is None:
            raise SystemExit(f"--merge: {key_c} is not on the page (its first stage has not been published)")
        entry = merged(old, entry)
    write_manifest(site, mod, entry, a.manifest_in, a.lists)
    return 0


def merged(old: dict, new: dict) -> dict:
    """A run's manifest entry with a later stage's fields added: their spans' members, gates and changes, grids,
    normals; the earlier stage's regimes kept; the fetch, bytes and wall time summed."""
    out = dict(old, fields={**old["fields"], **new["fields"]}, made=new["made"])
    spans_old = {e["key"]: e for e in old["spans"]}
    for e in new["spans"]:
        o = spans_old.setdefault(e["key"], {k: e[k] for k in ("key", "first", "last", "label")} | {"n": {}, "k": {}})
        for k in ("n", "k", "change", "change6", "change48"):
            if e.get(k):
                o[k] = {**o.get(k, {}), **e[k]}
    out["spans"] = list(spans_old.values())
    for k in ("prev", "prev6", "prev48"):
        out[k] = old.get(k) or new.get(k)
    out["normals"] = {**old["normals"], **{k: v for k, v in new["normals"].items() if k.startswith("t2m")}}
    out["fetch"] = {k: old["fetch"].get(k, 0) + new["fetch"].get(k, 0) for k in ("bytes", "messages", "seconds")}
    out["bytes"], out["wall_s"] = old["bytes"] + new["bytes"], old["wall_s"] + new["wall_s"]
    return out


def load_manifest(path: Path | None, mod: Model) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {"model": mod.key, "label": mod.label, "source": mod.source, "cycles": []}


def retained(items: list[dict], when: str, now: str) -> list[dict]:
    """Of the items (each with a time `when`, "%Y-%m-%dT%H:%MZ"), the ones the store still keeps, newest first: every
    one from the RETAIN_MIN minutes before `now`, and the newest older one."""

    def t(s: str) -> dt.datetime:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%MZ")

    items = sorted(items, key=lambda x: x[when], reverse=True)
    cut = t(now) - dt.timedelta(minutes=RETAIN_MIN)
    return [x for x in items if t(x[when]) >= cut] + [x for x in items if t(x[when]) < cut][:1]


def write_manifest(site: Path, mod: Model, entry: dict, manifest_in: Path | None, lists: Path | None) -> None:
    """The KEEP_RUNS newest runs, newest first; the lists for publish_data.sh: the cycle to publish (none for a run
    older than those kept), the cycles to prune and the floor below which the store is pruned. A run the manifest
    stops naming stays in the store (`spare`, with the time it left) for RETAIN_MIN minutes and until a newer one has
    left before it (retained), so a page still holding an older manifest (the deploy and the browsers' cache lag the
    store by minutes) never names a pruned run."""
    mpath = site / "assets/models" / f"ensmaps_{mod.key}.json"
    man = load_manifest(manifest_in or mpath, mod)
    prior = {c["cycle"] for c in man.get("cycles", [])} | {x["cycle"] for x in man.get("spare", [])}
    cycles = sorted(
        [c for c in man.get("cycles", []) if c["cycle"] != entry["cycle"]] + [entry],
        key=lambda c: c["cycle"],
        reverse=True,
    )
    keep, drop = cycles[:KEEP_RUNS], cycles[KEEP_RUNS:]
    listed = {c["cycle"] for c in keep}
    was = [x for x in man.get("spare", []) if x["cycle"] not in listed]
    left = [{"cycle": c["cycle"], "since": entry["made"]} for c in drop if c["cycle"] != entry["cycle"]]
    spare = retained(left + was, "since", entry["made"])
    man.update(model=mod.key, label=mod.label, source=mod.source, cycles=keep, spare=spare, updated=entry["made"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, separators=(",", ":")))
    data = f"{DATA}/{mod.key}"
    if lists:
        lists.mkdir(parents=True, exist_ok=True)
        stale = entry["cycle"] not in listed
        held = listed | {x["cycle"] for x in spare}
        # a rebuild of a run already on the page adds and replaces its files, deleting none the older manifest names
        again = "+" if entry["cycle"] in prior else ""
        (lists / "publish.txt").write_text("" if stale else f"{again}{data}/{entry['cycle']}\n")
        (lists / "prune.txt").write_text(
            "".join(f"{data}/{x['cycle']}\n" for x in left + was if x["cycle"] not in held)
        )
        (lists / "floor.txt").write_text(f"{data}:{min(held)}\n")
    log(f"manifest {mpath.name}: keep {sorted(listed, reverse=True)}, in the store too {[x['cycle'] for x in spare]}")


def cmd_plan(a) -> int:
    """Print the newest complete cycle (of the model's start hours) not yet in the manifest (nothing if none; with
    --started, the newest whose first day is complete, to be followed as the rest lands), and its
    stage when it is one: "<cycle> z500,msl" when --fields asks for the first IFS-ENS stage only, "<cycle> t2m" when
    the newest run on the page is still without its 2 m temperature and those files are complete. With --cycle: that
    cycle, unless the manifest has it (complete) or a newer one (--force plans it anyway)."""
    mod = MODELS[a.model]
    man = load_manifest(Path(a.manifests) / f"ensmaps_{a.model}.json", mod)
    have = {c["cycle"] for c in man.get("cycles", [])}
    held = {c["cycle"]: set(c.get("fields", FIELDS)) for c in man.get("cycles", [])}
    partial = {c["cycle"] for c in man.get("cycles", []) if c.get("through")}  # checkpoints of a run being followed
    want = tuple(f for f in FIELDS if f in a.fields.split(",")) if a.fields else FIELDS

    def stage(fs) -> str:
        return "" if fs == FIELDS else " " + ",".join(fs)

    if a.cycle:
        if a.cycle in held and "t2m" not in held[a.cycle] and "t2m" in want and not a.force:
            print(f"{a.cycle} t2m")
        elif have and max(have) >= a.cycle and not a.force:
            log(f"{a.cycle}: the page already has {max(have)}")
        else:
            print(a.cycle + stage(want))
        return 0
    every = 24 // len(mod.runs)  # 6 or 12 h between starts
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    newest = now.replace(minute=0, second=0, microsecond=0, hour=now.hour // every * every)
    t_end = time.time() + 60 * a.wait_min
    while True:
        c = newest
        for _ in range(48 // every):
            k, d, h = f"{c:%Y%m%d%H}", f"{c:%Y%m%d}", f"{c:%H}"
            if k in partial:  # its follower died before the end: the whole run again once it is complete (not a second
                # follower, which would race the first)
                if not a.started and mod.ready(d, h, mod.last(h), want):
                    print(k + stage(want))
                return 0
            if k in held:  # the newest published: nothing newer is complete; its 2 m temperature may be due
                if "t2m" not in held[k] and "t2m" in want and mod.ready(d, h, mod.last(h), ("t2m",)):
                    print(f"{k} t2m")
                return 0
            if have and max(have) > k:
                return 0
            upto = full_days(pd.Timestamp(c), mod.last(h))[0][1][-1] if a.started else mod.last(h)
            if mod.ready(d, h, upto, want):
                print(k + stage(want))
                return 0
            c -= dt.timedelta(hours=every)
        # the newest run is due but its last step not there yet (a dispatch fired as an earlier step landed): wait
        due = now - newest >= dt.timedelta(hours=mod.runs[f"{newest:%H}"][1]) - dt.timedelta(minutes=30)
        if not due or time.time() >= t_end:
            return 0
        time.sleep(60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "assets" / "models")
    p.add_argument("--cycle", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--wait-min", type=float, default=0.0, help="wait up to this long for a run that is due")
    p.add_argument("--fields", default="", help="IFS-ENS: z500,msl plans the first stage only (default: every field)")
    p.add_argument("--started", action="store_true", help="plan a run once its first day is complete (to follow it)")
    r = sub.add_parser("run")
    r.add_argument("--model", choices=tuple(MODELS), required=True)
    r.add_argument("--cycle", required=True)
    r.add_argument("--site", type=Path, default=ROOT)
    r.add_argument("--normals", type=Path, help="the 2 m temperature normals (release ensmaps-ref-v1)")
    r.add_argument("--prev", type=Path, help="the statistics of the run 12 h earlier (save_stats)")
    r.add_argument("--prev6", type=Path, help="the statistics of the run 6 h earlier")
    r.add_argument("--prev48", type=Path, help="the statistics of the run 48 h earlier (the trend over two days)")
    r.add_argument("--save", type=Path, help="write this run's statistics here")
    r.add_argument("--manifest-in", type=Path, help="the manifest on main (default: the site's)")
    r.add_argument("--regimes-out", type=Path, default=ROOT / "assets" / "regimes" / "data",
                   help="where the regime labels (ensmaps_members_<model>.json) and drift archive go")  # fmt: skip
    r.add_argument("--lists", type=Path, help="write publish.txt, prune.txt and floor.txt here")
    r.add_argument("--fields-out", type=Path, help="write the members' 2.5-deg anomalies for the clusters here")
    r.add_argument("--cluster-drift", type=Path, help="the clusters' drift archive of this model (read, and extended)")
    r.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    r.add_argument("--fields", default="", help="IFS-ENS stages: z500,msl first, then t2m with --merge (default all)")
    r.add_argument("--merge", action="store_true", help="add the fields to the run's entry already on the page")
    r.add_argument("--follow", type=float, default=0.0, help="wait up to this many minutes for steps still landing")
    r.add_argument("--checkpoint", default="", help="shell command that publishes a checkpoint (with --follow)")
    a = ap.parse_args()
    if a.cmd == "plan":
        return cmd_plan(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
