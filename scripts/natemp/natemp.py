#!/usr/bin/env python3
"""North America 2 m temperature from the IFS-ENS on its 0.1-deg grid and from GEPS on its 0.5-deg grid: the members'
daily means against the 1991-2020 normal, and how the run moved them from the run 12 hours earlier.

Data: the 0.1-deg IFS-ENS surface files on the Google Cloud mirror (interpolated by ECMWF from the model's ~9 km grid;
the 0.1-deg files carry the perturbed members only), parameter 2t at the steps a full day needs, byte-ranged
(rangefetch: index -> the 50 messages -> coalesced ranges) and decoded in this thread; the grid's own points over the
box (natemp_ref.LAT, LON) are kept, nothing is regridded. A day is a calendar UTC day, the mean of 06, 12, 18 and 24 UTC:
a 00Z run has 15 full days, a 12Z run 14 (its first and last partial days are not read). Each member's daily mean less
the day's normal (natemp_ref.Normals: ERA5-Land, ERA5 over the sea) is its anomaly; without the normals (release
natemp-ref-v1) only the change frames are drawn, which need none: over the same days the normal cancels.

GEPS (--model geps): the control and 20 perturbed members on the MSC Datamart's 0.5-deg grid, the finest it publishes
(one allmbrs file per step, 2 m temperature 3-hourly to 192 h, 6-hourly after), through scripts/geps/datamart.py; the
same days, windows and change test, its own frames and statistics. Its normal is the 0.1-deg normal averaged over each
0.5-deg cell (scripts/lib/cellmeans.py) and moved by LAPSE from the normal's orography (the IFS's ~9 km orography, which
ERA5-Land shares: the IFS control's surface geopotential averaged over the cell) to GEPS's own surface height, so a GEPS
cell is compared with a normal at its own height. Its anomaly gate is the fewest of its members a two-sided 5 % sign
test tells from an even split (16 of 21).

Frames, per view: days 1-5, 6-10 and 11-15 of the run (from its first full day; the last can be shorter, its label
says which days), then each day.
  anom    the members' mean anomaly, coloured only where at least AGREE of the members give it its sign
  change  this run's members' mean less the previous run's (12 h earlier) over the same calendar days (the days both
          runs hold), coloured at every point and dotted where significant: Welch's t-test on the two runs' members
          (their means and standard deviations, the previous run's from its statistics file), Benjamini-Hochberg over
          the map's points at FDR_Q. Without the previous run's statistics there are no change frames.
Writes the frames (assets/sst/anim/natemp_anom, natemp_change; frames branch), assets/sst/anim/natemp_manifest.json,
the summary assets/sst/data/natemp.json, and this run's statistics for the next run (--save: per day and per window,
the mean (float32) and standard deviation (float16) of the members' daily-mean temperatures).

    python scripts/natemp/natemp.py resolve --wait-min 45      # prints "YYYYMMDD HH"
    python scripts/natemp/natemp.py run --date 20261023 --time 00 --normals ref/natemp_normals.npz \
        --prev stats/2026102212.npz --save stats/2026102300.npz
    python scripts/natemp/natemp.py resolve --model geps --wait-min 45
    python scripts/natemp/natemp.py run --model geps --date 20261023 --time 00 --normals ref/natemp_normals.npz \
        --prev geps_stats/2026102212.npz --save geps_stats/2026102300.npz
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before eccodes)

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
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
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ecmwf"), str(ROOT / "scripts" / "lib")]
import natemp_ref as R

ANIM = ROOT / "assets" / "sst" / "anim"
DATA = ROOT / "assets" / "sst" / "data"
MEMBERS = 50
MIN_MEMBERS = 40  # a run with fewer complete members is not drawn
# colour an anomaly only where this share of the members gives it its sign: the smallest share a two-sided sign test
# against an even split passes at 5 % for any 40 to 50 members (p 0.015-0.044)
AGREE = 2 / 3
FDR_Q = 0.10
PERIOD = 5  # days per window
SAMPLES = (6, 12, 18, 24)  # hours after a day's 00 UTC: the four fields of its mean
LAST_STEP = 360
DUE_H = 13.0  # a run's 0.1-deg ensemble is on the mirror ~13.3 h after its start
AHEAD = 2  # steps fetched ahead of the decoding
RANGE_WORKERS = 12  # ranged GETs per step (two steps in flight stay under the mirror's per-IP ceiling)
KF = 1.8  # K -> degF for anomalies and changes
GLOBAL_2T = ("2t", 3600, 1801, 0, 90.0)  # shortName, Ni, Nj, jScansPositively, first latitude
NY, NX = R.LAT.size, R.LON.size
# GEPS: the MSC Datamart's 0.5-deg grid (720 x 361), its points over the same box
LAT_G = np.round(np.arange(1540, 179, -10) / 20.0, 1)  # 77N..9N
LON_G = np.round(np.arange(-360, -51) / 2.0, 1)  # 180W..26W
GEPS_MEMBERS, GEPS_MIN = 21, 16  # the control (number 0) and 20 perturbed members
GEPS_2T, GEPS_HGT = "TMP_TGL_2m", "HGT_SFC_0"
GEPS_DUE_H = 5.3  # a GEPS run is complete on the Datamart ~5 h 05 after its start
LAPSE = 6.5e-3  # K/m: the GEPS normal is moved to GEPS's surface height at this rate
G = 9.80665
DAY_NOTE = "a day is the mean of 06, 12, 18 and 24 UTC · {} grid"
CHANGE_NOTE = f"dotted: significant (Welch t-test, false discovery rate {FDR_Q:.0%})"  # the legend of the dots
FOOT = {
    "anom": "Normals: {} · ",  # the normals' label (natemp_ref.META), then DAY_NOTE
    "change": "Both runs over the same calendar days · ",
}


def log(*a):
    print(*a, flush=True)


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def full_days(init: pd.Timestamp) -> list[tuple[pd.Timestamp, list[int]]]:
    """[(day, steps of its four fields)] for the run's calendar UTC days whose four fields are all forecast steps."""
    out = []
    for day in pd.date_range(init.normalize(), (init + pd.Timedelta(hours=LAST_STEP)).normalize()):
        steps = [int((day + pd.Timedelta(hours=h) - init) / pd.Timedelta(hours=1)) for h in SAMPLES]
        if steps[0] >= 6 and steps[-1] <= LAST_STEP:
            out.append((day, steps))
    return out


def windows(ndays: int, anchor: int = 0) -> list[tuple[int, int]]:
    """(first, last) day indices of the PERIOD-day windows from `anchor`, the last one cut at the run's last day."""
    return [(k, min(k + PERIOD - 1, ndays - 1)) for k in range(anchor, ndays, PERIOD)][:3]


def fetch_step(date: str, hh: str, step: int) -> tuple[list[bytes], int]:
    """The 2t messages of the perturbed members at one step, and the bytes read (network only: runs in threads)."""
    import rangefetch as rf

    idx = rf.fetch_index(date, hh, "ifs", step, "ef", resol="0p1", levtype="sfc")
    want = sorted(rf.select(idx, param="2t", type="pf"), key=lambda e: e["_offset"])
    if not want:
        raise RuntimeError(f"step {step}: no 2t pf messages in the index")
    path = rf.path_for(date, hh, "ifs", step, "ef", resol="0p1", levtype="sfc") + ".grib2"
    stats: dict = {}
    blob = rf.fetch_ranges(path, rf.coalesce(want), workers=RANGE_WORKERS, stats=stats)
    lengths = [e["_length"] for e in want]  # the messages, back to back in offset order
    return [blob[e - n : e] for e, n in zip(np.cumsum(lengths), lengths)], sum(stats.values())


def decode(msg: bytes) -> tuple[int, np.ndarray]:
    """(member number, 2t on the box (K)) of one global 0.1-deg message, whatever its first longitude (the open data's
    starts at 180 E); eccodes runs here, in the calling thread only (decoding in the fetch threads has crashed the
    interpreter in this repository)."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        key = {k: eccodes.codes_get(gid, k) for k in ("shortName", "number", "Ni", "Nj", "jScansPositively")}
        lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
        v = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)
    if (key["shortName"], key["Ni"], key["Nj"], key["jScansPositively"], lat0) != GLOBAL_2T:
        raise RuntimeError(f"unexpected message {key} first point {lat0}, {lon0}")
    cols = (round((R.LON[0] - lon0) % 360 * 10) + np.arange(NX)) % 3600
    return int(key["number"]), v.reshape(1801, 3600)[R.ROW0 : R.ROW0 + NY][:, cols].astype(np.float32)


def split_grib(blob: bytes) -> list[bytes]:
    """The GRIB2 messages of a file's bytes, by their own length fields (no eccodes: this runs in the fetch threads)."""
    out, i = [], 0
    while (i := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[i + 8 : i + 16], "big")
        out.append(blob[i : i + n])
        i += n
    return out


def geps_file(date: str, hh: str, var: str, step: int) -> bytes:
    """One allmbrs file of a GEPS run from the MSC Datamart (network only: runs in threads)."""
    sys.path.insert(0, str(ROOT / "scripts" / "geps"))
    import datamart

    blob = datamart.get(datamart.url(date, var, step, hh), timeout=300)
    if blob is None:
        raise RuntimeError(f"GEPS {date} {hh}Z +{step} h: no {var} on the Datamart")
    return blob


def fetch_geps(date: str, hh: str, step: int) -> tuple[list[bytes], int]:
    """The members' 2t messages at one GEPS step, and the bytes read."""
    blob = geps_file(date, hh, GEPS_2T, step)
    return split_grib(blob), len(blob)


def decode_geps(msg: bytes) -> tuple[int, np.ndarray]:
    """(member number, the field on the 0.5-deg box) of one global GEPS message (the control is 0); eccodes in the
    calling thread only, as decode."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
        try:
            num = int(eccodes.codes_get(gid, "perturbationNumber"))
        except eccodes.KeyValueNotFoundError:
            num = 0
        v = eccodes.codes_get_values(gid).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    if (ni, nj) != (720, 361):
        raise RuntimeError(f"unexpected GEPS grid {ni} x {nj}")
    if lat0 < 0:  # south first: flip to 90N first
        v = v[::-1]
    rows = np.rint((90.0 - LAT_G) / 0.5).astype(int)
    cols = np.rint(((LON_G - lon0) % 360) / 0.5).astype(int) % 720
    return num, v[rows][:, cols].astype(np.float32)


class OnCells:
    """The 0.1-deg normals averaged over a coarser grid's cells and moved by LAPSE to that model's surface height
    (`shift`, K, from orography_shift)."""

    def __init__(self, N: R.Normals, lat: np.ndarray, lon: np.ndarray, shift: np.ndarray):
        self.N, self.lat, self.lon, self.shift, self.meta = N, lat, lon, shift, N.meta

    def at(self, times) -> np.ndarray:
        import cellmeans as CM

        res = float(self.lon[1] - self.lon[0])
        return CM.coarsen(self.N.at(times), R.LAT, R.LON, 0.1, self.lat, self.lon, res) + self.shift


def orography_shift(date: str, hh: str) -> np.ndarray:
    """(LAT_G, LON_G) K: -LAPSE x (GEPS's surface height - the IFS's orography averaged over the GEPS cell), the
    normals' orography being the IFS's (ERA5-Land runs on its ~9 km grid). GEPS: the control's HGT at step 0; the IFS:
    its control's surface geopotential at step 0 of the newest run on the Google Cloud mirror, 0.25 deg."""
    import cellmeans as CM
    import eccodes
    import rangefetch as rf

    zg = dict(decode_geps(m) for m in split_grib(geps_file(date, hh, GEPS_HGT, 0)))[0]
    # the Datamart's GEPS surface height is in DECAMETRES though its GRIB parameter says geopotential metres (24 Oct
    # 2026: 12 bits at a binary scale of -2, values -14..1010; the box's highest ground, Greenland and Alaska, reads
    # ~300-600): metres when the box's maximum says so, should the encoding be corrected
    if zg.max() < 1000.0:
        zg = zg * 10.0
    t0 = pd.Timestamp(f"{date} {hh}:00")
    for back in range(5):  # the orography does not change: the newest control run on the mirror (it lands after GEPS)
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
    gid = eccodes.codes_new_from_message(split_grib(blob)[0])
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
        inc = eccodes.codes_get_double(gid, "iDirectionIncrementInDegrees")
        up = eccodes.codes_get(gid, "jScansPositively")
        z = eccodes.codes_get_values(gid).reshape(nj, ni) / G
    finally:
        eccodes.codes_release(gid)
    lat = lat0 + (inc if up else -inc) * np.arange(nj)
    dz = zg - CM.coarsen(z, lat, lon0 + inc * np.arange(ni), inc, LAT_G, LON_G, 0.5, 360.0)
    log(f"  GEPS surface minus the IFS orography over its cells: mean |dz| {np.mean(np.abs(dz)):.0f} m, largest "
        f"{np.max(np.abs(dz)):.0f} m")  # fmt: skip
    return (-LAPSE * dz).astype(np.float32)


def sign_k(n: int) -> int:
    """The fewest of n members that must share a sign for a two-sided sign test against an even split to pass at 5 %."""
    from scipy.stats import binomtest

    return next(k for k in range(n // 2 + 1, n + 1) if binomtest(k, n).pvalue < 0.05)


@dataclass(frozen=True, eq=False)
class Model:
    """What tells the two ensembles apart: grid, members, data source and output names."""

    key: str
    label: str
    lat: np.ndarray
    lon: np.ndarray
    members: int  # the most members a run has
    first: int  # the GRIB number of the first (IFS-ENS pf 1..50, GEPS the control 0..20)
    min_members: int  # a run with fewer complete members is not drawn
    res: str
    source: str  # the summary's model line
    fetch: Callable
    decode: Callable
    prefix: str  # frame directories, manifest and summary

    def share(self, n: int) -> float:
        """The share of the n members that must give the anomaly its sign for it to be coloured."""
        return AGREE if self.key == "ifs" else sign_k(n) / n

    @property
    def regions(self) -> dict:
        return {"anom": (f"{self.prefix}_anom", "Anomaly"), "change": (f"{self.prefix}_change", "Change since the previous run")}  # fmt: skip


MODELS = {
    "ifs": Model("ifs", "IFS-ENS", R.LAT, R.LON, MEMBERS, 1, MIN_MEMBERS, "0.1°",
                 "IFS-ENS perturbed members, 0.1° open data (Google Cloud mirror)", fetch_step, decode, "natemp"),
    "geps": Model("geps", "GEPS", LAT_G, LON_G, GEPS_MEMBERS, 0, GEPS_MIN, "0.5°",
                  "GEPS control and 20 perturbed members, 0.5° (MSC Datamart)", fetch_geps, decode_geps, "natemp_geps"),
}  # fmt: skip


def day_normals(N: R.Normals, days) -> np.ndarray:
    """(day, NY, NX) float32 K: the normal curve averaged over each day's four forecast times, as the day's mean is."""
    return np.stack([N.at([d + pd.Timedelta(hours=h) for h in SAMPLES]).mean(0) for d in days]).astype(np.float32)


def read_run(date: str, hh: str, days: list, fetch=None, model: Model | None = None):
    """The members' daily-mean temperatures (member, day, lat, lon) K, which member-days hold all four fields
    (member, day), and the fetch's totals. Steps are fetched AHEAD of the decoding; each day is closed at its last
    field."""
    model = model or MODELS["ifs"]
    fetch = fetch or model.fetch
    n, ny, nx = model.members, model.lat.size, model.lon.size
    A = np.full((n, len(days), ny, nx), np.nan, np.float32)
    done = np.zeros((n, len(days)), bool)
    last = {st[-1]: k for k, (_, st) in enumerate(days)}
    acc = np.zeros((n, ny, nx), np.float32)
    cnt = np.zeros(n, int)
    todo = deque(s for _, st in days for s in st)
    nbytes = nmsg = 0
    t0 = time.time()
    with ThreadPoolExecutor(AHEAD) as ex:
        live: deque = deque()
        while todo or live:
            while todo and len(live) < AHEAD:
                s = todo.popleft()
                live.append((s, ex.submit(fetch, date, hh, s)))
            s, fut = live.popleft()
            msgs, nb = fut.result()
            nbytes += nb
            seen = set()
            for msg in msgs:
                num, f = model.decode(msg)
                i = num - model.first
                if 0 <= i < n and num not in seen:
                    seen.add(num)
                    acc[i] += f
                    cnt[i] += 1
            nmsg += len(msgs)
            if s in last:
                k = last[s]
                full = cnt == len(SAMPLES)
                A[full, k] = acc[full] / len(SAMPLES)
                done[:, k] = full
                acc[:] = 0.0
                cnt[:] = 0
                gb, sec = nbytes / 1e9, time.time() - t0
                log(f"  day {k + 1} {days[k][0]:%Y-%m-%d}: {int(full.sum())} members, {gb:.2f} GB, {sec:.0f} s")
    return A, done, {"bytes": nbytes, "messages": nmsg, "seconds": round(time.time() - t0)}


def member_mean(A: np.ndarray, mem: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """(member, NY, NX) float32: each of the members `mem` of A (member, day, NY, NX)'s mean over days k0..k1."""
    s = np.zeros((mem.size, *A.shape[2:]), np.float32)
    for k in range(k0, k1 + 1):
        s += A[mem, k]
    return s / (k1 - k0 + 1)


# ── statistics ────────────────────────────────────────────────────────────────────────────────────────────────────────
def agreed(x: np.ndarray, share: float = AGREE) -> tuple[np.ndarray, np.ndarray]:
    """(the members' mean, where at least `share` of the members share its sign) for members x (member, ...)."""
    m = x.mean(0)
    same = (np.sign(x) == np.sign(m)[None]).mean(0)
    return m, (same >= share - 1e-9) & (m != 0)


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


def significant(p: np.ndarray, family: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg at FDR_Q over the points of `family` (the map's points); False elsewhere."""
    from quantmet.stats import benjamini_hochberg

    out = np.zeros(p.shape, bool)
    out[family] = benjamini_hochberg(p[family], alpha=FDR_Q)
    return out


def key(days, k0: int, k1: int) -> str:
    return f"{days[k0][0]:%Y-%m-%d}/{days[k1][0]:%Y-%m-%d}"


def save_stats(path: Path, init: pd.Timestamp, days: list, A: np.ndarray, mem: np.ndarray) -> None:
    """Mean and standard deviation of the members' daily-mean temperatures (K) per day and per window from day 1 and
    day 2 - the windows the next run (12 h later) compares with: its days 1-5 start on this run's day 1 or 2."""
    spans = [(k, k) for k in range(len(days))] + sorted(set(windows(len(days), 0) + windows(len(days), 1)))
    keys, mean, sd = [], [], []
    for k0, k1 in spans:
        x = member_mean(A, mem, k0, k1)
        keys.append(key(days, k0, k1))
        mean.append(x.mean(0))
        sd.append(x.std(0, ddof=1).astype(np.float16))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, init=f"{init:%Y%m%d%H}", n=mem.size, keys=np.array(keys), mean=np.stack(mean), sd=np.stack(sd)
    )
    log(f"  statistics for the next run: {path} ({path.stat().st_size / 1e6:.0f} MB, {len(keys)} spans)")


def load_stats(path: Path | None, want_init: pd.Timestamp) -> dict | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as z:
        if str(z["init"]) != f"{want_init:%Y%m%d%H}":
            log(f"  {path}: holds run {z['init']}, not {want_init:%Y%m%d%H}; no change frames")
            return None
        return {
            "n": int(z["n"]),
            "keys": {str(k): i for i, k in enumerate(z["keys"])},
            "mean": z["mean"],
            "sd": z["sd"],
        }


# ── frames ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def span_text(days, k0: int, k1: int) -> tuple[str, str]:
    """(e.g. 'days 1–5', 'Fri Oct 23 – Tue Oct 27') or ('day 3', 'Sun Oct 25')."""
    d0, d1 = days[k0][0], days[k1][0]
    if k0 == k1:
        return f"day {k0 + 1}", f"{d0:%a %b %-d}"
    return f"days {k0 + 1}–{k1 + 1}", f"{d0:%a %b %-d} – {d1:%a %b %-d}"


def frame_label(days, k0: int, k1: int) -> str:
    what, when = span_text(days, k0, k1)
    return f"{what[0].upper()}{what[1:]} · {when}"


def shares(value: np.ndarray, gate: np.ndarray, inmap: np.ndarray) -> dict:
    """Over the map's points: the share coloured warm (positive), cold and neutral (the gate failed), and the coloured
    extremes (degF)."""
    v, g = value[inmap], gate[inmap]
    col = v[g]
    return {
        "warm": round(float((g & (v > 0)).mean()), 3),
        "cold": round(float((g & (v < 0)).mean()), 3),
        "neutral": round(float((~g).mean()), 3),
        "max_f": round(float(col.max()), 1) if col.size else None,
        "min_f": round(float(col.min()), 1) if col.size else None,
    }


def render(jobs: list, canvas, procs: int) -> None:
    """Draw the frames, in forked workers that share the canvas (natemp_maps.warm first reads the shapes here)."""
    global _CANVAS  # noqa: PLW0603  the workers inherit the canvas by fork
    import natemp_maps as M

    _CANVAS = canvas
    M.warm()
    if procs <= 1:
        for j in jobs:
            _draw(j)
        return
    with mp.get_context("fork").Pool(procs) as pool:
        pool.map(_draw, jobs, chunksize=1)


_CANVAS = None  # the layout, inherited by the forked render workers


def _draw(job: dict) -> None:
    import natemp_maps as M

    M.draw(_CANVAS, **job)


def spans(ndays: int) -> list[tuple[str, int, int]]:
    """(frame name, first day, last day) of the frames: the windows from day 1, then each day."""
    return [(f"P{i + 1}", k0, k1) for i, (k0, k1) in enumerate(windows(ndays))] + [
        (f"D{k + 1:02d}", k, k) for k in range(ndays)
    ]


def prev_row(days, k0: int, k1: int, prev: dict) -> tuple[int, int] | None:
    """(last day, row of the previous run's statistics) for the span k0..k1 cut to the days the previous run holds;
    None when it does not hold day k0 or has no statistics for the cut span."""
    last = max(k.split("/")[1] for k in prev["keys"])
    while k1 >= k0 and f"{days[k1][0]:%Y-%m-%d}" > last:
        k1 -= 1
    row = prev["keys"].get(key(days, k0, k1)) if k1 >= k0 else None
    return None if row is None else (k1, row)


def run(date: str, hh: str, normals: Path, prev_path: Path | None, save: Path | None, procs: int, fetch=None,
        model: str = "ifs"):  # fmt: skip
    import natemp_maps as M

    mod = MODELS[model]
    init = pd.Timestamp(f"{date} {hh}:00")
    prev_init = init - pd.Timedelta(hours=12)
    days = full_days(init)
    N = R.Normals(normals) if normals.exists() else None
    log(f"{mod.label} {init:%Y-%m-%d %H}Z: {len(days)} full days {days[0][0]:%Y-%m-%d}..{days[-1][0]:%Y-%m-%d}")
    if N is None:
        log(f"::notice::no normals ({normals}): change frames only")
    elif model == "geps":
        N = OnCells(N, mod.lat, mod.lon, orography_shift(date, hh))
    A, done, fetched = read_run(date, hh, days, fetch, mod)
    mem = np.flatnonzero(done.all(1))
    log(
        f"  {mem.size} of {mod.members} members complete; {fetched['bytes'] / 1e9:.2f} GB, {fetched['messages']} messages"
    )
    if mem.size < mod.min_members:
        log(f"::error::only {mem.size} complete members")
        return 1
    canvas = M.Canvas(mod.lat, mod.lon)
    normal = day_normals(N, [d for d, _ in days]) if N else None
    prev = load_stats(prev_path, prev_init)
    run_s = f"{init:%a %b %-d %Y %H}Z"
    label = N.meta.get("label", R.META["label"]) if N else ""
    share = mod.share(mem.size)
    k = int(np.ceil(share * mem.size - 1e-9))
    day_note = DAY_NOTE.format(mod.res)
    foot = {"anom": FOOT["anom"].format(label) + day_note, "change": FOOT["change"] + day_note}
    neutral = {"anom": f"members split (fewer than {k} of {mem.size} agree on the sign)", "change": CHANGE_NOTE}
    regions = mod.regions
    for rid, _label in regions.values():
        (ANIM / rid).mkdir(parents=True, exist_ok=True)
        for old in (ANIM / rid).glob("*.webp"):
            old.unlink()
    jobs, frames, summary = [], {"anom": [], "change": []}, {"anom": [], "change": []}

    def add(kind, name, k0, k1, value, gate, title, sub):
        out = ANIM / regions[kind][0] / f"{name}.webp"
        jobs.append(
            dict(
                kind=kind, value=value, gate=gate, title=title, sub=sub, neutral=neutral[kind], foot=foot[kind], out=out
            )
        )
        frames[kind].append({"file": out.name, "date": f"{days[k0][0]:%Y-%m-%d}", "label": frame_label(days, k0, k1)})
        row = {"frame": name, "first": f"{days[k0][0]:%Y-%m-%d}", "last": f"{days[k1][0]:%Y-%m-%d}"}
        summary[kind].append(row | shares(value, gate, canvas.inmap))

    for name, k0, k1 in spans(len(days)):
        x = member_mean(A, mem, k0, k1)
        if normal is not None:
            m, gate = agreed(x - normal[k0 : k1 + 1].mean(0), share)
            what, when = span_text(days, k0, k1)
            sub = (
                f"Run {run_s} · {mem.size} members · ensemble mean against the 1991–2020 normal, coloured where at "
                f"least {k} of the {mem.size} members agree on its sign · contours every 10 °F"
            )
            add("anom", name, k0, k1, m * KF, gate, f"{mod.label} 2 m temperature anomaly: {what}, {when}", sub)
        cut = prev_row(days, k0, k1, prev) if prev else None
        if cut is None:
            continue
        c1, row = cut
        y = x if c1 == k1 else member_mean(A, mem, k0, c1)
        my, sy = y.mean(0), y.std(0, ddof=1)
        m0, s0 = prev["mean"][row], prev["sd"][row].astype(np.float32)
        sig = significant(welch_p(my, sy, mem.size, m0, s0, prev["n"]), canvas.inmap)
        what, when = span_text(days, k0, c1)
        sub = (
            f"Run {run_s} minus {prev_init:%a %b %-d %H}Z · {mem.size} and {prev['n']} members · ensemble-mean change, "
            f"dotted where significant (Welch t-test, false discovery rate {FDR_Q:.0%}) · contours every 10 °F"
        )
        title = f"{mod.label} 2 m temperature, change from the previous run: {what}, {when}"
        add("change", name, k0, c1, (my - m0) * KF, sig, title, sub)
    if save:
        save_stats(save, init, days, A, mem)
    del A  # 3 GB the render workers need not inherit
    t0 = time.time()
    render(jobs, canvas, procs)
    size = sum(j["out"].stat().st_size for j in jobs)
    log(f"  {len(jobs)} frames ({size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    if not frames["change"]:
        log(f"::notice::no statistics of the previous run ({prev_init:%Y-%m-%d %H}Z): no change frames this run")
    write_outputs(mod, init, prev_init, mem.size, share, days, frames, summary, fetched, N.meta if N else None)
    return 0


def write_outputs(mod: Model, init, prev_init, n, share, days, frames, summary, fetched, normals_meta) -> None:
    regions = mod.regions
    doc = {
        "ver": f"{init:%Y%m%d%H}",
        "init": f"{init:%Y-%m-%dT%H}:00Z",
        "normals": normals_meta is not None,  # natemp.yml redraws a run drawn without them once they exist
        "default": regions["anom" if frames["anom"] else "change"][0],
        "selectorLabel": "View",
        "regions": {},
    }
    for kind, (rid, label) in regions.items():
        if frames[kind]:
            fr = [dict(f, idx=i) for i, f in enumerate(frames[kind])]
            doc["regions"][rid] = {"label": label, "n_frames": len(fr), "frames": fr}
    manifest, summary_path = ANIM / f"{mod.prefix}_manifest.json", DATA / f"{mod.prefix}.json"
    manifest.write_text(json.dumps(doc))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
                "init": doc["init"],
                "model": mod.source,
                "members": n,
                "days": [f"{d:%Y-%m-%d}" for d, _ in days],
                "agree": round(share, 4),
                "fdr_q": FDR_Q,
                "previous_run": f"{prev_init:%Y-%m-%dT%H}:00Z",
                "change_frames": bool(frames["change"]),
                "shares_over": "the map's grid points",
                "anom": summary["anom"],
                "change": summary["change"],
                "fetch": fetched,
                "normals": normals_meta or "not built yet (natemp-ref.yml): change frames only",
            },
            indent=1,
        )
    )
    log(f"  -> {manifest}, {summary_path}")


# ── the cycle ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def resolve_geps(wait_min: float) -> tuple[str, str]:
    """The GEPS run that is due (its start GEPS_DUE_H or more ago), waited for up to wait_min for its last step needed;
    if it does not arrive, the newest of the two runs before it that is complete."""
    sys.path.insert(0, str(ROOT / "scripts" / "geps"))
    import datamart

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    due = (now - dt.timedelta(hours=GEPS_DUE_H)).replace(minute=0, second=0, microsecond=0)
    due = due.replace(hour=12 if due.hour >= 12 else 0)

    def ready(t):
        return datamart.exists(datamart.url(f"{t:%Y%m%d}", GEPS_2T, LAST_STEP, f"{t:%H}"))

    t_end = time.time() + 60 * wait_min
    while not ready(due) and time.time() < t_end:
        time.sleep(120)
    for t in (due, due - dt.timedelta(hours=12), due - dt.timedelta(hours=24)):
        if ready(t):
            return f"{t:%Y%m%d}", f"{t:%H}"
    raise SystemExit(f"no GEPS run complete to +{LAST_STEP} h since {due - dt.timedelta(hours=24):%Y-%m-%d %H}Z")


def resolve(wait_min: float) -> tuple[str, str]:
    """The run that is due (its start DUE_H or more ago), waited for up to wait_min; if it does not complete, the newest
    run that is complete on the mirror (cycles.resolve)."""
    import cycles

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.timedelta(hours=DUE_H)
    due = now.replace(minute=0, second=0, microsecond=0, hour=12 if now.hour >= 12 else 0)
    d, h = f"{due:%Y%m%d}", f"{due:%H}"
    if cycles.wait_ready(d, h, cycles.IFS_ENS_0P1, wait_min):
        return d, h
    return cycles.resolve(cycles.IFS_ENS_0P1, what="IFS-ENS 0.1-deg 2t")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--model", choices=tuple(MODELS), default="ifs")
    r.add_argument("--wait-min", type=float, default=0.0)
    b = sub.add_parser("run")
    b.add_argument("--model", choices=tuple(MODELS), default="ifs")
    b.add_argument("--date", required=True)
    b.add_argument("--time", required=True, choices=("00", "12"))
    b.add_argument("--normals", type=Path, default=R.OUT)
    b.add_argument("--prev", type=Path, help="the previous run's statistics (save_stats)")
    b.add_argument("--save", type=Path, help="write this run's statistics here")
    b.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    if a.cmd == "resolve" and a.model == "geps":
        print(*resolve_geps(a.wait_min))
        return 0
    if a.cmd == "resolve":
        import cycles

        try:
            d, h = resolve(a.wait_min)
        except cycles.CycleUnavailable as e:
            log(f"ERROR: {e}")
            return 1
        print(d, h)
        return 0
    return run(a.date, a.time, a.normals, a.prev, a.save, a.procs, model=a.model)


if __name__ == "__main__":
    raise SystemExit(main())
