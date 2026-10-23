#!/usr/bin/env python3
"""North America 2 m temperature from the IFS-ENS on its 0.1-deg grid: the 50 perturbed members' daily means against
the 1991-2020 normal, and how the run moved them from the run 12 hours earlier.

Data: the 0.1-deg IFS-ENS surface files on the Google Cloud mirror (interpolated by ECMWF from the model's ~9 km grid;
the 0.1-deg files carry the perturbed members only), parameter 2t at the steps a full day needs, byte-ranged
(rangefetch: index -> the 50 messages -> coalesced ranges) and decoded in this thread; the grid's own points over the
box (natemp_ref.LAT, LON) are kept, nothing is regridded. A day is a calendar UTC day, the mean of 06, 12, 18 and 24 UTC:
a 00Z run has 15 full days, a 12Z run 14 (its first and last partial days are not read). Each member's daily mean less
the day's normal (natemp_ref.Normals: ERA5-Land, ERA5 over the sea) is its anomaly; until the normals exist (release
natemp-ref-v1) only the change frames are drawn, which need none: over the same days the normal cancels.

Frames, per view: days 1-5, 6-10 and 11-15 of the run (from its first full day; the last can be shorter, its label
says which days), then each day.
  anom    the members' mean anomaly, coloured only where at least AGREE of the members give it its sign
  change  this run's members' mean less the previous run's (12 h earlier) over the same calendar days (the days both
          runs hold), coloured only where significant: Welch's t-test on the two runs' members (their means and
          standard deviations, the previous run's from its statistics file), Benjamini-Hochberg over the map's points
          at FDR_Q. Without the previous run's statistics there are no change frames.
Writes the frames (assets/sst/anim/natemp_anom, natemp_change; frames branch), assets/sst/anim/natemp_manifest.json,
the summary assets/sst/data/natemp.json, and this run's statistics for the next run (--save: per day and per window,
the mean (float32) and standard deviation (float16) of the members' daily-mean temperatures).

    python scripts/natemp/natemp.py resolve --wait-min 45      # prints "YYYYMMDD HH"
    python scripts/natemp/natemp.py run --date 20261023 --time 00 --normals ref/natemp_normals.npz \
        --prev stats/2026102212.npz --save stats/2026102300.npz
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ecmwf")]
import natemp_ref as R

ANIM = ROOT / "assets" / "sst" / "anim"
MANIFEST = ANIM / "natemp_manifest.json"
SUMMARY = ROOT / "assets" / "sst" / "data" / "natemp.json"
REGIONS = {"anom": ("natemp_anom", "Anomaly"), "change": ("natemp_change", "Change since the previous run")}
MEMBERS = 50
MIN_MEMBERS = 40  # a run with fewer complete members is not drawn
AGREE = 0.8  # colour an anomaly only where this share of the members gives it its sign
FDR_Q = 0.10
PERIOD = 5  # days per window
SAMPLES = (6, 12, 18, 24)  # hours after a day's 00 UTC: the four fields of its mean
LAST_STEP = 360
DUE_H = 13.0  # a run's 0.1-deg ensemble is on the mirror ~13.3 h after its start
AHEAD = 2  # steps fetched ahead of the decoding
RANGE_WORKERS = 12  # ranged GETs per step (two steps in flight stay under the mirror's per-IP ceiling)
KF = 1.8  # K -> degF for anomalies and changes
GLOBAL_2T = ("2t", 3600, 1801, 0, 90.0, 0.0)  # shortName, Ni, Nj, jScansPositively, first lat, first lon
NY, NX = R.LAT.size, R.LON.size
DAY_NOTE = "a day is the mean of 06, 12, 18 and 24 UTC · 0.1° grid"
NEUTRAL = {"anom": f"members split (under {AGREE:.0%} agree on the sign)", "change": "no significant change"}
FOOT = {
    "anom": "Normals: ERA5-Land 1991–2020 over land, ERA5 over the sea, as a smooth daily curve · " + DAY_NOTE,
    "change": "Both runs over the same calendar days · " + DAY_NOTE,
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
    """(member number, 2t on the box (K)) of one message; eccodes runs here, in the calling thread only (decoding in
    the fetch threads has crashed the interpreter in this repository)."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        key = {k: eccodes.codes_get(gid, k) for k in ("shortName", "number", "Ni", "Nj", "jScansPositively")}
        lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
        v = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)
    if (key["shortName"], key["Ni"], key["Nj"], key["jScansPositively"], lat0, lon0) != GLOBAL_2T:
        raise RuntimeError(f"unexpected message {key} first point {lat0}, {lon0}")
    return int(key["number"]), v.reshape(1801, 3600)[R.ROW0 : R.ROW0 + NY, R.COL0 : R.COL0 + NX].astype(np.float32)


def day_normals(N: R.Normals, days) -> np.ndarray:
    """(day, NY, NX) float32 K: the normal curve averaged over each day's four forecast times, as the day's mean is."""
    return np.stack([N.at([d + pd.Timedelta(hours=h) for h in SAMPLES]).mean(0) for d in days]).astype(np.float32)


def read_run(date: str, hh: str, days: list, fetch=fetch_step):
    """The members' daily-mean temperatures (member, day, NY, NX) K, which member-days hold all four fields
    (member, day), and the fetch's totals. Steps are fetched AHEAD of the decoding; each day is closed at its last
    field."""
    A = np.full((MEMBERS, len(days), NY, NX), np.nan, np.float32)
    done = np.zeros((MEMBERS, len(days)), bool)
    last = {st[-1]: k for k, (_, st) in enumerate(days)}
    acc = np.zeros((MEMBERS, NY, NX), np.float32)
    cnt = np.zeros(MEMBERS, int)
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
                num, f = decode(msg)
                if 1 <= num <= MEMBERS and num not in seen:
                    seen.add(num)
                    acc[num - 1] += f
                    cnt[num - 1] += 1
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
def agreed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the members' mean, where at least AGREE of the members share its sign) for members x (member, ...)."""
    m = x.mean(0)
    same = (np.sign(x) == np.sign(m)[None]).mean(0)
    return m, (same >= AGREE) & (m != 0)


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


def run(date: str, hh: str, normals: Path, prev_path: Path | None, save: Path | None, procs: int, fetch=fetch_step):
    import natemp_maps as M

    init = pd.Timestamp(f"{date} {hh}:00")
    prev_init = init - pd.Timedelta(hours=12)
    days = full_days(init)
    N = R.Normals(normals) if normals.exists() else None
    log(f"IFS-ENS {init:%Y-%m-%d %H}Z: {len(days)} full days {days[0][0]:%Y-%m-%d}..{days[-1][0]:%Y-%m-%d}")
    if N is None:
        log(f"::notice::no normals ({normals}): change frames only")
    A, done, fetched = read_run(date, hh, days, fetch)
    mem = np.flatnonzero(done.all(1))
    log(f"  {mem.size} of {MEMBERS} members complete; {fetched['bytes'] / 1e9:.2f} GB, {fetched['messages']} messages")
    if mem.size < MIN_MEMBERS:
        log(f"::error::only {mem.size} complete members")
        return 1
    canvas = M.Canvas(R.LAT, R.LON)
    normal = day_normals(N, [d for d, _ in days]) if N else None
    prev = load_stats(prev_path, prev_init)
    run_s = f"{init:%a %b %-d %Y %H}Z"
    for rid, _label in REGIONS.values():
        (ANIM / rid).mkdir(parents=True, exist_ok=True)
        for old in (ANIM / rid).glob("*.webp"):
            old.unlink()
    jobs, frames, summary = [], {"anom": [], "change": []}, {"anom": [], "change": []}

    def add(kind, name, k0, k1, value, gate, title, sub):
        out = ANIM / REGIONS[kind][0] / f"{name}.webp"
        jobs.append(
            dict(
                kind=kind, value=value, gate=gate, title=title, sub=sub, neutral=NEUTRAL[kind], foot=FOOT[kind], out=out
            )
        )
        frames[kind].append({"file": out.name, "date": f"{days[k0][0]:%Y-%m-%d}", "label": frame_label(days, k0, k1)})
        row = {"frame": name, "first": f"{days[k0][0]:%Y-%m-%d}", "last": f"{days[k1][0]:%Y-%m-%d}"}
        summary[kind].append(row | shares(value, gate, canvas.inmap))

    for name, k0, k1 in spans(len(days)):
        x = member_mean(A, mem, k0, k1)
        if normal is not None:
            m, gate = agreed(x - normal[k0 : k1 + 1].mean(0))
            what, when = span_text(days, k0, k1)
            sub = (
                f"Run {run_s} · {mem.size} members · ensemble mean against the 1991–2020 normal, coloured where at "
                f"least {AGREE:.0%} of the members agree on its sign · contours every 10 °F"
            )
            add("anom", name, k0, k1, m * KF, gate, f"IFS-ENS 2 m temperature anomaly: {what}, {when}", sub)
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
            f"coloured where significant (Welch t-test, false discovery rate {FDR_Q:.0%}) · contours every 10 °F"
        )
        title = f"IFS-ENS 2 m temperature, change from the previous run: {what}, {when}"
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
    write_outputs(init, prev_init, mem.size, days, frames, summary, fetched, N.meta if N else None)
    return 0


def write_outputs(init, prev_init, n, days, frames, summary, fetched, normals_meta) -> None:
    doc = {
        "ver": f"{init:%Y%m%d%H}",
        "init": f"{init:%Y-%m-%dT%H}:00Z",
        "normals": normals_meta is not None,  # natemp.yml redraws a run drawn without them once they exist
        "default": REGIONS["anom" if frames["anom"] else "change"][0],
        "selectorLabel": "View",
        "regions": {},
    }
    for kind, (rid, label) in REGIONS.items():
        if frames[kind]:
            fr = [dict(f, idx=i) for i, f in enumerate(frames[kind])]
            doc["regions"][rid] = {"label": label, "n_frames": len(fr), "frames": fr}
    MANIFEST.write_text(json.dumps(doc))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
                "init": doc["init"],
                "model": "IFS-ENS perturbed members, 0.1° open data (Google Cloud mirror)",
                "members": n,
                "days": [f"{d:%Y-%m-%d}" for d, _ in days],
                "agree": AGREE,
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
    log(f"  -> {MANIFEST}, {SUMMARY}")


# ── the cycle ─────────────────────────────────────────────────────────────────────────────────────────────────────────
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
    r.add_argument("--wait-min", type=float, default=0.0)
    b = sub.add_parser("run")
    b.add_argument("--date", required=True)
    b.add_argument("--time", required=True, choices=("00", "12"))
    b.add_argument("--normals", type=Path, default=R.OUT)
    b.add_argument("--prev", type=Path, help="the previous run's statistics (save_stats)")
    b.add_argument("--save", type=Path, help="write this run's statistics here")
    b.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    if a.cmd == "resolve":
        import cycles

        try:
            d, h = resolve(a.wait_min)
        except cycles.CycleUnavailable as e:
            log(f"ERROR: {e}")
            return 1
        print(d, h)
        return 0
    return run(a.date, a.time, a.normals, a.prev, a.save, a.procs)


if __name__ == "__main__":
    raise SystemExit(main())
