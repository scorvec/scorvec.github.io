#!/usr/bin/env python3
"""Tropical-cyclone vertical wind shear and environment, per active storm, along the IFS control (ECMWF's
unperturbed run, HRES until May 2026: the lead run, and the 330 K maps), the 50 AIFS-ENS perturbed members (each on its own track), AIFS single, GFS and, near the US, HRRR. Physics in
shearlib.py, data in sheardata.py, the synoptic maps in shearmaps.py, observations (TDR tilt, vortex-message fixes,
best track, SHIPS) in shearobs.py.

    python scripts/tcshear/tcshear.py                        # newest ready cycle (00/06/12/18Z)
    python scripts/tcshear/tcshear.py --date 20261007 --time 12 --members 4 --no-maps    # a quick dev run

Writes (repo-relative):
  assets/sst/anim/tcshear/<SID>.json        per storm: every model / member record along its track + observations
  assets/sst/anim/tcshear/<SID>_F<hhh>.webp  330 K PV + shear map per 6 h (IFS control)    -> frames branch
  assets/sst/anim/tcshear_manifest.json     the storm list (main; stamped with frames_sha by the publisher)
  assets/sst/data/tcshear_verify.json       running verification archive (analysis shear vs SHIPS, tilt vs TDR)
Exits in seconds with an empty storm list when no named storm exists. A cycle is built once AIFS-ENS and the IFS control
are both on the Google mirror to day 5 (resolve_cycle). A run that was not fully read (AIFS single not on the mirror yet,
or a failed fetch) is published with complete=false, pending=[model] and rebuilt by a later dispatch once the model is
there (closed without it at T+DET_WAIT_H).
Map grid: the maps are drawn from the IFS control's 0.25-degree run with the cycle; its 0.1-degree run reaches
the mirror ~2 h later, and the next dispatch redraws only the maps from it (maps_only; --map-resol auto), on the same
boxes and steps. The diagnostics, the hodographs and the verification archive stay on the 0.25-degree run. The
manifest's maps_resol says which grid the maps are on.
"""

from __future__ import annotations

import pyproj  # noqa: F401  side effect: load first, before eccodes/xarray load eckit's own libproj (double free at exit)

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "scripts" / "ecmwf"),
    str(ROOT / "scripts" / "mjo" / "src"),
    str(ROOT / "scripts" / "models"),
]
import sheardata as D
import shearlib as S

OUT_DIR = ROOT / "assets" / "sst" / "anim" / "tcshear"
MANIFEST = ROOT / "assets" / "sst" / "anim" / "tcshear_manifest.json"
VERIFY = ROOT / "assets" / "sst" / "data" / "tcshear_verify.json"
CACHE = ROOT / "scripts" / "ecmwf" / "cache" / "tcshear"
DET_STEPS = tuple(range(0, 121, 6))
MEM_STEPS = tuple(range(0, 121, 12))
HRRR_STEPS = tuple(range(0, 49, 6))
MODELS = {"aifs-ens": "AIFS-ENS", "aifs": "AIFS single", "ifs": "IFS control", "gfs": "GFS", "hrrr": "HRRR"}
LEAD = "ifs"  # the lead run: the bold one on the page, the maps, the storm list's summary
RUNS = ("ifs", "aifs", "gfs", "hrrr")  # the single runs, lead first (the page's order)
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"


def log(*a):
    print(*a, flush=True)


# ── cycles ────────────────────────────────────────────────────────────────────────────────────────────────────────
def age_h(init):
    """Hours since the cycle's initial time (now, UTC)."""
    return (dt.datetime.now(dt.UTC).replace(tzinfo=None) - init).total_seconds() / 3600.0


def resolve_cycle(date=None, hh=None, back=4):
    """Newest cycle (00/06/12/18Z) with the AIFS-ENS tracker file and perturbed members to 120 h on the Google mirror,
    and the lead run (the IFS control) there too (det_on_mirror). The lead lands after AIFS-ENS (06/18Z ~T+6.4 h, 00/12Z
    ~T+7.5 h); a cycle still without it at T+DET_WAIT_H is taken without it."""
    import cycles

    need = (cycles.Need("aifs-ens", "pf", step=120, every=tuple(range(0, 121, 6))),)
    if date and hh:
        return date, f"{int(hh):02d}"
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    c = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour // 6) * 6)
    for _ in range(back + 1):
        d, h = f"{c:%Y%m%d}", f"{c:%H}"
        ok, why = cycles.is_ready(d, h, need)
        tf = f"{GOOGLE}/{d}/{h}z/aifs-ens/0p25/enfo/{d}{h}0000-360h-enfo-tf.bufr"
        if ok and D.http_get(tf, rng=(0, 99), tries=2, src="probe") is None:
            ok, why = False, "AIFS-ENS tracker file missing"
        if ok and not det_on_mirror(LEAD, d, h):
            if age_h(c) < DET_WAIT_H:
                ok, why = False, f"{MODELS[LEAD]} not fully on the mirror yet"
            else:
                why += f"; {MODELS[LEAD]} still missing at T+{age_h(c):.1f} h, taken without it"
        if ok:
            log(f"cycle {d} {h}Z: {why}")
            return d, h
        log(f"  {d} {h}Z not ready: {why}")
        c -= dt.timedelta(hours=6)
    raise SystemExit("no cycle ready on the Google mirror")


def oper_tf(model, d, h):
    """The deterministic run's tracker file on the Google mirror (name carries the run length). None when it is not
    there or cannot be fetched (a flaky fetch degrades to 'this model missing', never fails the cycle)."""
    import cycles

    names = cycles._list_names(f"{d}/{h}z/{model}/0p25/oper/") or []
    tf = [n for n in names if n.endswith("-oper-tf.bufr")]
    if not tf:
        return None
    fp = CACHE / f"{model}_{d}{h}_tf.bufr"
    if not fp.exists():
        try:
            b = D.http_get(f"{GOOGLE}/{tf[0]}", src="ecmwf-google")
        except Exception as e:  # noqa: BLE001
            log(f"::warning::{model} tracker file {d} {h}Z: {e}")
            return None
        if b is None:
            return None
        fp.write_bytes(b)
    return fp


def oper_steps(model, d, h, steps, resol="0p25"):
    """Steps of a deterministic ECMWF run fully on the mirror (.index and .grib2; at 0.1 deg the pressure-level file, all
    the maps read). The IFS control runs to 144 h at 06/18Z and 360 h at 00/12Z, AIFS single to 360 h: every step to
    120 h exists once the run is uploaded. A listing failure returns no steps (counted as missing, so the cycle is
    retried)."""
    import cycles
    import rangefetch as rf

    names = set(cycles._list_names(f"{d}/{h}z/{model}/{resol}/oper/") or [])
    rel = {s: rf.path_for(d, h, model, s, "fc", stream="oper", resol=resol, levtype="pl") for s in steps}
    return [s for s in steps if rel[s] + ".grib2" in names and rel[s] + ".index" in names]


# The deterministic ECMWF runs: (model on the mirror, key in the product). The IFS control, the lead, reaches the Google mirror
# after AIFS-ENS (06/18Z ~T+6.4 h, 00/12Z to 360 h ~T+7.5 h, 12Z ~19:34Z), so resolve_cycle waits for it. A cycle built
# without a run fully read is published with complete=false and pending=[model]; a later dispatch rebuilds it once the
# model is on the mirror. Past T+DET_WAIT_H the cycle is closed without it.
DET = (("ifs", "ifs"), ("aifs-single", "aifs"))
DET_WAIT_H = 9.0


def close_cycle(man, det_full, init):
    """complete / pending / missing in the manifest from which deterministic runs were read in full. A run not yet (fully)
    on the mirror keeps the cycle open (complete=false, pending=[model]) for a later dispatch, up to T+DET_WAIT_H."""
    missing = [m for m, ok in det_full.items() if not ok]
    man.pop("pending", None)
    man.pop("missing", None)
    if missing and age_h(init) < DET_WAIT_H:
        man["complete"] = False
        man["pending"] = missing
        log(
            f"  published without {', '.join(missing)} (not fully on the Google mirror yet): the cycle stays open "
            f"until T+{DET_WAIT_H:.0f} h"
        )
    else:
        man["complete"] = True
        if missing:
            man["missing"] = missing
            log(f"::warning::{init:%Y%m%d %H}Z published without {', '.join(missing)}")
    return man


def det_on_mirror(model, d, h):
    """True when the run's tracker file and every step of DET_STEPS are on the Google mirror."""
    import cycles

    names = cycles._list_names(f"{d}/{h}z/{model}/0p25/oper/")
    if not names or not any(n.endswith("-oper-tf.bufr") for n in names):
        return False
    return len(oper_steps(model, d, h, DET_STEPS)) == len(DET_STEPS)


# ── tracks ────────────────────────────────────────────────────────────────────────────────────────────────────────
def track_at(tr, h):
    st = np.asarray(tr["steps"])
    if h in st:
        k = int(np.where(st == h)[0][0])
        return float(tr["lat"][k]), float(tr["lon"][k])
    return None


def storms_from(msgs, min_tracks=10):
    """Named storms present at t=0 in at least min_tracks of the tracks (members, or 1 for a single run). -> list of
    dicts."""
    out = []
    for m in msgs:
        if not m["named"]:
            continue
        if sum(0 in list(t["steps"]) for t in m["tracks"].values()) < min_tracks:
            continue
        sid = m["id"].strip().upper()
        out.append(dict(tid=sid, name=m["name"].strip().title(), tracks=m["tracks"]))
    return out


def motion(track, h, half=6):
    """Storm motion (speed m/s, heading deg toward) from the centred difference of the track positions."""
    a, b = track_at(track, h - half), track_at(track, h + half)
    span = 2 * half
    if a is None:
        a, span = track_at(track, h), half
    if b is None:
        b, span = track_at(track, h), (half if span == 2 * half else 0)
    if a is None or b is None or span == 0:
        return None
    x, y = S.offset_km(a[0], a[1], b[0], b[1])
    spd = math.hypot(float(x), float(y)) * 1e3 / (span * 3600)
    return [round(spd, 1), round(S.bearing_of(float(x), float(y)))]


# ── one model's pass ──────────────────────────────────────────────────────────────────────────────────────────────
def analyse_box(b, la, lo, oi, want_pv=True):
    if b is None or not D.complete(b):
        return None
    return S.analyze(b, la, lo, D.sst_at(oi, la, lo), want_pv=want_pv)


def run_ecmwf_ens(d, h, storms, oi, members):
    """The AIFS-ENS perturbed members 1..members (12-hourly), each on its own tracker positions (self-tracked from the
    seed when the cycle has no ensemble tracker file). -> {sid: {member: [record per step]}}"""
    res = {s["sid"]: {} for s in storms}
    for step in MEM_STEPS:
        t0 = time.time()
        cbm = {}
        for s in storms:
            for m in range(1, members + 1) if s.get("seed") else s["tracks"]:
                if m > members:
                    continue
                tr = s["tracks"].get(m)
                p = track_at(tr, step) if tr else None
                if p:
                    cbm.setdefault(m, []).append(((m, s["sid"]), p[0], p[1]))
                elif s.get("seed"):  # no ensemble tracker file (hindcasts): self-track
                    found = s.setdefault("found", {}).setdefault(m, [])
                    g = track_at(s["seed"], 0) if step == 0 else (_extrap(found, step) if found else None)
                    if g:
                        cbm.setdefault(m, []).append(((m, s["sid"], "g"), g[0], g[1]))
        if not cbm:
            continue
        boxes = D.ecmwf_boxes("aifs-ens", d, h, step, "pf", "enfo", cbm)
        if boxes is None:
            log(f"  aifs-ens pf +{step}h not on the mirror")
            continue
        for m, lst in cbm.items():
            for key, la_trk, lo_trk in lst:
                la, lo = la_trk, lo_trk  # the tracker position, replaced by the centre found when re-centred
                if len(key) == 3:
                    b = boxes.get(key)
                    if b is None or not D.complete(b):
                        continue
                    c = S.find_centre(b, la, lo, r_search_km=250.0 if step == 0 else 400.0)
                    if c is None:
                        continue
                    s_ = next(x for x in storms if x["sid"] == key[1])
                    s_["found"][m].append((step, c[0], c[1]))
                    la, lo = c[0], c[1]
                r = analyse_box(boxes.get(key), la, lo, oi, want_pv=True)
                if r:
                    r["h"] = step
                    res[key[1]].setdefault(m, []).append(r)
        log(f"  aifs-ens +{step:3d}h {time.time() - t0:5.1f} s  {D.LEDGER.report()}")
    return res


def det_tracks(model, d, h, storms):
    """{sid: track} of a deterministic run's own tracker for the listed storms (matched by tracker id); None when the run
    has no readable tracker file for this cycle."""
    import tclib as TC

    fpt = oper_tf(model, d, h)
    if fpt is None:
        log(f"  {model}: no tracker file for {d} {h}Z")
        return None
    tr = {}
    try:
        for m in storms_from(TC.decode_all(fpt, types=(0,)), min_tracks=1):
            for s in storms:
                if m["tid"] == s["tid"] and 0 in m["tracks"]:
                    tr[s["sid"]] = m["tracks"][0]
    except Exception as e:  # noqa: BLE001
        log(f"::warning::{model} tracker file {d} {h}Z unreadable: {e}")
        fpt.unlink(missing_ok=True)
        return None
    return tr


def run_ecmwf_det(model, d, h, storms, tracks, oi, maps=False):
    """AIFS single ('aifs-single') or the IFS control ('ifs') along the run's own tracker positions; maps: also each storm's
    map box (storm["mapbox"]) at every step where the run tracks it.
    -> ({sid: records}, full: every step of DET_STEPS was on the mirror and read without error, {sid: {step: box}})."""
    res, mapbox = {}, {}
    steps = oper_steps(model, d, h, DET_STEPS)
    full = len(steps) == len(DET_STEPS)
    for step in steps:
        cbm, extra = {0: []}, {}
        for s in storms:
            tr = tracks.get(s["sid"])
            p = track_at(tr, step) if tr else None
            if p:
                cbm[0].append(((0, s["sid"]), p[0], p[1]))
                if maps and s.get("mapbox"):
                    extra[("map", s["sid"])] = s["mapbox"]
        if not cbm[0]:
            continue
        try:
            boxes = D.ecmwf_boxes(model, d, h, step, "fc", "oper", cbm, extra=extra or None)
        except Exception as e:  # noqa: BLE001
            log(f"::warning::{model} +{step}h: {e}")
            full = False
            continue
        if boxes is None:
            full = False
            continue
        for (_, sid), la, lo in cbm[0]:
            r = analyse_box(boxes.get((0, sid)), la, lo, oi)
            if r:
                r["h"] = step
                res.setdefault(sid, []).append(r)
        for key in extra:
            if key in boxes:
                mapbox.setdefault(key[1], {})[step] = boxes[key]
    log(f"  {model}: {len(steps)} of {len(DET_STEPS)} steps on the mirror  {D.LEDGER.report()}")
    return res, full, mapbox


def _extrap(found, step):
    if len(found) >= 2:
        (h1, a1, o1), (h2, a2, o2) = found[-2], found[-1]
        f = (step - h2) / max(1, h2 - h1)
        return a2 + f * (a2 - a1), o2 + f * (((o2 - o1 + 180) % 360) - 180)
    return found[-1][1], found[-1][2]


def guess(s, step):
    """First guess for a self-tracked model: the median of the AIFS-ENS members' positions at that step."""
    pts = [track_at(t, step) for t in s["tracks"].values()]
    pts = [p for p in pts if p]
    if len(pts) < 5:
        pts = [(a, o) for f in s.get("found", {}).values() for (h_, a, o) in f if h_ == step]
    if len(pts) < 5:
        return track_at(s["seed"], step) if s.get("seed") else None
    return float(np.median([p[0] for p in pts])), float(
        np.median([((p[1] - pts[0][1] + 180) % 360) - 180 + pts[0][1] for p in pts])
    )


def self_track(prev, step, s, b, first=None):
    """Next centre of a self-tracked run: search 350 km around the persistence-extrapolated position (or around the
    observed / first-guess position at t=0)."""
    if first is not None:
        return S.find_centre(b, first[0], first[1], r_search_km=250.0)
    if not prev:
        return None
    if len(prev) >= 2:
        (h1, a1, o1), (h2, a2, o2) = prev[-2], prev[-1]
        f = (step - h2) / max(1, h2 - h1)
        g = (a2 + f * (a2 - a1), o2 + f * (((o2 - o1 + 180) % 360) - 180))
    else:
        g = prev[-1][1:]
    return S.find_centre(b, g[0], g[1], r_search_km=350.0)


def run_gfs(d, h, storms, oi, obs0):
    res = {}
    trk = {s["sid"]: [] for s in storms}
    for step in DET_STEPS:
        cent = []
        for s in storms:
            g = guess(s, step)
            if g is None and trk[s["sid"]]:
                g = trk[s["sid"]][-1][1:]
            if g is None:
                continue
            cent.append(((0, s["sid"]), g[0], g[1]))
        if not cent:
            break
        try:
            boxes = D.gfs_boxes(d, h, step, cent, half_lat=18.0)
        except Exception as e:  # noqa: BLE001
            log(f"  gfs +{step}h failed: {e}")
            continue
        if boxes is None:
            log(f"  gfs +{step}h not posted")
            break
        for (_, sid), _, _ in cent:
            b = boxes.get((0, sid))
            if b is None or not D.complete(b):
                continue
            s = next(x for x in storms if x["sid"] == sid)
            first = (obs0.get(sid) or guess(s, 0)) if step == 0 else None
            if step > 0 and not trk[sid]:
                continue
            c = self_track(trk[sid], step, s, b, first=first)
            if c is None:
                continue
            trk[sid].append((step, c[0], c[1]))
            r = S.analyze(b, c[0], c[1], D.sst_at(oi, c[0], c[1]))
            r["h"] = step
            res.setdefault(sid, []).append(r)
    log(f"  gfs done  {D.LEDGER.report()}")
    return res, {
        sid: dict(steps=[x[0] for x in v], lat=np.array([x[1] for x in v]), lon=np.array([x[2] for x in v]))
        for sid, v in trk.items()
        if v
    }


def run_hrrr(d, h, storms, oi, seeds):
    """HRRR (CONUS 3 km, coarsened to 12 km) only where the whole 0-500 km disc is inside its domain."""
    res = {}
    trk = {s["sid"]: [] for s in storms}
    for step in HRRR_STEPS:
        live = []
        for s in storms:
            g = track_at(seeds.get(s["sid"]), step) if seeds.get(s["sid"]) else guess(s, step)
            if g and 18.0 <= g[0] <= 52.0 and -135.0 <= g[1] <= -60.0:
                live.append((s, g))
        if not live:
            continue
        try:
            F = D.hrrr_fields(d, h, step)
        except Exception as e:  # noqa: BLE001
            log(f"  hrrr +{step}h failed: {e}")
            continue
        if F is None:
            log(f"  hrrr +{step}h not posted")
            break
        for s, g in live:
            b = D.hrrr_box(F, g[0], g[1], half_lat=18.0)
            c = self_track(trk[s["sid"]], step, s, b, first=g if not trk[s["sid"]] else None)
            if c is None:
                continue
            b = D.hrrr_box(F, c[0], c[1])
            r = S.analyze(b, c[0], c[1], D.sst_at(oi, c[0], c[1]))
            if "env" not in r:
                continue  # the disc is not inside the HRRR domain
            trk[s["sid"]].append((step, c[0], c[1]))
            r["h"] = step
            res.setdefault(s["sid"], []).append(r)
    log(f"  hrrr done  {D.LEDGER.report()}")
    return res


# ── assembly ──────────────────────────────────────────────────────────────────────────────────────────────────────
def add_motion(recs):
    """Storm motion per record from the run's own centres, and the shear direction relative to motion."""
    tr = dict(steps=[r["h"] for r in recs], lat=[r["lat"] for r in recs], lon=[r["lon"] for r in recs])
    for r in recs:
        mv = motion(tr, r["h"], half=6 if (r["h"] % 12 or len(recs) > 15) else 12)
        if mv is None:
            mv = motion(tr, r["h"], half=12)
        if mv:
            r["mot"] = mv
            if "deep" in r:
                # heading of the shear vector (toward) relative to the motion heading: 0 = shear along motion,
                # +90 = shear points to the LEFT of motion
                sh_to = (r["deep"][1] + 180.0) % 360.0
                r["sh_mot"] = round(((mv[1] - sh_to) + 180.0) % 360.0 - 180.0)
    return recs


def summary(per_member):
    """Percentiles across members per step for the fan: {field: {h: [p10, p25, p50, p75, p90, n]}}."""
    out = {}
    fields = {
        "deep": lambda r: r.get("deep", [None])[0],
        "gs": lambda r: r.get("gs"),
        "vi": lambda r: r.get("vi"),
        "vi_env": lambda r: r.get("vi_env"),
        "tilt": lambda r: (r.get("tilt") or [None])[0] if r.get("sst") is not None else None,  # over water only
        "pi": lambda r: r.get("pi"),
        "chi": lambda r: r.get("chi"),
        "low": lambda r: r.get("low", [None])[0],
        "up": lambda r: r.get("up", [None])[0],
        "d2pvu": lambda r: r.get("d2pvu"),
        "rh600": lambda r: r.get("rh600"),
    }
    for f, get in fields.items():
        by = {}
        for recs in per_member.values():
            for r in recs:
                v = get(r)
                if v is not None:
                    by.setdefault(r["h"], []).append(v)
        out[f] = {
            str(hh): [round(float(np.percentile(v, q)), 3) for q in (10, 25, 50, 75, 90)] + [len(v)]
            for hh, v in sorted(by.items())
            if len(v) >= 10
        }
    return out


def mapbox_for(tr):
    """Fixed map region for a storm: the lead run's track over its first 72 h, with room for the upstream trough."""
    la, lo = np.asarray(tr["lat"], float), np.asarray(tr["lon"], float)
    st = np.asarray(tr["steps"])
    sel = st <= 72
    la, lo = la[sel], lo[sel]
    lo = lo[0] + ((lo - lo[0] + 180) % 360 - 180)
    if np.nanmean(la) >= 0:  # 13 deg poleward (the trough), 9 equatorward
        a0, a1 = max(-70.0, la.min() - 9.0), min(70.0, la.max() + 13.0)
        if a1 - a0 < 26:
            a1 = a0 + 26
    else:  # Southern Hemisphere: poleward is south
        a0, a1 = max(-70.0, la.min() - 13.0), min(70.0, la.max() + 9.0)
        if a1 - a0 < 26:
            a0 = a1 - 26
    b0, b1 = lo.min() - 24.0, lo.max() + 14.0
    if b1 - b0 > 60:
        b0 = b1 - 60
    return (round(a0), round(a1), round(b0) % 360.0, round(b0) % 360.0 + round(b1 - b0))


# ── the maps' second stage: the lead run's 0.1-degree fields ───────────────────────────────────────────────────────
def lead_map_resol(d, h):
    """'0p1' once the lead run's 0.1-degree pressure levels are on the mirror for every step of DET_STEPS its
    0.25-degree run has (~2 h after it), else '0p25'."""
    fine = oper_steps(LEAD, d, h, DET_STEPS, "0p1")
    return "0p1" if fine and fine == oper_steps(LEAD, d, h, DET_STEPS) else "0p25"


def redraw_maps(man, d, h, init, resol) -> bool:
    """Redraw every storm's maps of the cycle in OUT_DIR from the lead run's `resol` fields, at the steps and on the map
    boxes its 0.25-degree maps used (the per-storm JSON keeps the box, the lead run's records and the observations); the
    diagnostics stay the 0.25-degree run's. Nothing changes unless every map box of every step is read."""
    import shearmaps as M
    from rangefetch import RESOL_LABEL

    docs = {st["sid"]: json.loads((OUT_DIR / f"{st['sid']}.json").read_text()) for st in man["storms"]}
    docs = {sid: doc for sid, doc in docs.items() if doc.get("frames")}
    if any(not doc.get("mapbox") for doc in docs.values()):
        log(f"::warning::{d} {h}Z: a storm's map box was not kept; its {RESOL_LABEL[resol]} maps cannot be drawn")
        return False
    steps = sorted({int(f.rsplit("_F", 1)[1][:3]) for doc in docs.values() for f in doc["frames"]})
    boxes = {}
    for step in steps:
        extra = {
            ("map", sid): tuple(doc["mapbox"])
            for sid, doc in docs.items()
            if any(f.endswith(f"_F{step:03d}.webp") for f in doc["frames"])
        }
        try:
            b = D.ecmwf_boxes(LEAD, d, h, step, "fc", "oper", {0: []}, extra=extra, resol=resol, params=("u", "v", "t"))
        except Exception as e:  # noqa: BLE001
            b = None
            log(f"::warning::{LEAD} {resol} +{step}h: {e}")
        if b is None or any(k not in b for k in extra):
            log(f"::warning::{d} {h}Z: the {RESOL_LABEL[resol]} map boxes at +{step} h not read; maps unchanged")
            return False
        for (_, sid), box in b.items():
            boxes.setdefault(sid, {})[step] = box
    for st in man["storms"]:
        doc = docs.get(st["sid"])
        if doc is None:
            continue
        storm = dict(sid=doc["sid"], name=doc["name"], mapbox=doc["mapbox"])
        recs = doc["models"].get(LEAD, {}).get("run", [])
        doc["frames"] = M.render(storm, boxes[doc["sid"]], recs, init, OUT_DIR, doc["obs"], RESOL_LABEL[resol])
        (OUT_DIR / f"{doc['sid']}.json").write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
        st["frames"] = len(doc["frames"])
    man["maps_resol"] = resol
    log(f"  maps redrawn at {RESOL_LABEL[resol]} for {len(docs)} storm(s), {len(steps)} steps  {D.LEDGER.report()}")
    return True


def maps_only(man, d, h, init, resol) -> int:
    """The cycle is published with 0.25-degree maps and the lead run's `resol` run is now out: fetch the published storm
    files back from the frames branch (the publisher replaces the directory whole) and redraw only the maps. On any
    failure the directory is emptied, so nothing is published and the 0.25-degree maps stay."""
    import shutil
    import subprocess

    seed = ROOT / "scripts" / "lib" / "fetch_frames_ci.sh"
    subprocess.run(["bash", str(seed), str(OUT_DIR.relative_to(ROOT))], cwd=ROOT, check=False)  # checked below
    ok = all((OUT_DIR / f"{st['sid']}.json").exists() for st in man["storms"])
    if not ok:
        log(f"::warning::{d} {h}Z: the published storm files could not be fetched back; the 0.25-degree maps stay")
    if not ok or not redraw_maps(man, d, h, init, resol):
        shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True)
        return 0
    MANIFEST.write_text(json.dumps(man, separators=(",", ":")))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--no-hrrr", action="store_true")
    ap.add_argument("--only", default="", help="comma list of storm ids (dev)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-h", type=int, default=120, help="dev: last forecast hour")
    ap.add_argument("--tag", default="", help="hc: a hindcast, written to tcshear_hc (never over the live product)")
    ap.add_argument(
        "--map-resol",
        choices=("auto", "0p25", "0p1"),
        default="auto",
        help="the maps' grid (auto: 0.1 deg once the lead run's 0.1-degree run is on the mirror)",
    )
    a = ap.parse_args()
    global DET_STEPS, MEM_STEPS, HRRR_STEPS, OUT_DIR, MANIFEST  # noqa: PLW0603  run settings set once from the command line before any use
    if a.tag:
        OUT_DIR = OUT_DIR.parent / f"tcshear_{a.tag}"
        MANIFEST = MANIFEST.parent / f"tcshear_{a.tag}_manifest.json"
    DET_STEPS = tuple(x for x in DET_STEPS if x <= a.max_h)
    MEM_STEPS = tuple(x for x in MEM_STEPS if x <= a.max_h)
    HRRR_STEPS = tuple(x for x in HRRR_STEPS if x <= a.max_h)
    T0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d, h = resolve_cycle(a.date, a.time)
    init = dt.datetime.strptime(d + h, "%Y%m%d%H")
    prev = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    map_resol = lead_map_resol(d, h) if a.map_resol == "auto" else a.map_resol
    if prev.get("init") == f"{init:%Y-%m-%dT%H:00Z}" and not a.force:
        if prev.get("complete"):
            maps = any(st.get("frames") for st in prev.get("storms", []))
            if maps and map_resol != "0p25" and prev.get("maps_resol", "0p25") != map_resol:
                log(f"{d} {h}Z is published with 0.25-degree maps; redrawing them from the {map_resol} run")
                return maps_only(prev, d, h, init, map_resol)
            log(f"{d} {h}Z already published; nothing to do")
            return 0
        pend = prev.get("pending") or []
        if pend and not any(det_on_mirror(m, d, h) for m in pend):
            if age_h(init) < DET_WAIT_H:
                log(f"{d} {h}Z published without {', '.join(pend)}; still not on the Google mirror, nothing to do yet")
                return 0
            prev["complete"] = True
            prev["missing"] = pend
            prev.pop("pending", None)
            MANIFEST.write_text(json.dumps(prev, separators=(",", ":")))
            log(
                f"::warning::{d} {h}Z closed without {', '.join(pend)}: not on the Google mirror by T+{DET_WAIT_H:.0f} h"
            )
            return 0
        log(f"{d} {h}Z was published without {', '.join(pend) or 'some models'}; rebuilding it now")

    import tclib as TC

    fp = CACHE / f"aifs_ens_tf_{d}{h}.bufr"
    if not fp.exists():
        b = D.http_get(f"{GOOGLE}/{d}/{h}z/aifs-ens/0p25/enfo/{d}{h}0000-360h-enfo-tf.bufr", src="ecmwf-google")
        if b:
            fp.write_bytes(b)
    if fp.exists():
        storms = storms_from(TC.decode_all(fp, types=(4,)))
    else:
        # before ECMWF published AIFS-ENS tracks (hindcasts): storms from the AIFS single tracker, every AIFS-ENS member
        # self-tracked from it (closed MSLP low, 250 km search at t=0, then 400 km around its own extrapolated motion)
        fps = oper_tf("aifs-single", d, h)
        if fps is None:
            raise SystemExit("no tracker file for this cycle")
        storms = storms_from(TC.decode_all(fps, types=(0,)), min_tracks=1)
        for s in storms:
            s["seed"] = s["tracks"][0]
            s["tracks"] = {}
        log("  no AIFS-ENS tracker file: members self-tracked from the AIFS single track")
    import shearobs as O

    active = O.nhc_active()
    for s in storms:
        s["sid"], s["basin"] = O.match_sid(s, active, init)
    if a.only:
        storms = [s for s in storms if s["sid"] in a.only.split(",") or s["tid"] in a.only.split(",")]
    log(
        f"{len(storms)} storm(s): "
        + ", ".join(f"{s['name']} ({s['sid']}, {len(s['tracks']) or 'self-tracked'} tracks)" for s in storms)
    )
    man = dict(v=1, dir=OUT_DIR.name, init=f"{init:%Y-%m-%dT%H:00Z}", storms=[], complete=False)
    if not storms:
        man["complete"] = True
        man["built"] = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%MZ}"
        MANIFEST.write_text(json.dumps(man, separators=(",", ":")))
        log("no named storms: empty manifest written")
        return 0

    oi = D.oisst(init, CACHE / "oisst")
    obs0 = {s["sid"]: O.observed_position(s, active) for s in storms}
    ens = run_ecmwf_ens(d, h, storms, oi, a.members) if a.members > 0 else {s["sid"]: {} for s in storms}
    tracks = {model: det_tracks(model, d, h, storms) for model, _ in DET}
    for s in storms:  # the maps follow the lead run: a storm it does not track has none
        tr = (tracks[LEAD] or {}).get(s["sid"])
        s["mapbox"] = mapbox_for(tr) if tr and not a.no_maps else None
    det, det_full, mapbox = {}, {}, {}
    for model, key in DET:
        det_full[model] = False
        if tracks[model] is not None:
            det[key], det_full[model], mb = run_ecmwf_det(model, d, h, storms, tracks[model], oi, maps=model == LEAD)
            mapbox.update(mb)
    det["gfs"], gtrk = run_gfs(d, h, storms, oi, obs0)
    if not a.no_hrrr:
        det["hrrr"] = run_hrrr(d, h, storms, oi, gtrk)

    built = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%MZ}"
    for s in storms:
        sid = s["sid"]
        mem = {m: add_motion(sorted(v, key=lambda r: r["h"])) for m, v in ens[sid].items()}
        models = {
            "aifs-ens": {
                "members": {str(m): v for m, v in sorted(mem.items())},
                "fan": summary(dict(mem.items())),
            }
        }
        for key, rr in det.items():
            if rr.get(sid):
                models[key] = {"run": add_motion(sorted(rr[sid], key=lambda r: r["h"]))}
        try:
            obs = O.observations(sid, init)
        except Exception as e:  # noqa: BLE001
            log(f"::warning::observations for {sid}: {e}")
            obs = dict(btk=[], fcst=None, vdm=[], tdr=[], ships=None, src=[])
        frames = []
        if mapbox.get(sid):
            import shearmaps as M

            frames = M.render(s, mapbox[sid], models.get(LEAD, {}).get("run", []), init, OUT_DIR, obs)
        doc = dict(
            v=1,
            sid=sid,
            name=s["name"],
            basin=s["basin"],
            tid=s["tid"],
            init=man["init"],
            built=built,
            levels=list(S.LEVELS),
            models=models,
            obs=obs,
            frames=frames,
            mapbox=s["mapbox"],
            method=dict(
                r_remove_km=S.R_REMOVE_KM,
                r_env_km=S.R_ENV_KM,
                annulus_km=list(S.ANN_KM),
                r_centroid_km=S.R_CENTROID_KM,
                oisst=oi[3] if oi else None,
            ),
        )
        (OUT_DIR / f"{sid}.json").write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
        c0 = next((models[k]["run"][0] for k in RUNS if k in models), {})  # the lead's analysis, else the next run's
        man["storms"].append(
            dict(
                sid=sid,
                name=s["name"],
                basin=s["basin"],
                json=f"assets/sst/anim/{OUT_DIR.name}/{sid}.json",
                lat=c0.get("lat"),
                lon=c0.get("lon"),
                deep=c0.get("deep"),
                vi=c0.get("vi"),
                tilt=c0.get("tilt"),
                n_mem=len(mem),
                frames=len(frames),
                models=sorted(models),
            )
        )
        try:
            if not a.tag:
                O.update_verify(VERIFY, sid, s["name"], init, models, obs, RUNS)
        except Exception as e:  # noqa: BLE001
            log(f"  verify archive: {e}")
    man["maps_resol"] = "0p25"
    if map_resol != "0p25" and not a.no_maps:  # the 0.1-degree run is already out: redraw the maps from it
        redraw_maps(man, d, h, init, map_resol)
    close_cycle(man, det_full, init)
    man["built"] = built
    man["bytes"] = D.LEDGER.report()
    man["runtime_s"] = round(time.time() - T0)
    MANIFEST.write_text(json.dumps(man, separators=(",", ":")))
    log(f"done in {time.time() - T0:.0f} s; GB fetched {D.LEDGER.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
