#!/usr/bin/env python3
"""The tropical-cyclone storm board, from ECMWF's two ensembles (diagnostic only: where the storms go, when they turn
and transition, how deep they get; nothing about what a storm does to the flow).

Inputs, one 00 or 12 UTC cycle, from the Google Cloud mirror only:
  tracks   ECMWF's tracker on AIFS-ENS and IFS-ENS (tclib.tracker_file): the 50 perturbed members of each. The
           deterministic runs in the same files (AIFS single, IFS control) are drawn as thin reference lines. The AIFS-ENS
           control is not used.
  history  the official positions of the storms with an id in the tracker files of the HIST_CYCLES cycles before
           (12-hourly), put in front of every member's track for the recurvature test (history)
  msl      each ensemble's own 50 members, 12-hourly to day 10 on a 0.5-degree grid: every member is followed past the
           tracker's last fix as a surface low (tclib.follow_low) and the depth of the system it becomes is measured
           (depth_after)
Storms: every storm with an official id (observed at t=0) and every developing system that at least SUPPORT_FRAC of the
members of either ensemble take to tropical-storm strength (nh_storms gathers a system's genesis points across the
tracker's provisional ids; pair_genesis pairs the two ensembles' systems by where they are).
Per member, on its tracker track with the observed history in front (outcome):
  now         the turn from westward to eastward motion (tclib.recurvature) falls at or before t=0 + NOW_H, or the storm
              already moves east in the subtropics at t=0
  by          the turn comes later, by day N_DAYS (dated)
  no          no turn by day N_DAYS, and the track tells: it lasts to day N_DAYS or ends while still moving west
  unresolved  the track ends first while not moving west (stalled, or turning)
  absent      the member does not carry the storm (a developing system it does not develop)
Post-tropical (post_tropical_h): the tracker's last fix, when it falls north of PT_LAT before day N_DAYS and before the
run's end; ECMWF's tracker stops at or near extratropical transition, and the low is followed on from there (dashed).
Pooled odds: the members of both ensembles together, with a 90 % Wilson interval for the share. Watch: a storm in the
North Pacific or the North Atlantic whose pooled chance of recurving by day WATCH_DAY (now or by then) has a lower 90 %
bound of at least WATCH_P, with the median member's turn by day WATCH_DAY. With one ensemble's file missing the other
is used alone, and the board says which is missing. The track record (tc_verify.py ->
scripts/mjo/data/reference/tc_verify.json) is quoted with the odds.
Outputs: data/tc_board.json and per storm tc/tracks_<storm>.webp, tc/depth_<storm>.webp. Status "ok", "none" (the tracker
files were read and hold no Northern-Hemisphere storm) or "missing" (no tracker file could be had or read).
    python src/tc_ens.py --date 20261018 --time 00 --out-dir ../../assets/sst
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tclib as TC

SUPPORT_FRAC = 0.4  # share of an ensemble's members that must take a developing system to tropical-storm strength for
# it to be shown (one ensemble is enough); a storm observed at t=0 is always shown
N_DAYS = 10  # the outcomes' horizon
NOW_H = 12  # a turn at or before t=0 + NOW_H is "recurving now"
PT_LAT = 30.0  # a member whose tracker track ends north of this latitude has become post-tropical there
WATCH_P = 0.5  # the watch: lower 90 % bound of the pooled chance of recurving by day WATCH_DAY at least WATCH_P
WATCH_DAY = 10
Z90 = 1.6449  # two-sided 90 % normal quantile (Wilson interval)
HIST_CYCLES = 4  # observed history: 48 h of official positions, 12-hourly
MSL_STEPS = tuple(range(0, TC.FOLLOW_H_MAX + 1, 12))
MAX_PER_STORM = 10  # per-storm figures on the board
CACHE = Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc"
REF = Path(__file__).resolve().parents[1] / "data" / "reference" / "tc_verify.json"
UNAVAILABLE = "Track data unavailable for this cycle"
OUTCOMES = ("now", "by", "no", "unresolved", "absent")
OUT_LAB = {
    "now": "recurving now",
    "by": f"recurves by day {N_DAYS}",
    "no": f"no recurvature by day {N_DAYS}",
    "unresolved": "unresolved",
    "absent": "does not carry it",
}
OUT_COL = {"now": "#2b0b4f", "by": "#d9546f", "no": "#9a9a9a", "unresolved": "#c9b38a", "absent": "#ffffff"}
BASIN_OF = {"W": "WP", "E": "EP", "C": "EP", "L": "NA", "A": "NI", "B": "NI"}
BASIN_LAB = {
    "WP": "western North Pacific",
    "EP": "eastern North Pacific",
    "NA": "North Atlantic",
    "NI": "North Indian Ocean",
}
NI_LON = 100.0  # west of 100E a storm is in the North Indian Ocean: listed, never on watch


def slug(key):
    """File-safe storm key ('ifs:71W' -> 'ifs-71W')."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(key))


def _dup(tr, other) -> bool:
    for hh in np.intersect1d(tr["steps"], other["steps"]):
        a = TC.at(tr, int(hh))
        b = TC.at(other, int(hh))
        if TC._km(a[0], a[1], b[0], b[1]) < 150:
            return True
    return False


GEN_KM, GEN_H = 600.0, 48  # genesis points this close in space and time are one developing system
TS_WIND = 17.0  # m/s: tropical-storm strength
MIN_COMPANION = 3  # a group under SUPPORT_FRAC is kept only as the OTHER ensemble's view of a system that passes there
# ("IFS-ENS: 14 of 50 members", not "not in IFS-ENS"); never on its own
MATCH_H = 72  # hours of shared median track compared when pairing the two ensembles' developing systems


def median_track(s: dict) -> dict:
    """Lead hour -> median member (lat, lon) where at least a quarter of the members (and 3, or every track of a system
    with fewer) have a position. Longitudes are unwrapped about the first member before the median, so a system on the
    dateline is not averaged to the far side."""
    tr = list(s["tracks"].values())
    need = max(min(3, len(tr)), 0.25 * len(tr))
    out = {}
    for h in np.unique(np.concatenate([t["steps"] for t in tr])):
        ps = [q for q in (TC.at(t, int(h)) for t in tr) if q is not None]
        if len(ps) >= need:
            lo0 = ps[0][1]
            out[int(h)] = (
                float(np.median([q[0] for q in ps])),
                float(np.median([(q[1] - lo0 + 180) % 360 - 180 + lo0 for q in ps])),
            )
    return out


def genesis_h(t):
    """Hour a member track first reaches tropical-storm strength, or None."""
    w = np.nan_to_num(t["wind"], nan=0.0)
    return int(t["steps"][int(np.argmax(w >= TS_WIND))]) if (w >= TS_WIND).any() else None


def gen_h(s: dict) -> float:
    """Median hour at which a developing system's members reach tropical-storm strength."""
    return float(np.median([g for g in (genesis_h(t) for t in s["tracks"].values()) if g is not None] or [0]))


def track_gap(sa: dict, si: dict) -> float:
    """Median distance (km) between two systems' median tracks over the MATCH_H hours after the earlier of their median
    genesis hours (inf if fewer than 3 shared steps there). Anchored at genesis because the tracker also follows the
    pre-genesis disturbances, which scatter: 8 Oct 00Z AIFS 72W / IFS 73W were 736 km apart over their first shared
    hours (h48-96) but 86-90 km apart once both had developed (h120-144)."""
    h0 = min(gen_h(sa), gen_h(si))
    ma, mi = median_track(sa), median_track(si)
    common = [h for h in sorted(set(ma) & set(mi)) if h0 <= h <= h0 + MATCH_H]
    if len(common) < 3:
        return np.inf
    return float(np.median([TC._km(*ma[h], *mi[h]) for h in common]))


def match_gap(sa: dict, si: dict) -> float:
    """How far apart two developing systems are, for deciding whether they are one: track_gap wherever their median tracks
    share 3 steps after genesis, else the distance between their genesis points if those are within GEN_H of each other
    (inf otherwise). A pair closer than GEN_KM is the same system."""
    d = track_gap(sa, si)
    if not np.isfinite(d) and abs(gen_h(sa) - gen_h(si)) <= GEN_H:
        d = TC._km(*sa["genesis"], *si["genesis"])
    return d


def region_of(lat, lon):
    """A plain-language place for a developing system (Atlantic and eastern Pacific; None elsewhere)."""
    lo = (lon + 180) % 360 - 180
    for name, la0, la1, lo0, lo1 in (
        ("Bay of Campeche", 17.5, 22.5, -98.5, -90.0),
        ("Gulf of Mexico", 21.0, 31.0, -98.5, -80.5),
        ("western Caribbean", 9.0, 22.0, -89.0, -75.0),
        ("eastern Caribbean", 9.0, 19.0, -75.0, -60.0),
        ("off the Southeast US coast", 24.0, 36.0, -82.0, -70.0),
        ("central tropical Atlantic", 7.0, 25.0, -60.0, -35.0),
        ("eastern tropical Atlantic", 5.0, 25.0, -35.0, -15.0),
        ("subtropical Atlantic", 25.0, 40.0, -70.0, -20.0),
        ("off southern Mexico", 8.0, 20.0, -110.0, -90.0),
        ("eastern Pacific", 5.0, 25.0, -140.0, -110.0),
    ):
        if la0 <= lat <= la1 and lo0 <= lo <= lo1:
            return name
    return None


_PACIFIC_REGIONS = {"off southern Mexico", "eastern Pacific"}


def basin_of(sid, lat, lon):
    """'L' Atlantic, 'E' eastern Pacific, else the tracker id's letter. From WHERE the system develops when that is known:
    a Pacific disturbance that crosses the isthmus keeps its E id (e.g. an IFS 71E developing in the Bay of Campeche)."""
    r = region_of(lat, lon)
    if r is None:
        return sid[-1:]
    return "E" if r in _PACIFIC_REGIONS else "L"


def nh_storms(msgs: list[dict], min_frac: float = SUPPORT_FRAC, nmem: int = TC.NMEM) -> list[dict]:
    """One ensemble's Northern-Hemisphere systems (members only, tclib.decode_all types=(4,)): the storms with an official
    id, plus developing systems reached by >= min_frac of DISTINCT members at tropical-storm strength.

    ECMWF's tracker splits one developing system across several provisional ids (e.g. a Bay of Campeche system in one
    00Z AIFS-ENS run sat under 70L, 71L, 72L, ... with 9 / 13 / 7 ... tropical-storm members each, so every id failed
    the 40 % test on its own while 31 of 51 distinct members developed it). Genesis is therefore clustered across ids:
    each member track that reaches tropical-storm strength contributes its genesis point (where and when it first does),
    points within GEN_KM and GEN_H are linked, and a cluster counts its distinct members (each member's strongest track
    is kept once). A genesis track that duplicates the same member's track of a named storm (within 150 km at a shared
    step) is dropped first. A cluster below min_frac is kept as weak (shown only beside the other ensemble's system)."""
    named = [m for m in msgs if m["named"]]
    keep = []
    for m in named:
        tr = {k: t for k, t in m["tracks"].items() if (t["lat"] > 0).any() and len(t["steps"]) > 1}
        if tr and (m["obs"] is None or m["obs"][0] > 0):
            keep.append(dict(m, tracks=tr))
    pts = []  # (id, member, track, lat, lon, step) at genesis
    for m in msgs:
        if m["named"]:
            continue
        for k, t in m["tracks"].items():
            if not ((t["lat"] > 0).any() and len(t["steps"]) > 1) or any(
                k in n["tracks"] and _dup(t, n["tracks"][k]) for n in named
            ):
                continue
            w = np.nan_to_num(t["wind"], nan=0.0)
            if w.max() < TS_WIND:
                continue
            i = int(np.argmax(w >= TS_WIND))
            if t["lat"][i] > 0:
                pts.append((m["id"], k, t, float(t["lat"][i]), float(t["lon"][i]), int(t["steps"][i])))
    bas = [basin_of(p_[0], p_[3], p_[4]) for p_ in pts]
    par = list(range(len(pts)))  # union-find, single linkage

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (
                bas[i] == bas[j]
                and abs(pts[i][5] - pts[j][5]) <= GEN_H
                and TC._km(pts[i][3], pts[i][4], pts[j][3], pts[j][4]) < GEN_KM
            ):  # same basin letter: the Bay of
                # Campeche and the Gulf of Tehuantepec are ~300 km apart across the isthmus and must never merge
                par[root(i)] = root(j)
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(root(i), []).append(pts[i])
    used = set()  # one tracker id can head several groups:
    for g in sorted(groups.values(), key=lambda g: -len({x[1] for x in g})):  # the largest keeps the bare id
        best = {}
        for sid, k, t, la, lo, _h in g:  # one track per member: its strongest
            if k not in best or np.nanmax(t["wind"]) > np.nanmax(best[k][1]["wind"]):
                best[k] = (sid, t, la, lo)
        weak = len(best) < min_frac * nmem
        if weak and len(best) < MIN_COMPANION:
            continue
        ids = {}
        for sid, _, _, _ in best.values():
            ids[sid] = ids.get(sid, 0) + 1
        main_id = max(ids, key=ids.get)
        n_ = 2
        while main_id in used:
            main_id = f"{max(ids, key=ids.get)}-{n_}"
            n_ += 1
        used.add(main_id)
        gla = float(np.median([v[2] for v in best.values()]))
        glo = TC.lon_median([v[3] for v in best.values()])
        keep.append(
            dict(
                id=main_id,
                name=main_id,
                named=False,
                obs=None,
                weak=weak,
                ids=sorted(ids),
                region=region_of(gla, glo),
                basin=basin_of(main_id[:3], gla, glo),
                genesis=(round(gla, 1), round(glo, 1)),
                tracks={k: v[1] for k, v in best.items()},
            )
        )
    return keep


def pair_genesis(per: dict) -> dict:
    """The AIFS-ENS and IFS-ENS developing systems that are one system: {id(IFS-ENS system): AIFS-ENS system}.

    The two trackers number the same system differently (e.g. AIFS 71L = IFS 70L), so systems are matched by WHERE they
    are, not by id. Genesis medians alone are not enough: a system that reaches 17 m/s ~2 days later in one ensemble can
    have genesis medians 1,000+ km apart along one track (median tracks 12 km apart at the closest), and genesis points
    ignore WHEN (a system can form where another formed a week earlier). A pair is therefore judged by match_gap, same
    basin, not both below SUPPORT_FRAC; the closest pairs are matched first."""
    cand = []
    for sa in (x for x in per.get("aifs", []) if not x["named"]):
        for si in (x for x in per.get("ifs", []) if not x["named"]):
            if sa["basin"] != si["basin"] or (sa.get("weak") and si.get("weak")):
                continue
            d = match_gap(sa, si)
            if d < GEN_KM:
                cand.append((d, sa, si))
    pair, taken = {}, set()
    for _, sa, si in sorted(cand, key=lambda c: c[0]):
        if id(si) in pair or id(sa) in taken:
            continue
        pair[id(si)] = sa
        taken.add(id(sa))
    return pair


def assemble(per: dict) -> list[dict]:
    """The board's storms from both ensembles' systems: storms with an official id by id; a developing system under the
    AIFS-ENS id when pair_genesis pairs it with an IFS-ENS one, else per ensemble. Systems below SUPPORT_FRAC in every
    ensemble that carries them are left out (the support rule)."""
    storms = {}
    pair = pair_genesis(per)  # id(ifs system) -> aifs system
    for m, lst in per.items():
        for s in lst:
            if s["named"]:
                key = s["id"]
            elif m == "ifs" and id(s) in pair:
                key = "gen:" + pair[id(s)]["id"]
            else:
                key = ("gen:" if m == "aifs" else f"{m}:") + s["id"]
            ref = pair[id(s)] if (m == "ifs" and id(s) in pair) else s
            st = storms.setdefault(
                key,
                dict(
                    key=key,
                    id=ref["id"],
                    name=ref["name"],
                    named=s["named"],
                    region=ref.get("region"),
                    basin=ref.get("basin") or s["id"][-1:],
                    tag="IFS-ENS " if key.startswith("ifs:") else "",
                    obs=s.get("obs"),
                    models={},
                ),
            )
            st["models"][m] = s
            if st["obs"] is None:
                st["obs"] = s.get("obs")
    return [v for v in storms.values() if not all(x.get("weak") for x in v["models"].values())]


def match_det(s: dict, det: list[dict]):
    """The deterministic run's track of a board storm (det: tclib.decode_all types=(0,) of the same file), or None. A storm
    with an id: the same id. A developing system: the deterministic system of the same basin closest by match_gap
    (below GEN_KM), its track taken as a one-member system."""
    if s["named"]:
        x = next((d for d in det if d["named"] and d["id"] == s["id"]), None)
        return x["tracks"][0] if x else None
    best, bd = None, GEN_KM
    for d in det:
        t = d["tracks"][0]
        g = genesis_h(t)
        if d["named"] or g is None:
            continue
        k = int(np.where(t["steps"] == g)[0][0])
        one = dict(tracks={0: t}, genesis=(float(t["lat"][k]), float(t["lon"][k])))
        if basin_of(d["id"], *one["genesis"]) != s["basin"]:
            continue
        for S in s["models"].values():
            dd = match_gap(one, S)
            if dd < bd:
                best, bd = t, dd
    return best


# ── per member: the observed history, recurvature, post-tropical, depth ────────────────────────────────────────────
def history(date: str, hh: str) -> dict:
    """{storm id: [(h, lat, lon), ...]} with h < 0: the official positions of the storms with an id in the tracker files of
    the HIST_CYCLES cycles before this one (IFS-ENS, else AIFS-ENS; a cycle with neither is a gap)."""
    import pandas as pd

    t0 = pd.Timestamp(f"{date}T{int(hh):02d}:00")
    out = {}
    for k in range(1, HIST_CYCLES + 1):
        t = t0 - pd.Timedelta(hours=12 * k)
        for m in ("ifs", "aifs"):
            fp = TC.tracker_file(m, f"{t:%Y%m%d}", f"{t:%H}", CACHE, quiet=True)
            if fp is None:
                continue
            try:
                msgs = TC.decode_all(fp, types=(0, 4))
            except Exception as e:  # noqa: BLE001  an unreadable earlier file is a gap in the history
                print(f"  history: {fp.name} unreadable ({e!r})", flush=True)
                fp.unlink(missing_ok=True)
                continue
            for x in msgs:
                if x["named"] and x["obs"] is not None:
                    out.setdefault(x["id"], []).append((-12 * k, *x["obs"]))
            break
    return {sid: sorted(v) for sid, v in out.items()}


def with_history(tr: dict, hist) -> dict:
    """The track with the observed positions before its first fix in front (pmsl and wind NaN there)."""
    hs = [x for x in hist or () if x[0] < int(tr["steps"][0])]
    if not hs:
        return tr
    n = len(hs)
    return dict(
        steps=np.r_[[x[0] for x in hs], tr["steps"]].astype(int),
        lat=np.r_[[x[1] for x in hs], tr["lat"]],
        lon=np.r_[[x[2] for x in hs], tr["lon"]],
        pmsl=np.r_[np.full(n, np.nan), tr["pmsl"]],
        wind=np.r_[np.full(n, np.nan), tr["wind"]],
    )


def outcome(t: dict, run_h: int) -> tuple[str, int | None]:
    """(outcome, turn lead h) of one track with its observed history in front: "now", "by", "no" or "unresolved" (see
    the module docstring). run_h: the run's length, the latest a track can be followed."""
    st = t["steps"]
    r = TC.recurvature(t)
    if r is not None and r <= NOW_H:
        return "now", r
    u = TC.zonal_motion(t)
    k0 = np.where(st == 0)[0]
    if r is None and len(k0) and u[k0[0]] >= TC.REC_EAST and t["lat"][k0[0]] >= 20.0:
        return "now", 0  # already in the westerlies at the start
    if r is not None and r <= 24 * N_DAYS:
        return "by", r
    if st[-1] >= min(24 * N_DAYS, run_h - 6):
        return "no", None
    fin = np.where(np.isfinite(u))[0]
    if len(fin) and u[fin[-1]] <= -1.0:
        return "no", None  # it ended while still moving west
    return "unresolved", None


def post_tropical_h(tr: dict, run_h: int) -> int | None:
    """Lead of the tracker's last fix when it falls north of PT_LAT, by day N_DAYS and before the run's end, else None."""
    h = int(tr["steps"][-1])
    return h if tr["lat"][-1] >= PT_LAT and h <= 24 * N_DAYS and h < run_h - 6 else None


ET_LAT = 35.0  # depth after transition: the lowest central pressure north of this latitude ...
SYS_KM = 1200.0  # ... on the storm's own (followed) track or in any closed low this close to its latest position ...
SYS_H = 72  # ... from the tracker's last fix to this many hours later
EXTRAP_H = 48  # once the track ends, the search centre moves on with its last motion for up to this long
MIN_DEPTH_MEMBERS = 20  # the depth chart needs this many members (both ensembles) reaching ET_LAT
END_H = TC.FOLLOW_H_MAX - 12  # a member deepest this late may still be deepening when the follow stops at day 10


def depth_after(tr: dict, ext: dict, Mk, lat, lon, steps, land=None) -> dict | None:
    """The depth of the SYSTEM a member's storm becomes: dict(pmin, h, lat, lon, on_track) or None.

    During phasing the low is often double-centred and ECMWF's tracker can stay with a decaying southern remnant while the
    main low re-forms 1000+ km to the north (Choi-Wan, 2 Oct 00Z: 20 of 21 "shallow" members had a 935-955 hPa low next to
    their followed path). Depth is therefore the lowest central pressure, north of ET_LAT, of the storm's own extended
    track (ext: tracker + follow_low) and of any closed low within SYS_KM of its latest position over SYS_H hours from the
    tracker's last fix. A weak low (>= tclib.FOLLOW_WEAK_HPA) over land is not part of it (heat / lee lows; land:
    tclib.LandMask, None = everywhere), and nothing at or above tclib.FOLLOW_FILLED_HPA counts (a 1017 hPa "low" is a
    filled-in storm, not a depth: 18 Oct 2026 00Z, Champi). Mk: the member's msl (steps, lat, lon) hPa."""
    stl = list(np.asarray(steps, int))
    m = (
        (ext["lat"] >= ET_LAT)
        & (ext["steps"] <= TC.FOLLOW_H_MAX)
        & np.isfinite(ext["pmsl"])
        & (ext["pmsl"] < TC.FOLLOW_FILLED_HPA)
    )
    cands = [
        (float(ext["pmsl"][i]), int(ext["steps"][i]), float(ext["lat"][i]), float(ext["lon"][i]), True)
        for i in np.where(m)[0]
    ]  # (p, h, lat, lon, on_track)
    on = [int(h) for h in tr["steps"] if int(h) in set(stl)]
    if on:
        h_beg, h_end = on[-1], int(tr["steps"][-1])  # SYS_H after the LAST fix (often +6 h off-grid)
        for h in stl:
            if h < h_beg or h > min(h_end + SYS_H, TC.FOLLOW_H_MAX):
                continue
            before = np.where(ext["steps"] <= h)[0]
            c_la, c_lo = float(ext["lat"][before[-1]]), float(ext["lon"][before[-1]])
            gap = h - int(ext["steps"][before[-1]])
            if gap > 0 and len(before) >= 2:  # the track has ended: move the search centre
                j1, j0 = before[-1], before[-2]  # on with the last motion, for up to EXTRAP_H
                dt = (int(ext["steps"][j1]) - int(ext["steps"][j0])) or 6
                f_ = min(gap, EXTRAP_H) / dt
                c_la = float(np.clip(c_la + f_ * (ext["lat"][j1] - ext["lat"][j0]), -80, 80))
                c_lo = float(((c_lo + f_ * (((ext["lon"][j1] - ext["lon"][j0] + 180) % 360) - 180) + 180) % 360) - 180)
            for la, lo, p, _ in TC.closed_lows(Mk[stl.index(h)], lat, lon, (c_la, c_lo), SYS_KM):
                if (
                    la >= ET_LAT
                    and p < TC.FOLLOW_FILLED_HPA
                    and not (p >= TC.FOLLOW_WEAK_HPA and (land is None or land(la, lo)))
                ):
                    cands.append((float(p), int(h), float(la), float(lo), False))
    if not cands:
        return None
    p, h, la, lo, ontr = min(cands, key=lambda c: c[0])
    return dict(pmin=round(p, 1), h=h, lat=la, lon=lo, on_track=ontr)


def load_msl(model: str, date: str, hh: str):
    """One ensemble's 50 members' mean sea-level pressure ({member: (steps, lat, lon) hPa}, lat, lon, steps): 12-hourly to
    day 10 on a 0.5-degree grid, 10-82N, through the shared ECMWF store (Google mirror)."""
    import xarray as xr
    import store as ecmwf

    p = ecmwf.ensure(
        ecmwf.Cycle(date, f"{int(hh):02d}"), ecmwf.Spec(TC.MODEL_DIR[model], "pf", "msl", "sfc", (), MSL_STEPS)
    )
    with xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        da = ds["msl"]
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude").sortby("latitude")
        da = (
            da.sel(latitude=slice(10.0, 82.0)).coarsen(latitude=2, longitude=2, boundary="trim").mean() / 100.0
        ).load()
    out = {int(n): da.sel(number=n).values.astype("float32") for n in da.number.values}
    return out, da.latitude.values, da.longitude.values, (da.step / np.timedelta64(1, "h")).values.astype(int)


def analyse(S: dict | None, hist, run_h: int, msl=None, land=None) -> dict:
    """Every member of one ensemble for one storm: {member: dict(outcome, rec_h, pt_h, track, ext, depth)}; S = that
    ensemble's system (None: no member carries it), msl = load_msl's result or None (no follow, no depth)."""
    res = {}
    for k in range(1, TC.NMEM + 1):
        tr = (S or {}).get("tracks", {}).get(k)
        if tr is None:
            res[k] = dict(outcome="absent", rec_h=None, pt_h=None, track=None, ext=None, depth=None)
            continue
        o, r = outcome(with_history(tr, hist), run_h)
        ext, dep = tr, None
        if msl is not None and k in msl[0]:
            M, lat, lon, steps = msl
            cont = TC.follow_low(M[k], lat, lon, steps, tr, h_max=TC.FOLLOW_H_MAX, merge=True, land=land)
            ext = TC.extended(tr, cont)
            dep = depth_after(tr, ext, M[k], lat, lon, steps, land)
        res[k] = dict(outcome=o, rec_h=r, pt_h=post_tropical_h(tr, run_h), track=tr, ext=ext, depth=dep)
    return res


def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    """Wilson score interval for a share k / n (two-sided; z = 1.645 for 90 %)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _q(v, qs=(10, 50, 90)):
    return [round(float(x), 1) for x in np.percentile(v, qs)] if len(v) else None


def summarise(rs: list[dict]) -> dict:
    """The board's numbers for a set of member results (one ensemble, or both pooled)."""
    n = len(rs)
    cnt = {o: sum(r["outcome"] == o for r in rs) for o in OUTCOMES}
    k = cnt["now"] + cnt["by"]
    lo, hi = wilson(k, n)
    turn = sorted(r["rec_h"] for r in rs if r["outcome"] in ("now", "by"))
    allm = sorted([r["rec_h"] for r in rs if r["outcome"] in ("now", "by")] + [np.inf] * (n - k))
    med_all = float(np.median(allm)) if n else np.inf  # median member, a member without a turn counting as never
    pt = [r["pt_h"] for r in rs if r["pt_h"] is not None]
    dep = [r["depth"]["pmin"] for r in rs if r["depth"] is not None]
    return dict(
        members=n,
        carried=n - cnt["absent"],
        counts=cnt,
        p_recurve=round(k / n, 3) if n else None,
        p_recurve_90=[round(lo, 3), round(hi, 3)] if n else None,
        recurve_h_p10_p50_p90=_q(turn),
        median_member_turn_h=None if not np.isfinite(med_all) else round(med_all, 1),
        p_post_tropical=round(len(pt) / n, 3) if n else None,
        post_tropical_h_p10_p50_p90=_q(pt),
        depth_members=len(dep),
        depth_hpa_p10_p50_p90=_q(dep),
        deepest_at_end=sum(1 for r in rs if r["depth"] is not None and r["depth"]["h"] >= END_H),
        below_960=int(np.sum(np.array(dep) < 960)),
        below_950=int(np.sum(np.array(dep) < 950)),
    )


def det_summary(tr, hist, run_h: int) -> dict | None:
    """The deterministic run's own track, outcome and post-tropical time (no follow: a reference line)."""
    if tr is None:
        return None
    o, r = outcome(with_history(tr, hist), run_h)
    return dict(
        outcome=o,
        turn_h=r,
        post_tropical_h=post_tropical_h(tr, run_h),
        min_pmsl=None if not np.isfinite(tr["pmsl"]).any() else round(float(np.nanmin(tr["pmsl"])), 1),
        h=[int(h) for h in tr["steps"]],
        lat=[round(float(v), 2) for v in tr["lat"]],
        lon=[round(float(v), 2) for v in tr["lon"]],
        pmsl=[None if not np.isfinite(v) else round(float(v), 1) for v in tr["pmsl"]],
    )


def watch_basin(s: dict, med_h) -> str | None:
    """The basin of the storm's letter (the id's, or where a developing system forms: basin_of), except that a Pacific
    storm is placed by the members' median position nearest the median turn (Nolo 2026 was 15E but recurved near 168E:
    a western-Pacific recurvature) and anything west of NI_LON is North Indian. None in the Southern Hemisphere. A
    longitude split between the eastern Pacific and the Atlantic would put a system off southern Mexico (95W) in the
    Atlantic."""
    base = BASIN_OF.get(str(s.get("basin") or s["id"])[-1:].upper())
    allt = dict(enumerate(t for S in s["models"].values() for t in S["tracks"].values()))
    mt = median_track(dict(tracks=allt)) if allt else {}
    if mt:
        h = min(mt, key=lambda x: abs(x - (med_h if med_h is not None else 0)))
        la, lo = mt[h]
        if la <= 0:
            return None
        lon360 = lo % 360
        if lon360 < NI_LON:
            return "NI"
        if base in ("WP", "EP"):
            return "WP" if lon360 < 180 else "EP"
    return base


def _stormlab(s):
    if s["named"]:
        return s["name"].title()
    tag = s.get("tag", "") + s["id"]  # "IFS-ENS 71W": the two trackers number systems independently
    return f"possible new storm, {s['region']} ({tag})" if s.get("region") else f"possible new storm ({tag})"


def _pct(x) -> str:
    return f"{100 * x:.0f} %"


def watch_lines(storms: list[dict], mstat: dict, init) -> tuple[list[str], list[str]]:
    """(the watch line(s), keys of the storms on watch)."""
    on, lines = [s for s in storms if s.get("watch")], []
    if on:
        parts = []
        for s in on:
            q = s["pooled"]
            d = TC.date_lab(init, q["median_member_turn_h"] / 24, "short")
            parts.append(
                f"{_stormlab(s)}: {_pct(q['p_recurve'])} of {q['members']} members recurve by day {WATCH_DAY} "
                f"(90 % interval {_pct(q['p_recurve_90'][0])}–{_pct(q['p_recurve_90'][1])}), median member ~{d}"
            )
        lines.append("Recurvature watch: " + "; ".join(parts) + ".")
    else:
        lines.append("No recurvature watch.")  # the rule is in tc_board.json's rules and on the page's About text
    gone = [m for m in TC.MODELS if mstat.get(m) != "ok"]
    have = [m for m in TC.MODELS if mstat.get(m) == "ok"]
    if gone and have:
        lines.append(
            f"{', '.join(TC.MODELS[m] for m in gone)}: {UNAVAILABLE.lower()}; the odds use "
            f"{', '.join(TC.MODELS[m] for m in have)} alone."
        )
    return lines, [s["key"] for s in on]


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────
REC_BOUNDS = [0, 2, 4, 6, 8, 10]
REC_COLS = ["#6a2c91", "#a63e8c", "#d9546f", "#f17c4a", "#f8b33f"]  # the turn date by 2-day bins ("now" is darker)


def _member_col(r):
    if r["outcome"] == "by":
        from matplotlib.colors import BoundaryNorm, ListedColormap

        return ListedColormap(REC_COLS)(BoundaryNorm(REC_BOUNDS, len(REC_COLS))(r["rec_h"] / 24))
    return OUT_COL[r["outcome"]] if r["outcome"] != "absent" else "#9a9a9a"


def render_tracks(s: dict, init, out_png: Path, mstat: dict):
    """One storm: each ensemble's members on a Lambert fan (coloured by their recurvature outcome and date, the part
    followed past the tracker dashed), the observed history, the deterministic run as a thin black line and the member
    nearest the median track dated; beside them the outcome shares and the timing of the turn and of the transition.
    Layout in inches."""
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    pts_la, pts_lo = [], []
    for m in s["res"]:
        for r in s["res"][m].values():
            if r["ext"] is not None:
                pts_la.append(r["ext"]["lat"])
                pts_lo.append(r["ext"]["lon"])
    for h in s.get("hist") or []:
        pts_la.append([h[1]])
        pts_lo.append([h[2]])
    lon0, lon1, lat0, lat1 = TC.track_sector(np.concatenate(pts_la), np.concatenate(pts_lo))
    asp = TC.sector_aspect(lon0, lon1, lat0, lat1)
    W, top, bot = 16.5, 0.78, 1.12
    mh = min(5.6 * asp, 5.6)  # each map at most 5.6 in wide and high, true aspect
    mw = mh / asp
    hb = max(mh, 4.6)  # the band the maps and the right-hand panels share
    H = top + hb + bot
    fig = plt.figure(figsize=(W, H))

    def fy(y):  # inches from the bottom -> figure fraction
        return y / H

    fig.text(
        0.12 / W,
        1 - 0.1 / H,
        f"{_stormlab(s)} — AIFS-ENS and IFS-ENS tracks (50 members each), init {init:%d %b %Y %HZ}",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    cmap, norm = ListedColormap(REC_COLS), BoundaryNorm(REC_BOUNDS, len(REC_COLS))
    for c, m in enumerate(TC.MODELS):
        x0 = 0.12 + c * (mw + 0.3)
        ax = TC.sector_axes(fig, [x0 / W, fy(bot + (hb - mh) / 2), mw / W, mh / H], lon0, lon1, lat0, lat1)
        S = s["models"].get(m)
        q = s["summary"].get(m)
        if mstat.get(m) != "ok":
            ttl = f"{TC.MODELS[m]}: {UNAVAILABLE[0].lower() + UNAVAILABLE[1:]}"
        elif S is None:
            ttl = f"{TC.MODELS[m]}: no member carries it"
        else:
            ttl = (
                f"{TC.MODELS[m]}: {q['carried']} of {q['members']} members carry it · "
                f"{_pct(q['p_recurve'])} recurve by day {N_DAYS}"
            )
        fig.text(x0 / W, fy(bot + hb + 0.08), ttl, fontsize=10.2, va="bottom")
        if S is None:
            continue
        res = s["res"][m]
        for r in sorted(res.values(), key=lambda r: r["outcome"] == "by"):  # the dated members on top
            if r["ext"] is None:
                continue
            col = _member_col(r)
            x = r["ext"]
            fol = x.get("followed", np.zeros(len(x["steps"]), bool))
            ax.plot(x["lon"][~fol], x["lat"][~fol], color=col, lw=0.85, alpha=0.8, transform=ccrs.Geodetic(), zorder=3)
            if fol.any():
                j0 = max(int(np.where(fol)[0][0]) - 1, 0)
                ax.plot(
                    x["lon"][j0:],
                    x["lat"][j0:],
                    color=col,
                    lw=0.75,
                    alpha=0.75,
                    ls=(0, (3, 1.6)),
                    transform=ccrs.Geodetic(),
                    zorder=3,
                )
            if r["outcome"] == "by":
                p_ = TC.at(r["track"], r["rec_h"])
                if p_ is not None:
                    ax.plot(
                        p_[1],
                        p_[0],
                        marker="o",
                        ms=3.0,
                        color=col,
                        mec="#000",
                        mew=0.3,
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
            if not s["named"]:
                g = genesis_h(r["track"])
                if g is not None:
                    pg = TC.at(r["track"], g)
                    ax.plot(
                        pg[1],
                        pg[0],
                        marker="D",
                        ms=3.4,
                        mfc="#fff",
                        mec="#222",
                        mew=0.6,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )
        kref = ref_member({k: r["ext"] for k, r in res.items() if r["ext"] is not None})
        tr0 = res[kref]["ext"]
        ax.plot(tr0["lon"], tr0["lat"], color="#111", lw=3.0, transform=ccrs.Geodetic(), zorder=7)
        ax.plot(tr0["lon"], tr0["lat"], color=_member_col(res[kref]), lw=1.7, transform=ccrs.Geodetic(), zorder=7)
        last = []
        for kd in range(1, int(tr0["steps"][-1] // 24) + 1):
            qd = TC.at(tr0, 24 * kd)
            if qd is None:
                continue
            ax.plot(
                qd[1],
                qd[0],
                marker="o",
                ms=4.6,
                mfc="#fff",
                mec="#111",
                mew=1.0,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
            xy = ax.projection.transform_point(qd[1], qd[0], ccrs.PlateCarree())
            disp = ax.transData.transform(xy)
            if any(np.hypot(*(disp - o)) < 34 for o in last):  # skip a label that would overlap
                continue
            last.append(disp)
            ax.annotate(
                TC.date_lab(init, kd, "short"),
                xy,
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.6,
                fontweight="bold",
                color="#111",
                zorder=9,
                path_effects=[pe.Stroke(linewidth=2.4, foreground="#fff"), pe.Normal()],
            )
        dt_ = s["det"].get(m)
        if dt_ is not None:
            ax.plot(dt_["lon"], dt_["lat"], color="#000", lw=1.0, ls=(0, (6, 2)), transform=ccrs.Geodetic(), zorder=8)
        hist = s.get("hist") or []
        if hist:
            hl = np.array([[h[1], h[2]] for h in hist] + [[tr0["lat"][0], tr0["lon"][0]]])
            ax.plot(hl[:, 1], hl[:, 0], color="#333", lw=2.6, transform=ccrs.Geodetic(), zorder=8)
            ax.plot(
                hl[:-1, 1],
                hl[:-1, 0],
                ls="none",
                marker="o",
                ms=3.6,
                color="#333",
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
        ax.plot(tr0["lon"][0], tr0["lat"][0], marker="X", ms=10, color="#000", transform=ccrs.PlateCarree(), zorder=9)
    # right-hand column: the outcome shares, their key, and the timing
    rx = 0.12 + 2 * (mw + 0.3) + 0.75
    rw = W - rx - 0.3
    rows = [m for m in TC.MODELS if mstat.get(m) == "ok"] + (["pooled"] if len(s["summary"]) > 1 else [])
    bh = 0.3 * len(rows) + 0.1
    ytop = bot + hb
    ax = fig.add_axes([rx / W, fy(ytop - bh), rw / W, bh / H])
    for i, key in enumerate(rows):
        q = s["pooled"] if key == "pooled" else s["summary"][key]
        left = 0.0
        for o in OUTCOMES:
            w = 100 * q["counts"][o] / q["members"]
            if w <= 0:
                continue
            ax.barh(
                i, w, left=left, color=OUT_COL[o], edgecolor="#777" if o == "absent" else "none", lw=0.6, height=0.66
            )
            if w >= 9:
                ax.text(
                    left + w / 2,
                    i,
                    f"{w:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    color="#fff" if o in ("now", "by") else "#222",
                )
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["both" if r == "pooled" else TC.MODELS[r] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=8.5)
    ax.set_title(f"Recurvature by day {N_DAYS}, % of members", fontsize=10.2, loc="left")
    ax.legend(
        handles=[Patch(fc=OUT_COL[o], ec="#777" if o == "absent" else "none", label=OUT_LAB[o]) for o in OUTCOMES],
        fontsize=8.2,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.28),
        frameon=False,
        borderaxespad=0,
        handlelength=1.2,
        columnspacing=1.0,
    )
    tb = bot + 0.5  # the timing panel: from above its two-line date labels to under the legend
    th = ytop - bh - 1.45 - tb
    ax2 = fig.add_axes([rx / W, fy(tb), rw / W, th / H])
    width = 0.42
    for c, (m, colr) in enumerate((("aifs", "#1f5fa8"), ("ifs", "#c0392b"))):
        if m not in s["summary"]:
            continue
        res = s["res"][m]
        if s["named"]:
            d = np.array([r["rec_h"] / 24 for r in res.values() if r["outcome"] == "by"])
        else:
            d = np.array(
                [
                    g / 24
                    for g in (genesis_h(r["track"]) for r in res.values() if r["track"] is not None)
                    if g is not None
                ]
            )
        h_, _ = np.histogram(d, bins=np.arange(0, N_DAYS + 1))
        ax2.bar(
            np.arange(N_DAYS) + 0.5 + (c - 0.5) * width,
            100 * h_ / TC.NMEM,
            width=width,
            color=colr,
            alpha=0.85,
            label=TC.MODELS[m],
        )
        pt = np.array([r["pt_h"] / 24 for r in res.values() if r["pt_h"] is not None])
        ax2.plot(
            range(N_DAYS + 1), [100 * np.sum(pt <= N) / TC.NMEM for N in range(N_DAYS + 1)], color=colr, lw=1.6, ls="--"
        )
    ax2.set_xlim(0, N_DAYS)
    ax2.set_ylim(0, 100)
    TC.date_axis(ax2, init, N_DAYS, "x", step=2, fmt="stack")
    ax2.tick_params(labelsize=8)
    ax2.set_ylabel("% of members", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    ax2.set_title(
        ("Turn date" if s["named"] else "Reaching 17 m/s")
        + f" (bars) · post-tropical north of {PT_LAT:.0f}°N\n(dashed, cumulative: the tracker's last fix)",
        fontsize=9.6,
        loc="left",
    )
    cax = fig.add_axes([0.45 / W, fy(0.62), 0.34 * (2 * mw) / W, 0.13 / H])  # inset: the end labels fit
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=REC_BOUNDS)
    cb.set_ticklabels([TC.date_lab(init, b, "short") for b in REC_BOUNDS], fontsize=8)
    cb.set_label("member colour = date of the turn (dot)", fontsize=8.6)
    fig.text(
        (0.12 + 0.39 * 2 * mw) / W,
        fy(0.86),
        f"near-black: {OUT_LAB['now']} · grey: {OUT_LAB['no']} · tan: {OUT_LAB['unresolved']} · dashed: followed past the "
        "tracker as a surface low (post-tropical)\nthick dated line: the member nearest the median track · thin black dashes: "
        "the deterministic run (AIFS single, IFS control)\ndark grey: the observed track of the last 48 h · X = start"
        + ("" if s["named"] else " · ◇ where a member first reaches 17 m/s"),
        fontsize=8.2,
        color="#333",
        va="top",
    )
    fig.text(
        0.12 / W,
        fy(0.06),
        "ECMWF tropical-cyclone tracker on AIFS-ENS and IFS-ENS (open data, CC BY 4.0), perturbed members; each followed past "
        "the tracker's last fix on its own ensemble's sea-level pressure (to 72 h, day 10 at most).",
        fontsize=8,
        color="#6f6b64",
        va="bottom",
    )
    fig.savefig(out_png, dpi=90, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


REF_COVER = 0.75  # ref_member: share of the ensemble's 24-h leads a member must carry


def ref_member(tracks):
    """The member drawn thick and dated: the member nearest the ensemble's median track: mean distance from the members'
    median position at every 24-h lead after t=0 where at least half of the members (and 3) carry the storm, among the
    members that carry it at >= REF_COVER of those leads (fewest-gaps fallback). t=0 is left out because all members sit
    at one point there, and the coverage test keeps a member with almost no fixes from winning on a few short
    distances."""
    keys = list(tracks)
    if len(keys) < 3:
        return max(keys, key=lambda k: len(tracks[k]["steps"]))
    hmax = max(int(t["steps"][-1]) for t in tracks.values())
    med = {}
    for h in range(24, hmax + 1, 24):
        ps = [q for q in (TC.at(tracks[k], h) for k in keys) if q is not None]
        if len(ps) >= max(3, 0.5 * len(keys)):
            med[h] = (float(np.median([q[0] for q in ps])), TC.lon_median([q[1] for q in ps]))
    if not med:
        return max(keys, key=lambda k: len(tracks[k]["steps"]))
    score = {}
    for k in keys:
        ds = [TC._km(*a, *med[h]) for h in med for a in [TC.at(tracks[k], h)] if a is not None]
        score[k] = (len(ds) / len(med), float(np.mean(ds)) if ds else np.inf)
    ok = [k for k in keys if score[k][0] >= REF_COVER]
    if ok:
        return min(ok, key=lambda k: score[k][1])
    return min(keys, key=lambda k: (-score[k][0], score[k][1]))


def _lonlab(lo: float) -> str:
    return f"{abs(lo):.0f}°{'E' if lo >= 0 else 'W'}" if abs(lo) < 179.5 else "180°"


def render_depth(s: dict, init, out_png: Path):
    """How deep the storm gets after it reaches the mid-latitudes, both ensembles: the members' followed tracks, where and
    when each is deepest, the depth distribution and a factual summary. Layout in inches."""
    import textwrap

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    et = {(m, k): r for m in s["res"] for k, r in s["res"][m].items() if r["depth"] is not None}
    W = 17.0
    pm = np.array([r["depth"]["pmin"] for r in et.values()])
    n = len(pm)
    dmin = np.array([r["depth"]["h"] for r in et.values()]) / 24.0
    n_merge = sum(1 for r in et.values() if not r["depth"]["on_track"])
    b960, b950 = int((pm < 960).sum()), int((pm < 950).sum())
    per = {m: np.array([r["depth"]["pmin"] for (mm, _), r in et.items() if mm == m]) for m in TC.MODELS}
    lines = [
        f"• {n} members' lows reach {ET_LAT:.0f}°N ("
        + ", ".join(f"{TC.MODELS[m]} {len(v)}" for m, v in per.items() if len(v))
        + f"). {b960} deepen below 960 hPa and {b950} below 950; the median is {np.median(pm):.0f} hPa (10th–90th percentile "
        f"{np.percentile(pm, 10):.0f}–{np.percentile(pm, 90):.0f})"
        + "".join(f"; {TC.MODELS[m]} median {np.median(v):.0f} hPa" for m, v in per.items() if len(v))
        + ".",
        f"• The deepest point comes around day {np.median(dmin):.1f} (10th–90th percentile day {np.percentile(dmin, 10):.1f}–"
        f"{np.percentile(dmin, 90):.1f}), near {np.median([r['depth']['lat'] for r in et.values()]):.0f}°N "
        f"{_lonlab(TC.lon_median([r['depth']['lon'] for r in et.values()]))} (median over members).",
        f"• {n_merge} of {n} members reach their deepest point in a low the storm merges with (◆), not on its own followed "
        "path: during the transition the centre often re-forms along the jet, and ECMWF's tracker stops at that point.",
    ]
    late = int(np.sum(dmin * 24 >= END_H))
    if late:
        lines.append(
            f"• {late} of {n} members are deepest in the last 12 hours before day 10, where the follow stops: their lows may "
            "deepen further."
        )
    T0 = 0.85  # the map's top (inches from the figure's top): the title and the key line above it, the method at the foot
    H = T0 + 4.6 + 0.6 + 0.35 + 0.42 * len(lines) + 0.25 + 0.95
    fig = plt.figure(figsize=(W, H))

    def fx(x):
        return x / W

    def fy(y):
        return 1 - y / H

    fig.suptitle(
        f"How deep {_stormlab(s)} gets after it reaches the mid-latitudes — AIFS-ENS + IFS-ENS, init {init:%d %b %Y %HZ}",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=fy(0.22),
        va="top",
    )
    vmin, vmax = np.floor(np.percentile(pm, 2) / 5) * 5, np.ceil(np.percentile(pm, 98) / 5) * 5
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin, vmax + 5)

    def shown(r):  # a member's path up to a day after its deepest point (decaying lows clutter)
        x = r["ext"]
        keep = x["steps"] <= r["depth"]["h"] + 24
        return {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(x["steps"]) else v) for k, v in x.items()}

    paths = {k: shown(r) for k, r in et.items()}
    allla = np.concatenate([q["lat"] for q in paths.values()] + [np.array([r["depth"]["lat"] for r in et.values()])])
    allo = np.concatenate(
        [q["lon"] % 360 for q in paths.values()] + [np.array([r["depth"]["lon"] % 360 for r in et.values()])]
    )
    alt = ((allo + 180) % 360) - 180  # a storm crossing 0E reads better in -180..180
    if np.ptp(alt) < np.ptp(allo):
        allo = alt
    la0, la1 = max(np.percentile(allla, 0.5) - 5, 22), min(np.percentile(allla, 99.5) + 5, 82)  # the mid-latitude phase
    mid = np.array([q for q in zip(allo, allla) if q[1] >= 22]) if (allla >= 22).sum() > 10 else np.c_[allo, allla]
    lo0, lo1 = np.percentile(mid[:, 0], 0.5) - 8, np.percentile(mid[:, 0], 99.5) + 8
    bh = 4.6  # map height; the fan keeps its true aspect and
    asp = TC.sector_aspect(lo0, lo1, la0, la1)  # the histogram takes the width that is left
    w_ = min(bh / asp, 10.4)
    h_ = w_ * asp
    ax = TC.sector_axes(fig, [fx(0.3), fy(T0 + 4.6 - (bh - h_) / 2), fx(w_), h_ / H], lo0, lo1, la0, la1)
    hx = 0.3 + w_ + 1.0
    hw = W - hx - 0.35
    for key, r in sorted(et.items(), key=lambda kv: -kv[1]["depth"]["pmin"]):
        x = paths[key]
        e = r["depth"]
        fol = x.get("followed", np.zeros(len(x["steps"]), bool))
        col = cmap(norm(e["pmin"]))
        ax.plot(x["lon"][~fol], x["lat"][~fol], color="#9a9a9a", lw=0.5, alpha=0.7, transform=ccrs.Geodetic())
        if fol.any():
            j0 = max(int(np.where(fol)[0][0]) - 1, 0)
            ax.plot(
                x["lon"][j0:], x["lat"][j0:], color=col, lw=1.1, alpha=0.85, ls=(0, (3, 1.5)), transform=ccrs.Geodetic()
            )
        ax.scatter(
            [e["lon"]],
            [e["lat"]],
            s=30 if e["on_track"] else 26,
            color=col,
            edgecolors="#000",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=5,
            marker=("o" if key[0] == "aifs" else "s") if e["on_track"] else "D",
        )
    fig.text(
        0.02,
        fy(T0 - 0.13),
        "Map: grey = the tracker's track · dashed = followed as a surface low (to a day after its deepest point) · "
        "●/■ deepest on that path (AIFS-ENS/IFS-ENS) · ◆ deepest in the low it merged with · colours: depth, as in the "
        "histogram",
        fontsize=9.8,
        color="#333",
        va="bottom",
    )
    ah = fig.add_axes([fx(hx), fy(T0 + 4.6), fx(hw), 4.25 / H])
    bw = 2 if np.ptp(pm) <= 50 else 4  # 2-hPa bins, 4 when the members spread wide
    bins = np.arange(np.floor(pm.min() / bw) * bw, np.ceil(pm.max() / bw) * bw + bw + 0.01, bw)
    _cnt, edges, patches = ah.hist(pm, bins=bins, edgecolor="#fff")
    for pa, lft in zip(patches, edges[:-1]):
        pa.set_facecolor(cmap(norm(lft + bw / 2)))  # the bars are the map's colour key
    ah.set_xlabel(f"deepest central pressure north of {ET_LAT:.0f}°N (hPa)", fontsize=10)
    ah.set_ylabel("members", fontsize=10)
    ah.set_title(
        f"{n} members · median {np.median(pm):.0f} hPa · {b960} below 960 · {b950} below 950", fontsize=10.2, loc="left"
    )
    ah.grid(alpha=0.3)
    yy = T0 + 4.6 + 0.75
    fig.text(0.02, fy(yy - 0.1), "What the members show", fontsize=12.5, fontweight="bold", va="top")
    yy += 0.3
    for ln in lines:
        wrapped = textwrap.fill(ln, 175, subsequent_indent="   ")
        fig.text(0.02, fy(yy), wrapped, fontsize=10.8, va="top", color="#222")
        yy += 0.22 * (wrapped.count("\n") + 1) + 0.18
    fig.text(
        0.02,
        fy(H - 0.12),
        f"Depth = the lowest central pressure, below {TC.FOLLOW_FILLED_HPA:.0f} hPa and north of {ET_LAT:.0f}°N, of the "
        f"storm's own followed low or of any low it merges with (within {SYS_KM:,.0f} km, up to {SYS_H} h after the tracker's "
        "last fix).\n"
        "Tracks: ECMWF tropical-cyclone tracker on AIFS-ENS and IFS-ENS (open data, CC BY 4.0), 50 perturbed members each, "
        "continued as the surface-pressure minimum (closed lows at least 2 hPa below their surroundings; jumps of up to\n"
        "1,800 km in 12 h only along the motion, while the low is not decaying and onto a low below 1000 hPa; never onto a low "
        "that already existed as a separate system unless the storm merged into it; a low of 1000 hPa or more only over the\n"
        "sea north of 30°N; at most 72 h past the tracker's last fix). Pressure: each ensemble's own mean sea-level pressure, "
        "12-hourly, 0.5°.",
        fontsize=8.8,
        color="#6f6b64",
        va="bottom",
    )
    fig.savefig(out_png, dpi=80, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def track_record() -> dict | None:
    """What the verification harness says about the odds on the board, if it has run: its one-line summaries, and per
    chance (recurvature, post_tropical) and sample (pooled, ifs) whether it is skilful (True), shown no skill (False)
    or has no track record yet (None)."""
    if not REF.exists():
        return None
    v = json.loads(REF.read_text())
    skill = {
        k: {m: (v[k][m].get("skilful") if v[k][m].get("track_record") else None) for m in ("pooled", "ifs")}
        for k in ("recurvature", "post_tropical")
    }
    return dict(built=v.get("built"), period=v.get("period"), lines=v.get("summary", {}), skill=skill)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="assets/sst")
    ap.add_argument("--no-msl", action="store_true", help="no follow past the tracker and no depth (no msl download)")
    ap.add_argument("--members", help="also write the board storms' member tracks (for tc_maps.py) to this file")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    t0 = time.time()
    hh = f"{int(a.time):02d}"
    init = pd.Timestamp(f"{a.date}T{hh}:00")
    out = Path(a.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    per, det, mstat, run_h = {}, {}, {}, {}
    for m in TC.MODELS:
        fp = TC.tracker_file(m, a.date, hh, CACHE)
        try:
            if fp is not None:
                per[m] = nh_storms(TC.decode_all(fp, types=(4,)))
                det[m] = TC.decode_all(fp, types=(0,))
                run_h[m] = TC.run_hours(fp)
        except Exception as e:  # noqa: BLE001  one unreadable file leaves the other ensemble
            TC.report(f"{TC.MODELS[m]} tracker file {a.date} {hh}Z unreadable ({e!r})", "warning")
            per.pop(m, None)
            det.pop(m, None)
            if fp is not None:
                fp.unlink(missing_ok=True)
        mstat[m] = "ok" if m in per else "missing"
    hist = history(a.date, hh) if per else {}
    storms = assemble(per)
    msl, land = {}, None
    if storms and not a.no_msl:
        land = TC.load_land(a.date, hh)  # the follower's land rule (heat / lee lows)
        for m in per:
            try:
                msl[m] = load_msl(m, a.date, hh)
            except Exception as e:  # noqa: BLE001  the tracks stand without the follow-on
                TC.report(
                    f"{TC.MODELS[m]} sea-level pressure unavailable ({e!r}); its members end at the tracker's last fix",
                    "warning",
                )
    for s in storms:
        s["hist"] = hist.get(s["id"]) if s["named"] else None
        s["res"] = {m: analyse(s["models"].get(m), s["hist"], run_h[m], msl.get(m), land) for m in per}
        s["summary"] = {m: summarise(list(r.values())) for m, r in s["res"].items()}
        s["pooled"] = summarise([r for m in s["res"] for r in s["res"][m].values()])
        s["det"] = {}
        for m in per:
            tr = match_det(s, det[m])
            s["det"][m] = det_summary(tr, s["hist"], run_h[m])
        q = s["pooled"]
        s["basin_watch"] = watch_basin(s, q["median_member_turn_h"])
        s["watch"] = bool(
            s["basin_watch"] in ("WP", "EP", "NA")
            and q["p_recurve_90"] is not None
            and q["p_recurve_90"][0] >= WATCH_P
            and q["median_member_turn_h"] is not None
            and q["median_member_turn_h"] <= 24 * WATCH_DAY
        )
    storms.sort(key=lambda s: (not s["named"], -(s["pooled"]["p_recurve"] or 0)))
    status = "missing" if not per else ("ok" if storms else "none")
    print(
        "  storms: "
        + (
            ", ".join(
                f"{_stormlab(s)} ("
                + ", ".join(f"{TC.MODELS[m]} {s['summary'][m]['carried']}" for m in s["summary"])
                + f"; recurve {_pct(s['pooled']['p_recurve'])}{', WATCH' if s['watch'] else ''})"
                for s in storms
            )
            or "none"
        ),
        flush=True,
    )
    tcd = out / "tc"
    tcd.mkdir(exist_ok=True)
    for f in list(tcd.glob("tracks_*.webp")) + list(tcd.glob("depth_*.webp")):
        f.unlink()  # departed storms' figures go (the folder's phase_* files are tc_phase.py's)
    for s in storms[:MAX_PER_STORM]:
        sl = slug(s["key"])
        try:
            render_tracks(s, init, tcd / f"tracks_{sl}.webp", mstat)
            s["tracks_file"] = f"tc/tracks_{sl}.webp"
        except Exception as e:  # noqa: BLE001  one storm's figure must not stop the board
            TC.report(f"tracks figure for {_stormlab(s)} failed ({e!r})", "warning")
        if s["pooled"]["depth_members"] >= MIN_DEPTH_MEMBERS:
            try:
                render_depth(s, init, tcd / f"depth_{sl}.webp")
                s["depth_file"] = f"tc/depth_{sl}.webp"
            except Exception as e:  # noqa: BLE001
                TC.report(f"depth figure for {_stormlab(s)} failed ({e!r})", "warning")
    lines, on = watch_lines(storms, mstat, init) if per else ([f"Recurvature watch: {UNAVAILABLE.lower()}."], [])
    js = {
        "init": f"{init:%Y-%m-%dT%H:%MZ}",
        "built": f"{pd.Timestamp.now('UTC'):%Y-%m-%dT%H:%MZ}",
        "status": status,
        # complete: both tracker files read and, with storms, both ensembles' pressure for the follow-on; tc.yml rebuilds
        # an incomplete cycle at its later slots
        "complete": len(per) == len(TC.MODELS) and (not storms or a.no_msl or len(msl) == len(per)),
        "models": {m: {"members": TC.NMEM, "status": mstat[m], "deterministic": TC.DET[m]} for m in TC.MODELS},
        "msl": {m: ("ok" if m in msl else "missing") for m in per},
        "rules": {
            "support": f"observed at t=0, or carried to tropical-storm strength by at least {_pct(SUPPORT_FRAC)} of the members of either ensemble",
            "outcomes": {o: OUT_LAB[o] for o in OUTCOMES},
            "now_h": NOW_H,
            "horizon_days": N_DAYS,
            "post_tropical": f"the tracker's last fix north of {PT_LAT:.0f}°N, by day {N_DAYS}",
            "watch": f"North Pacific or North Atlantic; pooled chance of recurving by day {WATCH_DAY} with a lower 90 % bound of at least {_pct(WATCH_P)}, median member's turn by day {WATCH_DAY}",
            "interval": "Wilson 90 %",
        },
        "watch": {
            "level": "missing" if status == "missing" else ("watch" if on else "none"),
            "storms": on,
            "lines": lines,
        },
        "track_record": track_record(),
        "storms": [],
    }
    for s in storms:
        e = {
            "key": s["key"],
            "slug": slug(s["key"]),
            "id": s["id"],
            "name": _stormlab(s),
            "named": s["named"],
            "basin": s["basin_watch"],
            "basin_label": BASIN_LAB.get(s["basin_watch"]),
            "obs": None if s["obs"] is None else [round(s["obs"][0], 2), round(s["obs"][1], 2)],
            "history": [[h, round(la, 2), round(lo, 2)] for h, la, lo in s["hist"] or []],
            "watch": s["watch"],
            "pooled": s["pooled"],
            "models": s["summary"],
            "deterministic": {TC.DET[m]: v for m, v in s["det"].items() if v is not None},
        }
        for k in ("tracks_file", "depth_file"):
            if k in s:
                e[k] = s[k]
        if not s["named"]:
            e["genesis_h_p10_p50_p90"] = _q(
                [
                    g
                    for m in s["res"]
                    for r in s["res"][m].values()
                    if r["track"] is not None
                    for g in [genesis_h(r["track"])]
                    if g is not None
                ]
            )
        js["storms"].append(e)
    (out / "data" / "tc_board.json").write_text(json.dumps(js, separators=(",", ":")))
    if (
        a.members
    ):  # each member's tracker track and its followed continuation, for the maps (a work file, not published)
        mem = {
            slug(s["key"]): {
                m: [
                    dict(
                        h=[int(h) for h in r["ext"]["steps"]],
                        lat=[round(float(v), 2) for v in r["ext"]["lat"]],
                        lon=[round(float(v), 2) for v in r["ext"]["lon"]],
                        followed=[bool(v) for v in r["ext"].get("followed", np.zeros(len(r["ext"]["steps"]), bool))],
                    )
                    for r in s["res"][m].values()
                    if r["ext"] is not None
                ]
                for m in s["res"]
            }
            for s in storms
        }
        Path(a.members).write_text(json.dumps({"init": js["init"], "storms": mem}, separators=(",", ":")))
    if status == "missing":
        TC.report(f"TC board {a.date} {hh}Z: {UNAVAILABLE.lower()} (both tracker files missing)", "error")
    print("  " + " | ".join(lines), flush=True)
    print(f"  wrote data/tc_board.json ({status}, {len(storms)} storms) in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
