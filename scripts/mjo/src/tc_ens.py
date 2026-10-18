#!/usr/bin/env python3
"""Tropical cyclones, ENSEMBLE view - diagnostic only: what the AIFS-ENS and IFS-ENS members say about each storm's
recurvature, and how deep each storm's low gets once it reaches the mid-latitudes.

Tracks: ECMWF's tropical-cyclone tracker output in open data (Google mirror only), one BUFR per model and cycle:
  aifs-ens  .../aifs-ens/0p25/enfo/{d}{hh}0000-360h-enfo-tf.bufr   control (type 1, number 51) + 50 perturbed (type 4)
  ifs-ens   .../ifs/0p25/enfo/{d}{hh}0000-360h-enfo-tf.bufr        50 perturbed (type 4)
  Subsets of type 0 are the deterministic runs (AIFS-single / IFS HRES: identical to their own oper tf files) and are
  excluded.
Recurvature per member: tc_jet.recurvature (24-h zonal motion from <= -1 to >= +2 m/s).
Depth after transition (AIFS-ENS): each member's low followed past the tracker (tc_jet.follow_low) on 12-hourly msl to
day 10; depth = the lowest pressure north of ET_LAT of the storm or of a low it merges with (et_depth).
Ensemble support (supported): the AIFS-ENS control's storms drawn on the TC-jet card and in the phase space are those
observed at t=0 or carried by at least SUPPORT_FRAC of the members of either ensemble.

The products describe the storms only, not what a storm does to the flow.
Outputs record a status: "ok", "none" (the tracker files were read and hold no Northern-Hemisphere storm) or "missing"
(a tracker file could not be had or read; the figures then say that the track data are unavailable).
    python src/tc_ens.py --date 20261002 --time 00 --out-dir ../../assets/sst
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import tc_jet as TC

URLS = {
    "aifs": TC.URL,
    "ifs": "https://storage.googleapis.com/ecmwf-open-data/{d}/{h:02d}z/ifs/0p25/enfo/{d}{h:02d}0000-360h-enfo-tf.bufr",
}
MLAB = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
NMEM = {"aifs": 51, "ifs": 50}  # members in each tracker file (AIFS-ENS: control + 50; IFS-ENS: 50, no control)
MAX_PER_STORM = 10  # per-storm figures for the page's storm picker (assets/sst/tc/)
SUPPORT_FRAC = 0.4  # share of an ensemble's members that must develop a system (reach tropical-storm strength) for it
# to be listed as a possible new storm, and for a storm the control alone develops to be drawn on
# the TC-jet card and in the phase space (one ensemble is enough)


def slug(key):
    """File-safe storm key ('aifs:71W' -> 'aifs-71W')."""
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(key))


MISSING = -1e99
CACHE = Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc"


def fetch(model: str, date: str, hour: int) -> Path | None:
    """One model's tracker BUFR for one cycle (cached; tc_jet.get_bufr retries); None if it cannot be had."""
    return TC.get_bufr(
        URLS[model].format(d=date, h=hour),
        CACHE / f"{model}_ens_tf_{date}{hour:02d}.bufr",
        f"{MLAB[model]} tracker file {date} {hour:02d}Z",
    )


def decode_all(fp: Path, types=(1, 4)) -> list[dict]:
    """Every storm message with all ensemble members (types 1 and 4): {id, name, named, tracks: {member: track}}.
    The control is member 0 here (51 in the file); track = dict(steps, lat, lon(-180..180), pmsl hPa, wind m/s).
    types=(0,): the deterministic run instead (AIFS single / IFS HRES oper tf files, or the type-0 subset), as member 0
    (used by scripts/tcshear)."""
    import eccodes as ec

    out = []
    with open(fp, "rb") as f:
        while True:
            h = ec.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                ec.codes_set(h, "unpack", 1)
                sid = ec.codes_get(h, "#1#stormIdentifier").strip()
                name = ec.codes_get(h, "#1#longStormName").strip()
                ft = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleForecastType"))
                mem = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleMemberNumber"))
                n = len(ft)

                def arr(key, h=h, n=n):
                    a = np.atleast_1d(ec.codes_get_array(h, key)).astype(float)
                    return np.repeat(a, n) if a.size == 1 else a

                rep = ec.codes_get(h, "#1#delayedDescriptorReplicationFactor")
                cols = []  # (step, lat, lon, p, w) arrays over subsets
                a_la, a_lo = arr("#2#latitude"), arr("#2#longitude")
                o_la, o_lo = arr("#1#latitude"), arr("#1#longitude")
                a_la = np.where(a_la > MISSING, a_la, o_la)
                a_lo = np.where(a_lo > MISSING, a_lo, o_lo)
                cols.append(
                    (np.zeros(n), a_la, a_lo, arr("#1#pressureReducedToMeanSeaLevel"), arr("#1#windSpeedAt10M"))
                )
                cols.extend(
                    (
                        arr(f"#{i}#timePeriod"),
                        arr(f"#{2 * i + 2}#latitude"),
                        arr(f"#{2 * i + 2}#longitude"),
                        arr(f"#{i + 1}#pressureReducedToMeanSeaLevel"),
                        arr(f"#{i + 1}#windSpeedAt10M"),
                    )
                    for i in range(1, rep + 1)
                )
                tracks = {}
                for j in range(n):
                    if ft[j] not in types:
                        continue  # type 0 = the deterministic run
                    rows = [
                        (c[0][j], c[1][j], c[2][j], c[3][j], c[4][j])
                        for c in cols
                        if c[1][j] > MISSING and c[0][j] > MISSING
                    ]
                    if not rows:
                        continue
                    st, la, lo, p, w = (np.array(x, float) for x in zip(*rows))
                    m = 0 if ft[j] in (0, 1) else int(mem[j])
                    tracks[m] = dict(
                        steps=st.astype(int),
                        lat=la,
                        lon=(lo + 180) % 360 - 180,
                        pmsl=np.where(p > MISSING, p / 100.0, np.nan),
                        wind=np.where(w > MISSING, w, np.nan),
                    )
                named = not (sid[:2].isdigit() and int(sid[:2]) >= 70)
                if tracks:
                    out.append(dict(id=sid, name=name, named=named, tracks=tracks))
            finally:
                ec.codes_release(h)
    return out


def _dup(tr, other) -> bool:
    for hh in np.intersect1d(tr["steps"], other["steps"]):
        a = TC.at(tr, int(hh))
        b = TC.at(other, int(hh))
        if TC._km(a[0], a[1], b[0], b[1]) < 150:
            return True
    return False


GEN_KM, GEN_H = 600.0, 48  # genesis points this close in space and time are one developing system
TS_WIND = 17.0  # m/s: tropical-storm strength
MIN_COMPANION = 3  # a group under min_frac is kept only as the OTHER model's view of a system that
# passes there ("IFS-ENS: 14 of 50 members", not "not in IFS-ENS"); never on its own


MATCH_H = 72  # hours of shared median track compared when pairing the two models' developing systems


def median_track(s: dict) -> dict:
    """Lead hour -> median member (lat, lon) where at least a quarter of the members (and 3, or every track of a system with
    fewer, such as the control alone) have a position. Longitudes are unwrapped about the first member before the median,
    so a system on the dateline is not averaged to the far side."""
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


def lon_median(lons) -> float:
    """Median longitude (-180..180) that is safe across the dateline and 0 deg: the median offset from the circular mean.
    (A plain np.median of longitudes straddling the dateline can sit far from the cluster: 168.2E for a genesis cluster
    whose circular centre is 177.6E.)"""
    lo = np.asarray(lons, float)
    a = np.deg2rad(lo)
    c = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    return float((c + np.median((lo - c + 180) % 360 - 180) + 180) % 360 - 180)


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


def nh_storms(msgs: list[dict], min_frac: float = SUPPORT_FRAC, nmem: int = 51) -> list[dict]:
    """Named NH storms, plus developing systems reached by >= min_frac of DISTINCT members at tropical-storm strength.

    ECMWF's tracker splits one developing system across several provisional ids (e.g. a Bay of Campeche system in one
    00Z AIFS-ENS run sat under 70L, 71L, 72L, ... with 9 / 13 / 7 ... tropical-storm members each, so every id failed
    the 40 % test on its own while 31 of 51 distinct members developed it). Genesis is therefore clustered across ids:
    each member track that reaches tropical-storm strength contributes its genesis point (where and when it first does),
    points within GEN_KM and GEN_H are linked, and a cluster counts its distinct members (each member's strongest track
    is kept once). A genesis track that duplicates the same member's track of a named storm (within 150 km at a shared
    step) is dropped first."""
    named = [m for m in msgs if m["named"]]
    keep = []
    for m in named:
        tr = {k: t for k, t in m["tracks"].items() if (t["lat"] > 0).any() and len(t["steps"]) > 1}
        if tr:
            keep.append(dict(m, tracks=tr, ctl_ids=[m["id"]] if 0 in tr else []))
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
        glo = lon_median([v[3] for v in best.values()])
        # the control's own tracker id(s) in this cluster: the control track tc_phase / the TC-jet card carry under its id
        # can sit in a cluster named after another id (e.g. control 72L as member 0 of cluster 73L)
        ctl_ids = sorted({sid for sid, k, _, _, _, _ in g if k == 0})
        keep.append(
            dict(
                id=main_id,
                name=main_id,
                named=False,
                weak=weak,
                ids=sorted(ids),
                ctl_ids=ctl_ids,
                region=region_of(gla, glo),
                basin=basin_of(main_id[:3], gla, glo),
                genesis=(round(gla, 1), round(glo, 1)),
                tracks={k: v[1] for k, v in best.items()},
            )
        )
    return keep


def stats(s: dict, nmem: int) -> dict:
    """Per model and storm: recurvature lead per member, P(recurve), lead-time spread, P(track ended by day N)."""
    rec = {k: TC.recurvature(t) for k, t in s["tracks"].items()}
    ended = {k: int(t["steps"][-1]) for k, t in s["tracks"].items()}
    lat_end = {k: float(t["lat"][-1]) for k, t in s["tracks"].items()}
    days = np.array([r / 24 for r in rec.values() if r is not None])
    return dict(
        n_tracked=len(s["tracks"]),
        n_members=nmem,
        recurve=rec,
        ended=ended,
        lat_end=lat_end,
        p_recurve=round(float(np.sum([r is not None for r in rec.values()])) / nmem, 3),
        rec_day_q=[round(float(x), 2) for x in np.percentile(days, [10, 50, 90])] if len(days) else None,
    )


# ── the ensembles' systems, and which of the control's storms they support ────────────────────────────────────────────
def ensemble_systems(date: str, hour: int) -> tuple[dict, dict]:
    """Each ensemble's Northern-Hemisphere systems for one cycle (nh_storms) and the status of its tracker file:
    ({model: systems}, {model: "ok" | "missing"}). A model whose file cannot be had or read is "missing" and absent from
    the first dict (an unreadable cached file is deleted, so the next run fetches it again)."""
    per, status = {}, {}
    for m in MLAB:
        fp = None
        try:
            fp = fetch(m, date, hour)
            if fp is not None:
                per[m] = nh_storms(decode_all(fp), nmem=NMEM[m])
        except Exception as e:
            TC.report(f"{MLAB[m]} tracker file {date} {hour:02d}Z unreadable ({e!r})", "warning")
            if fp is not None:
                fp.unlink(missing_ok=True)
        status[m] = "ok" if m in per else "missing"
    return per, status


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


def match_control(s: dict, systems: list[dict]) -> dict | None:
    """The system in one ensemble's list (nh_storms) that is the AIFS-ENS control's storm s (a tc_jet.decode track), or
    None. An observed storm: the system under its official id. A developing one: the AIFS-ENS cluster that holds the
    control's own track (ctl_ids), else the closest developing system of the same basin by match_gap (below GEN_KM), the
    control's track taken as a one-member system."""
    if s["named"]:
        return next((x for x in systems if x["named"] and x["id"] == s["id"]), None)
    gen = [x for x in systems if not x["named"]]
    own = next((x for x in gen if s["id"] in x.get("ctl_ids", ())), None)
    if own is not None:
        return own
    g = genesis_h(s)
    if g is None:
        return None
    k = int(np.where(s["steps"] == g)[0][0])
    one = dict(tracks={0: s}, genesis=(float(s["lat"][k]), float(s["lon"][k])))
    bas = basin_of(s["id"], *one["genesis"])
    best, bd = None, GEN_KM
    for x in gen:
        if x["basin"] == bas:
            d = match_gap(one, x)
            if d < bd:
                best, bd = x, d
    return best


def control_support(storms: list[dict], per: dict, frac: float = SUPPORT_FRAC) -> tuple[list[dict], list[dict]]:
    """Split the AIFS-ENS control's storms (tc_jet.nh_storms) into those drawn and those left out: (kept, dropped).

    Kept: a storm observed at t=0 (tc_jet.decode "observed"), or one that at least `frac` of the members of AIFS-ENS or of
    IFS-ENS carry, i.e. develop to tropical-storm strength in the matched system (match_control; for IFS-ENS, which has
    no control, the IFS-ENS system pair_genesis pairs with the control's AIFS-ENS system comes first). Every storm gets
    support = {model: share of members, None where that model's tracker file is missing}."""
    pair = pair_genesis(per)  # id(IFS-ENS system) -> AIFS-ENS system
    partner = {id(pair[id(si)]): si for si in per.get("ifs", []) if id(si) in pair}
    kept, dropped = [], []
    for s in storms:
        sup, mine = {}, None
        for m in MLAB:
            if m not in per:
                sup[m] = None
                continue
            x = partner.get(id(mine)) if (m == "ifs" and mine is not None) else None
            if x is None:
                x = match_control(s, per[m])
            if m == "aifs":
                mine = x
            sup[m] = round(len(x["tracks"]) / NMEM[m], 3) if x is not None else 0.0
        s_sup = dict(s, support=sup)
        ok = s_sup.get("observed", s_sup["named"]) or any(v is not None and v >= frac for v in sup.values())
        (kept if ok else dropped).append(s_sup)
    return kept, dropped


def _share(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f} %"


def supported(storms: list[dict], date: str, hour: int) -> tuple[list[dict], list[dict], dict]:
    """The control's storms the ensembles support, for the TC-jet card (dt_charts) and the phase space (tc_phase):
    (kept, dropped, {model: "ok" | "missing"}). The storms left out are logged with their support."""
    per, status = ensemble_systems(date, hour)
    kept, dropped = control_support(storms, per)
    for s in dropped:
        print(
            f"  left out {TC.label(s)}: not observed at t=0; carried by "
            + ", ".join(f"{MLAB[m]} {_share(v)}" for m, v in s["support"].items())
            + f" of members (needs {100 * SUPPORT_FRAC:.0f} % in one ensemble)",
            flush=True,
        )
    return kept, dropped, status


# ── step 3: how deep the storm gets after it reaches the mid-latitudes ────────────────────────────────────────────────
# A recurving storm that phases with a mid-latitude upper trough can become an exceptionally deep extratropical low, often
# just after ECMWF's tracker stops following it. Each member's low is followed past the tracker's last fix
# (tc_jet.follow_low), and its lowest central pressure north of ET_LAT is the storm's depth.
ET_LAT = 35.0
ET_H_MAX = 240
MIN_ET_MEMBERS = 20


def load_msl(date: str, hh: str, h_max=ET_H_MAX):
    """{member: (steps, lat, lon) hPa} at 12-h steps to h_max on a 0.5-degree grid, 10-82N; member 0 = control."""
    import xarray as xr
    import store as ecmwf

    cyc = ecmwf.Cycle(date, hh)
    S = tuple(range(0, h_max + 1, 12))
    out = {}
    lat = lon = None
    for t in ("cf", "pf"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", t, "msl", "sfc", (), S))
        da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["msl"]
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude").sortby("latitude")
        da = da.sel(latitude=slice(10.0, 82.0)).coarsen(latitude=2, longitude=2, boundary="trim").mean() / 100.0
        if "number" in da.dims:
            for k, num in enumerate(da.number.values):
                out[int(num)] = da.isel(number=k).values.astype("float32")
        else:
            out[0] = da.values.astype("float32")
        lat, lon = da.latitude.values, da.longitude.values
        steps = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return out, lat, lon, steps


SYS_KM = 1200.0  # the "system": any closed low this close to the storm's latest position ...
SYS_H = 72  # ... from the tracker's last fix to this many hours later
EXTRAP_H = 48  # once the track ends, the search centre moves on with its last motion for up to this long


def et_depth(s: dict, M: dict, lat, lon, steps, land=None) -> dict:
    """Per member: the track extended past the tracker (tc_jet.follow_low) and the depth of the SYSTEM it becomes.

    During phasing the low is often double-centred and ECMWF's tracker can stay with a decaying southern remnant while the
    main low re-forms 1000+ km to the north (Choi-Wan, 2 Oct 00Z: 20 of 21 "shallow" members had a 935-955 hPa low next to
    their followed path). Depth is therefore the lowest central pressure, north of ET_LAT, of the storm's own track and of
    any closed low within SYS_KM of its latest position over SYS_H hours from the tracker's last fix: the remnant together
    with whatever low it merges into. A weak low (>= tc_jet.FOLLOW_WEAK_HPA) over land is not part of it (heat / lee lows;
    land: tc_jet.LandMask, None = everywhere). {member: dict(ext, pmin, h, lat, lon, on_track, n_followed)}."""
    res = {}
    stl = list(np.asarray(steps, int))
    for k, tr in s["tracks"].items():
        if k not in M:
            continue
        cont = TC.follow_low(M[k], lat, lon, steps, tr, h_max=ET_H_MAX, land=land)
        ext = TC.extended(tr, cont)
        cands = []  # (p, h, lat, lon, on_track)
        m = (ext["lat"] >= ET_LAT) & (ext["steps"] <= ET_H_MAX) & np.isfinite(ext["pmsl"])
        cands.extend(
            (float(ext["pmsl"][i]), int(ext["steps"][i]), float(ext["lat"][i]), float(ext["lon"][i]), True)
            for i in np.where(m)[0]
        )
        on = [int(h) for h in tr["steps"] if int(h) in set(stl)]
        if on:
            h_beg, h_end = on[-1], int(tr["steps"][-1])  # SYS_H after the LAST fix (often +6 h off-grid)
            for h in stl:
                if h < h_beg or h > min(h_end + SYS_H, ET_H_MAX):
                    continue
                before = np.where(ext["steps"] <= h)[0]
                c_la, c_lo = float(ext["lat"][before[-1]]), float(ext["lon"][before[-1]])
                gap = h - int(ext["steps"][before[-1]])
                if gap > 0 and len(before) >= 2:  # the track has ended: move the search centre
                    j1, j0 = before[-1], before[-2]  # on with the last motion, for up to EXTRAP_H
                    dt = (int(ext["steps"][j1]) - int(ext["steps"][j0])) or 6
                    f_ = min(gap, EXTRAP_H) / dt
                    c_la = float(np.clip(c_la + f_ * (ext["lat"][j1] - ext["lat"][j0]), -80, 80))
                    c_lo = float(
                        ((c_lo + f_ * (((ext["lon"][j1] - ext["lon"][j0] + 180) % 360) - 180) + 180) % 360) - 180
                    )
                for la, lo, p, _ in TC.closed_lows(M[k][stl.index(h)], lat, lon, (c_la, c_lo), SYS_KM):
                    if la >= ET_LAT and not (p >= TC.FOLLOW_WEAK_HPA and (land is None or land(la, lo))):
                        cands.append((float(p), int(h), float(la), float(lo), False))
        if not cands:
            continue
        p, h, la, lo, ontr = min(cands, key=lambda c: c[0])
        res[k] = dict(ext=ext, pmin=round(p, 1), h=h, lat=la, lon=lo, on_track=ontr, n_followed=len(cont["steps"]))
    return res


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _stormlab(s):
    if s["named"]:
        return s["name"].title()
    tag = s.get("tag", "") + s["id"]  # "IFS-ENS 71W": the two trackers number systems independently
    return f"possible new storm, {s['region']} ({tag})" if s.get("region") else f"possible new storm ({tag})"


def genesis_h(t):
    """Hour a member track first reaches tropical-storm strength, or None."""
    w = np.nan_to_num(t["wind"], nan=0.0)
    return int(t["steps"][int(np.argmax(w >= TS_WIND))]) if (w >= TS_WIND).any() else None


REF_COVER = 0.75  # ref_member: share of the ensemble's 24-h leads a member must carry


def ref_member(tracks):
    """The member drawn thick and dated: the AIFS control where there is one, else the member nearest the ensemble's
    median track (IFS-ENS open data carries no control track): mean distance from the members' median position at every
    24-h lead after t=0 where at least half of the members (and 3) carry the storm, among the members that carry it at
    >= REF_COVER of those leads (fewest-gaps fallback). t=0 is left out because all members sit at one point there, and
    the coverage test keeps a member with almost no fixes from winning on a few short distances."""
    if 0 in tracks:
        return 0
    keys = list(tracks)
    if len(keys) < 3:
        return max(keys, key=lambda k: len(tracks[k]["steps"]))
    hmax = max(int(t["steps"][-1]) for t in tracks.values())
    med = {}
    for h in range(24, hmax + 1, 24):
        ps = [q for q in (TC.at(tracks[k], h) for k in keys) if q is not None]
        if len(ps) >= max(3, 0.5 * len(keys)):
            med[h] = (float(np.median([q[0] for q in ps])), lon_median([q[1] for q in ps]))
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


UNAVAILABLE = "Track data unavailable for this cycle"


def render_tracks(storms: list[dict], st: dict, init, out_png: Path, mstat: dict | None = None):
    """Track ensembles, one row per storm (up to four): each model's members coloured by recurvature date, and the
    recurvature (or, for a developing system, genesis) timing. mstat = {model: "ok" | "missing"}: a model whose tracker
    file is missing is said to be unavailable, never "not in" it."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    rows = storms[:4]
    nrow = max(1, len(rows))
    H = 4.6 * nrow + 1.7
    fig = plt.figure(figsize=(16.5, H))
    fig.suptitle(
        f"Tropical-cyclone track ensembles — AIFS-ENS (51) and IFS-ENS (50), init {init:%d %b %Y %HZ}",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=1 - 0.25 / H,
    )
    top, bot = 1 - 0.85 / H, 1.1 / H  # figure fractions for the storm rows; the top margin clears the title
    rh = (top - bot) / nrow
    cmap = ListedColormap(["#3b0f70", "#6a2c91", "#9a3f8f", "#c9517f", "#ec6a5b", "#f9944a", "#fbc04b", "#f2e86d"])
    bounds = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    norm = BoundaryNorm(bounds, cmap.N)
    mstat = mstat or dict.fromkeys(MLAB, "ok")
    gone = [m for m in MLAB if mstat.get(m) != "ok"]
    if not rows:
        if len(gone) == len(MLAB):
            msg = UNAVAILABLE
        elif gone:
            msg = (
                f"No tropical storms in the Northern Hemisphere in {' or '.join(MLAB[m] for m in MLAB if m not in gone)} "
                f"({', '.join(MLAB[m] for m in gone)}: {UNAVAILABLE[0].lower() + UNAVAILABLE[1:]})"
            )
        else:
            msg = "No tropical storms in the Northern Hemisphere in either ensemble"
        fig.text(0.5, 0.5, msg, ha="center", fontsize=14, color="#6f6b64")
    for i, s in enumerate(rows):
        allt = [t for m in ("aifs", "ifs") for t in s["models"].get(m, {"tracks": {}})["tracks"].values()]
        la = np.concatenate([t["lat"] for t in allt])
        lo = np.concatenate([t["lon"] for t in allt])
        clon = lon_median(lo)  # safe across the dateline and 0 deg (a plain median of lon % 360 is not)
        rel = (lo - clon + 180) % 360 - 180
        ext = [rel.min() - 4, rel.max() + 4, max(-5, la.min() - 4), min(75, la.max() + 5)]  # in the map's own frame
        fig.text(
            0.035, top - i * rh + 0.3 / H, _stormlab(s), fontsize=12.5, fontweight="bold", va="top"
        )  # once per row
        for c, m in enumerate(("aifs", "ifs")):
            ax = fig.add_axes(
                [0.035 + c * 0.33, top - (i + 1) * rh + 0.04 * rh, 0.30, 0.86 * rh],
                projection=ccrs.PlateCarree(central_longitude=clon),
            )
            ax.set_extent(ext, crs=ccrs.PlateCarree(central_longitude=clon))
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efece4")
            ax.coastlines("50m", lw=0.6, color="#555")
            ax.gridlines(
                draw_labels=dict(bottom="x", left="y"), lw=0.3, color="#999", alpha=0.6, x_inline=False, y_inline=False
            )
            S = s["models"].get(m)
            if not S:
                ax.text(
                    0.5,
                    0.5,
                    f"{MLAB[m]}: {UNAVAILABLE[0].lower() + UNAVAILABLE[1:]}" if m in gone else f"not in {MLAB[m]}",
                    transform=ax.transAxes,
                    ha="center",
                    color="#6f6b64",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff", ec="none", alpha=0.9),
                )
                continue
            rec = st[s["key"]][m]["recurve"]
            kref = ref_member(S["tracks"])
            for k, t in S["tracks"].items():
                r = rec.get(k)
                col = "#9a9a9a" if r is None else cmap(norm(r / 24))
                ax.plot(
                    t["lon"],
                    t["lat"],
                    color=col,
                    lw=0.9,
                    alpha=0.75,
                    transform=ccrs.Geodetic(),
                    zorder=4 if r is not None else 3,
                )
                dd = [TC.at(t, h) for h in range(24, int(t["steps"][-1]) + 1, 24)]  # a dot every 24 h: the timeline
                dd = np.array([q for q in dd if q is not None])
                if len(dd):
                    ax.plot(
                        dd[:, 1],
                        dd[:, 0],
                        ls="none",
                        marker="o",
                        ms=1.9,
                        color=col,
                        alpha=0.8,
                        transform=ccrs.PlateCarree(),
                        zorder=4,
                    )
                if r is not None:
                    pos = TC.at(t, r)
                    ax.plot(
                        pos[1],
                        pos[0],
                        marker="o",
                        ms=3.2,
                        color=col,
                        mec="#000",
                        mew=0.3,
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                if not s["named"]:
                    g = genesis_h(t)
                    if g is not None:
                        pg = TC.at(t, g)
                        ax.plot(
                            pg[1],
                            pg[0],
                            marker="D",
                            ms=3.6,
                            mfc="#fff",
                            mec="#222",
                            mew=0.6,
                            transform=ccrs.PlateCarree(),
                            zorder=6,
                        )
            tr0 = S["tracks"][kref]  # the dated reference line
            r0 = rec.get(kref)
            ax.plot(tr0["lon"], tr0["lat"], color="#111", lw=3.4, transform=ccrs.Geodetic(), zorder=7)
            ax.plot(
                tr0["lon"],
                tr0["lat"],
                color="#9a9a9a" if r0 is None else cmap(norm(r0 / 24)),
                lw=2.0,
                transform=ccrs.Geodetic(),
                zorder=7,
            )
            import matplotlib.patheffects as pe

            last = []
            for kd in range(1, int(tr0["steps"][-1] // 24) + 1):
                q = TC.at(tr0, 24 * kd)
                if q is None:
                    continue
                ax.plot(
                    q[1],
                    q[0],
                    marker="o",
                    ms=5,
                    mfc="#fff",
                    mec="#111",
                    mew=1.0,
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                )
                xy = ax.projection.transform_point(q[1], q[0], ccrs.PlateCarree())
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
            ax.plot(
                tr0["lon"][0], tr0["lat"][0], marker="X", ms=11, color="#000", transform=ccrs.PlateCarree(), zorder=9
            )
            q = st[s["key"]][m]
            ax.set_title(
                f"{MLAB[m]}: {q['n_tracked']} of {q['n_members']} members tracked, {100 * q['p_recurve']:.0f}% recurve",
                fontsize=10.5,
                loc="left",
            )
        ax = fig.add_axes([0.71, top - (i + 1) * rh + 0.16 * rh, 0.27, 0.70 * rh])
        width = 0.42
        for c, (m, colr) in enumerate((("aifs", "#1f5fa8"), ("ifs", "#c0392b"))):
            if m not in s["models"]:
                continue
            q = st[s["key"]][m]
            if s["named"]:
                d = np.array([r / 24 for r in q["recurve"].values() if r is not None])
                lab = f"{MLAB[m]} recurves ({100 * q['p_recurve']:.0f}%)"
            else:  # a developing system: WHEN members reach tropical-storm strength
                d = np.array(
                    [g / 24 for g in (genesis_h(t) for t in s["models"][m]["tracks"].values()) if g is not None]
                )
                lab = f"{MLAB[m]}: {len(d)} of {q['n_members']} members"
                if len(d):
                    p10, p50, p90 = np.percentile(d, [10, 50, 90])
                    lab += (
                        f"\n  median {TC.date_lab(init, p50, 'short')}; 10–90 %: {TC.date_lab(init, p10, 'short')}"
                        f" – {TC.date_lab(init, p90, 'short')}"
                    )
            h_, _ = np.histogram(d, bins=np.arange(0, 16))
            ax.bar(
                np.arange(15) + 0.5 + (c - 0.5) * width,
                100 * h_ / q["n_members"],
                width=width,
                color=colr,
                alpha=0.85,
                label=lab,
            )
            if s["named"]:
                endd = np.array([e / 24 for e in q["ended"].values()])
                frac = [100 * np.sum(endd < N) / q["n_members"] for N in range(16)]
                ax.plot(range(16), frac, color=colr, lw=1.6, ls="--", label=f"{MLAB[m]} track ended (cumulative)")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 100 if s["named"] else max(10, ax.get_ylim()[1]))
        ax.set_ylabel("% of members")
        # ticks every 3 days: daily two-line dates run into each other in this narrow panel. Legend "best" for named
        # storms: a fixed upper-left legend would cover the early-recurvature bars and the steep cumulative lines
        TC.date_axis(ax, init, 15, "x", step=3, fmt="stack")
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=8.2, loc="best" if s["named"] else "upper right", frameon=True, framealpha=0.85, edgecolor="none"
        )
        ax.set_title(
            "Recurvature date (bars) · track ended (lines)"
            if s["named"]
            else "When members reach tropical-storm strength (17 m/s)",
            fontsize=10.5,
            loc="left",
        )

    cax = fig.add_axes([0.08, 0.55 / H, 0.5, 0.18 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=bounds)
    cb.set_ticklabels([TC.date_lab(init, b, "short") for b in bounds], fontsize=8)
    cb.set_label(
        "member colour = recurvature date (dot = the turn) · grey = no recurvature · small dots every 24 h · X = start\n"
        "thick dated line: AIFS control (IFS-ENS has no control: the member nearest the median track)"
        + ("" if all(x["named"] for x in rows) else " · ◇ where a member first reaches 17 m/s"),
        fontsize=8.6,
    )
    fig.savefig(out_png, dpi=90, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def _lonlab(lo: float) -> str:
    return f"{abs(lo):.0f}°{'E' if lo >= 0 else 'W'}" if abs(lo) < 179.5 else "180°"


def render_depth(s: dict, et: dict, init, out_png: Path, note: str | None = None):
    """How deep the storm gets after reaching the mid-latitudes - diagnostic only: the members' followed tracks, where and
    when each is deepest, the depth distribution and a factual summary. Nothing about what the storm does to the flow.
    Layout in inches."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    W = 17.0
    ttl = (
        f"How deep {_stormlab(s)} gets after it reaches the mid-latitudes — AIFS-ENS, init {init:%d %b %Y %HZ}"
        if s is not None
        else f"Tropical cyclones after transition — AIFS-ENS, init {init:%d %b %Y %HZ}"
    )
    if s is None or not et:
        fig = plt.figure(figsize=(W, 3.0))
        fig.suptitle(ttl, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.95)
        fig.text(
            0.5,
            0.45,
            note or "No storm reaches the mid-latitudes in enough AIFS-ENS members for this chart.",
            ha="center",
            fontsize=13,
            color="#6f6b64",
            wrap=True,
        )
        fig.savefig(out_png, dpi=80, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
        plt.close(fig)
        return
    pm = np.array([e["pmin"] for e in et.values()])
    n = len(pm)
    dmin = np.array([e["h"] for e in et.values()]) / 24.0
    n_merge = sum(1 for e in et.values() if not e["on_track"])
    b960, b950 = int((pm < 960).sum()), int((pm < 950).sum())
    lines = [
        f"• {n} members' lows reach 35°N. {b960} deepen below 960 hPa and {b950} below 950; the median is {np.median(pm):.0f} hPa "
        f"(10th–90th percentile {np.percentile(pm, 10):.0f}–{np.percentile(pm, 90):.0f})."
        + (f" The control: {et[0]['pmin']:.0f} hPa." if 0 in et else ""),
        f"• The deepest point comes around day {np.median(dmin):.1f} (10th–90th percentile day {np.percentile(dmin, 10):.1f}–"
        f"{np.percentile(dmin, 90):.1f}), near {np.median([e['lat'] for e in et.values()]):.0f}°N "
        f"{_lonlab(lon_median([e['lon'] for e in et.values()]))} (median over members).",
        f"• {n_merge} of {n} members reach their deepest point in a low the storm merges with (◆), not on its own followed path: "
        f"during the transition the centre often re-forms along the jet, and ECMWF's tracker stops at that point.",
    ]
    H = 1.25 + 4.6 + 0.6 + 0.35 + 0.42 * len(lines) + 0.25 + 0.8
    fig = plt.figure(figsize=(W, H))

    def fx(x):
        return x / W

    def fy(y):
        return 1 - y / H

    fig.suptitle(ttl, fontsize=15, fontweight="bold", x=0.02, ha="left", y=fy(0.22), va="top")
    fig.text(
        0.02,
        fy(0.55),
        f"Each member's low is followed past the tracker's last fix (dashed). Depth = the lowest central pressure north of 35°N "
        f"of the storm or of any low it merges with (within {SYS_KM:.0f} km, up to {SYS_H} h after the tracker's last fix).\n"
        f"Colours: depth, as in the histogram.",
        fontsize=10.2,
        color="#444",
        va="top",
    )
    vmin, vmax = np.floor(np.percentile(pm, 2) / 5) * 5, np.ceil(np.percentile(pm, 98) / 5) * 5
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin, vmax + 5)

    def shown(e):  # a member's path up to a day after its deepest point (decaying lows clutter)
        x = e["ext"]
        keep = x["steps"] <= e["h"] + 24
        return {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(x["steps"]) else v) for k, v in x.items()}

    paths = {k: shown(e) for k, e in et.items()}
    allla = np.concatenate([q["lat"] for q in paths.values()] + [np.array([e["lat"] for e in et.values()])])
    allo = np.concatenate([q["lon"] % 360 for q in paths.values()] + [np.array([e["lon"] % 360 for e in et.values()])])
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
    import cartopy.crs as ccrs

    ax = TC.sector_axes(fig, [fx(0.3), fy(1.25 + 4.6 - (bh - h_) / 2), fx(w_), h_ / H], lo0, lo1, la0, la1)
    hx = 0.3 + w_ + 1.0
    hw = W - hx - 0.35
    for k, e in sorted(et.items(), key=lambda kv: -kv[1]["pmin"]):
        x = paths[k]
        fol = x.get("followed", np.zeros(len(x["steps"]), bool))
        col = cmap(norm(e["pmin"]))
        ax.plot(x["lon"][~fol], x["lat"][~fol], color="#9a9a9a", lw=0.5, alpha=0.7, transform=ccrs.Geodetic())
        if fol.any():
            j0 = max(np.where(fol)[0][0] - 1, 0)
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
            marker="o" if e["on_track"] else "D",
        )
    if 0 in et:
        x = paths[0]
        ax.plot(x["lon"], x["lat"], color="#000", lw=2.2, transform=ccrs.Geodetic(), zorder=6)
        ax.scatter(
            [et[0]["lon"]],
            [et[0]["lat"]],
            s=150,
            marker="*",
            color="#000",
            edgecolors="#fff",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    fig.text(
        0.02,
        fy(1.12),
        "Map: grey = the tracker's track · dashed = followed as a surface low (to a day after its deepest point) · "
        "● deepest on that path · ◆ deepest in the low it merged with · black = the control (★ its deepest)",
        fontsize=9.8,
        color="#333",
        va="bottom",
    )
    ah = fig.add_axes([fx(hx), fy(1.25 + 4.6), fx(hw), 4.25 / H])
    bins = np.arange(np.floor(pm.min() / 2) * 2, np.ceil(pm.max() / 2) * 2 + 2.01, 2)
    _cnt, edges, patches = ah.hist(pm, bins=bins, edgecolor="#fff")
    for pa, lft in zip(patches, edges[:-1]):
        pa.set_facecolor(cmap(norm(lft + 1)))
    if 0 in et:
        ah.axvline(et[0]["pmin"], color="#000", lw=1.6, ls="--")
        ah.text(et[0]["pmin"], ah.get_ylim()[1] * 0.97, " control", fontsize=9.5, va="top")
    ah.set_xlabel("deepest central pressure north of 35°N (hPa)", fontsize=10)
    ah.set_ylabel("members", fontsize=10)
    ah.set_title(
        f"{n} members · median {np.median(pm):.0f} hPa · {b960} below 960 · {b950} below 950", fontsize=10.2, loc="left"
    )
    ah.grid(alpha=0.3)
    yy = 1.25 + 4.6 + 0.75
    fig.text(0.02, fy(yy - 0.1), "What the members show", fontsize=12.5, fontweight="bold", va="top")
    yy += 0.3
    for ln in lines:
        wrapped = textwrap.fill(ln, 175, subsequent_indent="   ")
        fig.text(0.02, fy(yy), wrapped, fontsize=10.8, va="top", color="#222")
        yy += 0.22 * (wrapped.count("\n") + 1) + 0.18
    fig.text(
        0.02,
        fy(H - 0.12),
        "Tracks: ECMWF tropical-cyclone tracker on AIFS-ENS (open data, CC BY 4.0), control + 50 members, continued as the "
        "surface-pressure minimum (closed lows at least 2 hPa below their surroundings; jumps of up to 1,800 km in 12 h only along the\n"
        "motion, while the low is not decaying and onto a low below 1000 hPa; never onto a low that already existed as a separate system; "
        "a briefly double-centred low is one system; a low of 1000 hPa or more\nonly over the sea north of 30°N; at most 72 h past the "
        "tracker's last fix). Pressure: AIFS-ENS mean sea-level pressure, 12-hourly, 0.5°.",
        fontsize=8.8,
        color="#6f6b64",
        va="bottom",
    )
    fig.savefig(out_png, dpi=80, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="assets/sst")
    ap.add_argument("--no-z500", action="store_true", help="skip the depth chart (no surface-pressure download)")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    t0 = time.time()
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    out = Path(a.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    per, mstat = ensemble_systems(a.date, int(a.time))
    # merge the two models: named storms by id; a developing system under the AIFS-ENS id when pair_genesis pairs it with an
    # IFS-ENS one, else per model
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
            storms.setdefault(
                key,
                dict(
                    key=key,
                    id=ref["id"],
                    name=ref["name"],
                    named=s["named"],
                    region=ref.get("region"),
                    tag="IFS-ENS " if key.startswith("ifs:") else "",
                    models={},
                ),
            )["models"][m] = s
    for key in [k for k, v in storms.items() if all(x.get("weak") for x in v["models"].values())]:
        del storms[key]  # below the bar in every model: not shown
    st = {k: {m: stats(s["models"][m], NMEM[m]) for m in s["models"]} for k, s in storms.items()}
    order = sorted(
        storms.values(), key=lambda s: (not s["named"], -max(st[s["key"]][m]["p_recurve"] for m in s["models"]))
    )
    print(
        "  storms: "
        + ", ".join(
            f"{_stormlab(s)} " + "/".join(f"{MLAB[m]} {100 * st[s['key']][m]['p_recurve']:.0f}%" for m in s["models"])
            for s in order
        ),
        flush=True,
    )
    status = "missing" if not per else ("ok" if order else "none")
    render_tracks(order, st, init, out / "tcens_tracks.webp", mstat)
    # one figure per storm for the page's storm picker; the folder is this script's and tc_phase's alone - stale storms'
    # files are removed here
    tcd = out / "tc"
    tcd.mkdir(exist_ok=True)
    for f in list(tcd.glob("tracks_*.webp")) + list(tcd.glob("depth_*.webp")):
        f.unlink()
    for s in order[:MAX_PER_STORM]:
        try:
            render_tracks([s], st, init, tcd / f"tracks_{slug(s['key'])}.webp", mstat)
        except Exception as e:
            print(f"  per-storm tracks {s['key']} failed ({e!r})", flush=True)
    # step 2 - diagnostic only (see the module docstring): how deep each named storm's low gets once it reaches the
    # mid-latitudes, from surface pressure alone (no 500 hPa download).
    depth, depth_status = {}, ("missing" if status == "missing" else "ok")
    cand = [s for s in order if "aifs" in s["models"] and s["named"]]
    if not a.no_z500 and cand:
        try:
            import store as ecmwf

            M, mlat, mlon, msteps = load_msl(a.date, a.time)
            land = TC.load_land(ecmwf.Cycle(a.date, a.time))  # the follower's land rule (heat / lee lows)
            for s in cand:
                et = et_depth(s["models"]["aifs"], M, mlat, mlon, msteps, land=land)
                if len(et) < MIN_ET_MEMBERS:
                    continue
                dm = np.median([e["h"] for e in et.values()]) / 24
                late = dm > ET_H_MAX / 24 - 1.5 or np.mean([e["h"] >= ET_H_MAX - 12 for e in et.values()]) > 1 / 3
                depth[s["key"]] = (et, late)
        except Exception as e:
            TC.report(f"depth after transition: AIFS-ENS surface pressure unavailable ({e!r})", "warning")
            depth, depth_status = {}, "missing"
    ok = sorted(
        [k for k, v in depth.items() if not v[1]], key=lambda k: np.median([e["pmin"] for e in depth[k][0].values()])
    )
    late_names = [_stormlab(storms[k]) for k, v in depth.items() if v[1]]
    for n_, fn in enumerate(("tcens_depth.webp", "tcens_depth_2.webp")):
        if n_ < len(ok):
            render_depth(storms[ok[n_]], depth[ok[n_]][0], init, out / fn)
        elif depth_status == "missing" or status == "missing":
            render_depth(
                None,
                {},
                init,
                out / fn,
                note=UNAVAILABLE if status == "missing" else "Surface-pressure data unavailable for this cycle",
            )
        else:
            msg = (
                ("No other storm" if n_ else "No storm")
                + " reaches the mid-latitudes and deepens within day 10 in at least "
                f"{MIN_ET_MEMBERS} AIFS-ENS members this cycle."
            )
            if late_names:
                msg += " Deepening after day 10 (beyond this chart): " + ", ".join(late_names) + "."
            render_depth(None, {}, init, out / fn, note=msg)
    for k, (et_, _late) in depth.items():
        try:
            render_depth(storms[k], et_, init, out / "tc" / f"depth_{slug(k)}.webp")
        except Exception as e:
            print(f"  per-storm depth {k} failed ({e!r})", flush=True)
    if depth:
        print(
            "  depth: "
            + "; ".join(
                f"{_stormlab(storms[k])} {len(v[0])} members, median {np.median([e['pmin'] for e in v[0].values()]):.0f} hPa"
                + (" (deepens after day 10)" if v[1] else "")
                for k, v in depth.items()
            ),
            flush=True,
        )
    js = {
        "init": f"{init:%Y-%m-%dT%H:%MZ}",
        "status": status,
        "depth_status": depth_status,
        "models": {m: {"members": NMEM[m], "status": mstat[m]} for m in NMEM},
        "depth_storms": ok[:2],
        "storms": [],
    }
    for s in order:
        e = {
            "key": s["key"],
            "id": slug(s["key"]) if s["key"].startswith("ifs:") else s["id"],  # unique: an IFS-only
            "name": _stormlab(s),
            "named": s["named"],
            "models": {},
            "slug": slug(s["key"]),  # system can share the AIFS id
            # the AIFS-ENS control's tracker id(s) for this system: the id tc_phase and the TC-jet card (tcphase.json,
            # tcjet.json) use. The picker and tc_alert match through it, never by id string (a control 72L can be member 0
            # of cluster 73L, or control 71L sit in "71L-2" and 72L in "71L")
            "control_ids": list(s["models"]["aifs"].get("ctl_ids", [])) if "aifs" in s["models"] else [],
        }
        if (out / "tc" / f"tracks_{e['slug']}.webp").exists():
            e["tracks_file"] = f"tc/tracks_{e['slug']}.webp"
        if (out / "tc" / f"depth_{e['slug']}.webp").exists():
            e["depth_file"] = f"tc/depth_{e['slug']}.webp"
        mdl = "aifs" if "aifs" in s["models"] else "ifs"  # a reference track for the map: the
        tr = s["models"][mdl]["tracks"]  # AIFS control where it has one,
        k0 = ref_member(tr)  # the same dated line as the figure
        e["track"] = {
            "model": mdl,
            "member": int(k0),
            "h": [int(h) for h in tr[k0]["steps"]],
            "lat": [round(float(v), 2) for v in tr[k0]["lat"]],
            "lon": [round(float(v), 2) for v in tr[k0]["lon"]],
        }
        for m in s["models"]:
            q = st[s["key"]][m]
            e["models"][m] = {
                "tracked": q["n_tracked"],
                "members": q["n_members"],
                "p_recurve": q["p_recurve"],
                "recurve_day_p10_p50_p90": q["rec_day_q"],
                "p_track_ended_by_day": {
                    str(N): round(
                        float(np.mean([v < N * 24 for v in q["ended"].values()])) * q["n_tracked"] / q["n_members"], 3
                    )
                    for N in (3, 5, 7, 10)
                },
            }
            if not s["named"]:  # when members reach tropical-storm strength (hours)
                gh = [g for g in (genesis_h(t) for t in s["models"][m]["tracks"].values()) if g is not None]
                if gh:
                    e["models"][m]["genesis_h_p10_p50_p90"] = [round(float(np.percentile(gh, q))) for q in (10, 50, 90)]
        if s["key"] in depth:
            et_, late_ = depth[s["key"]]
            pm_ = np.array([v["pmin"] for v in et_.values()])
            e["depth"] = {
                "members": len(et_),
                "deepens_after_day10": bool(late_),
                "pmin_p10_p50_p90": [round(float(np.percentile(pm_, q)), 1) for q in (10, 50, 90)],
                "below_960": int((pm_ < 960).sum()),
                "below_950": int((pm_ < 950).sum()),
                "day_of_min_median": round(float(np.median([v["h"] for v in et_.values()]) / 24), 2),
                "merged_low_minima": int(sum(not v["on_track"] for v in et_.values())),
                "control_pmin": et_[0]["pmin"] if 0 in et_ else None,
            }
        js["storms"].append(e)
    (out / "data" / "tcens.json").write_text(json.dumps(js))
    if status == "missing":
        TC.report(
            f"TC ensembles {a.date} {int(a.time):02d}Z: {UNAVAILABLE.lower()} (both tracker files missing)", "error"
        )
    print(
        f"  wrote tcens_tracks.webp, tcens_depth(_2).webp, data/tcens.json ({status}) in {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
