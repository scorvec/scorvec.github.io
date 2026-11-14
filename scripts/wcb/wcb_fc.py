#!/usr/bin/env python3
"""Warm conveyor belts in the 50 IFS-ENS members, and the maps of where the ensemble puts them (elias.py).

DATA. ECMWF IFS-ENS open data, the 00 UTC run's 50 perturbed members (open data has no control on pressure levels),
12-hourly, from the Google Cloud mirror only (scripts/ecmwf/rangefetch.py) and never cached. Per member and step the
21 fields of elias.LEVELS - u, v and q at 1000, 850, 700, 500 and 300 hPa, t at 700, 500 and 300 hPa, geopotential
height at 850, 500 and 200 hPa - of which only the rows from 90N to 7.5S are fetched (the leading bytes of each
message: rangefetch.fetch_rows), and the 300 hPa divergence whole (the outflow's irrotational wind is a global
inversion); for the potential vorticity at 300 hPa, also u, v and t at 400 and 250 hPa, the levels either side. The
first and last 24 h of the fetch serve the neighbouring leads' ascent alone and take the ten fields the ascent needs.
Each 0.25 degree field is reduced to the networks' 1 degree grid by the mean over the 1 degree box
centred on each point (box_mean).

IDENTIFICATION. Every member at every step on its own, by the published ELIAS 2.0 networks (elias.py; the models,
thresholds and climatology are the unzipped archive of release wcb-ref-v1, --archive): the ascent first, at every
step; then the inflow, whose last predictor is the same member's ascent probability 24 h later, and the outflow,
whose last predictor is its ascent probability 24 h earlier. A column holds a stage where the probability exceeds the
valid day's decision threshold. The leads run from 24 h to 24 h before --last-step: those with all three stages.

PRODUCT. Per lead and stage, the share of members with the stage over the column, drawn where it is above how often
the trajectories put it there around the valid date (the archive's 30-day climatology): a one-sided binomial test of
the members' count against that frequency (floored at P_FLOOR: the climatology has columns with none), kept at the
false discovery rate FDR over the map -> loop frames + manifest, and a json of the numbers behind them and what the
fetch cost.

TROPOPAUSE. Per lead, what the outflow does to the potential vorticity (PV) at 300 hPa: each member's Ertel PV there
(potential_vorticity) and the rate at which its own irrotational wind advects it - where a belt's outflow blows
across the tropopause it replaces stratospheric air with tropospheric and builds the ridge (Archambault et al. 2013,
Mon. Wea. Rev. 141, 2325-2346; Grams and Archambault 2016, Mon. Wea. Rev. 144, 3847-3869). Drawn: the members' mean
rate where they agree on its sign (two-sided sign test, false discovery rate FDR over the map's rows), the median
member's tropopause (half the members with at least TROPOPAUSE PVU) and the outflow's share -> a second loop in the
same manifest.

    python wcb_fc.py run --date 20261114 --archive ARCHIVE --out wcb_fc.npz       # fetch and identify
    python wcb_fc.py render --fc wcb_fc.npz --archive ARCHIVE --out-dir assets/sst
    python wcb_fc.py forecast --date 20261114 --archive ARCHIVE --out-dir assets/sst
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "ecmwf"))
sys.path.insert(0, str(HERE.parent / "strat"))
import elias as E
import ifs_ens as IE  # the IFS-ENS publication check (scripts/ecmwf/cycles.py's shared rule)

MODEL, LABEL, NAME = "ifs", "IFS-ENS", "wcb_ifs"
HOUR = "00"
STEP_H = 12
MEMBERS = 50
MIN_SHARE = 0.9  # a lead with fewer than this share of the members complete in a stage is dropped
SOURCES = ["google"]  # the Google Cloud mirror only
STEP_TRIES = 3
NJ, NI = 721, 4 * E.LON.size  # the open data's 0.25 degree grid, rows from 90N
NORTH = 4 * (E.ROWS.size - 1) + 3  # its leading rows that E.ROWS takes: 90N to 7.5S
PARAM = {"u": "u", "v": "v", "q": "q", "t": "t", "z": "gh"}  # elias' names in the open data
FIELDS = [(k, lev) for k, levels in E.LEVELS.items() for lev in levels]
ASCENT_FIELDS = [(k, lev) for k, lev in FIELDS if (k, lev) in {
    ("u", 850), ("v", 850), ("t", 700), ("q", 700), ("u", 300), ("v", 300), ("z", 500), ("z", 200), ("v", 500),
    ("q", 500),
}]  # fmt: skip
DIV = ("d", 300)
PV_LEVS = (400, 300, 250)  # potential vorticity at the middle level, its vertical derivatives over the three
PV_FIELDS = [(k, lev) for k in ("u", "v", "t") for lev in PV_LEVS]
FULL_FIELDS = FIELDS + [x for x in PV_FIELDS if x not in FIELDS]
TROPOPAUSE = 2.0  # PVU
PV_MAPS = ("strat", "adv", "neg")  # outflow_pv's maps, kept per lead
METRE = E.R_EARTH * np.pi / 180.0  # per degree
COS = np.cos(np.deg2rad(E.LAT))[:, None]
STAGE = list(E.STAGES)
CELLS = E.LAT.size * E.LON.size
FDR = 0.05
P_FLOOR = 0.005  # the lowest climatological frequency a count is tested against
MAP_LAT = (10.0, 80.0)
INK, MUTED = "#1a1a1a", "#6f6b64"
P_LEV = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # the jet cards' probability scale
P_COL = ["#fff3c4", "#fde28a", "#fdc458", "#fca03f", "#f7792f", "#e8512a", "#cc2f2f", "#a41d3f", "#7a1450", "#4d0f52"]
CLIM_LEV = (2, 5, 10)  # % contours of the trajectory climatology
PV_NAME = NAME + "_pv"
SECTORS = (("North Pacific and North America", 210.0), ("North Atlantic and Eurasia", -10.0))  # central longitudes
ADV_LEV = np.array([-8, -6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6, 8])  # PVU per day
BELT_LEV = (20, 50, 80)  # % of members with the outflow
BELT = "#0b6e4f"
TITLE = {"inflow": "Inflow (below 800 hPa)", "ascent": "Ascent (800–400 hPa)", "outflow": "Outflow (above 400 hPa)"}
CREDIT = (
    "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25° reduced to 1°, Google Cloud mirror · "
    "identification: ELIAS 2.0, Quinting & Grams 2022 (models, thresholds and trajectory climatology: Zenodo, "
    "doi:10.5281/zenodo.5154980, CC BY 4.0)"
)


# ── fetch and identify ──────────────────────────────────────────────────────────────────────────────────────────────
def lead_steps(last_step):
    return list(range(0, int(last_step) + 1, STEP_H))


def shown_steps(last_step):
    """The leads with all three stages: the ascent is needed LAG_H either side."""
    return [s for s in lead_steps(last_step) if s >= E.LAG_H and s + E.LAG_H <= last_step]


def _from(v, lon0, start):
    """The columns of a 0.25 degree field whose first lies at lon0, in the order that begins at `start` (deg E)."""
    return np.roll(v, -round(((start - lon0) % 360.0) * 4), axis=-1)


def box_mean(values, lat, lon):
    """rangefetch.fetch_rows' reduction of one message: the NORTH leading rows of a 0.25 degree field -> E.ROWS x
    E.LON, each point the mean over the 1 degree box centred on it (weights 1/2, 1, 1, 1, 1/2 each way; at 90N the
    half box south of the pole)."""
    if values.shape != (NORTH, NI) or lat[0] != 90.0 or lat[0] - lat[1] != 0.25:
        raise RuntimeError(f"not the 0.25 degree grid from 90N: {values.shape}, latitudes from {lat[0]}")
    v = _from(np.asarray(values, np.float64), lon[0], E.LON[0])
    v = np.concatenate([v[:, -2:], v, v[:, :2]], axis=1)
    n = E.LON.size
    x = 0.5 * v[:, 0::4][:, :n] + v[:, 1::4][:, :n] + v[:, 2::4][:, :n] + v[:, 3::4][:, :n] + 0.5 * v[:, 4::4]
    out = np.empty((E.ROWS.size, n), np.float32)
    out[-1] = (x[0] + x[1] + 0.5 * x[2]) / 10.0
    k = 4 * np.arange(1, E.ROWS.size)
    out[-2::-1] = (0.5 * x[k - 2] + x[k - 1] + x[k] + x[k + 1] + 0.5 * x[k + 2]) / 16.0
    return out


def _global(buf, eccodes):
    """A whole 0.25 degree message as irrotational_wind takes it: rows from 90N without the south pole, columns
    from 0E."""
    gid = eccodes.codes_new_from_message(buf)
    try:
        grid = (eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni"))
        north = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        if grid != (NJ, NI) or north != 90.0 or eccodes.codes_get(gid, "jScansPositively"):
            raise RuntimeError(f"not the 0.25 degree grid from 90N: {grid}, latitudes from {north}")
        lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        return _from(eccodes.codes_get_values(gid).reshape(NJ, NI), lon0, 0.0)[:-1]
    finally:
        eccodes.codes_release(gid)


def _fields(stack, keys):
    """(member, field, ROWS, LON) in the order of `keys` -> elias' input: the humidity never negative, the
    geopotential from its height."""
    f = {}
    for j, (k, lev) in enumerate(keys):
        a = stack[:, j].astype(np.float64)
        f.setdefault(k, {})[lev] = np.maximum(a, 0.0) if k == "q" else a * E.G if k == "z" else a
    return f


def _ddx(a):
    """d/dx of a field on E.LAT, periodic in longitude (elias' own differences stop at the date line, as the
    networks were trained)."""
    return (np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1)) / (2.0 * METRE * COS)


def potential_vorticity(f):
    """Ertel potential vorticity (PVU) at the middle of PV_LEVS on E.LAT x E.LON, from u, v (m/s) and t (K) on
    E.ROWS at those levels: -g [(f + zeta) dtheta/dp - dv/dp dtheta/dx + du/dp dtheta/dy]. The vertical derivatives
    are centred in ln p, theta's through ln theta (it is close to a power of p)."""
    at = PV_LEVS[1]
    lnp = np.log(100.0 * np.array(PV_LEVS))
    inner = (..., slice(1, -1), slice(None))  # E.ROWS -> E.LAT

    def ddp(a):
        return np.gradient(a, lnp, axis=0)[1] / (100.0 * at)

    def ddy(a):
        return (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * METRE)

    u, v = (np.stack([f[k][lev] for lev in PV_LEVS]) for k in ("u", "v"))
    th = np.stack([f["t"][lev] * (1000.0 / lev) ** E.KAPPA for lev in PV_LEVS])
    cos = np.cos(np.deg2rad(E.ROWS))[:, None]
    vort = 2.0 * E.OMEGA * np.sin(np.deg2rad(E.LAT))[:, None] + _ddx(v[1][inner]) - ddy(u[1] * cos) / COS
    th_p = (th[1] * ddp(np.log(th)))[inner]
    return -1.0e6 * E.G * (vort * th_p - ddp(v)[inner] * _ddx(th[1][inner]) + ddp(u)[inner] * ddy(th[1]))


def outflow_pv(f, chi):
    """One step's members -> the tropopause maps (PV_MAPS) on E.LAT x E.LON: `strat`, how many members have
    stratospheric air (at least TROPOPAUSE) at the level; `adv`, the members' mean rate (PVU per day) at which their
    own irrotational wind `chi` (member, [eastward, northward], LAT, LON) advects the potential vorticity; `neg`, how
    many members have that rate negative."""
    pv = potential_vorticity(f)
    adv = -86400.0 * (chi[:, 0] * _ddx(pv) + chi[:, 1] * np.gradient(pv, axis=-2) / METRE)
    return dict(
        strat=(pv >= TROPOPAUSE).sum(axis=0).astype(np.uint8),
        adv=adv.mean(axis=0).astype(np.float32),
        neg=(adv < 0.0).sum(axis=0).astype(np.uint8),
    )


def _net(root, stage):
    return E.Net(Path(root) / "models" / f"cnn_{E.STAGES[stage]}.h5")


def _valid(date, step):
    return pd.Timestamp(f"{date} {HOUR}:00") + pd.Timedelta(hours=int(step))


def _scratch(scratch, step):
    return Path(scratch) / f"s{step:03d}.npz"


def first_pass(job):
    """One step of the run for perturbed members 1..n: the fields fetched and reduced, the ascent identified; with
    `full`, every stage's fields and the irrotational wind speed for second_pass, and the tropopause maps
    (outflow_pv). The members, their ascent probabilities and (full) the fields, the speed and those maps go to the
    scratch file. -> {step, members, cost}; None when the step could not be fetched."""
    import eccodes
    import rangefetch as rf

    date, step, n, root, scratch, full = job
    keys = FULL_FIELDS if full else ASCENT_FIELDS
    need = [(PARAM[k], lev) for k, lev in keys] + ([DIV] if full else [])
    for attempt in range(1, STEP_TRIES + 1):
        try:
            t0 = time.time()
            idx = rf.fetch_index(date, HOUR, MODEL, step, "ef", sources=SOURCES)
            at = {
                (int(e["number"]), e["param"], int(e["levelist"])): e
                for e in rf.select(
                    idx,
                    param=sorted({p for p, _ in need}),
                    levelist=sorted({lev for _, lev in need}),
                    numbers=list(range(1, n + 1)),
                    type="pf",
                )
            }
            lacking = sorted(set(need) - {k[1:] for k in at})
            if lacking:  # not a transport failure: the run does not carry these fields
                print(f"  step {step:3d}: the run lacks {lacking}; skipped", flush=True)
                return None
            members = [m for m in range(1, n + 1) if all((m, *x) in at for x in need)]
            rel = rf.path_for(date, HOUR, MODEL, step, "ef") + ".grib2"
            want = [at[(m, PARAM[k], lev)] for m in members for k, lev in keys]
            wire = {}
            rows = rf.fetch_rows(rel, want, tuple(range(NORTH)), sources=SOURCES, stats=wire, each=box_mean)
            stack = np.stack(rows).reshape(len(members), len(keys), E.ROWS.size, E.LON.size)
            del rows
            kept = dict(members=np.array(members))
            if full:
                div = sorted((at[(m, *DIV)] for m in members), key=lambda e: e["_offset"])
                blob = rf.fetch_ranges(rel, rf.coalesce(div), sources=SOURCES, stats=wire)
                wind, pos = {}, 0
                t1 = time.time()
                for e in div:
                    field = _global(bytes(blob[pos : pos + e["_length"]]), eccodes)
                    wind[int(e["number"])] = E.irrotational_wind(field)
                    pos += e["_length"]
                if pos != len(blob):
                    raise RuntimeError(f"{len(blob)} bytes for {pos} indexed")
                del blob
                want += div
                chi = np.stack([wind[m] for m in members])
                kept.update(
                    stack=stack[:, : len(FIELDS)],
                    speed=np.hypot(chi[:, 0], chi[:, 1]).astype(np.float32),
                    **outflow_pv(_fields(stack[:, [keys.index(x) for x in PV_FIELDS]], PV_FIELDS), chi),
                )
            else:
                t1 = time.time()
            image = E.ascent_image(_fields(stack, keys), E.climatology(root, "ascent", _valid(date, step)))
            np.savez(_scratch(scratch, step), ascent=_net(root, "ascent")(image), **kept)
            cost = dict(
                messages=len(want),
                bytes=int(sum(e["_length"] for e in want)),
                wire=int(sum(wire.values())),
                fetch_s=round(t1 - t0, 1),
                identify_s=round(time.time() - t1, 1),
            )
            print(
                f"  step {step:3d}: {len(members)} members, {cost['messages']} messages, {cost['bytes'] / 1e6:,.0f} MB "
                f"({cost['wire'] / 1e6:,.0f} on the wire); fetch {cost['fetch_s']:.0f} s, ascent "
                f"{cost['identify_s']:.0f} s",
                flush=True,
            )
            return dict(step=step, members=kept["members"], cost=cost)
        except Exception as e:  # noqa: BLE001  one step must not end the run; the leads that need it are dropped
            print(f"  step {step:3d}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            time.sleep(20 * attempt)
    return None


def second_pass(job):
    """One lead from the scratch files: the ascent's footprint, and the inflow and the outflow of the members that
    also have the ascent LAG_H later and earlier. -> {step, foot (stage, n, packed columns), have (stage, n): the
    members identified, the step's tropopause maps (PV_MAPS) and how many members they hold (pv_members),
    identify_s}; member m is row m - 1."""
    date, step, n, root, scratch = job
    t0 = time.time()
    valid = _valid(date, step)
    with np.load(_scratch(scratch, step)) as z:
        members, ascent, speed, f = z["members"], z["ascent"], z["speed"], _fields(z["stack"], FIELDS)
        pv = {k: z[k] for k in PV_MAPS}
    foot, have = np.zeros((len(STAGE), n, CELLS), bool), np.zeros((len(STAGE), n), bool)

    def put(stage, who, prob):
        k = STAGE.index(stage)
        foot[k, who - 1] = (prob > E.thresholds(root, stage, valid)).reshape(who.size, CELLS)
        have[k, who - 1] = True

    put("ascent", members, ascent)
    for stage, other in (("inflow", step + E.LAG_H), ("outflow", step - E.LAG_H)):
        with np.load(_scratch(scratch, other)) as z:
            who, i, j = np.intersect1d(members, z["members"], return_indices=True)
            lagged = z["ascent"][j]
        sub = {k: {lev: a[i] for lev, a in levels.items()} for k, levels in f.items()}
        image = E.inflow_image(sub, lagged) if stage == "inflow" else E.outflow_image(sub, speed[i], lagged)
        put(stage, who, _net(root, stage)(image))
    took = round(time.time() - t0, 1)
    print(f"  step {step:3d}: inflow and outflow of {have.sum(axis=1).min()} members in {took:.0f} s", flush=True)
    return dict(step=step, foot=np.packbits(foot, axis=-1), have=have, pv_members=members.size, identify_s=took, **pv)


def run(date, last_step, root, out, workers=3, wait_min=0.0, members=MEMBERS):
    """Fetch and identify the run for perturbed members 1..members -> npz (per-lead member footprints, tropopause
    maps and the fetch's cost). Waits up to wait_min for the run to be on the Google mirror to last_step."""
    shown = shown_steps(last_step)
    if not shown:
        raise SystemExit(f"no lead has the ascent {E.LAG_H} h either side within {last_step} h")
    deadline = time.time() + 60 * wait_min
    while not IE.published(date, HOUR, step=last_step):
        if time.time() >= deadline:
            raise SystemExit(f"IFS-ENS {date} {HOUR}Z is not on the Google mirror to step {last_step}")
        print(f"  IFS-ENS {date} {HOUR}Z not complete on the Google mirror yet; checking again in 2 min", flush=True)
        time.sleep(120)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as scratch, ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [(date, s, members, str(root), scratch, s in shown) for s in lead_steps(last_step)]
        got = {r["step"]: r for r in ex.map(first_pass, jobs) if r is not None}
        enough = {s for s, r in got.items() if r["members"].size >= MIN_SHARE * members}
        ready = [s for s in shown if {s - E.LAG_H, s, s + E.LAG_H} <= enough]
        done = list(ex.map(second_pass, [(date, s, members, str(root), scratch) for s in ready]))
    wall = time.time() - t0
    keep = [r for r in done if r["have"].sum(axis=1).min() >= MIN_SHARE * members]
    if not keep:
        raise SystemExit("no lead was identified with enough members in every stage")
    cost = {k: sum(r["cost"][k] for r in got.values()) for k in ("messages", "bytes", "wire")}
    cost.update(
        steps=len(got),
        identify_s=round(sum(r["cost"]["identify_s"] for r in got.values()) + sum(r["identify_s"] for r in done), 1),
        wall_s=round(wall, 1),
        workers=workers,
    )
    print(
        f"IFS-ENS {date} {HOUR}Z: {cost['steps']} steps, {cost['messages']:,} messages, {cost['bytes'] / 1e9:.2f} GB "
        f"wanted, {cost['wire'] / 1e9:.2f} GB on the wire, {len(keep)} leads identified, {wall:.0f} s wall",
        flush=True,
    )
    np.savez_compressed(
        out,
        init=np.datetime64(_valid(date, 0), "s"),
        steps=np.array([r["step"] for r in keep]),
        foot=np.stack([r["foot"] for r in keep]),
        have=np.stack([r["have"] for r in keep]),
        pv_members=np.array([r["pv_members"] for r in keep]),
        cost=json.dumps(cost),
        **{k: np.stack([r[k] for r in keep]) for k in PV_MAPS},
    )
    return Path(out)


def load_run(path):
    """npz from `run` -> its arrays: init, steps, foot (lead, stage, n, LAT, LON), have (lead, stage, n), the
    tropopause maps (PV_MAPS: lead, LAT, LON) with pv_members (lead), cost."""
    with np.load(path) as f:
        got = {k: f[k] for k in ("steps", "have", "pv_members", *PV_MAPS)}
        got.update(init=pd.Timestamp(f["init"][()]), cost=json.loads(str(f["cost"])))
        foot = np.unpackbits(f["foot"], axis=-1, count=CELLS).astype(bool)
    return dict(got, foot=foot.reshape(*foot.shape[:-1], E.LAT.size, E.LON.size))


# ── the product ─────────────────────────────────────────────────────────────────────────────────────────────────────
def risk(foot, normal):
    """(member, LAT, LON) footprints of one stage at one lead against its climatological frequency -> (the share of
    the members, where their count is above that frequency at the false discovery rate FDR)."""
    from quantmet.stats import benjamini_hochberg
    from scipy.stats import binom

    k = foot.sum(axis=0)
    p = binom.sf(k - 1, foot.shape[0], np.maximum(np.nan_to_num(normal), P_FLOOR))
    return k / foot.shape[0], benjamini_hochberg(p, FDR)


def agreed(neg, n, where):
    """Where the members share a sign: a two-sided sign test of `neg` negative among n, kept at the false discovery
    rate FDR over the cells of `where`."""
    from quantmet.stats import benjamini_hochberg
    from scipy.stats import binom

    p = np.minimum(1.0, 2.0 * binom.sf(np.maximum(neg, n - neg) - 1.0, n, 0.5))
    return benjamini_hochberg(np.where(where, p, np.nan), FDR)


def products(run, root):
    """Per lead and stage: the share of members, where it is drawn and the climatological frequency. Per lead: the
    share of members with stratospheric air at the tropopause maps' level (trop), the mean advection of potential
    vorticity by the irrotational wind (adv) and where the members agree on its sign (sure)."""
    init, steps, foot, have, n = (run[k] for k in ("init", "steps", "foot", "have", "pv_members"))
    lat = E.LAT[:, None]
    band = np.broadcast_to((lat >= MAP_LAT[0]) & (lat <= MAP_LAT[1]), (E.LAT.size, E.LON.size))
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=int(s)) for s in steps])
    normal = np.array([[E.climatology(root, stage, v) for stage in STAGE] for v in valid])
    got = [[risk(foot[i, k, have[i, k]], normal[i, k]) for k in range(len(STAGE))] for i in range(len(steps))]
    return dict(
        init=init,
        steps=np.asarray(steps),
        valid=valid,
        members=[int(have[i].sum(axis=1).min()) for i in range(len(steps))],
        share=np.array([[g[0] for g in row] for row in got], np.float32),
        shown=np.array([[g[1] for g in row] for row in got]),
        normal=normal.astype(np.float32),
        trop=run["strat"] / n[:, None, None],
        adv=run["adv"],
        sure=np.array([agreed(run["neg"][i].astype(float), int(n[i]), band) for i in range(len(steps))]),
        band=band,
    )


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────────────
def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import house

    house.apply()

    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    return plt


def cyclic(f):
    """E.LON (180W-179E) -> 0-360E, the first column repeated so the maps close."""
    g = np.roll(f, E.LON.size // 2, axis=-1)
    return np.concatenate([g, g[..., :1]], axis=-1), np.arange(0.0, 361.0)


def nh_map(fig, rect, title, centre=180.0, half=179.995):
    """A map of MAP_LAT, `half` degrees of longitude either side of `centre`."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree(central_longitude=centre)
    ax = fig.add_axes(rect, projection=proj)
    ax.set_extent((-half, half, *MAP_LAT), crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333", zorder=4)
    gl = ax.gridlines(
        draw_labels=True, lw=0.3, color="#bbb", xlocs=range(-180, 181, 30), ylocs=range(10, 81, 10), zorder=3
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8.5, "color": INK}
    ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", color=INK)
    return ax


def frame(pr, i, out, dpi=90):
    """One lead: the three stages' member shares where they are above the climatological frequency, with that
    frequency as contours."""
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = mpl()
    map_w, left, right, top, gap, bot = 11.4, 0.55, 1.05, 1.35, 0.5, 0.62
    mh = map_w * (MAP_LAT[1] - MAP_LAT[0]) / 360.0
    W, H = left + map_w + right, top + 3 * mh + 2 * gap + bot
    fig = plt.figure(figsize=(W, H))
    cmap = ListedColormap(P_COL)
    v = pr["valid"][i]
    for k, stage in enumerate(STAGE):
        rect = [left / W, (bot + (2 - k) * (mh + gap)) / H, map_w / W, mh / H]
        ax = nh_map(fig, rect, TITLE[stage])
        f, lon = cyclic(100 * np.where(pr["shown"][i, k], pr["share"][i, k], 0.0))
        cf = ax.contourf(
            lon, E.LAT, f, levels=P_LEV, cmap=cmap, norm=BoundaryNorm(P_LEV, cmap.N), transform=ccrs.PlateCarree()
        )
        g, _ = cyclic(100 * pr["normal"][i, k])
        cs = ax.contour(
            lon, E.LAT, g, levels=CLIM_LEV, colors=INK, linewidths=(0.5, 0.8, 1.2), transform=ccrs.PlateCarree()
        )
        ax.clabel(cs, fmt="%d%%", fontsize=7.5, inline_spacing=2)
        cax = fig.add_axes([rect[0] + rect[2] + 0.12 / W, rect[1] + 0.05 * rect[3], 0.16 / W, 0.9 * rect[3]])
        cb = fig.colorbar(cf, cax=cax, ticks=P_LEV[::2])
        cb.set_label("% of members", fontsize=9)
        cb.ax.tick_params(labelsize=8.5)
    fig.text(
        left / W,
        1 - 0.14 / H,
        f"Warm conveyor belts, day {int(pr['steps'][i]) / 24:g} (valid {v:%a %-d %b %HZ}) — {LABEL} "
        f"{pr['members'][i]} members, init {pr['init']:%Y-%m-%d %HZ}",
        fontsize=13.5,
        fontweight="bold",
        va="top",
        color=INK,
    )
    sub = (
        "Shading: share of members with that stage of a warm conveyor belt over the column, where it is above the "
        f"usual frequency for the date (binomial test, false discovery rate {100 * FDR:g} %). Contours: the usual "
        "frequency, from the air-parcel trajectories the networks were trained on. The air of a belt rises from the "
        "inflow through the ascent to the outflow within two days."
    )
    fig.text(left / W, 1 - 0.5 / H, textwrap.fill(sub, int(W * 12.4)), fontsize=9.4, va="top", color=MUTED)
    fig.text(left / W, 0.06 / H, textwrap.fill(CREDIT, int(W * 15.5)), fontsize=8.2, va="bottom", color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def frame_pv(pr, i, out, dpi=90):
    """One lead: the rate at which the irrotational wind advects the potential vorticity at the middle of PV_LEVS,
    where the members agree on its sign, with the median member's tropopause there and the outflow's share."""
    import cartopy.crs as ccrs
    from matplotlib.colors import from_levels_and_colors

    plt = mpl()
    map_w, left, right, top, gap, bot = 11.4, 0.55, 1.05, 1.35, 0.5, 0.62
    mh = map_w * (MAP_LAT[1] - MAP_LAT[0]) / 180.0
    W, H = left + map_w + right, top + 2 * mh + gap + bot
    fig = plt.figure(figsize=(W, H))
    shades = plt.get_cmap("PuOr")(np.linspace(0.06, 0.94, ADV_LEV.size + 1))
    shades[ADV_LEV.size // 2] = 1.0  # between the innermost levels: white
    cmap, norm = from_levels_and_colors(ADV_LEV, shades, extend="both")
    k, lev, v = STAGE.index("outflow"), PV_LEVS[1], pr["valid"][i]
    adv, lon = cyclic(np.where(pr["sure"][i], pr["adv"][i], 0.0))
    trop, _ = cyclic(100 * pr["trop"][i])
    belt, _ = cyclic(100 * np.where(pr["shown"][i, k], pr["share"][i, k], 0.0))
    for j, (title, centre) in enumerate(SECTORS):
        rect = [left / W, (bot + (1 - j) * (mh + gap)) / H, map_w / W, mh / H]
        ax = nh_map(fig, rect, title, centre, 90.0)
        data = dict(transform=ccrs.PlateCarree())
        cf = ax.contourf(lon, E.LAT, adv, levels=ADV_LEV, cmap=cmap, norm=norm, extend="both", **data)
        ax.contour(lon, E.LAT, belt, levels=BELT_LEV, colors=BELT, linewidths=(0.8, 1.3, 1.8), zorder=5, **data)
        ax.contour(lon, E.LAT, trop, levels=[50], colors=INK, linewidths=1.8, zorder=6, **data)
        cax = fig.add_axes([rect[0] + rect[2] + 0.12 / W, rect[1] + 0.05 * rect[3], 0.16 / W, 0.9 * rect[3]])
        cb = fig.colorbar(cf, cax=cax, ticks=ADV_LEV)
        cb.set_label("PVU per day", fontsize=9)
        cb.ax.tick_params(labelsize=8.5)
    fig.text(
        left / W,
        1 - 0.14 / H,
        f"Belt outflow and the tropopause, day {int(pr['steps'][i]) / 24:g} (valid {v:%a %-d %b %HZ}) "
        f"— {LABEL} {pr['members'][i]} members, init {pr['init']:%Y-%m-%d %HZ}",
        fontsize=13.5,
        fontweight="bold",
        va="top",
        color=INK,
    )
    sub = (
        f"Shading: change of the {lev} hPa potential vorticity by the outflow (irrotational) wind, the members’ "
        f"mean where they agree on its sign (sign test, false discovery rate {100 * FDR:g} %); orange lowers it, "
        f"which builds the ridge. Black line: the tropopause ({TROPOPAUSE:g} PVU) at {lev} hPa in the median member. "
        f"Green contours: {', '.join(map(str, BELT_LEV[:-1]))} and {BELT_LEV[-1]} % of members with a belt’s "
        "outflow, where that is above the usual frequency."
    )
    fig.text(left / W, 1 - 0.5 / H, textwrap.fill(sub, int(W * 12.4)), fontsize=9.4, va="top", color=MUTED)
    fig.text(left / W, 0.06 / H, textwrap.fill(CREDIT, int(W * 15.5)), fontsize=8.2, va="bottom", color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def _loop(anim, pr, draw, note):
    """One loop's frames under `anim`, drawn by `draw` -> their manifest entries; note(i) ends lead i's label."""
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, v in enumerate(pr["valid"]):
        fn = f"F{i:02d}.webp"
        draw(pr, i, anim / fn)
        label = f"day {pr['steps'][i] / 24:g} · valid {v:%a %b %d %HZ} · {note(i)}"
        frames.append({"idx": i, "file": fn, "date": f"{v:%Y-%m-%d}", "label": label})
    print(f"  {len(frames)} map frames -> {anim}", flush=True)
    return {"n_frames": len(frames), "frames": frames}


def render(fc_path, root, out_dir):
    """Both loops' frames, their manifest and the json under out_dir (assets/sst) from a run's npz."""
    run = load_run(fc_path)
    pr = products(run, root)
    out_dir = Path(out_dir)
    peak = 100 * np.where(pr["shown"], pr["share"], 0.0).max(axis=(2, 3))  # (lead, stage)
    agreed_adv = np.where(pr["sure"] & pr["band"], pr["adv"], 0.0)
    low, high = agreed_adv.min(axis=(1, 2)), agreed_adv.max(axis=(1, 2))
    belts = _loop(out_dir / "anim" / NAME, pr, frame, lambda i: f"outflow up to {peak[i, STAGE.index('outflow')]:.0f}%")
    pv = _loop(out_dir / "anim" / PV_NAME, pr, frame_pv, lambda i: f"ridge building up to {abs(low[i]):.1f} PVU/day")
    mani = {
        "ver": int(time.time()),
        "regions": {
            NAME: {"label": f"Warm conveyor belts ({LABEL})", **belts},
            PV_NAME: {"label": f"Outflow and the tropopause ({LABEL})", **pv},
        },
    }
    (out_dir / "anim" / f"{NAME}_manifest.json").write_text(json.dumps(mani))
    doc = dict(
        model=LABEL,
        init=f"{run['init']:%Y-%m-%dT%H:%MZ}",
        fdr=FDR,
        members=pr["members"],
        lead_h=[int(s) for s in pr["steps"]],
        valid=[f"{v:%Y-%m-%dT%H:%MZ}" for v in pr["valid"]],
        max_share={stage: [round(float(x) / 100, 3) for x in peak[:, k]] for k, stage in enumerate(STAGE)},
        outflow_pv=dict(
            level_hpa=PV_LEVS[1],
            tropopause_pvu=TROPOPAUSE,
            members=[int(x) for x in run["pv_members"]],
            lowest_pvu_day=[round(float(x), 2) for x in low],
            highest_pvu_day=[round(float(x), 2) for x in high],
        ),
        fetch=run["cost"],
    )
    data = out_dir / "data" / f"{NAME}.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "forecast"):
        s = sub.add_parser(name)
        s.add_argument("--date", required=True, help="YYYYMMDD of the 00 UTC run")
        s.add_argument("--last-step", type=int, default=192, help="last step fetched; the leads end 24 h before it")
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--members", type=int, default=MEMBERS, help="perturbed members 1..N")
        s.add_argument("--wait-min", type=float, default=0.0, help="wait this long for the run on the mirror")
        s.add_argument("--archive", required=True, help="the unzipped ELIAS 2.0 archive (release wcb-ref-v1)")
        s.add_argument("--out", default="wcb_fc.npz")
        if name == "forecast":
            s.add_argument("--out-dir", required=True)
    s = sub.add_parser("render")
    s.add_argument("--fc", required=True)
    s.add_argument("--archive", required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.fc, a.archive, a.out_dir)
        return 0
    run(a.date, a.last_step, a.archive, a.out, a.workers, a.wait_min, a.members)
    if a.cmd == "forecast":
        render(a.out, a.archive, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
