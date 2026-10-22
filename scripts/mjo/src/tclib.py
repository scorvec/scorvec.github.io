#!/usr/bin/env python3
"""Shared tools for the tropical-cyclone products (tc_ens.py, tc_maps.py, tc_phase.py, tc_verify.py; scripts/tcshear).

Tracks: ECMWF's tropical-cyclone tracker in the open data (Google Cloud mirror only), one BUFR per model and cycle:
  .../{model}/0p25/enfo/{d}{hh}0000-{N}h-enfo-tf.bufr    ifs (IFS-ENS) and aifs-ens (AIFS-ENS); N = the run's length
  Each storm message holds every subset: ensembleForecastType 4 = a perturbed member (1-50), 1 = the AIFS-ENS control
  (member 51 in the file), 0 = the deterministic run of the same system (the IFS control, HRES until May 2026, in the
  IFS-ENS file, AIFS single in the
  AIFS-ENS file; identical to their own oper tf files). Positions 6-hourly; storms with an official id carry the observed
  position at t=0. decode_all reads them; the TC products use the perturbed members and the deterministic runs only.
Recurvature: the 24-h zonal motion turning from westward to eastward (recurvature).
After the tracker's last fix (ECMWF's tracker stops at or near extratropical transition) a storm is followed on as a
surface low (follow_low).
Outflow (Archambault, Bosart, Keyser & Cordeira 2013, MWR 141, 2325-2346; Archambault et al. 2015, MWR 143, 1122-1141):
negative PV advection by the irrotational wind, -v_chi . grad(PV), in the 300-200 hPa layer (irrotational,
pv_advection): v_chi from the layer-mean wind by a spherical-harmonic inversion (wind200_vpot, T106 ~1.7 deg), PV
smoothed 1 deg first.
Fields of the deterministic runs (IFS control, AIFS single) come byte-ranged from the Google mirror (oper_fields), the
IFS control's at 0.25 or 0.1 degrees.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))

A_EARTH = 6.371e6
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"
MISSING = -1e99
FETCH_TRIES = 3  # attempts per tracker file
FETCH_WAIT_S = (20, 60)  # seconds before the second and the third attempt
UA = {"User-Agent": "scorvec.com tropical-cyclone products (+https://scorvec.com)"}
MODELS = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}  # the ensembles, by their key in the products
MODEL_DIR = {"aifs": "aifs-ens", "ifs": "ifs"}  # ... and their directory on the mirror
DET = {"aifs": "AIFS single", "ifs": "IFS control"}  # the deterministic run in each ensemble's tracker file
NMEM = 50  # perturbed members per ensemble


def report(msg: str, level: str | None = None) -> None:
    """Log msg; with level "warning" or "error" also as a GitHub Actions annotation and, when the job summary file is
    set, as a line of the job summary."""
    print(f"  {msg}", flush=True)
    if level:
        print(f"::{level}::{msg}", flush=True)
        summ = os.environ.get("GITHUB_STEP_SUMMARY")
        if summ:
            with open(summ, "a") as f:
                f.write(f"- TC {level}: {msg}\n")


def get_bufr(url: str, fp: Path, label: str, quiet: bool = False) -> Path | None:
    """A tracker BUFR from url, cached at fp: FETCH_TRIES attempts with backoff (none after a 404). Only a response that
    is a BUFR file is accepted, and it is written atomically, so a broken download is never cached. None if every
    attempt fails (the caller records the product's input as missing); quiet: no warning annotation (an optional
    file)."""
    import requests

    if fp.exists() and fp.stat().st_size > 100:
        return fp
    fp.parent.mkdir(parents=True, exist_ok=True)
    why = ""
    for i in range(FETCH_TRIES):
        if i:
            time.sleep(FETCH_WAIT_S[min(i - 1, len(FETCH_WAIT_S) - 1)])
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and r.content[:4] == b"BUFR":
                tmp = fp.with_name(fp.name + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(fp)
                return fp
            why = f"HTTP {r.status_code}" if r.status_code != 200 else f"not a BUFR file ({len(r.content)} bytes)"
            if r.status_code == 404:
                break  # not published: asking again does not help
        except requests.RequestException as e:
            why = f"{type(e).__name__}: {str(e)[:100]}"
        print(f"  {label}: attempt {i + 1}/{FETCH_TRIES} failed ({why})", flush=True)
    report(f"{label} unavailable ({why})", None if quiet else "warning")
    return None


def tracker_file(model: str, date: str, hh: str, cache: Path, quiet: bool = False) -> Path | None:
    """One ensemble's tracker BUFR for one cycle (model "ifs" or "aifs"), cached as {model}_ens_tf_{date}{hh}_{N}h.bufr;
    None if it is not on the mirror. The name carries the run's length N (240 h for IFS-ENS until May 2026, 360 h
    since), so it is looked up in the listing (run_hours reads it back)."""
    import cycles

    hh = f"{int(hh):02d}"
    have = [p for p in cache.glob(f"{model}_ens_tf_{date}{hh}_*h.bufr") if p.stat().st_size > 100]
    if have:
        return have[0]
    names = cycles._list_names(f"{date}/{hh}z/{MODEL_DIR[model]}/0p25/enfo/")
    tf = sorted(n for n in names or [] if re.search(r"-\d+h-enfo-tf\.bufr$", n))
    if not tf:
        report(f"{MODELS[model]} tracker file {date} {hh}Z not on the Google mirror", None if quiet else "warning")
        return None
    run = re.search(r"-(\d+)h-enfo-tf\.bufr$", tf[-1]).group(1)
    return get_bufr(
        f"{GOOGLE}/{tf[-1]}",
        cache / f"{model}_ens_tf_{date}{hh}_{run}h.bufr",
        f"{MODELS[model]} tracker file {date} {hh}Z",
        quiet,
    )


def run_hours(fp: Path) -> int:
    """The run length (h) of a tracker file cached by tracker_file."""
    return int(re.search(r"_(\d+)h\.bufr$", fp.name).group(1))


def decode_all(fp: Path, types=(1, 4)) -> list[dict]:
    """Every storm message with the subsets of the given ensembleForecastTypes: {id, name, named, obs, tracks: {member:
    track}}. Members 1-50 are the perturbed ones (type 4); the control (type 1, 51 in the file) and the deterministic run
    (type 0) are both member 0, so ask for one of them at a time. track = dict(steps, lat, lon(-180..180), pmsl hPa,
    wind m/s). obs = the observed (lat, lon) at t=0 of a storm with an official id, else None."""
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
                        continue
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
                obs = (float(o_la[0]), float((o_lo[0] + 180) % 360 - 180)) if o_la[0] > MISSING else None
                if tracks:
                    out.append(dict(id=sid, name=name, named=named, obs=obs, tracks=tracks))
            finally:
                ec.codes_release(h)
    return out


FETCHED: dict = {}  # bytes oper_fields has fetched in this process, per mirror (rangefetch.fetch_ranges stats)


def oper_fields(model: str, date: str, hh: str, step: int, pl=(), levels=(), sfc=(), resol: str = "0p25") -> dict:
    """One step of a deterministic run (model "ifs" = the IFS control, or "aifs-single"), byte-ranged from the Google mirror:
    {"lat" (ascending), "lon" (0..360), (param, level): field for each pl param on each level, param: field for each sfc
    param}. resol "0p1": the IFS control's 0.1-degree files. Raises RuntimeError when the step is not on the mirror or a
    field is missing."""
    import eccodes
    import rangefetch as rf

    hh = f"{int(hh):02d}"
    levtypes = (("pl",) if pl else ()) + (("sfc",) if sfc else ())
    try:
        files = rf.oper_files(date, hh, model, int(step), resol, levtypes)
    except rf.IndexUnavailable as e:
        raise RuntimeError(f"{model} {resol} {date} {hh}Z +{step} h: not on the Google mirror") from e
    sel = []
    for rel, idx in files:
        want = (
            [e for e in rf.select(idx, param=list(pl), levelist=list(levels)) if e.get("levtype") == "pl"] if pl else []
        )
        want += [e for e in rf.select(idx, param=list(sfc)) if e.get("levtype") == "sfc"] if sfc else []
        sel.append((rel, want))
    n = sum(len(want) for _, want in sel)
    if n != len(pl) * len(levels) + len(sfc):
        raise RuntimeError(f"{model} {resol} {date} {hh}Z +{step} h: {n} of the fields in the index")
    blob = b"".join(rf.fetch_ranges(rel, rf.coalesce(want), stats=FETCHED) for rel, want in sel if want)
    out = {}
    i = 0
    while (j := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[j + 8 : j + 16], "big")  # GRIB2 section 0: the message's total length
        g = eccodes.codes_new_from_message(blob[j : j + n])
        i = j + n
        try:
            nm, lt = eccodes.codes_get(g, "shortName"), eccodes.codes_get(g, "typeOfLevel")
            ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
            la0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
            dj = eccodes.codes_get(g, "jDirectionIncrementInDegrees")
            lo0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
            di = eccodes.codes_get(g, "iDirectionIncrementInDegrees")
            up = eccodes.codes_get(g, "jScansPositively")
            v = eccodes.codes_get_values(g).reshape(nj, ni).astype(np.float32)
            key = (nm, int(eccodes.codes_get(g, "level"))) if lt == "isobaricInhPa" else nm
        finally:
            eccodes.codes_release(g)
        lat = la0 + (dj if up else -dj) * np.arange(nj)
        if lat[0] > lat[-1]:
            v, lat = v[::-1], lat[::-1]
        lon = (lo0 + di * np.arange(ni)) % 360
        o = np.argsort(lon)
        out[key], out["lat"], out["lon"] = v[:, o], lat, lon[o]
    if any(k not in out for k in [(p, int(L)) for p in pl for L in levels] + list(sfc)):
        raise RuntimeError(f"{model} {date} {hh}Z +{step} h: a field was missing from the messages")
    return out


def _km(la1, lo1, la2, lo2) -> float:
    p1, p2, dl = np.deg2rad(la1), np.deg2rad(la2), np.deg2rad(lo2 - lo1)
    return float(
        A_EARTH / 1000 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
    )


def zonal_motion(s: dict) -> np.ndarray:
    """The storm's 24-h zonal motion (m/s, east positive) at each point of the track: the longitude change from 12 h
    before to 12 h after; NaN where the track has no position at either end."""
    st, la = np.asarray(s["steps"], int), s["lat"]
    lo = np.unwrap(np.deg2rad(s["lon"])) * 180 / np.pi
    pos = {int(h): i for i, h in enumerate(st)}
    u = np.full(len(st), np.nan)
    for k, h in enumerate(st):
        a, b = pos.get(int(h) - 12), pos.get(int(h) + 12)
        if a is not None and b is not None:
            u[k] = (lo[b] - lo[a]) * 111.2e3 * np.cos(np.deg2rad(la[k])) / (24 * 3600)
    return u


REC_WEST, REC_EAST = 0.0, 2.0  # m/s: the turn from westward or poleward motion to eastward motion
REC_HOLD_H = 24  # h: the eastward motion must last this long (or to the track's end)


def recurvature(s: dict) -> int | None:
    """Lead (h) where the storm's 24-h zonal motion turns from westward or poleward (<= REC_WEST) to lasting eastward
    motion (>= REC_EAST at consecutive points spanning REC_HOLD_H, or half that up to the end of the track): the last point at or
    below REC_WEST before that eastward run. None if it never turns (a storm that moves east from its first position has
    no turn). A westward threshold of -1 m/s missed storms that move north before turning east (14 of 50 AIFS-ENS
    members of Choi-Wan, 2 Oct 2026 00Z, forecast track and one day of history: 6-18 m/s eastward at 41-51N, never
    faster than 0.9 m/s westward); without the hold, a 6-18 h eastward jog at 17N made 6 of its members "recurve" on
    day 1."""
    st = s["steps"]
    if len(st) < 5:
        return None
    u = zonal_motion(s)
    fin = np.where(np.isfinite(u))[0]
    if not len(fin):
        return None
    k = 0
    while k < len(u):
        if not u[k] >= REC_EAST:
            k += 1
            continue
        j = k
        while j + 1 < len(u) and u[j + 1] >= REC_EAST and st[j + 1] - st[j] <= 12:
            j += 1
        west = np.where(u[:k] <= REC_WEST)[0]
        held = st[j] - st[k]
        if len(west) and (held >= REC_HOLD_H or (j == fin[-1] and held >= REC_HOLD_H / 2)):
            return int(st[west[-1]])
        k = j + 1
    return None


def at(s: dict, h: int):
    """(lat, lon) of the storm at lead h if the track has a position there, else None."""
    k = np.where(s["steps"] == h)[0]
    return (float(s["lat"][k[0]]), float(s["lon"][k[0]])) if len(k) else None


def irrotational(u2d, v2d, lat, lon):
    """(u_chi, v_chi) interpolated onto the target lat/lon grid (lat ascending, lon 0..360)."""
    import xarray as xr
    from wind200_vpot import irrotational_wind, velocity_potential

    chi, dlat, dlon = velocity_potential(u2d, v2d)
    uc, vc = irrotational_wind(chi, dlat, dlon)
    dlon_c = np.asarray(dlon, float)
    if dlon_c[-1] < 360.0 - 1e-6:  # close the circle so interpolation reaches 360
        dlon_c = np.r_[dlon_c, dlon_c[0] + 360.0]
        uc = np.concatenate([uc, uc[:, :1]], axis=1)
        vc = np.concatenate([vc, vc[:, :1]], axis=1)
    keep = np.r_[True, np.diff(dlat) != 0]  # DH2 lats are unique, but be safe
    dlat, uc, vc = np.asarray(dlat)[keep], uc[keep], vc[keep]
    U = xr.DataArray(uc, dims=("lat", "lon"), coords={"lat": dlat, "lon": dlon_c}).sortby("lat")
    V = xr.DataArray(vc, dims=("lat", "lon"), coords={"lat": dlat, "lon": dlon_c}).sortby("lat")
    tgt = dict(lat=xr.DataArray(lat, dims="y"), lon=xr.DataArray(lon % 360, dims="x"))
    return U.interp(**tgt).values, V.interp(**tgt).values


def pv_advection(uchi, vchi, pv_layer, lat, lon, sigma_deg=1.0):
    """-v_chi . grad(PV) in PVU/day on the (lat, lon) grid; PV in PVU, Gaussian-smoothed by sigma_deg first (the 0.25 deg
    PV gradient is grid-noisy; v_chi is already T106-smooth, so this matches the two scales)."""
    from scipy.ndimage import gaussian_filter

    n = sigma_deg / abs(lat[1] - lat[0])
    pv_layer = gaussian_filter(np.nan_to_num(pv_layer, nan=0.0), sigma=(n, n), mode=("nearest", "wrap"))
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.clip(np.cos(phi), 1e-3, None)[:, None]
    dlam = lam[1] - lam[0]
    dpdx = (np.roll(pv_layer, -1, -1) - np.roll(pv_layer, 1, -1)) / (2 * dlam) / (A_EARTH * cosphi)
    dpdy = np.gradient(pv_layer, phi, axis=0) / A_EARTH
    return -(uchi * dpdx + vchi * dpdy) * 86400.0


def within(lat, lon, clat, clon, r_km) -> np.ndarray:
    """Boolean (lat, lon) mask of points within r_km of (clat, clon)."""
    la = np.deg2rad(lat)[:, None]
    lo = np.deg2rad(lon)[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    return cosd >= np.cos(r_km / (A_EARTH / 1000))


# 12 colours: a busy chart can carry 9 or more storms, and a shorter cycle would give two of them the same colour
COLS = [
    "#c0392b",
    "#1f5fa8",
    "#23874f",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#7f8c8d",
    "#b7950b",
    "#e84393",
    "#2c3e50",
    "#7cb342",
    "#6d4c41",
]


def date_lab(init, d, fmt="two"):
    """Valid-time label for forecast day d.
    two: 'Tue 6 Oct\nday 2'; one: 'Tue 6 Oct (day 2)'; short: '6 Oct'. The hour is shown when the run is not a 00Z one."""
    import pandas as pd

    t = pd.Timestamp(init) + pd.Timedelta(hours=round(float(d) * 24))
    hh = "" if t.hour == 0 else f" {t:%H}Z"
    if fmt == "short":
        return f"{t.day} {t:%b}{hh}"
    if fmt == "stack":
        return f"{t.day} {t:%b}{hh}\nday {float(d):g}"
    if fmt == "one":
        return f"{t:%a} {t.day} {t:%b}{hh} (day {float(d):g})"
    return f"{t:%a} {t.day} {t:%b}{hh}\nday {float(d):g}"


def date_axis(ax, init, dmax, axis="x", step=None, fmt=None):
    """Integer forecast days on an axis, labelled with their valid dates."""
    step = step or (1 if dmax <= 8 else 2)
    ticks = np.arange(0, float(dmax) + 1e-6, step)
    labs = [date_lab(init, t, fmt or ("two" if axis == "x" else "one")) for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, fontsize=8.6)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=8.6)


# ── following the storm after the tracker's last fix ──────────────────────────────────────────────────────────────────
# ECMWF's tracker stops at (or soon after) extratropical transition, which is exactly when a recurving storm phases with
# a mid-latitude trough and can deepen explosively (Choi-Wan 2026-10-02 00Z: dropped at +120 h near 40N 153E, 979 hPa;
# the control's low is 950 hPa near 57N 170E 24 h later, and 36 of 50 members are below 960 hPa at day 6). follow_low
# continues the track as the surface-pressure minimum, conservatively, so a different low is never picked up.
FOLLOW_REACH_KM = 1800.0  # per 12 h: a transitioning storm's centre can re-form far downstream along the jet
FOLLOW_SLOW_KM = 900.0  # a step longer than this must point roughly along the previous motion ...
FOLLOW_CONE_DEG = 75.0  # ... within this angle
FOLLOW_MIN_DEPTH = 2.0  # hPa below the 400-600 km ring mean: a closed low
FOLLOW_OTHER_KM = 500.0  # a low that existed 12 h earlier within this distance of a candidate (and was not our storm)
# makes that candidate a different system
FOLLOW_SAME_KM = 900.0  # a low within this distance of our storm at the previous step is part of the same system (a
# phasing storm is often double-centred for a step), so it never blocks a candidate
FOLLOW_MIDLAT = 30.0  # long jumps (re-forming along the jet) only north of this latitude; in the tropics a step is
# limited to FOLLOW_SLOW_KM
FOLLOW_DECAY_HPA = 15.0  # once the low has filled this much from its deepest, long jumps stop (a decaying low does not
# re-form downstream; a long hop then lands on another system)
FOLLOW_FILLED_HPA = (
    1010.0  # a low that has filled to this central pressure is no longer followed (over high terrain the
)
# sea-level reduction makes weak "lows" that are not storms)
FOLLOW_MERGE_KM = 1500.0  # merge=True: when the storm's own low is gone, a low that existed a step earlier within this
# distance of it, north of FOLLOW_MIDLAT, along the motion and within reach, has absorbed it
# (extratropical transition by merger: Choi-Wan 2026-10-02 12Z ran into a 988 hPa low 1,300 km
# downstream at +108 h, which became 943 hPa by +132 h)
# Without the three rules below the follower walks weak lows onto the continents (12 Oct 12Z: Koguma, dropped by the
# tracker at +66 h at 995 hPa, would be carried as a 998-1006 hPa low across the Pacific, then hop 1,061 km from eastern
# Washington into the Colorado lee lows and sit there to day 10; Simon, dropped at 1007 hPa off Sinaloa, would be followed
# into the Sonoran heat low). The rules still reach the deepest point of every
# Choi-Wan run (2 Oct 00Z / 12Z, 4 Oct 00Z: 929-948 hPa, 18-36 h after the tracker's last fix).
FOLLOW_MAX_H = 72  # no continuation more than this long after the tracker's last fix: the re-intensification after
# transition comes within 1-2 days, after that the low is an ordinary extratropical cyclone
# (tc_ens's system window SYS_H is the same 72 h)
FOLLOW_WEAK_HPA = 1000.0  # a low at or above this central pressure is followed only over the sea north of FOLLOW_MIDLAT
# (over land it is a heat or lee low; in the tropics, where the tracker dropped a weak storm, a
# dissipating remnant: Simon), and is never the target of a long jump (> FOLLOW_SLOW_KM: only a
# real cyclone re-forms downstream along the jet)
FOLLOW_H_MAX = 240  # every caller follows to day 10 at most (track ensembles and depth, maps, phase space)


class LandMask:
    """is_land(lat, lon) from a land-sea mask on a regular grid (nearest point; lat ascending, lon 0..360)."""

    def __init__(self, land, lat, lon):
        self.land, self.lat0, self.dlat = np.asarray(land, bool), float(lat[0]), float(lat[1] - lat[0])
        self.lon0, self.dlon, self.nlon = float(lon[0]), float(lon[1] - lon[0]), len(lon)

    def __call__(self, lat, lon) -> bool:
        j = int(np.clip(round((lat - self.lat0) / self.dlat), 0, self.land.shape[0] - 1))
        i = int(np.rint(((lon % 360.0) - self.lon0) / self.dlon)) % self.nlon
        return bool(self.land[j, i])


def load_land(date: str, hh: str) -> LandMask | None:
    """The IFS control's land-sea mask (step 0, one 0.25-degree field byte-ranged from the Google mirror) for follow_low's
    land rule; None if it cannot be had (follow_low then treats every weak low as over land)."""
    try:
        f = oper_fields("ifs", date, hh, 0, sfc=("lsm",))
        return LandMask(f["lsm"] >= 0.5, f["lat"], f["lon"])
    except Exception as e:  # noqa: BLE001  never let the mask take a product down
        print(f"  land-sea mask unavailable ({e}); weak lows are not followed anywhere", flush=True)
        return None


def _gc_km(lat, lon, clat, clon):
    """Great-circle distance (km) from (clat, clon) to every point of a (lat, lon) grid (1-D axes)."""
    la = np.deg2rad(np.asarray(lat))[:, None]
    lo = np.deg2rad(np.asarray(lon))[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    return A_EARTH / 1000 * np.arccos(np.clip(cosd, -1, 1))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return float(
        np.rad2deg(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    )


def closed_lows(f, lat, lon, centre, radius_km, smooth_deg=0.5, sep_deg=2.5):
    """Closed lows in field f (hPa, lat ascending, lon 0..360) within radius_km of centre=(lat, lon):
    list of (lat, lon(-180..180), central pressure, depth). A low = a local minimum of the lightly smoothed field over a
    sep_deg box whose raw central pressure (min within 120 km) is >= FOLLOW_MIN_DEPTH below the 400-600 km ring mean."""
    from scipy.ndimage import gaussian_filter, minimum_filter

    n = abs(lat[1] - lat[0])
    lon360 = np.asarray(lon) % 360
    clat, clon = centre
    dl = radius_km / 111.0 + 7.0
    j = np.where(np.abs(lat - clat) <= dl)[0]
    dlon = ((lon360 - (clon % 360) + 180) % 360) - 180
    i = np.where(np.abs(dlon) <= dl / max(np.cos(np.deg2rad(min(abs(clat) + dl, 85))), 0.1))[0]
    if not len(j) or not len(i):
        return []
    sub = np.nan_to_num(f[np.ix_(j, i)], nan=1013.0)
    fs = gaussian_filter(sub, sigma=smooth_deg / n, mode="nearest")
    mn = minimum_filter(fs, size=max(3, int(np.rint(sep_deg / n))) | 1, mode="nearest")
    out = []
    la_s, lo_s = lat[j], lon360[i]
    dist = _gc_km(la_s, lo_s, clat, clon % 360)
    for a, b in np.argwhere((fs == mn) & (dist <= radius_km)):
        if a in (0, len(j) - 1) or b in (0, len(i) - 1):
            continue  # on the window edge: not resolved
        c_la, c_lo = float(la_s[a]), float(lo_s[b])
        dd = _gc_km(la_s, lo_s, c_la, c_lo)
        ring, core = (dd >= 400) & (dd <= 600), dd <= 120
        if not ring.any() or not core.any():
            continue
        pmin = float(sub[core].min())
        depth = float(sub[ring].mean() - pmin)
        if depth >= FOLLOW_MIN_DEPTH:
            out.append((c_la, ((c_lo + 180) % 360) - 180, pmin, depth))
    return out


def _step_to(lat0, lon0, dlat, dlon_deg):
    return lat0 + dlat, ((lon0 + dlon_deg + 180) % 360) - 180


def follow_low(msl, lat, lon, steps_h, track, h_max=FOLLOW_H_MAX, merge=False, land=None):
    """Continue `track` (dict steps/lat/lon[/pmsl]) beyond its last fix as a surface low, every field step.

    msl: (len(steps_h), nlat, nlon) hPa, lat ascending, lon 0..360; steps_h: lead hours (12-hourly). Starts at the
    tracker's LAST fix (6-hourly, so the first field step can be 6 h on) and goes to h_max, at most FOLLOW_MAX_H past that
    fix. Each step: the candidates are the closed lows within reach of the previous position (FOLLOW_SLOW_KM of motion per
    12 h, scaled with the step, plus FOLLOW_REACH_KM - FOLLOW_SLOW_KM for a centre re-forming downstream); a candidate
    further than FOLLOW_SLOW_KM (scaled) must lie north of FOLLOW_MIDLAT, within FOLLOW_CONE_DEG of the previous motion and
    be deeper than FOLLOW_WEAK_HPA, and the low must not have filled FOLLOW_DECAY_HPA (a cyclone re-forming downstream
    along the jet, as in a phasing transition, is allowed; a jump sideways or backwards, or onto a weak low, is not); a weak
    candidate (>= FOLLOW_WEAK_HPA) is excluded over land and south of FOLLOW_MIDLAT (land: a LandMask; None = treat every
    point as land, i.e. weak lows are never followed); a candidate that is the continuation of ANOTHER low already present
    a field step earlier (within FOLLOW_OTHER_KM of it, and closer to it than our storm was) is excluded. The survivor
    nearest to the persistence first guess (previous position + previous displacement) is taken; none = the end of the
    track.
    merge=True: if no candidate survives but one was excluded only as another system's continuation, and that system lay
    within FOLLOW_MERGE_KM of our storm a step earlier, both north of FOLLOW_MIDLAT, the storm has merged into it: the
    track continues with the merged low (flagged in `merged` from that step on).
    Returns the CONTINUATION only: dict(steps, lat, lon, pmsl, merged), every step after the tracker's last fix."""
    steps_h = np.asarray(steps_h, int)
    empty = dict(
        steps=np.array([], int), lat=np.array([]), lon=np.array([]), pmsl=np.array([]), merged=np.array([], bool)
    )
    st = np.asarray(track["steps"], int)
    if not len(st):
        return empty
    # start at the tracker's LAST fix, not at the last fix on a (12-hourly) field step: continuing a track that ends at
    # +66 h from +60 h would make extended() run back in time (36 42 48 54 66 60 72), and the first hop, measured from the
    # earlier fix, could reach a low that is out of range of the real last position
    h0 = int(st[-1])
    k0 = len(st) - 1
    la, lo = float(track["lat"][k0]), float(track["lon"][k0])
    p_last = float(np.asarray(track["pmsl"], float)[-1]) if "pmsl" in track else np.nan
    prev = np.where(st <= h0 - 12)[0]
    if len(prev):
        kp = prev[-1]
        dt = (h0 - int(st[kp])) / 12.0
        v_lat = (la - float(track["lat"][kp])) / dt
        v_lon = (((lo - float(track["lon"][kp]) + 180) % 360) - 180) / dt
    else:
        v_lat = v_lon = 0.0
    h_stop = h0 + FOLLOW_MAX_H if h_max is None else min(int(h_max), h0 + FOLLOW_MAX_H)
    out = {"steps": [], "lat": [], "lon": [], "pmsl": [], "merged": []}
    h_last = h0
    merged = False
    for k in np.where(steps_h > h0)[0]:
        h = int(steps_h[k])
        if h > h_stop:
            break
        dt12 = (h - h_last) / 12.0
        # reach = the storm's own motion (FOLLOW_SLOW_KM per 12 h, scaled with the step) plus the offset of a centre
        # re-forming downstream (the rest of FOLLOW_REACH_KM, not scaled): 1,800 km in 12 h, 1,350 km in the 6 h after a
        # last fix between field steps (Choi-Wan 2 Oct 12Z: last fix +114 h, merges at +120 h into a low 929 km
        # downstream; a linear 900 km loses it, an unscaled 1,800 km lets 2 Oct 00Z member 50 hop 1,604 km NNW in 6 h)
        reach = FOLLOW_SLOW_KM * dt12 + (FOLLOW_REACH_KM - FOLLOW_SLOW_KM)
        cands = closed_lows(msl[k], lat, lon, (la, lo), reach)
        if not cands:
            break
        mot = _bearing(la, lo, *_step_to(la, lo, v_lat, v_lon)) if (abs(v_lat) + abs(v_lon)) > 0.2 else None
        # the lows of the previous FIELD step (12 h before this one), with our storm's position at that time: the last fix
        # can fall between field steps (+66 h), and the tracker's own fix at the field step is then the right reference
        kb = np.where(steps_h <= h_last)[0]
        if len(kb):
            hb = int(steps_h[kb[-1]])
            pb = at(track, hb) if hb < h0 else None
            o_la, o_lo = pb if pb is not None else (la, lo)
            before = closed_lows(msl[kb[-1]], lat, lon, (o_la, o_lo), reach + FOLLOW_OTHER_KM)
        else:
            before, o_la, o_lo = [], la, lo
        g_la, g_lo = _step_to(la, lo, v_lat * dt12, v_lon * dt12)
        # decay guard: filled FOLLOW_DECAY_HPA from the deepest of the followed lows and the tracker's last fix
        ref = [p for p in [p_last] + out["pmsl"] if np.isfinite(p)]
        p_now = out["pmsl"][-1] if out["pmsl"] else p_last
        decayed = bool(ref) and np.isfinite(p_now) and p_now - min(ref) >= FOLLOW_DECAY_HPA
        best, bd = None, np.inf
        mbest, mbd = None, np.inf  # merger candidates (merge=True only)
        for c_la, c_lo, pmin, _depth in cands:
            dist = _km(la, lo, c_la, c_lo)
            if pmin >= FOLLOW_FILLED_HPA:
                continue
            if pmin >= FOLLOW_WEAK_HPA and (c_la < FOLLOW_MIDLAT or land is None or land(c_la, c_lo)):
                continue  # weak: over land a heat / lee low, in the
                # tropics a dissipating remnant
            if dist > FOLLOW_SLOW_KM * dt12:
                if mot is None or min(la, c_la) < FOLLOW_MIDLAT or decayed or pmin >= FOLLOW_WEAK_HPA:
                    continue
                dang = abs(((_bearing(la, lo, c_la, c_lo) - mot + 180) % 360) - 180)
                if dang > FOLLOW_CONE_DEG:
                    continue
            other, absorber = False, False
            for b_la, b_lo, _, _ in before:
                d_ours = _km(b_la, b_lo, o_la, o_lo)
                if d_ours < FOLLOW_SAME_KM:
                    continue  # our own storm (or its second centre) a step earlier
                dbc = _km(b_la, b_lo, c_la, c_lo)
                if dbc < FOLLOW_OTHER_KM and dbc < dist:
                    other = True
                    absorber = d_ours <= FOLLOW_MERGE_KM and min(la, c_la, b_la) >= FOLLOW_MIDLAT
                    break
            dg = _km(g_la, g_lo, c_la, c_lo)
            if other:
                if merge and absorber and dg < mbd:
                    mbest, mbd = (c_la, c_lo, pmin), dg
                continue
            if dg < bd:
                best, bd = (c_la, c_lo, pmin), dg
        if best is None and mbest is not None:
            best, merged = mbest, True  # absorbed by a pre-existing low
        if best is None:
            break
        c_la, c_lo, pmin = best
        v_lat = (c_la - la) / dt12
        v_lon = (((c_lo - lo + 180) % 360) - 180) / dt12
        la, lo, h_last = c_la, c_lo, h
        out["steps"].append(h)
        out["lat"].append(la)
        out["lon"].append(lo)
        out["pmsl"].append(round(pmin, 1))
        out["merged"].append(merged)
    typ = {"steps": int, "merged": bool}
    return {k: np.array(v, typ.get(k, float)) for k, v in out.items()} if out["steps"] else empty


def extended(track, cont):
    """The tracker's track followed by its continuation, as one track dict (pmsl NaN where the tracker had none).
    Continuation steps at or before the tracker's last fix are dropped, so the steps always increase."""
    if cont is None or not len(cont["steps"]):
        return track
    keep = np.asarray(cont["steps"]) > int(np.max(track["steps"]))
    if not keep.all():
        cont = {
            k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in cont.items()
        }
        if not len(cont["steps"]):
            return track
    p = track.get("pmsl", np.full(len(track["steps"]), np.nan))
    out = dict(
        steps=np.r_[track["steps"], cont["steps"]].astype(int),
        lat=np.r_[track["lat"], cont["lat"]],
        lon=np.r_[track["lon"], cont["lon"]],
        pmsl=np.r_[p, cont["pmsl"]],
        followed=np.r_[np.zeros(len(track["steps"]), bool), np.ones(len(cont["steps"]), bool)],
    )
    if "wind" in track:  # the tracker's wind; none for the followed low
        out["wind"] = np.r_[track["wind"], np.full(len(cont["steps"]), np.nan)]
    if "merged" in cont:  # from the step the storm was absorbed by a pre-existing low
        out["merged"] = np.r_[np.zeros(len(track["steps"]), bool), np.asarray(cont["merged"], bool)]
    return out


# ── map helper ───────────────────────────────────────────────────────────────────────────────────────────────────────
# A 160-200-degree-wide mid-latitude sector on a flat lat-lon grid squeezed to fit ("auto" aspect) flattens Siberia,
# Alaska and Canada into slivers. sector_axes() draws a sector the way weather maps do: Lambert
# conformal with standard parallels inside the band, the outline following the sector's own meridians and latitude
# circles (a fan, no empty corners), true aspect.
def lon_median(lons) -> float:
    """Median longitude (-180..180) that is safe across the dateline and 0 deg: the median offset from the circular mean.
    (A plain np.median of longitudes straddling the dateline can sit far from the cluster: 168.2E for a genesis cluster
    whose circular centre is 177.6E.)"""
    lo = np.asarray(lons, float)
    a = np.deg2rad(lo)
    c = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    return float((c + np.median((lo - c + 180) % 360 - 180) + 180) % 360 - 180)


def track_sector(lats, lons):
    """Sector bounds (lon0, lon1 in a continuous 0..360+ frame, lat0, lat1) around a cloud of track positions: the 1st to
    99th percentiles with a margin (one stray member does not set the frame), at least 40 by 25 degrees."""
    lo = np.asarray(lons, float)
    c = lon_median(lo)
    rel = (lo - c + 180) % 360 - 180
    lon0, lon1 = c + np.percentile(rel, 1) - 6, c + np.percentile(rel, 99) + 6
    if lon1 - lon0 < 40:
        m = (lon0 + lon1) / 2
        lon0, lon1 = m - 20, m + 20
    la = np.asarray(lats, float)
    lat0, lat1 = max(-5.0, np.percentile(la, 1) - 4), min(80.0, np.percentile(la, 99) + 5)
    if lat1 - lat0 < 25:
        lat1 = min(80.0, lat0 + 25)
    return lon0 % 360, lon0 % 360 + (lon1 - lon0), lat0, lat1


def sector_projection(lon0, lon1, lat0, lat1):
    import cartopy.crs as ccrs

    clon = (((lon0 + lon1) / 2.0) + 180.0) % 360.0 - 180.0
    span = lat1 - lat0
    return ccrs.LambertConformal(
        central_longitude=clon,
        central_latitude=(lat0 + lat1) / 2.0,
        standard_parallels=(lat0 + span / 4.0, lat1 - span / 4.0),
        cutoff=max(lat0 - 5, -10),
    )


def sector_aspect(lon0, lon1, lat0, lat1, n=90):
    """height / width of the sector's outline in the projection (for laying out axes before drawing)."""
    import cartopy.crs as ccrs

    proj = sector_projection(lon0, lon1, lat0, lat1)
    lo = np.r_[np.linspace(lon0, lon1, n), np.linspace(lon1, lon0, n)]
    la = np.r_[np.full(n, lat0), np.full(n, lat1)]
    xy = proj.transform_points(ccrs.PlateCarree(), lo, la)
    return float((xy[:, 1].max() - xy[:, 1].min()) / (xy[:, 0].max() - xy[:, 0].min()))


def sector_axes(fig, rect, lon0, lon1, lat0, lat1, land=True, coast_res="50m", n=90):
    """A GeoAxes at rect=[x, y, w, h] (figure fraction) showing the sector lon0..lon1 (0..360, may exceed 360), lat0..lat1
    as a Lambert fan: true aspect, the outline along the sector's edges, coastlines, borders and light land."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.path as mpath

    proj = sector_projection(lon0, lon1, lat0, lat1)
    ax = fig.add_axes(rect, projection=proj)
    lo = np.r_[np.linspace(lon0, lon1, n), np.linspace(lon1, lon0, n), lon0]
    la = np.r_[np.full(n, lat0), np.full(n, lat1), lat0]
    xy = proj.transform_points(ccrs.PlateCarree(), lo, la)[:, :2]
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_boundary(mpath.Path(xy), transform=ax.transData)
    if land:
        ax.add_feature(cfeature.LAND, facecolor="#f2f1ec", zorder=0)
    ax.coastlines(coast_res, lw=0.6, color="#333")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#777")
    ax.gridlines(lw=0.3, color="#999", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(0, 91, 15))
    return ax


def fit_text(fig, text, width_in: float) -> None:
    """Shrink a figure text's font until it is at most width_in inches wide (a title with a long storm name and the grid
    would otherwise run off a narrow map)."""
    w = text.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    if w > width_in:
        text.set_fontsize(text.get_fontsize() * width_in / w)
