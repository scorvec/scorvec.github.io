#!/usr/bin/env python3
"""GEFS and GEPS for the days 1-15 stratosphere products: one shared fetch per model per cycle, reduced on the way in
to exactly what each product reads, so no product downloads a field another has already pulled.

THE MODELS
  gefs  NOAA GEFS, control gec00 + 30 perturbed members, pgrb2a 0.5 deg, every message byte-ranged from NOAA's S3
        bucket (noaa-gefs-pds) through its .idx file - the site rule for NOAA models. Every cycle reaches 384 h.
  geps  ECCC GEPS, control + 20 perturbed members, 0.5 deg, the MSC Datamart's raw `allmbrs` GRIBs (one file per
        variable, level and step, all 21 members in it) through scripts/geps/datamart.py (the site's honest User-Agent,
        retries, the fetch ledger), cached under strat_maps.DL with the names nh_vortex and strat_maps use, so a job
        that runs both downloads each file once. The 00Z and 12Z cycles reach 384 h. Its isobaric levels stop at 10 hPa
        and carry only 50 and 10 hPa above 100 hPa (index probed 2026-10-20: HGT at 11 levels 1000..10 hPa, TMP at 10,
        UGRD/VGRD at 12), which is why the E-P flux and the upper-stratosphere products have no GEPS: the flux's
        divergence needs the stratosphere resolved in the vertical, and the upper stratosphere lies above 10 hPa.

THE PARTS (fetch only what the caller needs; the job that runs every product fetches them all at once)
  vortex  daily 0-384 h      UGRD 10/100, HGT 100     u60 (member, lead, 2) zonal-mean u at 60N, 10 and 100 hPa;
                                                       zcap (member, lead) 65-90N height at 100 hPa       nh_vortex
  drip    daily 0-360 h      HGT at the 11 drip levels caps (member, lead, level) 65-90N height  forecast_drip --cache
  flux    daily 0-360 h      VGRD 100, TMP 100        flux_tot/_k1/_k2 (member, lead, lat) 100 hPa [v'T'] by latitude
                                                       (heatflux100.fluxes: per member, waves 1-72)   heatflux100_ext
                                                       flux_sib/_can (member, lead) v*T* over Siberia / Canada
                                                       (heatflux100.regional: the regional reflection index)
  waves   daily 0-360 h      HGT 500/100/10           zwave (member, lead, level, lat, lon) on 1 deg       wave_maps
  winds   12-hourly 0-360 h  UGRD, VGRD, HGT 10/100   wind (lead, level, u|v|gh, lat, lon) ensemble mean, 0.5 deg,
                                                       step 0 the CONTROL (the analysis)                 vortex_winds
Members are kept whole only on 1 deg and only for the wave maps; everything else is reduced as each step arrives.
The reductions of one cycle sit in data/strat_ens/{model}_{YYYYMMDDHH}.npz (control = member 0).

    python strat_ens.py fetch --model gefs --date 20261020 --time 00   # every part
    python strat_ens.py fetch --model geps --date 20261020 --time 00 --parts flux,drip
    python strat_ens.py fetch --model geps --date 20261021 --time 12 --skip-shown   # nothing when main shows it
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import geosfp_cap as FP

DATA = HERE / "data" / "strat_ens"
MODELS = {"gefs": "GEFS", "geps": "GEPS"}
CREDIT = {
    "gefs": "NOAA GEFS (AWS Open Data, public domain), control + 30 members, 0.5°",
    "geps": "ECCC GEPS (MSC Datamart, ECCC Data Servers End-use Licence), control + 20 members, 0.5°",
}
LEV_DRIP = tuple(int(x) for x in FP.LEVELS)  # 1000 ... 10 hPa, the drip's 11 levels
LEV_WAVE = (500, 100, 10)  # wave_maps.LEVELS
LEV_WIND = (10, 100)  # vortex_winds.LEVELS
LEADS_DAILY = tuple(range(0, 361, 24))  # the days 1-15 products: analysis + 15 days
LEADS_VORTEX = tuple(range(0, 385, 24))  # nh_vortex shows every ensemble to its end (day 16)
LEADS_12H = tuple(range(0, 361, 12))  # the vortex-wind loop
PARTS = ("vortex", "drip", "flux", "waves", "winds")
# the arrays each part keeps (the last one decides which members are complete) and the members a part needs
PART_KEYS = {
    "vortex": ("u60", "zcap"),
    "drip": ("caps",),
    "flux": ("flux_sib", "flux_can", "flux_tot", "flux_k1", "flux_k2"),
    "waves": ("zwave",),
    "winds": ("wind_n", "wind"),
}
FLOOR = {"gefs": 16, "geps": 11}  # of 31 and 21
LAT = np.arange(90.0, -90.0 - 1e-6, -0.5)  # both models' 0.5 deg grid, north to south
LON = np.arange(0.0, 360.0 - 1e-6, 0.5)  # from 0 E
GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]


def need(parts, L: int) -> set[tuple[str, int]]:
    """{(variable, level hPa)} the given parts read at lead L hours."""
    out = set()
    if "vortex" in parts and L in LEADS_VORTEX:
        out |= {("UGRD", 10), ("UGRD", 100), ("HGT", 100)}
    if "drip" in parts and L in LEADS_DAILY:
        out |= {("HGT", lev) for lev in LEV_DRIP}
    if "flux" in parts and L in LEADS_DAILY:
        out |= {("VGRD", 100), ("TMP", 100)}
    if "waves" in parts and L in LEADS_DAILY:
        out |= {("HGT", lev) for lev in LEV_WAVE}
    if "winds" in parts and L in LEADS_12H:
        out |= {(v, lev) for v in ("UGRD", "VGRD", "HGT") for lev in LEV_WIND}
    return out


def leads_for(parts) -> list[int]:
    return sorted({L for L in (*LEADS_VORTEX, *LEADS_12H) if need(parts, L)})


def path(model: str, date: str, hh: str) -> Path:
    return DATA / f"{model}_{date}{hh}.npz"


# ---------------------------------------------------------------------------------------------------------- GEFS ---
def _gefs_lib():
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_reforecast as R  # .idx parsing, byte ranges, decoding (north-to-south)

    return R


def gefs_url(date: str, hh: str, mem: str, L: int) -> str:
    return f"{GEFS_S3}/gefs.{date}/{hh}/atmos/pgrb2ap5/{mem}.t{hh}z.pgrb2a.0p50.f{L:03d}"


def gefs_ready(date: str, hh: str, last_h: int = 384) -> bool:
    """True when the last perturbed member's last step is on S3."""
    try:
        _gefs_lib().get(gefs_url(date, hh, GEFS_MEMBERS[-1], last_h) + ".idx")
        return True
    except Exception:  # noqa: BLE001
        return False


def gefs_step(date: str, hh: str, L: int, want: set, workers: int = 32) -> dict:
    """{member index: {(var, lev): (lat, lon) float32}} for one lead; a member missing any field is left out."""
    R = _gefs_lib()
    step = "anl" if L == 0 else f"{L} hour fcst"

    def one(mi):
        url = gefs_url(date, hh, GEFS_MEMBERS[mi], L)
        try:
            idx = R.index(url)
        except Exception:  # noqa: BLE001  (not on S3: the member is left out of this step)
            return mi, None
        out = {}
        for var, lev in want:
            hit = [e for e in idx if e[4] == var and e[2] == f"{lev} mb" and e[3] == step]
            if not hit:
                return mi, None
            out[(var, lev)] = R.decode(R.get(url, (hit[0][0], hit[0][1]))).astype("float32")
        return mi, out

    with ThreadPoolExecutor(workers) as ex:
        return {mi: o for mi, o in ex.map(one, range(len(GEFS_MEMBERS))) if o is not None}


# ---------------------------------------------------------------------------------------------------------- GEPS ---
def geps_ready(date: str, hh: str, last_h: int = 384) -> bool:
    import strat_maps as sm

    return sm.datamart.exists(sm.GEPS.format(d=date, c=hh, v="HGT", lev=10, L=last_h))


def _messages(f: Path) -> dict:
    """{member: (lat, lon) float32 on LAT x LON} from one allmbrs GRIB (control = perturbationNumber 0)."""
    import eccodes

    out = {}
    with open(f, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
                v = eccodes.codes_get_values(gid).reshape(nj, ni).astype("float32")
                la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                try:
                    num = int(eccodes.codes_get(gid, "perturbationNumber"))
                except Exception:  # noqa: BLE001  (a control message may carry none)
                    num = 0
            finally:
                eccodes.codes_release(gid)
            if (nj, ni) != (LAT.size, LON.size):
                raise ValueError(f"{f.name}: grid {nj}x{ni}, expected {LAT.size}x{LON.size}")
            if la0 < 0:  # south to north: flip to 90 -> -90
                v = v[::-1]
            out[num] = np.roll(v, round(lo0 / 0.5), axis=-1)  # column j at j x 0.5 deg E
    return out


def geps_step(date: str, hh: str, L: int, want: set) -> dict:
    """{member: {(var, lev): (lat, lon)}} for one lead from the Datamart files (cached in strat_maps.DL); empty when
    any file of the step is missing."""
    import strat_maps as sm

    def one(vl):
        f = sm.fetch(date, hh, vl[0], vl[1], L)
        return vl, (_messages(f) if f is not None else None)

    per = {}
    with ThreadPoolExecutor(sm.datamart.WORKERS) as ex:  # polite: the Datamart is a shared public service
        for vl, msgs in ex.map(one, sorted(want)):
            if msgs is None:
                return {}
            for num, v in msgs.items():
                per.setdefault(num, {})[vl] = v
    return {m: d for m, d in per.items() if len(d) == len(want)}


# -------------------------------------------------------------------------------------------------------- reduce ---
def cap(field: np.ndarray) -> float:
    """65-90N cos-weighted mean of the zonal mean (forecast_drip.cap_of, nh_vortex.cap_mean)."""
    m = LAT >= 65.0
    w = np.cos(np.deg2rad(LAT[m]))
    return float((np.nanmean(field[m], axis=-1) * w).sum() / w.sum())


def _empty(key: str, nmem: int) -> np.ndarray:
    if key == "wind_n":
        return np.zeros(len(LEADS_12H), "int16")
    if key in ("flux_sib", "flux_can"):
        return np.full((nmem, len(LEADS_DAILY)), np.nan, "float32")
    if key.startswith("flux_"):
        return np.full((nmem, len(LEADS_DAILY), LAT.size), np.nan, "float32")
    shape = {
        "u60": (nmem, len(LEADS_VORTEX), 2),
        "zcap": (nmem, len(LEADS_VORTEX)),
        "caps": (nmem, len(LEADS_DAILY), len(LEV_DRIP)),
        "zwave": (nmem, len(LEADS_DAILY), len(LEV_WAVE), LAT.size // 2 + 1, LON.size // 2),
        "wind": (len(LEADS_12H), len(LEV_WIND), 3, LAT.size, LON.size),
    }[key]
    return np.full(shape, np.nan, "float32")


def fetch(model: str, date: str, hh: str, parts=PARTS, workers: int = 32) -> Path:
    """Fetch and reduce the parts not yet in this cycle's file; returns the file. Raises when a step has no member
    complete, or a part ends with fewer complete members than FLOOR."""
    import xarray as xr

    import heatflux100 as HF

    dest = path(model, date, hh)
    have = {}
    if dest.exists():
        with np.load(dest) as z:
            have = {k: z[k] for k in z.files}
    todo = [p for p in parts if f"done_{p}" not in have]
    if not todo:
        return dest
    nmem = len(GEFS_MEMBERS) if model == "gefs" else 21
    acc = {k: _empty(k, nmem) for p in todo for k in PART_KEYS[p]}
    j60 = int(np.argmin(np.abs(LAT - 60.0)))
    t0, nbytes = time.time(), 0
    for L in leads_for(todo):
        want = need(todo, L)
        got = gefs_step(date, hh, L, want, workers) if model == "gefs" else geps_step(date, hh, L, want)
        if not got:
            raise SystemExit(f"{MODELS[model]} {date} {hh}Z: no member complete at +{L} h")
        ms = sorted(got)
        nbytes += sum(a.nbytes for d in got.values() for a in d.values())
        if "vortex" in todo and L in LEADS_VORTEX:
            i = LEADS_VORTEX.index(L)
            for m in ms:
                acc["u60"][m, i] = [got[m][("UGRD", lev)][j60].mean() for lev in (10, 100)]
                acc["zcap"][m, i] = cap(got[m][("HGT", 100)])
        if L in LEADS_DAILY:
            i = LEADS_DAILY.index(L)
            if "drip" in todo:
                for m in ms:
                    acc["caps"][m, i] = [cap(got[m][("HGT", lev)]) for lev in LEV_DRIP]
            if "waves" in todo:
                for m in ms:
                    acc["zwave"][m, i] = [got[m][("HGT", lev)][::2, ::2] for lev in LEV_WAVE]
            if "flux" in todo:
                vt = [
                    xr.DataArray(
                        np.stack([got[m][(var, 100)] for m in ms])[:, None],
                        dims=("number", "step", "latitude", "longitude"),
                        coords={"latitude": LAT},
                    )
                    for var in ("VGRD", "TMP")
                ]
                _, f = HF.fluxes(*vt)  # per member, waves 1-72
                for k in ("tot", "k1", "k2"):
                    acc[f"flux_{k}"][ms, i] = f[k][:, 0]
                reg = HF.regional(LAT, LON, vt[0].values[:, 0], vt[1].values[:, 0])  # Siberia and Canada v*T*
                for k in HF.REGIONS:
                    acc[f"flux_{k}"][ms, i] = reg[k]
        if "winds" in todo and L in LEADS_12H:
            i = LEADS_12H.index(L)
            if L == 0 and 0 not in got:
                raise SystemExit(f"{MODELS[model]} {date} {hh}Z: no control at the analysis step")
            use = [0] if L == 0 else ms  # step 0: the control IS the analysis; the members' mean smooths it
            for li, lev in enumerate(LEV_WIND):
                for fi, var in enumerate(("UGRD", "VGRD", "HGT")):
                    acc["wind"][i, li, fi] = np.mean([got[m][(var, lev)] for m in use], axis=0)
            acc["wind_n"][i] = len(use)
        print(
            f"  {MODELS[model]} {date} {hh}Z +{L:03d} h: {len(ms)} members x {len(want)} fields "
            f"({time.time() - t0:.0f} s, {nbytes / 1e9:.1f} GB decoded)",
            flush=True,
        )
    for p in todo:
        if p == "winds":
            continue
        a = acc[PART_KEYS[p][-1]]
        n = int(np.isfinite(a.reshape(a.shape[0], -1)).all(axis=1).sum())
        if n < FLOOR[model]:
            raise SystemExit(f"{MODELS[model]} {date} {hh}Z {p}: only {n} complete members")
    out = {**have, **acc, **{f"done_{p}": np.array(True) for p in todo}}
    out.update(
        model=np.array(model),
        init=np.array(f"{date}{hh}"),
        lat=LAT,
        lon=LON,
        leads_daily=np.array(LEADS_DAILY),
        leads_vortex=np.array(LEADS_VORTEX),
        leads_12h=np.array(LEADS_12H),
        lev_drip=np.array(LEV_DRIP),
        lev_wave=np.array(LEV_WAVE),
        lev_wind=np.array(LEV_WIND),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".part.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(dest)
    print(
        f"  {MODELS[model]} {date} {hh}Z: {', '.join(todo)} -> {dest.name} "
        f"({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s)",
        flush=True,
    )
    return dest


def load(model: str, date: str, hh: str, parts) -> dict:
    """This cycle's reductions for the parts asked, fetching whatever is missing first."""
    with np.load(fetch(model, date, hh, parts)) as z:
        return {k: z[k] for k in z.files}


def complete_members(d: dict, key: str) -> np.ndarray:
    """Indices of the members (rows of a member-first array) with every value of `key` present."""
    a = d[key]
    return np.flatnonzero(np.isfinite(a.reshape(a.shape[0], -1)).all(axis=1))


def resolve(model: str, date: str, hh: str, last_h: int = 384, back: int = 2) -> tuple[str, str]:
    """The asked-for cycle if it has reached last_h, else the newest of the `back` previous 12-hourly cycles that has
    (strat-ifs.yml passes the IFS-ENS cycle; GEFS and GEPS are normally complete hours before it)."""
    ready = gefs_ready if model == "gefs" else geps_ready
    t = dt.datetime.strptime(date + hh, "%Y%m%d%H")
    for k in range(back + 1):
        c = t - dt.timedelta(hours=12 * k)
        d, h = c.strftime("%Y%m%d"), c.strftime("%H")
        if ready(d, h, last_h):
            if k:
                print(f"  {MODELS[model]} {date} {hh}Z not complete to {last_h} h; using {d} {h}Z", flush=True)
            return d, h
    raise SystemExit(f"no {MODELS[model]} cycle complete to {last_h} h within {12 * back} h of {date} {hh}Z")


def shown(model: str, ref: str = "origin/main") -> str:
    """The cycle (YYYYMMDDHH) that EVERY one of the model's days 1-15 products shows on `ref` (the oldest of them), or
    "" when one has not been published. Read from git, not the checkout: a queued run checks out an older commit."""

    def doc(rel: str) -> dict | None:
        r = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{ref}:{rel}"], check=False, capture_output=True, text=True
        )
        return json.loads(r.stdout) if r.returncode == 0 else None

    got = []
    for rel, key, fmt in (
        (f"assets/sst/data/{model}_drip.json", "init", "%Y-%m-%d %HZ"),
        (f"assets/sst/data/{model}_heatflux100.json", "init", "%Y-%m-%dT%H:%M:%S"),
        (f"assets/sst/anim/wave1_maps_{model}_manifest.json", "ver", "%Y%m%d%H"),
        (f"assets/sst/anim/wave2_maps_{model}_manifest.json", "ver", "%Y%m%d%H"),
        (f"assets/sst/anim/vortex_winds_{model}_manifest.json", "ver", "%Y%m%d%H"),
    ):
        d = doc(rel)
        if not d or not d.get(key):
            return ""
        got.append(dt.datetime.strptime(d[key], fmt).strftime("%Y%m%d%H"))
    return min(got)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch and reduce one cycle (the asked-for one, or the newest complete before it)")
    f.add_argument("--model", required=True, choices=tuple(MODELS))
    f.add_argument("--date", required=True)
    f.add_argument("--time", required=True)
    f.add_argument("--parts", default=",".join(PARTS))
    f.add_argument("--github-output", action="store_true", help="write the cycle used as <model>_date/_time lines")
    f.add_argument(
        "--skip-shown",
        action="store_true",
        help="fetch nothing (and write <model>_skip=true) when main already shows this cycle in every product",
    )
    a = ap.parse_args()
    parts = [p for p in a.parts.split(",") if p]
    bad = set(parts) - set(PARTS)
    if bad:
        ap.error(f"unknown part(s): {', '.join(sorted(bad))}")
    date, hh = resolve(a.model, a.date, f"{int(a.time):02d}")
    skip = a.skip_shown and shown(a.model) >= f"{date}{hh}"
    if skip:
        print(f"  {MODELS[a.model]} {date} {hh}Z is already on the page in every product; nothing to fetch", flush=True)
    else:
        fetch(a.model, date, hh, parts)
    if a.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"{a.model}_date={date}\n{a.model}_time={hh}\n{a.model}_skip={str(skip).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
