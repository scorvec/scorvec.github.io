#!/usr/bin/env python3
"""The GEFSv12 reforecast's 35-day runs reduced as eddy_extract.py reduces ERA5 and CMIP6, for testing the forecast
model's eddy feedback and stratosphere-to-surface coupling against ERA5 on the same days (eddy_feedback_gefs.py).
Runs in Actions (.github/workflows/eddy-feedback-gefs.yml), one winter per job.

The reforecast (NOAA, public on AWS: noaa-gefs-retrospective) runs 11 members to 35 days from 00Z every Wednesday,
2000-2019. Each member's u at 10, 100, 250 and 500 hPa, v at 250 and 500 hPa and sea-level pressure are read for the
valid days 1-34, a message at a time by byte range from each file's index, 6-hourly (00/06/12/18Z valid) and averaged
to daily means as ERA5's are; nothing raw is kept. Per init, member and day (eddy_extract's reductions):
  u (10, 100, 250, 500 hPa) and uv (250, 500 hPa) by latitude, slp by latitude, slp_p, nao

    python gefs_eddy_extract.py run --winter 2009 --out out/          # Wednesdays 1 Oct 2009 - 30 Apr 2010
    python gefs_eddy_extract.py run --inits 2010010600 --members c00,p01 --out out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eddy_extract as EX

BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
MEMBERS = ("c00", *(f"p{k:02d}" for k in range(1, 11)))
U_LEV, V_LEV = (10, 100, 250, 500), (250, 500)
DAYS = np.arange(1, 35)  # valid days after the 00Z init
STEPS = (0, 6, 12, 18)
FILES = {  # (lead range, field) -> file stem; the first range holding a lead wins
    "u": (("Days:1-10", "ugrd_pres_abv700mb"), ("Days:10-35", "ugrd_pres")),
    "v": (("Days:1-10", "vgrd_pres_abv700mb"), ("Days:10-35", "vgrd_pres")),
    "msl": (("Days:1-10", "pres_msl"), ("Days:10-35", "pres_msl")),
}
FIRST, LAST = pd.Timestamp("2000-01-01"), pd.Timestamp("2019-12-31")
SESSION = requests.Session()


def url(init: pd.Timestamp, member: str, days: str, stem: str) -> str:
    i = f"{init:%Y%m%d%H}"
    return f"{BASE}/{init:%Y}/{i}/{member}/{days}/{stem}_{i}_{member}.grib2"


def get(u: str, rng: tuple[int, int] | None = None) -> bytes:
    for attempt in range(6):
        try:
            h = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
            r = SESSION.get(u, headers=h, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException as err_:
            err = err_
            time.sleep(3 * (attempt + 1))
    raise err


def index(u: str) -> dict:
    """{(field, level hPa or 0 for sea level, lead hours): (first byte, last byte)} from the file's .idx."""
    lines = get(u + ".idx").decode().strip().splitlines()
    out = {}
    for k, line in enumerate(lines):
        p = line.split(":")
        var, lev, fc = p[3], p[4], p[5]
        m = re.match(r"(\d+) hour fcst", fc)
        lead = int(m.group(1)) if m else (0 if fc == "anl" else None)
        if lead is None:
            continue
        if var in ("UGRD", "VGRD") and lev.endswith(" mb"):
            key = (var[0].lower(), int(float(lev[:-3])), lead)
        elif var in ("PRES", "PRMSL") and lev == "mean sea level":
            key = ("msl", 0, lead)
        else:
            continue
        end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else None
        out[key] = (int(p[1]), end)
    return out


def decode(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One GRIB2 message -> (values on ascending latitudes, lat, lon)."""
    import eccodes

    h = eccodes.codes_new_from_message(blob)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        la0, la1 = (
            eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
            eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees"),
        )
        lo0, dlo = (
            eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
            eccodes.codes_get(h, "iDirectionIncrementInDegrees"),
        )
        x = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    lat, lon = np.linspace(la0, la1, nj), (lo0 + dlo * np.arange(ni)) % 360.0
    if lat[0] > lat[-1]:
        lat, x = lat[::-1], x[::-1]
    return x, lat, lon


class Grids:
    """eddy_extract's reductions, set up once per grid (days 1-10 and 10-35 can differ)."""

    def __init__(self):
        self.p = {}

    def pressure(self, lat, lon):
        key = (len(lat), len(lon))
        if key not in self.p:
            self.p[key] = EX.Pressure(lat, lon, "GEFSv12")
        return self.p[key]


def member_run(init: pd.Timestamp, member: str) -> dict:
    """One member's daily reductions for days 1-34 (NaN where a message is missing)."""
    idx = {}
    for field, files in FILES.items():
        for days, stem in files:
            u = url(init, member, days, stem)
            try:
                idx[(field, days)] = (u, index(u))
            except requests.HTTPError:
                idx[(field, days)] = (u, {})

    def locate(field, lev, lead):
        for days, _ in FILES[field]:
            u, ix = idx[(field, days)]
            if (field, lev, lead) in ix:
                return u, ix[(field, lev, lead)]
        return None

    nj = len(EX.LAT)
    out = {
        "u": np.full((len(DAYS), len(U_LEV), nj), np.nan, "float32"),
        "uv": np.full((len(DAYS), len(V_LEV), nj), np.nan, "float32"),
        "slp": np.full((len(DAYS), nj), np.nan, "float32"),
        **{k: np.full(len(DAYS), np.nan, "float32") for k in ("slp_p", "nao")},
    }
    G = Grids()
    wanted = [("u", L) for L in U_LEV] + [("v", L) for L in V_LEV] + [("msl", 0)]

    def fetch(job):
        field, lev, lead = job
        hit = locate(field, lev, lead)
        if hit is None:
            return job, None
        u, (a, b) = hit
        return job, decode(get(u, (a, b if b is not None else a + 50_000_000)))

    with ThreadPoolExecutor(16) as ex:
        for k, d in enumerate(DAYS):
            jobs = [(f, L, 24 * int(d) + s) for f, L in wanted for s in STEPS]
            got = dict(ex.map(fetch, jobs))
            if any(v is None for v in got.values()):
                continue
            # The day's mean field on its coarsest grid: day 10 straddles the 0.25 deg days 1-10 files and the
            # days 10-35 ones, and every grid starts at 90N, 0E, so a finer one is thinned to it exactly.
            nlon = min(x[2].size for x in got.values())
            lat, lon = next(x[1:] for x in got.values() if x[2].size == nlon)
            nh = lat >= EX.LAT[0] - 5.0
            mean = {}
            for f, L in wanted:
                xs = [got[(f, L, 24 * int(d) + s)] for s in STEPS]
                mean[(f, L)] = np.mean([x[0][:: x[2].size // nlon, :: x[2].size // nlon][nh] for x in xs], 0)
            lat = lat[nh]
            out["u"][k] = EX.to_lat(np.stack([mean[("u", L)] for L in U_LEV]).mean(-1), lat)
            uu, vv = (np.stack([mean[(f, L)] for L in V_LEV]) for f in ("u", "v"))
            out["uv"][k] = EX.zonal(uu, vv, lat)[1]
            ps, pp, nao = G.pressure(lat, lon)(mean[("msl", 0)][None] / 100.0)
            out["slp"][k], out["slp_p"][k], out["nao"][k] = ps[0], pp[0], nao[0]
    return out


def unit(args):
    init, member = args
    t0 = time.time()
    try:
        return init, member, member_run(pd.Timestamp(init), member), round(time.time() - t0)
    except Exception as err:  # noqa: BLE001  (one member's failure is logged, not fatal)
        return init, member, f"{type(err).__name__}: {str(err)[:200]}", round(time.time() - t0)


def wednesdays(winter: int) -> list[pd.Timestamp]:
    d = pd.date_range(f"{winter}-10-01", f"{winter + 1}-04-30", freq="W-WED")
    return [t for t in d if FIRST <= t <= LAST]


def run(inits: list[pd.Timestamp], members, out: Path, name: str, workers: int) -> dict:
    t0 = time.time()
    dest = out / f"GEFSv12_{name}.npz"
    res = {
        "u": np.full((len(inits), len(members), len(DAYS), len(U_LEV), len(EX.LAT)), np.nan, "float32"),
        "uv": np.full((len(inits), len(members), len(DAYS), len(V_LEV), len(EX.LAT)), np.nan, "float32"),
        "slp": np.full((len(inits), len(members), len(DAYS), len(EX.LAT)), np.nan, "float32"),
        **{k: np.full((len(inits), len(members), len(DAYS)), np.nan, "float32") for k in ("slp_p", "nao")},
    }
    pos = {(str(i), m): (a, b) for a, i in enumerate(inits) for b, m in enumerate(members)}
    failed, done = [], 0
    with ProcessPoolExecutor(workers) as ex:
        for init, member, r, sec in ex.map(unit, [(str(i), m) for i in inits for m in members]):
            done += 1
            if isinstance(r, str):
                failed.append(f"{init} {member}: {r}")
                print(f"    {init} {member}: FAILED {r}", flush=True)
                continue
            a, b = pos[(init, member)]
            for k, x in r.items():
                res[k][a, b] = x
            missing = int(np.isnan(r["slp_p"]).sum())
            jet, slp = np.nanmax(np.nanmean(r["u"][:, U_LEV.index(250)], 0)), np.nanmean(r["slp"])
            print(f"    {init[:10]} {member}: {sec} s, 250 hPa jet {jet:.0f} m/s, mean slp {slp:.0f} hPa"
                  f"{f', {missing} days missing' if missing else ''} ({done}/{len(pos)}, {time.time() - t0:.0f} s)",
                  flush=True)  # fmt: skip
    # physical range: a wrong level, unit or grid in the decode would fail it
    jet = np.nanmax(np.nanmean(res["u"][..., U_LEV.index(250), :], (0, 1, 2)))
    if not (15 < jet < 70 and 990 < np.nanmean(res["slp"]) < 1035 and np.nanmax(np.abs(res["uv"])) < 600):
        raise ValueError(f"values out of range: 250 hPa jet {jet:.1f} m/s, slp {np.nanmean(res['slp']):.1f} hPa")
    part = dest.with_suffix(".part.npz")
    np.savez_compressed(
        part, init=np.array([f"{i:%Y-%m-%d}" for i in inits]), members=np.array(members), days=DAYS,
        u_levels=np.array(U_LEV), uv_levels=np.array(V_LEV), lat=EX.LAT.astype("float32"), **res,
    )  # fmt: skip
    part.replace(dest)
    ok = float(np.isfinite(res["slp_p"]).mean())
    return {"run": dest.stem, "status": "ok" if ok > 0.9 else f"only {ok:.0%} of member-days", "inits": len(inits),
            "members": len(members), "complete": round(ok, 4), "failed": failed[:50], "seconds": round(time.time() - t0)}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the winters, as a JSON list (for the workflow's matrix)")
    p.add_argument("--winters", default="1999-2019")
    r = sub.add_parser("run")
    r.add_argument("--winter", type=int, help="the Wednesdays of 1 October of this year to 30 April")
    r.add_argument("--inits", default="", help="comma-separated YYYYMMDDHH (instead of --winter)")
    r.add_argument("--members", default=",".join(MEMBERS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        w0, w1 = (int(x) for x in a.winters.split("-"))
        print(json.dumps([{"winter": w} for w in range(w0, w1 + 1) if wednesdays(w)]))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.inits:
        inits, name = [pd.Timestamp(f"{x[:8]}T{x[8:10]}") for x in a.inits.split(",")], "test"
    else:
        inits, name = wednesdays(a.winter), str(a.winter)
    try:
        res = run(inits, a.members.split(","), out, name, a.workers)
    except Exception as err:  # noqa: BLE001
        res = {"run": f"GEFSv12_{name}", "status": f"failed: {type(err).__name__}: {str(err)[:300]}"}
    print({k: v for k, v in res.items() if k != "failed"}, flush=True)
    (out / f"{res['run']}_log.json").write_text(json.dumps(res))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
