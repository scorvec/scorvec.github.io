#!/usr/bin/env python3
"""The GEFSv12 reforecast's 100 hPa eddy heat flux, for scoring the weeks 3-5 heat-flux counts of ssw.html
(ssw35_skill.py). Runs in Actions (.github/workflows/ssw35-skill.yml), one winter per job.

The same 35-day runs as gefs_eddy_extract.py (11 members, 00Z Wednesdays, 2000-2019, NOAA's public AWS bucket, read
a message at a time by byte range): v and T at 100 hPa at 00 and 12 UTC of valid days 1-34, the single fields the ERA5
series (reference/vt100_era5.nc) is built from. Each field's zonal-mean [v'T'] by latitude over zonal wavenumbers
1-72, cos(lat)-weighted over 45-75N, as heatflux100.fluxes and band_mean compute it (K_MAX, BAND); the day's value is
the mean of its two. Per init, member and day: tot, k1, k2 (K m/s); nothing raw is kept.

    python gefs_vt100_extract.py run --winter 2009 --out out/          # Wednesdays 1 Oct 2009 - 30 Apr 2010
    python gefs_vt100_extract.py run --inits 2010010600 --members c00 --out out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gefs_eddy_extract as GX

FILES = {
    "v": (("Days:1-10", "vgrd_pres_abv700mb"), ("Days:10-35", "vgrd_pres")),
    "t": (("Days:1-10", "tmp_pres_abv700mb"), ("Days:10-35", "tmp_pres")),
}
LEVEL = 100
STEPS = (0, 12)  # valid hours: ERA5's single fields
K_MAX, BAND = 72, (45.0, 75.0)  # heatflux100's wavenumbers and band
KEYS = ("tot", "k1", "k2")


def flux(v: np.ndarray, t: np.ndarray, lat: np.ndarray) -> dict[str, float]:
    """The band-mean [v'T'] of one pair of fields (lat ascending, lon), heatflux100.fluxes + band_mean."""
    m = (lat >= BAND[0]) & (lat <= BAND[1])
    n = v.shape[-1]
    per_k = 2.0 * np.real(
        np.fft.rfft(v[m], axis=-1)[:, : K_MAX + 1] * np.conj(np.fft.rfft(t[m], axis=-1)[:, : K_MAX + 1])
    )
    per_k /= n**2
    w = np.cos(np.deg2rad(lat[m]))
    w /= w.sum()
    return {"tot": float(w @ per_k[:, 1:].sum(-1)), "k1": float(w @ per_k[:, 1]), "k2": float(w @ per_k[:, 2])}


def member_run(init: pd.Timestamp, member: str) -> np.ndarray:
    """(day, key) band-mean fluxes for days 1-34 (NaN where a message is missing)."""
    idx = {}
    for field, files in FILES.items():
        for days, stem in files:
            u = GX.url(init, member, days, stem)
            try:
                idx[(field, days)] = (u, GX.index(u))
            except requests.HTTPError:
                idx[(field, days)] = (u, {})

    def fetch(job):
        field, lead = job
        for days, _ in FILES[field]:
            u, ix = idx[(field, days)]
            if (field, LEVEL, lead) in ix:
                a, b = ix[(field, LEVEL, lead)]
                return job, GX.decode(GX.get(u, (a, b if b is not None else a + 50_000_000)))
        return job, None

    out = np.full((len(GX.DAYS), len(KEYS)), np.nan, "float32")
    with ThreadPoolExecutor(8) as ex:
        for k, d in enumerate(GX.DAYS):
            got = dict(ex.map(fetch, [(f, 24 * int(d) + s) for f in FILES for s in STEPS]))
            if any(x is None for x in got.values()):
                continue
            per = []
            for s in STEPS:
                (v, lat, _), (t, lat_t, _) = got[("v", 24 * int(d) + s)], got[("t", 24 * int(d) + s)]
                if v.shape != t.shape or not np.allclose(lat, lat_t):
                    raise ValueError(f"v and T grids differ at lead {24 * int(d) + s} h")
                per.append([flux(v, t, lat)[key] for key in KEYS])
            out[k] = np.mean(per, 0)
    return out


def unit(args):
    init, member = args
    t0 = time.time()
    try:
        return init, member, member_run(pd.Timestamp(init), member), round(time.time() - t0)
    except Exception as err:  # noqa: BLE001  (one member's failure is logged, not fatal)
        return init, member, f"{type(err).__name__}: {str(err)[:200]}", round(time.time() - t0)


def run(inits: list[pd.Timestamp], members: list[str], out: Path, name: str, workers: int) -> dict:
    t0 = time.time()
    res = np.full((len(inits), len(members), len(GX.DAYS), len(KEYS)), np.nan, "float32")
    pos = {(str(i), m): (a, b) for a, i in enumerate(inits) for b, m in enumerate(members)}
    failed, done = [], 0
    with ProcessPoolExecutor(workers) as ex:
        for init, member, r, sec in ex.map(unit, list(pos)):
            done += 1
            if isinstance(r, str):
                failed.append(f"{init} {member}: {r}")
                print(f"    {init} {member}: FAILED {r}", flush=True)
                continue
            res[pos[(init, member)]] = r
            print(f"    {init[:10]} {member}: {sec} s, mean flux {np.nanmean(r[:, 0]):.1f} K m/s"
                  f" ({done}/{len(pos)}, {time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    # physical range: a wrong level, field or sign would fail it (winter 100 hPa [v'T'] 45-75N runs about 0-60 K m/s)
    m = float(np.nanmean(res[..., 0]))
    if not 2 < m < 40:
        raise ValueError(f"mean 100 hPa heat flux {m:.1f} K m/s is out of range")
    dest = out / f"GEFSv12vt_{name}.npz"
    part = dest.with_suffix(".part.npz")
    np.savez_compressed(part, init=np.array([f"{i:%Y-%m-%d}" for i in inits]), members=np.array(members),
                        days=GX.DAYS, keys=np.array(KEYS), vt=res)  # fmt: skip
    part.replace(dest)
    ok = float(np.isfinite(res[..., 0]).mean())
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
    r.add_argument("--members", default=",".join(GX.MEMBERS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        w0, w1 = (int(x) for x in a.winters.split("-"))
        print(json.dumps([{"winter": w} for w in range(w0, w1 + 1) if GX.wednesdays(w)]))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.inits:
        inits, name = [pd.Timestamp(f"{x[:8]}T{x[8:10]}") for x in a.inits.split(",")], "test"
    else:
        inits, name = GX.wednesdays(a.winter), str(a.winter)
    try:
        res = run(inits, a.members.split(","), out, name, a.workers)
    except Exception as err:  # noqa: BLE001
        res = {"run": f"GEFSv12vt_{name}", "status": f"failed: {type(err).__name__}: {str(err)[:300]}"}
    print({k: v for k, v in res.items() if k != "failed"}, flush=True)
    (out / f"{res['run']}_log.json").write_text(json.dumps(res))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
