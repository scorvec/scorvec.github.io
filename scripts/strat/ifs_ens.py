#!/usr/bin/env python3
"""IFS-ENS for the stratosphere products: one shared, byte-ranged fetch per cycle, all 50 members (2026-09-27; user:
"5 ifs-ens members is not enough for the stratosphere plots" and "We also add in the ifs-ens for all strat products").

WHAT ECMWF OPEN DATA CARRIES (index probed 2026-09-27)
  enfo  the ensemble: the 50 PERTURBED members only at pressure levels - gh, t, u, v (and d, q, r, vo, w) on 1000, 925,
        850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 and 10 hPa, 0.25 deg. NO control (type=cf) at any pressure
        level, and nothing above 10 hPa. One `-enfo-ef` file per step, ~8,500 messages in arrival order (no member or
        level blocks), so any 50-member selection is a scatter of 0.3-0.7 MB ranges: run with ECMWF_RANGE_GAP_MB=0.5.
  oper  HRES: the same levels at step 0 - the 4D-Var analysis the ensemble is started from. The maps' analysis frames
        use it (hres_analysis), since the ensemble has no control there.

WHEN IT IS ON THE MIRROR (Google object times, 00/12Z cycles of 2026-09-24..27): step 0 ~07:40Z / 19:40Z, day 15
~08:50Z / 20:55Z - after strat.yml's 07:35 / 19:35 slots, which is why IFS-ENS renders in its own workflow
(strat-ifs.yml, 09:05 / 21:05) and why `resolve` waits for the day-15 index before anything is fetched.

THE SHARED FILES (the store's canonical cache, scripts/ecmwf/cache/{cycle}/ifs/), fetched once, read by every product:
  gh   11 drip levels 1000..10 hPa, daily 0-360 h    ~3.5 GB   forecast_drip; wave1_maps (500/100/10); nh_vortex (100)
  u,v  10 and 100 hPa, 12-hourly 0-360 h             ~4.0 GB   vortex_winds; nh_vortex (u); heatflux100 (v at 100)
  t    100 hPa, daily 0-360 h                        ~0.5 GB   heatflux100
The E-P flux (u, v, t at all 14 levels, ~23 GB a cycle) is streamed step by step in qbo_duct.py instead of cached.

GOOGLE MIRROR ONLY (user rule 2026-09-06): every request here names the Google mirror and nothing else; a cycle the
mirror does not have yet is skipped with a warning, never fetched from another mirror.

    python ifs_ens.py resolve --wait-min 45 --github-output   # cycle + ready flag for the workflow
    python ifs_ens.py fetch --date 20260927 --time 00          # stock the three shared files
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import rangefetch as rf                                                  # noqa: E402
import store as E                                                        # noqa: E402

SOURCES = ["google"]                         # never another mirror, not even as a fallback
MEMBERS = 50                                 # the whole perturbed ensemble; there is no control at pressure levels
STEPS = tuple(E.STEPS)                       # 0..360 h by 24
STEPS_12H = tuple(range(0, 361, 12))
LEV_GH = (1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 10)   # = geosfp_cap.LEVELS, the drip's levels
LABEL = "IFS-ENS"
CREDIT = "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25°, Google Cloud mirror"


def spec_gh() -> E.Spec:
    return E.Spec("ifs", "pf", "gh", "pl", LEV_GH, STEPS)


def spec_uv() -> E.Spec:
    return E.Spec("ifs", "pf", ("u", "v"), "pl", (10, 100), STEPS_12H)


def spec_t100() -> E.Spec:
    return E.Spec("ifs", "pf", "t", "pl", (100,), STEPS)


def _index_url(date: str, time_: str, step: int, stream: str = "enfo", kind: str = "ef") -> str:
    return f"{rf.MIRRORS['google']}/{rf.path_for(date, time_, 'ifs', step, kind, stream=stream)}.index"


def published(date: str, time_: str, step: int = 360) -> bool:
    """True when the step's .index is on the Google mirror. Day 15 is the last thing ECMWF disseminates, so it gates
    the whole cycle."""
    try:
        r = requests.head(_index_url(date, time_, step), timeout=(5, 20))
        return r.status_code == 200
    except Exception:                                                    # noqa: BLE001
        return False


def _aifs_published(date: str, time_: str) -> bool:
    try:
        r = requests.head(f"{rf.MIRRORS['google']}/{rf.path_for(date, time_, 'aifs-ens', 360, 'pf')}.index",
                          timeout=(5, 20))
        return r.status_code == 200
    except Exception:                                                    # noqa: BLE001
        return False


def expected_cycle(now: dt.datetime | None = None) -> tuple[str, str]:
    """The cycle the stratosphere page is on: the newest 00/12Z whose AIFS-ENS day 15 is on the Google mirror (IFS
    trails it by about two hours). Falls back to the newest nominal cycle at least 9 h old."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    c = now.replace(minute=0, second=0, microsecond=0, hour=12 if now.hour >= 12 else 0)
    for _ in range(4):
        if _aifs_published(f"{c:%Y%m%d}", f"{c:%H}"):
            return f"{c:%Y%m%d}", f"{c:%H}"
        c -= dt.timedelta(hours=12)
    c = now - dt.timedelta(hours=9)
    c = c.replace(minute=0, second=0, microsecond=0, hour=12 if c.hour >= 12 else 0)
    return f"{c:%Y%m%d}", f"{c:%H}"


def resolve(date: str | None, time_: str | None, wait_min: float = 0.0) -> tuple[str, str, bool]:
    """(date, hour, ready). Waits up to wait_min for the day-15 index; never looks anywhere but Google."""
    if not date or not time_:
        date, time_ = expected_cycle()
    deadline = time.monotonic() + 60.0 * wait_min
    while True:
        if published(date, time_):
            return date, time_, True
        if time.monotonic() >= deadline:
            return date, time_, False
        print(f"  IFS-ENS {date} {time_}Z day 15 not on the Google mirror yet; waiting", flush=True)
        time.sleep(60)


def _set_env():
    """Pin the store to Google for this process, whatever the caller exported."""
    os.environ["ECMWF_SOURCES"] = "google"
    os.environ["ECMWF_MULTISOURCE"] = "0"
    E.SOURCES = ["google"]


def ensure(cycle: E.Cycle, spec: E.Spec) -> Path:
    _set_env()
    return E.ensure(cycle, spec)


def fetch_all(cycle: E.Cycle) -> dict:
    """Stock the three shared files; returns {name: path} for the ones that arrived (a failure is reported, not
    raised, so one missing file does not take the others down)."""
    out = {}
    for name, spec in (("gh", spec_gh()), ("uv", spec_uv()), ("t100", spec_t100())):
        t0 = time.time()
        try:
            p = ensure(cycle, spec)
            out[name] = p
            mb = p.stat().st_size / 1e6
            print(f"  IFS-ENS {name}: {mb:,.0f} MB in {time.time() - t0:.0f} s ({mb / max(time.time() - t0, 0.1):.0f} MB/s)",
                  flush=True)
        except Exception as e:                                           # noqa: BLE001
            print(f"  IFS-ENS {name}: FAILED ({type(e).__name__}: {str(e)[:120]})", flush=True)
    return out


def hres_analysis(cycle: E.Cycle, params: tuple, levels: tuple) -> Path:
    """IFS HRES step 0 (the analysis) for params x levels, byte-ranged from the Google mirror into the cycle cache."""
    params = tuple(params)
    levels = tuple(int(x) for x in levels)
    p = E.CACHE / cycle.tag / "ifs" / f"an_{'-'.join(params)}_{'-'.join(str(x) for x in levels)}.grib2"
    if p.exists() and p.stat().st_size > 0:
        return p
    idx = rf.fetch_index(cycle.date, cycle.time, "ifs", 0, "fc", stream="oper", sources=SOURCES)
    want = rf.select(idx, param=list(params), levelist=list(levels))
    if len(want) < len(params) * len(levels):
        raise RuntimeError(f"HRES analysis: {len(want)} of {len(params) * len(levels)} messages in the index")
    blob = rf.fetch_ranges(rf.path_for(cycle.date, cycle.time, "ifs", 0, "fc", stream="oper") + ".grib2",
                           rf.coalesce(want), sources=SOURCES)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_bytes(blob)
    os.replace(tmp, p)
    return p


def open_field(path, short: str, lev: int, chunks=None):
    """Lazy DataArray of one shortName/level; member and step dimensions kept."""
    import xarray as xr
    kw = dict(engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": short, "level": int(lev)},
                                                    indexpath=""))
    if chunks is not None:
        kw["chunks"] = chunks
    ds = xr.open_dataset(path, **kw)
    return ds[short] if short in ds else ds[list(ds.data_vars)[0]]


def step_hours(da):
    import numpy as np
    if "step" not in da.dims:
        return np.array([0])
    return (da.step.values / np.timedelta64(1, "h")).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("resolve", "fetch"))
    ap.add_argument("--date"); ap.add_argument("--time")
    ap.add_argument("--wait-min", type=float, default=0.0)
    ap.add_argument("--github-output", action="store_true")
    a = ap.parse_args()
    if a.cmd == "resolve":
        d, t, ok = resolve(a.date, a.time, a.wait_min)
        print(f"cycle {d} {t}Z: IFS-ENS {'on the Google mirror' if ok else 'NOT on the Google mirror'}", flush=True)
        if a.github_output and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"date={d}\ntime={t}\nready={'true' if ok else 'false'}\n")
        return 0
    if not (a.date and a.time):
        ap.error("fetch needs --date and --time")
    if not published(a.date, a.time):
        print(f"::warning::IFS-ENS {a.date} {a.time}Z is not on the Google mirror; nothing fetched", flush=True)
        return 0
    got = fetch_all(E.Cycle(a.date, a.time))
    tot = sum(p.stat().st_size for p in got.values()) / 1e9
    print(f"  shared IFS-ENS files: {len(got)}/3, {tot:.2f} GB", flush=True)
    return 0 if len(got) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
