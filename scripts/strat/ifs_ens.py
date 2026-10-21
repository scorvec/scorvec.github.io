#!/usr/bin/env python3
"""IFS-ENS and the IFS control for the stratosphere products: one shared, byte-ranged fetch per cycle, all 50 members
and the control, read by every stratosphere product.

WHAT ECMWF OPEN DATA CARRIES (index probed 2026-09-27)
  enfo  the ensemble: the 50 PERTURBED members only at pressure levels - gh, t, u, v (and d, q, r, vo, w) on 1000, 925,
        850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 and 10 hPa, 0.25 deg. NO control (type=cf) at any pressure
        level, and nothing above 10 hPa. One `-enfo-ef` file per step, ~8,500 messages in arrival order (no member or
        level blocks), so any 50-member selection is a scatter of 0.3-0.7 MB ranges: run with ECMWF_RANGE_GAP_MB=0.5.
  oper  the IFS control (ECMWF's unperturbed 9 km run, called HRES until May 2026): the same levels at every step to
        360 h (0..144 by 3, then by 6; probed 2026-10-21), one `-oper-fc` file per step. Its step 0 is the 4D-Var
        analysis the ensemble starts from, so the ensemble maps' analysis frames are the control's.

WHEN IT IS ON THE MIRROR (Google object times, 00/12Z cycles of 2026-09-24..27): step 0 ~07:40Z / 19:40Z, day 15
~08:50Z / 20:55Z - after strat.yml's 07:35 / 19:35 slots, which is why IFS-ENS renders in its own workflow
(strat-ifs.yml, 09:05 / 21:05) and why `resolve` waits for the day-15 index before anything is fetched. The control's
whole run lands within a few minutes, ~07:36Z / 19:36Z (2026-10-21), over an hour before the ensemble's day 15.

THE SHARED FILES (the store's canonical cache, scripts/ecmwf/cache/{cycle}/ifs/), fetched once, read by every product:
  gh   11 drip levels 1000..10 hPa, daily 0-360 h    ~3.5 GB   forecast_drip; wave_maps (500/100/10); nh_vortex (100)
  u,v  10 and 100 hPa, 12-hourly 0-360 h             ~4.0 GB   vortex_winds; nh_vortex (u); heatflux100 (v at 100)
  t    100 hPa, daily 0-360 h                        ~0.5 GB   heatflux100
and the control's (type fc), ~0.3 GB together:
  ctrl_gh    gh at the 11 drip levels, daily 0-360 h              forecast_drip; wave_maps
  ctrl_wind  u, v and gh at 10 and 100 hPa, 12-hourly 0-360 h     vortex_winds; nh_vortex; heatflux100 (v at 100)
  ctrl_t100  t at 100 hPa, daily 0-360 h                          heatflux100
The E-P flux (u, v, t at all 14 levels, ~23 GB a cycle) is streamed step by step in qbo_duct.py instead of cached.

GOOGLE MIRROR ONLY: every request here names the Google mirror and nothing else; a cycle the
mirror does not have yet is skipped with a warning, never fetched from another mirror.

    python ifs_ens.py resolve --wait-min 45 --github-output   # cycle + ready flag for the workflow
    python ifs_ens.py fetch --date 20260927 --time 00          # stock the six shared files
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import cycles as C
import store as E

SOURCES = ["google"]  # never another mirror, not even as a fallback
MEMBERS = 50  # the whole perturbed ensemble; there is no control at pressure levels
STEPS = tuple(E.STEPS)  # 0..360 h by 24
STEPS_12H = tuple(range(0, 361, 12))
LEV_GH = (1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 10)  # = geosfp_cap.LEVELS, the drip's levels
LABEL = "IFS-ENS"
CREDIT = "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25°, Google Cloud mirror"
CTRL_LABEL = "IFS control"
CTRL_CREDIT = "ECMWF open data (CC BY 4.0), IFS control run, 0.25°, Google Cloud mirror"
CONTROL = (C.Need("ifs", "fc", "oper", every=C.IFS_STEPS),)  # the control's -oper-fc files, 0..360 h


def spec_gh() -> E.Spec:
    return E.Spec("ifs", "pf", "gh", "pl", LEV_GH, STEPS)


def spec_uv() -> E.Spec:
    return E.Spec("ifs", "pf", ("u", "v"), "pl", (10, 100), STEPS_12H)


def spec_t100() -> E.Spec:
    return E.Spec("ifs", "pf", "t", "pl", (100,), STEPS)


def spec_ctrl_gh() -> E.Spec:
    return E.Spec("ifs", "fc", "gh", "pl", LEV_GH, STEPS)


def spec_ctrl_wind() -> E.Spec:
    return E.Spec("ifs", "fc", ("u", "v", "gh"), "pl", (10, 100), STEPS_12H)


def spec_ctrl_t100() -> E.Spec:
    return E.Spec("ifs", "fc", "t", "pl", (100,), STEPS)


def published(date: str, time_: str, step: int = 360) -> bool:
    """True when IFS-ENS is COMPLETELY on the Google mirror to `step`: that step's index and GRIB, and every step up to
    it uploaded (the shared rule, scripts/ecmwf/cycles.py). Day 15 is the last thing ECMWF disseminates."""
    return C.is_ready(date, time_, (C.Need("ifs", "ef", step=step, every=C.IFS_STEPS),))[0]


def ctrl_published(date: str, time_: str) -> bool:
    """True when the IFS control is completely on the Google mirror to 360 h (it lands before the ensemble)."""
    return C.is_ready(date, time_, CONTROL)[0]


def expected_cycle(now: dt.datetime | None = None) -> tuple[str, str]:
    """The cycle the stratosphere page is on: the newest 00/12Z whose AIFS-ENS is completely on the Google mirror -
    the SAME rule, through the same helper, that strat.yml resolves its cycle with, so both halves of the page share
    one initialisation (IFS trails AIFS by about two hours). There is no clock fallback: when no cycle within
    two of the newest is ready this raises cycles.CycleUnavailable and the cycle job fails visibly."""
    return C.resolve(C.AIFS_ENS, now=now, what="the stratosphere page (AIFS-ENS)")


def resolve(date: str | None, time_: str | None, wait_min: float = 0.0) -> tuple[str, str, bool]:
    """(date, hour, ready). Waits up to wait_min for IFS-ENS day 15 to be complete; never looks anywhere but Google."""
    if not date or not time_:
        date, time_ = expected_cycle()
    return date, time_, C.wait_ready(date, time_, C.IFS_ENS, wait_min)


def _set_env():
    """Pin the store to Google for this process, whatever the caller exported."""
    os.environ["ECMWF_SOURCES"] = "google"
    os.environ["ECMWF_MULTISOURCE"] = "0"
    E.SOURCES = ["google"]


def ensure(cycle: E.Cycle, spec: E.Spec) -> Path:
    _set_env()
    return E.ensure(cycle, spec)


SHARED = {
    "gh": spec_gh,
    "uv": spec_uv,
    "t100": spec_t100,
    "ctrl_gh": spec_ctrl_gh,
    "ctrl_wind": spec_ctrl_wind,
    "ctrl_t100": spec_ctrl_t100,
}


def fetch_all(cycle: E.Cycle) -> dict:
    """Stock the shared files; returns {name: path} for the ones that arrived (a failure is reported, not raised, so
    one missing file does not take the others down)."""
    out = {}
    for name, spec_of in SHARED.items():
        spec = spec_of()
        t0 = time.time()
        try:
            p = ensure(cycle, spec)
            out[name] = p
            mb = p.stat().st_size / 1e6
            print(
                f"  IFS {name}: {mb:,.0f} MB in {time.time() - t0:.0f} s ({mb / max(time.time() - t0, 0.1):.0f} MB/s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  IFS {name}: FAILED ({type(e).__name__}: {str(e)[:120]})", flush=True)
    return out


def open_field(path, short: str, lev: int, chunks=None):
    """Lazy DataArray of one shortName/level; member and step dimensions kept."""
    import xarray as xr

    kw = dict(
        engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": short, "level": int(lev)}, indexpath="")
    )
    if chunks is not None:
        kw["chunks"] = chunks
    ds = xr.open_dataset(path, **kw)
    return ds[short] if short in ds else ds[next(iter(ds.data_vars))]


def step_hours(da):
    import numpy as np

    if "step" not in da.dims:
        return np.array([0])
    return (da.step.values / np.timedelta64(1, "h")).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("resolve", "fetch"))
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument("--wait-min", type=float, default=0.0)
    ap.add_argument("--github-output", action="store_true")
    a = ap.parse_args()
    if a.cmd == "resolve":
        try:
            d, t, ok = resolve(a.date, a.time, a.wait_min)
        except C.CycleUnavailable as e:
            print(f"::error::{e}", flush=True)
            return 1
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
    print(f"  shared IFS files: {len(got)}/{len(SHARED)}, {tot:.2f} GB", flush=True)
    return 0 if len(got) == len(SHARED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
