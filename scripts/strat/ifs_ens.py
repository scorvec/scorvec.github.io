#!/usr/bin/env python3
"""IFS-ENS and the IFS control for the stratosphere products: one shared, byte-ranged fetch per cycle, all 50 members
and the control, read by every stratosphere product.

WHAT ECMWF OPEN DATA CARRIES (index probed 2026-09-27)
  enfo  the ensemble: the 50 PERTURBED members only at pressure levels - gh, t, u, v (and d, q, r, vo, w) on 1000, 925,
        850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 and 10 hPa, 0.25 deg. NO control (type=cf) at any pressure
        level, and nothing above 10 hPa. One `-enfo-ef` file per step, ~8,500 messages in arrival order (no member or
        level blocks), so any 50-member selection is a scatter of 0.3-0.7 MB ranges (rangefetch's 0.5 MB gap).
  oper  the IFS control (ECMWF's unperturbed 9 km run, called HRES until May 2026): the same levels at every step to
        360 h (0..144 by 3, then by 6; probed 2026-10-21), one `-oper-fc` file per step. Its step 0 is the 4D-Var
        analysis the ensemble starts from, so the ensemble maps' analysis frames are the control's.

WHEN IT IS ON THE MIRROR (Google object times): the control's whole run within a few minutes, ~07:36Z / 19:36Z; the
ensemble over ~50 min after it, step 0 ~07:40Z / 19:40Z to day 15 ~08:31Z / 20:31Z (2026-10-25), a 12-hourly step every
~2 min. `resolve` takes the cycle the stratosphere page is on (AIFS-ENS complete, as strat.yml) and waits only for the
ensemble to BEGIN landing; `land` then follows it, taking each step the moment its index and GRIB are both on the mirror
(cycles.complete_steps), so when day 15 lands only the last steps are left to fetch.

THE SHARED FILES (the store's canonical cache, scripts/ecmwf/cache/{cycle}/ifs/), fetched once, read by every product:
  gh       9 drip levels 1000..50 hPa (not 10, 100), daily 0-360 h   ~3.1 GB   forecast_drip; wave_maps (500);
                                                                               polarcap (500); regimes (500)
  gh_wind  gh at 10 and 100 hPa, daily 0-360 h                      ~0.65 GB  forecast_drip; wave_maps; polarcap;
                                                                               nh_vortex (100); vortex_winds
  uv       u, v at 10 and 100 hPa, 12-hourly 0-360 h                ~3.4 GB   vortex_winds; nh_vortex (u); heatflux100
                                                                               (v at 100); polar_descent (edge v)
  t100     t at 100 hPa, daily 0-360 h                              ~0.25 GB  heatflux100; polar_descent (edge t)
  t10      t at 10 hPa, daily 0-360 h                               ~0.2 GB   polarcap (the 10 hPa maps); polar_descent
  msl      sea-level pressure, daily 0-360 h                        ~0.4 GB   polarcap
  (gh_mid  gh at 10 and 100 hPa at the 12-hourly steps between those, ~0.6 GB: the vortex-wind loop's own frames)
and the control's (type fc), ~0.2 GB together:
  ctrl_gh    gh at the 11 drip levels, daily 0-360 h              forecast_drip; wave_maps; polarcap (500/100/10)
  ctrl_wind  u, v and gh at 10 and 100 hPa, 12-hourly 0-360 h     vortex_winds; nh_vortex; heatflux100; polar_descent
  ctrl_t100  t at 100 hPa, daily 0-360 h                          heatflux100; polar_descent
  ctrl_msl   sea-level pressure, daily 0-360 h                    polarcap
and, of the members and the control, only the leading rows of each field (ROWS: each message's first bytes, the rows
decoded from them, store.ensure_rows / ccsds.py), daily 0-360 h:
  edge       v and t at 10-400 hPa (8 levels), rows 65.5-65N         polar_descent (the residual descent: the eddy heat
  ctrl_edge  the same for the control                ~0.6 GB        flux and the stability on the cap's 65.25N edge);
             v at 10/100 and t at 100 (and 10) hPa are decoded from the whole files above, which hold those messages
  w          vertical velocity at 9 levels 850..10 hPa,            polar_descent (the column, sectors and maps),
  ctrl_w     90N-48.25N, kept as its 1.5-deg cells   ~1.0 GB        instead of ~3.8 GB for the whole fields
The E-P flux (u, v, t at all 14 levels, ~23 GB a cycle) is streamed step by step in qbo_duct.py instead of cached.

GOOGLE MIRROR ONLY: every request here names the Google mirror and nothing else; a cycle the
mirror does not have yet is skipped with a warning, never fetched from another mirror.

    python ifs_ens.py resolve --wait-min 15 --github-output             # cycle + whether IFS-ENS is landing
    python ifs_ens.py land --date 20261025 --time 12 --names uv,t100 --wait-min 100   # follow the run as it lands
    python ifs_ens.py fetch --date 20260927 --time 00                    # a complete cycle: every shared file and row
    python ifs_ens.py wait --date 20261025 --time 12 --wait-min 100     # until IFS-ENS is complete (exit 1 if not)
    python ifs_ens.py paths --date 20261025 --time 12 --names gh_wind,t10   # their cache files (a job hands them on)
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
STEPS_MID = tuple(s for s in STEPS_12H if s not in STEPS)  # 12, 36, ..., 348 h
LEV_GH = (1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 10)  # the drip's levels to 10 hPa (forecast_drip.LEVELS_10)
LEV_WIND = (10, 100)  # the vortex-wind levels: u, v and gh at both every 12 h (the other products take the daily steps)
LABEL = "IFS-ENS"
CREDIT = "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25°, Google Cloud mirror"
CTRL_LABEL = "IFS control"
CTRL_CREDIT = "ECMWF open data (CC BY 4.0), IFS control run, 0.25°, Google Cloud mirror"
CONTROL = (C.Need("ifs", "fc", "oper", every=C.IFS_STEPS),)  # the control's -oper-fc files, 0..360 h


def spec_gh() -> E.Spec:
    """gh at the drip's levels other than 10 and 100 hPa (spec_gh_wind holds those), daily."""
    return E.Spec("ifs", "pf", "gh", "pl", tuple(L for L in LEV_GH if L not in LEV_WIND), STEPS)


def spec_gh_wind() -> E.Spec:
    return E.Spec("ifs", "pf", "gh", "pl", LEV_WIND, STEPS)


def spec_gh_mid() -> E.Spec:
    """gh at 10 and 100 hPa at the 12-hourly steps between the daily ones: the vortex-wind loop's alone."""
    return E.Spec("ifs", "pf", "gh", "pl", LEV_WIND, STEPS_MID)


def gh_spec(lev: int) -> E.Spec:
    """The shared daily gh file that holds level lev."""
    return spec_gh_wind() if int(lev) in LEV_WIND else spec_gh()


def spec_uv() -> E.Spec:
    return E.Spec("ifs", "pf", ("u", "v"), "pl", LEV_WIND, STEPS_12H)


def spec_t100() -> E.Spec:
    return E.Spec("ifs", "pf", "t", "pl", (100,), STEPS)


def spec_t10() -> E.Spec:
    return E.Spec("ifs", "pf", "t", "pl", (10,), STEPS)


def spec_w() -> E.Spec:
    import polar_descent as PD

    return E.Spec("ifs", "pf", "w", "pl", PD.LEVELS_EC, STEPS)


def spec_msl() -> E.Spec:
    return E.Spec("ifs", "pf", "msl", "sfc", (), STEPS)


def spec_ctrl_gh() -> E.Spec:
    return E.Spec("ifs", "fc", "gh", "pl", LEV_GH, STEPS)


def spec_ctrl_wind() -> E.Spec:
    return E.Spec("ifs", "fc", ("u", "v", "gh"), "pl", LEV_WIND, STEPS_12H)


def spec_ctrl_t100() -> E.Spec:
    return E.Spec("ifs", "fc", "t", "pl", (100,), STEPS)


def spec_ctrl_w() -> E.Spec:
    import polar_descent as PD

    return E.Spec("ifs", "fc", "w", "pl", PD.LEVELS_EC, STEPS)


def spec_ctrl_msl() -> E.Spec:
    return E.Spec("ifs", "fc", "msl", "sfc", (), STEPS)


def spec_ctrl_t10() -> E.Spec:
    return E.Spec("ifs", "fc", "t", "pl", (10,), STEPS_12H)  # t10_maps.py fetches it itself (t10.yml), not shared


EDGE_LEVELS = (10, 50, 100, 150, 200, 250, 300, 400)  # the residual's levels, and 400 for the stability at 300
EDGE_ROWS = (98, 99, 100)  # 65.5, 65.25 and 65N on the 0.25-deg grid: the cap's edge 65.25N and the rows either side
W_ROWS = tuple(range(168))  # 90N to 48.25N: polar_descent's cells from 49.5N (southern edge 48.75N), a row to spare


def _w_cells(vals, lat, lon):
    import polar_descent as PD

    return PD.w_cells(vals, lat, lon)


def spec_edge() -> E.Spec:
    return E.Spec("ifs", "pf", ("v", "t"), "pl", EDGE_LEVELS, STEPS)


def spec_ctrl_edge() -> E.Spec:
    return E.Spec("ifs", "fc", ("v", "t"), "pl", EDGE_LEVELS, STEPS)


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
    """(date, hour, landing): the cycle (expected_cycle unless given), and whether IFS-ENS has begun to land on it -
    its first step complete on the Google mirror - waiting up to wait_min for that. The run is followed from there
    (land). Never looks anywhere but Google."""
    if not date or not time_:
        date, time_ = expected_cycle()
    return date, time_, C.wait_ready(date, time_, C.IFS_ENS_LANDING, wait_min)


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
    "gh_wind": spec_gh_wind,
    "uv": spec_uv,
    "t100": spec_t100,
    "t10": spec_t10,
    "msl": spec_msl,
    "ctrl_gh": spec_ctrl_gh,
    "ctrl_wind": spec_ctrl_wind,
    "ctrl_t100": spec_ctrl_t100,
    "ctrl_msl": spec_ctrl_msl,
}
FILES = {**SHARED, "gh_mid": spec_gh_mid}  # every whole-field file `land` can stock (gh_mid: vortex_winds.py's own)
# name -> (spec, rows, per-message reduction, its tag in the cache name, the whole files of FILES whose messages it
# decodes instead of fetching their leading bytes again: v at 10 and 100 hPa is in uv, t at 100 and 10 hPa in t100/t10)
ROWS = {
    "edge": (spec_edge, EDGE_ROWS, None, "", ("uv", "t100", "t10")),
    "ctrl_edge": (spec_ctrl_edge, EDGE_ROWS, None, "", ("ctrl_wind", "ctrl_t100")),
    "w": (spec_w, W_ROWS, _w_cells, "cells", ()),
    "ctrl_w": (spec_ctrl_w, W_ROWS, _w_cells, "cells", ()),
}


def ensure_rows(cycle: E.Cycle, name: str) -> Path:
    """The ROWS entry `name` of a cycle (store.ensure_rows)."""
    _set_env()
    spec_of, rows, reduce, tag, local = ROWS[name]
    return E.ensure_rows(cycle, spec_of(), rows, reduce, tag, local=tuple(FILES[n]() for n in local))


# The file stream each spec type is in, and what "the whole run is on the mirror" means for it.
STREAM = {"pf": C.IFS_ENS[0], "fc": CONTROL[0]}


def land(cycle: E.Cycle, names, wait_min: float = 0.0, poll_s: float = 30.0, workers: int = 4) -> dict:
    """Stock the files `names` (of FILES and ROWS) of a cycle step by step as the run lands on the Google mirror: each
    step as soon as both its index and its GRIB are there (cycles.complete_steps), `workers` steps at a time, one index
    fetch per step, its whole-field files first and then its row files (which decode what those hold). Each file is
    joined once all its steps are in, so when the last step lands only it is left to fetch. Ends when the whole run is
    on the mirror too (the products check that before they read). Returns {name: path}; TimeoutError naming what is
    missing when that is not so wait_min after the start (0: take what the mirror has, no waiting)."""
    from concurrent.futures import ThreadPoolExecutor

    _set_env()
    bad = [n for n in names if n not in FILES and n not in ROWS]
    if bad:
        raise ValueError(f"no such IFS file: {', '.join(bad)}")
    spec = {n: FILES[n]() if n in FILES else ROWS[n][0]() for n in names}
    final = {
        n: E.path(cycle, s) if n in FILES else E.rows_path(cycle, s, ROWS[n][1], ROWS[n][3]) for n, s in spec.items()
    }
    left = {n: set() if final[n].exists() else set(s.steps) for n, s in spec.items()}
    deadline = time.monotonic() + 60.0 * wait_min
    failures: dict = {}
    t0 = time.time()

    def one(key):
        stream_type, step = key
        todo = sorted((n for n in names if step in left[n] and spec[n].type == stream_type), key=lambda n: n in ROWS)
        stats, t1 = {}, time.time()
        idx = E.step_index(cycle, spec[todo[0]], step)
        for n in todo:
            if n in FILES:
                E.fetch_step(cycle, spec[n], step, idx, stats)
            else:
                _, rows, reduce, tag, local = ROWS[n]
                E.rows_step(cycle, spec[n], rows, step, reduce, tag, idx, tuple(FILES[m]() for m in local), stats)
        mb = sum(stats.values()) / 1e6
        print(f"  {CTRL_LABEL if stream_type == 'fc' else LABEL} +{step}h: {', '.join(todo)} - {mb:,.0f} MB in "
              f"{time.time() - t1:.0f} s", flush=True)  # fmt: skip
        return todo

    while True:
        pending = {(spec[n].type, s) for n in names for s in left[n]}
        if not pending:
            break
        ready = {t: C.complete_steps(cycle.date, cycle.time, STREAM[t]) for t in {t for t, _ in pending}}
        doable = sorted((k for k in pending if ready[k[0]] is None or k[1] in ready[k[0]]), key=lambda k: k[1])
        if doable:
            with ThreadPoolExecutor(workers) as ex:
                futs = {k: ex.submit(one, k) for k in doable}
            failed = False
            for k, f in futs.items():
                try:
                    for n in f.result():
                        left[n].discard(k[1])
                except Exception as e:  # noqa: BLE001
                    if isinstance(e, E.NotPublished) and ready[k[0]] is None:
                        continue  # the listing failed and the step is not there yet: the next pass asks again
                    failed, failures[k] = True, failures.get(k, 0) + 1
                    print(f"  {CTRL_LABEL if k[0] == 'fc' else LABEL} +{k[1]}h: {type(e).__name__}: {str(e)[:160]}",
                          flush=True)  # fmt: skip
                    if failures[k] >= 3:
                        raise
            if any(ready[t] is not None for t, _ in doable):
                time.sleep(10 if failed else 0)
                continue  # more may have landed meanwhile
        if time.monotonic() >= deadline:
            miss = {n: sorted(left[n]) for n in names if left[n]}
            raise TimeoutError(
                f"{cycle.tag} not complete on the Google mirror after {wait_min:.0f} min: "
                + "; ".join(f"{n} {len(v)} steps ({v[0]}h..{v[-1]}h)" for n, v in miss.items())
            )
        time.sleep(poll_s)
    with ThreadPoolExecutor(workers) as ex:  # each file joined from its parts (nothing left to fetch)
        got = dict(
            zip(names, ex.map(lambda n: E.ensure(cycle, spec[n]) if n in FILES else ensure_rows(cycle, n), names))
        )
    for t in sorted({s.type for s in spec.values()}):
        if not C.wait_ready(cycle.date, cycle.time, (STREAM[t],), max(0.0, (deadline - time.monotonic()) / 60.0)):
            raise TimeoutError(f"{cycle.tag}: the {'control' if t == 'fc' else 'IFS-ENS'} run is not complete on the "
                               "Google mirror")  # fmt: skip
    tot = sum(p.stat().st_size for p in got.values()) / 1e9
    print(
        f"  IFS files {', '.join(names)}: {tot:.2f} GB on disk, done {time.time() - t0:.0f} s after the start",
        flush=True,
    )
    return got


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
    ap.add_argument("cmd", choices=("resolve", "land", "fetch", "wait", "paths"))
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument(
        "--names", help="land / fetch / paths: the files, comma-separated (fetch: every shared file and row file)"
    )
    ap.add_argument("--wait-min", type=float, default=0.0)
    ap.add_argument("--github-output", action="store_true")
    a = ap.parse_args()
    if a.cmd == "resolve":
        try:
            d, t, ok = resolve(a.date, a.time, a.wait_min)
        except C.CycleUnavailable as e:
            print(f"::error::{e}", flush=True)
            return 1
        print(
            f"cycle {d} {t}Z: IFS-ENS {'landing on the Google mirror' if ok else 'NOT on the Google mirror'}",
            flush=True,
        )
        if a.github_output and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"date={d}\ntime={t}\nready={'true' if ok else 'false'}\n")
        return 0
    if not (a.date and a.time):
        ap.error(f"{a.cmd} needs --date and --time")
    if a.cmd == "wait":
        if C.wait_ready(a.date, a.time, C.IFS_ENS, a.wait_min):
            return 0
        print(f"::error::IFS-ENS {a.date} {a.time}Z is not complete on the Google mirror after {a.wait_min:.0f} min")
        return 1
    names = a.names.split(",") if a.names else [*SHARED, *ROWS]
    if (
        a.cmd == "paths"
    ):  # where the whole-field files are in the store's cache, with their sidecars (a job hands them on)
        for n in names:
            p = E.path(E.Cycle(a.date, a.time), FILES[n]())
            print(f"{p}\n{p}.json")
        return 0
    if a.cmd == "fetch" and not published(a.date, a.time):
        print(f"::warning::IFS-ENS {a.date} {a.time}Z is not on the Google mirror; nothing fetched", flush=True)
        return 0
    try:
        land(E.Cycle(a.date, a.time), names, a.wait_min if a.cmd == "land" else 0.0)
    except TimeoutError as e:
        print(f"::error::{e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
