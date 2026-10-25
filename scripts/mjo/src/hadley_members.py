#!/usr/bin/env python3
"""The members of one 00Z ensemble run for both Hadley forecast products, fetched once.

The overturning and Walker loops (overturn_fc.py) read u and v on 13 levels of every member at the 00 UTC state of lead
days 0..15; the outflow and descent loops (hadley_branch_fc.py) read u and v on ten of those levels with t, omega, z500
and sea-level pressure at the same states. Fetched apart, the winds came down twice. Here each lead day is fetched once
and handed to both: ECMWF (AIFS-ENS, IFS-ENS) from the Google Cloud mirror, DAY_PROCS lead days at once, the u/v
file read by overturn.reduce_file and then by hadley_branch_ref.states, which finds it on disk, adds the other fields
and deletes them all (~1 GB a day, never kept); GEFS member-steps from NOAA's S3 bucket in threads, both products'
messages in one byte-ranged pass per member-step, decoded here (eccodes in the fetch threads crashed the interpreter).

    python src/hadley_members.py --model ifs --date 20261023 --out runs [--only overturn|hbranch]
        -> runs/overturn_ifs.npz (vbar, ud: member x lead day x level) and runs/hbranch_ifs.npz (hadley_branch_fc's
           reduction; none when the ensemble is not drawn there)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import hadley_branch_fc as HF
import hadley_branch_ref as H
import hadley_branches as HB
import overturn as O
from mmsf import LEVELS

LEADS = H.LEADS  # lead days 0..15, the 00 UTC state of each (overturn_fc.LEADS too)
GEFS_UV = {(v, f"{L} mb") for v in ("UGRD", "VGRD") for L in LEVELS}  # overturn's GEFS messages
WORKERS = 16
# Lead days of an ECMWF ensemble at once (each ~1 GB fetched, then reduced): the reduction is the longer half, so this
# many keep the cores and the network busy together. Run with the numerical libraries single-threaded (OMP_NUM_THREADS=1
# etc.) so the processes do not fight over the cores.
DAY_PROCS = int(os.environ.get("HADLEY_DAY_PROCS", "4"))


def _arrays(n_ov: int, n_hb: int):
    """vbar (member, lead, lev, 721) and ud (member, lead, lev, 173) for overturn; (member, lead, field, row, lon) for
    hbranch. NaN until a member's lead day is read."""
    vb = np.full((n_ov, len(LEADS), len(LEVELS), O.LAT.size), np.nan, np.float32)
    ud = np.full((n_ov, len(LEADS), len(LEVELS), O.NLON_UD), np.nan, np.float32)
    x = np.full((n_hb, len(LEADS), len(H.FIELDS), H.ROWS.size, HB.LON.size), np.nan, np.float32)
    return vb, ud, x


def _ecmwf_day(job):
    """One lead day of an ECMWF ensemble: (j, vbar, ud, fields), each with the members first; None where not wanted."""
    import store as ecmwf

    model, init, j, overturn, hbranch = job
    cfg, L = O.MODELS[model], LEADS[j]
    cyc, s = ecmwf.Cycle(f"{init:%Y%m%d}", "00"), (24 * L,)
    types = (["cf"] if cfg["cf"] else []) + ["pf"]
    members = ([0] if cfg["cf"] else []) + list(range(1, ecmwf.PF_MEMBERS + 1))
    want = list(range(1, H.MODELS[model]["members"] + 1))
    vb, ud, x = _arrays(len(members) if overturn else 0, len(want) if hbranch else 0)
    vb, ud, x = vb[:, :1], ud[:, :1], x[:, :1]  # this day alone
    if overturn:
        for typ in types:
            p = ecmwf.ensure(cyc, ecmwf.Spec(cfg["store"], typ, ("u", "v"), "pl", tuple(LEVELS), s))
            mem = [0] if typ == "cf" else list(range(1, ecmwf.PF_MEMBERS + 1))
            v, u = O.reduce_file(p, mem, s)
            rows = [members.index(m) for m in mem]
            vb[rows, 0], ud[rows, 0] = v[:, 0], u[:, 0]
            if typ == "cf" or not hbranch:  # the members' file is hbranch's too: states() deletes it
                p.unlink(missing_ok=True)
    if hbranch:
        got = H.states(model, init, s, want)
        for i, m in enumerate(want):
            f = H.derived(got.pop((m, s[0]), {}))
            if f is not None:
                x[i, 0] = f
    n_o = int(np.isfinite(vb[:, 0, 0, 0]).sum()) if overturn else "-"
    n_h = int(np.isfinite(x[:, 0, 0, 0, 0]).sum()) if hbranch else "-"
    print(f"  {cfg['label']} day {L}: {n_o} members (overturning), {n_h} (outflow and descent)", flush=True)
    return j, vb[:, 0], ud[:, 0], x[:, 0]


def ecmwf_members(model: str, init: pd.Timestamp, overturn: bool, hbranch: bool):
    """(vbar, ud) for overturn (None when not wanted) and the hbranch members' fields (None when not wanted). The lead
    days run DAY_PROCS at a time in forked processes, each fetching its own day (one request budget, the store's
    per-file locks) while the others reduce theirs."""
    import multiprocessing as mp

    import store as ecmwf

    cfg = O.MODELS[model]
    members = ([0] if cfg["cf"] else []) + list(range(1, ecmwf.PF_MEMBERS + 1))
    vb, ud, x = _arrays(len(members) if overturn else 0, H.MODELS[model]["members"] if hbranch else 0)
    jobs = [(model, init, j, overturn, hbranch) for j in range(len(LEADS))]
    with mp.get_context("fork").Pool(DAY_PROCS) as pool:
        for j, v, u, f in pool.imap_unordered(_ecmwf_day, jobs):
            vb[:, j], ud[:, j], x[:, j] = v, u, f
    return ((vb, ud) if overturn else None), (x if hbranch else None)


def gefs_members(init: pd.Timestamp, overturn: bool, hbranch: bool):
    """As ecmwf_members for GEFS: every member-step fetched once with both products' messages; a member-step that
    fails stays NaN in both."""
    date = f"{init:%Y%m%d}"
    members = list(range(O.GEFS_PF + 1)) if overturn else list(range(1, H.MODELS["gefs"]["members"] + 1))
    want = list(range(1, H.MODELS["gefs"]["members"] + 1))
    vb, ud, x = _arrays(len(members) if overturn else 0, len(want) if hbranch else 0)

    def wants(m: int) -> set:
        return (GEFS_UV if overturn else set()) | (H.GEFS_WANT if hbranch and m in want else set())

    jobs = deque((i, j) for i in range(len(members)) for j in range(len(LEADS)))
    with ThreadPoolExecutor(WORKERS) as ex:
        live = deque()
        while jobs or live:
            while jobs and len(live) < 2 * WORKERS:  # at most this far ahead of the decoding, which runs here
                i, j = jobs.popleft()
                m, s = members[i], 24 * LEADS[j]
                live.append((i, j, ex.submit(H.gefs_member_step, date, m, s, wants(m))))
            i, j, fut = live.popleft()
            m = members[i]
            try:
                msgs = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  GEFS {init:%Y-%m-%d} member {m} day {LEADS[j]}: {type(e).__name__} {str(e)[:80]}", flush=True)
                continue
            if overturn:
                vb[i, j], ud[i, j] = O.gefs_reduce(msgs)  # takes u and v on its levels, passes over the rest
            if hbranch and m in want:
                st: dict = {}
                for g in msgs:
                    short, lev, _, _, f = H.decode(g)
                    H._put(st, short, lev, f)
                f = H.derived(st)
                if f is not None:
                    x[want.index(m), j] = f
    for j, L in enumerate(LEADS):
        n_o = int(np.isfinite(vb[:, j, 0, 0]).sum()) if overturn else "-"
        n_h = int(np.isfinite(x[:, j, 0, 0, 0]).sum()) if hbranch else "-"
        print(f"  GEFS day {L}: {n_o} members (overturning), {n_h} (outflow and descent)", flush=True)
    return ((vb, ud) if overturn else None), (x if hbranch else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(O.MODELS))
    ap.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    ap.add_argument("--out", required=True, type=Path, help="directory for overturn_<model>.npz, hbranch_<model>.npz")
    ap.add_argument("--only", choices=("overturn", "hbranch"), help="one product (the other already shows this run)")
    a = ap.parse_args()
    init = pd.Timestamp(a.date)
    overturn = a.only != "hbranch"
    hbranch = a.only != "overturn" and HF.drawn(a.model, init)
    if not (overturn or hbranch):
        return 0
    if a.model == "gefs":
        ov, x = gefs_members(init, overturn, hbranch)
    else:
        ov, x = ecmwf_members(a.model, init, overturn, hbranch)
    a.out.mkdir(parents=True, exist_ok=True)
    if ov is not None:
        np.savez_compressed(a.out / f"overturn_{a.model}.npz", vb=ov[0], ud=ov[1])
        print(f"wrote {a.out / f'overturn_{a.model}.npz'}", flush=True)
    if x is not None and HF.save_run(a.model, init, x, a.out / f"hbranch_{a.model}.npz"):
        print(f"wrote {a.out / f'hbranch_{a.model}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
