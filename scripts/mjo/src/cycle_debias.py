"""
Remove the AIFS-ENS 00Z/12Z cycle bias from forecast RMM ("windshield wiper").

At the SAME valid time, 12Z-initialised AIFS-ENS runs forecast a tropical wind
state whose wind-RMM is systematically offset from 00Z-initialised runs — about
(+0.2, -0.35) in (RMM1, RMM2) by day 6, growing roughly linearly with lead
(verified Jul 2026 over 85 runs / 408 same-valid-time ensemble-mean pairs; the
offset is identical whether the runs are compared at 00Z or 12Z valid times, so
it is a property of the published fields, not of our sampling or projection).
Alternating 00Z/12Z frames therefore rocked back and forth in the animation.

Fix: treat the 00Z family as the reference (obs history and the CPC convention
are 00Z-anchored) and subtract from each 12Z run the trailing-window mean
(12Z minus same-date-00Z) ensemble-mean offset as a function of lead, estimated
from the raw ensemble-mean trajectories archived here each run.

Store: data/reference/ensmean_history.json
    {"<YYYYMMDD>_<hh>z": {"leads": [...], "rmm1": [...], "rmm2": [...]}, ...}
RAW (uncorrected) ensemble means only — archiving corrected 12Z trajectories
would collapse future offset estimates toward zero and decay the correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = Path("data/reference/ensmean_history.json")
MAX_RUNS = 120     # cap the committed file (~2 months of 2 cycles/day)
N_PAIRS = 15       # trailing 00Z/12Z pairs used for the offset estimate
INDEX_V = 2        # RMM definition version (2 = full WH04 with precip pseudo-OLR,
                   # 2026-08-16); entries from another version are never paired —
                   # mixing definitions would bake the definition change into the
                   # 12Z offset estimate
MIN_PAIRS = 3      # below this, apply no correction (archive still warming up)


def _init_ts(key: str) -> pd.Timestamp:
    return pd.Timestamp(f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[9:11]}:00")


def record(date: str, time: str, leads, rmm1, rmm2, path: Path = ARCHIVE) -> None:
    """Archive one run's RAW ensemble-mean trajectory (idempotent per cycle)."""
    hist = json.loads(path.read_text()) if path.exists() else {}
    hist[f"{date}_{time}z"] = {
        "v": INDEX_V,
        "leads": [round(float(x), 3) for x in leads],
        "rmm1": [round(float(x), 4) for x in rmm1],
        "rmm2": [round(float(x), 4) for x in rmm2],
    }
    keep = sorted(hist, key=_init_ts)[-MAX_RUNS:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: hist[k] for k in keep}))


def offset_for(date: str, time: str, leads, path: Path = ARCHIVE
               ) -> tuple[np.ndarray, np.ndarray]:
    """(offset_rmm1, offset_rmm2) to SUBTRACT from a run at the given leads.

    00Z runs are the reference family → zero. For a 12Z run, average the
    ensemble-mean (12Z minus same-date-00Z) difference at common valid times
    over the last N_PAIRS archived pairs strictly before this init (no
    lookahead, so regenerated history matches what a live run would have done),
    binned by the 12Z run's lead and linearly interpolated to ``leads``.
    """
    leads = np.asarray(leads, dtype=float)
    zero = (np.zeros_like(leads), np.zeros_like(leads))
    if int(time) != 12 or not path.exists():
        return zero

    hist = json.loads(path.read_text())
    init = _init_ts(f"{date}_{time}z")
    diffs: dict[float, list[tuple[float, float]]] = {}
    n_used = 0
    for key in sorted(hist, key=_init_ts, reverse=True):
        if n_used >= N_PAIRS:
            break
        k_init = _init_ts(key)
        if key[9:11] != "12" or k_init >= init:
            continue
        mate = f"{key[:8]}_00z"                    # same-date 00Z run, 12 h earlier
        if mate not in hist:
            continue
        b, a = hist[key], hist[mate]
        if b.get("v") != INDEX_V or a.get("v") != INDEX_V:
            continue                   # never pair across RMM definitions
        a_by_valid = {_init_ts(mate) + pd.Timedelta(days=ld): i
                      for i, ld in enumerate(a["leads"])}
        matched = False
        for i, ld in enumerate(b["leads"]):
            vt = k_init + pd.Timedelta(days=ld)
            j = a_by_valid.get(vt)
            if j is None:
                continue
            diffs.setdefault(round(ld, 3), []).append(
                (b["rmm1"][i] - a["rmm1"][j], b["rmm2"][i] - a["rmm2"][j]))
            matched = True
        n_used += matched
    if n_used < MIN_PAIRS:
        return zero

    grid = np.array(sorted(diffs))
    off1 = np.array([np.mean([d[0] for d in diffs[g]]) for g in grid])
    off2 = np.array([np.mean([d[1] for d in diffs[g]]) for g in grid])
    return np.interp(leads, grid, off1), np.interp(leads, grid, off2)
