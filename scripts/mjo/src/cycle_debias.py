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
MAX_RUNS = 250  # cap the committed file: 2 cycles a day over the 120 days the rain channel's mean is taken on
N_PAIRS = 7  # trailing 00Z/12Z pairs used for the offset estimate (15 until 2026-10-02: the offset has grown
# from ~0.4 to ~1.1 RMM units at day 14 since July, and a shorter window tracks it better -
# replayed on the archive, day-10/14 residual 0.31/0.33 -> 0.27/0.31)
INDEX_V = 2  # RMM definition version (2 = full WH04 with precip pseudo-OLR,
# 2026-08-16); entries from another version are never paired —
# mixing definitions would bake the definition change into the
# 12Z offset estimate
MIN_PAIRS = 3  # below this, apply no correction (archive still warming up)
MIN_PAIRS_WIND = 1  # the wind-only channels were first archived 2026-10-02; their raw offset (~1 RMM unit at day 14)
# is several times the pair-to-pair scatter (~0.2), so even one pair removes most of it


def _init_ts(key: str) -> pd.Timestamp:
    return pd.Timestamp(f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[9:11]}:00")


def record(date: str, time: str, leads, rmm1, rmm2, path: Path = ARCHIVE, rmm1_wind=None, rmm2_wind=None) -> None:
    """Archive one run's RAW ensemble-mean trajectory (idempotent per cycle). The wind-only channels (used by the
    ENSO-removed product) are archived too since 2026-10-02 - they carry their own, different 12Z offset."""
    hist = json.loads(path.read_text()) if path.exists() else {}
    ent = {
        "v": INDEX_V,
        "leads": [round(float(x), 3) for x in leads],
        "rmm1": [round(float(x), 4) for x in rmm1],
        "rmm2": [round(float(x), 4) for x in rmm2],
    }
    if rmm1_wind is not None and rmm2_wind is not None:
        ent["rmm1_wind"] = [round(float(x), 4) for x in rmm1_wind]
        ent["rmm2_wind"] = [round(float(x), 4) for x in rmm2_wind]
    hist[f"{date}_{time}z"] = ent
    keep = sorted(hist, key=_init_ts)[-MAX_RUNS:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: hist[k] for k in keep}))


def offset_for(
    date: str, time: str, leads, path: Path = ARCHIVE, keys=("rmm1", "rmm2")
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
        mate = f"{key[:8]}_00z"  # same-date 00Z run, 12 h earlier
        if mate not in hist:
            continue
        b, a = hist[key], hist[mate]
        if b.get("v") != INDEX_V or a.get("v") != INDEX_V:
            continue  # never pair across RMM definitions
        if any(k not in b or k not in a for k in keys):
            continue  # runs archived before this channel was
        a_by_valid = {_init_ts(mate) + pd.Timedelta(days=ld): i for i, ld in enumerate(a["leads"])}
        matched = False
        for i, ld in enumerate(b["leads"]):
            vt = k_init + pd.Timedelta(days=ld)
            j = a_by_valid.get(vt)
            if j is None:
                continue
            diffs.setdefault(round(ld, 3), []).append((b[keys[0]][i] - a[keys[0]][j], b[keys[1]][i] - a[keys[1]][j]))
            matched = True
        n_used += matched
    if n_used < (MIN_PAIRS_WIND if keys[0].endswith("_wind") else MIN_PAIRS):
        return zero

    grid = np.array(sorted(diffs))
    off1 = np.array([np.mean([d[0] for d in diffs[g]]) for g in grid])
    off2 = np.array([np.mean([d[1] for d in diffs[g]]) for g in grid])
    return np.interp(leads, grid, off1), np.interp(leads, grid, off2)


RAIN_DAYS = 120  # the low-frequency window of Wheeler & Hendon (2004), as the winds' map (refresh_map120.py)
RAIN_MIN_RUNS = 20  # 00Z runs needed before the rain channel's mean is removed


def rain_channel_mean(date: str, time: str, leads, ratio, path: Path = ARCHIVE) -> tuple[np.ndarray, np.ndarray]:
    """(m1, m2) to SUBTRACT from the three-channel RMM at the given leads: the mean, over the 00Z runs of the last
    RAIN_DAYS days before this run, of what the rain (pseudo-OLR) channel adds to the index at each lead,
    rmm - ratio x rmm_wind (ratio: pc_wind_std / pc_std for the two modes, which puts the wind-only value on the
    three-channel scale).

    Wheeler & Hendon remove the previous 120-day mean from every field. The winds have it removed as a map; the
    rain is a model accumulation against an observed normal, so its 120-day mean is not ENSO alone but the model's
    own rain bias too, lead by lead, and left in it is a standing offset of the whole forecast (about -0.23 in RMM1
    in autumn 2026) that the wind-only analysed track does not have. A projection is linear, so the mean of the
    channel's part of the index is the part of the channel's mean. Zeros until RAIN_MIN_RUNS runs are archived."""
    leads = np.asarray(leads, dtype=float)
    zero = (np.zeros_like(leads), np.zeros_like(leads))
    if not path.exists():
        return zero
    hist = json.loads(path.read_text())
    init = _init_ts(f"{date}_{time}z")
    rows = [
        e
        for k, e in hist.items()
        if k[9:11] == "00"
        and pd.Timedelta(0) < init - _init_ts(k) <= pd.Timedelta(days=RAIN_DAYS)
        and e.get("v") == INDEX_V
        and "rmm1_wind" in e
    ]
    if len(rows) < RAIN_MIN_RUNS:
        return zero
    grid = np.array(sorted({round(ld, 3) for e in rows for ld in e["leads"]}))
    part = []
    for a, b, r in (("rmm1", "rmm1_wind", ratio[0]), ("rmm2", "rmm2_wind", ratio[1])):
        by = {g: [] for g in grid}
        for e in rows:
            for ld, x, w in zip(e["leads"], e[a], e[b], strict=True):
                by[round(ld, 3)].append(x - r * w)
        part.append(np.interp(leads, grid, [np.mean(by[g]) for g in grid]))
    return part[0], part[1]
