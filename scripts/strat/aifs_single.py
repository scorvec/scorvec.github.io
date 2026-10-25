"""AIFS single, ECMWF's deterministic AIFS, in the stratosphere products: the fields they share, fetched once a run.

WHAT ITS OPEN DATA CARRIES (index probed 2026-10-25): one -oper-fc file per step, 0.25 deg, runs at 00, 06, 12 and
18Z, every 6 h to 360 h. Pressure levels 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 and 10 hPa
(gh, z, t, u, v, w; q to 50 hPa): nothing between 50 and 10 hPa and nothing above 10 hPa, so no E-P flux and no upper
stratosphere. Sea-level pressure among the surface fields. Heights are read as gh (geopotential metres): packed in 12
bits it is within 0.53 m of z/g at 10 hPa (0.13 m at 1000 hPa, mean difference under 1 mm) at half z's bytes.

WHEN: the whole run is on the Google mirror ~7 h 15 min after its start (scripts/ecmwf/cycles.py --profile
aifs-single), ~15 min before AIFS-ENS, so its products have their own workflow (strat-aifs1.yml) and never wait on, or
delay, the AIFS-ENS job.

THE SHARED FILES (the store's cache, scripts/ecmwf/cache/{cycle}/aifs-single/), whole fields, fetched by whichever
product asks first (the store's per-file locks) and read by the others:
  wind   u, v and gh at 10 and 100 hPa, 12-hourly 0-360 h   vortex_winds; nh_vortex (u at 60N, the 100 hPa cap);
                                                            wave_maps, polarcap and forecast_drip (gh at 100 and 10)
  gh500  gh at 500 hPa, daily 0-360 h                       wave_maps, polarcap, forecast_drip
The products' other fields are their own, the leading rows of each message only (store.ensure_rows): forecast_drip's
cap at its eight other levels, polarcap's sea-level pressure and 10 hPa temperature.

ONE RUN: no members, so nothing is gated on member agreement. AIFS single is not AIFS-ENS: no AIFS-ENS correction is
ever applied to it (the dripping paint removes its own drift, measured on its own runs: forecast_drip.py
--measure-drift aifs1).

GOOGLE MIRROR ONLY (the store's rule): a run the mirror does not have yet is not fetched from anywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as E

STEPS = tuple(E.STEPS)  # 0..360 h by 24
STEPS_12H = tuple(range(0, 361, 12))
HEIGHTS = (500, 100, 10)  # the levels of gh in the shared files (wave_maps.LEVELS)


def spec_wind() -> E.Spec:
    return E.Spec("aifs-single", "fc", ("u", "v", "gh"), "pl", (10, 100), STEPS_12H)


def spec_gh500() -> E.Spec:
    return E.Spec("aifs-single", "fc", "gh", "pl", (500,), STEPS)


def height(cycle: E.Cycle, lev: int) -> Path:
    """The shared file holding gh at `lev` (one of HEIGHTS), fetched if it is not on disk yet."""
    if lev not in HEIGHTS:
        raise ValueError(f"no shared AIFS single height file at {lev} hPa (only {HEIGHTS})")
    return E.ensure(cycle, spec_gh500() if lev == 500 else spec_wind())
