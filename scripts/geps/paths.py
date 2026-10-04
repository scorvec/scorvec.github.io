"""Where the GEPS subseasonal pipeline reads and writes. One place, so nothing else carries a path.

The pipeline moved off the laptop on 2026-10-04 (user rule: no downloads on the laptop). It used to live in
~/data_archive/geps_subx next to its 150 GB of hindcast and reanalysis inputs; in Actions it has only the compact
DERIVED reference files (scripts/geps/make_reference.py, built once on the laptop from those local stores and
published as the release asset geps-ref-v1) plus whatever it fetches for the new cycle.

  REF    read-only reference: lead-dependent GEPS8 model climatology packs, the ERA5 1991-2020 minus 2001-2020 shift,
         teleconnection patterns, hindcast skill/calibration tables, regime centroids          (GEPS_REF)
  STATE  small persistent state carried between runs on the frames branch: GEPS day-0 analysis caches (the observed
         tails), refreshed CPC/PSL/Long Paddock index files, the previous cycle's compact anomalies and members (the
         change panels and previous-run overlays), the vortex series                            (GEPS_STATE)
  WORK   per-run scratch: full-resolution live files, rendered figures and frames, the GRIB cache (GEPS_WORK)

Every directory is gitignored; the defaults sit inside the repo so the frames-branch seed/publish helpers, which take
repo-relative paths, can carry STATE.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SITE_ROOT = Path(os.environ.get("SITE_ROOT", REPO))
SITE_ASSETS = SITE_ROOT / "assets" / "geps"

REF = Path(os.environ.get("GEPS_REF", HERE / "ref"))
STATE = Path(os.environ.get("GEPS_STATE", HERE / "state"))
WORK = Path(os.environ.get("GEPS_WORK", HERE / "work"))

CLIM = REF                       # geps8_clim_<tag>.npz, era5_shift.nc
TC_REF = REF / "telecon"         # static teleconnection / skill reference
TC_STATE = STATE / "telecon"     # refreshed observed indices, analysis tails, json summaries
ANA = TC_STATE / "analysis"      # GEPS day-0 ensemble-mean fields, 2.5 deg, one npz per day
ARCHIVE = STATE / "runs"         # the previous cycle, compact (change panels, previous-run overlays)
LIVE = WORK / "live"
FIGS = WORK / "figs"

# Laptop-only inputs of the one-off reference builds (never read in Actions, never downloaded again: user rule
# 2026-10-04; catalogue ~/data_stores/INDEX.md).
LOCAL_STORE = Path(os.environ.get("GEPS_LOCAL_STORE", Path.home() / "data_archive" / "geps_subx"))

# Honest identification for every upstream request (user rule 2026-10-04).
USER_AGENT = "scorvec.com data pipeline (+https://scorvec.com)"
UA = {"User-Agent": USER_AGENT}


def ensure() -> None:
    for d in (TC_STATE, ANA, ARCHIVE, LIVE, FIGS, SITE_ASSETS):
        d.mkdir(parents=True, exist_ok=True)
