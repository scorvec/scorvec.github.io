#!/usr/bin/env python3
"""Build the 2 m-temperature ensemble products for a cycle — AIFS-ENS + ECMWF
IFS-ENS ensemble MEANS (via the established store / open-data pipeline):

  1. run-to-run delta : current run − previous (24 h) run at the same valid time,
                        2-panel (AIFS | IFS), Day 0–15 lead-slider animation.
  2. 48-h run-to-run   : for each valid time, the least-squares slope (°C/day) of the
     forecast trend     combined-ensemble forecast across the last 48 h of runs — which
                        way the forecast for that date is trending. 1-panel animation.

Both need run history (the archive); they populate as cycles accrue (delta needs ≥2
same-hour runs, the trend ≥3). TODO: add 6-hourly 2 m-temp steps for true daily means.

    python src/run_temps.py --date 20260604 --run 00
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fetch as F
import render as R
import archive as A

ENS = ["ifs", "aifs"]            # panel order: IFS then AIFS
VAR = "t2m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    ap.add_argument("--out-root", default="../../assets/ens")
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date}T{a.run}:00")
    out = Path(a.out_root); anim = out / "anim"

    print(f"== t2m: fetching {ENS} (ensemble means) ==", flush=True)
    meds = {}
    for e in ENS:
        try:
            meds[e] = F.fetch(e, a.date, a.run, VAR)
        except Exception as ex:                                  # noqa: BLE001
            print(f"  {e} t2m: FAILED ({repr(ex)[:80]})", flush=True)
    if not meds:
        print("  no ensembles fetched; aborting"); return 1
    meds["combined"] = np.nanmean(np.stack([meds[e] for e in meds]), axis=0).astype("float32")

    # archive each run's mean first (incl. combined) so the change/trend can look back
    for e in meds:
        A.archive_run(e, VAR, init, meds[e])

    # 1. run-to-run delta (2-panel AIFS|IFS), same valid time vs the 24 h-earlier run
    print("== run-to-run delta (vs previous cycle) ==", flush=True)
    fields = {e: ch for e in ENS
              if (ch := A.change_fields(e, VAR, init, meds[e], 24)) is not None}
    if fields:
        R.render_change(fields, {e: meds[e] for e in fields}, VAR, "dprev", init,
                        anim / "t2m_dprev", anim / "t2m_dprev_manifest.json")
    else:
        print("  no run 24 h earlier archived yet — skipping the delta this cycle", flush=True)

    # 2. 48-h run-to-run forecast trend (combined, 1-panel animation)
    print("== 48-h run-to-run forecast trend (combined) ==", flush=True)
    tr = A.trend_fields("combined", VAR, init, meds["combined"], hours_back=48)
    if tr is not None:
        R.render_trend(tr, VAR, init, anim / "t2m_trend48", anim / "t2m_trend48_manifest.json")
    else:
        print("  need 3 archived runs (now, −24h, −48h) for the trend — skipping", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
