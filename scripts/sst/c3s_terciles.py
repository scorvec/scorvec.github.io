#!/usr/bin/env python3
"""Fetch what the C3S tercile maps need: forecast members and the matching hindcast.

The anomaly maps on seasonal.html come from the postprocessed collection, which already
holds the anomaly against each centre's own hindcast and is tiny. Tercile probabilities
cannot be taken from an anomaly alone -- they need the hindcast DISTRIBUTION per grid
point (the 33rd and 67th percentiles of that model's own climate) and the spread of the
live members around it. So this pulls the raw fields instead:

  hindcast   1993-2016, the C3S common period, same start month, leads 1-6
  forecast   this issue's members, leads 1-6

on the 1 deg grid the SEAS5 page already uses, so every system is counted on one grid.
ECMWF SEAS5 is NOT downloaded: scripts/sst/data/seas5/{hindcast,forecast} already holds
exactly these fields for the SEAS5 page (hc_gl_{MM}.grib / fc_gl_{YYYYMM}.grib) and the
reader below points at them.

    python scripts/sst/c3s_terciles.py                 # this month, every system
    python scripts/sst/c3s_terciles.py --only dwd_22 --kind hc
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, _client

STORE = HERE / "data" / "c3s" / "terc"
SEAS5 = HERE / "data" / "seas5"
DATASET = "seasonal-monthly-single-levels"
HC_YEARS = [str(y) for y in range(1993, 2017)]          # the C3S common hindcast period
LEADS = ["1", "2", "3", "4", "5", "6"]
GRID = ["1", "1"]
# variable -> (CDS name, grib short name, factor to display units)
VARS = {
    "t2m": ("2m_temperature", "t2m", 1.0),
    "tp":  ("total_precipitation", "tprate", 86400.0 * 1000.0),
}


def local_seas5(kind: str, issue: str) -> Path:
    """The SEAS5 page's own global file, which holds t2m, tprate and sst on the 1 deg grid."""
    return (SEAS5 / "hindcast" / f"hc_gl_{issue[4:6]}.grib") if kind == "hc" \
        else (SEAS5 / "forecast" / f"fc_gl_{issue}.grib")


def path_for(kind: str, centre: str, system: str, var: str, issue: str) -> Path:
    if centre == "ecmwf":
        return local_seas5(kind, issue)
    stem = f"{kind}_{centre}_{system}_{var}_" + (issue[4:6] if kind == "hc" else issue)
    return STORE / f"{stem}.grib"


def fetch(kind: str, centre: str, system: str, var: str, issue: str, force: bool = False) -> bool:
    p = path_for(kind, centre, system, var, issue)
    if centre == "ecmwf":
        ok = p.exists()
        print(f"  {centre}/{system} {var} {kind}: {'local SEAS5 file' if ok else 'MISSING ' + str(p)}", flush=True)
        return ok
    if p.exists() and not force and p.stat().st_size > 1_000_000:
        print(f"  {centre}/{system} {var} {kind}: cached ({p.stat().st_size / 1e6:.0f} MB)", flush=True)
        return True
    STORE.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": str(system),
        "variable": [VARS[var][0]], "product_type": ["monthly_mean"],
        "year": HC_YEARS if kind == "hc" else [issue[:4]],
        "month": [issue[4:6]], "leadtime_month": LEADS,
        "data_format": "grib", "grid": GRID,
    }
    tmp = p.with_suffix(".part")
    t0 = time.time()
    try:
        _client().retrieve(DATASET, req, str(tmp))
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system} {var} {kind}: FAILED {str(e)[:120]}", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(p)
    print(f"  {centre}/{system} {var} {kind}: {p.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--only", default="", help="centre_system")
    ap.add_argument("--kind", default="hc,fc")
    ap.add_argument("--vars", default="t2m,tp")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    kinds = [k for k in a.kind.split(",") if k in ("hc", "fc")]
    want = [v for v in a.vars.split(",") if v in VARS]
    got = need = 0
    for entry in MODELS:
        centre, system = entry[0], str(entry[1])
        if a.only and f"{centre}_{system}" != a.only:
            continue
        for var in want:
            for kind in kinds:
                need += 1
                got += bool(fetch(kind, centre, system, var, a.issue, a.force))
    print(f"{got}/{need} file(s) ready under {STORE}")
    # the multi-model tercile is only honest if most systems are in it
    return 0 if got >= 0.75 * need else 1


if __name__ == "__main__":
    raise SystemExit(main())
