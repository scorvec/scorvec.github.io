#!/usr/bin/env python3
"""C3S multi-model seasonal anomaly maps — every centre, one selectable model at a time.

WHY THIS EXISTS. seas5_outlook.py pulls SEAS5's 51 members globally and caches 17 GB per start
month; doing that for all seven C3S centres would be ~100 GB and days of queue. It is also
unnecessary for maps: C3S publishes `seasonal-postprocessed-single-levels`, "Seasonal forecast
anomalies on single levels" — the ensemble-mean anomaly ALREADY taken against each model's own
1993-2016 hindcast, which is exactly the quantity a model-vs-model comparison wants, at roughly
1/50th the volume and with no hindcast download at all.

So: members stay with the Niño boxes (c3s_nino34.py, 499 members, the plume), and the maps come
from the postprocessed anomalies. SEAS5 keeps its own member-based products (calibrated terciles,
threshold days, P-E) because only SEAS5 has the members on disk — it is simply one of the models
you can select here.

    python scripts/sst/c3s_maps.py --issue 202609            # all models
    python scripts/sst/c3s_maps.py --issue 202609 --only ecmwf_51
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, _client                      # one model registry, one CDS client

DATASET = "seasonal-postprocessed-single-levels"
CACHE = HERE / "data" / "c3s" / "maps"
LEADS = ["1", "2", "3", "4", "5", "6"]
# The anomaly names in the postprocessed dataset differ from the raw ones.
FIELDS = {
    "t2m":  "2m_temperature_anomaly",
    "tp":   "total_precipitation_anomalous_rate_of_accumulation",
    "mslp": "mean_sea_level_pressure_anomaly",
    "sst":  "sea_surface_temperature_anomaly",
}


def path_for(key: str, centre: str, system: int, issue: str) -> Path:
    return CACHE / issue / f"{centre}_{system}_{key}.grib"


def fetch(key: str, centre: str, system: int, issue: str, force: bool = False) -> Path | None:
    dest = path_for(key, centre, system, issue)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": str(system),
        "variable": [FIELDS[key]],
        "product_type": ["ensemble_mean"],
        "year": [issue[:4]], "month": [issue[4:6]], "leadtime_month": LEADS,
        "data_format": "grib",
    }
    try:
        _client().retrieve(DATASET, req, str(dest))
        print(f"  {centre}/{system} {key}: {dest.stat().st_size/1e6:.1f} MB", flush=True)
        return dest
    except Exception as e:                                            # noqa: BLE001
        msg = str(e).replace("\n", " ")
        print(f"  {centre}/{system} {key}: unavailable ({msg[:110]})", file=sys.stderr, flush=True)
        dest.unlink(missing_ok=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True, help="YYYYMM")
    ap.add_argument("--only", default="", help="centre_system, e.g. ecmwf_51")
    ap.add_argument("--fields", default="t2m,tp,mslp,sst")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    want = [k for k in a.fields.split(",") if k in FIELDS]
    got = 0
    for entry in MODELS:
        centre, system = entry[0], entry[1]
        if a.only and f"{centre}_{system}" != a.only:
            continue
        for key in want:
            if fetch(key, centre, system, a.issue, a.force):
                got += 1
    print(f"{got} field(s) cached under {CACHE / a.issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
