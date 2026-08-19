#!/usr/bin/env python3
"""Per-river inflow VOLUME (AporCaudal, m3/s) from XM — the hydrological target.

The stack has always modelled `AporEner` (kWh), which is volume x the head
available to that water through its whole downstream cascade. That makes
the target a product of hydrology AND fleet configuration, and the fleet
half moves: the AporEner/AporCaudal coefficient spans 27x across rivers
(BETANIA 14k, BOGOTA 385k kWh per m3/s) and 10 of 19 long-record rivers
shifted >10% since 2010. CAUCA SALVAJINA steps 20,278 -> 63,190 between
2022 and 2023 at Hidroituango commissioning — a 3.1x jump in the target
with no rainfall in it.

Rain drives water, not kilowatt-hours. This caches the volumetric series
so the hydrology can be fitted on its own terms and converted to energy
only at reporting time.

    python scripts/sst/xm_caudal_history.py [--from 2000-01-01]

Output: ~/colombia_hydro/raw/aporcaudal_daily.json.gz
        {"YYYY-MM-DD": {"RIVER": m3_per_s, ...}, ...}
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

API = "https://servapibi.xm.com.co/daily"
OUT = Path.home() / "colombia_hydro" / "raw" / "aporcaudal_daily.json.gz"
METRIC = "AporCaudal"


def fetch_range(d0: datetime, d1: datetime, have: dict) -> dict:
    out = dict(have)
    cur = d0
    n_new = 0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        # skip a fully-cached window; the series is final once published
        span = [(cur + timedelta(days=k)).strftime("%Y-%m-%d")
                for k in range((end - cur).days + 1)]
        if all(s in out for s in span):
            cur = end + timedelta(days=1)
            continue
        r = None
        for attempt in range(4):
            try:
                r = requests.post(API, json={"MetricId": METRIC,
                                             "StartDate": f"{cur:%Y-%m-%d}",
                                             "EndDate": f"{end:%Y-%m-%d}",
                                             "Entity": "Rio"}, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:                        # noqa: BLE001
                if attempt == 3:
                    print(f"  {cur:%Y-%m} FAILED {repr(e)[:60]}", flush=True)
                    r = None
                    break
                time.sleep(4 * (attempt + 1))
        if r is not None:
            for it in r.json().get("Items", []):
                day = it["Date"]
                for e in it.get("DailyEntities", []):
                    out.setdefault(day, {})[e["Name"].strip().upper()] = \
                        float(e["Value"])
                    n_new += 1
        if cur.month == 1 or (end - d0).days % 360 < 30:
            print(f"  {cur:%Y-%m}: {len(out)} days cached", flush=True)
        cur = end + timedelta(days=1)
    print(f"  {n_new} river-day values added")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default="2000-01-01")
    a = ap.parse_args(argv)
    have = {}
    if OUT.exists():
        with gzip.open(OUT, "rt") as fh:
            have = json.load(fh)
        print(f"existing cache: {len(have)} days")
    d0 = datetime.strptime(a.d0, "%Y-%m-%d")
    d1 = datetime.utcnow() - timedelta(days=1)
    out = fetch_range(d0, d1, have)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    days = sorted(out)
    print(f"wrote {OUT}: {len(days)} days {days[0]}..{days[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
