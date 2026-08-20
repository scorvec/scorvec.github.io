#!/usr/bin/env python3
"""INPE/CPTEC MERGE daily precipitation — the gauge-anchored truth for Brazil.

IMERG is satellite retrieval; MERGE is that same GPM field merged with
Brazil's surface gauge network by CPTEC. Over Brazil it is the better
reference, and it is on the **same 0.1 deg grid as our IMERG cache**, so
the two compare cell for cell with no regridding.

Domain 239.95-339.95 E, -60.05 to 32.25 N (all of South America), ~0.4 MB
per day.

    python scripts/sst/merge_cptec.py --backfill 30

Output: ~/brazil_hydro/raw/merge/MERGE_CPTEC_YYYYMMDD.grib2
"""
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path.home() / "brazil_hydro" / "raw" / "merge"
URL = ("https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/"
       "{y}/{m}/MERGE_CPTEC_{y}{m}{d}.grib2")


def fetch_day(day: datetime, force=False) -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"MERGE_CPTEC_{day:%Y%m%d}.grib2"
    if dest.exists() and dest.stat().st_size > 10000 and not force:
        return True
    u = URL.format(y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}")
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "scorvec-hydro/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  {day:%Y-%m-%d}: {repr(e)[:50]}", flush=True)
        return False
    dest.write_bytes(data)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    ok = 0
    for k in range(a.backfill, 0, -1):
        ok += bool(fetch_day(today - timedelta(days=k), a.force))
    print(f"MERGE: {ok}/{a.backfill} days available in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
