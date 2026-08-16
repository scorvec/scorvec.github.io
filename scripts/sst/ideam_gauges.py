#!/usr/bin/env python3
"""IDEAM rain-gauge daily totals (datos.gov.co Socrata feed s54a-sgyg).

Server-side aggregation: one request per day returns per-station daily
totals (sum of sub-hourly tips) with coordinates — a few thousand rows
instead of ~300k raw observations. Cached one JSON per day under the
private research repo's raw/ tree; a rerun only fetches missing days.

Timebase note (ledger 2026-08-16): the feed's timestamps are Colombia
local (UTC-5) with no zone marker; gauge "days" are local calendar days
while the IMERG cache uses UTC days. The 5-hour offset is irrelevant for
7/30-day accumulations and acceptable for single-day comparison maps.

    python scripts/sst/ideam_gauges.py --days 30      # backfill cache
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/s54a-sgyg.json"
CACHE = Path.home() / "colombia_hydro" / "raw" / "gauges"
MAX_MM_DAY = 450.0                 # physical fence: world-class daily totals top ~400


def fetch_day(day: datetime) -> dict[str, dict]:
    """{station_code: {la, lo, mm}} for one local calendar day (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{day:%Y%m%d}.json"
    if f.exists():
        cached = json.loads(f.read_text())
        # empty days are re-checked for ~45 days: the feed has real multi-week
        # ingest holes (Jul 25-Aug 11 2026 verified server-side) that IDEAM
        # may backload later
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - day).days
        if cached or age > 45:
            return cached
    d0 = f"{day:%Y-%m-%d}T00:00:00"
    d1 = f"{day + timedelta(days=1):%Y-%m-%d}T00:00:00"
    q = {
        "$select": "codigoestacion,latitud,longitud,sum(valorobservado) AS mm",
        "$where": f"fechaobservacion >= '{d0}' AND fechaobservacion < '{d1}'",
        "$group": "codigoestacion,latitud,longitud",
        "$limit": "50000",
    }
    url = BASE + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "scorvec-hydro/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = json.load(r)
    out = {}
    for row in rows:
        try:
            la, lo, mm = float(row["latitud"]), float(row["longitud"]), float(row["mm"])
        except (KeyError, ValueError):
            continue
        if not (0 <= mm <= MAX_MM_DAY):
            continue
        out[row["codigoestacion"]] = {"la": round(la, 5), "lo": round(lo, 5),
                                      "mm": round(mm, 2)}
    f.write_text(json.dumps(out, separators=(",", ":")))
    return out


def fetch_range(end: datetime, n: int) -> dict[datetime, dict]:
    out = {}
    for k in range(n):
        d = end - timedelta(days=k)
        try:
            out[d] = fetch_day(d)
            print(f"  gauges {d:%Y-%m-%d}: {len(out[d])} stations", flush=True)
        except Exception as e:                        # noqa: BLE001
            print(f"  gauges {d:%Y-%m-%d} failed ({repr(e)[:50]})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    end = datetime.now(timezone.utc) - timedelta(days=1)
    fetch_range(end.replace(tzinfo=None), a.days)
