#!/usr/bin/env python3
"""IDEAM rain gauges at HOURLY resolution (Socrata s54a-sgyg).

The gauges report every **10 minutes**, not daily - `ideam_gauges.py` was
collapsing that to daily totals server-side because daily was all the
verification needed. For the sub-daily question it is the wrong
aggregation to throw away.

This matters because the half-hourly IMERG route to the same question is
badly constrained: GES DISC throttling puts a 400-day pull ~13 hours out,
and `pick_days` selects on rainfall, so the sample is conditioned on the
predictor and collapses the rain-inflow relationship by range restriction
(measured r = -0.036 against a full-record +0.46 on 2026-08-19).

The gauge route has neither problem. `date_extract_hh` aggregates
server-side, so one request returns every station-hour in the country for
a day - ~9,600 rows in a couple of seconds - and it covers EVERY day in
the cache rather than a rain-selected subset. It is also finer than IMERG
half-hourly, and it is an independent instrument, so it can verify the
satellite's sub-daily structure rather than merely echo it.

Timebase: `fechaobservacion` is Colombia local (UTC-5) with no zone
marker, so `hr` is LOCAL hour. Convert before comparing with IMERG UTC.

    python scripts/sst/ideam_gauges_hourly.py --backfill        # all cached days
    python scripts/sst/ideam_gauges_hourly.py --days 30

Output: one gzipped JSON per day under raw/gauges_hourly/
        {station: {"la":, "lo":, "h": {hour: mm}}}
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/s54a-sgyg.json"
CACHE = Path.home() / "colombia_hydro" / "raw" / "gauges_hourly"
DAILY_CACHE = Path.home() / "colombia_hydro" / "raw" / "gauges"
MAX_MM_HOUR = 200.0          # physical fence; world records sit near 300 mm/h
# Socrata throttles unauthenticated clients. The first backfill ran flat out
# and wrote 176 of 764 days before every later request began failing - and
# because a retry-exhausted day returns {} WITHOUT writing a file, the run
# still exited 0 and looked complete. Pace the requests and report failures
# loudly rather than discovering the hole downstream.
PACE_S = 1.5                 # minimum gap between requests
_last_call = [0.0]


def fetch_day(day: datetime, retries: int = 4) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{day:%Y%m%d}.json.gz"
    if f.exists():
        try:
            with gzip.open(f, "rt") as fh:
                cached = json.load(fh)
            age = (datetime.now(timezone.utc).replace(tzinfo=None) - day).days
            if cached or age > 45:
                return cached
        except (EOFError, OSError, ValueError):
            pass                                  # corrupt/partial -> refetch
    d0 = f"{day:%Y-%m-%d}T00:00:00"
    d1 = f"{day:%Y-%m-%d}T23:59:59"
    q = {
        "$select": ("codigoestacion,latitud,longitud,"
                    "date_extract_hh(fechaobservacion) AS hr,"
                    "sum(valorobservado) AS mm"),
        "$where": f"fechaobservacion >= '{d0}' AND fechaobservacion <= '{d1}'",
        "$group": ("codigoestacion,latitud,longitud,"
                   "date_extract_hh(fechaobservacion)"),
        "$limit": "200000",
    }
    url = BASE + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "scorvec-hydro/1.0"})
    rows = None
    for attempt in range(retries):
        gap = time.time() - _last_call[0]
        if gap < PACE_S:
            time.sleep(PACE_S - gap)
        _last_call[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                rows = json.load(r)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"   {day:%Y-%m-%d} FAILED {repr(e)[:60]}", flush=True)
                return None                      # None = failure, {} = no data
            time.sleep(min(60, 5 * 2 ** attempt))   # exponential, capped
    out: dict = {}
    for row in rows or []:
        try:
            code = row["codigoestacion"]
            la, lo = float(row["latitud"]), float(row["longitud"])
            hr, mm = int(row["hr"]), float(row["mm"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= mm <= MAX_MM_HOUR) or not (0 <= hr <= 23):
            continue
        st = out.setdefault(code, {"la": la, "lo": lo, "h": {}})
        st["h"][str(hr)] = round(mm, 2)
    with gzip.open(f, "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--backfill", action="store_true",
                    help="every day the DAILY gauge cache already holds")
    a = ap.parse_args(argv)

    if a.backfill:
        days = sorted(datetime.strptime(p.stem, "%Y%m%d")
                      for p in DAILY_CACHE.glob("*.json"))
    else:
        today = datetime.now(timezone.utc).replace(tzinfo=None)
        days = [today - timedelta(days=i) for i in range(a.days, 0, -1)]

    todo = [d for d in days if not (CACHE / f"{d:%Y%m%d}.json.gz").exists()]
    print(f"{len(days)} days requested, {len(todo)} to fetch", flush=True)
    ok = empty = failed = 0
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        r = fetch_day(d)
        if r is None:
            failed += 1
        elif r:
            ok += 1
        else:
            empty += 1
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"   {i}/{len(todo)}  {ok} data, {empty} empty, "
                  f"{failed} FAILED  ({el/i:.2f}s/day, "
                  f"~{(len(todo)-i)*el/i/60:.0f} min left)", flush=True)
    print(f"done: {ok} days with data, {empty} empty, {failed} failed -> {CACHE}")
    if failed:
        print(f"WARNING: {failed} days could not be fetched and wrote no file. "
              f"Rerun to retry them; do not treat this archive as complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
