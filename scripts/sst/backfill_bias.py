#!/usr/bin/env python3
"""Backfill NWP rain cycles from the AWS ECMWF open-data archive.

The bucket (ecmwf-forecasts.s3.eu-central-1.amazonaws.com) retains past
cycles. For each backfill date we byte-range-fetch ONLY the tp messages
(via the per-step .index files) for a small member subset, and write
them as GRIB files named exactly like the live MJO downloads
(scripts/mjo/data/aifs/{model}_{date}_00z.{typ}.tp.grib2) — the
forecast engine then ingests, archives and verifies them through its
normal path, no special cases. 8 members is plenty for ensemble-mean
bias ratios; the fan always uses the newest (real, 51-member) cycles.

Steps fetched: 24..168 h (daily boundaries, leads 1-7 — the bands that
matter for bias correction). S3 throttles aggressively: paced requests
with exponential backoff on 503.

    python scripts/sst/backfill_bias.py [--days 21]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GRIB_DIR = REPO / "scripts" / "mjo" / "data" / "aifs"
ARCH = Path.home() / "colombia_hydro" / "raw" / "fcst_rain"
BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STEPS = list(range(24, 169, 24))
N_PF = 8                                   # pf members per model (ifs)
N_PF_AIFS = 7                              # + cf = 8 for aifs


def get(url: str, rng: str | None = None, tries: int = 7) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header("Range", f"bytes={rng}")
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(0.12)               # pacing — S3 throttles bursts
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2.0 * (k + 1))
        except Exception:                  # noqa: BLE001
            time.sleep(2.0 * (k + 1))
    return None


def fetch_tp(url_base: str, want) -> bytes | None:
    """Concatenated tp GRIB messages chosen by `want(entry)->bool`."""
    idx = get(url_base + ".index")
    if idx is None:
        return None
    out = bytearray()
    for line in idx.decode().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("param") != "tp" or not want(e):
            continue
        off, ln = int(e["_offset"]), int(e["_length"])
        blob = get(url_base + ".grib2", rng=f"{off}-{off + ln - 1}")
        if blob is None:
            return None
        out += blob
    return bytes(out) if out else None


def backfill_date(date: str) -> bool:
    """One 00Z cycle for both models. True if anything was written."""
    wrote = False
    # ── IFS: single -ef file per step, take pf members 1..N_PF ─────────────
    dst = GRIB_DIR / f"ifs_{date}_00z.pf.tp.grib2"
    arch = ARCH / f"ifs_{date}_00z.json.gz"
    if not dst.exists() and not arch.exists():
        buf = bytearray()
        ok = True
        for s in STEPS:
            u = f"{BASE}/{date}/00z/ifs/0p25/enfo/{date}000000-{s}h-enfo-ef"
            b = fetch_tp(u, lambda e: e.get("type") == "pf"
                         and int(e.get("number", 0)) <= N_PF)
            if b is None:
                ok = False
                break
            buf += b
        if ok and buf:
            dst.write_bytes(bytes(buf))
            print(f"  ifs {date}: {len(buf)/1e6:.0f} MB", flush=True)
            wrote = True
        else:
            print(f"  ifs {date}: unavailable", flush=True)
    # ── AIFS: separate -cf and -pf files per step ──────────────────────────
    for typ, want in (("cf", lambda e: True),
                      ("pf", lambda e: int(e.get("number", 0)) <= N_PF_AIFS)):
        dst = GRIB_DIR / f"aifs_{date}_00z.{typ}.tp.grib2"
        arch = ARCH / f"aifs_{date}_00z.json.gz"
        if dst.exists() or arch.exists():
            continue
        buf = bytearray()
        ok = True
        for s in STEPS:
            u = (f"{BASE}/{date}/00z/aifs-ens/0p25/enfo/"
                 f"{date}000000-{s}h-enfo-{typ}")
            b = fetch_tp(u, want)
            if b is None:
                ok = False
                break
            buf += b
        if ok and buf:
            dst.write_bytes(bytes(buf))
            print(f"  aifs {date} {typ}: {len(buf)/1e6:.0f} MB", flush=True)
            wrote = True
        else:
            print(f"  aifs {date} {typ}: unavailable", flush=True)
    return wrote


def main() -> int:
    ndays = 21
    if "--days" in sys.argv:
        ndays = int(sys.argv[sys.argv.index("--days") + 1])
    today = datetime.now(timezone.utc).date()
    n = 0
    for k in range(2, ndays + 2):          # start 2 days back (live covers today)
        date = (today - timedelta(days=k)).strftime("%Y%m%d")
        print(f"cycle {date} 00Z", flush=True)
        if backfill_date(date):
            n += 1
    print(f"backfilled {n} cycle-dates — run colombia_forecast.py to "
          f"extract + verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
