#!/usr/bin/env python3
"""CDS fetch for the SEAS5 weather-regime product (seas5_regimes.py).

SEAS5 (C3S system 51) 500 hPa geopotential from `seasonal-original-pressure-levels`, 12-hourly
steps 12..5160 h (215 days), on the 2.5 deg grid of the regime centroids (the CDS regrids), over
one box that covers both sectors: 80-20N, 170W-30E (25 x 81 points, ~4 KB a field).

  forecast   51 members, one request per start          ~90 MB   data/seas5/regimes/fc_z500_{YYYYMM}.grib
  hindcast   25 members x 1981-2016 (the full SEAS5     ~1.6 GB  data/seas5/regimes/hc_z500_{MM}_{y0}_{y1}.grib
             hindcast), three years a request (32k fields), per start month

Requests run one at a time: the CDS rejects parallel requests from one key. Everything is cached
forever; a rerun only pulls what is missing. The AWS Planette icechunk copy of SEAS5 holds the
hindcast too, but its forecast pressure levels stop at the October 2025 start and a sector read
costs whole-globe chunks (~25 GB a start month), so the CDS is the one source for both.

    python seas5_regimes_fetch.py forecast --issue 202609
    python seas5_regimes_fetch.py hindcast --month 09
    python seas5_regimes_fetch.py all --issue 202609        # forecast, then that start's hindcast
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import DATA, CENTRE, SYSTEM, _client                    # noqa: E402

DIR = DATA / "regimes"
AREA = [80, -170, 20, 30]                          # N, W, S, E: both sectors (EA -90..30, NA -170..-30)
GRID = [2.5, 2.5]
HOURS = [str(h) for h in range(12, 5161, 12)]      # 430 steps: 12 h .. 5160 h (215 days)
HC_CHUNK = 3                                       # hindcast years per request
# the whole SEAS5 hindcast, not just the C3S 1993-2016 reference period: n = 36 years for the skill test
HC_YEARS = [str(y) for y in range(1981, 2017)]


def fc_file(ym: str) -> Path:
    return DIR / f"fc_z500_{ym}.grib"


def hc_chunks() -> list[list[str]]:
    ys = list(HC_YEARS)
    return [ys[i:i + HC_CHUNK] for i in range(0, len(ys), HC_CHUNK)]


def hc_file(month: str, years: list[str]) -> Path:
    return DIR / f"hc_z500_{month}_{years[0]}_{years[-1]}.grib"


def hc_files(month: str) -> list[Path]:
    return [hc_file(month, ys) for ys in hc_chunks()]


def _retrieve(req: dict, dest: Path, what: str) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS z500 {what} …", flush=True)
            _client().retrieve("seasonal-original-pressure-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                                                   # noqa: BLE001
            msg = str(e)
            print(f"    attempt {attempt + 1} failed ({msg[:160]})", flush=True)
            if "not available" in msg.lower() or "no data" in msg.lower():
                return False
            time.sleep(30)
    return False


def _req(years: list[str], month: str) -> dict:
    return {"originating_centre": CENTRE, "system": SYSTEM, "variable": ["geopotential"], "pressure_level": ["500"],
            "year": years, "month": [month], "day": ["01"], "leadtime_hour": HOURS,
            "area": AREA, "grid": GRID, "data_format": "grib"}


def fetch_forecast(ym: str) -> bool:
    return _retrieve(_req([ym[:4]], ym[4:]), fc_file(ym), f"forecast {ym} (51 members x 430 steps)")


def fetch_hindcast(month: str) -> bool:
    ok = True
    for ys in hc_chunks():
        ok &= _retrieve(_req(ys, month), hc_file(month, ys),
                        f"hindcast start {month} {ys[0]}-{ys[-1]} (25 members x {len(ys)} yr x 430 steps)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["forecast", "hindcast", "all"])
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--month", default=None, help="hindcast start month MM (default: the issue's)")
    a = ap.parse_args()
    ok = True
    if a.what in ("forecast", "all"):
        ok &= fetch_forecast(a.issue)
    if a.what in ("hindcast", "all"):
        ok &= fetch_hindcast(a.month or a.issue[4:])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
