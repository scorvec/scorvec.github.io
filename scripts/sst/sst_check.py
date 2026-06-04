#!/usr/bin/env python3
"""Is a new OISST day up at PSL yet? Prints PSL's latest available day + the file's
Last-Modified and compares with what the site currently shows — WITHOUT running the
pipeline or downloading the ~240 MB annual files. Read-only.

    python scripts/sst/sst_check.py

Exit code: 0 = site is current, 2 = a newer OISST day is available at PSL.
"""
from __future__ import annotations
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

PSL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
DODS = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"
ANOM = f"sst.day.anom.{dt.datetime.now(dt.timezone.utc).year}.nc"


def last_modified(fname: str) -> str:
    req = urllib.request.Request(f"{PSL}/{fname}", method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers.get("Last-Modified", "?")


def psl_latest_day(fname: str):
    import pandas as pd
    import xarray as xr
    ds = xr.open_dataset(f"{DODS}/{fname}", decode_times=True)   # OPeNDAP: metadata only
    return pd.to_datetime(ds.time.values[-1]).date()


def site_day():
    p = Path(__file__).resolve().parents[2] / "assets" / "sst" / "manifest.json"
    if p.exists():
        return json.loads(p.read_text()).get("sst_valid_day")
    return None


def main() -> int:
    lm = last_modified(ANOM)
    latest = psl_latest_day(ANOM)
    site = site_day()
    print(f"PSL latest OISST day : {latest}   (file Last-Modified {lm})")
    print(f"site currently shows : {site}")
    newer = site is None or str(latest) > str(site)
    print("→ NEW DAY AVAILABLE — run the pipeline to ingest it" if newer
          else "→ up to date (nothing to fetch)")
    return 2 if newer else 0


if __name__ == "__main__":
    raise SystemExit(main())
