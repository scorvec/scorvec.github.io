#!/usr/bin/env python3
"""
Fetch equatorial-Pacific subsurface temperature (depth x longitude x time) from
the NOAA/PMEL TAO/TRITON DISDEL delivery system, and parse the ASCII into a tidy
xarray Dataset.

The DISDEL "deliver" button submits a GET to /cgi-tao/cover.cgi (script
disdel/nojava.csh) with parameters P1..P22; the response is an HTML page that
links to a freshly generated cache file (…/t_xyzt_dy.ascii). We replicate that:
build the request -> parse the link -> download the ASCII -> parse to arrays.

  Variable  P2=t (subsurface temp)      Averaging P3=dy (daily)
  Equator   P20=P21=0 (lat 0)           Lons P18=137 (137E) .. P19=265 (95W)
  Format    P11=ascii  Structure P10=one (single multi-site file)

Usage:
    python tao_subsurface.py --days 150 --out data/tao_eq_temp.nc
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HOST = "https://www.pmel.noaa.gov"
CGI = HOST + "/cgi-tao/cover.cgi"
MISSING = -9.99
HEADERS = {"User-Agent": "Mozilla/5.0 (research; SST/RONI El Nino monitor)"}


def deliver(start: datetime, end: datetime, dest: Path) -> Path:
    """Request equatorial subsurface temperature over [start, end] and download
    the delivered t_xyzt_dy.ascii to ``dest``."""
    params = {
        "script": "disdel/nojava.csh",
        "P1": "deliv", "P2": "t", "P3": "dy",
        "P4": start.year, "P5": start.month, "P6": f"{start.day:02d}",
        "P7": end.year,   "P8": end.month,   "P9": f"{end.day:02d}",
        "P10": "one", "P11": "ascii", "P12": "None",
        "P13": "tt",                       # array: TAO/TRITON
        "P14": "anonymous", "P15": "anonymous",
        "P16": "anonymous", "P17": "disdel",
        "P18": "137", "P19": "265",        # min/max lon (137E .. 95W)
        "P20": "0", "P21": "0",            # min/max lat (equator)
        "P22": "html",
    }
    url = CGI + "?" + urllib.parse.urlencode(params)
    print(f"requesting DISDEL delivery {start:%Y-%m-%d} -> {end:%Y-%m-%d} …")
    html = _get(url).decode("latin-1", "replace")

    m = re.search(r'href="\s*(/cache-tao/[^"]*?_xyzt_dy\.ascii)\s*"', html)
    if not m:
        raise RuntimeError("No t_xyzt_dy.ascii link in DISDEL response "
                           "(check date range / availability)")
    data_url = HOST + m.group(1).strip()
    print(f"  downloading {data_url.split('/')[-1]} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(data_url))
    print(f"  saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def parse(path: Path) -> xr.Dataset:
    """Parse a TAO t_xyzt_dy.ascii (multi-site daily T(z)) into a Dataset with
    dims (time, depth, longitude); temperature in °C, missing -> NaN."""
    text = Path(path).read_text(errors="replace")
    lons = _floats(re.search(r"^\s*Lon\s*=\s*(.+)$", text, re.M).group(1))
    depths = _floats(re.search(r"^\s*Depth\s*=\s*(.+)$", text, re.M).group(1))
    ndep = len(depths)

    # Each site block lists only its own depths and an "Index:" line giving the
    # 1-based positions of those columns in the global Depth grid. Track the
    # current longitude (Location) and column->global-depth mapping (Index),
    # then scatter each daily row's values into the global (time, depth) grid.
    records = {}  # (time, lon) -> full-length profile (NaN-filled)
    cur_lon = None
    didx = None
    for line in text.splitlines():
        loc = re.match(r"\s*Location:\s+\S+\s+\S+\s+\(\s*(\d+),", line)
        if loc:
            cur_lon = lons[int(loc.group(1)) - 1]
            continue
        if line.lstrip().startswith("Index:"):
            didx = [int(x) - 1 for x in line.split()[1:]]
            continue
        row = re.match(r"\s*(\d{8})\s+\d{4}\s+(.*)$", line)
        if row and cur_lon is not None and didx:
            vals = row.group(2).split()[:len(didx)]
            if len(vals) < len(didx):
                continue
            try:
                v = np.array([float(x) for x in vals], float)
            except ValueError:
                continue
            t = pd.to_datetime(row.group(1), format="%Y%m%d")
            prof = records.setdefault((t, cur_lon), np.full(ndep, np.nan))
            prof[didx] = np.where(np.isclose(v, MISSING), np.nan, v)

    times = sorted({t for t, _ in records})
    ulons = sorted({lo for _, lo in records})
    arr = np.full((len(times), ndep, len(ulons)), np.nan)
    ti = {t: i for i, t in enumerate(times)}
    li = {lo: i for i, lo in enumerate(ulons)}
    for (t, lo), prof in records.items():
        arr[ti[t], :, li[lo]] = prof

    ds = xr.Dataset(
        {"temp": (("time", "depth", "longitude"), arr)},
        coords={"time": times, "depth": depths, "longitude": ulons},
        attrs={"source": "NOAA/PMEL TAO/TRITON DISDEL (t_xyzt_dy)",
               "units": "degC", "note": "daily mean valid 1200 UTC; SST at ~1 m"},
    )
    ds["depth"].attrs["units"] = "m"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=150, help="lookback window")
    ap.add_argument("--start", help="YYYYMMDD (overrides --days)")
    ap.add_argument("--end", help="YYYYMMDD (default: today UTC)")
    ap.add_argument("--ascii", default="data/tao_eq_temp.ascii")
    ap.add_argument("--out", default="data/tao_eq_temp.nc")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y%m%d") if args.end else datetime.now(timezone.utc)
    start = (datetime.strptime(args.start, "%Y%m%d") if args.start
             else end - timedelta(days=args.days))

    ascii_path = deliver(start, end, Path(args.ascii))
    ds = parse(ascii_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)
    nz = int(np.isfinite(ds["temp"]).sum())
    print(f"\nparsed -> {args.out}")
    print(f"  dims: {dict(ds.sizes)}  |  {nz} finite values")
    print(f"  longitudes (degE): {list(ds.longitude.values)}")
    print(f"  depths (m): {list(ds.depth.values.astype(int))}")
    print(f"  time: {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
