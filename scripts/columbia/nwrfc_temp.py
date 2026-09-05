"""NWRFC observed daily maximum and minimum temperature, on the divisions.

The NWRFC publishes the point temperature forcing behind its own river
forecasts as XML: `qte24max` and `qte24min`, 24 hours to 12Z -- the SAME day
boundary as the Stage IV precipitation this page already uses, which is the
reason to prefer it over any gridded product that would need its own window.

Why this source and not PRISM or URMA: both stop at the Canadian border and
would miss the three headwater divisions that are 20% of the basin. This
network runs from 40.7 N to 53.0 N, 2,007 stations, 184 of them north of
49 N. All 42 divisions get stations; only SATSOP has fewer than three.

**NWRFC retains about ten days.** There is no deeper archive, no
subdirectory, no directory listing -- the download portal lists twenty files
and that is all there is. So every day not captured is gone, and this script
exists to accrue a record from today rather than to backfill one.

**These are station-network means, not areal means, and the difference is
not small.** Inside the Columbia above Arrow the stations run 1,310 to 7,700
feet and 43 to 70 F on a single day, so the basin figure depends on where in
the elevation distribution the network happens to sit. `n` and the mean
station elevation are stored beside every value so the bias is visible and a
later elevation-corrected version can be checked against this one. Do not
treat these as the basin's true mean temperature.

    python scripts/columbia/nwrfc_temp.py            # today's file
    python scripts/columbia/nwrfc_temp.py --list     # what NWRFC still holds
-> columbia/data/temp/{date}.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

TEMP = os.path.join(P.DATA, "temp")
LIST = "https://www.nwrfc.noaa.gov/misc/downloads/downloads.php?type={t}"
FILE = "https://www.nwrfc.noaa.gov/weather/xml/{fn}"
KINDS = {"max": "observed_temperature_points", "min": "observed_temperature_points"}
ROW = re.compile(
    r'lid="([^"]+)"\s+per0="(-?\d+)"[^>]*?latitude="(-?[\d.]+)"\s+'
    r'longitude="(-?[\d.]+)"\s+elevation="(-?\d+)"')
MIN_STATIONS = 3


def listing(kind: str = "observed_temperature_points"):
    """[(filename, valid date)] newest first, from the portal's own JSON."""
    raw = P.get(LIST.format(t=kind), tries=3)
    if not raw:
        return []
    try:
        rows = json.loads(raw.decode("utf-8", "ignore"))
    except Exception:
        return []
    out = []
    for r in rows:
        fn = r.get("fn")
        if not fn:
            continue
        m = re.search(r"_(\d{8})\d{4}\.xml$", fn)
        if m:
            d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            out.append((fn, d.isoformat()))
    return out


def parse(xml: str):
    """(lon, lat, value F, elevation ft) arrays."""
    rows = ROW.findall(xml)
    if not rows:
        return None
    return (np.array([float(r[3]) for r in rows]),
            np.array([float(r[2]) for r in rows]),
            np.array([float(r[1]) for r in rows]),
            np.array([float(r[4]) for r in rows]))


_GEOM = None


def division_means(lon, lat, val, elev):
    """{code: {t, n, elev}} -- the mean of the stations inside each division."""
    global _GEOM
    from shapely import contains_xy
    if _GEOM is None:
        _GEOM = P.geoms()
    out = {}
    for c, g in _GEOM.items():
        m = contains_xy(g, lon, lat)
        n = int(m.sum())
        if n < MIN_STATIONS:
            continue
        out[c] = {"t": round(float(val[m].mean()), 1), "n": n,
                  "elev": int(round(float(elev[m].mean())))}
    return out


def fetch_day(fn_max: str, fn_min: str, date: str, force=False):
    os.makedirs(TEMP, exist_ok=True)
    p = os.path.join(TEMP, f"{date}.json")
    if os.path.exists(p) and not force:
        return json.load(open(p))
    got = {}
    for field, fn in (("max", fn_max), ("min", fn_min)):
        if not fn:
            continue
        raw = P.get(FILE.format(fn=fn), tries=3, timeout=180)
        if not raw:
            continue
        parsed = parse(raw.decode("utf-8", "ignore"))
        if not parsed:
            continue
        got[field] = division_means(*parsed)
    if not got.get("max") or not got.get("min"):
        return None
    rec = {"date": date, "source": "NWRFC qte24max/qte24min, 24 h to 12Z",
           "units": "F", "basis": "mean of stations inside each division, NOT elevation corrected",
           "max": got["max"], "min": got["min"]}
    json.dump(rec, open(p, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show what NWRFC still holds")
    ap.add_argument("--all", action="store_true", help="take every day still on the server")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    rows = listing()
    by_date = {}
    for fn, d in rows:
        by_date.setdefault(d, {})["max" if "24max" in fn else "min"] = fn
    if a.list:
        print(f"  NWRFC holds {len(by_date)} days:")
        for d in sorted(by_date, reverse=True):
            print(f"    {d}  {' '.join(sorted(by_date[d]))}")
        return
    days = sorted(by_date) if a.all else sorted(by_date)[-1:]
    ok = miss = 0
    for d in days:
        pair = by_date[d]
        r = fetch_day(pair.get("max"), pair.get("min"), d, a.force)
        if r:
            ok += 1
            n = len(r["max"])
            mx = np.mean([v["t"] for v in r["max"].values()])
            mn = np.mean([v["t"] for v in r["min"].values()])
            print(f"  {d}: {n} divisions, mean of division means {mn:.0f}/{mx:.0f} F")
        else:
            miss += 1
            print(f"  {d}: incomplete, not written")
    print(f"  temp: {ok} written, {miss} unavailable "
          f"({len(os.listdir(TEMP)) if os.path.isdir(TEMP) else 0} days held here)")


if __name__ == "__main__":
    main()
