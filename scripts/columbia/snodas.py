"""SNODAS snow water equivalent, averaged onto the NWRFC water-supply divisions.

Why SNODAS rather than a precipitation grid: water supply in this basin is
snowpack, and SNODAS is the only openly available analysis that covers the
WHOLE basin. PRISM stops at 49.9375 N, which leaves out the Columbia above
Arrow Dam, the Kootenai and the Middle Columbia upper tributaries -- 55,508 of
274,679 sq mi, 20.2% of the basin, and precisely the snow-dominated fifth of
it. SNODAS's masked grid reaches 52.875 N, which clears the northernmost
division (52.87 N) by a hair; measured coverage is 0.987 on all 42.

    https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{YYYY}/{MM}_{Mon}/SNODAS_{YYYYMMDD}.tar

about 15 MB a day, 30 September 2003 to present, no key. Product 1034 is SWE.

Three things the documentation does not make obvious, each of which cost a
run to find:
  * the header member is `.txt.gz`, not the `.Hdr.gz` the file naming implies;
  * the data is big-endian int16 -- and 32767 appears in the raw field as a
    fill alongside the documented -9999, so both have to be masked or a
    division mean comes back as tens of metres of snow;
  * the PNW subset is 1485 x 1888 cells, so the dense (division, cell) weight
    matrix the Stage IV path uses would be 942 MB. Here the weights are kept
    as one index array per division instead.

    python scripts/columbia/snodas.py --date 2026-03-01
    python scripts/columbia/snodas.py --season 2026          # Oct 2025-Sep 2026
    python scripts/columbia/snodas.py --history --day-of-month 1,15
-> columbia/data/snow/{date}.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import os
import sys
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402  (shares divisions(), geoms(), CACHE, get())

SNOW = os.path.join(P.DATA, "snow")     # alongside the Stage IV obs archive
URL = ("https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{y}/{m:02d}_{mon}/"
       "SNODAS_{d}.tar")
FIRST = dt.date(2003, 10, 1)
SWE_PRODUCT = "1034"
FILL = (-9999, 32767)
# The window the divisions live in. Subsetting before anything else keeps the
# whole job inside a few hundred MB.
BOX = (-125.0, -109.0, 40.5, 53.5)


def _url(d: dt.date) -> str:
    return URL.format(y=d.year, m=d.month, mon=d.strftime("%b"),
                      d=d.strftime("%Y%m%d"))


def _grid(hdr: str):
    """(nx, ny, lons, lats) from a SNODAS .txt header."""
    H = {}
    for line in hdr.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            H[k.strip()] = v.strip()
    nx, ny = int(H["Number of columns"]), int(H["Number of rows"])
    x0, x1 = float(H["Minimum x-axis coordinate"]), float(H["Maximum x-axis coordinate"])
    y0, y1 = float(H["Minimum y-axis coordinate"]), float(H["Maximum y-axis coordinate"])
    lons = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    lats = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny      # north-to-south rows
    return nx, ny, lons, lats


_IDX = None


def _division_index(lons, lats):
    """{code: flat indices into the subset} -- sparse, not a dense matrix.

    Cached on the subset shape. A dense (42, 2.8M) float matrix is 942 MB;
    the index arrays are a few MB.
    """
    global _IDX
    if _IDX is not None:
        return _IDX
    os.makedirs(P.CACHE, exist_ok=True)
    p = os.path.join(P.CACHE, f"snodas_idx_{len(lats)}x{len(lons)}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        _IDX = {k: z[k] for k in z.files}
        return _IDX
    from shapely import contains_xy
    lon2, lat2 = np.meshgrid(lons, lats)
    flon, flat = lon2.ravel(), lat2.ravel()
    out = {}
    for c, geom in P.geoms().items():
        x0, y0, x1, y1 = geom.bounds
        # a cheap bounding-box pass first; contains_xy on 2.8M points per
        # division would take minutes
        cand = np.flatnonzero((flon >= x0) & (flon <= x1) & (flat >= y0) & (flat <= y1))
        if cand.size:
            inside = contains_xy(geom, flon[cand], flat[cand])
            cand = cand[inside]
        out[c] = cand.astype(np.int32)
    np.savez_compressed(p, **out)
    _IDX = out
    return out


def swe_for(date: str, tries: int = 3):
    """{code: mean SWE mm} for one date, or None when the day is not published."""
    d = dt.date.fromisoformat(date)
    raw = P.get(_url(d), tries=tries, timeout=300)
    if not raw or len(raw) < 100000:
        return None
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw))
        mem = [n for n in tf.getnames() if SWE_PRODUCT in n]
        hdr = gzip.decompress(tf.extractfile(
            [n for n in mem if n.endswith("txt.gz")][0]).read()).decode("latin-1")
        dat = gzip.decompress(tf.extractfile(
            [n for n in mem if n.endswith("dat.gz")][0]).read())
    except Exception:
        return None
    nx, ny, lons, lats = _grid(hdr)
    a = np.frombuffer(dat, dtype=">i2").astype(np.float32).reshape(ny, nx)
    jm = (lons > BOX[0]) & (lons < BOX[1])
    im = (lats > BOX[2]) & (lats < BOX[3])
    sub = a[np.ix_(im, jm)]
    for f in FILL:
        sub = np.where(sub == f, np.nan, sub)
    flat = sub.ravel()
    idx = _division_index(lons[jm], lats[im])
    out = {}
    for c, ix in idx.items():
        if ix.size == 0:
            continue
        v = flat[ix]
        good = np.isfinite(v)
        if good.mean() < 0.9:            # same completeness rule as Stage IV
            continue
        out[c] = round(float(v[good].mean()), 1)
    return out or None


def store(date: str, force: bool = False):
    os.makedirs(SNOW, exist_ok=True)
    p = os.path.join(SNOW, f"{date}.json")
    if os.path.exists(p) and not force:
        return json.load(open(p))
    v = swe_for(date)
    if not v:
        return None
    rec = {"date": date, "source": "snodas_masked_swe_1034", "units": "mm", "div": v}
    json.dump(rec, open(p, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="one YYYY-MM-DD")
    ap.add_argument("--season", type=int, help="water year N = Oct N-1 .. Sep N")
    ap.add_argument("--history", action="store_true", help="every season from 2004")
    ap.add_argument("--day-of-month", default="1,15",
                    help="which days to take in --history/--season (all = every day)")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between fetches")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    days: list[str] = []
    if a.date:
        days = [a.date]
    else:
        seasons = range(2004, dt.date.today().year + 1) if a.history else \
            ([a.season] if a.season else [])
        want = None if a.day_of_month == "all" else \
            {int(x) for x in a.day_of_month.split(",")}
        for s in seasons:
            d = dt.date(s - 1, 10, 1)
            end = min(dt.date(s, 9, 30), dt.date.today() - dt.timedelta(days=1))
            while d <= end:
                if d >= FIRST and (want is None or d.day in want):
                    days.append(d.isoformat())
                d += dt.timedelta(days=1)
    if not days:
        ap.error("give --date, --season or --history")

    todo = [d for d in days if a.force or
            not os.path.exists(os.path.join(SNOW, f"{d}.json"))]
    print(f"  snodas: {len(todo)} of {len(days)} days to fetch", flush=True)
    ok = miss = 0
    for i, d in enumerate(todo, 1):
        if store(d, a.force):
            ok += 1
        else:
            miss += 1
        if i % 20 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)}  ok {ok}  unavailable {miss}", flush=True)
        time.sleep(a.pause)
    print(f"  snodas done: {ok} written, {miss} unavailable")


if __name__ == "__main__":
    main()
