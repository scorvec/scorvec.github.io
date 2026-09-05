"""A day-of-year snow climatology from the SNODAS record, per division.

The day browser can show rain against a normal because NWRFC publishes
monthly precipitation normals. Snow has no such published normal here, so the
anomaly has to be taken against SNODAS's own record -- which is exactly the
right basis anyway: an anomaly is only meaningful against the climatology of
the same dataset, and mixing SNODAS against, say, a SNOTEL normal would put a
model difference into what reads as a weather signal.

Two honest limits, both carried in the output so the page can state them:

  * the record is 23 water years, so each day-of-year rests on roughly 23
    values before smoothing and rather more after;
  * before water year 2026 the archive is twice monthly, so the smoothing
    window is what supplies most of the sample. `n` is stored per point.

Sampled on a 5-day grid with a +/-12 day window and interpolated by the page:
snow climatology is smooth in time, so 74 points carry it as faithfully as
366 would at a fifth of the size.

    python scripts/columbia/snow_climo.py
-> columbia/data/snow_climo.json
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

SNOW = os.path.join(P.DATA, "snow")
OUT = os.path.join(P.DATA, "snow_climo.json")
GRID = list(range(1, 367, 5))
WINDOW = 12
MIN_N = 8


def doy(d: str) -> int:
    return dt.date.fromisoformat(d).timetuple().tm_yday


def main():
    area = P.area()
    recs = []
    for p in sorted(glob.glob(os.path.join(SNOW, "????-??-??.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        d = os.path.basename(p)[:-5]
        if not r.get("div"):
            continue
        recs.append((doy(d), r["div"], r.get("depth") or {}))
    if not recs:
        print("  no SNODAS records"); return
    print(f"  {len(recs)} analyses")

    codes = sorted({c for _, div, _ in recs for c in div})
    names = list(codes) + [n for n in P.COMPOSITES]

    def value(div, dep, name, field):
        src = div if field == "swe" else dep
        if name in P.COMPOSITES:
            cs = [c for c in P.members_of(name) if c in src]
            if not cs:
                return None
            w = np.array([area[c] for c in cs], float)
            return float(np.average([src[c] for c in cs], weights=w))
        return src.get(name)

    out = {}
    for name in names:
        cols = {}
        for field in ("swe", "depth"):
            vals, ns = [], []
            for g in GRID:
                # circular window on the day of year, so 1 January borrows
                # from late December rather than starting a fresh sample
                acc = []
                for dy, div, dep in recs:
                    dd = abs(dy - g)
                    if min(dd, 366 - dd) <= WINDOW:
                        v = value(div, dep, name, field)
                        if v is not None:
                            acc.append(v)
                vals.append(round(float(np.mean(acc)), 1) if len(acc) >= MIN_N else None)
                ns.append(len(acc))
            cols[field] = vals
            cols[field + "_n"] = ns
        out[name] = cols

    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "grid": GRID, "window_days": WINDOW, "min_n": MIN_N,
           "units": {"swe": "mm", "depth": "cm"},
           "source": "SNODAS masked 1034/1036 on the NWRFC divisions",
           "note": ("day-of-year mean over the SNODAS record; the archive is twice "
                    "monthly before water year 2026, so the +/-12 day window supplies "
                    "most of the sample -- n is given per point"),
           "climo": out}
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    sz = os.path.getsize(OUT)
    k = out["Columbia abv The Dalles"]["swe"]
    peak = max((v for v in k if v is not None), default=None)
    print(f"  {len(out)} basins, {len(GRID)} points each, {sz/1000:.0f} kB")
    print(f"  Columbia abv The Dalles SWE climatology peaks at {peak} mm "
          f"(day {GRID[k.index(peak)]}), n {out['Columbia abv The Dalles']['swe_n'][k.index(peak)]}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
