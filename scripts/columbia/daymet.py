"""Daymet on the NWRFC divisions: 46 water years of rain, temperature and snow.

Why Daymet. Every question the ENSO page asks was underpowered on 23 water
years -- the December-rainfall correlations were nine points, enough for two
shared trends to masquerade as r = -0.92. Daymet runs 1980-2025, covers North
America INCLUDING the Canadian headwaters that PRISM and URMA miss, and
carries precipitation, tmax, tmin and SWE on one grid, so the three
quantities are internally consistent and their anomalies comparable.

Access is the single-pixel API: no key, and one request returns a point's
whole 46-year daily series in about two seconds.

    https://daymet.ornl.gov/single-pixel/api/data?lat=&lon=&vars=&start=&end=

Two things to keep straight:

  * Daymet SWE is MODELLED -- its own snow model driven by its own
    temperature and precipitation -- not an observation-based analysis like
    SNODAS. Good for asking how snowpack co-varies with ENSO over 46 years;
    not interchangeable with the SNODAS record, and the two should never be
    concatenated into one series.
  * Daymet uses a 365-day calendar and drops 31 December in leap years, so a
    yearday is not a calendar day without conversion.

Basin values are the cos(latitude)-weighted mean of sample points drawn on a
regular grid inside each division. That is an areal mean rather than the
station-network mean the NWRFC temperature archive gives, which is the other
reason to have it.

    python scripts/columbia/daymet.py --probe          # points per division
    python scripts/columbia/daymet.py [--points 20]
-> columbia/data/daymet_monthly.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

OUT = os.path.join(P.DATA, "daymet_monthly.json")
CACHE = os.path.join(P.CACHE, "daymet")
API = ("https://daymet.ornl.gov/single-pixel/api/data"
       "?lat={lat:.4f}&lon={lon:.4f}&vars=prcp,tmax,tmin,swe&start={y0}-01-01&end={y1}-12-31")
Y0, Y1 = 1980, 2025
VARS = ("prcp", "tmax", "tmin", "swe")


def sample_points(geom, n_target: int):
    """~n_target points on a regular grid inside the polygon."""
    from shapely import contains_xy
    x0, y0, x1, y1 = geom.bounds
    for k in range(4, 60):
        xs = np.linspace(x0, x1, k)
        ys = np.linspace(y0, y1, k)
        gx, gy = np.meshgrid(xs, ys)
        m = contains_xy(geom, gx.ravel(), gy.ravel())
        if m.sum() >= n_target:
            pts = np.c_[gx.ravel()[m], gy.ravel()[m]]
            if len(pts) > n_target:                     # thin evenly, keep the spread
                idx = np.linspace(0, len(pts) - 1, n_target).astype(int)
                pts = pts[idx]
            return [(float(b), float(a)) for a, b in pts]   # (lat, lon)
    c = geom.representative_point()
    return [(float(c.y), float(c.x))]


def fetch_point(lat, lon, tries=3):
    """A point's daily series as a DataFrame, cached on disk."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{lat:.4f}_{lon:.4f}.csv")
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        raw = open(p, "rb").read()
    else:
        raw = P.get(API.format(lat=lat, lon=lon, y0=Y0, y1=Y1), tries=tries, timeout=300)
        if not raw or len(raw) < 10000:
            return None
        open(p, "wb").write(raw)
    try:
        txt = raw.decode("utf-8", "ignore")
        head = txt.split("\n", 8)
        elev = next((float(l.split(":")[1].split()[0]) for l in head if l.startswith("Elevation")), None)
        d = pd.read_csv(io.StringIO(txt), skiprows=6)
    except Exception:
        return None
    d.columns = [c.split(" ")[0] for c in d.columns]
    if "prcp" not in d.columns:
        return None
    d.attrs["elev"] = elev
    return d


def monthly(d):
    """Daily -> monthly, on the CALENDAR month a Daymet yearday falls in.

    Daymet drops 31 December in leap years, so the yearday must be turned
    into a real date per year rather than mapped by a fixed table.
    """
    import pandas as pd
    date = [dt.date(int(y), 1, 1) + dt.timedelta(days=int(j) - 1)
            for y, j in zip(d.year.values, d.yday.values)]
    d = d.assign(mo=[x.month for x in date])
    g = d.groupby(["year", "mo"])
    out = g.agg(prcp=("prcp", "sum"), tmax=("tmax", "mean"),
                tmin=("tmin", "mean"), swe=("swe", "mean"), swe_max=("swe", "max"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=20, help="sample points per division")
    ap.add_argument("--pause", type=float, default=0.6)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    geoms = P.geoms()
    plan = {c: sample_points(g, a.points) for c, g in geoms.items()}
    tot = sum(len(v) for v in plan.values())
    if a.probe:
        print(f"  {len(plan)} divisions, {tot} points")
        for c in list(plan)[:6]:
            print(f"    {c:12s} {len(plan[c]):3d}")
        print(f"  at ~2 s a point that is about {tot * 2 / 60:.0f} minutes")
        return

    print(f"  {tot} points over {len(plan)} divisions, {Y0}-{Y1}", flush=True)
    per, elevs, done, miss = {}, {}, 0, 0
    for c, pts in plan.items():
        acc, w, ev = [], [], []
        for lat, lon in pts:
            d = fetch_point(lat, lon)
            done += 1
            if d is None:
                miss += 1
                continue
            acc.append(monthly(d))
            w.append(np.cos(np.radians(lat)))       # area weight
            if d.attrs.get("elev") is not None:
                ev.append(d.attrs["elev"])
            time.sleep(a.pause)
        if not acc:
            continue
        w = np.array(w, float); w /= w.sum()
        base = acc[0]
        stacked = {k: np.zeros(len(base)) for k in ("prcp", "tmax", "tmin", "swe", "swe_max")}
        for frame, wi in zip(acc, w):
            f = frame.reindex(base.index)
            for k in stacked:
                stacked[k] += wi * f[k].values
        per[c] = {"index": [[int(y), int(m)] for y, m in base.index],
                  **{k: [round(float(v), 2) for v in stacked[k]] for k in stacked}}
        elevs[c] = int(round(float(np.mean(ev)))) if ev else None
        print(f"    {c:12s} {len(acc):3d} pts, mean elev {elevs[c]} m  ({done}/{tot})", flush=True)

    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "Daymet V4 R1 single-pixel API, ORNL DAAC",
           "period": [Y0, Y1], "points_per_division": a.points,
           "units": {"prcp": "mm/month", "tmax": "degC", "tmin": "degC",
                     "swe": "kg/m2 monthly mean", "swe_max": "kg/m2 monthly max"},
           "note": ("cos(lat)-weighted mean of grid points inside each division; "
                    "Daymet SWE is MODELLED, not an analysis like SNODAS, and the two "
                    "must not be concatenated"),
           "elev_m": elevs, "div": per}
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    print(f"  {len(per)} divisions, {miss} points unavailable, "
          f"{os.path.getsize(OUT)/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
