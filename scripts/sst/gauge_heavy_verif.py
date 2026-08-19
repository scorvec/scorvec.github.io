#!/usr/bin/env python3
"""Does IMERG under-measure heavy rain? Gauge-referenced verification.

The delta model captures only ~37% of observed inflow-spike amplitude.
One candidate cause is upstream: if the satellite flattens the heavy tail
of the rainfall it is fed, no downstream model can put the spike back.

This checks that directly against IDEAM gauges (764 cached days,
2024-07 -> 2026-08, up to ~497 stations/day), on the RAW IMERG field -
never the gauge-corrected or gauge-blended one, which contain the gauge
and would make the comparison circular.

Two stratifications, because either one alone is misleading:

  by GAUGE bin     answers the user's question - when it really rained
                   hard, what did the satellite say? Regression to the
                   mean guarantees the satellite looks low here even if
                   it were unbiased, so the size of the shortfall matters,
                   not its sign.
  by SATELLITE bin the mirror image - when the satellite claimed heavy
                   rain, what fell? An unbiased sensor is low in the first
                   and high in the second; a genuinely under-measuring one
                   is low in the first and NOT high in the second.

Scale caveat, stated up front: a gauge is a point, an IMERG cell is a
~11 km area mean. Point extremes exceed area means for pure sampling
reasons, so part of any heavy-end shortfall is geometry, not sensor
error. The by-satellite panel is what separates them.

Timebase: gauge days are Colombia local (UTC-5), the IMERG cache is UTC.
Both alignments are scored so the offset cannot be mistaken for bias.

    python scripts/sst/gauge_heavy_verif.py [--min-stations 50]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402

PRIV = Path.home() / "colombia_hydro"
GAUGES = PRIV / "raw" / "gauges"
OUT = PRIV / "out" / "gauge_heavy_verif.json"
MAX_MM = 450.0
BINS = [0.1, 1, 5, 10, 20, 40, 60, 100, 1e9]
LBL = ["0.1-1", "1-5", "5-10", "10-20", "20-40", "40-60", "60-100", ">100"]


def pair(shift_days=0, min_stations=20):
    """(gauge_mm, imerg_raw_mm) at every station-day, RAW satellite field."""
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    G, S = [], []
    files = sorted(glob.glob(str(GAUGES / "*.json")))
    used = 0
    for f in files:
        day = Path(f).stem
        st = json.loads(Path(f).read_text())
        if len(st) < min_stations:
            continue
        d = (np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}")
             + np.timedelta64(shift_days, "D"))
        npy = IP.DAILY_CACHE / f"{str(d).replace('-','')}.npy"
        if not npy.exists():
            continue
        grid = np.load(npy)                       # RAW, no gauge correction
        used += 1
        for v in st.values():
            mm = float(v["mm"])
            if not (0 <= mm <= MAX_MM):
                continue
            j = int(np.searchsorted(lons, v["lo"]))
            i = int(np.searchsorted(lats, v["la"]))
            if not (0 < i < len(lats) and 0 < j < len(lons)):
                continue
            G.append(mm)
            S.append(float(grid[i, j]))
    return np.array(G), np.array(S), used


def table(x, y, xlab, ylab):
    """Mean of y in bins of x, with counts."""
    idx = np.digitize(x, BINS) - 1
    rows = []
    for b in range(len(LBL)):
        m = idx == b
        if m.sum() < 20:
            rows.append((LBL[b], int(m.sum()), np.nan, np.nan, np.nan))
            continue
        rows.append((LBL[b], int(m.sum()), float(x[m].mean()),
                     float(y[m].mean()), float(y[m].mean() / x[m].mean())))
    print(f"  {xlab+' bin':>10} {'n':>7} {'mean '+xlab:>10} "
          f"{'mean '+ylab:>10} {'ratio':>7}")
    for lb, n, mx, my, r in rows:
        if np.isnan(mx):
            print(f"  {lb:>10} {n:>7}          -          -       -")
        else:
            print(f"  {lb:>10} {n:>7} {mx:10.1f} {my:10.1f} {r:7.2f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stations", type=int, default=20)
    a = ap.parse_args(argv)

    best = None
    print("timebase check (gauge local vs IMERG UTC) - correlation by shift:")
    for sh in (-1, 0, 1):
        g, s, used = pair(sh, a.min_stations)
        wet = (g > 0.1) | (s > 0.1)
        r = float(np.corrcoef(g[wet], s[wet])[0, 1])
        print(f"   shift {sh:+d} d: {used:4} days, {len(g):7} station-days, "
              f"r = {r:.3f}")
        if best is None or r > best[0]:
            best = (r, sh, g, s, used)
    r, sh, g, s, used = best
    print(f"  -> using shift {sh:+d} d\n")

    print(f"{len(g):,} station-days over {used} days. "
          f"gauge mean {g.mean():.2f} mm, IMERG raw {s.mean():.2f} mm "
          f"(overall ratio {s.mean()/g.mean():.3f})\n")

    print("A. stratified by GAUGE - 'when it really rained, what did IMERG say?'")
    ra = table(g, s, "gauge", "imerg")
    print("\nB. stratified by IMERG - 'when IMERG claimed heavy, what fell?'")
    rb = table(s, g, "imerg", "gauge")

    # heavy-end detection: of gauge days above 40mm, what fraction does the
    # satellite even place above 20mm
    for thr, det in ((40, 20), (60, 30), (100, 50)):
        m = g >= thr
        if m.sum() >= 20:
            print(f"\ngauge >= {thr} mm (n={m.sum()}): IMERG mean "
                  f"{s[m].mean():5.1f} mm, median {np.median(s[m]):5.1f}, "
                  f"{100*(s[m]>=det).mean():4.1f}% reach {det} mm, "
                  f"{100*(s[m]<5).mean():4.1f}% under 5 mm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        dict(shift=sh, n=len(g), days=used,
             by_gauge=ra, by_imerg=rb,
             gauge_mean=float(g.mean()), imerg_mean=float(s.mean())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
