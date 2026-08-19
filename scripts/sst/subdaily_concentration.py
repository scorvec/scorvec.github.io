#!/usr/bin/env python3
"""Does WITHIN-DAY rainfall concentration explain inflow beyond the total?

The last untested explanation for the spike-amplitude gap. A basin-mean
daily total cannot distinguish 40 mm falling in three hours from 40 mm
spread evenly across a day, and only the first overwhelms infiltration
and routes quickly to the reservoir. If that distinction carries signal,
daily forcing is throwing it away.

Run on the HOURLY GAUGE archive (`ideam_gauges_hourly.py`), not the
half-hourly IMERG pull. The IMERG sample is selected on rainfall by
`pick_days`, so it is conditioned on the predictor and cannot even
reproduce the rain-inflow relationship (r = -0.036 against a full-record
+0.46). The gauge archive is unselected and covers every cached day.

Metrics per basin-day, all computed on the basin's gauge-network mean
hourly series:

  total      daily sum (mm)
  top3h      share of the day's rain in its wettest 3 hours
  peak       peak hourly rate / daily total
  entropy    Shannon entropy of the hourly shares, 1 = uniform, 0 = spike

A CONTROL is run first and reported prominently: unless daily total
reproduces its known positive correlation with next-day dInflow on this
sample, no weaker result from it is interpretable.

    python scripts/sst/subdaily_concentration.py [--min-gauges 8]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import region_weights_energy                # noqa: E402
import perfect_rain_backtest as PR                                 # noqa: E402
import delta_backtest_long as DB                                   # noqa: E402

PRIV = Path.home() / "colombia_hydro"
HOURLY = PRIV / "raw" / "gauges_hourly"
OUT = PRIV / "out" / "subdaily_concentration.json"
REGIONS = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


def collect(min_gauges):
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    W = region_weights_energy(lons, lats, REGIONS)
    masks = {r: (np.asarray(W[r]).reshape(len(lats), len(lons)) > 0) for r in W}

    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    di = {s: i for i, s in enumerate(dates)}
    stat = {}
    for b in REGIONS:
        y = d["y"][b]
        dyf = np.full(len(y), np.nan)
        dyf[1:] = y[1:] - y[:-1]          # rise INTO day t (see ALIGNMENT)
        stat[b] = (np.nanmean(dyf), np.nanstd(dyf))

    rows = []
    for f in sorted(glob.glob(str(HOURLY / "*.json.gz"))):
        day = Path(f).stem.split(".")[0]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        if iso not in di:
            continue
        i = di[iso]
        if i < 1:
            continue
        try:
            with gzip.open(f, "rt") as fh:
                st = json.load(fh)
        except (EOFError, OSError, ValueError):
            continue
        if not st:
            continue
        pts = []
        for v in st.values():
            j = int(np.abs(lons - v["lo"]).argmin())
            k = int(np.abs(lats - v["la"]).argmin())
            h = np.zeros(24)
            for hh, mm in v["h"].items():
                h[int(hh)] = mm
            pts.append((k, j, h))
        for r in masks:
            sel = [p for p in pts if masks[r][p[0], p[1]]]
            if len(sel) < min_gauges:
                continue
            H = np.mean([p[2] for p in sel], axis=0)      # basin mean, per hour
            tot = H.sum()
            if tot < 2.0:                                  # concentration is
                continue                                   # meaningless when dry
            srt = np.sort(H)[::-1]
            p = H / tot
            nz = p[p > 0]
            y = d["y"][r]
            # ALIGNMENT: AporEner for day t already contains the response to
            # day-t rain (tropical catchments respond within hours), so the
            # delta that rain[t] drives is y[t]-y[t-1]. Pairing rain[t] with
            # y[t+1]-y[t] pairs it with the RECESSION that follows and flips
            # the sign: measured +0.363 vs -0.193 pooled over basins. This
            # repo has made that mistake once before; do not reintroduce it.
            if not (np.isfinite(y[i]) and np.isfinite(y[i - 1])):
                continue
            mu, sd = stat[r]
            rows.append(dict(
                day=iso, region=r, ng=len(sel), tot=float(tot),
                top3h=float(srt[:3].sum() / tot),
                peak=float(srt[0] / tot),
                ent=float(-(nz * np.log(nz)).sum() / np.log(24)),
                zdy=float(((y[i] - y[i - 1]) - mu) / sd),
                ztot=float(tot)))
    return rows


def partial(x, y, z):
    from scipy import stats
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return r, float(2 * (1 - stats.t.cdf(abs(t), n - 3)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gauges", type=int, default=8)
    a = ap.parse_args(argv)
    rows = collect(a.min_gauges)
    n = len(rows)
    days = len({r["day"] for r in rows})
    print(f"{n} basin-days from {days} distinct days "
          f"(>= {a.min_gauges} gauges, >2 mm)\n")
    if n < 100:
        print("too few for inference — rerun once the backfill completes")
        return 0

    tot = np.array([r["tot"] for r in rows])
    zdy = np.array([r["zdy"] for r in rows])
    t3 = np.array([r["top3h"] for r in rows])
    pk = np.array([r["peak"] for r in rows])
    en = np.array([r["ent"] for r in rows])
    # rain enters the response nonlinearly; use log1p so the control is a
    # fair test of association rather than of functional form
    lt = np.log1p(tot)

    ctrl = float(np.corrcoef(lt, zdy)[0, 1])
    print(f"CONTROL  log daily total vs standardised same-day dInflow: "
          f"r = {ctrl:+.3f}")
    print("  like-for-like full-record benchmark: +0.363 pooled "
          "(+0.27..+0.41 by basin)")
    if ctrl < 0.15:
        print("  -> CONTROL FAILS. Nothing weaker is interpretable on this")
        print("     sample; reporting no concentration result.\n")
        return 0
    print("  -> control holds; concentration test is interpretable\n")

    print(f"concentration spread: top3h share {t3.min():.2f}-{t3.max():.2f} "
          f"(mean {t3.mean():.2f})\n")
    print(f"  {'metric':18} {'raw r':>8} {'partial r | rain':>18} {'p':>8}")
    res = {"n": n, "days": days, "control": ctrl}
    for nm, v in (("wettest-3h share", t3), ("peak-hour share", pk),
                  ("temporal entropy", en)):
        r0 = float(np.corrcoef(v, zdy)[0, 1])
        pr, pv = partial(v, zdy, lt)
        res[nm] = {"r": r0, "partial": pr, "p": pv}
        print(f"  {nm:18} {r0:8.3f} {pr:18.3f} {pv:8.3f}")
    print(f"\n  smallest partial r detectable at n={n}: ~{2.8/np.sqrt(n):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
