#!/usr/bin/env python3
"""Is IMERG's heavy-rain shortfall real, or regression to the mean?

gauge_heavy_basin.py finds IMERG at 3.8x the gauge basin mean on light
days and 0.53x on days above 30 mm. That pattern is exactly what a
genuinely intensity-dependent retrieval bias looks like - and also
exactly what pure regression to the mean produces when you bin on a
NOISY reference. The gauge basin mean is itself an estimate from ~19
stations, so binning on it selects for gauge-side noise and drags the
ratio down at the top end even for a perfect sensor.

The fix is a split-half control. Randomly halve the gauges in each
region-day: bin on half A, verify against half B. A and B share no
measurement noise, so the noise that defined the bin cannot bias the
reference. Two readings then separate cleanly:

  B/A ratio      the pure regression-to-the-mean baseline. A perfect
                 instrument compared this way still shows < 1 at the top.
  IMERG/B ratio  the real bias, on the same axis and same days.

If IMERG/B tracks B/A, the shortfall was an artefact. If IMERG/B falls
well below it, the satellite really is flattening heavy rain.

Seeded, and averaged over N_REP independent splits.

    python scripts/sst/gauge_split_half.py [--min-gauges 8] [--reps 40]
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
from hydro_region_rain import region_weights_energy                # noqa: E402

PRIV = Path.home() / "colombia_hydro"
GAUGES = PRIV / "raw" / "gauges"
OUT = PRIV / "out" / "gauge_split_half.json"
REGIONS = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BINS = [0.5, 2, 5, 10, 15, 20, 30, 1e9]
LBL = ["0.5-2", "2-5", "5-10", "10-15", "15-20", "20-30", ">30"]


def collect(min_gauges):
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    W = region_weights_energy(lons, lats, REGIONS)
    masks = {r: (np.asarray(W[r]).reshape(len(lats), len(lons)) > 0) for r in W}
    days = []
    for f in sorted(glob.glob(str(GAUGES / "*.json"))):
        day = Path(f).stem
        st = json.loads(Path(f).read_text())
        npy = IP.DAILY_CACHE / f"{day}.npy"
        if not st or not npy.exists():
            continue
        grid = np.load(npy)
        pts = []
        for v in st.values():
            mm = float(v["mm"])
            if not (0 <= mm <= 450):
                continue
            j = int(np.abs(lons - v["lo"]).argmin())
            i = int(np.abs(lats - v["la"]).argmin())
            pts.append((i, j, mm))
        for r in masks:
            sel = [(i, j, mm) for i, j, mm in pts if masks[r][i, j]]
            if len(sel) < min_gauges:
                continue
            days.append((np.array([x[2] for x in sel]),
                         np.array([grid[x[0], x[1]] for x in sel])))
    return days


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gauges", type=int, default=8)
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args(argv)

    days = collect(a.min_gauges)
    print(f"{len(days)} region-days with >= {a.min_gauges} gauges "
          f"(median {np.median([len(g) for g, _ in days]):.0f} gauges)\n")

    rng = np.random.default_rng(20260819)
    acc = {b: dict(nA=[], A=[], B=[], S=[]) for b in range(len(LBL))}
    for _ in range(a.reps):
        for gv, sv in days:
            n = len(gv)
            idx = rng.permutation(n)
            ia, ib = idx[: n // 2], idx[n // 2:]
            A, B = gv[ia].mean(), gv[ib].mean()
            S = sv[ib].mean()            # IMERG at half-B's own pixels
            b = int(np.digitize(A, BINS) - 1)
            if 0 <= b < len(LBL):
                acc[b]["A"].append(A)
                acc[b]["B"].append(B)
                acc[b]["S"].append(S)

    print("binned on gauge half A; verified against gauge half B and IMERG")
    print(f"  {'bin (A)':>9} {'n/rep':>7} {'meanA':>7} {'meanB':>7} "
          f"{'B/A':>6} {'IMERG':>7} {'S/B':>6} {'excess':>7}")
    tab = []
    for b in range(len(LBL)):
        if len(acc[b]["A"]) < 10 * a.reps / 10:
            continue
        A = np.mean(acc[b]["A"]); B = np.mean(acc[b]["B"])
        S = np.mean(acc[b]["S"])
        n = len(acc[b]["A"]) / a.reps
        ba, sb = B / A, S / B
        tab.append([LBL[b]] + [float(x) for x in (n, A, B, ba, S, sb, sb / ba)])
        print(f"  {LBL[b]:>9} {n:7.0f} {A:7.1f} {B:7.1f} {ba:6.2f} "
              f"{S:7.1f} {sb:6.2f} {sb/ba:7.2f}")
    print("\n  B/A    = regression-to-the-mean baseline (perfect sensor)")
    print("  S/B    = IMERG vs an independent gauge half")
    print("  excess = (S/B)/(B/A); 1.00 means IMERG is no worse than the")
    print("           artefact alone, < 1 means a real intensity-dependent bias")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(reps=a.reps, table=tab), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
