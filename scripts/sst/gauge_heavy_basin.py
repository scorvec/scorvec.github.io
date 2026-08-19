#!/usr/bin/env python3
"""Heavy-rain verification at the scale the model actually consumes.

Point gauge vs 11 km IMERG pixel correlates only 0.22 daily over Andean
terrain - mostly scale mismatch, not sensor error, and useless as a bias
estimate. The inflow model never sees point rain: it sees an
energy-weighted BASIN MEAN. So the decision-relevant question is whether
IMERG flattens the heavy tail of the *basin mean*, where averaging many
gauges against many pixels removes most of the sampling confound.

Three quantities per region-day, gauges >= MIN_G:

  gauge_mean       mean of every IDEAM gauge inside the region
  imerg_at_gauges  raw IMERG sampled at THOSE SAME pixels - isolates
                   retrieval bias from spatial sampling
  imerg_area       the energy-weighted region mean the model is fed
  model            archived AIFS/IFS ensemble-mean forecast, lead 1

Stratified by gauge_mean. Raw IMERG only - the gauge-corrected and
gauge-blended fields contain the gauges and would be circular.

    python scripts/sst/gauge_heavy_basin.py [--min-gauges 5]
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

PRIV = Path.home() / "colombia_hydro"
GAUGES = PRIV / "raw" / "gauges"
ARCH = PRIV / "raw" / "fcst_rain"
OUT = PRIV / "out" / "gauge_heavy_basin.json"
BINS = [0.5, 2, 5, 10, 15, 20, 30, 1e9]
LBL = ["0.5-2", "2-5", "5-10", "10-15", "15-20", "20-30", ">30"]


def model_lead1():
    """{(region, valid_date): ens-mean mm/day} at lead 1 from each cycle."""
    out = {}
    for f in sorted(ARCH.glob("*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        be = d.get("basins_energy") or {}
        valid = d.get("valid") or []
        if len(valid) < 2:
            continue
        for rg, arr in be.items():
            a = np.asarray(arr, dtype=float)
            n = int(d.get("n_members") or 1)
            if a.size % n == 0 and a.size // n == len(valid):
                a = a.reshape(n, -1).mean(0)
            elif a.size != len(valid):
                continue
            out.setdefault((rg, valid[1]), []).append(float(a[1]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gauges", type=int, default=5)
    a = ap.parse_args(argv)

    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    regions = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
    W = region_weights_energy(lons, lats, regions)
    regions = sorted(W)
    masks = {r: (np.asarray(W[r]).reshape(len(lats), len(lons)) > 0)
             for r in regions}
    wts = {r: np.asarray(W[r]).reshape(len(lats), len(lons)) for r in regions}
    fc = model_lead1()
    print(f"{len(regions)} regions; {len(fc)} region-days of lead-1 forecast")

    rows = []
    for f in sorted(glob.glob(str(GAUGES / "*.json"))):
        day = Path(f).stem
        st = json.loads(Path(f).read_text())
        if not st:
            continue
        npy = IP.DAILY_CACHE / f"{day}.npy"
        if not npy.exists():
            continue
        grid = np.load(npy)
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        pts = []
        for v in st.values():
            mm = float(v["mm"])
            if not (0 <= mm <= 450):
                continue
            j = int(np.abs(lons - v["lo"]).argmin())
            i = int(np.abs(lats - v["la"]).argmin())
            pts.append((i, j, mm))
        if not pts:
            continue
        for r in regions:
            m = masks[r]
            sel = [(i, j, mm) for i, j, mm in pts if m[i, j]]
            if len(sel) < a.min_gauges:
                continue
            gm = float(np.mean([x[2] for x in sel]))
            sat_pt = float(np.mean([grid[x[0], x[1]] for x in sel]))
            sat_ar = float((grid * wts[r]).sum())
            rows.append(dict(day=iso, region=r, n=len(sel), gauge=gm,
                             imerg_at_gauges=sat_pt, imerg_area=sat_ar,
                             model=fc.get((r, iso), np.nan)))

    print(f"{len(rows)} region-days with >= {a.min_gauges} gauges\n")
    g = np.array([x["gauge"] for x in rows])
    sp = np.array([x["imerg_at_gauges"] for x in rows])
    sa = np.array([x["imerg_area"] for x in rows])
    md = np.array([x["model"] for x in rows])
    ng = np.array([x["n"] for x in rows])
    print(f"median gauges per region-day: {np.median(ng):.0f}")
    w = g > 0.1
    print(f"correlation vs gauge  -  IMERG@gauges {np.corrcoef(g[w],sp[w])[0,1]:.3f}"
          f"   IMERG area {np.corrcoef(g[w],sa[w])[0,1]:.3f}")
    mm = np.isfinite(md) & w
    if mm.sum() > 30:
        print(f"                          model(lead1) "
              f"{np.corrcoef(g[mm],md[mm])[0,1]:.3f}  (n={mm.sum()})")
    print()

    idx = np.digitize(g, BINS) - 1
    print(f"  {'gauge bin':>9} {'n':>6} {'gauge':>7} {'IMERG@g':>9} {'ratio':>6} "
          f"{'IMERGarea':>10} {'ratio':>6} {'model':>7} {'ratio':>6}")
    tab = []
    for b in range(len(LBL)):
        k = idx == b
        if k.sum() < 10:
            continue
        km = k & np.isfinite(md)
        mdl = md[km].mean() if km.sum() >= 10 else np.nan
        row = (LBL[b], int(k.sum()), g[k].mean(), sp[k].mean(),
               sp[k].mean()/g[k].mean(), sa[k].mean(),
               sa[k].mean()/g[k].mean(), mdl,
               (mdl/g[km].mean() if km.sum() >= 10 else np.nan))
        tab.append(row)
        ms = f"{row[7]:7.1f} {row[8]:6.2f}" if np.isfinite(row[7]) else "      -      -"
        print(f"  {row[0]:>9} {row[1]:>6} {row[2]:7.1f} {row[3]:9.1f} "
              f"{row[4]:6.2f} {row[5]:10.1f} {row[6]:6.2f} {ms}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(n=len(rows), table=tab), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
