#!/usr/bin/env python3
"""Frequency Bias Index and conditional bias as a function of PREDICTED rain.

Conditioning matters and the two directions answer different questions:

  by TRUTH      characterises the instrument. Regression to the mean
                dominates it (see gauge_split_half.py), so it is the wrong
                axis to calibrate on.
  by PREDICTION what you actually have operationally: you observe an
                IMERG or model value and need to know what it implies.
                This is the axis a correction must be built on.

Reported here, all on the energy-weighted basin mean the inflow model
actually consumes, referenced to the IDEAM gauge network mean:

  FBI(t)   = P(pred >= t) / P(obs >= t). > 1 over-forecasts that
             threshold's frequency, < 1 under-forecasts it. Independent
             of pairing, so it is not degraded by timing error.
  bias(p)  = mean(obs | pred in bin) / mean(pred | bin) - the
             multiplicative factor a correction would apply.

Raw IMERG only; the gauge-corrected and blended fields contain the gauges.

    python scripts/sst/rain_bias_curve.py [--min-gauges 8]
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
OUT = PRIV / "out" / "rain_bias_curve.json"
REGIONS = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
THRESH = [0.5, 1, 2, 5, 10, 15, 20, 30, 40]
PBINS = [0.5, 2, 5, 10, 15, 20, 30, 1e9]
PLBL = ["0.5-2", "2-5", "5-10", "10-15", "15-20", "20-30", ">30"]


def model_index(leads=(1, 2, 3)):
    out = {}
    for f in sorted(ARCH.glob("*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        be = d.get("basins_energy") or {}
        valid = d.get("valid") or []
        n = int(d.get("n_members") or 1)
        mdl = d.get("model")
        for rg, arr in be.items():
            a = np.asarray(arr, dtype=float)
            if a.size % n == 0 and a.size // n == len(valid):
                a = a.reshape(n, -1).mean(0)
            elif a.size != len(valid):
                continue
            for L in leads:
                if L < len(valid):
                    out.setdefault((mdl, L, rg, valid[L]), []).append(float(a[L]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def collect(min_gauges):
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    W = region_weights_energy(lons, lats, REGIONS)
    masks = {r: (np.asarray(W[r]).reshape(len(lats), len(lons)) > 0) for r in W}
    wts = {r: np.asarray(W[r]).reshape(len(lats), len(lons)) for r in W}
    fc = model_index()
    rows = []
    for f in sorted(glob.glob(str(GAUGES / "*.json"))):
        day = Path(f).stem
        st = json.loads(Path(f).read_text())
        npy = IP.DAILY_CACHE / f"{day}.npy"
        if not st or not npy.exists():
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
        for r in masks:
            sel = [(i, j, mm) for i, j, mm in pts if masks[r][i, j]]
            if len(sel) < min_gauges:
                continue
            rows.append(dict(
                region=r, day=iso, ng=len(sel),
                gauge=float(np.mean([x[2] for x in sel])),
                imerg=float((grid * wts[r]).sum()),
                aifs=fc.get(("aifs", 1, r, iso), np.nan),
                ifs=fc.get(("ifs", 1, r, iso), np.nan)))
    return rows


def fbi(pred, obs, label):
    print(f"\n  {label}: FBI = P(pred>=t) / P(obs>=t)   (n={len(pred)})")
    print(f"    {'thresh':>8} {'P(obs)':>8} {'P(pred)':>9} {'FBI':>7}")
    out = []
    for t in THRESH:
        po = float((obs >= t).mean())
        pp = float((pred >= t).mean())
        if po * len(obs) < 5:
            continue
        out.append([t, po, pp, pp / po if po > 0 else np.nan])
        print(f"    {t:8.1f} {po:8.3f} {pp:9.3f} {pp/po:7.2f}")
    return out


def cond(pred, obs, label):
    print(f"\n  {label}: bias by PREDICTED amount")
    print(f"    {'pred bin':>9} {'n':>6} {'mean pred':>10} {'mean obs':>9} "
          f"{'obs/pred':>9} {'correction':>11}")
    idx = np.digitize(pred, PBINS) - 1
    out = []
    for b in range(len(PLBL)):
        k = idx == b
        if k.sum() < 10:
            continue
        mp, mo = float(pred[k].mean()), float(obs[k].mean())
        out.append([PLBL[b], int(k.sum()), mp, mo, mo / mp])
        print(f"    {PLBL[b]:>9} {k.sum():6} {mp:10.1f} {mo:9.1f} "
              f"{mo/mp:9.2f} {'x%.2f' % (mo/mp):>11}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gauges", type=int, default=8)
    a = ap.parse_args(argv)
    rows = collect(a.min_gauges)
    g = np.array([x["gauge"] for x in rows])
    s = np.array([x["imerg"] for x in rows])
    print(f"{len(rows)} region-days, >= {a.min_gauges} gauges "
          f"(median {np.median([x['ng'] for x in rows]):.0f})")

    res = {"imerg": {"fbi": fbi(s, g, "IMERG (raw, basin mean)"),
                     "cond": cond(s, g, "IMERG (raw, basin mean)")}}

    for mdl in ("aifs", "ifs"):
        v = np.array([x[mdl] for x in rows])
        m = np.isfinite(v)
        if m.sum() < 30:
            print(f"\n  {mdl.upper()}: only {m.sum()} gauge-verifiable "
                  f"region-days - NOT REPORTED (needs accumulation)")
            continue
        res[mdl] = {"fbi": fbi(v[m], g[m], f"{mdl.upper()} lead-1"),
                    "cond": cond(v[m], g[m], f"{mdl.upper()} lead-1")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
