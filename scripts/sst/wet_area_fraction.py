#!/usr/bin/env python3
"""Rainfall EXTENT as a predictor, from the IMERG grid we already hold.

The hourly-gauge work found that within-day concentration predicts a
SMALLER inflow rise for the same daily total (partial r = -0.162), and
that a spatial measure - the fraction of gauges reporting rain - predicts
a LARGER one (+0.119). The two correlate only -0.315 and are partly
independent, so extent is its own mechanism, not a restatement of timing.

Extent is the more useful half. Timing needs a forecast of convective
structure that NWP does not have at day 3+ (realistic model gain there:
+0.003). Extent is measurable straight off the IMERG grid across the full
2000-2026 record, and models resolve rainfall AREA far better than
convective timing, so it has an operational path.

A basin mean of 8 mm can be 8 mm everywhere or 40 mm over a fifth of the
catchment. Those route very differently, and the mean cannot tell them
apart. This computes, per region-day, the energy-weighted fraction of the
catchment exceeding a set of thresholds, then tests whether that adds
skill to the delta model under the usual blocked CV.

    python scripts/sst/wet_area_fraction.py --build     # cache the fractions
    python scripts/sst/wet_area_fraction.py --test      # blocked-CV test

Cache: ~/colombia_hydro/raw/wet_area.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import (region_weights_energy,               # noqa: E402
                               gauge_correction, gauge_blend_field)

PRIV = Path.home() / "colombia_hydro"
CACHE = PRIV / "raw" / "wet_area.npz"
REGIONS = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
THRESH = [1.0, 5.0, 10.0, 20.0]


def build():
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    F = gauge_correction(lons, lats)
    W = region_weights_energy(lons, lats, REGIONS)
    # normalise each region's weights so the "fraction" is a true weighted
    # share of the generating catchment, not an unnormalised sum
    Wn = {}
    for r in REGIONS:
        w = np.asarray(W[r], float).ravel()
        Wn[r] = w / max(w.sum(), 1e-12)
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    print(f"{len(files)} daily grids, {len(REGIONS)} regions, "
          f"{len(THRESH)} thresholds", flush=True)
    dates, out = [], []
    for i, f in enumerate(files):
        g = gauge_blend_field(np.load(f) * F, f.stem, lons, lats).ravel()
        row = []
        for r in REGIONS:
            w = Wn[r]
            for t in THRESH:
                row.append(float(w[g >= t].sum()))
        dates.append(f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}")
        out.append(row)
        if (i + 1) % 2000 == 0:
            print(f"   {i+1}/{len(files)}", flush=True)
    cols = [f"{r}|{t}" for r in REGIONS for t in THRESH]
    np.savez_compressed(CACHE, dates=np.array(dates), frac=np.array(out),
                        cols=np.array(cols))
    print(f"wrote {CACHE}  frac{np.array(out).shape}")


def test(basin):
    import catchment_geometry_test as G
    import inflow_delta_model as M
    import perfect_rain_backtest as PR
    import delta_backtest_long as DB
    from scipy import stats

    z = np.load(CACHE, allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    zt = np.load(G.TRACED_NPZ, allow_pickle=True)
    common = np.intersect1d(np.intersect1d(dates, zt["dates"]), z["dates"])
    _, i1, _ = np.intersect1d(dates, common, return_indices=True)
    _, iz, _ = np.intersect1d(z["dates"], common, return_indices=True)
    traced, _ = G.load_catch(G.TRACED_NPZ, basin, common)
    D = dict(y=d["y"][basin][i1], pooled=d["rain"][basin][i1],
             roni=d["roni"][i1], stor=d["stor"][basin][i1], traced=traced)
    fr = {t: z["frac"][iz, cols.index(f"{basin}|{t}")] for t in THRESH}
    print(f"{basin}: {len(i1)} days")
    for t in THRESH:
        print(f"   wet-area fraction >= {t:4.0f} mm: mean {fr[t].mean():.3f} "
              f"sd {fr[t].std():.3f}")

    def build_X(variant, tau):
        X = G.build("traced", D, tau)
        if variant == "base":
            return X
        add = []
        for t in ([1.0, 10.0] if variant == "extent" else [float(variant)]):
            a = fr[t]
            add += [M.lagged(a, 0), M.lagged(M.ema(a, tau), 0)]
        return np.column_stack([X] + add)

    def run(variant):
        y = D["y"]; n = len(y)
        dy = np.full(n, np.nan); dy[1:] = y[1:] - y[:-1]
        pred = np.full(n, np.nan)
        for a, b in M.blocks(n, M.N_OUTER):
            te = np.zeros(n, bool); te[a:b] = True
            tr = ~te.copy()
            tr[max(0, a - M.EMBARGO):min(n, b + M.EMBARGO)] = False
            best = None
            inner = np.where(tr)[0]; cut = inner[int(0.75 * len(inner))]
            itr = tr & (np.arange(n) < cut)
            iva = tr & (np.arange(n) >= cut + M.EMBARGO)
            for tau in (2, 4, 7, 12, 20, 30):
                X = build_X(variant, tau)
                m1 = itr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
                m2 = iva & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
                if m1.sum() < 400 or m2.sum() < 100:
                    continue
                A = np.column_stack([np.ones(m1.sum()), X[m1]])
                be, *_ = np.linalg.lstsq(A, dy[m1], rcond=None)
                p = np.column_stack([np.ones(m2.sum()), X[m2]]) @ be
                s = np.corrcoef(p, dy[m2])[0, 1]
                if best is None or s > best[0]:
                    best = (s, tau)
            tau = best[1] if best else 7
            X = build_X(variant, tau)
            mtr = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            mte = te & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            A = np.column_stack([np.ones(mtr.sum()), X[mtr]])
            be, *_ = np.linalg.lstsq(A, dy[mtr], rcond=None)
            pred[mte] = np.column_stack([np.ones(mte.sum()), X[mte]]) @ be
        ok = np.isfinite(pred) & np.isfinite(dy)
        o, p = dy[ok], pred[ok]
        rise = o > np.percentile(o, 75); sp = o > np.percentile(o, 90)
        return dict(r=float(np.corrcoef(p, o)[0, 1]),
                    rr=float(np.corrcoef(p[rise], o[rise])[0, 1]),
                    hit=float((sp & (p > np.percentile(p, 90))).sum() / sp.sum()),
                    amp=float(p[sp].mean() / o[sp].mean()),
                    rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                    nf=build_X("base" if False else "base", 7).shape[1])

    print(f"\n  {'variant':16} {'r(dy)':>7} {'r_rise':>8} {'capture':>8} "
          f"{'amp':>6} {'rmse':>8}")
    res = {}
    for v in ("base", "1.0", "10.0", "extent"):
        r = run(v)
        res[v] = r
        lab = {"base": "base (traced)", "1.0": "+ area>=1mm",
               "10.0": "+ area>=10mm", "extent": "+ both"}[v]
        print(f"  {lab:16} {r['r']:7.3f} {r['rr']:8.3f} {r['hit']:8.3f} "
              f"{r['amp']:6.2f} {r['rmse']:8.3f}")
    b = res["base"]
    print()
    for v in ("1.0", "10.0", "extent"):
        x = res[v]
        print(f"  vs base [{v:6}]  r {x['r']-b['r']:+.4f}  "
              f"rise {x['rr']-b['rr']:+.4f}  capture {x['hit']-b['hit']:+.4f}  "
              f"amp {x['amp']-b['amp']:+.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--basin", default="ANTIOQUIA")
    a = ap.parse_args(argv)
    if a.build or not CACHE.exists():
        build()
    if a.test:
        test(a.basin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
