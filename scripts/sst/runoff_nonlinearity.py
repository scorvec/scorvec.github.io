#!/usr/bin/env python3
"""Does a saturation-dependent runoff gain recover spike amplitude?

Both rainfall explanations for the spike deficit are now dead: the
satellite tail is not flattened (gauge verification, 2026-08-19) and
bias-correcting the input moves model skill by +0.0000 (stage B test).
The model still predicts only ~36% of observed spike amplitude, so the
deficit must sit in the RESPONSE, not the forcing.

The physical suspicion is that the linear reservoir is the wrong shape. A
linear kernel applies one fixed gain: 10 mm on a saturated catchment and
10 mm on a dry one produce the same predicted rise. Real catchments do
not work that way - runoff coefficient climbs steeply with antecedent
wetness, and again with rainfall intensity once infiltration capacity is
exceeded. A symmetric linear filter asked to fit an asymmetric,
threshold-like process will split the difference: under-predicting wet
spikes and over-predicting dry ones, which is exactly an amplitude
deficit.

Variants tested, all linear-in-parameters so the same blocked CV and
in-fold tau selection apply and nothing buys its own skill:

  base        current traced-catchment model
  wet         + rain x antecedent-wetness (slow kernel)
  flow        + rain x current flow state (y is itself a wetness proxy)
  stor        + rain x reservoir storage
  intensity   + max(rain - p90, 0), a piecewise-linear intensity break
  wet+int     both mechanisms together

Scored on spike amplitude and capture, not just r - a variant can raise
r while still flattening spikes, which is the failure mode that matters.

    python scripts/sst/runoff_nonlinearity.py [--basin ANTIOQUIA]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catchment_geometry_test as G                                # noqa: E402
import inflow_delta_model as M                                     # noqa: E402
import perfect_rain_backtest as PR                                 # noqa: E402
import delta_backtest_long as DB                                   # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "out" / "runoff_nonlinearity.json"
TAUS = (2, 4, 7, 12, 20, 30)


def zscore(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 1e-9 else 1.0)


def build(mode, D, tau):
    """mode = geometry+variant, e.g. 'traced|wet'."""
    geom, var = mode.split("|")
    y, pooled, roni, stor = D["y"], D["pooled"], D["roni"], D["stor"]
    lg, em = M.lagged, M.ema
    cols = [lg(y, 1) - M.baseline_series(y, 365)]
    if geom == "pooled":
        cols += [lg(pooled, 0), lg(em(pooled, tau), 0)]
    else:
        for nm, (a, e) in sorted(D[geom].items(), key=lambda x: -x[1][1]):
            cols += [lg(a, 0), lg(em(a, tau), 0)]
    slow = lg(em(pooled, 90), 0)
    cols += [slow, roni, stor]

    # wetness proxies, all strictly causal (lagged before use)
    r0 = lg(pooled, 0)
    W_wet = zscore(slow)                       # antecedent rain memory
    W_flow = zscore(lg(y, 1))                  # current flow state
    W_stor = zscore(stor)

    if var in ("wet", "wet+int"):
        cols += [r0 * W_wet, lg(em(pooled, tau), 0) * W_wet]
    if var == "flow":
        cols += [r0 * W_flow, lg(em(pooled, tau), 0) * W_flow]
    if var == "stor":
        cols += [r0 * W_stor, lg(em(pooled, tau), 0) * W_stor]
    if var in ("intensity", "wet+int"):
        thr = np.nanpercentile(pooled, 90)
        cols += [np.maximum(r0 - thr, 0.0)]
    return np.column_stack(cols)


def run(mode, D):
    y = D["y"]
    n = len(y)
    dy = np.full(n, np.nan)
    dy[1:] = y[1:] - y[:-1]
    pred = np.full(n, np.nan)
    nfeat = build(mode, D, 7).shape[1]

    for a, b in M.blocks(n, M.N_OUTER):
        te = np.zeros(n, bool); te[a:b] = True
        tr = ~te.copy()
        tr[max(0, a - M.EMBARGO):min(n, b + M.EMBARGO)] = False
        inner = np.where(tr)[0]
        cut = inner[int(0.75 * len(inner))]
        itr = tr & (np.arange(n) < cut)
        iva = tr & (np.arange(n) >= cut + M.EMBARGO)
        best = None
        for tau in TAUS:
            X = build(mode, D, tau)
            m1 = itr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            m2 = iva & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            if m1.sum() < 400 or m2.sum() < 100:
                continue
            A = np.column_stack([np.ones(m1.sum()), X[m1]])
            beta, *_ = np.linalg.lstsq(A, dy[m1], rcond=None)
            p = np.column_stack([np.ones(m2.sum()), X[m2]]) @ beta
            sc = np.corrcoef(p, dy[m2])[0, 1]
            if best is None or sc > best[0]:
                best = (sc, tau)
        tau = best[1] if best else 7
        X = build(mode, D, tau)
        mtr = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        mte = te & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        A = np.column_stack([np.ones(mtr.sum()), X[mtr]])
        beta, *_ = np.linalg.lstsq(A, dy[mtr], rcond=None)
        pred[mte] = np.column_stack([np.ones(mte.sum()), X[mte]]) @ beta

    ok = np.isfinite(pred) & np.isfinite(dy)
    o, p = dy[ok], pred[ok]
    rise = o > np.percentile(o, 75)
    spike = o > np.percentile(o, 90)
    pthr = np.percentile(p, 90)
    return dict(mode=mode, n_features=int(nfeat),
                r=float(np.corrcoef(p, o)[0, 1]),
                r_rise=float(np.corrcoef(p[rise], o[rise])[0, 1]),
                spike_hit=float((spike & (p > pthr)).sum() / max(spike.sum(), 1)),
                spike_amp=float(p[spike].mean() / o[spike].mean()),
                rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                pred=p, obs=o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="ANTIOQUIA")
    a = ap.parse_args(argv)

    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    zt = np.load(G.TRACED_NPZ, allow_pickle=True)
    common = np.intersect1d(dates, zt["dates"])
    _, i1, _ = np.intersect1d(dates, common, return_indices=True)
    traced, _ = G.load_catch(G.TRACED_NPZ, a.basin, common)
    D = dict(y=d["y"][a.basin][i1], pooled=d["rain"][a.basin][i1],
             roni=d["roni"][i1], stor=d["stor"][a.basin][i1], traced=traced)

    print(f"{a.basin}: {len(i1)} days, {len(traced)} traced catchments\n")
    hdr = (f"{'variant':12} {'cols':>5} {'r(dy)':>7} {'r_rise':>8} "
           f"{'spike_hit':>10} {'amp':>6} {'rmse':>8}")
    print(hdr); print("-" * len(hdr))
    res = {}
    for var in ("base", "wet", "flow", "stor", "intensity", "wet+int"):
        r = run(f"traced|{var}", D)
        res[var] = {k: v for k, v in r.items() if k not in ("pred", "obs")}
        print(f"{var:12} {r['n_features']:5} {r['r']:7.3f} {r['r_rise']:8.3f} "
              f"{r['spike_hit']:10.3f} {r['spike_amp']:6.2f} {r['rmse']:8.3f}")
    b = res["base"]
    print(f"\nvs base:")
    for k, v in res.items():
        if k == "base":
            continue
        print(f"  {k:10} r {v['r']-b['r']:+.4f}  rise {v['r_rise']-b['r_rise']:+.4f}"
              f"  capture {v['spike_hit']-b['spike_hit']:+.4f}"
              f"  amplitude {v['spike_amp']-b['spike_amp']:+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
