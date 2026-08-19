#!/usr/bin/env python3
"""Does true upstream delineation beat the SZH unions? Blocked-CV decision.

ANTIOQUIA is 48.8% of national inflow energy and kept its headroom after
the catchment split helped CENTRO. The suspected cause was geometry, not
method: the shipped xm_river_catchments.geojson gives ANTIOQUIA's 16
rivers only **5 distinct shapes**, because IDEAM subzona 2701 alone is
shared by nine rivers carrying 43% of the region's energy. Rain that is
averaged over one polygon cannot be decomposed by any downstream model.

delineate_catchments.py replaces those unions with HydroBASINS lev-12
upstream traces from the dam outlets - 11 distinct shapes, each the
INCREMENTAL (local) area so a cascade's nested reservoirs do not restate
each other's rain.

This scores three geometries on the same target, folds and embargo:

    pooled   one ANTIOQUIA energy-weighted basin mean      (current v3)
    szh      the 5 distinct IDEAM subzona unions
    traced   the 11 distinct HydroBASINS incremental traces

tau is chosen INSIDE each training fold, never on the test block, so the
extra flexibility of the wider feature sets cannot buy itself skill.
Headline score is out-of-sample r on dy; the number that matters for the
stated goal is r on RISE days and capture of the top-decile spikes.

    python scripts/sst/catchment_geometry_test.py [--basin ANTIOQUIA]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import delta_backtest_long as DB                                    # noqa: E402
import inflow_delta_model as M                                      # noqa: E402

PRIV = Path.home() / "colombia_hydro"
SZH_NPZ = PRIV / "raw" / "catchment_rain.npz"
TRACED_NPZ = PRIV / "raw" / "catchment_rain_traced.npz"
OUT_JSON = PRIV / "out" / "catchment_geometry_test.json"
TAUS = (2, 4, 7, 12, 20, 30)
MIN_E = 1.0                  # GWh/day floor for a catchment to earn a column


def load_catch(npz, basin, dates):
    """{river: anomaly series aligned to `dates`} collapsing identical masks.

    `dates` must already be the common axis across every cache being
    compared - the SZH and traced caches were built on different days and
    cover different spans, and scoring them on different records would
    make the comparison meaningless.
    """
    z = np.load(npz, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    R, C, meta = z["rain"], z["clim"], z["meta"].item()
    _, i1, i2 = np.intersect1d(dates, z["dates"], return_indices=True)
    assert len(i1) == len(dates), f"{npz.name} misses {len(dates)-len(i1)} days"
    groups = {}
    for i, k in enumerate(keys):
        if meta[k]["region"] != basin:
            continue
        groups.setdefault(R[:, i].tobytes(), []).append(i)
    out = {}
    for g in groups.values():
        e = sum(meta[keys[j]]["energy_gwh"] for j in g)
        if e < MIN_E:
            continue
        name = "+".join(meta[keys[j]]["river"] for j in g)
        out[name] = ((R[:, g[0]] - C[:, g[0]])[i2], e)
    return out, i1


def build(mode, D, tau):
    y, pooled, roni, stor = D["y"], D["pooled"], D["roni"], D["stor"]
    lg, em = M.lagged, M.ema
    cols = [lg(y, 1) - M.baseline_series(y, 365)]
    if mode == "pooled":
        cols += [lg(pooled, 0), lg(em(pooled, tau), 0)]
    else:
        for nm, (a, e) in sorted(D[mode].items(), key=lambda x: -x[1][1]):
            cols += [lg(a, 0), lg(em(a, tau), 0)]
    cols += [lg(em(pooled, 90), 0), roni, stor]
    return np.column_stack(cols)


def run(mode, D):
    y = D["y"]
    n = len(y)
    dy = np.full(n, np.nan)
    dy[1:] = y[1:] - y[:-1]
    pred = np.full(n, np.nan)
    nfeat = build(mode, D, 7).shape[1]

    for a, b in M.blocks(n, M.N_OUTER):
        te = np.zeros(n, bool)
        te[a:b] = True
        tr = ~te.copy()
        tr[max(0, a - M.EMBARGO):min(n, b + M.EMBARGO)] = False

        # tau chosen on an inner split of the TRAINING block only
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
            s = np.corrcoef(p, dy[m2])[0, 1]
            if best is None or s > best[0]:
                best = (s, tau)
        tau = best[1] if best else 7

        X = build(mode, D, tau)
        mtr = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        mte = te & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        A = np.column_stack([np.ones(mtr.sum()), X[mtr]])
        beta, *_ = np.linalg.lstsq(A, dy[mtr], rcond=None)
        pred[mte] = np.column_stack([np.ones(mte.sum()), X[mte]]) @ beta

    ok = np.isfinite(pred) & np.isfinite(dy)
    o, p = dy[ok], pred[ok]
    thr = np.percentile(o, 90)
    rise = o > np.percentile(o, 75)
    spike = o > thr
    # spike capture: of the true top-decile rises, how many does the model
    # also place in ITS top decile
    pthr = np.percentile(p, 90)
    hit = float((spike & (p > pthr)).sum() / max(spike.sum(), 1))
    return dict(
        mode=mode, n=int(ok.sum()), n_features=int(nfeat),
        r=float(np.corrcoef(p, o)[0, 1]),
        r_rise=float(np.corrcoef(p[rise], o[rise])[0, 1]),
        r_spike=float(np.corrcoef(p[spike], o[spike])[0, 1]),
        spike_hit=hit,
        spike_amp=float(p[spike].mean() / o[spike].mean()),
        rmse=float(np.sqrt(np.mean((p - o) ** 2))),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="ANTIOQUIA")
    a = ap.parse_args(argv)

    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    # common axis across the model record and BOTH catchment caches
    zs, zt = np.load(SZH_NPZ, allow_pickle=True), np.load(TRACED_NPZ,
                                                          allow_pickle=True)
    common = np.intersect1d(np.intersect1d(dates, zs["dates"]), zt["dates"])
    _, i1, _ = np.intersect1d(dates, common, return_indices=True)
    szh, _ = load_catch(SZH_NPZ, a.basin, common)
    traced, _ = load_catch(TRACED_NPZ, a.basin, common)

    D = dict(y=d["y"][a.basin][i1], pooled=d["rain"][a.basin][i1],
             roni=d["roni"][i1], stor=d["stor"][a.basin][i1],
             szh=szh, traced=traced)

    print(f"{a.basin}: {len(i1)} days ({common[0]}..{common[-1]})  "
          f"szh={len(szh)} distinct  traced={len(traced)} distinct "
          f"(>= {MIN_E} GWh/d)")
    print(f"  szh    : {', '.join(sorted(szh, key=lambda k: -szh[k][1]))}")
    print(f"  traced : {', '.join(sorted(traced, key=lambda k: -traced[k][1]))}")
    print()
    hdr = (f"{'geometry':10} {'cols':>5} {'r(dy)':>8} {'r_rise':>8} "
           f"{'r_spike':>8} {'spike_hit':>10} {'amp':>7} {'rmse':>8}")
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for mode in ("pooled", "szh", "traced"):
        r = run(mode, D)
        res[mode] = r
        print(f"{mode:10} {r['n_features']:5} {r['r']:8.3f} {r['r_rise']:8.3f} "
              f"{r['r_spike']:8.3f} {r['spike_hit']:10.3f} "
              f"{r['spike_amp']:7.2f} {r['rmse']:8.3f}")

    b, w = res["traced"], res["szh"]
    print(f"\ntraced vs szh:  r {b['r']-w['r']:+.3f}   "
          f"rise {b['r_rise']-w['r_rise']:+.3f}   "
          f"spike-capture {b['spike_hit']-w['spike_hit']:+.3f}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
