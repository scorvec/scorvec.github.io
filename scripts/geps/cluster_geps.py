#!/usr/bin/env python3
"""Ensemble scenario clustering of the 21 members' weekly 500 hPa patterns.

Not regime classification: no fixed centroids. For each sector and week the
members' weekly-mean 500 hPa anomalies (de-drifted, re-based, 2.5 deg) are
split into K clusters by k-means (cos-lat weighted, 30 restarts, best
inertia), and each cluster is shown as its mean map with its member count.
That answers the question the ensemble mean cannot: when the mean is a blend
of two circulations, which scenarios are actually in the ensemble and how
many members back each.

The number of clusters is CHOSEN, not fixed. For k = 2, 3, 4 the between-
cluster share of member spread is compared with a null: the members'
deviations rotated in member space by random orthogonal matrices (same
spatial covariance, no grouping) and re-clustered N_NULL times, equal
k-means restarts on both sides. The largest k whose share beats the 90th
percentile of its null, the one with the largest margin, is shown; if
none does, the ensemble is "one scenario with spread" and the k = 2 split
is drawn faded for reference.
This is the Ferranti & Corti (2011) recipe ECMWF uses to pick its cluster
count, and with 21 members its verdict is usually one or two scenarios.
Calibration on this ensemble (2026-09-03, NA week 3): Gaussian ensembles
pass 5-7% of the time; a two-scenario ensemble with groups 3 sd apart along
EOF1 is caught 60% of the time at k = 2, 4 sd apart 95% — but only 25% and
50% with a forced k = 3, which is why k = 3 alone was abandoned. Groups
closer than ~2 sd cannot be told from a continuum with 21 members. Each cluster mean is also
labelled with its nearest weather regime (build_regimes.py) and the pattern
correlation, which ties the two products together.

    python cluster_geps.py --cycle 20260903
    -> figs/, assets/geps/  geps_clusters_{ea,na}.webp, clusters.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.vq import kmeans2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths                                                     # noqa: E402
sys.path.insert(0, str(paths.REPO / "scripts" / "regimes"))
# Regime centroids, names, sectors and colours from the shared regime code (scripts/regimes/regime_core.py), which
# replaced regimes_geps.py's own copies on 2026-09-28 - the import of those copies is what broke this script then.
import regime_core as RC                                         # noqa: E402
import telecon_geps as tg                                        # noqa: E402

SECTORS, KREG, season_of = RC.SECTORS, RC.K, RC.season_of
TC = paths.TC_STATE
FIGS = paths.FIGS
SITE = paths.SITE_ASSETS
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
KS = (2, 3, 4)                   # cluster counts tested; the largest significant one is shown
N_NULL = 100                     # random rotations for the null
NULL_PCT = 95                    # three k's are tested per week: 95th keeps the family-wise
                                 # false-positive rate near 14% rather than 27% at the 90th
RESTARTS = 12                    # the same for the real members and every null draw


def _kmeans_ratio(Z, k, rng, restarts):
    best = None
    for _ in range(restarts):
        cen, lab = kmeans2(Z, k, minit="++", seed=rng.integers(1 << 30), iter=60)
        inertia = ((Z - cen[lab]) ** 2).sum()
        if best is None or inertia < best[1]:
            best = (lab, inertia)
    lab, inertia = best
    total = ((Z - Z.mean(0)) ** 2).sum()
    return lab, (1.0 - inertia / total if total > 0 else 0.0)


def cluster_week(X, w):
    """X: (members, lat, lon). Tests k in KS against the rotation null and
    returns (k_shown, labels ordered by cluster size, significant, tests)
    where tests = {k: (ratio, null 90th pct)}."""
    n = X.shape[0]
    Z = (X * w).reshape(n, -1)
    D = Z - Z.mean(0)
    rng = np.random.default_rng(11)
    tests, labs = {}, {}
    for k in KS:
        lab, ratio = _kmeans_ratio(Z, k, rng, RESTARTS)
        null = []
        for _ in range(N_NULL):
            Q, _r = np.linalg.qr(rng.standard_normal((n, n)))
            null.append(_kmeans_ratio(Q @ D, k, rng, RESTARTS)[1])
        tests[k] = (float(ratio), float(np.percentile(null, NULL_PCT)))
        labs[k] = lab
    # among the k's that beat their null, the one with the largest margin
    # (ECMWF picks by significance, not by size); none -> one scenario
    sig = [k for k in KS if tests[k][0] > tests[k][1]]
    k_show, significant = ((max(sig, key=lambda k: tests[k][0] - tests[k][1]), True) if sig
                           else (KS[0], False))
    lab = labs[k_show]
    sizes = np.bincount(lab, minlength=k_show)
    order = np.argsort(sizes)[::-1]
    remap = {old: new for new, old in enumerate(order)}
    return k_show, np.array([remap[x] for x in lab]), significant, tests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    a = ap.parse_args()
    base = pd.Timestamp(a.cycle)
    season = season_of(base.month)
    rg = RC.Ref()
    z = tg.members_anom("zg500", a.cycle, base)                    # (L, number, lat, lon)
    if z is None:
        print("no zg500 members"); return 1
    day = z.L.values.astype(int)
    js = {"cycle": a.cycle, "season": season, "k_tested": list(KS), "null_pct": NULL_PCT, "n_null": N_NULL,
          "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "sectors": {}}
    for sk, spec in SECTORS.items():
        secv, lat, lon = RC.sector_cut(z.values, spec)                # (L, number, lat, lon), lon -180..180
        w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        cent, _, _ = rg.centroids(sk, season)
        names = [rg.name(sk, season, k) for k in range(KREG)]
        fam_col = rg.family_colors(sk)
        colors = [fam_col[rg.family_index(sk, season, k)] for k in range(KREG)]
        rows = []
        for wi, (w0, w1) in enumerate(WEEKS):
            sel = (day >= w0) & (day <= w1)
            X = secv[sel].mean(0)                                  # (number, lat, lon)
            kc, lab, significant, tests = cluster_week(X, w)
            ratio, thr = tests[kc]
            clusters = []
            for c in range(kc):
                m = X[lab == c].mean(0)
                # nearest regime by pattern correlation, for the label
                mw = (m * w).ravel(); mw = mw - mw.mean()
                cors = []
                for k in range(KREG):
                    cw = (cent[k] * w).ravel(); cw = cw - cw.mean()
                    cors.append(float(mw @ cw / (np.linalg.norm(mw) * np.linalg.norm(cw) + 1e-9)))
                kbest = int(np.argmax(cors))
                clusters.append(dict(n=int((lab == c).sum()), mean=m, regime=kbest, regime_name=names[kbest],
                                     r=cors[kbest], members=np.where(lab == c)[0].tolist()))
            rows.append(dict(week=wi + 1, w0=w0, w1=w1, k=kc, ratio=ratio, thr=thr, noise=not significant,
                             tests=tests, ens=X.mean(0), clusters=clusters, lab=lab))
            print(f"  {sk} week {wi+1}: k={kc} sizes " + "/".join(str(c["n"]) for c in clusters)
                  + " · " + " ".join(f"k{k}:{t[0]*100:.0f}%/{t[1]*100:.0f}%" for k, t in tests.items())
                  + ("  one scenario" if not significant else f"  {kc} SCENARIOS")
                  + "  " + ", ".join(f"{c['regime_name']} r{c['r']:+.2f}" for c in clusters), flush=True)
        # figure: rows = weeks, cols = ensemble mean + KC clusters
        proj = ccrs.LambertConformal(central_longitude=np.mean(spec["lon"]), standard_parallels=(35, 55))
        ncol = max(r["k"] for r in rows) + 1            # no empty columns
        fig = plt.figure(figsize=(3.9 * ncol, 2.95 * len(rows) + 1.3))
        lim = float(np.nanpercentile(np.abs(np.stack([c["mean"] for r in rows for c in r["clusters"]])), 99))
        levels = np.linspace(-lim, lim, 21)
        for ri, r in enumerate(rows):
            panels = [("ensemble mean", r["ens"], None)] + [
                (f"scenario {c+1} · {cl['n']} members ({cl['n']/len(r['lab'])*100:.0f}%)", cl["mean"], cl)
                for c, cl in enumerate(r["clusters"])]
            for ci, (ttl, fld, cl) in enumerate(panels):
                rh = 0.89 / len(rows)
                ax = fig.add_axes([0.01 + ci * (0.98 / ncol), 0.965 - (ri + 1) * rh + 0.012,
                                   0.98 / ncol - 0.012, rh - 0.070], projection=proj)
                ax.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
                m = ax.contourf(lon, lat, fld, levels=levels, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
                ax.contour(lon, lat, fld, levels=levels[::2], colors="k", linewidths=0.3, transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="#333")
                if cl is not None:
                    col = colors[cl["regime"]]
                    for sp in ax.spines.values():
                        sp.set_edgecolor(col); sp.set_linewidth(1.8 if cl["r"] >= 0.5 else 0.8)
                    sub = f"nearest regime: {cl['regime_name']} (r {cl['r']:+.2f})"
                else:
                    d0 = base + pd.Timedelta(days=r["w0"] - 1); d1 = base + pd.Timedelta(days=r["w1"] - 1)
                    ttl = f"week {r['week']} · {d0:%b %d} – {d1:%b %d} · ensemble mean"
                    sub = (f"{r['k']} distinct scenarios: k={r['k']} explains {r['ratio']*100:.0f}% of spread, "
                           f"null {r['thr']*100:.0f}%" if not r["noise"] else
                           f"one scenario with spread (no k in {KS[0]}–{KS[-1]} beats its null)")
                ax.set_title(f"{ttl}\n{sub}", fontsize=7.6, fontweight="bold" if cl is None else "normal",
                             color=("#8a1d1d" if (cl is None and r["noise"]) else
                                    "#1f1f1f" if (cl is None or cl["r"] >= 0.5) else "#5a5650"))
                if r["noise"] and cl is not None:
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", alpha=0.45,
                                               zorder=5, lw=0))
                    ax.text(0.5, 0.5, "not distinct", transform=ax.transAxes, ha="center", va="center", fontsize=13,
                            color="#5a5650", alpha=0.8, fontweight="bold", zorder=6)
        cax = fig.add_axes([0.32, 0.040, 0.36, 0.011])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal"); cb.set_label("500 hPa height anomaly (m)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        fig.suptitle(f"{spec['label']} — 500 hPa scenarios: k-means clusters of the 21 members' weekly patterns · "
                     f"GEPS init {base:%Y-%m-%d}", fontsize=12, fontweight="bold", y=0.995, va="top")
        fig.text(0.5, 0.004, f"k = {KS[0]}–{KS[-1]} tested per week against a null of {N_NULL} random rotations of the "
                 f"members (same covariance, no grouping), the largest k beating its {NULL_PCT}th percentile shown; "
                 f"groups closer than ~2 sd cannot be told from a continuum with 21 members; frame colour and "
                 f"label = nearest weather regime", ha="center", va="bottom", fontsize=7.4, color="#5a5650")
        for d in (FIGS, SITE):
            fig.savefig(d / f"geps_clusters_{sk}.webp", dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        print(f"  geps_clusters_{sk}.webp")
        js["sectors"][sk] = {"label": spec["label"], "weeks": [
            dict(week=r["week"], w0=r["w0"], w1=r["w1"], k=r["k"], distinct=not r["noise"],
                 tests={str(k): [round(t[0], 3), round(t[1], 3)] for k, t in r["tests"].items()},
                 clusters=[dict(n=c["n"], regime=c["regime_name"], r=round(c["r"], 2), members=c["members"])
                           for c in r["clusters"]]) for r in rows]}
    for d in (TC, SITE):
        d.mkdir(parents=True, exist_ok=True)
        (d / "clusters.json").write_text(json.dumps(js))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
