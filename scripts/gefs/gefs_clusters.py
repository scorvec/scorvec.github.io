#!/usr/bin/env python3
"""Ensemble scenario clustering of the GEFS members' weekly 500 hPa patterns - the GEPS page's cluster_geps.py ported to
the GEFS extended page (2026-09-26).

Not regime classification: no fixed centroids. For each sector and week the members' weekly-mean 500 hPa anomalies
(the daily 2.5 deg member fields de-drifted against the lead-matched GEFSv12 reforecast climatology and re-based to
1991-2020, gefs_patterns.member_anoms - the same field the GEPS page clusters; weekly = days 7k-6 .. 7k) are split into
k clusters by k-means (sqrt-cos-lat weighted, best of 12 restarts), each shown as its mean map with its member count.

The number of clusters is CHOSEN, not fixed (Ferranti & Corti 2011, ECMWF's recipe): for k = 2, 3, 4 the between-
cluster share of member spread is compared with a null - the members' deviations rotated in member space by random
orthogonal matrices (same spatial covariance, no grouping), re-clustered 100 times with the same restarts. Among the
k's that beat the 95th percentile of their null, the one with the largest margin is shown; if none does, the ensemble
is "one scenario with spread" and the k = 2 split is drawn faded. The 95th keeps the family-wise false-positive rate of
three tests near 14%. What the test can resolve with this ensemble's size is measured by build_gefs_patterns_hindcast.py
--cluster-power (synthetic ensembles of the GEFS member count drawn from observed weekly 500 hPa anomalies) and quoted
in the footnote from the calibration file. Each cluster mean is labelled with its nearest weather regime.

    python gefs_clusters.py --npz /tmp/gefs_work/gefs2_20260925.npz --date 20260925 --site . --cache /tmp/gefs_work
    -> assets/gefs/gefs_clusters_{ea,na}.webp, assets/gefs/data/gefs_clusters.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import textwrap
from pathlib import Path

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

warnings.filterwarnings("ignore", message="One of the clusters is empty")     # that restart just loses on inertia

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_patterns as GP                                           # noqa: E402

KS = (2, 3, 4)
N_NULL = 100
NULL_PCT = 95
RESTARTS = 12
COLORS = ["#b4453c", "#2c5fa8", "#3f9b6a", "#c8781e"]          # regimes 1-4 (GEPS)


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


def cluster_week(X, w, seed=11):
    """X (members, lat, lon) -> (clusters shown, labels ordered by cluster size, significant,
    {k: (ratio, null pct), "shown": the k tested})."""
    n = X.shape[0]
    Z = (X * w).reshape(n, -1)
    D = Z - Z.mean(0)
    rng = np.random.default_rng(seed)
    tests, labs = {}, {}
    for k in KS:
        lab, ratio = _kmeans_ratio(Z, k, rng, RESTARTS)
        null = []
        for _ in range(N_NULL):
            Q, _r = np.linalg.qr(rng.standard_normal((n, n)))
            null.append(_kmeans_ratio(Q @ D, k, rng, RESTARTS)[1])
        tests[k] = (float(ratio), float(np.percentile(null, NULL_PCT)))
        labs[k] = lab
    sig = [k for k in KS if tests[k][0] > tests[k][1]]
    k_show, significant = ((max(sig, key=lambda k: tests[k][0] - tests[k][1]), True) if sig else (KS[0], False))
    lab = labs[k_show]
    sizes = np.bincount(lab, minlength=k_show)
    order = [c for c in np.argsort(sizes)[::-1] if sizes[c] > 0]     # an empty cluster (rare) is dropped
    remap = {old: new for new, old in enumerate(order)}
    return len(order), np.array([remap[x] for x in lab]), significant, {**tests, "shown": k_show}


def power_note(ref, n_members):
    """The resolvable separation, from the calibration file's synthetic-ensemble test, if it was run for this size."""
    p = ref.calib_meta.get("cluster_power") if ref.calib_meta else None
    p = {int(k): v for k, v in (p or {}).items() if k.isdigit()}    # the Gaussian test, keyed by member count
    if not p:
        return ""
    m = min(p, key=lambda k: abs(k - n_members))
    row = p[m]
    return (f"; synthetic test with {m} Gaussian members: two equal groups {row['sep_60']:.1f} sd apart along the "
            f"leading EOF are found 60% of the time, {row['sep_90']:.1f} sd apart 90%, a single Gaussian "
            f"continuum passes {row['fp'] * 100:.0f}%")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ap = GP.common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    base, out, data, runs, cache, ref, clim, z = GP.setup(a)
    if clim is None:
        print("no daily climatology (gefs-clim-v2): clusters skipped"); return 0
    season = GP.season_of(base.month)
    n_members = int(len(z["members"]))
    z500 = GP.member_anoms(z, base, ref, {"zg500": clim["zg500"]})["zg500"]          # (35, M, 73, 144)
    js = {"cycle": a.date, "season": season, "k_tested": list(KS), "null_pct": NULL_PCT, "n_null": N_NULL,
          "n_members": n_members, "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
          "valid_convention": "day d valid on init + (d - 1)", "test": bool(GP.TEST_NOTE), "sectors": {}}
    for sk, spec in GP.SECTORS.items():
        sec, lat, lon = GP.sector_cut(z500, spec)
        w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        cent, _, _ = ref.centroids(sk, season)
        names = [ref.regime_name(sk, season, k) for k in range(GP.K)]
        rows = []
        for wi, (w0, w1) in enumerate(GP.WEEKS):
            X = sec[w0 - 1:w1].mean(0)                                                # (M, lat, lon)
            kc, lab, significant, tests = cluster_week(X, w)
            ratio, thr = tests[tests.pop("shown")]                  # the tested k (kc < it only if a cluster emptied)
            clusters = []
            for c in range(kc):
                m = X[lab == c].mean(0)
                mw = (m * w).ravel(); mw = mw - mw.mean()
                cors = []
                for k in range(GP.K):
                    cw = (cent[k] * w).ravel(); cw = cw - cw.mean()
                    cors.append(float(mw @ cw / (np.linalg.norm(mw) * np.linalg.norm(cw) + 1e-9)))
                kb = int(np.argmax(cors))
                clusters.append(dict(n=int((lab == c).sum()), mean=m, regime=kb, regime_name=names[kb], r=cors[kb],
                                     members=[str(x) for x in np.asarray(z["members"])[lab == c]]))
            rows.append(dict(week=wi + 1, w0=w0, w1=w1, k=kc, ratio=ratio, thr=thr, noise=not significant, tests=tests,
                             ens=X.mean(0), clusters=clusters, lab=lab))
            print(f"  {sk} week {wi + 1}: k={kc} sizes " + "/".join(str(c["n"]) for c in clusters)
                  + " · " + " ".join(f"k{k}:{t[0] * 100:.0f}%/{t[1] * 100:.0f}%" for k, t in tests.items())
                  + ("  one scenario" if not significant else f"  {kc} SCENARIOS")
                  + "  " + ", ".join(f"{c['regime_name']} r{c['r']:+.2f}" for c in clusters), flush=True)
        proj = ccrs.LambertConformal(central_longitude=np.mean(spec["lon"]), standard_parallels=(35, 55))
        ncol = max(r["k"] for r in rows) + 1
        foot = (f"k = {KS[0]}–{KS[-1]} tested per week against a null of {N_NULL} random rotations of the "
                f"members (same covariance, no grouping), the k beating its {NULL_PCT}th percentile by most shown"
                + power_note(ref, n_members) + "; frame colour and label = nearest weather regime; anomalies against "
                "the lead-matched GEFSv12 reforecast climatology (2000–2019), re-based to 1991–2020 (ERA5 shift)")
        foot = textwrap.wrap(foot, int(3.9 * ncol * 17))
        H0 = 2.95 * len(rows) + 1.3                                      # the GEPS geometry, plus room per footnote line
        extra = 0.13 * max(len(foot) - 1, 0)
        H = H0 + extra
        Y = lambda f: (f * H0 + extra) / H                               # noqa: E731  (GEPS fractions -> this height)
        fig = plt.figure(figsize=(3.9 * ncol, H))
        lim = float(np.nanpercentile(np.abs(np.stack([c["mean"] for r in rows for c in r["clusters"]])), 99))
        levels = np.linspace(-lim, lim, 21)
        for ri, r in enumerate(rows):
            panels = [("ensemble mean", r["ens"], None)] + [
                (f"scenario {c + 1} · {cl['n']} members ({cl['n'] / len(r['lab']) * 100:.0f}%)", cl["mean"], cl)
                for c, cl in enumerate(r["clusters"])]
            for ci, (ttl, fld, cl) in enumerate(panels):
                rh = 0.89 / len(rows)
                ax = fig.add_axes([0.01 + ci * (0.98 / ncol), Y(0.965 - (ri + 1) * rh + 0.012), 0.98 / ncol - 0.012,
                                   (rh - 0.070) * H0 / H], projection=proj)
                ax.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
                m = ax.contourf(lon, lat, fld, levels=levels, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
                ax.contour(lon, lat, fld, levels=levels[::2], colors="k", linewidths=0.3, transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="#333")
                if cl is not None:
                    col = COLORS[cl["regime"]]
                    for sp in ax.spines.values():
                        sp.set_edgecolor(col); sp.set_linewidth(1.8 if cl["r"] >= 0.5 else 0.8)
                    sub = f"nearest regime: {cl['regime_name']} (r {cl['r']:+.2f})"
                else:
                    d0 = base + dt.timedelta(days=r["w0"] - 1); d1 = base + dt.timedelta(days=r["w1"] - 1)
                    ttl = f"week {r['week']} · {d0:%b %d} – {d1:%b %d} · ensemble mean"
                    sub = (f"{r['k']} distinct scenarios: k={r['k']} explains {r['ratio'] * 100:.0f}% of spread, "
                           f"null {r['thr'] * 100:.0f}%" if not r["noise"] else
                           f"one scenario with spread (no k in {KS[0]}–{KS[-1]} beats its null)")
                ax.set_title(f"{ttl}\n{sub}", fontsize=7.6, fontweight="bold" if cl is None else "normal",
                             color=("#8a1d1d" if (cl is None and r["noise"]) else
                                    "#1f1f1f" if (cl is None or cl["r"] >= 0.5) else GP.MUTED))
                if r["noise"] and cl is not None:
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", alpha=0.45,
                                               zorder=5, lw=0))
                    ax.text(0.5, 0.5, "not distinct", transform=ax.transAxes, ha="center", va="center", fontsize=13,
                            color=GP.MUTED, alpha=0.8, fontweight="bold", zorder=6)
        cax = fig.add_axes([0.32, Y(0.040), 0.36, 0.011 * H0 / H])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal"); cb.set_label("500 hPa height anomaly (m)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        title = (f"{spec['label']} — 500 hPa scenarios: k-means clusters of the {n_members} members' weekly patterns · "
                 f"GEFS init {base:%Y-%m-%d}" + GP.TEST_NOTE)
        fig.suptitle("\n".join(textwrap.wrap(title, int(3.9 * ncol * 8.2))), fontsize=12, fontweight="bold",
                     y=1 - 0.005 * H0 / H, va="top")
        fig.text(0.5, 0.004 * H0 / H, "\n".join(foot), ha="center", va="bottom",
                 fontsize=7.4, color=GP.MUTED)
        GP.save_webp(fig, out / f"gefs_clusters_{sk}.webp")
        plt.close(fig)
        print(f"  gefs_clusters_{sk}.webp", flush=True)
        js["sectors"][sk] = {"label": spec["label"], "weeks": [
            dict(week=r["week"], w0=r["w0"], w1=r["w1"], k=r["k"], distinct=not r["noise"],
                 tests={str(k): [round(t[0], 3), round(t[1], 3)] for k, t in r["tests"].items()},
                 clusters=[dict(n=c["n"], regime=c["regime_name"], r=round(c["r"], 2), members=c["members"])
                           for c in r["clusters"]]) for r in rows]}
    (data / "gefs_clusters.json").write_text(json.dumps(js))
    return 0


if __name__ == "__main__":
    sys.exit(main())
