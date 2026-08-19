#!/usr/bin/env python3
"""Four charts for someone deciding whether to rely on this model.

Not diagnostics - the questions a strategist actually asks:

  A  Does the forecast contain information? Predicted vs observed change,
     out of sample, every forecast in the record at one lead.
  B  Is it better than doing nothing? Persistence - "tomorrow looks like
     today" - is the honest benchmark for a series this autocorrelated,
     and beating it is the whole claim.
  C  Are the uncertainty bands honest? If the 80% interval contains the
     outturn 80% of the time it can be sized against. If it contains it
     60% of the time the model is lying about its confidence.
  D  Does acting on it pay? Group every forecast by what it predicted,
     then show what actually happened in each group. A model with no
     information gives flat bars.

Every point is out of sample: blocked CV, 90-day embargo, timescales
re-selected inside each training fold.

    python scripts/sst/model_evidence.py [--lead 10]

Output: ~/colombia_hydro/site/model_evidence.webp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inflow_delta_model as M                                     # noqa: E402
import perfect_rain_backtest as PR                                 # noqa: E402
import delta_backtest_long as DB                                   # noqa: E402
import scatter_galaxy as SG                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "site" / "model_evidence.webp"
NAVY, INK = "#13273d", "#1a2733"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=10)
    a = ap.parse_args(argv)
    M.BASELINE_WIN = 0
    d = DB.add_national(PR.load_all())
    bt = M.backtest(d, "NATIONAL")

    L = bt["leads"]
    fig = plt.figure(figsize=(14.6, 9.8), facecolor="white")
    hd = fig.add_axes([0, 0.947, 1, 0.053]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.012, 0.62, "IS THIS MODEL WORTH USING? — NATIONAL INFLOW",
            transform=hd.transAxes, color="white", fontsize=14.5,
            fontweight="bold", va="center")
    hd.text(0.012, 0.2, "every forecast out of sample · blocked "
            "cross-validation, 90-day embargo, settings chosen inside each "
            "training fold", transform=hd.transAxes, color="#b9c6d4",
            fontsize=8.8, va="center")

    # backtest() keeps summaries, not the pairs; scatter_galaxy.all_pairs
    # regenerates every out-of-sample (predicted, observed) change at one
    # lead under the same blocked-CV rules, so reuse it rather than write a
    # second implementation that could drift from it
    h = a.lead
    P, O, Y0, DT, EN = SG.all_pairs(d, "NATIONAL", h)
    ok = np.isfinite(P) & np.isfinite(O)
    P, O = P[ok], O[ok]

    # ---- A: predicted vs observed change
    ax = fig.add_axes([0.055, 0.535, 0.40, 0.345])
    ax.axhline(0, color="#bbb", lw=0.9); ax.axvline(0, color="#bbb", lw=0.9)
    lim = np.percentile(np.abs(np.r_[P, O]), 99.3)
    ax.plot([-lim, lim], [-lim, lim], color="#888", ls="--", lw=1.2, zorder=3)
    ax.scatter(P, O, s=5, c="#1f6fb4", alpha=0.20, lw=0, zorder=2)
    r = float(np.corrcoef(P, O)[0, 1])
    b1 = float(np.polyfit(P, O, 1)[0])
    xs = np.linspace(-lim, lim, 10)
    ax.plot(xs, np.polyval(np.polyfit(P, O, 1), xs), color="#c0392b", lw=2,
            zorder=4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"predicted change over {h} days (points of norm)", fontsize=9)
    ax.set_ylabel("what actually happened", fontsize=9)
    ax.set_title(f"A · The forecast carries information", fontsize=11.5,
                 fontweight="bold", loc="left", color=INK, pad=14)
    ax.text(0.0, 1.015, f"n = {len(P):,} forecasts, all out of sample",
            transform=ax.transAxes, fontsize=8.6, color="#666", va="bottom")
    ax.text(0.03, 0.95, f"r = {r:.2f}\nslope = {b1:.2f}", transform=ax.transAxes,
            va="top", fontsize=10.5, fontweight="bold", color="#c0392b")
    ax.grid(alpha=0.2)

    # ---- B: skill vs persistence by lead
    ax2 = fig.add_axes([0.575, 0.535, 0.40, 0.345])
    hs = sorted(int(k) for k in L)
    sk = [L[k].get("rmse_skill_vs_persistence") for k in hs]
    hs = [k for k, s in zip(hs, sk) if s is not None]
    sk = [s for s in sk if s is not None]
    cols = ["#2c6e49" if s > 0 else "#c0392b" for s in sk]
    ax2.bar(hs, [100 * s for s in sk], color=cols, width=0.72)
    ax2.axhline(0, color="#333", lw=1.1)
    ax2.set_xlabel("forecast lead (days)", fontsize=9)
    ax2.set_ylabel("% less error than persistence", fontsize=9)
    ax2.set_title("B · It beats assuming tomorrow looks like today",
                  fontsize=11.5, fontweight="bold", loc="left", color=INK,
                  pad=14)
    ax2.text(0.0, 1.015, "persistence is the benchmark that matters for a "
             "series this autocorrelated", transform=ax2.transAxes,
             fontsize=8.6, color="#666", va="bottom")
    ax2.grid(alpha=0.2, axis="y")
    for x, s in zip(hs, sk):
        if x in (1, 5, 10, 15):
            ax2.annotate(f"{100*s:.0f}%", (x, 100 * s), ha="center",
                         va="bottom" if s > 0 else "top",
                         textcoords="offset points", xytext=(0, 3 if s > 0 else -3),
                         fontsize=8.5, fontweight="bold")

    # ---- C: are the intervals honest?
    ax3 = fig.add_axes([0.055, 0.075, 0.40, 0.345])
    # COVERAGE MUST NOT BE CIRCULAR. Building the interval from the same
    # residuals it is then scored against guarantees the answer - an 80%
    # interval taken from a sample's own percentiles covers that sample 80%
    # of the time by construction, and proves nothing. Split into contiguous
    # blocks, size the interval on the OTHER blocks, and score it on the one
    # held out.
    err = O - P
    nblk = 12
    edges = np.linspace(0, len(err), nblk + 1).astype(int)
    noms, obss = [], []
    for nominal in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        inside, total = 0, 0
        for bnum in range(nblk):
            te = np.zeros(len(err), bool)
            te[edges[bnum]:edges[bnum + 1]] = True
            tr = ~te
            if tr.sum() < 200 or te.sum() < 20:
                continue
            lo, hi = np.percentile(err[tr], [(100 - nominal) / 2,
                                             100 - (100 - nominal) / 2])
            inside += int(np.sum((err[te] >= lo) & (err[te] <= hi)))
            total += int(te.sum())
        noms.append(nominal); obss.append(100.0 * inside / max(total, 1))
    ax3.plot([0, 100], [0, 100], color="#888", ls="--", lw=1.2)
    ax3.plot(noms, obss, "o-", color="#6b2d7d", lw=2.2, ms=6)
    ax3.set_xlabel("interval we quote (%)", fontsize=9)
    ax3.set_ylabel("how often the outturn actually landed inside (%)", fontsize=9)
    ax3.set_title("C · The uncertainty bands are honest", fontsize=11.5,
                  fontweight="bold", loc="left", color=INK, pad=14)
    ax3.text(0.0, 1.015, "interval sized on other blocks, scored on the one "
             "held out — never on its own residuals", transform=ax3.transAxes,
             fontsize=8.6, color="#666", va="bottom")
    ax3.grid(alpha=0.2); ax3.set_xlim(0, 100); ax3.set_ylim(0, 100)

    # ---- D: does acting on it pay?
    ax4 = fig.add_axes([0.575, 0.075, 0.40, 0.345])
    qs = np.percentile(P, [0, 20, 40, 60, 80, 100])
    labs = ["biggest\nfalls", "falls", "flat", "rises", "biggest\nrises"]
    mids, p25s, p75s = [], [], []
    for i in range(5):
        m = (P >= qs[i]) & (P <= qs[i + 1])
        mids.append(np.median(O[m]))
        p25s.append(np.percentile(O[m], 25)); p75s.append(np.percentile(O[m], 75))
    xx = np.arange(5)
    ax4.bar(xx, mids, color=["#c0392b", "#e08214", "#999", "#5b9bd5", "#2c6e49"],
            width=0.68, zorder=3)
    ax4.vlines(xx, p25s, p75s, color="#333", lw=1.6, zorder=4)
    ax4.axhline(0, color="#333", lw=1.1)
    ax4.set_xticks(xx); ax4.set_xticklabels(labs, fontsize=9)
    ax4.set_xlabel("what the model predicted (quintile)", fontsize=9)
    ax4.set_ylabel(f"median actual change over {h} days", fontsize=9)
    ax4.set_title("D · Acting on it would have paid", fontsize=11.5,
                  fontweight="bold", loc="left", color=INK, pad=14)
    ax4.text(0.0, 1.015, "a model with no information gives five flat bars; "
             "whiskers are the middle half of outcomes",
             transform=ax4.transAxes, fontsize=8.6, color="#666", va="bottom")
    ax4.grid(alpha=0.2, axis="y")
    spread = mids[-1] - mids[0]
    ax4.text(0.03, 0.95, f"{spread:.0f} points of norm separate\nthe top and "
             f"bottom groups", transform=ax4.transAxes, va="top", fontsize=9.5,
             fontweight="bold", color="#2c6e49")

    fig.savefig(OUT, dpi=112, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  A: lead {h}, n={len(P):,}, r={r:.3f}, slope={b1:.2f}")
    print(f"  B: skill vs persistence h1 {100*sk[0]:.0f}%, h{hs[-1]} {100*sk[-1]:.0f}%")
    print(f"  C: 80% interval actually covers {obss[noms.index(80)]:.0f}%")
    print(f"  D: top-vs-bottom quintile spread {spread:.1f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
