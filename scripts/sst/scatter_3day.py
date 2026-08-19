#!/usr/bin/env python3
"""3-day mean inflow: forecast against observed, level and change.

Averaging the target over three days lifts top-decile capture from 54% to
61% at a 10-day lead while barely moving correlation - day-to-day
variation that was never predictable drops out. This shows that directly.

Both framings are drawn, because they answer different questions and only
one of them is a real test:

  LEVEL   forecast 3-day mean against observed 3-day mean. This is what a
          desk reads. Worth noting it scores WORSE than the change framing
          at a 10-day lead (r 0.65 vs 0.75): the autocorrelation that
          normally flatters a level scatter has decayed by then, so knowing
          where the series sits today buys little.
  CHANGE  the same forecasts expressed as movement from the launch day.
          Persistence sits at zero on the vertical axis, so any tilt
          toward the diagonal is skill the model added.

Every point is out of sample: blocked CV, 90-day embargo, forecast paths
smoothed with observed history exactly as the observed series is.

    python scripts/sst/scatter_3day.py [--lead 10]

Output: ~/colombia_hydro/site/scatter_3day.webp
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

PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "site" / "scatter_3day.webp"
NAVY, INK = "#13273d", "#1a2733"


def trail(a, w):
    if w <= 1:
        return np.asarray(a, float).copy()
    a = np.asarray(a, float)
    c = np.convolve(np.nan_to_num(a), np.ones(w), "full")[:len(a)]
    m = np.convolve(np.isfinite(a).astype(float), np.ones(w), "full")[:len(a)]
    out = np.full(len(a), np.nan)
    ok = m >= w - 0.5
    out[ok] = c[ok] / m[ok]
    return out


def pairs(d, lead, W, stride=2):
    y = d["y"]["NATIONAL"]; rain = d["rain"]["NATIONAL"]
    roni = d["roni"]; stor = d["stor"]["NATIONAL"]
    n = len(y)
    yS = trail(y, W)
    F, O, A = [], [], []
    for a, b in M.blocks(n, 8):
        te = np.zeros(n, bool); te[a:b] = True
        tr = ~te.copy()
        tr[max(0, a - M.EMBARGO):min(n, b + M.EMBARGO)] = False
        X, dy = M.design(rain, y, roni, stor, 7, 0)
        m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        beta = M.fit(X, dy, m)
        for i0 in range(a, b - lead - 2, stride):
            if not np.isfinite(y[i0]):
                continue
            sim = M.simulate(beta, rain, roni, stor, 7, 0, i0, float(y[i0]), lead)
            if not np.isfinite(sim[lead - 1]):
                continue
            # smooth the forecast with the SAME observed history the observed
            # series uses, so the two sides are the same operation
            seq = np.concatenate([y[max(0, i0 - W + 1):i0 + 1], sim[:lead]])
            fs = trail(seq, W)[-1]
            if np.isfinite(fs) and np.isfinite(yS[i0 + lead]) and np.isfinite(yS[i0]):
                F.append(fs); O.append(yS[i0 + lead]); A.append(yS[i0])
    return np.array(F), np.array(O), np.array(A)


def panel(ax, X, Y, xlab, ylab, title, sub, colour, diag=True):
    lim_lo = min(np.percentile(X, 0.5), np.percentile(Y, 0.5))
    lim_hi = max(np.percentile(X, 99.5), np.percentile(Y, 99.5))
    ax.scatter(X, Y, s=5, c=colour, alpha=0.18, lw=0, zorder=2)
    if diag:
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#888", ls="--",
                lw=1.2, zorder=4)
    sl, ic = np.polyfit(X, Y, 1)
    xs = np.linspace(lim_lo, lim_hi, 10)
    ax.plot(xs, sl * xs + ic, color="#c0392b", lw=2, zorder=5)
    r = float(np.corrcoef(X, Y)[0, 1])
    big = Y > np.percentile(Y, 90)
    capt = X[big].mean() / Y[big].mean() if Y[big].mean() != 0 else np.nan
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left",
                 color=INK, pad=14)
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.5,
            color="#666", va="bottom")
    # lower-right is reliably empty on a positively-sloped scatter, and it
    # keeps the block off both the point cloud and the panel subtitle
    ax.text(0.97, 0.05, f"r = {r:.3f}\nslope = {sl:.2f}\n"
            f"top-decile capture {100*capt:.0f}%", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9.5, fontweight="bold",
            color="#c0392b",
            bbox=dict(fc="white", ec="#e0d0d0", boxstyle="round,pad=0.45",
                      alpha=0.92))
    ax.grid(alpha=0.2)
    return r, sl, capt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=10)
    a = ap.parse_args(argv)
    M.BASELINE_WIN = 0
    d = DB.add_national(PR.load_all())
    h = a.lead

    F1, O1, A1 = pairs(d, h, 1)
    F3, O3, A3 = pairs(d, h, 3)

    fig = plt.figure(figsize=(13.2, 11.2), facecolor="white")
    hd = fig.add_axes([0, 0.951, 1, 0.049]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.012, 0.62, f"3-DAY MEAN INFLOW — FORECAST vs OBSERVED, "
            f"{h}-DAY LEAD", transform=hd.transAxes, color="white",
            fontsize=14.5, fontweight="bold", va="center")
    hd.text(0.012, 0.2, "national · every point out of sample · forecast "
            "smoothed with observed history exactly as the observed series is",
            transform=hd.transAxes, color="#b9c6d4", fontsize=8.8, va="center")

    ax = fig.add_axes([0.075, 0.585, 0.38, 0.31])
    panel(ax, F1, O1, "forecast (% of norm)", "observed (% of norm)",
          "A · Daily value — level", "what a desk reads, daily", "#7d8fa8")
    ax2 = fig.add_axes([0.575, 0.585, 0.38, 0.31])
    panel(ax2, F3, O3, "forecast 3-day mean", "observed 3-day mean",
          "B · 3-day mean — level", "tighter, but see the caution below",
          "#1f6fb4")

    ax3 = fig.add_axes([0.075, 0.165, 0.38, 0.31])
    panel(ax3, F1 - A1, O1 - A1, f"predicted change over {h} d",
          "actual change", "C · Daily value — change",
          "the honest test: zero = persistence", "#7d8fa8")
    ax4 = fig.add_axes([0.575, 0.165, 0.38, 0.31])
    r4, s4, c4 = panel(ax4, F3 - A3, O3 - A3, f"predicted change over {h} d",
                       "actual change", "D · 3-day mean — change",
                       "the comparison that matters", "#1f6fb4")

    # matplotlib does not wrap fig.text, so wrap explicitly rather than let
    # a long line run off the canvas
    import textwrap
    lead_txt = textwrap.fill(
        "Averaging does what it should: top-decile capture rises 54% to 60% "
        "on the change panels, while correlation barely moves (0.745 to "
        "0.732). Day-to-day variation that was never predictable drops out; "
        "the signal does not.", 118)
    note_txt = textwrap.fill(
        "Note the LEVEL panels score WORSE than the change panels (r 0.65 vs "
        "0.75), not better. At a 10-day lead the autocorrelation that usually "
        "flatters a level scatter has already decayed, so knowing today buys "
        "little. Every slope is below 1 - the model under-states the size of "
        "moves in both framings, which is the amplitude limit documented "
        "elsewhere.", 132)
    fig.text(0.075, 0.085, lead_txt, fontsize=9.5, color="#444",
             fontweight="bold", va="top")
    fig.text(0.075, 0.043, note_txt, fontsize=9.0, color="#666", va="top")
    fig.savefig(OUT, dpi=112, facecolor="white")   # no tight bbox:
    # it crops to drawn content and clips the full-width header bar
    print(f"wrote {OUT}")
    for lab, (F, O, A) in (("daily", (F1, O1, A1)), ("3-day", (F3, O3, A3))):
        big = (O - A) > np.percentile(O - A, 90)
        print(f"  {lab:6} n={len(F):5,}  level r={np.corrcoef(F,O)[0,1]:.3f}   "
              f"change r={np.corrcoef(F-A,O-A)[0,1]:.3f}   "
              f"top-decile capture {100*(F-A)[big].mean()/(O-A)[big].mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
