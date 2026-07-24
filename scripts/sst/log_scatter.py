#!/usr/bin/env python3
"""Log-space rain->inflow scatterplots (weekly blocks) for the hydro page.

Rain and inflow are both heavily right-skewed, so Pearson on raw mm hides the
relationship a log-log view makes obvious: non-overlapping 7-day totals plot
as a clean power law on free-flowing rivers. Two figures:

  validation_log_regions.webp   six regions
  validation_log_rivers.webp    five clean rivers + Bogota N.R. (the
                                operations-dominated counterexample)

r shown is Pearson in log10 space on the plotted (non-overlapping) blocks,
with Spearman rank alongside. Rain is shifted by the best daily lag (0-6 d,
chosen in log space on stride-1 blocks) before blocking.

Inputs: out/river_series.json (river_corr.py) + colombia_region_rain.json
    python scripts/sst/log_scatter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_region_rain import ORDER, COLORS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = Path.home() / "colombia_hydro" / "out"
PNG_REG = REPO / "colombia_hydro" / "validation_log_regions.webp"
PNG_RIV = REPO / "colombia_hydro" / "validation_log_rivers.webp"
K = 7                                   # block length, days
RIVERS = ["GUAVIO", "SAN CARLOS", "SOGAMOSO", "SINU URRA", "GRANDE",
          "BOGOTA N.R."]               # last one is the regulated counterexample


def block_stride(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-day sums (stride 1), NaN when any member missing."""
    c = np.convolve(np.nan_to_num(x), np.ones(k), "full")[:len(x)]
    bad = np.convolve((~np.isfinite(x)).astype(float), np.ones(k), "full")[:len(x)]
    c[bad > 0] = np.nan
    c[:k - 1] = np.nan
    return c


def best_lag(rr: np.ndarray, ii: np.ndarray) -> int:
    """Daily lag (rain leads) maximizing log-space Pearson on stride-1 blocks."""
    rb, ib = block_stride(rr, K), block_stride(ii, K)
    best, bl = -9.0, 0
    for lag in range(0, 7):
        xr = np.roll(rb, lag); xr[:lag] = np.nan
        m = np.isfinite(xr) & np.isfinite(ib) & (xr > 0) & (ib > 0)
        if m.sum() > 60:
            cc = float(np.corrcoef(np.log10(xr[m]), np.log10(ib[m]))[0, 1])
            if cc > best:
                best, bl = cc, lag
    return bl


def weekly_points(rr: np.ndarray, ii: np.ndarray, lag: int):
    """Non-overlapping K-day mean rain (lagged) and inflow, positive pairs."""
    rs = np.roll(rr, lag).astype(float); rs[:lag] = np.nan
    n = (len(rr) // K) * K
    rw = np.nanmean(rs[:n].reshape(-1, K), axis=1)
    iw = ii[:n].reshape(-1, K)
    iw = np.where(np.isfinite(iw).all(axis=1), iw.mean(axis=1), np.nan)
    m = np.isfinite(rw) & np.isfinite(iw) & (rw > 0) & (iw > 0)
    return rw[m], iw[m]


def panel(ax, x, y, color, title, unit_y="GWh/day"):
    lr = float(np.corrcoef(np.log10(x), np.log10(y))[0, 1])
    sr = float(spearmanr(x, y).statistic)
    ax.scatter(x, y, s=16, color=color, alpha=0.6, edgecolors="none")
    b, a = np.polyfit(np.log10(x), np.log10(y), 1)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 40)
    ax.plot(xs, 10 ** (a + b * np.log10(xs)), color="0.2", lw=1.4, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title + f"  —  r$_{{log}}$={lr:.2f} · ρ={sr:.2f}",
                 fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel(f"{K}-day mean rain (mm/day)", fontsize=8)
    ax.set_ylabel(f"{K}-day mean inflow ({unit_y})", fontsize=8)
    ax.tick_params(labelsize=7.5, which="both")
    ax.grid(alpha=0.22, which="both")
    return lr, sr


def main() -> int:
    S = json.load(open(OUT / "river_series.json"))
    RJ = json.load(open(REPO / "assets" / "sst" / "data" /
                        "colombia_region_rain.json"))["regions"]
    common = sorted(set(S["dates"]) & set(RJ[ORDER[0]]["dates"]))
    iS = [S["dates"].index(d) for d in common]
    iR = [RJ[ORDER[0]]["dates"].index(d) for d in common]
    n = len(common)
    tonan = lambda a: np.array([np.nan if v is None else v for v in a], float)

    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for ax, reg in zip(axs.ravel(), ORDER):
        rr = np.array(RJ[reg]["mm"], float)[iR]
        ii = np.full(n, np.nan)
        for name, rv in S["rivers"].items():
            if rv["region"] == reg:
                v = tonan(rv["inflow"])[iS]
                ii = np.where(np.isfinite(v), np.nan_to_num(ii) + v, ii)
        lag = best_lag(rr, ii)
        x, y = weekly_points(rr, ii, lag)
        lr, _ = panel(ax, x, y, COLORS[reg], f"{reg} (rain led {lag} d)")
        print(f"  {reg}: n={len(x)} weeks, r_log={lr:.2f}, lag={lag}d", flush=True)
    fig.suptitle("Weekly rain vs inflow, log-log — each dot is one (non-overlapping) week\n"
                 "r$_{log}$ = Pearson in log space · ρ = Spearman rank",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PNG_REG, dpi=120, bbox_inches="tight", facecolor="white")

    fig2, axs2 = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for ax, name in zip(axs2.ravel(), RIVERS):
        rv = S["rivers"][name]
        rr, ii = tonan(rv["rain"])[iS], tonan(rv["inflow"])[iS]
        lag = best_lag(rr, ii)
        x, y = weekly_points(rr, ii, lag)
        reg = rv["region"]
        regulated = name == "BOGOTA N.R."
        title = f"{name.title()} [{reg.title()}]" + (" — regulated" if regulated else "")
        panel(ax, x, y, "#8a8a8a" if regulated else COLORS[reg], title)
    fig2.suptitle("The same view per river — five free-flowing rivers and one "
                  "operations-dominated series\nweekly blocks, log-log",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    fig2.savefig(PNG_RIV, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"wrote {PNG_REG.name} + {PNG_RIV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
