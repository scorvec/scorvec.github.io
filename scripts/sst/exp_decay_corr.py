#!/usr/bin/env python3
"""Exponential die-off rainfall model vs XM inflows (API / linear reservoir).

Instead of boxcar 3-day sums, filter daily rain with an exponential memory
    S_t = R_t + a·S_{t-1},   a = exp(-1/tau)
— the impulse response of a linear reservoir with recession constant tau —
and correlate S (optionally shifted by a pure routing delay) against DAILY
inflow. Scanning tau maps each region/river's effective catchment memory.

Raw and anomaly versions are both scanned: long-tau raw filters partly
reconstruct the seasonal cycle, so the anomaly version (rain minus IMERG
harmonic clim, inflow minus its own harmonic fit) is the honest test at
tau ≳ 10 d; raw is shown for continuity with the 3-day validation.

Inputs:  ~/colombia_hydro/out/river_series.json   (from river_corr.py)
         assets/sst/data/colombia_region_rain.json
Outputs: ~/colombia_hydro/out/exp_decay.json
         colombia_hydro/exp_decay_regions.webp    (r vs tau curves)
         colombia_hydro/exp_decay_rivers.webp     (per-river best r + tau)

    python scripts/sst/exp_decay_corr.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_region_rain import ORDER, COLORS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = Path.home() / "colombia_hydro" / "out"
SERIES = OUT / "river_series.json"
RAIN_JSON = REPO / "assets" / "sst" / "data" / "colombia_region_rain.json"
PNG_REG = REPO / "colombia_hydro" / "exp_decay_regions.webp"
PNG_RIV = REPO / "colombia_hydro" / "exp_decay_rivers.webp"
TAUS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0]
LAGS = range(0, 8)                      # pure routing delay on top of the memory
MIN_N = 90


def efilt(x: np.ndarray, tau: float, seg_id: np.ndarray) -> np.ndarray:
    """Exponential accumulation, reset at record gaps, spin-up (3·tau) masked."""
    a = float(np.exp(-1.0 / tau))
    out = np.full(len(x), np.nan)
    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        s = 0.0
        for j, i in enumerate(idx):
            s = (0.0 if not np.isfinite(x[i]) else x[i]) + a * s
            if j >= int(np.ceil(3 * tau)):
                out[i] = s
    return out


def _harm_anom(y: np.ndarray, doy: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w),
                         np.cos(2 * w), np.sin(2 * w)])
    fin = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[fin], y[fin], rcond=None)
    return y - X @ coef


def scan_tau(rr, cl, ii, seg_id, doy):
    """{variant: {tau: (lag, r)}} + best per variant, correlating daily inflow."""
    ia = _harm_anom(ii, doy)
    curves = {"raw": {}, "anom": {}}
    for tau in TAUS:
        f_raw = efilt(rr, tau, seg_id)
        f_anm = efilt(rr - cl, tau, seg_id)
        for name, xs, ys in (("raw", f_raw, ii), ("anom", f_anm, ia)):
            best = (0, -9.0)
            for lag in LAGS:
                xr_ = np.roll(xs, lag); xr_[:lag] = np.nan
                m = np.isfinite(xr_) & np.isfinite(ys) & (np.roll(seg_id, lag) == seg_id)
                if m.sum() > MIN_N:
                    cc = float(np.corrcoef(xr_[m], ys[m])[0, 1])
                    if cc > best[1]:
                        best = (lag, cc)
            if best[1] > -9:
                curves[name][tau] = best
    out = {}
    for name, c in curves.items():
        if c:
            bt = max(c, key=lambda t: c[t][1])
            out[name] = dict(tau=bt, lag=c[bt][0], r=round(c[bt][1], 3),
                             curve={str(t): dict(lag=l, r=round(r, 3))
                                    for t, (l, r) in c.items()})
    return out


def main() -> int:
    S = json.load(open(SERIES))
    RJ = json.load(open(RAIN_JSON))["regions"]
    # the two inputs are regenerated at different times — align on common dates
    common = sorted(set(S["dates"]) & set(RJ[ORDER[0]]["dates"]))
    iS = [S["dates"].index(d) for d in common]
    iR = [RJ[ORDER[0]]["dates"].index(d) for d in common]
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in common]
    n = len(dts)
    print(f"{n} aligned days ({common[0]} … {common[-1]})", flush=True)
    seg_id = np.zeros(n, int)
    for i in range(1, n):
        seg_id[i] = seg_id[i - 1] + ((dts[i] - dts[i - 1]).days > 1)
    doy = np.array([d.timetuple().tm_yday for d in dts], float)
    tonan = lambda a: np.array([np.nan if v is None else v for v in a], float)[iS]
    results = {"regions": {}, "rivers": {}}
    print(f"{'series':<32} {'variant':>7} {'tau':>5} {'lag':>4} {'r':>6}")
    for reg in ORDER:
        rr = np.array(RJ[reg]["mm"], float)[iR]
        cl = np.array(RJ[reg]["clim"], float)[iR]
        ii = np.full(n, np.nan)
        for name, rv in S["rivers"].items():
            if rv["region"] == reg:
                v = tonan(rv["inflow"])
                ii = np.where(np.isfinite(v), np.nan_to_num(ii) + v, ii)
        res = scan_tau(rr, cl, ii, seg_id, doy)
        results["regions"][reg] = res
        for vn, r in res.items():
            print(f"{reg:<32} {vn:>7} {r['tau']:>5.1f} {r['lag']:>4d} {r['r']:>6.2f}")

    for name, rv in S["rivers"].items():
        ii = tonan(rv["inflow"])
        if np.isfinite(ii).sum() < MIN_N:
            continue
        res = scan_tau(tonan(rv["rain"]), tonan(rv["clim"]), ii, seg_id, doy)
        res["region"] = rv["region"]
        results["rivers"][name] = res
        b = max((res[v] for v in ("raw", "anom") if v in res), key=lambda r: r["r"])
        print(f"{name:<32} {'best':>7} {b['tau']:>5.1f} {b['lag']:>4d} {b['r']:>6.2f}")

    (OUT / "exp_decay.json").write_text(json.dumps(results, indent=1))

    # region figure: r(tau) curves, raw + anom, vs the 3-day boxcar baseline
    try:
        box = json.load(open(OUT / "validation_summary.json"))
    except FileNotFoundError:
        box = {}
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.6), sharex=True, sharey=True)
    for ax, reg in zip(axs.ravel(), ORDER):
        res = results["regions"][reg]
        for vn, ls in (("raw", "-"), ("anom", "--")):
            if vn not in res:
                continue
            cv = res[vn]["curve"]
            ts = sorted(float(t) for t in cv)
            ax.plot(ts, [cv[str(t)]["r"] for t in ts], ls, color=COLORS[reg],
                    lw=1.8 if vn == "raw" else 1.4, label=vn)
            b = res[vn]
            ax.plot(b["tau"], b["r"], "o", color=COLORS[reg], ms=5)
        if reg in box:
            ax.axhline(box[reg]["corr"], color="0.35", lw=1.0, ls=":",
                       label=f"3-day boxcar ({box[reg]['corr']:.2f})")
        bb = res.get("anom") or res["raw"]         # headline the deseasonalized number
        ax.set_title(f"{reg} — anom r={bb['r']:.2f} at τ={bb['tau']:g}d",
                     fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30])
        ax.set_xticklabels(["1", "2", "5", "10", "30"])
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7.5)
        if reg == ORDER[0]:
            ax.legend(fontsize=6.5, loc="lower right")
    for ax in axs[-1]:
        ax.set_xlabel("memory τ (days)", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("r (daily inflow)", fontsize=8)
    fig.suptitle("Exponential rainfall memory — r vs decay timescale τ, per region\n"
                 "S$_t$ = R$_t$ + e$^{-1/τ}$·S$_{t-1}$ vs daily inflow · dotted = 3-day boxcar baseline",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PNG_REG, dpi=120, bbox_inches="tight", facecolor="white")

    # river figure: best r with tau annotation, ordered by inflow share
    try:
        share = {r["river"]: r["share_pct"]
                 for r in json.load(open(OUT / "river_corr.json"))["rivers"]}
    except FileNotFoundError:
        share = {}
    fig2, axs2 = plt.subplots(2, 3, figsize=(13.2, 8.6))
    for ax, reg in zip(axs2.ravel(), ORDER):
        rr = sorted((n_ for n_, r in results["rivers"].items()
                     if r["region"] == reg and "anom" in r),
                    key=lambda n_: share.get(n_, 0.0))
        ys = np.arange(len(rr))
        for y, name in zip(ys, rr):
            # anomaly variant only: raw + long tau partly reconstructs the
            # seasonal cycle, which would flatter exactly the wrong rivers
            b = results["rivers"][name]["anom"]
            ax.barh(y, b["r"], color=COLORS[reg], alpha=0.85, edgecolor="k", lw=0.4)
            ax.text(max(b["r"], 0) + 0.015, y,
                    f"τ={b['tau']:g}d · {share.get(name, 0):.0f}%",
                    va="center", fontsize=6, color="0.35")
        ax.set_yticks(ys)
        ax.set_yticklabels([n_.title()[:22] for n_ in rr], fontsize=6.5)
        ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-0.05, 1.0)
        ax.set_title(reg, fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel("best r (exp memory, daily)", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.25)
    fig2.suptitle("Exponential rainfall memory per river — anomaly basis (seasonal cycle removed)\n"
                  "labels: fitted memory τ · share of region inflow",
                  fontsize=11.5, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2.savefig(PNG_RIV, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"wrote {PNG_REG.name} + {PNG_RIV.name} + exp_decay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
