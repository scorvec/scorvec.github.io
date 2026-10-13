#!/usr/bin/env python3
"""Figures for the conservation-and-drag intercomparison, from scripts/amdrag/data/results.json only.
Writes WebP files to <site>/studies/model-drag-intercomparison/.

    python scripts/amdrag/figures.py --site .
Colours: the site's categorical palette in fixed order (one hue per model, never cycled); ERA5 is the ink reference.
Solid lines = physics models, dashed = AI models, dotted = the GDPS hybrid (nudged toward an AI model since May 2026).
Filled markers = significant after Benjamini-Hochberg (q = 0.10); hollow = not significant.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                     # noqa: E402

SLUG = "model-drag-intercomparison"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#6f6b64", "#e1e0d9", "#fcfcfb"
ORDER = ["gfs", "gefs", "aigfs", "ifs-ens", "aifs-single", "aifs-ens", "gdps", "geps"]
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
LBL = {"gfs": "GFS", "gefs": "GEFS", "aigfs": "AIGFS (AI)", "ifs-ens": "IFS-ENS", "aifs-single": "AIFS (AI)",
       "aifs-ens": "AIFS-ENS (AI)", "gdps": "GDPS (hybrid)", "geps": "GEPS", "era5": "ERA5"}
LS = {"physics": "-", "AI": "--", "hybrid": ":"}
DIV = LinearSegmentedColormap.from_list("bgr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b3b2", "#e34948",
                                                "#8a1c1b"])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})
REGLBL = {"ORO": "rough terrain", "OCX": "ocean 30-65°", "OCO": "other ocean", "LND": "other land"}
RANGE_LBL = {"HIM": "Himalaya/Tibet", "ROC": "Rockies", "AND": "Andes", "GRL": "Greenland", "ANT": "Antarctica",
             "ALP": "Alps/Europe"}


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    from PIL import Image
    Image.open(buf).save(out / f"{name}.webp", "WEBP", quality=88, method=6)
    print("wrote", out / f"{name}.webp")


def series(ax, key, rows, kind, offset=0.0, scale=1.0, label=None):
    """Mean +- 95 % CI by lead; filled = significant (FDR), hollow = not."""
    x = np.array([r["lead"] for r in rows], float) + offset
    m = np.array([np.nan if r["mean"] is None else r["mean"] for r in rows], float) * scale
    lo = np.array([np.nan if r.get("lo") is None else r["lo"] for r in rows], float) * scale
    hi = np.array([np.nan if r.get("hi") is None else r["hi"] for r in rows], float) * scale
    sig = np.array([bool(r.get("sig")) for r in rows])
    ok = np.isfinite(m)
    if not ok.any():
        return
    ax.plot(x[ok], m[ok], LS[kind], color=COL[key], lw=1.8, label=label or LBL[key])
    ax.vlines(x[ok], lo[ok], hi[ok], color=COL[key], lw=0.9, alpha=0.7)
    ax.scatter(x[ok & sig], m[ok & sig], s=26, color=COL[key], zorder=3, edgecolor=SURF, linewidth=0.6)
    ax.scatter(x[ok & ~sig], m[ok & ~sig], s=26, facecolor=SURF, edgecolor=COL[key], zorder=3, linewidth=1.1)


def legend_below(fig, axes, ncol=5):
    h, l = [], []
    for ax in np.atleast_1d(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def kinds(R):
    return {k: v["kind"] for k, v in R["models"].items()}


def fig_budget(R, out):
    K = kinds(R)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.25, wspace=0.22)
    keys = [k for k in ORDER if k in R["models"]]
    for i, k in enumerate(keys):
        M = R["models"][k]
        off = (i - len(keys) / 2) * 0.06
        series(axes[0], k, M["budget_bulk"], K[k], off)
        if "budget_own" in M:
            series(axes[1], k, M["budget_own"], K[k], off)
        ef = M["explained_bulk"]
        axes[2].plot(range(1, len(ef) + 1), ef, LS[K[k]], color=COL[k], lw=1.8, label=LBL[k])
    e = R["era5"].get("budget")
    for ax, key in ((axes[0], "resid_bulk"), (axes[1], "resid_own")):
        ax.axhline(0, color=INK2, lw=0.8)
        if e:
            b = e[key]
            ax.axhspan(b["lo"], b["hi"], color="#c3c2b7", alpha=0.35, lw=0)
            ax.axhline(b["mean"], color=INK, lw=1.2, ls="-", label="ERA5 (analysis days)")
        ax.set_xlabel("forecast day")
        ax.set_ylabel("dM/dt − torques (Hadley)")
    if e:
        axes[2].axhline(e["explained_bulk"], color=INK, lw=1.2, label="ERA5 (analysis days)")
    axes[0].set_title("(a) budget residual, common bulk friction\nM = all own levels + mass term (mass drift removed)", fontsize=10)
    axes[1].set_title("(b) budget residual, the model's own published stress\nGFS, GEFS: turbulent + GWD; IFS: turbulent only",
                      fontsize=10)
    axes[2].set_title("(c) share of the daily AAM change explained\nby mountain + bulk friction torque", fontsize=10)
    axes[2].set_xlabel("forecast day")
    axes[2].set_ylim(-0.6, 1.0)
    fig.suptitle("Does each model's AAM change match its own torques? 00Z inits from 14 Sep 2026, global, daily means",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    legend_below(fig, axes, ncol=9)
    save(fig, out, "budget_residual")


def fig_mass(R, out):
    K = kinds(R)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.2)
    for i, k in enumerate([k for k in ORDER if k in R["models"]]):
        M = R["models"][k]
        series(axes[0], k, M["mass_drift_hPa"], K[k], (i - 4) * 0.06)
        if "dry_mass_drift_hPa" in M:
            series(axes[1], k, M["dry_mass_drift_hPa"], K[k], (i - 4) * 0.06)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("forecast day")
        ax.set_ylabel("global-mean surface pressure change (hPa)")
    axes[0].set_title("(a) total mass (all models)", fontsize=10)
    axes[1].set_title("(b) dry air: total minus water vapour (models that publish column water)", fontsize=10)
    fig.suptitle("Mass conservation: global-mean surface-pressure drift since the analysis, mean and 95 % CI over inits",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    legend_below(fig, axes, ncol=8)
    save(fig, out, "mass_drift")


def fig_drift(R, out, ref="own"):
    K = kinds(R)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.25, wspace=0.08)
    src = "drift_own" if ref == "own" else "drift_era5"
    for i, k in enumerate([k for k in ORDER if k in R["models"]]):
        M = R["models"][k]
        for ax, part in zip(axes, ("G", "S", "N")):
            series(ax, k, M[src][f"MrC11_b|{part}"], K[k], (i - 4) * 0.06)
    for ax, t in zip(axes, ("(a) global", "(b) Southern Hemisphere", "(c) Northern Hemisphere")):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("forecast day")
    axes[0].set_ylabel("relative AAM, forecast − reference (10²⁵ kg m² s⁻¹)")
    what = "its own analysis valid the same day" if ref == "own" else "ERA5 valid the same day, minus the day-0 offset"
    fig.suptitle(f"Free-forecast AAM drift (wind term, 11 common levels) against {what}", fontsize=11.5, color=INK,
                 x=0.01, ha="left")
    legend_below(fig, axes, ncol=8)
    save(fig, out, f"aam_drift_{ref}")


def fig_torques(R, out):
    keys = [k for k in ORDER if k in R["models"]]
    terms = [("Tm_b", "mountain torque, own grid"), ("Tfb_b", "bulk friction torque"), ("own", "own published stress")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.27, wspace=0.18)
    for ax, (tk, ttl) in zip(axes, terms):
        xs = []
        for i, k in enumerate(keys):
            ag = R["models"][k]["torque_bias_agg"].get(f"{tk}|G")
            if not ag:
                continue
            for j, (lead, shade) in enumerate((("d1", 0.45), ("d6_10", 1.0))):
                b = ag[lead]
                if b["mean"] is None or not np.isfinite(b["mean"]):
                    continue
                x = i + (j - 0.5) * 0.36
                ax.bar(x, b["mean"], width=0.34, color=COL[k], alpha=shade, edgecolor=SURF, linewidth=1.5)
                if b.get("lo") is not None and np.isfinite(b["lo"]):
                    ax.vlines(x, b["lo"], b["hi"], color=INK2, lw=0.9)
                if not b.get("sig"):
                    ax.text(x, b["mean"], "n.s.", ha="center", va="bottom" if b["mean"] >= 0 else "top",
                            fontsize=7, color=MUTED)
            xs.append(i)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([LBL[k].replace(" (AI)", "\n(AI)").replace(" (hybrid)", "\n(hybrid)") for k in keys],
                           fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("forecast − ERA5, same valid day (Hadley)")
    fig.suptitle("Global surface torques against ERA5: day 1 (pale) and days 6–10 (solid), mean and 95 % CI over inits",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    save(fig, out, "torque_bias")


def fig_winds(R, out):
    keys = [k for k in ORDER if k in R["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.2, wspace=0.28)
    for ax, (wk, ttl, unit) in zip(axes, (("spd10_rh|own", "10 m wind speed", "m/s"),
                                          ("u850_rh|own", "850 hPa zonal wind", "m/s"))):
        Mx = np.full((len(keys), 8), np.nan)
        Sg = np.zeros_like(Mx, bool)
        for i, k in enumerate(keys):
            for r in R["models"][k]["winds"][wk]:
                if r["lead"] == 10 or (k == "gdps" and r["lead"] == 10):
                    j = C.RHKEYS.index(r["region"])
                    Mx[i, j] = r["mean"] if r["mean"] is not None else np.nan
                    Sg[i, j] = bool(r.get("sig"))
        lim = max(0.3, np.nanmax(np.abs(Mx)) if np.isfinite(Mx).any() else 0.3)
        ax.imshow(np.zeros_like(Mx), cmap=LinearSegmentedColormap.from_list("g", ["#f0efec", "#f0efec"]), aspect="auto")
        ax.imshow(np.where(Sg, Mx, np.nan), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isfinite(Mx[i, j]):
                    col = (INK if abs(Mx[i, j]) < 0.6 * lim else "white") if Sg[i, j] else MUTED
                    ax.text(j, i, f"{Mx[i, j]:+.2f}" + ("" if Sg[i, j] else "\nn.s."), ha="center", va="center",
                            fontsize=7.5, color=col, fontweight="bold" if Sg[i, j] else "normal")
        ax.set_xticks(range(8))
        short = {"ORO": "rough\nterrain", "OCX": "ocean\n30-65°", "OCO": "other\nocean", "LND": "other\nland"}
        ax.set_xticklabels([f"{short[x.split('_')[0]]}\n{x.split('_')[1]}H" for x in C.RHKEYS], fontsize=7.5)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([LBL[k] for k in keys], fontsize=8.5)
        ax.grid(False)
        ax.set_title(f"{ttl}: day-10 forecast − own analysis ({unit})", fontsize=10)
    fig.suptitle("Low-level wind drift by region at day 10 (GDPS: day 10, its last). Coloured = significant (FDR 10 %)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    save(fig, out, "wind_drift")


def fig_mountain(R, out):
    keys = [k for k in ORDER if k in R["models"]] + (["era5"] if "ranges" in R.get("era5", {}) else [])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.17, wspace=0.18, hspace=0.42)
    for r, ax in enumerate(axes.ravel()):
        rk = C.RKEYS[r]
        for i, k in enumerate(keys):
            src = R["era5"]["ranges"] if k == "era5" else R["models"][k]["ranges"]
            if "Tm_r" not in src:
                continue
            vals = [src[t]["mean"][r] for t in ("Tm_r", "Tm05_r", "Tm15_r")]
            col = INK if k == "era5" else COL[k]
            for j, (v, a) in enumerate(zip(vals, (1.0, 0.6, 0.3))):
                ax.bar(i + (j - 1) * 0.27, v, width=0.25, color=col, alpha=a, edgecolor=SURF, linewidth=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([LBL[k].split(" ")[0] for k in keys], fontsize=7.5, rotation=45, ha="right")
        ax.set_title(RANGE_LBL[rk], fontsize=10)
        ax.grid(axis="x", visible=False)
        if r % 3 == 0:
            ax.set_ylabel("mean mountain torque (Hadley)")
    fig.suptitle("Resolved mountain torque by range, days 1–10 (ERA5: analysis days): each model's own grid (solid), "
                 "p_s and h averaged to 0.5° (mid) and 1.5° (pale)", fontsize=11.5, color=INK, x=0.01, ha="left")
    save(fig, out, "mountain_resolution")


def fig_orography(R, out):
    keys = [k for k in ORDER if k in R["models"]] + (["era5"] if "orography" in R.get("era5", {}) else [])
    ranges = C.RKEYS + ["land"]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.27)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        sc = (R["era5"] if k == "era5" else R["models"][k])["orography"]["scales"]
        if not sc:
            continue
        v = [sc.get(r, {}).get("rms_fine05", np.nan) for r in ranges]
        ax.bar(np.arange(len(ranges)) + (i - len(keys) / 2) * w, v, width=w * 0.92,
               color=INK if k == "era5" else COL[k], label=LBL[k], edgecolor=SURF, linewidth=0.6)
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels([RANGE_LBL.get(r, "all land") for r in ranges])
    ax.set_ylabel("RMS of h minus its 0.5° mean (m)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("How much sub-0.5° relief each model's own orography keeps (as published on its output grid)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=9, fontsize=9)
    save(fig, out, "orography_scales")


# WGNE Drag Project Report 1 (Zadra et al. 2013), Fig. 13, January 2012, 00-24 h: NH peak values (1e19 N m) read off
# the published figure (+-0.05): PBL over land, SGO, total physics (all surfaces).
WGNE2012 = {"pbl_land": {"IFS": 0.85, "GDPS": 0.52, "others": (0.33, 0.55)},
            "sgo": {"IFS": 0.2, "GDPS": 0.6, "others": (0.4, 1.25)},
            "total": {"IFS": 1.85, "GDPS": 1.9, "others": (1.6, 2.35)}}


def fig_wgne(R, out):
    W = R.get("wgne") or {}
    if not W:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, wspace=0.18, hspace=0.3)
    cols = {"gfs": COL["gfs"], "gefs": COL["gefs"], "ifs-ens": COL["ifs-ens"], "era5": INK}
    names = {"gfs": "GFS", "gefs": "GEFS control", "ifs-ens": "IFS (turbulent incl. TOFD only)", "era5": "ERA5"}
    for row, mon in enumerate((1, 7)):
        for col, (part, ttl) in enumerate((("pbl_land", "boundary-layer (turbulent) stress over land"),
                                           ("sgo", "subgrid-orography (GWD + blocking) stress"),
                                           ("total", "total subgrid stress, all surfaces"))):
            ax = axes[row, col]
            for m in ("gfs", "gefs", "ifs-ens", "era5"):
                d = W.get(f"{m}|{mon}")
                if not d or (part != "pbl_land" and not d["has_sgo"]):
                    continue
                ax.plot(d["lat"], d[part]["zonal"], color=cols[m], lw=1.8 if m != "era5" else 1.4,
                        label=f"{names[m]} (n = {d['n']})")
            ax.axhline(0, color=INK2, lw=0.6)
            if mon == 1 and part in WGNE2012:
                w = WGNE2012[part]
                ax.add_patch(plt.Rectangle((32, w["others"][0]), 14, w["others"][1] - w["others"][0], color="#c3c2b7",
                                           alpha=0.45, lw=0, label="2012 high-resolution models, peak range"))
                ax.plot([32, 46], [w["IFS"]] * 2, color=COL["ifs-ens"], lw=3, alpha=0.5, label="2012 IFS peak")
                ax.plot([32, 46], [w["GDPS"]] * 2, color=COL["gdps"], lw=3, alpha=0.5, label="2012 GDPS peak")
            if mon == 7 and part == "total":                       # July 2012 SH peak range (WGNE Report 1, Fig. 14)
                ax.add_patch(plt.Rectangle((-55, 2.2), 15, 0.3, color="#c3c2b7", alpha=0.45, lw=0,
                                           label="2012 high-resolution models, peak range"))
            ax.set_title(f"{ttl}\n{'January' if mon == 1 else 'July'} 2026, first 24 h", fontsize=9.5)
            ax.set_xlim(-90, 90)
            if col == 0:
                ax.set_ylabel("[τx] a³ cos²φ (10¹⁹ N m)")
            if row == 1:
                ax.set_xlabel("latitude")
    fig.suptitle("Day-1 subgrid surface stress in the WGNE Drag Project's form (Report 1, Fig. 13: Jan / Jul 2012), "
                 "2026 versions of the models that publish it", fontsize=11.5, color=INK, x=0.01, ha="left")
    legend_below(fig, axes.ravel(), ncol=4)
    save(fig, out, "wgne_day1")


def fig_levels(R, out):
    K = kinds(R)
    keys = [k for k in ORDER if k in R["models"] and R["models"][k].get("level_drift_d10")]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.22, wspace=0.08)
    for k in keys:
        rows = R["models"][k]["level_drift_d10"]
        for ax, part in zip(axes, ("S", "N")):
            rr = [r for r in rows if r["part"] == part]
            p = [r["level"] for r in rr]
            m = [r["mean"] for r in rr]
            ax.plot(m, p, LS[K[k]], color=COL[k], lw=1.6, label=LBL[k])
            ax.scatter([x for x, r in zip(m, rr) if r.get("sig")], [y for y, r in zip(p, rr) if r.get("sig")],
                       color=COL[k], s=20, zorder=3)
    for ax, t in zip(axes, ("(a) Southern Hemisphere", "(b) Northern Hemisphere")):
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("relative AAM in the layer, day 10 − own analysis (10²⁵ kg m² s⁻¹)")
    axes[0].invert_yaxis()
    axes[0].set_yscale("log")
    axes[0].set_yticks(C.C11)
    axes[0].set_yticklabels([str(x) for x in C.C11])
    axes[0].set_ylabel("pressure (hPa)")
    fig.suptitle("Where in the column the day-10 AAM drift sits (11 common levels; dots = significant, FDR 10 %)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    legend_below(fig, axes, ncol=8)
    save(fig, out, "level_drift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=".")
    a = ap.parse_args()
    R = json.loads((C.DATA / "results.json").read_text())
    out = Path(a.site) / "studies" / SLUG
    fig_budget(R, out)
    fig_mass(R, out)
    fig_drift(R, out, "own")
    fig_drift(R, out, "era5")
    fig_torques(R, out)
    fig_winds(R, out)
    fig_mountain(R, out)
    fig_orography(R, out)
    fig_wgne(R, out)
    fig_levels(R, out)


if __name__ == "__main__":
    main()
