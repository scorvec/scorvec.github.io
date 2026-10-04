#!/usr/bin/env python3
"""Figures for the 2026-09-30 SSW precursor study report (reads the s1/s1b/s2/s3 outputs; writes PNGs to
~/research/ssw_structure/figs). Research only; nothing here goes to the site."""
from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sswr_common as C  # noqa: E402

INK, MUTED, GRID, NS = "#1f2328", "#6f6b64", "#e7e5e1", "#b9b6b0"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 9.5, "axes.titlecolor": INK})
D = C.DATA
LAGS = np.arange(-90, 41)


def foot(fig, text, fontsize=7.4):
    paras = [textwrap.fill(p, 175) for p in text.split("\n")]
    fig.text(0.01, 0.005, "\n".join(paras), fontsize=fontsize, color=MUTED, va="bottom")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, z, tag, col, title, ylab="sd", xl=(-90, 40), fdr_key="fdr"):
    c, lo, hi = z[f"{tag}|comp"], z[f"{tag}|lo"], z[f"{tag}|hi"]
    sig = z[f"{tag}|{fdr_key}"].astype(bool)
    ax.fill_between(LAGS, lo, hi, color="#efedea", lw=0, label="95 % of null composites")
    ax.axhline(0, color=NS, lw=0.8); ax.axvline(0, color=NS, lw=0.8, ls=":")
    ax.plot(LAGS, c, color=NS, lw=1.4)
    ax.plot(LAGS, np.where(sig, c, np.nan), color=col, lw=2.6)
    p05 = z[f"{tag}|p"] < 0.05
    ax.plot(LAGS[p05 & ~sig], c[p05 & ~sig], ls="none", marker="o", ms=2.2, color=col, alpha=0.6)
    for a, b, lab in ((-48, -44, ""), (-31, -23, ""), (-30, -17, "")):
        pass
    ax.set_xlim(*xl); ax.set_title(title, loc="left"); ax.set_ylabel(ylab, color=MUTED)
    style(ax)


def fig_q1(z):
    rows = [("u10o", "u10", "u(60°N, 10 hPa)", S1), ("Ttr", "Ttr", "Tropical 100–50 hPa layer T, 0–15°N", S2),
            ("dTtr", "dTtr", "Its 10-day tendency (centred)", S3)]
    fig, axs = plt.subplots(3, 2, figsize=(10.5, 7.6), sharex=True, dpi=150)
    for i, (k0, k1, lab, col) in enumerate(rows):
        panel(axs[i, 0], z, f"{k0}|N0|base", col, f"{lab} — original test")
        panel(axs[i, 1], z, f"{k1}|N1|base", col, f"{lab} — eligible dates")
        for ax in axs[i]:
            if k1 == "u10":
                for a, b in ((-48, -44), (-31, -23)):
                    ax.axvspan(a, b, color=S4, alpha=0.12, lw=0)
            if k1 == "Ttr":
                ax.axvspan(-30, -17, color=S4, alpha=0.12, lw=0)
            if k1 == "dTtr":
                ax.axvspan(-43, -39, color=S4, alpha=0.12, lw=0)
    for ax in axs[-1]:
        ax.set_xlabel("days from the central date")
    fig.suptitle("Figure 1. The site composite (left) and the same composite against ELIGIBLE comparison dates (right)",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "28 MERRA-2 events, each measured from its own day −90..−61 mean (as on ssw_timing.webp). Left: the site's "
             "series and null (same calendar day ±10 d, any other winter). Right: gap-filled u series; null dates with the event's "
             "own westerly history (±15 d, Nov–Mar).\nThick colour = significant after FDR 10 % over the 131 lags; dots = p < 0.05 "
             "unadjusted only; grey band = 95 % of 3,000 null composites; amber = the windows the site figure calls significant.",
             fontsize=7.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(C.FIGS / "f1_eligible.png"); plt.close(fig)


def fig_q2(z):
    rows = [("u10|N1|anom", "u(60°N,10 hPa), absolute anomaly", S1), ("Ttr|N1|anom", "Tropical 100–50 hPa T, absolute anomaly", S2),
            ("qbo50|N1|anom", "QBO, CPC 50 hPa index (standardised)", S5), ("Ttr_noqe|N1|anom", "Tropical T with QBO and RONI regressed out", S2),
            ("Ttr|N3|base", "Tropical T, vortex-matched eligible null (base mode)", S2), ("Ttr_resid|N1|base", "Tropical T minus its vortex/heat-flux prediction (base)", S2)]
    fig, axs = plt.subplots(3, 2, figsize=(10.5, 7.6), sharex=True, dpi=150)
    for ax, (tag, lab, col) in zip(axs.ravel(), rows):
        panel(ax, z, tag, col, lab)
        ax.axvspan(-90, -61, color=NS, alpha=0.12, lw=0)
    for ax in axs[-1]:
        ax.set_xlabel("days from the central date")
    fig.suptitle("Figure 2. What the baseline hides: absolute anomalies against eligible dates, and the tropical signal controlled",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Top four panels: no per-event baseline (grey shading = the site's baseline window −90..−61). Bottom: base "
             "mode with the vortex-matched null (N3) and with the vortex-history regression removed. Thick colour = FDR 10 % over lags.",
             fontsize=7.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(C.FIGS / "f2_background.png"); plt.close(fig)


def fig_curves(F):
    keys = [("U", "Vortex strength u(60°N,10 hPa), sd"), ("HF40", "40-day mean 100 hPa heat flux, sd"),
            ("dT", "Trailing 10-day change of tropical T, sd"), ("T", "Tropical 100–50 hPa T, sd"),
            ("QBO", "QBO 50 hPa, sd (negative = easterly)"), ("ENSO", "RONI (°C, previous season)")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.4), dpi=150)
    for ax, (k, lab) in zip(axs.ravel(), keys):
        c = F["curves"][k]; b = np.array(c["bins"], float)
        mids = [(max(b[j], -2.25 if k != "ENSO" else -1.75) + min(b[j + 1], 2.25)) / 2 for j in range(len(b) - 1)]
        for w, col, off in (("1-14", S2, -0.04), ("21-42", S1, 0.04)):
            rr = c["rows"][w]
            x = [m + off for m, r in zip(mids, rr) if r]; y = [r["p"] for r in rr if r]
            lo = [r["p"] - r["lo"] for r in rr if r]; hi = [r["hi"] - r["p"] for r in rr if r]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, marker="o", ms=4, lw=1.6, capsize=2,
                        label=f"onset in {w} d" if k == "U" else None)
            ax.axhline(c["base"][w], color=col, lw=0.8, ls="--")
        ax.set_title(lab, loc="left", fontsize=8.8); ax.set_ylim(0, 0.5); style(ax)
        ax.set_ylabel("P(major SSW onset)", color=MUTED)
    axs[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 3. Forward probability of a major SSW on all eligible Nov–Mar days, 1980/81–2025/26 (5,898 days, 46 winters)",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Empirical frequency in bins of each predictor; bars = 95 % winter-block bootstrap intervals; dashed = base rate. "
             "Bins hold 51–1,854 days but far fewer independent winters (9–46), so neighbouring bins are not independent.",
             fontsize=7.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(C.FIGS / "f3_curves.png"); plt.close(fig)


def fig_skill(F):
    models = ["QE", "QE+U", "QE+T", "QE+dT", "QE+HF", "QE+HF40", "QE+U+dT", "QE+U+HF", "QE+U+T+dT+HF+HF40"]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 4.2), sharey=True, dpi=150)
    for ax, w in zip(axs, F["windows"]):
        for i, m in enumerate(models):
            r = F["skill"][w][m]["vsCLIM"]
            bsr = r["dBS"] / r["bss"] if r["bss"] != 0 else np.nan
            lo, hi = np.array(r["ci"]) / bsr
            col = S1 if (r["fdr10"] and r["bss"] > 0) else (S2 if r["fdr10"] else NS)
            ax.plot([lo, hi], [i, i], color=col, lw=2)
            ax.plot(r["bss"], i, "o", color=col, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"onset in {w} days", loc="left"); style(ax)
        ax.set_xlabel("BSS vs seasonal climatology")
    axs[0].set_yticks(range(len(models))); axs[0].set_yticklabels(models); axs[0].invert_yaxis()
    fig.suptitle("Figure 4. Out-of-sample skill (leave-one-winter-out), Brier skill score with 95 % winter-block bootstrap interval",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Blue = better than climatology after FDR 10 % across all window × model × reference tests; grey = not "
             "significant. QE = seasonal + QBO + ENSO; U vortex; T tropical T; dT its trailing 10-day change; HF heat flux; HF40 its 40-day mean.",
             fontsize=7.4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(C.FIGS / "f4_skill.png"); plt.close(fig)


NAMES = {"u10": "u(60N,10hPa)", "vT100": "heat flux 100 hPa", "jet_lat": "jet latitude", "jet_width": "jet width",
         "depth_ratio": "depth ratio u50/u10", "top_shear": "u(1 hPa) − u(10 hPa)", "u60_100": "u(60N,100hPa)",
         "qy_edge": "edge PV gradient (QG)", "qy_pole": "poleward-flank PV gradient", "bt_neg_upper": "β − u_yy < 0 share",
         "qg_neg_upper": "QG q_y < 0 share", "bt_neg_flank": "flank β − u_yy < 0 share", "n2k1": "refr. index k=1",
         "n2k2": "refr. index k=2", "n2k2_pos": "k=2 propagating share", "cen_lat": "centroid latitude", "aspect": "aspect ratio (log)",
         "area": "vortex area", "u60_50": "u(60N,50hPa)", "u60_1": "u(60N,1hPa)"}


def fig_struct_comp(z3):
    keys = ["jet_lat", "jet_width", "depth_ratio", "top_shear", "qy_edge", "qy_pole", "bt_neg_upper", "qg_neg_upper",
            "n2k2", "cen_lat", "aspect", "area"]
    fig, axs = plt.subplots(4, 3, figsize=(11, 9.2), sharex=True, dpi=150)
    for ax, k in zip(axs.ravel(), keys):
        panel(ax, z3, k, S5, NAMES[k])
        ax.set_xlim(-90, 30)
    for ax in axs[-1]:
        ax.set_xlabel("days from the central date")
    fig.suptitle("Figure 5. Vortex structure before the 28 events: absolute anomalies against eligible dates",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Standardised by day of year. Thick colour = FDR 10 % over lags; dots = p < 0.05 unadjusted only. "
             "MERRA-2 zonal means (PV gradient, instability shares, jet, depth); NCEP R1 10 hPa heights (centroid, aspect, area).",
             fontsize=7.4)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(C.FIGS / "f5_struct_comp.png"); plt.close(fig)


def fig_struct_skill(F):
    keys = list(F["struct"]["21-42"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True, dpi=150)
    for ax, w in zip(axs, ("21-42", "1-14")):
        for i, k in enumerate(keys):
            for t, col, dy in (("S_vs_U", S1, -0.18), ("UxS_vs_U", S2, 0.18)):
                r = F["struct"][w][k][t]
                bsr = r["dBS"] / r["bss"] if r["bss"] != 0 else np.nan
                lo, hi = np.array(r["ci"]) / bsr
                good = r["fdr10"] and r["bss"] > 0
                ax.plot([lo, hi], [i + dy, i + dy], color=col if good else NS, lw=1.8)
                ax.plot(r["bss"], i + dy, "o" if good else ("x" if r["fdr10"] else "o"), color=col if (good or r["fdr10"]) else NS, ms=4)
        ax.axvline(0, color=MUTED, lw=0.8); style(ax)
        ax.set_title(f"onset in {w} days: gain over seasonal + QBO + ENSO + strength", loc="left")
        ax.set_xlabel("BSS relative to the strength-only model")
        ax.set_xlim(-0.06, 0.1)
    axs[0].set_yticks(range(len(keys))); axs[0].set_yticklabels([NAMES[k] for k in keys]); axs[0].invert_yaxis()
    axs[1].plot([], [], color=S1, lw=2, label="+ structure S"); axs[1].plot([], [], color=S2, lw=2, label="+ S and strength × S")
    axs[1].legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.suptitle("Figure 6. Does structure add to strength? Leave-one-winter-out skill gain, 95 % winter-block bootstrap",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Coloured circle = significant GAIN after FDR 10 % over 16 metrics × 4 windows × 3 tests; coloured × = "
             "significant LOSS (over-fitting); grey = not significant.", fontsize=7.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(C.FIGS / "f6_struct_skill.png"); plt.close(fig)


def fig_cases():
    M = pd.read_pickle(C.DATA / "metrics_daily.pkl")
    u = C.u10()
    Z = {"U": C.std_anom(u), "bt": C.std_anom(M.bt_neg_upper), "qp": C.std_anom(M.qy_pole),
         "asp": C.std_anom(np.log(M.aspect.where((M.aspect >= 1) & (M.aspect < 20)))), "ts": C.std_anom(M.top_shear)}
    E = C.events()
    cases = [("2010/11", 2010, "no midwinter SSW"), ("2019/20", 2019, "no SSW, record-strong"),
             ("2008/09", 2008, "split 24 Jan 2009"), ("2012/13", 2012, "split 7 Jan 2013"),
             ("1988/89", 1988, "split 21 Feb 1989"), ("2015/16", 2015, "no midwinter SSW")]
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.2), dpi=150, sharey=True)
    for ax, (lab, y, note) in zip(axs.ravel(), cases):
        a, b = pd.Timestamp(y, 11, 1), pd.Timestamp(y + 1, 3, 31)
        for k, col, nm in (("U", S1, "strength u10"), ("bt", S2, "β − u_yy < 0 share"), ("qp", S3, "poleward PV gradient"),
                           ("asp", S5, "aspect ratio")):
            s = Z[k][a:b].rolling(5, center=True, min_periods=3).mean()
            ax.plot(s.index, s.values, color=col, lw=1.6 if k == "U" else 1.1, label=nm)
        for d in E.date[(E.date >= a) & (E.date <= b)]:
            ax.axvline(d, color=INK, lw=1, ls="--")
        ax.axhline(0, color=NS, lw=0.8); ax.axhline(1, color=NS, lw=0.6, ls=":")
        ax.set_title(f"{lab}: {note}", loc="left"); style(ax); ax.set_ylim(-3.5, 3.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axs[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Figure 7. Case studies: strong vortices with and without a following SSW (5-day means, standardised)",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    foot(fig, "Dashed = central date of a major SSW. Descriptive only (single winters need no test); the spell-level "
             "test across all 29 strong-vortex spells is in Table 4.", fontsize=7.4)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(C.FIGS / "f7_cases.png"); plt.close(fig)


def main():
    z = np.load(D / "s1_composites.npz")
    z3 = np.load(D / "s3_structure.npz")
    F = json.loads((D / "s2_forward.json").read_text())
    fig_q1(z); fig_q2(z); fig_curves(F); fig_skill(F); fig_struct_comp(z3); fig_struct_skill(F); fig_cases()
    print("figures in", C.FIGS)


if __name__ == "__main__":
    main()
