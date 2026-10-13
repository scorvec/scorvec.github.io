#!/usr/bin/env python3
"""Figures for the El Nino mountain-torque study (torque_enso_study.py / torque_budget_2026.py outputs).

    python scripts/mjo/src/torque_enso_figs.py --out <dir with the study outputs>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401  (before anything that may pull eccodes; repo rule)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import torque_enso_study as S                                           # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#6f6b64", "#e1e0d9", "#fcfcfb"
C_MTN, C_FRIC, C_GWD, C_NET, C_RES = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DIV = LinearSegmentedColormap.from_list("bgr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                                                "#f4b3b2", "#e34948", "#8a1c1b"])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
SE = ["DJF", "MAM", "JJA", "SON"]


def fig_q1(out: Path):
    comp = pd.read_csv(out / "q1_compare.csv")
    cm = pd.read_csv(out / "q1_cmip6_summary.csv") if (out / "q1_cmip6_summary.csv").exists() else None
    sh = json.loads((out / "q1_shift_test.json").read_text())
    panels = [("lp_mean", "EN-NEU", "standing (>=30 d) torque shift", "Hadley"),
              ("n_pos", "EN-NEU", "positive EVENTS per season (<30 d, >1 sd)", "events"),
              ("amp_pos", "EN-NEU", "positive event amplitude", "Hadley"),
              ("nx_pos", "EN-NEU", "total-anomaly positive events (>1 sd)", "events"),
              ("excess", None, "total-anomaly events beyond\nwhat the mean shift alone gives", "events")]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.4), constrained_layout=True)
    for ax, (mt, cmp_, ttl, unit) in zip(axes, panels):
        M = np.full((len(S.RANGES), 4), np.nan); SG = np.zeros_like(M, bool)
        for i, r in enumerate(S.RANGES):
            for j, se in enumerate(SE):
                if mt == "excess":
                    d = sh.get(r, {}).get(se)
                    if d:
                        M[i, j] = d["excess"]; SG[i, j] = d.get("fdr", False)
                else:
                    c = comp[(comp.range == r) & (comp.season == se) & (comp.cmp == cmp_) & (comp.metric == mt)]
                    if len(c):
                        M[i, j] = c["diff"].iloc[0]; SG[i, j] = bool(c["sig"].iloc[0])
        lim = np.nanmax(np.abs(M)) or 1
        shown = np.where(SG, M, np.nan)
        ax.imshow(np.zeros_like(M), cmap=LinearSegmentedColormap.from_list("g", ["#f0efec", "#f0efec"]),
                  aspect="auto")
        im = ax.imshow(shown, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    txt = f"{M[i, j]:+.1f}" if SG[i, j] else f"{M[i, j]:+.1f}\nn.s."
                    col = (INK if abs(M[i, j]) < 0.6 * lim else "white") if SG[i, j] else MUTED
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=col,
                            fontweight="bold" if SG[i, j] else "normal")
                    if cm is not None and mt in ("lp_mean", "n_pos", "nx_pos", "amp_pos"):
                        q = cm[(cm.metric == mt) & (cm.range == S.RANGES[i]) & (cm.season == SE[j])]
                        if len(q) and bool(q.robust.iloc[0]):
                            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                                       ec=INK, lw=1.6))
        ax.set_xticks(range(4)); ax.set_xticklabels(SE)
        ax.set_yticks(range(len(S.RANGES))); ax.set_yticklabels([S.RNAME[r] for r in S.RANGES], fontsize=8.5)
        ax.grid(False)
        ax.set_title(f"{ttl}\nEl Nino minus neutral ({unit})", fontsize=9.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("ERA5 1979-2025 mountain torque by range: El Nino minus neutral season-years.  Coloured = FDR 10% "
                 "significant (permutation of season-years); black box = robust in CMIP6 (19 models, FDR + >=80% sign "
                 "agreement)", fontsize=10.5, color=INK)
    fig.savefig(out / "fig_q1_events_vs_shift.png", dpi=130)
    plt.close(fig)


def fig_q2_lag(out: Path):
    lr = pd.read_csv(out / "q2_era5_lagreg.csv")
    cs = pd.read_csv(out / "q2_cmip6_slopes_summary.csv")
    r = json.loads((out / "results.json").read_text())
    tend = r["q2_era5"]["tendency_column"]
    ser = [("mtn_glob", "mountain", C_MTN), ("fric_glob", "friction (turbulent)", C_FRIC),
           ("gwd_glob", "gravity-wave drag", C_GWD), ("T_glob", "sum of torques", C_NET)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, constrained_layout=True)
    ax = axes[0]
    for k, lab, col in ser + [(tend, "dM/dt (motion + mass)", INK), ("R_glob", "residual dM/dt - torques", C_RES)]:
        g = lr[lr.series == k].sort_values("lag")
        ax.plot(g.lag, g.slope, color=col, lw=2, label=lab)
        ax.scatter(g.lag[g.sig], g.slope[g.sig], color=col, s=26, zorder=3)
        ax.scatter(g.lag[~g.sig], g.slope[~g.sig], facecolor=SURF, edgecolor=col, s=22, zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_title("ERA5 1979-2025 (filled = FDR-significant, 24-month block bootstrap)", fontsize=10)
    ax.set_xlabel("lag (months; torque after Nino-3.4 > 0)"); ax.set_ylabel("Hadley per K of Nino-3.4")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax = axes[1]
    for k, lab, col in [("mtn_glob", "mountain", C_MTN), ("fric_glob", "surface stress (tauu)", C_FRIC),
                        ("gwd_glob", "orographic GWD (13 models)", C_GWD), ("T_mf_glob", "mountain + tauu", C_NET),
                        ("dmdt_glob", "dM/dt (motion term)", INK)]:
        g = cs[cs.series == k].sort_values("lag")
        if g.empty:
            continue
        ax.fill_between(g.lag, g.q25, g.q75, color=col, alpha=0.16, lw=0)
        ax.plot(g.lag, g["median"], color=col, lw=2, label=lab)
        ax.scatter(g.lag[g.robust], g["median"][g.robust], color=col, s=26, zorder=3)
        ax.scatter(g.lag[~g.robust], g["median"][~g.robust], facecolor=SURF, edgecolor=col, s=22, zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=1)
    n = r["q2_cmip6"]["n_models"]
    ax.set_title(f"CMIP6 historical 1950-2014, {n} models: median, IQR band (filled = robust)", fontsize=10)
    ax.set_xlabel("lag (months)")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle("Regression of the global AAM budget terms on Nino-3.4 (detrended monthly anomalies)",
                 fontsize=11.5, color=INK)
    fig.savefig(out / "fig_q2_lagreg.png", dpi=130)
    plt.close(fig)


def fig_q2_contrib(out: Path):
    r = json.loads((out / "results.json").read_text())
    ce = r["q2_era5"]["contributions_lags_m6_m1"]
    cs = pd.read_csv(out / "q2_cmip6_contrib_summary.csv").set_index("series")
    tend = r["q2_era5"]["tendency_column"]
    rows = [("AAM tendency (dM/dt)", tend, "dmdt_glob", INK), ("sum of torques", "T_glob", "T_mf_glob", C_NET),
            ("mountain, global", "mtn_glob", "mtn_glob", C_MTN), ("friction, global", "fric_glob", "fric_glob", C_FRIC),
            ("GWD, global", "gwd_glob", "gwd_glob", C_GWD), ("ERA5 residual", "R_glob", None, C_RES),
            ("mtn Andes", "mtn5_AND", "mtn_AND", C_MTN), ("mtn Rockies", "mtn5_ROC", "mtn_ROC", C_MTN),
            ("mtn Himalaya/Tibet", "mtn5_HIM", "mtn_HIM", C_MTN), ("mtn Greenland", "mtn5_GRL", "mtn_GRL", C_MTN),
            ("mtn Antarctica", "mtn5_ANT", "mtn_ANT", C_MTN), ("mtn Alps", "mtn5_ALP", "mtn_ALP", C_MTN),
            ("mtn 30S-0", "mtn_b3000s", "mtn_b3000s", C_MTN), ("mtn 0-30N", "mtn_b0030n", "mtn_b0030n", C_MTN),
            ("mtn 30-60N", "mtn_b3060n", "mtn_b3060n", C_MTN), ("fric 30S-0", "fric_b3000s", "fric_b3000s", C_FRIC),
            ("fric 0-30N", "fric_b0030n", "fric_b0030n", C_FRIC), ("fric 30-60N", "fric_b3060n", "fric_b3060n", C_FRIC),
            ("fric 60-30S", "fric_b6030s", "fric_b6030s", C_FRIC)]
    fig, ax = plt.subplots(figsize=(9.5, 7.6), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ke, kc, col) in zip(y, rows):
        e = ce.get(ke)
        if e:
            sig = e.get("fdr", False)
            ax.barh(yi + 0.17, e["est"], height=0.32, color=col if sig else "#d9d8d2", edgecolor=col, lw=1)
            ax.plot([e["lo"], e["hi"]], [yi + 0.17] * 2, color=INK2, lw=1)
        if kc and kc in cs.index:
            c = cs.loc[kc]
            ax.plot([c.q25, c.q75], [yi - 0.17] * 2, color=INK, lw=2.2)
            ax.scatter([c["median"]], [yi - 0.17], marker="D", s=34, color=col if c.robust else SURF,
                       edgecolor=INK, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r_[0] for r_ in rows], fontsize=9)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("AAM supplied in the 6 months before the Nino-3.4 anomaly (1e25 kg m2 s-1 per K)")
    ax.set_title("Who builds the El Nino AAM rise?  Bars: ERA5 1979-2025 (95% block-bootstrap CI; grey = n.s. after FDR)\n"
                 "Diamonds: CMIP6 median with IQR (filled = robust: FDR across models + >=80% sign agreement)",
                 fontsize=9.8)
    fig.savefig(out / "fig_q2_contributions.png", dpi=130)
    plt.close(fig)


def fig_2026(out: Path):
    b = json.loads((out / "budget2026.json").read_text())
    g = pd.DataFrame(b["hourly_residual_by_end_hour"])
    sp = pd.read_csv(out / "budget2026_spectrum.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.bar(g["time"], g["mean"], color=[C_RES if h in (10, 22) else C_MTN for h in g["time"]], width=0.75)
    ax.errorbar(g["time"], g["mean"], yerr=g["std"], fmt="none", ecolor=INK2, lw=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("end hour of the 1-h interval (UTC)"); ax.set_ylabel("residual dM/dt - torques (Hadley)")
    wt = b["window_test"]
    ax.set_title(f"ERA5 hourly 2026: inside 4D-Var windows {wt['inside_mean']:+.1f} +/- {wt['inside_sd']:.1f} Hadley;\n"
                 f"09->10 and 21->22 UTC (window starts) {wt['across_mean']:+.0f} +/- {wt['across_sd']:.0f}", fontsize=9.5)
    ax = axes[1]
    m = sp.cpd > 0
    ax.loglog(sp.cpd[m], sp.psd_dMdt[m], color=INK, lw=1.4, label="dM/dt")
    ax.loglog(sp.cpd[m], sp.psd_T[m], color=C_NET, lw=1.4, label="sum of torques")
    ax.loglog(sp.cpd[m], sp.psd_R[m], color=C_RES, lw=1.6, label="residual")
    for f in (1, 2, 3, 4):
        ax.axvline(f, color=GRID, lw=0.8)
    ax.set_xlabel("frequency (cycles per day)"); ax.set_ylabel("PSD (Hadley^2 per cpd)")
    ax.set_title("Hourly 2026 spectra: the residual is lines at 2, 4, 6... cpd (twice-daily increments)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    att = b["attribution"]
    names = list(att)
    v = [att[k]["v2026"] for k in names]; vd = [att[k]["v2026_detrended"] for k in names]
    hs = [att[k]["hist_sd"] for k in names]
    yy = np.arange(len(names))[::-1]
    cols = [C_MTN if "mountain" in k or "Andes" in k or "ranges" in k else (C_GWD if k == "GWD" else C_FRIC) for k in names]
    ax.barh(yy + 0.17, v, height=0.32, color=cols)
    ax.barh(yy - 0.17, vd, height=0.32, color=cols, alpha=0.45)
    ax.errorbar(np.zeros(len(names)), yy, xerr=hs, fmt="none", ecolor=INK2, lw=6, alpha=0.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, label="2026 anomaly vs 1991-2020 clim (upper bar)"),
                       Patch(color=INK2, alpha=0.45, label="same, ERA5 1979-2025 trend removed (lower)"),
                       Patch(color=INK2, alpha=0.18, label="+/-1 sd of the window, 1979-2025")],
              fontsize=7.5, frameon=False, loc="lower right")
    dM = b["window_variants"]["hourly daily-means, motion+mass, anomaly vs 4x clim"]
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("integral Apr 15 - Sep 19 2026 (1e25 kg m2 s-1)")
    ax.set_title(f"2026 rise: AAM (motion+mass) {dM['dM']:+.1f}; torques {dM['torque']:+.1f}; residual {dM['residual']:+.1f}\n"
                 f"(the same window's residual, 1979-2025: {b['history_window_residual']['mean']:+.1f} +/- "
                 f"{b['history_window_residual']['sd']:.1f})", fontsize=9.5)
    fig.suptitle("ERA5 2026 AAM budget from HOURLY fields (torque_1h_2026): total AAM = motion + mass term",
                 fontsize=11.5, color=INK)
    fig.savefig(out / "fig_2026_budget.png", dpi=130)
    plt.close(fig)


def fig_q3_maps(out: Path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    z = np.load(out / "q3_z500_composites.npz")
    cases = ["HIM_DJF", "ROC_DJF", "AND_DJF"]
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(15, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(len(cases), 3)
    lev = np.linspace(-60, 60, 13)
    for i, c in enumerate(cases):
        lat = z[f"{c}__lat"]; lon = z[f"{c}__lon"]
        lon2 = np.where(lon > 180, lon - 360, lon) if c != "ROC_DJF" else lon
        for j, (k, ttl) in enumerate([("all", "all positive events"), ("en", "El Nino seasons"),
                                       ("diff", "El Nino minus neutral")]):
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree(central_longitude=float(np.mean([lon.min(), lon.max()]))))
            F = z[f"{c}__{k}"]; SG = z[f"{c}__sig_{k}"] if f"{c}__sig_{k}" in z else z[f"{c}__sig_all"]
            Fm = np.where(SG, F, np.nan)
            ax.contourf(lon, lat, Fm, levels=lev, cmap=DIV, extend="both", transform=ccrs.PlateCarree())
            ax.contour(lon, lat, F, levels=[l for l in lev if l != 0], colors=INK2, linewidths=0.5,
                       transform=ccrs.PlateCarree())
            ax.coastlines(lw=0.6, color=INK2)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            n = {"all": z[f"{c}__n_all"], "en": z[f"{c}__n_en"], "diff": f"{z[f'{c}__n_en']} vs {z[f'{c}__n_neu']}"}[k]
            frac = float(np.mean(SG))
            ax.set_title(f"{c.replace('_', ' ')}: {ttl} (n={n})\nshaded = FDR 10% ({frac:.0%} of cells)" if frac > 0
                         else f"{c.replace('_', ' ')}: {ttl} (n={n})\nno significant cells (FDR 10%)", fontsize=9)
    fig.suptitle("ERA5 z500 (<30-day part, m) averaged 3-1 days BEFORE positive mountain-torque events; contours = all "
                 "values (10 m), shading only where significant", fontsize=11, color=INK)
    fig.savefig(out / "fig_q3_precursor_z500.png", dpi=120)
    plt.close(fig)


def fig_q3_leadlag(out: Path):
    ll = pd.read_csv(out / "q3_leadlag.csv")
    cases = [("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA")]
    fig, axes = plt.subplots(1, len(cases), figsize=(16, 4.0), sharey=True, constrained_layout=True)
    ser = [("dMdt", "global dM/dt", INK), ("M", "global AAM", C_NET), ("u200jet", "subtropical u200 (zonal mean)", C_FRIC)]
    for ax, (r, se) in zip(axes, cases):
        for k, lab, col in ser:
            g = ll[(ll.range == r) & (ll.season == se) & (ll.series == k)].sort_values("lag")
            ax.plot(g.lag, g.r, color=col, lw=2, label=lab)
            ax.scatter(g.lag[g.sig], g.r[g.sig], color=col, s=16, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=1); ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_title(f"{S.RNAME[r]} {se}", fontsize=10)
        ax.set_xlabel("lag (days; series AFTER the torque > 0)")
    axes[0].set_ylabel("correlation (<30-day parts)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("ERA5 lead-lag of each range's high-pass mountain torque (dots = FDR-significant, Bretherton N_eff)",
                 fontsize=11, color=INK)
    fig.savefig(out / "fig_q3_leadlag.png", dpi=130)
    plt.close(fig)


def fig_slp(out: Path):
    import cartopy.crs as ccrs
    z = np.load(out / "slp_enso_maps.npz")
    lat, lon = z["lat"], z["lon"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.4), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)),
                             constrained_layout=True)
    lev = np.linspace(-3, 3, 13)
    for ax, se in zip(axes, ["DJF", "MAM"]):
        F = z[f"{se}__slope"]; SG = z[f"{se}__sig_slope"]
        ax.contourf(lon, lat, np.where(SG, F, np.nan), levels=lev, cmap=DIV, extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, F, levels=[l for l in lev if abs(l) > 1e-9], colors=INK2, linewidths=0.4,
                   transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color=INK2)
        for r, (lo0, lo1, la0, la1) in {"HIM": (70, 105, 25, 45), "ROC": (230, 260, 30, 60), "AND": (280, 295, -55, 10)}.items():
            ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], color=INK, lw=1.2, transform=ccrs.PlateCarree())
        ax.set_title(f"{se}: seasonal-mean SLP regressed on the ONI, ERA5 1991-2020 (hPa per unit ONI); "
                     f"shaded = FDR 10% ({float(np.mean(SG)):.0%} of cells); boxes = site torque ranges", fontsize=9.5)
    fig.savefig(out / "fig_slp_enso.png", dpi=120)
    plt.close(fig)


# ============================================================================ site figures (phone-width layouts)
DATA = S.REF / "torque_enso_study"                       # the committed study tables
WEBP = dict(dpi=110, pil_kwargs={"quality": 88}, bbox_inches="tight", pad_inches=0.06)
SITE_RC = {"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
           "legend.fontsize": 9.5}
SHORT = {"HIM": "Himalaya/Tibet", "ROC": "Rockies", "AND": "Andes", "GRL": "Greenland", "ANT": "Antarctica", "ALP": "Alps"}


def f1(x):
    t = f"{x:+.1f}"
    return "0.0" if t in ("+0.0", "-0.0") else t.replace("-", "\u2212")


def _grid_panel(ax, M, SG, title, unit, robust=None, fmt=None):
    lim = np.nanmax(np.abs(M)) or 1
    ax.imshow(np.zeros_like(M), cmap=LinearSegmentedColormap.from_list("g", ["#f0efec", "#f0efec"]), aspect="auto")
    ax.imshow(np.where(SG, M, np.nan), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            if SG[i, j]:
                col = INK if abs(M[i, j]) < 0.6 * lim else "white"
                ax.text(j, i, f1(M[i, j]), ha="center", va="center", fontsize=11, color=col, fontweight="bold")
            else:
                ax.text(j, i - 0.13, f1(M[i, j]), ha="center", va="center", fontsize=9.5, color=MUTED)
                ax.text(j, i + 0.22, "n.s.", ha="center", va="center", fontsize=8.5, color=MUTED)
            if robust is not None and robust[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=INK, lw=1.8))
    ax.set_xticks(range(4)); ax.set_xticklabels(SE)
    ax.set_yticks(range(len(S.RANGES))); ax.set_yticklabels([SHORT[r] for r in S.RANGES])
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{title} ({unit})", loc="left", fontsize=11.5)


def _cells(comp, cm, metric, cmp_="EN-NEU"):
    M = np.full((len(S.RANGES), 4), np.nan); SG = np.zeros_like(M, bool); RB = np.zeros_like(M, bool)
    for i, r in enumerate(S.RANGES):
        for j, se in enumerate(SE):
            c = comp[(comp.range == r) & (comp.season == se) & (comp.cmp == cmp_) & (comp.metric == metric)]
            if len(c):
                M[i, j] = c["diff"].iloc[0]; SG[i, j] = bool(c["sig"].iloc[0])
            if cm is not None:
                q = cm[(cm.metric == metric) & (cm.range == r) & (cm.season == se)]
                RB[i, j] = bool(len(q) and q.robust.iloc[0])
    return M, SG, RB


def site_q1(data: Path, dest: Path):
    comp = pd.read_csv(data / "q1_compare.csv")
    cm = pd.read_csv(data / "q1_cmip6_summary.csv")
    sh = json.loads((data / "q1_shift_test.json").read_text())
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 10.4), constrained_layout=True)
        for ax, (mt, ttl, unit) in zip(axes, [("lp_mean", "Standing shift: the 30-day and longer part", "Hadley"),
                                              ("n_pos", "Synoptic events per season, peaks above +1 sd", "number"),
                                              ("amp_pos", "Amplitude of those events", "Hadley")]):
            M, SG, RB = _cells(comp, cm, mt)
            _grid_panel(ax, M, SG, ttl, unit, RB)
        fig.suptitle("ERA5 1979–2025 range torque: El Niño minus neutral", fontsize=12.5,
                     color=INK, x=0.02, ha="left")
        fig.savefig(dest / "q1_shift_vs_events.webp", **WEBP)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 7.0), constrained_layout=True)
        M, SG, RB = _cells(comp, cm, "nx_pos")
        _grid_panel(axes[0], M, SG, "Events on the total anomaly (peaks above +1 sd)", "events", RB)
        E = np.full((len(S.RANGES), 4), np.nan); ESG = np.zeros_like(E, bool)
        for i, r in enumerate(S.RANGES):
            for j, se in enumerate(SE):
                d = sh.get(r, {}).get(se)
                if d:
                    E[i, j] = d["excess"]; ESG[i, j] = bool(d.get("fdr", False))
        _grid_panel(axes[1], E, ESG, "Those events beyond the pure-shift prediction", "events")
        fig.suptitle("ERA5 1979–2025: threshold events and the shift test", fontsize=12.5, color=INK,
                     x=0.02, ha="left")
        fig.savefig(dest / "q1_threshold_events.webp", **WEBP)
        plt.close(fig)


def site_slp(data: Path, dest: Path):
    import cartopy.crs as ccrs
    z = np.load(data / "slp_enso_maps.npz")
    lat, lon = z["lat"], z["lon"]
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=200)),
                                 constrained_layout=True)
        lev = np.linspace(-3, 3, 13)
        for ax, se in zip(axes, ["DJF", "MAM"]):
            F = z[f"{se}__slope"]; SG = z[f"{se}__sig_slope"]
            cf = ax.contourf(lon, lat, np.where(SG, F, np.nan), levels=lev, cmap=DIV, extend="both", transform=ccrs.PlateCarree())
            ax.contour(lon, lat, F, levels=[l for l in lev if abs(l) > 1e-9], colors=INK2, linewidths=0.35,
                       transform=ccrs.PlateCarree())
            ax.coastlines(lw=0.5, color=INK2)
            ax.set_extent([60, 345, -70, 75], crs=ccrs.PlateCarree())
            for r, (lo0, lo1, la0, la1) in {"HIM": (70, 105, 25, 45), "ROC": (230, 260, 30, 60), "AND": (280, 295, -55, 10)}.items():
                ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], color=INK, lw=1.3, transform=ccrs.PlateCarree())
                ax.text(lo1 + 2, la1 - 3, SHORT[r], fontsize=9, color=INK, transform=ccrs.PlateCarree())
            ax.set_title(f"{se}: shaded where FDR 10% significant ({float(np.mean(SG)):.0%} of cells)", loc="left", fontsize=11)
        cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40)
        cb.set_label("hPa per unit ONI")
        fig.suptitle("ERA5 1991–2020 seasonal-mean sea-level pressure regressed on the ONI", fontsize=12.5, color=INK,
                     x=0.02, ha="left")
        fig.savefig(dest / "slp_on_oni.webp", **WEBP)
        plt.close(fig)


def site_q2(data: Path, dest: Path):
    lr = pd.read_csv(data / "q2_era5_lagreg.csv")
    cs = pd.read_csv(data / "q2_cmip6_slopes_summary.csv")
    r = json.loads((data / "results.json").read_text())
    tend = r["q2_era5"]["tendency_column"]
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.6), sharex=True, sharey=True, constrained_layout=True)
        ax = axes[0]
        for k, lab, col in [("mtn_glob", "mountain", C_MTN), ("fric_glob", "friction", C_FRIC), ("gwd_glob", "gravity-wave drag", C_GWD),
                            ("T_glob", "sum of the torques", C_NET), (tend, "dM/dt", INK), ("R_glob", "dM/dt minus torques", C_RES)]:
            g = lr[lr.series == k].sort_values("lag")
            ax.plot(g.lag, g.slope, color=col, lw=2.2, label=lab)
            ax.scatter(g.lag[g.sig], g.slope[g.sig], color=col, s=24, zorder=3)
            ax.scatter(g.lag[~g.sig], g.slope[~g.sig], facecolor=SURF, edgecolor=col, s=20, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title("ERA5 1979–2025 (filled dots: FDR-significant)", loc="left")
        ax.set_ylabel("Hadley per K of Niño-3.4")
        ax = axes[1]
        for k, lab, col in [("mtn_glob", "mountain", C_MTN), ("fric_glob", "surface stress (tauu)", C_FRIC),
                            ("gwd_glob", "orographic GWD (13 models)", C_GWD), ("T_mf_glob", "mountain + tauu", C_NET),
                            ("dmdt_glob", "dM/dt", INK)]:
            g = cs[cs.series == k].sort_values("lag")
            ax.fill_between(g.lag, g.q25, g.q75, color=col, alpha=0.16, lw=0)
            ax.plot(g.lag, g["median"], color=col, lw=2.2, label=lab)
            ax.scatter(g.lag[g.robust], g["median"][g.robust], color=col, s=24, zorder=3)
            ax.scatter(g.lag[~g.robust], g["median"][~g.robust], facecolor=SURF, edgecolor=col, s=20, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(f"CMIP6 historical 1950–2014, {r['q2_cmip6']['n_models']} models: median and IQR (filled: robust)",
                     loc="left")
        ax.set_ylabel("Hadley per K of Niño-3.4")
        ax.set_xlabel("lag in months (positive: the torque comes after Niño-3.4)")
        ax.set_xticks([-6, -3, 0, 3, 6, 9, 12])
        ngw = r["q2_cmip6"].get("n_models_with_gwd", 13)
        hs = [Line2D([], [], color=c_, lw=2.2, label=l_) for c_, l_ in (
            (C_MTN, "mountain"), (C_FRIC, "friction (CMIP6: surface stress tauu)"),
            (C_GWD, f"gravity-wave drag (CMIP6: orographic, {ngw} models)"), (C_NET, "sum of the torques (CMIP6: mountain + tauu)"),
            (INK, "dM/dt"), (C_RES, "dM/dt minus torques (ERA5 only)"))]
        fig.legend(handles=hs, loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
        fig.suptitle("Global AAM budget terms regressed on Niño-3.4", fontsize=12.5, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "q2_lag_regression.webp", **WEBP)
        plt.close(fig)
        # contributions
        ce = r["q2_era5"]["contributions_lags_m6_m1"]
        csum = pd.read_csv(data / "q2_cmip6_contrib_summary.csv").set_index("series")
        rows = [("dM/dt", tend, "dmdt_glob", INK), ("sum of torques", "T_glob", "T_mf_glob", C_NET),
                ("mountain", "mtn_glob", "mtn_glob", C_MTN), ("friction", "fric_glob", "fric_glob", C_FRIC),
                ("gravity-wave drag", "gwd_glob", "gwd_glob", C_GWD), ("ERA5 residual", "R_glob", None, C_RES),
                ("Andes", "mtn5_AND", "mtn_AND", C_MTN), ("Rockies", "mtn5_ROC", "mtn_ROC", C_MTN),
                ("Himalaya/Tibet", "mtn5_HIM", "mtn_HIM", C_MTN), ("Greenland", "mtn5_GRL", "mtn_GRL", C_MTN),
                ("Antarctica", "mtn5_ANT", "mtn_ANT", C_MTN), ("Alps", "mtn5_ALP", "mtn_ALP", C_MTN),
                ("mountain 30S–0", "mtn_b3000s", "mtn_b3000s", C_MTN), ("mountain 0–30N", "mtn_b0030n", "mtn_b0030n", C_MTN),
                ("mountain 30–60N", "mtn_b3060n", "mtn_b3060n", C_MTN), ("friction 30S–0", "fric_b3000s", "fric_b3000s", C_FRIC),
                ("friction 0–30N", "fric_b0030n", "fric_b0030n", C_FRIC), ("friction 30–60N", "fric_b3060n", "fric_b3060n", C_FRIC),
                ("friction 30–60S", "fric_b6030s", "fric_b6030s", C_FRIC)]
        fig, ax = plt.subplots(figsize=(6.8, 8.4), constrained_layout=True)
        y = np.arange(len(rows))[::-1].astype(float)
        for yi, (lab, ke, kc, col) in zip(y, rows):
            e = ce.get(ke)
            if e:
                sig = e.get("fdr", False)
                ax.barh(yi + 0.18, e["est"], height=0.34, color=col if sig else "#e4e3dd", edgecolor=col, lw=1)
                ax.plot([e["lo"], e["hi"]], [yi + 0.18] * 2, color=INK2, lw=1)
            if kc and kc in csum.index:
                c = csum.loc[kc]
                ax.plot([c.q25, c.q75], [yi - 0.18] * 2, color=INK, lw=2.2)
                ax.scatter([c["median"]], [yi - 0.18], marker="D", s=36, color=col if c.robust else SURF, edgecolor=INK, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([r_[0] for r_ in rows])
        for yy in (12.5, 6.5):
            ax.axhline(yy, color="#c3c2b7", lw=0.8)
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel("AAM supplied in the six months before Niño-3.4 (10²⁵ kg m² s⁻¹ per K)")
        ax.set_title("Bars: ERA5 1979–2025 with 95% CI (pale = not significant)\n"
                     "Diamonds: CMIP6 median with IQR (filled = robust)", loc="left", fontsize=10.5)
        fig.suptitle("Where the El Niño AAM rise comes from", fontsize=12.5, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "q2_contributions.webp", **WEBP)
        plt.close(fig)


def site_q3(data: Path, dest: Path):
    import cartopy.crs as ccrs
    ll = pd.read_csv(data / "q3_leadlag.csv")
    cases = [("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA")]
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4), sharex=True, sharey=True, constrained_layout=True)
        for ax, (r, se) in zip(axes.ravel(), cases):
            for k, lab, col in [("dMdt", "global dM/dt", INK), ("M", "global AAM", C_NET),
                                ("u200jet", "subtropical u200", C_FRIC)]:
                g = ll[(ll.range == r) & (ll.season == se) & (ll.series == k)].sort_values("lag")
                ax.plot(g.lag, g.r, color=col, lw=2, label=lab)
                ax.scatter(g.lag[g.sig], g.r[g.sig], color=col, s=12, zorder=3)
            ax.axhline(0, color="#c3c2b7", lw=1); ax.axvline(0, color="#c3c2b7", lw=1)
            ax.set_title(f"{SHORT[r]}, {se}", loc="left")
        for ax in axes[1]:
            ax.set_xlabel("lag (days after the torque)")
        for ax in axes[:, 0]:
            ax.set_ylabel("correlation")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
        fig.suptitle("ERA5 1979–2025: <30-day mountain torque against AAM and the jet", fontsize=12.5, color=INK,
                     x=0.02, ha="left")
        fig.savefig(dest / "q3_lead_lag.webp", **WEBP)
        plt.close(fig)
        z = np.load(data / "q3_z500_composites.npz")
        cz = [("HIM_DJF", "Himalaya/Tibet, DJF"), ("ROC_DJF", "Rockies, DJF"), ("AND_DJF", "Andes, DJF")]
        asp = []
        for c, _ in cz:
            la_, lo_ = z[f"{c}__lat"], z[f"{c}__lon"]
            asp.append((la_.max() - la_.min()) / (lo_.max() - lo_.min()))
        mapw = 3.3; gap = 0.15; left = 0.12; tit = 0.52; top = 0.45; cbh = 1.0
        Wf = left + 2 * mapw + gap + 0.12
        Hf = top + sum(tit + a_ * mapw for a_ in asp) + cbh
        fig = plt.figure(figsize=(Wf, Hf))
        lev = np.linspace(-60, 60, 13)
        cf = None
        ycur = Hf - top
        for i, (c, nm) in enumerate(cz):
            lat = z[f"{c}__lat"]; lon = z[f"{c}__lon"]
            hmap = asp[i] * mapw
            ycur -= tit + hmap
            for j, (k, ttl) in enumerate([("all", "all positive events"), ("diff", "El Niño minus neutral")]):
                x0 = left + j * (mapw + gap)
                ax = fig.add_axes([x0 / Wf, ycur / Hf, mapw / Wf, hmap / Hf],
                                  projection=ccrs.PlateCarree(central_longitude=float(np.mean([lon.min(), lon.max()]))))
                F = z[f"{c}__{k}"]; SG = z[f"{c}__sig_{k}"]
                cf = ax.contourf(lon, lat, np.where(SG, F, np.nan), levels=lev, cmap=DIV, extend="both", transform=ccrs.PlateCarree())
                ax.contour(lon, lat, F, levels=[l for l in lev if l != 0], colors=INK2, linewidths=0.45, transform=ccrs.PlateCarree())
                ax.coastlines(lw=0.5, color=INK2)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                n = z[f"{c}__n_all"] if k == "all" else f"{z[f'{c}__n_en']} v {z[f'{c}__n_neu']}"
                frac = float(np.mean(SG))
                sub = f"{frac:.0%} of cells significant" if frac > 0 else "no significant cells"
                ax.set_title(f"{nm}: {ttl}\nn = {n}; {sub}", loc="left", fontsize=10)
        cax = fig.add_axes([left / Wf, 0.55 / Hf, (2 * mapw + gap) / Wf, 0.14 / Hf])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
        cb.set_label("z500, <30-day part (m); shaded where FDR 10% significant; contours every 10 m")
        fig.suptitle("ERA5: 500 hPa height 1–3 days before positive torque events", fontsize=12.5, color=INK,
                     x=0.02, y=1 - 0.12 / Hf, ha="left", va="top")
        fig.savefig(dest / "q3_z500_precursors.webp", **WEBP)
        plt.close(fig)


def site_2026(data: Path, dest: Path):
    import torque_budget_2026 as B
    b = json.loads((data / "budget2026.json").read_text())
    g = pd.DataFrame(b["hourly_residual_by_end_hour"])
    sp = pd.read_csv(data / "budget2026_spectrum.csv")
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.8), constrained_layout=True)
        ax = axes[0]
        jump = g["time"].isin([10, 22])
        ax.bar(g["time"], np.where(jump, 17.0, g["mean"]), color=[C_RES if h in (10, 22) else C_MTN for h in g["time"]], width=0.75)
        ax.errorbar(g["time"][~jump], g["mean"][~jump], yerr=g["std"][~jump], fmt="none", ecolor=INK2, lw=1)
        for _, rr in g[jump].iterrows():
            right = rr["time"] < 20
            ax.text(rr["time"] + (0.55 if right else -0.55), 15.6, f"{rr['mean']:+.0f} ± {rr['std']:.0f}".replace("-", "\u2212"),
                    ha="left" if right else "right", va="center", fontsize=9.5, color=INK,
                    bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
        ax.set_ylim(0, 17)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("end of the one-hour interval (UTC)"); ax.set_ylabel("dM/dt minus torques (Hadley)")
        wt = b["window_test"]
        ax.set_title(f"Inside the 4D-Var windows {wt['inside_mean']:+.1f} ± {wt['inside_sd']:.1f} Hadley an hour;\n"
                     "the 09→10 and 21→22 UTC bars (window starts) are cut off and labelled", loc="left", fontsize=10.5)
        ax = axes[1]
        m = sp.cpd > 0
        ax.loglog(sp.cpd[m], sp.psd_dMdt[m], color=INK, lw=1.4, label="dM/dt")
        ax.loglog(sp.cpd[m], sp.psd_T[m], color=C_NET, lw=1.4, label="sum of torques")
        ax.loglog(sp.cpd[m], sp.psd_R[m], color=C_RES, lw=1.6, label="dM/dt minus torques")
        ax.set_xlabel("frequency (cycles per day)"); ax.set_ylabel("power (Hadley² per cpd)")
        ax.set_title("Spectra: the residual is a comb of lines at whole cycles per day", loc="left", fontsize=10.5)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
        fig.suptitle("ERA5 hourly, 1 March – 3 October 2026: the AAM budget hour by hour", fontsize=12.5, color=INK,
                     x=0.02, ha="left")
        fig.savefig(dest / "b2026_hourly.webp", **WEBP)
        plt.close(fig)
        # cumulative anomalies since 15 April (descriptive)
        h = B.load_hourly()
        dh = S.load_era5()
        cols = ["mtn_glob", "fric_glob", "gwd_glob", "aam_glob", "aamm_glob", "mtn_AND"] + [f"mtn_{r}" for r in B.OTHER]
        hd = h[cols].resample("D").mean()
        an = {c: hd[c] - B.clim_on(dh, c, hd.index) for c in cols}
        w0 = pd.Timestamp(B.W0)
        sl = slice(w0, hd.index[-1])
        Mt = (an["aam_glob"] + an["aamm_glob"]).loc[sl]
        other = sum(an[f"mtn_{r}"] for r in B.OTHER)
        rest = an["mtn_glob"] - an["mtn_AND"] - other
        cum = lambda s: s.loc[sl].cumsum() * S.HADLEY_DAY_TO_AAM          # noqa: E731
        T = an["mtn_glob"] + an["fric_glob"] + an["gwd_glob"]
        fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
        ax.plot(Mt.index, Mt - Mt.iloc[0], color=INK, lw=2.6, label="AAM anomaly change (motion + mass)")
        ax.plot(Mt.index, cum(T), color=C_NET, lw=2.4, label="sum of the torque anomalies")
        ax.plot(Mt.index, cum(an["mtn_AND"]), color=C_MTN, lw=1.8, label="Andes")
        ax.plot(Mt.index, cum(other + rest), color=C_MTN, lw=1.4, ls="--", label="all other mountain")
        ax.plot(Mt.index, cum(an["fric_glob"]), color=C_FRIC, lw=1.8, label="friction")
        ax.plot(Mt.index, cum(an["gwd_glob"]), color=C_GWD, lw=1.4, label="gravity-wave drag")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.axvline(pd.Timestamp(B.W1), color=GRID, lw=1)
        ax.set_ylabel("10²⁵ kg m² s⁻¹ since 15 April")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.suptitle("ERA5 2026: the AAM rise against the torques that should explain it", fontsize=12.5, color=INK,
                     x=0.02, ha="left")
        fig.savefig(dest / "b2026_cumulative.webp", **WEBP)
        plt.close(fig)
        att = b["attribution"]
        names = list(att)
        fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
        yy = np.arange(len(names))[::-1].astype(float)
        cols_ = [C_MTN if k in ("Andes", "other ranges", "rest of mountain") else (C_GWD if k == "GWD" else C_FRIC) for k in names]
        ax.barh(yy, [att[k]["hist_sd"] * 2 for k in names], left=[-att[k]["hist_sd"] for k in names], height=0.7,
                color="#ecebe5", lw=0, label="±1 sd of the same window, 1979–2025")
        ax.barh(yy + 0.15, [att[k]["v2026"] for k in names], height=0.3, color=cols_, label="2026")
        ax.barh(yy - 0.15, [att[k]["v2026_detrended"] for k in names], height=0.3, color=cols_, alpha=0.45,
                label="2026, ERA5 trend removed")
        labs = {"Andes": "Andes", "other ranges": "other five ranges", "rest of mountain": "mountain elsewhere",
                "friction tropics": "friction, tropics", "friction NH ext": "friction, NH extratropics",
                "friction SH ext": "friction, SH extratropics", "GWD": "gravity-wave drag"}
        def rk(n):
            top_ = 49 - n
            if top_ <= n:
                return f"{ordn(top_)} highest of 48"
            return f"{ordn(n)} lowest of 48"
        ax.set_yticks(yy); ax.set_yticklabels([f"{labs[k]} ({rk(att[k]['rank_of_48'])})" for k in names])
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel("15 Apr – 19 Sep integral (10²⁵ kg m² s⁻¹)")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=C_MTN, label="2026 (mountain blue, friction orange)"),
                           Patch(color=C_MTN, alpha=0.45, label="2026, ERA5 trend removed"),
                           Patch(color="#ecebe5", label="±1 sd of the window, 1979–2025")], frameon=False, fontsize=9.5,
                  loc="outside lower center", ncol=1)
        fig.suptitle("ERA5 2026: each term's share against its own history", fontsize=12.5, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "b2026_attribution.webp", **WEBP)
        plt.close(fig)


def site_increments(data: Path, dest: Path):
    r = json.loads((data / "increments.json").read_text())
    lat = -87.5 + 5 * np.arange(36)
    U = 1e3                                                     # 1e25 -> 1e22 kg m2/s
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.6), sharex=True, constrained_layout=True)
        ax = axes[0]
        for k, lab, col in (("10", "09 UTC window start", C_MTN), ("22", "21 UTC window start", C_FRIC)):
            p = r["profile"][k]
            m = np.array(p["mean"]) * U; lo = np.array(p["lo"]) * U; hi = np.array(p["hi"]) * U; sg = np.array(p["sig"])
            ax.fill_between(lat, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(lat, m, color=col, lw=2.2, label=f"{lab} (upper panel)")
            ax.scatter(lat[sg], m[sg], color=col, s=18, zorder=3)
            ax.scatter(lat[~sg], m[~sg], facecolor=SURF, edgecolor=col, s=16, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_ylabel("10²² kg m² s⁻¹ per 5° band")
        g1, g2 = r["profile"]["10"]["global"], r["profile"]["22"]["global"]
        ax.set_title("Total (wind + mass term) per window; global sum\n"
                     f"09 UTC {num_(g1['mean'] * U)} [{num_(g1['lo'] * U)}, {num_(g1['hi'] * U)}], "
                     f"21 UTC {num_(g2['mean'] * U)} [{num_(g2['lo'] * U)}, {num_(g2['hi'] * U)}] ×10²² kg m² s⁻¹", loc="left", fontsize=10.5)
        ax = axes[1]
        for k, lab, col in (("r", "wind (motion) term (lower panel)", INK), ("m", "mass term (lower panel)", C_GWD)):
            p = r["profile"]["both" + k]
            m = np.array(p["mean"]) * U; lo = np.array(p["lo"]) * U; hi = np.array(p["hi"]) * U; sg = np.array(p["sig"])
            ax.fill_between(lat, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(lat, m, color=col, lw=2.2, label=lab)
            ax.scatter(lat[sg], m[sg], color=col, s=18, zorder=3)
            ax.scatter(lat[~sg], m[~sg], facecolor=SURF, edgecolor=col, s=16, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_ylabel("10²² kg m² s⁻¹ per band, per day")
        ax.set_xlabel("latitude")
        ax.set_xticks(range(-90, 91, 30)); ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
        ax.set_title("Both window starts together, split into wind and mass terms", loc="left")
        h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
        d0, d1, n = r["days"]
        fig.suptitle(f"ERA5 AAM increments by latitude, {d0[:7]} to {d1[:7]}",
                     fontsize=12, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "inc_profile.webp", **WEBP)
        plt.close(fig)
        # seasonal cycle (blocks: month x 5-degree band; hatched = not significant; grey = no data)
        Mm = pd.read_csv(data / "inc_season_mean.csv", index_col=0)
        SG = pd.read_csv(data / "inc_season_sig.csv", index_col=0).values.astype(bool)
        months = Mm.index.tolist()
        V = Mm.values * U
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.6), constrained_layout=True, gridspec_kw=dict(height_ratios=[1, 2.4]))
        ax = axes[0]
        gm = r["global_by_month"]
        x = np.arange(len(months))
        ax.bar(x, [gm[m]["mean"] * U for m in months], color=C_NET, width=0.75)
        ax.errorbar(x, [gm[m]["mean"] * U for m in months],
                    yerr=[1.96 * gm[m]["sd"] / np.sqrt(gm[m]["n"]) * U for m in months], fmt="none", ecolor=INK2, lw=1)
        ax.set_xticks(x[::3]); ax.set_xticklabels([m for m in months][::3], fontsize=9)
        ax.set_ylabel("10²² per day")
        ax.set_title("Global increment per day, by month (±95%)", loc="left")
        ax = axes[1]
        lev = nice_levels(np.nanpercentile(np.abs(V), 98))
        norm = BoundaryNorm(lev, DIV.N, extend="both")
        xe = np.arange(len(months) + 1) - 0.5
        ye = np.arange(-90, 91, 5)
        nod = ~np.isfinite(V) | (np.abs(V) < 1e-9)
        ax.pcolormesh(xe, ye, np.where(nod, 1.0, np.nan).T, cmap=LinearSegmentedColormap.from_list("nd", ["#d9d8d2"] * 2), shading="flat")
        ax.pcolormesh(xe, ye, np.where(nod, np.nan, V).T, cmap=DIV, norm=norm, shading="flat")
        ns = np.where(~SG & ~nod, 1.0, np.nan)
        ax.pcolor(xe, ye, ns.T, hatch="////", alpha=0.0, shading="flat")
        ax.set_xticks(x[::3]); ax.set_xticklabels([m for m in months][::3], fontsize=9)
        ax.set_yticks(range(-90, 91, 30)); ax.set_yticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
        ax.grid(False)
        ax.set_title(f"By latitude: hatched = not significant (effective-n t-test, FDR 10%; {float(SG[~nod].mean()):.0%} of cells significant)",
                     loc="left", fontsize=9.5)
        import matplotlib.cm as mcm
        cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=DIV), ax=ax, orientation="horizontal", fraction=0.05, pad=0.08,
                          aspect=40, ticks=lev[::2])
        cb.set_label("10²² kg m² s⁻¹ per 5° band per day")
        fig.suptitle("ERA5 increments of angular momentum: the seasonal cycle", fontsize=12, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "inc_season.webp", **WEBP)
        plt.close(fig)
        # vertical section: AM density per 100 hPa (colour x area = momentum on linear-pressure axes), blocks
        sec = data / "inc_section.npz"
        if sec.exists():
            z = np.load(sec)
            levs = z["level"]
            e = np.empty(len(levs) + 1); e[1:-1] = 0.5 * (levs[1:] + levs[:-1]); e[0] = 0.0
            e[-1] = levs[-1] + 0.5 * (levs[-1] - levs[-2])             # the layer edges used by torque_increments_u37.py
            dpn = np.diff(e)                                              # nominal layer thickness (hPa)
            ye = np.arange(-90, 91, 5)
            fr = json.loads((data / "increments.json").read_text())
            fig = plt.figure(figsize=(6.8, 9.8), constrained_layout=True)
            gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 2.6, 1.0, 2.6])
            tro = levs > 100; stra = levs <= 100
            dens = {w: z[f"{w}_mean"] * U / (dpn[:, None] / 100.0) for w in ("J10", "J22")}
            # grey: more than half of the layer below ground in that band (surface pressure estimated from ERA5's
            # orography on the 2.5-degree maps grid with a standard atmosphere)
            below = np.zeros((len(levs), 36), bool)
            fm = S.REF / "torque_u37maps_2025_2026.nc"
            if fm.exists():
                import xarray as xr
                hz = xr.open_dataset(fm)["h_mean"]
                psf = 1013.25 * (1 - 2.25577e-5 * np.clip(hz.values, 0, None)) ** 5.25588
                la_ = hz.lat.values; wcos = np.cos(np.deg2rad(la_))
                for jb in range(36):
                    rows = (la_ >= -90 + 5 * jb) & (la_ < -85 + 5 * jb)
                    if not rows.any():
                        continue
                    pr = psf[rows]; wr = np.repeat(wcos[rows][:, None], pr.shape[1], 1)
                    for kk in range(len(levs)):
                        f_ = np.clip(np.minimum(e[kk + 1], pr) - np.minimum(e[kk], pr), 0, None) / dpn[kk]
                        below[kk, jb] = (f_ * wr).sum() / wr.sum() < 0.5
            allv = np.concatenate([dens[w][tro].ravel() for w in dens])
            lev_t = nice_levels(np.nanpercentile(np.abs(allv), 98))
            alls = np.concatenate([dens[w][stra].ravel() for w in dens])
            lev_s = nice_levels(np.nanpercentile(np.abs(alls), 98))
            nt, ns_ = BoundaryNorm(lev_t, DIV.N, extend="both"), BoundaryNorm(lev_s, DIV.N, extend="both")
            import matplotlib.cm as mcm
            for j, (w, ttl) in enumerate((("J10", "09 UTC window start"), ("J22", "21 UTC window start"))):
                V2 = dens[w]; sg = z[f"{w}_sig"]; nod = z[f"{w}_nodata"] | below
                for k, (mk, norm, axrow) in enumerate(((stra, ns_, 2 * j), (tro, nt, 2 * j + 1))):
                    ax = fig.add_subplot(gs[axrow])
                    idx = np.where(mk)[0]
                    pe = np.concatenate([e[idx], [e[idx[-1] + 1]]])
                    Vm = V2[idx]; Sm = sg[idx]; Nm = nod[idx]
                    ax.pcolormesh(ye, pe, np.where(Nm, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("nd", ["#d9d8d2"] * 2), shading="flat")
                    ax.pcolormesh(ye, pe, np.where(Nm, np.nan, Vm), cmap=DIV, norm=norm, shading="flat")
                    ax.pcolor(ye, pe, np.where(~Sm & ~Nm, 1.0, np.nan), hatch="////", alpha=0.0, shading="flat")
                    ax.grid(False)
                    if k == 0:
                        ax.set_ylim(e[idx[-1] + 1], 0)
                        ax.set_yticks([100, 50, 0])
                        fs = fr["u37_frac_sig_with_data"][w]
                        ax.set_title(f"{ttl}: 100–1 hPa levels, own scale, drawn 3× taller than its mass; "
                                     f"{fs:.0%} of cells with data significant", loc="left", fontsize=9)
                        ax.set_xticklabels([])
                        cb = fig.colorbar(mcm.ScalarMappable(norm=ns_, cmap=DIV), ax=ax, orientation="vertical", fraction=0.04,
                                          pad=0.01, ticks=lev_s[::2])
                    else:
                        ax.set_ylim(e[-1], e[idx[0]])
                        ax.set_yticks([1000, 850, 700, 500, 300, 125])
                        ax.set_title(f"{ttl}: 1000–125 hPa levels", loc="left", fontsize=9.5)
                        cb = fig.colorbar(mcm.ScalarMappable(norm=nt, cmap=DIV), ax=ax, orientation="vertical", fraction=0.04,
                                          pad=0.01, ticks=lev_t[::2])
                        if j == 1:
                            ax.set_xticks(range(-90, 91, 30)); ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
                        else:
                            ax.set_xticklabels([])
                    ax.set_ylabel("hPa")
                    cb.ax.tick_params(labelsize=8)
            n = int(fr["u37_J10"]["n_days"])
            fig.suptitle(f"ERA5 wind-term AAM increments by latitude and pressure ({n} days)\n"
                         "10²² kg m² s⁻¹ per 5° band per 100 hPa, per window; axes linear in pressure;\n"
                         "hatched = not significant; grey = more than half the layer below ground",
                         fontsize=10, color=INK, x=0.02, ha="left")
            fig.savefig(dest / "inc_section.webp", **WEBP)
            plt.close(fig)
        # AIFS-ENS forecast drift
        fd = r.get("fc_drift")
        if fd:
            fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
            for reg, lab, col in (("global", "global", INK), ("nh", "Northern Hemisphere", C_MTN), ("sh", "Southern Hemisphere", C_FRIC)):
                d = pd.DataFrame(fd["by_region"][reg])
                ax.fill_between(d.lead, d.lo, d.hi, color=col, alpha=0.14, lw=0)
                ax.plot(d.lead, d["mean"], color=col, lw=2.2, label=lab)
                ax.scatter(d.lead[d.sig], d["mean"][d.sig], color=col, s=20, zorder=3)
                ax.scatter(d.lead[~d.sig], d["mean"][~d.sig], facecolor=SURF, edgecolor=col, s=16, zorder=3)
            ax.axhline(0, color="#c3c2b7", lw=1)
            ax.set_xlabel("lead (days)"); ax.set_ylabel("forecast minus verifying analysis (10²⁵ kg m² s⁻¹)")
            fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
            i0, i1, n = fd["inits"]
            ax.set_title(f"{n} inits, {i0[:10]} to {i1[:10]}; filled: 95% interval\nexcludes zero (weekly-block bootstrap)",
                         loc="left", fontsize=10)
            fig.suptitle("AIFS-ENS ensemble-mean angular momentum drift", fontsize=12, color=INK, x=0.02, ha="left")
            fig.savefig(dest / "fc_drift.webp", **WEBP)
            plt.close(fig)


def site_increment_maps(data: Path, dest: Path):
    """Low-level increment maps (u37maps mode) and the regional split."""
    import cartopy.crs as ccrs
    import xarray as xr
    f = S.REF / "torque_u37maps_2025_2026.nc"
    if not f.exists() or not (data / "inc_maps_sig.nc").exists():
        return
    z = xr.open_dataset(f); sg = xr.open_dataset(data / "inc_maps_sig.nc")
    r = json.loads((data / "increments_maps.json").read_text())
    lat, lon = z.lat.values, z.lon.values
    lonc = np.concatenate([lon, [lon[0] + 360]])

    def cyc(F):
        return np.concatenate([F, F[:, :1]], axis=1)
    hm = z["h_mean"].values

    def draw(ax, F, S_, lev, ttl):
        cf = ax.contourf(lonc, lat, cyc(np.where(S_, F, np.nan)), levels=lev, cmap=DIV, extend="both", transform=ccrs.PlateCarree())
        ax.contour(lonc, lat, cyc(hm), levels=[500, 1500, 3000], colors=[INK2, INK2, INK], linewidths=[0.4, 0.7, 1.0],
                   transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.4, color="#8a877f")
        ax.set_global()
        ax.set_title(ttl, loc="left", fontsize=10.5)
        return cf
    with plt.rc_context(SITE_RC):
        # (1) u at 925 and 850 hPa, both window starts (per day)
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.6), subplot_kw=dict(projection=ccrs.Robinson(central_longitude=10)),
                                 constrained_layout=True)
        lev = np.linspace(-0.5, 0.5, 11)
        for ax, L in zip(axes, ("925", "850")):
            k = f"u{L}_both_ALL"
            fs_ = r.get(f"frac_sig_{k}", 0)
            cf = draw(ax, z[f"{k}_mean"].values, sg[f"{k}_sig"].values, lev,
                      f"{L} hPa zonal wind: shaded where significant ({fs_:.0%} of cells, FDR 10%)")
        cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40)
        cb.set_label("increment of u per day (m/s, both window starts); contours: terrain 500, 1500, 3000 m")
        fig.suptitle(f"ERA5 low-level wind increments, {r['days'].split(' (')[0].replace('..', ' to ')}", fontsize=12, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "inc_maps_u.webp", **WEBP)
        plt.close(fig)
        # (2) the >=850 hPa angular momentum increment, DJF vs JJA
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.6), subplot_kw=dict(projection=ccrs.Robinson(central_longitude=10)),
                                 constrained_layout=True)
        V = np.concatenate([z[f"amL_both_{s}_mean"].values.ravel() for s in ("DJF", "JJA")])
        lev = nice_levels(float(np.nanpercentile(np.abs(V), 98)))
        for ax, s in zip(axes, ("DJF", "JJA")):
            k = f"amL_both_{s}"
            cf = draw(ax, z[f"{k}_mean"].values, sg[f"{k}_sig"].values, lev,
                      f"{s}: shaded where significant ({r.get('frac_sig_' + k, 0):.0%} of cells, FDR 10%)")
        cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40)
        cb.set_label("AM increment at and below 850 hPa (10²⁰ kg m² s⁻¹ per 2.5° cell per day)")
        fig.suptitle("ERA5 low-level angular momentum increments by season", fontsize=12, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "inc_maps_amL_season.webp", **WEBP)
        plt.close(fig)
        # (3) the regional split
        sp = r["split"]
        regs = [("ORO", "terrain and 500 km downstream"), ("OCX", "ocean 30–65°"), ("OCO", "other ocean"), ("LND", "other land")]
        fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
        y = np.arange(len(regs))[::-1].astype(float)
        cols = {"ALL": INK2, "DJF": C_MTN, "JJA": C_FRIC}
        for j, s in enumerate(("ALL", "DJF", "JJA")):
            for yi, (rk, lab) in zip(y, regs):
                d = sp[f"amL_{s}_{rk}"]
                yy = yi + 0.25 - 0.25 * j
                ax.barh(yy, 100 * d["share"], height=0.22, color=cols[s], label={"ALL": "all days"}.get(s, s) if yi == y[0] else None)
                ax.plot([100 * d["lo"], 100 * d["hi"]], [yy] * 2, color=INK, lw=1)
        for yi, (rk, lab) in zip(y, regs):
            ax.plot([100 * r["area"][rk]] * 2, [yi - 0.42, yi + 0.42], ls="--", lw=1.3, color=INK, zorder=4,
                    label="share of the Earth's area" if yi == y[0] else None)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in regs])
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel("share of the global increment below 850 hPa (%)")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=9.5)
        fig.suptitle("Where ERA5's low-level angular momentum increment goes", fontsize=12, color=INK, x=0.02, ha="left")
        ax.set_title("bars: share with 95% interval; dashed: share of the area", loc="left", fontsize=10)
        fig.savefig(dest / "inc_maps_split.webp", **WEBP)
        plt.close(fig)


def site_gfz(data: Path, dest: Path):
    """The explainer's opening figure: GFZ global AAM by ENSO phase (torque_gfz_enso.py tables)."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    s = pd.read_csv(data / "gfz_enso_series.csv", parse_dates=["month"]).set_index("month")
    r = json.loads((data / "gfz_enso.json").read_text())
    o = r["oni"]
    sl = o["slope_per_K"]["est"]
    C_EN, C_LN, C_NE = "#d8423f", "#2a78d6", "#c3c2b7"
    s = s[s.aam3.notna()]
    ph = np.where(s.oni >= 0.5, "EN", np.where(s.oni <= -0.5, "LN", "NEU"))
    lod = r["lod_ms_per_1e25"]
    with plt.rc_context(SITE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.4), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1.15]})
        ax = axes[0]
        t0 = s.index.values                                              # each month drawn as a flat block, no seams
        t1 = (s.index + pd.offsets.MonthBegin(1)).values
        xs, ys = np.ravel(np.column_stack([t0, t1])), np.repeat(s.aam3.values, 2)
        for k, c in (("EN", C_EN), ("NEU", C_NE), ("LN", C_LN)):
            ax.fill_between(xs, 0, ys, where=np.repeat(ph == k, 2), color=c, lw=0)
        ax.plot(s.index + pd.Timedelta(days=15), sl * s.oni, color=INK, lw=0.9)
        ax.axhline(0, color=INK2, lw=0.8)
        for lab, t_, up in (("1982–83", "1983-02", True), ("1997–98", "1997-09", True), ("2015–16", "2016-02", True),
                            ("2026", "2026-08", True), ("1988–89", "1989-02", False), ("1999–2000", "2000-03", False),
                            ("2010–11", "2010-08", False)):
            v = float(s.aam3.loc[t_].iloc[0])
            ax.annotate(lab, (pd.Timestamp(t_), v), xytext=(0, 4 if up else -4), textcoords="offset points",
                        ha="right" if lab == "2026" else "center", va="bottom" if up else "top", fontsize=9, color=INK)
        ax.set_ylim(-3.4, 4.2)
        ax.set_xlim(pd.Timestamp("1978-09-01"), pd.Timestamp("2026-12-31"))
        ax.xaxis.set_major_locator(mdates.YearLocator(10, month=1)); ax.xaxis.set_minor_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
        sec = ax.secondary_yaxis("right", functions=(lambda x: x * lod, lambda y: y / lod))
        sec.set_ylabel("length of day (ms)")
        ax.set_title("3-month means, coloured by the ONI of the same season", loc="left")
        ax = axes[1]
        for k, c in (("EN", C_EN), ("NEU", C_NE), ("LN", C_LN)):
            m_ = ph == k
            ax.scatter(s.oni[m_], s.aam3[m_], s=11, color=c, alpha=0.45, lw=0)
        xx = np.array([-2.2, 2.8])
        ax.plot(xx, sl * xx, color=INK, lw=1.4)
        for k, c in (("EN", C_EN), ("NEU", C_NE), ("LN", C_LN)):
            m_ = ph == k
            e_ = o[k]
            ax.errorbar(float(s.oni[m_].mean()), e_["est"], yerr=[[e_["est"] - e_["lo"]], [e_["hi"] - e_["est"]]], fmt="o",
                        ms=9, mfc=c, mec=INK, ecolor=INK, elinewidth=1.6, capsize=4, zorder=5)
        y26 = s.loc["2026-05":]
        ax.plot(y26.oni, y26.aam3, color=INK, lw=1, zorder=4)
        ax.scatter(y26.oni, y26.aam3, s=26, facecolor=SURF, edgecolor=INK, lw=1.2, zorder=6)
        ax.annotate("2026, AMJ to JAS", (float(y26.oni.iloc[-1]), float(y26.aam3.iloc[-1])), xytext=(-8, 2),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
        ax.axhline(0, color="#c3c2b7", lw=1); ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlim(-2.2, 2.8); ax.set_ylim(-3.4, 4.2)
        ax.set_xlabel("ONI (K)"); ax.set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
        ax.set_title(f"r = {o['r']['est']:.2f}; slope {num_(sl, 2)} {ci_(o['slope_per_K'])} ×10²⁵ per K;\n"
                     "large dots: phase means with 95% intervals", loc="left")
        hs = [Patch(color=C_EN, label="El Niño (ONI ≥ +0.5)"), Patch(color=C_NE, label="neutral"),
              Patch(color=C_LN, label="La Niña (ONI ≤ −0.5)"), Line2D([], [], color=INK, lw=1.2, label=f"ONI × {sl:.2f} (the fit)")]
        fig.legend(handles=hs, loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
        fig.suptitle(f"GFZ (ECMWF analyses), {o['span'][0][:4]}–2026: global angular momentum\nis high in El Niño and low in La Niña",
                     fontsize=12.5, color=INK, x=0.02, ha="left")
        fig.savefig(dest / "gfz_aam_enso.webp", **WEBP)
        plt.close(fig)


def ci_(d, k=2):
    return f"[{num_(d['lo'], k)}, {num_(d['hi'], k)}]"


def nice_levels(lim, n=10):
    """Round symmetric colour levels around zero."""
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    raw = lim / (n / 2)
    mag = 10 ** np.floor(np.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda c: abs(c * mag - raw)) * mag
    k = int(np.ceil(lim / step))
    return np.arange(-k, k + 1) * step


def ordn(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def num_(x, d=0):
    return (f"{x:+.{d}f}").replace("-", "−")


def site_all(site: Path, data: Path = DATA):
    dest = site / "studies" / "torque-enso"
    dest.mkdir(parents=True, exist_ok=True)
    for f in (site_gfz, site_q1, site_slp, site_q2, site_q3, site_2026, site_increments, site_increment_maps):
        f(data, dest)
        print("ok", f.__name__)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="study output dir (scratch figures)")
    ap.add_argument("--site", default=None, help="site root: write the study page figures to studies/torque-enso/")
    ap.add_argument("--only", default="q1,q2lag,q2c,2026,q3maps,q3ll,slp")
    a = ap.parse_args()
    if a.site:
        site_all(Path(a.site))
        return 0
    out = Path(a.out)
    fn = {"q1": fig_q1, "q2lag": fig_q2_lag, "q2c": fig_q2_contrib, "2026": fig_2026, "q3maps": fig_q3_maps,
          "q3ll": fig_q3_leadlag, "slp": fig_slp}
    for k in a.only.split(","):
        try:
            fn[k](out)
            print("ok", k)
        except Exception as e:                                                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", k, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
