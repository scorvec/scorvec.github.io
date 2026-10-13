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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="q1,q2lag,q2c,2026,q3maps,q3ll,slp")
    a = ap.parse_args()
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
