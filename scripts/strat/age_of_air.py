#!/usr/bin/env python3
"""Render the "Age of stratospheric air" item of stratosphere.html from the committed reference
scripts/strat/reference/age_of_air.nc (built once on the laptop by build_age_of_air.py from the MERRA-2 GMI replay).
A STATIC product: the workflow (age-of-air.yml) runs when this script or the reference changes, or on dispatch.

Outputs (assets/sst/):
  aoa_clim_{djf,mam,jja,son,ann}.webp   latitude-height mean age, by season
  aoa_cycle.webp                        the seasonal cycle at 50 hPa: boxes, and latitude x month (significant cells)
  aoa_trend.webp, aoa_trend04.webp      linear trend 1980-2019 and 2004-2019, significant cells only, + the balloon box
  aoa_enso.webp                         regression on ENSO (RONI), significant cells only, + the lag scan
  aoa_qbo.webp                          regression on the QBO (30 and 50 hPa winds), significant cells only
  data/age_of_air.json                  the numbers quoted on the page

    python scripts/strat/age_of_air.py [--out-root .]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REF = HERE / "reference" / "age_of_air.nc"

INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
NAVY, SIENNA = "#24466e", "#b4532a"
H = 7.0
PT, PB = 1.0, 200.0
SEAS_LABEL = {"DJF": "December–February", "MAM": "March–May", "JJA": "June–August", "SON": "September–November",
              "ANN": "annual mean"}
# sequential (age) and diverging (anomaly) palettes; the diverging one is the site's BDC section palette
SEQ = ["#f7f4ea", "#efe6c9", "#e2d3a3", "#cfbd82", "#b3a56b", "#8f8c5e", "#6b7556", "#4c5f53", "#34494d", "#223543",
       "#172636", "#0e1a28"]
# diverging classes with no white class: every shaded (= significant) cell gets a colour
DIV10 = ["#3b4f8f", "#5a78b4", "#8ea8d2", "#b3c5e3", "#d6e0f0", "#f7dcc6", "#f3c9a8", "#e59c6c", "#cc6a3a", "#9c3f1c"]
UNDER, OVER = "#26345f", "#6e2710"
DIV = ["#26345f", "#3b4f8f", "#5a78b4", "#8ea8d2", "#c3d1e8", "#ffffff", "#f3c9a8", "#e59c6c", "#cc6a3a", "#9c3f1c",
       "#6e2710"]


def header(fig, title, sub, x=0.055):
    W, Hh = fig.get_figwidth(), fig.get_figheight()
    y = 1 - 0.28 / Hh
    fig.text(x, y, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    lines = textwrap.wrap(sub, width=int((1 - 2 * x) * W * 12.6))
    fig.text(x, y - 0.36 / Hh, "\n".join(lines), ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)
    return y - (0.36 + 0.19 * len(lines)) / Hh


def footer(fig, text, x=0.055):
    """Wrapped to the figure width; explicit newlines start a new paragraph."""
    W = fig.get_figwidth()
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=int((1 - 2 * x) * W * 14.2))
    fig.text(x, 0.014, "\n".join(lines), ha="left", va="bottom", fontsize=8.6, color=MUTED, linespacing=1.4)
    return len(lines)


def cbar(fig, cf, rect, ticks, label, extend=None):
    cax = fig.add_axes(rect)
    kw = {"extend": extend} if extend else {}
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks, **kw)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.4, colors=INK)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(label, fontsize=9, color=INK, labelpad=3)
    return cb


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 90})
    plt.close(fig)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa1a9")
    ax.tick_params(colors=INK, labelsize=9.2, length=3)


def pax(ax, pt=PT, pb=PB, km=True):
    ax.set_yscale("log"); ax.set_ylim(pb, pt)
    yt = [p for p in [200, 150, 100, 70, 50, 30, 20, 10, 5, 3, 2, 1, 0.5, 0.3, 0.1] if pt <= p <= pb]
    ax.set_yticks(yt); ax.set_yticklabels([f"{p:g}" for p in yt]); ax.minorticks_off()
    ax.set_ylabel("hPa", fontsize=9.2, color=INK)
    if km:
        sec = ax.secondary_yaxis("right", functions=(lambda p: -H * np.log(np.clip(p, 1e-3, None) / 1000.0),
                                                     lambda z: 1000.0 * np.exp(-z / H)))
        from matplotlib.ticker import FixedLocator, FuncFormatter
        sec.yaxis.set_major_locator(FixedLocator([z for z in range(10, 70, 5)]))
        sec.yaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))
        sec.yaxis.set_minor_locator(FixedLocator([]))
        sec.set_ylabel("log-pressure height, km", fontsize=8.8, color=MUTED)
        sec.tick_params(colors=MUTED, labelsize=8.4)


def latax(ax):
    ax.set_xlim(-90, 90)
    xt = np.arange(-90, 91, 30)
    ax.set_xticks(xt); ax.set_xticklabels([f"{abs(x)}°{'S' if x < 0 else ('N' if x > 0 else '')}" for x in xt])


def clabels(ax, cs, fmt="{:g}"):
    ax.clabel(cs, fontsize=7.8, fmt=lambda v: fmt.format(v), inline_spacing=2)


# ------------------------------------------------------------------------------------------------------- figures
def fig_clim(ds, season, out):
    lev, lat = ds.lev.values, ds.lat.values
    k = (lev <= PB) & (lev >= PT)
    A = ds.clim_season.sel(season=season).values[k]
    lv = np.arange(0, 6.01, 0.5)
    cmap = ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.04, 0.86, len(lv) - 1)))
    cmap.set_over(plt.get_cmap("YlGnBu")(0.95))
    norm = BoundaryNorm(lv, cmap.N)
    fig = plt.figure(figsize=(11.4, 6.9), dpi=130)
    top = header(fig, f"MERRA-2 GMI · Mean age of stratospheric air · {SEAS_LABEL[season]}, {ds.attrs['period'][:4]}–"
                      f"{ds.attrs['period'][9:13]}",
                 "Zonal-mean age of the GMI clock tracer: the mean time, in years, since the air last touched the ground. "
                 "Air enters through the tropical tropopause, rises in the tropics and spreads poleward; the oldest air is "
                 "in the polar upper stratosphere.")
    ax = fig.add_axes([0.07, 0.235, 0.83, top - 0.27])
    cf = ax.contourf(lat, lev[k], A, levels=lv, cmap=cmap, norm=norm, extend="max")
    cs1 = ax.contour(lat, lev[k], A, levels=np.arange(0.5, 3.01, 0.5), colors=[INK], linewidths=0.6, alpha=0.6)
    cs2 = ax.contour(lat, lev[k], A, levels=np.arange(3.5, 6.01, 0.5), colors=["white"], linewidths=0.6, alpha=0.8)
    clabels(ax, cs1); clabels(ax, cs2)
    pax(ax); latax(ax); style(ax)
    cbar(fig, cf, [0.32, 0.135, 0.38, 0.02], lv[::2], "mean age, years", "max")
    v = json.loads(ds.attrs["validation"])
    footer(fig, f"Annual mean at 50 hPa (~20.5 km): tropics {v['50hPa_tropics_10S_10N']:.1f}, 35–55°N {v['50hPa_NH_35_55N']:.1f}, "
                f"35–55°S {v['50hPa_SH_35_55S']:.1f} years; at 10 hPa (~31 km) 35–55°N {v['10hPa_NH_35_55N']:.1f}. Referenced to "
                "the surface: balloon and aircraft estimates referenced to the tropical tropopause read ~0.3 years less. "
                "Source: NASA GMAO MERRA-2 GMI replay (monthly means, NCCS OPeNDAP).")
    save(fig, out)


def fig_cycle(ds, out):
    lev, lat = ds.lev.values, ds.lat.values
    k50 = int(np.argmin(np.abs(lev - 50)))
    C = ds.clim_month.values[:, k50]              # (12, lat)
    S = ds.iav_month.values[:, k50]
    ny = int(ds.attrs["period"][9:13]) - int(ds.attrs["period"][:4]) + 1
    fig = plt.figure(figsize=(12.4, 6.6), dpi=130)
    top = header(fig, f"MERRA-2 GMI · Seasonal cycle of mean age at 50 hPa · {ds.attrs['period'][:4]}–{ds.attrs['period'][9:13]}",
                 "Left: the monthly climatology in five latitude bands, shaded ±1 standard deviation of single years. "
                 "Right: each month's departure from the annual mean at every latitude, drawn only where it differs "
                 "significantly from zero (t-test over the years, false-discovery rate 10 %).")
    ax = fig.add_axes([0.06, 0.2, 0.36, top - 0.27]); style(ax)
    m = np.arange(1, 13)
    bands = [("tropics 10°S–10°N", -10, 10, "#b4532a"), ("35–55°N", 35, 55, "#24466e"), ("35–55°S", -55, -35, "#5a78b4"),
             ("65–90°N", 65, 90, "#1c2430"), ("65–90°S", -90, -65, "#8a8f96")]
    for lab, lo, hi, col in bands:
        w = np.cos(np.deg2rad(lat)) * ((lat >= lo) & (lat <= hi))
        mu = (C * w).sum(1) / w.sum(); sd = np.sqrt(((S ** 2) * w).sum(1) / w.sum())
        ax.fill_between(m, mu - sd, mu + sd, color=col, alpha=0.13, lw=0)
        ax.plot(m, mu, color=col, lw=2.2, label=lab)
    ax.set_xticks(m); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_xlim(1, 12)
    ax.set_ylabel("mean age, years", fontsize=9.2, color=INK); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=3, columnspacing=1.2,
              handlelength=1.6)
    ax.set_title("50 hPa, by latitude band", loc="left", fontsize=11, fontweight="bold", color=INK)
    # right: anomaly from the annual mean, significance by t over years (months are one per year: n = ny)
    from scipy import stats
    anom = C - C.mean(0, keepdims=True)
    se = S / np.sqrt(ny)
    t = anom / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), ny - 1)
    sig = _bh(p)
    ax2 = fig.add_axes([0.52, 0.2, 0.44, top - 0.27]); style(ax2)
    lv = np.array([-9, -6, -4, -2, -1, 0, 1, 2, 4, 6, 9])
    cmap = ListedColormap(DIV10); cmap.set_over(OVER); cmap.set_under(UNDER)
    norm = BoundaryNorm(lv, cmap.N)
    mm = np.r_[0.5, m + 0.0]
    Z = np.where(sig, anom * 12, np.nan)
    cf = ax2.pcolormesh(np.r_[m - 0.5, 12.5], _edges(lat), Z.T, cmap=cmap, norm=norm, shading="flat")
    cs = ax2.contour(m, lat, (anom * 12).T, levels=[-6, -4, -2, 2, 4, 6], colors=[INK], linewidths=0.6, alpha=0.6)
    clabels(ax2, cs)
    ax2.set_xticks(m); ax2.set_xticklabels(list("JFMAMJJASOND")); ax2.set_xlim(0.5, 12.5)
    ax2.set_ylim(-90, 90); ax2.set_yticks(np.arange(-90, 91, 30))
    ax2.set_yticklabels([f"{abs(y)}°{'S' if y < 0 else ('N' if y > 0 else '')}" for y in range(-90, 91, 30)])
    ax2.set_title("50 hPa, departure from the annual mean (months)", loc="left", fontsize=11, fontweight="bold", color=INK)
    cbar(fig, cf, [0.60, 0.1, 0.28, 0.018], lv, "younger ← months → older", "both")
    footer(fig, f"{int(sig.sum())} of {sig.size} latitude-month cells significant. Source: NASA GMAO MERRA-2 GMI replay, "
                "clock-tracer age, monthly zonal means.")
    save(fig, out)
    return {"sig_cells": int(sig.sum()), "cells": int(sig.size)}


def _edges(x):
    x = np.asarray(x, float)
    mid = (x[1:] + x[:-1]) / 2
    return np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]


def _bh(p, alpha=0.10):
    flat = np.sort(p[np.isfinite(p)])
    out = np.zeros(p.shape, bool)
    if not flat.size:
        return out
    ok = flat <= alpha * np.arange(1, flat.size + 1) / flat.size
    if ok.any():
        out = np.isfinite(p) & (p <= flat[np.where(ok)[0].max()])
    return out


def section(ax, ds, name, scale, lv, cbl=None, fig=None, cax=None):
    """Regression section: shading only where significant (FDR mask from the builder), climatology contours."""
    levm, lat = ds.levm.values, ds.lat.values
    coef = ds[f"{name}_coef"].values * scale
    sig = ds[f"{name}_sig"].values.astype(bool)
    cmap = ListedColormap(DIV10); cmap.set_over(OVER); cmap.set_under(UNDER)
    norm = BoundaryNorm(lv, cmap.N)
    Z = np.where(sig, coef, np.nan)
    cf = ax.pcolormesh(lat, levm, np.ma.masked_invalid(Z), cmap=cmap, norm=norm, shading="nearest")
    lev = ds.lev.values; k = (lev <= PB) & (lev >= PT)
    cs = ax.contour(lat, lev[k], ds.clim_season.sel(season="ANN").values[k], levels=[1, 2, 3, 4, 5],
                    colors=["#59636f"], linewidths=0.7, linestyles="dashed")
    clabels(ax, cs, "{:g} yr")
    pax(ax, km=False); latax(ax); style(ax)
    if cax is not None:
        cax.remove()
        cbar(fig, cf, cax.get_position().bounds, lv, cbl or "", "both")
    return int(sig.sum()), int(sig.size)


def fig_trend(ds, out, which="trend"):
    """which = "trend" (whole record) or "trend04" (the sub-period from sub_period_start)."""
    fits = json.loads(ds.attrs["box_fits"]); eng = json.loads(ds.attrs["engel_comparison"])
    per = ds.attrs["period"]; y0 = int(ds.attrs["sub_period_start"]) if which == "trend04" else int(per[:4])
    y1 = int(per[9:13])
    fig = plt.figure(figsize=(12.6, 7.1), dpi=130)
    top = header(fig, f"MERRA-2 GMI · Trend in the mean age of stratospheric air · {y0}–{y1}",
                 "Linear trend of monthly mean age, estimated together with ENSO and the QBO (AR(1) errors"
                 + (", volcanic years left out" if which == "trend" else "") + "). Shaded only where significant after a "
                 "false-discovery-rate control of 10 % over the section; dashed contours: the climatological age (years).")
    ax = fig.add_axes([0.06, 0.29, 0.50, top - 0.33])
    lv = np.array([-0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3])
    cax = fig.add_axes([0.13, 0.175, 0.36, 0.02])
    ns, nc = section(ax, ds, which, 1.0, lv, "younger ← years per decade → older", fig, cax)
    ax.set_title("trend, years per decade", loc="left", fontsize=11, fontweight="bold", color=INK)
    if ns == 0:
        ax.text(0.5, 0.5, "no significant trend anywhere", transform=ax.transAxes, ha="center", fontsize=11, color=MUTED)
    # right: the balloon-comparison box, whole record, with both trend lines
    ax2 = fig.add_axes([0.66, 0.29, 0.31, top - 0.33]); style(ax2)
    t = pd.to_datetime(ds.time.values)
    raw = pd.Series(ds["box_engel_30_50N_30_5hPa"].values, index=t)
    ann = raw.groupby(t.year).mean()
    ax2.plot(ann.index, ann.values, "o-", color=NAVY, ms=3.2, lw=1.4, label="MERRA-2 GMI, annual mean")
    bf = fits["engel_30_50N_30_5hPa"]["fit"]
    for key, a0, col, lab in (("trend", ann.index.min(), SIENNA, f"{ann.index.min()}–{y1}"),
                              ("trend04", int(ds.attrs["sub_period_start"]), "#5a78b4", f"{ds.attrs['sub_period_start']}–{y1}")):
        f = bf[key]
        seg = ann[ann.index >= a0]
        yy = np.array([seg.index.min(), seg.index.max()], float)
        mid = seg.mean() - f["coef"] * (seg.index.values.mean() - 2000) / 10
        ax2.plot(yy, mid + f["coef"] * (yy - 2000) / 10, color=col, lw=2.2 if key == which else 1.4,
                 ls="-" if key == which else (0, (4, 2)),
                 label=f"{lab}: {f['coef']:+.2f} ± {1.96 * f['se']:.2f} yr/decade" + ("" if f["p"] < 0.05 else ", n.s."))
    for s_ in ("1992-01", "2000-01", "2010-01"):
        ax2.axvline(int(s_[:4]) - 0.5, color="#b9bec5", lw=0.8, ls=":")
    ax2.text(1991.3, 0.02, "MERRA-2 streams", transform=ax2.get_xaxis_transform(), fontsize=7.8, color=MUTED, ha="right")
    ax2.grid(color=GRID, lw=0.6); ax2.set_axisbelow(True)
    ax2.set_ylabel("mean age, years", fontsize=9.2, color=INK)
    ax2.set_title("30–50°N, 30–5 hPa (24–35 km)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax2.legend(fontsize=8.2, frameon=False, loc="upper right")
    f = bf[which]
    sigtxt = "significant" if f["p"] < 0.05 else "not significant"
    step = bf["late_minus_early"]
    footer(fig, f"{ns} of {nc} cells significant. Right: the latitudes and heights of the balloon record of Engel et al. (2017), "
                f"who found +0.15 ± 0.18 years per decade for 1975–2016, not significant; MERRA-2 GMI gives {f['coef']:+.2f} ± "
                f"{1.96 * f['se']:.2f} for {y0}–{y1} (95 %, p = {f['p']:.2g}, {sigtxt}). Most of the 40-year decline is a drop of "
                f"{abs(step):.1f} years between the early 1990s and 2003 (mean of {ds.attrs['sub_period_start']}–{y1} minus "
                "1980–1991); after 2004 the box shows no significant trend. The drop coincides with changes in the observing "
                "system MERRA-2 assimilates (a new production stream from 1992, AMSU from 1998, Aura MLS temperatures from 2004), "
                "so it is not clean evidence of a climate trend. Chemistry–climate models simulate air growing younger as "
                "greenhouse warming speeds the circulation; the balloon record does not confirm it. Source: NASA GMAO MERRA-2 "
                "GMI replay.")
    save(fig, out)
    return {"sig_cells": ns, "cells": nc, "engel_box": f, "engel_period_fit": eng, "early_to_late_step": step}


def fig_enso(ds, out):
    lag = int(ds.attrs["enso_lag_months"]); scan = json.loads(ds.attrs["enso_lag_scan"])
    fits = json.loads(ds.attrs["box_fits"])
    per = ds.attrs["period"]
    fig = plt.figure(figsize=(12.6, 6.9), dpi=130)
    top = header(fig, f"MERRA-2 GMI · ENSO signal in the mean age of stratospheric air · {per[:4]}–{per[9:13]}",
                 f"Change in mean age per +1 °C of the relative Niño-3.4 index (RONI) {lag} month{'s' if lag != 1 else ''} "
                 "earlier, fitted together with the trend and the QBO (AR(1) errors, volcanic years left out). "
                 "Shaded only where significant (false-discovery rate 10 %); dashed: the climatological age.")
    ax = fig.add_axes([0.06, 0.25, 0.50, top - 0.29])
    lv = np.array([-1, -0.6, -0.4, -0.25, -0.1, 0, 0.1, 0.25, 0.4, 0.6, 1])
    cax = fig.add_axes([0.13, 0.135, 0.36, 0.02])
    ns, nc = section(ax, ds, "enso", 12.0, lv, "younger ← months per °C → older", fig, cax)
    ax.set_title("months of age per +1 °C RONI", loc="left", fontsize=11, fontweight="bold", color=INK)
    if ns == 0:
        ax.text(0.5, 0.5, "no significant ENSO signal", transform=ax.transAxes, ha="center", fontsize=11, color=MUTED)
    ax2 = fig.add_axes([0.66, 0.25, 0.31, top - 0.29]); style(ax2)
    L = np.array(sorted(int(k) for k in scan)); c = np.array([scan[str(x)]["coef_months_per_K"] for x in L])
    tv = np.array([scan[str(x)]["t"] for x in L])
    se = np.abs(c / np.where(tv != 0, tv, np.nan))
    cols = [SIENNA if abs(x) >= 1.96 else "#b9bec5" for x in tv]
    ax2.bar(L, c, color=cols, width=0.7)
    ax2.errorbar(L, c, yerr=1.96 * se, fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax2.axhline(0, color="#9aa1a9", lw=0.8)
    ax2.set_xticks(L); ax2.set_xlabel("RONI leads age by (months)", fontsize=9, color=INK)
    ax2.set_ylabel("months of age per °C", fontsize=9, color=INK)
    ax2.set_title("tropics 10°S–10°N, 70 hPa, by lag", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax2.grid(color=GRID, lw=0.6, axis="y"); ax2.set_axisbelow(True)
    f = fits["tropics_70hPa"]["fit"]["enso"]
    footer(fig, f"{ns} of {nc} cells significant. Bars: coefficient ± 95 % interval at each lag; coloured where |t| ≥ 1.96. "
                f"The map uses the lag with the largest |t| ({lag} month{'s' if lag != 1 else ''}): tropical 70 hPa {12 * f['coef']:+.2f} months per °C "
                f"(p = {f['p']:.3f}). Picking the lag from the same data favours a signal; the neighbouring lags show how robust "
                "it is. Source: NASA GMAO MERRA-2 GMI replay; CPC RONI.")
    save(fig, out)
    return {"sig_cells": ns, "cells": nc, "lag": lag, "tropics_70hPa": f}


def fig_qbo(ds, out):
    sd = json.loads(ds.attrs["qbo_sd_ms"])
    per = ds.attrs["period"]
    fig = plt.figure(figsize=(12.6, 6.9), dpi=130)
    top = header(fig, f"MERRA-2 GMI · QBO signal in the mean age of stratospheric air · {per[:4]}–{per[9:13]}",
                 "Change in mean age per one standard deviation of westerly equatorial wind at 50 hPa (left, "
                 f"{sd['u50']:.0f} m/s) and at 30 hPa (right, {sd['u30']:.0f} m/s), fitted together with each other, the trend "
                 "and ENSO. Shaded only where significant (false-discovery rate 10 %); dashed: the climatological age.")
    lv = np.array([-1.5, -1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1, 1.5])
    res = {}
    for i, (nm, lab) in enumerate((("qbo50", "50 hPa westerly"), ("qbo30", "30 hPa westerly"))):
        ax = fig.add_axes([0.06 + 0.475 * i, 0.25, 0.40, top - 0.29])
        cax = fig.add_axes([0.28, 0.135, 0.45, 0.02]) if i == 0 else None
        ns, nc = section(ax, ds, nm, 12.0, lv, "younger ← months per standard deviation → older", fig, cax)
        ax.set_title(f"per +1 sd of {lab}", loc="left", fontsize=11, fontweight="bold", color=INK)
        res[nm] = {"sig_cells": ns, "cells": nc}
        if i == 1:
            ax.set_ylabel("")
    fits = json.loads(ds.attrs["box_fits"])
    f50 = fits["tropics_50hPa"]["fit"]["qbo50"]
    footer(fig, f"Cells significant: {res['qbo50']['sig_cells']} (50 hPa index) and {res['qbo30']['sig_cells']} (30 hPa index) of "
                f"{res['qbo50']['cells']}. Tropical 50 hPa: {12 * f50['coef']:+.2f} months per sd of 50 hPa westerly "
                f"(p = {f50['p']:.3g}). Westerly shear brings a secondary circulation that sinks at the equator and rises "
                "in the subtropics, so the equatorial air below westerlies is older. Source: NASA GMAO MERRA-2 GMI replay; "
                "CPC QBO indices (CDAS).")
    save(fig, out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(REPO))
    a = ap.parse_args()
    root = Path(a.out_root)
    ds = xr.open_dataset(REF)
    js = {"source": ds.attrs["source"], "period": ds.attrs["period"], "validation": json.loads(ds.attrs["validation"]),
          "record_checks": json.loads(ds.attrs["record_checks"]), "box_fits": json.loads(ds.attrs["box_fits"]),
          "enso_lag_months": int(ds.attrs["enso_lag_months"]), "enso_lag_scan": json.loads(ds.attrs["enso_lag_scan"]),
          "n_sig": json.loads(ds.attrs["n_sig"]), "n_cells": int(ds.attrs["n_cells"]),
          "regression": ds.attrs["regression"], "fdr": ds.attrs["fdr"]}
    for s in ("DJF", "MAM", "JJA", "SON", "ANN"):
        fig_clim(ds, s, root / "assets" / "sst" / f"aoa_clim_{s.lower()}.webp")
    js["cycle"] = fig_cycle(ds, root / "assets" / "sst" / "aoa_cycle.webp")
    js["trend"] = fig_trend(ds, root / "assets" / "sst" / "aoa_trend.webp")
    js["trend04"] = fig_trend(ds, root / "assets" / "sst" / "aoa_trend04.webp", "trend04")
    js["enso"] = fig_enso(ds, root / "assets" / "sst" / "aoa_enso.webp")
    js["qbo"] = fig_qbo(ds, root / "assets" / "sst" / "aoa_qbo.webp")
    (root / "assets" / "sst" / "data").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "sst" / "data" / "age_of_air.json").write_text(json.dumps(js, indent=1))
    print(json.dumps({k: js[k] for k in ("n_sig", "enso_lag_months")}, indent=1))


if __name__ == "__main__":
    main()
