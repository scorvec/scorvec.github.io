#!/usr/bin/env python3
"""Upper stratosphere, 100 -> 1 hPa, four models side by side: render side (2026-09-27; user: "let's add GDPS, GEFS,
GFS and GEOS-FP (allow the user to toggle between and compare the upper stratosphere)").

Reads what upperstrat_data.py left in --data (one npz per model that fetched; a missing model is skipped, never faked),
the GEOS FP analysis tail (reference seed + assets/sst/data/upperstrat_tail.json) and the MERRA-2 climatology
(reference/upperstrat_clim.nc, build_upperstrat_clim.py). Writes, under --out (assets/sst):

  upperstrat_ts_{model}_{hemi}.webp    u(60) and polar-cap T at 1, 5 and 10 hPa: tail + forecast against MERRA-2
  upperstrat_sec_{model}_{hemi}.webp   height-time section 100 -> 1 hPa of u(60) and the cap-T anomaly
  anim/upperstrat_{model}_{hemi}/F00..F10.webp + anim/upperstrat_maps_manifest.json
                                       polar maps of height and temperature at 5 and 1 hPa, days 0-10
  data/upperstrat.json                 the numbers: day-0 offsets from GEOS FP, day 0/5/10/last per level
model = gefs | gfs | geos | gdps | all ("all" overlays / tiles every model that is present).

Everything is RAW model output. Nothing is drift-corrected (no reforecast with these levels is used yet); the only
cross-model calibration shown is each model's day-0 offset from the GEOS FP 00Z analysis, printed on the figures.
Nothing above 1 hPa is drawn: GEFS and GDPS publish nothing higher, and the top levels of every model are damped.

    python upperstrat.py --data scripts/strat/data/upperstrat --out assets/sst
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
import matplotlib.dates as mdates                                            # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap                   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import upperstrat_data as UD                                                  # noqa: E402

CLIM = HERE / "reference" / "upperstrat_clim.nc"
ORDER = ["gefs", "gfs", "geos", "gdps"]
NAME = {"gefs": "GEFS", "gfs": "GFS", "geos": "GEOS FP", "gdps": "GDPS", "all": "All four models"}
# validated with the dataviz palette checker on the white figure ground (worst adjacent CVD dE 10.2, normal 17.0)
COL = {"gefs": "#c0392b", "gfs": "#2563b8", "geos": "#0e8f6e", "gdps": "#9a6700"}
INK, MUTED, GRID, CLIMC, BAND = "#1c2430", "#6f6b64", "#d9dde2", "#6f6b64", "#e6e8ec"
TS_LEVELS = [1.0, 5.0, 10.0]
TAIL_TS, TAIL_SEC, TAIL_SEC_ALL = 45, 30, 10
X_AHEAD = 35                                                   # every time series runs to day 35 so the toggle keeps its axes
TOPS = {"gefs": "output stops at 1 hPa, its top published level, not far below the model lid",
        "gfs": "output to 0.01 hPa; lid near 80 km",
        "gdps": "output stops at 1 hPa; lid at 0.1 hPa",
        "geos": "output to 0.1 hPa; lid at 0.01 hPa"}
U_LEV = np.arange(-60, 101, 10)
TA_LEV = np.arange(-20, 21, 2)
T_MAP = {5.0: np.arange(195, 286, 5), 1.0: np.arange(220, 306, 5)}   # fallback only: see map_scales()
T_STEPS = (2.0, 2.5, 4.0, 5.0, 10.0)                             # explicit ladder, never thirds of a unit
SCALE: dict = {}
Z_INT = {5.0: 200.0, 1.0: 250.0}                                # m


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------- inputs ---
def load_models(d: Path) -> dict:
    out = {}
    for m in ORDER:
        p = d / f"{m}.npz"
        if not p.exists():
            log(f"  {m}: no data this run")
            continue
        z = np.load(p, allow_pickle=False)
        init = pd.Timestamp(str(z["init"]))
        leads = np.asarray(z["leads_h"]) // 24
        out[m] = dict(init=init, levels=[float(x) for x in z["levels"]], days=leads.astype(int),
                      valid=pd.DatetimeIndex([init + pd.Timedelta(days=int(k)) for k in leads]),
                      members=[str(x) for x in z["members"]], u60=z["u60"].astype("float64"), capT=z["capT"].astype("float64"),
                      maps=z["maps"], note=str(z["note"]))
        log(f"  {m}: init {init:%Y-%m-%d}, {len(out[m]['members'])} member(s), days 0-{leads[-1]}, {len(out[m]['levels'])} levels")
    return out


def load_tail(live: Path) -> dict:
    h = UD.load_tail(live)
    if not h:
        return {}
    idx = pd.DatetimeIndex(sorted(h))
    return {hemi: {var: pd.DataFrame([h[k][hemi][var] for k in sorted(h)], index=idx, columns=UD.LEV_ALL)
                   for var in ("u60", "capT")} for hemi in UD.HEMIS}


class Clim:
    def __init__(self):
        self.ds = xr.open_dataset(CLIM).load() if CLIM.exists() else None
        self.attrs = self.ds.attrs if self.ds is not None else {}

    def at(self, hemi, var, stat, dates, L):
        if self.ds is None:
            return np.full(len(dates), np.nan)
        dates = pd.DatetimeIndex(dates)
        d = UD_doy(dates)
        return self.ds[stat].sel(hemi=hemi, var=var, lev=L).values[d]

    def grid(self, hemi, var, stat, dates, levels):
        return np.column_stack([self.at(hemi, var, stat, dates, L) for L in levels])      # (date, level)


def UD_doy(t: pd.DatetimeIndex) -> np.ndarray:
    d = t.dayofyear.values - 1
    return np.where(t.is_leap_year & (t.month > 2), d - 1, d)


def hemi_i(h):
    return 0 if h == "nh" else 1


def series(M, m, hemi, var, L, kind="det"):
    """(valid, values) of one model/hemisphere/variable/level; GEFS: kind = mean | p10 | p90 | control."""
    if L not in M[m]["levels"]:
        return None, None
    j = M[m]["levels"].index(L)
    a = M[m][var][hemi_i(hemi), :, :, j]                                   # (member, lead)
    if kind in ("det", "control"):
        v = a[0]
    elif kind == "mean":
        v = a.mean(0)
    else:
        v = np.percentile(a, 10 if kind == "p10" else 90, axis=0)
    return M[m]["valid"], v


def day0_offsets(M) -> dict:
    """model day 0 minus the GEOS FP 00Z analysis (its forecast's step 0), same init date only."""
    out = {}
    if "geos" not in M:
        return out
    g = M["geos"]
    for m in M:
        if m == "geos" or M[m]["init"] != g["init"]:
            continue
        out[m] = {}
        for hemi in UD.HEMIS:
            out[m][hemi] = {}
            for var in ("u60", "capT"):
                d = {}
                for L in M[m]["levels"]:
                    if L in g["levels"]:
                        a = M[m][var][hemi_i(hemi), :, 0, M[m]["levels"].index(L)]
                        b = g[var][hemi_i(hemi), 0, 0, g["levels"].index(L)]
                        d[f"{L:g}"] = round(float(a[0] - b), 2)                   # GEFS: the control
                out[m][hemi][var] = d
    return out


# ------------------------------------------------------------------------------------------------ plot helpers ---
def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")
    ax.tick_params(colors=INK, labelsize=9)


def date_axis(ax, x0, x1):
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))


def end_labels(ax, items, min_sep=0.075):
    """Direct labels at line ends, staggered so they never overlap: items = [(x, y, text, colour)]; the label text is
    ink, a short coloured tick carries the identity."""
    if not items:
        return
    tr = ax.transData + ax.transAxes.inverted()
    pts = sorted([(tr.transform((mdates.date2num(x), y))[1], x, y, t, c) for x, y, t, c in items if np.isfinite(y)])
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    for yy, (_, x, y, t, c) in zip(ys, pts):
        xa = tr.transform((mdates.date2num(x), y))[0]
        ax.annotate(t, xy=(xa, yy), xycoords="axes fraction", xytext=(8, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=8.5, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
        ax.annotate("", xy=(xa, yy), xycoords="axes fraction", xytext=(xa + 0.012, yy), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=c, lw=2.2))


def diverging(levels, cmap="RdBu_r"):
    """A BoundaryNorm colormap whose white sits at zero even when the levels are asymmetric."""
    base = plt.get_cmap(cmap)
    lo, hi = levels[0], levels[-1]
    span = max(-lo, hi)
    mids = 0.5 * (levels[:-1] + levels[1:])
    cols = [base(0.5 + 0.5 * m / span) for m in mids]
    cm = ListedColormap(cols)
    cm.set_under(base(0.5 + 0.5 * lo / span * 1.05)); cm.set_over(base(min(1.0, 0.5 + 0.5 * hi / span * 1.05)))
    return cm, BoundaryNorm(levels, cm.N)


def hemi_word(h):
    return "Northern" if h == "nh" else "Southern"


def ns(h):
    return "N" if h == "nh" else "S"


def fmt_off(v, unit):
    return "n/a" if v is None else f"{v:+.1f} {unit}"


# ------------------------------------------------------------------------------------------------ time series ---
def render_ts(M, tail, clim, offs, which, hemi, path):
    models = ORDER if which == "all" else [which]
    models = [m for m in models if m in M]
    if not models:
        return False
    init = max(M[m]["init"] for m in models)
    x0, x1 = init - pd.Timedelta(days=TAIL_TS), init + pd.Timedelta(days=X_AHEAD)
    span = pd.date_range(x0, x1)
    fig = plt.figure(figsize=(13.2, 10.6), dpi=110)
    gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.13, left=0.055, right=0.945, top=0.835, bottom=0.055)
    title = (f"{NAME[which]} · upper stratosphere, {hemi_word(hemi)} Hemisphere" if which == "all"
             else f"{NAME[which]} {init:%d %b %Y} 00Z · upper stratosphere, {hemi_word(hemi)} Hemisphere")
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=15, fontweight="bold", color=INK)
    if which == "all":
        l1 = "Runs: " + " · ".join(f"{NAME[m]} {M[m]['init']:%d %b} 00Z to day {int(M[m]['days'][-1])}" for m in models)
        l2 = "Raw model output, not drift-corrected; GEFS drawn as its 10–90 % member band and mean."
    else:
        l1 = (f"{len(M[which]['members'])} members: 10–90 % band, mean and control" if which == "gefs" else "Deterministic run") \
             + f", to day {int(M[which]['days'][-1])}" + (f" · {M[which]['note']}" if M[which]["note"] else "")
        l2 = f"Raw model output, not drift-corrected · {NAME[which]}: {TOPS[which]}."
    l3 = (f"Zonal-mean wind at 60°{ns(hemi)} (left) and polar-cap temperature, 60–90°{ns(hemi)} (right). "
          f"Black: GEOS FP analyses, daily means. Grey: MERRA-2 {period(clim, hemi)}, mean and 10–90 % range.")
    fig.text(0.055, 0.952, "\n".join((l1, l2, l3)), fontsize=10, color=MUTED, va="top", linespacing=1.45)
    handles = {}
    for r, L in enumerate(TS_LEVELS):
        for c, var in enumerate(("u60", "capT")):
            ax = fig.add_subplot(gs[r, c]); style(ax)
            lo, hi = clim.at(hemi, var, "p10", span, L), clim.at(hemi, var, "p90", span, L)
            handles["band"] = ax.fill_between(span, lo, hi, color=BAND, lw=0, label="MERRA-2 10–90 %")
            handles["clim"], = ax.plot(span, clim.at(hemi, var, "mean", span, L), color=CLIMC, lw=1.4, ls="--",
                                       label="MERRA-2 mean")
            if tail:
                s = tail[hemi][var][L]
                s = s[(s.index >= x0) & (s.index < init + pd.Timedelta(days=1))]
                if len(s):
                    handles["obs"], = ax.plot(s.index, s.values, color=INK, lw=2.0, label="GEOS FP analyses")
            labs = []
            for m in models:
                if m == "gefs":
                    v, p10 = series(M, m, hemi, var, L, "p10")
                    if v is None:
                        continue
                    _, p90 = series(M, m, hemi, var, L, "p90")
                    _, mean = series(M, m, hemi, var, L, "mean")
                    _, ctl = series(M, m, hemi, var, L, "control")
                    handles["gefs_band"] = ax.fill_between(v, p10, p90, color=COL[m], alpha=0.17, lw=0, label="GEFS members 10–90 %")
                    handles[m], = ax.plot(v, mean, color=COL[m], lw=2.4, label="GEFS mean")
                    if which == "gefs":
                        handles["ctl"], = ax.plot(v, ctl, color=COL[m], lw=1.1, ls=(0, (4, 2)), label="GEFS control")
                    labs.append((v[-1], mean[-1], NAME[m], COL[m]))
                else:
                    v, y = series(M, m, hemi, var, L)
                    if v is None:
                        continue
                    handles[m], = ax.plot(v, y, color=COL[m], lw=2.2, label=NAME[m])
                    labs.append((v[-1], y[-1], NAME[m], COL[m]))
            ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":")
            if var == "u60":
                ax.axhline(0, color=INK, lw=0.8)
            date_axis(ax, x0, x1)
            unit = "m/s" if var == "u60" else "K"
            ax.set_ylabel(unit, fontsize=9.5, color=INK)
            what = f"u at 60°{ns(hemi)}" if var == "u60" else f"polar-cap T, 60–90°{ns(hemi)}"
            ax.set_title(f"{L:g} hPa · {what}", loc="left", fontsize=11.5, fontweight="bold", color=INK)
            # day-0 offset from the GEOS FP analysis
            txt = []
            for m in models:
                if m in offs and f"{L:g}" in offs[m][hemi][var]:
                    txt.append(f"{NAME[m]} {offs[m][hemi][var][f'{L:g}']:+.1f}")
            if txt:
                ax.text(0.01, 0.03, "day 0 minus GEOS FP: " + ", ".join(txt) + f" {unit}", transform=ax.transAxes,
                        fontsize=8, color=MUTED, va="bottom", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8))
            ys = [l.get_ydata() for l in ax.lines if len(np.atleast_1d(l.get_ydata())) > 2]
            yy = np.concatenate([np.asarray(y, float) for y in ys] + [lo, hi])
            yy = yy[np.isfinite(yy)]
            if len(yy):
                pad = 0.08 * (yy.max() - yy.min() + 1)
                ax.set_ylim(yy.min() - pad, yy.max() + pad)
            if which == "all" and r == 0:
                end_labels(ax, labs)                               # after the limits: it works in axes fractions
    order = [k for k in ["obs", "clim", "band"] + ORDER + ["gefs_band", "ctl"] if k in handles]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc="upper left",
               bbox_to_anchor=(0.05, 0.893), ncol=min(len(order), 8), fontsize=9, frameon=False, handlelength=2.2)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    return True


def period(clim, hemi):
    a = clim.attrs.get(hemi, "")
    import re
    m = re.search(r"(\d{4})-(\d{4})", a)
    return f"{m.group(1)}–{m.group(2)}" if m else "climatology"


# ----------------------------------------------------------------------------------------------- section (z-t) ---
def section_panel(ax_u, ax_t, M, m, tail, clim, hemi, tail_days, show_y=True, cm_u=None, cm_t=None):
    init = M[m]["init"]
    lv = M[m]["levels"]
    valid = M[m]["valid"]
    a_u = M[m]["u60"][hemi_i(hemi)].mean(0)                                  # (lead, level); GEFS: ensemble mean
    a_t = M[m]["capT"][hemi_i(hemi)].mean(0) - clim.grid(hemi, "capT", "mean", valid, lv)
    lp = -np.log(np.array(lv))
    xs = mdates.date2num(valid)
    ms = []
    if tail:
        tu = tail[hemi]["u60"]; tt = tail[hemi]["capT"]
        sel = (tu.index >= init - pd.Timedelta(days=tail_days)) & (tu.index < init)
        if sel.sum() >= 2:
            xt = mdates.date2num(tu.index[sel])
            lpt = -np.log(np.array(UD.LEV_ALL))
            ms.append(ax_u.contourf(xt, lpt, tu.values[sel].T, levels=U_LEV, cmap=cm_u[0], norm=cm_u[1], extend="both"))
            ax_u.contour(xt, lpt, tu.values[sel].T, levels=[0], colors=INK, linewidths=1.2)
            an = tt.values[sel] - clim.grid(hemi, "capT", "mean", tu.index[sel], UD.LEV_ALL)
            ax_t.contourf(xt, lpt, an.T, levels=TA_LEV, cmap=cm_t[0], norm=cm_t[1], extend="both")
    cu = ax_u.contourf(xs, lp, a_u.T, levels=U_LEV, cmap=cm_u[0], norm=cm_u[1], extend="both")
    ax_u.contour(xs, lp, a_u.T, levels=[0], colors=INK, linewidths=1.2)
    ct = ax_t.contourf(xs, lp, a_t.T, levels=TA_LEV, cmap=cm_t[0], norm=cm_t[1], extend="both")
    x0 = mdates.date2num(init - pd.Timedelta(days=tail_days))
    for ax in (ax_u, ax_t):
        ax.set_xlim(x0, xs[-1])
        ax.set_ylim(-np.log(100.0), -np.log(1.0))
        ax.axvline(xs[0], color=INK, lw=1.2)
        ticks = [100, 50, 30, 20, 10, 5, 3, 2, 1]
        ax.set_yticks(-np.log(ticks)); ax.set_yticklabels([f"{t:g}" for t in ticks] if show_y else [])
        ax.tick_params(labelsize=8.5, colors=INK)
        # the model's own levels, as ticks on the right edge: anything between them is interpolated
        for L in lv:
            ax.plot([1.0, 1.012], [(-np.log(L) + np.log(100)) / (np.log(100) - np.log(1))] * 2, transform=ax.transAxes,
                    color=INK, lw=1.0, clip_on=False)
        ndays = xs[-1] - x0
        # ~4 labels a panel: in the four-model compare the 10-day panels are narrow, and 3-day ticks ran the
        # last label of one panel into the first of the next ("Oct 7Sep 17")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(3, int(round(ndays / 4.0)))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return cu, ct


def render_sec(M, tail, clim, which, hemi, path):
    models = [m for m in (ORDER if which == "all" else [which]) if m in M]
    if not models:
        return False
    cm_u, cm_t = diverging(U_LEV), diverging(TA_LEV)
    if which == "all":
        fig = plt.figure(figsize=(15.0, 8.6), dpi=110)
        wr = [TAIL_SEC_ALL + int(M[m]["days"][-1]) for m in models]
        gs = fig.add_gridspec(2, len(models), width_ratios=wr, hspace=0.26, wspace=0.12, left=0.045, right=0.915,
                              top=0.845, bottom=0.07)
        for k, m in enumerate(models):
            au, at = fig.add_subplot(gs[0, k]), fig.add_subplot(gs[1, k])
            cu, ct = section_panel(au, at, M, m, tail, clim, hemi, TAIL_SEC_ALL, show_y=(k == 0), cm_u=cm_u, cm_t=cm_t)
            lab = f"{NAME[m]}{' mean' if m == 'gefs' else ''} · {M[m]['init']:%d %b} 00Z"
            au.set_title(lab, loc="left", fontsize=11, fontweight="bold", color=COL[m])
        title = f"All four models · u at 60°{ns(hemi)} and polar-cap temperature anomaly, 100 → 1 hPa, {hemi_word(hemi)} Hemisphere"
        sub = (f"Last {TAIL_SEC_ALL} days = GEOS FP analyses (left of the black line), then each model to its own range. "
               f"Same colour scale in every panel. Raw output; T anomaly against MERRA-2 {period(clim, hemi)}.")
    else:
        m = which
        fig = plt.figure(figsize=(13.2, 8.8), dpi=110)
        gs = fig.add_gridspec(2, 1, hspace=0.25, left=0.07, right=0.9, top=0.855, bottom=0.07)
        au, at = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        cu, ct = section_panel(au, at, M, m, tail, clim, hemi, TAIL_SEC, cm_u=cm_u, cm_t=cm_t)
        au.set_title(f"Zonal-mean wind at 60°{ns(hemi)}{', ensemble mean' if m == 'gefs' else ''} (black contour: u = 0)",
                     loc="left", fontsize=11.5, fontweight="bold", color=INK)
        at.set_title(f"Polar-cap temperature 60–90°{ns(hemi)}, anomaly from MERRA-2 {period(clim, hemi)}",
                     loc="left", fontsize=11.5, fontweight="bold", color=INK)
        for ax in (au, at):
            ax.set_ylabel("hPa", fontsize=9.5, color=INK)
        title = (f"{NAME[m]} {M[m]['init']:%d %b %Y} 00Z · u at 60°{ns(hemi)} and polar-cap temperature, 100 → 1 hPa, "
                 f"{hemi_word(hemi)} Hemisphere")
        sub = (f"Left of the black line: GEOS FP analyses, last {TAIL_SEC} days; right: {NAME[m]} to day {int(M[m]['days'][-1])}"
               f"{' (ensemble mean of ' + str(len(M[m]['members'])) + ' members)' if m == 'gefs' else ''}.\n"
               f"Ticks on the right edge = the model's output levels (between them the shading is interpolated). "
               f"Raw output, not drift-corrected · {NAME[m]}: {TOPS[m]}." + (f" {M[m]['note'].capitalize()}." if M[m]["note"] else ""))
    fig.suptitle(title, x=0.045 if which == "all" else 0.07, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.045 if which == "all" else 0.07, 0.95, sub, fontsize=10, color=MUTED, va="top", linespacing=1.5)
    right = 0.925 if which == "all" else 0.915
    for cs, (y0, y1), lab, lev in ((cu, (0.49, 0.845), "u, m/s", U_LEV), (ct, (0.07, 0.425), "T anomaly, K", TA_LEV)):
        cax = fig.add_axes([right, y0 + 0.02, 0.012, y1 - y0 - 0.04])
        cb = fig.colorbar(cs, cax=cax, ticks=lev[::2])
        cb.ax.tick_params(labelsize=8); cb.set_label(lab, fontsize=9)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------------------------------- maps ---
_NAT = {}


def native(north):
    if north not in _NAT:
        import geos_pv
        lat = UD.LAT_N if north else -UD.LAT_N
        _NAT[north] = geos_pv.Native(north, lat, UD.LON_1, n=360)
    return _NAT[north]


def map_scales(M):
    """One temperature scale per (hemisphere, level) for THIS run, shared by every model and every frame, so a toggle
    between models (or a step through the loop) never changes the colours under you; set from the 1st-99th
    percentiles of all models' fields together, on the first step of the ladder that gives at most 18 bins."""
    for hi, hemi in enumerate(UD.HEMIS):
        for li, L in enumerate(UD.MAP_LEVELS):
            v = np.concatenate([M[m]["maps"][hi, :, 1, li].ravel() for m in M])
            v = v[np.isfinite(v)]
            if not len(v):
                SCALE[(hemi, L)] = T_MAP[L]
                continue
            lo, hi_ = np.percentile(v, [1, 99])
            for st in T_STEPS:
                a, b = np.floor(lo / st) * st, np.ceil(hi_ / st) * st
                if (b - a) / st <= 18:
                    break
            SCALE[(hemi, L)] = np.arange(a, b + st / 2, st)


def cbar_ticks(cb, lv):
    tk = lv[::2] if len(lv) > 10 else lv
    cb.set_ticks(tk); cb.set_ticklabels([f"{t:g}" for t in tk])


def map_panel(fig, rect, north, Z, T, L, title, title_col=INK):
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    import matplotlib.path as mpath
    nat = native(north)
    ax = fig.add_axes(rect, projection=nat.proj)
    ax.set_xlim(nat.x[0], nat.x[-1]); ax.set_ylim(nat.y[0], nat.y[-1])
    th = np.linspace(0, 2 * np.pi, 240)
    ax.set_boundary(mpath.Path(np.column_stack([np.sin(th), np.cos(th)]) * 0.5 + 0.5), transform=ax.transAxes)
    ax.spines["geo"].set_edgecolor("#9aa3ad")
    ax.set_title(title, fontsize=10, fontweight="bold", color=title_col, pad=3)
    lv = SCALE.get(("nh" if north else "sh", L), T_MAP[L])
    cmap = plt.get_cmap("OrRd")
    cm = ListedColormap(cmap(np.linspace(0.04, 0.88, len(lv) + 1)))
    nm = BoundaryNorm(lv, cm.N, extend="both")
    if Z is None:
        ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=MUTED)
        ax.axis("off")
        return None
    t = nat(T); z = nat(Z)
    m = ax.contourf(nat.x, nat.y, t, levels=lv, cmap=cm, norm=nm, extend="both")
    zl = np.arange(np.floor(np.nanmin(z) / Z_INT[L]) * Z_INT[L], np.nanmax(z) + Z_INT[L], Z_INT[L])
    cs = ax.contour(nat.x, nat.y, z, levels=zl, colors="#14202e", linewidths=0.75)
    try:
        ax.clabel(cs, cs.levels[::2], fmt=lambda v: f"{v / 10:.0f}", fontsize=6.5, inline=True, inline_spacing=2)
    except Exception:                                                        # noqa: BLE001
        pass
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="#4a5563")
    ax.gridlines(crs=ccrs.PlateCarree(), lw=0.3, color="#9aa3ad", ylocs=[60 if north else -60], xlocs=[], draw_labels=False)
    return m


def map_fields(M, m, hemi, valid):
    """(Z, T) per MAP level for the model's map day that verifies at `valid`, or None."""
    k = (valid - M[m]["init"]).days
    if k < 0 or k >= M[m]["maps"].shape[1]:
        return None
    a = M[m]["maps"][hemi_i(hemi), k]                                        # (var z/t, level, lat, lon)
    if not np.isfinite(a).any():
        return None
    return {L: (a[0, li].astype("float64"), a[1, li].astype("float64")) for li, L in enumerate(UD.MAP_LEVELS)}


def render_maps(M, which, hemi, out_anim: Path, base_init):
    north = hemi == "nh"
    models = [m for m in (ORDER if which == "all" else [which]) if m in M]
    if not models:
        return None
    d = out_anim / f"upperstrat_{which}_{hemi}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("F*.webp"):
        old.unlink()
    init = base_init if which == "all" else M[which]["init"]
    frames = []
    for day in range(UD.MAP_DAYS):
        valid = init + pd.Timedelta(days=day)
        if which == "all":
            fig = plt.figure(figsize=(14.4, 8.3), dpi=100)
            w = 0.905 / len(models)
            ms = None
            for k, m in enumerate(models):
                f = map_fields(M, m, hemi, valid)
                for r, L in enumerate(UD.MAP_LEVELS):
                    ax_rect = [0.02 + k * w, 0.475 - r * 0.43, w * 0.97, 0.40]
                    lead = (valid - M[m]["init"]).days
                    ttl = f"{NAME[m]} · {L:g} hPa · {'analysis' if lead == 0 else f'day {lead}'}"
                    mm = map_panel(fig, ax_rect, north, None if f is None else f[L][0], None if f is None else f[L][1], L, ttl,
                                   title_col=COL[m])
                    if mm is not None:
                        ms = ms or {}
                        ms[L] = mm
            if not ms:
                plt.close(fig)
                continue
            for r, L in enumerate(UD.MAP_LEVELS):
                if L in ms:
                    cax = fig.add_axes([0.935, 0.475 - r * 0.43 + 0.03, 0.009, 0.34])
                    cb = fig.colorbar(ms[L], cax=cax); cb.ax.tick_params(labelsize=7.5)
                    cbar_ticks(cb, SCALE.get((hemi, L), T_MAP[L]))
                    cb.set_label(f"T {L:g} hPa, K", fontsize=8)
            fig.suptitle(f"Height (dam) and temperature at 5 and 1 hPa · {hemi_word(hemi)} Hemisphere · valid {valid:%a %d %b %Y} 00Z",
                         x=0.03, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
            fig.text(0.03, 0.948, "Four models at the same valid time; GEFS = its control run. Raw output, one colour scale "
                     f"for every panel and frame of this run; contours every {Z_INT[5.0] / 10:.0f} dam (5 hPa) and {Z_INT[1.0] / 10:.0f} dam (1 hPa).",
                     fontsize=9.5, color=MUTED)
        else:
            f = map_fields(M, which, hemi, valid)
            if f is None:
                continue
            fig = plt.figure(figsize=(10.4, 5.9), dpi=100)
            ms = {}
            for c, L in enumerate(UD.MAP_LEVELS):
                ms[L] = map_panel(fig, [0.015 + c * 0.5, 0.13, 0.47, 0.72], north, f[L][0], f[L][1], L,
                                  f"{L:g} hPa · height (dam) and temperature")
                cax = fig.add_axes([0.1 + c * 0.5, 0.075, 0.30, 0.022])
                cb = fig.colorbar(ms[L], cax=cax, orientation="horizontal")
                cbar_ticks(cb, SCALE.get((hemi, L), T_MAP[L]))
                cb.ax.tick_params(labelsize=7.5); cb.set_label(f"temperature at {L:g} hPa, K", fontsize=8, labelpad=1)
            lead = "analysis" if day == 0 else f"day {day}"
            src = f"{NAME[which]} control" if which == "gefs" else NAME[which]
            fig.suptitle(f"{src} {M[which]['init']:%d %b} 00Z · {hemi_word(hemi)} Hemisphere, 5 and 1 hPa — {lead}, "
                         f"valid {valid:%a %d %b} 00Z", x=0.015, y=0.975, ha="left", fontsize=12, fontweight="bold", color=INK)
            fig.text(0.015, 0.915, f"Raw output · contours every {Z_INT[5.0] / 10:.0f} dam (5 hPa) and {Z_INT[1.0] / 10:.0f} dam "
                     f"(1 hPa) · {TOPS[which]}", fontsize=9, color=MUTED)
        fp = d / f"F{day:02d}.webp"
        fig.savefig(fp, format="webp", facecolor="white", pil_kwargs={"quality": 84, "method": 6})
        plt.close(fig)
        frames.append({"idx": len(frames), "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": ("analysis" if day == 0 else f"day {day}") + f" · {valid:%a %d %b}"})
    return frames


# -------------------------------------------------------------------------------------------------------- main ---
def summary(M, tail, clim, offs):
    doc = {"source": "upperstrat.py: GEFS/GFS (NOAA S3), GDPS (MSC Datamart), GEOS FP (NASA GMAO NCCS OPeNDAP); "
                     "MERRA-2 climatology; raw model output, not drift-corrected",
           "levels_shown": TS_LEVELS, "climatology": {h: clim.attrs.get(h) for h in UD.HEMIS},
           "day0_minus_geosfp": offs, "models": {}}
    for m in M:
        g = M[m]
        e = {"init": f"{g['init']:%Y-%m-%d} 00Z", "members": len(g["members"]), "last_day": int(g["days"][-1]),
             "levels": g["levels"], "top": TOPS[m], "note": g["note"], "hemis": {}}
        for hemi in UD.HEMIS:
            e["hemis"][hemi] = {}
            for var in ("u60", "capT"):
                e["hemis"][hemi][var] = {}
                for L in TS_LEVELS:
                    if L not in g["levels"]:
                        continue
                    row = {}
                    for day in sorted({0, 5, 10, int(g["days"][-1])}):
                        if day > g["days"][-1]:
                            continue
                        a = g[var][hemi_i(hemi), :, day, g["levels"].index(L)]
                        vd = g["init"] + pd.Timedelta(days=day)
                        c = clim.at(hemi, var, "mean", [vd], L)[0]
                        r = {"valid": f"{vd:%Y-%m-%d}", "value": round(float(a.mean() if m == "gefs" else a[0]), 2),
                             "clim_mean": round(float(c), 2) if np.isfinite(c) else None}
                        if m == "gefs":
                            r.update(p10=round(float(np.percentile(a, 10)), 2), p90=round(float(np.percentile(a, 90)), 2),
                                     control=round(float(a[0]), 2), frac_below_0=round(float((a < 0).mean()), 3))
                        row[f"d{day}"] = r
                    e["hemis"][hemi][var][f"{L:g}"] = row
        doc["models"][m] = e
    if tail:
        last = tail["nh"]["u60"].index[-1]
        doc["analysis_tail"] = {"last_day": f"{last:%Y-%m-%d}", "days": len(tail["nh"]["u60"]),
                                **{h: {v: {f"{L:g}": round(float(tail[h][v][L].iloc[-1]), 2) for L in TS_LEVELS}
                                       for v in ("u60", "capT")} for h in UD.HEMIS}}
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "upperstrat"))
    ap.add_argument("--out", default=str(REPO / "assets" / "sst"))
    ap.add_argument("--tail", default=str(REPO / "assets" / "sst" / "data" / "upperstrat_tail.json"))
    ap.add_argument("--only", default="", help="comma list of ts,sec,maps (default all)")
    a = ap.parse_args()
    out = Path(a.out); out = out if out.is_absolute() else REPO / out
    only = set(a.only.split(",")) if a.only else {"ts", "sec", "maps"}
    M = load_models(Path(a.data))
    if not M:
        log("no model data at all; nothing rendered")
        return 1
    tail = load_tail(Path(a.tail))
    clim = Clim()
    offs = day0_offsets(M)
    (out / "data").mkdir(parents=True, exist_ok=True)
    whiches = [m for m in ORDER if m in M] + ["all"]
    for w in whiches:
        for hemi in UD.HEMIS:
            if "ts" in only and render_ts(M, tail, clim, offs, w, hemi, out / f"upperstrat_ts_{w}_{hemi}.webp"):
                log(f"  ts  {w} {hemi}")
            if "sec" in only and render_sec(M, tail, clim, w, hemi, out / f"upperstrat_sec_{w}_{hemi}.webp"):
                log(f"  sec {w} {hemi}")
    if "maps" in only:
        mpath = out / "anim" / "upperstrat_maps_manifest.json"
        man = json.loads(mpath.read_text()) if mpath.exists() else {"regions": {}}
        base_init = max(M[m]["init"] for m in M)
        map_scales(M)
        man.update({"ver": f"{base_init:%Y%m%d}00", "days": UD.MAP_DAYS, "selectorLabel": "Model",
                    "default": "upperstrat_all_nh"})
        for w in whiches:
            for hemi in UD.HEMIS:
                fr = render_maps(M, w, hemi, out / "anim", base_init)
                if fr:
                    man["regions"][f"upperstrat_{w}_{hemi}"] = {"label": f"{NAME[w]}, {hemi_word(hemi)} Hemisphere",
                                                                "n_frames": len(fr), "frames": fr}
                    log(f"  maps {w} {hemi}: {len(fr)} frames")
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(man))
    (out / "data" / "upperstrat.json").write_text(json.dumps(summary(M, tail, clim, offs), separators=(",", ":")))
    log(f"wrote {out / 'data' / 'upperstrat.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
