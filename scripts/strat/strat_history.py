#!/usr/bin/env python3
"""Render the "Stratosphere history" items of circulation.html from the committed reference
(scripts/strat/reference/strat_history.nc + strat_history_events.json, built once on the laptop by
build_strat_history.py). Static products: nothing here changes unless the reference is rebuilt, so the
workflow (strat-history.yml) runs on a change to this script or the reference, or on dispatch.

Outputs (assets/sst/):
  data/strat_history.json            every winter's daily series + climatology + catalogue (interactive chart, table)
  strat_hist_timeline.webp           SSWs and strong-vortex events, winter by winter, with ENSO and QBO
  strat_hist_gallery.webp            the 10 hPa vortex at every SSW (NCEP R1 height)
  strat_hist_drip_<set>.webp         polar-cap height anomaly, time-height, around each kind of event
  strat_hist_drip_{cmip6,m2u}_<set>.webp  the same in u(60N): thousands of CMIP6 events, and MERRA-2 drawn alike
                                     (reference/cmip6_drip.nc from build_cmip6_drip.py)
  strat_hist_follows_<set>_<win>.webp  ERA5 2 m temperature and 500 hPa height after each kind of event
  strat_hist_ao.webp                 the Arctic Oscillation after SSWs and strong-vortex events
  strat_hist_regions.webp            how often each region ran cold after each kind of event
  strat_hist_nao.webp                the NAO after SSWs and strong-vortex events (CPC daily NAO)
  strat_hist_{follows_cmip6_<set>_<win>,ao_cmip6,nao_cmip6,regions_cmip6}.webp  the same questions in 9 CMIP6 models
                                     (reference/cmip6_follow.nc from build_cmip6_follow.py)
  strat_hist_rates.webp              SSW odds by ENSO, QBO and the solar cycle (sunspot number)

    python scripts/strat/strat_history.py [--out-root .]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
matplotlib.rcParams["hatch.color"] = "#8a8f96"
matplotlib.rcParams["hatch.linewidth"] = 0.5

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REF_NC = HERE / "reference" / "strat_history.nc"
REF_JS = HERE / "reference" / "strat_history_events.json"

INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
WARM, COOL, SV_C = "#b0402a", "#2b5f8a", "#2b5f8a"
SET_LABEL = {"ssw_all": "sudden warmings", "ssw_deep": "deep sudden warmings", "ssw_shallow": "shallow sudden warmings",
             "ssw_split": "vortex splits", "ssw_disp": "vortex displacements", "sv": "strong-vortex events"}
SET_SHORT = {"ssw_all": "All SSWs", "ssw_deep": "Deep", "ssw_shallow": "Shallow", "ssw_split": "Splits",
             "ssw_disp": "Displacements", "sv": "Strong vortex"}
WIN_LABEL = {"d1_30": "days 1–30", "d31_60": "days 31–60"}
SRC = "MERRA-2 zonal means 1980–2026 (NASA GMAO)"
ANALOGS = ["1982/83", "1997/98", "2009/10", "2015/16", "2023/24"]


def fig_header(fig, title, sub, x=0.045, y=None):
    """Title and a subtitle wrapped to the figure width; positions in inches so every figure size agrees.
    Returns the figure fraction just below the subtitle."""
    import textwrap
    W, H = fig.get_figwidth(), fig.get_figheight()
    y = 1 - 0.28 / H if y is None else y
    fig.text(x, y, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    lines = textwrap.wrap(sub, width=int((1 - 2 * x) * W * 12.6))
    fig.text(x, y - 0.36 / H, "\n".join(lines), ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)
    return y - (0.36 + 0.19 * len(lines)) / H


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"  {out.name}  {out.stat().st_size / 1e3:.0f} KB")


def style(ax):
    ax.tick_params(colors=INK, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fmt_date(s):
    return pd.Timestamp(s).strftime("%-d %b %Y")


# ---------------------------------------------------------------- JSON for the interactive chart + table
def write_json(D, E, out: Path):
    t = pd.DatetimeIndex(D.time.values)
    md = [f"{m:02d}-{d:02d}" for m, d in [(x.month, x.day) for x in pd.date_range("2001-10-01", "2002-05-31")]]
    series = {"u10": D.u10.to_series(), "t10": D.t10.to_series(), "z100s": D.z100s.to_series()}
    out_vars = {}
    for k, s in series.items():
        rows = {}
        for w in E["winters"]:
            y = w["y"]
            seg = s[f"{y}-10-01":f"{y + 1}-05-31"]
            seg = seg[~((seg.index.month == 2) & (seg.index.day == 29))]
            rows[w["winter"]] = [None if not np.isfinite(v) else round(float(v), 2 if k == "z100s" else 1) for v in seg.values]
        A = np.array([[np.nan if v is None else v for v in r] for r in rows.values() if len(r) == len(md)])
        clim = {q: [round(float(v), 2) for v in np.nanpercentile(A, qq, axis=0)] for q, qq in
                (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))}
        clim["min"] = [round(float(v), 2) for v in np.nanmin(A, 0)]
        clim["max"] = [round(float(v), 2) for v in np.nanmax(A, 0)]
        out_vars[k] = {"clim": clim, "winters": rows}
    out_vars["u10"].update(label="Zonal-mean wind at 60°N, 10 hPa", units="m/s", zero=True)
    out_vars["t10"].update(label="Polar-cap temperature at 10 hPa (65–90°N)", units="K", zero=False)
    out_vars["z100s"].update(label="Polar-cap height anomaly at 100 hPa (65–90°N)", units="σ", zero=True)
    j = {"built": E["built"], "period": E["period"], "days": md, "vars": out_vars, "winters": E["winters"],
         "ssw": E["ssw"], "strong_vortex": E["strong_vortex"], "analogs": ANALOGS, "current": E.get("current"),
         "source": SRC + "; NCEP R1 10 hPa height (split/displacement); CPC daily AO; CPC RONI; CPC QBO 50 hPa; SILSO sunspot number via SWPC"}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(j, open(out, "w"), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    print(f"  {out.name}  {out.stat().st_size / 1e3:.0f} KB")


# ---------------------------------------------------------------- timeline
def timeline(E, out):
    W = E["winters"]
    fig = plt.figure(figsize=(13.4, 7.0), dpi=130)
    n_ssw = sum(w["n_ssw"] for w in W)
    fig_header(fig, "Every northern winter since 1980: sudden warmings and strong-vortex events",
               f"{n_ssw} major SSWs in {len(W)} winters ({sum(1 for w in W if w['n_ssw'])} winters with at least one) and "
               f"{len(E['strong_vortex'])} strong-vortex events. Columns tinted by ENSO (CPC RONI, DJF); "
               f"QBO at 50 hPa (Oct–Nov) along the bottom. {SRC}.")
    ax = fig.add_axes([0.06, 0.24, 0.92, 0.58])
    x = np.arange(len(W))
    for i, w in enumerate(W):
        c = {"El Niño": "#f3d7c9", "La Niña": "#d3e2ef"}.get(w["enso"])
        if c:
            ax.axvspan(i - 0.5, i + 0.5, color=c, lw=0, zorder=0)
    base = lambda d, y: (pd.Timestamp(d) - pd.Timestamp(y, 11, 1)).days
    for e in E["ssw"]:
        y = int(e["winter"][:4]); i = y - W[0]["y"]
        if not 0 <= i < len(W):
            continue
        yy = base(e["date"], y)
        mk = "D" if e["type"] == "split" else "o"
        if e.get("marginal"):
            ax.scatter(i, yy, s=60, marker=mk, facecolor="white", edgecolor="#9aa3ad", lw=1.2, zorder=4)
            continue
        deep = e["depth"] == "deep"
        ax.scatter(i, yy, s=95, marker=mk, facecolor=WARM if deep else "white", edgecolor=WARM, lw=1.8, zorder=5)
    for s in E["strong_vortex"]:
        d = pd.Timestamp(s["date"]); y = d.year if d.month >= 7 else d.year - 1; i = y - W[0]["y"]
        if 0 <= i < len(W):
            ax.scatter(i, base(s["date"], y), s=70, marker="^", color=SV_C, zorder=4)
    ticks = [pd.Timestamp(2001, m, 1) for m in (11, 12)] + [pd.Timestamp(2002, m, 1) for m in (1, 2, 3, 4)]
    ax.set_yticks([(d - pd.Timestamp(2001, 11, 1)).days for d in ticks])
    ax.set_yticklabels([d.strftime("1 %b") for d in ticks])
    ax.set_ylim(152, -3)
    ax.set_xlim(-0.6, len(W) - 0.4)
    ax.set_xticks(x[::2]); ax.set_xticklabels([w["winter"] for w in W][::2], rotation=60, ha="right", fontsize=8.5)
    style(ax); ax.grid(axis="x", visible=False)
    for i, w in enumerate(W):
        ax.text(i, 1.03, w["qbo"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=COOL if w["qbo"] == "W" else WARM, fontweight="bold")
    ax.text(-0.012, 1.03, "QBO", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    h = [plt.Line2D([], [], marker="o", ls="", ms=9, mfc=WARM, mec=WARM, label="displacement, deep"),
         plt.Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=WARM, mew=1.8, label="displacement, shallow"),
         plt.Line2D([], [], marker="D", ls="", ms=8, mfc=WARM, mec=WARM, label="split, deep"),
         plt.Line2D([], [], marker="D", ls="", ms=8, mfc="white", mec=WARM, mew=1.8, label="split, shallow"),
         plt.Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="#9aa3ad", mew=1.2, label="unconfirmed (one reanalysis)"),
         plt.Line2D([], [], marker="^", ls="", ms=9, color=SV_C, label="strong-vortex event"),
         plt.Rectangle((0, 0), 1, 1, color="#f3d7c9", label="El Niño winter"),
         plt.Rectangle((0, 0), 1, 1, color="#d3e2ef", label="La Niña winter")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.52, -0.005))
    save(fig, out)


# ---------------------------------------------------------------- gallery
def polar_axes(fig, rect, lat0=30, central=0.0):
    import cartopy.crs as ccrs
    ax = fig.add_axes(rect, projection=ccrs.NorthPolarStereo(central_longitude=central))
    ax.set_extent([-180, 180, lat0, 90], ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 120)
    ax.set_boundary(mpath.Path(np.vstack([np.sin(th), np.cos(th)]).T * 0.5 + 0.5), transform=ax.transAxes)
    return ax


def cyclic(f, lon):
    return np.concatenate([f, f[..., :1]], -1), np.concatenate([lon, [lon[0] + 360]])


def gallery(D, E, out):
    import cartopy.crs as ccrs
    ev = E["ssw"]
    n = len(ev); nc = 6; nr = int(np.ceil(n / nc))
    fig = plt.figure(figsize=(13.4, 2.25 * nr + 1.2), dpi=120)
    H = fig.get_figheight()
    below = fig_header(fig, "The polar vortex at every sudden warming since 1980",
               f"10 hPa geopotential height (NCEP R1) on the day, from −3 to +7, when the vortex was most evenly split or "
               f"furthest off the pole. Bold contour = the vortex edge ({E['vortex_edge_m'] / 1000:.2f} km, the DJFM mean at 60°N). "
               f"Split = two separate low centres, the smaller at least a quarter of the larger. Grey = unconfirmed reversal.")
    lev = np.arange(28.8, 31.81, 0.15)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.BoundaryNorm(lev, cmap.N, extend="both")
    lat, lon = D.mlat.values, D.mlon.values
    top = below - 0.12 / H
    cw, ch = 0.94 / nc, (top - 0.05) / nr
    pc = ccrs.PlateCarree()
    for k, e in enumerate(ev):
        r, c = divmod(k, nc)
        ax = polar_axes(fig, [0.03 + c * cw + 0.004, top - (r + 1) * ch + 0.02, cw - 0.008, ch - 0.035])
        f = D.z10_map.sel(ssw=e["date"]).values / 1000
        fc, lc = cyclic(f, lon)
        ax.contourf(lc, lat, fc, levels=lev, cmap=cmap, norm=norm, extend="both", transform=pc)
        ax.contour(lc, lat, fc, levels=[E["vortex_edge_m"] / 1000], colors=INK, linewidths=1.8, transform=pc)
        ax.coastlines(lw=0.35, color="#555")
        lab = f"{fmt_date(e['date'])}"
        sub = "unconfirmed" if e.get("marginal") else f"{e['type']}, {e['depth']}"
        ax.set_title(f"{lab}\n{sub}", fontsize=8.6, color=INK if not e.get("marginal") else MUTED, pad=2, linespacing=1.15)
    cax = fig.add_axes([0.3, 0.018, 0.4, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(lev[::2]); cb.ax.tick_params(labelsize=8); cb.set_label("10 hPa height, km", fontsize=8.5, color=INK)
    save(fig, out)


# ---------------------------------------------------------------- dripping paint
IDX_NAME = {"ao": ("Arctic Oscillation", "AO", "CPC daily AO"), "nao": ("North Atlantic Oscillation", "NAO", "CPC daily NAO")}


def ao_rows(D, E, s, ix="ao"):
    """Event paths and the calendar-matched baseline of a daily index (ix = "ao" or "nao") for one event set."""
    dates = list(D.ssw_used.values) if s != "sv" else None
    if s == "sv":
        return D[f"{ix}_sv"].values, D[f"{ix}_base_sv"].values
    keep = {"ssw_all": lambda e: True, "ssw_deep": lambda e: e["depth"] == "deep", "ssw_shallow": lambda e: e["depth"] == "shallow",
            "ssw_split": lambda e: e["type"] == "split", "ssw_disp": lambda e: e["type"] == "displacement"}[s]
    ok = {e["date"] for e in E["ssw"] if not e.get("marginal") and keep(e)}
    rows = [i for i, d in enumerate(dates) if d in ok]
    return D[f"{ix}_ssw"].values[rows], D[f"{ix}_base_ssw"].values


SIG_NOTE = "Coloured only where significant: t-test across events, false-discovery rate 10 % over the whole chart (Wilks 2016, about 5 % overall)"


def pval(t, n):
    from scipy import stats
    return 2 * stats.t.sf(np.abs(np.asarray(t, float)), max(n - 1, 1))


def fdr(p, alpha=0.10):
    """Benjamini-Hochberg over every cell; alpha_FDR 0.10 holds the chance of any false positive in a spatially
    correlated field near 0.05 (Wilks 2016). Returns a boolean mask of the significant cells."""
    q = np.asarray(p, float); ok = np.isfinite(q); v = np.sort(q[ok]); n = v.size
    if not n:
        return np.zeros(q.shape, bool)
    passed = v[v <= alpha * np.arange(1, n + 1) / n]
    return ok & (q <= passed.max()) if passed.size else np.zeros(q.shape, bool)


def ao_sig(A, base, win=7):
    """Event AO paths (7-day means), their mean, 95 % interval and an FDR-significance mask of mean - baseline."""
    sm = pd.DataFrame(A.T).rolling(win, center=True, min_periods=4).mean().values.T
    bs = pd.Series(base).rolling(win, center=True, min_periods=4).mean().values
    n = np.isfinite(sm).sum(0); m = np.nanmean(sm, 0); se = np.nanstd(sm, 0, ddof=1) / np.sqrt(n)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, np.maximum(n - 1, 1))
    sig = fdr(pval((m - bs) / se, int(np.median(n))))
    return sm, m, m - tcrit * se, m + tcrit * se, bs, sig


_WT = {}


def window_tests(D, E, ix="ao"):
    """Index (AO or NAO) mean over days 1-30 and 31-60 minus the same calendar days in other years, per event set:
    one-sample t-test, Benjamini-Hochberg at 10 % over the index's 12 (set, window) tests. Cached per index;
    {(set, win): (mean, p, significant)}."""
    if ix in _WT:
        return _WT[ix]
    from scipy import stats
    lag = D.lag.values; keys, vals = [], []
    for s in SET_LABEL:
        A, base = ao_rows(D, E, s, ix)
        for w, (a, b) in (("d1_30", (1, 30)), ("d31_60", (31, 60))):
            k = (lag >= a) & (lag <= b); x = np.nanmean(A[:, k], 1) - np.nanmean(base[k])
            keys.append((s, w)); vals.append((float(x.mean()), float(stats.ttest_1samp(x, 0).pvalue)))
    sig = fdr([v[1] for v in vals])
    _WT[ix] = {k: (v[0], v[1], bool(g)) for k, v, g in zip(keys, vals, sig)}
    return _WT[ix]


def wt_text(D, E, s):
    W = window_tests(D, E)
    return "   ".join(f"days {w[1:].replace('_', '–')}: {W[(s, w)][0]:+.2f} " + (f"(p = {W[(s, w)][1]:.3f}, significant)" if W[(s, w)][2]
                     else "(not significant)") for w in ("d1_30", "d31_60"))


def draw_sig_line(ax, x, y, sig, col, label, lw=2.8):
    ax.plot(x, y, color=col, lw=1.1, ls=(0, (2, 2)), alpha=0.8)
    ys = np.where(sig, y, np.nan)
    ax.plot(x, ys, color=col, lw=lw, label=label)


def drip(D, E, s, out):
    n = E["sets"][s]
    lag = D.lag.values
    lev = D.lev.values
    keep = lev >= 1.0
    m = D.drip_mean.sel(set=s).values[:, keep].T
    sig = fdr(pval(D.drip_t.sel(set=s).values[:, keep].T, n))
    p = lev[keep]
    fig = plt.figure(figsize=(13.4, 8.6), dpi=125)
    below = fig_header(fig, f"Dripping paint: polar-cap height around the {n} {SET_LABEL[s]} since 1980",
               f"Standardised 65–90°N geopotential height anomaly, mean of {n} events; day 0 = "
               f"{'the first easterly day at 60°N, 10 hPa' if s != 'sv' else 'the 10 hPa annular index first crossing +1.5'}. "
               f"Red = high cap heights (weak vortex, negative AO), blue = low. {SIG_NOTE}; grey = not significant. {SRC}.")
    ax = fig.add_axes([0.07, 0.36, 0.84, below - 0.365])
    ax.set_facecolor("#eceef1")
    lv = np.arange(-2.4, 2.41, 0.3)
    mm = np.where(sig, m, np.nan)
    cf = ax.contourf(lag, p, mm, levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(lag, p, np.where(sig, m, np.nan), levels=[x for x in lv if abs(x) > 1e-6], colors="#333", linewidths=0.35)
    if not sig.any():
        ax.text(0.5, 0.5, "No significant anomaly anywhere on this chart", transform=ax.transAxes, ha="center", fontsize=13, color=INK)
    ax.set_yscale("log"); ax.set_ylim(1000, 1)
    ax.set_yticks([1000, 500, 300, 100, 50, 30, 10, 5, 3, 1]); ax.set_yticklabels(["1000", "500", "300", "100", "50", "30", "10", "5", "3", "1"])
    ax.axvline(0, color=INK, lw=1.0)
    for L in (100, 10):
        ax.axhline(L, color="#555", lw=0.5, ls=":")
    ax.set_ylabel("pressure, hPa", fontsize=10, color=INK)
    ax.set_xlim(lag[0], lag[-1]); ax.tick_params(labelbottom=False)
    style(ax); ax.grid(False)
    cax = fig.add_axes([0.925, 0.36, 0.012, below - 0.365])
    cb = fig.colorbar(cf, cax=cax); cb.set_label("standard deviations", fontsize=9); cb.ax.tick_params(labelsize=8)
    A, base = ao_rows(D, E, s)
    sm, mean, lo, hi, bs, sg = ao_sig(A, base)
    col = WARM if s != "sv" else COOL
    ax2 = fig.add_axes([0.07, 0.08, 0.84, 0.23])
    ax2.fill_between(lag, lo, hi, color=col, alpha=0.16, lw=0, label="95 % interval of the mean")
    draw_sig_line(ax2, lag, mean, sg, col, f"mean of {len(sm)} events, solid where significant")
    ax2.plot(lag, bs, color=INK, lw=1.1, ls="--", label="same dates, other years")
    ax2.axhline(0, color="#555", lw=0.8); ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xlim(lag[0], lag[-1]); ax2.set_ylim(-2.2, 1.6)
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); ax2.set_ylabel("AO, 7-day mean", fontsize=10, color=INK)
    style(ax2); ax2.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    ax2.set_title("Arctic Oscillation at the surface (CPC daily index); dotted = not significant day by day", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax2.text(0.995, 0.96, "30-day means vs other years: " + wt_text(D, E, s), transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.6, color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    save(fig, out)
    return bool(sig.any()), bool(sg[(lag > 0)].any())


# ---------------------------------------------------------------- dripping paint in u(60N): CMIP6 and MERRA-2
REF_DRIP = HERE / "reference" / "cmip6_drip.nc"      # build_cmip6_drip.py (laptop, once)
U_SETS = ["ssw_all", "ssw_deep", "ssw_shallow", "sv"]
U_SRC = {"cmip6": "CMIP6", "m2u": "MERRA-2"}


def robust(X, agree=0.8, min_models=5):
    """Across-model test on X (model, ...): mean, and robust = one-sample t-test passing Benjamini-Hochberg FDR 10 %
    over every cell AND >= `agree` of the models on the sign (the house CMIP6 rule)."""
    from scipy import stats
    n = np.isfinite(X).sum(0)
    m = np.nanmean(X, 0); sd = np.nanstd(X, 0, ddof=1)
    t = m / (sd / np.sqrt(np.maximum(n, 1)))
    p = np.where(n >= min_models, 2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1)), np.nan)
    same = np.maximum((X > 0).sum(0), (X < 0).sum(0)) / np.maximum(n, 1)
    return m, fdr(p) & (same >= agree), n, same


def smooth7(a):
    return pd.DataFrame(np.atleast_2d(a).T).rolling(7, center=True, min_periods=4).mean().values.T


_UWT = {}


def u_window_tests(R):
    """Strip (700 hPa u') means over days 1-30 and 31-60 per set and source, tested against zero: CMIP6 across models
    (t-test + 80 % sign agreement), MERRA-2 across events (t-test); Benjamini-Hochberg 10 % within each source's 8 tests."""
    if _UWT:
        return _UWT
    from scipy import stats
    lag = R.lag.values
    for src in U_SRC:
        keys, vals = [], []
        for s in U_SETS:
            X = R.cmip6_strip.sel(set=s).values if src == "cmip6" else R[f"m2_strip_{s.replace('ssw_all', 'ssw').replace('ssw_', '')}"].values
            for w, (a, b) in (("d1_30", (1, 30)), ("d31_60", (31, 60))):
                k = (lag >= a) & (lag <= b); x = np.nanmean(X[:, k], 1); x = x[np.isfinite(x)]
                same = max((x > 0).sum(), (x < 0).sum()) / len(x)
                keys.append((s, w)); vals.append((float(x.mean()), float(stats.ttest_1samp(x, 0).pvalue), same, len(x)))
        sig = fdr([v[1] for v in vals])
        for k, v, g in zip(keys, vals, sig):
            _UWT[(src, k[0], k[1])] = (v[0], v[1], bool(g) and (src != "cmip6" or v[2] >= 0.8), v[2], v[3])
    return _UWT


def drip_u(R, src, s, out):
    lag = R.lag.values
    if src == "cmip6":
        p = R.plev.values
        m, sig, nmod, _ = robust(R.cmip6_comp.sel(set=s).values)
        m, sig = m.T, sig.T
        nev, nm, yrs = int(R.cmip6_n.sel(set=s).sum()), int(R.sizes["model"]), int(R.cmip6_years.sum())
        who = (f"mean of {nm} CMIP6 models' composites, models weighted equally ({yrs:,} model-years of control, historical, "
               f"scenario and AMIP runs)")
        test = ("Coloured only where robust: across-model t-test, false-discovery rate 10 % over the chart, and at least 80 % "
                "of the models agree on the sign")
        title = f"Dripping paint in {nev:,} model {SET_LABEL[s]}: u(60°N) in {nm} CMIP6 models"
    else:
        p = R.m2lev.values
        n = int(R.m2_n.sel(set=s))
        m = R.m2_mean.sel(set=s).values.T
        sig = fdr(pval(R.m2_t.sel(set=s).values.T, n))
        nev = n
        who = f"mean of {n} events, MERRA-2 zonal means 1980–2026 (NASA GMAO)"
        test = SIG_NOTE
        title = f"Dripping paint: u(60°N) around the {n} {SET_LABEL[s]} since 1980, MERRA-2"
    d0 = ("the first easterly day at 60°N, 10 hPa" if s != "sv" else "u(60°N, 10 hPa) first 1.5 standard deviations above normal")
    fig = plt.figure(figsize=(13.4, 8.6), dpi=125)
    below = fig_header(fig, title,
               f"Zonal-mean wind at 60°N, standardised, sign flipped so a weaker vortex is red, as in the polar-cap height charts; "
               f"{who}; day 0 = {d0}. {test}; grey = not significant.")
    ax = fig.add_axes([0.07, 0.36, 0.84, below - 0.365])
    ax.set_facecolor("#eceef1")
    lv = np.arange(-2.4, 2.41, 0.3)
    cf = ax.contourf(lag, p, np.where(sig, m, np.nan), levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(lag, p, np.where(sig, m, np.nan), levels=[x for x in lv if abs(x) > 1e-6], colors="#333", linewidths=0.35)
    if not sig.any():
        ax.text(0.5, 0.5, "No significant anomaly anywhere on this chart", transform=ax.transAxes, ha="center", fontsize=13, color=INK)
    ax.set_yscale("log"); ax.set_ylim(1000, 1)
    ax.set_yticks([1000, 500, 300, 100, 50, 30, 10, 5, 3, 1]); ax.set_yticklabels(["1000", "500", "300", "100", "50", "30", "10", "5", "3", "1"])
    ax.axvline(0, color=INK, lw=1.0)
    for L in (100, 10):
        ax.axhline(L, color="#555", lw=0.5, ls=":")
    ax.axhline(float(R.attrs.get("strip_hpa", 700)), color="#555", lw=0.5, ls=(0, (1, 3)))
    ax.set_ylabel("pressure, hPa", fontsize=10, color=INK)
    ax.set_xlim(lag[0], lag[-1]); ax.tick_params(labelbottom=False)
    style(ax); ax.grid(False)
    cax = fig.add_axes([0.925, 0.36, 0.012, below - 0.365])
    cb = fig.colorbar(cf, cax=cax); cb.set_label("standard deviations (−u′)", fontsize=9); cb.ax.tick_params(labelsize=8)
    col = WARM if s != "sv" else COOL
    ax2 = fig.add_axes([0.07, 0.08, 0.84, 0.23])
    sp = int(R.attrs.get("strip_hpa", 700))
    if src == "cmip6":
        X = smooth7(R.cmip6_strip.sel(set=s).values)
        for x in X:
            ax2.plot(lag, x, color="#9aa3ad", lw=0.7, alpha=0.9)
        mean, sg, _, _ = robust(X)
        ax2.plot([], [], color="#9aa3ad", lw=0.7, label="each model")
        draw_sig_line(ax2, lag, mean, sg, col, "mean of the models, solid where robust")
        lo_hi = X
    else:
        key = {"ssw_all": "ssw", "ssw_deep": "deep", "ssw_shallow": "shallow", "sv": "sv"}[s]
        X = smooth7(R[f"m2_strip_{key}"].values)
        from scipy import stats
        nn = np.isfinite(X).sum(0); mean = np.nanmean(X, 0); se = np.nanstd(X, 0, ddof=1) / np.sqrt(nn)
        tc = stats.t.ppf(0.975, np.maximum(nn - 1, 1))
        sg = fdr(pval(mean / se, int(np.median(nn))))
        ax2.fill_between(lag, mean - tc * se, mean + tc * se, color=col, alpha=0.16, lw=0, label="95 % interval of the mean")
        draw_sig_line(ax2, lag, mean, sg, col, f"mean of {len(X)} events, solid where significant")
        lo_hi = np.vstack([mean - tc * se, mean + tc * se])
    ax2.axhline(0, color="#555", lw=0.8); ax2.axvline(0, color=INK, lw=1.0)
    top = float(np.nanmax(np.abs(lo_hi))) if np.isfinite(lo_hi).any() else 1.0
    ax2.set_xlim(lag[0], lag[-1]); ax2.set_ylim(-max(0.6, 1.15 * top), max(0.6, 1.15 * top))
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); ax2.set_ylabel(f"u′ {sp} hPa, 7-day mean", fontsize=10, color=INK)
    style(ax2); ax2.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    ax2.set_title(f"u(60°N) at {sp} hPa, standardised: a daily annular-mode proxy (negative ≈ negative AO); dotted = not significant",
                  loc="left", fontsize=10.2, fontweight="bold", color=INK)
    W = u_window_tests(R)
    txt = "   ".join(f"days {w[1:].replace('_', '–')}: {W[(src, s, w)][0]:+.2f} "
                     + (f"({int(round(W[(src, s, w)][3] * W[(src, s, w)][4]))}/{W[(src, s, w)][4]} models, " if src == "cmip6" else "(")
                     + (f"p = {W[(src, s, w)][1]:.1g}, significant)" if W[(src, s, w)][2] else "not significant)")
                     for w in ("d1_30", "d31_60"))
    ax2.text(0.995, 0.96, "30-day means vs normal: " + txt, transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.6, color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    save(fig, out)
    return {"n": nev, "cells_sig": int(sig.sum()), "strip_sig_after_0": bool(sg[lag > 0].any())}


# ---------------------------------------------------------------- surface maps
def follows_map(D, E, s, w, out):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    n = E["sets"][s]
    lat, lon = D.lat.values, D.lon.values
    m = D.t2m_mean.sel(set=s, window=w).values
    sig = fdr(pval(D.t2m_t.sel(set=s, window=w).values, n))
    z = D.z500_mean.sel(set=s, window=w).values
    zsig = fdr(pval(D.z500_t.sel(set=s, window=w).values, n))
    fig = plt.figure(figsize=(10.4, 11.2), dpi=120)
    below = fig_header(fig, f"2 m temperature, {WIN_LABEL[w]} after {SET_LABEL[s]}",
               f"Mean anomaly over {n} events (ERA5, against each point's seasonal cycle and trend); contours: 500 hPa height "
               f"anomaly every 20 m (dashed negative). {SIG_NOTE}; both fields are blank where not significant. "
               f"North America at the bottom.", x=0.05)
    ax = polar_axes(fig, [0.03, 0.1, 0.94, below - 0.14], lat0=20, central=-80)
    pc = ccrs.PlateCarree()
    lv = np.arange(-3.5, 3.51, 0.5)
    mc, lc = cyclic(np.where(sig, m, np.nan), lon)
    cf = ax.contourf(lc, lat, mc, levels=lv, cmap="RdBu_r", extend="both", transform=pc)
    zc, _ = cyclic(np.where(zsig, z, np.nan), lon)
    zl = [x for x in np.arange(-200, 201, 20) if x != 0]
    if zsig.any():
        cs = ax.contour(lc, lat, zc, levels=zl, colors=INK, linewidths=0.9, transform=pc)
        ax.clabel(cs, fmt="%d", fontsize=7)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
    ax.gridlines(lw=0.4, color="#999", ylocs=[30, 45, 60, 75], xlocs=np.arange(-180, 181, 45))
    if not sig.any():
        ax.text(0.5, 0.5, "No significant temperature signal", transform=ax.transAxes, ha="center", va="center", fontsize=15,
                color=INK, bbox=dict(facecolor="white", edgecolor="#9aa3ad", boxstyle="round,pad=0.5"))
    cax = fig.add_axes([0.2, 0.065, 0.6, 0.013])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
    cb.ax.set_title("2 m temperature anomaly, K (significant cells only)", fontsize=9, color=INK, pad=3)
    fig.text(0.05, 0.018, f"{int(sig.sum())} of {sig.size} grid cells significant for temperature, {int(zsig.sum())} for 500 hPa height.",
             fontsize=8.8, color=MUTED)
    save(fig, out)
    return int(sig.sum()), int(zsig.sum())


def ao_figure(D, E, out, ix="ao"):
    name, short, src = IDX_NAME[ix]
    from scipy import stats
    fig = plt.figure(figsize=(13.4, 7.8), dpi=125)
    below = fig_header(fig, f"The {name} after sudden warmings and strong-vortex events",
               f"{src}, 7-day running mean, around {E['sets']['ssw_all']} SSWs and {E['sets']['sv']} strong-vortex events "
               f"(MERRA-2 dates, 1980–2026). Lines are solid on days where the mean differs from the same calendar days in other years "
               f"(t-test, false-discovery rate 10 % across days), dotted where it does not; shading = 95 % interval of the mean. "
               f"30-day means are the more powerful test (box). "
               f"Thin lines: the same calendar days in other years. Lower panel: share of events with a negative {short}, solid where "
               f"a binomial test against that share in other years (thin line) is significant.")
    lag = D.lag.values
    ax = fig.add_axes([0.07, 0.42, 0.9, below - 0.445])
    ax2 = fig.add_axes([0.07, 0.08, 0.9, 0.27])
    for s, col, lab in (("ssw_all", WARM, "after SSWs"), ("sv", COOL, "after strong-vortex events")):
        A, base = ao_rows(D, E, s, ix)
        sm, mean, lo, hi, bs, sg = ao_sig(A, base)
        ax.fill_between(lag, lo, hi, color=col, alpha=0.14, lw=0)
        draw_sig_line(ax, lag, mean, sg, col, f"mean {lab} (n={len(sm)})")
        ax.plot(lag, bs, color=col, lw=0.8, alpha=0.5)
        # the null for "% negative" is the share on the same calendar days in other years (the NAO's winter mean is not
        # zero: ~30 % of those days are negative); references built before it was stored fall back to 50 %
        key = f"{ix}_base_pneg_{'sv' if s == 'sv' else 'ssw'}"
        p0 = np.clip(D[key].values.astype(float), 0.01, 0.99) if key in D else np.full(len(lag), 0.5)
        k = np.isfinite(sm).sum(0); neg = (sm < 0).sum(0)
        pb = np.array([stats.binomtest(int(a), int(b), float(q)).pvalue if b else np.nan for a, b, q in zip(neg, k, p0)])
        draw_sig_line(ax2, lag, 100 * neg / np.maximum(k, 1), fdr(pb), col, lab, lw=2.4)
        ax2.plot(lag, 100 * p0, color=col, lw=0.8, alpha=0.5)
    W = window_tests(D, E, ix)
    rows = [f"{'30-day means vs other years':<30s}{'days 1–30':>16s}{'days 31–60':>16s}"]
    for s_ in SET_LABEL:
        cell = lambda w: (f"{W[(s_, w)][0]:+.2f}" + (" *" if W[(s_, w)][2] else "  n.s.")).rjust(16)
        rows.append(f"{SET_SHORT[s_] + ' (n=' + str(E['sets'][s_]) + ')':<30s}{cell('d1_30')}{cell('d31_60')}")
    ax.text(0.995, 0.97, "\n".join(rows) + "\n* significant (t-test, false-discovery rate 10 % over the 12 tests)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.2, family="monospace", color=INK, bbox=dict(facecolor="white", edgecolor="#c9ccd1", pad=4))
    ax.axhline(0, color="#555", lw=0.8); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-40, 90); ax.set_ylim(-1.8, 1.8); ax.set_ylabel(f"{short}, 7-day mean", fontsize=10, color=INK)
    ax.tick_params(labelbottom=False); style(ax); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.axvline(0, color=INK, lw=1)
    ax2.set_xlim(-40, 90); ax2.set_ylim(0, 100); ax2.set_ylabel(f"% of events {short} < 0", fontsize=10, color=INK)
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); style(ax2)
    ax2.text(-39, 2, "thin lines: the share on the same calendar days in other years", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    save(fig, out)


def regions_figure(E, out):
    from scipy import stats
    R = list(E["regions"])
    S = ["ssw_all", "ssw_deep", "ssw_shallow", "ssw_split", "ssw_disp", "sv"]
    Ws = ("d1_30", "d31_60")
    cells = {(w, r, s): np.array(E["region_values"][f"{s}|{w}|{r}"]) for w in Ws for r in R for s in S}
    keys = list(cells)
    # the question is "did it turn cold": a binomial test of the share of events colder than normal against the region's
    # own chance of a cold 30-day window, two-sided
    p = np.array([stats.binomtest(int((cells[k] < 0).sum()), len(cells[k]), E["region_base_pcold"][f"{k[0]}|{k[1]}"]).pvalue for k in keys])
    sig = dict(zip(keys, fdr(p)))
    fig = plt.figure(figsize=(13.4, 6.8), dpi=125)
    below = fig_header(fig, "Did it turn cold? Regional 2 m temperature after each kind of event",
               "Land-only box means, ERA5, against each point's seasonal cycle and trend. A cell shows the mean anomaly and the share "
               "of events colder than normal only where that share differs significantly from the region's normal chance of a cold "
               "30-day window (under each region; binomial test, false-discovery rate 10 % over all 72 cells). n.s. = not significant.")
    for k, w in enumerate(Ws):
        ax = fig.add_axes([0.13 + k * 0.44, 0.06, 0.4, below - 0.2])
        M = np.array([[cells[(w, r, s)].mean() if sig[(w, r, s)] else np.nan for s in S] for r in R])
        ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_facecolor("#f4f5f7")
        for i, r in enumerate(R):
            for j, s_ in enumerate(S):
                v = cells[(w, r, s_)]
                if sig[(w, r, s_)]:
                    ax.text(j, i, f"{v.mean():+.1f} K\n{100 * np.mean(v < 0):.0f}% cold", ha="center", va="center", fontsize=8.8,
                            color="white" if abs(v.mean()) > 1.3 else INK, linespacing=1.2, fontweight="bold")
                else:
                    ax.text(j, i, "n.s.", ha="center", va="center", fontsize=8.5, color="#8a8f96")
        ax.set_xticks(range(len(S))); ax.set_xticklabels([f"{SET_SHORT[s_]}\n(n={E['sets'][s_]})" for s_ in S], fontsize=8.6)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([f"{r}\n{E['region_base_pcold'][f'{w}|{r}'] * 100:.0f}% normally" for r in R] if k == 0 else [], fontsize=8.8)
        ax.set_title(WIN_LABEL[w] + " after", fontsize=11, fontweight="bold", color=INK, pad=34)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    save(fig, out)
    return {f"{w}|{r}|{s_}": round(float(cells[(w, r, s_)].mean()), 2) for (w, r, s_), v in sig.items() if v}


# ---------------------------------------------------------------- "What usually follows" in CMIP6 (build_cmip6_follow.py)
REF_FOLLOW = HERE / "reference" / "cmip6_follow.nc"
F_SETS = ["ssw_all", "ssw_deep", "ssw_shallow", "sv"]
F_TEST = ("across-model t-test, false-discovery rate 10 % over {what}, and at least 80 % of the models agree on the sign")


def with_pole(f, lat):
    """Append a 90N row (the zonal mean of the northernmost row): the 2.5-degree box centres stop at 88.75N."""
    top = np.nanmean(f[-1:], 1, keepdims=True) if np.isfinite(f[-1]).any() else np.full((1, 1), np.nan)
    return np.vstack([f, np.repeat(top, f.shape[1], 1)]), np.append(lat, 90.0)


def f_meta(F):
    return int(F.sizes["model"]), int(F.years.sum()), int(F.members.sum())


def follows_map_cmip6(F, s, w, out, lim):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lat, lon = F.lat.values, F.lon.values
    m, sig, _, _ = robust(F.tas.sel(set=s, window=w).values)
    pm, psig, _, _ = robust(F.psl.sel(set=s, window=w).values)
    nev = int(F.n.sel(set=s).sum()); nm, yrs, _ = f_meta(F)
    fig = plt.figure(figsize=(10.4, 11.2), dpi=120)
    below = fig_header(fig, f"2 m temperature, {WIN_LABEL[w]} after {nev:,} {SET_LABEL[s]} in CMIP6",
               f"Mean of the {nm} CMIP6 models' composites, models weighted equally ({yrs:,} model-years). Monthly anomalies from each "
               f"run's own running climatology; the window is overlap-weighted over calendar months (each day carries its month's "
               f"anomaly), so it is smoother than the ERA5 version. Contours: sea-level pressure anomaly every 0.5 hPa (dashed "
               f"negative), in place of ERA5's 500 hPa height. Shaded and contoured only where robust: "
               + F_TEST.format(what="the map") + "; blank = not robust. North America at the bottom.", x=0.05)
    ax = polar_axes(fig, [0.03, 0.1, 0.94, below - 0.105], lat0=20, central=-80)
    pc = ccrs.PlateCarree()
    lv = np.round(np.arange(-lim, lim + 1e-9, lim / 7), 3)
    mc, la = with_pole(np.where(sig, m, np.nan), lat)
    mc, lc = cyclic(mc, lon)
    cf = ax.contourf(lc, la, mc, levels=lv, cmap="RdBu_r", extend="both", transform=pc)
    if psig.any():
        zc, _ = with_pole(np.where(psig, pm, np.nan), lat)
        zc, _ = cyclic(zc, lon)
        zl = [x for x in np.arange(-6, 6.01, 0.5) if abs(x) > 1e-9]
        cs = ax.contour(lc, la, zc, levels=zl, colors=INK, linewidths=0.9, transform=pc)
        ax.clabel(cs, fmt="%g", fontsize=7)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
    ax.gridlines(lw=0.4, color="#999", ylocs=[30, 45, 60, 75], xlocs=np.arange(-180, 181, 45))
    if not sig.any():
        ax.text(0.5, 0.5, "No robust temperature signal", transform=ax.transAxes, ha="center", va="center", fontsize=15,
                color=INK, bbox=dict(facecolor="white", edgecolor="#9aa3ad", boxstyle="round,pad=0.5"))
    cax = fig.add_axes([0.2, 0.065, 0.6, 0.013])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
    cb.ax.set_title("2 m temperature anomaly, K (robust cells only)", fontsize=9, color=INK, pad=3)
    fig.text(0.05, 0.018, f"{int(sig.sum())} of {sig.size} grid cells (2.5°) robust for temperature, {int(psig.sum())} for sea-level pressure.",
             fontsize=8.8, color=MUTED)
    save(fig, out)
    return int(sig.sum()), int(psig.sum())


_FT = {}


def f_ix_tests(F, index, what="pneg"):
    """Per index: the 8 (set, window) cells tested across models - the mean (what="mean") or the share of events below
    zero minus each model's normal share (what="pneg") - BH-FDR 10 % over the 8, AND >= 80 % sign agreement.
    {(set, win): (multi-model value, p, robust, agreement, n models, normal share)}"""
    key = (index, what)
    if key in _FT:
        return _FT[key]
    from scipy import stats
    keys, vals = [], []
    for s in F_SETS:
        for w in ("d1_30", "d31_60"):
            mean = F.ix_mean.sel(set=s, window=w, index=index).values.astype(float)
            pneg = F.ix_pneg.sel(set=s, window=w, index=index).values.astype(float)
            base = F.ix_base_pneg.sel(window=w, index=index).values.astype(float)
            x = mean if what == "mean" else pneg - base
            x = x[np.isfinite(x)]
            same = max((x > 0).sum(), (x < 0).sum()) / len(x)
            val = float(x.mean()) if what == "mean" else float(np.nanmean(pneg))
            keys.append((s, w)); vals.append((val, float(stats.ttest_1samp(x, 0).pvalue), same, len(x), float(np.nanmean(base))))
    sig = fdr([v[1] for v in vals])
    _FT[key] = {k: (v[0], v[1], bool(g and v[2] >= 0.8), float(v[2]), int(v[3]), v[4]) for k, v, g in zip(keys, vals, sig)}
    return _FT[key]


def ao_cmip6(F, out):
    nm, yrs, _ = f_meta(F)
    fig = plt.figure(figsize=(13.4, 7.8), dpi=125)
    below = fig_header(fig, f"The annular mode after sudden warmings and strong-vortex events in {nm} CMIP6 models",
               f"There is no daily surface pressure in this sample, so the daily Arctic Oscillation is stood in for by the zonal-mean "
               f"wind at 60°N, 700 hPa, standardised (weaker westerlies ≈ negative AO), 7-day running mean; {int(F.n.sel(set='ssw_all').sum()):,} "
               f"SSWs and {int(F.n.sel(set='sv').sum()):,} strong-vortex events, {yrs:,} model-years. Lines: mean of the models' "
               f"composites, solid where robust (" + F_TEST.format(what="the days") + "); band = range across the models. Lower panel: "
               f"share of events below zero, solid where it differs robustly from each model's normal share for those calendar days "
               f"(thin line). Box: 30-day means, with the monthly sea-level-pressure annular mode as a cross-check.")
    lag = F.lag.values
    ax = fig.add_axes([0.07, 0.42, 0.9, below - 0.44])
    ax2 = fig.add_axes([0.07, 0.08, 0.9, 0.27])
    for s, col, lab in (("ssw_all", WARM, "after SSWs"), ("sv", COOL, "after strong-vortex events")):
        X = F.u700_path.sel(set=s).values.astype(float)
        mean, sg, _, _ = robust(X)
        ax.fill_between(lag, np.nanmin(X, 0), np.nanmax(X, 0), color=col, alpha=0.13, lw=0)
        draw_sig_line(ax, lag, mean, sg, col, f"mean of the models {lab} (n={int(F.n.sel(set=s).sum()):,})")
        P = F.u700_pneg.sel(set=s).values.astype(float); Bn = F.u700_base_pneg.sel(set=s).values.astype(float)
        _, sgp, _, _ = robust(P - Bn)
        draw_sig_line(ax2, lag, 100 * np.nanmean(P, 0), sgp, col, lab, lw=2.4)
        ax2.plot(lag, 100 * np.nanmean(Bn, 0), color=col, lw=0.8, alpha=0.5)
    rows = [f"{'30-day means':<16s}{'u′ 700 hPa (daily)':>26s}{'SLP annular mode (monthly)':>30s}",
            f"{'':<16s}{'d1–30':>13s}{'d31–60':>13s}{'d1–30':>15s}{'d31–60':>15s}"]
    Tu, Tn = f_ix_tests(F, "u700", "mean"), f_ix_tests(F, "nam", "mean")
    for s_ in F_SETS:
        cell = lambda T, w, width: (f"{T[(s_, w)][0]:+.2f}" + (" *" if T[(s_, w)][2] else " n.s.")).rjust(width)
        rows.append(f"{SET_SHORT[s_]:<16s}{cell(Tu, 'd1_30', 13)}{cell(Tu, 'd31_60', 13)}{cell(Tn, 'd1_30', 15)}{cell(Tn, 'd31_60', 15)}")
    ax.text(0.995, 0.97, "\n".join(rows) + "\n* robust (across-model t-test, false-discovery rate 10 % over each index's 8 cells,\n"
            "  ≥ 80 % of models agree); n.s. = not robust. Units: standard deviations.", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.0, family="monospace", color=INK, bbox=dict(facecolor="white", edgecolor="#c9ccd1", pad=4))
    ax.axhline(0, color="#555", lw=0.8); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-40, 90); ax.set_ylim(-0.9, 0.9); ax.set_ylabel("u′ 700 hPa, 7-day mean (σ)", fontsize=10, color=INK)
    ax.tick_params(labelbottom=False); style(ax); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.axhline(50, color=INK, lw=0.6, ls=":"); ax2.axvline(0, color=INK, lw=1)
    ax2.set_xlim(-40, 90); ax2.set_ylim(25, 75); ax2.set_ylabel("% of events below zero", fontsize=10, color=INK)
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); style(ax2)
    ax2.text(-39, 26.5, "thin lines: the models' normal share for the same calendar days", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    save(fig, out)


def nao_cmip6(F, out):
    nm, yrs, _ = f_meta(F)
    import re as _re
    r = _re.search(r'"r_cpc_djfm": ([0-9.]+)', F.attrs.get("nao", ""))
    rtxt = f" (ERA5 projected the same way correlates {float(r.group(1)):.2f} with CPC's monthly NAO in Dec–Mar, 1991–2020)" if r else ""
    Tm, Tp = f_ix_tests(F, "nao", "mean"), f_ix_tests(F, "nao", "pneg")
    fig = plt.figure(figsize=(13.4, 7.8), dpi=125)
    below = fig_header(fig, f"The North Atlantic Oscillation after sudden warmings and strong-vortex events in {nm} CMIP6 models",
               f"A monthly index (there is no daily pressure in this sample): each run's sea-level-pressure anomaly over 20–80°N, "
               f"90°W–40°E projected on ERA5's NAO pattern for that calendar month (leading EOF of 1991–2020 monthly SLP){rtxt}, "
               f"standardised per model and month; a window's value is overlap-weighted over calendar months. Dots: each model; "
               f"bars: the mean of the models, coloured where robust (" + F_TEST.format(what="the 8 cells of each panel") +
               "), grey and marked n.s. where not.")
    top = fig.add_axes([0.07, 0.47, 0.9, below - 0.49])
    bot = fig.add_axes([0.07, 0.08, 0.9, 0.3])
    xs, labels = [], []
    k = 0
    for s in F_SETS:
        for w in ("d1_30", "d31_60"):
            xs.append(k); labels.append(f"{SET_SHORT[s]}\n{WIN_LABEL[w]}"); k += 1
        k += 0.6
    col = {"ssw_all": WARM, "ssw_deep": WARM, "ssw_shallow": WARM, "sv": COOL}
    i = 0
    for s in F_SETS:
        for w in ("d1_30", "d31_60"):
            x = xs[i]; i += 1
            v = F.ix_mean.sel(set=s, window=w, index="nao").values.astype(float)
            mean, _, rob = Tm[(s, w)][:3]
            top.bar(x, mean, 0.7, color=col[s] if rob else "#d5d8dc", alpha=0.85 if rob else 1.0, zorder=2)
            top.scatter(np.full(len(v), x) + np.linspace(-0.22, 0.22, len(v)), v, s=14, color=INK, alpha=0.55, zorder=3, lw=0)
            edge = (max(mean, np.nanmax(v)) + 0.05) if mean >= 0 else (min(mean, np.nanmin(v)) - 0.05)      # clear of the dots
            top.text(x, edge, f"{mean:+.2f}" if rob else "n.s.", ha="center",
                     va="bottom" if mean >= 0 else "top", fontsize=8.6, color=INK if rob else MUTED, fontweight="bold" if rob else "normal")
            pn = F.ix_pneg.sel(set=s, window=w, index="nao").values.astype(float)
            bs = F.ix_base_pneg.sel(window=w, index="nao").values.astype(float)
            d = 100 * (pn - bs); dm = float(np.nanmean(d)); robp = Tp[(s, w)][2]
            bot.bar(x, dm, 0.7, color=col[s] if robp else "#d5d8dc", alpha=0.85 if robp else 1.0, zorder=2)
            bot.scatter(np.full(len(d), x) + np.linspace(-0.22, 0.22, len(d)), d, s=14, color=INK, alpha=0.55, zorder=3, lw=0)
            edge = (max(dm, np.nanmax(d)) + 1.5) if dm >= 0 else (min(dm, np.nanmin(d)) - 1.5)
            bot.text(x, edge, f"{100 * Tp[(s, w)][0]:.0f}% vs {100 * Tp[(s, w)][5]:.0f}% normally" if robp else "n.s.",
                     ha="center", va="bottom" if dm >= 0 else "top", fontsize=8.2, color=INK if robp else MUTED)
    for a, lab in ((top, "NAO, standard deviations"), (bot, "events with NAO < 0,\npoints above normal")):
        a.axhline(0, color="#555", lw=0.8); style(a); a.set_ylabel(lab, fontsize=9.5, color=INK); a.set_xlim(-0.8, xs[-1] + 0.8)
        a.grid(axis="x", visible=False)
    top.set_xticks(xs); top.set_xticklabels([]); bot.set_xticks(xs); bot.set_xticklabels(labels, fontsize=8.6)
    v = F.ix_mean.sel(index="nao").values
    top.set_ylim(min(-0.3, 1.3 * float(np.nanmin(v))), max(0.3, 1.3 * float(np.nanmax(v))))
    dl = 100 * (F.ix_pneg.sel(index="nao").values - F.ix_base_pneg.sel(index="nao").values[None])
    bot.set_ylim(min(-10.0, 1.3 * float(np.nanmin(dl))), max(10.0, 1.3 * float(np.nanmax(dl))))
    save(fig, out)


def regions_cmip6(F, out):
    R = list(F.region.values)
    Ws = ("d1_30", "d31_60")
    nm, yrs, _ = f_meta(F)
    X = np.array([[[F.ix_pneg.sel(set=s, window=w, index=f"reg_{r}").values - F.ix_base_pneg.sel(window=w, index=f"reg_{r}").values
                    for s in F_SETS] for r in R] for w in Ws], float)                     # (win, region, set, model)
    X = np.moveaxis(X, -1, 0)                                                            # (model, win, region, set)
    _, sig, _, _ = robust(X)
    fig = plt.figure(figsize=(13.4, 6.8), dpi=125)
    below = fig_header(fig, f"Did it turn cold? Regional 2 m temperature after each kind of event, {nm} CMIP6 models",
               f"Land-weighted box means of monthly anomalies (each run's own running climatology; windows overlap-weighted over calendar "
               f"months), {yrs:,} model-years. A cell shows the mean anomaly and the share of events colder than normal only where "
               f"that share differs robustly from the model's own normal chance of a cold window (under each region): "
               + F_TEST.format(what="all 48 cells") + ". n.s. = not robust.")
    for k, w in enumerate(Ws):
        ax = fig.add_axes([0.13 + k * 0.44, 0.06, 0.4, below - 0.2])
        M = np.array([[float(np.nanmean(F.ix_mean.sel(set=s, window=w, index=f"reg_{r}").values)) if sig[k, i, j] else np.nan
                       for j, s in enumerate(F_SETS)] for i, r in enumerate(R)])
        ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
        ax.set_facecolor("#f4f5f7")
        for i, r in enumerate(R):
            for j, s_ in enumerate(F_SETS):
                if sig[k, i, j]:
                    v = float(np.nanmean(F.ix_mean.sel(set=s_, window=w, index=f"reg_{r}").values))
                    pc = 100 * float(np.nanmean(F.ix_pneg.sel(set=s_, window=w, index=f"reg_{r}").values))
                    ax.text(j, i, f"{v:+.2f} K\n{pc:.0f}% cold", ha="center", va="center", fontsize=8.8,
                            color="white" if abs(v) > 0.8 else INK, linespacing=1.2, fontweight="bold")
                else:
                    ax.text(j, i, "n.s.", ha="center", va="center", fontsize=8.5, color="#8a8f96")
        ax.set_xticks(range(len(F_SETS)))
        ax.set_xticklabels([f"{SET_SHORT[s_]}\n(n={int(F.n.sel(set=s_).sum()):,})" for s_ in F_SETS], fontsize=8.6)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([f"{r}\n{100 * float(np.nanmean(F.ix_base_pneg.sel(window=w, index=f'reg_{r}').values)):.0f}% normally"
                            for r in R] if k == 0 else [], fontsize=8.8)
        ax.set_title(WIN_LABEL[w] + " after", fontsize=11, fontweight="bold", color=INK, pad=34)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    save(fig, out)
    return {f"{w}|{r}|{s_}": round(float(np.nanmean(F.ix_mean.sel(set=s_, window=w, index=f"reg_{r}").values)), 2)
            for k, w in enumerate(Ws) for i, r in enumerate(R) for j, s_ in enumerate(F_SETS) if sig[k, i, j]}


def render_follow_cmip6(A, summary):
    F = xr.open_dataset(REF_FOLLOW).load()
    ms = []
    for s in F_SETS:
        for w in ("d1_30", "d31_60"):
            m, sig, _, _ = robust(F.tas.sel(set=s, window=w).values)
            if sig.any():
                ms.append(np.nanpercentile(np.abs(m[sig]), 99))
    lim = float(np.ceil(max(ms + [0.35]) / 0.35) * 0.35) if ms else 1.4       # one colour scale for every CMIP6 map
    summary["cmip6_maps"] = {f"{s}|{w}": follows_map_cmip6(F, s, w, A / f"strat_hist_follows_cmip6_{s}_{w}.webp", lim)
                             for s in F_SETS for w in ("d1_30", "d31_60")}
    ao_cmip6(F, A / "strat_hist_ao_cmip6.webp")
    nao_cmip6(F, A / "strat_hist_nao_cmip6.webp")
    summary["cmip6_regions"] = regions_cmip6(F, A / "strat_hist_regions_cmip6.webp")
    summary["cmip6_index"] = {f"{ix}|{what}|{k[0]}|{k[1]}": [round(v[0], 3), round(v[1], 5), v[2], round(v[3], 2)]
                              for ix in ("u700", "nam", "nao") for what in ("mean", "pneg") for k, v in f_ix_tests(F, ix, what).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(REPO))
    a = ap.parse_args()
    root = Path(a.out_root)
    A = root / "assets" / "sst"
    D = xr.open_dataset(REF_NC).load()
    E = json.load(open(REF_JS))
    write_json(D, E, A / "data" / "strat_history.json")
    timeline(E, A / "strat_hist_timeline.webp")
    gallery(D, E, A / "strat_hist_gallery.webp")
    summary = {"drip": {}, "maps": {}}
    for s in D.set.values:
        summary["drip"][str(s)] = drip(D, E, str(s), A / f"strat_hist_drip_{s}.webp")
        for w in D.window.values:
            summary["maps"][f"{s}|{w}"] = follows_map(D, E, str(s), str(w), A / f"strat_hist_follows_{s}_{w}.webp")
    if REF_DRIP.exists():                                   # u(60N) drips: CMIP6 and MERRA-2 (build_cmip6_drip.py)
        R = xr.open_dataset(REF_DRIP).load()
        summary["drip_u"] = {f"{src}|{s}": drip_u(R, src, s, A / f"strat_hist_drip_{src}_{s}.webp") for src in U_SRC for s in U_SETS}
    ao_figure(D, E, A / "strat_hist_ao.webp")
    if REF_FOLLOW.exists():                                 # CMIP6 "What usually follows" (build_cmip6_follow.py)
        render_follow_cmip6(A, summary)
    summary["regions"] = regions_figure(E, A / "strat_hist_regions.webp")
    summary["windows"] = {f"{k[0]}|{k[1]}": [round(v[0], 2), round(v[1], 4), v[2]] for k, v in window_tests(D, E).items()}
    if "nao_ssw" in D:                                      # CPC daily NAO around the same events (build_strat_history --add-nao)
        ao_figure(D, E, A / "strat_hist_nao.webp", ix="nao")
        summary["windows_nao"] = {f"{k[0]}|{k[1]}": [round(v[0], 2), round(v[1], 4), v[2]] for k, v in window_tests(D, E, "nao").items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
