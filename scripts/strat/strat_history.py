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
  strat_hist_follows_<set>_<win>.webp  ERA5 2 m temperature and 500 hPa height after each kind of event
  strat_hist_ao.webp                 the Arctic Oscillation after SSWs and strong-vortex events
  strat_hist_regions.webp            how often each region ran cold after each kind of event
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
def ao_rows(D, E, s):
    dates = list(D.ssw_used.values) if s != "sv" else None
    if s == "sv":
        return D.ao_sv.values, D.ao_base_sv.values
    keep = {"ssw_all": lambda e: True, "ssw_deep": lambda e: e["depth"] == "deep", "ssw_shallow": lambda e: e["depth"] == "shallow",
            "ssw_split": lambda e: e["type"] == "split", "ssw_disp": lambda e: e["type"] == "displacement"}[s]
    ok = {e["date"] for e in E["ssw"] if not e.get("marginal") and keep(e)}
    idx = [i for i, d in enumerate(dates) if d in ok]
    return D.ao_ssw.values[idx], D.ao_base_ssw.values


def drip(D, E, s, out):
    n = E["sets"][s]
    lag = D.lag.values
    lev = D.lev.values
    keep = lev >= 1.0
    m = D.drip_mean.sel(set=s).values[:, keep].T
    tt = D.drip_t.sel(set=s).values[:, keep].T
    p = lev[keep]
    fig = plt.figure(figsize=(13.4, 8.4), dpi=125)
    ev = "sudden warmings" if s != "sv" else "strong-vortex events"
    fig_header(fig, f"Dripping paint: polar-cap height around the {n} {SET_LABEL[s]} since 1980",
               f"Standardised 65–90°N geopotential height anomaly, mean of {n} events, day 0 = "
               f"{'the first easterly day at 60°N, 10 hPa' if s != 'sv' else 'the 10 hPa annular index first crossing +1.5'}. "
               f"Red = high cap heights (weak vortex, negative AO), blue = low. Hatched where |t| < 2. {SRC}.")
    ax = fig.add_axes([0.07, 0.36, 0.84, 0.47])
    lv = np.arange(-2.4, 2.41, 0.3)
    cf = ax.contourf(lag, p, m, levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(lag, p, m, levels=[x for x in lv if abs(x) > 1e-6], colors="#333", linewidths=0.35)
    ax.contourf(lag, p, np.abs(tt), levels=[0, 2], colors="none", hatches=[".."])
    ax.set_yscale("log"); ax.set_ylim(1000, 1)
    ax.set_yticks([1000, 500, 300, 100, 50, 30, 10, 5, 3, 1]); ax.set_yticklabels(["1000", "500", "300", "100", "50", "30", "10", "5", "3", "1"])
    ax.axvline(0, color=INK, lw=1.0)
    for L in (100, 10):
        ax.axhline(L, color="#555", lw=0.5, ls=":")
    ax.set_ylabel("pressure, hPa", fontsize=10, color=INK)
    ax.set_xlim(lag[0], lag[-1])
    ax.tick_params(labelbottom=False)
    style(ax); ax.grid(False)
    ax.text(0.005, 1.01, "stratosphere above ~250 hPa, troposphere below", transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    cax = fig.add_axes([0.925, 0.36, 0.012, 0.47])
    cb = fig.colorbar(cf, cax=cax); cb.set_label("standard deviations", fontsize=9); cb.ax.tick_params(labelsize=8)
    # AO underneath
    A, base = ao_rows(D, E, s)
    ax2 = fig.add_axes([0.07, 0.08, 0.84, 0.23])
    sm = pd.DataFrame(A.T).rolling(7, center=True, min_periods=4).mean().values.T
    for r in sm:
        ax2.plot(lag, r, color="#9aa3ad", lw=0.6, alpha=0.55)
    q25, q75 = np.nanpercentile(sm, 25, 0), np.nanpercentile(sm, 75, 0)
    col = WARM if s != "sv" else COOL
    ax2.fill_between(lag, q25, q75, color=col, alpha=0.18, lw=0, label="middle half of events")
    ax2.plot(lag, np.nanmean(sm, 0), color=col, lw=2.6, label=f"mean of {len(sm)} events")
    ax2.plot(lag, pd.Series(base).rolling(7, center=True, min_periods=4).mean(), color=INK, lw=1.2, ls="--", label="same dates, other years")
    ax2.axhline(0, color="#555", lw=0.8); ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xlim(lag[0], lag[-1]); ax2.set_ylim(-4, 3)
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); ax2.set_ylabel("AO, 7-day mean", fontsize=10, color=INK)
    style(ax2)
    ax2.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    ax2.set_title("Arctic Oscillation at the surface (CPC daily index)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    save(fig, out)


# ---------------------------------------------------------------- surface maps
def follows_map(D, E, s, w, out):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    n = E["sets"][s]
    lat, lon = D.lat.values, D.lon.values
    m = D.t2m_mean.sel(set=s, window=w).values
    tt = D.t2m_t.sel(set=s, window=w).values
    z = D.z500_mean.sel(set=s, window=w).values
    fig = plt.figure(figsize=(10.4, 11.0), dpi=120)
    fig_header(fig, f"2 m temperature, {WIN_LABEL[w]} after {SET_LABEL[s]}",
               f"Mean anomaly over {n} events (ERA5, against each point's seasonal cycle and trend); contours: 500 hPa height "
               f"anomaly every 20 m (dashed negative). Dots: |t| ≥ 2 across events. North America at the bottom.", x=0.05)
    ax = polar_axes(fig, [0.03, 0.1, 0.94, 0.78], lat0=20, central=-80)
    pc = ccrs.PlateCarree()
    lv = np.arange(-3.5, 3.51, 0.5)
    mc, lc = cyclic(m, lon)
    cf = ax.contourf(lc, lat, mc, levels=lv, cmap="RdBu_r", extend="both", transform=pc)
    zc, _ = cyclic(z, lon)
    zl = [x for x in np.arange(-200, 201, 20) if x != 0]
    cs = ax.contour(lc, lat, zc, levels=zl, colors=INK, linewidths=0.8, transform=pc)
    ax.clabel(cs, fmt="%d", fontsize=7)
    sig = np.abs(tt) >= 2
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    pick = sig & (np.arange(len(lat))[:, None] % 2 == 0) & (np.arange(len(lon))[None, :] % 2 == 0)
    ax.scatter(LO[pick], LA[pick], s=1.2, color="#222", transform=pc, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
    ax.gridlines(lw=0.4, color="#999", ylocs=[30, 45, 60, 75], xlocs=np.arange(-180, 181, 45))
    cax = fig.add_axes([0.2, 0.065, 0.6, 0.013])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
    cb.ax.set_title("2 m temperature anomaly, K", fontsize=9, color=INK, pad=3)
    fig.text(0.05, 0.018, "Composites of so few events are noisy: read the large, dotted patterns, not single spots.",
             fontsize=8.8, color=MUTED)
    save(fig, out)


def ao_figure(D, E, out):
    fig = plt.figure(figsize=(13.4, 7.6), dpi=125)
    fig_header(fig, "The Arctic Oscillation after sudden warmings and strong-vortex events",
               f"CPC daily AO, 7-day running mean, around {E['sets']['ssw_all']} SSWs and {E['sets']['sv']} strong-vortex events "
               f"(MERRA-2 dates, 1980–2026). Lower panel: share of events with a negative AO, against the same calendar days in every other year.")
    lag = D.lag.values
    ax = fig.add_axes([0.07, 0.42, 0.9, 0.42])
    ax2 = fig.add_axes([0.07, 0.08, 0.9, 0.27])
    for s, col, lab in (("ssw_all", WARM, "after SSWs"), ("sv", COOL, "after strong-vortex events")):
        A, base = ao_rows(D, E, s)
        sm = pd.DataFrame(A.T).rolling(7, center=True, min_periods=4).mean().values.T
        ax.fill_between(lag, np.nanpercentile(sm, 25, 0), np.nanpercentile(sm, 75, 0), color=col, alpha=0.16, lw=0)
        ax.plot(lag, np.nanmean(sm, 0), color=col, lw=2.8, label=f"mean {lab} (n={len(sm)}), shading = middle half")
        neg = np.mean(sm < 0, 0) * 100
        ax2.plot(lag, neg, color=col, lw=2.2, label=lab)
    ax.axhline(0, color="#555", lw=0.8); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-40, 90); ax.set_ylim(-3.2, 2.4); ax.set_ylabel("AO, 7-day mean", fontsize=10, color=INK)
    ax.tick_params(labelbottom=False); style(ax); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.axhline(50, color=INK, lw=1, ls="--"); ax2.axvline(0, color=INK, lw=1)
    ax2.set_xlim(-40, 90); ax2.set_ylim(0, 100); ax2.set_ylabel("% of events AO < 0", fontsize=10, color=INK)
    ax2.set_xlabel("days from the event", fontsize=10, color=INK); style(ax2)
    ax2.text(-39, 52, "50 %: a coin flip", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    save(fig, out)


def regions_figure(E, out):
    R = list(E["regions"])
    S = ["ssw_all", "ssw_deep", "ssw_shallow", "ssw_split", "ssw_disp", "sv"]
    fig = plt.figure(figsize=(13.4, 6.6), dpi=125)
    fig_header(fig, "Did it turn cold? Regional 2 m temperature after each kind of event",
               "Land-only box means, ERA5, against each point's seasonal cycle and trend. Each cell: the mean anomaly and the share "
               "of events that ran colder than normal. Under each region: how often any 30-day window starting in Dec–Mar runs cold. "
               "With 10–28 events, a share within about 15 points of that is indistinguishable from chance.")
    for k, w in enumerate(("d1_30", "d31_60")):
        ax = fig.add_axes([0.13 + k * 0.44, 0.06, 0.4, 0.66])
        M = np.array([[np.mean(E["region_values"][f"{s}|{w}|{r}"]) for s in S] for r in R])
        P = np.array([[np.mean(np.array(E["region_values"][f"{s}|{w}|{r}"]) < 0) * 100 for s in S] for r in R])
        ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i in range(len(R)):
            for j in range(len(S)):
                ax.text(j, i, f"{M[i, j]:+.1f} K\n{P[i, j]:.0f}% cold", ha="center", va="center", fontsize=8.6,
                        color="white" if abs(M[i, j]) > 1.3 else INK, linespacing=1.2)
        ax.set_xticks(range(len(S))); ax.set_xticklabels([f"{SET_SHORT[s]}\n(n={E['sets'][s]})" for s in S], fontsize=8.6)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([f"{r}\n{E['region_base_pcold'][f'{w}|{r}'] * 100:.0f}% normally" for r in R] if k == 0 else [], fontsize=8.8)
        ax.set_title(WIN_LABEL[w] + " after", fontsize=11, fontweight="bold", color=INK, pad=34)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    save(fig, out)


# ---------------------------------------------------------------- base rates
def rates(E, out):
    W = pd.DataFrame(E["winters"])
    W["any"] = W.n_ssw > 0
    cur = E.get("current") or {}
    fig = plt.figure(figsize=(13.4, 6.6), dpi=125)
    below = fig_header(fig, "How often does a winter get a sudden warming? By ENSO, QBO and the solar cycle",
               f"{len(W)} winters 1980/81–2025/26, {int(W.n_ssw.sum())} major SSWs (Nov–Mar, MERRA-2). Bars: share of winters "
               f"with at least one; under the percentage, SSWs per winter and the number of winters. Sunspots split at the "
               f"median winter ({E['ssn_median']:.0f}). With 5–30 winters per bar, one winter moves a bar by 3–20 points.")
    groups = [("ENSO (RONI, DJF)", lambda c: W[W.enso == c], ["El Niño", "neutral", "La Niña"], cur.get("enso")),
              ("QBO at 50 hPa, Oct–Nov", lambda c: W[W.qbo == c], ["E", "W"], cur.get("qbo")),
              ("Sunspots, Oct–Feb", lambda c: W[W.sun == c], ["high", "low"], cur.get("sun")),
              ("ENSO and QBO together", lambda c: W[(W.enso == c[0]) & (W.qbo == c[1])],
               [(e, q) for e in ["El Niño", "neutral", "La Niña"] for q in ["E", "W"]], (cur.get("enso"), cur.get("qbo")))]
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 2, 2, 6], left=0.06, right=0.99, bottom=0.2, top=below - 0.1, wspace=0.28)
    for k, (title, sub, cats, hl) in enumerate(groups):
        ax = fig.add_subplot(gs[0, k])
        v = [sub(c) for c in cats]
        pct = [100 * x["any"].mean() if len(x) else 0 for x in v]
        cols = [WARM if c == hl else "#d9b8a8" for c in cats]
        ax.bar(range(len(cats)), pct, color=cols, width=0.72)
        for i, (p_, x) in enumerate(zip(pct, v)):
            ax.text(i, p_ + 9.5, f"{p_:.0f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
            ax.text(i, p_ + 1.5, f"{x.n_ssw.sum() / max(len(x), 1):.2f}/winter\nn = {len(x)}", ha="center", va="bottom", fontsize=7.8,
                    color=MUTED, linespacing=1.1)
        lab = [c if isinstance(c, str) else f"{c[0]}\nQBO {c[1]}" for c in cats]
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(lab, fontsize=9.2)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100])
        if k:
            ax.set_yticklabels([])
        ax.axhline(100 * W["any"].mean(), color=INK, lw=0.9, ls="--")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")
        style(ax); ax.grid(axis="x", visible=False)
    fig.axes[0].set_ylabel("% of winters with an SSW", fontsize=10, color=INK)
    fig.axes[0].text(2.45, 100 * W["any"].mean() + 1.5, f"all winters {100 * W['any'].mean():.0f}%", ha="right", va="bottom", fontsize=8.3, color=MUTED)
    if cur:
        fig.text(0.06, 0.03, f"Dark bars = where 2026/27 sits as of {cur.get('as_of', '')}: {cur.get('text', '')}",
                 fontsize=9.2, color=INK)
    save(fig, out)


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
    for s in D.set.values:
        drip(D, E, str(s), A / f"strat_hist_drip_{s}.webp")
        for w in D.window.values:
            follows_map(D, E, str(s), str(w), A / f"strat_hist_follows_{s}_{w}.webp")
    ao_figure(D, E, A / "strat_hist_ao.webp")
    regions_figure(E, A / "strat_hist_regions.webp")
    rates(E, A / "strat_hist_rates.webp")


if __name__ == "__main__":
    main()
