#!/usr/bin/env python3
"""Maps and charts of the Americas model-rain bias field (build_bias.py output) for rain-bias.html (2026-10-13).

  {comp}_{lead}_{season}_{abs|rel}.webp   one per comparison of build_bias.COMPARISONS (each model in ONE cycle: "current"
                                          IFS-ENS 50r1 + AIFS-ENS v2, "previous" 49r1 + v1, "ifs49" IFS-ENS 49r1 alone);
                                          shaded only where significant (block bootstrap, BH-FDR q 0.20 since 2026-10-13), grey elsewhere
  change_{model}_{old}_{new}_{lead}.webp  bias change across an upgrade (matched calendar days a year apart)
  truth_{season}.webp                     IMERG Late minus Final (Late's own error, the truth adjustment)
  leads_regions_{comp}.webp               regional mean bias by lead, 90 % intervals

    python scripts/precip_bias/render_bias.py --fields scripts/precip_bias/data/bias_fields.npz \
        --json assets/precip_bias/bias_summary.json --out assets/precip_bias
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G                                          # noqa: E402
from build_bias import LEADS, MODELS, REGIONS, SEASONS, COMPARISONS   # noqa: E402

ABS_LEV = [-6, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
REL_LEV = [-80, -60, -40, -25, -15, -5, 5, 15, 25, 40, 60, 100, 150]
CHG_LEV = [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3]
INSIG = "#dcdcdc"
SEASON_TXT = {"all": "all seasons", "DJF": "Dec-Feb", "MAM": "Mar-May", "JJA": "Jun-Aug", "SON": "Sep-Nov"}
EXTENT = [-170, -30, -60, 60]
DEG, QTXT, CITXT = 1, "BH-FDR q 0.10", "95 %"                    # set from the summary in main()
SHORT = {"Southeast South America": "SE South America", "Northeast Brazil": "NE Brazil", "East Pacific ITCZ": "E Pacific ITCZ"}


def cmap_for(levels):
    """Diverging brown (too dry) / green (too wet) with white on the bin that holds zero, each side spaced on its own
    (the percent levels are not symmetric)."""
    n = len(levels) + 1                                   # with both extensions
    z = int(np.searchsorted(levels, 0.0))                 # index of the bin containing zero
    cm = plt.get_cmap("BrBG")
    neg = cm(np.linspace(0.04, 0.42, z)) if z else np.empty((0, 4))
    pos = cm(np.linspace(0.58, 0.96, n - z - 1)) if n - z - 1 else np.empty((0, 4))
    cols = np.vstack([neg, [[1, 1, 1, 1]], pos])
    return ListedColormap(cols), BoundaryNorm(levels, n, extend="both")


def draw(ax, data, sig, levels, title, dry=None):
    cm, nm = cmap_for(levels)
    h = 0.5 * (G.LAT[1] - G.LAT[0])                                  # half a box (the grid comes from the fields file)
    lon_e = np.append(G.LON - h, G.LON[-1] + h); lat_e = np.append(G.LAT - h, G.LAT[-1] + h)
    show = np.where(sig & np.isfinite(data), data, np.nan)
    grey = np.where(~sig & np.isfinite(data), 1.0, np.nan)
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.pcolormesh(lon_e, lat_e, grey, cmap=ListedColormap([INSIG]), vmin=0, vmax=2, transform=ccrs.PlateCarree(), rasterized=True)
    pm = ax.pcolormesh(lon_e, lat_e, show, cmap=cm, norm=nm, transform=ccrs.PlateCarree(), rasterized=True)
    if dry is not None and dry.any():                      # too dry to express as a percent: hatched, not white
        plt.rcParams["hatch.color"] = "#9a9a9a"; plt.rcParams["hatch.linewidth"] = 0.5
        ax.contourf(G.LON, G.LAT, dry.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#333")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#555")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#777")
    for name, (y0, y1, x0, x1) in REGIONS.items():
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], lw=0.6, color="#222", ls=(0, (2, 2)), transform=ccrs.PlateCarree())
        lab = SHORT.get(name, name); east = x1 > -45
        ax.text((x1 - 0.6) if east else (x0 + 0.6), y1 - 0.6, lab, fontsize=6.5, va="top", ha="right" if east else "left",
                color="#111", transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="square,pad=0.1", fc=(1, 1, 1, 0.6), ec="none"))
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    return pm


def map_fig(panels, levels, unit, suptitle, foot, path):
    """panels: [(data, sig, title)]; geometry from the domain aspect (no letterboxing), fixed-inch text bands."""
    import textwrap
    n = len(panels)
    w_panel = 5.6; h_panel = w_panel * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])
    W = w_panel * n + 0.15 * (n - 1) + 0.2
    lines = textwrap.wrap(foot, width=int(W * 17))
    TOP, BOT = 0.62, 0.78 + 0.13 * len(lines)
    H = h_panel + TOP + BOT
    fig = plt.figure(figsize=(W, H), dpi=150)
    gs = fig.add_gridspec(1, n, left=0.1 / W, right=1 - 0.1 / W, top=1 - TOP / H, bottom=BOT / H, wspace=0.15 / w_panel)
    pm = None
    for i, pan in enumerate(panels):
        d, s, t = pan[:3]
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        pm = draw(ax, d, s, levels, t, dry=pan[3] if len(pan) > 3 else None)
    fig.text(0.1 / W, 1 - 0.1 / H, suptitle, fontsize=10.5, ha="left", va="top", weight="bold")
    cb_y = 0.13 * len(lines) + 0.42
    cax = fig.add_axes([0.5 - 2.6 / W, cb_y / H, 5.2 / W, 0.12 / H])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=7, pad=1.5); cb.ax.set_xticklabels([f"{l:g}".replace("-", "−") for l in levels])
    fig.text(0.5, (cb_y + 0.17) / H, unit, fontsize=8, ha="center", va="bottom")
    fig.text(0.1 / W, 0.06 / H, "\n".join(lines), fontsize=7, color="#444", ha="left", va="bottom", linespacing=1.25)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)


def lead_chart(summary, ck, path):
    comp = summary["comparisons"][ck]
    leads = [k for k in ("day1", "d2_3", "d4_7", "d8_10", "d11_14")]
    regs = list(REGIONS)
    ncol = 3; nrow = int(np.ceil(len(regs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(11, 2.3 * nrow + 0.6), dpi=150, sharex=True)
    col = {"ifs": "#1f5fa8", "aifs-ens": "#c2571a"}
    for ax, reg in zip(axs.ravel(), regs):
        for j, (m, lab) in enumerate(MODELS.items()):
            if m not in comp:
                continue
            lab = f"{lab} {comp[m]['cycle']}"
            xs, ys, lo, hi, sg = [], [], [], [], []
            for i, lk in enumerate(leads):
                r = comp[m]["leads"].get(lk, {}).get("all", {}).get("regions", {}).get(reg)
                if not r:
                    continue
                xs.append(i + (j - 0.5) * 0.18); ys.append(r["bias"]); lo.append(r["lo"]); hi.append(r["hi"]); sg.append(r["sig"])
            if not xs:
                continue
            ys = np.array(ys)
            ax.errorbar(xs, ys, yerr=[ys - np.array(lo), np.array(hi) - ys], fmt="none", ecolor=col[m], lw=1, capsize=2)
            ax.scatter(xs, ys, s=22, facecolors=[col[m] if s else "white" for s in sg], edgecolors=col[m], zorder=3,
                       label=lab if reg == regs[0] else None)
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_title(reg, fontsize=9, loc="left"); ax.tick_params(labelsize=7.5)
        ax.set_xticks(range(len(leads))); ax.set_xticklabels([LEADS[k][2] for k in leads], fontsize=7)
    for ax in axs.ravel()[len(regs):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("mm/day", fontsize=8)
    first = comp[next(m for m in MODELS if m in comp)]
    who = " and ".join(f"{MODELS[m]} {comp[m]['cycle']}" for m in MODELS if m in comp)
    same = "the same " if sum(m in comp for m in MODELS) > 1 else ""
    fig.suptitle(f"Rain bias by lead, regional means vs gauge-adjusted IMERG - {who}\n{same}{first['runs']} 00Z runs "
                 f"({first['first']} to {first['last']})", fontsize=10, x=0.01, ha="left", weight="bold")
    axs.ravel()[0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.text(0.01, 0.005, f"Bars: {CITXT} interval (21-day block bootstrap). Filled marker: significant (interval excludes 0); hollow: not.",
             fontsize=7, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(path, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.fields); S = json.loads(Path(a.json).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    G.LAT, G.LON = z["lat"], z["lon"]; G.NY, G.NX = len(G.LAT), len(G.LON)
    global DEG, QTXT, CITXT
    DEG = S.get("grid_deg", 1); QTXT = f"BH-FDR q {S.get('q_fdr', 0.10):.2f}"; CITXT = f"{100 * S.get('ci_level', 0.95):.0f} %"
    keys = set(str(k) for k in z["keys"])
    get = lambda k, v: z[f"{k}#{v}"]
    made = []
    for ck, C in S["comparisons"].items():
        ms = [m for m in MODELS if m in C]
        for lk, (_, _, ltxt) in LEADS.items():
            for sk in SEASONS:
                ks = [f"{ck}|{m}|{lk}|{sk}" for m in ms]
                if not all(k in keys for k in ks):
                    continue
                infos = [C[m]["leads"][lk][sk] for m in ms]
                who = " and ".join(f"{MODELS[m]} {C[m]['cycle']}" for m in ms)
                period = f"{infos[0]['first']} to {infos[0]['last']}"
                sig_txt = ", ".join(f"{MODELS[m]} {100 * i['frac_sig']:.0f} %" for m, i in zip(ms, infos))
                foot = (f"{'The same ' if len(ms) > 1 else ''}{infos[0]['runs']} 00Z runs{' for both models' if len(ms) > 1 else ''}, "
                        f"members 1-6, {DEG}° boxes; one model cycle each (never pooled across upgrades). Truth: IMERG V07 Late adjusted "
                        f"by its seasonal difference from the gauge-adjusted Final. Grey: not significant (21-day block bootstrap, "
                        f"{QTXT}); significant cells {sig_txt}. Percent maps: hatched where the observed mean is below 0.5 mm/day.")
                for kind, lev, unit in (("abs", ABS_LEV, "forecast minus observed rain (mm/day)"), ("rel", REL_LEV, "forecast / observed - 1 (%)")):
                    pan = []
                    for m, k in zip(ms, ks):
                        d = get(k, "bias") if kind == "abs" else 100 * get(k, "rel")
                        dry = (np.isfinite(get(k, "bias")) & ~np.isfinite(d)) if kind == "rel" else None
                        pan.append((d, get(k, "sig") & np.isfinite(d), f"{MODELS[m]} {C[m]['cycle']} {ltxt}, {SEASON_TXT[sk]}", dry))
                    pth = out / f"{ck}_{lk}_{sk}_{kind}.webp"
                    map_fig(pan, lev, unit, f"{who}: rain bias vs IMERG - {ltxt}, {SEASON_TXT[sk]}, {period}", foot, pth)
                    made.append(pth.name)
        if any(f"{ck}|{m}|d4_7|all" in keys for m in ms):
            lead_chart(S, ck, out / f"leads_regions_{ck}.webp"); made.append(f"leads_regions_{ck}.webp")
    for uk, U in S.get("upgrades", {}).items():
        m, ch = uk.split("|"); old, new = ch.split("->")
        for lk, (_, _, ltxt) in LEADS.items():
            k = f"change|{uk}|{lk}"
            if k not in keys:
                continue
            u = U["leads"][lk]
            foot = (f"{U['runs']} pairs of 00Z runs on the same calendar day a year apart ({new}: {U['new'][0]} to {U['new'][1]}; "
                    f"{old}: {U['old'][0]} to {U['old'][1]}). Green: the new cycle rains more relative to IMERG. Grey: no "
                    f"significant change ({QTXT}); {100 * u['frac_sig']:.0f} % of cells changed. The two years had different "
                    f"weather, so part of any change is the weather, not the model. Pattern correlation of the two bias fields "
                    f"r = {u['pattern_r']:.2f}.")
            pth = out / f"change_{m}_{old}_{new}_{lk}.webp"
            map_fig([(get(k, "bias"), get(k, "sig"), f"{MODELS[m]} {new} minus {old}, {ltxt}")], CHG_LEV, "change in forecast minus observed rain (mm/day)",
                    f"{MODELS[m]} {old} -> {new}: change in rain bias - {ltxt}", foot, pth)
            made.append(pth.name)
    for sk in ("DJF", "MAM", "JJA", "SON"):
        k = f"truth|{sk}"
        if k in keys:
            n = S.get("truth_days", {}).get(sk, 0)
            p = out / f"truth_{sk}.webp"
            map_fig([(get(k, "bias"), get(k, "sig"), f"IMERG Late minus Final, {SEASON_TXT[sk]}")], ABS_LEV, "Late minus Final (mm/day)",
                    f"IMERG V07 Late minus gauge-adjusted Final - {SEASON_TXT[sk]}, {n} days, {S['imerg_final'][0]} to {S['imerg_final'][1]}",
                    f"Late's own error: added to every forecast-minus-Late bias on this page. Grey: not significant ({QTXT}).", p)
            made.append(p.name)
    S["figures"] = sorted(made)
    Path(a.json).write_text(json.dumps(S, indent=1))
    print(f"{len(made)} figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
