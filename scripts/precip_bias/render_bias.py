#!/usr/bin/env python3
"""Maps and charts of the Americas model-rain bias field (build_bias.py output) for rain-bias.html (2026-10-13).

  {comp}_{lead}_{season}_{abs|rel}.webp   comp "ifs_full" (IFS-ENS, whole record, one panel) or "common" (IFS-ENS and
                                          AIFS-ENS over the same runs, two panels); shaded only where significant
                                          (block bootstrap, BH-FDR q 0.10), grey elsewhere
  truth_{season}.webp                     IMERG Late minus Final (Late's own error, the truth adjustment)
  leads_regions.webp                      regional mean bias by lead, both models, 95 % intervals

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
from build_bias import LEADS, MODELS, REGIONS, SEASONS   # noqa: E402

ABS_LEV = [-6, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
REL_LEV = [-80, -60, -40, -25, -15, -5, 5, 15, 25, 40, 60, 100, 150]
INSIG = "#dcdcdc"
SEASON_TXT = {"all": "all seasons", "DJF": "Dec-Feb", "MAM": "Mar-May", "JJA": "Jun-Aug", "SON": "Sep-Nov"}
EXTENT = [-170, -30, -60, 60]
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


def draw(ax, data, sig, levels, title):
    cm, nm = cmap_for(levels)
    lon_e = np.append(G.LON - 0.5, G.LON[-1] + 0.5); lat_e = np.append(G.LAT - 0.5, G.LAT[-1] + 0.5)
    show = np.where(sig & np.isfinite(data), data, np.nan)
    grey = np.where(~sig & np.isfinite(data), 1.0, np.nan)
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.pcolormesh(lon_e, lat_e, grey, cmap=ListedColormap([INSIG]), vmin=0, vmax=2, transform=ccrs.PlateCarree(), rasterized=True)
    pm = ax.pcolormesh(lon_e, lat_e, show, cmap=cm, norm=nm, transform=ccrs.PlateCarree(), rasterized=True)
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
    for i, (d, s, t) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        pm = draw(ax, d, s, levels, t)
    fig.text(0.1 / W, 1 - 0.1 / H, suptitle, fontsize=10.5, ha="left", va="top", weight="bold")
    cb_y = 0.13 * len(lines) + 0.42
    cax = fig.add_axes([0.5 - 2.6 / W, cb_y / H, 5.2 / W, 0.12 / H])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=7, pad=1.5)
    fig.text(0.5, (cb_y + 0.17) / H, unit, fontsize=8, ha="center", va="bottom")
    fig.text(0.1 / W, 0.06 / H, "\n".join(lines), fontsize=7, color="#444", ha="left", va="bottom", linespacing=1.25)
    fig.savefig(path, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)


def lead_chart(summary, path):
    comp = summary["comparisons"].get("common") or summary["comparisons"]["ifs_full"]
    leads = [k for k in ("day1", "d2_3", "d4_7", "d8_10", "d11_14")]
    regs = list(REGIONS)
    ncol = 3; nrow = int(np.ceil(len(regs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(11, 2.3 * nrow + 0.6), dpi=150, sharex=True)
    col = {"ifs": "#1f5fa8", "aifs-ens": "#c2571a"}
    for ax, reg in zip(axs.ravel(), regs):
        for j, (m, lab) in enumerate(MODELS.items()):
            if m not in comp:
                continue
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
    first = comp[next(iter(comp))]
    fig.suptitle(f"Rain bias by lead, regional means vs gauge-adjusted IMERG - IFS-ENS and AIFS-ENS, the same "
                 f"{first['runs']} 00Z runs ({first['first']} to {first['last']})", fontsize=10, x=0.01, ha="left", weight="bold")
    axs.ravel()[0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.text(0.01, 0.005, "Bars: 95 % interval (21-day block bootstrap). Filled marker: significant (|z| > 1.96); hollow: not.",
             fontsize=7, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.fields); S = json.loads(Path(a.json).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = set(str(k) for k in z["keys"])
    get = lambda k, v: z[f"{k}#{v}"]
    made = []
    for lk, (_, _, ltxt) in LEADS.items():
        for sk in SEASONS:
            # IFS-ENS, whole record
            k = f"ifs_full|ifs|{lk}|{sk}"
            if k in keys:
                info = S["comparisons"]["ifs_full"]["ifs"]; li = info["leads"][lk][sk]
                period = f"{info['first']} to {info['last']}"
                foot = (f"{li['runs']} 00Z runs, members 1-6, 1° boxes. Truth: IMERG V07 Late adjusted by its seasonal difference from the "
                        f"gauge-adjusted Final. Grey: not significant (21-day block bootstrap, BH-FDR q 0.10); "
                        f"{100 * li['frac_sig']:.0f} % of cells significant.")
                for kind, lev, unit in (("abs", ABS_LEV, "forecast minus observed rain (mm/day)"), ("rel", REL_LEV, "forecast / observed - 1 (%)")):
                    d = get(k, "bias") if kind == "abs" else 100 * get(k, "rel")
                    sg = get(k, "sig") & np.isfinite(d)
                    p = out / f"ifs_full_{lk}_{sk}_{kind}.webp"
                    map_fig([(d, sg, f"IFS-ENS {ltxt}, {SEASON_TXT[sk]}")], lev, unit,
                            f"IFS-ENS rain bias vs IMERG - {ltxt}, {SEASON_TXT[sk]}, {period}", foot, p)
                    made.append(p.name)
            # IFS-ENS vs AIFS-ENS, the same runs
            k1, k2 = f"common|ifs|{lk}|{sk}", f"common|aifs-ens|{lk}|{sk}"
            if k1 in keys and k2 in keys:
                info = S["comparisons"]["common"]["ifs"]; li = info["leads"][lk][sk]
                li2 = S["comparisons"]["common"]["aifs-ens"]["leads"][lk][sk]
                period = f"{info['first']} to {info['last']}"
                foot = (f"The same {li['runs']} 00Z runs for both models, members 1-6, 1° boxes. Truth: IMERG V07 Late adjusted to the gauge-adjusted "
                        f"Final. Grey: not significant (21-day block bootstrap, BH-FDR q 0.10); significant cells IFS-ENS "
                        f"{100 * li['frac_sig']:.0f} %, AIFS-ENS {100 * li2['frac_sig']:.0f} %.")
                for kind, lev, unit in (("abs", ABS_LEV, "forecast minus observed rain (mm/day)"), ("rel", REL_LEV, "forecast / observed - 1 (%)")):
                    pan = []
                    for kk, lab in ((k1, "IFS-ENS"), (k2, "AIFS-ENS")):
                        d = get(kk, "bias") if kind == "abs" else 100 * get(kk, "rel")
                        pan.append((d, get(kk, "sig") & np.isfinite(d), f"{lab} {ltxt}, {SEASON_TXT[sk]}"))
                    p = out / f"common_{lk}_{sk}_{kind}.webp"
                    map_fig(pan, lev, unit, f"IFS-ENS and AIFS-ENS rain bias vs IMERG - {ltxt}, {SEASON_TXT[sk]}, {period}", foot, p)
                    made.append(p.name)
    for sk in ("DJF", "MAM", "JJA", "SON"):
        k = f"truth|{sk}"
        if k in keys:
            n = S.get("truth_days", {}).get(sk, 0)
            p = out / f"truth_{sk}.webp"
            map_fig([(get(k, "bias"), get(k, "sig"), f"IMERG Late minus Final, {SEASON_TXT[sk]}")], ABS_LEV, "Late minus Final (mm/day)",
                    f"IMERG V07 Late minus gauge-adjusted Final - {SEASON_TXT[sk]}, {n} days, {S['imerg_final'][0]} to {S['imerg_final'][1]}",
                    "Late's own error: added to every forecast-minus-Late bias on this page. Grey: not significant (BH-FDR q 0.10).", p)
            made.append(p.name)
    lead_chart(S, out / "leads_regions.webp"); made.append("leads_regions.webp")
    S["figures"] = sorted(made)
    Path(a.json).write_text(json.dumps(S, indent=1))
    print(f"{len(made)} figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
