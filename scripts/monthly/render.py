#!/usr/bin/env python3
"""Draw the month-ahead outlook figures from assets/monthly/data/outlook_<YYYYMM>.{npz,json} (forecast.py).

    python scripts/monthly/render.py                 # the outlook named by latest.json
    python scripts/monthly/render.py --target 2026-11

Figures (assets/monthly/, webp), for each variable (z500, t2m, prcp):
    latest_<var>_<region>_<kind>[_nomask].webp   region nh / na / eu, kind anom (expected anomaly) or prob (tercile
                                                 probabilities), with and without the skill mask
    outlook_<YYYYMM>_<var>_<kind>.webp           the hemispheric maps of every issue, kept (skill mask on)
    components_<var>.webp                        each component's weighted contribution and the sum
    blend_<var>.webp                             statistical forecast, model mean and blend (when there is guidance)
    skill_<var>.webp                             cross-validated hindcast skill of this target month
"""

from __future__ import annotations

import datetime

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_grid as G
import mo_method as M

REPO = HERE.parents[1]
ASSETS = REPO / "assets" / "monthly"
DATA = ASSETS / "data"
LAT0 = 20.0
INK, MUTED = "#1a1a1a", "#5b6470"
CREDIT = "scorvec.com · Claude's month-ahead forecast · ERA5 statistics, blended with GEFS days 1–15 where it adds skill · anomalies vs 1991–2020"

LEVELS = {
    "z500": np.array([-90, -75, -60, -48, -36, -24, -15, -8, 8, 15, 24, 36, 48, 60, 75, 90], float),
    "t2m": np.array([-4, -3, -2.5, -2, -1.5, -1, -0.6, -0.3, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 4], float),
    "prcp": np.array([-3, -2, -1.5, -1, -0.6, -0.3, -0.15, 0.15, 0.3, 0.6, 1, 1.5, 2, 3], float),
}
CMAP = {"z500": "RdBu_r", "t2m": "RdBu_r", "prcp": "BrBG"}
UNIT = {"z500": "m", "t2m": "°C", "prcp": "mm/day"}
TITLE = {"z500": "500 hPa height anomaly", "t2m": "2 m temperature anomaly", "prcp": "precipitation anomaly"}
NAME = {"z500": "500 hPa height", "t2m": "2 m temperature", "prcp": "precipitation"}
# tercile categories: what "above" and "below" normal mean for each variable, and their colours
CAT = {
    "z500": (("above", "Reds"), ("below", "Blues")),
    "t2m": (("above", "Reds"), ("below", "Blues")),
    "prcp": (("above", "Greens"), ("below", "YlOrBr")),
}
PROB_LEVELS = np.array([40, 50, 60, 70, 80, 100], float)

REGIONS = {
    "nh": ("Northern Hemisphere", None, None),
    "na": ("North America", ccrs.LambertConformal(central_longitude=-100, central_latitude=45), [-160, -55, 21, 72]),
    "eu": ("Europe", ccrs.LambertConformal(central_longitude=15, central_latitude=52), [-25, 55, 32, 72]),
}


def grid(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = v.reshape(len(G.LAT_NH), len(G.LON))
    f, lon = add_cyclic_point(f, coord=G.LON)
    return f, lon


def polar_ax(fig, rect):
    ax = fig.add_axes(rect, projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-180, 180, LAT0, 90], crs=ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.vstack([np.sin(th), np.cos(th)]).T * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.45, edgecolor="#3a3a3a")
    ax.gridlines(lw=0.3, color="#9aa3ad", alpha=0.6, xlocs=range(-180, 180, 30), ylocs=[30, 45, 60, 75])
    return ax


def map_ax(fig, rect, region: str):
    if region == "nh":
        return polar_ax(fig, rect)
    _, proj, ext = REGIONS[region]
    ax = fig.add_axes(rect, projection=proj)
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#3a3a3a")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#777")
    if region == "na":
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#999")
    ax.gridlines(lw=0.3, color="#9aa3ad", alpha=0.6, xlocs=range(-180, 180, 20), ylocs=range(10, 90, 10))
    return ax


def shade(ax, v, var, alpha=1.0):
    f, lon = grid(v)
    m = ax.contourf(
        lon, G.LAT_NH, f, levels=LEVELS[var], cmap=CMAP[var], extend="both", alpha=alpha, transform=ccrs.PlateCarree()
    )
    m.set_edgecolor("face")
    return m


def hatch(ax, sig):
    """Fade and hatch where the hindcast shows no significant skill."""
    sg, lon = grid(sig.astype(float))
    ax.contourf(lon, G.LAT_NH, sg, levels=[-1, 0.5], colors=["white"], alpha=0.55, transform=ccrs.PlateCarree())
    ax.contourf(lon, G.LAT_NH, sg, levels=[-1, 0.5], colors="none", hatches=["////"], transform=ccrs.PlateCarree())


def prob_layers(o: dict, var: str, raw: bool = False) -> list[tuple[np.ndarray, str]]:
    """The tercile-probability map as CPC draws it: where above (below) normal is the most likely category and
    reaches 40 %, its probability in that category's colours; elsewhere blank (near normal or equal chances). The
    issued probabilities are equal chances where the hindcast has no skill; raw=True shows them unmasked."""
    suf = "_raw" if raw else ""
    pa, pb = 100 * o[f"{var}_p_above{suf}"], 100 * o[f"{var}_p_below{suf}"]
    pn = 100 - pa - pb
    up = np.where((pa >= 40) & (pa > pb) & (pa > pn), pa, np.nan)
    dn = np.where((pb >= 40) & (pb > pa) & (pb > pn), pb, np.nan)
    return [(up, CAT[var][0][1]), (dn, CAT[var][1][1])]


def draw_prob(ax, o, var, raw=False):
    for f, cmap in prob_layers(o, var, raw):
        g, lon = grid(f)
        cm = plt.get_cmap(cmap)
        cols = cm(np.linspace(0.3, 0.95, len(PROB_LEVELS) - 1))
        m = ax.contourf(lon, G.LAT_NH, g, levels=PROB_LEVELS, colors=cols, transform=ccrs.PlateCarree())
        m.set_edgecolor("face")


def prob_legend(fig, rect, var):
    """Two small colour bars: above normal and below normal."""
    for (lab, cmap), x0 in zip(CAT[var], (rect[0], rect[0] + rect[2] / 2 + 0.02)):
        cax = fig.add_axes([x0, rect[1], rect[2] / 2 - 0.02, rect[3]])
        cols = plt.get_cmap(cmap)(np.linspace(0.3, 0.95, len(PROB_LEVELS) - 1))
        cmp = mcolors.ListedColormap(cols)
        norm = mcolors.BoundaryNorm(PROB_LEVELS, cmp.N)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax, orientation="horizontal")
        cb.set_ticks(PROB_LEVELS[:-1])
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(f"probability of {lab} normal (%)", fontsize=8.5)


def outlook_map(o: dict, s: dict, var: str, region: str, kind: str, path: Path, mask: bool = True) -> None:
    fc = o[f"{var}_fc"]
    nh = region == "nh"
    fig = plt.figure(figsize=(7.6, 8.4) if nh else (8.6, 7.4))
    rect = [0.04, 0.135, 0.92, 0.735] if nh else [0.03, 0.15, 0.94, 0.71]
    ax = map_ax(fig, rect, region)
    if kind == "anom":
        shade(ax, fc, var)
    else:
        draw_prob(ax, o, var, raw=not mask)
    if mask:
        hatch(ax, o[f"{var}_sig"])
    if var == "z500" and kind == "anom":
        full, lon2 = grid(o["z500_clim"] + fc)
        cs = ax.contour(
            lon2,
            G.LAT_NH,
            full / 10.0,
            levels=np.arange(480, 600, 6),
            colors=INK,
            linewidths=0.7,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6.5, fmt="%d")
    if kind == "anom":
        cax = fig.add_axes([0.14, 0.09, 0.72, 0.02])
        sm = plt.cm.ScalarMappable(cmap=CMAP[var], norm=mcolors.BoundaryNorm(LEVELS[var], 256, extend="both"))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(f"{TITLE[var]} ({UNIT[var]})", fontsize=8.5)
    else:
        prob_legend(fig, [0.1, 0.09, 0.8, 0.02], var)
    sv = s["vars"][var]
    what = TITLE[var] if kind == "anom" else f"{NAME[var]}, tercile probabilities"
    where = "" if nh else f" · {REGIONS[region][0]}"
    fig.text(0.04, 0.965, f"{s['target_label']}: {what}{where}", fontsize=13.5, fontweight="bold", color=INK)
    sub = f"Issued {s['issued']} from ERA5 through {s['window'].split('–')[1].strip()}"
    b = s.get("blend", {}).get(var, {})
    if b.get("issued"):
        init = datetime.date.fromisoformat(b["init"])
        sub += f" and GEFS days {b['days'][0]}–{b['days'][1]} ({init.day} {init:%b} run)"
    if var == "z500" and kind == "anom":
        sub += " · contours: forecast height (dam)"
    if kind == "prob":
        sub += " · blank: no category above 40% (or equal chances)"
    fig.text(0.04, 0.937, sub, fontsize=8.6, color=MUTED)
    if b.get("issued"):
        bs, ss = b["scores"]["blend"], b["scores"]["stat"]
        skill = (
            f"Cross-validated NH ACC {sv['nh_acc_mean']:+.2f} ({s['hindcast'][0]}–{s['hindcast'][1]}); "
            f"{b['years'][0]}–{b['years'][1]}: blend {bs['acc']:+.2f}, statistical alone {ss['acc']:+.2f}"
        )
    else:
        skill = (
            f"Hindcast {s['hindcast'][0]}–{s['hindcast'][1]} (cross-validated): NH ACC {sv['nh_acc_mean']:+.2f}, "
            f"RPSS {sv['rpss_nh']:+.2f}; skill significant over {100 * sv['skill_area_sig']:.0f}% of 20–90N"
        )
    skill += " (hatched: not significant)" if mask else " (skill mask off)"
    fig.text(0.04, 0.912, skill, fontsize=8, color=MUTED)
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=105 if nh else 100, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def components_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """The statistical forecast's five weighted components and their sum (before any blend with GEFS)."""
    contrib = o[f"{var}_contrib"]
    beta = o[f"{var}_beta"]
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w = (w * (G.LAT_NH[:, None] >= LAT0)).ravel()
    fig = plt.figure(figsize=(13.2, 7.6))
    panels = [*list(enumerate(M.COMPONENTS)), (None, "sum")]
    for j, (i, c) in enumerate(panels):
        r, col = divmod(j, 3)
        ax = polar_ax(fig, [0.01 + col * 0.33, 0.52 - r * 0.44, 0.31, 0.37])
        v = contrib[i] if i is not None else o.get(f"{var}_stat_fc", o[f"{var}_fc"])
        shade(ax, v, var)
        if i is not None:
            wb = float((w * beta[:, i]).sum() / w.sum())
            lab = f"{M.NAMES[c]}\nmean weight {wb:.2f}"
        else:
            lab = "Statistical forecast (sum of the five)"
        ax.set_title(lab, fontsize=9.5, fontweight="bold", pad=3)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP[var], norm=mcolors.BoundaryNorm(LEVELS[var], 256, extend="both"))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"{TITLE[var]} ({UNIT[var]}), each component times its consolidation weight", fontsize=8)
    fig.text(0.01, 0.965, f"{s['target_label']}: how the {NAME[var]} forecast is built", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=95, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def blend_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """The statistical forecast, GEFS's first half of the month and their blend, with the weights and the
    shared-years cross-validated scores (against the whole month); the issued one is marked."""
    b = s["blend"][var]
    sc = b["scores"]
    issued = "blend" if b["issued"] else "stat"
    stat = o.get(f"{var}_stat_fc", o[f"{var}_fc"])
    panels = [
        ("stat", stat, "Statistical (this page's method)"),
        ("models", o[f"{var}_models_fc"], f"GEFS, days {b['days'][0]}–{b['days'][1]} ({b['init']} run)"),
        ("blend", o[f"{var}_blend_fc"], "Blend"),
    ]
    fig = plt.figure(figsize=(13.2, 5.9))
    for j, (k, fld, lab) in enumerate(panels):
        ax = polar_ax(fig, [0.01 + j * 0.33, 0.17, 0.31, 0.66])
        shade(ax, fld, var)
        line = f"ACC {sc[k]['acc']:+.2f} · RPSS {sc[k]['rpss']:+.2f}"
        if k == "blend":
            line += f" · weights {b['weight_nh']['stat']:.2f} / {b['weight_nh']['models']:.2f}"
        mark = "  ← issued" if k == issued else ""
        ax.set_title(f"{lab}{mark}\n{line}", fontsize=9.5, fontweight="bold", pad=4)
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP[var], norm=mcolors.BoundaryNorm(LEVELS[var], 256, extend="both"))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"{TITLE[var]} ({UNIT[var]}); scores: cross-validated, {b['years'][0]}–{b['years'][1]}", fontsize=8)
    fig.text(
        0.01,
        0.95,
        f"{s['target_label']}: {NAME[var]}, statistical forecast and GEFS",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=92, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def skill_map(fig, rect, field, sig, title, label):
    ax = polar_ax(fig, rect)
    f, lon = grid(field)
    lv = np.array([-0.6, -0.45, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75])
    m = ax.contourf(lon, G.LAT_NH, f, levels=lv, cmap="PuOr_r", extend="both", transform=ccrs.PlateCarree())
    m.set_edgecolor("face")
    if sig is not None:
        sg, lon = grid(sig.astype(float))
        ax.contour(lon, G.LAT_NH, sg, levels=[0.5], colors=INK, linewidths=0.9, transform=ccrs.PlateCarree())
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, ticks=lv[::2])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=7.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)


def skill_fig(o: dict, s: dict, var: str, path: Path) -> None:
    sv = s["vars"][var]
    mon = M.target_label(0, int(s["target"][5:])).split()[0]
    fig = plt.figure(figsize=(15.0, 5.6))
    skill_map(
        fig,
        [0.0, 0.14, 0.3, 0.72],
        o[f"{var}_r"],
        o[f"{var}_sig"],
        "Correlation with ERA5",
        "black line: significant (one-sided 5%, FDR 10%)",
    )
    skill_map(
        fig,
        [0.3, 0.14, 0.3, 0.72],
        o[f"{var}_msss_trend"],
        None,
        "Skill beyond the trend (MSSS vs trend alone)",
        f"> 0: better than the trend component by itself (20–90N: {sv['msss_trend_nh']:+.2f})",
    )
    bx = fig.add_axes([0.66, 0.2, 0.32, 0.63])
    acc = sv["nh_acc_by_year"]
    ys = np.array([int(k) for k in acc])
    vals = np.array(list(acc.values()))
    bx.bar(ys, vals, color=np.where(vals >= 0, "#c2410c", "#2563eb"), width=0.8, label="forecast")
    tr = sv.get("acc_trend_by_year")
    if tr:
        bx.plot(ys, [tr[str(y)] for y in ys], color=INK, lw=0.9, marker="o", ms=2.2, label="trend alone")
        bx.legend(fontsize=7.5, loc="lower left", frameon=False)
    bx.axhline(0, color=INK, lw=0.6)
    bx.set_ylim(-1, 1)
    bx.set_ylabel("anomaly correlation, 20–90N", fontsize=9)
    bx.set_title(
        f"By year: mean {vals.mean():+.2f} (trend alone {sv['acc_trend_mean']:+.2f}), RPSS {sv['rpss_nh']:+.2f}",
        fontsize=9.5,
        fontweight="bold",
    )
    bx.tick_params(labelsize=8)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.text(
        0.01,
        0.95,
        f"Cross-validated hindcast {s['hindcast'][0]}–{s['hindcast'][1]}: {mon} {NAME[var]}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(0.5, 0.008, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=92, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    a = ap.parse_args()
    if a.target:
        tag = a.target.replace("-", "")
    else:
        tag = json.loads((DATA / "latest.json").read_text())["target"].replace("-", "")
    s = json.loads((DATA / f"outlook_{tag}.json").read_text())
    with np.load(DATA / f"outlook_{tag}.npz") as f:
        o = {k: f[k] for k in f.files}
    ASSETS.mkdir(parents=True, exist_ok=True)
    for var in s["vars"]:
        for region in REGIONS:
            for kind in ("anom", "prob"):
                for mask, suf in ((True, ""), (False, "_nomask")):
                    p = ASSETS / f"latest_{var}_{region}_{kind}{suf}.webp"
                    outlook_map(o, s, var, region, kind, p, mask)
                    if region == "nh" and mask:
                        shutil.copyfile(p, ASSETS / f"outlook_{tag}_{var}_{kind}.webp")
        components_fig(o, s, var, ASSETS / f"components_{var}.webp")
        if var in s.get("blend", {}) and f"{var}_blend_fc" in o:
            blend_fig(o, s, var, ASSETS / f"blend_{var}.webp")
        skill_fig(o, s, var, ASSETS / f"skill_{var}.webp")
        print(f"drew {var}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
