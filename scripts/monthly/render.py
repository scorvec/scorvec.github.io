#!/usr/bin/env python3
"""Draw the month-ahead outlook figures from assets/monthly/data/outlook_<YYYYMM>.{npz,json} (forecast.py).

    python scripts/monthly/render.py                 # the outlook named by latest.json
    python scripts/monthly/render.py --target 2026-11

Figures (assets/monthly/, webp):
    outlook_<YYYYMM>_<var>[_nomask].webp     the outlook maps with and without the skill mask (kept for every issue);
                                             latest_*.webp copies
    components_<var>.webp                    each component's weighted contribution and the sum
    skill_<var>.webp                         cross-validated hindcast skill of this target month
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
CREDIT = (
    "scorvec.com · Claude's month-ahead forecast: statistical, from ERA5 (no dynamical model) · anomalies vs 1991–2020"
)

LEVELS = {
    "z500": np.array([-90, -75, -60, -48, -36, -24, -15, -8, 8, 15, 24, 36, 48, 60, 75, 90], float),
    "t2m": np.array([-4, -3, -2.5, -2, -1.5, -1, -0.6, -0.3, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 4], float),
}
UNIT = {"z500": "m", "t2m": "°C"}
TITLE = {"z500": "500 hPa height anomaly", "t2m": "2 m temperature anomaly"}


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


def shade(ax, v, var, alpha=1.0):
    f, lon = grid(v)
    m = ax.contourf(
        lon, G.LAT_NH, f, levels=LEVELS[var], cmap="RdBu_r", extend="both", alpha=alpha, transform=ccrs.PlateCarree()
    )
    m.set_edgecolor("face")
    return m


def outlook_map(o: dict, s: dict, var: str, path: Path, mask: bool = True) -> None:
    fc = o[f"{var}_fc"]
    sig = o[f"{var}_sig"].astype(float)
    fig = plt.figure(figsize=(7.6, 8.4))
    ax = polar_ax(fig, [0.04, 0.12, 0.92, 0.74])
    shade(ax, fc, var)
    if mask:  # where the hindcast shows no significant skill the forecast is drawn faded under grey hatching
        sg, lon = grid(sig)
        ax.contourf(lon, G.LAT_NH, sg, levels=[-1, 0.5], colors=["white"], alpha=0.55, transform=ccrs.PlateCarree())
        ax.contourf(lon, G.LAT_NH, sg, levels=[-1, 0.5], colors="none", hatches=["////"], transform=ccrs.PlateCarree())
    if var == "z500":
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
    cax = fig.add_axes([0.14, 0.085, 0.72, 0.02])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.BoundaryNorm(LEVELS[var], 256, extend="both"))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(f"{TITLE[var]} ({UNIT[var]})", fontsize=8.5)
    sv = s["vars"][var]
    fig.text(0.04, 0.965, f"{s['target_label']}: {TITLE[var]}", fontsize=14, fontweight="bold", color=INK)
    sub = f"Issued {s['issued']} from ERA5 through {s['window'].split('–')[1].strip()}"
    if var == "z500":
        sub += " · contours: forecast monthly-mean height (dam)"
    fig.text(0.04, 0.937, sub, fontsize=9, color=MUTED)
    fig.text(
        0.04,
        0.912,
        f"Hindcast {s['hindcast'][0]}–{s['hindcast'][1]} (cross-validated): mean NH ACC {sv['nh_acc_mean']:+.2f}; "
        f"skill significant over {100 * sv['skill_area_sig']:.0f}% of 20–90N"
        + (" (hatched: not significant, FDR 10%)" if mask else " (skill mask off)"),
        fontsize=8,
        color=MUTED,
    )
    fig.text(0.5, 0.025, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def components_fig(o: dict, s: dict, var: str, path: Path) -> None:
    contrib = o[f"{var}_contrib"]
    beta = o[f"{var}_beta"]
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w = (w * (G.LAT_NH[:, None] >= LAT0)).ravel()
    fig = plt.figure(figsize=(13.2, 7.6))
    panels = [*list(enumerate(M.COMPONENTS)), (None, "sum")]
    for j, (i, c) in enumerate(panels):
        r, col = divmod(j, 3)
        ax = polar_ax(fig, [0.01 + col * 0.33, 0.52 - r * 0.44, 0.31, 0.37])
        v = contrib[i] if i is not None else o[f"{var}_fc"]
        shade(ax, v, var)
        if i is not None:
            wb = float((w * beta[:, i]).sum() / w.sum())
            lab = f"{M.NAMES[c]}\nmean weight {wb:.2f}"
        else:
            lab = "Forecast (sum of the five)"
        ax.set_title(lab, fontsize=9.5, fontweight="bold", pad=3)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.018])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.BoundaryNorm(LEVELS[var], 256, extend="both"))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"{TITLE[var]} ({UNIT[var]}), each component times its consolidation weight", fontsize=8)
    fig.text(0.01, 0.965, f"{s['target_label']}: how the {TITLE[var]} forecast is built", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=95, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def skill_fig(o: dict, s: dict, var: str, path: Path) -> None:
    r = o[f"{var}_r"]
    sig = o[f"{var}_sig"].astype(float)
    fig = plt.figure(figsize=(12.0, 5.6))
    ax = polar_ax(fig, [0.01, 0.1, 0.42, 0.78])
    f, lon = grid(r)
    lv = np.array([-0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6, 0.75])
    m = ax.contourf(lon, G.LAT_NH, f, levels=lv, cmap="PuOr_r", extend="both", transform=ccrs.PlateCarree())
    m.set_edgecolor("face")
    sg, lon = grid(sig)
    ax.contour(lon, G.LAT_NH, sg, levels=[0.5], colors=INK, linewidths=0.9, transform=ccrs.PlateCarree())
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("correlation, forecast vs ERA5 (black line: significant, FDR 10%)", fontsize=8)
    ax.set_title(
        f"Cross-validated skill, {M.target_label(0, int(s['target'][5:])).split()[0]} {TITLE[var]}",
        fontsize=10,
        fontweight="bold",
    )
    bx = fig.add_axes([0.52, 0.17, 0.45, 0.66])
    acc = s["vars"][var]["nh_acc_by_year"]
    ys = np.array([int(k) for k in acc])
    vals = np.array(list(acc.values()))
    bx.bar(ys, vals, color=np.where(vals >= 0, "#c2410c", "#2563eb"), width=0.8)
    bx.axhline(0, color=INK, lw=0.6)
    bx.axhline(vals.mean(), color=MUTED, lw=0.9, ls="--")
    bx.set_ylim(-1, 1)
    bx.set_ylabel("anomaly correlation, 20–90N", fontsize=9)
    bx.set_title(
        f"Hindcast by year (mean {vals.mean():+.2f}; positive in {100 * (vals > 0).mean():.0f}% of years)",
        fontsize=10,
        fontweight="bold",
    )
    bx.tick_params(labelsize=8)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.text(0.5, 0.02, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=95, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
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
    for var in M.TARGETS:
        for mask, suf in ((True, ""), (False, "_nomask")):
            p = ASSETS / f"outlook_{tag}_{var}{suf}.webp"
            outlook_map(o, s, var, p, mask)
            shutil.copyfile(p, ASSETS / f"latest_{var}{suf}.webp")
        components_fig(o, s, var, ASSETS / f"components_{var}.webp")
        skill_fig(o, s, var, ASSETS / f"skill_{var}.webp")
        print(f"drew {var}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
