#!/usr/bin/env python3
"""Draw the outlook figures of the newest issue (latest.json): the month and the three seasons (forecast.py).

    python scripts/monthly/render.py

Figures (assets/monthly/, webp), per target key (m1 = the month, s1-s3 = the seasons from it) and variable (z500,
t2m, prcp):
    fc_<key>_<var>_<region>_<kind>.webp   region nh / na / eu, kind anom (expected anomaly) or prob (tercile
                                          probabilities), dotted where the hindcast has significant skill
    ingredients_<key>_<var>.webp          the statistical forecast, the C3S mean, GEFS + GEPS (the month) and the issue
    components_<key>_<var>.webp           the statistical forecast's six weighted components and their sum
    skill_<key>_<var>.webp                the issued forecast's hindcast correlation and the yearly scores
    outlook_<YYYYMM>_<var>_<kind>.webp    the month's hemispheric maps of every issue, kept
    update_<var>.webp                     an update's two parts (forecast.py --update)
"""

from __future__ import annotations

import argparse
import calendar
import datetime
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import house

house.apply()
import numpy as np
from cartopy.util import add_cyclic_point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_grid as G

REPO = HERE.parents[1]
ASSETS = REPO / "assets" / "monthly"
DATA = ASSETS / "data"
LAT0 = 20.0
INK, MUTED = house.INK, house.MUTED
CREDIT = "scorvec.com · Claude's long-range forecast · anomalies vs 1991–2020"

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
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.45, edgecolor="#3a3a3a", zorder=4)
    ax.gridlines(lw=0.3, color="#9aa3ad", alpha=0.6, xlocs=range(-180, 180, 30), ylocs=[30, 45, 60, 75])
    return ax


def map_ax(fig, rect, region: str):
    if region == "nh":
        return polar_ax(fig, rect)
    _, proj, ext = REGIONS[region]
    ax = fig.add_axes(rect, projection=proj)
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#3a3a3a", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#777", zorder=4)
    if region == "na":
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#999", zorder=4)
    ax.gridlines(lw=0.3, color="#9aa3ad", alpha=0.6, xlocs=range(-180, 180, 20), ylocs=range(10, 90, 10))
    return ax


def shade(ax, v, var):
    f, lon = grid(v)
    m = ax.contourf(lon, G.LAT_NH, f, levels=LEVELS[var], cmap=CMAP[var], extend="both", transform=ccrs.PlateCarree())
    m.set_edgecolor("face")
    return m


def skill_dots(ax, sig):
    """Dots over the points with significant hindcast skill (house.stipple); the forecast's colours show everywhere."""
    sg, lon = grid(sig.astype(float))
    house.stipple(ax, lon, G.LAT_NH, sg >= 0.5, transform=ccrs.PlateCarree())


def prob_layers(o: dict, var: str) -> list[tuple[np.ndarray, str]]:
    """The tercile-probability map as CPC draws it: where above (below) normal is the most likely category and
    reaches 40 %, its probability in that category's colours; elsewhere blank (near normal or equal chances). The
    probabilities before the skill test sets the points without skill to equal chances (p_*_raw): the dots say where
    the hindcast backs them."""
    pa, pb = 100 * o[f"{var}_p_above_raw"], 100 * o[f"{var}_p_below_raw"]
    pn = 100 - pa - pb
    up = np.where((pa >= 40) & (pa > pb) & (pa > pn), pa, np.nan)
    dn = np.where((pb >= 40) & (pb > pa) & (pb > pn), pb, np.nan)
    return [(up, CAT[var][0][1]), (dn, CAT[var][1][1])]


def draw_prob(ax, o, var):
    for f, cmap in prob_layers(o, var):
        g, lon = grid(f)
        cols = plt.get_cmap(cmap)(np.linspace(0.3, 0.95, len(PROB_LEVELS) - 1))
        m = ax.contourf(lon, G.LAT_NH, g, levels=PROB_LEVELS, colors=cols, transform=ccrs.PlateCarree())
        m.set_edgecolor("face")


def anom_bar(fig, rect, var, label):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=CMAP[var], norm=mcolors.BoundaryNorm(LEVELS[var], 256, extend="both"))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=LEVELS[var])
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_edgecolor(house.EDGE)
    cb.set_label(label, fontsize=8.5, color=house.INK2)


def prob_legend(fig, rect, var):
    """Two small colour bars: above normal and below normal."""
    for (lab, cmap), x0 in zip(CAT[var], (rect[0], rect[0] + rect[2] / 2 + 0.02), strict=True):
        cax = fig.add_axes([x0, rect[1], rect[2] / 2 - 0.02, rect[3]])
        cols = plt.get_cmap(cmap)(np.linspace(0.3, 0.95, len(PROB_LEVELS) - 1))
        cmp = mcolors.ListedColormap(cols)
        norm = mcolors.BoundaryNorm(PROB_LEVELS, cmp.N)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax, orientation="horizontal")
        cb.set_ticks(PROB_LEVELS[:-1])
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.outline.set_edgecolor(house.EDGE)
        cb.set_label(f"probability of {lab} normal (%)", fontsize=8.5, color=house.INK2)


def inputs_text(s: dict, var: str) -> str:
    """'statistical + C3S (6 centres, Nov issue) + GEFS 15 Nov + GEPS 12 Nov': what the issued forecast is made of."""
    parts = ["statistical"]
    c = s.get("c3s", {}).get(var, {})
    if c.get("issued"):
        mon = calendar.month_abbr[int(c["issue"][4:])]
        parts.append(f"C3S ({len(c['systems'])} centres, {mon} issue)")
    b = s.get("blend", {}).get(var, {})
    if b.get("issued"):
        for k, r in b["runs"].items():
            init = datetime.date.fromisoformat(r["init"])
            parts.append(f"{k} {init.day} {init:%b}")
    return " + ".join(parts)


def head(fig, s: dict, var: str, what: str, sub2: str, y0: float = 0.965, x0: float = 0.04) -> None:
    """Title, inputs line and skill line, left on the figure from (x0, y0) down."""
    fig.text(x0, y0, f"{s['target_label']} · {what}", fontsize=13.5, fontweight="semibold", color=INK)
    sub = (
        f"Claude's forecast: {inputs_text(s, var)} · issued {s['issued']}, ERA5 to {s['window'].split('–')[1].strip()}"
    )
    if s.get("update"):
        u = s["update"]
        d1 = datetime.date.fromisoformat(u["medium"]["days"][1])
        c = datetime.datetime.strptime(u["medium"]["cycle"], "%Y%m%d%H")
        sub = (f"Update {s['issued']} (of {u['of']}): days 1–{d1.day} IFS-ENS and AIFS-ENS ({c.day} {c:%b} {c:%H}Z), "
               f"the rest statistical with their vortex")  # fmt: skip
    fig.text(x0, y0 - 0.03, sub, fontsize=min(8.8, 8.8 * 112 / len(sub)), color=house.INK2)
    fig.text(x0, y0 - 0.055, sub2, fontsize=min(8.2, 8.2 * 118 / len(sub2)), color=MUTED)


def skill_line(s: dict, var: str, kind: str) -> str:
    iv = s["vars"][var]["issued"]
    y0, y1 = iv["years"]
    out = f"Hindcast {y0}–{y1}, cross-validated: ACC {iv['acc']:+.2f}, RPSS {iv['rpss']:+.2f} · dots: significant skill"
    if var == "z500" and kind == "anom":
        out += " · contours: height (dam)"
    if kind == "prob":
        out += " · white: no category above 40 %"
    return out


def outlook_map(o: dict, s: dict, var: str, region: str, kind: str, path: Path) -> None:
    fc = o[f"{var}_fc"]
    nh = region == "nh"
    fig = plt.figure(figsize=(7.6, 8.25) if nh else (8.6, 7.6))
    ax = map_ax(fig, [0.03, 0.12, 0.94, 0.76], region)
    if kind == "anom":
        shade(ax, fc, var)
    else:
        draw_prob(ax, o, var)
    skill_dots(ax, o[f"{var}_sig"])
    if var == "z500" and kind == "anom":
        full, lon2 = grid(o["z500_clim"] + fc)
        cs = ax.contour(lon2, G.LAT_NH, full / 10.0, levels=np.arange(480, 600, 6), colors=INK, linewidths=0.7,
                        transform=ccrs.PlateCarree(), zorder=5)  # fmt: skip
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6.5, fmt="%d")
    ax.apply_aspect()  # the map's drawn box: the titles and the colour bar hug it, and the canvas is cropped to them
    b = ax.get_position()
    if kind == "anom":
        anom_bar(fig, [b.x0 + 0.1 * b.width, b.y0 - 0.055, 0.8 * b.width, 0.018], var, f"{TITLE[var]} ({UNIT[var]})")
    else:
        prob_legend(fig, [b.x0 + 0.05 * b.width, b.y0 - 0.055, 0.9 * b.width, 0.018], var)
    what = TITLE[var] if kind == "anom" else f"{NAME[var]}, tercile probabilities"
    if not nh:
        what += f" · {REGIONS[region][0]}"
    head(fig, s, var, what, skill_line(s, var, kind), y0=b.y1 + 0.075, x0=b.x0)
    fig.text(b.x0 + b.width / 2, b.y0 - 0.112, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=105 if nh else 100, bbox_inches="tight", pad_inches=0.12,
                pil_kwargs={"quality": 86, "method": 6})  # fmt: skip
    plt.close(fig)


def panel_row(fig, panels: list, var: str, top: float = 0.82, bar: str = "") -> None:
    """Hemispheric maps side by side (title, field) with one colour bar under them."""
    n = len(panels)
    wd = 0.98 / n
    for j, (lab, fld, sig) in enumerate(panels):
        ax = polar_ax(fig, [0.01 + j * wd, 0.16, wd - 0.01, top - 0.16])
        shade(ax, fld, var)
        if sig is not None:
            skill_dots(ax, sig)
        ax.set_title(lab, fontsize=9.2, fontweight="semibold", pad=4, loc="center")
    anom_bar(fig, [0.25, 0.085, 0.5, 0.025], var, bar or f"{TITLE[var]} ({UNIT[var]})")


def ingredients_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """What the issued forecast is made of: the statistical forecast, the C3S mean, GEFS + GEPS (the month) and the issue,
    each with its weight and its cross-validated ACC on the stage's hindcast years."""
    c, b = s.get("c3s", {}).get(var, {}), s.get("blend", {}).get(var, {})
    stat = o.get(f"{var}_stat_fc", o[f"{var}_fc"])
    panels = [
        (f"Statistical\nACC {s['vars'][var]['nh_acc_mean']:+.2f} ({s['hindcast'][0]}–{s['hindcast'][1]})", stat, None)
    ]
    if f"{var}_c3s_fc" in o and c:
        sc = c["scores"]["c3s"]
        mark = f"weight {c['weight_nh']['c3s']:.2f}" if c["issued"] else "not used"
        panels.append((f"C3S, {len(c['systems'])} centres\nACC {sc['acc']:+.2f} ({c['years'][0]}–{c['years'][1]}) · {mark}",
                       o[f"{var}_c3s_fc"], None))  # fmt: skip
    if f"{var}_models_fc" in o and b:
        first = datetime.date(int(s["target"][:4]), int(s["target"][5:]), 1)
        mark = f"weight {b['weight_nh']['models']:.2f}" if b["issued"] else "not used"
        days = max(r["days"][1] for r in b["runs"].values())
        panels.append((f"{' + '.join(b['models'])}, 1–{days} {first:%b}\nACC {b['scores']['models']['acc']:+.2f} · {mark}",
                       o[f"{var}_models_fc"], None))  # fmt: skip
    iv = s["vars"][var]["issued"]
    panels.append((f"Issued: {' + '.join(iv['inputs'])}\nACC {iv['acc']:+.2f} ({iv['years'][0]}–{iv['years'][1]})",
                   o[f"{var}_fc"], o[f"{var}_sig"]))  # fmt: skip
    n = len(panels)
    fig = plt.figure(figsize=(3.9 * n + 0.4, 5.4))
    panel_row(fig, panels, var)
    fig.text(0.01, 0.95, f"{s['target_label']} · {NAME[var]}: what the forecast is made of", fontsize=12.5,
             fontweight="semibold")  # fmt: skip
    fig.text(0.5, 0.012, CREDIT + " · ACC: cross-validated hindcast, 20–90N", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=88, pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def components_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """The statistical forecast's weighted components (those it was issued with) and their sum (before any blend)."""
    contrib = o[f"{var}_contrib"]
    beta = o[f"{var}_beta"]
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w = (w * (G.LAT_NH[:, None] >= LAT0)).ravel()
    panels = [*list(enumerate(s["components"])), (None, "sum")]
    fig = plt.figure(figsize=(14.0, 7.4))
    for j, (i, c) in enumerate(panels):
        r, col = divmod(j, 4)
        ax = polar_ax(fig, [0.005 + col * 0.25, 0.53 - r * 0.42, 0.24, 0.36])
        v = contrib[i] if i is not None else o.get(f"{var}_stat_fc", o[f"{var}_fc"])
        shade(ax, v, var)
        lab = (
            f"{s['components'][c]}\nmean weight {float((w * beta[:, i]).sum() / w.sum()):.2f}"
            if i is not None
            else "Sum"
        )
        ax.set_title(lab, fontsize=9.2, fontweight="semibold", pad=3, loc="center")
    anom_bar(fig, [0.77, 0.27, 0.2, 0.025], var, f"{UNIT[var]}, component × weight")
    fig.text(0.01, 0.965, f"{s['target_label']} · {NAME[var]}: the statistical forecast's components", fontsize=12.5,
             fontweight="semibold")  # fmt: skip
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=85, pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def skill_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """The issued forecast's hindcast correlation at each point (significant inside the line) and the statistical
    forecast's anomaly correlation year by year against the trend alone."""
    sv = s["vars"][var]
    iv = sv["issued"]
    fig = plt.figure(figsize=(12.6, 5.5))
    ax = polar_ax(fig, [0.0, 0.14, 0.4, 0.72])
    f, lon = grid(o.get(f"{var}_issued_r", o[f"{var}_r"]))
    lv = np.array([-0.6, -0.45, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75])
    m = ax.contourf(lon, G.LAT_NH, f, levels=lv, cmap="PuOr_r", extend="both", transform=ccrs.PlateCarree())
    m.set_edgecolor("face")
    sg, lon = grid(o[f"{var}_sig"].astype(float))
    ax.contour(lon, G.LAT_NH, sg, levels=[0.5], colors=INK, linewidths=0.9, transform=ccrs.PlateCarree())
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, ticks=lv[::2])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("correlation with ERA5 · black line: significant (one-sided 5 %, FDR 10 %)", fontsize=8)
    ax.set_title(f"Issued forecast ({' + '.join(iv['inputs'])}), {iv['years'][0]}–{iv['years'][1]}", fontsize=9.5,
                 fontweight="semibold", pad=3, loc="center")  # fmt: skip
    bx = fig.add_axes([0.5, 0.2, 0.47, 0.63])
    acc = sv["nh_acc_by_year"]
    ys = np.array([int(k) for k in acc])
    vals = np.array(list(acc.values()))
    bx.bar(ys, vals, color=np.where(vals >= 0, house.WARM, house.COOL), width=0.8, label="statistical forecast")
    tr = sv["acc_trend_by_year"]
    bx.plot(ys, [tr[str(y)] for y in ys], color=INK, lw=0.9, marker="o", ms=2.2, label="trend alone")
    bx.legend(fontsize=8, loc="lower left")
    bx.axhline(0, color=INK, lw=0.6)
    bx.set_ylim(-1, 1)
    bx.set_ylabel("anomaly correlation, 20–90N")
    bx.set_title(f"Statistical forecast by year: mean {vals.mean():+.2f} (trend alone {sv['acc_trend_mean']:+.2f})",
                 fontsize=9.5)  # fmt: skip
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.text(0.01, 0.95, f"{s['target_label']} · {NAME[var]}: cross-validated hindcast skill", fontsize=12.5,
             fontweight="semibold")  # fmt: skip
    fig.text(0.5, 0.008, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def update_fig(o: dict, s: dict, var: str, path: Path) -> None:
    """An update's two parts and the month: the IFS-ENS and AIFS-ENS days 1-7, the statistical forecast of the days
    from the 8th with the predicted vortex, and their days-weighted sum (update.py)."""
    u = s["update"]
    d0, d1 = (datetime.date.fromisoformat(x) for x in u["medium"]["days"])
    c = datetime.datetime.strptime(u["medium"]["cycle"], "%Y%m%d%H")
    last = datetime.date(d1.year, d1.month, calendar.monthrange(d1.year, d1.month)[1])
    wt = u["weights"]
    panels = [
        (f"IFS-ENS and AIFS-ENS, {d0.day}–{d1.day} {d1:%b}\n{c.day} {c:%b} {c:%H}Z runs · weight {wt['first_week']:.2f}", o[f"{var}_mr_fc"], None),
        (f"Statistical, {d1.day + 1}–{last.day} {last:%b}, with the vortex\nof {d0.day}–{d1.day} {d1:%b} · weight {wt['rest']:.2f}", o[f"{var}_stat_fc"], None),
        ("The month\n(the issued forecast)", o[f"{var}_fc"], o[f"{var}_sig"]),
    ]  # fmt: skip
    fig = plt.figure(figsize=(12.4, 5.4))
    panel_row(fig, panels, var, bar=f"{TITLE[var]} ({UNIT[var]}), each part's own anomaly")
    fig.text(0.01, 0.95, f"{s['target_label']} · {NAME[var]}: the update's two parts", fontsize=12.5,
             fontweight="semibold")  # fmt: skip
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def load(name: str) -> tuple[dict, dict]:
    s = json.loads((DATA / name).read_text())
    with np.load(DATA / name.replace(".json", ".npz")) as f:
        return s, {k: f[k] for k in f.files}


def draw_target(key: str, name: str) -> None:
    s, o = load(name)
    for var in s["vars"]:
        for region in REGIONS:
            for kind in ("anom", "prob"):
                outlook_map(o, s, var, region, kind, ASSETS / f"fc_{key}_{var}_{region}_{kind}.webp")
        if key == "m1":
            for kind in ("anom", "prob"):
                shutil.copyfile(ASSETS / f"fc_m1_{var}_nh_{kind}.webp", ASSETS / f"{name[:-5]}_{var}_{kind}.webp")
            if s.get("update"):
                update_fig(o, s, var, ASSETS / f"update_{var}.webp")
        ingredients_fig(o, s, var, ASSETS / f"ingredients_{key}_{var}.webp")
        components_fig(o, s, var, ASSETS / f"components_{key}_{var}.webp")
        skill_fig(o, s, var, ASSETS / f"skill_{key}_{var}.webp")
    print(f"drew {key} ({name})", flush=True)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    latest = json.loads((DATA / "latest.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw_target("m1", latest["file"])
    for k, name in enumerate(latest.get("seasons", []), 1):
        draw_target(f"s{k}", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
