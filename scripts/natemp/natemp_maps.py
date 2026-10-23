"""The North America 2 m temperature maps of natemp.py: Lambert conformal, the 0.1-deg grid's cells drawn as they are.

Each pixel of the map takes the value of the grid point whose 0.1-deg cell holds it (nearest neighbour in latitude and
longitude, worked out once per layout: Canvas), so every cell keeps its own colour and true shape and nothing is
smoothed or regridded. At MAP_W pixels across EXTENT a cell is 3.4 px tall and 2.2 px wide at 50N (1.7 px at 60N).
Colour = the field in 2 degF steps on a fixed scale with saturating ends, only where the frame's gate passes (the rest
takes the neutral NEUTRAL fill); contours every 10 degF through the coloured part (dashed below zero); Natural Earth 10 m coastlines,
lakes, borders and the US, Canadian and Mexican states and provinces on top. Colour bar ticks in degF and degC.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgba
from matplotlib.patches import Rectangle

PROJ = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=40.0, standard_parallels=(33.0, 45.0))
EXTENT = (-3850e3, 3400e3, -2850e3, 4500e3)  # projected metres: Alaska to Newfoundland, southern Mexico to 76N
DPI = 100
MAP_W, MAP_H = 2175, 2205  # map pixels; MAP_H / MAP_W is the extent's aspect, so no pixel is resampled
LEFT, BAR_COL = 12, 213  # left margin and the colour bar's column (px)
TOP, BOTTOM = 128, 70  # title and footer bands (px)
FIG_W, FIG_H = LEFT + MAP_W + BAR_COL, TOP + MAP_H + BOTTOM
NEUTRAL = "#d9d6d0"  # where the gate fails: members split / no significant change
INK, MUTED = "#1f1f1f", "#6f6b64"
LAKE_RANK = 5  # Natural Earth lakes of scalerank <= this (the larger ~140 over the box)
STATE_COUNTRIES = ("USA", "CAN", "MEX")

# (cold arm, warm arm) key colours from the centre outward; each arm is sampled at its number of bins
KINDS = {
    "anom": dict(
        edges=np.arange(-30, 31, 2),
        cold=["#e3edf7", "#b5d2eb", "#7fb0da", "#4d8cc6", "#2b67ab", "#1f4789", "#232f69", "#1b1a49"],
        warm=["#fde7da", "#f8c3a3", "#f0976b", "#e36a43", "#cc3f2c", "#a3202a", "#76102d", "#4f0a2a"],
        ext=("#110d33", "#36061f"),
        fticks=range(-30, 31, 6),
        cticks=range(-16, 17, 2),
        contours=(-30, -20, -10, 10, 20, 30),
    ),
    "change": dict(
        edges=np.arange(-20, 21, 2),
        cold=["#ece7f4", "#d0c5e5", "#ae9dd2", "#8a74bd", "#6a51a6", "#4c348e", "#352170", "#21134f"],
        warm=["#fdebce", "#fbd197", "#f6b05c", "#eb892b", "#d0661a", "#a6480f", "#783109", "#4d2006"],
        ext=("#150b37", "#341503"),
        fticks=range(-20, 21, 4),
        cticks=range(-10, 11, 2),
        contours=(-20, -10, 10, 20),
    ),
}


def arm(keys: list, n: int) -> list:
    """n colours along the key colours, from the palest (the bin next to zero) to the darkest."""
    cm = LinearSegmentedColormap.from_list("arm", keys)
    return [cm(x) for x in np.linspace(0.0, 1.0, n)]


@functools.cache
def palette(kind: str):
    """(colour table for the codes of np.digitize on the edges + NEUTRAL, ListedColormap, BoundaryNorm)."""
    K = KINDS[kind]
    n = (len(K["edges"]) - 1) // 2
    bins = arm(K["cold"], n)[::-1] + arm(K["warm"], n)
    under, over = to_rgba(K["ext"][0]), to_rgba(K["ext"][1])
    table = np.array([under, *bins, over, to_rgba(NEUTRAL)])
    cmap = ListedColormap(bins).with_extremes(under=under, over=over)
    return (table * 255).round().astype(np.uint8), cmap, BoundaryNorm(K["edges"], cmap.N)


class Canvas:
    """The layout's pixel -> grid-point map and the grid points' projected coordinates, for a grid of rows `lat`
    (north first) and columns `lon` at 0.1 deg."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        x0, x1, y0, y1 = EXTENT
        x = x0 + (np.arange(MAP_W) + 0.5) * (x1 - x0) / MAP_W
        y = y1 - (np.arange(MAP_H) + 0.5) * (y1 - y0) / MAP_H
        X, Y = np.meshgrid(x, y)
        ll = ccrs.PlateCarree().transform_points(PROJ, X, Y)
        j = np.rint((lat[0] - ll[..., 1]) * 10).astype(np.int64)
        i = np.rint((ll[..., 0] - lon[0]) * 10).astype(np.int64)
        ok = (j >= 0) & (j < lat.size) & (i >= 0) & (i < lon.size)
        if not ok.all():
            raise SystemExit(f"the grid misses {int((~ok).sum())} pixels of the map")
        self.pix = j * lon.size + i
        g = PROJ.transform_points(ccrs.PlateCarree(), *np.meshgrid(lon, lat))
        self.X, self.Y = g[..., 0], g[..., 1]
        # the grid points inside the map's rectangle
        self.inmap = (np.abs(self.X - (x0 + x1) / 2) <= (x1 - x0) / 2) & (
            np.abs(self.Y - (y0 + y1) / 2) <= (y1 - y0) / 2
        )


@functools.cache
def shapes() -> dict:
    """The Natural Earth 10 m lines over the box, read once (call it before forking render workers: concurrent first
    downloads in the workers corrupt the shapefiles)."""
    import cartopy.io.shapereader as shp
    from shapely.geometry import box

    near = box(-180.0, 5.0, -20.0, 85.0)

    def read(category, name, keep=lambda a: True):
        r = shp.Reader(shp.natural_earth("10m", category, name))
        return [g for rec, g in zip(r.records(), r.geometries()) if g.intersects(near) and keep(rec.attributes)]

    return {
        "coast": read("physical", "coastline"),
        "lakes": read("physical", "lakes", lambda a: a.get("scalerank", 99) <= LAKE_RANK),
        "borders": read("cultural", "admin_0_boundary_lines_land"),
        "states": read("cultural", "admin_1_states_provinces_lines", lambda a: a.get("ADM0_A3") in STATE_COUNTRIES),
    }


def features(ax) -> None:
    from cartopy.feature import ShapelyFeature

    pc = ccrs.PlateCarree()
    S = shapes()
    for key, lw, colour in (
        ("states", 0.65, "#3a3a3a"),
        ("lakes", 0.7, "#202020"),
        ("borders", 1.15, "#141414"),
        ("coast", 1.0, "#141414"),
    ):
        ax.add_feature(ShapelyFeature(S[key], pc), facecolor="none", edgecolor=colour, linewidth=lw, zorder=4)


def fmt(v: float) -> str:
    return "0" if v == 0 else f"{v:+.0f}".replace("-", "−")


def colour_bar(fig, kind: str) -> None:
    """Vertical bar right of the map: degF on its left, degC on its right."""
    K = KINDS[kind]
    _, cmap, norm = palette(kind)
    x = (LEFT + MAP_W + 84) / FIG_W
    h0, h1 = BOTTOM + 30, TOP + 58  # px kept free below and above the bar
    cax = fig.add_axes([x, h0 / FIG_H, 38 / FIG_W, (FIG_H - h0 - h1) / FIG_H])
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        extend="both",
        extendfrac=0.03,
        ticks=list(K["fticks"]),
    )
    cb.outline.set_linewidth(0.8)
    cax.yaxis.set_ticks_position("left")
    cax.set_yticklabels([fmt(v) for v in K["fticks"]])
    cax.tick_params(labelsize=19, length=7, width=1.0, colors=INK)
    sec = cax.secondary_yaxis("right", functions=(lambda f: f / 1.8, lambda c: c * 1.8))
    sec.set_ticks(list(K["cticks"]), labels=[fmt(v) for v in K["cticks"]])
    sec.tick_params(labelsize=17, length=6, width=1.0, colors=MUTED)
    top = 1 - (h1 - 14) / FIG_H
    fig.text(x - 6 / FIG_W, top, "°F", fontsize=19, ha="right", va="bottom", color=INK, fontweight="bold")
    fig.text(x + 44 / FIG_W, top, "°C", fontsize=17, ha="left", va="bottom", color=MUTED, fontweight="bold")


def text(fig, x_px: float, y_px: float, s: str, size: float, room_px: float, **kw):
    """fig.text at pixel (x, y), its font shrunk below `size` if it would run past room_px; returns its width (px)."""
    t = fig.text(x_px / FIG_W, y_px / FIG_H, s, fontsize=size, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width
    if w > room_px:
        t.set_fontsize(size * room_px / w)
    return min(w, room_px)


def draw(canvas: Canvas, kind: str, value, gate, title: str, sub: str, neutral: str, foot: str, out: Path):
    """One frame: `value` (degF on the box's grid) coloured where `gate`, NEUTRAL (legend text `neutral`) elsewhere;
    contours of the gated field every 10 degF, dashed below zero."""
    K = KINDS[kind]
    table, _, _ = palette(kind)
    code = np.where(gate & np.isfinite(value), np.digitize(np.nan_to_num(value), K["edges"]), len(table) - 1)
    img = table[code.ravel()[canvas.pix]]
    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI)
    ax = fig.add_axes([LEFT / FIG_W, BOTTOM / FIG_H, MAP_W / FIG_W, MAP_H / FIG_H], projection=PROJ)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.imshow(img, extent=EXTENT, origin="upper", interpolation="nearest", transform=PROJ, zorder=1)
    cs = ax.contour(
        canvas.X,
        canvas.Y,
        np.where(gate, value, np.nan),
        levels=K["contours"],
        colors=INK,
        linewidths=1.0,
        negative_linestyles="dashed",
        zorder=3,
    )
    cs.set(path_effects=[pe.withStroke(linewidth=2.6, foreground="white", alpha=0.55)])
    features(ax)
    ax.spines["geo"].set_linewidth(1.0)
    colour_bar(fig, kind)
    room = FIG_W - 2 * LEFT
    text(fig, LEFT, FIG_H - 16, title, 30, room, fontweight="bold", va="top", color=INK)
    text(fig, LEFT, FIG_H - 76, sub, 19, room, va="top", color="#333333")
    sw = 34  # the neutral swatch, then its legend, then the footer right-aligned in the room left
    box = Rectangle((LEFT / FIG_W, 18 / FIG_H), sw / FIG_W, sw / FIG_H, transform=fig.transFigure)
    box.set(facecolor=NEUTRAL, edgecolor="#8a8680", linewidth=0.8)
    fig.patches.append(box)
    used = LEFT + sw + 12 + text(fig, LEFT + sw + 12, 35, neutral, 17, room / 3, va="center", color=INK)
    text(fig, FIG_W - 14, 35, foot, 15.5, FIG_W - 14 - used - 40, va="center", ha="right", color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def warm() -> None:
    """Read the shapes and draw them once in this process, before render workers fork from it."""
    import io

    fig = plt.figure(figsize=(4, 4), dpi=20)
    ax = fig.add_axes([0, 0, 1, 1], projection=PROJ)
    ax.set_extent(EXTENT, crs=PROJ)
    features(ax)
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)
