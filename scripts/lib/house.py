"""The site's matplotlib house style: the page's own type and colours on every figure.

    import house
    house.apply()                      # once, before the first figure (after matplotlib.use("Agg"))
    fig, ax = plt.subplots(figsize=house.size(13, 9 / 16))
    house.source(fig, "NOAA OISST v2.1 · anomaly vs 1991–2020")
    house.stipple(ax, lon, lat, skilful, transform=ccrs.PlateCarree())   # dots where a forecast map has skill
    house.legend_room(ax)              # once a panel is drawn: its lines pass clear of its legend
    house.spaced(cb.ax.xaxis)          # a colour bar's level labels thinned where they would run together
    house.clear(fig)                   # last, before saving: the figure's own text stays off its panels

Type: Source Sans 3 from scripts/lib/fonts when the files are there (the site's text face, OFL), else DejaVu Sans;
nothing below 8 pt at the canvas size. Colours: the site tokens (assets/site.css); titles set left, semibold; legends
without a frame line. Geometry, grids, spines, dpi and the colour maps stay each product's own (loop frames must keep
their size from frame to frame, and the maps keep their detailed palettes).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib import transforms as mtransforms

FONTS = Path(__file__).resolve().parent / "fonts"
FAMILY = "Source Sans 3"

# the site tokens (assets/site.css html:root)
INK = "#1c1c1a"
INK2 = "#4a4744"
MUTED = "#6f6b64"
RULE = "#e4e2dc"
EDGE = "#c9c5bb"
ACCENT = "#274b7a"
WARM = "#b4541f"
COOL = "#3f6f8f"
GRID = "#e8e6e0"

RC = {
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "axes.labelcolor": INK2,
    "axes.edgecolor": EDGE,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": EDGE,
    "ytick.color": EDGE,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "none",
    "legend.fancybox": False,
    "figure.titlesize": 13.5,
    "figure.titleweight": "semibold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
}


def fonts() -> str:
    """Register the bundled Source Sans 3 files; the family name to use (DejaVu Sans when they are absent)."""
    files = sorted(FONTS.glob("SourceSans3-*.ttf")) + sorted(FONTS.glob("SourceSans3-*.otf"))
    for f in files:
        font_manager.fontManager.addfont(str(f))
    return FAMILY if files else "DejaVu Sans"


def apply() -> None:
    """Set the house style on matplotlib's rcParams (idempotent)."""
    mpl.rcParams.update(RC)
    mpl.rcParams["font.family"] = [
        fonts(),
        "DejaVu Sans",
    ]  # (named outright: a character the first lacks is drawn from the second)
    mpl.rcParams["mathtext.fontset"] = "dejavusans"


def size(width: float, ratio: float) -> tuple[float, float]:
    """A figure size in inches from its width and height-to-width ratio (the data's own aspect: no letterboxing)."""
    return (width, width * ratio)


def source(fig, text: str, y: float = 0.004) -> None:
    """The small source and method line along a figure's bottom edge."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)


def stipple(ax, lon, lat, mask, transform=None, zorder: float = 3.5, color: str = INK2) -> None:
    """Dots over the cells of a map where ``mask`` is true (lat x lon, 1-D coordinates), with no fill: the house mark
    for significant skill on a forecast map, which shows its colours everywhere. A global grid's seam closes when its
    longitudes span the globe (the first column is repeated at +360)."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return
    lon = np.asarray(lon, float)
    step = lon[1] - lon[0]
    if abs(lon[-1] - lon[0] + step - 360) < step / 2:  # a global grid not yet closed: close the seam
        lon, mask = np.r_[lon, lon[0] + 360.0], np.concatenate([mask, mask[:, :1]], axis=1)
    kw = {} if transform is None else {"transform": transform}
    cs = ax.contourf(
        lon, lat, mask.astype(float), levels=[0.5, 1.5], colors="none", hatches=[".."], zorder=zorder, **kw
    )
    for c in getattr(cs, "collections", [cs]):  # the dots' colour, whatever the rcParams, and no outline
        c.set_edgecolor(color)
        c.set_linewidth(0.0)


def _extents(ax, renderer) -> list:
    """The boxes of one panel's frame, titles, axis labels and ticks, legend and texts, in pixels."""
    parts = [ax.title, ax._left_title, ax._right_title, ax.get_legend(), *ax.texts]
    boxes = [a.get_window_extent(renderer) for a in parts if a is not None and a.get_visible()]
    boxes += [b for b in (ax.xaxis.get_tightbbox(renderer), ax.yaxis.get_tightbbox(renderer)) if b is not None]
    return [ax.bbox, *(b for b in boxes if b.width > 0 and b.height > 0)]


def clear(fig, dpi: float | None = None, gap: float = 5.0) -> None:
    """Keep the text set on the figure itself (title, subtitle, footnote, source line) off its panels. Where a line above
    the panels reaches a panel's title, or one below them reaches a tick label, an axis label or a colour bar's label,
    the figure grows by the shortfall plus ``gap`` points and every panel keeps its size; a figure that is already clear
    is left as it is, and so is one laid out by an engine. Call it last, before saving, with the resolution it is saved
    at (hinted text is not the same height in points at every resolution); not on loop frames, which must keep one size
    from frame to frame."""
    if fig.get_layout_engine() is not None:
        return
    own = fig.dpi
    fig.set_dpi(dpi or own)
    _clear(fig, gap)
    fig.set_dpi(own)


def _clear(fig, gap: float) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h, pad = fig.bbox.height, gap * fig.dpi / 72
    axes = [ax for ax in fig.axes if ax.get_visible()]
    body = [b for ax in axes for b in _extents(ax, r)] + [leg.get_window_extent(r) for leg in fig.legends]
    top, bottom = max(ax.bbox.y1 for ax in axes), min(ax.bbox.y0 for ax in axes)  # of the panels' frames
    texts = [(t, t.get_window_extent(r)) for t in fig.texts if t.get_visible() and t.get_text().strip()]
    head = [t for t, e in texts if e.y1 > top and e.y0 > bottom]
    foot = [t for t, e in texts if e.y0 < bottom and e.y1 < top]

    def short(group, below: bool) -> float:
        """How far the texts of ``group`` reach into the panels under (above) them, plus the gap; 0 when clear."""
        need = 0.0
        for t, e in texts:
            for b in body if t in group else ():
                if b.x0 < e.x1 and e.x0 < b.x1 and (b.y0 < e.y1 if below else e.y0 < b.y1):  # (not one beyond the text)
                    need = max(need, (b.y1 - e.y0 if below else e.y1 - b.y0) + pad)
        return need

    up, down = short(head, True), short(foot, False)
    if up + down < 0.5:
        return
    h2 = h + up + down

    for a in [*fig.texts, *fig.lines, *fig.patches, *fig.artists]:
        if a.get_transform().contains_branch(fig.transFigure):  # in figure fractions: the same pixels, dy higher
            dy = 0 if a in foot else down + up if a in head else down
            a.set_transform(mtransforms.Affine2D().scale(1, h / h2).translate(0, dy / h2) + a.get_transform())
    for leg in fig.legends:
        e = leg.get_window_extent(r)
        leg.borderaxespad = 0
        leg.set_loc("lower left")
        leg.set_bbox_to_anchor((e.x0 / fig.bbox.width, (e.y0 + down) / h2), transform=fig.transFigure)
    for ax in axes:  # (an inset placed by a locator follows its parent whatever is set here)
        p = ax.get_position(original=True)
        ax.set_position([p.x0, (p.y0 * h + down) / h2, p.width, p.height * h / h2])
    fig.set_figheight(h2 / fig.dpi)


def legend_room(ax, gap: float = 4.0) -> None:
    """Stretch a panel's linear y axis until the lines and bars under its legend pass clear of it, by the measured
    shortfall plus ``gap`` points; for a legend along the panel's top or bottom edge, and nothing where they are clear
    already. Call it once the panel's data, limits and legend are set."""
    leg = ax.get_legend()
    if leg is None or ax.get_yscale() != "linear":
        return
    fig = ax.figure
    fig.canvas.draw()
    box, frame, pad = leg.get_window_extent(fig.canvas.get_renderer()), ax.bbox, gap * fig.dpi / 72
    ys = []  # the heights, in pixels, that the data reaches across the legend's width
    for a in (*ax.lines, *ax.patches):
        if not a.get_visible() or not a.get_transform().contains_branch(ax.transData):  # (not a line across the panel)
            continue
        v = a.get_transform().transform_path(a.get_path()).vertices
        v = v[np.isfinite(v).all(1)]
        ys.append(v[(v[:, 0] >= box.x0 - pad) & (v[:, 0] <= box.x1 + pad), 1])
        x0, y0, x1, y1 = v[:-1, 0], v[:-1, 1], v[1:, 0], v[1:, 1]
        dx = np.where(x1 == x0, 1.0, x1 - x0)
        for edge in (box.x0 - pad, box.x1 + pad):  # where a segment crosses the legend's side
            t = (edge - x0) / dx
            ys.append((y0 + (y1 - y0) * t)[(x1 != x0) & (t > 0) & (t < 1)])
    ys = np.concatenate(ys) if ys else np.empty(0)
    ys = ys[(ys > frame.y0) & (ys < frame.y1)]
    if not ys.size:
        return
    lo, hi = ax.get_ylim()
    if box.y0 + box.y1 > frame.y0 + frame.y1:  # a legend along the top: room above the data
        have, need = (box.y0 - pad - frame.y0) / frame.height, (ys.max() - frame.y0) / frame.height
        if need > have > 0.25:
            ax.set_ylim(lo, lo + (hi - lo) * need / have)
    else:
        have, need = (frame.y1 - box.y1 - pad) / frame.height, (frame.y1 - ys.min()) / frame.height
        if need > have > 0.25:
            ax.set_ylim(hi - (hi - lo) * need / have, hi)


def spaced(axis, gap: float = 3.0) -> None:
    """Thin an axis's tick labels where neighbours run together: every second (third, ...) kept, counted in from both
    ends so that a scale symmetric about zero stays so. For fixed ticks, a colour bar's levels; call it once the
    figure's layout is set."""
    fig = axis.figure
    fig.canvas.draw()
    rend, horiz = fig.canvas.get_renderer(), axis.axis_name == "x"
    labs = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
    span = [(b.x0, b.x1) if horiz else (b.y0, b.y1) for b in (t.get_window_extent(rend) for t in labs)]
    order = sorted(range(len(labs)), key=lambda i: span[i][0])
    n, pad = len(order), gap * fig.dpi / 72
    for k in range(1, max(n, 2)):
        keep = [order[i] for i in range(n) if min(i, n - 1 - i) % k == 0]
        if all(span[b][0] - span[a][1] >= pad for a, b in itertools.pairwise(keep)):
            break
    for i, t in enumerate(labs):
        t.set_visible(i in keep)
