"""The site's matplotlib house style: the page's own type and colours on every figure.

    import house
    house.apply()                      # once, before the first figure (after matplotlib.use("Agg"))
    fig, ax = plt.subplots(figsize=house.size(13, 9 / 16))
    house.source(fig, "NOAA OISST v2.1 · anomaly vs 1991–2020")

Type: Source Sans 3 from scripts/lib/fonts when the files are there (the site's text face, OFL), else DejaVu Sans;
nothing below 8 pt at the canvas size. Colours: the site tokens (assets/site.css); titles set left, semibold; legends
without a frame line. Geometry, grids, spines, dpi and the colour maps stay each product's own (loop frames must keep
their size from frame to frame, and the maps keep their detailed palettes).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

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
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [fonts(), "DejaVu Sans"]
    mpl.rcParams["mathtext.fontset"] = "dejavusans"


def size(width: float, ratio: float) -> tuple[float, float]:
    """A figure size in inches from its width and height-to-width ratio (the data's own aspect: no letterboxing)."""
    return (width, width * ratio)


def source(fig, text: str, y: float = 0.004) -> None:
    """The small source and method line along a figure's bottom edge."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
