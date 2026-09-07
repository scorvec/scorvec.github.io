#!/usr/bin/env python3
"""One map style for the outlook pages (user 2026-09-07: "redesign the maps to be a similar style
for consistency"): the SEAS5 single-map look — plate carrée, most of the world, bold title top-left,
one-line subtitle, 50 m coastlines, light gridlines with labels, a horizontal colour bar below.
seas5_build._global_map draws exactly this; sfs_maps / sfs_daily import it.

    fig, ax, H, pc = open_map(kind="atm")            # global, cut at 40 °E (sst: 20 °E)
    fig, ax, H, pc = open_map(extent=[-180, 180, 20, 90], central=-140)
    ... draw with transform=pc ...
    features(ax, land_only=False); heading(fig, H, title, sub); colorbar(fig, H, mappable, label)
    save(fig, out)
"""
from __future__ import annotations
import numpy as np

CENTRAL = {"sst": -160.0, "atm": -140.0}
LAT = {"sst": (-70, 70), "atm": (-60, 85)}
WIDTH = 14.0


def open_map(kind: str = "atm", extent=None, central: float | None = None, width: float = WIDTH, top: float = 0.95, bot: float = 0.85):
    """→ (fig, ax, H, PlateCarree). `extent` = [W, E, S, N] in degrees (None → global by kind)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    c = CENTRAL.get(kind, CENTRAL["atm"]) if central is None else central
    proj = ccrs.PlateCarree(central_longitude=c)
    if extent is None:
        lat0, lat1 = LAT.get(kind, LAT["atm"]); lon_span = 360.0
    else:
        lat0, lat1 = extent[2], extent[3]; lon_span = extent[1] - extent[0]
    map_h = width * (lat1 - lat0) / lon_span; H = map_h + top + bot
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=proj)
    if extent is None or (extent[1] - extent[0]) >= 360:
        ax.set_extent([-180, 180, lat0, lat1], crs=proj)
    else:
        ax.set_extent([extent[0], extent[1], lat0, lat1], crs=pc)
    return fig, ax, H, pc


def features(ax, land_only: bool = False, states: bool = True, gridlines: bool = True):
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    if land_only:
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    if states:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#999", zorder=3)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
        gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}


def heading(fig, H: float, title: str, sub: str, title_size: float = 13.5, sub_size: float = 8.6, wrap: int | None = None):
    """Bold title top-left, subtitle under it; `wrap` = characters per subtitle line for narrow figures."""
    if wrap:
        import textwrap
        sub = "\n".join(textwrap.wrap(sub, wrap))
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=title_size, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, sub, fontsize=sub_size, color="#444", va="top", linespacing=1.3)


def colorbar(fig, H: float, mappable, label: str, levels=None, extend: str = "both"):
    cax = fig.add_axes([0.25, 0.42 / H, 0.50, 0.14 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend, spacing="uniform")
    cb.set_label(label, fontsize=8.5); cb.ax.tick_params(labelsize=7)
    if levels is not None and len(levels) > 16:
        cb.set_ticks([x for x in levels if abs(x - round(x)) < 1e-6])
    return cb


def symmetric_levels(vmax: float, n: int = 6, gap: float | None = None) -> list[float]:
    """n bins each side of zero with a white band ±gap (default vmax/n/2) in the middle."""
    step = vmax / n; gap = step / 2 if gap is None else gap
    pos = [gap] + [step * k for k in range(1, n + 1)]
    return [-x for x in pos[::-1]] + pos


def save(fig, out, dpi: int = 125, quality: int = 86):
    import matplotlib.pyplot as plt
    fig.savefig(out, dpi=dpi, pil_kwargs={"quality": quality, "method": 6}); plt.close(fig)
