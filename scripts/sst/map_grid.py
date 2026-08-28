"""Shared graticule helpers for the ENSO-site cartopy maps.

Two fixes applied site-wide (2026-07-26):
  - tick values are MULTIPLES of the step (10°N, 0°, 10°S …), never anchored to
    the extent edge (the old `range(la0, la1, dlat)` produced 13°N / 3°N / 7°S)
  - dashed reference lines where the equator / dateline / Greenwich meridian
    cross the map, slightly heavier than the graticule
"""
from __future__ import annotations

import numpy as np
import cartopy.crs as ccrs


def lat_ticks(la0: float, la1: float, step: float) -> list:
    """Multiples of `step` inside [la0, la1]."""
    t = np.arange(np.ceil(la0 / step) * step, la1 + 1e-6, step)
    return [float(round(x, 2)) for x in t]


def lon_ticks(lo0: float, lo1: float, step: float) -> list:
    """Multiples of `step` inside [lo0, lo1] (0–360 input), as −180..180 values
    for cartopy's gridliner."""
    t = np.arange(np.ceil(lo0 / step) * step, lo1 + 1e-6, step)
    return [float(round(((x + 180) % 360) - 180, 2)) for x in t]


def add_ref_lines(ax, extent, color="0.35", lw=0.7) -> None:
    """Dashed equator / dateline / Greenwich lines where they cross the map.
    `extent` = (lon0, lon1, lat0, lat1) with longitudes in 0–360."""
    lo0, lo1, la0, la1 = extent
    style = dict(color=color, lw=lw, ls=(0, (6, 4)), zorder=3.5,
                 transform=ccrs.PlateCarree())
    if la0 < 0 < la1:                                       # equator
        xs = np.linspace(lo0, lo1, 80)
        ax.plot(xs, np.zeros_like(xs), **style)
    for ref in (180.0, 360.0):                              # dateline, Greenwich
        if lo0 < ref < lo1:
            x = ((ref + 180) % 360) - 180
            ax.plot([x, x], [la0, la1], **style)
