"""Area means of a regular latitude-longitude field over coarser cells (conservative regridding by overlap).

Each fine point stands for its h-wide cell; a coarse cell of `width` centred on its point takes the mean of the fine
cells it overlaps, weighted by the overlap and by cos(latitude), the way WeatherBench 2 regrids ERA5 onto its 1.5-deg
cells. Longitudes may run around the globe (period 360: a cell at 0 E takes the fine cells on both sides of it).

    m = coarsen(f, lat, lon, 0.25, clat, clon, 1.5, period=360.0)   # f (..., lat, lon) -> (..., clat, clon)
"""

from __future__ import annotations

import numpy as np


def weights(cells: np.ndarray, pts: np.ndarray, h: float, width: float, period: float | None = None) -> np.ndarray:
    """(cells, pts): the overlap of each point's h-wide cell with each width-wide cell, normalised per cell; with a
    period, the distance between centres is taken the short way round."""
    d = pts[None] - cells[:, None]
    if period:
        d = (d + period / 2) % period - period / 2
    w = np.clip(np.minimum(width / 2, d + h / 2) - np.maximum(-width / 2, d - h / 2), 0.0, None)
    return w / w.sum(1, keepdims=True)


def coarsen(f, lat, lon, h: float, clat, clon, width: float, period: float | None = None) -> np.ndarray:
    """The area means over the cells (clat, clon) of f (..., lat, lon) on h-deg points; leading dimensions kept."""
    wy = weights(np.asarray(clat), np.asarray(lat), h, width) * np.cos(np.deg2rad(lat))[None]
    wx = weights(np.asarray(clon), np.asarray(lon), h, width, period)
    return (wy @ np.asarray(f) @ wx.T) / (wy.sum(1)[:, None] * wx.sum(1)[None])
