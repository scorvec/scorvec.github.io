"""Regular latitude-longitude grids: first-order conservative remapping between them, and the 366-day calendar of
their daily normals. Shared by the 500 hPa verification (scripts/verify/z500_acc.py) and the days 1-15 teleconnection
forecasts (scripts/ensmaps/telecon15.py, ensmaps_telecon.py), so a field is put on a grid one way everywhere.

    from latlon import G15, remap
    z15 = remap(z, lat, lon, G15)    # any regular global grid, either way up, longitudes from anywhere
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd


class Grid:
    """A regular latitude-longitude grid of cell centres, north to south, longitudes from 0 east."""

    def __init__(self, step: float, south: float = -90.0):
        self.lat = np.round(90.0 - step * np.arange(round((90.0 - south) / step) + 1), 6)
        self.lon = np.round(step * np.arange(round(360.0 / step)), 6)
        self.shape = (self.lat.size, self.lon.size)


G15 = Grid(1.5)  # WMO's standard verification grid, WeatherBench 2's 240 x 121
G25N = Grid(2.5, south=20.0)  # the teleconnection patterns' NCEP/NCAR grid, 20-90N


def _edges(c: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Edges of the cells of regularly spaced centres ``c`` (ascending), clipped to [lo, hi] (the poles)."""
    d = c[1] - c[0]
    e = np.concatenate([c - d / 2, c[-1:] + d / 2])
    return np.clip(e, lo, hi) if lo is not None else e


def _overlap(src: np.ndarray, dst: np.ndarray, period: float | None = None) -> np.ndarray:
    """(destination cells, source cells): the length shared by each pair of cells, of ascending edge arrays; with a
    period (longitude) a source cell counts once more a period either side."""
    shifts = (-period, 0.0, period) if period else (0.0,)
    out = np.zeros((dst.size - 1, src.size - 1))
    for s in shifts:
        lo = np.maximum(dst[:-1, None], src[None, :-1] + s)
        hi = np.minimum(dst[1:, None], src[None, 1:] + s)
        out += np.clip(hi - lo, 0.0, None)
    return out


@functools.cache
def _weights(lat0: float, lat1: float, nlat: int, lon0: float, dlon: float, nlon: int, g: Grid) -> tuple:
    """The remapping onto ``g`` of a regular source grid, as (rows, columns) weight matrices whose rows sum to one,
    whether the source runs south to north and the order that sorts its longitudes from 0 east."""
    lat, up = np.linspace(lat0, lat1, nlat), lat0 < lat1
    lat = lat if up else lat[::-1]
    lon = np.mod(lon0 + dlon * np.arange(nlon), 360.0)
    order = np.argsort(lon, kind="stable")
    s_lat = np.sin(np.radians(_edges(lat, -90.0, 90.0)))
    d_lat = np.sin(np.radians(_edges(g.lat[::-1], -90.0, 90.0)))
    wy = _overlap(s_lat, d_lat)[::-1]  # rows north to south, as g
    wx = _overlap(_edges(lon[order]), _edges(g.lon), 360.0)
    return wy / wy.sum(1)[:, None], wx / wx.sum(1)[:, None], up, order


def remap(v: np.ndarray, lat: np.ndarray, lon: np.ndarray, g: Grid) -> np.ndarray:
    """A global field on its own regular grid (centres ``lat`` x ``lon``, either way up) as the area mean over each
    cell of ``g``: first-order conservative remapping, latitude weighted by the cells' areas (overlaps in sin(lat))."""
    wy, wx, up, order = _weights(
        float(lat[0]), float(lat[-1]), lat.size, float(lon[0]), float(lon[1] - lon[0]), lon.size, g
    )
    v = np.asarray(v, np.float64)
    v = (v if up else v[::-1])[:, order]  # south to north, as the weights
    return (wy @ v @ wx.T).astype(np.float32)


def slot366(t: pd.DatetimeIndex) -> np.ndarray:
    """Row of the 366-day calendar: 29 February = 59, 1 March = 60 in every year."""
    return np.asarray(t.dayofyear - 1 + ((t.month > 2) & ~t.is_leap_year))
