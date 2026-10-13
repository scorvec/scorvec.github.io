"""The Americas 1° grid shared by the model-rain bias work (2026-10-13, user: a rainfall bias field "vs IMERG or other
sources (Stage IV for the USA)" - "One for North and South America would be fine to start").

Cells are 1° boxes with centres at half degrees: latitude 59.5°S..59.5°N (120 rows, ascending), longitude
169.5°W..30.5°W (140 columns). Poleward of 60° IMERG is unreliable (snow and ice surfaces, few passive-microwave
retrievals), so the domain stops there: Alaska and northern Canada are left out on purpose.

  ECMWF open data (0.25° points, 90°N -> 90°S, from 180°W): each box is the trapezoid-weighted mean of the 5 x 5 points on
  its edges and inside (edge points half weight), so the box is centred exactly on its cell centre.
  IMERG (0.1° cells, centres 89.95°S.. / 179.95°W..): the plain mean of the 10 x 10 cells inside, a cell needing >= 50 % valid.
"""
from __future__ import annotations

import numpy as np

LAT = np.arange(-59.5, 60.0, 1.0)              # 120 rows, ascending
LON = np.arange(-169.5, -30.0, 1.0)            # 140 columns
NY, NX = len(LAT), len(LON)

# ECMWF 0.25° open-data grid: row j = 90 - 0.25 j (721), column i = -180 + 0.25 i (1440)
_EJ0, _EJ1 = int((90 - 60) / 0.25), int((90 + 60) / 0.25)            # rows 120..600 = 60°N..60°S
_EI0, _EI1 = int((-170 + 180) / 0.25), int((-30 + 180) / 0.25)        # columns 40..600 = 170°W..30°W
_W5 = np.array([0.5, 1.0, 1.0, 1.0, 0.5]) / 4.0


def ecmwf_to_1deg(v: np.ndarray) -> np.ndarray:
    """(721, 1440) ECMWF 0.25° field -> (NY, NX) 1° box means, ascending latitude."""
    a = np.asarray(v, dtype="float64").reshape(721, 1440)[_EJ0:_EJ1 + 1, _EI0:_EI1 + 1]          # (481, 561), 60°N first
    rows = sum(_W5[k] * a[k::4][:NY] for k in range(5))                                          # (120, 561)
    box = sum(_W5[k] * rows[:, k::4][:, :NX] for k in range(5))                                  # (120, 140)
    return box[::-1].astype("float32")                                                           # north-first -> ascending


# IMERG 0.1° cells: lat index k = (lat + 89.95) / 0.1, lon index m = (lon + 179.95) / 0.1
IM_LAT0, IM_LAT1 = 300, 1499                   # 59.95°S .. 59.95°N
IM_LON0, IM_LON1 = 100, 1499                   # 169.95°W .. 30.05°W


def imerg_to_1deg(p: np.ndarray) -> np.ndarray:
    """(lon 1400, lat 1200) IMERG 0.1° subset in mm/day -> (NY, NX) 1° box means (NaN where < 50 % valid)."""
    p = np.where(np.isfinite(p) & (p >= 0), p, np.nan).T                                         # (1200, 1400)
    blk = p.reshape(NY, 10, NX, 10)
    n = np.isfinite(blk).sum((1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(blk, axis=(1, 3))
    m[n < 50] = np.nan
    return m.astype("float32")


def area_weights() -> np.ndarray:
    return np.repeat(np.cos(np.deg2rad(LAT))[:, None], NX, axis=1)
