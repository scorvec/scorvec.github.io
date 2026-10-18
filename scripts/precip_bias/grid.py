"""The Americas 1° grid shared by the model-rain bias work.

Cells are 1° boxes with centres at half degrees: latitude 59.5°S..59.5°N (120 rows, ascending), longitude
169.5°W..30.5°W (140 columns). Poleward of 60° IMERG is unreliable (snow and ice surfaces, few passive-microwave
retrievals), so the domain stops there: Alaska and northern Canada are left out on purpose.

  Model grids (regular, every 1/k degree with points on whole degrees: ECMWF open data 0.25°, GEFS and GEPS 0.5°): each box
  is the trapezoid-weighted mean of the (k+1) x (k+1) points on its edges and inside (edge points half weight), so the box
  is centred exactly on its cell centre. canonical() first brings a decoded field to rows 90°N -> 90°S, columns from 180°W.
  IMERG (0.1° cells, centres 89.95°S.. / 179.95°W..): the plain mean of the 10 x 10 cells inside, a cell needing >= 50 % valid.

use_domain("global") (the tropical rainfall forecast) switches every function here to the global band: the same 120 rows,
360 columns 179.5°W..179.5°E; the box at 179.5°E takes its eastern edge points from the 180°W column (the seam wraps).
"""

from __future__ import annotations

import numpy as np

LAT = np.arange(-59.5, 60.0, 1.0)  # 120 rows, ascending
LON = np.arange(-169.5, -30.0, 1.0)  # 140 columns
NY, NX = len(LAT), len(LON)

DOMAIN = "americas"


def canonical(v: np.ndarray, lat_first: float, lon_first: float) -> np.ndarray:
    """A decoded global regular field (rows from either pole, columns from any longitude) -> rows 90°N -> 90°S, columns
    from 180°W. GEFS starts at 90°N 0°E, GEPS at 90°S 0°E, ECMWF open data at 90°N 180°."""
    v = np.asarray(v)
    if lat_first < 0:
        v = v[::-1]
    return np.roll(v, round(((lon_first + 180.0) % 360.0) * v.shape[1] / 360.0), axis=1)


def to_1deg(full: np.ndarray, res: float) -> np.ndarray:
    """(rows 90°N -> 90°S, columns from 180°W) global field every `res` degrees (0.25 or 0.5) -> (NY, NX) 1° box means,
    ascending latitude."""
    k = round(1 / res)  # points per degree
    w = np.r_[0.5, np.ones(k - 1), 0.5] / k  # trapezoid weights of the k + 1 points across a box
    full = np.asarray(full, dtype="float64")
    j0, j1 = round(30 / res), round(150 / res)  # rows of 60°N .. 60°S
    if DOMAIN == "global":  # 180°W .. 180°E: the 180°E column is the 180°W one again
        a = np.concatenate([full[j0 : j1 + 1], full[j0 : j1 + 1, :1]], axis=1)
    else:
        a = full[j0 : j1 + 1, round(10 / res) : round(150 / res) + 1]  # 170°W .. 30°W
    rows = sum(w[i] * a[i::k][:NY] for i in range(k + 1))
    box = sum(w[i] * rows[:, i::k][:, :NX] for i in range(k + 1))
    return box[::-1].astype("float32")  # north-first -> ascending


def ecmwf_to_1deg(v: np.ndarray) -> np.ndarray:
    """(721, 1440) ECMWF 0.25° open-data field (row j = 90 - 0.25 j, column i = -180 + 0.25 i) -> (NY, NX) 1° box means."""
    return to_1deg(np.asarray(v).reshape(721, 1440), 0.25)


# IMERG 0.1° cells: lat index k = (lat + 89.95) / 0.1, lon index m = (lon + 179.95) / 0.1
IM_LAT0, IM_LAT1 = 300, 1499  # 59.95°S .. 59.95°N
IM_LON0, IM_LON1 = 100, 1499  # 169.95°W .. 30.05°W


def imerg_to_1deg(p: np.ndarray) -> np.ndarray:
    """(lon 1400, lat 1200) IMERG 0.1° subset in mm/day -> (NY, NX) 1° box means (NaN where < 50 % valid)."""
    p = np.where(np.isfinite(p) & (p >= 0), p, np.nan).T  # (1200, 1400)
    blk = p.reshape(NY, 10, NX, 10)
    n = np.isfinite(blk).sum((1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(blk, axis=(1, 3))
    m[n < 50] = np.nan
    return m.astype("float32")


def use_domain(name: str) -> None:
    """Rebind the grid to "americas" (the default) or "global" (60°S-60°N, all longitudes). Call before anything else."""
    global DOMAIN, LON, NX, IM_LON0, IM_LON1  # noqa: PLW0603  module grid rebound once at start-up (--domain)
    if name == "americas":
        return
    if name != "global":
        raise ValueError(f"unknown domain {name!r}")
    DOMAIN = "global"
    LON = np.arange(-179.5, 180.0, 1.0)
    NX = len(LON)
    IM_LON0, IM_LON1 = 0, 3599  # 179.95°W .. 179.95°E


def area_weights() -> np.ndarray:
    return np.repeat(np.cos(np.deg2rad(LAT))[:, None], NX, axis=1)


def coarsen(a: np.ndarray, k: int) -> np.ndarray:
    """(..., NY, NX) on the CURRENT grid -> (..., NY/k, NX/k) area-weighted box means of k x k cells; a box needs at
    least half its cells valid. Call before set_coarse."""
    if k == 1:
        return a
    ny, nx = a.shape[-2:]
    if ny % k or nx % k:
        raise ValueError(f"grid {ny}x{nx} not divisible by {k}")
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, nx))
    b = a.reshape((*a.shape[:-2], ny // k, k, nx // k, k))
    ww = w.reshape(ny // k, k, nx // k, k)
    ok = np.isfinite(b)
    s = (np.where(ok, b, 0.0) * ww).sum(axis=(-3, -1))
    d = (ok * ww).sum(axis=(-3, -1))
    n = ok.sum(axis=(-3, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= (k * k + 1) // 2, s / d, np.nan).astype(a.dtype)


def refine(r2: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """A positive 2° box field (a rain ratio) -> the 1° grid, bilinear in log, longitude periodic, latitude clamped at the
    edges."""
    from scipy.interpolate import RegularGridInterpolator

    lr = np.log(np.where(np.isfinite(r2) & (r2 > 0), r2, 1.0))
    lonp = np.concatenate([[lon2[-1] - 360], lon2, [lon2[0] + 360]])
    lrp = np.concatenate([lr[:, -1:], lr, lr[:, :1]], axis=1)
    f = RegularGridInterpolator((lat2, lonp), lrp, bounds_error=False, fill_value=None)
    la = np.clip(LAT, lat2[0], lat2[-1])
    pts = np.stack(np.meshgrid(la, LON, indexing="ij"), -1).reshape(-1, 2)
    return np.exp(f(pts)).reshape(NY, NX).astype("float32")


def set_coarse(k: int) -> None:
    """Rebind LAT / LON / NY / NX to the k-degree box grid (after the data have been coarsened)."""
    global LAT, LON, NY, NX  # noqa: PLW0603  module grid rebound once after the data are coarsened
    if k == 1:
        return
    LAT = LAT.reshape(-1, k).mean(1)
    LON = LON.reshape(-1, k).mean(1)
    NY, NX = len(LAT), len(LON)
