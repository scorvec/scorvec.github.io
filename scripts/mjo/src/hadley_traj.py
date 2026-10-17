#!/usr/bin/env python3
"""Back-trajectories from the subtropical descent: how long ago the sinking air left the tropical outflow layer.

Each day at 00 UTC, parcels are released on a RES grid over 15-35 deg of each hemisphere at RELEASE_P and traced back
DAYS_BACK days through the 6-hourly winds (u, v, omega on hadley_branches.LEVELS, 1.5 deg grid), with a STEP_S
midpoint (second-order Runge-Kutta) step and the winds linear in time, longitude, latitude and pressure. A parcel's
transit time is how long before its release it was last in the tropical outflow layer: equatorward of OUTFLOW_LAT and
within hadley_branches.OUT_DEPTH below the dynamic tropopause (from that time's wind and temperatures). Per release, sector
(hadley_branches.SUBS, read over 15-35 deg) and release level, the area-weighted histogram of transit times in BIN_D
days, and the weight of the parcels that were not in the outflow layer at any time in the DAYS_BACK days.

From a set of histograms (a 7-day window of releases), two numbers: the share of the sinking air that left the
outflow layer within DAYS_BACK days, and the median transit time of that share - the speed of the cell's upper and
descending branches. The ERA5 normals come from the same code on ERA5's winds (build_branch_clim.py).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hadley_branches as HB

RELEASE_P = (500.0,)  # hPa (at 700 hPa most of the air is not traced to the outflow within DAYS_BACK, and the median
# of the rest sits against the horizon: a statistic the 30-day cap decides)
RES = 3.0  # deg, release grid
RELEASE_LAT = np.arange(15.0, 35.0, RES)  # 15 ... 33
RELEASE_LON = np.arange(0.0, 360.0, RES)
DAYS_BACK = 30
STEP_S = 3600.0
OUTFLOW_LAT = 20.0
BIN_D = 0.25
NBIN = int(DAYS_BACK / BIN_D)
P_ASC = np.array(sorted(HB.LEVELS), float)  # 50 ... 1000 hPa: the winds' level axis
DT_DATA = 6 * 3600.0
DEG = 180.0 / np.pi


@dataclass
class Winds:
    """6-hourly winds on the 1.5 deg grid: t0 (s, the first time), u, v (m/s), w (Pa/s) as (time, P_ASC, lat, lon)
    and the dynamic tropopause's pressure ptrop (hPa) as (time, lat, lon), which only the back-trajectories need."""

    t0: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    ptrop: np.ndarray | None = None


def release_points():
    """(lat, lon, p, area weight) of every parcel: both hemispheres, every level."""
    la, lo = np.meshgrid(np.r_[RELEASE_LAT, -RELEASE_LAT], RELEASE_LON, indexing="ij")
    la, lo = np.tile(la.ravel(), len(RELEASE_P)), np.tile(lo.ravel(), len(RELEASE_P))
    p = np.repeat(RELEASE_P, la.size // len(RELEASE_P))
    return la, lo, p, np.cos(np.deg2rad(la))


def _corners(W: Winds, t, lat, lon, p=None):
    """Flat indices and weights of the 8 (time, lat, lon) or 16 (time, p, lat, lon) neighbours."""
    nt, nl, ny, nx = W.u.shape
    tt = (t - W.t0) / DT_DATA
    it = np.clip(np.floor(tt).astype(int), 0, nt - 2)
    at = np.clip(tt - it, 0.0, 1.0)
    y = (90.0 - lat) / 1.5
    j = np.clip(np.floor(y).astype(int), 0, ny - 2)
    bj = np.clip(y - j, 0.0, 1.0)
    x = (lon % 360.0) / 1.5
    i = np.floor(x).astype(int) % nx
    bi = x - np.floor(x)
    i1 = (i + 1) % nx
    out = []
    if p is None:
        for dt, wt in ((0, 1 - at), (1, at)):
            for jj, wj in ((j, 1 - bj), (j + 1, bj)):
                for ii, wi in ((i, 1 - bi), (i1, bi)):
                    out.append((((it + dt) * ny + jj) * nx + ii, wt * wj * wi))
        return out
    k = np.clip(np.searchsorted(P_ASC, p) - 1, 0, nl - 2)
    bk = np.clip((p - P_ASC[k]) / (P_ASC[k + 1] - P_ASC[k]), 0.0, 1.0)
    for dt, wt in ((0, 1 - at), (1, at)):
        for kk, wk in ((k, 1 - bk), (k + 1, bk)):
            for jj, wj in ((j, 1 - bj), (j + 1, bj)):
                for ii, wi in ((i, 1 - bi), (i1, bi)):
                    out.append(((((it + dt) * nl + kk) * ny + jj) * nx + ii, wt * wk * wj * wi))
    return out


def _gather(flat: np.ndarray, corners) -> np.ndarray:
    return sum(flat[idx] * w for idx, w in corners)


def _velocity(W: Winds, flat, t, lat, lon, p):
    """(dlon/dt deg/s, dlat/dt deg/s, dp/dt hPa/s) at the parcels."""
    c = _corners(W, t, lat, lon, p)
    u, v, w = (_gather(f, c) for f in flat)
    cosp = np.maximum(np.cos(np.deg2rad(lat)), 0.02)
    return u / (HB.A * cosp) * DEG, v / HB.A * DEG, w / 100.0


def _advance(lat, lon, p, k, h):
    lat = lat + h * k[1]
    over = np.abs(lat) > 90.0  # across a pole: back down the other side, half a turn round
    lat = np.where(over, np.sign(lat) * 180.0 - lat, lat)
    lon = np.where(over, lon + 180.0, lon + h * k[0]) % 360.0
    return lat, lon, np.clip(p + h * k[2], P_ASC[0], P_ASC[-1])


def _rk2(W: Winds, flat, t, lat, lon, p, h):
    """One midpoint (second-order Runge-Kutta) step of h seconds from time t."""
    k1 = _velocity(W, flat, t, lat, lon, p)
    mid = _advance(lat, lon, p, k1, 0.5 * h)
    return _advance(lat, lon, p, _velocity(W, flat, t + 0.5 * h, *mid), h)


def paths(W: Winds, t_start: float, lat, lon, p, hours: float) -> np.ndarray:
    """Forward trajectories of parcels from (lat, lon, p) at t_start (s) for `hours`: (record, 3, parcel), the
    6-hourly (lat, lon, p), the first record the release point."""
    flat = (W.u.reshape(-1), W.v.reshape(-1), W.w.reshape(-1))
    lat, lon, p = (np.asarray(x, float) for x in (lat, lon, p))
    rec, t = [np.stack([lat, lon, p])], t_start
    for s in range(1, int(hours * 3600 / STEP_S) + 1):
        lat, lon, p = _rk2(W, flat, t, lat, lon, p, STEP_S)
        t += STEP_S
        if (s * STEP_S) % DT_DATA == 0:
            rec.append(np.stack([lat, lon, p]))
    return np.stack(rec)


def trace(W: Winds, t_release: float, keep_tracks: bool = False):
    """Back-trajectories of release_points() from t_release (s): the transit time (days; NaN if the parcel was not in
    the outflow layer within DAYS_BACK), the source latitude and longitude, and with keep_tracks the 6-hourly
    (lat, lon, p) of every parcel back to its source (NaN after it)."""
    lat, lon, p, _ = release_points()
    flat = (W.u.reshape(-1), W.v.reshape(-1), W.w.reshape(-1))
    tp = W.ptrop.reshape(-1)
    tau, src_lat, src_lon = (np.full(lat.size, np.nan) for _ in range(3))
    tracks = [] if keep_tracks else None
    h = -STEP_S
    n = int(DAYS_BACK * 86400 / STEP_S)
    for s in range(1, n + 1):
        lat, lon, p = _rk2(W, flat, t_release + (s - 1) * h, lat, lon, p, h)
        t = t_release + s * h
        new = np.isnan(tau) & (np.abs(lat) <= OUTFLOW_LAT)
        if new.any():
            ptrop = _gather(tp, _corners(W, t, lat, lon))
            new &= p <= ptrop + HB.OUT_DEPTH
            tau[new], src_lat[new], src_lon[new] = s * STEP_S / 86400.0, lat[new], lon[new]
        if keep_tracks and (s * STEP_S) % DT_DATA == 0:
            alive = np.isnan(tau) | new
            tracks.append(np.where(alive, np.stack([lat, lon, p]), np.nan))
    return tau, src_lat, src_lon, (np.stack(tracks) if keep_tracks else None)


def histograms(tau: np.ndarray) -> np.ndarray:
    """(sector, level, NBIN + 1) area-weighted counts: the transit-time bins, then the parcels with none."""
    lat, lon, p, wt = release_points()
    out = np.zeros((len(HB.SUBS), len(RELEASE_P), NBIN + 1))
    b = np.where(np.isnan(tau), NBIN, np.clip(np.ceil(np.nan_to_num(tau) / BIN_D).astype(int) - 1, 0, NBIN - 1))
    for i, (hemi, w, e) in enumerate(HB.SUBS.values()):
        m = ((lat > 0) == (hemi == "N")) & HB.lon_mask(w, e, lon)
        for j, lev in enumerate(RELEASE_P):
            mm = m & (p == lev)
            out[i, j] = np.bincount(b[mm], weights=wt[mm], minlength=NBIN + 1)
    return out


def stats(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(share from the outflow, median transit time in days of that share) from (..., NBIN + 1) histograms."""
    found = h[..., :NBIN].sum(-1)
    share = found / h.sum(-1)
    cum = np.cumsum(h[..., :NBIN], -1) / np.maximum(found, 1e-12)[..., None]
    edges = BIN_D * np.arange(1, NBIN + 1)
    med = np.full(found.shape, np.nan)
    for ix in np.ndindex(found.shape):
        if found[ix] > 0:
            med[ix] = np.interp(0.5, np.r_[0.0, cum[ix]], np.r_[0.0, edges])
    return share, med


_W: Winds | None = None  # the winds a forked worker reads (set before the pool is made)


def _one(t_release: float) -> np.ndarray:
    return histograms(trace(_W, t_release)[0])


def trace_many(W: Winds, releases, workers: int = 4) -> np.ndarray:
    """histograms() of several releases (s), in forked worker processes that share W."""
    global _W  # noqa: PLW0603  the workers inherit W by fork
    _W = W
    import multiprocessing as mp

    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return np.stack(list(ex.map(_one, releases)))
