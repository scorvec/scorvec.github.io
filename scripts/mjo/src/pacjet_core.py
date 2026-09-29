#!/usr/bin/env python3
"""Shared pieces of the North Pacific jet product (pacjet.py live, build_pacjet_*.py reference builders).

Grids (all regular, latitude ascending, longitude 0-360):
  work  0.5 deg over 10-80N, 100E-60W (100-300E): the maps, the jet axis, the probabilities. Built from the 0.25 deg
        open-data fields by an area-conservative 0.25 -> 0.5 average ([1/4, 1/2, 1/4] per axis), the same operation for
        every model so no model is sharper than another on the page.
  eof   1.5 deg, the ERA5 WeatherBench-2 grid (10.5-79.5N, 100.5-298.5E): 3 x 3 means of the work grid, which is exactly
        conservative because the 1.5 deg boxes are unions of nine 0.5 deg boxes. The jet-phase EOFs, the drift and the
        skill live here. The EOF sector is its 100.5-240E part (Winters et al. 2019: 10-80N, 100E-120W).

The jet axis is the latitude of the maximum 250 hPa wind speed at each longitude between 15 and 70N, kept only where
that maximum reaches AXIS_MIN; the line is broken where it jumps more than AXIS_JUMP degrees between neighbouring
longitudes (a split flow hands over from one branch to the other), so no vertical connector is drawn.

Jet phases (Winters et al. 2019): a day's point (PC1, PC2) in the plane of the two leading EOFs. Inside the unit circle
it is "neutral"; outside, the phase is the axis nearest to the point: extension (+PC1), retraction (-PC1), poleward shift
(+PC2), equatorward shift (-PC2).
"""
from __future__ import annotations

import numpy as np

LEVEL = 250
WORK_LAT = np.round(np.arange(10.0, 80.0 + 1e-6, 0.5), 3)
WORK_LON = np.round(np.arange(100.0, 300.0 + 1e-6, 0.5), 3)
EOF_LAT = np.round(np.arange(10.5, 79.5 + 1e-6, 1.5), 3)
EOF_LON = np.round(np.arange(100.5, 298.5 + 1e-6, 1.5), 3)
SECTOR_LON = (100.5, 240.0)                  # EOF sector, 100E-120W
MAP_EXTENT = (100.0, 300.0, 12.0, 75.0)       # lon0, lon1, lat0, lat1 of the drawn maps
AXIS_LAT = (15.0, 70.0)
AXIS_MIN = 30.0                               # m/s: weaker maxima are not a jet core
AXIS_JUMP = 3.0                               # deg: larger steps between neighbouring longitudes break the line
CORE = 50.0                                   # m/s: "jet core" threshold of the probability maps
PHASES = ("extension", "poleward", "retraction", "equatorward")
PHASE_LABEL = {"extension": "Jet extension", "retraction": "Jet retraction", "poleward": "Poleward shift",
               "equatorward": "Equatorward shift", "neutral": "Neutral"}


def _smooth_half(a, axis):
    """0.25 -> 0.5 deg conservative along one axis: out[i] = 1/4 a[2i-1] + 1/2 a[2i] + 1/4 a[2i+1]."""
    a = np.moveaxis(a, axis, -1)
    out = 0.25 * a[..., 0:-2:2] + 0.5 * a[..., 1:-1:2] + 0.25 * a[..., 2::2]
    return np.moveaxis(out, -1, axis)


def native_slices(lat025, lon025):
    """Index arrays into a 0.25 deg global grid (any latitude order, longitudes 0..359.75) covering the work grid
    padded by one 0.25 deg point on each side, in ascending latitude."""
    lat025 = np.round(np.asarray(lat025, float), 3); lon025 = np.round(np.asarray(lon025, float) % 360, 3)
    want_lat = np.round(np.arange(WORK_LAT[0] - 0.25, WORK_LAT[-1] + 0.25 + 1e-6, 0.25), 3)
    want_lon = np.round(np.arange(WORK_LON[0] - 0.25, WORK_LON[-1] + 0.25 + 1e-6, 0.25), 3)
    li = np.array([int(np.flatnonzero(lat025 == v)[0]) for v in want_lat])
    lj = np.array([int(np.flatnonzero(lon025 == v)[0]) for v in want_lon])
    return li, lj


def to_work(f025, li, lj):
    """(..., nlat025, nlon025) native field -> (..., 141, 401) on the 0.5 deg work grid."""
    sub = np.take(np.take(f025, li, axis=-2), lj, axis=-1)
    return _smooth_half(_smooth_half(sub, -2), -1).astype("float32")


def to_eof(fw):
    """(..., 141, 401) work grid -> (..., 47, 133) on the 1.5 deg grid (3 x 3 block means centred on the 1.5 deg points)."""
    i0 = int(np.flatnonzero(WORK_LAT == EOF_LAT[0])[0]) - 1
    j0 = int(np.flatnonzero(WORK_LON == EOF_LON[0])[0]) - 1
    a = fw[..., i0:i0 + 3 * EOF_LAT.size, j0:j0 + 3 * EOF_LON.size]
    s = a.shape[:-2]
    a = a.reshape(*s, EOF_LAT.size, 3, EOF_LON.size, 3)
    return a.mean(axis=(-3, -1)).astype("float32")


def sector_mask():
    return (EOF_LON >= SECTOR_LON[0]) & (EOF_LON <= SECTOR_LON[1])


def jet_axis(speed, lat=WORK_LAT, lat_range=AXIS_LAT, vmin=AXIS_MIN):
    """speed (..., nlat, nlon) -> (..., nlon) latitude of the maximum within lat_range, NaN below vmin, with a
    three-point parabolic refinement of the maximum so the line is not stepped at the grid spacing."""
    sel = (lat >= lat_range[0]) & (lat <= lat_range[1])
    s = speed[..., sel, :]
    la = lat[sel]
    k = np.argmax(s, axis=-2)
    smax = np.take_along_axis(s, k[..., None, :], axis=-2)[..., 0, :]
    km = np.clip(k - 1, 0, la.size - 1); kp = np.clip(k + 1, 0, la.size - 1)
    y0 = np.take_along_axis(s, km[..., None, :], axis=-2)[..., 0, :]
    y2 = np.take_along_axis(s, kp[..., None, :], axis=-2)[..., 0, :]
    den = y0 - 2 * smax + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where((k > 0) & (k < la.size - 1) & (den < 0), 0.5 * (y0 - y2) / den, 0.0)
    dl = float(la[1] - la[0])
    ax = la[k] + np.clip(off, -0.5, 0.5) * dl
    return np.where(smax >= vmin, ax, np.nan).astype("float32")


def smooth_axis(ax, n=7):
    """Running median along longitude (n points, NaN-aware; NaN where fewer than half the window is defined), so the
    drawn axis is not stepped by grid-scale noise in where the maximum sits."""
    ax = np.asarray(ax, float)
    h = n // 2
    pad = np.pad(ax, [(0, 0)] * (ax.ndim - 1) + [(h, h)], constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, n, axis=-1)
    ok = np.isfinite(win).sum(-1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(win, axis=-1)
    return np.where(np.isfinite(ax) & (ok > h), med, np.nan)


def axis_segments(lon, ax, jump=AXIS_JUMP):
    """One axis (nlon,) -> list of (lon, lat) arrays, broken at NaNs and at jumps > `jump` degrees."""
    ok = np.isfinite(ax)
    segs, cur = [], []
    for j in range(lon.size):
        if not ok[j]:
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = []
            continue
        if cur and abs(ax[j] - cur[-1][1]) > jump:
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = []
        cur.append((lon[j], ax[j]))
    if len(cur) > 1:
        segs.append(np.array(cur))
    return segs


def harm_basis(doy, nh=3):
    w = 2 * np.pi * np.asarray(doy, dtype=float) / 365.25
    cols = [np.ones_like(w)]
    for h in range(1, nh + 1):
        cols += [np.cos(h * w), np.sin(h * w)]
    return np.stack(cols, axis=-1)


def harm_eval(coef, doy):
    """coef (nharm, ...) evaluated at doy (n,) -> (n, ...)."""
    nh = (coef.shape[0] - 1) // 2
    return np.tensordot(harm_basis(doy, nh), coef, axes=(-1, 0))


def project(anom_eof, proj):
    """anom (..., 47, 133) on the 1.5 deg grid, proj (2, 47, 133) (zero outside the sector) -> PCs (..., 2) in sigma."""
    return np.einsum("...ij,kij->...k", np.nan_to_num(anom_eof), proj)


def phase_of(pc1, pc2):
    """Vectorised Winters et al. (2019) phase: 0 extension, 1 poleward, 2 retraction, 3 equatorward, -1 neutral."""
    pc1 = np.asarray(pc1, float); pc2 = np.asarray(pc2, float)
    r = np.hypot(pc1, pc2)
    ang = np.degrees(np.arctan2(pc2, pc1)) % 360
    ph = (((ang + 45) % 360) // 90).astype(int)                 # 0: -45..45 (ext), 1: 45..135 (pol), 2 (ret), 3 (eqw)
    return np.where(np.isfinite(r) & (r >= 1.0), ph, -1)
