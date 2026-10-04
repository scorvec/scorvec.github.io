#!/usr/bin/env python3
"""Linear equatorial β-plane shallow-water model (Gill 1980) — the idealised leg of the
Walker-basins product. Pure numpy so it runs in any job.

    εu − yv = −p_x,   εv + yu = −p_y,   εp + u_x + v_y = −Q         (nondimensional)

with x, y scaled by L = √(c/β) and time by T = 1/√(cβ); c = 30 m/s (first baroclinic mode with
a moist-reduced phase speed), ε = 0.1 (Gill's value, a 4.4-day damping). Heating Q is positive
where the SST anomaly is positive AND the climatological SST supports deep convection. The
response is stepped to steady state with periodic x and a sponge poleward of 24°.
"""
from __future__ import annotations

import numpy as np

C_WAVE, BETA, A_EARTH = 30.0, 2.28e-11, 6.371e6
L_GILL = np.sqrt(C_WAVE / BETA)
T_GILL = 1.0 / np.sqrt(C_WAVE * BETA)
EPS = 0.1
LON2 = np.arange(0.0, 360.0, 2.0)
LAT2 = np.arange(-30.0, 30.1, 2.0)
BAND = 5.0


def gill_response(Q: np.ndarray, lat=LAT2, lon=LON2, eps=EPS):
    """Steady (u, v, p) on the (lat, lon) grid for heating Q(lat, lon): the linear system

        εu − yv + p_x = 0,   εv + yu + p_y = 0,   εp + u_x + v_y = −Q

    solved DIRECTLY as one sparse matrix (centred differences, periodic in x, v = 0 at the
    north/south walls, damping raised in a sponge poleward of 24°). A time-stepped version was
    unstable — the explicit Coriolis term grows at high |y| — and a direct solve is exact anyway.
    A weak 4th-order smoothing term suppresses the 2Δx checkerboard the collocated grid admits."""
    import scipy.sparse.linalg as spla
    A, walls = gill_matrix(lat, lon, eps)
    ny, nx = Q.shape; N = ny * nx
    b = _gill_rhs(Q, walls)
    sol = spla.spsolve(A, b)
    u = sol[:N].reshape(ny, nx); v = sol[N:2 * N].reshape(ny, nx); p = sol[2 * N:].reshape(ny, nx)
    return u, v, p


def gill_matrix(lat=LAT2, lon=LON2, eps=EPS):
    """The sparse operator of gill_response (CSC) and the wall-row indices, built once so many heating
    fields can share one factorisation (gill_solver)."""
    import scipy.sparse as sp
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    y = np.deg2rad(lat) * A_EARTH / L_GILL
    x = np.deg2rad(lon) * A_EARTH / L_GILL
    dx = x[1] - x[0]; dy = y[1] - y[0]
    ny, nx = lat.size, lon.size; N = ny * nx
    damp = eps + 0.6 * np.clip((np.abs(lat) - 24.0) / 6.0, 0, 1)
    idx = np.arange(N).reshape(ny, nx)
    # x-derivative, periodic
    Dx = (sp.diags([np.ones(nx - 1), -np.ones(nx - 1)], [1, -1], shape=(nx, nx)).tolil())
    Dx[0, nx - 1] = -1; Dx[nx - 1, 0] = 1
    Dx = sp.kron(sp.identity(ny), Dx.tocsr() / (2 * dx), format="csr")
    # y-derivative, centred inside, one-sided at the walls
    Dy1 = sp.diags([np.ones(ny - 1), -np.ones(ny - 1)], [1, -1], shape=(ny, ny)).tolil() / (2 * dy)
    Dy1[0, 0] = -1 / dy; Dy1[0, 1] = 1 / dy; Dy1[ny - 1, ny - 1] = 1 / dy; Dy1[ny - 1, ny - 2] = -1 / dy
    Dy = sp.kron(Dy1.tocsr(), sp.identity(nx), format="csr")
    Yd = sp.diags(np.repeat(y, nx)); Ed = sp.diags(np.repeat(damp, nx))
    # 4th-order smoothing in x (checkerboard control), negligible at resolved scales
    Lx = sp.kron(sp.identity(ny), _periodic_laplacian(nx), format="csr")
    S = 0.02 * (Lx @ Lx)
    A = sp.bmat([[Ed + S, -Yd, Dx],
                 [Yd, Ed + S, Dy],
                 [Dx, Dy, Ed + S]], format="lil")
    # walls: v = 0 on the first and last latitude rows
    walls = [N + idx[j, i] for j in (0, ny - 1) for i in range(nx)]
    for r in walls:
        A.rows[r] = [r]; A.data[r] = [1.0]
    return A.tocsc(), walls


def _gill_rhs(Q: np.ndarray, walls) -> np.ndarray:
    N = Q.size
    b = np.concatenate([np.zeros(N), np.zeros(N), -np.nan_to_num(Q).ravel()])
    b[walls] = 0.0
    return b


def gill_solver(lat=LAT2, lon=LON2, eps=EPS):
    """A factorised gill_response: returns solve(Q) -> (u, v, p), identical to gill_response(Q, lat, lon, eps)
    but one LU factorisation serves every heating field (hundreds of weekly fields in a calibration)."""
    import scipy.sparse.linalg as spla
    A, walls = gill_matrix(lat, lon, eps)
    lu = spla.splu(A)
    ny, nx = len(lat), len(lon); N = ny * nx

    def solve(Q: np.ndarray):
        sol = lu.solve(_gill_rhs(Q, walls))
        return sol[:N].reshape(ny, nx), sol[N:2 * N].reshape(ny, nx), sol[2 * N:].reshape(ny, nx)
    return solve


# Latent heat released per unit rain: L_v × 1 kg m⁻² day⁻¹ = 2.5e6 J / 86400 s ≈ 28.9 W m⁻² (column mean).
LV_WM2_PER_MMDAY = 2.5e6 / 86400.0


def heating_from_precip(p_anom: np.ndarray, lat=LAT2, full=18.0, zero=22.0):
    """Column latent heating (W m⁻²) from a rainfall ANOMALY (mm/day) on the Gill grid: 1 mm/day ≈ 28.9 W m⁻²,
    confined to the deep tropics with a linear taper from `full` to `zero` degrees (poleward of ~20° rain is
    mostly frontal and the equatorial β-plane does not apply). No SST mask: rain is the heating itself, so a
    dry anomaly is a cooling anomaly. Projection on the first baroclinic mode and the conversion to Gill units
    are absorbed by the calibrated scale factor of the consumer (gill_rain.py)."""
    latw = np.clip((zero - np.abs(np.asarray(lat, float))) / (zero - full), 0, 1)[:, None]
    return LV_WM2_PER_MMDAY * np.nan_to_num(p_anom) * latw


def _periodic_laplacian(n: int):
    import scipy.sparse as sp
    L = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], shape=(n, n)).tolil()
    L[0, n - 1] = 1; L[n - 1, 0] = 1
    return L.tocsr()


def heating_from_ssta(ssta2: np.ndarray, sst_clim2: np.ndarray, lat=LAT2):
    """Heating ∝ SST anomaly where the TOTAL SST (climatology + anomaly) supports deep convection:
    tapered over 26–28 °C so there is no cliff, confined to about 20°S–20°N, lightly smoothed in
    longitude. Using the total rather than the climatological SST is what lets a warm cold tongue
    become convective — the nonlinearity that makes an El Niño response larger than a linear
    scaling of a weak one."""
    total = np.nan_to_num(sst_clim2, nan=0.0) + np.nan_to_num(ssta2)
    taper = np.clip((total - 26.0) / 2.0, 0, 1)
    latw = np.clip((22.0 - np.abs(lat)) / 4.0, 0, 1)[:, None]
    Q = np.nan_to_num(ssta2) * taper * latw
    return (Q + 0.5 * (np.roll(Q, 1, 1) + np.roll(Q, -1, 1))) / 2.0


def band_u(u: np.ndarray, lat=LAT2, half=BAND):
    m = np.abs(lat) <= half; w = np.cos(np.deg2rad(lat[m]))
    return (u[m] * w[:, None]).sum(0) / w.sum()


def basin_mask(lon=LON2, lo0=0.0, lo1=360.0):
    """Longitude mask that may wrap past 360 (e.g. the Atlantic 285–380 = 75°W–20°E)."""
    l = lon % 360
    if lo1 <= 360:
        return (l >= lo0) & (l < lo1)
    return (l >= lo0) | (l < lo1 - 360)
