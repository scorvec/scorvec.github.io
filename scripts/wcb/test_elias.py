#!/usr/bin/env python3
"""Offline checks of elias.py: the network's convolution and upsampling against their definitions written out cell
by cell, the finite differences and the irrotational wind against fields with known answers, and the standardising
and wrapping of a network's image. (check_elias.py holds the whole module to the published pipeline.)

    python scripts/wcb/test_elias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elias as E


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_conv():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, 6, 3)).astype(np.float32)
    w = rng.normal(size=(3, 3, 3, 5)).astype(np.float32)
    b = rng.normal(size=5).astype(np.float32)
    ref = np.zeros((2, 4, 6, 5))
    for i in range(4):
        for j in range(6):
            for di in range(3):
                for dj in range(3):
                    if 0 <= i + di - 1 < 4 and 0 <= j + dj - 1 < 6:
                        ref[:, i, j] += x[:, i + di - 1, j + dj - 1] @ w[di, dj]
    ref = np.maximum(ref + b, 0.0)
    check(np.abs(E._conv(x, w, b) - ref).max() < 1e-5, "3x3 convolution")


def test_up():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    w = rng.normal(size=(3, 3, 5, 2)).astype(np.float32)
    b = rng.normal(size=2).astype(np.float32)
    ref = np.zeros((2, 6, 8, 2))
    for i in range(3):
        for j in range(4):
            for di in range(3):
                for dj in range(3):
                    if 2 * i + di < 6 and 2 * j + dj < 8:
                        ref[:, 2 * i + di, 2 * j + dj] += x[:, i, j] @ w[di, dj]
    check(np.abs(E._up(x, w, b) - (ref + b)).max() < 1e-5, "transposed convolution")


def test_differences():
    """Solid-body rotation (u = U cos lat) has the vorticity 2 U sin lat / a and no divergence; a zonal wavenumber 2
    in v has the vorticity dv/dx away from the two edge columns."""
    lat, lon = np.deg2rad(E.ROWS)[:, None], np.deg2rad(E.LON)[None, :]
    u = 30.0 * np.cos(lat) * np.ones_like(lon)
    exact = 2.0 * 30.0 * np.sin(np.deg2rad(E.LAT))[:, None] / E.R_EARTH
    check(np.abs(E._vorticity(u, 0.0 * u) - exact).max() < 1e-3 * np.abs(exact).max(), "vorticity of a rotation")
    check(np.abs(E._divergence(u, 0.0 * u)).max() < 1e-12, "divergence of a rotation")
    v = 10.0 * np.sin(2.0 * lon) * np.ones_like(lat)
    exact = 20.0 * np.cos(2.0 * lon) / (E.R_EARTH * np.cos(np.deg2rad(E.LAT))[:, None])
    err = np.abs(E._vorticity(0.0 * v, v) - exact)[:, 1:-1] / np.abs(exact).max(axis=-1, keepdims=True)
    check(err.max() < 1e-3, f"vorticity of a wave {err.max():.1e}")


def test_irrotational_speed():
    """A velocity potential A sin lat + B cos lat cos lon: its Laplacian is -2 / a2 times itself, its gradient
    (-B sin lon, A cos lat - B sin lat cos lon) / a."""
    a, b = 3.0e7, 5.0e7
    east, north = np.deg2rad(E.LON)[None, :], np.deg2rad(E.LAT)[:, None]
    exact = np.hypot(b * np.sin(east), a * np.cos(north) - b * np.sin(north) * np.cos(east)) / E.R_EARTH
    for n, lmax in ((180, 89), (720, 179), (720, 359)):
        lat = np.deg2rad(90.0 - 180.0 / n * np.arange(n))[:, None]
        lon = np.deg2rad(180.0 / n * np.arange(2 * n))[None, :]
        div = -2.0 / E.R_EARTH**2 * (a * np.sin(lat) + b * np.cos(lat) * np.cos(lon))
        speed = E.irrotational_speed(div, lmax)
        check(speed.shape == exact.shape, f"shape {speed.shape}")
        check(np.abs(speed - exact).max() < 1e-6 * exact.max(), f"irrotational speed, {n} rows, degree {lmax}")


def test_image():
    rng = np.random.default_rng(2)
    fields = [rng.normal(3.0, 2.0, size=(4, E.LAT.size, E.LON.size)) for _ in range(4)]
    x = E._image([*fields, fields[0][0]])  # the fifth the same for every member
    check(x.shape == (4, 96, 448, 5) and x.dtype == np.float32, f"image {x.shape} {x.dtype}")
    core = x[:, :, E.PAD : -E.PAD]
    w = np.cos(np.deg2rad(E.LAT))[:, None, None] / (np.cos(np.deg2rad(E.LAT)).sum() * E.LON.size)
    mean = (core * w).sum(axis=(1, 2))
    var = (core**2 * w).sum(axis=(1, 2))
    check(np.abs(mean).max() < 1e-5 and np.abs(var - 1.0).max() < 1e-4, "standardised over the domain")
    check(np.array_equal(x[:, :, : E.PAD], core[:, :, -E.PAD :]), "wrapped west")
    check(np.array_equal(x[:, :, -E.PAD :], core[:, :, : E.PAD]), "wrapped east")
    check(np.array_equal(x[0, ..., 4], x[3, ..., 4]), "a shared predictor")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
