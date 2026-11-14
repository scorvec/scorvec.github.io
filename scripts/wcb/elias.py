"""Warm conveyor belts from gridded fields: the ELIAS 2.0 models of Quinting and Grams (2022).

SOURCE. Quinting, J. F., and C. M. Grams (2022), Geosci. Model Dev. 15, 715-730, doi:10.5194/gmd-15-715-2022: three
convolutional networks (U-Nets) trained on trajectories in ERA-Interim give the conditional probability that a
1 degree column holds the inflow (below 800 hPa), the ascent (800-400 hPa) or the outflow (above 400 hPa) of a warm
conveyor belt, each from five predictors; a column counts where the probability exceeds the calendar day's decision
threshold. The models, the thresholds and the trajectory climatology are the archive at Zenodo,
doi:10.5281/zenodo.5154980 (CC BY 4.0), kept unmodified as release wcb-ref-v1 (wcb-ref.yml). The predictors here
follow that archive's code (MIT licence: ELIAS_LICENSE.txt beside this file) down to its finite differences, so the
networks are handed what they were trained on; check_elias.py holds this module to the archive's own pipeline.

GRID. 1 degree. The networks' domain is LAT (6S-89N) by LON (180W-179E); the fields come on ROWS (7S-90N: one row
either side for the centred differences) as {name: {level in hPa: array (..., 98, 360)}}, with u and v (m/s),
q (kg/kg, never negative), t (K) and z (geopotential, m2/s2) on LEVELS. A vertical derivative is centred on the
levels either side in the archive's level set (NEIGH). Longitude differences are one-sided at 180W and 179E, as in
the archive.

STAGES. The ascent's fifth predictor is the trajectory climatology of the ascent around the day; the inflow's is the
ascent probability LAG_H (24 h) later, the outflow's the ascent probability 24 h earlier. The outflow's second is the
speed of the irrotational wind at 300 hPa, here from the divergence itself (irrotational_speed).

NETWORKS. Net reads a model's weights (Keras HDF5) and applies them as plain array arithmetic: nine blocks of a 3x3
convolution, batch normalisation (folded into the convolution) and ReLU, 2x2 maximum pooling down four times, four
transposed convolutions back up with the skip connections, a 1x1 convolution and a sigmoid.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pyshtools as pysh
from netCDF4 import Dataset
from scipy.special import expit

LAT = np.arange(-6.0, 90.0)
LON = np.arange(-180.0, 180.0)
ROWS = np.arange(-7.0, 91.0)
STAGES = {"inflow": "GT800", "ascent": "MIDTROP", "outflow": "LT400"}  # the archive's names
WIND_LEVS = (1000, 850, 700, 500, 300)
LEVELS = {"u": WIND_LEVS, "v": WIND_LEVS, "q": WIND_LEVS, "t": (700, 500, 300), "z": (850, 500, 200)}
NEIGH = {700: (850, 500), 500: (700, 300), 300: (500, 200)}  # d/dp at a level: the levels below and above it
LAG_H = 24
PAD = 44  # columns wrapped around the date line on either side of a network's image (448 = 360 + 2 * 44)
LMAX = 179  # the irrotational wind's truncation: what a 1 degree grid holds
BATCH = 10  # images through a network at a time (memory)
BN_EPS = 1.0e-3  # Keras' batch normalisation default

R_EARTH = 6371200.0  # the archive's constants
RD = 287.04
RV = 461.6
CP = 1004.5
G = 9.80665
OMEGA = 2.0 * np.pi / 86400.0
KAPPA = RD / CP
EPSIL = RD / RV

_K = R_EARTH * np.pi / 180.0  # metres per degree
_COS = np.cos(np.deg2rad(ROWS))[:, None]
_IN = slice(1, -1)  # ROWS -> LAT
_W = np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)


def _ddx(a):
    """d/dx on LAT of a field on ROWS."""
    return np.gradient(a[..., _IN, :], axis=-1) / (_K * _COS[_IN])


def _ddy(a):
    """d/dy on LAT of a field on ROWS."""
    return (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * _K)


def _ddp(f, lev):
    """d/dp (per Pa) at `lev` of a field given per level."""
    below, above = NEIGH[lev]
    return (f[below] - f[above]) / (100.0 * (below - above))


def _vorticity(u, v):
    return _ddx(v) - _ddy(u * _COS) / _COS[_IN]


def _divergence(u, v):
    """The archive's form: the meridional term is d(v cos lat)/dy, not divided by cos lat."""
    return _ddx(u) + _ddy(v * _COS)


def _rh(t, q, p):
    """Relative humidity (%) over water (Bolton 1980) at pressure p (hPa)."""
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 273.15 + 243.5))
    return 100.0 * (q / (1.0 - q)) / (EPSIL * es / p)


def _theta_e(t, q, p):
    """Equivalent potential temperature (K; Bolton 1980) at pressure p (hPa)."""
    r = q / (1.0 - q)
    dry = t * (1000.0 / p) ** (KAPPA * (1.0 - 0.28 * r))
    with np.errstate(divide="ignore"):  # q = 0: the exponent below is zero whatever the condensation temperature
        tl = 1.0 / (1.0 / (t - 55.0) - np.log(_rh(t, q, p) / 100.0) / 2840.0) + 55.0
    return dry * np.exp((3.376 / tl - 0.00254) * 1.0e3 * r * (1.0 + 0.81 * r))


def _thickness_advection(f, lev):
    """Advection of dz/dp at `lev` by the wind there."""
    zp = _ddp(f["z"], lev)
    return -f["u"][lev][..., _IN, :] * _ddx(zp) - f["v"][lev][..., _IN, :] * _ddy(zp)


def _moist_pv(f, lev=500):
    """Potential vorticity of the equivalent potential temperature on a pressure level, positive cyclonic."""
    u, v = f["u"], f["v"]
    the = {k: _theta_e(f["t"][k], f["q"][k], k) for k in (lev, *NEIGH[lev])}
    vort = 2.0 * OMEGA * np.sin(np.deg2rad(LAT))[:, None] + _ddx(v[lev]) - _ddy(u[lev])
    pv = -G * (
        -_ddp(v, lev)[..., _IN, :] * _ddx(the[lev])
        + _ddp(u, lev)[..., _IN, :] * _ddy(the[lev])
        + vort * _ddp(the, lev)[..., _IN, :]
    )
    return np.where(LAT[:, None] < 0, -pv, pv)


def _stability(f, lev=500):
    """Static stability -(R T / (p theta)) d theta/dp."""
    th = {k: f["t"][k] * (1000.0 / k) ** KAPPA for k in (lev, *NEIGH[lev])}
    return (-(RD * f["t"][lev]) / (100.0 * lev * th[lev]) * _ddp(th, lev))[..., _IN, :]


def _image(channels):
    """A network's input (..., 96, 448, 5) from its five predictors on LAT x LON: each standardised by its
    area-weighted mean and standard deviation over the domain, then wrapped around the date line. (The archive's
    constant multipliers drop out of the standardisation but for their sign, which the callers keep.)"""
    x = np.stack(np.broadcast_arrays(*channels), axis=-1)
    w = (_W / _W.sum())[..., None]
    x = x - (x * w).sum(axis=(-3, -2), keepdims=True)
    x = x / np.sqrt((x * x * w).sum(axis=(-3, -2), keepdims=True))
    return np.concatenate([x[..., -PAD:, :], x, x[..., :PAD, :]], axis=-2).astype(np.float32)


def ascent_image(f, clim):
    """The ascent network's input: 850 hPa relative vorticity, 700 hPa relative humidity, 300 hPa thickness
    advection, 500 hPa meridional moisture flux and the day's ascent climatology (climatology)."""
    return _image(
        [
            _vorticity(f["u"][850], f["v"][850]),
            _rh(f["t"][700], f["q"][700], 700)[..., _IN, :],
            _thickness_advection(f, 300),
            (f["v"][500] * f["q"][500])[..., _IN, :],
            clim,
        ]
    )


def inflow_image(f, ascent_later):
    """The inflow network's input: 700 hPa thickness advection, 850 hPa meridional moisture flux, 1000 hPa moisture
    flux convergence, 500 hPa moist potential vorticity and the ascent probability LAG_H later."""
    u, v, q = f["u"], f["v"], f["q"]
    return _image(
        [
            _thickness_advection(f, 700),
            (v[850] * q[850])[..., _IN, :],
            -_divergence(u[1000] * q[1000], v[1000] * q[1000]),
            _moist_pv(f),
            ascent_later,
        ]
    )


def outflow_image(f, speed, ascent_earlier):
    """The outflow network's input: 300 hPa relative humidity, the 300 hPa irrotational wind speed (on LAT x LON:
    irrotational_speed), 500 hPa static stability, 300 hPa relative vorticity and the ascent probability LAG_H
    earlier."""
    return _image(
        [
            _rh(f["t"][300], f["q"][300], 300)[..., _IN, :],
            speed,
            _stability(f),
            _vorticity(f["u"][300], f["v"][300]),
            ascent_earlier,
        ]
    )


def irrotational_speed(div, lmax=LMAX):
    """Speed (m/s) of the irrotational wind on LAT x LON from the divergence (1/s) on a Driscoll-Healy grid - rows
    from 90N without the south pole, columns from 0E, twice as many columns as rows - truncated at degree `lmax`
    (89, 179 or 359): the velocity potential's harmonics are -a2 / (l (l + 1)) times the divergence's, the wind is
    its gradient."""
    c = pysh.expand.SHExpandDH(np.asarray(div, float), sampling=2, lmax_calc=lmax)
    deg = np.arange(1.0, lmax + 1.0)
    c[:, 0] = 0.0
    c[:, 1:] *= (-(R_EARTH**2) / (deg * (deg + 1.0)))[None, :, None]
    speed = np.hypot(*pysh.expand.MakeGradientDH(c, lmax=lmax, sampling=2, radius=R_EARTH))
    per_degree = speed.shape[0] // 180
    return np.roll(speed[::per_degree, ::per_degree][(90 - LAT).astype(int)], 180, axis=-1)


def thresholds(root, stage, day):
    """The decision thresholds (LAT x LON) of `stage` on the date `day`, from the unzipped archive at `root`; the
    archive gives 29 February those of the 28th."""
    mmdd = "0228" if (day.month, day.day) == (2, 29) else f"{day:%m%d}"
    with Dataset(Path(root) / "thresholds" / f"ocv_{mmdd}") as nc:
        return np.ma.filled(nc.variables[STAGES[stage]][:, :].astype(float), np.nan)


def climatology(root, stage, day):
    """How often the trajectories put a belt's `stage` over each column around the date `day` (the archive's 30-day
    running mean, LAT x LON): its files are global from 90S and 180W, with 180E repeated."""
    with Dataset(Path(root) / "data" / f"hit_{day:%m%d}") as nc:
        return np.ma.filled(nc.variables[STAGES[stage]][0, 0, 84:180, :-1].astype(float), np.nan)


def _layers(f, kind):
    """The layers of one kind that hold weights, in the order Keras numbered them as it built the model."""
    found = []
    for name in f:
        m = re.fullmatch(rf"{kind}(?:_(\d+))?", name)
        if m and len(f[name]):
            found.append((int(m.group(1) or 0), name))
    return [f[name][name] for _, name in sorted(found)]


def _conv(x, w, b):
    """A 3x3 convolution (zero padded: Keras 'same'), bias and ReLU: x (n, h, w, c_in), w (3, 3, c_in, c_out)."""
    n, h, wd, _ = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros((n * h * wd, w.shape[-1]), np.float32)
    for i in range(3):
        for j in range(3):
            out += xp[:, i : i + h, j : j + wd].reshape(n * h * wd, -1) @ w[i, j]
    out += b
    return np.maximum(out, 0.0, out=out).reshape(n, h, wd, -1)


def _up(x, w, b):
    """A transposed 3x3 convolution of stride 2 (Keras 'same': twice the size): every input cell adds its kernel to
    the output from twice its index on, and the last row and column are dropped; w (3, 3, c_in, c_out)."""
    n, h, wd, c = x.shape
    flat = x.reshape(n * h * wd, c)
    out = np.zeros((n, 2 * h + 1, 2 * wd + 1, w.shape[-1]), np.float32)
    for i in range(3):
        for j in range(3):
            out[:, i : i + 2 * h : 2, j : j + 2 * wd : 2] += (flat @ w[i, j]).reshape(n, h, wd, -1)
    return out[:, : 2 * h, : 2 * wd] + b


class Net:
    """One stage's network, from the archive's models/cnn_<name>.h5."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            conv = [(g["kernel:0"][()], g["bias:0"][()]) for g in _layers(f, "conv2d")]
            norm = [
                [g[k + ":0"][()] for k in ("gamma", "beta", "moving_mean", "moving_variance")]
                for g in _layers(f, "batch_normalization")
            ]
            up = [(g["kernel:0"][()], g["bias:0"][()]) for g in _layers(f, "conv2d_transpose")]
        if (len(conv), len(norm), len(up)) != (10, 9, 4):
            raise ValueError(f"{path}: {len(conv)} convolutions, {len(norm)} normalisations, {len(up)} upsamplings")
        self.blocks = []
        for (w, b), (gamma, beta, mean, var) in zip(conv[:9], norm, strict=True):
            scale = gamma / np.sqrt(var + BN_EPS)
            self.blocks.append((w * scale, (b - mean) * scale + beta))
        # Keras stores a transposed convolution's kernel as (3, 3, c_out, c_in)
        self.ups = [(np.ascontiguousarray(w.transpose(0, 1, 3, 2)), b) for w, b in up]
        self.head = (conv[9][0][0, 0], conv[9][1])

    def __call__(self, x, trace=None):
        """Probabilities (n, 96, 360) from images (n, 96, 448, 5). `trace`, a list, collects the output of every
        block and every upsampling in the order they run."""
        return np.concatenate([self._forward(x[i : i + BATCH], trace) for i in range(0, len(x), BATCH)])

    def _forward(self, x, trace):
        def note(a):
            if trace is not None:
                trace.append(a)
            return a

        skips = []
        for k in range(4):
            x = note(_conv(x, *self.blocks[k]))
            skips.append(x)
            n, h, wd, c = x.shape
            x = x.reshape(n, h // 2, 2, wd // 2, 2, c).max(axis=(2, 4))
        x = note(_conv(x, *self.blocks[4]))
        for k in range(4):
            x = note(_up(x, *self.ups[k]))
            x = note(_conv(np.concatenate([x, skips.pop()], axis=-1), *self.blocks[5 + k]))
        n, h, wd, c = x.shape
        w, b = self.head
        return expit(x.reshape(n * h * wd, c) @ w + b).reshape(n, h, wd)[:, :, PAD:-PAD]
