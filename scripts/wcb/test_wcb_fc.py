#!/usr/bin/env python3
"""Offline checks of wcb_fc.py: the reduction of the open data's 0.25 degree rows to the networks' 1 degree grid
against fields with known box means, the leads a fetch serves, the risk map's test on synthetic footprints, and the
potential vorticity, its advection and the members' sign test against atmospheres with known answers.

    python scripts/wcb/test_wcb_fc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elias as E
import wcb_fc as W

LAT = 90.0 - 0.25 * np.arange(W.NORTH)  # the fetched rows
LON = 180.0 + 0.25 * np.arange(W.NI)  # the open data's columns begin at the date line


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_box_mean_grid():
    check(LAT[-1] == E.ROWS[0] - 0.5, f"the fetched rows end half a degree south of the domain: {LAT[-1]}")
    out = W.box_mean(np.full((W.NORTH, W.NI), 7.0, np.float32), LAT, LON)
    check(out.shape == (E.ROWS.size, E.LON.size) and out.dtype == np.float32, f"{out.shape} {out.dtype}")
    check(np.allclose(out, 7.0), "a constant stays")


def test_box_mean_latitude():
    out = W.box_mean(np.repeat(LAT[:, None], W.NI, axis=1), LAT, LON)
    check(np.allclose(out[:-1], E.ROWS[:-1, None], atol=1e-5), "a field linear in latitude: the centre's value")
    check(np.allclose(out[-1], 89.8, atol=1e-5), "at the pole the half box: (90 + 89.75 + 89.5 / 2) / 2.5")


def test_box_mean_longitude():
    wave = np.repeat(np.sin(np.deg2rad(LON))[None], W.NORTH, axis=0)
    damp = (1.0 + 2.0 * np.cos(np.deg2rad(0.25)) + np.cos(np.deg2rad(0.5))) / 4.0  # the box mean of a wavenumber one
    want = damp * np.sin(np.deg2rad(E.LON))
    out = W.box_mean(wave, LAT, LON)
    check(np.allclose(out, want[None], atol=1e-6), "wavenumber one, columns from 180E, comes out on E.LON")
    k = 4 * 37  # the same field handed over from another first column
    again = W.box_mean(np.roll(wave, -k, axis=1), LAT, np.roll(LON, -k))
    check(np.allclose(again, out, atol=1e-6), "whatever the first column")
    try:
        W.box_mean(wave[:-1], LAT[:-1], LON)
    except RuntimeError:
        pass
    else:
        raise AssertionError("another grid is refused")


def test_leads():
    check(W.lead_steps(48) == [0, 12, 24, 36, 48], "12-hourly steps")
    check(W.shown_steps(48) == [24], "the ascent a day either side")
    check(W.shown_steps(192) == list(range(24, 169, 12)), "days 1 to 7 from a fetch to 192 h")
    check(set(W.ASCENT_FIELDS) < set(W.FIELDS) and len(W.ASCENT_FIELDS) == 10, "the ascent's ten fields")
    check(W.FULL_FIELDS[: len(W.FIELDS)] == W.FIELDS, "the stages' fields come first on a shown step")
    check(len(W.PV_FIELDS) == 9 and set(W.PV_FIELDS) <= set(W.FULL_FIELDS), "the PV's nine fields")
    check(len(W.FULL_FIELDS) == len(set(W.FULL_FIELDS)) == 27, "each field fetched once")


def test_risk():
    n = 50
    foot = np.zeros((n, E.LAT.size, E.LON.size), bool)
    foot[:, 40:44, 100:110] = True  # every member
    foot[:20, 60:64, 200:210] = True  # two in five
    foot[:2, 10, 300] = True  # two members where a belt is common
    normal = np.full((E.LAT.size, E.LON.size), 0.01)
    normal[10, 300] = 0.2
    normal[80:, :] = 0.0  # columns the climatology never has: tested against the floor
    foot[0, 85, 5] = True  # one member there
    share, shown = W.risk(foot, normal)
    check(np.allclose(share[40:44, 100:110], 1.0) and np.allclose(share[60:64, 200:210], 0.4), "the members' share")
    check(shown[40:44, 100:110].all() and shown[60:64, 200:210].all(), "drawn where well above the usual frequency")
    check(not shown[10, 300], "two of fifty where one in five is usual: not drawn")
    check(not shown[85, 5], "one member where the climatology has none: not drawn")
    check(shown.sum() == 80, f"nothing drawn where no member has a belt: {int(shown.sum())}")


T0, P_AT = 220.0, 100.0 * W.PV_LEVS[1]
PHI, PHI_ROWS = np.deg2rad(E.LAT)[:, None], np.deg2rad(E.ROWS)[:, None]
CORIOLIS = 2.0 * E.OMEGA * np.sin(PHI)
THETA_AT = T0 * (1000.0 / W.PV_LEVS[1]) ** E.KAPPA
STABLE = E.G * E.KAPPA * THETA_AT / P_AT  # -g dtheta/dp of an isothermal atmosphere at the level


def atmosphere(n=1, u=None, t=None):
    """n identical members on E.ROWS x E.LON: wind u(level) (default at rest) and temperature t(level) (default T0)."""
    shape = (n, E.ROWS.size, E.LON.size)
    f = {k: {lev: np.zeros(shape) for lev in W.PV_LEVS} for k in ("u", "v")}
    f["t"] = {lev: np.full(shape, T0) for lev in W.PV_LEVS}
    for lev in W.PV_LEVS:
        if u is not None:
            f["u"][lev] += u(lev)
        if t is not None:
            f["t"][lev] += t(lev)
    return f


def test_potential_vorticity():
    rest = 1.0e6 * CORIOLIS * STABLE
    pv = W.potential_vorticity(atmosphere())[0]
    check(pv.shape == (E.LAT.size, E.LON.size), f"on LAT x LON: {pv.shape}")
    check(np.allclose(pv, rest, rtol=1e-10, atol=1e-12), "an isothermal atmosphere at rest: f times its stability")
    check(abs(pv[E.LAT == 45, 0][0] - 2.99) < 0.01, f"2.99 PVU at 45N: {pv[E.LAT == 45, 0][0]:.3f}")
    speed = 40.0  # solid-body rotation: relative vorticity 2 U sin(lat) / a
    pv = W.potential_vorticity(atmosphere(u=lambda lev: speed * np.cos(PHI_ROWS)))[0]
    want = rest * (1.0 + speed / (E.R_EARTH * E.OMEGA))
    check(np.allclose(pv, want, rtol=1e-4, atol=1e-9), "solid-body rotation adds its vorticity")
    shear, warm = 20.0, 0.5  # u = shear ln(p / level), nothing at the level; temperature rising `warm` K a degree
    pv = W.potential_vorticity(
        atmosphere(u=lambda lev: shear * np.log(lev / W.PV_LEVS[1]), t=lambda lev: warm * (E.ROWS[:, None] - 45.0))
    )[0]
    theta = (T0 + warm * (E.LAT[:, None] - 45.0)) * THETA_AT / T0
    want = 1.0e6 * E.G * (CORIOLIS * E.KAPPA * theta / P_AT - (shear / P_AT) * (warm * THETA_AT / T0) / W.METRE)
    check(np.allclose(pv, want, rtol=1e-9, atol=1e-9), "vertical shear across a temperature gradient")


def test_outflow_pv():
    n, wind = 7, 5.0  # every member: an isothermal atmosphere at rest crossed by a northward irrotational wind
    chi = np.zeros((n, 2, E.LAT.size, E.LON.size))
    chi[:, 1] = wind
    got = W.outflow_pv(atmosphere(n), chi)
    check(set(got) == set(W.PV_MAPS), f"the maps kept per lead: {sorted(got)}")
    want = -86400.0 * wind * 1.0e6 * 2.0 * E.OMEGA * np.cos(PHI) * STABLE / E.R_EARTH
    check(np.allclose(got["adv"][1:-1], want[1:-1], rtol=1e-4), "low PV carried poleward")
    check((got["neg"][1:-1] == n).all(), "negative in every member")
    north = np.broadcast_to(1.0e6 * CORIOLIS * STABLE >= W.TROPOPAUSE, got["strat"].shape)
    check(np.array_equal(got["strat"], n * north), "stratospheric air where the PV reaches the tropopause's")
    chi[:, 1] = 0.0
    chi[:, 0] = wind
    check(np.abs(W.outflow_pv(atmosphere(n), chi)["adv"]).max() < 1e-9, "no change along a PV contour")


def test_agreed():
    n = 50
    neg = np.full((E.LAT.size, E.LON.size), 25.0)
    neg[40:44, 100:110] = 50  # every member negative
    neg[60:64, 200:210] = 2  # all but two positive
    neg[70, 5] = 31  # a split a coin gives one time in ten
    neg[2, 300] = 50  # outside the cells tested
    where = np.zeros(neg.shape, bool)
    where[10:, :] = True
    sure = W.agreed(neg, n, where)
    check(sure[40:44, 100:110].all() and sure[60:64, 200:210].all(), "kept where the members share a sign")
    check(sure.sum() == 80, f"and nowhere else: {int(sure.sum())}")


def test_cyclic():
    f, lon = W.cyclic(np.repeat(E.LON[None], 2, axis=0))
    check(f.shape == (2, 361) and lon[0] == 0.0 and lon[-1] == 360.0, "0-360E with the first column repeated")
    check(np.array_equal(f[0] % 360.0, lon % 360.0), "the columns follow their longitudes")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
