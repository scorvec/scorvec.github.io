#!/usr/bin/env python3
"""Offline checks of wcb_fc.py: the reduction of the open data's 0.25 degree rows to the networks' 1 degree grid
against fields with known box means, the leads a fetch serves, and the risk map's test on synthetic footprints.

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


def test_cyclic():
    f, lon = W.cyclic(np.repeat(E.LON[None], 2, axis=0))
    check(f.shape == (2, 361) and lon[0] == 0.0 and lon[-1] == 360.0, "0-360E with the first column repeated")
    check(np.array_equal(f[0] % 360.0, lon % 360.0), "the columns follow their longitudes")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
