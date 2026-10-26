#!/usr/bin/env python3
"""Offline checks of nh_vortex.py's 10 hPa temperature reductions and its two sudden-warming shares, on synthetic
fields (no network, no figures).

    python scripts/strat/test_nh_vortex.py

  weights     polar_t's cap mean is the area mean: T = sin(latitude) on a fine grid against its exact 60-90N area mean,
              and a field equal to 1 everywhere but south of 60N, which must not leak in
  gradient    a pole warmer than 60N gives a gradient above zero, a colder one below, a flat field zero, whichever way
              the latitudes run
  grids       the per-message rows reduction (90-60N rows of ECMWF's 0.25-deg grid) and the DataArray one (a global
              0.5-deg grid, north to south, members and steps) agree with polar_t of the zonal mean
  shares      the wind criterion counts a member easterly at 60N on some day to day 15; the WMO definition only a member
              easterly with the gradient above zero on the same day
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nh_vortex as NV


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_weights():
    lat = np.round(np.arange(90.0, -90.01, -0.01), 6)
    tcap = NV.polar_t(np.sin(np.deg2rad(lat)), lat)[0]
    s0 = np.sin(np.deg2rad(NV.TCAP[0]))
    exact = (1.0 - s0**2) / 2.0 / (1.0 - s0)  # the area mean of sin(lat) over 60-90N
    check(abs(tcap - exact) < 1e-3, f"cap mean {tcap:.5f}, the area mean is {exact:.5f}")
    t = np.where(lat >= NV.TCAP[0], 1.0, 50.0)
    check(abs(NV.polar_t(t, lat)[0] - 1.0) < 1e-12, "the cap mean reaches south of 60N")


def test_gradient():
    lat = np.arange(40.0, 90.01, 1.0)
    for order in (slice(None), slice(None, None, -1)):
        warm_pole = 200.0 + (lat - 60.0)  # warmer toward the pole: reversed
        cold_pole = 200.0 - (lat - 60.0)  # the winter normal: colder toward the pole
        r, n, f = (NV.polar_t(t[order], lat[order]) for t in (warm_pole, cold_pole, np.full(lat.size, 210.0)))
        check(r[1] > 0 and n[1] < 0 and abs(f[1]) < 1e-12, f"gradient signs {r[1]:.2f} {n[1]:.2f} {f[1]:.2e}")
        check(26.0 < r[1] < 30.0, f"85-90N mean minus 60N on a 1 K/degree field: {r[1]:.2f}")
        check(abs(f[0] - 210.0) < 1e-9, "a flat field's cap mean")


def test_grids():
    rng = np.random.default_rng(1)
    lat = 90.0 - 0.25 * np.array(NV.T_ROWS)
    rows = 210.0 + 0.3 * (lat[:, None] - 60.0) + rng.normal(0.0, 2.0, (lat.size, 1440))
    want = NV.polar_t(rows.mean(-1), lat)
    check(np.allclose(NV._t10(rows, lat, None), want, atol=1e-3), "rows reduction against polar_t")
    check(lat[-1] == NV.PHI, "the rows end at 60N")
    glat = np.arange(90.0, -90.01, -0.5)
    f = 220.0 + 0.2 * glat[None, None, :, None] + rng.normal(0.0, 1.0, (3, 2, glat.size, 720))
    da = xr.DataArray(f, dims=("number", "step", "latitude", "longitude"), coords={"latitude": glat})
    q = NV.polar_t_da(da)
    check(q.dims == ("number", "step", "q"), f"dims {q.dims}")
    check(np.allclose(q.values, NV.polar_t(f.mean(-1), glat), atol=1e-9), "DataArray reduction against polar_t")


def test_shares():
    init = pd.Timestamp("2026-11-01 00:00")
    t = init + pd.to_timedelta(np.arange(17), unit="D")
    u = pd.DataFrame(20.0, index=t, columns=range(4))
    dt = pd.DataFrame(-10.0, index=t, columns=range(4))
    u.iloc[3, 0], dt.iloc[3, 0] = -2.0, 1.0  # easterly and reversed on day 3: both
    u.iloc[3, 1], dt.iloc[5, 1] = -2.0, 1.0  # easterly on day 3, reversed on day 5: the wind criterion only
    dt.iloc[4, 2] = 3.0  # reversed, never easterly: neither
    u.iloc[16, 3], dt.iloc[16, 3] = -2.0, 1.0  # both on day 16, after day 15: neither
    check(NV.ssw_chance(u, init) == 0.5, f"wind criterion {NV.ssw_chance(u, init)}")
    check(NV.wmo_chance(u, dt, init) == 0.25, f"WMO definition {NV.wmo_chance(u, dt, init)}")
    check(NV.wmo_chance(u, dt.iloc[:, :2], init) == 0.5, "a member without the gradient is left out")
    check(NV.wmo_chance(u, dt * np.nan, init) is None, "no member with the gradient")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
