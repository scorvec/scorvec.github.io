#!/usr/bin/env python3
"""Offline checks of jet_branches.py (the grid reduction, the wind layer mean, the gradient, the season calendar, the
groups, the identification on a synthetic stepped tropopause) and of the event rules of build_jet_branches_ref.py.

    python scripts/mjo/src/test_jet_branches.py

Four 90-degree sectors of longitude, each with its own tropopause:
  0-90E     two steps: tropical -> subtropical at 30N, subtropical -> polar at 50N, 40 m/s
            -> a subtropical jet only near 30N, a polar jet only near 50N, no superposition
  90-180E   one step: tropical -> polar at 40N, 40 m/s -> both jets at 40N
  180-270E  the two steps of 0-90E with a 20 m/s wind -> nothing (the wind criterion)
  270-360E  the two steps spread over 12 degrees of latitude, 40 m/s -> nothing (the gradient criterion)
Two versions of it:
  isentropic   the layers' PV given directly (0.5 PVU below the step, 5.5 PVU above): the identification itself, where
               the one-step sector is a superposition (both layers step in the same columns)
  end to end   temperature and wind on the nine pressure levels from a tropopause (pressure p_T, potential temperature
               theta_T) that changes with latitude: theta falls 0.1 K/hPa downward through the troposphere and rises
               0.6 K/hPa upward through the stratosphere, so PV is about 1 PVU below the tropopause and 4-7 PVU above
               it. Tropopauses: tropical (100 hPa, 375 K), subtropical (180 hPa, 335 K), polar (320 hPa, 305 K). A
               single-valued tropopause slopes, so the one step crosses the polar layer's channel a degree poleward
               of the subtropical layer's (40.5N and 39.5N): both jets, two rows apart, a wall not yet vertical
               enough to be a superposition on this grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_jet_branches_ref as B
import jet_branches as JB
import pacjet_core as PC

KAPPA = 0.2857
SECTORS = ("two steps", "one step", "weak wind", "spread steps")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def step(lat, at, width):
    return 0.5 * (1.0 + np.tanh((lat - at) / width))


def sectors(sharp, broad=6.0, lon=JB.LON):
    """Per-longitude sector index, step width and wind speed on a ring of longitudes."""
    sector = (lon // 90).astype(int)
    return sector, np.where(sector == 3, broad, sharp), np.where(sector == 2, 20.0, 40.0)


def column(theta_t, p_t, p):
    """theta (K) at pressures p (hPa) below and above a tropopause at p_t with theta_t."""
    return np.where(p >= p_t, theta_t - 0.1 * (p - p_t), theta_t + 0.6 * (p_t - p))


def synthetic(sharp=1.0, lat=JB.BAND, lon=JB.LON):
    """Temperature and wind on the nine levels over a band (by default the 0.5 deg one)."""
    sector, w, speed = sectors(sharp, lon=lon)
    p = np.asarray(JB.LEVS, float)[:, None, None]
    la = lat[:, None]
    s30, s50, s40 = step(la, 30, w), step(la, 50, w), step(la, 40, w)
    theta_t = np.where(sector == 1, 375 - 70 * s40, 375 - 40 * s30 - 30 * s50)
    p_t = np.where(sector == 1, 100 + 220 * s40, 100 + 80 * s30 + 140 * s50)
    th = column(theta_t[None], p_t[None], p)
    t = th * (p / 1000.0) ** KAPPA
    u = np.broadcast_to(speed[None, None, :], t.shape).copy()
    return t, u, np.zeros_like(u), lat, lon


def synthetic_isentropic(sharp=1.0):
    """{layer: PV on its four isentropes} and the 400-100 hPa wind over the 0.5 deg band."""
    lat = JB.BAND
    sector, w, speed = sectors(sharp)
    at = {"polar": np.where(sector == 1, 40.0, 50.0), "stj": np.where(sector == 1, 40.0, 30.0)}
    iso = {}
    for name, thetas in JB.LAYERS.items():
        pv = 0.5 + 5.0 * step(lat[:, None], at[name][None, :], w[None, :])
        iso[name] = np.stack([pv * (1 + 0.02 * (k - 1.5)) for k in range(len(thetas))])  # the layer mean is pv
    return iso, np.broadcast_to(speed[None, :], (lat.size, JB.LON.size)), lat


def rows_with(mask, lat, lon0, lon1):
    """Latitudes of the 10-80N output rows where `mask` holds somewhere in lon0..lon1 (a sector's interior: the steps
    jump between sectors, which makes zonal gradients at their edges)."""
    sl = JB.output_rows(lat)
    cols = (JB.LON >= lon0) & (JB.LON <= lon1)  # noqa: SIM300  coordinate array LON, not a constant
    return lat[sl][mask[sl][:, cols].any(axis=1)]


def test_grid():
    rng = np.random.default_rng(1)
    lat025 = np.round(np.arange(90.0, -90.0 - 1e-6, -0.25), 3)  # north to south, as ERA5 and open data store it
    rows = JB.band_rows025(lat025)
    check(lat025[rows[0]] == 2.25 and lat025[rows[-1]] == 87.75 and np.all(np.diff(lat025[rows]) > 0), "band rows")
    f = rng.normal(size=(2, rows.size, 1440))
    h = JB.to_half(f)
    check(h.shape == (2, JB.BAND.size, JB.LON.size), f"to_half shape {h.shape}")
    # the operator is pacjet_core's: identical to it away from the longitude seam (0.5..359.5E from 0.25..359.75E)
    ref = PC._smooth_half(PC._smooth_half(f, -2)[..., 1:], -1)
    check(np.allclose(h[..., 1:], ref), "to_half differs from pacjet_core's operator")
    # periodic: the 0E column averages 359.75, 0 and 0.25E
    col = 0.25 * f[..., 359 * 4 + 3] + 0.5 * f[..., 0] + 0.25 * f[..., 1]
    want = 0.25 * col[..., 0:-2:2] + 0.5 * col[..., 1:-1:2] + 0.25 * col[..., 2::2]
    check(np.allclose(h[..., 0], want), "to_half is not periodic in longitude")
    # conservative: a field periodic in longitude keeps its zonal sum
    check(np.allclose(h.sum(-1) * 2, PC._smooth_half(f, -2).sum(-1)), "to_half does not conserve the zonal sum")
    check(np.allclose(JB.to_half(np.full((rows.size, 1440), 7.0)), 7.0), "a constant field changes")


def test_wind():
    shape = (len(JB.LEVS), 3, 4)
    u = np.full(shape, 30.0)
    v = np.full(shape, 40.0)
    check(np.allclose(JB.wind_layer_mean(u, v), 50.0), "uniform 50 m/s")
    p = np.asarray(JB.LEVS, float)[:, None, None]
    s = np.broadcast_to(100.0 - 0.1 * (p - 100.0), shape)  # linear in p: 70 m/s at 400 hPa, 100 at 100 hPa
    check(np.allclose(JB.wind_layer_mean(s, 0 * s), 85.0), "a speed linear in p averages to its middle value")
    s2 = s.copy()
    s2[JB.LEVS.index(700)] = 500.0  # levels outside 400-100 hPa do not count
    check(np.allclose(JB.wind_layer_mean(s2, 0 * s2), 85.0), "a level outside the layer leaks in")


def test_gradient():
    lat = JB.BAND
    f = np.broadcast_to(np.sin(np.deg2rad(lat))[:, None], (lat.size, 720))
    g = JB.grad_mag(f, lat)
    want = np.cos(np.deg2rad(lat))[:, None] / JB.A_EARTH
    check(np.allclose(g[1:-1], want[1:-1], rtol=1e-4), "|grad sin(lat)| = cos(lat)/a")
    lon = np.deg2rad(JB.LON)
    f = np.broadcast_to(np.cos(lon)[None, :], (lat.size, 720))
    g = JB.grad_mag(f, lat)
    want = np.abs(np.sin(lon))[None, :] / (JB.A_EARTH * np.cos(np.deg2rad(lat))[:, None])
    check(np.allclose(g, want, rtol=1e-4, atol=1e-12), "|grad cos(lon)| = |sin(lon)|/(a cos(lat))")
    pv = np.broadcast_to(np.linspace(0, 5, lat.size)[:, None], (lat.size, 720))
    gc = JB.channel_gradient(pv, lat)
    inside = (pv >= 1) & (pv <= 3)
    check(np.isfinite(gc[inside]).all() and np.isnan(gc[~inside]).all(), "the channel mask")


def test_season():
    days = np.array(
        ["2019-11-01", "2019-12-31", "2020-01-01", "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-31"],
        "datetime64[D]",
    )
    check(list(JB.season_slot(days)) == [0, 60, 61, 119, 120, 121, 151], f"leap season {JB.season_slot(days)}")
    days = np.array(
        ["2018-11-01", "2019-02-28", "2019-03-01", "2019-03-31", "2019-10-31", "2019-04-01"], "datetime64[D]"
    )
    check(list(JB.season_slot(days)) == [0, 119, 121, 151, -1, -1], f"common season {JB.season_slot(days)}")
    x = JB.slot_position()
    check(x[0] == 0 and x[119] == 119 and x[120] == 119.5 and x[121] == 120 and x[151] == 150, "positions")
    samples = np.full(152, 60.0)
    samples[120] = 16.0
    counts = 0.02 * samples[:, None] * np.ones((152, 5))
    check(np.allclose(JB.smooth_season(counts, samples), 0.02), "a constant frequency is kept")
    counts = np.zeros(152)
    counts[75] = 60.0  # every analysis of one day
    f = JB.smooth_season(counts, samples)
    check(np.isclose(f[75], 60.0 / (31 * 60.0)) and f[75 + 16] == 0 and f[75 - 15] > 0, "the 31-day window")


def test_groups():
    lat = np.array([40.0, 40.0, 40.0, 40.0, 40.0])
    lon = np.array([150.0, 160.0, 170.0, 230.0, 300.0])  # 10 deg at 40N is ~850 km; 60 deg ~4,900 km
    lab = JB.groups(lat, lon)
    check(lab[0] == lab[1] == lab[2] and len({lab[2], lab[3], lab[4]}) == 3, f"groups {lab}")
    check(abs(JB.great_circle_km(0, 0, 0, 90) - np.pi / 2 * JB.A_EARTH / 1000) < 1e-6, "great circle")
    check(JB.groups([40.0], [10.0]).tolist() == [0], "one point is one group")


def test_near():
    m = np.zeros((JB.LAT.size, JB.LON.size), bool)
    pts = [(45.0, 180.0), (75.0, 359.5), (12.0, 3.0)]  # mid-latitude, high latitude across the seam, low latitude
    for la, lo in pts:
        m[np.isclose(JB.LAT, la), np.isclose(JB.LON, lo)] = True
    got = JB.near(m)
    d = np.min([JB.great_circle_km(JB.LAT[:, None], JB.LON[None, :], la, lo) for la, lo in pts], axis=0)
    check(np.array_equal(got, d <= JB.NEAR_KM), f"near: {int((got != (d <= JB.NEAR_KM)).sum())} columns differ")
    check(np.array_equal(JB.near(np.stack([m, ~m]))[0], got), "near over a stack")
    check(not JB.near(np.zeros_like(m)).any(), "near of nothing")


def test_masks_025():
    # the whole path from 0.25 deg fields, as the forecasts and the ERA5 reference take it
    lat025 = np.round(np.arange(90.0, -90.0 - 1e-6, -0.25), 3)
    lat = lat025[JB.band_rows025(lat025)]
    t, u, v, _, _ = synthetic(lat=lat, lon=np.round(np.arange(0.0, 360.0, 0.25), 3))
    m = JB.masks_025(t, u, v)
    check(m.shape == (3, JB.LAT.size, JB.LON.size) and m.dtype == bool, f"masks_025 {m.shape} {m.dtype}")
    cols = (JB.LON >= 20) & (JB.LON <= 70)
    rows = {b: JB.LAT[m[i][:, cols].any(axis=1)] for i, b in enumerate(JB.BRANCHES)}
    check(rows["stj"].size and np.all(np.abs(rows["stj"] - 30) <= 1.5), f"0.25 deg: subtropical rows {rows['stj']}")
    check(rows["polar"].size and np.all(np.abs(rows["polar"] - 50) <= 1.5), f"0.25 deg: polar rows {rows['polar']}")
    weak = (JB.LON >= 200) & (JB.LON <= 250)
    check(not m[:, :, weak].any(), "0.25 deg: jets under a weak wind")


def test_identification_isentropic():
    iso, vbar, lat = synthetic_isentropic()
    g = JB.layer_gradients(iso, lat)
    m = JB.identify(vbar, g["polar"], g["stj"])
    r = {k: rows_with(m[k], lat, 20, 70) for k in JB.BRANCHES}
    check(r["stj"].size and np.all(np.abs(r["stj"] - 30) <= 1.5), f"two steps: subtropical jet rows {r['stj']}")
    check(r["polar"].size and np.all(np.abs(r["polar"] - 50) <= 1.5), f"two steps: polar jet rows {r['polar']}")
    check(r["sup"].size == 0, f"two steps: superposition rows {r['sup']}")
    r = {k: rows_with(m[k], lat, 110, 160) for k in JB.BRANCHES}
    check(r["sup"].size and np.all(np.abs(r["sup"] - 40) <= 1.5), f"one step: superposition rows {r['sup']}")
    check(np.array_equal(r["sup"], r["polar"]), "one step: a polar jet without the subtropical one")
    for k, (lo0, lo1) in enumerate(((200, 250), (290, 340)), start=2):
        r = {b: rows_with(m[b], lat, lo0, lo1) for b in JB.BRANCHES}
        check(all(x.size == 0 for x in r.values()), f"{SECTORS[k]}: identified {r}")
    # an undefined isentrope removes the column's layer PV
    iso["polar"][0, :, :10] = np.nan
    check(np.isnan(JB.layer_gradients(iso, lat)["polar"][:, 1:9]).all(), "a column without its layer")


def test_identification_end_to_end():
    t, u, v, lat, lon = synthetic()
    d = JB.diagnose(t, u, v, lat, lon)
    m = JB.identify(d["vbar"], d["grad"]["polar"], d["grad"]["stj"])
    r = {k: rows_with(m[k], lat, 20, 70) for k in JB.BRANCHES}
    check(r["stj"].size and np.all(np.abs(r["stj"] - 30) <= 1.5), f"two steps: subtropical jet rows {r['stj']}")
    check(r["polar"].size and np.all(np.abs(r["polar"] - 50) <= 1.5), f"two steps: polar jet rows {r['polar']}")
    check(r["sup"].size == 0, f"two steps: superposition rows {r['sup']}")
    r = {k: rows_with(m[k], lat, 110, 160) for k in JB.BRANCHES}
    check(r["polar"].size and r["stj"].size, f"one step: rows {r}")
    check(np.all(np.abs(np.concatenate([r["polar"], r["stj"]]) - 40) <= 1.0), f"one step: rows {r}")
    for k, (lo0, lo1) in enumerate(((200, 250), (290, 340)), start=2):
        r = {b: rows_with(m[b], lat, lo0, lo1) for b in JB.BRANCHES}
        check(all(x.size == 0 for x in r.values()), f"{SECTORS[k]}: identified {r}")
    # the step's gradient is well above the threshold, the background (f alone) far below it
    sl, cols = JB.output_rows(lat), (lon >= 20) & (lon <= 70)
    g = d["grad"]["polar"][sl][:, cols]
    check(np.nanmax(g) > 1.2 * JB.THRESH["polar"], f"the polar step's gradient {np.nanmax(g):.2e}")
    check(np.nanmin(g) < 0.1 * JB.THRESH["polar"], f"the background gradient {np.nanmin(g):.2e}")
    check(np.allclose(d["vbar"][:, cols], 40.0), "the wind layer mean of a uniform wind")


def test_events():
    counts = np.array([0] * 50 + list(range(1, 11)) * 2)  # 20 analyses with 1..10 columns
    st = B.region_stats(counts)
    check(st["threshold"] == 10 and st["share_any"] == round(100 * 20 / 70, 2), f"region stats {st}")
    m = np.zeros((JB.LAT.size, JB.LON.size), bool)
    m[JB.LAT == 35.0, (JB.LON >= 140) & (JB.LON <= 149.5)] = True  # 20 columns in a row over ~800 km
    m[JB.LAT == 35.0, (JB.LON >= 170) & (JB.LON <= 171)] = True  # 3 columns ~1,800 km further east
    g = JB.groups_of(m, JB.region_mask(JB.LAT, JB.LON, "wnp"))
    check([x[0] for x in g] == [20, 3] and abs(g[0][2] - 144.75) < 1e-6 and g[0][1] == 35.0, f"groups {g}")
    t = pd.date_range("2000-01-01", periods=8, freq="12h")
    found = [
        [(20, 35.0, 145.0)],
        [(25, 36.0, 150.0)],  # ~470 km on: the same event
        [],
        [(30, 37.0, 162.0)],  # 24 h after the last, ~1,100 km on: still the same event
        [(19, 37.0, 175.0)],  # below the threshold: no event
        [],
        [],
        [(22, 37.0, 175.0), (21, 37.0, 300.0)],  # 48 h after the last: two new events
    ]
    ev = JB.track(t, found, thr=20)
    check(len(ev) == 3, f"events {ev}")
    check(ev[0]["analyses"] == 3 and ev[0]["peak_columns"] == 30 and ev[0]["end"] == t[3], f"first event {ev[0]}")
    check(ev[1]["onset"] == ev[2]["onset"] == t[7] and ev[1]["lon"] == 175.0, f"the last two {ev[1:]}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
