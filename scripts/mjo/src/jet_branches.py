#!/usr/bin/env python3
"""Polar jet, subtropical jet and their vertical superposition, identified grid column by grid column.

The scheme of Winters & Martin (2014), Handlos & Martin (2016), Christenson et al. (2017) and Winters et al. (2020).
A column holds a polar (subtropical) jet when both hold:
  1. the 400-100 hPa wind speed, averaged over the layer with pressure weighting, exceeds V_MIN = 30 m/s;
  2. the horizontal PV gradient of the 315-330 K (340-355 K) layer, where the layer's PV lies in the 1-3 PVU channel,
     reaches 1.4e-5 (0.9e-5) PVU/m.
A polar and a subtropical jet in the same column at the same time is a jet superposition: one steep tropopause wall
from the polar to the tropical tropopause instead of two steps.

How the papers' wording is applied here (they name the quantities, not the arithmetic):
  layer PV        the mean of the PV on the layer's isentropes at 5 K spacing (315, 320, 325, 330 K and 340-355 K):
                  the papers interpolate to 5 K isentropes and compute PV per isentropic layer, and the group's
                  diagnostics show "PV in the 315-330 K layer". A column whose layer leaves the 700-100 hPa data
                  (an isentrope below 700 hPa or above 100 hPa) has no layer PV and cannot hold that jet.
  channel         1 <= layer PV <= 3 PVU at the column; the gradient is the centred-difference |grad PV| on the sphere.
  wind            the trapezoid integral of the speed |V| over 400, 300, 250, 200, 150, 100 hPa divided by 300 hPa
                  (the "integral average" / "pressure-weighted" 400-100 hPa wind of Koch et al. 2006).
The thresholds are empirical and grid-dependent: Christenson et al. chose theirs from cross-sections through jets,
above the median |grad PV| of the channel columns with wind above 30 m/s (0.64e-5 PVU/m on the 2.5 deg NCEP-NCAR
grid); Winters et al. (2020) used the values above on the 0.5 deg CFSR grid, which this grid matches. On ERA5
1991-2020 they give the North American superposition count threshold (the top 10 %) of Winters et al., 18 columns.
The ERA5 reference (build_jet_branches_ref.py) measures how much the other readings of the layer and channel wording,
and the thresholds, change the climatology.

Grid. ERA5 and ECMWF open data come at 0.25 deg; both are reduced to 0.5 deg by the area-conservative [1/4, 1/2, 1/4]
average of pacjet_core (every source gets the same operation) over the band BAND_LAT, wider than the 10-80N output
LAT so that every derivative at the output rows is centred. PV is Ertel PV on the nine pressure levels LEVS
(quantmet.pv, unsmoothed: the conservative reduction is the only filter), interpolated linearly in theta to the
isentropes.

The layers are the cold-season (November-March) ones; outside it the jets live on other isentropes and the scheme
does not apply. Day-of-season helpers (season_slot, smooth_season) index the climatology on a 152-slot season
(1 November to 31 March with 29 February as its own slot) so leap and common years line up.

Maps and events. A 0.5 deg column mask is a ribbon one to three columns wide, with isolated columns scattered around
it, so the maps are of the neighbourhood: a branch counts at a column when it is identified within NEAR_KM of it
(near), in the forecasts and in the ERA5 reference alike. A superposition event in a region is a group of superposed
columns, each within LINK_KM of another, at least as large as the region's threshold (groups_of; Winters et al. 2020).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from quantmet.degdays import doy366
from quantmet.pv import ertel_pv, on_isentrope

sys.path.insert(0, str(Path(__file__).parent))
import pacjet_core as PC

A_EARTH = 6.371e6
LEVS = (700, 600, 500, 400, 300, 250, 200, 150, 100)  # hPa: the open-data levels the DT charts use
WIND_LEVS = (400, 300, 250, 200, 150, 100)  # the wind criterion's layer
V_MIN = 30.0  # m/s
CHANNEL = (1.0, 3.0)  # PVU
LAYERS = {"polar": (315.0, 320.0, 325.0, 330.0), "stj": (340.0, 345.0, 350.0, 355.0)}
THRESH = {"polar": 1.4e-5, "stj": 0.9e-5}  # PVU/m on the 0.5 deg grid (Winters et al. 2020)
BRANCHES = ("polar", "stj", "sup")
BAND_LAT = (2.5, 87.5)  # rows of the 0.5 deg working band (from 0.25 deg rows 2.25..87.75)
BAND = np.round(np.arange(BAND_LAT[0], BAND_LAT[1] + 1e-6, 0.5), 3)
LAT = np.round(np.arange(10.0, 80.0 + 1e-6, 0.5), 3)  # output rows
LON = np.round(np.arange(0.0, 360.0, 0.5), 3)
# Superposition regions (lon 0-360): the western North Pacific box of Handlos & Martin (2016, 2021) and the North
# American domain of Winters et al. (2020).
REGIONS = {
    "wnp": dict(label="Western North Pacific", lat=(30.0, 40.0), lon=(135.0, 175.0)),
    "na": dict(label="North America", lat=(10.0, 80.0), lon=(220.0, 310.0)),
}
LINK_KM = 1000.0  # superposed columns within this distance of each other form one group (Winters et al. 2020)
TRACK_KM, TRACK_H = 1500.0, 30.0  # an event continues one whose centre was this close this recently (Winters et al.)
NEAR_KM = 300.0  # the maps' neighbourhood: a branch within this distance of the column
SMOOTH_HALF = 15  # days: the climatology is a centred 31-day window over the season


# ── grid ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def band_rows025(lat025) -> np.ndarray:
    """Indices into a 0.25 deg latitude axis (any order) of the rows BAND_LAT needs, ordered south to north."""
    lat025 = np.round(np.asarray(lat025, float), 3)
    want = np.round(np.arange(BAND_LAT[0] - 0.25, BAND_LAT[1] + 0.25 + 1e-6, 0.25), 3)
    return np.array([int(np.flatnonzero(lat025 == v)[0]) for v in want])


def to_half(f025) -> np.ndarray:
    """(..., rows, 1440) on the 0.25 deg rows of band_rows025 (south to north, longitudes 0..359.75) -> the 0.5 deg
    band (..., BAND, LON): pacjet_core's conservative average along both axes, periodic in longitude."""
    f = np.concatenate([f025[..., -1:], f025, f025[..., :1]], axis=-1)
    return PC._smooth_half(PC._smooth_half(f, -2), -1)


# ── the diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────
def wind_layer_mean(u, v, levs=LEVS) -> np.ndarray:
    """Pressure-weighted mean wind speed over 400-100 hPa: the trapezoid integral of |V| in p over WIND_LEVS / 300 hPa.
    u, v (lev, lat, lon) on `levs` (hPa)."""
    idx = [list(levs).index(p) for p in WIND_LEVS]
    s = np.hypot(np.asarray(u, float)[idx], np.asarray(v, float)[idx])
    p = np.asarray(WIND_LEVS, float)
    dp = p[:-1] - p[1:]
    return np.tensordot(dp, 0.5 * (s[:-1] + s[1:]), axes=(0, 0)) / (p[0] - p[-1])


def grad_mag(f, lat) -> np.ndarray:
    """|grad f| per metre on the sphere: centred differences, periodic in longitude (a global ring of columns)."""
    phi = np.deg2rad(np.asarray(lat, float))
    dlam = 2 * np.pi / f.shape[-1]
    fx = (np.roll(f, -1, -1) - np.roll(f, 1, -1)) / (2 * dlam * A_EARTH * np.cos(phi)[:, None])
    fy = np.gradient(f, phi, axis=-2) / A_EARTH
    return np.hypot(fx, fy)


def channel_gradient(pv, lat) -> np.ndarray:
    """|grad pv| (PVU/m) where 1 <= pv <= 3 PVU, NaN elsewhere (and where pv or a neighbour is undefined)."""
    g = grad_mag(pv, lat)
    with np.errstate(invalid="ignore"):
        inside = (pv >= CHANNEL[0]) & (pv <= CHANNEL[1])
    return np.where(inside, g, np.nan)


def layer_gradients(iso, lat) -> dict:
    """{layer: PV on its isentropes (4, lat, lon)} -> {layer: the in-channel gradient of the layer-mean PV}. A column
    where any of the layer's isentropes is undefined has no layer PV."""
    return {name: channel_gradient(pv.mean(axis=0), lat) for name, pv in iso.items()}


def diagnose(t, u, v, lat, lon, levs=LEVS) -> dict:
    """One analysis time on one grid. t, u, v (lev, lat, lon) on `levs` hPa, lat south to north, lon a global ring.

    Returns 'vbar' (the 400-100 hPa wind), 'theta' (lev, lat, lon), 'pv' {layer: PV on each of its isentropes
    (4, lat, lon)} and 'grad' {layer: the in-channel gradient of the layer-mean PV}, all on the input rows."""
    pv, th = ertel_pv(t, u, v, np.asarray(levs, float) * 100.0, lat, lon, smooth=False)
    iso = {name: np.stack([on_isentrope(pv, th, u, v, th0)["pv"] for th0 in thetas]) for name, thetas in LAYERS.items()}
    return {"vbar": wind_layer_mean(u, v, levs), "theta": th, "pv": iso, "grad": layer_gradients(iso, lat)}


def identify(vbar, g_polar, g_stj, thresh=THRESH) -> dict:
    """Boolean masks 'polar', 'stj' and 'sup' from the wind and the two layers' in-channel gradients."""
    with np.errstate(invalid="ignore"):
        windy = vbar > V_MIN
        polar = windy & (g_polar >= thresh["polar"])
        stj = windy & (g_stj >= thresh["stj"])
    return {"polar": polar, "stj": stj, "sup": polar & stj}


def masks_025(t, u, v) -> np.ndarray:
    """t, u, v (lev, rows, 1440) on the 0.25 deg rows of band_rows025 -> (3, LAT, LON) boolean masks of BRANCHES: the
    whole identification of one analysis or one member, as the ERA5 reference makes it."""
    d = diagnose(*(to_half(x) for x in (t, u, v)), BAND, LON)
    sl = output_rows(BAND)
    m = identify(d["vbar"][sl], d["grad"]["polar"][sl], d["grad"]["stj"][sl])
    return np.stack([m[b] for b in BRANCHES])


def output_rows(lat) -> slice:
    """The slice of `lat` (south to north) that is the 10-80N output."""
    lat = np.round(np.asarray(lat, float), 3)
    i0 = int(np.flatnonzero(lat == LAT[0])[0])
    i1 = int(np.flatnonzero(lat == LAT[-1])[0])
    return slice(i0, i1 + 1)


# ── regions and groups ──────────────────────────────────────────────────────────────────────────────────────────────
def region_mask(lat, lon, region) -> np.ndarray:
    """(lat, lon) boolean mask of a REGIONS box (edges included)."""
    box = REGIONS[region]
    la = (np.asarray(lat) >= box["lat"][0]) & (np.asarray(lat) <= box["lat"][1])
    lo = (np.asarray(lon) >= box["lon"][0]) & (np.asarray(lon) <= box["lon"][1])
    return la[:, None] & lo[None, :]


def groups(lat, lon, km=LINK_KM) -> np.ndarray:
    """Single-linkage groups of points (degrees): two points are in one group when a chain of points, each within
    `km` of the next, joins them. Returns a group label per point (0..n_groups-1)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    if lat.size < 2:
        return np.zeros(lat.size, int)
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    chord = 2 * np.sin(km / (2 * A_EARTH / 1000.0))  # the chord of a `km` great-circle arc on the unit sphere
    lab = fcluster(linkage(xyz, method="single"), t=chord, criterion="distance")
    return np.unique(lab, return_inverse=True)[1]


def great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between points given in degrees."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(np.asarray(lon2, float) - np.asarray(lon1, float))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return A_EARTH / 1000.0 * np.arccos(np.clip(c, -1.0, 1.0))


def groups_of(mask, in_region, lat=LAT, lon=LON) -> list:
    """The superposed columns of one analysis inside a region -> [(columns, centre lat, centre lon)], largest first
    (groups by `groups`; the centre longitude is the circular mean)."""
    ii, jj = np.nonzero(mask & in_region)
    if not ii.size:
        return []
    la, lo = np.asarray(lat)[ii], np.asarray(lon)[jj]
    lab = groups(la, lo)
    out = []
    for g in range(lab.max() + 1):
        sel = lab == g
        z = np.exp(1j * np.deg2rad(lo[sel])).mean()
        out.append((int(sel.sum()), float(la[sel].mean()), float(np.rad2deg(np.angle(z)) % 360)))
    return sorted(out, reverse=True)


def track(times, found, thr) -> list:
    """Events from per-analysis groups (`found`, one groups_of list per time). A group of >= thr columns is an event at
    its analysis; it continues the event of a group up to TRACK_H earlier whose centre is within TRACK_KM (the
    nearest), else starts one. -> [{onset, end, analyses, peak_columns, lat, lon}] (lat/lon: the first centre)."""
    rows, recent = [], []  # recent: (time, lat, lon, event index)
    window = np.timedelta64(int(TRACK_H), "h")
    for t, gs in zip(times, found):
        recent = [r for r in recent if (t - r[0]) <= window]
        now = []
        for size, la, lo in gs:
            if size < thr:
                continue
            near_ = [(great_circle_km(la, lo, r[1], r[2]), r[3]) for r in recent if r[0] < t]
            near_ = [x for x in near_ if x[0] <= TRACK_KM]
            if near_:
                e = min(near_)[1]
                ev = rows[e]
                if ev["end"] != t:  # two groups of one analysis joining one event count it once
                    ev["end"], ev["analyses"] = t, ev["analyses"] + 1
                ev["peak_columns"] = max(ev["peak_columns"], size)
            else:
                e = len(rows)
                rows.append(dict(onset=t, end=t, analyses=1, peak_columns=size, lat=round(la, 2), lon=round(lo, 2)))
            now.append((t, la, lo, e))
        recent += now
    return rows


def near(mask, lat=LAT, km=NEAR_KM) -> np.ndarray:
    """(..., lat, lon) boolean -> True at the columns with the mask within `km` (great circle) of them. Longitudes are
    a global ring; rows beyond the given ones do not count. Each target row takes the source rows within `km`
    meridionally, each widened by the half-width in columns the great circle allows between the two latitudes."""
    from scipy.ndimage import maximum_filter1d

    mask = np.asarray(mask, bool)
    lat = np.asarray(lat, float)
    phi = np.deg2rad(lat)
    dlon = 360.0 / mask.shape[-1]
    ang = km / (A_EARTH / 1000.0)
    reach = int(np.floor(np.rad2deg(ang) / (lat[1] - lat[0]) + 1e-9))
    out = np.zeros_like(mask)
    n = lat.size
    for k in range(-reach, reach + 1):
        tgt = np.arange(max(0, -k), min(n, n - k))
        src = tgt + k
        c = (np.cos(ang) - np.sin(phi[tgt]) * np.sin(phi[src])) / (np.cos(phi[tgt]) * np.cos(phi[src]))
        half = np.floor(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))) / dlon + 1e-9).astype(int)
        for h in np.unique(half):
            rows = half == h
            grown = maximum_filter1d(mask[..., src[rows], :], size=2 * h + 1, axis=-1, mode="wrap")
            out[..., tgt[rows], :] |= grown
    return out


# ── the season ──────────────────────────────────────────────────────────────────────────────────────────────────────
def season_slot(times) -> np.ndarray:
    """0..151: 1 November = 0, 31 December = 60, 1 January = 61, 29 February = 120, 1 March = 121, 31 March = 151 in
    every year (a 366-slot day of year, quantmet.degdays.doy366, folded onto the season). -1 outside November-March."""
    s = doy366(times)
    return np.where(s >= 305, s - 305, np.where(s <= 90, s + 61, -1))


def slot_position() -> np.ndarray:
    """Each slot's day of the season on the common-year calendar (1 November = 0, 31 March = 150): 29 February sits at
    119.5, between 28 February and 1 March."""
    x = np.arange(152, dtype=float)
    return np.where(x < 120, x, np.where(x == 120, 119.5, x - 1))


def smooth_season(counts, samples, half=SMOOTH_HALF, at=None) -> np.ndarray:
    """Frequency (0..1) per slot from per-slot counts (152, ...) and the number of analyses per slot (152,): the
    counts over all slots within `half` days (common-year calendar, so 29 February adds its own few analyses) divided
    by their analyses. The window is cut at 1 November and 31 March; the season's ends rest on fewer analyses.
    `at`: only these slots (in that order), instead of all 152."""
    x = slot_position()
    w = (np.abs(x[:, None] - x[None, :]) <= half).astype(float)  # (slot, slot)
    if at is not None:
        w = w[np.asarray(at, int)]
    n = w @ np.asarray(samples, float)
    c = np.tensordot(w, np.asarray(counts, float), axes=(1, 0))
    return c / n.reshape((-1,) + (1,) * (c.ndim - 1))
