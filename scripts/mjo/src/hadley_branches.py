#!/usr/bin/env python3
"""The Hadley cells' upper and descending branches: the poleward outflow beneath the tropopause, the subsiding air under
it and the subtropical ridges the descent builds - definitions shared by the ERA5 climatology (build_branch_clim.py),
the analysis pipeline and its maps and tracker (hadley_branch_maps.py) and the trajectories (hadley_traj.py).

Every source is first put on WeatherBench2's 1.5 deg grid (the ERA5 climatology's own) on the 13 standard levels
LEVELS that ERA5 and the ECMWF open data share, and the fields of a day are those of its four synoptic times' mean:
  ptrop             the lapse-rate tropopause (WMO: the lowest level above 550 hPa where the lapse rate falls below
                    2 K/km and stays below it on average for 2 km), found on the standard levels as Reichler et al.
                    (2003, GRL 30, 2042) do: half-level lapse rates in p^kappa, the crossing interpolated (hPa)
  vout, uout        the irrotational (divergent) wind averaged by mass over the OUT_DEPTH below the local tropopause:
                    the layer the Hadley outflow fills, at ~100-250 hPa in the deep tropics and lower where the
                    tropopause descends poleward. The divergent wind is solved at OUT_LEVELS through the site's
                    velocity-potential solver and kept to spherical-harmonic degree KEEP (large scales only)
  wmid              omega averaged by mass over DESC_LAYER (Pa/s, positive = descent), to degree KEEP
  z500              geopotential height at 500 hPa (m): the 588-dam line is the subtropical ridge as forecasters draw it
  msl               mean sea-level pressure (hPa): the surface subtropical highs

Indices of a (7-day-mean) set of fields, per basin sector and hemisphere (SUBS, RIDGE, HIGHS):
  out_<S>   poleward vout over 10-30 deg in the sector's hemisphere (m/s)
  desc_<S>  wmid over 15-35 deg (hPa/day, positive = descent)
  r588_<S>  area of the sector's 10-45 deg band where z500 >= Z588 (10^6 km^2); zmax_<S> its highest z500 (m)
  high_<S>  the subtropical high's central pressure (hPa), with its latitude and longitude (hlat_, hlon_)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from wind200_vpot import irrotational_wind, truncated, velocity_potential

LAT = np.linspace(90.0, -90.0, 121)  # WeatherBench2's rows, north to south
LON = np.arange(0.0, 360.0, 1.5)
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)  # hPa, surface upward
OUT_LEVELS = (100, 150, 200, 250, 300, 400)  # every outflow layer lies inside these
OUT_DEPTH = 150.0  # hPa below the tropopause
DESC_LAYER = (300.0, 700.0)  # hPa
KEEP = 42  # spherical-harmonic degree kept in the divergent wind and omega (~500 km)
LMAX15 = 106  # the solver's grid for 1.5 deg input (the site's default)
FIELDS = ("vout", "uout", "ptrop", "wmid", "z500", "msl")
A = 6.371e6
G, CP, KAPPA = 9.80665, 1004.64, 287.04 / 1004.64
LAPSE_C = 2.0e-3  # K/m, the WMO criterion
TROP_RANGE = (75.0, 550.0)  # hPa, where a tropopause is looked for
Z588 = 5880.0
PA_S_TO_HPA_DAY = 864.0  # 1 Pa/s = 864 hPa/day
# (hemisphere, west, east) on 0..360 (east < west wraps through 0). SUBS: the eastern-basin subsidence regions under each
# subtropical high, where the outflow and the descent are read (east of the convergence zones: in January the South
# Pacific band at 160-80W took in the SPCZ's ascent); RIDGE: the whole basin, for the 588-dam ridge.
SUBS = {
    "npac": ("N", 180.0, 240.0),
    "natl": ("N", 300.0, 345.0),
    "spac": ("S", 230.0, 285.0),
    "satl": ("S", 330.0, 10.0),
    "sind": ("S", 60.0, 110.0),
    "nh": ("N", 0.0, 360.0),
    "sh": ("S", 0.0, 360.0),
}
RIDGE = {
    "npac": ("N", 140.0, 240.0),
    "natl": ("N", 280.0, 350.0),
    "spac": ("S", 200.0, 290.0),
    "satl": ("S", 320.0, 20.0),
    "sind": ("S", 40.0, 115.0),
    "nh": ("N", 0.0, 360.0),
    "sh": ("S", 0.0, 360.0),
}
# subtropical high: (lat south edge, lat north edge, west, east), over ocean only (a box reaching the coast found the winter
# Great Basin high instead of the Pacific one)
HIGHS = {
    "npac": (20.0, 45.0, 170.0, 230.0),
    "natl": (20.0, 50.0, 285.0, 345.0),
    "spac": (-45.0, -15.0, 220.0, 285.0),
    "satl": (-45.0, -15.0, 325.0, 10.0),
    "sind": (-45.0, -15.0, 45.0, 110.0),
}
OUT_BAND, DESC_BAND, RIDGE_BAND = (10.0, 30.0), (15.0, 35.0), (10.0, 45.0)


def lon_mask(w: float, e: float, lon: np.ndarray = LON) -> np.ndarray:
    return (w <= lon) & (e > lon) if w < e else (w <= lon) | (e > lon)


def lat_mask(hemi: str, band: tuple, lat: np.ndarray = LAT) -> np.ndarray:
    lo, hi = band
    return (lo <= lat) & (hi >= lat) if hemi == "N" else (-lo >= lat) & (-hi <= lat)


@lru_cache(maxsize=4)
def _weights(lat_src: tuple, lon_src: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(121, nlat) and (240, nlon) conservative weights onto the 1.5 deg grid: area overlaps of source cells of the
    source spacing with the target cells (latitude by sin, longitude periodic)."""
    la, lo = np.asarray(lat_src), np.asarray(lon_src)
    hla, hlo = 0.5 * abs(la[1] - la[0]), 0.5 * abs(lo[1] - lo[0])
    edges = np.clip(np.r_[90.0, 0.5 * (LAT[:-1] + LAT[1:]), -90.0], -90.0, 90.0)
    Wa = np.zeros((LAT.size, la.size))
    for j in range(LAT.size):
        hi, low = np.minimum(edges[j], la + hla), np.maximum(edges[j + 1], la - hla)
        Wa[j] = np.where(hi > low, np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(low)), 0.0)
    Wo = np.zeros((LON.size, lo.size))
    for i, c in enumerate(LON):
        d = (lo - c + 180.0) % 360.0 - 180.0  # source centre relative to the target centre
        Wo[i] = np.clip(np.minimum(d + hlo, 0.75) - np.maximum(d - hlo, -0.75), 0.0, None)
    return Wa / Wa.sum(1, keepdims=True), Wo / Wo.sum(1, keepdims=True)


def coarse15(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A (..., lat, lon) field on any regular global grid -> the 1.5 deg grid, conservatively (identity on that grid)."""
    if lat.size == LAT.size and lon.size == LON.size and np.allclose(lat, LAT) and np.allclose(lon % 360, LON):
        return np.asarray(f, float)
    Wa, Wo = _weights(tuple(np.round(lat, 6)), tuple(np.round(lon % 360, 6)))
    return Wa @ np.asarray(f, float) @ Wo.T  # matmul broadcasts over any leading axes


def _da(f: np.ndarray) -> xr.DataArray:
    return xr.DataArray(f, dims=("latitude", "longitude"), coords={"latitude": LAT, "longitude": LON})


def _on_grid(fields: np.ndarray, dlat: np.ndarray, dlon: np.ndarray) -> np.ndarray:
    """(n, lat, lon) fields on the solver's DH2 grid -> the 1.5 deg grid."""
    pts = xr.DataArray(fields, dims=("c", "latitude", "longitude"), coords={"latitude": dlat, "longitude": dlon})
    if dlon[-1] < 360.0 - 1e-6:  # the DH2 grid carries the 360 deg column; interpolation needs one
        pts = xr.concat([pts, pts.isel(longitude=[0]).assign_coords(longitude=[dlon[0] + 360.0])], dim="longitude")
    return pts.sortby("latitude").interp(latitude=LAT, longitude=LON).values


def divergent_wind(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(uchi, vchi) on the 1.5 deg grid from 1.5 deg u, v, to degree KEEP."""
    chi, dlat, dlon = velocity_potential(_da(u), _da(v), lmax=LMAX15, keep=KEEP)
    uchi, vchi = irrotational_wind(chi, dlat, dlon)
    out = _on_grid(np.stack([uchi, vchi]), dlat, dlon)
    return out[0], out[1]


def large_scale(f: np.ndarray) -> np.ndarray:
    """A 1.5 deg scalar kept to degree KEEP."""
    g, dlat, dlon = truncated(_da(f), KEEP, lmax=LMAX15)
    return _on_grid(g[None], dlat, dlon)[0]


PK = (np.asarray(LEVELS, float) * 100.0) ** KAPPA  # p^kappa of the levels (Pa^kappa), decreasing upward


def _at_pk(t: np.ndarray, pkx: np.ndarray) -> np.ndarray:
    """(LEVELS, n) profiles at p^kappa = pkx (n), linear in p^kappa, clipped to the profile's ends."""
    x = -PK
    k = np.clip(np.searchsorted(x, -pkx), 1, x.size - 1)
    w = np.clip((-pkx - x[k - 1]) / (x[k] - x[k - 1]), 0.0, 1.0)
    cols = np.arange(t.shape[1])
    return t[k - 1, cols] * (1.0 - w) + t[k, cols] * w


def tropopause(t: np.ndarray) -> np.ndarray:
    """The lapse-rate tropopause (hPa) of (LEVELS, ...) temperatures (K), as Reichler et al. (2003): the lapse rate at
    each half level in p^kappa, each upward crossing of LAPSE_C interpolated in p^kappa, the lowest one inside
    TROP_RANGE whose mean lapse rate over the 2 km above stays below LAPSE_C. NaN where there is none."""
    shape = t.shape[1:]
    t = t.reshape(len(LEVELS), -1)
    pk = PK[:, None]
    phk, th = 0.5 * (pk[1:] + pk[:-1]), 0.5 * (t[1:] + t[:-1])
    gam = (t[1:] - t[:-1]) / (pk[1:] - pk[:-1]) * phk / th * G / CP  # -dT/dz at the half levels (K/m)
    out = np.full(t.shape[1], np.nan)
    for j in range(1, gam.shape[0]):
        c = np.flatnonzero(np.isnan(out) & (gam[j - 1] >= LAPSE_C) & (gam[j] < LAPSE_C))
        if not c.size:
            continue
        f = (LAPSE_C - gam[j - 1, c]) / (gam[j, c] - gam[j - 1, c])
        pk_tp = phk[j - 1, 0] + f * (phk[j, 0] - phk[j - 1, 0])
        t_tp = _at_pk(t[:, c], pk_tp)
        p_tp = pk_tp ** (1.0 / KAPPA)
        p_2km = p_tp * np.exp(-2000.0 * G / (287.04 * t_tp))  # hypsometric, at the tropopause temperature
        ok = (t_tp - _at_pk(t[:, c], p_2km**KAPPA)) / 2000.0 < LAPSE_C
        ok &= (p_tp >= TROP_RANGE[0] * 100.0) & (p_tp <= TROP_RANGE[1] * 100.0)
        out[c[ok]] = p_tp[ok] / 100.0
    return out.reshape(shape)


def _fill_rows(f: np.ndarray) -> np.ndarray:
    """NaN points of a (lat, lon) field set to their latitude row's median (rare: no tropopause in range)."""
    return np.where(np.isnan(f), np.nanmedian(f, axis=1, keepdims=True), f)


def layer_mean(f: np.ndarray, plev, top, bottom, n: int = 31) -> np.ndarray:
    """Mass-weighted mean over [top, bottom] (hPa, scalars or per point) of (len(plev), ...) values linear in p between
    the levels plev (ascending), the layer clipped to the levels' span."""
    plev = np.asarray(plev, float)
    shape = f.shape[1:]
    f = f.reshape(plev.size, -1)
    top = np.clip(np.broadcast_to(top, shape).ravel(), plev[0], plev[-1])
    bottom = np.clip(np.broadcast_to(bottom, shape).ravel(), plev[0], plev[-1])
    pp = top[None] + np.linspace(0.0, 1.0, n)[:, None] * (bottom - top)[None]
    k = np.clip(np.searchsorted(plev, pp), 1, plev.size - 1)
    w = (pp - plev[k - 1]) / (plev[k] - plev[k - 1])
    vals = np.take_along_axis(f, k - 1, axis=0) * (1.0 - w) + np.take_along_axis(f, k, axis=0) * w
    trap = np.full(n, 1.0)
    trap[[0, -1]] = 0.5
    return (np.tensordot(trap, vals, axes=(0, 0)) / trap.sum()).reshape(shape)


def fields(u: np.ndarray, v: np.ndarray, w: np.ndarray, t: np.ndarray, z500: np.ndarray, msl: np.ndarray) -> dict:
    """The six fields from (LEVELS, lat, lon) u, v (m/s), omega (Pa/s) and temperature (K) on the 1.5 deg grid, the
    500 hPa height (m) and the sea-level pressure (Pa)."""
    lev = list(LEVELS)
    ptrop = _fill_rows(tropopause(t))
    chi = [divergent_wind(u[lev.index(L)], v[lev.index(L)]) for L in OUT_LEVELS]
    vout = layer_mean(np.stack([c[1] for c in chi]), OUT_LEVELS, ptrop, ptrop + OUT_DEPTH)
    uout = layer_mean(np.stack([c[0] for c in chi]), OUT_LEVELS, ptrop, ptrop + OUT_DEPTH)
    desc = [L for L in sorted(LEVELS) if DESC_LAYER[0] <= L <= DESC_LAYER[1]]
    wmid = large_scale(layer_mean(w[[lev.index(L) for L in desc]], desc, *DESC_LAYER))
    return {"vout": vout, "uout": uout, "ptrop": ptrop, "wmid": wmid, "z500": z500, "msl": msl / 100.0}


def _area_mean(f: np.ndarray, mlat: np.ndarray, mlon: np.ndarray) -> float:
    w = np.cos(np.deg2rad(LAT[mlat]))[:, None] * np.ones(mlon.sum())
    return float((f[np.ix_(mlat, mlon)] * w).sum() / w.sum())


CELL_AREA = (A**2) * np.deg2rad(1.5) * np.deg2rad(1.5) * np.cos(np.deg2rad(LAT))[:, None] / 1e12  # 10^6 km^2


def indices(F: dict, z588: float = Z588) -> dict:
    """The tracker's numbers from one set of (7-day-mean) fields. z588 lets the ERA5 years be read at the threshold the
    rise of the 500 hPa surface puts them at (build_branch_clim.py)."""
    out = {}
    for s, (hemi, w, e) in SUBS.items():
        mlon = lon_mask(w, e)
        sgn = 1.0 if hemi == "N" else -1.0
        out[f"out_{s}"] = sgn * _area_mean(F["vout"], lat_mask(hemi, OUT_BAND), mlon)
        out[f"desc_{s}"] = PA_S_TO_HPA_DAY * _area_mean(F["wmid"], lat_mask(hemi, DESC_BAND), mlon)
    for s, (hemi, w, e) in RIDGE.items():
        mr = np.ix_(lat_mask(hemi, RIDGE_BAND), lon_mask(w, e))
        z = F["z500"][mr]
        out[f"r588_{s}"] = float((CELL_AREA * np.ones((1, LON.size)))[mr][z >= z588].sum())
        out[f"zmax_{s}"] = float(z.max())
    for s, (la0, la1, w, e) in HIGHS.items():
        mlat, mlon = (la0 <= LAT) & (la1 >= LAT), lon_mask(w, e)
        p = F["msl"][np.ix_(mlat, mlon)]
        i, j = np.unravel_index(np.argmax(p), p.shape)
        out[f"high_{s}"], out[f"hlat_{s}"], out[f"hlon_{s}"] = float(p[i, j]), float(LAT[mlat][i]), float(LON[mlon][j])
    return out


def ridge_ladder(z500: np.ndarray, thresholds: np.ndarray) -> dict:
    """Per sector: the ridge-band area at or above each threshold (10^6 km^2), and the band's mean z500 (m) - what the
    climatology needs to read the 588-dam area at the threshold a sector's trend in z500 implies for each year."""
    out = {}
    for s, (hemi, w, e) in RIDGE.items():
        mr = np.ix_(lat_mask(hemi, RIDGE_BAND), lon_mask(w, e))
        z, a = z500[mr], (CELL_AREA * np.ones((1, LON.size)))[mr]
        out[s] = (np.array([a[z >= t].sum() for t in thresholds]), float((z * a).sum() / a.sum()))
    return out
