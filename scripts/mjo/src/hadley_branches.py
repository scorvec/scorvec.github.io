#!/usr/bin/env python3
"""The Hadley cells' upper and descending branches: the poleward outflow at 200 hPa, the subsiding air under it and the
subtropical ridges that the descent builds - definitions shared by the climatology (build_branch_clim.py), the analysis
archive and the maps and tracker (hadley_branch_maps.py).

Fields, all on WeatherBench2's 1.5 deg grid (the ERA5 climatology's own), 00 UTC:
  vchi200, uchi200  the irrotational (divergent) wind at 200 hPa, from velocity potential (wind200_vpot, the site's one
                    solver, truncated at LMAX for every source alike): the Hadley outflow is its poleward part
  w500              omega at 500 hPa (Pa/s, positive = descent)
  z500              geopotential height at 500 hPa (m): the 588-dam line is the subtropical ridge as forecasters draw it
  msl               mean sea-level pressure (hPa): the surface subtropical highs
A 0.25 deg source is averaged conservatively onto the grid (coarse15); WeatherBench2's ERA5 already is that grid.

Indices of a (7-day-mean) set of fields, per basin sector and hemisphere (SUBS, RIDGE, HIGHS):
  out_<S>   poleward vchi200 over 10-30 deg in the sector's hemisphere (m/s)
  desc_<S>  omega 500 over 15-35 deg (hPa/day, positive = descent)
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
from wind200_vpot import irrotational_wind, velocity_potential

LAT = np.linspace(90.0, -90.0, 121)  # WeatherBench2's rows, north to south
LON = np.arange(0.0, 360.0, 1.5)
FIELDS = ("vchi200", "uchi200", "w500", "z500", "msl")
A = 6.371e6
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


def _lon_mask(w: float, e: float) -> np.ndarray:
    return (w <= LON) & (e > LON) if w < e else (w <= LON) | (e > LON)


def _lat_mask(hemi: str, band: tuple) -> np.ndarray:
    lo, hi = band
    return (lo <= LAT) & (hi >= LAT) if hemi == "N" else (-lo >= LAT) & (-hi <= LAT)


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
    """A (lat, lon) field on any regular global grid -> the 1.5 deg grid, conservatively (identity on that grid)."""
    if lat.size == LAT.size and lon.size == LON.size and np.allclose(lat, LAT) and np.allclose(lon % 360, LON):
        return np.asarray(f, float)
    Wa, Wo = _weights(tuple(np.round(lat, 6)), tuple(np.round(lon % 360, 6)))
    return Wa @ np.asarray(f, float) @ Wo.T


def divergent_wind(u200: xr.DataArray, v200: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """(uchi, vchi) at 200 hPa on the 1.5 deg grid, through the site's velocity-potential solver."""
    chi, dlat, dlon = velocity_potential(u200, v200)
    uchi, vchi = irrotational_wind(chi, dlat, dlon)
    pts = xr.DataArray(
        np.stack([uchi, vchi]), dims=("c", "latitude", "longitude"), coords={"latitude": dlat, "longitude": dlon}
    )
    if dlon[-1] < 360.0 - 1e-6:  # the solver's DH2 grid carries the 360 deg column; interpolation needs one
        pts = xr.concat([pts, pts.isel(longitude=[0]).assign_coords(longitude=[dlon[0] + 360.0])], dim="longitude")
    out = pts.sortby("latitude").interp(latitude=LAT, longitude=LON).values
    return out[0], out[1]


def fields(u200, v200, w500, z500, msl) -> dict:
    """The five 1.5 deg fields from 2-D (latitude, longitude) DataArrays on any global grid; w500 in Pa/s, z500 in m,
    msl in Pa."""
    uchi, vchi = divergent_wind(u200, v200)

    def c(da):
        da = da.sortby("latitude", ascending=False)
        return coarse15(da.values, da.latitude.values, da.longitude.values)

    return {"vchi200": vchi, "uchi200": uchi, "w500": c(w500), "z500": c(z500), "msl": c(msl) / 100.0}


def _area_mean(f: np.ndarray, mlat: np.ndarray, mlon: np.ndarray) -> float:
    w = np.cos(np.deg2rad(LAT[mlat]))[:, None] * np.ones(mlon.sum())
    return float((f[np.ix_(mlat, mlon)] * w).sum() / w.sum())


CELL_AREA = (A**2) * np.deg2rad(1.5) * np.deg2rad(1.5) * np.cos(np.deg2rad(LAT))[:, None] / 1e12  # 10^6 km^2


def indices(F: dict, z588: float = Z588) -> dict:
    """The tracker's numbers from one set of (7-day-mean) fields. z588 lets the ERA5 years be read at the threshold the
    rise of the 500 hPa surface puts them at (build_branch_clim.py)."""
    out = {}
    for s, (hemi, w, e) in SUBS.items():
        mlon = _lon_mask(w, e)
        sgn = 1.0 if hemi == "N" else -1.0
        out[f"out_{s}"] = sgn * _area_mean(F["vchi200"], _lat_mask(hemi, OUT_BAND), mlon)
        out[f"desc_{s}"] = PA_S_TO_HPA_DAY * _area_mean(F["w500"], _lat_mask(hemi, DESC_BAND), mlon)
    for s, (hemi, w, e) in RIDGE.items():
        mr = np.ix_(_lat_mask(hemi, RIDGE_BAND), _lon_mask(w, e))
        z = F["z500"][mr]
        out[f"r588_{s}"] = float((CELL_AREA * np.ones((1, LON.size)))[mr][z >= z588].sum())
        out[f"zmax_{s}"] = float(z.max())
    for s, (la0, la1, w, e) in HIGHS.items():
        mlat, mlon = (la0 <= LAT) & (la1 >= LAT), _lon_mask(w, e)
        p = F["msl"][np.ix_(mlat, mlon)]
        i, j = np.unravel_index(np.argmax(p), p.shape)
        out[f"high_{s}"], out[f"hlat_{s}"], out[f"hlon_{s}"] = float(p[i, j]), float(LAT[mlat][i]), float(LON[mlon][j])
    return out


def ridge_ladder(z500: np.ndarray, thresholds: np.ndarray) -> dict:
    """Per sector: the ridge-band area at or above each threshold (10^6 km^2), and the band's mean z500 (m) - what the
    climatology needs to read the 588-dam area at the threshold a sector's trend in z500 implies for each year."""
    out = {}
    for s, (hemi, w, e) in RIDGE.items():
        mr = np.ix_(_lat_mask(hemi, RIDGE_BAND), _lon_mask(w, e))
        z, a = z500[mr], (CELL_AREA * np.ones((1, LON.size)))[mr]
        out[s] = (np.array([a[z >= t].sum() for t in thresholds]), float((z * a).sum() / a.sum()))
    return out
