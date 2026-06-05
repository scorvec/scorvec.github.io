#!/usr/bin/env python3
"""Shared constants for the ensemble-anomaly monitor.

Per-variable analysis domains:
  z500 → full NORTHERN HEMISPHERE (the wave pattern is hemispheric), polar stereo.
  t2m  → NORTH AMERICA (sensible-weather focus), Lambert conformal.
Both on a 0.5° lat-lon grid; all sources regrid to the variable's grid.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
G = 9.80665                                  # geopotential → height

# Northern-hemisphere grid (z500): 0–90 N, all longitudes, 0.5°.
NH_LAT = np.arange(90.0, -0.01, -0.5)        # 90N → 0   (181)
NH_LON = np.arange(0.0, 359.51, 0.5)         # 0 → 359.5 (720)
# North-America grid (t2m): 15–75 N, 170–50 W, 0.5°.
NA_LAT = np.arange(75.0, 14.99, -0.5)        # 75N → 15N (121)
NA_LON = np.arange(190.0, 310.01, 0.5)       # 190 → 310 E (241)

# Per-variable grid + the ERA5 .sel box (with margin for interpolation) + map region.
GRIDS = {
    "z500": dict(lat=NH_LAT, lon=NH_LON, region="nh",
                 sel=dict(latitude=slice(91, -1))),                     # full NH, all lon
    "t2m": dict(lat=NA_LAT, lon=NA_LON, region="na",
                sel=dict(latitude=slice(78, 12), longitude=slice(186, 314))),
}


def grid(var: str):
    g = GRIDS[var]
    return g["lat"], g["lon"]


# Forecast lead steps: daily, Day 0–15 (hours).
LEADS = list(range(0, 361, 24))              # 0,24,…,360  (16 frames)

# ERA5 zarr stores for the climatology (anon gcsfs).
#   ARCO  — native 0.25°, but pressure-level vars are chunked with ALL 37 levels
#           together, so a single-level read drags down all 37 (~37× waste). Fine for
#           SURFACE vars (single level → no waste), where we want full 0.25° detail.
#   WB2_15 — WeatherBench2 1.5° (6-hourly). Tiny per-field, only 13 levels. A 500 mb
#           HEIGHT normal is smooth, so 1.5° interpolated up to our 0.5° grid is
#           lossless in practice — and it sidesteps the 37-level waste entirely.
ARCO   = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2_15 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# var → fields. era5 = store variable; clim_store = which zarr to read normals from;
# ecmwf = open-data param; gefs = Herbie search.
VARS = {
    "z500": dict(label="500 mb geopotential height", units="dam",
                 era5="geopotential", era5_level=500, era5_scale=1.0 / G / 10.0,  # m²/s² → dam
                 clim_store=WB2_15,                                               # 1.5° (smooth field)
                 model_scale=0.1,                                                 # model gpm → dam
                 gefs_search=r":HGT:500 mb:", ecmwf_param="gh", ecmwf_levtype="pl", ecmwf_levelist=[500],
                 aifs_param="z", aifs_scale=1.0 / G / 10.0,                       # AIFS uses z (m²/s²)
                 geps="HGT_ISBL_0500"),
    "t2m": dict(label="2 m temperature", units="K",
                era5="2m_temperature", era5_level=None, era5_scale=1.0, model_scale=1.0,
                clim_store=ARCO,                                                  # 0.25° native (surface)
                gefs_search=r":TMP:2 m above ground:", ecmwf_param="2t", ecmwf_levtype="sfc",
                ecmwf_levelist=None, aifs_param="2t", aifs_scale=1.0, geps="TMP_TGL_2m"),
}

PERIODS = {"30yr": (1991, 2020), "10yr": (2014, 2023)}
ENSEMBLES = ["gefs", "ifs", "aifs", "geps"]                 # the fetched models (panel order)
ENS_LABEL = {"gefs": "GEFS", "ifs": "ECMWF IFS-ENS", "aifs": "AIFS-ENS", "geps": "GEPS (CMC)",
             "allmean": "All-Ensemble Mean",                # derived consensus (not fetched)
             "combined": "Combined AIFS-ENS + ECMWF IFS-ENS"}
