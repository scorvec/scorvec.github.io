#!/usr/bin/env python3
"""Short-range model viewer data for models.html: HRRR, RRFS and RDPS fields on their native grids.

Fields: simulated IR satellite, cloud ceiling, visibility, temperature, dewpoint, vertically integrated smoke, 80 m and
10 m winds, surface gust, incoming shortwave, reflectivity with precipitation type, MSLP and run accumulations, plus the
convective, upper-air and snow-band fields listed in FIELDS.

No figures are rendered here. Each field and forecast hour becomes ONE 8-bit lossless WebP on the model's NATIVE grid
(HRRR/RRFS 3 km Lambert 1799 x 1059, RDPS 10 km rotated lat-lon, cropped to North America), and the page draws them
with WebGL: the colour table, zoom, pan, hover values and the vector boundaries are all client-side, so switching
field or hour costs one small file and zooming costs nothing.

Encoding (codes 0-254 data, 255 = missing / "none"); the page holds the same table (models.html, FIELDS):
  ir      K,  linear, 170 K + 0.625 K * code
  ceil    ft above ground, log from 50 to 50,000 ft; 255 = no ceiling; 0 = below 50 ft
  vis     statute miles, log from 0.03 to 10 mi; 254 = 10 mi or more (the METAR ceiling: above 10 mi is noise that
          tripled the file)
  t2m     deg F, linear, PER-FRAME offset (floor of the frame minimum), 0.5 F steps (1 F if the frame spans > 127 F)
  td2m    as t2m
  smoke   mg m-2 column smoke, log from 0.1 to 3,000; 0 = below 0.1
  wind80  m/s, EARTH-relative u in R and v in G, -50.8 + 0.4 * code (the page draws speed, arrows and hover values)
  wind10  as wind80, 10 m above ground
  gust    m/s, linear, 0.25 m/s steps from 0 (0-63.5 m/s = 0-123 kt; 254 = 63.5 or more): the model's instantaneous
          surface gust; the page draws the 10 m wind's barbs on it
  sw      W m-2, linear 5 W m-2 steps
  refl    HRRR/RRFS: R = dBZ / 0.5 at 1 km AGL (0 = below 5 dBZ), G = precipitation type (see ptype_flags);
          RDPS: R = precipitation rate, log 0.1-100 mm/h, G = type (RDPS_PTYPE); manifest enc refl2 / rate2
  apcp, asnow, afzra, aip   accumulation since the run started, 16-bit (R high byte, G low byte) in 0.01 in (snow 0.1 in)
  mslp    hPa, 940 + 0.5 * code in R, G = 1 where the surface pressure is below 850 hPa (terrain above ~1,500 m, where
          the sea-level reduction makes fake highs and lows - the page places no H/L there), on every SUB-th grid point
Every file is written with its image row 0 at the NORTH (grid row j = ny - 1 - image row).

Sources (NOAA from the AWS open-data buckets byte-ranged via the .idx; ECCC from the MSC Datamart):
  hrrr   noaa-hrrr-bdp-pds  hrrr.tHHz.wrfsfcfFF.grib2. 00/06/12/18Z to f48 only (the hourly 18-h runs are not used).
         IR = SBT124 (the GOES-East 10.3 um window band as UPP labels it; SBT114 is the GOES-West copy - verified by
         the limb darkening: SBT124 - SBT114 is +11 K at 70W and -1.4 K at 120W). Ceiling = HGT at "cloud ceiling"
         (geopotential metres ABOVE SEA LEVEL - it never falls below the terrain) minus HGT at the surface.
         Smoke = COLMD (column mass density of smoke). DSWRF is instantaneous.
  rrfs   noaa-rrfs-ops-pds  rrfs.YYYYMMDD/HH/rrfs.tHHz.2dfld.3km.fFFF.conus.grib2 (the HRRR grid, identically).
         00/06/12/18Z to f84, published hourly to f36 then 3-hourly (storage); the other hours only write
         sub-hourly 2-D files to f15-18 (not used).
         IR = SBTA1613 (GOES-16 ABI band 13, 10.3 um). Ceiling as HRRR. Smoke = COLMD of "particulate organic matter
         dry" (RRFS-SD's smoke tracer). DSWRF instantaneous (the file also has a 1-h mean). The operational files
         (from the 2026-10-15 12Z cycle) carry no FRZR: freezing rain is then derived like sleet (FZRA_DERIVED).
  rdps   dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km/HH/FFF/, 00/06/12/18Z to f84.
         No simulated brightness temperature, ceiling or visibility exist in the RDPS output. The IR slot shows the
         OLR-derived brightness temperature Tb = (OLR / sigma)^(1/4) from UpwardLongwaveRadiationFlux_NTAtm (an
         instantaneous field, PDT 4.0) - a
         broadband quantity, so clear skies read ~20-30 K colder than a 10.3 um window band. Shortwave = the
         difference of consecutive DownwardShortwaveRadiationFlux-Accum_Sfc steps / 3600 s = the MEAN over the
         previous hour. Smoke from RAQDPS-FireWork (model_raqdps, same Datamart, same 0.09 deg rotated grid family
         but a smaller 729 x 599 window): PM2.5-WildfireSmokePlume_EAtm, column mass density, 00/12Z to f72; a 06/18Z
         RDPS cycle takes the RAQDPS run 6 h older, matched on valid time.

    python scripts/models/models.py plan --models hrrr,rrfs,rdps --manifests assets/models
    python scripts/models/models.py run --model hrrr --cycle 2026093012 --site . --procs 4 --lists /tmp/mv
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "snowband"))
import snowband as sb  # the shared, validated transport (S3 / Datamart)
from manifest_merge import prune_ecape  # hrrr.json is shared with the ECAPE job
import points as PTS  # meteogram point series (points.json)
import upper as UPR
import winter as WIN  # FRAM ice, Cobb snow ratio, snow-depth increase  upper-air fields: 925/250 hPa, dynamic tropopause, 500 hPa

UA = {"User-Agent": "scorvec-models/1.0 (+https://scorvec.com/models.html)"}
SIGMA = 5.670374419e-8
DATA = "assets/models/data"

# ── encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────
FIELDS = {
    "ir": dict(enc="lin", off=170.0, step=0.625),
    "ceil": dict(enc="log", lo=50.0, hi=50000.0),
    "vis": dict(enc="log", lo=0.03, hi=10.0),  # 254 = 10 mi or more, as a METAR reports it
    "t2m": dict(enc="linf"),
    "td2m": dict(enc="linf"),
    "smoke": dict(enc="log", lo=0.1, hi=3000.0),
    "wind80": dict(enc="lin2", off=-50.8, step=0.4),
    "wind10": dict(enc="lin2", off=-50.8, step=0.4),
    "gust": dict(enc="lin", off=0.0, step=0.25),
    "sw": dict(enc="lin", off=0.0, step=5.0),
    "refl": dict(enc="refl2"),  # HRRR/RRFS: dBZ in R, precipitation type in G; RDPS: precipitation rate ("rate2")
    # mean-sea-level pressure, overlay only (isobars on the reflectivity map): 0.5 hPa from 940 hPa, stored
    # on every SUB-th grid point (12 km HRRR/RRFS, 20 km RDPS) - smooth, and the isobars are smoothed further anyway
    "mslp": dict(enc="lin", off=940.0, step=0.5),
    # precipitation accumulated SINCE THE RUN STARTED (the page differences two hours for any window): 16-bit, R = high
    # byte, G = low byte, in `unit` inches (liquid 0.01 in, to 655 in; snow 0.1 in). 0.001 in cost twice the bytes (the
    # low byte wraps every 0.26 in and turns to noise) for a precision no QPF product quotes
    "apcp": dict(enc="acc16", unit=0.01),  # total precipitation, liquid equivalent
    "asnow": dict(enc="acc16", unit=0.1),  # snowfall (HRRR/RRFS: the model's ASNOW; RDPS: 10:1 from its snow water)
    "afzra": dict(enc="acc16", unit=0.01),  # freezing rain, liquid equivalent (model's own FRZR / FreezingRain-Accum)
    "aip": dict(
        enc="acc16", unit=0.01
    ),  # sleet, liquid equivalent (HRRR/RRFS: APCP while the sleet type is on; RDPS: IcePellets-Accum)
    # convection and the boundary layer: updraft helicity and its swaths, max updraft, echo tops and PBL height.
    # HRRR wrfsfc / RRFS 2dfld; RDPS has only the PBL height.
    #   uh25 / uh03  updraft helicity 2-5 / 0-3 km AGL, the model's max over the hour ending at the valid time (MXUPHL), m2 s-2:
    #                code = round(254 sqrt(UH / 400)) (25 m2/s2 -> 64, 75 -> 110; 254 = 400 or more), 0 below 1 m2/s2
    #   uvv          max upward vertical velocity over the hour, 100-1000 hPa (MAXUVV), m/s: 0.25 m/s steps, 0 below 2 m/s
    #   *s           the RUN SWATH: running maximum since the start of the run (stored, one small file per hour - the page
    #                would otherwise need every earlier hour to show one map)
    #   etop         echo top (RETOP, the 18.3 dBZ top, m above sea level) in kft, 1 kft steps; 255 = no echo
    #   hpbl         PBL height (HPBL; RDPS PlanetaryBoundaryLayerHeight), m above ground, 25 m steps to 6,350 m
    "uh25": dict(enc="uh", top=400.0),
    "uh25s": dict(enc="uh", top=400.0),
    "uh03": dict(enc="uh", top=400.0),
    "uh03s": dict(enc="uh", top=400.0),
    "uvv": dict(enc="uvv", step=0.25),
    "uvvs": dict(enc="uvv", step=0.25),
    "etop": dict(enc="etop"),
    "hpbl": dict(enc="lin", off=0.0, step=25.0),
    # upper air (upper.py), every 3rd hour. "uvz": R, G = earth-relative u, v (off + step * code), B = height,
    # per-frame offset and step (manifest "q", like linf). 925 hPa: +-63.5 m/s in 0.5 m/s (~1 kt); 250 hPa: +-101.6 m/s
    # in 0.8 m/s (the jet reaches ~100 m/s). "dt": R = potential temperature on 2 PVU, 250 K + 0.6 K * code (to 402 K),
    # G, B = the wind on it as 250 hPa. "vz": R = 500 hPa absolute vorticity, 1e-5 s-1, -30 + 0.5 * code; G = height.
    # winter accumulations (winter.py), same 16-bit "since the run started" encoding
    "afram": dict(enc="acc16", unit=0.01),  # radial ice accretion, FRAM (Sanders and Barjenbruch 2016)
    "acobb": dict(enc="acc16", unit=0.1),  # snowfall with the Cobb column snow ratio (HRRR)
    "as10": dict(enc="acc16", unit=0.1),  # snowfall at 10:1 from the snow-phase liquid (HRRR, RRFS)
    "asnod": dict(enc="acc16", unit=0.1),  # snow-depth increase (sum of positive hourly changes)
    "ua925": dict(enc="uvz", off=-63.5, step=0.5),
    "ua250": dict(enc="uvz", off=-101.6, step=0.8),
    "dt": dict(enc="dt", th=(250.0, 0.6), off=-101.6, step=0.8),
    "z500": dict(enc="vz", vo=(-30.0, 0.5)),
    # 500 hPa wind for the barbs on the z500 map: a companion 2-channel frame (u, v earth-relative, as 250 hPa) on the
    # z500 grid - a 4th channel in z500 would be WebP alpha, which the canvas decode premultiplies (not bit-exact)
    "w500": dict(enc="lin2", off=-101.6, step=0.8),
    # potential vorticity (upper.py), 12 km (RDPS 20 km), every 3 h.
    # PV code = (PVU + 2) / 0.0625 (-2 .. +13.9 PVU; 255 = masked). pvu: 300-250 hPa layer mean (1 channel; barbs and heights
    # come from the ua250 frame); pvl: 850-700 hPa layer mean, G, B = 850 hPa wind (as 925), masked where the ground is above
    # 850 hPa; pv315..pv340: PV on the isentrope, G = its pressure, 100 + 2.5 hPa * code, masked outside 100-700 hPa;
    # w315..w340: the wind on it (as 250 hPa)
    "pvu": dict(enc="pv1", off=-2.0, step=0.0625),
    "pvl": dict(enc="pv3", off=-2.0, step=0.0625, uv=(-63.5, 0.5)),
    **{f"pv{int(t)}": dict(enc="pvth", off=-2.0, step=0.0625, p=(100.0, 2.5)) for t in UPR.THETAS},
    **{f"w{int(t)}": dict(enc="lin2", off=-101.6, step=0.8) for t in UPR.THETAS},
    # snow-band diagnostics: the SAME computation as the retired snowbands.html (scripts/snowband/snowband.py
    # smooth_diagnose / ingredients: Gaussian sigma 18 km on the native grid, evaluated on 12 km - the HRRR's "sub"
    # grid - or the RDPS's own 10 km grid), every 3 h from +3 h, in season (SB_SEASON). 255 = missing / masked in every
    # channel unless said otherwise.
    #   sbfg   R = 700 hPa Petterssen frontogenesis, K per 100 km per 3 h, -20.125 + 0.25 * code (255: 700 hPa within 25 hPa
    #          of the ground); G = 700 hPa theta, 240.2 + 0.4 K * code; B = 700 hPa height, 2402.5 + 5 m * code (offsets half a
    #          step off round numbers: no code equals a contour level)
    #   sbdef  R = axis of dilatation, counter-clockwise from EAST (earth-relative), 2 degrees * code (0-89; 255 where 700 hPa is
    #          near the ground or the deformation is below 6e-5 s-1, where no axis is drawn); G = total deformation, 2e-6 s-1 * code (the page draws the axes where it is in the top
    #          15 % of the area in view and >= 6e-5 s-1, as snowband.py did per region)
    #   sbepv  R = EPV* 750-500 hPa layer mean, PVU, -2.5 + 0.025 * code, only where the layer is saturated (RH >= 80 %,
    #          over ice below 0 C; 255 elsewhere); G = that layer's RH in % (50 = 50 or less, to 126; 127 = none) + 128 where theta_es falls
    #          with height (upright CI); B = 800-600 hPa frontogenesis as sbfg R
    #   sbdgz  R = max -omega in the saturated -12..-18 C layer, microbar/s, -20 + 0.5 * code (255: no saturated DGZ);
    #          G = DGZ depth, 5 hPa * code
    #   sbing  R = 60 x the number of ingredients (0-3; bilinear on the page, so the 1.5 / 2.5 edges are smooth as a
    #          contourf), G = which: 1 frontogenesis, 2 EPV*, 4 DGZ lift; B = 250 x the fraction of the cell with model
    #          snow at >= 25 dBZ composite reflectivity (RDPS: >= 1 mm in the hour)
    **{f: dict(enc=f) for f in ("sbfg", "sbdef", "sbepv", "sbdgz", "sbing")},
}
SB_FIELDS = ("sbfg", "sbdef", "sbepv", "sbdgz", "sbing")
ACC_METHOD = {
    "hrrr": dict(
        apcp="APCP",
        asnow="ASNOW, the HRRR's own snowfall (variable snow density)",
        afzra="FRZR, the model's accumulated freezing rain (liquid equivalent, not ice accretion)",
        aip="APCP accumulated in hours whose precipitation type is sleet (priority FZRA > IP > SN > RA; liquid equivalent)",
    ),
    "rrfs": dict(
        apcp="APCP",
        asnow="ASNOW, the RRFS's own snowfall",
        afzra="FRZR, the model's accumulated freezing rain (liquid equivalent)",
        aip="APCP accumulated in periods whose precipitation type is sleet (priority FZRA > IP > SN > RA; liquid equivalent; 3-hourly past 36 h)",
    ),
    "rdps": dict(
        apcp="Precip-Accum",
        asnow="10:1 from the RDPS's accumulated snow water equivalent (not a model snowfall)",
        afzra="FreezingRain-Accum (liquid equivalent)",
        aip="IcePellets-Accum (liquid equivalent)",
    ),
}
_FRAM = (
    "radial ice accretion on power lines and branches from the Freezing Rain Accumulation Model (Sanders and Barjenbruch "
    "2016): each hour's ice-to-liquid ratio from the freezing-rain rate, the 2 m wet-bulb temperature and the 10 m wind, "
    "times the model's freezing-rain liquid ({src}), x 0.394 for radial ice"
)
_SNOD = "the sum of the hourly INCREASES of the model's snow depth ({src}); melting and settling are not subtracted"
ACC_METHOD["hrrr"].update(
    afram=_FRAM.format(src="FRZR"),
    asnod=_SNOD.format(src="SNOD"),
    acobb=(
        "snowfall = column snow-to-liquid ratio x the snow-phase liquid (APCP where the surface type is snow, "
        "half where it is a rain-snow mix). Ratio: Cobb and Waldstreicher (2005), in the form of the NBM v4.1 Cobb method "
        "(one of the NBM's four snow-ratio methods) - layers saturated "
        "with respect to ice every 25 hPa from 925 to 300 hPa, weighted by upward motion and humidity, the 2011 "
        "temperature table, limited to 3-30; no melting or compaction"
    ),
    as10="10 x the snow-phase liquid (APCP where the surface type is snow, half where it is a rain-snow mix)",
)
ACC_METHOD["rrfs"].update(
    afram=_FRAM.format(src="FRZR; 3-hourly past 36 h"),
    asnod=_SNOD.format(src="SNOD; 3-hourly past 36 h"),
    as10=ACC_METHOD["hrrr"]["as10"],
)
ACC_METHOD["rdps"].update(afram=_FRAM.format(src="FreezingRain-Accum"), asnod=_SNOD.format(src="SnowDepth"))
# Freezing rain for a cycle whose files carry no FRZR (the operational RRFS from the 2026-10-15 12Z cycle): APCP in the
# periods whose precipitation type is freezing rain, accumulated like sleet; the FRAM ice uses that liquid. Checked per
# cycle, so FRZR is used again if it comes back; any other model still requires its FRZR.
FZRA_DERIVED = {
    "rrfs": dict(
        afzra="APCP accumulated in periods whose precipitation type is freezing rain (priority FZRA > IP > SN > RA; liquid equivalent; 3-hourly past 36 h; the operational RRFS writes no FRZR)",
        afram=_FRAM.format(src="APCP in freezing-rain periods; 3-hourly past 36 h"),
    ),
}
ORDER = [
    "ir",
    "refl",
    "ceil",
    "vis",
    "t2m",
    "td2m",
    "smoke",
    "wind80",
    "wind10",
    "gust",
    "sw",
    "mslp",
    "apcp",
    "asnow",
    "afzra",
    "aip",
    "uh25",
    "uh25s",
    "uh03",
    "uh03s",
    "uvv",
    "uvvs",
    "etop",
    "hpbl",
    "ua925",
    "ua250",
    "dt",
    "z500",
    "w500",
    "pvu",
    "pvl",
    *[f"{k}{int(t)}" for k in ("pv", "w") for t in UPR.THETAS],
    "afram",
    "acobb",
    "as10",
    "asnod",
    *SB_FIELDS,
]
PV_FIELDS = ("pvu", "pvl", *tuple(f"{k}{int(t)}" for k in ("pv", "w") for t in UPR.THETAS))
UA_FIELDS = ("ua925", "ua250", "dt", "z500", "w500", *PV_FIELDS)
UA_ON = os.environ.get("MV_UA", "0") == "1"  # models.yml sets MV_UA=1 (the switch for the upper-air fetch)
# models with upper air (MV_UA_MODELS); the RRFS prslev path (prs_url, UA_GRID, upper.UA_LEVS_RRFS) is off by default
UA_MODELS = set(os.environ.get("MV_UA_MODELS", "hrrr,rdps").split(","))


def ua_on(model):
    return UA_ON and model in UA_MODELS


# the grid each upper-air field is stored on: HRRR/RRFS 925/250 hPa on the 6 km half-resolution grid, the dynamic tropopause
# and 500 hPa (computed from fields smoothed to ~12 km) on the MSLP files' 12 km grid; RDPS native 10 km and 20 km
UA_GRID = {
    "hrrr": dict(ua925="lo", ua250="lo", dt="sub", z500="sub", w500="sub"),
    "rrfs": dict(ua925="lo", ua250="lo", dt="sub", z500="sub", w500="sub"),
    "rdps": dict(ua925="main", ua250="main", dt="sub", z500="sub", w500="sub"),
}
SUB = {"hrrr": 4, "rrfs": 4, "rdps": 2}
# Smoke is seasonal (off from 1 October to 31 March): a run whose UTC date falls outside (start month, start day,
# end month, end day) inclusive fetches, encodes and publishes no smoke for any model -
# HRRR/RRFS COLMD, RDPS via RAQDPS-FireWork (and its separate grid fetch) - and points.json carries no smoke series. The
# field is then absent from the manifest and models.html hides its button. It comes back on its own from Apr 1.
SMOKE_SEASON = (4, 1, 9, 30)


# Snow-band diagnostics: a season gate on the run's UTC date, inclusive (month, day, month, day): 1 October to
# 30 April. MV_SB_SEASON="MM-DD:MM-DD" overrides; MV_SB=force computes them out of season (testing,
# models.yml input "snowband"), MV_SB=off never. HRRR and RDPS only.
SB_MODELS = ("hrrr", "rdps")
SB_GRID = {"hrrr": "sub", "rdps": "main"}  # the 12 km "sub" grid = snowband.py's evaluation grid; RDPS 10 km as is


def sb_season():
    a, b = os.environ.get("MV_SB_SEASON", "10-01:04-30").split(":")
    return tuple(int(x) for x in a.split("-")) + tuple(int(x) for x in b.split("-"))


def sb_in_season(cyc_dt):
    m0, d0, m1, d1 = sb_season()
    md = (cyc_dt.month, cyc_dt.day)
    return (md >= (m0, d0) or md <= (m1, d1)) if (m0, d0) > (m1, d1) else ((m0, d0) <= md <= (m1, d1))


def sb_on(model, cyc_dt):
    mode = os.environ.get("MV_SB", "auto")
    return model in SB_MODELS and mode != "off" and (mode == "force" or sb_in_season(cyc_dt))


def sb_hour(lead):
    """every 3 h from +3 h (the HRRR's analysis hour has no usable vertical motion; snowband.py likewise)"""
    return lead >= 3 and lead % 3 == 0


def sb_window_text():
    m0, d0, m1, d1 = sb_season()
    mon = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return f"{d0} {mon[m0 - 1]} – {d1} {mon[m1 - 1]}"


def smoke_on(cyc_dt):
    m0, d0, m1, d1 = SMOKE_SEASON
    return (m0, d0) <= (cyc_dt.month, cyc_dt.day) <= (m1, d1)


# Half-resolution copies for zoomed-out views (for low-powered devices on slow connections): <fff>_lo.webp = every
# 2nd point of the native grid (plain decimation, so every encoding - 16-bit accumulations, precipitation-type flags,
# the 255 mask - keeps its meaning; grid j = 0, the southern row, is kept), listed as fields[f]["lo"] = "lo" with
# grids["lo"]. The page uses them while a half-resolution cell is <= 1.5 device pixels.
# 3 km models only: on the 10 km RDPS grid the page would never pick them. Measured on 2026100118 HRRR: lo/full bytes
# 0.28-0.36 per field, +33 % on the cycle.
LO = {"hrrr": 2, "rrfs": 2}
CONV_MAX = ("uh25", "uh03", "uvv")  # hourly-max fields: no f00, and a run swath each
HPBL_GRID = {"hrrr": "lo", "rrfs": "lo", "rdps": "sub"}
LO_SKIP = {
    *PV_FIELDS,
    *SB_FIELDS,
    "hpbl",
    "uh25",
    "uh25s",
    "uh03",
    "uh03s",
    "uvv",
    "uvvs",  # narrow tracks: every-2nd-point copies would drop them (tiny anyway)
    "mslp",
    "ua925",
    "ua250",
    "dt",
    "z500",
    "w500",
}


def decimate(arr, f):
    """stored image (row 0 north) -> every f-th grid point, keeping grid row 0 (the last stored row)"""
    return np.ascontiguousarray(arr[::-1][::f, ::f][::-1])


def q_lin(x, off, step):
    c = np.clip(np.rint((x - off) / step), 0, 254)
    c = np.where(np.isfinite(x), c, 255)
    return c.astype(np.uint8)


def q_log(x, lo, hi):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = 1 + np.rint((np.log(np.maximum(x, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 253)
    c = np.clip(c, 1, 254)
    c = np.where(x < lo, 0, c)
    c = np.where(np.isfinite(x), c, 255)
    return c.astype(np.uint8)


def q_linf(x):
    """Per-frame offset: 0.5 F steps from the floor of the frame minimum, 1 F (or coarser) if the frame is too wide."""
    fin = np.isfinite(x)
    if not fin.any():
        return np.full(x.shape, 255, np.uint8), [0.0, 0.5]
    lo, hi = float(np.floor(np.nanmin(x))), float(np.nanmax(x))
    step = 0.5
    while (hi - lo) / step > 254:
        step = {0.5: 1.0, 1.0: 1.5, 1.5: 2.0}.get(step, step * 2)
    return q_lin(x, lo, step), [lo, step]


# precipitation type code (G channel of "refl"): 0 no type flag, 1 rain, 2 snow, 3 freezing rain, 4 ice pellets (sleet),
# 5 rain-snow mix. HRRR/RRFS categorical flags can overlap; priority FZRA > IP > SN > RA, except that RA and SN together
# (and nothing else) is "mix". RDPS codes (PrecipType-Instant, measured 2026-09-30 against 2 m / 850 hPa temperature -
# the Datamart does NOT follow GRIB table 4.201 literally): 1 rain (T2 16 C), 2 rain-snow mix (T2 +2 C, T850 -4.5 C),
# 3 freezing rain (T2 -1 C, T850 0 C), 4 ice pellets (T2 -0.7 C), 5 snow (T2 -3 C), 6 no precipitation.
RDPS_PTYPE = {1: 1, 2: 5, 3: 3, 4: 4, 5: 2}
REFL_MIN = 5.0  # dBZ below this -> code 0 (no echo); also clears the type, so empty sky compresses
RATE_LO, RATE_HI = 0.1, 100.0  # RDPS precipitation rate, mm/h, log scale


def ptype_flags(ra, sn, fz, ip):
    t = np.zeros(ra.shape, np.uint8)
    t[ra > 0.5] = 1
    t[sn > 0.5] = 2
    t[(ra > 0.5) & (sn > 0.5)] = 5
    t[ip > 0.5] = 4
    t[fz > 0.5] = 3
    return t


def encode(name, x):
    """-> (uint8 image array, row 0 north; per-frame params or None)"""
    f = FIELDS[name]
    if f["enc"] == "uh":
        c = np.where(x >= 1.0, np.clip(np.rint(254 * np.sqrt(np.maximum(x, 0) / f["top"])), 1, 254), 0)
        return np.where(np.isfinite(x), c, 255).astype(np.uint8)[::-1], None
    if f["enc"] == "uvv":
        # below 2 m/s -> 0 (the map is blank there anyway): the 1-2 m/s of ordinary rising air filled the swaths with noise
        c = np.where(x >= 2.0, np.clip(np.rint(x / f["step"]), 8, 254), 0)
        return np.where(np.isfinite(x), c, 255).astype(np.uint8)[::-1], None
    if f["enc"] == "etop":  # x = echo top, kft; NaN = no echo
        c = np.where(np.isfinite(x), np.clip(np.rint(x), 0, 254), 255)
        return c.astype(np.uint8)[::-1], None
    if f["enc"] == "acc16":
        v = np.clip(np.rint(np.nan_to_num(x, nan=0.0) / f["unit"]), 0, 65535).astype(np.uint16)
        c = np.dstack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), np.zeros(v.shape, np.uint8)])
        return c[::-1], None
    if f["enc"] in ("pv1", "pv3", "pvth"):
        pv = q_lin(x[1], f["off"], f["step"])
        if f["enc"] == "pv1":
            return pv[::-1], None
        if f["enc"] == "pv3":
            c = np.dstack([pv, q_lin(x[2], *f["uv"]), q_lin(x[3], *f["uv"])])
        else:
            c = np.dstack([pv, q_lin(x[2], *f["p"]), np.zeros(pv.shape, np.uint8)])
        c[..., 1:][pv == 255] = 255
        return c[::-1], None
    if f["enc"] in SB_FIELDS:
        return encode_sb(f["enc"], x), None
    if f["enc"] in ("uvz", "dt", "vz"):
        z3 = np.zeros(x[1].shape, np.uint8)
        if f["enc"] == "uvz":  # ("uvz", u, v, height)
            zc, p = UPR.q_auto(x[3])
            c = np.dstack([q_lin(x[1], f["off"], f["step"]), q_lin(x[2], f["off"], f["step"]), zc])
        elif f["enc"] == "dt":  # ("dt", theta, u, v)
            c = np.dstack([q_lin(x[1], *f["th"]), q_lin(x[2], f["off"], f["step"]), q_lin(x[3], f["off"], f["step"])])
            p = None
        else:  # ("vz", absolute vorticity 1e-5 s-1, height)
            zc, p = UPR.q_auto(x[2])
            c = np.dstack([q_lin(x[1], *f["vo"]), zc, z3])
        return c[::-1], p
    if name == "mslp":  # ("mslp", hPa, high-terrain mask): R = pressure, G = 1 where
        c = np.dstack([q_lin(x[1], f["off"], f["step"]), x[2], np.zeros(x[2].shape, np.uint8)])  # surface p < 850 hPa
        p = None
    elif f["enc"] == "lin":
        c, p = q_lin(x, f["off"], f["step"]), None
    elif f["enc"] == "log":
        c, p = q_log(x, f["lo"], f["hi"]), None
    elif f["enc"] == "linf":
        c, p = q_linf(x)
    elif f["enc"] == "refl2" and x[0] == "refl":  # ("refl", dBZ, type)
        dbz, t = x[1], x[2]
        r = np.where(np.isfinite(dbz) & (dbz >= REFL_MIN), np.clip(np.rint(dbz / 0.5), 1, 254), 0).astype(np.uint8)
        c = np.dstack([r, np.where(r > 0, t, 0).astype(np.uint8), np.zeros(r.shape, np.uint8)])
        p = None
    elif f["enc"] == "refl2":  # ("rate", mm/h, type): RDPS
        rate, t = x[1], x[2]
        r = q_log(rate, RATE_LO, RATE_HI)
        r = np.where((r == 255) | (t == 0), 0, r).astype(np.uint8)
        c = np.dstack([r, np.where(r > 0, t, 0).astype(np.uint8), np.zeros(r.shape, np.uint8)])
        p = None
    else:  # lin2: (u, v)
        u, v = x
        c = np.dstack([q_lin(u, f["off"], f["step"]), q_lin(v, f["off"], f["step"]), np.zeros(u.shape, np.uint8)])
        p = None
    return c[::-1], p


def encode_sb(kind, x):
    """snow-band fields (see FIELDS) -> uint8 image, row 0 north"""

    def z(a):
        return np.zeros(a.shape, np.uint8)

    if kind == "sbfg":
        fg, th, zz = x[1:]
        # offsets half a step off the round numbers: no stored value then equals a contour level (theta every 2 K,
        # height every 30 m, frontogenesis 2/4/8/16), so a flat field cannot make a plateau AT a level - those draw
        # rectangular staircase lines round the plateau's cells
        c = np.dstack([q_lin(fg, -20.125, 0.25), q_lin(th, 240.2, 0.4), q_lin(zz, 2402.5, 5.0)])
    elif kind == "sbdef":
        ang, D = x[1:]
        ok = np.isfinite(ang) & np.isfinite(D)
        # the angle only where an axis can be drawn (D >= 6e-5 s-1, the page's floor): elsewhere it is noise that tripled
        # the file (96 -> ~30 KB/h on the HRRR, measured 2026-10-09)
        # and the deformation below that floor as 0: the page's threshold is max(85th percentile in view, 6e-5), which
        # values below the floor can never change (a quantile above the floor is the same with them clamped)
        big = ok & (np.nan_to_num(D) >= 6e-5)
        r = np.where(big, np.mod(np.rint(np.nan_to_num(ang) / 2.0), 90), 255).astype(np.uint8)  # 2-degree steps
        g = np.where(ok, np.where(big, np.clip(np.rint(np.nan_to_num(D) / 2e-6), 30, 254), 0), 255).astype(np.uint8)
        c = np.dstack([r, g, z(r)])
    elif kind == "sbepv":
        epv, rh, ci, fgl = x[1:]
        # RH below 50 % stored as 50 ("50 % or less"): only the 80 % outline and the hover use it; the dry noise below
        # cost a third of the RDPS file
        g = np.where(np.isfinite(rh), np.clip(np.rint(np.nan_to_num(rh)), 50, 126), 127) + 128 * (np.asarray(ci) > 0)
        c = np.dstack([q_lin(epv, -2.5, 0.025), g.astype(np.uint8), q_lin(fgl, -20.125, 0.25)])
    elif kind == "sbdgz":
        lift, depth = x[1:]
        g = np.where(np.isfinite(depth), np.clip(np.rint(np.nan_to_num(depth) / 5.0), 0, 254), 255).astype(np.uint8)
        c = np.dstack([q_lin(lift, -20.0, 0.5), g, z(g)])
    else:  # sbing
        n, flags, snow = x[1:]
        c = np.dstack(
            [
                (60 * np.asarray(n)).astype(np.uint8),
                np.asarray(flags).astype(np.uint8),
                np.clip(np.rint(250 * np.nan_to_num(snow)), 0, 250).astype(np.uint8),
            ]
        )
    return c[::-1]


def webp(arr):
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, "WEBP", lossless=True, quality=100, method=4)
    return b.getvalue()


# ── models ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def leads_of(model, cyc):
    """Forecast hours published. RRFS: hourly to 36 h, 3-hourly to 84 h (storage); others hourly."""
    n = MODELS[model]["length"](cyc)
    if model == "rrfs":
        return list(range(37)) + list(range(39, n + 1, 3))
    return list(range(n + 1))


MODELS = {
    # HRRR: only the runs that go out to 48 hours - 00/06/12/18Z
    "hrrr": dict(
        label="HRRR",
        every=6,
        length=lambda c: 48,
        lag_h=(1, 30),
        source="NOAA HRRR v4 (AWS open data, noaa-hrrr-bdp-pds)",
    ),
    "rrfs": dict(
        label="RRFS",
        every=6,
        length=lambda c: 84,
        lag_h=(3, 40),
        source="NOAA RRFS v1.0 (AWS open data, noaa-rrfs-ops-pds)",
    ),
    "rdps": dict(
        label="RDPS",
        every=6,
        length=lambda c: 84,
        lag_h=(3, 40),
        source="ECCC RDPS 10 km and RAQDPS-FireWork (MSC Datamart)",
    ),
}
# simulated reflectivity 1 km above ground (what a radar's lowest scans see and what the precipitation type describes; the
# composite REFC would paint elevated echo and bright bands in ptype colours) + the instantaneous categorical types
PT = {
    "refd": r"^REFD:1000 m above ground:",
    "crain": r"^CRAIN:surface:(anl|\d+ hour fcst)",
    "csnow": r"^CSNOW:surface:(anl|\d+ hour fcst)",
    "cfrzr": r"^CFRZR:surface:(anl|\d+ hour fcst)",
    "cicep": r"^CICEP:surface:(anl|\d+ hour fcst)",
}
# run-total accumulations; the HRRR writes 24 and 48 h as "0-1 day acc" / "0-2 day acc"
ACC = {
    "apcp": r"^APCP:surface:0-\d+ (hour|day) acc",
    "asnow": r"^ASNOW:surface:0-\d+ (hour|day) acc",
    "frzr": r"^FRZR:surface:0-\d+ (hour|day) acc",
}
# 10 m wind and the instantaneous surface gust; the anchors keep out the hourly-max WIND / MAXUW / MAXVW
# lines that share the 10 m level ("5-6 hour max fcst"). Same names in HRRR wrfsfc and RRFS 2dfld (checked 2026-10-04).
CONV = {
    "uh25": r"^MXUPHL:5000-2000 m above ground:",
    "uh03": r"^MXUPHL:3000-0 m above ground:",
    "uvv": r"^MAXUVV:100-1000 mb",
    "etop": r"^RETOP:",
    "hpbl": r"^HPBL:surface:",
}
WG = {
    "snod": r"^SNOD:surface:(anl|\d+ hour fcst):?$",
    "gust": r"^GUST:surface:(anl|\d+ hour fcst):?$",
    "u10": r"^UGRD:10 m above ground:(anl|\d+ hour fcst):?$",
    "v10": r"^VGRD:10 m above ground:(anl|\d+ hour fcst):?$",
}
IDX_WANT = {  # field -> regex on "VAR:LEVEL:TIME[:extra]" of a wgrib2 .idx line
    # MSLP: HRRR MSLMA (MAPS reduction, the only MSLP in wrfsfc); RRFS MSLET (NCEP's chart reduction; no PRMSL in 2dfld)
    "hrrr": {
        "mslp": r"^MSLMA:mean sea level:",
        "psfc": r"^PRES:surface:",
        "ir": r"^SBT124:top of atmosphere:",
        "ceilh": r"^HGT:cloud ceiling:",
        "zsfc": r"^HGT:surface:",
        "vis": r"^VIS:surface:",
        "t2m": r"^TMP:2 m above ground:",
        "td2m": r"^DPT:2 m above ground:",
        "smoke": r"^COLMD:entire atmosphere",
        "u80": r"^UGRD:80 m above ground:",
        "v80": r"^VGRD:80 m above ground:",
        "sw": r"^DSWRF:surface:(anl|\d+ hour fcst):?$",
        **WG,
        **PT,
        **ACC,
        **CONV,
    },
    "rrfs": {
        "mslp": r"^MSLET:mean sea level:",
        "psfc": r"^PRES:surface:",
        "ir": r"^SBTA1613:top of atmosphere:",
        "ceilh": r"^HGT:cloud ceiling:",
        "zsfc": r"^HGT:surface:",
        "vis": r"^VIS:surface:",
        "t2m": r"^TMP:2 m above ground:",
        "td2m": r"^DPT:2 m above ground:",
        "smoke": r"^COLMD:entire atmosphere.*Particulate organic matter dry",
        "u80": r"^UGRD:80 m above ground:",
        "v80": r"^VGRD:80 m above ground:",
        "sw": r"^DSWRF:surface:(anl|\d+ hour fcst):?$",
        **WG,
        **PT,
        **ACC,
        **CONV,
    },
}
RDPS_VARS = {
    "ir": "UpwardLongwaveRadiationFlux_NTAtm",
    "t2m": "AirTemp_AGL-2m",
    "td2m": "DewPoint_AGL-2m",
    "u80": "WindU_AGL-80m",
    "v80": "WindV_AGL-80m",
    "u10": "WindU_AGL-10m",
    "v10": "WindV_AGL-10m",
    "gust": "WindGust_AGL-10m",
    "swacc": "DownwardShortwaveRadiationFlux-Accum_Sfc",
    "prate": "PrecipRate_Sfc",
    "ptype": "PrecipType-Instant_Sfc",
    "mslp": "Pressure_MSL",
    "psfc": "Pressure_Sfc",
    "apcp": "Precip-Accum_Sfc",
    "swe": "Snow-Accum_Sfc",
    "frzr": "FreezingRain-Accum_Sfc",
    "ipacc": "IcePellets-Accum_Sfc",
    "hpbl": "PlanetaryBoundaryLayerHeight_Sfc",
    "snod": "SnowDepth_Sfc",
}
RDPS_BOX = (-172.0, -45.0, 17.0, 80.0)  # crop of the RDPS grid (lon0, lon1, lat0, lat1): North America


def file_url(model, date, cyc, lead):
    if model == "hrrr":
        return f"{sb.hrrr_base(date, cyc)}.wrfsfcf{lead:02d}.grib2"
    if model == "rrfs":
        return f"{sb.rrfs_base(date, cyc)}.2dfld.3km.f{lead:03d}.conus.grib2"
    raise ValueError(model)


def prs_url(model, date, cyc, lead):
    """The isobaric-level file (upper air): HRRR wrfprs, RRFS prslev."""
    if model == "hrrr":
        return f"{sb.hrrr_base(date, cyc)}.wrfprsf{lead:02d}.grib2"
    return f"{sb.rrfs_base(date, cyc)}.prslev.3km.f{lead:03d}.conus.grib2"


def ua_want(model="hrrr"):
    """.idx regexes of the upper-air messages (HRRR wrfprs / RRFS prslev share the names): T, u, v on UA_LEVS for the
    dynamic tropopause and 500 hPa vorticity, and the 925 / 250 / 500 hPa heights and 925 hPa wind."""
    w = {}
    for p in UPR.levs_of(model):
        for v in ("TMP", "UGRD", "VGRD"):
            w[f"ua:{v}:{p}"] = rf"^{v}:{p} mb:(anl|\d+ hour fcst):?$"
    for v, p in (("UGRD", 925), ("VGRD", 925), ("HGT", 925), ("HGT", 250), ("HGT", 500)):
        w[f"ua:{v}:{p}"] = rf"^{v}:{p} mb:(anl|\d+ hour fcst):?$"
    return w


def rdps_ua_jobs(date, cyc, lead):
    j = {
        f"ua:T:{p}": rdps_url(date, cyc, lead, f"AirTemp_IsbL-{p:04d}")
        for p in sorted(set(UPR.RDPS_T_LEVS) | set(UPR.RDPS_PV_LEVS))
    }
    for p in sorted(set(UPR.RDPS_UV_LEVS) | set(UPR.RDPS_PV_LEVS) | {925}):
        j[f"ua:U:{p}"] = rdps_url(date, cyc, lead, f"WindU_IsbL-{p:04d}")
        j[f"ua:V:{p}"] = rdps_url(date, cyc, lead, f"WindV_IsbL-{p:04d}")
    for p in (925, 250, 500):
        j[f"ua:Z:{p}"] = rdps_url(date, cyc, lead, f"GeopotentialHeight_IsbL-{p:04d}")
    j["ua:AV:500"] = rdps_url(date, cyc, lead, "AbsoluteVorticity_IsbL-0500")
    j["ua:PDT"] = rdps_url(date, cyc, lead, "Pressure_PVU-2")
    return j


def sb_want_lcc():
    """.idx regexes of the snow-band columns in HRRR wrfprs (snowband.py read_levels): T, u, v, omega, q on sb.LEVS
    (950-450 hPa every 25) and the 700 hPa height. Keys shared with the upper-air (ua:) and Cobb (cb:VVEL) sets, so a
    message wanted by two is fetched once."""
    w = {}
    for p in sb.LEVS:
        for v, k in (
            ("TMP", "ua:TMP"),
            ("UGRD", "ua:UGRD"),
            ("VGRD", "ua:VGRD"),
            ("VVEL", "cb:VVEL"),
            ("SPFH", "sb:SPFH"),
        ):
            w[f"{k}:{p}"] = rf"^{v}:{p} mb:(anl|\d+ hour fcst):?$"
    w["ua:HGT:700"] = r"^HGT:700 mb:(anl|\d+ hour fcst):?$"
    return w


def rdps_sb_jobs(date, cyc, lead):
    """the RDPS snow-band inputs (snowband.py read_levels / read_surface): T, u, v, q on sb.LEVS_RDPS, omega at 850/700/
    500 hPa, the 700 hPa height and the 1-h precipitation (keys shared with the upper-air set)"""
    j = {}
    for p in sb.LEVS_RDPS:
        j[f"ua:T:{p}"] = rdps_url(date, cyc, lead, f"AirTemp_IsbL-{p:04d}")
        j[f"ua:U:{p}"] = rdps_url(date, cyc, lead, f"WindU_IsbL-{p:04d}")
        j[f"ua:V:{p}"] = rdps_url(date, cyc, lead, f"WindV_IsbL-{p:04d}")
        j[f"sb:Q:{p}"] = rdps_url(date, cyc, lead, f"SpecificHumidity_IsbL-{p:04d}")
    for p in sb.RDPS_W_LEVS:
        j[f"sb:W:{p}"] = rdps_url(date, cyc, lead, f"VerticalVelocity_IsbL-{p:04d}")
    j["ua:Z:700"] = rdps_url(date, cyc, lead, "GeopotentialHeight_IsbL-0700")
    j["sb:PA1"] = rdps_url(date, cyc, lead, "Precip-Accum1h_Sfc")
    return j


def rdps_url(date, cyc, lead, var):
    return (
        f"https://dd.weather.gc.ca/{date}/WXO-DD/model_rdps/10km/{cyc:02d}/{lead:03d}/"
        f"{date}T{cyc:02d}Z_MSC_RDPS_{var}_RLatLon0.09_PT{lead:03d}H.grib2"
    )


def raq_url(date, cyc, lead):
    return (
        f"https://dd.weather.gc.ca/{date}/WXO-DD/model_raqdps/10km/grib2/{cyc:02d}/{lead:03d}/"
        f"{date}T{cyc:02d}Z_MSC_RAQDPS_PM2.5-WildfireSmokePlume_EAtm_RLatLon0.09_PT{lead:03d}H.grib2"
    )


def raq_for(cycle_dt):
    """RAQDPS runs 00/12Z only: the newest run at or before the RDPS cycle, and the lead offset to add."""
    r = cycle_dt.replace(hour=0 if cycle_dt.hour < 12 else 12)
    return r, int((cycle_dt - r).total_seconds() // 3600)


# ── transport ───────────────────────────────────────────────────────────────────────────────────────────────────────
_FETCH = {}


def fetcher(model):
    if model not in _FETCH:
        _FETCH[model] = {
            "hrrr": lambda: sb.Fetcher("S3 HRRR", 16),
            "rrfs": lambda: sb.Fetcher("S3 RRFS", 16),
            "rdps": lambda: sb.Fetcher("MSC Datamart", 12, interval=0.02),
        }[model]()
    return _FETCH[model]


def head_ok(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def idx_lines(f, url):
    body = f.get(url + ".idx", kind="idx").decode()
    rows = [l.split(":") for l in body.splitlines() if l.strip()]
    out = []
    for i, p in enumerate(rows):
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else -1
        out.append((int(p[1]), end, ":".join(p[3:])))
    return out


def fetch_idx_fields(model, url, want):
    """{key: GRIB message bytes} for the first .idx line matching each regex, one range per message in parallel."""
    f = fetcher(model)
    lines = idx_lines(f, url)
    sel = {}
    for key, rx in want.items():
        for s, e, desc in lines:
            if re.search(rx, desc):
                sel[key] = (s, e)
                break
    missing = set(want) - set(sel)
    if missing:
        raise sb.Missing(f"{url}: no {sorted(missing)}")
    keys = sorted(sel, key=lambda k: sel[k][0])
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(keys, ex.map(lambda k: f.get(url, sel[k]), keys)))


# ── grids ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def grid_desc(buf, crop=None):
    """The grid as the page needs it to map screen -> grid index, from a GRIB message."""
    import eccodes

    g = eccodes.codes_new_from_message(buf)
    try:
        t = eccodes.codes_get(g, "gridType")

        def gv(k):
            return eccodes.codes_get(g, k)

        if t == "lambert":
            d = dict(
                type="lcc",
                nx=gv("Nx"),
                ny=gv("Ny"),
                lat1=gv("Latin1InDegrees"),
                lat2=gv("Latin2InDegrees"),
                lon0=((gv("LoVInDegrees") + 180) % 360) - 180,
                la1=gv("latitudeOfFirstGridPointInDegrees"),
                lo1=((gv("longitudeOfFirstGridPointInDegrees") + 180) % 360) - 180,
                dx=gv("DxInMetres"),
                dy=gv("DyInMetres"),
                R=6371229.0,
            )
        elif t == "rotated_ll":
            d = dict(
                type="rotll",
                nx=gv("Ni"),
                ny=gv("Nj"),
                splat=gv("latitudeOfSouthernPoleInDegrees"),
                splon=gv("longitudeOfSouthernPoleInDegrees"),
                la1=gv("latitudeOfFirstGridPointInDegrees"),
                lo1=gv("longitudeOfFirstGridPointInDegrees"),
                di=gv("iDirectionIncrementInDegrees"),
                dj=gv("jDirectionIncrementInDegrees"),
            )
            if gv("iScansNegatively") or not gv("jScansPositively"):
                raise ValueError("unexpected scan order")
        else:
            raise ValueError(t)
        if crop:
            (j0, j1), (i0, i1) = crop
            d.update(i0=int(i0), j0=int(j0), nx=int(i1 - i0), ny=int(j1 - j0))
        return d
    finally:
        eccodes.codes_release(g)


def grid_latlon(buf):
    _v, g = sb.decode(buf, want_grid=True)
    return g["lat"], g["lon"], g["grid_rel"]


def alpha_of(lat, lon):
    """Angle of the grid's +i axis counter-clockwise from east, per point (grid-relative -> earth-relative winds)."""
    dlon = (np.gradient(lon, axis=1) + 180) % 360 - 180
    return np.arctan2(np.gradient(lat, axis=1), dlon * np.cos(np.deg2rad(lat))).astype("float32")


def to_earth(u, v, alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    return u * ca - v * sa, u * sa + v * ca


# ── per-hour work (runs in a worker process) ───────────────────────────────────────────────────────────────────────
_CTX = {}


def _init(ctx):
    _CTX.update(ctx)


def dec(buf, crop=None):
    v = sb.decode(buf)
    if crop is not None:
        (j0, j1), (i0, i1) = crop
        v = v[j0:j1, i0:i1]
    return v


def f_of_k(k):
    return (k - 273.15) * 1.8 + 32.0 if np.nanmean(k) > 150 else k * 1.8 + 32.0


def to_grid(u, v, alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    return u * ca + v * sa, -u * sa + v * ca


def dec_units(buf, crop=None):
    """values and the GRIB units string"""
    import eccodes

    g = eccodes.codes_new_from_message(buf)
    try:
        u = eccodes.codes_get(g, "units")
    except Exception:  # noqa: BLE001
        u = ""
    finally:
        eccodes.codes_release(g)
    return dec(buf, crop), u


def height_m(buf, crop, lev):
    """geopotential height in metres whatever the file's unit (dam or gpm): the factor that puts the median where the
    level's heights live"""
    z, u = dec_units(buf, crop)
    if u.lower() in ("dam", "decametre", "decameters"):
        return z * 10.0
    med = float(np.nanmedian(z))
    expect = {925: 750.0, 850: 1450.0, 700: 3000.0, 500: 5600.0, 250: 10300.0}[lev]
    return z * 10.0 if abs(med * 10 - expect) < abs(med - expect) else z


def pv_products(pv, th, U, V, LV, earth, ps_sub):
    """the PV charts from Ertel PV on the pressure levels (upper.py); U, V grid-relative, earth(u, v) -> earth-relative"""
    out = {}

    def f32(a):
        return np.asarray(a, np.float32)

    if all(p in LV for p in UPR.PV_UP):
        out["pvu"] = ("pv1", f32(UPR.layer_mean(pv, LV, UPR.PV_UP)))
    if all(p in LV for p in UPR.PV_LO):
        lo = UPR.layer_mean(pv, LV, UPR.PV_LO)
        k8 = LV.index(850)
        u8, v8 = earth(U[k8], V[k8])
        if ps_sub is not None:
            bad = ps_sub < 85000.0
            lo, u8, v8 = (np.where(bad, np.nan, a) for a in (lo, u8, v8))
        out["pvl"] = ("pv3", f32(lo), f32(u8), f32(v8))
    for t in UPR.THETAS:
        o = UPR.on_theta(pv, th, U, V, LV, t, pmax=700)
        ue, ve = earth(o["u"], o["v"])
        out[f"pv{int(t)}"] = ("pvth", f32(o["pv"]), f32(o["p"]))
        out[f"w{int(t)}"] = (f32(ue), f32(ve))
    return out


def ua_lcc(blobs):
    """HRRR / RRFS upper air (upper.py) -> {field: tuple for encode()}"""
    out = {}
    F = _CTX["sub"]
    al, rel = _CTX.get("alpha"), _CTX.get("grid_rel")

    def earth(u, v, a):
        return to_earth(u, v, a) if rel and a is not None else (u, v)

    ps = dec(blobs["psfc"]) if "psfc" in blobs else None
    for lev, name in ((925, "ua925"), (250, "ua250")):
        ks = [f"ua:UGRD:{lev}", f"ua:VGRD:{lev}", f"ua:HGT:{lev}"]
        if not all(k in blobs for k in ks):
            continue
        u, v, z = (dec(blobs[k])[::2, ::2] for k in ks)  # the 6 km "lo" grid (every 2nd point)
        if lev == 925 and ps is not None:  # below ground: the model extrapolated
            bad = ps[::2, ::2] < 92500.0
            u, v, z = (np.where(bad, np.nan, a) for a in (u, v, z))
        ue, ve = earth(u, v, al[::2, ::2] if al is not None else None)
        out[name] = ("uvz", ue, ve, z)
    LV = _CTX.get("ua_levs", UPR.UA_LEVS)
    if all(f"ua:{v}:{p}" in blobs for v in ("TMP", "UGRD", "VGRD") for p in LV):

        def st(v):
            return np.stack([UPR.smooth_dec(dec(blobs[f"ua:{v}:{p}"]), F) for p in LV]).astype(np.float64)

        T, U, V = st("TMP"), st("UGRD"), st("VGRD")
        asub = al[::F, ::F] if al is not None else None
        if not rel and asub is not None:  # the PV formula wants grid-relative winds
            U, V = to_grid(U, V, asub[None])
        lat = _CTX["lat_sub"]
        m = UPR.lcc_mapfac(lat, _CTX["lat1"], _CTX["lat2"])
        pv, th, absv = UPR.ertel_pv(T, U, V, LV, lat, m, _CTX["dx"] * F)
        n7 = sum(p <= 700 for p in LV)  # the DT search stays on 100-700 hPa
        d = UPR.dt_downward(pv[:n7], th[:n7], U[:n7], V[:n7], LV[:n7])
        try:
            out.update(
                pv_products(
                    pv,
                    th,
                    U,
                    V,
                    LV,
                    lambda u, v: to_earth(u, v, asub) if asub is not None else (u, v),
                    ps[::F, ::F] if ps is not None else None,
                )
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"::warning::PV charts failed ({e!r})", flush=True)
        ue, ve = to_earth(d["u"], d["v"], asub) if asub is not None else (d["u"], d["v"])
        out["dt"] = ("dt", d["theta"].astype(np.float32), ue.astype(np.float32), ve.astype(np.float32))
        if "ua:HGT:500" in blobs:
            av = UPR._121(UPR._121(absv[LV.index(500)]))
            out["z500"] = ("vz", (av * 1e5).astype(np.float32), UPR.smooth_dec(dec(blobs["ua:HGT:500"]), F))
            k5 = LV.index(500)
            u5, v5 = to_earth(U[k5], V[k5], asub) if asub is not None else (U[k5], V[k5])
            out["w500"] = (np.asarray(u5, np.float32), np.asarray(v5, np.float32))
    return out


def ua_rdps(blobs, crop):
    """RDPS upper air: winds and heights on its own grid, the dynamic tropopause at the model's own 2-PVU pressure"""
    out = {}
    F = _CTX["sub"]
    al, rel = _CTX.get("alpha"), _CTX.get("grid_rel")

    def earth(u, v, a):
        return to_earth(u, v, a) if rel and a is not None else (u, v)

    ps = None
    if "psfc" in blobs:
        ps = dec(blobs["psfc"], crop)
        ps = ps if np.nanmean(ps) > 5000 else ps * 100.0
    for lev, name in ((925, "ua925"), (250, "ua250")):
        ks = [f"ua:U:{lev}", f"ua:V:{lev}", f"ua:Z:{lev}"]
        if not all(k in blobs for k in ks):
            continue
        u, v = dec(blobs[ks[0]], crop), dec(blobs[ks[1]], crop)
        z = height_m(blobs[ks[2]], crop, lev)
        if lev == 925 and ps is not None:
            bad = ps < 92500.0
            u, v, z = (np.where(bad, np.nan, a) for a in (u, v, z))
        ue, ve = earth(u, v, al)
        out[name] = ("uvz", ue, ve, z)
    need = [f"ua:T:{p}" for p in UPR.RDPS_T_LEVS] + [f"ua:{c}:{p}" for c in "UV" for p in UPR.RDPS_UV_LEVS] + ["ua:PDT"]
    if all(k in blobs for k in need):
        pd = dec(blobs["ua:PDT"], crop).astype(np.float64)
        pd = pd / 100.0 if np.nanmedian(pd) > 2000 else pd  # Pa -> hPa
        T = np.stack([dec(blobs[f"ua:T:{p}"], crop) for p in UPR.RDPS_T_LEVS]).astype(np.float64)
        T = T + 273.15 if np.nanmean(T) < 150 else T
        lv = np.asarray(UPR.RDPS_T_LEVS, np.float64)
        th = T * (1000.0 / lv[:, None, None]) ** UPR.KAPPA
        U = np.stack([dec(blobs[f"ua:U:{p}"], crop) for p in UPR.RDPS_UV_LEVS]).astype(np.float64)
        V = np.stack([dec(blobs[f"ua:V:{p}"], crop) for p in UPR.RDPS_UV_LEVS]).astype(np.float64)
        pq = np.where(np.isfinite(pd), pd, 100.0)
        tdt = UPR.interp_lnp(th, UPR.RDPS_T_LEVS, pq)
        udt, vdt = UPR.interp_lnp(U, UPR.RDPS_UV_LEVS, pq), UPR.interp_lnp(V, UPR.RDPS_UV_LEVS, pq)
        udt, vdt = earth(udt, vdt, al)
        out["dt"] = ("dt", *tuple(UPR.smooth_dec(a, F) for a in (tdt, udt, vdt)))
    PL = UPR.RDPS_PV_LEVS
    if all(f"ua:{c}:{p}" in blobs for c in "TUV" for p in PL) and _CTX.get("rd_rlat") is not None:
        try:

            def st(c):
                return np.stack([UPR.smooth_dec(dec(blobs[f"ua:{c}:{p}"], crop), F) for p in PL]).astype(np.float64)

            T, U, V = st("T"), st("U"), st("V")
            T = T + 273.15 if np.nanmean(T) < 150 else T
            asub = al[::F, ::F] if al is not None else None
            if not rel and asub is not None:
                U, V = to_grid(U, V, asub[None])
            g = _CTX["rd_grid"]
            pv, th = UPR.ertel_pv_rot(
                T,
                U,
                V,
                PL,
                _CTX["rd_rlat"][:, None] * np.ones((1, T.shape[2])),
                _CTX["lat_sub"],
                g["di"] * F,
                g["dj"] * F,
            )
            out.update(
                pv_products(
                    pv,
                    th,
                    U,
                    V,
                    PL,
                    lambda u, v: to_earth(u, v, asub) if asub is not None else (u, v),
                    ps[::F, ::F] if ps is not None else None,
                )
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"::warning::RDPS PV charts failed ({e!r})", flush=True)
    if "ua:Z:500" in blobs and "ua:AV:500" in blobs:
        av = dec(blobs["ua:AV:500"], crop)
        av = av * 1e5 if np.nanmedian(np.abs(av)) < 0.01 else av  # s-1 -> 1e-5 s-1
        out["z500"] = ("vz", UPR.smooth_dec(av, F), UPR.smooth_dec(height_m(blobs["ua:Z:500"], crop, 500), F))
        if "ua:U:500" in blobs and "ua:V:500" in blobs:
            u5, v5 = earth(dec(blobs["ua:U:500"], crop), dec(blobs["ua:V:500"], crop), al)
            out["w500"] = (UPR.smooth_dec(u5, F), UPR.smooth_dec(v5, F))
    return out


def sb_grid(model):
    """snowband.Grid over the whole viewer grid (no crop), built once per worker"""
    if "_sbgrid" not in _CTX:
        _CTX["_sbgrid"] = sb.Grid(
            dict(lat=_CTX["sb_lat"], lon=_CTX["sb_lon"], grid_rel=bool(_CTX.get("grid_rel"))),
            sb.SRC_DX[model],
            box=None,
        )
    return _CTX["_sbgrid"]


def sb_fields(model, blobs, crop):
    """The snow-band diagnostics of one hour -> {field: tuple for encode()}, computed by snowband.py itself
    (smooth_diagnose, ingredients); this only feeds it the fields and re-expresses the dilatation axis earth-relative."""
    gr = sb_grid(model)
    if model == "rdps":
        levs = list(sb.LEVS_RDPS)
        need = (
            [f"ua:{c}:{p}" for c in "TUV" for p in levs]
            + [f"sb:Q:{p}" for p in levs]
            + [f"sb:W:{p}" for p in sb.RDPS_W_LEVS]
            + ["ua:Z:700", "psfc"]
        )
        miss = [k for k in need if k not in blobs]
        if miss:
            print(f"  snow-band: missing {miss[:6]}", flush=True)
            return {}
        W3 = {p: dec(blobs[f"sb:W:{p}"], crop) for p in sb.RDPS_W_LEVS}
        key = dict(T="ua:T", u="ua:U", v="ua:V", q="sb:Q")

        def level(name, k):
            p = levs[k]
            if name == "w":
                return sb.rdps_omega(W3, p)
            a = dec(blobs[f"{key[name]}:{p}"], crop)
            return sb.rdps_kelvin(a) if name == "T" else a

        z7 = sb.rdps_z_m(dec(blobs["ua:Z:700"], crop))
        ps = dec(blobs["psfc"], crop)
        ps = ps if np.nanmean(ps) > 5000 else ps * 100.0
    else:
        levs = list(sb.LEVS)
        need = [f"{k}:{p}" for k in ("ua:TMP", "ua:UGRD", "ua:VGRD", "cb:VVEL", "sb:SPFH") for p in levs] + [
            "ua:HGT:700",
            "psfc",
        ]
        miss = [k for k in need if k not in blobs]
        if miss:
            print(f"  snow-band: missing {miss[:6]}", flush=True)
            return {}
        key = dict(T="ua:TMP", u="ua:UGRD", v="ua:VGRD", w="cb:VVEL", q="sb:SPFH")

        def level(name, k):
            return dec(blobs[f"{key[name]}:{levs[k]}"])

        z7 = height_m(blobs["ua:HGT:700"], None, 700)
        ps = dec(blobs["psfc"])
    d = sb.smooth_diagnose(gr, levs, level, z7, ps)

    def f32(a):
        return np.asarray(a, np.float32)

    # the axis of dilatation: snowband.py gives its angle from the grid's +i axis; + the angle of that axis from east
    ang = np.degrees(d["alpha"] + gr.alpha[:: gr.step, :: gr.step])
    D = np.where(d["ground700"], np.nan, d["D"])
    ang = np.where(np.isfinite(D), ang, np.nan)
    n, fl, el, ll = sb.ingredients(d)
    if model == "rdps":
        if "sb:PA1" in blobs and "ptype" in blobs:
            snow = (
                (np.rint(np.nan_to_num(dec(blobs["ptype"], crop), nan=6)) == 5)
                & (np.nan_to_num(dec(blobs["sb:PA1"], crop)) >= 1.0)
            ).astype(np.float32)
        else:
            snow = np.full(n.shape, np.nan, np.float32)
    else:
        if "sb:REFC" in blobs and "csnow" in blobs:
            hit = ((dec(blobs["csnow"]) >= 0.5) & (np.nan_to_num(dec(blobs["sb:REFC"])) >= 25.0)).astype(np.float32)
            snow = UPR.smooth_dec(hit, gr.step)  # fraction of the 12 km cell (block centred on it)
        else:
            snow = np.full(n.shape, np.nan, np.float32)
    if snow.shape != n.shape:
        raise ValueError(f"snow mask {snow.shape} vs diagnostics {n.shape}")
    epv = np.where(d["sat"], d["epv"], np.nan)
    return {
        "sbfg": ("sbfg", f32(d["fg"]), f32(d["th7"]), f32(d["z7"])),
        "sbdef": ("sbdef", f32(ang), f32(D)),
        "sbepv": ("sbepv", f32(epv), f32(d["rh_l"]), np.asarray(d["ci"], bool), f32(d["fg_layer"])),
        "sbdgz": ("sbdgz", f32(d["dgz_lift"]), f32(d["dgz_depth"])),
        "sbing": ("sbing", n.astype(np.uint8), (fl + 2 * el + 4 * ll).astype(np.uint8), f32(snow)),
    }


def winter_hour(blobs, crop, pt, apcp, phys):
    """This hour's inputs to the winter accumulations (winter.py), for the sequential pass: freezing-rain liquid so far
    (mm), 2 m wet-bulb (C) and 10 m wind (kt) where freezing rain has fallen or is falling, snow depth (in), and on the
    HRRR the Cobb and Kuchera snow ratios where the surface type is snow or mix. Mostly-empty fields: compressed npz."""
    out = {}
    shape = phys["t2m"].shape if "t2m" in phys else None
    fz = (
        np.maximum(dec(blobs["frzr"], crop), 0).astype(np.float32)
        if "frzr" in blobs
        else (np.zeros(shape, np.float32) if shape else None)
    )
    if fz is not None:
        if not _CTX.get("fzra_derived"):  # derived freezing rain reaches FRAM through the sequential pass instead
            out["frzr"] = fz
        where = (fz > 0) | ((pt == 3) if pt is not None else False)
        if where.any() and "t2m" in phys and "td2m" in phys and "wind10" in phys:
            tc, tdc = (phys["t2m"] - 32) / 1.8, (phys["td2m"] - 32) / 1.8
            tw = WIN.wetbulb_stull(tc, WIN.rh_from_td(tc, tdc))
            u, v = phys["wind10"]
            out["tw"] = np.where(where, tw, np.nan).astype(np.float16)
            out["ws"] = np.where(where, np.hypot(u, v) * 1.943844, np.nan).astype(np.float16)
    if "snod" in blobs:
        sd, un = dec_units(blobs["snod"], crop)
        out["snod"] = (np.maximum(np.nan_to_num(sd), 0) * (39.3701 if un.lower() in ("m", "") else 0.393701)).astype(
            np.float32
        )
    if pt is not None:
        sf = np.where(pt == 2, 2, np.where(pt == 5, 1, 0)).astype(np.uint8)  # snow 1.0, rain-snow mix 0.5 (x 2)
        out["sf"] = sf
        if "asnow" in phys:
            out["asnow"] = phys["asnow"].astype(np.float32)
        if (sf > 0).any() and all(f"cb:RH:{p}" in blobs for p in WIN.COBB_LEVS) and "psfc" in blobs:
            T = np.stack([dec(blobs[f"ua:TMP:{p}"]) for p in WIN.COBB_LEVS])
            RH = np.stack([dec(blobs[f"cb:RH:{p}"]) for p in WIN.COBB_LEVS])
            OM = np.stack([dec(blobs[f"cb:VVEL:{p}"]) for p in WIN.COBB_LEVS])
            ps = dec(blobs["psfc"])
            slr, none = WIN.cobb_slr(T, RH, OM, WIN.COBB_LEVS, ps)
            t2k = (phys["t2m"] - 32) / 1.8 + 273.15
            out["slr"] = np.where(sf > 0, slr, np.nan).astype(np.float16)
            out["slrk"] = np.where(sf > 0, WIN.kuchera(T, WIN.COBB_LEVS, ps, t2k), np.nan).astype(np.float16)
            out["slr_none"] = np.array([int((none & (sf > 0)).sum()), int((sf > 0).sum())])
    return out


def process_hour(model, lead, blobs, outdir):
    """Decode, convert, quantise and write every field of one forecast hour. -> {field: params-or-None}, bytes"""
    t0 = time.time()
    crop = _CTX.get("crop")
    alpha = _CTX.get("alpha")
    phys = {}
    mfull = None
    if model in ("hrrr", "rrfs"):
        if "ir" in blobs:
            phys["ir"] = dec(blobs["ir"])
        if "ceilh" in blobs:
            z = _CTX["zsfc"]
            c = dec(blobs["ceilh"])
            phys["ceil"] = np.maximum(c - z, 0.0) * 3.28084  # m ASL -> ft above ground; NaN = none
        if "vis" in blobs:
            phys["vis"] = dec(blobs["vis"]) / 1609.344
        for k in ("t2m", "td2m"):
            if k in blobs:
                phys[k] = f_of_k(dec(blobs[k]))
        if "smoke" in blobs:
            phys["smoke"] = dec(blobs["smoke"]) * 1e6  # kg m-2 -> mg m-2
        if "u80" in blobs and "v80" in blobs:
            u, v = dec(blobs["u80"]), dec(blobs["v80"])
            phys["wind80"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "u10" in blobs and "v10" in blobs:
            u, v = dec(blobs["u10"]), dec(blobs["v10"])
            phys["wind10"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "gust" in blobs:
            phys["gust"] = dec(blobs["gust"])
        if "sw" in blobs:
            phys["sw"] = dec(blobs["sw"])
        for k in CONV_MAX:
            if k in blobs:
                phys[k] = dec(blobs[k])
        if "etop" in blobs:
            et = dec(blobs["etop"])
            if lead in (3, 12):  # units / missing-value check in the log
                fin = et[np.isfinite(et)]
                print(
                    f"  f{lead:03d} RETOP: finite {fin.size / et.size:.3f}, min {fin.min() if fin.size else 'n/a'}, "
                    f"p50/p99 of > 0: {np.percentile(fin[fin > 0], [50, 99]) if (fin > 0).any() else 'n/a'}, "
                    f"share <= 0: {(fin <= 0).mean() if fin.size else 'n/a'}",
                    flush=True,
                )
            phys["etop"] = np.where(np.isfinite(et) & (et > 0), et * 3.28084 / 1000.0, np.nan)
        if "hpbl" in blobs:  # 6 km (smoothed, every 2nd point): at 3 km in 25 m steps it was
            phys["hpbl"] = UPR.smooth_dec(
                dec(blobs["hpbl"]), 2
            )  # 0.64 MB/h, the largest field of the run (measured 2026-10-07)
        if "mslp" in blobs and "psfc" in blobs:
            F = _CTX["sub"]
            ps = dec(blobs["psfc"])[::F, ::F]
            mfull = dec(blobs["mslp"]) / 100.0
            phys["mslp"] = ("mslp", mfull[::F, ::F], (ps < 85000.0).astype(np.uint8))
        pt = None
        if all(k in blobs for k in ("refd", "crain", "csnow", "cfrzr", "cicep")):
            pt = ptype_flags(*(dec(blobs[k]) for k in ("crain", "csnow", "cfrzr", "cicep")))
            phys["refl"] = ("refl", dec(blobs["refd"]), pt)
        mm = 1 / 25.4
        apcp = dec(blobs["apcp"]) if "apcp" in blobs else np.zeros(_CTX["zsfc"].shape, np.float32)
        phys["apcp"] = apcp * mm
        phys["asnow"] = dec(blobs["asnow"]) * 39.3701 if "asnow" in blobs else np.zeros_like(apcp)  # m of snow -> in
        fz_derived = _CTX.get("fzra_derived", False)
        if not fz_derived:
            phys["afzra"] = dec(blobs["frzr"]) * mm if "frzr" in blobs else np.zeros_like(apcp)
        # sleet (and freezing rain without FRZR): needs the previous processed hour, so the cumulative APCP and the type
        # flags go to a side file for the sequential pass in cmd_run
        seq = dict(
            apcp=apcp.astype(np.float32),
            ip=(pt == 4).astype(np.uint8) if pt is not None else np.zeros(apcp.shape, np.uint8),
        )
        if fz_derived:
            seq["fz"] = (pt == 3).astype(np.uint8) if pt is not None else np.zeros(apcp.shape, np.uint8)
        try:
            seq.update(winter_hour(blobs, None, pt, apcp, phys))
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"::warning::{model} f{lead:03d}: winter inputs failed ({e!r})", flush=True)
        np.savez_compressed(Path(outdir) / f"_seq_{lead:03d}.npz", **seq)
        if any(k.startswith("ua:") for k in blobs):
            try:
                phys.update(ua_lcc(blobs))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"::warning::{model} f{lead:03d}: upper air failed ({e!r})", flush=True)
        if any(k.startswith("sb:") for k in blobs):
            try:
                phys.update(sb_fields(model, blobs, None))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"::warning::{model} f{lead:03d}: snow-band diagnostics failed ({e!r})", flush=True)
    else:
        if "ir" in blobs:
            olr = dec(blobs["ir"], crop)
            with np.errstate(invalid="ignore"):
                phys["ir"] = (np.maximum(olr, 1.0) / SIGMA) ** 0.25
        for k in ("t2m", "td2m"):
            if k in blobs:
                phys[k] = f_of_k(dec(blobs[k], crop))
        if "u80" in blobs and "v80" in blobs:
            u, v = dec(blobs["u80"], crop), dec(blobs["v80"], crop)
            phys["wind80"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "u10" in blobs and "v10" in blobs:
            u, v = dec(blobs["u10"], crop), dec(blobs["v10"], crop)
            phys["wind10"] = to_earth(u, v, alpha) if _CTX.get("grid_rel") else (u, v)
        if "gust" in blobs:
            phys["gust"] = dec(blobs["gust"], crop)
        if "swacc" in blobs and "swacc_prev" in blobs:
            a1, a0 = dec(blobs["swacc"], crop), dec(blobs["swacc_prev"], crop)
            phys["sw"] = np.maximum(a1 - a0, 0.0) / 3600.0
        if "hpbl" in blobs:  # 20 km, the MSLP files' grid (native: 0.3 MB/h x 85 h)
            phys["hpbl"] = UPR.smooth_dec(dec(blobs["hpbl"], crop), _CTX["sub"])
        if "smoke" in blobs:
            phys["smoke"] = dec(blobs["smoke"]) * 1e6
        if "mslp" in blobs and "psfc" in blobs:
            F = _CTX["sub"]
            v = dec(blobs["mslp"], crop)
            ps = dec(blobs["psfc"], crop)
            ps = ps if np.nanmean(ps) > 5000 else ps * 100.0
            mfull = v / 100.0 if np.nanmean(v) > 5000 else v
            phys["mslp"] = ("mslp", mfull[::F, ::F], (ps[::F, ::F] < 85000.0).astype(np.uint8))
        mm = 1 / 25.4
        acc = {"apcp": "apcp", "afzra": "frzr", "aip": "ipacc"}
        shape = None
        for name, key in acc.items():
            if key in blobs:
                phys[name] = np.maximum(dec(blobs[key], crop), 0) * mm
                shape = phys[name].shape
        if "swe" in blobs:
            phys["asnow"] = np.maximum(dec(blobs["swe"], crop), 0) * mm * 10.0  # 10:1, labelled so on the page
        if lead == 0 and shape is None and alpha is not None:  # f00: nothing has fallen yet
            for name in ("apcp", "asnow", "afzra", "aip"):
                phys[name] = np.zeros(alpha.shape, np.float32)
        if "prate" in blobs and "ptype" in blobs:
            code = np.rint(np.nan_to_num(dec(blobs["ptype"], crop), nan=6)).astype(int)
            t = np.zeros(code.shape, np.uint8)
            for k, v in RDPS_PTYPE.items():
                t[code == k] = v
            phys["refl"] = ("rate", dec(blobs["prate"], crop) * 3600.0, t)  # kg m-2 s-1 -> mm/h
        try:  # FRAM ice and the snow-depth increase (sequential pass)
            seq = winter_hour(blobs, crop, None, None, phys)
            if seq:
                np.savez_compressed(Path(outdir) / f"_seq_{lead:03d}.npz", **seq)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"::warning::{model} f{lead:03d}: winter inputs failed ({e!r})", flush=True)
        if any(k.startswith("ua:") for k in blobs):
            try:
                phys.update(ua_rdps(blobs, crop))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"::warning::{model} f{lead:03d}: upper air failed ({e!r})", flush=True)
        if any(k.startswith("sb:") for k in blobs):
            try:
                phys.update(sb_fields(model, blobs, crop))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"::warning::{model} f{lead:03d}: snow-band diagnostics failed ({e!r})", flush=True)
    pts = {}
    try:  # meteogram points + county temperatures: never fail the maps for them
        if _CTX.get("ps") is not None:
            pts = sample_points(model, phys, mfull, _CTX["ps"], _CTX.get("rs"))
    except Exception as e:  # noqa: BLE001
        print(f"  f{lead:03d}: point sampling failed: {e!r}", flush=True)
        pts = {}
    meta, nbytes, maxc = {}, 0, {}
    lo = LO.get(model)
    for name, x in phys.items():
        arr, p = encode(name, x)
        b = webp(arr)
        d = Path(outdir) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{lead:03d}.webp").write_bytes(b)
        nbytes += len(b)
        if lo and name not in LO_SKIP:
            bl = webp(decimate(arr, lo))
            (d / f"{lead:03d}_lo.webp").write_bytes(bl)
            nbytes += len(bl)
        meta[name] = p
        if name in CONV_MAX:  # codes for the run-swath pass (monotonic encodings: max of
            maxc[name] = arr  # codes = code of the max)
    if maxc:
        np.savez_compressed(Path(outdir) / f"_max_{lead:03d}.npz", **maxc)
    return lead, meta, nbytes, time.time() - t0, pts


def sample_points(model, phys, mfull, ps, rs=None):
    """One forecast hour at every meteogram point (points.py SPEC): bilinear for continuous fields, nearest cell for
    ceiling, visibility, reflectivity / rate and precipitation type. -> {name: float32 (npoints,)}, NaN = missing."""
    out = {}

    def near(a):
        v = ps.nearest(a).astype(np.float32)
        v[~ps.inside] = np.nan
        return v

    for k in ("t2m", "td2m", "sw", "apcp", "asnow", "afzra", "aip"):
        if k in phys:
            out[k] = ps.bilinear(phys[k])
    if "wind80" in phys:
        u, v = (ps.bilinear(c) for c in phys["wind80"])
        out["ws80"] = np.hypot(u, v)
        out["wd80"] = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    if "wind10" in phys:
        u, v = (ps.bilinear(c) for c in phys["wind10"])
        out["ws10"] = np.hypot(u, v)
        out["wd10"] = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    if "gust" in phys:
        out["gust"] = ps.bilinear(phys["gust"])
    if "ceil" in phys:
        out["ceil"] = near(phys["ceil"])
    if "vis" in phys:
        out["vis"] = np.minimum(near(phys["vis"]), 10.0)  # 10 = 10 mi or more (the map's and a METAR's ceiling)
    if "refl" in phys:
        kind, val, t = phys["refl"]
        if kind == "refl":
            d = near(val)
            out["refl"] = np.where(d >= REFL_MIN, d, np.nan)
        else:
            r = near(val)
            out["prate"] = np.where(np.isfinite(r) & (r < 0.05), 0.0, r)
        tt = near(t)
        out["ptype"] = tt
    if "smoke" in phys:
        smp = rs if model == "rdps" else ps  # RDPS smoke is on the RAQDPS grid
        if smp is not None:
            out["smoke"] = smp.bilinear(phys["smoke"])
    if mfull is not None:
        out["mslp"] = ps.bilinear(mfull)
    return out


# ── fetch plans ─────────────────────────────────────────────────────────────────────────────────────────────────────
def fetch_hour(model, date, cyc, lead, cyc_dt, skip=()):
    """-> {key: bytes} for one forecast hour (keys as in IDX_WANT / RDPS_VARS, less `skip`)."""
    smoke = smoke_on(cyc_dt)
    if model in ("hrrr", "rrfs"):
        want = {
            k: v
            for k, v in IDX_WANT[model].items()
            if k != "zsfc"
            and k not in skip
            and not (lead == 0 and (k in ACC or k in CONV_MAX))  # f00: no accumulations or hourly maxima
            and (smoke or k != "smoke")
        }
        out = fetch_idx_fields(model, file_url(model, date, cyc, lead), want)
        pw = dict(ua_want(model)) if (UPR.ua_hour(lead) and ua_on(model)) else {}
        # the Cobb snow ratio (HRRR): T, RH and omega every 25 hPa 925-300 hPa, fetched only for hours with snow somewhere
        if model == "hrrr" and UA_ON and "csnow" in out and np.nanmax(sb.decode(out["csnow"])) > 0.5:
            for p in WIN.COBB_LEVS:
                pw[f"ua:TMP:{p}"] = rf"^TMP:{p} mb:(anl|\d+ hour fcst):?$"
                pw[f"cb:RH:{p}"] = rf"^RH:{p} mb:(anl|\d+ hour fcst):?$"
                pw[f"cb:VVEL:{p}"] = rf"^VVEL:{p} mb:(anl|\d+ hour fcst):?$"
        if sb_on(model, cyc_dt) and sb_hour(lead):
            pw.update(sb_want_lcc())
            try:  # composite reflectivity for the model-snow outline
                out.update(
                    fetch_idx_fields(model, file_url(model, date, cyc, lead), {"sb:REFC": r"^REFC:entire atmosphere"})
                )
            except Exception as e:  # noqa: BLE001
                print(f"::warning::{model} f{lead:03d}: no REFC for the snow-band outline ({e!r})", flush=True)
        if pw:
            # upper air from the isobaric file: never fails the hour (the surface maps are the core product)
            for k in range(2):
                try:
                    out.update(fetch_idx_fields(model, prs_url(model, date, cyc, lead), pw))
                    break
                except Exception as e:  # noqa: BLE001
                    if k:
                        print(f"::warning::{model} f{lead:03d}: no upper air this hour ({e!r})", flush=True)
                    else:
                        time.sleep(15)
        return out
    f = fetcher("rdps")
    names = dict(RDPS_VARS)
    if lead == 0:
        names.pop("ir", None)  # no OLR at the analysis time
    jobs = {k: rdps_url(date, cyc, lead, v) for k, v in names.items()}
    if lead >= 1:
        jobs["swacc_prev"] = rdps_url(date, cyc, lead - 1, RDPS_VARS["swacc"])
    if UPR.ua_hour(lead) and ua_on("rdps"):
        jobs.update(rdps_ua_jobs(date, cyc, lead))
    if sb_on("rdps", cyc_dt) and sb_hour(lead):
        jobs.update(rdps_sb_jobs(date, cyc, lead))
    r_dt, off = raq_for(cyc_dt)
    if smoke and lead + off <= 72:
        jobs["smoke"] = raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, lead + off)
    out = {}

    def one(kv):
        k, url = kv
        try:
            return k, f.get(url)
        except sb.Missing:
            return k, None

    with ThreadPoolExecutor(8) as ex:
        for k, b in ex.map(one, jobs.items()):
            if b is not None:
                out[k] = b
    if lead == 0:
        out.pop("swacc", None)
    return out


def complete(model, cyc_dt):
    """Is the cycle's LAST forecast hour published (for RDPS, and the matching RAQDPS run to f72)?"""
    date, cyc = cyc_dt.strftime("%Y%m%d"), cyc_dt.hour
    n = MODELS[model]["length"](cyc)
    if model in ("hrrr", "rrfs"):
        return head_ok(file_url(model, date, cyc, n) + ".idx")
    ok = head_ok(rdps_url(date, cyc, n, RDPS_VARS["swacc"]))
    if not ok:
        return False
    if not smoke_on(cyc_dt):  # out of the smoke season: no RAQDPS to wait for
        return True
    r_dt, _off = raq_for(cyc_dt)
    if head_ok(raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, 72)):
        return True
    # RAQDPS late: wait for it up to 3 h after the RDPS run is complete, then go without smoke (the page says so)
    return (dt.datetime.utcnow() - cyc_dt).total_seconds() > (MODELS[model]["lag_h"][0] + 5) * 3600


# ── manifest ────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_manifest(path, model):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return dict(model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=[])


def retain(cycles, model):
    """Newest first. Keep only the newest cycle (storage); KEEP_<MODEL> overrides.
    HRRR runs that are not 00/06/12/18Z (the retired 18-h hourly runs) are always dropped."""
    keep_n = int(os.environ.get(f"KEEP_{model.upper()}", "1"))
    cycles = sorted(cycles, key=lambda c: c["cycle"], reverse=True)
    ok = [c for c in cycles if model != "hrrr" or int(c["cycle"][8:10]) % 6 == 0]
    keep = ok[:keep_n]
    return keep, [c for c in cycles if c not in keep]


# ── commands ────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan(a):
    """Print one line per model cycle to process: the newest COMPLETE cycle not already in that model's manifest, and
    for HRRR also the newest complete 48-h cycle if it is newer than the published one."""
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    todo = []
    for model in a.models.split(","):
        m = MODELS[model]
        man = load_manifest(Path(a.manifests) / f"{model}.json", model)
        have = {c["cycle"] for c in man.get("cycles", [])}
        cands = []
        for h in range(m["lag_h"][0], m["lag_h"][1] + 1):
            c = now - dt.timedelta(hours=h)
            if c.hour % m["every"] == 0:
                cands.append(c)
        picked = None
        for c in cands:
            key = c.strftime("%Y%m%d%H")
            if key in have:
                break  # newest published: nothing newer is complete
            if complete(model, c):
                picked = c
                break
        if picked:
            todo.append((model, picked.strftime("%Y%m%d%H")))
    for model, key in todo:
        print(f"{model} {key}")


def cmd_run(a):
    model, key = a.model, a.cycle
    cyc_dt = dt.datetime.strptime(key, "%Y%m%d%H")
    date, cyc = key[:8], cyc_dt.hour
    n = MODELS[model]["length"](cyc)
    leads = leads_of(model, cyc)
    if os.environ.get("MV_LEADS"):  # testing only: a subset of the hours
        leads = [int(x) for x in os.environ["MV_LEADS"].split(",")]
    site = Path(a.site)
    outdir = site / DATA / model / key
    if outdir.exists():
        import shutil

        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t_start = time.time()

    # static context: grid description, terrain (ceiling AGL), wind rotation, the RDPS crop
    ctx, grids = {}, {}
    if model in ("hrrr", "rrfs"):
        z = fetch_idx_fields(model, file_url(model, date, cyc, 0), {"zsfc": IDX_WANT[model]["zsfc"]})["zsfc"]
        lat, lon, grel = grid_latlon(z)
        ctx.update(zsfc=sb.decode(z), grid_rel=grel, alpha=alpha_of(lat, lon) if grel else None)
        grids["main"] = grid_desc(z)
        ctx.update(
            ua_levs=UPR.levs_of(model),
            lat_sub=lat[:: SUB[model], :: SUB[model]].astype(np.float64),
            lat1=grids["main"]["lat1"],
            lat2=grids["main"]["lat2"],
            dx=grids["main"]["dx"],
        )
        if sb_on(model, cyc_dt):
            ctx.update(sb_lat=np.asarray(lat, np.float64), sb_lon=np.asarray(lon, np.float64))
        del lat, lon
        h1 = next((h for h in leads if h > 0), None)
        if model in FZRA_DERIVED and h1 is not None:
            lines = idx_lines(fetcher(model), file_url(model, date, cyc, h1))
            ctx["fzra_derived"] = not any(re.search(ACC["frzr"], desc) for _, _, desc in lines)
            if ctx["fzra_derived"]:
                print(
                    f"  {model} {key}: no FRZR in the files; freezing rain derived from APCP and the type", flush=True
                )
    else:
        f = fetcher("rdps")
        b = f.get(rdps_url(date, cyc, 1, RDPS_VARS["u80"]))
        lat, lon, grel = grid_latlon(b)
        m = (lat >= RDPS_BOX[2]) & (lat <= RDPS_BOX[3]) & (lon >= RDPS_BOX[0]) & (lon <= RDPS_BOX[1])
        jj, ii = np.where(m)
        crop = ((int(jj.min()), int(jj.max()) + 1), (int(ii.min()), int(ii.max()) + 1))
        sl = (slice(*crop[0]), slice(*crop[1]))
        ctx.update(crop=crop, grid_rel=grel, alpha=alpha_of(lat, lon)[sl] if grel else None)
        grids["main"] = grid_desc(b, crop)
        gm, F0 = grids["main"], SUB[model]  # PV on the rotated grid: its own latitude of each sub row
        ny0 = -(-gm["ny"] // F0)
        ctx.update(
            rd_grid=gm,
            rd_rlat=gm["la1"] + (gm.get("j0", 0) + F0 * np.arange(ny0)) * gm["dj"],
            lat_sub=UPR.smooth_dec(lat[sl], F0).astype(np.float64),
        )
        if sb_on(model, cyc_dt):
            ctx.update(sb_lat=np.asarray(lat[sl], np.float64), sb_lon=np.asarray(lon[sl], np.float64))
        r_dt, off = raq_for(cyc_dt)
        try:
            if not smoke_on(cyc_dt):
                raise sb.Missing("out of the smoke season (SMOKE_SEASON)")
            rb = f.get(raq_url(r_dt.strftime("%Y%m%d"), r_dt.hour, off + 1))
            grids["raq"] = grid_desc(rb)
        except Exception as e:  # noqa: BLE001
            print(f"  RAQDPS {r_dt:%Y%m%d%H}: {e} - no smoke layer for this cycle")
        del lat, lon
    ctx["sub"] = F = SUB[model]
    g = dict(grids["main"])
    if g["type"] == "lcc":
        g.update(dx=g["dx"] * F, dy=g["dy"] * F)
    else:
        g.update(di=g["di"] * F, dj=g["dj"] * F, i0=g.get("i0", 0) / F, j0=g.get("j0", 0) / F)
    g.update(nx=-(-g["nx"] // F), ny=-(-g["ny"] // F))
    grids["sub"] = g
    if LO.get(model):  # the half-resolution copies' grid
        F2, g2 = LO[model], dict(grids["main"])
        if g2["type"] == "lcc":
            g2.update(dx=g2["dx"] * F2, dy=g2["dy"] * F2)
        else:
            g2.update(di=g2["di"] * F2, dj=g2["dj"] * F2, i0=g2.get("i0", 0) / F2, j0=g2.get("j0", 0) / F2)
        g2.update(nx=-(-g2["nx"] // F2), ny=-(-g2["ny"] // F2))
        grids["lo"] = g2
    # meteogram points (airports + extra cities), located on the grid
    # once; the workers sample them while each hour's fields are in memory (points.py)
    pt_ids = []
    try:
        pt_ids, plat, plon = PTS.load_points()
        ctx["ps"] = PTS.Sampler(grids["main"], plat, plon)
        if "raq" in grids:
            ctx["rs"] = PTS.Sampler(grids["raq"], plat, plon)
        print(f"  points: {int(ctx['ps'].inside.sum())} of {len(pt_ids)} on the grid", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::{model} {key}: no point series this run ({e!r})", flush=True)
        ctx.pop("ps", None), ctx.pop("rs", None), ctx.pop("cs", None)
    print(
        f"{model} {key}: {len(leads)} hours, grid {grids['main']['nx']} x {grids['main']['ny']}, "
        f"setup {time.time() - t_start:.1f}s",
        flush=True,
    )

    # fetch (threads) -> decode/quantise/encode (processes)
    per_field = {k: {} for k in ORDER}
    samples = {}  # lead -> {point field: values}
    total_bytes, failed = 0, []
    t_fetch = 0.0
    hour_workers = 4 if model in ("hrrr", "rrfs") else 3
    with (
        ProcessPoolExecutor(a.procs, initializer=_init, initargs=(ctx,)) as pool,
        ThreadPoolExecutor(hour_workers) as fx,
    ):
        skip = ("frzr",) if ctx.get("fzra_derived") else ()

        def fetch_one(lead):
            t = time.time()
            for k in range(3):
                try:
                    return lead, fetch_hour(model, date, cyc, lead, cyc_dt, skip), time.time() - t
                except (sb.Missing, sb.ShortRead) as e:
                    if k == 2:
                        print(f"  f{lead:03d}: {e}", flush=True)
                        return lead, None, time.time() - t
                    time.sleep(20)

        futs = []
        for lead, blobs, secs in fx.map(fetch_one, leads):
            t_fetch += secs
            if not blobs:
                failed.append(lead)
                continue
            futs.append(pool.submit(process_hour, model, lead, blobs, str(outdir)))
        for fu in as_completed(futs):
            lead, meta, nb, secs, pts = fu.result()
            samples[lead] = pts
            total_bytes += nb
            for name, p in meta.items():
                per_field[name][lead] = p
    # sequential pass over the published hours in order: sleet = APCP fallen between consecutive hours while the sleet type
    # is on (HRRR/RRFS), freezing rain the same way where the files carry no FRZR (FZRA_DERIVED), and the winter
    # accumulations (winter.py): FRAM radial ice, snowfall with the Cobb ratio (HRRR) and at 10:1 (HRRR/RRFS), the
    # snow-depth increase. Each is a running total written like the other accumulations.
    seq = sorted(outdir.glob("_seq_*.npz"))
    if seq:
        prev, cums, plead = {}, {}, None
        st = dict(slr=[], slrk=[], none=0, nsnow=0)

        def emit(name, arr_in, lead):
            nonlocal total_bytes
            arr, _ = encode(name, arr_in)
            b = webp(arr)
            (outdir / name).mkdir(exist_ok=True)
            (outdir / name / f"{lead:03d}.webp").write_bytes(b)
            total_bytes += len(b)
            if LO.get(model):
                bl = webp(decimate(arr, LO[model]))
                (outdir / name / f"{lead:03d}_lo.webp").write_bytes(bl)
                total_bytes += len(bl)
            per_field[name][lead] = None

        for fpath in seq:
            lead = int(fpath.stem[5:])
            z = np.load(fpath)
            hrs = float(lead - plead) if plead is not None else 1.0
            plead = lead

            def inc(key, z=z):
                cur = z[key].astype(np.float64)
                d = np.maximum(cur - prev.get(key, cur if key == "snod" else 0.0), 0)
                prev[key] = cur
                return d

            def add(name, x):
                cums[name] = cums.get(name, 0) + x
                return cums[name]

            liq = None  # this period's freezing-rain liquid, in (FRAM)
            if "apcp" in z:
                dap = inc("apcp") / 25.4  # in
                cum = add("aip", dap * z["ip"])
                emit("aip", cum, lead)
                if ctx.get("ps") is not None:
                    samples.setdefault(lead, {})["aip"] = ctx["ps"].bilinear(cum)
                if "fz" in z:
                    liq = dap * z["fz"]
                    cum = add("afzra", liq)
                    emit("afzra", cum, lead)
                    if ctx.get("ps") is not None:
                        samples.setdefault(lead, {})["afzra"] = ctx["ps"].bilinear(cum)
                if "sf" in z:
                    sl = dap * z["sf"] / 2.0  # snow-phase liquid, in
                    emit("as10", add("as10", 10.0 * sl), lead)
                    if model == "hrrr":
                        slr = (
                            np.nan_to_num(z["slr"].astype(np.float32), nan=WIN.SLR_NONE)
                            if "slr" in z
                            else np.full(sl.shape, WIN.SLR_NONE, np.float32)
                        )
                        emit("acobb", add("acobb", slr * sl), lead)
                        wet = sl > 0.01
                        if "slr" in z and wet.any():
                            st["slr"].append(np.asarray(z["slr"], np.float32)[wet])
                            st["slrk"].append(np.asarray(z["slrk"], np.float32)[wet])
                            st["none"] += int(z["slr_none"][0])
                            st["nsnow"] += int(z["slr_none"][1])
            if "frzr" in z:
                liq = inc("frzr") / 25.4
            if liq is not None:
                if "tw" in z and (liq > 0).any():
                    rad = WIN.fram_radial(
                        liq,
                        hrs,
                        np.nan_to_num(z["tw"].astype(np.float32), nan=0.0),
                        np.nan_to_num(z["ws"].astype(np.float32), nan=0.0),
                    )
                else:
                    rad = np.zeros(liq.shape, np.float32)
                emit("afram", add("afram", rad), lead)
            if "snod" in z:
                emit("asnod", add("asnod", inc("snod")), lead)
            last_z = {k: z[k] for k in ("asnow",) if k in z}
            fpath.unlink()
        if model == "hrrr" and "acobb" in cums:  # the snow-ratio check printed with every run
            c, t10, an = cums["acobb"], cums.get("as10"), last_z.get("asnow")
            msg = f"  snow ratio check: hours x points with snow liquid > 0.01 in: {sum(len(a) for a in st['slr'])}"
            if st["slr"]:
                sa, sk = np.concatenate(st["slr"]), np.concatenate(st["slrk"])
                msg += (
                    f"; Cobb SLR p10/p50/p90 {np.percentile(sa, 10):.1f}/{np.percentile(sa, 50):.1f}/{np.percentile(sa, 90):.1f} mean {sa.mean():.1f}"
                    f"; Kuchera p50 {np.median(sk):.1f} mean {sk.mean():.1f}; no saturated layer {100 * st['none'] / max(st['nsnow'], 1):.0f} % of snow points"
                )
            if an is not None:
                m = (c > 0.1) | (an > 0.1)
                if m.any():
                    msg += (
                        f"; run totals over {int(m.sum())} cells with > 0.1 in: Cobb {c[m].mean():.2f} in, 10:1 {t10[m].mean():.2f}, "
                        f"ASNOW {an[m].mean():.2f}, r(Cobb, ASNOW) {np.corrcoef(c[m], an[m])[0, 1]:.2f}"
                    )
            print(msg, flush=True)
    # run swaths: the running maximum of each hourly-max field since the start of the run (255 = missing is skipped)
    run_max = {}
    for fpath in sorted(outdir.glob("_max_*.npz")):
        lead = int(fpath.stem[5:])
        z = np.load(fpath)
        for name in z.files:
            c = z[name]
            cur = run_max.get(name)
            cur = c.copy() if cur is None else np.where(c == 255, cur, np.where(cur == 255, c, np.maximum(cur, c)))
            run_max[name] = cur
            b = webp(cur)
            (outdir / (name + "s")).mkdir(exist_ok=True)
            (outdir / (name + "s") / f"{lead:03d}.webp").write_bytes(b)
            total_bytes += len(b)
            per_field[name + "s"][lead] = None
        fpath.unlink()
    fetch_stats = dict(
        requests=sum(f.nreq for f in _FETCH.values()), mb=round(sum(f.nbytes for f in _FETCH.values()) / 1e6, 1)
    )
    fields = {}
    for name in ORDER:
        hrs = sorted(per_field[name])
        if not hrs:
            continue
        e = dict(
            hours=hrs,
            grid="raq"
            if (model == "rdps" and name == "smoke")
            else "sub"
            if name == "mslp"
            else UA_GRID[model].get(name, "sub")
            if name in UA_FIELDS
            else HPBL_GRID[model]
            if name == "hpbl"
            else SB_GRID[model]
            if name in SB_FIELDS
            else "main",
        )
        if name in SB_FIELDS:
            e["enc"] = name
        fd = FIELDS[name]
        if fd["enc"] in ("uvz", "vz"):
            e.update(enc=fd["enc"], q=[per_field[name][h] for h in hrs])
        if fd["enc"] in ("uvz", "dt"):
            e.update(enc=fd["enc"], uv=[fd["off"], fd["step"]])
        if fd["enc"] == "dt":
            e["th"] = list(fd["th"])
        if fd["enc"] == "vz":
            e["vo"] = list(fd["vo"])
        if FIELDS[name]["enc"] == "linf":
            e["q"] = [per_field[name][h] for h in hrs]
        if name == "refl":
            e["enc"] = "rate2" if model == "rdps" else "refl2"
        if fd["enc"] in ("uh", "uvv", "etop"):
            e["enc"] = fd["enc"]
        if FIELDS[name]["enc"] == "acc16":
            method = FZRA_DERIVED[model].get(name) if ctx.get("fzra_derived") else None
            e.update(enc="acc16", unit=FIELDS[name]["unit"], method=method or ACC_METHOD[model][name])
        if LO.get(model) and name not in LO_SKIP and e["grid"] == "main":
            e["lo"] = "lo"
        fields[name] = e
    if model == "rdps" and "smoke" in fields:
        r_dt, off = raq_for(cyc_dt)
        fields["smoke"]["from"] = f"RAQDPS-FireWork {r_dt:%Y-%m-%d %H}Z"
    hours_all = sorted({h for e in fields.values() for h in e["hours"]})
    extra_files = (
        write_points(model, key, cyc_dt, hours_all, pt_ids, samples, outdir) if ctx.get("ps") is not None else {}
    )
    total_bytes += sum(v["bytes"] for v in extra_files.values())
    wall = time.time() - t_start
    entry = dict(
        cycle=key,
        init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"),
        length=n,
        fields=fields,
        grids=grids,
        hours=sorted({h for e in fields.values() for h in e["hours"]}),
        missing_hours=failed,
        bytes=total_bytes,
        fetch=fetch_stats,
        wall_s=round(wall, 1),
        made=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
    )
    entry.update(extra_files)  # "points": {...}, "points_bin": {...}
    # the snow-band season as this run saw it: the page says "in season 1 October - 30 April" instead of an empty map
    entry["snowband"] = dict(
        models=list(SB_MODELS),
        window=sb_window_text(),
        active=sb_in_season(cyc_dt),
        forced=os.environ.get("MV_SB", "auto") == "force" and not sb_in_season(cyc_dt),
        computed=any(f in fields for f in SB_FIELDS),
    )
    # the run may not publish a cycle that lost hours in the middle (a partial run would be kept for two cycles)
    ok = not failed or failed == [0]
    print(
        f"{model} {key}: {total_bytes / 1e6:.1f} MB in {sum(len(f['hours']) for f in fields.values())} files, "
        f"fetched {fetch_stats['mb']} MB in {fetch_stats['requests']} requests, wall {wall:.0f}s, "
        f"missing hours {failed}",
        flush=True,
    )
    for name, e in fields.items():
        sz = sum((outdir / name / f"{h:03d}.webp").stat().st_size for h in e["hours"])
        szl = sum((outdir / name / f"{h:03d}_lo.webp").stat().st_size for h in e["hours"]) if e.get("lo") else 0
        print(
            f"  {name:7s} {len(e['hours']):3d} h  {sz / 1e6:7.1f} MB  ({sz / len(e['hours']) / 1e3:.0f} KB/h)"
            + (f"  + lo {szl / 1e6:.1f} MB ({szl / max(sz, 1) * 100:.0f} %)" if szl else "")
        )
    if not ok:
        print(f"::error::{model} {key}: hours {failed} failed - not publishing a partial cycle")
        sys.exit(1)

    # manifest + retention
    mpath = site / "assets/models" / f"{model}.json"
    man = load_manifest(Path(a.manifest_in) if a.manifest_in else mpath, model)
    cycles = [c for c in man.get("cycles", []) if c["cycle"] != key] + [entry]
    keep, drop = retain(cycles, model)
    man.update(
        model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=keep, updated=entry["made"]
    )
    # the ECAPE job's section (hrrr only; scripts/models/manifest_merge.py): entries older than every kept cycle go, and
    # their directories with them
    ec_drop = prune_ecape(man) if man.get("ecape") else []
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, separators=(",", ":")))
    if a.lists:
        L = Path(a.lists)
        L.mkdir(parents=True, exist_ok=True)
        # a cycle older than everything retention keeps (a forced re-run of an old cycle) is not published at all
        # one line per FIELD directory, not the cycle directory: publish_data.sh replaces what it is handed, and the ECAPE
        # job may already have put ecape_mu/ ... mlcape/ into this cycle's directory (it must survive this publish)
        (L / "publish.txt").write_text(
            ""
            if any(c["cycle"] == key for c in drop)
            else "".join(f"{DATA}/{model}/{key}/{name}\n" for name in fields)
            + "".join(f"{DATA}/{model}/{key}/{v['file']}\n" for v in extra_files.values())
        )
        (L / "prune.txt").write_text(
            "".join(f"{DATA}/{model}/{c}\n" for c in sorted({c["cycle"] for c in drop} | set(ec_drop)))
        )
        # PRUNE_OLDER floor: any cycle directory older than every kept cycle is garbage (an orphan from a run whose
        # manifest commit failed, or ECAPE for a cycle this job never published)
        (L / "floor.txt").write_text(f"{DATA}/{model}:{min(c['cycle'] for c in keep)}\n" if keep else "")
        (L / "cycle.json").write_text(json.dumps(entry))
    print(f"manifest: keep {[c['cycle'] for c in keep]}, drop {[c['cycle'] for c in drop]}")


def write_points(model, key, cyc_dt, hours, ids, samples, outdir):
    """points.json (meteograms) from the per-hour samples. -> manifest entries."""
    t0 = time.time()
    out = {}
    try:
        names = sorted({n for s in samples.values() for n in s if not n.startswith("_")})
        series = {n: np.full((len(ids), len(hours)), np.nan, np.float32) for n in names}
        for j, h in enumerate(hours):
            for n, v in samples.get(h, {}).items():
                if n in series:
                    series[n][:, j] = v
        if model in ("hrrr", "rrfs") and 0 in hours:  # f00 has no accumulation messages: nothing has fallen
            for n in ("apcp", "asnow", "afzra", "aip"):
                if n in series:
                    series[n][:, 0] = (
                        np.where(np.isfinite(series["t2m"][:, 0]), 0.0, np.nan) if "t2m" in series else 0.0
                    )
        doc = PTS.build(
            model,
            key,
            hours,
            ids,
            series,
            extra=dict(
                init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"),
                ptype_codes={
                    "0": "none",
                    "1": "rain",
                    "2": "snow",
                    "3": "freezing rain",
                    "4": "sleet",
                    "5": "rain-snow mix",
                },
            ),
        )
        b = PTS.dumps(doc).encode()
        (Path(outdir) / "points.json").write_bytes(b)
        out["points"] = dict(file="points.json", bytes=len(b), n=len(doc["ids"]), fields=list(doc["fields"]))
        # the same series per station (a click reads one with a Range request; points.py)
        bb, head = PTS.bin_bytes(doc)
        (Path(outdir) / "points.bin").write_bytes(bb)
        out["points_bin"] = dict(file="points.bin", bytes=len(bb), head=head, n=len(doc["ids"]))
        import gzip

        print(
            f"  points.json {len(b) / 1e3:.0f} KB ({len(gzip.compress(b, 6)) / 1e3:.0f} KB gzipped), "
            f"{len(doc['ids'])} points x {len(hours)} h x {len(doc['fields'])} fields; points.bin {len(bb) / 1e3:.0f} KB "
            f"(head {head / 1e3:.1f} KB, {(len(bb) - head) / max(len(doc['ids']), 1):.0f} B per station); "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"::warning::{model} {key}: point / degree-day files not written ({e!r})", flush=True)
    return out


def cmd_stamp(a):
    """After the frames push: record the frames commit that carries the cycle (the jsDelivr fallback pins it)."""
    p = Path(a.manifest)
    man = json.loads(p.read_text())
    for c in man["cycles"]:
        if c["cycle"] == a.cycle:
            c["sha"] = a.sha
    p.write_text(json.dumps(man, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--models", default="hrrr,rrfs,rdps")
    p.add_argument("--manifests", default="assets/models")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--cycle", required=True)
    p.add_argument("--site", default=".")
    p.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    p.add_argument("--lists", default="")
    p.add_argument("--manifest-in", default="", help="the manifest to extend (main's CURRENT copy); default the site's")
    p = sub.add_parser("stamp")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cycle", required=True)
    p.add_argument("--sha", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "stamp": cmd_stamp}[a.cmd](a)


if __name__ == "__main__":
    main()
