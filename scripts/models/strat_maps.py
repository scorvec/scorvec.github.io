#!/usr/bin/env python3
"""Stratosphere model maps for strat-models.html: the GFS, the GDPS, the AIFS single and GEOS FP on their pressure levels
from 100 hPa up and (GFS, GDPS, GEOS FP) on the isentropic surfaces their levels support, both hemispheres, every 12
hours (GFS to 16 days, GDPS to 10, AIFS single to 15, GEOS FP to 10 from 00 UTC and 5 from 12 UTC). The 00 and 12 UTC
runs; the AIFS single's from 00, 06, 12 and 18 UTC (its 06/18 UTC runs reach 15 days too).

Each model is read on its latitude-longitude output grid (GFS 0.5 deg; GDPS 0.15 deg, averaged over 3 x 3 cells to
0.45 deg; AIFS single 0.25 deg, one 1-2-1 pass and every second point to 0.5 deg; GEOS FP every second point of its
0.25 x 0.3125 deg grid, interpolated in longitude to 0.5 deg) and every field and step becomes ONE lossless WebP per
hemisphere - the format of models.html
(scripts/models/models.py), drawn by the page with WebGL on a polar stereographic projection. Nothing is rendered here.

  gfs    NOAA GFS (FV3, 127 layers to 80 km). Maps on its 15 levels 100-0.4 hPa; PV from T, u, v on 27 levels, 400 hPa
         to 0.4 hPa (the 0.5 deg files' 21 from 400 hPa up, and the six 25 hPa levels between 125 and 375 hPa from the
         "b" files), on 350-1600 K; the smoothed vertical motion (below).
  gdps   ECCC GDPS (GEM, 15 km). The MSC Datamart has T, z, u, v above 100 hPa only at 50, 30, 20, 10, 5 and 1 hPa, and
         vertical velocity only from 200 hPa down (listings of 2026-10-24): maps on 100-1 hPa, no vertical motion, PV
         from 16 levels 400-1 hPa on 350-850 K only (between 5 and 1 hPa there is no level for 11 km; the GFS run
         measures what that costs, check "gdpslev" below).
  aifs   ECMWF AIFS single (the deterministic AIFS, 0.25 deg open data). Above 100 hPa it publishes 50 and 10 hPa only:
         maps on 100, 50 and 10 hPa, no PV (a column of 100, 50, 10 hPa cannot resolve d theta/dp on the isentropes)
         and no vertical motion (the AIFS-ENS's w failed the site's residual-circulation checks; the AIFS's is not used).
  geos   NASA GMAO GEOS FP (the assimilation's forecasts, 72 layers to 0.01 hPa; day 0 is its analysis): maps on its 14
         levels 100-0.4 hPa that the GFS shares, and PV on 350-1600 K from the model's OWN Ertel PV (EPV, published on
         its pressure levels; given the same 1-2-1 pass as the PV computed here), interpolated from its 22 levels 400-0.4
         hPa; no vertical motion, no absolute-vorticity or 2-PVU check (nothing is differentiated).

Files, per hemisphere (nh: 0-90 N, sh: 90 S-0), rows south -> north, columns from 180 W less PAD cells to 180 E plus
PAD cells (the grid from 180 W with PAD columns repeated on each side, so the page can sample and smooth across the date
line without wrapping):
  t<lev>   R = temperature, 160 K + 0.6 K * code; G = geopotential height, per-frame offset and step (manifest "q")
  w<lev>   R, G = earth-relative u, v, one per-frame offset and step for both (manifest "q")
  pv<K>    R = Lait-scaled Ertel potential vorticity, -20 + 0.5 * code in PVU at the 475 K scale (see below), sign
           reversed in the Southern Hemisphere so the vortex is positive in both; G = the Montgomery streamfunction
           divided by g, metres, per-frame offset and step; B = the pressure of the surface, log 0.2-600 hPa
  o<lev>   (GFS) R = Eulerian vertical motion, hPa/day, DESCENT POSITIVE (omega in Pa/s x 864), smoothed (below):
           -2 X + 4 X / 254 * code, X = the level's colour range (manifest "omega" "ranges")
  255 = missing in every channel. <lev> is the pressure in hPa with "p" for the point (0p4 = 0.4 hPa).

Potential vorticity. Ertel PV on the pressure levels (hydrostatic, isobaric form, spherical derivatives, centred and
periodic in longitude; vertical derivatives in ln p on the actual levels):
    PV = -g [ (zeta + f) dtheta/dp - dv/dp dtheta/dx + du/dp dtheta/dy ]          1 PVU = 1e-6 K m2 kg-1 s-1
with one 1-2-1 pass in latitude and longitude; the rows at the poles take the mean of the ring next to them. PV, the
Montgomery streamfunction M = cp T + g z and ln p are then interpolated to each isentrope linearly in ln theta, column
by column (theta = T (1000 hPa / p)^(R/cp)). PV grows ~ theta^(9/2) with height in a resting atmosphere, so the page
colours the Lait (1994) scaled value PV (theta / 475 K)^(-9/2): a vortex reads alike on every surface, and at 475 K the
scaled value is the PV itself. The page converts back for its hover values and legend ticks.

Vertical motion (GFS). The GFS's own omega (VVEL, Pa/s, published on all 15 map levels), smoothed hard: raw omega on a
0.5 deg grid in the stratosphere is mostly gravity waves. In space, a Gaussian of standard deviation VV_SIGMA_KM along
the sphere (smooth_sphere: exact along each latitude circle in Fourier space, then along the meridians, continued over
the poles); in time, a centred running mean over 72 h of the 12-hourly steps (trapezoidal: weight 1 inside +-36 h, 1/2
at +-36 h), shortened near the ends of the run so it stays centred (24 h at +12 h, 48 h at +24 h; none at +0 and
+384 h, which have no vertical motion; the manifest's "win" gives the hours each step covers). The page's
colour range per level is fixed by `omega-clim`: the 99th percentile of |omega| poleward of 20 deg, smoothed the same
way, over 48 samples of GFS analyses (the 15th of each month, July 2022 - June 2026, the seven 00/12 UTC analyses of
the 72 h around it), rounded up -> strat_omega_ranges.json beside this script. Without that file a run writes no
vertical motion.

Checks, per run (manifest "checks"; the page's method section shows them):
  absv/<lev>     zeta + f from these winds against the model's own absolute vorticity (GFS ABSV at 50 and 10 hPa,
                 GDPS AbsoluteVorticity at 250 and 200 hPa: its only levels above 500 hPa), 20-85 deg
  pv2/<hemi>     the 2-PVU surface from this PV (searched downward from 100 hPa, models.html's dynamic-tropopause rule)
                 against the model's own 2-PVU pressure (from its native levels), 30-80 deg
  gdpslev/<K>    (GFS) PV recomputed from the GFS on exactly the GDPS's 16 levels against the GFS's own 27-level PV, per
                 isentrope: what the GDPS's coarser levels cost (pattern correlation, polar-cap mean ratio)
  units          (GDPS) what each field turned out to be in, read from its magnitudes, not its GRIB keys (ECCC files
                 have carried decametres under a metres label): temperature in K, heights against a standard
                 atmosphere at every level (the 10 hPa median near 31,000 m), winds, vorticity, the 2-PVU pressure
  cap            the polar-cap (poleward of 60 deg, cos-weighted) mean PV and log-mean pressure at 475 and 850 K on every
                 step: the quantities GEOS FP's run on the page reports (its own EPV), so the page can
                 set the models side by side for the same day

Sources. GFS (hard rule: NOAA models from AWS S3, never NOMADS): noaa-gfs-bdp-pds, byte ranges of only the wanted
messages via the .idx files (scripts/snowband/snowband.py's validated transport); +384 h reaches the bucket ~5 h 15 min
after the start (2026-10-23/24). GDPS: the dated MSC Datamart tree (dd.weather.gc.ca/YYYYMMDD/WXO-DD/model_gdps/15km,
the files scripts/strat/upperstrat_data.py and scripts/models/samerica.py read) through scripts/geps/datamart.py (the
site's honest User-Agent, retries, a request ledger) with modest concurrency; one global file per variable, level and
step, read whole (they cannot be cut); +240 h lands ~4 h 50 min after the start, and readiness is a HEAD on two of the
last step's files. AIFS single (hard rule: ECMWF open data from the Google Cloud mirror only): byte ranges of t, gh, u, v
on its three levels through scripts/ecmwf/rangefetch.py (the index, coalesced ranges, the request budget); +360 h is
on the mirror ~7 h 15 min after the start, and readiness is scripts/ecmwf/cycles.py's (every step's file present).
GEOS FP: NASA's NCCS OPeNDAP server (fcast/inst3_3d_asm_Np, 3-hourly), one request per step for t, h, epv, u, v on 400-0.4
hPa at every second point (~90 MB, read with a small DAP binary parser); readiness is the last step's top level holding
data (the collection is listed before its last steps are written).

    python scripts/models/strat_maps.py plan --model gfs --manifests assets/models
    python scripts/models/strat_maps.py run --model gdps --cycle 2026102412 --site . --procs 3 --lists /tmp/strat
    python scripts/models/strat_maps.py omega-clim --procs 4          # once: the GFS vertical-motion colour ranges
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "snowband"))
sys.path.insert(0, str(HERE.parent / "geps"))
sys.path.insert(0, str(HERE.parent / "ecmwf"))
import datamart as DM  # the MSC Datamart fetcher (honest User-Agent, retries, ledger)
import models as MV  # the viewer's quantiser and WebP writer (models.html decodes the same codes)
import rangefetch as RF  # ECMWF open data, Google Cloud mirror only
import snowband as sb  # the shared, validated S3 transport
import upper as UPR  # per-frame height codes (q_auto) and the downward 2-PVU search

S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.{f}.0p50.f{s:03d}"
DATAMART = (
    "https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{h}/{s:03d}/{d}T{h}Z_MSC_GDPS_{v}_LatLon0.15_PT{s:03d}H.grib2"
)
GFS_LEVELS = (100, 70, 50, 40, 30, 20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4)
GDPS_COL = (400, 350, 300, 275, 250, 225, 200, 175, 150, 100, 50, 30, 20, 10, 5, 1)
GEOS = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/inst3_3d_asm_Np/inst3_3d_asm_Np.{d}_{h}"
GEOS_LEV = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300,
            250, 200, 150, 100, 70, 50, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1)  # fmt: skip
GEOS_COL = GEOS_LEV[GEOS_LEV.index(400) : GEOS_LEV.index(0.4) + 1]  # 22 levels, one contiguous slab
UA = {"User-Agent": "scorvec.com data pipeline (+https://scorvec.com)"}
MODELS = {
    "gfs": dict(
        label="GFS",
        source="NOAA GFS (FV3, 127 layers to 80 km) on its 0.5 deg output grid: NOAA Open Data Dissemination, AWS S3",
        steps=tuple(range(0, 385, 12)),
        levels=GFS_LEVELS,  # the maps' pressure levels, hPa
        col_a=(400, 350, 300, 250, 200, 150, *GFS_LEVELS),  # the PV columns' levels in pgrb2.0p50 ...
        col_b=(375, 325, 275, 225, 175, 125),  # ... and in pgrb2b.0p50
        thetas=(350, 400, 475, 600, 850, 1200, 1600),
        absv=(50, 10),
        omega=True,
    ),
    "gdps": dict(
        label="GDPS",
        source="ECCC GDPS (GEM, 15 km) on its 0.15 deg grid averaged to 0.45 deg: MSC Datamart (Data Source: Environment and Climate Change Canada)",
        steps=tuple(range(0, 241, 12)),
        levels=(100, 50, 30, 20, 10, 5, 1),
        col=GDPS_COL,
        thetas=(350, 400, 475, 600, 850),
        absv=(250, 200),
        omega=False,
    ),
    "geos": dict(
        label="GEOS FP",
        source="NASA GMAO GEOS FP on its 0.25 x 0.3125 deg grid, every second point, at 0.5 deg; its own Ertel PV: NCCS OPeNDAP",
        steps=tuple(range(0, 241, 12)),
        steps_by_hour={"12": tuple(range(0, 121, 12))},  # the 12 UTC forecast runs 5 days
        levels=tuple(p for p in GFS_LEVELS if p in GEOS_LEV),
        col=GEOS_COL,
        thetas=(350, 400, 475, 600, 850, 1200, 1600),
        absv=(),
        omega=False,
        epv=True,
    ),
    "aifs": dict(
        label="AIFS single",
        source="ECMWF AIFS single (the deterministic AIFS) on its 0.25 deg grid, smoothed to 0.5 deg: ECMWF open data (CC BY 4.0), Google Cloud mirror",
        steps=tuple(range(0, 361, 12)),
        every=6,  # hours between the runs planned (the others: 12)
        levels=(100, 50, 10),
        col=(100, 50, 10),
        thetas=(),
        absv=(),
        omega=False,
    ),
}
QC_STEPS = (0, 120, 240)  # steps with the model's own vorticity and 2-PVU surface fetched as a check
PV2 = {"nh": "PV=2e-06 (Km^2/kg/s) surface", "sh": "PV=-2e-06 (Km^2/kg/s) surface"}  # the GFS's names
G0, CP, KAPPA, OMEGA, A_E, RD = 9.80665, 1004.64, 0.2857, 7.2921e-5, 6.371229e6, 287.04
PAD = 4  # the columns repeated on each side of the date line
T_OFF, T_STEP = 160.0, 0.6  # temperature code, K
PV_OFF, PV_STEP, PV_REF = -20.0, 0.5, 475.0  # Lait-scaled PV code, PVU at the PV_REF scale
P_LO, P_HI = 0.2, 600.0  # pressure code of the isentropic surfaces, log, hPa
WIND_STEPS = (0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)  # m/s, the finest that spans the frame
CAP_LAT, CAP_THETAS = 60.0, (475, 850)
HEMIS = ("nh", "sh")
HPA_DAY = 864.0  # Pa/s -> hPa/day
VV_SIGMA_KM, VV_HALF_H = 600.0, 36  # the vertical motion's smoothing: Gaussian sigma along the sphere; half the window
VV_REF = HERE / "strat_omega_ranges.json"  # its colour range per level (omega-clim)
CLIM_FROM, CLIM_MONTHS, CLIM_LAT = (2022, 7), 48, 20.0  # omega-clim: the 15th of each month from July 2022
CLIM_BINS = np.linspace(-4.0, 2.0, 601)  # log10 |omega|, hPa/day
# a standard atmosphere's heights (m) at the GDPS's levels, for its units check (the GFS run's global means of
# 2026-10-24, rounded): a field off by more than 20 % from these is not in metres
Z_STD = {400: 7200, 350: 8100, 300: 9200, 275: 9800, 250: 10400, 225: 11000, 200: 11800, 175: 12600, 150: 13600}
Z_STD |= {100: 16300, 50: 20500, 30: 23700, 20: 26300, 10: 30900, 5: 35600, 1: 47500}


def steps_of(model, hh):
    """The model's forecast steps (h) for a cycle starting at hh UTC."""
    M = MODELS[model]
    return M.get("steps_by_hour", {}).get(hh, M["steps"])


def lev_key(p):
    """10 -> "10", 0.4 -> "0p4" (the files' level names)."""
    return f"{p:g}".replace(".", "p")


def data_dir(model):
    return f"assets/models/data/strat/{model}"


def gfs_url(date, hh, step, b=False):
    return S3.format(d=date, h=hh, f="pgrb2b" if b else "pgrb2", s=step)


def gdps_url(date, hh, step, var):
    return DATAMART.format(d=date, h=hh, s=step, v=var)


def col_of(model):
    """The PV column's levels, bottom (400 hPa) to top."""
    M = MODELS[model]
    return sorted(M.get("col") or M["col_a"] + M["col_b"], reverse=True)


# ── grids ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def hemi(a, h):
    """A global field (rows 90 N -> 90 S with the equator in the middle row, columns from 0 E) -> one hemisphere, rows
    south -> north (equator to pole or pole to equator), columns from 180 W with PAD columns repeated on each side."""
    ny = a.shape[-2]
    eq = ny // 2
    rows = a[..., eq::-1, :] if h == "nh" else a[..., ny - 1 : eq - 1 : -1, :]
    nx = a.shape[-1]
    w = np.concatenate([rows[..., nx // 2 :], rows[..., : nx // 2]], axis=-1)  # from 180 W
    return np.concatenate([w[..., -PAD:], w, w[..., :PAD]], axis=-1)


def grid_desc(h, nx, ny, di):
    """The page's grid: regular lat-lon, row j = 0 at the south (la1), column i = 0 at lo1; w0 = where the page starts
    counting longitude (columns PAD.. hold 180 W .. 180 E once each)."""
    return dict(
        type="llw",
        nx=nx,
        ny=ny,
        lo1=round(-180.0 - PAD * di, 6),
        la1=0.0 if h == "nh" else -90.0,
        di=di,
        dj=di,
        w0=-180.0,
    )


def half(a):
    """0.25 -> 0.5 deg: one 1-2-1 pass, then every second point (the poles and 0 E kept)."""
    return _121(a)[..., ::2, ::2]


def coarsen3(a):
    """3 x 3 cell means at every third cell (columns periodic, the pole rows kept): 0.15 -> 0.45 deg, row 0 and
    column 0 staying where they are."""
    b = (np.roll(a, 1, -1) + a + np.roll(a, -1, -1)) / 3.0
    c = b.copy()
    c[..., 1:-1, :] = (b[..., :-2, :] + b[..., 1:-1, :] + b[..., 2:, :]) / 3.0
    return c[..., ::3, ::3]


# ── physics ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _121(a):
    """One 1-2-1 pass in latitude (edges kept) and longitude (periodic)."""
    b = a.copy()
    b[..., 1:-1, :] = 0.25 * a[..., :-2, :] + 0.5 * a[..., 1:-1, :] + 0.25 * a[..., 2:, :]
    return 0.25 * np.roll(b, 1, -1) + 0.5 * b + 0.25 * np.roll(b, -1, -1)


def poles(a, lat):
    """The rows at the poles (where the longitude derivatives are undefined) take the zonal mean of the ring next to
    them."""
    for j in np.flatnonzero(np.abs(lat) > 89.99):
        k = j + 1 if j + 1 < lat.size and abs(lat[j + 1]) < 89.99 else j - 1
        a[..., j, :] = np.nanmean(a[..., k, :], axis=-1, keepdims=True)
    return a


def ertel_pv(T, U, V, plev, lat):
    """Ertel PV (PVU) and zeta + f on pressure levels of a global regular lat-lon grid (module docstring). T, U, V:
    (nlev, ny, nx), earth-relative winds; plev hPa (any order); lat (ny,) degrees, the grid's rows. -> pv, absv, theta"""
    p = np.asarray(plev, np.float64)[:, None, None] * 100.0
    lnp = np.log(p[:, 0, 0])
    th = T * (1e5 / p) ** KAPPA
    phi = np.radians(np.asarray(lat, np.float64))
    cph = np.cos(phi)[:, None]
    cph[np.abs(lat) > 89.99] = np.nan  # the poles: no longitude derivative (filled from the next ring)
    dlam = math.radians(360.0 / T.shape[-1])

    def ddx(q):  # (1 / a cos phi) dq/dlambda, centred and periodic
        return (np.roll(q, -1, -1) - np.roll(q, 1, -1)) / (2 * dlam * A_E * cph)

    def ddy(q):  # (1 / a) dq/dphi
        return np.gradient(q, phi, axis=-2) / A_E

    def ddp(q):  # dq/dp = (1 / p) dq/dln p on the actual levels
        return np.gradient(q, lnp, axis=0) / p

    with np.errstate(invalid="ignore", divide="ignore"):
        absv = ddx(V) - ddy(U * np.nan_to_num(cph)) / cph + (2 * OMEGA * np.sin(phi))[:, None]
        pv = -G0 * (absv * ddp(th) - ddp(V) * ddx(th) + ddp(U) * ddy(th)) * 1e6
    return poles(_121(poles(pv, lat)), lat), poles(absv, lat), th


def on_theta(th, fields, theta0):
    """Each of `fields` (nlev, ny, nx) on the isentrope theta0, linear in ln theta between the two levels around it
    (theta increasing upward: the first crossing counted from the bottom); NaN where theta0 is outside the column."""
    lt = np.log(th)
    k = (th < theta0).sum(axis=0) - 1
    bad = (k < 0) | (k > th.shape[0] - 2)
    k = np.clip(k, 0, th.shape[0] - 2)[None]
    a, b = np.take_along_axis(lt, k, 0)[0], np.take_along_axis(lt, k + 1, 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (math.log(theta0) - a) / (b - a)
    bad |= ~((w >= 0) & (w <= 1))
    out = []
    for f in fields:
        x0, x1 = np.take_along_axis(f, k, 0)[0], np.take_along_axis(f, k + 1, 0)[0]
        out.append(np.where(bad, np.nan, x0 + w * (x1 - x0)))
    return out


def smooth_sphere(a, lat, sigma_km=VV_SIGMA_KM):
    """A Gaussian of standard deviation sigma_km along the sphere on the last two axes of a global field (rows
    90 N -> 90 S, periodic columns): along each latitude circle in Fourier space (exact on the circle: zonal wavenumber
    k is kept by exp(-(k sigma / (a cos phi))^2 / 2), so near the poles only the zonal mean is left), then along the
    meridians, continued over each pole onto the opposite meridian."""
    ny, nx = a.shape[-2:]
    sig = sigma_km * 1e3
    k = np.arange(nx // 2 + 1)
    cph = np.maximum(np.cos(np.radians(lat)), 1e-9)[:, None]
    b = np.fft.irfft(np.fft.rfft(a, axis=-1) * np.exp(-0.5 * (k[None, :] * sig / (A_E * cph)) ** 2), n=nx, axis=-1)
    sr = sig / (A_E * math.radians(abs(lat[1] - lat[0])))  # sigma in rows
    half = math.ceil(4 * sr)

    def over(rows):  # the rows beyond a pole: the same latitudes on the opposite meridian
        return np.roll(rows[..., ::-1, :], nx // 2, axis=-1)

    ext = np.concatenate([over(b[..., 1 : half + 1, :]), b, over(b[..., ny - 1 - half : ny - 1, :])], axis=-2)
    w = np.exp(-0.5 * (np.arange(-half, half + 1) / sr) ** 2)
    out = np.zeros_like(b)
    for i, wi in enumerate(w / w.sum()):
        out += wi * ext[..., i : i + ny, :]
    return out


def window(steps, t, half=VV_HALF_H):
    """The centred running mean at step t of the 12-hourly `steps`: {step: weight}, trapezoidal over +-h hours (weight
    1 inside, 1/2 at the ends; with 12-hourly steps this cancels the diurnal tide exactly), h = `half`, or less near the
    ends of the run so the window stays centred; {} where h would be under 12 h (the run's first and last step)."""
    h = min(half, t - steps[0], steps[-1] - t)
    if h < 12:
        return {}
    w = {s: 1.0 if abs(s - t) < h else 0.5 for s in steps if abs(s - t) <= h}
    tot = sum(w.values())
    return {s: v / tot for s, v in w.items()}


def lait(pv, theta0, sign):
    """PV scaled to the PV_REF surface (Lait 1994, theta^(-9/2)), sign reversed in the Southern Hemisphere."""
    return sign * pv * (theta0 / PV_REF) ** -4.5


def cap_mean(x, lat, logmean=False):
    """cos-weighted mean poleward of CAP_LAT of a hemisphere's (rows, columns) field, the padding columns left out."""
    rows = np.abs(lat) >= CAP_LAT
    w = np.cos(np.radians(lat[rows]))
    v = x[rows][:, PAD:-PAD]
    v = np.log(v) if logmean else v
    m = float((np.nanmean(v, axis=1) * w).sum() / w.sum())
    return math.exp(m) if logmean else m


# ── encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────
def q_logp(p):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.rint((np.log(p) - math.log(P_LO)) / (math.log(P_HI) - math.log(P_LO)) * 254)
    return np.where(np.isfinite(p), np.clip(c, 0, 254), 255).astype(np.uint8)


def image(*chans):
    """Channels (rows south -> north) -> a 3-channel uint8 image, row 0 at the north (the empty third channel of a
    two-channel frame is zero)."""
    while len(chans) < 3:
        chans = (*chans, np.zeros_like(chans[0]))
    return np.dstack(chans)[::-1]


def write(outdir, h, name, step, arr):
    d = Path(outdir) / h / name
    d.mkdir(parents=True, exist_ok=True)
    b = MV.webp(arr)
    (d / f"{step:03d}.webp").write_bytes(b)
    return len(b)


# ── fetching: every model -> {(name, level): global array, rows 90 N -> 90 S, columns from 0 E}, lat, info ───────────
#   names: t (K), z (m), u, v (m/s), w (omega, Pa/s), absv (1/s); ("pv2", hemi): the model's own 2-PVU pressure (Pa)
_FETCH: dict = {}


def fetcher():
    if "f" not in _FETCH:
        _FETCH["f"] = sb.Fetcher("S3 GFS", 16)
    return _FETCH["f"]


def fetch_gfs(date, hh, step):
    f = fetcher()
    M = MODELS["gfs"]
    n0, b0 = f.nreq, f.nbytes
    names = {"TMP": "t", "HGT": "z", "UGRD": "u", "VGRD": "v"}

    def want(levels):
        return {(v, f"{p:g} mb") for v in names for p in levels}

    wa = want(M["col_a"]) | {("VVEL", f"{p:g} mb") for p in M["levels"]}
    if step in QC_STEPS:
        wa |= {("ABSV", f"{p} mb") for p in M["absv"]} | {("PRES", s) for s in PV2.values()}
    a, grid = sb.fetch_messages(f, gfs_url(date, hh, step), wa)
    b, _ = sb.fetch_messages(f, gfs_url(date, hh, step, b=True), want(M["col_b"]))
    lat = grid["lat"][:, 0].astype(np.float64)
    if a[("TMP", "10 mb")].shape != (361, 720) or abs(lat[0] - 90) > 1e-6 or abs(lat[-1] + 90) > 1e-6:
        raise RuntimeError(f"unexpected grid {a[('TMP', '10 mb')].shape}, lat {lat[0]}..{lat[-1]}")
    out = {}
    for (v, lev), x in (a | b).items():
        if v == "PRES":
            out[("pv2", "nh" if lev == PV2["nh"] else "sh")] = x
        else:
            out[({**names, "VVEL": "w", "ABSV": "absv"}[v], float(lev[:-3]))] = x
    return out, lat, dict(requests=f.nreq - n0, bytes=f.nbytes - b0, di=0.5)


def decode_ll(buf):
    """One regular lat-lon GRIB2 message -> (values, rows 90 N -> 90 S, columns from 0 E; its GRIB units)."""
    import eccodes as ec

    g = ec.codes_new_from_message(buf)
    try:

        def gv(k):
            return ec.codes_get(g, k)

        if gv("gridType") != "regular_ll":
            raise ValueError(f"unexpected grid {gv('gridType')}")
        ni, nj, di = gv("Ni"), gv("Nj"), gv("iDirectionIncrementInDegrees")
        v = ec.codes_get_values(g).reshape(nj, ni).astype(np.float64)
        if gv("bitmapPresent"):
            v[v == gv("missingValue")] = np.nan
        south_first, lo0 = gv("jScansPositively") == 1, gv("longitudeOfFirstGridPointInDegrees")
        la0, la1 = gv("latitudeOfFirstGridPointInDegrees"), gv("latitudeOfLastGridPointInDegrees")
        whole_circle = abs(ni * di - 360.0) < 1e-6 or abs((ni - 1) * di - 360.0) < 1e-6
        if abs(abs(la0) - 90) > 1e-6 or abs(la0 + la1) > 1e-6 or not whole_circle:
            raise ValueError(f"not a global pole-to-pole grid: lat {la0}..{la1}, {ni} x {di} deg")
        try:
            units = gv("units")
        except Exception:  # noqa: BLE001
            units = ""
    finally:
        ec.codes_release(g)
    if south_first:
        v = v[::-1]
    if abs((ni - 1) * di - 360.0) < 1e-6:  # a repeated column at 360 E
        v, ni = v[:, :-1], ni - 1
    return np.roll(v, -round(((-lo0) % 360.0) / di), axis=1), units


GDPS_VAR = {"t": "AirTemp", "z": "GeopotentialHeight", "u": "WindU", "v": "WindV"}


def fetch_gdps(date, hh, step):
    M = MODELS["gdps"]
    jobs = {(n, float(p)): f"{var}_IsbL-{p:04d}" for n, var in GDPS_VAR.items() for p in M["col"]}
    if step in QC_STEPS:
        jobs |= {("absv", float(p)): f"AbsoluteVorticity_IsbL-{p:04d}" for p in M["absv"]}
        jobs[("pv2", "both")] = "Pressure_PVU-2"
    urls = {k: gdps_url(date, hh, step, v) for k, v in jobs.items()}
    with ThreadPoolExecutor(DM.WORKERS) as ex:
        bodies = dict(zip(urls, ex.map(DM.get, urls.values())))
    gone = [urls[k].rsplit("/", 1)[-1] for k, b in bodies.items() if b is None]
    if gone:
        raise RuntimeError(f"{len(gone)} GDPS files missing, e.g. {gone[0]}")
    out, units = {}, {}
    for k, body in bodies.items():
        x, u = decode_ll(body)
        out[k] = coarsen3(x)
        units[k] = u
    n = out[("t", 10.0)].shape
    if n != (401, 800):
        raise RuntimeError(f"unexpected GDPS grid {n} after coarsening")
    lat = np.linspace(90.0, -90.0, n[0])
    info = dict(requests=len(urls), bytes=sum(len(b) for b in bodies.values()), di=0.45, units=gdps_units(out, units))
    if ("pv2", "both") in out:
        pv2 = out.pop(("pv2", "both"))
        out[("pv2", "nh")] = out[("pv2", "sh")] = pv2
    return out, lat, info


def gdps_units(out, grib_units):
    """Bring every GDPS field to the units of the module (K, m, m/s, 1/s, Pa) by its magnitude, IN PLACE, and say what
    was found; anything that matches no expected unit stops the step."""
    rep = {}
    t = np.nanmedian(np.stack([out[("t", p)] for p in (10.0, 50.0)]))
    if not 150.0 < t < 330.0:
        if -130.0 < t < 60.0:
            for k in [k for k in out if k[0] == "t"]:
                out[k] += 273.15
            rep["t"] = "deg C, converted to K"
        else:
            raise RuntimeError(f"GDPS temperature median {t:.1f}: neither K nor deg C")
    else:
        rep["t"] = "K"
    zr = {}
    for (n, p), x in out.items():
        if n != "z":
            continue
        r = float(np.nanmedian(x)) / Z_STD[int(p)]
        if 0.8 < r < 1.25:
            f = 1.0
        elif 0.08 < r < 0.125:
            f = 10.0  # decametres
        elif 7.8 < r < 12.3:
            f = 1.0 / G0  # geopotential, m2/s2
        else:
            raise RuntimeError(f"GDPS height at {p:g} hPa: median {np.nanmedian(x):.0f} against ~{Z_STD[int(p)]} m")
        zr[f] = zr.get(f, 0) + 1
        out[(n, p)] = x * f
    rep["z"] = {1.0: "m", 10.0: "dam, converted to m", 1.0 / G0: "geopotential, divided by g"}.get(
        zr and max(zr, key=zr.get)
    )
    if len(zr) > 1:
        rep["z_mixed"] = {str(k): v for k, v in zr.items()}
    rep["z10_median_m"] = round(float(np.nanmedian(out[("z", 10.0)])))
    umax = max(float(np.nanmax(np.abs(out[(n, p)]))) for (n, p) in out if n in ("u", "v"))
    if not 20.0 < umax < 300.0:
        raise RuntimeError(f"GDPS wind maximum {umax:.0f}: not m/s")
    rep["uv"] = f"m/s ({grib_units[('u', 10.0)]}; largest |u|, |v| {umax:.0f})"
    if ("pv2", "both") in out:
        pm = float(np.nanmedian(out[("pv2", "both")]))
        if 50.0 < pm < 1000.0:
            out[("pv2", "both")] *= 100.0
            rep["pv2"] = "hPa, converted to Pa"
        elif 5e3 < pm < 1e5:
            rep["pv2"] = "Pa"
        else:
            raise RuntimeError(f"GDPS 2-PVU pressure median {pm:.0f}: neither Pa nor hPa")
    for k in [k for k in out if k[0] == "absv"]:  # f at 45 deg is 1.03e-4 1/s: a field 10^n times that is in 10^-n
        n = round(math.log10(max(float(np.nanmedian(np.abs(out[k]))), 1e-30) / 1e-4))
        if n:
            out[k] *= 10.0**-n
        rep["absv"] = "1/s" if not n else f"1e{-n} 1/s, rescaled"
    return rep


AIFS_PARAM = {"t": "t", "gh": "z", "u": "u", "v": "v"}  # open-data param -> this module's name


def fetch_aifs(date, hh, step):
    M = MODELS["aifs"]
    ((rel, idx),) = RF.oper_files(date, hh, "aifs-single", step)
    want = RF.select(idx, param=list(AIFS_PARAM), levelist=list(M["levels"]))
    if len(want) != len(AIFS_PARAM) * len(M["levels"]):
        raise RuntimeError(
            f"AIFS single +{step} h: {len(want)} of {len(AIFS_PARAM) * len(M['levels'])} messages indexed"
        )
    ranges = RF.coalesce(want)
    blob = RF.fetch_ranges(rel, ranges)  # the messages, concatenated in the ranges' order
    out, pos = {}, 0
    for e in (e for _, _, es in ranges for e in es):
        x, _ = decode_ll(blob[pos : pos + e["_length"]])
        pos += e["_length"]
        out[(AIFS_PARAM[e["param"]], float(e["levelist"]))] = half(x)
    n = out[("t", 10.0)].shape
    if n != (361, 720):
        raise RuntimeError(f"unexpected AIFS single grid {n} after halving")
    for p in M["levels"]:  # units by magnitude, as the GDPS's: K and metres, or the step is not used
        t, z = float(np.nanmedian(out[("t", float(p))])), float(np.nanmedian(out[("z", float(p))]))
        if not 150.0 < t < 330.0 or not 0.8 < z / Z_STD[p] < 1.25:
            raise RuntimeError(
                f"AIFS single {p} hPa: median temperature {t:.1f}, height {z:.0f} m (~{Z_STD[p]} m expected)"
            )
    lat = np.linspace(90.0, -90.0, n[0])
    return out, lat, dict(requests=len(ranges) + 1, bytes=len(blob), di=0.5)


def dods(url: str, shape: tuple, tries: int = 4) -> dict:
    """{name: array} of a DAP2 binary response (.dods) of float32 Grid variables of one shape. The server sends them in
    the dataset's order, not the request's: the DDS before "Data:" names them; after it each Grid is its array then its
    map vectors (float64), every vector led by its length twice (XDR)."""
    import re

    import requests

    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=300)
            r.raise_for_status()
            body = r.content
            cut = body.index(b"\nData:\n")
            names = re.findall(r"ARRAY:\s*Float32 (\w+)\[", body[:cut].decode("ascii", "replace"))
            i, out = cut + 7, {}
            for name in names:
                n = struct.unpack(">I", body[i : i + 4])[0]
                if n != math.prod(shape):
                    raise RuntimeError(f"DAP array {name} of {n} values, {math.prod(shape)} expected")
                out[name] = np.frombuffer(body, ">f4", n, i + 8).reshape(shape).astype(np.float64)
                i += 8 + 4 * n
                for _ in shape:  # the Grid's maps, one per dimension
                    i += 8 + 8 * struct.unpack(">I", body[i : i + 4])[0]
            return out
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"  DAP read failed ({str(e)[:100]}); again in {30 * (k + 1)} s", flush=True)
            time.sleep(30 * (k + 1))


GEOS_VARS = {"t": "t", "h": "z", "epv": "epv", "u": "u", "v": "v"}  # GEOS FP name -> this module's


def fetch_geos(date, hh, step):
    """One step of the GEOS FP forecast: every variable on GEOS_COL at every second point (0.5 x 0.625 deg), rows
    flipped to 90 N first, columns rolled to 0 E and interpolated (periodic, linear) to 0.5 deg."""
    k0, k1 = GEOS_LEV.index(GEOS_COL[0]), GEOS_LEV.index(GEOS_COL[-1])
    j = step // 3
    hyper = f"[{j}:{j}][{k0}:{k1}][0:2:720][0:2:1151]"
    url = GEOS.format(d=date, h=hh) + ".dods?" + ",".join(v + hyper for v in GEOS_VARS)
    shape = (1, k1 - k0 + 1, 361, 576)
    arrs = dods(url, shape)
    x = np.arange(720) * 0.5 / 0.625  # 0.5-deg columns from 0 E on the 0.625-deg grid rolled to 0 E
    i0 = np.floor(x).astype(int)
    w = x - i0
    out = {}
    for name, mine in GEOS_VARS.items():
        raw = arrs[name]
        b = raw[0, :, ::-1]  # (lev, lat 90 N -> 90 S, lon from 180 W)
        b[np.abs(b) > 1e10] = np.nan  # below ground
        b = np.roll(b, -288, axis=-1)  # 180 W + 288 x 0.625 deg = 0 E
        b = b[..., i0] * (1 - w) + b[..., (i0 + 1) % 576] * w
        for i, p in enumerate(GEOS_COL):
            out[(mine, float(p))] = b[i]
    lat = np.linspace(90.0, -90.0, 361)
    for p in (100, 50, 10, 1):  # units by magnitude, as the GDPS's: K, metres, SI PV, or the step is not used
        t, z = float(np.nanmedian(out[("t", float(p))])), float(np.nanmedian(out[("z", float(p))]))
        if not 150.0 < t < 330.0 or not 0.8 < z / Z_STD[p] < 1.25:
            raise RuntimeError(
                f"GEOS FP {p} hPa: median temperature {t:.1f}, height {z:.0f} m (~{Z_STD[p]} m expected)"
            )
    q = float(np.nanmedian(np.abs(out[("epv", 50.0)][lat >= 60])))
    if not 3e-6 < q < 3e-4:
        raise RuntimeError(f"GEOS FP 50 hPa polar |EPV| median {q:.2e}: not K m2 kg-1 s-1")
    return out, lat, dict(requests=1, bytes=sum(a.size for a in arrs.values()) * 4, di=0.5)


def geos_complete(date, hh, last) -> bool:
    """The forecast's last step holds data at its top level (NCCS lists a collection before its last steps)."""
    j, k = last // 3, GEOS_LEV.index(GEOS_COL[-1])
    try:
        a = dods(GEOS.format(d=date, h=hh) + f".dods?t[{j}:{j}][{k}:{k}][360:360][0:0]", (1, 1, 1, 1), tries=1)["t"]
    except Exception:  # noqa: BLE001  (not posted yet)
        return False
    return bool(np.isfinite(a).all() and (np.abs(a) < 1e10).all())


# ── one step (a worker process: fetch, compute, encode) ──────────────────────────────────────────────────────────────
def process_step(model, date, hh, step, outdir):
    """-> (step, {file: {hemi: q}}, bytes, checks, (requests, bytes fetched), extras)"""
    t0 = time.time()
    M = MODELS[model]
    msgs, lat, info = {"gfs": fetch_gfs, "gdps": fetch_gdps, "aifs": fetch_aifs, "geos": fetch_geos}[model](
        date, hh, step
    )
    t_fetch = time.time() - t0
    col = col_of(model)

    def stack(v):
        return np.stack([msgs[(v, float(p))] for p in col]).astype(np.float64)

    T, U, V, Z = stack("t"), stack("u"), stack("v"), stack("z")
    if M["thetas"] and M.get("epv"):  # the model's own Ertel PV (SI), smoothed as ertel_pv smooths its own
        th = T * (1e5 / (np.asarray(col, np.float64)[:, None, None] * 100.0)) ** KAPPA
        pv = poles(_121(poles(stack("epv") * 1e6, lat)), lat)
    elif M["thetas"]:
        pv, absv, th = ertel_pv(T, U, V, col, lat)
    if M["thetas"]:
        lnp = np.log(np.asarray(col, np.float64))[:, None, None] * np.ones_like(T)
        mg = (CP * T + G0 * Z) / G0  # the Montgomery streamfunction over g, metres
    meta, nbytes, extras = {}, 0, {}
    hlat = {h: hemi(lat[:, None], h)[:, 0] for h in HEMIS}

    for p in M["levels"]:
        k = col.index(p)
        for h in HEMIS:
            t, z, u, v = (hemi(x[k], h) for x in (T, Z, U, V))
            zc, zq = UPR.q_auto(z)
            nbytes += write(outdir, h, f"t{lev_key(p)}", step, image(MV.q_lin(t, T_OFF, T_STEP), zc))
            wc, wq = UPR.q_auto(np.stack([u, v]), steps=WIND_STEPS)
            nbytes += write(outdir, h, f"w{lev_key(p)}", step, image(wc[0], wc[1]))
            meta.setdefault(f"t{lev_key(p)}", {})[h] = zq
            meta.setdefault(f"w{lev_key(p)}", {})[h] = wq
    if step == 0:  # the levels' mean height (the page's labels): cos-weighted over the globe
        w = np.cos(np.radians(lat))[:, None] * np.ones((1, T.shape[-1]))
        extras["zmean_km"] = {
            lev_key(p): round(float((Z[col.index(p)] * w).sum() / w.sum()) / 1000, 1) for p in M["levels"]
        }

    cap = {h: {} for h in HEMIS}
    for th0 in M["thetas"]:
        q_pv, q_m, q_lnp = on_theta(th, (pv, mg, lnp), th0)
        for h in HEMIS:
            sign = 1.0 if h == "nh" else -1.0
            x, m, pp = hemi(q_pv, h), hemi(q_m, h), np.exp(hemi(q_lnp, h))
            mc, mq = UPR.q_auto(m)
            img = image(MV.q_lin(lait(x, th0, sign), PV_OFF, PV_STEP), mc, q_logp(pp))
            nbytes += write(outdir, h, f"pv{th0}", step, img)
            meta.setdefault(f"pv{th0}", {})[h] = mq
            if th0 in CAP_THETAS:
                cap[h][str(th0)] = [round(cap_mean(sign * x, hlat[h]), 1), round(cap_mean(pp, hlat[h], True), 1)]
            if step == 0:  # the surfaces' polar-cap pressure (the page's labels)
                extras.setdefault("cap_hpa", {}).setdefault(h, {})[str(th0)] = round(cap_mean(pp, hlat[h], True), 1)

    if M["omega"]:
        W = np.stack([msgs[("w", float(p))] for p in M["levels"]]).astype(np.float64)
        np.save(Path(outdir) / f"_vv_{step:03d}.npy", smooth_sphere(W * HPA_DAY, lat).astype(np.float32))  # vv_level

    checks = {"cap": cap}
    if step in QC_STEPS and M["absv"]:
        checks.update(check_absv(msgs, absv, col, lat, M["absv"]))
        checks.update(check_pv2(msgs, pv, col, lat))
        if model == "gfs":
            checks.update(check_gdps_levels(T, U, V, col, lat, pv, th, M["thetas"]))
    if "units" in info:
        checks["units"] = info["units"]
    checks["secs"], checks["fetch_secs"] = round(time.time() - t0, 1), round(t_fetch, 1)
    extras["di"] = info["di"]
    return step, meta, nbytes, checks, (info["requests"], info["bytes"]), extras


def process_step_retry(model, date, hh, step, outdir, tries=3):
    """process_step, tried again after a pause (a dropped connection); the step number alone when every try failed."""
    for k in range(tries):
        try:
            return process_step(model, date, hh, step, outdir)
        except Exception as e:  # noqa: BLE001
            print(f"  +{step:03d} h attempt {k + 1}: {e!r}"[:400], flush=True)
            if k < tries - 1:
                time.sleep(30)
    return step


def vv_level(outdir, k, p, x_range, steps):
    """One level's vertical-motion files: the running mean of the smoothed steps (process_step's _vv files), both
    hemispheres. -> bytes written"""
    one = {s: np.load(Path(outdir) / f"_vv_{s:03d}.npy", mmap_mode="r") for s in steps}
    nbytes = 0
    for t in steps:
        w = window(steps, t)
        if not w:
            continue
        x = sum(wt * one[s][k].astype(np.float64) for s, wt in w.items())
        for h in HEMIS:
            img = image(MV.q_lin(hemi(x, h), -2 * x_range, 4 * x_range / 254))
            nbytes += write(outdir, h, f"o{lev_key(p)}", t, img)
    return nbytes


def vv_ranges():
    """omega-clim's file and the vertical motion's colour range per GFS level from it, or (None, None)."""
    try:
        doc = json.loads(VV_REF.read_text())
        return doc, {lev_key(p): float(doc["levels"][lev_key(p)]["range"]) for p in GFS_LEVELS}
    except Exception:  # noqa: BLE001
        return None, None


# ── checks ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def fin(x, nd):
    """x rounded, or None where it is not finite (the manifest is JSON: no NaN)."""
    return round(float(x), nd) if np.isfinite(x) else None


def band(lat, lo, hi):
    a = np.abs(lat)
    return (a >= lo) & (a <= hi)


def check_absv(msgs, absv, col, lat, levels):
    """zeta + f from this module's derivatives against the model's own absolute vorticity, 20-85 deg, both
    hemispheres together."""
    out = {}
    rows = band(lat, 20, 85)
    for p in levels:
        ours, theirs = absv[col.index(p)][rows].ravel(), msgs[("absv", float(p))][rows].ravel().astype(np.float64)
        ok = np.isfinite(ours) & np.isfinite(theirs)
        a, b = ours[ok], theirs[ok]
        out[f"absv/{p}"] = dict(
            n=int(ok.sum()),
            r=fin(np.corrcoef(a, b)[0, 1], 4),
            rms_ratio=fin(np.sqrt((a**2).mean() / (b**2).mean()), 4),
            mae_rel=fin(np.abs(a - b).mean() / np.abs(b).mean(), 4),
        )
    return out


def check_pv2(msgs, pv, col, lat):
    """The 2-PVU pressure from this PV (|PV|, searched downward from 100 hPa through the levels to 400 hPa, as
    models.html's dynamic tropopause) against the model's own, 30-80 deg, where both lie between 100 and 400 hPa."""
    ks = [col.index(p) for p in sorted(p for p in col if 100 <= p <= 400)]  # 100 hPa first
    levs = [col[k] for k in ks]
    stack = np.abs(pv[ks])
    dt_ = UPR.dt_downward(stack, stack, stack, stack, levs)
    out = {}
    for h, sel in (("nh", lat > 0), ("sh", lat < 0)):
        rows = sel & band(lat, 30, 80)
        ours = np.where(dt_["top"] | dt_["none"], np.nan, dt_["p"])[rows]
        theirs = msgs[("pv2", h)][rows] / 100.0
        ok = np.isfinite(ours) & np.isfinite(theirs) & (theirs >= 100) & (theirs <= 400)
        d = (ours[ok] - theirs[ok]).astype(np.float64)
        out[f"pv2/{h}"] = dict(n=int(ok.sum()))
        if ok.sum() > 100:
            out[f"pv2/{h}"] |= dict(
                bias_hpa=fin(d.mean(), 1),
                mae_hpa=fin(np.abs(d).mean(), 1),
                r=fin(np.corrcoef(ours[ok], theirs[ok])[0, 1], 3),
                within_25hpa=fin((np.abs(d) <= 25).mean(), 3),
            )
    return out


def check_gdps_levels(T, U, V, col, lat, pv, th, thetas):
    """The GFS's PV recomputed on exactly the GDPS's levels against its own on all 27, on every isentrope: the pattern
    correlation poleward of 20 deg and the ratio of the polar-cap means (what the GDPS's coarser levels cost)."""
    ks = [col.index(p) for p in col if p in GDPS_COL]
    pv_s, _, th_s = ertel_pv(T[ks], U[ks], V[ks], [col[k] for k in ks], lat)
    rows = band(lat, 20, 90)
    out = {}
    for th0 in thetas:
        a, b = on_theta(th, (pv,), th0)[0], on_theta(th_s, (pv_s,), th0)[0]
        ok = np.isfinite(a) & np.isfinite(b) & rows[:, None]
        e = dict(n=int(ok.sum()))
        if ok.sum() > 100:
            e["r"] = fin(np.corrcoef(a[ok], b[ok])[0, 1], 4)
            for h in HEMIS:
                hl = hemi(lat[:, None], h)[:, 0]
                e[f"cap_ratio_{h}"] = fin(cap_mean(hemi(b, h), hl) / cap_mean(hemi(a, h), hl), 4)
        out[f"gdpslev/{th0}"] = e
    return out


def summarise(checks):
    """The per-step checks -> the manifest's: medians over the check steps (and min, max), and the units the GDPS's
    fields were found in (the first step's; every step must agree)."""
    out = {}
    qc = [checks[s] for s in sorted(checks, key=int) if "pv2/nh" in checks[s]]
    keys = sorted({(k, m) for c in qc for k, v in c.items() if isinstance(v, dict) and "n" in v for m in v if m != "n"})
    for k, m in keys:
        vals = [c[k][m] for c in qc if k in c and isinstance(c[k].get(m), (int, float))]
        if vals:
            out[f"{k}/{m}"] = dict(median=round(float(np.median(vals)), 4), min=min(vals), max=max(vals))
    units = [checks[s]["units"] for s in sorted(checks, key=int) if "units" in checks[s]]
    if units:
        found = {}
        for u in units:
            for k, v in u.items():
                if k not in ("uv", "z10_median_m"):
                    found.setdefault(k, set()).add(json.dumps(v, sort_keys=True))
        if any(len(v) > 1 for v in found.values()):
            raise SystemExit(f"::error::the GDPS fields came in different units on different steps: {found}")
        out["units"] = {k: json.loads(next(iter(v))) for k, v in found.items()} | {
            "uv": units[0]["uv"],
            "z10_median_m": int(np.median([u["z10_median_m"] for u in units])),
        }
    return out


# ── plan / run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def complete(model, cyc_dt):
    """The run's last step is published: both GFS 0.5 deg files' .idx, or two of the GDPS's last files (the top and
    the bottom of the column; they land together), answer a HEAD; the AIFS single's every step on the Google mirror."""
    date, hh = cyc_dt.strftime("%Y%m%d"), cyc_dt.strftime("%H")
    last = steps_of(model, hh)[-1]
    if model == "gfs":
        return all(MV.head_ok(gfs_url(date, hh, last, b) + ".idx") for b in (False, True))
    if model == "aifs":
        return RF.cycle_complete(date, hh, "aifs-single", last_step=last, kind="fc")
    if model == "geos":
        return geos_complete(date, hh, last)
    return all(DM.exists(gdps_url(date, hh, last, v)) for v in ("AirTemp_IsbL-0001", "WindV_IsbL-0400"))


def load_manifest(path, model):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return dict(model=model, label=MODELS[model]["label"], source=MODELS[model]["source"], cycles=[])


def cmd_plan(a):
    """Print the newest complete cycle (00/12Z; the AIFS single's every 6 h) not yet in the manifest (nothing if none),
    looking back 48 h. With --cycle: that cycle, unless the manifest has it or a newer one (--force plans it anyway)."""
    man = load_manifest(Path(a.manifests) / f"strat_{a.model}.json", a.model)
    have = {c["cycle"] for c in man.get("cycles", [])}
    if a.cycle:
        if have and max(have) >= a.cycle and not a.force:
            print(f"{a.cycle}: the page already has {max(have)}", file=sys.stderr)
        else:
            print(a.cycle)
        return
    every = MODELS[a.model].get("every", 12)
    now = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    c = now.replace(hour=now.hour - now.hour % every)
    for _ in range(48 // every):
        key = c.strftime("%Y%m%d%H")
        if key in have:
            return  # the newest published: nothing newer is complete
        if complete(a.model, c):
            print(key)
            return
        print(f"  {a.model} {key}: not complete yet", file=sys.stderr)
        c -= dt.timedelta(hours=every)


def cmd_run(a):
    model, key = a.model, a.cycle
    M = MODELS[model]
    cyc_dt = dt.datetime.strptime(key, "%Y%m%d%H")
    date, hh = key[:8], key[8:]
    steps = list(steps_of(model, hh))
    if os.environ.get("STRAT_STEPS"):  # testing only: a subset of the steps
        steps = [int(x) for x in os.environ["STRAT_STEPS"].split(",")]
    site = Path(a.site)
    data = data_dir(model)
    outdir = site / data / key
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t_start = time.time()
    per = {}  # file -> {step: {hemi: q}}
    checks, failed, extras = {}, [], {}
    nbytes = nreq = nb = 0
    with ProcessPoolExecutor(a.procs) as pool:
        futs = [pool.submit(process_step_retry, model, date, hh, s, str(outdir)) for s in steps]
        for fu in as_completed(futs):
            r = fu.result()
            if isinstance(r, int):
                failed.append(r)
                continue
            step, meta, b, ck, (n, bb), ex = r
            for name, q in meta.items():
                per.setdefault(name, {})[step] = q
            checks[str(step)] = ck
            extras.update(ex)
            nbytes, nreq, nb = nbytes + b, nreq + n, nb + bb
            print(
                f"  {model} +{step:03d} h: {b / 1e6:.2f} MB, fetched {bb / 1e6:.0f} MB in {n} requests in "
                f"{ck['fetch_secs']} s, {ck['secs']} s in all",
                flush=True,
            )
    if failed:
        print(f"::error::{model} {key}: steps {sorted(failed)} failed - not publishing a partial cycle")
        sys.exit(1)
    hours = sorted(steps)
    fields = {}
    for name, hs in per.items():
        e = dict(hours=sorted(hs))
        e["q"] = {h: [hs[s][h] for s in e["hours"]] for h in HEMIS}
        fields[name] = e
    omega = None
    if M["omega"]:
        ref, ranges = vv_ranges()
        if ranges:
            with ProcessPoolExecutor(a.procs) as pool:
                futs = [
                    pool.submit(vv_level, str(outdir), k, p, ranges[lev_key(p)], hours)
                    for k, p in enumerate(M["levels"])
                ]
                nbytes += sum(fu.result() for fu in futs)
            vh = [t for t in hours if window(hours, t)]
            fields.update({f"o{lev_key(p)}": dict(hours=vh) for p in M["levels"]})
            omega = dict(
                sigma_km=VV_SIGMA_KM,
                window_h=2 * VV_HALF_H,
                ranges=ranges,
                win=[[min(w), max(w)] for w in (window(hours, t) for t in vh)],
                ref=dict(stat=ref["stat"], samples=ref["samples"], period=ref["period"]),
            )
        else:
            print(f"::warning::no {VV_REF.name}: no vertical-motion maps this run (strat_maps.py omega-clim)")
        for f in outdir.glob("_vv_*.npy"):
            f.unlink()
    di = extras["di"]
    wall = time.time() - t_start
    ny, nx = round(90 / di) + 1, round(360 / di) + 2 * PAD
    cap = (
        {h: {str(t): [checks[str(s)]["cap"][h][str(t)] for s in hours] for t in CAP_THETAS} for h in HEMIS}
        if M["thetas"]
        else None
    )  # [PV, hPa] per step
    entry = dict(
        cycle=key,
        init=cyc_dt.strftime("%Y-%m-%dT%H:00Z"),
        length=max(hours),
        hours=hours,
        levels=list(M["levels"]),
        thetas=list(M["thetas"]),
        pv_levels=col_of(model) if M["thetas"] else [],
        grids={h: grid_desc(h, nx, ny, di) for h in HEMIS},
        enc=dict(
            t=dict(off=T_OFF, step=T_STEP),
            pv=dict(off=PV_OFF, step=PV_STEP, ref=PV_REF, sh_sign=-1),
            p=dict(lo=P_LO, hi=P_HI),
        ),
        fields=fields,
        zmean_km=extras.get("zmean_km", {}),
        cap_hpa=extras.get("cap_hpa", {}),
        **({"cap": dict(lat=CAP_LAT, values=cap)} if cap else {}),
        **({"omega": omega} if omega else {}),
        bytes=nbytes,
        fetch=dict(requests=nreq, mb=round(nb / 1e6, 1)),
        checks=summarise(checks),
        wall_s=round(wall, 1),
        made=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
    )
    nfiles = sum(len(e["hours"]) for e in fields.values()) * len(HEMIS)
    print(
        f"{model} {key}: {nbytes / 1e6:.1f} MB in {nfiles} files, fetched {nb / 1e6:.0f} MB in {nreq} requests, "
        f"wall {wall:.0f} s",
        flush=True,
    )
    for kind in ("t", "w", "pv", "o"):
        sz = sum(p.stat().st_size for h in HEMIS for p in (outdir / h).glob(f"{kind}[0-9]*/*.webp"))
        print(f"  {kind:3s} files {sz / 1e6:6.1f} MB")
    print("checks: " + json.dumps(entry["checks"]), flush=True)

    mpath = site / "assets/models" / f"strat_{model}.json"
    man = load_manifest(Path(a.manifest_in) if a.manifest_in else mpath, model)
    cycles = sorted(
        [c for c in man.get("cycles", []) if c["cycle"] != key] + [entry], key=lambda c: c["cycle"], reverse=True
    )
    # The newest run and those back to the run 12 h before it: the page draws the newest and its change from that run
    # (the steps are 12-hourly, so the valid times pair), and a visitor's copy of the old manifest (or a deploy not out
    # yet) still finds the previous run's maps in the store.
    n_keep = 1 + 12 // M.get("every", 12)
    keep, drop = cycles[:n_keep], cycles[n_keep:]
    man.update(model=model, label=M["label"], source=M["source"], cycles=keep, updated=entry["made"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, separators=(",", ":")))
    if a.lists:
        L = Path(a.lists)
        L.mkdir(parents=True, exist_ok=True)
        stale = any(c["cycle"] == key for c in drop)  # a re-run of a cycle older than the one shown: not published
        (L / "publish.txt").write_text("" if stale else f"{data}/{key}\n")
        (L / "prune.txt").write_text("".join(f"{data}/{c['cycle']}\n" for c in drop))
        (L / "floor.txt").write_text(f"{data}:{keep[-1]['cycle']}\n")
    print(f"manifest: keep {[c['cycle'] for c in keep]}, drop {[c['cycle'] for c in drop]}")


def clim_sample(day):
    """One sample of omega-clim: the vertical motion of the 72 h around 00 UTC on `day` from the seven 00/12 UTC
    analyses in it, averaged and smoothed as the maps are (both linear, so in either order). -> (day, histogram of
    log10 |omega| per level poleward of CLIM_LAT, cos-weighted) or (day, None) when an analysis is missing."""
    keys = [("VVEL", f"{p:g} mb") for p in GFS_LEVELS]
    t0 = dt.datetime.combine(day, dt.time())
    acc = lat = None
    try:
        for off, wt in window(range(-VV_HALF_H, VV_HALF_H + 1, 12), 0).items():
            c = t0 + dt.timedelta(hours=off)
            msgs, grid = sb.fetch_messages(fetcher(), gfs_url(c.strftime("%Y%m%d"), c.strftime("%H"), 0), set(keys))
            x = wt * HPA_DAY * np.stack([msgs[k] for k in keys]).astype(np.float64)
            acc = x if acc is None else acc + x
            lat = grid["lat"][:, 0].astype(np.float64)
    except sb.Missing as e:
        print(f"  {day}: {e}", flush=True)
        return day.isoformat(), None
    x = smooth_sphere(acc, lat)
    rows = np.abs(lat) >= CLIM_LAT
    w = np.cos(np.radians(lat[rows]))[:, None] * np.ones((1, x.shape[-1]))
    with np.errstate(divide="ignore"):
        hist = [np.histogram(np.log10(np.abs(x[k][rows])), bins=CLIM_BINS, weights=w)[0] for k in range(len(keys))]
    print(f"  {day}: done", flush=True)
    return day.isoformat(), np.stack(hist)


def nice_up(v):
    """The smallest of 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^n at or above v."""
    e = 10.0 ** math.floor(math.log10(v))
    return next(round(m * e, 10) for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v * (1 - 1e-9))


def cmd_omega_clim(a):
    """The vertical motion's colour range per level: the 99th percentile of |omega| (smoothed as the maps are) poleward
    of CLIM_LAT over CLIM_MONTHS monthly samples of GFS analyses, rounded up (nice_up) -> VV_REF."""
    y0, m0 = CLIM_FROM
    days = [dt.date(y0 + (m0 - 1 + i) // 12, (m0 - 1 + i) % 12 + 1, 15) for i in range(CLIM_MONTHS)]
    with ProcessPoolExecutor(a.procs) as pool:
        got = [(d, h) for d, h in pool.map(clim_sample, days) if h is not None]
    if len(got) < 0.9 * len(days):
        raise SystemExit(f"::error::only {len(got)} of {len(days)} samples - not writing {VV_REF.name}")
    H = sum(h for _, h in got)
    levels = {}
    for k, p in enumerate(GFS_LEVELS):
        c = np.cumsum(H[k]) / H[k].sum()
        q = {
            n: float(10 ** CLIM_BINS[1:][np.searchsorted(c, f)]) for n, f in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        }
        levels[lev_key(p)] = {n: float(f"{v:.3g}") for n, v in q.items()} | {"range": nice_up(q["p99"])}
        print(f"  {p:5g} hPa: " + ", ".join(f"{n} {v:.3g}" for n, v in levels[lev_key(p)].items()) + " hPa/day")
    doc = dict(
        stat=f"99th percentile of |omega| poleward of {CLIM_LAT:g} deg, cos-weighted, rounded up",
        units="hPa/day, descent positive",
        smoothing=f"Gaussian sigma {VV_SIGMA_KM:g} km along the sphere; {2 * VV_HALF_H} h centred mean of 00/12 UTC analyses",
        period=f"{days[0]:%Y-%m} to {days[-1]:%Y-%m}, the 15th of each month",
        samples=len(got),
        missing=sorted({d.isoformat() for d in days} - {d for d, _ in got}),
        levels=levels,
        made=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
    )
    VV_REF.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {VV_REF} from {len(got)} samples")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--manifests", default="assets/models")
    p.add_argument("--cycle", default="", help="YYYYMMDDHH: plan this cycle (skipped when already published)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--cycle", required=True)
    p.add_argument("--site", default=".")
    p.add_argument("--procs", type=int, default=4, help="steps processed at once (each fetches its own)")
    p.add_argument("--lists", default="")
    p.add_argument("--manifest-in", default="", help="the manifest to extend (main's CURRENT copy)")
    p = sub.add_parser("omega-clim")
    p.add_argument("--procs", type=int, default=4, help="samples fetched at once")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "omega-clim": cmd_omega_clim}[a.cmd](a)


if __name__ == "__main__":
    main()
