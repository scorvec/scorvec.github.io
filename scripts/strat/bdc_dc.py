#!/usr/bin/env python3
"""Brewer-Dobson circulation by DOWNWARD CONTROL: MERRA-2 climatology + GEOS FP analyses and 10-day forecast.

Built 2026-09-27 to replace the continuity-based geos_bdc.py figures (user: "Those charts look like a complete mess").

WHY DOWNWARD CONTROL. psi* from the Eulerian integral of [v] minus the eddy term (geos_bdc.py) fails from once-daily
analyses: tested on identical days, neither GEOS FP nor MERRA-2 gives the boreal-winter maximum of tropical upwelling
(DJF/JJA ~0.8-1.0) and tropical w* from omega is ~0 or downward - the analysed mean meridional circulation of the tropical
stratosphere (~0.3 mm/s) is below the noise of a single 00Z zonal mean. Downward control (Haynes et al. 1991) builds the
residual circulation from the zonal momentum forcing, which the analyses constrain far better.

METHOD. Steady TEM zonal momentum balance in pressure coordinates (Andrews, Holton and Leovy 1987):
    -fhat v* = D = (1/(a cos phi)) div F + X ,   fhat = f - (1/(a cos phi)) d(u cos phi)/d phi
    F_phi = a cos phi (u_p [v'th']/th_p - [u'v']),   F_p = a cos phi (fhat [v'th']/th_p - [u'omega'])
    psi*(phi, p) = -(2 pi a cos phi / g) * integral from 0.1 hPa to p of D / fhat dp'          (1e9 kg/s)
  X is the parameterized gravity-wave drag (orographic + non-orographic). The analysis increment is NOT included (it
  exists for neither the forecast nor the MERRA-2 GMI replay; see VALIDATION for its size). Zonal means and eddy
  covariances come from 00Z fields point-sampled at 1 x 1.25 deg, on GEOS FP's pressure levels 0.1-200 hPa for BOTH
  analyses (MERRA-2's extra 125 and 85 hPa levels are dropped: derivatives and the integral are grid-dependent).
  Index = tropical upward mass flux across 70 and 100 hPa: psi* averaged over +-2 deg around the NH turnaround latitude
  minus the same around the SH one, with the turnaround latitudes FIXED from the MERRA-2 mean circulation for the day of
  year (extremum over 15-40 deg of the 2-deg-smoothed 3-harmonic climatology). Fixing them keeps the index linear in
  psi*, so a 30-day mean of daily values is the index of the 30-day-mean circulation. The earlier choice - each day's
  own max minus min - is biased high by the day-to-day noise (MERRA-2 70 hPa annual mean 9.4 instead of 6.4).
  Single days are noisy (sd ~4 at 70 hPa: transient wave forcing, which the steady balance maps straight into v*), so
  the product is the trailing 30-day mean. Section: 30-day mean psi*; |lat| < 15 is interpolated in sin(lat) (uniform
  tropical upwelling) because fhat -> 0 there.

DATA.
  Climatology: MERRA-2 GMI replay (MERRA-2 meteorology), NCCS OPeNDAP merra2_gmi/inst3_3d_met_Np (u, v, T, omega at 00Z,
    every 10th day; fetch_m2gmi_tem.py -> data/m2gmi_tem/tem_YYYY.npz) + merra2_gmi/tavg24_3d_dad_Np daily DUDTGWD
    (fetch_m2gmi_gwd.py -> gwd_YYYY.npz). Years = whatever tem_YYYY.npz exist when `clim` runs (listed in
    clim_merra2.json); rerun `clim` as more land.
  GEOS FP analyses: assim/inst3_3d_asm_Np 00Z daily (u, v, T, omega, 0.1-200 hPa) + assim/tavg3_3d_udt_Nv DUDTGWD on
    native levels 1-43 (pure pressure levels above 164 hPa), calendar-day mean of the 01:30-19:30 windows, interpolated in
    log p -> data/bdc_dc/fp_tail/YYYYMMDD.npz (also DUDTANA, the analysis-increment tendency, as a diagnostic).
  GEOS FP forecast: fcast/inst3_3d_asm_Np.<YYYYMMDD_00> 00Z days 0-10 -> data/bdc_dc/fp_fcst/<cycle>.npz. The fcast
    directory publishes NO wind-tendency collection (checked 2026-09-26: no udt/GWD file), so the forecast holds the mean
    GEOS FP analysis drag of the last 5 days fixed.
  Every OPeNDAP read is sequential, retried with backoff, and validated (missing/fill, all-zero or constant levels,
  physical ranges) - a parallel OPeNDAP read has silently returned zeros before.

VALIDATION (2026-09-27; numbers also in data/bdc_dc/*.json)
  MERRA-2 climatology (15 years 1991-94, 1996-2001, 2004, 2006-09 on file 2026-09-27; 552 days): 70 hPa tropical
    upward mass flux 6.4 x 1e9 kg/s annual mean (literature for reanalysis downward control ~6-7), Dec 9.1 / Jan 8.1
    vs Jul 5.0 / Aug 5.2, DJF/JJA 1.52; 100 hPa 10.0, DJF/JJA 1.54. Resolved-wave part 5.3 (DJF/JJA 1.40), gravity-
    wave part 1.05 (DJF/JJA 2.3). Single days scatter with sd ~4 (70 hPa) about the normal; the year-to-year variance of
    30-day means is not resolvable from every-10th-day samples (estimate -0.44 +- 0.33 at 70 hPa, -0.40 +- 0.53 at
    100 hPa), so the band and the tests use 0 + one standard error.
  Same days, GEOS FP vs MERRA-2 (38 days, every 20th day of 2018-19, each with its own drag): 70 hPa 5.67 vs 6.55,
    GEOS FP - MERRA-2 = -0.88 +- 0.24 (ratio 0.87 +- 0.04), same-day r 0.91; the gravity-wave part is -0.53 +- 0.05
    (GEOS FP's parameterized drag is 54 % of MERRA-2's), the resolved part -0.35 +- 0.23 (not significant). 100 hPa
    -0.54 +- 0.39 (ratio 0.95, r 0.85; drag -0.93 +- 0.07, resolved +0.40 +- 0.38). Seasonal means at 70 hPa -1.2 DJF,
    -0.3 MAM, -1.5 JJA, -0.4 SON: not distinguishable (one-way ANOVA p 0.21; 0.09 at 100 hPa), and scaling only the drag
    part does not remove the spread, so the shift is one additive constant: GEOS FP values are shifted +0.88 (70 hPa) /
    +0.54 (100 hPa), and psi* by its per-point mean difference, before any comparison with the MERRA-2 normal; the
    shift's standard error enters every test. Caveat: GEOS FP's model version (and GWD tuning) has changed since
    2018-19; its 2026 drag part (0.32 at 70 hPa) against MERRA-2's normal drag on the same days (0.74) is consistent
    with the 2018-19 ratio.
  GEOS FP analysis tail, 106 daily analyses 2026-06-13 -> 09-26: unshifted mean 5.27 at 70 hPa vs the MERRA-2 normal
    for the same days 5.44 (8.17 vs 8.20 at 100 hPa); 68 % / 75 % of single days inside MERRA-2's single-day 10-90 %
    range (80 % expected). Daily index: lag-1 autocorrelation 0.59 with a ~6-day oscillation; Bartlett variance
    inflation 1.39 (70 hPa) / 1.63 (100 hPa) for a 30-day mean.
  Forecast day 0 (00Z 2026-09-26) = the analysis of that time: 8.21 vs 8.14 at 70 hPa, 11.56 vs 11.48 at 100 hPa
    (the state is identical; only the persisted vs actual drag differs), inside MERRA-2's single-day 10-90 % range for
    the date (1.6-11.1 and 4.3-14.9).
  Forecast drag, hindcast on the tail (index from each day's own state): holding the last 5 days' GEOS FP drag gives
    RMSE 0.09-0.11 (70 hPa) / 0.20-0.24 (100 hPa) and bias < 0.02 at 1-10 days; MERRA-2's day-of-year drag instead is
    biased +0.43 / +0.99 (RMSE 0.44 / 1.02) because GEOS FP's drag is half of MERRA-2's. Persistence is used.
  Analysis increment (excluded throughout): in GEOS FP it would add +0.99 (70 hPa) / +1.59 (100 hPa) over the tail -
    more than GEOS FP's own gravity-wave part (0.32 / 0.58); in MERRA-2 (monthly M2TMNPUDT DUDTANA, 12 of the years)
    +0.45 / +0.68 annual mean.
  Re-reading a tail day and a forecast cycle from the server reproduced the cached fields bit for bit.
  Last 30 days (2026-08-28 -> 09-26): 70 hPa 7.43 on the MERRA-2 scale vs normal 5.56 (+33 %, z 1.65, p 0.10, not
    significant at 5 %); 100 hPa 9.58 vs 8.20 (+17 %, p 0.33). Section: no point passes FDR 10 %. Context, not tested:
    strong El Nino (raises tropical upwelling) and QBO westerlies at 20-50 hPa (lower it).
  Live update: ~6 min on a normal day (one analysis day ~100 s, the 11-step forecast ~4 min, render ~1 s);
  7.2 min measured with two analysis days + a full forecast; catching up after an outage is capped at 6 min of
  analysis days per run. The tail backfill is ~100 s a day.

STATISTICS (the site rule: every claim states its test).
  Index: z-test of the GEOS FP 30-day mean against the MERRA-2 normal for the same days; the variance adds the GEOS FP
  mean's sampling noise (MERRA-2 day-to-day variance / n, inflated by the Bartlett sum of the GEOS FP daily
  autocorrelation to 10 days), MERRA-2's
  year-to-year variance of 30-day means (estimate clipped at 0 plus one standard error - conservative), the normal's
  standard error, and the overlap offset's. Band = MERRA-2 10-90 % range of a 30-day mean of daily values from the same
  variance model. Section: the same z-test per point, Benjamini-Hochberg false-discovery rate 10 % over 100-1 hPa and
  15-80 deg; only passing points are shaded.

USAGE (LAPTOP; nothing here writes into assets/ or commits)
    python bdc_dc.py clim                        # MERRA-2 climatology from the local archive (~15 s)
    python bdc_dc.py overlap                     # one-off: 2018-19 same-day GEOS FP vs MERRA-2 (~2.5 min a day)
    python bdc_dc.py tail --days 106 --step 1    # one-off backfill of the analysis tail (~100 s a day)
    python bdc_dc.py live                        # daily: new analysis days + newest forecast + figures
    python bdc_dc.py render                      # figures + JSON from the caches only
  Outputs in data/bdc_dc/: bdc_dc_upwelling.{webp,png}, bdc_dc_section.{webp,png}, bdc_dc.json, clim_merra2.{npz,json},
  clim_merra2_sections.json (the climatological psi* section by month).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
M2DIR = HERE / "data" / "m2gmi_tem"                      # MERRA-2 GMI tem_YYYY.npz + gwd_YYYY.npz (fetch_m2gmi_*.py)
OUT = HERE / "data" / "bdc_dc"
TAIL = OUT / "fp_tail"                                   # one npz per GEOS FP analysis day
FCST = OUT / "fp_fcst"                                   # one npz per GEOS FP 00Z forecast cycle
OVL = OUT / "overlap"                                    # 2018-19 MERRA-2 GMI vs GEOS FP, same days

NCCS = "https://opendap.nccs.nasa.gov/dods"
ASSIM_NP = f"{NCCS}/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Np"
ASSIM_UDT = f"{NCCS}/GEOS-5/fp/0.25_deg/assim/tavg3_3d_udt_Nv"
FCAST_NP = f"{NCCS}/GEOS-5/fp/0.25_deg/fcast/inst3_3d_asm_Np"
M2GMI_NP = f"{NCCS}/merra2_gmi/inst3_3d_met_Np"

A_E, G, K, P0, OM = 6.371e6, 9.80665, 0.2857, 1000.0, 7.292e-5
# Common pressure levels, top -> bottom (hPa): GEOS FP's Np levels from 0.1 to 200 hPa. MERRA-2 GMI also has 125 and 85
# hPa; they are dropped so both analyses see the same vertical grid (derivatives and the integral are grid-dependent).
LEVS = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100, 150, 200], float)
# GEOS FP L72 native mid-level pressures (hPa) for levels 1-43, from delp with p_top = 0.01 hPa; levels 1-41 are pure
# pressure levels (zonal sd of p = 0), 42-43 vary by <2 hPa. Measured 2026-09-26 from tavg3_3d_udt_Nv.
P_NATIVE = np.array([0.015, 0.0263, 0.0401, 0.0568, 0.0777, 0.1045, 0.1396, 0.1854, 0.2449, 0.3218, 0.4204, 0.5463,
                     0.706, 0.9073, 1.16, 1.4757, 1.8679, 2.3526, 2.9483, 3.6765, 4.5617, 5.6318, 6.9183, 8.4564,
                     10.2849, 12.4602, 15.0503, 18.1244, 21.761, 26.0491, 31.0889, 36.9927, 43.9097, 52.0159, 61.4957,
                     72.5579, 85.439, 100.5144, 118.25, 139.115, 163.6615, 192.4131, 225.8709])
STRIDE_FP, STRIDE_M2 = 4, 2                              # 0.25 x 0.3125 -> 1 x 1.25 deg ; 0.5 x 0.625 -> 1 x 1.25 deg
UDT_HOURS = (1.5, 7.5, 13.5, 19.5)                       # tavg3 centres: a calendar-day mean, as MERRA-2's tavg24 GWD


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------------------------------------
def downward_control(lat, p_hpa, ubar, tbar, vT, uv, uw, X):
    """Steady downward control. Inputs on (lev, lat), p ascending (top -> bottom), eddy covariances with omega in Pa/s,
    X = parameterized drag (m/s^2). Returns psi* (1e9 kg/s) from the total, the resolved EP-flux part and the X part."""
    phi = np.deg2rad(lat); c = np.cos(phi); cc = np.where(np.abs(c) < 1e-3, 1e-3, c)
    p = p_hpa * 100.0
    th = tbar * (P0 / p_hpa)[:, None] ** K
    vth = vT * (P0 / p_hpa)[:, None] ** K
    th_p = np.gradient(th, p, axis=0)
    th_p = np.where(np.abs(th_p) < 1e-7, -1e-7, th_p)
    u_p = np.gradient(ubar, p, axis=0)
    f = 2 * OM * np.sin(phi)
    fhat = f[None, :] - np.gradient(ubar * c[None, :], phi, axis=1) / (A_E * cc[None, :])
    Fphi = A_E * c[None, :] * (u_p * vth / th_p - uv)
    Fp = A_E * c[None, :] * (fhat * vth / th_p - uw)
    divF = np.gradient(Fphi * c[None, :], phi, axis=1) / (A_E * cc[None, :]) + np.gradient(Fp, p, axis=0)
    D_ep = divF / (A_E * cc[None, :])
    fh = np.where(np.abs(fhat) < 1e-6, np.nan, fhat)

    def integ(Dx):
        q = Dx / fh
        cum = np.concatenate([np.zeros((1, q.shape[1])),
                              np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(p)[:, None], axis=0)])
        return -(2 * np.pi * A_E * c[None, :] / G) * cum / 1e9
    return integ(D_ep + X), integ(D_ep), integ(X)


IDX_LEVS = (100, 70, 50)
TURN_HALF = 2.0                                           # the index averages psi* over +-2 deg around each turnaround


def lev_index(level):
    return int(np.argmin(np.abs(LEVS - level)))


def turn_table(psi_clim_fn, lat):
    """Turnaround latitudes of the MERRA-2 mean circulation by day of year: the extremum of the climatological psi*
    (smoothed with a 2-deg Gaussian in latitude) over 15-40 deg in each hemisphere, at each index level ->
    (366, len(IDX_LEVS), 2) latitudes (south, north). Fixing them makes the index LINEAR in psi*: the mean of daily
    values equals the value of the mean circulation. Taking each day's own max-min instead inflated the MERRA-2 70 hPa
    annual mean from 6.5 to 9.4 x 1e9 kg/s, because the max of a noisy profile is biased high."""
    from scipy.ndimage import gaussian_filter1d
    n = (lat >= 15) & (lat <= 40); s = (lat <= -15) & (lat >= -40)
    out = np.zeros((366, len(IDX_LEVS), 2))
    for d in range(1, 367):
        ps = psi_clim_fn(d)
        for i, L in enumerate(IDX_LEVS):
            row = gaussian_filter1d(np.nan_to_num(ps[lev_index(L)]), 2.0, mode="nearest")
            out[d - 1, i, 0] = lat[s][np.argmin(row[s])]; out[d - 1, i, 1] = lat[n][np.argmax(row[n])]
    return out


_TL = None


def get_tl():
    global _TL
    if _TL is None:
        _TL = load(OUT / "clim_merra2.npz")["turn"]
    return _TL


def band_mean(row, lat, c):
    lo, hi = (c - TURN_HALF, c + TURN_HALF)
    lo, hi = (max(lo, 15), min(hi, 40)) if c > 0 else (max(lo, -40), min(hi, -15))
    return float(np.nanmean(row[(lat >= lo - 1e-6) & (lat <= hi + 1e-6)]))


def upflux(row, lat, doy, i_lev, tl=None):
    """Tropical upward mass flux (1e9 kg/s) across a level: psi* averaged over +-2 deg around the NH turnaround latitude
    minus the same around the SH one, both from the MERRA-2 climatology for the day of year (turn_table)."""
    tl = get_tl() if tl is None else tl
    s, n = tl[int(doy) - 1, i_lev]
    return band_mean(row, lat, n) - band_mean(row, lat, s)


def dc_psi(z, X=None):
    X = z["gwd"] if X is None else X
    return downward_control(z["lat"], LEVS, z["ubar"], z["tbar"], z["vT"], z["uv"], z["uw"], X)


def index_vals(tot, ep, gw, lat, doy, tl=None):
    """KEYS from psi* (total, EP part, GWD part); ep/gw may carry only the IDX_LEVS rows (then indexed by position)."""
    r = {}
    for i, L in enumerate(IDX_LEVS):
        k = lev_index(L)
        r[f"F{L}"] = upflux(tot[k], lat, doy, i, tl)
        for nm, arr in (("ep", ep), ("gw", gw)):
            row = arr[k] if arr.shape[0] == len(LEVS) else arr[i]
            r[f"F{L}_{nm}"] = upflux(row, lat, doy, i, tl)
    return r


def dc_record(z, doy, X=None):
    """Downward control for one reduced record -> dict with psi (total) and the index KEYS."""
    tot, ep, gw = dc_psi(z, X)
    return {"psi": tot, **index_vals(tot, ep, gw, z["lat"], doy)}


# ----------------------------------------------------------------------------------------------------------------------
# NCCS OPeNDAP reads (sequential; every field validated - a parallel OPeNDAP read has silently returned zeros before)
# ----------------------------------------------------------------------------------------------------------------------
class Bad(ValueError):
    pass


def _clean(a):
    a = np.asarray(a, dtype="float64")
    return np.where(np.abs(a) > 1e10, np.nan, a)


def check_state(f):
    """Reject a snapshot with missing, fill, all-zero or out-of-range levels."""
    for k, a in f.items():
        if np.isfinite(a).mean() < 0.995:
            raise Bad(f"{k}: {100 * (1 - np.isfinite(a).mean()):.1f}% missing")
        for i in range(a.shape[0]):
            lv = a[i]
            if (lv == 0).mean() > 0.02 or not np.nanstd(lv) > 0:
                raise Bad(f"{k} level {i}: zero/constant field")
    t = f["t"]
    if not (140 < np.nanmin(t) and np.nanmax(t) < 360):
        raise Bad(f"t out of range {np.nanmin(t):.0f}-{np.nanmax(t):.0f}")
    if np.nanmax(np.abs(f["u"])) > 300 or np.nanmax(np.abs(f["v"])) > 250:
        raise Bad("wind out of range")
    if np.nanmax(np.abs(f["omega"])) > 20:
        raise Bad("omega out of range")


def reduce_state(f):
    """Zonal means and eddy covariances on (lev, lat) from full fields (lev, lat, lon)."""
    zm = {k: np.nanmean(f[k], axis=-1) for k in f}
    d = {k: f[k] - zm[k][..., None] for k in f}
    return {"ubar": zm["u"], "vbar": zm["v"], "tbar": zm["t"], "wbar": zm["omega"],
            "vT": np.nanmean(d["v"] * d["t"], -1), "uv": np.nanmean(d["u"] * d["v"], -1),
            "uw": np.nanmean(d["u"] * d["omega"], -1)}


def retry(fn, what, tries=4, base=30):
    err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                                           # noqa: BLE001
            err = e
            log(f"  retry {i + 1}/{tries} {what}: {str(e)[:120]}")
            time.sleep(base * (i + 1))
    raise err


def open_ds(url, recent=False):
    """Open an OPeNDAP dataset; with recent=True the time axis must reach within 3 days of now (the GrADS server has
    served a garbled time axis while it was rewriting it)."""
    import xarray as xr

    def go():
        ds = xr.open_dataset(url)
        if recent:
            last = pd.Timestamp(ds.time.values[-1])
            if not (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=3) < last):
                raise Bad(f"time axis ends {last}")
        return ds
    return retry(go, f"open {url.rsplit('/', 1)[-1]}", base=60)


def read_state_fp(ds, t):
    """GEOS FP inst3_3d_asm_Np (assim or fcast) at time t -> reduced dict on LEVS (top -> bottom)."""
    lev = ds.lev.values
    i0 = int(np.argmin(np.abs(lev - LEVS[-1])))                  # 200 hPa; lev is 1000 -> 0.1
    assert np.allclose(lev[i0:][::-1], LEVS), "unexpected GEOS FP levels"

    def go():
        sub = ds[["u", "v", "t", "omega"]].sel(time=t).isel(lev=slice(i0, None), lat=slice(None, None, STRIDE_FP),
                                                             lon=slice(None, None, STRIDE_FP)).load()
        f = {k: _clean(sub[k].values)[::-1] for k in ("u", "v", "t", "omega")}
        check_state(f)
        return reduce_state(f), ds.lat.values[::STRIDE_FP]
    return retry(go, f"state {t}")


def read_state_m2(ds, t):
    """MERRA-2 GMI inst3_3d_met_Np at time t -> reduced dict on LEVS (125 and 85 hPa dropped)."""
    lev = ds.lev.values
    i0 = int(np.argmin(np.abs(lev - LEVS[-1])))
    keep = np.array([int(np.argmin(np.abs(lev[i0:] - L))) for L in LEVS[::-1]])
    assert np.allclose(lev[i0:][keep][::-1], LEVS), "unexpected MERRA-2 levels"

    def go():
        sub = ds[["u", "v", "t", "omega"]].sel(time=t).isel(lev=slice(i0, None), lat=slice(None, None, STRIDE_M2),
                                                             lon=slice(None, None, STRIDE_M2)).load()
        f = {k: _clean(sub[k].values)[keep][::-1] for k in ("u", "v", "t", "omega")}
        check_state(f)
        return reduce_state(f), ds.lat.values[::STRIDE_M2]
    return retry(go, f"m2 state {t}")


def native_to_levs(zm):
    """(43 native levels, lat) -> (LEVS, lat), linear in log p."""
    x = np.log(P_NATIVE); xi = np.log(LEVS)
    return np.stack([np.interp(xi, x, zm[:, j]) for j in range(zm.shape[1])], axis=1)


def read_udt_fp(ds, day):
    """GEOS FP daily-mean parameterized GWD (and the analysis-increment tendency, a diagnostic) on (LEVS, lat), m/s^2,
    from the four tavg3 windows centred 01:30-19:30 of the calendar day; at least two must be valid."""
    tt = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in UDT_HOURS]
    tix = pd.DatetimeIndex(ds.time.values)
    gw, an = [], []
    for t in tt:
        if t not in tix:
            log(f"  udt {t} not on the server (axis ends {tix[-1]})")
            continue

        def go():
            sub = ds[["dudtgwd", "dudtana"]].sel(time=t).isel(lev=slice(0, len(P_NATIVE)), lat=slice(None, None, STRIDE_FP),
                                                             lon=slice(None, None, STRIDE_FP)).load()
            g, a = _clean(sub["dudtgwd"].values), _clean(sub["dudtana"].values)
            for nm, x in (("dudtgwd", g), ("dudtana", a)):
                if np.isfinite(x).mean() < 0.995 or (x == 0).mean() > 0.5 or np.nanmax(np.abs(x)) > 5e-2:
                    raise Bad(f"{nm} at {t}: missing/zero/out of range")
            return np.nanmean(g, -1), np.nanmean(a, -1)
        try:
            g, a = retry(go, f"udt {t}")
        except Exception as e:                                           # noqa: BLE001
            log(f"  udt {t} skipped: {str(e)[:100]}")
            continue
        gw.append(g); an.append(a)
    if len(gw) < 2:
        raise Bad(f"only {len(gw)} valid tavg3 GWD windows for {day}")
    return native_to_levs(np.mean(gw, 0)), native_to_levs(np.mean(an, 0)), len(gw)


def save(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: (np.asarray(v, "float32") if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v)
                                for k, v in rec.items()})
    tmp.replace(path)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------------------------------
# Fetch jobs
# ----------------------------------------------------------------------------------------------------------------------
def fetch_tail(days=90, step=1, budget_min=None):
    """GEOS FP analyses at 00Z for the last `days` days (every `step`-th day counting back from the newest), resumable."""
    t_start = time.time()
    ds = open_ds(ASSIM_NP, True); du = open_ds(ASSIM_UDT, True)
    last = pd.Timestamp(ds.time.values[-1])
    newest = last.normalize()
    targets = [newest - pd.Timedelta(days=k) for k in range(0, days + 1, step)]
    todo = [d for d in targets if not (TAIL / f"{d:%Y%m%d}.npz").exists()]
    # a day whose GWD came from fewer than four windows (the newest day, read before it closed) is redone later
    for d in targets:
        f = TAIL / f"{d:%Y%m%d}.npz"
        if f.exists() and int(load(f)["n_udt"]) < 4 and d not in todo and pd.Timestamp(du.time.values[-1]) >= d + pd.Timedelta(hours=19.5):
            todo.append(d)
    log(f"tail: {len(todo)} of {len(targets)} days to fetch (newest analysis {last})")
    for d in sorted(todo, reverse=True):
        if budget_min and (time.time() - t_start) / 60 > budget_min:
            log("tail: time budget reached, rest on the next run"); break
        t0 = time.time()
        try:
            st, lat = read_state_fp(ds, d)
            gwd, ana, n = read_udt_fp(du, d)
        except Exception as e:                                           # noqa: BLE001
            log(f"  {d:%Y-%m-%d} FAILED: {str(e)[:120]}"); continue
        save(TAIL / f"{d:%Y%m%d}.npz", {**st, "gwd": gwd, "ana": ana, "n_udt": n, "lat": lat, "lev": LEVS,
                                        "time": np.array(f"{d:%Y-%m-%d}")})
        log(f"  {d:%Y-%m-%d} ok ({n} GWD windows) {time.time() - t0:.0f}s")


def forecast_cycles():
    lst = retry(lambda: urllib.request.urlopen(FCAST_NP, timeout=120).read().decode(), "fcast listing")
    return sorted(set(re.findall(r"inst3_3d_asm_Np\.(\d{8}_00)", lst)))


def fetch_forecast(cyc=None):
    """Latest (or given) 00Z GEOS FP forecast, days 0-10 at 00Z, cached per cycle."""
    cyc = cyc or forecast_cycles()[-1]
    dest = FCST / f"{cyc}.npz"
    if dest.exists():
        log(f"forecast {cyc}: cached"); return dest
    ds = open_ds(f"{FCAST_NP}/inst3_3d_asm_Np.{cyc}")
    steps = [t for t in pd.DatetimeIndex(ds.time.values) if t.hour == 0]
    if len(steps) < 11:
        raise Bad(f"forecast {cyc} has only {len(steps)} 00Z steps")
    recs, lat = [], None
    for t in steps:
        t0 = time.time()
        st, lat = read_state_fp(ds, t)
        recs.append(st); log(f"  forecast {cyc} {t:%m-%d} {time.time() - t0:.0f}s")
    save(dest, {**{k: np.stack([r[k] for r in recs]) for k in recs[0]}, "lat": lat, "lev": LEVS,
                "valid": np.array([f"{t:%Y-%m-%d}" for t in steps]), "cycle": np.array(cyc)})
    return dest


def overlap_dates():
    return [pd.Timestamp(y, 1, 1) + pd.Timedelta(days=k) for y in (2018, 2019) for k in range(0, 365, 20)]


def fetch_overlap():
    """MERRA-2 GMI and GEOS FP on the same 2018-19 days (every 20th day, a subset of the MERRA-2 GWD dates)."""
    dm = open_ds(M2GMI_NP); ds = open_ds(ASSIM_NP, True); du = open_ds(ASSIM_UDT, True)
    for d in overlap_dates():
        fm, ff = OVL / f"m2_{d:%Y%m%d}.npz", OVL / f"fp_{d:%Y%m%d}.npz"
        t0 = time.time()
        try:
            if not fm.exists():
                st, lat = read_state_m2(dm, d)
                save(fm, {**st, "lat": lat, "lev": LEVS, "time": np.array(f"{d:%Y-%m-%d}")})
            if not ff.exists():
                st, lat = read_state_fp(ds, d)
                gwd, ana, n = read_udt_fp(du, d)
                save(ff, {**st, "gwd": gwd, "ana": ana, "n_udt": n, "lat": lat, "lev": LEVS, "time": np.array(f"{d:%Y-%m-%d}")})
        except Exception as e:                                           # noqa: BLE001
            log(f"  overlap {d:%Y-%m-%d} FAILED: {str(e)[:120]}"); continue
        log(f"  overlap {d:%Y-%m-%d} ok {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------------------------------
# MERRA-2 climatology samples
# ----------------------------------------------------------------------------------------------------------------------
def m2_records():
    """Every MERRA-2 GMI day that has both the state (tem_YYYY) and the daily GWD (gwd_YYYY), on LEVS."""
    for tf in sorted(glob.glob(str(M2DIR / "tem_*.npz"))):
        y = Path(tf).stem.split("_")[1]
        gf = M2DIR / f"gwd_{y}.npz"
        if not gf.exists():
            continue
        t, g = np.load(tf), np.load(gf)
        lev = t["lev"]; keep = np.array([int(np.argmin(np.abs(lev - L))) for L in LEVS])
        assert np.allclose(lev[keep], LEVS) and np.allclose(g["lev"][keep], LEVS) and np.allclose(g["lat"], t["lat"])
        gt = {str(x): i for i, x in enumerate(g["time"])}
        for i, tt in enumerate(t["time"]):
            j = gt.get(str(tt))
            if j is None:
                continue
            rec = {k: t[k][i][keep].astype("float64") for k in ("ubar", "vbar", "tbar", "omegabar", "vT", "uv", "uw")}
            rec["wbar"] = rec.pop("omegabar")
            rec["gwd"] = np.nan_to_num(g["dudtgwd"][j][keep].astype("float64"))
            rec["lat"], rec["time"] = t["lat"], str(tt)
            bad = [k for k in ("ubar", "tbar", "vT", "uv", "uw") if not np.isfinite(rec[k]).all() or (rec[k] == 0).all(axis=1).any()]
            if bad:
                log(f"  MERRA-2 {tt}: rejected ({', '.join(bad)} missing or all-zero at a level)"); continue
            yield rec


def build_clim_samples():
    """Downward control for every MERRA-2 day on file -> OUT/clim_merra2_samples.npz."""
    T, PS, EP, GW, U = [], [], [], [], []
    lat = None
    ki = [lev_index(L) for L in IDX_LEVS]
    for rec in m2_records():
        tot, ep, gw = dc_psi(rec)
        T.append(rec["time"]); PS.append(tot); EP.append(ep[ki]); GW.append(gw[ki]); U.append(rec["ubar"]); lat = rec["lat"]
    save(OUT / "clim_merra2_samples.npz", {"time": np.array(T), "psi": np.array(PS), "psi_ep": np.array(EP),
                                           "psi_gw": np.array(GW), "ubar": np.array(U), "lat": lat, "lev": LEVS})
    yrs = sorted({t[:4] for t in T})
    log(f"MERRA-2 samples: {len(T)} days, years {', '.join(yrs)}")
    return OUT / "clim_merra2_samples.npz"


def harm(doy, nh=3):
    x = 2 * np.pi * (np.asarray(doy, float) - 1) / 365.25
    return np.column_stack([np.ones(len(x))] + [f(k * x) for k in range(1, nh + 1) for f in (np.cos, np.sin)])


def circ_dd(doy, c):
    """Signed day-of-year distance, wrapped."""
    return (np.asarray(doy) - c + 182.5) % 365.25 - 182.5


def build_clim():
    """MERRA-2 downward-control climatology -> OUT/clim_merra2.npz (+ monthly numbers in clim_merra2.json).
    Index (F70, F100 and their EP / GWD parts): 3-harmonic mean by day of year; single-day 10/50/90 % from residuals
    pooled over +-30 days and all years, smoothed with 2 harmonics; residual variance by doy (2-harmonic fit of r^2).
    Section: 3-harmonic mean of psi* per (level, latitude) and plain monthly means; GWD: 3-harmonic mean per cell."""
    build_clim_samples()
    z = load(OUT / "clim_merra2_samples.npz")
    T = pd.DatetimeIndex(pd.to_datetime(z["time"])); doy = T.dayofyear.values
    X = harm(doy); days = np.arange(1, 367); Xd = harm(days); X2 = harm(days, 2)
    out = {"doy": days, "lat": z["lat"], "lev": LEVS, "years": np.array(sorted(set(T.year))), "n_days": len(T)}
    PS = z["psi"].reshape(len(T), -1)
    ok = np.isfinite(PS).all(0)
    cp = np.full((X.shape[1], PS.shape[1]), np.nan)
    cp[:, ok] = np.linalg.lstsq(X, PS[:, ok], rcond=None)[0]
    out["psi_coef"] = cp.reshape(X.shape[1], *z["psi"].shape[1:])
    out["turn"] = tl = turn_table(lambda d: np.tensordot(harm([d])[0], out["psi_coef"], axes=1), z["lat"])
    rows = [index_vals(z["psi"][i], z["psi_ep"][i], z["psi_gw"][i], z["lat"], doy[i], tl) for i in range(len(T))]
    for key in rows[0]:
        z[key] = np.array([r[key] for r in rows])
    save(OUT / "clim_merra2_index.npz", {k: z[k] for k in rows[0]} | {"time": z["time"]})
    js = {"n_days": int(len(T)), "years": [int(y) for y in sorted(set(T.year))], "monthly": {}, "annual_mean": {},
          "djf_over_jja": {}}
    m = T.month.values
    for key in ("F70", "F100", "F50", "F70_ep", "F70_gw", "F100_ep", "F100_gw"):
        y = z[key].astype(float)
        c = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ c
        out[f"{key}_mean"] = Xd @ c
        qs = {q: np.array([np.quantile(r[np.abs(circ_dd(doy, d0)) <= 30], q) for d0 in days]) for q in (0.1, 0.5, 0.9)}
        for q, arr in qs.items():
            out[f"{key}_q{int(q * 100)}"] = Xd @ c + X2 @ np.linalg.lstsq(X2, arr, rcond=None)[0]
        out[f"{key}_var"] = np.maximum(X2 @ np.linalg.lstsq(harm(doy, 2), r ** 2, rcond=None)[0], 0.05 * r.var())
        js["monthly"][key] = {int(k): round(float(y[m == k].mean()), 3) for k in range(1, 13)}
        js["annual_mean"][key] = round(float(y.mean()), 3)
        djf, jja = np.isin(m, [12, 1, 2]), np.isin(m, [6, 7, 8])
        js["djf_over_jja"][key] = round(float(y[djf].mean() / y[jja].mean()), 3)
    # year-to-year variance of 30-day means: var(3-sample window means) minus the sampling part, pooled over the year
    for key in ("F70", "F100"):
        r = z[key] - X @ np.linalg.lstsq(X, z[key], rcond=None)[0]
        vt, vn = [], []
        for d0 in range(1, 366, 10):
            w = np.abs(circ_dd(doy, d0)) <= 15
            g = pd.DataFrame({"y": T.year[w], "r": r[w]}).groupby("y").r
            vt.append(g.mean().var()); vn.append((g.var() / g.count()).mean())
        vt, vn = np.array(vt), np.array(vn)
        # each year's window mean is a mean of ~3 samples 10 days apart (lag-10 autocorrelation ~0, so independent)
        est = float(np.mean(vt - vn)); ny = len(set(T.year)); se = float(np.mean(vt) * np.sqrt(2 / (ny - 1)) / np.sqrt(12))
        out[f"{key}_via30"] = np.array(est); out[f"{key}_via30_se"] = np.array(se)
        js[f"{key}_interannual_var_30d"] = {"estimate": round(est, 3), "se": round(se, 3)}
    out["psi_month"] = np.array([np.nanmean(z["psi"][m == k], 0) for k in range(1, 13)])
    G = np.array([r["gwd"] for r in m2_records()])            # same order as the samples
    cg = np.linalg.lstsq(X, G.reshape(len(T), -1), rcond=None)[0]
    out["gwd_coef"] = cg.reshape(X.shape[1], *G.shape[1:])
    save(OUT / "clim_merra2.npz", out)
    global _TL
    _TL = None
    (OUT / "clim_merra2.json").write_text(json.dumps(js, indent=1))
    # the climatological psi* section by month, 200-0.1 hPa, every 2 deg (tropics |lat| < 15 filled as in the figures)
    lat = z["lat"]; jj = np.where(np.isclose(lat % 2, 0))[0]
    sec = {"units": "1e9 kg/s, positive = clockwise (NH cell)", "lev_hPa": LEVS.tolist(), "lat": lat[jj].tolist(),
           "years": js["years"], "note": "|lat| < 15 interpolated in sin(lat) (downward control undefined near the equator)",
           "psi_by_month": {int(k + 1): np.round(fill_tropics(out["psi_month"][k], lat)[:, jj], 3).tolist() for k in range(12)}}
    (OUT / "clim_merra2_sections.json").write_text(json.dumps(sec))
    log(f"climatology: {len(T)} MERRA-2 days, years {js['years']}; F70 annual {js['annual_mean']['F70']}, "
        f"DJF/JJA {js['djf_over_jja']['F70']}")
    return out


def clim_at(C, key, doy):
    doy = np.clip(np.asarray(doy), 1, 366) - 1
    return C[key][doy]


def psi_clim(C, doy):
    return np.tensordot(harm(np.atleast_1d(doy))[0], C["psi_coef"], axes=1)


def gwd_clim(C, doy):
    return np.tensordot(harm(np.atleast_1d(doy))[0], C["gwd_coef"], axes=1)


# ----------------------------------------------------------------------------------------------------------------------
# GEOS FP series
# ----------------------------------------------------------------------------------------------------------------------
KEYS = ("F100", "F100_ep", "F100_gw", "F70", "F70_ep", "F70_gw", "F50", "F50_ep", "F50_gw")


def tail_frame():
    """Every cached GEOS FP analysis day -> (DataFrame of the index and its parts, psi stack, gwd stack, records)."""
    rows, PS, GW, recs = [], [], [], []
    for f in sorted(f for f in TAIL.glob("*.npz") if re.fullmatch(r"\d{8}\.npz", f.name)):   # never a .tmp being written
        z = load(f)
        doy = pd.Timestamp(str(z["time"])).dayofyear
        r = dc_record(z, doy)
        ra = dc_record(z, doy, X=z["gwd"] + z["ana"])                         # diagnostic: with the analysis increment
        rows.append({"t": pd.Timestamp(str(z["time"])), **{k: r[k] for k in KEYS}, "F70_ana": ra["F70"] - r["F70"],
                     "F100_ana": ra["F100"] - r["F100"], "n_udt": int(z["n_udt"])})
        PS.append(r["psi"]); GW.append(z["gwd"]); recs.append(z)
    d = pd.DataFrame(rows).set_index("t")
    return d, np.array(PS), np.array(GW), recs


def persisted_gwd(tail_d, GW, day, window=5):
    """Mean GEOS FP analysis GWD over the `window` days ending on `day` (the forecast's drag, held fixed)."""
    w = (tail_d.index <= day) & (tail_d.index > day - pd.Timedelta(days=window))
    if not w.any():
        raise Bad(f"no GEOS FP GWD within {window} days of {day}")
    return GW[w].mean(0), int(w.sum())


def forecast_frame(path, X):
    """Downward control for every 00Z step of a cached forecast with drag X (lev, lat)."""
    z = load(path)
    rows, PS = [], []
    for i, v in enumerate(z["valid"]):
        rec = {k: z[k][i] for k in ("ubar", "tbar", "vT", "uv", "uw")}
        rec["lat"] = z["lat"]
        r = dc_record(rec, pd.Timestamp(str(v)).dayofyear, X=X)
        rows.append({"t": pd.Timestamp(str(v)), **{k: r[k] for k in KEYS}}); PS.append(r["psi"])
    return pd.DataFrame(rows).set_index("t"), np.array(PS), str(z["cycle"])


# ----------------------------------------------------------------------------------------------------------------------
# GEOS FP vs MERRA-2 on the same days (2018-19)
# ----------------------------------------------------------------------------------------------------------------------
OFFSET_CACHE = OUT / "overlap_offset.pkl"          # the computed offset, for runs without the overlap files (Actions)


def overlap_offset():
    """Same-day GEOS FP minus MERRA-2 (each with its own GWD) for the index and psi*; None when no pairs exist.
    Computed from data/bdc_dc/overlap + the MERRA-2 gwd files when they are on disk (and cached), else read from the
    cache (release bdc-ref-v1)."""
    import pickle
    if not any(OVL.glob("m2_*.npz")) and OFFSET_CACHE.exists():
        return pickle.loads(OFFSET_CACHE.read_bytes())
    out = _overlap_offset()
    if out is not None:
        OFFSET_CACHE.write_bytes(pickle.dumps(out))
    return out


def _overlap_offset():
    pairs = []
    for fm in sorted(f for f in OVL.glob("m2_*.npz") if re.fullmatch(r"m2_\d{8}\.npz", f.name)):
        day = fm.stem.split("_")[1]
        ff = OVL / f"fp_{day}.npz"
        g = M2DIR / f"gwd_{day[:4]}.npz"
        if not (ff.exists() and g.exists()):
            continue
        gz = np.load(g); j = [i for i, t in enumerate(gz["time"]) if str(t).replace("-", "") == day]
        if not j:
            continue
        keep = np.array([int(np.argmin(np.abs(gz["lev"] - L))) for L in LEVS])
        m2 = load(fm); m2["gwd"] = np.nan_to_num(gz["dudtgwd"][j[0]][keep].astype(float))
        fp = load(ff)
        doy = pd.Timestamp(day).dayofyear
        pairs.append((day, dc_record(m2, doy), dc_record(fp, doy)))
    if len(pairs) < 5:
        return None
    out = {"n": len(pairs), "days": [p[0] for p in pairs]}
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(pairs), (2000, len(pairs)))
    for key in KEYS:
        a = np.array([p[1][key] for p in pairs]); b = np.array([p[2][key] for p in pairs])
        d = b - a
        rb = b[boot].mean(1) / a[boot].mean(1)
        out[key] = {"m2": float(a.mean()), "fp": float(b.mean()), "diff": float(d.mean()),
                    "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))), "ratio": float(b.mean() / a.mean()),
                    "ratio_se": float(rb.std()), "r": float(np.corrcoef(a, b)[0, 1])}
    dpsi = np.array([p[2]["psi"] - p[1]["psi"] for p in pairs])
    out["psi_diff"] = np.nanmean(dpsi, 0)
    out["psi_diff_se"] = np.nanstd(dpsi, 0, ddof=1) / np.sqrt(len(pairs))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Statistics for the figures
# ----------------------------------------------------------------------------------------------------------------------
def interannual_var(Y, T, half=15):
    """Year-to-year variance of `2*half`-day means of Y (N, ...) about the harmonic climatology, pooled over the year:
    var(per-year window means) minus their sampling part (within-year variance / count). Returns (estimate, se)."""
    doy = T.dayofyear.values; X = harm(doy)
    Yf = Y.reshape(len(T), -1)
    R = Yf - X @ np.linalg.lstsq(X, np.nan_to_num(Yf), rcond=None)[0]
    vt, vn = [], []
    years = np.array(T.year)
    for d0 in range(1, 366, 10):
        w = np.abs(circ_dd(doy, d0)) <= half
        ys = np.unique(years[w])
        means = np.array([R[w & (years == y)].mean(0) for y in ys])
        within = np.array([R[w & (years == y)].var(0, ddof=1) / (w & (years == y)).sum() for y in ys
                           if (w & (years == y)).sum() > 1])
        vt.append(means.var(0, ddof=1)); vn.append(within.mean(0))
    vt, vn = np.array(vt), np.array(vn)
    ny = len(np.unique(years))
    est = (vt - vn).mean(0)
    se = vt.mean(0) * np.sqrt(2 / (ny - 1)) / np.sqrt(365 / (2 * half))     # ~12 independent windows a year
    return est.reshape(Y.shape[1:]), se.reshape(Y.shape[1:])


def harm_mean_se(Y, T, doys):
    """Standard error of the 3-harmonic climatological mean averaged over `doys`, from the regression residuals."""
    X = harm(T.dayofyear.values); Yf = Y.reshape(len(T), -1)
    coef = np.linalg.lstsq(X, np.nan_to_num(Yf), rcond=None)[0]
    s2 = ((Yf - X @ coef) ** 2).sum(0) / (len(T) - X.shape[1])
    xm = harm(doys).mean(0)
    lev = float(xm @ np.linalg.inv(X.T @ X) @ xm)
    return np.sqrt(s2 * lev).reshape(Y.shape[1:])


ACF_LAGS = 10


def acf_inflation(res, times, lags=ACF_LAGS):
    """Variance inflation of a mean of DAILY values, 1 + 2 sum_k (1 - k/(L+1)) rho_k (Bartlett window, L = 10 days),
    from the analysis residuals (axis 0) put on a daily grid; clipped to [1, 5]. Measured on 107 daily GEOS FP analyses
    (2026-09-27) the 70 hPa index has rho_1 = 0.59 but a ~6-day oscillation (rho_3 = -0.44, rho_6 = +0.31), so an AR(1)
    model (inflation 3.9) would overstate the noise of a 30-day mean about threefold; the Bartlett sum gives ~1.3."""
    t = pd.DatetimeIndex(times)
    grid = pd.date_range(t.min(), t.max(), freq="D")
    R = np.full((len(grid),) + res.shape[1:], np.nan)
    R[grid.get_indexer(t)] = res
    R = R - np.nanmean(R, 0)
    v = np.nanmean(R * R, 0)
    infl = np.ones(res.shape[1:])
    for k in range(1, lags + 1):
        rk = np.nanmean(R[k:] * R[:-k], 0) / v
        infl = infl + 2 * (1 - k / (lags + 1)) * np.nan_to_num(rk)
    return np.clip(infl, 1, 5)


def bh_fdr(p, alpha=0.10):
    """Benjamini-Hochberg: boolean mask of discoveries among finite p (Wilks 2016, alpha_FDR)."""
    p = np.asarray(p, float); ok = np.isfinite(p)
    ps = np.sort(p[ok]); n = ps.size
    if n == 0:
        return np.zeros_like(p, bool)
    below = ps <= alpha * np.arange(1, n + 1) / n
    thr = ps[below].max() if below.any() else -1
    return ok & (p <= thr)


def norm_p(z):
    from math import erfc, sqrt
    return np.vectorize(lambda x: erfc(abs(x) / sqrt(2)) if np.isfinite(x) else np.nan)(z)


def fill_tropics(psi, lat, edge=15.0):
    """Replace |lat| < edge (downward control undefined as fhat -> 0) by linear interpolation in sin(lat) between the
    edges - the profile of uniform tropical upwelling."""
    out = psi.copy(); s = np.sin(np.deg2rad(lat))
    i0, i1 = int(np.argmin(np.abs(lat + edge))), int(np.argmin(np.abs(lat - edge)))
    w = (s[i0 + 1:i1] - s[i0]) / (s[i1] - s[i0])
    out[:, i0 + 1:i1] = psi[:, [i0]] + (psi[:, [i1]] - psi[:, [i0]]) * w[None, :]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
NAVY, SIENNA, BAND, NORMAL = "#24466e", "#b4532a", "#e2e5e9", "#59636f"
WIN = 30                                                  # days in the running / section mean


def _style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa1a9")
    ax.tick_params(colors=INK, labelsize=9.5, length=3)


def noise_model(C, an, key):
    """Day-to-day behaviour of the GEOS FP index about the MERRA-2 normal: the variance inflation of a mean of daily
    values (acf_inflation) and the lag-1 autocorrelation, from the analysis tail."""
    res = (an[key] - clim_at(C, f"{key}_mean", an.index.dayofyear.values)).values
    step = float(np.median(np.diff(an.index.values).astype("timedelta64[h]").astype(float))) / 24
    infl = float(acf_inflation(res[:, None], an.index)[0])
    r = res - res.mean()
    return {"step_days": step, "inflation": infl, "rho_lag1": float(np.sum(r[1:] * r[:-1]) / np.sum(r * r))}


def band_sd(C, key, doys, nm):
    """sd of a WIN-day mean of DAILY values for these days of year: MERRA-2 year-to-year variance (estimate clipped at 0
    plus one standard error) + MERRA-2 day-to-day variance / WIN, inflated for the daily autocorrelation."""
    var_ia = max(float(C[f"{key}_via30"]), 0.0) + float(C[f"{key}_via30_se"])
    return np.sqrt(var_ia + clim_at(C, f"{key}_var", doys) / WIN * nm["inflation"])


def index_stats(C, S, key, an, off, nm):
    """Last-WIN-day GEOS FP mean vs the MERRA-2 normal for the same days. z-test; the variance holds the GEOS FP mean's
    own sampling noise (from its samples, AR(1)-inflated), MERRA-2's year-to-year variance of WIN-day means (estimate
    clipped at 0 plus one standard error: conservative), the normal's standard error and, if applied, the overlap
    offset's."""
    end = an.index.max(); w = an.index > end - pd.Timedelta(days=WIN)
    x = an[key][w]; doys = x.index.dayofyear.values
    mu = float(np.mean(clim_at(C, f"{key}_mean", doys)))
    T = pd.DatetimeIndex(pd.to_datetime(S["time"]))
    var_ia = max(float(C[f"{key}_via30"]), 0.0) + float(C[f"{key}_via30_se"])
    var_mu = float(harm_mean_se(S[key], T, doys) ** 2)
    # n samples spread over WIN days: the daily-mean inflation applies to the daily-equivalent count
    n_eq = min(len(x), WIN / nm["step_days"])
    var_x = float(np.mean(clim_at(C, f"{key}_var", doys)) / n_eq * nm["inflation"])
    delta, var_d = 0.0, 0.0
    if off is not None:
        delta, var_d = off[key]["diff"], off[key]["diff_se"] ** 2
    xa = float(x.mean()) - delta
    z = (xa - mu) / np.sqrt(var_x + var_ia + var_mu + var_d)
    p = float(norm_p(z))
    return {"window": [f"{x.index.min():%Y-%m-%d}", f"{end:%Y-%m-%d}"], "n": int(len(x)), "geosfp_mean": round(float(x.mean()), 2),
            "offset_removed": round(delta, 2), "geosfp_adjusted": round(xa, 2), "merra2_normal": round(mu, 2),
            "anomaly": round(xa - mu, 2), "anomaly_pct": round(100 * (xa - mu) / mu, 1), "z": round(float(z), 2),
            "p": round(p, 3), "significant_5pct": bool(p < 0.05),
            "sd_parts": {"geosfp_sampling": round(np.sqrt(var_x), 2), "merra2_interannual": round(np.sqrt(var_ia), 2),
                         "normal_se": round(np.sqrt(var_mu), 2), "offset_se": round(np.sqrt(var_d), 2)}}


YLIM = {"F70": (0, 12), "F100": (0, 18)}          # fixed scales; the top steps up by 2 only if a line needs it


def fig_upwelling(C, an, fc, cyc, stats, notes, off, nm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cyc_t = pd.Timestamp(cyc[:8])
    lead = fc[fc.index > an.index.max()]
    both = pd.concat([an, lead])
    minp = int(0.8 * WIN / nm["F70"]["step_days"])
    t0 = an.index.min() + pd.Timedelta(days=WIN - 1)
    t1 = fc.index.max() + pd.Timedelta(days=1)
    days = pd.date_range(t0 - pd.Timedelta(days=1), t1, freq="D"); doy = days.dayofyear.values
    fig = plt.figure(figsize=(12.4, 7.9), dpi=130)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.28, left=0.058, right=0.985, top=0.88, bottom=0.135)
    for i, (key, lev) in enumerate((("F70", 70), ("F100", 100))):
        ax = fig.add_subplot(gs[i]); _style(ax)
        mu = clim_at(C, f"{key}_mean", doy); sd = band_sd(C, key, doy, nm[key])
        ax.fill_between(days, mu - 1.2816 * sd, mu + 1.2816 * sd, color=BAND, lw=0, label="MERRA-2 10–90 % range of 30-day means")
        ax.plot(days, mu, color=NORMAL, lw=1.5, ls=(0, (5, 3)),
                label=f"MERRA-2 normal ({len(C['years'])} years {C['years'][0]}–{C['years'][-1]})")
        delta = stats[key]["offset_removed"]
        run = (both[key] - delta).rolling(f"{WIN}D", min_periods=minp).mean()
        ra = run[(run.index <= an.index.max()) & (run.index >= t0)]
        rf = run[run.index >= an.index.max()]
        scale = " (on the MERRA-2 scale)" if off is not None else ""
        ax.plot(ra.index, ra.values, color=NAVY, lw=2.6, label=f"GEOS FP analyses{scale}")
        ax.plot(rf.index, rf.values, color=SIENNA, lw=2.6, ls=(0, (2.5, 1.4)), label="including the GEOS FP forecast days")
        if len(ra):
            ax.plot([ra.index[-1]], [ra.values[-1]], "o", ms=5, color=NAVY)
        ax.axvline(an.index.max(), color=MUTED, lw=0.8, ls=":")
        top = YLIM[key][1]
        hi = np.nanmax(np.r_[run.values[np.isfinite(run.values)], mu + 1.2816 * sd])
        while hi > top - 0.5:
            top += 2
        ax.set_xlim(t0, t1); ax.set_ylim(YLIM[key][0], top)
        ax.set_ylabel("10⁹ kg s⁻¹", fontsize=9.5, color=INK)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1 if (t1 - t0).days < 60 else 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        s = stats[key]
        sig = (f"significant at 5 %, p = {s['p']:.2f}" if s["significant_5pct"] else f"not significant, p = {s['p']:.2f}")
        ax.set_title(f"{lev} hPa", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_title(f"{WIN} days to {an.index.max():%b %-d}: {s['geosfp_adjusted']:.1f} vs normal {s['merra2_normal']:.1f} "
                     f"({s['anomaly_pct']:+.0f} %, {sig})", loc="right", fontsize=10, color=INK)
        if i == 0:
            ax.legend(fontsize=9.2, frameon=False, ncol=2, loc="upper left", handlelength=2.6, columnspacing=2.2)
            ax.text(an.index.max() + pd.Timedelta(hours=12), 0.035, "forecast →", transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color=MUTED, va="bottom")
    fig.suptitle(f"GEOS FP · Brewer–Dobson tropical upwelling · analyses to {an.index.max():%b %-d}, forecast from "
                 f"{cyc[:4]}-{cyc[4:6]}-{cyc[6:8]} 00Z", x=0.058, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.058, 0.93, f"Upward mass flux between the turnaround latitudes by downward control (resolved waves + gravity-wave "
             f"drag), trailing {WIN}-day mean", fontsize=10, color=MUTED, va="top")
    fig.text(0.058, 0.018, notes, fontsize=8.8, color=MUTED, va="bottom", linespacing=1.45)
    fig.savefig(path.with_suffix(".webp"), format="webp", pil_kwargs={"quality": 90})
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def section_stats(C, S, an, PS_an, off, lat):
    """Last-WIN-day mean psi* anomaly per cell with the same variance terms as index_stats, BH-FDR at 10 %."""
    end = an.index.max(); w = np.asarray(an.index > end - pd.Timedelta(days=WIN))
    doys = an.index[w].dayofyear.values
    T = pd.DatetimeIndex(pd.to_datetime(S["time"]))
    clim = np.mean([psi_clim(C, d) for d in doys], 0)
    X = PS_an[w]; xm = np.nanmean(X, 0)
    # residuals of the whole tail about the MERRA-2 normal, for the autocorrelation inflation per cell
    res = PS_an - np.array([psi_clim(C, d) for d in an.index.dayofyear.values])
    infl = acf_inflation(res, an.index)
    var_x = np.nanvar(X, 0, ddof=1) / w.sum() * infl
    est, se = interannual_var(S["psi"], T, half=WIN // 2)
    var_ia = np.maximum(est, 0) + se
    var_mu = harm_mean_se(S["psi"], T, doys) ** 2
    delta, var_d = (off["psi_diff"], off["psi_diff_se"] ** 2) if off is not None else (0.0, 0.0)
    anom = xm - delta - clim
    z = anom / np.sqrt(var_x + var_ia + var_mu + var_d)
    p = norm_p(z)
    dom = ((LEVS >= 1) & (LEVS <= 100))[:, None] & ((np.abs(lat) >= 15) & (np.abs(lat) <= 80))[None, :]
    p = np.where(dom, p, np.nan)
    sig = bh_fdr(p, 0.10)
    return {"mean": xm, "clim": clim, "anom": anom, "z": z, "sig": sig, "n": int(w.sum()),
            "window": (an.index[w].min(), end), "inflation": infl}


def flow_arrows(ax, lat, p, psi, css, min_px=110, avoid_lat=9.0):
    """Arrowheads on the Psi* contours in the direction of the residual circulation (2026-09-27, user: "can we add arrows
    to the residual circulation?"). With v* = (g / 2 pi a cos phi) dPsi/dp and w* ~ -(g / 2 pi a^2 cos phi) dPsi/dphi the
    flow runs along the contours, clockwise round a positive (NH) cell and anticlockwise round a negative (SH) one in this
    view (north right, height up): up in the tropics, poleward aloft, down at high latitude. The direction is taken on
    the drawn axes (latitude against log pressure), so each head lies along its line as the eye sees it: in display
    pixels, (u, w) = (-dPsi/dy, dPsi/dx). One head per contour piece, two on long ones; none inside |lat| < avoid_lat,
    where the label sits."""
    from scipy.interpolate import RegularGridInterpolator
    T = ax.transData
    o = np.argsort(p); pp, ps = p[o], psi[o]
    xd = T.transform(np.c_[lat, np.full(len(lat), pp[0])])[:, 0]
    yd = T.transform(np.c_[np.zeros(len(pp)), pp])[:, 1]
    gy, gx = np.gradient(ps, yd, xd)
    iu = RegularGridInterpolator((pp, lat), -gy, bounds_error=False, fill_value=np.nan)
    iw = RegularGridInterpolator((pp, lat), gx, bounds_error=False, fill_value=np.nan)
    for cs in css:
        for segs in cs.allsegs:
            for seg in segs:
                if len(seg) < 4:
                    continue
                D = T.transform(seg)
                arc = np.r_[0, np.cumsum(np.hypot(*np.diff(D, axis=0).T))]
                if arc[-1] < min_px:
                    continue
                for frac in ((0.5,) if arc[-1] < 3 * min_px else (0.3, 0.7)):
                    i = int(np.clip(np.searchsorted(arc, frac * arc[-1]), 1, len(seg) - 2))
                    la, pr = seg[i]
                    if abs(la) < avoid_lat or not (pp[0] < pr < pp[-1]):
                        continue
                    t = D[i + 1] - D[i - 1]
                    v = np.array([float(iu([[pr, la]])[0]), float(iw([[pr, la]])[0])])
                    if not np.isfinite(v).all() or np.hypot(*t) == 0:
                        continue
                    t = t / np.hypot(*t) * (1.0 if t @ v >= 0 else -1.0)
                    a, b = T.inverted().transform(np.vstack([D[i] - 5 * t, D[i] + 5 * t]))
                    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                                                    mutation_scale=12, shrinkA=0, shrinkB=0), zorder=5)


def fig_section(C, sec, lat, cyc, off, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.ndimage import gaussian_filter1d
    k = (LEVS >= 1) & (LEVS <= 100)
    p = LEVS[k]
    # contours: the mean circulation, tropics filled, smoothed in latitude for display only (1.5-deg Gaussian)
    mean = gaussian_filter1d(fill_tropics(sec["mean"], lat), 1.5, axis=1, mode="nearest")[k]
    clim = sec["clim"][k]
    # shading: strength vs normal in % (+ = stronger in either hemisphere), significant cells only
    floor = 0.05 * np.nanmax(np.abs(clim), axis=1, keepdims=True)
    pct = np.where(np.abs(clim) > floor, 100 * sec["anom"][k] * np.sign(clim) / np.abs(clim), np.nan)
    shade = np.where(sec["sig"][k], pct, np.nan)
    lv = [-100, -60, -40, -25, -10, 10, 25, 40, 60, 100]
    cols = ["#3b4f8f", "#5a78b4", "#8ea8d2", "#c3d1e8", "#ffffff", "#f3c9a8", "#e59c6c", "#cc6a3a", "#9c3f1c"]
    cmap = ListedColormap(cols); cmap.set_over("#6e2710"); cmap.set_under("#26345f")
    norm = BoundaryNorm(lv, cmap.N)
    fig = plt.figure(figsize=(11.6, 6.7), dpi=130)
    ax = fig.add_axes([0.065, 0.255, 0.915, 0.60])
    cf = ax.contourf(lat, p, np.ma.masked_invalid(shade) if np.isfinite(shade).any() else np.zeros_like(pct),
                     levels=lv, cmap=cmap, norm=norm, extend="both")
    cl = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    cs1 = ax.contour(lat, p, mean, levels=cl, colors=INK, linewidths=0.95)
    cs2 = ax.contour(lat, p, mean, levels=-cl[::-1], colors=INK, linewidths=0.95, linestyles="dashed")
    for cs in (cs1, cs2):
        ax.clabel(cs, fontsize=7.8, fmt=lambda v: f"{v:g}", inline_spacing=2)
    ax.axvspan(-15, 15, facecolor="none", edgecolor="#c3c8ce", hatch="////", lw=0)
    ax.set_yscale("log"); ax.set_ylim(100, 1); ax.set_xlim(-80, 80)
    yt = [100, 70, 50, 30, 20, 10, 5, 3, 2, 1]
    ax.set_yticks(yt); ax.set_yticklabels([str(y) for y in yt]); ax.minorticks_off()
    ax.set_xticks(np.arange(-80, 81, 20))
    ax.set_xticklabels([f"{abs(x)}°{'S' if x < 0 else ('N' if x > 0 else '')}" for x in range(-80, 81, 20)])
    _style(ax); ax.grid(False)
    flow_arrows(ax, lat, p, mean, (cs1, cs2))
    ax.set_ylabel("hPa", fontsize=9.5, color=INK)
    ax.text(0, 0.975, "interpolated:\ndownward control\nundefined near\nthe equator", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.8, color=MUTED, bbox=dict(fc="white", ec="none", pad=1.5))
    d0, d1 = sec["window"]
    fig.suptitle(f"GEOS FP · Brewer–Dobson circulation Ψ* · {WIN}-day mean to {d1:%b %-d %Y}",
                 x=0.065, y=0.972, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.065, 0.922, f"Residual streamfunction by downward control, mean of {sec['n']} analyses {d0:%b %-d}–{d1:%b %-d}, "
             "10⁹ kg s⁻¹ · solid: NH cell · dashed: SH cell · arrows: flow direction", fontsize=10, color=MUTED, va="top")
    nsig = int(sec["sig"].sum())
    cax = fig.add_axes([0.30, 0.155, 0.45, 0.023])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lv)
    cb.ax.tick_params(labelsize=8.5, colors=INK); cb.outline.set_visible(False)
    cb.set_label("weaker ← circulation strength, % of MERRA-2 normal → stronger", fontsize=9.5, color=INK)
    foot = (f"Shading only where significant: z-test per point against MERRA-2 ({len(C['years'])} years {C['years'][0]}–"
            f"{C['years'][-1]}; day-to-day and year-to-year variance),\nfalse-discovery rate 10 % over 100–1 hPa and 15–80°."
            + (f" The GEOS FP–MERRA-2 difference measured on {off['n']} same days in 2018–19 is removed first." if off is not None else "")
            + "\n" + (f"{nsig} points pass." if nsig else "No point passes: no significant anomaly in this window."))
    fig.text(0.065, 0.02, foot, fontsize=8.8, color=MUTED, va="bottom", linespacing=1.45)
    fig.savefig(path.with_suffix(".webp"), format="webp", pil_kwargs={"quality": 90})
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# Render + live
# ----------------------------------------------------------------------------------------------------------------------
GWD_PERSIST_DAYS = 5
TAIL_DAYS = 106                                            # analysis history used and drawn (the 30-day line starts 30 days in)


def gwd_persistence_test(C, an, GW, recs, leads=(1, 5, 10)):
    """Hindcast of the forecast drag on the analysis tail: the index from each day's own state with (a) its own drag,
    (b) the mean drag of the 5 days ending `lead` days earlier (what the forecast uses), (c) the MERRA-2 day-of-year drag.
    Returns the bias and RMSE of (b) and (c) against (a)."""
    out = {}
    for lead in leads:
        e = {"persist": {"F70": [], "F100": []}, "merra2_clim": {"F70": [], "F100": []}}
        for i, d in enumerate(an.index):
            w = (an.index <= d - pd.Timedelta(days=lead)) & (an.index > d - pd.Timedelta(days=lead + GWD_PERSIST_DAYS))
            if not w.any():
                continue
            doy = d.dayofyear
            act = dc_record(recs[i], doy)
            per = dc_record(recs[i], doy, X=GW[w].mean(0))
            cli = dc_record(recs[i], doy, X=gwd_clim(C, doy))
            for key in ("F70", "F100"):
                e["persist"][key].append(per[key] - act[key]); e["merra2_clim"][key].append(cli[key] - act[key])
        out[f"lead_{lead}d"] = {m: {k: {"bias": round(float(np.mean(v)), 3), "rmse": round(float(np.sqrt(np.mean(np.square(v)))), 3),
                                        "n": len(v)} for k, v in dd.items()} for m, dd in e.items()}
    return out


def render():
    """Everything from the caches: figures (webp + png) and bdc_dc.json in OUT."""
    if not (OUT / "clim_merra2.npz").exists():
        build_clim()
    C = load(OUT / "clim_merra2.npz"); S = load(OUT / "clim_merra2_samples.npz") | load(OUT / "clim_merra2_index.npz")
    an, PS_an, GW, recs = tail_frame()
    keep = np.asarray(an.index > an.index.max() - pd.Timedelta(days=TAIL_DAYS))
    an, PS_an, GW, recs = an[keep], PS_an[keep], GW[keep], [r for r, k in zip(recs, keep) if k]
    lat = recs[0]["lat"]
    off = overlap_offset()
    fcf = sorted(f for f in FCST.glob("*.npz") if re.fullmatch(r"\d{8}_00\.npz", f.name))[-1]
    cyc = str(load(fcf)["cycle"]); cyc_t = pd.Timestamp(cyc[:8])
    X_fc, n_gw = persisted_gwd(an, GW, cyc_t, GWD_PERSIST_DAYS)
    fc, PS_fc, _ = forecast_frame(fcf, X_fc)
    gwd_note = (f"Gravity-wave drag: GEOS FP's own analyses; the forecast holds the last {GWD_PERSIST_DAYS} days' mean drag "
                "(GEOS FP publishes no forecast wind tendencies).")
    stats, val, nm = {}, {}, {}
    for key in ("F70", "F100"):
        nm[key] = noise_model(C, an, key)
        stats[key] = index_stats(C, S, key, an, off, nm[key])
        stats[key]["noise_model"] = {k: round(v, 3) for k, v in nm[key].items()}
        q10, q90 = clim_at(C, f"{key}_q10", an.index.dayofyear.values), clim_at(C, f"{key}_q90", an.index.dayofyear.values)
        inside = float(np.mean((an[key].values >= q10) & (an[key].values <= q90)))
        f0 = fc[key].iloc[0]; a0 = an[key].get(cyc_t, np.nan)
        dd = int(cyc_t.dayofyear)
        val[key] = {"analysis_days": int(len(an)), "analysis_mean": round(float(an[key].mean()), 2),
                    "merra2_normal_same_days": round(float(np.mean(clim_at(C, f"{key}_mean", an.index.dayofyear.values))), 2),
                    "share_inside_merra2_10_90": round(inside, 2),
                    "forecast_day0": round(float(f0), 2), "analysis_same_day": None if np.isnan(a0) else round(float(a0), 2),
                    "merra2_q10_q50_q90_day0": [round(float(clim_at(C, f"{key}_q{q}", dd)), 2) for q in (10, 50, 90)],
                    "forecast_d1_10_mean": round(float(fc[key].iloc[1:].mean()), 2),
                    "merra2_normal_d1_10": round(float(np.mean(clim_at(C, f"{key}_mean", fc.index[1:].dayofyear.values))), 2),
                    "ep_part_mean": round(float(an[f"{key}_ep"].mean()), 2), "gw_part_mean": round(float(an[f"{key}_gw"].mean()), 2),
                    "ana_increment_would_add": round(float(an[f"{key}_ana"].mean()), 2)}
    if off is None:
        scale_note = "GEOS FP is not adjusted to MERRA-2 (no same-day comparison on file)."
    else:
        scale_note = (f"GEOS FP is shifted {-off['F70']['diff']:+.1f} (70 hPa) and {-off['F100']['diff']:+.1f} (100 hPa): its mean "
                      f"same-day difference from MERRA-2 on {off['n']} days in 2018–19, of which the weaker gravity-wave drag is "
                      f"{-off['F70_gw']['diff']:+.1f} / {-off['F100_gw']['diff']:+.1f}.")
    notes = (gwd_note + "\n" + scale_note + "\nTest: z-test of the " + f"{WIN}-day mean against MERRA-2's day-to-day and "
             "year-to-year variance plus the errors of the normal and of the shift.")
    fig_upwelling(C, an, fc, cyc, stats, notes, off, nm, OUT / "bdc_dc_upwelling")
    sec = section_stats(C, S, an, PS_an, off, lat)
    fig_section(C, sec, lat, cyc, off, OUT / "bdc_dc_section")
    js = {"cycle": cyc, "made": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "units": "1e9 kg/s",
          "method": "downward control: psi* = -(2 pi a cos phi / g) * int_top^p (div F / (a cos phi) + X) / fhat dp",
          "gwd_source": {"analysis": "GEOS FP assim tavg3_3d_udt_Nv DUDTGWD, calendar-day mean of 4 windows",
                         "forecast": f"mean of the last {GWD_PERSIST_DAYS} analysis days ({n_gw} on file), held fixed"},
          "climatology": {"source": "MERRA-2 GMI replay (merra2_gmi inst3_3d_met_Np + tavg24_3d_dad_Np), 00Z every 10th day",
                          "years": [int(y) for y in C["years"]], "n_days": int(C["n_days"])},
          "last30": stats, "validation": val, "gwd_forecast_hindcast": gwd_persistence_test(C, an, GW, recs),
          "overlap_2018_19": None if off is None else {"n_days": off["n"], **{k: {kk: round(vv, 3) for kk, vv in off[k].items()}
                                                                               for k in KEYS}},
          "section": {"window": [f"{sec['window'][0]:%Y-%m-%d}", f"{sec['window'][1]:%Y-%m-%d}"], "n": sec["n"],
                      "significant_cells": int(sec["sig"].sum()), "test": "z-test per cell, BH-FDR 10 %, 100-1 hPa, 15-80 deg"},
          "analysis": {f"{t:%Y-%m-%d}": {k: round(float(an.loc[t, k]), 2) for k in ("F70", "F70_ep", "F70_gw", "F100")}
                       for t in an.index},
          "forecast": {f"{t:%Y-%m-%d}": {k: round(float(fc.loc[t, k]), 2) for k in ("F70", "F70_ep", "F70_gw", "F100")}
                       for t in fc.index}}
    # the plotted lines: trailing 30-day means on the MERRA-2 scale, the normal and the 10-90 % band
    lead = fc[fc.index > an.index.max()]
    both = pd.concat([an, lead])
    days = pd.date_range(an.index.min() + pd.Timedelta(days=WIN - 1), fc.index.max(), freq="D")
    minp = int(0.8 * WIN / nm["F70"]["step_days"])
    ser = {}
    for key in ("F70", "F100"):
        run = (both[key] - stats[key]["offset_removed"]).rolling(f"{WIN}D", min_periods=minp).mean()
        mu = clim_at(C, f"{key}_mean", days.dayofyear.values); sd = band_sd(C, key, days.dayofyear.values, nm[key])
        ser[key] = {f"{t:%Y-%m-%d}": {"geosfp_30d": (round(float(run[t]), 2) if t in run.index and np.isfinite(run[t]) else None),
                                      "forecast": bool(t > an.index.max()), "normal": round(float(m), 2),
                                      "p10": round(float(m - 1.2816 * b), 2), "p90": round(float(m + 1.2816 * b), 2)}
                    for t, m, b in zip(days, mu, sd)}
    js["series_30d"] = ser
    (OUT / "bdc_dc.json").write_text(json.dumps(js, indent=1))
    return js


def live(tail_days=10, budget_min=6):
    """Daily step: missing analysis days of the last `tail_days` (normally just the newest; a longer outage is caught up
    within the time budget over successive runs), the newest forecast, figures. Target < 15 min; ~6 min measured."""
    t0 = time.time()
    fetch_tail(tail_days, 1, budget_min=budget_min)
    try:
        fetch_forecast()
    except Exception as e:                                               # noqa: BLE001
        log(f"forecast fetch failed ({str(e)[:120]}); rendering with the newest cached cycle")
    js = render()
    js["runtime_s"] = round(time.time() - t0)
    (OUT / "bdc_dc.json").write_text(json.dumps(js, indent=1))
    log(f"live update done in {js['runtime_s'] / 60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["clim", "tail", "forecast", "overlap", "render", "live"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "clim":
        build_clim()
    elif a.cmd == "tail":
        fetch_tail(a.days, a.step)
    elif a.cmd == "forecast":
        fetch_forecast()
    elif a.cmd == "overlap":
        fetch_overlap()
    elif a.cmd == "render":
        print(json.dumps({k: v for k, v in render().items() if k not in ("analysis", "forecast")}, indent=1))
    elif a.cmd == "live":
        live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
