#!/usr/bin/env python3
"""Teleconnection loading patterns, CPC-style, on NCEP/NCAR R1 — built once.

What this produces is the set of fixed spatial "projectors" that turn a daily
500 hPa height or sea-level-pressure anomaly map into a standardized index
value, plus the standardization constants, plus a validation of each index
against the one CPC publishes. telecon_geps.py then applies the same
projectors to every GEPS member at every lead, and to the NCEP analyses of the
last weeks for the observed tail, so forecast and observation are on one
footing.

Method, and why each choice
  Reanalysis  NCEP/NCAR R1 at 2.5 deg — the dataset CPC's operational patterns
              are defined on (Barnston & Livezey 1987; CPC "Northern Hemisphere
              Teleconnection Patterns"). Using it, rather than ERA5, is what
              lets the indices here be checked against CPC's own numbers, and
              the 2.5 deg grid is more than enough for hemispheric patterns.
  Base period 1991-2020 throughout (CPC's current standardization base).
  NH patterns Defined by REGRESSION ON CPC'S OWN MONTHLY INDICES, month by
              month. For each pattern and calendar month, the map of monthly
              500 hPa height anomalies 20-90N (standardized by calendar month,
              weighted by sqrt(cos lat)) regressed on the CPC index over the
              months within +-1 of that calendar month, 1991-2020; the nine
              maps of a month are combined into a least-squares projector so
              each index is read with the others accounted for. Why not an
              independent rotated PCA: CPC derives its modes separately for
              every calendar month and the PNA in particular changes shape
              with season, so one all-year varimax set reproduced CPC's PNA
              at only r 0.54 (monthly) — the seasonal regression takes it to
              0.74 out of sample. The reproduction is VALIDATED, not assumed:
              patterns fitted on 1991-2005 are scored on 2006-2020 against
              CPC's monthly indices for every pattern and against CPC's daily
              NAO and PNA; those out-of-sample r are what the page reports,
              while the operational projector is refitted on all 30 years.
  AO / AAO    First EOF of monthly sea-level-pressure anomalies poleward of 20
              (Thompson & Wallace 1998; CPC uses 1000 hPa / 700 hPa height for
              the daily AO / AAO, which SLP tracks closely). Signed so that the
              positive phase has low pressure over the pole.
  EPO / WPO   PSL's box definitions on 500 hPa height anomalies (metres, not
              standardized): EPO = (20-35N, 160-125W) minus (55-65N, 160-125W);
              WPO = (25-40N, 140E-150W) minus (50-70N, 140E-150W), area-
              weighted box means, validated against PSL's daily series.
  SOI / EQSOI CPC's two sea-level-pressure ENSO indices, as box differences on
              the R1 slp anomaly (hPa, not standardized per station — with
              the projector fixed by definition the station scaling makes no
              difference to the correlation, r 0.977 vs 0.980, and the raw
              difference keeps the index in one unit). SOI = Tahiti minus
              Darwin, each a 5 x 5 deg box (3 x 3 grid points) centred on the
              nearest 2.5 deg point to the station: Tahiti (17.5S, 149.5W) ->
              20-15S 207.5-212.5E, Darwin (12.5S, 130.9E) -> 15-10S 127.5-
              132.5E (the single-point version scores 0.977 monthly vs CPC,
              the box 0.980). EQSOI = the eastern equatorial Pacific box
              (5S-5N, 130-80W) minus the Indonesia box (5S-5N, 90-140E),
              CPC's definition verbatim. Both are NEGATIVE in El Nino.
              Validated monthly against CPC's published series (soi, the
              STANDARDIZED table; reqsoi.for) over 1991-2020 — a definition,
              not a fit, so every month is out of sample — and the SOI also
              daily against the Long Paddock (Queensland) daily SOI, the
              only daily SOI published anywhere (BoM's is monthly).
  NAM 100     Northern annular mode at 100 hPa: EOF1 of monthly 100 hPa height
              anomalies poleward of 20N with the hemispheric mean removed first
              (Baldwin & Dunkerton 2001 use the cap-versus-midlatitude
              structure, and the secular height trend at 100 hPa, ~+19 m per
              decade, would otherwise masquerade as a weak vortex). Signed so
              the positive phase has low heights over the pole. No hindcast:
              the GEPS8 archive carries no 100 hPa height, so this index gets
              no calibration and says so.
  Seasonal sd The displayed unit is the SEASONAL standard deviation of the daily
              index: sigma_doy, the std of the daily projection within +-15
              days of each day of year over 1991-2020. Indices with a strong
              annual cycle in amplitude (AO, AAO, NAM, EPO, WPO, which are
              projected from raw metres/hPa) otherwise read as permanently
              quiet in summer and every +-0.5 threshold is easy in January
              and hard in July. "+1" here means one seasonal sd for that
              time of year. The annual sigma_daily is kept for the CPC
              comparison: the published CPC/PSL series are annually
              standardized and are rescaled by sigma_daily/sigma_doy when
              drawn beside ours; the validation r is computed on the
              annually standardized series, apples to apples.
  Daily index Daily anomaly (vs a 31-day-smoothed day-of-year climatology),
              the same standardization and weighting, projected with the
              least-squares projector of the rotated modes, then divided by
              the 1991-2020 standard deviation of the daily projection — the
              procedure CPC uses for its daily AO/NAO/PNA. The projector is
              the one for the valid month. The daily index is
              therefore in standard deviations of the DAILY index, and +1 on
              a single day is a routine excursion; a weekly mean of +1 is not.

Outputs (telecon/)
  patterns.nc       projector[index, month, lat, lon] (already weighted; standardized
                    fields use std_month), std_month[month, lat, lon] for z500,
                    clim_doy_{z500,slp}[doy, lat, lon], sigma_daily[index],
                    regression maps for the figure; per-index attributes.
  validation.json   out-of-sample (2006-2020) correlation of each NH index
                    with CPC's monthly series and, for NAO/PNA/AO/AAO, daily.
  geps_telecon_patterns.webp   the loading patterns as regression maps (m per
                    standard deviation), one panel per index, r annotated.

    python build_telecon_patterns.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS                             # noqa: E402  laptop-only inputs/outputs
TC = _LS / "telecon"
NCEP, CPC = TC / "ncep", TC / "cpc"
FIGS = _LS / "figs"
Y0, Y1 = 1991, 2020
HALF = 1                      # +-1 calendar month window for the seasonal regression
SPLIT = 2005                  # fit <= SPLIT, validate after, then refit on everything
# CPC column name -> our short id and long name
CPC_NAMES = {"NAO": ("nao", "North Atlantic Oscillation"),
             "EA": ("ea", "East Atlantic"),
             "WP": ("wp", "West Pacific"),
             "EP/NP": ("epnp", "East Pacific / North Pacific"),
             "PNA": ("pna", "Pacific / North American"),
             "EA/WR": ("eawr", "East Atlantic / West Russia"),
             "SCA": ("sca", "Scandinavia"),
             "TNH": ("tnh", "Tropical / Northern Hemisphere"),
             "POL": ("pol", "Polar / Eurasia")}


# ── data ────────────────────────────────────────────────────────────────────
def monthly(var, level=None):
    d = xr.open_dataset(NCEP / f"{var}.mon.mean.nc")
    a = d[var]
    if level is not None:
        a = a.sel(level=level)
    return a.sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31")).load()


def daily(var, years, level=None, lat=None):
    parts = []
    for y in years:
        p = NCEP / f"{var}.{y}.nc"
        if not p.exists():
            continue
        d = xr.open_dataset(p)
        a = d[var]
        if level is not None:
            a = a.sel(level=level)
        if lat is not None:
            a = a.sel(lat=lat)
        a = a.load().astype("float32")
        if var == "slp" and float(a.max()) > 2000:      # daily files are Pa, monthly hPa
            a = a / 100.0
        parts.append(a)
        d.close()
    a = xr.concat(parts, dim="time")
    # R1 daily files are daily means; drop any duplicate stamps at year seams
    _, idx = np.unique(a.time.values, return_index=True)
    return a.isel(time=np.sort(idx))


def doy_clim(a):
    """Day-of-year mean over the base years, 31-day circular running mean."""
    doy = a.time.dt.dayofyear
    c = a.groupby(doy).mean("time")
    c = c.reindex(dayofyear=np.arange(1, 367)).interpolate_na("dayofyear")
    v = c.values
    ext = np.concatenate([v[-15:], v, v[:15]], axis=0)
    k = np.ones(31) / 31.0
    sm = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 0, ext)
    return c.copy(data=sm.astype("float32")).rename(dayofyear="doy")


# The observed-index readers (CPC/PSL/Long Paddock) live in the runtime module telecon_io.py, which the
# live product and this build share.
from telecon_io import (cpc_monthly, cpc_daily, psl_daily, cpc_soi_monthly,      # noqa: E402,F401
                        cpc_eqsoi_monthly, longpaddock_daily_soi)


BOXES = {  # PSL definitions: (south box) minus (north box), 500 hPa height anomaly in m
    "epo": ("Eastern Pacific Oscillation", (20, 35, 200, 235), (55, 65, 200, 235)),
    "wpo": ("Western Pacific Oscillation", (25, 40, 140, 210), (50, 70, 140, 210)),
}


SLP_BOXES = {  # CPC definitions on the slp anomaly (hPa): (first box) minus (second box)
    "soi": ("Southern Oscillation Index", (-20, -15, 207.5, 212.5), (-15, -10, 127.5, 132.5)),
    "eqsoi": ("Equatorial SOI", (-5, 5, 230, 280), (-5, 5, 90, 140)),
}


def box_weights(lat, lon, box):
    la0, la1, lo0, lo1 = box
    m = ((lat[:, None] >= la0) & (lat[:, None] <= la1)
         & (lon[None, :] >= lo0) & (lon[None, :] <= lo1))
    wgt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * m
    return wgt / wgt.sum()


# ── maths ───────────────────────────────────────────────────────────────────
def varimax(A, gamma=1.0, q=200, tol=1e-7):
    p, k = A.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        B = A @ R
        u, s, vt = np.linalg.svd(A.T @ (B ** 3 - (gamma / p) * B @ np.diag((B ** 2).sum(0))))
        R = u @ vt
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return A @ R


def sigma_by_doy(raw, times, half=15):
    """Std of a daily series within +-half days of each day of year (circular),
    pooled over all years."""
    doy = pd.DatetimeIndex(times).dayofyear.values
    out = np.zeros(366)
    for d in range(1, 367):
        dist = np.abs(((doy - d + 183) % 366) - 183)
        out[d - 1] = raw[dist <= half].std()
    return out


def svd_safe(Z):
    """Thin SVD via the time-time eigenproblem. numpy's LAPACK SVD failed to
    converge on the 360 x 4176 standardized-anomaly matrix (macOS Accelerate);
    the n x n Gram matrix is tiny, symmetric and eigh never fails."""
    C = Z @ Z.T
    ev, U = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    ev, U = np.clip(ev[o], 0, None), U[:, o]
    s = np.sqrt(ev)
    keep = s > s[0] * 1e-8
    Vt = (Z.T @ U[:, keep] / s[keep]).T
    return U[:, keep], s[keep], Vt


def rpca(Z, nmodes):
    """Z: (time, space), already standardized/weighted, no NaN.
    Returns rotated loadings R (space, k) and the least-squares projector
    P = R (R'R)^-1 so that the mode time series are Z @ P."""
    n = Z.shape[0]
    Zc = Z - Z.mean(0)
    u, s, vt = svd_safe(Zc)
    A = vt[:nmodes].T * (s[:nmodes] / np.sqrt(n - 1))       # loadings, sqrt-eigval scaled
    R = varimax(A)
    P = R @ np.linalg.inv(R.T @ R)
    return R, P, (s ** 2 / (s ** 2).sum())[:nmodes]


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    TC.mkdir(exist_ok=True)
    years = list(range(Y0, Y1 + 1))
    out_proj, out_attrs, regmaps, validation, out_sigdoy = {}, {}, {}, {}, {}

    # ---- NH 500 hPa: seasonal regression patterns on CPC's indices ----
    zm = monthly("hgt", 500).sel(lat=slice(90, 20))
    lat, lon = zm.lat.values, zm.lon.values
    w = np.sqrt(np.clip(np.cos(np.deg2rad(lat.astype("float64"))), 0, None))[:, None] \
        * np.ones((1, lon.size))
    mon = zm.time.dt.month
    clim_m = zm.groupby(mon).mean("time")
    std_m = zm.groupby(mon).std("time")
    anom_m = zm.groupby(mon) - clim_m
    Zm = ((anom_m.groupby(mon) / std_m) * w).values.reshape(zm.sizes["time"], -1)
    tidx = pd.DatetimeIndex(zm.time.values).to_period("M").to_timestamp()
    cpc = cpc_monthly().reindex(tidx)
    names = [n for n in CPC_NAMES if n in cpc]
    yrs, mons = tidx.year.values, tidx.month.values

    def fit_month(m, mask):
        near = np.isin(mons, [((m - 1 + d) % 12) + 1 for d in range(-HALF, HALF + 1)])
        R = []
        for n in names:
            c = cpc[n].values
            ok = mask & near & np.isfinite(c)
            if ok.sum() < 20:
                R.append(np.zeros(Zm.shape[1])); continue
            cc = (c[ok] - c[ok].mean()) / c[ok].std()
            R.append(Zm[ok].T @ cc / (cc @ cc))
        R = np.array(R).T
        return R, R @ np.linalg.inv(R.T @ R + 1e-6 * np.eye(len(names)))

    # validation split first
    train, test = yrs <= SPLIT, yrs > SPLIT
    Pv = {m: fit_month(m, train)[1] for m in range(1, 13)}
    Tv = np.array([Zm[i] @ Pv[mons[i]] for i in range(len(tidx))])
    r_month = {}
    for k, n in enumerate(names):
        c = cpc[n].values
        ok = test & np.isfinite(c)
        r_month[n] = float(np.corrcoef(Tv[ok, k], c[ok])[0, 1]) if ok.sum() > 24 else np.nan
    # daily 500 hPa, NH: climatology from all base years
    zd = daily("hgt", years, 500, lat=slice(90, 20))
    clim_d = doy_clim(zd)
    ad = zd.groupby(zd.time.dt.dayofyear) - clim_d.rename(doy="dayofyear")
    ad = ad.drop_vars("dayofyear", errors="ignore")
    dmon = zd.time.dt.month.values
    sd = std_m.sel(month=zd.time.dt.month).values
    Zd = ((ad.values / sd) * w).reshape(zd.sizes["time"], -1)
    dyr = zd.time.dt.year.values
    Tdv = np.array([Zd[i] @ Pv[dmon[i]] for i in range(len(dmon))])
    r_daily = {}
    for n in names:
        sid = CPC_NAMES[n][0]
        cd = cpc_daily(sid)
        if cd is None:
            continue
        k = names.index(n)
        dv = pd.Series(Tdv[:, k], index=zd.time.values)[dyr > SPLIT]
        j = dv.index.intersection(cd.index)
        r_daily[n] = float(np.corrcoef(dv.loc[j], cd.loc[j])[0, 1])
    # operational projector: all 30 years
    Pm = {m: fit_month(m, np.ones_like(train, bool))[1] for m in range(1, 13)}
    Td = np.array([Zd[i] @ Pm[dmon[i]] for i in range(len(dmon))])
    sig = Td.std(0)
    sig_doy = {}
    for k, n in enumerate(names):
        sid, long = CPC_NAMES[n]
        sig_doy[sid] = sigma_by_doy(Td[:, k], zd.time.values)
        proj = np.zeros((12, 73, 144), dtype="float32")
        for m in range(1, 13):
            proj[m - 1, :lat.size] = Pm[m][:, k].reshape(lat.size, lon.size)
        out_proj[sid] = proj
        # regression of the raw monthly anomaly on CPC's index, all months, m per sd
        c = cpc[n].values
        ok = np.isfinite(c)
        cc = (c[ok] - c[ok].mean()) / c[ok].std()
        reg = anom_m.values.reshape(zm.sizes["time"], -1)[ok].T @ cc / (cc @ cc)
        regmaps[sid] = (reg.reshape(lat.size, lon.size), "z500", lat, lon)
        val = {"cpc": n, "r_monthly": round(r_month[n], 3), "validation": f"out of sample, {SPLIT+1}-{Y1}"}
        if n in r_daily:
            val["r_daily"] = round(r_daily[n], 3)
        validation[sid] = val
        out_attrs[sid] = dict(name=long, field="z500", standardize=1, demean=0,
                              sigma_daily=float(sig[k]), hemisphere="nh")
        out_sigdoy[sid] = sig_doy[sid]
        print(f"  {sid:5s} {long:35s} r_month {r_month[n]:+.2f}"
              + (f"  r_daily {r_daily[n]:+.2f}" if n in r_daily else "") + "  (out of sample)")

    # ---- AO / AAO from SLP EOF1 -----------------------------------------
    sm = monthly("slp")
    sdall = daily("slp", years)
    clim_slp_d = doy_clim(sdall)
    for sid, long, hemi, latsl, cpcname in (
            ("ao", "Arctic Oscillation", "nh", slice(90, 20), "ao"),
            ("aao", "Antarctic Oscillation (SAM)", "sh", slice(-20, -90), "aao")):
        s_m = sm.sel(lat=latsl)
        lat2, lon2 = s_m.lat.values, s_m.lon.values
        w2 = np.sqrt(np.clip(np.cos(np.deg2rad(lat2.astype("float64"))), 0, None))[:, None] * np.ones((1, lon2.size))
        mon2 = s_m.time.dt.month
        an2 = (s_m.groupby(mon2) - s_m.groupby(mon2).mean("time"))
        Z2 = (an2.values * w2).reshape(s_m.sizes["time"], -1)
        Z2 = Z2 - Z2.mean(0)
        u, s, vt = svd_safe(Z2)
        e1 = vt[0]
        cap = np.abs(lat2) >= 70
        if e1.reshape(lat2.size, lon2.size)[cap].mean() > 0:
            e1 = -e1                                    # positive phase = low polar pressure
        expl1 = float(s[0] ** 2 / (s ** 2).sum() * 100)
        s_d = sdall.sel(lat=latsl)
        ad2 = (s_d.groupby(s_d.time.dt.dayofyear)
               - clim_slp_d.sel(lat=latsl).rename(doy="dayofyear")).drop_vars("dayofyear", errors="ignore")
        raw = (ad2.values * w2).reshape(s_d.sizes["time"], -1) @ e1
        sig1 = float(raw.std())
        out_sigdoy[sid] = sigma_by_doy(raw, s_d.time.values)
        proj = np.zeros((12, 73, 144), dtype="float32")
        rows = np.where((sm.lat.values <= latsl.start) & (sm.lat.values >= latsl.stop))[0]
        proj[:, rows] = e1.reshape(lat2.size, lon2.size)[None]
        out_proj[sid] = proj
        idx_m = Z2 @ e1
        reg = an2.values.reshape(s_m.sizes["time"], -1).T @ idx_m / (idx_m @ idx_m) * idx_m.std()
        regmaps[sid] = (reg.reshape(lat2.size, lon2.size), "slp", lat2, lon2)
        dv = pd.Series(raw / sig1, index=s_d.time.values)
        val = {"cpc": cpcname.upper(), "explained_pct": round(expl1, 1),
               "validation": f"{Y0}-{Y1} (EOF, no fit to CPC)"}
        cd = cpc_daily(cpcname)
        if cd is not None:
            j = dv.index.intersection(cd.index)
            val["r_daily"] = round(float(np.corrcoef(dv.loc[j], cd.loc[j])[0, 1]), 3)
        validation[sid] = val
        out_attrs[sid] = dict(name=long, field="slp", standardize=0, demean=0,
                              sigma_daily=sig1, hemisphere=hemi)
        print(f"  {sid:5s} {long:35s} EOF1 {expl1:.1f}%"
              + (f"  r_daily {val['r_daily']:+.2f}" if "r_daily" in val else ""))

    # ---- EPO / WPO: PSL box differences on raw 500 hPa anomalies -----------
    latg, long_ = sm.lat.values, sm.lon.values
    wg = np.sqrt(np.clip(np.cos(np.deg2rad(latg.astype("float64"))), 0, None))[:, None] \
        * np.ones((1, long_.size))
    # projector in the weighted-field convention: sum((anom * w) * P) = box mean diff
    zd_full = daily("hgt", years, 500)                     # whole globe, for the box index
    clim_full = doy_clim(zd_full)
    ad_full = (zd_full.groupby(zd_full.time.dt.dayofyear) - clim_full.rename(doy="dayofyear")) \
        .drop_vars("dayofyear", errors="ignore")
    for sid, (long_name, south, north) in BOXES.items():
        P = (box_weights(latg, long_, south) - box_weights(latg, long_, north))
        with np.errstate(divide="ignore", invalid="ignore"):
            Pw = np.where(wg > 0, P / wg, 0.0).astype("float32")
        raw = np.tensordot(ad_full.values, P, axes=([1, 2], [0, 1]))
        sig = float(raw.std())
        out_sigdoy[sid] = sigma_by_doy(raw, zd_full.time.values)
        dv = pd.Series(raw / sig, index=zd_full.time.values)
        val = {"cpc": None, "validation": f"{Y0}-{Y1} vs PSL daily (definition, no fit)"}
        ps = psl_daily(sid)
        if ps is not None:
            j = dv.index.intersection(ps.index)
            r = float(np.corrcoef(dv.loc[j], ps.loc[j])[0, 1])
            if r < 0:
                Pw, raw, r = -Pw, -raw, -r
                dv = -dv
            val["r_daily"] = round(r, 3)
            val["reference"] = "PSL"
        proj = np.zeros((12, 73, 144), dtype="float32")
        proj[:] = Pw[None]
        out_proj[sid] = proj
        # regression map for the figure: monthly global anomaly on the monthly-mean daily index
        mi = dv.resample("MS").mean().reindex(pd.DatetimeIndex(zm.time.values).to_period("M").to_timestamp())
        zfull_m = monthly("hgt", 500)
        an_full_m = zfull_m.groupby(zfull_m.time.dt.month) - zfull_m.groupby(zfull_m.time.dt.month).mean("time")
        ok = np.isfinite(mi.values)
        cc = (mi.values[ok] - mi.values[ok].mean()) / mi.values[ok].std()
        reg = an_full_m.values.reshape(zfull_m.sizes["time"], -1)[ok].T @ cc / (cc @ cc)
        regmaps[sid] = (reg.reshape(latg.size, long_.size)[latg >= 20], "z500", latg[latg >= 20], long_)
        validation[sid] = val
        out_attrs[sid] = dict(name=long_name, field="z500", standardize=0, demean=0,
                              sigma_daily=sig, hemisphere="nh")
        print(f"  {sid:5s} {long_name:35s} box index"
              + (f"  r_daily {val['r_daily']:+.2f} vs PSL" if "r_daily" in val else ""))

    # ---- SOI / EQSOI: CPC box differences on the slp anomaly (hPa) ---------
    # sdall / clim_slp_d are the daily slp and its day-of-year climatology from
    # the AO block; the anomaly is global, the projector picks the boxes.
    ad_slp = (sdall.groupby(sdall.time.dt.dayofyear) - clim_slp_d.rename(doy="dayofyear")) \
        .drop_vars("dayofyear", errors="ignore")
    tm_slp = pd.DatetimeIndex(sdall.time.values)
    slp_m = sm.groupby(sm.time.dt.month) - sm.groupby(sm.time.dt.month).mean("time")
    for sid, (long_name, pos_box, neg_box) in SLP_BOXES.items():
        P = box_weights(latg, long_, pos_box) - box_weights(latg, long_, neg_box)
        with np.errstate(divide="ignore", invalid="ignore"):
            Pw = np.where(wg > 0, P / wg, 0.0).astype("float32")
        raw = np.tensordot(ad_slp.values, P, axes=([1, 2], [0, 1]))
        sig = float(raw.std())
        out_sigdoy[sid] = sigma_by_doy(raw, tm_slp.values)
        dv = pd.Series(raw / sig, index=tm_slp)
        mi = dv.resample("MS").mean()
        cpcm = cpc_soi_monthly() if sid == "soi" else cpc_eqsoi_monthly()
        j = mi.index.intersection(cpcm.index)
        j = j[(j.year >= Y0) & (j.year <= Y1)]
        r_m = float(np.corrcoef(mi.loc[j], cpcm.loc[j])[0, 1])
        j2 = j[j.year > SPLIT]
        r_m2 = float(np.corrcoef(mi.loc[j2], cpcm.loc[j2])[0, 1])
        val = {"cpc": sid.upper(), "reference": "CPC", "r_monthly": round(r_m, 3),
               "r_monthly_after_split": round(r_m2, 3),
               "validation": f"{Y0}-{Y1} vs CPC monthly (definition, no fit; {SPLIT+1}-{Y1} r {r_m2:+.3f})",
               "boxes": {"positive": list(pos_box), "negative": list(neg_box)}}
        if sid == "soi":
            lp = longpaddock_daily_soi()
            if lp is not None:
                jd = dv.index.intersection(lp.index)
                val["r_daily_ref"] = round(float(np.corrcoef(dv.loc[jd], lp.loc[jd])[0, 1]), 3)
                val["daily_reference"] = "Long Paddock (Qld)"
        assert r_m > 0, f"{sid}: sign disagrees with CPC (r {r_m:+.2f})"
        proj = np.zeros((12, 73, 144), dtype="float32")
        proj[:] = Pw[None]
        out_proj[sid] = proj
        # monthly means of the annually standardized daily index have this std:
        # the factor that puts CPC's monthly (unit-variance) value in daily-index units
        monthly_sd = float(mi.loc[str(Y0):str(Y1)].std())
        # regression map for the figure: monthly global slp anomaly on the monthly-mean daily index
        mi_a = mi.reindex(pd.DatetimeIndex(sm.time.values).to_period("M").to_timestamp())
        ok = np.isfinite(mi_a.values)
        cc = (mi_a.values[ok] - mi_a.values[ok].mean()) / mi_a.values[ok].std()
        reg = slp_m.values.reshape(sm.sizes["time"], -1)[ok].T @ cc / (cc @ cc)
        regmaps[sid] = (reg.reshape(latg.size, long_.size), "slp", latg, long_)
        validation[sid] = val
        out_attrs[sid] = dict(name=long_name, field="slp", standardize=0, demean=0,
                              sigma_daily=sig, hemisphere="tropics", monthly_sd=monthly_sd)
        print(f"  {sid:5s} {long_name:35s} box index  r_month {r_m:+.2f} vs CPC ({SPLIT+1}-{Y1} {r_m2:+.2f})"
              + (f"  r_daily {val['r_daily_ref']:+.2f} vs Long Paddock" if "r_daily_ref" in val else ""))

    # ---- NAM at 100 hPa: EOF1 of demeaned height anomalies 20-90N ----------
    z1 = monthly("hgt", 100).sel(lat=slice(90, 20))
    lat1, lon1 = z1.lat.values, z1.lon.values
    w1 = np.sqrt(np.clip(np.cos(np.deg2rad(lat1.astype("float64"))), 0, None))[:, None] \
        * np.ones((1, lon1.size))
    cosw = np.cos(np.deg2rad(lat1))[:, None] * np.ones((1, lon1.size))

    def demean_h(a):                         # remove the cos-weighted 20-90N mean
        return a - (a * cosw).sum(axis=(-2, -1), keepdims=True) / cosw.sum()

    mon1 = z1.time.dt.month
    an1 = demean_h((z1.groupby(mon1) - z1.groupby(mon1).mean("time")).values)
    Z1 = (an1 * w1).reshape(z1.sizes["time"], -1)
    Z1 = Z1 - Z1.mean(0)
    u, s1, vt = svd_safe(Z1)
    e1 = vt[0]
    if e1.reshape(lat1.size, lon1.size)[lat1 >= 70].mean() > 0:
        e1 = -e1
    expl1 = float(s1[0] ** 2 / (s1 ** 2).sum() * 100)
    zd1 = daily("hgt", years, 100, lat=slice(90, 20))
    clim_d1 = doy_clim(zd1)
    ad1 = (zd1.groupby(zd1.time.dt.dayofyear) - clim_d1.rename(doy="dayofyear")) \
        .drop_vars("dayofyear", errors="ignore").values
    raw1 = (demean_h(ad1) * w1).reshape(zd1.sizes["time"], -1) @ e1
    sig1 = float(raw1.std())
    out_sigdoy["nam100"] = sigma_by_doy(raw1, zd1.time.values)
    proj = np.zeros((12, 73, 144), dtype="float32")
    proj[:, :lat1.size] = e1.reshape(lat1.size, lon1.size)[None]
    out_proj["nam100"] = proj
    idx1 = Z1 @ e1
    reg1 = an1.reshape(z1.sizes["time"], -1).T @ idx1 / (idx1 @ idx1) * idx1.std()
    regmaps["nam100"] = (reg1.reshape(lat1.size, lon1.size), "z100", lat1, lon1)
    validation["nam100"] = {"cpc": None, "explained_pct": round(expl1, 1),
                            "validation": "EOF1 of demeaned 100 hPa height, no external reference"}
    out_attrs["nam100"] = dict(name="Northern Annular Mode, 100 hPa", field="z100", standardize=0,
                               demean=1, sigma_daily=sig1, hemisphere="nh")
    clim_doy_z100 = np.zeros((366, 73, 144), "float32")
    clim_doy_z100[:, :lat1.size] = clim_d1.values
    print(f"  nam100 Northern Annular Mode, 100 hPa      EOF1 {expl1:.1f}%  (no hindcast, no reference)")

    # ---- save ------------------------------------------------------------
    ids = list(out_proj)
    ds = xr.Dataset({
        "projector": (("index", "month", "lat", "lon"), np.stack([out_proj[i] for i in ids])),
        "std_month_z500": (("month", "lat", "lon"),
                           np.zeros((12, 73, 144), "float32")),
        "clim_doy_z500": (("doy", "lat", "lon"), np.zeros((366, 73, 144), "float32")),
        "clim_doy_slp": (("doy", "lat", "lon"), clim_slp_d.values.astype("float32")),
        "clim_doy_z100": (("doy", "lat", "lon"), clim_doy_z100),
        "sigma_daily": (("index",), np.array([out_attrs[i]["sigma_daily"] for i in ids])),
        "sigma_doy": (("index", "doy"), np.stack([out_sigdoy[i] for i in ids]).astype("float32")),
        "standardize": (("index",), np.array([out_attrs[i]["standardize"] for i in ids])),
    }, coords={"index": ids, "lat": sm.lat.values, "lon": sm.lon.values,
               "month": np.arange(1, 13), "doy": np.arange(1, 367)})
    ds["std_month_z500"].values[:, :lat.size] = std_m.values
    ds["clim_doy_z500"].values[:, :lat.size] = clim_d.values
    for i in ids:
        for k, v in out_attrs[i].items():
            ds["projector"].attrs[f"{i}_{k}"] = v
    ds.attrs.update(source="NCEP/NCAR Reanalysis 1 (PSL), CPC teleconnection indices",
                    base=f"{Y0}-{Y1}", method="see build_telecon_patterns.py")
    ds.to_netcdf(TC / "patterns.nc")
    meta = {i: {**out_attrs[i], **validation[i]} for i in ids}
    (TC / "validation.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(ids)} indices -> patterns.nc")

    # ---- figure: regression maps ----------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    n = len(ids)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(3.4 * ncol, 3.65 * nrow + 0.7))
    for i, sid in enumerate(ids):
        reg, fld, la, lo = regmaps[sid]
        hemi = out_attrs[sid]["hemisphere"]
        south = hemi == "sh"
        proj = (ccrs.PlateCarree(central_longitude=180) if hemi == "tropics" else
                ccrs.SouthPolarStereo() if south else ccrs.NorthPolarStereo())
        ax = fig.add_subplot(nrow, ncol, i + 1, projection=proj)
        if hemi == "tropics":
            ax.set_extent([-180, 180, -60, 60], ccrs.PlateCarree(central_longitude=180))
        else:
            ax.set_extent([-180, 180, -90 if south else 20, -20 if south else 90], ccrs.PlateCarree())
        lo_w = np.r_[lo, 360.0]
        rg = np.concatenate([reg, reg[:, :1]], axis=1)
        lim = float(np.nanpercentile(np.abs(rg), 99))
        m = ax.pcolormesh(lo_w, la, rg, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          transform=ccrs.PlateCarree(), shading="auto")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="#444")
        ax.gridlines(lw=0.3, color="#999", alpha=0.6)
        for bx in (SLP_BOXES.get(sid) or BOXES.get(sid) or (None,))[1:]:      # box outlines
            la0, la1, lo0, lo1 = bx
            ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], color="#111", lw=1.0,
                    transform=ccrs.PlateCarree())
        v = validation[sid]
        ref = v.get("reference", "CPC")
        rtxt = (f"r vs CPC monthly {v['r_monthly']:+.2f} (out of sample)" if "r_monthly" in v else "") + \
               (f"\nr vs {ref} daily {v['r_daily']:+.2f}" + (" (out of sample)" if "r_monthly" in v else "")
                if "r_daily" in v else "") + \
               (f"\nr vs {v['daily_reference']} daily {v['r_daily_ref']:+.2f}" if "r_daily_ref" in v else "")
        ax.set_title(f"{sid.upper()} — {out_attrs[sid]['name']}\n"
                     f"{ {'z500': '500 hPa height', 'z100': '100 hPa height'}.get(fld, 'sea-level pressure')}, "
                     f"{'hPa' if fld == 'slp' else 'm'} per +1 sd"
                     + (f" · EOF1 {v['explained_pct']}%" if "explained_pct" in v else ""),
                     fontsize=8.4, fontweight="bold")
        ax.text(0.5, -0.04, rtxt, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color="#4a4640")
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.14,
                          aspect=28)
        cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"Teleconnection loading patterns\nregression of the monthly field on each index, "
                 f"NCEP/NCAR R1 {Y0}–{Y1} · CPC indices at 500 hPa, SLP EOF1 for AO/AAO, SLP boxes for SOI/EQSOI",
                 fontsize=11.5, fontweight="bold", y=0.995, va="top")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.905, bottom=0.03, hspace=0.42, wspace=0.08)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "geps_telecon_patterns.webp", dpi=110, facecolor="white",
                pil_kwargs={"quality": 88, "method": 6})
    print("  geps_telecon_patterns.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
