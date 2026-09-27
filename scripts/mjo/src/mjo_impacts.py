#!/usr/bin/env python3
"""MJO impact composites by RMM phase and calendar month, CMIP6 against observed (2026-09-27; user: "cmip6 mjo
composites for the different phases by month"). LAPTOP analysis on the reductions written by cmip6_mjo_extract.py and
mjo_obs_prepare.py; writes the compact site reference scripts/mjo/data/reference/mjo_impacts_site.npz(+ .json) that the
Actions renderer (mjo_impacts_render.py) draws.

RMM for a CMIP6 run (Wheeler & Hendon 2004, projected onto the observed W&H EOFs in data/reference/eofs.nc):
  15S-15N cos-weighted means of OLR (rlut), u850 and u250 (the `day` table has no 200 hPa: 250 hPa stands in) ->
  anomalies from the run's own mean + 3 annual harmonics (calendar-aware: 360-day and noleap runs use their own year)
  -> minus the previous 120-day mean -> each field divided by its own zonal-and-time standard deviation -> projected
  on the observed EOFs -> RMM1/RMM2 BAND-PASSED 20-100 days and renormalised to unit variance (removes ENSO: see
  bandpass()). Phase from atan2(RMM2, RMM1) in the standard 45-degree sectors (phase 1 = [-180, -135) degrees); active
  = amplitude >= 1.
Observed RMM: BoM (1979-2024-02), band-passed the same way (BoM removed ENSO only up to 2013).
Impact fields: daily anomalies from each run's own mean + 3 harmonics + a linear trend, per grid cell.
Composites: mean anomaly on day t+lag over the active days t in phase p whose calendar month falls in the centred
three-month window m-1..m+1 (so "January" = December-February days; it triples the observed events per map), lags
0/5/10/15 d.
Precipitation also as % of the month's normal (cells with a normal under 0.5 mm/day are not scored).

Tests (the site rule: only significant cells are drawn):
  CMIP6  each model's composite (members pooled; a model needs >= 30 active days in the phase-month); across-model
         one-sample t-test, Benjamini-Hochberg FDR 10 % over the map AND >= 80 % of the models agreeing on the sign.
  obs    days are autocorrelated and an MJO passage lasts days, so the unit is the EVENT: a run of consecutive active
         days in one phase. Event-block bootstrap (2,000 resamples) of the day-weighted composite -> z -> two-sided p,
         BH-FDR 10 % over the map; a phase-month needs >= 8 events to be tested at all.

    python mjo_impacts.py validate          # wind-only RMM from ERA5 u850/u200 vs BoM, and the u250-for-u200 proxy
    python mjo_impacts.py screen            # MJO realism of every candidate model (E/W ratio, propagation)
    python mjo_impacts.py compose M1 M2 ... # per-model composites for the screened models -> archive
    python mjo_impacts.py obs               # observed composites + event bootstrap -> archive
    python mjo_impacts.py reference         # multi-model statistics + masks -> the committed site reference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "data" / "reference"
ARC = Path.home() / "data_archive" / "cmip6_mjo"
RED, OBS, COMP = ARC / "red", ARC / "obs", ARC / "comp"
LAGS = (0, 5, 10, 15)
SITE_LAGS = (0, 10)
FIELDS = ("tas_na", "tas_sa", "pr_na", "pr_sa", "z500_nh")
MIN_DAYS, MIN_EVENTS, AGREE, ALPHA = 30, 8, 0.8, 0.10
# Model screen, FIXED before any composite is looked at (Ahn et al. 2020, Le et al. 2021 find good-MJO CMIP6 models at an
# east/west ratio of ~2 and a propagation pattern r of ~0.8 against observations). Fallback if fewer than 6 pass: 1.8 / 0.75.
EW_MIN, PROP_MIN = 2.0, 0.80
EW_MIN2, PROP_MIN2 = 1.8, 0.75
# near-duplicate atmospheres: at most CAP per family, the best (E/W x r) first, so the across-model test is not stacked
FAMILY = {"EC-Earth3": "ecearth", "EC-Earth3-Veg": "ecearth", "EC-Earth3-CC": "ecearth", "EC-Earth3-AerChem": "ecearth",
          "EC-Earth3-Veg-LR": "ecearth", "CMCC-CM2-SR5": "cmcc", "CMCC-ESM2": "cmcc", "CMCC-CM2-HR4": "cmcc",
          "CESM2": "cesm", "CESM2-WACCM": "cesm", "CESM2-FV2": "cesm", "TaiESM1": "cesm",
          "MPI-ESM1-2-HR": "mpi", "MPI-ESM1-2-LR": "mpi", "MPI-ESM-1-2-HAM": "mpi", "AWI-ESM-1-1-LR": "mpi",
          "HadGEM3-GC31-LL": "ukmo", "HadGEM3-GC31-MM": "ukmo", "UKESM1-0-LL": "ukmo", "ACCESS-CM2": "ukmo",
          "CNRM-CM6-1": "cnrm", "CNRM-CM6-1-HR": "cnrm", "CNRM-ESM2-1": "cnrm", "BCC-CSM2-MR": "bcc", "BCC-ESM1": "bcc",
          "INM-CM4-8": "inm", "INM-CM5-0": "inm", "IPSL-CM6A-LR": "ipsl", "IPSL-CM5A2-INCA": "ipsl"}
FAMILY_CAP = 2
WIN = 1          # a "month" is the centred three-month window (m-1, m, m+1): triples the observed events per map


# ----------------------------------------------------------------------------------------------------------- basics
def year_frac(dates, calendar):
    d = pd.Series(dates)
    y = d.str.slice(0, 4).astype(int).values; m = d.str.slice(5, 7).astype(int).values; dd = d.str.slice(8, 10).astype(int).values
    if calendar in ("360_day",):
        doy = (m - 1) * 30 + dd; n = 360.0
    else:
        cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        leap = ((y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))) & (calendar not in ("noleap", "365_day"))
        doy = cum[m - 1] + dd + ((m > 2) & leap)
        n = np.where(leap, 366.0, 365.0) if calendar not in ("noleap", "365_day") else 365.0
    return y, m, (doy - 1) / n


def design(frac, t, harm=3, trend=True):
    cols = [np.ones_like(frac)]
    for k in range(1, harm + 1):
        cols += [np.cos(2 * np.pi * k * frac), np.sin(2 * np.pi * k * frac)]
    if trend:
        cols.append((t - t.mean()) / (t.std() + 1e-9))
    return np.column_stack(cols)


def anomalies(Y, frac, t, trend=True):
    """Y (n, ...) -> anomalies from mean + 3 harmonics (+ linear trend); also the fitted climatology without trend."""
    sh = Y.shape; Y2 = Y.reshape(sh[0], -1).astype(np.float64)
    ok = np.isfinite(Y2)
    X = design(frac, t, trend=trend)
    Yf = np.where(ok, Y2, 0.0)
    B = np.linalg.lstsq(X, Yf, rcond=None)[0]
    fit = X @ B
    clim = design(frac, t, trend=False) @ B[: 1 + 6]
    A = np.where(ok, Y2 - fit, np.nan)
    return A.reshape(sh).astype(np.float32), clim.reshape(sh).astype(np.float32)


def band_mean(a, lat):
    """cos-weighted latitude mean (t, lat, lon) -> (t, lon), NaN-aware: CMIP6 u850 is masked below ground over the Andes and
    the East African highlands; any longitude still empty is filled along longitude (cyclic)."""
    a = a.astype(np.float64); w = np.cos(np.deg2rad(lat))[None, :, None]
    ok = np.isfinite(a)
    num = np.nansum(np.where(ok, a, 0.0) * w, axis=1); den = np.sum(ok * w, axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    bad = ~np.isfinite(out)
    if bad.any():
        nl = out.shape[1]; x = np.arange(nl)
        for t in np.flatnonzero(bad.any(axis=1)):
            g = np.isfinite(out[t])
            if g.sum() >= 2:
                out[t, ~g] = np.interp(x[~g], np.r_[x[g] - nl, x[g], x[g] + nl], np.tile(out[t, g], 3))
    return out


def rmm_from_band(olr, u850, uup, frac, t, eofs, olr_sign=1.0):
    """Daily (t, lon) band means -> RMM1, RMM2 (unit variance), amplitude, phase."""
    comps = []
    for x in (olr * olr_sign, u850, uup):
        a, _ = anomalies(x, frac, t, trend=False)
        a = a.astype(np.float64)
        c = np.cumsum(np.nan_to_num(a), axis=0)
        prev = np.full_like(a, np.nan)
        prev[120:] = (c[119:-1] - np.concatenate([np.zeros((1, a.shape[1])), c[:-121]])) / 120.0
        a = a - prev
        sd = np.nanstd(a)
        a = a / sd if sd > 0 else np.zeros_like(a)
        comps.append(a)
    V = np.concatenate(comps, axis=1)
    e1 = np.concatenate([eofs["eof_olr"][0], eofs["eof_u850"][0], eofs["eof_u200"][0]])
    e2 = np.concatenate([eofs["eof_olr"][1], eofs["eof_u850"][1], eofs["eof_u200"][1]])
    r1 = V @ e1; r2 = V @ e2
    r1 = r1 / np.nanstd(r1); r2 = r2 / np.nanstd(r2)
    return r1, r2


def phase_amp(r1, r2):
    ang = np.degrees(np.arctan2(r2, r1))
    ph = (np.floor((ang + 180.0) / 45.0).astype(int) % 8) + 1
    return ph, np.hypot(r1, r2)


def load_eofs():
    import xarray as xr
    e = xr.open_dataset(REF / "eofs.nc")
    return {k: e[k].values for k in ("eof_olr", "eof_u850", "eof_u200")}


def events(ph, active):
    """Event id per day: a run of consecutive active days in one phase; -1 when inactive."""
    ev = np.full(len(ph), -1); k = -1
    for i in range(len(ph)):
        if active[i]:
            if i == 0 or not active[i - 1] or ph[i] != ph[i - 1]:
                k += 1
            ev[i] = k
    return ev


def bh(p, alpha=ALPHA):
    p = np.asarray(p, float); m = np.isfinite(p); out = np.zeros(p.shape, bool)
    if m.sum() == 0:
        return out
    q = p[m]; o = np.argsort(q); n = len(q)
    thr = alpha * np.arange(1, n + 1) / n
    ok = q[o] <= thr
    if ok.any():
        k = np.max(np.flatnonzero(ok)); sel = np.zeros(n, bool); sel[o[: k + 1]] = True
        out[np.flatnonzero(m)[sel]] = True
    return out


# ------------------------------------------------------------------------------------------------ ENSO removal
# BoM's RMM removes the ENSO (SST1) signal only up to 2013; from 2014 only the 120-day mean is gone, and a strong El Nino's
# standing OLR/wind pattern then projects on RMM as a slow "phase 6-8" (2015/16 DJF active on 89 % of days, 2023/24 on 81 %,
# against 64 % in strong El Ninos of the ENSO-removed era). A linear Nino-3.4 regression (the W&H/BoM route) was tried first
# and removed almost nothing (R^2 0.04; 2015/16 0.89 -> 0.81, 2023/24 rose), so BOTH indices are band-passed 20-100 days
# instead (bandpass(); historical composites need no real-time filter), which removes ENSO, the seasonal cycle and synoptic
# noise alike, for the observed index over the whole 1979-2024 record and for every CMIP6 run.
def _prev120(x):
    c = np.cumsum(np.nan_to_num(x)); out = np.full_like(x, np.nan, dtype=float)
    out[120:] = (c[119:-1] - np.concatenate([[0.0], c[:-121]])) / 120.0
    return out


def daily_n34(months, vals, dates):
    """Monthly Nino-3.4 -> anomalies (own calendar-month mean and quadratic trend removed) -> daily by linear interpolation
    between mid-months -> predictors [N34, N34 - previous 120-day mean]."""
    m = pd.Series(months); mon = m.str.slice(5, 7).astype(int).values
    v = np.asarray(vals, float).copy()
    for k in range(1, 13):
        v[mon == k] -= np.nanmean(v[mon == k])
    tt = np.arange(len(v), dtype=float)
    ok = np.isfinite(v)
    v[ok] -= np.polyval(np.polyfit(tt[ok], v[ok], 2), tt[ok])
    # calendar-agnostic time axis in months (360-day and noleap runs have dates pandas cannot parse)
    mid = m.str.slice(0, 4).astype(int).values * 12 + m.str.slice(5, 7).astype(int).values - 1 + 0.5
    D = pd.Series(dates)
    dd = (D.str.slice(0, 4).astype(int).values * 12 + D.str.slice(5, 7).astype(int).values - 1
          + (D.str.slice(8, 10).astype(int).values - 0.5) / 31.0)
    d = np.interp(dd, mid, np.nan_to_num(v))
    return np.column_stack([d, d - np.nan_to_num(_prev120(d))])


def remove_enso(r1, r2, X, fit, renorm=True):
    """OLS of RMM1/RMM2 on X over the fit mask, removed everywhere the fit applies; returns r1, r2 and the R^2."""
    ok = fit & np.isfinite(r1) & np.isfinite(r2)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    out, r2s = [], []
    for r in (r1, r2):
        b = np.linalg.lstsq(A, r[ok], rcond=None)[0]
        pred = np.column_stack([np.ones(len(r)), X]) @ b
        new = r.copy(); new[fit] = r[fit] - (pred[fit] - b[0])
        r2s.append(1 - np.nanvar(r[ok] - A @ b) / np.nanvar(r[ok]))
        if renorm:
            new = new / np.nanstd(new)
        out.append(new)
    return out[0], out[1], r2s


def bandpass(x, lo=20.0, hi=100.0, edge=60):
    """20-100-day FFT band-pass of a daily series (NaN gaps linearly filled first), renormalised to unit variance; the first
    and last `edge` days are set to NaN (filter end effects). Removes ENSO, the seasonal cycle and synoptic noise."""
    x = np.asarray(x, float).copy(); n = len(x)
    ok = np.isfinite(x)
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    x -= x.mean()
    F = np.fft.rfft(x); f = np.fft.rfftfreq(n)
    F[(f < 1.0 / hi) | (f > 1.0 / lo)] = 0.0
    y = np.fft.irfft(F, n=n)
    y[:edge] = np.nan; y[-edge:] = np.nan
    return y / np.nanstd(y)


def obs_rmm():
    """BoM RMM, band-passed 20-100 d over the whole 1979-2024 record (ENSO-free throughout, the same filter as the models)."""
    bom = pd.read_csv(OBS / "rmm_bom.csv")
    bom = bom[bom.date >= "1979-01-01"].reset_index(drop=True)
    full = pd.date_range(bom.date.iloc[0], bom.date.iloc[-1]).strftime("%Y-%m-%d")
    b = bom.set_index("date").reindex(full)
    r1, r2 = bandpass(b.rmm1.values), bandpass(b.rmm2.values)
    ph, amp = phase_amp(r1, r2)
    return np.array(full), ph, amp, b.phase.values, b.amp.values, None


# ------------------------------------------------------------------------------------------------------- composites
def composite(anom, ph, active, months, lags=LAGS, ev=None, boot=0, rng=None):
    """anom (n, cells) -> mean[lag, phase, month, cells], days[phase, month]; with ev and boot: SE from an event bootstrap."""
    n, nc = anom.shape
    M = np.full((len(lags), 8, 12, nc), np.nan, np.float32); D = np.zeros((8, 12), int); NE = np.zeros((8, 12), int)
    SE = np.full_like(M, np.nan) if boot else None
    for p in range(1, 9):
        for mo in range(1, 13):
            win = [((mo - 2 + k) % 12) + 1 for k in range(-WIN, WIN + 1)]          # centred window of months
            idx = np.flatnonzero(active & (ph == p) & np.isin(months, win))
            D[p - 1, mo - 1] = len(idx)
            if ev is not None:
                NE[p - 1, mo - 1] = len(np.unique(ev[idx]))
            if len(idx) == 0:
                continue
            for li, L in enumerate(lags):
                j = idx + L; j = j[j < n]
                if len(j) == 0:
                    continue
                X = anom[j]
                M[li, p - 1, mo - 1] = np.nanmean(X, axis=0)
                if boot and ev is not None:
                    e = ev[j - L]; ue, inv = np.unique(e, return_inverse=True)
                    if len(ue) < 2:
                        continue
                    S = np.zeros((len(ue), nc)); C = np.zeros(len(ue))
                    np.add.at(S, inv, np.nan_to_num(X)); np.add.at(C, inv, 1.0)
                    R = rng.integers(0, len(ue), size=(boot, len(ue)))
                    W = np.zeros((boot, len(ue))); np.add.at(W, (np.repeat(np.arange(boot), len(ue)), R.ravel()), 1.0)
                    bm = (W @ S) / (W @ C)[:, None]
                    SE[li, p - 1, mo - 1] = bm.std(axis=0)
    return M, D, NE, SE


def month_normal(clim, months):
    """Normal for each centred window (same months as the composites), (12, cells)."""
    return np.stack([np.nanmean(clim[np.isin(months, [((mo - 2 + k) % 12) + 1 for k in range(-WIN, WIN + 1)])], axis=0)
                     for mo in range(1, 13)])


# --------------------------------------------------------------------------------------------------------- commands
def validate():
    """Wind-only RMM from ERA5 u850/u200 (store, 1991-2020) against BoM; then u250 in place of u200 (WB2, 2016-2018)."""
    import xarray as xr
    eofs = load_eofs()
    bom = pd.read_csv(OBS / "rmm_bom.csv").set_index("date")
    store = Path.home() / "era5_store" / "wb2_1p5_daily_global"
    lat_b = np.arange(-15, 15.01, 1.5)

    def band(var, y):
        a = xr.open_dataset(store / var / f"{var}_{y}.nc"); v = a[list(a.data_vars)[0]].transpose("time", "latitude", "longitude")
        v = v.sel(latitude=slice(-15, 15)) if v.latitude[0] < v.latitude[-1] else v.sel(latitude=slice(15, -15))
        v = v.interp(longitude=np.arange(0, 360, 2.5), kwargs={"fill_value": "extrapolate"})
        return pd.DatetimeIndex(v.time.values).strftime("%Y-%m-%d").values, band_mean(v.values, v.latitude.values)
    D, U8, U2 = [], [], []
    for y in range(1991, 2021):
        d, u8 = band("u850", y); _, u2 = band("u200", y); D += list(d); U8.append(u8); U2.append(u2)
    D = np.array(D); U8 = np.concatenate(U8); U2 = np.concatenate(U2)
    _, _, frac = year_frac(D, "standard"); t = np.arange(len(D), dtype=float)
    # wind-only: OLR part set to zero
    r1, r2 = rmm_from_band(np.zeros_like(U8), U8, U2, frac, t, {**eofs, "eof_olr": np.zeros_like(eofs["eof_olr"])})
    b = bom.reindex(D)
    ok = np.isfinite(r1) & b.rmm1.notna().values
    c1 = np.corrcoef(r1[ok], b.rmm1.values[ok])[0, 1]; c2 = np.corrcoef(r2[ok], b.rmm2.values[ok])[0, 1]
    ph, amp = phase_amp(r1, r2)
    act = ok & (amp >= 1) & (b.amp.values >= 1)
    same = np.mean(ph[act] == b.phase.values[act]); near = np.mean(((ph[act] - b.phase.values[act]) % 8) <= 1) + 0
    near = np.mean(np.minimum((ph[act] - b.phase.values[act]) % 8, (b.phase.values[act] - ph[act]) % 8) <= 1)
    print(f"ERA5 wind-only RMM vs BoM 1991-2020: r(RMM1) {c1:.3f}, r(RMM2) {c2:.3f}; phase identical on {100 * same:.0f} % "
          f"and within one phase on {100 * near:.0f} % of days active in both")
    res = {"era5_windonly_vs_bom": {"r_rmm1": round(c1, 3), "r_rmm2": round(c2, 3), "same_phase": round(same, 3),
                                   "within_one": round(near, 3), "years": "1991-2020"}}
    # u250 vs u200 (WB2 1.5-degree zarr, 00Z, 2014-2018)
    ds = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                      storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(time=slice("2014-01-01", "2018-12-31"), level=[850, 250, 200])
    u = u.isel(time=(u.time.dt.hour == 0).values).sel(latitude=slice(-15, 15))
    u = u.interp(longitude=np.arange(0, 360, 2.5)).transpose("time", "level", "latitude", "longitude").load()
    d2 = pd.DatetimeIndex(u.time.values).strftime("%Y-%m-%d").values
    _, _, f2 = year_frac(d2, "standard"); t2 = np.arange(len(d2), dtype=float)
    bm = {L: band_mean(u.sel(level=L).values, u.latitude.values) for L in (850, 250, 200)}
    z = {**eofs, "eof_olr": np.zeros_like(eofs["eof_olr"])}
    a1, a2 = rmm_from_band(np.zeros_like(bm[850]), bm[850], bm[200], f2, t2, z)
    b1, b2 = rmm_from_band(np.zeros_like(bm[850]), bm[850], bm[250], f2, t2, z)
    ok = np.isfinite(a1) & np.isfinite(b1)
    pa, aa = phase_amp(a1, a2); pb, ab = phase_amp(b1, b2)
    act = ok & (aa >= 1) & (ab >= 1)
    res["u250_for_u200"] = {"r_rmm1": round(float(np.corrcoef(a1[ok], b1[ok])[0, 1]), 3),
                            "r_rmm2": round(float(np.corrcoef(a2[ok], b2[ok])[0, 1]), 3),
                            "same_phase": round(float(np.mean(pa[act] == pb[act])), 3),
                            "amp_ratio": round(float(np.nanmedian(ab[ok] / aa[ok])), 3), "years": "2014-2018, 00Z, wind-only"}
    print("u250 in place of u200 (wind-only RMM, ERA5 2014-2018):", res["u250_for_u200"])
    (ARC / "validate.json").write_text(json.dumps(res, indent=1))


def _spectra_prop(band_pr, lat, dates, calendar):
    """E/W ratio (k 1-3, 30-96 d, Nov-Apr) and the Indian Ocean lag-regression Hovmoller of 10S-10N rain."""
    y, m, frac = year_frac(dates, calendar)
    sel = np.abs(lat) <= 10.0 + 1e-6
    x = band_mean(band_pr[:, sel, :], lat[sel])
    a, _ = anomalies(x, frac, np.arange(len(x), dtype=float), trend=True)
    a = np.nan_to_num(a.astype(np.float64))
    n, nl = a.shape
    east = west = 0.0
    for yy in np.unique(y)[:-1]:
        s = np.flatnonzero((y == yy) & (m == 11) & (np.r_[0, np.diff(m)] != 0))
        if not len(s):
            continue
        s = s[0]; seg = a[s: s + 180]
        if len(seg) < 180:
            continue
        seg = (seg - seg.mean(0)) * np.hanning(180)[:, None]
        F = np.fft.fft2(seg)                                           # (freq, k)
        P = np.abs(F) ** 2
        fr = np.fft.fftfreq(180); kk = np.fft.fftfreq(nl, 1.0 / nl)
        per = np.where(fr != 0, 1.0 / np.abs(fr), np.inf)
        fb = (per >= 30) & (per <= 96)
        for ki in (1, 2, 3):
            # eastward: frequency and wavenumber of opposite sign in numpy's convention exp(i(k x - w t)) mapping
            e_ = P[np.ix_(fb & (fr > 0), kk == -ki)].sum() + P[np.ix_(fb & (fr < 0), kk == ki)].sum()
            w_ = P[np.ix_(fb & (fr > 0), kk == ki)].sum() + P[np.ix_(fb & (fr < 0), kk == -ki)].sum()
            east += e_; west += w_
    ew = east / west if west > 0 else np.nan
    # 20-100 d band-pass via FFT, Nov-Apr days, regressed on the 75-85E box
    F = np.fft.rfft(a, axis=0); f = np.fft.rfftfreq(n)
    keep = (f >= 1 / 100) & (f <= 1 / 20)
    bp = np.fft.irfft(F * keep[:, None], n=n, axis=0)
    lon = np.arange(0, 360, 2.5)
    box = bp[:, (lon >= 75) & (lon <= 85)].mean(1)
    win = np.isin(m, [11, 12, 1, 2, 3, 4])
    lags = np.arange(-20, 21)
    H = np.zeros((len(lags), nl))
    idx = np.flatnonzero(win)
    idx = idx[(idx >= 20) & (idx < n - 20)]
    bx = box[idx]; bx = (bx - bx.mean()) / bx.std()
    for i, L in enumerate(lags):
        H[i] = (bp[idx + L] * bx[:, None]).mean(0)
    return float(ew), H


def screen():
    import glob
    base = OBS / "gpcp.npz"
    o = np.load(base)
    lat = np.arange(-15, 15.01, 2.5)
    ew_o, H_o = _spectra_prop(o["pr_band"].astype(np.float32), lat, o["dates"], "standard")
    lon = np.arange(0, 360, 2.5); reg = (lon >= 60) & (lon <= 180)
    print(f"GPCP 1997-2024: E/W {ew_o:.2f}", flush=True)
    rows = [{"model": "OBS (GPCP)", "member": "", "ew": round(ew_o, 2), "prop_r": 1.0}]
    for f in sorted(glob.glob(str(RED / "*_pr.npz"))):
        name = Path(f).name[:-7]; model, member = name.rsplit("_", 1)
        d = np.load(f)
        ew, H = _spectra_prop(d["pr_band"].astype(np.float32), lat, d["dates"], str(d["calendar"]))
        a, b = H[:, reg].ravel(), H_o[:, reg].ravel()
        r = float(np.corrcoef(a, b)[0, 1])
        rows.append({"model": model, "member": member, "ew": round(ew, 2), "prop_r": round(r, 3)})
        print(f"{model:18s} {member:12s} E/W {ew:5.2f}  propagation r {r:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ARC / "screen.csv", index=False)
    np.savez_compressed(ARC / "screen_hov.npz", obs=H_o)


def select():
    """Apply the fixed screen and the family cap to screen.csv -> selected.json (the models used)."""
    df = pd.read_csv(ARC / "screen.csv")
    mod = df[~df.model.str.startswith("OBS")].copy()
    ew, pr_ = EW_MIN, PROP_MIN
    mod["pass"] = (mod.ew >= ew) & (mod.prop_r >= pr_)
    if mod["pass"].sum() < 6:
        ew, pr_ = EW_MIN2, PROP_MIN2
        mod["pass"] = (mod.ew >= ew) & (mod.prop_r >= pr_)
    mod["score"] = mod.ew * mod.prop_r
    mod["family"] = mod.model.map(lambda m: FAMILY.get(m, m))
    chosen = []
    for fam, g in mod[mod["pass"]].sort_values("score", ascending=False).groupby("family", sort=False):
        chosen += list(g.model[:FAMILY_CAP])
    out = {"thresholds": {"ew": ew, "prop_r": pr_}, "family_cap": FAMILY_CAP, "selected": sorted(chosen),
           "screen": [dict(model=r.model, member=r.member, ew=float(r.ew), prop_r=float(r.prop_r), **{"pass": bool(r["pass"])})
                      for _, r in mod.sort_values("score", ascending=False).iterrows()]
                     + [dict(model="OBS (GPCP)", member="", ew=float(df[df.model.str.startswith("OBS")].ew.iloc[0]), prop_r=1.0,
                             **{"pass": True})]}
    (ARC / "selected.json").write_text(json.dumps(out, indent=1))
    print(f"thresholds E/W >= {ew}, r >= {pr_}: {int(mod['pass'].sum())} pass; after the family cap {len(chosen)}: {sorted(chosen)}")


def _member_rmm(model, member, eofs):
    r = np.load(RED / f"{model}_{member}_rlut.npz"); u = np.load(RED / f"{model}_{member}_ua.npz")
    dates = r["dates"]; assert (u["dates"] == dates).all(), "rlut/ua dates differ"
    cal = str(r["calendar"]); lat = r["lat_band"]
    _, months, frac = year_frac(dates, cal); t = np.arange(len(dates), dtype=float)
    olr = band_mean(r["rlut_band"].astype(np.float32), lat)
    u8 = band_mean(u["u850_band"].astype(np.float32), lat); u2 = band_mean(u["u250_band"].astype(np.float32), lat)
    r1, r2 = rmm_from_band(olr, u8, u2, frac, t, eofs)
    r1, r2 = bandpass(r1), bandpass(r2)                               # ENSO-free, as the observed index
    r2s = [np.nan, np.nan]
    ph, amp = phase_amp(r1, r2)
    return dates, cal, months, frac, t, ph, amp, r1, r2, r2s


def compose(models):
    """Per-model composites (members pooled, day-weighted), saved to comp/<model>.npz."""
    import glob
    eofs = load_eofs(); COMP.mkdir(parents=True, exist_ok=True)
    for model in models:
        mems = sorted({Path(f).name[len(model) + 1:-7] for f in glob.glob(str(RED / f"{model}_*_ua.npz"))})
        acc = {}; days = np.zeros((8, 12)); stats = []
        for mem in mems:
            dates, cal, months, frac, t, ph, amp, r1, r2, ens_r2 = _member_rmm(model, mem, eofs)
            active = np.isfinite(amp) & (amp >= 1)
            # propagation sanity: mean phase step on consecutive active days (eastward = +)
            st = ((ph[1:] - ph[:-1] + 4) % 8 - 4)[active[1:] & active[:-1]]
            stats.append({"member": mem, "active_frac": round(float(active.mean()), 3),
                          "enso_r2_rmm1": round(float(ens_r2[0]), 3), "enso_r2_rmm2": round(float(ens_r2[1]), 3),
                          "east_steps": round(float(np.mean(st > 0)), 3), "west_steps": round(float(np.mean(st < 0)), 3)})
            for fld in FIELDS:
                var, grid = fld.split("_")
                f = RED / f"{model}_{mem}_{var}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                if not (d["dates"] == dates).all():
                    j = pd.Index(d["dates"]).get_indexer(dates)
                    if (j < 0).any():
                        print(f"  {model} {mem} {fld}: dates misaligned, skipped"); continue
                    Y = d[f"{var}_{grid}"][j].astype(np.float32)
                else:
                    Y = d[f"{var}_{grid}"].astype(np.float32)
                sh = Y.shape; A, C = anomalies(Y, frac, t, trend=True)
                A = A.reshape(sh[0], -1)
                M, D, _, _ = composite(A, ph, active, months)
                acc.setdefault(fld, [np.zeros_like(np.nan_to_num(M)), np.zeros(M.shape[:3] + (1,))])
                acc[fld][0] += np.nan_to_num(M) * D[None, :, :, None]; acc[fld][1] += D[None, :, :, None]
                if var == "pr":
                    Nm = month_normal(C.reshape(sh[0], -1), months)
                    acc.setdefault(f"{fld}_normal", [np.zeros_like(Nm), 0]); acc[f"{fld}_normal"][0] += Nm; acc[f"{fld}_normal"][1] += 1
            days += composite(np.zeros((len(ph), 1)), ph, active, months, lags=(0,))[1]
            print(f"  {model} {mem}: active {stats[-1]['active_frac']:.2f}, eastward steps {stats[-1]['east_steps']:.2f}", flush=True)
        out = {"days": days}
        for k, (s, n) in acc.items():
            out[k] = (s / np.maximum(n, 1)).astype(np.float32) if not k.endswith("_normal") else (s / max(n, 1)).astype(np.float32)
            if not k.endswith("_normal"):
                out[k][np.broadcast_to(n, out[k].shape) == 0] = np.nan
        np.savez_compressed(COMP / f"{model}.npz", **out)
        (COMP / f"{model}.json").write_text(json.dumps({"members": mems, "stats": stats}))
        print(f"{model}: {len(mems)} members, fields {sorted(k for k in out if k != 'days')}", flush=True)


def obs():
    rng = np.random.default_rng(20260927)
    od, oph, oamp, _, _, _ = obs_rmm()
    bom = pd.DataFrame({"phase": oph, "amp": oamp}, index=od)
    out = {}; meta = {}
    for fld, (f, key) in {"tas_na": ("era5_t2m.npz", "tas_na"), "tas_sa": ("era5_t2m.npz", "tas_sa"),
                          "z500_nh": ("era5_z500.npz", "z500_nh"), "pr_na": ("gpcp.npz", "pr_na"),
                          "pr_sa": ("gpcp.npz", "pr_sa")}.items():
        d = np.load(OBS / f)
        dates = d["dates"]
        end = "2024-02-24"
        k = dates <= end
        dates = dates[k]; Y = d[key][k].astype(np.float32)
        b = bom.reindex(dates)
        ph = b.phase.fillna(0).astype(int).values; amp = b.amp.values
        active = np.isfinite(amp) & (amp >= 1)
        _, months, frac = year_frac(dates, "standard"); t = np.arange(len(dates), dtype=float)
        sh = Y.shape; A, C = anomalies(Y, frac, t, trend=True); A = A.reshape(sh[0], -1)
        ev = events(ph, active)
        M, D, NE, SE = composite(A, ph, active, months, lags=SITE_LAGS, ev=ev, boot=2000, rng=rng)
        out[fld] = M; out[f"{fld}_se"] = SE; out[f"{fld}_days"] = D; out[f"{fld}_events"] = NE
        if fld.startswith("pr"):
            out[f"{fld}_normal"] = month_normal(C.reshape(sh[0], -1), months)
        meta[fld] = {"period": f"{dates[0]}..{dates[-1]}", "days": int(len(dates))}
        print(f"obs {fld}: {meta[fld]}, median events per phase-month {int(np.median(NE))}", flush=True)
    np.savez_compressed(COMP / "OBS.npz", **out)
    (COMP / "OBS.json").write_text(json.dumps(meta))


def reference(models):
    """Multi-model means + robust masks, observed composites + FDR masks, for SITE_LAGS -> the committed reference."""
    from scipy import stats as st
    li = [LAGS.index(L) for L in SITE_LAGS]
    O = np.load(COMP / "OBS.npz")
    R = {}; summ = {"models": models, "lags": list(SITE_LAGS), "fields": list(FIELDS), "robust_frac": {}, "obs_sig_frac": {}}
    for fld in FIELDS:
        Ms, Ds, used = [], [], []
        for m in models:
            f = COMP / f"{m}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            if fld not in z.files:
                continue
            Ms.append(z[fld][li]); Ds.append(z["days"]); used.append(m)
        variants = [(fld, None)] + ([(f"{fld.replace('pr_', 'prpct_')}", "pct")] if fld.startswith("pr") else [])
        for name, kind in variants:
            if Ms:
                S = np.stack(Ms)                                          # (model, lag, phase, month, cells)
                Dd = np.stack(Ds)                                         # (model, phase, month)
                if kind == "pct":
                    Nm = np.stack([np.load(COMP / f"{m}.npz")[f"{fld}_normal"] for m in used])   # (model, 12, cells)
                    S = 100.0 * S / np.where(Nm[:, None, None] >= 0.5, Nm[:, None, None], np.nan)
                S = np.where((Dd >= MIN_DAYS)[:, None, :, :, None], S, np.nan)
                n = np.sum(np.isfinite(S), axis=0)
                mm = np.nanmean(S, axis=0)
                tt, p = st.ttest_1samp(S, 0.0, axis=0, nan_policy="omit")
                p = np.asarray(p, float); p[n < 5] = np.nan
                agree = np.maximum(np.nansum(S > 0, 0), np.nansum(S < 0, 0)) / np.maximum(n, 1)
                sig = np.zeros(p.shape, bool)
                for a in range(p.shape[0]):
                    for b in range(p.shape[1]):
                        for c in range(p.shape[2]):
                            sig[a, b, c] = bh(p[a, b, c]) & (agree[a, b, c] >= AGREE)
                R[f"cmip6_{name}"] = np.where(sig, mm, np.nan).astype(np.float16)
                R[f"cmip6_{name}_raw"] = mm.astype(np.float16)
                R[f"cmip6_{name}_nmod"] = n.max(axis=-1).astype(np.int16)
                summ["robust_frac"][name] = round(float(np.mean(sig)), 3)
                summ.setdefault("models_used", {})[name] = used
            # observed
            M = O[fld]; SE = O[f"{fld}_se"]
            if kind == "pct":
                Nm = O[f"{fld}_normal"]
                M = 100.0 * M / np.where(Nm[None, None] >= 0.5, Nm[None, None], np.nan)
                SE = 100.0 * SE / np.where(Nm[None, None] >= 0.5, Nm[None, None], np.nan)
            zz = M / SE
            pp = 2 * st.norm.sf(np.abs(zz))
            NE = O[f"{fld}_events"]
            pp = np.where((NE >= MIN_EVENTS)[None, :, :, None], pp, np.nan)
            sig = np.zeros(pp.shape, bool)
            for a in range(pp.shape[0]):
                for b in range(pp.shape[1]):
                    for c in range(pp.shape[2]):
                        sig[a, b, c] = bh(pp[a, b, c])
            R[f"obs_{name}"] = np.where(sig, M, np.nan).astype(np.float16)
            R[f"obs_{name}_raw"] = M.astype(np.float16)
            R[f"obs_{name}_events"] = NE.astype(np.int16)
            summ["obs_sig_frac"][name] = round(float(np.mean(sig)), 3)
    from cmip6_mjo_extract import GRIDS
    # teleconnection realism and CMIP6-vs-observed agreement (raw composites, lag +10 d)
    def wpat(a, b, lat):
        w = np.repeat(np.cos(np.deg2rad(lat)), a.size // len(lat))
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            return np.nan
        a, b, w = a[ok], b[ok], w[ok]
        a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
        return float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b)))
    L10 = SITE_LAGS.index(10)
    latn = GRIDS["nh"][0]; nh20 = np.repeat(latn >= 20, len(GRIDS["nh"][1]))
    oz = O["z500_nh"][L10].astype(np.float64)                             # (phase, month, cells)
    tele = {}
    for m in models:
        f = COMP / f"{m}.npz"
        if not f.exists() or "z500_nh" not in np.load(f).files:
            continue
        mz = np.load(f)["z500_nh"][LAGS.index(10)].astype(np.float64)
        rs = [wpat(mz[q, 0][nh20], oz[q, 0][nh20], latn[latn >= 20]) for q in range(8)]      # January window = DJF
        amp = np.nanmean([np.nanstd(mz[q, 0][nh20]) / np.nanstd(oz[q, 0][nh20]) for q in range(8)])
        tele[m] = {"djf_lag10_pattern_r": round(float(np.nanmean(rs)), 3), "amp_ratio": round(float(amp), 2)}
    summ["teleconnection_djf_lag10"] = tele
    agree = {}
    for name in [k[len("cmip6_"):-len("_raw")] for k in R if k.startswith("cmip6_") and k.endswith("_raw")]:
        g = name.split("_")[-1]; lat = GRIDS[g][0]
        mm = R[f"cmip6_{name}_raw"][1].astype(np.float64); oo = R[f"obs_{name}_raw"][1].astype(np.float64)
        rr = [[wpat(mm[q, mo], oo[q, mo], lat) for mo in range(12)] for q in range(8)]
        ar = [[np.nanstd(mm[q, mo]) / np.nanstd(oo[q, mo]) for mo in range(12)] for q in range(8)]
        agree[name] = {"pattern_r_lag10_median": round(float(np.nanmedian(rr)), 3),
                       "pattern_r_lag10_by_month": [round(float(np.nanmean([rr[q][mo] for q in range(8)])), 3) for mo in range(12)],
                       "amp_ratio_lag10_median": round(float(np.nanmedian(ar)), 2)}
    summ["cmip6_vs_obs"] = agree
    for g in ("na", "sa", "nh"):
        R[f"lat_{g}"] = GRIDS[g][0]; R[f"lon_{g}"] = GRIDS[g][1]
    selj = json.loads((ARC / "selected.json").read_text()) if (ARC / "selected.json").exists() else {}
    summ["screen"] = selj.get("screen", []); summ["thresholds"] = selj.get("thresholds", {})
    summ["validation"] = json.loads((ARC / "validate.json").read_text()) if (ARC / "validate.json").exists() else {}
    mem = {m: json.loads((COMP / f"{m}.json").read_text())["members"] for m in models if (COMP / f"{m}.json").exists()}
    summ["members"] = mem
    summ["members_note"] = (f"{sum(len(v) for v in mem.values())} historical members in all (up to two per model), "
                            f"1979&ndash;2014; the 500&nbsp;hPa composites use each model&rsquo;s first member.")
    summ["window_note"] = "each month is the centred three-month window (the month and its neighbours)"
    summ["index_note"] = "RMM1/RMM2 band-passed 20-100 d and renormalised, observed (BoM, 1979-2024) and CMIP6 alike"
    summ["ensocheck"] = json.loads((ARC / "ensocheck.json").read_text()) if (ARC / "ensocheck.json").exists() else {}
    # the committed reference carries only what the renderer draws (masked fields + counts); the unmasked composites that
    # the comparisons above used stay in the local archive
    np.savez_compressed(COMP / "reference_raw.npz", **{k: v for k, v in R.items() if k.endswith("_raw")})
    np.savez_compressed(REF / "mjo_impacts_site.npz", **{k: v for k, v in R.items() if not k.endswith("_raw")})
    (REF / "mjo_impacts_site.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def ensocheck(models):
    """DJF active-day fraction (amplitude >= 1) by ENSO class, observed (before/after the fix) and CMIP6."""
    n = json.loads((HERE.parents[2] / "assets" / "sst" / "data" / "nino_history.json").read_text())
    oni = dict(zip(n["months"], n["series"]["oni"]["anom"]))
    d, ph, amp, ph0, amp0, r2s = obs_rmm()
    df = pd.DataFrame({"date": d, "amp": amp, "amp0": amp0})
    df["y"] = df.date.str.slice(0, 4).astype(int); df["m"] = df.date.str.slice(5, 7).astype(int)
    df = df[df.m.isin([12, 1, 2])]; df["season"] = np.where(df.m == 12, df.y + 1, df.y)
    rows = []
    for sy, g in df.groupby("season"):
        fin = g[np.isfinite(g.amp.values.astype(float))]
        if len(g) < 85 or len(fin) < 75:                               # filter end effects blank the record's last season
            continue
        o = oni.get(f"{sy}-01")
        rows.append((sy, o, float((fin.amp >= 1).mean()), float((g.amp0 >= 1).mean())))
    t = pd.DataFrame(rows, columns=["season", "oni", "active_fixed", "active_raw"])
    def cls(o):
        return "strong El Nino" if o >= 1.5 else "El Nino" if o >= 0.5 else "La Nina" if o <= -0.5 else "neutral"
    t["cls"] = t.oni.map(cls)
    out = {"method": "RMM1/RMM2 band-passed 20-100 d (whole record) and renormalised", "obs": {}, "obs_by_season": {}}
    for era, sel in (("1979-2013", t.season <= 2013), ("2014-2024", t.season >= 2014), ("all", t.season > 0)):
        g = t[sel]
        out["obs"][era] = {c: {"n": int((g.cls == c).sum()), "active_raw": round(float(g[g.cls == c].active_raw.mean()), 2),
                               "active_fixed": round(float(g[g.cls == c].active_fixed.mean()), 2)}
                           for c in ("strong El Nino", "El Nino", "neutral", "La Nina") if (g.cls == c).any()}
    for sy in (2016, 2024):
        r = t[t.season == sy]
        if len(r):
            out["obs_by_season"][f"{sy - 1}/{str(sy)[2:]}"] = {"raw": round(float(r.active_raw.iloc[0]), 2),
                                                             "fixed": round(float(r.active_fixed.iloc[0]), 2)}
    from scipy import stats as st
    g = t[t.season <= 2024]
    a, b = g[g.cls == "strong El Nino"].active_fixed, g[g.cls == "neutral"].active_fixed
    out["obs_mwu_strong_vs_neutral_fixed_p"] = round(float(st.mannwhitneyu(a, b).pvalue), 3) if len(a) and len(b) else None
    # CMIP6: per run, DJF seasons classed by the run's own DJF Nino-3.4 anomaly (in its own sd units x obs sd)
    import glob
    eofs = load_eofs(); mod = {}
    for m in models:
        vals = {c: [] for c in ("strong El Nino", "El Nino", "neutral", "La Nina")}
        for f in glob.glob(str(RED / f"{m}_*_ua.npz")):
            mem = Path(f).name[len(m) + 1:-7]
            if not (RED / f"{m}_{mem}_n34.npz").exists():
                continue
            dates, cal, months, frac, tt, ph_, amp_, *_ = _member_rmm(m, mem, eofs)
            z = np.load(RED / f"{m}_{mem}_n34.npz"); mm = pd.Series(z["months"]); v = z["n34"].astype(float)
            mon = mm.str.slice(5, 7).astype(int).values
            for k in range(1, 13):
                v[mon == k] -= v[mon == k].mean()
            ser = dict(zip(mm, v))
            yy = pd.Series(dates).str.slice(0, 4).astype(int).values
            season = np.where(months == 12, yy + 1, yy)
            for sy in np.unique(season):
                sel_ = (season == sy) & np.isin(months, [12, 1, 2]) & np.isfinite(amp_)
                if sel_.sum() < 75:
                    continue
                o = np.mean([ser.get(f"{sy - 1}-12", np.nan), ser.get(f"{sy}-01", np.nan), ser.get(f"{sy}-02", np.nan)])
                if np.isfinite(o):
                    vals[cls(o)].append(float((amp_[sel_] >= 1).mean()))
        mod[m] = {c: {"n": len(v_), "active": round(float(np.mean(v_)), 2)} for c, v_ in vals.items() if v_}
    out["cmip6"] = mod
    (ARC / "ensocheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "validate":
        validate()
    elif cmd == "screen":
        screen()
    elif cmd == "select":
        select()
    elif cmd == "ensocheck":
        ensocheck(sys.argv[2:])
    elif cmd == "compose":
        compose(sys.argv[2:])
    elif cmd == "obs":
        obs()
    elif cmd == "reference":
        reference(sys.argv[2:])
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
