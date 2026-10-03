#!/usr/bin/env python3
"""Wheeler-Kiladis wavenumber-frequency spectrum of tropical OLR (Wheeler & Kiladis 1999), climatology and "now".

Data: outgoing longwave radiation (OLR, positive up, W m-2) between 15S and 15N on the 2.5-degree grid, daily means.
  climatology   NOAA Interpolated OLR (Liebmann & Smith 1996; the data of WK99 itself), 1991-2020, PSL OPeNDAP, read
                ONE year at a time (parallel PSL reads have returned silent zeros here). Built once on the laptop:
                `build` -> reference/wk_clim.nc; the raw subset is cached outside the repo (WK_CACHE). WeatherBench2's
                ERA5 `mean_top_net_long_wave_radiation_flux` was the first plan: it is all NaN in every WB2 store.
  observed      ERA5 from the ARCO store (hourly 0.25 deg; the 00/06/12/18Z hourly means averaged per day, box-averaged
                to 2.5 deg), ~6 days behind; cached in reference/wk_olr_recent.nc and extended each run.
  source ratio  "now" is ERA5 + IFS, the background is NOAA OLR: the two do not carry the same variance at every
                (frequency, wavenumber) (NOAA's is interpolated from twice-daily AVHRR passes). `build` measures
                P_ERA5 / P_NOAA on the same 2021-22 days, smoothed like the background, and the "now" ratio is taken
                against background x that factor, so a source difference cannot read as an active wave.
  GMGSI         the site's satellite OLR proxy (scripts/sst/build_olr_realtime.py: NOAA GMGSI window-channel Tb ->
                Ohring-Gruber OLR, 15S-15N band mean, 2.5 deg, daily to today; scripts/sst/metar/olr_rt.nc). A band
                mean keeps only the SYMMETRIC waves (the antisymmetric part cancels), so it gets its own one-panel
                figure (wk_gmgsi.webp) against a band-mean NOAA background. It has no overlap with NOAA OLR (it starts
                2025-04), so its source ratio is chained: GMGSI/ERA5 on their overlap (ERA5 band means,
                reference/era5_olr_band.nc) x ERA5/NOAA on 2021-22, both smoothed like the background.
  gap + forecast  IFS-ENS open data (Google Cloud mirror only): `ttr`, top net thermal radiation accumulated from the
                start (J m-2, negative), so daily OLR = -(ttr(t+24h) - ttr(t)) / 86400. The days between the last ERA5 day
                and the cycle are filled from earlier 00Z cycles' first day (ensemble mean); the forecast is days 1-15 of
                the current cycle, MEMBER BY MEMBER (N_MEMBERS perturbed members): each member's 96-day record gets its
                own spectrum and the POWERS are averaged, because the spectrum of the ensemble mean loses the variance
                the members no longer agree on and would read as dying waves.

Method (WK99, as in NCL's wkSpaceTime): anomalies against the 1991-2020 day-of-year mean (3 harmonics); 96-day segments
(climatology: overlapping by 60 days); each segment detrended per grid point and tapered with a 10 % split cosine bell
in time; split into the parts symmetric and antisymmetric about the equator; 2-D FFT in longitude and time per latitude;
power summed over 0-15 deg. The background is the mean of the symmetric and antisymmetric climatological power,
smoothed by 1-2-1 passes in wavenumber (5/10/20/40 passes as frequency rises) and 10 in frequency. Normalised power =
power / background. Dispersion curves: equatorial shallow-water waves at equivalent depths 12, 25 and 50 m.

"Now" (one 96-day window) has few degrees of freedom: the power is smoothed with one 1-2-1 pass in frequency and one in
wavenumber (about 2 x 5.3 = 10.6 equivalent degrees of freedom; DOF_NOW), and a bin is shaded only where its ratio to the
background passes a chi-square test at Benjamini-Hochberg FDR 10 % across the plotted domain (the site's significance
rule); unshaded bins are not significantly above the climatological background.

    python wk_spectrum.py build                                     # laptop, once, ~20-30 min
    python wk_spectrum.py now --date 20261003 --out ../../../assets/sst/wk_now.webp   # daily (olr-waves.yml), from scripts/mjo/src
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "data" / "reference"
CLIM = REF / "wk_clim.nc"
RECENT = REF / "wk_olr_recent.nc"
NOAA_OLR = "https://psl.noaa.gov/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc"
WK_CACHE = Path(__import__("os").environ.get("WK_CACHE", str(Path.home() / "data_archive" / "wk")))
CAL = ("2021-01-01", "2022-12-31")                         # ERA5 vs NOAA overlap for the source ratio
GMGSI = HERE.parents[1] / "sst" / "metar" / "olr_rt.nc"         # band-mean satellite OLR proxy, daily to today
# GMGSI's daily value is the mean of TWO snapshots (00/12Z), and a day missing one keeps the other: over land that
# swaps the diurnal cycle in and out day to day. Its power over ERA5's rises with frequency (median 0.9 below 0.1 cpd,
# 1.3 at 0.2-0.25, 2.5 at 0.4-0.5, 4.6 near k = 0) and varies window to window, so no one calibration removes it.
# GMGSI is tested only at frequencies where 90 % of wavenumbers (|k| <= 15) keep GMGSI/ERA5 within 1.2 -- read off the
# calibration, not the result (on the 2026-10-03 build: periods of 8 days and longer).
GMGSI_TOL = 1.2


def gmgsi_fmax(clim) -> float:
    f, k = clim.freq.values, clim.k.values
    r = (clim["src_ratio_band_gmgsi"] / clim["src_ratio_band_era5"]).values[:, np.abs(k) <= KMAX]
    ok = [i for i in range(1, len(f)) if np.percentile(r[i], 90) <= GMGSI_TOL]
    fmax = 0.0
    for i in range(1, len(f)):                                                 # contiguous from the lowest frequency
        if i not in ok:
            break
        fmax = f[i]
    return float(fmax)
ERA5_BAND = REF / "era5_olr_band.nc"                           # ERA5 band means (cos-weighted), 2023-08 on
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "mean_top_net_long_wave_radiation_flux"
LATS = np.arange(-15.0, 15.01, 2.5)                       # 13 latitudes, symmetric about the equator (NOAA OLR grid)
LONS = np.arange(0.0, 360.0, 2.5)                         # 144
HALF = 1.25                                               # box half-width for regridding 0.25-deg fields
NSEG, STEP = 96, 36                                       # WK99: 96-day segments overlapping by 60 days
NH = 3
N_MEMBERS = int(__import__("os").environ.get("WK_MEMBERS", "20"))
DOF_NOW = 2 * 2 * (16 / 6)                                # two independent 1-2-1 passes, chi-square dof ~10.7
KMAX, FMAX = 15, 0.5                                      # daily data: 0.5 cycles per day is the Nyquist frequency


def artefact(freq, kk):
    """NOAA interpolated OLR carries a spurious EASTWARD peak at k = 14 near 9 days (ratio 2-10, harmonic at k = 28):
    the polar orbiters' ~14 orbits a day aliased into the twice-daily sampling. Masked in the plots and left out of
    the significance test, never read as a wave."""
    return (kk[None, :] >= 13) & (kk[None, :] <= 15) & (freq[:, None] >= 0.09) & (freq[:, None] <= 0.135)


# ── shared maths ──────────────────────────────────────────────────────────────────────────────────────────────────
def design(doy):
    w = 2 * np.pi * np.asarray(doy, float) / 365.25
    cols = [np.ones_like(w)]
    for k in range(1, NH + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.column_stack(cols)


def taper(n, frac=0.1):
    """Split cosine bell: 10 % at each end."""
    w = np.ones(n); m = int(round(frac * n))
    x = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    w[:m] = x; w[-m:] = x[::-1]
    return w


def segment_power(seg):
    """seg (time, lat, lon) anomaly, lat symmetric about 0 -> (P_sym, P_asym) over (freq, k) with freq >= 0 and k
    signed so that k > 0 = eastward. Power summed over 0..15 deg."""
    nt, ny, nx = seg.shape
    t = np.arange(nt)
    a = seg - seg.mean(0)
    # detrend per grid point
    tc = t - t.mean()
    a = a - np.einsum("t,yx->tyx", tc, (tc[:, None, None] * a).sum(0) / (tc ** 2).sum())
    a = a * taper(nt)[:, None, None]
    half = ny // 2                                         # equator index
    up, dn = a[:, half:, :], a[:, half::-1, :]             # lat >= 0 and its mirror
    sym, asym = 0.5 * (up + dn), 0.5 * (up - dn)
    out = []
    for f in (sym, asym):
        F = np.fft.fft2(f, axes=(0, 2)) / (nt * nx)        # (freq, lat, k)
        P = (np.abs(F) ** 2).sum(1)                        # sum over latitudes
        P = np.fft.fftshift(P, axes=(0, 1))
        out.append(P)
    freq = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0))      # cycles per day
    kk = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))   # zonal wavenumber
    # numpy's exp(-i(kx + wt)) convention: positive frequency with POSITIVE k is a WESTWARD wave; flip k
    res = []
    for P in out:
        pos = freq >= 0
        Q = P[pos][:, ::-1]
        res.append(Q)
    return freq[freq >= 0], -kk[::-1], res[0], res[1]


def smooth121(a, axis, n):
    a = a.copy()
    for _ in range(n):
        p = np.pad(a, [(1, 1) if i == axis else (0, 0) for i in range(a.ndim)], mode="edge")
        sl = lambda s: tuple(s if i == axis else slice(None) for i in range(a.ndim))
        a = 0.25 * p[sl(slice(0, -2))] + 0.5 * p[sl(slice(1, -1))] + 0.25 * p[sl(slice(2, None))]
    return a


def background(psym, pasym, freq):
    bg = 0.5 * (psym + pasym)
    for lo, hi, n in ((0.0, 0.1, 5), (0.1, 0.2, 10), (0.2, 0.3, 20), (0.3, 9.9, 40)):
        m = (freq >= lo) & (freq < hi)
        if m.any():
            bg[m] = smooth121(bg, 1, n)[m]
    return smooth121(bg, 0, 10)


def anomaly_band(olr, times, coef):
    """(time, lon) band mean minus the band-mean harmonic climatology coef (h, lon)."""
    X = design(pd.DatetimeIndex(times).dayofyear)
    return olr - X @ coef


def anomaly(olr, times, coef):
    X = design(pd.DatetimeIndex(times).dayofyear)
    return olr - np.tensordot(X, coef, axes=(1, 0))


# ── climatology (laptop, once) ────────────────────────────────────────────────────────────────────────────────────
def noaa_years(y0: int, y1: int) -> xr.DataArray:
    """NOAA interpolated OLR on LATS x LONS, daily, years y0..y1 -- one year per request, sequentially, each checked
    (finite, 60-380 W m-2) and cached as WK_CACHE/noaa_olr_{year}.nc."""
    WK_CACHE.mkdir(parents=True, exist_ok=True)
    out, ds = [], None
    for y in range(y0, y1 + 1):
        f = WK_CACHE / f"noaa_olr_{y}.nc"
        if not f.exists():
            if ds is None:
                ds = xr.open_dataset(NOAA_OLR)
            for attempt in range(3):
                x = ds["olr"].sel(time=str(y), lat=slice(15.01, -15.01)).load().sortby("lat")
                v = x.values
                if np.isfinite(v).all() and v.min() > 60 and v.max() < 380 and len(x.time) >= 365:
                    break
                print(f"  {y}: bad read (finite {np.isfinite(v).mean():.3f}, range {np.nanmin(v):.0f}-{np.nanmax(v):.0f}), retry",
                      flush=True)
                time.sleep(5)
            else:
                raise SystemExit(f"NOAA OLR {y}: three bad reads")
            x.astype("float32").to_netcdf(f)
        x = xr.open_dataarray(f).load()
        assert np.allclose(x.lat.values, LATS) and np.allclose(x.lon.values, LONS), f"grid {y}"
        out.append(x.rename(lat="latitude", lon="longitude").transpose("time", "latitude", "longitude"))
        print(f"  {y}: NOAA OLR {len(x.time)} days, mean {float(x.mean()):.1f} W/m2", flush=True)
    return xr.concat(out, "time")


def band_power(a):
    """a (time, lon) band-mean anomaly -> (freq >= 0, k eastward > 0, P), the same conventions as segment_power."""
    nt, nx = a.shape
    t = np.arange(nt); a = a - a.mean(0); tc = t - t.mean()
    a = a - np.outer(tc, (tc[:, None] * a).sum(0) / (tc ** 2).sum())
    a = a * taper(nt)[:, None]
    P = np.fft.fftshift(np.abs(np.fft.fft2(a) / (nt * nx)) ** 2)
    freq = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0)); kk = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    pos = freq >= 0
    return freq[pos], -kk[::-1], P[pos][:, ::-1]


def band_mean_power(an):
    P, n = 0, 0
    for s in range(0, an.shape[0] - NSEG + 1, STEP):
        freq, kk, p = band_power(an[s:s + NSEG])
        P = P + p; n += 1
    return freq, kk, P / n, n


def band_coef(times, series):
    X = design(pd.DatetimeIndex(times).dayofyear)
    return np.linalg.lstsq(X, series, rcond=None)[0]


def seg_mean_power(an):
    ps, pa, n = 0, 0, 0
    for s in range(0, an.shape[0] - NSEG + 1, STEP):
        freq, kk, a_s, a_a = segment_power(an[s:s + NSEG])
        ps = ps + a_s; pa = pa + a_a; n += 1
    return freq, kk, ps / n, pa / n, n


def build(a) -> int:
    t0 = time.time()
    olr = noaa_years(1991, 2020)
    X = design(pd.DatetimeIndex(olr.time.values).dayofyear)
    Y = olr.values.reshape(olr.sizes["time"], -1)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0].reshape(X.shape[1], len(LATS), len(LONS))
    an = olr.values - np.tensordot(X, coef, axes=(1, 0))
    freq, kk, ps, pa, n = seg_mean_power(an)
    bg = background(ps, pa, freq)
    # source ratio: ERA5 (ARCO, as "now" reads it) against NOAA on the SAME days, anomalies against the same harmonics
    days = pd.date_range(*CAL, freq="D")
    noaa = noaa_years(int(CAL[0][:4]), int(CAL[1][:4])).sel(time=slice(*CAL))
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    with ThreadPoolExecutor(8) as ex:
        era = list(ex.map(lambda d: arco_day(ds, d), days))
    ok = [i for i, e in enumerate(era) if e is not None]
    assert len(ok) == len(days), f"ARCO missing {len(days) - len(ok)} days in {CAL}"
    era = np.stack(era)
    nt = pd.DatetimeIndex(noaa.time.values)
    assert (nt == days).all(), "NOAA calibration days"
    _, _, es, ea, nc = seg_mean_power(anomaly(era, days, coef))
    _, _, ns_, na_, _ = seg_mean_power(anomaly(noaa.values, days, coef))
    sm = lambda P: background(P, P, freq)                                       # noqa: E731  same smoothing as the background
    r_sym, r_asym = sm(es) / sm(ns_), sm(ea) / sm(na_)
    dom = (np.abs(kk)[None, :] <= KMAX) & (freq[:, None] > 0) & (freq[:, None] <= FMAX)
    print(f"  ERA5/NOAA power, {CAL[0]}..{CAL[1]} ({nc} segments): sym median {np.median(r_sym[dom]):.2f} "
          f"(5-95% {np.percentile(r_sym[dom], 5):.2f}-{np.percentile(r_sym[dom], 95):.2f}), asym median "
          f"{np.median(r_asym[dom]):.2f}", flush=True)
    # band means (for GMGSI): NOAA background, ERA5/NOAA, then GMGSI/ERA5 on their own overlap
    bcoef = coef.mean(1)                                                        # (h, lon): harmonics of the band mean
    _, _, pb, nb = band_mean_power(an.mean(1))
    bg_band = background(pb, pb, freq)
    _, _, eb, _ = band_mean_power(anomaly_band(era.mean(1), days, bcoef))
    _, _, nbp, _ = band_mean_power(anomaly_band(noaa.values.mean(1), days, bcoef))
    r_band_era5 = sm(eb) / sm(nbp)
    g = xr.open_dataset(GMGSI).olr_15; e5 = xr.open_dataset(ERA5_BAND).olr
    both = sorted(set(pd.DatetimeIndex(g.time.values)) & set(pd.DatetimeIndex(e5.time.values)))
    both = pd.DatetimeIndex(both)
    assert len(both) >= 3 * NSEG, f"GMGSI/ERA5 overlap only {len(both)} days"
    run = pd.date_range(both[0], both[-1]); assert len(run) == len(both), "GMGSI/ERA5 overlap has gaps"
    _, _, gp, ng = band_mean_power(anomaly_band(g.sel(time=both).values, both, bcoef))
    _, _, ep, _ = band_mean_power(anomaly_band(e5.sel(time=both).values, both, bcoef))
    r_band_g_e = sm(gp) / sm(ep)
    r_band_gmgsi = r_band_g_e * r_band_era5
    print(f"  band means: ERA5/NOAA median {np.median(r_band_era5[dom]):.2f}; GMGSI/ERA5 ({both[0]:%Y-%m-%d}..{both[-1]:%Y-%m-%d}, "
          f"{ng} segments) median {np.median(r_band_g_e[dom]):.2f} (5-95% {np.percentile(r_band_g_e[dom], 5):.2f}-"
          f"{np.percentile(r_band_g_e[dom], 95):.2f}); GMGSI/NOAA median {np.median(r_band_gmgsi[dom]):.2f}", flush=True)
    out = xr.Dataset({"p_sym": (("freq", "k"), ps), "p_asym": (("freq", "k"), pa), "background": (("freq", "k"), bg),
                      "p_band": (("freq", "k"), pb), "background_band": (("freq", "k"), bg_band),
                      "src_ratio_band_era5": (("freq", "k"), r_band_era5), "src_ratio_band_gmgsi": (("freq", "k"), r_band_gmgsi),
                      "coef_band": (("h", "lon"), bcoef.astype("float32")),
                      "src_ratio_sym": (("freq", "k"), r_sym), "src_ratio_asym": (("freq", "k"), r_asym),
                      "coef": (("h", "lat", "lon"), coef.astype("float32"))},
                     coords={"freq": freq, "k": kk, "lat": LATS, "lon": LONS},
                     attrs={"source": "NOAA Interpolated OLR (Liebmann & Smith 1996) 1991-2020, PSL, 2.5 deg",
                            "src_ratio": f"ERA5 ARCO / NOAA interp OLR power on {CAL[0]}..{CAL[1]}, smoothed like the background",
                            "src_ratio_band_gmgsi": f"GMGSI/ERA5 band-mean power {both[0]:%Y-%m-%d}..{both[-1]:%Y-%m-%d} x ERA5/NOAA {CAL[0]}..{CAL[1]}",
                            "segments": n, "segment_days": NSEG, "step_days": STEP, "harmonics": NH})
    REF.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(CLIM)
    print(f"wrote {CLIM} ({n} segments, {time.time() - t0:.0f}s)")
    return 0


# ── observed tail (ARCO ERA5) ─────────────────────────────────────────────────────────────────────────────────────
def to15(f, lat, lon):
    """0.25-deg global field -> the 2.5-deg 15S-15N grid by box means (+-HALF deg around each target point)."""
    lat = np.asarray(lat); lon = np.asarray(lon) % 360
    out = np.empty((len(LATS), len(LONS)), np.float32)
    o = np.argsort(lon); lon_s = lon[o]; f = f[:, o]
    ext_lon = np.r_[lon_s - 360, lon_s, lon_s + 360]; ext = np.concatenate([f, f, f], 1)
    cs = np.cumsum(np.pad(ext, ((0, 0), (1, 0))), 1)
    for j, y in enumerate(LATS):
        rows = np.abs(lat - y) <= HALF + 0.01
        sub = cs[rows]
        i0 = np.searchsorted(ext_lon, LONS - HALF - 0.01); i1 = np.searchsorted(ext_lon, LONS + HALF + 0.01)
        out[j] = ((sub[:, i1] - sub[:, i0]) / (i1 - i0)).mean(0)
    return out


def arco_day(ds, day):
    vals = []
    for hh in (0, 6, 12, 18):
        t = pd.Timestamp(day) + pd.Timedelta(hours=hh)
        try:
            x = ds[VAR].sel(time=t).sel(latitude=slice(16, -16))
            a = np.asarray(x.values, dtype=np.float32)
        except Exception:                                                   # noqa: BLE001
            return None
        if not np.isfinite(a).all():
            return None
        vals.append(-a)
    m = np.mean(vals, 0)
    return to15(m, x.latitude.values, x.longitude.values)


def update_recent(until: pd.Timestamp, days: int = 140) -> xr.DataArray:
    have = xr.open_dataarray(RECENT).load() if RECENT.exists() else None
    want = pd.date_range(until - pd.Timedelta(days=days), until, freq="D")
    got = set(pd.DatetimeIndex(have.time.values)) if have is not None else set()
    todo = [d for d in want if d not in got]
    if todo:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        with ThreadPoolExecutor(8) as ex:
            res = dict(zip(todo, ex.map(lambda d: arco_day(ds, d), todo)))
        new = [(d, r) for d, r in res.items() if r is not None]
        if new:
            da = xr.DataArray(np.stack([r for _, r in new]), dims=("time", "lat", "lon"),
                              coords={"time": [d for d, _ in new], "lat": LATS, "lon": LONS}, name="olr")
            have = da if have is None else xr.concat([have, da], "time")
        print(f"  ERA5 (ARCO): +{len(new)} days, {len(todo) - len(new)} not published yet", flush=True)
    have = have.sortby("time")
    have = have.sel(time=have.time >= np.datetime64(until - pd.Timedelta(days=days + 30)))
    have.attrs.update(units="W m-2", source="ERA5 ARCO, daily mean of 00/06/12/18Z hourly means, 2.5 deg box means")
    tmp = RECENT.with_suffix(".tmp.nc"); have.to_netcdf(tmp); tmp.replace(RECENT)
    return have


# ── IFS-ENS (Google mirror only) ──────────────────────────────────────────────────────────────────────────────────
def _ifs_ttr(date, step, members):
    sys.path.insert(0, str(HERE.parents[1] / "ecmwf"))
    import rangefetch as rf
    import eccodes as ec
    idx = rf.fetch_index(date, "00", "ifs", step, "ef", stream="enfo")
    want = [e for e in idx if e.get("param") == "ttr" and e.get("type") == "pf" and int(e.get("number", 0)) in members]
    want = sorted(want, key=lambda e: e["_offset"])
    blob = rf.fetch_ranges(rf.path_for(date, "00", "ifs", step, "ef", stream="enfo") + ".grib2", rf.coalesce(want, max_gap=512 * 1024))
    out, pos = {}, 0
    for e in want:
        h = ec.codes_new_from_message(blob[pos:pos + e["_length"]]); pos += e["_length"]
        ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
        la0, la1 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"), ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lo0, di = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees"), ec.codes_get(h, "iDirectionIncrementInDegrees")
        num = ec.codes_get(h, "number")
        v = ec.codes_get_values(h).reshape(nj, ni); ec.codes_release(h)
        lat = np.linspace(la0, la1, nj); lon = lo0 + di * np.arange(ni)
        keep = np.abs(lat) <= 16
        out[num] = (v[keep], lat[keep], lon)
    return out


def ifs_daily(date, members, steps=range(24, 361, 24)) -> np.ndarray:
    """(member, day, lat, lon) daily-mean OLR for days 1..15 of the 00Z cycle on `date`."""
    prev = None
    days = []
    for st in [0] + list(steps):
        if st == 0:
            prev = {m: None for m in members}
            continue
        cur = _ifs_ttr(date, st, members)
        d = []
        for m in members:
            v, lat, lon = cur[m]
            p = prev.get(m)
            acc = v if p is None else v - p
            d.append(to15(-acc / 86400.0, lat, lon))
            prev[m] = v
        days.append(np.stack(d))
    return np.stack(days, 1)


# ── now ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def window_power(series, times, clim):
    an = anomaly(series, times, clim["coef"].values)
    freq, kk, ps, pa = segment_power(an)
    return freq, kk, ps, pa


def bh(p, q=0.10):
    pf = p.ravel(); o = np.argsort(pf); n = len(pf)
    thr = q * np.arange(1, n + 1) / n
    passed = pf[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool); out[o[:k]] = True
    return out.reshape(p.shape)


def now(a) -> int:
    from scipy.stats import chi2
    clim = xr.open_dataset(CLIM)
    init = pd.Timestamp(a.date)
    obs = update_recent(init - pd.Timedelta(days=1))
    last_obs = pd.Timestamp(obs.time.values[-1])
    members = list(range(1, N_MEMBERS + 1))
    # gap days (last ERA5 day + 1 .. init - 1) from earlier cycles' day 1, member mean
    gap_days = pd.date_range(last_obs + pd.Timedelta(days=1), init - pd.Timedelta(days=1), freq="D")
    gap = []
    for d in gap_days:
        try:
            g = ifs_daily(f"{d:%Y%m%d}", members[:10], steps=[24]).mean(0)[0]
            gap.append(g)
        except Exception as e:                                             # noqa: BLE001
            print(f"  gap {d:%Y-%m-%d}: IFS-ENS day 1 unavailable ({str(e)[:60]})", flush=True)
            gap.append(None)
    fc = ifs_daily(f"{init:%Y%m%d}", members)                              # (member, 15, lat, lon)
    print(f"  IFS-ENS {init:%Y-%m-%d} 00Z: {fc.shape[0]} members x {fc.shape[1]} days, OLR mean {np.nanmean(fc):.1f}", flush=True)
    otimes = pd.DatetimeIndex(obs.time.values)
    obs_arr = obs.values
    if any(g is None for g in gap):                                        # bridge any missing gap day linearly
        good = [i for i, g in enumerate(gap) if g is not None]
        for i, g in enumerate(gap):
            if g is None:
                lo = max([j for j in good if j < i], default=None); hi = min([j for j in good if j > i], default=None)
                gap[i] = gap[lo] if hi is None and lo is not None else gap[hi] if lo is None and hi is not None else \
                    (obs_arr[-1] if lo is None else 0.5 * (gap[lo] + gap[hi]))
    gtimes = pd.DatetimeIndex(gap_days)
    ftimes = pd.date_range(init, periods=fc.shape[1], freq="D")
    gap_arr = np.stack(gap) if gap else np.empty((0,) + obs_arr.shape[1:])
    # 1) observed-only window ending at the last ERA5 day
    sel = otimes > last_obs - pd.Timedelta(days=NSEG)
    freq, kk, ps_o, pa_o = window_power(obs_arr[sel][-NSEG:], otimes[sel][-NSEG:], clim)
    # 2) window ending at forecast day 15: per member, powers averaged
    n_fc = fc.shape[1]; n_gap = len(gap_days); n_obs = NSEG - n_fc - n_gap
    base = np.concatenate([obs_arr[-n_obs:], gap_arr], 0)
    btimes = otimes[-n_obs:].append(gtimes)
    ps_f = pa_f = 0
    for m in range(fc.shape[0]):
        rec = np.concatenate([base, fc[m]], 0)
        _, _, s, p = window_power(rec, btimes.append(ftimes), clim)
        ps_f = ps_f + s; pa_f = pa_f + p
    ps_f, pa_f = ps_f / fc.shape[0], pa_f / fc.shape[0]
    bg = clim["background"].values
    res = {}
    for tag, (s, p) in {"obs": (ps_o, pa_o), "fc": (ps_f, pa_f)}.items():
        for part, P in (("sym", s), ("asym", p)):
            Ps = smooth121(smooth121(P, 0, 1), 1, 1)
            ratio = Ps / (bg * clim[f"src_ratio_{part}"].values)                   # ERA5/IFS against a NOAA background
            pv = chi2.sf(ratio * DOF_NOW, DOF_NOW)
            dom = (np.abs(kk)[None, :] <= KMAX) & (freq[:, None] > 0) & (freq[:, None] <= FMAX) & ~artefact(freq, kk)
            sig = np.zeros_like(ratio, bool)
            sig[dom] = bh(pv[dom])
            res[(tag, part)] = (ratio, sig)
    render_now(res, freq, kk, clim, init, last_obs, n_gap, n_fc, fc.shape[0], Path(a.out))
    # GMGSI runs to yesterday whatever the IFS cycle (the daily job uses the previous day's 00Z, not yet on the mirror at 02:40Z)
    gsum = gmgsi_panel(clim, pd.Timestamp.now("UTC").tz_localize(None).normalize(), Path(a.out).with_name("wk_gmgsi.webp"))
    render_clim(clim, Path(a.out).with_name("wk_clim.webp"))
    summary = {"init": f"{init:%Y-%m-%d}", "era5_last": f"{last_obs:%Y-%m-%d}", "gap_days": int(n_gap),
               "forecast_days": int(n_fc), "members": int(fc.shape[0]),
               "obs_window": [f"{(last_obs - pd.Timedelta(days=NSEG - 1)):%Y-%m-%d}", f"{last_obs:%Y-%m-%d}"],
               "fc_window": [f"{btimes[0]:%Y-%m-%d}", f"{ftimes[-1]:%Y-%m-%d}"], "gmgsi": gsum}
    js = Path(a.out).parent / "data" / "wk_spectrum.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(summary))
    print("summary", summary)
    return 0


# ── GMGSI (band mean, symmetric waves only) ──────────────────────────────────────────────────────────────────────
def gmgsi_panel(clim, init, out: Path) -> dict | None:
    """The last 96 complete days of the site's GMGSI OLR proxy (through the day before `init`), band-mean spectrum
    against the NOAA band-mean background x the chained GMGSI/NOAA source ratio, same smoothing and test as "now"."""
    from scipy.stats import chi2
    if "background_band" not in clim or not GMGSI.exists():
        print("  GMGSI: no band background in wk_clim.nc (rerun build) or no store; skipped", flush=True)
        return None
    g = xr.open_dataset(GMGSI).olr_15.load()
    g = g.sel(time=g.time < np.datetime64(init))                               # today's entry may be a partial day
    t = pd.DatetimeIndex(g.time.values)[-NSEG:]
    if len(t) < NSEG or (t[-1] - t[0]).days != NSEG - 1 or not np.isfinite(g.sel(time=t).values).all():
        print(f"  GMGSI: the last {NSEG} days are not complete; skipped", flush=True)
        return None
    an = anomaly_band(g.sel(time=t).values, t, clim["coef_band"].values)
    freq, kk, P = band_power(an)
    Ps = smooth121(smooth121(P, 0, 1), 1, 1)
    ratio = Ps / (clim["background_band"].values * clim["src_ratio_band_gmgsi"].values)
    fcut = gmgsi_fmax(clim)
    dom = (np.abs(kk)[None, :] <= KMAX) & (freq[:, None] > 0) & (freq[:, None] <= fcut + 1e-9) & ~artefact(freq, kk)
    pv = chi2.sf(ratio * DOF_NOW, DOF_NOW)
    sig = np.zeros_like(ratio, bool); sig[dom] = bh(pv[dom])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    cf = _panel(ax, freq, kk, ratio, [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10], "YlOrRd", "sym",
                f"GMGSI satellite OLR, {t[0]:%d %b} – {t[-1]:%d %b %Y}\nband mean 15°S–15°N: symmetric waves only", sig=sig)
    _labels(ax, "sym")
    hi = fcut + 0.5 / NSEG                                                     # the upper edge of the last tested bin
    ytop = min(FMAX, 2 * hi)                                                   # zoom: the tested part fills the panel
    ax.set_ylim(0, ytop)
    for txt in ax.texts:                                                       # the full-range period labels above the zoom
        txt.set_clip_on(True)
    ax.axhspan(hi, ytop, facecolor="#e9e7e2", edgecolor="#c9c5bd", hatch="///", lw=0, zorder=0.5)
    for d in (8, 10, 15, 20):                                                  # period guides inside the zoom
        if 1 / d < ytop:
            ax.axhline(1 / d, color="#bbb", lw=0.6, ls=":"); ax.text(KMAX - 0.3, 1 / d, f"{d} d", fontsize=7, color="#6f6b64",
                                                                         ha="right", va="bottom")
    ax.text(-7, (hi + ytop) / 2, f"periods under {1 / fcut:.1f} days not tested:\nGMGSI's two snapshots a day put\n"
            "diurnal-cycle noise here", ha="center", va="center", fontsize=8.5, color="#6f6b64",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    if not sig.any():
        ax.text(0.5, 0.36, f"nothing significantly above the background\n(largest ratio {np.nanmax(ratio[dom]):.1f}; "
                f"about {chi2_need(dom.sum()):.1f} needed)", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#6f6b64", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12, aspect=40)
    cb.set_label("power / 1991–2020 background; shaded only where significant (chi-square, BH FDR 10 %)", fontsize=8.5)
    fig.suptitle("Which tropical waves are active: GMGSI OLR, last 96 days", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.03, "OLR estimated from NOAA GMGSI geostationary infrared (window-channel brightness temperature, "
             "Ohring-Gruber), to yesterday.\nBackground: NOAA OLR 1991–2020 band means x the GMGSI/NOAA power ratio, chained "
             "through ERA5 (GMGSI starts in 2025).\nA standing anomaly (e.g. an El Niño's) sits at zero frequency and is "
             "detrended out of a 96-day window: it does not show here.", fontsize=7.6, color="#6f6b64")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out, flush=True)
    return {"window": [f"{t[0]:%Y-%m-%d}", f"{t[-1]:%Y-%m-%d}"], "tested_min_period_days": round(1 / fcut, 1),
            "max_ratio": round(float(np.nanmax(ratio[dom])), 2),
            "significant_bins": int(sig.sum())}


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────
def dispersion(ax, part, kmin=-KMAX, kmax=KMAX):
    """Equatorial shallow-water dispersion curves (Matsuno 1966) at h = 12, 25, 50 m, in cycles per day."""
    g, beta, a_e = 9.81, 2.28e-11, 6.371e6
    kk = np.linspace(kmin, kmax, 400)
    kw = 2 * np.pi * kk / (2 * np.pi * a_e)                                # rad m-1
    for h in (12, 25, 50):
        c = np.sqrt(g * h)
        cpd = lambda w: w * 86400 / (2 * np.pi)
        lines = []
        if part == "sym":
            lines.append((kk[kk > 0], cpd(c * kw[kk > 0]), "Kelvin"))
            for n in (1,):
                w = -beta * kw / (kw ** 2 + (2 * n + 1) * beta / c)
                lines.append((kk[kk < 0], cpd(w[kk < 0]), "ER"))
                # n = 1 inertia-gravity: w^2 = (2n+1) beta c + c^2 k^2 (approx.)
                lines.append((kk, cpd(np.sqrt((2 * n + 1) * beta * c + (c * kw) ** 2)), "IG1"))
        else:
            # mixed Rossby-gravity (n = 0), westward branch and eastward inertia-gravity n = 0
            w = kw * c / 2 * (1 + np.sqrt(1 + 4 * beta / (kw ** 2 * c)))
            w_m = kw * c / 2 * (1 - np.sqrt(1 + 4 * beta / (kw ** 2 * c)))
            lines.append((kk[kk < 0], cpd(np.abs(w_m[kk < 0])), "MRG"))
            lines.append((kk[kk > 0], cpd(w[kk > 0]), "EIG"))
            lines.append((kk, cpd(np.sqrt(5 * beta * c + (c * kw) ** 2)), "IG2"))
        for x, y, lab in lines:
            ax.plot(x, y, color="#333", lw=0.7, ls="--" if h == 25 else "-", alpha=0.75)
    ax.axvline(0, color="#777", lw=0.6, ls=":")
    for per in (3, 6, 30):
        ax.axhline(1 / per, color="#999", lw=0.5, ls=":")
        ax.text(kmax - 0.3, 1 / per, f"{per} d", fontsize=7, color="#6f6b64", ha="right", va="bottom")


def _panel(ax, freq, kk, field, levels, cmap, part, title, sig=None):
    import matplotlib.pyplot as plt  # noqa: F401
    m = (np.abs(kk) <= KMAX); f = (freq > 0) & (freq <= FMAX)
    K, Fq = np.meshgrid(kk[m], freq[f])
    Z = field[np.ix_(f, m)]
    if sig is not None:
        Z = np.where(sig[np.ix_(f, m)], Z, np.nan)
    cf = ax.contourf(K, Fq, Z, levels=levels, cmap=cmap, extend="max")
    dispersion(ax, part)
    ax.set_xlim(-KMAX, KMAX); ax.set_ylim(0, FMAX)
    ax.set_xlabel("zonal wavenumber (eastward > 0)", fontsize=9)
    ax.set_ylabel("frequency (cycles per day)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    return cf


def _labels(ax, part):
    if part == "sym":
        for x, y, t in ((8, 0.33, "Kelvin"), (-9, 0.06, "n=1 ER"), (2, 0.025, "MJO"), (-11, 0.46, "n=1 WIG"), (12, 0.47, "n=1 EIG")):
            ax.text(x, y, t, fontsize=8, color="#111", weight="bold", ha="center", clip_on=True)
    else:
        for x, y, t in ((-8, 0.28, "MRG"), (8, 0.42, "n=0 EIG")):
            ax.text(x, y, t, fontsize=8, color="#111", weight="bold", ha="center", clip_on=True)


def render_clim(clim, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freq, kk = clim.freq.values, clim.k.values
    bg = clim["background"].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    lev = [1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8]
    for ax, part, P in zip(axes, ("sym", "asym"), (clim["p_sym"].values, clim["p_asym"].values)):
        R = P / bg; R[artefact(freq, kk)] = np.nan
        cf = _panel(ax, freq, kk, R, lev, "YlOrRd", part,
                    f"{'Symmetric' if part == 'sym' else 'Antisymmetric'}: power / background")
        _labels(ax, part)
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.13, aspect=50)
    cb.set_label("power divided by the smoothed background (ratios above 1.1 shown; every shaded bin is significant over "
                 f"{int(clim.attrs['segments'])} segments)", fontsize=8.5)
    fig.suptitle("Wheeler–Kiladis spectrum of tropical OLR, 15°S–15°N — NOAA OLR 1991–2020 climatology", fontsize=12,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02, "96-day segments overlapping by 60 days. Curves: equatorial shallow-water waves at equivalent "
             "depths 12, 25 (dashed) and 50 m. Blank at k = 13–15 near 9 days: a polar-orbiter sampling artefact, not a wave. "
             "NOAA Interpolated OLR (Liebmann & Smith 1996), NOAA PSL.", fontsize=8, color="#6f6b64")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


def chi2_need(n, q=0.10):
    """Ratio a single bin needs to pass BH at FDR q when it is the strongest of n (p <= q / n)."""
    from scipy.stats import chi2
    return chi2.isf(q / n, DOF_NOW) / DOF_NOW


def render_now(res, freq, kk, clim, init, last_obs, n_gap, n_fc, nmem, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.6), sharex=True, sharey=True)
    lev = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
    w0 = last_obs - pd.Timedelta(days=NSEG - 1)
    titles = {"obs": f"Observed (ERA5), {w0:%d %b} – {last_obs:%d %b %Y}",
              "fc": f"Ending at forecast day {n_fc}: {NSEG - n_fc - n_gap} d ERA5 + {n_gap} d IFS-ENS day 1\n+ {n_fc} d "
                    f"IFS-ENS from {init:%d %b} 00Z ({nmem} members)"}
    for i, tag in enumerate(("obs", "fc")):
        for j, part in enumerate(("sym", "asym")):
            ratio, sig = res[(tag, part)]
            ax = axes[i, j]
            cf = _panel(ax, freq, kk, ratio, lev, "YlOrRd", part,
                        f"{titles[tag]}\n{'Symmetric' if part == 'sym' else 'Antisymmetric'}", sig=sig)
            _labels(ax, part)
            if not sig.any():
                dom = (np.abs(kk)[None, :] <= KMAX) & (freq[:, None] > 0) & (freq[:, None] <= FMAX) & ~artefact(freq, kk)
                ax.text(0.5, 0.8, f"nothing significantly above the background\n(largest ratio {np.nanmax(ratio[dom]):.1f}; "
                        f"about {chi2_need(dom.sum()):.1f} needed)", transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#6f6b64", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
            if i == 0:
                ax.set_xlabel("")
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.035, pad=0.07, aspect=50)
    cb.set_label("power in this 96-day window divided by the 1991–2020 background; shaded only where significant "
                 "(chi-square, BH FDR 10 %)", fontsize=8.5)
    fig.suptitle(f"Which tropical waves are active: Wheeler–Kiladis spectrum of OLR, 15°S–15°N, last 96 days — "
                 f"IFS-ENS {init:%Y-%m-%d} 00Z", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Forecast power is computed member by member and averaged (the ensemble mean's spectrum would "
             "lose the waves the members disagree on). Background: NOAA OLR 1991–2020 x the ERA5/NOAA power ratio measured on 2021–22.\n"
             "A standing anomaly (e.g. an El Niño's) sits at zero frequency and is detrended out of a 96-day window: it does not show here.\n"
             "ERA5 (Copernicus C3S / ECMWF); ECMWF IFS-ENS open data (CC BY 4.0); NOAA Interpolated OLR (NOAA PSL).",
             fontsize=8, color="#6f6b64")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    n = sub.add_parser("now"); n.add_argument("--date", required=True); n.add_argument("--out", default="../../assets/sst/wk_now.webp")
    a = ap.parse_args()
    return build(a) if a.cmd == "build" else now(a)


if __name__ == "__main__":
    raise SystemExit(main())
