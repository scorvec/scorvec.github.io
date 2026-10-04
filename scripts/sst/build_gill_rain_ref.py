#!/usr/bin/env python3
"""Reference for the Gill rain-forcing product (gill_rain.py) — built once on the laptop.

Stages (each caches its output under scripts/sst/data/gill_rain_ref/, gitignored):

  imerg-clim     IMERG V07 Late daily, 1° over 30°S–30°N (imerg_tropics.py fetch, 2001–2025): day-of-year
                 harmonic climatology (mean + 4 harmonics per cell) — the base the live anomalies use.
  era5-weekly    ERA5 (local store, 1.5° daily) 7-day means 2001–2020: χ200 by the same spherical-harmonic
                 inversion as the live AIFS analyses, anomalies against walker_chi_clim.nc exactly as the
                 live product; u850/v850 anomalies against their own harmonic climatology.
  era5-monthly   ERA5 monthly χ200 anomalies 1991–2020 (same convention) for the GPCP check.
  gpcp           GPCP v2.3 monthly (PSL file, one HTTPS download) anomalies 1991–2020.
  aifs           AIFS-ENS control, 00Z cycles 2025-07-01 onward (Google mirror only): week-1 / week-2 rain
                 and χ200 at mid-week, for the forecast rain bias and the χ drift, against IMERG Late and
                 the AIFS 0-h analyses.
  fit            Gill (1980) forced by rain-anomaly heating; heating scale α (χ200) and β (850 hPa wind) and
                 damping ε fitted to the weekly ERA5 anomalies 20°S–20°N (ε chosen by out-of-sample share),
                 year-block bootstrap CIs, a same-season/other-year permutation null; the GPCP monthly check;
                 writes scripts/mjo/data/reference/gill_rain_ref.nc + gill_rain_calibration.json.

    python scripts/sst/build_gill_rain_ref.py <stage>
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts" / "mjo" / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ecmwf"))
import imerg_tropics as IT                                                     # noqa: E402
import gill_model as GM                                                        # noqa: E402
from gill_rain import to1deg_025, to_gill, chi_anom, gill_fields, pattern_stats, harm_basis   # noqa: E402

WORK = HERE / "data" / "gill_rain_ref"
IMERG = HERE / "data" / "imerg_trop"
REF = ROOT / "scripts" / "mjo" / "data" / "reference"
OUT_NC = REF / "gill_rain_ref.nc"
OUT_JSON = REF / "gill_rain_calibration.json"
STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global"
CLIM_Y0, CLIM_Y1 = 2001, 2025
CAL_Y0, CAL_Y1 = 2001, 2020
NHARM = 4
STAT_LAT = 20.0
EPS_GRID = (0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)


# ── IMERG climatology ─────────────────────────────────────────────────────────
def load_imerg(d0: date, d1: date) -> tuple[pd.DatetimeIndex, np.ndarray]:
    days, arrs = [], []
    d = d0
    while d <= d1:
        f = IMERG / f"{d:%Y}" / f"{d:%Y%m%d}.npy"
        if f.exists():
            days.append(pd.Timestamp(d)); arrs.append(np.load(f))
        d += timedelta(days=1)
    return pd.DatetimeIndex(days), np.array(arrs, dtype="float32")


def stage_imerg_clim() -> None:
    t0 = time.time()
    days, P = load_imerg(date(CLIM_Y0, 1, 1), date(CLIM_Y1, 12, 31))
    nexp = (date(CLIM_Y1, 12, 31) - date(CLIM_Y0, 1, 1)).days + 1
    print(f"  IMERG Late {len(days)}/{nexp} days loaded ({time.time() - t0:.0f}s)", flush=True)
    B = harm_basis(days.dayofyear.values, NHARM)
    Y = P.reshape(len(days), -1).astype("float64")
    ok = np.isfinite(Y)
    coef = np.zeros((B.shape[1], Y.shape[1]))
    allok = ok.all(0)
    c, *_ = np.linalg.lstsq(B, np.where(ok, Y, 0.0)[:, allok], rcond=None); coef[:, allok] = c
    for j in np.where(~allok)[0]:                                    # rare: cells with a missing day
        m = ok[:, j]
        coef[:, j] = np.linalg.lstsq(B[m], Y[m, j], rcond=None)[0] if m.sum() > 1000 else np.nan
    # residual check: day-of-year mean vs harmonic fit
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / "imerg_clim.npz", coef=coef.reshape(B.shape[1], len(IT.LAT1), len(IT.LON1)).astype("float32"),
                        ndays=len(days), nexp=nexp)
    mean = coef[0].reshape(len(IT.LAT1), len(IT.LON1))
    print(f"  harmonic clim: tropical mean {np.nanmean(mean):.2f} mm/day, max {np.nanmax(mean):.1f}; "
          f"{(~allok).sum()} cells with gaps ({(time.time() - t0) / 60:.1f} min)", flush=True)


def imerg_clim_eval(coef: np.ndarray, doy) -> np.ndarray:
    B = harm_basis(np.atleast_1d(doy), NHARM)                           # (t, h)
    return np.einsum("th,hij->tij", B, coef)


# ── ERA5 weekly / monthly ─────────────────────────────────────────────────────
def era5_daily(var: str, y0: int, y1: int) -> xr.DataArray:
    parts = []
    for y in range(y0, y1 + 1):
        ds = xr.open_dataset(STORE / var / f"{var}_{y}.nc"); da = ds[list(ds.data_vars)[0]]
        da = da.assign_coords(latitude=np.round(da.latitude.values.astype("float64"), 3), longitude=np.round(da.longitude.values.astype("float64"), 3))
        parts.append(da.transpose("time", "latitude", "longitude").load()); ds.close()
    return xr.concat(parts, dim="time", join="exact").sortby("time").sortby("latitude")


def week_starts(y0=CAL_Y0, y1=CAL_Y1) -> pd.DatetimeIndex:
    s = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="7D")
    return s[s + pd.Timedelta(days=6) <= pd.Timestamp(f"{y1}-12-31")]


def stage_era5_weekly() -> None:
    import walker_chi as wc
    t0 = time.time()
    clim = xr.open_dataset(REF / "walker_chi_clim.nc").load()
    ws = week_starts()
    out = {}
    for var in ("u200", "v200", "u850", "v850"):
        da = era5_daily(var, CAL_Y0, CAL_Y1)
        wk = np.stack([da.sel(time=slice(s, s + pd.Timedelta(days=6))).mean("time").values for s in ws])
        out[var] = xr.DataArray(wk, dims=("time", "latitude", "longitude"), coords={"time": ws, "latitude": da.latitude, "longitude": da.longitude})
        print(f"  {var}: {len(ws)} weeks ({time.time() - t0:.0f}s)", flush=True)
    chis = []
    for k in range(len(ws)):
        chi, lat, lon = wc.chi_strip(out["u200"].isel(time=k), out["v200"].isel(time=k))
        chis.append(chi)
        if k % 200 == 0:
            print(f"    χ200 {k}/{len(ws)} ({time.time() - t0:.0f}s)", flush=True)
    chis = np.array(chis, "float64")
    mid = ws + pd.Timedelta(days=3)
    lat, lon = clim.latitude.values, clim.longitude.values
    an = chi_anom(chis, pd.DatetimeIndex(mid), clim)
    # 850 hPa wind anomalies on the Gill grid: harmonic climatology (mean + 2 harmonics) of the weekly means
    B = harm_basis(pd.DatetimeIndex(mid).dayofyear.values, 2)
    wind = {}
    for var in ("u850", "v850"):
        g = out[var].sortby("latitude").interp(latitude=GM.LAT2, longitude=GM.LON2).values           # (t, 31, 180)
        Y = g.reshape(len(ws), -1)
        c, *_ = np.linalg.lstsq(B, Y, rcond=None)
        wind[var] = (Y - B @ c).reshape(g.shape).astype("float32")
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / "era5_weekly.npz", start=ws.values.astype("datetime64[D]"), chi=an.astype("float32"), lat=lat, lon=lon,
                        u850=wind["u850"], v850=wind["v850"])
    print(f"  wrote era5_weekly.npz ({(time.time() - t0) / 60:.1f} min)", flush=True)


def stage_era5_monthly() -> None:
    import walker_chi as wc
    from build_walker_basins_clim import monthly_uv
    t0 = time.time()
    clim = xr.open_dataset(REF / "walker_chi_clim.nc").load()
    u, v = monthly_uv("u200"), monthly_uv("v200")
    chis = np.array([wc.chi_strip(u.isel(time=k), v.isel(time=k))[0] for k in range(u.sizes["time"])], "float64")
    t = pd.DatetimeIndex(u.time.values) + pd.Timedelta(days=14)
    an = chi_anom(chis, t, clim)
    np.savez_compressed(WORK / "era5_monthly.npz", time=pd.DatetimeIndex(u.time.values).values.astype("datetime64[D]"), chi=an.astype("float32"),
                        lat=clim.latitude.values, lon=clim.longitude.values)
    print(f"  wrote era5_monthly.npz ({(time.time() - t0) / 60:.1f} min)", flush=True)


def stage_gpcp() -> None:
    import requests
    f = WORK / "gpcp_precip.mon.mean.nc"
    if not f.exists():
        r = requests.get("https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc", timeout=300); r.raise_for_status()
        f.write_bytes(r.content)
    ds = xr.open_dataset(f, drop_variables=["time_bnds"])
    p = ds["precip"].sel(time=slice("1991-01-01", "2020-12-31")).sortby("lat").load()
    if p.sizes["time"] != 360 or not (1.5 < float(p.sel(lat=slice(-30, 30)).mean()) < 6):
        raise SystemExit(f"GPCP implausible: {p.sizes} mean {float(p.mean()):.2f}")
    t = pd.DatetimeIndex(p.time.values); mos = t.month.values
    clim = np.stack([p.values[mos == m].mean(0) for m in range(1, 13)])
    an = p.values - clim[mos - 1]
    g = xr.DataArray(an, dims=("time", "lat", "lon"), coords={"time": p.time, "lat": p.lat, "lon": p.lon})
    g = g.interp(lat=GM.LAT2, lon=GM.LON2, kwargs={"fill_value": None})
    np.savez_compressed(WORK / "gpcp_monthly.npz", time=t.values.astype("datetime64[D]"), anom=g.values.astype("float32"))
    print(f"  GPCP anomalies {g.shape}, tropical sd {float(np.nanstd(g.values)):.2f} mm/day", flush=True)


# ── AIFS-ENS hindcast (bias + drift) ──────────────────────────────────────────
def _aifs_cycle(d: str) -> str:
    import store as ecmwf
    import walker_chi as wc
    o = WORK / "aifs" / f"{d}.npz"
    if o.exists():
        return "cached"
    cyc = ecmwf.Cycle(d, "00")
    try:
        tpf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "tp", "sfc", (), (168, 336)))
        uvf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", ("u", "v"), "pl", (200,), (96, 264)))
    except Exception as ex:                                            # noqa: BLE001
        return f"missing ({repr(ex)[:60]})"
    kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = xr.open_dataset(tpf, **kw)["tp"].sortby("latitude")
    tp = tp.assign_coords(longitude=tp.longitude % 360).sortby("longitude")
    if tp.attrs.get("units", "") in ("m", "metre", "metres"):
        tp = tp * 1000.0
    a168 = tp.sel(step=pd.Timedelta(hours=168)).values; a336 = tp.sel(step=pd.Timedelta(hours=336)).values
    lat, lon = tp.latitude.values, tp.longitude.values
    w1 = to1deg_025(a168, lat, lon) / 7.0; w2 = to1deg_025(a336 - a168, lat, lon) / 7.0
    ds = xr.open_dataset(uvf, **kw)
    chis = []
    for h in (96, 264):
        uu = ds["u"].sel(step=pd.Timedelta(hours=h)); vv = ds["v"].sel(step=pd.Timedelta(hours=h))
        uu = uu.squeeze(drop=True); vv = vv.squeeze(drop=True)
        chis.append(wc.chi_strip(uu, vv)[0])
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o, w1=w1.astype("float32"), w2=w2.astype("float32"), chi96=chis[0], chi264=chis[1])
    import shutil
    shutil.rmtree(ecmwf.CACHE / cyc.tag, ignore_errors=True)
    return "ok"


def stage_aifs() -> None:
    os.environ.setdefault("ECMWF_SOURCES", "google"); os.environ.setdefault("ECMWF_MULTISOURCE", "0")
    os.environ.setdefault("ECMWF_CACHE", str(WORK / "ecmwf_cache"))
    d0 = date(2025, 7, 1); d1 = date.today() - timedelta(days=16)
    days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range((d1 - d0).days + 1)]
    procs = int(os.environ.get("GILL_PROCS", "3"))
    from multiprocessing import get_context
    t0 = time.time()
    with get_context("spawn").Pool(procs) as pool:
        for i, (d, st) in enumerate(zip(days, pool.imap(_aifs_cycle, days))):
            if st != "cached":
                print(f"  {d}: {st}  [{i + 1}/{len(days)}, {(time.time() - t0) / 60:.1f} min]", flush=True)


def _aifs_an(d: str) -> str:
    """AIFS-ENS control step-0 χ200 (the analysis) of a 00Z cycle — the verification for the χ drift."""
    import store as ecmwf
    import walker_chi as wc
    o = WORK / "aifs_an" / f"{d}.npy"
    if o.exists():
        return "cached"
    cyc = ecmwf.Cycle(d, "00")
    try:
        uvf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", ("u", "v"), "pl", (200,), (0,)))
    except Exception as ex:                                            # noqa: BLE001
        return f"missing ({repr(ex)[:60]})"
    ds = xr.open_dataset(uvf, engine="cfgrib", backend_kwargs={"indexpath": ""})
    chi = wc.chi_strip(ds["u"].squeeze(drop=True), ds["v"].squeeze(drop=True))[0]
    o.parent.mkdir(parents=True, exist_ok=True); np.save(o, chi)
    import shutil
    shutil.rmtree(ecmwf.CACHE / cyc.tag, ignore_errors=True)
    return "ok"


def stage_aifs_an() -> None:
    os.environ.setdefault("ECMWF_SOURCES", "google"); os.environ.setdefault("ECMWF_MULTISOURCE", "0")
    os.environ.setdefault("ECMWF_CACHE", str(WORK / "ecmwf_cache"))
    d0 = date(2025, 7, 5); d1 = date.today() - timedelta(days=2)
    days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range((d1 - d0).days + 1)]
    from multiprocessing import get_context
    t0 = time.time()
    with get_context("spawn").Pool(int(os.environ.get("GILL_PROCS", "3"))) as pool:
        for i, (d, st) in enumerate(zip(days, pool.imap(_aifs_an, days))):
            if st != "cached":
                print(f"  {d}: {st}  [{i + 1}/{len(days)}, {(time.time() - t0) / 60:.1f} min]", flush=True)


# ── Gill responses for the calibration sets ───────────────────────────────────
def imerg_windows(starts: pd.DatetimeIndex, ndays) -> np.ndarray:
    """IMERG Late anomaly means over [start, start + n) on the Gill grid, against the harmonic climatology."""
    coef = np.load(WORK / "imerg_clim.npz")["coef"].astype("float64")
    out = np.full((len(starts), GM.LAT2.size, GM.LON2.size), np.nan)
    for k, s in enumerate(starts):
        n = ndays[k] if np.ndim(ndays) else ndays
        days = pd.date_range(s, periods=int(n), freq="D")
        arr = [np.load(IMERG / f"{d:%Y}" / f"{d:%Y%m%d}.npy") for d in days if (IMERG / f"{d:%Y}" / f"{d:%Y%m%d}.npy").exists()]
        if len(arr) < 0.85 * n:
            continue
        ok = [d for d in days if (IMERG / f"{d:%Y}" / f"{d:%Y%m%d}.npy").exists()]
        a = np.nanmean(np.array(arr, "float64"), 0) - imerg_clim_eval(coef, pd.DatetimeIndex(ok).dayofyear.values).mean(0)
        out[k] = to_gill(np.nan_to_num(a))
    return out


def _gill_chunk(args):
    eps, P = args
    solve = GM.gill_solver(eps=eps)
    C, U, V = [], [], []
    for p in P:
        if not np.isfinite(p).all():
            C.append(None); U.append(None); V.append(None); continue
        c, u, v = gill_fields(p, solve)
        C.append(c.astype("float32")); U.append(u.astype("float32")); V.append(v.astype("float32"))
    return C, U, V


def gill_batch(P: np.ndarray, eps: float, procs: int = 12):
    from multiprocessing import get_context
    chunks = np.array_split(np.arange(len(P)), procs)
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(_gill_chunk, [(eps, P[c]) for c in chunks])
    C, U, V = [], [], []
    for c, u, v in res:
        C += c; U += u; V += v
    shp_c = next(x.shape for x in C if x is not None); shp_u = next(x.shape for x in U if x is not None)
    f = lambda L, shp: np.array([x if x is not None else np.full(shp, np.nan, "float32") for x in L])
    return f(C, shp_c), f(U, shp_u), f(V, shp_u)


def stage_gill() -> None:
    """Weekly IMERG anomalies 2001–2020 → Gill χ200/u/v per unit scale for every ε; monthly GPCP and IMERG
    at the default ε for the monthly check (recomputed at the fitted ε by `fit`)."""
    t0 = time.time()
    z = np.load(WORK / "era5_weekly.npz")
    ws = pd.DatetimeIndex(z["start"])
    P = imerg_windows(ws, 7)
    np.save(WORK / "imerg_weekly_gill.npy", P.astype("float32"))
    print(f"  IMERG weekly anomalies {P.shape}, {np.isnan(P[:, 0, 0]).sum()} weeks short ({time.time() - t0:.0f}s)", flush=True)
    for eps in EPS_GRID:
        f = WORK / f"gill_weekly_eps{eps:g}.npz"
        if f.exists():
            continue
        C, U, V = gill_batch(P, eps)
        np.savez_compressed(f, chi=C, u=U, v=V)
        print(f"  ε {eps:g}: {len(C)} weekly Gill responses ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # monthly: IMERG 2001–2020 means
    mz = pd.date_range(f"{CAL_Y0}-01-01", f"{CAL_Y1}-12-01", freq="MS")
    Pm = imerg_windows(mz, np.array(mz.days_in_month))
    np.save(WORK / "imerg_monthly_gill.npy", Pm.astype("float32"))
    print(f"  IMERG monthly anomalies {Pm.shape} ({time.time() - t0:.0f}s)", flush=True)


# ── the fit ───────────────────────────────────────────────────────────────────
def _w_chi(lat, lon):
    m = np.abs(lat) <= STAT_LAT
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * m[:, None]
    if np.isclose(lon[-1] - lon[0], 360.0):
        w[:, -1] = 0.0                                                  # the DH2 grid repeats 0° as 360°
    return w / w.sum()


def _alpha(O, G, w, idx):
    return float((w * G[idx] * O[idx]).sum() / (w * G[idx] ** 2).sum())


def _ev(O, G, w, idx, a):
    return float(1 - (w * (O[idx] - a * G[idx]) ** 2).sum() / (w * O[idx] ** 2).sum())


def _pattern_r(O, G, w):
    """Centred weighted pattern correlation per sample."""
    om = (w * O).sum((-2, -1), keepdims=True); gm = (w * G).sum((-2, -1), keepdims=True)
    num = (w * (O - om) * (G - gm)).sum((-2, -1))
    den = np.sqrt((w * (O - om) ** 2).sum((-2, -1)) * (w * (G - gm) ** 2).sum((-2, -1)))
    return num / np.maximum(den, 1e-30)


def _seasons(months):
    return np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])[np.asarray(months) - 1]


def fit_all() -> None:
    rng = np.random.default_rng(20260928)
    t0 = time.time()
    z = np.load(WORK / "era5_weekly.npz")
    ws = pd.DatetimeIndex(z["start"]); lat, lon = z["lat"], z["lon"]
    O = z["chi"].astype("float64"); Uo = z["u850"].astype("float64"); Vo = z["v850"].astype("float64")
    w = _w_chi(lat, lon)
    wg = np.cos(np.deg2rad(GM.LAT2))[:, None] * (np.abs(GM.LAT2) <= STAT_LAT)[:, None] * np.ones((1, GM.LON2.size)); wg /= wg.sum()
    years = ws.year.values; mid = ws + pd.Timedelta(days=3)
    ok = np.isfinite(np.load(WORK / "imerg_weekly_gill.npy")[:, 0, 0])
    idx_all = np.where(ok)[0]
    odd = idx_all[years[idx_all] % 2 == 1]; even = idx_all[years[idx_all] % 2 == 0]
    # ε scan
    scan = {}
    for eps in EPS_GRID:
        G = np.load(WORK / f"gill_weekly_eps{eps:g}.npz")["chi"].astype("float64")
        a_all = _alpha(O, G, w, idx_all)
        a_o, a_e = _alpha(O, G, w, odd), _alpha(O, G, w, even)
        sse = (w * (O[even] - a_o * G[even]) ** 2).sum() + (w * (O[odd] - a_e * G[odd]) ** 2).sum()
        oos = float(1 - sse / (w * O[idx_all] ** 2).sum())
        r = _pattern_r(O[idx_all], G[idx_all], w)
        scan[eps] = {"alpha": a_all, "share_in": _ev(O, G, w, idx_all, a_all), "share_oos": oos, "r_median": float(np.median(r))}
        print(f"  ε {eps:5g}: α {a_all:.3g}  share in {scan[eps]['share_in']:.3f}  OOS {oos:.3f}  median r {scan[eps]['r_median']:.3f}", flush=True)
    eps = max(scan, key=lambda e: scan[e]["share_oos"])
    zz = np.load(WORK / f"gill_weekly_eps{eps:g}.npz")
    G = zz["chi"].astype("float64"); Gu = zz["u"].astype("float64"); Gv = zz["v"].astype("float64")
    alpha = scan[eps]["alpha"]
    r_w = _pattern_r(O[idx_all], G[idx_all], w)
    # 850 hPa wind scale
    num = (wg * (Gu[idx_all] * Uo[idx_all] + Gv[idx_all] * Vo[idx_all])).sum(); den = (wg * (Gu[idx_all] ** 2 + Gv[idx_all] ** 2)).sum()
    beta = float(num / den)
    ev_wind = float(1 - (wg * ((Uo[idx_all] - beta * Gu[idx_all]) ** 2 + (Vo[idx_all] - beta * Gv[idx_all]) ** 2)).sum() / (wg * (Uo[idx_all] ** 2 + Vo[idx_all] ** 2)).sum())
    rv = (wg * (Gu[idx_all] * Uo[idx_all] + Gv[idx_all] * Vo[idx_all])).sum((1, 2)) / np.sqrt((wg * (Gu[idx_all] ** 2 + Gv[idx_all] ** 2)).sum((1, 2)) * (wg * (Uo[idx_all] ** 2 + Vo[idx_all] ** 2)).sum((1, 2)))
    ru = _pattern_r(Uo[idx_all], Gu[idx_all], wg)
    # 28-day blocks: four consecutive weeks inside the record
    blocks = [idx_all[i:i + 4] for i in range(0, len(idx_all) - 3, 4) if idx_all[i + 3] - idx_all[i] == 3]
    O28 = np.array([O[b].mean(0) for b in blocks]); G28 = np.array([G[b].mean(0) for b in blocks])
    b_idx = np.arange(len(blocks)); y28 = np.array([years[b[0]] for b in blocks]); m28 = np.array([mid[b[1]].month for b in blocks])
    r_28 = _pattern_r(O28, G28, w)
    share_28 = _ev(O28, G28, w, b_idx, alpha); alpha_28 = _alpha(O28, G28, w, b_idx)
    # year-block bootstrap
    uy = np.unique(years[idx_all]); byy = {y: idx_all[years[idx_all] == y] for y in uy}; by28 = {y: b_idx[y28 == y] for y in uy}
    boot = {"alpha": [], "share": [], "r_med": [], "share28": [], "r28_med": [], "beta": [], "ev_wind": []}
    for _ in range(2000):
        ys = rng.choice(uy, len(uy), replace=True)
        ii = np.concatenate([byy[y] for y in ys]); jj = np.concatenate([by28[y] for y in ys])
        a = _alpha(O, G, w, ii); boot["alpha"].append(a); boot["share"].append(_ev(O, G, w, ii, a))
        rr = r_w[np.searchsorted(idx_all, ii)]; boot["r_med"].append(np.median(rr))
        boot["share28"].append(_ev(O28, G28, w, jj, a)); boot["r28_med"].append(np.median(r_28[jj]))
        nb = (wg * (Gu[ii] * Uo[ii] + Gv[ii] * Vo[ii])).sum(); db = (wg * (Gu[ii] ** 2 + Gv[ii] ** 2)).sum(); bb = nb / db
        boot["beta"].append(bb)
        boot["ev_wind"].append(1 - (wg * ((Uo[ii] - bb * Gu[ii]) ** 2 + (Vo[ii] - bb * Gv[ii]) ** 2)).sum() / (wg * (Uo[ii] ** 2 + Vo[ii] ** 2)).sum())
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    # null: rain from another year, same time of year (nearest start day-of-year)
    doy = ws.dayofyear.values
    slot = {}
    for y in uy:
        iy = byy[y]
        slot[y] = (iy, doy[iy])
    def partner(t, y2):
        iy, dy = slot[y2]
        d = np.abs(((dy - doy[t]) + 182) % 365 - 182)
        return iy[np.argmin(d)]
    null_share, null_r, null_r28 = [], [], []
    for p in range(500):
        perm = uy.copy()
        while True:
            rng.shuffle(perm)
            if not np.any(perm == uy):
                break
        mp = dict(zip(uy, perm))
        src = np.array([partner(t, mp[years[t]]) for t in idx_all])
        Gp = G[src]
        a = float((w * Gp * O[idx_all]).sum() / (w * Gp ** 2).sum())
        null_share.append(float(1 - (w * (O[idx_all] - a * Gp) ** 2).sum() / (w * O[idx_all] ** 2).sum()))
        rp = _pattern_r(O[idx_all], Gp, w)
        null_r.append(rp)
        pos = {t: k for k, t in enumerate(idx_all)}
        G28p = np.array([Gp[[pos[t] for t in b]].mean(0) for b in blocks])
        null_r28.append(_pattern_r(O28, G28p, w))
    null_r = np.concatenate(null_r); null_r28 = np.concatenate(null_r28)
    share_in = scan[eps]["share_in"]
    p_share = float((1 + np.sum(np.array(null_share) >= share_in)) / (1 + len(null_share)))
    q95 = {"w7": float(np.percentile(null_r, 95)), "w28": float(np.percentile(null_r28, 95))}
    seas = _seasons(mid[idx_all].month.values); seas28 = _seasons(m28)
    by_season = {}
    for s in ("DJF", "MAM", "JJA", "SON"):
        ii = idx_all[seas == s]
        by_season[s] = {"share": round(_ev(O, G, w, ii, alpha), 3), "r_median": round(float(np.median(r_w[seas == s])), 3),
                        "frac_sig": round(float(np.mean(r_w[seas == s] > q95["w7"])), 3), "n_weeks": int(len(ii))}
    print(f"  ε = {eps:g}: α {alpha:.3g} [{ci['alpha'][0]:.3g}, {ci['alpha'][1]:.3g}], weekly share {share_in:.3f} [{ci['share'][0]:.3f}, {ci['share'][1]:.3f}] "
          f"OOS {scan[eps]['share_oos']:.3f}, median r {np.median(r_w):.3f} [{ci['r_med'][0]:.3f}, {ci['r_med'][1]:.3f}]; null share p {p_share:.4f}, "
          f"null r q95 {q95['w7']:.3f} / 28-d {q95['w28']:.3f}", flush=True)
    print(f"  28-day: share {share_28:.3f} [{ci['share28'][0]:.3f}, {ci['share28'][1]:.3f}], median r {np.median(r_28):.3f}, α28 {alpha_28:.3g}", flush=True)
    print(f"  850 hPa wind: β {beta:.3g} [{ci['beta'][0]:.3g}, {ci['beta'][1]:.3g}], share {ev_wind:.3f} [{ci['ev_wind'][0]:.3f}, {ci['ev_wind'][1]:.3f}], "
          f"median vector r {np.median(rv):.3f}, median u r {np.median(ru):.3f}", flush=True)
    print(f"  by season: {by_season}", flush=True)
    # monthly check: GPCP 1991–2020 and IMERG 2001–2020 against ERA5 monthly χ200
    zm = np.load(WORK / "era5_monthly.npz"); tm = pd.DatetimeIndex(zm["time"]); Om = zm["chi"].astype("float64")
    gp = np.load(WORK / "gpcp_monthly.npz"); Pg = np.nan_to_num(gp["anom"].astype("float64"))
    Gg, _, _ = gill_batch(Pg, eps)
    Pim = np.load(WORK / "imerg_monthly_gill.npy").astype("float64"); Gi, _, _ = gill_batch(np.nan_to_num(Pim), eps)
    im_idx = np.where((tm.year >= CAL_Y0) & (tm.year <= CAL_Y1))[0]
    monthly = {}
    for name, Gm, sel in (("gpcp_1991_2020", Gg, np.arange(len(tm))), ("imerg_2001_2020", Gi, None)):
        Ox = Om if sel is not None else Om[im_idx]
        ii = np.arange(len(Ox)); yy = (tm if sel is not None else tm[im_idx]).year.values
        a = _alpha(Ox, Gm, w, ii); rr = _pattern_r(Ox, Gm, w)
        bs_a, bs_s, bs_r = [], [], []
        uyy = np.unique(yy)
        for _ in range(2000):
            ys = rng.choice(uyy, len(uyy), replace=True); jj = np.concatenate([ii[yy == y] for y in ys])
            ab = _alpha(Ox, Gm, w, jj); bs_a.append(ab); bs_s.append(_ev(Ox, Gm, w, jj, ab)); bs_r.append(np.median(rr[jj]))
        monthly[name] = {"alpha": a, "alpha_ci": [float(np.percentile(bs_a, 2.5)), float(np.percentile(bs_a, 97.5))],
                         "share": _ev(Ox, Gm, w, ii, a), "share_ci": [float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5))],
                         "share_with_weekly_alpha": _ev(Ox, Gm, w, ii, alpha),
                         "r_median": float(np.median(rr)), "r_median_ci": [float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))], "n": int(len(ii))}
        print(f"  monthly {name}: α {a:.3g} {monthly[name]['alpha_ci']}, share {monthly[name]['share']:.3f} {monthly[name]['share_ci']}, "
              f"with weekly α {monthly[name]['share_with_weekly_alpha']:.3f}, median r {monthly[name]['r_median']:.3f}", flush=True)
    # forecast references from the AIFS hindcast
    rain_bias, chi_drift, fc_skill = aifs_references()
    coef = np.load(WORK / "imerg_clim.npz")["coef"]
    Tg = 1.0 / np.sqrt(GM.C_WAVE * GM.BETA)
    cal = {"eps": eps, "damp_days": float(Tg / eps / 86400.0), "alpha": alpha, "alpha_ci": ci["alpha"], "beta": beta, "beta_ci": ci["beta"],
           "share_w7": share_in, "share_w7_ci": ci["share"], "share_w7_oos": scan[eps]["share_oos"], "share_w7_p_null": p_share,
           "null_share_q95": float(np.percentile(null_share, 95)),
           "r_w7_median": float(np.median(r_w)), "r_w7_median_ci": ci["r_med"], "r_w7_iqr": [float(np.percentile(r_w, 25)), float(np.percentile(r_w, 75))],
           "frac_w7_significant": float(np.mean(r_w > q95["w7"])),
           "share_w28": share_28, "share_w28_ci": ci["share28"], "r_w28_median": float(np.median(r_28)), "r_w28_median_ci": ci["r28_med"],
           "frac_w28_significant": float(np.mean(r_28 > q95["w28"])), "alpha_w28": alpha_28,
           "wind850_share": ev_wind, "wind850_share_ci": ci["ev_wind"], "wind850_vector_r_median": float(np.median(rv)), "u850_r_median": float(np.median(ru)),
           "null_r_q95": q95, "by_season": by_season, "eps_scan": {f"{k:g}": {kk: round(vv, 4) for kk, vv in v.items()} for k, v in scan.items()},
           "monthly": monthly, "forecast": fc_skill,
           "n_weeks": int(len(idx_all)), "n_blocks28": int(len(blocks)), "calibration_years": f"{CAL_Y0}-{CAL_Y1}", "rain_clim_years": f"{CLIM_Y0}-{CLIM_Y1}",
           "stat_band_deg": STAT_LAT, "c_wave_ms": GM.C_WAVE, "taper_deg": [18.0, 22.0]}
    ds = xr.Dataset(
        {"rain_clim": (("harm", "lat1", "lon1"), coef.astype("float32")),
         "rain_bias": (("lead", "harm1", "lat1", "lon1"), rain_bias.astype("float32")),
         "chi_drift": (("lead", "latitude", "longitude"), chi_drift.astype("float32"))},
        coords={"harm": np.arange(coef.shape[0]), "harm1": np.arange(3), "lat1": IT.LAT1, "lon1": IT.LON1, "lead": [1, 2], "latitude": lat, "longitude": lon},
        attrs={"alpha": alpha, "beta": beta, "eps": eps, "taper": json.dumps([18.0, 22.0]), "rain_nharm": NHARM,
               "calibration": json.dumps(_round(cal)),
               "note": "rain_clim: IMERG V07 Late daily 1-deg harmonic climatology (mean + 4 harmonics, 2001-2025), mm/day; "
                       "rain_bias: AIFS-ENS control week-lead rain minus IMERG Late (mean + annual harmonic of mid-week day of year), 00Z cycles 2025-07 on, 3x3 smoothed; "
                       "chi_drift: AIFS-ENS control chi200 at mid-week (96 h / 264 h) minus the verifying control analysis, mean; "
                       "alpha/beta: Gill chi200 and 850 hPa wind scales per unit of gill_rain.gill_fields at eps, fitted to ERA5 weekly anomalies 2001-2020, 20S-20N"})
    ds.to_netcdf(OUT_NC, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    OUT_JSON.write_text(json.dumps(_round(cal), indent=1))
    print(f"wrote {OUT_NC} ({OUT_NC.stat().st_size / 1e6:.2f} MB) and {OUT_JSON.name} in {(time.time() - t0) / 60:.1f} min", flush=True)


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.4g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def aifs_references():
    """Rain bias (lead, 3 harmonics, 60, 360) and χ drift (lead, lat, lon) from the AIFS-ENS control hindcast,
    plus the week-1/2 anomaly skill of the corrected forecast rain against IMERG Late (pattern r, 20°S–20°N)."""
    files = sorted((WORK / "aifs").glob("*.npz"))
    coef = np.load(WORK / "imerg_clim.npz")["coef"].astype("float64")
    diffs = {1: [], 2: []}; doys = {1: [], 2: []}; pairs = {1: [], 2: []}; drift = {1: [], 2: []}
    for f in files:
        d0 = pd.Timestamp(f.stem)
        z = np.load(f)
        for lead, key, h in ((1, "w1", 96), (2, "w2", 264)):
            s = d0 + pd.Timedelta(days=7 * (lead - 1))
            days = pd.date_range(s, periods=7, freq="D")
            obs = [IMERG / f"{d:%Y}" / f"{d:%Y%m%d}.npy" for d in days]
            if all(p.exists() for p in obs):
                o = np.mean([np.load(p) for p in obs], 0).astype("float64")
                diffs[lead].append(z[key].astype("float64") - o); doys[lead].append((s + pd.Timedelta(days=3.5)).dayofyear)
                clim = imerg_clim_eval(coef, days.dayofyear.values).mean(0)
                pairs[lead].append((z[key].astype("float64") - clim, o - clim, (s + pd.Timedelta(days=3.5)).dayofyear))
            va = (d0 + pd.Timedelta(hours=h)).strftime("%Y%m%d")
            an = WORK / "aifs_an" / f"{va}.npy"
            if an.exists():
                drift[lead].append(z["chi96" if lead == 1 else "chi264"].astype("float64") - np.load(an).astype("float64"))
    bias = np.zeros((2, 3, len(IT.LAT1), len(IT.LON1))); skill = {}
    for k, lead in enumerate((1, 2)):
        D = np.array(diffs[lead]); B = harm_basis(np.array(doys[lead]), 1)
        c, *_ = np.linalg.lstsq(B, D.reshape(len(D), -1), rcond=None)
        c = c.reshape(3, len(IT.LAT1), len(IT.LON1))
        # 3×3 box smoothing (periodic in longitude)
        sm = sum(np.roll(np.roll(c, i, -1), 0, -2) for i in (-1, 0, 1)) / 3.0
        pad = np.concatenate([sm[:, :1], sm, sm[:, -1:]], 1)
        bias[k] = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0
        # skill of the corrected anomalies (in-sample bias; descriptive)
        rs_raw, rs_cor = [], []
        m = np.abs(IT.LAT1) <= STAT_LAT; wv = np.cos(np.deg2rad(IT.LAT1[m]))[:, None] * np.ones((1, len(IT.LON1)))
        for fa, oa, dy in pairs[lead]:
            b = np.einsum("h,hij->ij", harm_basis([dy], 1)[0], bias[k])
            for lst, f_ in ((rs_raw, fa), (rs_cor, fa - b)):
                x = f_[m]; y = oa[m]; ww = wv / wv.sum()
                xm, ym = (ww * x).sum(), (ww * y).sum()
                lst.append((ww * (x - xm) * (y - ym)).sum() / np.sqrt((ww * (x - xm) ** 2).sum() * (ww * (y - ym) ** 2).sum()))
        skill[f"week{lead}"] = {"n_cycles": len(pairs[lead]), "rain_anom_r_median_raw": float(np.median(rs_raw)), "rain_anom_r_median_corrected": float(np.median(rs_cor)),
                                "mean_bias_tropics_mm_day": float(np.mean(np.array(diffs[lead])[:, m])), "n_drift": len(drift[lead])}
        print(f"  AIFS week {lead}: {len(pairs[lead])} cycles, rain-anomaly r vs IMERG median raw {np.median(rs_raw):.2f} → corrected {np.median(rs_cor):.2f}; "
              f"tropical mean bias {skill[f'week{lead}']['mean_bias_tropics_mm_day']:+.2f} mm/day; χ drift from {len(drift[lead])} cycles", flush=True)
    chi = np.array([np.mean(drift[1], 0), np.mean(drift[2], 0)])
    return bias, chi, skill


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "fit"
    {"imerg-clim": stage_imerg_clim, "era5-weekly": stage_era5_weekly, "era5-monthly": stage_era5_monthly,
     "gpcp": stage_gpcp, "aifs": stage_aifs, "aifs-an": stage_aifs_an, "gill": stage_gill, "fit": fit_all}[stage]()
