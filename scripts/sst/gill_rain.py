#!/usr/bin/env python3
"""The Gill (1980) model forced by observed and forecast tropical rainfall, against the observed circulation.

Heating: latent heat of the rainfall ANOMALY (1 mm/day ≈ 28.9 W m⁻² column mean), 20°S–20°N with a taper to
22°, on the first baroclinic mode (Gill's shallow-water model: c = 30 m/s, damping ε fitted). One heating scale
α for the upper-level velocity potential χ200 and one β for the 850 hPa wind were fitted ONCE against ERA5
weekly anomalies 2001–2020 (build_gill_rain_ref.py → scripts/mjo/data/reference/gill_rain_ref.nc); nothing is
re-tuned per week, so the pattern correlations and explained shares shown are honest out-of-fit numbers.

Rows: the last four 7-day windows and their 28-day mean, observed:
  rain anomaly  IMERG V07 Late daily (GES DISC OPeNDAP, 1°), against its own 2001–2025 harmonic climatology;
  Gill          χ200 response (upper-level wind = −(lower-level wind), same spherical-harmonic inversion as the
                observations) and the Gill 850 hPa wind as vectors;
  observed      χ200 anomaly from the AIFS-ENS 0-h analyses (the walker_chi history), vs ERA5 1991–2020;
  residual      observed − Gill: what tropical latent heating on the first baroclinic mode does not explain.
Then the forecast strip, week 1 and week 2 of the latest AIFS-ENS cycle: the ensemble-mean rain anomaly (the
AIFS mean rain bias for that lead, measured against IMERG Late over 2025-07 → 2026-09, removed), its Gill
response, the ensemble-mean forecast χ200 anomaly (its lead-dependent drift against the AIFS analyses
removed) and forecast − Gill.

The IMERG cache (1° daily, last 150 days) lives on the frames branch (assets/sst/anim/gillrain); a missing
cache costs ~2 min of OPeNDAP requests. Runs in mjo.yml after walker.py (χ history) and wind200_vpot.py
(ensemble-mean u/v 200 in the store cache); needs EARTHDATA_USERNAME / EARTHDATA_PASSWORD and earthaccess.

    SST_SITE_ROOT=... python scripts/sst/gill_rain.py --date 20260928 --time 00
"""
from __future__ import annotations

import argparse
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
import gill_model as GM                                                          # noqa: E402
import imerg_tropics as IT                                                       # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", ROOT))
ASSETS = SITE / "assets" / "sst"
REF = ROOT / "scripts" / "mjo" / "data" / "reference"
REFNC = REF / "gill_rain_ref.nc"
CHICLIM = REF / "walker_chi_clim.nc"
CACHE = ASSETS / "anim" / "gillrain" / "imerg_late_1deg.nc"
KEEP_DAYS = 150
NWEEKS = 4
STAT_LAT = 20.0
INK, MUTED = "#1a1a1a", "#6f6b64"
QSCALE = 60.0                      # m/s per panel width for the 850 hPa box-mean vectors (2 m/s ≈ one box spacing)
QKEY = 2.0
DPI = 140


# ── shared helpers (also used by the calibration) ─────────────────────────────
def harm_basis(doy, n: int) -> np.ndarray:
    w = 2 * np.pi * np.asarray(doy, float) / 365.25
    cols = [np.ones_like(w)]
    for k in range(1, n + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.stack(cols, -1)


def to_gill(p1: np.ndarray) -> np.ndarray:
    """1° field(s) (..., 60, 360) on IT.LAT1 × IT.LON1 → the Gill grid (..., 31, 180): 2×2 box means (centres on odd
    degrees), then a half-cell linear shift onto the even-degree Gill points (periodic in longitude)."""
    a = np.asarray(p1, float)
    b = a.reshape(*a.shape[:-2], 30, 2, 180, 2).mean(axis=(-3, -1))           # centres lat −29…29, lon 1…359
    lon = 0.5 * (b + np.roll(b, 1, axis=-1))                                  # → lon 0, 2, …, 358
    mid = 0.5 * (lon[..., :-1, :] + lon[..., 1:, :])                           # → lat −28 … 28
    first = lon[..., :1, :]; last = lon[..., -1:, :]                           # ±30: nearest (heating is zero there)
    return np.concatenate([first, mid, last], axis=-2)


def chi_anom(chi: np.ndarray, times: pd.DatetimeIndex, clim: xr.Dataset) -> np.ndarray:
    """χ200 anomaly: minus the ERA5 1991–2020 harmonic climatology and that month's 1991–2020 trend (the same
    convention as walker_chi_weekly.chi_anomalies)."""
    times = pd.DatetimeIndex(times)
    doy = times.dayofyear.values; ang = 2 * np.pi * doy / 365.25
    B = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)
    c = clim.chi_coef.sel(level=200).values
    yr = times.year.values + (doy - 1) / 365.25
    sl = clim.chi_slope.sel(level=200).values[times.month.values - 1]
    return chi - np.einsum("th,hij->tij", B, c) - sl * (yr - 2005.5)[:, None, None]


_GLOBAL_LAT = np.arange(-90.0, 90.1, 2.0)


def gill_fields(P_gill: np.ndarray, solve, taper=(18.0, 22.0)):
    """Rain anomaly on the Gill grid (mm/day) → (χ200 strip on the walker_chi grid, u_low, v_low) per unit scale.
    Upper-level wind = −(lower-level wind) (first baroclinic mode), faded to zero over 24–30° (the sponge),
    embedded in a global field and inverted with the same spherical-harmonic solver as the observed χ."""
    import walker_chi as wc
    Q = GM.heating_from_precip(P_gill, GM.LAT2, *taper)
    u, v, _ = solve(Q)
    fade = np.clip((30.0 - np.abs(GM.LAT2)) / 6.0, 0, 1)[:, None]
    ug = np.zeros((_GLOBAL_LAT.size, GM.LON2.size)); vg = np.zeros_like(ug)
    j0 = int(np.where(np.isclose(_GLOBAL_LAT, GM.LAT2[0]))[0][0])
    ug[j0:j0 + GM.LAT2.size] = -u * fade; vg[j0:j0 + GM.LAT2.size] = -v * fade
    # 0…358 only: velocity_potential maps longitude mod 360, so an appended 360 would duplicate 0, and the
    # Gauss–Legendre longitudes it interpolates to stop short of 358 (lmax 42: 0…355.8)
    mk = lambda f: xr.DataArray(f, dims=("latitude", "longitude"), coords={"latitude": _GLOBAL_LAT, "longitude": GM.LON2})
    chi, _, _ = wc.chi_strip(mk(ug), mk(vg))
    return chi.astype("float64"), u, v


def wind_boxes(u: np.ndarray, v: np.ndarray, half_lat=18.0):
    """Gill-grid (31 × 180, 2°) winds → 6° × 12° box means centred on 0, ±6, ±12, ±18° and 5, 17, … °E."""
    rows = [c for c in np.arange(-half_lat, half_lat + 0.1, 6.0)]
    U, V, LA, LO = [], [], [], []
    for c in rows:
        j = np.where(np.abs(GM.LAT2 - c) <= 2.01)[0]
        U.append(u[j].mean(0).reshape(-1, 6).mean(1)); V.append(v[j].mean(0).reshape(-1, 6).mean(1))
        LO.append(GM.LON2.reshape(-1, 6).mean(1)); LA.append(np.full(GM.LON2.size // 6, c))
    return (np.array(LA), np.array(LO)), np.array(U), np.array(V)


def strip_weights(lat, nlon, half=STAT_LAT):
    m = np.abs(lat) <= half
    return m, np.cos(np.deg2rad(lat[m]))[:, None] * np.ones((1, nlon))


def pattern_stats(obs: np.ndarray, mod: np.ndarray, lat) -> dict:
    """Centred pattern correlation and explained share 1 − Σw(obs − mod)² / Σw obs² over |lat| ≤ STAT_LAT."""
    m, w = strip_weights(lat, obs.shape[-1])
    o = obs[m]; g = mod[m]; w = w / w.sum()
    om, gm = (w * o).sum(), (w * g).sum()
    r = float((w * (o - om) * (g - gm)).sum() / np.sqrt((w * (o - om) ** 2).sum() * (w * (g - gm) ** 2).sum() + 1e-30))
    share = float(1 - (w * (o - g) ** 2).sum() / ((w * o ** 2).sum() + 1e-30))
    return {"r": r, "share": share, "rms_obs": float(np.sqrt((w * o ** 2).sum())), "rms_res": float(np.sqrt((w * (o - g) ** 2).sum()))}


def to1deg_025(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """0.25° field (lat asc, lon 0..359.75) → 1° cell means on IT.LAT1 × IT.LON1 (trapezoid weights)."""
    w = np.array([0.5, 1, 1, 1, 0.5]) / 4.0
    li = {round(float(x), 2): i for i, x in enumerate(lat)}
    out = np.zeros((len(IT.LAT1), len(IT.LON1)))
    ff = np.concatenate([f, f[:, :1]], 1)                              # wrap for the 360 edge
    rows = []
    for c in IT.LAT1:
        idx = [li[round(c - 0.5 + 0.25 * k, 2)] for k in range(5)]
        rows.append((w[:, None] * ff[idx]).sum(0))
    R = np.array(rows)                                                  # (60, 1441)
    for j in range(len(IT.LON1)):
        out[:, j] = (R[:, 4 * j:4 * j + 5] * w[None]).sum(1)
    return out


# ── observed rain cache ───────────────────────────────────────────────────────
def load_cache() -> xr.DataArray | None:
    if CACHE.exists():
        try:
            return xr.open_dataarray(CACHE).load()
        except Exception as ex:                                              # noqa: BLE001
            print(f"  cache unreadable ({str(ex)[:60]}); rebuilding", flush=True)
    return None


def update_rain(need_from: date, need_to: date) -> xr.DataArray:
    old = load_cache()
    have = set() if old is None else {pd.Timestamp(t).date() for t in old.time.values}
    want = [need_from + timedelta(days=k) for k in range((need_to - need_from).days + 1)]
    miss = [d for d in want if d not in have]
    new = []
    if miss:
        urls = IT.granule_urls(min(miss), max(miss))
        for d in miss:
            if d in urls:
                g = IT.fetch_day(urls[d])
                if g is not None:
                    new.append((d, g))
        print(f"  IMERG Late: {len(miss)} days missing from the cache, {len(new)} fetched "
              f"({sum(1 for d in miss if d not in urls)} not yet published)", flush=True)
    if new:
        da = xr.DataArray(np.stack([g for _, g in new]), dims=("time", "lat", "lon"),
                          coords={"time": pd.DatetimeIndex([pd.Timestamp(d) for d, _ in new]), "lat": IT.LAT1, "lon": IT.LON1}, name="precipitation",
                          attrs={"units": "mm/day", "source": "IMERG V07 Late daily (GPM_3IMERGDL), 1-degree box means of the 0.1-degree field"})
        da = xr.concat([old, da], dim="time").sortby("time") if old is not None else da
        da = da.isel(time=slice(-KEEP_DAYS, None))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp.nc")
        da.to_netcdf(tmp, encoding={"precipitation": {"zlib": True, "complevel": 4, "dtype": "float32"}}); tmp.replace(CACHE)
        return da
    if old is None:
        raise SystemExit("no IMERG Late data (cache empty and nothing fetched)")
    return old


# ── forecast (AIFS-ENS, from the store cache the MJO job already filled) ──────
def forecast_fields(dt: str, hh: str, clim_chi: xr.Dataset, ref: xr.Dataset):
    import store as ecmwf
    import download_aifs
    import wind200_vpot as w2
    import walker_chi as wc
    cyc = ecmwf.Cycle(dt, hh)
    steps = list(download_aifs.rmm_steps(hh))                                  # 00Z-anchored daily valid times
    first = steps[0] if hh == "00" else steps[1]                               # first 00Z valid time
    init = pd.Timestamp(f"{dt}T{hh}:00")
    t0 = init + pd.Timedelta(hours=first)
    out = []
    # rain: ensemble mean of cf + 50 pf, accumulated from init
    tp_steps = tuple(s for s in steps if s > 0)
    acc = {}
    for typ in ("pf", "cf"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "tp", "sfc", (), tp_steps))
        da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["tp"]
        if da.attrs.get("units", "").strip() in ("m", "metre", "metres"):
            da = da * 1000.0
        da = da.sortby("latitude").sel(latitude=slice(-30.2, 30.2))
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
        for h in (first + 168, first + 336) + ((first,) if first > 0 else ()):
            f = da.sel(step=pd.Timedelta(hours=h))
            n = f.sizes.get("number", 1)
            acc.setdefault(h, [0.0, 0])
            acc[h][0] = acc[h][0] + (f.sum("number") if "number" in f.dims else f).values; acc[h][1] += n
        lat, lon = da.latitude.values, da.longitude.values
    em = {h: s / n for h, (s, n) in acc.items()}
    base = em.get(first, 0.0)
    wk_rain = [to1deg_025(em[first + 168] - base, lat, lon) / 7.0, to1deg_025(em[first + 336] - em[first + 168], lat, lon) / 7.0]
    # χ200: ensemble-mean u/v 200 at the daily 00Z steps (wind200_vpot's own ensemble mean, a store cache hit)
    ds = w2._ens_mean(cyc, steps, steps)
    sh = (ds.step / np.timedelta64(1, "h")).round().astype(int).values
    chis = {}
    for i, h in enumerate(sh):
        if first <= h <= first + 336:
            chis[int(h)] = wc.chi_strip(ds["u"].isel(step=i), ds["v"].isel(step=i))[0].astype("float64")
    trap = np.array([0.5] + [1.0] * 6 + [0.5]) / 7.0
    coef = ref["rain_clim"].values
    for k, lead in enumerate((1, 2)):
        a = t0 + pd.Timedelta(days=7 * (lead - 1)); b = a + pd.Timedelta(days=6)
        days = pd.date_range(a, b, freq="D")
        clim = np.einsum("th,hij->tij", harm_basis(days.dayofyear.values, ref.attrs["rain_nharm"]), coef).mean(0)
        mid = a + pd.Timedelta(days=3.5)
        bias = np.einsum("h,hij->ij", harm_basis([mid.dayofyear], 1)[0], ref["rain_bias"].sel(lead=lead).values)
        ra = wk_rain[k] - clim - bias
        hs = [first + 24 * (7 * (lead - 1) + j) for j in range(8)]
        if not all(h in chis for h in hs):
            print(f"  forecast week {lead}: χ steps missing, skipped", flush=True); continue
        chi_w = sum(trap[j] * chis[h] for j, h in enumerate(hs))
        times = pd.DatetimeIndex([a + pd.Timedelta(days=3.5)])
        ca = chi_anom(chi_w[None], times, clim_chi)[0] - ref["chi_drift"].sel(lead=lead).values
        out.append({"lead": lead, "start": a, "end": b, "rain": ra, "chi": ca})
        print(f"  forecast week {lead} {a:%m-%d}–{b:%m-%d}: rain anomaly sd {np.nanstd(ra):.2f} mm/day", flush=True)
    return init, out


# ── figure ────────────────────────────────────────────────────────────────────
def render(out: Path, rows: list, fc_rows: list, init, lat, lon, cal: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import textwrap
    vp_cmap = LinearSegmentedColormap.from_list("vpot", ["#1b5e20", "#43a047", "#86c98a", "#cfe8cf", "#ffffff", "#fbe2bd", "#f0a64b", "#df6a1e", "#a8330f"])
    vp_lev = np.array([-16, -12, -8, -5, -3, -1.5, 1.5, 3, 5, 8, 12, 16], float)
    rain_cmap = LinearSegmentedColormap.from_list("rain", ["#7a3b0c", "#b8641f", "#e0a55c", "#f3dcb4", "#ffffff", "#c6e6d6", "#6cc3a8", "#1f8f86", "#0b4f6c"])
    rain_lev = np.array([-12, -8, -5, -3, -1.5, -0.5, 0.5, 1.5, 3, 5, 8, 12], float)
    nobs, nfc = len(rows), len(fc_rows)
    nrow = nobs + nfc
    head_in, foot_in, row_in, gap_in = 1.40, 0.80, 1.02, (0.42 if nfc else 0.0)
    H = head_in + foot_in + row_in * nrow + gap_in
    W = 16.0
    fig = plt.figure(figsize=(W, H))
    pc = ccrs.PlateCarree(central_longitude=180)
    vnorm = BoundaryNorm(vp_lev, vp_cmap.N, extend="both"); rnorm = BoundaryNorm(rain_lev, rain_cmap.N, extend="both")
    lef, rig = 0.035, 0.995
    colw = (rig - lef) / 4.0

    def row_axes(r):
        y_top = H - head_in - row_in * r - (gap_in if r >= nobs else 0.0)
        out_ = []
        for c in range(4):
            out_.append(fig.add_axes([lef + c * colw + 0.004, (y_top - row_in + 0.24) / H, colw - 0.008, (row_in - 0.30) / H], projection=pc))
        return out_, y_top

    def draw(ax, field, glat, glon, kind, last_row):
        f = np.concatenate([field, field[:, :1]], 1); gl_ = np.concatenate([glon, [glon[0] + 360]])
        if kind == "rain":
            cf = ax.contourf(gl_, glat, f, levels=rain_lev, cmap=rain_cmap, norm=rnorm, extend="both", transform=ccrs.PlateCarree())
        else:
            cf = ax.contourf(gl_, glat, f / 1e6, levels=vp_lev, cmap=vp_cmap, norm=vnorm, extend="both", transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.45, color="#444"); ax.add_feature(cfeature.LAND, facecolor="#efece6", zorder=0)
        ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, lw=0.25, color="#bbb", xlocs=range(0, 360, 60), ylocs=[-20, 0, 20])
        for y in (-STAT_LAT, STAT_LAT):
            ax.plot([0, 359.99], [y, y], color="#777", lw=0.4, ls=":", transform=ccrs.PlateCarree())
        return cf

    def note(ax, txt, bold=False):
        ax.text(0.0, -0.06, txt, transform=ax.transAxes, fontsize=8.6, va="top", color=INK, fontweight="bold" if bold else "normal")

    def sig_txt(st, key):
        q = cal["null_r_q95"][key]
        return "significant" if st["r"] > q else "not significant"

    cols = ["IMERG Late rain anomaly", "Gill response: χ200 + 850 hPa wind", "Observed χ200 (AIFS-ENS analyses)", "Observed − Gill (residual)"]
    fcols = ["AIFS-ENS rain anomaly (bias removed)", "Gill response to the forecast rain", "AIFS-ENS χ200 (drift removed)", "Forecast − Gill (residual)"]
    cf_r = cf_v = None
    allrows = [(r, False) for r in rows] + [(r, True) for r in fc_rows]
    for i, (rw, is_fc) in enumerate(allrows):
        axs, y_top = row_axes(i)
        last = (i == nrow - 1)
        head = fcols if is_fc else cols
        if i == 0 or (is_fc and i == nobs):
            for c in range(4):
                axs[c].set_title(head[c], fontsize=10, loc="left", fontweight="bold", pad=3)
        cf_r = draw(axs[0], to_gill(rw["rain"]), GM.LAT2, GM.LON2, "rain", last)
        cf_v = draw(axs[1], rw["gill"], lat, lon, "chi", last)
        # 850 hPa wind as 6° × 12° box means (a point sample of a response to 2° heating is mostly grid-scale noise)
        bl, bu, bv = wind_boxes(rw["u850"], rw["v850"])
        big = np.hypot(bu, bv) > 0.3
        q = axs[1].quiver(bl[1][big], bl[0][big], bu[big], bv[big], transform=ccrs.PlateCarree(), scale=QSCALE, scale_units="width",
                          width=0.0032, headwidth=3.6, headlength=4.2, color="#1b365d", alpha=0.9)
        if i == 0:
            axs[1].quiverkey(q, 0.95, 1.10, QKEY, f"{QKEY:g} m/s", labelpos="W", coordinates="axes", fontproperties={"size": 8.4}, labelsep=0.05)
        draw(axs[2], rw["obs"], lat, lon, "chi", last)
        draw(axs[3], rw["obs"] - rw["gill"], lat, lon, "chi", last)
        st = rw["stats"]
        key = "w28" if rw.get("days", 7) > 8 else "w7"
        span = f"{rw['start'].day} {rw['start']:%b} –\n{rw['end'].day} {rw['end']:%b}"
        lab = f"Week {rw['lead']}\n{span}" if is_fc else (f"28 days\n{span}" if rw.get("days", 7) > 8 else span)
        axs[0].text(-0.012, 0.5, lab, transform=axs[0].transAxes, rotation=90, ha="right", va="center", fontsize=8.6, linespacing=1.1, fontweight="bold", color=INK)
        note(axs[0], f"{rw['nrain']} IMERG days" if not is_fc else f"ensemble mean of 51 members, {init:%d %b %H}Z")
        note(axs[1], f"r {st['r']:+.2f} vs {'forecast' if is_fc else 'observed'} ({sig_txt(st, key)}) · explains {100 * max(st['share'], -9.99):.0f}%".replace("-", "−"), bold=True)
        note(axs[2], f"RMS {st['rms_obs'] / 1e6:.1f}" + ("" if is_fc else f" · {rw['nchi']} analyses"))
        note(axs[3], f"RMS {st['rms_res'] / 1e6:.1f}  ({100 * st['rms_res'] ** 2 / max(st['rms_obs'] ** 2, 1e-9):.0f}% of the variance)")
    if nfc:
        yb = (H - head_in - row_in * nobs - gap_in * 0.35) / H
        fig.text(lef, yb, f"FORECAST — AIFS-ENS {init:%d %b %Y %H}Z, ensemble mean", fontsize=10, fontweight="bold", color="#1b365d", va="center")
        fig.add_artist(plt.Line2D([lef, rig], [yb - 0.10 / H, yb - 0.10 / H], transform=fig.transFigure, color="#1b365d", lw=0.8))
    # colour bars
    yb = 0.40 / H
    cax = fig.add_axes([0.06, yb, 0.30, 0.10 / H]); cb = fig.colorbar(cf_r, cax=cax, orientation="horizontal", ticks=rain_lev); cb.ax.tick_params(labelsize=8.4)
    cb.set_label("rain anomaly, mm/day  (1 mm/day ≈ 29 W m⁻² of latent heating)", fontsize=9)
    cax2 = fig.add_axes([0.44, yb, 0.42, 0.10 / H]); cb2 = fig.colorbar(cf_v, cax=cax2, orientation="horizontal", ticks=vp_lev); cb2.ax.tick_params(labelsize=8.4)
    cb2.set_label("χ200 anomaly, 10⁶ m² s⁻¹  (green = upper-level divergence, rising air)", fontsize=9)
    fig.text(rig, 0.06 / H, "IMERG V07 Late · AIFS-ENS · ERA5 · Gill (1980)", fontsize=8, color=MUTED, ha="right")
    fig.text(lef, 1 - 0.14 / H, "What tropical rainfall explains of the upper-level circulation: the Gill (1980) model forced by observed and forecast rain",
             fontsize=15.5, fontweight="bold", ha="left", va="top", color=INK)
    blurb = (f"Heating = latent heat of the rainfall anomaly (1 mm/day ≈ 29 W m⁻²) on the first baroclinic mode, 20°S–20°N. One heating scale and the damping "
             f"(ε = {cal['eps']:g}, a {cal['damp_days']:.1f}-day time scale) were fitted once to ERA5 weekly anomalies 2001–2020 and are not re-tuned: there the Gill χ200 explains "
             f"{100 * cal['share_w7']:.0f}% of the weekly anomaly variance (95% CI {100 * cal['share_w7_ci'][0]:.0f}–{100 * cal['share_w7_ci'][1]:.0f}%, out of sample "
             f"{100 * cal['share_w7_oos']:.0f}%), median pattern r {cal['r_w7_median']:.2f}. Statistics over {STAT_LAT:.0f}°S–{STAT_LAT:.0f}°N (dotted); "
             f"a week's r is called significant when it beats 95% of same-season weeks paired with another year's rain (r > {cal['null_r_q95']['w7']:.2f}; 28-day r > {cal['null_r_q95']['w28']:.2f}). "
             "The residual is what latent heating in the deep tropics does not account for (extratropical forcing, higher vertical modes, the model's simplicity); it is not attributed further.")
    fig.text(lef, 1 - 0.50 / H, "\n".join(textwrap.wrap(blurb, 232)), fontsize=8.6, color=MUTED, va="top", linespacing=1.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white"); plt.close(fig)
    print(f"saved {out}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    import walker_chi as wc
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=""); ap.add_argument("--time", default="")
    ap.add_argument("--history", default=str(ASSETS / "anim" / "walker" / wc.HIST_NAME))
    ap.add_argument("--no-forecast", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for p in (REFNC, CHICLIM):
        if not p.exists():
            raise SystemExit(f"reference missing: {p}")
    ref = xr.open_dataset(REFNC).load(); clim = xr.open_dataset(CHICLIM).load()
    cal = json.loads(ref.attrs["calibration"])
    solve = GM.gill_solver(eps=float(ref.attrs["eps"]))
    alpha, beta = float(ref.attrs["alpha"]), float(ref.attrs["beta"])
    taper = tuple(json.loads(ref.attrs["taper"]))
    # observed χ
    h = wc.load_history(Path(a.history))
    if h is None:
        raise SystemExit("no χ history")
    h = h.sel(latitude=clim.latitude, longitude=clim.longitude, method="nearest")
    lat, lon = clim.latitude.values, clim.longitude.values
    tch = pd.DatetimeIndex(h.time.values)
    an = chi_anom(h.sel(level=200).values.astype("float64"), tch, clim)
    # observed rain up to the last full day both records cover
    last_chi_day = (tch[-1] - pd.Timedelta(hours=12)).normalize().date()   # a day needs its 00Z and 12Z analyses
    today = pd.Timestamp.now("UTC").date()
    rain = update_rain(today - timedelta(days=7 * NWEEKS + 12), min(today - timedelta(days=1), last_chi_day))
    tr = pd.DatetimeIndex(rain.time.values)
    last = min(tr[-1].date(), last_chi_day)
    print(f"  χ history to {tch[-1]:%Y-%m-%d %HZ}, IMERG Late to {tr[-1]:%Y-%m-%d}; windows end {last}", flush=True)
    coef = ref["rain_clim"].values; nh = int(ref.attrs["rain_nharm"])
    rows = []
    spans = [(last - timedelta(days=7 * (k + 1) - 1), last - timedelta(days=7 * k)) for k in range(NWEEKS)][::-1]
    spans.append((last - timedelta(days=7 * NWEEKS - 1), last))
    for s, e in spans:
        rsel = (tr >= pd.Timestamp(s)) & (tr <= pd.Timestamp(e))
        csel = (tch >= pd.Timestamp(s)) & (tch < pd.Timestamp(e) + pd.Timedelta(days=1))
        ndays = (e - s).days + 1
        if rsel.sum() < 0.85 * ndays or csel.sum() < 1.5 * ndays:
            print(f"  window {s}–{e}: rain {int(rsel.sum())} d, χ {int(csel.sum())} analyses — skipped", flush=True); continue
        days = tr[rsel]
        ra = rain.values[rsel].mean(0) - np.einsum("th,hij->tij", harm_basis(days.dayofyear.values, nh), coef).mean(0)
        chi_g, u, v = gill_fields(to_gill(ra), solve, taper)
        obs = an[csel].mean(0)
        g = alpha * chi_g
        st = pattern_stats(obs, g, lat)
        rows.append({"start": s, "end": e, "days": ndays, "rain": ra, "gill": g, "u850": beta * u, "v850": beta * v, "obs": obs,
                     "stats": st, "nrain": int(rsel.sum()), "nchi": int(csel.sum())})
        print(f"  {s:%m-%d}–{e:%m-%d}: Gill vs observed χ200 r {st['r']:+.2f}, explains {100 * st['share']:.0f}%", flush=True)
    if not rows:
        raise SystemExit("no complete observed windows")
    fc_rows, init = [], None
    if not a.no_forecast and a.date and a.time:
        try:
            init, fcs = forecast_fields(a.date, a.time, clim, ref)
            for f in fcs:
                chi_g, u, v = gill_fields(to_gill(f["rain"]), solve, taper)
                g = alpha * chi_g
                st = pattern_stats(f["chi"], g, lat)
                fc_rows.append({"lead": f["lead"], "start": f["start"], "end": f["end"], "days": 7, "rain": f["rain"], "gill": g, "u850": beta * u, "v850": beta * v,
                                "obs": f["chi"], "stats": st, "nrain": 0, "nchi": 0})
                print(f"  forecast week {f['lead']}: Gill vs AIFS χ200 r {st['r']:+.2f}, explains {100 * st['share']:.0f}%", flush=True)
        except Exception as ex:                                               # noqa: BLE001
            print(f"::warning::forecast strip unavailable ({repr(ex)[:120]})", flush=True)
    render(ASSETS / "gill_rain.webp", rows, fc_rows, init, lat, lon, cal)
    q = cal["null_r_q95"]
    rec = lambda r, fc: {"start": str(r["start"])[:10], "end": str(r["end"])[:10], "r": round(r["stats"]["r"], 3), "share": round(r["stats"]["share"], 3),
                         "significant": bool(r["stats"]["r"] > q["w28" if r["days"] > 8 else "w7"]),
                         "rms_observed_1e6": round(r["stats"]["rms_obs"] / 1e6, 3), "rms_residual_1e6": round(r["stats"]["rms_res"] / 1e6, 3),
                         **({"lead_week": r["lead"]} if fc else {"imerg_days": r["nrain"], "analyses": r["nchi"]})}
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "last_analysis": tch[-1].strftime("%Y-%m-%dT%HZ"), "imerg_last_day": tr[-1].strftime("%Y-%m-%d"),
           "forecast_init": init.strftime("%Y-%m-%dT%HZ") if init is not None else None, "stat_band_deg": STAT_LAT,
           "observed": [rec(r, False) for r in rows], "forecast": [rec(r, True) for r in fc_rows], "calibration": cal}
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    (ASSETS / "data" / "gill_rain.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote gill_rain.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
