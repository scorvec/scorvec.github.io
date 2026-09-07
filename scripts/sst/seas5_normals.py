#!/usr/bin/env python3
"""SEAS5 anomaly and tercile maps against observed normals, monthly and seasonal.

The page's default terciles are relative to SEAS5's own 1993–2016 hindcast, which
removes bias but answers "warmer than the model's 1993–2016" — a low bar in a
warming climate. This module puts the same members against four references:

  hc      the model's hindcast climatology (the default; bias-free by construction)
  obs30   ERA5 1991–2020, the WMO standard normal
  obs10   ERA5 2016–2025, the last decade
  trend   ERA5 1991–2025 linear trend extrapolated to the forecast month's year:
          what "normal" is expected to be NOW, so a warm anomaly here is warm even
          for today's climate

For the observed references the members are first moved into observed space with
a mean bias correction per grid point and month: member − hindcast mean + ERA5
1993–2016 mean (precipitation and solar radiation multiplicatively). Anomalies are
ensemble mean minus the reference mean (precipitation as % of the reference).
Terciles: for temperature and heights the boundaries are reference mean ± 0.4307 σ
where σ is the ERA5 1991–2020 standard deviation (detrended when the reference is
the trend); for precipitation and radiation the empirical 33rd/67th percentiles of
the 30 observed years, scaled to the reference mean. Probabilities are member
fractions. Monthly for the six lead months and seasonal for the three overlapping
seasons, one 3 × 3 figure per (variable, reference, anomaly | tercile).

Needs the ERA5 monthly means from seas5_era5.py (Americas, 1°, 1991–2025).
Output: assets/sst/seas5_norm_{var}_{ref}_{anom|std|terc}_{period}.webp (one map per period) + data/seas5_normals.json.
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, fc_path, hc_path  # noqa: E402
from seas5_build import G0, SEASON_LEADS, TERC_BINS, TERC_PALETTES, _open, head_text, load_field, map_layout, season_label, valid_months  # noqa: E402

ERA5 = Path(__file__).resolve().parent / "data" / "seas5" / "era5"
OUT_JSON = ASSETS / "data" / "seas5_normals.json"
Z_TERC = 0.4307                                                       # ±0.4307 σ bounds the middle third of a normal

VARS = {
    # key: (label, seas5 kind, seas5 var, era5 file, era5 shortName, factor to display units, units, multiplicative)
    # t2m / tp / z500 draw on the global 1° kinds (gl, gl_z500) when present, the Americas kinds otherwise
    # (user 2026-09-07: "better to just show global"); their ERA5 references come from the global monthly
    # files era5_gl_t / era5_gl_z (seas5_era5.py) or, until those land, the local store.
    "t2m": ("2 m temperature", "gl", "t2m", "gl_t", "t2m", 1.0, "°C", False),
    "tp": ("Precipitation", "gl", "tprate", "gl_t", "tp", 1.0, "mm/day", True),
    "z500": ("500 hPa height", "gl_z500", "z", "gl_z", "z", 1.0 / G0, "m", False),
    "si10": ("10 m wind speed", "energy", "si10", "am_sfc", "si10", 1.0, "m/s", False),
    "ssrd": ("Surface solar radiation", "energy", "ssrd", "am_sfc", "ssrd", 1.0, "W/m²", True),
    # derived: precipitation minus evaporation, the surface water balance (mm/day; negative = net drying)
    "pme": ("P − E water balance", "water", "e", "am_e", "e", 1.0, "mm/day", False),
}
REFS = {"hc": "SEAS5 hindcast 1993–2016", "obs30": "ERA5 1991–2020 normal", "obs10": "ERA5 2016–2025 normal", "trend": "ERA5 trend extrapolated to the forecast year"}
LEVELS = {"pme": [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3], "t2m": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4], "z500": [-60, -45, -30, -20, -10, -5, 5, 10, 20, 30, 45, 60],
          "si10": [-1.5, -1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1, 1.5], "tp": [40, 55, 70, 85, 95, 105, 120, 145, 180, 250], "ssrd": [80, 86, 92, 96, 98, 102, 104, 108, 114, 120]}
STD_LEVELS = [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3]   # standardised anomaly, σ of the reference's interannual spread
CMAPS = {"pme": "BrBG", "t2m": "RdBu_r", "z500": "RdBu_r", "si10": "PuOr_r", "tp": "BrBG", "ssrd": "RdYlBu_r"}
LAND_ONLY = {"t2m", "tp", "si10", "ssrd", "pme"}


# ── ERA5 ─────────────────────────────────────────────────────────────────────
_ERA: dict = {}


def era5_monthly(key: str, short: str):
    """(vals[time, lat, lon] in display units, years, months, lat, lon), cached per (key, short).
    ("am_e", "e") is the derived P − E: ERA5 precipitation plus ERA5 evaporation (negative upward)."""
    if (key, short) in _ERA:
        return _ERA[(key, short)]
    if key == "am_e" and short == "e":
        tp, ev = era5_monthly("am_sfc", "tp"), era5_monthly("am_e", "e_raw")
        if tp is None or ev is None:
            return None
        _ERA[(key, short)] = (tp[0] + ev[0], tp[1], tp[2], tp[3], tp[4])
        return _ERA[(key, short)]
    gshort = "e" if short == "e_raw" else short
    p = ERA5 / f"era5_{key}_1991-2025.grib"
    if not p.exists():
        # LOCAL STORE FIRST: ~/era5_store holds daily t2m (global), z500 (global to 2020) and
        # NH precipitation; the CDS file is only the fallback for what the store lacks.
        local = {"t2m": "t2m", "z": "z500", "tp": "tp"}.get(gshort)
        if local is None:
            return None
        import era5_local
        m = era5_local.monthly(local)
        if m is None:
            return None
        m = era5_local.to_lon180(m)
        t = m.time.values
        yrs = np.array([int(str(x)[:4]) for x in t]); mos = np.array([int(str(x)[5:7]) for x in t])
        v = m.values.astype(np.float64)
        if gshort == "t2m":
            v = v - 273.15
        _ERA[(key, short)] = (v, yrs, mos, m.latitude.values, m.longitude.values)
        return _ERA[(key, short)]
    ds = _open(p, shortName={"t2m": "2t", "si10": "10si"}.get(gshort, gshort))   # GRIB names differ from the netCDF ones
    da = ds[list(ds.data_vars)[0]].transpose("time", "latitude", "longitude")
    t = da.time.values
    yrs = np.array([int(str(x)[:4]) for x in t]); mos = np.array([int(str(x)[5:7]) for x in t])
    v = da.values.astype(np.float64)
    if gshort in ("tp", "e"):
        v = v * 1000.0                                                # m/day → mm/day (evaporation negative upward)
    elif gshort == "z":
        v = v / G0
    elif gshort == "ssrd":
        v = v / 86400.0                                               # J/m² per day → W/m²
    elif gshort == "t2m":
        v = v - 273.15
    _ERA[(key, short)] = (v, yrs, mos, da.latitude.values, da.longitude.values)
    ds.close()
    return _ERA[(key, short)]


def _sel_month(v, yrs, mos, m, y0, y1):
    sel = (mos == m) & (yrs >= y0) & (yrs <= y1)
    return v[sel], yrs[sel]


def references(var: str, month: int, year: int) -> dict | None:
    """Per grid point for one calendar month: means for each reference, σ for tercile bounds,
    the 1993–2016 mean (to align the model) and the observed 1991–2020 sample (for precip terciles)."""
    _, _, _, ekey, eshort, _, _, mult = VARS[var]
    r = era5_monthly(ekey, eshort)
    if r is None:
        return None
    v, yrs, mos, lat, lon = r
    s30, y30 = _sel_month(v, yrs, mos, month, 1991, 2020)
    s10, y10 = _sel_month(v, yrs, mos, month, 2016, 2025)
    s9316, _ = _sel_month(v, yrs, mos, month, 1993, 2016)
    sall, yall = _sel_month(v, yrs, mos, month, 1991, 2025)
    if len(s30) < 25 or len(s10) < 4:
        return None
    # NaN-safe: a store year with missing months would otherwise poison the means
    s30 = s30[np.isfinite(s30).all(axis=(1, 2))]; s10 = s10[np.isfinite(s10).all(axis=(1, 2))]
    ok = np.isfinite(sall).all(axis=(1, 2)); sall, yall = sall[ok], yall[ok]
    x = yall - yall.mean()
    slope = (x[:, None, None] * (sall - sall.mean(0))).sum(0) / (x ** 2).sum()
    trend_val = sall.mean(0) + slope * (year - yall.mean())
    resid_sd = (sall - (sall.mean(0) + slope * x[:, None, None])).std(0, ddof=2)
    return dict(lat=lat, lon=lon, m9316=s9316.mean(0), sample30=s30,
                mean={"obs30": s30.mean(0), "obs10": s10.mean(0), "trend": trend_val},
                sd={"obs30": s30.std(0, ddof=1), "obs10": s30.std(0, ddof=1), "trend": resid_sd},
                span={"obs30": f"{y30.min()}–{y30.max()}", "obs10": f"{y10.min()}–{y10.max()}", "trend": f"{yall.min()}–{yall.max()} fit"})


# ── model fields ─────────────────────────────────────────────────────────────
def model_fields(ym: str, var: str):
    label, kind, mvar, _, _, fac, _, _ = VARS[var]
    if var == "pme":
        tp = model_fields(ym, "tp")
        f, h = fc_path("water", ym), hc_path("water", ym[4:])
        if tp is None or not (f.exists() and h.exists()):
            return None
        fe, lat, lon = load_field(f, "e"); he, _, _ = load_field(h, "e")
        return tp[0] + fe * 86400.0 * 1000, tp[1] + he * 86400.0 * 1000, lat, lon   # e is m/s (rate), negative upward
    f, h = fc_path(kind, ym), hc_path(kind, ym[4:])
    if not (f.exists() and h.exists()):
        alt = {"gl": "sfc", "gl_z500": "z500"}.get(kind)              # Americas fallback for the global kinds
        if alt is None:
            return None
        f, h = fc_path(alt, ym), hc_path(alt, ym[4:])
        if not (f.exists() and h.exists()):
            return None
    fc, lat, lon = load_field(f, mvar); hc, _, _ = load_field(h, mvar)
    if var == "tp":
        fc, hc = fc * 86400.0 * 1000, hc * 86400.0 * 1000
    elif var == "ssrd":
        fc, hc = fc / 86400.0, hc / 86400.0
    elif var == "t2m":
        fc, hc = fc - 273.15, hc - 273.15
    else:
        fc, hc = fc * fac, hc * fac
    return fc, hc, lat, lon


def _regrid_to(src, slat, slon, lat, lon, reach: float = 1.1):
    """Nearest-neighbour selection of a [lat, lon] (or [..., lat, lon]) ERA5 field onto the model
    grid. Target points farther than `reach` degrees from any source point are NaN — the store's
    precipitation covers 0–90°N only, and without this the equator row would be smeared over
    South America."""
    ilat = np.array([int(np.argmin(np.abs(slat - v))) for v in lat]); ilon = np.array([int(np.argmin(np.abs(slon - v))) for v in lon])
    out = src[..., ilat[:, None], ilon[None, :]].astype(np.float64, copy=True)
    far_lat = np.abs(slat[ilat] - lat) > reach; far_lon = np.abs(slon[ilon] - lon) > reach
    out[..., far_lat, :] = np.nan; out[..., :, far_lon] = np.nan
    return out


# ── products ─────────────────────────────────────────────────────────────────
def panels_for(ym: str, var: str, ref: str):
    """→ list of dict(title, anom[lat,lon], below, above, ens_anom_units) for 6 months + 3 seasons."""
    mf = model_fields(ym, var)
    if mf is None:
        return None, None, None
    fc, hc, lat, lon = mf
    hc_mean = np.nanmean(hc, axis=0)                                  # [lead, lat, lon]
    mult = VARS[var][7]
    vm = valid_months(ym)
    out = []

    valid_year = int(vm[0][:4])

    def one(members, hcm, refs_list, title):
        nonlocal valid_year
        """members [sample, lat, lon] model values; hcm [lat, lon] hindcast mean; refs_list: per-month reference dicts (1 or 3)."""
        if ref == "hc":
            hsub = hcs
            ny = 24
            yr = np.arange(ny) - (ny - 1) / 2                      # samples are member-major, year fastest
            if var == "z500":
                # heights carry the warming trend: the hindcast reference is its linear trend at the
                # valid year (per grid point), and the spread is the residual spread — same as the caps
                hy = hsub.reshape(-1, ny, *hsub.shape[1:])         # [member, year, lat, lon]
                ym_ = np.nanmean(hy, axis=0)                         # per-year ensemble mean
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr ** 2).sum()
                target = (valid_year - 1993) - (ny - 1) / 2
                hcm_ref = ym_.mean(0) + slope * target
                resid = hsub - (hcm + slope[None] * np.tile(yr, hsub.shape[0] // ny)[:, None, None])
                lo, hi = np.nanpercentile(resid + hcm_ref[None], [100 / 3, 200 / 3], axis=0)
                a = np.nanmean(members, 0) - hcm_ref
                below = (members < lo[None]).mean(0); above = (members > hi[None]).mean(0)
                sd = np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0)
                return dict(title=title, anom=a, below=below, above=above, std=a / np.where(sd > 0, sd, np.nan))
            a = np.nanmean(members, 0) - hcm
            lo, hi = np.nanpercentile(hsub, [100 / 3, 200 / 3], axis=0)
            anom = (np.nanmean(members, 0) / hcm * 100.0) if mult else a
            below = (members < lo[None]).mean(0); above = (members > hi[None]).mean(0)
            yr_means = np.nanmean(hsub.reshape(-1, ny, *hsub.shape[1:]), axis=0)   # interannual spread, not member noise
            sd = np.nanstd(yr_means, axis=0)
            return dict(title=title, anom=anom, below=below, above=above, std=a / np.where(sd > 0, sd, np.nan))
        # observed space: average the per-month references over the season
        m9316 = np.mean([_regrid_to(r["m9316"], r["lat"], r["lon"], lat, lon) for r in refs_list], axis=0)
        rmean = np.mean([_regrid_to(r["mean"][ref], r["lat"], r["lon"], lat, lon) for r in refs_list], axis=0)
        if mult:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = members * (m9316 / np.where(hcm > 1e-6, hcm, np.nan))[None]
            samp = np.mean([_regrid_to(r["sample30"], r["lat"], r["lon"], lat, lon) for r in refs_list], axis=0)   # [30, lat, lon]
            scale = rmean / np.where(np.nanmean(samp, 0) > 1e-6, np.nanmean(samp, 0), np.nan)
            lo, hi = np.nanpercentile(samp, [100 / 3, 200 / 3], axis=0) * scale
            anom = np.nanmean(corr, 0) / rmean * 100.0
        else:
            corr = members - hcm[None] + m9316[None]
            sd = np.sqrt(np.mean([_regrid_to(r["sd"][ref], r["lat"], r["lon"], lat, lon) ** 2 for r in refs_list], axis=0)) / np.sqrt(len(refs_list) if len(refs_list) > 1 else 1)
            if len(refs_list) > 1:                                     # seasonal σ from the seasonal-mean series, not the monthly one
                samp = np.mean([_regrid_to(r["sample30"], r["lat"], r["lon"], lat, lon) for r in refs_list], axis=0)
                sd = samp.std(0, ddof=1)
            lo, hi = rmean - Z_TERC * sd, rmean + Z_TERC * sd
            anom = np.nanmean(corr, 0) - rmean
        below = (corr < lo[None]).mean(0); above = (corr > hi[None]).mean(0)
        if mult:
            samp_sd = np.nanstd(samp, axis=0)
            std = (np.nanmean(corr, 0) - rmean) / np.where(samp_sd > 0, samp_sd, np.nan)
        else:
            std = (np.nanmean(corr, 0) - rmean) / np.where(sd > 0, sd, np.nan)
        return dict(title=title, anom=anom, below=below, above=above, std=std)

    refs_cache = {}
    panels_for.last_span = None
    for L, v in enumerate(vm):
        y, m = int(v[:4]), int(v[5:])
        hcs = hc[:, L]; valid_year = y
        rl = [] if ref == "hc" else [refs_cache.setdefault((y, m), references(var, m, y))]
        if ref != "hc" and rl[0] is None:
            return None, None, None
        if ref != "hc" and panels_for.last_span is None:
            panels_for.last_span = rl[0]["span"][ref]
        pnl = one(fc[:, L], hc_mean[L], rl, f"{calendar.month_abbr[m]} {y}"); pnl["key"] = f"{y}_{m:02d}"; out.append(pnl)
    for leads in SEASON_LEADS:
        idx = [Lx - 1 for Lx in leads]
        hcs = hc[:, idx].mean(1); valid_year = int(vm[idx[1]][:4])
        rl = [] if ref == "hc" else [refs_cache[(int(vm[i][:4]), int(vm[i][5:]))] for i in idx]
        y0s, y1s = vm[idx[0]][:4], vm[idx[-1]][:4]
        pnl = one(fc[:, idx].mean(1), hc_mean[idx].mean(0), rl, f"{season_label(ym, leads)} {y0s if y0s == y1s else y0s + '–' + y1s[2:]}")
        pnl["key"] = f"{season_label(ym, leads)}_{y0s}"; out.append(pnl)
    return out, lat, lon


def render(ym: str, var: str, ref: str, kind: str, panels, lat, lon, out_dir: Path) -> dict:
    """One image per period (single map, user 2026-09-07): global plate carrée for fields on a global
    grid (cut at 40 °E), the Americas box otherwise. → {period_key: file}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    import textwrap

    label, _, _, _, _, _, units, mult = VARS[var]
    glob = (lon.max() - lon.min()) > 300
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=-140.0 if glob else -90.0)
    bins = TERC_BINS
    warm, cool = TERC_PALETTES["warm"], TERC_PALETTES["cool"]
    if var in ("tp", "pme"):
        warm, cool = TERC_PALETTES["wet"], TERC_PALETTES["dry"]
    if var == "ssrd":
        warm, cool = TERC_PALETTES["sunny"], TERC_PALETTES["dull"]
    y0, m0 = ym[:4], int(ym[4:])
    what = {"anom": "anomaly", "terc": "most likely tercile", "std": "standardised anomaly"}[kind]
    if ref == "hc":
        sub = "Reference: the model's own 1993–2016 hindcast at the same lead (bias-free by construction)."
    else:
        span = getattr(panels_for, "last_span", None)
        sub = ("Members moved into observed space with a per-point mean bias correction (member − hindcast mean + ERA5 1993–2016 mean"
               + (", multiplicatively" if mult else "") + "), then compared with " + REFS[ref] + (f" (ERA5 years used: {span})" if span else "") + ".")
    if kind == "terc":
        sub += "  White: no category reaches 40 %; near-normal not drawn."
    if kind == "std":
        sub += "  Ensemble-mean anomaly divided by the reference's year-to-year standard deviation (±1σ is a typical year's swing)."
    sub = "\n".join(textwrap.wrap(sub, 190))
    order = np.argsort(lon); lon_s = lon[order]
    lat_a = lat; flip = lat[0] > lat[-1]
    if flip: lat_a = lat[::-1]
    if glob:
        lat0, lat1 = -60, 85; lon_span = 360.0
    else:
        lat0, lat1, lon_span = -60, 75, 140.0
    W = 14.0; map_h = W * (lat1 - lat0) / lon_span; top, bot = 1.0, (0.95 if kind != "terc" else 1.05); H = map_h + top + bot
    files = {}
    for pnl in panels:
        def prep(a):
            a = a[:, order]
            if flip: a = a[::-1]
            if glob:
                a, lo = add_cyclic_point(a, coord=lon_s); return a, lo
            return a, lon_s
        fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=proj)
        if glob: ax.set_extent([-180, 180, lat0, lat1], crs=proj)
        else: ax.set_extent([-170, -30, lat0, lat1], crs=pc)
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
        if kind in ("anom", "std"):
            lev = LEVELS[var] if kind == "anom" else STD_LEVELS
            a, lo = prep(pnl["anom"] if kind == "anom" else pnl["std"])
            mesh = ax.pcolormesh(lo, lat_a, a, cmap=plt.get_cmap(CMAPS[var], len(lev) - 1), norm=BoundaryNorm(lev, len(lev) - 1), transform=pc, shading="auto", zorder=1)
        else:
            normal = 1.0 - pnl["above"] - pnl["below"]
            for arr, other, cols in ((pnl["above"], pnl["below"], warm), (pnl["below"], pnl["above"], cool)):
                show = np.where((arr >= 0.40) & (arr >= np.maximum(other, normal)), arr, np.nan)
                a, lo = prep(show)
                mesh = ax.pcolormesh(lo, lat_a, a, cmap=ListedColormap(cols), norm=BoundaryNorm(bins, len(cols)), transform=pc, shading="auto", zorder=1)
        if var in LAND_ONLY:
            ax.add_feature(cfeature.OCEAN, facecolor="#fff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#fff", zorder=2)
        ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#999", zorder=3)
        gl_ = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
        gl_.top_labels = gl_.right_labels = False; gl_.xlabel_style = gl_.ylabel_style = {"size": 7, "color": "#555"}
        fig.text(0.03, 1 - 0.14 / H, f"SEAS5 {label}: {what} vs {REFS[ref]} · {pnl['title']} · {calendar.month_name[m0]} {y0} issue", fontsize=13, fontweight="bold", va="top")
        fig.text(0.03, 1 - 0.50 / H, sub, fontsize=8.4, color="#444", va="top")
        if kind in ("anom", "std"):
            cax = fig.add_axes([0.3, 0.42 / H, 0.4, 0.14 / H]); cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
            cb.set_label(("standardised anomaly (σ)" if kind == "std" else "% of reference" if mult else f"ensemble-mean anomaly ({units})"), fontsize=8.5); cb.ax.tick_params(labelsize=7)
        else:
            h1 = [Patch(color=c, label=f"{int(bins[i]*100)}–{int(min(bins[i+1],1)*100)}%") for i, c in enumerate(warm)]
            h2 = [Patch(color=c, label=f"{int(bins[i]*100)}–{int(min(bins[i+1],1)*100)}%") for i, c in enumerate(cool)]
            l1 = fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.04, 0.004), ncol=6, frameon=False, title="Above normal most likely", fontsize=8, title_fontsize=8.5)
            fig.add_artist(l1)
            fig.legend(handles=h2, loc="lower right", bbox_to_anchor=(0.96, 0.004), ncol=6, frameon=False, title="Below normal most likely", fontsize=8, title_fontsize=8.5)
        out = out_dir / f"seas5_norm_{var}_{ref}_{kind}_{pnl['key']}.webp"
        fig.savefig(out, dpi=110, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
        files[pnl["key"]] = out.name
    return files


def build(ym: str, only_vars=None, only_refs=None) -> None:
    t0 = time.time()
    ASSETS.mkdir(parents=True, exist_ok=True)
    man = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "refs": REFS,
           "vars": {k: dict(label=v[0], units=v[6]) for k, v in VARS.items()}, "figures": {}, "periods": [], "extent": {}}
    for var in (only_vars or VARS):
        for ref in (only_refs or REFS):
            panels, lat, lon = panels_for(ym, var, ref)
            if panels is None:
                print(f"  {var} vs {ref}: fields not on disk — skipped", flush=True); continue
            if not man["periods"]:
                man["periods"] = [dict(key=p["key"], label=p["title"], kind="season" if "_" in p["key"] and not p["key"][0].isdigit() else "month") for p in panels]
            man["extent"][var] = "global" if (lon.max() - lon.min()) > 300 else "americas"
            for kind in ("anom", "std", "terc"):
                files = render(ym, var, ref, kind, panels, lat, lon, ASSETS)
                man["figures"][f"{var}|{ref}|{kind}"] = files
                print(f"  wrote {len(files)} maps: {var} {ref} {kind}", flush=True)
    if OUT_JSON.exists():                                            # keep figure sets from a partial earlier run of the same issue
        old = json.loads(OUT_JSON.read_text())
        if old.get("issue") == ym and isinstance(next(iter(old.get("figures", {}).values()), None), dict):
            old["figures"].update(man["figures"]); man["figures"] = old["figures"]
            man["extent"] = {**old.get("extent", {}), **man["extent"]}
    OUT_JSON.write_text(json.dumps(man, separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({len(man['figures'])} figure sets) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--vars", nargs="*"); ap.add_argument("--refs", nargs="*")
    a = ap.parse_args()
    build(a.issue, a.vars, a.refs)
