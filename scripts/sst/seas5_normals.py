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

Inputs (since 2026-10-04, so it runs in Actions): this issue's forecast GRIBs plus derived
tables (seas5_ref.py) — hc_normals_{gl,gl_z500,energy,pme}_{MM} (hindcast mean / tercile bounds /
interannual σ per period) and obs_normals_{var} (ERA5 1991–2025 per-calendar-month statistics,
derived once on the laptop from the ERA5 monthly GRIBs in data/seas5/era5/; never fetched again).
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
from seas5_outlook import ASSETS, DATA, fc_path, hc_path  # noqa: E402
from seas5_build import G0, SEASON_LEADS, TERC_BINS, TERC_PALETTES, _open, head_text, load_field, map_layout, season_label, valid_months  # noqa: E402

ERA5 = DATA / "era5"                       # local ERA5 monthly GRIBs: read by derive_obs_var() on the laptop only
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


# ── ERA5 (DERIVE TIME ONLY: reads the laptop's local files, never runs in Actions) ─
_ERA: dict = {}


def era5_monthly(key: str, short: str):
    """(vals[time, lat, lon] in display units, years, months, lat, lon), cached per (key, short).
    ("am_e", "e") is the derived P − E: ERA5 precipitation plus ERA5 evaporation (negative upward).
    Used only by derive_obs_var() on the laptop: the local ERA5 monthly GRIBs (pulled once in 2026-09,
    kept forever) or the local store; the build reads the obs_normals_* tables instead."""
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


def _references_raw(var: str, month: int) -> dict | None:
    """Per grid point for one calendar month, from the full ERA5 record: means for each reference,
    σ for tercile bounds, the 1993–2016 mean (to align the model) and the observed 1991–2020 sample."""
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
    resid_sd = (sall - (sall.mean(0) + slope * x[:, None, None])).std(0, ddof=2)
    return dict(lat=lat, lon=lon, m9316=s9316.mean(0), sample30=s30, mean30=s30.mean(0), mean10=s10.mean(0),
                sall_mean=sall.mean(0), slope=slope, yall_mean=float(yall.mean()), sd30=s30.std(0, ddof=1), resid_sd=resid_sd,
                span={"obs30": f"{y30.min()}–{y30.max()}", "obs10": f"{y10.min()}–{y10.max()}", "trend": f"{yall.min()}–{yall.max()} fit"})


def derive_obs_var(var: str) -> None:
    """obs_normals_{var}: per calendar month (axis 0 = Jan..Dec) on the ERA5 grid, everything the
    observed references consume. Windows w* are the 3-month seasons STARTING in that month (wrapping
    the year, the same element-wise averaging of the 30-year samples as before)."""
    import seas5_ref as R
    mult = VARS[var][7]
    per = [_references_raw(var, m) for m in range(1, 13)]
    if any(p is None for p in per):
        print(f"  obs normals {var}: ERA5 reference incomplete — not written", flush=True)
        return
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    keys = ["m9316", "mean30", "mean10", "sall_mean", "slope"] + ([] if mult else ["sd30", "resid_sd"])
    arrays = {k: f32([p[k] for p in per]) for k in keys}
    arrays["yall_mean"] = np.array([p["yall_mean"] for p in per])
    win = [np.mean([per[(m + j) % 12]["sample30"] for j in range(3)], axis=0) for m in range(12)]
    if mult:
        q = [np.nanpercentile(p["sample30"], [100 / 3, 200 / 3], axis=0) for p in per]
        arrays["sp33"], arrays["sp67"] = f32([x[0] for x in q]), f32([x[1] for x in q])
        arrays["sstd"] = f32([np.nanstd(p["sample30"], axis=0) for p in per])
        q = [np.nanpercentile(w, [100 / 3, 200 / 3], axis=0) for w in win]
        arrays["wp33"], arrays["wp67"] = f32([x[0] for x in q]), f32([x[1] for x in q])
        arrays["wstd"] = f32([np.nanstd(w, axis=0) for w in win])
    else:
        arrays["wsd"] = f32([w.std(0, ddof=1) for w in win])
    R.save(R.obs_ref(f"normals_{var}"), meta={"span": [p["span"] for p in per], "mult": mult},
           lat=per[0]["lat"].astype(np.float64), lon=per[0]["lon"].astype(np.float64), **arrays)


_OBS: dict = {}


def _obs_table(var: str) -> dict | None:
    if var not in _OBS:
        import seas5_ref as R
        _OBS[var] = R.load(R.obs_ref(f"normals_{var}"))
    return _OBS[var]


def references(var: str, month: int, year: int) -> dict | None:
    """Per grid point for one calendar month (from obs_normals_{var}): means for each reference, σ for
    tercile bounds, the 1993–2016 mean (to align the model), and `mi` (month index) for the window stats."""
    t = _obs_table(var)
    if t is None:
        return None
    i = month - 1
    mult = bool(t["meta"]["mult"])
    trend_val = t["sall_mean"][i].astype(np.float64) + t["slope"][i] * (year - t["yall_mean"][i])
    d = dict(lat=t["lat"], lon=t["lon"], m9316=t["m9316"][i], mi=i, t=t,
             mean={"obs30": t["mean30"][i], "obs10": t["mean10"][i], "trend": trend_val},
             span=t["meta"]["span"][i])
    if not mult:
        d["sd"] = {"obs30": t["sd30"][i], "obs10": t["sd30"][i], "trend": t["resid_sd"][i]}
    return d


# ── model fields ─────────────────────────────────────────────────────────────
# hindcast tables: one per group of variables that share a raw hindcast kind
HC_GROUPS = {"normals_gl": (["t2m", "tp"], ["m:gl"]), "normals_gl_z500": (["z500"], ["m:gl_z500"]),
             "normals_energy": (["si10", "ssrd"], ["m:energy"]), "normals_pme": (["pme"], ["m:water", "m:sfc"])}
VAR_GROUP = {v: g for g, (vs, _) in HC_GROUPS.items() for v in vs}


def _to_units(var: str, a: np.ndarray) -> np.ndarray:
    if var == "tp":
        return a * 86400.0 * 1000
    if var == "ssrd":
        return a / 86400.0
    if var == "t2m":
        return a - 273.15
    return a * VARS[var][5]


def model_fields(ym: str, var: str, hindcast: bool = False):
    """This issue's members (fc[sample, lead, lat, lon], lat, lon) in display units, or None. With
    `hindcast` the same from the RAW hindcast GRIB of start month ym[4:] — derive time only."""
    path = (lambda kind: hc_path(kind, ym[4:])) if hindcast else (lambda kind: fc_path(kind, ym))
    _, kind, mvar, _, _, _, _, _ = VARS[var]
    if var == "pme":
        # evaporation is an Americas-box pull, so P − E uses the Americas precipitation (not the global kind)
        w, s = path("water"), path("sfc")
        if not (w.exists() and s.exists()):
            return None
        e, lat, lon = load_field(w, "e"); tp, _, _ = load_field(s, "tprate")
        return (tp + e) * 86400.0 * 1000, lat, lon                    # e is m/s (rate), negative upward
    f = path(kind)
    if not f.exists():
        return None
    v, lat, lon = load_field(f, mvar)
    return _to_units(var, v), lat, lon


def _periods():
    """(leads) for the six months then the three seasons — the panel order."""
    return [(L,) for L in range(1, 7)] + [tuple(s) for s in SEASON_LEADS]


def derive_hc(group: str, month: str) -> None:
    """hc_{group}_{MM}: per variable and period (6 months + 3 seasons, axis 0) everything the 'hc'
    reference and the observed-space bias correction take from the 600 hindcast samples: hcm (mean),
    lo/hi (33rd/67th percentiles of the samples), sd (σ of the 24 per-year ensemble means). z500 is
    detrended: ymean/slope of the per-year means, rlo/rhi the percentiles of the residuals (the trend
    line is added back at the valid year at build time), sd the residual σ."""
    import seas5_ref as R
    arrays = {}
    for var in HC_GROUPS[group][0]:
        mf = model_fields("2000" + month, var, hindcast=True)
        if mf is None:
            raise FileNotFoundError(f"raw hindcast for {var} {month} not on disk")
        hc, lat, lon = mf
        hc_mean = np.nanmean(hc, axis=0)
        ny = 24; yr = np.arange(ny) - (ny - 1) / 2                    # samples are member-major, year fastest
        acc = {k: [] for k in (("hcm", "ymean", "slope", "rlo", "rhi", "sd") if var == "z500" else ("hcm", "lo", "hi", "sd"))}
        for leads in _periods():
            idx = [L - 1 for L in leads]
            hsub = hc[:, idx[0]] if len(idx) == 1 else hc[:, idx].mean(1)
            hcm = hc_mean[idx[0]] if len(idx) == 1 else hc_mean[idx].mean(0)
            acc["hcm"].append(hcm)
            if var == "z500":
                ym_ = np.nanmean(hsub.reshape(-1, ny, *hsub.shape[1:]), axis=0)
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr ** 2).sum()
                resid = hsub - (hcm + slope[None] * np.tile(yr, hsub.shape[0] // ny)[:, None, None])
                rlo, rhi = np.nanpercentile(resid, [100 / 3, 200 / 3], axis=0)
                acc["ymean"].append(ym_.mean(0)); acc["slope"].append(slope); acc["rlo"].append(rlo); acc["rhi"].append(rhi)
                acc["sd"].append(np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0))
            else:
                lo, hi = np.nanpercentile(hsub, [100 / 3, 200 / 3], axis=0)
                acc["lo"].append(lo); acc["hi"].append(hi)
                acc["sd"].append(np.nanstd(np.nanmean(hsub.reshape(-1, ny, *hsub.shape[1:]), axis=0), axis=0))
        for k, v in acc.items():
            arrays[f"{var}_{k}"] = np.asarray(v, dtype=np.float32)
        arrays["lat"], arrays["lon"] = lat, lon
        del hc
    R.save(R.hc_ref(group, month), meta={"vars": HC_GROUPS[group][0], "periods": [list(p) for p in _periods()]}, **arrays)


REF_UNITS = {g: dict(raw=raws, fn=(lambda month, g=g: derive_hc(g, month))) for g, (_, raws) in HC_GROUPS.items()}
OBS_UNITS = {f"normals_{v}": (lambda v=v: derive_obs_var(v)) for v in VARS}


def _regrid_to(src, slat, slon, lat, lon, reach: float = 1.1):
    """Nearest-neighbour selection of a [lat, lon] (or [..., lat, lon]) ERA5 field onto the model
    grid. Target points farther than `reach` degrees from any source point are NaN — the store's
    precipitation covers 0–90°N only, and without this the equator row would be smeared over
    South America."""
    dlon = lambda a, b: np.abs((a - b + 180.0) % 360.0 - 180.0)          # circular: the 180° column must not fall in a gap
    ilat = np.array([int(np.argmin(np.abs(slat - v))) for v in lat]); ilon = np.array([int(np.argmin(dlon(slon, v))) for v in lon])
    out = src[..., ilat[:, None], ilon[None, :]].astype(np.float64, copy=True)
    far_lat = np.abs(slat[ilat] - lat) > reach; far_lon = dlon(slon[ilon], lon) > reach
    out[..., far_lat, :] = np.nan; out[..., :, far_lon] = np.nan
    return out


# ── products ─────────────────────────────────────────────────────────────────
def panels_for(ym: str, var: str, ref: str):
    """→ list of dict(title, anom[lat,lon], below, above, std) for 6 months + 3 seasons."""
    import seas5_ref as R
    mf = model_fields(ym, var)
    if mf is None:
        return None, None, None
    fc, lat, lon = mf
    H = R.need(R.hc_ref(VAR_GROUP[var], ym[4:]), f"SEAS5 hindcast statistics for {var}")
    mult = VARS[var][7]
    vm = valid_months(ym)
    out = []

    def one(members, p, refs_list, title, valid_year):
        """members [sample, lat, lon] model values; p the period index into the hindcast table;
        refs_list: per-month reference dicts (1 or 3)."""
        hcm = H[f"{var}_hcm"][p]
        if ref == "hc":
            if var == "z500":
                # heights carry the warming trend: the hindcast reference is its linear trend at the
                # valid year (per grid point), and the spread is the residual spread — same as the caps
                ny = 24
                target = (valid_year - 1993) - (ny - 1) / 2
                hcm_ref = H["z500_ymean"][p] + H["z500_slope"][p] * target
                lo, hi = H["z500_rlo"][p] + hcm_ref, H["z500_rhi"][p] + hcm_ref
                a = np.nanmean(members, 0) - hcm_ref
                below = (members < lo[None]).mean(0); above = (members > hi[None]).mean(0)
                sd = H["z500_sd"][p]
                return dict(title=title, anom=a, below=below, above=above, std=a / np.where(sd > 0, sd, np.nan))
            a = np.nanmean(members, 0) - hcm
            lo, hi = H[f"{var}_lo"][p], H[f"{var}_hi"][p]
            anom = (np.nanmean(members, 0) / hcm * 100.0) if mult else a
            below = (members < lo[None]).mean(0); above = (members > hi[None]).mean(0)
            sd = H[f"{var}_sd"][p]                                     # interannual spread, not member noise
            return dict(title=title, anom=anom, below=below, above=above, std=a / np.where(sd > 0, sd, np.nan))
        # observed space: average the per-month references over the season
        rg = lambda a, r: _regrid_to(a, r["lat"], r["lon"], lat, lon)
        r0 = refs_list[0]; T = r0["t"]; season = len(refs_list) > 1
        m9316 = np.mean([rg(r["m9316"], r) for r in refs_list], axis=0)
        rmean = np.mean([rg(r["mean"][ref], r) for r in refs_list], axis=0)
        if mult:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = members * (m9316 / np.where(hcm > 1e-6, hcm, np.nan))[None]
            smean = np.mean([rg(T["mean30"][r["mi"]], r) for r in refs_list], axis=0)     # mean of the 30-year sample
            scale = rmean / np.where(smean > 1e-6, smean, np.nan)
            p33, p67, ssd = (T["wp33"], T["wp67"], T["wstd"]) if season else (T["sp33"], T["sp67"], T["sstd"])
            lo, hi = rg(p33[r0["mi"]], r0) * scale, rg(p67[r0["mi"]], r0) * scale
            anom = np.nanmean(corr, 0) / rmean * 100.0
        else:
            corr = members - hcm[None] + m9316[None]
            # seasonal σ from the seasonal-mean series, not the monthly one
            sd = rg(T["wsd"][r0["mi"]], r0) if season else rg(r0["sd"][ref], r0)
            lo, hi = rmean - Z_TERC * sd, rmean + Z_TERC * sd
            anom = np.nanmean(corr, 0) - rmean
        below = (corr < lo[None]).mean(0); above = (corr > hi[None]).mean(0)
        if mult:
            samp_sd = rg(ssd[r0["mi"]], r0)
            std = (np.nanmean(corr, 0) - rmean) / np.where(samp_sd > 0, samp_sd, np.nan)
        else:
            std = (np.nanmean(corr, 0) - rmean) / np.where(sd > 0, sd, np.nan)
        return dict(title=title, anom=anom, below=below, above=above, std=std)

    refs_cache = {}
    panels_for.last_span = None
    for p, leads in enumerate(_periods()):
        idx = [L - 1 for L in leads]
        months = [(int(vm[i][:4]), int(vm[i][5:])) for i in idx]
        valid_year = months[len(months) // 2][0]
        rl = []
        if ref != "hc":
            # the trend reference is evaluated at each month's own year (seasons straddling New Year included)
            rl = [refs_cache.setdefault((y, m), references(var, m, y)) for y, m in months]
            if any(r is None for r in rl):
                return None, None, None
            if panels_for.last_span is None:
                panels_for.last_span = rl[0]["span"][ref]
        members = fc[:, idx[0]] if len(idx) == 1 else fc[:, idx].mean(1)
        if len(idx) == 1:
            y, m = months[0]
            title, key = f"{calendar.month_abbr[m]} {y}", f"{y}_{m:02d}"
        else:
            y0s, y1s = vm[idx[0]][:4], vm[idx[-1]][:4]
            title = f"{season_label(ym, leads)} {y0s if y0s == y1s else y0s + '–' + y1s[2:]}"
            key = f"{season_label(ym, leads)}_{y0s}"
        pnl = one(members, p, rl, title, valid_year); pnl["key"] = key; out.append(pnl)
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
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#6f6b64", zorder=3)
        gl_ = ax.gridlines(draw_labels=True, linewidth=0.3, color="#6f6b64", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
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
            try:
                panels, lat, lon = panels_for(ym, var, ref)
            except Exception as e:                                    # noqa: BLE001 — one variable must not sink the rest
                print(f"  {var} vs {ref}: FAILED ({str(e)[:120]})", flush=True); continue
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
