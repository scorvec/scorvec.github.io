#!/usr/bin/env python3
"""Daily global climate monitor: ERA5 anomalies, loops, regional series,
monthly means — plus the IMERG daily precip anomaly.

ERA5 (anonymous ARCO-ERA5 hourly zarr, ~5-day lag): daily means are the
average of the 00/06/12/18Z synoptic steps — exactly the sampling of the
WeatherBench2 climatology (1991–2020, harmonic, built by build_era5_clim.py),
and 6× cheaper than 24 hourly reads.

Products (assets/climate/):
  t2m_anom.webp / z500_anom.webp / mslp_anom.webp     latest-day maps
  anim/{t2m,z500,mslp}/YYYYMMDD.webp + anim/manifest.json   30-day loops
  t2m_regions_daily.csv    daily regional anomaly series (appended each run;
                           backfilled 1979→ by build_t2m_regions.py)
  monthly/{var}_{YYYYMM}.webp   recent complete months (CDS monthly means;
                           skipped quietly when no CDS credentials — the
                           rendered webps are committed, so CI never needs CDS)
  precip_anom.webp         IMERG daily anomaly (needs imerg_clim_global.nc)
  manifest.json            {era5_day, imerg_day, months, generated}

    python scripts/climate/climate_monitor.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets" / "climate"
ANIM = ASSETS / "anim"
SERIES_CSV = ASSETS / "t2m_regions_daily.csv"
TELE_CSV = ASSETS / "teleconnections_daily.csv"
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "sst"))

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
PC = ccrs.PlateCarree()
SYNOPTIC_H = [0, 6, 12, 18]              # matches the WB2 6-hourly climatology
LOOP_DAYS = 30
MONTHLY_KEEP = 4                          # recent complete months to show

ERA5_VARS = {
    "t2m":  dict(src="2m_temperature",          level=None, scale=1.0,          offset=-273.15,
                 title="2 m temperature anomaly", cbar="°C",  vmax=12,
                 cds_var="2m_temperature", cds_ds="reanalysis-era5-single-levels-monthly-means"),
    "mslp": dict(src="mean_sea_level_pressure", level=None, scale=0.01,         offset=0.0,
                 title="mean sea-level pressure anomaly", cbar="hPa", vmax=24,
                 cds_var="mean_sea_level_pressure", cds_ds="reanalysis-era5-single-levels-monthly-means"),
    "z500": dict(src="geopotential",            level=500,  scale=1 / 9.80665,   offset=0.0,
                 title="500 hPa geopotential height anomaly", cbar="m", vmax=240,
                 cds_var="geopotential", cds_ds="reanalysis-era5-pressure-levels-monthly-means"),
}

_BWR = LinearSegmentedColormap.from_list("cm_anom", [
    (0.00, "#1a3a8f"), (0.15, "#2b6fd6"), (0.32, "#7db8e8"), (0.46, "#cfe6f5"),
    (0.50, "#ffffff"),
    (0.54, "#fbe3d0"), (0.68, "#f3a072"), (0.85, "#d9402a"), (1.00, "#7a0d18")])
_PRC = LinearSegmentedColormap.from_list("prc_anom", [
    "#5a3410", "#9c6b1e", "#d2a24a", "#ecd9a6", "#ffffff",
    "#bce3cf", "#5cbf9a", "#1f8f8f", "#1b5a8a"])
# daily totals: same family as the site's other IMERG accumulation loops
_PRC_TOT = LinearSegmentedColormap.from_list("prc_tot", [
    "#2b3a6b", "#3b76c4", "#3fb0b0", "#52c452", "#c8d63f",
    "#f0a800", "#e2502a", "#b3247a", "#ffffff"])
PRECIP_TOTAL_LEVELS = [1, 2, 5, 10, 20, 35, 50, 75, 100, 150]
PRECIP_MONTHLY_LEVELS = [25, 50, 100, 175, 275, 400]


# ── shared math ──────────────────────────────────────────────────────────────
def eval_clim(coef: np.ndarray, doy: int) -> np.ndarray:
    x = 2 * np.pi * doy / 365.0
    nharm = (coef.shape[0] - 1) // 2
    out = coef[0].astype("float64").copy()
    for h in range(1, nharm + 1):
        out += coef[2 * h - 1] * np.cos(h * x) + coef[2 * h] * np.sin(h * x)
    return out


def doy365(ts: pd.Timestamp) -> int:
    d = 28 if (ts.month, ts.day) == (2, 29) else ts.day
    return pd.Timestamp(2001, ts.month, d).dayofyear - 1


def monthly_clim(coef: np.ndarray, year: int, month: int) -> np.ndarray:
    days = pd.date_range(f"{year}-{month:02d}-01", periods=1, freq="MS")
    ndays = pd.Timestamp(year, month, 1).days_in_month
    fields = [eval_clim(coef, doy365(pd.Timestamp(year, month, d + 1))) for d in range(ndays)]
    return np.mean(fields, axis=0)


# ── regions (masks on the 1.5° climatology grid) ─────────────────────────────
REGION_ORDER = ["global", "nhem", "shem", "tropics", "arctic", "antarctic",
                "land", "ocean", "americas", "eurafrica", "asiapacific"]
REGION_LABELS = {
    "global": "Global", "nhem": "N. Hemisphere", "shem": "S. Hemisphere",
    "tropics": "Tropics (20S–20N)", "arctic": "Arctic (66.5–90N)",
    "antarctic": "Antarctic (90–66.5S)", "land": "Land", "ocean": "Ocean",
    "americas": "Americas", "eurafrica": "Europe–Africa", "asiapacific": "Asia–Pacific",
}
_LSM = None


def _land_mask(lats, lons) -> np.ndarray:
    """Fractional land mask on the 1.5° grid (from ARCO's static field, cached)."""
    global _LSM
    p = DATA / "lsm15.npy"
    if _LSM is None:
        if p.exists():
            _LSM = np.load(p)
        else:
            ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
            lsm = ds["land_sea_mask"]
            if "time" in lsm.dims:
                lsm = lsm.isel(time=0)
            _LSM = lsm.interp(latitude=lats, longitude=lons).values
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, _LSM)
    return _LSM


def region_means(anom: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Cosine-weighted means of a (lat, lon) anomaly field for every region."""
    LO, LA = np.meshgrid(lons % 360, lats)
    w = np.cos(np.deg2rad(LA))
    land = _land_mask(lats, lons) >= 0.5
    masks = {
        "global": np.ones_like(LA, bool),
        "nhem": LA >= 0, "shem": LA < 0,
        "tropics": (LA >= -20) & (LA <= 20),
        "arctic": LA >= 66.5, "antarctic": LA <= -66.5,
        "land": land, "ocean": ~land,
        "americas": (LO >= 190) & (LO <= 330) & (LA >= -60) & (LA <= 75),
        "eurafrica": (((LO >= 330) | (LO <= 60)) & (LA >= -40) & (LA <= 75)),
        "asiapacific": (LO > 60) & (LO < 190) & (LA >= -50) & (LA <= 75),
    }
    out = {}
    for k, m in masks.items():
        ww = w * m
        out[k] = round(float(np.nansum(anom * ww) / np.nansum(ww)), 3)
    return out


# ── ERA5 access ──────────────────────────────────────────────────────────────
def open_arco() -> xr.Dataset:
    return xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})


def latest_complete_day(ds: xr.Dataset) -> pd.Timestamp | None:
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    probe = ds["2m_temperature"].sel(latitude=0.0, longitude=0.0, method="nearest")
    for back in range(4, 16):
        d = today - pd.Timedelta(days=back)
        v = probe.sel(time=[d + pd.Timedelta(hours=h) for h in SYNOPTIC_H]).values
        if np.isfinite(v).all():
            return d
    return None


def era5_daily_mean(ds: xr.Dataset, day: pd.Timestamp, spec: dict) -> xr.DataArray:
    da = ds[spec["src"]]
    if spec["level"] is not None:
        da = da.sel(level=spec["level"])
    sel = da.sel(time=[day + pd.Timedelta(hours=h) for h in SYNOPTIC_H])
    return sel.mean("time").compute() * spec["scale"] + spec["offset"]


# ── rendering ────────────────────────────────────────────────────────────────
def render_global(field, lats, lons, title, cbar_label, out: Path,
                  cmap=_BWR, vmax=None, levels=None):
    fig = plt.figure(figsize=(12.2, 6.6), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    if levels is not None:
        lev = [-x for x in reversed(levels)] + list(levels)
        norm = BoundaryNorm(lev, cmap.N, extend="both")
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm,
                           transform=PC, shading="auto", rasterized=True)
    else:
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                           transform=PC, shading="auto", rasterized=True)
    ax.coastlines(linewidth=0.5, color="#333", resolution="110m")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25,
                   edgecolor="#666", alpha=0.5)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.012, shrink=0.82,
                      fraction=0.028, extend="both")
    cb.set_label(cbar_label, fontsize=10); cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, loc="left", pad=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.relative_to(ASSETS.parent)}", flush=True)


def render_precip_total(field, lats, lons, title, out: Path):
    fig = plt.figure(figsize=(12.2, 6.6), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    norm = BoundaryNorm(PRECIP_TOTAL_LEVELS, _PRC_TOT.N, extend="max")
    wet = np.ma.masked_less(field, PRECIP_TOTAL_LEVELS[0])
    im = ax.pcolormesh(lons, lats, wet, cmap=_PRC_TOT, norm=norm,
                       transform=PC, shading="auto", rasterized=True)
    ax.set_facecolor("#f4f2ec")
    ax.coastlines(linewidth=0.5, color="#333", resolution="110m")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25,
                   edgecolor="#666", alpha=0.5)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.012, shrink=0.82,
                      fraction=0.028, extend="max")
    cb.set_label("mm/day", fontsize=10); cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, loc="left", pad=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.relative_to(ASSETS.parent)}", flush=True)


# ── daily pipeline: loops + latest maps + region series ──────────────────────
def run_era5() -> str | None:
    clims = {}
    for name in ERA5_VARS:
        p = HERE / f"era5_clim_{name}.nc"
        if not p.exists():
            print(f"  {p.name} missing — run build_era5_clim.py first", flush=True)
            return None
        clims[name] = xr.open_dataset(p)
    lats = clims["t2m"].latitude.values
    lons = clims["t2m"].longitude.values

    ds = open_arco()
    day = latest_complete_day(ds)
    if day is None:
        print("  no complete ERA5 day found in probe window", flush=True)
        return None
    print(f"  ERA5 latest complete day: {day:%Y-%m-%d}", flush=True)

    loop_dates = pd.date_range(day - pd.Timedelta(days=LOOP_DAYS - 1), day)
    series = (pd.read_csv(SERIES_CSV, index_col=0) if SERIES_CSV.exists()
              else pd.DataFrame(columns=REGION_ORDER))
    # teleconnection state (present once build_z500_indices.py has run)
    tele = None
    tele_pat = None
    if TELE_CSV.exists() and (HERE / "tele_patterns.nc").exists():
        tele = pd.read_csv(TELE_CSV, index_col=0, parse_dates=True)
        tele_pat = xr.open_dataset(HERE / "tele_patterns.nc")

    manifest_regions = {}
    for name, spec in ERA5_VARS.items():
        coef = clims[name]["coef"].values
        vdir = ANIM / name; vdir.mkdir(parents=True, exist_ok=True)
        entries = []
        for d in loop_dates:
            fp = vdir / f"{d:%Y%m%d}.webp"
            need_series = (name == "t2m" and d.strftime("%Y-%m-%d") not in series.index)
            need_tele = (name == "z500" and tele is not None
                         and pd.Timestamp(d.date()) not in tele.index)
            if not fp.exists() or need_series or need_tele:
                daily = era5_daily_mean(ds, d, spec)
                daily15 = daily.interp(latitude=lats, longitude=lons)
                anom = daily15.values - eval_clim(coef, doy365(d))
                if not fp.exists():
                    render_global(anom, lats, lons,
                                  f"ERA5 {spec['title']} — {d:%Y-%m-%d} (vs 1991–2020)",
                                  spec["cbar"], fp, vmax=spec["vmax"])
                if need_series:
                    series.loc[d.strftime("%Y-%m-%d")] = region_means(anom, lats, lons)
                if need_tele:
                    tele.loc[pd.Timestamp(d.date())] = _tele_project_day(
                        anom, lats, tele_pat, pd.Timestamp(d.date()))
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": f"{d:%Y-%m-%d}", "label": f"{d:%a %b %d, %Y}"})
        keep = {f"{d:%Y%m%d}" for d in loop_dates}
        for old in vdir.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        # latest-day static map = copy of the newest frame
        import shutil
        shutil.copyfile(vdir / f"{day:%Y%m%d}.webp", ASSETS / f"{name}_anom.webp")
        manifest_regions[name] = {"label": spec["title"], "frames": entries}

    (ANIM / "manifest.json").write_text(json.dumps(
        {"ver": f"{day:%Y%m%d}", "selectorLabel": "Field",
         "default": "t2m", "regions": manifest_regions}))
    series = series[[c for c in REGION_ORDER if c in series.columns]]
    series.sort_index().round(3).to_csv(SERIES_CSV, index_label="date")
    print(f"  series through {series.index.max()} ({len(series)} rows)", flush=True)
    if tele is not None:
        tele = tele[~tele.index.duplicated(keep="last")].sort_index()
        tele.to_csv(TELE_CSV, index_label="date")
        render_tele_table(tele)
    return f"{day:%Y-%m-%d}"


def _tele_project_day(anom: np.ndarray, lats: np.ndarray,
                      pat: xr.Dataset, d: pd.Timestamp) -> dict:
    """Standardized daily indices: projection of the day's NH Z500 anomaly onto
    OUR rotated-PCA loading patterns (see build_z500_indices.py)."""
    sel = lats >= float(pat.lat.min())
    w = np.sqrt(np.cos(np.deg2rad(lats[sel])))[:, None]
    flat = (anom[sel] * w).reshape(-1)
    k = doy365(d)
    out = {}
    for name in [str(i) for i in pat.index.values]:
        v = pat["pattern"].sel(index=name).values.reshape(-1)
        raw = float(flat @ v / (v @ v))
        mu = float(pat[f"{name}_mu"].values[k])
        sg = float(pat[f"{name}_sd"].values[k])
        out[name] = round((raw - mu) / sg, 3)
    return out


def render_tele_table(tele: pd.DataFrame, n_months: int = 13):
    """Monthly-mean teleconnection table: values printed in colored cells."""
    monthly = tele.resample("MS").mean().tail(n_months)
    names = ["nao", "pna", "epo", "wpo", "ao"]
    M = monthly[names].T.values
    fig, ax = plt.subplots(figsize=(12.2, 3.4), dpi=100)
    im = ax.imshow(M, cmap=_BWR, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(names)), [n.upper() for n in names], fontsize=10)
    ax.set_xticks(range(len(monthly)),
                  [f"{m:%b\n%Y}" if m.month == 1 or i == 0 else f"{m:%b}"
                   for i, m in enumerate(monthly.index)], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold",
                        color="white" if abs(v) > 1.4 else "#111")
    ax.set_title("Monthly-mean teleconnection indices — Z500-based, standardized vs 1991–2020"
                 f"  ·  latest month is partial", fontsize=11, loc="left", pad=8)
    fig.text(0.005, 0.01,
             "Computed in-house: varimax-rotated PCA of our ERA5 Z500 monthly anomalies (1991–2020 "
             "base, 20–90N); AO = unrotated EOF1. Daily values = projection onto the loading patterns.",
             fontsize=7, color="#888")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=8)
    fig.savefig(ASSETS / "teleconnections_monthly.webp", dpi=100, facecolor="white",
                bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print("  wrote teleconnections_monthly.webp", flush=True)


# ── monthly means (CDS; skipped without credentials) ─────────────────────────
def run_monthly() -> list:
    months = []
    today = datetime.now(timezone.utc).date()
    first_of_this = pd.Timestamp(today.year, today.month, 1)
    wanted = [(first_of_this - pd.DateOffset(months=k)) for k in range(1, MONTHLY_KEEP + 1)]
    mdir = ASSETS / "monthly"; mdir.mkdir(parents=True, exist_ok=True)
    for v in list(ERA5_VARS) + ["precip"]:
        (mdir / v).mkdir(parents=True, exist_ok=True)
    missing = [(m, v) for m in wanted for v in ERA5_VARS
               if not (mdir / v).joinpath(f"{m:%Y%m}.webp").exists()]
    months = [f"{m:%Y-%m}" for m in wanted]
    if not missing:
        return months
    try:
        import cdsapi
        c = cdsapi.Client(quiet=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  monthly: CDS unavailable ({repr(e)[:60]}); keeping existing maps", flush=True)
        return months
    import tempfile
    clims = {v: xr.open_dataset(HERE / f"era5_clim_{v}.nc") for v in ERA5_VARS}
    lats = clims["t2m"].latitude.values; lons = clims["t2m"].longitude.values
    for m, name in missing:
        spec = ERA5_VARS[name]
        req = {"product_type": "monthly_averaged_reanalysis",
               "variable": spec["cds_var"], "year": f"{m.year}",
               "month": f"{m.month:02d}", "time": "00:00",
               "grid": "1.5/1.5", "format": "netcdf"}
        if spec["level"] is not None:
            req["pressure_level"] = str(spec["level"])
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
            target = tf.name
        try:
            c.retrieve(spec["cds_ds"], req, target)
            dm = xr.open_dataset(target)
            v = dm[[k for k in dm.data_vars if dm[k].ndim >= 2][0]].squeeze()
            laname = "latitude" if "latitude" in v.coords else "lat"
            loname = "longitude" if "longitude" in v.coords else "lon"
            v = v.rename({laname: "latitude", loname: "longitude"}).sortby("latitude")
            v = v.interp(latitude=lats, longitude=lons)
            field = v.values * spec["scale"] + spec["offset"]
            anom = field - monthly_clim(clims[name]["coef"].values, m.year, m.month)
            render_global(anom, lats, lons,
                          f"ERA5 {spec['title']} — {m:%B %Y} monthly mean (vs 1991–2020)",
                          spec["cbar"], (mdir / name).joinpath(f"{m:%Y%m}.webp"),
                          vmax=max(2.0, spec["vmax"] / 2.5))
        except Exception as e:                               # noqa: BLE001
            print(f"  monthly {name} {m:%Y-%m} failed ({repr(e)[:70]})", flush=True)
    return months


# ── IMERG ────────────────────────────────────────────────────────────────────
IMERG_STORE = DATA / "imerg_global"          # per-day strided global grids (~1 MB each)


def _imerg_recent_days(n: int = 10) -> dict:
    """Ensure the last n IMERG Early daily global 0.5° grids are cached; return
    {YYYYMMDD: (lat, lon) mm/day} for the days available."""
    from imerg_precip import _login
    from build_imerg_clim_global import _opendap_url, _fetch_day
    import earthaccess
    _login()
    IMERG_STORE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    gs = earthaccess.search_data(short_name="GPM_3IMERGDE", version="07",
                                 temporal=(f"{now - timedelta(days=n + 2):%Y-%m-%d}",
                                           f"{now:%Y-%m-%d}"))
    by_day = {}
    for g in gs:
        url = _opendap_url(g)
        if url:
            by_day[url.split(".3IMERG.")[1][:8]] = url
    sess = None
    out = {}
    for ymd, url in sorted(by_day.items())[-n:]:
        p = IMERG_STORE / f"{ymd}.npy"
        if not p.exists():
            if sess is None:
                sess = earthaccess.get_requests_https_session()
            _, grid = _fetch_day((url, 1, sess))
            if grid is None:
                continue
            np.save(p, grid.astype("float32"))
        out[ymd] = np.load(p)
    for old in IMERG_STORE.glob("*.npy"):                    # prune beyond ~2n
        if old.stem not in by_day and len(list(IMERG_STORE.glob("*.npy"))) > 2 * n:
            old.unlink()
    return out


def _imerg_days_for(dates) -> dict:
    """Ensure specific past days' global 0.5° grids are cached; return subset."""
    from imerg_precip import _login
    from build_imerg_clim_global import _opendap_url, _fetch_day
    import earthaccess
    IMERG_STORE.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in IMERG_STORE.glob("*.npy")}
    need = [d for d in dates if f"{d:%Y%m%d}" not in have]
    if need:
        _login()
        gs = earthaccess.search_data(short_name="GPM_3IMERGDE", version="07",
                                     temporal=(f"{min(need):%Y-%m-%d}",
                                               f"{max(need) + pd.Timedelta(days=1):%Y-%m-%d}"))
        by_day = {}
        for g in gs:
            url = _opendap_url(g)
            if url:
                by_day[url.split(".3IMERG.")[1][:8]] = url
        sess = earthaccess.get_requests_https_session()
        import concurrent.futures as cf
        def work(d):
            ymd = f"{d:%Y%m%d}"
            if ymd not in by_day:
                return
            _, grid = _fetch_day((by_day[ymd], 1, sess))
            if grid is not None:
                np.save(IMERG_STORE / f"{ymd}.npy", grid.astype("float32"))
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(work, need))
    out = {}
    for d in dates:
        p = IMERG_STORE / f"{d:%Y%m%d}.npy"
        if p.exists():
            out[f"{d:%Y%m%d}"] = np.load(p)
    return out


def run_imerg() -> str | None:
    """Daily precip TOTALS as a 30-day loop (joins the ERA5 anim manifest as a
    fourth field) + monthly accumulation ANOMALIES for the recent complete
    months. Daily anomalies were dropped deliberately: daily rain is
    intermittent, so anomalies only make sense aggregated (monthly)."""
    clim_p = HERE / "imerg_clim_global.nc"
    if not clim_p.exists():
        print(f"  {clim_p.name} missing — run build_imerg_clim_global.py first", flush=True)
        return None
    cds = xr.open_dataset(clim_p)
    coef = cds["coef"].values
    lats, lons = cds.lat.values, cds.lon.values

    days = _imerg_recent_days(LOOP_DAYS + 3)
    if not days:
        print("  no recent IMERG Early daily granules found", flush=True)
        return None
    ymds = sorted(days)[-LOOP_DAYS:]
    day = pd.Timestamp(f"{ymds[-1][:4]}-{ymds[-1][4:6]}-{ymds[-1][6:]}")

    # 30-day daily-totals loop
    vdir = ANIM / "precip"; vdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for ymd in ymds:
        d = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
        fp = vdir / f"{ymd}.webp"
        if not fp.exists():
            render_precip_total(days[ymd], lats, lons,
                                f"GPM IMERG daily precipitation — {d:%Y-%m-%d} (Early run)", fp)
        entries.append({"idx": len(entries), "file": fp.name,
                        "date": f"{d:%Y-%m-%d}", "label": f"{d:%a %b %d, %Y}"})
    for old in vdir.glob("*.webp"):
        if old.stem not in set(ymds):
            old.unlink()
    manp = ANIM / "manifest.json"
    if manp.exists():
        man = json.loads(manp.read_text())
        man["regions"]["precip"] = {"label": "daily precipitation (IMERG)",
                                    "frames": entries}
        manp.write_text(json.dumps(man))

    # monthly accumulation anomalies (from Early dailies; needs the full month)
    mdir = ASSETS / "monthly"; mdir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    first_of_this = pd.Timestamp(today.year, today.month, 1)
    for k in range(1, MONTHLY_KEEP + 1):
        m = first_of_this - pd.DateOffset(months=k)
        out = (mdir / "precip").joinpath(f"{m:%Y%m}.webp")
        if out.exists():
            continue
        mdays = pd.date_range(m, m + pd.DateOffset(months=1) - pd.Timedelta(days=1))
        grids = _imerg_days_for(mdays)
        if len(grids) < len(mdays) - 2:                      # tolerate a couple gaps
            print(f"  monthly precip {m:%Y-%m}: only {len(grids)}/{len(mdays)} days — skipped",
                  flush=True)
            continue
        tot = np.sum(list(grids.values()), axis=0)
        ctot = np.sum([eval_clim(coef, doy365(d)) for d in mdays], axis=0)
        fig_levels = PRECIP_MONTHLY_LEVELS
        render_global(tot - ctot, lats, lons,
                      f"GPM IMERG precipitation anomaly — {m:%B %Y} (vs 2001–2025)",
                      "mm", out, cmap=_PRC, levels=fig_levels)
    return f"{day:%Y-%m-%d}"


def write_monthly_manifest():
    """Manifest for the sst_anim viewer: one region per field, frames = the
    committed monthly maps (oldest→newest), so the page can animate them."""
    mdir = ASSETS / "monthly"
    labels = {"t2m": "2 m temperature", "z500": "500 hPa height",
              "mslp": "MSLP", "precip": "precipitation"}
    regions = {}
    for v in ["t2m", "z500", "mslp", "precip"]:
        vd = mdir / v
        if not vd.exists():
            continue
        files = sorted(vd.glob("[0-9]*.webp"), key=lambda f: f.stem)[-12:]
        frames = [{"idx": i, "file": f.name, "date": f"{f.stem[:4]}-{f.stem[4:6]}-01",
                   "label": pd.Timestamp(f"{f.stem[:4]}-{f.stem[4:6]}-01").strftime("%b %Y")}
                  for i, f in enumerate(files)]
        if frames:
            regions[v] = {"label": labels[v] + " monthly anomaly", "frames": frames}
    if regions:
        ver = max(f["date"] for r in regions.values() for f in r["frames"]).replace("-", "")
        (mdir / "manifest.json").write_text(json.dumps(
            {"ver": ver, "selectorLabel": "Field", "default": "t2m", "regions": regions}))
        print("  wrote monthly/manifest.json", flush=True)


def main() -> int:
    print("ERA5 daily (maps + loops + regional series):", flush=True)
    era5_day = run_era5()
    print("Monthly means:", flush=True)
    months = run_monthly()
    print("IMERG precip anomaly:", flush=True)
    imerg_day = run_imerg()
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_monthly_manifest()
    (ASSETS / "manifest.json").write_text(json.dumps({
        "era5_day": era5_day, "imerg_day": imerg_day, "months": months,
        "loop_days": LOOP_DAYS,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}))
    print("wrote manifest.json", flush=True)
    return 0 if era5_day else 1


if __name__ == "__main__":
    sys.exit(main())
