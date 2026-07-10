#!/usr/bin/env python3
"""Daily global climate monitor: ERA5 anomalies + IMERG precip anomaly.

ERA5 (via the public ARCO-ERA5 hourly zarr, anonymous): the latest complete
day (~5-day lag; found by probing backward for non-NaN data) of daily-mean
2 m temperature, MSLP, and 500 hPa geopotential height, as anomalies against
the 1991–2020 climatology built by build_era5_clim.py (WeatherBench2 1.5°).

IMERG: the latest GPM IMERG Early daily total at 0.5° (server-side OPeNDAP
stride), as an anomaly against the 2001–2025 global climatology built by
build_imerg_clim_global.py.

Products (assets/climate/):
  t2m_anom.webp, z500_anom.webp, mslp_anom.webp, precip_anom.webp
  manifest.json  {era5_day, imerg_day, generated}

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
SITE_ROOT = Path(sys.argv[0]).resolve().parents[2] if len(sys.argv) else HERE.parents[1]
ASSETS = HERE.parents[1] / "assets" / "climate"
sys.path.insert(0, str(HERE.parent / "sst"))

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
PC = ccrs.PlateCarree()

ERA5_VARS = {
    "t2m":  dict(src="2m_temperature",          level=None, scale=1.0,         offset=-273.15,
                 title="2 m temperature anomaly", cbar="°C",  vmax=12),
    "mslp": dict(src="mean_sea_level_pressure", level=None, scale=0.01,        offset=0.0,
                 title="mean sea-level pressure anomaly", cbar="hPa", vmax=24),
    "z500": dict(src="geopotential",            level=500,  scale=1 / 9.80665,  offset=0.0,
                 title="500 hPa geopotential height anomaly", cbar="m", vmax=240),
}

# diverging blue-white-red (same family as the SST anomaly maps)
_BWR = LinearSegmentedColormap.from_list("cm_anom", [
    (0.00, "#1a3a8f"), (0.15, "#2b6fd6"), (0.32, "#7db8e8"), (0.46, "#cfe6f5"),
    (0.50, "#ffffff"),
    (0.54, "#fbe3d0"), (0.68, "#f3a072"), (0.85, "#d9402a"), (1.00, "#7a0d18")])
# dry(brown) <-> wet(teal) for precip anomaly (same family as the IMERG products)
_PRC = LinearSegmentedColormap.from_list("prc_anom", [
    "#5a3410", "#9c6b1e", "#d2a24a", "#ecd9a6", "#ffffff",
    "#bce3cf", "#5cbf9a", "#1f8f8f", "#1b5a8a"])
PRECIP_LEVELS = [2, 5, 10, 20, 35, 60, 100]                 # mm/day, symmetric


def eval_clim(coef: np.ndarray, doy365: int) -> np.ndarray:
    """Harmonic clim(doy) from (1+2H, ny, nx) coefficients; doy365 in 0..364."""
    x = 2 * np.pi * doy365 / 365.0
    nharm = (coef.shape[0] - 1) // 2
    out = coef[0].astype("float64").copy()
    for h in range(1, nharm + 1):
        out += coef[2 * h - 1] * np.cos(h * x) + coef[2 * h] * np.sin(h * x)
    return out


def doy365(ts: pd.Timestamp) -> int:
    d = 28 if (ts.month, ts.day) == (2, 29) else ts.day
    return pd.Timestamp(2001, ts.month, d).dayofyear - 1


# ── ERA5 ─────────────────────────────────────────────────────────────────────
def open_arco() -> xr.Dataset:
    return xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})


def latest_complete_day(ds: xr.Dataset) -> pd.Timestamp | None:
    """Newest UTC day whose 24 hourly steps are all present (ARCO's time axis is
    a pre-allocated template — beyond-data steps read as NaN)."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    probe = ds["2m_temperature"].sel(latitude=0.0, longitude=0.0, method="nearest")
    for back in range(4, 16):
        d = today - pd.Timedelta(days=back)
        v = probe.sel(time=slice(d, d + pd.Timedelta(hours=23))).values
        if len(v) == 24 and np.isfinite(v).all():
            return d
    return None


def era5_daily_mean(ds: xr.Dataset, day: pd.Timestamp, spec: dict) -> xr.DataArray:
    da = ds[spec["src"]]
    if spec["level"] is not None:
        da = da.sel(level=spec["level"])
    sel = da.sel(time=slice(day, day + pd.Timedelta(hours=23)))
    return sel.mean("time").compute() * spec["scale"] + spec["offset"]


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
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="#666",
                   alpha=0.5)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.012, shrink=0.82,
                      fraction=0.028, extend="both")
    cb.set_label(cbar_label, fontsize=10); cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, loc="left", pad=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def run_era5() -> str | None:
    clims = {}
    for name in ERA5_VARS:
        p = HERE / f"era5_clim_{name}.nc"
        if not p.exists():
            print(f"  {p.name} missing — run build_era5_clim.py first", flush=True)
            return None
        clims[name] = xr.open_dataset(p)
    ds = open_arco()
    day = latest_complete_day(ds)
    if day is None:
        print("  no complete ERA5 day found in probe window", flush=True)
        return None
    print(f"  ERA5 latest complete day: {day:%Y-%m-%d}", flush=True)
    k = doy365(day)
    for name, spec in ERA5_VARS.items():
        cds = clims[name]
        daily = era5_daily_mean(ds, day, spec)
        # evaluate clim on its native 1.5° grid, take the data onto it
        daily15 = daily.interp(latitude=cds.latitude.values, longitude=cds.longitude.values)
        anom = daily15.values - eval_clim(cds["coef"].values, k)
        render_global(anom, cds.latitude.values, cds.longitude.values,
                      f"ERA5 {spec['title']} — {day:%Y-%m-%d} (vs 1991–2020)",
                      spec["cbar"], ASSETS / f"{name}_anom.webp", vmax=spec["vmax"])
    return f"{day:%Y-%m-%d}"


# ── IMERG ────────────────────────────────────────────────────────────────────
def run_imerg() -> str | None:
    clim_p = HERE / "imerg_clim_global.nc"
    if not clim_p.exists():
        print(f"  {clim_p.name} missing — run build_imerg_clim_global.py first", flush=True)
        return None
    cds = xr.open_dataset(clim_p)
    from imerg_precip import _login
    from build_imerg_clim_global import _opendap_url, _fetch_day, STRIDE   # noqa: F401
    import earthaccess
    _login()
    now = datetime.now(timezone.utc)
    gs = earthaccess.search_data(short_name="GPM_3IMERGDE", version="07",
                                 temporal=(f"{now - timedelta(days=6):%Y-%m-%d}",
                                           f"{now:%Y-%m-%d}"))
    if not gs:
        print("  no recent IMERG Early daily granules found", flush=True)
        return None
    by_day = {}
    for g in gs:
        url = _opendap_url(g)
        if url:
            by_day[url.split(".3IMERG.")[1][:8]] = url
    ymd = sorted(by_day)[-1]
    day = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
    sess = earthaccess.get_requests_https_session()
    _, grid = _fetch_day((by_day[ymd], 1, sess))
    if grid is None:
        print("  IMERG fetch failed", flush=True)
        return None
    anom = grid - eval_clim(cds["coef"].values, doy365(day))
    render_global(anom, cds.lat.values, cds.lon.values,
                  f"GPM IMERG daily precipitation anomaly — {day:%Y-%m-%d} (vs 2001–2025)",
                  "mm/day", ASSETS / "precip_anom.webp",
                  cmap=_PRC, levels=PRECIP_LEVELS)
    return f"{day:%Y-%m-%d}"


def main() -> int:
    print("ERA5 anomalies:", flush=True)
    era5_day = run_era5()
    print("IMERG precip anomaly:", flush=True)
    imerg_day = run_imerg()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "manifest.json").write_text(json.dumps({
        "era5_day": era5_day, "imerg_day": imerg_day,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}))
    print("wrote manifest.json", flush=True)
    return 0 if (era5_day or imerg_day) else 1


if __name__ == "__main__":
    sys.exit(main())
