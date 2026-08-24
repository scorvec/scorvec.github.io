#!/usr/bin/env python3
"""
NDJFM winter temperature outlook for ~100 US cities from RONI + climate trend.

Method (deliberately minimal — the point is what these two predictors alone say):
  1. Winter (Nov–Mar) mean RONI per year from cached ERSST v5:
     Niño-3.4 anomaly minus 20°S–20°N tropical-mean anomaly (base 1991–2020),
     3-month-smoothed, averaged over NDJFM. Winters 1950/51 → 2025/26.
  2. City NDJFM mean 2-m temperature per winter from the cached ERA5 monthly
     t2m (2°, 1950→present), sampled at the nearest grid cell, expressed as an
     anomaly vs the city's 1991–2020 NDJFM normal.
  3. Per city, ordinary least squares:  T_anom = a + b·(year−2000) + c·RONI.
     The trend term is always kept; the RONI term is kept only where its
     coefficient is significant at 90% (two-sided t-test) — "if one exists".
  4. Forecast NDJFM 2026/27 = a + b·26 + c·RONI_assumed, mapped and contoured
     over CONUS (cubic-buffered linear interpolation of the city values,
     clipped to the national outline).

Usage:
    python roni_winter_forecast.py [--roni 1.9] [--out-dir plots]

RONI assumption defaults to the site's latest daily RONI (assets/sst/
manifest.json); override with --roni. Outputs (PNG + per-city CSV) stay local.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ERSST = REPO / "scripts" / "sst" / "data" / "ersst_v5_mnmean.nc"
ERA5 = REPO / "scripts" / "sst" / "data" / "era5_t2m_mon.nc"
MANIFEST = REPO / "assets" / "sst" / "manifest.json"

Y0, Y1 = 1950, 2025            # winters 1950/51 … 2025/26 (year = Nov year)
CLIM0, CLIM1 = 1991, 2020      # anomaly base (winter start years)
TARGET = 2026                  # forecast winter 2026/27
WINTER_MONTHS = (11, 12, 1, 2, 3)
P_SIG = 0.10                   # keep the RONI term only below this p-value

# ~100 cities, roughly evenly spread over CONUS (name, lat, lon degE-360-safe)
CITIES = [
    ("Seattle WA", 47.61, -122.33), ("Spokane WA", 47.66, -117.43),
    ("Portland OR", 45.52, -122.68), ("Boise ID", 43.62, -116.21),
    ("Missoula MT", 46.87, -113.99), ("Billings MT", 45.78, -108.50),
    ("Bismarck ND", 46.81, -100.78), ("Fargo ND", 46.88, -96.79),
    ("Minneapolis MN", 44.98, -93.27), ("Duluth MN", 46.79, -92.10),
    ("Sioux Falls SD", 43.55, -96.73), ("Rapid City SD", 44.08, -103.23),
    ("Casper WY", 42.87, -106.31), ("Salt Lake City UT", 40.76, -111.89),
    ("Reno NV", 39.53, -119.81), ("Sacramento CA", 38.58, -121.49),
    ("San Francisco CA", 37.77, -122.42), ("Fresno CA", 36.74, -119.79),
    ("Los Angeles CA", 34.05, -118.24), ("San Diego CA", 32.72, -117.16),
    ("Las Vegas NV", 36.17, -115.14), ("Phoenix AZ", 33.45, -112.07),
    ("Tucson AZ", 32.22, -110.97), ("Flagstaff AZ", 35.20, -111.65),
    ("Albuquerque NM", 35.08, -106.65), ("El Paso TX", 31.76, -106.49),
    ("Denver CO", 39.74, -104.99), ("Grand Junction CO", 39.06, -108.55),
    ("Cheyenne WY", 41.14, -104.82), ("North Platte NE", 41.12, -100.77),
    ("Omaha NE", 41.26, -95.93), ("Wichita KS", 37.69, -97.34),
    ("Goodland KS", 39.35, -101.71), ("Oklahoma City OK", 35.47, -97.52),
    ("Tulsa OK", 36.15, -95.99), ("Amarillo TX", 35.19, -101.85),
    ("Lubbock TX", 33.58, -101.86), ("Dallas TX", 32.78, -96.80),
    ("Austin TX", 30.27, -97.74), ("San Antonio TX", 29.42, -98.49),
    ("Houston TX", 29.76, -95.37), ("Corpus Christi TX", 27.80, -97.40),
    ("Brownsville TX", 25.90, -97.50), ("Midland TX", 32.00, -102.08),
    ("Shreveport LA", 32.52, -93.75), ("New Orleans LA", 29.95, -90.07),
    ("Baton Rouge LA", 30.45, -91.15), ("Little Rock AR", 34.75, -92.29),
    ("Fayetteville AR", 36.06, -94.16), ("Kansas City MO", 39.10, -94.58),
    ("St Louis MO", 38.63, -90.20), ("Springfield MO", 37.22, -93.30),
    ("Des Moines IA", 41.59, -93.62), ("Cedar Rapids IA", 41.98, -91.67),
    ("Madison WI", 43.07, -89.40), ("Milwaukee WI", 43.04, -87.91),
    ("Green Bay WI", 44.51, -88.02), ("Chicago IL", 41.88, -87.63),
    ("Springfield IL", 39.80, -89.64), ("Indianapolis IN", 39.77, -86.16),
    ("Fort Wayne IN", 41.08, -85.14), ("Detroit MI", 42.33, -83.05),
    ("Grand Rapids MI", 42.96, -85.66), ("Marquette MI", 46.54, -87.40),
    ("Columbus OH", 39.96, -83.00), ("Cleveland OH", 41.50, -81.69),
    ("Cincinnati OH", 39.10, -84.51), ("Louisville KY", 38.25, -85.76),
    ("Lexington KY", 38.04, -84.50), ("Nashville TN", 36.16, -86.78),
    ("Memphis TN", 35.15, -90.05), ("Knoxville TN", 35.96, -83.92),
    ("Jackson MS", 32.30, -90.18), ("Birmingham AL", 33.52, -86.80),
    ("Mobile AL", 30.69, -88.04), ("Atlanta GA", 33.75, -84.39),
    ("Savannah GA", 32.08, -81.09), ("Jacksonville FL", 30.33, -81.66),
    ("Orlando FL", 28.54, -81.38), ("Tampa FL", 27.95, -82.46),
    ("Miami FL", 25.76, -80.19), ("Tallahassee FL", 30.44, -84.28),
    ("Columbia SC", 34.00, -81.03), ("Charleston SC", 32.78, -79.93),
    ("Charlotte NC", 35.23, -80.84), ("Raleigh NC", 35.78, -78.64),
    ("Asheville NC", 35.60, -82.55), ("Richmond VA", 37.54, -77.44),
    ("Norfolk VA", 36.85, -76.29), ("Roanoke VA", 37.27, -79.94),
    ("Washington DC", 38.90, -77.04), ("Baltimore MD", 39.29, -76.61),
    ("Charleston WV", 38.35, -81.63), ("Pittsburgh PA", 40.44, -79.99),
    ("Philadelphia PA", 39.95, -75.17), ("Harrisburg PA", 40.27, -76.88),
    ("New York NY", 40.71, -74.01), ("Albany NY", 42.65, -73.75),
    ("Buffalo NY", 42.89, -78.88), ("Syracuse NY", 43.05, -76.15),
    ("Hartford CT", 41.77, -72.67), ("Boston MA", 42.36, -71.06),
    ("Providence RI", 41.82, -71.41), ("Burlington VT", 44.48, -73.21),
    ("Concord NH", 43.21, -71.54), ("Portland ME", 43.66, -70.26),
    ("Caribou ME", 46.87, -68.01),
]


def winter_mean(da: xr.DataArray, start_year: int) -> xr.DataArray | None:
    """NDJFM mean for the winter starting in `start_year` (Nov)."""
    months = ([f"{start_year}-11", f"{start_year}-12"]
              + [f"{start_year+1}-{m:02d}" for m in (1, 2, 3)])
    sel = da.sel(time=[t for t in months])
    if sel.sizes["time"] != 5:
        return None
    return sel.mean("time")


def roni_history() -> pd.Series:
    """NDJFM-mean RONI per winter start year, from ERSST v5."""
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice(f"{CLIM0}-01-01", f"{CLIM1}-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wmean(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    n34 = wmean(anom.sel(lat=slice(-5, 5), lon=slice(190, 240)))
    trop = wmean(anom.sel(lat=slice(-20, 20)))
    roni_m = (n34 - trop).rolling(time=3, center=True, min_periods=2).mean()
    s = roni_m.to_series()
    out = {}
    for y in range(Y0, Y1 + 1):
        months = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                 [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
        vals = [s.get(m, np.nan) for m in months]
        if np.isfinite(vals).sum() >= 4:
            out[y] = float(np.nanmean(vals))
    return pd.Series(out)


def city_winters() -> tuple[pd.DataFrame, pd.Series]:
    """(winters × cities) NDJFM t2m anomaly table (°F) + city normals check."""
    ds = xr.open_dataset(ERA5)
    t2m = ds["t2m"]
    if float(t2m.max()) > 200:           # Kelvin
        t2m = t2m - 273.15
    t2m = t2m.assign_coords(lon=t2m["lon"] % 360).sortby("lon").sortby("lat")

    # monthly series at each city's nearest grid cell
    lats = xr.DataArray([c[1] for c in CITIES], dims="city")
    lons = xr.DataArray([c[2] % 360 for c in CITIES], dims="city")
    pts = t2m.sel(lat=lats, lon=lons, method="nearest").load()   # (time, city)

    rows = {}
    for y in range(Y0, Y1 + 1):
        months = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                 [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
        try:
            sel = pts.sel(time=months)
        except KeyError:
            continue
        rows[y] = sel.mean("time").values
    df = pd.DataFrame.from_dict(rows, orient="index",
                                columns=[c[0] for c in CITIES])
    normals = df.loc[CLIM0:CLIM1].mean()
    anom_f = (df - normals) * 9.0 / 5.0            # °C anomaly → °F anomaly
    return anom_f, normals


def fit_and_forecast(anom: pd.DataFrame, roni: pd.Series, roni_now: float):
    """Per-city OLS on [year, RONI]; returns forecast + diagnostics DataFrame."""
    years = anom.index.intersection(roni.index)
    X_year = (years - 2000).to_numpy(dtype=float)
    X_roni = roni.loc[years].to_numpy()
    out = []
    for i, (name, lat, lon) in enumerate(CITIES):
        y = anom.loc[years, name].to_numpy()
        A = np.column_stack([np.ones_like(X_year), X_year, X_roni])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = len(y) - 3
        s2 = (resid @ resid) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        t_roni = coef[2] / np.sqrt(cov[2, 2])
        p_roni = 2 * stats.t.sf(abs(t_roni), dof)
        r = np.corrcoef(A @ coef, y)[0, 1]
        c_used = coef[2] if p_roni < P_SIG else 0.0
        fc = coef[0] + coef[1] * (TARGET - 2000) + c_used * roni_now
        trend_part = coef[0] + coef[1] * (TARGET - 2000)
        out.append(dict(city=name, lat=lat, lon=lon, forecast_F=fc,
                        trend_F=trend_part, roni_F=c_used * roni_now,
                        roni_coef=coef[2], roni_p=p_roni,
                        trend_per_decade=coef[1] * 10, fit_r=r))
    return pd.DataFrame(out)


def draw_map(fc: pd.DataFrame, roni_now: float, out_path: Path):
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    fig = plt.figure(figsize=(13, 8.6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-120.5, -73, 22.5, 50.5], crs=ccrs.PlateCarree())

    # interpolate city values onto a fine lat/lon mesh
    gx, gy = np.meshgrid(np.arange(-125, -66, 0.25), np.arange(24, 50.1, 0.25))
    pts = fc[["lon", "lat"]].to_numpy()
    gz = griddata(pts, fc["forecast_F"].to_numpy(), (gx, gy), method="linear")
    gz_n = griddata(pts, fc["forecast_F"].to_numpy(), (gx, gy), method="nearest")
    gz = np.where(np.isnan(gz), gz_n, gz)          # fill edges beyond hull

    vmax = max(2.0, np.ceil(np.abs(fc["forecast_F"]).max() * 2) / 2)
    levels = np.arange(-vmax, vmax + 0.25, 0.5)
    cf = ax.contourf(gx, gy, gz, levels=levels, cmap="RdBu_r", extend="both",
                     transform=ccrs.PlateCarree(), alpha=0.85)
    cl = ax.contour(gx, gy, gz, levels=levels[::2], colors="#444",
                    linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cl, fmt=lambda v: f"{v:+.0f}", fontsize=7)

    # clip everything to the CONUS outline
    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path as MplPath
    from cartopy.mpl.patch import geos_to_path
    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    usa = next(r.geometry for r in shpreader.Reader(shp).records()
               if r.attributes["ADM0_A3"] == "USA")
    from shapely.geometry import box
    usa = usa.intersection(box(-125, 24, -66, 50))     # CONUS only — Alaska in the
    clip = MplPath.make_compound_path(*geos_to_path(usa))  # clip path blows up the layout bbox
    tr = ccrs.PlateCarree()._as_mpl_transform(ax)
    for cs in (cf, cl):
        arts = getattr(cs, "collections", None) or [cs]   # mpl ≥3.10: ContourSet is the artist
        for art in arts:
            art.set_clip_path(clip, transform=tr)

    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.4, edgecolor="#666")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6, edgecolor="#333")
    ax.coastlines("50m", lw=0.6, color="#333")

    sig = fc["roni_F"] != 0.0
    ax.scatter(fc.lon[sig], fc.lat[sig], s=10, c="k", marker="o",
               transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(fc.lon[~sig], fc.lat[~sig], s=12, facecolors="none",
               edgecolors="k", lw=0.6, marker="o",
               transform=ccrs.PlateCarree(), zorder=5)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.03, aspect=45, shrink=0.8)
    cb.set_label("NDJFM 2-m temperature anomaly vs 1991–2020 (°F)", fontsize=10)
    ax.set_title(f"Winter 2026–27 (NDJFM) outlook — RONI + linear trend only — "
                 f"assumed RONI {roni_now:+.1f} °C",
                 fontsize=13, loc="left", pad=8)
    fig.text(0.015, 0.015,
             "Per-city OLS on ERA5 NDJFM means (2°, 1950/51–2025/26): "
             "anomaly = a + b·year + c·RONI (ERSST-derived, Niño-3.4 − tropical mean). "
             "RONI term kept only where significant at 90% (filled dots; open = trend only). "
             "Statistical outlook, not a dynamical forecast.",
             fontsize=7.5, color="#555")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out_path}")  # noqa: T201


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roni", type=float, default=None,
                    help="assumed NDJFM RONI (default: site's latest daily RONI)")
    ap.add_argument("--out-dir", default=str(HERE / "plots"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roni_now = args.roni
    if roni_now is None:
        roni_now = float(json.loads(MANIFEST.read_text())["daily_roni"])
        print(f"assumed NDJFM RONI from site manifest: {roni_now:+.2f}")

    roni = roni_history()
    print(f"RONI winters: {roni.index.min()}–{roni.index.max()} "
          f"(latest complete {roni.index.max()}: {roni.iloc[-1]:+.2f})")
    anom, _ = city_winters()
    print(f"city winters: {anom.shape[0]} × {anom.shape[1]} cities")

    fc = fit_and_forecast(anom, roni, roni_now)
    nsig = int((fc.roni_F != 0).sum())
    print(f"RONI term significant (p<{P_SIG}) at {nsig}/{len(fc)} cities; "
          f"forecast range {fc.forecast_F.min():+.1f} … {fc.forecast_F.max():+.1f} °F")

    fc.round(3).to_csv(out_dir / "roni_winter_forecast_2026.csv", index=False)
    draw_map(fc, roni_now, out_dir / "roni_winter_forecast_2026.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
