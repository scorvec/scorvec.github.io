#!/usr/bin/env python3
"""Cold-spell probability from C3S DAILY ensemble members.

P(at least one run of >= SPELL_DAYS consecutive days with daily-mean T2m
below normal - 1 sigma) per gridpoint, through the winter window — the
duration-aware cold risk a monthly mean cannot express.

Data: seasonal-original-single-levels (the full member-level sub-daily
fields), 2m temperature, winter-window leadtime_hours only (6-hourly,
averaged to daily means per member). ECMWF first; UKMO/ECCC join as their
downloads land.

Bias handling, in two documented steps:
  1. model drift: subtract the model's monthly hindcast climatology for that
     valid month & lead (from the c3s_t2m monthly cache), interpolated to the
     model grid — the same bias+drift removal the monthly products use;
  2. daily reference: add back the ERA5 1993-2016 monthly mean, then compare
     against the ERA5 1991-2020 DAILY normal and sigma (per calendar day,
     +/-7-day window smoothing) from the local WB2 daily store (1.5 deg NH).
Net: member daily anomaly vs the observed daily normal, with model drift
removed at monthly scale. Spells are counted on that anomaly < -1 sigma.

    python c3s_coldspell.py --fetch --issue 202608     # queue daily downloads
    python c3s_coldspell.py --issue 202608             # build product
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3s_nino34 as c3s
from c3s_t2m_winter import (DATA as T2M_DATA, BOX, open_fields,
                            month_samples, era5_monthly)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "c3s_daily"
ASSETS = c3s.ASSETS
WB2_T2M = Path("~/era5_store/wb2_1p5_daily/t2m").expanduser()

SPELL_DAYS = 5
SIGMA = 1.0
DAILY_MODELS = [("ecmwf", "51", "ECMWF SEAS5"),
                ("ukmo", "610", "UKMO GloSea6"),
                ("eccc", "4", "ECCC GEM5-NEMO"),
                ("eccc", "5", "ECCC CanESM5")]


def winter_hours(issue: str):
    """6-hourly leadtime_hours covering Dec 1 .. Feb 28 (clipped to the
    integration length) for this issue's init date."""
    init = pd.Timestamp(f"{issue[:4]}-{issue[4:]}-01")
    dec1 = pd.Timestamp(f"{init.year}-12-01")
    end = pd.Timestamp(f"{init.year + 1}-03-01")
    h0 = int((dec1 - init).total_seconds() // 3600)
    h1 = min(int((end - init).total_seconds() // 3600), 5160)
    return [str(h) for h in range(h0, h1, 6)], dec1


def _fetch_daily(centre, system, label, year, month, hours, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    req = {"originating_centre": centre, "system": system,
           "variable": "2m_temperature",
           "year": year, "month": month, "day": "01",
           "leadtime_hour": hours, "area": BOX, "data_format": "grib"}
    try:
        c3s._client().retrieve("seasonal-original-single-levels", req, str(dest))
        print(f"{label} {year}: ✓ {dest.stat().st_size/1e6:.0f} MB", flush=True)
        return True
    except Exception as e:                                # noqa: BLE001
        print(f"{label} {year}: failed ({str(e)[:100]})", file=sys.stderr)
        return False


RENEW_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
              "surface_solar_radiation_downwards"]


def _fetch_daily_renew(centre, system, label, year, month, hours, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    req = {"originating_centre": centre, "system": system,
           "variable": RENEW_VARS,
           "year": year, "month": month, "day": "01",
           "leadtime_hour": hours, "area": BOX, "data_format": "grib"}
    try:
        c3s._client().retrieve("seasonal-original-single-levels", req, str(dest))
        print(f"{label} renew {year}: ✓ {dest.stat().st_size/1e6:.0f} MB", flush=True)
        return True
    except Exception as e:                                # noqa: BLE001
        print(f"{label} renew {year}: failed ({str(e)[:90]})", file=sys.stderr)
        return False


def fetch(issue: str, hindcast: bool = False) -> None:
    """Forecast members; with hindcast=True also the 1993-2016 daily hindcast
    (winter window only) — needed to correct the daily stream against its OWN
    climatology: the C3S monthly and daily streams disagree by a spatially
    varying ~3 C (found 2026-08-11), so monthly-climo correction of daily
    members is invalid."""
    DATA.mkdir(parents=True, exist_ok=True)
    hours, _ = winter_hours(issue)
    print(f"winter window: {len(hours)} 6-hourly steps")
    for centre, system, label in DAILY_MODELS:
        _fetch_daily(centre, system, label, issue[:4], issue[4:], hours,
                     DATA / f"daily_{centre}_{system}_{issue}.grib")
        if centre != "ukmo":
            _fetch_daily_renew(centre, system, label, issue[:4], issue[4:], hours,
                               DATA / f"renew_{centre}_{system}_{issue}.grib")
        if hindcast:
            if centre == "ukmo":          # lagged system: day-01 gives a 2-member sliver
                continue
            for y in range(1993, 2017):
                _fetch_daily(centre, system, label, str(y), issue[4:], hours,
                             DATA / f"dailyhc_{centre}_{system}_{y}{issue[4:]}.grib")
                _fetch_daily_renew(centre, system, label, str(y), issue[4:], hours,
                                   DATA / f"renewhc_{centre}_{system}_{y}{issue[4:]}.grib")


def era5_daily_climo(y0=1991, y1=2020, halfwin=7):
    """(doy_mean, doy_std) DataArrays (dayofyear, lat, lon) from the WB2 daily
    store, +/-halfwin-day window pooling."""
    das = []
    for y in range(y0, y1 + 1):
        f = WB2_T2M / f"t2m_{y}.nc"
        das.append(xr.open_dataset(f)["t2m"])
    da = xr.concat(das, dim="time").transpose("time", "latitude", "longitude")
    doy = da.time.dt.dayofyear
    mean = np.full((366,) + da.shape[1:], np.nan, np.float32)
    std = np.full_like(mean, np.nan)
    vals = da.values
    doyv = doy.values
    for d in range(1, 367):
        lo, hi = d - halfwin, d + halfwin
        sel = ((doyv - d + 183) % 366 - 183)
        m = np.abs(sel) <= halfwin
        mean[d - 1] = vals[m].mean(axis=0)
        std[d - 1] = vals[m].std(axis=0)
    coords = dict(dayofyear=np.arange(1, 367),
                  latitude=da.latitude.values, longitude=da.longitude.values)
    return (xr.DataArray(mean, coords=coords, dims=("dayofyear", "latitude", "longitude")),
            xr.DataArray(std, coords=coords, dims=("dayofyear", "latitude", "longitude")))


def spell_prob(z, thresh=-1.0, days=5):
    """P-per-sample of >= `days` consecutive True along axis 1."""
    cold = z < thresh
    run = np.zeros(cold.shape[0:1] + cold.shape[2:], bool)
    streak = np.zeros_like(run, dtype=np.int16)
    for d in range(cold.shape[1]):
        streak = np.where(cold[:, d], streak + 1, 0)
        run |= streak >= days
    return run


def era5_base_rate(clim_mean, clim_std, thresh=-1.0, days=5, y0=1991, y1=2020):
    """Climatological P(>=1 spell) per gridpoint: same test on 30 observed
    winters (Dec 1 - Feb 28), cached per (thresh, days) variant."""
    tag = f"{abs(thresh):.0f}sig_{days}d"
    cache = DATA / f"era5_spell_base_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["base"]
    winters = []
    for y in range(y0, y1):
        a = xr.open_dataset(WB2_T2M / f"t2m_{y}.nc")["t2m"]
        b = xr.open_dataset(WB2_T2M / f"t2m_{y+1}.nc")["t2m"]
        da = xr.concat([a.sel(time=slice(f"{y}-12-01", None)),
                        b.sel(time=slice(None, f"{y+1}-02-28"))], dim="time")
        da = da.transpose("time", "latitude", "longitude")
        doy = da.time.dt.dayofyear.values
        cm = clim_mean.sel(dayofyear=xr.DataArray(doy, dims="time")).values
        cs = clim_std.sel(dayofyear=xr.DataArray(doy, dims="time")).values
        z = (da.values - cm) / np.maximum(cs, 0.5)
        winters.append(spell_prob(z[None], thresh, days)[0])
    base = np.mean(winters, axis=0)
    np.savez_compressed(cache, base=base)
    return base


def permian_product(panels, tlat, lon_plot, issue):
    """Freeze-off risk for the Texas producing basins: ABSOLUTE thresholds on
    the bias-corrected daily-mean member temperatures (the metric that maps to
    wellhead freeze-offs), zoomed to the southern Plains, plus Midland-point
    member statistics. Uri (Feb 2021) benchmark at Midland: ~7 consecutive
    daily means below 0 °C on this 1.5° grid."""
    jm = np.argmin(np.abs(tlat - 32.0))
    km = np.argmin(np.abs(lon_plot - (-102.0)))
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True,
                             subplot_kw={"projection": ccrs.LambertConformal(
                                 central_longitude=-100, central_latitude=32)})
    metrics = [("P(≥3 consec. days < 0 °C) %", 0.0, 3, [5, 10, 15, 20, 30, 40, 55, 70]),
               ("P(≥5 consec. days < 0 °C) %  [Uri-class]", 0.0, 5, [2, 5, 8, 12, 18, 25, 35, 50]),
               ("P(any day < −5 °C) %", -5.0, 1, [5, 10, 15, 20, 30, 40, 55, 70])]
    stats_txt = []
    for ax, (title, thr, nd, levels) in zip(axes, metrics):
        per = [spell_prob(P["abs_daily"], thr, nd).mean(axis=0) for P in panels.values()]
        fld = np.mean(per, axis=0)
        cf = ax.contourf(lon_plot, tlat, 100 * fld, levels=levels, cmap="PuBu",
                         extend="max", transform=ccrs.PlateCarree())
        ax.set_extent([-108, -93, 25.5, 37.5], ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="0.25")
        ax.add_feature(cfeature.BORDERS, lw=0.5, edgecolor="0.35", facecolor="none")
        ax.add_feature(cfeature.STATES, lw=0.4, edgecolor="0.4", facecolor="none")
        ax.plot(-102.1, 31.9, marker="*", ms=14, color="#c62828",
                transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.02,
                          fraction=0.05, aspect=30)
        cb.ax.tick_params(labelsize=7.5)
        stats_txt.append(f"{title.split(')')[0]}) Midland: {100*fld[jm, km]:.0f}%")
    # Midland member distribution: freezing days + longest run, pooled members
    pool_fd, pool_run = [], []
    for P in panels.values():
        a = P["abs_daily"][:, :, jm, km]
        pool_fd.extend((a < 0).sum(axis=1).tolist())
        streak = np.zeros(a.shape[0], np.int16)
        longest = np.zeros(a.shape[0], np.int16)
        for d in range(a.shape[1]):
            streak = np.where(a[:, d] < 0, streak + 1, 0)
            longest = np.maximum(longest, streak)
        pool_run.extend(longest.tolist())
    pool_fd, pool_run = np.array(pool_fd), np.array(pool_run)
    fig.suptitle(f"Permian / Texas freeze-off risk — daily C3S members, issue "
                 f"{issue[:4]}-{issue[4:]} · Dec–Feb window", fontsize=13.5,
                 fontweight="bold")
    fig.text(0.5, 0.002,
             f"★ Midland gridpoint (1.5°): freezing days DJF median {np.median(pool_fd):.0f} "
             f"(P90 {np.percentile(pool_fd, 90):.0f}) · longest sub-0°C run median "
             f"{np.median(pool_run):.0f} d (P90 {np.percentile(pool_run, 90):.0f} d) · "
             f"P(run ≥5 d, Uri-class) {100*(pool_run >= 5).mean():.0f}% · "
             f"{len(pool_fd)} members · bias-corrected daily means, ERA5-referenced",
             fontsize=8.6, ha="center", color="0.25")
    out = ASSETS / "c3s_permian_freeze.webp"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print("  Midland: " + " · ".join(stats_txt))
    print(f"saved {out}")


def build(issue: str, out: Path):
    hours, dec1 = winter_hours(issue)
    print("ERA5 daily climatology (1991–2020, ±7 d) …", flush=True)
    clim_mean, clim_std = era5_daily_climo()
    glat = clim_mean.latitude.values
    glon = clim_mean.longitude.values
    gsel_lat = (glat >= 16) & (glat <= 72)
    gsel_lon = (glon >= 190) & (glon <= 310)
    tlat, tlon = glat[gsel_lat], glon[gsel_lon]
    cm_t = clim_mean.isel(latitude=gsel_lat, longitude=gsel_lon)
    cs_t = clim_std.isel(latitude=gsel_lat, longitude=gsel_lon)
    print("ERA5 climatological spell base rates …", flush=True)
    bases = {sg: era5_base_rate(clim_mean, clim_std, -sg, SPELL_DAYS)
             [np.ix_(gsel_lat, gsel_lon)] for sg in (1.0, 2.0)}

    panels = {}
    for centre, system, label in DAILY_MODELS:
        dest = DATA / f"daily_{centre}_{system}_{issue}.grib"
        if not dest.exists():
            continue
        print(f"{label}: loading daily members …", flush=True)
        ds = xr.open_dataset(dest, engine="cfgrib", backend_kwargs={"indexpath": ""})
        if "number" in ds.dims and ds.sizes["number"] < 10:
            print(f"  {label}: only {ds.sizes['number']} members (lagged sliver) — skipped")
            ds.close()
            continue
        da = ds[[v for v in ds.data_vars][0]] - 273.15
        vt = pd.to_datetime(np.asarray(ds["valid_time"].values))
        norm = pd.DatetimeIndex(vt).normalize()
        udays = pd.DatetimeIndex(sorted(set(norm)))
        udays = udays[udays >= dec1]
        # daily means, then coarsen to the ERA5 1.5° scoring grid (scoring on
        # the model's finer native grid against 1.5° σ inflates |z|)
        lon_model = da.longitude.values
        lonm360 = np.where(lon_model < 0, lon_model + 360, lon_model)
        da = da.assign_coords(longitude=lonm360).sortby("longitude").sortby("latitude")
        arr = da.transpose("number", "step", "latitude", "longitude")
        daily = []
        for d in udays:
            sel = np.where(norm == d)[0]
            daily.append(arr.isel(step=sel).mean("step"))
        dailyx = xr.concat(daily, dim="day").transpose("number", "day", "latitude", "longitude")
        dailyx = dailyx.interp(latitude=tlat, longitude=tlon)
        daily_v = dailyx.values                        # (mem, day, lat, lon)

        # model monthly drift removal on the scoring grid
        hcp = T2M_DATA / f"hc_{centre}_{system}_{issue[4:]}.grib"
        hc, hvt = open_fields(hcp)
        hlon = hc.longitude.values
        hc = hc.assign_coords(longitude=np.where(hlon < 0, hlon + 360, hlon))
        anom = np.empty_like(daily_v)
        for mth in sorted(set(udays.month)):
            sel = udays.month == mth
            mclim = month_samples(hc, hvt, mth).mean(axis=0)
            mclim = xr.DataArray(mclim, coords=dict(latitude=hc.latitude.values,
                                                    longitude=hc.longitude.values),
                                 dims=("latitude", "longitude")).sortby("latitude")                 .sortby("longitude").interp(latitude=tlat, longitude=tlon).values
            e5m = era5_monthly(1993, 2016, mth, tlat, np.where(tlon > 180, tlon - 360, tlon))
            anom[:, sel] = daily_v[:, sel] - mclim[None, None] + e5m[None, None]

        doy = udays.dayofyear.values
        cm = cm_t.sel(dayofyear=xr.DataArray(doy, dims="day")).transpose(
            "day", "latitude", "longitude").values
        cs = cs_t.sel(dayofyear=xr.DataArray(doy, dims="day")).transpose(
            "day", "latitude", "longitude").values
        z = (anom - (cm - 273.15)) / np.maximum(cs, 0.5)
        runs = {sg: spell_prob(z, -sg, SPELL_DAYS).mean(axis=0) for sg in (1.0, 2.0)}
        panels[label] = dict(p=runs[1.0], p2=runs[2.0], abs_daily=anom,
                             nmem=z.shape[0], ndays=z.shape[1], udays=udays)
        print(f"  {label}: {z.shape[0]} members × {z.shape[1]} days · "
              f"P(1σ)={runs[1.0].mean():.2f} (base {bases[1.0].mean():.2f}) · "
              f"P(2σ)={runs[2.0].mean():.2f} (base {bases[2.0].mean():.2f})")

    if not panels:
        raise SystemExit("no daily data on disk yet")
    combo = np.mean([P["p"] for P in panels.values()], axis=0)
    combo2 = np.mean([P["p2"] for P in panels.values()], axis=0)
    lon_plot = np.where(tlon > 180, tlon - 360, tlon)
    permian_product(panels, tlat, lon_plot, issue)

    n = len(panels) + 6
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.6 * nrow),
                             constrained_layout=True,
                             subplot_kw={"projection": ccrs.LambertConformal(
                                 central_longitude=-100, central_latitude=45)})
    axes = np.atleast_1d(axes).ravel()
    levels = [5, 10, 15, 20, 30, 40, 50, 65, 80]
    lev2 = [1, 2.5, 5, 7.5, 10, 15, 20, 30, 40]
    plots = ([("MM1", combo, "Blues", levels, f"Multi-model · −1σ ≥{SPELL_DAYS}d"),
              ("Base", bases[1.0], "Blues", levels, "ERA5 base rate · −1σ"),
              ("Diff", combo - bases[1.0], "RdBu", np.linspace(-0.4, 0.4, 17),
               "Multi-model − base · −1σ"),
              ("MM2", combo2, "Blues", lev2, f"Multi-model · −2σ ≥{SPELL_DAYS}d"),
              ("Base2", bases[2.0], "Blues", lev2, "ERA5 base rate · −2σ"),
              ("Diff2", combo2 - bases[2.0], "RdBu", np.linspace(-0.15, 0.15, 16),
               "Multi-model − base · −2σ")]
             + [(f"m{i}", P["p"], "Blues", levels, f"{lab} · −1σ  ({P['nmem']} mem)")
                for i, (lab, P) in enumerate(panels.items())])
    for ax, (key, fld, cmap, lev, title) in zip(axes, plots):
        diff = key.startswith("Diff")
        cf = ax.contourf(lon_plot, tlat, 100 * fld,
                         levels=([x * 100 for x in lev] if diff else lev),
                         cmap=cmap, extend="both" if diff else "max",
                         transform=ccrs.PlateCarree())
        ax.set_extent([-168, -52, 17, 71], ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="0.25")
        ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.35", facecolor="none")
        ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="0.55", facecolor="none")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.02,
                          fraction=0.05, aspect=30)
        cb.ax.tick_params(labelsize=7)
    for ax in axes[len(plots):]:
        ax.set_visible(False)
    fig.suptitle(f"Cold-spell probability — ≥{SPELL_DAYS} consecutive days < normal − {SIGMA}σ · "
                 f"daily C3S members, issue {issue[:4]}-{issue[4:]} (Dec 1 → integration end)",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.002,
             "Daily-mean member T2m coarsened to the ERA5 1.5° scoring grid · model drift removed via monthly hindcast climatology · "
             "normal & σ: ERA5 1991–2020 daily (±7 d window) · base rate: identical spell test on the 30 observed winters",
             fontsize=8.3, ha="center", color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=pd.Timestamp.utcnow().strftime("%Y%m"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--hindcast", action="store_true")
    ap.add_argument("--out", default=str(ASSETS / "c3s_winter_coldspell.webp"))
    args = ap.parse_args()
    if args.fetch:
        fetch(args.issue, hindcast=args.hindcast)
        return
    build(args.issue, Path(args.out))


if __name__ == "__main__":
    main()
