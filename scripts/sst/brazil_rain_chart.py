#!/usr/bin/env python3
"""Brazil rainfall forecast products from the corrected-IMERG pipeline.

1. rain_fans.webp — per-basin corrected-IMERG daily rain vs harmonic
   norm with the bias-corrected AIFS+IFS ensemble fan (engine's
   out["rain"] quantiles).
2. forecast_map.webp — blended skill-corrected most-likely total over
   the common ~15-day window (per-basin/lead bias factors rasterized,
   models blended by verified weights) + % of normal vs a gridded
   harmonic climatology fitted from the corrected IMERG cache.

    python scripts/sst/brazil_rain_chart.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                   # noqa: E402
from brazil_model import basin_weights, MAJORS              # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim      # noqa: E402
from matplotlib.path import Path as MplPath                 # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "brazil_hydro"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
CORR_NPZ = PRIV / "raw" / "imerg_gauge_corr.npz"
MODELS_JSON = PRIV / "out" / "brazil_models.json"
VERIF = PRIV / "out" / "fcst_verif.json"
FAN_JSON = REPO / "brazil_hydro" / "data" / "ena_forecast.json"
BASINS_GJ = PRIV / "out" / "brazil_basins.geojson"
GRIB_DIR = REPO / "scripts" / "mjo" / "data" / "aifs"
SITE = REPO / "brazil_hydro"

NAVY = "#13273d"
RAIN_COLS = ["#f6f7f5", "#d9edcf", "#a5d99b", "#57b86b", "#1f9e89",
             "#2380b9", "#20539c", "#5b3f9e", "#93357f", "#c2185b"]
RAIN_CMAP = ListedColormap(RAIN_COLS)
RAIN_CMAP.set_over("#7a1240")
LEV_15D = [0, 5, 10, 25, 50, 75, 100, 150, 200, 300, 450]
PCT_COLS = ["#7a4a12", "#a8702a", "#cd9d57", "#e8cf9e", "#f6f5f0",
            "#c9e7c2", "#7cc87c", "#2e9e4f", "#1d6fb8", "#6a3d9a"]
PCT_CMAP = ListedColormap(PCT_COLS)
PCT_CMAP.set_over("#3f1f66")
LEV_PCT = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 1000]
EXT = [-75.5, -33.5, -34.5, 6.0]
BANDS = [(1, 3), (4, 7), (8, 15)]


def band_of(lead):
    return next((i for i, (a, b) in enumerate(BANDS) if a <= lead <= b),
                len(BANDS) - 1)


def rain_fans(fan):
    tc = json.loads(TRUTH.read_text())
    dm = json.loads(MODELS_JSON.read_text())["params"] if MODELS_JSON.exists() else {}
    rdates = [datetime.strptime(d, "%Y%m%d") for d in tc["dates"]]
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in rdates])
    when = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = when - timedelta(days=100)
    t1 = when + timedelta(days=18)
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 12.5), sharex=True)
    for ax, b in zip(axes.flat, MAJORS):
        if b not in tc:
            ax.set_axis_off()
            continue
        obs = np.array(tc[b], float)
        cl = (np.array(dm[b]["clim365_mmday"], float)[doy - 1]
              if b in dm else np.full(len(obs), np.nan))
        m = [i for i, d in enumerate(rdates) if d >= t0]
        td = [rdates[i] for i in m]
        ax.bar(td, obs[m], width=1.0, color="#a8c6e2", lw=0)
        k7 = np.convolve(np.where(np.isfinite(obs), obs, 0),
                         np.ones(7) / 7, "full")[:len(obs)]
        ax.plot(td, k7[m], color="#c62828", lw=1.4, label="7-day mean")
        ax.plot(td, cl[m], color="#1f4e8c", lw=1.4, label="seasonal norm")
        if fan is not None and b in fan.get("rain", {}):
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            q = fan["rain"][b]
            ok = [j for j, v in enumerate(q["p50"]) if v is not None]
            fdo = [fd[j] for j in ok]
            ax.fill_between(fdo, [q["p10"][j] for j in ok],
                            [q["p90"][j] for j in ok], color="#e08214",
                            alpha=0.22, lw=0)
            ax.fill_between(fdo, [q["p25"][j] for j in ok],
                            [q["p75"][j] for j in ok], color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(fdo, [q["p50"][j] for j in ok], color="#b35806",
                    lw=1.5, ls="--", label="bias-corrected ens forecast")
        wk = np.nanmean(obs[m][-7:])
        wkc = np.nanmean(cl[m][-7:])
        ttl = b
        if np.isfinite(wk) and wkc > 0.2:
            ttl += f"  ·  last 7 d: {100 * wk / wkc:.0f}% of norm"
        ax.set_title(ttl, fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlim(t0, t1)
        ax.set_ylim(bottom=0)
        ax.axvline(when, color="0.55", lw=0.7, ls=":")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("mm/day", fontsize=7.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if b == MAJORS[0]:
            ax.legend(fontsize=6.6, loc="upper left")
    fig.suptitle("Brazil basin rainfall — gauge-corrected IMERG vs the 25-yr "
                 "seasonal norm, with the bias-corrected AIFS-ENS + IFS-ENS "
                 "ensemble fan", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(SITE / "rain_fans.webp", dpi=110)
    plt.close(fig)
    print("wrote brazil_hydro/rain_fans.webp", flush=True)


def grid_clim():
    """25-yr (2001-2025) IMERG-Final harmonic coefficients, full grid."""
    import xarray as xr
    return xr.open_dataset(CLIM_NC)["coef"].values, "2001\u20132025"


def forecast_map(verif):
    import xarray as xr
    from scipy.ndimage import gaussian_filter
    latest = {}
    for f in sorted(glob.glob(str(GRIB_DIR / "*_*z.pf.tp.grib2"))):
        m = re.match(r"(aifs|ifs)_(\d{8})_(\d{2})z", Path(f).name)
        if m:
            latest[m.group(1)] = (m.group(2), m.group(3))
    if not latest:
        return

    def daily_fields(model, date, hh):
        parts = []
        for typ in (("cf", "pf") if model == "aifs" else ("pf",)):
            pth = GRIB_DIR / f"{model}_{date}_{hh}z.{typ}.tp.grib2"
            if not pth.exists():
                continue
            ds = xr.open_dataset(pth, engine="cfgrib", chunks={},
                                 backend_kwargs={"filter_by_keys":
                                                 {"shortName": "tp"},
                                                 "indexpath": ""})
            da = ds["tp"]
            if da.attrs.get("units", "").strip() in ("m", "metre", "metres"):
                da = da * 1000.0
            if da.longitude.values.max() > 180:
                da = da.assign_coords(longitude=(da.longitude + 180) % 360
                                      - 180)
            da = da.sortby("longitude").sortby("latitude")
            da = da.sel(longitude=slice(EXT[0], EXT[1]),
                        latitude=slice(EXT[2], EXT[3]))
            if "number" not in da.dims:
                da = da.expand_dims("number")
            parts.append(da.compute())
        da = parts[0] if len(parts) == 1 else xr.concat(parts, dim="number")
        steps_h = (da.step.values / np.timedelta64(1, "h")).astype(int)
        order = np.argsort(steps_h)
        steps_h = steps_h[order]
        v = da.isel(step=order).mean("number")\
              .transpose("step", "latitude", "longitude").values
        init = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
        bh = np.concatenate([[0], steps_h])
        bv = np.concatenate([np.zeros((1,) + v.shape[1:]), v], axis=0)
        days_, buckets = [], []
        for k in range(len(bh) - 1):
            t0 = init + np.timedelta64(int(bh[k]), "h")
            if bh[k + 1] - bh[k] == 24 and t0 == t0.astype("datetime64[D]"):
                days_.append(t0.astype("datetime64[D]"))
                buckets.append(np.clip(bv[k + 1] - bv[k], 0, None))
        return days_, np.array(buckets), da.longitude.values, da.latitude.values

    fields = {m: daily_fields(m, d, h) for m, (d, h) in latest.items()}
    sets = [set(f[0]) for f in fields.values()]
    common = sorted(set.intersection(*sets) if len(sets) > 1 else sets[0])
    if len(common) < 5:
        return
    lons, lats = list(fields.values())[0][2], list(fields.values())[0][3]
    W = basin_weights(lons, lats, set(MAJORS))
    masks = {b: w > 0 for b, w in W.items()}
    init0 = min(np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
                for d, _ in latest.values())
    w_ai = float(np.mean([verif["weight_aifs"][b][bi]
                          for b in verif["weight_aifs"]
                          for bi in verif["weight_aifs"][b]]))
    blended = np.zeros((len(lats), len(lons)))
    for model, (days_, buckets, _, _) in fields.items():
        w = w_ai if model == "aifs" else 1 - w_ai
        acc = np.zeros_like(blended)
        for dd, bucket in zip(days_, buckets):
            if dd not in common:
                continue
            lead = max(int((dd - init0).astype(int)), 1)
            bi = str(band_of(lead))
            Fr = np.full(blended.shape, np.nan)
            for b, msk in masks.items():
                Fr[msk] = verif["bias_factors"][b][bi][model]
            Fr = np.where(np.isfinite(Fr), Fr, np.nanmean(
                [verif["bias_factors"][b][bi][model]
                 for b in verif["bias_factors"]]))
            acc += bucket * gaussian_filter(Fr, 1.5)
        blended += w * acc

    # corrected clim over the same days on the fine grid -> model grid
    ml, mt = IP._grid_axes()
    flons = np.sort(IP._LON[ml])
    flats = np.sort(IP._LAT[mt])
    coef, clim_years = grid_clim()
    doys = np.array([min(d.item().timetuple().tm_yday, 365) for d in common])
    clim_sum = np.sum([np.clip(eval_clim(coef, int(dy)), 0, None)
                       for dy in doys], axis=0)
    Ffine = np.ones((len(flats), len(flons)))
    if CORR_NPZ.exists():
        zc = np.load(CORR_NPZ)
        if zc["F"].shape == Ffine.shape:
            Ffine = zc["F"]
    ii = np.array([int(np.argmin(np.abs(flats - la))) for la in lats])
    jj = np.array([int(np.argmin(np.abs(flons - lo))) for lo in lons])
    clm = (clim_sum * Ffine)[np.ix_(ii, jj)]     # gauge correction at eval
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(clm > 2.0, 100.0 * blended / clm, np.nan)

    gj = json.loads(BASINS_GJ.read_text())
    rings = {ft["properties"]["basin"]: ft["geometry"]["coordinates"]
             for ft in gj["features"] if ft["properties"]["basin"] in MAJORS}
    span = (f"{common[0].astype(object):%b %d} – "
            f"{common[-1].astype(object):%b %d}")
    fig = plt.figure(figsize=(14.5, 8.6))
    for k, (field, cmap, lev, ttl, msk_ocean) in enumerate([
            (blended, RAIN_CMAP, LEV_15D,
             f"Skill-corrected most-likely total (mm) · {span}", False),
            (pct, PCT_CMAP, LEV_PCT,
             f"Percent of normal (IMERG {clim_years} climatology, "
             "gauge-corrected)", True)]):
        ax = fig.add_axes([0.035 + k * 0.485, 0.09, 0.45, 0.80],
                          projection=ccrs.PlateCarree())
        nrm = BoundaryNorm(lev, cmap.N)
        pm = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=nrm,
                           shading="nearest", transform=ccrs.PlateCarree(),
                           rasterized=True)
        if msk_ocean:
            ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                           facecolor="#e6ebf0", zorder=3)
        ax.coastlines(resolution="50m", lw=0.8, color="#2b2b2b", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                       edgecolor="#4a4a4a", zorder=4)
        for b, polys in rings.items():
            for p in polys:
                arr = np.array(p[0])
                ax.plot(arr[:, 0], arr[:, 1], color=NAVY, lw=1.0,
                        transform=ccrs.PlateCarree(), zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground="white")])
        ax.set_extent(EXT, ccrs.PlateCarree())
        ax.set_title(ttl, fontsize=11, fontweight="bold", loc="left")
        cax = fig.add_axes([0.06 + k * 0.485, 0.045, 0.4, 0.016])
        cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="max")
        cb.set_ticks(lev[1:-1] if cmap is PCT_CMAP else lev[:-1])
        cb.ax.tick_params(labelsize=7.5)
    inits_s = " · ".join(f"{m.upper()} {d} {h}Z"
                         for m, (d, h) in latest.items())
    fig.suptitle(f"Brazil — next {len(common)} days rainfall, blended "
                 f"AIFS-ENS + IFS-ENS ensemble means, per-basin/lead bias "
                 f"factors from the corrected-IMERG verification · {inits_s}",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.savefig(SITE / "forecast_map.webp", dpi=115)
    plt.close(fig)
    print("wrote brazil_hydro/forecast_map.webp", flush=True)


def main() -> int:
    fan = json.loads(FAN_JSON.read_text()) if FAN_JSON.exists() else None
    verif = json.loads(VERIF.read_text())
    rain_fans(fan)
    forecast_map(verif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
