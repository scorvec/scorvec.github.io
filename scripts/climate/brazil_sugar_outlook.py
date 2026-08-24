#!/usr/bin/env python3
"""
Sugar-crop weather outlook for the 2026/27 Brazil summer.

Center-South cane belt (SP/N-Paraná/Triângulo, ~90% of output): NDJFM is
the vegetative development window for the 2027/28 crop, while Nov–Dec is
the tail of the 2026/27 crush — rain there halts harvesters and dilutes
ATR. Northeast coastal belt: harvest runs Sep–Mar in-season.

Panels:
 1. C3S 10-system MME precip anomaly map (Nov–Jan) + belt boxes
 2. CS belt Nov–Dec precip: history vs pooled C3S members (crush tail)
 3. CS belt NDJ precip: history vs members (2027/28 crop development)
 4. CS belt NDJFM temperature: history vs v2 forecast (heat/evap stress)

Output: ~/brazil_sugar_outlook.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/c3s/scripts"))
import brazil_summer_fcst_v2 as v2                           # noqa: E402
import f4_lib as F                                           # noqa: E402

ERA5P = HERE.parent / "sst" / "data" / "era5_precip_mon.nc"
CS = dict(lat=(-24.0, -19.5), lon=(-53.0, -46.0))
NE = dict(lat=(-10.0, -7.0), lon=(-37.0, -34.5))
STEP = {11: 3, 12: 4, 1: 5}
ANALOGS = {1982: "82/83", 1997: "97/98", 2015: "15/16", 2023: "23/24"}


def belt_avg(da, box):
    d = da.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
    if d["lat"].size == 0:
        d = da.sel(lat=slice(box["lat"][1], box["lat"][0]),
                   lon=slice(*box["lon"]))
    w = np.cos(np.deg2rad(d["lat"]))
    return d.weighted(w).mean(("lat", "lon"))


def era5_series(path, var_guess, months, box):
    ds = xr.open_dataset(path)
    var = var_guess if var_guess in ds else list(ds.data_vars)[0]
    da = ds[var].sortby("lat")
    da = da.assign_coords(lon=(((da["lon"] + 180) % 360) - 180)).sortby("lon")
    t = pd.DatetimeIndex(da["time"].values).to_period("M").to_timestamp()
    da = da.assign_coords(time=t)
    if var != "t2m" and float(da.max()) < 1.0:
        da = da * 1000.0
    s = belt_avg(da, box).to_series()
    out = {}
    for y in range(1950, 2026):
        stamps = [pd.Timestamp(y if m >= 8 else y + 1, m, 1) for m in months]
        v = [s.get(x, np.nan) for x in stamps]
        if np.isfinite(v).sum() == len(months):
            out[y] = float(np.mean(v))
    return pd.Series(out)


def c3s_members(var, months, box):
    """Pooled member belt means: anomaly vs own hindcast clim (mm/day or °C),
    plus each system's hindcast-clim belt value count."""
    pool = []
    for mdl in F.models_present("sa_fc"):
        ds = F.load_sa(mdl)
        da_fc = ds[f"fc_{var}"].rename(latitude="lat", longitude="lon")
        da_hc = ds[f"hc_{var}"].rename(latitude="lat", longitude="lon")
        steps = [STEP[m] for m in months]
        fcb = belt_avg(da_fc.isel(step=steps), box).mean("step")
        hcb = belt_avg(da_hc.isel(step=steps), box).mean("step").mean("sample")
        pool.extend((fcb - hcb).values.tolist())
    return np.array(pool)


def mme_precip_map(lat_t, lon_t):
    out = []
    for mdl in F.models_present("sa_fc"):
        ds = F.load_sa(mdl)
        a = (ds["fc_tp"].isel(step=[3, 4, 5]).mean(("number", "step"))
             - ds["hc_tp"].isel(step=[3, 4, 5]).mean(("sample", "step")))
        a = a.rename(latitude="lat", longitude="lon").sortby("lat")
        out.append(a.interp(lat=lat_t, lon=lon_t).values)
    return np.nanmean(out, axis=0)


def hist_panel(ax, hist, members, title, xlabel, note, hist_clim=None):
    ax.hist(hist.values, bins=22, color="#b8cbe0", edgecolor="#5b7ea6",
            lw=0.5, density=True, label="ERA5 1950/51–2025/26")
    n10 = float(hist.iloc[-10:].mean())
    ax.axvline(n10, color="#555", lw=1.0, ls="--")
    for y, lb in ANALOGS.items():
        if y in hist.index:
            ax.axvline(hist[y], color="#e8890c", lw=1.0, alpha=0.8)
            ax.text(hist[y], ax.get_ylim()[1] * 0.97, lb, rotation=90,
                    fontsize=7, color="#a35f05", ha="right", va="top")
    if members is not None:
        base = float(hist.loc[1993:2016].mean()) if hist_clim is None \
            else hist_clim
        mem_abs = members + base
        ax.hist(mem_abs, bins=26, density=True, histtype="step",
                color="#7a5ba6", lw=1.6, label="C3S members (abs.)")
        ax.axvline(float(np.mean(mem_abs)), color="#5b3e8f", lw=2.2,
                   label=f"C3S MME {np.mean(mem_abs):.2f}")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks([])
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8,
            color="#333", style="italic")
    ax.legend(fontsize=7.5, loc="upper right")


def main() -> int:
    d = v2.compute()
    lat_e, lon_e = d["lat"], d["lon"]

    fig = plt.figure(figsize=(17.5, 9.5), dpi=150)

    # 1: precip anomaly map + belts
    ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax.set_extent(v2.EXT, crs=ccrs.PlateCarree())
    pm = mme_precip_map(lat_e, lon_e)
    lv = np.linspace(-2.0, 2.0, 17)
    cf = ax.contourf(lon_e, lat_e, pm, levels=lv, cmap="BrBG",
                     extend="both", transform=ccrs.PlateCarree())
    for box, cc, lb in ((CS, "#d9402a", "Center-South belt"),
                        (NE, "#2b6fd6", "NE coastal belt")):
        ax.add_patch(mpatches.Rectangle(
            (box["lon"][0], box["lat"][0]),
            box["lon"][1] - box["lon"][0], box["lat"][1] - box["lat"][0],
            fill=False, edgecolor=cc, lw=2.0,
            transform=ccrs.PlateCarree(), label=lb))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                   edgecolor="#333")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25,
                   edgecolor="#888")
    ax.coastlines("50m", lw=0.6, color="#333")
    cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("mm/day", fontsize=9)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("C3S MME precip anomaly, Nov–Jan", loc="left",
                 fontsize=10.5)

    # 2: CS harvest tail Nov–Dec precip
    h_nd = era5_series(ERA5P, "tp", [11, 12], CS)
    m_nd = c3s_members("tp", [11, 12], CS)
    hist_panel(fig.add_subplot(2, 2, 2), h_nd, m_nd,
               "Center-South belt: Nov–Dec rain — 2026/27 crush tail",
               "Nov–Dec mean precip (mm/day)",
               "wet = harvester stoppages,\ndiluted ATR at season end")

    # 3: CS development NDJ precip
    h_ndj = era5_series(ERA5P, "tp", [11, 12, 1], CS)
    m_ndj = c3s_members("tp", [11, 12, 1], CS)
    hist_panel(fig.add_subplot(2, 2, 3), h_ndj, m_ndj,
               "Center-South belt: Nov–Jan rain — 2027/28 crop development",
               "Nov–Jan mean precip (mm/day)",
               "dry = smaller 2027/28 cane crop\n(moisture drives tonnage)")

    # 4: CS temp NDJFM (v2 forecast)
    ds5 = xr.open_dataset(v2.v1.ERA5)
    t2 = ds5["t2m"].sortby("lat")
    t2 = t2.assign_coords(lon=(((t2["lon"] + 180) % 360) - 180)).sortby("lon")
    tt = pd.DatetimeIndex(t2["time"].values)
    h_t = era5_series(v2.v1.ERA5, "t2m", [11, 12, 1, 2, 3], CS)
    la = lat_e
    fcs = []
    for m in v2.MONTHS:
        fld = xr.DataArray(d["final_abs"][m],
                           coords=dict(lat=lat_e, lon=lon_e),
                           dims=("lat", "lon"))
        fcs.append(float(belt_avg(fld, CS)))
    fc_t = float(np.mean(fcs))
    ax4 = fig.add_subplot(2, 2, 4)
    hist_panel(ax4, h_t, None,
               "Center-South belt: NDJFM temperature",
               "NDJFM mean t2m (°C)",
               "heat = evapotranspiration stress,\nhigher irrigation demand")
    ax4.axvline(fc_t, color="#d9402a", lw=2.5)
    rank = int((h_t.values >= fc_t).sum()) + 1
    ax4.text(fc_t - 0.04, ax4.get_ylim()[1] * 0.55,
             f"v2 forecast {fc_t:.1f} °C\n(#{rank}/{len(h_t) + 1})  ",
             color="#d9402a", fontsize=9, fontweight="bold", ha="right")

    fig.suptitle("Sugar-crop weather outlook — Brazil 2026/27 summer",
                 fontsize=13.5, x=0.02, ha="left")
    fig.text(0.01, 0.005,
             "CS belt 19.5–24°S, 46–53°W (≈90% of cane). C3S member "
             "distributions anchored to the ERA5 1993–2016 belt mean; "
             "orange lines = strong-Niño analog summers.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    out = Path.home() / "brazil_sugar_outlook.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print(f"wrote {out}")
    print(f"CS NDJFM temp forecast {fc_t:.2f} (rank {rank}); "
          f"ND precip MME anom {m_nd.mean():+.2f} mm/day; "
          f"NDJ {m_ndj.mean():+.2f} mm/day")
    return 0


if __name__ == "__main__":
    sys.exit(main())
