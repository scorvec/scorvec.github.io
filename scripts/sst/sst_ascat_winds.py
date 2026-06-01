#!/usr/bin/env python3
"""Daily 10°S–10°N Pacific surface-wind map from Copernicus Marine.

Pulls the gap-filled gridded scatterometer (ASCAT-based) L4 NRT wind product,
averages the latest full UTC day over the equatorial Pacific, and renders wind
SPEED (shaded) with direction VECTORS.

Dataset: cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H (WIND_GLO_PHY_L4_NRT_012_004)
Auth: copernicusmarine credentials (~/.copernicusmarine or env vars
      COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD).

    python sst_ascat_winds.py --out assets/sst/ascat_winds.webp
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine as cm

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"

DATASET = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
LAT = (-10.0, 10.0)
LON = (120.0, 290.0)          # 120°E … 70°W (eastern Indian Ocean → eastern Pacific)
SPEED_MAX = 14.0


def latest_day(out: Path):
    ds = cm.open_dataset(dataset_id=DATASET)
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    ds = ds.sortby("latitude")
    ds = ds[["eastward_wind", "northward_wind"]].sel(
        latitude=slice(*LAT), longitude=slice(*LON))
    # the last few daily slots are placeholders not yet filled (NRT lag), so walk
    # back to the most recent day that actually has data.
    last = pd.to_datetime(ds.time.values[-1]).normalize()
    for day in pd.date_range(last, periods=8, freq="-1D"):
        dsd = ds.sel(time=str(day.date())).mean("time").compute()   # daily mean
        if np.isfinite(dsd["eastward_wind"].values).any():
            return dsd, day
    raise SystemExit("no populated day found in the last 8 days of the L4 product")


def plot(dsd: xr.Dataset, day: pd.Timestamp, out: Path):
    lon = dsd.longitude.values
    lat = dsd.latitude.values
    u = dsd["eastward_wind"].values
    v = dsd["northward_wind"].values
    spd = np.hypot(u, v)

    proj = ccrs.PlateCarree(central_longitude=180)
    PC = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(13, 3.4), subplot_kw=dict(projection=proj))
    ax.set_extent([LON[0], LON[1], LAT[0], LAT[1]], crs=PC)

    # levels start at 1 m/s so calm areas (<1) stay unfilled → white background
    cf = ax.contourf(lon, lat, spd, levels=np.arange(1.0, SPEED_MAX + .01, 1.0),
                     cmap="YlGnBu", extend="max", transform=PC)
    # sparser (~every 2.5°), smaller arrows for legibility
    s = max(1, int(round(2.5 / float(np.diff(lon).mean()))))
    q = ax.quiver(lon[::s], lat[::s], u[::s, ::s], v[::s, ::s], transform=PC,
                  scale=460, width=0.0012, color="0.15", zorder=4)
    ax.quiverkey(q, 0.92, 1.18, 10, "10 m s⁻¹", labelpos="E", fontproperties={"size": 8})

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#d9d6cf", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#555", linewidth=0.4, zorder=3)
    ax.axhline(0, color="k", lw=0.5, ls=":", zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}

    ax.set_title(f"Equatorial Pacific surface wind (10°S–10°N) — {day:%Y-%m-%d}\n"
                 f"speed (shaded) + direction · gap-filled scatterometer (ASCAT) L4",
                 fontsize=11, fontweight="bold", loc="left")
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.018, pad=0.02, extend="max")
    cb.set_label("wind speed (m s⁻¹)", fontsize=8); cb.ax.tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({day:%Y-%m-%d}; mean speed {np.nanmean(spd):.1f}, "
          f"max {np.nanmax(spd):.1f} m/s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ASSETS / "ascat_winds.webp"))
    args = ap.parse_args()
    dsd, day = latest_day(Path(args.out))
    plot(dsd, day, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
