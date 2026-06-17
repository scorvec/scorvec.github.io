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
import json
import os
import shutil
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
import copernicusmarine as cm

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
ANIM_DIR = ASSETS / "anim" / "ascat"               # committed daily image frames (animator)
MANIFEST = ASSETS / "anim" / "ascat_manifest.json"
DATA_DIR = HERE / "data" / "ascat"                 # local-only daily u/v field archive
KEEP_DAYS = 120                                    # rolling window for frames + data

DATASET = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
LAT = (-10.0, 10.0)
LON = (120.0, 290.0)          # 120°E … 70°W (eastern Indian Ocean → eastern Pacific)
SPEED_MAX = 14.0


def _prune(d: Path, pattern: str) -> None:
    """Drop dated files (<YYYY-MM-DD>.*) older than KEEP_DAYS."""
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=KEEP_DAYS)
    for f in d.glob(pattern):
        try:
            if pd.to_datetime(f.stem) < cutoff:
                f.unlink()
        except Exception:                          # noqa: BLE001 — skip non-dated names
            pass


def archive_data(dsd: xr.Dataset, day: pd.Timestamp) -> None:
    """Keep the daily-mean equatorial u/v field for later analysis (local, gitignored)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / f"{day:%Y-%m-%d}.nc"
    if not p.exists():
        enc = {v: {"zlib": True, "complevel": 5} for v in dsd.data_vars}
        dsd.to_netcdf(p, encoding=enc)                  # ~5–10× smaller than raw
        print(f"  archived data {p.name} ({p.stat().st_size/1e6:.1f} MB)")
    _prune(DATA_DIR, "*.nc")


def build_manifest() -> None:
    """Rebuild the animator manifest from the dated frame archive (oldest→newest), pruned."""
    _prune(ANIM_DIR, "*.webp")
    dated = []
    for f in sorted(ANIM_DIR.glob("*.webp")):
        try:
            dated.append((pd.to_datetime(f.stem), f))
        except Exception:                          # noqa: BLE001
            pass
    dated.sort()
    frames = [{"idx": i, "file": f.name, "date": f"{d:%Y-%m-%d}", "label": f"{d:%a %b %d, %Y}"}
              for i, (d, f) in enumerate(dated)]
    manifest = {"ver": int(time.time()), "days": len(frames),
                "regions": {"ascat": {
                    "label": "Equatorial Pacific surface wind (10°S–10°N) — gap-filled scatterometer (ASCAT) L4",
                    "n_frames": len(frames), "frames": frames}}}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest))
    print(f"  manifest: {len(frames)} frames ({frames[0]['date']}…{frames[-1]['date']})" if frames
          else "  manifest: no frames")


def daily_means(n_days: int):
    """Most recent `n_days` populated daily-mean u/v fields (newest first). The last
    few NRT slots are placeholders, so days with no finite data are skipped."""
    ds = cm.open_dataset(dataset_id=DATASET)
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude").sortby("latitude")
    ds = ds[["eastward_wind", "northward_wind"]].sel(
        latitude=slice(*LAT), longitude=slice(*LON))
    first = pd.to_datetime(ds.time.values[0]).normalize()
    last = pd.to_datetime(ds.time.values[-1]).normalize()
    out = []
    for day in pd.date_range(last, periods=n_days + 10, freq="-1D"):
        if day < first:                                 # past the start of the NRT window
            break
        try:
            dsd = ds.sel(time=str(day.date())).mean("time").compute()   # daily mean
        except Exception:                               # noqa: BLE001
            continue
        # Require substantial coverage, not just one finite point: the newest NRT slot is
        # often a partial swath (a sliver of finite data) that passes an .any() check but
        # renders as a near-empty, collapsed map. Demand the gap-filled L4 be mostly populated.
        if np.isfinite(dsd["eastward_wind"].values).mean() > 0.5:
            out.append((dsd, day))
        if len(out) >= n_days:
            break
    if not out:
        raise SystemExit("no populated day found in the recent L4 product")
    return out


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
    ap.add_argument("--backfill", type=int, default=1,
                    help="render the last N populated days into the archive (bootstrap)")
    args = ap.parse_args()

    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    pairs = daily_means(max(1, args.backfill))          # newest first
    for dsd, day in pairs:
        archive_data(dsd, day)                          # keep the daily field (local)
        frame = ANIM_DIR / f"{day:%Y-%m-%d}.webp"
        if not frame.exists():                          # idempotent — only render new days
            plot(dsd, day, frame)
    newest = pairs[0][1]
    shutil.copy2(ANIM_DIR / f"{newest:%Y-%m-%d}.webp", Path(args.out))   # latest static image
    build_manifest()                                    # refresh the animator manifest
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
