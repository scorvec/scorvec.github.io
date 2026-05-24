"""Synoptic map renderer (national + regional zooms).

For a given HRRR cycle, render static PNG maps of:
  - 80m wind speed (m/s), with wind plants overlaid
  - DSWRF surface shortwave radiation (W/m²), with solar plants overlaid

Each variable produces a set of maps for the National view plus several
regional zooms. The full set per variable is configured below in
WIND_REGIONS and SOLAR_REGIONS.

Output:
    assets/synoptic/wind/<region>/F00.png ... F48.png  (overwritten each cycle)
    assets/synoptic/wind/<region>/manifest.json
    assets/synoptic/wind/regions.json
    (same shape for solar)

The viewer (viewer.html) reads regions.json to populate its region
dropdown, then loads the selected region's manifest.json.

Usage:
    python render_maps.py                  # latest extended cycle
    python render_maps.py 2026-05-24 18    # specific cycle
"""
from __future__ import annotations

import argparse
import json
import sys
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from herbie import Herbie

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================================
# Constants and paths
# ============================================================================

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets" / "synoptic"
WIND_DIR = ASSETS / "wind"
SOLAR_DIR = ASSETS / "solar"

WIND_INV_CSV = HERE.parent / "data" / "uswtdb.csv"
SOLAR_INV_CSV = HERE.parent / "solar" / "data" / "uspvdb.csv"

FORECAST_HOURS = list(range(0, 49))

MIN_WIND_PLANT_MW = 10.0
MIN_SOLAR_PLANT_MW = 5.0

PC = ccrs.PlateCarree()


# ============================================================================
# Region definitions
# ============================================================================

@dataclass
class Region:
    """A named geographic region with its bounding box and projection."""
    id: str                       # short slug, used as directory name
    label: str                    # human-readable for the dropdown
    extent: tuple                 # (west, east, south, north) in degrees
    proj_lon: float               # central meridian for LCC
    proj_lat: float = 37.5        # central latitude
    standard_parallels: tuple = (33.0, 45.0)
    figsize: tuple = (13, 7.5)


# Wind regions (national + 7 zoomed)
WIND_REGIONS = [
    Region("national", "National (CONUS)",
           extent=(-125, -66, 24, 50), proj_lon=-96.0, figsize=(13, 7.5)),
    Region("ercot", "ERCOT (Texas)",
           extent=(-107, -93, 25.5, 37.0), proj_lon=-99.5,
           standard_parallels=(28, 36), figsize=(10, 8.5)),
    Region("spp", "SPP (Plains)",
           extent=(-106, -89.5, 31, 49), proj_lon=-97.5,
           standard_parallels=(34, 46), figsize=(10, 10)),
    Region("miso", "MISO (Midwest)",
           extent=(-104, -82, 29, 49.5), proj_lon=-92.0,
           standard_parallels=(34, 46), figsize=(11, 9.5)),
    Region("pjm", "PJM (Mid-Atlantic)",
           extent=(-90.5, -73.5, 35.0, 43.5), proj_lon=-81.5,
           standard_parallels=(36, 42), figsize=(12, 7)),
    Region("nyiso", "NYISO (New York)",
           extent=(-80.5, -71.5, 40.0, 45.5), proj_lon=-76.0,
           standard_parallels=(41, 44.5), figsize=(11, 7)),
    Region("caiso", "CAISO (California)",
           extent=(-125, -114, 32, 42.5), proj_lon=-120.0,
           standard_parallels=(33, 41), figsize=(9, 10)),
    Region("bpa", "BPA (Pacific NW)",
           extent=(-125, -110, 41.5, 49.5), proj_lon=-118.0,
           standard_parallels=(43, 48), figsize=(11, 7.5)),
]

# Solar regions (national + 5 zoomed; some overlap with wind)
SOLAR_REGIONS = [
    Region("national", "National (CONUS)",
           extent=(-125, -66, 24, 50), proj_lon=-96.0, figsize=(13, 7.5)),
    Region("ercot", "ERCOT (Texas)",
           extent=(-107, -93, 25.5, 37.0), proj_lon=-99.5,
           standard_parallels=(28, 36), figsize=(10, 8.5)),
    Region("caiso", "CAISO (California)",
           extent=(-125, -114, 32, 42.5), proj_lon=-120.0,
           standard_parallels=(33, 41), figsize=(9, 10)),
    Region("se", "Southeast",
           extent=(-92, -75, 24, 37.5), proj_lon=-83.5,
           standard_parallels=(27, 35), figsize=(11, 8)),
    Region("isone", "ISO-NE (New England)",
           extent=(-74, -66.5, 40.5, 47.5), proj_lon=-70.5,
           standard_parallels=(42, 46), figsize=(8.5, 9)),
    Region("miso", "MISO (Midwest)",
           extent=(-104, -82, 29, 49.5), proj_lon=-92.0,
           standard_parallels=(34, 46), figsize=(11, 9.5)),
    Region("pjm", "PJM (Mid-Atlantic)",
           extent=(-90.5, -73.5, 35.0, 43.5), proj_lon=-81.5,
           standard_parallels=(36, 42), figsize=(12, 7)),
]


# ============================================================================
# Colormaps (unchanged)
# ============================================================================

def wind_speed_cmap():
    stops = [
        (0,   "#f0f0f0"),
        (3,   "#c8d8e8"),
        (6,   "#7fb3d8"),
        (9,   "#5cb854"),
        (12,  "#f9d34c"),
        (16,  "#f08c30"),
        (20,  "#dc3636"),
        (25,  "#8b1a1a"),
    ]
    return LinearSegmentedColormap.from_list(
        "wind_speed",
        list(zip([s[0]/25.0 for s in stops], [s[1] for s in stops])),
        N=256,
    )


def solar_dswrf_cmap():
    stops = [
        (0,    "#0a1a2c"),
        (50,   "#2e4366"),
        (200,  "#c0c0c0"),
        (400,  "#f5edd0"),
        (600,  "#f9d34c"),
        (800,  "#f7913a"),
        (1000, "#dc3636"),
        (1200, "#7a1010"),
    ]
    return LinearSegmentedColormap.from_list(
        "solar_dswrf",
        list(zip([s[0]/1200.0 for s in stops], [s[1] for s in stops])),
        N=256,
    )


def plant_cf_cmap():
    stops = [
        (0,   "#333333"),
        (25,  "#666666"),
        (50,  "#aaaaaa"),
        (75,  "#f0c060"),
        (100, "#dc6e2a"),
    ]
    return LinearSegmentedColormap.from_list(
        "plant_cf",
        list(zip([s[0]/100.0 for s in stops], [s[1] for s in stops])),
        N=256,
    )


# ============================================================================
# Helpers
# ============================================================================

def find_latest_extended_cycle() -> datetime:
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                              microsecond=0, tzinfo=None)
    for hours_back in range(2, 24):
        candidate = now - timedelta(hours=hours_back)
        if candidate.hour in (0, 6, 12, 18):
            try:
                H = Herbie(candidate, model="hrrr", product="sfc", fxx=0)
                if H.grib is not None:
                    return candidate
            except Exception:
                continue
    raise RuntimeError("No recent extended HRRR cycle available")


def fetch_hrrr_field(cycle: datetime, fxx: int, search: str):
    """Fetch one HRRR variable by search regex. Returns DataArray or None.

    Herbie's subset mechanism caches results locally under
    ~/data/hrrr/<date>/subset_<hash>__<filename> — so calling this
    function with the same (cycle, fxx, search) more than once is free
    after the first call. Different `search` strings produce different
    cached files, but within the same Python session they're held in
    memory by xarray.
    """
    try:
        H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx)
        ds = H.xarray(search)
        var = list(ds.data_vars)[0]
        return ds[var]
    except Exception as e:
        print(f"    fetch failed ({search} F{fxx:02d}): {e}", file=sys.stderr)
        return None


def marker_size_wind(cap_mw: np.ndarray, region: Region) -> np.ndarray:
    """Wind plant marker size — slightly larger in zoomed regions for visibility."""
    base = 0.5 if region.id == "national" else 0.9
    return np.clip(base * np.sqrt(np.maximum(cap_mw, 1.0)), 2.0, 18.0) ** 2


def marker_size_solar(cap_mw: np.ndarray, region: Region) -> np.ndarray:
    base = 0.6 if region.id == "national" else 1.0
    return np.clip(base * np.sqrt(np.maximum(cap_mw, 1.0)), 2.0, 22.0) ** 2


def plants_within_extent(plants: pd.DataFrame, extent: tuple) -> pd.DataFrame:
    """Return only plants whose lat/lon fall within the bounding box."""
    if plants.empty:
        return plants
    w, e, s, n = extent
    return plants[
        (plants["xlong"] >= w) & (plants["xlong"] <= e) &
        (plants["ylat"] >= s) & (plants["ylat"] <= n)
    ]


# ============================================================================
# Plant loading
# ============================================================================

def load_wind_plants_with_forecast(cycle_str: str) -> pd.DataFrame:
    capacity_csv = (HERE.parent.parent / "assets" / "wind_forecast_data"
                    / f"capacity_plant_{cycle_str}.csv")
    if capacity_csv.exists():
        return pd.read_csv(capacity_csv)
    if WIND_INV_CSV.exists():
        df = pd.read_csv(WIND_INV_CSV, low_memory=False)
        return df.rename(columns={"t_cap": "p_cap_kw"})
    return pd.DataFrame()


def load_solar_plants_with_forecast(cycle_str: str):
    forecast_csv = (HERE.parent.parent / "assets" / "solar_forecast_data"
                    / f"forecast_plant_{cycle_str}.csv")
    capacity_csv = (HERE.parent.parent / "assets" / "solar_forecast_data"
                    / f"capacity_plant_{cycle_str}.csv")
    if not (forecast_csv.exists() and capacity_csv.exists()):
        if SOLAR_INV_CSV.exists():
            return pd.read_csv(SOLAR_INV_CSV, low_memory=False), None
        return pd.DataFrame(), None
    cap = pd.read_csv(capacity_csv)
    fc = pd.read_csv(forecast_csv, parse_dates=["valid_time"])
    pivot = fc.pivot_table(index="case_id", columns="valid_time",
                            values="MW_AC", aggfunc="sum")
    return cap, pivot


# ============================================================================
# Map rendering
# ============================================================================

def render_wind_map(field, valid_time: datetime, fxx: int,
                    plants: pd.DataFrame, cycle: datetime,
                    region: Region, out_path: Path) -> None:
    """Render one 80m wind speed map for a specific region."""
    cmap = wind_speed_cmap()
    proj = ccrs.LambertConformal(
        central_longitude=region.proj_lon,
        central_latitude=region.proj_lat,
        standard_parallels=region.standard_parallels,
    )

    fig, ax = plt.subplots(figsize=region.figsize, dpi=110,
                            subplot_kw=dict(projection=proj))
    ax.set_extent(region.extent, crs=PC)

    grid_lats = field.latitude.values
    grid_lons = field.longitude.values
    grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
    speeds = field.values

    im = ax.pcolormesh(
        grid_lons, grid_lats, speeds,
        cmap=cmap, vmin=0, vmax=25,
        transform=PC, shading="auto",
        rasterized=True, zorder=1,
    )

    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#222222", linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#000000", linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#000000", linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor="#cfdce6", edgecolor="#000000",
                   linewidth=0.3, zorder=2)

    # Plant overlay — hollow rings with dark outlines
    p = plants_within_extent(plants, region.extent)
    if not p.empty:
        cap_col = "p_cap_ac" if "p_cap_ac" in p.columns else (
            "p_cap" if "p_cap" in p.columns else None)
        if cap_col is not None:
            mask = (p[cap_col] >= MIN_WIND_PLANT_MW) & \
                   p[["xlong", "ylat"]].notna().all(axis=1)
            p = p[mask]
            sizes = marker_size_wind(p[cap_col].values, region)
            ax.scatter(
                p["xlong"].values, p["ylat"].values,
                s=sizes, facecolors="none",
                edgecolors="#0a1a2c", linewidths=0.9,
                alpha=0.9, zorder=5, transform=PC,
            )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.85)
    cbar.set_label("80m Wind Speed (m/s)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle.strftime("%Y-%m-%d %HZ")
    ax.set_title(
        f"HRRR 80m Wind Speed · {region.label}\n"
        f"Cycle {cycle_str}  ·  F{fxx:02d}  ·  Valid {valid_str}",
        fontsize=12, loc="left", pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def render_solar_map(field, valid_time: datetime, fxx: int,
                     plants: pd.DataFrame, plant_mw_at_time: Optional[pd.Series],
                     cycle: datetime, region: Region, out_path: Path) -> None:
    """Render one DSWRF map for a specific region."""
    cmap = solar_dswrf_cmap()
    cf_cmap = plant_cf_cmap()
    proj = ccrs.LambertConformal(
        central_longitude=region.proj_lon,
        central_latitude=region.proj_lat,
        standard_parallels=region.standard_parallels,
    )

    fig, ax = plt.subplots(figsize=region.figsize, dpi=110,
                            subplot_kw=dict(projection=proj))
    ax.set_extent(region.extent, crs=PC)

    grid_lats = field.latitude.values
    grid_lons = field.longitude.values
    grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
    dswrf = field.values

    im = ax.pcolormesh(
        grid_lons, grid_lats, dswrf,
        cmap=cmap, vmin=0, vmax=1100,
        transform=PC, shading="auto",
        rasterized=True, zorder=1,
    )

    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#444444", linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#222222", linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#222222", linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor="#9bb5c4", edgecolor="#222222",
                   linewidth=0.3, zorder=2)

    p = plants_within_extent(plants, region.extent)
    if not p.empty:
        cap_col = "p_cap_ac" if "p_cap_ac" in p.columns else None
        if cap_col is not None:
            mask = (p[cap_col] >= MIN_SOLAR_PLANT_MW) & \
                   p[["xlong", "ylat"]].notna().all(axis=1)
            p = p[mask].copy()
            sizes = marker_size_solar(p[cap_col].values, region)
            if plant_mw_at_time is not None:
                mw = p["case_id"].map(plant_mw_at_time).fillna(0.0)
                cf = (mw / p[cap_col]).clip(0, 1).values * 100.0
                edge_colors = cf_cmap(cf / 100.0)
            else:
                edge_colors = "#222222"
            ax.scatter(
                p["xlong"].values, p["ylat"].values,
                s=sizes, facecolors="none",
                edgecolors=edge_colors, linewidths=1.0,
                alpha=0.9, zorder=5, transform=PC,
            )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.85)
    cbar.set_label("Surface DSWRF (W/m²)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle.strftime("%Y-%m-%d %HZ")
    ax.set_title(
        f"HRRR Downward Shortwave · {region.label}\n"
        f"Cycle {cycle_str}  ·  F{fxx:02d}  ·  Valid {valid_str}",
        fontsize=12, loc="left", pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ============================================================================
# Driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?")
    parser.add_argument("hour", nargs="?")
    parser.add_argument("--skip-wind", action="store_true")
    parser.add_argument("--skip-solar", action="store_true")
    parser.add_argument("--regions-only", help="Comma-separated region IDs to render")
    args = parser.parse_args()

    if args.date and args.hour:
        cycle = datetime.strptime(f"{args.date} {args.hour}", "%Y-%m-%d %H")
    else:
        cycle = find_latest_extended_cycle()
    print(f"Cycle: {cycle:%Y-%m-%d %H:%MZ}")
    cycle_str = cycle.strftime("%Y%m%dT%HZ")

    # Region subset for development/testing
    wind_regions = WIND_REGIONS
    solar_regions = SOLAR_REGIONS
    if args.regions_only:
        wanted = set(args.regions_only.split(","))
        wind_regions = [r for r in wind_regions if r.id in wanted]
        solar_regions = [r for r in solar_regions if r.id in wanted]

    # Load plant inventories once at the start
    if not args.skip_wind:
        print("\nLoading wind plants...")
        wind_plants = load_wind_plants_with_forecast(cycle_str)
        print(f"  {len(wind_plants):,} wind plants loaded")
        for r in wind_regions:
            (WIND_DIR / r.id).mkdir(parents=True, exist_ok=True)

    if not args.skip_solar:
        print("Loading solar plants...")
        solar_plants, solar_pivot = load_solar_plants_with_forecast(cycle_str)
        print(f"  {len(solar_plants):,} solar plants loaded")
        for r in solar_regions:
            (SOLAR_DIR / r.id).mkdir(parents=True, exist_ok=True)

    # Per-region manifest accumulators
    wind_manifests = {r.id: [] for r in wind_regions} if not args.skip_wind else {}
    solar_manifests = {r.id: [] for r in solar_regions} if not args.skip_solar else {}

    t_total = time_module.time()
    n_fetches = 0
    n_renders = 0

    # OUTER LOOP: forecast hours. INNER: fetch all needed fields ONCE,
    # then render every (variable × region) from those cached fields.
    # This is the key change — Herbie caches by (cycle, fxx, search),
    # and within a single Python process xarray keeps the DataArrays in
    # memory, so we can reuse them across regions without re-fetching.
    for fxx in FORECAST_HOURS:
        valid_time = cycle + timedelta(hours=fxx)
        print(f"\nF{fxx:02d} → valid {valid_time:%Y-%m-%d %H:%MZ}", flush=True)

        # Fetch wind components (only if rendering wind)
        wind_speed_field = None
        if not args.skip_wind:
            u80 = fetch_hrrr_field(cycle, fxx, ":UGRD:80 m above ground")
            v80 = fetch_hrrr_field(cycle, fxx, ":VGRD:80 m above ground")
            n_fetches += 2
            if u80 is not None and v80 is not None:
                wind_speed_field = u80.copy()
                wind_speed_field.values = np.sqrt(u80.values ** 2 + v80.values ** 2)
            else:
                print(f"  skipping wind F{fxx:02d}: UGRD/VGRD unavailable")

        # Fetch DSWRF (only if rendering solar)
        dswrf_field = None
        if not args.skip_solar:
            dswrf_field = fetch_hrrr_field(cycle, fxx, ":DSWRF:surface")
            n_fetches += 1
            if dswrf_field is None:
                print(f"  skipping solar F{fxx:02d}: DSWRF unavailable")

        # Render wind for every region
        if wind_speed_field is not None:
            for region in wind_regions:
                out_path = WIND_DIR / region.id / f"F{fxx:02d}.png"
                render_wind_map(wind_speed_field, valid_time, fxx,
                                 wind_plants, cycle, region, out_path)
                n_renders += 1
                wind_manifests[region.id].append({
                    "fxx": fxx,
                    "valid_time": valid_time.isoformat() + "Z",
                    "valid_label": valid_time.strftime("%a %m/%d %H:%MZ"),
                    "file": f"F{fxx:02d}.png",
                })
            print(f"  wind: rendered {len(wind_regions)} regions")

        # Render solar for every region
        if dswrf_field is not None:
            mw_at_time = None
            if solar_pivot is not None:
                ts = pd.Timestamp(valid_time)
                if ts in solar_pivot.columns:
                    mw_at_time = solar_pivot[ts]
            for region in solar_regions:
                out_path = SOLAR_DIR / region.id / f"F{fxx:02d}.png"
                render_solar_map(dswrf_field, valid_time, fxx,
                                  solar_plants, mw_at_time, cycle,
                                  region, out_path)
                n_renders += 1
                solar_manifests[region.id].append({
                    "fxx": fxx,
                    "valid_time": valid_time.isoformat() + "Z",
                    "valid_label": valid_time.strftime("%a %m/%d %H:%MZ"),
                    "file": f"F{fxx:02d}.png",
                })
            print(f"  solar: rendered {len(solar_regions)} regions")

    # Write per-region manifests + regions.json for each variable
    if not args.skip_wind:
        for region in wind_regions:
            manifest = {
                "cycle": cycle.strftime("%Y-%m-%d %HZ"),
                "cycle_compact": cycle_str,
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "variable": "80m wind speed",
                "units": "m/s",
                "region_id": region.id,
                "region_label": region.label,
                "frames": wind_manifests[region.id],
            }
            (WIND_DIR / region.id / "manifest.json").write_text(
                json.dumps(manifest, indent=2))
        regions_meta = {
            "regions": [
                {"id": r.id, "label": r.label, "default": (r.id == "national")}
                for r in wind_regions
            ],
        }
        (WIND_DIR / "regions.json").write_text(json.dumps(regions_meta, indent=2))

    if not args.skip_solar:
        for region in solar_regions:
            manifest = {
                "cycle": cycle.strftime("%Y-%m-%d %HZ"),
                "cycle_compact": cycle_str,
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "variable": "Surface downward shortwave",
                "units": "W/m²",
                "region_id": region.id,
                "region_label": region.label,
                "frames": solar_manifests[region.id],
            }
            (SOLAR_DIR / region.id / "manifest.json").write_text(
                json.dumps(manifest, indent=2))
        regions_meta = {
            "regions": [
                {"id": r.id, "label": r.label, "default": (r.id == "national")}
                for r in solar_regions
            ],
        }
        (SOLAR_DIR / "regions.json").write_text(json.dumps(regions_meta, indent=2))

    elapsed = time_module.time() - t_total
    print(f"\nTotal: {n_fetches} fetches + {n_renders} renders in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
