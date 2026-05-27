"""Synoptic map renderer (multi-variable, multi-region).

For a given HRRR cycle, render static PNG maps of multiple
atmospheric variables (wind, solar radiation, cloud ceiling,
visibility, reflectivity), each at multiple region zooms (national
plus a handful of ISO/regional footprints).

Renders are parallelized across (variable, region) pairs within each
forecast hour using a process pool. This gives ~5-7x speedup on a
modern multicore machine since matplotlib/cartopy rendering is the
dominant cost and is embarrassingly parallel.

Output:
    assets/synoptic/<variable>/<region>/F00.png ... F48.png
    assets/synoptic/<variable>/<region>/manifest.json
    assets/synoptic/variables.json       (list of variables for viewer dropdown)
    assets/synoptic/regions.json         (list of regions for viewer dropdown)

Adding a new variable: just append a `Variable(...)` to VARIABLES below.
Adding a new region: append a `Region(...)` to REGIONS. No other code changes.

Usage:
    python render_maps.py                  # latest extended cycle
    python render_maps.py 2026-05-24 18    # specific cycle
    python render_maps.py --variables wind,solar    # subset
    python render_maps.py --regions national,ercot  # subset
    python render_maps.py --workers 4               # control parallelism
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from herbie import Herbie

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================================
# Constants
# ============================================================================

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets" / "synoptic"

WIND_INV_CSV = HERE.parent / "data" / "uswtdb.csv"
SOLAR_INV_CSV = HERE.parent / "solar" / "data" / "uspvdb.csv"

FORECAST_HOURS = list(range(0, 49))

MIN_WIND_PLANT_MW = 30.0
MIN_SOLAR_PLANT_MW = 5.0

PC = ccrs.PlateCarree()


# ============================================================================
# Region definitions — unified across all variables
# ============================================================================

@dataclass
class Region:
    id: str
    label: str
    extent: tuple                 # (west, east, south, north)
    proj_lon: float
    proj_lat: float = 37.5
    standard_parallels: tuple = (33.0, 45.0)
    figsize: tuple = (13, 7.5)


REGIONS = [
    Region("national", "National (CONUS)",
           extent=(-125, -66, 24, 50), proj_lon=-96.0, figsize=(13, 7.5)),
    Region("northwest", "Northwest",
           extent=(-125, -110, 41.5, 49.5), proj_lon=-118.0,
           standard_parallels=(43, 48), figsize=(11, 7.5)),
    Region("caiso", "CAISO (California)",
           extent=(-125, -114, 32, 42.5), proj_lon=-120.0,
           standard_parallels=(33, 41), figsize=(9, 10)),
    Region("spp", "SPP (Plains)",
           extent=(-106, -89.5, 31, 49), proj_lon=-97.5,
           standard_parallels=(34, 46), figsize=(10, 10)),
    Region("ercot", "ERCOT (Texas)",
           extent=(-107, -93, 25.5, 37.0), proj_lon=-99.5,
           standard_parallels=(28, 36), figsize=(10, 8.5)),
    Region("miso", "MISO (Midwest)",
           extent=(-104, -82, 29, 49.5), proj_lon=-92.0,
           standard_parallels=(34, 46), figsize=(11, 9.5)),
    Region("southeast", "Southeast",
           extent=(-92, -75, 24, 37.5), proj_lon=-83.5,
           standard_parallels=(27, 35), figsize=(11, 8)),
    Region("pjm", "PJM (Mid-Atlantic)",
           extent=(-90.5, -73.5, 35.0, 43.5), proj_lon=-81.5,
           standard_parallels=(36, 42), figsize=(12, 7)),
    Region("newengland", "New England",
           extent=(-74, -66.5, 40.5, 47.5), proj_lon=-70.5,
           standard_parallels=(42, 46), figsize=(8.5, 9)),
]


# ============================================================================
# Variable definitions
# ============================================================================

@dataclass
class Variable:
    """A renderable HRRR variable with its colormap, label, units, etc.

    `grib_search` is the search regex(es) for Herbie. For composite
    variables (like wind speed from UGRD+VGRD), use multiple searches
    and provide a `combine` function that takes the dict of DataArrays
    and returns a single combined DataArray.
    """
    id: str
    label: str
    units: str
    grib_searches: list           # list of search regexes
    combine: Optional[Callable] = None    # callback that builds a single field from multiple
    overlay: str = "none"         # "none", "wind", or "solar"
    title: str = ""               # display title for the figure
    vmin: float = 0.0
    vmax: float = 100.0
    cmap_factory: Optional[Callable] = None
    norm_factory: Optional[Callable] = None
    cbar_format: str = "%g"


# Composite functions for variables computed from multiple GRIB messages

def _combine_wind_speed(fields: dict, searches: list):
    """Compute scalar wind speed from U and V components."""
    u = fields.get(searches[0])
    v = fields.get(searches[1])
    if u is None or v is None:
        return None
    out = u.copy()
    out.values = np.sqrt(u.values ** 2 + v.values ** 2)
    return out


def _passthrough(fields: dict, searches: list):
    """Just return the only field."""
    return fields.get(searches[0])


# ---------- Colormaps ----------

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


def reflectivity_cmap():
    """Standard NWS reflectivity colormap (≈MRMS scheme)."""
    stops = [
        (-30, "#ffffff00"),     # transparent below 5
        (5,   "#8ecde0"),       # light blue
        (10,  "#6ba6c0"),
        (15,  "#4f8a9e"),
        (20,  "#5cba3a"),       # green
        (25,  "#4a9c2c"),
        (30,  "#388e1e"),
        (35,  "#f9d34c"),       # yellow
        (40,  "#f5a623"),       # orange
        (45,  "#e85d2c"),       # red-orange
        (50,  "#cf2618"),       # red
        (55,  "#9a1b14"),
        (60,  "#bd5d92"),       # pink/magenta
        (65,  "#e183c7"),
        (75,  "#ffffff"),       # white at extreme
    ]
    # Normalize to [0,1] over the range -30 to 75
    span = 75.0 - (-30.0)
    return LinearSegmentedColormap.from_list(
        "refd",
        list(zip([(s[0] - (-30)) / span for s in stops], [s[1] for s in stops])),
        N=256,
    )


def visibility_cmap():
    """Aviation visibility categories: LIFR red → IFR orange → MVFR yellow → VFR pale.

    Thresholds in statute miles: LIFR <1, IFR 1-3, MVFR 3-5, VFR >5.
    HRRR units are meters; 1 mi = 1609 m, 3 mi = 4828 m, 5 mi = 8047 m.
    """
    # Build a continuous-ish ramp through aviation categories.
    # Smoother than strict categorical but still respects thresholds.
    stops = [
        (0,      "#7a0010"),       # below 0.5 mi: extreme low
        (805,    "#cf2618"),       # 0.5 mi: LIFR
        (1609,   "#e85d2c"),       # 1 mi: IFR ceiling
        (4828,   "#f5a623"),       # 3 mi: IFR/MVFR boundary
        (8047,   "#f9d34c"),       # 5 mi: MVFR/VFR boundary
        (12000,  "#c8e8a8"),       # 7.5 mi: light VFR
        (16000,  "#f0f4e8"),       # full visibility: pale
    ]
    span = 16000.0
    return LinearSegmentedColormap.from_list(
        "vis",
        list(zip([s[0] / span for s in stops], [s[1] for s in stops])),
        N=256,
    )


def ceiling_cmap():
    """Cloud ceiling: low ceilings = red (LIFR/IFR), high = light, no ceiling = transparent.

    HRRR units: meters above ground. Aviation thresholds:
      <152m   (500 ft): LIFR
      152-305 (500-1000 ft): IFR
      305-914 (1000-3000 ft): MVFR
      >914    (3000 ft): VFR
    """
    stops = [
        (0,     "#7a0010"),      # 0 ft: extreme low ceiling
        (152,   "#cf2618"),      # 500 ft: LIFR
        (305,   "#e85d2c"),      # 1000 ft: IFR
        (610,   "#f5a623"),      # 2000 ft
        (914,   "#f9d34c"),      # 3000 ft: MVFR
        (1524,  "#c8e8a8"),      # 5000 ft
        (3048,  "#9bd0e8"),      # 10000 ft
        (6096,  "#f0f4e8"),      # 20000 ft: high ceiling / clear
    ]
    span = 6096.0
    return LinearSegmentedColormap.from_list(
        "ceiling",
        list(zip([s[0] / span for s in stops], [s[1] for s in stops])),
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


# ---------- Variable list ----------

VARIABLES = [
    Variable(
        id="wind",
        label="80m Wind Speed",
        title="HRRR 80m Wind Speed",
        units="m/s",
        grib_searches=[
            ":UGRD:80 m above ground",
            ":VGRD:80 m above ground",
        ],
        combine=_combine_wind_speed,
        overlay="wind",
        vmin=0, vmax=25,
        cmap_factory=wind_speed_cmap,
    ),
    Variable(
        id="solar",
        label="Surface Shortwave (DSWRF)",
        title="HRRR Surface Downward Shortwave",
        units="W/m²",
        grib_searches=[":DSWRF:surface"],
        combine=_passthrough,
        overlay="solar",
        vmin=0, vmax=1100,
        cmap_factory=solar_dswrf_cmap,
    ),
    Variable(
        id="reflectivity",
        label="1 km Reflectivity",
        title="HRRR Simulated Reflectivity (1 km AGL)",
        units="dBZ",
        grib_searches=[":REFD:1000 m above ground"],
        combine=_passthrough,
        overlay="none",
        vmin=-30, vmax=75,
        cmap_factory=reflectivity_cmap,
    ),
    Variable(
        id="visibility",
        label="Surface Visibility",
        title="HRRR Surface Visibility",
        units="m",
        grib_searches=[":VIS:surface"],
        combine=_passthrough,
        overlay="none",
        vmin=0, vmax=16000,
        cmap_factory=visibility_cmap,
        cbar_format="%.0f",
    ),
    Variable(
        id="ceiling",
        label="Cloud Ceiling",
        title="HRRR Cloud Ceiling (m AGL)",
        units="m AGL",
        grib_searches=[":HGT:cloud ceiling"],
        combine=_passthrough,
        overlay="none",
        vmin=0, vmax=6096,
        cmap_factory=ceiling_cmap,
        cbar_format="%.0f",
    ),
]


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
    try:
        H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx)
        ds = H.xarray(search)
        var = list(ds.data_vars)[0]
        return ds[var]
    except Exception as e:
        print(f"    fetch failed ({search} F{fxx:02d}): {e}", file=sys.stderr)
        return None


def marker_size_wind(cap_mw: np.ndarray, region: Region) -> np.ndarray:
    base = 0.32 if region.id == "national" else 0.55
    return np.clip(base * np.sqrt(np.maximum(cap_mw, 1.0)), 3.5, 9.0) ** 2


def marker_size_solar(cap_mw: np.ndarray, region: Region) -> np.ndarray:
    base = 0.6 if region.id == "national" else 1.0
    return np.clip(base * np.sqrt(np.maximum(cap_mw, 1.0)), 2.0, 22.0) ** 2


def plants_within_extent(plants: pd.DataFrame, extent: tuple) -> pd.DataFrame:
    if plants.empty:
        return plants
    w, e, s, n = extent
    return plants[
        (plants["xlong"] >= w) & (plants["xlong"] <= e) &
        (plants["ylat"] >= s) & (plants["ylat"] <= n)
    ]


# ============================================================================
# Plant data loading
# ============================================================================

def load_wind_plants(cycle_str: str) -> pd.DataFrame:
    capacity_csv = (HERE.parent.parent / "assets" / "wind_forecast_data"
                    / f"capacity_plant_{cycle_str}.csv")
    if capacity_csv.exists():
        return pd.read_csv(capacity_csv)
    if WIND_INV_CSV.exists():
        df = pd.read_csv(WIND_INV_CSV, low_memory=False)
        return df.rename(columns={"t_cap": "p_cap_kw"})
    return pd.DataFrame()


def load_solar_plants(cycle_str: str):
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
# Map rendering — generic
# ============================================================================

def _draw_features(ax, dark_borders: bool = True):
    """Draw state/coastline/border features. Dark for light-background maps
    (wind, ceiling, vis), lighter for dark-background maps (solar)."""
    state_color = "#222222" if dark_borders else "#aaaaaa"
    line_color = "#000000" if dark_borders else "#dddddd"
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor=state_color, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor=line_color, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor=line_color, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor="#cfdce6", edgecolor=line_color,
                   linewidth=0.3, zorder=2)


def _overlay_wind_plants(ax, plants, region):
    p = plants_within_extent(plants, region.extent)
    if p.empty:
        return
    if "capacity_MW" in p.columns:
        cap_col = "capacity_MW"
    elif "p_cap_ac" in p.columns:
        cap_col = "p_cap_ac"
    elif "p_cap" in p.columns:
        cap_col = "p_cap"
    else:
        return
    mask = (p[cap_col] >= MIN_WIND_PLANT_MW) & p[["xlong", "ylat"]].notna().all(axis=1)
    p = p[mask]
    if p.empty:
        return
    sizes = marker_size_wind(p[cap_col].values, region)
    ax.scatter(
        p["xlong"].values, p["ylat"].values,
        s=sizes, facecolors="none",
        edgecolors="#0a1a2c", linewidths=0.45,
        alpha=0.85, zorder=5, transform=PC,
    )


def _overlay_solar_plants(ax, plants, plant_mw_at_time, region):
    """Overlay solar plants with capacity-factor-colored rings.

    `plant_mw_at_time` can be a dict mapping case_id → MW (preferred,
    cheap to pickle) or a pandas Series with same semantics, or None.
    """
    p = plants_within_extent(plants, region.extent)
    if p.empty or "p_cap_ac" not in p.columns:
        return
    mask = (p["p_cap_ac"] >= MIN_SOLAR_PLANT_MW) & \
           p[["xlong", "ylat"]].notna().all(axis=1)
    p = p[mask].copy()
    if p.empty:
        return
    sizes = marker_size_solar(p["p_cap_ac"].values, region)
    if plant_mw_at_time is not None:
        if isinstance(plant_mw_at_time, dict):
            mw = p["case_id"].map(plant_mw_at_time).fillna(0.0)
        else:
            mw = p["case_id"].map(plant_mw_at_time).fillna(0.0)
        cf = (mw / p["p_cap_ac"]).clip(0, 1).values * 100.0
        edge_colors = plant_cf_cmap()(cf / 100.0)
    else:
        edge_colors = "#222222"
    ax.scatter(
        p["xlong"].values, p["ylat"].values,
        s=sizes, facecolors="none",
        edgecolors=edge_colors, linewidths=1.0,
        alpha=0.9, zorder=5, transform=PC,
    )


def _share_array(arr: np.ndarray) -> dict:
    """Copy an ndarray into a SharedMemory block and return a reference dict.

    The caller is responsible for tracking the returned 'name' and
    calling _release_shared(name) once all workers have finished.
    """
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    # Create an ndarray view backed by the shared buffer and copy data in
    shared_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_view[:] = arr[:]
    return {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }


def _release_shared(refs: list) -> None:
    """Close and unlink all shared memory blocks. Call at end of run.

    `refs` is a list of name strings (just the SHM names).
    """
    for name in refs:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  warn: failed to unlink shm {name}: {e}", file=sys.stderr)


def _attach_array(ref: dict) -> tuple:
    """Inside a worker: reconstruct an ndarray view of a SharedMemory block.

    Returns (ndarray, shm_handle). The shm_handle must be kept alive
    until the ndarray is no longer needed (Python doesn't track the
    backing buffer otherwise). The caller closes the handle after use.
    """
    shm = shared_memory.SharedMemory(name=ref["name"])
    arr = np.ndarray(ref["shape"], dtype=np.dtype(ref["dtype"]),
                     buffer=shm.buf)
    return arr, shm


def render_map(values: np.ndarray, grid_lats: np.ndarray, grid_lons: np.ndarray,
               valid_time: datetime, fxx: int,
               variable_id: str, region_id: str, cycle: datetime,
               out_path: Path, overlay_records=None,
               overlay_mw_at_time=None) -> None:
    """Generic map renderer. Plain numpy arrays in, PNG out."""
    variable = next(v for v in VARIABLES if v.id == variable_id)
    region = next(r for r in REGIONS if r.id == region_id)

    cmap = variable.cmap_factory()
    proj = ccrs.LambertConformal(
        central_longitude=region.proj_lon,
        central_latitude=region.proj_lat,
        standard_parallels=region.standard_parallels,
    )

    fig, ax = plt.subplots(figsize=region.figsize, dpi=110,
                            subplot_kw=dict(projection=proj))
    ax.set_extent(region.extent, crs=PC)

    # Ceiling has sentinel values (~20000 m) where no ceiling exists.
    # We need a writable local copy here since `values` may be a view
    # into shared memory that we don't want to mutate for other workers.
    if variable.id == "ceiling":
        values = np.where(values > 6500, np.nan, values)

    im = ax.pcolormesh(
        grid_lons, grid_lats, values,
        cmap=cmap, vmin=variable.vmin, vmax=variable.vmax,
        transform=PC, shading="auto",
        rasterized=True, zorder=1,
    )

    _draw_features(ax)

    if variable.overlay == "wind" and overlay_records is not None:
        _overlay_wind_plants(ax, overlay_records, region)
    elif variable.overlay == "solar" and overlay_records is not None:
        _overlay_solar_plants(ax, overlay_records, overlay_mw_at_time, region)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.85, format=variable.cbar_format)
    cbar.set_label(f"{variable.label} ({variable.units})", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle.strftime("%Y-%m-%d %HZ")
    ax.set_title(
        f"{variable.title} · {region.label}\n"
        f"Cycle {cycle_str}  ·  F{fxx:02d}  ·  Valid {valid_str}",
        fontsize=12, loc="left", pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _render_task(args: dict) -> str:
    """Worker-side entry point.

    `args["values"]` is the per-field ndarray, passed via task pickling.
    Grid arrays are shared via SharedMemory and attached here.
    """
    # Attach shared grid arrays (these are reused across all 2,205 tasks)
    grid_lats_arr, lats_shm = _attach_array(args["grid_lats_ref"])
    grid_lons_arr, lons_shm = _attach_array(args["grid_lons_ref"])
    try:
        render_map(
            values=args["values"],
            grid_lats=grid_lats_arr,
            grid_lons=grid_lons_arr,
            valid_time=args["valid_time"],
            fxx=args["fxx"],
            variable_id=args["variable_id"],
            region_id=args["region_id"],
            cycle=args["cycle"],
            out_path=args["out_path"],
            overlay_records=args.get("overlay_records"),
            overlay_mw_at_time=args.get("overlay_mw_at_time"),
        )
    finally:
        lats_shm.close()
        lons_shm.close()
    return args["variable_id"]


# ============================================================================
# Driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?")
    parser.add_argument("hour", nargs="?")
    parser.add_argument("--variables", help="Comma-separated variable IDs")
    parser.add_argument("--regions", help="Comma-separated region IDs")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of render worker processes (0 = auto, all cores)")
    args = parser.parse_args()

    if args.date and args.hour:
        cycle = datetime.strptime(f"{args.date} {args.hour}", "%Y-%m-%d %H")
    else:
        cycle = find_latest_extended_cycle()
    print(f"Cycle: {cycle:%Y-%m-%d %H:%MZ}")
    cycle_str = cycle.strftime("%Y%m%dT%HZ")

    variables = VARIABLES
    if args.variables:
        wanted = set(args.variables.split(","))
        variables = [v for v in variables if v.id in wanted]
    regions = REGIONS
    if args.regions:
        wanted = set(args.regions.split(","))
        regions = [r for r in regions if r.id in wanted]

    # Aggressive default: use ALL available cores. User said memory is
    # not a constraint, so we don't hold back.
    if args.workers > 0:
        n_workers = args.workers
    else:
        n_workers = max(1, os.cpu_count() or 4)

    print(f"Variables ({len(variables)}): {[v.id for v in variables]}")
    print(f"Regions ({len(regions)}): {[r.id for r in regions]}")
    print(f"Workers: {n_workers} (out of {os.cpu_count()} cores)")

    # Plant data
    need_wind = any(v.overlay == "wind" for v in variables)
    need_solar = any(v.overlay == "solar" for v in variables)
    wind_plants = load_wind_plants(cycle_str) if need_wind else pd.DataFrame()
    if need_wind:
        print(f"Loaded {len(wind_plants):,} wind plants")
    solar_plants, solar_pivot = (load_solar_plants(cycle_str)
                                  if need_solar else (pd.DataFrame(), None))
    if need_solar:
        print(f"Loaded {len(solar_plants):,} solar plants")

    for v in variables:
        for r in regions:
            (ASSETS / v.id / r.id).mkdir(parents=True, exist_ok=True)
    manifests = {(v.id, r.id): [] for v in variables for r in regions}

    t_total = time_module.time()

    # PHASE 1: Fetch all HRRR fields up front, sequentially.
    print(f"\n[1/2] Fetching {len(variables) * len(FORECAST_HOURS)} field-hours...")
    t_fetch = time_module.time()

    # Memory strategy:
    #   - Grid lats/lons are identical across all 245 (variable, hour) pairs,
    #     so we put them in shared memory ONCE and reference them in every
    #     task. Saves 245x duplication.
    #   - Each field's data array is passed as a regular task argument
    #     (pickled per task). With bounded worker queue, only a few tasks
    #     are in flight at once, so this caps memory naturally and avoids
    #     /dev/shm size limits (typically 64 MB on Linux containers).
    field_cache = {}   # (variable_id, fxx) → values ndarray (float32)
    grid_lats_ref = None
    grid_lons_ref = None
    shm_names_to_release = []

    n_fetched = 0
    n_skipped = 0
    for fxx in FORECAST_HOURS:
        for variable in variables:
            fields = {}
            for search in variable.grib_searches:
                arr = fetch_hrrr_field(cycle, fxx, search)
                if arr is None:
                    fields = None
                    break
                fields[search] = arr
            if fields is None:
                field_cache[(variable.id, fxx)] = None
                n_skipped += 1
                continue
            combined = variable.combine(fields, variable.grib_searches)
            if combined is None:
                field_cache[(variable.id, fxx)] = None
                n_skipped += 1
                continue

            # Shared-memory grid (built lazily on first fetch)
            if grid_lats_ref is None:
                grid_lats = np.ascontiguousarray(combined.latitude.values,
                                                  dtype=np.float32)
                grid_lons_raw = np.ascontiguousarray(combined.longitude.values,
                                                     dtype=np.float32)
                grid_lons = np.where(grid_lons_raw > 180,
                                      grid_lons_raw - 360, grid_lons_raw)
                grid_lats_ref = _share_array(grid_lats)
                grid_lons_ref = _share_array(grid_lons)
                shm_names_to_release.append(grid_lats_ref["name"])
                shm_names_to_release.append(grid_lons_ref["name"])
                print(f"  shared-memory grid: {grid_lats.shape} "
                      f"({(grid_lats.nbytes + grid_lons.nbytes) / 1e6:.1f} MB)")

            # Per-field values: keep in main-process memory only.
            # Will be passed through pickling to workers (cheap with
            # bounded queue depth).
            values = np.ascontiguousarray(combined.values, dtype=np.float32)
            field_cache[(variable.id, fxx)] = values
            n_fetched += 1

        if (fxx + 1) % 12 == 0:
            print(f"    ... F00-F{fxx:02d}: {n_fetched} OK, {n_skipped} skipped "
                  f"({time_module.time()-t_fetch:.0f}s elapsed)")

    print(f"  fetched in {time_module.time() - t_fetch:.0f}s "
          f"({n_fetched} OK, {n_skipped} skipped)")
    main_mem_mb = sum(v.nbytes for v in field_cache.values() if v is not None) / 1e6
    print(f"  main-process field cache: {main_mem_mb:.0f} MB across {n_fetched} arrays")

    # PHASE 2: Build task list, dispatch to pool.
    print(f"\n[2/2] Rendering all (variable × region × hour) PNGs in parallel...")

    mw_by_hour = {}
    if need_solar and solar_pivot is not None:
        for fxx in FORECAST_HOURS:
            valid_time = cycle + timedelta(hours=fxx)
            ts = pd.Timestamp(valid_time)
            mw_by_hour[fxx] = (solar_pivot[ts].to_dict()
                                if ts in solar_pivot.columns else None)
    else:
        for fxx in FORECAST_HOURS:
            mw_by_hour[fxx] = None

    tasks = []
    for variable in variables:
        overlay_records = (wind_plants if variable.overlay == "wind"
                            else solar_plants if variable.overlay == "solar"
                            else None)
        for fxx in FORECAST_HOURS:
            values = field_cache.get((variable.id, fxx))
            if values is None:
                continue
            valid_time = cycle + timedelta(hours=fxx)
            for region in regions:
                tasks.append({
                    # Per-task: each gets a reference to the same in-memory
                    # array (no copy until pickling, which happens
                    # per-submit by ProcessPoolExecutor). With a bounded
                    # worker queue, only a few tasks are pickled at once.
                    "values": values,
                    "grid_lats_ref": grid_lats_ref,
                    "grid_lons_ref": grid_lons_ref,
                    "valid_time": valid_time,
                    "fxx": fxx,
                    "variable_id": variable.id,
                    "region_id": region.id,
                    "cycle": cycle,
                    "out_path": ASSETS / variable.id / region.id / f"F{fxx:02d}.png",
                    "overlay_records": overlay_records,
                    "overlay_mw_at_time": (mw_by_hour[fxx]
                                            if variable.overlay == "solar" else None),
                })

    print(f"  {len(tasks)} render tasks queued ({n_workers} workers)")
    t_render = time_module.time()
    n_done = 0
    n_failed = 0
    progress_every = max(1, len(tasks) // 40)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_render_task, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    n_done += 1
                    manifests[(task["variable_id"], task["region_id"])].append({
                        "fxx": task["fxx"],
                        "valid_time": task["valid_time"].isoformat() + "Z",
                        "valid_label": task["valid_time"].strftime("%a %m/%d %H:%MZ"),
                        "file": f"F{task['fxx']:02d}.png",
                    })
                except Exception as e:
                    n_failed += 1
                    print(f"  ERROR {task['variable_id']}/{task['region_id']}/F{task['fxx']:02d}: {e}",
                          file=sys.stderr)
                if n_done % progress_every == 0:
                    pct = 100 * n_done / len(tasks)
                    rate = n_done / max(time_module.time() - t_render, 1)
                    eta = (len(tasks) - n_done) / max(rate, 0.1)
                    print(f"    {n_done}/{len(tasks)} ({pct:.0f}%) "
                          f"rate={rate:.1f}/s eta={eta:.0f}s")
    finally:
        print(f"  releasing {len(shm_names_to_release)} shared-memory blocks...")
        _release_shared(shm_names_to_release)

    print(f"  rendered in {time_module.time() - t_render:.0f}s "
          f"({n_done} OK, {n_failed} failed)")

    # Sort manifest frames by fxx (parallel completion order is arbitrary)
    for key in manifests:
        manifests[key].sort(key=lambda x: x["fxx"])

    # Per-(variable, region) manifests
    for v in variables:
        for r in regions:
            manifest = {
                "cycle": cycle.strftime("%Y-%m-%d %HZ"),
                "cycle_compact": cycle_str,
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "variable_id": v.id,
                "variable_label": v.label,
                "units": v.units,
                "region_id": r.id,
                "region_label": r.label,
                "frames": manifests[(v.id, r.id)],
            }
            (ASSETS / v.id / r.id / "manifest.json").write_text(
                json.dumps(manifest, indent=2))

    # Top-level dropdown metadata
    variables_meta = {
        "variables": [
            {"id": v.id, "label": v.label, "units": v.units,
             "default": (v.id == "wind")}
            for v in variables
        ],
    }
    (ASSETS / "variables.json").write_text(json.dumps(variables_meta, indent=2))

    regions_meta = {
        "regions": [
            {"id": r.id, "label": r.label,
             "default": (r.id == "national")}
            for r in regions
        ],
    }
    (ASSETS / "regions.json").write_text(json.dumps(regions_meta, indent=2))

    total = time_module.time() - t_total
    print(f"\nTotal: {n_done} renders in {total:.0f}s "
          f"({total/max(n_done,1):.2f}s/render avg, {n_workers} workers)")


if __name__ == "__main__":
    main()
