"""Synoptic map renderer (multi-model, multi-variable, multi-region).

For a given HRRR cycle, render static PNG maps of multiple
atmospheric variables (wind, solar radiation, cloud ceiling,
visibility, reflectivity), each at multiple region zooms (national
plus a handful of ISO/regional footprints). The same variables are
also rendered from RRFS (same cycle, same grid — the RRFS
2dfld.3km.conus product is bit-identical to the HRRR CONUS grid),
plus 3-panel HRRR/RRFS/difference comparison products.

Renders are parallelized across (variable, region) pairs within each
forecast hour using a process pool. This gives ~5-7x speedup on a
modern multicore machine since matplotlib/cartopy rendering is the
dominant cost and is embarrassingly parallel.

Output:
    assets/synoptic/<variable>/<region>/F00.webp ... F48.webp   (HRRR — unchanged)
    assets/synoptic/rrfs_<variable>/<region>/...                (RRFS)
    assets/synoptic/t2m_diff/<region>/...                       (3-panel diffs)
    assets/synoptic/precip_diff/<region>/...
    assets/synoptic/<product>/<region>/manifest.json
    assets/synoptic/variables.json       (list of products for viewer dropdown)
    assets/synoptic/regions.json         (list of regions for viewer dropdown)

Adding a new variable: just append a `Variable(...)` to VARIABLES below.
Adding a new region: append a `Region(...)` to REGIONS. No other code changes.
Adding a new diff product: append a `DiffProduct(...)` to DIFF_PRODUCTS.

Usage:
    python render_maps.py                  # latest extended cycle
    python render_maps.py 2026-05-24 18    # specific cycle
    python render_maps.py --variables wind,solar    # subset (rrfs_wind, t2m_diff... ok too)
    python render_maps.py --regions national,ercot  # subset
    python render_maps.py --workers 4               # control parallelism
    python render_maps.py --models hrrr             # skip RRFS + diffs (CI default
                                                    #  via SYNOPTIC_MODELS env var)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
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

try:
    import julia_bridge as _jb
except Exception:                                              # noqa: BLE001
    _jb = None
import matplotlib.colors as mcolors
from herbie import Herbie

import rrfs_fetch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _prewarm_cartopy_cache() -> None:
    """Force cartopy to download its Natural Earth shapefiles in the main
    process before workers start. Without this, each worker downloads
    independently the first time it touches a feature, which is slow
    on CI (4 workers × ~150 MB of shapefiles, downloaded serially within
    each worker) and pollutes logs with DownloadWarning messages.
    """
    import warnings
    print("  warming cartopy shapefile cache (one-time, ~150 MB)...",
          flush=True)
    t0 = time_module.time()
    for feature, scale in [
        (cfeature.COASTLINE, "10m"),   # high-res coast on every plot
        (cfeature.STATES, "50m"),
        (cfeature.BORDERS, "50m"),
        (cfeature.STATES, "110m"),
        (cfeature.BORDERS, "110m"),
    ]:
        f = feature.with_scale(scale)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Touching .geometries() triggers the download
                list(f.geometries())[:1]
        except Exception as e:
            print(f"    warn: cartopy prewarm failed for {f}: {e}",
                  file=sys.stderr)
    print(f"  cartopy cache warmed in {time_module.time()-t0:.0f}s",
          flush=True)


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
           extent=(-125, -66, 24, 50), proj_lon=-96.0, figsize=(15.4, 8.0)),
    Region("northwest", "Northwest",
           extent=(-125, -94, 38.5, 50), proj_lon=-109.5,
           standard_parallels=(41, 48), figsize=(16.0, 8.0)),
    Region("southwest", "Southwest",
           extent=(-125, -94, 24, 38.5), proj_lon=-109.5,
           standard_parallels=(28, 36), figsize=(15.4, 8.0)),
    Region("northeast", "Northeast",
           extent=(-94, -66, 34, 50), proj_lon=-80.0,
           standard_parallels=(38, 47), figsize=(11.0, 8.0)),
    Region("southeast", "Southeast",
           extent=(-94, -75, 24, 38.5), proj_lon=-84.5,
           standard_parallels=(28, 36), figsize=(9.6, 8.0)),
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

    `rrfs_searches` overrides the search strings for the RRFS idx
    lookup (rrfs_fetch.py). Most fields share the exact HRRR strings,
    so None means "use grib_searches". Only needed where the two
    models' idx entries diverge (smoke's species-split COLMD, DSWRF's
    instantaneous-vs-averaged pair, run-total APCP).
    """
    id: str
    label: str
    units: str
    grib_searches: list           # list of search regexes
    rrfs_searches: Optional[list] = None  # RRFS idx searches (None → grib_searches)
    min_fxx: int = 0              # first hour the field exists (APCP has no F00)
    combine: Optional[Callable] = None    # callback that builds a single field from multiple
    overlay: str = "none"         # "none", "wind", or "solar"
    wind_vectors: bool = False    # draw U/V direction arrows over the field
    title: str = ""               # display title for the figure
    vmin: float = 0.0
    vmax: float = 100.0
    cmap_factory: Optional[Callable] = None
    norm_factory: Optional[Callable] = None
    cbar_format: str = "%g"
    transform: Optional[Callable] = None  # unit conversion on the numpy field
    point_values: bool = False            # stamp sampled values on the map
    cbar_extend: str = "neither"
    cbar_ticks: Optional[list] = None


# Approximate number of wind arrows across the frame width. The array
# stride is derived from this and the sliced grid shape so arrow spacing
# looks consistent regardless of region zoom.
WIND_ARROW_COLS = 52


# Composite functions for variables computed from multiple GRIB messages

def _combine_wind_speed(fields: dict, searches: list):
    """Compute scalar wind speed from U and V components.

    Also stashes the raw U/V arrays in the returned array's .attrs so the
    driver can cache them for direction arrows, without breaking the
    single-field return contract the rest of the pipeline expects.
    """
    u = fields.get(searches[0])
    v = fields.get(searches[1])
    if u is None or v is None:
        return None
    out = u.copy()
    out.values = np.sqrt(u.values ** 2 + v.values ** 2)
    out.attrs["_wind_u"] = np.ascontiguousarray(u.values, dtype=np.float32)
    out.attrs["_wind_v"] = np.ascontiguousarray(v.values, dtype=np.float32)
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



# ── 2 m temperature: banded 2 °F scale, one hue family per ~10 °F decade ──
# You can read the decade from the hue (purples below 0 °F, blues to
# freezing, teals/greens through the 40s–60s, yellow 70s, orange 80s, red
# 90s, maroon 100s, pink past 110) and the shade gives the 2 °F step inside
# it — high fidelity without losing the at-a-glance banding.
_T2M_ANCHORS = [
    (-30, "#f6e8fb"), (-20, "#d5a6ea"), (-10, "#9a5ccc"), (0, "#5b2d91"),
    (10, "#31329b"), (20, "#2f62d4"), (32, "#41a3ea"),
    (40, "#39c3b7"), (50, "#2fa053"), (60, "#83cd45"),
    (70, "#f6e73b"), (80, "#f5a623"), (90, "#e8542a"),
    (100, "#b81c1c"), (110, "#7a0d33"), (116, "#f3a0c8"),
]
_T2M_BOUNDS = np.arange(-30.0, 116.1, 2.0)


def _t2m_scale():
    lo, hi = _T2M_ANCHORS[0][0], _T2M_ANCHORS[-1][0]
    pos = [(a - lo) / (hi - lo) for a, _ in _T2M_ANCHORS]
    base = mcolors.LinearSegmentedColormap.from_list(
        "t2m_base", list(zip(pos, [c for _, c in _T2M_ANCHORS])), N=1024)
    colors = base(np.linspace(0.0, 1.0, len(_T2M_BOUNDS) - 1))
    cmap = mcolors.ListedColormap(colors, name="t2m")
    # Under-color must NOT be white: air colder than -30 °F is real
    # (northern Plains every winter) and would vanish into the map
    # background. Neutral gray reads as "off the cold end of the scale".
    cmap.set_under("#c9ced4")
    cmap.set_over("#ffd9ec")
    return cmap


def t2m_cmap():
    return _t2m_scale()


def t2m_norm():
    return mcolors.BoundaryNorm(_T2M_BOUNDS, len(_T2M_BOUNDS) - 1)


# ── vertically integrated smoke: log-spaced bands, mg m⁻² ────────────────
# Clean air is blank; plumes climb yellow → orange → red → brown → near-
# black, the palette every HRRR-Smoke viewer trains people on. Bounds are
# log-spaced because plume mass spans three orders of magnitude — calibrated
# against a live fire-season field (2026-07-14 18Z: CONUS p50 ≈ 2, p95 ≈ 55,
# p99 ≈ 435, max ≈ 9200 mg/m²): ambient ≲5 stays blank.
SMOKE_BOUNDS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]


def smoke_cmap():
    colors = ["#f7f3d0", "#f2e28a", "#f0c94f", "#eda63b", "#e87a2c",
              "#d94f1e", "#b8331c", "#8c2418", "#5e1a12"]
    cmap = mcolors.ListedColormap(colors, name="smoke")
    cmap.set_over("#1a0a06")
    return cmap


def smoke_norm():
    return mcolors.BoundaryNorm(SMOKE_BOUNDS, len(SMOKE_BOUNDS) - 1)


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
        wind_vectors=True,
        vmin=0, vmax=25,
        cmap_factory=wind_speed_cmap,
    ),
    Variable(
        id="t2m",
        label="2 m Temperature",
        title="HRRR 2 m Temperature",
        units="°F",
        grib_searches=[":TMP:2 m above ground"],
        combine=_passthrough,
        transform=lambda v: (v - 273.15) * 9.0 / 5.0 + 32.0,
        vmin=-30, vmax=116,
        cmap_factory=t2m_cmap,
        norm_factory=t2m_norm,
        point_values=True,
        cbar_extend="both",
        cbar_ticks=list(range(-30, 117, 10)),
        cbar_format="%d",
    ),
    Variable(
        id="smoke",
        label="Vertically Integrated Smoke",
        title="HRRR Vertically Integrated Smoke",
        units="mg/m²",
        grib_searches=[":COLMD:entire atmosphere"],
        # RRFS splits COLMD by aerosol species (HRRR carries one total-smoke
        # column). The wildfire-smoke tracer is the fine particulate organic
        # matter line; \d+ because the forecast-hour spec varies per fxx.
        rrfs_searches=[r":COLMD:entire atmosphere \(considered as a single "
                       r"layer\):(?:anl|\d+ hour fcst):aerosol=Particulate "
                       r"organic matter dry:aerosol_size <2\.5e-06"],
        combine=_passthrough,
        transform=lambda v: v * 1.0e6,        # kg/m² → mg/m²
        vmin=0, vmax=5000,
        cmap_factory=smoke_cmap,
        norm_factory=smoke_norm,
        cbar_extend="max",
        cbar_format="%g",
    ),
    Variable(
        id="solar",
        label="Surface Shortwave (DSWRF)",
        title="HRRR Surface Downward Shortwave",
        units="W/m²",
        grib_searches=[":DSWRF:surface"],
        # RRFS carries BOTH an instantaneous DSWRF ("N hour fcst") and a
        # time-averaged one ("(N-1)-N hour ave fcst"). Match only the
        # instantaneous line to align with HRRR's instantaneous DSWRF.
        rrfs_searches=[r":DSWRF:surface:(?:anl|\d+ hour fcst):"],
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
# Difference products (3-panel HRRR / RRFS / RRFS−HRRR)
# ============================================================================
#
# The RRFS 2dfld.3km.conus grid is bit-identical to the HRRR CONUS grid
# (lat/lon max abs diff exactly 0.0, verified against live files), so the
# difference panel is a pure elementwise subtraction — no regridding.

# ── run-total precip accumulation: NWS-style banded scale, inches ─────────
# Bounds follow the conventional QPE display breaks; below 0.01" is masked
# transparent so dry areas stay blank.
PRECIP_BOUNDS = [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10]


def precip_accum_cmap():
    colors = ["#b4f0b4", "#57c357", "#1d9e1d", "#106b10",   # greens
              "#f9f96b", "#f5c14c", "#ef8433",              # yellow → orange
              "#dc3d24", "#a51d1d",                         # reds
              "#8b34a5", "#d67ae0"]                         # purples
    cmap = mcolors.ListedColormap(colors, name="precip_accum")
    cmap.set_over("#f6c9ee")     # >10": pale magenta, off the top of the scale
    return cmap


def precip_accum_norm():
    return mcolors.BoundaryNorm(PRECIP_BOUNDS, len(PRECIP_BOUNDS) - 1)


# ── difference-panel scales ───────────────────────────────────────────────
# t2m: continuous symmetric diverging, ±6 °F covers the typical model
# spread while saturating real disagreements. precip: banded BrBG (brown =
# RRFS drier, green = RRFS wetter) with |Δ| < 0.05" masked transparent so
# trivial noise doesn't paint the whole map.
PRECIP_DIFF_BOUNDS = [-2, -1, -0.5, -0.25, -0.1, -0.05,
                      0.05, 0.1, 0.25, 0.5, 1, 2]


def t2m_diff_cmap():
    return matplotlib.colormaps["RdBu_r"]


def precip_diff_cmap():
    return matplotlib.colormaps["BrBG"]


def precip_diff_norm():
    # 256 so the banded intervals span the full continuous BrBG ramp.
    return mcolors.BoundaryNorm(PRECIP_DIFF_BOUNDS, 256)


@dataclass
class DiffProduct:
    """A 3-panel HRRR vs RRFS comparison (left HRRR, middle RRFS, right
    RRFS−HRRR).

    `source` is the Variable whose per-model fields feed the panels: it
    supplies the fetch searches + unit transform, and may (t2m) or may
    not (precip) also be rendered as a standalone single-model product.
    Panel 1-2 styling comes from the panel_* fields; the diff panel gets
    its own symmetric diverging scale.
    """
    id: str
    label: str                    # dropdown label
    field_title: str              # "<field>" part of the figure title
    units: str
    source: Variable
    panel_cmap_factory: Callable
    panel_norm_factory: Optional[Callable] = None
    panel_vmin: float = 0.0
    panel_vmax: float = 1.0
    panel_mask_below: Optional[float] = None   # mask small values transparent
    panel_cbar_ticks: Optional[list] = None
    panel_cbar_format: str = "%g"
    panel_cbar_extend: str = "neither"
    diff_cmap_factory: Optional[Callable] = None
    diff_norm_factory: Optional[Callable] = None
    diff_vmin: float = -1.0
    diff_vmax: float = 1.0
    diff_mask_abs_below: Optional[float] = None  # mask |Δ| below this
    diff_cbar_format: str = "%g"


# Run-total precip is fetch-only (feeds precip_diff, never rendered as a
# single-model product, so it's NOT in VARIABLES). Both models carry a
# ":APCP:surface:0-N hour acc fcst" run-total message; the "0-" prefix
# keeps the hourly (N-1)-N bucket from matching. No APCP exists at F00.
_PRECIP_SOURCE = Variable(
    id="precip",
    label="Run-Total Precipitation",
    title="Run-Total Precipitation",
    units="in",
    grib_searches=[":APCP:surface:0-"],
    rrfs_searches=[r":APCP:surface:0-\d+ hour acc fcst"],
    min_fxx=1,
    combine=_passthrough,
    transform=lambda v: v / 25.4,     # kg/m² (= mm) → inches
)

_T2M_SOURCE = next(v for v in VARIABLES if v.id == "t2m")

DIFF_PRODUCTS = [
    DiffProduct(
        id="t2m_diff",
        label="2 m Temperature: RRFS vs HRRR",
        field_title="2 m Temperature",
        units="°F",
        source=_T2M_SOURCE,
        panel_cmap_factory=t2m_cmap,
        panel_norm_factory=t2m_norm,
        panel_cbar_ticks=list(range(-30, 117, 20)),
        panel_cbar_format="%d",
        panel_cbar_extend="both",
        diff_cmap_factory=t2m_diff_cmap,
        diff_vmin=-6.0, diff_vmax=6.0,
        diff_cbar_format="%g",
    ),
    DiffProduct(
        id="precip_diff",
        label="Total Precip: RRFS vs HRRR",
        field_title="Run-Total Precipitation",
        units="in",
        source=_PRECIP_SOURCE,
        panel_cmap_factory=precip_accum_cmap,
        panel_norm_factory=precip_accum_norm,
        panel_mask_below=PRECIP_BOUNDS[0],
        panel_cbar_extend="max",
        diff_cmap_factory=precip_diff_cmap,
        diff_norm_factory=precip_diff_norm,
        diff_mask_abs_below=0.05,
        diff_cbar_format="%g",
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


def searches_for(variable: Variable, model: str) -> list:
    """The idx/Herbie search strings for a (variable, model) pair.

    RRFS defaults to the HRRR strings; only fields whose idx entries
    diverge (smoke, DSWRF, APCP) carry an explicit rrfs_searches.
    """
    if model == "rrfs" and variable.rrfs_searches is not None:
        return variable.rrfs_searches
    return variable.grib_searches


def fetch_model_field(model: str, cycle: datetime, fxx: int, search: str):
    """Dispatch a single-field fetch to the right model backend.

    HRRR keeps the existing Herbie path untouched; RRFS goes through the
    direct idx + byte-range fetcher (rrfs_fetch.py).
    """
    if model == "rrfs":
        return rrfs_fetch.fetch_rrfs_field(cycle, fxx, search)
    return fetch_hrrr_field(cycle, fxx, search)


def marker_size_wind(cap_mw: np.ndarray, region: Region) -> np.ndarray:
    # Markers sized so even the densest clusters (West Texas / Panhandle,
    # which all fall in the Southwest region) read as distinct rings
    # rather than a black blob. The minimum is the key knob: many small
    # farms each drawn at the floor are what merge. Keep it small. The
    # rings are a light location overlay, not meant to obscure the field.
    base = 0.22 if region.id == "national" else 0.36
    return np.clip(base * np.sqrt(np.maximum(cap_mw, 1.0)), 1.3, 6.5) ** 2


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
    base = HERE.parent.parent / "assets" / "wind_forecast_data"
    # Canonical static-fleet file (overwritten each cycle, so git dedups it);
    # fall back to the legacy cycle-stamped name for back-compat.
    capacity_csv = base / "capacity_plant.csv"
    if not capacity_csv.exists():
        capacity_csv = base / f"capacity_plant_{cycle_str}.csv"
    if capacity_csv.exists():
        df = pd.read_csv(capacity_csv)
        print(f"  wind overlay: loaded {len(df):,} plants "
              f"from {capacity_csv.name}", flush=True)
        return df
    # IMPORTANT: do NOT fall back to the raw turbine inventory
    # (uswtdb.csv, ~75k turbines). That database is per-TURBINE with a
    # 't_cap' (kW) column, not per-PLANT, so plotting it dumps tens of
    # thousands of mis-sized rings that blob the whole map. A missing
    # per-cycle CSV means the wind run hasn't committed it yet (an
    # ordering race) — better to render the field + arrows with NO plant
    # rings than to render obviously-wrong ones.
    print(f"  WARN: {capacity_csv.name} not found — rendering wind maps "
          f"WITHOUT plant rings (wind run may not have committed it yet).",
          file=sys.stderr, flush=True)
    return pd.DataFrame()


def load_solar_plants(cycle_str: str):
    base = HERE.parent.parent / "assets" / "solar_forecast_data"
    # The solar ring overlay is STATIC: _overlay_solar_plants sizes rings purely
    # off p_cap_ac (the per-cycle forecast MW is accepted-but-unused). So we load
    # ONLY the static capacity fleet and never the per-cycle forecast_plant CSV —
    # the overlay must not gate or slow the render on per-cycle plant data, and the
    # fleet only changes when it's updated by hand. Returns (cap, None); the None
    # pivot flows through main() as overlay_mw_at_time=None, which the overlay
    # ignores anyway. Legacy cycle-stamped capacity name kept as a fallback.
    capacity_csv = base / "capacity_plant.csv"
    if not capacity_csv.exists():
        capacity_csv = base / f"capacity_plant_{cycle_str}.csv"
    if not capacity_csv.exists():
        print(f"  WARN: {capacity_csv.name} not found — rendering solar maps "
              f"WITHOUT plant markers.", file=sys.stderr, flush=True)
        return pd.DataFrame(), None
    cap = pd.read_csv(capacity_csv)
    print(f"  solar overlay: loaded {len(cap):,} plants "
          f"from {capacity_csv.name} (static)", flush=True)
    return cap, None


# ============================================================================
# Map rendering — generic
# ============================================================================

def _draw_features(ax, dark_borders: bool = True, scale: str = "50m",
                   coast_scale: str = "10m"):
    """Draw state/coastline/border features. Dark for light-background maps
    (wind, ceiling, vis), lighter for dark-background maps (solar).

    `scale` controls Natural Earth resolution for state/border lines: "110m"
    is much faster to draw (fewer vertices) and looks fine at the national
    zoom; "50m" gives crisper lines for the zoomed regional views.

    `coast_scale` is separate (and finer, "10m" by default) so coastlines are
    high-resolution on every plot regardless of the state/border scale — the
    coast is the most-scrutinized line against the data fields. Lakes are NOT
    drawn: a filled lake polygon sits on top of (and masks) the 1 km
    reflectivity field over the Great Lakes.
    """
    state_color = "#222222" if dark_borders else "#aaaaaa"
    line_color = "#000000" if dark_borders else "#dddddd"
    ax.add_feature(cfeature.STATES.with_scale(scale),
                   edgecolor=state_color, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale(coast_scale),
                   edgecolor=line_color, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale(scale),
                   edgecolor=line_color, linewidth=0.6, zorder=3)


# HRRR CONUS Lambert Conformal grid parameters (from GRIB metadata).
# GRIB U/V winds are GRID-relative, not earth-relative. Speed is
# rotation-invariant (so the speed field is fine), but direction arrows
# must rotate U/V to earth-relative or they point up to ~20deg wrong near
# the grid edges. Convergence angle gamma = cone * (lon - LoV), with the
# tangent-cone constant = sin(reference latitude).
_HRRR_LOV = -97.5
_HRRR_CONE = math.sin(math.radians(38.5))


def _grid_to_earth_winds(u, v, lons):
    """Rotate grid-relative (u, v) to earth-relative using the Lambert
    grid convergence angle at each point's longitude."""
    gamma = np.radians(_HRRR_CONE * (lons - _HRRR_LOV))
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    u_e = u * cosg - v * sing
    v_e = u * sing + v * cosg
    return u_e, v_e


def _overlay_wind_arrows(ax, u, v, grid_lats, grid_lons):
    """Draw wind direction arrows (quiver) over the speed field.

    Subsamples the grid to ~WIND_ARROW_COLS arrows across the frame width
    so the arrow density looks consistent whether the region is national
    or a zoomed quadrant. Arrow length scales gently with wind speed; the
    field color already conveys magnitude, so arrows mainly show flow.
    """
    ny, nx = u.shape
    # Stride from target column count; same stride both axes keeps arrows
    # on a square-ish lattice in index space.
    stride = max(1, int(round(nx / WIND_ARROW_COLS)))
    us = u[::stride, ::stride]
    vs = v[::stride, ::stride]
    lons = grid_lons[::stride, ::stride]
    lats = grid_lats[::stride, ::stride]

    # Rotate grid-relative winds to earth-relative so arrows point true.
    us, vs = _grid_to_earth_winds(us, vs, lons)

    ax.quiver(
        lons, lats, us, vs,
        transform=PC, zorder=4,
        color="#1a1a1a", alpha=0.55,
        scale=840,            # larger scale → shorter arrows
        scale_units="width",
        width=0.0011,         # shaft thickness (fraction of axes width)
        headwidth=4, headlength=4.5, headaxislength=4,
        minshaft=1.5, minlength=0,
    )


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
        edgecolors="#0a1a2c", linewidths=0.4,
        alpha=0.62, zorder=5, transform=PC,
    )


def _overlay_solar_plants(ax, plants, plant_mw_at_time, region):
    """Overlay solar plants with white capacity-sized ring markers.

    `plant_mw_at_time` is accepted but unused — kept for signature
    compatibility with the wind overlay path.
    """
    p = plants_within_extent(plants, region.extent)
    if p.empty or "p_cap_ac" not in p.columns:
        return
    mask = (p["p_cap_ac"] >= MIN_SOLAR_PLANT_MW) & \
           p[["xlong", "ylat"]].notna().all(axis=1)
    p = p[mask]
    if p.empty:
        return
    sizes = marker_size_solar(p["p_cap_ac"].values, region)
    ax.scatter(
        p["xlong"].values, p["ylat"].values,
        s=sizes, facecolors="none",
        edgecolors="#ffffff", linewidths=0.85,
        alpha=0.7, zorder=5, transform=PC,
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


# Lattice→gridpoint mapping cache for _overlay_point_values. The HRRR grid
# and each region's slice are fixed, so the nearest-gridpoint search (the
# expensive part: ~130 argmins over a ~120k-cell strided grid) yields the
# same (lon, lat, j, i) list for every frame of a region. Computed once per
# worker process, reused across all its frames. Keyed by (region_id, shape)
# so a grid change can never serve stale indices.
_POINT_LATTICE_CACHE = {}


def _point_lattice(grid_lats, grid_lons, region):
    key = (region.id, grid_lats.shape)
    cached = _POINT_LATTICE_CACHE.get(key)
    if cached is not None:
        return cached
    lon0, lon1, lat0, lat1 = region.extent
    ncol = 40 if region.id == "national" else 34
    nrow = max(5, int(round(ncol * (lat1 - lat0) / max(lon1 - lon0, 1) * 1.4)))
    mlon = (lon1 - lon0) * 0.045
    mlat = (lat1 - lat0) * 0.06
    lons = np.linspace(lon0 + mlon, lon1 - mlon, ncol)
    lats = np.linspace(lat0 + mlat, lat1 - mlat, nrow)
    st = 4                                    # stride: 12 km lookups are plenty
    gla, glo = grid_lats[::st, ::st], grid_lons[::st, ::st]
    coslat = np.cos(np.radians(np.clip(gla, -80, 80)))
    points = []
    for la in lats:
        for lo in lons:
            d2 = (gla - la) ** 2 + ((glo - lo) * coslat) ** 2
            j, i = np.unravel_index(np.argmin(d2), d2.shape)
            if d2[j, i] > 1.0:                # lattice point off the HRRR grid
                continue
            points.append((lo, la, j * st, i * st))
    _POINT_LATTICE_CACHE[key] = points
    return points


def _overlay_point_values(ax, values, grid_lats, grid_lons, region, fmt="%.0f"):
    """Stamp sampled field values on a regular lat/lon lattice — the map
    stops being colour-only and starts answering "what's the number HERE".
    Nearest-gridpoint sampling on a strided copy keeps it O(lattice)."""
    import matplotlib.patheffects as mpe
    halo = [mpe.withStroke(linewidth=0.9, foreground="white")]
    fs = 2.8 if region.id == "national" else 3.2
    for lo, la, j, i in _point_lattice(grid_lats, grid_lons, region):
        v = values[j, i]
        if not np.isfinite(v):
            continue
        ax.text(lo, la, fmt % v, transform=PC, fontsize=fs,
                ha="center", va="center", color="#151515",
                fontweight="bold", zorder=6, path_effects=halo)


def render_map(values: np.ndarray, grid_lats: np.ndarray, grid_lons: np.ndarray,
               valid_time: datetime, fxx: int,
               variable_id: str, region_id: str, cycle: datetime,
               out_path: Path, overlay_records=None,
               overlay_mw_at_time=None, wind_uv=None,
               model: str = "hrrr") -> None:
    """Generic map renderer. Plain numpy arrays in, WebP out.

    Optimizations:
      - per-region array slicing: zoomed regions process ~25-40% of the
        full grid instead of 100% (national uses the full grid)
      - WebP @ quality 82: ~30% smaller than equivalent PNG, lossless-ish
        for these flat-color maps, allows higher DPI without bloat
      - 110m Natural Earth features for national, 50m for zoomed regions
      - cached projection + slice objects per region
      - bbox_inches="tight" crops residual whitespace (cheap now that
        figsizes match each region's geographic aspect ratio)
    """
    variable = next(v for v in VARIABLES if v.id == variable_id)
    region = next(r for r in REGIONS if r.id == region_id)

    cmap = variable.cmap_factory()
    proj = _get_projection(region_id)

    fig, ax = plt.subplots(figsize=region.figsize, dpi=100,
                            subplot_kw=dict(projection=proj))
    ax.set_extent(region.extent, crs=PC)

    # For zoomed regions, slice the field + grid to the region's index
    # bounding box so pcolormesh processes far fewer cells (the full CONUS
    # grid is ~1.9M cells; a quadrant is ~25-40% of that). National uses
    # the full grid. Slicing also yields a writable copy, so the masking
    # below won't mutate shared memory.
    sl = _get_region_slice(region_id, grid_lats, grid_lons)
    if sl is not None:
        values = values[sl]
        grid_lats = grid_lats[sl]
        grid_lons = grid_lons[sl]

    # Slice the wind U/V components the same way (for direction arrows).
    u_arr = v_arr = None
    if wind_uv is not None:
        u_arr, v_arr = wind_uv
        if sl is not None:
            u_arr = u_arr[sl]
            v_arr = v_arr[sl]

    # Ceiling has sentinel values (~20000 m) where no ceiling exists.
    # (np.where returns a fresh array, so this is safe even for the
    # national view where `values` is still a shared-memory view.)
    if variable.id == "ceiling":
        values = np.where(values > 6500, np.nan, values)

    # Reflectivity: mask out "no echo" (low dBZ) so those areas render as
    # the white background instead of faint blue. Standard radar displays
    # show no-precip as blank. 5 dBZ is the conventional threshold.
    if variable.id == "reflectivity":
        values = np.where(values < 5.0, np.nan, values)

    # Smoke: clean air stays blank (white), like every HRRR-Smoke display.
    if variable.id == "smoke":
        values = np.where(values < SMOKE_BOUNDS[0], np.nan, values)

    norm = variable.norm_factory() if variable.norm_factory else None
    mesh_kw = dict(norm=norm) if norm is not None else dict(
        vmin=variable.vmin, vmax=variable.vmax)
    # Fast path: nearest-neighbor regrid onto a regular projected raster
    # (index cached per model+region by julia_bridge) + imshow. ~3-4x faster
    # than curvilinear pcolormesh at HRRR-national size; falls back to
    # pcolormesh if the bridge is unavailable.
    im = None
    if _jb is not None:
        try:
            e = region.extent
            bx = np.concatenate([np.linspace(e[0], e[1], 50), np.full(50, e[1]),
                                 np.linspace(e[1], e[0], 50), np.full(50, e[0])])
            by = np.concatenate([np.full(50, e[2]), np.linspace(e[2], e[3], 50),
                                 np.full(50, e[3]), np.linspace(e[3], e[2], 50)])
            pts = proj.transform_points(PC, bx, by)
            ext_xy = (float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0])),
                      float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1])))
            key = f"{model}_{region.id}"
            idx, mask = _jb._regrid_index(key, grid_lats, grid_lons, proj, ext_xy)
            reg = np.where(mask, np.asarray(values, np.float32).ravel()[idx],
                           np.nan).reshape(880, 1560)
            im = ax.imshow(reg, extent=[ext_xy[0], ext_xy[1], ext_xy[2], ext_xy[3]],
                           origin="lower", cmap=cmap, interpolation="nearest",
                           transform=proj, zorder=1, **mesh_kw)
        except Exception as exc:                               # noqa: BLE001
            print(f"  regrid fast-path failed ({str(exc)[:60]}) — pcolormesh")
            im = None
    if im is None:
        im = ax.pcolormesh(
            grid_lons, grid_lats, values,
            cmap=cmap, transform=PC, shading="auto",
            rasterized=True, zorder=1, **mesh_kw,
        )

    feat_scale = "110m" if region.id == "national" else "50m"
    _draw_features(ax, scale=feat_scale)

    # Wind direction arrows over the speed field. Subsampled to a fixed
    # ~visual density so spacing looks consistent across region zooms.
    if u_arr is not None and v_arr is not None:
        _overlay_wind_arrows(ax, u_arr, v_arr, grid_lats, grid_lons)

    if variable.overlay == "wind" and overlay_records is not None:
        _overlay_wind_plants(ax, overlay_records, region)
    elif variable.overlay == "solar" and overlay_records is not None:
        _overlay_solar_plants(ax, overlay_records, overlay_mw_at_time, region)

    if variable.point_values:
        _overlay_point_values(ax, values, grid_lats, grid_lons, region)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.015, shrink=0.92, fraction=0.040,
                        format=variable.cbar_format,
                        extend=variable.cbar_extend)
    if variable.cbar_ticks:
        cbar.set_ticks(variable.cbar_ticks)
    cbar.set_label(f"{variable.label} ({variable.units})", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle.strftime("%Y-%m-%d %HZ")
    # Variable titles are written for HRRR; the same styling serves RRFS
    # with just the model name swapped (same grid, same field, same units).
    title = (variable.title if model == "hrrr"
             else variable.title.replace("HRRR", "RRFS"))
    ax.set_title(
        f"{title} · {region.label}\n"
        f"Cycle {cycle_str}  ·  F{fxx:02d}  ·  Valid {valid_str}",
        fontsize=12, loc="left", pad=8,
    )

    # bbox_inches="tight" crops the saved image to actual content (map +
    # colorbar + title), eliminating surrounding white canvas from any
    # figsize/extent aspect-ratio mismatch. With figsizes now matched to
    # each region's geographic ratio, there's little to trim, so the cost
    # is small. pad_inches keeps a thin uniform border.
    # WebP @ quality 82 is visually lossless for these flat-color maps.
    fig.savefig(out_path, dpi=140,
                facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)


# National panel row is the reference geometry for diff figures; other
# regions scale their width by geographic aspect so panels stay square-ish.
_NATIONAL_ASPECT = 15.4 / 8.0


def render_diff_map(hrrr_values: np.ndarray, rrfs_values: np.ndarray,
                    grid_lats: np.ndarray, grid_lons: np.ndarray,
                    valid_time: datetime, fxx: int,
                    diff_id: str, region_id: str, cycle: datetime,
                    out_path: Path) -> None:
    """3-panel comparison renderer: HRRR | RRFS | RRFS − HRRR.

    Both fields arrive on the SAME grid (RRFS 2dfld.3km.conus is
    bit-identical to HRRR CONUS), so the right panel is a plain
    elementwise subtraction. Panels 1-2 share the source variable's
    normal colormap and ONE colorbar; the diff panel gets its own
    symmetric diverging scale. Kept on pcolormesh for consistency with
    the single-panel renderer; the region slice is computed once and
    reused for all three panels.
    """
    product = next(d for d in DIFF_PRODUCTS if d.id == diff_id)
    region = next(r for r in REGIONS if r.id == region_id)
    proj = _get_projection(region_id)

    # Width scales with the region's geographic aspect (national = 19.5,
    # clamped so the squarer eastern regions don't collapse too narrow).
    aspect = region.figsize[0] / region.figsize[1]
    fig_w = max(12.0, min(21.0, 19.5 * aspect / _NATIONAL_ASPECT))
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, 5.4), dpi=100,
        subplot_kw=dict(projection=proj), constrained_layout=True)

    # One slice serves all three panels (same grid → same index bbox).
    sl = _get_region_slice(region_id, grid_lats, grid_lons)
    if sl is not None:
        hrrr_values = hrrr_values[sl]
        rrfs_values = rrfs_values[sl]
        grid_lats = grid_lats[sl]
        grid_lons = grid_lons[sl]

    diff = rrfs_values - hrrr_values

    # Mask small forecast values (dry areas for precip) and trivial
    # differences so those cells render as blank background. np.where
    # returns fresh arrays, so the shared-memory inputs are never mutated.
    if product.panel_mask_below is not None:
        hrrr_values = np.where(hrrr_values < product.panel_mask_below,
                               np.nan, hrrr_values)
        rrfs_values = np.where(rrfs_values < product.panel_mask_below,
                               np.nan, rrfs_values)
    if product.diff_mask_abs_below is not None:
        diff = np.where(np.abs(diff) < product.diff_mask_abs_below,
                        np.nan, diff)

    panel_cmap = product.panel_cmap_factory()
    panel_norm = (product.panel_norm_factory()
                  if product.panel_norm_factory else None)
    panel_kw = (dict(norm=panel_norm) if panel_norm is not None
                else dict(vmin=product.panel_vmin, vmax=product.panel_vmax))
    diff_cmap = product.diff_cmap_factory()
    diff_norm = (product.diff_norm_factory()
                 if product.diff_norm_factory else None)
    diff_kw = (dict(norm=diff_norm) if diff_norm is not None
               else dict(vmin=product.diff_vmin, vmax=product.diff_vmax))

    feat_scale = "110m" if region.id == "national" else "50m"
    panels = [
        ("HRRR", hrrr_values, panel_cmap, panel_kw),
        ("RRFS", rrfs_values, panel_cmap, panel_kw),
        ("RRFS − HRRR", diff, diff_cmap, diff_kw),
    ]
    ims = []
    for ax, (name, values, cmap, mesh_kw) in zip(axes, panels):
        ax.set_extent(region.extent, crs=PC)
        ims.append(ax.pcolormesh(
            grid_lons, grid_lats, values,
            cmap=cmap, transform=PC, shading="auto",
            rasterized=True, zorder=1, **mesh_kw,
        ))
        _draw_features(ax, scale=feat_scale)
        ax.set_title(name, fontsize=10, pad=4)

    # ONE shared colorbar for the two forecast panels, a separate one for
    # the diff panel — the two scales are unrelated, so a single bar
    # would be misleading.
    cbar = fig.colorbar(ims[0], ax=list(axes[:2]),
                        orientation="horizontal",
                        fraction=0.055, pad=0.03, aspect=48,
                        format=product.panel_cbar_format,
                        extend=product.panel_cbar_extend)
    if product.panel_cbar_ticks:
        cbar.set_ticks(product.panel_cbar_ticks)
    cbar.set_label(f"{product.field_title} ({product.units})", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    dbar = fig.colorbar(ims[2], ax=axes[2],
                        orientation="horizontal",
                        fraction=0.055, pad=0.03, aspect=24,
                        format=product.diff_cbar_format, extend="both")
    dbar.set_label(f"RRFS − HRRR ({product.units})", fontsize=9)
    dbar.ax.tick_params(labelsize=8)

    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle.strftime("%Y-%m-%d %HZ")
    fig.suptitle(
        f"{product.field_title} — HRRR vs RRFS · {region.label} · "
        f"Cycle {cycle_str}  ·  F{fxx:02d}  ·  Valid {valid_str}",
        fontsize=12,
    )

    fig.savefig(out_path, dpi=140,
                facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.08,
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)


# Projection objects are expensive to construct (~50-100ms each).
# Cache them per region so each render reuses the same instance.
_PROJ_CACHE = {}


def _get_projection(region_id: str):
    if region_id not in _PROJ_CACHE:
        region = next(r for r in REGIONS if r.id == region_id)
        _PROJ_CACHE[region_id] = ccrs.LambertConformal(
            central_longitude=region.proj_lon,
            central_latitude=region.proj_lat,
            standard_parallels=region.standard_parallels,
        )
    return _PROJ_CACHE[region_id]


# Per-region index-slice cache. HRRR's grid is curvilinear (2D lat/lon),
# so a geographic bbox maps to a skewed region of index space — we take
# the index bounding box that contains all in-extent cells, plus a margin
# so edge quads render fully. Computed once per region from the (fixed)
# grid, then reused for every frame of that region. Zoomed regions end up
# processing ~25-40% of the full grid instead of 100%.
_SLICE_CACHE = {}


def _get_region_slice(region_id: str, grid_lats: np.ndarray,
                      grid_lons: np.ndarray):
    """Return (row_slice, col_slice) bounding the region's extent in the
    grid index space, or None for the national view (use full grid).

    Cached per region. `grid_lats`/`grid_lons` are the full 2D arrays.
    """
    if region_id == "national":
        return None
    if region_id in _SLICE_CACHE:
        return _SLICE_CACHE[region_id]

    region = next(r for r in REGIONS if r.id == region_id)
    w, e, s, n = region.extent
    # Margin in degrees so edge cells/quads aren't clipped at the border.
    mlon, mlat = 1.0, 1.0
    inside = (
        (grid_lons >= w - mlon) & (grid_lons <= e + mlon) &
        (grid_lats >= s - mlat) & (grid_lats <= n + mlat)
    )
    if not inside.any():
        # Fallback: no cells matched (shouldn't happen) — use full grid
        _SLICE_CACHE[region_id] = None
        return None

    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    sl = (slice(r0, r1), slice(c0, c1))
    _SLICE_CACHE[region_id] = sl
    return sl


def _render_task(args: dict) -> str:
    """Worker-side entry point (single-model AND diff tasks).

    Single tasks: `args["values"]` is the per-field ndarray, passed via
    task pickling. Diff tasks: BOTH models' fields ride the same
    SharedMemory mechanism as the grid (two shared arrays per
    (product, hour), referenced by all 5 region tasks instead of being
    pickled 5x each). Grid arrays are always shared and attached here.
    """
    # Attach shared grid arrays (these are reused across all tasks)
    grid_lats_arr, lats_shm = _attach_array(args["grid_lats_ref"])
    grid_lons_arr, lons_shm = _attach_array(args["grid_lons_ref"])
    extra_shms = []
    try:
        if args.get("kind") == "diff":
            hrrr_arr, hrrr_shm = _attach_array(args["hrrr_ref"])
            extra_shms.append(hrrr_shm)
            rrfs_arr, rrfs_shm = _attach_array(args["rrfs_ref"])
            extra_shms.append(rrfs_shm)
            render_diff_map(
                hrrr_values=hrrr_arr,
                rrfs_values=rrfs_arr,
                grid_lats=grid_lats_arr,
                grid_lons=grid_lons_arr,
                valid_time=args["valid_time"],
                fxx=args["fxx"],
                diff_id=args["diff_id"],
                region_id=args["region_id"],
                cycle=args["cycle"],
                out_path=args["out_path"],
            )
        else:
            staged = False
            if (_jb is not None and _jb.available()
                    and args.get("overlay_records") is None
                    and args.get("wind_uv") is None):
                try:
                    _stage_for_julia(args, grid_lats_arr, grid_lons_arr)
                    staged = True
                except Exception as e:                         # noqa: BLE001
                    print(f"  julia staging failed ({str(e)[:60]}) — matplotlib")
            if not staged:
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
                    wind_uv=args.get("wind_uv"),
                    model=args.get("model", "hrrr"),
                )
    finally:
        for shm in extra_shms:
            shm.close()
        lats_shm.close()
        lons_shm.close()
    return args["out_id"]


def _stage_for_julia(args, grid_lats, grid_lons):
    """Serialize one plain frame for the Julia rasterizer (visual parity with
    render_map: same slicing, sentinel masking, colormap, titles)."""
    variable = next(v for v in VARIABLES if v.id == args["variable_id"])
    if variable.point_values:
        raise RuntimeError("point-value overlay frame")
    region = next(r for r in REGIONS if r.id == args["region_id"])
    values = args["values"]
    sl = _get_region_slice(region.id, grid_lats, grid_lons)
    if sl is not None:
        values = values[sl]
        grid_lats = grid_lats[sl]
        grid_lons = grid_lons[sl]
    if variable.id == "ceiling":
        values = np.where(values > 6500, np.nan, values)
    if variable.id == "reflectivity":
        values = np.where(values < 5.0, np.nan, values)
    if variable.id == "smoke":
        values = np.where(values < SMOKE_BOUNDS[0], np.nan, values)
    proj = _get_projection(region.id)
    cmap = variable.cmap_factory()
    norm = variable.norm_factory() if variable.norm_factory else None
    spec = _jb.serialize_cmap(cmap, norm, variable.vmin, variable.vmax)
    # projected bbox of the PlateCarree extent rectangle (matplotlib parity)
    e = region.extent
    bx = np.concatenate([np.linspace(e[0], e[1], 50), np.full(50, e[1]),
                         np.linspace(e[1], e[0], 50), np.full(50, e[0])])
    by = np.concatenate([np.full(50, e[2]), np.linspace(e[2], e[3], 50),
                         np.full(50, e[3]), np.linspace(e[3], e[2], 50)])
    pts = proj.transform_points(PC, bx, by)
    extent_xy = (float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0])),
                 float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1])))
    feat_scale = "110m" if region.id == "national" else "50m"
    overlays = _jb.ensure_overlays(region.id, proj, feat_scale)
    model = args.get("model", "hrrr")
    title = variable.title if model == "hrrr" else variable.title.replace("HRRR", "RRFS")
    valid_str = args["valid_time"].strftime("%Y-%m-%d %H:%MZ")
    cycle_str = args["cycle"].strftime("%Y-%m-%d %HZ")
    full_title = (f"{title} · {region.label}\n"
                  f"Cycle {cycle_str}  ·  F{args['fxx']:02d}  ·  Valid {valid_str}")
    frame_id = Path(args["out_path"]).stem + "_" + region.id + "_" + model
    _jb.stage_frame(frame_id=frame_id, values=values, lats=grid_lats,
                    lons=grid_lons, proj=proj, extent_xy=extent_xy,
                    cmap_spec=spec, title=full_title,
                    cbar_label=f"{variable.label} ({variable.units})",
                    cbar_ticks=variable.cbar_ticks, figsize=region.figsize,
                    out_path=str(args["out_path"]), overlays_npz=overlays)


# ============================================================================
# Driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?")
    parser.add_argument("hour", nargs="?")
    parser.add_argument("--variables",
                        help="Comma-separated product IDs (base variables, "
                             "rrfs_* variants, and *_diff products)")
    parser.add_argument("--regions", help="Comma-separated region IDs")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of render worker processes (0 = auto, all cores)")
    # Env-var gate so CI can pin HRRR-only (SYNOPTIC_MODELS=hrrr) without
    # touching the invocation; local runs default to both models + diffs.
    parser.add_argument("--models",
                        default=os.environ.get("SYNOPTIC_MODELS", "hrrr,rrfs"),
                        help="Comma-separated models: hrrr,rrfs (default both; "
                             "diff products render only when both are present)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ("hrrr", "rrfs")]
    if bad or not models:
        parser.error(f"--models must be a subset of hrrr,rrfs (got {args.models!r})")

    if args.date and args.hour:
        cycle = datetime.strptime(f"{args.date} {args.hour}", "%Y-%m-%d %H")
    else:
        cycle = find_latest_extended_cycle()
    print(f"Cycle: {cycle:%Y-%m-%d %H:%MZ}")
    cycle_str = cycle.strftime("%Y%m%dT%HZ")

    # RRFS publication lags HRRR; we deliberately use the SAME cycle for
    # both models (apples-to-apples diffs) and just drop the RRFS half
    # when that cycle isn't on the bucket yet — HRRR proceeds exactly as
    # before, and the next run picks RRFS up.
    if "rrfs" in models and not rrfs_fetch.fetch_rrfs_availability(cycle):
        print(f"WARN: RRFS cycle {cycle:%Y-%m-%d %HZ} not available yet — "
              f"proceeding HRRR-only.", file=sys.stderr, flush=True)
        models = [m for m in models if m != "rrfs"]

    # ---- Product selection -------------------------------------------------
    # Three product families share one render pipeline:
    #   <var>       single-panel HRRR   (output layout unchanged)
    #   rrfs_<var>  single-panel RRFS   (assets/synoptic/rrfs_<var>/...)
    #   <x>_diff    3-panel HRRR/RRFS/difference (needs BOTH models)
    wanted = set(args.variables.split(",")) if args.variables else None
    hrrr_vars = [v for v in VARIABLES
                 if "hrrr" in models and (wanted is None or v.id in wanted)]
    rrfs_vars = [v for v in VARIABLES
                 if "rrfs" in models
                 and (wanted is None or f"rrfs_{v.id}" in wanted)]
    diffs = [d for d in DIFF_PRODUCTS
             if "hrrr" in models and "rrfs" in models
             and (wanted is None or d.id in wanted)]

    regions = REGIONS
    if args.regions:
        wanted_r = set(args.regions.split(","))
        regions = [r for r in regions if r.id in wanted_r]

    # Everything below keys renders + manifests off `product_meta`:
    # (out_id, dropdown label, units) per rendered product.
    product_meta = (
        [{"out_id": v.id, "label": v.label, "units": v.units}
         for v in hrrr_vars]
        + [{"out_id": f"rrfs_{v.id}", "label": f"RRFS: {v.label}",
            "units": v.units} for v in rrfs_vars]
        + [{"out_id": d.id, "label": d.label, "units": d.units}
           for d in diffs]
    )

    # Which base variables each model must actually download. Diff
    # products piggyback their source variable onto both models' lists;
    # the dict-by-id keeps t2m from being fetched twice when both the
    # single-panel product and t2m_diff are requested.
    fetch_specs = {m: {} for m in models}
    for v in hrrr_vars:
        fetch_specs["hrrr"][v.id] = v
    for v in rrfs_vars:
        fetch_specs["rrfs"][v.id] = v
    for d in diffs:
        for m in ("hrrr", "rrfs"):
            fetch_specs[m][d.source.id] = d.source

    # Aggressive default: use ALL available cores. User said memory is
    # not a constraint, so we don't hold back.
    if args.workers > 0:
        n_workers = args.workers
    else:
        n_workers = max(1, os.cpu_count() or 4)

    print(f"Models ({len(models)}): {models}")
    print(f"Products ({len(product_meta)}): {[p['out_id'] for p in product_meta]}")
    print(f"Regions ({len(regions)}): {[r.id for r in regions]}")
    print(f"Workers: {n_workers} (out of {os.cpu_count()} cores)")

    # Plant data (overlays apply to both models' single-panel renders)
    need_wind = any(v.overlay == "wind" for v in hrrr_vars + rrfs_vars)
    need_solar = any(v.overlay == "solar" for v in hrrr_vars + rrfs_vars)
    wind_plants = load_wind_plants(cycle_str) if need_wind else pd.DataFrame()
    if need_wind:
        print(f"Loaded {len(wind_plants):,} wind plants")
    solar_plants, solar_pivot = (load_solar_plants(cycle_str)
                                  if need_solar else (pd.DataFrame(), None))
    if need_solar:
        print(f"Loaded {len(solar_plants):,} solar plants")

    for p in product_meta:
        for r in regions:
            (ASSETS / p["out_id"] / r.id).mkdir(parents=True, exist_ok=True)
    manifests = {(p["out_id"], r.id): []
                 for p in product_meta for r in regions}

    t_total = time_module.time()

    # PHASE 1: Fetch all model fields up front, sequentially.
    n_field_hours = sum(
        len([f for f in FORECAST_HOURS if f >= v.min_fxx])
        for m in models for v in fetch_specs[m].values())
    print(f"\n[1/2] Fetching {n_field_hours} field-hours...")
    t_fetch = time_module.time()

    # Memory strategy:
    #   - Grid lats/lons are identical across all (model, variable, hour)
    #     tuples — the RRFS 2dfld grid is bit-identical to HRRR CONUS —
    #     so we put them in shared memory ONCE and reference them in every
    #     task. Saves huge duplication.
    #   - Each field's data array is passed as a regular task argument
    #     (pickled per task). With bounded worker queue, only a few tasks
    #     are in flight at once, so this caps memory naturally and avoids
    #     /dev/shm size limits (typically 64 MB on Linux containers).
    #   - Diff products are the exception: each (product, hour) pair's two
    #     model fields go into shared memory once and are referenced by
    #     all 5 region tasks (see the task-build phase below).
    field_cache = {}   # (model, variable_id, fxx) → values ndarray (float32)
    wind_uv_cache = {} # (model, variable_id, fxx) → (u_arr, v_arr) for arrows
    grid_lats_ref = None
    grid_lons_ref = None
    shm_names_to_release = []

    n_fetched = 0
    n_skipped = 0
    for fxx in FORECAST_HOURS:
        for model in models:
            for variable in fetch_specs[model].values():
                if fxx < variable.min_fxx:
                    continue           # e.g. APCP does not exist at F00
                searches = searches_for(variable, model)
                fields = {}
                for search in searches:
                    arr = fetch_model_field(model, cycle, fxx, search)
                    if arr is None:
                        fields = None
                        break
                    fields[search] = arr
                if fields is None:
                    field_cache[(model, variable.id, fxx)] = None
                    n_skipped += 1
                    continue
                combined = variable.combine(fields, searches)
                if combined is None:
                    field_cache[(model, variable.id, fxx)] = None
                    n_skipped += 1
                    continue

                # Shared-memory grid (built lazily on first fetch; either
                # model can seed it since the grids are identical)
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
                if variable.transform is not None:
                    values = np.ascontiguousarray(variable.transform(values),
                                                  dtype=np.float32)
                field_cache[(model, variable.id, fxx)] = values
                # If this variable draws direction arrows, stash its U/V too.
                if variable.wind_vectors:
                    u = combined.attrs.get("_wind_u")
                    v = combined.attrs.get("_wind_v")
                    if u is not None and v is not None:
                        wind_uv_cache[(model, variable.id, fxx)] = (u, v)
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
    single_products = ([("hrrr", v) for v in hrrr_vars]
                       + [("rrfs", v) for v in rrfs_vars])
    for model, variable in single_products:
        # HRRR keeps its original output layout (assets/synoptic/<var>/)
        # so existing URLs/manifests never move; RRFS lands beside it
        # under a rrfs_ prefix.
        out_id = variable.id if model == "hrrr" else f"rrfs_{variable.id}"
        overlay_records = (wind_plants if variable.overlay == "wind"
                            else solar_plants if variable.overlay == "solar"
                            else None)
        for fxx in FORECAST_HOURS:
            values = field_cache.get((model, variable.id, fxx))
            if values is None:
                continue
            uv = wind_uv_cache.get((model, variable.id, fxx))  # None unless wind
            valid_time = cycle + timedelta(hours=fxx)
            for region in regions:
                tasks.append({
                    "kind": "single",
                    "model": model,
                    "out_id": out_id,
                    "values": values,
                    "wind_uv": uv,
                    "grid_lats_ref": grid_lats_ref,
                    "grid_lons_ref": grid_lons_ref,
                    "valid_time": valid_time,
                    "fxx": fxx,
                    "variable_id": variable.id,
                    "region_id": region.id,
                    "cycle": cycle,
                    "out_path": ASSETS / out_id / region.id / f"F{fxx:02d}.webp",
                    "overlay_records": overlay_records,
                    "overlay_mw_at_time": (mw_by_hour[fxx]
                                            if variable.overlay == "solar" else None),
                })

    # Diff tasks: only hours where BOTH models' fields were fetched. The
    # two fields ride the same SharedMemory mechanism as the grid — one
    # shared copy per (product, hour) serves all 5 region tasks instead
    # of pickling both CONUS arrays 5x each (~15 MB/hour-product in shm;
    # only materializes when both models actually ran).
    for d in diffs:
        for fxx in FORECAST_HOURS:
            if fxx < d.source.min_fxx:
                continue           # e.g. no APCP at F00 → skip precip_diff F00
            hrrr_values = field_cache.get(("hrrr", d.source.id, fxx))
            rrfs_values = field_cache.get(("rrfs", d.source.id, fxx))
            if hrrr_values is None or rrfs_values is None:
                continue
            hrrr_ref = _share_array(hrrr_values)
            rrfs_ref = _share_array(rrfs_values)
            shm_names_to_release.append(hrrr_ref["name"])
            shm_names_to_release.append(rrfs_ref["name"])
            valid_time = cycle + timedelta(hours=fxx)
            for region in regions:
                tasks.append({
                    "kind": "diff",
                    "out_id": d.id,
                    "diff_id": d.id,
                    "hrrr_ref": hrrr_ref,
                    "rrfs_ref": rrfs_ref,
                    "grid_lats_ref": grid_lats_ref,
                    "grid_lons_ref": grid_lons_ref,
                    "valid_time": valid_time,
                    "fxx": fxx,
                    "region_id": region.id,
                    "cycle": cycle,
                    "out_path": ASSETS / d.id / region.id / f"F{fxx:02d}.webp",
                })

    # Sort tasks by (region, product, fxx) so each worker tends to stay
    # on the same region for several consecutive tasks. Helps cartopy
    # reuse its internal projection state across renders.
    tasks.sort(key=lambda t: (t["region_id"], t["out_id"], t["fxx"]))

    print(f"  {len(tasks)} render tasks queued ({n_workers} workers)")

    # Pre-warm cartopy's shapefile cache in the main process so workers
    # don't each download independently (each worker would otherwise
    # download ~150 MB the first time it draws a coastline/state border).
    _prewarm_cartopy_cache()

    t_render = time_module.time()
    n_done = 0
    n_failed = 0
    progress_every = max(1, len(tasks) // 40)

    # Use "spawn" context to avoid fork-related SharedMemory resource-tracker
    # issues. With fork, workers inherit the parent's resource tracker, which
    # can cause SHM blocks to be cleaned up early when one worker exits.
    # Spawn starts each worker fresh, no inheritance.
    mp_ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_render_task, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    n_done += 1
                    manifests[(task["out_id"], task["region_id"])].append({
                        "fxx": task["fxx"],
                        "valid_time": task["valid_time"].isoformat() + "Z",
                        "valid_label": task["valid_time"].strftime("%a %m/%d %H:%MZ"),
                        "file": f"F{task['fxx']:02d}.webp",
                    })
                except Exception as e:
                    n_failed += 1
                    print(f"  ERROR {task['out_id']}/{task['region_id']}/F{task['fxx']:02d}: {e}",
                          file=sys.stderr)
                if n_done % progress_every == 0:
                    pct = 100 * n_done / len(tasks)
                    rate = n_done / max(time_module.time() - t_render, 1)
                    eta = (len(tasks) - n_done) / max(rate, 0.1)
                    print(f"    {n_done}/{len(tasks)} ({pct:.0f}%) "
                          f"rate={rate:.1f}/s eta={eta:.0f}s",
                          flush=True)
    finally:
        print(f"  releasing {len(shm_names_to_release)} shared-memory blocks...")
        _release_shared(shm_names_to_release)

    # Julia batch pass over frames the workers staged instead of rendering.
    if _jb is not None and _jb.available():
        done, failed = _jb.render_staged()
        for fid in failed:
            try:
                meta, vals = _jb.load_failed(fid)
                # fall back through matplotlib using the staged arrays: the
                # frame_id encodes out stem + region + model; re-render from
                # the original task list entry
                cand = [tk for tk in tasks
                        if Path(tk["out_path"]).stem + "_" + tk["region_id"]
                        + "_" + tk.get("model", "hrrr") == fid]
                if cand:
                    tk = cand[0]
                    render_map(values=tk["values"], grid_lats=None,
                               grid_lons=None, valid_time=tk["valid_time"],
                               fxx=tk["fxx"], variable_id=tk["variable_id"],
                               region_id=tk["region_id"], cycle=tk["cycle"],
                               out_path=tk["out_path"], model=tk.get("model", "hrrr"))
            except Exception as e:                             # noqa: BLE001
                print(f"  fallback failed for {fid}: {str(e)[:80]}")
            finally:
                _jb.clear_frame(fid)

    print(f"  rendered in {time_module.time() - t_render:.0f}s "
          f"({n_done} OK, {n_failed} failed)")

    # Sort manifest frames by fxx (parallel completion order is arbitrary)
    for key in manifests:
        manifests[key].sort(key=lambda x: x["fxx"])

    # Per-(product, region) manifests
    for p in product_meta:
        for r in regions:
            manifest = {
                "cycle": cycle.strftime("%Y-%m-%d %HZ"),
                "cycle_compact": cycle_str,
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "variable_id": p["out_id"],
                "variable_label": p["label"],
                "units": p["units"],
                "region_id": r.id,
                "region_label": r.label,
                "frames": manifests[(p["out_id"], r.id)],
            }
            (ASSETS / p["out_id"] / r.id / "manifest.json").write_text(
                json.dumps(manifest, indent=2))

    # Top-level dropdown metadata. Only rewritten on UNFILTERED runs: a
    # partial run (--variables/--regions, e.g. a local test or one-field
    # backfill) would otherwise shrink the live dropdowns to just the
    # subset it rendered, hiding every other variable's existing frames.
    # The FULL catalog (HRRR + RRFS + diffs) is written regardless of
    # --models for the same reason: an HRRR-only CI fallback run must not
    # hide the RRFS/diff frames the local pipeline already rendered.
    if not args.variables:
        entries = [
            {"id": v.id, "label": v.label, "units": v.units,
             "default": (v.id == "wind"), "group": "HRRR"}
            for v in VARIABLES
        ]
        entries += [
            {"id": f"rrfs_{v.id}", "label": f"RRFS: {v.label}",
             "units": v.units, "default": False, "group": "RRFS"}
            for v in VARIABLES
        ]
        entries += [
            {"id": d.id, "label": d.label, "units": d.units,
             "default": False, "group": "RRFS vs HRRR"}
            for d in DIFF_PRODUCTS
        ]
        (ASSETS / "variables.json").write_text(
            json.dumps({"variables": entries}, indent=2))

    if not args.regions:
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
