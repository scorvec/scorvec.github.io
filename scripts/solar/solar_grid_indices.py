"""Pre-compute HRRR grid indices for every plant in the USPVDB inventory.

The HRRR grid is fixed (Lambert Conformal Conic, 3km resolution, 1059x1799 cells).
So the nearest grid cell for each plant is also fixed. Compute it once,
cache as CSV, reuse forever (or until HRRR ever changes its grid).

Usage:
    python solar_grid_indices.py            # compute and cache
    python solar_grid_indices.py --refresh  # force recompute
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from herbie import Herbie
from datetime import datetime, timedelta, timezone

from solar_inventory import load_uspvdb


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
GRID_INDEX_CSV = DATA_DIR / "uspvdb_grid_indices.csv"


def find_recent_cycle() -> datetime:
    """Find any recent HRRR cycle (just need the grid, not specific data)."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                              microsecond=0, tzinfo=None)
    for hours_back in range(2, 10):
        candidate = now - timedelta(hours=hours_back)
        try:
            H = Herbie(candidate, model="hrrr", product="sfc", fxx=0)
            if H.grib is not None:
                return candidate
        except Exception:
            continue
    raise RuntimeError("No recent HRRR cycle available")


def build_grid_indices(force: bool = False) -> pd.DataFrame:
    """Compute (iy, ix) HRRR grid indices for each plant in USPVDB.

    Returns DataFrame with columns: case_id, ylat, xlong, iy, ix,
                                     grid_lat, grid_lon
    """
    if GRID_INDEX_CSV.exists() and not force:
        print(f"  Cached: {GRID_INDEX_CSV}")
        return pd.read_csv(GRID_INDEX_CSV)

    print(f"  Building HRRR grid indices for USPVDB plants...")
    inv = load_uspvdb()
    print()
    print(f"  Loading HRRR grid from recent cycle...")
    cycle = find_recent_cycle()
    H = Herbie(cycle, model="hrrr", product="sfc", fxx=0)
    # Fetch any small field just to get the grid coordinates
    ds = H.xarray(":TMP:2 m above ground")
    var = list(ds.data_vars)[0]
    field = ds[var]
    grid_lats = field.latitude.values   # 2D, shape (ny, nx)
    grid_lons = field.longitude.values  # 2D, shape (ny, nx), in 0-360 convention
    print(f"  HRRR grid: {grid_lats.shape} cells")

    print(f"  Computing nearest cell for {len(inv):,} plants...")
    lats = inv["ylat"].values
    lons = inv["xlong"].values
    lons_360 = lons % 360.0

    iy_arr = np.empty(len(inv), dtype=np.int32)
    ix_arr = np.empty(len(inv), dtype=np.int32)
    glat_arr = np.empty(len(inv), dtype=np.float32)
    glon_arr = np.empty(len(inv), dtype=np.float32)

    # We can either do brute-force (slow but simple) or use cKDTree.
    # KDTree is much faster but requires care with the haversine metric.
    # For ~6k plants × 2M grid cells, brute-force takes ~1 min once.
    # Since we cache the result, this is fine.
    for i, (lat, lon) in enumerate(zip(lats, lons_360)):
        d2 = (grid_lats - lat) ** 2 + (grid_lons - lon) ** 2
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        iy_arr[i] = iy
        ix_arr[i] = ix
        glat_arr[i] = grid_lats[iy, ix]
        glon_arr[i] = grid_lons[iy, ix]
        if (i + 1) % 1000 == 0:
            print(f"    ... {i+1:,}/{len(inv):,} plants")

    out = pd.DataFrame({
        "case_id": inv["case_id"].values,
        "ylat": lats,
        "xlong": lons,
        "iy": iy_arr,
        "ix": ix_arr,
        "grid_lat": glat_arr,
        "grid_lon": glon_arr,
    })
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(GRID_INDEX_CSV, index=False)

    # Sanity check: how far is each plant from its assigned grid cell?
    # 3 km HRRR grid means we expect ~1.5 km mean offset
    out["dlat_km"] = (out["grid_lat"] - out["ylat"]) * 111.0
    # cos(lat) for longitude scaling
    cos_lat = np.cos(np.deg2rad(out["ylat"]))
    out["dlon_km"] = (out["grid_lon"] - (out["xlong"] % 360)) * 111.0 * cos_lat
    out["dist_km"] = np.sqrt(out["dlat_km"]**2 + out["dlon_km"]**2)
    print(f"\n  Cached {len(out):,} grid indices: {GRID_INDEX_CSV}")
    print(f"  Distance to nearest cell:")
    print(f"    mean:   {out['dist_km'].mean():.2f} km")
    print(f"    median: {out['dist_km'].median():.2f} km")
    print(f"    p95:    {out['dist_km'].quantile(0.95):.2f} km")
    print(f"    max:    {out['dist_km'].max():.2f} km")

    return out.drop(columns=["dlat_km", "dlon_km", "dist_km"])


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    print("=" * 70)
    print("USPVDB → HRRR grid index builder")
    print("=" * 70)
    df = build_grid_indices(force=force)
    print(f"\nSample (5 rows):")
    print(df.head(5).to_string(index=False))
