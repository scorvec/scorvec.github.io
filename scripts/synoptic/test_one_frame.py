#!/usr/bin/env python3
"""Render ONE wind frame to test the direction arrows before a full run.

Usage (from scripts/synoptic/, same dir as render_maps.py):
    python3 test_one_frame.py 2026-05-29 18 --region southwest --fxx 12

Writes the single frame to /tmp/test_wind_frame.webp and prints the path.
Pick a daytime/windy fxx and a region with lots of turbines (southwest =
Texas/Plains) so the arrows are easy to see.
"""
import sys
import argparse
from datetime import datetime, timedelta

# Import the real machinery from render_maps.py
import render_maps as rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD cycle date")
    ap.add_argument("hour", type=int, help="cycle hour (0/6/12/18)")
    ap.add_argument("--region", default="southwest",
                    help="region id (default southwest)")
    ap.add_argument("--fxx", type=int, default=12,
                    help="forecast hour to render (default 12)")
    ap.add_argument("--out", default="/tmp/test_wind_frame.webp")
    args = ap.parse_args()

    cycle = datetime.strptime(f"{args.date} {args.hour:02d}", "%Y-%m-%d %H")
    fxx = args.fxx

    # Prewarm cartopy once (downloads shapefiles if not cached)
    rm._prewarm_cartopy_cache()

    # Find the wind variable
    variable = next(v for v in rm.VARIABLES if v.id == "wind")
    region = next(r for r in rm.REGIONS if r.id == args.region)

    print(f"Fetching wind U/V for {cycle} F{fxx:02d} ...")
    fields = {}
    for search in variable.grib_searches:
        arr = rm.fetch_hrrr_field(cycle, fxx, search)
        if arr is None:
            print(f"  FAILED to fetch: {search}")
            sys.exit(1)
        fields[search] = arr
    combined = variable.combine(fields, variable.grib_searches)
    if combined is None:
        print("  combine returned None")
        sys.exit(1)

    import numpy as np
    values = np.ascontiguousarray(combined.values, dtype=np.float32)
    u = combined.attrs.get("_wind_u")
    v = combined.attrs.get("_wind_v")
    print(f"  U/V present: {u is not None and v is not None}")

    grid_lats = np.ascontiguousarray(combined.latitude.values, dtype=np.float32)
    grid_lons_raw = np.ascontiguousarray(combined.longitude.values, dtype=np.float32)
    grid_lons = np.where(grid_lons_raw > 180, grid_lons_raw - 360, grid_lons_raw)

    # Load wind plants so the rings show too (so you see arrows + rings)
    try:
        wind_plants = rm.load_wind_plants(cycle.strftime("%Y%m%dT%HZ"))
        print(f"  loaded {len(wind_plants):,} wind plants")
    except Exception as e:
        print(f"  (no wind plants loaded: {e})")
        wind_plants = None

    valid_time = cycle + timedelta(hours=fxx)
    print(f"Rendering {args.region} F{fxx:02d} → {args.out}")
    rm.render_map(
        values=values,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        valid_time=valid_time,
        fxx=fxx,
        variable_id="wind",
        region_id=args.region,
        cycle=cycle,
        out_path=args.out,
        overlay_records=wind_plants,
        overlay_mw_at_time=None,
        wind_uv=(u, v),
    )
    print(f"\nDone. Open {args.out} to inspect the arrows.")


if __name__ == "__main__":
    main()
