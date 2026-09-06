#!/usr/bin/env python3
"""West-coast landfall transect for the AR monitor — built once, committed.

For each 0.25° latitude from 24°N to 61°N, the first land longitude scanning east from
165°W across the North American mainland (Vancouver Island and Haida Gwaii count as coast;
small islands and the Alaska Peninsula's Aleutian tail do not), and one grid point offshore
of it — the point where landfalling IVT is read. Plus a set of named locations used for the
AR-scale table. Land from Natural Earth 50 m polygons.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from cartopy.io import shapereader

OUT = Path(__file__).resolve().parent / "data" / "coast_transect.json"
NAMED = [  # name, lat, lon(°E) of the offshore reading point
    ("Baja / Ensenada", 31.75, -117.0), ("San Diego", 32.75, -117.5), ("Los Angeles", 34.0, -119.0), ("Santa Barbara", 34.5, -120.5),
    ("San Francisco Bay", 37.75, -123.0), ("Eureka / North Coast", 40.75, -124.5), ("Oregon coast", 44.5, -124.5), ("Astoria / Columbia mouth", 46.25, -124.5),
    ("Olympic Peninsula", 47.75, -125.0), ("Vancouver Island", 49.25, -126.5), ("Haida Gwaii", 53.25, -133.0), ("Southeast Alaska", 57.0, -136.5), ("Prince William Sound", 60.0, -147.0),
]

land = unary_union([g for g in shapereader.Reader(shapereader.natural_earth("50m", "physical", "land")).geometries()])
big = [g for g in getattr(land, "geoms", [land]) if g.area > 0.8]      # mainland + Vancouver Island, Haida Gwaii, the SE Alaska islands (deg²)
mainland = prep(unary_union(big))
rows = []
for lat in np.arange(24.0, 58.01, 0.25):                       # north of ~58°N the coast turns east–west (Gulf of Alaska): named points cover it
    coast = None
    start = -142.0 if lat >= 54 else -132.0 if lat >= 48 else -128.0
    for lon in np.arange(start, -95.0, 0.25):
        if mainland.contains(Point(lon, lat)):
            coast = float(lon); break
    if coast is None:
        continue
    rows.append({"lat": round(float(lat), 2), "coast_lon": coast, "point_lon": round(coast - 0.5, 2)})
OUT.write_text(json.dumps({"transect": rows, "named": [dict(name=n, lat=la, lon=lo) for n, la, lo in NAMED],
                           "note": "first land longitude (Natural Earth 50m pieces > 0.8 deg^2, so Vancouver Island / Haida Gwaii / SE Alaska islands count) scanning east per 0.25 deg latitude, 24-58N; point_lon = 0.5 deg offshore"}, indent=0))
print(f"{len(rows)} latitudes; sample:", rows[::20])
