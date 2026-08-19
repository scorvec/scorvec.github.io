#!/usr/bin/env python3
"""True upstream catchment delineation from dam outlets (HydroBASINS lev-12).

The shipped xm_river_catchments.geojson assigns each XM river an IDEAM
subzona hidrografica (SZH) polygon. SZHs are administrative drainage
units, not contributing areas, and several XM rivers share one: ANTIOQUIA
has 16 rivers but only **5 distinct shapes**. Per-catchment rain
decomposition cannot see what the geometry does not separate, which is
why ANTIOQUIA - 49% of national inflow energy - kept its headroom after
the catchment split helped CENTRO.

This replaces the SZH union with the real thing: locate the lev-12 unit
containing each dam, walk the NEXT_DOWN graph upstream, dissolve. The
result is the actual area whose rain reaches that reservoir.

    python scripts/sst/delineate_catchments.py --region ANTIOQUIA
    python scripts/sst/delineate_catchments.py --all --write

Inputs   ~/colombia_hydro/raw/hybas_sa_lev12_v1c.shp   (local, HydroSHEDS)
         outlet coordinates (OUTLETS below; OSM-derived, hand-verified)
Output   ~/colombia_hydro/data/xm_river_catchments_traced.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
HYBAS = PRIV / "raw" / "hybas_sa_lev12_v1c.shp"
REPO_DATA = Path(__file__).resolve().parents[2] / "colombia_hydro" / "data"
OUT_GJ = REPO_DATA / "xm_river_catchments_traced.geojson"
BBOX = (-79.5, -4.5, -66.5, 13.0)

# Dam / reservoir outlets keyed by the XM river name exactly as it appears
# in xm_river_catchments.geojson. Coordinates are the IMPOUNDMENT (dam
# wall or reservoir outlet), never the powerhouse: for the EPM cascade the
# powerhouse can sit 5-15 km downstream through a tunnel, and tracing from
# it would sweep in area the reservoir never sees.
#
# `area_hint` is a published/expected contributing area in km^2 where one
# is known, used only as a plausibility check on the trace - NOT as truth.
OUTLETS = {
  "ANTIOQUIA": {
    # --- verified: distinct impoundment, area consistent with published ---
    "GUATAPE":        dict(lon=-75.2085, lat=6.2563,  src="osm:resv Penol-Guatape"),
    # NOT the OSM "Embalse Riogrande II" centroid (-75.4930,6.5160): that
    # polygon's centre sits in a 357 km2 headwater arm. The dam wall is one
    # unit downstream, giving 1154 km2 - consistent with Riogrande II's
    # published ~1050 km2 contributing area.
    "GRANDE":         dict(lon=-75.4084, lat=6.5045,  src="osm:dam-wall unit, Riogrande II"),
    "PORCE2 CP":      dict(lon=-75.1486, lat=6.8053,  src="osm:dam Represa Porce II"),
    "PORCE III":      dict(lon=-75.1390, lat=6.9393,  src="osm:dam Represa Porce III"),
    "GUADALUPE":      dict(lon=-75.1893, lat=6.8393,  src="osm:plant Guadalupe IV"),
    "SAN CARLOS":     dict(lon=-74.8403, lat=6.2113,  src="osm:dam Represa Punchina"),
    "NARE CP":        dict(lon=-74.9386, lat=6.2939,  src="osm:dam Represa Playas"),
    "A. SAN LORENZO": dict(lon=-74.9975, lat=6.3804,  src="osm:dam San Lorenzo-Jaguas"),
    "ITUANGO":        dict(lon=-75.6615, lat=7.1352,  src="osm:dam Presa de Hidro Ituango"),
    "CARLOS LLERAS":  dict(lon=-75.2519, lat=6.5205,  src="osm:plant Carlos Lleras"),
    # --- HYPOTHESIS, not verified. "ESCUELA DE MINAS" is 11.9% of ANTIOQUIA
    #     energy and reads like a GAUGE name, not a river: most likely the
    #     Rio Medellin/Aburra station at the Universidad Nacional Facultad
    #     de Minas, i.e. upper Aburra valley inflow (525 km2). Enabled only
    #     under --hypo so the backtest can settle it against the SZH
    #     fallback rather than my assuming it. ---
    "ESCUELA DE MINAS": dict(lon=-75.5906, lat=6.2758,
                             src="HYPOTHESIS upper Aburra @ Fac.Minas", hypo=True),
    # --- tributary diversions into Troneras. 3.3% of ANTIOQUIA energy
    #     combined; OSM maps no separate intake and both fall in the SAME
    #     lev-12 unit as Guadalupe IV, so lev-12 cannot resolve them. They
    #     intentionally share GUADALUPE's shape. ---
    "TENCHE":         dict(lon=-75.2516, lat=6.7780,  src="osm:dam Troneras (shares GUADALUPE)"),
    "CONCEPCION":     dict(lon=-75.2516, lat=6.7780,  src="osm:dam Troneras (shares GUADALUPE)"),
    # QUEBRADONA dropped: carries no AporEner weight, and its diversion
    # reservoir falls inside GRANDE's traced catchment anyway.
  },
}

NESTED_OK = True


def load_basins(bbox=BBOX):
    g = gpd.read_file(HYBAS, bbox=bbox, engine="pyogrio")
    return g


def reverse_index(g):
    """NEXT_DOWN -> [HYBAS_ID, ...]"""
    rev = defaultdict(list)
    for hid, nd in zip(g["HYBAS_ID"].values, g["NEXT_DOWN"].values):
        if nd:
            rev[int(nd)].append(int(hid))
    return rev


def upstream_ids(start, rev, cap=200000):
    """Every basin draining into `start`, inclusive."""
    seen = {int(start)}
    q = deque([int(start)])
    while q:
        cur = q.popleft()
        for up in rev.get(cur, ()):
            if up not in seen:
                seen.add(up)
                q.append(up)
                if len(seen) > cap:
                    raise RuntimeError(f"runaway trace at {start}")
    return seen


def locate(g, sidx, lon, lat):
    """lev-12 unit containing the point (nearest centroid if none)."""
    pt = Point(lon, lat)
    hits = list(sidx.query(pt, predicate="contains"))
    if hits:
        return g.iloc[hits[0]]
    # point fell in a sliver gap - snap to nearest polygon
    d = g.geometry.distance(pt)
    return g.iloc[int(d.values.argmin())]


def trace_one(g, sidx, rev, lon, lat):
    unit = locate(g, sidx, lon, lat)
    ids = upstream_ids(unit["HYBAS_ID"], rev)
    sel = g[g["HYBAS_ID"].isin(ids)]
    geom = sel.geometry.union_all()
    return dict(
        n_units=len(sel),
        area_km2=float(sel["SUB_AREA"].sum()),
        up_area_km2=float(unit["UP_AREA"]),
        outlet_unit=int(unit["HYBAS_ID"]),
        geometry=geom,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ANTIOQUIA")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--hypo", action="store_true",
                    help="include outlets flagged hypo=True (unverified)")
    ap.add_argument("--full", action="store_true",
                    help="emit full upstream polygons instead of incremental")
    a = ap.parse_args(argv)

    print(f"loading {HYBAS.name} ...", flush=True)
    g = load_basins()
    g["_id"] = g["HYBAS_ID"].astype("int64")
    sidx = g.sindex
    rev = reverse_index(g)
    print(f"  {len(g)} lev-12 units, median {g['SUB_AREA'].median():.0f} km2")
    print()

    traced = {}
    for riv, o in OUTLETS.get(a.region, {}).items():
        if o.get("hypo") and not a.hypo:
            print(f"  {riv:16} skipped (unverified hypothesis; --hypo to include)")
            continue
        unit = locate(g, sidx, o["lon"], o["lat"])
        traced[riv] = dict(ids=upstream_ids(unit["HYBAS_ID"], rev),
                           outlet_unit=int(unit["HYBAS_ID"]),
                           **{k: v for k, v in o.items() if k != "hypo"})

    for riv, r in traced.items():
        r["inside"] = sorted(o for o, s in traced.items()
                             if o != riv and r["ids"] < s["ids"])
        r["contains"] = sorted(o for o, s in traced.items()
                               if o != riv and s["ids"] < r["ids"])
    for riv, r in traced.items():
        sub = set()
        for o in r["contains"]:
            sub |= traced[o]["ids"]
        r["inc_ids"] = r["ids"] - sub

    feats, seen = [], {}
    for riv, r in traced.items():
        use = r["ids"] if a.full else r["inc_ids"]
        if not use:
            print(f"  {riv:16} EMPTY incremental area "
                  f"(covered by {','.join(r['contains'])}) - skipped")
            continue
        sel = g[g["_id"].isin(use)]
        full_km2 = float(g[g["_id"].isin(r["ids"])]["SUB_AREA"].sum())
        inc_km2 = float(sel["SUB_AREA"].sum())
        sig = tuple(sorted(use))
        dup = seen.get(sig)
        seen.setdefault(sig, riv)
        note = f"   == {dup}" if dup else ""
        nest = f"   nested-in[{','.join(r['inside'])}]" if r["inside"] else ""
        print(f"  {riv:16} {len(sel):4}u  incr {inc_km2:8.1f}  "
              f"full {full_km2:9.1f} km2{nest}{note}")
        feats.append({
            "type": "Feature",
            "properties": dict(river=riv, region=a.region,
                               area_km2=round(inc_km2, 1),
                               full_area_km2=round(full_km2, 1),
                               n_units=len(sel), outlet_unit=r["outlet_unit"],
                               outlet_lon=r["lon"], outlet_lat=r["lat"],
                               nested_in=r["inside"], contains=r["contains"],
                               source=r["src"],
                               method="hydrobasins12_" +
                                      ("full" if a.full else "incremental")),
            "geometry": sel.geometry.union_all().__geo_interface__,
        })

    print()
    print(f"{len(feats)} catchments, {len(seen)} distinct shapes "
          f"(SZH unions gave 5)")
    if a.write and feats:
        OUT_GJ.write_text(json.dumps({"type": "FeatureCollection",
                                      "features": feats}))
        print(f"wrote -> {OUT_GJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
