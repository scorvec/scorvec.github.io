#!/usr/bin/env python3
"""Per-river catchment polygons — one feature per active XM river.

Each river's catchment is the union of its IDEAM subzonas (the same
river→SZH provenance in build_hydro_regions.REGION_SZH that drives the
region polygons and the per-river validation). Unlike the region products,
features here legitimately OVERLAP: rivers on shared subzonas (lumped
gauges) and nested drainage (Ituango's mid-Cauca contains the Caldas
run-of-river catchments) keep their full physical footprint.

Attributes: river, code (XM ListadoRios — joins directly against the
per-river AporEner/AporCaudal API), region, szh, area_km2.

Outputs:
    ~/colombia_hydro/out/xm_river_catchments.geojson
    ~/colombia_hydro/out/xm_river_catchments.shp.zip

    python scripts/sst/build_river_catchments.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hydro_regions import RAW, OUT, SIMPLIFY_DEG
from river_corr import river_szh
from validate_region_rain import ORDER

RIVERS_JSON = RAW / "xm_listado_rios.json"


def main() -> int:
    import geopandas as gpd
    from shapely.ops import unary_union
    rios = json.load(open(RIVERS_JSON))
    rivers = []
    for it in rios["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("Status") == "ACTIVO" and v["HydroRegion"] in ORDER:
                rivers.append(dict(river=v["Name"].strip().upper(),
                                   code=v["Code"], region=v["HydroRegion"]))

    print("loading IDEAM zonificación …", flush=True)
    gdfz = gpd.read_file(RAW / "ideam_zonificacion.geojson")
    gdfz["cod_szh"] = gdfz["cod_szh"].astype(float).astype(int)

    feats = []
    for rv in rivers:
        szhs = river_szh(rv["river"], rv["region"])
        if not szhs:
            print(f"  !! {rv['river']}: no subzona match — skipped")
            continue
        g = unary_union(gdfz[gdfz["cod_szh"].isin(szhs)].geometry.values)
        g = g.simplify(SIMPLIFY_DEG).buffer(0)
        feats.append(dict(river=rv["river"], code=rv["code"], region=rv["region"],
                          szh=",".join(str(s) for s in sorted(szhs)), geometry=g))
    out = gpd.GeoDataFrame(feats, crs="EPSG:4326")
    ea = out.to_crs("+proj=cea")
    out["area_km2"] = (ea.geometry.area / 1e6).round(0)
    for _, r in out.sort_values(["region", "river"]).iterrows():
        print(f"  {r['region']:<10} {r['river']:<28} {r['area_km2']:>9,.0f} km²  SZH {r['szh']}")

    OUT.mkdir(parents=True, exist_ok=True)
    gj = OUT / "xm_river_catchments.geojson"
    out.to_file(gj, driver="GeoJSON")
    shp_dir = OUT / "xm_river_catchments"
    shp_dir.mkdir(exist_ok=True)
    out.to_file(shp_dir / "xm_river_catchments.shp")
    zp = OUT / "xm_river_catchments.shp.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    print(f"wrote {gj.name} ({gj.stat().st_size/1e6:.1f} MB) + {zp.name} "
          f"({len(out)} rivers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
