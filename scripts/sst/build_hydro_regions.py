#!/usr/bin/env python3
"""Build polygons for XM's hydrological regions (Colombia's grid operator).

XM reports hydro inflows ("aportes hídricos") for six regions — ANTIOQUIA,
CALDAS, CARIBE, CENTRO, ORIENTE, VALLE — each an aggregate of contributing
river basins (API: servapibi.xm.com.co ListadoRios, saved to
raw/xm_listado_rios.json). Two polygon constructions, both emitted:

  ideam        official IDEAM Zonificación Hidrográfica subzonas (SZH), CC-BY
               from datos.gov.co (5kjg-nuda): each XM river is mapped to the
               subzona(s) holding its catchment and regions are the union.
               Whole-subzona granularity slightly overcounts where a dam sits
               mid-subzona — the price of using only official units.
  hydrobasins  HydroSHEDS HydroBASINS SA level-12 upstream traces from each
               region's outlet dam(s) via NEXT_DOWN topology — the sharper
               "actual contributing area" version. Antioquia's Cauca trace
               subtracts Valle's Salvajina catchment so regions stay disjoint.

Inputs (download once, see docstring commands in repo history):
    ~/colombia_hydro/raw/ideam_zonificacion.geojson
    ~/colombia_hydro/raw/hybas_sa_lev12_v1c.shp
Outputs:
    scripts/sst/colombia_hydro_regions.geojson          (ideam, drives the maps)
    scripts/sst/colombia_hydro_regions_hydrobasins.geojson
    ~/colombia_hydro/out/xm_regions_{ideam,hydrobasins}.shp.zip

    python scripts/sst/build_hydro_regions.py
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

RAW = Path.home() / "colombia_hydro" / "raw"
OUT = Path.home() / "colombia_hydro" / "out"
HERE = Path(__file__).resolve().parent

# XM river -> IDEAM subzona (cod_szh) curation. Comments carry the XM river
# names each subzona covers; kept alongside the region for provenance.
REGION_SZH = {
    "ANTIOQUIA": {
        2308: "NARE CP, GUATAPE, SAN CARLOS, A. SAN LORENZO, SAN MIGUEL",
        2307: "A. SAN LORENZO (Samaná Norte)",
        2701: "PORCE2 CP, PORCE III, GRANDE, GUADALUPE, TENCHE, CONCEPCION, "
              "ESCUELA DE MINAS, QUEBRADONA, CARLOS LLERAS",
        2702: "DESV. EEPPM (NEC,PAJ,DOL)",
        # ITUANGO: the FULL intermediate Cauca, Salvajina dam → Puerto Valdivia.
        # Everything in ZH26 except the upstream-of-Salvajina set (2601-2603,
        # 2627 → VALLE) and the below-dam set (2624-2626). Subzonas 2613/2614/
        # 2615 legitimately OVERLAP the CALDAS region (run-of-river plants
        # upstream, Ituango downstream): XM regions are not spatially disjoint.
        **{c: "ITUANGO (intermediate Cauca)" for c in
           (2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613,
            2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2628,
            2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637)},
    },
    "CALDAS": {
        2305: "MIEL I, DESV. MANSO", 2302: "DESV. GUARINO",
        2615: "CHINCHINA", 2613: "CAMPOALEGRE, SAN EUGENIO, ESTRELLA (Otún)",
        2614: "CAMPOALEGRE, SAN EUGENIO (Risaralda)",
    },
    "CARIBE": {1301: "SINU URRA"},
    "CENTRO": {
        2101: "EL QUIMBO, BETANIA CP", 2102: "BETANIA CP", 2103: "BETANIA CP",
        2104: "BETANIA CP", 2105: "BETANIA CP", 2106: "BETANIA CP",
        2108: "BETANIA CP (Yaguará)", 2109: "BETANIA CP",
        2116: "PRADO", 2120: "BOGOTA N.R., SAN FRANCISCO, DESV. SAN MARCOS",
        2204: "AMOYA", 2207: "CUCUANA",
        2401: "SOGAMOSO", 2402: "SOGAMOSO", 2403: "SOGAMOSO", 2405: "SOGAMOSO",
    },
    "ORIENTE": {
        3506: "GUAVIO", 3507: "BATA, DESV. CHIVOR, DESV. BATATAS",
        3503: "CHUZA, BLANCO (Chingaza)",
    },
    "VALLE": {
        2601: "CAUCA SALVAJINA", 2602: "CAUCA SALVAJINA (Palacé)",
        2603: "CAUCA SALVAJINA", 2627: "CAUCA SALVAJINA (Piendamó)",
        5310: "ALTOANCHICAYA, DIGUA",
        5407: "CALIMA",
    },
}

# HydroBASINS outlets: (lat, lon) of the most-downstream dam per system.
# subtract= removes an upstream region's catchment so regions stay disjoint.
REGION_OUTLETS = {
    "ANTIOQUIA": [dict(name="Ituango (Cauca)", lat=7.137, lon=-75.664,
                       subtract="VALLE:Salvajina"),
                  dict(name="Playas (Nare)", lat=6.365, lon=-74.862),
                  dict(name="Porce III", lat=6.939, lon=-75.140)],
    "CALDAS": [dict(name="Miel I", lat=5.561, lon=-74.887),
               dict(name="Guarinó desv.", lat=5.24, lon=-75.05)],
    "CARIBE": [dict(name="Urrá I", lat=8.014, lon=-76.189)],
    "CENTRO": [dict(name="Betania", lat=2.690, lon=-75.436),
               dict(name="Sogamoso", lat=7.100, lon=-73.406),
               dict(name="Prado", lat=3.75, lon=-74.85),
               dict(name="Amoyá", lat=3.725, lon=-75.472),
               dict(name="La Guaca (Bogotá)", lat=4.556, lon=-74.336)],
    "ORIENTE": [dict(name="Guavio", lat=4.725, lon=-73.483),
                dict(name="Chivor (Batá)", lat=4.901, lon=-73.297),
                dict(name="Chuza (Chingaza)", lat=4.57, lon=-73.72)],
    "VALLE": [dict(name="Salvajina", lat=2.950, lon=-76.685),
              dict(name="Alto Anchicayá", lat=3.553, lon=-76.884),
              dict(name="Calima", lat=3.932, lon=-76.525)],
}

SIMPLIFY_DEG = 0.004          # ~400 m — map-scale fidelity, small files


def build_ideam(gdfz):
    import geopandas as gpd
    from shapely.ops import unary_union
    feats = []
    for region, szhs in REGION_SZH.items():
        sel = gdfz[gdfz["cod_szh"].isin(szhs)]
        missing = set(szhs) - set(sel["cod_szh"])
        if missing:
            raise SystemExit(f"{region}: SZH codes not found: {missing}")
        geom = unary_union(sel.geometry.values).simplify(SIMPLIFY_DEG)
        feats.append(dict(region=region, source="IDEAM SZH",
                          szh=",".join(str(int(c)) for c in sorted(szhs)),
                          rivers="; ".join(sorted(set(szhs.values()))),
                          geometry=geom))
        print(f"  ideam {region}: {len(sel)} subzonas, "
              f"{geom.area * (111.1**2) * np.cos(np.radians(5)):,.0f} km² approx")
    return gpd.GeoDataFrame(feats, crs="EPSG:4326")


def build_hydrobasins():
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union
    print("loading HydroBASINS lev12 …", flush=True)
    hb = gpd.read_file(RAW / "hybas_sa_lev12_v1c.shp",
                       bbox=(-78.5, -1.0, -71.0, 12.0),
                       columns=["HYBAS_ID", "NEXT_DOWN"])
    up = {}
    for hid, nd in zip(hb["HYBAS_ID"].values, hb["NEXT_DOWN"].values):
        up.setdefault(nd, []).append(hid)
    geom_by_id = dict(zip(hb["HYBAS_ID"].values, hb.geometry.values))
    sidx = hb.sindex

    def catchment(lat, lon):
        hit = list(sidx.query(Point(lon, lat), predicate="within"))
        if not hit:
            raise SystemExit(f"no hybas polygon at {lat},{lon}")
        seed = hb.iloc[hit[0]]["HYBAS_ID"]
        seen, stack = {seed}, [seed]
        while stack:
            for u in up.get(stack.pop(), []):
                if u not in seen:
                    seen.add(u); stack.append(u)
        return seen

    catch_cache = {}
    for region, outs in REGION_OUTLETS.items():
        for o in outs:
            catch_cache[f"{region}:{o['name'].split(' ')[0]}"] = \
                catchment(o["lat"], o["lon"])
    feats = []
    for region, outs in REGION_OUTLETS.items():
        ids = set()
        for o in outs:
            c = catchment(o["lat"], o["lon"])
            if "subtract" in o:
                c = c - catch_cache[o["subtract"]]
            ids |= c
        geom = unary_union([geom_by_id[i] for i in ids]).simplify(SIMPLIFY_DEG)
        feats.append(dict(region=region, source="HydroBASINS lev12 trace",
                          outlets="; ".join(o["name"] for o in outs),
                          geometry=geom))
        print(f"  hybas {region}: {len(ids)} lev12 cells", flush=True)
    return gpd.GeoDataFrame(feats, crs="EPSG:4326")


def write_outputs(gdf, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    gj = HERE / ("colombia_hydro_regions.geojson" if tag == "ideam"
                 else f"colombia_hydro_regions_{tag}.geojson")
    gdf.rename(columns={"region": "name"}).to_file(gj, driver="GeoJSON")
    shp_dir = OUT / f"xm_regions_{tag}"
    shp_dir.mkdir(exist_ok=True)
    gdf.to_file(shp_dir / f"xm_regions_{tag}.shp")
    zp = OUT / f"xm_regions_{tag}.shp.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    print(f"  wrote {gj.name} ({gj.stat().st_size/1e6:.1f} MB) + {zp}")


def main():
    import geopandas as gpd
    print("loading IDEAM zonificación …", flush=True)
    gdfz = gpd.read_file(RAW / "ideam_zonificacion.geojson")
    gdfz["cod_szh"] = gdfz["cod_szh"].astype(float).astype(int)
    write_outputs(build_ideam(gdfz), "ideam")
    write_outputs(build_hydrobasins(), "hydrobasins")


if __name__ == "__main__":
    main()
