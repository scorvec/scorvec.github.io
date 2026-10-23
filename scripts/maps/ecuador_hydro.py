#!/usr/bin/env python3
"""Ecuador's hydropower: the plants, their dams and intakes, and the catchment each one draws on.

Catchments: HydroBASINS level 12 (HydroSHEDS, CC BY 4.0), traced upstream through NEXT_DOWN from the sub-basin that holds
each dam or intake (not the powerhouse: run-of-river schemes take their water well upstream). Rivers: HydroRIVERS
(ORD_FLOW). HydroSHEDS' download site refuses scripted requests, so both layers come from the Earth Engine catalogue
copies (WWF/HydroSHEDS/v1/Basins/hybas_12; WWF/HydroSHEDS/v1/FreeFlowingRivers with RIV_ORD as ORD_FLOW) exported to
GeoJSON for the box. Plants and dams: one OpenStreetMap Overpass query, saved as `--osm`:

    [out:json][timeout:120];area["ISO3166-1"="EC"][admin_level=2]->.ec;
    (nwr["power"="plant"]["plant:source"="hydro"](area.ec);nwr["waterway"="dam"]["name"](area.ec););out center tags;

The major schemes are listed in SCHEMES with the dam or intake their water comes through; plants sharing a reservoir or
a tailrace (the Paute cascade, Agoyan and San Francisco) share a catchment.

    python scripts/maps/ecuador_hydro.py --basins basins.geojson --rivers rivers.geojson --osm osm.json \
        --out ecuador_hydro.png [--geojson catchments.geojson]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import pyproj  # noqa: F401  (load order: before the GDAL/PROJ stack)
import geopandas as gpd
from shapely.geometry import Point

BBOX = (-81.3, -5.2, -75.1, 1.6)  # lon0, lat0, lon1, lat1
# scheme: (intake / dam lon, lat, the point's name, plants fed with MW, river, approximate?)
SCHEMES = {
    "Coca Codo Sinclair": (-77.686, -0.197, "Salado headworks", [("Coca Codo Sinclair", 1500)], "Coca", False),
    "Paute cascade": (
        -78.566,
        -2.594,
        "Daniel Palacios dam (Amaluza)",
        [("Mazar", 170), ("Molino", 1100), ("Sopladora", 487)],
        "Paute",
        False,
    ),
    "Agoyan - San Francisco": (
        -78.382,
        -1.398,
        "Agoyan dam",
        [("Agoyan", 156), ("San Francisco", 230)],
        "Pastaza",
        False,
    ),
    "Toachi-Pilaton": (
        -78.916,
        -0.368,
        "Toachi-Pilaton headworks",
        [("Toachi-Pilaton", 254)],
        "Toachi / Pilaton",
        False,
    ),
    "Minas San Francisco": (
        -79.599,
        -3.310,
        "powerhouse (intake not mapped)",
        [("Minas San Francisco", 270)],
        "Jubones",
        True,
    ),
    "Marcel Laniado": (-79.754, -0.926, "Daule-Peripa dam", [("Marcel Laniado de Wind", 213)], "Daule", False),
    "Delsitanisagua": (-79.017, -3.981, "Delsitanisagua dam", [("Delsitanisagua", 180)], "Zamora", False),
    "Pucara": (-78.397, -1.074, "Pisayambo reservoir", [("Pucara", 73)], "Pisayambo", False),
    "Manduriacu": (-78.912, 0.215, "Manduriacu dam", [("Manduriacu", 65)], "Guayllabamba", False),
}
LABEL_AT = {  # label centres clear of the catchments; a leader runs to the dam or intake
    "Coca Codo Sinclair": (-76.85, -0.55),
    "Paute cascade": (-77.7, -2.95),
    "Agoyan - San Francisco": (-77.35, -1.6),
    "Toachi-Pilaton": (-79.45, 0.2),
    "Minas San Francisco": (-80.5, -3.25),
    "Marcel Laniado": (-79.55, -0.55),
    "Delsitanisagua": (-78.2, -4.2),
    "Pucara": (-77.55, -1.05),
    "Manduriacu": (-78.75, 0.8),
}
SNAP_DEG = 0.055  # ~6 km
CLUSTER_DEG = 0.15  # other plants this close share one label
LABEL_MW = 25  # label the other plants (or clusters) at least this large
INNER = {"Paute cascade": (-78.622, -2.595, "Mazar dam")}  # a dam inside a scheme's catchment: drawn as a point


def upstream(basins: gpd.GeoDataFrame, hid: int) -> list[int]:
    up = defaultdict(list)
    for h, nd in zip(basins.HYBAS_ID, basins.NEXT_DOWN):
        up[nd].append(h)
    out, todo = [], [hid]
    while todo:
        h = todo.pop()
        out.append(h)
        todo.extend(up.get(h, []))
    return out


def _plain(s: str) -> str:
    """Lower case without accents: 'Central Hidroeléctrica Pucará' -> 'central hidroelectrica pucara'."""
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def _short(name: str) -> str:
    """'Central Hidroeléctrica Nayón' -> 'Nayón', 'ESTACIÓN HIDROABANICO' -> 'Abanico'."""
    s = re.sub(r"^(la\s+)?(central\s+|estaci[oó]n\s+)?(hidroel[eé]ctric[ao]s?\s*)?", "", name, flags=re.I).strip(' "')
    s = re.sub(r"^hidro(?=\w)", "", s, flags=re.I)
    s = s.title() if s.isupper() else s
    return (s[:1].upper() + s[1:])[:26]


def clusters(rows: list[dict]) -> list[list[dict]]:
    """Group plants within CLUSTER_DEG of a larger one, largest first."""
    groups: list[list[dict]] = []
    for q in sorted(rows, key=lambda q: -q["mw"]):
        g = next(
            (
                g
                for g in groups
                if abs(g[0]["lon"] - q["lon"]) < CLUSTER_DEG and abs(g[0]["lat"] - q["lat"]) < CLUSTER_DEG
            ),
            None,
        )
        if g:
            g.append(q)
        else:
            groups.append([q])
    return groups


def osm_plants(path: str):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for e in d.get("elements", []):
        t = e.get("tags", {})
        if t.get("power") != "plant":
            continue
        c = e.get("center") or e
        v = (t.get("plant:output:electricity") or "").replace(",", ".")
        m = re.match(r"\s*([\d.]+)\s*(MW|kW|GW)?", v)
        mw = None
        if m:
            x = float(m.group(1))
            mw = x / 1000 if m.group(2) == "kW" else x * 1000 if m.group(2) == "GW" else x
        rows.append({"name": t.get("name", ""), "mw": mw, "lon": c["lon"], "lat": c["lat"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--basins", required=True)
    ap.add_argument("--rivers", required=True)
    ap.add_argument("--osm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geojson")
    a = ap.parse_args()

    basins = gpd.read_file(a.basins, bbox=BBOX)[["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA", "geometry"]]
    print(f"{len(basins)} level-12 sub-basins in the box")
    sindex = basins.sindex
    catch = []
    for name, (lon, lat, where, plants, river, approx) in SCHEMES.items():
        if approx:  # a powerhouse beside its river: the main-stem sub-basin nearby (largest upstream area within ~6 km)
            hit = list(sindex.query(Point(lon, lat).buffer(SNAP_DEG), predicate="intersects"))
            hit = [max(hit, key=lambda i: basins.iloc[i].UP_AREA)] if hit else []
        else:
            hit = list(sindex.query(Point(lon, lat), predicate="within"))
        if not hit:
            raise SystemExit(f"{name}: the point is in no sub-basin")
        ids = set(upstream(basins, int(basins.iloc[hit[0]].HYBAS_ID)))
        sel = basins[basins.HYBAS_ID.isin(ids)]
        area = float(sel.SUB_AREA.sum())
        mw = sum(p[1] for p in plants)
        catch.append(
            {
                "scheme": name,
                "river": river,
                "mw": mw,
                "area_km2": round(area),
                "approx": approx,
                "point": where,
                "geometry": sel.union_all(),
            }
        )
        print(f"  {name}: {len(sel)} sub-basins, {area:,.0f} km2, {mw} MW ({river})")
    cg = gpd.GeoDataFrame(catch, crs=basins.crs).sort_values("area_km2", ascending=False)
    if a.geojson:
        cg.to_file(a.geojson, driver="GeoJSON")

    rivers = gpd.read_file(a.rivers, bbox=BBOX)
    rivers = rivers[rivers["ORD_FLOW"] <= 6]
    plants = osm_plants(a.osm)

    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 12.2))
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.82], projection=ccrs.PlateCarree())
    ax.set_extent((BBOX[0], BBOX[2], BBOX[1], BBOX[3]), ccrs.PlateCarree())
    ax.add_feature(cf.OCEAN.with_scale("10m"), facecolor="#dfe9f2", zorder=0)
    ax.add_feature(cf.LAND.with_scale("10m"), facecolor="#f6f4ef", zorder=0)
    ax.add_feature(cf.BORDERS.with_scale("10m"), edgecolor="0.35", lw=0.9, zorder=5)
    ax.add_feature(cf.COASTLINE.with_scale("10m"), edgecolor="0.35", lw=0.7, zorder=5)
    cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4", "#b15928"]
    for (_, r), col in zip(cg.iterrows(), cols):
        ax.add_geometries(
            [r.geometry],
            ccrs.PlateCarree(),
            facecolor=col,
            alpha=0.38,
            edgecolor=col,
            lw=1.2,
            zorder=2,
            hatch="//" if r.approx else None,
        )
        lx, ly = LABEL_AT[r.scheme]
        dam = SCHEMES[r.scheme]
        ax.plot([lx, dam[0]], [ly, dam[1]], color="0.35", lw=0.6, zorder=6, transform=ccrs.PlateCarree())
        lab = f"{r.scheme}\n{r.mw:,} MW · {r.area_km2:,} km²" + (" (approx.)" if r.approx else "")
        ax.text(
            lx,
            ly,
            lab,
            fontsize=8,
            ha="center",
            va="center",
            color="#222",
            zorder=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": col, "alpha": 0.85, "lw": 0.8},
        )
    for o, lw in ((6, 0.35), (5, 0.6), (4, 0.9), (3, 1.3), (2, 1.8), (1, 2.3)):
        sub = rivers[rivers["ORD_FLOW"] == o]
        if len(sub):
            ax.add_geometries(sub.geometry, ccrs.PlateCarree(), facecolor="none", edgecolor="#3b78b0", lw=lw, zorder=3)
    for lon, lat in [v[:2] for v in SCHEMES.values()] + [v[:2] for v in INNER.values()]:
        ax.plot(lon, lat, marker="^", ms=8, mfc="#222", mec="white", mew=0.8, zorder=9, transform=ccrs.PlateCarree())
    # one circle per major scheme (its plants' total, at the dam or intake); the other mapped plants on their own
    covered = {_plain(n) for *_x, ps, _r, _a in SCHEMES.values() for n, _mw in ps}
    others = [
        q
        for q in plants
        if q["mw"] and not any(c in _plain(q["name"]) for c in covered) and "casa de maquinas" not in _plain(q["name"])
    ]
    big = max(sum(mw for _n, mw in v[3]) for v in SCHEMES.values())
    size = lambda mw: 12 + 900 * (mw / big) ** 0.75  # noqa: E731
    for lon, lat, _w, ps, _r, _a in SCHEMES.values():
        ax.scatter(
            lon,
            lat,
            s=size(sum(mw for _n, mw in ps)),
            color="#c0392b",
            alpha=0.5,
            edgecolor="#7b241c",
            lw=0.7,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
    for q in others:
        ax.scatter(
            q["lon"],
            q["lat"],
            s=size(q["mw"]),
            color="#c0392b",
            alpha=0.5,
            edgecolor="#7b241c",
            lw=0.6,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
    for g in clusters(others):
        if sum(q["mw"] for q in g) < LABEL_MW:
            continue
        ax.text(
            max(q["lon"] for q in g) + 0.07,
            g[0]["lat"],
            "\n".join(f"{_short(q['name'])} {q['mw']:,.0f} MW" for q in g),
            fontsize=7,
            color="#7b241c",
            va="center",
            zorder=8,
            transform=ccrs.PlateCarree(),
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="0.6")
    gl.top_labels = gl.right_labels = False
    for mw, lab in ((50, "50 MW"), (500, "500 MW"), (1500, "1,500 MW")):
        ax.scatter([], [], s=size(mw), color="#c0392b", alpha=0.5, edgecolor="#7b241c", label=lab)
    ax.plot([], [], "^", color="#222", ms=8, label="dam / intake the catchment is traced from")
    ax.legend(
        loc="lower left",
        fontsize=8.5,
        framealpha=0.9,
        title="Installed capacity (a scheme at its dam or intake)",
        title_fontsize=8.5,
    )
    tot = int(cg.mw.sum())
    fig.text(
        0.03,
        0.965,
        "Ecuador's hydropower: plants, dams and the catchments that feed them",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.945,
        f"Shaded: the catchment upstream of each major scheme's dam or intake ({tot:,} MW of capacity); "
        "circles: installed capacity",
        fontsize=9.5,
        color="#444",
    )
    fig.text(
        0.03,
        0.1,
        "Catchments traced upstream through HydroBASINS level-12 sub-basins (HydroSHEDS; Lehner & Grill 2013, "
        "CC BY 4.0); rivers: HydroSHEDS free-flowing rivers; plants and dams: © OpenStreetMap contributors (ODbL); the major schemes' "
        "capacities are the plants' published ratings.\nThe Paute cascade "
        "(Mazar, Molino, Sopladora) shares the catchment above the Daniel Palacios dam; San Francisco takes Agoyan's "
        "outflow. Minas San Francisco is traced from its powerhouse (its Jubones intake is not mapped), so its area is "
        "slightly too large.",
        fontsize=8,
        color="#6f6b64",
        va="top",
        wrap=True,
    )
    fig.savefig(a.out, dpi=120, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
