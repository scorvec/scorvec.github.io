#!/usr/bin/env python3
"""Render the two colombia_hydro maps: contributing basins + XM display view.

Colors follow XM's Sinergox dashboard palette; the department clusters in the
display view replicate XM's official "Mapa Hidrología SIN" exactly (decoded
2026-07-23 from sinergox.xm.com.co/hdrlg/Paginas/Informes/MapaHidrologiaSIN.aspx).
Note XM's display is deliberately coarse: Cauca, Huila, Tolima and Risaralda
are unshaded despite hosting plants (Salvajina, Betania/El Quimbo, Prado/Amoyá,
Campoalegre) — the black basin outlines carry the truth.

    python scripts/sst/render_hydro_maps.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from shapely.geometry import shape
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTDIR = REPO / "colombia_hydro"

# XM Sinergox region palette (sampled from the official dashboard cards)
COLORS = {"ANTIOQUIA": "#68B79F", "CALDAS": "#4F5BE3", "CARIBE": "#F0A169",
          "CENTRO": "#F5D76E", "ORIENTE": "#C0608D", "VALLE": "#43128F"}

# XM's official department aggregates (Mapa Hidrología SIN)
XM_DEPTS = {
    "ANTIOQUIA": ["Antioquia"],
    "CALDAS": ["Caldas"],
    "CARIBE": ["La Guajira", "Cesar", "Magdalena", "Atlántico", "Bolívar",
               "Sucre", "Córdoba"],
    "CENTRO": ["Cundinamarca", "Bogota", "Meta"],
    "ORIENTE": ["Santander", "Norte de Santander", "Boyacá", "Casanare", "Arauca"],
    "VALLE": ["Valle del Cauca"],
}
RIVERS = {
 "ANTIOQUIA": "Nare · Guatapé · San Carlos\nPorce II/III · Grande · Guadalupe\nItuango (mid-Cauca, net of Caldas)",
 "CALDAS": "Miel I · Guarinó · Manso\nChinchiná · Campoalegre",
 "CARIBE": "Sinú (Urrá)",
 "CENTRO": "Betania · El Quimbo · Bogotá N.R.\nPrado · Amoyá · Sogamoso",
 "ORIENTE": "Guavio · Batá (Chivor)\nChuza · Blanco",
 "VALLE": "Salvajina · Alto Anchicayá\nDigua · Calima"}
LABEL = {"ANTIOQUIA": (-78.55, 7.9), "CALDAS": (-78.2, 4.75), "CARIBE": (-74.55, 8.95),
         "CENTRO": (-71.55, 5.6), "ORIENTE": (-71.85, 3.35), "VALLE": (-78.4, 1.9)}
ARROW = {"ANTIOQUIA": (-75.7, 7.0), "CALDAS": (-75.35, 5.35), "CARIBE": (-76.0, 7.9),
         "CENTRO": (-73.3, 6.2), "ORIENTE": (-73.4, 4.55), "VALLE": (-76.75, 2.85)}


MAGDALENA_COLOR = "#8a5a00"        # burnt umber — distinct from every region hue


def _magdalena(ax):
    """Overlay the Magdalena–Cauca macro-basin divide (IDEAM AH boundary)."""
    mg = gpd.read_file(HERE / "magdalena_outline.geojson")
    ax.add_geometries([mg.geometry.iloc[0]], ccrs.PlateCarree(),
                      facecolor="none", edgecolor=MAGDALENA_COLOR,
                      linewidth=1.6, linestyle=(0, (6, 3)), zorder=4, alpha=0.9)


def basins_map(regions):
    fig = plt.figure(figsize=(9.8, 11.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-79.8, -70.4, 0.3, 10.3])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f3ee")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e8f0f5")
    ax.coastlines("50m", lw=0.7, color="0.4")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6, color="0.55")
    ax.add_feature(cfeature.RIVERS.with_scale("50m"), lw=0.4, alpha=0.5)
    for name in ["ANTIOQUIA", "CENTRO", "ORIENTE", "VALLE", "CARIBE", "CALDAS"]:
        rr = regions[regions["name"] == name].iloc[0]
        ax.add_geometries([rr.geometry], ccrs.PlateCarree(), facecolor=COLORS[name],
                          alpha=0.6, edgecolor="k", linewidth=0.7, zorder=3)
    _magdalena(ax)
    plants = json.load(open(HERE / "colombia_hydro_plants.json"))["plants"]
    ax.scatter([p["lon"] for p in plants], [p["lat"] for p in plants],
               s=[18 + p["mw"] / 45 for p in plants], c="k", marker="^", zorder=6,
               label="hydro plant (size ∝ MW)")
    ax.plot([], [], color=MAGDALENA_COLOR, lw=1.6, ls=(0, (6, 3)),
            label="Magdalena–Cauca basin divide")
    for name, (lx, ly) in LABEL.items():
        axx, ayy = ARROW[name]
        ax.annotate(f"{name}\n{RIVERS[name]}", xy=(axx, ayy), xytext=(lx, ly),
                    fontsize=8.2, fontweight="bold", ha="center", va="center",
                    color=COLORS[name], zorder=7,
                    path_effects=[mpe.withStroke(linewidth=2.6, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=1.1, alpha=0.8))
    gl = ax.gridlines(draw_labels=True, lw=0.2, ls=":", color="0.6")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title("XM hydrological regions — Colombia's power-system basins\n"
                 "membership: XM ListadoRios · polygons: IDEAM Zonificación Hidrográfica "
                 "(disjoint — upstream regions keep shared basins)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.savefig(OUTDIR / "xm_regions_map.webp", dpi=125, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def departments_map(regions):
    rd = shpreader.Reader(shpreader.natural_earth(resolution="10m",
         category="cultural", name="admin_1_states_provinces"))
    depts = [(r.attributes["name"], shape(r.geometry)) for r in rd.records()
             if r.attributes["admin"] == "Colombia"]
    d2r = {d: reg for reg, ds in XM_DEPTS.items() for d in ds}
    fig = plt.figure(figsize=(8.6, 10.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-79.6, -66.6, -4.5, 12.8])
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e8f0f5")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f3ee")
    matched = set()
    for dname, dgeom in depts:
        reg = d2r.get(dname)
        if reg:
            matched.add(dname)
        ax.add_geometries([dgeom], ccrs.PlateCarree(),
                          facecolor=COLORS[reg] if reg else "#eceae4",
                          alpha=0.85 if reg else 1.0,
                          edgecolor="white", linewidth=0.7, zorder=3)
    missing = set(d2r) - matched
    if missing:
        print(f"  WARNING: departments not found in Natural Earth: {missing}")
    for _, rr in regions.iterrows():
        ax.add_geometries([rr.geometry], ccrs.PlateCarree(), facecolor="none",
                          edgecolor="k", linewidth=0.9, zorder=5)
    _magdalena(ax)
    ax.coastlines("50m", lw=0.6, color="0.4")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6, color="0.55")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85) for c in COLORS.values()]
    labels = [k.title() for k in COLORS]
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], color=MAGDALENA_COLOR, lw=1.6, ls=(0, (6, 3))))
    labels.append("Magdalena–Cauca divide")
    ax.legend(handles, labels, loc="lower left", fontsize=8.5,
              title="Region (XM department groups)", title_fontsize=8.5, framealpha=0.95)
    ax.set_title("XM hydro regions — official department-scale display\n"
                 "replicates XM's Mapa Hidrología SIN · black outlines = actual contributing basins",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(OUTDIR / "xm_regions_departments.webp", dpi=125,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def topo_map(regions):
    """Basins over shaded terrain (NOAA DEM mosaic export, ~30 arc-sec)."""
    import numpy as np
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    from PIL import Image as PILImage
    tif = PILImage.open(Path.home() / "colombia_hydro" / "raw" / "etopo_colombia.tif")
    z = np.array(tif, dtype="float32")
    ext = (-80.5, -69.5, -5.5, 13.5)                    # matches the export bbox
    ls = LightSource(azdeg=315, altdeg=40)
    # muted terrain so the region overlays stay the loudest layer
    terrain = LinearSegmentedColormap.from_list("land", [
        (0.00, "#9db98e"), (0.12, "#b5c69c"), (0.30, "#ddd6b2"),
        (0.55, "#cbb59a"), (0.78, "#bfb2a8"), (1.00, "#f5f3ef")])
    zl = np.clip(z, 0, None)
    rgb = ls.shade(zl, cmap=terrain, blend_mode="soft",
                   vmin=0, vmax=4800, vert_exag=0.12, dx=900, dy=900)
    rgb[z <= 0] = (0.78, 0.86, 0.92, 1.0)               # sea
    fig = plt.figure(figsize=(9.8, 11.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-79.8, -70.4, 0.3, 10.3])
    ax.imshow(rgb, origin="upper", extent=ext, transform=ccrs.PlateCarree(),
              interpolation="bilinear", zorder=1)
    ax.coastlines("50m", lw=0.7, color="0.25")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6, color="0.3")
    for name in ["ANTIOQUIA", "CENTRO", "ORIENTE", "VALLE", "CARIBE", "CALDAS"]:
        rr = regions[regions["name"] == name].iloc[0]
        ax.add_geometries([rr.geometry], ccrs.PlateCarree(),
                          facecolor=COLORS[name], alpha=0.5,
                          edgecolor="k", linewidth=1.3, zorder=3)
    _magdalena(ax)
    plants = json.load(open(HERE / "colombia_hydro_plants.json"))["plants"]
    ax.scatter([p["lon"] for p in plants], [p["lat"] for p in plants],
               s=[18 + p["mw"] / 45 for p in plants], c="k", marker="^", zorder=6)
    for name, (lx, ly) in LABEL.items():
        axx, ayy = ARROW[name]
        ax.annotate(name, xy=(axx, ayy), xytext=(lx, ly),
                    fontsize=9.5, fontweight="bold", ha="center", va="center",
                    color=COLORS[name], zorder=7,
                    path_effects=[mpe.withStroke(linewidth=3.0, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=1.2))
    gl = ax.gridlines(draw_labels=True, lw=0.2, ls=":", color="0.5")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title("XM hydrological regions over terrain\n"
                 "the cordilleras carve the basins — dams sit where rivers drop off the ranges",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(OUTDIR / "xm_regions_topo.webp", dpi=125, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    regions = gpd.read_file(HERE / "colombia_hydro_regions.geojson")
    basins_map(regions)
    departments_map(regions)
    topo_map(regions)
    # re-stamp the page's cache-busters so browsers refetch the new renders
    import re
    from datetime import datetime, timezone
    page = OUTDIR / "index.html"
    cb = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    page.write_text(re.sub(r"(xm_regions_[a-z]+\.webp)\?v=\w+", rf"\1?v={cb}",
                           page.read_text()))
    print("wrote xm_regions_{map,departments,topo}.webp (+ page cache-bust)")


if __name__ == "__main__":
    main()
