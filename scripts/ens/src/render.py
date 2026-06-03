#!/usr/bin/env python3
"""Render ensemble-anomaly map frames (Day 0–15 lead slider) for a (variable,
product). Each frame is an N-panel map (one panel per ensemble), so a single lead
slider drives all models at once. Reuses the sst_anim.html viewer via a
per-(var,product) manifest with a build "ver" cache-buster.

Per-variable map: z500 → full NH on a NorthPolarStereo (cyclic-closed at 0/360);
t2m → North America on a LambertConformal.

Products: anom30 / anom10 (vs ERA5 norms) — Phase 1–3.  dprev / d48 land in Phase 4.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).parent))
from common import REF, grid, GRIDS, LEADS, VARS, ENS_LABEL

# GrADS/StormVista-style diverging ramp: purple-blue (negative) → white → orange-red-
# magenta (positive). Saturated mid/outer bands to match StormVista's punch; only the
# innermost ±2 band stays white. Sampled onto each product's discrete level set below.
_RAMP = mcolors.LinearSegmentedColormap.from_list("sv_anom", [
    "#5e0a99", "#7d1fc4", "#2e2ee0", "#0d4fe6", "#1f7ff0", "#3fa0f5", "#79c2f7", "#b6dffb",
    "#ffffff", "#ffffff",
    "#ffe89a", "#ffcf45", "#ffab14", "#ff8000", "#ff4d00", "#e62200", "#bd0000", "#85002e"])

# Symmetric discrete level edges per (kind, var) — these set the colorbar bands.
_LEVELS = {
    ("anom", "z500"): [-32, -28, -24, -20, -16, -12, -8, -4, -2, 2, 4, 8, 12, 16, 20, 24, 28, 32],
    ("chg",  "z500"): [-12, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10, 12],
    ("chg",  "t2m"):  [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8],
}

# Median-height contour overlay (z500 only), every 6 dam.
_GPH_CONTOURS = np.arange(480, 601, 6)


def _disc(levels):
    """Discrete (cmap, norm) sampling the StormVista ramp across `levels` bands."""
    n = len(levels) - 1
    cmap = mcolors.ListedColormap([_RAMP(i / (n - 1)) for i in range(n)])
    cmap.set_under(_RAMP(0.0)); cmap.set_over(_RAMP(1.0))
    return cmap, mcolors.BoundaryNorm(levels, cmap.N)


PRODUCTS = {  # anomaly products: product → (clim period, cbar label)
    "anom30": ("30yr", "anomaly vs 1991–2020"),
}
CHANGE = {    # change products: product → (hours_back, cbar label)
    "dprev": (24, "change vs previous run"),
    "d48":   (48, "48-hour trend"),
}

# A circular axes boundary for the polar-stereo panels (drawn once, reused per panel).
_THETA = np.linspace(0, 2 * np.pi, 200)
_CIRCLE = mpath.Path(np.vstack([np.sin(_THETA), np.cos(_THETA)]).T * 0.5 + 0.5)


def _plot_cfg(var: str) -> dict:
    """Per-variable plotting config: projection, target grid, and (NA only) the
    plotting longitudes shifted to -180..180."""
    lat, lon = grid(var)
    region = GRIDS[var]["region"]
    if region == "nh":
        return dict(region="nh", lat=lat, lon=lon,
                    proj=ccrs.NorthPolarStereo(central_longitude=-100))
    return dict(region="na", lat=lat, lon=lon,
                plon=np.where(lon > 180, lon - 360, lon),
                proj=ccrs.LambertConformal(central_longitude=-100, central_latitude=45,
                                           standard_parallels=(30, 60)))


def _anomalies(meds: dict, var: str, period: str, init: pd.Timestamp):
    """meds {ens:(lead,lat,lon)} → {ens:(lead,lat,lon)} anomalies vs the day-of-year
    normal (looked up at each forecast valid date's day-of-year)."""
    cl = xr.open_dataarray(REF / f"ens_clim_{var}_{period}.nc")     # (dayofyear, lat, lon)
    doys = [int((init + pd.Timedelta(hours=ld)).dayofyear) for ld in LEADS]
    clim = np.stack([cl.sel(dayofyear=d).values for d in doys])     # (lead, lat, lon)
    return {e: m - clim for e, m in meds.items()}


def _panel(ax, field, overlay, levels, cmap, norm, title, cfg):
    """One panel: discrete-band fill of `field` (anomaly/change) + black contour
    overlay of `overlay` (the ensemble-median height, z500 only)."""
    if cfg["region"] == "nh":
        ax.set_extent([-180, 180, 15, 90], crs=ccrs.PlateCarree())
        ax.set_boundary(_CIRCLE, transform=ax.transAxes)
        x = cfg["lon"]
        ff, xf = add_cyclic_point(field, coord=x)               # close 0/360 seam
    else:
        ax.set_extent([-168, -52, 14, 76], crs=ccrs.PlateCarree())
        x = cfg["plon"]; ff, xf = field, x
    cf = ax.contourf(xf, cfg["lat"], ff, levels=levels, cmap=cmap, norm=norm,
                     extend="both", transform=ccrs.PlateCarree())
    if overlay is not None:                                     # median height contours
        if cfg["region"] == "nh":
            oo, xo = add_cyclic_point(overlay, coord=x)
        else:
            oo, xo = overlay, x
        cs = ax.contour(xo, cfg["lat"], oo, levels=_GPH_CONTOURS, colors="k",
                        linewidths=0.4, transform=ccrs.PlateCarree())
        ax.clabel(cs, levels=cs.levels[::2], fmt="%d", fontsize=5, inline=True)
    # edgecolor (NOT color=) + facecolor none: color= fills the land polygons gray.
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="0.35", facecolor="none")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="0.45", facecolor="none")
    ax.set_title(title, fontsize=9, fontweight="bold")
    return cf


_SHORT = {"z500": "500 mb height", "t2m": "2 m temperature"}


def _render_frames(fields: dict, overlay: dict, var: str, product: str, kind: str,
                   cbl: str, init: pd.Timestamp, anim_dir: Path, manifest_path: Path) -> int:
    """Generic per-(var,product) renderer. `fields` = {ensemble:(nlead,lat,lon)} values
    to fill (anomaly OR change); `overlay` = {ensemble:(nlead,lat,lon)} median height to
    contour (or None). Shared by the anomaly and change products."""
    cfg = _plot_cfg(var)
    levels = _LEVELS[(kind, var)]
    cmap, norm = _disc(levels)
    ens = list(fields)                                              # panel order
    nc = 1 if len(ens) == 1 else 2
    nr = int(np.ceil(len(ens) / nc))
    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, ld in enumerate(LEADS):
        valid = init + pd.Timedelta(hours=ld)
        fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 4.7 * nr),
                                 subplot_kw={"projection": cfg["proj"]})
        axes = np.atleast_1d(axes).ravel()
        cf = None
        for j, e in enumerate(ens):
            ov = overlay[e][i] if overlay is not None else None
            cf = _panel(axes[j], fields[e][i], ov, levels, cmap, norm, ENS_LABEL[e], cfg)
        for ax in axes[len(ens):]:
            ax.axis("off")
        # two-line title that fits the figure width
        fig.suptitle(f"{_SHORT[var]} — {cbl} ({VARS[var]['units']})\n"
                     f"init {init:%Y-%m-%d %HZ}  ·  Day {ld // 24}  ·  valid {valid:%a %d %b %Y}",
                     fontsize=9.5, fontweight="bold", linespacing=1.3)
        cax = fig.add_axes([0.25, 0.045, 0.5, 0.016])
        fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=levels, extend="both"
                     ).ax.tick_params(labelsize=6)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.085, wspace=0.04, hspace=0.12)
        fp = anim_dir / f"F{i:02d}.webp"
        fig.savefig(fp, dpi=115); plt.close(fig)
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"Day {ld // 24} · valid {valid:%a %b %d}"})
    region = anim_dir.name                                 # match the frame directory
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {region: {"label": f"{_SHORT[var]} — {cbl}",
                                 "n_frames": len(frames), "frames": frames}}}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(mani))
    print(f"  rendered {len(frames)} frames + {manifest_path.name} ({len(ens)} panels)")
    return 0


def render_product(meds: dict, var: str, product: str, init: pd.Timestamp,
                   anim_dir: Path, manifest_path: Path) -> int:
    """Anomaly product (anom30): fill = model median − ERA5 normal; overlay = median height."""
    period, cbl = PRODUCTS[product]
    anoms = _anomalies(meds, var, period, init)
    overlay = meds if var == "z500" else None
    return _render_frames(anoms, overlay, var, product, "anom", cbl, init, anim_dir, manifest_path)


def render_change(fields: dict, meds: dict, var: str, product: str, init: pd.Timestamp,
                  anim_dir: Path, manifest_path: Path) -> int:
    """Change product (dprev / d48): fill = current median − earlier run (same valid
    time); overlay = current median height."""
    _, cbl = CHANGE[product]
    overlay = ({e: meds[e] for e in fields} if var == "z500" else None)
    return _render_frames(fields, overlay, var, product, "chg", cbl, init, anim_dir, manifest_path)
