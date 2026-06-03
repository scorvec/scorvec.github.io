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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).parent))
from common import REF, grid, GRIDS, LEADS, VARS, ENS_LABEL

PRODUCTS = {  # product → (clim period or None, colorbar label prefix, fixed ±limit by var)
    "anom30": ("30yr", "anomaly vs 1991–2020", {"z500": 18.0, "t2m": 12.0}),
    "anom10": ("10yr", "anomaly vs 2014–2023", {"z500": 18.0, "t2m": 12.0}),
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


def _panel(ax, field, lim, title, cfg):
    # pcolormesh (not contourf): contourf + cyclic-point leaves gray gaps on the
    # polar-stereo projection. pcolormesh renders every cell cleanly and needs no
    # cyclic padding.
    fld = np.ma.masked_invalid(field)
    if cfg["region"] == "nh":
        ax.set_extent([-180, 180, 15, 90], crs=ccrs.PlateCarree())
        ax.set_boundary(_CIRCLE, transform=ax.transAxes)
        x = cfg["lon"]
    else:
        ax.set_extent([-168, -52, 14, 76], crs=ccrs.PlateCarree())
        x = cfg["plon"]
    cf = ax.pcolormesh(x, cfg["lat"], fld, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="auto", transform=ccrs.PlateCarree())
    # edgecolor (NOT color=) + facecolor none: color= sets facecolor too, which fills
    # the land polygons gray and paints over the field.
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="0.3", facecolor="none")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="0.3", facecolor="none")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25, edgecolor="0.5", facecolor="none")
    ax.set_title(title, fontsize=10, fontweight="bold")
    return cf


def render_product(meds: dict, var: str, product: str, init: pd.Timestamp,
                   anim_dir: Path, manifest_path: Path) -> int:
    period, cbl, lims = PRODUCTS[product]
    lim = lims[var]
    cfg = _plot_cfg(var)
    anoms = _anomalies(meds, var, period, init)
    ens = list(meds)                                                # panel order
    nc = 1 if len(ens) == 1 else 2
    nr = int(np.ceil(len(ens) / nc))
    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, ld in enumerate(LEADS):
        valid = init + pd.Timedelta(hours=ld)
        fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 4.4 * nr),
                                 subplot_kw={"projection": cfg["proj"]})
        axes = np.atleast_1d(axes).ravel()
        cf = None
        for j, e in enumerate(ens):
            cf = _panel(axes[j], anoms[e][i], lim, ENS_LABEL[e], cfg)
        for ax in axes[len(ens):]:
            ax.axis("off")
        fig.suptitle(f"{VARS[var]['label']} — {cbl}   ·   init {init:%Y-%m-%d %HZ}   ·   "
                     f"Day {ld // 24} (valid {valid:%a %Y-%m-%d})", fontsize=11, fontweight="bold")
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.018])
        fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both"
                     ).set_label(f"{VARS[var]['label']} {cbl} ({VARS[var]['units']})", fontsize=8)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.09, wspace=0.04, hspace=0.12)
        fp = anim_dir / f"F{i:02d}.webp"
        fig.savefig(fp, dpi=110); plt.close(fig)
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"Day {ld // 24} · valid {valid:%a %b %d}"})
    region = f"{var}_{product}"
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {region: {"label": f"{VARS[var]['label']} — {cbl}",
                                 "n_frames": len(frames), "frames": frames}}}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(mani))
    print(f"  rendered {len(frames)} frames + {manifest_path.name} ({len(ens)} panels)")
    return 0
