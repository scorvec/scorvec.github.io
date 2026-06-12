#!/usr/bin/env python3
"""Equatorial Pacific zonal-current + thermocline section — Copernicus Marine 1/12° ocean model.

A depth × longitude slice along the equator (1.5°S-1.5°N mean), 160°E-90°W, of the daily
analysis-forecast: zonal current (shaded; eastward = red) with the 20 °C isotherm (the thermocline)
overlaid. It shows the Equatorial Undercurrent (the eastward subsurface core), the westward South
Equatorial Current at the surface, and the thermocline tilt — all of which respond as El Niño
matures (relaxing trades → eastward surface anomalies, a flattening/deepening eastern thermocline).

    python scripts/sst/eq_current_section.py --out assets/sst/eq_current_section.webp
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import copernicusmarine as cm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cmems"                  # gitignored scratch
CUR = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
TEM = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
LON0, LON1, LATB, DMAX = 160, 270, 1.5, 400      # 160°E-90°W, ±1.5° lat, 0-400 m


def _pull(ds: str, var: str) -> xr.DataArray:
    """Latest daily field, equator-band (±1.5°) mean → (depth, longitude)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"sec_{var}.nc"
    cm.subset(dataset_id=ds, variables=[var], minimum_longitude=LON0, maximum_longitude=LON1,
              minimum_latitude=-LATB, maximum_latitude=LATB, minimum_depth=0, maximum_depth=DMAX,
              start_datetime=str(date.today() - timedelta(days=4)),
              end_datetime=str(date.today() + timedelta(days=1)),
              output_filename=out.name, output_directory=str(CACHE), overwrite=True)
    return xr.open_dataset(out)[var].isel(time=-1)


def render(u: xr.DataArray, t: xr.DataArray, out: Path) -> None:
    when = np.datetime_as_string(u["time"].values, unit="D")
    lon = u["longitude"].values; dep = u["depth"].values
    U = u.mean("latitude").values                       # (depth, lon)
    T = t.mean("latitude").values
    fig, ax = plt.subplots(figsize=(12, 5.6))
    lev = np.arange(-1.2, 1.21, 0.15)
    pm = ax.contourf(lon, dep, U, levels=lev, cmap="RdBu_r", extend="both")
    cs = ax.contour(lon, dep, T, levels=[20], colors="black", linewidths=2.0)       # thermocline
    ax.clabel(cs, fmt="20°C", fontsize=8)
    ax.contour(lon, dep, T, levels=range(12, 30, 2), colors="0.35", linewidths=0.4, alpha=0.6)
    ax.set_ylim(DMAX, 0); ax.set_xlim(LON0, LON1)
    ax.set_xticks([160, 180, 200, 220, 240, 260])
    ax.set_xticklabels(["160E", "180", "160W", "140W", "120W", "100W"])
    ax.set_ylabel("depth (m)"); ax.set_xlabel("longitude")
    ax.axvline(190, color="0.5", lw=0.4, ls=":"); ax.axvline(240, color="0.5", lw=0.4, ls=":")
    ax.text(215, -14, "Niño-3.4", ha="center", fontsize=8, color="0.4")
    ax.set_title(f"Equatorial Pacific zonal current & thermocline (1.5°S–1.5°N)  ·  {when}\n"
                 "red = eastward (incl. the Equatorial Undercurrent); black = 20°C isotherm",
                 fontsize=10, pad=12)
    cb = fig.colorbar(pm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cb.set_label("zonal current (m s⁻¹)   ·   eastward +")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  ({when})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "assets" / "sst" / "eq_current_section.webp"))
    args = ap.parse_args(argv)
    render(_pull(CUR, "uo"), _pull(TEM, "thetao"), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
