#!/usr/bin/env python3
"""Tropical Pacific surface-current animator — Copernicus Marine 1/12° ocean model.

A rolling ~1-month loop of the daily surface current over the equatorial Pacific (15°S-15°N,
130°E-80°W): current *speed* shaded with streamlines tracing the flow. It lays out the equatorial
current system — the westward South Equatorial Current on the equator, the eastward North
Equatorial Counter Current near 5-10°N, the western-boundary and Central-American jets — and, as
El Niño matures, eastward surface flow building on the equator.

Frames are daily PNG/WebP named by date; the first build backfills the window in one block pull,
later runs append the new day(s). Feeds sst_anim.html (region "eq_cur_map"). Runs in run_local_sst.

    python scripts/sst/eq_current_map.py            # append new day(s) + refresh the loop
    python scripts/sst/eq_current_map.py --days 31  # backfill the rolling window
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cmems"
ANIM = HERE.parent.parent / "assets" / "sst" / "anim" / "eq_cur_map"
MANIFEST = HERE.parent.parent / "assets" / "sst" / "anim" / "eq_cur_map_manifest.json"
CUR = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
LON0, LON1, LATB = 130, 280, 15.0
KEEP_DAYS = 150   # back past Apr 1 2026 — first major downwelling Kelvin wave of the event


def pull_block(d0: date, d1: date) -> xr.Dataset:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "map_block.nc"
    cm.subset(dataset_id=CUR, variables=["uo", "vo"], minimum_longitude=LON0, maximum_longitude=LON1,
              minimum_latitude=-LATB, maximum_latitude=LATB, minimum_depth=0, maximum_depth=1,
              start_datetime=str(d0), end_datetime=str(d1),
              output_filename=out.name, output_directory=str(CACHE), overwrite=True)
    return xr.open_dataset(out).isel(depth=0)


def render_frame(day: xr.Dataset, dt: datetime, out: Path) -> None:
    lon = day["longitude"].values; lat = day["latitude"].values
    u = day["uo"].values; v = day["vo"].values
    spd = np.sqrt(u ** 2 + v ** 2)
    proj = ccrs.PlateCarree(central_longitude=180); pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 5.0)); ax = plt.axes(projection=proj)
    ax.set_extent([LON0, LON1, -LATB, LATB], crs=pc)
    pm = ax.contourf(lon, lat, spd, levels=np.arange(0, 1.41, 0.1), cmap="turbo", extend="max", transform=pc)
    s = 5
    ax.streamplot(lon[::s], lat[::s], u[::s, ::s], v[::s, ::s], transform=pc,
                  color="white", linewidth=0.6, density=3.0, arrowsize=0.7)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.7", zorder=3)
    ax.coastlines("50m", color="0.3", linewidth=0.5, zorder=4)
    ax.axhline(0, color="white", lw=0.5, ls=(0, (4, 3)), alpha=0.6, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-200, -180, -160, -140, -120, -100, -80])
    gl.ylocator = mticker.FixedLocator([-15, -10, -5, 0, 5, 10, 15])
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    ax.set_title(f"Tropical Pacific surface currents (15°S–15°N)  ·  {dt:%Y-%m-%d}\n"
                 "shading = speed · streamlines = flow direction", fontsize=11)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.07, aspect=48, shrink=0.72)
    cb.set_label("current speed (m s⁻¹)")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=KEEP_DAYS, help="ensure frames over the last N days")
    args = ap.parse_args(argv)
    ANIM.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    want = [today - timedelta(days=k) for k in range(min(args.days, KEEP_DAYS) - 1, -1, -1)]
    need = [d for d in want if not (ANIM / f"{d:%Y%m%d}.webp").exists()]
    if need:
        block = pull_block(min(need), max(need))               # one subset for all missing days
        bdates = pd.to_datetime(block["time"].values).date
        for d in need:
            i = np.argmin([abs((bd - d).days) for bd in bdates])
            if abs((bdates[i] - d).days) > 1:
                continue                                        # day not in the model output
            render_frame(block.isel(time=i), datetime(d.year, d.month, d.day), ANIM / f"{d:%Y%m%d}.webp")
            print(f"  rendered {d}", flush=True)
    # trim + manifest
    cutoff = today - timedelta(days=KEEP_DAYS)
    entries = []
    for fp in sorted(ANIM.glob("*.webp")):
        try:
            dt = datetime.strptime(fp.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if dt < cutoff:
            fp.unlink(); continue
        entries.append({"idx": len(entries), "file": fp.name,
                        "date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%-d %b %Y")})
    MANIFEST.write_text(json.dumps({"ver": today.strftime("%Y%m%d"),
                                    "regions": {"eq_cur_map": {"label": "Tropical Pacific surface currents",
                                                               "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
