#!/usr/bin/env python3
"""IMERG precipitation ANOMALY loops — departure from the IMERG climatology (7d/14d/30d).

Reanalysis-free: anomaly = recent N-day accumulation (IMERG Early daily) − the climatological
N-day accumulation for the same day-of-year, where the climatology is IMERG's own 20-year
Final record (harmonic coeffs from build_imerg_clim.py). Shows where it's been wetter
(green) or drier (brown) than normal. Same Colombia+Brazil framing, an `Accumulation`-style
dropdown over the windows. Reuses imerg_precip's data layer (daily grids, region, geometry).

    python scripts/sst/imerg_precip_anom.py        # ensure recent anomaly frames for all windows
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg_precip as IP                                   # shared data layer + region
from build_imerg_clim import eval_clim, OUT as CLIM_NC

ANIM_ROOT = IP.ANIM_ROOT
CFG = IP.CFG

# anomaly windows (all daily-based): id, N days, # daily frames in the loop, symmetric mm levels
ANOM_WINDOWS = [
    dict(id="precip_anom_7d",  days=7,  span=10, label="7-day",
         levels=[5, 10, 20, 40, 70, 120]),
    dict(id="precip_anom_14d", days=14, span=10, label="14-day",
         levels=[10, 20, 40, 70, 120, 200]),
    dict(id="precip_anom_30d", days=30, span=10, label="30-day",
         levels=[15, 30, 60, 110, 180, 300]),
    dict(id="precip_anom_mtd", days=31, mtd=True, span=10, label="month-to-date",
         levels=[15, 30, 60, 110, 180, 300]),
    # ENSO-development-season running total: everything since May 1 of the
    # current development year (rolls to the new May 1 each spring). The daily
    # cache accretes locally, so the growing window costs no re-downloads.
    dict(id="precip_anom_may1", since=(5, 1), span=10, label="since May 1",
         levels=[30, 60, 120, 220, 380, 600]),
]
DEFAULT_WINDOW = "precip_anom_14d"


def _n(win: dict, ft) -> int:
    """N days for a frame — fixed, day-of-month of the last day (month-to-date),
    or days since the window's anchor date (since=(month, day))."""
    if win.get("since"):
        mo, dy = win["since"]
        yr = ft.year if (ft.month, ft.day) >= (mo, dy) else ft.year - 1
        return (ft.date() - ft.date().replace(year=yr, month=mo, day=dy)).days
    return (ft - timedelta(days=1)).day if win.get("mtd") else win["days"]

# diverging dry(brown) ↔ wet(blue-green) — white at zero
_STOPS = ["#5a3410", "#9c6b1e", "#d2a24a", "#ecd9a6", "#ffffff",
          "#bce3cf", "#5cbf9a", "#1f8f8f", "#1b5a8a"]
ANOM_CMAP = LinearSegmentedColormap.from_list("precip_anom", _STOPS)


def clim_accum(coef: np.ndarray, end: datetime, n: int) -> np.ndarray:
    """Climatological n-day precip accumulation (mm) for the n days ending at `end`."""
    tot = None
    for k in range(1, n + 1):
        c = eval_clim(coef, (end - timedelta(days=k)).timetuple().tm_yday)   # mm/day
        tot = c.copy() if tot is None else tot + c
    return tot


def render_anom(field: np.ndarray, vlabel: str, out: Path, win: dict):
    lo0, lo1, la0, la1 = CFG["extent"]; clon = CFG["clon"]; scale = CFG["scale"]
    ml, mt = IP._grid_axes()
    LON, LAT = IP._LON[ml], IP._LAT[mt]
    off = IP._offset(LON, clon)
    lev = [-x for x in reversed(win["levels"])] + [x for x in win["levels"]]
    norm = BoundaryNorm(lev, ANOM_CMAP.N, extend="both")
    proj = ccrs.PlateCarree(central_longitude=clon)

    fig = plt.figure(figsize=CFG["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([IP._offset(lo0, clon), IP._offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#f3f1ec")                            # light bg for the diverging field
    cf = ax.contourf(off, LAT, field, levels=lev, cmap=ANOM_CMAP, norm=norm,
                     extend="both", transform=proj)
    ax.coastlines(linewidth=0.7, color="#555", resolution=scale)
    if CFG["borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.55, edgecolor="#666")
    if CFG["states"]:
        geoms = IP._state_geoms(CFG["states"], scale)
        if geoms:
            ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="none",
                              edgecolor="#888", linewidth=0.35, zorder=3)
    basin = IP._magdalena_geom()
    if basin:                                              # dark outline reads on the light anomaly bg
        ax.add_geometries(basin, crs=ccrs.PlateCarree(), facecolor="none",
                          edgecolor="#111", linewidth=1.2, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.4, linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([((t + 180) % 360) - 180
                  for t in range(int(lo0), int(lo1) + 1) if t % CFG["dlon"] == 0])
    gl.ylocator = mticker.FixedLocator(range(int(la0), int(la1) + 1, CFG["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=42, extend="both")
    cb.set_label("precip anomaly vs 2001–2025 climatology (mm)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(f"IMERG — {win['label']} precipitation anomaly  ·  {vlabel}", fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-scale", type=float, default=1.0)
    ap.parse_args(argv)
    if not CLIM_NC.exists():
        print(f"climatology {CLIM_NC.name} not built yet — run build_imerg_clim.py first", flush=True)
        return 1
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    _, anchor_day = IP._latest_anchors()
    step = timedelta(days=1)
    regions = {}
    for win in ANOM_WINDOWS:
        ftimes = [anchor_day - k * step for k in range(win["span"])][::-1]
        need = set()
        for ft in ftimes:
            for k in range(1, _n(win, ft) + 1):
                need.add((ft - k * step).replace(hour=0, minute=0, second=0, microsecond=0))
        IP.ensure_daily(need)
        anim = ANIM_ROOT / win["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            n = _n(win, ft)
            if n < 1:                          # frame at/before a "since" anchor date
                continue
            if not fp.exists():
                recent = IP.trailing_sum(IP.DAILY_CACHE, ft, n, step, "%Y%m%d")
                if recent is None:
                    continue
                anom = recent - clim_accum(coef, ft, n)
                if win.get("mtd"):
                    e = ft - step
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                else:
                    vlabel = f"{(ft - timedelta(days=n)):%b %d} – {(ft - step):%b %d, %Y}"
                render_anom(anom, vlabel, fp, win)
                print(f"  {win['id']}: rendered {tag}", flush=True)
            label = (f"MTD → {(ft - step):%b %-d}" if win.get("mtd")
                     else f"{(ft - timedelta(days=n)):%b %d} – {(ft - step):%b %d}")
            entries.append({"idx": len(entries), "file": fp.name, "date": f"{ft:%Y-%m-%d}", "label": label})
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[win["id"]] = {"label": f"{win['label']} anomaly", "frames": entries}

    manifest = ANIM_ROOT / "precip_anom_manifest.json"
    manifest.write_text(json.dumps({"ver": anchor_day.strftime("%Y%m%d"),
                                    "selectorLabel": "Window", "default": DEFAULT_WINDOW, "regions": regions}))
    print(f"wrote {len(regions)} windows + {manifest.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
