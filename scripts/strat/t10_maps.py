#!/usr/bin/env python3
"""Polar temperature at 10 hPa from the IFS control: the warming of a sudden warming, where it sits and how strong.

One polar map per 12 h step to day 15 and per hemisphere: the temperature's departure from its ERA5 1991-2020 normal
of that day and hour shaded (reference/t10_clim.nc, build_t10_clim.py), and the 10 hPa geopotential height contoured,
which outlines the vortex (the flow runs along the height lines). Fields: t at 10 hPa and the height from the IFS
control's shared files (ifs_ens.py ctrl_t10, ctrl_wind; ECMWF open data, Google Cloud mirror), 0.25 deg.

Run on its own as soon as the control is on the mirror (t10.yml; ~60 MB a run): the control lands about an hour and a
half before the ensemble the other IFS products wait for.

    python scripts/strat/t10_maps.py [--date 20261023 --time 12] --anim-dir assets/sst/anim
        -> anim/t10_hres_{nh,sh}/F00..F30.webp + anim/t10_hres_manifest.json (no date: the newest control run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "lib"))
import harmonic_clim as HC
import ifs_ens as IE
import vortex_winds as VW

CLIM = HERE / "reference" / "t10_clim.nc"
PREFIX = "t10_hres"
STEPS_12H = IE.STEPS_12H
LAT_EDGE = 20.0
HEIGHT_STEP = 24  # dam
LEVELS = np.arange(-36, 37, 4)  # K
PC = ccrs.PlateCarree()


def newest() -> tuple[str, str]:
    """The newest 00/12Z control run completely on the Google mirror: the one started at least 7 h ago, else the one
    before."""
    t = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=7)
    for back in range(2):
        c = (t - pd.Timedelta(hours=12 * back)).floor("12h")
        if IE.ctrl_published(f"{c:%Y%m%d}", f"{c:%H}"):
            return f"{c:%Y%m%d}", f"{c:%H}"
    raise SystemExit("no IFS control run complete on the Google mirror in the last day")


def fields(date: str, time: str):
    """(temperature K, height dam), both (step, latitude, longitude), from the IFS control's shared files."""
    if not IE.ctrl_published(date, time):
        raise SystemExit(f"IFS control {date} {time}Z is not on the Google mirror yet")
    cyc = IE.E.Cycle(date, time)
    t = VW._mean_field(IE.ensure(cyc, IE.spec_ctrl_t10()), "t", 10)
    gh = VW._mean_field(IE.ensure(cyc, IE.spec_ctrl_wind()), "gh", 10) / 10.0
    return t.load(), gh.load()


def normal_at(clim: xr.Dataset, valid: pd.Timestamp, like: xr.DataArray) -> xr.DataArray:
    """The ERA5 normal at that day and hour, on the forecast's grid (bilinear, across the date line too)."""
    n = HC.evaluate(clim["coef"].values, HC.times_of([valid]))[0]
    lon = clim.longitude.values
    da = xr.DataArray(
        np.concatenate([n, n[:, :1]], axis=1),
        dims=("latitude", "longitude"),
        coords={"latitude": clim.latitude.values, "longitude": np.append(lon, lon[0] + 360.0)},
    )
    target = like.longitude.values
    out = da.interp(latitude=like.latitude.values, longitude=np.where(target < lon[0], target + 360.0, target))
    return out.assign_coords(longitude=target)


def render(anom: xr.DataArray, gh: xr.DataArray, hemi: str, init: pd.Timestamp, step_h: int, out: Path) -> None:
    north = hemi == "nh"
    fig = plt.figure(figsize=(8.4, 8.0), dpi=125)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], left=0.03, right=0.9, top=0.86, bottom=0.075, wspace=0.05)
    proj = ccrs.NorthPolarStereo(central_longitude=0) if north else ccrs.SouthPolarStereo(central_longitude=0)
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    ax.set_extent([-180, 180, LAT_EDGE, 90] if north else [-180, 180, -90, -LAT_EDGE], crs=PC)
    VW.circular_boundary(ax)
    sel = anom.latitude >= LAT_EDGE - 2 if north else anom.latitude <= -LAT_EDGE + 2
    a, z = anom.where(sel, drop=True), gh.where(sel, drop=True)
    lon = np.append(a.longitude.values, a.longitude.values[0] + 360.0)  # close the seam at the date line
    cyc = lambda x: np.concatenate([x, x[:, :1]], axis=1)  # noqa: E731
    cf = ax.contourf(
        lon, a.latitude, cyc(a.values), levels=LEVELS, cmap="RdBu_r", extend="both", transform=PC, zorder=1
    )
    lo, hi = np.floor(np.nanmin(z) / HEIGHT_STEP) * HEIGHT_STEP, np.ceil(np.nanmax(z) / HEIGHT_STEP) * HEIGHT_STEP
    cs = ax.contour(lon, z.latitude, cyc(z.values), levels=np.arange(lo, hi + 1, HEIGHT_STEP), colors="#111",
                    linewidths=0.9, transform=PC, zorder=3)  # fmt: skip
    ax.clabel(cs, fmt="%d", fontsize=7, inline_spacing=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#2b2b2b", linewidth=0.6, zorder=2)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.5, zorder=4)
    gl.ylocator = plt.FixedLocator([40, 60, 80] if north else [-80, -60, -40])
    cb = fig.colorbar(cf, cax=fig.add_subplot(gs[0, 1]), ticks=LEVELS[::2])
    cb.set_label("temperature minus its 1991–2020 normal, K", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    valid = init + pd.Timedelta(hours=step_h)
    tag = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle(
        f"Polar temperature — 10 hPa, {'northern' if north else 'southern'} hemisphere",
        fontsize=14,
        fontweight="bold",
        x=0.03,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.03,
        0.935,
        f"ECMWF IFS control{' analysis' if not step_h else ''} · {init:%Y-%m-%d %HZ} {tag} · valid {valid:%a %d %b %HZ}\n"
        f"shaded: temperature anomaly from ERA5 1991–2020 at that day and hour · black: height, dam, every "
        f"{HEIGHT_STEP} dam",
        fontsize=9,
        color="#555",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.03,
        0.012,
        "One run, not an ensemble: by the second week a warming here is one realisation of many.\n"
        "The colour scale is fixed, the same in both hemispheres.",
        fontsize=8,
        color="#6f6b64",
        ha="left",
        va="bottom",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", help="YYYYMMDD (with --time; default: the newest control run on the mirror)")
    ap.add_argument("--time", help="00 or 12")
    ap.add_argument("--anim-dir", type=Path, help="frames ROOT: t10_hres_<hemi>/ and the manifest")
    ap.add_argument("--print-cycle", action="store_true", help="print the newest control run (YYYYMMDDHH) and stop")
    a = ap.parse_args()
    if a.print_cycle:
        print("".join(newest()))
        return 0
    if a.anim_dir is None:
        ap.error("--anim-dir is required")
    if not CLIM.exists():
        raise SystemExit(f"{CLIM.name} missing: build it with t10-clim.yml")
    clim = xr.open_dataset(CLIM).load()
    date, time = (a.date, f"{int(a.time):02d}") if a.date and a.time else newest()
    t, gh = fields(date, time)
    init = pd.Timestamp(f"{date}T{time}:00")
    regions = {}
    for hemi in VW.HEMIS:
        d = a.anim_dir / f"{PREFIX}_{hemi}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        frames = []
        for i, step_h in enumerate(STEPS_12H):
            valid = init + pd.Timedelta(hours=step_h)
            ti, zi = VW.at_step(t, step_h), VW.at_step(gh, step_h)
            render(ti - normal_at(clim, valid, ti), zi, hemi, init, step_h, d / f"F{i:02d}.webp")
            frames.append({"idx": i, "file": f"F{i:02d}.webp", "date": f"{valid:%Y-%m-%d}",
                           "label": f"F{step_h:03d} · valid {valid:%a %d %b %HZ}"})  # fmt: skip
        regions[f"{PREFIX}_{hemi}"] = {
            "label": f"Polar temperature — 10 hPa, {'northern' if hemi == 'nh' else 'southern'} hemisphere, IFS control",
            "n_frames": len(frames),
            "frames": frames,
        }
        print(f"  {hemi.upper()}: {len(frames)} frames -> {d}", flush=True)
    winter = "nh" if init.month in (10, 11, 12, 1, 2, 3) else "sh"
    man = {"ver": f"{date}{time}", "days": len(STEPS_12H), "selectorLabel": "Hemisphere",
           "default": f"{PREFIX}_{winter}", "regions": regions}  # fmt: skip
    (a.anim_dir / f"{PREFIX}_manifest.json").write_text(json.dumps(man))
    print(f"  wrote {PREFIX}_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
