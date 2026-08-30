#!/usr/bin/env python3
"""
Stratospheric vortex winds: 10 and 100 hPa, both hemispheres.

Four polar panels — 10 hPa NH / SH on the top row, 100 hPa NH / SH on the
bottom — showing wind SPEED (shaded) with STREAMLINES over it, from the
AIFS-ENS control.

Why these two levels, in this pairing:

  10 hPa   is where the vortex itself lives and where an SSW is declared (the
           WMO diagnostic is a reversal of the zonal-mean zonal wind at 10 hPa,
           60 deg). A coherent ring of fast westerlies here is a strong,
           undisturbed vortex; a displaced or split circulation is visible
           directly as the streamlines stop encircling the pole.
  100 hPa  is the level that decides whether any of that reaches the surface.
           Anomalies confined to 10 hPa routinely go nowhere; the ones that
           show at 100 hPa are the ones with a path to the troposphere. Putting
           the two on one figure is the whole point - the question is never
           "what is the vortex doing" but "is it doing it deep enough to
           matter".

Both hemispheres because the SH vortex is the cleaner laboratory (stronger,
less wave-disturbed, and its spring breakdown is a scheduled event), and
because the reader should be able to see at a glance which hemisphere is in
its active season.

The speed scales are per-level and fixed, NOT per-panel: 10 hPa winds run far
stronger than 100 hPa, so a shared scale would flatten the lower level into a
single colour, while a per-panel scale would make NH and SH incomparable at a
glance - which is exactly the comparison the figure exists to support.

Data rides the pull the rest of the stratosphere pipeline already makes: u at
10/100 comes from the AAM levels (LEVELS_AAM includes both), and v is ensured
through the shared ECMWF store, control only. No new download of consequence.

Usage:
  python vortex_winds.py --date 20260829 --time 12 --out assets/sst/vortex_winds.webp
  python vortex_winds.py --step 120        # a forecast hour rather than analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf                                    # shared download manager

PC = ccrs.PlateCarree()
LEVELS = (10, 100)
# 12-hourly rather than the store's daily STEPS. The vortex can displace or
# split over a day, so daily frames alias the very evolution the loop exists to
# show. Verified against the open-data index: u, v and z are all published at
# both levels on every 12 h step. 31 frames to day 15.
STEPS_12H = tuple(range(0, 361, 12))
# Equatorward edge of each panel. 20 deg keeps the subtropical jet's poleward
# flank in view, which is what you want for judging whether the 100 hPa flow is
# connected to the troposphere rather than sitting on top of it.
LAT_EDGE = 20.0
# Streamline input is coarsened to this stride (lat, lon) - see panel().
STREAM_STRIDE = (3, 4)
# The still shows where the vortex lives and where an SSW is declared.
STILL_LEVEL = 10

# Per-level speed scales (m/s). Chosen from the climatological range at each
# level rather than from one cycle, so the colour of a given wind speed means
# the same thing every day - the figure is meant to be compared against itself
# over a season.
# Speeds below the first edge are left WHITE. A summer hemisphere is near-calm
# from pole to subtropics and tinting all of it says "look here" about nothing;
# blanking it makes the active hemisphere and the jet cores read immediately.
# Thresholds are per level because 10 hPa runs far stronger than 100.
SPEED = {
    10:  dict(levels=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
              label="10 hPa wind speed (m s$^{-1}$)"),
    100: dict(levels=[10, 15, 20, 25, 30, 35, 40, 45, 50, 60],
              label="100 hPa wind speed (m s$^{-1}$)"),
}
# Sequential, perceptually ordered, and light at the bottom so the streamlines
# (dark) stay legible over weak flow where they matter most.
CMAP = "YlGnBu"


def open_field(path, short: str, lev: int):
    """The whole forecast for one shortName/level, step dimension intact.

    Opened once per (var, level) rather than once per step: the loop walks 31
    steps and re-opening would decode the same grib 31 times.
    """
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": short, "level": lev}, indexpath=""))
    return ds[short] if short in ds else ds[list(ds.data_vars)[0]]


def at_step(da, step_h: int):
    if "step" not in da.dims:
        return da
    steps = (da.step.values / np.timedelta64(1, "h")).astype(int)
    if step_h not in set(steps):
        raise SystemExit(f"step {step_h}h unavailable (have {steps.min()}..{steps.max()})")
    return da.isel(step=int(np.where(steps == step_h)[0][0]))


def fetch(date: str, time: str):
    """u and v at both levels, every 12 h step, control only."""
    cyc = ecmwf.Cycle(date, time)
    upath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", LEVELS, STEPS_12H))
    vpath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", LEVELS, STEPS_12H))
    return {lev: (open_field(upath, "u", lev), open_field(vpath, "v", lev))
            for lev in LEVELS}


def circular_boundary(ax):
    """Clip a polar panel to a circle.

    A polar-stereographic axes left square shows a corner of the opposite
    hemisphere and a lot of empty projection space; the circle is what makes
    these read as a vortex rather than as a map that happens to be round.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def panel(ax, u, v, lev, hemi):
    """One hemisphere at one level: speed shaded, streamlines over."""
    lat = u.latitude.values
    lon = u.longitude.values
    north = hemi == "NH"
    sel = (lat >= LAT_EDGE) if north else (lat <= -LAT_EDGE)
    la = lat[sel]
    U = u.values[sel, :]
    V = v.values[sel, :]
    spd = np.hypot(U, V)

    ax.set_extent([-180, 180, LAT_EDGE if north else -90,
                   90 if north else -LAT_EDGE], crs=PC)
    circular_boundary(ax)

    cfg = SPEED[lev]
    # "min" extend would colour the sub-threshold band; leaving it unset means
    # anything below levels[0] is simply not drawn, so the white page shows
    # through - which is the point.
    cf = ax.contourf(lon, la, spd, levels=cfg["levels"], cmap=CMAP,
                     extend="max", transform=PC, zorder=1)

    # Streamlines need a monotonically increasing latitude axis; the GRIB comes
    # north-to-south, and for the SH selection that leaves it descending.
    #
    # Coarsened first, and this is the whole cost of the figure. Measured on the
    # SH 10 hPa panel: streamplot at the native 0.25 deg takes 94.5 s while
    # every other element of the panel together takes 0.2 s - cartopy has to
    # transform ~400k wind vectors into projection space before matplotlib
    # integrates a single streamline. At density 2.2 the integrator cannot
    # resolve anything near 0.25 deg, so that work is discarded: dropping to
    # 0.75 x 1.0 deg gives a visually identical panel in 8.0 s (12x). Shading
    # keeps the full grid - contourf costs 0.1 s and the crisp speed field is
    # what the eye actually reads.
    ys, xs = STREAM_STRIDE
    las, lons = la[::ys], lon[::xs]
    order = np.argsort(las)
    ax.streamplot(lons, las[order], U[::ys, ::xs][order, :],
                  V[::ys, ::xs][order, :], transform=PC,
                  density=2.2, linewidth=0.55, arrowsize=0.65,
                  color="#22303a", zorder=3)

    # The u = 0 contour is the vortex edge in the zonal sense: inside it the
    # flow is westerly and waves can propagate, outside it they cannot.
    #
    # Smoothed before contouring. The critical line is a planetary-scale
    # feature, but in a summer hemisphere u sits near zero over the whole cap,
    # so the raw contour shatters into hundreds of specks that carry no
    # information and bury the flow underneath. sigma is small (~1 deg) and
    # wraps in longitude so the dateline is not a seam.
    Us = gaussian_filter(U, sigma=(2, 4), mode=("nearest", "wrap"))
    ax.contour(lon, la, Us, levels=[0], colors="#b3122b", linewidths=1.4,
               transform=PC, zorder=4)

    # Geography reads FIRST: darker and heavier than the streamlines (0.55) so
    # the eye locates the vortex over the map rather than hunting for the
    # coast among the flow lines.
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#0d0d0d",
                   linewidth=1.05, zorder=6)
    gl = ax.gridlines(linewidth=0.35, color="#8a8a8a", alpha=0.5, zorder=6)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    # 60 deg gets its own heavier ring: the WMO sudden-stratospheric-warming
    # diagnostic is the zonal-mean zonal wind at 10 hPa, 60 deg, so this is the
    # latitude the reader is implicitly asked to judge.
    lat60 = 60.0 if north else -60.0
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat60), transform=PC,
            color="#1a1a1a", linewidth=1.35, linestyle=(0, (5, 3)), zorder=7)

    peak = float(np.nanmax(spd))
    ax.set_title(f"{hemi} · {lev} hPa    max {peak:.0f} m s$^{{-1}}$",
                 fontsize=10.5, pad=6)
    return cf


def render(uv, lev, date, time, step_h, out_path: Path):
    """ONE level, both hemispheres, side by side.

    Unstacked for the same reason wave1_maps was: 10 and 100 hPa shared a 2x2
    figure, which made the animation nearly square and very tall, pushing the
    scrubber below the fold and making the loop awkward to control. One level
    per frame halves the height, and the animator's selector switches levels.
    """
    fig = plt.figure(figsize=(11.4, 6.3), dpi=125)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                          left=0.02, right=0.93, top=0.83, bottom=0.085,
                          wspace=0.06)
    cf = None
    for c, hemi in enumerate(("NH", "SH")):
        proj = (ccrs.NorthPolarStereo(central_longitude=0) if hemi == "NH"
                else ccrs.SouthPolarStereo(central_longitude=0))
        ax = fig.add_subplot(gs[0, c], projection=proj)
        u, v = uv
        cf = panel(ax, u, v, lev, hemi)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(cf, cax=cax, extend="max")
    cb.set_label(SPEED[lev]["label"], fontsize=9)
    cb.ax.tick_params(labelsize=8)

    init = pd.Timestamp(f"{date}T{time}:00")
    valid = init + pd.Timedelta(hours=step_h)
    tag = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle(f"Stratospheric vortex winds — {lev} hPa",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.925,
             f"ECMWF AIFS-ENS control · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z "
             f"{tag} · valid {valid:%a %d %b %HZ} · speed shaded, streamlines "
             f"overlaid · red: u = 0 (vortex edge) · dashed ring: 60°",
             fontsize=9, color="#555", ha="left")
    fig.text(0.02, 0.020,
             "Colour scales are fixed per level, so a colour means the same wind "
             "speed every day; NH and SH share the scale and are directly comparable.",
             fontsize=8, color="#777", ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor="white",
                pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out_path


def build_loop(full, date, time, anim_root: Path, manifest: Path) -> int:
    """One frame per 12 h step PER LEVEL, and a manifest with a level selector.

    anim_root is the animation ROOT (assets/sst/anim); this writes
    vortex_10/ and vortex_100/ under it, one region each.
    """
    import json
    init = pd.Timestamp(f"{date}T{time}:00")
    regions = {}
    for lev in LEVELS:
        d = anim_root / f"vortex_{lev}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        frames = []
        for i, step_h in enumerate(STEPS_12H):
            u, v = full[lev]
            render((at_step(u, step_h), at_step(v, step_h)), lev, date, time,
                   step_h, d / f"F{i:02d}.webp")
            valid = init + pd.Timedelta(hours=step_h)
            frames.append({"idx": i, "file": f"F{i:02d}.webp",
                           "date": f"{valid:%Y-%m-%d}",
                           "label": f"F{step_h:03d} · valid {valid:%a %d %b %HZ}"})
            print(f"    {lev} hPa  F{step_h:03d}", flush=True)
        regions[f"vortex_{lev}"] = {
            "label": f"Vortex winds — {lev} hPa, both hemispheres",
            "n_frames": len(frames), "frames": frames}
        print(f"  {lev} hPa: {len(frames)} frames -> {d}")
    man = {"ver": f"{date}{time}", "days": len(STEPS_12H),
           "selectorLabel": "Level", "default": "vortex_10",
           "regions": regions}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man))
    print(f"  wrote {manifest.name} with {len(regions)} regions")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD")
    ap.add_argument("--time", required=True, help="cycle hour, 00 or 12")
    ap.add_argument("--step", type=int, default=0, help="forecast hour (default 0)")
    ap.add_argument("--out-dir", default="assets/sst",
                    help="the still is written here as vortex_10.webp")
    ap.add_argument("--anim-dir", help="frames ROOT; vortex_10/ and vortex_100/ go under it")
    ap.add_argument("--manifest", help="animator manifest path")
    a = ap.parse_args(argv)

    full = fetch(a.date, a.time)
    # ONE still, at STILL_LEVEL - the loop carries both levels as regions and
    # the animator switches between them, so a second still would just be an
    # unused file committed every cycle.
    outd = Path(a.out_dir)
    u, v = full[STILL_LEVEL]
    print(f"  wrote {render((at_step(u, a.step), at_step(v, a.step)), STILL_LEVEL, a.date, a.time, a.step, outd / f'vortex_{STILL_LEVEL}.webp')}")
    if a.anim_dir and a.manifest:
        build_loop(full, a.date, a.time, Path(a.anim_dir), Path(a.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
