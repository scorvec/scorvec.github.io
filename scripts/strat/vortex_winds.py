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
# Equatorward edge of each panel. 20 deg keeps the subtropical jet's poleward
# flank in view, which is what you want for judging whether the 100 hPa flow is
# connected to the troposphere rather than sitting on top of it.
LAT_EDGE = 20.0

# Per-level speed scales (m/s). Chosen from the climatological range at each
# level rather than from one cycle, so the colour of a given wind speed means
# the same thing every day - the figure is meant to be compared against itself
# over a season.
SPEED = {
    10:  dict(levels=np.arange(0, 121, 10), label="10 hPa wind speed (m s$^{-1}$)"),
    100: dict(levels=np.arange(0, 61, 5),   label="100 hPa wind speed (m s$^{-1}$)"),
}
# Sequential, perceptually ordered, and light at the bottom so the streamlines
# (dark) stay legible over weak flow where they matter most.
CMAP = "YlGnBu"


def open_field(path, short: str, lev: int, step_h: int):
    """One (lat, lon) field for a shortName/level/step out of a cached grib."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": short, "level": lev}, indexpath=""))
    da = ds[short] if short in ds else ds[list(ds.data_vars)[0]]
    if "step" in da.dims:
        steps = (da.step.values / np.timedelta64(1, "h")).astype(int)
        if step_h not in set(steps):
            raise SystemExit(f"step {step_h}h not in {short}@{lev} "
                             f"(have {steps.min()}..{steps.max()})")
        da = da.isel(step=int(np.where(steps == step_h)[0][0]))
    return da


def fetch(date: str, time: str, step_h: int):
    """u and v at both levels for one step. u rides the AAM pull; v is small."""
    cyc = ecmwf.Cycle(date, time)
    steps = tuple(ecmwf.STEPS)
    upath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", LEVELS, steps))
    vpath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", LEVELS, steps))
    out = {}
    for lev in LEVELS:
        u = open_field(upath, "u", lev, step_h)
        v = open_field(vpath, "v", lev, step_h)
        out[lev] = (u, v)
    return out


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
    cf = ax.contourf(lon, la, spd, levels=cfg["levels"], cmap=CMAP,
                     extend="max", transform=PC, zorder=1)

    # Streamlines need a monotonically increasing latitude axis; the GRIB comes
    # north-to-south, and for the SH selection that leaves it descending.
    order = np.argsort(la)
    ax.streamplot(lon, la[order], U[order, :], V[order, :], transform=PC,
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

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#2b2b2b",
                   linewidth=0.6, zorder=5)
    gl = ax.gridlines(linewidth=0.35, color="#8a8a8a", alpha=0.5, zorder=6)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    # 60 deg gets its own heavier ring: the WMO sudden-stratospheric-warming
    # diagnostic is the zonal-mean zonal wind at 10 hPa, 60 deg, so this is the
    # latitude the reader is implicitly asked to judge.
    lat60 = 60.0 if north else -60.0
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat60), transform=PC,
            color="#444", linewidth=0.9, linestyle=(0, (5, 3)), zorder=6)

    peak = float(np.nanmax(spd))
    ax.set_title(f"{hemi} · {lev} hPa    max {peak:.0f} m s$^{{-1}}$",
                 fontsize=10.5, pad=6)
    return cf


def render(fields, date, time, step_h, out_path: Path):
    fig = plt.figure(figsize=(11.4, 11.0), dpi=125)
    # Rows are levels, columns hemispheres; the right column is narrower only by
    # the colourbar gutter, so the four panels stay the same physical size and
    # NH/SH remain directly comparable.
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045],
                          left=0.02, right=0.93, top=0.90, bottom=0.055,
                          wspace=0.06, hspace=0.14)
    for r, lev in enumerate(LEVELS):
        cf = None
        for c, hemi in enumerate(("NH", "SH")):
            proj = (ccrs.NorthPolarStereo(central_longitude=0) if hemi == "NH"
                    else ccrs.SouthPolarStereo(central_longitude=0))
            ax = fig.add_subplot(gs[r, c], projection=proj)
            u, v = fields[lev]
            cf = panel(ax, u, v, lev, hemi)
        cax = fig.add_subplot(gs[r, 2])
        cb = fig.colorbar(cf, cax=cax, extend="max")
        cb.set_label(SPEED[lev]["label"], fontsize=9)
        cb.ax.tick_params(labelsize=8)

    valid = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle("Stratospheric vortex winds — 10 and 100 hPa",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.943,
             f"ECMWF AIFS-ENS control · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z "
             f"{valid} · speed shaded, streamlines overlaid · red: u = 0 "
             f"(vortex edge) · dashed ring: 60\u00b0, the SSW diagnostic latitude",
             fontsize=9, color="#555", ha="left")
    fig.text(0.02, 0.018,
             "Per-level colour scales are fixed, so a colour means the same wind "
             "speed every day; NH and SH share a scale at each level and are "
             "directly comparable.",
             fontsize=8, color="#777", ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor="white",
                pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD")
    ap.add_argument("--time", required=True, help="cycle hour, 00 or 12")
    ap.add_argument("--step", type=int, default=0, help="forecast hour (default 0)")
    ap.add_argument("--out", default="assets/sst/vortex_winds.webp")
    a = ap.parse_args(argv)

    fields = fetch(a.date, a.time, a.step)
    p = render(fields, a.date, a.time, a.step, Path(a.out))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
