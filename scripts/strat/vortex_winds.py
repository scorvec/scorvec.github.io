#!/usr/bin/env python3
"""
Stratospheric vortex winds: 10 and 100 hPa, each hemisphere on its own.

One polar map per frame - a level and a hemisphere - showing wind SPEED (shaded, with thin speed contours) with
WIND BARBS over it and the GEOPOTENTIAL HEIGHT contoured in black, from the AIFS-ENS ENSEMBLE MEAN (25 perturbed
members; the analysis frame is the control). Barbs, not streamlines: a streamline through the pole is integrated
across a point where the grid's u and v have no fixed direction, and it tangled there; a barb is one vector, regridded
onto the map with its direction rotated into the projection. The height lines are the flow's own contours (the wind
runs along them), so a displaced or stretched vortex reads from them at a glance; the shading says how fast.

The pole itself is left blank poleward of POLE_MASK_LAT: a lat-lon grid's pole row is one point repeated at every
longitude, the regridding gives the shading a seam there and the barbs a knot, and nothing about the vortex is read
within four degrees of the pole.

Why the mean and not member 0. Member 0 is one
realisation, and by the second week its vortex shape is largely its own: on the
Sep 26 00Z run it pinched into a near-split at day 13-15 on a wave-2 heat-flux
burst above the members' 90th percentile, while the mean wave-2 flux stayed near
normal. Shown alone it reads as a forecast of a split. The mean is the part of
the flow the members agree on. It is also SMOOTHER and WEAKER than any single
member wherever they disagree, so a split carried by a few members appears as a
stretched, slackened vortex rather than two centres - that is the honest
reading of such an ensemble. u and v are averaged (linear), and the speed shaded
is the speed of the mean wind, the same field the barbs show.

Why these two levels, in this pairing:

  10 hPa   is where the vortex itself lives and where an SSW is declared (the
           WMO diagnostic is a reversal of the zonal-mean zonal wind at 10 hPa,
           60 deg). A coherent ring of fast westerlies here is a strong,
           undisturbed vortex; a displaced or split circulation is visible
           directly as the barbs stop circling the pole.
  100 hPa  is the level that decides whether any of that reaches the surface.
           Anomalies confined to 10 hPa routinely go nowhere; the ones that
           show at 100 hPa are the ones with a path to the troposphere. Putting
           the two on one figure is the whole point - the question is never
           "what is the vortex doing" but "is it doing it deep enough to
           matter".

Both hemispheres, as separate loops: the SH vortex is the cleaner laboratory (stronger, less wave-disturbed, and its
spring breakdown is a scheduled event); the loop opens on the hemisphere in its winter.

The speed scales are per-level and fixed, NOT per-panel: 10 hPa winds run far
stronger than 100 hPa, so a shared scale would flatten the lower level into a
single colour, while a per-panel scale would make NH and SH incomparable at a
glance - which is exactly the comparison the figure exists to support.

Data: u, v and z at 10 and 100 hPa for 25 perturbed members every 12 h (~2.8 GB a
cycle through the shared ECMWF store, Google mirror), plus the control at step 0.
Opened with dask one step at a time so the 25-member mean never holds more than
one step's members in memory.

IFS-ENS: --model ifs draws the mean of ALL 50
IFS-ENS perturbed members from the shared IFS-ENS u/v file (ifs_ens.py: 10 and 100 hPa, 12-hourly, Google Cloud mirror
only) and a 12-hourly geopotential-height pull of the same members into vortex_ifs_10/ and vortex_ifs_100/, same scales. Open data has no IFS control at pressure levels, so its
analysis frame is the IFS HRES step 0, the analysis the ensemble starts from.

Usage:
  python vortex_winds.py --date 20260829 --time 12 --anim-dir assets/sst/anim \
      --manifest assets/sst/anim/vortex_winds_manifest.json
  python vortex_winds.py --model ifs --date 20260927 --time 00 --anim-dir assets/sst/anim \
      --manifest assets/sst/anim/vortex_winds_ifs_manifest.json
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import xarray as xr
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf  # shared download manager

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
# Barb input is coarsened to this stride (lat, lon) before cartopy regrids it onto the map - see panel().
BARB_STRIDE = (4, 4)
BARB_GRID = 26  # barbs across the map, each way
# barbs in m/s, as the shading: half barb 5, full 10, pennant 50
BARBS = dict(half=5, full=10, flag=50)
HEMIS = ("nh", "sh")
# Speed shading, speed contours and barbs stop here: the grid's pole row has no direction (see the docstring).
POLE_MASK_LAT = 86.0
# Geopotential height contours (dam): the interval per level, chosen for seven or eight lines across a winter cap
# (the 100 hPa map carries dense barbs as well, so its lines are the sparser).
HEIGHT_STEP = {10: 24, 100: 16}
# The u = 0 vortex edge stops at this latitude. u is the eastward component, which has no fixed direction at the pole:
# a flow across the pole changes the sign of u from one side of it to the other and draws a false ring around it.
EDGE_LAT_MAX = 80.0
# Per-level speed scales (m/s). Chosen from the climatological range at each
# level rather than from one cycle, so the colour of a given wind speed means
# the same thing every day - the figure is meant to be compared against itself
# over a season.
# Speeds below the first edge are left WHITE. A summer hemisphere is near-calm
# from pole to subtropics and tinting all of it says "look here" about nothing;
# blanking it makes the active hemisphere and the jet cores read immediately.
# Thresholds are per level because 10 hPa runs far stronger than 100.
SPEED = {
    10: dict(levels=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], label="10 hPa wind speed (m s$^{-1}$)"),
    100: dict(levels=[10, 15, 20, 25, 30, 35, 40, 45, 50, 60], label="100 hPa wind speed (m s$^{-1}$)"),
}
# Sequential, perceptually ordered, and light at the bottom so the barbs
# (dark) stay legible over weak flow where they matter most.
CMAP = "YlGnBu"


def open_field(path, short: str, lev: int):
    """The whole forecast for one shortName/level, step dimension intact.

    Opened once per (var, level) rather than once per step: the loop walks 31
    steps and re-opening would decode the same grib 31 times.
    """
    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": short, "level": lev}, indexpath="")
    )
    return ds[short] if short in ds else ds[next(iter(ds.data_vars))]


def at_step(da, step_h: int):
    if "step" not in da.dims:
        return da
    steps = (da.step.values / np.timedelta64(1, "h")).astype(int)
    if step_h not in set(steps):
        raise SystemExit(f"step {step_h}h unavailable (have {steps.min()}..{steps.max()})")
    return da.isel(step=int(np.where(steps == step_h)[0][0]))


# Perturbed members averaged. 25, as for the wave-1 maps, the heat flux and the
# WAF: the mean of u and v converges well before 50, and the pull scales with N.
MEMBERS = 25


def _mean_field(path, short: str, lev: int):
    """The member mean for one shortName/level, every step, one step in memory at a time."""
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        chunks={"step": 1},
        backend_kwargs=dict(filter_by_keys={"shortName": short, "level": lev}, indexpath=""),
    )
    da = ds[short] if short in ds else ds[next(iter(ds.data_vars))]
    if "number" in da.dims:
        da = da.mean("number")
    return da.compute()


def _with_step(da, step_h: int):
    """A single-step field as a 1-step array, member and level coordinates dropped."""
    if "step" not in da.dims:
        da = da.expand_dims(step=[np.timedelta64(step_h, "h").astype("timedelta64[ns]")])
    return da.drop_vars([c for c in da.coords if c not in ("step", "latitude", "longitude")])


def fetch(date: str, time: str, members: int = MEMBERS):
    """u and v at both levels, every 12 h step: the analysis from the CONTROL, every
    forecast step as the mean of `members` perturbed members.

    Step 0 is the control because the perturbed members differ from the analysis
    only by their perturbations - averaging them smooths the very state the
    analysis is (same reasoning as wave_maps.fetch_analysis). If the perturbed
    pull fails outright, the control stands in and the subtitle says so.
    """
    cyc = ecmwf.Cycle(date, time)
    fc_steps = tuple(s for s in STEPS_12H if s)
    ana = {sh: ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", sh, "pl", LEVELS, (0,))) for sh in FIELDS}
    try:
        pf = {sh: ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", sh, "pl", LEVELS, fc_steps, members)) for sh in FIELDS}
        source = f"ensemble mean of {members} members"
    except Exception as e:  # noqa: BLE001
        print(f"  perturbed pull failed ({type(e).__name__}: {e}); falling back to the control", flush=True)
        pf = {sh: ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", sh, "pl", LEVELS, fc_steps)) for sh in FIELDS}
        source = "control (member 0) - the perturbed pull failed"
    out = {}
    for lev in LEVELS:
        trio = []
        for sh in FIELDS:
            a = _with_step(_mean_field(ana[sh], sh, lev), 0)
            f = _mean_field(pf[sh], sh, lev)
            f = f.drop_vars([c for c in f.coords if c not in ("step", "latitude", "longitude")])
            trio.append(xr.concat([a, f], dim="step"))
        trio[2] = trio[2] / (G0 * 10.0)  # geopotential (m2 s-2) -> height in dam
        out[lev] = tuple(trio)
    return out, source


def fetch_ifs(date: str, time: str):
    """IFS-ENS: u and v at both levels, every 12 h step, the forecast steps as the mean of ALL 50 perturbed members
    (the shared IFS-ENS file) and step 0 from the IFS HRES analysis (no ensemble control on open data)."""
    import ifs_ens as IE

    if not IE.published(date, time):
        raise SystemExit(f"IFS-ENS {date} {time}Z is not on the Google mirror yet")
    cyc = IE.E.Cycle(date, time)
    uv = IE.ensure(cyc, IE.spec_uv())
    gh = IE.ensure(cyc, IE.E.Spec("ifs", "pf", "gh", "pl", LEVELS, STEPS_12H))  # geopotential height, gpm, 12-hourly
    ana = IE.hres_analysis(cyc, ("u", "v", "gh"), LEVELS)
    out = {}
    for lev in LEVELS:
        trio = []
        for sh, path in (("u", uv), ("v", uv), ("gh", gh)):
            a = _with_step(_mean_field(ana, sh, lev), 0)
            f = _mean_field(path, sh, lev)
            f = f.isel(step=np.flatnonzero(IE.step_hours(f) > 0))
            f = f.drop_vars([c for c in f.coords if c not in ("step", "latitude", "longitude")])
            trio.append(xr.concat([a, f], dim="step"))
        trio[2] = trio[2] / 10.0  # gpm -> dam
        out[lev] = tuple(trio)
    return out, f"ensemble mean of {IE.MEMBERS} members"


FIELDS = ("u", "v", "z")  # the AIFS-ENS fields (z = geopotential, m2 s-2)
G0 = 9.80665
MODEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
PREFIX = {"aifs": "vortex", "ifs": "vortex_ifs"}  # frame directories / manifest regions


def circular_boundary(ax):
    """Clip a polar panel to a circle.

    A polar-stereographic axes left square shows a corner of the opposite
    hemisphere and a lot of empty projection space; the circle is what makes
    these read as a vortex rather than as a map that happens to be round.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def panel(ax, u, v, z, lev, hemi):
    """One hemisphere at one level: speed shaded and contoured, barbs over, geopotential height in black."""
    lat = u.latitude.values
    lon = u.longitude.values
    north = hemi == "NH"
    sel = (lat >= LAT_EDGE) if north else (lat <= -LAT_EDGE)
    la = lat[sel]
    U = u.values[sel, :]
    V = v.values[sel, :]
    Z = z.values[sel, :]
    spd = np.hypot(U, V)
    spd[np.abs(la) > POLE_MASK_LAT, :] = np.nan  # the pole left blank (docstring)

    ax.set_extent([-180, 180, LAT_EDGE if north else -90, 90 if north else -LAT_EDGE], crs=PC)
    circular_boundary(ax)

    cfg = SPEED[lev]
    # "min" extend would colour the sub-threshold band; leaving it unset means
    # anything below levels[0] is simply not drawn, so the white page shows
    # through - which is the point.
    # the shading and the u = 0 line get a cyclic column: without it the last longitude and 360 deg are not joined,
    # and a white seam runs from the edge to the pole
    spd_c, lon_c = add_cyclic_point(spd, coord=lon)
    cf = ax.contourf(lon_c, la, spd_c, levels=cfg["levels"], cmap=CMAP, extend="max", transform=PC, zorder=1)
    # cartopy cuts the fill at 180 deg; stroking each piece in its own colour hides the hairline there
    cf.set_edgecolor("face")
    # the same speed levels as thin lines, so a core and its gradient read without comparing shades
    ax.contour(
        lon_c, la, spd_c, levels=cfg["levels"], colors="#1b3a4d", linewidths=0.35, alpha=0.4, transform=PC, zorder=2
    )
    # geopotential height: the flow follows these, so the vortex's shape and centre read from them directly
    step = HEIGHT_STEP[lev]
    Zc, _ = add_cyclic_point(Z, coord=lon)
    zlev = np.arange(np.floor(np.nanmin(Zc) / step) * step, np.nanmax(Zc) + step, step)
    cz = ax.contour(lon_c, la, Zc, levels=zlev, colors="#111111", linewidths=0.7, transform=PC, zorder=5)
    ax.clabel(cz, fontsize=7, fmt="%d", inline=True, inline_spacing=2)

    # Barbs from a coarsened grid. cartopy regrids the vectors onto a regular grid in map space (regrid_shape),
    # rotating each into the projection, and that transform of every input point is the figure's cost: from the
    # native 0.25 deg it is ~400k points, from 1 deg ~26k, and the regridded barbs look the same.
    #
    # Rows poleward of POLE_MASK_LAT are left out, as the shading is: the pole row is one point repeated at every
    # longitude, each copy with its own u, v, so its vectors land on a single spot of the map pointing in different
    # directions.
    ys, xs = BARB_STRIDE
    rows = np.flatnonzero(np.abs(la) <= POLE_MASK_LAT)[::ys]
    ax.barbs(
        lon[::xs],
        la[rows],
        U[rows, ::xs],
        V[rows, ::xs],
        transform=PC,
        regrid_shape=BARB_GRID,
        barb_increments=BARBS,
        length=5.4,
        linewidth=0.6,
        color="#22303a",
        flip_barb=not north,  # southern-hemisphere convention: the feathers on the other side
        sizes=dict(emptybarb=0.0),  # calm points draw nothing: a summer hemisphere would be a field of circles
        zorder=3,
    )

    # The u = 0 contour is the vortex edge in the zonal sense: inside it the
    # flow is westerly and waves can propagate, outside it they cannot.
    #
    # Smoothed before contouring. The critical line is a planetary-scale
    # feature, but in a summer hemisphere u sits near zero over the whole cap,
    # so the raw contour shatters into hundreds of specks that carry no
    # information and bury the flow underneath. sigma is small (~1 deg) and
    # wraps in longitude so the dateline is not a seam.
    Us = gaussian_filter(U, sigma=(2, 4), mode=("nearest", "wrap"))
    Us[np.abs(la) > EDGE_LAT_MAX, :] = np.nan
    ax.contour(lon_c, la, add_cyclic_point(Us), levels=[0], colors="#b3122b", linewidths=1.4, transform=PC, zorder=4)

    # Geography reads FIRST: darker and heavier than the barbs (0.6) so the eye
    # locates the vortex over the map rather than hunting for the coast among them.
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#0d0d0d", linewidth=1.05, zorder=6)
    gl = ax.gridlines(linewidth=0.35, color="#8a8a8a", alpha=0.5, zorder=6)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    # 60 deg gets its own heavier ring: the WMO sudden-stratospheric-warming
    # diagnostic is the zonal-mean zonal wind at 10 hPa, 60 deg, so this is the
    # latitude the reader is implicitly asked to judge.
    lat60 = 60.0 if north else -60.0
    ax.plot(
        np.linspace(-180, 180, 361),
        np.full(361, lat60),
        transform=PC,
        color="#1a1a1a",
        linewidth=1.35,
        linestyle=(0, (5, 3)),
        zorder=7,
    )

    peak = float(np.nanmax(spd))
    ax.set_title(f"max {peak:.0f} m s$^{{-1}}$", fontsize=10, loc="right", pad=4)
    return cf


def render(uv, lev, hemi, date, time, step_h, out_path: Path, source: str = "control", model: str = "aifs"):
    """ONE level, ONE hemisphere: a single polar map and its colour bar (the animator's selector switches between the
    level-and-hemisphere loops)."""
    fig = plt.figure(figsize=(8.4, 8.0), dpi=125)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], left=0.03, right=0.9, top=0.83, bottom=0.075, wspace=0.05)
    north = hemi == "NH"
    proj = ccrs.NorthPolarStereo(central_longitude=0) if north else ccrs.SouthPolarStereo(central_longitude=0)
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    u, v, z = uv
    cf = panel(ax, u, v, z, lev, hemi)
    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(cf, cax=cax, extend="max")
    cb.set_label(SPEED[lev]["label"], fontsize=9)
    cb.ax.tick_params(labelsize=8)

    init = pd.Timestamp(f"{date}T{time}:00")
    valid = init + pd.Timedelta(hours=step_h)
    tag = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle(
        f"Stratospheric vortex winds — {lev} hPa, {'northern' if north else 'southern'} hemisphere",
        fontsize=14,
        fontweight="bold",
        x=0.03,
        ha="left",
        y=0.975,
    )
    if step_h == 0 and model == "ifs":
        what = "ECMWF IFS HRES analysis, the state IFS-ENS starts from (open data has no IFS-ENS control)"
    else:
        what = f"ECMWF {MODEL[model]} " + ("control (the analysis)" if step_h == 0 else source)
    fig.text(
        0.03,
        0.935,
        f"{what} · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z {tag} · "
        f"valid {valid:%a %d %b %HZ}\n"
        f"{'speed' if step_h == 0 else 'speed of the mean wind'} shaded, "
        f"{'barbs' if step_h == 0 else 'its barbs'} in m s$^{{-1}}$ (half 5, full 10, pennant 50)\n"
        f"black: geopotential height, dam, every {HEIGHT_STEP[lev]} dam · thin lines: the speed levels · red: u = 0 (vortex edge, to 80°)\n"
        f"dashed ring: 60° · the disc at the pole: left blank poleward of {POLE_MASK_LAT:.0f}°",
        fontsize=9,
        color="#555",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.03,
        0.012,
        "Where the members disagree the mean is weaker and smoother than any one of them: a split carried by a few\n"
        "members shows as a stretched, slackened vortex, not two centres. Colour scales are fixed per level, the same "
        "in both hemispheres.",
        fontsize=8,
        color="#6f6b64",
        ha="left",
        va="bottom",
        linespacing=1.4,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # loop frames: WebP q65 method 6 (~30-35 % fewer bytes than q82; text, contours and colour bars unchanged at 1x)
    fig.savefig(out_path, dpi=125, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
    plt.close(fig)
    return out_path


def build_loop(full, date, time, anim_root: Path, manifest: Path, source: str = "control", model: str = "aifs") -> int:
    """One frame per 12 h step per level and hemisphere, and a manifest with a selector over the four.

    anim_root is the animation ROOT (assets/sst/anim); this writes <prefix>_<level>_<nh|sh>/ under it, one region
    each. The manifest opens on 10 hPa in the hemisphere that is in its winter half (October-March: the north).
    """
    import json

    init = pd.Timestamp(f"{date}T{time}:00")
    regions = {}
    for lev in LEVELS:
        for h in HEMIS:
            d = anim_root / f"{PREFIX[model]}_{lev}_{h}"
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("F*.webp"):
                old.unlink()
            frames = []
            for i, step_h in enumerate(STEPS_12H):
                u, v, z = full[lev]
                render(
                    (at_step(u, step_h), at_step(v, step_h), at_step(z, step_h)),
                    lev,
                    h.upper(),
                    date,
                    time,
                    step_h,
                    d / f"F{i:02d}.webp",
                    source,
                    model,
                )
                valid = init + pd.Timedelta(hours=step_h)
                frames.append(
                    {
                        "idx": i,
                        "file": f"F{i:02d}.webp",
                        "date": f"{valid:%Y-%m-%d}",
                        "label": f"F{step_h:03d} · valid {valid:%a %d %b %HZ}",
                    }
                )
                print(f"    {lev} hPa {h.upper()}  F{step_h:03d}", flush=True)
            regions[f"{PREFIX[model]}_{lev}_{h}"] = {
                "label": f"Vortex winds — {lev} hPa, {'northern' if h == 'nh' else 'southern'} hemisphere, "
                f"{MODEL[model]} ensemble mean",
                "n_frames": len(frames),
                "frames": frames,
            }
            print(f"  {lev} hPa {h.upper()}: {len(frames)} frames -> {d}")
    winter = "nh" if init.month in (10, 11, 12, 1, 2, 3) else "sh"
    man = {
        "ver": f"{date}{time}",
        "days": len(STEPS_12H),
        "selectorLabel": "Level and hemisphere",
        "default": f"{PREFIX[model]}_10_{winter}",
        "regions": regions,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man))
    print(f"  wrote {manifest.name} with {len(regions)} regions")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD")
    ap.add_argument("--time", required=True, help="cycle hour, 00 or 12")
    ap.add_argument("--anim-dir", required=True, help="frames ROOT; <prefix>_<level>_<nh|sh>/ go under it")
    ap.add_argument("--manifest", required=True, help="animator manifest path")
    ap.add_argument("--model", choices=("aifs", "ifs"), default="aifs")
    a = ap.parse_args(argv)

    if a.model == "ifs":
        full, source = fetch_ifs(a.date, a.time)
    else:
        full, source = fetch(a.date, a.time)
    return build_loop(full, a.date, a.time, Path(a.anim_dir), Path(a.manifest), source, a.model)


if __name__ == "__main__":
    sys.exit(main())
