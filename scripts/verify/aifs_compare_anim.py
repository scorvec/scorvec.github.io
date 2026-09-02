#!/usr/bin/env python3
"""AIFS single vs AIFS-ENS control — side-by-side forecast animators.

Two loops per 00Z/12Z cycle, every 6 h through hour 360:
  * North America: MSLP + 1000-500 thickness + 6-h precip (TropicalTidbits style)
  * NH polar: 500 hPa height, contour-filled
the physical-space companion to the spectral-fidelity study.

Rendering is Julia/CairoMakie (scripts/julia/aifs_render.jl), fed by staged
regular-grid fields: Python does every projection exactly once (nearest-
neighbour warp indices onto projected grids; Natural Earth polylines projected
and cached), then writes one small npz per frame; N_JULIA parallel Julia
processes stride the frame list. Full render of both loops: well under a
minute vs ~15 min for the matplotlib path (kept as --engine mpl fallback).

Data (~400 MB/cycle) flows through the shared ecmwf store (scripts/ecmwf) —
per-file locks, GRIB message-count integrity, mirror fallbacks. A non-blocking
flock makes concurrent invocations (pipeline + manual) impossible.

    python aifs_compare_anim.py [--date YYYYMMDD --time 00] [--engine mpl]
"""
from __future__ import annotations
import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf
ANIM = REPO / "assets" / "sst" / "anim" / "aifs_compare"
MANIFEST = REPO / "assets" / "sst" / "anim" / "aifs_compare_manifest.json"
ANIM_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500"
MANIFEST_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500_manifest.json"
ANIM_T = REPO / "assets" / "sst" / "anim" / "aifs_t2m"
MANIFEST_T = REPO / "assets" / "sst" / "anim" / "aifs_t2m_manifest.json"
STAGE = HERE / "data" / "anim_stage"
OVCACHE = HERE / "data"
JULIA_SCRIPT = REPO / "scripts" / "julia" / "aifs_render.jl"
JULIA_PROJ = REPO / "scripts" / "julia"
N_JULIA = 6

STEPS = list(range(0, 361, 6))
G = 9.80665
EXTENT = [-128, -63, 22, 55]                  # North America zoom
P_LEVELS = [0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 50]
P_COLORS = ["#d7ecd7", "#a8d8a8", "#6cbf6c", "#2d8f2d", "#ffe57d",
            "#ffbb4d", "#ff8c33", "#e85c29", "#c62828"]


def fetch(date: str, hh: str) -> dict:
    """All four files through the shared store; returns {(model,kind): path}."""
    cyc = ecmwf.Cycle(date, hh)
    S = tuple(STEPS)
    out = {}
    for mkey, model, typ in (("single", "aifs-single", "fc"),
                             ("ens", "aifs-ens", "cf")):
        out[(mkey, "sfc")] = ecmwf.ensure(
            cyc, ecmwf.Spec(model, typ, ("msl", "tp"), "sfc", (), S))
        out[(mkey, "2t")] = ecmwf.ensure(
            cyc, ecmwf.Spec(model, typ, "2t", "sfc", (), S))
        out[(mkey, "z")] = ecmwf.ensure(
            cyc, ecmwf.Spec(model, typ, "z", "pl", (1000, 500), S))
    return out


def load_all(paths):
    out = {}
    for mkey in ("single", "ens"):
        kw = dict(engine="cfgrib")
        msl = xr.open_dataset(paths[(mkey, "sfc")], backend_kwargs=dict(
            filter_by_keys={"shortName": "msl"}, indexpath=""), **kw)
        tp = xr.open_dataset(paths[(mkey, "sfc")], backend_kwargs=dict(
            filter_by_keys={"shortName": "tp"}, indexpath=""), **kw)
        z = xr.open_dataset(paths[(mkey, "z")],
                            backend_kwargs={"indexpath": ""}, **kw)
        t2 = xr.open_dataset(paths[(mkey, "2t")],
                             backend_kwargs={"indexpath": ""}, **kw)
        out[mkey] = dict(msl=msl["msl"], tp=tp["tp"], z=z["z"], t2m=t2["t2m"])
    return out


# ---------------------------------------------------------------- projections
def _projections():
    import cartopy.crs as ccrs
    return (ccrs.LambertConformal(central_longitude=-97, central_latitude=39),
            ccrs.NorthPolarStereo(central_longitude=-100))


def _proj_grid(proj, extent, nx, glat, glon):
    """Regular grid in projected coords covering `extent`, plus flat nearest-
    neighbour indices into the (lat descending, lon 0..360) 0.25 deg grid."""
    import cartopy.crs as ccrs
    n = 160
    lons = np.linspace(extent[0], extent[1], n)
    lats = np.linspace(extent[2], extent[3], n)
    ex = np.concatenate([lons, lons, np.full(n, extent[0]), np.full(n, extent[1])])
    ey = np.concatenate([np.full(n, extent[2]), np.full(n, extent[3]), lats, lats])
    p = proj.transform_points(ccrs.PlateCarree(), ex, ey)
    ok = np.isfinite(p[:, 0])
    x0, x1 = p[ok, 0].min(), p[ok, 0].max()
    y0, y1 = p[ok, 1].min(), p[ok, 1].max()
    ny = int(round(nx * (y1 - y0) / (x1 - x0)))
    XX, YY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ll = ccrs.PlateCarree().transform_points(proj, XX, YY)
    lon, lat = ll[..., 0], ll[..., 1]
    valid = np.isfinite(lon) & np.isfinite(lat)
    lon = np.where(valid, lon, 0.0)
    lat = np.where(valid, lat, 90.0)
    dlat = abs(glat[1] - glat[0])
    ilat = np.clip(np.round((glat[0] - lat) / dlat), 0, len(glat) - 1).astype(np.int64)
    ilon = (np.round(((lon - glon[0]) % 360) / dlat).astype(np.int64)) % len(glon)
    return dict(x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1),
                flat=ilat * len(glon) + ilon, valid=valid)


def _iter_lines(geom):
    gt = geom.geom_type
    if gt == "LineString":
        yield geom
    elif gt in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_lines(g)
    elif gt == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            yield from _iter_lines(g)


def _project_lines(proj, geoms, bbox, coarse=0.0):
    """Concatenate projected polylines (NaN-separated), clipped to bbox."""
    import cartopy.crs as ccrs
    x0, x1, y0, y1 = bbox
    padx, pady = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    xs, ys = [], []
    for geom in geoms:
        for line in _iter_lines(geom):
            if line is None:
                continue
            c = np.asarray(line.coords)
            if coarse and len(c) > 400:
                c = c[::2]
            p = proj.transform_points(ccrs.PlateCarree(), c[:, 0], c[:, 1])
            x, y = p[:, 0], p[:, 1]
            m = (np.isfinite(x) & np.isfinite(y) & (x > x0 - padx) & (x < x1 + padx)
                 & (y > y0 - pady) & (y < y1 + pady))
            if not m.any():
                continue
            xs.append(np.where(m, x, np.nan))
            xs.append([np.nan])
            ys.append(np.where(m, y, np.nan))
            ys.append([np.nan])
    if not xs:
        return np.array([np.nan]), np.array([np.nan])
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32))


def _overlays_na(proj, bbox):
    out = OVCACHE / "ov_na.npz"
    if out.exists():
        return out
    import cartopy.feature as cfeature
    feats = dict(
        coast=cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
        borders=cfeature.NaturalEarthFeature("cultural",
                                             "admin_0_boundary_lines_land", "50m"),
        states=cfeature.NaturalEarthFeature("cultural",
                                            "admin_1_states_provinces_lines", "50m"),
        lakes=cfeature.NaturalEarthFeature("physical", "lakes", "50m"))
    d = {}
    for k, f in feats.items():
        d[f"{k}_x"], d[f"{k}_y"] = _project_lines(proj, f.geometries(), bbox)
    np.savez_compressed(out, **d)
    return out


def _overlays_nh(proj, bbox):
    out = OVCACHE / "ov_nh.npz"
    if out.exists():
        return out
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    d = {}
    for k, f in (("coast", cfeature.NaturalEarthFeature("physical", "coastline",
                                                        "110m")),
                 ("borders", cfeature.NaturalEarthFeature(
                     "cultural", "admin_0_boundary_lines_land", "110m"))):
        d[f"{k}_x"], d[f"{k}_y"] = _project_lines(proj, f.geometries(), bbox,
                                                  coarse=1.0)
    gx, gy = [], []
    lons = np.linspace(-180, 180, 361)
    for latc in (30, 50, 70):
        p = proj.transform_points(ccrs.PlateCarree(), lons, np.full_like(lons, latc))
        gx += [p[:, 0], [np.nan]]
        gy += [p[:, 1], [np.nan]]
    lats = np.linspace(20, 88, 69)
    for lonc in range(-180, 180, 60):
        p = proj.transform_points(ccrs.PlateCarree(), np.full_like(lats, lonc), lats)
        gx += [p[:, 0], [np.nan]]
        gy += [p[:, 1], [np.nan]]
    d["grid_x"] = np.concatenate(gx).astype(np.float32)
    d["grid_y"] = np.concatenate(gy).astype(np.float32)
    np.savez_compressed(out, **d)
    return out


def _warp(field, g, smooth=0.0):
    """NN warp onto the projected grid; optional light Gaussian smoothing for
    contoured fields (kills terrain noise in MSLP/thickness that otherwise
    spawns hundreds of labelled 2-px closed contours over the Rockies).
    Precip is never smoothed — its texture is the point of the comparison."""
    v = field.ravel()[g["flat"]]
    if smooth:
        from scipy.ndimage import gaussian_filter
        v = gaussian_filter(v, smooth)
    return np.where(g["valid"], v, np.nan).astype(np.float32)


# -------------------------------------------------------------------- staging
def stage(date: str, hh: str, paths) -> Path:
    t0 = time.time()
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all(paths)
    glat = F["single"]["msl"].latitude.values
    glon = F["single"]["msl"].longitude.values
    lcc, nps = _projections()
    gna = _proj_grid(lcc, EXTENT, 660, glat, glon)
    gnh = _proj_grid(nps, [-179.9, 180, 20, 90], 560, glat, glon)
    ovna = _overlays_na(lcc, (gna["x0"], gna["x1"], gna["y0"], gna["y1"]))
    ovnh = _overlays_nh(nps, (gnh["x0"], gnh["x1"], gnh["y0"], gnh["y1"]))
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(ovna, STAGE / "ov_na.npz")
    shutil.copy(ovnh, STAGE / "ov_nh.npz")

    V = {}
    for m in ("single", "ens"):
        msl = F[m]["msl"].transpose("step", "latitude", "longitude")
        tp = F[m]["tp"].transpose("step", "latitude", "longitude")
        z = F[m]["z"].transpose("step", "isobaricInhPa", "latitude", "longitude")
        sidx = {int(s / np.timedelta64(1, "h")): i
                for i, s in enumerate(msl.step.values)}
        levs = list(z.isobaricInhPa.values)
        zv = z.values
        z5 = zv[:, levs.index(500)] / G / 10.0
        V[m] = dict(msl=msl.values / 100.0, tp=tp.values,
                    thk=z5 - zv[:, levs.index(1000)] / G / 10.0,
                    z5=z5, sidx=sidx)

    frames_c, frames_z = [], []
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        d = {}
        for m, pre in (("single", "s_"), ("ens", "e_")):
            k = V[m]["sidx"][s]
            kp = V[m]["sidx"].get(s - 6)
            pr = (np.zeros_like(V[m]["tp"][k]) if s == 0
                  else np.clip(V[m]["tp"][k] - V[m]["tp"][kp], 0, None))
            d[pre + "pr"] = _warp(pr, gna)
            d[pre + "msl"] = _warp(V[m]["msl"][k], gna, smooth=2.0)
            d[pre + "thk"] = _warp(V[m]["thk"][k], gna, smooth=2.0)
        np.savez(STAGE / f"cmp{i:02d}.npz", **d)
        np.savez(STAGE / f"z5{i:02d}.npz",
                 s_z5=_warp(V["single"]["z5"][V["single"]["sidx"][s]], gnh, smooth=1.2),
                 e_z5=_warp(V["ens"]["z5"][V["ens"]["sidx"][s]], gnh, smooth=1.2))
        lab = f"h{s:03d} · {valid:%b %d %HZ}"
        frames_c.append(dict(
            id=f"cmp{i:02d}", npz=f"cmp{i:02d}.npz", idx=i, step=s,
            date=f"{valid:%Y-%m-%d}", label=lab,
            title=(f"MSLP (hPa) · 1000–500 thickness (dam, 540 bold) · 6-h precip"
                   f" — hour {s} · valid {valid:%a %b %d %HZ}"
                   f" · init {base:%Y-%m-%d %HZ}")))
        frames_z.append(dict(
            id=f"z5{i:02d}", npz=f"z5{i:02d}.npz", idx=i, step=s,
            date=f"{valid:%Y-%m-%d}", label=lab,
            title=(f"500 hPa geopotential height · NH — hour {s}"
                   f" · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}")))

    spec = dict(staging=str(STAGE), loops=[
        dict(kind="compare", overlays="ov_na.npz",
             xlim=[gna["x0"], gna["x1"]], ylim=[gna["y0"], gna["y1"]],
             p_levels=P_LEVELS, p_colors=P_COLORS, frames=frames_c),
        dict(kind="z500", overlays="ov_nh.npz",
             xlim=[gnh["x0"], gnh["x1"]], ylim=[gnh["y0"], gnh["y1"]],
             frames=frames_z)])
    sp = STAGE / "spec.json"
    sp.write_text(json.dumps(spec))
    print(f"staged {len(STEPS)}x2 frames in {time.time() - t0:.1f}s", flush=True)
    return sp


# ------------------------------------------------------------- julia + webp
def render_julia(specpath: Path) -> bool:
    t0 = time.time()
    procs = [subprocess.Popen(
        ["julia", "--project=" + str(JULIA_PROJ), "--startup-file=no", "-O1",
         str(JULIA_SCRIPT), str(specpath), str(p + 1), str(N_JULIA)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for p in range(N_JULIA)]
    ok = True
    for p in procs:
        out, _ = p.communicate()
        for ln in out.strip().splitlines():
            print(f"  [jl] {ln}", flush=True)
        ok &= p.returncode == 0
    n = len(list(STAGE.glob("*.png")))
    print(f"julia render: {n} frames in {time.time() - t0:.1f}s", flush=True)
    return ok and n == 2 * len(STEPS)


def publish(specpath: Path):
    """png -> webp into the site anim dirs + manifests."""
    from PIL import Image
    spec = json.loads(specpath.read_text())
    t0 = time.time()
    for loop, outdir, mpath, key, label in (
            (spec["loops"][0], ANIM, MANIFEST, "aifs_compare",
             "AIFS single vs AIFS-ENS control — MSLP/thickness/precip"),
            (spec["loops"][1], ANIM_Z, MANIFEST_Z, "aifs_z500",
             "AIFS single vs AIFS-ENS control — 500 hPa height (NH)")):
        outdir.mkdir(parents=True, exist_ok=True)
        for old in outdir.glob("F*.webp"):
            old.unlink()
        frames = []
        for fr in loop["frames"]:
            im = Image.open(STAGE / (fr["id"] + ".png"))
            fn = f"F{fr['idx']:02d}.webp"
            im.save(outdir / fn, quality=92, method=6)
            frames.append({"idx": fr["idx"], "file": fn, "date": fr["date"],
                           "label": fr["label"]})
        man = {"ver": int(time.time()), "days": len(frames),
               "regions": {key: {"label": label, "n_frames": len(frames),
                                 "frames": frames}}}
        mpath.write_text(json.dumps(man))
        print(f"published {len(frames)} -> {outdir.name} "
              f"({time.time() - t0:.1f}s)", flush=True)


# ------------------------------------------------- matplotlib fallback path
# ---------------------------------------------------------------- matplotlib
# Three loops, every frame drawn by one function; frames are rendered in
# parallel across the machine's cores (fork: the loaded fields are inherited,
# nothing is pickled). Serial on a 4-core runner the three loops took the
# best part of an hour (2026-09-02) - past the step timeout, so nothing was
# ever published.
Z500_CLIM = REPO / "scripts" / "verify" / "data" / "clim" / "clim_1p5.npz"
LOOPS = {
    "compare": dict(anim=ANIM, manifest=MANIFEST, region="aifs_compare",
                    label="AIFS single vs AIFS-ENS control — MSLP/thickness/precip"),
    "z500": dict(anim=ANIM_Z, manifest=MANIFEST_Z, region="aifs_z500",
                 label="AIFS single vs AIFS-ENS control — 500 hPa height anomaly + height (NH)"),
    "t2m": dict(anim=ANIM_T, manifest=MANIFEST_T, region="aifs_t2m",
                label="AIFS single vs AIFS-ENS control — 2 m temperature anomaly (North America)"),
}
_G = {}                                   # fields shared with forked workers


def z500_clim_on(lat, lon, doy, var="z500"):
    """ERA5 1991-2020 ±7 d day-of-year climatology on the model grid: z500 in
    dam, t2m in °C. The climatology lives on a 1.5° NH grid (see
    aifs_station_verify.py); NaN south of the equator, which neither loop shows."""
    if not Z500_CLIM.exists():
        return None
    d = np.load(Z500_CLIM)
    if var not in d.files:
        return None
    c = d[var][min(int(doy), 366) - 1].astype(float)             # (120, 240)
    if var == "t2m" and np.nanmean(c) > 100:
        c = c - 273.15
    # The WB2 source grid ends at 358.5E, so the climatology's last 1.5° column
    # (centre 359.125E) was never filled: NaN there spread into a wedge from
    # the pole to the Greenwich meridian on every anomaly map. Fill it by
    # wrapping between its neighbours.
    if np.isnan(c[:, -1]).all() and not np.isnan(c[:, -2]).all():
        c[:, -1] = 0.5 * (c[:, -2] + c[:, 0])
    clat = 90.0 - 0.25 * (np.arange(720).reshape(120, 6).mean(axis=1))
    clon = 0.25 * (np.arange(1440).reshape(240, 6).mean(axis=1))
    da = xr.DataArray(c, coords=dict(latitude=clat, longitude=clon),
                      dims=("latitude", "longitude"))
    # pad BOTH ends: the 1.5° cell centres run 0.75..359.25, so a 0.25° point
    # between 359.25 and 360 or between 0 and 0.75 fell outside the range and
    # came back NaN - a white wedge from the pole to the Greenwich meridian
    da = xr.concat([da.isel(longitude=-1).assign_coords(longitude=clon[-1] - 360.0),
                    da,
                    da.isel(longitude=0).assign_coords(longitude=clon[0] + 360.0)],
                   dim="longitude")
    lon360 = np.mod(lon, 360.0)
    out = da.interp(latitude=("y", lat), longitude=("x", lon360), method="linear").values
    return out / 10.0 if var == "z500" else out


def _subset(lat, lon, arr, lat0, lat1, lon0, lon1):
    """Cut a global (lat, lon) field to a lon/lat box (lon in -180..180) and
    return 2-D meshgrids: contouring 1.3 M global points onto a projection was
    what made a frame cost 24 s on a runner; the box is ~5% of that, and
    transform_first=True lets cartopy project the points before contouring."""
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180)
    lon180 = lon180[order]; arr = arr[:, order]
    mi = (lat >= lat0) & (lat <= lat1)
    mj = (lon180 >= lon0) & (lon180 <= lon1)
    X, Y = np.meshgrid(lon180[mj], lat[mi])
    return X, Y, arr[np.ix_(mi, mj)]


NA_BOX = (12.0, 66.0, -150.0, -40.0)          # generous around EXTENT for the LCC corners
TF = dict(transform_first=True)


def _na_axes(ax, cfeature, ccrs):
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.coastlines(lw=1.1, color="0.05")
    ax.add_feature(cfeature.BORDERS, lw=0.8, edgecolor="0.15", facecolor="none")
    ax.add_feature(cfeature.STATES, lw=0.55, edgecolor="0.3", facecolor="none")
    ax.add_feature(cfeature.LAKES, lw=0.55, edgecolor="0.3", facecolor="none")


def _prewarm_features():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    for proj, ext in ((ccrs.LambertConformal(central_longitude=-97, central_latitude=39), EXTENT),
                      (ccrs.NorthPolarStereo(central_longitude=-100), [-180, 180, 20, 90])):
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(ext, ccrs.PlateCarree())
        ax.coastlines(lw=0.5)
        for f in (cfeature.BORDERS, cfeature.STATES, cfeature.LAKES):
            ax.add_feature(f, lw=0.3, facecolor="none")
        fig.canvas.draw()
        plt.close(fig)


def _draw_frame(job):
    """Render one frame of one loop; returns the manifest entry. One retry:
    a transient failure must not take the whole loop down."""
    for attempt in (1, 2):
        try:
            return _draw_frame_once(job)
        except Exception as e:                            # noqa: BLE001
            if attempt == 2:
                print(f"  frame {job} failed twice: {str(e)[:90]}", flush=True)
                return None
            time.sleep(2)


def _draw_frame_once(job):
    kind, i, s = job
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    F, base = _G["F"], _G["base"]
    valid = base + pd.Timedelta(hours=s)
    sd = pd.Timedelta(hours=s)
    lat, lon = _G["lat"], _G["lon"]
    if kind == "z500":
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6), constrained_layout=True,
                                 subplot_kw={"projection": proj})
        levels = np.arange(486, 601, 6)
        alev = np.arange(-27, 27.1, 3)
        clim = z500_clim_on(lat, lon, valid.dayofyear)
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS control")):
            z5 = F[mkey]["z"].sel(step=sd, isobaricInhPa=500).values / G / 10.0
            # polar view: subset to the hemisphere (the saving) but let cartopy
            # transform the contours (transform_first folds the grid at the
            # pole and produced a solid black frame, 2026-09-02)
            nh = lat >= 12.0
            latn = lat[nh]
            # Start the longitude axis at the projection's antimeridian (80E for
            # a -100 central meridian) so the data wrap and the map boundary
            # coincide: otherwise contourf leaves a white seam from the pole to
            # the boundary where its polygons are clipped.
            lon360 = np.mod(lon, 360.0)
            order = np.argsort(lon360)                       # -180..180 input → 0..360 monotonic
            lon360 = lon360[order]
            k = int(np.argmin(np.abs(lon360 - 80.0)))
            lonr = np.concatenate([lon360[k:], lon360[:k] + 360.0])
            zs = z5[nh][:, order]
            z5c, lonc = add_cyclic_point(np.roll(zs, -k, axis=1), coord=lonr)
            if clim is not None:
                a_s = (z5 - clim)[nh][:, order]
                ac, _ = add_cyclic_point(np.roll(a_s, -k, axis=1), coord=lonr)
                cf = ax.contourf(lonc, latn, ac, levels=alev, cmap="RdBu_r", extend="both",
                                 transform=ccrs.PlateCarree())
            else:
                cf = ax.contourf(lonc, latn, z5c, levels=levels, cmap="turbo", extend="both",
                                 transform=ccrs.PlateCarree())
            cl = ax.contour(lonc, latn, z5c, levels=levels, colors="k", linewidths=0.55,
                            transform=ccrs.PlateCarree())
            ax.clabel(cl, levels=levels[::2], fmt="%d", fontsize=6.5, inline_spacing=2)
            ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
            ax.coastlines(lw=0.8, color="0.15")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.3", facecolor="none")
            ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 50, 70], xlocs=range(-180, 181, 60))
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("500 hPa height anomaly vs 1991–2020 (dam) · contours: height (dam, every 6)"
                     if clim is not None else "500 hPa height (dam)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"500 hPa height anomaly + height · NH — hour {s} · valid {valid:%a %b %d %HZ}"
                     f" · init {base:%Y-%m-%d %HZ}", fontsize=12, fontweight="bold")
    elif kind == "t2m":
        proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), constrained_layout=True,
                                 subplot_kw={"projection": proj})
        alev = np.arange(-16, 16.1, 2)
        tlev = np.arange(-40, 46, 5)
        clim = z500_clim_on(lat, lon, valid.dayofyear, var="t2m")
        if clim is None:
            plt.close(fig)
            return None
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS control")):
            t2 = F[mkey]["t2m"].sel(step=sd).values - 273.15
            # The climatology is a DAILY MEAN (WB2), so an instantaneous field
            # against it reads cold every morning and warm every evening. The
            # anomaly is therefore of the 24-h mean centred on the valid time
            # (the 6-h steps within ±12 h, truncated at the run's ends); the
            # contours stay instantaneous.
            steps = F[mkey]["t2m"].step.values
            win = [st for st in steps if abs((st - sd) / np.timedelta64(1, "h")) <= 12]
            t24 = F[mkey]["t2m"].sel(step=win).mean("step").values - 273.15
            X, Y, T = _subset(lat, lon, t2, *NA_BOX)
            _, _, A = _subset(lat, lon, t24 - clim, *NA_BOX)
            cf = ax.contourf(X, Y, A, levels=alev, cmap="RdBu_r", extend="both",
                             transform=ccrs.PlateCarree(), **TF)
            ct = ax.contour(X, Y, T, levels=tlev, colors="0.25", linewidths=0.5,
                            transform=ccrs.PlateCarree(), **TF)
            ax.clabel(ct, levels=tlev[::2], fmt="%d", fontsize=6, inline_spacing=2)
            c0 = ax.contour(X, Y, T, levels=[0], colors="#1565c0", linewidths=1.5,
                            transform=ccrs.PlateCarree(), **TF)
            ax.clabel(c0, fmt="%d°C", fontsize=6.5, inline_spacing=2)
            _na_axes(ax, cfeature, ccrs)
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("24-h mean 2 m temperature anomaly vs 1991–2020 (°C) · contours: 2 m temperature now (°C, every 5; 0 °C blue)",
                     fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"2 m temperature anomaly + temperature — hour {s} · valid {valid:%a %b %d %HZ}"
                     f" · init {base:%Y-%m-%d %HZ}", fontsize=12, fontweight="bold")
    else:                                                   # compare: MSLP / thickness / precip
        proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), constrained_layout=True,
                                 subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS control")):
            d = F[mkey]
            msl = d["msl"].sel(step=sd).values / 100.0
            thk = ((d["z"].sel(step=sd, isobaricInhPa=500)
                    - d["z"].sel(step=sd, isobaricInhPa=1000)).values / G / 10.0)
            if s == 0:
                pr = np.zeros_like(msl)
            else:
                pr = d["tp"].sel(step=sd).values - d["tp"].sel(step=sd - pd.Timedelta(hours=6)).values
            X, Y, PR = _subset(lat, lon, np.clip(pr, 0, None), *NA_BOX)
            _, _, TH = _subset(lat, lon, thk, *NA_BOX)
            _, _, MS = _subset(lat, lon, msl, *NA_BOX)
            cf = ax.contourf(X, Y, PR, levels=P_LEVELS, colors=P_COLORS,
                             extend="max", transform=ccrs.PlateCarree(), **TF)
            for rng, col in (((410, 540), "#1565c0"), ((546, 620), "#c62828")):
                ct = ax.contour(X, Y, TH, levels=np.arange(rng[0], rng[1], 6), colors=col,
                                linewidths=0.6, linestyles="--", transform=ccrs.PlateCarree(), **TF)
                if len(ct.levels):
                    ax.clabel(ct, levels=list(ct.levels)[::2], fmt="%d", fontsize=5.5, inline_spacing=2)
            c540 = ax.contour(X, Y, TH, levels=[540], colors="#1565c0", linewidths=1.6,
                              linestyles="--", transform=ccrs.PlateCarree(), **TF)
            ax.clabel(c540, fmt="%d", fontsize=6, inline_spacing=2)
            cs = ax.contour(X, Y, MS, levels=np.arange(940, 1061, 4), colors="k", linewidths=0.75,
                            transform=ccrs.PlateCarree(), **TF)
            ax.clabel(cs, levels=np.arange(940, 1061, 8), fmt="%d", fontsize=6)
            _na_axes(ax, cfeature, ccrs)
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("6-h precipitation (mm)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"MSLP (hPa) · 1000–500 thickness (dam, 540 bold) · 6-h precip — hour {s}"
                     f" · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold")
    out = LOOPS[kind]["anim"] / f"F{i:02d}.webp"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return (kind, {"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                   "label": f"h{s:03d} · {valid:%b %d %HZ}"})


def render_mpl(date: str, hh: str, paths, kinds=("compare", "z500", "t2m"), workers=None):
    import multiprocessing as mp
    import os
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all(paths)
    # cfgrib is lazy: read everything once in the parent so forked workers
    # never touch the GRIB files concurrently
    for mkey in F:
        for k in list(F[mkey]):
            F[mkey][k] = F[mkey][k].load()
    _G.update(F=F, base=base, lat=F["single"]["msl"].latitude.values,
              lon=F["single"]["msl"].longitude.values)
    for kind in kinds:
        LOOPS[kind]["anim"].mkdir(parents=True, exist_ok=True)
        for old in LOOPS[kind]["anim"].glob("F*.webp"):
            old.unlink()
    # Pre-warm cartopy's Natural Earth cache in the PARENT: on a fresh runner
    # every forked worker otherwise downloads the same shapefiles at once and
    # one of them reads a half-written file (struct.error, 2026-09-02).
    _prewarm_features()
    jobs = [(kind, i, s) for kind in kinds for i, s in enumerate(STEPS)]
    workers = workers or max(1, min(os.cpu_count() or 2, 8))
    t0 = time.time()
    frames = {k: [] for k in kinds}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_draw_frame, jobs, chunksize=2), 1):
            if res:
                frames[res[0]].append(res[1])
            if n % 30 == 0:
                print(f"  {n}/{len(jobs)} frames in {time.time() - t0:.0f}s ({workers} workers)", flush=True)
    for kind in kinds:
        fr = sorted(frames[kind], key=lambda f: f["idx"])
        if not fr:
            print(f"  {kind}: no frames rendered — manifest untouched", flush=True)
            continue
        man = {"ver": int(time.time()), "days": len(fr),
               "regions": {LOOPS[kind]["region"]: {"label": LOOPS[kind]["label"],
                                                   "n_frames": len(fr), "frames": fr}}}
        LOOPS[kind]["manifest"].write_text(json.dumps(man))
        print(f"wrote {len(fr)} {kind} frames + manifest", flush=True)
    print(f"rendered {len(jobs)} frames in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default="00")
    ap.add_argument("--engine", choices=("julia", "mpl"), default="julia")
    args = ap.parse_args()

    lockdir = REPO / "assets" / "sst" / "anim"
    lockdir.mkdir(parents=True, exist_ok=True)
    lockf = (lockdir / ".aifs_anim.lock").open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another aifs animator instance is running — aborting")

    paths = fetch(args.date, args.time)
    (HERE / "data").mkdir(parents=True, exist_ok=True)      # overlay caches live here; absent on a fresh runner
    if args.engine == "julia" and shutil.which("julia") is None:
        print("julia not on PATH (Actions runner) — rendering with matplotlib", flush=True)
        args.engine = "mpl"
    if args.engine == "julia":
        sp = stage(args.date, args.time, paths)
        if render_julia(sp):
            publish(sp)
        else:
            print("julia render incomplete — falling back to matplotlib",
                  file=sys.stderr, flush=True)
            render_mpl(args.date, args.time, paths)
    else:
        render_mpl(args.date, args.time, paths)


if __name__ == "__main__":
    main()
