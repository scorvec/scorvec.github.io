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
        out[mkey] = dict(msl=msl["msl"], tp=tp["tp"], z=z["z"])
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
            im.save(outdir / fn, quality=82, method=4)
            frames.append({"idx": fr["idx"], "file": fn, "date": fr["date"],
                           "label": fr["label"]})
        man = {"ver": int(time.time()), "days": len(frames),
               "regions": {key: {"label": label, "n_frames": len(frames),
                                 "frames": frames}}}
        mpath.write_text(json.dumps(man))
        print(f"published {len(frames)} -> {outdir.name} "
              f"({time.time() - t0:.1f}s)", flush=True)


# ------------------------------------------------- matplotlib fallback path
def render_mpl(date: str, hh: str, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all(paths)
    ANIM.mkdir(parents=True, exist_ok=True)
    for old in ANIM.glob("F*.webp"):
        old.unlink()
    frames = []
    lat = F["single"]["msl"].latitude.values
    lon = F["single"]["msl"].longitude.values
    proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1),
                                 constrained_layout=True,
                                 subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"),
                                  ("AIFS single", "AIFS-ENS control")):
            d = F[mkey]
            sd = pd.Timedelta(hours=s)
            msl = d["msl"].sel(step=sd).values / 100.0
            thk = ((d["z"].sel(step=sd, isobaricInhPa=500)
                    - d["z"].sel(step=sd, isobaricInhPa=1000)).values / G / 10.0)
            if s == 0:
                pr = np.zeros_like(msl)
            else:
                pr = (d["tp"].sel(step=sd).values
                      - d["tp"].sel(step=sd - pd.Timedelta(hours=6)).values)
            cf = ax.contourf(lon, lat, np.clip(pr, 0, None), levels=P_LEVELS,
                             colors=P_COLORS, extend="max",
                             transform=ccrs.PlateCarree())
            for rng, col in (((410, 540), "#1565c0"), ((546, 620), "#c62828")):
                ct = ax.contour(lon, lat, thk, levels=np.arange(rng[0], rng[1], 6),
                                colors=col, linewidths=0.6, linestyles="--",
                                transform=ccrs.PlateCarree())
                if len(ct.levels):
                    ax.clabel(ct, levels=list(ct.levels)[::2],
                              fmt="%d", fontsize=5.5, inline_spacing=2)
            c540 = ax.contour(lon, lat, thk, levels=[540], colors="#1565c0",
                              linewidths=1.6, linestyles="--",
                              transform=ccrs.PlateCarree())
            ax.clabel(c540, fmt="%d", fontsize=6, inline_spacing=2)
            cs = ax.contour(lon, lat, msl, levels=np.arange(940, 1061, 4),
                            colors="k", linewidths=0.75,
                            transform=ccrs.PlateCarree())
            ax.clabel(cs, levels=np.arange(940, 1061, 8), fmt="%d", fontsize=6)
            ax.set_extent(EXTENT, ccrs.PlateCarree())
            ax.coastlines(lw=1.1, color="0.05")
            ax.add_feature(cfeature.BORDERS, lw=0.8, edgecolor="0.15",
                           facecolor="none")
            ax.add_feature(cfeature.STATES, lw=0.55, edgecolor="0.3",
                           facecolor="none")
            ax.add_feature(cfeature.LAKES, lw=0.55, edgecolor="0.3",
                           facecolor="none")
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal",
                          pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("6-h precipitation (mm)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"MSLP (hPa) · 1000–500 thickness (dam, 540 bold) · "
                     f"6-h precip — hour {s} · valid {valid:%a %b %d %HZ} · "
                     f"init {base:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold")
        out = ANIM / f"F{i:02d}.webp"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        frames.append({"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"h{s:03d} · {valid:%b %d %HZ}"})
        if i % 12 == 0:
            print(f"  frame {i}/{len(STEPS)}", flush=True)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"aifs_compare": {
               "label": "AIFS single vs AIFS-ENS control — MSLP/thickness/precip",
               "n_frames": len(frames), "frames": frames}}}
    MANIFEST.write_text(json.dumps(man))
    print(f"wrote {len(frames)} frames + manifest")
    render_z500_mpl(F, base)


def render_z500_mpl(F, base):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ANIM_Z.mkdir(parents=True, exist_ok=True)
    for old in ANIM_Z.glob("F*.webp"):
        old.unlink()
    frames = []
    lat = F["single"]["z"].latitude.values
    lon = F["single"]["z"].longitude.values
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    levels = np.arange(486, 601, 6)
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6),
                                 constrained_layout=True,
                                 subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"),
                                  ("AIFS single", "AIFS-ENS control")):
            z5 = (F[mkey]["z"].sel(step=pd.Timedelta(hours=s),
                                   isobaricInhPa=500).values / G / 10.0)
            cf = ax.contourf(lon, lat, z5, levels=levels, cmap="turbo",
                             extend="both", transform=ccrs.PlateCarree())
            cl = ax.contour(lon, lat, z5, levels=levels[::2], colors="k",
                            linewidths=0.5, transform=ccrs.PlateCarree())
            ax.clabel(cl, levels=levels[::4], fmt="%d", fontsize=6,
                      inline_spacing=2)
            ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
            ax.coastlines(lw=0.8, color="0.15")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.3",
                           facecolor="none")
            ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 50, 70],
                         xlocs=range(-180, 181, 60))
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal",
                          pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("500 hPa height (dam)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"500 hPa geopotential height · NH — hour {s} · "
                     f"valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold")
        out = ANIM_Z / f"F{i:02d}.webp"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        frames.append({"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"h{s:03d} · {valid:%b %d %HZ}"})
        if i % 12 == 0:
            print(f"  z500 frame {i}/{len(STEPS)}", flush=True)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"aifs_z500": {
               "label": "AIFS single vs AIFS-ENS control — 500 hPa height (NH)",
               "n_frames": len(frames), "frames": frames}}}
    MANIFEST_Z.write_text(json.dumps(man))
    print(f"wrote {len(frames)} z500 frames + manifest")


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
