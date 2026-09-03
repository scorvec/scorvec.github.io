#!/usr/bin/env python3
"""
Render the gridded ECAPE fields produced by ecape_grid.

Three panels are worth showing, and the third is the point of the whole exercise:

  ecape_ml / ecape_mu   entraining CAPE for the mixed-layer and most-unstable
                        parcels - what a real updraft of finite width actually
                        gets, rather than the undiluted textbook value.
  ratio                 ECAPE / CAPE. Nobody publishes this, and it is the field
                        that carries the new information: two places with the
                        same 3000 J/kg of CAPE can hand an updraft very different
                        amounts of usable buoyancy depending on how dry the
                        surroundings are and how strong the storm-relative flow
                        is. Low ratio = entrainment-hostile.

Usage:
  python render_ecape.py <stem> --outdir assets/ecape [--field ecape_ml] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PC = ccrs.PlateCarree()

# CAPE-like fields share the SPC-ish stepped palette so ECAPE and CAPE can be
# compared by eye across panels; the ratio gets its own sequential scale.
CAPE_LEVELS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
# One colour per bin PLUS one for the over-range wedge that extend="max" adds.
CAPE_COLORS = ["#ffffff", "#e8f4ea", "#c9e8c9", "#9fd89f", "#ffe9a8", "#ffc95c",
               "#ff9a3c", "#f4633a", "#dc3c2e", "#b81f36", "#8c1046", "#59062c"]
# The ratio is NOT capped at 1. Peters' formulation nets a storm-relative
# kinetic-energy gain against the entrainment loss, so ratio > 1 means vigorous
# inflow more than pays for the mixing (see skewt/methodology.html). That regime
# is the whole point, so it gets its own hot band above 1.0 rather than being
# flattened into the top of a 0-1 ramp.
RATIO_LEVELS = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
RATIO_COLORS = ["#3b1f4e", "#43307d", "#3c5a9a", "#2f7fa0", "#2f9e8f", "#57b56d",
                "#9ac64f", "#d8d13f", "#f4e07a",            # 0.9-1.0
                "#ff9a3c", "#e8503a", "#a8143c", "#6d0026"]  # >1: inflow wins

FIELD_META = {
    "ecape_ml": dict(title="ECAPE — mixed-layer parcel",
                     cbar="ECAPE (J kg$^{-1}$)", kind="cape"),
    "ecape_mu": dict(title="ECAPE — most-unstable parcel",
                     cbar="ECAPE (J kg$^{-1}$)", kind="cape"),
    "mlcape":   dict(title="MLCAPE — undiluted", cbar="CAPE (J kg$^{-1}$)", kind="cape"),
    "mucape":   dict(title="MUCAPE — undiluted", cbar="CAPE (J kg$^{-1}$)", kind="cape"),
    "ratio_ml": dict(title="ECAPE / CAPE — mixed-layer parcel",
                     cbar="fraction of CAPE retained", kind="ratio"),
    "ratio_mu": dict(title="ECAPE / CAPE — most-unstable parcel",
                     cbar="fraction of CAPE retained", kind="ratio"),
}


def hrrr_crs(g):
    """HRRR's native Lambert conformal, from the GRIB grid definition."""
    return ccrs.LambertConformal(
        central_longitude=g["lov"], central_latitude=g["latin1"],
        standard_parallels=(g["latin1"], g["latin2"]),
        globe=ccrs.Globe(ellipse="sphere", semimajor_axis=6371229.0,
                         semiminor_axis=6371229.0))


def load(stem: Path):
    meta = json.loads(stem.with_suffix(".json").read_text())
    ec_meta = json.loads(Path(str(stem) + "_ecape.json").read_text())
    nf, ny, nx = ec_meta["shape"]
    arr = np.fromfile(str(stem) + "_ecape.f32", dtype=np.float32).reshape(nf, ny, nx)
    fields = {n: arr[i] for i, n in enumerate(ec_meta["fields"])}
    # Ratio is only meaningful where there is buoyancy to lose; below ~100 J/kg
    # it is the quotient of two near-zero numbers and pure noise, so mask it.
    for tag, e, c in (("ml", "ecape_ml", "mlcape"), ("mu", "ecape_mu", "mucape")):
        r = np.full_like(fields[e], np.nan)
        ok = fields[c] > 100.0
        r[ok] = fields[e][ok] / fields[c][ok]
        fields[f"ratio_{tag}"] = r
    return meta, ec_meta, fields


def render(field, name, meta, out_path: Path):
    g = meta["grid"]
    ny, nx = g["ny"], g["nx"]
    fm = FIELD_META[name]
    crs = hrrr_crs(g)

    # Grid coordinates in the native projection: HRRR is a regular Lambert grid,
    # so build x/y from the corner and spacing rather than carrying 2-D lat/lon.
    x0, y0 = crs.transform_point(g["lon1"], g["lat1"], PC)
    x = x0 + np.arange(nx) * g["dx"]
    y = y0 + np.arange(ny) * g["dy"]

    # Panel sized to the domain aspect so there is no letterboxing.
    aspect = (nx * g["dx"]) / (ny * g["dy"])
    W, L, R, T, B = 12.0, 0.015, 0.885, 0.90, 0.035
    ax_h = (R - L) * W / aspect
    fig = plt.figure(figsize=(W, ax_h / (T - B)), dpi=115)
    gs = fig.add_gridspec(1, 1, left=L, right=R, top=T, bottom=B)
    ax = fig.add_subplot(gs[0], projection=crs)
    ax.set_extent([x[0], x[-1], y[0], y[-1]], crs=crs)

    if fm["kind"] == "cape":
        cmap = mcolors.ListedColormap(CAPE_COLORS)
        norm = mcolors.BoundaryNorm(CAPE_LEVELS, cmap.N, extend="max")
        im = ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading="auto",
                           transform=crs, rasterized=True, zorder=1)
    else:
        cmap = mcolors.ListedColormap(RATIO_COLORS)
        norm = mcolors.BoundaryNorm(RATIO_LEVELS, cmap.N, extend="max")
        im = ax.pcolormesh(x, y, np.ma.masked_invalid(field), cmap=cmap, norm=norm,
                           shading="auto", transform=crs, rasterized=True, zorder=1)

    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#5a5a5a",
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#333",
                   linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#333",
                   linewidth=0.6, zorder=3)

    c = meta["cycle"]
    from datetime import datetime, timedelta
    init = datetime(int(c["date"][:4]), int(c["date"][4:6]), int(c["date"][6:]), int(c["hour"]))
    fxx = int(c.get("fxx", 0))
    valid = init + timedelta(hours=fxx)
    stamp = f"HRRR {init:%Y-%m-%d %H}Z"
    stamp += f" F{fxx:02d}" if fxx else " analysis"
    stamp += f" · valid {valid:%a %b %d %H}Z"
    fig.suptitle(f"{fm['title']} — {stamp}", fontsize=13, fontweight="bold",
                 x=L, ha="left")
    ax.set_title("SHARPlib · Peters et al. (2023) entraining CAPE · HRRR 3 km native levels",
                 fontsize=8.5, loc="left", color="#555")

    cax = fig.add_axes([R + 0.012, B, 0.013, T - B])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(fm["cbar"], fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115, facecolor="white",
                pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)
    return out_path


def _render_one(args):
    """Pool worker. Re-loads from the scratch rather than receiving the array:
    the four panels together are ~30 MB of float32, and pickling that to every
    worker costs more than the memmap read it replaces."""
    stem, name, dest = args
    meta, _, fields = load(Path(stem))
    render(fields[name], name, meta, Path(dest))
    d = fields[name]
    finite = d[np.isfinite(d)]
    return name, dest, float(finite.max()), float(np.percentile(finite, 99))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--outdir", default="assets/ecape")
    ap.add_argument("--field", action="append",
                    help="field to render (repeatable); default: the four headline ones")
    ap.add_argument("--all", action="store_true", help="render every field")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel panel renders (1 disables the pool)")
    ap.add_argument("--anim-root",
                    help="write animation frames to <root>/<field>/F<fxx>.webp "
                         "instead of flat <outdir>/<field>.webp")
    a = ap.parse_args(argv)

    stem = Path(a.stem)
    meta, ec_meta, fields = load(stem)
    if a.all:
        names = list(FIELD_META)
    elif a.field:
        names = a.field
    else:
        # ratio_ml dropped 2026-08-29: the MU ratio is the one that gets read,
        # and the ML ratio tracked it closely enough not to earn its own panel.
        names = ["ecape_ml", "ecape_mu", "ratio_mu"]

    # Forecast hours share one output directory, so anything past the analysis
    # gets an _fNN suffix - otherwise a multi-hour run silently overwrites
    # itself and the page shows whichever hour happened to render last.
    fxx = int(meta["cycle"].get("fxx", 0))
    suffix = "" if fxx == 0 else f"_f{fxx:02d}"
    outdir = Path(a.outdir)
    anim_root = Path(a.anim_root) if a.anim_root else None
    jobs = []
    for n in names:
        if n not in fields:
            print(f"  skip {n}: not in output", file=sys.stderr)
            continue
        dest = (anim_root / n / f"F{fxx:02d}.webp") if anim_root \
            else (outdir / f"{n}{suffix}.webp")
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(stem), n, str(dest)))

    # The panels are independent cartopy renders, so run them in a pool. Serial
    # they are ~5 s for four; this is the last easy second to take off the
    # per-frame critical path.
    if len(jobs) > 1 and a.jobs != 1:
        # Spawn rather than Linux's default fork. This was a guess at the cause
        # of `struct.error: unpack requires a buffer of N bytes` on a GitHub
        # runner, and it did NOT fix it - the pool still dies there, just with a
        # different N. The real cause is unresolved. Left as spawn because it
        # matches the start method this is actually tested under on macOS.
        #
        # Not worth chasing further: the fallback below costs ~3 s a frame
        # against a 135 s kernel on that runner (~2%), and the pool works fine
        # locally, which is where the 1.7x actually matters.
        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=min(len(jobs), a.jobs or 4),
                                     mp_context=ctx) as ex:
                results = list(ex.map(_render_one, jobs))
        except Exception as e:                                      # noqa: BLE001
            # Rendering serially is ~2 s slower per frame. Losing the frame
            # entirely is worse, so degrade rather than fail the run.
            print(f"  parallel render failed ({type(e).__name__}: {e}); "
                  f"falling back to serial", file=sys.stderr, flush=True)
            results = [_render_one(j) for j in jobs]
    else:
        results = [_render_one(j) for j in jobs]
    for name, dest, mx, p99 in results:
        print(f"  {name:9s} -> {dest}  (max {mx:.2f}, p99 {p99:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
