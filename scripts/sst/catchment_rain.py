#!/usr/bin/env python3
"""Per-catchment basin rain — the spatial detail a basin mean throws away.

A region's rain has been reduced to one energy-weighted mean per basin.
Over Colombian terrain that is lossy: CENTRO's SOGAMOSO (Santander) and
EL QUIMBO (Huila, ~600 km south) carry 32% and 24% of the region's
generation energy and their daily rain correlates just **0.212**.
Averaging them merges two independent weather regimes into one number,
and no amount of model flexibility recovers it — gradient boosting on the
basin mean made CENTRO *worse* (−0.022) while decomposing into
catchments made it *better* (+0.024).

ANTIOQUIA is the case that matters: 15 catchments, 49% of national
inflow energy, split across the Cauca and Magdalena valleys with the
Central Cordillera between them.

This builds and caches, for every catchment with meaningful generation:
  * daily gauge-corrected basin rain (the same blended truth the rest of
    the stack calibrates on)
  * the matching harmonic climatology, so anomalies are per-catchment
    rather than against a regional mean

One pass over the daily archive, all catchments at once — the cost is
reading the grids, not the dot products.

    python scripts/sst/catchment_rain.py [--min-energy 0.3]

Output: ~/colombia_hydro/raw/catchment_rain.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import (CATCH_GJ, gauge_correction,          # noqa: E402
                               gauge_blend_field, _river_energy,
                               _regulated_rivers)
from build_imerg_clim import OUT as CLIM_NC, eval_clim             # noqa: E402
from matplotlib.path import Path as MplPath                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
CACHE = PRIV / "raw" / "catchment_rain.npz"

# CO_CATCH_GJ swaps the catchment geometry without touching any caller:
# the default is the IDEAM subzona (SZH) union set, and
# xm_river_catchments_traced.geojson is the HydroBASINS lev-12 upstream
# trace built by delineate_catchments.py. Set CO_CATCH_CACHE alongside it
# so the two geometries do not overwrite each other's cache.
_GJ_OVERRIDE = os.environ.get("CO_CATCH_GJ")
if _GJ_OVERRIDE:
    CATCH_GJ = Path(_GJ_OVERRIDE)
CACHE = Path(os.environ.get("CO_CATCH_CACHE", CACHE))


def build_masks(lons, lats, min_energy=0.3, include_regulated=False):
    """{(region, river): flat cos-lat weight vector} for catchments worth using.

    Regulated rivers are excluded by default for the same reason they are
    excluded from the region weighting: their inflow is a release
    decision, not a rainfall response. Tiny catchments are dropped
    because a mask of two or three cells is mostly noise.
    """
    egy, reg = _river_energy(), _regulated_rivers()
    gj = json.loads(Path(CATCH_GJ).read_text())
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel() % 360, LA.ravel()])
    cos = np.cos(np.deg2rad(LA))
    out, meta = {}, {}
    for ft in gj["features"]:
        pr = ft["properties"]
        rg, riv = pr.get("region"), pr.get("river")
        if not rg or not riv:
            continue
        if riv in reg and not include_regulated:
            continue
        e = float(egy.get(riv, 0.0))
        if e < min_energy:
            continue
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        m = np.zeros(LO.shape, bool)
        for pl in polys:
            ring = np.array(pl[0])
            if ring.ndim != 2 or len(ring) < 4:
                continue
            inside = MplPath(np.column_stack([ring[:, 0] % 360, ring[:, 1]])
                             ).contains_points(pts).reshape(LO.shape)
            # interior rings are real holes, not decoration: an INCREMENTAL
            # traced catchment is its full upstream area minus the nested
            # catchments below it, and those subtractions arrive as holes.
            # Ignoring them would hand PORCE III its whole 3,768 km2 back
            # and re-create the collinearity the trace exists to remove.
            for hole in pl[1:]:
                h = np.array(hole)
                if h.ndim == 2 and len(h) >= 4:
                    inside &= ~MplPath(
                        np.column_stack([h[:, 0] % 360, h[:, 1]])
                    ).contains_points(pts).reshape(LO.shape)
            m |= inside
        w = np.where(m, cos, 0.0)
        if w.sum() <= 0:
            continue
        key = f"{rg}|{riv}"
        if key in out:                       # merge duplicate features
            w = w + out[key].reshape(LO.shape) * meta[key]["cells"]
        out[key] = (w / w.sum()).ravel()
        meta[key] = {"region": rg, "river": riv, "energy_gwh": round(e, 3),
                     "cells": int((w > 0).sum())}
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-energy", type=float, default=0.3,
                    help="GWh/day floor; below this a catchment mask is "
                         "a handful of cells and mostly noise")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    if CACHE.exists() and not a.force:
        z = np.load(CACHE, allow_pickle=True)
        if int(z["nfiles"]) == len(files):
            m = z["meta"].item()
            print(f"cache current: {len(m)} catchments, {len(files)} days")
            return 0

    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    F = gauge_correction(lons, lats)
    W, meta = build_masks(lons, lats, a.min_energy)
    byreg = {}
    for k, v in meta.items():
        byreg.setdefault(v["region"], []).append(v)
    print(f"{len(W)} catchments >= {a.min_energy} GWh/day:")
    for r in sorted(byreg):
        tot = sum(x["energy_gwh"] for x in byreg[r])
        print(f"   {r:11} {len(byreg[r]):3d} catchments  {tot:6.1f} GWh/d  "
              f"cells {min(x['cells'] for x in byreg[r])}-"
              f"{max(x['cells'] for x in byreg[r])}")

    import xarray as xr
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    keys = sorted(W)
    Wm = np.stack([W[k] for k in keys])                 # (ncatch, ncell)
    cc = {}
    dates, R, C = [], [], []
    for i, f in enumerate(files):
        g = gauge_blend_field(np.load(f) * F, f.stem, lons, lats).ravel()
        doy = min(datetime.strptime(f.stem, "%Y%m%d").timetuple().tm_yday, 365)
        if doy not in cc:
            cc[doy] = (eval_clim(coef, doy) * F).ravel()
        dates.append(f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}")
        R.append(Wm @ g)
        C.append(Wm @ cc[doy])
        if i % 2000 == 0:
            print(f"   {i}/{len(files)}", flush=True)
    R = np.asarray(R); C = np.asarray(C)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, dates=np.array(dates), keys=np.array(keys),
                        rain=R, clim=C, meta=np.array(meta, dtype=object),
                        nfiles=len(files))
    print(f"\nwrote {CACHE}  rain{R.shape}  ({R.nbytes/1e6:.0f} MB raw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
