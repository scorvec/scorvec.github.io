#!/usr/bin/env python3
"""The land fraction of every cell of the verification's 0.5 deg common grid, for the rain maps (mverify_scores.rain_map),
which test only cells that are mostly land: over open water and the Great Lakes Stage IV is radar alone, far from the
gauges that correct it, and misses light rain at range, so a model's "too many wet days" there is partly the
analysis's.

Natural Earth's 1:50m land less its lakes (public domain, through cartopy), sampled 5 x 5 in each cell. Cell centres
are on whole and half degrees, as the common grid's. Built once, offline:

    python build_land_mask.py     # -> scripts/verify/reference/land_0p5.npz: lat (361), lon (720, from 180W), land (%)
"""

from __future__ import annotations

from pathlib import Path

import cartopy.io.shapereader as shpreader
import numpy as np
import shapely
from shapely.ops import unary_union

OUT = Path(__file__).resolve().parent / "reference" / "land_0p5.npz"
STEP, SUB = 0.5, 5


def geometry(name: str):
    return unary_union(list(shpreader.Reader(shpreader.natural_earth("50m", "physical", name)).geometries()))


def main() -> None:
    land = shapely.difference(geometry("land"), geometry("lakes"))
    shapely.prepare(land)
    lat = np.arange(-90.0, 90.0 + STEP / 2, STEP)
    lon = np.arange(-180.0, 180.0, STEP)
    off = (np.arange(SUB) + 0.5) / SUB * STEP - STEP / 2  # sample points across a cell
    frac = np.zeros((lat.size, lon.size), np.uint8)
    for j, la in enumerate(lat):
        y = np.clip(la + off, -90, 90)
        xx, yy = np.meshgrid(lon[:, None] + off[None, :], y, indexing="xy")  # (SUB, lon x SUB)
        inside = shapely.contains_xy(land, xx.ravel(), yy.ravel()).reshape(SUB, lon.size, SUB)
        frac[j] = np.rint(100 * inside.mean(axis=(0, 2))).astype(np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, lat=lat, lon=lon, land=frac)
    print(f"{OUT}: {int((frac >= 50).sum())} of {frac.size} cells at least half land")


if __name__ == "__main__":
    main()
