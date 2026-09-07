#!/usr/bin/env python3
"""County-mean monthly dewpoint from PRISM (tdmean, 4 km, CONUS, 1895 → last month).

nClimDiv has no humidity variable, so dewpoint comes from PRISM's station-based monthly mean
dewpoint grids (https://data.prism.oregonstate.edu/time_series/us/an/4km/tdmean/monthly/).
Once, the county polygons (US Atlas counties-10m, lon/lat) are rasterised onto the PRISM grid →
prism_county_mask.npz (committed, ~1 MB). Each run appends the months missing from the archive
(and re-reads the last eight, which PRISM still revises) and writes:

  assets/climate/anim/data/tdmean_county.bin    Int16, °F × 100, [county][month], −32768 = missing
  assets/climate/anim/data/tdmean_index.json    fips list, first month, number of months, cell counts

The archive lives on the frames branch with the rest of the climate data; a copy of the initial
backfill is kept in scripts/climate/trends/seed/ so a fresh runner does not need 4 GB of PRISM.

    python scripts/climate/trends/prism_tdmean.py --build-mask      (once, needs rasterio+shapely)
    python scripts/climate/trends/prism_tdmean.py --update          (append missing months)
    python scripts/climate/trends/prism_tdmean.py --backfill        (everything from 1895)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[2]))
OUT = SITE / "assets" / "climate" / "anim" / "data"
SEED = HERE / "seed"
MASK = HERE / "prism_county_mask.npz"
CACHE = HERE.parent / "data" / "prism"
BASE = "https://data.prism.oregonstate.edu/time_series/us/an/4km/tdmean/monthly/"
ATLAS = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"
UA = {"User-Agent": "Mozilla/5.0 (scorvec.com climate trends; contact: site owner)"}
FIRST = (1895, 1)
REVISE_MONTHS = 8


def _get(url: str, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as e:                                          # noqa: BLE001
            last = e; time.sleep(5 * (k + 1))
    raise last


def month_tif(y: int, m: int) -> tuple[np.ndarray, dict] | None:
    """(array °C with NaN, profile) for one month; cached zip in scripts/climate/data/prism/."""
    import rasterio
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"prism_tdmean_us_25m_{y}{m:02d}.zip"
    p = CACHE / name
    if not p.exists():
        try:
            data = _get(f"{BASE}{y}/{name}")
        except Exception as e:                                          # noqa: BLE001
            print(f"    {y}-{m:02d}: not available ({str(e)[:50]})", flush=True); return None
        p.write_bytes(data)
    with zipfile.ZipFile(p) as z:
        tif = [n for n in z.namelist() if n.endswith(".tif")][0]
        with rasterio.open(io.BytesIO(z.read(tif))) as ds:
            a = ds.read(1).astype("float32"); nd = ds.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[a < -90] = np.nan
            prof = {"transform": ds.transform, "width": ds.width, "height": ds.height, "crs": str(ds.crs)}
    return a, prof



def build_mask_real() -> None:
    import rasterio.features
    from shapely.geometry import shape
    a, prof = month_tif(2020, 1)
    topo = json.loads(_get(ATLAS))
    counties = _topo_features(topo, "counties")
    shapes = [(shape(f["geometry"]), i + 1) for i, f in enumerate(counties) if f["geometry"] is not None]
    grid = rasterio.features.rasterize(shapes, out_shape=(prof["height"], prof["width"]), transform=prof["transform"], fill=0, dtype="int32", all_touched=False)
    fips = [f["id"] for f in counties]
    valid = np.isfinite(a)
    counts = np.bincount(grid[valid], minlength=len(fips) + 1)[1:]
    np.savez_compressed(MASK, grid=grid.astype("int16"), fips=np.array(fips), counts=counts.astype("int32"), width=prof["width"], height=prof["height"])
    print(f"  mask: {len(fips)} counties, {int((counts > 0).sum())} with PRISM cells (CONUS), grid {prof['height']}×{prof['width']}", flush=True)


def _topo_features(topo: dict, name: str) -> list:
    """Minimal TopoJSON → GeoJSON features (arcs with delta-encoding and transform), enough for us-atlas."""
    tr = topo.get("transform"); scale = tr["scale"] if tr else (1, 1); trans = tr["translate"] if tr else (0, 0)
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0; pts = []
        for dx, dy in arc:
            if tr:
                x += dx; y += dy; pts.append((x * scale[0] + trans[0], y * scale[1] + trans[1]))
            else:
                pts.append((dx, dy))
        arcs.append(pts)
    def ring(idx):
        out = []
        for i in idx:
            pts = arcs[~i][::-1] if i < 0 else arcs[i]
            if out and out[-1] == pts[0]:
                pts = pts[1:]
            out.extend(pts)
        if out and out[0] != out[-1]:
            out.append(out[0])
        return out if len(out) >= 4 else None
    def polygon(rings):
        rr = [r for r in (ring(x) for x in rings) if r]
        return rr if rr else None
    feats = []
    for g in topo["objects"][name]["geometries"]:
        t = g["type"]
        if t == "Polygon":
            pr = polygon(g["arcs"]); geom = {"type": "Polygon", "coordinates": pr} if pr else None
        elif t == "MultiPolygon":
            polys = [pp for pp in (polygon(poly) for poly in g["arcs"]) if pp]
            geom = {"type": "MultiPolygon", "coordinates": polys} if polys else None
        else:
            geom = None
        feats.append({"id": g["id"], "geometry": geom, "properties": g.get("properties", {})})
    return feats


def load_archive() -> tuple[np.ndarray | None, dict | None]:
    for d in (OUT, SEED):
        b, j = d / "tdmean_county.bin", d / "tdmean_index.json"
        if b.exists() and j.exists():
            idx = json.loads(j.read_text())
            arr = np.fromfile(b, dtype="<i2").reshape(len(idx["fips"]), idx["n_months"])
            return arr, idx
    return None, None


def save_archive(arr: np.ndarray, idx: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arr.astype("<i2").tofile(OUT / "tdmean_county.bin"); (OUT / "tdmean_index.json").write_text(json.dumps(idx, separators=(",", ":")))
    SEED.mkdir(exist_ok=True)
    shutil.copy(OUT / "tdmean_county.bin", SEED / "tdmean_county.bin"); shutil.copy(OUT / "tdmean_index.json", SEED / "tdmean_index.json")


def county_means(a: np.ndarray, grid: np.ndarray, n: int) -> np.ndarray:
    valid = np.isfinite(a) & (grid > 0)
    s = np.bincount(grid[valid], weights=a[valid], minlength=n + 1)[1:]
    c = np.bincount(grid[valid], minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def update(backfill: bool) -> None:
    m = np.load(MASK, allow_pickle=False); grid = m["grid"].astype("int32"); fips = [str(f) for f in m["fips"]]; n = len(fips)
    arr, idx = load_archive()
    now = time.gmtime(); last_target = (now.tm_year, now.tm_mon - 1) if now.tm_mon > 1 else (now.tm_year - 1, 12)
    total = (last_target[0] - FIRST[0]) * 12 + (last_target[1] - FIRST[1]) + 1
    if arr is None or backfill or idx["fips"] != fips:
        arr = np.full((n, total), -32768, dtype="<i2"); idx = {"fips": fips, "first": f"{FIRST[0]}-{FIRST[1]:02d}", "n_months": total, "counts": m["counts"].tolist(), "units": "degF x100"}
        have = set()
    else:
        if idx["n_months"] < total:
            arr = np.concatenate([arr, np.full((n, total - idx["n_months"]), -32768, dtype="<i2")], axis=1); idx["n_months"] = total
        have = {k for k in range(total) if (arr[:, k] != -32768).any()}
    revise = set(range(max(0, total - REVISE_MONTHS), total))
    todo = [k for k in range(total) if k not in have or k in revise]
    print(f"  archive {n} counties × {total} months; {len(todo)} month(s) to read", flush=True)
    t0 = time.time(); done = 0
    # prefetch the zips in parallel (the reads are fast; the downloads were the whole wait)
    from concurrent.futures import ThreadPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    def _fetch(k):
        y, mo = FIRST[0] + (FIRST[1] - 1 + k) // 12, (FIRST[1] - 1 + k) % 12 + 1
        name = f"prism_tdmean_us_25m_{y}{mo:02d}.zip"; p = CACHE / name
        if p.exists() and k not in revise:
            return
        try:
            p.write_bytes(_get(f"{BASE}{y}/{name}"))
        except Exception as e:                                          # noqa: BLE001
            print(f"    {y}-{mo:02d}: download failed ({str(e)[:40]})", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, _ in enumerate(ex.map(_fetch, todo)):
            if i % 200 == 0:
                print(f"    prefetch {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    for k in todo:
        y, mo = FIRST[0] + (FIRST[1] - 1 + k) // 12, (FIRST[1] - 1 + k) % 12 + 1
        r = month_tif(y, mo)
        if r is None:
            continue
        a, _ = r
        f = county_means(a, grid, n) * 9 / 5 + 32.0
        arr[:, k] = np.where(np.isfinite(f), np.round(f * 100), -32768).astype("<i2")
        done += 1
        if done % 120 == 0:
            print(f"    {y}-{mo:02d} ({done}/{len(todo)}, {time.time() - t0:.0f}s)", flush=True)
    idx["last"] = f"{last_target[0]}-{last_target[1]:02d}"; idx["updated"] = time.strftime("%Y-%m-%d", now)
    save_archive(arr, idx)
    print(f"  wrote tdmean archive ({done} months read, {time.time() - t0:.0f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-mask", action="store_true"); ap.add_argument("--update", action="store_true"); ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    if a.build_mask or not MASK.exists():
        build_mask_real()
    if a.update or a.backfill:
        update(backfill=a.backfill)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
