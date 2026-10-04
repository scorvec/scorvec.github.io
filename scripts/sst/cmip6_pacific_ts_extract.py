#!/usr/bin/env python3
"""Monthly Pacific surface temperature for the CMIP6 ENSO-impact members (2026-09-27; user: "it would also be useful to
look at the impact of east-based events and of the PDO index as well (trying to separate out the ENSO parts as much as
possible)"). LAPTOP job; streams the public Pangeo CMIP6 zarr stores (Google Cloud, anonymous), nothing raw is kept.

For exactly the members of the ENSO-impact study (every data/cmip6_enso/<source>_<member>.nc; 16 models, 429 historical
members), Amon `ts` 1950-2014:
  ts   (month, lat, lon)  ocean surface temperature on a 2.5-degree grid, 30S-70N x 100E-70W (centres), area-mean
                          coarsened from the native grid (area-overlap weights, ocean cells only). Native cells whose centre
                          is land (Natural Earth 50 m) are dropped; sea ice (ts < 271.4 K) is NaN. A 2.5-degree box needs
                          >= 50 % of its native weight in ocean, else it is NaN (land). Stored int16, 0.01 K about 273.15.
  gm   (month)            60S-60N ocean-mean ts (ice excluded): the global mean the PDO definition removes.
Output: data/cmip6_enso/pacific_ts/<source>_<member>.npz (resumable: a finished file is skipped).

    python scripts/sst/cmip6_pacific_ts_extract.py [--workers 2] [--only MODEL]
"""
from __future__ import annotations

import argparse
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "cmip6_enso"
OUT = SRC / "pacific_ts"
CAT = HERE.parent / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"
SO = {"token": "anon"}
LAT = np.arange(-30.0, 70.01, 2.5)
LON = np.arange(100.0, 290.01, 2.5)
Y0, Y1 = 1950, 2014
ICE = 271.4
_LAND = None
_GRIDS = {}


def land_geom():
    global _LAND
    if _LAND is None:
        import cartopy.io.shapereader as shp
        import shapely
        from shapely.ops import unary_union
        _LAND = unary_union(list(shp.Reader(shp.natural_earth("50m", "physical", "land")).geometries()))
        shapely.prepare(_LAND)
    return _LAND


def _bounds(c, periodic=False):
    """Cell edges from centres (midpoints; the outer edges mirror the first/last half-spacing)."""
    m = 0.5 * (c[1:] + c[:-1])
    lo = c[0] - (m[0] - c[0]) if not periodic else c[0] - 0.5 * ((c[0] - c[-1]) % 360)
    hi = c[-1] + (c[-1] - m[-1]) if not periodic else c[-1] + 0.5 * ((c[0] - c[-1]) % 360)
    return np.r_[lo, m, hi]


def _overlap(src_edges, dst_edges, periodic=False):
    """(n_dst, n_src) overlap lengths of 1-D cells; longitudes are tried at -360/0/+360 shifts."""
    O = np.zeros((len(dst_edges) - 1, len(src_edges) - 1))
    for sh in ((-360.0, 0.0, 360.0) if periodic else (0.0,)):
        s0, s1 = src_edges[:-1] + sh, src_edges[1:] + sh
        O += np.clip(np.minimum(dst_edges[1:, None], s1[None]) - np.maximum(dst_edges[:-1, None], s0[None]), 0, None)
    return O


def grid_ops(lat, lon):
    """For a native grid: the sparse AREA-OVERLAP mean operator onto the 2.5-degree boxes (ocean cells only; works for
    native grids finer or coarser than 2.5 degrees), the boxes that are >= 50 % ocean, and the 60S-60N ocean weights.
    Cached per grid (members of a model share it)."""
    key = (len(lat), len(lon), float(lat[0]), float(lon[0]))
    if key in _GRIDS:
        return _GRIDS[key]
    import shapely
    lo180 = np.where(lon > 180, lon - 360, lon)
    X, Y = np.meshgrid(lo180, lat)
    ocean = ~shapely.contains_xy(land_geom(), X.ravel(), Y.ravel())
    la_e = np.clip(_bounds(lat), -90, 90); lo_e = _bounds(lon, periodic=True)
    Ala = _overlap(np.sin(np.deg2rad(la_e)), np.sin(np.deg2rad(np.r_[LAT - 1.25, LAT[-1] + 1.25])))   # area ~ d(sin lat)
    Alo = _overlap(lo_e, np.r_[LON - 1.25, LON[-1] + 1.25], periodic=True)
    M_all = sparse.kron(sparse.csr_matrix(Ala), sparse.csr_matrix(Alo)).tocsr()          # (boxes, native)
    M_oc = (M_all @ sparse.diags(ocean.astype(float))).tocsr()
    frac = np.asarray(M_oc.sum(1)).ravel() / np.maximum(np.asarray(M_all.sum(1)).ravel(), 1e-12)
    w = np.repeat(np.diff(np.sin(np.deg2rad(la_e))), len(lon))
    gw = w * ocean * (np.abs(Y.ravel()) <= 60)
    _GRIDS[key] = (ocean, M_oc, frac >= 0.5, gw)
    return _GRIDS[key]


def one(args):
    src, member, store = args
    dest = OUT / f"{src}_{member}.npz"
    if dest.exists():
        return f"{src} {member}: have"
    t0 = time.time()
    err = ""
    for attempt in range(3):
        try:
            ds = xr.open_zarr(store, storage_options=SO, consolidated=True)
            ds = ds.sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31"))
            if "lon" in ds.coords:
                ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon")
            v = ds["ts"].transpose("time", "lat", "lon")
            lat, lon = v.lat.values.astype(float), v.lon.values.astype(float)
            x = v.values.astype("float32")                                  # (time, lat, lon), the whole native field
            months = [f"{t.year:04d}-{t.month:02d}" for t in pd.to_datetime([str(t)[:10] for t in v.time.values])]
            if len(months) != (Y1 - Y0 + 1) * 12:
                raise ValueError(f"{len(months)} months")
            ocean, M, okbox, gw = grid_ops(lat, lon)
            X = x.reshape(len(months), -1)
            valid = np.isfinite(X) & (X >= ICE)
            Xf = np.where(valid, X, 0.0)
            num = (M @ Xf.T).T; den = (M @ valid.T.astype("float32")).T
            box = np.where((den > 0) & okbox[None, :], num / np.maximum(den, 1e-12), np.nan)
            wsum = valid @ gw
            gm = (Xf @ gw) / np.maximum(wsum, 1e-12)
            q = np.where(np.isfinite(box), np.round((box - 273.15) * 100), -32768).astype("int16")
            tmp = dest.with_suffix(".part.npz")
            np.savez_compressed(tmp, ts=q.reshape(len(months), len(LAT), len(LON)), gm=gm.astype("float32"),
                                months=np.array(months), lat=LAT, lon=LON, source_id=src, member_id=member)
            Path(str(tmp)).replace(dest)
            return f"{src} {member}: {time.time() - t0:.0f}s native {len(lat)}x{len(lon)}"
        except Exception as e:                                             # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"; time.sleep(10)
    return f"{src} {member}: FAILED {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = sorted(Path(f).stem for f in glob.glob(str(SRC / "*.nc")) if ".part" not in f)
    cat = pd.read_csv(CAT)
    c = cat[(cat.table_id == "Amon") & (cat.experiment_id == "historical") & (cat.variable_id == "ts")]
    c = c.sort_values("version").groupby(["source_id", "member_id"]).zstore.last()
    jobs = []
    for name in want:
        src, member = name.rsplit("_", 1)
        if a.only and src != a.only:
            continue
        if (src, member) not in c.index:
            print(f"{name}: no ts store", flush=True); continue
        jobs.append((src, member, c.loc[(src, member)]))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(f"{len(jobs)} members to extract ({a.workers} workers)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            if (i + 1) % 10 == 0 or "FAILED" in r or i < 3:
                print(f"[{i + 1}/{len(jobs)}] {r}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
