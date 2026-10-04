#!/usr/bin/env python3
"""Monthly 500 hPa geopotential height for the CMIP6 ENSO-impact members, Northern Hemisphere (2026-09-27; user: "can you
also show 500mb height anomalies (actually just a show a flat northern hemisphere view)"). LAPTOP job; streams the public
Pangeo CMIP6 zarr stores (Google Cloud, anonymous), nothing raw is kept.

COST. Every Pangeo Amon zg store chunks all 19 pressure levels (and the whole globe) together, so one level costs the
full 3-D read: 366 GB compressed for the 397 study members that have Amon zg (32 have none; no AERday zg500 exists for
any). The default therefore takes at most MAX_PER_MODEL members per model (167 GB): the tests count each model once, so
10 members give each model's mean as well as 65 would; `--max-per-model 0` takes them all (resumable, adds the rest).

Per member, Amon zg at 500 hPa, 1949-12 .. 2014-12 (781 months, so DJF 1950 is complete): area-overlap mean onto a
2.5-degree grid 0-90N x 0-357.5E (centres), metres, int16 (z - 5000 m, 1 m steps). Output
data/cmip6_enso/z500/<source>_<member>.npz (resumable).

    python scripts/sst/cmip6_z500_extract.py [--workers 2] [--max-per-model 10] [--only MODEL] [--limit N]
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
OUT = SRC / "z500"
CAT = HERE.parent / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"
SO = {"token": "anon"}
LAT = np.arange(0.0, 90.01, 2.5)
LON = np.arange(0.0, 357.51, 2.5)
T0, T1 = "1949-12-01", "2014-12-31"
NMON = 781
MAX_PER_MODEL = 10
_OPS = {}


def _bounds(c, periodic=False):
    m = 0.5 * (c[1:] + c[:-1])
    lo = c[0] - (m[0] - c[0]) if not periodic else c[0] - 0.5 * ((c[0] - c[-1]) % 360)
    hi = c[-1] + (c[-1] - m[-1]) if not periodic else c[-1] + 0.5 * ((c[0] - c[-1]) % 360)
    return np.r_[lo, m, hi]


def _overlap(src_edges, dst_edges, periodic=False):
    O = np.zeros((len(dst_edges) - 1, len(src_edges) - 1))
    for sh in ((-360.0, 0.0, 360.0) if periodic else (0.0,)):
        s0, s1 = src_edges[:-1] + sh, src_edges[1:] + sh
        O += np.clip(np.minimum(dst_edges[1:, None], s1[None]) - np.maximum(dst_edges[:-1, None], s0[None]), 0, None)
    return O


def ops(lat, lon):
    """Separable area-overlap operators (native -> 2.5 deg), row-normalised."""
    key = (len(lat), len(lon), float(lat[0]), float(lon[0]))
    if key not in _OPS:
        la_e = np.clip(_bounds(lat), -90, 90); lo_e = _bounds(lon, periodic=True)
        dst_la = np.clip(np.r_[LAT - 1.25, LAT[-1] + 1.25], -90, 90)
        A = _overlap(np.sin(np.deg2rad(la_e)), np.sin(np.deg2rad(dst_la)))
        B = _overlap(lo_e, np.r_[LON - 1.25, LON[-1] + 1.25], periodic=True)
        _OPS[key] = (A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True))
    return _OPS[key]


def one(args):
    src, member, store = args
    dest = OUT / f"{src}_{member}.npz"
    if dest.exists():
        return f"{src} {member}: have", 0.0
    t0 = time.time(); err = ""
    for attempt in range(3):
        try:
            ds = xr.open_zarr(store, storage_options=SO, consolidated=True)
            v = ds["zg"].sel(time=slice(T0, T1))
            plev = v.plev.values
            k = int(np.argmin(np.abs(plev - (50000.0 if plev.max() > 2000 else 500.0))))
            if abs(plev[k] - (50000.0 if plev.max() > 2000 else 500.0)) > 1:
                raise ValueError(f"no 500 hPa level: {plev}")
            v = v.isel(plev=k)
            if "lon" in v.coords:
                v = v.assign_coords(lon=v.lon % 360).sortby("lon")
            v = v.sortby("lat").transpose("time", "lat", "lon")
            months = [f"{t.year:04d}-{t.month:02d}" for t in pd.to_datetime([str(t)[:10] for t in v.time.values])]
            if len(months) != NMON:
                raise ValueError(f"{len(months)} months")
            x = v.values.astype("float32")
            A, B = ops(v.lat.values.astype(float), v.lon.values.astype(float))
            box = np.einsum("il,tlm,jm->tij", A, x, B, optimize=True)
            q = np.round(box - 5000.0).clip(-32767, 32767).astype("int16")
            tmp = dest.with_suffix(".part.npz")
            np.savez_compressed(tmp, z=q, months=np.array(months), lat=LAT, lon=LON, units=str(ds["zg"].attrs.get("units", "")),
                                source_id=src, member_id=member)
            Path(str(tmp)).replace(dest)
            return f"{src} {member}: {time.time() - t0:.0f}s native {v.sizes['lat']}x{v.sizes['lon']} mean {float(box.mean()):.0f} m", time.time() - t0
        except Exception as e:                                             # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"; time.sleep(10)
    return f"{src} {member}: FAILED {err}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-model", type=int, default=MAX_PER_MODEL)
    ap.add_argument("--only"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = sorted(Path(f).stem for f in glob.glob(str(SRC / "*.nc")) if ".part" not in f)
    cat = pd.read_csv(CAT)
    c = cat[(cat.table_id == "Amon") & (cat.experiment_id == "historical") & (cat.variable_id == "zg")]
    c = c.sort_values("version").groupby(["source_id", "member_id"]).zstore.last()
    per, jobs = {}, []
    for name in want:                       # sorted, so the first members of each model (r1, r10, r11 ...) are taken
        src, member = name.rsplit("_", 1)
        if (a.only and src != a.only) or (src, member) not in c.index:
            continue
        if a.max_per_model and per.get(src, 0) >= a.max_per_model:
            continue
        per[src] = per.get(src, 0) + 1
        jobs.append((src, member, c.loc[(src, member)]))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(f"{len(jobs)} members, {len(per)} models ({a.workers} workers): {per}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (r, _) in enumerate(ex.map(one, jobs)):
            if (i + 1) % 5 == 0 or "FAILED" in r or i < 3:
                print(f"[{i + 1}/{len(jobs)}] {r}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
