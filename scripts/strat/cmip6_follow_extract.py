#!/usr/bin/env python3
"""Monthly 2 m temperature and sea-level pressure over 20-90N for every CMIP6 member of the u(60N) drips, for the CMIP6
"What usually follows" views (2026-09-27; user: "Will you also update 'what usually follows' with CMIP6?").

LAPTOP job, run once, resumable per member. Streams Amon tas and Amon psl from the public Pangeo CMIP6 zarr stores on
Google Cloud (anonymous), chunk by chunk, for exactly the members that have a daily u(60N) extract in
scripts/strat/data/cmip6 (build_cmip6_drip.files_for over its GOOD models), limited to the years that extract covers,
and keeps only an AREA-MEAN 2.5-degree grid over 20-90N (every native cell's cos(lat)-weighted value summed into the
2.5-degree box its centre falls in - not point sampling). Nothing raw is kept.

Why sea-level pressure and not 500 hPa height: Amon zg chunks carry all 19 levels, ~1.5 TB to stream for this sample;
psl is one level. The ERA5 maps contour 500 hPa height; the CMIP6 maps contour sea-level pressure, and say so.

Output: <data/cmip6>/follow/<model>_<experiment>_<member>.npz with
    tas  int16 (month, lat, lon)  (K - 260) x 100            -> 0.01 K
    psl  int16 (month, lat, lon)  (Pa - 101300)               -> 1 Pa
    year, month  int16 (month)     the model calendar's year and month
    lat, lon     box centres (21.25..88.75, 1.25..358.75)

    python scripts/strat/cmip6_follow_extract.py [--workers 6] [--jobs 2] [--only MODEL]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_cmip6_drip as B                                       # noqa: E402
import cmip6_strat_extract as X                                    # noqa: E402

OUT = B.DATA / "follow"
DLAT = DLON = 2.5
LAT_E = np.arange(20.0, 90.0 + 1e-6, DLAT)
LON_E = np.arange(0.0, 360.0 + 1e-6, DLON)
LAT_C = 0.5 * (LAT_E[1:] + LAT_E[:-1])
LON_C = 0.5 * (LON_E[1:] + LON_E[:-1])
SCALE = {"tas": (260.0, 100.0), "psl": (101300.0, 1.0)}             # value = int16 / scale + offset


def coarsen_matrix(lat, lon):
    """Sparse (boxes x native cells) matrix of cos(lat) weights, rows normalised: box mean = M @ field.ravel()."""
    la2, lo2 = np.meshgrid(lat, lon % 360.0, indexing="ij")
    ib = np.searchsorted(LAT_E, la2.ravel(), side="right") - 1
    jb = np.searchsorted(LON_E, lo2.ravel(), side="right") - 1
    ok = (la2.ravel() >= LAT_E[0]) & (ib >= 0) & (ib < len(LAT_C)) & (jb >= 0) & (jb < len(LON_C))
    w = np.cos(np.deg2rad(la2.ravel()))[ok]
    rows = (ib * len(LON_C) + jb)[ok]
    cols = np.arange(la2.size)[ok]
    M = sparse.csr_matrix((w, (rows, cols)), shape=(len(LAT_C) * len(LON_C), la2.size))
    s = np.asarray(M.sum(1)).ravel()
    empty = int((s == 0).sum())
    M = sparse.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)) @ M
    return M.tocsr(), empty


def reduce_var(zstore, var, y0, y1, workers):
    ds = xr.open_zarr(zstore, storage_options=X.SO, consolidated=True,
                      decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    v = ds[var]
    yrs = np.array([t.year for t in ds.time.values]); mon = np.array([t.month for t in ds.time.values])
    keep = np.where((yrs >= y0) & (yrs <= y1))[0]
    if not len(keep):
        return None
    lat, lon = v.lat.values, v.lon.values
    M, empty = coarsen_matrix(lat, lon)
    if empty:
        raise RuntimeError(f"{empty} empty 2.5-degree boxes: native grid too coarse")
    ct = (v.encoding.get("chunks") or (216,))[0]
    i0, i1 = int(keep[0]), int(keep[-1]) + 1
    starts = list(range(i0 - (i0 % ct), i1, ct))
    out = np.full((i1 - i0, len(LAT_C), len(LON_C)), np.nan, "float32")

    def one(s):
        a0, a1 = max(s, i0), min(s + ct, i1)
        for attempt in range(5):
            try:
                a = v.isel(time=slice(a0, a1)).values.astype("float32")
                a = np.where(np.abs(a) > 1e10, np.nan, a)
                n = a.shape[0]
                flat = a.reshape(n, -1)
                bad = ~np.isfinite(flat)
                if bad.any():                                    # a missing cell leaves its box NaN, never biased low
                    cnt = (M @ bad.T.astype("float32")).T
                    flat = np.where(bad, 0.0, flat)
                    r = (M @ flat.T).T
                    r[cnt > 0] = np.nan
                else:
                    r = (M @ flat.T).T
                return a0, r.reshape(n, len(LAT_C), len(LON_C)).astype("float32")
            except Exception as e:                               # noqa: BLE001
                err = e; time.sleep(5 * (attempt + 1))
        raise err
    with ThreadPoolExecutor(workers) as ex:
        for a0, r in ex.map(one, starts):
            out[a0 - i0:a0 - i0 + len(r)] = r
    return out, yrs[i0:i1], mon[i0:i1], v.encoding.get("dtype"), float(v.nbytes / v.shape[0])


def run_member(model, exp, member, f, cat, workers):
    dest = OUT / f"{model}_{exp}_{member}.npz"
    if dest.exists():
        return model, exp, member, "have", 0.0, 0.0
    t0 = time.time()
    Y = B.read_member(f)[0]
    y0, y1 = int(Y.min()), int(Y.max())
    res, nbytes = {}, 0.0
    for var in ("tas", "psl"):
        zs = X.store(cat, model, exp, member, "Amon", var)
        if zs is None:
            return model, exp, member, f"no Amon {var}", 0.0, time.time() - t0
        r = reduce_var(zs, var, y0, y1, workers)
        if r is None:
            return model, exp, member, f"Amon {var} has no months in {y0}-{y1}", 0.0, time.time() - t0
        res[var] = r; nbytes += r[4] * len(r[1])
    (a, ya, ma), (b, yb, mb) = res["tas"][:3], res["psl"][:3]
    ka = {(y, m): i for i, (y, m) in enumerate(zip(ya, ma))}
    common = [(i, ka[(y, m)]) for i, (y, m) in enumerate(zip(yb, mb)) if (y, m) in ka]
    ib, ia = (np.array(x) for x in zip(*common))
    enc = {}
    for var, arr in (("tas", a[ia]), ("psl", b[ib])):
        off, sc = SCALE[var]
        q = np.round((arr - off) * sc)
        enc[var] = np.where(np.isfinite(q), np.clip(q, -32767, 32767), -32768).astype("int16")
    tmp = dest.with_suffix(".part.npz")
    np.savez_compressed(tmp, tas=enc["tas"], psl=enc["psl"], year=ya[ia].astype("int16"), month=ma[ia].astype("int16"),
                        lat=LAT_C, lon=LON_C)
    tmp.replace(dest)
    return model, exp, member, f"{len(ia)} months {y0}-{y1}", nbytes, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cat = X.catalog()
    todo = [(m, exp, mem, f) for m in B.GOOD if not a.only or m == a.only for f, exp, mem in B.files_for(m)]
    print(f"{len(todo)} members", flush=True)
    t0 = time.time(); tot = 0.0; done = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_member, m, e, mem, f, cat, a.workers) for m, e, mem, f in todo]
        for fu in futs:
            try:
                model, exp, member, msg, nb, dt = fu.result()
            except Exception as e:                               # noqa: BLE001  keep going; a rerun resumes
                print(f"  FAILED: {str(e)[:200]}", flush=True); continue
            done += 1; tot += nb
            el = time.time() - t0
            print(f"  [{done}/{len(todo)}] {model} {exp} {member}: {msg}  {nb / 1e9:.2f} GB in {dt:.0f} s  "
                  f"(total {tot / 1e9:.1f} GB, {tot / 1e6 / max(el, 1):.0f} MB/s raw, {el / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
