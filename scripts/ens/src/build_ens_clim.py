#!/usr/bin/env python3
"""ERA5 day-of-year NORMALS of z500 / t2m on the common North-America grid.
Periods: 30-yr (1991–2020) and 10-yr (2014–2023).

Method (no harmonics, no sub-sampling):
  1. Read EVERY day of the period (12Z) for the NA box and take the per-day-of-year
     mean over the years → raw daily normal (≈30 / ≈10 samples per calendar day).
     Reads run in a PROCESS pool (each worker decompresses independently — real
     parallelism, unlike a thread pool which the codec's GIL serialises).
  2. Fit a PERIODIC smoothing spline across day-of-year per grid point (GCV-chosen,
     year wrapped) to remove the residual day-to-day sampling noise.

    python src/build_ens_clim.py --var z500 --period 30yr --workers 8

Output: data/reference/ens_clim_<var>_<period>.nc — DataArray (dayofyear 1..366, lat, lon).
"""
from __future__ import annotations
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

sys.path.insert(0, str(Path(__file__).parent))
from common import REF, VARS, PERIODS, GRIDS, grid

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CHUNK = 50           # fields per worker task — small enough for fast first-signal /
                     # fine progress + checkpoint granularity, big enough that per-task
                     # zarr-open overhead stays small


def _accumulate(args):
    """Worker: read a slice of daily times, interp each to the common grid, and
    accumulate per-day-of-year sum/count. Returns (sum[366,lat,lon], count[366]).
    Appends ONE line per field to FIELD_LOG so progress is watchable in real time
    (O_APPEND line writes from each process are atomic)."""
    gkey, era5var, level, times, flog, store = args
    g = GRIDS[gkey]; tlat, tlon = g["lat"], g["lon"]
    wid = os.getpid()
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(store), chunks=None)
    da = ds[era5var]
    if level is not None:
        da = da.sel(level=level)
    ssum = np.zeros((366, len(tlat), len(tlon)), "float32"); scnt = np.zeros(366, "int32")
    for t in times:
        t0 = time.time()
        try:
            fld = da.sel(time=t)
            if float(fld.longitude.min()) < 0:               # -180..180 → 0..360
                fld = fld.assign_coords(longitude=(fld.longitude % 360)).sortby("longitude")
            fld = fld.sortby("latitude")                     # ascending for interp
            # close the 0/360 seam: append a wrapped column (lon = min+360) so interp
            # to a target reaching ~359.5 stays IN range (else extrapolates → NaN)
            wrap = fld.isel(longitude=0).assign_coords(longitude=float(fld.longitude[0]) + 360.0)
            fld = xr.concat([fld, wrap], dim="longitude")
            f = (fld.interp(latitude=tlat, longitude=tlon)
                    .transpose("latitude", "longitude").values.astype("float32"))
        except Exception as e:                               # noqa: BLE001
            with open(flog, "a") as lf:
                lf.write(f"{time.strftime('%H:%M:%S')} w{wid} {pd.Timestamp(t):%Y-%m-%d} FAIL {repr(e)[:50]}\n")
            continue
        d = int(pd.Timestamp(t).dayofyear)
        ssum[d - 1] += f; scnt[d - 1] += 1
        with open(flog, "a") as lf:
            lf.write(f"{time.strftime('%H:%M:%S')} w{wid} {pd.Timestamp(t):%Y-%m-%d} 12Z "
                     f"ok  {time.time()-t0:4.1f}s  {np.nanmin(f):.0f}-{np.nanmax(f):.0f}\n")
    return ssum, scnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, choices=list(VARS))
    ap.add_argument("--period", required=True, choices=list(PERIODS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    v = VARS[args.var]; y0, y1 = PERIODS[args.period]
    store = v["clim_store"]
    out = REF / f"ens_clim_{args.var}_{args.period}.nc"

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(store), chunks=None)
    want = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="1D") + pd.Timedelta(hours=12)
    era5_times = pd.to_datetime(ds.time.values)                        # convert once
    times = list(want[want.isin(era5_times)])
    print(f"reading EVERY day: {len(times)} ERA5 {args.var} fields ({y0}-{y1}), "
          f"{args.workers} processes …", flush=True)

    LAT, LON = grid(args.var)
    ntot = len(times)
    # Many small chunks (not one big group per worker) so as_completed gives a live
    # progress counter as each chunk lands — at the cost of a few extra zarr opens.
    # Each chunk has a stable index (times + CHUNK are deterministic), so we can
    # checkpoint the running sum + the set of finished chunk-ids and RESUME after a
    # crash/sleep/kill instead of re-downloading everything. Checkpoint lives in /tmp
    # (NOT the repo) and is deleted on success.
    all_chunks = [(i, times[j:j + CHUNK]) for i, j in enumerate(range(0, ntot, CHUNK))]
    ckpt = Path("/tmp") / f"ens_ckpt_{args.var}_{args.period}.npz"
    flog = Path("/tmp") / f"ens_fields_{args.var}_{args.period}.log"   # per-FIELD live log
    flog.write_text(f"# {args.var} {args.period}: one line per ERA5 field as it lands "
                    f"({ntot} total)  ·  tail -f this file\n")
    ssum = np.zeros((366, len(LAT), len(LON)), "float64"); scnt = np.zeros(366, "int32")
    done_ids: set[int] = set()
    if ckpt.exists():
        z = np.load(ckpt)
        ssum, scnt, done_ids = z["ssum"], z["scnt"], set(z["done"].tolist())
        print(f"  resuming from checkpoint: {len(done_ids)}/{len(all_chunks)} chunks "
              f"({int(scnt.sum())} fields) already done", flush=True)
    todo = [(i, ch) for i, ch in all_chunks if i not in done_ids]
    base = int(scnt.sum())                                # fields done before this session
    done = base
    t0 = time.time(); last_flush = t0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_accumulate, (args.var, v["era5"], v["era5_level"], ch, str(flog), store)): i
                for i, ch in todo}
        for fut in as_completed(futs):
            i = futs[fut]; s, c = fut.result()
            ssum += s; scnt += c; done_ids.add(i); done += int(c.sum())
            el = time.time() - t0; rate = (done - base) / el if el else 0
            eta = (ntot - done) / rate / 60 if rate else 0
            print(f"  read {done}/{ntot} fields ({100 * done / ntot:.0f}%)  "
                  f"{el / 60:.1f} min this run · ETA {eta:.1f} min", flush=True)
            if time.time() - last_flush > 150:           # checkpoint ~every 2.5 min
                tmp = ckpt.with_suffix(".tmp.npz")
                np.savez(tmp, ssum=ssum, scnt=scnt, done=np.array(sorted(done_ids)))
                os.replace(tmp, ckpt); last_flush = time.time()
    print(f"  read+mean done in {(time.time()-t0)/60:.1f} min", flush=True)
    ckpt.unlink(missing_ok=True)                          # success → drop the checkpoint

    raw = (ssum / np.maximum(scnt, 1)[:, None, None] * v["era5_scale"]).astype("float32")
    REF.mkdir(parents=True, exist_ok=True)

    # No smoothing: the per-day-of-year mean (over every day of the period) IS the
    # normal we use directly. The cyclic-padded interp above means there's no 0/360
    # seam NaN, so this writes clean.
    xr.DataArray(raw, dims=("dayofyear", "latitude", "longitude"),
                 coords={"dayofyear": np.arange(1, 367), "latitude": LAT, "longitude": LON},
                 attrs={"var": args.var, "units": v["units"], "period": f"{y0}-{y1}",
                        "samples_per_doy": int(np.median(scnt)),
                        "note": "ERA5 per-day-of-year mean (every day, NO smoothing) — used directly as normal"}
                 ).to_netcdf(out, encoding={"__xarray_dataarray_variable__":
                                            {"zlib": True, "complevel": 4}})
    print(f"wrote {out}  ({len(times)} days, total {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
