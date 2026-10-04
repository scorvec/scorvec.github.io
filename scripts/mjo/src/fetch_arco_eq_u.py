#!/usr/bin/env python3
"""ERA5 (ARCO) daily equatorial-band zonal wind for RMM self-standardization.

Supersedes the MERRA-2 sweep (fetch_m2_eq_u.py): calibrate pc_wind_std on the
same reanalysis stream the operational index consumes (ERA5/AIFS).

Reads the native 0.25 deg 37-level ARCO store per timestep. One chunk = one
hour x all levels x full globe, so u850+u200 and the whole 15S-15N band arrive
in the same read — latitude subsetting saves nothing on the wire, only decode;
the real cost knob is snapshots per day (ARCO_HOURS=12 for a 4x cheaper
1x/day snapshot; band-mean u diurnal cycle < 0.3 m/s). Stores the cos-weighted
15S-15N band mean per longitude at 1 deg, all 37 levels.

Yearly files, day-resumable, partitionable [k n] (GCS tolerates parallel
workers, unlike GES DISC's 2-connection cap).
Output: data/reference/arco_eq_u/arco_eq_u_<YYYY>.nc  (uband: time, lev, lon)

    python src/fetch_arco_eq_u.py [k n]
"""
from pathlib import Path
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parents[1] / "data" / "reference" / "arco_eq_u"
OUT.mkdir(parents=True, exist_ok=True)
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
HOURS = [int(h) for h in os.environ.get("ARCO_HOURS", "0,6,12,18").split(",")]
Y0, Y1 = 2003, 2024


def open_u():
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
    u = ds["u_component_of_wind"]
    lat = u.latitude.values
    latsl = slice(15.05, -15.05) if lat[0] > lat[-1] else slice(-15.05, 15.05)
    return u.sel(latitude=latsl).isel(longitude=slice(0, 1440, 4))


def fetch_day(u, d):
    for attempt in range(3):
        try:
            times = [d + pd.Timedelta(hours=h) for h in HOURS]
            sub = u.sel(time=times).compute()     # (nt, lev, lat, lon)
            v = sub.values
            if not (np.isnan(v).mean() < 0.01 and np.nanmax(np.abs(v)) < 400):
                raise ValueError("failed sanity check")
            w = np.cos(np.deg2rad(sub.latitude.values))
            band = np.nansum(v * w[None, None, :, None], axis=2) / w.sum()
            band = band.mean(axis=0)              # daily mean -> (lev, lon)
            return dict(lev=sub.level.values, lon=sub.longitude.values,
                        uband=band.astype("float32")), u
        except Exception:                         # noqa: BLE001
            time.sleep(5)
            try:
                u = open_u()                      # fresh handle after stream errors
            except Exception:                     # noqa: BLE001
                pass
    return None, u


def save(f, pieces, buf):
    merged = xr.concat(pieces + buf, dim="time").sortby("time")
    merged = merged.isel(
        time=~pd.Index(merged.time.values).duplicated(keep="last"))
    tmp = f.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(f)
    return merged


def main():
    k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
    u = open_u()
    years = [y for i, y in enumerate(range(Y0, Y1 + 1)) if i % n == k]
    for y in years:
        f = OUT / f"arco_eq_u_{y}.nc"
        have, done = None, set()
        if f.exists():
            have = xr.open_dataset(f).load(); have.close()
            done = set(pd.DatetimeIndex(have.time.values).normalize())
        todo = [d for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")
                if d not in done]
        if not todo:
            print(f"{y}: complete ({len(done)} days)", flush=True)
            continue
        print(f"{y}: fetching {len(todo)} days", flush=True)
        pieces = [have] if have is not None else []
        buf, t0 = [], time.time()
        for d in todo:
            r, u = fetch_day(u, d)
            if r is None:
                print(f"  {d.date()}: FAILED", flush=True)
                continue
            buf.append(xr.Dataset(
                {"uband": (("time", "lev", "lon"), r["uband"][None])},
                coords={"time": [d], "lev": r["lev"], "lon": r["lon"]}))
            if len(buf) >= 15:
                pieces, buf = [save(f, pieces, buf)], []
                print(f"  ... {d.date()} ({time.time()-t0:.0f}s)", flush=True)
        if buf:
            save(f, pieces, buf)
        print(f"{y}: done in {time.time()-t0:.0f}s", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
