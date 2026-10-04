#!/usr/bin/env python3
"""MERRA-2 daily equatorial-band zonal wind for RMM self-standardization.

Fetches U (42 levels, 15S-15N, 2.5 deg lon) daily means of 0/6/12/18Z from
M2I3NPASM via GES DISC OPeNDAP, storing the cos-weighted 15S-15N band mean
per longitude. One contiguous full-profile hyperslab per day (faster than
level subsets — DAP request latency dominates volume).

Purpose: run the operational wind-only RMM projection over ~2 decades and set
pc_wind_std per mode from the std of THAT series (the WH04 standardization,
measured on a modern-reanalysis data stream instead of NCEP 1979-2001 —
the mismatch that inflated amplitudes 2.2x, caught 2026-08-10).

Yearly files, day-resumable, partitionable [k n].
Output: data/reference/m2_eq_u/m2_eq_u_<YYYY>.nc  (uband: time, lev, lon)

    python src/fetch_m2_eq_u.py [k n]
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parents[1] / "data" / "reference" / "m2_eq_u"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4"
LATSL = slice(150, 211, 2)          # 15S-15N at 1 deg (M2 lat = -90 + 0.5*i)
LONSL = slice(0, 576, 4)            # 2.5 deg
TSL = slice(0, 8, 2)                # 0/6/12/18Z
Y0, Y1 = 2003, 2024


def stream(y):
    if y <= 1991: return ["100", "101"]
    if y <= 2000: return ["200", "201"]
    if y <= 2010: return ["300", "301"]
    return ["400", "401"]


def fetch_day(d):
    for attempt in range(3):
        for s in stream(d.year):
            url = (f"{BASE}/{d.year}/{d.month:02d}/"
                   f"MERRA2_{s}.inst3_3d_asm_Np.{d:%Y%m%d}.nc4")
            try:
                with xr.open_dataset(url, engine="netcdf4") as ds:
                    sub = ds["U"].isel(time=TSL, lat=LATSL, lon=LONSL).compute()
                u = sub.values                       # (4, lev, lat, lon)
                if not (np.isnan(u[:, sub.lev.values <= 200]).mean() < 0.01
                        and np.nanmax(np.abs(u)) < 400):
                    continue
                w = np.cos(np.deg2rad(sub.lat.values))
                with np.errstate(invalid="ignore"):
                    band = np.nansum(u * w[None, None, :, None], axis=2) / w.sum()
                    band = np.nanmean(band, axis=0)  # daily mean -> (lev, lon)
                return dict(lev=sub.lev.values, lon=sub.lon.values,
                            uband=band.astype("float32"))
            except Exception:                        # noqa: BLE001
                continue
        time.sleep(8)
    return None


def main():
    k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
    years = [y for i, y in enumerate(range(Y0, Y1 + 1)) if i % n == k]
    for y in years:
        f = OUT / f"m2_eq_u_{y}.nc"
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
            r = fetch_day(d)
            if r is None:
                print(f"  {d.date()}: FAILED", flush=True)
                continue
            buf.append(xr.Dataset(
                {"uband": (("time", "lev", "lon"), r["uband"][None])},
                coords={"time": [d], "lev": r["lev"], "lon": r["lon"]}))
            if len(buf) >= 30:
                merged = xr.concat(pieces + buf, dim="time").sortby("time")
                merged = merged.isel(
                    time=~pd.Index(merged.time.values).duplicated(keep="last"))
                tmp = f.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(f)
                pieces, buf = [merged], []
                print(f"  ... {pd.Timestamp(merged.time.values[-1]).date()} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if buf:
            merged = xr.concat(pieces + buf, dim="time").sortby("time")
            merged = merged.isel(
                time=~pd.Index(merged.time.values).duplicated(keep="last"))
            tmp = f.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(f)
        print(f"{y}: done in {time.time()-t0:.0f}s", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
