#!/usr/bin/env python3
"""ARCO ERA5 tail (2023 -> present) for the daily store + NAM tower, combined.

The ar/full_37 zarr chunks span ALL 37 levels at 0.25 deg (~154 MB decoded per
timestep per pressure variable), so each geopotential chunk is fetched ONCE and
serves both products:

  zplev 13-level z field, daily mean of 0/6/12/18Z on the store's 1.5 deg
        0-90N grid -> ~/era5_store/wb2_1p5_daily/zplev/zplev_<Y>.nc
  z500  the 500 hPa slice of the same read -> z500/z500_<Y>.nc
  cap_z 13-level 65-90N cos-weighted cap mean (NAM tower rungs)
        + cap_slp from MSLP -> scripts/telecon/data/nam_tower_era5.nc (append,
        dedup keep-last, so the 10-day WB2 2023 stub is replaced)
  slp   full MSLP field (hPa), same grid -> slp/slp_<Y>.nc
  t2m   daily mean, same store grid -> t2m_<Y>.nc  (cheap surface chunks)

u200/v200/prcp are NOT fetched: each is another full-chunk variable (~19 s per
timestep) with no current downstream need past 2023.

Month-level resumable: monthly pieces cached under <var>/tail_pieces/, year
files assembled from them. ~19 s per z timestep, 3 parallel fetchers.

    python arco_tail.py            # 2023 -> latest available
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STORE = Path(os.environ.get("ERA5_STORE", "~/era5_store")).expanduser()
LAYER = STORE / "wb2_1p5_daily"
TOWER = Path(__file__).resolve().parent.parent / "telecon" / "data" / "nam_tower_era5.nc"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665
HOURS = [0, 6, 12, 18]
LEVS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
RANGES = {1000: (-400, 700), 925: (200, 1300), 850: (800, 1900),
          700: (2200, 3400), 600: (3400, 4800), 500: (4700, 6100),
          400: (6300, 7800), 300: (8000, 9800), 250: (9200, 11100),
          200: (10500, 12500), 150: (12000, 14500), 100: (14500, 17000),
          50: (18000, 21200)}

ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
zfull = ds["geopotential"]
t2 = ds["2m_temperature"]
msl = ds["mean_sea_level_pressure"]

lat = ds.latitude.values                      # 90 .. -90 descending
i15 = np.where(np.isin(lat, np.arange(0, 90.1, 1.5)))[0]   # store grid rows
lat15 = lat[i15]                              # descending 90..0 -> flip later
icap = np.where(lat >= 65)[0][::4]            # cap rows at 1 deg
wcap = np.cos(np.deg2rad(lat[icap]))
ilon15 = np.arange(0, 1440, 6)


LI = None                                                     # set at first fetch


def fetch_z(ts):
    """One geopotential chunk -> (z13 1.5deg 0-90N field, cap_z[13])."""
    global LI
    for attempt in range(4):
        try:
            x = zfull.sel(time=ts).compute().values / G      # (37, 721, 1440)
            if LI is None:
                LI = [int(np.where(ds.level.values == L)[0][0]) for L in LEVS]
            z13 = x[np.ix_(LI, i15, ilon15)]                 # (13, 61, 240)
            capz = np.array([np.average(x[j][icap].mean(axis=1), weights=wcap)
                             for j in LI])
            return z13.astype("float32"), capz
        except Exception:                                     # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def fetch_surface(ts):
    """(t2m field, slp field hPa, cap-mean slp hPa) for one timestep."""
    for attempt in range(4):
        try:
            t = t2.sel(time=ts).compute().values
            s = msl.sel(time=ts).compute().values / 100.0
            caps = np.average(s[icap].mean(axis=1), weights=wcap)
            return (t[np.ix_(i15, ilon15)].astype("float32"),
                    s[np.ix_(i15, ilon15)].astype("float32"), float(caps))
        except Exception:                                     # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def find_end():
    """Latest day with finite t2m (ARCO pads the time axis with NaN)."""
    d = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for back in range(2, 45):
        ts = d - pd.Timedelta(days=back)
        try:
            v = t2.sel(time=ts + pd.Timedelta(hours=12)).isel(
                latitude=slice(0, 4), longitude=slice(0, 4)).compute().values
            if np.isfinite(v).all():
                return ts
        except Exception:                                     # noqa: BLE001
            continue
    raise SystemExit("no recent finite ARCO day found")


def month_piece(y, m, end):
    """Compute one month; returns dict of monthly datasets or None if cached."""
    pdir = LAYER / "tail_pieces"
    pdir.mkdir(parents=True, exist_ok=True)
    fp = pdir / f"tail_{y}{m:02d}.nc"
    if fp.exists():
        with xr.open_dataset(fp) as c:
            return c.load()
    d0 = pd.Timestamp(y, m, 1)
    d1 = min(d0 + pd.offsets.MonthEnd(0), end)
    if d0 > end:
        return None
    days = pd.date_range(d0, d1)
    tss = [d + pd.Timedelta(hours=h) for d in days for h in HOURS]
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        zres = list(ex.map(fetch_z, tss))
    with ThreadPoolExecutor(3) as ex:
        sres = list(ex.map(fetch_surface, tss))
    nh = len(HOURS)
    nd, nlat = len(days), len(i15)
    zplev = np.stack([r[0] for r in zres]).reshape(nd, nh, len(LEVS), nlat, 240).mean(1)
    capz = np.stack([r[1] for r in zres]).reshape(nd, nh, len(LEVS)).mean(1)
    t2m = np.stack([r[0] for r in sres]).reshape(nd, nh, nlat, 240).mean(1)
    slpf = np.stack([r[1] for r in sres]).reshape(nd, nh, nlat, 240).mean(1)
    caps = np.array([r[2] for r in sres]).reshape(nd, nh).mean(1)
    z500 = zplev[:, LEVS.index(500)]
    out = xr.Dataset(
        {"zplev": (("time", "level", "latitude", "longitude"), zplev[:, :, ::-1, :]),
         "z500": (("time", "latitude", "longitude"), z500[:, ::-1, :]),  # -> 0..90 asc
         "t2m": (("time", "latitude", "longitude"), t2m[:, ::-1, :]),
         "slp": (("time", "latitude", "longitude"), slpf[:, ::-1, :]),
         "cap_z": (("time", "level"), capz),
         "cap_slp": ("time", caps)},
        coords={"time": days, "latitude": lat15[::-1],
                "longitude": ds.longitude.values[ilon15],
                "level": np.array(LEVS, float)})
    ok = (np.isfinite(zplev).all() and np.isfinite(t2m).all()
          and np.isfinite(slpf).all()
          and 4500 < z500.mean() < 6500 and 240 < t2m.mean() < 320
          and 940 < caps.mean() < 1060)
    for L, (lo, hi) in RANGES.items():
        x = out.cap_z.sel(level=L).values
        ok &= bool((x > lo).all() and (x < hi).all())
    if not ok:
        print(f"  {y}-{m:02d}: GATE FAIL — not cached", flush=True)
        return None
    tmp = fp.with_suffix(f".tmp{os.getpid()}.nc")
    out.to_netcdf(tmp); os.replace(tmp, fp)
    print(f"  {y}-{m:02d}: {len(days)} days in {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    end = find_end()
    print(f"ARCO data through {end.date()}", flush=True)
    for y in range(2023, end.year + 1):
        months = [month_piece(y, m, end) for m in range(1, 13)]
        months = [m for m in months if m is not None]
        if not months:
            continue
        yr = xr.concat(months, dim="time").sortby("time")
        for var in ("zplev", "z500", "t2m", "slp"):
            fp = LAYER / var / f"{var}_{y}.nc"
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(f".tmp{os.getpid()}.nc")
            enc = {var: dict(zlib=True, complevel=1)} if var == "zplev" else None
            dims = (("time", "level", "longitude", "latitude") if var == "zplev"
                    else ("time", "longitude", "latitude"))   # store convention
            yr[var].transpose(*dims).rename(var).to_netcdf(tmp, encoding=enc)
            os.replace(tmp, fp)
        # tower append
        have = xr.open_dataset(TOWER).load(); have.close()
        new = yr[["cap_z", "cap_slp"]]
        merged = xr.concat([have, new], dim="time").sortby("time")
        merged = merged.isel(time=~pd.Index(merged.time.values).duplicated(keep="last"))
        tmp = TOWER.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(TOWER)
        print(f"{y}: store {yr.sizes['time']} days; tower -> "
              f"{pd.Timestamp(merged.time.values[-1]).date()}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
