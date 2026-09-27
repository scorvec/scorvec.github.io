#!/usr/bin/env python3
"""Observed inputs for the MJO impact composites, on the same grids as cmip6_mjo_extract.py (LAPTOP, once).

  rmm    BoM RMM (Wheeler & Hendon 2004), 1974-06 -> 2024-02 (the BoM series ends there)  -> obs/rmm_bom.csv
  t2m    ERA5 2 m temperature, ~/era5_store/wb2_1p5_daily_global (1959-2026; dimension order differs by year, so
         transposed by name), 1979-2024 -> na / sa grids                                      -> obs/era5_t2m.npz
  z500   ERA5 500 hPa height, ~/era5_store/wb2_1p5_daily (NH, metres), 1979-2024 -> nh grid     -> obs/era5_z500.npz
  gpcp   GPCP 1DD v1.3 daily precipitation (NOAA NCEI CDR, 1 deg), 1997-2024 -> band / na / sa  -> obs/gpcp.npz
         (PSL's CPC-unified and OLR files were returning 502 on 2026-09-27; NCEI serves GPCP per day.)

    python mjo_obs_prepare.py rmm t2m z500 gpcp
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmip6_mjo_extract import GRIDS, _interp                                     # noqa: E402

OBS = Path.home() / "data_archive" / "cmip6_mjo" / "obs"
STORE = Path.home() / "era5_store"
UA = {"User-Agent": "Mozilla/5.0 (scorvec research)"}
Y0, Y1 = 1979, 2024


def rmm():
    url = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
    txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode()
    rows = []
    for ln in txt.splitlines()[2:]:
        p = ln.split()
        if len(p) < 7:
            continue
        y, m, d = int(p[0]), int(p[1]), int(p[2]); r1, r2, ph, amp = float(p[3]), float(p[4]), int(p[5]), float(p[6])
        if abs(r1) > 900 or abs(amp) > 900:
            continue
        rows.append((f"{y:04d}-{m:02d}-{d:02d}", r1, r2, ph, amp))
    df = pd.DataFrame(rows, columns=["date", "rmm1", "rmm2", "phase", "amp"])
    df.to_csv(OBS / "rmm_bom.csv", index=False)
    print(f"RMM: {len(df)} days {df.date.iloc[0]} .. {df.date.iloc[-1]}; amp>=1 on {100 * (df.amp >= 1).mean():.0f} %")


def _era5(var, sub, grids, conv=None):
    outs = {g: [] for g in grids}; dates = []
    for y in range(Y0, Y1 + 1):
        f = STORE / sub / var / f"{var}_{y}.nc"
        ds = xr.open_dataset(f)
        a = ds[list(ds.data_vars)[0]].transpose("time", "latitude", "longitude")
        lat = a.latitude.values.astype(float); lon = a.longitude.values.astype(float) % 360
        so = np.argsort(lon)
        v = a.values.astype(np.float32)[:, :, so]
        if conv:
            v = conv(v)
        for g in grids:
            glat, glon = GRIDS[g]
            outs[g].append(_interp(v, lat, lon[so], glat, glon).astype(np.float16))
        dates += [pd.Timestamp(t).strftime("%Y-%m-%d") for t in a.time.values]
        ds.close()
    return np.array(dates), {g: np.concatenate(v) for g, v in outs.items()}


def t2m():
    d, o = _era5("t2m", "wb2_1p5_daily_global", ["na", "sa"])
    m = float(np.nanmean(o["na"][:365].astype(float)))
    assert 250 < m < 300, f"t2m units? mean {m}"
    np.savez_compressed(OBS / "era5_t2m.npz", dates=d, tas_na=o["na"], tas_sa=o["sa"])
    print(f"ERA5 t2m: {len(d)} days, NA mean {m:.1f} K")


def z500():
    d, o = _era5("z500", "wb2_1p5_daily", ["nh"])
    m = float(np.nanmean(o["nh"][:365].astype(float)))
    assert 5000 < m < 6000, f"z500 units? mean {m} (expected metres)"
    np.savez_compressed(OBS / "era5_z500.npz", dates=d, z500_nh=o["nh"])
    print(f"ERA5 z500: {len(d)} days, NH mean {m:.0f} m")


def gpcp(y0=1997, y1=2024):
    base = "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access"
    cache = OBS / "gpcp_raw"; cache.mkdir(parents=True, exist_ok=True)
    names = []
    for y in range(y0, y1 + 1):
        idx = urllib.request.urlopen(urllib.request.Request(f"{base}/{y}/", headers=UA), timeout=120).read().decode()
        names += [(y, n) for n in sorted(set(re.findall(r"gpcp_v01r03_daily_d\d{8}_c\d{8}\.nc", idx)))]
    print(f"GPCP: {len(names)} daily files listed", flush=True)

    def one(yn):
        y, n = yn
        f = cache / n
        if f.exists() and f.stat().st_size > 100_000:
            return f
        for k in range(4):
            try:
                data = urllib.request.urlopen(urllib.request.Request(f"{base}/{y}/{n}", headers=UA), timeout=120).read()
                if data[:4] not in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
                    raise ValueError("not netCDF")
                f.write_bytes(data)
                return f
            except Exception:                                   # noqa: BLE001
                time.sleep(5 * (k + 1))
        return None
    with ThreadPoolExecutor(4) as ex:
        files = [f for f in ex.map(one, names) if f is not None]
    outs = {g: [] for g in ("band", "na", "sa")}; dates = []
    for f in sorted(files):
        ds = xr.open_dataset(f)
        p = ds["precip"].isel(time=0).values.astype(np.float32)
        p = np.where((p < 0) | (p > 1000), np.nan, p)
        lat = ds.latitude.values.astype(float); lon = ds.longitude.values.astype(float)
        for g in outs:
            glat, glon = GRIDS[g]
            outs[g].append(_interp(p[None], lat, lon, glat, glon)[0].astype(np.float16))
        dates.append(pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d"))
        ds.close()
    np.savez_compressed(OBS / "gpcp.npz", dates=np.array(dates), **{f"pr_{g}": np.stack(v) for g, v in outs.items()})
    print(f"GPCP: {len(dates)} days {dates[0]} .. {dates[-1]} (of {len(names)} listed)")


def main() -> int:
    OBS.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[1:]:
        {"rmm": rmm, "t2m": t2m, "z500": z500, "gpcp": gpcp}[c]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
