#!/usr/bin/env python3
"""Monthly North American 2 m temperature and precipitation, all 12 months, for the CMIP6 members that also have monthly
500 hPa height (cmip6_z500_extract.py; 147 members of the 16 ENSO-study models) - the PDO forced-response test month by
month (2026-09-27; user: "I'm also interested in PDO's impact on months in the rest of the year and also impacts on north
america temp and precip"). LAPTOP job; public Pangeo CMIP6 zarr stores (Google Cloud, anonymous); nothing raw kept.

Amon tas and pr, 1950-01 .. 2014-12, bilinear to 2.5 deg over 15-75N, 170W-50W (the grid of cmip6_monthly_tas_extract.py),
int16: tas (K - 273.15) x 100, pr mm/day x 100. Output data/cmip6_enso/na_monthly/<source>_<member>.npz (resumable).

    python scripts/sst/cmip6_na_monthly_extract.py [--workers 2] [--limit N]
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

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "cmip6_enso"
OUT = SRC / "na_monthly"
CAT = HERE.parent / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"
SO = {"token": "anon"}
LAT = np.arange(15.0, 75.01, 2.5)
LON = np.arange(190.0, 310.01, 2.5)


def one(job):
    src, member, stores = job
    dest = OUT / f"{src}_{member}.npz"
    if dest.exists():
        return f"{src} {member}: have"
    t0 = time.time()
    for attempt in range(3):
        try:
            out = {}
            for var in ("tas", "pr"):
                ds = xr.open_zarr(stores[var], storage_options=SO, consolidated=True)
                v = ds[var].sel(time=slice("1950-01-01", "2014-12-31"))
                v = v.assign_coords(lon=v.lon % 360).sortby("lon").sortby("lat")
                months = [f"{t.year:04d}-{t.month:02d}" for t in pd.to_datetime([str(t)[:10] for t in v.time.values])]
                if len(months) != 780:
                    raise ValueError(f"{var}: {len(months)} months")
                x = v.interp(lat=LAT, lon=LON).values.astype("float32")
                q = (x - 273.15) * 100 if var == "tas" else x * 86400.0 * 100
                out[var] = np.where(np.isfinite(q), np.round(q), -32768).clip(-32768, 32767).astype("int16")
                out["months"] = np.array(months)
            np.savez_compressed(dest, **out, lat=LAT, lon=LON, source_id=src, member_id=member)
            return f"{src} {member}: {time.time() - t0:.0f}s"
        except Exception as e:                                             # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:160]}"; time.sleep(10)
    return f"{src} {member}: FAILED {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(Path(f).stem for f in glob.glob(str(SRC / "z500" / "*.npz")) if ".part" not in f)
    cat = pd.read_csv(CAT)
    c = cat[(cat.table_id == "Amon") & (cat.experiment_id == "historical") & cat.variable_id.isin(["tas", "pr"])]
    c = c.sort_values("version").groupby(["source_id", "member_id", "variable_id"]).zstore.last().unstack("variable_id")
    jobs = []
    for n in names:
        src, mem = n.rsplit("_", 1)
        if (src, mem) in c.index and c.loc[(src, mem)].notna().all():
            jobs.append((src, mem, c.loc[(src, mem)].to_dict()))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(f"{len(jobs)} members ({a.workers} workers)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            if (i + 1) % 10 == 0 or "FAILED" in r or i < 2:
                print(f"[{i + 1}/{len(jobs)}] {r}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
