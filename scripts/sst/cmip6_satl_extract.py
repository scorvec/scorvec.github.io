#!/usr/bin/env python3
"""South Atlantic subtropical dipole (SASD) and South American surface fields for one CMIP6 historical member.

From the public Pangeo CMIP6 zarr stores (Google Cloud, anonymous), Amon 1950-2014:
  sw, ne, n34   monthly box means of `ts` on the native grid (cos-latitude weights, every cell centre in the box): the
                site's SASD poles (sst-roni.py SASD_SW 30-10W 40-30S, SASD_NE 20W-0 25-15S; the index is SW - NE, the
                Morioka convention) and Nino-3.4 (170-120W, 5S-5N). All three boxes are open ocean.
  pr, tas, psl  (month, lat, lon) on a 2.5-degree grid, 60S-15N x 90W-0 (bilinear from the native grid): mm/day, degC,
                hPa.
  calendar      the store's CF calendar (month lengths for day-weighted seasonal means).

    python scripts/sst/cmip6_satl_extract.py plan --catalog pangeo-cmip6.csv --members 3          # JSON matrix
    python scripts/sst/cmip6_satl_extract.py run --catalog pangeo-cmip6.csv --model MIROC6 --out out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARS = ("ts", "pr", "tas", "psl")
Y0, Y1 = 1950, 2014
LAT = np.arange(-60.0, 15.01, 2.5)
LON = np.arange(270.0, 360.01, 2.5)
BOXES = {"sw": ((-40, -30), (330, 350)), "ne": ((-25, -15), (340, 360)), "n34": ((-5, 5), (190, 240))}
RIPF = re.compile(r"r(\d+)i(\d+)p(\d+)f(\d+)")


def catalog(path: str) -> pd.DataFrame:
    c = pd.read_csv(path)
    return c[(c.experiment_id == "historical") & (c.table_id == "Amon") & c.variable_id.isin(VARS)]


def members(cat: pd.DataFrame, model: str, n: int) -> list[str]:
    """The first n members (by realisation, then the rest of the label) that publish all four variables."""
    have = cat[cat.source_id == model].groupby("member_id").variable_id.nunique()
    ripf = {m: tuple(map(int, g.groups())) for m in have.index if have[m] == len(VARS) and (g := RIPF.fullmatch(m))}
    if not ripf:
        return []
    # one initialisation/physics/forcing variant (the lowest), its realisations in order
    ipf = min(v[1:] for v in ripf.values())
    return sorted((m for m, v in ripf.items() if v[1:] == ipf), key=lambda m: ripf[m][0])[:n]


def store(cat: pd.DataFrame, model: str, member: str, var: str) -> str:
    g = cat[(cat.source_id == model) & (cat.member_id == member) & (cat.variable_id == var)]
    g = g.assign(native=g.grid_label.str.startswith("gn")).sort_values(["native", "version"])
    return g.zstore.iloc[-1]


def open_var(zstore: str, var: str) -> xr.DataArray:
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(zstore), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[var]
    da = da.rename({d: n for d, n in (("latitude", "lat"), ("longitude", "lon")) if d in da.dims})
    da = da.sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31"))
    return da.assign_coords(lon=da.lon % 360).sortby("lon").sortby("lat")


def box_mean(da: xr.DataArray, box) -> np.ndarray:
    (la0, la1), (lo0, lo1) = box
    sub = da.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
    w = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(w).mean(("lat", "lon")).values.astype("float32")


def regrid(da: xr.DataArray) -> np.ndarray:
    # the first column again at +360 so the domain's 0-degree edge interpolates instead of falling off the grid
    pad = da.isel(lon=[0])
    full = xr.concat([da, pad.assign_coords(lon=pad.lon + 360)], "lon")
    return full.interp(lat=LAT, lon=LON).values.astype("float32")


def run(cat: pd.DataFrame, model: str, n: int, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for member in members(cat, model, n):
        t0 = time.time()
        try:
            fields, months, calendar = {}, None, None
            for var in VARS:
                da = open_var(store(cat, model, member, var), var)
                # only the latitudes in use travel into memory: the boxes for ts, the domain (one cell wider) else
                da = da.sel(lat=slice(-42.5, 7.5) if var == "ts" else slice(LAT[0] - 5, LAT[-1] + 5)).load()
                ym = [f"{t.year:04d}-{t.month:02d}" for t in da.time.values]
                if len(ym) != (Y1 - Y0 + 1) * 12:
                    raise ValueError(f"{var}: {len(ym)} months")
                months = months or ym
                if ym != months:
                    raise ValueError(f"{var}: months differ from ts")
                calendar = calendar or str(da.time.dt.calendar)
                if var == "ts":
                    fields.update({k: box_mean(da, b) for k, b in BOXES.items()})
                else:
                    x = regrid(da)
                    fields[var] = x * 86400.0 if var == "pr" else x - 273.15 if var == "tas" else x / 100.0
            np.savez_compressed(
                out / f"{model}_{member}.npz",
                months=np.array(months),
                calendar=calendar,
                lat=LAT,
                lon=LON,
                source_id=model,
                member_id=member,
                **fields,
            )
            done += 1
            print(f"{model} {member}: {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001  one bad member must not lose the model's others
            print(f"::warning::{model} {member}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0 if done else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--model")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    cat = catalog(a.catalog)
    if a.cmd == "plan":
        models = sorted(m for m in cat.source_id.unique() if members(cat, m, 1))
        print(json.dumps({"model": models}))
        return 0
    return run(cat, a.model, a.members, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
