#!/usr/bin/env python3
"""NOAA SFS Beta (v1.0 prototype) Niño-3.4 forecast feed.

The Seasonal Forecast System beta (UFS-based CFSv2 successor, NRT since
March 2026) publishes monthly-mean forecast zarr on NODD: 31 members
initialized at the start of each month, 12 monthly leads, with an
11-member reforecast (1991+) per init month. Chunks are (member, 1 lead,
globe), so a Niño-3.4 read is a handful of ~MB chunk fetches — no
subscription, no auth.

Anomalies are computed against the SFS's OWN reforecast climatology for
the same init month and lead (1991-2020, 11 members × 30 years pooled) —
model drift and bias are removed per-lead exactly as C3S models are
anomalized against their own hindcasts on the ENSO forecasts page. The
clim differs from C3S's 1993-2016 hindcast window by ≲0.1 °C for
Niño-3.4; the page footnote carries that caveat.

Feed: assets/sfs/data/sfs_nino34.json (members × leads, plus clim meta).
Clim cache: scripts/sfs/data/clim_n34_MM.csv (one build per init month,
~360 chunk reads; cheap forever after).

    python scripts/sfs/sfs_nino.py                # current issue month
    python scripts/sfs/sfs_nino.py --issue 202608
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "assets" / "sfs" / "data" / "sfs_nino34.json"
CLIMDIR = HERE / "data"

BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
CLIM_Y0, CLIM_Y1 = 1991, 2020

# Niño-3.4: 5S-5N, 170W-120W (190-240E on the 0-359 ocean grid)
LAT = slice(-5, 5)
LON = slice(190, 240)


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def n34_forecast(issue: str) -> np.ndarray:
    """(member, lead) absolute Niño-3.4 SST from the NRT monthly store."""
    ds = _open(f"{BASE}/forecast/{issue}/ocn_monthly.zarr")
    box = ds.SST.sel(latitude=LAT, longitude=LON)
    w = np.cos(np.deg2rad(box.latitude))
    return box.weighted(w).mean(("latitude", "longitude")).values


def n34_clim(month: int) -> np.ndarray:
    """Per-lead reforecast climatology (12,), cached to CSV."""
    cache = CLIMDIR / f"clim_n34_{month:02d}.csv"
    if cache.exists():
        return pd.read_csv(cache)["clim"].values
    ds = _open(f"{BASE}/reforecast/{month:02d}/ocn_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    box = ds.SST.sel(latitude=LAT, longitude=LON)
    w = np.cos(np.deg2rad(box.latitude))
    series = box.weighted(w).mean(("latitude", "longitude"))  # (init, member, lead)
    print(f"clim {month:02d}: pooling {series.sizes['init']} years x "
          f"{series.sizes['member']} members ...", flush=True)
    clim = series.mean(("init", "member")).values
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lead": np.arange(len(clim)), "clim": np.round(clim, 4)}
                 ).to_csv(cache, index=False)
    print(f"clim {month:02d}: cached -> {cache.name}", flush=True)
    return clim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    month = int(issue[4:6])

    try:
        fc = n34_forecast(issue)                     # (31, 12) absolute degC
    except Exception as e:                           # noqa: BLE001
        print(f"SFS {issue}: forecast store not readable ({str(e)[:80]})",
              file=sys.stderr)
        return 1
    clim = n34_clim(month)                           # (12,)
    anom = fc - clim[None, :]

    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")
    valid = [(t0 + pd.DateOffset(months=k)).strftime("%Y-%m")
             for k in range(anom.shape[1])]
    ok = np.isfinite(anom).all(axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "system": "NOAA SFS beta v1.0 prototype (UFS coupled)",
        "clim": f"own reforecast {CLIM_Y0}-{CLIM_Y1}, 11 members, per init month & lead",
        "valid_months": valid,
        "members": np.round(anom[ok], 3).tolist(),
    }, separators=(",", ":")))
    mn = np.nanmean(anom[ok], axis=0)
    print(f"SFS {issue}: {int(ok.sum())} members -> {OUT.relative_to(REPO)}")
    print("  ens-mean anom:", np.round(mn, 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
