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
TROP_LAT = slice(-20, 20)                        # RONI tropical mean band
SIGMA_PATH = REPO / "scripts" / "sst" / "roni_sigma.json"


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def _boxes(ds):
    """(n34, trop) member series in °C — same convention as c3s_nino34.py:
    Niño-3.4 box mean and the 20°S-20°N all-longitude tropical mean."""
    out = []
    for latsel, lonsel in ((LAT, LON), (TROP_LAT, None)):
        box = ds.SST.sel(latitude=latsel)
        if lonsel is not None:
            box = box.sel(longitude=lonsel)
        w = np.cos(np.deg2rad(box.latitude))
        out.append(box.weighted(w).mean(("latitude", "longitude")))
    return out


def n34_forecast(issue: str):
    """(member, lead) absolute Niño-3.4 and tropical-mean SST (NRT store)."""
    ds = _open(f"{BASE}/forecast/{issue}/ocn_monthly.zarr")
    n34, trop = _boxes(ds)
    return n34.values, trop.values


def n34_clim(month: int):
    """Per-lead reforecast climatology for both boxes, cached to CSV."""
    cache = CLIMDIR / f"clim_n34_{month:02d}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if "clim_trop" in df.columns:
            return df["clim"].values, df["clim_trop"].values
    ds = _open(f"{BASE}/reforecast/{month:02d}/ocn_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    n34, trop = _boxes(ds)                        # (init, member, lead)
    print(f"clim {month:02d}: pooling {n34.sizes['init']} years x "
          f"{n34.sizes['member']} members ...", flush=True)
    cn = n34.mean(("init", "member")).values
    ct = trop.mean(("init", "member")).values
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lead": np.arange(len(cn)), "clim": np.round(cn, 4),
                  "clim_trop": np.round(ct, 4)}).to_csv(cache, index=False)
    print(f"clim {month:02d}: cached -> {cache.name}", flush=True)
    return cn, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    month = int(issue[4:6])

    try:
        fc, ftrop = n34_forecast(issue)              # (31, 12) absolute degC
    except Exception as e:                           # noqa: BLE001
        print(f"SFS {issue}: forecast store not readable ({str(e)[:80]})",
              file=sys.stderr)
        return 1
    clim, clim_trop = n34_clim(month)                # (12,) each
    anom = fc - clim[None, :]
    # RONI: relative index (N34 anom minus tropical-mean anom), rescaled per
    # VALID calendar month by the site's sigma(ONI)/sigma(relative) table —
    # identical convention to the C3S feed (c3s_nino34.py).
    scale_tab = json.loads(SIGMA_PATH.read_text())["scale_by_month"]
    rel = anom - (ftrop - clim_trop[None, :])
    vmonths = [((month - 1 + k) % 12) + 1 for k in range(anom.shape[1])]
    rnino = rel * np.array([scale_tab.get(str(m), 1.0) for m in vmonths])[None, :]

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
        "members_rnino": np.round(rnino[ok], 3).tolist(),
    }, separators=(",", ":")))
    mn = np.nanmean(anom[ok], axis=0)
    print(f"SFS {issue}: {int(ok.sum())} members -> {OUT.relative_to(REPO)}")
    print("  ens-mean anom:", np.round(mn, 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
