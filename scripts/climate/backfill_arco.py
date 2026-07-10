#!/usr/bin/env python3
"""Backfill the 2023–present gap in the climate series directly from ARCO-ERA5
(the CDS daily-statistics product is under an ECMWF known-issue and hangs).

Reads t2m + z500 synoptic (00/06/12/18Z) means month by month from the
anonymous ARCO hourly zarr, interpolates to the 1.5° climatology grid, and for
each day computes:
  • z500 NH anomaly  → appends to the z500_nh store (per-year npz)
  • regional t2m anomaly means → gap_t2m_regions.csv
  • teleconnection index projections onto the fitted patterns → gap_tele.csv

Writes only backfill-owned files (never the shared series CSVs), so it can run
alongside the daily monitor. `--merge` then folds the gap files into the main
CSVs in one cheap pass. Resumable: skips months already in the gap files.

    python scripts/climate/backfill_arco.py           # fetch + compute the gap
    python scripts/climate/backfill_arco.py --merge   # fold into the main CSVs
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import climate_monitor as CM
from build_z500_indices import STORE, LAT_MIN

GAP_T2M = HERE / "data" / "gap_t2m_regions.csv"
GAP_TELE = HERE / "data" / "gap_tele.csv"
START = "2023-01-01"


def _load(rows_path):
    if rows_path.exists():
        df = pd.read_csv(rows_path, index_col=0)
        return {k: v.to_dict() for k, v in df.iterrows()}
    return {}


def main() -> int:
    clim_t = xr.open_dataset(HERE / "era5_clim_t2m.nc")
    clim_z = xr.open_dataset(HERE / "era5_clim_z500.nc")
    lats, lons = clim_t.latitude.values, clim_t.longitude.values
    sel = lats >= LAT_MIN
    tele_pat = xr.open_dataset(HERE / "tele_patterns.nc")

    t2m_rows, tele_rows = _load(GAP_T2M), _load(GAP_TELE)
    ds = CM.open_arco()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    months = pd.date_range(START, today, freq="MS")

    z_da = ds["geopotential"].sel(level=500)
    t_da = ds["2m_temperature"]

    for m0 in months:
        tag = f"{m0:%Y-%m}"
        m1 = m0 + pd.offsets.MonthBegin(1)
        days = pd.date_range(m0, min(m1 - pd.Timedelta(days=1), today - pd.Timedelta(days=5)))
        days = [d for d in days if d.strftime("%Y-%m-%d") not in t2m_rows]
        if not days:
            continue
        stamps = [d + pd.Timedelta(hours=h) for d in days for h in CM.SYNOPTIC_H]
        try:
            tt = t_da.sel(time=stamps).compute()
            zz = z_da.sel(time=stamps).compute()
        except Exception as e:                                # noqa: BLE001
            print(f"  {tag}: ARCO read failed ({repr(e)[:60]}); stopping", flush=True)
            break
        # group the 4 synoptic steps into daily means
        for d in days:
            iso = d.strftime("%Y-%m-%d")
            k = CM.doy365(d)
            sl = slice(None)
            t_day = tt.sel(time=[d + pd.Timedelta(hours=h) for h in CM.SYNOPTIC_H]).mean("time")
            z_day = zz.sel(time=[d + pd.Timedelta(hours=h) for h in CM.SYNOPTIC_H]).mean("time")
            t15 = t_day.interp(latitude=lats, longitude=lons).values - 273.15
            z15 = (z_day.interp(latitude=lats, longitude=lons).values) / 9.80665
            t_anom = t15 - CM.eval_clim(clim_t["coef"].values, k)
            z_anom = z15 - CM.eval_clim(clim_z["coef"].values, k)
            t2m_rows[iso] = CM.region_means(t_anom, lats, lons)
            tele_rows[iso] = CM._tele_project_day(z_anom, lats, tele_pat, pd.Timestamp(d))
            # also grow the z500 NH store for future refits
            _append_store(d.year, iso, z_anom[sel].astype("float16"))
        _save(GAP_T2M, t2m_rows, CM.REGION_ORDER)
        _save(GAP_TELE, tele_rows, ["nao", "pna", "epo", "wpo", "ao"])
        print(f"  {tag}: +{len(days)} days (gap files now {len(t2m_rows)} rows)", flush=True)
    print("backfill fetch complete", flush=True)
    return 0


_STORE_CACHE: dict = {}


def _append_store(year, iso, nh_anom):
    p = STORE / f"{year}.npz"
    if year not in _STORE_CACHE:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            _STORE_CACHE[year] = {d: a for d, a in zip(z["dates"], z["anoms"])}
        else:
            _STORE_CACHE[year] = {}
    _STORE_CACHE[year][iso] = nh_anom
    ks = sorted(_STORE_CACHE[year])
    np.savez_compressed(p, dates=np.array(ks),
                        anoms=np.array([_STORE_CACHE[year][k] for k in ks]))


def _save(path, rows, order):
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[[c for c in order if c in df.columns]]
    df.sort_index().round(3).to_csv(path, index_label="date")


def merge() -> int:
    for gap, main in [(GAP_T2M, HERE.parents[1] / "assets" / "climate" / "t2m_regions_daily.csv"),
                      (GAP_TELE, HERE.parents[1] / "assets" / "climate" / "teleconnections_daily.csv")]:
        if not gap.exists():
            print(f"  {gap.name} missing; skip"); continue
        g = pd.read_csv(gap, index_col=0)
        m = pd.read_csv(main, index_col=0)
        merged = pd.concat([m, g[~g.index.isin(m.index)]]).sort_index()
        merged.to_csv(main, index_label="date")
        print(f"  merged {len(g)} gap rows into {main.name} → {len(merged)} total "
              f"({merged.index.min()} … {merged.index.max()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(merge() if "--merge" in sys.argv else main())
