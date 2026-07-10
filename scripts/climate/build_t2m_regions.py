#!/usr/bin/env python3
"""Backfill the daily regional 2 m temperature anomaly series (1979 → present).

Two sources, both on the 1.5° grid the climatology lives on:
  • 1979 – 2022-12-31: WeatherBench2's 6-hourly ERA5 (00/06/12/18Z) — fast,
    anonymous, no queue.
  • 2023-01-01 → present: CDS `derived-era5-single-levels-daily-statistics`
    (daily_mean, regridded to 1.5° server-side) via cdsapi (~small request).
The daily monitor (climate_monitor.py) appends each new day from ARCO after
this backfill, so this script normally runs once.

Output: assets/climate/t2m_regions_daily.csv (committed) — date + one column
per region, °C anomaly vs the committed 1991–2020 climatology.

    python scripts/climate/build_t2m_regions.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from climate_monitor import eval_clim, doy365, region_means, REGION_ORDER, SERIES_CSV  # noqa: E402

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
START = "1979-01-01"
WB2_END = "2022-12-31"


def anomaly_rows(dates, fields, clim_coef, lats, lons) -> dict:
    rows = {}
    for d, f in zip(dates, fields):
        anom = f - eval_clim(clim_coef, doy365(pd.Timestamp(d)))
        rows[pd.Timestamp(d).strftime("%Y-%m-%d")] = region_means(anom, lats, lons)
    return rows


def main() -> int:
    clim = xr.open_dataset(HERE / "era5_clim_t2m.nc")
    coef = clim["coef"].values
    lats, lons = clim.latitude.values, clim.longitude.values
    rows: dict = {}
    if SERIES_CSV.exists():
        old = pd.read_csv(SERIES_CSV, index_col=0)
        rows = {k: v.to_dict() for k, v in old.iterrows()}
        print(f"resuming with {len(rows)} existing rows")

    # ---- WB2: 1979 → 2022 ----
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(WB2), chunks={"time": 1464})
    da = ds["2m_temperature"]
    for yr in range(int(START[:4]), int(WB2_END[:4]) + 1):
        if f"{yr}-06-15" in rows:
            continue                                        # year already done
        sel = da.sel(time=str(yr))
        daily = (sel.resample(time="1D").mean()
                 .transpose("time", "latitude", "longitude").compute()) - 273.15
        rows.update(anomaly_rows(daily.time.values, daily.values, coef, lats, lons))
        print(f"  WB2 {yr} done", flush=True)
        if yr % 5 == 0:
            _save(rows)

    # ---- CDS: 2023 → present (daily_mean t2m at 1.5°) ----
    try:
        import cdsapi
        import tempfile
        c = cdsapi.Client(quiet=True)
        today = datetime.now(timezone.utc).date()
        for yr in range(2023, today.year + 1):
            if f"{yr}-06-15" in rows and yr < today.year:
                continue
            probe = [d for d in pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
                     if d.strftime("%Y-%m-%d") not in rows and d.date() < today]
            if not probe:
                continue
            months = sorted({d.month for d in probe})
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
                target = tf.name
            print(f"  CDS request {yr} months {months} …", flush=True)
            c.retrieve("derived-era5-single-levels-daily-statistics", {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": str(yr),
                "month": [f"{m:02d}" for m in months],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1-hourly",
                "grid": "1.5/1.5",
            }, target)
            dsy = xr.open_dataset(target)
            v = dsy[[k for k in dsy.data_vars if "t2m" in k or "2m" in k][0]]
            tname = "valid_time" if "valid_time" in v.dims else "time"
            laname = "latitude" if "latitude" in v.coords else "lat"
            loname = "longitude" if "longitude" in v.coords else "lon"
            v = v.rename({tname: "time", laname: "latitude", loname: "longitude"})
            v = v.sortby("latitude")                        # match clim orientation (S→N)
            v = v.interp(latitude=lats, longitude=lons)     # snap to clim grid exactly
            vals = v.transpose("time", "latitude", "longitude").values - 273.15
            rows.update(anomaly_rows(v.time.values, vals, coef, lats, lons))
            print(f"  CDS {yr}: {len(v.time)} days", flush=True)
            _save(rows)
    except Exception as e:                                   # noqa: BLE001
        print(f"  CDS part failed/skipped ({repr(e)[:90]}); series ends at WB2 unless "
              "the monitor has been appending", file=sys.stderr)

    _save(rows)
    return 0


def _save(rows: dict):
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[[r for r in REGION_ORDER if r in df.columns]]
    df.sort_index().round(3).to_csv(SERIES_CSV, index_label="date")
    print(f"  saved {SERIES_CSV.name}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    sys.exit(main())
