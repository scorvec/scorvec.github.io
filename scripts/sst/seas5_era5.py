#!/usr/bin/env python3
"""ERA5 monthly means as the observed reference for the SEAS5 page — pulled once.

Two products need an observed record next to the model:

  * Normals. The tercile and anomaly maps are, by default, relative to SEAS5's
    own 1993–2016 hindcast. To express them against an observed 30-year normal
    (1991–2020), a 10-year normal (2016–2025) and a trend extrapolated to the
    forecast year, we need ERA5 monthly 2 m temperature, precipitation, 500 hPa
    height, 10 m wind speed and surface solar radiation over the Americas at 1°
    for 1991–2025.
  * Skill. Hindcast teleconnection and stratospheric indices are scored against
    the ERA5 equivalents (NAO, PNA, AO, vortex wind, QBO, SOI), so each plume can
    carry the model's own out-of-sample correlation for that month.

Both are single retrievals per dataset, cached under scripts/sst/data/seas5/era5/.

    python scripts/sst/seas5_era5.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import DATA, _client  # noqa: E402

ERA5 = DATA / "era5"
YEARS = [str(y) for y in range(1991, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

PULLS = {
    # Americas, 1°: the fields the tercile / anomaly maps use
    "am_sfc": dict(dataset="reanalysis-era5-single-levels-monthly-means",
                   req=dict(product_type=["monthly_averaged_reanalysis"],
                            variable=["2m_temperature", "total_precipitation", "10m_wind_speed", "surface_solar_radiation_downwards"],
                            year=YEARS, month=MONTHS, time=["00:00"], area=[75, -170, -60, -30], grid=[1.0, 1.0], data_format="grib")),
    "am_z500": dict(dataset="reanalysis-era5-pressure-levels-monthly-means",
                    req=dict(product_type=["monthly_averaged_reanalysis"], variable=["geopotential"], pressure_level=["500"],
                             year=YEARS, month=MONTHS, time=["00:00"], area=[75, -170, -60, -30], grid=[1.0, 1.0], data_format="grib")),
    # evaporation for P − E (kept separate so the first pull's request stays cached)
    "am_e": dict(dataset="reanalysis-era5-single-levels-monthly-means",
                 req=dict(product_type=["monthly_averaged_reanalysis"], variable=["evaporation"],
                          year=YEARS, month=MONTHS, time=["00:00"], area=[75, -170, -60, -30], grid=[1.0, 1.0], data_format="grib")),
    # global 2°: teleconnection and stratosphere references
    "gl_sfc": dict(dataset="reanalysis-era5-single-levels-monthly-means",
                   req=dict(product_type=["monthly_averaged_reanalysis"], variable=["mean_sea_level_pressure", "sea_surface_temperature"],
                            year=YEARS, month=MONTHS, time=["00:00"], area=[90, -180, -90, 180], grid=[2.0, 2.0], data_format="grib")),
    "gl_pl": dict(dataset="reanalysis-era5-pressure-levels-monthly-means",
                  req=dict(product_type=["monthly_averaged_reanalysis"], variable=["geopotential", "u_component_of_wind"],
                           pressure_level=["10", "30", "50", "100", "200", "500", "1000"],
                           year=YEARS, month=MONTHS, time=["00:00"], area=[90, -180, -90, 180], grid=[2.0, 2.0], data_format="grib")),
}


def path(key: str) -> Path:
    return ERA5 / f"era5_{key}_1991-2025.grib"


def fetch(keys=None) -> dict:
    ERA5.mkdir(parents=True, exist_ok=True)
    got = {}
    for key in keys or PULLS:
        dest = path(key)
        if dest.exists() and dest.stat().st_size > 0:
            got[key] = True; continue
        spec = PULLS[key]; tmp = dest.with_suffix(f".part{os.getpid()}")
        ok = False
        for attempt in range(3):
            t0 = time.time()
            try:
                print(f"  ERA5 {key} …", flush=True)
                _client().retrieve(spec["dataset"], spec["req"], str(tmp))
                if tmp.exists() and tmp.stat().st_size > 0:
                    os.replace(tmp, dest); ok = True
                    print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                    break
            except Exception as e:                                # noqa: BLE001
                print(f"    {key}: attempt {attempt + 1} failed ({str(e)[:120]})", flush=True)
                time.sleep(30)
        got[key] = ok
    return got


if __name__ == "__main__":
    got = fetch(sys.argv[1:] or None)
    print("ERA5:", got)
