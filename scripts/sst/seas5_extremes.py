#!/usr/bin/env python3
"""Threshold-day frequencies from SEAS5's 6-hourly members, as % of the observed normal.

North America (CONUS box): cold days, daily minimum 2 m temperature at or below 0, −10 and
−20 °C. Brazil: hot days, daily maximum above 35, 37 and 40 °C (user, 2026-09-07). For every
member and month the count of threshold days per grid cell is compared with the CPC
1991–2020 frequency of the same event on the same calendar month, after shifting each
member by the model's monthly mean bias (hindcast month mean minus the CPC (Tmax+Tmin)/2
month mean at the cell, 1993–2016) so a warm-biased model does not read as
fewer freezes. Products: per-month maps of % of normal, and population-weighted national
series (expected threshold days per month, normal, member spread).

Data:
  * SEAS5 seasonal-original-single-levels, maximum/minimum 2 m temperature in the last 24 h,
    6-hourly, in the same monthly chunks as seas5_popT (files `{region}_{ym}_m{k}_x.grib`).
  * NOAA CPC Global Unified daily Tmax/Tmin (0.5°, land), 1991–2020, the two boxes via PSL
    OPeNDAP, cached under data/seas5/cpc/{tx|tn}_{region}_{year}.nc. (The local ERA5 store
    has daily MEANS only, and the CDS derived-daily ERA5 jobs failed server-side, 2026-09-07.)

    python seas5_extremes.py fetch [--issue 202609]     # SEAS5 chunks, this issue + previous
    python seas5_extremes.py cpc   [--region us|br]     # the CPC normals, once (~1 h)
    python seas5_extremes.py build [--issue 202609]
"""
from __future__ import annotations
import argparse, calendar, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, DATA, CENTRE, SYSTEM, CLIM_YEARS, _client, previous_issues   # noqa: E402
from seas5_popT import REGIONS, SIXH, MONTHS, month_hours                        # noqa: E402

ERA5D = DATA / "era5" / "daily"
YEARS = list(range(1991, 2021))
# threshold sets per region: (set key, daily statistic, label, thresholds °C, operator); more thresholds
# and both tails per region (user 2026-09-07: "add more temperature thresholds for these regions")
THRESH = {
    "us": [("cold", "tn", "cold days, T min ≤", [5, 0, -5, -10, -15, -20], "le"),
           ("hot", "tx", "hot days, T max ≥", [30, 35, 38, 40], "ge")],
    "br": [("hot", "tx", "hot days, T max ≥", [30, 32, 35, 37, 40], "ge"),
           ("cool", "tn", "cool nights, T min ≤", [15, 10, 5], "le")],
}


def xchunk_path(region: str, ym: str, k: int) -> Path:
    return SIXH / f"{region}_{ym}_m{k}_x.grib"


def fetch_chunk(region: str, ym: str, k: int) -> bool:
    dest = xchunk_path(region, ym, k)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    hours = [h for h in month_hours(ym, k) if int(h) <= 5160]
    req = {"originating_centre": CENTRE, "system": SYSTEM,
           "variable": ["maximum_2m_temperature_in_the_last_24_hours", "minimum_2m_temperature_in_the_last_24_hours"],
           "year": [ym[:4]], "month": [ym[4:]], "day": ["01"], "leadtime_hour": hours,
           "area": REGIONS[region][2], "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS extremes {region} {ym} month {k} ({len(hours)} steps) …", flush=True)
            _client().retrieve("seasonal-original-single-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower() or "not found" in msg.lower():
                print(f"    {region} {ym} m{k}: no data on the CDS ({msg[:80]})", flush=True); return False
            print(f"    {region} {ym} m{k}: attempt {attempt + 1} failed ({msg[:120]})", flush=True)
            time.sleep(30)
    return False


def hchunk_path(region: str, month: str, k: int) -> Path:
    return SIXH / f"hc_{region}_{month}_m{k}_x.grib"


def day_hours(ym: str, k: int) -> list[str]:
    """The 24-hour extremes exist at daily steps only: one step per day of forecast month k."""
    hrs = [int(h) for h in month_hours(ym, k) if int(h) <= 5160]
    return [str(h) for h in hrs if h % 24 == 0]


def fetch_hindcast(region: str, ym: str, k: int) -> bool:
    """SEAS5 hindcast (1993–2016, 25 members) daily max/min for forecast month k of this start
    month — the model's own Tmin/Tmax climatology, so thresholds can be quantile-mapped per cell
    (the mean bias understates the Tmin bias: SEAS5's diurnal range is too small, 2026-09-07)."""
    dest = hchunk_path(region, ym[4:], k)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    hours = day_hours(ym, k)
    req = {"originating_centre": CENTRE, "system": SYSTEM,
           "variable": ["maximum_2m_temperature_in_the_last_24_hours", "minimum_2m_temperature_in_the_last_24_hours"],
           "year": list(CLIM_YEARS), "month": [ym[4:]], "day": ["01"], "leadtime_hour": hours,
           "area": REGIONS[region][2], "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS hindcast extremes {region} start {ym[4:]} month {k} ({len(hours)} days × {len(CLIM_YEARS)} yr) …", flush=True)
            _client().retrieve("seasonal-original-single-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).replace("\n", " ")
            print(f"    hindcast {region} {ym[4:]} m{k}: attempt {attempt + 1} failed ({msg[:120]})", flush=True)
            time.sleep(30)
    return False


def fetch(ym: str, regions=None, issues=None) -> dict:
    got = {}
    for iss in issues or (ym, previous_issues(ym, 1)[0]):
        for region in regions or list(REGIONS):
            for k in range(1, MONTHS + 1):
                got[(iss, region, k)] = fetch_chunk(region, iss, k)
    for region in regions or list(REGIONS):
        for k in range(1, MONTHS + 1):
            got[("hc", region, k)] = fetch_hindcast(region, ym, k)
    return got


CPC = DATA / "cpc"


def cpc_path(stat: str, region: str, year: int) -> Path:
    return CPC / f"{stat}_{region}_{year}.nc"


def fetch_cpc(regions=None, years=None, stats=None) -> dict:
    """NOAA CPC Global Unified daily Tmax / Tmin (0.5°, land only, 1979–), subset over the region
    boxes through PSL's OPeNDAP, one (stat, region, year) at a time — sequential on purpose:
    parallel PSL reads have returned silently corrupt arrays before. Each file is validated
    (shape, NaN share, plausible range) before it is kept. ~30 s per year and box."""
    import xarray as xr
    got = {}
    CPC.mkdir(parents=True, exist_ok=True)
    for region in regions or list(REGIONS):
        n_, w_, s_, e_ = REGIONS[region][2]
        for year in years or YEARS:
            for stat in stats or ("tx", "tn"):
                dest = cpc_path(stat, region, year)
                if dest.exists() and dest.stat().st_size > 0:
                    got[(stat, region, year)] = True; continue
                var = "tmax" if stat == "tx" else "tmin"; ok = False
                for attempt in range(3):
                    t0 = time.time()
                    try:
                        ds = xr.open_dataset(f"https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/{var}.{year}.nc")
                        da = ds[var].sel(lat=slice(n_, s_), lon=slice(w_ % 360, e_ % 360)).load(); ds.close()
                        a = da.values; nan = float(np.isnan(a).mean()); mx = float(np.nanmax(a)); mn = float(np.nanmin(a))
                        if a.shape[0] < 365 or nan > 0.95 or nan < 0.02 or mx > 60 or mn < -80 or not np.isfinite(mx):
                            raise ValueError(f"implausible: shape {a.shape} nan {nan:.2f} range {mn:.0f}..{mx:.0f}")
                        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
                        da.to_dataset(name=var).to_netcdf(dest); ok = True
                        print(f"  CPC {var} {region} {year}: {a.shape} nan {nan:.2f} max {mx:.1f} in {time.time() - t0:.0f}s", flush=True)
                        break
                    except Exception as e:                        # noqa: BLE001
                        print(f"    CPC {var} {region} {year}: attempt {attempt + 1} failed ({str(e)[:120]})", flush=True)
                        time.sleep(20)
                got[(stat, region, year)] = ok
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "cpc", "build"])
    ap.add_argument("--issue", default=None)
    ap.add_argument("--region", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--stat", nargs="*", default=None, help="tx and/or tn")
    a = ap.parse_args(argv)
    import datetime as _dt
    ym = a.issue or _dt.datetime.utcnow().strftime("%Y%m")
    if a.cmd == "fetch":
        got = fetch(ym, a.region); bad = [k for k, v in got.items() if not v]
        print(f"fetch {ym}: {len(got) - len(bad)} ok, {len(bad)} missing {bad}", flush=True)
    elif a.cmd == "cpc":
        got = fetch_cpc(a.region, a.years, a.stat); bad = [k for k, v in got.items() if not v]
        print(f"cpc: {len(got) - len(bad)} ok, {len(bad)} missing {bad}", flush=True)
    else:
        from seas5_extremes_build import build                    # written separately
        build(ym)
    return 0


if __name__ == "__main__":
    sys.exit(main())
