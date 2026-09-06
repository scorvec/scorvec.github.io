#!/usr/bin/env python3
"""ECMWF SEAS5 seasonal outlook page — one system, every read we care about.

Built for the September 2026 issue while the rest of the C3S multi-system set
was still unpublished (the multi-model page, c3s_nino34.py, waits for a quorum
of centres; this one does not). Everything is SEAS5 (C3S originating_centre
ecmwf, system 51): 51 members for the forecast, 25 for the 1993–2016 hindcast.

What it makes, per issue month:

  1. ENSO index plumes, this issue against the previous three issues:
       Niño-1+2, Niño-3, Niño-3.4, Niño-4, relative Niño-3.4 (RONI-scaled,
       same method as the multi-model page), and the Trans-Niño index
       (Trenberth & Stepaniak 2001: standardised Niño-1+2 minus Niño-4, the
       east–west gradient). Each anomaly is the member minus SEAS5's own
       hindcast climatology for the same start month and lead, so drift is
       removed exactly as on the multi-model page.
  2. Other SST indices from the same fields: PDO (projection of the North
       Pacific anomaly onto the ERSST EOF in reference/pdo_pattern.nc, global
       mean removed, NCEI scale), AMO (North Atlantic 0–60°N mean minus the
       60°S–60°N global mean — the "relative" form, so warming does not read
       as AMO), IOD (west minus east dipole), Atlantic Niño (ATL3).
  3. Calibrated tercile probabilities of 2 m temperature, precipitation and
       500 hPa height over North and South America, for three overlapping
       seasons. Terciles are the model's own: thresholds come from the 600
       hindcast samples (24 years × 25 members) at each grid point, season and
       lead, so a member is counted against what SEAS5 itself does in a normal
       year — the standard first-order calibration that removes mean bias and
       spread bias per point. Shown CPC-style: most-likely category, shaded by
       its probability, blank where no category clears 40 %.
  4. Stratosphere: polar-cap (60–90°) geopotential-height anomalies at 100, 50
       and 10 hPa for both hemispheres, per lead, against the hindcast.

Data cache: scripts/sst/data/seas5/ (gitignored). Outputs: assets/sst/seas5_*.webp
and assets/sst/data/seas5_outlook.json.

    python scripts/sst/seas5_outlook.py fetch  [--issue 202609] [--previous 3]
    python scripts/sst/seas5_outlook.py build  [--issue 202609]
    python scripts/sst/seas5_outlook.py        # fetch, then build

Requires ~/.cdsapirc. Hindcast pulls are large (hundreds of MB per start month)
and queue on the CDS for a long time; they are cached forever once down.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "seas5"
SIGMA_PATH = HERE / "roni_sigma.json"
PDO_PATTERN = HERE / "reference" / "pdo_pattern.nc"

CENTRE, SYSTEM = "ecmwf", "51"
LEADS = ["1", "2", "3", "4", "5", "6"]          # leadtime_month 1 = the start month itself
MAXLEAD = 6
CLIM_YEARS = [str(y) for y in range(1993, 2017)]

# ── what to pull. One entry per (dataset, region); the hindcast reuses the same
# request with the year list swapped. Areas are [N, W, S, E] in −180..180.
KINDS = {
    # global SST on the 2° grid: every index is a box mean or a 2° pattern projection
    "sst": dict(dataset="seasonal-monthly-single-levels",
                variable=["sea_surface_temperature"], area=[70, -180, -70, 180], grid=[2.0, 2.0]),
    # the Americas at 1°: 75°N–60°S, 170°W–30°W
    "sfc": dict(dataset="seasonal-monthly-single-levels",
                variable=["2m_temperature", "total_precipitation"], area=[75, -170, -60, -30], grid=[1.0, 1.0]),
    "z500": dict(dataset="seasonal-monthly-pressure-levels",
                 variable=["geopotential"], pressure_level=["500"], area=[75, -170, -60, -30], grid=[1.0, 1.0]),
    # polar caps, 2° is plenty for a cap mean
    "polar_n": dict(dataset="seasonal-monthly-pressure-levels",
                    variable=["geopotential"], pressure_level=["10", "50", "100"], area=[90, -180, 60, 180], grid=[2.0, 2.0]),
    "polar_s": dict(dataset="seasonal-monthly-pressure-levels",
                    variable=["geopotential"], pressure_level=["10", "50", "100"], area=[-60, -180, -90, 180], grid=[2.0, 2.0]),
    # teleconnections (added 2026-09-06): NH + tropics heights at 500/1000 and msl on 2°
    "nh_z": dict(dataset="seasonal-monthly-pressure-levels",
                 variable=["geopotential"], pressure_level=["500", "1000"], area=[90, -180, -30, 180], grid=[2.0, 2.0]),
    "nh_msl": dict(dataset="seasonal-monthly-single-levels",
                   variable=["mean_sea_level_pressure"], area=[90, -180, -30, 180], grid=[2.0, 2.0]),
    # stratospheric zonal wind: vortex (60°N/S at 10 hPa) and the QBO (equatorial 10–50 hPa)
    "strat_u": dict(dataset="seasonal-monthly-pressure-levels",
                    variable=["u_component_of_wind"], pressure_level=["10", "30", "50"], area=[90, -180, -90, 180], grid=[2.0, 2.0]),
    # subtropical jet over the Pacific and the Americas
    "u200": dict(dataset="seasonal-monthly-pressure-levels",
                 variable=["u_component_of_wind"], pressure_level=["200"], area=[75, -180, -60, -30], grid=[2.0, 2.0]),
    # energy fields over the Americas at 1°
    "energy": dict(dataset="seasonal-monthly-single-levels",
                   variable=["10m_wind_speed", "surface_solar_radiation_downwards"], area=[75, -170, -60, -30], grid=[1.0, 1.0]),
    # water balance: evaporation (negative upward, m of water per day) for P − E drought maps
    "water": dict(dataset="seasonal-monthly-single-levels",
                  variable=["evaporation"], area=[75, -170, -60, -30], grid=[1.0, 1.0]),
}
# the immediately previous issue is pulled in full (change maps, polar-cap
# comparison); older issues only need SST (the plume-evolution lines)
PREVIOUS_KINDS = ["sst"]
PREVIOUS_FULL = list(KINDS)


def _client():
    import cdsapi
    return cdsapi.Client(timeout=900, quiet=True, progress=False, wait_until_complete=True, retry_max=1)


def _retrieve(kind: str, years: list[str], month: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    k = KINDS[kind]
    req = {"originating_centre": CENTRE, "system": SYSTEM, "product_type": ["monthly_mean"],
           "variable": k["variable"], "year": list(years), "month": [month], "leadtime_month": LEADS,
           "area": k["area"], "grid": k["grid"], "data_format": "grib"}
    if "pressure_level" in k:
        req["pressure_level"] = k["pressure_level"]
    tmp = dest.with_suffix(f".part{os.getpid()}")                    # parallel fetchers never share a partial file
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS {kind} {month} years={years[0]}..{years[-1]} → {dest.name} …", flush=True)
            _client().retrieve(k["dataset"], req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                              # noqa: BLE001
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower() or "not found" in msg.lower():
                print(f"    {kind} {month}: no data on the CDS — skipped ({msg[:80]})", flush=True)
                return False
            print(f"    {kind} {month}: attempt {attempt + 1} failed ({msg[:120]})", flush=True)
            time.sleep(30)
    return False


def fc_path(kind: str, ym: str) -> Path:
    return DATA / "forecast" / f"fc_{kind}_{ym}.grib"


def hc_path(kind: str, month: str) -> Path:
    return DATA / "hindcast" / f"hc_{kind}_{month}.grib"


def previous_issues(ym: str, n: int) -> list[str]:
    y, m = int(ym[:4]), int(ym[4:])
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y}{m:02d}")
    return out


def fetch(ym: str, n_prev: int, kinds=None) -> dict:
    """Pull everything for the issue (and SST for the previous issues). Returns
    {(kind, ym|hc-month): ok}. Order: the issue's forecasts first (cheap, and
    they tell us whether the issue is out at all), then hindcasts, then the
    previous issues."""
    got = {}
    month = ym[4:]
    want = [k for k in KINDS if (kinds is None or k in kinds)]
    for kind in want:
        got[("fc", kind, ym)] = _retrieve(kind, [ym[:4]], month, fc_path(kind, ym))
    if kinds is None and not got[("fc", "sst", ym)]:
        print(f"SEAS5 {ym} is not on the CDS yet — nothing more to do", flush=True)
        return got
    for kind in want:
        got[("hc", kind, month)] = _retrieve(kind, CLIM_YEARS, month, hc_path(kind, month))
    for i, prev in enumerate(previous_issues(ym, n_prev)):
        for kind in [k for k in (PREVIOUS_FULL if i == 0 else PREVIOUS_KINDS) if (kinds is None or k in kinds)]:
            ok = _retrieve(kind, [prev[:4]], prev[4:], fc_path(kind, prev))
            got[("fc", kind, prev)] = ok
            if ok:
                got[("hc", kind, prev[4:])] = _retrieve(kind, CLIM_YEARS, prev[4:], hc_path(kind, prev[4:]))
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["fetch", "build", "all"])
    ap.add_argument("--issue", default=None, help="YYYYMM (default: this month)")
    ap.add_argument("--previous", type=int, default=3, help="earlier issues to draw for the plume evolution")
    ap.add_argument("--kinds", nargs="*", help="fetch only these field kinds (a parallel worker)")
    args = ap.parse_args(argv)
    import datetime as _dt
    ym = args.issue or _dt.datetime.utcnow().strftime("%Y%m")
    if args.cmd in ("fetch", "all"):
        got = fetch(ym, args.previous, args.kinds)
        bad = [k for k, v in got.items() if not v]
        print(f"fetch {ym}: {len(got) - len(bad)} ok, {len(bad)} missing" + (f" {bad}" if bad else ""), flush=True)
    if args.cmd in ("build", "all"):
        from seas5_build import build                         # heavy imports live there
        build(ym, args.previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
