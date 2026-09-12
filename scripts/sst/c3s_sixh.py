#!/usr/bin/env python3
"""6-hourly members for every C3S system: the fetch the daily impact products need.

The monthly members already on disk answer "how warm is the month". They cannot answer
"how many days below freezing", "how long is the cold run", "how many windy days" -- those
are daily questions and need the sub-daily fields. `seasonal-original-single-levels`
carries them for all nine centres, but only inside a box: a global 6-hourly pull for eight
systems is not a thing anyone should do, and the products that use it are regional anyway
(the United States and Brazil).

Sizes measured on the ECMWF card this mirrors: ~120 MB per country per issue for 51
members. Systems with 120+ members cost proportionally more, so a full issue across seven
more systems and two countries is a few gigabytes -- hence: one region-month chunk per
request, resumable, skip what is already on disk, and a --probe mode that fetches exactly
one chunk so the cost is measured before it is spent.

    SST_SITE_ROOT=. python scripts/sst/c3s_sixh.py --probe --only ukmo_610
    SST_SITE_ROOT=. python scripts/sst/c3s_sixh.py --issue 202609 --regions us,br
-> scripts/sst/data/c3s/sixh/{centre}_{system}_{region}_{issue}_m{k}.grib
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, _client
from seas5_popT import REGIONS, month_hours
from seas5_popT import chunk_path as seas5_chunk_path

STORE = HERE / "data" / "c3s" / "sixh"
DATASET = "seasonal-original-single-levels"
MONTHS = 6
# variable set per product family; t2m alone unlocks the daily temperature products, which
# is where this starts. Winds and snowfall are their own (equally large) pulls.
VARS = {"t2m": ["2m_temperature"],
        "wind": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "snow": ["snowfall"]}


def path_for(centre: str, system: str, region: str, issue: str, k: int, var: str = "t2m") -> Path:
    if centre == "ecmwf" and var == "t2m":
        # the SEAS5 card already holds exactly this chunk; never fetch it twice
        p = seas5_chunk_path(region, issue, k)
        if p.exists():
            return p
    stem = f"{centre}_{system}_{region}_{issue}_m{k}" + ("" if var == "t2m" else f"_{var}")
    return STORE / f"{stem}.grib"


def start_dates(centre: str, system: str, issue: str):
    """The start dates this system's members carry, read off the monthly file already cached.

    A single-start system (ECMWF) returns one; a LAGGED one returns many -- UKMO's September
    issue is 62 members spread over 31 start days in August. The sub-daily collection exposes
    those starts individually (asking for day 01 alone returned 2 of the 62 members), so the
    request has to name them, and the lead window has to be widened to cover the target month
    from the earliest of them.
    """
    import xarray as xr
    from c3s_terciles import path_for as terc_path
    p = terc_path("fc", centre, system, "t2m", issue)
    if not p.exists():
        return []
    try:
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
    except Exception:                                                 # noqa: BLE001
        return []
    out = sorted({str(t)[:10] for t in np.atleast_1d(ds["time"].values)})
    ds.close()
    return out


def fetch_chunk(centre: str, system: str, region: str, issue: str, k: int, var: str = "t2m") -> bool:
    dest = path_for(centre, system, region, issue, k, var)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    STORE.mkdir(parents=True, exist_ok=True)
    hours = month_hours(issue, k)
    starts = start_dates(centre, system, issue)
    if len(starts) > 1:
        # lagged ensemble: one request per start MONTH, every start day in it, and a lead
        # window widened by the spread of the starts so the target month is covered from
        # the earliest of them. Alignment is done later on valid time, which the file carries.
        ym = starts[0][:7].replace("-", "")
        days = sorted({d[8:10] for d in starts if d[:7] == starts[0][:7]})
        back = (dt.date(int(issue[:4]), int(issue[4:6]), 1)
                - dt.date(*(int(x) for x in starts[0].split("-")))).days
        lo, hi = int(hours[0]) + back * 24, int(hours[-1]) + back * 24
        hours = [str(h) for h in range(max(lo - 744, 6), hi + 1, 6)]
    else:
        ym, days = issue, ["01"]
    req = {"originating_centre": centre, "system": str(system), "variable": VARS[var],
           "year": [ym[:4]], "month": [ym[4:6]], "day": days, "leadtime_hour": hours,
           "area": REGIONS[region][2], "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(".part")
    t0 = time.time()
    try:
        _client().retrieve(DATASET, req, str(tmp))
    except Exception as e:                                            # noqa: BLE001
        msg = str(e).replace("\n", " ")
        print(f"  {centre}/{system} {region} m{k} {var}: FAILED {msg[:110]}", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    if not (tmp.exists() and tmp.stat().st_size > 0):
        return False
    os.replace(tmp, dest)
    print(f"  {centre}/{system} {region} m{k} {var}: {dest.stat().st_size / 1e6:.0f} MB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--regions", default="us,br")
    ap.add_argument("--only", default="", help="centre_system")
    ap.add_argument("--var", default="t2m", choices=sorted(VARS))
    ap.add_argument("--months", type=int, default=MONTHS)
    ap.add_argument("--probe", action="store_true", help="one chunk only, to measure the cost")
    a = ap.parse_args()
    regions = [r for r in a.regions.split(",") if r in REGIONS]
    got = need = 0
    t0 = time.time()
    for entry in MODELS:
        centre, system = entry[0], str(entry[1])
        if a.only and f"{centre}_{system}" != a.only:
            continue
        for region in regions:
            for k in range(1, a.months + 1):
                need += 1
                got += bool(fetch_chunk(centre, system, region, a.issue, k, a.var))
                if a.probe:
                    print(f"probe: {got}/{need} chunk(s) in {(time.time() - t0) / 60:.1f} min")
                    return 0 if got else 1
    print(f"{got}/{need} chunk(s) ready under {STORE} in {(time.time() - t0) / 60:.1f} min")
    return 0 if got >= 0.75 * need else 1


if __name__ == "__main__":
    raise SystemExit(main())
