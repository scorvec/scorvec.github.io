#!/usr/bin/env python3
"""6-hourly 2 m temperature members for every C3S system: the fetch behind the
population-weighted daily card (c3s_popt_daily).

The monthly members already on disk answer "how warm is the month"; a daily series needs
the sub-daily field. `seasonal-original-single-levels` carries it for all nine centres, but
only inside a box: a global 6-hourly pull for eight systems is not a thing anyone should do,
and the card is regional anyway (the United States and Brazil).

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
import c3s_ref as CR
from seas5_popT import chunk_path as seas5_chunk_path

STORE = HERE / "data" / "c3s" / "sixh"
DATASET = "seasonal-original-single-levels"
MONTHS = 6


def path_for(centre: str, system: str, region: str, issue: str, k: int) -> Path:
    if centre == "ecmwf":
        # the SEAS5 card already holds exactly this chunk; never fetch it twice
        p = seas5_chunk_path(region, issue, k)
        if p.exists():
            return p
    return STORE / f"{centre}_{system}_{region}_{issue}_m{k}.grib"


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
    import c3s_ref as CR

    t = CR.load(CR.is_terc(centre, system, "t2m", issue))  # the tercile build keeps them (raw deleted)
    if t is not None and "starts" in t:
        return [str(x) for x in t["starts"]]
    p = terc_path("fc", centre, system, "t2m", issue)
    if not p.exists():
        return []
    try:
        ds = xr.open_dataset(
            p, engine="cfgrib", backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")}
        )
    except Exception:  # noqa: BLE001
        return []
    out = sorted({str(t)[:10] for t in np.atleast_1d(ds["time"].values)})
    ds.close()
    return out


def fetch_chunk(centre: str, system: str, region: str, issue: str, k: int) -> bool:
    dest = path_for(centre, system, region, issue, k)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    STORE.mkdir(parents=True, exist_ok=True)
    hours = month_hours(issue, k)
    starts = start_dates(centre, system, issue)
    if not starts:
        # without the start dates a lagged system's request would name day 01 alone and come back
        # with a sliver of its members (2 of UKMO's 62) -- wait for the tercile build's table
        print(f"  {centre}/{system} {region} m{k}: start dates unknown (no t2m members yet); skipped", flush=True)
        return False
    if len(starts) > 1:
        # lagged ensemble: one request per start MONTH, every start day in it, and a lead
        # window widened by the spread of the starts so the target month is covered from
        # the earliest of them. Alignment is done later on valid time, which the file carries.
        ym = starts[0][:7].replace("-", "")
        days = sorted({d[8:10] for d in starts if d[:7] == starts[0][:7]})
        back = (dt.date(int(issue[:4]), int(issue[4:6]), 1) - dt.date(*(int(x) for x in starts[0].split("-")))).days
        lo, hi = int(hours[0]) + back * 24, int(hours[-1]) + back * 24
        hours = [str(h) for h in range(max(lo - 744, 6), hi + 1, 6)]
    else:
        ym, days = issue, ["01"]
    req = {
        "originating_centre": centre,
        "system": str(system),
        "variable": ["2m_temperature"],
        "year": [ym[:4]],
        "month": [ym[4:6]],
        "day": days,
        "leadtime_hour": hours,
        "area": REGIONS[region][2],
        "grid": [1.0, 1.0],
        "data_format": "grib",
    }
    tmp = dest.with_suffix(".part")
    t0 = time.time()
    try:
        _client().retrieve(DATASET, req, str(tmp))
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        print(f"  {centre}/{system} {region} m{k}: FAILED {msg[:100]}", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    if not (tmp.exists() and tmp.stat().st_size > 0):
        return False
    os.replace(tmp, dest)
    print(
        f"  {centre}/{system} {region} m{k}: {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--regions", default="us,br")
    ap.add_argument("--only", default="", help="centre_system")
    ap.add_argument("--months", type=int, default=MONTHS)
    ap.add_argument("--probe", action="store_true", help="one chunk only, to measure the cost")
    ap.add_argument("--order", default="", help="comma list of centre_system, fetched in this order")
    ap.add_argument(
        "--budget-min", type=float, default=0.0, help="stop STARTING new chunks after this many minutes (0 = no limit)"
    )
    a = ap.parse_args()
    regions = [r for r in a.regions.split(",") if r in REGIONS]
    got = need = 0
    t0 = time.time()
    order = [x.strip() for x in a.order.split(",") if x.strip()]
    entries = list(MODELS)
    if order:  # cheapest systems first, so value accrues early
        entries.sort(key=lambda e: order.index(f"{e[0]}_{e[1]}") if f"{e[0]}_{e[1]}" in order else len(order))
    for entry in entries:
        centre, system = entry[0], str(entry[1])
        if a.only and f"{centre}_{system}" != a.only:
            continue
        for region in regions:
            for k in range(1, a.months + 1):
                need += 1
                if CR.is_sixh(centre, system, region, a.issue, k).exists():
                    got += 1  # already reduced to its table (c3s_popt_daily.daily_for)
                    continue
                if a.budget_min > 0 and (time.time() - t0) / 60 >= a.budget_min:
                    print(f"  budget of {a.budget_min:.0f} min spent; the rest is left for the next run", flush=True)
                    print(f"{got}/{need} chunk(s) ready (budget stop)")
                    return 0 if got >= 0.75 * need else 1
                ok = fetch_chunk(centre, system, region, a.issue, k)
                if ok:
                    # reduce at once to the daily card's table and drop the raw chunk (Actions)
                    from c3s_popt_daily import daily_for

                    vals, _days = daily_for(centre, system, region, a.issue, k)
                    ok = vals is not None
                    p = path_for(centre, system, region, a.issue, k)
                    if ok and CR.discard_raw() and STORE in p.parents:
                        p.unlink(missing_ok=True)
                got += ok
                if a.probe:
                    print(f"probe: {got}/{need} chunk(s) in {(time.time() - t0) / 60:.1f} min")
                    return 0 if got else 1
    print(f"{got}/{need} chunk(s) ready under {STORE} in {(time.time() - t0) / 60:.1f} min")
    return 0 if got >= 0.75 * need else 1


if __name__ == "__main__":
    raise SystemExit(main())
