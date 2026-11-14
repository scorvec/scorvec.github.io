#!/usr/bin/env python3
"""6-hourly 2 m temperature members for every C3S system: the fetch behind the
population-weighted daily card (c3s_popt_daily).

The monthly members already on disk answer "how warm is the month"; a daily series needs
the sub-daily field. `seasonal-original-single-levels` carries it for all nine centres, but
only inside a box: a global 6-hourly pull for seven systems is not a thing anyone should do,
and the card is regional anyway (the United States and Brazil).

Sizes measured on the ECMWF card this mirrors: ~120 MB per country per issue for 51
members. Systems with 120+ members cost proportionally more, so a full issue across six
more systems and two countries is a few gigabytes -- hence: region-month chunks, resumable,
skip what is already on disk, and a --probe mode that fetches exactly one chunk so the cost
is measured before it is spent.

A request waits hours for its place in the CDS queue and then runs in minutes, so the
number of requests sets the pace. A system with one start date takes every month still
missing in ONE request per region (six months are 728 fields against the CDS limit of
10000), split here into the month chunks; those systems go first. A lagged system's month
is 7564 fields with every start day in it, so it stays one request per month.

    SST_SITE_ROOT=. python scripts/sst/c3s_sixh.py --probe --only ukmo_610
    SST_SITE_ROOT=. python scripts/sst/c3s_sixh.py --issue 202609 --regions us,br
-> scripts/sst/data/c3s/sixh/{centre}_{system}_{region}_{issue}_m{k}.grib
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, cds_fetch, queue_budget
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


def request(centre: str, system: str, region: str, ym: str, days: list[str], hours: list[str]) -> dict:
    return {
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


def fetch_chunk(centre: str, system: str, region: str, issue: str, k: int, starts: list[str]) -> bool:
    """Forecast month k of a LAGGED system: one request per start MONTH, every start day in it, and a lead window
    widened by the spread of the starts so the target month is covered from the earliest of them. Alignment is done
    later on valid time, which the file carries."""
    dest = path_for(centre, system, region, issue, k)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    hours = month_hours(issue, k)
    ym = starts[0][:7].replace("-", "")
    days = sorted({d[8:10] for d in starts if d[:7] == starts[0][:7]})
    back = (dt.date(int(issue[:4]), int(issue[4:6]), 1) - dt.date(*(int(x) for x in starts[0].split("-")))).days
    lo, hi = int(hours[0]) + back * 24, int(hours[-1]) + back * 24
    hours = [str(h) for h in range(max(lo - 744, 6), hi + 1, 6)]
    req = request(centre, system, region, ym, days, hours)
    return cds_fetch(DATASET, req, dest, f"{centre}/{system} {region} m{k}")


def split_months(src: Path, issue: str, dests: dict[int, Path]) -> None:
    """src's messages written to dests[k], k the forecast month of the issue their valid time belongs to: month k's
    lead hours (month_hours) run from 06 UTC on its first day to 00 UTC on the first day of the next month. A message
    of a month not in dests is dropped, and a month no message came for gets no file."""
    import eccodes

    y0, m0 = int(issue[:4]), int(issue[4:6])
    parts = {k: p.with_name(p.name + ".part") for k, p in dests.items()}
    count = dict.fromkeys(dests, 0)
    with contextlib.ExitStack() as stack:
        out = {k: stack.enter_context(open(p, "wb")) for k, p in parts.items()}
        f = stack.enter_context(open(src, "rb"))
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                stamp = f"{eccodes.codes_get(h, 'validityDate'):08d}{eccodes.codes_get(h, 'validityTime'):04d}"
                valid = dt.datetime.strptime(stamp, "%Y%m%d%H%M") - dt.timedelta(hours=6)
                k = (valid.year - y0) * 12 + valid.month - m0 + 1
                if k in out:
                    out[k].write(eccodes.codes_get_message(h))
                    count[k] += 1
            finally:
                eccodes.codes_release(h)
    for k, part in parts.items():
        if count[k]:
            part.replace(dests[k])
        else:
            part.unlink()


def fetch_months(centre: str, system: str, region: str, issue: str, ks: list[int]) -> bool:
    """Forecast months ks of a system with ONE start date, in one request, split into the month chunks; False when
    the request did not come. The split keeps only the months asked for, so a request taken over from an earlier run
    that holds more months serves too."""
    dests = {k: path_for(centre, system, region, issue, k) for k in ks}
    dests = {k: p for k, p in dests.items() if not (p.exists() and p.stat().st_size > 0)}
    if not dests:
        return True
    hours = sorted({int(h) for k in dests for h in month_hours(issue, k)})
    span = f"m{min(dests)}" if len(dests) == 1 else f"m{min(dests)}-m{max(dests)}"
    src = STORE / f"{centre}_{system}_{region}_{issue}_months.grib"
    src.unlink(missing_ok=True)  # one a stopped run left may hold other months
    req = request(centre, system, region, issue, ["01"], [str(h) for h in hours])
    if not cds_fetch(DATASET, req, src, f"{centre}/{system} {region} {span}"):
        return False
    split_months(src, issue, dests)
    src.unlink()
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
    if a.budget_min > 0:
        queue_budget(a.budget_min)
    order = [x.strip() for x in a.order.split(",") if x.strip()]
    entries = [(e[0], str(e[1])) for e in MODELS if not a.only or f"{e[0]}_{e[1]}" == a.only]
    starts = {e: start_dates(*e, a.issue) for e in entries}
    entries.sort(key=lambda e: len(starts[e]) > 1)  # one start date: a whole region in one request, so those first
    if order:
        entries.sort(key=lambda e: order.index(f"{e[0]}_{e[1]}") if f"{e[0]}_{e[1]}" in order else len(order))
    for centre, system in entries:
        lagged = len(starts[centre, system]) > 1
        for region in regions:
            need += a.months
            ks = [k for k in range(1, a.months + 1) if not CR.is_sixh(centre, system, region, a.issue, k).exists()]
            got += a.months - len(ks)  # already reduced to their tables (c3s_popt_daily.daily_for)
            if not starts[centre, system]:
                # without the start dates a lagged system's request would name day 01 alone and come back with a
                # sliver of its members (2 of UKMO's 62) -- wait for the tercile build's table
                if ks:
                    print(
                        f"  {centre}/{system} {region}: start dates unknown (no t2m members yet); skipped", flush=True
                    )
                continue
            if a.probe:
                ks = ks[:1]
            requests = [[k] for k in ks] if lagged else [ks] if ks else []  # the months each request asks for
            for group in requests:
                if a.budget_min > 0 and (time.time() - t0) / 60 >= a.budget_min:
                    print(f"  budget of {a.budget_min:.0f} min spent; the rest is left for the next run", flush=True)
                    print(f"{got}/{need} chunk(s) ready (budget stop)")
                    return 0 if got >= 0.75 * need else 1
                if lagged:
                    ok = fetch_chunk(centre, system, region, a.issue, group[0], starts[centre, system])
                else:
                    ok = fetch_months(centre, system, region, a.issue, group)
                if ok:  # reduce at once to the daily card's tables
                    from c3s_popt_daily import daily_for

                    got += sum(daily_for(centre, system, region, a.issue, k)[0] is not None for k in group)
                if a.probe:
                    print(f"probe: {got}/{need} chunk(s) in {(time.time() - t0) / 60:.1f} min")
                    return 0 if got else 1
    print(f"{got}/{need} chunk(s) ready under {STORE} in {(time.time() - t0) / 60:.1f} min")
    return 0 if got >= 0.75 * need else 1


if __name__ == "__main__":
    raise SystemExit(main())
