#!/usr/bin/env python3
"""
Print the most recent 00/06/12/18Z HRRR extended-run cycle that is
likely available on AWS, in ISO-8601 format suitable for `run.py --cycle`.

Extended runs (00/06/12/18Z) go out to fxx=48; off-cycle runs cap at 18.
Output is naive UTC, e.g. "2026-04-30T18:00".

Usage in CI:
    CYCLE=$(python find_latest_extended_cycle.py)
    python run.py --cycle "$CYCLE" --fxx-end 48 ...

Tunables:
    --latency-hours   How long to wait after init before assuming the
                      full f00..f48 set has landed on AWS. Default 2.5h
                      (HRRR full forecast completes ~70 min after init,
                      AWS sync adds ~30 min).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

EXTENDED_HOURS = (0, 6, 12, 18)


def latest_extended_cycle(latency_hours: float = 2.5,
                          now: datetime | None = None) -> datetime:
    """Find the most recent 00/06/12/18Z cycle older than `latency_hours`."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=latency_hours)

    # Walk back day by day, then hour by hour, until we find one ≤ cutoff
    for day_offset in range(3):
        d = (cutoff - timedelta(days=day_offset)).date()
        for h in sorted(EXTENDED_HOURS, reverse=True):
            candidate = datetime(d.year, d.month, d.day, h,
                                 tzinfo=timezone.utc)
            if candidate <= cutoff:
                return candidate

    raise RuntimeError("No extended cycle found within the search window")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latency-hours", type=float, default=2.5,
                   help="Wait this many hours after cycle init before "
                        "assuming GRIBs are available")
    p.add_argument("--format", default="iso",
                   choices=["iso", "compact", "filename"],
                   help="iso=2026-04-30T18:00, compact=20260430T18Z, "
                        "filename=20260430_18 (filesystem-safe)")
    args = p.parse_args(argv)

    cycle = latest_extended_cycle(latency_hours=args.latency_hours)
    cycle_naive = cycle.replace(tzinfo=None)

    if args.format == "iso":
        print(cycle_naive.strftime("%Y-%m-%dT%H:00"))
    elif args.format == "compact":
        print(cycle_naive.strftime("%Y%m%dT%HZ"))
    elif args.format == "filename":
        print(cycle_naive.strftime("%Y%m%d_%H"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
