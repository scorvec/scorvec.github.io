#!/usr/bin/env python3
"""
Print the most recent 00/06/12/18Z HRRR extended cycle whose **F48** GRIB is
actually published on AWS, probed live via Herbie.

Unlike scripts/find_latest_extended_cycle.py (a clock-only latency guess that
assumes the set has landed N hours after init), this returns a cycle only once
its full f00..f48 set is really on AWS: F48 is the LAST file HRRR writes, so its
presence means the set is complete. That lets the synoptic poll render as soon
as the data is ready — sooner than the conservative 2.5 h guess, and never on a
half-published cycle.

Walks back from the most-recent-passed extended cycle and returns the first one
whose F48 exists (so a just-passed cycle that isn't ready yet falls through to
the previous, already-rendered cycle, which the caller no-ops on).

Exit 0 + prints the cycle if one is found in the lookback window; exit 1 with no
output if none is available yet (caller should no-op and re-poll).

Usage in run_local_synoptic.sh:
    ISO=$(python latest_available_cycle.py --format iso) || { echo "waiting"; exit 0; }
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime, timedelta, timezone

EXTENDED_HOURS = (0, 6, 12, 18)


def _candidates(now: datetime, max_lookback_hours: float):
    """Yield extended cycles, most-recent-passed first, within the lookback."""
    base = now.replace(minute=0, second=0, microsecond=0)
    for back in range(int(max_lookback_hours) + 1):
        c = base - timedelta(hours=back)
        if c.hour in EXTENDED_HOURS:
            yield c


def latest_available(max_lookback_hours: float = 24,
                     priority=("aws",),
                     now: datetime | None = None) -> datetime | None:
    """Most recent extended cycle whose F48 GRIB is on AWS, or None."""
    from herbie import Herbie

    now = now or datetime.now(timezone.utc)
    # Herbie chatter (and any source-probe prints) must not pollute stdout, which
    # the shell captures for the cycle string — route it all to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        for c in _candidates(now, max_lookback_hours):
            try:
                H = Herbie(c.strftime("%Y-%m-%d %H:00"), model="hrrr",
                           product="sfc", fxx=48, priority=list(priority),
                           verbose=False)
                if H.grib is not None:
                    return c
            except Exception as e:  # noqa: BLE001 — any probe failure ⇒ try older
                print(f"  probe {c:%Y-%m-%d %HZ} failed: {e!r}", file=sys.stderr)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-lookback-hours", type=float, default=24,
                   help="How far back to probe for an available extended cycle.")
    p.add_argument("--priority", default="aws",
                   help="Comma-separated Herbie source priority (default: aws).")
    p.add_argument("--format", default="iso",
                   choices=["iso", "compact", "filename"],
                   help="iso=2026-04-30T18:00, compact=20260430T18Z, "
                        "filename=20260430_18 (filesystem-safe)")
    args = p.parse_args(argv)

    cycle = latest_available(max_lookback_hours=args.max_lookback_hours,
                             priority=tuple(args.priority.split(",")))
    if cycle is None:
        return 1  # nothing ready yet; caller no-ops

    cycle = cycle.replace(tzinfo=None)
    if args.format == "iso":
        print(cycle.strftime("%Y-%m-%dT%H:00"))
    elif args.format == "compact":
        print(cycle.strftime("%Y%m%dT%HZ"))
    elif args.format == "filename":
        print(cycle.strftime("%Y%m%d_%H"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
