"""Every member of a 10 hPa u(60) ensemble forecast: the members that turn easterly, drawn and counted week by week.

Shared by the days 1-35 vortex figures (scripts/strat/strat_series.py for GEPS, scripts/gefs/gefs_live.py for GEFS)
and the extended SSW index they write (ssw35_<model>.json, read by scripts/strat/ssw_summary.py).

A reversal is a daily u(60) below zero on a northern-winter day (November to March), the WMO wind criterion; the
southern hemisphere and the other months count none (spring's final warming is not a sudden warming).
"""

from __future__ import annotations

import numpy as np

WINTER = (11, 12, 1, 2, 3)
WEEKS = ((1, 7), (8, 14), (15, 21), (22, 28), (29, 35))  # forecast days, day 1 = the first valid day after the start
EASTERLY = "#c0392b"
MEMBER = "#6f87a6"


def first_reversal(times, M, north: bool) -> np.ndarray:
    """Index of each member's first winter easterly day, -1 for none. times: valid times (lead,); M: (lead, member)."""
    M = np.asarray(M, float)
    if not north:
        return np.full(M.shape[1], -1)
    winter = np.array([t.month in WINTER for t in times])
    east = (M < 0) & winter[:, None]
    return np.where(east.any(0), east.argmax(0), -1)


def draw_members(ax, times, M, north: bool) -> None:
    """Every member as a thin line, the reversing ones in a warm colour over the rest, with legend handles."""
    M = np.asarray(M, float)
    rev = first_reversal(times, M, north) >= 0
    for k in np.argsort(rev, kind="stable"):  # reversing members last, on top
        ax.plot(
            times,
            M[:, k],
            color=EASTERLY if rev[k] else MEMBER,
            lw=1.1 if rev[k] else 0.7,
            alpha=0.9 if rev[k] else 0.55,
            zorder=2,
        )
    ax.plot([], [], color=MEMBER, lw=0.9, label=f"members ({M.shape[1]})")
    if rev.any():
        ax.plot([], [], color=EASTERLY, lw=1.2, label=f"members turning easterly ({int(rev.sum())})")


def weekly_index(times, M, north: bool, start) -> dict:
    """Members whose first winter reversal falls in each forecast week, and how many have reversed by its end.
    start: the forecast's start (a date or datetime); day d = the valid day d days after it. A week the series
    ends one day short of (GEFS's daily means stop at day 34) is kept, with the day it ends on."""
    M = np.asarray(M, float)
    fr = first_reversal(times, M, north)
    day0 = start.date() if hasattr(start, "date") else start
    day_of = np.array([(t.date() - day0).days for t in times])
    first_day = np.where(fr >= 0, day_of[np.maximum(fr, 0)], 10**6)
    weeks = []
    for a, b in WEEKS:
        end = min(b, int(day_of.max()))
        if end < b - 1:
            break
        weeks.append(
            {
                "days": [a, end],
                "first": int(((first_day >= a) & (first_day <= end)).sum()),
                "by_end": int((first_day <= end).sum()),
            }
        )
    return {"members": int(M.shape[1]), "weeks": weeks}
