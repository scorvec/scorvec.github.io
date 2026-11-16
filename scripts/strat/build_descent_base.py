#!/usr/bin/env python3
"""How often the polar-cap height anomaly is one unbroken column from 100 hPa down, in MERRA-2: the base rate the
descent watch (descent_watch.py) sets each ensemble's share of members against.

A day counts for a level when the 65-90N cap height stands at THRESHOLD standard deviations or more above its normal
at every level of CHAIN from 100 hPa down to that one, on that same day; heights standardised as the dripping paint's
are (build_drip_ref.Ref, the MERRA-2 fit). The rate for a calendar day pools every year's days within WINDOW slots of
it on the 366-slot calendar (29 February has a slot of its own), November to April.

The rate is known only as well as about 46 winters give it, and a column lasts for days, so the days of a winter are
not separate trials: `upper` is the 95th percentile of the rate over N_BOOT resamples of whole winters. The watch
tests the members against `upper`, so "more than usual" means more than the record can call usual.

    python build_descent_base.py            # -> reference/descent_base.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from quantmet.degdays import doy366

from build_drip_ref import Ref
from descent_watch import BASE, CHAIN, THRESHOLD, connected

WINDOW = 15  # days either side
SEASON = (11, 12, 1, 2, 3, 4)
N_BOOT = 2000
SEED = 20261116


def main() -> int:
    import build_strat_history as B

    d = B.load_m2()
    t = pd.DatetimeIndex(d.time.values)
    ref = Ref()
    z = np.column_stack([ref.z(t, L, d.cap_z.sel(lev=L).values.astype(float)) for L in CHAIN])
    hit = connected(z)
    slot = np.asarray(doy366(t))
    days = pd.date_range("2000-01-01", "2000-12-31", freq="D")  # a leap year: every calendar day, 29 February too
    days = days[days.month.isin(SEASON)]
    winter = np.asarray(t.year + (t.month >= 7))
    ws = np.unique(winter)
    pick = np.random.default_rng(SEED).integers(0, len(ws), (N_BOOT, len(ws)))
    weight = np.stack([np.bincount(p, minlength=len(ws)) for p in pick]).astype(float)  # (draw, winter)
    rate, upper = {}, {}
    for day in days:
        gap = np.abs(slot - int(doy366([day])[0]))
        near = np.minimum(gap, 366 - gap) <= WINDOW
        rate[f"{day:%m-%d}"] = [round(float(v), 4) for v in hit[near].mean(0)]
        wi = np.searchsorted(ws, winter[near])
        hits = np.stack(
            [np.bincount(wi, hit[near][:, k], len(ws)) for k in range(len(CHAIN))], axis=1
        )  # (winter, level)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (weight @ hits) / (weight @ np.bincount(wi, minlength=len(ws)))[:, None]
        upper[f"{day:%m-%d}"] = [round(float(v), 4) for v in np.nanpercentile(boot, 95, axis=0)]
    doc = {
        "source": f"MERRA-2 cap_z (65-90N), {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}, standardised as reference/drip_std.json",
        "chain_hpa": list(CHAIN),
        "threshold_sd": THRESHOLD,
        "window_days": WINDOW,
        "rate": rate,
        "upper": upper,
    }
    BASE.write_text(json.dumps(doc, separators=(",", ":")))
    for key in ("11-20", "12-15", "01-15", "02-15", "03-15", "04-15"):
        print(
            f"  {key}: "
            + ", ".join(
                f"{L:.0f} hPa {100 * v:.1f} (to {100 * u:.1f}) %" for L, v, u in zip(CHAIN, rate[key], upper[key])
            )
        )
    print(f"wrote {BASE} ({len(rate)} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
