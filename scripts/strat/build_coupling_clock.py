#!/usr/bin/env python3
"""Reference for the coupling clock on ssw.html: after the MERRA-2 sudden warmings since 1980, how the wind at 60N,
100 hPa over the days after the 10 hPa reversal told when the surface pattern of a warming began.

    python build_coupling_clock.py      # -> reference/u100_std.json, reference/coupling_clock.json

INPUTS. MERRA-2 daily zonal means (build_strat_history.load_m2: the raw archive, its QC and gap filling): u at 60N,
100 hPa and the 65-90N cap height at 1000 hPa; the SSW surface-imprint index since 1959 and the 28 MERRA-2 sudden
warmings it was tested on (reference/ssw_imprint.nc); the dripping paint's standardisation (reference/drip_std.json).

DEFINITIONS
  wind      u(60N, 100 hPa) against its normal, in sd, fitted as the dripping paint's polar cap is (build_drip_ref.fit:
            4 annual harmonics and a trend, a 3-harmonic sd; stored as reference/u100_std.json so ssw_summary.py puts
            the forecasts on the same scale), averaged over days 0-10 after the onset (the first easterly day at
            10 hPa): what a days 1-15 forecast reaches of an onset in its first days.
  start     the first day 1-60 after the onset whose next 14 days average +1.5 or more on the imprint index: a surface
            signal that lasts, not a noisy week. None within 60 days ranks after every start.
  strength  the imprint index over days 11-40.
  sea level the 1000 hPa polar-cap height over days 6-15, on the dripping paint's scale (sd): the drips' AO proxy.
            Kept day by day from day 0 as well, so a forecast reaching only part of those days is set against the
            same days of the past warmings.
TESTS. Spearman rank correlation of the wind with the start day and with the strength, against N_PERM shuffles of
the wind (two-sided). The page shows the clock only when the start-day test passes.
THIRDS. The warmings in three groups by the wind; per group the start days, the share under way within SOON days, the
strength and the sea-level range, which the page reads each forecast's place against.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_strat_history as B
from build_drip_ref import Ref, fit

IMPRINT = HERE / "reference" / "ssw_imprint.nc"
U100_STD = HERE / "reference" / "u100_std.json"
OUT = HERE / "reference" / "coupling_clock.json"
WIND_DAYS = (0, 10)
START_SPAN, START_THR, START_MAX = 14, 1.5, 60
STRENGTH_DAYS = (11, 40)
SEA_DAYS = (6, 15)
SOON = 10
N_PERM = 20000
ALPHA = 0.05


def over(s: pd.Series, onset: pd.Timestamp, days: tuple[int, int]) -> float:
    return float(s.loc[onset + pd.Timedelta(days=days[0]) : onset + pd.Timedelta(days=days[1])].mean())


def start_day(index: pd.Series, onset: pd.Timestamp) -> int | None:
    ahead = index.rolling(START_SPAN).mean().shift(-(START_SPAN - 1))  # each day's mean with the 13 days after it
    x = ahead.loc[onset + pd.Timedelta(days=1) : onset + pd.Timedelta(days=START_MAX)]
    hit = np.flatnonzero(x.values >= START_THR)
    return int(hit[0]) + 1 if hit.size else None


def test(x: np.ndarray, y: np.ndarray, rng) -> dict:
    rho = stats.spearmanr(x, y).statistic
    null = np.array([stats.spearmanr(rng.permutation(x), y).statistic for _ in range(N_PERM)])
    return {"rho": round(float(rho), 3), "p": round(float((np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERM + 1)), 5)}


def main() -> int:
    d = B.load_m2()
    t = pd.DatetimeIndex(d.time.values)
    u100 = d.u60.sel(lev=100.0).values.astype(float)
    b, bs, floor, vfloor = fit(u100, t)
    U100_STD.write_text(json.dumps({
        "source": f"MERRA-2 u at 60N, 100 hPa (zonal mean, daily mean of 00/06/12/18Z), {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}, "
                  "build_strat_history QC and gap filling",
        "method": "as reference/drip_std.json (build_drip_ref.fit): mean 4 annual harmonics + linear trend per decade "
                  "from 2000; sd sqrt of a 3-harmonic fit to the squared residual (top 0.1 % clipped), floored at a "
                  "third of its seasonal maximum",
        "levels": {"100.0": {"b": [float(v) for v in b], "bs": [float(v) for v in bs], "sd_floor": floor,
                             "var_floor": vfloor, "step": False}},
    }, indent=1))  # fmt: skip
    wind = pd.Series(Ref(U100_STD).z(t, 100, u100), t)
    sea = pd.Series(Ref().z(t, 1000, d.cap_z.sel(lev=1000.0).values.astype(float)), t)
    with xr.open_dataset(IMPRINT) as f:
        index = f["index"].to_series()
        onsets = [pd.Timestamp(str(e)) for e in f.event.values]

    rows = [{
        "onset": f"{e:%Y-%m-%d}",
        "wind": round(over(wind, e, WIND_DAYS), 2),
        "start": start_day(index, e),
        "strength": round(over(index, e, STRENGTH_DAYS), 2),
        "sea": round(over(sea, e, SEA_DAYS), 2),
        "sea_daily": [round(float(v), 2) for v in sea.loc[e : e + pd.Timedelta(days=SEA_DAYS[1])]],
    } for e in onsets]  # fmt: skip
    w = np.array([r["wind"] for r in rows])
    s = np.array([START_MAX + 1 if r["start"] is None else r["start"] for r in rows])
    rng = np.random.default_rng(20261030)
    tests = {"start": test(w, s, rng), "strength": test(w, np.array([r["strength"] for r in rows]), rng)}

    groups = np.array_split(np.argsort(w, kind="stable"), 3)
    edges = [None, *(round(float((w[a[-1]] + w[c[0]]) / 2), 2) for a, c in itertools.pairwise(groups)), None]
    thirds = []
    for k, part in enumerate(groups):
        g = [rows[i] for i in part]
        st = s[part]
        thirds.append({
            "wind": [edges[k], edges[k + 1]],
            "n": len(g),
            "onsets": [r["onset"] for r in g],
            "start_quartiles": [int(np.percentile(st, q, method="nearest")) for q in (25, 50, 75)],
            "soon": int((st <= SOON).sum()),
            "never": int((st > START_MAX).sum()),
            "strength_mean": round(float(np.mean([r["strength"] for r in g])), 2),
            "strength_positive": int(sum(r["strength"] > 0 for r in g)),
            "sea_mean": round(float(np.mean([r["sea"] for r in g])), 2),
            "sea_range": [min(r["sea"] for r in g), max(r["sea"] for r in g)],
        })  # fmt: skip

    OUT.write_text(json.dumps({
        "built": f"{pd.Timestamp.now('UTC'):%Y-%m-%d}",
        "period": f"{t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}",
        "definitions": {"wind_days": list(WIND_DAYS), "start_span": START_SPAN, "start_threshold": START_THR,
                        "start_max": START_MAX, "strength_days": list(STRENGTH_DAYS), "sea_days": list(SEA_DAYS),
                        "soon": SOON, "n_perm": N_PERM, "alpha": ALPHA},
        "tests": tests,
        "passes": tests["start"]["p"] < ALPHA,
        "thirds": thirds,
        "events": rows,
    }, indent=1))  # fmt: skip
    print(f"wrote {OUT.name}: {len(rows)} warmings; wind vs start {tests['start']}, vs strength {tests['strength']}")
    for k, g in enumerate(thirds):
        print(
            f"  third {k + 1}: wind {g['wind']}, n {g['n']}, start quartiles {g['start_quartiles']}, within {SOON} d "
            f"{g['soon']}, never {g['never']}, strength {g['strength_mean']:+.2f} ({g['strength_positive']} > 0), "
            f"sea level {g['sea_mean']:+.2f} {g['sea_range']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
