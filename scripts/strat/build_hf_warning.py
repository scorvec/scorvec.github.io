#!/usr/bin/env python3
"""The SSW warning threshold for the 100 hPa eddy heat flux products (reference/heatflux100_warning.json).

Question: how much upward wave activity, sustained, has preceded the sudden stratospheric warmings on record? The
heat flux products then read every member's forecast against that level. Everything here comes from two committed
references: ERA5's daily 45-75 N [v'T'] at 100 hPa (reference/vt100_era5.nc, 00/12 UTC averaged, from 1990-11) and
the SSW central dates of the stratosphere history (reference/strat_history_events.json, CP07 reversals; marginal
cases left out), so the record is the winters from 1990/91.

Metric: the trailing W-day mean of the daily flux. W = 10 by default: the vortex responds to the time-integrated
flux (Polvani & Waugh 2004), but a warning needs the pulse, not the season. On the record the 10-day mean has the
best skill and the sharpest odds (skill 0.67 against 0.63 at 20 days and 0.66 at 40; a warning day's chance of an
SSW within 30 days 0.35 against 0.24 and 0.17, the 40-day figure not separable from the base rate), and it leads
the central date by 13 days in the median, so a 15-day forecast can move it. A day is a WARNING when the metric is
at or above the threshold T. The outcome scored is an SSW central date within the next LEAD days (30).

Threshold: T maximises the Peirce skill score, hit rate minus false-alarm rate, over
  * the events: the metric's maximum in the 30 days up to each central date (one trial per SSW), and
  * the quiet record: every November-March day of the record that is not within 30 days before or 60 days after a
    central date, scored on the metric's maximum over the following 30 days (so the two sides describe the same
    quantity: the highest W-day mean a 30-day window reaches).
The threshold and the window are chosen on the record, so the record's own scores at that choice flatter it. The
scores published are therefore OUT OF SAMPLE: each winter is scored with the window (of WINDOWS) and threshold chosen
on the other winters (leave one winter out), pooled over the winters:
  * hit rate, false-alarm rate and their difference (the Peirce skill score),
  * precision  P(SSW within 30 days | warning day)   and the base rate P(SSW within 30 days | any winter day).
Their 90 % intervals come from a bootstrap over winters (2000 draws) that repeats the choice in every draw and scores
it on the winters the draw left out. The in-sample scores are kept beside them (in_sample) to show the difference.
The lead is the days from a winter's first warning day to its SSW at the published threshold.
Nothing is published where the interval of the out-of-sample skill score includes zero.

    python scripts/strat/build_hf_warning.py            # writes reference/heatflux100_warning.json
    python scripts/strat/build_hf_warning.py --window 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path(__file__).resolve().parent / "reference"
SERIES = REF / "vt100_era5.nc"
EVENTS = REF / "strat_history_events.json"
OUT = REF / "heatflux100_warning.json"
LEAD = 30  # days ahead an SSW counts as the outcome of a warning
PRE = 30  # days before a central date in which the precursor pulse is looked for
QUIET_AFTER = 60  # days after a central date left out of the quiet record (the vortex is recovering)
WINTER_MONTHS = (11, 12, 1, 2, 3)
WINDOWS = (10, 20, 40)  # the windows the choice is made among
GRID = np.round(np.arange(10.0, 40.0, 0.5), 1)  # the thresholds tried, K m/s
N_BOOT = 2000
SEED = 20261017


def winter_of(d: pd.Timestamp) -> int:
    """The winter a date belongs to, named by its January: Nov-Dec go with the following year."""
    return d.year + 1 if d.month >= 7 else d.year


def load():
    v = xr.open_dataset(SERIES)
    s = pd.Series(v["nh_tot"].mean("hour").values, index=pd.DatetimeIndex(v.time.values)).asfreq("D")
    gaps = int(s.isna().sum())
    s = s.interpolate(limit=3)
    first = s.index[0] + pd.Timedelta(days=200)  # the first winter needs a full autumn behind it
    ev = sorted(
        pd.Timestamp(e["date"])
        for e in json.loads(EVENTS.read_text())["ssw"]
        if pd.Timestamp(e["date"]) >= first and not e.get("marginal")
    )
    return s, ev, gaps


def tables(s: pd.Series, ev: list, W: int):
    m = s.rolling(W, min_periods=W).mean()
    mx = m[::-1].rolling(LEAD, min_periods=LEAD).max()[::-1]  # the metric's maximum over the next LEAD days
    pre = pd.Series([m.loc[d - pd.Timedelta(days=PRE) : d].max() for d in ev], index=ev)
    near = pd.Series(False, index=s.index)
    for d in ev:
        near.loc[d - pd.Timedelta(days=PRE) : d + pd.Timedelta(days=QUIET_AFTER)] = True
    winter = s.index.month.isin(WINTER_MONTHS)
    quiet = mx[winter & ~near.values].dropna()
    # outcome of every winter day: an SSW central date within the next LEAD days
    hit30 = pd.Series(False, index=s.index)
    for d in ev:
        hit30.loc[d - pd.Timedelta(days=LEAD) : d] = True
    days = m[winter].dropna()
    return m, pre, quiet, days, hit30.loc[days.index]


def skill(pre: pd.Series, quiet: pd.Series, T: float) -> tuple[float, float, float]:
    hit = float((pre >= T).mean())
    fa = float((quiet >= T).mean())
    return hit - fa, hit, fa


class Record:
    """One window's tables split by winter, so a set of winters can choose a threshold and another be scored."""

    def __init__(self, s: pd.Series, ev: list, W: int):
        self.m, self.pre, self.quiet, self.days, self.out30 = tables(s, ev, W)
        self.winters = sorted({winter_of(d) for d in self.days.index})
        wq = np.array([winter_of(d) for d in self.quiet.index])
        wd = np.array([winter_of(d) for d in self.days.index])
        self.ev = {w: [d for d in ev if winter_of(d) == w] for w in self.winters}
        self.q = {w: self.quiet.values[wq == w] for w in self.winters}
        self.d = {w: (self.days.values[wd == w], self.out30.values[wd == w]) for w in self.winters}

    def parts(self, ws):
        """(events' precursor maxima, quiet maxima, winter days' metric, their outcomes) of the winters ws, repeats kept."""
        pre = self.pre.loc[[d for w in ws for d in self.ev[w]]].values
        q = np.concatenate([self.q[w] for w in ws]) if len(ws) else np.zeros(0)
        d = np.concatenate([self.d[w][0] for w in ws]) if len(ws) else np.zeros(0)
        o = np.concatenate([self.d[w][1] for w in ws]) if len(ws) else np.zeros(0, bool)
        return pre, q, d, o


def choose(recs: dict, ws) -> tuple[int, float]:
    """The window and threshold with the best Peirce skill on the winters ws (the first of ties)."""
    best = (-np.inf, WINDOWS[0], float(GRID[0]))
    for W, r in recs.items():
        pre, q, _, _ = r.parts(ws)
        if not len(pre) or not len(q):
            continue
        for T in GRID:
            pss = float((pre >= T).mean() - (q >= T).mean())
            if pss > best[0]:
                best = (pss, W, float(T))
    return best[1], best[2]


def counts(recs: dict, fit, test) -> np.ndarray:
    """The winters `test` scored with the choice made on `fit`: (events, events warned, quiet windows, quiet windows
    warned, warning days, warning days followed by an SSW, days, days followed by an SSW)."""
    W, T = choose(recs, fit)
    pre, q, d, o = recs[W].parts(test)
    on = d >= T
    return np.array([len(pre), (pre >= T).sum(), len(q), (q >= T).sum(), on.sum(), o[on].sum(), len(d), o.sum()], float)


def rates(c: np.ndarray) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        hit, fa = c[1] / c[0], c[3] / c[2]
        return {"hit_rate": hit, "false_alarm_rate": fa, "peirce_skill": hit - fa, "precision": c[5] / c[4],
                "base_rate": c[7] / c[6]}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    W = a.window
    s, ev, gaps = load()
    recs = {w: Record(s, ev, w) for w in dict.fromkeys((*WINDOWS, W))}
    r = recs[W]
    m, pre, quiet, days, out30, winters = r.m, r.pre, r.quiet, r.days, r.out30, r.winters
    scores = [skill(pre, quiet, T) for T in GRID]
    i = int(np.argmax([x[0] for x in scores]))
    T, (pss, hit, fa) = float(GRID[i]), scores[i]
    on = days >= T
    precision = float(out30[on].mean())
    base = float(out30.mean())
    # the lead from a winter's first warning day to its SSW (events whose precursor crossed T)
    lead_of = {}
    for d in ev:
        win = m.loc[d - pd.Timedelta(days=PRE) : d]
        c = win[win >= T]
        if len(c):
            lead_of[d] = int((d - c.index[0]).days)
    leads = list(lead_of.values())
    # out of sample: every winter scored with the choice made on the others
    fam = {w: recs[w] for w in WINDOWS}
    oos = rates(sum(counts(fam, [x for x in winters if x != w], [w]) for w in winters))
    # its intervals: winters drawn with replacement choose, the winters left out are scored
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        draw = rng.choice(winters, size=len(winters), replace=True)
        left = sorted(set(winters) - set(draw.tolist()))
        c = counts(fam, list(draw), left)
        if c[0] and c[2]:
            boot.append(rates(c))

    def ci(key):
        x = [b[key] for b in boot]
        return [round(float(np.nanpercentile(x, 5)), 3), round(float(np.nanpercentile(x, 95)), 3)]

    doc = {
        "built": datetime.now(UTC).strftime("%Y-%m-%d"),
        "metric": f"trailing {W}-day mean of the daily (00/12 UTC) 100 hPa [v'T'], 45-75N, K m s-1",
        "window_days": W,
        "lead_days": LEAD,
        "threshold": T,
        "record": f"{s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d} (ERA5), {len(winters)} winters, {gaps} interpolated days",
        "n_events": len(ev),
        "events": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "max_pre30": round(float(pre.loc[d]), 1),
                "lead_first_warning": lead_of.get(d),
            }
            for d in ev
        ],
        "scores": "out of sample: each winter scored with the window and threshold chosen on the other winters",
        "hit_rate": round(oos["hit_rate"], 3),
        "false_alarm_rate": round(oos["false_alarm_rate"], 3),
        "peirce_skill": round(oos["peirce_skill"], 3),
        "peirce_skill_ci90": ci("peirce_skill"),
        "precision": round(oos["precision"], 3),
        "precision_ci90": ci("precision"),
        "base_rate": round(oos["base_rate"], 3),
        "base_rate_ci90": ci("base_rate"),
        "in_sample": {
            "hit_rate": round(hit, 3),
            "false_alarm_rate": round(fa, 3),
            "peirce_skill": round(pss, 3),
            "precision": round(precision, 3),
            "base_rate": round(base, 3),
        },
        "n_warning_days": int(on.sum()),
        "n_winter_days": len(days),
        "lead_days_first_warning": {
            "median": float(np.median(leads)) if leads else None,
            "p25": float(np.percentile(leads, 25)) if leads else None,
            "p75": float(np.percentile(leads, 75)) if leads else None,
            "n": len(leads),
        },
        "quiet_quantiles": {
            k: round(float(quiet.quantile(q)), 1) for k, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
        },
        "winter_metric_quantiles": {
            k: round(float(days.quantile(q)), 1) for k, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        },
        "skill_by_threshold": [
            [float(t), round(x[0], 3), round(x[1], 3), round(x[2], 3)] for t, x in zip(GRID, scores)
        ],
        "significant": bool(ci("peirce_skill")[0] > 0),
    }
    print(
        f"W={W} d: T={T} K m/s; in sample hit {hit:.2f}, false alarm {fa:.2f}, PSS {pss:.2f}, "
        f"P(SSW<=30 d | warning) {precision:.2f} vs base {base:.2f}; out of sample hit {oos['hit_rate']:.2f}, false "
        f"alarm {oos['false_alarm_rate']:.2f}, PSS {oos['peirce_skill']:.2f} {ci('peirce_skill')}, P(SSW<=30 d | warning) "
        f"{oos['precision']:.2f} {ci('precision')} vs base {oos['base_rate']:.2f}; lead median "
        f"{doc['lead_days_first_warning']['median']} d (n {len(leads)}); {len(ev)} events, {on.sum()} warning days of {len(days)}"
    )
    if not doc["significant"]:
        print("::warning::skill interval includes zero: nothing written")
        return 1
    if not a.dry_run:
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
