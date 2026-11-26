#!/usr/bin/env python3
"""Skill of the weeks 3-5 sudden-warming index (ssw.html, ssw_summary.extended_part) in the GEFSv12 reforecast.

The index counts, per forecast week, the members whose drift-corrected u(60N, 10 hPa) first turns easterly that week
on a November-March day (scripts/lib/vortex_members.first_reversal). Here the same count is made in the reforecast's
35-day runs (11 members, 00Z Wednesdays, 2000-2019; the gefs_eddy_extract.py extracts of eddy-feedback-gefs.yml) and
scored against MERRA-2's u(60N, 10 hPa) on the same valid days (strat_history.nc), with the event defined exactly as
for a member: the first easterly winter day of days 1-34 falls in the week.

Drift correction as gefs_live.vortex_series draws the members: raw - M(start, lead) + the observed normal of the valid
day, M = the members' mean at that lead over the starts of the OTHER winters within 15 days of the start's day of year
(common-year calendar), the normal = MERRA-2 1991-2020 by day of year, 31-day running mean.

Score: Brier skill of the member share against the climatological probability (the observed event rate over the other
winters' starts within 15 days of the start), weeks 1-5 (week 5 = days 29-34, the extract's last day) and days 15-34
together; winter-block bootstrap (5000) for the interval and a one-sided p (share of resamples with BSS <= 0).

    python ssw35_skill.py --gefs DIR --strat reference/strat_history.nc --out reference/ssw35_skill.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import vortex_members as VM

sys.path.insert(0, str(HERE.parent / "mjo" / "src"))
from doycal import noleap_doy

WEEKS = ((1, 7), (8, 14), (15, 21), (22, 28), (29, 34), (15, 34))
POOL = 15  # days either side of a start's day of year
NBOOT = 5000
ALPHA = 0.05


def load(d: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Every extract's starts and u(60N, 10 hPa) (start, member, day) with its lead days."""
    zs = [np.load(p) for p in sorted(d.glob("GEFSv12_*.npz")) if "test" not in p.name and ".part" not in p.name]
    z0 = zs[0]
    k10 = [int(x) for x in z0["u_levels"]].index(10)
    j60 = int(np.argmin(np.abs(z0["lat"].astype(float) - 60.0)))
    init = pd.DatetimeIndex(np.concatenate([z["init"] for z in zs]))
    u = np.concatenate([z["u"][..., k10, j60] for z in zs]).astype(float)
    o = np.argsort(init.values)
    return init[o], u[o], z0["days"].astype(int)


def cday(t) -> np.ndarray:
    """Common-year day 1..365 (29 February joins 28 February)."""
    return np.floor(noleap_doy(t)).astype(int)


def winter_of(t: pd.DatetimeIndex) -> np.ndarray:
    return np.where(t.month >= 7, t.year, t.year - 1)


def pooled(doy: np.ndarray, win: np.ndarray, i: int) -> np.ndarray:
    """The starts of the other winters within POOL days of start i (on the circular common-year calendar)."""
    dd = np.abs(doy - doy[i])
    return (np.minimum(dd, 365 - dd) <= POOL) & (win != win[i])


def first_week(times: list, M: np.ndarray, lead: np.ndarray) -> np.ndarray:
    """Lead day of each column's first easterly winter day, a large number for none. M: (lead, column)."""
    fr = VM.first_reversal(times, M, True)
    return np.where(fr >= 0, lead[np.maximum(fr, 0)], 10**6)


def bss(p: np.ndarray, o: np.ndarray, c: np.ndarray, win: np.ndarray, rng) -> dict:
    """Brier skill of p against c for outcomes o, with its winter-block bootstrap."""
    ws = np.unique(win)
    bf = np.array([((p - o) ** 2)[win == w].sum() for w in ws])
    bc = np.array([((c - o) ** 2)[win == w].sum() for w in ws])
    est = 1 - bf.sum() / bc.sum()
    k = rng.integers(0, len(ws), (NBOOT, len(ws)))
    boot = 1 - bf[k].sum(1) / bc[k].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "bss": round(float(est), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "p": round(float((boot <= 0).mean()), 4),
        "events": int(o.sum()),
        "starts": len(o),
        "mean_share_event": round(float(p[o == 1].mean()), 3) if o.any() else None,
        "mean_share_none": round(float(p[o == 0].mean()), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gefs", type=Path, required=True, help="directory of GEFSv12_*.npz extracts")
    ap.add_argument("--strat", type=Path, required=True, help="strat_history.nc (MERRA-2 u(60N, 10 hPa))")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    init, u, lead = load(a.gefs)
    with xr.open_dataset(a.strat) as ds:
        obs = ds["u10"].to_series().astype(float)
    obs.index = pd.DatetimeIndex(obs.index).normalize()
    base = obs["1991":"2020"]
    nd = cday(base.index.values)
    norm = pd.Series(base.values).groupby(nd).mean().reindex(range(1, 366)).values
    norm = np.convolve(np.r_[norm[-15:], norm, norm[:15]], np.ones(31) / 31, "valid")  # (365,) by common-year day

    valid = [pd.DatetimeIndex(init[i] + pd.to_timedelta(lead, "D")) for i in range(len(init))]
    O = np.array([obs.reindex(v).values for v in valid])  # (start, lead)
    # starts whose 34 days touch November-March; the others can hold no event in either source
    keep = np.isfinite(O).all(1) & np.isfinite(u).all((1, 2)) & np.isin(init.month, (10, 11, 12, 1, 2, 3))
    init, u, O, valid = init[keep], u[keep], O[keep], [v for v, k in zip(valid, keep) if k]
    win, doy = winter_of(init), cday(init.values)
    em = u.mean(1)  # (start, lead)
    out = {"source": "GEFSv12 reforecast 00Z Wednesdays, 11 members; MERRA-2 u(60N, 10 hPa)", "weeks": {}}
    shares, outcome = {}, {}
    for kind in ("corrected", "raw"):
        F = np.empty((len(init), len(WEEKS)))
        for i in range(len(init)):
            m = u[i].T  # (lead, member)
            if kind == "corrected":
                M = em[pooled(doy, win, i)].mean(0)
                m = m - M[:, None] + norm[cday(valid[i].values) - 1][:, None]
            fd = first_week(list(valid[i]), m, lead)
            F[i] = [((fd >= w0) & (fd <= w1)).mean() for w0, w1 in WEEKS]
            if kind == "corrected":
                fo = first_week(list(valid[i]), O[i][:, None], lead)[0]
                outcome[i] = [int(w0 <= fo <= w1) for w0, w1 in WEEKS]
        shares[kind] = F
    Ob = np.array([outcome[i] for i in range(len(init))], float)
    C = np.array([Ob[pooled(doy, win, i)].mean(0) for i in range(len(init))])
    rng = np.random.default_rng(20261126)
    for k, (w0, w1) in enumerate(WEEKS):
        row = {kind: bss(shares[kind][:, k], Ob[:, k], C[:, k], win, rng) for kind in shares}
        row["passes"] = bool(row["corrected"]["bss"] > 0 and row["corrected"]["p"] < ALPHA)
        out["weeks"][f"{w0}-{w1}"] = row
        r = row["corrected"]
        print(
            f"days {w0:2d}-{w1:2d}: {r['events']:3d} events in {r['starts']} starts; corrected BSS {r['bss']:+.3f}"
            f" [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p {r['p']:.4f}; raw BSS {row['raw']['bss']:+.3f};"
            f" mean share {r['mean_share_event']} with an event, {r['mean_share_none']} without",
            flush=True,
        )
    out["winters"] = f"{win.min()}-{win.max() + 1}"
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
