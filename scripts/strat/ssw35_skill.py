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
together; winter-block bootstrap (5000) for the interval and a one-sided p (share of resamples with BSS <= 0). Also the
share's ROC area (discrimination alone) and the Brier skill of the share mapped to a chance by a logistic fit on the
other winters (whether a calibrated chance would have skill where the raw share has none).

    python ssw35_skill.py --gefs DIR --strat reference/strat_history.nc [--vt DIR] --out reference/ssw35_skill.json
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
FLUX_Z = 0.5  # a heat-flux week: build_ssw_checklist.THRESHOLDS['flux'] (sd above the ERA5 normal)


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


def winter_starts(t: pd.DatetimeIndex) -> np.ndarray:
    """Starts whose 34 days touch November-March; the others can hold no reversal in either source."""
    return np.isin(t.month, (10, 11, 12, 1, 2, 3))


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


def roc_area(p: np.ndarray, o: np.ndarray, win: np.ndarray, rng) -> dict:
    """ROC area of p for o (ties count half), with its winter-block bootstrap; p = share of resamples at or below 0.5."""

    def area(pp, oo):
        pos, neg = pp[oo == 1], pp[oo == 0]
        if not len(pos) or not len(neg):
            return np.nan
        return float(
            ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
        )

    ws = np.unique(win)
    rows = [np.flatnonzero(win == w) for w in ws]
    boot = np.array(
        [
            area(p[i], o[i])
            for i in (np.concatenate([rows[k] for k in rng.integers(0, len(ws), len(ws))]) for _ in range(NBOOT))
        ]
    )
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "auc": round(area(p, o), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "p": round(float((boot <= 0.5).mean()), 4),
    }


def recalibrated(p: np.ndarray, o: np.ndarray, c: np.ndarray, win: np.ndarray, rng) -> dict:
    """Brier skill of the share mapped to a chance by a logistic fit on the other winters (leave one winter out)."""
    q = np.empty_like(p)
    for w in np.unique(win):
        tr = win != w
        X = np.c_[np.ones(tr.sum()), p[tr]]
        b = np.zeros(2)
        for _ in range(50):  # Newton's method on the log-likelihood, lightly ridged for the rare-event fits
            m = 1 / (1 + np.exp(-(X @ b)))
            H = X.T @ (X * (m * (1 - m))[:, None]) + 1e-6 * np.eye(2)
            b = b + np.linalg.solve(H, X.T @ (o[tr] - m))
        q[~tr] = 1 / (1 + np.exp(-(b[0] + b[1] * p[~tr])))
    r = bss(q, o, c, win, rng)
    return {k: r[k] for k in ("bss", "ci95", "p")}


def scores(p: np.ndarray, o: np.ndarray, c: np.ndarray, win: np.ndarray, rng) -> dict:
    return bss(p, o, c, win, rng) | {"roc": roc_area(p, o, win, rng), "recalibrated": recalibrated(p, o, c, win, rng)}


def reversal_skill(gefs: Path, strat: Path, rng) -> dict:
    """The first-reversal counts, drift-corrected (as on the page) and raw, week by week."""
    init, u, lead = load(gefs)
    with xr.open_dataset(strat) as ds:
        obs = ds["u10"].to_series().astype(float)
    obs.index = pd.DatetimeIndex(obs.index).normalize()
    base = obs["1991":"2020"]
    norm = pd.Series(base.values).groupby(cday(base.index.values)).mean().reindex(range(1, 366)).values
    norm = np.convolve(np.r_[norm[-15:], norm, norm[:15]], np.ones(31) / 31, "valid")  # (365,) by common-year day

    valid = [pd.DatetimeIndex(init[i] + pd.to_timedelta(lead, "D")) for i in range(len(init))]
    O = np.array([obs.reindex(v).values for v in valid])  # (start, lead)
    keep = np.isfinite(O).all(1) & np.isfinite(u).all((1, 2)) & winter_starts(init)
    init, u, O, valid = init[keep], u[keep], O[keep], [v for v, k in zip(valid, keep) if k]
    win, doy = winter_of(init), cday(init.values)
    em = u.mean(1)  # (start, lead)
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
    out = {"winters": f"{win.min()}-{win.max() + 1}", "weeks": {}}
    for k, (w0, w1) in enumerate(WEEKS):
        row = {kind: scores(shares[kind][:, k], Ob[:, k], C[:, k], win, rng) for kind in shares}
        row["passes"] = bool(row["corrected"]["bss"] > 0 and row["corrected"]["p"] < ALPHA)
        out["weeks"][f"{w0}-{w1}"] = row
        r = row["corrected"]
        print(
            f"reversal days {w0:2d}-{w1:2d}: {r['events']:3d} events in {r['starts']} starts; corrected BSS"
            f" {r['bss']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p {r['p']:.4f}; raw BSS"
            f" {row['raw']['bss']:+.3f}; mean share {r['mean_share_event']} with an event, {r['mean_share_none']} without;"
            f" ROC area {r['roc']['auc']:.3f} [{r['roc']['ci95'][0]:.3f}, {r['roc']['ci95'][1]:.3f}] p {r['roc']['p']:.4f};"
            f" recalibrated BSS {r['recalibrated']['bss']:+.3f} [{r['recalibrated']['ci95'][0]:+.3f},"
            f" {r['recalibrated']['ci95'][1]:+.3f}] p {r['recalibrated']['p']:.4f}",
            flush=True,
        )
    return out


def heat_skill(vt: Path, era5: Path, clim: Path, rng) -> dict:
    """The weekly 100 hPa heat flux: the members with a heat-flux week (the checklist's 7-day mean of at least FLUX_Z
    standard deviations above the ERA5 normal) scored as a probability, and the ensemble mean's weekly strength scored
    by its correlation with ERA5's. Raw, and with the lead-dependent mean error of the other winters' starts within
    POOL days removed (the page removes each model's significant mean error)."""
    from quantmet.degdays import doy366

    zs = [np.load(p) for p in sorted(vt.glob("GEFSv12vt_*.npz")) if "test" not in p.name and ".part" not in p.name]
    init = pd.DatetimeIndex(np.concatenate([z["init"] for z in zs]))
    keys = [str(k) for k in zs[0]["keys"]]
    f = np.concatenate([z["vt"][..., keys.index("tot")] for z in zs]).astype(float)  # (start, member, day)
    lead = zs[0]["days"].astype(int)
    o = np.argsort(init.values)
    init, f = init[o], f[o]
    with xr.open_dataset(era5) as ds:
        x = ds["nh_tot"].mean("hour").to_series().astype(float)
    x.index = pd.DatetimeIndex(x.index).normalize()
    with xr.open_dataset(clim) as ds:
        mu, sd = ds["nh_tot_mean"].values, ds["nh_tot_sd"].values
    valid = [pd.DatetimeIndex(init[i] + pd.to_timedelta(lead, "D")) for i in range(len(init))]
    s = np.array([doy366(v.values) for v in valid])
    Z = (f - mu[s][:, None]) / sd[s][:, None]
    O = (np.array([x.reindex(v).values for v in valid]) - mu[s]) / sd[s]
    keep = np.isfinite(O).all(1) & np.isfinite(Z).all((1, 2)) & winter_starts(init)
    init, Z, O = init[keep], Z[keep], O[keep]
    win, doy = winter_of(init), cday(init.values)
    out = {"winters": f"{win.min()}-{win.max() + 1}", "threshold_sd": FLUX_Z, "weeks": {}}
    for kind in ("corrected", "raw"):
        Zk = Z
        if kind == "corrected":
            err = Z.mean(1) - O  # (start, lead)
            Zk = Z - np.array([err[pooled(doy, win, i)].mean(0) for i in range(len(init))])[:, None]
        for w0, w1 in WEEKS:
            sl = slice(w0 - lead[0], w1 - lead[0] + 1)
            zm, zo = Zk[:, :, sl].mean(-1), O[:, sl].mean(-1)  # (start, member), (start,)
            p, ob = (zm >= FLUX_Z).mean(1), (zo >= FLUX_Z).astype(float)
            c = np.array([ob[pooled(doy, win, i)].mean() for i in range(len(init))])
            row = out["weeks"].setdefault(f"{w0}-{w1}", {})
            row[kind] = scores(p, ob, c, win, rng) | {"acc": acc(zm.mean(1), zo, win, rng)}
            if kind == "raw":
                r, a = row["corrected"], row["corrected"]["acc"]
                row["passes"] = bool(r["bss"] > 0 and r["p"] < ALPHA)
                print(
                    f"heat flux days {w0:2d}-{w1:2d}: {r['events']:3d} heat-flux weeks in {r['starts']} starts;"
                    f" corrected BSS {r['bss']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p {r['p']:.4f}"
                    f" (raw {row['raw']['bss']:+.3f}); ROC area {r['roc']['auc']:.3f} p {r['roc']['p']:.4f};"
                    f" recalibrated BSS {r['recalibrated']['bss']:+.3f} p {r['recalibrated']['p']:.4f};"
                    f" ensemble-mean strength r {a['r']:+.2f}"
                    f" [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] p {a['p']:.4f} (raw {row['raw']['acc']['r']:+.2f})",
                    flush=True,
                )
    return out


def acc(f: np.ndarray, o: np.ndarray, win: np.ndarray, rng) -> dict:
    """Correlation of f with o over the starts, with its winter-block bootstrap."""
    ws = np.unique(win)
    rows = [np.flatnonzero(win == w) for w in ws]
    est = float(np.corrcoef(f, o)[0, 1])
    boot = np.empty(NBOOT)
    for b in range(NBOOT):
        i = np.concatenate([rows[k] for k in rng.integers(0, len(ws), len(ws))])
        boot[b] = np.corrcoef(f[i], o[i])[0, 1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "r": round(est, 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "p": round(float((boot <= 0).mean()), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gefs", type=Path, required=True, help="directory of GEFSv12_*.npz extracts (u)")
    ap.add_argument("--strat", type=Path, required=True, help="strat_history.nc (MERRA-2 u(60N, 10 hPa))")
    ap.add_argument("--vt", type=Path, help="directory of GEFSv12vt_*.npz extracts (100 hPa heat flux)")
    ap.add_argument("--era5-vt", type=Path, default=HERE / "reference" / "vt100_era5.nc")
    ap.add_argument("--vt-clim", type=Path, default=HERE / "reference" / "vt100_clim.nc")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(20261126)
    out = {"source": "GEFSv12 reforecast 00Z Wednesdays, 11 members; MERRA-2 u(60N, 10 hPa); ERA5 100 hPa [v'T']"}
    out["reversal"] = reversal_skill(a.gefs, a.strat, rng)
    if a.vt:
        out["heat_flux"] = heat_skill(a.vt, a.era5_vt, a.vt_clim, rng)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
