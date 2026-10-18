#!/usr/bin/env python3
"""Himalayan mountain torque -> North Pacific jet phase indices: a SIGNED lagged regression on ERA5 1991-2020 (built
once, offline), the research view of pacjet.py.

The previous product composited the jet after torque days >= +1.5 sigma only and then anchored that composite on the
forecast's torque peak whatever its sign. This asks the symmetric question instead: for every lag L from -10 to +20
days, the least-squares slope of the jet index at t + L on the torque anomaly at t, both standardised within the
season window (so the slope is also the correlation and beta^2 the share of variance), for torque days in each
in-season month's three-month window.

Uncertainty honours the autocorrelation of both series (the torque decorrelates in a few days, the jet indices in about
a week): a moving-block bootstrap over the time-ordered torque days with BLOCK-day blocks (1000 draws) gives the 90%
interval and a two-sided percentile p-value, and the p-values of all lags and both indices in a window are held to a
false discovery rate of 10% (Benjamini-Hochberg; Wilks 2016). The effective sample size from the lag-1
autocorrelations is reported as a cross-check.

The torque is the live product's own (torque_map_anim.py, read by pacjet.py from torque_ranges.json): the
Himalaya/Tibet day-mean mountain-torque anomaly, 5-degree h dp_s/dlam over 70-105E 25-45N, from ERA5 1991-2020
(build_torque_range_sd.day_anomalies; high pressure west of the range and low east = braking, negative). Each month's
window mean and sigma of it go into the output (torque_scale), and pacjet.py standardises the forecast with exactly
those, so forecast and predictor are one quantity on one scale.

    python src/build_pacjet_torque.py            -> scripts/mjo/data/reference/pacjet_torque.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.stats import benjamini_hochberg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_torque_range_sd as BRS

REF = HERE.parent / "data" / "reference"
LAGS = list(range(-10, 21))
BLOCK, NBOOT, ALPHA = 30, 1000, 0.10


def bh(p, alpha=ALPHA):
    return benjamini_hochberg(p, alpha)  # quantmet.stats (was a local copy; identical)


def main() -> int:
    with xr.open_dataset(REF / "pacjet_ref.nc") as ds:
        ref = ds.load()
    t = pd.DatetimeIndex(ref.time.values)
    tq = BRS.day_anomalies()["HIM"].reindex(t).to_numpy(float)  # the last day has no ERA5 window: NaN, dropped
    pc = ref.pc.values.astype(float)
    mon = t.month.values
    n = len(t)
    rng = np.random.default_rng(7)
    out = {
        "lags": LAGS,
        "block_days": BLOCK,
        "n_boot": NBOOT,
        "alpha_fdr": ALPHA,
        "note": "beta = regression of the standardised jet index at t+lag on the standardised Himalaya/Tibet "
        "day-mean mountain torque at t (the live product's form: 5-degree h dp_s/dlam, ERA5 1991-2020), torque days "
        "in each month's 3-month window; 90% moving-block bootstrap CI; significance: bootstrap p, BH FDR 10% over "
        "lags x indices. torque_scale: each window's torque mean and sigma (Hadley), the forecast's standardisation",
        "torque_scale": {},
        "windows": {},
    }
    for m in range(1, 13):
        months = [((m - 2) % 12) + 1, m, (m % 12) + 1]
        days = np.flatnonzero(np.isin(mon, months) & np.isfinite(tq))
        days = days[(days + min(LAGS) >= 0) & (days + max(LAGS) < n)]
        x = tq[days]
        out["torque_scale"][str(m)] = {"mean": round(float(x.mean()), 3), "sd": round(float(x.std()), 3)}
        if not bool(ref.in_season.sel(month=m)):
            continue
        x = (x - x.mean()) / x.std()
        Y = np.stack([pc[days + L] for L in LAGS], 1)  # (n, lag, 2)
        Y = (Y - Y.mean(0)) / Y.std(0)
        beta = (x[:, None, None] * Y).mean(0)  # (lag, 2)
        nb = int(np.ceil(len(days) / BLOCK))
        boots = np.empty((NBOOT, len(LAGS), 2))
        for b in range(NBOOT):
            st = rng.integers(0, len(days) - BLOCK + 1, nb)
            idx = (st[:, None] + np.arange(BLOCK)[None]).ravel()[: len(days)]
            xb = x[idx]
            Yb = Y[idx]
            xb = (xb - xb.mean()) / xb.std()
            Yb = (Yb - Yb.mean(0)) / Yb.std(0)
            boots[b] = (xb[:, None, None] * Yb).mean(0)
        lo, hi = np.percentile(boots, 5, 0), np.percentile(boots, 95, 0)
        p = np.minimum(1.0, 2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0)))
        sig = bh(p.ravel()).reshape(p.shape)
        r1x = np.corrcoef(x[:-1], x[1:])[0, 1]
        r1y = [np.corrcoef(pc[days[:-1], k], pc[days[:-1] + 1, k])[0, 1] for k in range(2)]
        neff = [len(days) * (1 - r1x * r) / (1 + r1x * r) for r in r1y]
        lab = "–".join(pd.Timestamp(2001, mm, 1).strftime("%b") for mm in (months[0], months[-1])) + " torque days"
        win = {
            "label": lab,
            "n_days": len(days),
            "n_eff_lag1": [round(v) for v in neff],
            "torque_r1": round(float(r1x), 2),
        }
        for k, name in enumerate(("pc1", "pc2")):
            win[name] = {
                "beta": np.round(beta[:, k], 3).tolist(),
                "lo": np.round(lo[:, k], 3).tolist(),
                "hi": np.round(hi[:, k], 3).tolist(),
                "p": np.round(p[:, k], 4).tolist(),
                "sig": sig[:, k].tolist(),
            }
        out["windows"][str(m)] = win
        best = [
            (name, LAGS[int(np.argmax(np.abs(beta[:, k]) * (np.array(LAGS) >= 1)))])
            for k, name in enumerate(("pc1", "pc2"))
        ]
        txt = "; ".join(
            f"{nm} max |beta| at +{L} d = {beta[LAGS.index(L), k]:+.2f} [{lo[LAGS.index(L), k]:+.2f}, "
            f"{hi[LAGS.index(L), k]:+.2f}] {'SIG' if sig[LAGS.index(L), k] else 'n.s.'}"
            for k, (nm, L) in enumerate(best)
        )
        print(
            f"  {pd.Timestamp(2001, m, 1):%b} ({lab}, n {len(days)}, n_eff {win['n_eff_lag1']}): {txt}; "
            f"significant lags {sig.sum()} of {sig.size}",
            flush=True,
        )
    (REF / "pacjet_torque.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {REF / 'pacjet_torque.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
