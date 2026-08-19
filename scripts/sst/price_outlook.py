#!/usr/bin/env python3
"""Spot-price (bolsa) outlook driven by the inflow forecast.

Colombia's spot price is a hydro-thermal dispatch outcome: when reservoirs
are full and inflows strong, hydro sets the margin and the price sits near
variable cost; when water is short, thermal plant sets it and the price
runs toward the scarcity cap. So a price view falls out of the inflow view
almost for free — with the important caveat that the mapping is loose.

Fitted on monthly means, 2015-2026, in LOG space (price is multiplicative
and strictly positive, and its residuals are far closer to lognormal than
normal):

    log P ~ inflow%norm + storage + ONI + seasonal harmonics
            + log(scarcity price)

Incremental R2, monthly, in-sample:

    inflow only                       0.332
    + storage                         0.347
    + ONI                             0.398
    + seasonal harmonics              0.463
    + log scarcity price              0.720     <- the thermal cost anchor
    all                               0.754

The scarcity price matters that much because it is the ceiling the market
runs toward under stress, and it is indexed to fuel costs — without it the
model cannot tell a dry 2017 (price ~106 COP/kWh) from a dry 2024 (~676).
It is a published regulatory value that moves slowly, so carrying it
forward at its latest level is defensible; that assumption is stated in
the output rather than hidden.

Uncertainty is honest and wide: the residual sd is ~0.36 in logs, i.e. a
**x1.44 one-sigma multiplicative spread**. Price is reported with that
propagated through, and the intervals should be read as genuinely broad.

    python scripts/sst/price_outlook.py

Output: ~/colombia_hydro/out/price_outlook.json
        ~/colombia_hydro/site/price_outlook.webp
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
RAW = PRIV / "raw"
OUT_JSON = PRIV / "out" / "price_outlook.json"
OUT_PNG = PRIV / "site" / "price_outlook.webp"
QS = [10, 25, 50, 75, 90]


def monthly_frame():
    import perfect_rain_backtest as PR
    import delta_backtest_long as DB
    b = json.load(gzip.open(RAW / "bolsa_daily.json.gz", "rt"))
    es = json.load(gzip.open(RAW / "escasez_daily.json.gz", "rt"))
    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    di = {s: i for i, s in enumerate(dates)}
    y, st, on = d["y"]["NATIONAL"], d["stor"]["NATIONAL"], d["roni"]
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in b.items():
        try:
            p = float(v)
        except (TypeError, ValueError):
            continue
        i = di.get(k)
        if i is None or not np.isfinite(y[i]) or not np.isfinite(st[i]):
            continue
        m = k[:7]
        agg[m]["p"].append(p)
        agg[m]["y"].append(y[i])
        agg[m]["s"].append(st[i])
        agg[m]["o"].append(on[i])
        try:
            agg[m]["e"].append(float(es[k]))
        except (KeyError, TypeError, ValueError):
            pass
    mo = sorted(agg)
    F = {k: np.array([np.mean(agg[m][k]) if agg[m][k] else np.nan for m in mo])
         for k in ("p", "y", "s", "o", "e")}
    F["month"] = np.array([int(m[5:7]) for m in mo])
    F["key"] = np.array(mo)
    return F


def design(y, s, o, month, esc):
    ph = 2 * np.pi * month / 12.0
    return np.column_stack([np.ones(len(y)), y, s, o,
                            np.sin(ph), np.cos(ph),
                            np.log(np.clip(esc, 1, None))])


def fit_cv(F):
    """Blocked (contiguous-year) CV so the residual spread is out-of-sample."""
    ok = np.all(np.isfinite(np.column_stack(
        [F["p"], F["y"], F["s"], F["o"], F["e"]])), axis=1)
    lp = np.log(np.clip(F["p"], 1, None))
    X = design(F["y"], F["s"], F["o"], F["month"], F["e"])
    yrs = np.array([k[:4] for k in F["key"]])
    uy = sorted(set(yrs[ok]))
    resid = []
    for hold in uy:
        te = ok & (yrs == hold)
        tr = ok & (yrs != hold)
        if te.sum() < 3 or tr.sum() < 40:
            continue
        be, *_ = np.linalg.lstsq(X[tr], lp[tr], rcond=None)
        resid.append(lp[te] - X[te] @ be)
    resid = np.concatenate(resid) if resid else np.zeros(1)
    beta, *_ = np.linalg.lstsq(X[ok], lp[ok], rcond=None)
    ins = lp[ok] - X[ok] @ beta
    r2 = 1 - (ins ** 2).sum() / ((lp[ok] - lp[ok].mean()) ** 2).sum()
    return beta, float(np.std(resid)), float(r2), int(ok.sum())


def main() -> int:
    F = monthly_frame()
    beta, sd_oos, r2, n = fit_cv(F)
    print(f"fitted on {n} months; in-sample R2 {r2:.3f}; "
          f"out-of-sample residual sd {sd_oos:.3f} log "
          f"(x{np.exp(sd_oos):.2f} one-sigma)")

    # drive with the seasonal inflow outlook
    nat = json.loads((PRIV / "out" / "national_inflow.json").read_text())
    rows = nat["monthly_forecast"]["months"]
    esc_now = float(F["e"][np.isfinite(F["e"])][-1])
    stor_now = float(F["s"][-1])
    oni_now = float(F["o"][-1])
    print(f"carried forward: scarcity price {esc_now:.0f} COP/kWh, "
          f"storage anom {stor_now:+.1f}, ONI {oni_now:+.2f}")

    # CURRENT-REGIME ANCHOR. The structural fit runs low right now: over
    # the last three months actual/model has a median of ~1.5. Something the
    # model does not carry - fuel costs, contracting, unit outages, bidding
    # behaviour - is holding prices above what water alone explains. Ignoring
    # that ships a level we already know is wrong; extrapolating it forever
    # assumes a transient is permanent. So the recent bias is applied in full
    # at lead 1 and decayed to 1.0 by lead 6, which is the same logic the
    # rain stage uses for its model bias factors.
    Xh = design(F["y"], F["s"], F["o"], F["month"], F["e"])
    okh = np.all(np.isfinite(Xh), axis=1) & np.isfinite(F["p"])
    ratio = F["p"][okh] / np.exp(Xh[okh] @ beta)
    anchor = float(np.median(ratio[-3:])) if okh.sum() >= 3 else 1.0
    print(f"current-regime anchor: actual/model = {anchor:.2f} over the last "
          f"3 months, decayed to 1.00 by lead 6")

    out = []
    for r in rows:
        key = r["month"]
        mn = int(str(key)[5:7])
        # propagate the inflow interval AND the price model's own residual
        centre, lo, hi = r["pct_p50"], r["pct_p10"], r["pct_p90"]
        pr = {}
        for lab, iv in (("p10", lo), ("p50", centre), ("p90", hi)):
            X = design(np.array([iv]), np.array([stor_now]),
                       np.array([oni_now]), np.array([mn]),
                       np.array([esc_now]))
            pr[lab] = float(np.exp((X @ beta)[0]))
        # inflow p10 -> HIGH price, so swap ends
        lead = int(r.get("lead", 1))
        w = max(0.0, 1.0 - (lead - 1) / 5.0)         # 1.0 at lead 1 -> 0 at 6
        adj = 1.0 + (anchor - 1.0) * w
        pr = {k: v * adj for k, v in pr.items()}
        band_lo, band_hi = min(pr["p10"], pr["p90"]), max(pr["p10"], pr["p90"])
        mid = pr["p50"]
        z = 1.2816                                  # 80% interval
        out.append({
            "month": str(key), "regime_adj": round(adj, 3),
            "inflow_p50": centre,
            "price_p50": round(mid, 1),
            "price_p10": round(min(band_lo, mid) / np.exp(z * sd_oos), 1),
            "price_p90": round(max(band_hi, mid) * np.exp(z * sd_oos), 1),
        })

    print(f"\n{'month':10}{'inflow %norm':>14}{'price p10':>11}"
          f"{'price p50':>11}{'price p90':>11}   COP/kWh")
    for r in out:
        print(f"{r['month']:10}{r['inflow_p50']:14.0f}{r['price_p10']:11.0f}"
              f"{r['price_p50']:11.0f}{r['price_p90']:11.0f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"r2_in_sample": round(r2, 3), "regime_anchor": round(anchor, 3), "oos_resid_sd_log": round(sd_oos, 3),
         "one_sigma_multiplier": round(float(np.exp(sd_oos)), 2),
         "scarcity_price_held": esc_now, "months": out}, indent=1))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
