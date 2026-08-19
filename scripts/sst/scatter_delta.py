#!/usr/bin/env python3
"""Out-of-sample scatter: forecast vs observed inflow DELTA at a fixed lead.

The delta at lead h is what a trader actually takes a position on: not
"what will inflow be" but "how far will it move from where it is now",

    observed  = y[t0+h] - y[t0]        predicted = y_hat[t0+h] - y[t0]

Zero on the y-axis is the persistence forecast, so any point away from
the vertical zero line is the model taking a position, and the scatter's
tilt toward the 1:1 line is whether those positions paid.

Each event window is replayed with that window AND a 90-day embargo
removed from training, so every point is out-of-sample: a forecast the
model would have issued that morning, having never seen the event.

    python scripts/sst/scatter_delta.py [--lead 10] [--series NATIONAL]

Outputs: ~/colombia_hydro/out/scatter_delta_h{lead}.json
         ~/colombia_hydro/site/scatter_delta_h{lead}.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import inflow_delta_model as M                                      # noqa: E402
import delta_backtest_long as DB                                    # noqa: E402

PRIV = Path.home() / "colombia_hydro"
EMBARGO = 90
EVENTS = [("2015-08:2016-01", "2015-16 super El Nino"),
          ("2023-08:2024-03", "2023-24 El Nino"),
          ("2009-08:2010-03", "2009-10 El Nino"),
          ("2010-09:2011-04", "2010-11 La Nina")]


def pairs_for(d, series, a0, a1, lead):
    """(predicted delta, observed delta, y0, date) for every init in the window."""
    dates = d["dates"]
    y, rain = d["y"][series], d["rain"][series]
    roni, stor = d["roni"], d["stor"][series]
    n = len(y)
    ym = np.array([str(x)[:7] for x in dates])
    inper = (ym >= a0) & (ym <= a1)
    hi = np.where(inper)[0]
    if len(hi) < 30:
        return None
    tr = ~inper
    tr[max(0, hi[0] - EMBARGO):min(n, hi[-1] + EMBARGO + 1)] = False
    tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
    X, dy = M.design(rain, y, roni, stor, tau, lag)
    m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
    beta = M.fit(X, dy, m)
    sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
    kf_h, ks_h = M.ema(rain, tau), M.ema(rain, M.TAU_SLOW)
    P, O, Y0, DT = [], [], [], []
    for i0 in hi:
        j = i0 + lead
        if not np.isfinite(y[i0]) or j >= n or not np.isfinite(y[j]):
            continue
        sim = PR.fwd(beta, off, sh, kf_h, ks_h, rain, roni, stor,
                     tau, lag, i0, y[i0], None)
        v = sim[lead - 1]
        if not np.isfinite(v):
            continue
        P.append(y[i0] + off[lead - 1] + sh[lead - 1] * (v - y[i0]) - y[i0])
        O.append(y[j] - y[i0])
        Y0.append(y[i0]); DT.append(str(dates[i0]))
    if len(P) < 20:
        return None
    P, O = np.asarray(P), np.asarray(O)
    sl = float(np.dot(P, O) / max(np.dot(P, P), 1e-9))
    return {"tau": int(tau), "lag": int(lag), "n": len(P),
            "pred": P.tolist(), "obs": O.tolist(), "y0": Y0, "dates": DT,
            "r": round(float(np.corrcoef(P, O)[0, 1]), 3),
            "rmse": round(float(np.sqrt(np.mean((P - O) ** 2))), 2),
            "rmse_persistence": round(float(np.sqrt(np.mean(O ** 2))), 2),
            "skill": round(1 - float(np.sqrt(np.mean((P - O) ** 2)))
                           / max(float(np.sqrt(np.mean(O ** 2))), 1e-9), 3),
            "slope_obs_on_pred": round(sl, 3),
            "hit_rate_sign": round(float(np.mean(np.sign(P) == np.sign(O))), 3),
            "mean_level": round(float(np.mean(Y0)), 1)}


def figure(res, lead, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    ev = [(k, lab) for k, lab in EVENTS if k in res]
    fig = plt.figure(figsize=(15.6, 8.8))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.012, 0.62, f"FORECAST vs OBSERVED INFLOW CHANGE — "
            f"{lead}-DAY LEAD, {series}", transform=hd.transAxes, color="white",
            fontsize=15, fontweight="bold", va="center")
    hd.text(0.012, 0.2, "every point is out-of-sample: the event window and a "
            "90-day embargo were removed from training",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    lim = 0
    for k, _ in ev:
        lim = max(lim, np.percentile(np.abs(res[k]["obs"]), 99.5),
                  np.percentile(np.abs(res[k]["pred"]), 99.5))
    lim = float(np.ceil(lim / 10) * 10)
    for i, (k, lab) in enumerate(ev):
        ax = fig.add_axes([0.048 + i * 0.238, 0.10, 0.195, 0.74])
        R = res[k]
        o, p, y0 = np.asarray(R["obs"]), np.asarray(R["pred"]), np.asarray(R["y0"])
        sc = ax.scatter(o, p, c=y0, cmap="RdYlBu", s=22, alpha=0.85,
                        edgecolor="white", linewidth=0.3, vmin=40, vmax=160)
        ax.plot([-lim, lim], [-lim, lim], color="#111", lw=1.2, ls="--",
                zorder=1, label="perfect (1:1)")
        b = np.polyfit(o, p, 1)
        xs = np.array([-lim, lim])
        ax.plot(xs, b[0] * xs + b[1], color="#c62828", lw=1.6, zorder=2,
                label=f"fit, slope {b[0]:.2f}")
        ax.axhline(0, color="0.5", lw=0.9)
        ax.axvline(0, color="0.5", lw=0.9)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("observed change, pts of norm", fontsize=9)
        if i == 0:
            ax.set_ylabel(f"forecast change at +{lead} d", fontsize=9)
        ax.set_title(f"{lab}\nr={R['r']:.2f}  skill={R['skill']:+.2f}  "
                     f"sign {R['hit_rate_sign']*100:.0f}%  n={R['n']}",
                     fontsize=9.8, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.2, loc="upper left")
        if i == len(ev) - 1:
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
            cb.set_label("inflow at issue, % of norm", fontsize=8)
            cb.ax.tick_params(labelsize=7.5)
    fig.text(0.048, 0.022, "the vertical zero line IS the persistence forecast — "
             "points away from it are the model taking a position, and tilt toward "
             "the dashed 1:1 line is whether those positions paid. "
             "Colour shows how wet the basin already was at issue.",
             fontsize=8, color="#5a6b7a")
    png = PRIV / "site" / f"scatter_delta_h{lead}.webp"
    fig.savefig(png, dpi=118); plt.close(fig)
    print(f"wrote {png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--series", default="NATIONAL")
    ap.add_argument("--baseline", type=int, default=365)
    a = ap.parse_args()
    M.BASELINE_WIN = a.baseline
    d = DB.add_national(PR.load_all())
    res = {}
    print(f"{a.series}, lead +{a.lead} d — out-of-sample event replays\n")
    print(f"{'event':24}{'n':>5}{'r':>7}{'slope':>7}{'RMSE':>7}{'pers':>7}"
          f"{'skill':>8}{'sign%':>7}{'mean lvl':>9}")
    for k, lab in EVENTS:
        a0, a1 = k.split(":")
        R = pairs_for(d, a.series, a0, a1, a.lead)
        if not R:
            continue
        res[k] = R
        print(f"{lab:24}{R['n']:5d}{R['r']:7.3f}{R['slope_obs_on_pred']:7.2f}"
              f"{R['rmse']:7.2f}{R['rmse_persistence']:7.2f}{R['skill']:+8.3f}"
              f"{R['hit_rate_sign']*100:7.0f}{R['mean_level']:9.1f}", flush=True)
    if not res:
        return 1
    allp = np.concatenate([res[k]["pred"] for k in res])
    allo = np.concatenate([res[k]["obs"] for k in res])
    pooled = {"n": int(len(allp)),
              "r": round(float(np.corrcoef(allp, allo)[0, 1]), 3),
              "rmse": round(float(np.sqrt(np.mean((allp - allo) ** 2))), 2),
              "rmse_persistence": round(float(np.sqrt(np.mean(allo ** 2))), 2),
              "hit_rate_sign": round(float(np.mean(np.sign(allp) == np.sign(allo))), 3)}
    pooled["skill"] = round(1 - pooled["rmse"] / pooled["rmse_persistence"], 3)
    print(f"\n{'POOLED':24}{pooled['n']:5d}{pooled['r']:7.3f}{'':7}"
          f"{pooled['rmse']:7.2f}{pooled['rmse_persistence']:7.2f}"
          f"{pooled['skill']:+8.3f}{pooled['hit_rate_sign']*100:7.0f}")
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "series": a.series, "lead_days": a.lead, "pooled": pooled,
           "events": {k: {kk: vv for kk, vv in res[k].items()
                          if kk not in ("pred", "obs", "y0", "dates")} for k in res},
           "note": "delta = y[t0+lead] - y[t0]; zero on the forecast axis is "
                   "persistence. Out-of-sample: window + 90-day embargo removed."}
    (PRIV / "out" / f"scatter_delta_h{a.lead}.json").write_text(json.dumps(out, indent=1))
    try:
        figure(res, a.lead, a.series)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
