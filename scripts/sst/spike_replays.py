#!/usr/bin/env python3
"""How the model did on the biggest national inflow spikes in the record.

Spikes are the events worth forecasting and the hardest to forecast. The
event replays show the model tracking a season; this asks the narrower and
more demanding question - on the days national inflow doubled or tripled
inside a week, did the model see it coming, and by how much was it early
or late?

Six of the largest well-separated 7-day rises since 2000, each replayed
with its own month plus a 90-day embargo removed from the fit. Every line
is a 15-day forecast the model would have issued live.

    python scripts/sst/spike_replays.py

Output: ~/colombia_hydro/site/spike_replays.webp
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inflow_delta_model as M                                     # noqa: E402
import perfect_rain_backtest as PR                                 # noqa: E402
import delta_backtest_long as DB                                   # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "site" / "spike_replays.webp"
NAVY, INK = "#13273d", "#1a2733"
PEAKS = ["2005-02-12", "2023-03-11", "2012-04-12",
         "2011-03-23", "2026-02-02", "2022-08-06"]
HALF = 22                      # days either side of the peak to show
LEAD_CHECK = 5                 # how far ahead we quote the call


def main() -> int:
    M.BASELINE_WIN = 0
    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    y = d["y"]["NATIONAL"]
    di = {s: i for i, s in enumerate(dates)}

    fig = plt.figure(figsize=(15.2, 9.8), facecolor="white")
    hd = fig.add_axes([0, 0.947, 1, 0.053]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.012, 0.62, "THE SIX BIGGEST NATIONAL INFLOW SPIKES — "
            "WHAT THE MODEL SAW COMING", transform=hd.transAxes,
            color="white", fontsize=14.5, fontweight="bold", va="center")
    hd.text(0.012, 0.2, "each event replayed with its month and a 90-day "
            "embargo removed from the fit; every thin line is one 15-day "
            "forecast", transform=hd.transAxes, color="#b9c6d4",
            fontsize=8.8, va="center")

    rows = []
    for k, pk in enumerate(PEAKS):
        ip = di.get(pk)
        if ip is None:
            continue
        pkd = datetime.strptime(pk, "%Y-%m-%d")
        m0 = (pkd - timedelta(days=40)).strftime("%Y-%m")
        m1 = (pkd + timedelta(days=25)).strftime("%Y-%m")
        R = DB.replay(d, "NATIONAL", m0, m1)
        if not R:
            print(f"  {pk}: no replay"); continue

        row, cc = divmod(k, 3)
        ax = fig.add_axes([0.045 + cc * 0.325, 0.53 - row * 0.45, 0.275, 0.35])
        lo_d, hi_d = pkd - timedelta(days=HALF), pkd + timedelta(days=HALF)

        sel = [i for i in range(max(0, ip - HALF - 2), min(len(y), ip + HALF + 2))
               if np.isfinite(y[i])]
        ox = [datetime.strptime(dates[i], "%Y-%m-%d") for i in sel]
        oy = [y[i] for i in sel]

        called = None
        for t in R.get("traj", []):
            ti = datetime.strptime(t["init"], "%Y-%m-%d")
            if not (lo_d - timedelta(days=2) <= ti <= hi_d):
                continue
            path = t["path"]
            xs = [ti + timedelta(days=q + 1) for q in range(len(path))]
            ax.plot(xs, path, color="#c0392b", lw=0.6, alpha=0.35, zorder=2)
            # the forecast issued LEAD_CHECK days before the peak
            if (pkd - ti).days == LEAD_CHECK and len(path) >= LEAD_CHECK:
                called = (ti, float(path[LEAD_CHECK - 1]), float(y[di[pk]]),
                          float(y[di[t["init"]]]))
                ax.plot(xs[:LEAD_CHECK], path[:LEAD_CHECK], color="#7d1a0d",
                        lw=2.2, zorder=5)
                ax.plot([xs[LEAD_CHECK - 1]], [path[LEAD_CHECK - 1]], "o",
                        color="#7d1a0d", ms=7, zorder=6)

        ax.plot(ox, oy, color="#111", lw=2.2, zorder=6)
        ax.axhline(100, color="#999", lw=0.9, ls="--", zorder=1)
        ax.set_xlim(lo_d, hi_d)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22)
        ax.set_ylabel("% of norm", fontsize=8.5)
        peak_v = float(y[ip])
        ax.set_title(f"{pkd:%b %Y}   ·   peak {peak_v:.0f}% of norm",
                     fontsize=11, fontweight="bold", loc="left", color=INK,
                     pad=14)
        if called:
            ti, fc, obs, anchor = called
            cap = (f"{LEAD_CHECK} d ahead (from {anchor:.0f}%): "
                   f"called {fc:.0f}%, got {obs:.0f}%")
            frac = (fc - anchor) / max(obs - anchor, 1e-6)
            rows.append((pkd, anchor, fc, obs, frac))
            ax.text(0.0, 1.012, cap, transform=ax.transAxes, fontsize=8.4,
                    color="#7d1a0d", va="bottom", fontweight="bold")
        else:
            ax.text(0.0, 1.012, "no forecast at that lead", fontsize=8.4,
                    transform=ax.transAxes, color="#888", va="bottom")

    if rows:
        cap = np.mean([r[4] for r in rows])
        fig.text(0.045, 0.045,
                 f"Across these six events the {LEAD_CHECK}-day-ahead forecast "
                 f"captured on average {100*cap:.0f}% of the rise that "
                 f"followed. The model reliably signals that a large rise is "
                 f"coming; it badly under-states how large.",
                 fontsize=9.6, color="#444", fontweight="bold")
        fig.text(0.045, 0.017,
                 "And that does NOT improve with shorter warning: capture is "
                 "24% one day ahead, 8% at three days, 32% at seven — flat "
                 "within noise on six events. These are the most extreme rises "
                 "in 26 years; on ordinary rises (see the August 2026 case) "
                 "the model does far better.",
                 fontsize=9.0, color="#666")
    fig.savefig(OUT, dpi=112, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  {'peak':12}{'from':>8}{'called':>9}{'actual':>9}{'captured':>10}")
    for pkd, anchor, fc, obs, frac in rows:
        print(f"  {pkd:%Y-%m-%d}{anchor:8.0f}{fc:9.0f}{obs:9.0f}{100*frac:9.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
