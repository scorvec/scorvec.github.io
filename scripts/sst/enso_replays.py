#!/usr/bin/env python3
"""Daily forecast replays through El Nino and La Nina events.

delta_backtest_long keeps trajectories only for its single headline event.
This keeps them for every window, because the question "does it work in a
La Nina as well as an El Nino" cannot be answered from one episode.

For each window the model is fitted with that window AND a 90-day embargo
removed, then a fresh 15-day forecast is launched from every day inside
it. Every line on the chart is therefore a forecast the model would have
produced live, drawn over what actually happened.

    python scripts/sst/enso_replays.py

Output: ~/colombia_hydro/site/enso_replays.webp
"""
from __future__ import annotations

import json
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
OUT = PRIV / "site" / "enso_replays.webp"
NAVY, INK = "#13273d", "#1a2733"

EVENTS = [
    ("2015-08:2016-03", "El Niño 2015-16", "strongest in the record, ONI +2.65",
     "#c0392b"),
    ("2023-08:2024-03", "El Niño 2023-24", "the most recent strong event",
     "#c0392b"),
    ("2010-09:2011-04", "La Niña 2010-11", "ONI −1.64, the wet extreme",
     "#1f6fb4"),
    ("2007-11:2008-05", "La Niña 2007-08", "ONI −1.64, a second wet case",
     "#1f6fb4"),
]


def main() -> int:
    M.BASELINE_WIN = 0                 # match the operational configuration
    d = DB.add_national(PR.load_all())

    fig = plt.figure(figsize=(15.0, 10.6), facecolor="white")
    hd = fig.add_axes([0, 0.947, 1, 0.053]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.012, 0.62, "NATIONAL INFLOW — EVERY DAILY FORECAST, "
            "THROUGH FOUR ENSO EVENTS", transform=hd.transAxes, color="white",
            fontsize=14.5, fontweight="bold", va="center")
    hd.text(0.012, 0.2, "each thin line is one 15-day forecast; the window and "
            "a 90-day embargo are removed from training before any of them "
            "are made", transform=hd.transAxes, color="#b9c6d4", fontsize=8.8,
            va="center")

    summary = {}
    for k, (ev, title, sub, col) in enumerate(EVENTS):
        a0, a1 = ev.split(":")
        R = DB.replay(d, "NATIONAL", a0, a1)
        if not R:
            print(f"  {ev}: no replay"); continue
        row, cc = divmod(k, 2)
        ax = fig.add_axes([0.055 + cc * 0.495, 0.53 - row * 0.45, 0.43, 0.35])

        # observed
        dates = np.array([str(x) for x in d["dates"]])
        y = d["y"]["NATIONAL"]
        sel = [i for i, s_ in enumerate(dates)
               if a0 <= s_[:7] <= a1 and np.isfinite(y[i])]
        ox = [datetime.strptime(dates[i], "%Y-%m-%d") for i in sel]
        oy = [y[i] for i in sel]

        for j, t in enumerate(R.get("traj", [])):
            ti = datetime.strptime(t["init"], "%Y-%m-%d")
            path = t["path"]
            xs = [ti + timedelta(days=q + 1) for q in range(len(path))]
            ax.plot(xs, path, color=col, lw=0.55, alpha=0.30, zorder=2)
        ax.plot(ox, oy, color="#111", lw=2.0, zorder=6, label="observed")
        ax.axhline(100, color="#999", lw=1.0, ls="--", zorder=1)
        ax.plot([], [], color=col, lw=1.4, alpha=0.75,
                label=f"{len(R.get('traj', []))} forecasts, 15 d each")

        sk = R["leads"].get(10, {}).get("skill_vs_persistence")
        sk1 = R["leads"].get(1, {}).get("skill_vs_persistence")
        summary[ev] = (sk1, sk)
        ax.set_title(f"{title}", fontsize=12, fontweight="bold", loc="left",
                     color=INK, pad=15)
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8.8,
                color="#666", va="bottom")
        ax.text(1.0, 1.015, f"beats persistence by {100*sk:.0f}% at 10 days"
                if sk else "", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.8, color="#2c6e49", fontweight="bold")
        ax.set_ylabel("% of norm", fontsize=9)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=8)

    fig.text(0.055, 0.035,
             "Read the spread, not the individual lines: where the bundle sits "
             "below the black line the model was too dry, above it too wet. "
             "Through all four events the bundle tracks the observed path "
             "rather than drifting back to normal, which is the property that "
             "matters in a prolonged anomaly.",
             fontsize=9.2, color="#444", wrap=True)
    fig.savefig(OUT, dpi=112, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")
    for ev, (s1, s10) in summary.items():
        print(f"  {ev}: skill vs persistence  h1 {s1:+.3f}   h10 {s10:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
