#!/usr/bin/env python3
"""Two national back-tested cases for the methodology report.

A skill table says the model is right on average. A trader wants to know
what it did on the days that mattered. These are the two hardest kinds of
day in this record, both at NATIONAL level and both genuinely out of
sample:

  A  2015-16 super El Nino - a slow collapse from ~102% of norm to 52%
     over nine months. Each month is forecast at leads 1, 3 and 6 with
     that month and its neighbours withheld from the fit.

  B  17-18 Aug 2026 - a sudden near-tripling of national inflow. The fan
     is re-issued from 12 and 15 August using only the rain forecast that
     existed those mornings, with the whole of August 2026 plus a 90-day
     embargo removed from the fit.

    python scripts/sst/backtest_cases.py

Output: ~/colombia_hydro/site/backtest_cases.webp
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
PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "site" / "backtest_cases.webp"
NAVY, INK = "#13273d", "#1a2733"


def main() -> int:
    nat = json.loads((PRIV / "out" / "national_inflow.json").read_text())
    sc = nat["showcase"]

    fig = plt.figure(figsize=(12.4, 9.4), facecolor="white")
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.013, 0.62, "BACK-TESTED CASES — NATIONAL INFLOW",
            transform=hd.transAxes, color="white", fontsize=14.5,
            fontweight="bold", va="center")
    hd.text(0.013, 0.2, "both genuinely out of sample: the event period and a "
            "surrounding embargo are removed before fitting",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")

    # ---------------- A: 2015-16 drought ----------------
    ax = fig.add_axes([0.07, 0.56, 0.88, 0.33])
    ks = list(sc)
    x = np.arange(len(ks))
    obs = np.array([sc[k]["obs"] for k in ks], float)
    ax.axhline(100, color="#999", lw=1.0, ls="--", zorder=1)
    for lead, col, lw in ((1, "#1f7a4d", 2.0), (3, "#b35806", 1.7),
                          (6, "#7b5ea7", 1.7)):
        v = np.array([sc[k].get(f"lead{lead}", np.nan) for k in ks], float)
        ax.plot(x, v, "o-", color=col, lw=lw, ms=4.5,
                label=f"forecast issued {lead} month{'s' if lead > 1 else ''} ahead")
    ax.plot(x, obs, "o-", color="#111", lw=2.8, ms=6, zorder=6,
            label="what actually happened")
    ax.set_xticks(x); ax.set_xticklabels(ks, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("national inflow, % of norm", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 0.0))
    ax.set_title("A · 2015-16 super El Niño — a nine-month "
                 "collapse to 52% of norm", fontsize=11.5, fontweight="bold",
                 loc="left", color=INK, pad=16)
    e1 = np.nanmean(np.abs([sc[k].get("lead1", np.nan) for k in ks] - obs))
    e6 = np.nanmean(np.abs([sc[k].get("lead6", np.nan) for k in ks] - obs))
    # captions sit ABOVE the axes, never over the data
    ax.text(1.0, 1.02, f"mean error {e1:.1f} pts at 1 month, "
            f"{e6:.1f} pts at 6 months", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color="#555")

    # ---------------- B: Aug 2026 spike ----------------
    ax2 = fig.add_axes([0.07, 0.09, 0.88, 0.34])
    import perfect_rain_backtest as PR
    import delta_backtest_long as DB
    d = DB.add_national(PR.load_all())
    dates = np.array([str(v) for v in d["dates"]])
    y = d["y"]["NATIONAL"]
    sel = [i for i, s_ in enumerate(dates)
           if "2026-08-01" <= s_ <= "2026-08-31" and np.isfinite(y[i])]
    ox = [datetime.strptime(dates[i], "%Y-%m-%d") for i in sel]
    oy = [y[i] for i in sel]
    ax2.axhline(100, color="#999", lw=1.0, ls="--", zorder=1)
    ax2.plot(ox, oy, "o-", color="#111", lw=2.6, ms=5, zorder=6,
             label="what actually happened")
    # the two re-issued forecasts, as reported by hindcast_fan
    issues = [("2026-08-12", 49, [(17, 99, 75, 132)], "#1f7a4d", -26),
              ("2026-08-15", 59, [(17, 105, 85, 132)], "#b35806", 12)]
    for init, anchor, pts, col, dy_lab in issues:
        d0 = datetime.strptime(init, "%Y-%m-%d")
        for day, med, lo, hi in pts:
            dt = datetime(2026, 8, day)
            ax2.plot([d0, dt], [anchor, med], "--", color=col, lw=1.8,
                     zorder=4)
            ax2.plot([dt, dt], [lo, hi], color=col, lw=6, alpha=0.30,
                     solid_capstyle="butt", zorder=3)
            ax2.plot([dt], [med], "o", color=col, ms=8, zorder=5,
                     label=f"forecast issued {d0:%d %b} (anchor {anchor}%)")
            # the two issues land within 6 points of each other, so their
            # labels are offset vertically or they overprint
            ax2.annotate(f"issued {d0:%d %b}: {med}%  [{lo}–{hi}]", (dt, med),
                         textcoords="offset points", xytext=(16, dy_lab),
                         fontsize=8.5, color=col, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.set_ylabel("national inflow, % of norm", fontsize=9.5)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 0.98))
    ax2.set_title("B · 17-18 August 2026 — a sudden near-tripling, "
                  "called five days ahead", fontsize=11.5, fontweight="bold",
                  loc="left", color=INK, pad=16)
    ax2.text(1.0, 1.02, "observed 132% sits at the top edge of the interval "
             "— direction right, magnitude under-shot",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             color="#555")
    ax2.set_ylim(min(oy) - 8, max(oy) + 14)          # headroom for the legend
    ax2.set_xlim(ox[0] - timedelta(days=1), ox[-1] + timedelta(days=5))

    fig.savefig(OUT, dpi=115, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  A: 2015-16, mean error {e1:.1f} pts (1 mo) / {e6:.1f} pts (6 mo)")
    print(f"  B: Aug 2026 spike, called 5 days ahead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
