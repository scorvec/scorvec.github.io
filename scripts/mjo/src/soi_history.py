#!/usr/bin/env python3
"""Historical daily SOI: the full consistent LongPaddock/BoM daily record with
30-day and 90-day running means and their all-time record lines.

Data: LongPaddock's daily Troup SOI (10·(ΔP − m)/σ on the fixed 1887–1989 base,
ΔP = Tahiti − Darwin MSLP) — the one series computed the same way end to end.
Daily values exist from June 1991; the file is refreshed each cycle by
soi_forecast.fetch_obs, so this just reads the same cache.

Figure (two panels):
  top    — the full record: 30-day mean (light) and 90-day mean (bold), with
           horizontal record-max/min lines for each smoothing (value + the date
           the record was set), zero line, and ±8 Troup El Niño/La Niña guides.
  bottom — the last 24 months: daily SOI as faint bars + the same two running
           means and record lines, so today reads against the whole record.

    python src/soi_history.py --out ../../assets/sst/soi_history.webp
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from soi_forecast import fetch_obs

MIN_COVER = 0.8                      # a window must be ≥80% present to count


def running(series: pd.Series, days: int) -> pd.Series:
    """Trailing running mean on the daily calendar, NaN where coverage is poor."""
    s = series.asfreq("D")
    return s.rolling(days, min_periods=int(days * MIN_COVER)).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../assets/sst/soi_history.webp")
    ap.add_argument("--cache", default="data/soi/DailySOI.txt")
    args = ap.parse_args()
    obs = fetch_obs(Path(args.cache))
    soi = obs["SOI"].dropna()
    m30, m90 = running(soi, 30), running(soi, 90)
    recs = {}
    for name, s in (("30", m30), ("90", m90)):
        recs[name] = dict(hi=float(s.max()), hi_d=s.idxmax(),
                          lo=float(s.min()), lo_d=s.idxmin())
    print(f"  record 30-d: {recs['30']['lo']:+.1f} ({recs['30']['lo_d']:%b %Y}) … "
          f"{recs['30']['hi']:+.1f} ({recs['30']['hi_d']:%b %Y})")
    print(f"  record 90-d: {recs['90']['lo']:+.1f} ({recs['90']['lo_d']:%b %Y}) … "
          f"{recs['90']['hi']:+.1f} ({recs['90']['hi_d']:%b %Y})")

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.4, 8.6),
                                 gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.28))
    C30, C90 = "#64b5f6", "#0d47a1"

    def rec_lines(ax):
        for name, col in (("30", C30), ("90", C90)):
            for key, lab in (("hi", "max"), ("lo", "min")):
                v = recs[name][key]; d = recs[name][key + "_d"]
                ax.axhline(v, color=col, lw=0.9, ls=":", alpha=0.85)
                ax.annotate(f"record {name}-d {lab} {v:+.1f} ({d:%b %Y})",
                            xy=(1.0, v), xycoords=("axes fraction", "data"),
                            xytext=(-4, 2 if key == "hi" else -9),
                            textcoords="offset points", fontsize=6.8, color=col,
                            ha="right", va="bottom" if key == "hi" else "top")

    # ── full record ──
    a0.plot(m30.index, m30.values, color=C30, lw=0.7, alpha=0.85, label="30-day mean")
    a0.plot(m90.index, m90.values, color=C90, lw=1.5, label="90-day mean")
    a0.axhline(0, color="0.4", lw=0.8)
    for y, lab in ((8, "sustained > +8 ⇒ La Niña"), (-8, "sustained < −8 ⇒ El Niño")):
        a0.axhline(y, color="0.65", lw=0.8, ls="--")
        a0.text(m30.index[0], y, " " + lab, fontsize=6.8, color="0.45",
                va="bottom" if y > 0 else "top")
    rec_lines(a0)
    a0.set_xlim(m30.index[0], m30.index[-1] + pd.Timedelta(days=120))
    a0.grid(True, alpha=0.2)
    a0.set_title(f"Daily SOI (Troup, 1887–1989 base) — full consistent daily record, "
                 f"{soi.index[0]:%b %Y} – {soi.index[-1]:%b %d %Y}",
                 fontsize=11.5, fontweight="bold", loc="left")
    a0.set_ylabel("SOI"); a0.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # ── last 24 months ──
    t0 = soi.index[-1] - pd.DateOffset(months=24)
    d = soi[soi.index >= t0]
    a1.bar(d.index, d.values, width=1.0, color=np.where(d.values >= 0, "#90caf9", "#ffab91"),
           alpha=0.55, linewidth=0)
    a1.plot(m30[m30.index >= t0].index, m30[m30.index >= t0].values, color=C30, lw=1.6)
    a1.plot(m90[m90.index >= t0].index, m90[m90.index >= t0].values, color=C90, lw=2.2)
    a1.axhline(0, color="0.4", lw=0.8)
    a1.axhline(8, color="0.65", lw=0.8, ls="--"); a1.axhline(-8, color="0.65", lw=0.8, ls="--")
    rec_lines(a1)
    a1.set_xlim(t0, soi.index[-1] + pd.Timedelta(days=10))
    a1.grid(True, alpha=0.2)
    cur30 = m30.dropna().iloc[-1]; cur90 = m90.dropna().iloc[-1]
    a1.set_title(f"Last 24 months — daily values (bars) · latest 30-d {cur30:+.1f}, "
                 f"90-d {cur90:+.1f}", fontsize=10.5, fontweight="bold", loc="left")
    a1.set_ylabel("SOI")
    fig.text(0.5, 0.005,
             "LongPaddock (Qld DES) daily Troup SOI — 10·(ΔP−m)/σ, ΔP = Tahiti−Darwin MSLP, "
             "fixed 1887–1989 monthly base · running means require ≥80% daily coverage · "
             "record lines span the full daily record",
             ha="center", fontsize=7.5, color="0.4")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  latest: 30-d {cur30:+.1f} · 90-d {cur90:+.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
