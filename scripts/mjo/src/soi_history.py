#!/usr/bin/env python3
"""Historical SOI: 130+ years of monthly Troup SOI plus the consistent daily
record with 30/90-day running means and all-time record lines.

Two series from the Queensland Government's LongPaddock service (CC BY 4.0), both on
Troup's fixed 1887–1989 base. They stay separate panels with their own records
rather than a splice: calendar-month means and 30/90-day running means are
different statistics.
  · Monthly Troup SOI, the whole record — each value 10·(ΔP−m)/σ from
    monthly-mean Tahiti−Darwin MSLP, one method across the whole span. Read from
    MonthlySOIPhase1887-1989Base.txt (SOI and SOI phase by month): the plain
    monthly file, MonthlySOI1887-1989Base.txt, starts only in June 1991.
  · Daily Troup SOI (DailySOI1887-1989Base.txt), Jun 1991–present — refreshed
    each cycle by soi_forecast.fetch_obs.

Figure (three panels):
  top    — the monthly record: monthly SOI (light) + 3-month mean (bold),
           record monthly / 3-month max-min lines (value + date set).
  middle — the daily era: 30-day and 90-day running means with their records.
  bottom — the last 24 months: daily bars + the running means and records.

    python src/soi_history.py --out ../../assets/sst/soi_history.webp
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from soi_forecast import SOI_URL, fetch_obs
from webget import cached

MIN_COVER = 0.8  # a window must be ≥80% present to count
MONTHLY_URL = SOI_URL.replace("DailySOI", "MonthlySOIPhase")  # same directory and base as the daily file


def running(series: pd.Series, days: int) -> pd.Series:
    """Trailing running mean on the daily calendar, NaN where coverage is poor."""
    s = series.asfreq("D")
    return s.rolling(days, min_periods=int(days * MIN_COVER)).mean()


def fetch_monthly(cache: Path) -> pd.Series:
    """LongPaddock monthly Troup SOI (1887–1989 base) -> Series indexed by month start.

    Reads one "Year Month SOI [Phase]" row per month, or one row per year with the twelve months across. The copy is
    reused for a day, then refreshed with a conditional GET."""
    rows = []
    for ln in cached(MONTHLY_URL, cache, max_age_h=24).splitlines():
        tok = ln.split()
        if tok and tok[0].isdigit() and 1800 < int(tok[0]) < 2200:
            rows.append(tok)
    # a month row: an integer month second, the SOI third (the phase may follow); a year row: twelve values
    n_month = sum(3 <= len(r) <= 4 and r[1].isdigit() and 1 <= int(r[1]) <= 12 for r in rows)
    n_year = sum(len(r) == 13 for r in rows)
    if max(n_month, n_year) <= len(rows) / 2:
        raise RuntimeError(f"LongPaddock monthly SOI: unknown layout ({len(rows)} rows)")
    vals = {}
    for r in rows:
        cells = [(int(r[1]), r[2])] if n_month > n_year else list(enumerate(r[1:13], 1))
        for m, tok in cells:
            try:
                v = float(tok)
            except ValueError:
                continue
            if abs(v) < 99:  # drop missing-value sentinels
                vals[pd.Timestamp(int(r[0]), m, 1)] = v
    s = pd.Series(vals).sort_index()
    if len(s) < 1000:
        raise RuntimeError(f"LongPaddock monthly SOI parse looks wrong ({len(s)} values, first {s.index.min()})")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../assets/sst/soi_history.webp")
    ap.add_argument("--cache", default="data/soi/DailySOI.txt")
    args = ap.parse_args()
    obs = fetch_obs(Path(args.cache))
    soi = obs["SOI"].dropna()
    mon = fetch_monthly(Path(args.cache).parent / "MonthlySOIPhase1887-1989Base.txt")
    mon3 = mon.rolling(3, min_periods=3).mean()
    m30, m90 = running(soi, 30), running(soi, 90)
    recs = {}
    for name, s in (("30", m30), ("90", m90), ("mon", mon), ("mon3", mon3)):
        recs[name] = dict(hi=float(s.max()), hi_d=s.idxmax(), lo=float(s.min()), lo_d=s.idxmin())
    print(
        f"  monthly record ({mon.index[0]:%Y}–): {recs['mon']['lo']:+.1f} "
        f"({recs['mon']['lo_d']:%b %Y}) … {recs['mon']['hi']:+.1f} ({recs['mon']['hi_d']:%b %Y})"
    )
    print(
        f"  record 30-d: {recs['30']['lo']:+.1f} ({recs['30']['lo_d']:%b %Y}) … "
        f"{recs['30']['hi']:+.1f} ({recs['30']['hi_d']:%b %Y})"
    )
    print(
        f"  record 90-d: {recs['90']['lo']:+.1f} ({recs['90']['lo_d']:%b %Y}) … "
        f"{recs['90']['hi']:+.1f} ({recs['90']['hi_d']:%b %Y})"
    )

    fig, (aM, a0, a1) = plt.subplots(
        3, 1, figsize=(11.4, 12.4), gridspec_kw=dict(height_ratios=[1.05, 1, 0.95], hspace=0.32)
    )
    C30, C90 = "#64b5f6", "#0d47a1"
    CM, CM3 = "#ce93d8", "#6a1b9a"

    def rec_lines(ax, names):
        LBL = {"30": "30-d", "90": "90-d", "mon": "monthly", "mon3": "3-mo"}
        COL = {"30": C30, "90": C90, "mon": CM, "mon3": CM3}
        for name in names:
            for key, lab in (("hi", "max"), ("lo", "min")):
                v = recs[name][key]
                d = recs[name][key + "_d"]
                ax.axhline(v, color=COL[name], lw=0.9, ls=":", alpha=0.85)
                ax.annotate(
                    f"record {LBL[name]} {lab} {v:+.1f} ({d:%b %Y})",
                    xy=(1.0, v),
                    xycoords=("axes fraction", "data"),
                    xytext=(-4, 2 if key == "hi" else -9),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=COL[name],
                    ha="right",
                    va="bottom" if key == "hi" else "top",
                    zorder=6,
                    # a white ground: bars and lines can run through the min labels
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
                )

    def troup_lines(ax):
        """The dashed ±8 lines, labelled in every panel: the classic Troup threshold for a sustained event (the
        Bureau of Meteorology now uses ±7, as the forecast view does)."""
        for y, lab in (
            (8, "+8, classic Troup threshold: sustained above ⇒ La Niña"),
            (-8, "−8, classic Troup threshold: sustained below ⇒ El Niño"),
        ):
            ax.axhline(y, color="0.65", lw=0.8, ls="--")
            ax.annotate(
                lab,
                xy=(0.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(4, 2 if y > 0 else -2),
                textcoords="offset points",
                fontsize=6.8,
                color="0.4",
                va="bottom" if y > 0 else "top",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
            )

    # ── monthly record ──
    aM.plot(mon.index, mon.values, color=CM, lw=0.55, alpha=0.8, label="monthly SOI")
    aM.plot(mon3.index, mon3.values, color=CM3, lw=1.3, label="3-month mean")
    aM.axhline(0, color="0.4", lw=0.8)
    troup_lines(aM)
    rec_lines(aM, ("mon", "mon3"))
    aM.set_ylim(recs["mon"]["lo"] - 6, recs["mon"]["hi"] + 6)  # room for record labels
    aM.set_xlim(mon.index[0], mon.index[-1] + pd.Timedelta(days=700))
    aM.grid(True, alpha=0.2)
    aM.set_title(
        f"Monthly SOI (LongPaddock Troup, 1887–1989 base) — {mon.index[0]:%Y}–{mon.index[-1]:%b %Y}: "
        f"latest {mon.iloc[-1]:+.1f}",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    aM.set_ylabel("SOI")
    aM.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # ── daily record ──
    a0.plot(m30.index, m30.values, color=C30, lw=0.7, alpha=0.85, label="30-day mean")
    a0.plot(m90.index, m90.values, color=C90, lw=1.5, label="90-day mean")
    a0.axhline(0, color="0.4", lw=0.8)
    troup_lines(a0)
    rec_lines(a0, ("30", "90"))
    a0.set_ylim(recs["30"]["lo"] - 6, recs["30"]["hi"] + 6)  # room for record labels
    a0.set_xlim(m30.index[0], m30.index[-1] + pd.Timedelta(days=120))
    a0.grid(True, alpha=0.2)
    a0.set_title(
        f"Daily SOI (LongPaddock Troup, 1887–1989 base) — the consistent daily record, "
        f"{soi.index[0]:%b %Y} – {soi.index[-1]:%b %d %Y}",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    a0.set_ylabel("SOI")
    a0.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # ── last 24 months ──
    t0 = soi.index[-1] - pd.DateOffset(months=24)
    d = soi[soi.index >= t0]
    a1.bar(d.index, d.values, width=1.0, color=np.where(d.values >= 0, "#90caf9", "#ffab91"), alpha=0.55, linewidth=0)
    a1.plot(m30[m30.index >= t0].index, m30[m30.index >= t0].values, color=C30, lw=1.6)
    a1.plot(m90[m90.index >= t0].index, m90[m90.index >= t0].values, color=C90, lw=2.2)
    a1.axhline(0, color="0.4", lw=0.8)
    troup_lines(a1)
    rec_lines(a1, ("30", "90"))
    a1.set_xlim(t0, soi.index[-1] + pd.Timedelta(days=10))
    a1.grid(True, alpha=0.2)
    cur30 = m30.dropna().iloc[-1]
    cur90 = m90.dropna().iloc[-1]
    a1.set_title(
        f"Last 24 months — daily values (bars) · latest 30-d {cur30:+.1f}, 90-d {cur90:+.1f}",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    a1.set_ylabel("SOI")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.05)
    fig.text(
        0.5,
        0.005,
        f"Source: Queensland Government, LongPaddock (CC BY 4.0) · monthly and daily Troup SOI, fixed 1887–1989 base, "
        f"monthly {mon.index[0]:%Y}–present, daily Jun 1991–present\n"
        "(calendar-month and running-mean records are kept apart) · "
        "10·(ΔP−m)/σ, ΔP = Tahiti−Darwin MSLP · running means require ≥80% daily coverage",
        ha="center",
        fontsize=7.5,
        color="0.4",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  latest: 30-d {cur30:+.1f} · 90-d {cur90:+.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
