#!/usr/bin/env python3
"""Historical SOI: 130+ years of monthly Troup SOI plus the consistent daily
record with 30/90-day running means and all-time record lines.

Two consistently-computed series (small base differences keep them as separate
panels rather than a splice):
  · BoM monthly Troup SOI, 1876–present (ftp.bom.gov.au soiplaintext.html) —
    each value 10·(ΔP−m)/σ from monthly-mean Tahiti−Darwin MSLP, one method
    across the whole span.
  · LongPaddock daily Troup SOI (fixed 1887–1989 base), Jun 1991–present —
    refreshed each cycle by soi_forecast.fetch_obs.

Figure (three panels):
  top    — the monthly record since 1893: monthly SOI (light) + 3-month mean
           (bold), record monthly / 3-month max-min lines (value + date set).
  middle — the daily era: 30-day and 90-day running means with their records.
  bottom — the last 24 months: daily bars + the running means and records.

    python src/soi_history.py --out ../../assets/sst/soi_history.webp
"""
from __future__ import annotations
import argparse, io, re, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from soi_forecast import fetch_obs

MIN_COVER = 0.8                      # a window must be ≥80% present to count
BOM_URL = "ftp://ftp.bom.gov.au/anon/home/ncc/www/sco/soi/soiplaintext.html"


def running(series: pd.Series, days: int) -> pd.Series:
    """Trailing running mean on the daily calendar, NaN where coverage is poor."""
    s = series.asfreq("D")
    return s.rolling(days, min_periods=int(days * MIN_COVER)).mean()


def fetch_bom_monthly(cache: Path) -> pd.Series:
    """BoM monthly Troup SOI (1893–present) from the FTP plain-text table,
    cached with fallback like the daily file."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(BOM_URL, timeout=60) as r:
            txt = r.read().decode("utf-8", "replace")
        cache.write_text(txt)
    except Exception as e:                                # noqa: BLE001
        if not cache.exists():
            raise
        print(f"  BoM monthly fetch failed ({repr(e)[:60]}); using cached {cache.name}")
        txt = cache.read_text()
    vals = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*(\d{4})\s+(.+)$", ln)
        if not m or not (1800 < int(m.group(1)) < 2200):
            continue
        y = int(m.group(1))
        for i, tok in enumerate(m.group(2).split()[:12]):
            try:
                vals[pd.Timestamp(y, i + 1, 1)] = float(tok)
            except ValueError:
                continue
    s = pd.Series(vals).sort_index()
    if len(s) < 1000:
        raise RuntimeError(f"BoM monthly SOI parse looks wrong ({len(s)} values)")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../assets/sst/soi_history.webp")
    ap.add_argument("--cache", default="data/soi/DailySOI.txt")
    args = ap.parse_args()
    obs = fetch_obs(Path(args.cache))
    soi = obs["SOI"].dropna()
    mon = fetch_bom_monthly(Path(args.cache).parent / "soiplaintext.html")
    mon3 = mon.rolling(3, min_periods=3).mean()
    m30, m90 = running(soi, 30), running(soi, 90)
    recs = {}
    for name, s in (("30", m30), ("90", m90), ("mon", mon), ("mon3", mon3)):
        recs[name] = dict(hi=float(s.max()), hi_d=s.idxmax(),
                          lo=float(s.min()), lo_d=s.idxmin())
    print(f"  monthly record ({mon.index[0]:%Y}–): {recs['mon']['lo']:+.1f} "
          f"({recs['mon']['lo_d']:%b %Y}) … {recs['mon']['hi']:+.1f} ({recs['mon']['hi_d']:%b %Y})")
    print(f"  record 30-d: {recs['30']['lo']:+.1f} ({recs['30']['lo_d']:%b %Y}) … "
          f"{recs['30']['hi']:+.1f} ({recs['30']['hi_d']:%b %Y})")
    print(f"  record 90-d: {recs['90']['lo']:+.1f} ({recs['90']['lo_d']:%b %Y}) … "
          f"{recs['90']['hi']:+.1f} ({recs['90']['hi_d']:%b %Y})")

    fig, (aM, a0, a1) = plt.subplots(3, 1, figsize=(11.4, 12.4),
                                     gridspec_kw=dict(height_ratios=[1.05, 1, 0.95],
                                                      hspace=0.32))
    C30, C90 = "#64b5f6", "#0d47a1"
    CM, CM3 = "#ce93d8", "#6a1b9a"

    def rec_lines(ax, names):
        LBL = {"30": "30-d", "90": "90-d", "mon": "monthly", "mon3": "3-mo"}
        COL = {"30": C30, "90": C90, "mon": CM, "mon3": CM3}
        for name in names:
            for key, lab in (("hi", "max"), ("lo", "min")):
                v = recs[name][key]; d = recs[name][key + "_d"]
                ax.axhline(v, color=COL[name], lw=0.9, ls=":", alpha=0.85)
                ax.annotate(f"record {LBL[name]} {lab} {v:+.1f} ({d:%b %Y})",
                            xy=(1.0, v), xycoords=("axes fraction", "data"),
                            xytext=(-4, 2 if key == "hi" else -9),
                            textcoords="offset points", fontsize=6.8, color=COL[name],
                            ha="right", va="bottom" if key == "hi" else "top")

    # ── monthly record since 1893 ──
    aM.plot(mon.index, mon.values, color=CM, lw=0.55, alpha=0.8, label="monthly SOI")
    aM.plot(mon3.index, mon3.values, color=CM3, lw=1.3, label="3-month mean")
    aM.axhline(0, color="0.4", lw=0.8)
    aM.axhline(8, color="0.65", lw=0.8, ls="--"); aM.axhline(-8, color="0.65", lw=0.8, ls="--")
    rec_lines(aM, ("mon", "mon3"))
    aM.set_ylim(recs["mon"]["lo"] - 6, recs["mon"]["hi"] + 6)   # room for record labels
    aM.set_xlim(mon.index[0], mon.index[-1] + pd.Timedelta(days=700))
    aM.grid(True, alpha=0.2)
    aM.set_title(f"Monthly SOI (BoM Troup) — {mon.index[0]:%Y}–{mon.index[-1]:%b %Y}: "
                 f"latest {mon.iloc[-1]:+.1f}", fontsize=11.5, fontweight="bold", loc="left")
    aM.set_ylabel("SOI"); aM.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # ── daily record ──
    a0.plot(m30.index, m30.values, color=C30, lw=0.7, alpha=0.85, label="30-day mean")
    a0.plot(m90.index, m90.values, color=C90, lw=1.5, label="90-day mean")
    a0.axhline(0, color="0.4", lw=0.8)
    for y, lab in ((8, "sustained > +8 ⇒ La Niña"), (-8, "sustained < −8 ⇒ El Niño")):
        a0.axhline(y, color="0.65", lw=0.8, ls="--")
        a0.text(m30.index[0], y, " " + lab, fontsize=6.8, color="0.45",
                va="bottom" if y > 0 else "top")
    rec_lines(a0, ("30", "90"))
    a0.set_ylim(recs["30"]["lo"] - 6, recs["30"]["hi"] + 6)     # room for record labels
    a0.set_xlim(m30.index[0], m30.index[-1] + pd.Timedelta(days=120))
    a0.grid(True, alpha=0.2)
    a0.set_title(f"Daily SOI (LongPaddock Troup, 1887–1989 base) — the consistent daily record, "
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
    rec_lines(a1, ("30", "90"))
    a1.set_xlim(t0, soi.index[-1] + pd.Timedelta(days=10))
    a1.grid(True, alpha=0.2)
    cur30 = m30.dropna().iloc[-1]; cur90 = m90.dropna().iloc[-1]
    a1.set_title(f"Last 24 months — daily values (bars) · latest 30-d {cur30:+.1f}, "
                 f"90-d {cur90:+.1f}", fontsize=10.5, fontweight="bold", loc="left")
    a1.set_ylabel("SOI")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.05)
    fig.text(0.5, 0.005,
             "Monthly: BoM Troup SOI (ftp.bom.gov.au), one method 1876–present · daily: LongPaddock Troup SOI,\n"
             "fixed 1887–1989 base, Jun 1991–present (slightly different normalizations — records are per-series) · "
             "10·(ΔP−m)/σ, ΔP = Tahiti−Darwin MSLP · running means require ≥80% daily coverage",
             ha="center", fontsize=7.5, color="0.4")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  latest: 30-d {cur30:+.1f} · 90-d {cur90:+.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
