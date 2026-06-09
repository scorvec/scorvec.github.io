#!/usr/bin/env python3
"""Hourly SOI estimator from Darwin + Tahiti airport METARs.

The official Southern Oscillation Index is a Troup standardisation of the
Tahiti-minus-Darwin mean-sea-level-pressure difference, published only monthly (and
LongPaddock's daily version still lags 1-2 days). This estimates it in ~real time from
the hourly QNH reported in the Darwin (YPDN) and Tahiti / Faa'a (NTAA) METARs.

Per run it pulls the recent METARs, merges them into a rolling hourly pressure history
(committed CSV), forms the pressure difference, and applies the Troup monthly normals
(recovered from the LongPaddock daily file). Two curves are produced: the raw hourly SOI
(very noisy — the atmospheric tide + weather) and a 24-hour running mean (tide-removed,
the meaningful nowcast), the latter bias-corrected to the published LongPaddock daily SOI
so it sits on the same scale. Light enough for an hourly GitHub Action.

    python scripts/sst/hourly_soi.py --out assets/sst/soi_hourly.webp
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
CSV = HERE / "metar" / "soi_hourly.csv"               # rolling hourly pressure history (committed)
METAR_API = "https://aviationweather.gov/api/data/metar?ids=YPDN,NTAA&format=json&hours={h}"
SOI_URL = ("https://data.longpaddock.qld.gov.au/SeasonalClimateOutlook/"
           "SouthernOscillationIndex/SOIDataFiles/DailySOI1887-1989Base.txt")
STATIONS = {"YPDN": "darwin", "NTAA": "tahiti"}
PLOT_DAYS = 28


# ----------------------------------------------------------------------------- METAR
def fetch_metar(hours: int = 72) -> pd.DataFrame:
    """Recent METARs → hourly-mean QNH (hPa) per station, indexed by UTC hour."""
    import json
    url = METAR_API.format(h=hours)
    obs = json.loads(urllib.request.urlopen(url, timeout=60).read().decode())
    rows = []
    for o in obs:
        p = o.get("slp") or o.get("altim")             # slp if present (it isn't for these), else QNH
        if p is None:
            continue
        rows.append((pd.Timestamp(o["reportTime"]).tz_localize(None),
                     STATIONS.get(o.get("icaoId")), float(p)))
    df = pd.DataFrame(rows, columns=["time", "stn", "hpa"]).dropna()
    wide = df.pivot_table(index="time", columns="stn", values="hpa", aggfunc="mean")
    return wide.resample("1h").mean()                  # hourly means (averages the 30-min obs)


def update_history(new: pd.DataFrame) -> pd.DataFrame:
    """Merge the new hourly observations into the committed rolling CSV."""
    CSV.parent.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        old = pd.read_csv(CSV, parse_dates=["time"]).set_index("time")
        hist = new.combine_first(old).combine_first(new)   # new wins on overlap
        hist.update(new)
    else:
        hist = new
    hist = hist.sort_index()
    for c in ("darwin", "tahiti"):
        if c not in hist:
            hist[c] = np.nan
    hist[["darwin", "tahiti"]].to_csv(CSV)
    return hist


# ------------------------------------------------------------------- LongPaddock SOI
def longpaddock():
    """LongPaddock daily SOI (1887-1989 base): DataFrame[date]→SOI plus Troup normals."""
    txt = urllib.request.urlopen(SOI_URL, timeout=60).read().decode()
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    df["date"] = pd.to_datetime(df["Year"].astype(int).astype(str), format="%Y") + \
        pd.to_timedelta(df["Day"].astype(int) - 1, unit="D")
    df = df.set_index("date").sort_index()
    df["diff"] = df["Tahiti"] - df["Darwin"]
    # Troup normals: per calendar month, mean and SD of the (MSLP) pressure difference
    g = df.dropna(subset=["diff"])
    normals = {m: (sub["diff"].mean(), sub["diff"].std())
               for m, sub in g.groupby(g.index.month)}
    return df["SOI"], normals


def soi_of(diff_hpa, months, normals):
    """Troup SOI from a Tahiti−Darwin pressure difference (hPa) for each month."""
    mean = np.array([normals[m][0] for m in months])
    sd = np.array([normals[m][1] for m in months])
    return 10.0 * (diff_hpa - mean) / sd


# --------------------------------------------------------------------------- compute
def estimate(hist: pd.DataFrame, lp_soi: pd.Series, normals: dict):
    pdiff = (hist["tahiti"] - hist["darwin"]).dropna()
    raw = pd.Series(soi_of(pdiff.values, pdiff.index.month, normals), index=pdiff.index)
    pdiff_24 = pdiff.rolling(24, min_periods=18).mean()                 # tide-removed
    soi24 = pd.Series(soi_of(pdiff_24.values, pdiff_24.index.month, normals),
                      index=pdiff_24.index).dropna()
    # bias-correct the 24-h mean to the published LongPaddock daily SOI over the overlap
    lp = lp_soi.reindex(soi24.resample("1D").mean().index).dropna()
    ours_d = soi24.resample("1D").mean().reindex(lp.index)
    both = pd.concat([lp, ours_d], axis=1).dropna()
    bias = float((both.iloc[:, 0] - both.iloc[:, 1]).tail(30).median()) if len(both) else 0.0
    return raw + bias, soi24 + bias, bias


# ------------------------------------------------------------------------------ plot
def plot(raw, soi24, lp_soi, bias, out: Path):
    t0 = soi24.index.max() - pd.Timedelta(days=PLOT_DAYS)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    lp = lp_soi[lp_soi.index >= t0]
    ax.plot(lp.index, lp.values, color="#888", lw=1.4, marker="o", ms=3,
            label="LongPaddock daily SOI", zorder=2)
    r = raw[raw.index >= t0]
    ax.plot(r.index, r.values, color="#7aa6c2", lw=0.8, alpha=0.7,
            label="METAR hourly SOI (raw)", zorder=3)
    s = soi24[soi24.index >= t0]
    ax.plot(s.index, s.values, color="#d62728", lw=2.4, label="METAR 24-h mean SOI", zorder=4)
    last = s.dropna().iloc[-1]
    ax.scatter([s.dropna().index[-1]], [last], s=46, color="#d62728", zorder=6)
    ax.annotate(f"{last:+.0f}\n{s.dropna().index[-1]:%b %-d %HZ}", (s.dropna().index[-1], last),
                textcoords="offset points", xytext=(8, 0), va="center", fontsize=9,
                color="#d62728", fontweight="bold")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axhspan(-7, -40, color="#d62728", alpha=0.05); ax.axhspan(7, 40, color="#1f77b4", alpha=0.05)
    ax.set_ylabel("SOI (Troup)")
    ax.set_title("Hourly SOI estimate — Darwin (YPDN) + Tahiti (NTAA) METAR  ·  "
                 "negative ⇒ El Niño-favorable", fontsize=11)
    ax.set_xlim(t0, s.dropna().index[-1] + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9); ax.grid(alpha=0.15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out} (latest 24-h SOI {last:+.1f}; bias {bias:+.1f}; "
          f"{len(soi24)} hourly pts)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/soi_hourly.webp")
    ap.add_argument("--hours", type=int, default=72)
    args = ap.parse_args(argv)
    hist = update_history(fetch_metar(args.hours))
    lp_soi, normals = longpaddock()
    raw, soi24, bias = estimate(hist, lp_soi, normals)
    plot(raw, soi24, lp_soi, bias, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
