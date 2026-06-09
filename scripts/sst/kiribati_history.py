#!/usr/bin/env python3
"""Tarawa westerly-wind-burst activity across recent El Niño onsets (IEM METAR archive).

Compares the equatorial zonal wind at Tarawa (Kiribati) through the onset years of the
recent strong-ish El Niños — 2009, 2015, 2023 — against the current year, to see how the
trade weakening / westerly-wind-burst activity built up each time. Data: the Iowa
Environmental Mesonet ASOS archive (station NGTT/NGTA, back to 2003), pulled gently
(one request per year, with backoff). The fixed analog years are cached so refreshes only
re-fetch the current year.

    python scripts/sst/kiribati_history.py --out assets/sst/kiribati_history.webp
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CACHE = HERE / "metar" / "kiribati_history_monthly.csv"
ANALOGS = {2009: "#e41a1c", 2015: "#4daf4a", 2023: "#984ea3"}     # onset years (+ current)
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
       "station=NGTT&station=NGTA&data=drct&data=sknt&"
       "year1={y}&month1=1&day1=1&year2={y}&month2=12&day2=31&"
       "tz=Etc/UTC&format=onlycomma&missing=M")
KT2MS = 0.514444


def fetch_year(y: int) -> pd.DataFrame | None:
    """One IEM request for the whole year (both Tarawa ids), with rate-limit backoff."""
    for attempt in range(5):
        try:
            txt = urllib.request.urlopen(IEM.format(y=y), timeout=180).read().decode()
        except Exception as e:
            print(f"    {y}: fetch error {repr(e)[:60]}; retry", flush=True); time.sleep(15); continue
        if "Too many requests" in txt:
            print(f"    {y}: rate-limited, backing off", flush=True); time.sleep(25); continue
        df = pd.read_csv(io.StringIO(txt))
        if "drct" not in df:
            return None
        df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
        df["drct"] = pd.to_numeric(df["drct"], errors="coerce")
        df["sknt"] = pd.to_numeric(df["sknt"], errors="coerce")
        return df.dropna(subset=["valid", "drct", "sknt"])
    return None


def monthly_stats(df: pd.DataFrame, y: int) -> pd.DataFrame:
    u = -df["sknt"] * KT2MS * np.sin(np.deg2rad(df["drct"]))      # eastward; westerly +
    g = pd.DataFrame({"u": u.values}, index=df["valid"].values)
    um = g["u"].resample("MS").mean()
    wf = (g["u"] > 0).resample("MS").mean() * 100.0              # % westerly obs
    n = g["u"].resample("MS").count()
    out = pd.DataFrame({"u_mean": um, "west_frac": wf, "n": n}).dropna()
    out = out[out["n"] >= 50]                                     # need decent monthly sampling
    out["year"] = y; out["month"] = out.index.month
    return out[["year", "month", "u_mean", "west_frac", "n"]].reset_index(drop=True)


def build_table(out_years) -> pd.DataFrame:
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(
        columns=["year", "month", "u_mean", "west_frac", "n"])
    cur = datetime.now(timezone.utc).year
    keep = cache.copy()
    for i, y in enumerate(out_years):
        cached = (keep.year == y).any()
        if cached and y != cur:
            print(f"  {y}: cached", flush=True); continue
        if i:
            time.sleep(6)                                        # be gentle to IEM
        df = fetch_year(y)
        if df is None or df.empty:
            print(f"  {y}: no data", flush=True); continue
        keep = keep[keep.year != y]
        keep = pd.concat([keep, monthly_stats(df, y)], ignore_index=True)
        print(f"  {y}: {len(df):,} obs", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    keep.sort_values(["year", "month"]).to_csv(CACHE, index=False)
    return keep


def plot(tab: pd.DataFrame, out: Path):
    cur = datetime.now(timezone.utc).year
    years = sorted(set(ANALOGS) | {cur})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    for y in years:
        sub = tab[tab.year == y].sort_values("month")
        if sub.empty:
            continue
        c = "#111" if y == cur else ANALOGS.get(y, "#999")
        lw = 2.8 if y == cur else 1.8
        lab = f"{y} (now)" if y == cur else f"{y}–{str(y + 1)[2:]}"
        axes[0].plot(sub.month, sub.u_mean, color=c, lw=lw, marker="o", ms=3, label=lab)
        axes[1].plot(sub.month, sub.west_frac, color=c, lw=lw, marker="o", ms=3, label=lab)
    axes[0].axhline(0, color="0.6", lw=0.7)
    axes[0].set_ylabel("monthly-mean zonal wind\n(m s⁻¹) · westerly +")
    axes[0].set_title("Tarawa westerly-wind-burst activity across El Niño onsets — "
                      "monthly METAR wind (IEM)", fontsize=11)
    axes[1].set_ylabel("westerly obs\n(% of the month)")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(list("JFMAMJJASOND"))
    axes[1].set_xlabel("month of onset year")
    for ax in axes:
        ax.grid(alpha=0.15); ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/kiribati_history.webp")
    args = ap.parse_args(argv)
    years = sorted(set(ANALOGS) | {datetime.now(timezone.utc).year})
    tab = build_table(years)
    plot(tab, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
