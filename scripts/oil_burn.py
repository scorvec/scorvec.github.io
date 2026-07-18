#!/usr/bin/env python3
"""Oil burned for US electricity, in thousand barrels per day (kb/d), by region.

EIA-930 daily petroleum ("OIL") net generation per grid region (EIA Open Data
API v2, daily-fuel-type-data, July 2018 → present), converted to implied crude
burn with the standard EIA equivalences:

    kb/d = MWh/day × (heat rate 10.33 MMBtu/MWh) / (5.80 MMBtu/bbl) / 1000
         ≈ MWh/day × 1.781 / 1000

Oil-fired generation is tiny nationally but concentrated and event-driven —
Florida (year-round peakers), New England and New York (winter gas scarcity),
Hawaii/PR excluded (not in US48 regions). The chart shows where and when the
fleet actually burns oil, and how it has evolved since 2018.

Figure (assets/power_data/oil_burn.webp):
  top    — stacked 30-day-mean kb/d by region since Jul 2018 (top burners
           explicit, the rest grouped), with notable spikes readable.
  bottom — last 24 months, daily: US48 total + the top regions as lines.

    EIA_API_KEY=… python scripts/oil_burn.py
(The key is also read from ~/.eia_key if the env var is unset.)

Later: extend to natural gas (same endpoint, fueltype NG, Bcf/d via
7.15 MMBtu/MWh gas fleet heat rate / 1.037 MMBtu/Mcf).
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "power_data" / "oil_burn.webp"

EIA_KEY = os.environ.get("EIA_API_KEY", "") or (
    Path.home().joinpath(".eia_key").read_text().strip()
    if Path.home().joinpath(".eia_key").exists() else "")

REGIONS = {  # EIA-930 region respondents (conterminous US)
    "CAL": "California", "CAR": "Carolinas", "CENT": "Central",
    "FLA": "Florida", "MIDA": "Mid-Atlantic", "MIDW": "Midwest",
    "NE": "New England", "NY": "New York", "NW": "Northwest",
    "SE": "Southeast", "SW": "Southwest", "TEN": "Tennessee", "TEX": "Texas",
}
START = "2018-07-01"                       # EIA-930 fuel-mix record begins
HEAT_RATE = 10.33                          # MMBtu/MWh, EIA avg petroleum fleet
MMBTU_BBL = 5.80                           # MMBtu per barrel (EIA convention)
BBL_PER_MWH = HEAT_RATE / MMBTU_BBL        # ≈ 1.781
EXPLICIT = ["FLA", "NE", "NY", "MIDA", "SE"]   # shown individually; rest grouped


def fetch_daily_oil() -> pd.DataFrame:
    """Daily OIL net generation (MWh) per region → DataFrame[date × region]."""
    url = "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "daily", "data[0]": "value",
            "facets[fueltype][]": "OIL", "facets[timezone][]": "Eastern",
            "start": START, "end": date.today().isoformat(),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        for i, r_id in enumerate(REGIONS):
            params[f"facets[respondent][{i}]"] = r_id
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        chunk = r.json()["response"]["data"]
        rows += chunk
        if len(chunk) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = (df.pivot_table(index="period", columns="respondent", values="value",
                           aggfunc="sum")
              .reindex(columns=list(REGIONS)).sort_index())
    # negatives are metering noise on a ~0 baseline; oil burn can't be negative
    return wide.clip(lower=0.0)


def main() -> int:
    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set (env or ~/.eia_key).", file=sys.stderr)
        return 1
    gen = fetch_daily_oil()                                    # MWh/day
    kbd = gen * BBL_PER_MWH / 1000.0                           # kb/d
    print(f"  {len(kbd)} days, {kbd.index[0]:%Y-%m-%d} → {kbd.index[-1]:%Y-%m-%d}")
    tot = kbd.sum(axis=1)
    print(f"  latest US48 total: {tot.dropna().iloc[-1]:.1f} kb/d · "
          f"record daily {tot.max():.1f} kb/d on {tot.idxmax():%Y-%m-%d}")

    sm = kbd.rolling(30, min_periods=20).mean()
    order = kbd.mean().sort_values(ascending=False)
    explicit = [r for r in order.index if r in EXPLICIT]
    other = [r for r in REGIONS if r not in explicit]
    stack = sm[explicit].copy()
    stack["Other regions"] = sm[other].sum(axis=1)
    labels = [REGIONS[r] for r in explicit] + ["Other regions"]
    colors = ["#d95f02", "#1f78b4", "#33a02c", "#6a3d9a", "#e31a1c", "#b0b0b0"]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.6, 8.8),
                                 gridspec_kw=dict(height_ratios=[1.15, 1],
                                                  hspace=0.3))
    a0.stackplot(stack.index, [stack[c].fillna(0).values for c in stack.columns],
                 labels=labels, colors=colors, alpha=0.9, lw=0)
    a0.set_xlim(stack.index[0], stack.index[-1])
    a0.grid(True, alpha=0.2)
    a0.set_ylabel("thousand barrels / day")
    a0.set_title("Oil burned for US electricity — 30-day mean by grid region, "
                 f"Jul 2018 – {kbd.index[-1]:%b %Y}",
                 fontsize=12, fontweight="bold", loc="left")
    a0.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)

    t0 = kbd.index[-1] - pd.DateOffset(months=24)
    a1.plot(tot[tot.index >= t0].index, tot[tot.index >= t0].values,
            color="#222", lw=1.0, alpha=0.55, label="US48 total (daily)")
    a1.plot(sm[sm.index >= t0].index, sm.loc[sm.index >= t0, explicit].sum(axis=1)
            + sm.loc[sm.index >= t0, other].sum(axis=1),
            color="#222", lw=2.0, label="US48 total (30-d mean)")
    for r, c in zip(explicit[:3], colors):
        d = kbd.loc[kbd.index >= t0, r]
        a1.plot(d.index, d.values, color=c, lw=0.9, alpha=0.8, label=REGIONS[r])
    a1.set_xlim(t0, kbd.index[-1])
    a1.grid(True, alpha=0.2)
    a1.set_ylabel("thousand barrels / day")
    a1.set_title("Last 24 months — daily", fontsize=10.5, fontweight="bold",
                 loc="left")
    a1.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.06)
    fig.text(0.5, 0.005,
             "EIA-930 daily net generation from petroleum by region (Eastern time) · "
             f"kb/d = MWh × {HEAT_RATE} MMBtu/MWh ÷ {MMBTU_BBL} MMBtu/bbl ≈ MWh × "
             f"{BBL_PER_MWH:.2f}/1000 · conterminous US only",
             ha="center", fontsize=7.5, color="0.4")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
