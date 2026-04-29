"""
ERCOT Hourly Load vs Population-Weighted Temperature
-----------------------------------------------------
Pulls 365 days of:
  - ERCOT system demand  →  EIA Open Data API (free key)
  - Hourly ASOS obs      →  Iowa State Mesonet (no key needed)

Outputs: assets/ercot_load_temp.png

Usage:
  export EIA_API_KEY="your_key_here"
  python scripts/ercot_plot.py
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import StringIO
from datetime import datetime, timedelta, timezone

# ── CONFIG ───────────────────────────────────────────────────────────────
STATIONS = {
    "KIAH": 0.28,   # Houston (Intercontinental)
    "KHOU": 0.08,   # Houston (Hobby)
    "KDFW": 0.22,   # Dallas/Fort Worth
    "KSAT": 0.14,   # San Antonio
    "KAUS": 0.12,   # Austin
    "KELP": 0.05,   # El Paso
    "KCRP": 0.03,   # Corpus Christi
    "KAMA": 0.03,   # Amarillo
    "KBRO": 0.05,   # Brownsville
}

EIA_KEY  = os.environ.get("EIA_API_KEY", "")
OUT_PATH = "assets/ercot_load_temp.png"

end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=365)


# ── 1. EIA ERCOT LOAD ────────────────────────────────────────────────────
def fetch_ercot_load():
    print("Fetching ERCOT load from EIA...")
    url     = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    records = []
    offset  = 0

    while True:
        params = {
            "api_key":              EIA_KEY,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[type][]":       "D",
            "facets[respondent][]": "ERCO",
            "start":   start_dt.strftime("%Y-%m-%dT%H"),
            "end":     end_dt.strftime("%Y-%m-%dT%H"),
            "length":  5000,
            "offset":  offset,
            "sort[0][column]":    "period",
            "sort[0][direction]": "asc",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        records.extend(batch)
        print(f"  {len(records)} records fetched...")
        if len(batch) < 5000:
            break
        offset += 5000

    df = pd.DataFrame(records)
    df["period"]  = pd.to_datetime(df["period"], utc=True)
    df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df = (df[["period", "load_mw"]]
            .dropna()
            .drop_duplicates("period")
            .set_index("period")
            .sort_index())
    print(f"  Done — {len(df)} hourly records.")
    return df


# ── 2. ASOS TEMPERATURE ──────────────────────────────────────────────────
def fetch_asos(station):
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       start_dt.year,  "month1": start_dt.month,  "day1": start_dt.day,
        "year2":       end_dt.year,    "month2": end_dt.month,    "day2": end_dt.day,
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "direct":      "no",
        "report_type": 3,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(lines)), na_values=["M", "T", " "])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"valid": "time", "tmpf": "temp_f"})
    df["time"]   = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    df["temp_f"] = pd.to_numeric(df["temp_f"], errors="coerce")
    df = df[["time", "temp_f"]].dropna().groupby("time")["temp_f"].mean()
    return df


def fetch_weighted_temp():
    print("Fetching ASOS temperatures...")
    weighted = None
    total_w  = 0.0

    for station, weight in STATIONS.items():
        try:
            s = fetch_asos(station) * weight
            weighted = s if weighted is None else weighted.add(s, fill_value=0)
            total_w += weight
            print(f"  {station}: OK ({len(s)} obs)")
        except Exception as e:
            print(f"  {station}: SKIPPED — {e}")
        time.sleep(4)

    temp = weighted / total_w
    temp.name = "temp_f"
    print(f"  Done — {len(temp)} weighted hourly values.")
    return temp


# ── 3. PLOT ──────────────────────────────────────────────────────────────
def make_plot(df):
    df = df.copy()
    df["hour"] = (df.index.hour - 6) % 24   # rough Central time

    # Remove obvious bad data
    df = df[(df["load_gw"] > 20) & (df["load_gw"] < 90)]
    df = df[(df["temp_f"]  >  0) & (df["temp_f"]  < 115)]

    BG, PANEL, LIGHT, MUTED = "#0f0f0d", "#181816", "#e8e6e0", "#5a5855"

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    cmap = cm.twilight_shifted
    norm = mcolors.Normalize(vmin=0, vmax=23)

    ax.scatter(
        df["temp_f"], df["load_gw"],
        c=df["hour"], cmap=cmap, norm=norm,
        s=4, alpha=0.55, linewidths=0, rasterized=True,
    )

    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Hour of Day (CT)", color=LIGHT, fontsize=9, labelpad=10)
    cbar.set_ticks([0, 6, 12, 18, 23])
    cbar.set_ticklabels(["Midnight", "6 AM", "Noon", "6 PM", "11 PM"])
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(MUTED)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a28")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Population-Weighted Temperature (°F)", color=LIGHT, fontsize=10, labelpad=8)
    ax.set_ylabel("ERCOT System Load (GW)",               color=LIGHT, fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)

    date_str = datetime.now(timezone.utc).strftime("Updated %B %d, %Y")
    ax.set_title(
        "ERCOT Hourly Load vs. Temperature  ·  Last 365 Days",
        color=LIGHT, fontsize=13, fontweight="normal", loc="left", pad=14,
    )
    ax.text(0.99, 1.012, date_str,
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.09,
            "Sources: EIA Open Data (ERCOT demand)  ·  Iowa State Mesonet (ASOS obs)",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Plot saved → {OUT_PATH}")


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not EIA_KEY:
        raise SystemExit("ERROR: EIA_API_KEY environment variable not set.")

    load = fetch_ercot_load()
    temp = fetch_weighted_temp()

    df = pd.DataFrame({"load_mw": load["load_mw"], "temp_f": temp}).dropna()
    df["load_gw"] = df["load_mw"] / 1000.0

    print(f"\nMerged dataset: {len(df)} hourly points.")
    make_plot(df)
    print("Done.")
