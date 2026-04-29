"""
ERCOT Hourly Load vs Population-Weighted Temperature
-----------------------------------------------------
Pulls 365 days of:
  - ERCOT system demand  →  EIA Open Data API (free key)
  - Hourly ASOS obs      →  Iowa State Mesonet (no key needed)

Outputs: assets/ercot_load_temp.png
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import StringIO
from datetime import datetime, timedelta, timezone

# ── STATIONS & POPULATION WEIGHTS ───────────────────────────────────────
# Weights are approximate % of ERCOT-zone population each city represents
STATIONS = {
    "KIAH":  0.30,   # Houston
    "KDFW":  0.22,   # Dallas
    "KSAT":  0.14,   # San Antonio
    "KAUS":  0.12,   # Austin
    "KFTW":  0.08,   # Fort Worth
    "KHOU":  0.06,   # Houston (secondary, averages with KIAH)
    "KELP":  0.04,   # El Paso
    "KCRP":  0.02,   # Corpus Christi
    "KAMA":  0.02,   # Amarillo
}

EIA_KEY = os.environ["EIA_API_KEY"]

# ── DATE RANGE (last 365 days, UTC) ─────────────────────────────────────
end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=365)


# ── 1. FETCH EIA ERCOT LOAD ─────────────────────────────────────────────
def fetch_ercot_load(start, end, api_key):
    """Download hourly ERCOT system demand from EIA API v2."""
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    records = []
    offset  = 0
    page    = 5000

    while True:
        params = {
            "api_key":           api_key,
            "frequency":         "hourly",
            "data[0]":           "value",
            "facets[type][]":    "D",          # D = demand
            "facets[respondent][]": "ERCO",    # ERCOT balancing authority
            "start":  start.strftime("%Y-%m-%dT%H"),
            "end":    end.strftime("%Y-%m-%dT%H"),
            "length": page,
            "offset": offset,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        records.extend(batch)
        print(f"  EIA: fetched {len(records)} records so far...")
        if len(batch) < page:
            break
        offset += page

    df = pd.DataFrame(records)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["period", "load_mw"]].dropna().sort_values("period")
    df = df.drop_duplicates("period").set_index("period")
    print(f"  EIA: {len(df)} hourly load records loaded.")
    return df


# ── 2. FETCH ASOS TEMPERATURE ────────────────────────────────────────────
def fetch_asos(station, start, end):
    """Download hourly ASOS obs from Iowa State Mesonet (tmpf in °F)."""
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       start.year,  "month1": start.month,  "day1": start.day,
        "year2":       end.year,    "month2": end.month,    "day2": end.day,
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "direct":      "no",
        "report_type": 3,           # 3 = hourly METAR
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    # Iowa State prepends a few comment lines before the CSV header
    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(lines)), na_values=["M", "T", " "])

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"valid": "time", "tmpf": "temp_f"})
    df["time"]   = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    df["temp_f"] = pd.to_numeric(df["temp_f"], errors="coerce")
    df = df[["time", "temp_f"]].dropna()
    df = df.groupby("time")["temp_f"].mean()   # average within each hour
    print(f"  {station}: {len(df)} hourly obs loaded.")
    return df


# ── 3. BUILD POPULATION-WEIGHTED TEMPERATURE ────────────────────────────
def weighted_temperature(stations, start, end):
    """Return an hourly Series of population-weighted mean temperature (°F)."""
    weighted_sum = None
    total_weight = 0.0

    for station, weight in stations.items():
        try:
            series = fetch_asos(station, start, end) * weight
            weighted_sum = series if weighted_sum is None else weighted_sum.add(series, fill_value=0)
            total_weight += weight
        except Exception as e:
            print(f"  WARNING: could not fetch {station}: {e}")
        time.sleep(3)   # be polite to Iowa State's servers

    # Normalise in case any stations failed
    weighted_temp = weighted_sum / total_weight
    weighted_temp.name = "temp_f"
    return weighted_temp


# ── 4. PLOT ──────────────────────────────────────────────────────────────
def make_plot(df, out_path):
    # Convert °F → °F (keep), compute hour in Central Time (UTC-6 approx)
    df = df.copy()
    df["temp_f"]  = df["temp_f"]
    df["hour_ct"] = (df.index.hour - 6) % 24     # rough CST offset

    # Drop obvious outliers (load < 20 GW or temp outside 0–115°F)
    df = df[(df["load_gw"] > 20) & (df["load_gw"] < 90)]
    df = df[(df["temp_f"]  > 0)  & (df["temp_f"]  < 115)]

    # ── Style ──
    BG     = "#0f0f0d"
    PANEL  = "#181816"
    LIGHT  = "#e8e6e0"
    MUTED  = "#666460"

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Circular hour colormap (twilight_shifted wraps midnight nicely)
    cmap   = cm.twilight_shifted
    norm   = mcolors.Normalize(vmin=0, vmax=23)
    colors = cmap(norm(df["hour_ct"].values))

    sc = ax.scatter(
        df["temp_f"], df["load_gw"],
        c=df["hour_ct"], cmap=cmap, norm=norm,
        s=4, alpha=0.55, linewidths=0, rasterized=True,
    )

    # Colourbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Hour of Day (CT)", color=LIGHT, fontsize=9, labelpad=10)
    cbar.set_ticks([0, 6, 12, 18, 23])
    cbar.set_ticklabels(["Midnight", "6 AM", "Noon", "6 PM", "11 PM"])
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(MUTED)

    # Axes styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#333330")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)
    ax.set_xlabel("Population-Weighted Temperature (°F)", fontsize=10, labelpad=8)
    ax.set_ylabel("ERCOT System Load (GW)", fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)

    # Title block
    date_str = datetime.now(timezone.utc).strftime("Updated %B %d, %Y")
    ax.set_title(
        "ERCOT Hourly Load vs. Temperature — Last 365 Days",
        color=LIGHT, fontsize=13, fontweight="normal",
        loc="left", pad=14,
    )
    ax.text(
        0.99, 1.012, date_str,
        transform=ax.transAxes, ha="right", va="bottom",
        color=MUTED, fontsize=8,
    )
    ax.text(
        0.99, -0.09,
        "Sources: EIA Open Data (ERCOT demand) · Iowa State Mesonet (ASOS obs)",
        transform=ax.transAxes, ha="right", va="top",
        color=MUTED, fontsize=7.5,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching ERCOT load...")
    load = fetch_ercot_load(start_dt, end_dt, EIA_KEY)

    print("Fetching ASOS temperatures...")
    temp = weighted_temperature(STATIONS, start_dt, end_dt)

    print("Merging and plotting...")
    df = pd.DataFrame({"load_mw": load["load_mw"], "temp_f": temp})
    df = df.dropna()
    df["load_gw"] = df["load_mw"] / 1000.0

    make_plot(df, "assets/ercot_load_temp.png")
    print("Done.")
