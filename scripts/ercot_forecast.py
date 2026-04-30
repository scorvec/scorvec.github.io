"""
ERCOT 48-Hour Load Forecast — LightGBM + Open-Meteo
-----------------------------------------------------
Uses Open-Meteo API (free, no key, HRRR-based) to fetch point forecasts
at nine Texas stations, then runs the trained LightGBM model.

Replaces Herbie/cfgrib/GRIB approach — no native library dependencies,
no memory issues, much faster.

Reads:  assets/lgbm_model.pkl
        assets/model_meta.json
Writes: assets/ercot_forecast.png

Usage:
  python scripts/ercot_forecast.py
  python scripts/ercot_forecast.py --run 12   # ignored, kept for CLI compat
"""

import os
import json
import argparse
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone

# ── CONFIG ───────────────────────────────────────────────────────────────
STATIONS = {
    "KIAH": {"lat": 29.984, "lon": -95.368, "weight": 0.28},
    "KHOU": {"lat": 29.645, "lon": -95.279, "weight": 0.08},
    "KDFW": {"lat": 32.897, "lon": -97.044, "weight": 0.22},
    "KSAT": {"lat": 29.534, "lon": -98.470, "weight": 0.14},
    "KAUS": {"lat": 30.194, "lon": -97.670, "weight": 0.12},
    "KELP": {"lat": 31.807, "lon": -106.376,"weight": 0.05},
    "KCRP": {"lat": 27.770, "lon": -97.511, "weight": 0.03},
    "KAMA": {"lat": 35.219, "lon": -101.706,"weight": 0.03},
    "KBRO": {"lat": 25.906, "lon": -97.432, "weight": 0.05},
}

# Open-Meteo variables (correct names per docs)
OM_VARIABLES = "temperature_2m,dew_point_2m,wind_speed_10m,cloud_cover"

MODEL_PATH = "assets/lgbm_model.pkl"
META_PATH  = "assets/model_meta.json"
OUT_PATH   = "assets/ercot_forecast.png"
BG, PANEL, LIGHT, MUTED = "#0f0f0d", "#181816", "#e8e6e0", "#5a5855"


# ── 1. FETCH OPEN-METEO FORECASTS ─────────────────────────────────────────
def fetch_open_meteo(station_name, lat, lon):
    """
    Fetch HRRR-driven point forecast from Open-Meteo's /v1/gfs endpoint,
    which automatically uses HRRR (3 km, hourly updates) for US locations.
    Returns DataFrame indexed by UTC time with columns:
      temp_f, dwpt_f, wind_mph, cloud_pct
    """
    url = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           OM_VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "forecast_days":    2,
        "timezone":         "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]

    df = pd.DataFrame({
        "time":      pd.to_datetime(data["time"], utc=True),
        "temp_f":    data["temperature_2m"],
        "dwpt_f":    data["dew_point_2m"],
        "wind_mph":  data["wind_speed_10m"],
        "cloud_pct": data["cloud_cover"],
    }).set_index("time")

    return df


def fetch_all_stations():
    """
    Fetch forecasts for all stations and return population-weighted means.
    """
    print("Fetching Open-Meteo HRRR forecasts...")
    weighted = None
    total_w  = 0.0

    for name, info in STATIONS.items():
        try:
            df  = fetch_open_meteo(name, info["lat"], info["lon"])
            w   = info["weight"]
            wdf = df * w
            weighted = wdf if weighted is None else weighted.add(wdf, fill_value=0)
            total_w += w
            print(f"  {name}: OK ({len(df)} hours)")
        except Exception as e:
            print(f"  {name}: SKIPPED — {e}")

    if weighted is None:
        raise SystemExit("ERROR: All stations failed — cannot build forecast.")
    result = weighted / total_w
    print(f"  Done — {len(result)} weighted hourly values.")
    return result


# ── 2. BUILD FEATURES ─────────────────────────────────────────────────────
def build_features(obs_df, training_start, features):
    origin = pd.Timestamp(training_start, tz="UTC")
    df = pd.DataFrame(index=obs_df.index)
    df["hour_ct"]          = (df.index.hour - 5) % 24
    df["dow"]              = df.index.dayofweek
    df["month"]            = df.index.month
    df["temp_f"]           = obs_df["temp_f"]
    df["dwpt_f"]           = obs_df["dwpt_f"]
    df["wind_mph"]         = obs_df["wind_mph"]
    df["cloud_pct"]        = obs_df["cloud_pct"]
    df["days_since_start"] = (df.index - origin).days
    df["is_weekend"]       = (df["dow"] >= 5).astype(int)
    return df[features]


# ── 3. PLOT ───────────────────────────────────────────────────────────────
def make_forecast_plot(load_fcst, run_dt, meta):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

    # Convert index to Central Time for display, then strip tz so matplotlib
    # doesn't try to convert back to UTC
    load_fcst_ct = load_fcst.copy()
    load_fcst_ct.index = load_fcst_ct.index.tz_convert("America/Chicago").tz_localize(None)

    cmap     = cm.twilight_shifted
    norm     = mcolors.Normalize(vmin=0, vmax=23)
    ct_hours = load_fcst_ct.index.hour
    times    = load_fcst_ct.index
    values   = load_fcst_ct.values

    for i in range(len(times) - 1):
        if np.isnan(values[i]) or np.isnan(values[i+1]):
            continue
        ax.plot([times[i], times[i+1]], [values[i], values[i+1]],
                color=cmap(norm(ct_hours[i])), linewidth=2.5,
                solid_capstyle="round")

    # "Now" line in Central Time, also tz-naive
    now_ct = pd.Timestamp.now(tz="America/Chicago").tz_localize(None)
    ax.axvline(now_ct, color=LIGHT, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(now_ct, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 75,
            "  Now", color=LIGHT, fontsize=7.5, va="top", alpha=0.55)

    # Day shading also tz-naive
    day_start = pd.Timestamp(times[0].date())
    for d in range(4):
        ds = day_start + pd.Timedelta(days=d)
        de = ds + pd.Timedelta(days=1)
        if d % 2 == 0:
            ax.axvspan(ds, de, color="#1a1a18", alpha=0.35, zorder=0)

    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Hour of Day (CT)", color=LIGHT, fontsize=9, labelpad=10)
    cbar.set_ticks([0, 6, 12, 18, 23])
    cbar.set_ticklabels(["Midnight","6 AM","Noon","6 PM","11 PM"])
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(MUTED)

    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a28")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Date / Time (Central)", color=LIGHT, fontsize=10, labelpad=8)
    ax.set_ylabel("ERCOT Load Forecast (GW)", color=LIGHT, fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d\n%-I %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    valid = values[~np.isnan(values)]
    if len(valid):
        ax.set_ylim(max(20, valid.min() - 3), min(90, valid.max() + 3))

    run_dt_ct = run_dt.astimezone(pd.Timestamp.now("America/Chicago").tz)
    run_str   = run_dt_ct.strftime("%Y-%m-%d %-I %p %Z")
    test_r2   = meta.get("test_r2", "—")
    ax.set_title(
        f"ERCOT 48-Hour Load Forecast  ·  Generated {run_str}  ·  "
        f"Model Test R² = {test_r2}",
        color=LIGHT, fontsize=12, fontweight="normal", loc="left", pad=14)
    ax.text(0.99, 1.012,
            pd.Timestamp.now("America/Chicago").strftime("Updated %B %d, %Y · %-I:%M %p %Z"),
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.12,
            "LightGBM · Open-Meteo HRRR · 2m temp + dewpoint · 10m wind · cloud cover",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Forecast saved → {OUT_PATH}")


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=None,
                        help="HRRR run hour (ignored — Open-Meteo always serves latest)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"ERROR: {MODEL_PATH} not found. Run ercot_plot.py first.")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Model loaded — Test R² = {meta['test_r2']}  "
          f"(trained on {meta['n_train']:,} hours)")

    # Fetch forecasts
    obs_df = fetch_all_stations()

    # Trim to 48 hours from now
    now    = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_dt = now + timedelta(hours=48)
    obs_df = obs_df[(obs_df.index >= now) & (obs_df.index <= end_dt)]

    # Build features and run model
    X = build_features(obs_df, meta["training_start"], meta["features"])
    load_pred = np.clip(model.predict(X), 20, 90)
    load_fcst = pd.Series(load_pred, index=obs_df.index, name="load_gw_fcst")

    valid = load_fcst.dropna()
    print(f"Forecast: {valid.min():.1f} – {valid.max():.1f} GW  ({len(valid)} hours)")

    run_dt = now
    make_forecast_plot(load_fcst, run_dt, meta)
    print("Done.")
