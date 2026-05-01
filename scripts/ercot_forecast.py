"""
ERCOT 48-Hour Load Forecast — LightGBM + Open-Meteo
-----------------------------------------------------
Uses Open-Meteo API (free, no key, HRRR-based) to fetch point forecasts
at nine Texas stations, then runs the trained LightGBM model.

Reads:  assets/lgbm_model.pkl
        assets/model_meta.json
Writes: assets/ercot_forecast.png   (static fallback)
        assets/ercot_forecast.html  (interactive Plotly chart)

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

EIA_KEY    = os.environ.get("EIA_API_KEY", "")
MODEL_PATH = "assets/lgbm_model.pkl"
META_PATH  = "assets/model_meta.json"
OUT_PATH   = "assets/ercot_forecast.png"
HTML_PATH  = "assets/ercot_forecast.html"
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


# ── 4. ERCOT 5-MINUTE ACTUALS (via gridstatus) ────────────────────────────
def fetch_recent_actuals(hours_back=24):
    """
    Fetch recent ERCOT actual load directly from ERCOT at 5-minute resolution
    using the gridstatus library. Returns Series in GW.
    """
    try:
        import gridstatus
    except ImportError:
        print("  gridstatus not installed — skipping actuals.")
        return None

    print(f"Fetching last {hours_back}h of ERCOT 5-min actuals...")
    try:
        end   = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(hours=hours_back)
        ercot = gridstatus.Ercot()
        df = ercot.get_load(
            start=start.strftime("%Y-%m-%d %H:%M"),
            end=end.strftime("%Y-%m-%d %H:%M"),
        )

        # gridstatus returns "Time" column in US/Central, "Load" in MW
        df = df.rename(columns={"Time": "time", "Load": "load_mw"})
        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("America/Chicago")
        df["time"] = df["time"].dt.tz_convert("UTC")
        df["load_gw"] = df["load_mw"] / 1000.0
        df = df[["time", "load_gw"]].dropna().drop_duplicates("time").set_index("time").sort_index()

        print(f"  Got {len(df)} 5-minute actual records.")
        return df["load_gw"]
    except Exception as e:
        print(f"  Failed to fetch ERCOT actuals: {e}")
        return None


# ── 5. INTERACTIVE PLOTLY CHART ──────────────────────────────────────────
def make_interactive_plot(load_fcst, run_dt, meta, actuals=None):
    """
    Generate a self-contained interactive HTML chart with Plotly.
    Hover tooltip shows full timestamp, day of week, and load.
    Optionally overlays recent ERCOT actuals from EIA.
    """
    import plotly.graph_objects as go

    # Convert forecast to Central Time, drop tz so JS doesn't apply browser tz
    df = pd.DataFrame({"load_gw": load_fcst.values}, index=load_fcst.index)
    df.index = df.index.tz_convert("America/Chicago").tz_localize(None)
    df["hour"]  = df.index.hour
    df["label"] = df.index.strftime("%a %b %-d · %-I %p")

    # Hover text per forecast point
    hover_fcst = [
        f"<b>{lbl}</b><br>Forecast: {ld:.2f} GW"
        for lbl, ld in zip(df["label"], df["load_gw"])
    ]

    # Twilight-shifted-ish gradient mapped to hour of day
    twilight_hex = [
        "#1a1a2e", "#1f2240", "#27314f", "#314058", "#3b4f5e", "#465e62",
        "#536c64", "#637765", "#778066", "#8c8869", "#a18e6f", "#b6917a",
        "#c79289", "#d39198", "#dc8da6", "#e088b3", "#df85bf", "#d683c9",
        "#c283ce", "#a784ce", "#8585c9", "#6685bf", "#4f80ad", "#3a7095",
    ]
    point_colors = [twilight_hex[h] for h in df["hour"]]

    fig = go.Figure()

    # ── Actuals trace (if available) ──────────────────────────────────────
    if actuals is not None and len(actuals) > 0:
        ac = actuals.copy()
        ac.index = ac.index.tz_convert("America/Chicago").tz_localize(None)
        ac_labels = ac.index.strftime("%a %b %-d · %-I:%M %p")
        hover_ac = [
            f"<b>{lbl}</b><br>Actual: {ld:.2f} GW"
            for lbl, ld in zip(ac_labels, ac.values)
        ]
        fig.add_trace(go.Scatter(
            x=ac.index, y=ac.values,
            mode="lines",
            line=dict(color="#a8a59f", width=1.8),
            hovertext=hover_ac, hoverinfo="text",
            name="Actual (5-min)",
            opacity=0.85,
        ))

    # ── Forecast trace ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["load_gw"],
        mode="lines+markers",
        line=dict(color="#7e9fc7", width=3, shape="spline", smoothing=1.3),
        marker=dict(size=7, color=point_colors, line=dict(width=0)),
        hovertext=hover_fcst, hoverinfo="text",
        name="Forecast",
    ))

    # ── "Now" reference line ──────────────────────────────────────────────
    now_ct = pd.Timestamp.now(tz="America/Chicago").tz_localize(None)
    now_str = now_ct.strftime("%Y-%m-%d %H:%M:%S")
    fig.add_shape(
        type="line",
        x0=now_str, x1=now_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="#e8e6e0", width=1, dash="dot"),
        opacity=0.5,
    )
    fig.add_annotation(
        x=now_str, y=1, yref="paper",
        text="Now", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color="#e8e6e0", size=11),
        xshift=4,
    )

    # ── Axis ranges ───────────────────────────────────────────────────────
    # X: tightly bounded to actual data range
    all_times = list(df.index)
    if actuals is not None and len(actuals) > 0:
        all_times = list(ac.index) + all_times
    x_min = min(all_times).strftime("%Y-%m-%d %H:%M:%S")
    x_max = max(all_times).strftime("%Y-%m-%d %H:%M:%S")

    # Y: data range with 15% padding above and below
    all_loads = list(df["load_gw"].values)
    if actuals is not None and len(actuals) > 0:
        all_loads += list(ac.values)
    y_lo, y_hi = min(all_loads), max(all_loads)
    y_pad = (y_hi - y_lo) * 0.15
    y_min = max(0, y_lo - y_pad)
    y_max = y_hi + y_pad

    # ── Day shading ───────────────────────────────────────────────────────
    day_start = pd.Timestamp(min(all_times).date())
    end_date  = pd.Timestamp(max(all_times).date()) + pd.Timedelta(days=1)
    n_days    = (end_date - day_start).days + 1
    for d in range(n_days):
        ds = day_start + pd.Timedelta(days=d)
        de = ds + pd.Timedelta(days=1)
        if d % 2 == 0:
            fig.add_shape(
                type="rect",
                x0=ds.strftime("%Y-%m-%d %H:%M:%S"),
                x1=de.strftime("%Y-%m-%d %H:%M:%S"),
                y0=0, y1=1, yref="paper",
                fillcolor="#1a1a18", opacity=0.4,
                layer="below", line_width=0,
            )

    # ── Layout ────────────────────────────────────────────────────────────
    run_dt_ct = run_dt.astimezone(pd.Timestamp.now("America/Chicago").tz)
    run_str   = run_dt_ct.strftime("%b %-d, %-I %p %Z")
    test_r2   = meta.get("test_r2", "—")

    fig.update_layout(
        title=dict(
            text=(f"<b>ERCOT 48-Hour Load Forecast</b>"
                  f"<span style='font-size:13px;color:#888580;'>"
                  f"  ·  Generated {run_str}  ·  Test R² = {test_r2}</span>"),
            font=dict(color="#e8e6e0", size=16, family="Inter, sans-serif"),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(
            title=dict(text="Date / Time (Central)",
                       font=dict(color="#e8e6e0", size=12)),
            color="#5a5855",
            gridcolor="#2a2a28",
            showgrid=True, gridwidth=0.5,
            tickformat="%-m/%-d<br>%-I %p",
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title=dict(text="ERCOT Load (GW)",
                       font=dict(color="#e8e6e0", size=12)),
            color="#5a5855",
            gridcolor="#2a2a28",
            showgrid=True, gridwidth=0.5,
            range=[y_min, y_max],
        ),
        plot_bgcolor="#181816",
        paper_bgcolor="#0f0f0d",
        font=dict(family="Inter, sans-serif", color="#5a5855"),
        margin=dict(l=70, r=40, t=70, b=80),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1a18",
            bordercolor="#2a2a28",
            font=dict(color="#e8e6e0", size=12, family="Inter, sans-serif"),
        ),
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(26,26,24,0.6)",
            bordercolor="#2a2a28", borderwidth=1,
            font=dict(color="#e8e6e0", size=11),
        ),
        height=480,
    )

    fig.add_annotation(
        text="LightGBM · Open-Meteo HRRR · 2m temp + dewpoint · 10m wind · cloud cover",
        xref="paper", yref="paper",
        x=0.99, y=-0.16,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(color="#5a5855", size=10, family="Inter, sans-serif"),
    )

    fig.write_html(
        HTML_PATH,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "modeBarButtonsToRemove":
                ["select2d", "lasso2d", "autoScale2d"]},
    )
    print(f"Interactive chart saved → {HTML_PATH}")


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

    # Fetch recent actuals for context
    actuals = fetch_recent_actuals(hours_back=24)

    make_forecast_plot(load_fcst, run_dt, meta)
    make_interactive_plot(load_fcst, run_dt, meta, actuals=actuals)
    print("Done.")
