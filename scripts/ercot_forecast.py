"""
ERCOT 48-Hour Load Forecast — LightGBM
----------------------------------------
Loads trained LightGBM model, downloads HRRR F00–F48:
  - 2 m temperature + dewpoint   (TMP, DPT)
  - 10 m wind components         (UGRD, VGRD) → wind speed in mph
  - Total cloud cover, entire atmosphere (TCDC)
Builds feature vector and runs inference for each forecast hour.

Reads:  assets/lgbm_model.pkl     (written by ercot_plot.py)
        assets/model_meta.json    (training metadata)
Writes: assets/ercot_forecast.png

Usage:
  python scripts/ercot_forecast.py
  python scripts/ercot_forecast.py --run 12
"""

import os
import json
import argparse
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
from herbie import Herbie

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

MODEL_PATH = "assets/lgbm_model.pkl"
META_PATH  = "assets/model_meta.json"
OUT_PATH   = "assets/ercot_forecast.png"
BG, PANEL, LIGHT, MUTED = "#0f0f0d", "#181816", "#e8e6e0", "#5a5855"


# ── 1. HRRR RUN DETECTION ────────────────────────────────────────────────
def get_hrrr_run(force_run_hour=None):
    now = datetime.now(timezone.utc)
    if force_run_hour is not None:
        run_dt = now.replace(hour=force_run_hour, minute=0, second=0, microsecond=0)
        if run_dt > now:
            run_dt -= timedelta(days=1)
        return run_dt
    for run_hour in [18, 12, 6, 0]:
        run_dt = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        if (now - run_dt).total_seconds() >= 90 * 60:
            return run_dt
    yesterday = now - timedelta(days=1)
    return yesterday.replace(hour=18, minute=0, second=0, microsecond=0)


# ── 2. HRRR DOWNLOAD ─────────────────────────────────────────────────────
def _nearest_value(ds, lat, lon, var=None):
    """Return the nearest-grid-point scalar; auto-selects the first data var if var is None."""
    lats = ds.latitude.values
    lons  = ds.longitude.values
    dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    idx  = np.unravel_index(dist.argmin(), dist.shape)
    if var is None:
        var = list(ds.data_vars)[0]
    return float(ds[var].values[idx])


def extract_at_point(ds, lat, lon, var):
    """Extract a field stored in Kelvin and return degrees Fahrenheit."""
    return (_nearest_value(ds, lat, lon, var) - 273.15) * 9/5 + 32


def fetch_hrrr(run_dt):
    print(f"Downloading HRRR {run_dt.strftime('%Y-%m-%d %H')}Z  F00–F48...")
    records = []
    for fxx in range(49):
        valid_time = run_dt + timedelta(hours=fxx)
        row = {"valid_time": valid_time, "fxx": fxx}
        try:
            H     = Herbie(run_dt.replace(tzinfo=None), model="hrrr", fxx=fxx, verbose=False)
            ds_t  = H.xarray("TMP:2 m above ground")
            ds_d  = H.xarray("DPT:2 m above ground")
            ds_u  = H.xarray("UGRD:10 m above ground")
            ds_v  = H.xarray("VGRD:10 m above ground")
            ds_tc = H.xarray("TCDC:entire atmosphere")
            for name, info in STATIONS.items():
                t_f  = extract_at_point(ds_t, info["lat"], info["lon"], "t2m")
                d_f  = extract_at_point(ds_d, info["lat"], info["lon"], "d2m")
                u_ms = _nearest_value(ds_u,  info["lat"], info["lon"], "u10")
                v_ms = _nearest_value(ds_v,  info["lat"], info["lon"], "v10")
                tcdc = _nearest_value(ds_tc, info["lat"], info["lon"])
                row[f"{name}_t"]    = t_f
                row[f"{name}_d"]    = d_f
                row[f"{name}_wspd"] = np.sqrt(u_ms**2 + v_ms**2) * 2.23694  # m/s → mph
                row[f"{name}_tcdc"] = tcdc
            print(f"  F{fxx:02d} ✓")
        except Exception as e:
            print(f"  F{fxx:02d} ✗  {e}")
            for name in STATIONS:
                row[f"{name}_t"]    = np.nan
                row[f"{name}_d"]    = np.nan
                row[f"{name}_wspd"] = np.nan
                row[f"{name}_tcdc"] = np.nan
        records.append(row)

    df = pd.DataFrame(records).set_index("valid_time")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


# ── 3. FEATURE BUILDER ────────────────────────────────────────────────────
def build_forecast_features(hrrr_df, training_start, features):
    """Build the same feature vector used during training."""
    total_w = sum(v["weight"] for v in STATIONS.values())
    wt     = sum(hrrr_df[f"{n}_t"]    * i["weight"] for n, i in STATIONS.items()) / total_w
    wd     = sum(hrrr_df[f"{n}_d"]    * i["weight"] for n, i in STATIONS.items()) / total_w
    wspd   = sum(hrrr_df[f"{n}_wspd"] * i["weight"] for n, i in STATIONS.items()) / total_w
    wcloud = sum(hrrr_df[f"{n}_tcdc"] * i["weight"] for n, i in STATIONS.items()) / total_w

    origin = pd.Timestamp(training_start, tz="UTC")
    df = pd.DataFrame(index=hrrr_df.index)
    df["hour_ct"]          = (df.index.hour - 5) % 24
    df["dow"]              = df.index.dayofweek
    df["month"]            = df.index.month
    df["temp_f"]           = wt
    df["dwpt_f"]           = wd
    df["wind_mph"]         = wspd
    df["cloud_pct"]        = wcloud
    df["days_since_start"] = (df.index - origin).days
    df["is_weekend"]       = (df["dow"] >= 5).astype(int)

    return df[features]


# ── 4. PLOT ───────────────────────────────────────────────────────────────
def make_forecast_plot(load_fcst, run_dt, meta):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

    cmap = cm.twilight_shifted
    norm = mcolors.Normalize(vmin=0, vmax=23)
    ct_hours = (load_fcst.index.hour - 5) % 24
    times    = load_fcst.index
    values   = load_fcst.values

    for i in range(len(times) - 1):
        if np.isnan(values[i]) or np.isnan(values[i+1]):
            continue
        ax.plot([times[i], times[i+1]], [values[i], values[i+1]],
                color=cmap(norm(ct_hours[i])), linewidth=2.5,
                solid_capstyle="round")

    now = datetime.now(timezone.utc)
    ax.axvline(now, color=LIGHT, linewidth=0.8, linestyle=":", alpha=0.5)

    day_start = times[0].normalize()
    for d in range(3):
        ds = day_start + pd.Timedelta(days=d)
        de = ds + pd.Timedelta(days=1)
        if d % 2 == 0:
            ax.axvspan(ds, de, color="#1a1a18", alpha=0.4, zorder=0)

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
    ax.set_xlabel("Date / Time (UTC)", color=LIGHT, fontsize=10, labelpad=8)
    ax.set_ylabel("ERCOT Load Forecast (GW)", color=LIGHT, fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d\n%HZ"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    valid = values[~np.isnan(values)]
    if len(valid):
        ax.set_ylim(max(20, valid.min()-3), min(90, valid.max()+3))

    run_str = run_dt.strftime("%Y-%m-%d %H")
    test_r2 = meta.get("test_r2", "—")
    ax.set_title(
        f"ERCOT 48-Hour Load Forecast  ·  HRRR {run_str}Z  ·  "
        f"Model Test R² = {test_r2}",
        color=LIGHT, fontsize=12, fontweight="normal", loc="left", pad=14)
    ax.text(0.99, 1.012,
            datetime.now(timezone.utc).strftime("Generated %B %d, %Y %H:%MZ"),
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.12,
            "LightGBM · HRRR 2m temp + dewpoint · 10m wind speed · total cloud cover",
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
                        help="Force HRRR run hour (0, 6, 12, or 18)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"ERROR: {MODEL_PATH} not found. Run ercot_plot.py first.")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"Model loaded — Test R² = {meta['test_r2']}  "
          f"(trained on {meta['n_train']:,} hours)")

    run_dt  = get_hrrr_run(force_run_hour=args.run)
    hrrr_df = fetch_hrrr(run_dt)

    X_fcst = build_forecast_features(hrrr_df, meta["training_start"], meta["features"])

    load_pred = np.clip(model.predict(X_fcst), 20, 90)
    load_fcst = pd.Series(load_pred, index=hrrr_df.index, name="load_gw_fcst")

    valid = load_fcst.dropna()
    print(f"Forecast range: {valid.min():.1f} – {valid.max():.1f} GW  "
          f"({len(valid)} hours)")

    make_forecast_plot(load_fcst, run_dt, meta)
    print("Done.")
