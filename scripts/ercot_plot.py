"""
ERCOT Load Forecasting — LightGBM Model
----------------------------------------
Pulls 3 years of:
  - ERCOT system demand  →  EIA Open Data API (free key)
  - Hourly ASOS obs      →  Iowa State Mesonet (tmpf, dwpf, sknt, skyc)

Features: hour_ct, dow, month, temp_f, dwpt_f, wind_mph, cloud_pct,
          days_since_start, is_weekend
Target:   load_gw

Outputs:
  assets/ercot_load_temp.png          — raw scatter coloured by hour
  assets/ercot_diurnal.png            — average diurnal cycle by day of week
  assets/ercot_curves.png             — actual vs predicted on test set
  assets/ercot_feature_importance.png — LightGBM feature importances
  assets/lgbm_model.pkl               — trained model (loaded by forecast script)
  assets/model_meta.json              — training metadata incl. CV R² per fold

Usage:
  export EIA_API_KEY="your_key_here"
  python ercot_plot.py
"""

import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import StringIO
from datetime import datetime, timedelta, timezone
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# ── CONFIG ───────────────────────────────────────────────────────────────
STATIONS = {
    "KIAH": 0.28, "KHOU": 0.08, "KDFW": 0.22,
    "KSAT": 0.14, "KAUS": 0.12, "KELP": 0.05,
    "KCRP": 0.03, "KAMA": 0.03, "KBRO": 0.05,
}

FEATURES = ["hour_ct", "dow", "month", "temp_f", "dwpt_f",
            "wind_mph", "cloud_pct", "days_since_start", "is_weekend"]
TARGET    = "load_gw"
TEST_DAYS = 90

EIA_KEY  = os.environ.get("EIA_API_KEY", "")
end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=365 * 3)

BG, PANEL, LIGHT, MUTED = "#0f0f0d", "#181816", "#e8e6e0", "#5a5855"
DOW_NAMES  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
DOW_COLORS = ["#4a90d9","#5ba85c","#e8a838","#d95f4a","#9b59b6","#e87d4a","#48b8b8"]

# METAR sky-condition code → cloud-cover percentage (max-okta convention)
SKY_TO_PCT = {
    "CLR": 0.0, "SKC": 0.0, "CAVOK": 0.0, "NSC": 0.0,
    "FEW": 18.75, "SCT": 43.75, "BKN": 75.0, "OVC": 100.0, "VV": 100.0,
}

FEATURE_LABELS = {
    "hour_ct":          "Hour of Day",
    "dow":              "Day of Week",
    "month":            "Month",
    "temp_f":           "Temperature (°F)",
    "dwpt_f":           "Dewpoint (°F)",
    "wind_mph":         "Wind Speed (mph)",
    "cloud_pct":        "Cloud Cover (%)",
    "days_since_start": "Days Since Start\n(Load Growth Trend)",
    "is_weekend":       "Weekend Flag",
}


# ── 1. EIA ERCOT LOAD ────────────────────────────────────────────────────
def fetch_ercot_load():
    print("Fetching ERCOT load from EIA (3 years)...")
    url, records, offset = "https://api.eia.gov/v2/electricity/rto/region-data/data/", [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "hourly",
            "data[0]": "value", "facets[type][]": "D",
            "facets[respondent][]": "ERCO",
            "start": start_dt.strftime("%Y-%m-%dT%H"),
            "end":   end_dt.strftime("%Y-%m-%dT%H"),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
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
    df = (df[["period","load_mw"]].dropna()
            .drop_duplicates("period").set_index("period").sort_index())
    print(f"  Done — {len(df)} hourly records.")
    return df


# ── 2. ASOS ───────────────────────────────────────────────────────────────
def fetch_asos(station):
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station, "data": "tmpf,dwpf,sknt,skyc1,skyc2,skyc3",
        "year1": start_dt.year, "month1": start_dt.month, "day1": start_dt.day,
        "year2": end_dt.year,   "month2": end_dt.month,   "day2": end_dt.day,
        "tz": "UTC", "format": "comma", "latlon": "no",
        "direct": "no", "report_type": 3,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(lines)), na_values=["M", "T", " "])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"valid": "time", "tmpf": "temp_f", "dwpf": "dwpt_f",
                             "sknt": "wind_kts"})
    df["time"]     = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    df["temp_f"]   = pd.to_numeric(df["temp_f"],   errors="coerce")
    df["dwpt_f"]   = pd.to_numeric(df["dwpt_f"],   errors="coerce")
    df["wind_kts"] = pd.to_numeric(df["wind_kts"], errors="coerce")
    df["wind_mph"] = df["wind_kts"] * 1.15078

    # Take the maximum cloud coverage across the three reported sky layers
    for col in ["skyc1", "skyc2", "skyc3"]:
        if col in df.columns:
            df[col] = df[col].str.strip().map(SKY_TO_PCT)
    sky_cols = [c for c in ["skyc1", "skyc2", "skyc3"] if c in df.columns]
    df["cloud_pct"] = df[sky_cols].max(axis=1) if sky_cols else np.nan

    return (df[["time", "temp_f", "dwpt_f", "wind_mph", "cloud_pct"]]
            .dropna(subset=["temp_f", "dwpt_f"])
            .groupby("time").mean())


def fetch_weighted_obs():
    print("Fetching ASOS weather obs (3 years)...")
    wt_temp = wt_dwpt = wt_wind = wt_cloud = None
    total_w = 0.0
    for station, weight in STATIONS.items():
        try:
            obs  = fetch_asos(station)
            wt   = obs["temp_f"]    * weight
            wd   = obs["dwpt_f"]    * weight
            ww   = obs["wind_mph"]  * weight
            wc   = obs["cloud_pct"] * weight
            wt_temp  = wt if wt_temp  is None else wt_temp.add(wt,  fill_value=0)
            wt_dwpt  = wd if wt_dwpt  is None else wt_dwpt.add(wd,  fill_value=0)
            wt_wind  = ww if wt_wind  is None else wt_wind.add(ww,  fill_value=0)
            wt_cloud = wc if wt_cloud is None else wt_cloud.add(wc, fill_value=0)
            total_w += weight
            print(f"  {station}: OK ({len(obs)} obs)")
        except Exception as e:
            print(f"  {station}: SKIPPED — {e}")
        time.sleep(4)
    out = pd.DataFrame({
        "temp_f":    wt_temp  / total_w,
        "dwpt_f":    wt_dwpt  / total_w,
        "wind_mph":  wt_wind  / total_w,
        "cloud_pct": wt_cloud / total_w,
    })
    print(f"  Done — {len(out)} weighted hourly values.")
    return out


# ── 3. FEATURES ───────────────────────────────────────────────────────────
def build_features(df, training_start):
    """Add time-based ML features. temp_f/dwpt_f/wind_mph/cloud_pct assumed present."""
    df = df.copy()
    origin = pd.Timestamp(training_start, tz="UTC")
    df["hour_ct"]          = (df.index.hour - 6) % 24
    df["dow"]              = df.index.dayofweek
    df["month"]            = df.index.month
    df["days_since_start"] = (df.index - origin).days
    df["is_weekend"]       = (df["dow"] >= 5).astype(int)
    return df


# ── 4. MODEL TRAINING ─────────────────────────────────────────────────────
def train_model(df):
    """
    Run 5-fold time-series CV, then train final model on all data except
    the last TEST_DAYS days. Returns (model, train_r2, test_r2, df_test, cv_r2).
    """
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df[TARGET] > 20) & (df[TARGET] < 90)]

    cutoff   = df.index.max() - pd.Timedelta(days=TEST_DAYS)
    df_train = df[df.index <= cutoff]
    df_test  = df[df.index >  cutoff]

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]

    lgbm_params = dict(
        learning_rate=0.05,
        max_depth=7,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # 5-fold time-series cross-validation on the training window
    tscv  = TimeSeriesSplit(n_splits=5)
    cv_r2 = []
    print("  Time-series CV (5 folds):")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xf_tr,  yf_tr  = X_train.iloc[tr_idx],  y_train.iloc[tr_idx]
        Xf_val, yf_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        m = lgb.LGBMRegressor(n_estimators=500, **lgbm_params)
        m.fit(Xf_tr, yf_tr)
        r2 = r2_score(yf_val, m.predict(Xf_val))
        cv_r2.append(r2)
        print(f"    Fold {fold}: val R² = {r2:.4f}")
    print(f"  CV mean R² = {np.mean(cv_r2):.4f}  ±  {np.std(cv_r2):.4f}")

    # Final model with early stopping against held-out test set
    model = lgb.LGBMRegressor(n_estimators=1000, **lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2  = r2_score(y_test,  model.predict(X_test))
    print(f"  Train R² = {train_r2:.4f}   Test R² = {test_r2:.4f}")
    print(f"  Best iteration: {model.best_iteration_}")

    return model, train_r2, test_r2, df_test, cv_r2


# ── 5. SCATTER PLOT ───────────────────────────────────────────────────────
def make_scatter_plot(df):
    df = df.copy()
    df = df[(df[TARGET] > 20) & (df[TARGET] < 90)]
    df = df[(df["temp_f"] > 0) & (df["temp_f"] < 115)]
    cmap = cm.twilight_shifted
    norm = mcolors.Normalize(vmin=0, vmax=23)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    ax.scatter(df["temp_f"], df[TARGET],
               c=df["hour_ct"], cmap=cmap, norm=norm,
               s=4, alpha=0.4, linewidths=0, rasterized=True)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    _style_cbar(cbar)
    _style_ax(ax,
        xlabel="Population-Weighted Temperature (°F)",
        ylabel="ERCOT System Load (GW)",
        title="ERCOT Hourly Load vs. Temperature  ·  Last 3 Years",
        footnote="Sources: EIA Open Data (ERCOT demand)  ·  Iowa State Mesonet (ASOS obs)")
    os.makedirs("assets", exist_ok=True)
    fig.tight_layout()
    fig.savefig("assets/ercot_load_temp.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Scatter plot saved → assets/ercot_load_temp.png")


# ── 6. DIURNAL CYCLE PLOT ────────────────────────────────────────────────
def make_diurnal_plot(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    hours = np.arange(24)
    hour_labels = [f"{h%12 or 12}{'am' if h<12 else 'pm'}" for h in hours]
    for dow in range(7):
        sub     = df[df["dow"] == dow]
        profile = sub.groupby("hour_ct")[TARGET].agg(["mean","std"]).reindex(hours)
        color   = DOW_COLORS[dow]
        ls      = "--" if dow >= 5 else "-"
        ax.plot(hours, profile["mean"], color=color, linewidth=2.2,
                linestyle=ls, label=DOW_NAMES[dow], zorder=3)
        ax.fill_between(hours,
                        profile["mean"] - profile["std"],
                        profile["mean"] + profile["std"],
                        color=color, alpha=0.10, zorder=2)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a28")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(hours)
    ax.set_xticklabels(hour_labels, color=MUTED, fontsize=7.5, rotation=45, ha="right")
    ax.set_xlabel("Hour of Day (CT)", color=LIGHT, fontsize=10, labelpad=8)
    ax.set_ylabel("Average ERCOT Load (GW)", color=LIGHT, fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.15, fontsize=8,
              facecolor="#1a1a18", edgecolor="#2a2a28", labelcolor=LIGHT, ncol=2)
    date_str = datetime.now(timezone.utc).strftime("Updated %B %d, %Y")
    ax.set_title("ERCOT Average Diurnal Load Cycle by Day of Week  ·  Last 3 Years",
                 color=LIGHT, fontsize=13, fontweight="normal", loc="left", pad=14)
    ax.text(0.99, 1.012, date_str,
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.13,
            "Shaded band = ±1 std dev  ·  Dashed = weekend  ·  EIA Open Data",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)
    fig.tight_layout()
    fig.savefig("assets/ercot_diurnal.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Diurnal plot saved → assets/ercot_diurnal.png")


# ── 7. ACTUAL VS PREDICTED PLOT ───────────────────────────────────────────
def make_curves_plot(model, df_test, test_r2):
    y_actual = df_test[TARGET].values
    y_pred   = model.predict(df_test[FEATURES])
    hours    = df_test["hour_ct"].values

    cmap = cm.twilight_shifted
    norm = mcolors.Normalize(vmin=0, vmax=23)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

    ax.scatter(y_actual, y_pred,
               c=hours, cmap=cmap, norm=norm,
               s=5, alpha=0.4, linewidths=0, rasterized=True)

    lo = min(y_actual.min(), y_pred.min()) - 1
    hi = max(y_actual.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], color=LIGHT, linewidth=1.0,
            linestyle="--", alpha=0.4, zorder=5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.text(0.05, 0.95, f"R² = {test_r2:.4f}",
            transform=ax.transAxes, color=LIGHT, fontsize=14,
            fontweight="normal", va="top",
            bbox=dict(facecolor="#1a1a18", edgecolor="#2a2a28",
                      alpha=0.8, pad=6, boxstyle="round,pad=0.4"))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035)
    _style_cbar(cbar)

    _style_ax(ax,
        xlabel="Actual ERCOT Load (GW)",
        ylabel="Predicted ERCOT Load (GW)",
        title=f"LightGBM: Actual vs Predicted  ·  {TEST_DAYS}-Day Test Set",
        footnote=f"LightGBM · {len(FEATURES)} features · "
                 f"Trained on 3 years of data · Coloured by hour of day")

    fig.tight_layout()
    fig.savefig("assets/ercot_curves.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Actual vs predicted plot saved → assets/ercot_curves.png")


# ── 8. FEATURE IMPORTANCE PLOT ────────────────────────────────────────────
def make_importance_plot(model):
    importances = model.feature_importances_
    labels      = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    order       = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

    colors = [cm.twilight_shifted(v) for v in np.linspace(0.2, 0.85, len(order))]
    bars   = ax.barh(
        [labels[i] for i in order],
        importances[order],
        color=[colors[j] for j in range(len(order))],
        edgecolor="none", height=0.6,
    )

    for bar, val in zip(bars, importances[order]):
        ax.text(val + importances.max()*0.01, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}", va="center", color=MUTED, fontsize=8)

    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a28")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Feature Importance (LightGBM split gain)", color=LIGHT,
                  fontsize=10, labelpad=8)
    ax.set_facecolor(PANEL)
    ax.grid(axis="x", color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    date_str = datetime.now(timezone.utc).strftime("Updated %B %d, %Y")
    ax.set_title("LightGBM Feature Importances  ·  ERCOT Load Model",
                 color=LIGHT, fontsize=13, fontweight="normal", loc="left", pad=14)
    ax.text(0.99, 1.012, date_str,
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.12,
            "Split gain importance — higher = more predictive  ·  "
            "EIA Open Data  ·  Iowa State Mesonet",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)

    fig.tight_layout()
    fig.savefig("assets/ercot_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Feature importance plot saved → assets/ercot_feature_importance.png")


# ── STYLE HELPERS ─────────────────────────────────────────────────────────
def _style_cbar(cbar):
    cbar.set_label("Hour of Day (CT)", color=LIGHT, fontsize=9, labelpad=10)
    cbar.set_ticks([0, 6, 12, 18, 23])
    cbar.set_ticklabels(["Midnight","6 AM","Noon","6 PM","11 PM"])
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(MUTED)


def _style_ax(ax, xlabel, ylabel, title, footnote):
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a28")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=LIGHT, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=LIGHT, fontsize=10, labelpad=8)
    ax.grid(color="#2a2a28", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_title(title, color=LIGHT, fontsize=13, fontweight="normal", loc="left", pad=14)
    date_str = datetime.now(timezone.utc).strftime("Updated %B %d, %Y")
    ax.text(0.99, 1.012, date_str,
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.text(0.99, -0.09, footnote,
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not EIA_KEY:
        raise SystemExit("ERROR: EIA_API_KEY environment variable not set.")

    load = fetch_ercot_load()
    obs  = fetch_weighted_obs()

    df = pd.DataFrame({
        "load_mw":   load["load_mw"],
        "temp_f":    obs["temp_f"],
        "dwpt_f":    obs["dwpt_f"],
        "wind_mph":  obs["wind_mph"],
        "cloud_pct": obs["cloud_pct"],
    }).dropna(subset=["load_mw", "temp_f", "dwpt_f"])
    df["load_gw"] = df["load_mw"] / 1000.0

    training_start = start_dt.strftime("%Y-%m-%d")
    df = build_features(df, training_start)

    print(f"\nDataset: {len(df)} hourly points  "
          f"({start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')})")

    make_scatter_plot(df)
    make_diurnal_plot(df)

    print("\nTraining LightGBM...")
    model, train_r2, test_r2, df_test, cv_r2 = train_model(df)

    make_curves_plot(model, df_test, test_r2)
    make_importance_plot(model)

    os.makedirs("assets", exist_ok=True)
    joblib.dump(model, "assets/lgbm_model.pkl")
    meta = {
        "training_start": training_start,
        "training_end":   end_dt.strftime("%Y-%m-%d"),
        "features":       FEATURES,
        "train_r2":       round(train_r2, 4),
        "test_r2":        round(test_r2,  4),
        "cv_r2_folds":    [round(r, 4) for r in cv_r2],
        "cv_r2_mean":     round(float(np.mean(cv_r2)), 4),
        "cv_r2_std":      round(float(np.std(cv_r2)),  4),
        "n_train":        int(len(df) - len(df_test)),
        "n_test":         int(len(df_test)),
        "best_iteration": int(model.best_iteration_),
    }
    with open("assets/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel saved → assets/lgbm_model.pkl")
    print(f"Metadata  → assets/model_meta.json")
    print("Done.")
