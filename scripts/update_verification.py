#!/usr/bin/env python3
"""
Daily incremental verification update for the historical dashboard card.

Maintains a rolling 14-day verification CSV at
`assets/wind_forecast_data/verification.csv` by:

  1. Loading any existing verification.csv (resume from last run)
  2. Stacking all forecast_iso_*.csv files in the archive
  3. Selecting the day-ahead representative lead (12-24h) per
     (region, valid_time)
  4. Pulling actuals and curtailment from gridstatus.io ONLY for the
     date range we don't already have
  5. Merging and saving the rolling 14-day window
  6. Generating the verification dashboard HTML

Designed for low API usage: pulls just the missing days, ~5-10 requests
per run. Run it manually once a day or weekly:

    cd scripts/
    export GRIDSTATUS_API_KEY=<key>
    python update_verification.py

Then `git add assets/ && git commit && git push` to publish.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Only these regions get curtailment overlay because they're the only ones
# where we have curtailment data we trust. Adding more here just means
# adding to CURTAILMENT_DATASETS and possibly a new parser branch.
DASHBOARD_REGIONS = ["ERCOT", "SPP", "CAISO"]

# Rolling window — historical card shows this many days.
HISTORY_DAYS = 14

# Day-ahead representative lead window. Forecasts initialized 12-24 hours
# before valid_time approximate the day-ahead market window.
LEAD_MIN_HRS = 12.0
LEAD_MAX_HRS = 24.0

ARCHIVE_DIR = Path("../assets/wind_forecast_data")
VER_PATH = ARCHIVE_DIR / "verification.csv"
DASHBOARD_PATH = Path("../assets/wind_verification.html")

ISO_FUEL_MIX = {
    "ERCOT":   ("ercot_fuel_mix",   "wind"),
    "SPP":     ("spp_fuel_mix",     "wind"),
    "CAISO":   ("caiso_fuel_mix",   "wind"),
}

CURTAILMENT_DATASETS = {
    "CAISO":  "caiso_curtailment",
    "SPP":    "spp_ver_curtailments",
    "ERCOT":  "ercot_hourly_wind_report",
}


# ---------------------------------------------------------------------------
# Forecast archive loader (same as backtest.py)
# ---------------------------------------------------------------------------

def load_forecast_archive(archive_dir: Path) -> pd.DataFrame:
    files = sorted(archive_dir.glob("forecast_iso_*.csv"))
    rows = []
    for path in files:
        cycle_str = path.stem.split("_")[-1]
        cycle = pd.Timestamp(cycle_str.replace("Z", ""))
        df = pd.read_csv(path, parse_dates=["valid_time"])
        region_col = "iso" if "iso" in df.columns else "ba_code"
        df = df.rename(columns={region_col: "region"})
        df["cycle"] = cycle
        df["lead_hours"] = (df["valid_time"] - cycle).dt.total_seconds() / 3600.0
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def select_day_ahead(fc: pd.DataFrame) -> pd.DataFrame:
    """Pick one forecast per (region, valid_time) in the day-ahead window."""
    in_window = fc[(fc["lead_hours"] >= LEAD_MIN_HRS)
                   & (fc["lead_hours"] <= LEAD_MAX_HRS)]
    idx = in_window.groupby(["region", "valid_time"])["cycle"].idxmax()
    return in_window.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# gridstatus fetchers (copy of backtest.py logic, keeping only ERCOT/SPP/CAISO)
# ---------------------------------------------------------------------------

def pull_actuals(client, regions: list[str], start: str, end: str) -> pd.DataFrame:
    pieces = []
    for iso_label in regions:
        if iso_label not in ISO_FUEL_MIX:
            continue
        dataset, col = ISO_FUEL_MIX[iso_label]
        print(f"  {iso_label}...", end=" ", flush=True)
        try:
            df = client.get_dataset(dataset, start=start, end=end)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        ts_col = "interval_start_utc" if "interval_start_utc" in df.columns else "Time"
        df = df.rename(columns={ts_col: "valid_time", col: "actual_MW"})
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)
        hourly = (df.set_index("valid_time")[["actual_MW"]]
                    .resample("1h").mean()
                    .reset_index())
        hourly["region"] = iso_label
        pieces.append(hourly[["region", "valid_time", "actual_MW"]])
        print(f"{len(hourly)} hrs")
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def pull_curtailment(client, regions: list[str], start: str, end: str) -> pd.DataFrame:
    pieces = []
    for iso_label in regions:
        if iso_label not in CURTAILMENT_DATASETS:
            continue
        dataset = CURTAILMENT_DATASETS[iso_label]
        print(f"  {iso_label}...", end=" ", flush=True)
        try:
            df = client.get_dataset(dataset, start=start, end=end)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        ts_col = "interval_start_utc" if "interval_start_utc" in df.columns else "Time"
        df = df.rename(columns={ts_col: "valid_time"})
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_localize(None)

        if iso_label == "CAISO":
            wind_mask = df["fuel_type"].astype(str).str.strip().str.lower() == "wind"
            wind = df[wind_mask]
            if len(wind) == 0:
                print(f"FAILED (no wind rows)")
                continue
            out = (wind.groupby("valid_time")["curtailment_mw"].sum()
                       .resample("1h").mean().reset_index()
                       .rename(columns={"curtailment_mw": "curtailment_MW"}))
        elif iso_label == "SPP":
            # Canonical pair only — see backtest.py for the explanation
            canonical_cols = [c for c in df.columns
                              if c in ("wind_redispatch_curtailments",
                                       "wind_manual_curtailments")]
            if not canonical_cols:
                print(f"FAILED (no canonical wind cols)")
                continue
            df["_total"] = df[canonical_cols].sum(axis=1)
            out = (df.set_index("valid_time")[["_total"]]
                      .resample("1h").mean().reset_index()
                      .rename(columns={"_total": "curtailment_MW"}))
        elif iso_label == "ERCOT":
            for c in ("actual_system_wide", "hsl_system_wide",
                      "cop_hsl_system_wide"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["actual_system_wide"])
            if len(df) == 0:
                print(f"FAILED (no populated actuals)")
                continue
            df["_potential"] = df["hsl_system_wide"].fillna(
                df["cop_hsl_system_wide"])
            df["_curt"] = (df["_potential"] - df["actual_system_wide"]).clip(lower=0)
            if "publish_time_utc" in df.columns:
                df = df.sort_values("publish_time_utc")
                df = df.drop_duplicates(subset=["valid_time"], keep="last")
            out = (df.set_index("valid_time")[["_curt"]]
                      .resample("1h").mean().reset_index()
                      .rename(columns={"_curt": "curtailment_MW"}))
        else:
            continue

        out["region"] = iso_label
        pieces.append(out[["region", "valid_time", "curtailment_MW"]])
        print(f"{len(out)} hrs, mean {out['curtailment_MW'].mean():.0f} MW")
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if "GRIDSTATUS_API_KEY" not in os.environ:
        print("ERROR: set GRIDSTATUS_API_KEY environment variable")
        return 1

    print("=" * 70)
    print("Verification updater")
    print("=" * 70)

    # Forecast archive
    fc = load_forecast_archive(ARCHIVE_DIR)
    if fc.empty:
        print(f"ERROR: no forecast archive found at {ARCHIVE_DIR}")
        return 1
    fc = fc[fc["region"].isin(DASHBOARD_REGIONS)]
    fc_da = select_day_ahead(fc)

    # Trim to rolling window
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=HISTORY_DAYS)
    fc_da = fc_da[fc_da["valid_time"] >= cutoff]
    print(f"Forecast archive: {len(fc_da):,} (region, valid_time) rows in window")
    if fc_da.empty:
        print("No forecasts within the rolling window — nothing to do.")
        return 0

    # Existing verification (resume)
    existing = pd.DataFrame()
    if VER_PATH.exists():
        existing = pd.read_csv(VER_PATH, parse_dates=["valid_time"])
        existing = existing[existing["valid_time"] >= cutoff]
        # Drop rows beyond the dashboard regions in case we trimmed
        existing = existing[existing["region"].isin(DASHBOARD_REGIONS)]
        print(f"Existing verification: {len(existing):,} rows "
              f"({existing['valid_time'].min()} → {existing['valid_time'].max()})")

    # Compute date range we still need actuals for
    if not existing.empty:
        already_have = set(zip(existing["region"], existing["valid_time"]))
        needed_mask = ~fc_da.apply(
            lambda r: (r["region"], r["valid_time"]) in already_have, axis=1)
        needed = fc_da[needed_mask]
    else:
        needed = fc_da

    if needed.empty:
        print("Already up to date.")
        # Still regenerate dashboard in case styling/code changed
        _build_dashboard(existing)
        return 0

    start_str = needed["valid_time"].min().strftime("%Y-%m-%d")
    end_str = (needed["valid_time"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Need to fetch: {start_str} → {end_str} "
          f"({len(needed):,} (region, valid_time) rows)")

    from gridstatusio import GridStatusClient
    client = GridStatusClient()

    print("\nFetching actuals:")
    actuals_new = pull_actuals(client, DASHBOARD_REGIONS, start_str, end_str)
    print("\nFetching curtailment:")
    curtail_new = pull_curtailment(client, DASHBOARD_REGIONS, start_str, end_str)

    # Build new verification rows
    ver_new = needed[["region", "valid_time", "MW", "cycle", "lead_hours"]].rename(
        columns={"MW": "forecast_MW"})
    if not actuals_new.empty:
        ver_new = ver_new.merge(actuals_new,
                                 on=["region", "valid_time"], how="left")
    if not curtail_new.empty:
        ver_new = ver_new.merge(curtail_new,
                                 on=["region", "valid_time"], how="left")

    # Merge with existing
    ver = pd.concat([existing, ver_new], ignore_index=True)
    ver = ver.drop_duplicates(subset=["region", "valid_time"], keep="last")
    ver = ver.sort_values(["region", "valid_time"]).reset_index(drop=True)

    # Compute derived columns (regenerate to be safe)
    if "curtailment_MW" in ver.columns:
        ver["curtailment_MW"] = ver["curtailment_MW"].fillna(0.0)
        ver["forecast_minus_curtail_MW"] = ver["forecast_MW"] - ver["curtailment_MW"]
        ver["error_after_curtail_MW"] = (ver["forecast_minus_curtail_MW"]
                                          - ver["actual_MW"])
    ver["error_MW"] = ver["forecast_MW"] - ver["actual_MW"]

    # Save
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ver.to_csv(VER_PATH, index=False)
    print(f"\nWrote {VER_PATH}: {len(ver):,} rows")

    # Print summary
    print("\nCurrent error metrics:")
    for region in DASHBOARD_REGIONS:
        sub = ver[ver["region"] == region]
        if sub.empty:
            continue
        e = (sub["forecast_MW"] - sub["actual_MW"]).dropna()
        if "forecast_minus_curtail_MW" in sub.columns:
            ea = (sub["forecast_minus_curtail_MW"] - sub["actual_MW"]).dropna()
            print(f"  {region:6s}  n={len(e):4d}  "
                  f"physics_bias={e.mean():+7.0f} MW  "
                  f"after_curt_bias={ea.mean():+7.0f} MW")
        else:
            print(f"  {region:6s}  n={len(e):4d}  "
                  f"physics_bias={e.mean():+7.0f} MW")

    _build_dashboard(ver)
    return 0


def _build_dashboard(ver: pd.DataFrame) -> None:
    """Generate the historical verification HTML."""
    from verification_dashboard import build_verification_dashboard
    if ver is None or ver.empty:
        print("No data to build dashboard from")
        return
    print(f"\nBuilding dashboard: {DASHBOARD_PATH}")
    # Ensure the CSV exists for the dashboard builder
    ver.to_csv(VER_PATH, index=False)
    build_verification_dashboard(
        csv_path=VER_PATH,
        output_html=DASHBOARD_PATH,
        default_region="ERCOT",
        plotly_js="cdn",
        theme="dark",
    )


if __name__ == "__main__":
    sys.exit(main())
