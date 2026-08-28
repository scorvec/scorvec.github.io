#!/usr/bin/env python3
"""
Daily incremental verification update for the historical dashboard card.

Maintains a rolling 14-day verification CSV at
`assets/wind_forecast_data/verification.csv` by:

  1. Loading any existing verification.csv (resume from last run)
  2. Stacking all forecast_iso_*.csv files in the archive
  3. Selecting the day-ahead representative lead (12-24h) per
     (region, valid_time)
  4. Pulling actuals directly from the ISOs (iso_direct_fetchers) ONLY
     for the date range we don't already have
  5. Merging and saving the rolling 14-day window
  6. Generating the verification dashboard HTML

The gridstatus.io hosted API was removed 2026-07: actuals come from the
free ISO public endpoints (ERCOT/MISO direct HTTP; SPP/CAISO via the
open-source gridstatus scraping library). Curtailment, which only the
hosted API carried, is no longer collected. Run manually:

    cd scripts/
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
elif sys.path[0] != _HERE:
    sys.path.remove(_HERE)
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
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Verification updater")
    print("=" * 70)

    # Environment diagnostic — helps debug "library not installed" issues
    import sys
    print(f"Python: {sys.executable}")
    try:
        import gridstatus as _gs
        print(f"gridstatus library: v{_gs.__version__} (open-source, direct ISO scraping)")
    except ImportError as e:
        print(f"gridstatus library: NOT INSTALLED ({e})")

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

    # Compute date range we still need actuals for. A (region, valid_time)
    # pair counts as "already have" only if it exists AND has a non-null
    # actual_MW — otherwise the row was inserted before gridstatus had
    # published the actual, and we need to retry.
    if not existing.empty and "actual_MW" in existing.columns:
        resolved = existing[existing["actual_MW"].notna()]
        already_have = set(zip(resolved["region"], resolved["valid_time"]))
        needed_mask = ~fc_da.apply(
            lambda r: (r["region"], r["valid_time"]) in already_have, axis=1)
        needed = fc_da[needed_mask]
    elif not existing.empty:
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

    # Actuals come directly from the ISOs' free public endpoints
    # (ERCOT/MISO direct HTTP; SPP/CAISO via the open-source gridstatus
    # scraping library). Set USE_ISO_DIRECT=0 to skip fetching entirely.
    actuals_new = pd.DataFrame()
    if os.environ.get("USE_ISO_DIRECT", "1") == "1":
        print("\nFetching actuals (direct from ISOs):")
        try:
            from iso_direct_fetchers import pull_actuals_direct
            actuals_new = pull_actuals_direct(DASHBOARD_REGIONS,
                                              start_str, end_str)
        except Exception as e:
            print(f"  ISO-direct fetchers errored ({e}); skipping")

    # Build new verification rows
    ver_new = needed[["region", "valid_time", "MW", "cycle", "lead_hours"]].rename(
        columns={"MW": "forecast_MW"})
    if not actuals_new.empty:
        ver_new = ver_new.merge(actuals_new,
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
    if ver is None or ver.empty:
        print("No data to build dashboard from")
        return
    try:
        from verification_dashboard import build_verification_dashboard
    except ImportError as e:
        print(f"WARNING: verification_dashboard.py not found ({e}); "
              f"CSV was written but dashboard HTML was not built.")
        print(f"  __file__ = {__file__}")
        print(f"  _HERE    = {_HERE}")
        print(f"  cwd      = {os.getcwd()}")
        print(f"  sys.path[0:5] = {sys.path[:5]}")
        print(f"  Files in _HERE: {sorted(os.listdir(_HERE))[:10]}…")
        return
    print(f"\nBuilding dashboard: {DASHBOARD_PATH}")
    # Ensure the CSV exists for the dashboard builder
    ver.to_csv(VER_PATH, index=False)
    try:
        build_verification_dashboard(
            csv_path=VER_PATH,
            output_html=DASHBOARD_PATH,
            default_region="ERCOT",
            plotly_js="cdn",
            theme="dark",
        )
    except Exception as e:
        print(f"WARNING: dashboard build failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
