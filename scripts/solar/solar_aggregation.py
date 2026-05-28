"""Solar forecast aggregation to ISO/region totals.

Joins per-plant forecast with capacity inventory, applies a state →
ISO/region mapping, and sums MW per region per hour.

Output: assets/solar_forecast_data/forecast_region_<cycle>.csv
        Long format: region, valid_time, MW_AC, capacity_MW

State → region mapping uses simple state codes for a first pass.
Some states (TX, NM, MT) span multiple BAs; we assign each state to
its predominant ISO. Refinements using EIA-861 BA boundaries can
come later.

Usage:
    python solar_aggregation.py                # latest cycle
    python solar_aggregation.py 20260523T18Z   # specific cycle
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "assets" / "solar_forecast_data"


# State → ISO/region mapping. Predominant ISO per state.
STATE_TO_REGION = {
    # ERCOT (Texas, most of it)
    "TX": "ERCOT",

    # CAISO
    "CA": "CAISO",

    # SPP
    "NE": "SPP", "KS": "SPP", "OK": "SPP", "AR": "SPP", "SD": "SPP",
    "ND": "SPP", "MT": "SPP",   # Montana is split; most renewable in WAPA/SPP

    # MISO
    "IL": "MISO", "IN": "MISO", "IA": "MISO", "MO": "MISO", "MI": "MISO",
    "WI": "MISO", "MN": "MISO", "MS": "MISO", "LA": "MISO",
    # Note: parts of IL/IN/MI/KY are PJM but most utility solar in MISO
    "KY": "MISO",

    # PJM
    "PA": "PJM", "NJ": "PJM", "MD": "PJM", "DE": "PJM", "VA": "PJM",
    "WV": "PJM", "OH": "PJM", "DC": "PJM",

    # Southeast (non-RTO)
    "FL": "Southeast", "GA": "Southeast", "NC": "Southeast",
    "SC": "Southeast", "AL": "Southeast", "TN": "Southeast",

    # ISO-NE
    "CT": "ISO-NE", "MA": "ISO-NE", "RI": "ISO-NE", "VT": "ISO-NE",
    "NH": "ISO-NE", "ME": "ISO-NE",

    # NYISO
    "NY": "NYISO",

    # Other (Western non-CAISO)
    "WA": "West", "OR": "West", "ID": "West", "WY": "West", "NV": "West",
    "UT": "West", "CO": "West", "AZ": "West", "NM": "West",
}

# Regions we'll surface on the dashboard
DASHBOARD_REGIONS = ["ERCOT", "CAISO", "MISO", "PJM", "SPP", "Southeast",
                     "ISO-NE", "NYISO"]


def get_latest_cycle() -> Optional[str]:
    """Find the most recent forecast file in DATA_DIR."""
    if not DATA_DIR.exists():
        return None
    files = sorted(DATA_DIR.glob("forecast_plant_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    # filename pattern: forecast_plant_YYYYMMDDThhZ.csv
    return files[0].stem.replace("forecast_plant_", "")


def main():
    print("=" * 70)
    print("Solar forecast aggregation")
    print("=" * 70)

    # 1. Determine cycle
    if len(sys.argv) >= 2:
        cycle_str = sys.argv[1]
    else:
        cycle_str = get_latest_cycle()
        if cycle_str is None:
            print(f"ERROR: no forecast files found in {DATA_DIR}")
            return 1

    print(f"\nCycle: {cycle_str}")

    # 2. Load forecast + capacity CSVs
    forecast_path = DATA_DIR / f"forecast_plant_{cycle_str}.csv"
    capacity_path = DATA_DIR / f"capacity_plant_{cycle_str}.csv"
    if not forecast_path.exists():
        print(f"ERROR: {forecast_path} not found")
        return 1
    if not capacity_path.exists():
        print(f"ERROR: {capacity_path} not found")
        return 1

    forecast = pd.read_csv(forecast_path, parse_dates=["valid_time"])
    capacity = pd.read_csv(capacity_path)
    forecast["case_id"] = forecast["case_id"].astype(str)
    capacity["case_id"] = capacity["case_id"].astype(str)
    print(f"  Loaded forecast: {len(forecast):,} rows")
    print(f"  Loaded capacity: {len(capacity):,} plants")

    # 3. Apply region mapping
    capacity["region"] = (capacity["p_state"]
                           .astype(str).str.upper()
                           .map(STATE_TO_REGION)
                           .fillna("Other"))

    # 4. Join + aggregate
    merged = forecast.merge(
        capacity[["case_id", "region", "p_cap_ac"]],
        on="case_id", how="left",
    )
    n_unmapped = merged["region"].isna().sum()
    if n_unmapped > 0:
        print(f"  WARN: {n_unmapped} forecast rows with no region mapping")

    # Aggregate generation by region × valid_time
    agg = (merged.groupby(["region", "valid_time"], dropna=False)["MW_AC"]
                 .sum().reset_index())

    # Add static capacity per region (sum of nameplate)
    region_cap = (capacity.groupby("region")["p_cap_ac"]
                          .sum().reset_index()
                          .rename(columns={"p_cap_ac": "capacity_MW"}))
    agg = agg.merge(region_cap, on="region", how="left")

    # Capacity factor for convenience
    agg["CF"] = (agg["MW_AC"] / agg["capacity_MW"]).clip(lower=0, upper=1.0)

    # Sort
    agg = agg.sort_values(["region", "valid_time"]).reset_index(drop=True)

    # 5. Save
    out_path = DATA_DIR / f"forecast_region_{cycle_str}.csv"
    agg.to_csv(out_path, index=False)
    print(f"\nWrote {len(agg):,} rows → {out_path}")

    # 6. Summary
    print()
    print("=" * 70)
    print("Region summary:")
    print("=" * 70)
    summary = (agg.groupby("region")
                  .agg(capacity_MW=("capacity_MW", "first"),
                       peak_MW=("MW_AC", "max"),
                       mean_MW=("MW_AC", "mean")))
    summary["peak_CF"] = (summary["peak_MW"] / summary["capacity_MW"] * 100).round(1)
    summary = summary.sort_values("capacity_MW", ascending=False)
    print(summary.round(0).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
