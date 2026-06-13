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
    "FL": "Southeast (non-RTO)", "GA": "Southeast (non-RTO)", "NC": "Southeast (non-RTO)",
    "SC": "Southeast (non-RTO)", "AL": "Southeast (non-RTO)", "TN": "Southeast (non-RTO)",

    # ISO-NE
    "CT": "ISO-NE", "MA": "ISO-NE", "RI": "ISO-NE", "VT": "ISO-NE",
    "NH": "ISO-NE", "ME": "ISO-NE",

    # NYISO
    "NY": "NYISO",

    # Other (Western non-CAISO)
    "WA": "WECC (non-CAISO)", "OR": "WECC (non-CAISO)", "ID": "WECC (non-CAISO)", "WY": "WECC (non-CAISO)", "NV": "WECC (non-CAISO)",
    "UT": "WECC (non-CAISO)", "CO": "WECC (non-CAISO)", "AZ": "WECC (non-CAISO)", "NM": "WECC (non-CAISO)",
}

# Regions we'll surface on the dashboard. MUST cover every region the
# state→region map can emit, or the dashboard's "National" (= sum of these)
# silently undercounts — "West" (AZ/NV/NM/UT/CO, ~24 GW) was missing.
DASHBOARD_REGIONS = ["ERCOT", "CAISO", "WECC (non-CAISO)", "MISO",
                     "Southeast (non-RTO)", "PJM", "SPP", "ISO-NE", "NYISO"]

# Grid-scale cut: the dashboard counts only transmission-connected solar
# (what balancing authorities meter as generation), so totals line up with
# grid actuals. Smaller distribution/BTM-sized plants show up as reduced
# demand, not generation. Mirrors GRID_SCALE_MIN_MW in scripts/power_hero.py.
GRID_SCALE_MIN_MW = 10.0


# Balancing-authority code → display region. Preferred over the state map
# because it matches how grid operators actually meter solar (e.g. ISO-NE
# vs the distribution BTM it excludes; CISO ≠ the non-CAISO California BAs
# LDWP/BANC/IID, which belong to WECC "West"). Anything not listed falls
# back to STATE_TO_REGION. Codes are EIA-860M / USPVDB p_pwr_reg BA codes.
BA_TO_REGION = {
    # RTOs / ISOs (the BA code is the region)
    "ERCO": "ERCOT", "CISO": "CAISO", "MISO": "MISO", "PJM": "PJM",
    "SWPP": "SPP", "ISNE": "ISO-NE", "NYIS": "NYISO",
    # Southeast (non-RTO SERC/FRCC)
    "FPL": "Southeast (non-RTO)", "SOCO": "Southeast (non-RTO)", "CPLE": "Southeast (non-RTO)",
    "DUK": "Southeast (non-RTO)", "FPC": "Southeast (non-RTO)", "TVA": "Southeast (non-RTO)",
    "TEC": "Southeast (non-RTO)", "SCEG": "Southeast (non-RTO)", "SC": "Southeast (non-RTO)",
    "SEPA": "Southeast (non-RTO)", "FMPP": "Southeast (non-RTO)", "SEC": "Southeast (non-RTO)",
    "TAL": "Southeast (non-RTO)", "LGEE": "Southeast (non-RTO)", "JEA": "Southeast (non-RTO)",
    "AEC": "Southeast (non-RTO)", "GVL": "Southeast (non-RTO)", "HST": "Southeast (non-RTO)",
    "SPA": "SPP",
    # West (WECC non-CAISO — incl. non-CAISO California BAs)
    "NEVP": "WECC (non-CAISO)", "PACE": "WECC (non-CAISO)", "PSCO": "WECC (non-CAISO)", "AZPS": "WECC (non-CAISO)",
    "PNM": "WECC (non-CAISO)", "SRP": "WECC (non-CAISO)", "LDWP": "WECC (non-CAISO)", "IPCO": "WECC (non-CAISO)",
    "WACM": "WECC (non-CAISO)", "TEPC": "WECC (non-CAISO)", "IID": "WECC (non-CAISO)", "EPE": "WECC (non-CAISO)",
    "WALC": "WECC (non-CAISO)", "AVRN": "WECC (non-CAISO)", "PACW": "WECC (non-CAISO)", "BANC": "WECC (non-CAISO)",
    "DOPD": "WECC (non-CAISO)", "BPAT": "WECC (non-CAISO)", "PGE": "WECC (non-CAISO)", "NWMT": "WECC (non-CAISO)",
    "WAUW": "WECC (non-CAISO)", "AVA": "WECC (non-CAISO)", "PSEI": "WECC (non-CAISO)", "SCL": "WECC (non-CAISO)",
    "TIDC": "WECC (non-CAISO)", "GCPD": "WECC (non-CAISO)", "CHPD": "WECC (non-CAISO)", "GWA": "WECC (non-CAISO)",
    "GRID": "WECC (non-CAISO)", "DEAA": "WECC (non-CAISO)", "HGMA": "WECC (non-CAISO)", "GRIF": "WECC (non-CAISO)",
    "WAUE": "WECC (non-CAISO)",
}


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

    # 2. Load forecast + capacity CSVs. Capacity is now the canonical static-fleet
    #    file (unstamped); fall back to the legacy cycle-stamped name for back-compat.
    forecast_path = DATA_DIR / f"forecast_plant_{cycle_str}.csv"
    capacity_path = DATA_DIR / "capacity_plant.csv"
    if not capacity_path.exists():
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

    # 3. Apply region mapping — prefer the real balancing authority (BA),
    #    fall back to the state map for any plant without a BA code.
    state_region = (capacity["p_state"].astype(str).str.upper()
                                       .map(STATE_TO_REGION))
    if "BA" in capacity.columns:
        ba_region = (capacity["BA"].astype(str).str.upper()
                                   .map(BA_TO_REGION))
        n_by_ba = ba_region.notna().sum()
        capacity["region"] = ba_region.fillna(state_region).fillna("Other")
        print(f"  Region mapping: {n_by_ba:,}/{len(capacity):,} plants via BA, "
              f"{len(capacity) - n_by_ba:,} via state fallback")
    else:
        print("  WARN: capacity_plant.csv has no BA column; using state map only")
        capacity["region"] = state_region.fillna("Other")

    # 3b. Grid-scale cut — keep only transmission-connected solar (what BAs
    #     meter as generation) so regional totals line up with grid actuals.
    #     Distribution/BTM-sized plants (<GRID_SCALE_MIN_MW, or unassigned to a
    #     BA) are dropped; they appear as reduced demand, not generation.
    before_n, before_gw = len(capacity), capacity["p_cap_ac"].sum() / 1000
    gs = capacity["p_cap_ac"] >= GRID_SCALE_MIN_MW
    if "BA" in capacity.columns:
        gs &= capacity["BA"].notna()
    capacity = capacity[gs].copy()
    print(f"  Grid-scale filter (≥{GRID_SCALE_MIN_MW:.0f} MW): "
          f"{len(capacity):,}/{before_n:,} plants kept, "
          f"{capacity['p_cap_ac'].sum() / 1000:.1f}/{before_gw:.1f} GW")

    # 4. Join + aggregate (inner: forecast rows for dropped plants are excluded)
    merged = forecast.merge(
        capacity[["case_id", "region", "p_cap_ac"]],
        on="case_id", how="inner",
    )

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
