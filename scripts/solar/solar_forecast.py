"""Solar forecast generator (one HRRR cycle, 48 forecast hours).

For a given HRRR cycle, fetch radiation + temperature fields for each
forecast hour (F00 through F48), sample at each USPVDB plant location,
run the physics model, and save the per-plant per-hour AC power
forecast to CSV.

Output:
    assets/solar_forecast_data/forecast_plant_<cycle>.csv
        Per-plant per-hour: case_id, p_name, eia_id, p_state, valid_time, MW_AC
    assets/solar_forecast_data/capacity_plant_<cycle>.csv
        Per-plant static info: case_id, p_name, eia_id, p_state, lat, lon,
                               p_cap_ac, p_axis, p_sys_type

Usage:
    python solar_forecast.py                # current cycle
    python solar_forecast.py 2026-05-23 18  # specific cycle
"""
from __future__ import annotations

import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from herbie import Herbie

from solar_inventory import load_solar_inventory
from solar_grid_indices import build_grid_indices
from solar_power_model import compute_plant_power, resolve_plant_config


# Forecast horizon: HRRR cycle hours to process. F00-F18 are produced
# every cycle; F19-F48 only on 00/06/12/18Z cycles. We default to F00-F48
# and skip any that aren't available.
FORECAST_HOURS = list(range(0, 49))

# Output directory (relative to scripts/solar/, matches wind pipeline pattern)
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent.parent / "assets" / "solar_forecast_data"

# Fields to fetch each hour
HRRR_FIELDS = {
    "dswrf": ":DSWRF:surface",
    "vbdsf": ":VBDSF:surface",
    "vddsf": ":VDDSF:surface",
    "tmp":   ":TMP:2 m above ground",
}


def parse_cycle_arg(argv) -> datetime:
    """Parse cycle from CLI (YYYY-MM-DD HH), or default to most recent."""
    if len(argv) >= 3:
        date_str, hour_str = argv[1], argv[2]
        return datetime.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H")
    # Default: most recent 00/06/12/18Z extended cycle (which has F49+)
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                              microsecond=0, tzinfo=None)
    # Walk back hour by hour, prefer 00/06/12/18 for extended runs
    for hours_back in range(2, 24):
        candidate = now - timedelta(hours=hours_back)
        if candidate.hour in (0, 6, 12, 18):
            try:
                H = Herbie(candidate, model="hrrr", product="sfc", fxx=0)
                if H.grib is not None:
                    return candidate
            except Exception:
                continue
    raise RuntimeError("No suitable HRRR cycle available")


def fetch_field_for_hour(cycle: datetime, fxx: int, field_key: str):
    """Fetch one variable for one forecast hour. Returns DataArray or None."""
    try:
        H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx)
        ds = H.xarray(HRRR_FIELDS[field_key])
        var = list(ds.data_vars)[0]
        return ds[var]
    except Exception as e:
        return None


def sample_using_indices(field, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Fast vectorized sampling using pre-computed grid indices."""
    return field.values[iy, ix].astype(np.float32)


def main():
    print("=" * 70)
    print("Solar forecast generator")
    print("=" * 70)

    # 1. Determine cycle
    cycle = parse_cycle_arg(sys.argv)
    cycle_str = cycle.strftime("%Y%m%dT%HZ")
    print(f"\nCycle: {cycle.strftime('%Y-%m-%d %H:%MZ')} ({cycle_str})")

    # 2. Load inventory + grid indices
    print("\n[1/4] Loading inventory and grid indices...")
    inv = load_solar_inventory()
    grid_idx = build_grid_indices(force=False)
    # Join inventory with indices by case_id
    inv = inv.merge(grid_idx[["case_id", "iy", "ix"]], on="case_id", how="left")
    missing = inv["iy"].isna().sum()
    if missing > 0:
        print(f"  WARN: {missing} plants missing grid indices (will skip)")
        inv = inv.dropna(subset=["iy", "ix"]).copy()
    inv["iy"] = inv["iy"].astype(int)
    inv["ix"] = inv["ix"].astype(int)
    print(f"  {len(inv):,} plants ready")

    # Pre-resolve plant configs (avoids re-doing in inner loop)
    print("  Pre-resolving plant configs...")
    configs = [
        resolve_plant_config(pd.Series({
            "p_axis": getattr(row, "p_axis", np.nan),
            "p_tilt": getattr(row, "p_tilt", np.nan),
            "p_azimuth": getattr(row, "p_azimuth", np.nan),
            "p_cap_ac": row.p_cap_ac,
            "p_cap_dc": getattr(row, "p_cap_dc", np.nan),
            "ylat": row.ylat,
        }))
        for row in inv.itertuples(index=False)
    ]

    # 3. Loop over forecast hours
    print(f"\n[2/4] Fetching HRRR for {len(FORECAST_HOURS)} forecast hours...")
    iy_arr = inv["iy"].values
    ix_arr = inv["ix"].values
    lats_arr = inv["ylat"].values
    lons_arr = inv["xlong"].values

    all_rows = []
    fetch_time = 0.0
    physics_time = 0.0

    for fxx in FORECAST_HOURS:
        valid_time = cycle + timedelta(hours=fxx)
        print(f"  F{fxx:02d} → valid {valid_time.strftime('%Y-%m-%d %H:%MZ')} ... ",
              end="", flush=True)

        # Fetch all four fields for this hour
        t0 = time_module.time()
        fields = {}
        for key in HRRR_FIELDS:
            f = fetch_field_for_hour(cycle, fxx, key)
            if f is None:
                fields = None
                break
            fields[key] = f
        if fields is None:
            print("SKIP (one or more fields unavailable)")
            continue
        fetch_time += time_module.time() - t0

        # Sample at all plants
        dswrf = sample_using_indices(fields["dswrf"], iy_arr, ix_arr)
        vbdsf = sample_using_indices(fields["vbdsf"], iy_arr, ix_arr)
        vddsf = sample_using_indices(fields["vddsf"], iy_arr, ix_arr)
        tmp_k = sample_using_indices(fields["tmp"], iy_arr, ix_arr)

        # Run physics for each plant at this hour
        t0 = time_module.time()
        times_index = pd.DatetimeIndex([pd.Timestamp(valid_time).tz_localize("UTC")])
        mw_ac = np.zeros(len(inv), dtype=np.float32)
        for i, config in enumerate(configs):
            result = compute_plant_power(
                times_utc=times_index,
                lat=lats_arr[i], lon=lons_arr[i],
                config=config,
                dswrf=np.array([dswrf[i]]),
                beam=np.array([vbdsf[i]]),
                diffuse=np.array([vddsf[i]]),
                temp_k=np.array([tmp_k[i]]),
            )
            mw_ac[i] = result["mw_ac"][0]
        physics_time += time_module.time() - t0

        # Collect rows for this hour
        for i, row in enumerate(inv.itertuples(index=False)):
            if mw_ac[i] > 0.001:  # skip nighttime zeros to keep CSV tidy
                all_rows.append({
                    "case_id": row.case_id,
                    "valid_time": valid_time,
                    "MW_AC": mw_ac[i],
                })

        total_mw = mw_ac.sum()
        print(f"fleet={total_mw:>7,.0f} MW")

    print(f"\n  Total fetch time:   {fetch_time:.1f}s")
    print(f"  Total physics time: {physics_time:.1f}s")

    # 4. Write outputs
    print(f"\n[3/4] Writing per-plant forecast CSV...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df = pd.DataFrame(all_rows)
    if forecast_df.empty:
        print("ERROR: no forecast rows generated")
        return 1

    forecast_path = OUTPUT_DIR / f"forecast_plant_{cycle_str}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"  Wrote {len(forecast_df):,} rows → {forecast_path}")
    print(f"  Time range: {forecast_df['valid_time'].min()} → "
          f"{forecast_df['valid_time'].max()}")
    print(f"  Plants with nonzero output: "
          f"{forecast_df['case_id'].nunique():,}")

    # Per-plant static info (capacity, axis, location, ISO mapping later)
    print(f"\n[4/4] Writing per-plant capacity CSV...")
    capacity_df = inv[[
        "case_id", "p_name", "p_state", "p_cap_ac", "p_axis", "p_sys_type",
        "ylat", "xlong",
    ]].copy()
    # eia_id might not be in inventory; add empty if missing
    if "eia_id" in inv.columns:
        capacity_df["eia_id"] = inv["eia_id"].values

    capacity_path = OUTPUT_DIR / f"capacity_plant_{cycle_str}.csv"
    capacity_df.to_csv(capacity_path, index=False)
    print(f"  Wrote {len(capacity_df):,} plants → {capacity_path}")

    # Summary
    print()
    print("=" * 70)
    print("Forecast summary:")
    print("=" * 70)
    by_hour = forecast_df.groupby("valid_time")["MW_AC"].sum().sort_index()
    print(f"  Forecast hours: {len(by_hour)}")
    print(f"  Peak hour:     {by_hour.idxmax()} → {by_hour.max():>7,.0f} MW")
    print(f"  Min hour:      {by_hour.idxmin()} → {by_hour.min():>7,.0f} MW")
    print(f"  Mean hourly:   {by_hour.mean():>7,.0f} MW")
    print(f"  Total energy:  {by_hour.sum():>7,.0f} MWh over 48 hours")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
