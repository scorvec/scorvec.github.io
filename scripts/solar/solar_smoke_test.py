"""Solar pipeline smoke test.

End-to-end test of the foundational solar pipeline:
  1. Load USPVDB inventory
  2. Fetch HRRR DSWRF, VBDSF, VDDSF, TMP for one cycle/hour
  3. Sample each at every plant's lat/lon
  4. Compute crude per-plant MW estimate:
       MW = (DSWRF/1000) * capacity_AC * system_derate
     where system_derate = 0.86 (NREL ATB benchmark)

This is a deliberately oversimplified physics model — no panel
geometry, no tracker logic, no temperature derate. Just a sanity check
that the data flow works end to end. Real physics happens in later
sessions.

Usage:
    python solar_smoke_test.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from herbie import Herbie

from solar_inventory import load_uspvdb


# Crude physics constants
SYSTEM_DERATE = 0.86          # NREL ATB benchmark, accounts for inverter, wiring, soiling, etc.
STC_IRRADIANCE = 1000.0       # Standard test conditions irradiance, W/m²


def find_recent_cycle() -> datetime:
    """Find a recent HRRR cycle, preferring daytime UTC so we get
    nonzero DSWRF at most CONUS locations."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                              microsecond=0, tzinfo=None)
    # Walk back hour by hour until we find a cycle Herbie can access
    for hours_back in range(2, 10):
        candidate = now - timedelta(hours=hours_back)
        try:
            H = Herbie(candidate, model="hrrr", product="sfc", fxx=0)
            if H.grib is not None:
                return candidate
        except Exception:
            continue
    raise RuntimeError("No recent HRRR cycle available")


def fetch_field(H: Herbie, search: str, label: str):
    """Fetch one variable. Returns the xarray DataArray or None on failure."""
    try:
        ds = H.xarray(search)
        var = list(ds.data_vars)[0]
        return ds[var]
    except Exception as e:
        print(f"  WARN: {label} fetch failed: {e}")
        return None


def sample_at_points(field, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Sample `field` at each (lat, lon) point. Returns array of values
    aligned with input arrays. Uses brute-force nearest-neighbor since
    HRRR is a 2D Lambert conformal grid (no monotonic 1D coords).

    For ~6000 plants this takes a few seconds; if it becomes a
    bottleneck we can optimize with cKDTree later.
    """
    grid_lats = field.latitude.values   # 2D
    grid_lons = field.longitude.values  # 2D, in 0-360 convention
    values = field.values               # 2D

    # Convert input lons to 0-360 to match grid
    lons_360 = lons % 360.0

    out = np.empty(len(lats), dtype=np.float32)
    for i, (lat, lon) in enumerate(zip(lats, lons_360)):
        d2 = (grid_lats - lat) ** 2 + (grid_lons - lon) ** 2
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        out[i] = values[iy, ix]
    return out


def main() -> int:
    print("=" * 70)
    print("Solar pipeline smoke test")
    print("=" * 70)

    # 1. Load inventory
    print("\n[1/4] Loading USPVDB inventory...")
    inv = load_uspvdb()

    # 2. Find a recent HRRR cycle
    print("\n[2/4] Finding recent HRRR cycle...")
    cycle = find_recent_cycle()
    print(f"  Cycle: {cycle.strftime('%Y-%m-%d %H:%MZ')}")

    H = Herbie(cycle, model="hrrr", product="sfc", fxx=0)

    # 3. Fetch radiation fields
    print("\n[3/4] Fetching HRRR radiation + temp...")
    dswrf = fetch_field(H, ":DSWRF:surface", "DSWRF")
    vbdsf = fetch_field(H, ":VBDSF:surface", "VBDSF")
    vddsf = fetch_field(H, ":VDDSF:surface", "VDDSF")
    tmp2m = fetch_field(H, ":TMP:2 m above ground", "TMP 2m")

    if dswrf is None:
        print("ERROR: DSWRF unavailable, can't compute solar output")
        return 1

    print(f"  DSWRF shape: {dswrf.shape}")
    print(f"  DSWRF range (full grid): "
          f"{float(dswrf.min()):.1f} to {float(dswrf.max()):.1f} W/m^2")

    # 4. Sample at each plant and compute crude MW
    print(f"\n[4/4] Sampling at {len(inv):,} plant locations...")
    lats = inv["ylat"].values
    lons = inv["xlong"].values

    inv["dswrf"] = sample_at_points(dswrf, lats, lons)
    if vbdsf is not None:
        inv["vbdsf"] = sample_at_points(vbdsf, lats, lons)
    if vddsf is not None:
        inv["vddsf"] = sample_at_points(vddsf, lats, lons)
    if tmp2m is not None:
        inv["tmp2m_k"] = sample_at_points(tmp2m, lats, lons)
        inv["tmp2m_c"] = inv["tmp2m_k"] - 273.15

    # Crude MW estimate
    inv["MW_crude"] = (inv["dswrf"] / STC_IRRADIANCE
                       * inv["p_cap_ac"] * SYSTEM_DERATE).clip(lower=0)
    # Cap at AC nameplate (real systems clip via inverter)
    inv["MW_crude"] = inv["MW_crude"].clip(upper=inv["p_cap_ac"])

    # Summary
    print()
    total_cap = inv["p_cap_ac"].sum()
    total_mw = inv["MW_crude"].sum()
    cf = (total_mw / total_cap * 100) if total_cap > 0 else 0
    print(f"Fleet-wide at {cycle.strftime('%Y-%m-%d %H:%MZ')}:")
    print(f"  Total capacity:       {total_cap:,.0f} MW AC")
    print(f"  Total estimated MW:   {total_mw:,.0f} MW")
    print(f"  Capacity factor:      {cf:.1f}%")
    print()

    # By state
    print("By state (top 10 by capacity):")
    by_state = inv.groupby("p_state").agg(
        n_plants=("p_name", "size"),
        cap_MW=("p_cap_ac", "sum"),
        gen_MW=("MW_crude", "sum"),
    )
    by_state["CF_pct"] = (by_state["gen_MW"] / by_state["cap_MW"] * 100).round(1)
    by_state = by_state.sort_values("cap_MW", ascending=False).head(10)
    print(by_state.to_string())
    print()

    # Sample individual plants — sort by capacity and pick a few
    print("Sample plants (top 10 by capacity):")
    sample = inv.nlargest(10, "p_cap_ac")[
        ["p_name", "p_state", "p_cap_ac", "dswrf", "MW_crude"]
    ].copy()
    sample.columns = ["name", "st", "cap_MW", "DSWRF", "MW"]
    print(sample.round(1).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
