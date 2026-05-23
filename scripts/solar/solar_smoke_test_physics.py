"""Solar pipeline smoke test (physics version).

End-to-end test using the proper physics model:
  1. Load USPVDB inventory
  2. Fetch HRRR DSWRF, VBDSF, VDDSF, TMP for one cycle/hour
  3. Sample each field at every plant's lat/lon
  4. Compute AC power per plant using solar_power_model:
     - Sun position from lat/lon/UTC time
     - Tracker geometry (fixed/single-axis/dual-axis)
     - POA irradiance from HRRR beam+diffuse (no empirical decomposition)
     - Cell temperature with INOCT model
     - DC power with PVWatts + temperature derate
     - AC power with inverter clipping at nameplate

Usage:
    python solar_smoke_test_physics.py
"""
from __future__ import annotations

import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from herbie import Herbie

from solar_inventory import load_uspvdb
from solar_power_model import compute_plant_power, resolve_plant_config


def find_recent_cycle() -> datetime:
    """Find a recent HRRR cycle that's published on AWS."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                              microsecond=0, tzinfo=None)
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
    """Fetch one variable. Returns xarray DataArray or None."""
    try:
        ds = H.xarray(search)
        var = list(ds.data_vars)[0]
        return ds[var]
    except Exception as e:
        print(f"  WARN: {label} fetch failed: {e}")
        return None


def sample_at_points(field, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Brute-force nearest-neighbor sampling on HRRR Lambert conformal grid."""
    grid_lats = field.latitude.values
    grid_lons = field.longitude.values
    values = field.values
    lons_360 = lons % 360.0

    out = np.empty(len(lats), dtype=np.float32)
    for i, (lat, lon) in enumerate(zip(lats, lons_360)):
        d2 = (grid_lats - lat) ** 2 + (grid_lons - lon) ** 2
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        out[i] = values[iy, ix]
    return out


def main() -> int:
    print("=" * 70)
    print("Solar pipeline smoke test (physics model)")
    print("=" * 70)

    # 1. Load inventory
    print("\n[1/5] Loading USPVDB inventory...")
    inv = load_uspvdb()

    # 2. Find a recent HRRR cycle
    print("\n[2/5] Finding recent HRRR cycle...")
    cycle = find_recent_cycle()
    print(f"  Cycle: {cycle.strftime('%Y-%m-%d %H:%MZ')}")
    H = Herbie(cycle, model="hrrr", product="sfc", fxx=0)

    # 3. Fetch radiation + temperature fields
    print("\n[3/5] Fetching HRRR fields...")
    dswrf = fetch_field(H, ":DSWRF:surface", "DSWRF")
    vbdsf = fetch_field(H, ":VBDSF:surface", "VBDSF")
    vddsf = fetch_field(H, ":VDDSF:surface", "VDDSF")
    tmp2m = fetch_field(H, ":TMP:2 m above ground", "TMP 2m")

    if any(x is None for x in [dswrf, vbdsf, vddsf, tmp2m]):
        print("ERROR: missing required HRRR fields, can't compute solar")
        return 1

    print(f"  DSWRF range: {float(dswrf.min()):.1f} to {float(dswrf.max()):.1f} W/m²")
    print(f"  VBDSF range: {float(vbdsf.min()):.1f} to {float(vbdsf.max()):.1f} W/m²")
    print(f"  VDDSF range: {float(vddsf.min()):.1f} to {float(vddsf.max()):.1f} W/m²")
    print(f"  TMP range:   {float(tmp2m.min())-273.15:.1f} to "
          f"{float(tmp2m.max())-273.15:.1f} °C")

    # 4. Sample at each plant location (~30 sec for 6500 plants)
    print(f"\n[4/5] Sampling at {len(inv):,} plant locations...")
    t0 = time_module.time()
    lats = inv["ylat"].values
    lons = inv["xlong"].values
    inv["dswrf"] = sample_at_points(dswrf, lats, lons)
    inv["vbdsf"] = sample_at_points(vbdsf, lats, lons)
    inv["vddsf"] = sample_at_points(vddsf, lats, lons)
    inv["tmp_k"] = sample_at_points(tmp2m, lats, lons)
    print(f"  Sampling done in {time_module.time()-t0:.1f}s")

    # 5. Run physics per plant
    print(f"\n[5/5] Running solar physics model for each plant...")
    t0 = time_module.time()
    times_index = pd.DatetimeIndex([pd.Timestamp(cycle).tz_localize("UTC")])

    mw_ac = np.zeros(len(inv), dtype=np.float32)
    mw_dc = np.zeros(len(inv), dtype=np.float32)
    poa = np.zeros(len(inv), dtype=np.float32)
    temp_cell = np.zeros(len(inv), dtype=np.float32)

    for i, row in enumerate(inv.itertuples(index=False)):
        config = resolve_plant_config(pd.Series({
            "p_axis": row.p_axis,
            "p_tilt": getattr(row, "p_tilt", np.nan),
            "p_azimuth": getattr(row, "p_azimuth", np.nan),
            "p_cap_ac": row.p_cap_ac,
            "p_cap_dc": getattr(row, "p_cap_dc", np.nan),
            "ylat": row.ylat,
        }))
        result = compute_plant_power(
            times_utc=times_index,
            lat=row.ylat, lon=row.xlong,
            config=config,
            dswrf=np.array([row.dswrf]),
            beam=np.array([row.vbdsf]),
            diffuse=np.array([row.vddsf]),
            temp_k=np.array([row.tmp_k]),
        )
        mw_ac[i] = result["mw_ac"][0]
        mw_dc[i] = result["mw_dc"][0]
        poa[i] = result["poa_global"][0]
        temp_cell[i] = result["temp_cell_c"][0]

        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1:,}/{len(inv):,} plants processed "
                  f"({time_module.time()-t0:.0f}s elapsed)")

    inv["MW_AC"] = mw_ac
    inv["MW_DC"] = mw_dc
    inv["POA_W_m2"] = poa
    inv["T_cell_C"] = temp_cell
    print(f"  Physics done in {time_module.time()-t0:.1f}s")

    # Summary
    print()
    print("=" * 70)
    total_cap = inv["p_cap_ac"].sum()
    total_gen = inv["MW_AC"].sum()
    cf = (total_gen / total_cap * 100) if total_cap > 0 else 0
    print(f"Fleet-wide at {cycle.strftime('%Y-%m-%d %H:%MZ')}:")
    print(f"  Total AC capacity:    {total_cap:,.0f} MW")
    print(f"  Total estimated MW:   {total_gen:,.0f} MW")
    print(f"  Capacity factor:      {cf:.1f}%")

    print()
    print("By state (top 10 by capacity):")
    by_state = inv.groupby("p_state").agg(
        n=("p_name", "size"),
        cap_MW=("p_cap_ac", "sum"),
        gen_MW=("MW_AC", "sum"),
    )
    by_state["CF_pct"] = (by_state["gen_MW"] / by_state["cap_MW"] * 100).round(1)
    by_state = by_state.sort_values("cap_MW", ascending=False).head(10)
    print(by_state.round(0).to_string())

    print()
    print("By tracking type:")
    by_axis = inv.groupby("p_axis").agg(
        n=("p_name", "size"),
        cap_MW=("p_cap_ac", "sum"),
        gen_MW=("MW_AC", "sum"),
    )
    by_axis["CF_pct"] = (by_axis["gen_MW"] / by_axis["cap_MW"] * 100).round(1)
    print(by_axis.round(0).to_string())

    print()
    print("Top 10 plants by capacity:")
    sample = inv.nlargest(10, "p_cap_ac")[
        ["p_name", "p_state", "p_cap_ac", "p_axis",
         "dswrf", "POA_W_m2", "T_cell_C", "MW_AC"]
    ].copy()
    sample.columns = ["name", "st", "cap_MW", "axis",
                      "DSWRF", "POA", "Tcell", "MW_AC"]
    print(sample.round(1).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
