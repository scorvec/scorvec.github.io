"""USPVDB inventory loader.

Loads the USGS US Large-Scale Solar Photovoltaic Database, filters to
operational utility-scale plants in CONUS, and exposes a clean
DataFrame for downstream processing. Mirrors the role of
turbine_inventory.py in the wind pipeline.

USPVDB v4.0 (April 2026) contains 6,611 facilities ≥1 MW AC.

If data/uspvdb.csv doesn't exist, this script will download the latest
version once and cache it locally. Subsequent runs read from cache.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# Where the CSV lives locally
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
USPVDB_LOCAL = DATA_DIR / "uspvdb.csv"

# USPVDB REST API endpoint. Documented at
#   https://energy.usgs.gov/uspvdb/api-doc/
# No authentication needed for read-only GETs. Base is /api/uspvdb/v1/,
# resource is /projects. Pagination via offset and limit query params.
USPVDB_API = "https://energy.usgs.gov/api/uspvdb/v1/projects"

# Browser-style UA in case the host filters non-browser requests
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}

# Match the wind pipeline: drop non-CONUS states (no HRRR coverage)
NONCONUS_STATES = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}


def download_uspvdb(force: bool = False) -> Path:
    """Download USPVDB via REST API if not cached. Returns path to local CSV.

    The API returns paginated JSON. We pull all pages, convert to a
    DataFrame, and write to disk as CSV for downstream loading.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if USPVDB_LOCAL.exists() and not force:
        sz = USPVDB_LOCAL.stat().st_size / 1024
        print(f"  USPVDB cached: {USPVDB_LOCAL} ({sz:.0f} KB)")
        return USPVDB_LOCAL

    print(f"  Downloading USPVDB via API {USPVDB_API} ...")
    all_records = []
    offset = 0
    limit = 1000
    while True:
        params = {"offset": offset, "limit": limit}
        try:
            r = requests.get(USPVDB_API, headers=_UA, params=params, timeout=60)
            r.raise_for_status()
            records = r.json()
        except Exception as e:
            raise RuntimeError(
                f"USPVDB API request failed: {e}\n"
                f"\n"
                f"Fallback: download manually from\n"
                f"  https://energy.usgs.gov/uspvdb/data\n"
                f"and save as {USPVDB_LOCAL}\n"
            )

        if not records:
            break
        all_records.extend(records)
        print(f"    offset {offset}: got {len(records)} records "
              f"(total so far: {len(all_records):,})")
        if len(records) < limit:
            break
        offset += limit
        if offset > 50000:  # safety cap, ~50k records max
            print(f"    safety cap reached at offset {offset}, stopping")
            break

    if not all_records:
        raise RuntimeError(
            "USPVDB API returned no records.\n"
            f"Fallback: download manually from "
            "https://energy.usgs.gov/uspvdb/data\n"
            f"and save as {USPVDB_LOCAL}"
        )

    df = pd.DataFrame(all_records)
    df.to_csv(USPVDB_LOCAL, index=False)
    print(f"  Saved: {USPVDB_LOCAL} ({len(all_records):,} records, "
          f"{USPVDB_LOCAL.stat().st_size/1024:.0f} KB)")
    return USPVDB_LOCAL


def load_uspvdb(force_download: bool = False) -> pd.DataFrame:
    """Load USPVDB and return a cleaned DataFrame.

    Returned columns:
        case_id        USPVDB project ID
        eia_id         EIA-860 plant ID (for ISO actuals matching)
        p_name         project name
        p_state        US state code
        p_year         year of commissioning
        xlong, ylat    centroid coordinates
        p_cap_dc       MW DC nameplate
        p_cap_ac       MW AC nameplate
        p_axis         tracking type: single-axis, fixed, dual-axis, unknown
        p_tilt         tilt angle (degrees, for fixed-tilt; NaN for trackers)
        p_azimuth      azimuth (degrees from north, for fixed-tilt)
        p_btech        panel technology: c-Si, CdTe, CIGS, a-Si, unknown
        p_type         site type: ground-mounted, rooftop, canopy, floating
        p_status       operational, under construction, etc.
    """
    csv_path = download_uspvdb(force=force_download)
    df = pd.read_csv(csv_path, low_memory=False)

    # USPVDB v4.0 actual schema (verified against API docs):
    #   case_id, multi_poly, eia_id, p_state, p_county, ylat, xlong,
    #   p_area, p_img_date, p_dig_conf, p_name, p_year, p_pwr_reg,
    #   p_tech_pri, p_tech_sec, p_axis, p_azimuth, p_tilt, p_battery,
    #   p_cap_ac, p_cap_dc, p_type, p_agrivolt, p_zscore, p_comm,
    #   p_sys_type
    # Schema is stable; we don't need a heavy rename map. Just check
    # required columns exist.
    required = {"case_id", "p_name", "xlong", "ylat", "p_cap_ac",
                "p_state", "p_axis", "p_sys_type"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"USPVDB CSV missing required columns: {missing}\n"
            f"Available columns: {sorted(df.columns)}\n"
            f"USPVDB schema may have changed; update solar_inventory.py."
        )

    # Add columns that might be missing in older versions
    for col, default in [
        ("eia_id", np.nan),
        ("p_year", np.nan),
        ("p_cap_dc", np.nan),
        ("p_tilt", np.nan),
        ("p_azimuth", np.nan),
        ("p_tech_pri", "PV"),
        ("p_tech_sec", "unknown"),
    ]:
        if col not in df.columns:
            df[col] = default

    # Balancing-authority code: USPVDB names it p_pwr_reg. Expose a
    # unified `BA` column so the combined inventory (USPVDB + EIA-860M,
    # which carries its own BA Code) aggregates to real BAs, not states.
    if "p_pwr_reg" not in df.columns:
        df["p_pwr_reg"] = np.nan
    df["BA"] = df["p_pwr_reg"]

    print(f"  Loaded USPVDB: {len(df):,} facilities")

    # All records in USPVDB are operational (LBNL only adds completed
    # facilities). No status filter needed.

    # Filter: CONUS only (drop AK, HI, PR, GU, VI, AS, MP)
    before = len(df)
    df = df[~df["p_state"].astype(str).str.upper().isin(NONCONUS_STATES)].copy()
    print(f"    CONUS filter:     {len(df):,} (dropped {before-len(df):,} non-CONUS)")

    # Filter: have coordinates and capacity > 0
    before = len(df)
    df = df.dropna(subset=["xlong", "ylat", "p_cap_ac"]).copy()
    df = df[df["p_cap_ac"] > 0].copy()
    print(f"    Sanity filter:    {len(df):,} (dropped {before-len(df):,} "
          f"missing coords or zero MW)")

    # Optional: drop rooftop and canopy if we want ground-mount only.
    # For now keep everything but report it.
    if "p_sys_type" in df.columns:
        sys_counts = df.groupby("p_sys_type").agg(
            n=("p_name", "size"), mw=("p_cap_ac", "sum"))
        print(f"\n    By system type:")
        print(sys_counts.to_string())

    # Reset index
    df = df.reset_index(drop=True)

    # Summary
    print(f"\n  Final inventory:  {len(df):,} solar plants")
    print(f"    Total MW AC:    {df['p_cap_ac'].sum():,.0f}")
    print(f"    Total MW DC:    {df['p_cap_dc'].sum():,.0f}" if df["p_cap_dc"].notna().any() else "")
    print(f"    By state (top 10):")
    by_state = (df.groupby("p_state")["p_cap_ac"].sum()
                  .sort_values(ascending=False).head(10))
    for state, mw in by_state.items():
        print(f"      {state}: {mw:,.0f} MW")
    print(f"\n    By tracking type:")
    by_axis = df.groupby("p_axis").agg(
        n=("p_name", "size"),
        mw=("p_cap_ac", "sum"),
    ).sort_values("mw", ascending=False)
    print(by_axis.to_string())

    return df


def load_solar_inventory(
    force_download: bool = False,
    use_eia860m: bool = True,
) -> pd.DataFrame:
    """Load USPVDB plus optional EIA-860M supplement for newer plants.

    This is the preferred entry point for the forecast pipeline. It
    returns a DataFrame combining:
      - USPVDB (primary): authoritative plant geometry, tracking, etc.
      - EIA-860M (supplement): plants commissioned since USPVDB release,
                                with default geometry (assumes single-axis)

    Set use_eia860m=False to skip supplement (useful for debugging or
    if EIA download fails).
    """
    uspvdb = load_uspvdb(force_download=force_download)

    if not use_eia860m:
        uspvdb["source"] = "USPVDB"
        return uspvdb

    # Import here to keep solar_eia860m optional
    try:
        # Try both layouts: scripts/solar/solar_eia860m.py and
        # scripts/solar_eia860m.py (depending on where this gets installed)
        try:
            from solar.solar_eia860m import load_eia860m_supplement
        except ImportError:
            from solar_eia860m import load_eia860m_supplement
    except ImportError as e:
        print(f"  Could not import solar_eia860m: {e}")
        print(f"  Skipping EIA-860M supplement; using USPVDB only.")
        uspvdb["source"] = "USPVDB"
        return uspvdb

    # Get the set of EIA Plant IDs already covered by USPVDB
    uspvdb_eia_ids = set(uspvdb["eia_id"].dropna().astype(int).tolist())
    print(f"\n  USPVDB covers {len(uspvdb_eia_ids):,} unique EIA plant IDs")

    print(f"\n  Loading EIA-860M supplement...")
    try:
        supplement = load_eia860m_supplement(uspvdb_eia_ids=uspvdb_eia_ids)
    except Exception as e:
        print(f"  EIA-860M supplement failed: {e}", file=sys.stderr)
        print(f"  Continuing with USPVDB only.")
        uspvdb["source"] = "USPVDB"
        return uspvdb

    if supplement.empty:
        uspvdb["source"] = "USPVDB"
        return uspvdb

    # Tag USPVDB rows for downstream provenance tracking
    uspvdb = uspvdb.copy()
    uspvdb["source"] = "USPVDB"

    # Align columns: take the union, add missing columns as NaN
    all_cols = sorted(set(uspvdb.columns) | set(supplement.columns))
    for col in all_cols:
        if col not in uspvdb.columns:
            uspvdb[col] = np.nan
        if col not in supplement.columns:
            supplement[col] = np.nan

    combined = pd.concat([uspvdb[all_cols], supplement[all_cols]],
                          ignore_index=True)
    print(f"\n  Combined inventory: {len(combined):,} plants "
          f"({combined['p_cap_ac'].sum():,.0f} MW)")

    # Compare to USPVDB alone
    delta_n = len(combined) - len(uspvdb)
    delta_mw = combined["p_cap_ac"].sum() - uspvdb["p_cap_ac"].sum()
    print(f"    Added {delta_n:,} plants, {delta_mw:,.0f} MW vs USPVDB alone")

    # State breakdown including supplement
    print(f"\n  Top 10 states by MW (USPVDB + EIA-860M):")
    by_state = (combined.groupby("p_state")["p_cap_ac"].sum()
                  .sort_values(ascending=False).head(10))
    for state, mw in by_state.items():
        eia_mw = supplement[supplement["p_state"] == state]["p_cap_ac"].sum()
        if eia_mw > 0:
            print(f"    {state}: {mw:,.0f} MW (+{eia_mw:,.0f} from EIA-860M)")
        else:
            print(f"    {state}: {mw:,.0f} MW")

    return combined


if __name__ == "__main__":
    print("=" * 70)
    print("USPVDB solar inventory loader")
    print("=" * 70)
    df = load_solar_inventory()
    print()
    print("Sample (5 random plants):")
    print(df.sample(min(5, len(df)), random_state=42)[
        ["case_id", "p_name", "p_state", "p_cap_ac", "p_axis", "source"]
    ].to_string(index=False))
