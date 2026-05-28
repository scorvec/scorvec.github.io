"""EIA-860M loader: supplements USPVDB with recently-built solar plants.

USPVDB releases annually (~April), so it can lag actual capacity by up
to a year. EIA-860M publishes monthly, with a ~6-week lag. We use it
to fill in plants that have come online since the most recent USPVDB
release.

This module returns a DataFrame in **USPVDB-compatible schema** for
plants NOT already in USPVDB. Downstream code (forecast pipeline)
treats EIA-only plants the same as USPVDB plants.

EIA-860M does NOT include tracking/tilt/azimuth metadata. We assume:
  - p_axis     = "single-axis"  (~73% of recent US installs)
  - p_tilt     = NaN              (tracker geometry handled in pvlib)
  - p_azimuth  = NaN

This means new EIA-only plants will be modeled as single-axis trackers
with default backtracking parameters. This is the dominant
configuration for recent (2023+) utility-scale installs and is the
best assumption without per-plant geometry.

Cached locally as data/eia860m_<year>_<month>.xlsx; downloaded once
per month.

Schema notes (EIA-860M "Operating" sheet, stable since ~2017):
  - Plant ID (also called "Plant Code"): integer, matches USPVDB eia_id
  - Plant Name: string
  - Generator ID: string (multiple per plant for phased projects)
  - Technology: filter to "Solar Photovoltaic"
  - Nameplate Capacity (MW): float, AC
  - Latitude / Longitude: float (sometimes 0/0 for new plants)
  - State: 2-letter
  - Status: "OP" = operating, "SB" = standby, etc.
  - Operating Year / Operating Month: when generator came online

Usage:
    from solar_eia860m import load_eia860m_supplement
    df = load_eia860m_supplement(uspvdb_eia_ids=set([1234, 5678, ...]))
    # df is in USPVDB schema, contains only plants not in uspvdb_eia_ids
"""
from __future__ import annotations

import calendar
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# EIA-860M latest URL pattern.
# Current month: /xls/<month>_generator<year>.xlsx
# Archived: /archive/xls/<month>_generator<year>.xlsx
EIA860M_CURRENT_BASE = "https://www.eia.gov/electricity/data/eia860m/xls"
EIA860M_ARCHIVE_BASE = "https://www.eia.gov/electricity/data/eia860m/archive/xls"

# Months in lowercase
MONTH_NAMES = [m.lower() for m in calendar.month_name[1:]]

# Browser-style UA in case host filters
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# CONUS filter, same as USPVDB
NONCONUS_STATES = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}


def _candidate_urls(today: date) -> list[tuple[str, int, int, str]]:
    """Build a list of (url, year, month, label) candidates, newest first.

    EIA-860M typically publishes the prior month's data in the 4th week
    of the current month. So in late May 2026, the latest available is
    likely April 2026.

    We try the previous 6 months and use the first one that downloads.
    For the current calendar month, the URL is at /xls/. For previous
    months, the URL is at /archive/xls/.
    """
    candidates = []
    for months_back in range(0, 7):
        # Walk backward
        target_year = today.year
        target_month = today.month - months_back
        while target_month <= 0:
            target_year -= 1
            target_month += 12
        month_name = MONTH_NAMES[target_month - 1]
        filename = f"{month_name}_generator{target_year}.xlsx"
        # The very latest month is in /xls/, earlier in /archive/xls/
        base = EIA860M_CURRENT_BASE if months_back == 0 else EIA860M_ARCHIVE_BASE
        candidates.append((f"{base}/{filename}", target_year, target_month,
                           f"{month_name.capitalize()} {target_year}"))
    return candidates


def download_eia860m(force: bool = False) -> tuple[Path, str]:
    """Download the most recent available EIA-860M file.

    Returns (local_path, label) where label is e.g. "April 2026".

    Caches locally as eia860m_<year>_<MM>.xlsx. Subsequent calls find
    the most recent cached file if any exists.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If forcing, skip cache check. Otherwise look for newest cached.
    if not force:
        cached = sorted(DATA_DIR.glob("eia860m_*.xlsx"))
        if cached:
            newest = cached[-1]
            label_parts = newest.stem.split("_")
            year = int(label_parts[1])
            month_num = int(label_parts[2])
            label = f"{MONTH_NAMES[month_num-1].capitalize()} {year}"
            sz = newest.stat().st_size / 1024 / 1024
            print(f"  EIA-860M cached: {newest.name} ({sz:.1f} MB, {label})")
            return newest, label

    today = date.today()
    candidates = _candidate_urls(today)

    for url, year, month, label in candidates:
        print(f"  Trying EIA-860M {label} ...")
        try:
            r = requests.get(url, headers=_UA, timeout=60, allow_redirects=True)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}, trying older month")
                continue
            if len(r.content) < 100_000:
                # Sanity: real file is ~5-10 MB. If we got something
                # tiny it's probably an error page.
                print(f"    file too small ({len(r.content)} bytes), skipping")
                continue
            local = DATA_DIR / f"eia860m_{year}_{month:02d}.xlsx"
            local.write_bytes(r.content)
            sz = local.stat().st_size / 1024 / 1024
            print(f"  EIA-860M downloaded: {local.name} ({sz:.1f} MB, {label})")
            return local, label
        except Exception as e:
            print(f"    download failed: {e}", file=sys.stderr)
            continue

    raise RuntimeError(
        "Failed to download any recent EIA-860M file.\n"
        f"Tried: {[c[3] for c in candidates]}\n"
        f"Manual download: https://www.eia.gov/electricity/data/eia860m\n"
        f"Place file in {DATA_DIR}/ named eia860m_<year>_<month>.xlsx"
    )


def load_eia860m_raw(force: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the Operating sheet of the latest EIA-860M.

    Returns (df, label) where label is the file's month-year.

    Filters: Operating status, Solar Photovoltaic technology, CONUS states.
    Raw columns preserved; aggregation and renaming done by
    load_eia860m_supplement().
    """
    local, label = download_eia860m(force=force)

    # EIA-860M sheet names have been stable: "Operating" tab has
    # operating generators. Other tabs are "Planned", "Retired",
    # "Canceled", etc. We want only operating + maybe standby.
    #
    # Header row position varies between releases (sometimes 1, sometimes
    # 2 banner rows before the real header). Auto-detect by reading the
    # first 10 rows headerless, then finding the row that contains
    # "Plant ID" or "Plant Code".
    print(f"  reading sheet 'Operating' from {local.name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Peek with no header to find where the column row is
        probe = pd.read_excel(local, sheet_name="Operating", header=None,
                              nrows=10, engine="openpyxl")
    header_row = None
    for i in range(len(probe)):
        row_strs = [str(x).strip() for x in probe.iloc[i].tolist()]
        if "Plant ID" in row_strs or "Plant Code" in row_strs:
            header_row = i
            break
    if header_row is None:
        raise RuntimeError(
            f"Could not find header row in EIA-860M file {local.name}. "
            f"First row values: {probe.iloc[0].tolist()[:8]}"
        )
    print(f"    header detected at row {header_row}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        df = pd.read_excel(local, sheet_name="Operating", header=header_row,
                           engine="openpyxl")
    print(f"    raw rows: {len(df):,}")

    # Standardize column names (EIA changes spaces/case occasionally)
    df.columns = [c.strip() for c in df.columns]

    # Required columns. The exact names have varied slightly over time.
    # We accept a few variants.
    col_aliases = {
        "Plant ID":              ["Plant ID", "Plant Code"],
        "Plant Name":            ["Plant Name"],
        "Generator ID":          ["Generator ID"],
        "Technology":            ["Technology"],
        "Nameplate Capacity":    ["Nameplate Capacity (MW)",
                                  "Net Summer Capacity (MW)"],
        "Latitude":              ["Latitude"],
        "Longitude":             ["Longitude"],
        "State":                 ["Plant State", "State"],
        "Status":                ["Status"],
        "Operating Year":        ["Operating Year"],
    }
    resolved = {}
    for canon, candidates in col_aliases.items():
        for c in candidates:
            if c in df.columns:
                resolved[canon] = c
                break
        else:
            raise RuntimeError(
                f"EIA-860M missing required column. Wanted one of "
                f"{candidates}, available: {sorted(df.columns)[:20]}..."
            )

    # Filter to Solar PV (Technology field)
    before = len(df)
    df = df[df[resolved["Technology"]].astype(str).str.strip()
              .str.lower() == "solar photovoltaic"].copy()
    print(f"    solar PV filter: {len(df):,} (dropped {before-len(df):,})")

    # Inspect Status values before filtering — EIA has used both
    # "OP" (two-letter code) and full names like "(OP) Operating" across
    # releases. Build a set of patterns considered "operating".
    status_col = resolved["Status"]
    raw_statuses = df[status_col].astype(str).str.strip()
    status_counts = raw_statuses.value_counts()
    print(f"    Status values present (top 5):")
    for val, count in status_counts.head(5).items():
        print(f"      {val!r}: {count:,}")

    # Match anything that looks like "operating" or "OP".
    # Common patterns: "OP", "Operating", "(OP) Operating"
    status_lower = raw_statuses.str.lower()
    is_operating = (
        (status_lower == "op")
        | (status_lower == "operating")
        | status_lower.str.contains(r"\(op\)", regex=True)
        | status_lower.str.contains(r"^operating\b", regex=True)
    )
    before = len(df)
    df = df[is_operating].copy()
    print(f"    OP status filter: {len(df):,} (dropped {before-len(df):,})")

    # Filter to CONUS
    before = len(df)
    df = df[~df[resolved["State"]].astype(str).str.upper()
              .isin(NONCONUS_STATES)].copy()
    print(f"    CONUS filter: {len(df):,} (dropped {before-len(df):,})")

    # Drop rows with missing/zero coordinates
    before = len(df)
    df = df.dropna(subset=[resolved["Latitude"], resolved["Longitude"],
                           resolved["Nameplate Capacity"]]).copy()
    df = df[(df[resolved["Latitude"]] != 0) | (df[resolved["Longitude"]] != 0)]
    df = df[df[resolved["Nameplate Capacity"]] > 0].copy()
    print(f"    coord+MW filter: {len(df):,} (dropped {before-len(df):,})")

    # Stash the column alias map for the next stage
    df.attrs["resolved_columns"] = resolved
    return df, label


def load_eia860m_supplement(
    uspvdb_eia_ids: Optional[set] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Build a USPVDB-compatible DataFrame from EIA-860M.

    Only includes plants whose Plant ID is NOT in `uspvdb_eia_ids`.
    Multiple generators at the same plant are summed (capacity) and
    averaged (lat/lon).

    Returns a DataFrame with USPVDB schema:
      case_id, eia_id, p_name, p_state, ylat, xlong, p_cap_ac, p_axis,
      p_tilt, p_azimuth, p_year, p_sys_type
    """
    if uspvdb_eia_ids is None:
        uspvdb_eia_ids = set()

    raw, label = load_eia860m_raw(force=force)
    cols = raw.attrs["resolved_columns"]

    # Filter: drop plants already in USPVDB
    before = len(raw)
    raw = raw[~raw[cols["Plant ID"]].isin(uspvdb_eia_ids)].copy()
    print(f"    USPVDB-overlap filter: {len(raw):,} generators "
          f"(dropped {before-len(raw):,} already in USPVDB)")

    if raw.empty:
        print("  No new plants in EIA-860M beyond USPVDB.")
        return pd.DataFrame()

    # Aggregate generators within the same plant (one row per plant)
    grouped = raw.groupby(cols["Plant ID"], as_index=False).agg({
        cols["Plant Name"]:            "first",
        cols["State"]:                 "first",
        cols["Latitude"]:              "mean",
        cols["Longitude"]:             "mean",
        cols["Nameplate Capacity"]:    "sum",
        cols["Operating Year"]:        "min",  # earliest commissioning
    })
    print(f"  Aggregated to {len(grouped):,} plants "
          f"(total {grouped[cols['Nameplate Capacity']].sum():,.0f} MW)")

    # Translate to USPVDB schema
    out = pd.DataFrame({
        # Synthesize case_id with EIA_ prefix to avoid collision with
        # USPVDB integer case_ids
        "case_id":     grouped[cols["Plant ID"]].apply(lambda p: f"EIA_{int(p)}"),
        "eia_id":      grouped[cols["Plant ID"]].astype(int),
        "p_name":      grouped[cols["Plant Name"]].astype(str),
        "p_state":     grouped[cols["State"]].astype(str),
        "ylat":        grouped[cols["Latitude"]].astype(float),
        "xlong":       grouped[cols["Longitude"]].astype(float),
        "p_cap_ac":    grouped[cols["Nameplate Capacity"]].astype(float),
        # Assume DC = AC × 1.30 (typical DC/AC ratio for new builds)
        "p_cap_dc":    grouped[cols["Nameplate Capacity"]].astype(float) * 1.30,
        # Single-axis is dominant for recent installs; safest default
        "p_axis":      "single-axis",
        # No fixed-tilt geometry for trackers
        "p_tilt":      np.nan,
        "p_azimuth":   np.nan,
        # Commissioning year
        "p_year":      grouped[cols["Operating Year"]].astype(float),
        # Assume ground-mounted (the dominant utility-scale type)
        "p_sys_type":  "ground-mounted",
        "p_tech_pri":  "PV",
        "p_tech_sec":  "unknown",
        # Mark as EIA-sourced for downstream awareness
        "source":      "EIA-860M",
        "eia860m_label": label,
    })

    # Top-line summary
    print(f"\n  EIA-860M supplement: {len(out):,} new plants, "
          f"{out['p_cap_ac'].sum():,.0f} MW AC")
    if len(out):
        print(f"    Top-5 new states by MW:")
        top = (out.groupby("p_state")["p_cap_ac"].sum()
                  .sort_values(ascending=False).head(5))
        for state, mw in top.items():
            print(f"      {state}: {mw:,.0f} MW")

    return out


if __name__ == "__main__":
    print("=" * 70)
    print("EIA-860M supplement loader")
    print("=" * 70)
    # When run standalone, dump all EIA-860M plants (no USPVDB filter)
    df = load_eia860m_supplement(uspvdb_eia_ids=set())
    print()
    if not df.empty:
        print("Sample (5 random plants):")
        print(df.sample(min(5, len(df)), random_state=42)[
            ["case_id", "p_name", "p_state", "p_cap_ac", "p_year"]
        ].to_string(index=False))
