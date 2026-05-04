"""
USWTDB inventory loader.

Source: https://eerscmap.usgs.gov/uswtdb/  (US Wind Turbine Database, USGS)
Native columns we care about:

    case_id    int     unique turbine identifier
    eia_id     int     EIA plant id (links to EIA-860 / 923)
    t_state    str     state abbreviation
    p_name     str     project / wind plant name
    t_manu     str     turbine manufacturer
    t_model    str     turbine model
    t_cap      int     turbine nameplate, kW
    t_hh       float   hub height, m
    t_rd       float   rotor diameter, m
    t_rsa      float   rotor swept area, m^2 (sometimes pre-computed)
    xlong      float   longitude
    ylat       float   latitude
    p_year     int     project commissioning year
    p_tnum     int     number of turbines in the project

This loader does *not* try to repair missing fields — it surfaces them so
downstream code can decide. The only non-trivial transform is computing
specific power when t_rsa or t_rd is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLS = [
    "case_id", "eia_id", "t_state", "p_name", "t_manu", "t_model",
    "t_cap", "t_hh", "t_rd", "xlong", "ylat",
]


# Names that match non-grid, demonstration, or educational installations.
# These appear in USWTDB but aren't ERCOT/SPP dispatched generation, so they
# should be excluded from forecasting. Patterns are matched as case-insensitive
# substrings against p_name.
NON_UTILITY_PATTERNS = (
    "museum",                      # American Windmill Museum, Sustainable Tech Museum
    "stadium",                     # Apogee Stadium Wind
    "test facility",               # UL Advanced Wind Turbine Test Facility
    "advanced wind turbine test",  # variants of the above
    "noresco",                     # NORESCO behind-the-meter installs
    "wtamu",                       # West Texas A&M
    "texas tech",                  # Texas Tech research turbines
    "special utility district",    # Mountain Peak SUD
)


def _flag_non_utility(p_name: pd.Series) -> pd.Series:
    """Return a boolean mask of rows whose p_name matches a non-utility pattern."""
    s = p_name.astype(str).str.lower()
    mask = pd.Series(False, index=p_name.index)
    for pat in NON_UTILITY_PATTERNS:
        mask |= s.str.contains(pat, na=False, regex=False)
    return mask


def _backfill_orphan_eia_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach orphan turbines to their parent plant by name.

    USWTDB sometimes has a few turbines at an existing plant without an
    eia_id while the rest of the plant's turbines have one. If exactly one
    eia_id is associated with a (p_name, t_state) pair across the dataset,
    inherit it for the orphans. Plants where every turbine is an orphan
    (e.g. brand-new 2025 plants not yet in EIA-860) are left alone — they
    still get correct BA routing via state fallback downstream.
    """
    df = df.copy()
    has_id = df["eia_id"].notna()
    # For each (p_name, t_state) bucket, count distinct eia_ids among the
    # cataloged turbines.
    cataloged = df[has_id]
    parent_map = (
        cataloged.groupby(["p_name", "t_state"])["eia_id"]
        .nunique()
        .reset_index(name="n_distinct")
    )
    # Only inherit from buckets with exactly one eia_id (avoid ambiguity).
    unambiguous = parent_map[parent_map["n_distinct"] == 1][["p_name", "t_state"]]
    eia_lookup = (
        cataloged.merge(unambiguous, on=["p_name", "t_state"])
        .drop_duplicates(["p_name", "t_state"])
        .set_index(["p_name", "t_state"])["eia_id"]
    )

    orphan_mask = ~has_id
    if orphan_mask.any() and not eia_lookup.empty:
        keys = list(zip(df.loc[orphan_mask, "p_name"],
                        df.loc[orphan_mask, "t_state"]))
        inherited = pd.Series(
            [eia_lookup.get(k, np.nan) for k in keys],
            index=df.index[orphan_mask],
        )
        n_filled = inherited.notna().sum()
        if n_filled > 0:
            df.loc[inherited.notna().reindex(df.index, fill_value=False),
                   "eia_id"] = inherited
            log.info("Backfilled eia_id for %d orphan turbine row(s) "
                     "via name match to parent plant", int(n_filled))
    return df


def load_uswtdb(path: str | Path) -> pd.DataFrame:
    """Load USWTDB CSV (or shapefile-derived CSV) into a DataFrame.

    Coerces numerics, drops rows with no nameplate or no coordinates
    (those are unusable for forecasting), excludes non-grid generators
    (museums, university test sites, stadiums), and backfills missing
    eia_ids for orphan turbines whose parent plant is unambiguously
    identifiable by name. Computes `specific_power_W_m2`.
    """
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"USWTDB missing expected columns: {missing}")

    # Numeric coercion (USWTDB occasionally has -99999 sentinel values).
    # Negative-as-sentinel only applies to fields that can't legitimately
    # be negative; longitudes are negative in the western hemisphere.
    nonneg_cols = ("t_cap", "t_hh", "t_rd", "p_tnum", "p_year")
    coords_cols = ("xlong", "ylat")
    for c in nonneg_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
    for c in coords_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Only the explicit -99999 sentinel for coordinates
            df.loc[df[c] <= -999, c] = np.nan

    n0 = len(df)
    df = df.dropna(subset=["t_cap", "xlong", "ylat"])
    df = df[df["t_cap"] > 0]

    # Drop non-utility installations (museums, university test sites, etc.)
    nu_mask = _flag_non_utility(df["p_name"])
    if nu_mask.any():
        nu_mw = df.loc[nu_mask, "t_cap"].sum() / 1000.0
        log.info("Dropped %d non-utility turbine row(s) (%.2f MW): %s",
                 int(nu_mask.sum()), nu_mw,
                 sorted(df.loc[nu_mask, "p_name"].unique().tolist()))
        df = df[~nu_mask].copy()

    # Backfill orphan eia_ids via parent plant name
    df = _backfill_orphan_eia_ids(df)

    log.info("Loaded USWTDB: %d → %d turbines after filtering",
             n0, len(df))

    # Specific power
    rotor_area = np.pi * (df["t_rd"].astype(float) / 2.0) ** 2
    df["specific_power_W_m2"] = 1000.0 * df["t_cap"] / rotor_area
    df.loc[~np.isfinite(df["specific_power_W_m2"]),
           "specific_power_W_m2"] = np.nan

    # Merge in synthetic plants if they exist alongside the USWTDB file.
    # These are real operating plants that USWTDB hasn't yet attributed
    # (e.g. brand-new projects whose turbines USGS detected via aerial
    # imagery but hasn't yet linked to a permitting record). Format
    # matches USWTDB columns; see data/synthetic_plants.csv.
    syn_path = Path(path).parent / "synthetic_plants.csv"
    if syn_path.exists():
        try:
            syn = pd.read_csv(syn_path, low_memory=False)
            # Coerce numerics same way as the main USWTDB load
            for c in ("t_cap", "t_hh", "t_rd", "p_tnum", "p_year",
                      "xlong", "ylat", "eia_id"):
                if c in syn.columns:
                    syn[c] = pd.to_numeric(syn[c], errors="coerce")
            # Filter same way (must have nameplate and coords)
            syn = syn.dropna(subset=["t_cap", "xlong", "ylat"])
            syn = syn[syn["t_cap"] > 0]
            if not syn.empty:
                # Compute specific power for synthetic rows too
                syn_area = np.pi * (syn["t_rd"].astype(float) / 2.0) ** 2
                syn["specific_power_W_m2"] = 1000.0 * syn["t_cap"] / syn_area
                # Append, keeping all columns aligned. Missing columns
                # from either side become NaN, which is fine — the
                # downstream code tolerates it.
                df = pd.concat([df, syn], ignore_index=True)
                log.info("Merged %d synthetic turbine row(s) (%.0f MW) "
                         "from %s",
                         len(syn), syn["t_cap"].sum() / 1000.0, syn_path.name)
        except Exception as e:
            log.warning("Failed to load synthetic plants from %s: %s",
                        syn_path, e)

    return df.reset_index(drop=True)


def fleet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state summary: turbine count and total MW."""
    return (df.groupby("t_state")
              .agg(n_turbines=("case_id", "count"),
                   total_MW=("t_cap", lambda s: s.sum() / 1000.0))
              .sort_values("total_MW", ascending=False))


def manufacturer_breakdown(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top manufacturer/model combos by installed MW. Useful for QA."""
    g = (df.groupby(["t_manu", "t_model"])
           .agg(n=("case_id", "count"),
                MW=("t_cap", lambda s: s.sum() / 1000.0))
           .sort_values("MW", ascending=False))
    return g.head(top_n)
