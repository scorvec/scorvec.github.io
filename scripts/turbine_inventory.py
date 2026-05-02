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


def load_uswtdb(path: str | Path) -> pd.DataFrame:
    """Load USWTDB CSV (or shapefile-derived CSV) into a DataFrame.

    Coerces numerics, drops rows with no nameplate or no coordinates
    (those are unusable for forecasting). Computes `specific_power_W_m2`.
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
    log.info("Loaded USWTDB: %d → %d turbines after dropping invalid rows",
             n0, len(df))

    # Specific power
    rotor_area = np.pi * (df["t_rd"].astype(float) / 2.0) ** 2
    df["specific_power_W_m2"] = 1000.0 * df["t_cap"] / rotor_area
    df.loc[~np.isfinite(df["specific_power_W_m2"]),
           "specific_power_W_m2"] = np.nan

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
