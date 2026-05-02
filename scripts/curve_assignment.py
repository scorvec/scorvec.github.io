"""
Curve assignment for the USWTDB fleet.

For each turbine row, decide which power curve to use, and record the
provenance so the assignment is auditable.

Resolution order (highest priority first):
  1. User-registered override         → `manual:<key>`
  2. oedb / windpowerlib catalog match → `oedb:<turbine_type>`
  3. NREL generic by specific power    → `nrel_generic:<bin_key>`

Each turbine row needs at minimum: t_manu, t_model, t_cap (kW), t_rd (m).
If t_rd is missing we estimate it from t_cap assuming a typical specific
power of 350 W/m² (reasonable for the modern US fleet) and flag the
assignment with a `_rotor_estimated` marker in the source string.

Usage
-----
    from turbine_inventory import load_uswtdb
    from curve_assignment import assign_curves

    inv = load_uswtdb("uswtdb.csv")
    inv = assign_curves(inv, use_oedb=True)
    # inv now has columns: curve_source, curve_obj
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import power_curves as pc
from power_curves import PowerCurve, select_nrel_generic

log = logging.getLogger(__name__)


def _estimate_rotor_diameter(rated_kW: float,
                             assumed_sp_W_m2: float = 350.0) -> float:
    """Back out rotor diameter assuming a typical specific power."""
    if not np.isfinite(rated_kW) or rated_kW <= 0:
        return np.nan
    area = 1000.0 * rated_kW / assumed_sp_W_m2
    return float(2.0 * np.sqrt(area / np.pi))


def _assign_one(manu: str, model: str, rated_kW: float,
                rotor_d: float, oedb_catalog) -> tuple[PowerCurve, str]:
    """Return (curve, source_string) for a single turbine."""
    # Tier 0: user override
    from oedb_catalog import normalize, MANUFACTURER_ALIASES
    mfr_n = MANUFACTURER_ALIASES.get(normalize(manu), normalize(manu))
    mod_n = normalize(model)
    user = pc.lookup_user(mfr_n, mod_n)
    if user is not None:
        return user, f"manual:{mfr_n}|{mod_n}"

    # Tier 1: oedb catalog
    if oedb_catalog is not None:
        cat = oedb_catalog.lookup(manu, model)
        if cat is not None:
            # Use USWTDB rotor diameter if catalog's is missing/zero
            if (not np.isfinite(cat.rotor_diameter_m)
                    or cat.rotor_diameter_m <= 0):
                cat = PowerCurve(
                    name=cat.name, rated_kW=cat.rated_kW,
                    rotor_diameter_m=float(rotor_d),
                    wind_speeds_m_s=cat.wind_speeds_m_s,
                    power_kW=cat.power_kW,
                    cut_in=cat.cut_in, cut_out=cat.cut_out,
                    iec_class=cat.iec_class, source=cat.source,
                )
            return cat, cat.source

    # Tier 2: NREL generic
    if not np.isfinite(rotor_d) or rotor_d <= 0:
        rotor_d = _estimate_rotor_diameter(rated_kW)
        rotor_estimated = True
    else:
        rotor_estimated = False

    if not np.isfinite(rotor_d) or rotor_d <= 0:
        # Last-ditch: assume the median modern rotor for the nameplate.
        rotor_d = 100.0

    key, curve = select_nrel_generic(rated_kW=rated_kW,
                                     rotor_diameter_m=rotor_d)
    suffix = "_rotor_estimated" if rotor_estimated else ""
    return curve, f"nrel_generic:{key}{suffix}"


def assign_curves(inventory: pd.DataFrame,
                  use_oedb: bool = True) -> pd.DataFrame:
    """Add `curve_source` and `curve_obj` columns to a turbine inventory.

    Expected input columns (USWTDB native names, after `turbine_inventory`
    has loaded them): t_manu, t_model, t_cap, t_rd.
    """
    oedb = None
    if use_oedb:
        try:
            from oedb_catalog import OEDBCatalog
            oedb = OEDBCatalog.load()
            log.info("Loaded oedb catalog with %d turbine types",
                     len(oedb.types_df))
        except Exception as e:
            log.warning("oedb catalog unavailable (%s); "
                        "falling back to NREL generic only", e)

    out = inventory.copy()
    sources: list[str] = []
    curves: list[PowerCurve] = []

    for _, row in out.iterrows():
        curve, src = _assign_one(
            manu=row.get("t_manu", ""),
            model=row.get("t_model", ""),
            rated_kW=float(row.get("t_cap", np.nan)),
            rotor_d=float(row.get("t_rd", np.nan)),
            oedb_catalog=oedb,
        )
        sources.append(src)
        curves.append(curve)

    out["curve_source"] = sources
    out["curve_obj"] = curves
    return out


def assignment_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Tabulate how many turbines / MW landed in each source bucket."""
    if "curve_source" not in inventory.columns:
        raise ValueError("Run assign_curves() first")

    g = inventory.groupby("curve_source").agg(
        n_turbines=("curve_source", "size"),
        total_MW=("t_cap", lambda s: s.sum() / 1000.0),
    ).sort_values("total_MW", ascending=False)
    g["pct_MW"] = 100.0 * g["total_MW"] / g["total_MW"].sum()
    return g
