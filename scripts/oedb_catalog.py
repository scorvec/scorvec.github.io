"""
Wrapper around the windpowerlib / oedb power curve catalog.

windpowerlib (https://github.com/wind-python/windpowerlib) ships a CSV of
curated power curves. Each curve is identified by a `turbine_type` string
like "V90/2000" or "E-126/4200" — typically `{rotor_diameter}/{rated_kW}`
prefixed by an OEM-specific letter code.

USWTDB stores manufacturer and model in separate columns (`t_manu`,
`t_model`) and the model strings differ from oedb conventions. This
module provides:

  - `OEDBCatalog`: wraps `windpowerlib.get_turbine_types()` into a lookup
    dataframe keyed on normalized (manufacturer, model) pairs.
  - `normalize()`: lowercase, strip punctuation, collapse whitespace.
  - `MANUFACTURER_ALIASES` and `MODEL_ALIASES`: hand-maintained synonym
    tables (USWTDB string → oedb canonical), covering the dozens of
    real-world naming differences. Extend as you find more.

Usage
-----
    cat = OEDBCatalog.load()
    pc = cat.lookup("GE Wind", "1.5SLE")      # → windpowerlib WindTurbine
    pc = cat.lookup("Siemens Gamesa", "SG 4.5-145")
    pc = cat.lookup("Vestas", "V90-1.8")      # tries aliases internally
    if pc is None:
        # No catalog match — caller falls back to NREL generic.
        ...

The first call to `OEDBCatalog.load()` triggers a download/parse of the
windpowerlib catalog (it is cached on disk by windpowerlib). After that
calls are in-memory.

This file does NOT hardcode curve numbers; the oedb is the source of truth.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from power_curves import PowerCurve

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize(s: str | float | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    `Vestas V90-1.8`   → `vestas v90 1 8`
    `GE 1.5sle`        → `ge 1 5sle`
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# Manufacturer aliases: USWTDB `t_manu` strings (left) → canonical key.
# The canonical key is also lowercase / punctuation-stripped (to match
# normalize()'s output).
MANUFACTURER_ALIASES: dict[str, str] = {
    "ge wind": "ge",
    "general electric": "ge",
    "ge energy": "ge",
    "ge renewable energy": "ge",
    "siemens gamesa renewable energy": "siemens gamesa",
    "siemens gamesa": "siemens gamesa",
    "sgre": "siemens gamesa",
    "siemens": "siemens",
    "siemens wind power": "siemens",
    "vestas wind systems": "vestas",
    "vestas": "vestas",
    "gamesa eolica": "gamesa",
    "gamesa": "gamesa",
    "mitsubishi heavy industries": "mitsubishi",
    "mitsubishi": "mitsubishi",
    "nordex acciona": "nordex",
    "nordex usa": "nordex",
    "nordex": "nordex",
    "acciona windpower": "acciona",
    "acciona": "acciona",
    "clipper windpower": "clipper",
    "clipper": "clipper",
    "suzlon energy": "suzlon",
    "suzlon": "suzlon",
    "goldwind americas": "goldwind",
    "goldwind": "goldwind",
    "enercon": "enercon",
    "repower": "senvion",
    "senvion": "senvion",
    "nordtank": "nordtank",
    "nordtank energy": "nordtank",
    "neg micon": "neg micon",
    "mitsubishi power systems": "mitsubishi",
}


# Model aliases. Keys are normalize(t_manu) + "|" + normalize(t_model);
# values are the windpowerlib catalog `turbine_type` strings.
# This table is the place where you encode "the USWTDB says X, oedb calls
# it Y". Keep it small and grow it from real lookup misses.
MODEL_ALIASES: dict[str, str] = {
    # GE
    "ge|1 5 sle":         "GE 1.5sle",
    "ge|1 5sle":          "GE 1.5sle",
    "ge|1 5 s":           "GE 1.5s",
    "ge|1 5s":            "GE 1.5s",
    "ge|1 5 xle":         "GE 1.5xle",
    "ge|1 5xle":          "GE 1.5xle",
    "ge|1 6 100":         "GE 1.6-100",
    "ge|1 7 100":         "GE 1.7-100",
    "ge|1 7 103":         "GE 1.7-103",
    "ge|2 5 100":         "GE 2.5-100",
    "ge|2 75 100":        "GE 2.75-100",
    "ge|2 75 103":        "GE 2.75-103",
    "ge|2 5 127":         "GE 2.5-127",
    # Vestas
    "vestas|v47":         "V47/660",
    "vestas|v47 660":     "V47/660",
    "vestas|v80 1 8":     "V80/1800",
    "vestas|v82 1 65":    "V82/1650",
    "vestas|v90 1 8":     "V90/1800",
    "vestas|v90 3 0":     "V90/3000",
    "vestas|v100 1 8":    "V100/1800",
    "vestas|v100 2 0":    "V100/2000",
    "vestas|v110 2 0":    "V110/2000",
    "vestas|v112 3 0":    "V112/3000",
    "vestas|v117 3 45":   "V117/3450",
    "vestas|v126 3 45":   "V126/3450",
    "vestas|v136 3 45":   "V136/3450",
    "vestas|v150 4 5":    "V150/4500",
    # Siemens / SGRE
    "siemens|swt 2 3 93":   "SWT-2.3-93",
    "siemens|swt 2 3 101":  "SWT-2.3-101",
    "siemens|swt 2 3 108":  "SWT-2.3-108",
    "siemens|swt 2 3 113":  "SWT-2.3-113",
    "siemens|swt 3 2 113":  "SWT-3.2-113",
    "siemens gamesa|sg 4 5 145": "SG 4.5-145",
    # Gamesa
    "gamesa|g87 2 0":     "G87/2000",
    "gamesa|g97 2 0":     "G97/2000",
    "gamesa|g114 2 0":    "G114/2000",
    "gamesa|g126 2 5":    "G126/2500",
}


# ---------------------------------------------------------------------------
# Catalog wrapper
# ---------------------------------------------------------------------------

@dataclass
class OEDBCatalog:
    """In-memory view of the windpowerlib power curve catalog."""

    types_df: pd.DataFrame  # one row per turbine_type with manufacturer

    @classmethod
    def load(cls) -> "OEDBCatalog":
        """Load via windpowerlib. Network/disk hit on first call."""
        try:
            from windpowerlib import get_turbine_types  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "windpowerlib is required for Tier-1 catalog matching. "
                "Install with: pip install windpowerlib"
            ) from e

        df = get_turbine_types(print_out=False)
        # Columns are typically: turbine_type, manufacturer, has_power_curve, has_cp_curve
        df = df.copy()
        df["mfr_norm"] = df["manufacturer"].map(normalize)
        df["model_norm"] = df["turbine_type"].map(normalize)
        return cls(types_df=df)

    def lookup(self, manufacturer: str, model: str) -> Optional[PowerCurve]:
        """Return a PowerCurve, or None if no match.

        Resolution order:
          1. Direct (mfr_alias, model_alias) lookup in MODEL_ALIASES.
          2. Normalized (manufacturer, model) prefix match in oedb.
          3. None — caller falls back to NREL generic.
        """
        mfr_n = MANUFACTURER_ALIASES.get(normalize(manufacturer),
                                         normalize(manufacturer))
        mod_n = normalize(model)
        if not mfr_n or not mod_n:
            return None

        alias_key = f"{mfr_n}|{mod_n}"
        canonical = MODEL_ALIASES.get(alias_key)

        if canonical is not None:
            row = self.types_df[
                self.types_df["turbine_type"].str.lower() == canonical.lower()
            ]
            if not row.empty:
                return self._row_to_curve(row.iloc[0])

        # Fallback: try normalized prefix match within manufacturer
        candidates = self.types_df[self.types_df["mfr_norm"] == mfr_n]
        if not candidates.empty:
            # Exact normalized model match
            exact = candidates[candidates["model_norm"] == mod_n]
            if not exact.empty:
                return self._row_to_curve(exact.iloc[0])
            # Substring match on the model token, both directions
            for _, row in candidates.iterrows():
                rn = row["model_norm"]
                if rn and (rn in mod_n or mod_n in rn):
                    return self._row_to_curve(row)

        return None

    @staticmethod
    def _row_to_curve(row: pd.Series) -> Optional[PowerCurve]:
        """Build a PowerCurve from a windpowerlib catalog row."""
        try:
            from windpowerlib import WindTurbine  # noqa: WPS433
        except ImportError:
            return None

        # windpowerlib needs hub_height & rotor_diameter; we want the curve
        # itself, which is independent of hub height. Pick the catalog
        # rotor_diameter if present, else 0 (we'll override at use time).
        wt = WindTurbine(
            turbine_type=row["turbine_type"],
            hub_height=80,  # placeholder
        )
        if wt.power_curve is None or wt.power_curve.empty:
            return None

        ws = wt.power_curve["wind_speed"].to_numpy(dtype=float)
        # windpowerlib stores power in W; convert to kW
        pw = wt.power_curve["value"].to_numpy(dtype=float) / 1000.0
        rated_kW = float(pw.max())
        # Rotor diameter: from WindTurbine attribute or NaN
        rotor_d = float(getattr(wt, "rotor_diameter", np.nan) or np.nan)
        if not np.isfinite(rotor_d):
            rotor_d = 0.0  # caller should override from USWTDB t_rd

        # Cut-in / cut-out from curve shape: first nonzero, last nonzero+1
        nonzero = np.flatnonzero(pw > 0)
        cut_in = float(ws[nonzero[0]]) if len(nonzero) else 3.0
        cut_out = float(ws[nonzero[-1]] + (ws[1] - ws[0])) if len(nonzero) else 25.0

        return PowerCurve(
            name=row["turbine_type"],
            rated_kW=rated_kW,
            rotor_diameter_m=rotor_d,
            wind_speeds_m_s=ws,
            power_kW=pw,
            cut_in=cut_in,
            cut_out=cut_out,
            iec_class="?",  # oedb does not always carry this
            source=f"oedb:{row['turbine_type']}",
        )
