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
    real-world naming differences.
  - Fuzzy matching: parses USWTDB model strings into (rotor_d, rated_kW)
    tuples and queries oedb by geometry when string-based matching fails.

Usage
-----
    cat = OEDBCatalog.load()
    pc = cat.lookup("GE Wind", "1.5SLE")
    pc = cat.lookup("Siemens Gamesa", "SG 4.5-145")
    pc = cat.lookup("Vestas", "V90-1.8")
    if pc is None:
        # No catalog match — caller falls back to NREL generic.
        ...

The first call to `OEDBCatalog.load()` triggers a download/parse of the
windpowerlib catalog (it is cached on disk by windpowerlib). After that
calls are in-memory.
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


# Manufacturer-prefix patterns that USGS sometimes embeds into the model
# column itself. e.g. t_manu="GE Wind", t_model="GE2.82-127" — the "GE" gets
# concatenated onto the model string. Stripping the prefix lets the alias
# table and geometry parser work on the real model identifier.
_MFR_PREFIX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ge",             re.compile(r"^ge\s*")),
    ("vestas",         re.compile(r"^vestas\s*")),
    ("siemens gamesa", re.compile(r"^sgre?\s*|^siemens gamesa\s*")),
    ("siemens",        re.compile(r"^siemens\s*|^swt\s*", re.IGNORECASE)),
    ("nordex",         re.compile(r"^nordex\s*|^n(?=\d)")),  # 'N131' starts with N
    ("gamesa",         re.compile(r"^gamesa\s*")),
    ("acciona",        re.compile(r"^acciona\s*|^aw\s*")),
    ("suzlon",         re.compile(r"^suzlon\s*|^s(?=\d)")),
    ("mitsubishi",     re.compile(r"^mitsubishi\s*|^mwt\s*")),
    ("clipper",        re.compile(r"^clipper\s*|^c(?=\d)")),
    ("goldwind",       re.compile(r"^goldwind\s*|^gw\s*")),
    ("senvion",        re.compile(r"^senvion\s*|^repower\s*")),
]


def strip_mfr_prefix(mfr_n: str, model_n: str) -> str:
    """Remove a manufacturer prefix that USGS concatenated into the model.

    `ge`         + `ge2 82 127`  → `2 82 127`
    `ge`         + `ge1 5 77`    → `1 5 77`
    `siemens`    + `swt 2 3 93`  → preserved (SWT is a real product line)
    `vestas`     + `v117 3 45`   → preserved (V is part of the model name)

    Only strips redundant repetitions of the manufacturer name; preserves
    canonical letter codes like V, SG, N, MWT that are part of the actual
    model identifier.
    """
    s = model_n.strip()
    # Only strip exact "ge" / "vestas" / "siemens gamesa" / etc — not
    # single-letter codes that are part of legitimate model names.
    redundant_strips = {
        "ge":              r"^ge\s+|^ge(?=\d)",
        "vestas":          r"^vestas\s+",
        "siemens gamesa":  r"^siemens gamesa\s+|^sgre\s+",
        "siemens":         r"^siemens\s+",
        "nordex":          r"^nordex\s+",
        "gamesa":          r"^gamesa\s+",
        "acciona":         r"^acciona\s+",
        "suzlon":          r"^suzlon\s+",
        "mitsubishi":      r"^mitsubishi\s+",
        "clipper":         r"^clipper\s+",
        "goldwind":        r"^goldwind\s+",
        "senvion":         r"^senvion\s+|^repower\s+",
    }
    if mfr_n in redundant_strips:
        s = re.sub(redundant_strips[mfr_n], "", s).strip()
    return s


# Manufacturer aliases: USWTDB `t_manu` strings (left) → canonical key.
MANUFACTURER_ALIASES: dict[str, str] = {
    "ge wind": "ge",
    "general electric": "ge",
    "ge energy": "ge",
    "ge renewable energy": "ge",
    "ge vernova": "ge",
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
MODEL_ALIASES: dict[str, str] = {
    # GE — variations on 1.5 MW family
    "ge|1 5 sle":         "GE 1.5sle",
    "ge|1 5sle":          "GE 1.5sle",
    "ge|1 5 s":           "GE 1.5s",
    "ge|1 5s":            "GE 1.5s",
    "ge|1 5 xle":         "GE 1.5xle",
    "ge|1 5xle":          "GE 1.5xle",
    "ge|1 5 sl":          "GE 1.5sl",
    "ge|1 5sl":           "GE 1.5sl",
    "ge|1 5 77":          "GE 1.5sle",
    "ge|1 5 82":          "GE 1.5sle",
    "ge|1 5 87":          "GE 1.5xle",
    "ge|sle":             "GE 1.5sle",
    "ge|xle":             "GE 1.5xle",
    # GE — newer USGS string format with embedded "GE" prefix
    # The strip_mfr_prefix() pass also rewrites these to "X Y Z" form,
    # but we keep both forms for robustness.
    "ge|ge1 5 77":        "GE 1.5sle",
    "ge|ge1 5 82":        "GE 1.5sle",
    "ge|ge1 5 87":        "GE 1.5xle",
    "ge|ge1 5sle":        "GE 1.5sle",
    "ge|ge1 5xle":        "GE 1.5xle",
    "ge|ge1 6 91":        "GE 1.6-100",       # 91m is GE 1.6 short-rotor
    "ge|ge1 6 100":       "GE 1.6-100",
    "ge|ge1 7 100":       "GE 1.7-100",
    "ge|ge1 7 103":       "GE 1.7-103",
    "ge|ge1 79 100":      "GE 1.7-100",       # 1.79 MW = uprated 1.7
    "ge|ge1 85 87":       "GE 1.85-87",
    "ge|ge2 3 116":       "GE 2.3-116",
    "ge|ge2 3 107":       "GE 2.3-107",
    "ge|ge2 3 127":       "GE 2.3-127",
    "ge|ge2 4 107":       "GE 2.4-107",
    "ge|ge2 5 100":       "GE 2.5-100",
    "ge|ge2 5 116":       "GE 2.5-116",
    "ge|ge2 5 127":       "GE 2.5-127",
    "ge|ge2 75 100":      "GE 2.75-100",
    "ge|ge2 75 103":      "GE 2.75-103",
    "ge|ge2 8 127":       "GE 2.8-127",
    "ge|ge2 82 127":      "GE 2.8-127",       # 2.82 MW uprate of 2.8-127
    "ge|ge3 0 130":       "GE 3.0-130",
    "ge|ge3 0 137":       "GE 3.0-137",
    "ge|ge3 4 137":       "GE 3.4-137",
    "ge|ge3 4 140":       "GE 3.4-137",       # 140m close enough to 137
    "ge|ge3 6 137":       "GE 3.6-137",
    "ge|ge3 8 137":       "GE 3.8-137",
    "ge|ge5 3 158":       "GE 5.3-158",
    # GE 1.6/1.7/2.x families (older USGS format without prefix)
    "ge|1 6 91":          "GE 1.6-100",
    "ge|1 6 100":         "GE 1.6-100",
    "ge|1 7 100":         "GE 1.7-100",
    "ge|1 7 103":         "GE 1.7-103",
    "ge|1 79 100":        "GE 1.7-100",
    "ge|1 85 87":         "GE 1.85-87",
    "ge|2 3 116":         "GE 2.3-116",
    "ge|2 3 107":         "GE 2.3-107",
    "ge|2 3 127":         "GE 2.3-127",
    "ge|2 4 107":         "GE 2.4-107",
    "ge|2 5 100":         "GE 2.5-100",
    "ge|2 5 116":         "GE 2.5-116",
    "ge|2 5 127":         "GE 2.5-127",
    "ge|2 75 100":        "GE 2.75-100",
    "ge|2 75 103":        "GE 2.75-103",
    "ge|2 8 127":         "GE 2.8-127",
    "ge|2 82 127":        "GE 2.8-127",
    "ge|3 0 130":         "GE 3.0-130",
    "ge|3 0 137":         "GE 3.0-137",
    "ge|3 4 137":         "GE 3.4-137",
    "ge|3 4 140":         "GE 3.4-137",
    "ge|3 6 137":         "GE 3.6-137",
    "ge|3 8 137":         "GE 3.8-137",
    "ge|cypress 5 3 158": "GE 5.3-158",
    # Vestas — V47 to V90
    "vestas|v47":         "V47/660",
    "vestas|v47 660":     "V47/660",
    "vestas|v52":         "V52/850",
    "vestas|v52 850":     "V52/850",
    "vestas|v66":         "V66/1750",
    "vestas|v66 1 65":    "V66/1750",
    "vestas|v80":         "V80/2000",
    "vestas|v80 1 8":     "V80/1800",
    "vestas|v80 2 0":     "V80/2000",
    "vestas|v82":         "V82/1650",
    "vestas|v82 1 65":    "V82/1650",
    "vestas|v90":         "V90/2000",
    "vestas|v90 1 8":     "V90/1800",
    "vestas|v90 2 0":     "V90/2000",
    "vestas|v90 3 0":     "V90/3000",
    # Vestas — V100 to V126
    "vestas|v100":        "V100/2000",
    "vestas|v100 1 8":    "V100/1800",
    "vestas|v100 2 0":    "V100/2000",
    "vestas|v110":        "V110/2000",
    "vestas|v110 2 0":    "V110/2000",
    "vestas|v110 2 2":    "V110/2200",
    "vestas|v112":        "V112/3000",
    "vestas|v112 3 0":    "V112/3000",
    "vestas|v112 3 3":    "V112/3300",
    "vestas|v117":        "V117/3450",
    "vestas|v117 3 3":    "V117/3300",
    "vestas|v117 3 45":   "V117/3450",
    "vestas|v117 4 2":    "V117/4200",
    "vestas|v117 4 3":    "V117/4300",
    # V120-2.2 has SP ≈ 195 W/m² (rotor 120m, 2.2 MW) — not in oedb.
    # The 12-bin generic for that SP range (low_sp_modern) is the best fit.
    "vestas|v126":        "V126/3450",
    "vestas|v126 3 3":    "V126/3300",
    "vestas|v126 3 45":   "V126/3450",
    "vestas|v126 3 6":    "V126/3600",
    # Vestas — V136 and up (modern fleet, big in ERCOT)
    "vestas|v136":        "V136/3450",
    "vestas|v136 3 45":   "V136/3450",
    "vestas|v136 3 6":    "V136/3600",
    "vestas|v136 4 0":    "V136/4000",
    "vestas|v136 4 2":    "V136/4200",
    "vestas|v150":        "V150/4200",
    "vestas|v150 4 0":    "V150/4000",
    "vestas|v150 4 2":    "V150/4200",
    "vestas|v150 4 5":    "V150/4500",
    "vestas|v150 5 6":    "V150/5600",
    "vestas|v155":        "V155/3300",
    "vestas|v162":        "V162/5600",
    "vestas|v162 5 6":    "V162/5600",
    "vestas|v162 6 0":    "V162/6000",
    "vestas|v162 6 2":    "V162/6200",
    # Siemens — SWT classics
    "siemens|swt 2 3 82":   "SWT-2.3-82",
    "siemens|swt 2 3 93":   "SWT-2.3-93",
    "siemens|swt 2 3 101":  "SWT-2.3-101",
    "siemens|swt 2 3 108":  "SWT-2.3-108",
    "siemens|swt 2 3 113":  "SWT-2.3-113",
    "siemens|swt 3 0 101":  "SWT-3.0-101",
    "siemens|swt 3 0 108":  "SWT-3.0-108",
    "siemens|swt 3 2 101":  "SWT-3.2-101",
    "siemens|swt 3 2 113":  "SWT-3.2-113",
    "siemens|swt dd 130":   "SWT-3.3-130",
    "siemens|swt dd 142":   "SWT-3.6-142",
    # Siemens Gamesa — SG family (post-2017 merger)
    "siemens gamesa|sg 2 1 114": "SG 2.1-114",
    "siemens gamesa|sg 2 6 114": "SG 2.6-114",
    "siemens gamesa|sg 2 7 129": "SG 2.7-129",
    "siemens gamesa|sg 3 4 132": "SG 3.4-132",
    "siemens gamesa|sg 4 5 145": "SG 4.5-145",
    "siemens gamesa|sg 5 0 145": "SG 5.0-145",
    "siemens gamesa|sg 5 8 155": "SG 5.8-155",
    "siemens gamesa|sg 5 8 170": "SG 5.8-170",
    "siemens gamesa|sg 6 0 170": "SG 6.0-170",
    # Gamesa — pre-merger
    "gamesa|g52":         "G52/850",
    "gamesa|g80":         "G80/2000",
    "gamesa|g83":         "G80/2000",
    "gamesa|g87":         "G87/2000",
    "gamesa|g87 2 0":     "G87/2000",
    "gamesa|g90":         "G90/2000",
    "gamesa|g90 2 0":     "G90/2000",
    "gamesa|g97":         "G97/2000",
    "gamesa|g97 2 0":     "G97/2000",
    "gamesa|g114":        "G114/2000",
    "gamesa|g114 2 0":    "G114/2000",
    "gamesa|g114 2 1":    "G114/2100",
    "gamesa|g126":        "G126/2500",
    "gamesa|g126 2 5":    "G126/2500",
    "gamesa|g132":        "G132/3300",
    "gamesa|g132 3 3":    "G132/3300",
    # Mitsubishi
    "mitsubishi|mwt 1000a":  "MWT 1000",
    "mitsubishi|mwt 95 2 4": "MWT-95/2.4",
    "mitsubishi|mwt 102 2 4":"MWT-102/2.4",
    # Nordex / Acciona
    "nordex|n100":        "N100/2500",
    "nordex|n100 2 5":    "N100/2500",
    "nordex|n117 2 4":    "N117/2400",
    "nordex|n117 3 0":    "N117/3000",
    "nordex|n131 3 0":    "N131/3000",
    "nordex|n131 3 6":    "N131/3600",
    "nordex|n149 4 0":    "N149/4000",
    "nordex|n149 4 5":    "N149/4500",
    "nordex|n155 4 5":    "N155/4500",
    "nordex|n163 5 7":    "N163/5700",
    "acciona|aw 70 1 5":  "AW 1500/70",
    "acciona|aw 77 1 5":  "AW 1500/77",
    "acciona|aw 82 1 5":  "AW 1500/82",
    "acciona|aw 100 3 0": "AW 3000/100",
    "acciona|aw 116 3 0": "AW 3000/116",
    "acciona|aw 125 3 0": "AW 3000/125",
    "acciona|aw 132 3 0": "AW 3000/132",
    # Suzlon
    "suzlon|s64":         "S64/1250",
    "suzlon|s88":         "S88/2100",
    "suzlon|s95":         "S95/2100",
    "suzlon|s97":         "S97/2100",
    # Goldwind
    "goldwind|gw 121 2 5": "GW 121/2500",
    "goldwind|gw 140 3 0": "GW 140/3000",
    "goldwind|gw 140 3 4": "GW 140/3400",
    "goldwind|gw 155 4 5": "GW 155/4500",
    # Clipper (legacy)
    "clipper|c89":        "C89/2500",
    "clipper|c93":        "C93/2500",
    "clipper|c96":        "C96/2500",
    "clipper|c99":        "C99/2500",
    # Senvion / Repower
    "senvion|mm82":       "MM82/2050",
    "senvion|mm92":       "MM92/2050",
    "senvion|3xm":        "3.4M104",
    "senvion|3 4m104":    "3.4M104",
}


# ---------------------------------------------------------------------------
# Model-string parsing (for fuzzy fallback)
# ---------------------------------------------------------------------------

# Common patterns in USWTDB t_model strings. We extract rotor diameter (m)
# and rated kW where possible, then match against oedb by geometry.
_MODEL_PARSERS: list[tuple[re.Pattern, str]] = [
    # "V117-3.45" / "V117 3 45" / "V126/3450" / "G114-2.0" / "N131-3.6"
    # Letter prefix + rotor + separator + rated_MW (decimal)
    (re.compile(r"^([a-z]+)\s*(\d{2,3})\s+(\d+)\s+(\d+)\s*$"),
     "letter_rotor_decimal_spaces"),
    (re.compile(r"^([a-z]+)\s*(\d{2,3})\s*[/\-]\s*(\d+)\s*[\.\,]\s*(\d+)\s*$"),
     "letter_rotor_decimal_punct"),
    # "GE 2.5-127" / "GE 1.5-77" / "SWT-2.3-93"
    # Optional letters + rated_MW (decimal) + separator + rotor
    (re.compile(r"^.*?(\d+)\s+(\d+)\s+(\d{2,3})\s*$"),
     "decimal_then_rotor_spaces"),
    (re.compile(r"^.*?(\d+)\s*[\.\,]\s*(\d+)\s*[/\-]\s*(\d{2,3})\s*$"),
     "decimal_then_rotor_punct"),
    # "V117/3300" — letter prefix + rotor + sep + rated_kW (integer ≥ 1000)
    (re.compile(r"^([a-z]+)\s*(\d{2,3})\s+(\d{4,5})\s*$"),
     "letter_rotor_kw_spaces"),
    (re.compile(r"^([a-z]+)\s*(\d{2,3})\s*[/\-]\s*(\d{4,5})\s*$"),
     "letter_rotor_kw_punct"),
]


def _parse_model_geometry(manu_norm: str, model_norm: str
                          ) -> tuple[Optional[float], Optional[float]]:
    """Try to extract (rotor_diameter_m, rated_kW) from a USWTDB model string.

    Returns (None, None) if the string doesn't match any known pattern.
    """
    s = model_norm.strip()
    for pat, kind in _MODEL_PARSERS:
        m = pat.match(s)
        if not m:
            continue
        try:
            if kind.startswith("letter_rotor_decimal"):
                # groups: (letters, rotor, mw_int, mw_frac)
                rotor = float(m.group(2))
                rated_mw = float(f"{m.group(3)}.{m.group(4)}")
                return rotor, rated_mw * 1000.0
            elif kind.startswith("decimal_then_rotor"):
                # groups: (mw_int, mw_frac, rotor)
                rated_mw = float(f"{m.group(1)}.{m.group(2)}")
                rotor = float(m.group(3))
                return rotor, rated_mw * 1000.0
            elif kind.startswith("letter_rotor_kw"):
                # groups: (letters, rotor, kw)
                rotor = float(m.group(2))
                rated_kw = float(m.group(3))
                if rated_kw >= 500:
                    return rotor, rated_kw
        except (ValueError, IndexError):
            continue
    return None, None


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
        df = df.copy()
        df["mfr_norm"] = df["manufacturer"].map(normalize)
        df["model_norm"] = df["turbine_type"].map(normalize)

        # Pre-extract (rotor, rated_kW) from oedb's own turbine_type strings
        # so fuzzy geometry matching is fast.
        rotors: list[float] = []
        rateds: list[float] = []
        for _, row in df.iterrows():
            r, k = _parse_model_geometry("", row["model_norm"])
            rotors.append(r if r is not None else np.nan)
            rateds.append(k if k is not None else np.nan)
        df["rotor_d_m"] = rotors
        df["rated_kW_parsed"] = rateds

        return cls(types_df=df)

    def lookup(self, manufacturer: str, model: str,
               *, uswtdb_rotor_d: Optional[float] = None,
               uswtdb_rated_kW: Optional[float] = None
               ) -> Optional[PowerCurve]:
        """Return a PowerCurve, or None if no match.

        Resolution order:
          1. Direct (mfr_alias, model_alias) lookup in MODEL_ALIASES.
          2. Normalized exact match within manufacturer.
          3. Normalized substring match (both directions) within manufacturer.
          4. Geometry-based fallback: look for an oedb model with the same
             manufacturer whose (rotor_d, rated_kW) is within ±5% of the
             USWTDB-supplied values (preferred) or a parsed model name.
          5. None — caller falls back to NREL generic.
        """
        mfr_n = MANUFACTURER_ALIASES.get(normalize(manufacturer),
                                         normalize(manufacturer))
        mod_n = normalize(model)
        if not mfr_n or not mod_n:
            return None

        # Also compute a "stripped" model variant where any redundant
        # manufacturer prefix embedded in the model string is removed.
        # USGS recently started writing GE models like "GE2.82-127" rather
        # than "2.82-127" — the strip lets either form match the same alias.
        mod_n_stripped = strip_mfr_prefix(mfr_n, mod_n)

        # 1. Alias table — try both variants
        for try_mod in (mod_n, mod_n_stripped):
            if not try_mod:
                continue
            alias_key = f"{mfr_n}|{try_mod}"
            canonical = MODEL_ALIASES.get(alias_key)
            if canonical is not None:
                row = self.types_df[
                    self.types_df["turbine_type"].str.lower() == canonical.lower()
                ]
                if not row.empty:
                    return self._row_to_curve(row.iloc[0])

        # 2-3. String match within manufacturer (try both variants)
        candidates = self.types_df[self.types_df["mfr_norm"] == mfr_n]
        if not candidates.empty:
            for try_mod in (mod_n, mod_n_stripped):
                if not try_mod:
                    continue
                exact = candidates[candidates["model_norm"] == try_mod]
                if not exact.empty:
                    return self._row_to_curve(exact.iloc[0])
                for _, row in candidates.iterrows():
                    rn = row["model_norm"]
                    if rn and (rn in try_mod or try_mod in rn):
                        return self._row_to_curve(row)

        # 4. Geometry fallback
        target_rotor = (uswtdb_rotor_d
                        if (uswtdb_rotor_d is not None
                            and np.isfinite(uswtdb_rotor_d))
                        else None)
        target_rated = (uswtdb_rated_kW
                        if (uswtdb_rated_kW is not None
                            and np.isfinite(uswtdb_rated_kW))
                        else None)
        if target_rotor is None or target_rated is None:
            # Try the stripped version first (more reliable parse), fall
            # back to the raw normalized form if that fails.
            for try_mod in (mod_n_stripped, mod_n):
                if not try_mod:
                    continue
                r_parsed, k_parsed = _parse_model_geometry(mfr_n, try_mod)
                if r_parsed is not None and target_rotor is None:
                    target_rotor = r_parsed
                if k_parsed is not None and target_rated is None:
                    target_rated = k_parsed
                if (target_rotor is not None
                        and target_rated is not None):
                    break

        if (target_rotor is not None and target_rated is not None
                and not candidates.empty):
            cand = candidates.dropna(subset=["rotor_d_m", "rated_kW_parsed"])
            if not cand.empty:
                rel_rotor = (cand["rotor_d_m"] - target_rotor) / target_rotor
                rel_rated = (cand["rated_kW_parsed"] - target_rated) / target_rated
                close = cand[(rel_rotor.abs() <= 0.05)
                             & (rel_rated.abs() <= 0.05)]
                if not close.empty:
                    score = (((close["rotor_d_m"] - target_rotor) / target_rotor) ** 2
                             + ((close["rated_kW_parsed"] - target_rated) / target_rated) ** 2)
                    best_idx = score.idxmin()
                    log.info("oedb fuzzy-match %s|%s → %s "
                             "(geom: %.0fm/%.0fkW vs %.0fm/%.0fkW)",
                             mfr_n, mod_n,
                             close.loc[best_idx, "turbine_type"],
                             target_rotor, target_rated,
                             close.loc[best_idx, "rotor_d_m"],
                             close.loc[best_idx, "rated_kW_parsed"])
                    return self._row_to_curve(close.loc[best_idx])

        return None

    @staticmethod
    def _row_to_curve(row: pd.Series) -> Optional[PowerCurve]:
        """Build a PowerCurve from a windpowerlib catalog row."""
        try:
            from windpowerlib import WindTurbine  # noqa: WPS433
        except ImportError:
            return None

        wt = WindTurbine(
            turbine_type=row["turbine_type"],
            hub_height=80,  # placeholder
        )
        if wt.power_curve is None or wt.power_curve.empty:
            return None

        ws = wt.power_curve["wind_speed"].to_numpy(dtype=float)
        pw = wt.power_curve["value"].to_numpy(dtype=float) / 1000.0
        rated_kW = float(pw.max())
        rotor_d = float(getattr(wt, "rotor_diameter", np.nan) or np.nan)
        if not np.isfinite(rotor_d):
            rotor_d = 0.0

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
            iec_class="?",
            source=f"oedb:{row['turbine_type']}",
        )
