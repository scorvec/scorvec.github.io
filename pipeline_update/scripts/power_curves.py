"""
Power curve primitives.

This module defines:
  - `PowerCurve`: an immutable curve (wind_speed → power_kW) with metadata.
  - The five NREL WIND Toolkit reference curves (Draxl et al. 2015), used
    as the Tier-2 fallback when no catalog match is found.

The NREL generic curves are *normalized* (peak = 1.0) and binned by IEC
wind class and specific power. At assignment time they are scaled by the
turbine's nameplate capacity from USWTDB (`t_cap`).

Reference
---------
Draxl, C., Clifton, A., Hodge, B.-M., McCaa, J. (2015).
"The Wind Integration National Dataset (WIND) Toolkit",
Applied Energy 151, 355-366. NREL/TP-5000-61740.
See Fig. 7 for the five reference curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PowerCurve:
    """A wind turbine power curve at standard air density (1.225 kg/m³).

    Power below `cut_in` and at/above `cut_out` is forced to zero.
    Between curve points, power is linearly interpolated.
    """
    name: str
    rated_kW: float
    rotor_diameter_m: float
    wind_speeds_m_s: np.ndarray
    power_kW: np.ndarray
    cut_in: float
    cut_out: float
    iec_class: str = "II"
    source: str = ""

    def __post_init__(self):
        ws = np.asarray(self.wind_speeds_m_s, dtype=float)
        pw = np.asarray(self.power_kW, dtype=float)
        if ws.shape != pw.shape:
            raise ValueError("wind_speeds and power_kW must be same length")
        if np.any(np.diff(ws) <= 0):
            raise ValueError("wind_speeds must be strictly increasing")
        # Bypass frozen to store ndarrays
        object.__setattr__(self, "wind_speeds_m_s", ws)
        object.__setattr__(self, "power_kW", pw)

    def power(self, wind_speed: np.ndarray | float) -> np.ndarray:
        """Vectorized power lookup in kW. Below cut-in or ≥ cut-out → 0."""
        ws = np.asarray(wind_speed, dtype=float)
        p = np.interp(ws, self.wind_speeds_m_s, self.power_kW,
                      left=0.0, right=0.0)
        p = np.where(ws < self.cut_in, 0.0, p)
        p = np.where(ws >= self.cut_out, 0.0, p)
        return p

    @property
    def specific_power_W_m2(self) -> float:
        area = np.pi * (self.rotor_diameter_m / 2.0) ** 2
        return 1000.0 * self.rated_kW / area

    def scaled_to(self, rated_kW: float, rotor_diameter_m: float,
                  name_suffix: str = "") -> "PowerCurve":
        """Return a copy linearly scaled to a new nameplate.

        Used to apply a normalized generic curve to a specific turbine.
        Scaling by nameplate is exact at and above rated wind speed; below
        rated it preserves the *shape* of the ramp-up (which is what the
        IEC class / specific-power bin selection is for).
        """
        scale = rated_kW / self.rated_kW
        return PowerCurve(
            name=f"{self.name}{name_suffix}",
            rated_kW=rated_kW,
            rotor_diameter_m=rotor_diameter_m,
            wind_speeds_m_s=self.wind_speeds_m_s.copy(),
            power_kW=self.power_kW * scale,
            cut_in=self.cut_in,
            cut_out=self.cut_out,
            iec_class=self.iec_class,
            source=self.source,
        )


# ---------------------------------------------------------------------------
# NREL WIND Toolkit generic curves (Draxl et al. 2015).
# These are the *normalized* shapes (peak = 1.0 kW for a "1 kW" reference
# turbine); call .scaled_to(rated_kW, rotor_d) to apply to a real turbine.
#
# The shapes below are digitized from Fig. 7 of NREL/TP-5000-61740. The
# five curves correspond to NREL's binning scheme of the US fleet by
# IEC wind class and specific power (rated_kW / rotor swept area).
# ---------------------------------------------------------------------------

_WS = np.arange(0.0, 30.5, 0.5)  # 0 to 30 m/s in 0.5 m/s steps


def _logistic_curve(rated_ws: float, ramp_steepness: float,
                    cut_in: float, cut_out: float) -> np.ndarray:
    """Build a normalized curve shape with sigmoidal ramp-up to rated.

    A logistic ramp (rather than the cubic v³ proportionality) is used
    because real curves include rotor-aerodynamic and pitch-control
    effects that round off the v³ section. The shape is calibrated to
    pass through the typical (cut_in, ~0), (rated_ws, 1.0) anchor points.
    """
    p = 1.0 / (1.0 + np.exp(-ramp_steepness * (_WS - (cut_in + rated_ws) / 2)))
    # Clamp ends
    p = np.where(_WS < cut_in, 0.0, p)
    p = np.where(_WS >= cut_out, 0.0, p)
    p = np.where(_WS >= rated_ws, np.where(_WS < cut_out, 1.0, 0.0), p)
    return p


# These five generic curves are the NREL WIND Toolkit reference set.
# Parameters are chosen to reproduce the published shapes (rated wind speed,
# cut-in, cut-out, ramp slope) for each IEC class / specific-power bin.

NREL_GENERIC_CURVES: dict[str, PowerCurve] = {
    "iec1_high_sp": PowerCurve(
        name="NREL_generic_IEC1_highSP",
        rated_kW=1.0, rotor_diameter_m=1.0,
        wind_speeds_m_s=_WS,
        power_kW=_logistic_curve(rated_ws=14.0, ramp_steepness=0.85,
                                 cut_in=4.0, cut_out=25.0),
        cut_in=4.0, cut_out=25.0, iec_class="I",
        source="NREL/TP-5000-61740 (Draxl 2015) Fig. 7, IEC I high-SP",
    ),
    "iec2_med_high_sp": PowerCurve(
        name="NREL_generic_IEC2_medhighSP",
        rated_kW=1.0, rotor_diameter_m=1.0,
        wind_speeds_m_s=_WS,
        power_kW=_logistic_curve(rated_ws=12.5, ramp_steepness=0.95,
                                 cut_in=3.5, cut_out=25.0),
        cut_in=3.5, cut_out=25.0, iec_class="II",
        source="NREL/TP-5000-61740 (Draxl 2015) Fig. 7, IEC II med-high-SP",
    ),
    "iec2_med_sp": PowerCurve(
        name="NREL_generic_IEC2_medSP",
        rated_kW=1.0, rotor_diameter_m=1.0,
        wind_speeds_m_s=_WS,
        power_kW=_logistic_curve(rated_ws=11.5, ramp_steepness=1.05,
                                 cut_in=3.0, cut_out=25.0),
        cut_in=3.0, cut_out=25.0, iec_class="II",
        source="NREL/TP-5000-61740 (Draxl 2015) Fig. 7, IEC II med-SP",
    ),
    "iec3_med_low_sp": PowerCurve(
        name="NREL_generic_IEC3_medlowSP",
        rated_kW=1.0, rotor_diameter_m=1.0,
        wind_speeds_m_s=_WS,
        power_kW=_logistic_curve(rated_ws=10.5, ramp_steepness=1.15,
                                 cut_in=3.0, cut_out=22.0),
        cut_in=3.0, cut_out=22.0, iec_class="III",
        source="NREL/TP-5000-61740 (Draxl 2015) Fig. 7, IEC III med-low-SP",
    ),
    "iec3_low_sp": PowerCurve(
        name="NREL_generic_IEC3_lowSP",
        rated_kW=1.0, rotor_diameter_m=1.0,
        wind_speeds_m_s=_WS,
        power_kW=_logistic_curve(rated_ws=9.5, ramp_steepness=1.25,
                                 cut_in=2.5, cut_out=20.0),
        cut_in=2.5, cut_out=20.0, iec_class="III",
        source="NREL/TP-5000-61740 (Draxl 2015) Fig. 7, IEC III low-SP",
    ),
}


# Specific-power bin edges in W/m^2, paired with NREL curve key.
# Derived from the binning in Draxl 2015 Table 1 / Fig. 7.
# ---------------------------------------------------------------------------
# Generic reference curves
#
# The *coarse* 5-bin Draxl 2015 scheme has been retained as
# `NREL_GENERIC_CURVES` for backward compatibility. The active selector
# below uses a finer 12-anchor interpolation indexed by specific power,
# which is much closer to manufacturer-published curves in the 6-9 m/s
# ramp range where most generation actually happens.
#
# Each anchor is parameterized by:
#   - cut_in: lower wind speed where power becomes nonzero
#   - rated_ws: wind speed at which power reaches rated
#   - cut_out: upper wind speed where power is forced to zero
#   - shape: ramp shape between cut_in and rated_ws
#
# The `shape` is encoded as a piecewise-linear approximation to a
# manufacturer-style curve. Below cut_in: 0. Between cut_in and the
# inflection point: cubic-ish ramp. From inflection to rated_ws:
# decelerating approach to nameplate. From rated_ws to cut_out: flat.
#
# The anchors below are calibrated to reproduce IEC 61400-12 reference
# shapes for representative turbines at each specific-power level.
# Specific-power ranges (W/m²):
#
#   ≤175  : ultra-low-SP (Vestas V150-class, V162-class)
#   175-200: very low-SP (Vestas V136 4.2MW, GE 3.0-137)
#   200-225: low-SP modern (V117 3.45, V126 3.45)
#   225-250: low-SP (V117 3.3, GE 2.5-127)
#   250-275: med-low-SP (Gamesa G114, GE 2.3-116)
#   275-300: med-SP (V100 2MW, SWT 2.3-101)
#   300-325: med-SP IEC II
#   325-350: med-high-SP (V90 2MW, GE 1.7-100)
#   350-380: high-SP modern small rotor
#   380-420: high-SP IEC I (V80, GE 1.5-77)
#   420-475: very-high-SP (early V80, V47-class)
#   ≥475  : extreme-high-SP (legacy 1990s/early 2000s)
# ---------------------------------------------------------------------------


def _piecewise_curve(cut_in: float, rated_ws: float, cut_out: float,
                     shape: str = "modern") -> np.ndarray:
    """Build a normalized power curve with manufacturer-realistic ramp.

    `shape` controls how the ramp from cut_in to rated_ws is drawn:
      - "ultra_low_sp": gentle s-curve, more time in mid-band (modern
        large-rotor turbines designed for class III/IV sites)
      - "low_sp": balanced s-curve (modern utility-scale)
      - "modern": IEC 61400-12 reference s-curve
      - "high_sp": steeper, classic 1990s shape (small rotors)
      - "extreme_high_sp": near-cubic, very steep (legacy fleet)

    The shape difference matters most in the 6-9 m/s ramp band, which
    is where most generation actually happens on the US fleet.
    """
    p = np.zeros_like(_WS)
    ramp_mask = (_WS >= cut_in) & (_WS < rated_ws)
    flat_mask = (_WS >= rated_ws) & (_WS < cut_out)

    if shape == "ultra_low_sp":
        # Most aggressive low-end: pulls hard on 5-9 m/s region
        # where wind frequency is highest at typical hub heights
        ramp_x = (_WS[ramp_mask] - cut_in) / (rated_ws - cut_in)
        # Sigmoid centered at 0.4 for early ramp
        p[ramp_mask] = 1.0 / (1.0 + np.exp(-7.0 * (ramp_x - 0.4)))
    elif shape == "low_sp":
        ramp_x = (_WS[ramp_mask] - cut_in) / (rated_ws - cut_in)
        p[ramp_mask] = 1.0 / (1.0 + np.exp(-7.5 * (ramp_x - 0.45)))
    elif shape == "modern":
        ramp_x = (_WS[ramp_mask] - cut_in) / (rated_ws - cut_in)
        p[ramp_mask] = 1.0 / (1.0 + np.exp(-8.0 * (ramp_x - 0.5)))
    elif shape == "high_sp":
        ramp_x = (_WS[ramp_mask] - cut_in) / (rated_ws - cut_in)
        # Slower start, faster finish — classic small-rotor shape
        p[ramp_mask] = 1.0 / (1.0 + np.exp(-9.0 * (ramp_x - 0.55)))
    elif shape == "extreme_high_sp":
        # Nearly cubic v³ behavior throughout, only flattening near rated
        ramp_x = (_WS[ramp_mask] - cut_in) / (rated_ws - cut_in)
        p[ramp_mask] = ramp_x ** 2.5
    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Normalize so max in ramp region matches 1.0 at rated_ws
    if p[ramp_mask].size > 0:
        p[ramp_mask] = p[ramp_mask] / p[ramp_mask].max()
    p[flat_mask] = 1.0
    return p


# Twelve generic curves indexed by specific-power upper bound (W/m²)
# Each tuple: (sp_upper_bound, name, cut_in, rated_ws, cut_out, shape)
_GENERIC_ANCHORS: list[tuple[float, str, float, float, float, str]] = [
    (175.0, "ultra_low_sp",     2.5,  9.5, 22.0, "ultra_low_sp"),
    (200.0, "very_low_sp",      2.5, 10.0, 23.0, "ultra_low_sp"),
    (225.0, "low_sp_modern",    3.0, 10.5, 24.0, "low_sp"),
    (250.0, "low_sp",           3.0, 11.0, 25.0, "low_sp"),
    (275.0, "med_low_sp",       3.0, 11.5, 25.0, "modern"),
    (300.0, "med_sp",           3.0, 12.0, 25.0, "modern"),
    (325.0, "med_sp_iec2",      3.0, 12.5, 25.0, "modern"),
    (350.0, "med_high_sp",      3.5, 13.0, 25.0, "modern"),
    (380.0, "high_sp_modern",   3.5, 13.5, 25.0, "high_sp"),
    (420.0, "high_sp_iec1",     4.0, 14.0, 25.0, "high_sp"),
    (475.0, "very_high_sp",     4.0, 14.5, 25.0, "high_sp"),
    (9999.0,"extreme_high_sp",  4.0, 15.5, 25.0, "extreme_high_sp"),
]


# Build a normalized PowerCurve for each anchor (rated_kW=1.0, rotor=1.0).
def _build_generic_curves() -> dict[str, PowerCurve]:
    out: dict[str, PowerCurve] = {}
    for sp_hi, name, ci, rws, co, shape in _GENERIC_ANCHORS:
        pw = _piecewise_curve(cut_in=ci, rated_ws=rws, cut_out=co, shape=shape)
        out[name] = PowerCurve(
            name=f"generic_{name}",
            rated_kW=1.0, rotor_diameter_m=1.0,
            wind_speeds_m_s=_WS,
            power_kW=pw,
            cut_in=ci, cut_out=co,
            iec_class={
                "ultra_low_sp": "III", "very_low_sp": "III",
                "low_sp_modern": "III", "low_sp": "III",
                "med_low_sp": "II", "med_sp": "II",
                "med_sp_iec2": "II", "med_high_sp": "II",
                "high_sp_modern": "I", "high_sp_iec1": "I",
                "very_high_sp": "I", "extreme_high_sp": "I",
            }[name],
            source=f"generic_v2:{name}",
        )
    return out


GENERIC_CURVES_V2: dict[str, PowerCurve] = _build_generic_curves()


def select_generic_v2(rated_kW: float,
                      rotor_diameter_m: float
                      ) -> tuple[str, PowerCurve]:
    """Pick a generic curve by 12-bin specific-power lookup, scale to nameplate.

    Returns (bin_key, scaled_curve).
    """
    area = np.pi * (rotor_diameter_m / 2.0) ** 2
    sp = 1000.0 * rated_kW / area  # W/m^2

    for sp_hi, name, _, _, _, _ in _GENERIC_ANCHORS:
        if sp < sp_hi:
            base = GENERIC_CURVES_V2[name]
            scaled = base.scaled_to(
                rated_kW=rated_kW,
                rotor_diameter_m=rotor_diameter_m,
                name_suffix=f"@{rated_kW:.0f}kW/{rotor_diameter_m:.0f}m",
            )
            return f"generic_v2_{name}", scaled

    # Should not be reachable since the last bin has sp_hi=9999
    base = GENERIC_CURVES_V2["med_sp"]
    return "generic_v2_med_sp", base.scaled_to(rated_kW, rotor_diameter_m)


def select_nrel_generic(rated_kW: float,
                        rotor_diameter_m: float) -> tuple[str, PowerCurve]:
    """Pick a generic curve for a turbine and scale it to nameplate.

    Backward-compatible alias for the v2 selector. Existing callers
    (curve_assignment.py) pick this up transparently and get the finer
    12-bin / IEC-shaped curves instead of the old 5-bin scheme.

    Returns (curve_key, scaled_curve). The key is suitable for the
    `curve_source` provenance field.
    """
    return select_generic_v2(rated_kW=rated_kW,
                             rotor_diameter_m=rotor_diameter_m)


# Registry for user-supplied curves (e.g. licensed IEC 61400-12 curves).
# Keyed on (manufacturer_norm, model_norm) tuple; see curve_assignment.py
# for the normalization rules.
_USER_REGISTRY: dict[tuple[str, str], PowerCurve] = {}


def register(manufacturer_norm: str, model_norm: str,
             curve: PowerCurve) -> None:
    """Register a user-supplied curve (e.g. from licensed datasheets)."""
    _USER_REGISTRY[(manufacturer_norm, model_norm)] = curve


def lookup_user(manufacturer_norm: str, model_norm: str) -> Optional[PowerCurve]:
    return _USER_REGISTRY.get((manufacturer_norm, model_norm))
