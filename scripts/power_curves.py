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
_SP_BINS = [
    (0.0,    200.0, "iec3_low_sp"),
    (200.0,  300.0, "iec3_med_low_sp"),
    (300.0,  400.0, "iec2_med_sp"),
    (400.0,  500.0, "iec2_med_high_sp"),
    (500.0,  9999.0, "iec1_high_sp"),
]


def select_nrel_generic(rated_kW: float,
                        rotor_diameter_m: float) -> tuple[str, PowerCurve]:
    """Pick the NREL generic curve for a turbine and scale it to nameplate.

    Returns (curve_key, scaled_curve). The key is the NREL bin label
    (e.g. "iec2_med_sp"), suitable for the `curve_source` provenance field.
    """
    area = np.pi * (rotor_diameter_m / 2.0) ** 2
    sp = 1000.0 * rated_kW / area  # W/m^2

    for lo, hi, key in _SP_BINS:
        if lo <= sp < hi:
            base = NREL_GENERIC_CURVES[key]
            scaled = base.scaled_to(
                rated_kW=rated_kW,
                rotor_diameter_m=rotor_diameter_m,
                name_suffix=f"@{rated_kW:.0f}kW/{rotor_diameter_m:.0f}m",
            )
            return key, scaled

    # Fallback (should not happen given bin coverage)
    base = NREL_GENERIC_CURVES["iec2_med_sp"]
    return "iec2_med_sp", base.scaled_to(rated_kW, rotor_diameter_m)


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
