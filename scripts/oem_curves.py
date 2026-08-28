"""
Curated power curves for high-volume turbine models that aren't in oedb.

windpowerlib's curated catalog (oedb) is European-centric and missing many
of the GE / Siemens / Vestas variants that dominate the US fleet. This
module fills the gap with curves digitized from manufacturer-published
datasheets and product brochures (publicly available, not IEC 61400-12
test reports).

These are NOT certified curves. They are manufacturer design-intent
curves at standard atmospheric conditions (1.225 kg/m^3, no turbulence,
no wake losses). Field-observed performance will differ; that's what
the loss factor and per-BA calibration are for.

Each entry is a list of (wind_speed_m_s, power_kW) anchor points,
linearly interpolated between. Below cut_in: 0. At/above cut_out: 0.
Rated-power flat-top is preserved.

Coverage priority (by ERCOT 2025 installed MW):
  GE 2.82-127    5,132 MW
  GE 2.5-127     2,190 MW
  V110/2000      1,492 MW
  GE 2.3-116     1,488 MW
  GE 1.7-100     1,386 MW  (also covers 1.79 uprate)
  GE 1.5-77      1,250 MW
  GE 1.85-87     1,217 MW
  V100/2000        978 MW
  SWT 2.3-93       961 MW
  GE 3.4-137       809 MW  (also covers 3.4-140 cypress)
  GE 2.5-116       690 MW
  GE 1.5-87        675 MW
  GE 1.6-91        614 MW
  GE 2.4-107       595 MW

Sources:
  GE: GE Renewable Energy onshore product brochures
  Vestas: V100, V110 brochures (2 MW platform)
  Siemens: SWT-2.3 family product sheet
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from power_curves import PowerCurve

log = logging.getLogger(__name__)


# GE 2.82-127  (uprate of 2.5-127, same rotor; SP ~ 222 W/m^2)
_CURVE_GE_2_82_127 = [
    (3.0, 0), (3.5, 25), (4.0, 90), (4.5, 175), (5.0, 285),
    (5.5, 425), (6.0, 600), (6.5, 810), (7.0, 1050), (7.5, 1320),
    (8.0, 1610), (8.5, 1900), (9.0, 2180), (9.5, 2440), (10.0, 2660),
    (10.5, 2780), (11.0, 2820), (12.0, 2820), (15.0, 2820), (20.0, 2820),
    (25.0, 2820), (25.5, 0),
]

# GE 2.5-127  (SP ~ 197 W/m^2)
_CURVE_GE_2_5_127 = [
    (3.0, 0), (3.5, 25), (4.0, 90), (4.5, 175), (5.0, 285),
    (5.5, 425), (6.0, 600), (6.5, 810), (7.0, 1050), (7.5, 1320),
    (8.0, 1610), (8.5, 1900), (9.0, 2180), (9.5, 2400), (10.0, 2490),
    (10.5, 2500), (11.0, 2500), (15.0, 2500), (20.0, 2500), (25.0, 2500),
    (25.5, 0),
]

# GE 2.3-116  (SP ~ 218 W/m^2)
_CURVE_GE_2_3_116 = [
    (3.0, 0), (3.5, 18), (4.0, 70), (4.5, 150), (5.0, 250),
    (5.5, 380), (6.0, 540), (6.5, 730), (7.0, 955), (7.5, 1210),
    (8.0, 1480), (8.5, 1760), (9.0, 2030), (9.5, 2225), (10.0, 2290),
    (10.5, 2300), (11.0, 2300), (15.0, 2300), (20.0, 2300), (25.0, 2300),
    (25.5, 0),
]

# GE 1.7-100  (SP ~ 216 W/m^2). Also used for 1.79 uprate.
_CURVE_GE_1_7_100 = [
    (3.0, 0), (3.5, 15), (4.0, 55), (4.5, 120), (5.0, 205),
    (5.5, 315), (6.0, 450), (6.5, 610), (7.0, 790), (7.5, 990),
    (8.0, 1200), (8.5, 1400), (9.0, 1580), (9.5, 1670), (10.0, 1700),
    (10.5, 1700), (11.0, 1700), (15.0, 1700), (20.0, 1700), (25.0, 1700),
    (25.5, 0),
]

# GE 1.5sle  (77m rotor, 1.5 MW; SP ~ 322 W/m^2)
_CURVE_GE_1_5_SLE = [
    (3.5, 0), (4.0, 30), (4.5, 85), (5.0, 155), (5.5, 240),
    (6.0, 340), (6.5, 455), (7.0, 590), (7.5, 745), (8.0, 910),
    (8.5, 1080), (9.0, 1245), (9.5, 1395), (10.0, 1500), (10.5, 1500),
    (11.0, 1500), (15.0, 1500), (20.0, 1500), (25.0, 1500), (25.5, 0),
]

# GE 1.85-87  (SP ~ 311 W/m^2)
_CURVE_GE_1_85_87 = [
    (3.0, 0), (3.5, 12), (4.0, 45), (4.5, 105), (5.0, 185),
    (5.5, 290), (6.0, 420), (6.5, 580), (7.0, 775), (7.5, 990),
    (8.0, 1210), (8.5, 1430), (9.0, 1620), (9.5, 1770), (10.0, 1840),
    (10.5, 1850), (11.0, 1850), (15.0, 1850), (20.0, 1850), (25.0, 1850),
    (25.5, 0),
]

# Vestas V110/2000  (SP ~ 211 W/m^2)
_CURVE_V110_2000 = [
    (3.0, 0), (3.5, 24), (4.0, 78), (4.5, 158), (5.0, 263),
    (5.5, 394), (6.0, 555), (6.5, 747), (7.0, 969), (7.5, 1219),
    (8.0, 1490), (8.5, 1759), (9.0, 1922), (9.5, 1985), (10.0, 2000),
    (10.5, 2000), (11.0, 2000), (15.0, 2000), (20.0, 2000), (25.0, 2000),
    (25.5, 0),
]

# Vestas V100/2000  (SP ~ 255 W/m^2)
_CURVE_V100_2000 = [
    (3.0, 0), (3.5, 18), (4.0, 65), (4.5, 138), (5.0, 235),
    (5.5, 357), (6.0, 511), (6.5, 698), (7.0, 919), (7.5, 1170),
    (8.0, 1432), (8.5, 1690), (9.0, 1900), (9.5, 1985), (10.0, 2000),
    (10.5, 2000), (11.0, 2000), (15.0, 2000), (20.0, 2000), (25.0, 2000),
    (25.5, 0),
]

# Siemens SWT-2.3-93  (SP ~ 339 W/m^2)
_CURVE_SWT_2_3_93 = [
    (3.5, 0), (4.0, 30), (4.5, 85), (5.0, 165), (5.5, 275),
    (6.0, 415), (6.5, 595), (7.0, 810), (7.5, 1060), (8.0, 1330),
    (8.5, 1610), (9.0, 1880), (9.5, 2095), (10.0, 2230), (10.5, 2290),
    (11.0, 2300), (12.0, 2300), (15.0, 2300), (20.0, 2300), (25.0, 2300),
    (25.5, 0),
]

# GE 3.4-137  (SP ~ 230 W/m^2). Also used for 3.4-140 cypress.
_CURVE_GE_3_4_137 = [
    (3.0, 0), (3.5, 28), (4.0, 100), (4.5, 205), (5.0, 340),
    (5.5, 515), (6.0, 730), (6.5, 990), (7.0, 1290), (7.5, 1620),
    (8.0, 1980), (8.5, 2360), (9.0, 2715), (9.5, 3050), (10.0, 3300),
    (10.5, 3400), (11.0, 3400), (15.0, 3400), (20.0, 3400), (25.0, 3400),
    (25.5, 0),
]

# GE 2.5-116  (SP ~ 237 W/m^2)
_CURVE_GE_2_5_116 = [
    (3.0, 0), (3.5, 20), (4.0, 80), (4.5, 165), (5.0, 275),
    (5.5, 415), (6.0, 590), (6.5, 795), (7.0, 1040), (7.5, 1310),
    (8.0, 1605), (8.5, 1905), (9.0, 2200), (9.5, 2400), (10.0, 2480),
    (10.5, 2500), (11.0, 2500), (15.0, 2500), (20.0, 2500), (25.0, 2500),
    (25.5, 0),
]

# GE 1.5xle  (87m rotor, 1.5 MW; SP ~ 252 W/m^2)
_CURVE_GE_1_5_XLE = [
    (3.0, 0), (3.5, 20), (4.0, 65), (4.5, 140), (5.0, 240),
    (5.5, 365), (6.0, 515), (6.5, 690), (7.0, 885), (7.5, 1090),
    (8.0, 1290), (8.5, 1430), (9.0, 1490), (9.5, 1500), (10.0, 1500),
    (10.5, 1500), (11.0, 1500), (15.0, 1500), (20.0, 1500), (25.0, 1500),
    (25.5, 0),
]

# GE 1.6-91  (SP ~ 246 W/m^2)
_CURVE_GE_1_6_91 = [
    (3.0, 0), (3.5, 17), (4.0, 60), (4.5, 130), (5.0, 220),
    (5.5, 335), (6.0, 475), (6.5, 640), (7.0, 825), (7.5, 1020),
    (8.0, 1220), (8.5, 1410), (9.0, 1545), (9.5, 1595), (10.0, 1600),
    (10.5, 1600), (11.0, 1600), (15.0, 1600), (20.0, 1600), (25.0, 1600),
    (25.5, 0),
]

# GE 2.4-107  (SP ~ 267 W/m^2)
_CURVE_GE_2_4_107 = [
    (3.0, 0), (3.5, 22), (4.0, 75), (4.5, 155), (5.0, 255),
    (5.5, 385), (6.0, 545), (6.5, 735), (7.0, 955), (7.5, 1200),
    (8.0, 1465), (8.5, 1735), (9.0, 1985), (9.5, 2200), (10.0, 2360),
    (10.5, 2400), (11.0, 2400), (15.0, 2400), (20.0, 2400), (25.0, 2400),
    (25.5, 0),
]


# Registry: keys are normalize(t_manu) + "|" + normalize(t_model). We accept
# multiple key variants per curve to cover both the older "X Y Z" USGS format
# and the newer prefix-embedded "GEX Y Z" format.
OEM_CURVE_REGISTRY: dict[str, tuple[float, list[tuple[float, float]], str]] = {
    "ge|ge2 82 127":      (127.0, _CURVE_GE_2_82_127, "GE 2.82-127"),
    "ge|2 82 127":        (127.0, _CURVE_GE_2_82_127, "GE 2.82-127"),
    "ge|ge2 8 127":       (127.0, _CURVE_GE_2_82_127, "GE 2.82-127"),
    "ge|2 8 127":         (127.0, _CURVE_GE_2_82_127, "GE 2.82-127"),

    "ge|ge2 5 127":       (127.0, _CURVE_GE_2_5_127, "GE 2.5-127"),
    "ge|2 5 127":         (127.0, _CURVE_GE_2_5_127, "GE 2.5-127"),

    "ge|ge2 3 116":       (116.0, _CURVE_GE_2_3_116, "GE 2.3-116"),
    "ge|2 3 116":         (116.0, _CURVE_GE_2_3_116, "GE 2.3-116"),

    "ge|ge1 7 100":       (100.0, _CURVE_GE_1_7_100, "GE 1.7-100"),
    "ge|1 7 100":         (100.0, _CURVE_GE_1_7_100, "GE 1.7-100"),
    "ge|ge1 79 100":      (100.0, _CURVE_GE_1_7_100, "GE 1.7-100"),
    "ge|1 79 100":        (100.0, _CURVE_GE_1_7_100, "GE 1.7-100"),

    "ge|ge1 5 77":        (77.0,  _CURVE_GE_1_5_SLE, "GE 1.5sle"),
    "ge|1 5 77":          (77.0,  _CURVE_GE_1_5_SLE, "GE 1.5sle"),
    "ge|ge1 5sle":        (77.0,  _CURVE_GE_1_5_SLE, "GE 1.5sle"),
    "ge|1 5sle":          (77.0,  _CURVE_GE_1_5_SLE, "GE 1.5sle"),
    "ge|sle":             (77.0,  _CURVE_GE_1_5_SLE, "GE 1.5sle"),

    "ge|ge1 85 87":       (87.0,  _CURVE_GE_1_85_87, "GE 1.85-87"),
    "ge|1 85 87":         (87.0,  _CURVE_GE_1_85_87, "GE 1.85-87"),

    "vestas|v110 2 0":    (110.0, _CURVE_V110_2000, "V110/2000"),
    "vestas|v110":        (110.0, _CURVE_V110_2000, "V110/2000"),

    "vestas|v100 2 0":    (100.0, _CURVE_V100_2000, "V100/2000"),
    "vestas|v100":        (100.0, _CURVE_V100_2000, "V100/2000"),

    "siemens|swt 2 3 93": (93.0,  _CURVE_SWT_2_3_93, "SWT-2.3-93"),
    "siemens|swt2 3 93":  (93.0,  _CURVE_SWT_2_3_93, "SWT-2.3-93"),

    "ge|ge3 4 137":       (137.0, _CURVE_GE_3_4_137, "GE 3.4-137"),
    "ge|3 4 137":         (137.0, _CURVE_GE_3_4_137, "GE 3.4-137"),
    "ge|ge3 4 140":       (140.0, _CURVE_GE_3_4_137, "GE 3.4-137"),
    "ge|3 4 140":         (140.0, _CURVE_GE_3_4_137, "GE 3.4-137"),

    "ge|ge2 5 116":       (116.0, _CURVE_GE_2_5_116, "GE 2.5-116"),
    "ge|2 5 116":         (116.0, _CURVE_GE_2_5_116, "GE 2.5-116"),

    "ge|ge1 5 87":        (87.0,  _CURVE_GE_1_5_XLE, "GE 1.5xle"),
    "ge|1 5 87":          (87.0,  _CURVE_GE_1_5_XLE, "GE 1.5xle"),
    "ge|ge1 5xle":        (87.0,  _CURVE_GE_1_5_XLE, "GE 1.5xle"),
    "ge|1 5xle":          (87.0,  _CURVE_GE_1_5_XLE, "GE 1.5xle"),
    "ge|xle":             (87.0,  _CURVE_GE_1_5_XLE, "GE 1.5xle"),

    "ge|ge1 6 91":        (91.0,  _CURVE_GE_1_6_91, "GE 1.6-91"),
    "ge|1 6 91":          (91.0,  _CURVE_GE_1_6_91, "GE 1.6-91"),

    "ge|ge2 4 107":       (107.0, _CURVE_GE_2_4_107, "GE 2.4-107"),
    "ge|2 4 107":         (107.0, _CURVE_GE_2_4_107, "GE 2.4-107"),
}


def _anchors_to_curve(anchors: list[tuple[float, float]],
                      rotor_d: float, name: str) -> PowerCurve:
    """Resample brochure anchors onto the standard 0.5 m/s grid using a
    monotonic cubic (PCHIP) spline through the ramp region.

    Why PCHIP over plain cubic spline: cubic splines can overshoot near
    inflection points, which on a power curve would push interpolated
    values above rated power between the last ramp anchor and the rated-
    power plateau. PCHIP preserves monotonicity by construction, so the
    ramp stays bounded by [0, rated_kW] and the flat-top stays flat.

    The cut-in and cut-out points are imposed exactly (zero on both
    sides) rather than letting the spline decide, since those are
    physical control limits, not curve features.
    """
    ws_anchor = np.array([a[0] for a in anchors], dtype=float)
    pw_anchor = np.array([a[1] for a in anchors], dtype=float)

    rated_kW = float(pw_anchor.max())
    nonzero = np.flatnonzero(pw_anchor > 0)
    cut_in = float(ws_anchor[nonzero[0]]) if len(nonzero) else 3.0
    if len(nonzero) >= 1 and nonzero[-1] + 1 < len(ws_anchor):
        cut_out = float(ws_anchor[nonzero[-1] + 1])
    else:
        cut_out = 25.5

    # Resample on a fine 0.1 m/s grid so the smoothing actually shows up
    # at sub-anchor wind speeds. Brochure anchors are at 0.5 m/s, but
    # HRRR-derived hub-height winds and the temperature-corrected winds
    # used in forecast.py can land anywhere in between, so we want the
    # power() lookup to capture the spline rather than snap to the
    # nearest 0.5 m/s tick.
    ws_grid = np.arange(0.0, 30.05, 0.1)

    # Build the interpolant on anchors that are in the operating range
    # (cut_in <= ws < cut_out and pw > 0). The flat-top region near rated
    # is included so the spline knows where to level off.
    op_mask = (ws_anchor >= cut_in) & (ws_anchor < cut_out)
    ws_op = ws_anchor[op_mask]
    pw_op = pw_anchor[op_mask]

    pw_grid = np.zeros_like(ws_grid)
    if len(ws_op) >= 2:
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(ws_op, pw_op, extrapolate=False)
            in_range = (ws_grid >= ws_op[0]) & (ws_grid <= ws_op[-1])
            pw_grid[in_range] = interp(ws_grid[in_range])
            # Beyond the last operating anchor but still below cut_out:
            # the curve is flat at rated. The PchipInterpolator returns
            # NaN past the last anchor (extrapolate=False), so fill the
            # gap explicitly.
            tail = (ws_grid > ws_op[-1]) & (ws_grid < cut_out)
            pw_grid[tail] = rated_kW
        except ImportError:
            # Fallback to linear if scipy is unavailable
            log.warning("scipy not available; falling back to linear "
                        "interpolation for OEM curve %s", name)
            pw_grid = np.interp(ws_grid, ws_anchor, pw_anchor,
                                left=0.0, right=0.0)
    else:
        pw_grid = np.interp(ws_grid, ws_anchor, pw_anchor,
                            left=0.0, right=0.0)

    # Hard zeros below cut-in and at/above cut-out
    pw_grid = np.where(ws_grid < cut_in, 0.0, pw_grid)
    pw_grid = np.where(ws_grid >= cut_out, 0.0, pw_grid)
    # Clip any tiny numerical excursions outside [0, rated]
    pw_grid = np.clip(pw_grid, 0.0, rated_kW)

    return PowerCurve(
        name=name,
        rated_kW=rated_kW,
        rotor_diameter_m=rotor_d,
        wind_speeds_m_s=ws_grid,
        power_kW=pw_grid,
        cut_in=cut_in,
        cut_out=cut_out,
        iec_class="?",
        source=f"oem:{name}",
    )


def lookup_oem_curve(mfr_norm: str, model_norm: str) -> Optional[PowerCurve]:
    """Look up a curated OEM curve. Returns None if no match.

    Inputs are the same normalized strings used by oedb_catalog: lowercase,
    punctuation-stripped, whitespace-collapsed.
    """
    key = f"{mfr_norm}|{model_norm}"
    entry = OEM_CURVE_REGISTRY.get(key)
    if entry is None:
        return None
    rotor_d, anchors, name = entry
    return _anchors_to_curve(anchors, rotor_d, name)


def registry_summary() -> list[tuple[str, float, float]]:
    """Return one row per unique curve: (name, rotor_d, rated_kW)."""
    seen: dict[str, tuple[float, float]] = {}
    for _, (rotor_d, anchors, name) in OEM_CURVE_REGISTRY.items():
        if name not in seen:
            rated_kW = max(p for _, p in anchors)
            seen[name] = (rotor_d, rated_kW)
    return [(name, rd, rk) for name, (rd, rk) in seen.items()]
