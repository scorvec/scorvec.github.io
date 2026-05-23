"""Solar power physics model.

Converts HRRR radiation fields (DSWRF, VBDSF, VDDSF) and surface
temperature into AC power for each plant.

Physics chain:
  1. Sun position from lat/lon/UTC time (pvlib.solarposition)
  2. Tracker geometry — fixed, single-axis, or dual-axis — determines
     panel tilt and azimuth (pvlib.tracking)
  3. Decompose HRRR beam + diffuse into plane-of-array (POA) irradiance
     using isotropic sky diffuse model
  4. Compute cell temperature from POA + ambient
  5. Compute DC power with temperature derate (PVWatts model)
  6. Convert DC → AC with inverter clipping at nameplate

Uses HRRR's native beam/diffuse split (no empirical decomposition models
like Erbs/DISC) since HRRR's radiative transfer scheme directly computes
this from cloud optical depth and aerosols.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# pvlib provides sun position, tracker geometry, POA irradiance.
# Install: pip install pvlib
import pvlib


# ============================================================================
# Physical constants and defaults
# ============================================================================

STC_IRRADIANCE = 1000.0       # W/m², standard test conditions
STC_TEMPERATURE = 25.0        # °C, standard test conditions
PANEL_TEMP_COEFF = -0.004     # 1/°C, crystalline-silicon module power tempco
TEMP_RISE_PER_KW_M2 = 30.0    # °C per 1000 W/m² POA, simple INOCT model
DEFAULT_DC_LOSSES = 0.05      # 5%, lumped soiling+wiring+mismatch (NREL PVWatts)
DEFAULT_INVERTER_EFF = 0.96   # 96% nominal inverter efficiency
DEFAULT_DC_AC_RATIO = 1.2     # If DC capacity unknown, assume 1.2× AC
DEFAULT_GCR = 0.35            # Ground coverage ratio for single-axis trackers
DEFAULT_AXIS_AZIMUTH = 180.0  # N-S tracker axis (sun moves E to W)
DEFAULT_MAX_TRACKER_ANGLE = 60.0  # ±60° rotation limit

# Albedo for diffuse-reflected ground irradiance. Generic land = 0.20.
DEFAULT_ALBEDO = 0.20


# ============================================================================
# Plant attribute normalization
# ============================================================================

@dataclass
class PlantConfig:
    """Resolved plant configuration with defaults applied."""
    axis: str                 # 'fixed', 'single-axis', 'dual-axis'
    tilt: float               # degrees from horizontal (fixed-tilt only)
    azimuth: float            # degrees E of N (fixed-tilt only)
    capacity_ac: float        # MW AC nameplate
    capacity_dc: float        # MW DC nameplate (after dc/ac ratio default)


def _normalize_axis(raw: Optional[str]) -> str:
    """Normalize p_axis values to one of: fixed, single-axis, dual-axis."""
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return "fixed"  # default for missing info (e.g., floating solar)
    s = str(raw).lower().strip()
    # Compound types use first listed (predominant)
    if "," in s:
        s = s.split(",")[0].strip()
    if "dual" in s:
        return "dual-axis"
    if "single" in s:
        return "single-axis"
    if "fix" in s:
        return "fixed"
    return "fixed"


def resolve_plant_config(row: pd.Series) -> PlantConfig:
    """Apply defaults to fill in missing plant attributes.

    USPVDB has occasional NaN values for p_tilt, p_azimuth, p_axis,
    p_cap_dc. We apply reasonable defaults so the physics doesn't blow up.
    """
    axis = _normalize_axis(row.get("p_axis"))

    # Tilt: latitude × 0.76 maximizes annual energy at fixed orientation,
    # a standard rule of thumb. Real plants tilt anywhere from 10° to 35°,
    # we default to lat-based.
    raw_tilt = row.get("p_tilt", np.nan)
    if pd.isna(raw_tilt) or raw_tilt <= 0:
        lat = float(row.get("ylat", 35.0))
        tilt = max(10.0, min(40.0, abs(lat) * 0.76))
    else:
        tilt = float(raw_tilt)

    # Azimuth: 180° (south-facing) is the default. USPVDB uses degrees
    # east of north (compass), which is what pvlib expects.
    raw_az = row.get("p_azimuth", np.nan)
    if pd.isna(raw_az):
        azimuth = 180.0
    else:
        azimuth = float(raw_az)

    capacity_ac = float(row.get("p_cap_ac", 0.0))

    raw_dc = row.get("p_cap_dc", np.nan)
    if pd.isna(raw_dc) or raw_dc <= 0:
        capacity_dc = capacity_ac * DEFAULT_DC_AC_RATIO
    else:
        capacity_dc = float(raw_dc)

    return PlantConfig(
        axis=axis, tilt=tilt, azimuth=azimuth,
        capacity_ac=capacity_ac, capacity_dc=capacity_dc,
    )


# ============================================================================
# Single-plant physics — vectorized over time
# ============================================================================

def compute_plant_power(
    times_utc: pd.DatetimeIndex,
    lat: float,
    lon: float,
    config: PlantConfig,
    dswrf: np.ndarray,         # W/m² broadband total
    beam: np.ndarray,          # W/m² broadband direct beam on horizontal (VBDSF)
    diffuse: np.ndarray,       # W/m² broadband diffuse (VDDSF)
    temp_k: np.ndarray,        # K, air temp at 2m
    albedo: float = DEFAULT_ALBEDO,
) -> dict:
    """Compute AC power timeseries for one plant.

    Returns dict with arrays (all same length as times_utc):
      'poa_global'   — plane-of-array irradiance, W/m²
      'temp_cell_c'  — cell temperature, °C
      'mw_dc'        — DC power before inverter, MW
      'mw_ac'        — AC power after inverter clipping, MW
    """
    n = len(times_utc)
    temp_c = temp_k - 273.15

    # 1. Sun position. pvlib expects tz-aware UTC datetimes.
    if times_utc.tz is None:
        times = times_utc.tz_localize("UTC")
    else:
        times = times_utc.tz_convert("UTC")
    sp = pvlib.solarposition.get_solarposition(times, lat, lon)
    apparent_zenith = sp["apparent_zenith"].values     # degrees from vertical
    apparent_azimuth = sp["azimuth"].values             # degrees E of N

    # 2. Tracker geometry → effective panel tilt and azimuth per timestep
    if config.axis == "single-axis":
        # N-S axis tracker. axis_azimuth=180° means axis runs N-S, panels
        # rotate to follow sun E-to-W. Backtracking on, typical GCR=0.35.
        # pvlib 0.13.1+ renamed apparent_azimuth→solar_azimuth (keeping
        # apparent_zenith as-is, awkwardly). Try new names first, fall
        # back to old.
        common_kwargs = dict(
            axis_tilt=0,                       # axis is horizontal
            axis_azimuth=DEFAULT_AXIS_AZIMUTH,
            max_angle=DEFAULT_MAX_TRACKER_ANGLE,
            backtrack=True,
            gcr=DEFAULT_GCR,
        )
        try:
            track = pvlib.tracking.singleaxis(
                apparent_zenith=pd.Series(apparent_zenith, index=times),
                solar_azimuth=pd.Series(apparent_azimuth, index=times),
                **common_kwargs,
            )
        except TypeError:
            # Older pvlib expects apparent_azimuth
            track = pvlib.tracking.singleaxis(
                apparent_zenith=pd.Series(apparent_zenith, index=times),
                apparent_azimuth=pd.Series(apparent_azimuth, index=times),
                **common_kwargs,
            )
        # singleaxis may return a DataFrame (older pvlib) or dict (newer).
        # Pull values either way.
        if hasattr(track, "values") and hasattr(track, "columns"):
            surface_tilt = track["surface_tilt"].values
            surface_azimuth = track["surface_azimuth"].values
        else:
            surface_tilt = np.asarray(track["surface_tilt"])
            surface_azimuth = np.asarray(track["surface_azimuth"])
        # When sun is below horizon, tracker model returns NaN. Fill so
        # we don't propagate NaN into the irradiance calculation.
        surface_tilt = np.nan_to_num(surface_tilt, nan=0.0)
        surface_azimuth = np.nan_to_num(surface_azimuth, nan=180.0)
    elif config.axis == "dual-axis":
        # Panels always face the sun directly. surface_tilt = sun zenith,
        # surface_azimuth = sun azimuth.
        surface_tilt = apparent_zenith
        surface_azimuth = apparent_azimuth
    else:
        # Fixed-tilt: constants
        surface_tilt = np.full(n, config.tilt)
        surface_azimuth = np.full(n, config.azimuth)

    # 3. Decompose HRRR horizontal beam → DNI (direct normal). HRRR gives
    # us VBDSF (beam on horizontal); pvlib wants DNI. Convert:
    #   DNI = VBDSF / cos(zenith)  when zenith < 90°, else 0
    zenith_rad = np.deg2rad(apparent_zenith)
    cos_zenith = np.cos(zenith_rad)
    # Guard: don't divide by ~0 at sunrise/sunset, would give nonsense
    cos_zenith_safe = np.where(cos_zenith > 0.05, cos_zenith, np.nan)
    dni = beam / cos_zenith_safe
    dni = np.where(apparent_zenith < 87.0, dni, 0.0)
    dni = np.nan_to_num(dni, nan=0.0).clip(min=0, max=1400)

    # 4. Get POA irradiance using HRRR's beam+diffuse split. pvlib's
    # get_total_irradiance uses an isotropic sky diffuse model by default,
    # which is a reasonable approximation.
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=apparent_zenith,
        solar_azimuth=apparent_azimuth,
        dni=dni,
        ghi=dswrf,
        dhi=diffuse,
        albedo=albedo,
        model="isotropic",
    )
    # Handle DataFrame or dict-of-arrays return shape
    if hasattr(poa, "values") and hasattr(poa, "columns"):
        poa_global = poa["poa_global"].values
    else:
        poa_global = np.asarray(poa["poa_global"])
    poa_global = np.nan_to_num(poa_global, nan=0.0).clip(min=0)

    # 5. Cell temperature: ambient + heating from absorbed irradiance
    # Simple INOCT-style model: rise = 30°C × (POA / 1000)
    temp_cell_c = temp_c + TEMP_RISE_PER_KW_M2 * (poa_global / STC_IRRADIANCE)

    # 6. DC power using PVWatts: P_dc = (POA/1000) × P_dc0 × (1 + γ(Tc-25))
    # Apply DC-side losses (soiling, wiring, mismatch) here.
    # pvlib 0.13+ renamed g_poa_effective → effective_irradiance.
    try:
        p_dc_kw = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=poa_global,
            temp_cell=temp_cell_c,
            pdc0=config.capacity_dc * 1000.0,
            gamma_pdc=PANEL_TEMP_COEFF,
        )
    except TypeError:
        p_dc_kw = pvlib.pvsystem.pvwatts_dc(
            g_poa_effective=poa_global,
            temp_cell=temp_cell_c,
            pdc0=config.capacity_dc * 1000.0,
            gamma_pdc=PANEL_TEMP_COEFF,
        )
    p_dc_kw = np.asarray(p_dc_kw) * (1.0 - DEFAULT_DC_LOSSES)
    p_dc_kw = np.maximum(p_dc_kw, 0.0)

    # 7. AC power with inverter efficiency + clipping at AC nameplate
    p_ac_kw = pvlib.inverter.pvwatts(
        pdc=p_dc_kw,
        pdc0=config.capacity_dc * 1000.0,
        eta_inv_nom=DEFAULT_INVERTER_EFF,
    )
    # Clip at AC nameplate (the inverter physically can't put out more)
    p_ac_kw = np.minimum(p_ac_kw, config.capacity_ac * 1000.0)
    p_ac_kw = np.maximum(p_ac_kw, 0.0)

    return {
        "poa_global": poa_global,
        "temp_cell_c": temp_cell_c,
        "mw_dc": p_dc_kw / 1000.0,
        "mw_ac": p_ac_kw / 1000.0,
    }


# ============================================================================
# Self-test
# ============================================================================

def _self_test():
    """Quick sanity check: midday clear-sky output for a 100 MW single-axis
    tracker plant in Phoenix should produce roughly 80-95 MW AC."""
    print("=" * 70)
    print("solar_power_model self-test")
    print("=" * 70)

    times = pd.date_range("2026-06-21 18:00", periods=1, freq="h", tz="UTC")
    config = PlantConfig(
        axis="single-axis", tilt=20.0, azimuth=180.0,
        capacity_ac=100.0, capacity_dc=130.0,
    )

    # Clear-sky June midday Phoenix: ~1000 W/m² DSWRF, mostly beam
    result = compute_plant_power(
        times_utc=times,
        lat=33.45, lon=-112.07,
        config=config,
        dswrf=np.array([1000.0]),
        beam=np.array([850.0]),       # ~85% beam on clear day
        diffuse=np.array([150.0]),
        temp_k=np.array([308.0]),     # 35°C (95°F)
    )
    print(f"\nPhoenix 100 MW single-axis @ June 21, 18Z (~11 AM local):")
    print(f"  Sun position: zenith ~22° (high), azimuth ~140° (SE)")
    print(f"  HRRR: GHI=1000, beam=850, diffuse=150, T_air=35°C")
    print(f"  POA:         {result['poa_global'][0]:.0f} W/m²")
    print(f"  T_cell:      {result['temp_cell_c'][0]:.1f}°C")
    print(f"  P_DC:        {result['mw_dc'][0]:.1f} MW (of 130 MW DC capacity)")
    print(f"  P_AC:        {result['mw_ac'][0]:.1f} MW (of 100 MW AC capacity)")

    if 75 <= result['mw_ac'][0] <= 100:
        print(f"\n  PASS: P_AC in expected 75-100 MW range")
    else:
        print(f"\n  WARN: P_AC outside expected 75-100 MW range")

    # And a nighttime case
    times_night = pd.date_range("2026-06-21 06:00", periods=1, freq="h", tz="UTC")
    result_night = compute_plant_power(
        times_utc=times_night,
        lat=33.45, lon=-112.07,
        config=config,
        dswrf=np.array([0.0]),
        beam=np.array([0.0]),
        diffuse=np.array([0.0]),
        temp_k=np.array([293.0]),
    )
    print(f"\nPhoenix same plant @ June 21, 06Z (~11 PM local, night):")
    print(f"  P_AC: {result_night['mw_ac'][0]:.1f} MW (expect 0)")
    if result_night['mw_ac'][0] < 0.01:
        print(f"  PASS")
    else:
        print(f"  WARN: nighttime output should be zero")


if __name__ == "__main__":
    _self_test()
