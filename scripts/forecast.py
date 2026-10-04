"""
Per-turbine power forecast: hub-height adjustment, density correction,
curve application, losses.

Step-by-step transform from HRRR 80m winds to per-turbine power:

  1. ws_hh = ws80 * (t_hh / 80) ** alpha           (hub-height adjust)
  2. rho   = P / (R_d * T)                         (air density at surface)
     rho_hh = rho * exp(-g * (t_hh - 0) / (R_d * T))   (small correction)
  3. ws_eq = ws_hh * (rho_hh / 1.225) ** (1/3)      (IEC 61400-12 density eq.)
  4. P     = curve.power(ws_eq)                    (kW per turbine)
  5. P_net = P * (1 - loss_factor)                 (availability + wake)

Notes
-----
- The shear exponent `alpha` defaults to 1/7 (~0.143), the engineering
  estimate for open terrain. For more accuracy, fit per-region from
  measured 10m / 80m HRRR ratios, or condition on stability. For most
  turbines in the modern fleet `t_hh` ≥ 80m, so the adjustment is small
  (e.g. 95m → factor 1.025).
- The density correction follows IEC 61400-12-1 Annex G: equivalent wind
  speed at standard density is `ws * (rho / 1.225)^(1/3)`. Standard density
  is 1.225 kg/m^3 at 15°C, sea level.
- Loss factor of 0.16 is a fleet-average rule of thumb (NREL ATB 2023):
  ~10% wake + ~6% availability/electrical/icing/curtailment. Tune per BA
  against EIA-923 monthly generation if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

R_DRY_AIR = 287.058       # J/(kg*K), specific gas constant for dry air
G = 9.80665               # m/s^2
RHO_STD = 1.225           # kg/m^3, IEC standard air density


@dataclass
class ForecastConfig:
    shear_exponent: float = 1.0 / 7.0
    loss_factor: float = 0.16
    rho_std: float = RHO_STD


def air_density(t2m_K: np.ndarray, psfc_Pa: np.ndarray) -> np.ndarray:
    """Surface air density via ideal gas law.

    HRRR `psfc` is true surface pressure (Pa) and `t2m` is 2m temperature.
    For hub-height density we apply a small hypsometric correction.
    """
    return psfc_Pa / (R_DRY_AIR * t2m_K)


def density_at_height(rho_sfc: np.ndarray, t2m_K: np.ndarray,
                      height_m: float) -> np.ndarray:
    """Hypsometric scaling of density from surface to a given height.

    Approximate for tens of meters: rho(h) = rho_sfc * exp(-g h / R_d T).
    """
    return rho_sfc * np.exp(-G * height_m / (R_DRY_AIR * t2m_K))


def hub_height_wind(ws_80: np.ndarray, hub_height_m: np.ndarray,
                    alpha: float = 1.0 / 7.0) -> np.ndarray:
    """Power-law shear from 80 m to hub height.

    ws_hh = ws_80 * (hub_height / 80)^alpha.

    For hub heights below 80 m (rare in modern fleet but exists for
    legacy turbines), the same formula extrapolates downward.
    """
    return ws_80 * (np.asarray(hub_height_m) / 80.0) ** alpha


def density_equivalent_wind(ws_hh: np.ndarray,
                            rho_hh: np.ndarray,
                            rho_std: float = RHO_STD) -> np.ndarray:
    """IEC 61400-12-1 density correction: ws_eq = ws * (rho / rho_std)^(1/3)."""
    return ws_hh * (rho_hh / rho_std) ** (1.0 / 3.0)


def forecast_turbine_power(turbine_row: pd.Series,
                           winds: pd.DataFrame,
                           cfg: ForecastConfig = ForecastConfig()
                           ) -> pd.DataFrame:
    """Run the full transform for one turbine across all forecast hours.

    Parameters
    ----------
    turbine_row : Series
        One row of the inventory DataFrame, after `assign_curves`.
        Must include t_cap, t_hh, curve_obj.
    winds : DataFrame
        Time-indexed (or long-format) wind data for this turbine, with
        columns ws80, t2m, psfc.

    Returns
    -------
    DataFrame with columns ws_hh, rho_hh, ws_eq, power_kW, power_net_kW.
    """
    curve = turbine_row["curve_obj"]
    hh = float(turbine_row["t_hh"]) if pd.notna(turbine_row.get("t_hh")) else 80.0

    ws80 = winds["ws80"].to_numpy()
    t2m = winds["t2m"].to_numpy()
    psfc = winds["psfc"].to_numpy()

    rho_sfc = air_density(t2m, psfc)
    rho_hh = density_at_height(rho_sfc, t2m, hh)

    ws_hh = hub_height_wind(ws80, hh, alpha=cfg.shear_exponent)
    ws_eq = density_equivalent_wind(ws_hh, rho_hh, rho_std=cfg.rho_std)

    p_gross = curve.power(ws_eq)
    p_net = p_gross * (1.0 - cfg.loss_factor)

    out = winds.copy()
    out["ws_hh"] = ws_hh
    out["rho_hh"] = rho_hh
    out["ws_eq"] = ws_eq
    out["power_kW"] = p_gross
    out["power_net_kW"] = p_net
    return out


def forecast_fleet(inventory: pd.DataFrame,
                   wind_samples: pd.DataFrame,
                   cfg: ForecastConfig = ForecastConfig()) -> pd.DataFrame:
    """Vectorized fleet forecast.

    Parameters
    ----------
    inventory : DataFrame
        After `assign_curves`. Indexed [0..N-1]; column `point_id` should
        match the `point_id` column in `wind_samples`.
    wind_samples : DataFrame
        Long-format with columns: point_id, valid_time, ws80, t2m, psfc.

    Returns
    -------
    DataFrame indexed (case_id, valid_time) with power_net_kW etc.
    """
    if "point_id" not in inventory.columns:
        # Default: row index is the point_id
        inventory = inventory.copy()
        inventory["point_id"] = inventory.index

    # Pull BA/ISO/state columns through if they exist on the inventory
    inv_cols = ["point_id", "case_id", "t_cap", "t_hh", "curve_obj",
                "eia_id", "p_name", "t_state"]
    for opt in ("ba_code", "iso", "ba_source"):
        if opt in inventory.columns:
            inv_cols.append(opt)

    merged = wind_samples.merge(
        inventory[inv_cols],
        on="point_id", how="inner",
    )

    # Compute density once
    rho_sfc = air_density(merged["t2m"].to_numpy(), merged["psfc"].to_numpy())
    hh = merged["t_hh"].fillna(80.0).to_numpy()
    rho_hh = density_at_height(rho_sfc, merged["t2m"].to_numpy(), hh)

    ws_hh = hub_height_wind(merged["ws80"].to_numpy(), hh,
                            alpha=cfg.shear_exponent)
    ws_eq = density_equivalent_wind(ws_hh, rho_hh, rho_std=cfg.rho_std)

    # Apply each turbine's curve. Curves may have different grids, so we
    # loop on unique curves to keep this vectorized within each curve.
    p_gross = np.zeros_like(ws_eq)
    curves = merged["curve_obj"].to_numpy()

    # Group rows by id(curve) — same PowerCurve object, vectorize together.
    curve_ids = np.array([id(c) for c in curves])
    for cid in np.unique(curve_ids):
        mask = curve_ids == cid
        c = curves[np.argmax(mask)]
        p_gross[mask] = c.power(ws_eq[mask])

    merged["ws_hh"] = ws_hh
    merged["rho_hh"] = rho_hh
    merged["ws_eq"] = ws_eq
    merged["power_kW"] = p_gross
    merged["power_net_kW"] = p_gross * (1.0 - cfg.loss_factor)
    return merged
