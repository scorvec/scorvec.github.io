"""
End-to-end pipeline: USWTDB + HRRR → BA/ISO wind generation forecast.

Operational example
-------------------
    from datetime import datetime, timezone
    from wf_pipeline import run_forecast

    df = run_forecast(
        uswtdb_path="data/uswtdb_v7_0_20240501.csv",
        eia860_path="data/2___Plant_Y2023.xlsx",
        cycle_dt=datetime(2026, 5, 1, 12, tzinfo=timezone.utc),
        fxx_range=range(0, 19),  # 0..18 hour forecast
    )
    # df has columns: ba_code, iso, valid_time, MW

Backcast example (one historical day, all 24 cycles or just 00Z)
----------------------------------------------------------------
    for hour in range(0, 24):
        cycle = datetime(2024, 1, 15, hour, tzinfo=timezone.utc)
        df = run_forecast(..., cycle_dt=cycle, fxx_range=[1])
        # save / accumulate
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from turbine_inventory import load_uswtdb
from curve_assignment import assign_curves, assignment_summary
from aggregation import (
    load_eia860_ba_map, attach_ba_iso, aggregate,
)

log = logging.getLogger(__name__)


def build_inventory(uswtdb_path: str | Path,
                    eia860_path: Optional[str | Path] = None,
                    use_oedb: bool = True) -> pd.DataFrame:
    """Build a forecast-ready inventory: USWTDB + curves + BA mapping."""
    inv = load_uswtdb(uswtdb_path)
    inv = assign_curves(inv, use_oedb=use_oedb)

    eia860 = None
    if eia860_path:
        eia860 = load_eia860_ba_map(eia860_path)
    inv = attach_ba_iso(inv, eia860)

    log.info("Inventory ready: %d turbines, %.0f MW",
             len(inv), inv["t_cap"].sum() / 1000.0)
    log.info("Curve source breakdown:\n%s", assignment_summary(inv))
    return inv


def run_forecast(uswtdb_path: str | Path,
                 cycle_dt: datetime,
                 fxx_range: Iterable[int],
                 eia860_path: Optional[str | Path] = None,
                 use_oedb: bool = True,
                 aggregation_level: str = "ba") -> pd.DataFrame:
    """End-to-end: build inventory, fetch HRRR, forecast, aggregate."""
    from hrrr_winds import fetch_hrrr_forecast, interp_to_points
    from forecast import forecast_fleet, ForecastConfig

    inv = build_inventory(uswtdb_path, eia860_path, use_oedb=use_oedb)

    ds = fetch_hrrr_forecast(cycle_dt, list(fxx_range))
    samples = interp_to_points(
        ds,
        lats=inv["ylat"].to_numpy(),
        lons=inv["xlong"].to_numpy(),
    )
    inv = inv.reset_index(drop=True)
    inv["point_id"] = inv.index

    fc = forecast_fleet(inv, samples, cfg=ForecastConfig())
    return aggregate(fc, level=aggregation_level)
