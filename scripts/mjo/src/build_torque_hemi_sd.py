#!/usr/bin/env python3
"""The climatological spread of the net surface-torque anomaly, globe and each hemisphere, by day of year.

torque_map_anim.py's budget series shows each forecast day's mean friction + mountain torque anomaly against ERA5
1991-2020, with IFS-ENS's own surface stress as the friction and h dp_s/dlambda as the mountain torque. Its ±1σ and
±2σ bands are this: the standard deviation of the same quantity in ERA5 - daily means (of four synoptic hours) of
ERA5's own turbulent surface stress torque and its 0.25-degree mountain torque, summed over the latitude bands of
each hemisphere (torque_daily_era5.csv.gz, torque_history_era5.py) - after the seasonal cycle (mean and two annual
harmonics) is removed, pooled over the 30 years and the days within ±POOL of each common-year day.

    python src/build_torque_hemi_sd.py        # -> data/reference/torque_hemi_sd.nc
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from doycal import noleap_doy

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
SRC = REF / "torque_daily_era5.csv.gz"
OUT = REF / "torque_hemi_sd.nc"
POOL = 15
BANDS = {"NH": ("b0030n", "b3060n", "b6090n"), "SH": ("b9060s", "b6030s", "b3000s")}


def main() -> int:
    d = pd.read_csv(SRC, parse_dates=["date"]).set_index("date")
    d = d[(d.index.year >= 1991) & (d.index.year <= 2020)]
    net = {
        "G": d["fric_glob"] + d["mtn_glob"],
        **{h: sum(d[f"fric_{b}"] + d[f"mtn_{b}"] for b in bs) for h, bs in BANDS.items()},
    }
    doy = noleap_doy(d.index)
    w = 2 * np.pi * doy / 365.0
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    sd = np.full((3, 366), np.nan)
    for i, k in enumerate(("G", "NH", "SH")):
        y = net[k].values
        res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        for day in range(1, 366):  # slots 1..365 are common-year days; slot 366 repeats 31 Dec
            band = np.abs(((doy - day + 182.5) % 365) - 182.5) <= POOL
            sd[i, day - 1] = res[band].std(ddof=1)
        sd[i, 365] = sd[i, 364]
        print(f"  {k}: σ {np.nanmin(sd[i]):.1f}-{np.nanmax(sd[i]):.1f} Hadley (all days {res.std(ddof=1):.1f})")
    xr.Dataset(
        {"sd": (("region", "doy"), sd)},
        coords={"region": ["G", "NH", "SH"], "doy": np.arange(1, 367)},
        attrs={
            "units": "Hadley (1e18 N m)",
            "note": "1-sigma of the daily-mean (friction + mountain) torque anomaly, ERA5 1991-2020: ERA5's own "
            f"turbulent surface stress and 0.25-degree mountain torque, seasonal cycle removed, pooled +-{POOL} days "
            "of year (build_torque_hemi_sd.py, from torque_daily_era5.csv.gz)",
        },
    ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
