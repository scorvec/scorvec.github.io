#!/usr/bin/env python3
"""ERA5 1991-2020 reference of the live by-range mountain torque (torque_map_anim.py): each range's day-mean
anomaly in the forecast's own form, and its 1-sigma -> data/reference/torque_range_sd.nc.

Input: torque_ranges5_era5_1991_2020.csv.gz (torque-history.yml, mode ranges5: torque_history_era5.py --ranges5 at
00 and 12 UTC every day), the site-exact 5-degree box sums of h dp_s/dlam over each RANGES box and the globe.

The forecast's day d is the 24 h ending at init + d days: the trapezoid of the 12-hourly values over that window
(weights 1/4, 1/2, 1/4) minus the ERA5 map climatology (torque_map_clim_coeffs.nc, the mean of its 00 and 12 UTC
coefficients) at the window's common-year day. The climatology is linear, so its box sum is the box-summed
coefficients evaluated at that day. ERA5 days here are the 00-00 UTC windows, the forecast days of a 00 UTC run (a
12 UTC run's are the same quantity half a day later): 1991-01-01 .. 2020-12-30, since the 31 December window would
need 1 January 2021.

torque_range_sd.nc is the 1-sigma of that anomaly per range, all days pooled: the +-1 and +-2 sigma bands of the
by-range figure (a day mean's sigma, smaller than that of single 00/12 UTC instants, which keep the 00-12 UTC tide
difference in the anomaly).

    python src/build_torque_range_sd.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from build_torque_map_clim import harmonics
from torque_history_era5 import RANGES

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
TABLE = REF / "torque_ranges5_era5_1991_2020.csv.gz"
MAP_CLIM = REF / "torque_map_clim_coeffs.nc"
OUT = REF / "torque_range_sd.nc"
Y0, Y1 = 1991, 2020
A, HU = 6.371e6, 1e18
# torque_history_era5 codes -> torque_map_anim.RANGES names (the globe is torque_range_sd.nc's "__global__")
NAMES = {
    "HIM": "Himalaya/Tibet",
    "ROC": "Rockies/W. N.America",
    "AND": "Andes",
    "GRL": "Greenland",
    "ANT": "Antarctica",
    "ALP": "Alps/Europe",
    "glob": "__global__",
}
BOXES = RANGES | {"glob": (0, 360, -90, 90)}


def _clim_boxes() -> np.ndarray:
    """(5, range) harmonic coefficients of each box's climatological torque, Hadley: the map climatology's mountain
    density (mean of its hours, as torque_map_anim._map_clim) times the block area, summed over the box's blocks."""
    with xr.open_dataset(MAP_CLIM) as c:
        coef = c["mountain"].mean("hour").values  # (5, lat, lon)
        clat, clon = c.latitude.values, c.longitude.values
    dA5 = (A**2) * np.cos(np.deg2rad(clat)) * np.deg2rad(abs(clon[1] - clon[0])) * np.deg2rad(abs(clat[1] - clat[0]))
    out = []
    for lo0, lo1, la0, la1 in (BOXES[k] for k in NAMES):
        ym = (clat >= la0) & (clat <= la1)
        xm = (clon >= lo0) & (clon <= lo1)
        out.append((coef * dA5[None, :, None])[:, ym][:, :, xm].sum((1, 2)) / HU)
    return np.stack(out, 1)


def day_anomalies() -> pd.DataFrame:
    """ERA5 day-mean mountain-torque anomaly (Hadley) of every range and the globe, the forecast's form, indexed by
    the 00-00 UTC window's date. Refuses an incomplete or non-finite table."""
    tab = pd.read_csv(TABLE, parse_dates=["time"]).set_index("time")
    days = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
    want = pd.DatetimeIndex(np.stack([days, days + pd.Timedelta(hours=12)], 1).ravel())
    if not tab.index.equals(want):
        raise SystemExit(f"{TABLE.name}: time axis is not 00 and 12 UTC every day {Y0}-{Y1} ({len(tab)} rows)")
    x = tab[[f"mtn5_{k}" for k in NAMES]].to_numpy(float)
    if not np.isfinite(x).all():
        raise SystemExit(f"{TABLE.name}: {(~np.isfinite(x)).sum()} non-finite values")
    x00, x12 = x[0::2], x[1::2]
    day = 0.25 * x00[:-1] + 0.5 * x12[:-1] + 0.25 * x00[1:]
    return pd.DataFrame(day - harmonics(days[:-1]) @ _clim_boxes(), index=days[:-1], columns=list(NAMES))


def main() -> int:
    an = day_anomalies()
    sd = an.std(ddof=0)
    old = {}
    if OUT.exists():
        with xr.open_dataset(OUT) as o:
            old = dict(zip(o["range"].values.tolist(), o["sd"].values.tolist()))
    for k, nm in NAMES.items():
        was = f" (file had {old[nm]:.2f})" if nm in old else ""
        print(f"  {nm:22s} mean {an[k].mean():+6.2f}  sigma {sd[k]:6.2f} Hadley{was}", flush=True)
    xr.Dataset(
        {"sd": ("range", [float(sd[k]) for k in NAMES])},
        coords={"range": list(NAMES.values())},
        attrs={
            "note": "1-sigma of the day-mean mountain-torque anomaly per barrier, the live forecast's form: 5-degree "
            "h dp_s/dlam box sum, trapezoid of 00/12 UTC over the 00-00 UTC day, minus the ERA5 map climatology "
            "(mean of its 00/12 UTC coefficients); ERA5 1991-01-01..2020-12-30 (build_torque_range_sd.py)",
            "n_samples": len(an),
        },
    ).to_netcdf(OUT)
    print(f"wrote {OUT} ({len(an)} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
