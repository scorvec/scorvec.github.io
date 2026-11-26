#!/usr/bin/env python3
"""Normals for the polar-cap dynamic-tropopause series (dt_cap.py), from the local ERA5 store.

Daily geopotential height on dt_cap.LEVS at 1.5 degrees (wb2_1p5_daily/zplev), the same tropopause and cap means as the
forecast (dt_cap.dt_height, dt_cap.cap_means). Normal = the latest WINTERS full winters (July-June years), no trend
term: per day of a 366-slot calendar, percentiles of the days within +-HALF days, smoothed over 15 days.
    python src/build_dt_cap_ref.py            # -> scripts/mjo/data/reference/dt_cap_clim.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
sys.path.insert(0, str(Path(__file__).parent))
import dt_cap as D
from localdirs import env_dir

WINTERS = 30
HALF = 7
SMOOTH = 15
PERCENTILES = ("2.5", "10", "50", "90", "97.5")
FIRST_YEAR = 1980


def daily() -> pd.DataFrame:
    """Daily cap means (m) from every zplev year file from FIRST_YEAR."""
    store = env_dir("ERA5_STORE", "wb2_1p5_daily", "zplev")
    parts = []
    for path in sorted(store.glob("zplev_*.nc")):
        if int(path.stem[-4:]) < FIRST_YEAR:
            continue
        z = xr.open_dataset(path)["zplev"].transpose("time", "level", "latitude", "longitude")
        z = z.assign_coords(latitude=np.round(z.latitude.values.astype(float), 3))
        z = z.assign_coords(longitude=np.round(z.longitude.values.astype(float), 3))
        z = z.sel(level=D.LEVS, latitude=D.LAT, longitude=D.LON)
        caps = D.cap_means(D.dt_height(z.values, D.LAT, D.LON), D.LAT, D.LON)
        parts.append(pd.DataFrame(caps, index=pd.DatetimeIndex(z.time.values)))
        print(path.stem, flush=True)
    return pd.concat(parts).sort_index()


def normals(df: pd.DataFrame, last: int) -> dict:
    t = df.index
    wy = np.asarray(t.year + (t.month >= 7))
    keep = (wy > last - WINTERS) & (wy <= last)
    slot = doy366(t.values)
    out = {}
    for name in df:
        x, s = df[name].values[keep], slot[keep]
        q = np.empty((len(PERCENTILES), 366))
        for k in range(366):
            near = np.abs((s - k + 183) % 366 - 183) <= HALF
            q[:, k] = np.percentile(x[near], [float(p) for p in PERCENTILES])
        kern = np.ones(SMOOTH) / SMOOTH
        q = np.stack([np.convolve(np.r_[r[-SMOOTH:], r, r[:SMOOTH]], kern, "same")[SMOOTH:-SMOOTH] for r in q])
        out[name] = {p: [round(float(v), 1) for v in q[i]] for i, p in enumerate(PERCENTILES)}
    return out


def main() -> int:
    df = daily()
    t = df.index
    full = sorted({y for y in t.year + (t.month >= 7) if ((t.year + (t.month >= 7)) == y).sum() >= 365})
    last = full[-1]
    ref = {
        "window": [last - WINTERS + 1, last],
        "slots": 366,
        "percentiles": list(PERCENTILES),
        "method": "2-PVU surface from ERA5 daily geopotential (wb2 1.5 deg), 700-50 hPa; cos-weighted 65-87N means",
        "offset": {"cap": 0.0, "atl": 0.0},
        **normals(df, last),
    }
    out = Path(__file__).resolve().parents[1] / "data/reference/dt_cap_clim.json"
    out.write_text(json.dumps(ref, separators=(",", ":")))
    djf = df[t.month.isin([12, 1, 2])]
    print(f"wrote {out.name}: winters {ref['window']}; DJF mean cap {djf.cap.mean():.0f} m, atl {djf.atl.mean():.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
