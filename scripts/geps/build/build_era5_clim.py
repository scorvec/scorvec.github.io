#!/usr/bin/env python3
"""ERA5 1991-2020 day-of-year normal, built from the local store.

Reads ~/era5_store/wb2_1p5_daily_global (the store managed by
scripts/era5/wb2_daily_store.py) rather than re-streaming WeatherBench2 — the
data was already there, and duplicating it would have been ~80 GB of pointless
traffic. The store holds daily means of the four synoptic hours, so no diurnal
aliasing to worry about.

Purpose: re-reference the maps onto a standard normal. The GEPS8 model
climatology is the right thing for removing model drift and stays the default,
but 2001-2020 is not a base anyone quotes. With this,

    A_era5(L) = [F(L) - M(L)] + [M(L=0) - O_era5]

keeps the lead-dependent drift removal and moves the result into the observed
frame. The second term is a bias-plus-base-period shift and is smooth in space,
which is why 1.5 deg is adequate for it while the anomaly itself stays at 1 deg.

    python build_era5_clim.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS                             # noqa: E402  laptop-only inputs/outputs
STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global"
OUTFMT = "era5_clim_{y0}-{y1}.nc"
# our tag -> (store var, unit conversion to match the GEPS live fields)
VARS = {"t2m": ("t2m", 1.0, 0.0), "mslp": ("slp", 100.0, 0.0),   # store slp is hPa
        "zg500": ("z500", 1.0, 0.0),
        "u850": ("u850", 1.0, 0.0), "u200": ("u200", 1.0, 0.0),
        "v850": ("v850", 1.0, 0.0), "v200": ("v200", 1.0, 0.0)}
STEP, HALFWIN = 5, 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()
    centres = np.arange(1, 366, STEP)
    out, coords = {}, None
    for tag, (var, mul, add) in VARS.items():
        fs = [STORE / var / f"{var}_{y}.nc" for y in range(a.y0, a.y1 + 1)]
        fs = [f for f in fs if f.exists()]
        if not fs:
            print(f"  {tag}: no files under {STORE / var} — skipped")
            continue
        acc = None
        cnt = np.zeros(len(centres), dtype=int)
        for f in fs:
            d = xr.open_dataset(f)
            v = d[var]
            # the store writes (time, longitude, latitude), not the usual
            # (time, lat, lon) — transpose explicitly rather than trusting order
            la = [c for c in v.dims if "lat" in c][0]
            lo = [c for c in v.dims if "lon" in c][0]
            v = v.transpose("time", la, lo)
            doy = pd.DatetimeIndex(d.time.values).dayofyear.values
            arr = v.values.astype("float64") * mul + add
            if acc is None:
                acc = np.zeros((len(centres),) + arr.shape[1:], dtype="float64")
                coords = {"lat": v[la].values, "lon": v[lo].values}
            for bi, c in enumerate(centres):
                dist = np.minimum(np.abs(doy - c), 365 - np.abs(doy - c))
                sel = np.where(dist <= HALFWIN)[0]
                if sel.size:
                    acc[bi] += arr[sel].sum(0)
                    cnt[bi] += sel.size
            d.close()
        m = (acc / np.maximum(cnt, 1)[:, None, None]).astype("float32")
        out[tag] = (("doy", "lat", "lon"), m)
        print(f"  {tag:6s} from {len(fs)} years, {cnt.min()}-{cnt.max()} days per bin, "
              f"mean {m.mean():.4g}", flush=True)
    if not out:
        raise SystemExit("nothing built — is the global store populated?")
    xr.Dataset(out, coords={"doy": centres, **coords},
               attrs={"title": f"ERA5 {a.y0}-{a.y1} day-of-year normal (WMO base)",
                      "source": str(STORE),
                      "note": "daily means of the 4 synoptic hours, 1.5 deg; used to "
                              "re-reference GEPS anomalies onto the observed normal",
                      "units": "t2m K, mslp Pa, zg500 m, winds m/s"}).to_netcdf(_LS / "climatology" / OUTFMT.format(y0=a.y0, y1=a.y1))
    OUT = _LS / "climatology" / OUTFMT.format(y0=a.y0, y1=a.y1)
    print(f"\nwrote {OUT.name} ({OUT.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
