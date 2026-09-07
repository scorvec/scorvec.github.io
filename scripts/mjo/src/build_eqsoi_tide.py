#!/usr/bin/env python3
"""Tidal offset of the Equatorial SOI box difference at a fixed UTC hour (laptop, once).

The ensembles' 24-hourly steps all fall at the cycle hour (00 or 12 UTC); the semidiurnal
atmospheric tide then puts the east box (130–80°W, local afternoon at 00 UTC) near its pressure
minimum and the west box (90–140°E, local morning) near its maximum, so an instantaneous
east-minus-west difference is biased by ~2 hPa against the daily means ERA5's climatology is
built from. This reads hourly ERA5 MSLP from ARCO at 00 and 12 UTC for one year, forms the box
difference, and records per month the mean of (D at that hour − the day's mean D from the local
daily store). Output: data/reference/eqsoi_tide.json  {hour: {month: offset_hPa}}.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import era5_store as es
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = json.loads((REF / "eqsoi_clim.json").read_text()); B = CLIM["boxes"]
YEAR = 2019
STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "slp"

def boxmean(da, b):
    d = da.sel(latitude=slice(b["lat"][1], b["lat"][0]) if float(da.latitude[0]) > float(da.latitude[-1]) else slice(b["lat"][0], b["lat"][1]), longitude=slice(b["lon"][0], b["lon"][1]))
    return float(d.weighted(np.cos(np.deg2rad(d.latitude))).mean(("latitude", "longitude")))

def main():
    t0 = time.time(); ds = es._arco()["mean_sea_level_pressure"]
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude") if float(ds.longitude.min()) < 0 else ds
    e = xr.open_dataset(STORE / f"slp_{YEAR}.nc"); dv = e[list(e.data_vars)[0]].transpose("time", "latitude", "longitude")
    def daily_D(day):
        f = dv.sel(time=day); return boxmean(f, B["east"]) - boxmean(f, B["west"])
    rows = []
    for day in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D"):
        dd = daily_D(day)
        for hh in (0, 12):
            f = ds.sel(time=day + pd.Timedelta(hours=hh)).load() / 100.0
            rows.append((day.month, hh, boxmean(f, B["east"]) - boxmean(f, B["west"]) - dd))
        if day.day == 1: print(f"  {day:%Y-%m} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["month", "hour", "off"])
    out = {str(hh): {str(m): round(float(v), 3) for m, v in df[df.hour == hh].groupby("month")["off"].mean().items()} for hh in (0, 12)}
    (REF / "eqsoi_tide.json").write_text(json.dumps({"year": YEAR, "note": "mean of (box difference at hour − daily-mean box difference), hPa, ERA5 ARCO hourly vs the daily store", "offset": out}, indent=1))
    print("offsets 00Z:", out["0"]); print("offsets 12Z:", out["12"])

if __name__ == "__main__":
    main()
