#!/usr/bin/env python3
"""Station calibration for the model Troup SOI (laptop, once; ~1 min + the ARCO tide pass).

The ensembles give Tahiti−Darwin MSL at the nearest grid points; the observed SOI is Troup's
station index. Instead of anchoring every run to the recent observed level (which moved the whole
forecast from cycle to cycle — user: "whipped around by the bias adjustment"), this builds a STATIC
map from grid-point difference to station difference:

  1. ERA5 1991–2020 daily MSLP (local store, 1.5°) at the points nearest Tahiti and Darwin →
     daily grid difference D_g; the LongPaddock daily file gives the station difference D_s.
     Per calendar month: D_s = a_m + b_m · D_g (least squares), with r and the residual SD.
  2. Tide: the ensembles' 24-hourly steps all fall at the cycle hour, so an instantaneous D_g
     carries the semidiurnal tide against the daily means of step 1. From ARCO hourly ERA5
     (one year) per month: mean of (D_g at 00/12 UTC − the day's mean D_g).

Output data/reference/soi_clim.json; soi_forecast.py applies
  D_s(model) = a_m + b_m · (D_g(model, hour) − tide[hour][m])   →   Troup SOI with the station
normals, no per-run offset. --tide-only / --no-tide split the two passes.
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "era5")); sys.path.insert(0, str(HERE.parents[1] / "lib"))
REF = HERE.parent / "data" / "reference"; OUT = REF / "soi_clim.json"
STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "slp"
TAHITI = dict(latitude=-17.5, longitude=210.4); DARWIN = dict(latitude=-12.4, longitude=130.9)

def station_diff() -> pd.Series:
    import soi_forecast as SF
    obs = SF.fetch_obs(HERE.parent / "data" / "soi" / "DailySOI.txt")
    return (obs["Tahiti"] - obs["Darwin"]).dropna()

def grid_diff_daily(y0=1991, y1=2020) -> pd.Series:
    parts = []
    for y in range(y0, y1 + 1):
        e = xr.open_dataset(STORE / f"slp_{y}.nc"); v = e[list(e.data_vars)[0]]
        d = v.sel(**TAHITI, method="nearest") - v.sel(**DARWIN, method="nearest")
        parts.append(d.to_series())
    s = pd.concat(parts).sort_index(); s.index = pd.DatetimeIndex(s.index).normalize()
    return s[~s.index.duplicated()]

def calibrate(dg: pd.Series, ds: pd.Series) -> dict:
    df = pd.DataFrame({"g": dg, "s": ds}).dropna(); df["m"] = df.index.month
    out = {}
    for m, g in df.groupby("m"):
        b, a = np.polyfit(g["g"].values, g["s"].values, 1); res = g["s"].values - (a + b * g["g"].values)
        out[str(m)] = dict(a=round(float(a), 3), b=round(float(b), 4), r=round(float(np.corrcoef(g["g"], g["s"])[0, 1]), 3),
                           n=int(len(g)), resid_sd=round(float(res.std()), 3), grid_mean=round(float(g["g"].mean()), 3), grid_sd=round(float(g["g"].std()), 3))
        print(f"  month {m:2d}: D_s = {a:+.2f} + {b:.3f}·D_g  r {out[str(m)]['r']:.3f}  resid sd {out[str(m)]['resid_sd']:.2f} hPa  (n {len(g)})")
    return out

def tide(year=2019) -> dict:
    import era5_store as es
    t0 = time.time(); ds = es._arco()["mean_sea_level_pressure"]
    if float(ds.longitude.min()) < 0: ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    e = xr.open_dataset(STORE / f"slp_{year}.nc"); dv = e[list(e.data_vars)[0]]
    dd = (dv.sel(**TAHITI, method="nearest") - dv.sel(**DARWIN, method="nearest")).to_series()
    pt = ds.sel(**TAHITI, method="nearest"); pd_ = ds.sel(**DARWIN, method="nearest")
    rows = []
    for hh in (0, 12):
        times = pd.date_range(f"{year}-01-01 {hh:02d}:00", f"{year}-12-31 {hh:02d}:00", freq="D")
        g = (pt.sel(time=times).load() - pd_.sel(time=times).load()).values / 100.0
        for t, v in zip(times, g):
            rows.append((t.month, hh, float(v - dd.loc[t.normalize()])))
        print(f"  hour {hh:02d}Z done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["month", "hour", "off"])
    out = {str(hh): {str(m): round(float(v), 3) for m, v in df[df.hour == hh].groupby("month")["off"].mean().items()} for hh in (0, 12)}
    for hh in out: print(f"  tide {hh}Z:", out[hh])
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tide-only", action="store_true"); ap.add_argument("--no-tide", action="store_true"); ap.add_argument("--year", type=int, default=2019)
    a = ap.parse_args()
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    if not a.tide_only:
        dg = grid_diff_daily(); ds = station_diff()
        cur.update(base="1991-2020", source="ERA5 wb2 1.5 deg daily slp (local store) vs LongPaddock station pressures",
                   points={"tahiti": TAHITI, "darwin": DARWIN}, calib=calibrate(dg, ds))
        OUT.write_text(json.dumps(cur, indent=1)); print(f"  wrote {OUT}")
    if not a.no_tide:
        cur["tide"] = tide(a.year); cur["tide_year"] = a.year
        OUT.write_text(json.dumps(cur, indent=1)); print(f"  wrote {OUT} (tide)")

if __name__ == "__main__":
    main()
