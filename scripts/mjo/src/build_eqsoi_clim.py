#!/usr/bin/env python3
"""Equatorial SOI reference (laptop, once): ERA5 1991–2020 box statistics and the calibration to
CPC's published monthly EQSOI.

EQSOI (CPC): standardised anomaly of the difference between the area-mean sea-level pressure of
the eastern equatorial Pacific box (5°N–5°S, 130–80°W) and the Indonesia box (5°N–5°S, 90–140°E);
negative in El Niño. Here: daily box-mean difference from the local ERA5 store (wb2 1.5° slp, hPa,
1991–2020), per-calendar-month mean and the standard deviation of the MONTHLY-mean difference
(CPC standardises monthly values), then a linear fit of CPC's series on the ERA5 z-score over the
overlap so live values sit on CPC's scale. Writes data/reference/eqsoi_clim.json.
"""
from __future__ import annotations
import json, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "slp"
OUT = Path(__file__).resolve().parent.parent / "data" / "reference" / "eqsoi_clim.json"
EAST = dict(lat=(-5, 5), lon=(230, 280)); WEST = dict(lat=(-5, 5), lon=(90, 140))
CPC = "https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"

def box(da, b):
    d = da.sel(latitude=slice(b["lat"][0], b["lat"][1])) if da.latitude[0] < da.latitude[-1] else da.sel(latitude=slice(b["lat"][1], b["lat"][0]))
    d = d.sel(longitude=slice(b["lon"][0], b["lon"][1]))
    w = np.cos(np.deg2rad(d.latitude)); return d.weighted(w).mean(("latitude", "longitude"))

def main():
    parts = []
    for f in sorted(STORE.glob("slp_*.nc")):
        y = int(f.stem.split("_")[-1])
        if not 1991 <= y <= 2020: continue
        ds = xr.open_dataset(f); da = ds[list(ds.data_vars)[0]].transpose("time", "latitude", "longitude")
        parts.append((box(da, EAST) - box(da, WEST)).to_series()); ds.close()
    d = pd.concat(parts).sort_index()                                 # daily east−west, hPa
    m = d.resample("MS").mean()
    mon = m.index.month
    clim = {int(k): {"mean": float(m[mon == k].mean()), "sd_monthly": float(m[mon == k].std(ddof=1)),
                     "sd_daily": float(d[d.index.month == k].std(ddof=1))} for k in range(1, 13)}
    # CPC calibration on the overlap
    txt = urllib.request.urlopen(CPC, timeout=60).read().decode()
    rows = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            for i, v in enumerate(p[1:]):
                v = float(v)
                if v < 900: rows.append((pd.Timestamp(int(p[0]), i + 1, 1), v))
    cpc = pd.Series(dict(rows)).sort_index()
    z = pd.Series([(m[t] - clim[t.month]["mean"]) / clim[t.month]["sd_monthly"] for t in m.index], index=m.index)
    both = pd.concat([z.rename("z"), cpc.rename("cpc")], axis=1).dropna()
    b, a = np.polyfit(both.z, both.cpc, 1); r = float(np.corrcoef(both.z, both.cpc)[0, 1])
    print(f"ERA5 z vs CPC EQSOI 1991–2020: n={len(both)} r={r:.3f} slope={b:.3f} intercept={a:.3f}")
    OUT.write_text(json.dumps({"base": "1991-2020", "boxes": {"east": EAST, "west": WEST}, "source": "ERA5 wb2 1.5 deg daily slp (hPa), local store",
                               "clim": clim, "cpc_fit": {"a": float(a), "b": float(b), "r": r, "n": int(len(both))},
                               "note": "EQSOI = a + b * (D - mean_m) / sd_monthly_m, D = east minus west box-mean SLP (hPa); negative in El Nino"}, indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
