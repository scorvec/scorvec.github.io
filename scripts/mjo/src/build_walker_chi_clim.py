#!/usr/bin/env python3
"""Reference for the weekly Walker-departure maps (walker_chi_weekly.py) — built once on the laptop.

Tropical velocity potential χ at 200 and 850 hPa from ERA5 1991–2020 monthly u/v (the local
era5_store, 1.5°), solved with the same spherical-harmonic inversion (lmax 42) and kept on the same
DH2 strip (|lat| ≤ 32°) as the live AIFS-ENS analyses, so climatology, regression and observation
share one grid. Per grid point, the same ENSO-first regression as the W (band-mean) reference:
monthly χ200 anomalies, detrended per calendar month, regressed (±1 month window, 90 samples) on
Niño-3.4 and its 3-month lag first, then on the Indian and Atlantic indices as the residuals after
ENSO — the indices come from walker_basins_clim.nc (idx_train), so the two references agree exactly.

Outputs: scripts/mjo/data/reference/walker_chi_clim.nc
    python src/build_walker_chi_clim.py        (~10 min: 720 velocity-potential solves)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from build_walker_basins_clim import monthly_uv, detrend_by_month, PREDICTORS, Y0, Y1   # noqa: E402
import walker_chi as wc                                                                    # noqa: E402

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "walker_chi_clim.nc"
WREF = REF / "walker_basins_clim.nc"


def main() -> int:
    t0 = time.time()
    u200, v200, u850, v850 = (monthly_uv(v) for v in ("u200", "v200", "u850", "v850"))
    print(f"  monthly u/v loaded {dict(u200.sizes)} ({time.time() - t0:.0f}s)", flush=True)
    times = u200.time.values
    chis = {}
    for lev, (u, v) in ((200, (u200, v200)), (850, (u850, v850))):
        rows = []
        for k in range(u.sizes["time"]):
            chi, lat, lon = wc.chi_strip(u.isel(time=k), v.isel(time=k))
            rows.append(chi)
            if k % 60 == 0:
                print(f"    χ{lev}: {k}/{u.sizes['time']} ({time.time() - t0:.0f}s)", flush=True)
        chis[lev] = np.array(rows, dtype="float64")                    # (t, lat, lon)
    # harmonic day-of-year climatology (monthly means sit mid-month), per level
    doy = pd.DatetimeIndex(times).dayofyear.values + 14
    w_ = 2 * np.pi * doy / 365.25
    B = np.stack([np.ones_like(w_), np.cos(w_), np.sin(w_), np.cos(2 * w_), np.sin(2 * w_)], -1)
    coef_clim = np.zeros((5, 2, lat.size, lon.size)); slope = np.zeros((12, 2, lat.size, lon.size)); anom = {}
    for i, lev in enumerate((200, 850)):
        c, *_ = np.linalg.lstsq(B, chis[lev].reshape(len(times), -1), rcond=None)
        coef_clim[:, i] = c.reshape(5, lat.size, lon.size)
        a, _, s = detrend_by_month(chis[lev], times)
        slope[:, i] = s; anom[lev] = a
    # regression of χ200 anomalies, ENSO first, basins as residuals — indices from the W reference
    wref = xr.open_dataset(WREF)
    tW = pd.DatetimeIndex(wref.time.values); tC = pd.DatetimeIndex(times)
    pos = {(d.year, d.month): i for i, d in enumerate(tW)}
    ii = np.array([pos[(d.year, d.month)] for d in tC])
    X = {k: wref.idx_train.sel(pred=k).values[ii].astype("float64") for k in PREDICTORS}
    mos = tC.month.values
    Y = anom[200].reshape(len(times), -1)
    coef = np.zeros((12, len(PREDICTORS), Y.shape[1])); r2 = np.zeros((12, 2, Y.shape[1]))
    for m in range(1, 13):
        win = np.isin(mos, [(m - 2) % 12 + 1, m, m % 12 + 1])
        ok = win & np.all([np.isfinite(X[k]) for k in PREDICTORS], axis=0)
        E = np.stack([X["n34"][ok], X["n34_lag3"][ok]], 1); Ec = E - E.mean(0)
        Yc = Y[ok] - Y[ok].mean(0)
        R = {}
        for k in ("dmi", "iob", "atl3", "tna"):
            xk = X[k][ok] - X[k][ok].mean()
            g, *_ = np.linalg.lstsq(Ec, xk, rcond=None); R[k] = xk - Ec @ g
        Xf = np.column_stack([Ec, R["dmi"], R["iob"], R["atl3"], R["tna"]])
        b_e, *_ = np.linalg.lstsq(Ec, Yc, rcond=None); b_f, *_ = np.linalg.lstsq(Xf, Yc, rcond=None)
        ss = (Yc ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[m - 1, 0] = 1 - ((Yc - Ec @ b_e) ** 2).sum(0) / ss
            r2[m - 1, 1] = 1 - ((Yc - Xf @ b_f) ** 2).sum(0) / ss
        coef[m - 1] = b_f
        print(f"    month {m:2d}: median R² ENSO {np.nanmedian(r2[m - 1, 0]):.2f} / full {np.nanmedian(r2[m - 1, 1]):.2f}", flush=True)
    ds = xr.Dataset(
        {"chi_coef": (("harm", "level", "latitude", "longitude"), coef_clim.astype("float32")),
         "chi_slope": (("month", "level", "latitude", "longitude"), slope.astype("float32")),
         "coef": (("month", "pred", "latitude", "longitude"), coef.reshape(12, len(PREDICTORS), lat.size, lon.size).astype("float32")),
         "r2": (("month", "stage", "latitude", "longitude"), r2.reshape(12, 2, lat.size, lon.size).astype("float32"))},
        coords={"harm": ["mean", "cos1", "sin1", "cos2", "sin2"], "level": [200, 850], "latitude": lat, "longitude": lon,
                "month": np.arange(1, 13), "pred": PREDICTORS, "stage": ["enso", "full"]},
        attrs={"base": f"{Y0}-{Y1}", "lmax": wc.LMAX, "units": "m2 s-1",
               "note": "chi from ERA5 monthly u/v (1.5 deg) by spherical-harmonic inversion, DH2 strip |lat|<=32; anomalies detrended per calendar month; "
                       "coef: chi200 anomaly per unit predictor, ENSO (n34, n34_lag3) first, basin indices as residuals after ENSO (same windows and indices as walker_basins_clim.nc)"})
    ds.to_netcdf(OUT, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
