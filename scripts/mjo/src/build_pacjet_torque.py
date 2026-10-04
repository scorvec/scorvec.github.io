#!/usr/bin/env python3
"""Himalayan mountain torque -> North Pacific jet phase indices: a SIGNED lagged regression on ERA5 1991-2020 (laptop,
once), the research view of pacjet.py.

The previous product composited the jet after torque days >= +1.5 sigma only and then anchored that composite on the
forecast's torque peak whatever its sign. This asks the symmetric question instead: for every lag L from -10 to +20
days, the least-squares slope of the jet index at t + L on the torque anomaly at t, both standardised within the
season window (so the slope is also the correlation and beta^2 the share of variance), for torque days in each
in-season month's three-month window.

Uncertainty honours the autocorrelation of both series (the torque decorrelates in a few days, the jet indices in about
a week): a moving-block bootstrap over the time-ordered torque days with BLOCK-day blocks (1000 draws) gives the 90%
interval and a two-sided percentile p-value, and the p-values of all lags and both indices in a window are held to a
false discovery rate of 10% (Benjamini-Hochberg; Wilks 2016). The effective sample size from the lag-1
autocorrelations is reported as a cross-check.

    python src/build_pacjet_torque.py --series   (once, ~10 min: WB2 surface pressure -> him_torque_1991_2020.nc)
    python src/build_pacjet_torque.py            -> scripts/mjo/data/reference/pacjet_torque.json

The torque series (unchanged from the 2026-09-06 product): -integral of p_s' dh/dx a cos(phi) dA over 70-105E 25-45N on the
WeatherBench-2 1.5 deg grid, 00Z+12Z daily mean, anomaly against a harmonic day-of-year climatology (same sign
convention as the torque product: high pressure west of the range and low east = braking, negative). Anomalies only:
the absolute mean at 1.5 deg is resolution junk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REF = HERE.parent / "data" / "reference"
LAGS = list(range(-10, 21))
BLOCK, NBOOT, ALPHA = 30, 1000, 0.10


WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
HIM = dict(lat=(25.0, 45.0), lon=(70.0, 105.0))
A_EARTH, G0, HADLEY = 6.371e6, 9.80665, 1e18


def build_series():
    """Stream WB2 surface pressure over the Himalaya/Tibet box 1991-2020 -> data/reference/him_torque_1991_2020.nc."""
    import time
    import gcsfs
    t0 = time.time()
    fs = gcsfs.GCSFileSystem(token="anon")
    z = xr.open_zarr(fs.get_mapper(WB2), chunks=None)
    box = dict(latitude=slice(HIM["lat"][0] - 1.5, HIM["lat"][1] + 1.5), longitude=slice(HIM["lon"][0] - 1.5, HIM["lon"][1] + 1.5))
    zs = z["geopotential_at_surface"]
    zs = zs.isel(time=0) if "time" in zs.dims else zs
    if not np.all(np.diff(z.latitude.values) > 0):
        z = z.sortby("latitude"); zs = zs.sortby("latitude")
    sp = z["surface_pressure"].sel(time=slice("1991-01-01", "2020-12-31")).sel(**box)
    sp = sp.sel(time=sp.time.dt.hour.isin([0, 12]))
    parts = []
    for y in range(1991, 2021):
        parts.append(sp.sel(time=str(y)).load().resample(time="1D").mean())
        print(f"    {y} ({time.time() - t0:.0f}s)", flush=True)
    sp = xr.concat(parts, "time").transpose("time", "latitude", "longitude")
    h = (zs.sel(**box).load() / G0).transpose("latitude", "longitude")
    lat, lon = sp.latitude.values, sp.longitude.values
    dlon, dlat = np.deg2rad(float(lon[1] - lon[0])), np.deg2rad(float(lat[1] - lat[0]))
    dx = dlon * A_EARTH * np.cos(np.deg2rad(lat))[:, None]
    dhdx = np.gradient(h.values, axis=1) / dx
    dA = (dlon * A_EARTH) * (dlat * A_EARTH) * np.cos(np.deg2rad(lat))[:, None]
    lever = A_EARTH * np.cos(np.deg2rad(lat))[:, None]
    ii = (lat >= HIM["lat"][0]) & (lat <= HIM["lat"][1]); jj = (lon >= HIM["lon"][0]) & (lon <= HIM["lon"][1])
    kern = (-dhdx * dA * lever)[ii][:, jj]
    tq = np.einsum("tij,ij->t", sp.values[:, ii][:, :, jj], kern) / HADLEY
    doy = sp.time.dt.dayofyear.values
    import pacjet_core as PC
    B = PC.harm_basis(doy, 2)
    coef, *_ = np.linalg.lstsq(B, tq, rcond=None)
    anom = (tq - B @ coef).astype("float32")
    da = xr.DataArray(anom, coords={"time": sp.time.values}, dims=("time",), name="him_torque_anom",
                      attrs={"units": "Hadley (1e18 N m)", "note": "-int p_s' dh/dx a cos(phi) dA over 70-105E 25-45N, WB2 1.5 deg, "
                                                                   "00Z+12Z mean, anomaly vs harmonic doy clim"})
    da.to_dataset().to_netcdf(REF / "him_torque_1991_2020.nc", encoding={"him_torque_anom": {"zlib": True, "complevel": 4}})
    print(f"  torque anomaly sigma {anom.std():.2f} Hadley -> him_torque_1991_2020.nc", flush=True)


def bh(p, alpha=ALPHA):
    q = np.sort(p[np.isfinite(p)])
    if q.size == 0:
        return np.zeros_like(p, bool)
    k = np.flatnonzero(q <= alpha * np.arange(1, q.size + 1) / q.size)
    return np.isfinite(p) & (p <= (q[k.max()] if k.size else -1))


def main() -> int:
    if "--series" in sys.argv:
        build_series()
        return 0
    ref = xr.open_dataset(REF / "pacjet_ref.nc").load()
    t = pd.DatetimeIndex(ref.time.values)
    tq = ref.him_torque_anom.values.astype(float)
    pc = ref.pc.values.astype(float)
    mon = t.month.values
    n = len(t)
    rng = np.random.default_rng(7)
    out = {"lags": LAGS, "block_days": BLOCK, "n_boot": NBOOT, "alpha_fdr": ALPHA,
           "note": "beta = regression of the standardised jet index at t+lag on the standardised Himalayan torque at t "
                   "(ERA5/WB2 1991-2020), torque days in each month's 3-month window; 90% moving-block bootstrap CI; "
                   "significance: bootstrap p, BH FDR 10% over lags x indices", "windows": {}}
    for m in range(1, 13):
        if not bool(ref.in_season.sel(month=m)):
            continue
        months = [((m - 2) % 12) + 1, m, (m % 12) + 1]
        days = np.flatnonzero(np.isin(mon, months) & np.isfinite(tq))
        days = days[(days + min(LAGS) >= 0) & (days + max(LAGS) < n)]
        x = tq[days]; x = (x - x.mean()) / x.std()
        Y = np.stack([pc[days + L] for L in LAGS], 1)                         # (n, lag, 2)
        Y = (Y - Y.mean(0)) / Y.std(0)
        beta = (x[:, None, None] * Y).mean(0)                                  # (lag, 2)
        nb = int(np.ceil(len(days) / BLOCK))
        boots = np.empty((NBOOT, len(LAGS), 2))
        for b in range(NBOOT):
            st = rng.integers(0, len(days) - BLOCK + 1, nb)
            idx = (st[:, None] + np.arange(BLOCK)[None]).ravel()[:len(days)]
            xb = x[idx]; Yb = Y[idx]
            xb = (xb - xb.mean()) / xb.std(); Yb = (Yb - Yb.mean(0)) / Yb.std(0)
            boots[b] = (xb[:, None, None] * Yb).mean(0)
        lo, hi = np.percentile(boots, 5, 0), np.percentile(boots, 95, 0)
        p = np.minimum(1.0, 2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0)))
        sig = bh(p.ravel()).reshape(p.shape)
        r1x = np.corrcoef(x[:-1], x[1:])[0, 1]
        r1y = [np.corrcoef(pc[days[:-1], k], pc[days[:-1] + 1, k])[0, 1] for k in range(2)]
        neff = [len(days) * (1 - r1x * r) / (1 + r1x * r) for r in r1y]
        lab = "–".join(pd.Timestamp(2001, mm, 1).strftime("%b") for mm in (months[0], months[-1])) + " torque days"
        win = {"label": lab, "n_days": int(len(days)), "n_eff_lag1": [round(v) for v in neff],
               "torque_r1": round(float(r1x), 2)}
        for k, name in enumerate(("pc1", "pc2")):
            win[name] = {"beta": np.round(beta[:, k], 3).tolist(), "lo": np.round(lo[:, k], 3).tolist(),
                         "hi": np.round(hi[:, k], 3).tolist(), "p": np.round(p[:, k], 4).tolist(), "sig": sig[:, k].tolist()}
        out["windows"][str(m)] = win
        best = [(name, LAGS[int(np.argmax(np.abs(beta[:, k]) * (np.array(LAGS) >= 1)))]) for k, name in enumerate(("pc1", "pc2"))]
        txt = "; ".join(f"{nm} max |beta| at +{L} d = {beta[LAGS.index(L), k]:+.2f} [{lo[LAGS.index(L), k]:+.2f}, "
                        f"{hi[LAGS.index(L), k]:+.2f}] {'SIG' if sig[LAGS.index(L), k] else 'n.s.'}" for k, (nm, L) in enumerate(best))
        print(f"  {pd.Timestamp(2001, m, 1):%b} ({lab}, n {len(days)}, n_eff {win['n_eff_lag1']}): {txt}; "
              f"significant lags {sig.sum()} of {sig.size}", flush=True)
    (REF / "pacjet_torque.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {REF / 'pacjet_torque.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
