#!/usr/bin/env python3
"""Reference for the Walker-basins product (walker_basins.py): what the Indian and Atlantic
Oceans do to the tropical zonal overturning once ENSO is taken out — built once on the laptop.

Two independent legs share one reference file:

  A. Statistical (ERA5 + ERSST, 1991–2020). The Walker circulation is measured as the two-level
     baroclinic proxy W(λ) = u_D(200 hPa) − u_D(850 hPa), the 5°S–5°N mean DIVERGENT zonal wind
     (velocity potential in spherical harmonics, the walker.py solver) — the upper minus lower
     branch of Ψ_W. Monthly W and the ERSSTv5 basin indices (Niño-3.4, the IOD dipole, the Indian
     Ocean basin mean, Atlantic Niño ATL3, tropical North Atlantic) are detrended per calendar
     month, then W is regressed, per calendar month (±1 month window, 90 samples), on Niño-3.4
     and its 3-month lag first; the Indian and Atlantic indices enter only as the RESIDUALS after
     that ENSO regression, so their coefficients are the ENSO-independent contribution.
  B. Idealised (Gill 1980). A linear equatorial β-plane shallow-water model with Rayleigh
     damping and Newtonian cooling is forced by heating proportional to the SST anomaly where
     the climatological SST exceeds the convective threshold. Its single free amplitude is
     calibrated ONCE against the ENSO regression (leg A) so that the responses to the Indian and
     Atlantic forcing are the model's own, not a fit.

Outputs: scripts/mjo/data/reference/walker_basins_clim.nc (+ _summary.json).
    python src/build_walker_basins_clim.py        (~10 min: 720 velocity-potential solves)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from wind200_vpot import velocity_potential, irrotational_wind   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sst"))
from gill_model import LAT2, LON2, C_WAVE, EPS   # noqa: E402

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "walker_basins_clim.nc"
STORE = Path.home() / "era5_store" / "wb2_1p5_daily_global"
ERSST = Path(__file__).resolve().parents[2] / "sst" / "data" / "ersst_v5_mnmean.nc"
Y0, Y1 = 1991, 2020
BAND = 5.0
LMAX = 42
# SST boxes (lon in 0–360)
BOXES = {
    "n34": ((-5, 5), (190, 240)),
    "dmi_w": ((-10, 10), (50, 70)), "dmi_e": ((-10, 0), (90, 110)),
    "iob": ((-20, 20), (40, 100)),
    "atl3": ((-3, 3), (340, 360)),
    "tna": ((5, 25), (305, 345)),
}
PREDICTORS = ["n34", "n34_lag3", "dmi", "iob", "atl3", "tna"]
GROUPS = {"enso": ["n34", "n34_lag3"], "indian": ["dmi", "iob"], "atlantic": ["atl3", "tna"]}
BASIN_LON = {"indian": (30, 110), "pacific": (110, 285), "atlantic": (285, 380)}   # atlantic wraps to 20°E
CONV_SST = 27.0


# ── leg A inputs ──────────────────────────────────────────────────────────────
def monthly_uv(var: str):
    files = [f for f in sorted((STORE / var).glob(f"{var}_*.nc")) if Y0 <= int(f.stem.split("_")[-1]) <= Y1]
    parts = []
    for f in files:
        ds = xr.open_dataset(f); da = ds[list(ds.data_vars)[0]]
        da = da.assign_coords(latitude=np.round(da.latitude.values.astype("float64"), 3), longitude=np.round(da.longitude.values.astype("float64"), 3))
        parts.append(da.transpose("time", "latitude", "longitude").resample(time="1MS").mean().load()); ds.close()
    return xr.concat(parts, dim="time", join="exact").sortby("time").sortby("latitude")


def walker_series():
    """W(t, lon2) = u_D200 − u_D850 band means, monthly 1991–2020, and the two branches."""
    t0 = time.time()
    u200, v200, u850, v850 = (monthly_uv(v) for v in ("u200", "v200", "u850", "v850"))
    print(f"  monthly u/v loaded {dict(u200.sizes)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for lev, (u, v) in (("200", (u200, v200)), ("850", (u850, v850))):
        rows = []
        for k in range(u.sizes["time"]):
            chi, dlat, dlon = velocity_potential(u.isel(time=k), v.isel(time=k), lmax=LMAX)
            uchi, _ = irrotational_wind(chi, dlat, dlon)
            band = np.abs(dlat) <= BAND; w = np.cos(np.deg2rad(dlat[band]))
            ud = (uchi[band] * w[:, None]).sum(0) / w.sum()
            rows.append(np.interp(LON2, np.concatenate([dlon, [dlon[0] + 360]]), np.concatenate([ud, [ud[0]]])))
            if k % 60 == 0:
                print(f"    u_D{lev}: {k}/{u.sizes['time']} ({time.time() - t0:.0f}s)", flush=True)
        out[lev] = np.array(rows)
    return u200.time.values, out["200"], out["850"]


def ersst():
    ds = xr.open_dataset(ERSST); sst = ds["sst"]
    sst = sst.sel(time=slice("1990-10-01", None)).load()
    lat, lon = sst.lat.values, sst.lon.values
    return sst, lat, lon


def box_mean(field, lat, lon, box):
    (la0, la1), (lo0, lo1) = box
    m = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * m
    f = np.where(np.isfinite(field), field, 0.0); w = np.where(np.isfinite(field), w, 0.0)
    return (f * w).sum(axis=(-2, -1)) / w.sum(axis=(-2, -1))


def detrend_by_month(values, times, y0=Y0, y1=Y1):
    """Anomaly vs the 1991–2020 calendar-month mean, minus the 1991–2020 linear trend of that month
    (extrapolated outside). Returns (anom, clim[12,...], slope[12,...], intercept year mean)."""
    t = pd.DatetimeIndex(times); yrs = t.year.values; mos = t.month.values
    v = np.asarray(values, float)
    clim = np.zeros((12, *v.shape[1:])); slope = np.zeros((12, *v.shape[1:])); anom = np.full(v.shape, np.nan)
    for m in range(1, 13):
        fit = (mos == m) & (yrs >= y0) & (yrs <= y1)
        x = yrs[fit] - (y0 + y1) / 2
        clim[m - 1] = np.nanmean(v[fit], axis=0)
        slope[m - 1] = (x[:, None, None] * (v[fit] - clim[m - 1])).reshape(len(x), -1).sum(0).reshape(v.shape[1:]) / (x ** 2).sum() if v.ndim == 3 else \
                       (x[:, None] * (v[fit] - clim[m - 1])).sum(0) / (x ** 2).sum() if v.ndim == 2 else (x * (v[fit] - clim[m - 1])).sum() / (x ** 2).sum()
        sel = mos == m
        anom[sel] = v[sel] - clim[m - 1] - slope[m - 1] * (yrs[sel] - (y0 + y1) / 2).reshape(-1, *([1] * (v.ndim - 1)))
    return anom, clim, slope


# ── leg B: Gill model (shared module, NaN-safe) ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sst"))
from gill_model import gill_response, heating_from_ssta, band_u   # noqa: E402


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-era5", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cache = REF / "_walker_basins_era5_cache.nc"
    if cache.exists() and a.skip_era5 or (cache.exists() and not a.skip_era5 and False):
        c = xr.open_dataset(cache); times, ud200, ud850 = c.time.values, c.ud200.values, c.ud850.values
    elif cache.exists():
        c = xr.open_dataset(cache); times, ud200, ud850 = c.time.values, c.ud200.values, c.ud850.values
        print("  ERA5 u_D from cache", flush=True)
    else:
        times, ud200, ud850 = walker_series()
        xr.Dataset({"ud200": (("time", "lon"), ud200), "ud850": (("time", "lon"), ud850)}, coords={"time": times, "lon": LON2}).to_netcdf(cache)
    W = ud200 - ud850                                               # (t, lon)
    mos = pd.DatetimeIndex(times).month.values
    # harmonic day-of-year climatology of the two branches (for the live product's anomalies)
    doy = pd.DatetimeIndex(times).dayofyear.values + 14            # monthly means sit mid-month
    w_ = 2 * np.pi * doy / 365.25
    B = np.stack([np.ones_like(w_), np.cos(w_), np.sin(w_), np.cos(2 * w_), np.sin(2 * w_)], -1)
    coef200, *_ = np.linalg.lstsq(B, ud200, rcond=None); coef850, *_ = np.linalg.lstsq(B, ud850, rcond=None)
    W_anom, W_clim, W_slope = detrend_by_month(W, times)

    # ── SST: ERSST anomalies, detrended, indices, ENSO pattern ──
    sst, elat, elon = ersst()
    sst_f = sst.values.astype("float64")
    tt = sst.time.values
    ssta, sst_clim, sst_slope = detrend_by_month(sst_f, tt)
    sel = (pd.DatetimeIndex(tt).year >= Y0) & (pd.DatetimeIndex(tt).year <= Y1)
    idx = {}
    for k in ("n34", "iob", "atl3", "tna"):
        idx[k] = box_mean(ssta, elat, elon, BOXES[k])
    idx["dmi"] = box_mean(ssta, elat, elon, BOXES["dmi_w"]) - box_mean(ssta, elat, elon, BOXES["dmi_e"])
    n34 = pd.Series(idx["n34"], index=pd.DatetimeIndex(tt))
    idx["n34_lag3"] = n34.shift(3).values
    smos = pd.DatetimeIndex(tt).month.values
    # align SST months to the W months
    tW = pd.DatetimeIndex(times); tS = pd.DatetimeIndex(tt)
    pos = {(d.year, d.month): i for i, d in enumerate(tS)}
    ii = np.array([pos[(d.year, d.month)] for d in tW])
    X = {k: v[ii] for k, v in idx.items()}
    # ENSO SST regression pattern per calendar month (for removing ENSO from an SST field)
    enso_pat = np.zeros((12, elat.size, elon.size)); pred_sd = np.zeros((12, len(PREDICTORS)))
    for m in range(1, 13):
        win = np.isin(smos, [(m - 2) % 12 + 1, m, m % 12 + 1]) & sel
        n = idx["n34"][win]; f = np.nan_to_num(ssta[win])
        enso_pat[m - 1] = np.einsum("t,tij->ij", n - n.mean(), f - f.mean(0)) / ((n - n.mean()) ** 2).sum()
        winW = np.isin(mos, [(m - 2) % 12 + 1, m, m % 12 + 1])
        pred_sd[m - 1] = [np.nanstd(X[k][winW]) for k in PREDICTORS]

    # ── regression per calendar month: ENSO first, basins as residuals ──
    nl = LON2.size
    coef = np.zeros((12, len(PREDICTORS), nl)); r2 = np.zeros((12, 4))
    resid_ops = {}
    for m in range(1, 13):
        win = np.isin(mos, [(m - 2) % 12 + 1, m, m % 12 + 1])
        ok = win & np.all([np.isfinite(X[k]) for k in PREDICTORS], axis=0)
        E = np.stack([X["n34"][ok], X["n34_lag3"][ok]], 1)
        Ec = E - E.mean(0)
        Yw = W_anom[ok]; Yc = Yw - Yw.mean(0)
        # basin residuals after ENSO
        R = {}
        for k in ("dmi", "iob", "atl3", "tna"):
            xk = X[k][ok] - X[k][ok].mean()
            g, *_ = np.linalg.lstsq(Ec, xk, rcond=None)
            R[k] = xk - Ec @ g
            resid_ops[(m, k)] = g
        Xf = np.column_stack([Ec, R["dmi"], R["iob"], R["atl3"], R["tna"]])
        def fit(Xm):
            b, *_ = np.linalg.lstsq(Xm, Yc, rcond=None); pred = Xm @ b
            return b, 1 - ((Yc - pred) ** 2).sum() / (Yc ** 2).sum()
        b_e, r_e = fit(Ec)
        _, r_ei = fit(np.column_stack([Ec, R["dmi"], R["iob"]]))
        _, r_ea = fit(np.column_stack([Ec, R["atl3"], R["tna"]]))
        b_f, r_f = fit(Xf)
        coef[m - 1] = b_f; r2[m - 1] = [r_e, r_ei, r_ea, r_f]
    print("  R² (all longitudes, by month): ENSO only / +Indian / +Atlantic / full", flush=True)
    for m in range(12):
        print(f"    {m + 1:2d}: {r2[m, 0]:.2f} / {r2[m, 1]:.2f} / {r2[m, 2]:.2f} / {r2[m, 3]:.2f}", flush=True)

    # ── Gill calibration: response to the ENSO SST pattern vs the ENSO regression of W ──
    # ERSST clim on the Gill grid for the convective mask
    def to_gill(field2d):
        da = xr.DataArray(field2d, coords={"lat": elat, "lon": elon}, dims=("lat", "lon"))
        return da.interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None}).values
    alphas = []; gill_enso = np.zeros((12, nl)); reg_enso = np.zeros((12, nl))
    for m in range(1, 13):
        Q = heating_from_ssta(to_gill(enso_pat[m - 1] * pred_sd[m - 1, 0]), to_gill(sst_clim[m - 1]))
        u, v, p = gill_response(Q)
        wg = -2.0 * band_u(u)                                        # first baroclinic: upper = −lower
        wr = coef[m - 1, 0] * pred_sd[m - 1, 0]                      # W per 1σ Niño-3.4 (same-month term)
        pac = (LON2 >= 120) & (LON2 <= 280); box = (LON2 >= 140) & (LON2 <= 200)
        # amplitude anchored on the Pacific Walker box mean (the headline quantity); a least-squares
        # slope over longitude would shrink it by the pattern correlation. r is reported as the shape check.
        alpha = wr[box].mean() / wg[box].mean()
        alphas.append(alpha); gill_enso[m - 1] = wg; reg_enso[m - 1] = wr
        print(f"    Gill calibration month {m:2d}: α {alpha:.3f}  pattern r {np.corrcoef(wg[pac], wr[pac])[0, 1]:+.2f}", flush=True)
    alpha = float(np.median(alphas))
    print(f"  Gill amplitude α = {alpha:.3f} (median of months; range {min(alphas):.3f}–{max(alphas):.3f})", flush=True)

    ds = xr.Dataset(
        {"ud200_coef": (("harm", "lon"), coef200), "ud850_coef": (("harm", "lon"), coef850),
         "w_clim": (("month", "lon"), W_clim), "w_slope": (("month", "lon"), W_slope),
         "coef": (("month", "pred", "lon"), coef), "r2": (("month", "stage"), r2), "pred_sd": (("month", "pred"), pred_sd),
         "sst_clim": (("month", "lat", "lon_e"), sst_clim.astype("float32")), "sst_slope": (("month", "lat", "lon_e"), sst_slope.astype("float32")),
         "enso_pattern": (("month", "lat", "lon_e"), enso_pat.astype("float32")),
         "resid_ops": (("month", "basin_pred", "enso_pred"), np.array([[resid_ops[(m, k)] for k in ("dmi", "iob", "atl3", "tna")] for m in range(1, 13)])),
         "gill_enso": (("month", "lon"), gill_enso), "reg_enso": (("month", "lon"), reg_enso),
         "w_train": (("time", "lon"), W_anom.astype("float32")),
         "idx_train": (("time", "pred"), np.stack([X[k] for k in PREDICTORS], 1).astype("float32"))},
        coords={"harm": ["mean", "cos1", "sin1", "cos2", "sin2"], "lon": LON2, "month": np.arange(1, 13), "pred": PREDICTORS,
                "stage": ["enso", "enso+indian", "enso+atlantic", "full"], "lat": elat, "lon_e": elon,
                "basin_pred": ["dmi", "iob", "atl3", "tna"], "enso_pred": ["n34", "n34_lag3"], "time": times},
        attrs={"gill_alpha": alpha, "gill_c": C_WAVE, "gill_eps": EPS, "conv_sst": CONV_SST, "base": f"{Y0}-{Y1}",
               "boxes": json.dumps(BOXES), "basin_lon": json.dumps(BASIN_LON),
               "note": "W = u_D200 - u_D850 (5S-5N band mean of the divergent zonal wind, m/s); anomalies detrended per calendar month; "
                       "basin indices enter the regression as residuals after Niño-3.4 (t, t-3 months); Gill model in nondimensional units, "
                       "W_gill = -2 * alpha * u_low(band)"})
    ds.to_netcdf(OUT, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    (REF / "walker_basins_summary.json").write_text(json.dumps({"r2_by_month": r2.round(3).tolist(), "gill_alpha": alpha, "gill_alpha_by_month": [round(x, 3) for x in alphas],
                                                                 "stages": ["enso", "enso+indian", "enso+atlantic", "full"]}, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
