#!/usr/bin/env python3
"""ERA5 reference for the North Pacific jet monitor (pacjet.py) — built once on the laptop.

Everything the live product needs to say "extended / retracted / shifted, and how unusual":

  1. 200 hPa zonal wind over the North Pacific sector (10–70°N, 100°E–120°W) from the LOCAL
     ERA5 store (~/era5_store/wb2_1p5_daily/u200, daily means, 1.5°): harmonic day-of-year
     climatology (mean + annual + semiannual) per grid point, and the σ(doy) of the anomaly.
  2. Jet-regime EOFs (Jaffe et al. 2011; Griffin & Martin 2017; Winters et al. 2019 use 250 hPa
     over the same sector): EOF1/EOF2 of the area-weighted daily anomalies for Nov–Mar 1991–2020.
     EOF1 is signed so that positive = jet EXTENSION (westerly anomaly in the exit region,
     30–40°N 170°E–150°W); EOF2 so that positive = POLEWARD shift (westerly anomaly north of the
     climatological axis). PCs are in units of their Nov–Mar standard deviation.
  3. The simple all-season indices, with their own doy climatology / σ: the exit-region mean
     anomaly (30–40°N, 170°E–150°W) and the jet terminus longitude (easternmost longitude reached
     by the ≥ 30 m/s core, walking east from 130°E along the 20–55°N maximum).
  4. A Himalayan mountain-torque history (70–105°E, 25–45°N) from WeatherBench-2 1.5° surface
     pressure (00Z+12Z mean) and orography, as the anomaly −∫ p_s' ∂h/∂x a cosφ dA vs the doy
     climatology (the same sign convention as the torque product: high pressure west of the
     range and low east = braking, negative). From it, the lead–lag correlation and event
     composites of the jet indices after strong torque days — the measured basis for the
     "torque → downstream jet" reading.

Outputs (committed, small):
  scripts/mjo/data/reference/pacjet_clim.nc   clim/σ fields, EOF patterns, index climatologies,
                                              the daily ERA5 index record 1991–2020
  scripts/mjo/data/reference/pacjet_lag.json  lag correlations + composites by season

    python src/build_pacjet_clim.py            (~10 min: WB2 surface-pressure stream dominates)
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

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT_NC, OUT_JSON = REF / "pacjet_clim.nc", REF / "pacjet_lag.json"
STORE = Path.home() / "era5_store" / "wb2_1p5_daily" / "u200"
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
CACHE = Path.home() / "mjo" / "era5_cache"

Y0, Y1 = 1991, 2020
SECTOR = dict(lat=(10.0, 70.0), lon=(100.0, 240.0))          # 100°E–120°W
EXIT = dict(lat=(30.0, 40.0), lon=(170.0, 210.0))            # 170°E–150°W
HIM = dict(lat=(25.0, 45.0), lon=(70.0, 105.0))
ZDOM = dict(lat=(20.0, 80.0), lon=(100.0, 260.0))            # 500 hPa domain for the downstream response, to 100°W
ALASKA = dict(lat=(55.0, 70.0), lon=(195.0, 235.0))          # Alaska ridge box, 165–125°W (the EPO centre, sign-flipped)
GOA = dict(lat=(40.0, 60.0), lon=(215.0, 240.0))             # Gulf of Alaska / West Coast ridge box, 145–120°W
COMP_LAGS = [0, 3, 6, 9, 12]
STORE_Z = Path.home() / "era5_store" / "wb2_1p5_daily" / "z500"
CORE_MS = 30.0                                               # jet-core threshold for the terminus
COLD = (11, 12, 1, 2, 3)
A_EARTH, G0, HADLEY = 6.371e6, 9.80665, 1e18
SEASONS = {"NDJFM": COLD, "SON": (9, 10, 11), "all": tuple(range(1, 13))}


# ── helpers ──────────────────────────────────────────────────────────────────
def harm_basis(doy):
    w = 2 * np.pi * np.asarray(doy, dtype=float) / 365.25
    return np.stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)], axis=-1)


def harm_fit(doy, y):
    """y [n, ...] → coefficients [5, ...] of the mean + annual + semiannual fit."""
    B = harm_basis(doy)
    coef, *_ = np.linalg.lstsq(B, y.reshape(len(y), -1), rcond=None)
    return coef.reshape(5, *y.shape[1:])


def harm_eval(coef, doy):
    return np.tensordot(harm_basis(doy), coef, axes=(-1, 0))


def sigma_doy(doy, resid, halfwidth=15):
    """Smoothed σ by day of year: std of the residuals within ±halfwidth days (circular)."""
    doy = np.asarray(doy); out = np.empty((366, *resid.shape[1:]))
    for d in range(1, 367):
        dd = np.abs(((doy - d + 183) % 366) - 183)
        out[d - 1] = np.nanstd(resid[dd <= halfwidth], axis=0)
    return out


def sel_box(da, box):
    return da.sel(latitude=slice(box["lat"][0], box["lat"][1]), longitude=slice(box["lon"][0], box["lon"][1]))


def to_lon360(da):
    lon = da.longitude.values
    return da.assign_coords(longitude=np.where(lon < 0, lon + 360, lon)).sortby("longitude")


# ── jet indices from a (time, lat, lon) 200 hPa u field on the sector grid ──
def exit_index(u_anom):
    b = sel_box(u_anom, EXIT); w = np.cos(np.deg2rad(b.latitude))
    return b.weighted(w).mean(("latitude", "longitude")).values


def terminus(u_abs, lon0=130.0):
    """Easternmost longitude of the ≥ CORE_MS core, walking east from lon0 along the 20–55°N maximum.
    NaN when no core ≥ CORE_MS sits in 120–160°E to start from (the summer state)."""
    band = u_abs.sel(latitude=slice(20, 55)).max("latitude")          # (time, lon)
    lon = band.longitude.values; v = band.values
    out = np.full(v.shape[0], np.nan)
    i0 = int(np.argmin(np.abs(lon - lon0)))
    for t in range(v.shape[0]):
        row = v[t]
        start = np.where(row[max(0, i0 - 8):i0 + 8] >= CORE_MS)[0]     # a core within ±12° of 130°E
        if start.size == 0:
            continue
        j = max(0, i0 - 8) + start[-1]
        while j + 1 < row.size and row[j + 1] >= CORE_MS:
            j += 1
        out[t] = lon[j]
    return out


# ── 1–3: 200 hPa climatology, EOFs, index record ────────────────────────────
def load_u200():
    files = sorted(STORE.glob("u200_*.nc"))
    files = [f for f in files if Y0 <= int(f.stem.split("_")[-1]) <= Y1]
    parts = []
    for f in files:
        ds = xr.open_dataset(f); da = ds[list(ds.data_vars)[0]]
        da = da.assign_coords(latitude=np.round(da.latitude.values.astype("float64"), 3),
                              longitude=np.round(da.longitude.values.astype("float64"), 3))
        parts.append(sel_box(to_lon360(da.transpose("time", "latitude", "longitude")), SECTOR).load())
        ds.close()
    u = xr.concat(parts, dim="time", join="exact").sortby("time").sortby("latitude")
    return u


def build_u200():
    t0 = time.time()
    u = load_u200()
    print(f"  u200 sector {dict(u.sizes)} {str(u.time.values[0])[:10]}–{str(u.time.values[-1])[:10]} ({time.time() - t0:.0f}s)", flush=True)
    doy = u.time.dt.dayofyear.values
    coef = harm_fit(doy, u.values)                                     # [5, lat, lon]
    anom = u.values - harm_eval(coef, doy)
    sd = sigma_doy(doy, anom)                                          # [366, lat, lon]
    lat, lon = u.latitude.values, u.longitude.values
    # EOFs on Nov–Mar, area-weighted
    mo = u.time.dt.month.values
    cold = np.isin(mo, COLD)
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
    X = (anom[cold] * w[None]).reshape(cold.sum(), -1)
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    expl = (S ** 2 / (S ** 2).sum())[:5]
    eofs = Vt[:2].reshape(2, lat.size, lon.size) / w[None]             # unweighted patterns (m/s per unit PC)
    # projector: PC = Σ anom·w·e_w  (e_w = weighted eigenvector), normalised by its cold-season σ
    e_w = Vt[:2].reshape(2, lat.size, lon.size)
    pcs = np.einsum("tij,kij->tk", anom * w[None], e_w)
    pc_sd = pcs[cold].std(0)
    # sign conventions
    exit_mask = ((lat >= EXIT["lat"][0]) & (lat <= EXIT["lat"][1]))[:, None] & ((lon >= EXIT["lon"][0]) & (lon <= EXIT["lon"][1]))[None]
    if eofs[0][exit_mask].mean() < 0:
        eofs[0] *= -1; e_w[0] *= -1; pcs[:, 0] *= -1
    axis_lat = lat[np.argmax(harm_eval(coef, np.array([15.0]))[0].mean(-1))]  # climatological jet axis latitude (mid-Jan)
    north = (lat > axis_lat + 5)[:, None] & np.ones((1, lon.size), bool)
    if eofs[1][north & ((lon >= 140) & (lon <= 220))[None]].mean() < 0:
        eofs[1] *= -1; e_w[1] *= -1; pcs[:, 1] *= -1
    pcs = pcs / pc_sd
    # regression maps: m/s of anomaly per 1σ of the PC — what the map contours show
    eofs = np.einsum("tij,tk->kij", anom[cold], pcs[cold]) / cold.sum()
    print(f"  EOF1/EOF2 explain {expl[0]:.1%} / {expl[1]:.1%} of Nov–Mar variance (next {expl[2]:.1%}); axis {axis_lat:.1f}°N", flush=True)
    # simple indices, their own doy climatology
    ua = xr.DataArray(anom, coords=u.coords, dims=u.dims)
    ex = exit_index(ua)
    ex_sd = sigma_doy(doy, ex[:, None])[:, 0]
    term = terminus(u)
    ok = np.isfinite(term)
    term_coef = harm_fit(doy[ok], term[ok][:, None])[:, 0]
    term_anom = term - harm_eval(term_coef, doy)
    term_sd = sigma_doy(doy, np.where(ok, term_anom, np.nan)[:, None])[:, 0]
    frac_defined = np.array([np.mean(ok[doy == d]) if (doy == d).any() else np.nan for d in range(1, 367)])
    ds = xr.Dataset(
        {"u_coef": (("harm", "latitude", "longitude"), coef.astype("float32")),
         "u_sd": (("doy", "latitude", "longitude"), sd.astype("float32")),
         "eof": (("mode", "latitude", "longitude"), eofs.astype("float32")),   # regression map, m/s per σ
         "proj": (("mode", "latitude", "longitude"), (e_w * w[None] / pc_sd[:, None, None]).astype("float32")),
         "explained": (("mode",), expl[:2]),
         "exit_sd": (("doy",), ex_sd.astype("float32")),
         "term_coef": (("harm",), term_coef), "term_sd": (("doy",), term_sd.astype("float32")),
         "term_defined_frac": (("doy",), frac_defined.astype("float32")),
         "pc": (("time", "mode"), pcs.astype("float32")), "exit": (("time",), (ex / ex_sd[doy - 1]).astype("float32")),
         "exit_ms": (("time",), ex.astype("float32")), "terminus": (("time",), term.astype("float32")),
         "terminus_anom": (("time",), (term_anom / term_sd[doy - 1]).astype("float32"))},
        coords={"harm": ["mean", "cos1", "sin1", "cos2", "sin2"], "latitude": lat, "longitude": lon, "doy": np.arange(1, 367),
                "mode": ["extension", "shift"], "time": u.time.values},
        attrs={"note": f"ERA5 (WB2 1.5° daily) 200 hPa u, {Y0}–{Y1}; sector {SECTOR}; exit box {EXIT}; EOFs on Nov–Mar; "
                       f"proj: PC_k = Σ anom·proj_k (σ units, Nov–Mar); terminus: easternmost lon of the ≥{CORE_MS:.0f} m/s core from 130°E",
               "axis_lat_jan": float(axis_lat)})
    return ds


# ── 4: Himalayan mountain-torque history from WB2 ────────────────────────────
def build_torque():
    import gcsfs
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"wb2_sp_himalaya_{Y0}-{Y1}.nc"
    if cpath.exists():
        ds = xr.open_dataset(cpath)
    else:
        t0 = time.time()
        fs = gcsfs.GCSFileSystem(token="anon")
        z = xr.open_zarr(fs.get_mapper(WB2), chunks=None)
        box = dict(latitude=slice(HIM["lat"][0] - 1.5, HIM["lat"][1] + 1.5), longitude=slice(HIM["lon"][0] - 1.5, HIM["lon"][1] + 1.5))
        zs = z["geopotential_at_surface"]
        zs = (zs.isel(time=0) if "time" in zs.dims else zs)
        lat_ok = np.all(np.diff(z.latitude.values) > 0)
        if not lat_ok:
            z = z.sortby("latitude"); zs = zs.sortby("latitude")
        sp = z["surface_pressure"].sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31")).sel(**box)
        sp = sp.sel(time=sp.time.dt.hour.isin([0, 12]))
        print(f"  streaming WB2 surface pressure {dict(sp.sizes)} …", flush=True)
        # read in yearly slabs to keep memory flat
        parts = []
        for y in range(Y0, Y1 + 1):
            s = sp.sel(time=str(y)).load()
            parts.append(s.resample(time="1D").mean())
            print(f"    {y} ({time.time() - t0:.0f}s)", flush=True)
        spd = xr.concat(parts, dim="time")
        ds = xr.Dataset({"sp": spd, "h": zs.sel(**box).load() / G0})
        ds.to_netcdf(cpath)
    sp, h = ds["sp"].transpose("time", "latitude", "longitude"), ds["h"].transpose("latitude", "longitude")
    lat, lon = sp.latitude.values, sp.longitude.values
    dx = np.deg2rad(float(lon[1] - lon[0])) * A_EARTH * np.cos(np.deg2rad(lat))[:, None]
    dhdx = np.gradient(h.values, axis=1) / dx
    dA = (np.deg2rad(float(lon[1] - lon[0])) * A_EARTH) * (np.deg2rad(float(lat[1] - lat[0])) * A_EARTH) * np.cos(np.deg2rad(lat))[:, None]
    lever = A_EARTH * np.cos(np.deg2rad(lat))[:, None]
    inner = (lat >= HIM["lat"][0]) & (lat <= HIM["lat"][1])
    innerx = (lon >= HIM["lon"][0]) & (lon <= HIM["lon"][1])
    kern = (-dhdx * dA * lever)[inner][:, innerx]
    tq = np.einsum("tij,ij->t", sp.values[:, inner][:, :, innerx], kern) / HADLEY  # Hadley, absolute
    doy = sp.time.dt.dayofyear.values
    tq_anom = tq - harm_eval(harm_fit(doy, tq[:, None]), doy)[:, 0]
    print(f"  Himalayan torque: mean {tq.mean():+.1f} Hadley, anomaly σ {tq_anom.std():.1f} (Nov–Mar {tq_anom[np.isin(sp.time.dt.month.values, COLD)].std():.1f})", flush=True)
    return xr.DataArray(tq_anom.astype("float32"), coords={"time": sp.time.values}, dims=("time",), name="him_torque_anom",
                        attrs={"units": "Hadley (1e18 N m)", "note": "−∫ p_s' ∂h/∂x a cosφ dA over 70–105°E 25–45°N, WB2 1.5°, 00Z+12Z mean, anomaly vs harmonic doy clim"})


# ── 500 hPa: the downstream response (Alaska ridge hypothesis) ───────────────
def load_z500():
    files = [f for f in sorted(STORE_Z.glob("z500_*.nc")) if Y0 <= int(f.stem.split("_")[-1]) <= Y1]
    parts = []
    for f in files:
        ds = xr.open_dataset(f); da = ds[list(ds.data_vars)[0]]
        da = da.assign_coords(latitude=np.round(da.latitude.values.astype("float64"), 3), longitude=np.round(da.longitude.values.astype("float64"), 3))
        parts.append(sel_box(to_lon360(da.transpose("time", "latitude", "longitude")), ZDOM).load())
        ds.close()
    z = xr.concat(parts, dim="time", join="exact").sortby("time").sortby("latitude")
    if float(z.mean()) > 20000:                                            # m² s⁻² in disguise
        z = z / G0
    return z


def build_z500(times) -> tuple[xr.Dataset, np.ndarray]:
    """(dataset with the z500 harmonic clim / σ / Alaska index climatology, daily anomaly array on `times`)."""
    t0 = time.time()
    z = load_z500().reindex(time=times)
    doy = pd.DatetimeIndex(times).dayofyear.values
    coef = harm_fit(doy, np.nan_to_num(z.values, nan=np.nanmean(z.values)))
    anom = (z.values - harm_eval(coef, doy)).astype("float32")
    sd = sigma_doy(doy, anom)
    lat, lon = z.latitude.values, z.longitude.values
    data = {"z_coef": (("harm", "zlat", "zlon"), coef.astype("float32")), "z_sd": (("doy", "zlat", "zlon"), sd.astype("float32"))}
    for name, box in (("alaska", ALASKA), ("goa", GOA)):
        bm = ((lat >= box["lat"][0]) & (lat <= box["lat"][1]))[:, None] & ((lon >= box["lon"][0]) & (lon <= box["lon"][1]))[None]
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * bm
        v = np.einsum("tij,ij->t", np.nan_to_num(anom), w) / w.sum()
        v_sd = sigma_doy(doy, v[:, None])[:, 0]
        data[f"{name}_sd"] = (("doy",), v_sd.astype("float32")); data[name] = (("time",), (v / v_sd[doy - 1]).astype("float32"))
        print(f"  z500 {name} box anomaly σ {np.nanstd(v):.0f} m", flush=True)
    print(f"  z500 {dict(z.sizes)} ({time.time() - t0:.0f}s)", flush=True)
    return xr.Dataset(data, coords={"zlat": lat, "zlon": lon}), anom


def composites(anom: np.ndarray, times, tq_anom: np.ndarray, sel_by_season: dict, lat, lon, lags=COMP_LAGS, n_null=300) -> tuple[xr.Dataset, dict]:
    """Mean z500 anomaly (m) at each lag after torque days ≥ 1.5σ (σ of that season), per season, with the
    fraction of random same-count date sets whose composite is more extreme (two-sided) — the significance
    field — and a tracker of the strongest significant positive anomaly (where the ridge sits) per lag."""
    rng = np.random.default_rng(1)
    out, track = {}, {}
    for sname, sel in sel_by_season.items():
        tq_sigma = np.where(sel, tq_anom / np.nanstd(tq_anom[sel]), -9.0)
        cand = np.where(sel & (tq_sigma >= 1.5))[0]
        ev = []
        for i in cand:
            if ev and i - ev[-1] < 10:
                if tq_sigma[i] > tq_sigma[ev[-1]]:
                    ev[-1] = i
                continue
            ev.append(i)
        ev = np.array(ev); pool = np.where(sel)[0]
        comp = np.full((len(lags), *anom.shape[1:]), np.nan, "float32"); pval = comp.copy()
        for k, L in enumerate(lags):
            ii = ev[ev + L < len(times)]
            c = np.nanmean(anom[ii + L], axis=0); comp[k] = c
            nulls = np.stack([np.nanmean(anom[np.clip(rng.choice(pool, size=len(ii), replace=False) + L, 0, len(times) - 1)], axis=0) for _ in range(n_null)])
            pval[k] = (np.abs(nulls) >= np.abs(c)[None]).mean(0)
            # where is the ridge: strongest positive anomaly significant at 5%, within 30–75°N east of the dateline region
            dom = ((lat >= 30) & (lat <= 75))[:, None] & ((lon >= 170) & (lon <= 260))[None]
            cs = np.where(dom & (pval[k] < 0.05) & (c > 0), c, np.nan)
            if np.isfinite(cs).any():
                j = np.unravel_index(np.nanargmax(cs), cs.shape)
                track.setdefault(sname, []).append({"lag": L, "lat": float(lat[j[0]]), "lon": float(lon[j[1]]), "amp_m": round(float(cs[j]), 1),
                                                    "sig_area_frac": round(float(np.isfinite(cs).sum() / dom.sum()), 3)})
            else:
                track.setdefault(sname, []).append({"lag": L, "lat": None, "lon": None, "amp_m": None, "sig_area_frac": 0.0})
        out[sname] = (comp, pval, len(ev))
    seasons = list(out)
    ds = xr.Dataset({"z500_comp": (("season", "lag", "zlat", "zlon"), np.stack([out[s][0] for s in seasons])),
                     "z500_p": (("season", "lag", "zlat", "zlon"), np.stack([out[s][1] for s in seasons])),
                     "n_events": (("season",), np.array([out[s][2] for s in seasons]))},
                    coords={"season": seasons, "lag": lags},
                    attrs={"note": "ERA5 500 hPa height anomaly (m) composited on Himalayan mountain-torque days ≥ +1.5σ (peaks ≥ 10 d apart); "
                                   f"z500_p = fraction of {n_null} random same-season date sets at least as extreme (two-sided)"})
    return ds, track


# ── lag relationships ────────────────────────────────────────────────────────
def lag_stats(ds: xr.Dataset, tq: xr.DataArray, extra: dict | None = None) -> dict:
    t = pd.DatetimeIndex(ds.time.values)
    tq = tq.reindex(time=ds.time.values)
    mo = t.month.values
    idx = {"extension": ds.pc.sel(mode="extension").values, "shift": ds.pc.sel(mode="shift").values,
           "exit": ds.exit.values, "terminus": ds.terminus_anom.values, **(extra or {})}
    lags = list(range(-10, 16))
    out = {"lags": lags, "seasons": {}, "event_threshold_sigma": 1.5}
    for sname, months in SEASONS.items():
        sel = np.isin(mo, months)
        x = tq.values.copy(); sd = np.nanstd(x[sel]); x = x / sd
        res = {"torque_sd_hadley": round(float(sd), 2), "corr": {}, "composite": {}, "n_events": 0}
        # events: local peaks ≥ 1.5σ within the season, ≥ 10 d apart
        cand = np.where(sel & (x >= 1.5))[0]
        events = []
        for i in cand:
            if events and i - events[-1] < 10:
                if x[i] > x[events[-1]]:
                    events[-1] = i
                continue
            events.append(i)
        res["n_events"] = len(events)
        rng = np.random.default_rng(0)
        for k, v in idx.items():
            cs, comp = [], []
            for L in lags:
                xi = x[:-L] if L > 0 else (x[-L:] if L < 0 else x)
                yi = v[L:] if L > 0 else (v[:len(v) + L] if L < 0 else v)
                si = sel[:-L] if L > 0 else (sel[-L:] if L < 0 else sel)
                ok = si & np.isfinite(xi) & np.isfinite(yi)
                cs.append(round(float(np.corrcoef(xi[ok], yi[ok])[0, 1]), 3) if ok.sum() > 100 else None)
                vals = [v[i + L] for i in events if 0 <= i + L < len(v) and np.isfinite(v[i + L])]
                comp.append(round(float(np.mean(vals)), 3) if len(vals) >= 5 else None)
            # null band for the composite: random in-season dates, same count
            pool = np.where(sel & np.isfinite(v))[0]
            nulls = []
            for _ in range(500):
                ri = rng.choice(pool, size=max(len(events), 1), replace=False)
                nulls.append([np.nanmean([v[i + L] for i in ri if 0 <= i + L < len(v)]) for L in lags])
            nulls = np.array(nulls)
            res["corr"][k] = cs
            res["composite"][k] = {"mean": comp, "null_p05": np.nanpercentile(nulls, 5, axis=0).round(3).tolist(),
                                   "null_p95": np.nanpercentile(nulls, 95, axis=0).round(3).tolist()}
        for tgt in ("alaska", "goa"):                                # the second link: does an extended jet lead the ridge?
            if tgt not in idx:
                continue
            e, ak = idx["extension"], idx[tgt]; chain = []
            for L in lags:
                xi = e[:-L] if L > 0 else (e[-L:] if L < 0 else e); yi = ak[L:] if L > 0 else (ak[:len(ak) + L] if L < 0 else ak)
                si = sel[:-L] if L > 0 else (sel[-L:] if L < 0 else sel); ok = si & np.isfinite(xi) & np.isfinite(yi)
                chain.append(round(float(np.corrcoef(xi[ok], yi[ok])[0, 1]), 3) if ok.sum() > 100 else None)
            res[f"corr_extension_to_{tgt}"] = chain
        out["seasons"][sname] = res
        best = max(((k, L, c) for k, cs in res["corr"].items() for L, c in zip(lags, cs) if c is not None and L >= 0), key=lambda z: abs(z[2]))
        print(f"  {sname}: {len(events)} events; strongest torque→jet correlation {best[0]} at lag +{best[1]} d: r {best[2]:+.2f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-torque", action="store_true")
    a = ap.parse_args()
    REF.mkdir(parents=True, exist_ok=True)
    ds = build_u200()
    zds, zanom = build_z500(ds.time.values)
    ds = ds.merge(zds)
    if not a.skip_torque:
        tq = build_torque()
        ds["him_torque_anom"] = tq.reindex(time=ds.time.values)
        stats = lag_stats(ds, tq, extra={"alaska": ds.alaska.values, "goa": ds.goa.values})
        mo = pd.DatetimeIndex(ds.time.values).month.values
        sel_by = {sname: np.isin(mo, months) for sname, months in SEASONS.items() if sname != "all"}
        comp, track = composites(zanom, ds.time.values, ds.him_torque_anom.values, sel_by, ds.zlat.values, ds.zlon.values)
        comp = comp.assign_coords(zlat=ds.zlat.values, zlon=ds.zlon.values)
        comp.to_netcdf(REF / "pacjet_composites.nc", encoding={v: {"zlib": True, "complevel": 4} for v in comp.data_vars})
        stats["ridge_track"] = track
        OUT_JSON.write_text(json.dumps(stats, separators=(",", ":")))
        print(f"  wrote {OUT_JSON} and pacjet_composites.nc (events {dict(zip(comp.season.values.tolist(), comp.n_events.values.tolist()))})", flush=True)
        for sname, rows in track.items():
            print("  ridge track", sname, "; ".join(f"+{r['lag']}d: {r['amp_m']} m at {r['lat']}N {r['lon']}E (sig {r['sig_area_frac']:.0%})" if r["lat"] is not None else f"+{r['lag']}d: none" for r in rows), flush=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(OUT_NC, encoding=enc)
    print(f"wrote {OUT_NC} ({OUT_NC.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
