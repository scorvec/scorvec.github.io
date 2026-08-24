#!/usr/bin/env python3
"""
Step 0 of the elevation-weighting question: does high-elevation rain carry
more inflow signal than the flat catchment mean?

Builds per-region elevation-band rain series (low/mid/high weight-mass
terciles of the ENERGY-weighted region masks — the same basis as the model's
truth cache) from the full gridded IMERG daily cache (2000-06 → present) and
ETOPO 2022 elevations, then screens each band against the kernel model's
actual target: raw daily Δy (inflow % of norm), using the model's dominant
hyperparameters (EMA tau=2, lag 0).

Outputs (private repo):
  ~/colombia_hydro/out/elevation_band_test.json
  ~/colombia_hydro/raw/etopo_colombia.nc        (cached DEM subset)
  ~/colombia_hydro/raw/elev_band_rain.json.gz   (band series cache)

    python scripts/sst/elevation_band_test.py
"""
from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imerg_precip as IP                              # noqa: E402
from hydro_region_rain import region_weights_energy    # noqa: E402

PRIV = Path.home() / "colombia_hydro"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
DEM_NC = PRIV / "raw" / "etopo_colombia.nc"
BAND_CACHE = PRIV / "raw" / "elev_band_rain.json.gz"
OUT = PRIV / "out" / "elevation_band_test.json"
INFLOW_JSON = HERE.parent.parent / "colombia_hydro" / "data" / "inflow_clim.json"
DAILY = HERE / "data" / "imerg" / "daily"

ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BANDS = ["low", "mid", "high"]
TAU, LAG = 2.0, 0                    # the model's dominant fitted kernel/lag

ETOPO_NCSS = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
              "?altitude%5B(-2):(13)%5D%5B(-81):(-69)%5D")   # ETOPO1, 1 arc-min


def dem_on_grid(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    if not DEM_NC.exists():
        print("downloading ETOPO 2022 subset…", flush=True)
        urllib.request.urlretrieve(ETOPO_NCSS, DEM_NC)
    ds = xr.open_dataset(DEM_NC)
    z = ds[[v for v in ds.data_vars if v in ("z", "altitude")][0]]
    la = "latitude" if "latitude" in z.dims else "lat"
    lo = "longitude" if "longitude" in z.dims else "lon"
    lon180 = np.where(lons > 180, lons - 360, lons)
    zi = z.sortby(la).interp({la: xr.DataArray(lats, dims="y"),
                              lo: xr.DataArray(lon180, dims="x")})
    return np.maximum(np.asarray(zi.values, float), 0.0)   # (lat, lon), sea→0


def band_weights(W: dict[str, np.ndarray], elev_flat: np.ndarray):
    """Split each region's flat weight vector into weight-mass terciles by
    elevation. Returns {(region, band): normalized flat weights} + stats."""
    out, stats = {}, {}
    for r, w in W.items():
        w = w.ravel()
        idx = np.where(w > 0)[0]
        order = idx[np.argsort(elev_flat[idx])]
        cum = np.cumsum(w[order]) / w[order].sum()
        edges = np.searchsorted(cum, [1 / 3, 2 / 3])
        groups = np.split(order, edges)
        stats[r] = {}
        for b, g in zip(BANDS, groups):
            wb = np.zeros_like(w)
            wb[g] = w[g]
            out[(r, b)] = wb / wb.sum()
            stats[r][b] = dict(
                mean_elev_m=float(np.average(elev_flat[g], weights=w[g])),
                min_elev_m=float(elev_flat[g].min()),
                max_elev_m=float(elev_flat[g].max()),
                n_cells=int(len(g)))
        out[(r, "flat")] = w / w.sum()
    return out, stats


def sweep_daily(vecs: dict) -> tuple[list[str], np.ndarray]:
    keys = list(vecs)
    M = np.stack([vecs[k] for k in keys])               # (nseries, ncells)
    files = sorted(DAILY.glob("*.npy"))
    dates, rows = [], np.empty((len(files), len(keys)))
    for i, f in enumerate(files):
        g = np.load(f).ravel()
        rows[i] = M @ np.nan_to_num(g, nan=0.0)
        dates.append(f.stem)
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    return dates, rows


def harmonic_clim(dates: np.ndarray, x: np.ndarray, nharm: int = 3) -> np.ndarray:
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    A = [np.ones_like(doy, float)]
    for k in range(1, nharm + 1):
        A += [np.cos(2 * np.pi * k * doy / 365.25),
              np.sin(2 * np.pi * k * doy / 365.25)]
    A = np.column_stack(A)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return np.maximum(A @ coef, 0.0)


def ema(x: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    a = 1.0 / tau
    for i in range(1, len(x)):
        out[i] = (1 - a) * out[i - 1] + a * x[i]
    return out


def main() -> int:
    # grid axes of the daily cache
    ml, mt = IP._grid_axes()
    lons, lats = IP._LON[ml] % 360, IP._LAT[mt]

    W = region_weights_energy(lons, lats, ORDER)
    elev = dem_on_grid(lats, lons).ravel()
    vecs, band_stats = band_weights(W, elev)
    for r in ORDER:
        s = band_stats[r]
        print(f"{r:10s} band mean elev (m): "
              + "  ".join(f"{b}={s[b]['mean_elev_m']:5.0f}" for b in BANDS))

    if BAND_CACHE.exists():
        c = json.loads(gzip.decompress(BAND_CACHE.read_bytes()))
        dates, keys = c["dates"], [tuple(k) for k in c["keys"]]
        rows = np.array(c["rows"])
        print(f"band cache: {len(dates)} days")
    else:
        print("sweeping gridded dailies…", flush=True)
        dates, rows = sweep_daily(vecs)
        keys = list(vecs)
        BAND_CACHE.write_bytes(gzip.compress(json.dumps(
            {"dates": dates, "keys": [list(k) for k in keys],
             "rows": np.round(rows, 4).tolist()}).encode()))
    dd = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates],
                  dtype="datetime64[D]")
    series = {k: rows[:, i] for i, k in enumerate(keys)}

    # sanity: flat reconstruction vs the model's truth cache
    tc = json.loads(TRUTH.read_text())
    td = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in tc["dates"]],
                  dtype="datetime64[D]")
    common, ai, bi = np.intersect1d(dd, td, return_indices=True)
    sanity = {}
    for r in ORDER:
        a, b = series[(r, "flat")][ai], np.array(tc[r], float)[bi]
        m = np.isfinite(a) & np.isfinite(b)
        sanity[r] = float(np.corrcoef(a[m], b[m])[0, 1])
    print("flat-vs-truth corr:", {r: round(v, 4) for r, v in sanity.items()})

    # inflow target
    inf = json.loads(INFLOW_JSON.read_text())["recent"]
    idl = np.array(inf["dates"], dtype="datetime64[D]")
    results = {}
    for r in ORDER:
        y = np.array(inf["pct_of_norm"][r], float)
        y[y == 0] = np.nan
        dy = np.full(len(y), np.nan)
        dy[1:] = y[1:] - y[:-1]
        res = {}
        for b in BANDS + ["flat"]:
            x = series[(r, b)]
            anom = x - harmonic_clim(dd, x)
            k = ema(anom, TAU)
            # align to inflow dates
            _, xi, yi = np.intersect1d(dd, idl, return_indices=True)
            xk, xr_, dyv = k[xi], anom[xi], dy[yi]
            if LAG:
                xk, xr_, dyv = xk[:-LAG], xr_[:-LAG], dyv[LAG:]
            m = np.isfinite(xk) & np.isfinite(dyv)
            n = int(m.sum())
            r_k = float(np.corrcoef(xk[m], dyv[m])[0, 1])
            r_x = float(np.corrcoef(xr_[m], dyv[m])[0, 1])
            h = n // 2
            res[b] = dict(r_kernel=round(r_k, 4), r_rainday=round(r_x, 4),
                          r_kernel_h1=round(float(np.corrcoef(
                              xk[m][:h], dyv[m][:h])[0, 1]), 4),
                          r_kernel_h2=round(float(np.corrcoef(
                              xk[m][h:], dyv[m][h:])[0, 1]), 4),
                          n=n)
        # joint 3-band OLS on dy: relative per-band coefficients
        _, xi, yi = np.intersect1d(dd, idl, return_indices=True)
        Xb = np.column_stack([
            ema(series[(r, b)] - harmonic_clim(dd, series[(r, b)]), TAU)[xi]
            for b in BANDS])
        dyv = dy[yi]
        m = np.isfinite(Xb).all(1) & np.isfinite(dyv)
        A = np.column_stack([np.ones(m.sum()), Xb[m]])
        coef, *_ = np.linalg.lstsq(A, dyv[m], rcond=None)
        res["joint_coef"] = {b: round(float(c), 4)
                             for b, c in zip(BANDS, coef[1:])}
        results[r] = res
        jc = res["joint_coef"]
        print(f"{r:10s} r_kernel " +
              " ".join(f"{b}={res[b]['r_kernel']:+.3f}" for b in BANDS + ['flat'])
              + f"   joint {jc}")

    OUT.write_text(json.dumps(
        {"generated": "2026-08-24", "tau": TAU, "lag": LAG,
         "weights_basis": "energy (same as truth cache)",
         "band_stats": band_stats, "sanity_flat_vs_truth": sanity,
         "results": results}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
