#!/usr/bin/env python3
"""Teleconnection indices (NAO, PNA, EPO, WPO, AO) — computed OURSELVES via
EOF / rotated-PCA on our own Z500 anomaly history (no borrowed point formulas;
CPC's published indices are used only as a validation check).

Method
  1. Daily Z500 anomalies 1979→present on the 1.5° grid, vs our 1991–2020
     harmonic climatology (WeatherBench2 through 2022; CDS daily statistics
     2023→present). NH cap 20–90N, √cos(φ) area weighting.
  2. Monthly means → PCA (SVD); the leading 10 loadings are varimax-rotated
     (CPC's RPCA approach, applied all-calendar-months for fixed year-round
     patterns). AO is the UNROTATED leading EOF of the same matrix — the
     annular mode by construction.
  3. Each rotated mode is identified as NAO / PNA / EPO / WPO by congruence
     with anchor dipoles at the classical centers of action (anchors identify
     and orient the mode; the INDEX is always the projection onto OUR pattern).
  4. Daily index = area-weighted projection of the daily anomaly field onto
     the pattern, standardized by harmonic day-of-year stats over 1991–2020.

Outputs
  assets/climate/teleconnections_daily.csv   (committed)
  scripts/climate/tele_patterns.nc           (committed: loadings + doy norms)
  scripts/climate/data/z500_nh/YYYY.npz      (gitignored daily-anomaly store)

    python scripts/climate/build_z500_indices.py
"""
from __future__ import annotations

import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from climate_monitor import eval_clim, doy365  # noqa: E402

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
START_YEAR = 1979
TELE_CSV = HERE.parents[1] / "assets" / "climate" / "teleconnections_daily.csv"
PATTERNS_NC = HERE / "tele_patterns.nc"
STORE = HERE / "data" / "z500_nh"                 # per-year npz of daily NH anomalies
LAT_MIN = 20.0
N_MODES = 10
BASE = (1991, 2020)
TELE_ORDER = ["nao", "pna", "epo", "wpo", "ao"]

# Anchor dipoles at classical centers of action — used ONLY to pick which
# rotated mode is which and to fix its sign, never to compute the index.
ANCHORS = {
    "nao": [(-1.0, 64.0, 338.0), (+1.0, 38.0, 335.0)],                    # Iceland vs Azores
    "pna": [(+1.0, 20.0, 200.0), (-1.0, 45.0, 195.0),
            (+1.0, 55.0, 245.0), (-1.0, 30.0, 275.0)],                    # Wallace–Gutzler centers
    "epo": [(+1.0, 28.0, 215.0), (-1.0, 60.0, 215.0)],                    # subtropics vs Alaska
    "wpo": [(+1.0, 60.0, 155.0), (-1.0, 30.0, 155.0)],                    # Kamchatka vs subtropical WPac
}


# ── daily anomaly store ──────────────────────────────────────────────────────
def _grid():
    clim = xr.open_dataset(HERE / "era5_clim_z500.nc")
    lats, lons = clim.latitude.values, clim.longitude.values
    sel = lats >= LAT_MIN
    return clim["coef"].values, lats, lons, sel


def _store_year(year: int, times, fields, coef, sel):
    """fields: (n, nlat, nlon) daily-mean Z500 (m). Save NH anomalies float16."""
    out_t, out_a = [], []
    for tt, f in zip(times, fields):
        ts = pd.Timestamp(tt).normalize()
        anom = f - eval_clim(coef, doy365(ts))
        out_t.append(ts.strftime("%Y-%m-%d"))
        out_a.append(anom[sel].astype("float16"))
    STORE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE / f"{year}.npz", dates=np.array(out_t), anoms=np.array(out_a))


def build_store():
    coef, lats, lons, sel = _grid()
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(WB2), chunks={"time": 1464})
    da = ds["geopotential"].sel(level=500)
    for yr in range(START_YEAR, 2023):
        if (STORE / f"{yr}.npz").exists():
            continue
        sely = da.sel(time=str(yr))
        daily = (sely.resample(time="1D").mean()
                 .transpose("time", "latitude", "longitude").compute()) / 9.80665
        _store_year(yr, daily.time.values, daily.values, coef, sel)
        print(f"  WB2 z500 {yr} stored", flush=True)

    try:
        import cdsapi
        import tempfile
        c = cdsapi.Client(quiet=True)
        today = datetime.now(timezone.utc).date()
        for yr in range(2023, today.year + 1):
            p = STORE / f"{yr}.npz"
            have = set(np.load(p, allow_pickle=True)["dates"]) if p.exists() else set()
            want = [d for d in pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
                    if d.strftime("%Y-%m-%d") not in have and d.date() < today]
            if not want:
                continue
            months = sorted({d.month for d in want})
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
                target = tf.name
            print(f"  CDS z500 request {yr} months {months} …", flush=True)
            c.retrieve("derived-era5-pressure-levels-daily-statistics", {
                "product_type": "reanalysis", "variable": "geopotential",
                "pressure_level": "500", "year": str(yr),
                "month": [f"{m:02d}" for m in months],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "daily_statistic": "daily_mean", "time_zone": "utc+00:00",
                "frequency": "1-hourly", "grid": "1.5/1.5",
            }, target)
            dsy = xr.open_dataset(target)
            v = dsy[[k for k in dsy.data_vars if dsy[k].ndim >= 3][0]]
            tname = "valid_time" if "valid_time" in v.dims else "time"
            v = v.rename({tname: "time"}).sortby("latitude")
            v = v.interp(latitude=lats, longitude=lons)
            vals = v.transpose("time", "latitude", "longitude").values / 9.80665
            merged = {}
            if p.exists():
                z = np.load(p, allow_pickle=True)
                merged = {d: a for d, a in zip(z["dates"], z["anoms"])}
            for tt, f in zip(v.time.values, vals):
                ts = pd.Timestamp(tt).normalize()
                anom = f - eval_clim(coef, doy365(ts))
                merged[ts.strftime("%Y-%m-%d")] = anom[sel].astype("float16")
            ks = sorted(merged)
            STORE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, dates=np.array(ks),
                                anoms=np.array([merged[k] for k in ks]))
            print(f"  CDS z500 {yr}: {len(v.time)} days stored", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  CDS part failed/skipped ({repr(e)[:90]})", file=sys.stderr)


def load_store():
    dates, anoms = [], []
    for p in sorted(STORE.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        dates += list(z["dates"])
        anoms.append(z["anoms"].astype("float32"))
    return pd.DatetimeIndex(dates), np.concatenate(anoms)    # (ndays, nlat_nh, nlon)


# ── EOF / rotated PCA ────────────────────────────────────────────────────────
def varimax(L: np.ndarray, gamma: float = 1.0, iters: int = 200, tol: float = 1e-7):
    """Varimax rotation of loading matrix L (n_cells × k)."""
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.sum(Lr ** 2, axis=0))))
        R = u @ vt
        d_new = float(np.sum(s))
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def fit_patterns(dates: pd.DatetimeIndex, anoms: np.ndarray, lats_nh, lons):
    """Monthly PCA + varimax; returns identified patterns (2-D, in weighted
    space) with signs fixed by the anchors, plus unrotated EOF1 as AO."""
    nlat, nlon = anoms.shape[1:]
    w = np.sqrt(np.cos(np.deg2rad(lats_nh)))[:, None] * np.ones((1, nlon))
    monthly = pd.DataFrame(
        (anoms * w).reshape(len(dates), -1)).set_index(dates).resample("MS").mean()
    base = monthly[(monthly.index.year >= BASE[0]) & (monthly.index.year <= BASE[1])]
    X = base.values - base.values.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    expl = (S ** 2 / np.sum(S ** 2))[:N_MODES]
    L = (Vt[:N_MODES].T * S[:N_MODES]) / np.sqrt(len(X) - 1)     # loadings (cells × k)
    Lr = varimax(L)

    patterns = {"ao": Vt[0].reshape(nlat, nlon)}                 # annular mode
    if patterns["ao"][lats_nh >= 70].mean() > 0:                 # AO+ = LOW polar heights
        patterns["ao"] = -patterns["ao"]

    def anchor_score(pat2d, anchors):
        return sum(amp * pat2d[int(np.argmin(np.abs(lats_nh - la))),
                               int(np.argmin(np.abs(lons % 360 - lo)))]
                   for amp, la, lo in anchors)

    taken = set()
    for name, anchors in ANCHORS.items():
        best, best_j, best_sign = -np.inf, None, 1.0
        for j in range(N_MODES):
            if j in taken:
                continue
            pat = Lr[:, j].reshape(nlat, nlon)
            sc = anchor_score(pat / (np.abs(pat).max() + 1e-9), anchors)
            if abs(sc) > best:
                best, best_j, best_sign = abs(sc), j, float(np.sign(sc))
        taken.add(best_j)
        patterns[name] = best_sign * Lr[:, best_j].reshape(nlat, nlon)
        print(f"  {name.upper():4} ← rotated mode {best_j + 1} "
              f"(|anchor congruence| {best:.2f})", flush=True)
    print(f"  explained variance (top {N_MODES}): "
          + " ".join(f"{e:.0%}" for e in expl), flush=True)
    return patterns, w


def project(anoms: np.ndarray, patterns: dict, w: np.ndarray) -> pd.DataFrame:
    """Raw index = area-weighted projection of each daily field onto a pattern."""
    flat = (anoms * w).reshape(anoms.shape[0], -1)
    out = {}
    for name, pat in patterns.items():
        v = pat.reshape(-1)
        out[name] = flat @ v / (v @ v)
    return pd.DataFrame(out)


def harmonic_doy_stats(series: pd.Series, nharm: int = 3):
    s = series[(series.index.year >= BASE[0]) & (series.index.year <= BASE[1])]
    doy = np.array([doy365(t) for t in s.index])
    mean_raw = np.full(365, np.nan)
    std_raw = np.full(365, np.nan)
    for d in range(365):
        v = s.values[doy == d]
        if len(v) >= 10:
            mean_raw[d] = np.mean(v)
            std_raw[d] = np.std(v)
    x = 2 * np.pi * np.arange(365) / 365.0
    cols = [np.ones(365)]
    for h in range(1, nharm + 1):
        cols += [np.cos(h * x), np.sin(h * x)]
    A = np.column_stack(cols)
    ok = ~np.isnan(mean_raw)
    mu = A @ np.linalg.lstsq(A[ok], mean_raw[ok], rcond=None)[0]
    sg = A @ np.linalg.lstsq(A[ok], std_raw[ok], rcond=None)[0]
    return mu, np.maximum(sg, 1e-9)


# ── CPC validation (check only — never an input) ─────────────────────────────
def _cpc_monthly():
    out = {}
    try:
        txt = urllib.request.urlopen(
            "https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh",
            timeout=60).read().decode(errors="ignore")
        rows, header = {}, None
        for line in txt.splitlines():
            p = line.split()
            if p and p[0].lower() == "yyyy":
                header = p
            elif header and len(p) == len(header) and p[0].isdigit():
                for name, val in zip(header[2:], p[2:]):
                    try:
                        rows.setdefault(name.upper(), {})[
                            pd.Timestamp(int(p[0]), int(p[1]), 1)] = float(val)
                    except ValueError:
                        pass
        for cpc, ours in (("NAO", "nao"), ("PNA", "pna"), ("WP", "wpo")):
            if cpc in rows:
                out[ours] = pd.Series(rows[cpc]).replace(-99.9, np.nan)
    except Exception as e:                                   # noqa: BLE001
        print(f"  CPC tele_index.nh unavailable ({repr(e)[:60]})", flush=True)
    try:
        txt = urllib.request.urlopen(
            "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/"
            "monthly.ao.index.b50.current.ascii", timeout=60).read().decode(errors="ignore")
        vals = {}
        for line in txt.splitlines():
            p = line.split()
            if len(p) == 3 and p[0].isdigit():
                vals[pd.Timestamp(int(p[0]), int(p[1]), 1)] = float(p[2])
        out["ao"] = pd.Series(vals)
    except Exception as e:                                   # noqa: BLE001
        print(f"  CPC AO table unavailable ({repr(e)[:60]})", flush=True)
    return out


def validate(daily: pd.DataFrame):
    monthly = daily.resample("MS").mean()
    cpc = _cpc_monthly()
    print("  validation vs CPC monthly (methodology check only):", flush=True)
    for k in TELE_ORDER:
        if k in cpc:
            j = monthly[k].to_frame("ours").join(cpc[k].rename("cpc")).dropna()
            r = j["ours"].corr(j["cpc"]) if len(j) > 24 else np.nan
            print(f"    {k.upper():4} r = {r:+.2f}  (n={len(j)})", flush=True)
        else:
            print(f"    {k.upper():4} — no CPC counterpart (operational index)", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    coef, lats, lons, sel = _grid()
    lats_nh = lats[sel]
    build_store()
    dates, anoms = load_store()
    print(f"  store: {len(dates)} days ({dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d})", flush=True)

    patterns, w = fit_patterns(dates, anoms, lats_nh, lons)
    raw = project(anoms, patterns, w).set_index(dates)

    norms, z = {}, {}
    for k in TELE_ORDER:
        mu, sg = harmonic_doy_stats(raw[k])
        norms[f"{k}_mu"] = ("doy", mu.astype("float32"))
        norms[f"{k}_sd"] = ("doy", sg.astype("float32"))
        doy = np.array([doy365(t) for t in raw.index])
        z[k] = (raw[k].values - mu[doy]) / sg[doy]
    daily = pd.DataFrame(z, index=raw.index).round(3)

    pat_arr = np.stack([patterns[k] for k in TELE_ORDER])
    xr.Dataset(
        {"pattern": (("index", "lat", "lon"), pat_arr.astype("float32")), **norms},
        coords={"index": TELE_ORDER, "lat": lats_nh, "lon": lons,
                "doy": np.arange(365)},
        attrs={"method": "monthly PCA of Z500 anomalies, varimax-rotated "
                         "(AO = unrotated EOF1), anchors identify/orient modes only; "
                         f"base {BASE[0]}-{BASE[1]}, NH {LAT_MIN}-90N, sqrt-cos weights",
               "built": str(datetime.now(timezone.utc))},
    ).to_netcdf(PATTERNS_NC)
    print(f"  wrote {PATTERNS_NC.name}", flush=True)

    daily.to_csv(TELE_CSV, index_label="date")
    print(f"  wrote {TELE_CSV.name}: {len(daily)} days", flush=True)
    validate(daily)
    return 0


if __name__ == "__main__":
    sys.exit(main())
