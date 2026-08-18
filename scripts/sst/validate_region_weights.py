#!/usr/bin/env python3
"""Validate the AREA -> ENERGY region-weighting change, end to end.

For every region, rebuild the basin rain series under both weighting bases,
refit the full v3 model (fast EMA + slow EMA180 + RONI + storage anomaly) and
report:
  in-sample r, split-half out-of-sample r (fit on one half, score the other),
  and the same for the kernel-only term.
Also reports how much of each region's inflow energy the weights actually
cover, and which rivers are excluded as regulated.

Output: ~/colombia_hydro/out/region_weights_validation.json
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                            # noqa: E402
from hydro_region_rain import (region_weights_area, region_weights_energy,   # noqa: E402
                               gauge_correction, gauge_blend_field,
                               _river_energy, _regulated_rivers)
from build_imerg_clim import OUT as CLIM_NC, eval_clim               # noqa: E402
from rain_inflow_model import ema, trail, TAUS, LAGS, YSMOOTH, TAU_SLOW  # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
GJ = HERE / "colombia_hydro_regions.geojson"
ICLIM = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
ENSO = REPO / "assets" / "sst" / "data" / "enso_daily.json"
OUT = PRIV / "out" / "region_weights_validation.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


_SCACHE = PRIV / "raw" / "validate_series_cache.npz"


def build_series(W, tag=""):
    import xarray as xr
    nf = len(list(IP.DAILY_CACHE.glob("*.npy")))
    if _SCACHE.exists():
        try:
            z = np.load(_SCACHE, allow_pickle=True)
            d = z["d"].item()
            if d.get("nf") == nf and tag in d:
                e = d[tag]
                return e["dates"], e["rain"], e["clim"]
        except Exception:                               # noqa: BLE001
            pass
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml]); lats = np.sort(IP._LAT[mt])
    F = gauge_correction(lons, lats)
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    dates, rain, clim = [], {r: [] for r in W}, {r: [] for r in W}
    for f in files:
        g = gauge_blend_field(np.load(f) * F, f.stem, lons, lats)
        doy = min(datetime.strptime(f.stem, "%Y%m%d").timetuple().tm_yday, 365)
        c = eval_clim(coef, doy) * F
        dates.append(f.stem)
        for r, w in W.items():
            rain[r].append(float((g * w).sum()))
            clim[r].append(float((c * w).sum()))
    rain = {r: np.array(v) for r, v in rain.items()}
    clim = {r: np.array(v) for r, v in clim.items()}
    if tag:
        d = {}
        if _SCACHE.exists():
            try:
                z = np.load(_SCACHE, allow_pickle=True)
                d = z["d"].item()
                if d.get("nf") != nf:
                    d = {}
            except Exception:                           # noqa: BLE001
                d = {}
        d["nf"] = nf
        d[tag] = {"dates": dates, "rain": rain, "clim": clim}
        np.savez_compressed(_SCACHE, d=np.array(d, dtype=object))
    return dates, rain, clim


def fit_eval(x_an, y, S, roni, train, test):
    """Best (tau,lag) chosen on train; r reported on test."""
    best = None
    for tau in TAUS:
        k = ema(np.where(np.isfinite(x_an), x_an, 0), tau)
        ks = ema(np.where(np.isfinite(x_an), x_an, 0), TAU_SLOW)
        for lag in LAGS:
            kl = np.roll(k, lag); kl[:lag] = np.nan
            ksl = np.roll(ks, lag); ksl[:lag] = np.nan
            X = np.column_stack([kl, ksl, roni, S])
            m = train & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if m.sum() < 120:
                continue
            A = np.column_stack([np.ones(m.sum()), X[m]])
            b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            fit = A @ b
            sc = np.corrcoef(fit, y[m])[0, 1]
            if best is None or sc > best[0]:
                best = (sc, tau, lag, b, X)
    if best is None:
        return np.nan, np.nan, None
    _, tau, lag, b, X = best
    mt_ = test & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    pred = b[0] + X[mt_] @ b[1:]
    ok = mt_.sum() > 30
    r_te = float(np.corrcoef(pred, y[mt_])[0, 1]) if ok else np.nan
    mae_te = float(np.mean(np.abs(pred - y[mt_]))) if ok else np.nan
    mtr = train & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    ptr = b[0] + X[mtr] @ b[1:]
    r_tr = float(np.corrcoef(ptr, y[mtr])[0, 1])
    return r_tr, r_te, (tau, lag), float(np.mean(np.abs(ptr - y[mtr]))), mae_te


def main() -> int:
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml]); lats = np.sort(IP._LAT[mt])
    WA = region_weights_area(GJ, lons, lats)
    WE = region_weights_energy(lons, lats, ORDER)
    if WE is None:
        raise SystemExit("energy weights unavailable")
    dates, rain_a, clim_a = build_series(WA, "area")
    _, rain_e, clim_e = build_series(WE, "energy")

    ic = json.loads(ICLIM.read_text())["full_pct_of_norm"]
    idx = {d: i for i, d in enumerate(ic["dates"])}
    ed = json.loads(ENSO.read_text())["daily"]
    emap = dict(zip(ed["dates"], ed["roni_d"]))
    with gzip.open(PRIV / "raw" / "storage_daily.json.gz", "rt") as f:
        pass
    from xm_storage import pct_anomaly_series
    sdates, sanom = pct_anomaly_series()
    smap_ = {str(d): {r: sanom[r][i] for r in ORDER} for i, d in enumerate(sdates)}

    n = len(dates)
    dd = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates]
    roni = np.array([emap.get(x, np.nan) for x in dd], float)
    for i in range(1, n):
        if not np.isfinite(roni[i]): roni[i] = roni[i - 1]
    for i in range(n - 2, -1, -1):
        if not np.isfinite(roni[i]): roni[i] = roni[i + 1]

    egy = _river_energy(); reg = _regulated_rivers()
    res = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "regulated_excluded": sorted(reg), "days": n,
           "window": f"{dates[0]}..{dates[-1]}", "regions": {}}
    h = n // 2
    f1 = np.zeros(n, bool); f1[:h] = True
    f2 = ~f1
    print(f"{'region':10} {'basis':7} {'in-samp r':>10} {'OOS h1→h2':>10} "
          f"{'OOS h2→h1':>10} {'MAE in':>8} {'MAE OOS':>8}")
    for r in ORDER:
        y = np.array([np.nan] * n)
        v = np.array(ic[r], float)
        for i, x in enumerate(dd):
            if x in idx: y[i] = v[idx[x]]
        y[y == 0] = np.nan
        y = trail(y, YSMOOTH)
        S = np.array([smap_.get(x, {}).get(r, np.nan) for x in dd], float)
        S = np.roll(S, 1); S[0] = np.nan
        for i in range(1, n):
            if not np.isfinite(S[i]): S[i] = S[i - 1]
        S = np.nan_to_num(S)
        row = {}
        for basis, rain, clim in (("area", rain_a, clim_a), ("energy", rain_e, clim_e)):
            x_an = rain[r] - clim[r]
            r_in, _, tl, mae_in, _ = fit_eval(x_an, y, S, roni,
                                              np.ones(n, bool), np.ones(n, bool))
            _, r12, _, _, m12 = fit_eval(x_an, y, S, roni, f1, f2)
            _, r21, _, _, m21 = fit_eval(x_an, y, S, roni, f2, f1)
            mo = float(np.nanmean([m12, m21]))
            row[basis] = {"r_in_sample": round(float(r_in), 3),
                          "r_oos_h1_to_h2": round(float(r12), 3),
                          "r_oos_h2_to_h1": round(float(r21), 3),
                          "mae_in_sample_pct": round(float(mae_in), 1),
                          "mae_oos_pct": round(mo, 1), "tau_lag": tl}
            print(f"{r:10} {basis:7} {r_in:10.3f} {r12:10.3f} {r21:10.3f}"
                  f"{mae_in:9.1f}{mo:9.1f}")
        row["delta_in_sample"] = round(row["energy"]["r_in_sample"] - row["area"]["r_in_sample"], 3)
        row["delta_mae_oos"] = round(row["energy"]["mae_oos_pct"]
                                     - row["area"]["mae_oos_pct"], 2)
        row["delta_oos_mean"] = round(
            np.nanmean([row["energy"]["r_oos_h1_to_h2"], row["energy"]["r_oos_h2_to_h1"]])
            - np.nanmean([row["area"]["r_oos_h1_to_h2"], row["area"]["r_oos_h2_to_h1"]]), 3)
        # energy coverage
        cov = {riv: round(egy.get(riv, 0), 2) for riv in egy}
        res["regions"][r] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    ds = [res["regions"][r]["delta_oos_mean"] for r in ORDER]
    di = [res["regions"][r]["delta_in_sample"] for r in ORDER]
    dmae = [res["regions"][r]["delta_mae_oos"] for r in ORDER]
    print(f"\nMEAN Δr in-sample {np.mean(di):+.3f} | Δr out-of-sample "
          f"{np.mean(ds):+.3f} | Δ MAE out-of-sample {np.mean(dmae):+.2f} pts "
          "(negative = energy better)")
    print("regulated excluded:", sorted(reg))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
