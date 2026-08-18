#!/usr/bin/env python3
"""End-to-end backtest of the AREA -> ENERGY basin-weighting change.

Runs the whole operational chain in-process under both weighting bases —
archived AIFS/IFS ensemble rain -> per-band bias correction -> anomaly vs
IMERG harmonic clim -> spliced onto the observed anomaly history -> fitted
EMA kernel -> inflow % of norm — and scores the prediction against XM's
observed inflow.  Everything downstream of the split is fit on TRAIN
cycles only (bias factors AND kernel), so the test cycles are clean.

The forecast is sampled on the area footprint under both bases (see
colombia_forecast._bas); only the TRUTH and the kernel fitted against it
change.  That is the question this answers: does energy-weighting the
truth survive the NWP step, or does it only look good against perfect
rain?

    python scripts/sst/backtest_weights_e2e.py

Writes ~/colombia_hydro/out/backtest_weights_e2e.json
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import (region_weights_area, region_weights_energy,  # noqa: E402
                               gauge_correction, gauge_blend_field)
from build_imerg_clim import OUT as CLIM_NC, eval_clim             # noqa: E402
from rain_inflow_model import ema, trail, TAUS, LAGS, YSMOOTH, TAU_SLOW  # noqa: E402
from colombia_forecast import (ARCH, ORDER, BANDS, K_PRIOR, PRIOR_RATIO,  # noqa: E402
                               REGIONS_GJ, _bas)

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
ICLIM = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
ENSO = REPO / "assets" / "sst" / "data" / "enso_daily.json"
OUT = PRIV / "out" / "backtest_weights_e2e.json"
TRAIN_FRAC = 0.6


def truth_series(W):
    """Gauge-blended corrected IMERG basin rain + clim, on weighting W."""
    import xarray as xr
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    F = gauge_correction(lons, lats)
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    dates, rain, clim = [], {r: [] for r in W}, {r: [] for r in W}
    for f in sorted(IP.DAILY_CACHE.glob("*.npy")):
        g = gauge_blend_field(np.load(f) * F, f.stem, lons, lats)
        doy = min(datetime.strptime(f.stem, "%Y%m%d").timetuple().tm_yday, 365)
        c = eval_clim(coef, doy) * F
        dates.append(f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}")
        for r, w in W.items():
            rain[r].append(float((g * w).sum()))
            clim[r].append(float((c * w).sum()))
    return dates, {r: np.array(v) for r, v in rain.items()}, \
        {r: np.array(v) for r, v in clim.items()}


def fit_kernel(x_an, y, roni, train):
    """v1+ENSO kernel (no storage term — this isolates the rain pathway)."""
    best = None
    for tau in TAUS:
        k = ema(np.nan_to_num(x_an), tau)
        ks = ema(np.nan_to_num(x_an), TAU_SLOW)
        for lag in LAGS:
            kl = np.roll(k, lag); kl[:lag] = np.nan
            ksl = np.roll(ks, lag); ksl[:lag] = np.nan
            X = np.column_stack([kl, ksl, roni])
            m = train & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if m.sum() < 120:
                continue
            A = np.column_stack([np.ones(m.sum()), X[m]])
            b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            sc = np.corrcoef(A @ b, y[m])[0, 1]
            if best is None or sc > best[0]:
                best = (sc, tau, lag, b)
    return best


def main() -> int:
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    WA = region_weights_area(REGIONS_GJ, lons, lats)
    WE = region_weights_energy(lons, lats, ORDER)

    cycles = []
    for f in sorted(glob.glob(str(ARCH / "*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            cycles.append(json.load(fh))
    cycles.sort(key=lambda c: (c["init_date"], c["init_hh"]))
    inits = sorted({c["init_date"] for c in cycles})
    cut = inits[int(len(inits) * TRAIN_FRAC)]
    print(f"{len(cycles)} cycles, {len(inits)} init days — train < {cut} <= test")

    ic = json.loads(ICLIM.read_text())["full_pct_of_norm"]
    iidx = {d: i for i, d in enumerate(ic["dates"])}
    ed = json.loads(ENSO.read_text())["daily"]
    emap = dict(zip(ed["dates"], ed["roni_d"]))

    res = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "cycles": len(cycles), "train_before": cut,
           "note": "forecast sampled on the AREA footprint under both bases; "
                   "only the truth and the kernel fitted to it differ",
           "bases": {}}

    for basis, W in (("area", WA), ("energy", WE)):
        dates, rain, clim = truth_series(W)
        dmap = {d: i for i, d in enumerate(dates)}
        n = len(dates)
        roni = np.array([emap.get(d, np.nan) for d in dates], float)
        for i in range(1, n):
            if not np.isfinite(roni[i]): roni[i] = roni[i - 1]
        for i in range(n - 2, -1, -1):
            if not np.isfinite(roni[i]): roni[i] = roni[i + 1]
        train_day = np.array([d.replace("-", "") < cut for d in dates])

        # --- bias factors from TRAIN cycles only -----------------------------
        acc = {m: {r: [[0.0, 0.0] for _ in BANDS] for r in ORDER}
               for m in ("aifs", "ifs")}
        for c in cycles:
            if c["init_date"] >= cut:
                continue
            d0 = np.datetime64(f"{c['init_date'][:4]}-{c['init_date'][4:6]}-{c['init_date'][6:8]}")
            for li, vd in enumerate(c["valid"]):
                i = dmap.get(vd)
                if i is None: continue
                lead = int((np.datetime64(vd) - d0).astype(int)) + 1
                band = next((bi for bi, (a, b) in enumerate(BANDS) if a <= lead <= b), None)
                if band is None: continue
                for r in ORDER:
                    fc = float(np.mean([mem[li] for mem in _bas(c, r)]))
                    acc[c["model"]][r][band][0] += fc
                    acc[c["model"]][r][band][1] += rain[r][i]
        cmean = {r: float(np.nanmean(clim[r])) for r in ORDER}
        Fac = {m: {r: [] for r in ORDER} for m in ("aifs", "ifs")}
        for m in ("aifs", "ifs"):
            for r in ORDER:
                for bi in range(len(BANDS)):
                    sf, so = acc[m][r][bi]
                    P = K_PRIOR * cmean[r]
                    Fac[m][r].append((so + P * PRIOR_RATIO[m]) / max(sf + P, 1e-6))

        # --- kernels from TRAIN days only ------------------------------------
        ker = {}
        for r in ORDER:
            y = np.full(n, np.nan)
            v = np.array(ic[r], float)
            for i, d in enumerate(dates):
                if d in iidx: y[i] = v[iidx[d]]
            y[y == 0] = np.nan
            y = trail(y, YSMOOTH)
            ker[r] = (fit_kernel(rain[r] - clim[r], y, roni, train_day), y)

        # --- score held-out cycles -------------------------------------------
        rows = {r: {bi: [[], []] for bi in range(len(BANDS))} for r in ORDER}
        for c in cycles:
            if c["init_date"] < cut:
                continue
            d0 = np.datetime64(f"{c['init_date'][:4]}-{c['init_date'][4:6]}-{c['init_date'][6:8]}")
            i0 = dmap.get(c["valid"][0])
            if i0 is None: continue
            for r in ORDER:
                fit, y = ker[r]
                if fit is None: continue
                _, tau, lag, b = fit
                an = (rain[r] - clim[r]).copy()          # observed history
                x = an.copy()
                for li, vd in enumerate(c["valid"]):
                    lead = int((np.datetime64(vd) - d0).astype(int)) + 1
                    band = next((bi for bi, (a, bb) in enumerate(BANDS) if a <= lead <= bb), None)
                    if band is None: continue
                    i = dmap.get(vd)
                    if i is None: continue
                    fc = float(np.mean([mem[li] for mem in _bas(c, r)])) * Fac[c["model"]][r][band]
                    x[i] = fc - clim[r][i]               # splice forecast in
                k = ema(np.nan_to_num(x), tau)
                ks = ema(np.nan_to_num(x), TAU_SLOW)
                for li, vd in enumerate(c["valid"]):
                    lead = int((np.datetime64(vd) - d0).astype(int)) + 1
                    band = next((bi for bi, (a, bb) in enumerate(BANDS) if a <= lead <= bb), None)
                    if band is None: continue
                    i = dmap.get(vd)
                    if i is None or i + lag >= n: continue
                    pred = b[0] + b[1] * k[i] + b[2] * ks[i] + b[3] * roni[i]
                    obs = y[i + lag]
                    if np.isfinite(obs):
                        rows[r][band][0].append(pred)
                        rows[r][band][1].append(obs)

        out = {}
        for r in ORDER:
            out[r] = {}
            for bi, (a, bb) in enumerate(BANDS):
                p, o = np.array(rows[r][bi][0]), np.array(rows[r][bi][1])
                if len(p) < 12:
                    out[r][f"d{a}-{bb}"] = None; continue
                out[r][f"d{a}-{bb}"] = {
                    "n": int(len(p)),
                    "r": round(float(np.corrcoef(p, o)[0, 1]), 3),
                    "mae_pct": round(float(np.mean(np.abs(p - o))), 1),
                    "clim_mae_pct": round(float(np.mean(np.abs(100.0 - o))), 1)}
        res["bases"][basis] = out
        print(f"\n=== {basis.upper()} ===")
        for r in ORDER:
            cells = "  ".join(
                f"{k}: r={v['r']:+.2f} mae={v['mae_pct']:.0f}" if v else f"{k}: --"
                for k, v in out[r].items())
            print(f"  {r:11}{cells}")

    A, E = res["bases"]["area"], res["bases"]["energy"]
    print(f"\n{'region':11}" + "".join(f"{f'd{a}-{b}':>14}" for a, b in BANDS))
    dr, dm = [], []
    for r in ORDER:
        cells = ""
        for a, b in BANDS:
            k = f"d{a}-{b}"
            if A[r][k] and E[r][k]:
                d = E[r][k]["r"] - A[r][k]["r"]; dr.append(d)
                dm.append(A[r][k]["mae_pct"] - E[r][k]["mae_pct"])
                cells += f"{d:+9.3f}{'':5}"
            else:
                cells += f"{'--':>14}"
        print(f"{r:11}{cells}")
    res["summary"] = {"mean_delta_r": round(float(np.mean(dr)), 3),
                      "mean_mae_improvement_pct": round(float(np.mean(dm)), 2),
                      "n_cells": len(dr)}
    print(f"\nEND-TO-END: mean Δr {np.mean(dr):+.3f} over {len(dr)} basin×band cells, "
          f"mean MAE improvement {np.mean(dm):+.2f} pct-of-norm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
