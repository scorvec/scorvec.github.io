#!/usr/bin/env python3
"""Per-basin rain→inflow model: exponential-memory IMERG rainfall vs
fleet-corrected inflow % of norm.

For each region: daily IMERG basin-mean rainfall anomaly (vs the harmonic
IMERG climatology) is passed through a causal exponential memory kernel
(EMA, time constant τ) and lagged; the fleet-corrected inflow
%-of-normal (from inflow_clim.json) is regressed on it. τ and lag are
scanned per basin; the best fit, the plain same-day correlation (for
comparison), and the fitted parameters are reported. A constant
multiplicative satellite bias cancels in correlation — the bias factors
matter for AIFS calibration, not for r.

Outputs:
  colombia_hydro/rain_inflow_model.webp     (per-region obs vs fitted + table)
  colombia_hydro/data/rain_inflow_model.json (params for the forecast step)

    python scripts/sst/rain_inflow_model.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                        # noqa: E402
from hydro_region_rain import region_weights, gauge_correction  # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim  # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
ENSO_JSON = REPO / "assets" / "sst" / "data" / "enso_daily.json"
OUT_PNG = REPO / "colombia_hydro" / "rain_inflow_model.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "rain_inflow_model.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
TAUS = [2, 4, 7, 12, 20, 30, 45, 60]
LAGS = range(0, 8)
YSMOOTH = 5                       # trailing-mean days on the inflow side
TAU_SLOW = 90                     # slow soil-moisture EMA, days


def ema(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential moving average, NaNs treated as climatology (0)."""
    a = 1.0 / tau
    out = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        vv = 0.0 if not np.isfinite(v) else v
        acc = (1 - a) * acc + a * vv
        out[i] = acc
    return out


def trail(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return np.convolve(np.where(np.isfinite(x), x, np.nan), k, mode="full")[:len(x)]


def main() -> int:
    # ── rain side: regional daily means over the whole IMERG cache ─────────
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    rdays = [f.stem for f in files]
    rdates = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in rdays], dtype="datetime64[D]")
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml])          # native -180..180 — geojson convention
    lats = np.sort(IP._LAT[mt])
    W = region_weights(REGIONS_GJ, lons, lats)
    F = gauge_correction(lons, lats)      # corrected = IMERG * F (1.0 off-footprint)
    import xarray as xr
    clim = xr.open_dataset(CLIM_NC)["coef"].values
    doys = np.array([min(int(np.datetime64(d).astype("datetime64[D]").item().timetuple().tm_yday), 365)
                     for d in rdates])
    rain = {r: np.full(len(files), np.nan) for r in ORDER}
    rain_clim = {r: np.full(len(files), np.nan) for r in ORDER}
    for i, f in enumerate(files):
        g = np.load(f) * F                # gauge-corrected rain; clim gets the
        c = eval_clim(clim, doys[i]) * F  # same field so anomalies stay consistent
        for r in ORDER:
            w = W[r]
            sw = w.sum()
            rain[r][i] = float((g * w).sum() / sw)
            rain_clim[r][i] = float((c * w).sum() / sw)
    print(f"rain series: {rdays[0]}..{rdays[-1]} ({len(files)} days)", flush=True)

    # ── inflow side: fleet-corrected % of norm ─────────────────────────────
    inf = json.loads(INFLOW_JSON.read_text())
    idates = np.array(inf["recent"]["dates"], dtype="datetime64[D]")
    pct = {r: np.array(inf["recent"]["pct_of_norm"][r], dtype=float) for r in ORDER}

    # align on common dates
    common, ri, ii = np.intersect1d(rdates, idates, return_indices=True)
    print(f"overlap: {common[0]}..{common[-1]} ({len(common)} days)", flush=True)

    # daily RONI on the common axis (the ET / antecedent-dryness channel:
    # El Nino = hotter, sunnier, drier soils -> less runoff per mm of rain)
    ed = json.loads(ENSO_JSON.read_text())["daily"]
    edates = np.array(ed["dates"], dtype="datetime64[D]")
    eroni = np.array(ed["roni_d"], dtype=float)
    roni = np.full(len(common), np.nan)
    _, ci, ei = np.intersect1d(common, edates, return_indices=True)
    roni[ci] = eroni[ei]
    # persistence-fill edges (RONI moves on monthly scales)
    for i in range(1, len(roni)):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i - 1]
    for i in range(len(roni) - 2, -1, -1):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i + 1]

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.5), sharex=True)
    params = {}
    t = common.astype("datetime64[s]").astype(datetime)
    for ax, r in zip(axes.flat, ORDER):
        x_anom = (rain[r] - rain_clim[r])[ri]                 # mm/day anomaly
        y = trail(pct[r], YSMOOTH)[ii]                        # smoothed % of norm
        ok0 = np.isfinite(y)
        best = None
        for tau in TAUS:
            k = ema(x_anom, tau)
            for lag in LAGS:
                xl = np.roll(k, lag)
                xl[:lag] = np.nan
                m = ok0 & np.isfinite(xl)
                if m.sum() < 120:
                    continue
                rr = float(np.corrcoef(xl[m], y[m])[0, 1])
                if best is None or rr > best[0]:
                    best = (rr, tau, lag, xl, m)
        r_daily = float(np.corrcoef(x_anom[np.isfinite(x_anom) & ok0],
                                    y[np.isfinite(x_anom) & ok0])[0, 1])
        rr, tau, lag, xl, m = best
        b, a = np.polyfit(xl[m], y[m], 1)
        fit = a + b * xl
        # + ENSO/ET term: y = a2 + b2*kernel + c2*RONI. c2<0 = same rain,
        # less inflow under El Nino (evapotranspiration + dry soils).
        m2 = m & np.isfinite(roni)
        X = np.column_stack([np.ones(m2.sum()), xl[m2], roni[m2]])
        beta, *_ = np.linalg.lstsq(X, y[m2], rcond=None)
        fit2 = beta[0] + beta[1] * xl + beta[2] * roni
        r2 = float(np.corrcoef(fit2[m2], y[m2])[0, 1])
        # v2: + slow soil-moisture kernel (90-day EMA of the same anomaly) —
        # antecedent wetness the fast kernel forgets; sharper wet-dry turns
        kslow = ema(x_anom, TAU_SLOW)
        ksl = np.roll(kslow, lag)
        ksl[:lag] = np.nan
        m3 = m2 & np.isfinite(ksl)
        X3 = np.column_stack([np.ones(m3.sum()), xl[m3], roni[m3], ksl[m3]])
        beta3, *_ = np.linalg.lstsq(X3, y[m3], rcond=None)
        fit3 = beta3[0] + beta3[1] * xl + beta3[2] * roni + beta3[3] * ksl
        r3 = float(np.corrcoef(fit3[m3], y[m3])[0, 1])
        params[r] = {"tau_days": tau, "lag_days": lag, "r": round(rr, 3),
                     "r_same_day": round(r_daily, 3),
                     "gain_pct_per_mmday": round(float(b), 2),
                     "intercept_pct": round(float(a), 1),
                     "r_with_enso": round(r2, 3),
                     "enso_coef_pct_per_roni": round(float(beta[2]), 1),
                     "gain2_pct_per_mmday": round(float(beta[1]), 2),
                     "intercept2_pct": round(float(beta[0]), 1),
                     "tau_slow_days": TAU_SLOW,
                     "r_v2": round(r3, 3),
                     "intercept3_pct": round(float(beta3[0]), 1),
                     "gain3_pct_per_mmday": round(float(beta3[1]), 2),
                     "enso3_coef_pct_per_roni": round(float(beta3[2]), 1),
                     "slow3_pct_per_mmday": round(float(beta3[3]), 2),
                     "n": int(m.sum())}
        ax.plot(t, y, color="#1f4e8c", lw=1.3, label="inflow, % of norm (5-d mean)")
        ax.plot(t, fit, color="#c62828", lw=1.2, alpha=0.9,
                label=f"fitted from rain (τ={tau} d, lag {lag} d)")
        ax.plot(t, fit2, color="#e08214", lw=1.1, alpha=0.9, ls="--",
                label=f"rain + ENSO term (r={r2:.2f})")
        ax.plot(t, fit3, color="#2e7d32", lw=1.0, alpha=0.85, ls=":",
                label=f"+ slow soil kernel (r={r3:.2f})")
        ax.axhline(100, color="0.6", lw=0.7, ls="--")
        ax.set_title(f"{r} — r: {r_daily:.2f} same-day → {rr:.2f} kernel → {r2:.2f} +ENSO "
                     f"→ {r3:.2f} +slow (c={beta[2]:+.0f}%/RONI)",
                     fontsize=10.5, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        ax.set_ylabel("% of norm", fontsize=8)
        if r == ORDER[0]:
            ax.legend(fontsize=7.5, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.suptitle("Rain → inflow, per basin: IMERG basin-rain anomaly through an "
                 "exponential memory kernel vs fleet-corrected inflow % of norm\n"
                 "kernel memory (τ) and lag fitted per basin · same-day correlation "
                 "shown for comparison — the memory is most of the skill",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window": f"{common[0]}..{common[-1]}",
        "note": ("RAIN IS GAUGE-CORRECTED IMERG (x F field) as of v2. "
                 "v2 model: y = intercept3 + gain3*EMA_tau + enso3*RONI + "
                 "slow3*EMA_90, all kernels lagged; y is fleet-corrected "
                 "inflow %% of norm, 5-day trailing mean. NWP forecasts must "
                 "be verified/bias-mapped against CORRECTED IMERG."),
        "rain_space": "gauge_corrected",
        "params": params,
    }, indent=1))
    print(json.dumps(params, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
