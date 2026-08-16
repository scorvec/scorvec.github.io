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
from hydro_region_rain import region_weights     # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim  # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
OUT_PNG = REPO / "colombia_hydro" / "rain_inflow_model.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "rain_inflow_model.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
TAUS = [2, 4, 7, 12, 20, 30, 45, 60]
LAGS = range(0, 8)
YSMOOTH = 5                       # trailing-mean days on the inflow side


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
    import xarray as xr
    clim = xr.open_dataset(CLIM_NC)["coef"].values
    doys = np.array([min(int(np.datetime64(d).astype("datetime64[D]").item().timetuple().tm_yday), 365)
                     for d in rdates])
    rain = {r: np.full(len(files), np.nan) for r in ORDER}
    rain_clim = {r: np.full(len(files), np.nan) for r in ORDER}
    for i, f in enumerate(files):
        g = np.load(f)
        c = eval_clim(clim, doys[i])
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
        params[r] = {"tau_days": tau, "lag_days": lag, "r": round(rr, 3),
                     "r_same_day": round(r_daily, 3),
                     "gain_pct_per_mmday": round(float(b), 2),
                     "intercept_pct": round(float(a), 1),
                     "n": int(m.sum())}
        ax.plot(t, y, color="#1f4e8c", lw=1.3, label="inflow, % of norm (5-d mean)")
        ax.plot(t, fit, color="#c62828", lw=1.2, alpha=0.9,
                label=f"fitted from rain (τ={tau} d, lag {lag} d)")
        ax.axhline(100, color="0.6", lw=0.7, ls="--")
        ax.set_title(f"{r} — r = {rr:.2f} (kernel) vs {r_daily:.2f} (same-day rain)",
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
        "note": ("y = intercept + gain * EMA_tau(rain anomaly, lagged); "
                 "y is fleet-corrected inflow % of norm, 5-day trailing mean; "
                 "multiplicative satellite bias cancels in r — apply region "
                 "bias factors when driving with AIFS forecasts"),
        "params": params,
    }, indent=1))
    print(json.dumps(params, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
