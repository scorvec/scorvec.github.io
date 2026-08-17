#!/usr/bin/env python3
"""Brazil rain->ENA kernel models per SIN basin.

Rain: the existing IMERG daily cache already covers Brazil (extent
277-328E, 37S-14N) — basin means over the grouped SIN geometry, with a
harmonic climatology fitted per basin from the cache window itself.
No gauge correction yet (OPEN; NWP bias factors absorb mean bias).

Target: ONS ENA as % of MLT (norms built in). Model per basin:

  y = c0 + c1*EMA_tau(rain anom, lag) + c2*EMA_180(rain anom)
      + c3*EAR_anom(-1d)

tau scanned to 120 d — continental basins carry long memory. EAR
anomaly (storage % vs its doy norm) is the measured-state regressor,
as in Colombia v3.

Outputs:
  ~/brazil_hydro/raw/imerg_basin_daily.json   (rain truth cache)
  ~/brazil_hydro/out/brazil_models.json       + copy in site data/
  brazil_hydro/models.webp                    (12 majors, obs vs fit + fan)

    python scripts/sst/brazil_model.py
"""
from __future__ import annotations

import gzip
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
import imerg_precip as IP                                   # noqa: E402
from rain_inflow_model import ema, trail                    # noqa: E402
from matplotlib.path import Path as MplPath                 # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "brazil_hydro"
BASINS_GJ = PRIV / "out" / "brazil_basins.geojson"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
ENA_CACHE = PRIV / "raw" / "ena_bacia_daily.json.gz"
EAR_CACHE = PRIV / "raw" / "ear_bacia_daily.json.gz"
OUT_JSON = PRIV / "out" / "brazil_models.json"
SITE_JSON = REPO / "brazil_hydro" / "data" / "brazil_models.json"
OUT_PNG = REPO / "brazil_hydro" / "models.webp"
FAN_JSON = REPO / "brazil_hydro" / "data" / "ena_forecast.json"

MAJORS = ["GRANDE", "PARANAIBA", "TIETE", "PARANAPANEMA", "PARANA", "IGUACU",
          "URUGUAI", "JACUI", "SAO FRANCISCO", "TOCANTINS", "AMAZONAS",
          "PARAIBA DO SUL"]
TAUS = [4, 7, 12, 20, 30, 45, 60, 90, 120]
LAGS = range(0, 8)
TAU_SLOW = 180
YSMOOTH = 3


def basin_weights(lons, lats, basins=None):
    gj = json.loads(BASINS_GJ.read_text())
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    W = {}
    for ft in gj["features"]:
        nm = ft["properties"]["basin"]
        if basins is not None and nm not in basins:
            continue
        inside = np.zeros(LO.shape, bool)
        for poly in ft["geometry"]["coordinates"]:
            inside |= MplPath(np.array(poly[0])).contains_points(pts)\
                .reshape(LO.shape)
        w = np.where(inside, np.cos(np.deg2rad(LA)), 0.0)
        if w.sum() > 0:
            W[nm] = w / w.sum()
    return W


def harm_clim(x: np.ndarray, doy: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    th = 2 * np.pi * doy / 365.0
    X = np.column_stack([np.ones_like(th), np.sin(th), np.cos(th),
                         np.sin(2 * th), np.cos(2 * th)])
    beta, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    return X @ beta


CORR_NPZ = Path.home() / "brazil_hydro" / "raw" / "imerg_gauge_corr.npz"


def rain_series() -> dict:
    """Basin-mean daily GAUGE-CORRECTED rain over the whole IMERG cache.
    Cache stamps the correction-field mtime and rebuilds when it changes."""
    cache = json.loads(TRUTH.read_text()) if TRUTH.exists() else {"dates": []}
    fmt = CORR_NPZ.stat().st_mtime if CORR_NPZ.exists() else 0.0
    if cache.get("corr_mtime") != fmt:
        cache = {"dates": [], "corr_mtime": fmt}
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    days = [f.stem for f in files]
    new = [d for d in days if d not in set(cache["dates"])]
    if new:
        ml, mt = IP._grid_axes()
        lons = np.sort(IP._LON[ml])                        # -180..180
        lats = np.sort(IP._LAT[mt])
        W = basin_weights(lons, lats, set(MAJORS))
        F = np.ones((len(lats), len(lons)))
        if CORR_NPZ.exists():
            z = np.load(CORR_NPZ)
            if z["F"].shape == F.shape:
                F = z["F"]
        for b in W:
            cache.setdefault(b, [])
        for d in new:
            g = np.load(IP.DAILY_CACHE / f"{d}.npy") * F
            for b, w in W.items():
                cache[b].append(round(float((g * w).sum()), 3))
        cache["dates"] = cache["dates"] + new
        TRUTH.parent.mkdir(parents=True, exist_ok=True)
        TRUTH.write_text(json.dumps(cache, separators=(",", ":")))
        print(f"rain truth: +{len(new)} days -> {len(cache['dates'])}",
              flush=True)
    return cache


def main() -> int:
    tc = rain_series()
    rdates = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in tc["dates"]],
                      dtype="datetime64[D]")
    doy = np.array([min(d.item().timetuple().tm_yday, 365) for d in rdates])

    with gzip.open(ENA_CACHE, "rt") as f:
        ena = json.load(f)
    with gzip.open(EAR_CACHE, "rt") as f:
        ear = json.load(f)

    fan = None
    if FAN_JSON.exists():
        f_ = json.loads(FAN_JSON.read_text())
        age = (np.datetime64(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
               - np.datetime64(f_["dates"][0])).astype(int)
        if age <= 2:
            fan = f_

    params = {}
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.0), sharex=True)
    for ax, b in zip(axes.flat, MAJORS):
        if b not in tc:
            ax.set_axis_off()
            continue
        rain = np.array(tc[b], float)
        clim = harm_clim(rain, doy)
        x_an = rain - clim

        edays = sorted(ena.get(b, {}))
        edates = np.array(edays, dtype="datetime64[D]")
        y_full = np.array([ena[b][d][1] for d in edays], float)   # % of MLT
        common, ri, ei = np.intersect1d(rdates, edates, return_indices=True)
        y = trail(y_full, YSMOOTH)[ei]
        xa = x_an[ri]

        # EAR anomaly vs doy norm, lagged 1 d
        sdays = sorted(ear.get(b, {}))
        sv = np.array([ear[b][d] for d in sdays], float)
        sdoy = np.array([min(np.datetime64(d).item().timetuple().tm_yday, 365)
                         for d in sdays])
        snorm = np.full(365, np.nan)
        for dd in range(1, 366):
            dist = np.minimum(np.abs(sdoy - dd), 365 - np.abs(sdoy - dd))
            m = dist <= 10
            if m.sum() > 40:
                snorm[dd - 1] = np.median(sv[m])
        sanom_full = sv - snorm[sdoy - 1]
        smap = dict(zip(sdays, sanom_full))
        sa = np.array([smap.get(str(d), np.nan) for d in common])
        sa = np.roll(sa, 1)
        sa[0] = np.nan
        for i in range(1, len(sa)):
            if not np.isfinite(sa[i]):
                sa[i] = sa[i - 1]

        ok0 = np.isfinite(y)
        best = None
        for tau in TAUS:
            k = ema(np.where(np.isfinite(xa), xa, 0), tau)
            for lag in LAGS:
                xl = np.roll(k, lag)
                xl[:lag] = np.nan
                m = ok0 & np.isfinite(xl)
                if m.sum() < 200:
                    continue
                rr = float(np.corrcoef(xl[m], y[m])[0, 1])
                if best is None or rr > best[0]:
                    best = (rr, tau, lag, xl)
        if best is None:
            ax.set_axis_off()
            continue
        r_k, tau, lag, xl = best
        ks = np.roll(ema(np.where(np.isfinite(xa), xa, 0), TAU_SLOW), lag)
        ks[:lag] = np.nan
        m4 = ok0 & np.isfinite(xl) & np.isfinite(ks) & np.isfinite(sa)
        X = np.column_stack([np.ones(m4.sum()), xl[m4], ks[m4], sa[m4]])
        beta, *_ = np.linalg.lstsq(X, y[m4], rcond=None)
        fit = beta[0] + beta[1] * xl + beta[2] * ks + beta[3] * sa
        r_v = float(np.corrcoef(fit[m4], y[m4])[0, 1])

        kf_full = ema(np.where(np.isfinite(xa), xa, 0), tau)
        ks_full = ema(np.where(np.isfinite(xa), xa, 0), TAU_SLOW)
        obs_now = float(y[ok0][-1])
        s_now = float(sa[np.isfinite(sa)][-1])
        fit_now = (beta[0] + beta[1] * kf_full[-1 - lag]
                   + beta[2] * ks_full[-1 - lag] + beta[3] * s_now)
        # harmonic clim by doy for the forecast engine
        clim365 = np.full(365, np.nan)
        for dd in range(1, 366):
            mm = doy == dd
            if mm.any():
                clim365[dd - 1] = float(np.nanmean(clim[mm]))
        bad = ~np.isfinite(clim365)
        clim365[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0],
                                 clim365[~bad])
        params[b] = {
            "tau_days": tau, "lag_days": lag,
            "r_kernel": round(r_k, 3), "r_full": round(r_v, 3),
            "coefs": [round(float(v), 4) for v in beta],
            "tau_slow": TAU_SLOW,
            "k_now": round(float(kf_full[-1]), 3),
            "ks_now": round(float(ks_full[-1]), 3),
            "ear_anom_now": round(s_now, 2),
            "obs_now_pct": round(obs_now, 1),
            "fit_now_pct": round(float(fit_now), 1),
            "clim365_mmday": np.round(clim365, 2).tolist(),
            "last_rain_day": str(rdates[-1]), "n": int(m4.sum())}

        t = common.astype("datetime64[s]").astype(datetime)
        ax.plot(t, y, color="#1f4e8c", lw=1.0, label="ENA, % of MLT")
        ax.plot(t, fit, color="#c62828", lw=1.0, alpha=0.9,
                label="fitted from basin rain + EAR state")
        if fan is not None and b in fan.get("basins", {}):
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            q = fan["basins"][b]
            ax.fill_between(fd, q["p10"], q["p90"], color="#e08214",
                            alpha=0.22, lw=0)
            ax.fill_between(fd, q["p25"], q["p75"], color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(fd, q["p50"], color="#b35806", lw=1.3, ls="--")
        ax.axhline(100, color="0.6", lw=0.6, ls="--")
        ax.set_title(f"{b} — r {r_k:.2f} kernel → {r_v:.2f} full · τ={tau}d",
                     fontsize=9.5, fontweight="bold", loc="left")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("% of MLT", fontsize=7.5)
        if b == MAJORS[0]:
            ax.legend(fontsize=6.8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.suptitle("Brazil rain → ENA models — IMERG basin rain through memory "
                 "kernels + EAR storage state, per SIN basin\n"
                 "target: ONS ENA as % of MLT · fits on the 2-year satellite "
                 "window · orange fan: AIFS+IFS ensemble forecast",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}", flush=True)

    payload = json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "note": ("y = c0 + c1*EMA_tau(rain anom, lag) + c2*EMA_180 + "
                 "c3*EAR_anom(-1d); y = ENA %% of MLT (3-d trail); rain = "
                 "raw IMERG basin mean (no gauge correction yet)"),
        "params": params}, indent=1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(payload)
    SITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SITE_JSON.write_text(payload)
    for b, p in params.items():
        print(f"  {b:16} r {p['r_kernel']:.2f}->{p['r_full']:.2f} "
              f"tau={p['tau_days']:3d} lag={p['lag_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
