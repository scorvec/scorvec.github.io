#!/usr/bin/env python3
"""Per-dam rain->inflow models: each major river driven ONLY by rain that
falls in its own upstream catchment (xm_river_catchments.geojson — IDEAM
subzona unions upstream of each dam; nested drainage kept, so Ituango's
polygon contains its upstream Caldas catchments by design).

For each of the DAMS: gauge-corrected IMERG masked to the catchment ->
anomaly vs the harmonic clim (same F field on both) -> v3-style fit
against the river's own inflow % of norm (successor names merged):

  y = a + b*EMA_tau(x, lag) + c*RONI + d*EMA_90(x) + e*S_anom(region)

Rivers whose kernel-only r < R_REGULATED are flagged REGULATED (operator
decisions, not rain, set their flow — Bogota N.R., Bata/Chivor,...);
their fits are still reported but the page says not to trust rain there.

Outputs:
  colombia_hydro/dam_models.webp        (12 panels, obs vs fit + fan)
  colombia_hydro/data/dam_models.json   (params + kernel state for the engine)

    python scripts/sst/dam_models.py
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
import imerg_precip as IP                                    # noqa: E402
from hydro_region_rain import gauge_correction               # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim       # noqa: E402
from rain_inflow_model import ema, trail, TAUS, LAGS, YSMOOTH, TAU_SLOW  # noqa: E402
from xm_inflow_history import SUCCESSOR                      # noqa: E402
from xm_storage import pct_anomaly_series                    # noqa: E402
from matplotlib.path import Path as MplPath                  # noqa: E402

REPO = HERE.parent.parent
CATCH_GJ = REPO / "colombia_hydro" / "data" / "xm_river_catchments.geojson"
APOR_CACHE = Path.home() / "colombia_hydro" / "raw" / "aporener_daily.json.gz"
ENSO_JSON = REPO / "assets" / "sst" / "data" / "enso_daily.json"
FAN_JSON = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
OUT_PNG = REPO / "colombia_hydro" / "dam_models.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "dam_models.json"

DAMS = ["ITUANGO", "SOGAMOSO", "GUAVIO", "BATA", "EL QUIMBO", "BETANIA CP",
        "SINU URRA", "CAUCA SALVAJINA", "GRANDE", "ESCUELA DE MINAS",
        "A. SAN LORENZO", "BOGOTA N.R."]
R_REGULATED = 0.30
WIN = 10


def catchment_weights(lons: np.ndarray, lats: np.ndarray,
                      rivers=None) -> dict[str, np.ndarray]:
    """cos-lat masks for each river's upstream catchment on any grid."""
    gj = json.loads(CATCH_GJ.read_text())
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    W = {}
    for ft in gj["features"]:
        nm = ft["properties"]["river"]
        if rivers is not None and nm not in rivers:
            continue
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        inside = np.zeros(LO.shape, bool)
        for poly in polys:
            inside |= MplPath(np.array(poly[0])).contains_points(pts).reshape(LO.shape)
        w = np.where(inside, np.cos(np.deg2rad(LA)), 0.0)
        if w.sum() == 0:                       # sub-cell catchment -> nearest cell
            arr = np.vstack([np.array(p[0]) for p in polys])
            i = int(np.argmin(np.abs(lats - arr[:, 1].mean())))
            j = int(np.argmin(np.abs(lons - arr[:, 0].mean())))
            w = np.zeros(LO.shape)
            w[i, j] = 1.0
        W[nm] = w / w.sum()
    return W


def river_series(days: list[str]) -> dict[str, np.ndarray]:
    """Per-dam daily inflow kWh with successor names merged."""
    with gzip.open(APOR_CACHE, "rt") as f:
        apor = json.load(f)
    pred = {v: k for k, v in SUCCESSOR.items()}      # current -> historical name
    out = {r: np.full(len(days), np.nan) for r in DAMS}
    for i, d in enumerate(days):
        dd = apor.get(d, {})
        for r in DAMS:
            v = dd.get(r)
            if v is None and r in pred:
                v = dd.get(pred[r])
            if v is not None:
                out[r][i] = v
    return out


def main() -> int:
    import xarray as xr
    region_of = {ft["properties"]["river"]: ft["properties"]["region"]
                 for ft in json.loads(CATCH_GJ.read_text())["features"]}

    # ── catchment-masked corrected rain over the whole IMERG cache ─────────
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    rdates = np.array([f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}" for f in files],
                      dtype="datetime64[D]")
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml])
    lats = np.sort(IP._LAT[mt])
    W = catchment_weights(lons, lats, set(DAMS))
    F = gauge_correction(lons, lats)
    clim = xr.open_dataset(CLIM_NC)["coef"].values
    doys = np.array([min(np.datetime64(d).item().timetuple().tm_yday, 365)
                     for d in rdates])
    rain = {r: np.full(len(files), np.nan) for r in DAMS}
    rain_clim = {r: np.full(len(files), np.nan) for r in DAMS}
    clim365 = {r: np.full(365, np.nan) for r in DAMS}
    for dy in range(1, 366):
        c = eval_clim(clim, dy) * F
        for r in DAMS:
            clim365[r][dy - 1] = float((c * W[r]).sum())
    for i, f in enumerate(files):
        g = np.load(f) * F
        for r in DAMS:
            rain[r][i] = float((g * W[r]).sum())
            rain_clim[r][i] = clim365[r][doys[i] - 1]
    print(f"catchment rain: {len(files)} days x {len(DAMS)} dams", flush=True)

    # ── river inflow % of own doy norm (full record, successors merged) ─────
    with gzip.open(APOR_CACHE, "rt") as f:
        apor = json.load(f)
    adays = sorted(apor)
    aser = river_series(adays)
    adates = np.array(adays, dtype="datetime64[D]")
    adoy = np.array([min(np.datetime64(d).item().timetuple().tm_yday, 365)
                     for d in adates])
    pct = {}
    for r in DAMS:
        v = aser[r]
        norm = np.full(365, np.nan)
        for dy in range(1, 366):
            dist = np.minimum(np.abs(adoy - dy), 365 - np.abs(adoy - dy))
            m = (dist <= WIN) & np.isfinite(v) & (v > 0)
            if m.sum() > 40:
                norm[dy - 1] = np.median(v[m])
        with np.errstate(invalid="ignore"):
            pct[r] = 100.0 * v / norm[adoy - 1]

    # ── shared regressors ───────────────────────────────────────────────────
    common, ri, ai = np.intersect1d(rdates, adates, return_indices=True)
    ed = json.loads(ENSO_JSON.read_text())["daily"]
    edates = np.array(ed["dates"], dtype="datetime64[D]")
    eroni = np.array(ed["roni_d"], float)
    roni = np.full(len(common), np.nan)
    _, ci, ei = np.intersect1d(common, edates, return_indices=True)
    roni[ci] = eroni[ei]
    for i in range(1, len(roni)):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i - 1]
    for i in range(len(roni) - 2, -1, -1):
        if not np.isfinite(roni[i]):
            roni[i] = roni[i + 1]
    roni_now = float(roni[np.isfinite(roni)][-1])
    sdates, sanom = pct_anomaly_series()
    stor = {rg: np.full(len(common), np.nan) for rg in set(region_of.values())}
    _, c2, s2 = np.intersect1d(common, sdates, return_indices=True)
    for rg in stor:
        stor[rg][c2] = sanom[rg][s2]
        stor[rg] = np.roll(stor[rg], 1)
        stor[rg][0] = np.nan

    # ── fits ────────────────────────────────────────────────────────────────
    fan = None
    if FAN_JSON.exists():
        f_ = json.loads(FAN_JSON.read_text())
        if "dams" in f_ and (np.datetime64(datetime.now(timezone.utc)
                             .strftime("%Y-%m-%d"))
                             - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    params = {}
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.0), sharex=True)
    t = common.astype("datetime64[s]").astype(datetime)
    for ax, r in zip(axes.flat, DAMS):
        x_an = (rain[r] - rain_clim[r])[ri]
        y = trail(pct[r], YSMOOTH)[ai]
        y[y == 0] = np.nan
        ok0 = np.isfinite(y)
        best = None
        for tau in TAUS:
            k = ema(np.where(np.isfinite(x_an), x_an, 0), tau)
            for lag in LAGS:
                xl = np.roll(k, lag)
                xl[:lag] = np.nan
                m = ok0 & np.isfinite(xl)
                if m.sum() < 120:
                    continue
                rr = float(np.corrcoef(xl[m], y[m])[0, 1])
                if best is None or rr > best[0]:
                    best = (rr, tau, lag, xl, m)
        if best is None:
            ax.set_title(f"{r} — insufficient data", fontsize=9)
            continue
        r_k, tau, lag, xl, m = best
        ks = np.roll(ema(np.where(np.isfinite(x_an), x_an, 0), TAU_SLOW), lag)
        ks[:lag] = np.nan
        sa = stor[region_of[r]]
        m4 = m & np.isfinite(ks) & np.isfinite(roni) & np.isfinite(sa)
        X = np.column_stack([np.ones(m4.sum()), xl[m4], roni[m4], ks[m4], sa[m4]])
        beta, *_ = np.linalg.lstsq(X, y[m4], rcond=None)
        fit = beta[0] + beta[1] * xl + beta[2] * roni + beta[3] * ks + beta[4] * sa
        r_v3 = float(np.corrcoef(fit[m4], y[m4])[0, 1])
        regulated = bool(r_k < R_REGULATED)

        # kernel state at the last rain day (for the engine's fan propagation)
        kf_full = ema(np.where(np.isfinite(x_an), x_an, 0), tau)
        ks_full = ema(np.where(np.isfinite(x_an), x_an, 0), TAU_SLOW)
        obs_now = y[np.isfinite(y)][-1] if np.isfinite(y).any() else None
        s_now = float(sa[np.isfinite(sa)][-1]) if np.isfinite(sa).any() else 0.0
        fit_now = (beta[0] + beta[1] * kf_full[-1 - lag] + beta[2] * roni_now
                   + beta[3] * ks_full[-1 - lag] + beta[4] * s_now)
        params[r] = {
            "region": region_of[r], "tau_days": tau, "lag_days": lag,
            "r_kernel": round(r_k, 3), "r_v3": round(r_v3, 3),
            "regulated": regulated,
            "coefs": [round(float(b), 4) for b in beta],
            "k_now": round(float(kf_full[-1]), 3),
            "ks_now": round(float(ks_full[-1]), 3),
            "storage_anom_now": round(s_now, 2), "roni_now": round(roni_now, 2),
            "obs_now_pct": round(float(obs_now), 1) if obs_now is not None else None,
            "fit_now_pct": round(float(fit_now), 1),
            "clim365_mmday": np.round(clim365[r], 2).tolist(),
            "last_rain_day": str(rdates[-1]),
            "n": int(m4.sum())}

        ax.plot(t, y, color="#1f4e8c", lw=1.1, label="inflow, % of own norm")
        ax.plot(t, fit, color="#c62828", lw=1.0, alpha=0.9,
                label="fitted from catchment rain")
        if fan is not None and r in fan.get("dams", {}):
            q = fan["dams"][r]
            ft_ = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            ax.fill_between(ft_, q["p10"], q["p90"], color="#e08214",
                            alpha=0.25, lw=0)
            ax.fill_between(ft_, q["p25"], q["p75"], color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(ft_, q["p50"], color="#b35806", lw=1.3, ls="--")
        ax.axhline(100, color="0.6", lw=0.6, ls="--")
        flag = "  [REGULATED]" if regulated else ""
        ax.set_title(f"{r} ({region_of[r]}) — r {r_k:.2f} kernel → {r_v3:.2f} "
                     f"full · τ={tau}d{flag}",
                     fontsize=9, fontweight="bold", loc="left",
                     color="#8a4a00" if regulated else "black")
        ax.tick_params(labelsize=7.5)
        ax.grid(lw=0.25, alpha=0.5)
        ax.set_ylabel("% of norm", fontsize=7.5)
        if r == DAMS[0]:
            ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.suptitle("Per-dam models — each river driven ONLY by rain in its own "
                 "upstream catchment (gauge-corrected IMERG × catchment mask)\n"
                 "fast + slow memory kernels + ENSO + regional storage state · "
                 "REGULATED = operator decisions dominate, rain skill is structurally low",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "note": ("y = c0 + c1*EMA_tau(catchment rain anom, lag) + c2*RONI + "
                 "c3*EMA_90 + c4*region_storage_anom; rain is gauge-corrected "
                 "IMERG masked to the dam's upstream catchment; k_now/ks_now "
                 "are kernel states at last_rain_day for fan propagation"),
        "params": params,
    }, indent=1))
    for r, p in params.items():
        print(f"  {r:18} r {p['r_kernel']:.2f}->{p['r_v3']:.2f} tau={p['tau_days']:2d} "
              f"{'REGULATED' if p['regulated'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
