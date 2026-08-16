#!/usr/bin/env python3
"""Head-to-head: do rain GAUGES beat IMERG at predicting inflows?

Three competitors per basin, identical treatment: (1) IMERG basin-mean
rain, (2) in-basin gauge-mean rain (stations inside the region polygon,
>=5 reporting required per day), (3) the standardized blend. Each series
is converted to anomalies against its OWN 2-year harmonic seasonal fit
(both sides get the same climatology treatment — no home-field
advantage), passed through the same exponential-memory kernel scan
(tau, lag), and correlated with the fleet-corrected inflow % of norm.
The multiple-regression r of both predictors together is the ceiling.

Outputs:
  colombia_hydro/gauge_vs_imerg.webp
  colombia_hydro/data/gauge_vs_imerg.json

    python scripts/sst/gauge_vs_imerg_inflow.py
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
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                        # noqa: E402
from hydro_region_rain import region_weights     # noqa: E402
from rain_inflow_model import ema, trail, TAUS, LAGS, YSMOOTH  # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
GAUGES = Path.home() / "colombia_hydro" / "raw" / "gauges"
OUT_PNG = REPO / "colombia_hydro" / "gauge_vs_imerg.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "gauge_vs_imerg.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
RCOL = {"ANTIOQUIA": "#1b7837", "CALDAS": "#762a83", "CARIBE": "#2166ac",
        "CENTRO": "#b2182b", "ORIENTE": "#e08214", "VALLE": "#35978f"}
MIN_ST = 5


def harmonic_anom(x: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Anomaly vs a 2-harmonic seasonal fit of the series itself."""
    ok = np.isfinite(x)
    th = 2 * np.pi * doy / 365.0
    X = np.column_stack([np.ones_like(th), np.sin(th), np.cos(th),
                         np.sin(2 * th), np.cos(2 * th)])
    beta, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    return x - X @ beta


def region_polys():
    gj = json.loads(REGIONS_GJ.read_text())
    out = {}
    for ft in gj["features"]:
        name = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out[name] = [MplPath(np.array(r[0])) for r in polys]
    return out


def kernel_best(x_anom: np.ndarray, y: np.ndarray):
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
                best = (rr, tau, lag, xl)
    return best


def main() -> int:
    # ── IMERG basin series ──────────────────────────────────────────────────
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    rdates = np.array([f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}" for f in files],
                      dtype="datetime64[D]")
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml])
    lats = np.sort(IP._LAT[mt])
    W = region_weights(REGIONS_GJ, lons, lats)
    imerg = {r: np.full(len(files), np.nan) for r in ORDER}
    for i, f in enumerate(files):
        g = np.load(f)
        for r in ORDER:
            w = W[r]
            imerg[r][i] = float((g * w).sum() / w.sum())

    # ── gauge basin series ──────────────────────────────────────────────────
    polys = region_polys()
    st_region: dict[str, str] = {}
    gs = {r: np.full(len(files), np.nan) for r in ORDER}
    gn = {r: np.zeros(len(files)) for r in ORDER}
    for i, d in enumerate(rdates):
        f = GAUGES / f"{str(d).replace('-', '')}.json"
        if not f.exists():
            continue
        day = json.loads(f.read_text())
        acc = {r: [] for r in ORDER}
        for code, g in day.items():
            reg = st_region.get(code, "?")
            if reg == "?":
                reg = None
                for r, ps in polys.items():
                    if r in ORDER and any(p.contains_point((g["lo"], g["la"])) for p in ps):
                        reg = r
                        break
                st_region[code] = reg
            if reg:
                acc[reg].append(g["mm"])
            st_region[code] = reg
        for r in ORDER:
            if len(acc[r]) >= MIN_ST:
                gs[r][i] = float(np.mean(acc[r]))
                gn[r][i] = len(acc[r])
    print("in-basin stations (mean/day):",
          {r: round(float(np.nanmean(np.where(gn[r] > 0, gn[r], np.nan))), 1)
           for r in ORDER}, flush=True)

    # ── inflow target ───────────────────────────────────────────────────────
    inf = json.loads(INFLOW_JSON.read_text())
    idates = np.array(inf["recent"]["dates"], dtype="datetime64[D]")
    common, ri, ii = np.intersect1d(rdates, idates, return_indices=True)
    doy = np.minimum(np.array([np.datetime64(d, "D").item().timetuple().tm_yday
                               for d in common]), 365)

    results = {}
    for r in ORDER:
        y = trail(np.array(inf["recent"]["pct_of_norm"][r], dtype=float), YSMOOTH)[ii]
        sr = {}
        xi = harmonic_anom(imerg[r][ri], doy)
        xg_raw = gs[r][ri]
        xg = harmonic_anom(xg_raw, doy)
        for name, x in [("imerg", xi), ("gauges", xg)]:
            b = kernel_best(np.where(np.isfinite(x), x, np.nan), y)
            sr[name] = {"r": round(b[0], 3), "tau": b[1], "lag": b[2]} if b else None
        # blend: standardized average where both exist
        zi = (xi - np.nanmean(xi)) / np.nanstd(xi)
        zg = (xg - np.nanmean(xg)) / np.nanstd(xg)
        blend = np.nanmean(np.vstack([zi, zg]), axis=0)
        blend[~(np.isfinite(zi) | np.isfinite(zg))] = np.nan
        b = kernel_best(blend, y)
        sr["blend"] = {"r": round(b[0], 3), "tau": b[1], "lag": b[2]} if b else None
        sr["n_gauge_days"] = int(np.isfinite(xg).sum())
        sr["mean_stations"] = round(float(np.nanmean(np.where(gn[r][ri] > 0, gn[r][ri], np.nan))), 1)
        results[r] = sr
        print(r, sr, flush=True)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    xpos = np.arange(len(ORDER))
    wdt = 0.26
    for k, (name, off, col) in enumerate([("imerg", -wdt, "#1f4e8c"),
                                          ("gauges", 0, "#c62828"),
                                          ("blend", wdt, "#4d8f4d")]):
        vals = [results[r][name]["r"] if results[r][name] else 0 for r in ORDER]
        ax.bar(xpos + off, vals, width=wdt, color=col,
               label={"imerg": "IMERG basin mean", "gauges": "in-basin gauges",
                      "blend": "blend (z-avg)"}[name])
        for x0, v in zip(xpos + off, vals):
            ax.text(x0, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{r}\n(~{results[r]['mean_stations']:.0f} gauges)"
                        for r in ORDER], fontsize=9)
    ax.set_ylabel("kernel-model correlation with inflow % of norm", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title("Who predicts inflows better — satellite, gauges, or both?\n"
                 "identical anomaly + memory-kernel treatment per source",
                 fontsize=12.5, fontweight="bold", loc="left")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=125)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results": results,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
