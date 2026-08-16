#!/usr/bin/env python3
"""23-year water-balance model: does daily P−E beat P for inflows?

WB2 ERA5 gives daily basin precipitation AND evapotranspiration
(latent-heat-flux-derived) from 1959; XM inflows (fleet-corrected % of
norm) run from 2000. On the 2000–2023 overlap, three kernel models per
basin, identical machinery:

  A. kernel(P′)          — rain anomaly alone (the current model's form)
  B. kernel(P′ − E′)     — daily effective precipitation
  C. kernel(P′) + c·E′k  — rain kernel plus a separate ET-kernel term

Skill is OUT-OF-SAMPLE: leave-one-year-out cross-validation, r computed
on the concatenated held-out predictions — 23 ENSO-spanning years make
the ET terms identifiable where the 2-year window could not.

Outputs:
  colombia_hydro/water_balance_23yr.webp
  colombia_hydro/data/water_balance_23yr.json
  cache: ~/colombia_hydro/raw/wb2_basin_pe_daily.json

    python scripts/sst/long_water_balance.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rain_inflow_model import ema, trail, TAUS, LAGS, YSMOOTH  # noqa: E402
from matplotlib.path import Path as MplPath                    # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
ET_RAW = Path.home() / "colombia_hydro" / "raw" / "era5_basin_et_daily.json"
PE_RAW = Path.home() / "colombia_hydro" / "raw" / "wb2_basin_pe_daily.json"
OUT_PNG = REPO / "colombia_hydro" / "water_balance_23yr.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "water_balance_23yr.json"
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


def basin_weights(lons, lats):
    gj = json.loads(REGIONS_GJ.read_text())
    rings = {}
    for ft in gj["features"]:
        nm = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        rings.setdefault(nm, []).extend(np.array(p[0]) for p in polys)
    LO, LA = np.meshgrid(lons, lats)
    W = {}
    for r in ORDER:
        inside = np.zeros(LO.shape, bool)
        for rr in rings[r]:
            inside |= MplPath(rr).contains_points(
                np.column_stack([LO.ravel(), LA.ravel()])).reshape(LO.shape)
        w = np.where(inside, np.cos(np.deg2rad(LA)), 0.0)
        if w.sum() == 0:
            arr = np.vstack(rings[r])
            i = int(np.argmin(np.abs(lats - arr[:, 1].mean())))
            j = int(np.argmin(np.abs(lons - arr[:, 0].mean())))
            w = np.zeros(LO.shape)
            w[i, j] = 1.0
        W[r] = w
    return W


def build_precip():
    """WB2 daily basin precipitation 2000-2023 (mm/day), cached with ET dates."""
    if PE_RAW.exists():
        return json.loads(PE_RAW.read_text())
    import xarray as xr
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    p6 = ds["total_precipitation_6hr"].sel(
        longitude=slice(279, 291), latitude=slice(-1, 11),
        time=slice("2000-01-01", "2023-01-09"))
    print("streaming P box", dict(p6.sizes), flush=True)
    pd_ = (p6.resample(time="1D").sum() * 1000.0).compute()   # mm/day
    lons = pd_.longitude.values - 360.0
    lats = pd_.latitude.values
    W = basin_weights(lons, lats)
    v = pd_.transpose("time", "latitude", "longitude").values
    out = {"dates": [f"{pd.Timestamp(t):%Y-%m-%d}" for t in pd_.time.values]}
    for r in ORDER:
        w = W[r]
        out[r] = np.round((v * w).sum(axis=(1, 2)) / w.sum(), 3).tolist()
        print(f"  P {r}: mean {np.mean(out[r]):.2f} mm/day", flush=True)
    PE_RAW.write_text(json.dumps(out, separators=(",", ":")))
    return out


def harm_anom(x, doy):
    ok = np.isfinite(x)
    th = 2 * np.pi * doy / 365.0
    X = np.column_stack([np.ones_like(th), np.sin(th), np.cos(th),
                         np.sin(2 * th), np.cos(2 * th)])
    beta, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    return x - X @ beta


def cv_fit(Xcols, y, years):
    """Leave-one-year-out CV; returns out-of-sample r."""
    yhat = np.full(len(y), np.nan)
    for yr in np.unique(years):
        tr = (years != yr) & np.isfinite(y) & np.all(np.isfinite(Xcols), axis=1)
        te = (years == yr) & np.isfinite(y) & np.all(np.isfinite(Xcols), axis=1)
        if tr.sum() < 500 or te.sum() < 100:
            continue
        A = np.column_stack([np.ones(tr.sum()), Xcols[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        yhat[te] = np.column_stack([np.ones(te.sum()), Xcols[te]]) @ beta
    m = np.isfinite(yhat) & np.isfinite(y)
    return float(np.corrcoef(yhat[m], y[m])[0, 1]), yhat


def main() -> int:
    P = build_precip()
    E = json.loads(ET_RAW.read_text())
    inf = json.loads(INFLOW_JSON.read_text())["full_pct_of_norm"]

    pdates = np.array(P["dates"], dtype="datetime64[D]")
    edates = np.array(E["dates"], dtype="datetime64[D]")
    idates = np.array(inf["dates"], dtype="datetime64[D]")
    common = np.intersect1d(np.intersect1d(pdates, edates), idates)
    pi = np.searchsorted(pdates, common)
    ei = np.searchsorted(edates, common)
    ii = np.searchsorted(idates, common)
    doy = np.minimum(np.array([np.datetime64(d, "D").item().timetuple().tm_yday
                               for d in common]), 365)
    years = common.astype("datetime64[Y]").astype(int) + 1970
    print(f"overlap {common[0]}..{common[-1]} ({len(common)} days)", flush=True)

    results = {}
    for r in ORDER:
        y = trail(np.array(inf[r], dtype=float), YSMOOTH)[ii]
        y[y == 0] = np.nan                      # zero-filled = missing
        p_a = harm_anom(np.array(P[r])[pi], doy)
        e_a = harm_anom(np.array(E[r])[ei], doy)
        # choose tau on model A in-sample (shared across models for fairness)
        best = (None, -9)
        for tau in TAUS:
            k = ema(p_a, tau)
            m = np.isfinite(y)
            rr = float(np.corrcoef(k[m], y[m])[0, 1])
            if rr > best[1]:
                best = (tau, rr)
        tau = best[0]
        kP = ema(p_a, tau)
        kPE = ema(p_a - e_a, tau)
        kE = ema(e_a, tau)
        rA, _ = cv_fit(kP[:, None], y, years)
        rB, _ = cv_fit(kPE[:, None], y, years)
        rC, _ = cv_fit(np.column_stack([kP, kE]), y, years)
        results[r] = {"tau": tau, "r_P": round(rA, 3), "r_PminusE": round(rB, 3),
                      "r_P_plus_Eterm": round(rC, 3)}
        print(r, results[r], flush=True)

    fig, ax = plt.subplots(figsize=(11.5, 6))
    xpos = np.arange(len(ORDER))
    wd = 0.26
    for k, (key, off, col, lab) in enumerate([
            ("r_P", -wd, "#1f4e8c", "P kernel"),
            ("r_PminusE", 0, "#c62828", "(P − E) kernel"),
            ("r_P_plus_Eterm", wd, "#4d8f4d", "P kernel + E term")]):
        vals = [results[r][key] for r in ORDER]
        ax.bar(xpos + off, vals, width=wd, color=col, label=lab)
        for x0, v in zip(xpos + off, vals):
            ax.text(x0, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{r}\n(τ={results[r]['tau']} d)" for r in ORDER], fontsize=9)
    ax.set_ylabel("out-of-sample r (leave-one-year-out, 2000–2023)", fontsize=10)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title("Does daily evapotranspiration improve the inflow model?\n"
                 "23-year ERA5 water balance, cross-validated by year",
                 fontsize=12.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=125)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window": f"{common[0]}..{common[-1]}",
        "method": "LOYO-CV; tau chosen on model A in-sample, shared across models",
        "results": results,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
