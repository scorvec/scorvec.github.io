#!/usr/bin/env python3
"""Validate IMERG basin rainfall against XM's observed inflows (aportes).

Fetches daily per-river inflow energy (AporEner, kWh) from XM's public API,
aggregates to the six hydrological regions via the same ListadoRios mapping
that defines the polygons, and correlates each region's inflow series with
its IMERG area-mean rainfall at lags 0–15 days. Basins integrate rain, so
inflow is compared against trailing-K-day rainfall means (K per region ≈
concentration time) as well as raw daily rain.

Outputs:
    ~/colombia_hydro/out/validation_rain_vs_inflow.png
    ~/colombia_hydro/out/validation_summary.json

    python scripts/sst/validate_region_rain.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAIN_JSON = REPO / "assets" / "sst" / "data" / "colombia_region_rain.json"
RIVERS_JSON = Path.home() / "colombia_hydro" / "raw" / "xm_listado_rios.json"
OUT = Path.home() / "colombia_hydro" / "out"
API = "https://servapibi.xm.com.co/daily"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
COLORS = {"ANTIOQUIA": "#68B79F", "CALDAS": "#4F5BE3", "CARIBE": "#F0A169",
          "CENTRO": "#F5D76E", "ORIENTE": "#C0608D", "VALLE": "#43128F"}


def river_region_map() -> dict[str, str]:
    d = json.load(open(RIVERS_JSON))
    out = {}
    for it in d["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("Status") == "ACTIVO" and v["HydroRegion"] in ORDER:
                out[v["Name"].strip().upper()] = v["HydroRegion"]
    return out


def fetch_aportes(d0: datetime, d1: datetime) -> dict[str, dict[str, float]]:
    """{date_iso: {river: kWh}} over [d0, d1] (chunked ≤30-day requests)."""
    out: dict[str, dict[str, float]] = {}
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        r = requests.post(API, json={"MetricId": "AporEner",
                                     "StartDate": f"{cur:%Y-%m-%d}",
                                     "EndDate": f"{end:%Y-%m-%d}",
                                     "Entity": "Rio"}, timeout=90)
        r.raise_for_status()
        for it in r.json().get("Items", []):
            day = it["Date"]
            for e in it.get("DailyEntities", []):
                out.setdefault(day, {})[e["Name"].strip().upper()] = float(e["Value"])
        cur = end + timedelta(days=1)
    return out


def main() -> int:
    rain = json.load(open(RAIN_JSON))["regions"]
    r2r = river_region_map()
    dates = rain[ORDER[0]]["dates"]
    d0 = datetime.strptime(dates[0], "%Y-%m-%d")
    d1 = datetime.strptime(dates[-1], "%Y-%m-%d")
    print(f"fetching XM aportes {d0:%Y-%m-%d} … {d1:%Y-%m-%d}", flush=True)
    apor = fetch_aportes(d0, d1)

    # region × day inflow (GWh); missing rivers on a day simply don't add
    inflow = {reg: np.full(len(dates), np.nan) for reg in ORDER}
    for i, ds in enumerate(dates):
        day = apor.get(ds, {})
        acc: dict[str, float] = {}
        for river, kwh in day.items():
            reg = r2r.get(river)
            if reg:
                acc[reg] = acc.get(reg, 0.0) + kwh
        for reg, v in acc.items():
            inflow[reg][i] = v / 1e6                        # kWh → GWh

    summary, scatter = {}, {}
    fig, axs = plt.subplots(3, 2, figsize=(12.2, 9.6), sharex=True)
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    for ax, reg in zip(axs.ravel(), ORDER):
        rr = np.array(rain[reg]["mm"], float)
        ii = inflow[reg]
        ok = np.isfinite(ii)
        # contiguous segments: the record may hold disjoint chunks (partial
        # backfill) — rolling sums must never straddle a gap
        dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        seg_id = np.zeros(len(dts), int)
        for i in range(1, len(dts)):
            seg_id[i] = seg_id[i - 1] + ((dts[i] - dts[i - 1]).days > 1)

        def roll3(x):
            """Trailing 3-day sum, NaN across segment joins / incomplete windows."""
            out = np.full(len(x), np.nan)
            for sid in np.unique(seg_id):
                m = seg_id == sid
                xx = x[m]
                sm = np.convolve(np.nan_to_num(xx), np.ones(3), "full")[:m.sum()]
                bad = np.convolve(np.isnan(xx).astype(float), np.ones(3), "full")[:m.sum()]
                sm[(bad > 0)] = np.nan
                sm[:2] = np.nan
                out[m] = sm
            return out

        r3, i3 = roll3(rr), roll3(np.where(ok, ii, np.nan))
        cl3 = roll3(np.array(rain[reg]["clim"], float))

        # inflow seasonal normal: harmonic (mean+annual+semiannual) fit on the
        # available record, so anomalies test rain-inflow coupling WITHOUT the
        # shared seasonal cycle inflating r
        doy = np.array([d.timetuple().tm_yday for d in dts], float)
        w = 2 * np.pi * doy / 365.25
        X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w),
                             np.cos(2 * w), np.sin(2 * w)])
        fin = np.isfinite(i3)
        coef, *_ = np.linalg.lstsq(X[fin], i3[fin], rcond=None)
        ia3 = i3 - X @ coef

        def scan(xs, ys):
            best = (0, 0.0)
            for lag in range(0, 16):
                xr_ = np.roll(xs, lag); xr_[:lag] = np.nan
                m = np.isfinite(xr_) & np.isfinite(ys) & (np.roll(seg_id, lag) == seg_id)
                if m.sum() > 30:
                    cc = float(np.corrcoef(xr_[m], ys[m])[0, 1])
                    if cc > best[1]:
                        best = (lag, cc)
            return best

        variants = {"raw": (r3, i3), "anom": (r3 - cl3, ia3)}
        bT, bB = 0, (0, 0.0)
        for T in range(1, 9):                       # burst: rain above baseline
            e3 = roll3(np.maximum(rr - T, 0.0))
            lg, cc = scan(e3, ia3)
            if cc > bB[1]:
                bT, bB = T, (lg, cc)
        variants[f"burst>{bT}mm"] = (roll3(np.maximum(rr - bT, 0.0)), ia3)
        results = {name: scan(x, y) for name, (x, y) in variants.items()}
        bname = max(results, key=lambda n: results[n][1])
        k, c = results[bname]
        xs, ys = variants[bname]
        sm = np.roll(xs, k); sm[:k] = np.nan
        sm = sm / 3.0
        print(f"  {reg}: " + "  ".join(
            f"{n} r={results[n][1]:.2f}@{results[n][0]}d" for n in results), flush=True)
        scatter[reg] = (sm, ys / 3.0, k, c, bname)
        summary[reg] = dict(best_lag_days=k, corr=round(c, 3), variant=bname,
                            all_variants={n: dict(lag=results[n][0], r=round(results[n][1], 3)) for n in results},
                            n_days=int(ok.sum()),
                            inflow_mean_gwh=round(float(np.nanmean(ii)), 2))

        ax2 = ax.twinx()
        ax.bar(dts, rr, width=1.0, color=COLORS[reg], alpha=0.35, lw=0)
        ax.plot(dts, sm, color=COLORS[reg], lw=1.6,
                label=f"3-d rain, led {k} d")
        ax2.plot(dts, i3 / 3.0, color="k", lw=1.3, label="3-d XM inflow (GWh/d)")
        ax.set_title(f"{reg} — r = {c:.2f} (3-d blocks, rain led {k} d)", fontsize=10,
                     fontweight="bold", loc="left")
        ax.set_ylabel("mm/day", fontsize=8); ax2.set_ylabel("GWh/day", fontsize=8)
        ax.tick_params(labelsize=7.5); ax2.tick_params(labelsize=7.5)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if reg == ORDER[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle("Validation — IMERG basin rainfall vs XM observed inflows (AporEner)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "rain: GPM IMERG area-weighted over the region polygons · "
             "inflow: XM daily aportes energía summed per region via ListadoRios · "
             "r = corr(3-day rain, 3-day inflow), lag scanned 0–15 d",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.012, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "validation_rain_vs_inflow.png", dpi=115,
                bbox_inches="tight", facecolor="white")

    # scatter view: the correlation itself, one dot per 3-day block
    fig2, axs2 = plt.subplots(2, 3, figsize=(12.6, 8.2))
    for ax, reg in zip(axs2.ravel(), ORDER):
        x, y, k, c, bname = scatter[reg]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=14, color=COLORS[reg], alpha=0.55,
                   edgecolors="none")
        if m.sum() > 2:
            b, a = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
            ax.plot(xs, a + b * xs, color="0.2", lw=1.4, ls="--")
        ax.set_title(f"{reg} — {bname} r = {c:.2f} · lead {k} d · n = {int(m.sum())}",
                     fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel(f"3-day rain [{bname}] (mm/day avg)", fontsize=8)
        ax.set_ylabel("3-day inflow anom (GWh/day)" if bname != "raw" else "3-day inflow (GWh/day avg)", fontsize=8)
        ax.tick_params(labelsize=7.5); ax.grid(alpha=0.22)
    fig2.suptitle("3-day rain vs 3-day inflow — each dot is one 3-day block",
                  fontsize=12.5, fontweight="bold")
    fig2.text(0.5, 0.005, "rain shifted by each region's best lag · "
              "XM AporEner summed per region · GPM IMERG area-weighted over the "
              "disjoint region polygons", ha="center", fontsize=7.5, color="0.4")
    fig2.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig2.savefig(OUT / "validation_scatter.png", dpi=115,
                 bbox_inches="tight", facecolor="white")
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT}/validation_rain_vs_inflow.png + validation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
