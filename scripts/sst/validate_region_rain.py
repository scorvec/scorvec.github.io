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

    summary = {}
    fig, axs = plt.subplots(3, 2, figsize=(12.2, 9.6), sharex=True)
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    for ax, reg in zip(axs.ravel(), ORDER):
        rr = np.array(rain[reg]["mm"], float)
        ii = inflow[reg]
        ok = np.isfinite(ii)
        # lag correlation: inflow vs trailing-K mean rain, K in 1..15
        best = (0, 0.0)
        for k in range(1, 16):
            kern = np.ones(k) / k
            sm = np.convolve(rr, kern, mode="full")[:len(rr)]
            m = ok & np.isfinite(sm)
            if m.sum() > 30:
                c = float(np.corrcoef(sm[m], ii[m])[0, 1])
                if c > best[1]:
                    best = (k, c)
        k, c = best
        kern = np.ones(max(k, 1)) / max(k, 1)
        sm = np.convolve(rr, kern, mode="full")[:len(rr)]
        summary[reg] = dict(best_trailing_days=k, corr=round(c, 3),
                            n_days=int(ok.sum()),
                            inflow_mean_gwh=round(float(np.nanmean(ii)), 2))
        print(f"  {reg}: r={c:.2f} at trailing {k} d "
              f"(mean inflow {np.nanmean(ii):.1f} GWh/d)", flush=True)

        ax2 = ax.twinx()
        ax.bar(dts, rr, width=1.0, color=COLORS[reg], alpha=0.35, lw=0)
        ax.plot(dts, sm, color=COLORS[reg], lw=1.6,
                label=f"rain, trailing {k}-d mean")
        ax2.plot(dts, ii, color="k", lw=1.3, label="XM inflow (GWh/d)")
        ax.set_title(f"{reg} — r = {c:.2f} (rain led {k} d)", fontsize=10,
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
             "r maximized over trailing-mean windows 1–15 d",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.012, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "validation_rain_vs_inflow.png", dpi=115,
                bbox_inches="tight", facecolor="white")
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT}/validation_rain_vs_inflow.png + validation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
