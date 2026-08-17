#!/usr/bin/env python3
"""Which basins matter for price, by month — the seasonal sensitivity map.

Index(basin, month) = MLT[GW] x gain15 x clim_rain(month)[mm/d] / 100
  gain15 = fast+slow kernel response realized within a 15-day window
  clim_rain = 25-yr corrected-IMERG basin climatology, monthly mean
i.e. the ENA swing (GW) produced by a 15-day rain anomaly equal in
magnitude to that month's climatological rain — a cross-month yardstick
for "how much can a 15-day forecast move the price-relevant supply".

Output: brazil_hydro/price_sensitivity_month.webp

    python scripts/sst/brazil_price_seasonality.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
PRIV = Path.home() / "brazil_hydro"
OUT = REPO / "brazil_hydro" / "price_sensitivity_month.webp"
REGION = {"IGUACU": "SOUTH", "URUGUAI": "SOUTH", "JACUI": "SOUTH",
          "GRANDE": "SE/CO", "PARANAIBA": "SE/CO", "TIETE": "SE/CO",
          "PARANAPANEMA": "SE/CO", "PARANA": "SE/CO",
          "PARAIBA DO SUL": "SE/CO", "SAO FRANCISCO": "NE",
          "TOCANTINS": "N", "AMAZONAS": "N"}
RCOL = {"SOUTH": "#1d6fb8", "SE/CO": "#b35806", "NE": "#2e9e4f",
        "N": "#6a3d9a"}
MONTH_DAYS = [(1, 31), (32, 59), (60, 90), (91, 120), (121, 151),
              (152, 181), (182, 212), (213, 243), (244, 273),
              (274, 304), (305, 334), (335, 365)]


def main() -> int:
    dm = json.load(open(PRIV / "out" / "brazil_models.json"))["params"]
    ena = json.load(gzip.open(PRIV / "raw" / "ena_bacia_daily.json.gz", "rt"))
    idx = {}
    mlts = {}
    for b, p in dm.items():
        d = ena.get(b, {})
        days = sorted(d)[-3650:]
        mw = np.array([d[k][0] for k in days], float)
        pc = np.array([d[k][1] for k in days], float)
        ok = np.isfinite(mw) & np.isfinite(pc) & (pc > 5)
        mlt = float(np.median(mw[ok] / (pc[ok] / 100))) / 1000    # GW
        mlts[b] = mlt
        tau, lag = p["tau_days"], p["lag_days"]
        c1, c2 = p["coefs"][1], p["coefs"][2]
        eff = max(15 - lag, 1)
        g15 = c1 * (1 - np.exp(-eff / tau)) + c2 * (1 - np.exp(-eff / 180))
        clim = np.array(p["clim365_mmday"], float)
        row = []
        for a, bnd in MONTH_DAYS:
            row.append(mlt * g15 * float(np.mean(clim[a - 1:bnd])) / 100)
        idx[b] = np.array(row)

    order = sorted(idx, key=lambda b: -np.abs(idx[b]).mean())
    M = np.array([idx[b] for b in order])
    Mplot = np.abs(M)

    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    pm = ax.pcolormesh(np.arange(13), np.arange(len(order) + 1), Mplot,
                       cmap="YlOrRd", vmin=0, vmax=np.nanmax(Mplot))
    for i, b in enumerate(order):
        for m in range(12):
            v = Mplot[i, m]
            ax.text(m + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if v > 0.55 * np.nanmax(Mplot) else "#333333",
                    fontweight="bold" if v == Mplot[:, m].max() else "normal")
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                        "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=9)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels(
        [f"{b}  ({REGION.get(b, '?')} · {mlts[b]:.1f} GW)" for b in order],
        fontsize=9)
    for lbl, b in zip(ax.get_yticklabels(), order):
        lbl.set_color(RCOL.get(REGION.get(b, ""), "#222"))
    ax.invert_yaxis()
    cb = fig.colorbar(pm, ax=ax, pad=0.015)
    cb.set_label("ENA swing (GW) from a climatological-magnitude "
                 "15-day rain anomaly", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Brazil — which basins can a 15-day forecast move, by month\n"
                 "index = MLT × 15-day kernel response × monthly climatological "
                 "rain · bold = month's most sensitive basin",
                 fontsize=12, fontweight="bold", loc="left")
    fig.text(0.01, 0.01,
             "South (blue) matters year-round; SE/CO (orange) surges in the "
             "Oct–Mar wet season, when forecasts hit the price-setting "
             "reservoirs directly. The long-memory giants stay pale all year.",
             fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
