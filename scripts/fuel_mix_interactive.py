#!/usr/bin/env python3
"""Interactive US fuel-mix explorer → assets/power_data/fuel_mix.html (Plotly).

Two linked interactive figures on one self-contained page:
  1. THE LONG VIEW — monthly US electric-power generation share by fuel since
     2001 (EIA electric-power-operational-data): the coal→gas transition and
     the wind/solar rise, as a 100%-stacked area chart with hover detail.
  2. DAILY FOSSIL BURN — oil (kb/d), gas (Bcf/d) and coal (kst/d) daily since
     Jul 2018 (EIA-930, same conversions as the static charts) with a range
     slider and the named winter storms annotated.

    EIA_API_KEY=… python scripts/fuel_mix_interactive.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fuel_burn import (EIA_KEY, fetch_daily, STORMS,               # noqa: E402
                       BBL_PER_MWH, MCF_PER_MWH, TON_PER_MWH)

OUT = HERE.parent / "assets" / "power_data" / "fuel_mix.html"

# fueltypeid → (label, group) for the monthly mix; grouped where EIA splits
FUEL_LABELS = {
    "COW": ("Coal", "Coal"), "NG": ("Natural gas", "Natural gas"),
    "PEL": ("Petroleum liquids", "Petroleum"), "PC": ("Petroleum coke", "Petroleum"),
    "NUC": ("Nuclear", "Nuclear"), "HYC": ("Hydro", "Hydro"),
    "WND": ("Wind", "Wind"), "SPV": ("Solar PV", "Solar"), "STH": ("Solar thermal", "Solar"),
    "GEO": ("Geothermal", "Other renewables"), "WWW": ("Wood", "Other renewables"),
    "WAS": ("Waste", "Other renewables"), "OOG": ("Other gases", "Other"),
    "OTH": ("Other", "Other"), "HPS": ("Pumped storage", "Other"),
}
GROUP_ORDER = ["Coal", "Natural gas", "Petroleum", "Nuclear", "Hydro",
               "Wind", "Solar", "Other renewables", "Other"]
GROUP_COLORS = {"Coal": "#4d4d4d", "Natural gas": "#d95f02", "Petroleum": "#7a2418",
                "Nuclear": "#7570b3", "Hydro": "#1f78b4", "Wind": "#33a02c",
                "Solar": "#f4c430", "Other renewables": "#a6d854", "Other": "#c9c6bd"}


def fetch_monthly_mix() -> pd.DataFrame:
    """Monthly US electric-power generation (thousand MWh) by fuel group, 2001+."""
    url = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "monthly", "data[0]": "generation",
            "facets[location][]": "US", "facets[sectorid][]": "98",   # electric power
            "start": "2001-01", "end": date.today().strftime("%Y-%m"),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        chunk = r.json()["response"]["data"]
        rows += chunk
        if len(chunk) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    df = df[df["fueltypeid"].isin(FUEL_LABELS)]
    df["group"] = df["fueltypeid"].map({k: v[1] for k, v in FUEL_LABELS.items()})
    df["period"] = pd.to_datetime(df["period"])
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    wide = df.pivot_table(index="period", columns="group", values="generation",
                          aggfunc="sum").reindex(columns=GROUP_ORDER)
    return wide.clip(lower=0.0)


def main() -> int:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set.", file=sys.stderr)
        return 1

    mix = fetch_monthly_mix()
    share = mix.div(mix.sum(axis=1), axis=0) * 100.0
    print(f"  monthly mix: {mix.index[0]:%Y-%m} → {mix.index[-1]:%Y-%m}")

    fig1 = go.Figure()
    for g in GROUP_ORDER:
        if g not in share or share[g].isna().all():
            continue
        fig1.add_trace(go.Scatter(
            x=share.index, y=share[g], name=g, mode="lines", stackgroup="one",
            line=dict(width=0.4, color=GROUP_COLORS[g]),
            hovertemplate=g + ": %{y:.1f}%<extra></extra>"))
    fig1.update_layout(
        title="US electric-power generation share by fuel — monthly since 2001",
        yaxis=dict(title="% of generation", range=[0, 100], ticksuffix="%"),
        hovermode="x unified", template="plotly_white", height=460,
        legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=50, b=20))

    oil = fetch_daily("OIL") * BBL_PER_MWH / 1000.0
    gas = fetch_daily("NG") * MCF_PER_MWH / 1e6
    coal = fetch_daily("COL") * TON_PER_MWH / 1000.0
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=("Oil (thousand barrels/day)",
                                         "Natural gas (Bcf/day)",
                                         "Coal (thousand short tons/day)"))
    for i, (df, color) in enumerate([(oil, "#7a2418"), (gas, "#d95f02"),
                                     (coal, "#4d4d4d")], start=1):
        tot = df.sum(axis=1)
        fig2.add_trace(go.Scatter(x=tot.index, y=tot.values, mode="lines",
                                  line=dict(width=1.0, color=color),
                                  name=["Oil", "Gas", "Coal"][i - 1],
                                  hovertemplate="%{y:.1f}<extra></extra>"),
                       row=i, col=1)
        sm = tot.rolling(30, min_periods=20).mean()
        fig2.add_trace(go.Scatter(x=sm.index, y=sm.values, mode="lines",
                                  line=dict(width=2.2, color=color),
                                  showlegend=False,
                                  hovertemplate="30-d: %{y:.1f}<extra></extra>"),
                       row=i, col=1)
    for nominal, name in STORMS:
        d0 = pd.Timestamp(nominal)
        tot = oil.sum(axis=1)
        win = tot[(tot.index >= d0 - pd.Timedelta(days=12)) &
                  (tot.index <= d0 + pd.Timedelta(days=12))]
        if win.empty:
            continue
        fig2.add_vline(x=win.idxmax(), line_dash="dot", line_color="#888", line_width=1)
        fig2.add_annotation(x=win.idxmax(), yref="paper", y=1.0, text=name,
                            showarrow=False, font=dict(size=9, color="#666"),
                            textangle=-90, yanchor="top", xshift=6)
    fig2.update_layout(
        title="Daily fossil burn for US power (EIA-930, thin = daily, bold = 30-day mean)",
        hovermode="x unified", template="plotly_white", height=680, showlegend=False,
        margin=dict(l=60, r=20, t=60, b=20))
    fig2.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=3, col=1)

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>US Fuel Mix — Interactive</title></head>"
        "<body style='margin:0;font-family:sans-serif;background:#fff'>"
        + fig1.to_html(full_html=False, include_plotlyjs="cdn")
        + fig2.to_html(full_html=False, include_plotlyjs=False)
        + "</body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"  wrote {OUT} ({len(html)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
