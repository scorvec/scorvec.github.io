#!/usr/bin/env python3
"""Heating-oil model, phase 1: where America heats with oil.

Fetches ACS 5-year B25040 (house heating fuel) by county, caches it, and
builds:
  - assets/power_data/heatoil_map.html   interactive county choropleth
    (fuel-oil/kerosene share of occupied households, hover = counts + fuels)
  - assets/power_data/heatoil_states.webp top-states bar chart

Later phases weight these households by ERA5 heating-degree-days for a daily
burn index, calibrated against EIA SEDS and validated on PADD-1A distillate.

    python census_fuel.py            # key read from ~/.census_key
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

YEAR = 2023
VARS = {"B25040_001E": "total", "B25040_002E": "utility_gas",
        "B25040_003E": "lp_gas", "B25040_004E": "electricity",
        "B25040_005E": "fuel_oil", "B25040_007E": "wood"}
CACHE = Path(__file__).parent / f"b25040_county_{YEAR}.csv"
OUT = Path(__file__).resolve().parents[2] / "assets" / "power_data"
COUNTY_GEO = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def fetch() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"fips": str})
    key = (Path.home() / ".census_key").read_text().strip()
    url = (f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,"
           f"{','.join(VARS)}&for=county:*&key={key}")
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.load(r)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for v in VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.rename(columns=VARS)
    df["fips"] = df["state"] + df["county"]
    df["oil_share"] = 100 * df["fuel_oil"] / df["total"].clip(lower=1)
    df["state_name"] = df["NAME"].str.split(", ").str[-1]
    df.to_csv(CACHE, index=False)
    return df


def build_map(df: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    with urllib.request.urlopen(COUNTY_GEO, timeout=60) as r:
        counties = json.load(r)
    hover = [(f"{n}<br>fuel oil/kerosene: {int(o):,} households ({s:.1f}%)"
              f"<br>total: {int(t):,} · gas {100*g/max(t,1):.0f}% · "
              f"electric {100*e/max(t,1):.0f}%")
             for n, o, s, t, g, e in zip(df["NAME"], df["fuel_oil"], df["oil_share"],
                                         df["total"], df["utility_gas"], df["electricity"])]
    fig = go.Figure(go.Choropleth(
        geojson=counties, locations=df["fips"], z=df["oil_share"],
        colorscale="YlOrRd", zmin=0, zmax=60,
        text=hover, hoverinfo="text",
        marker_line_width=0.1, marker_line_color="#999",
        colorbar=dict(title="% of households<br>heating with<br>fuel oil/kerosene",
                      thickness=14, len=0.7)))
    fig.update_layout(
        geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"Home heating with fuel oil / kerosene — ACS 5-year {YEAR}, by county",
                   x=0.5, font=dict(size=16)),
        paper_bgcolor="white")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / "heatoil_map.html", include_plotlyjs="cdn",
                   config={"displayModeBar": False})
    print(f"  wrote {OUT / 'heatoil_map.html'}")


def build_states(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = (df.groupby("state_name")[["total", "fuel_oil"]].sum()
            .assign(share=lambda x: 100 * x["fuel_oil"] / x["total"])
            .sort_values("fuel_oil", ascending=False).head(15))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    a1.barh(st.index[::-1], st["fuel_oil"][::-1] / 1e3, color="#c62828")
    a1.set_xlabel("households heating with fuel oil/kerosene (thousands)")
    a1.set_title("Most oil-heated households", fontsize=10.5, fontweight="bold")
    st2 = (df.groupby("state_name")[["total", "fuel_oil"]].sum()
             .assign(share=lambda x: 100 * x["fuel_oil"] / x["total"])
             .sort_values("share", ascending=False).head(15))
    a2.barh(st2.index[::-1], st2["share"][::-1], color="#e65100")
    a2.set_xlabel("% of households")
    a2.set_title("Highest fuel-oil share", fontsize=10.5, fontweight="bold")
    for a in (a1, a2):
        a.grid(True, axis="x", alpha=0.25)
        a.tick_params(labelsize=8.5)
    fig.suptitle(f"Where America heats with oil — ACS 5-year {YEAR}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "heatoil_states.webp", dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    us_oil = df["fuel_oil"].sum()
    print(f"  US oil-heated households: {us_oil:,.0f} "
          f"({100 * us_oil / df['total'].sum():.1f}%)")
    print(f"  wrote {OUT / 'heatoil_states.webp'}")


if __name__ == "__main__":
    d = fetch()
    build_map(d)
    build_states(d)
