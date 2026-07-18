#!/usr/bin/env python3
"""Northeast heating-oil burn by season, 20+ years (heatoil_seasons.html).

Measured: EIA prime-supplier sales of No. 2 fuel oil, New England (PADD 1A)
+ Central Atlantic (PADD 1B), monthly 1983 – Mar 2022 (the survey breakout
ended with EIA's 2022 systems transition; dnav is frozen there). Fetched
once from the keyless dnav hist_xls endpoint and committed to
prime_supplier_ne.csv.

Modeled: seasons after the survey gap come from the calibrated HDD model
(same states), scaled to the measured series on the overlap seasons and
drawn hatched so nobody mistakes them for survey data.

    python season_history.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[2] / "assets" / "power_data"
CACHE = HERE / "prime_supplier_ne.csv"
SERIES = {"padd1a": "C210011001", "padd1b": "C210012001"}
GAL_PER_BBL = 42.0
N_SEASONS = 25
# model states inside PADD 1A+1B (DE/DC missing from model: ~2% of oil homes)
NE_STATES = ["CT", "ME", "MA", "NH", "RI", "VT", "NY", "NJ", "PA", "MD"]


def measured_monthly() -> pd.DataFrame:
    """kbbl/month per sub-district; cached (source frozen at 2022-03)."""
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["date"], index_col="date")
    import requests
    cols = {}
    for name, sid in SERIES.items():
        r = requests.get(f"https://www.eia.gov/dnav/pet/hist_xls/{sid}m.xls",
                         timeout=60)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1",
                           skiprows=2)
        df.columns = ["date", "kgal_d"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna()
        cols[name] = pd.Series(
            (df["kgal_d"] * df["date"].dt.days_in_month / GAL_PER_BBL).values,
            index=df["date"])
    out = pd.DataFrame(cols)
    out.index.name = "date"
    out.to_csv(CACHE)
    print(f"  cached prime-supplier history to {CACHE.name}")
    return out


def season_of(ts) -> int:
    return ts.year if ts.month >= 7 else ts.year - 1


def label(y: int) -> str:
    return f"{y}–{str(y + 1)[-2:]}"


def model_daily_ne() -> pd.Series:
    """Modeled NE residential heating oil, kbbl/day (calibrated HDD model)."""
    hdd = pd.read_csv(HERE / "oil_hdd_daily.csv", parse_dates=["date"])
    cal = json.loads((HERE / "calibration.json").read_text())
    w = pd.read_csv(HERE / "oil_household_weights.csv")
    from hdd_index import STATE_ABBR
    st = w["NAME"].str.split(", ").str[-1].map(STATE_ABBR)
    hh = w.groupby(st)["fuel_oil"].sum()
    a_s, b_s = cal["a_state_gal_hh_yr"], cal["b_state_gal_hh_hdd"]
    a_m, b_m = np.mean(list(a_s.values())), np.mean(list(b_s.values()))
    kbbl = np.zeros(len(hdd))
    for s in NE_STATES:
        if s in hh.index and f"hdd_{s}" in hdd:
            kbbl += hh[s] * (a_s.get(s, a_m) / 365.0
                             + b_s.get(s, b_m) * hdd[f"hdd_{s}"].values) \
                    / GAL_PER_BBL / 1000.0
    return pd.Series(kbbl, index=pd.DatetimeIndex(hdd["date"]))


def main() -> int:
    import plotly.graph_objects as go

    meas = measured_monthly()
    mseason = meas.groupby(meas.index.map(season_of)).sum() / 1000.0  # Mbbl
    complete = meas.groupby(meas.index.map(season_of))["padd1a"].count() == 12
    mseason = mseason[complete]

    mod = model_daily_ne()
    md = mod.groupby(mod.index.map(season_of)).sum() / 1000.0          # Mbbl
    days = mod.groupby(mod.index.map(season_of)).count()

    overlap = [y for y in mseason.index if y in md.index and days[y] >= 360]
    scale = float((mseason.sum(axis=1)[overlap] / md[overlap]).mean()) \
        if overlap else 1.0
    print(f"  model→measured scale {scale:.2f} on overlap "
          f"{[label(y) for y in overlap]}")

    last_day = mod.index.max()
    cur = season_of(last_day)
    model_years = [y for y in sorted(md.index)
                   if y not in mseason.index and md[y] > 0]
    years = sorted(mseason.index)[-(N_SEASONS - len(model_years)):] + model_years

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[label(y) for y in years],
        y=[mseason["padd1a"].get(y, np.nan) for y in years],
        name="New England (PADD 1A, measured)", marker_color="#4a6fa5",
        hovertemplate="New England %{x}: %{y:.1f} Mbbl<extra></extra>"))
    fig.add_trace(go.Bar(
        x=[label(y) for y in years],
        y=[mseason["padd1b"].get(y, np.nan) for y in years],
        name="Central Atlantic (PADD 1B, measured)", marker_color="#9db8d9",
        hovertemplate="Central Atlantic %{x}: %{y:.1f} Mbbl<extra></extra>"))
    fig.add_trace(go.Bar(
        x=[label(y) for y in years],
        y=[md[y] * scale if y in model_years else np.nan for y in years],
        name="HDD model (survey discontinued)", marker_color="#c98a8a",
        marker_pattern_shape="/",
        marker_line=dict(color="#c62828", width=1.2),
        hovertemplate="modeled %{x}: %{y:.1f} Mbbl<extra></extra>"))
    totals = [mseason.sum(axis=1).get(y, md.get(y, np.nan) * scale)
              for y in years]
    fig.add_trace(go.Scatter(
        x=[label(y) for y in years], y=totals, mode="text",
        text=[f"{t:.0f}" + ("*" if y == cur and last_day.month not in (5, 6)
                            else "") for y, t in zip(years, totals)],
        textposition="top center", textfont=dict(size=11, color="#333"),
        showlegend=False, hoverinfo="skip"))
    part = (f" · *{label(cur)} to date (through {last_day:%b %d})"
            if last_day.month not in (5, 6) else "")
    fig.update_layout(
        barmode="stack", template="plotly_white",
        title=dict(text="Northeast heating oil burned per season (Jul–Jun)",
                   x=0.5, font=dict(size=17)),
        yaxis_title="million barrels per heating season",
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5,
                    xanchor="center", font=dict(size=12)),
        margin=dict(l=60, r=20, t=72, b=30), autosize=True,
        annotations=[dict(
            text="EIA prime-supplier No. 2 fuel oil sales, PADD 1A+1B "
                 "(breakout ended Mar 2022) · hatched bars: HDD model scaled "
                 f"on {len(overlap)} overlap seasons{part}",
            xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
            font=dict(size=11, color="#777"))])
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "heatoil_seasons.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True, "displaylogo": False})
    print(f"  {label(years[0])} … {label(years[-1])} "
          f"({len(model_years)} modeled); wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
