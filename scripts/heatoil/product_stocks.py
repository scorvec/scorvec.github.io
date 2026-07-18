#!/usr/bin/env python3
"""Refined-product inventories, US + ARA (product_stocks.html).

US: EIA weekly petroleum status stocks via the keyless dnav hist_xls
endpoint (current, updates every Wednesday). ARA proxy: JODI monthly
closing stocks for the Netherlands + Belgium (~2-month lag). Singapore
onshore stocks are confidential in JODI (Enterprise Singapore weekly is
subscription-only), so they cannot be included from a public source.

The JODI bulk file is a 645 MB CSV, so the filtered slice is committed
(jodi_product_stocks.csv) and refreshed only when stale.

    python product_stocks.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[2] / "assets" / "power_data"
JODI_CACHE = HERE / "jodi_product_stocks.csv"
JODI_URL = ("https://www.jodidata.org/_resources/files/downloads/oil-data/"
            "world_Secondary_CSV.zip")
JODI_MAX_AGE_DAYS = 14

US_SERIES = {"gasoline": "WGTSTUS1", "distillate": "WDISTUS1",
             "jet fuel": "WKJSTUS1", "residual": "WRESTUS1"}
ARA_PRODUCTS = {"GASOLINE": "gasoline", "GASDIES": "gasoil/diesel",
                "JETKERO": "jet/kero", "NAPHTHA": "naphtha",
                "RESFUEL": "residual"}
COLORS = {"gasoline": "#ff7f0e", "distillate": "#1f77b4",
          "gasoil/diesel": "#1f77b4", "jet fuel": "#9467bd",
          "jet/kero": "#9467bd", "naphtha": "#2ca02c",
          "residual": "#8c564b"}
YEARS = 5


def us_weekly() -> dict[str, pd.Series]:
    import requests
    out = {}
    for name, sid in US_SERIES.items():
        r = requests.get(f"https://www.eia.gov/dnav/pet/hist_xls/{sid}w.xls",
                         timeout=60)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1",
                           skiprows=2)
        df.columns = ["date", "kbbl"]
        df = df.dropna()
        out[name] = pd.Series(df["kbbl"].values / 1000.0,
                              index=pd.to_datetime(df["date"]))
    return out


def refresh_jodi() -> None:
    """Re-filter the JODI bulk file when the cache is old (slow: ~700 MB)."""
    import time
    import zipfile
    import requests
    age = (time.time() - JODI_CACHE.stat().st_mtime) / 86400 \
        if JODI_CACHE.exists() else 999
    if age < JODI_MAX_AGE_DAYS:
        return
    print(f"  JODI cache {age:.0f} d old — refreshing (large download)…")
    try:
        r = requests.get(JODI_URL, timeout=900)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        keep = []
        with zf.open(member) as fh:
            for ch in pd.read_csv(fh, chunksize=2_000_000):
                m = ch[(ch.REF_AREA.isin(["NL", "BE", "US"]))
                       & (ch.FLOW_BREAKDOWN == "CLOSTLV")
                       & (ch.UNIT_MEASURE == "KBBL")]
                keep.append(m)
        d = pd.concat(keep)
        d = d[pd.to_numeric(d.OBS_VALUE, errors="coerce").notna()]
        d[["REF_AREA", "TIME_PERIOD", "ENERGY_PRODUCT", "OBS_VALUE"]].to_csv(
            JODI_CACHE, index=False)
        print(f"  JODI cache refreshed → {d.TIME_PERIOD.max()}")
    except Exception as e:                                     # noqa: BLE001
        print(f"  JODI refresh failed ({e}); using committed cache")


def ara_monthly() -> pd.DataFrame:
    d = pd.read_csv(JODI_CACHE)
    d = d[d.REF_AREA.isin(["NL", "BE"])
          & d.ENERGY_PRODUCT.isin(ARA_PRODUCTS)].copy()
    d["OBS_VALUE"] = d.OBS_VALUE.astype(float)
    d["date"] = pd.to_datetime(d.TIME_PERIOD, format="%Y-%m")
    piv = (d.groupby(["date", "ENERGY_PRODUCT"])["OBS_VALUE"].sum()
           .unstack() / 1000.0)                                  # -> Mbbl
    return piv.rename(columns=ARA_PRODUCTS)


def main() -> int:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    us = us_weekly()
    ara = ara_monthly()
    t0 = pd.Timestamp.today() - pd.DateOffset(years=YEARS)

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.11,
        subplot_titles=(
            "United States — weekly stocks (EIA)",
            "ARA proxy: Netherlands + Belgium — monthly closing stocks (JODI)"))
    for name, s in us.items():
        s = s[s.index >= t0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=f"US {name}", legendgroup="us",
            line=dict(color=COLORS[name], width=1.8),
            hovertemplate=f"US {name} · %{{x|%b %d, %Y}}<br>"
                          "%{y:.1f} Mbbl<extra></extra>"), row=1, col=1)
    for name in ara.columns:
        s = ara[name].dropna()
        s = s[s.index >= t0]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=f"ARA {name}", legendgroup="ara",
            line=dict(color=COLORS[name], width=1.8, dash="solid"),
            hovertemplate=f"NL+BE {name} · %{{x|%b %Y}}<br>"
                          "%{y:.1f} Mbbl<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="million barrels",
                     gridcolor="rgba(0,0,0,0.08)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)")
    us_latest = max(s.index.max() for s in us.values())
    fig.update_layout(
        template="plotly_white", height=780, autosize=True,
        title=dict(text="Refined-product inventories — US & ARA region",
                   x=0.5, font=dict(size=17)),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5,
                    xanchor="center", font=dict(size=11)),
        margin=dict(l=60, r=25, t=80, b=20),
        annotations=list(fig.layout.annotations) + [dict(
            text=f"EIA WPSR through {us_latest:%b %d, %Y} · JODI through "
                 f"{ara.index.max():%b %Y} · Singapore onshore stocks are "
                 "not publicly reported (Enterprise Singapore, subscription)",
            xref="paper", yref="paper", x=0.5, y=1.045, showarrow=False,
            font=dict(size=10.5, color="#777"))])
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "product_stocks.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True, "displaylogo": False})
    print(f"  US weekly → {us_latest:%Y-%m-%d}; ARA → {ara.index.max():%Y-%m}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    refresh_jodi()
    raise SystemExit(main())
