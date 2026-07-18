#!/usr/bin/env python3
"""Heating-oil model, phase 2: the oil-weighted heating-degree-day index.

Daily mean 2 m temperature (ERA5 via the local store, 4 synoptic hours) at
each oil-heating county centroid → HDD65 → weighted by that county's
fuel-oil household count → one national daily index (plus a New England
sub-index, the heart of the market). Cached incrementally; the chart shows
the current heating season against the 2018+ climatological envelope, daily
and season-cumulative.

    python hdd_index.py [--backfill-start 2018-07-01]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "era5"))
import era5_store

HERE = Path(__file__).parent
CACHE = HERE / "oil_hdd_daily.csv"
OUT = Path(__file__).resolve().parents[2] / "assets" / "power_data"
NE_STATES = ("Connecticut", "Maine", "Massachusetts", "New Hampshire",
             "Rhode Island", "Vermont")
HOURS = ("00", "06", "12", "18")


STATE_ABBR = {
 "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
 "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
 "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
 "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
 "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI",
 "Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT",
 "Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
 "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
 "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
 "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
 "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
 "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"}


def county_cells():
    w = pd.read_csv(HERE / "oil_household_weights.csv")
    da = era5_store.get_t2m_conus("2024-01-15T12:00")          # grid template
    lats, lons = da.latitude.values, da.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    iy = np.abs(lats[:, None] - w["lat"].values[None, :]).argmin(axis=0)
    ix = np.abs(lons[:, None] - w["lon"].values[None, :]).argmin(axis=0)
    wt = w["fuel_oil"].values.astype(float)
    stname = w["NAME"].str.split(", ").str[-1]
    ne = stname.isin(NE_STATES).values
    st = stname.map(STATE_ABBR).fillna("XX").values
    return iy, ix, wt, ne, st


def day_hdd(day: pd.Timestamp, iy, ix, wt, ne, st, states):
    fields = []
    for h in HOURS:
        da = era5_store.get_t2m_conus(f"{day:%Y-%m-%d}T{h}:00")
        if bool(np.isnan(da.values).all()):
            return None
        fields.append(da.values)
    tmean_f = (np.mean(fields, axis=0)[iy, ix] - 273.15) * 9 / 5 + 32
    hdd = np.maximum(65.0 - tmean_f, 0.0)
    out = [float((hdd * wt).sum() / wt.sum()),
           float((hdd[ne] * wt[ne]).sum() / wt[ne].sum())]
    for s in states:
        m = st == s
        out.append(float((hdd[m] * wt[m]).sum() / max(wt[m].sum(), 1.0)))
    return out


TOP_STATES = ["NY", "PA", "MA", "CT", "ME", "NJ", "NH", "MD", "RI", "VA",
              "VT", "NC", "OH", "AK", "WA", "MI"]


def update(backfill_start: str):
    cols = ["date", "oil_hdd", "ne_hdd"] + [f"hdd_{s}" for s in TOP_STATES]
    have = pd.read_csv(CACHE, parse_dates=["date"]) if CACHE.exists() else \
        pd.DataFrame(columns=cols)
    if list(have.columns) != cols:                 # schema change → full recompute
        have = pd.DataFrame(columns=cols)
    done = set(have["date"])
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=6)   # ERA5 lag
    rows = []
    iy, ix, wt, ne, st = county_cells()
    todo = [d for d in pd.date_range(backfill_start, end, freq="D") if d not in done]
    if todo:
        print(f"  computing {len(todo)} day(s) …", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(d):
        r = day_hdd(d, iy, ix, wt, ne, st, TOP_STATES)
        return (d, *r) if r else None

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            if r:
                rows.append(r)
            if i and i % 300 == 0:
                print(f"    {i}/{len(todo)}", flush=True)
    if rows:
        new = pd.DataFrame(rows, columns=cols)
        have = (pd.concat([have, new]).drop_duplicates("date")
                .sort_values("date").reset_index(drop=True))
        have.to_csv(CACHE, index=False)
        print(f"  cache now {len(have)} days through {have['date'].max():%Y-%m-%d}")
    return have


def season_year(d):
    return d.year if d.month >= 7 else d.year - 1


def chart(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = df.copy()
    df["season"] = df["date"].apply(season_year)
    df["sday"] = [(d - pd.Timestamp(s, 7, 1)).days for d, s in zip(df["date"], df["season"])]
    cur = df["season"].max()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    past = df[df["season"] < cur]
    env = past.groupby("sday")["oil_hdd"]
    days = np.arange(0, 366)
    a1.fill_between(env.min().reindex(days).index, env.min().reindex(days),
                    env.max().reindex(days), color="0.88",
                    label=f"range {past['season'].min()}–{cur-1}")
    a1.plot(env.mean().reindex(days).index,
            env.mean().reindex(days).rolling(7, center=True, min_periods=1).mean(),
            color="0.35", lw=1.5, label="mean")
    seasons = sorted(df["season"].unique())[-10:]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(seasons)}
    for s in seasons[:-1]:                          # recent seasons, 7-day smoothed
        g = df[df["season"] == s].sort_values("sday")
        a1.plot(g["sday"], g["oil_hdd"].rolling(7, center=True, min_periods=1).mean(),
                color=colors[s], lw=0.9, alpha=0.85, label=f"{s}–{s+1}")
    cd = df[df["season"] == cur]
    a1.plot(cd["sday"], cd["oil_hdd"], color="#c62828", lw=1.9,
            label=f"{cur}–{cur+1} season")
    a1.set_title("Daily oil-weighted HDD (national, past seasons 7-day smoothed)",
                 fontsize=10.5, fontweight="bold")
    a1.set_ylabel("heating degree days (base 65°F)")
    for s in seasons:
        g = df[df["season"] == s].sort_values("sday")
        is_cur = s == cur
        a2.plot(g["sday"], g["oil_hdd"].cumsum(),
                color="#c62828" if is_cur else colors[s],
                lw=2.4 if is_cur else 1.1, alpha=1.0 if is_cur else 0.9,
                label=f"{s}–{s+1}")
    a2.set_title("Season-cumulative oil-weighted HDD", fontsize=10.5, fontweight="bold")
    for a in (a1, a2):
        a.grid(True, alpha=0.25)
        mo = [0, 62, 123, 184, 245, 306]
        a.set_xticks(mo)
        a.set_xticklabels(["Jul", "Sep", "Nov", "Jan", "Mar", "May"], fontsize=8.5)
        a.tick_params(labelsize=8)
    a1.legend(fontsize=6.8, loc="upper right", ncol=2)
    a2.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Heating-oil demand index — HDDs weighted by each county's "
                 "oil-heating households (ERA5 · ACS)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "heatoil_hdd.webp", dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print(f"  wrote {OUT / 'heatoil_hdd.webp'}")


def chart_interactive(df: pd.DataFrame):
    """Responsive plotly version of the two-panel HDD chart (heatoil_hdd.html).
    One legend entry per season toggles that season in both panels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = df.copy()
    df["season"] = df["date"].apply(season_year)
    df["sday"] = [(d - pd.Timestamp(s, 7, 1)).days for d, s in zip(df["date"], df["season"])]
    cur = df["season"].max()
    seasons = sorted(df["season"].unique())[-10:]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(seasons)}

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.06,
        subplot_titles=("Daily oil-weighted HDD (past seasons 7-day smoothed)",
                        "Season-cumulative oil-weighted HDD"))

    past = df[df["season"] < cur]
    env = past.groupby("sday")["oil_hdd"]
    days = np.arange(0, 366)
    lo, hi = env.min().reindex(days), env.max().reindex(days)
    fig.add_trace(go.Scatter(x=days, y=lo, line=dict(width=0),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=hi, fill="tonexty",
                             fillcolor="rgba(0,0,0,0.10)", line=dict(width=0),
                             name=f"range {past['season'].min()}–{cur-1}",
                             hoverinfo="skip"), row=1, col=1)
    mean7 = env.mean().reindex(days).rolling(7, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=days, y=mean7, name="mean",
                             line=dict(color="#595959", width=2),
                             hovertemplate="mean · %{y:.1f} HDD<extra></extra>"),
                  row=1, col=1)

    for s in seasons:
        g = df[df["season"] == s].sort_values("sday")
        is_cur = s == cur
        c = "#c62828" if is_cur else colors[s]
        name = f"{s}–{s+1}" + (" (current)" if is_cur else "")
        dates = g["date"].dt.strftime("%b %d, %Y")
        daily = (g["oil_hdd"] if is_cur else
                 g["oil_hdd"].rolling(7, center=True, min_periods=1).mean())
        fig.add_trace(go.Scatter(
            x=g["sday"], y=daily, name=name, legendgroup=name,
            line=dict(color=c, width=3 if is_cur else 1.7),
            opacity=1.0 if is_cur else 0.85, customdata=dates,
            hovertemplate=f"{name} · %{{customdata}}<br>%{{y:.1f}} HDD<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=g["sday"], y=g["oil_hdd"].cumsum(), name=name, legendgroup=name,
            showlegend=False, line=dict(color=c, width=3.5 if is_cur else 1.9),
            opacity=1.0 if is_cur else 0.9, customdata=dates,
            hovertemplate=f"{name} · %{{customdata}}<br>%{{y:,.0f}} cumulative HDD<extra></extra>"),
            row=1, col=2)

    mo = dict(tickvals=[0, 62, 123, 184, 245, 306],
              ticktext=["Jul", "Sep", "Nov", "Jan", "Mar", "May"])
    fig.update_xaxes(**mo, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="heating degree days (base 65°F)", row=1, col=1)
    fig.update_layout(
        title=dict(text="Heating-oil demand index — HDDs weighted by each county's "
                        "oil-heating households (ERA5 · ACS)",
                   x=0.5, font=dict(size=17)),
        template="plotly_white", hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5,
                    xanchor="center", font=dict(size=12)),
        margin=dict(l=60, r=20, t=80, b=20), autosize=True)
    out = OUT / "heatoil_hdd.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True, "displaylogo": False})
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-start", default="2018-07-01")
    args = ap.parse_args()
    daily = update(args.backfill_start)
    chart(daily)
    try:
        chart_interactive(daily)
    except Exception as e:                                     # noqa: BLE001
        print(f"  interactive HDD chart failed ({e}); static webp still current")
