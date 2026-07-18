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
                    env.max().reindex(days), color="0.85",
                    label=f"range {past['season'].min()}–{cur-1}")
    a1.plot(env.mean().reindex(days).index,
            env.mean().reindex(days).rolling(7, center=True, min_periods=1).mean(),
            color="0.4", lw=1.4, label="mean")
    cd = df[df["season"] == cur]
    a1.plot(cd["sday"], cd["oil_hdd"], color="#c62828", lw=1.6,
            label=f"{cur}–{cur+1} season")
    a1.set_title("Daily oil-weighted HDD (national)", fontsize=10.5, fontweight="bold")
    a1.set_ylabel("heating degree days (base 65°F)")
    for s, g in df.groupby("season"):
        cum = g.sort_values("sday")
        a2.plot(cum["sday"], cum["oil_hdd"].cumsum(),
                color="#c62828" if s == cur else "0.75",
                lw=2.2 if s == cur else 0.9,
                label=f"{s}–{s+1}" if s == cur else None)
    a2.set_title("Season-cumulative oil-weighted HDD", fontsize=10.5, fontweight="bold")
    for a in (a1, a2):
        a.grid(True, alpha=0.25)
        mo = [0, 62, 123, 184, 245, 306]
        a.set_xticks(mo)
        a.set_xticklabels(["Jul", "Sep", "Nov", "Jan", "Mar", "May"], fontsize=8.5)
        a.legend(fontsize=8, loc="upper right")
        a.tick_params(labelsize=8)
    fig.suptitle("Heating-oil demand index — HDDs weighted by each county's "
                 "oil-heating households (ERA5 · ACS)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "heatoil_hdd.webp", dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print(f"  wrote {OUT / 'heatoil_hdd.webp'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-start", default="2018-07-01")
    args = ap.parse_args()
    chart(update(args.backfill_start))
