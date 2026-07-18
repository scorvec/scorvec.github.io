#!/usr/bin/env python3
"""Heating-oil model, phase 3: calibrate HDDs to physical barrels.

Fit: SEDS annual residential distillate per state (DFRCP, thousand bbl) vs
that state's annual oil-weighted HDDs and oil-heating household count:

    gallons / household / year  =  a  +  b × HDD_year

pooled across the top oil states × years (household-weighted). Then the
real-time estimate is  Σ_states households × (a/365 + b × HDD_day) / 42
in barrels per day, charted against the seasonal envelope.

    python calibrate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[2] / "assets" / "power_data"
GAL_PER_BBL = 42.0
FIT_YEARS = range(2018, 2025)
FIT_STATES = ["NY", "PA", "MA", "CT", "ME", "NJ", "NH", "MD", "RI", "VT"]


def households_by_state():
    w = pd.read_csv(HERE / "oil_household_weights.csv")
    from hdd_index import STATE_ABBR
    st = w["NAME"].str.split(", ").str[-1].map(STATE_ABBR)
    return w.groupby(st)["fuel_oil"].sum()


def fit():
    hdd = pd.read_csv(HERE / "oil_hdd_daily.csv", parse_dates=["date"])
    seds = pd.read_csv(HERE / "seds_use_all_phy.csv")
    seds = seds[seds["MSN"] == "DFRCP"].set_index("State")
    hh = households_by_state()
    rows = []
    for s in FIT_STATES:
        col = f"hdd_{s}"
        if col not in hdd or s not in seds.index:
            continue
        for y in FIT_YEARS:
            ann = hdd[hdd["date"].dt.year == y]
            if len(ann) < 360 or str(y) not in seds.columns:
                continue
            hdd_y = float(ann[col].sum())
            cons_kbbl = float(seds.loc[s, str(y)])
            gal_hh = cons_kbbl * 1000 * GAL_PER_BBL / hh[s]
            rows.append((s, y, hdd_y, gal_hh, hh[s]))
    df = pd.DataFrame(rows, columns=["state", "year", "hdd", "gal_hh", "hh"])
    # STATE FIXED EFFECTS: gal_hh = a_state + b*hdd — cross-state structural
    # differences (primary vs supplemental heat, housing stock) dwarf the HDD
    # signal in a pooled fit; b is identified within-state across winters
    states = sorted(df["state"].unique())
    D = np.zeros((len(df), len(states)))
    for i, s in enumerate(states):
        D[df["state"].values == s, i] = 1.0
    X = np.column_stack([D, df["hdd"]])
    w = (df["hh"] / df["hh"].mean()).values
    beta = np.linalg.lstsq(X * np.sqrt(w[:, None]),
                           df["gal_hh"] * np.sqrt(w), rcond=None)[0]
    a_s = dict(zip(states, beta[:-1]))
    b = float(beta[-1])
    pred = X @ beta
    resid = df["gal_hh"] - pred
    within = df["gal_hh"] - D @ np.array([df[df.state == s]["gal_hh"].mean()
                                          for s in states])
    r2_within = 1 - (resid ** 2 * w).sum() / max((within ** 2 * w).sum(), 1e-9)
    print(f"  FE fit on {len(df)} state-years: b = {b:.3f} gal/hh/HDD "
          f"(within-R² {r2_within:.2f}); baselines e.g. "
          f"ME {a_s.get('ME', 0):.0f}, MD {a_s.get('MD', 0):.0f} gal/hh/yr")
    (HERE / "calibration.json").write_text(json.dumps(
        {"a_state_gal_hh_yr": {k: float(v) for k, v in a_s.items()},
         "b_gal_hh_hdd": b, "r2_within": float(r2_within), "n": len(df),
         "years": [min(FIT_YEARS), max(FIT_YEARS)]}))
    return a_s, b


def chart(fit_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hdd_index import season_year, TOP_STATES
    hdd = pd.read_csv(HERE / "oil_hdd_daily.csv", parse_dates=["date"])
    hh = households_by_state()
    a_s, b = fit_result
    a_mean = float(np.mean(list(a_s.values())))
    bbl = np.full(len(hdd), 0.0)
    hh_covered = 0.0
    for s in TOP_STATES:
        if s in hh.index and f"hdd_{s}" in hdd:
            a = a_s.get(s, a_mean)
            bbl += hh[s] * (a / 365.0 + b * hdd[f"hdd_{s}"].values) / GAL_PER_BBL
            hh_covered += hh[s]
    scale = hh.sum() / hh_covered            # extrapolate to all oil households
    hdd["kbd"] = bbl * scale / 1000.0
    hdd["season"] = hdd["date"].apply(season_year)
    hdd["sday"] = [(d - pd.Timestamp(s, 7, 1)).days
                   for d, s in zip(hdd["date"], hdd["season"])]
    cur = hdd["season"].max()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    past = hdd[hdd["season"] < cur]
    env = past.groupby("sday")["kbd"]
    days = np.arange(0, 366)
    ax.fill_between(days, env.min().reindex(days), env.max().reindex(days),
                    color="0.87", label=f"range {past['season'].min()}–{cur-1}")
    ax.plot(days, env.mean().reindex(days).rolling(7, center=True, min_periods=1).mean(),
            color="0.4", lw=1.5, label="mean")
    cd = hdd[hdd["season"] == cur]
    ax.plot(cd["sday"], cd["kbd"], color="#c62828", lw=1.8,
            label=f"{cur}–{cur+1} season")
    last = hdd.dropna(subset=["kbd"]).iloc[-1]
    ax.annotate(f"{last['kbd']:.0f} kb/d\n{last['date']:%b %d}",
                xy=(last["sday"], last["kbd"]), xytext=(10, 15),
                textcoords="offset points", fontsize=8.5, color="#c62828",
                fontweight="bold")
    mo = [0, 62, 123, 184, 245, 306]
    ax.set_xticks(mo)
    ax.set_xticklabels(["Jul", "Sep", "Nov", "Jan", "Mar", "May"], fontsize=9)
    ax.set_ylabel("estimated residential heating-oil demand (thousand bbl/day)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("US residential heating-oil demand estimate — "
                 "HDD model calibrated to EIA SEDS",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "demand = households × (baseline + slope × daily HDD), fitted on state-year "
             "SEDS residential distillate 2018–2024 · ERA5 (~6-day lag) · ACS households",
             ha="center", fontsize=7.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "heatoil_bbl.webp", dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print(f"  latest estimate: {last['kbd']:.0f} kb/d ({last['date']:%Y-%m-%d})")
    print(f"  wrote {OUT / 'heatoil_bbl.webp'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    chart(fit())
