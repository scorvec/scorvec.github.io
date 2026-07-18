#!/usr/bin/env python3
"""Fossil fuel burned for US electricity, by grid region: oil in thousand
barrels per day (kb/d), natural gas in billion cubic feet per day (Bcf/d),
and coal in thousand short tons per day (kst/d).

EIA-930 daily petroleum ("OIL") net generation per grid region (EIA Open Data
API v2, daily-fuel-type-data, July 2018 → present), converted to implied crude
burn with the standard EIA equivalences:

    kb/d = MWh/day × (heat rate 10.33 MMBtu/MWh) / (5.80 MMBtu/bbl) / 1000
         ≈ MWh/day × 1.781 / 1000

Oil-fired generation is tiny nationally but concentrated and event-driven —
Florida (year-round peakers), New England and New York (winter gas scarcity),
Hawaii/PR excluded (not in US48 regions). The chart shows where and when the
fleet actually burns oil, and how it has evolved since 2018.

Figure (assets/power_data/oil_burn.webp):
  top    — stacked 30-day-mean kb/d by region since Jul 2018 (top burners
           explicit, the rest grouped), with notable spikes readable.
  bottom — last 24 months, daily: US48 total + the top regions as lines.

(The key is also read from ~/.eia_key if the env var is unset.)

    EIA_API_KEY=… python scripts/fuel_burn.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_OIL = HERE.parent / "assets" / "power_data" / "oil_burn.webp"
OUT_GAS = HERE.parent / "assets" / "power_data" / "gas_burn.webp"
OUT_COAL = HERE.parent / "assets" / "power_data" / "coal_burn.webp"

EIA_KEY = os.environ.get("EIA_API_KEY", "") or (
    Path.home().joinpath(".eia_key").read_text().strip()
    if Path.home().joinpath(".eia_key").exists() else "")

REGIONS = {  # EIA-930 region respondents (conterminous US)
    "CAL": "California", "CAR": "Carolinas", "CENT": "Central",
    "FLA": "Florida", "MIDA": "Mid-Atlantic", "MIDW": "Midwest",
    "NE": "New England", "NY": "New York", "NW": "Northwest",
    "SE": "Southeast", "SW": "Southwest", "TEN": "Tennessee", "TEX": "Texas",
}
START = "2018-07-01"                       # EIA-930 fuel-mix record begins
HEAT_RATE = 10.33                          # MMBtu/MWh, EIA avg petroleum fleet
MMBTU_BBL = 5.80                           # MMBtu per barrel (EIA convention)
BBL_PER_MWH = HEAT_RATE / MMBTU_BBL        # ≈ 1.781
GAS_HR = 7.72                              # MMBtu/MWh, EIA avg gas fleet (CC-dominated)
MMBTU_MCF = 1.037                          # MMBtu per Mcf (EIA convention)
MCF_PER_MWH = GAS_HR / MMBTU_MCF           # ≈ 7.44
COAL_HR = 10.2                             # MMBtu/MWh, EIA avg coal fleet
MMBTU_TON = 19.2                           # MMBtu per short ton (US power coal mix)
TON_PER_MWH = COAL_HR / MMBTU_TON          # ≈ 0.531
# Named winter events annotated on the charts: (nominal date, label). The line
# is drawn at the DAILY-TOTAL PEAK within ±12 days of the nominal date, so the
# label lands on the actual burn spike, not on our guess of the storm timing.
STORMS = [("2021-02-15", "Uri"), ("2022-12-24", "Elliott"),
          ("2024-01-16", "Jan '24 outbreak"), ("2025-01-08", "Blair"),
          ("2025-01-21", "Enzo"), ("2026-02-05", "Fern")]

EXPLICIT = ["FLA", "NE", "NY", "MIDA", "SE"]         # oil: shown individually
EXPLICIT_GAS = ["TEX", "SE", "FLA", "MIDA", "MIDW"]  # gas: the big burners
EXPLICIT_COAL = ["MIDW", "MIDA", "SE", "CENT", "TEX"]  # coal heartland


def fetch_daily(fueltype: str) -> pd.DataFrame:
    """Daily net generation (MWh) for one fueltype per region → date × region."""
    url = "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "daily", "data[0]": "value",
            "facets[fueltype][]": fueltype, "facets[timezone][]": "Eastern",
            "start": START, "end": date.today().isoformat(),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        for i, r_id in enumerate(REGIONS):
            params[f"facets[respondent][{i}]"] = r_id
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        chunk = r.json()["response"]["data"]
        rows += chunk
        if len(chunk) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = (df.pivot_table(index="period", columns="respondent", values="value",
                           aggfunc="sum")
              .reindex(columns=list(REGIONS)).sort_index())
    # negatives are metering noise on a ~0 baseline; oil burn can't be negative
    return wide.clip(lower=0.0)


def render_fuel(vol: pd.DataFrame, unit: str, title: str, explicit_ids: list,
                out: Path, conv_note: str) -> None:
    """Two-panel chart: stacked 30-day-mean history by region + 24-month daily."""
    tot = vol.sum(axis=1)
    print(f"  {title}: latest {tot.dropna().iloc[-1]:.1f} {unit} · "
          f"record daily {tot.max():.1f} on {tot.idxmax():%Y-%m-%d}")
    sm = vol.rolling(30, min_periods=20).mean()
    order = vol.mean().sort_values(ascending=False)
    explicit = [r for r in order.index if r in explicit_ids]
    other = [r for r in REGIONS if r not in explicit]
    stack = sm[explicit].copy()
    stack["Other regions"] = sm[other].sum(axis=1)
    labels = [REGIONS[r] for r in explicit] + ["Other regions"]
    colors = ["#d95f02", "#1f78b4", "#33a02c", "#6a3d9a", "#e31a1c", "#b0b0b0"]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.6, 8.8),
                                 gridspec_kw=dict(height_ratios=[1.15, 1],
                                                  hspace=0.3))
    a0.stackplot(stack.index, [stack[c].fillna(0).values for c in stack.columns],
                 labels=labels, colors=colors, alpha=0.9, lw=0)
    a0.set_xlim(stack.index[0], stack.index[-1])
    a0.grid(True, alpha=0.2)
    a0.set_ylabel(unit)
    a0.set_title(f"{title} — 30-day mean by grid region, "
                 f"Jul 2018 – {vol.index[-1]:%b %Y}",
                 fontsize=12, fontweight="bold", loc="left")
    for nominal, name in STORMS:
        d0 = pd.Timestamp(nominal)
        win = tot[(tot.index >= d0 - pd.Timedelta(days=12)) &
                  (tot.index <= d0 + pd.Timedelta(days=12))]
        if win.empty:
            continue
        dpk = win.idxmax()
        a0.axvline(dpk, color="0.35", lw=0.8, ls=":", alpha=0.8)
        a0.annotate(name, xy=(dpk, a0.get_ylim()[1]), xytext=(3, -3),
                    textcoords="offset points", rotation=90, va="top", ha="left",
                    fontsize=10, fontweight="bold", color="0.3")
    a0.legend(fontsize=8, loc="upper left",
              ncol=2, framealpha=0.9)

    t0 = vol.index[-1] - pd.DateOffset(months=24)
    a1.plot(tot[tot.index >= t0].index, tot[tot.index >= t0].values,
            color="#222", lw=1.0, alpha=0.55, label="US48 total (daily)")
    a1.plot(sm[sm.index >= t0].index,
            sm.loc[sm.index >= t0].sum(axis=1),
            color="#222", lw=2.0, label="US48 total (30-d mean)")
    for r, c in zip(explicit[:3], colors):
        d = vol.loc[vol.index >= t0, r]
        a1.plot(d.index, d.values, color=c, lw=0.9, alpha=0.8, label=REGIONS[r])
    a1.set_xlim(t0, vol.index[-1])
    a1.grid(True, alpha=0.2)
    a1.set_ylabel(unit)
    a1.set_title("Last 24 months — daily", fontsize=10.5, fontweight="bold",
                 loc="left")
    a1.legend(fontsize=8, loc="best", ncol=2, framealpha=0.9)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.06)
    fig.text(0.5, 0.005,
             "EIA-930 daily net generation by region (Eastern time) · " + conv_note +
             " · conterminous US only",
             ha="center", fontsize=7.5, color="0.4")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def _residential_kbd() -> "pd.Series":
    """Modeled residential heating-oil demand (kb/d) from the committed
    heating-oil model outputs (see scripts/heatoil/)."""
    import json
    hdir = Path(__file__).parent / "heatoil"
    hdd = pd.read_csv(hdir / "oil_hdd_daily.csv", parse_dates=["date"])
    cal = json.loads((hdir / "calibration.json").read_text())
    w = pd.read_csv(hdir / "oil_household_weights.csv")
    import sys
    sys.path.insert(0, str(hdir))
    from hdd_index import STATE_ABBR, TOP_STATES
    st = w["NAME"].str.split(", ").str[-1].map(STATE_ABBR)
    hh = w.groupby(st)["fuel_oil"].sum()
    a_s, b_s = cal["a_state_gal_hh_yr"], cal["b_state_gal_hh_hdd"]
    a_m = sum(a_s.values()) / len(a_s)
    b_m = sum(b_s.values()) / len(b_s)
    bbl = pd.Series(0.0, index=hdd.index)
    covered = 0.0
    for s in TOP_STATES:
        if s in hh.index and f"hdd_{s}" in hdd:
            bbl += hh[s] * (a_s.get(s, a_m) / 365.0
                            + b_s.get(s, b_m) * hdd[f"hdd_{s}"]) / 42.0
            covered += hh[s]
    out = pd.Series((bbl * hh.sum() / covered / 1000.0).values,
                    index=pd.DatetimeIndex(hdd["date"]))
    return out


def render_combined_oil(power_kbd: "pd.DataFrame") -> None:
    """Weather-driven Northeast oil demand: MEASURED power-sector burn
    (EIA-930, ISNE+NYIS+PJM) stacked with the MODELED residential estimate."""
    try:
        res = _residential_kbd()
    except Exception as e:                                     # noqa: BLE001
        print(f"  combined-oil: residential model unavailable ({e}) — skipped")
        return
    iso = [c for c in ("NE", "NY", "MIDA") if c in power_kbd.columns]
    pw = power_kbd[iso].sum(axis=1)
    start = pd.Timestamp.today().normalize() - pd.DateOffset(months=30)
    pw = pw[pw.index >= start]
    res = res[res.index >= start]
    idx = pw.index.intersection(res.index)
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.fill_between(idx, 0, res.reindex(idx), color="#c62828", alpha=0.75,
                    label="residential heating oil (modeled, HDD×SEDS)")
    ax.fill_between(idx, res.reindex(idx), res.reindex(idx) + pw.reindex(idx),
                    color="#4a6fa5", alpha=0.8,
                    label="power-sector oil burn ISNE+NYIS+PJM (measured, EIA-930)")
    tail = res.reindex(idx) + pw.reindex(idx)
    for d, nm in STORMS:
        dd = pd.Timestamp(d)
        if dd in tail.index or (idx.min() < dd < idx.max()):
            ax.axvline(dd, color="0.4", lw=0.7, ls=":")
            ax.annotate(nm, xy=(dd, ax.get_ylim()[1] * 0.02), fontsize=10,
                        fontweight="bold", rotation=90, color="0.3", va="bottom")
    ax.set_ylabel("thousand barrels / day")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Northeast weather-driven oil demand — homes + power plants",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "residential = HDD model calibrated to EIA SEDS (weather only; ignores prices) · "
             "power = implied burn from EIA-930 generation (captures dual-fuel switching as it happens)",
             ha="center", fontsize=7.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT_OIL.parent / "combined_oil.webp", dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print(f"  wrote {OUT_OIL.parent / 'combined_oil.webp'}")


def main() -> int:
    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set (env or ~/.eia_key).", file=sys.stderr)
        return 1
    oil = fetch_daily("OIL")                                   # MWh/day
    print(f"  OIL: {len(oil)} days → {oil.index[-1]:%Y-%m-%d}")
    render_combined_oil(oil * BBL_PER_MWH / 1000.0)
    render_fuel(oil * BBL_PER_MWH / 1000.0, "thousand barrels / day",
                "Oil burned for US electricity", EXPLICIT, OUT_OIL,
                f"kb/d = MWh × {HEAT_RATE} MMBtu/MWh ÷ {MMBTU_BBL} MMBtu/bbl "
                f"≈ MWh × {BBL_PER_MWH:.2f}/1000 · NY is a LOWER BOUND (NYISO "
                "dual-fuel units are not fuel-split when they switch to oil)")
    gas = fetch_daily("NG")
    print(f"  NG: {len(gas)} days → {gas.index[-1]:%Y-%m-%d}")
    render_fuel(gas * MCF_PER_MWH / 1e6, "Bcf / day",
                "Natural gas burned for US electricity", EXPLICIT_GAS, OUT_GAS,
                f"Bcf/d = MWh × {GAS_HR} MMBtu/MWh ÷ {MMBTU_MCF} MMBtu/Mcf ÷ 10⁶")
    coal = fetch_daily("COL")
    print(f"  COL: {len(coal)} days → {coal.index[-1]:%Y-%m-%d}")
    render_fuel(coal * TON_PER_MWH / 1000.0, "thousand short tons / day",
                "Coal burned for US electricity", EXPLICIT_COAL, OUT_COAL,
                f"kst/d = MWh × {COAL_HR} MMBtu/MWh ÷ {MMBTU_TON} MMBtu/ton ÷ 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
