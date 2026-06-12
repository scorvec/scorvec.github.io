#!/usr/bin/env python3
"""EIA-930 live US grid feed for the wind+solar penetration tracker.

Pulls the last ~72 h of the conterminous-US (respondent ``US48``) hourly **fuel-type** net
generation from the EIA Open Data API v2 — ``electricity/rto/fuel-type-data`` — using the same
request idiom as ``ercot_plot.py``. We sum *all* fuel types per hour to get total US generation
(the granular fuel-type feed stays current, whereas the ``region-data`` US48 NG aggregate lags
many hours), and pull out WND / SOL for the wind/solar actuals. Total is the live denominator;
wind/solar actuals let the tracker sanity-check the HRRR estimate.

Writes ``assets/power_data/eia930_latest.json``. Fail-soft: on API error a prior file is kept.

    EIA_API_KEY=… python scripts/eia930_fetch.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "power_data" / "eia930_latest.json"
EIA_KEY = os.environ.get("EIA_API_KEY", "")
RESPONDENT = "US48"               # EIA-930 conterminous-US aggregate
LOOKBACK_H = 72


def _fetch_fueltypes(start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly US48 net generation by fuel type → DataFrame indexed by period (UTC), columns=fueltype, MW."""
    url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": RESPONDENT,
            "start": start.strftime("%Y-%m-%dT%H"), "end": end.strftime("%Y-%m-%dT%H"),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        rows.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    fuel_col = next((c for c in ("fueltype", "type") if c in df.columns), None)
    if fuel_col is None:
        raise KeyError(f"no fuel-type column in EIA response ({list(df.columns)})")
    return df.pivot_table(index="period", columns=fuel_col, values="value", aggfunc="sum").sort_index()


def main() -> int:
    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set.", file=sys.stderr)
        return 1
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=LOOKBACK_H)
    try:
        wide = _fetch_fueltypes(start, end)
    except Exception as e:                            # noqa: BLE001 — fail-soft, keep prior file
        print(f"  EIA-930 fetch failed ({repr(e)[:120]}); keeping existing {OUT.name}.")
        return 0
    if wide.empty:
        print(f"  EIA-930 returned no fuel-type data; keeping existing {OUT.name}.")
        return 0

    total = wide.sum(axis=1) / 1000.0                 # sum all fuels → total generation, GW
    wind = (wide["WND"] / 1000.0) if "WND" in wide else pd.Series(dtype="float64")
    solar = (wide["SOL"] / 1000.0) if "SOL" in wide else pd.Series(dtype="float64")
    total = total[total > 0]                          # drop empty trailing hours
    if total.empty:
        print(f"  EIA-930 total generation all zero/empty; keeping existing {OUT.name}.")
        return 0
    latest = total.index[-1]
    print(f"  fuels present: {list(wide.columns)} | total latest {latest:%Y-%m-%d %HZ}")

    df = pd.DataFrame({"total_GW": total, "wind_GW": wind, "solar_GW": solar}).reindex(total.index)
    series = [{"period": p.strftime("%Y-%m-%dT%H:%MZ"),
               "total_GW": round(float(r.total_GW), 2),
               "wind_GW": None if pd.isna(r.wind_GW) else round(float(r.wind_GW), 2),
               "solar_GW": None if pd.isna(r.solar_GW) else round(float(r.solar_GW), 2)}
              for p, r in df.iterrows()]
    g = lambda v: None if pd.isna(v) else round(float(v), 2)
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "source": "EIA-930 (Hourly Electric Grid Monitor) fuel-type-data via EIA API v2, respondent US48",
        "latest_period": latest.strftime("%Y-%m-%dT%H:%MZ"),
        "total_GW": g(df.loc[latest, "total_GW"]),
        "wind_actual_GW": g(df.loc[latest, "wind_GW"]),
        "solar_actual_GW": g(df.loc[latest, "solar_GW"]),
        "series": series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.replace(OUT)
    print(f"wrote {OUT.name}: {latest:%Y-%m-%d %HZ} total {out['total_GW']} GW, "
          f"wind {out['wind_actual_GW']}, solar {out['solar_actual_GW']} GW ({len(series)} hrs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
