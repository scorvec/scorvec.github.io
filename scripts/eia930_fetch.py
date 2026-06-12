#!/usr/bin/env python3
"""EIA-930 live US grid feed for the wind+solar penetration tracker.

Pulls the last ~72 h of the conterminous-US (respondent ``US48``) hourly grid data from the
EIA Open Data API v2 — the same API and request idiom already used in ``ercot_plot.py``:

  - total net generation   electricity/rto/region-data/data/      facets type=NG
  - wind / solar actuals    electricity/rto/fuel-type-data/data/   facets fueltype=WND / SOL

Writes ``assets/power_data/eia930_latest.json`` — the live denominator (total generation) plus
EIA's own wind/solar actuals (used on the tracker to sanity-check the HRRR estimate). Fail-soft:
if the API errors and a prior file exists, the prior file is kept (the tracker then flags the
penetration as stale) rather than overwriting it with partial data.

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


def _fetch(endpoint: str, facets: dict[str, str], start: datetime, end: datetime) -> pd.Series:
    """Hourly EIA v2 series (period→MW) for one endpoint + facet set, offset-paginated."""
    url = f"https://api.eia.gov/v2/electricity/rto/{endpoint}/data/"
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": RESPONDENT,
            "start": start.strftime("%Y-%m-%dT%H"), "end": end.strftime("%Y-%m-%dT%H"),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }
        for k, v in facets.items():
            params[k] = v
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        rows.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (df.dropna(subset=["value"]).drop_duplicates("period")
              .set_index("period")["value"].sort_index())


def main() -> int:
    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set.", file=sys.stderr)
        return 1
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=LOOKBACK_H)
    try:
        total = _fetch("region-data", {"facets[type][]": "NG"}, start, end)
        wind = _fetch("fuel-type-data", {"facets[fueltype][]": "WND"}, start, end)
        solar = _fetch("fuel-type-data", {"facets[fueltype][]": "SOL"}, start, end)
    except Exception as e:                            # noqa: BLE001 — fail-soft, keep prior file
        print(f"  EIA-930 fetch failed ({repr(e)[:120]}); keeping existing {OUT.name}.")
        return 0
    if total.empty:
        print(f"  EIA-930 returned no total generation; keeping existing {OUT.name}.")
        return 0

    df = pd.DataFrame({"total_GW": total / 1000.0,
                       "wind_GW": wind / 1000.0,
                       "solar_GW": solar / 1000.0}).sort_index()
    df = df[df["total_GW"].notna()]                   # total is the required denominator
    latest = df.index[-1]
    series = [{"period": p.strftime("%Y-%m-%dT%H:%MZ"),
               "total_GW": round(float(r.total_GW), 2),
               "wind_GW": None if pd.isna(r.wind_GW) else round(float(r.wind_GW), 2),
               "solar_GW": None if pd.isna(r.solar_GW) else round(float(r.solar_GW), 2)}
              for p, r in df.iterrows()]
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "source": "EIA-930 (Hourly Electric Grid Monitor) via EIA API v2, respondent US48",
        "latest_period": latest.strftime("%Y-%m-%dT%H:%MZ"),
        "total_GW": round(float(df.loc[latest, "total_GW"]), 2),
        "wind_actual_GW": None if pd.isna(df.loc[latest, "wind_GW"]) else round(float(df.loc[latest, "wind_GW"]), 2),
        "solar_actual_GW": None if pd.isna(df.loc[latest, "solar_GW"]) else round(float(df.loc[latest, "solar_GW"]), 2),
        "series": series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.replace(OUT)
    print(f"wrote {OUT.name}: {latest:%Y-%m-%d %HZ} total {out['total_GW']} GW, "
          f"wind {out['wind_actual_GW']}, solar {out['solar_actual_GW']} GW ({len(series)} hrs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
