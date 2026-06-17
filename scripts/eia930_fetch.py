#!/usr/bin/env python3
"""EIA-930 live US grid feed for the wind+solar penetration tracker.

Conterminous-US (respondent ``US48``) hourly data from the EIA Open Data API v2, ~72 h back,
using the same request idiom as ``ercot_plot.py``:

  - demand + net generation   electricity/rto/region-data        facets type=D / NG
  - wind / solar actuals       electricity/rto/fuel-type-data     fueltype WND / SUN  (solar = SUN!)

The US48 *net-generation* roll-up lags ~a day, so for the live denominator we use whichever of
demand (D) / generation (NG) has the more recent last hour — demand is normally the freshest, and
for the whole US48 it ≈ generation (small net interchange). Wind/solar actuals (from fuel-type) let
the tracker sanity-check the HRRR estimate. Fail-soft: on API error a prior file is kept.

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
RESPONDENT = "US48"
LOOKBACK_H = 72


def _get(endpoint: str, facets: dict[str, str]) -> list[dict]:
    url = f"https://api.eia.gov/v2/electricity/rto/{endpoint}/data/"
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now
    start = now - timedelta(hours=LOOKBACK_H)
    rows, offset = [], 0
    while True:
        params = {
            "api_key": EIA_KEY, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": RESPONDENT,
            "start": start.strftime("%Y-%m-%dT%H"), "end": end.strftime("%Y-%m-%dT%H"),
            "length": 5000, "offset": offset,
            "sort[0][column]": "period", "sort[0][direction]": "asc", **facets,
        }
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        rows.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    return rows


def _series(rows: list[dict]) -> pd.Series:
    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = (df.dropna(subset=["value"]).drop_duplicates("period")
           .set_index("period")["value"].sort_index() / 1000.0)        # MW → GW
    return s[s > 0]


def main() -> int:
    if not EIA_KEY:
        print("ERROR: EIA_API_KEY not set.", file=sys.stderr)
        return 1
    try:
        demand = _series(_get("region-data", {"facets[type][]": "D"}))
        gen = _series(_get("region-data", {"facets[type][]": "NG"}))
        wind = _series(_get("fuel-type-data", {"facets[fueltype][]": "WND"}))
        solar = _series(_get("fuel-type-data", {"facets[fueltype][]": "SUN"}))
    except Exception as e:                            # noqa: BLE001 — fail-soft, keep prior file
        print(f"  EIA-930 fetch failed ({repr(e)[:120]}); keeping existing {OUT.name}.")
        return 0

    cand = [(s.index[-1], k, s) for k, s in (("demand", demand), ("generation", gen)) if not s.empty]
    if not cand:
        print(f"  EIA-930 returned no demand/generation; keeping existing {OUT.name}.")
        return 0
    latest, kind, denom = max(cand, key=lambda c: c[0])       # most recent of demand/generation
    print(f"  latest: demand {demand.index[-1] if not demand.empty else '—'} | "
          f"gen {gen.index[-1] if not gen.empty else '—'} | wind {wind.index[-1] if not wind.empty else '—'} "
          f"| solar {solar.index[-1] if not solar.empty else '—'} → using {kind} @ {latest:%Y-%m-%d %HZ}")

    idx = denom.index
    df = pd.DataFrame({"total_GW": denom,
                       "wind_GW": wind.reindex(idx), "solar_GW": solar.reindex(idx)})
    g = lambda v: None if pd.isna(v) else round(float(v), 2)
    series = [{"period": p.strftime("%Y-%m-%dT%H:%MZ"), "total_GW": g(r.total_GW),
               "wind_GW": g(r.wind_GW), "solar_GW": g(r.solar_GW)} for p, r in df.iterrows()]
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "source": f"EIA-930 (Hourly Electric Grid Monitor) via EIA API v2, respondent US48 ({kind})",
        "denominator_kind": kind,
        "latest_period": latest.strftime("%Y-%m-%dT%H:%MZ"),
        "total_GW": g(df.loc[latest, "total_GW"]),
        "wind_actual_GW": g(wind.reindex([latest]).iloc[0]) if not wind.empty else None,
        "solar_actual_GW": g(solar.reindex([latest]).iloc[0]) if not solar.empty else None,
        "series": series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.replace(OUT)
    print(f"wrote {OUT.name}: {kind} {out['total_GW']} GW @ {latest:%Y-%m-%d %HZ}, "
          f"EIA wind {out['wind_actual_GW']} / solar {out['solar_actual_GW']} GW ({len(series)} hrs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
