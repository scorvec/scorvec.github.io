#!/usr/bin/env python3
"""Combined national wind+solar tracker for the renewables.html hero.

Reads the latest committed per-plant HRRR forecasts (wind + solar — independent cycles), the two
nameplate inventories, and the EIA-930 live feed (eia930_fetch.py), and writes one small JSON the
page renders into the gauge + stat cards + 48-hour forecast chart:

  - national wind / solar / combined GW over the overlapping forecast window
  - "now" combined GW + live grid-penetration % = HRRR combined ÷ EIA-930 total US net generation,
    evaluated at the latest hour both have (EIA lags ~1-2 h)
  - peak forecast GW + time, wind/solar split, and HRRR-vs-EIA-actual (model sanity check)

Penetration is left null (and flagged stale) if the EIA feed is missing, so the hero still shows
the forecast. CONUS utility-scale only (HRRR domain + EIA-860/USWTDB inventories).

    python scripts/power_hero.py
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
OUT = ASSETS / "power_data" / "power_hero.json"
EIA = ASSETS / "power_data" / "eia930_latest.json"


def _cycle(path: str) -> str:
    return Path(path).stem.replace("forecast_plant_", "")


def _national_gw(data_dir: str, power_col: str):
    """Latest per-plant forecast → national GW Series (hourly, UTC) + its cycle tag."""
    files = sorted(glob.glob(str(ASSETS / data_dir / "forecast_plant_*.csv")))
    if not files:
        return None, None
    f = files[-1]
    df = pd.read_csv(f, usecols=["valid_time", power_col])
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    s = (df.groupby("valid_time")[power_col].sum() / 1000.0).sort_index()   # MW → GW
    return s, _cycle(f)


def _at(series: pd.Series, t: pd.Timestamp):
    """Series value at time t by nearest hour (within 90 min), else None."""
    if series is None or series.empty:
        return None
    i = series.index.get_indexer([t], method="nearest")[0]
    if i < 0 or abs((series.index[i] - t).total_seconds()) > 5400:
        return None
    v = float(series.iloc[i])
    return None if np.isnan(v) else round(v, 2)


def main() -> int:
    wind, wcyc = _national_gw("wind_forecast_data", "MW")
    solar, scyc = _national_gw("solar_forecast_data", "MW_AC")
    if wind is None or solar is None:
        print("ERROR: missing wind or solar forecast_plant CSVs.")
        return 1

    # Common hourly grid over the overlap of the two forecast windows.
    lo = max(wind.index.min(), solar.index.min())
    hi = min(wind.index.max(), solar.index.max())
    idx = pd.date_range(lo, hi, freq="h", tz="UTC")
    wind = wind.reindex(idx).interpolate(limit_area="inside")
    solar = solar.reindex(idx).interpolate(limit_area="inside")
    combined = (wind + solar)

    namep = {
        "wind_GW": round(float(pd.read_csv(sorted(glob.glob(str(ASSETS / "wind_forecast_data" / "capacity_ba_*.csv")))[-1])["capacity_MW"].sum() / 1000.0), 1),
        "solar_GW": round(float(pd.read_csv(ASSETS / "solar_forecast_data" / "capacity_plant.csv")["p_cap_ac"].sum() / 1000.0), 1),
    }
    namep["combined_GW"] = round(namep["wind_GW"] + namep["solar_GW"], 1)

    now = pd.Timestamp.now(tz="UTC")
    eia = json.loads(EIA.read_text()) if EIA.exists() else None

    # Evaluate "now"/penetration at the latest hour EIA also has (else clamp to forecast).
    if eia and eia.get("latest_period"):
        t = pd.Timestamp(eia["latest_period"].replace("Z", "+00:00"))
        eia_total = eia.get("total_GW")
    else:
        t = min(max(now, idx[0]), idx[-1])
        eia_total = None
    t = min(max(t, idx[0]), idx[-1])
    cur = _at(combined, t)
    pen = round(100.0 * cur / eia_total, 1) if (cur is not None and eia_total) else None
    stale = eia is None or (now - t).total_seconds() > 3 * 3600

    fwd = combined[combined.index >= t].dropna()
    peak_t = fwd.idxmax() if not fwd.empty else combined.idxmax()

    forecast = [{"valid_time": p.strftime("%Y-%m-%dT%H:%MZ"),
                 "wind_GW": None if np.isnan(wind[p]) else round(float(wind[p]), 2),
                 "solar_GW": None if np.isnan(solar[p]) else round(float(solar[p]), 2),
                 "combined_GW": None if np.isnan(combined[p]) else round(float(combined[p]), 2)}
                for p in idx]

    out = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%MZ"),
        "wind_cycle": wcyc, "solar_cycle": scyc,
        "nameplate": namep,
        "now": {
            "valid_time": t.strftime("%Y-%m-%dT%H:%MZ"),
            "wind_GW": _at(wind, t), "solar_GW": _at(solar, t), "combined_GW": cur,
            "penetration_pct": pen,
            "eia_total_GW": eia_total,
            "eia_wind_GW": eia.get("wind_actual_GW") if eia else None,
            "eia_solar_GW": eia.get("solar_actual_GW") if eia else None,
            "stale": stale,
        },
        "peak": {"valid_time": peak_t.strftime("%Y-%m-%dT%H:%MZ"),
                 "combined_GW": round(float(combined[peak_t]), 1)},
        "forecast": forecast,
        "eia_recent": (eia.get("series", []) if eia else []),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.replace(OUT)
    print(f"wrote {OUT.name}: now {cur} GW combined "
          f"({'penetration '+str(pen)+'%' if pen else 'penetration n/a'}), "
          f"peak {out['peak']['combined_GW']} GW @ {out['peak']['valid_time']}, "
          f"nameplate {namep['combined_GW']} GW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
