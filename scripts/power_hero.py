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

# The combined penetration tracker counts only GRID-SCALE (transmission-
# connected) solar, so the numerator matches the EIA-930 grid-generation
# denominator. Plant size is the proxy: ≥10 MW is overwhelmingly
# transmission-interconnected utility solar; smaller plants are
# distribution-connected / BTM that the grid meters as reduced load, not
# generation (e.g. ISO-NE: ~3 GW of the ≥1 MW fleet vs <1 GW grid-reported).
# ≥10 MW keeps the big transmission BAs matched to grid (CISO/MISO ~1.0×)
# while pulling ISO-NE down to ~0.7 GW. One-line tunable.
GRID_SCALE_MIN_MW = 10.0


def _cycle(path: str) -> str:
    return Path(path).stem.replace("forecast_plant_", "")


def _national_gw(data_dir: str, power_col: str):
    """Latest per-plant forecast → national GW Series (hourly, UTC) + its cycle tag."""
    files = sorted(glob.glob(str(ASSETS / data_dir / "forecast_plant_*.csv")))
    if not files:
        return None, None
    f = files[-1]
    df = pd.read_csv(f, usecols=["valid_time", power_col])
    # feeds occasionally emit date-only rows (day boundaries) among full
    # timestamps — "mixed" parses per-element instead of erroring on the batch
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True, format="mixed")
    s = (df.groupby("valid_time")[power_col].sum() / 1000.0).sort_index()   # MW → GW
    return s, _cycle(f)


def _solar_gridscale_gw():
    """Latest solar forecast restricted to GRID-SCALE plants (≥ GRID_SCALE_MIN_MW,
    BA-assigned) → national GW Series + cycle tag + grid-scale nameplate GW.

    Distribution/BTM-sized plants are excluded so the combined tracker's solar
    matches grid-reported generation (and the EIA-930 denominator), rather than
    the full ≥1 MW fleet shown on the standalone solar dashboard."""
    files = sorted(glob.glob(str(ASSETS / "solar_forecast_data" / "forecast_plant_*.csv")))
    if not files:
        return None, None, 0.0
    f = files[-1]
    capp = ASSETS / "solar_forecast_data" / "capacity_plant.csv"
    cap_cols = list(pd.read_csv(capp, nrows=0).columns)
    use = ["case_id", "p_cap_ac"] + (["BA"] if "BA" in cap_cols else [])
    cap = pd.read_csv(capp, usecols=use)
    cap["case_id"] = cap["case_id"].astype(str)
    mask = cap["p_cap_ac"] >= GRID_SCALE_MIN_MW
    if "BA" in cap.columns:                       # "from the BA": drop unassigned plants
        mask &= cap["BA"].notna()
    keep = set(cap.loc[mask, "case_id"])
    nameplate_GW = round(float(cap.loc[mask, "p_cap_ac"].sum() / 1000.0), 1)

    df = pd.read_csv(f, usecols=["case_id", "valid_time", "MW_AC"])
    df["case_id"] = df["case_id"].astype(str)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True, format="mixed")
    df = df[df["case_id"].isin(keep)]
    s = (df.groupby("valid_time")["MW_AC"].sum() / 1000.0).sort_index()   # MW → GW
    return s, _cycle(f), nameplate_GW


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
    solar, scyc, solar_namep_GW = _solar_gridscale_gw()   # grid-scale only (≥10 MW, BA-assigned)
    if wind is None or solar is None:
        print("ERROR: missing wind or solar forecast_plant CSVs.")
        return 1

    # Common hourly grid over the overlap of the two forecast windows. The solar
    # CSV omits night (zero-output) rows, so solar.index understates its window —
    # derive the windows from the cycle stamps and treat absent solar hours as 0
    # (night), not NaN. Without this the combined series ended at the last *daytime*
    # solar hour, so a nighttime "now" fell outside it and penetration came back
    # undefined even though wind was generating (and solar got wrongly interpolated
    # across the night).
    w0 = pd.to_datetime(wcyc, format="%Y%m%dT%HZ", utc=True)
    s0 = pd.to_datetime(scyc, format="%Y%m%dT%HZ", utc=True)
    horizon = wind.index.max() - w0                 # actual forecast horizon (~48 h)
    lo = max(w0, s0)
    hi = min(wind.index.max(), s0 + horizon)
    if lo > hi:                                     # solar pipeline stale beyond
        idx = pd.date_range(w0, wind.index.max(), freq="h", tz="UTC")   # overlap:
        print("WARN: no wind/solar overlap — publishing wind-only")     # wind-only
    else:
        idx = pd.date_range(lo, hi, freq="h", tz="UTC")
    wind = wind.reindex(idx).interpolate(limit_area="inside")
    # short interior gaps are fetch hiccups, not night — interpolate up to 3 h;
    # longer gaps (real nights, which the CSV omits) still fill with zero
    solar = solar.reindex(idx).interpolate(limit=3, limit_area="inside").fillna(0.0)
    combined = wind + solar

    namep = {
        "wind_GW": round(float(pd.read_csv(sorted(glob.glob(str(ASSETS / "wind_forecast_data" / "capacity_ba_*.csv")))[-1])["capacity_MW"].sum() / 1000.0), 1),
        "solar_GW": solar_namep_GW,   # grid-scale (≥10 MW) nameplate, matching the tracker numerator
    }
    namep["combined_GW"] = round(namep["wind_GW"] + namep["solar_GW"], 1)

    now = pd.Timestamp.now(tz="UTC")
    eia = json.loads(EIA.read_text()) if EIA.exists() else None

    now_t = min(max(now, idx[0]), idx[-1])            # HRRR nowcast time, clamped to the forecast window
    cur = _at(combined, now_t)                        # combined GW "now"

    pen = eia_total = None
    pen_t, stale = now_t, True
    denom_kind = None

    # Denominator: most recent EIA US48 *actual* total (type D demand), aligned to its own
    # (1–2 h-lagged) hour so the numerator and denominator share the same hour. We deliberately
    # do NOT use the EIA day-ahead demand FORECAST for the current hour: the US48 forecast is a
    # sum across balancing authorities, and the bleeding-edge hours are routinely only a PARTIAL
    # sum (not all BAs have reported yet), which collapses to absurdly low values (e.g. ~70–90 GW
    # overnight) and pins penetration near 100 %. The somewhat-lagged actual is reliable.
    if eia and eia.get("latest_period") and eia.get("total_GW"):
        et = pd.Timestamp(eia["latest_period"].replace("Z", "+00:00"))
        ec = _at(combined, et)
        if ec is not None and idx[0] <= et <= now + pd.Timedelta(hours=1):
            pen = round(100.0 * ec / eia["total_GW"], 1)
            pen_t, eia_total, cur = et, eia["total_GW"], ec
            denom_kind, stale = eia.get("denominator_kind"), (now - et).total_seconds() > 3 * 3600

    fwd = combined[combined.index >= now_t].dropna()
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
            "valid_time": pen_t.strftime("%Y-%m-%dT%H:%MZ"),
            "wind_GW": _at(wind, pen_t), "solar_GW": _at(solar, pen_t), "combined_GW": cur,
            "penetration_pct": pen,
            "denominator_kind": denom_kind,
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
