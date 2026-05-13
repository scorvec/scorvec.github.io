"""Direct-from-ISO fetchers for actual wind generation.

These are used as a fallback (or replacement) for gridstatus.io's
hosted API to avoid burning through monthly row quotas.

Each fetcher returns a DataFrame with columns:
    region (str), valid_time (datetime, UTC, tz-naive), actual_MW (float)

Strategy:
    - ERCOT: hits the public fuel-mix dashboard JSON endpoint directly
      (no library dependency). Covers ~24-48 hours of recent history at
      5-minute resolution.
    - SPP, CAISO: use the open-source `gridstatus` library (different
      from the paid `gridstatusio` hosted API). It scrapes ISO public
      sites directly, has no rate limit, and no API key required.

None of these requires registration or an API key. Behavior on
failure is to print a warning and return an empty DataFrame; the
calling code can fall through to gridstatusio if it has quota.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests


_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


# ---------------------------------------------------------------------------
# ERCOT — direct from public dashboard JSON
# ---------------------------------------------------------------------------

ERCOT_FUEL_MIX_URL = (
    "https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json"
)


def fetch_ercot_actual_wind(start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    """Pull recent ERCOT wind actuals from the public fuel-mix dashboard.

    Covers roughly the last 24-48 hours at 5-minute resolution.
    Schema verified live 2026-05-13: data[date_str] is a dict keyed by
    timestamp strings, each value containing {"Wind": {"gen": MW}, ...}.
    """
    try:
        r = requests.get(ERCOT_FUEL_MIX_URL, headers=_UA, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ERCOT (direct)... FAILED ({e})")
        return pd.DataFrame()

    rows = []
    for _date_key, ts_dict in payload.get("data", {}).items():
        if not isinstance(ts_dict, dict):
            continue
        for ts_str, record in ts_dict.items():
            if not isinstance(record, dict):
                continue
            wind_entry = record.get("Wind")
            if not isinstance(wind_entry, dict):
                continue
            wind_mw = wind_entry.get("gen")
            if wind_mw is None:
                continue
            rows.append({"timestamp": ts_str, "wind": float(wind_mw)})

    if not rows:
        print(f"  ERCOT (direct)... FAILED (no wind rows in payload)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["valid_time"] = pd.to_datetime(df["timestamp"], utc=True,
                                       errors="coerce")
    df = df.dropna(subset=["valid_time"])
    df["valid_time"] = df["valid_time"].dt.tz_localize(None)

    hourly = (df.set_index("valid_time")[["wind"]]
                .resample("1h").mean()
                .reset_index()
                .rename(columns={"wind": "actual_MW"}))
    hourly["region"] = "ERCOT"

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (hourly["valid_time"] >= start_ts) & (hourly["valid_time"] < end_ts)
    hourly = hourly.loc[mask].dropna(subset=["actual_MW"]).reset_index(drop=True)

    print(f"  ERCOT (direct)... {len(hourly)} hrs")
    return hourly[["region", "valid_time", "actual_MW"]]


# ---------------------------------------------------------------------------
# SPP and CAISO — via the open-source gridstatus library
# ---------------------------------------------------------------------------

# The open-source `gridstatus` package (NOT gridstatusio, which is the
# paid hosted API) wraps direct scrapers for each ISO's public site.
# It's free, has no rate limit, and requires no API key. Install via
#   pip install gridstatus
# We import lazily so the module loads even when the library isn't
# present (e.g. local development).

def _gridstatus_caiso():
    from gridstatus import CAISO
    return CAISO()


def _gridstatus_spp():
    from gridstatus import SPP
    return SPP()


def fetch_spp_actual_wind(start: str, end: str) -> pd.DataFrame:
    """Pull SPP wind actuals using the open-source gridstatus library.

    Uses SPP.get_fuel_mix() which returns a DataFrame with hourly or
    sub-hourly fuel-mix values including a 'Wind' column. We resample
    to hourly.
    """
    try:
        iso = _gridstatus_spp()
    except ImportError:
        print(f"  SPP (direct)... SKIPPED (gridstatus library not installed)")
        return pd.DataFrame()

    try:
        # gridstatus SPP class signature: get_fuel_mix(date, end=None).
        # Using positional date with end kwarg works across versions.
        df = iso.get_fuel_mix(start, end=end)
    except Exception as e:
        print(f"  SPP (direct)... FAILED ({e})")
        return pd.DataFrame()

    if df is None or df.empty or "Wind" not in df.columns:
        cols = list(df.columns) if df is not None else "None"
        print(f"  SPP (direct)... FAILED (no Wind column; got {cols})")
        return pd.DataFrame()

    # gridstatus returns either a "Time" column or "Interval Start";
    # pick whichever is present
    ts_col = "Interval Start" if "Interval Start" in df.columns else "Time"
    out = df[[ts_col, "Wind"]].rename(
        columns={ts_col: "valid_time", "Wind": "actual_MW"})

    # Convert to UTC tz-naive for consistency with the rest of pipeline
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    if out["valid_time"].dt.tz is not None:
        out["valid_time"] = out["valid_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    hourly = (out.set_index("valid_time")[["actual_MW"]]
                 .resample("1h").mean()
                 .reset_index())
    hourly["region"] = "SPP"

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (hourly["valid_time"] >= start_ts) & (hourly["valid_time"] < end_ts)
    hourly = hourly.loc[mask].dropna(subset=["actual_MW"]).reset_index(drop=True)

    print(f"  SPP (direct)... {len(hourly)} hrs")
    return hourly[["region", "valid_time", "actual_MW"]]


def fetch_caiso_actual_wind(start: str, end: str) -> pd.DataFrame:
    """Pull CAISO wind actuals using the open-source gridstatus library.

    Uses CAISO.get_fuel_mix(). Note that CAISO data is at 5-minute
    resolution and is summed/averaged to hourly.
    """
    try:
        iso = _gridstatus_caiso()
    except ImportError:
        print(f"  CAISO (direct)... SKIPPED (gridstatus library not installed)")
        return pd.DataFrame()

    try:
        # gridstatus CAISO class signature: get_fuel_mix(date, end=None).
        # Cap end at "today" (Pacific) since tomorrow's file doesn't
        # exist yet and produces a noisy 404 from the library.
        from datetime import date as _date
        today_pt = pd.Timestamp.now(tz="America/Los_Angeles").normalize()
        end_capped = min(pd.to_datetime(end),
                         today_pt.tz_localize(None) + pd.Timedelta(days=1))
        df = iso.get_fuel_mix(start, end=end_capped.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"  CAISO (direct)... FAILED ({e})")
        return pd.DataFrame()

    if df is None or df.empty or "Wind" not in df.columns:
        cols = list(df.columns) if df is not None else "None"
        print(f"  CAISO (direct)... FAILED (no Wind column; got {cols})")
        return pd.DataFrame()

    ts_col = "Interval Start" if "Interval Start" in df.columns else "Time"
    out = df[[ts_col, "Wind"]].rename(
        columns={ts_col: "valid_time", "Wind": "actual_MW"})

    out["valid_time"] = pd.to_datetime(out["valid_time"])
    if out["valid_time"].dt.tz is not None:
        out["valid_time"] = out["valid_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    hourly = (out.set_index("valid_time")[["actual_MW"]]
                 .resample("1h").mean()
                 .reset_index())
    hourly["region"] = "CAISO"

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (hourly["valid_time"] >= start_ts) & (hourly["valid_time"] < end_ts)
    hourly = hourly.loc[mask].dropna(subset=["actual_MW"]).reset_index(drop=True)

    print(f"  CAISO (direct)... {len(hourly)} hrs")
    return hourly[["region", "valid_time", "actual_MW"]]


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def pull_actuals_direct(regions: list[str], start: str, end: str) -> pd.DataFrame:
    """Pull actual wind generation directly from each ISO.

    ERCOT uses our hand-rolled dashboard JSON parser.
    SPP and CAISO use the open-source gridstatus library.
    Returns a single DataFrame; empty results per ISO are dropped.
    """
    fns = {
        "ERCOT": fetch_ercot_actual_wind,
        "SPP": fetch_spp_actual_wind,
        "CAISO": fetch_caiso_actual_wind,
    }
    pieces = []
    for region in regions:
        fn = fns.get(region)
        if fn is None:
            continue
        df = fn(start, end)
        if not df.empty:
            pieces.append(df)
    if not pieces:
        return pd.DataFrame(columns=["region", "valid_time", "actual_MW"])
    return pd.concat(pieces, ignore_index=True)
