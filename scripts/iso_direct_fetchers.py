"""Direct-from-ISO fetchers for actual wind generation.

These are used as a fallback (or replacement) for gridstatus.io's
hosted API to avoid burning through monthly row quotas.

Each fetcher returns a DataFrame with columns:
    region (str), valid_time (datetime, UTC, tz-naive), actual_MW (float)

Coverage strategy:
    - ERCOT: dashboard JSON endpoint for system-wide wind+solar (no auth).
      This is what powers the ercot.com fuel-mix dashboard. Note:
      published as 5-minute aggregates, system-wide. Output is averaged
      to hourly.
    - SPP: CSV download from marketplace.spp.org for the OP-STWF
      (system-wide wind forecast vs actual) latest interval and
      historical files. No auth.
    - CAISO: daily renewables watch text file from
      content.caiso.com/green/renewrpt/. Hourly resolution, published
      for each day. No auth.

None of these requires registration or an API key. Behavior on
failure is to print a warning and return an empty DataFrame; the
calling code can fall through to gridstatus if available.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


_UA = {"User-Agent": "scorvec-wind-pipeline/1.0 (educational; contact via github.com/scorvec)"}


# ---------------------------------------------------------------------------
# ERCOT
# ---------------------------------------------------------------------------

ERCOT_FUEL_MIX_URL = (
    "https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json"
)


def fetch_ercot_actual_wind(start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    """Pull recent ERCOT wind actuals from the public fuel-mix dashboard.

    The dashboard endpoint serves the current and previous operating
    day broken down by fuel type at 5-minute resolution. It does NOT
    serve arbitrary historical windows — only roughly the last 36-48
    hours are available at any given time. For older data, fall back
    to gridstatus or another source.
    """
    try:
        r = requests.get(ERCOT_FUEL_MIX_URL, headers=_UA, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ERCOT (direct)... FAILED ({e})")
        return pd.DataFrame()

    # Schema: payload["data"] is a dict keyed by date string
    # ("YYYY-MM-DD"), each containing an array of 5-min records with
    # keys including "timestamp", "wind", "solar", "gas", ...
    rows = []
    for date_str, day_data in payload.get("data", {}).items():
        if not isinstance(day_data, list):
            continue
        for rec in day_data:
            ts = rec.get("timestamp")
            wind = rec.get("wind")
            if ts is None or wind is None:
                continue
            rows.append({"timestamp": ts, "wind": float(wind)})

    if not rows:
        print(f"  ERCOT (direct)... FAILED (no wind rows in payload)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ERCOT timestamps are CPT (UTC-5 standard / UTC-6 DST) ISO strings.
    # Parse as tz-aware Central, convert to UTC, drop tz.
    df["valid_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["valid_time"])
    if df["valid_time"].dt.tz is None:
        df["valid_time"] = (df["valid_time"]
                            .dt.tz_localize("America/Chicago",
                                            ambiguous="infer",
                                            nonexistent="NaT"))
    df["valid_time"] = (df["valid_time"].dt.tz_convert("UTC")
                                          .dt.tz_localize(None))

    # Average 5-min values up to hourly
    hourly = (df.set_index("valid_time")[["wind"]]
                .resample("1h").mean()
                .reset_index()
                .rename(columns={"wind": "actual_MW"}))
    hourly["region"] = "ERCOT"

    # Filter to requested window
    start_ts = pd.to_datetime(start).tz_localize(None) if pd.to_datetime(start).tz else pd.to_datetime(start)
    end_ts = pd.to_datetime(end).tz_localize(None) if pd.to_datetime(end).tz else pd.to_datetime(end)
    mask = (hourly["valid_time"] >= start_ts) & (hourly["valid_time"] < end_ts)
    hourly = hourly.loc[mask].reset_index(drop=True)

    print(f"  ERCOT (direct)... {len(hourly)} hrs")
    return hourly[["region", "valid_time", "actual_MW"]]


# ---------------------------------------------------------------------------
# SPP
# ---------------------------------------------------------------------------

# OP-STWF (system-wide wind forecast vs actual). Public marketplace path.
# Latest interval file is always at:
SPP_STWF_LATEST_URL = (
    "https://marketplace.spp.org/file-browser-api/download/"
    "operational-data?path=%2FOP-STWF-latestInterval.csv"
)


def fetch_spp_actual_wind(start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    """Pull recent SPP wind actuals from the public OP-STWF feed.

    The latest-interval file contains a rolling window (typically last
    several hours) of wind forecast vs actual at 5-minute granularity.
    For deeper history, SPP archives each interval's file by timestamp
    — fetching the historical window for verification is a future
    enhancement.
    """
    try:
        r = requests.get(SPP_STWF_LATEST_URL, headers=_UA, timeout=timeout)
        r.raise_for_status()
        csv_text = r.text
    except Exception as e:
        print(f"  SPP (direct)... FAILED ({e})")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        print(f"  SPP (direct)... FAILED to parse CSV ({e})")
        return pd.DataFrame()

    # SPP OP-STWF columns vary by file revision but typically include:
    #   GMTIntervalEnd, Wind Forecast MW (model name), Actual Wind MW
    # Find the actual-wind column flexibly.
    actual_col = None
    for c in df.columns:
        if "actual" in c.lower() and "wind" in c.lower():
            actual_col = c
            break
    ts_col = None
    for c in df.columns:
        if "gmt" in c.lower() or "interval" in c.lower():
            ts_col = c
            break

    if actual_col is None or ts_col is None:
        print(f"  SPP (direct)... FAILED (couldn't find columns; "
              f"available: {list(df.columns)[:8]}...)")
        return pd.DataFrame()

    df = df.rename(columns={ts_col: "valid_time", actual_col: "actual_MW"})
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid_time", "actual_MW"])
    df["valid_time"] = df["valid_time"].dt.tz_localize(None)

    hourly = (df.set_index("valid_time")[["actual_MW"]]
                .resample("1h").mean()
                .reset_index())
    hourly["region"] = "SPP"

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (hourly["valid_time"] >= start_ts) & (hourly["valid_time"] < end_ts)
    hourly = hourly.loc[mask].reset_index(drop=True)

    print(f"  SPP (direct)... {len(hourly)} hrs")
    return hourly[["region", "valid_time", "actual_MW"]]


# ---------------------------------------------------------------------------
# CAISO
# ---------------------------------------------------------------------------

CAISO_DAILY_TEMPLATE = (
    "http://content.caiso.com/green/renewrpt/{ymd}_DailyRenewablesWatch.txt"
)


def fetch_caiso_actual_wind(start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    """Pull recent CAISO wind actuals from the daily renewables watch.

    This file is published once per day per operating day. We loop
    through each date in the requested window and download
    YYYYMMDD_DailyRenewablesWatch.txt for each. The file is a
    tab-separated text format with a header block, an hourly
    renewables table, and another hourly non-renewables table.

    Returns hourly UTC. CAISO local time is America/Los_Angeles.
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # CAISO publishes by Pacific date; widen the window by a day on
    # each side to cover UTC/Pacific overlap.
    pacific_start = (start_ts - pd.Timedelta(days=1)).date()
    pacific_end = (end_ts + pd.Timedelta(days=1)).date()

    pieces = []
    cur = pacific_start
    while cur <= pacific_end:
        url = CAISO_DAILY_TEMPLATE.format(ymd=cur.strftime("%Y%m%d"))
        try:
            r = requests.get(url, headers=_UA, timeout=timeout)
            if r.status_code == 404:
                # Future date or not yet published; skip silently
                cur += timedelta(days=1)
                continue
            r.raise_for_status()
            text = r.text
        except Exception as e:
            print(f"  CAISO (direct) {cur}... FAILED ({e})")
            cur += timedelta(days=1)
            continue

        # Parse the renewables block. The file has a header line, then
        # an hourly table with columns:
        #   Hour, GEOTHERMAL, BIOMASS, BIOGAS, SMALL HYDRO, WIND TOTAL,
        #   SOLAR PV, SOLAR THERMAL
        try:
            renewables = pd.read_table(
                io.StringIO(text),
                sep=r"\s+",
                skiprows=2,
                header=None,
                names=["Hour", "GEOTHERMAL", "BIOMASS", "BIOGAS",
                       "SMALL_HYDRO", "WIND_TOTAL",
                       "SOLAR_PV", "SOLAR_THERMAL"],
                nrows=24,
                skipinitialspace=True,
                engine="python",
            )
        except Exception as e:
            print(f"  CAISO (direct) {cur}... FAILED to parse ({e})")
            cur += timedelta(days=1)
            continue

        # Hour 1-24 in Pacific; map to interval-start in UTC.
        # CAISO hour ending convention: hour=1 means 0:00-1:00 Pacific.
        for _, row in renewables.iterrows():
            try:
                hour_he = int(row["Hour"])
                wind_mw = float(row["WIND_TOTAL"])
            except (ValueError, TypeError):
                continue
            # interval start in Pacific = hour ending - 1
            local_dt = datetime.combine(cur, datetime.min.time()) \
                          + timedelta(hours=hour_he - 1)
            # Pacific local → UTC. Use pandas for DST handling.
            local_pd = pd.Timestamp(local_dt).tz_localize(
                "America/Los_Angeles", ambiguous="infer", nonexistent="NaT")
            if pd.isna(local_pd):
                continue
            utc_dt = local_pd.tz_convert("UTC").tz_localize(None)
            pieces.append({"valid_time": utc_dt, "actual_MW": wind_mw})

        cur += timedelta(days=1)

    if not pieces:
        print(f"  CAISO (direct)... FAILED (no data parsed)")
        return pd.DataFrame()

    df = pd.DataFrame(pieces)
    df = df.drop_duplicates(subset="valid_time").sort_values("valid_time")
    df["region"] = "CAISO"

    mask = (df["valid_time"] >= start_ts) & (df["valid_time"] < end_ts)
    df = df.loc[mask].reset_index(drop=True)

    print(f"  CAISO (direct)... {len(df)} hrs")
    return df[["region", "valid_time", "actual_MW"]]


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def pull_actuals_direct(regions: list[str], start: str, end: str) -> pd.DataFrame:
    """Pull actual wind generation directly from each ISO.

    Returns a single DataFrame with rows for whichever ISOs returned
    data. Empty results for individual ISOs are silently dropped.
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
