#!/usr/bin/env python3
"""XM national load (DemaReal): 26-year evolution + temperature link.

Load: POST /hourly MetricId=DemaReal Entity=Sistema, 24 hourly kWh
summed to daily national demand, 2000->present (incremental cache).

Temperature: WB2 ERA5 6h t2m (1.5 deg, 2000-2023) at the four biggest
demand centers, population-weighted (Bogota 8, Medellin 4, Cali 2.5,
Barranquilla 2.2 M) -> daily mean series (cached). Correlation is
computed on anomalies: load detrended (365-d trailing mean) and
weekday-adjusted; temperature vs day-of-year climatology.

Outputs:
  ~/colombia_hydro/raw/load_daily.json.gz     (cache)
  ~/colombia_hydro/raw/city_t2m_daily.json    (cache, 2000-2023)
  colombia_hydro/data/load.json               (series + fitted stats)

    python scripts/sst/xm_load.py [--backfill]
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
API = "https://servapibi.xm.com.co/hourly"
CACHE = Path.home() / "colombia_hydro" / "raw" / "load_daily.json.gz"
T2M_CACHE = Path.home() / "colombia_hydro" / "raw" / "city_t2m_daily.json"
OUT_JSON = REPO / "colombia_hydro" / "data" / "load.json"
START = "2000-01-01"
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
CITIES = {"bogota": (4.6, -74.1, 8.0), "medellin": (6.25, -75.6, 4.0),
          "cali": (3.45, -76.5, 2.5), "barranquilla": (11.0, -74.8, 2.2)}


def fetch_range(d0: datetime, d1: datetime) -> dict:
    out = {}
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        for attempt in range(3):
            try:
                r = requests.post(API, json={"MetricId": "DemaReal",
                                             "StartDate": f"{cur:%Y-%m-%d}",
                                             "EndDate": f"{end:%Y-%m-%d}",
                                             "Entity": "Sistema"}, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:                    # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  retry {cur:%Y-%m} ({repr(e)[:40]})", flush=True)
        for it in r.json().get("Items", []):
            for e in it.get("HourlyEntities", []):
                v = e["Values"]
                tot = sum(float(v[f"Hour{h:02d}"] or 0) for h in range(1, 25)
                          if v.get(f"Hour{h:02d}") not in (None, ""))
                if tot > 0:
                    out[it["Date"]] = tot
        print(f"  {cur:%Y-%m-%d}..{end:%Y-%m-%d}: {len(out)} days", flush=True)
        cur = end + timedelta(days=1)
    return out


def load_series(backfill: bool) -> dict:
    data = {}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    have = sorted(data)
    d1 = datetime.now() - timedelta(days=1)
    d0 = (datetime.strptime(START, "%Y-%m-%d") if backfill or not have
          else datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=2))
    if backfill and have:
        first = datetime.strptime(have[0], "%Y-%m-%d")
        if first > d0:
            data.update(fetch_range(d0, first - timedelta(days=1)))
        d0 = datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=2)
    if (d1 - d0).days >= 0:
        data.update(fetch_range(d0, d1))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def city_t2m() -> dict:
    """Population-weighted daily mean t2m (degC), 2000-2023, cached."""
    if T2M_CACHE.exists():
        return json.loads(T2M_CACHE.read_text())
    import xarray as xr
    import pandas as pd
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    t = ds["2m_temperature"].sel(time=slice("2000-01-01", "2023-01-09"))
    wsum = sum(w for _, _, w in CITIES.values())
    acc = None
    for name, (la, lo, w) in CITIES.items():
        cell = t.sel(latitude=la, longitude=lo % 360, method="nearest")
        print(f"  t2m {name} …", flush=True)
        v = cell.compute()
        acc = (w / wsum) * v if acc is None else acc + (w / wsum) * v
    daily = acc.resample(time="1D").mean() - 273.15
    out = {"dates": [f"{pd.Timestamp(x):%Y-%m-%d}" for x in daily.time.values],
           "t2m": np.round(daily.values, 2).tolist()}
    T2M_CACHE.write_text(json.dumps(out, separators=(",", ":")))
    return out


def main() -> int:
    backfill = "--backfill" in sys.argv[1:]
    data = load_series(backfill)
    days = sorted(data)
    dates = np.array(days, dtype="datetime64[D]")
    gw = np.array([data[d] for d in days]) / 1e6 / 24.0     # avg GW
    print(f"load: {days[0]}..{days[-1]} ({len(days)} days), "
          f"now ~{gw[-30:].mean():.2f} GW", flush=True)

    # ── temperature correlation (2000-2023 overlap) ─────────────────────────
    stats = {}
    try:
        tt = city_t2m()
        tdates = np.array(tt["dates"], dtype="datetime64[D]")
        tv = np.array(tt["t2m"], float)
        common, li, ti = np.intersect1d(dates, tdates, return_indices=True)
        L = gw[li]
        T = tv[ti]
        wd = np.array([d.item().weekday() for d in common])
        doy = np.array([min(d.item().timetuple().tm_yday, 365)
                        for d in common])
        # load anomaly: % vs 365-d trailing mean, weekday offsets removed
        tr = np.full(len(L), np.nan)
        for i in range(365, len(L)):
            tr[i] = L[i - 365:i].mean()
        la = 100 * (L / tr - 1)
        Wd = np.zeros(7)
        m0 = np.isfinite(la)
        for w in range(7):
            Wd[w] = np.nanmedian(la[m0 & (wd == w)])
        la = la - Wd[wd]
        # temp anomaly vs doy clim
        Ta = np.full(len(T), np.nan)
        for d_ in range(1, 366):
            m = doy == d_
            dist = np.minimum(np.abs(doy - d_), 365 - np.abs(doy - d_))
            mm = dist <= 10
            Ta[m] = T[m] - np.nanmean(T[mm])
        m = np.isfinite(la) & np.isfinite(Ta)
        r_daily = float(np.corrcoef(Ta[m], la[m])[0, 1])
        b = np.polyfit(Ta[m], la[m], 1)
        # monthly aggregation (the cleaner signal)
        mon = common.astype("datetime64[M]")
        umon = np.unique(mon[m])
        lam = np.array([np.nanmean(la[m & (mon == u)]) for u in umon])
        tam = np.array([np.nanmean(Ta[m & (mon == u)]) for u in umon])
        r_mon = float(np.corrcoef(tam, lam)[0, 1])
        bm = np.polyfit(tam, lam, 1)
        stats = {"n_days": int(m.sum()),
                 "r_daily": round(r_daily, 3),
                 "pct_per_degC_daily": round(float(b[0]), 2),
                 "r_monthly": round(r_mon, 3),
                 "pct_per_degC_monthly": round(float(bm[0]), 2),
                 "window": f"{common[0]}..{common[-1]}",
                 "cities": {k: v[2] for k, v in CITIES.items()}}
        print("temp link:", stats, flush=True)
    except Exception as e:                              # noqa: BLE001
        print(f"temperature link skipped: {repr(e)[:100]}")

    keep = dates > dates[-1] - np.timedelta64(3 * 365, "D")
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    annual = {int(y): round(float(np.nanmean(gw[yr == y])), 2)
              for y in np.unique(yr) if (yr == y).sum() > 300}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "units": "GW (daily average power)",
        "annual_mean_gw": annual,
        "temp_link": stats,
        "recent": {"dates": [str(x) for x in dates[keep]],
                   "gw": np.round(gw[keep], 3).tolist()},
        "full": {"dates": [str(x) for x in dates[::7]],
                 "gw": np.round(gw[::7], 3).tolist()},
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
