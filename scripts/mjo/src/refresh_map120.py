#!/usr/bin/env python3
"""Out-of-band, CDS-free refresh of the RMM 120-day low-frequency (ENSO) filter map.

run_rmm.py is deliberately CDS-free and only *reads* data/reference/wind_map120.nc, so the
Wheeler & Hendon low-frequency filter must be kept current by something else — this script,
run from a daily launchd job (com.scorvec.mjo.map120.plist). CDS is unreliable and NCEP/NCAR
R1 lags ~3 months, so the source here is ARCO-ERA5 on Google Cloud (anonymous, no CDS, ~5-7
day latency).

ARCO's realtime store is chunked one-hour × all-37-levels × globe (≈154 MB/timestep), so we
keep a small persistent cache of the daily band-mean U850/U200 on the EOF grid and pull only
the *new* days each run — the ~125-day bootstrap is the one heavy read, updates are a handful
of days. A daily 12 Z snapshot is used (not a 24-hour mean) to keep each read to one chunk;
over a 120-day average the diurnal/sub-daily signal is negligible. Refresh is gated to every
REFRESH_EVERY_DAYS (the map is a 120-day mean — it drifts <1%/day), and is fail-soft: any
fetch error or an incomplete (NaN) window leaves the existing map untouched.

    python src/refresh_map120.py            # update the cache + map if due (≥5 days old)
    python src/refresh_map120.py --force    # rebuild now regardless of age
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recent_analysis as ra            # _to_eof_grid, save_map120, MAP120_PATH, REF_DIR

ROOT = HERE.parent
CLIM_PATH = ROOT / "data" / "reference" / "climatology.nc"
CACHE = ra.REF_DIR / "arco_uwind_cache.nc"     # persistent daily band-mean (time, lon), CDS-free
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

SNAP_HOUR = 12                 # daily 12 Z snapshot (one ARCO chunk/day)
SAMPLE_STRIDE_DAYS = 1         # 1 = faithful daily sampling; bump to 2 to halve the ARCO read
WINDOW_DAYS = 120              # the low-frequency mean window (Wheeler & Hendon 2004)
CACHE_DAYS = 128               # rolling cache span (window + a few days margin)
ARCO_LAG_DAYS = 7              # target the latest day comfortably inside ARCO's ~5-day latency
REFRESH_EVERY_DAYS = 1         # rebuild cadence (daily; ARCO advances ~1 day/day at ~7-day latency)


def _open_arco() -> xr.Dataset:
    return xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})


def _pull_day(ds: xr.Dataset, day: pd.Timestamp):
    """One 12 Z snapshot → (u850, u200) band-mean on the 2.5° EOF grid (each shape (lon,)), or None."""
    t = pd.Timestamp(day.date()) + pd.Timedelta(hours=SNAP_HOUR)
    try:
        snap = ds["u_component_of_wind"].sel(time=t, level=[850, 200]).compute()   # reads one chunk
    except (KeyError, Exception):                              # noqa: BLE001
        return None
    if not np.isfinite(snap.values).any():
        return None
    u850 = ra._to_eof_grid(snap.sel(level=850, drop=True))
    u200 = ra._to_eof_grid(snap.sel(level=200, drop=True))
    return u850.values, u200.values


def update_cache() -> xr.Dataset | None:
    """Append any missing days (since the cache end, or a fresh bootstrap) to the band-mean cache."""
    ds_arco = _open_arco()
    last = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=ARCO_LAG_DAYS)
    start = last - pd.Timedelta(days=CACHE_DAYS)

    cache = xr.open_dataset(CACHE) if CACHE.exists() else None
    have = set(pd.to_datetime(cache.time.values).normalize()) if cache is not None else set()
    targets = pd.date_range(start, last, freq=f"{SAMPLE_STRIDE_DAYS}D")
    need = [d for d in targets if d.normalize() not in have]
    if not need:
        print("cache already current; no ARCO read needed.")
        return cache

    print(f"pulling {len(need)} day(s) from ARCO ({need[0].date()} → {need[-1].date()}) …", flush=True)
    rows850, rows200, times = [], [], []
    lon = np.arange(0, 360, ra.GRID_RES)
    for i, d in enumerate(need):
        r = _pull_day(ds_arco, d)
        if r is None:
            continue                                           # not yet available / all-NaN
        rows850.append(r[0]); rows200.append(r[1]); times.append(np.datetime64(d.date()))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(need)} …", flush=True)
    if not times:
        print("  no new finite days pulled.")
        return cache
    new = xr.Dataset({"u850": (("time", "lon"), np.array(rows850, dtype="float32")),
                      "u200": (("time", "lon"), np.array(rows200, dtype="float32"))},
                     coords={"time": times, "lon": lon})
    merged = new if cache is None else xr.concat([cache, new], dim="time")
    merged = merged.sortby("time")
    merged = merged.isel(time=~pd.Index(merged.time.values).duplicated(keep="last"))
    merged = merged.sel(time=merged.time >= np.datetime64((last - pd.Timedelta(days=CACHE_DAYS)).date()))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(CACHE)       # atomic
    print(f"cache: {merged.sizes['time']} days "
          f"({str(merged.time.values[0])[:10]}→{str(merged.time.values[-1])[:10]})", flush=True)
    return merged


def build_map(cache: xr.Dataset, clim: xr.Dataset) -> dict | None:
    """Trailing-120-day mean of the seasonal-anomaly band-mean → {u850, u200} on the EOF grid."""
    t = pd.to_datetime(cache.time.values)
    doy = xr.DataArray(t.dayofyear, dims="time")
    a850 = cache["u850"].values - clim["clim_u850"].sel(dayofyear=doy).values
    a200 = cache["u200"].values - clim["clim_u200"].sel(dayofyear=doy).values
    win = t > (t[-1] - pd.Timedelta(days=WINDOW_DAYS))
    if win.sum() < WINDOW_DAYS / max(SAMPLE_STRIDE_DAYS, 1) * 0.8:      # need a near-complete window
        print(f"  ⚠ only {int(win.sum())} samples in the 120-day window — too few; keeping old map.")
        return None
    m = {"u850": np.asarray(a850[win].mean(0)), "u200": np.asarray(a200[win].mean(0))}
    if not (np.isfinite(m["u850"]).all() and np.isfinite(m["u200"]).all()):
        print("  ⚠ non-finite values in the mean; keeping old map.")
        return None
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild regardless of the map's age")
    args = ap.parse_args(argv)

    p = ra.MAP120_PATH
    # Gate on the DATA, not the file mtime. The old mtime gate had two failure
    # modes that made the "daily" refresh run every other day at best: timedelta
    # .days truncates (23 h 59 m → 0 → skip), and any git checkout/rebase that
    # rewrites the file resets its mtime and re-arms the skip. The cache's last
    # data day vs the day ARCO should now hold is the question actually at stake.
    if not args.force and p.exists() and CACHE.exists():
        try:
            with xr.open_dataset(CACHE) as c:
                last_have = pd.Timestamp(c.time.values[-1])
            target = (pd.Timestamp(datetime.now(timezone.utc).date())
                      - pd.Timedelta(days=ARCO_LAG_DAYS))
            if last_have >= target:
                print(f"cache already reaches {last_have.date()} "
                      f"(target {target.date()}); nothing to do.")
                return 0
        except Exception:                        # noqa: BLE001 — unreadable cache: rebuild
            pass
    if not CLIM_PATH.exists():
        print("ERROR: missing climatology.nc — run src/setup_reference.py first.")
        return 1

    try:
        cache = update_cache()
    except Exception as e:                       # noqa: BLE001 — never leave a broken map
        print(f"  ⚠ ARCO cache update failed ({repr(e)[:120]}); keeping the existing map.")
        return 0
    if cache is None or not cache.sizes.get("time"):
        print("  no cache data; keeping the existing map.")
        return 0

    clim = xr.open_dataset(CLIM_PATH)
    mean120 = build_map(cache, clim)
    if mean120 is None:
        return 0
    mean120["window_end"] = str(cache.time.values[-1])[:10]
    ra.save_map120(mean120)
    print(f"wind_map120.nc refreshed from ARCO-ERA5 (latest cache day "
          f"{mean120['window_end']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
