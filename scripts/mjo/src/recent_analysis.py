"""
Recent analysis for the real-time RMM:

  1. The trailing 120-day-mean U850/U200 anomaly maps used to filter the
     low-frequency (interannual / ENSO) signal out of the AIFS forecast,
     following Wheeler & Hendon (2004).
  2. A self-consistent **wind-only observed RMM** track for the days leading
     into the forecast — computed with the same EOF projection as the forecast.

Two analysis sources are supported:
  - "era5" (default): ECMWF ERA5 via CDS, ~5-day latency — recent enough to
    reach within days of a real-time forecast.  Requested directly on the 2.5°
    EOF grid over the tropics, so downloads are small.
  - "ncep": NCEP/NCAR Reanalysis-1 — no CDS account needed, but lags real time
    by weeks-to-months, so the observed track ends well before the forecast.

OLR is not used: the forecast is wind-only and the NOAA interp-OLR product ends
in 2022.  The wind-only RMM reproduces the full 3-variable index almost exactly
(corr 0.99, amplitude ratio 1.00 over the training period).

Recent files live in data/reference/{era5_recent,ncep_recent}/ — separate from
the 1979–2001 training files so the EOF computation never picks them up.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from setup_reference import REF_DIR, lat_mean, download

GRID_RES = 2.5
LOOKBACK_DAYS = 190          # ≥ obs_days + 120 so trailing windows are complete

NCEP_DIR = REF_DIR / "ncep_recent"
NCEP_URL = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/"
            "pressure/uwnd.{year}.nc")

ERA5_DIR = REF_DIR / "era5_recent"

# Committed, CDS-free runtime inputs (built locally by seed_recent.py).
MAP120_PATH = REF_DIR / "wind_map120.nc"   # forecast 120-day filter map


def save_map120(mean120: dict, path: Path = MAP120_PATH) -> None:
    """Persist the 120-day-mean U850/U200 anomaly maps for CDS-free runs.

    An optional "window_end" (YYYY-MM-DD of the last analysis day in the
    120-day window) is stored as an attribute so CDS-free consumers can tell
    how stale the low-frequency filter is — the file's mtime can't (git
    checkouts rewrite it).
    """
    lon = np.arange(0, 360, GRID_RES)
    ds = xr.Dataset(
        {"u850": ("longitude", mean120["u850"]),
         "u200": ("longitude", mean120["u200"])},
        coords={"longitude": lon},
    )
    if mean120.get("window_end"):
        ds.attrs["window_end"] = str(mean120["window_end"])
    ds.to_netcdf(path)


def load_map120(path: Path = MAP120_PATH) -> dict:
    d = xr.open_dataset(path)
    out = {"u850": d["u850"].values, "u200": d["u200"].values}
    if "window_end" in d.attrs:
        out["window_end"] = str(d.attrs["window_end"])
    return out


# ── coordinate-name helpers (ERA5/CDS naming drifts between products) ──────────
def _coord(da, *names):
    for n in names:
        if n in da.coords or n in da.dims:
            return n
    raise KeyError(f"none of {names} in {list(da.coords)}")


def _to_eof_grid(da: xr.DataArray) -> xr.DataArray:
    """Return (time, lon) on the 0–357.5° 2.5° grid, latitude-band averaged."""
    lonn = _coord(da, "longitude", "lon")
    if float(da[lonn].min()) < 0:
        da = da.assign_coords({lonn: da[lonn] % 360}).sortby(lonn)
    lon_new = np.arange(0, 360, GRID_RES)
    da = da.interp({lonn: lon_new})
    return lat_mean(da)   # weighted 15°S–15°N mean, retains longitude


# ── ERA5 (CDS) ────────────────────────────────────────────────────────────────
ERA5_LAG_DAYS = 6        # ERA5T preliminary lags real time by ~5 days
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def _is_throttle(msg: str) -> bool:
    return ("rejected" in msg or "temporarily limited" in msg
            or "429" in msg or "too many" in msg.lower())


def _submit(c, req, attempts=40, wait=90):
    """Submit one request without waiting; retry if CDS rejects (per-account
    queued limit).  Returns a remote-job handle, or None on hard failure."""
    for i in range(1, attempts + 1):
        try:
            return c.retrieve("reanalysis-era5-pressure-levels", req)
        except Exception as e:
            msg = str(e)
            if _is_throttle(msg) and i < attempts:
                print(f"  submit throttled (try {i}/{attempts}); waiting {wait}s …",
                      flush=True)
                time.sleep(wait)
                continue
            print(f"  ERA5 submit failed: {msg[:120]}", flush=True)
            return None


def prefetch_era5(init: pd.Timestamp):
    """Download recent ERA5 U200/U850 (2.5°, tropics, 4×daily).  All grouped
    requests are *submitted up front* so they queue in parallel (one global
    queue wait instead of N), then downloaded as each becomes ready."""
    import cdsapi
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(quiet=True, wait_until_complete=False)

    start = init - pd.Timedelta(days=LOOKBACK_DAYS + 30)   # margin for 120-day window
    last  = init - pd.Timedelta(days=ERA5_LAG_DAYS)
    base = {"product_type": "reanalysis", "variable": "u_component_of_wind",
            "pressure_level": ["200", "850"], "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [20, -180, -20, 179.75], "grid": [GRID_RES, GRID_RES],
            "format": "netcdf"}

    # Group: full prior-year tail, full current-year months, capped latest month.
    # Tags encode the COVERAGE, not just the period: a later run with a wider
    # window gets a different filename and re-downloads. The old fixed tags
    # ("2026_full", "2026_07") made `tgt.exists()` reuse a file frozen at
    # whatever day it was first fetched — the recent tail silently never
    # advanced within a month, and new complete months never joined the "full"
    # group at all. Superseded files just sit unused (the dir is gitignored).
    groups = []   # (tag, year, [months], [days])
    py = start.year
    prev_months = [m for m in range(start.month, 13)] if py < init.year else []
    if prev_months:
        groups.append((f"{py}_m{start.month:02d}-12", py, prev_months, ALL_DAYS))
    full_months = [m for m in range(1, last.month)]               # complete months
    if full_months:
        groups.append((f"{init.year}_m01-{last.month - 1:02d}", init.year,
                       full_months, ALL_DAYS))
    cap_days = [f"{d:02d}" for d in range(1, last.day + 1)]        # partial latest month
    groups.append((f"{init.year}_{last.month:02d}d{last.day:02d}", init.year,
                   [last.month], cap_days))

    # 1. Submit everything that isn't already downloaded (parallel queueing).
    pending, files = [], []
    for tag, year, months, days in groups:
        tgt = ERA5_DIR / f"u_{tag}.nc"
        if tgt.exists():
            files.append(tgt)
            continue
        req = {**base, "year": [str(year)],
               "month": [f"{m:02d}" for m in months], "day": days}
        print(f"  submitting ERA5 {tag} ({len(months)} month(s)) …", flush=True)
        job = _submit(c, req)
        if job is not None:
            pending.append((tag, tgt, job))

    # 2. Download each as it finishes (.download blocks until that job is ready).
    for tag, tgt, job in pending:
        try:
            job.download(str(tgt))
            print(f"  downloaded ERA5 {tag}", flush=True)
            files.append(tgt)
        except Exception as e:
            print(f"  ERA5 {tag} download failed: {repr(e)[:120]}", flush=True)
    return files


def _load_recent_era5(init: pd.Timestamp):
    """Return (lm850, lm200, times, last_day) from recent ERA5."""
    files = prefetch_era5(init)
    if not files:
        raise RuntimeError("No ERA5 files downloaded (CDS unavailable)")
    ds = xr.open_mfdataset(files, combine="by_coords")
    uvar = "u" if "u" in ds else "u_component_of_wind"
    u    = ds[uvar]
    tname = _coord(u, "valid_time", "time")
    lname = _coord(u, "pressure_level", "level", "isobaricInhPa")
    if tname != "time":
        u = u.rename({tname: "time"})
    start = init - pd.Timedelta(days=LOOKBACK_DAYS + 30)
    u = u.sel(time=slice(str(start.date()), str(init.date())))

    # hourly → daily mean
    u = u.resample(time="1D").mean()
    u850 = u.sel({lname: 850}, drop=True)
    u200 = u.sel({lname: 200}, drop=True)

    lm850 = _to_eof_grid(u850).compute()
    lm200 = _to_eof_grid(u200).compute()
    times = pd.to_datetime(lm850.time.values)
    return lm850.values, lm200.values, times, times[-1]


# ── NCEP/NCAR Reanalysis-1 (no CDS account) ───────────────────────────────────
def _load_recent_ncep(init: pd.Timestamp):
    NCEP_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for year in range(init.year - 1, init.year + 1):
        dest = NCEP_DIR / f"uwnd.{year}.nc"
        download(NCEP_URL.format(year=year), dest)
        files.append(dest)

    ds   = xr.open_mfdataset(files, combine="by_coords")
    u    = ds["uwnd"]
    last = min(init, pd.Timestamp(pd.to_datetime(u.time.max().values)))
    start = last - pd.Timedelta(days=LOOKBACK_DAYS)
    u = u.sel(time=slice(str(start.date()), str(last.date())))
    lm850 = lat_mean(u.sel(level=850, drop=True)).compute()
    lm200 = lat_mean(u.sel(level=200, drop=True)).compute()
    times = pd.to_datetime(lm850.time.values)
    return lm850.values, lm200.values, times, last


# ── shared projection ─────────────────────────────────────────────────────────
def _trailing_120day_mean(arr: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """Per-longitude trailing 120-day running mean (calendar-day window)."""
    df   = pd.DataFrame(arr, index=times)
    full = df.reindex(pd.date_range(times.min(), times.max(), freq="D"))
    roll = full.rolling(window=120, min_periods=120).mean()
    return roll.reindex(times).values


def build(init, clim: xr.Dataset, eofs: xr.Dataset, obs_days: int = 60,
          source: str = "era5"):
    """Return (mean120, obs).

    mean120 : {"u850","u200"} trailing 120-day-mean anomaly maps at the latest
              analysis day — subtract from the forecast (held fixed in lead time).
    obs     : Dataset(time, rmm1, rmm2) — wind-only observed RMM, last obs_days.
    """
    init = pd.Timestamp(init)
    loader = {"era5": _load_recent_era5, "ncep": _load_recent_ncep}[source]
    lm850, lm200, times, last = loader(init)
    gap = (init - pd.Timestamp(last)).days

    doy  = xr.DataArray(times.dayofyear, dims="time")
    a850 = lm850 - clim["clim_u850"].sel(dayofyear=doy).values
    a200 = lm200 - clim["clim_u200"].sel(dayofyear=doy).values

    m850 = _trailing_120day_mean(a850, times)
    m200 = _trailing_120day_mean(a200, times)
    mean120 = {"u850": m850[-1], "u200": m200[-1]}   # fixed map at latest day

    # Wind-only observed RMM (same projection the forecast uses).
    e1 = np.concatenate([eofs["eof_u850"].sel(mode=1).values,
                         eofs["eof_u200"].sel(mode=1).values])
    e2 = np.concatenate([eofs["eof_u850"].sel(mode=2).values,
                         eofs["eof_u200"].sel(mode=2).values])
    w1 = float(eofs["pc_wind_std"].sel(mode=1))
    w2 = float(eofs["pc_wind_std"].sel(mode=2))

    f850 = (a850 - m850) / clim.attrs["std_u850"]
    f200 = (a200 - m200) / clim.attrs["std_u200"]
    comb = np.concatenate([f850, f200], axis=1)
    rmm1 = (comb @ e1) / w1
    rmm2 = (comb @ e2) / w2

    valid = ~np.isnan(rmm1)
    sel_t = times[valid][-obs_days:]
    obs = xr.Dataset(
        {"rmm1": ("time", rmm1[valid][-obs_days:]),
         "rmm2": ("time", rmm2[valid][-obs_days:])},
        coords={"time": sel_t},
        attrs={"source": source.upper()},
    )
    print(f"  recent analysis ({source}): {sel_t[0].date()} → {sel_t[-1].date()} "
          f"({len(sel_t)} obs days), 120-day map from {pd.Timestamp(last).date()}")
    if gap > 7:
        print(f"  ⚠ analysis lags init by {gap} days (data latency).")
    return mean120, obs
