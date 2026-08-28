"""Shared OISST v2.1 anomaly helper — anomalies vs the TRUE 1991–2020 base.

NCEI's published sst.day.anom files are computed against a 1971–2000
climatology (per the NCEI OISST FAQ), which silently mismatched the CPC
1991–2020 convention our charts claimed. Every SST consumer in this repo now
derives anomalies through this module instead:

    anomaly(t) = sst.day.mean(t) − sst.day.mean.ltm.1991-2020(day-of-year(t))

Both files come from NOAA PSL and share the same 1/4° grid (lon 0–360). The
climatology has 365 dummy-dated steps (non-leap Jan 1 … Dec 31); Feb 29 maps
to Feb 28. Memory discipline: `anom()` is eager (xarray without dask), so
callers slice the mean field FIRST (a day, a 60-day window); for long series
use the box-level helpers, which reduce before subtracting.
"""
from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PSL_BASE = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
MEAN_URL = PSL_BASE + "/sst.day.mean.{year}.nc"
LTM_NAME = "sst.day.mean.ltm.1991-2020.nc"
BASE_LABEL = "1991–2020"


# ── download helpers (moved from sst-roni.py so every consumer shares them) ──
def _remote_is_newer(url: str, dest: Path) -> bool:
    """HEAD the URL: True only if its Last-Modified is newer than the cached file
    (so we should refresh). Any failure → False = keep the cache; that is what
    makes the big PSL files download once and then reuse until PSL appends."""
    try:
        from email.utils import parsedate_to_datetime
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as h:
            lm = h.headers.get("Last-Modified")
        if not lm:
            return False
        return parsedate_to_datetime(lm).timestamp() > dest.stat().st_mtime + 60
    except Exception as e:                                   # noqa: BLE001
        print(f"  HEAD {url.rsplit('/', 1)[-1]} failed ({repr(e)[:60]}); keeping cache", flush=True)
        return False


def _nc_ok(path: Path) -> bool:
    """Integrity probe: PSL rewrites the current-year file in place, so a file
    caught mid-update can be full-sized yet unreadable (NetCDF: HDF error).
    Byte counts alone don't catch that — actually open it."""
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            ds.sizes
        return True
    except Exception:
        return False


def download(url: str, dest: Path, force: bool = False, tries: int = 4) -> Path:
    """Streaming download with retry. A genuine 404 raises immediately (so year
    fallbacks only fire when a file is truly absent); transient failures retry."""
    import shutil, time, http.client
    DATA.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if not _remote_is_newer(url, dest):
            if dest.suffix != ".nc" or _nc_ok(dest):
                print(f"  cached: {dest.name} ({dest.stat().st_size/1e6:.1f} MB; PSL not newer)")
                return dest
            print(f"  {dest.name}: cached file is corrupt → re-downloading", flush=True)
        else:
            print(f"  {dest.name}: PSL published newer data → refreshing", flush=True)
    print(f"  downloading {url} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if tmp.stat().st_size < 1_000_000:
                raise IOError(f"suspiciously small download ({tmp.stat().st_size} B)")
            if dest.suffix == ".nc" and not _nc_ok(tmp):
                raise IOError("netCDF probe failed (corrupt or mid-update file)")
            tmp.replace(dest)
            print(f"  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, IOError, OSError) as e:
            last = e
        wait = 15 * attempt                      # escalate: a mid-update PSL file needs time to finish writing
        print(f"  download attempt {attempt}/{tries} failed ({repr(last)[:80]}); retrying in {wait}s…", flush=True)
        try: tmp.unlink()
        except OSError: pass
        time.sleep(wait)
    raise last if last is not None else RuntimeError(f"download failed: {url}")


def ensure_mean(year: int, force: bool = False) -> Path:
    dest = DATA / f"sst.day.mean.{year}.nc"
    try:
        return download(MEAN_URL.format(year=year), dest, force=force)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise                    # genuinely absent year file → caller's previous-year fallback
        return _ensure_mean_ncei(year, dest)
    except Exception:                # PSL unreachable (outage) → NCEI day-file assembly
        return _ensure_mean_ncei(year, dest)


# ── NCEI fallback: assemble the PSL-style yearly file from NCEI daily files ──
# Same OISST v2.1 on the same grid, served by a different NOAA center on
# different infrastructure (the 2026-08-22 PSL outage motivated this). NCEI
# publishes finals with ~2 weeks lag plus `_preliminary` files for the recent
# window, giving the same currency as PSL. When PSL recovers with newer data,
# the normal Last-Modified check refreshes the cache from PSL wholesale.
NCEI_BASE = ("https://www.ncei.noaa.gov/data/"
             "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr")


def _ncei_fetch_day(day) -> xr.DataArray | None:
    """One day's sst field from NCEI (final, else preliminary); None if neither
    is published yet. Transient errors retry once, then raise."""
    import shutil, time
    for suffix in ("", "_preliminary"):
        url = (f"{NCEI_BASE}/{day:%Y%m}/oisst-avhrr-v02r01.{day:%Y%m%d}{suffix}.nc")
        tmp = DATA / f".ncei_day_{day:%Y%m%d}.nc"
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                with xr.open_dataset(tmp) as ds:
                    da = ds["sst"].squeeze("zlev", drop=True).load()
                tmp.unlink()
                return da
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    break            # this variant not published → try next suffix
                if attempt == 2:
                    raise
                time.sleep(10)
            except Exception:        # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(10)
            finally:
                tmp.unlink(missing_ok=True)
    return None


def _ensure_mean_ncei(year: int, dest: Path) -> Path:
    print("  PSL unreachable → assembling from NCEI daily files", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    end = min(pd.Timestamp(year=year, month=12, day=31), today)
    base = None
    start = pd.Timestamp(year=year, month=1, day=1)
    if dest.exists() and _nc_ok(dest):
        base = xr.open_dataset(dest)["sst"].load()
        start = pd.Timestamp(base["time"].values[-1]).normalize() + pd.Timedelta(days=1)
    new = []
    for d in pd.date_range(start, end, freq="D"):
        da = _ncei_fetch_day(d)
        if da is None:
            break                    # not yet published — nothing further exists
        new.append(da)
        print(f"    NCEI {d:%Y-%m-%d} ok", flush=True)
    if not new:
        if base is not None:
            print("  NCEI has nothing newer than the cache; using cached file")
            return dest
        raise RuntimeError("NCEI fallback: no daily files retrievable")
    merged = xr.concat(([base] if base is not None else []) + new, dim="time")
    tmp = dest.with_suffix(".nc.ncei_tmp")
    merged.to_dataset(name="sst").to_netcdf(
        tmp, encoding={"sst": {"zlib": True, "complevel": 1, "dtype": "float32"}})
    tmp.replace(dest)
    print(f"  NCEI-assembled {dest.name}: through "
          f"{pd.Timestamp(merged['time'].values[-1]):%Y-%m-%d} "
          f"({dest.stat().st_size/1e6:.0f} MB)", flush=True)
    return dest


# ── climatology ──────────────────────────────────────────────────────────────
_LTM = None


def ltm() -> xr.DataArray:
    """The 1991–2020 daily climatology (time=365, lat, lon), lazily opened once."""
    global _LTM
    if _LTM is None:
        p = download(PSL_BASE + "/" + LTM_NAME, DATA / LTM_NAME)
        ds = xr.open_dataset(p, decode_times=False)
        _LTM = ds["sst"]
    return _LTM


# non-leap day-of-year lookup, with Feb 29 folded onto Feb 28
_DOY = {}
for _m in range(1, 13):
    for _d in range(1, 32):
        try:
            _DOY[(_m, _d)] = pd.Timestamp(2001, _m, _d).dayofyear - 1
        except ValueError:
            pass
_DOY[(2, 29)] = _DOY[(2, 28)]


def clim_indices(times) -> np.ndarray:
    """ltm time-axis index (0..364) for each timestamp."""
    ts = pd.DatetimeIndex(times)
    return np.array([_DOY[(t.month, t.day)] for t in ts], dtype=int)


def clim_for(times, chunked: bool = False) -> xr.DataArray:
    """Climatology field aligned to the given timestamps (dims: time, lat, lon).

    chunked=True keeps the result dask-backed (chunked along time) — REQUIRED
    when aligning against a long multi-year axis, where an eager result would
    materialize tens of GB. Use with dask-backed means (open_mfdataset)."""
    idx = clim_indices(times)
    src = ltm().chunk({"time": 92}) if chunked else ltm()
    c = src.isel(time=xr.DataArray(idx, dims="time"))
    return c.assign_coords(time=pd.DatetimeIndex(times))


def anom(mean_da: xr.DataArray) -> xr.DataArray:
    """Gridded anomaly = mean − climatology(doy). EAGER without dask — slice the
    mean field to what you need (one day, a 60-day window) before calling."""
    return mean_da - clim_for(mean_da["time"].values)


# ── box-level series (reduce first — cheap for long/multi-year series) ───────
def _wmean_box(da: xr.DataArray, la: str, lo: str, lat_rng, lon_rng) -> xr.DataArray:
    """Cosine-lat-weighted box mean, honoring either latitude ordering; NaN
    (land) cells drop out of numerator and denominator via weighted mean."""
    lat = da[la]
    latsel = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    lonsel = (da[lo] >= lon_rng[0]) & (da[lo] <= lon_rng[1])
    box = da.where(latsel & lonsel, drop=True)
    w = np.cos(np.deg2rad(box[la]))
    return box.weighted(w).mean(dim=(la, lo), skipna=True)


_BOX_CLIM_CACHE: dict = {}


def box_clim(la: str, lo: str, lat_rng, lon_rng) -> np.ndarray:
    """365-value climatological box-mean series (same weighting as the data box,
    so box-mean(mean) − box-mean(clim) == box-mean(mean − clim) exactly)."""
    key = (lat_rng, lon_rng)
    if key not in _BOX_CLIM_CACHE:
        c = _wmean_box(ltm(), la, lo, lat_rng, lon_rng)
        _BOX_CLIM_CACHE[key] = np.asarray(c.values, dtype="float64")
    return _BOX_CLIM_CACHE[key]


def box_mean_series(da: xr.DataArray, la: str, lo: str, lat_rng, lon_rng) -> pd.Series:
    """Daily box-mean series of the field as-is (absolute SST when fed means)."""
    return _wmean_box(da, la, lo, lat_rng, lon_rng).to_series()


def box_anom_series(mean_da: xr.DataArray, la: str, lo: str, lat_rng, lon_rng) -> pd.Series:
    """Daily box-mean ANOMALY series (vs 1991–2020) from a daily-mean field."""
    absser = box_mean_series(mean_da, la, lo, lat_rng, lon_rng)
    clim = box_clim(la, lo, lat_rng, lon_rng)
    idx = clim_indices(absser.index)
    return absser - pd.Series(clim[idx], index=absser.index)
