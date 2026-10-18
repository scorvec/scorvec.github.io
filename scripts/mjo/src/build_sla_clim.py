#!/usr/bin/env python3
"""Baseline for the ocean Kelvin wave tracker: sea-level anomaly climatology, tropical Pacific.

Source: NOAA CoastWatch ERDDAP `noaacwBLENDEDsshDaily` (blended altimetry, daily, 0.25 deg, relative to
the DTU15 mean sea surface; continuous from Feb 2017), read anonymously. Sampled every second grid point
(0.5 deg) over 15S-15N, 120E-70W; the Pacific crosses the dateline, so each year is two requests.

At every grid point a least-squares fit over 2017-02 .. 2025-12 of
    mean + trend + 3 annual harmonics
so the product's anomaly = SLA minus this fit: the seasonal cycle and the rise of global mean sea level
come out, the intraseasonal Kelvin signal and the ENSO state stay in. Output:
    scripts/mjo/data/reference/sla_clim.nc   coef (8, lat, lon), t0 (fractional year of the trend origin)

    python build_sla_clim.py          # ~20 requests, a few minutes; yearly files cached in data/sla_raw/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from doycal import noleap_doy

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "reference" / "sla_clim.nc"
CACHE = HERE.parent / "data" / "sla_raw"
ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDsshDaily.nc"
LAT = (-15.125, 15.125)
T0 = 2021.5  # trend origin, mid-record
NHARM = 3


# CoastWatch's ERDDAP sits behind a proxy that answers 502 to any request running longer than ~10 s
# (measured 2026-10-09: 205 days of SLA or NBS wind = 502 every time, 30 days of SLA 7 s, 15 days 2-4 s), so
# every read is cut into CHUNK_DAYS pieces with explicit end dates.
CHUNK_DAYS = 15


def last_time(dataset_nc: str) -> pd.Timestamp:
    """Newest time step a CoastWatch griddap dataset has published (a ~100-byte request)."""
    import urllib.request

    base = dataset_nc.rsplit(".", 1)[0]
    with urllib.request.urlopen(f"{base}.csv?time%5B(last)%5D", timeout=60) as r:
        rows = r.read().decode().split()
    return pd.Timestamp(rows[-1]).tz_localize(None).normalize()


def time_chunks(t0, t1, days: int = CHUNK_DAYS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    a, b = pd.Timestamp(t0).normalize(), pd.Timestamp(t1).normalize()
    out = []
    while a <= b:
        e = min(a + pd.Timedelta(days=days - 1), b)
        out.append((a, e))
        a = e + pd.Timedelta(days=1)
    return out


def get_nc(q: str, tries: int = 4, what: str = "") -> xr.Dataset:
    """Download one griddap .nc subset with retries; raises the last error."""
    import tempfile
    import urllib.request

    for attempt in range(tries):
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
                urllib.request.urlretrieve(q, f.name)
                return xr.open_dataset(f.name).load()
        except Exception as e:  # noqa: BLE001
            print(f"    {what} retry {attempt + 1}: {str(e)[:80]}", flush=True)
            if attempt + 1 == tries:
                raise
            time.sleep(5 * (attempt + 1))


def url(t0: str, t1: str, lon0: float, lon1: float, lat=LAT, stride=2) -> str:
    return (
        f"{ERDDAP}?sla%5B({t0}T00:00:00Z):1:({t1}T00:00:00Z)%5D"
        f"%5B({lat[0]}):{stride}:({lat[1]})%5D%5B({lon0}):{stride}:({lon1})%5D"
    )


def fetch(t0: str, t1: str, lat=LAT, stride=2) -> xr.DataArray:
    """SLA over 120E-70W for [t0, t1] ("last" = the newest day published), longitudes as 0-360, in metres."""
    if t1 == "last":
        t1 = f"{last_time(ERDDAP):%Y-%m-%d}"
    pieces = []
    for a, b in time_chunks(t0, t1):
        parts = []
        for lo0, lo1 in ((120.125, 179.875), (-179.875, -70.125)):
            try:
                d = get_nc(url(f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}", lo0, lo1, lat, stride), what=f"SLA {a:%Y-%m-%d}")
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"could not fetch SLA {a:%Y-%m-%d}..{b:%Y-%m-%d} ({str(e)[:60]})") from e
            parts.append(d["sla"])
        pieces.append(xr.concat(parts, "longitude"))
    s = xr.concat(pieces, "time")
    s = s.assign_coords(longitude=(s.longitude % 360)).sortby("longitude").sortby("latitude")
    return s.where(np.abs(s) < 3)


def design(t: pd.DatetimeIndex) -> np.ndarray:
    yr = t.year + (t.dayofyear - 1) / 365.25  # trend coordinate in average years (365.25 d)
    w = 2 * np.pi * (noleap_doy(t) - 1) / 365.25  # common-year day: 1 March = 60 every year, 29 Feb = 59.5
    cols = [np.ones(len(t)), yr - T0]
    for k in range(1, NHARM + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.stack(cols, -1)


def evaluate(coef: np.ndarray, t: pd.DatetimeIndex) -> np.ndarray:
    """coef (8, lat, lon) -> climatology (time, lat, lon)."""
    return np.einsum("tk,kij->tij", design(t), coef)


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    years = []
    for y in range(2017, 2026):
        p = CACHE / f"sla_{y}.nc"
        if not p.exists():
            t0 = "2017-02-01" if y == 2017 else f"{y}-01-01"
            t_ = time.time()
            fetch(t0, f"{y}-12-31").to_netcdf(p)
            print(f"  {y}: fetched in {time.time() - t_:.0f} s", flush=True)
        years.append(xr.open_dataarray(p))
    s = xr.concat(years, "time").sortby("time")
    t = pd.DatetimeIndex(s.time.values).normalize()
    X = design(t)
    Y = s.values.reshape(len(t), -1)
    ok = np.isfinite(Y)
    coef = np.full((X.shape[1], Y.shape[1]), np.nan, np.float32)
    # most points are complete: solve those in one go, the rest point by point
    full = ok.all(0)
    coef[:, full] = np.linalg.lstsq(X, Y[:, full], rcond=None)[0]
    for j in np.flatnonzero(~full & (ok.sum(0) > 365 * 3)):
        m = ok[:, j]
        coef[:, j] = np.linalg.lstsq(X[m], Y[m, j], rcond=None)[0]
    coef = coef.reshape(X.shape[1], s.sizes["latitude"], s.sizes["longitude"])
    res = xr.Dataset(
        {"coef": (("k", "latitude", "longitude"), coef)},
        coords={"latitude": s.latitude.values, "longitude": s.longitude.values},
    )
    res.attrs.update(
        source="NOAA CoastWatch noaacwBLENDEDsshDaily (blended altimetry SLA vs DTU15 MSS)",
        period=f"{t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}",
        t0=T0,
        terms="mean, trend (m/yr from t0), 3 annual harmonics",
        units="m",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_netcdf(OUT, encoding={"coef": {"zlib": True, "complevel": 4}})
    eq = np.abs(res.latitude) <= 2
    print(
        f"wrote {OUT}  ({len(t)} days); equatorial trend {float(res.coef.sel(k=1).where(eq).mean()) * 1000:.1f} mm/yr, "
        f"annual amplitude {float(np.hypot(res.coef.sel(k=2), res.coef.sel(k=3)).where(eq).mean()) * 100:.1f} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
