#!/usr/bin/env python3
"""WeatherNext 2 (Google DeepMind, 64 members) for the ensemble verification (ens_verify.py), from Earth Engine.

Each member's fields at one lead come as one request, already on the common 0.5 deg grid (mverify_scores.GRID; rows
90N first, columns from 180W): every native 0.25 deg pixel weighted by its overlap with the cell times the cosine of
its latitude (``to_half``; the request grid is the cells themselves, so Earth Engine's averaged pyramid is never read).
mverify's own averaging onto that grid then leaves them as they are. They are written as netCDF that mverify opens like
any model's: one file a lead with the members stacked, and the terrain and land fraction WN2 was trained with (ERA5's,
from ARCO) averaged onto the same cells by mverify's remapping (``static``).

The fields scored are those of the other ensembles that WN2 has: 2 m temperature, sea-level pressure and 10 m wind
against the stations; 500 hPa height, 850 hPa temperature and the wind at 850 and 250 hPa against the radiosondes. It
has no 2 m dew point, and its rain is left out (the 6-hourly steps would cost four times the Earth Engine time).

Earth Engine: a service account's key (EE_SA_KEY, the path of its JSON) in the project that holds the free
noncommercial quota of Earth Engine compute time (EECU), shared with other work. Before each run the month's use is
read from Cloud Monitoring (``month_eecu_hours``) and nothing is fetched above EECU_MAX hours, or when it cannot be
read; the first quota or rate refusal stops the job without retrying (QuotaStop).

Only runs whose every lead lies more than 48 h in the past are fetched, the data Google licenses under CC BY 4.0.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
MEMBERS = 64
G0 = 9.80665
# name for mverify: (Earth Engine band, unit)
BANDS = {
    "t2m": ("2m_temperature", "K"),
    "msl": ("mean_sea_level_pressure", "Pa"),
    "u10": ("10m_u_component_of_wind", "m s-1"),
    "v10": ("10m_v_component_of_wind", "m s-1"),
    "z500": ("500_geopotential", "m2 s-2"),
    "t850": ("850_temperature", "K"),
    "u850": ("850_u_component_of_wind", "m s-1"),
    "v850": ("850_v_component_of_wind", "m s-1"),
    "u250": ("250_u_component_of_wind", "m s-1"),
    "v250": ("250_v_component_of_wind", "m s-1"),
}
LAT = np.round(90.0 - 0.5 * np.arange(361), 2)  # the common grid's cells: 90N to 90S, 180W to 179.5E
LON = np.round(-180.0 + 0.5 * np.arange(720), 2)
EECU_MAX = 125.0  # EECU-hours used this month above which nothing is fetched (the free quota is about 150)
WORKERS = 4  # requests at once
PUBLIC_AFTER = pd.Timedelta(hours=48)  # data older than this is CC BY 4.0
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


class QuotaStop(SystemExit):
    pass


def _key() -> dict:
    return json.loads(Path(os.environ["EE_SA_KEY"]).read_text())


def init() -> None:
    import ee

    key = _key()
    creds = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(creds, project=key["project_id"])
    ee.data.setWorkloadTag("verification")


def month_eecu_hours() -> float | None:
    """EECU-hours the key's project has used this calendar month (Cloud Monitoring, read-only); None if unreadable."""
    import urllib.parse
    import urllib.request

    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    key = _key()
    creds = service_account.Credentials.from_service_account_info(
        key, scopes=["https://www.googleapis.com/auth/monitoring.read"]
    )
    try:
        creds.refresh(Request())
    except Exception:  # noqa: BLE001  (any failure to authenticate: unreadable)
        return None
    now = pd.Timestamp.now(tz="UTC")
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q = {
        "filter": 'metric.type="earthengine.googleapis.com/project/cpu/usage_time"',
        "interval.startTime": f"{start:%Y-%m-%dT%H:%M:%S}Z",
        "interval.endTime": f"{now:%Y-%m-%dT%H:%M:%S}Z",
        "aggregation.alignmentPeriod": f"{max(60, int((now - start).total_seconds()))}s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
    }
    url = f"https://monitoring.googleapis.com/v3/projects/{key['project_id']}/timeSeries?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {creds.token}"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
    except (OSError, ValueError):
        return None
    pts = [p for ts in d.get("timeSeries", []) for p in ts["points"]]
    return sum(float(p["value"].get("doubleValue", p["value"].get("int64Value", 0))) for p in pts) / 3600


def budget() -> None:
    """Stop unless this month's EECU use is known and below EECU_MAX."""
    used = month_eecu_hours()
    if used is None:
        raise QuotaStop("Earth Engine use this month cannot be read (does the key's account have Monitoring Viewer?)")
    print(f"Earth Engine this month: {used:.1f} EECU-h", flush=True)
    if used >= EECU_MAX:
        raise QuotaStop(f"Earth Engine use {used:.1f} EECU-h is at the ceiling ({EECU_MAX:.0f}): nothing fetched")


def public(init_time: pd.Timestamp, last_lead_h: int) -> bool:
    """Whether every lead of the run lies more than PUBLIC_AFTER in the past (the CC BY 4.0 data)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return init_time + pd.Timedelta(hours=last_lead_h) < now - PUBLIC_AFTER


def to_half(im):
    """The 0.5 deg cell means: each native pixel weighted by its overlap with the cell times the cosine of its
    latitude (Earth Engine's reduceResolution weighs by overlap alone)."""
    import ee

    native = ee.Projection("EPSG:4326", [0.25, 0, -180.125, 0, -0.25, 90.125])
    half = ee.Projection("EPSG:4326", [0.5, 0, -180.25, 0, -0.5, 90.25])
    names = im.bandNames()
    c = ee.Image.pixelLonLat().select("latitude").multiply(np.pi / 180).cos().max(0).rename("w")
    both = im.multiply(c).addBands(c).setDefaultProjection(native)
    r = both.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(half)
    return r.select(names).divide(r.select("w"))


def _pixels(img) -> np.ndarray:
    """img's bands on the common grid as a structured array, retrying a transient failure twice."""
    import ee

    req = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": LON.size, "height": LAT.size},
            "affineTransform": {
                "scaleX": 0.5,
                "shearX": 0,
                "translateX": -180.25,
                "shearY": 0,
                "scaleY": -0.5,
                "translateY": 90.25,
            },
            "crsCode": "EPSG:4326",
        },
    }
    for attempt in range(3):
        try:
            return ee.data.computePixels(req)
        except ee.EEException as e:
            msg = str(e).lower()
            if any(k in msg for k in ("quota", "429", "too many", "rate", "restricted")):
                raise QuotaStop(f"Earth Engine refused: {e}") from e
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def lead_fields(init_time: pd.Timestamp, lead_h: int) -> dict[str, np.ndarray]:
    """Every member's fields at one lead: {name: (member, lat, lon) float32}."""
    import ee

    t = ee.Date(init_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    coll = (
        ee.ImageCollection(ASSET)
        .filter(ee.Filter.date(t, t.advance(1, "minute")))
        .filter(ee.Filter.eq("forecast_hour", lead_h))
    )
    names = list(BANDS)
    bands = [BANDS[n][0] for n in names]

    def member(m: int) -> np.ndarray:
        one = coll.filter(ee.Filter.eq("ensemble_member", str(m)))
        a = _pixels(to_half(ee.Image(one.first()).select(bands, names).toFloat()))
        return np.stack([a[n].astype(np.float32) for n in names])

    with ThreadPoolExecutor(WORKERS) as ex:
        stack = np.stack(list(ex.map(member, range(MEMBERS))))  # member, field, lat, lon
    return {n: stack[:, k] for k, n in enumerate(names)}


def write_lead(path: Path, init_time: pd.Timestamp, lead_h: int, fields: dict[str, np.ndarray]) -> Path:
    """One lead's members as netCDF that mverify.io.open reads (init_time and step, member stacked)."""
    import xarray as xr

    ds = xr.Dataset(
        {n: (("member", "latitude", "longitude"), a, {"units": BANDS[n][1]}) for n, a in fields.items()},
        coords={
            "member": np.arange(MEMBERS),
            "latitude": LAT,
            "longitude": LON,
            "init_time": np.datetime64(init_time.to_datetime64(), "ns"),
            "step": np.timedelta64(lead_h, "h").astype("timedelta64[ns]"),
        },
    )
    part = path.with_suffix(".part")
    ds.to_netcdf(part)
    part.rename(path)
    return path


def static(path: Path, init_time: pd.Timestamp) -> Path:
    """ERA5's terrain (m) and land fraction, the static fields WN2 was trained with, averaged onto the common grid by
    mverify's remapping (as it averages every model's), as netCDF of the run at step 0 (mverify opens a file by its
    times)."""
    import xarray as xr
    from mverify.coarsen import Remap
    from mverify.io.open import Grid

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    one = ds[["geopotential_at_surface", "land_sea_mask"]].isel(time=0).load()
    lat, lon = one["latitude"].values.astype(float), one["longitude"].values.astype(float)
    r = Remap(Grid(lat, lon), 0.5)
    rows = np.searchsorted(r.grid.lat, LAT)  # remapped rows run south to north, columns from 0E
    cols = np.searchsorted(r.grid.lon, LON % 360)
    if not (np.allclose(r.grid.lat[rows], LAT) and np.allclose(r.grid.lon[cols], LON % 360)):
        raise RuntimeError("the remapped ERA5 cells are not the common grid's")

    def on_grid(a: np.ndarray) -> np.ndarray:
        return r(a)[np.ix_(rows, cols)].astype(np.float32)

    out = xr.Dataset(
        {
            "orog": (("latitude", "longitude"), on_grid(one["geopotential_at_surface"].values / G0), {"units": "m"}),
            "lsm": (("latitude", "longitude"), on_grid(one["land_sea_mask"].values), {"units": "1"}),
        },
        coords={
            "latitude": LAT,
            "longitude": LON,
            "init_time": np.datetime64(init_time.to_datetime64(), "ns"),
            "step": np.timedelta64(0, "ns"),
        },
    )
    part = path.with_suffix(".part")
    out.to_netcdf(part)
    part.rename(path)
    return path
