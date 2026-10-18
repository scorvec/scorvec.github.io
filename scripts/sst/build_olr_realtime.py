#!/usr/bin/env python3
"""Real-time GMGSI OLR tail for the wave decomposition — global, daily, rolling window.

The Wheeler-Kiladis filter (olr_waves.py) needs a continuous equatorial OLR record up to
yesterday on the FULL global longitude circle, and no true-OLR product is current (NOAA's
interpolated OLR stops in 2022, ERA5 runs about a week behind). So this maintains it from the
GMGSI longwave-IR mosaic: per image it converts the McIDAS brightness temperature to OLR
(Ohring-Gruber), meridional-means the 15°S-15°N and 5°S-5°N bands and bins them to 2.5°
longitude cells; a day is the mean of its 00, 06, 12 and 18 UTC images.

Every stored day samples the same four phases of the diurnal cycle: a day is written only
once all four images are on the bucket (an image missing at its hour is taken from the next
or the previous hour), each run fills whichever days of the window are still missing, and a
store sampled at other hours is rebuilt. A day averaged over a varying set of images would
carry a varying share of the diurnal cycle over land, a bias no fixed baseline can remove.
The proxy runs a few W m⁻² off true OLR; build_olr_baseline.py measures that offset against
ERA5.

    python scripts/sst/build_olr_realtime.py   # fill the window up to yesterday
"""

from __future__ import annotations

import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "lib"))
from webget import get

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
STORE = HERE / "metar" / "olr_rt.nc"  # committed (rolling state for the daily Action)
S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
SIGMA = 5.67e-8
HOURS = (0, 6, 12, 18)  # images averaged into each daily mean: four phases of the diurnal cycle at every longitude
SAMPLING = ",".join(f"{h:02d}" for h in HOURS)  # stamped on the store; a store sampled otherwise is rebuilt
KEEP_DAYS = 550  # ~1.5 yr — margin for the 120-day low-pass + longer display
BANDS = {"15": 15.0, "05": 5.0}
GRID = np.arange(0, 360, 2.5)  # interp_OLR longitude centres (144 @ 2.5°)
WORKERS = 6  # concurrent days; downloads overlap, the netCDF decode is serialised
_H5LOCK = threading.Lock()  # HDF5/netCDF reads are not thread-safe


def _image(t: datetime) -> bytes | None:
    """The GMGSI LW file of the hour t, else of the next or the previous hour; None if none of the three is posted."""
    for dh in (0, 1, -1):
        pre = f"GMGSI_LW/{t + timedelta(hours=dh):%Y/%m/%d/%H}/"
        keys = re.findall(r"<Key>([^<]+)</Key>", get(f"{S3}/?list-type=2&prefix={pre}&max-keys=5", timeout=30).decode())
        if keys:
            return get(f"{S3}/{keys[0]}", timeout=90)
    return None


def _bands(raw: bytes) -> dict[str, np.ndarray]:
    """Synthetic OLR of one GMGSI LW image, per band, on the 2.5° grid."""
    with _H5LOCK, tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(raw)
        tf.flush()
        with xr.open_dataset(tf.name) as d:
            b = d["data"].isel(time=0).astype("float32").load()
            lat = d["lat"].load()
            lon = d["lon"].load()
    Tb = xr.where(b > 176, 418.0 - b, (660.0 - b) / 2.0)  # McIDAS IR byte → K
    lon1d = lon.isel(yc=lon.sizes["yc"] // 2).values % 360
    idx = (np.round(lon1d / 2.5).astype(int)) % len(GRID)
    out = {}
    for bd, half in BANDS.items():
        band = Tb.where((lat >= -half) & (lat <= half)).mean("yc")
        Tf = band * (1.228 - 1.106e-3 * band)
        olr = (SIGMA * Tf**4).values
        good = np.isfinite(olr)
        sums = np.bincount(idx[good], weights=olr[good], minlength=len(GRID))
        cnts = np.bincount(idx[good], minlength=len(GRID))
        out[bd] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return out


def _gmgsi_day(dd: date) -> dict[str, np.ndarray] | None:
    """Daily-mean synthetic OLR(lon) on the 2.5° grid for each band, or None until every HOURS image is posted."""
    per = []
    for h in HOURS:
        try:
            raw = _image(datetime(dd.year, dd.month, dd.day, h, tzinfo=UTC))
            if raw is None:
                return None
            per.append(_bands(raw))
        except Exception as e:  # noqa: BLE001  (a failed request or unreadable file leaves the day to the next run)
            print(f"  {dd} {h:02d}Z: {repr(e)[:90]}", flush=True)
            return None
    return {bd: np.nanmean([p[bd] for p in per], axis=0) for bd in BANDS}


def load() -> xr.Dataset | None:
    if STORE.exists():
        return xr.open_dataset(STORE)
    return None


def _write(rows: dict, today: date) -> xr.Dataset:
    """Atomically (re)write the rolling store from the accumulated rows."""
    cutoff = today - timedelta(days=KEEP_DAYS)
    times = sorted(t for t in rows["15"] if t > cutoff)
    out = xr.Dataset(
        {f"olr_{b}": (("time", "lon"), np.array([rows[b][t] for t in times], dtype="float32")) for b in BANDS},
        coords={"time": [np.datetime64(t, "D") for t in times], "lon": GRID},
        attrs={
            "source": "NOAA GMGSI longwave-IR mosaic, McIDAS brightness temperature to OLR (Ohring-Gruber)",
            "sampling": SAMPLING,
            "note": "daily mean of the images at these UTC hours; bands = |lat| <= 5 and 15 deg; W m-2",
        },
    )
    tmp = STORE.with_suffix(".nc.tmp")
    out.to_netcdf(tmp)
    tmp.replace(STORE)  # atomic: never lose the prior store
    return out


def main() -> int:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    rows = {b: {} for b in BANDS}
    ds = load()
    if ds is not None:
        if ds.attrs.get("sampling") == SAMPLING:  # carry the stored days
            for b in BANDS:
                for t, v in zip(pd.to_datetime(ds["time"].values).date, ds[f"olr_{b}"].values):
                    rows[b][t] = v
        else:
            print(f"store sampled at {ds.attrs.get('sampling', 'unrecorded')} UTC, not {SAMPLING}: rebuilding")
        ds.close()
    want = [today - timedelta(days=k) for k in range(KEEP_DAYS - 1, 0, -1)]
    want = [d for d in want if d not in rows["15"]]
    print(f"{len(want)} day(s) to fill", flush=True)

    t0 = time.time()
    n_new = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_gmgsi_day, dd): dd for dd in want}
        for fut in as_completed(futs):
            dd, r = futs[fut], fut.result()
            if r is None:
                print(f"  {dd}: incomplete, left for a later run", flush=True)
                continue
            for b in BANDS:
                rows[b][dd] = r[b]
            n_new += 1
            if n_new % 50 == 0:  # checkpoint so a crash can't undo progress
                _write(rows, today)
                print(f"  …{n_new} new days ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if not rows["15"]:
        raise SystemExit("no complete GMGSI day")
    out = _write(rows, today)
    print(
        f"wrote {STORE.name}: {out.sizes['time']} days × {out.sizes['lon']} lon "
        f"({str(out['time'].values[0])[:10]}→{str(out['time'].values[-1])[:10]}), {n_new} new, sampled {SAMPLING} UTC",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
