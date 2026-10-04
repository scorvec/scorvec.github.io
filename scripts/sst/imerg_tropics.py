#!/usr/bin/env python3
"""IMERG V07 Late daily rainfall over the tropics (30°S–30°N), on a 1° grid — the observed-rain input of
the Gill rain-forcing product (gill_rain.py) and of its calibration (build_gill_rain_ref.py).

Why IMERG Late: it is the only global satellite-gauge-calibrated rainfall with a latency of about one
day (the daily Late file for day D is on GES DISC by ~14 UTC on D+1); CPC's gauge analysis has no ocean,
CMORPH/GPCP daily lag weeks to months. The V07 Late record is reprocessed back to June 2000, so the
climatology can come from the same run as the live data (a matched base, not a GPCP or Final mean).

Access: GES DISC OPeNDAP, one request per day for `precipitation` over 30°S–30°N at the full 0.1°
(~3 MB compressed, ~3 s), then 1° box means (a cell needs ≥ 50 % valid 0.1° points). Earthdata login
from ~/.netrc on the laptop or EARTHDATA_USERNAME / EARTHDATA_PASSWORD in Actions (earthaccess).

    python scripts/sst/imerg_tropics.py fetch 2001-01-01 2025-12-31 --out DIR --procs 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

LAT1 = np.arange(-29.5, 30.0, 1.0)            # 60 rows, cell centres
LON1 = np.arange(0.5, 360.0, 1.0)             # 360 columns, 0–360 °E
SHORT = "GPM_3IMERGDL"
DATA = "https://data.gesdisc.earthdata.nasa.gov/data/"
OPENDAP = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/"
QUERY = "precipitation[0:0][0:1:3599][600:1:1199]"      # all longitudes, lat index 600..1199 = 29.95°S..29.95°N

_SESSION = None
_AUTH = False


def _login():
    global _AUTH
    import earthaccess
    if not _AUTH:
        strategy = "environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"
        earthaccess.login(strategy=strategy)
        _AUTH = True
    return earthaccess


def session():
    global _SESSION
    if _SESSION is None:
        ea = _login()
        _SESSION = ea.get_requests_https_session()
    return _SESSION


def granule_urls(d0: date, d1: date) -> dict:
    """{date: opendap base url} from a CMR search (month-sized chunks, retried)."""
    ea = _login()
    out = {}
    cur = d0
    while cur <= d1:
        nxt = min(d1, (cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1))
        for attempt in range(5):
            try:
                res = ea.search_data(short_name=SHORT, version="07", temporal=(cur.isoformat(), nxt.isoformat()))
                break
            except Exception as ex:                                  # noqa: BLE001 — CMR 500s under load
                if attempt == 4:
                    print(f"  CMR search {cur}..{nxt} failed: {repr(ex)[:80]}", flush=True); res = []
                time.sleep(3 * (attempt + 1))
        for r in res:
            for link in r.data_links():
                if link.endswith(".nc4") and "3B-DAY-L" in link:
                    ds = link.split(".3IMERG.")[1][:8]
                    out[datetime.strptime(ds, "%Y%m%d").date()] = link.replace(DATA, OPENDAP)
        cur = nxt + timedelta(days=1)
    return out


def to_1deg(p: np.ndarray) -> np.ndarray:
    """(lon 3600, lat 600) 0.1° mm/day → (60, 360) 1° box means on LAT1 × LON1 (NaN where < 50 % valid)."""
    p = np.where(np.isfinite(p) & (p >= 0), p, np.nan).T                       # (600, 3600)
    blk = p.reshape(60, 10, 360, 10)
    n = np.isfinite(blk).sum((1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(blk, axis=(1, 3))
    m[n < 50] = np.nan
    return np.roll(m, 180, axis=1).astype("float32")                          # −179.5…179.5 → 0.5…359.5


def fetch_day(url: str, tries: int = 4) -> np.ndarray | None:
    """One day, validated: tropical mean 1.5–6 mm/day, ≥ 95 % valid cells, max < 1000 mm/day."""
    import netCDF4                          # not h5py: two HDF5 libraries in one process (h5py's + netCDF4's wheels)
    s = session()                           # abort with "double free or corruption" at exit on the Actions runner
    for k in range(tries):
        try:
            r = s.get(url + ".nc4?" + QUERY, timeout=120)
            r.raise_for_status()
            with netCDF4.Dataset("imerg.nc4", mode="r", memory=r.content) as f:
                v = f.variables["precipitation"]; v.set_auto_maskandscale(False)
                a = np.asarray(v[0]).astype("float32")                            # (lon, lat)
            if a.shape != (3600, 600):
                raise ValueError(f"shape {a.shape}")
            g = to_1deg(a)
            fin = np.isfinite(g).mean(); mu = float(np.nanmean(g)); mx = float(np.nanmax(g))
            if fin < 0.95 or not (1.5 < mu < 6.0) or mx > 1000:
                raise ValueError(f"implausible: valid {fin:.2f} mean {mu:.2f} max {mx:.0f}")
            return g
        except Exception as ex:                                              # noqa: BLE001
            if k == tries - 1:
                print(f"  {url.rsplit('/', 1)[-1][23:31]} failed: {repr(ex)[:90]}", flush=True)
                return None
            time.sleep(4 * (k + 1))
    return None


def _worker(args):
    items, out = args
    out = Path(out); n = 0
    for d, url in items:
        f = out / f"{d:%Y}" / f"{d:%Y%m%d}.npy"
        if f.exists():
            continue
        g = fetch_day(url)
        if g is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, g); n += 1
    return n


def fetch_range(d0: date, d1: date, out: Path, procs: int = 1) -> int:
    urls = granule_urls(d0, d1)
    todo = sorted((d, u) for d, u in urls.items() if not (out / f"{d:%Y}" / f"{d:%Y%m%d}.npy").exists())
    print(f"  IMERG Late {d0}..{d1}: {len(urls)} granules, {len(todo)} to fetch", flush=True)
    if not todo:
        return 0
    if procs <= 1:
        return _worker((todo, str(out)))
    from multiprocessing import get_context
    chunks = [todo[i::procs] for i in range(procs)]
    with get_context("spawn").Pool(procs) as pool:
        return sum(pool.map(_worker, [(c, str(out)) for c in chunks]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("d0"); ap.add_argument("d1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    d0 = date.fromisoformat(a.d0); d1 = date.fromisoformat(a.d1)
    t0 = time.time()
    # year by year so progress is visible and a restart resumes cheaply
    y = d0.year; tot = 0
    while y <= d1.year:
        s = max(d0, date(y, 1, 1)); e = min(d1, date(y, 12, 31))
        n = fetch_range(s, e, Path(a.out), a.procs); tot += n
        print(f"{y}: {n} days fetched ({(time.time() - t0) / 60:.1f} min)", flush=True)
        y += 1
    print(f"DONE {tot} days", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
