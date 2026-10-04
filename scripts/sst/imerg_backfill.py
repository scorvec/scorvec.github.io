#!/usr/bin/env python3
"""Backfill the IMERG daily cache from the GES DISC OPeNDAP endpoint.

The operational path (imerg_precip.ensure_daily) downloads whole global
granules, ~30 MB each — fine for a few days, hopeless for 24 years.  This
pulls the same field through Hyrax with a server-side hyperslab, so each
day costs ~280 KB and ~0.8 s.  Verified bit-identical to the granule path
on an overlapping date (2024-08-01: max |diff| = 0.000000).

Grid, variable and orientation match imerg_precip._read_subset exactly,
so backfilled days drop straight into the existing cache and every
downstream consumer picks them up untouched.

NOTE ON CALIBRATION SPACE: the cache holds RAW IMERG; the gauge
correction field F is applied at read time and gauge_blend_field no-ops
on days with no station file.  Backfilled days are therefore
corrected-satellite, not gauge-blended — the IDEAM daily station archive
only starts in 2024.  That difference is intentional and must be carried
into any analysis that spans the join.

    python scripts/sst/imerg_backfill.py --start 2000-06-01 --end 2024-07-21
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                    # noqa: E402

BASE = ("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/"
        "GPM_3IMERGDE.07/{y}/{m:02d}/{fn}")
_LOCK = threading.Lock()
_DONE = {"n": 0, "fail": 0, "skip": 0}


def month_index(y: int, m: int) -> dict:
    """{date: filename} for one month, from the CMR granule listing."""
    import earthaccess
    d0 = date(y, m, 1)
    d1 = (d0 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    for attempt in range(4):
        try:
            res = earthaccess.search_data(
                short_name="GPM_3IMERGDE", version="07",
                temporal=(d0.isoformat(), d1.isoformat()), count=40)
            out = {}
            for r in res:
                fn = r.data_links()[0].split("/")[-1]
                stamp = fn.split(".3IMERG.")[1][:8]
                out[datetime.strptime(stamp, "%Y%m%d").date()] = fn
            return out
        except Exception:                               # noqa: BLE001
            time.sleep(2 + 3 * attempt)
    return {}


def fetch_day(session, d: date, fn: str, idx) -> bool:
    i0, i1, j0, j1 = idx
    dest = IP.DAILY_CACHE / f"{d:%Y%m%d}.npy"
    if dest.exists():
        with _LOCK:
            _DONE["skip"] += 1
        return True
    url = (BASE.format(y=d.year, m=d.month, fn=fn) +
           f".dap.nc4?dap4.ce=/precipitation%5B0%5D"
           f"%5B{i0}:{i1 - 1}%5D%5B{j0}:{j1 - 1}%5D")
    for attempt in range(3):
        try:
            r = session.get(url, timeout=240)
            if r.status_code != 200 or len(r.content) < 50_000:
                raise OSError(f"status {r.status_code} len {len(r.content)}")
            import xarray as xr
            with tempfile.NamedTemporaryFile(suffix=".nc4", delete=True) as f:
                f.write(r.content)
                f.flush()
                with xr.open_dataset(f.name, engine="h5netcdf") as ds:
                    a = np.squeeze(ds["precipitation"].values)
            g = np.where(a < 0, 0.0, a).T.astype("float32")
            if g.shape != (j1 - j0, i1 - i0):
                raise OSError(f"shape {g.shape}")
            tmp = dest.with_suffix(".tmp.npy")
            np.save(tmp, g)
            tmp.replace(dest)                       # atomic — no torn files
            with _LOCK:
                _DONE["n"] += 1
            return True
        except Exception:                           # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    with _LOCK:
        _DONE["fail"] += 1
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-06-01")
    ap.add_argument("--end", default="2024-07-21")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    d0 = datetime.strptime(a.start, "%Y-%m-%d").date()
    d1 = datetime.strptime(a.end, "%Y-%m-%d").date()

    ml, mt = IP._grid_axes()
    i0, i1 = int(np.argmax(ml)), int(len(ml) - np.argmax(ml[::-1]))
    j0, j1 = int(np.argmax(mt)), int(len(mt) - np.argmax(mt[::-1]))
    if not (ml[i0:i1].all() and mt[j0:j1].all()):
        raise SystemExit("region mask is not contiguous — hyperslab invalid")
    idx = (i0, i1, j0, j1)

    IP._login()
    import earthaccess
    session = earthaccess.get_requests_https_session()
    IP.DAILY_CACHE.mkdir(parents=True, exist_ok=True)

    months = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    print(f"backfill {d0}..{d1} — {len(months)} months, "
          f"{a.workers} workers", flush=True)

    t0 = time.time()
    for k, (y, m) in enumerate(months):
        idxm = month_index(y, m)
        todo = [(d, fn) for d, fn in sorted(idxm.items()) if d0 <= d <= d1
                and not (IP.DAILY_CACHE / f"{d:%Y%m%d}.npy").exists()]
        if todo:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(lambda t: fetch_day(session, t[0], t[1], idx), todo))
        el = time.time() - t0
        done = _DONE["n"] + _DONE["skip"]
        rate = _DONE["n"] / max(el, 1)
        left = (len(months) - k - 1) * 30 / max(rate, 0.1) / 60
        print(f"[{k + 1}/{len(months)}] {y}-{m:02d}  new={_DONE['n']} "
              f"skip={_DONE['skip']} fail={_DONE['fail']}  "
              f"{rate:.1f} d/s  ~{left:.0f} min left", flush=True)
    print(f"done: {_DONE['n']} new, {_DONE['skip']} already cached, "
          f"{_DONE['fail']} failed, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
