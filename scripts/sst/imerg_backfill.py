#!/usr/bin/env python3
"""Backfill the IMERG daily cache via OPeNDAP region subsets (GPM IMERG Final).

The nightly pipeline keeps ~recent daily grids from IMERG Early full granules;
the Colombia hydro validation wants a multi-year record. Full-granule downloads
(~10 MB/day) proved fragile on slow links, but build_imerg_clim.py showed the
robust path: server-side OPeNDAP subsets of GPM_3IMERGDF (Final, gauge-
corrected) at ~1 MB/day. This fills every missing date in the trailing --days
window with a Final subset, written in exactly the daily-cache format
(510×510 float32 mm/day, fill→0). Naturally resumable: existing dates skip.

Note: the cache ends up mixed Early (recent, nightly) + Final (backfilled) —
Final is the better product; the seam is harmless for correlation work.

    python scripts/sst/imerg_backfill.py [--days 730] [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
import time
import concurrent.futures as cf
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imerg_precip import DAILY_CACHE, _login
from build_imerg_clim import _opendap_url, _LO0, _LO1, _LA0, _LA1

RETRIES = 3


def _fetch(args) -> tuple[str, bool]:
    url, ymd, sess = args
    out = DAILY_CACHE / f"{ymd}.npy"
    for k in range(RETRIES):
        try:
            ds = xr.open_dataset(url, engine="pydap", session=sess)
            a = ds["precipitation"].sel(lon=slice(_LO0, _LO1),
                                        lat=slice(_LA0, _LA1)).values
            a = np.asarray(a, "float32")
            if a.ndim == 3:
                a = a[0]
            g = np.where(a < 0, 0.0, a).T                  # (lon,lat)→(lat,lon)
            assert g.shape == (510, 510), g.shape
            tmp = out.with_suffix(".tmp.npy")
            np.save(tmp, g)
            tmp.replace(out)
            return ymd, True
        except Exception as e:                             # noqa: BLE001
            if k == RETRIES - 1:
                print(f"  !! {ymd}: {repr(e)[:70]}", flush=True)
            time.sleep(2 * (k + 1))
    return ymd, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                               microsecond=0)
    want = [today - timedelta(days=k) for k in range(4, a.days + 1)]
    need = sorted(d for d in want
                  if not (DAILY_CACHE / f"{d:%Y%m%d}.npy").exists())
    print(f"{len(need)} missing days in trailing {a.days} "
          f"({need[0]:%Y-%m-%d} … {need[-1]:%Y-%m-%d})" if need else
          "nothing missing", flush=True)
    if not need:
        return 0

    import earthaccess
    _login()
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    url_by_day = {}
    # Final first (gauge-corrected), then Early daily for days past Final's
    # production latency — same 0.1° grid and daily mm, so the cache stays uniform
    for short in ("GPM_3IMERGDF", "GPM_3IMERGDE"):
        print(f"searching CMR for {short} granules …", flush=True)
        gs = earthaccess.search_data(
            short_name=short, version="07",
            temporal=(f"{need[0]:%Y-%m-%d}", f"{need[-1]:%Y-%m-%d}"))
        n_new = 0
        for g in gs:
            url = _opendap_url(g)
            if url:
                ymd = url.split(".3IMERG.")[1][:8]
                if ymd not in url_by_day:
                    url_by_day[ymd] = url; n_new += 1
        print(f"  {short}: covers {n_new} more days", flush=True)
    tasks, uncovered = [], 0
    sess = earthaccess.get_requests_https_session()
    for d in need:
        u = url_by_day.get(f"{d:%Y%m%d}")
        if u is None:
            uncovered += 1
            continue
        tasks.append((u, f"{d:%Y%m%d}", sess))
    print(f"{len(tasks)} days fetchable ({uncovered} uncovered)", flush=True)

    t0, ok = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (ymd, good) in enumerate(ex.map(_fetch, tasks), 1):
            ok += good
            if i % 25 == 0 or i == len(tasks):
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(tasks)} ({ok} ok) · {rate:.1f}/s · "
                      f"eta {(len(tasks) - i) / max(rate, 0.01):.0f}s", flush=True)
    print(f"done: {ok}/{len(tasks)} fetched in {time.time() - t0:.0f}s; "
          f"cache now {len(list(DAILY_CACHE.glob('*.npy')))} days", flush=True)
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
