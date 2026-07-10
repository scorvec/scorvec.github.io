#!/usr/bin/env python3
"""GLOBAL day-of-year IMERG precip climatology at 0.5° for the climate monitor.

Same recipe as scripts/sst/build_imerg_clim.py (IMERG Final daily 2001–2025,
harmonic fit per cell) but global, made tractable by server-side OPeNDAP
STRIDING: every 5th 0.1° cell → 720×360 (~1/25 of the bytes; the granule set
stays the ~73 GB archive but we move only ~3 GB total). Resumable per year.

Output: imerg_clim_global.nc (720×360×9 coefficients, ~9 MB, committed).

    python scripts/climate/build_imerg_clim_global.py
    python scripts/climate/build_imerg_clim_global.py --years 2024:2025   # test
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sst"))
from imerg_precip import _login                            # shared Earthdata auth

ACCUM = HERE / "data" / "imerg_global_accum.npz"           # resumable checkpoint (gitignored)
OUT = HERE / "imerg_clim_global.nc"                        # harmonic coeffs (committed)
BASE_YEARS = range(2001, 2026)
NHARM = 4
WORKERS = 12
STRIDE = 5                                                 # 0.1° → 0.5°
# strided axes of the fixed IMERG grid (lon −180..180, lat −90..90)
LON = (-179.95 + 0.1 * np.arange(3600))[::STRIDE]          # 720
LAT = (-89.95 + 0.1 * np.arange(1800))[::STRIDE]           # 360


def _opendap_url(g) -> str | None:
    for u in g["umm"].get("RelatedUrls", []):
        url = u.get("URL", "")
        if "opendap" in url.lower() and url.endswith(".nc4"):
            return "dap4://" + url.split("://", 1)[1]
    return None


def _fetch_day(args):
    """Server-side strided global grid (lat, lon) mm/day for one granule."""
    url, doy, sess = args
    try:
        ds = xr.open_dataset(url, engine="pydap", session=sess)
        a = ds["precipitation"].isel(lon=slice(0, 3600, STRIDE),
                                     lat=slice(0, 1800, STRIDE)).values
        a = np.asarray(a, "float32")
        if a.ndim == 3:
            a = a[0]
        return doy, np.where(a < 0, 0.0, a).T              # (lon,lat)→(lat,lon)
    except Exception:                                      # noqa: BLE001 — transient OPeNDAP
        return doy, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="override, e.g. 2024:2025")
    args = ap.parse_args(argv)
    years = BASE_YEARS
    if args.years:
        a, b = map(int, args.years.split(":")); years = range(a, b + 1)

    import earthaccess
    _login()

    if ACCUM.exists():
        z = np.load(ACCUM, allow_pickle=True)
        doy_sum = z["doy_sum"]; doy_cnt = z["doy_cnt"]; done = set(int(y) for y in z["done"])
        print(f"resuming: years done = {sorted(done)}", flush=True)
    else:
        doy_sum = np.zeros((367, len(LAT), len(LON)), "float32")
        doy_cnt = np.zeros(367, "int32"); done = set()

    for yr in years:
        if yr in done:
            continue
        sess = earthaccess.get_requests_https_session()
        gs = earthaccess.search_data(short_name="GPM_3IMERGDF", version="07",
                                     temporal=(f"{yr}-01-01", f"{yr}-12-31"))
        tasks = []
        for g in gs:
            url = _opendap_url(g)
            if not url:
                continue
            ymd = url.split(".3IMERG.")[1][:8]
            doy = int(time.strptime(ymd, "%Y%m%d").tm_yday)
            tasks.append((url, doy, sess))
        t0 = time.time(); ok = 0
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for doy, grid in ex.map(_fetch_day, tasks):
                if grid is None:
                    continue
                doy_sum[doy] += grid; doy_cnt[doy] += 1; ok += 1
        done.add(yr)
        ACCUM.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ACCUM, doy_sum=doy_sum, doy_cnt=doy_cnt,
                            done=np.array(sorted(done)))
        print(f"  {yr}: {ok}/{len(tasks)} days in {time.time()-t0:.0f}s (checkpointed)", flush=True)

    # daily-mean normals (mm/day) → harmonic coefficients
    with np.errstate(invalid="ignore"):
        clim = doy_sum[1:366] / np.maximum(doy_cnt[1:366, None, None], 1)
    x = 2 * np.pi * np.arange(365) / 365.0
    cols = [np.ones(365)]
    for h in range(1, NHARM + 1):
        cols += [np.cos(h * x), np.sin(h * x)]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, clim.reshape(365, -1), rcond=None)
    coef = beta.reshape((A.shape[1],) + clim.shape[1:]).astype("float32")
    xr.Dataset(
        {"coef": (("coef_idx", "lat", "lon"), coef)},
        coords={"lat": LAT, "lon": LON},
        attrs={"source": "GPM IMERG Final daily V07, OPeNDAP stride 5 (0.5deg)",
               "base": f"{min(BASE_YEARS)}-{max(BASE_YEARS)}", "nharm": NHARM,
               "units": "mm/day"},
    ).to_netcdf(OUT)
    print(f"wrote {OUT.name} ({OUT.stat().st_size // 1024} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
