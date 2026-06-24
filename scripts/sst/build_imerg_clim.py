#!/usr/bin/env python3
"""Day-of-year IMERG precip climatology (harmonic) for the precip-anomaly products.

Reanalysis-free: the climatology is built from IMERG's OWN record — 20 years (2001–2020) of
GPM IMERG **Final** daily (GPM_3IMERGDF V07, gauge-corrected). Region subsets are pulled
server-side via OPeNDAP (~3.6 GB, not the 73 GB of full global granules), accumulated into a
day-of-year mean, then fit per grid cell with a small harmonic (mean + NHARM annual
harmonics — enough for Colombia's bimodal wet seasons). The compact coefficients
(data/reference/imerg_precip_clim.nc, ~9 MB, committed) let the anomaly products evaluate
the climatological accumulation for any window/day-of-year without re-touching the archive.

Resumable: the day-of-year accumulator is checkpointed after each year (data/imerg/
clim_accum.npz); re-running resumes from the last completed year.

    python scripts/sst/build_imerg_clim.py            # full 2001–2020 build (multi-hour)
    python scripts/sst/build_imerg_clim.py --years 2019:2020   # a short test range
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imerg_precip import CFG, _login                       # share the region + Earthdata auth

import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ACCUM = HERE / "data" / "imerg" / "clim_accum.npz"         # resumable checkpoint (gitignored)
OUT = HERE / "imerg_precip_clim.nc"                        # harmonic coeffs (committed, like the other SST clim files)
BASE_YEARS = range(2001, 2026)                             # 25-year climatology baseline (2001–2025)
NHARM = 4                                                  # annual harmonics (annual+semi+...)
WORKERS = 12
# region edges in −180..180 for the OPeNDAP .sel()
_LO0 = CFG["extent"][0] - 360 if CFG["extent"][0] > 180 else CFG["extent"][0]
_LO1 = CFG["extent"][1] - 360 if CFG["extent"][1] > 180 else CFG["extent"][1]
_LA0, _LA1 = CFG["extent"][2], CFG["extent"][3]


def _opendap_url(g) -> str | None:
    for u in g["umm"].get("RelatedUrls", []):
        url = u.get("URL", "")
        if "opendap" in url.lower() and url.endswith(".nc4"):
            return "dap4://" + url.split("://", 1)[1]
    return None


def _fetch_day(args):
    """OPeNDAP region-subset (lat, lon) mm/day for one granule, with its day-of-year. None on error."""
    url, doy, sess = args
    try:
        ds = xr.open_dataset(url, engine="pydap", session=sess)
        a = ds["precipitation"].sel(lon=slice(_LO0, _LO1), lat=slice(_LA0, _LA1)).values
        a = np.asarray(a, "float32")
        if a.ndim == 3:
            a = a[0]
        return doy, np.where(a < 0, 0.0, a).T          # (lon,lat)→(lat,lon)
    except Exception:                                  # noqa: BLE001 — transient OPeNDAP/network
        return doy, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="override, e.g. 2019:2020")
    args = ap.parse_args(argv)
    years = BASE_YEARS
    if args.years:
        a, b = map(int, args.years.split(":")); years = range(a, b + 1)

    import earthaccess
    _login()

    # resume from checkpoint
    if ACCUM.exists():
        z = np.load(ACCUM, allow_pickle=True)
        doy_sum = z["doy_sum"]; doy_cnt = z["doy_cnt"]; done = set(int(y) for y in z["done"])
        print(f"resuming: years done = {sorted(done)}")
    else:
        ny = abs(_LA1 - _LA0); nx = abs(_LO1 - _LO0)        # placeholder; sized on first day
        doy_sum = None; doy_cnt = np.zeros(367, "int32"); done = set()

    for yr in years:
        if yr in done:
            continue
        sess = earthaccess.get_requests_https_session()     # fresh session each year (token refresh)
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
                if doy_sum is None:
                    doy_sum = np.zeros((367,) + grid.shape, "float32")
                doy_sum[doy] += grid; doy_cnt[doy] += 1; ok += 1
        done.add(yr)
        ACCUM.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ACCUM, doy_sum=doy_sum, doy_cnt=doy_cnt, done=np.array(sorted(done)))
        print(f"  {yr}: {ok}/{len(tasks)} days in {time.time()-t0:.0f}s  (checkpointed)", flush=True)

    # day-of-year mean → per-cell harmonic fit
    valid = doy_cnt[1:366] > 0
    doys = np.arange(1, 366)[valid]
    clim = (doy_sum[1:366][valid] / doy_cnt[1:366][valid, None, None]).astype("float32")  # (nd, lat, lon)
    nd, nlat, nlon = clim.shape
    X = np.ones((nd, 1 + 2 * NHARM))
    for k in range(1, NHARM + 1):
        X[:, 2 * k - 1] = np.cos(2 * np.pi * k * doys / 365.25)
        X[:, 2 * k] = np.sin(2 * np.pi * k * doys / 365.25)
    coef, *_ = np.linalg.lstsq(X, clim.reshape(nd, -1), rcond=None)   # (ncoef, ncells)
    coef = coef.reshape(-1, nlat, nlon).astype("float32")

    ml, mt = _grid_lonlat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"coef": (("ncoef", "lat", "lon"), coef)},
        coords={"lat": mt, "lon": ml},
        attrs={"nharm": NHARM, "years": f"{min(done)}-{max(done)}",
               "source": "GPM IMERG Final daily V07 (GPM_3IMERGDF)",
               "note": "harmonic day-of-year mean daily precip (mm/day); eval with build_imerg_clim.eval_clim"},
    ).to_netcdf(OUT, encoding={"coef": {"zlib": True, "complevel": 5}})
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {NHARM} harmonics, years {min(done)}–{max(done)})")
    return 0


def _grid_lonlat():
    """Region lon/lat axes matching the climatology grid (−180..180 lon)."""
    lon = -179.95 + 0.1 * np.arange(3600)
    lat = -89.95 + 0.1 * np.arange(1800)
    return lon[(lon >= _LO0) & (lon <= _LO1)], lat[(lat >= _LA0) & (lat <= _LA1)]


def eval_clim(coef: np.ndarray, doy: int) -> np.ndarray:
    """Climatological mean daily precip (mm/day) for a day-of-year from harmonic coeffs."""
    nharm = (coef.shape[0] - 1) // 2
    out = coef[0].copy()
    for k in range(1, nharm + 1):
        out += coef[2 * k - 1] * np.cos(2 * np.pi * k * doy / 365.25)
        out += coef[2 * k] * np.sin(2 * np.pi * k * doy / 365.25)
    return np.maximum(out, 0.0)                            # precip can't be negative


if __name__ == "__main__":
    raise SystemExit(main())
