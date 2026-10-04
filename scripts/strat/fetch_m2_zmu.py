#!/usr/bin/env python3
"""MERRA-2 daily zonal-mean u, 50S-50N, 100-10 hPa, 1980 -> latest (M2I3NPASM inst3_3d_asm_Np), for the QBO zero-wind-line
product (qbo_zeroline.py). LAPTOP job, resumable per year.

Route: Earthdata Cloud OPeNDAP (Hyrax, opendap.earthdata.nasa.gov) with a DAP4 constraint, so only U at 7 levels
(100 70 50 40 30 20 10 hPa), 201 latitudes (50S-50N, 0.5 deg), every 4th longitude (144 points, 2.5 deg) and the four
synoptic times 00/06/12/18Z come down: ~2.1 MB a day, ~3 s a request instead of the 1.1 GB granule. The GES DISC on-prem OPeNDAP that
fetched m2_strat is retired (410). Login from ~/.netrc via earthaccess (never printed).
Daily value = mean of the four synoptic times, then the zonal mean over the 144 sampled longitudes.
Output: scripts/strat/data/m2_zmu/zmu_<YYYY>.npz (time 'YYYY-MM-DD', lev, lat, u[time, lev, lat] float32)."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, tempfile, os
import earthaccess

OUT = Path(__file__).resolve().parent / "data" / "m2_zmu"; OUT.mkdir(parents=True, exist_ok=True)
Y0, Y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (1980, pd.Timestamp.today().year)
CE = "/U[0:2:7][24:1:30][80:1:280][0:4:575];/lat[80:1:280];/lev[24:1:30];/time[0:2:7]"   # /time must carry the same stride
earthaccess.login(strategy="netrc")
S = earthaccess.get_requests_https_session()


BASE = "https://opendap.earthdata.nasa.gov/collections/C1276812879-GES_DISC/granules/M2I3NPASM.5.12.4%3AMERRA2_{s}.inst3_3d_asm_Np.{d}.nc4"


def streams(day):
    """MERRA-2 production streams: 100 to 1991, 200 to 2000, 300 to 2010, 400 after; a handful of months were
    re-run as 401 (Sep 2020, Jun-Sep 2021), so 401 is tried whenever the primary stream fails. Built directly
    instead of a CMR search: the search took ~6 min a year against 1.4 min of fetching."""
    y = day.year
    s = 100 if y <= 1991 else 200 if y <= 2000 else 300 if y <= 2010 else 400
    return [s] + ([401] if s == 400 else [])


def day(dt):
    urls = [BASE.format(s=s, d=dt.strftime("%Y%m%d")) for s in streams(dt)]
    url = urls[0]
    err = None
    for attempt in range(5):
        try:
            r = None
            for url in urls:
                r = S.get(url + ".dap.nc4", params={"dap4.ce": CE}, timeout=180)
                if r.status_code == 200:
                    break
            r.raise_for_status()
            fd, tmp = tempfile.mkstemp(suffix=".nc4"); os.write(fd, r.content); os.close(fd)
            try:
                d = xr.open_dataset(tmp)
                u = d["U"].values.astype("float64")
                u = np.where(np.abs(u) > 1e10, np.nan, u)
                rec = dict(time=pd.Timestamp(d.time.values[0]).strftime("%Y-%m-%d"),
                           u=np.nanmean(np.nanmean(u, axis=0), axis=-1).astype("float32"),
                           lev=d.lev.values, lat=d.lat.values)
                d.close()
            finally:
                os.remove(tmp)
            return rec
        except Exception as e:                                                  # noqa: BLE001
            err = e; time.sleep(10 * (attempt + 1))
    print(f"  FAILED {dt.date()}: {str(err)[:120]}", flush=True)
    return None


for y in range(Y0, Y1 + 1):
    dest = OUT / f"zmu_{y}.npz"
    if dest.exists():
        prev = np.load(dest)
        full = len(prev["time"]) >= (366 if y % 4 == 0 else 365)
        if full:
            continue
    t0 = time.time()
    last = min(pd.Timestamp(y, 12, 31), pd.Timestamp.today().normalize() - pd.Timedelta(days=20))
    days = pd.date_range(f"{y}-01-01", last)
    with ThreadPoolExecutor(12) as ex:
        recs = sorted([r for r in ex.map(day, days) if r], key=lambda r: r["time"])
    by = days
    if not recs:
        print(f"{y}: nothing", flush=True); continue
    np.savez_compressed(dest, time=np.array([r["time"] for r in recs]), lev=recs[0]["lev"], lat=recs[0]["lat"],
                        u=np.stack([r["u"] for r in recs]))
    print(f"{y}: {len(recs)}/{len(by)} days in {(time.time() - t0) / 60:.1f} min (last {recs[-1]['time']})", flush=True)
print("done", flush=True)
