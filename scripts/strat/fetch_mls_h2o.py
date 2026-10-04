"""Aura MLS v6 water vapour over the tropics, one profile a day (laptop; also runnable in Actions WITH Earthdata login).

Source: NASA GES DISC, ML2H2O version 006 (MLS-Aura_L2GP-H2O_v06-0x), Level 2 swath files of ~3,500 profiles a day,
2004-08-01 on, ~2 days behind real time. Whole files are downloaded over Earthdata HTTPS (earthaccess, ~/.netrc or
EARTHDATA_USERNAME/_PASSWORD), reduced to a 10S-10N mean profile, and deleted.

Screening, from the MLS v6.0x Data Quality Document section 3.10.8: Status even, Quality > 0.7, Convergence < 2.0,
per-level precision > 0, and no value below 0.101 ppmv at 1 hPa or below. Levels 316-1 hPa are kept. Days with fewer than
100 good tropical profiles are stored with n = 0 (the whole day fails when the 190-GHz radiometer is switched off: since
May 2024 it is duty-cycled to about one week a month, since early 2026 to about four days, and on the other days every
profile carries an odd Status).

One small npz per day under data/tape/mls_days/, so the job resumes and a re-run only fetches what is missing.

    python scripts/strat/fetch_mls_h2o.py                      # 2004-08-01 .. today
    python scripts/strat/fetch_mls_h2o.py 2026-08-01 2026-09-30
"""
import os, sys, time, shutil
from datetime import date, timedelta
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "tape" / "mls_days"
TMP = Path(__file__).resolve().parent / "data" / "tape" / f"mls_tmp_{os.getpid()}"
P_MAX, P_MIN = 316.3, 0.99       # keep 316 .. 1 hPa
LAT = 10.0
MIN_PROFILES = 100


def login():
    import earthaccess
    strat = "environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"
    earthaccess.login(strategy=strat)
    return earthaccess


def reduce_file(path):
    import h5py
    with h5py.File(path, "r") as f:
        g = f["HDFEOS/SWATHS/H2O"]
        p = g["Geolocation Fields/Pressure"][:].astype(float)
        lat = g["Geolocation Fields/Latitude"][:]
        st = g["Data Fields/Status"][:]
        q = g["Data Fields/Quality"][:]
        cv = g["Data Fields/Convergence"][:]
        v = g["Data Fields/L2gpValue"][:].astype(float)
        pr = g["Data Fields/L2gpPrecision"][:]
    if v.ndim != 2 or v.shape[1] != len(p):
        raise ValueError(f"bad shape {v.shape}")
    kk = np.where((p <= P_MAX) & (p >= P_MIN))[0]
    ok = (st % 2 == 0) & (q > 0.7) & (cv < 2.0) & (np.abs(lat) <= LAT)
    low = p >= 1.0
    ok &= ~np.any((v[:, low] < 0.101e-6) & (v[:, low] > -900), axis=1)
    vv = v[:, kk].copy()
    vv[(pr[:, kk] <= 0) | (vv < -900)] = np.nan
    vv = vv[ok]
    n = int(ok.sum())
    if n < MIN_PROFILES:
        return p[kk], np.full(len(kk), np.nan), np.zeros(len(kk), int), n
    return p[kk], np.nanmean(vv, axis=0) * 1e6, np.isfinite(vv).sum(axis=0), n


def day_of(granule):
    b = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
    return date.fromisoformat(b)


def run(d0, d1):
    OUT.mkdir(parents=True, exist_ok=True)
    ea = login()
    y = d0.year
    while y <= d1.year:
        a, b = max(d0, date(y, 1, 1)), min(d1, date(y, 12, 31))
        gr = ea.search_data(short_name="ML2H2O", version="006", temporal=(a.isoformat(), b.isoformat()))
        todo = [g for g in gr if not (OUT / f"{day_of(g):%Y%m%d}.npz").exists() and a <= day_of(g) <= b]
        print(f"{y}: {len(gr)} granules, {len(todo)} to fetch", flush=True)
        for i in range(0, len(todo), 24):
            batch = todo[i:i + 24]
            TMP.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                files = ea.download(batch, str(TMP), threads=4, show_progress=False)
            except Exception as e:
                print(f"  download failed ({str(e)[:80]}); next pass", flush=True)
                time.sleep(30); continue
            byday = {}
            for f in files:
                s = Path(f).name.split("_")[-1].split(".")[0]       # 2025d010
                byday[date(int(s[:4]), 1, 1) + timedelta(int(s[5:]) - 1)] = f
            nd = 0
            for dday, f in byday.items():
                try:
                    p, h, cnt, n = reduce_file(f)
                except Exception as e:
                    print(f"  {dday}: unreadable ({str(e)[:60]}); next pass", flush=True)
                    continue
                np.savez_compressed(OUT / f"{dday:%Y%m%d}.npz", p=p.astype(np.float32), h2o=h.astype(np.float32),
                                    count=cnt.astype(np.int32), n=n)
                nd += 1
            shutil.rmtree(TMP, ignore_errors=True)
            print(f"  {i + len(batch)}/{len(todo)}: {nd} days, {time.time() - t0:.0f}s", flush=True)
        y += 1


if __name__ == "__main__":
    if len(sys.argv) > 2:
        run(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
    else:
        run(date(2004, 8, 1), date.today())
