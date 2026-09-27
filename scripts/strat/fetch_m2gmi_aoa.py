"""Age of stratospheric air from the MERRA-2 GMI replay: monthly zonal means, 1980-2019 (laptop, run once).

Source: NASA NCCS OPeNDAP, merra2_gmi/tavgM_3d_dac_Np (GMI chemistry replayed in MERRA-2 meteorology, monthly means
of the daily-average chemical fields, 44 pressure levels to 0.1 hPa, 0.5 x 0.625 deg), variable `aoadays` (the GMI
age-of-air clock tracer, in days). Only the levels from 250 hPa up are read, every 4th latitude (2 deg) and every 8th
longitude (5 deg) - the tracer is smooth in longitude in the stratosphere and a 72-point zonal mean of it differs from
the full 576-point mean by well under 0.01 yr (checked on 1980-01). ~6 s a month, one request per year.

SEQUENTIAL on purpose (NCCS, like every OPeNDAP server, returns partial reads under concurrency; see the PSL note), one
npz per year so the job resumes, and every year is validated before it is saved: 12 finite months, zonal means inside
0-8 years, no all-zero or constant level. A year that fails is not written and is retried on the next pass.

    python scripts/strat/fetch_m2gmi_aoa.py            # all missing years
    python scripts/strat/fetch_m2gmi_aoa.py 1980 1985  # a range
"""
import sys, time
from pathlib import Path
import numpy as np, xarray as xr

URL = "https://opendap.nccs.nasa.gov/dods/merra2_gmi/tavgM_3d_dac_Np"
OUT = Path(__file__).resolve().parent / "data" / "aoa"
K0 = 21            # lev index of 250 hPa (lev runs 1000 -> 0.1 hPa, 44 levels)
LAT_STEP, LON_STEP = 4, 8


def open_ds():
    for i in range(5):
        try:
            return xr.open_dataset(URL, engine="netcdf4")
        except Exception as e:          # NCCS drops the odd connection ("SSL connect error"); back off and retry
            print(f"open failed ({str(e)[:60]}), retry {i + 1}", flush=True)
            time.sleep(20 * (i + 1))
    raise RuntimeError("could not open " + URL)


def valid(zm):
    if not np.isfinite(zm).all():
        return "non-finite values"
    yrs = zm / 365.25
    if yrs.min() < -0.05 or yrs.max() > 8.0:
        return f"out of range {yrs.min():.2f}..{yrs.max():.2f}"
    for m in range(zm.shape[0]):
        for k in range(zm.shape[1]):
            if np.ptp(zm[m, k]) == 0:
                return f"constant level (month {m}, lev {k})"
    return None


def year(ds, y):
    f = OUT / f"aoa_{y}.npz"
    if f.exists():
        return f"{y}: have"
    t = ds["time"].values
    idx = np.where((t >= np.datetime64(f"{y}-01-01")) & (t < np.datetime64(f"{y + 1}-01-01")))[0]
    if len(idx) != 12:
        return f"{y}: {len(idx)} months on the server"
    for attempt in range(3):
        try:
            t0 = time.time()
            a = ds["aoadays"][idx[0]:idx[-1] + 1, K0:, ::LAT_STEP, ::LON_STEP].values.astype(np.float64)
            a[a > 1e14] = np.nan
            zm = np.nanmean(a, axis=-1)          # (12, lev, lat)
            bad = valid(zm)
            if bad:
                raise ValueError(bad)
            np.savez_compressed(f, aoa_days=zm.astype(np.float32), time=t[idx].astype("datetime64[D]"),
                                lev=ds["lev"].values[K0:], lat=ds["lat"].values[::LAT_STEP],
                                nlon=a.shape[-1], nan_frac=np.float32(np.isnan(a).mean()))
            return f"{y}: ok {time.time() - t0:.0f}s"
        except Exception as e:
            print(f"{y}: attempt {attempt + 1} failed: {str(e)[:100]}", flush=True)
            time.sleep(30)
    return f"{y}: FAILED, retry later"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1980, 2019)
    ds = open_ds()
    for y in range(y0, y1 + 1):
        print(year(ds, y), flush=True)


if __name__ == "__main__":
    main()
