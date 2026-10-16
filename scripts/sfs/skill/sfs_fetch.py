"""SFS beta2 reforecast -> weekly means on the ERA5 NH 1.5 deg grid, chunk by chunk (one member x one init x all leads)."""

import sys
import time
import urllib.request
import numpy as np
import numcodecs
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import YEARS, MONTHS, OUT, weekly, regrid

B = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta2/reforecast/{m:02d}/atm_daily.zarr"
VARS = {"t2m": ("TMP_2maboveground", 6, 1.0), "z500": ("HGT_500mb", 6, 1.0), "pr": ("PRATE_surface", 4, 86400.0)}
LAT = np.arange(-90.0, 90.0 + 1e-6, 1.0)
LON = np.arange(0.0, 360.0 - 1e-6, 1.0)
BLOSC = numcodecs.Blosc()


def one(job):
    tag, m, y, mem = job
    name, _, fac = VARS[tag]
    url = f"{B.format(m=m)}/{name}/{mem}.{y - 1991}.0.0.0"
    for k in range(5):
        try:
            raw = urllib.request.urlopen(url, timeout=180).read()
            break
        except Exception as e:  # noqa: BLE001
            if k == 4:
                return job, None, str(e)
            time.sleep(3 * (k + 1))
    a = np.frombuffer(BLOSC.decode(raw), "<f4").reshape(47, 181, 360).astype(np.float32)
    a = np.where(a > 1e19, np.nan, a) * fac
    w = weekly(a[:35])  # leads 0..34 = days 1..35
    return job, regrid(w, LAT, LON), None


def main():
    jobs = [(t, m, y, mem) for t, (_, nm, _) in VARS.items() for m in MONTHS for y in YEARS for mem in range(nm)]
    res = {t: np.full((2, len(YEARS), VARS[t][1], 4, 61, 240), np.nan, np.float32) for t in VARS}
    t0 = time.time()
    done = 0
    fails = []
    with ThreadPoolExecutor(10) as ex:
        for job, w, err in ex.map(one, jobs):
            t, m, y, mem = job
            if w is None:
                fails.append((job, err))
                continue
            res[t][MONTHS.index(m), YEARS.index(y), mem] = w
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT + "sfs_weekly.npz", **res)
    print(f"done {done}/{len(jobs)} in {time.time() - t0:.0f} s; failures {len(fails)} {fails[:3]}", flush=True)


if __name__ == "__main__":
    main()
