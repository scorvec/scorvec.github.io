"""SFS beta2 reforecast weekly means for the CFS-study years: reuse the 2001-2019 cache, fetch the rest chunk by chunk
(the reforecast zarr only - never the daily forecast store or the *_clim stores)."""

import sys
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import YEARS, MONTHS, OUT, OLD

sys.path.insert(0, OLD)
from sfs_fetch import one, VARS  # (one chunk = one member x one init, all leads)

OLDY = list(range(2001, 2020))


def main():
    S = np.load(OLD + "sfs_weekly.npz")
    res = {t: np.full((2, len(YEARS), VARS[t][1], 4, 61, 240), np.nan, np.float32) for t in VARS}
    need = []
    for yi, y in enumerate(YEARS):
        if y in OLDY:
            for t in VARS:
                res[t][:, yi] = S[t][:, OLDY.index(y)]
        else:
            need += [(t, m, y, mem) for t, (_, nm, _) in VARS.items() for m in MONTHS for mem in range(nm)]
    print(len(need), "chunks to fetch", flush=True)
    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(10) as ex:
        for k, (job, w, err) in enumerate(ex.map(one, need)):
            t, m, y, mem = job
            if w is None:
                fails.append((job, err))
                continue
            res[t][MONTHS.index(m), YEARS.index(y), mem] = w
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(need)}, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT + "sfs_weekly2.npz", years=np.array(YEARS), **res)
    print(f"done in {time.time() - t0:.0f} s; failures {fails}", flush=True)


if __name__ == "__main__":
    main()
