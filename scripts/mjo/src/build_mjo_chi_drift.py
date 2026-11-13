#!/usr/bin/env python3
"""Build the seed of the MJO chart's drift archive (mjo_chi.py): the AIFS-ENS control's 200 hPa velocity potential,
15°S-15°N, at days 1-15 of the given 00 UTC runs, from the Google mirror. Writes scripts/mjo/data/reference/
mjo_chi_drift_seed.nc (committed), which stands in for the rolling archive on the frames branch until that holds
DRIFT_WINDOW days of runs of its own. Each run's 0-h field is printed against the χ history's analysis of that
time: the two come from one solver, so they should agree to rounding. Run by hand; no workflow runs it.

    python src/build_mjo_chi_drift.py 20260729 20260803 20260808
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).parent))
import mjo_chi
import store as ecmwf
import walker_chi as wc

STEPS = tuple(range(24, 361, 24))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+", help="00 UTC run dates, YYYYMMDD")
    a = ap.parse_args()
    t0 = time.time()
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    hours = np.array(STEPS)
    hist = wc.load_history(wc.SEED)
    runs = []
    for date in sorted(set(a.dates)):
        cyc = ecmwf.Cycle(date, "00")
        try:
            u, v = (
                xr.open_dataset(ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", par, "pl", (200,), (0, *STEPS))), **ku)[
                    par
                ]
                for par in "uv"
            )
        except Exception as ex:  # noqa: BLE001
            print(f"  {cyc.tag}: not available ({str(ex)[:60]})", flush=True)
            continue
        fc = []
        for h in (0, *STEPS):
            step = np.timedelta64(h, "h")
            chi, lat, lon = wc.chi_strip(u.sel(step=step), v.sel(step=step))
            fc.append(mjo_chi.band_mean(chi.astype(float), lat))
        at = pd.Timestamp(date)
        if hist is not None and at in hist.time.values:
            an = mjo_chi.band_mean(hist.sel(level=200, time=at).values.astype(float), lat)
            check = f", 0 h against the history: rms {np.sqrt(((fc[0] - an) ** 2).mean()):.3f}"
        else:
            check = ""
        lon = lon[:-1] if np.isclose(lon[-1] - lon[0], 360.0) else lon
        runs.append(mjo_chi.drift_entry(at, hours, np.stack(fc[1:])[:, : lon.size], lon, 1))
        print(f"  {cyc.tag} ok ({len(runs)}, {time.time() - t0:.0f}s){check}", flush=True)
    if not runs:
        print("no run found: the seed stays")
        return 1
    ds = xr.concat(runs, dim="init")
    mjo_chi.write_drift(mjo_chi.DRIFT_SEED, ds)
    print(
        f"seed: {ds.sizes['init']} runs {str(ds.init.values[0])[:10]} … {str(ds.init.values[-1])[:10]}, "
        f"{mjo_chi.DRIFT_SEED.stat().st_size / 1e3:.0f} kB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
