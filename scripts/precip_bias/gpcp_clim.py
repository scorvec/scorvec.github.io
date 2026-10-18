#!/usr/bin/env python3
"""Weekly rainfall climatology for the tropical rainfall forecast.

GPCP V3.2 daily (GES DISC GPCPDAY.3.2, 0.5°, 2000-06 to 2021-09; IMERG Final-based inside 60°N-60°S, which is why it is
the climatology for a forecast bias-adjusted to IMERG Final), 2001-2020, as 2 x 2 means on the global 1° grid of grid.py.
For 73 pentads (start day of year 1, 6, ..., 361) the 7-day totals STARTING within +-10 days of the pentad's start, in
every year (21 x 20 = 420 overlapping samples per cell): mean, terciles and median, mm per week. Feb 29 is dropped.

    python scripts/precip_bias/gpcp_clim.py --out DIR [--threads 6]
-> DIR/gpcp_clim_week_global.npz: start_doy (73), mean / q33 / q50 / q67 (73, NY, NX) float32 mm/week, lat, lon, years, missing
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G
import imerg as IM  # (Earthdata session, IPv4-only login)

URL = "https://measures.gesdisc.eosdis.nasa.gov/data/GPCP/GPCPDAY.3.2/{y}/GPCPDAY_L3_{d}_V3.2.nc4"
Y0, Y1 = 2001, 2020
HALF = 10  # +- days around each pentad start
_H5LOCK = threading.Lock()  # HDF5/netCDF reads are not thread-safe (repo gotcha: the first
# Actions run segfaulted in 7 s with 6 threads decoding at once)


def fetch(d: date, tries: int = 5):
    """(NY, NX) mm/day on the global 1° grid, or None. GPCP rows run 89.75°N -> 89.75°S, columns 179.75°W -> 179.75°E."""
    import netCDF4

    for k in range(tries):
        try:
            r = IM._session().get(URL.format(y=d.year, d=f"{d:%Y%m%d}"), timeout=180)
            r.raise_for_status()
            with (
                _H5LOCK,
                netCDF4.Dataset("gpcp.nc4", mode="r", memory=r.content) as f,
            ):  # download in threads, decode one at a time
                v = f.variables["precip"]
                v.set_auto_maskandscale(False)
                a = np.asarray(v[0], dtype="float32")
                la = np.asarray(f.variables["lat"][:])
                lo = np.asarray(f.variables["lon"][:])
            if a.shape != (360, 720) or abs(la[0] - 89.75) > 1e-3 or abs(lo[0] + 179.75) > 1e-3:
                raise ValueError(f"grid {a.shape} lat0 {la[0]} lon0 {lo[0]}")
            a = np.where((a >= 0) & (a < 2000), a, np.nan)[60:300][::-1]  # 59.75°N..59.75°S -> ascending
            b = a.reshape(G.NY, 2, G.NX, 2)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(b, axis=(1, 3))
            m[np.isfinite(b).sum((1, 3)) < 2] = np.nan
            w = G.area_weights()
            ok = np.isfinite(m)
            mu = float(np.nansum(m * w) / (w * ok).sum())
            if ok.mean() < 0.95 or not (1.0 < mu < 6.0):
                raise ValueError(f"implausible: valid {ok.mean():.2f} mean {mu:.2f}")
            return m.astype("float32")
        except Exception as ex:  # noqa: BLE001
            if k == tries - 1:
                print(f"  {d} failed: {repr(ex)[:90]}", flush=True)
                return None
            time.sleep(3 * (k + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    days = [
        d
        for y in range(Y0, Y1 + 2)
        for d in (date(y, 1, 1) + timedelta(days=i) for i in range(366))
        if d.year == y and not (d.month == 2 and d.day == 29) and (y <= Y1 or d < date(Y1 + 1, 1, 8))
    ]
    t0 = time.time()
    IM._session()  # log in once, before the threads
    pr = np.full((len(days), G.NY, G.NX), np.nan, dtype="float32")
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, g in enumerate(ex.map(fetch, days)):
            if g is not None:
                pr[i] = g
            if i % 500 == 0:
                print(f"  {days[i]} ({i + 1}/{len(days)}, {(time.time() - t0) / 60:.0f} min)", flush=True)
    missing = [f"{d:%Y%m%d}" for d, g in zip(days, pr) if not np.isfinite(g).any()]
    if len(missing) > 0.02 * len(days):
        print(f"too many missing days: {len(missing)}", flush=True)
        return 1
    # 7-day totals starting on each day (a total touching a missing day or an invalid cell is NaN)
    c = np.concatenate(
        [np.zeros((1, *pr.shape[1:]), "float64"), np.cumsum(np.nan_to_num(pr, nan=0.0), axis=0, dtype="float64")]
    )
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(pr).all(axis=(1, 2)))])
    cellbad = np.concatenate(
        [np.zeros((1, *pr.shape[1:]), np.int16), np.cumsum(~np.isfinite(pr), axis=0, dtype=np.int16)]
    )
    n = len(days) - 6
    tot = (c[7 : 7 + n] - c[:n]).astype("float32")
    tot[(cellbad[7 : 7 + n] - cellbad[:n]) > 0] = np.nan
    tot[(bad[7 : 7 + n] - bad[:n]) > 0] = np.nan
    doy = np.array(
        [(d - date(d.year, 1, 1)).days + 1 - (1 if d.month > 2 and d.year % 4 == 0 else 0) for d in days[:n]]
    )  # 1..365
    yr = np.array([d.year for d in days[:n]])
    starts = np.arange(1, 366, 5)  # 73 pentads
    out = {k: np.full((len(starts), G.NY, G.NX), np.nan, "float32") for k in ("mean", "q33", "q50", "q67")}
    for p, s in enumerate(starts):
        dist = np.abs(((doy - s + 182) % 365) - 182)
        sel = (dist <= HALF) & (yr >= Y0) & (yr <= Y1)
        x = tot[sel]
        with np.errstate(invalid="ignore"):
            out["mean"][p] = np.nanmean(x, axis=0)
            q = np.nanpercentile(x, [100 / 3, 50, 200 / 3], axis=0)
        out["q33"][p], out["q50"][p], out["q67"][p] = q
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / "gpcp_clim_week_global.npz"
    np.savez_compressed(
        f,
        start_doy=starts,
        lat=G.LAT,
        lon=G.LON,
        years=np.array([Y0, Y1]),
        half_window=HALF,
        missing=np.array(missing, dtype=str),
        **out,
    )
    w = G.area_weights()
    print(
        f"{f.name}: {len(days) - len(missing)}/{len(days)} days, global-band mean {np.nansum(out['mean'].mean(0) * w) / w.sum() / 7:.2f} "
        f"mm/day, {(time.time() - t0) / 60:.0f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
