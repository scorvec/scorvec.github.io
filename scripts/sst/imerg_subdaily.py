#!/usr/bin/env python3
"""Half-hourly IMERG over the basins, for a sampled set of days.

Does WHEN the rain fell matter, beyond how much fell?

At 0.1-degree DAILY resolution the basin mean already carries the
available spatial information: the areal fraction above 10 mm correlates
slightly better with the inflow change (+0.472 vs +0.463) but adds almost
nothing in the model, and nothing at all for spikes. That points at
temporal structure as the missing piece — a 60 mm day delivered in six
hours is a different event from the same total spread evenly, and the
daily product cannot tell them apart.

Downloading half-hourly for 26 years is not affordable (48 granules a
day). Downloading it for a SAMPLE is: high-rain days carry the spikes we
care about, and matched controls keep the comparison honest. Granules are
subset server-side through OPeNDAP at ~71 KB each.

NOTE: in the half-hourly product the variable is root-level
`precipitation`; the daily product's `/Grid/precipitation` constraint
returns HTTP 400.

    python scripts/sst/imerg_subdaily.py --events 200 --controls 200

Output: ~/colombia_hydro/raw/subdaily/<YYYYMMDD>.npz   (48 x basin means)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import (gauge_correction, region_weights_energy)  # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "raw" / "subdaily"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BASE = ("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/"
        "GPM_3IMERGHHE.07/{y}/{doy:03d}/{fn}")
_LOCK = threading.Lock()


def pick_days(n_events, n_controls, seed=19):
    """High-rain days plus matched controls from the same month and year."""
    import perfect_rain_backtest as PR
    import national_inflow as NI
    # the SST runner rewrites aporener_daily.json.gz in place, so a read
    # can land mid-write and raise EOFError on the gzip stream. Retry
    # rather than abandoning a multi-hour download.
    d = W = None
    for attempt in range(6):
        try:
            d = PR.load_all()
            W = NI.basin_energy_weights()
            break
        except (EOFError, OSError, ValueError) as e:
            print(f"   input read failed ({repr(e)[:50]}), retry {attempt+1}/6",
                  flush=True)
            time.sleep(20)
    if d is None or W is None:
        raise SystemExit("could not read inputs after retries")
    nat = sum(np.nan_to_num(d["rain_abs"][b]) * W[b] for b in ORDER)
    dates = np.array([str(x) for x in d["dates"]])
    # the half-hourly Early product is reliably archived from ~2015
    ok = np.array([s >= "2015-01-01" for s in dates])
    idx = np.where(ok & np.isfinite(nat))[0]
    order = idx[np.argsort(-nat[idx])]
    events = list(order[:n_events])
    rng = np.random.default_rng(seed)
    ev = set(events)
    # controls: same calendar month, NOT a top-decile rain day
    thr = np.percentile(nat[idx], 90)
    pool = [i for i in idx if i not in ev and nat[i] < thr]
    months = [dates[i][5:7] for i in events]
    controls = []
    for m in months[:n_controls]:
        cand = [i for i in pool if dates[i][5:7] == m and i not in controls]
        if cand:
            controls.append(int(rng.choice(cand)))
    return ([dates[i] for i in events], [dates[i] for i in controls],
            {dates[i]: float(nat[i]) for i in list(events) + controls})


def fetch_day(day, W, session, idx):
    """48 half-hourly basin-mean rain values (mm/hr) for one day."""
    import earthaccess
    import xarray as xr
    dest = OUT / f"{day.replace('-', '')}.npz"
    if dest.exists():
        return "cached"
    i0, i1, j0, j1 = idx
    try:
        gr = earthaccess.search_data(short_name="GPM_3IMERGHHE", version="07",
                                     temporal=(day, day), count=60)
    except Exception:                                              # noqa: BLE001
        return "search-failed"
    if len(gr) < 40:
        return f"only {len(gr)} granules"
    dt = datetime.strptime(day, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    rows, stamps, miss = [], [], 0
    for g in sorted(gr, key=lambda x: x.data_links()[0]):
        fn = g.data_links()[0].split("/")[-1]
        url = (BASE.format(y=dt.year, doy=doy, fn=fn) +
               f".dap.nc4?dap4.ce=/precipitation%5B0%5D"
               f"%5B{i0}:{i1 - 1}%5D%5B{j0}:{j1 - 1}%5D")
        # GES DISC OPeNDAP returns 503 under sustained load — six workers
        # times 48 granules per day is enough to trigger it, and the first
        # run wrote only 26 of 217 days because of it. Exponential backoff
        # on 503 specifically, and a hard stop so a throttled run gives up
        # on the day rather than grinding.
        buf = None
        for attempt in range(5):
            try:
                r = session.get(url, timeout=180)
                if r.status_code == 200 and len(r.content) > 20_000:
                    buf = r.content
                    break
                if r.status_code in (503, 429, 500):
                    time.sleep(min(60, 4 * (2 ** attempt)))
                    continue
                break                                   # 4xx: no retry
            except Exception:                                      # noqa: BLE001
                time.sleep(4 * (attempt + 1))
        if buf is None:
            miss += 1
            if miss > 8:                                # server is unhappy
                return f"abandoned after {miss} misses"
            continue
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=True) as f:
            f.write(buf); f.flush()
            with xr.open_dataset(f.name, engine="h5netcdf") as ds:
                a = np.squeeze(ds[list(ds.data_vars)[0]].values)
        g2 = np.where(a < 0, 0.0, a).T.ravel()      # (lon,lat) -> (lat,lon)
        rows.append([float(g2 @ W[b]) for b in ORDER])
        stamps.append(fn.split("-S")[1][:6])
    if len(rows) < 40:
        return f"only {len(rows)} slices"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, rain=np.asarray(rows), basins=np.array(ORDER),
                        stamps=np.array(stamps), day=day)
    return f"{len(rows)} slices"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--controls", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    ev, ct, mag = pick_days(a.events, a.controls)
    days = ev + ct
    print(f"{len(ev)} event days + {len(ct)} controls = {len(days)}")
    print(f"   event rain  {np.mean([mag[d] for d in ev]):5.1f} mm/d, "
          f"control rain {np.mean([mag[d] for d in ct]):5.1f} mm/d")

    IP._login()
    import earthaccess
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    i0, i1 = int(np.argmax(ml)), int(len(ml) - np.argmax(ml[::-1]))
    j0, j1 = int(np.argmax(mt)), int(len(mt) - np.argmax(mt[::-1]))
    F = gauge_correction(lons, lats)
    Wr = region_weights_energy(lons, lats, ORDER)
    W = {b: (Wr[b] * F).ravel() for b in ORDER}     # correction folded in
    session = earthaccess.get_requests_https_session()
    done = {"n": 0}

    def work(day):
        r = fetch_day(day, W, session, (i0, i1, j0, j1))
        with _LOCK:
            done["n"] += 1
            if done["n"] % 20 == 0 or "only" in str(r) or "failed" in str(r):
                print(f"   {done['n']}/{len(days)}  {day}: {r}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, days))
    have = len(list(OUT.glob("*.npz")))
    print(f"\n{have} days cached in {OUT}  ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
