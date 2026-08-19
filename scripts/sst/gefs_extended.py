#!/usr/bin/env python3
"""GEFS extended-range rain (days ~10-35) over the Colombia basins.

Fills the gap the AIFS/IFS ensemble cannot reach.  The daily chain runs
to 15 days; the monthly chain starts at +1 month.  Between them sits a
fortnight with no dynamical rain at all, which is why the two halves
disagree at the seam (2026-08: the daily chain implied 81% of norm for
the month, the monthly model said 69%).

Source: noaa-gefs-pds, gefs.YYYYMMDD/00/atmos/pgrb2ap5 — 0.5 deg, APCP as
6-hour accumulations, out to f840.  33 members run to f384, 24 to f840.

Only the APCP message is fetched, via the .idx byte offsets, so a member
-hour costs ~100 KB instead of ~12 MB.  Same trick as backfill_bias.py.

    python scripts/sst/gefs_extended.py [--date YYYYMMDD] [--members 24]

Output: ~/colombia_hydro/raw/gefs_rain/gefs_<date>_00z.json.gz
        (basin-mean mm/day per member per valid day)
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "gefs_rain"
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
PREF = "gefs.{d}/00/atmos/pgrb2ap5"
FN = "{mem}.t00z.pgrb2a.0p50.f{h:03d}"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
FH0, FH1, STEP = 240, 840, 6          # day 10 -> day 35; overlap with AIFS
                                      # on purpose, so the two can be
                                      # cross-calibrated where they overlap
_LOCK = threading.Lock()


def idx_range(session_url: str):
    """(start, end) byte offsets of the APCP message, from the .idx."""
    try:
        txt = urllib.request.urlopen(session_url + ".idx", timeout=60).read().decode()
    except Exception:                                   # noqa: BLE001
        return None
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ":APCP:" not in ln:
            continue
        start = int(ln.split(":")[1])
        end = ""
        if i + 1 < len(lines):
            end = str(int(lines[i + 1].split(":")[1]) - 1)
        return start, end
    return None


def fetch_msg(url: str, rng) -> bytes | None:
    start, end = rng
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:                               # noqa: BLE001
            pass
    return None


def decode(buf: bytes, W):
    """Basin-mean mm from one APCP GRIB message."""
    import xarray as xr
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as f:
        f.write(buf); f.flush()
        ds = xr.open_dataset(f.name, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
        v = [x for x in ds.data_vars]
        da = ds[v[0]]
        lon = da.longitude.values
        if lon.max() > 180:
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
        da = da.sortby("longitude").sortby("latitude")
        g = np.squeeze(da.values)
        if W["_grid"] is None:
            from hydro_region_rain import region_weights_energy
            Wt = region_weights_energy(da.longitude.values, da.latitude.values,
                                       ORDER)
            if Wt is None:
                import c3s_precip as C
                Wt = C.coarse_weights(da.longitude.values, da.latitude.values)
            W["_grid"] = {b: Wt[b].ravel() for b in ORDER}
        flat = np.nan_to_num(g).ravel()
        return {b: float(np.dot(flat, W["_grid"][b])) for b in ORDER}




def run_cycle(d, mems, hrs, workers):
    """Ingest one cycle; returns the path written, or None."""
    pref = PREF.format(d=d)
    jobs = [(m, h) for m in mems for h in hrs]
    W = {"_grid": None}
    acc, done = {}, {"n": 0, "miss": 0}

    def work(job):
        m, h = job
        url = f"{BASE}/{pref}/{FN.format(mem=m, h=h)}"
        rng = idx_range(url)
        if rng is None:
            with _LOCK:
                done["miss"] += 1
            return
        buf = fetch_msg(url, rng)
        if not buf:
            with _LOCK:
                done["miss"] += 1
            return
        try:
            vals = decode(buf, W)
        except Exception:                               # noqa: BLE001
            with _LOCK:
                done["miss"] += 1
            return
        with _LOCK:
            acc.setdefault(m, {})[h] = vals
            done["n"] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    if not acc:
        print(f"  {d}: nothing fetched ({done['miss']} missing)", flush=True)
        return None

    # 6-hour accumulations -> daily totals on 00Z calendar days
    init = datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc)
    out = {}
    for m, byh in acc.items():
        daily = {}
        for h, vals in byh.items():
            # the message covers (h-6, h]; attribute it to the day it ends in,
            # less a second, so an h landing exactly at 00Z belongs to the day
            # that just finished
            k = (init + timedelta(hours=h)
                 - timedelta(seconds=1)).strftime("%Y-%m-%d")
            for b in ORDER:
                daily.setdefault(k, {}).setdefault(b, 0.0)
                daily[k][b] += vals[b]
            daily[k]["_n"] = daily[k].get("_n", 0) + 1
        for k, v in daily.items():
            if v.pop("_n", 0) != 4:                     # complete days only
                continue
            out.setdefault(k, {})[m] = {b: round(v[b], 3) for b in ORDER}

    ARCH.mkdir(parents=True, exist_ok=True)
    tag = "mean" if mems == ["geavg"] else "ens"
    dest = ARCH / f"gefs_{d}_00z_{tag}.json.gz"
    with gzip.open(dest, "wt") as f:
        json.dump({"model": "gefs", "init_date": d, "init_hh": "00",
                   "members": sorted(acc), "kind": tag,
                   "units": "mm/day, basin-mean on energy weights",
                   "generated": datetime.now(timezone.utc)
                   .strftime("%Y-%m-%d %H:%M UTC"), "days": out}, f,
                  separators=(",", ":"))
    days = sorted(out)
    print(f"  {d}: {done['n']} msgs ok / {done['miss']} missing, "
          f"{len(days)} days {days[0] if days else '-'}.."
          f"{days[-1] if days else '-'} -> {dest.name}", flush=True)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--members", type=int, default=31)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mean-only", action="store_true",
                    help="fetch only geavg — one message per step instead of "
                         "~24. Bias factors correct the ensemble MEAN, so the "
                         "mean product is all the calibration needs, which "
                         "makes backfilling past cycles affordable.")
    ap.add_argument("--backfill", type=int, default=0,
                    help="also ingest N earlier cycles, spaced --spacing days")
    ap.add_argument("--spacing", type=int, default=5)
    a = ap.parse_args()
    d = a.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    mems = (["geavg"] if a.mean_only
            else ["gec00"] + [f"gep{i:02d}" for i in range(1, a.members)])
    hrs = list(range(FH0, FH1 + 1, STEP))
    dates = [d]
    if a.backfill:
        d0 = datetime.strptime(d, "%Y%m%d")
        dates += [(d0 - timedelta(days=a.spacing * (i + 1))).strftime("%Y%m%d")
                  for i in range(a.backfill)]
    print(f"GEFS 00Z — {len(mems)} member(s) x {len(hrs)} steps x "
          f"{len(dates)} cycle(s)", flush=True)
    for dd in dates:
        run_cycle(dd, mems, hrs, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
