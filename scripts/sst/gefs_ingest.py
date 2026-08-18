#!/usr/bin/env python3
"""GEFS ensemble rainfall -> per-basin member archive (Brazil SIN basins).

Source: NOAA GEFS on AWS (noaa-gefs-pds, no auth). APCP is served as
6-hourly buckets per member per step; the .idx files give byte ranges so
we fetch ONLY the APCP messages. Two products:
  pgrb2sp25 (0.25 deg)  f006..f240   -> days 1-10, all 31 members  (engine)
  pgrb2ap5  (0.5 deg)   f246..f840   -> days 11-35 (00Z only)     (clusters)
Daily totals are calendar days (00Z->00Z UTC) built from the buckets.
Writes the same archive schema as the AIFS/IFS extraction so the Brazil
forecast engine treats GEFS as a third model:
  ~/brazil_hydro/raw/fcst_rain/gefs_YYYYMMDD_HHz.json.gz
    {model, init_date, init_hh, valid[], n_members, basins{b:[mem][lead]},
     source: "aws"|"local"}
When the user's early GEFS lands, point GEFS_LOCAL_DIR at it (same GRIB
files) and this script reads locally instead of AWS.

    python scripts/sst/gefs_ingest.py [--date YYYYMMDD --hh 00|12] [--days 15|35]
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_model import basin_weights, MAJORS                 # noqa: E402

PRIV = Path.home() / "brazil_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
CACHE = PRIV / "raw" / "gefs_grib"
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_LOCAL_DIR = os.environ.get("GEFS_LOCAL_DIR")               # user's early run
N_MEM = 30                                                        # gep01..gep30 (+gec00)
BOX = dict(lon0=-76.0, lon1=-33.0, lat0=-35.0, lat1=6.0)


def get(url, rng=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if rng:
                req.add_header("Range", f"bytes={rng}")
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (k + 1))
        except Exception:                                          # noqa: BLE001
            time.sleep(1.5 * (k + 1))
    return None


def apcp_range(idx_txt: str):
    """(offset, length) of the APCP message from an .idx text (or None)."""
    lines = idx_txt.splitlines()
    for i, ln in enumerate(lines):
        p = ln.split(":")
        if len(p) > 3 and p[3] == "APCP":
            off = int(p[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            return off, end
    return None


def fetch_member_steps(date, hh, mem, steps, res):
    """Concatenated APCP GRIB messages for one member over steps."""
    prod = "pgrb2sp25" if res == "0p25" else "pgrb2ap5"
    tag = "pgrb2s" if res == "0p25" else "pgrb2a"
    out = bytearray()
    got = []
    for s in steps:
        name = f"{mem}.t{hh}z.{tag}.{res}.f{s:03d}"
        url = f"{BASE}/gefs.{date}/{hh}/atmos/{prod}/{name}"
        cache = CACHE / date / hh / f"{name}.apcp"
        if cache.exists():
            out += cache.read_bytes()
            got.append(s)
            continue
        idx = get(url + ".idx")
        if idx is None:
            continue
        rng = apcp_range(idx.decode())
        if rng is None:
            continue
        blob = get(url, rng=f"{rng[0]}-{rng[1]}")
        if blob is None:
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
        out += blob
        got.append(s)
    return bytes(out), got


def decode_daily(blob: bytes, init: datetime, W: dict, lats_ref=None):
    """Sum 6h APCP buckets into calendar-day (UTC) totals; basin means."""
    import eccodes
    tmp = CACHE / f"_tmp_{os.getpid()}.grib2"
    tmp.write_bytes(blob)
    days = {}                     # date -> field accum
    Wg = None
    with open(tmp, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                end = int(eccodes.codes_get(gid, "endStep"))
                start = int(eccodes.codes_get(gid, "startStep"))
                if end - start != 6:
                    continue
                vend = init + timedelta(hours=end)
                # bucket belongs to the calendar day containing (vend - 3h)
                day = (vend - timedelta(hours=3)).date()
                if Wg is None:
                    ni = eccodes.codes_get(gid, "Ni"); nj = eccodes.codes_get(gid, "Nj")
                    la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                    lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                    dj = eccodes.codes_get(gid, "jDirectionIncrementInDegrees")
                    di = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
                    lats = la0 - dj * np.arange(nj)          # N->S
                    lons = (lo0 + di * np.arange(ni))
                    lons = np.where(lons > 180, lons - 360, lons)
                    ti = np.where((lats >= BOX["lat0"]) & (lats <= BOX["lat1"]))[0]
                    li = np.where((lons >= BOX["lon0"]) & (lons <= BOX["lon1"]))[0]
                    sub_lats, sub_lons = lats[ti], lons[li]
                    order = np.argsort(sub_lats)
                    Wg = basin_weights(sub_lons, sub_lats[order], set(MAJORS))
                    idx = (ti, li, order, ni, nj)
                v = eccodes.codes_get_values(gid).reshape(idx[4], idx[3])
                sub = v[np.ix_(idx[0], idx[1])][idx[2], :]
                acc = days.setdefault(day, [np.zeros_like(sub), 0])
                acc[0] += np.nan_to_num(sub)
                acc[1] += 1
            finally:
                eccodes.codes_release(gid)
    tmp.unlink(missing_ok=True)
    out = {}
    for d, (fld, n) in sorted(days.items()):
        if n < 4:                                                  # partial day
            continue
        out[str(d)] = {b: float((fld * w).sum()) for b, w in Wg.items()}
    return out


def ingest(date: str, hh: str, ndays: int) -> Path | None:
    dest = ARCH / f"gefs_{date}_{hh}z.json.gz"
    if dest.exists():
        return dest
    init = datetime.strptime(date + hh, "%Y%m%d%H")
    steps25 = list(range(6, min(ndays, 10) * 24 + 7, 6))
    steps5 = list(range(246, ndays * 24 + 7, 6)) if ndays > 10 else []
    members = ["gec00"] + [f"gep{i:02d}" for i in range(1, N_MEM + 1)]
    per_mem = []
    for m in members:
        blob, got = fetch_member_steps(date, hh, m, steps25, "0p25")
        daily = decode_daily(blob, init, None) if blob else {}
        if steps5:
            blob5, _ = fetch_member_steps(date, hh, m, steps5, "0p50")
            if blob5:
                daily.update(decode_daily(blob5, init, None))
        per_mem.append(daily)
        print(f"  {m}: {len(daily)} days", flush=True)
    valid = sorted(set.intersection(*[set(d) for d in per_mem if d]))
    if len(valid) < 3:
        print("  too few valid days — not archived")
        return None
    rec = {"model": "gefs", "init_date": date, "init_hh": hh, "valid": valid,
           "n_members": len(per_mem), "source": "local" if GEFS_LOCAL_DIR else "aws",
           "basins": {b: [[round(dm[v][b], 2) if v in dm else 0.0 for v in valid]
                          for dm in per_mem] for b in MAJORS if all(b in dm.get(valid[0], {}) for dm in per_mem)}}
    ARCH.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt") as f:
        json.dump(rec, f, separators=(",", ":"))
    print(f"wrote {dest.name}: {len(valid)} days x {len(per_mem)} members")
    return dest


def main() -> int:
    a = sys.argv[1:]
    ndays = int(a[a.index("--days") + 1]) if "--days" in a else 15
    if "--date" in a:
        dates = [(a[a.index("--date") + 1], a[a.index("--hh") + 1] if "--hh" in a else "00")]
    else:
        now = datetime.now(timezone.utc)
        dates = []
        for back in (0, 1):
            d = (now - timedelta(days=back))
            for hh in ("12", "00"):
                dates.append((d.strftime("%Y%m%d"), hh))
    for date, hh in dates:
        print(f"GEFS {date} {hh}Z …", flush=True)
        ingest(date, hh, ndays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
