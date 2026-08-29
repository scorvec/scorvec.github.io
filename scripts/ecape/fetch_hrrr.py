#!/usr/bin/env python3
"""
Fetch one HRRR native-level cycle and decode it into a flat float32 scratch file
for the ECAPE kernel.

Why native (wrfnat) and not pressure levels: ECAPE is a mixed-layer/most-unstable
parcel quantity, so it lives or dies on how well the boundary layer is resolved.
The isobaric product is 25 hPa spacing (~250 m near the ground); the native
hybrid levels are ~20-40 m down there. HRRR also publishes HGT *on* the hybrid
levels, so no hybrid-coordinate height reconstruction is needed - we read
geopotential height directly.

Only 6 of the ~1100 messages per level-set are wanted, so we byte-range them out
of the 693 MB file using the published .idx inventory (same trick the CPTEC
ingest uses). That is ~360 MB instead of 693 MB.

Output:
  <out>.f32   float32, C order, shape (NVAR, NLEV, ny, nx)
  <out>.json  grid metadata + level count + cycle, for the kernel and the renderer

Usage:
  python fetch_hrrr.py --out /tmp/hrrr_ecape           # latest available cycle
  python fetch_hrrr.py --date 20260828 --hour 12 --out /tmp/hrrr_ecape
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

S3 = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
# Order matters: the kernel indexes vars by this position.
# The .idx inventory uses wgrib2 names; eccodes reports its own shortName for the
# same message, so the decoder needs the second column to match on.
VARS = ["PRES", "HGT", "TMP", "SPFH", "UGRD", "VGRD"]
SHORTNAME = {"pres": "PRES", "gh": "HGT", "t": "TMP",
             "q": "SPFH", "u": "UGRD", "v": "VGRD"}
NVAR = len(VARS)
LEVEL_TAG = "hybrid level"
# HRRR publishes 50 hybrid levels; index 1 is nearest the ground and 50 is the
# model top (~17 hPa).
NLEV_MODEL = 50
# We keep only the lowest 42 (up to ~70 hPa). Nothing above a parcel's
# equilibrium level contributes to CAPE or to ECAPE, and the saving is real:
# 16% off the download and 16% off the kernel, which scales with level count.
#
# 42 was measured, not guessed. Against the full 50-level cube the ECAPE_ML,
# ECAPE_MU, MLCAPE and MUCAPE fields come out BIT-IDENTICAL over all 1,905,141
# columns - zero differing values. Trimming further is not safe: at 38 levels
# (~115 hPa top) SHARPlib throws "Both bottom and top must be MISSING" on 9,343
# columns whose parcels reach an EL above the truncated profile. Do not lower
# this without repeating that comparison.
NLEV = 42
TIMEOUT = 120


def _url(date: str, hour: int, product: str = "wrfnat", fxx: int = 0) -> str:
    return f"{S3}/hrrr.{date}/conus/hrrr.t{hour:02d}z.{product}f{fxx:02d}.grib2"


def fetch_index(url: str):
    """Return [(msg_no, offset, var, level)] from the .idx sidecar."""
    import urllib.request
    with urllib.request.urlopen(url + ".idx", timeout=TIMEOUT) as r:
        text = r.read().decode()
    rows = []
    for line in text.splitlines():
        p = line.split(":")
        if len(p) < 6:
            continue
        rows.append((int(p[0]), int(p[1]), p[3], p[4]))
    rows.sort(key=lambda t: t[1])
    return rows


# Only these cycles run out to 48 h; every other hour stops at F18.
EXTENDED_HOURS = (0, 6, 12, 18)


def latest_cycle(max_back_hours: int = 12, extended_only: bool = False,
                 fxx: int = 0):
    """Newest cycle whose wrfnat .idx is published. HRRR runs hourly but the
    native files land later than the surface ones, so walk back until one is
    actually there rather than trusting the clock.

    extended_only restricts the search to the 00/06/12/18 UTC cycles, which are
    the only ones carrying forecast hours past F18 - asking a 13Z cycle for F24
    is a 404, not a wait. `fxx` is probed rather than F00 so we pick a cycle
    whose *longest* needed hour has actually landed: the tail of a 48 h run
    publishes well after its analysis."""
    import urllib.error
    import urllib.request
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for back in range(max_back_hours * (4 if extended_only else 1)):
        t = now - timedelta(hours=back)
        if extended_only and t.hour not in EXTENDED_HOURS:
            continue
        u = _url(t.strftime("%Y%m%d"), t.hour, fxx=fxx)
        try:
            req = urllib.request.Request(u + ".idx", method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return t.strftime("%Y%m%d"), t.hour
        except (urllib.error.URLError, OSError):
            continue
    raise SystemExit("no HRRR wrfnat cycle found in the last "
                     f"{max_back_hours} h")


def wanted_ranges(rows):
    """Byte ranges for our 6 vars on all hybrid levels, plus the level ordering.

    Returns (ranges, keys) where ranges is [(start, end_inclusive)] and keys is
    the matching [(var, level_number)] so the decoder knows what each message is.
    """
    by_no = {}
    for i, (n, off, var, lev) in enumerate(rows):
        end = rows[i + 1][1] - 1 if i + 1 < len(rows) else None
        by_no[n] = (off, end, var, lev)
    ranges, keys = [], []
    for n in sorted(by_no):
        off, end, var, lev = by_no[n]
        if var not in VARS or not lev.endswith(LEVEL_TAG):
            continue
        k = int(lev.split()[0])                    # "37 hybrid level" -> 37
        if k > NLEV:                               # stratosphere: see NLEV above
            continue
        ranges.append((off, end))
        keys.append((var, k))
    return ranges, keys


def download(url: str, ranges, dest: Path, quiet=False, workers: int = 8):
    """Concatenate the requested byte ranges into one local GRIB2 file.

    Adjacent messages are merged into a single request - the wanted fields are
    interleaved with unwanted ones, but runs still collapse 300 messages into
    ~100 requests.

    Those requests then go out CONCURRENTLY. Issued one at a time the fetch runs
    at ~13.8 MB/s and is round-trip-latency bound, not bandwidth bound; eight at
    a time measures 41.3 MB/s, taking a 397 MB cycle from ~30 s to ~10 s.
    Sixteen is slower than eight (35.2 MB/s), so the pool is deliberately small.
    Results are reassembled in offset order - GRIB messages must land in the
    file in the order the index lists them.
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    merged = []
    for off, end in sorted(ranges):
        if merged and end is not None and merged[-1][1] is not None \
           and off <= merged[-1][1] + 1:
            merged[-1][1] = end
        else:
            merged.append([off, end])

    def grab(item):
        i, (off, end) = item
        rng = f"bytes={off}-" + ("" if end is None else str(end))
        req = urllib.request.Request(url, headers={"Range": rng})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return i, r.read()

    chunks = [None] * len(merged)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, buf in ex.map(grab, enumerate(merged)):
            chunks[i] = buf
            done += 1
            if not quiet and (done % 25 == 0 or done == len(merged)):
                print(f"  range {done}/{len(merged)}", flush=True)
    total = 0
    with open(dest, "wb") as fh:
        for buf in chunks:                 # offset order, not completion order
            fh.write(buf)
            total += len(buf)
    return total


def decode(grib: Path, out_stem: Path, cycle: dict, quiet=False):
    """Decode into (NVAR, NLEV, ny, nx) float32 and write the scratch + metadata."""
    import eccodes as ec

    arr = None
    ny = nx = None
    seen = np.zeros((NVAR, NLEV), bool)
    var_idx = {v: i for i, v in enumerate(VARS)}
    grid = {}

    with open(grib, "rb") as fh:
        n = 0
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                name = SHORTNAME.get(ec.codes_get(gid, "shortName"))
                lev = ec.codes_get(gid, "level")
                if name is None or ec.codes_get(gid, "typeOfLevel") != "hybrid" \
                   or not (1 <= lev <= NLEV):
                    continue
                if arr is None:
                    ny = ec.codes_get(gid, "Ny")
                    nx = ec.codes_get(gid, "Nx")
                    # memmap, not an in-RAM array: the full cube is 2.3 GB, and
                    # holding it resident is the one thing that would make this
                    # job marginal on a 7 GB Actions runner. Writing through to
                    # disk keeps peak RSS at a few tens of MB - the OS flushes
                    # pages as we go - and the kernel mmaps the same file next.
                    arr = np.memmap(out_stem.with_suffix(".f32"), np.float32,
                                    mode="w+", shape=(NVAR, NLEV, ny, nx))
                    # No NaN prefill: that would dirty all 2.3 GB for nothing.
                    # Coverage is guaranteed instead by the `seen` check below,
                    # which refuses to write metadata if any (var, level) is
                    # absent - so an unwritten cell can never reach the kernel.
                    grid = dict(
                        ny=int(ny), nx=int(nx),
                        lat1=float(ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")),
                        lon1=float(ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")),
                        latin1=float(ec.codes_get(gid, "Latin1InDegrees")),
                        latin2=float(ec.codes_get(gid, "Latin2InDegrees")),
                        lov=float(ec.codes_get(gid, "LoVInDegrees")),
                        dx=float(ec.codes_get(gid, "DxInMetres")),
                        dy=float(ec.codes_get(gid, "DyInMetres")),
                    )
                vals = ec.codes_get_values(gid).astype(np.float32)
                arr[var_idx[name], lev - 1] = vals.reshape(ny, nx)
                seen[var_idx[name], lev - 1] = True
                n += 1
                if not quiet and n % 50 == 0:
                    print(f"  decoded {n} messages", flush=True)
            finally:
                ec.codes_release(gid)

    if arr is None:
        raise SystemExit("no usable GRIB messages decoded")
    missing = int((~seen).sum())
    if missing:
        # A missing level would silently corrupt every column, so refuse rather
        # than emit a field with a hole in it.
        where = [(VARS[i], k + 1) for i, k in zip(*np.where(~seen))]
        raise SystemExit(f"missing {missing} (var, level) messages, e.g. {where[:5]}")

    f32 = out_stem.with_suffix(".f32")
    arr.flush()
    del arr
    meta = dict(cycle=cycle, vars=VARS, nlev=NLEV, grid=grid,
                shape=[NVAR, NLEV, grid["ny"], grid["nx"]],
                dtype="float32", order="C", scratch=str(f32.name))
    out_stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if not quiet:
        print(f"  wrote {f32} ({f32.stat().st_size/1e9:.2f} GB) "
              f"and {out_stem.with_suffix('.json').name}")
    return meta


def main(argv=None) -> int:
    global NLEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: latest available)")
    ap.add_argument("--hour", type=int, help="cycle hour UTC")
    ap.add_argument("--fxx", type=int, default=0, help="forecast hour")
    ap.add_argument("--out", help="output stem (no extension); not needed "
                                   "with --print-cycle")
    ap.add_argument("--keep-grib", action="store_true")
    ap.add_argument("--grib", help="decode this local GRIB instead of downloading "
                                   "(iteration aid; skips the fetch entirely)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--levels", type=int, default=NLEV,
                    help=f"hybrid levels to keep from the ground up "
                         f"(default {NLEV}; 38 is known to fail - see NLEV)")
    ap.add_argument("--extended-only", action="store_true",
                    help="only consider 00/06/12/18Z cycles (the 48 h runs)")
    ap.add_argument("--print-cycle", action="store_true",
                    help="resolve the cycle, print YYYYMMDDHH and exit. The "
                         "driver calls this ONCE and passes the result to every "
                         "forecast hour - re-resolving per hour could straddle "
                         "two model runs midway through a loop.")
    ap.add_argument("--probe-fxx", type=int, default=None,
                    help="probe this forecast hour when picking a cycle "
                         "(default: --fxx); use the longest hour you intend to "
                         "fetch so the whole run is known to be published")
    a = ap.parse_args(argv)

    NLEV = a.levels
    if a.date and a.hour is not None:
        date, hour = a.date, a.hour
    else:
        date, hour = latest_cycle(
            extended_only=a.extended_only,
            fxx=a.probe_fxx if a.probe_fxx is not None else a.fxx)
    if a.print_cycle:
        print(f"{date}{hour:02d}")
        return 0
    if not a.out:
        ap.error("--out is required unless --print-cycle is given")
    url = _url(date, hour, fxx=a.fxx)
    print(f"HRRR wrfnat {date} t{hour:02d}z f{a.fxx:02d}", flush=True)

    rows = fetch_index(url)
    ranges, keys = wanted_ranges(rows)
    got = {v: 0 for v in VARS}
    for v, _ in keys:
        got[v] += 1
    print(f"  index: {len(rows)} messages; want {len(ranges)} "
          f"({', '.join(f'{v}x{c}' for v, c in got.items())})", flush=True)
    for v, c in got.items():
        if c != NLEV:
            raise SystemExit(f"{v}: expected {NLEV} hybrid levels, index has {c}")

    out_stem = Path(a.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if a.grib:
        decode(Path(a.grib), out_stem, dict(date=date, hour=hour, fxx=a.fxx),
               quiet=a.quiet)
        return 0
    tmp = Path(tempfile.mkdtemp()) / "hrrr_subset.grib2"
    nbytes = download(url, ranges, tmp, quiet=a.quiet)
    print(f"  downloaded {nbytes/1e6:.1f} MB", flush=True)
    keep = Path(a.keep_grib_path) if getattr(a, "keep_grib_path", None) else None
    try:
        decode(tmp, out_stem, dict(date=date, hour=hour, fxx=a.fxx), quiet=a.quiet)
    finally:
        if a.keep_grib:
            dest = out_stem.with_suffix(".grib2")
            tmp.replace(dest)
            print(f"  kept {dest}")
        else:
            tmp.unlink(missing_ok=True)
        try:
            os.rmdir(tmp.parent)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
