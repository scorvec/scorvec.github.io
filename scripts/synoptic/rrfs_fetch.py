"""RRFS field fetcher — idx-driven HTTP byte-range downloads.

Mirrors what Herbie does for HRRR in render_maps.py, but hand-rolled:
Herbie's RRFS support lags the operational rrfs_a bucket layout, and all
we need is (idx lookup → byte range → cfgrib), so a ~100-line direct
fetcher is simpler and has no extra dependency surface.

Bucket layout (verified against live data 2026-07):
    https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.YYYYMMDD/HH/
        rrfs.tHHz.2dfld.3km.fFFF.conus.grib2       (+ .idx)
    FFF = 000..084, cycles 00/06/12/18Z.

The .idx format is identical to HRRR's wgrib2 style:
    msgnum:start_byte:d=YYYYMMDDHH:VAR:level:forecast spec:...

The RRFS 2dfld.3km.conus grid is BIT-IDENTICAL to the HRRR CONUS grid
(Lambert conformal, 1059x1799; lat/lon arrays compared with max abs diff
exactly 0.0), so the caller can difference RRFS and HRRR fields
elementwise with no regridding.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional

import requests
import xarray as xr

RRFS_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a"

# idx files are cached per (cycle, fxx): every variable fetched from the
# same GRIB file re-reads the same idx, so one small download serves all
# ~10 field lookups for that forecast hour.
_IDX_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rrfs_idx_cache")

_TIMEOUT = 30          # seconds, idx/HEAD requests
_DATA_TIMEOUT = 120    # seconds, byte-range GRIB downloads (a message is ~1-8 MB)


def _urls(cycle: datetime, fxx: int) -> tuple[str, str]:
    """Return (grib_url, idx_url) for a cycle/forecast-hour pair."""
    grib = (f"{RRFS_BASE}/rrfs.{cycle:%Y%m%d}/{cycle:%H}/"
            f"rrfs.t{cycle:%H}z.2dfld.3km.f{fxx:03d}.conus.grib2")
    return grib, grib + ".idx"


def _get_idx_lines(cycle: datetime, fxx: int) -> list:
    """Download (or read cached) idx for this cycle/hour, as a line list."""
    os.makedirs(_IDX_CACHE_DIR, exist_ok=True)
    cache = os.path.join(
        _IDX_CACHE_DIR, f"rrfs.{cycle:%Y%m%d}.t{cycle:%H}z.f{fxx:03d}.idx")
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        with open(cache) as f:
            return f.read().splitlines()
    _, idx_url = _urls(cycle, fxx)
    r = requests.get(idx_url, timeout=_TIMEOUT)
    r.raise_for_status()
    # Only cache complete, successful responses so a failed fetch can't
    # poison later lookups with a truncated idx.
    with open(cache, "w") as f:
        f.write(r.text)
    return r.text.splitlines()


def _line_matches(line: str, search: str) -> bool:
    """Match an idx line against a search string.

    Plain searches (":TMP:2 m above ground") are substring matches on the
    ":VAR:level:forecast spec:" text, same spirit as Herbie. Searches
    containing a backslash (e.g. r":DSWRF:surface:\\d+ hour fcst:") are
    treated as regexes — needed where the forecast-hour spec varies per
    fxx (RRFS instantaneous DSWRF, the species-split COLMD smoke tracer,
    and run-total APCP).
    """
    if "\\" in search:
        return re.search(search, line) is not None
    return search in line


def _byte_range(lines: list, search: str) -> Optional[tuple]:
    """Find the FIRST matching idx line; return (start_byte, end_byte).

    end_byte is inclusive (Range-header style), or None when the match is
    the last message in the file (open-ended range).
    """
    for i, line in enumerate(lines):
        if not _line_matches(line, search):
            continue
        start = int(line.split(":")[1])
        if i + 1 < len(lines):
            return start, int(lines[i + 1].split(":")[1]) - 1
        return start, None
    return None


def fetch_rrfs_field(cycle: datetime, fxx: int, search: str):
    """Fetch one RRFS GRIB message as an xarray DataArray, or None.

    Retries once on transient failure; returns None on persistent failure
    (the render driver already tolerates missing fields per-hour).
    """
    for attempt in (1, 2):
        tmp_path = None
        try:
            lines = _get_idx_lines(cycle, fxx)
            rng = _byte_range(lines, search)
            if rng is None:
                # Not a transient error: the field simply isn't in this
                # file (e.g. APCP at F00). No point retrying.
                print(f"    rrfs: no idx match for {search!r} F{fxx:02d}",
                      file=sys.stderr)
                return None
            start, end = rng
            headers = {"Range": f"bytes={start}-{'' if end is None else end}"}
            grib_url, _ = _urls(cycle, fxx)
            r = requests.get(grib_url, headers=headers, timeout=_DATA_TIMEOUT)
            r.raise_for_status()

            # cfgrib needs a real file; indexpath='' stops it from writing
            # a stray .idx sidecar next to the temp file.
            fd, tmp_path = tempfile.mkstemp(suffix=".grib2",
                                            dir=_IDX_CACHE_DIR)
            with os.fdopen(fd, "wb") as f:
                f.write(r.content)
            ds = xr.open_dataset(tmp_path, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
            try:
                var = list(ds.data_vars)[0]
                # .load() so the DataArray survives the temp file's deletion
                da = ds[var].load()
            finally:
                ds.close()
            return da
        except Exception as e:
            if attempt == 1:
                time.sleep(2)
                continue
            print(f"    rrfs fetch failed ({search} F{fxx:02d}): {e}",
                  file=sys.stderr)
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def fetch_rrfs_availability(cycle: datetime) -> bool:
    """True if this RRFS cycle exists on the bucket (HEAD the f000 idx).

    RRFS publication lags HRRR by up to a couple of hours, so the driver
    probes here and falls back to HRRR-only when the cycle isn't up yet.
    """
    _, idx_url = _urls(cycle, 0)
    try:
        r = requests.head(idx_url, timeout=_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False
