#!/usr/bin/env python3
"""GFS, ECMWF IFS and AIFS scored with mverify for verification.html.

Downloads only the fields that are scored (byte ranges from each file's index): 2 m temperature and dew point,
MSLP, 10 m wind, precipitation, 500 hPa height, 850 hPa temperature and 250 hPa wind every 12 h to day 10,
plus orography and the land-sea mask at step 0. ECMWF comes from the Google Cloud mirror only (as everywhere on
the site, see scripts/ecmwf/rangefetch.py); GFS from NOAA on AWS. Then

    mverify score GFS=... IFS=... AIFS=...   (one sample for all three, paired significance tests)

against an mverify observation database, and the scorecard goes to assets/verification/scores.json.

    python scripts/verify/mverify_scores.py --db obs.duckdb --inits 2026092600 2026100500 --work /tmp/fc
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets" / "verification" / "scores.json"
LEADS = range(0, 241, 12)
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
ECMWF_FIELDS = {
    ("2t", "sfc", None),
    ("2d", "sfc", None),
    ("msl", "sfc", None),
    ("10u", "sfc", None),
    ("10v", "sfc", None),
    ("tp", "sfc", None),
    ("gh", "pl", "500"),
    ("t", "pl", "850"),
    ("u", "pl", "250"),
    ("v", "pl", "250"),
}
ECMWF_STATIC = {("lsm", "sfc", None), ("z", "sfc", None)}
GFS_FIELDS = [
    ":TMP:2 m above ground:",
    ":DPT:2 m above ground:",
    ":PRMSL:mean sea level:",
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":APCP:surface:",
    ":HGT:500 mb:",
    ":TMP:850 mb:",
    ":UGRD:250 mb:",
    ":VGRD:250 mb:",
]
GFS_STATIC = [":HGT:surface:", ":LAND:surface:", ":SFCR:surface:"]
MODELS = {"GFS": "gfs", "IFS": "ifs", "AIFS": "aifs-single"}


def get(url: str, rng: str | None = None) -> bytes:
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"gave up on {url}")


def merged(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Adjacent byte ranges as one request each."""
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a == out[-1][1] + 1:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def fetch(model: str, init: str, lead: int, work: Path) -> None:
    dest = work / MODELS[model] / f"{init}_{lead:03d}.grib2"
    if dest.exists():
        return
    day, hh = init[:8], init[8:]
    if model == "GFS":
        url = f"{GFS}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"
        lines = get(url + ".idx").decode().splitlines()
        offsets = [int(line.split(":")[1]) for line in lines]
        want = GFS_FIELDS + (GFS_STATIC if lead == 0 else [])
        # the index gives no end for the last message, which is never one of these
        spans = [(offsets[i], offsets[i + 1] - 1) for i, line in enumerate(lines[:-1]) if any(f in line for f in want)]
    else:
        url = f"{GOOGLE}/{day}/{hh}z/{MODELS[model]}/0p25/oper/{day}{hh}0000-{lead}h-oper-fc"
        want = ECMWF_FIELDS | (ECMWF_STATIC if lead == 0 else set())
        spans = []
        for line in get(url + ".index").decode().splitlines():
            d = json.loads(line)
            if (d["param"], d["levtype"], d.get("levelist")) in want:
                spans.append((d["_offset"], d["_offset"] + d["_length"] - 1))
        url += ".grib2"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with open(part, "wb") as fh:
        for a, b in merged(spans):
            fh.write(get(url, f"{a}-{b}"))
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="mverify observation database")
    ap.add_argument("--inits", nargs=2, required=True, metavar=("FIRST", "LAST"), help="00Z runs, YYYYMMDDHH")
    ap.add_argument("--work", type=Path, required=True, help="download directory")
    a = ap.parse_args()
    inits = [
        t.strftime("%Y%m%d%H")
        for t in pd.date_range(
            pd.to_datetime(a.inits[0], format="%Y%m%d%H"), pd.to_datetime(a.inits[1], format="%Y%m%d%H"), freq="1D"
        )
    ]
    jobs = [(m, i, ld) for m in MODELS for i in inits for ld in LEADS]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(*j, a.work), jobs))

    from mverify import report
    from mverify.db.store import ObsStore
    from mverify.verify import verify

    with ObsStore.open(a.db, read_only=True) as store:
        res = verify({m: sorted((a.work / d).glob("*.grib2")) for m, d in MODELS.items()}, store)
    report.write_json(res.scorecard(), OUT)
    print(f"{len(res.pairs):,} pairs, {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
