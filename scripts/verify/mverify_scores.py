#!/usr/bin/env python3
"""GFS, ECMWF IFS, AIFS and ECCC GDPS scored with mverify for verification.html.

Downloads only the fields that are scored (byte ranges from each file's index), every 12 h to day 10:
  * surface: 2 m temperature and dew point, MSLP, 10 m wind and precipitation, plus orography and the
    land-sea mask at step 0;
  * upper air: height, temperature and wind at 1000-100 hPa and specific humidity at 1000-300 hPa
    (mverify.variables.LEVELS, HUMIDITY_LEVELS), scored against radiosondes. These are some 50 fields a
    step, so each run is downloaded, matched to the soundings and deleted before the next; its pairs are
    kept as Parquet and scored level by level at the end (mverify.verify.score_pair_files).
ECMWF comes from the Google Cloud mirror only (as everywhere on the site, see scripts/ecmwf/rangefetch.py);
GFS from NOAA on AWS; GDPS from the ECCC Datamart, which keeps about 30 days and has one file per field.
All models are scored on one sample with paired significance tests, and the scorecard goes to
assets/verification/scores.json.

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
from mverify import variables as V

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets" / "verification" / "scores.json"
LEADS = range(0, 241, 12)
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UPPER = [(b, lv) for lv in V.LEVELS for b in "ztuv"] + [("q", lv) for lv in V.HUMIDITY_LEVELS]
ECMWF_FIELDS = {
    ("2t", "sfc", None),
    ("2d", "sfc", None),
    ("msl", "sfc", None),
    ("10u", "sfc", None),
    ("10v", "sfc", None),
    ("tp", "sfc", None),
}
ECMWF_UPPER = {({"z": "gh"}.get(b, b), "pl", str(lv)) for b, lv in UPPER}
ECMWF_STATIC = {("lsm", "sfc", None), ("z", "sfc", None)}
GFS_FIELDS = [
    ":TMP:2 m above ground:",
    ":DPT:2 m above ground:",
    ":PRMSL:mean sea level:",
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":APCP:surface:",
]
GFS_UPPER = [f":{dict(z='HGT', t='TMP', u='UGRD', v='VGRD', q='SPFH')[b]}:{lv} mb:" for b, lv in UPPER]
GFS_STATIC = [":HGT:surface:", ":LAND:surface:", ":SFCR:surface:"]
DATAMART = "https://dd.weather.gc.ca"
GDPS_FIELDS = [
    "AirTemp_AGL-2m",
    "DewPoint_AGL-2m",
    "Pressure_MSL",
    "WindU_AGL-10m",
    "WindV_AGL-10m",
    "Precip-Accum_Sfc",
]
GDPS_UPPER = [
    f"{dict(z='GeopotentialHeight', t='AirTemp', u='WindU', v='WindV', q='SpecificHumidity')[b]}_IsbL-{lv:04d}"
    for b, lv in UPPER
]
GDPS_STATIC = ["GeopotentialHeight_Sfc", "LandWaterProportion_Sfc"]
MODELS = {"GFS": "gfs", "IFS": "ifs", "AIFS": "aifs-single", "GDPS": "gdps"}


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


def fetch(model: str, init: str, lead: int, dest: Path, upper: bool) -> None:
    """The surface fields (``upper`` False) or the pressure-level fields of one model step, into ``dest``."""
    if dest.exists():
        return
    day, hh = init[:8], init[8:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    if model == "GDPS":
        names = GDPS_UPPER if upper else GDPS_FIELDS + (GDPS_STATIC if lead == 0 else [])
        base = f"{DATAMART}/{day}/WXO-DD/model_gdps/15km/{hh}/{lead:03d}/{day}T{hh}Z_MSC_GDPS"
        urls = [
            f"{base}_{n}_LatLon0.15_PT{lead:03d}H.grib2"
            for n in names
            if not (lead == 0 and n == "Precip-Accum_Sfc")  # no accumulation at step 0
        ]
        # one file per field, so the fields of a step are fetched side by side
        with ThreadPoolExecutor(8) as ex:
            part.write_bytes(b"".join(ex.map(get, urls)))
        part.rename(dest)
        return
    if model == "GFS":
        url = f"{GFS}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"
        lines = get(url + ".idx").decode().splitlines()
        offsets = [int(line.split(":")[1]) for line in lines]
        want = GFS_UPPER if upper else GFS_FIELDS + (GFS_STATIC if lead == 0 else [])
        # the index gives no end for the last message, which is never one of these
        spans = [(offsets[i], offsets[i + 1] - 1) for i, line in enumerate(lines[:-1]) if any(f in line for f in want)]
    else:
        url = f"{GOOGLE}/{day}/{hh}z/{MODELS[model]}/0p25/oper/{day}{hh}0000-{lead}h-oper-fc"
        want = ECMWF_UPPER if upper else ECMWF_FIELDS | (ECMWF_STATIC if lead == 0 else set())
        spans = []
        for line in get(url + ".index").decode().splitlines():
            d = json.loads(line)
            if (d["param"], d["levtype"], d.get("levelist")) in want:
                spans.append((d["_offset"], d["_offset"] + d["_length"] - 1))
        url += ".grib2"
    with open(part, "wb") as fh:
        for a, b in merged(spans):
            fh.write(get(url, f"{a}-{b}"))
    part.rename(dest)


def download(jobs: list[tuple[str, str, int]], root: Path, upper: bool) -> None:
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(*j, root / MODELS[j[0]] / f"{j[1]}_{j[2]:03d}.grib2", upper), jobs))


def upper_pairs(init: str, work: Path, store) -> Path:
    """Radiosonde pairs of every model for one run, saved as Parquet; the run's files are deleted after."""
    from mverify.io.open import open_forecast
    from mverify.match import match

    out = work / "pairs" / f"{init}.parquet"
    if out.exists():
        return out
    root = work / "upper"
    download([(m, init, ld) for m in MODELS for ld in LEADS], root, upper=True)
    pairs = pd.concat(
        [
            match(open_forecast(sorted((root / d).glob(f"{init}_*.grib2")), name=m), store, networks=["RAOB"])
            for m, d in MODELS.items()
        ],
        ignore_index=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out.with_suffix(".part"))
    out.with_suffix(".part").rename(out)
    for f in root.glob(f"*/{init}_*.grib2"):
        f.unlink()
    return out


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
    download([(m, i, ld) for m in MODELS for i in inits for ld in LEADS], a.work, upper=False)

    from mverify import report
    from mverify.db.store import ObsStore
    from mverify.verify import Result, score_pair_files, verify

    with ObsStore.open(a.db, read_only=True) as store:
        surface = verify(
            {m: sorted((a.work / d).glob("*.grib2")) for m, d in MODELS.items()}, store, networks=["GHCNH"]
        )
        files = []
        for init in inits:
            files.append(upper_pairs(init, a.work, store))
            print(f"{init}: radiosonde pairs saved", flush=True)
    res = Result.concat([surface, score_pair_files(files)])
    report.write_json(res.scorecard(), OUT)
    print(f"{len(surface.pairs):,} surface pairs, {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
