#!/usr/bin/env python3
"""GFS, ECMWF IFS, AIFS and ECCC GDPS scored with mverify for verification.html.

Downloads only the fields that are scored (byte ranges from each file's index), every 12 h to day 10:
  * surface: 2 m temperature and dew point, pressure, 10 m wind and precipitation, plus orography and the
    land-sea mask at step 0. Pressure is surface pressure against station-level pressure, only where the model
    terrain is within 5 m of the station (mverify.match.MAX_DZ_PRESSURE), when every model publishes it, and
    mean sea-level pressure when one does not (pressure_variable);
  * upper air: height, temperature and wind at 1000-100 hPa and specific humidity at 1000-300 hPa
    (mverify.variables.LEVELS, HUMIDITY_LEVELS), scored against radiosondes. These are some 50 fields a
    step, so each run is downloaded, matched to the soundings and deleted before the next; its pairs are
    kept as Parquet and scored level by level at the end (mverify.verify.score_pair_files).
ECMWF comes from the Google Cloud mirror only (as everywhere on the site, see scripts/ecmwf/rangefetch.py);
GFS from NOAA on AWS; GDPS from the ECCC Datamart, which keeps about 30 days and has one file per field.
Every model is first averaged onto one 0.5 deg grid (mverify.coarsen, by area overlap, with its terrain and land-sea
mask), so the 0.15 deg GDPS and the 0.25 deg models are interpolated to the stations from cells of the same size. All
models are scored on one sample with paired significance tests, and the scorecard goes to
assets/verification/scores.json; rain against Stage IV in every cell of that grid, the bias tested cell by cell
(mverify.precip.maps), goes to assets/verification/precip_maps.json.

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
MAP_OUT = REPO / "assets" / "verification" / "stations.json"
RAIN_OUT = REPO / "assets" / "verification" / "precip_maps.json"
MAP_LEADS = [24, 72, 120, 168, 240]
MAP_VARS = ["t2m", "td2m", "wind10"] + [f"{b}{lv}" for lv in (850, 500, 250) for b in ("z", "t", "wind")]
MAP_VARS += ["q850", "q700"]
LEADS = range(0, 241, 12)
GRID = 0.5  # degrees: the common grid, coarser than every model's own resolution (AIFS, the coarsest, is ~31 km)
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UPPER = [(b, lv) for lv in V.LEVELS for b in "ztuv"] + [("q", lv) for lv in V.HUMIDITY_LEVELS]
ECMWF_FIELDS = {
    ("2t", "sfc", None),
    ("2d", "sfc", None),
    ("10u", "sfc", None),
    ("10v", "sfc", None),
    ("tp", "sfc", None),
}
ECMWF_UPPER = {({"z": "gh"}.get(b, b), "pl", str(lv)) for b, lv in UPPER}
ECMWF_STATIC = {("lsm", "sfc", None), ("z", "sfc", None)}
GFS_FIELDS = [
    ":TMP:2 m above ground:",
    ":DPT:2 m above ground:",
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
    "WindU_AGL-10m",
    "WindV_AGL-10m",
    "Precip-Accum_Sfc",
]
GDPS_UPPER = [
    f"{dict(z='GeopotentialHeight', t='AirTemp', u='WindU', v='WindV', q='SpecificHumidity')[b]}_IsbL-{lv:04d}"
    for b, lv in UPPER
]
GDPS_STATIC = ["GeopotentialHeight_Sfc", "LandWaterProportion_Sfc"]
# The pressure field of each model (ECMWF, GFS, GDPS) for the two ways of scoring pressure; one is added to the field
# lists for the run (pressure_variable)
PRESSURE = {
    "sp": (("sp", "sfc", None), ":PRES:surface:", "Pressure_Sfc"),
    "msl": (("msl", "sfc", None), ":PRMSL:mean sea level:", "Pressure_MSL"),
}
MODELS = {"GFS": "gfs", "IFS": "ifs", "AIFS": "aifs-single", "GDPS": "gdps"}
UA = "scorvec.com data pipeline (+https://scorvec.com)"  # the site's honest User-Agent (scripts/lib/webget.py)


def get(url: str, rng: str | None = None) -> bytes:
    for attempt in range(8):
        try:
            headers = {"User-Agent": UA, **({"Range": f"bytes={rng}"} if rng else {})}
            req = urllib.request.Request(url, headers=headers)
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
        spans, found = [], set()
        for line in get(url + ".index").decode().splitlines():
            d = json.loads(line)
            if (key := (d["param"], d["levtype"], d.get("levelist"))) in want:
                spans.append((d["_offset"], d["_offset"] + d["_length"] - 1))
                found.add(key)
        # a field a model never publishes drops it from every model's sample (verify.common_sample), so say so
        missing = [] if upper else sorted(k[0] for k in ECMWF_FIELDS - found if not (k[0] == "tp" and lead == 0))
        if missing:
            print(f"::warning::{model} {init} +{lead}h: no {', '.join(missing)} in the index", flush=True)
        url += ".grib2"
    with open(part, "wb") as fh:
        for a, b in merged(spans):
            fh.write(get(url, f"{a}-{b}"))
    part.rename(dest)


def pressure_variable(init: str) -> str:
    """'sp' when both ECMWF models publish surface pressure for the run (GFS and GDPS always do), else 'msl': a variable
    one model lacks would drop out of every model's sample (verify.common_sample)."""
    day, hh = init[:8], init[8:]
    for model in ("IFS", "AIFS"):
        index = get(f"{GOOGLE}/{day}/{hh}z/{MODELS[model]}/0p25/oper/{day}{hh}0000-12h-oper-fc.index").decode()
        if not any(json.loads(line)["param"] == "sp" for line in index.splitlines()):
            print(f"{model} publishes no surface pressure: scoring mean sea-level pressure", flush=True)
            return "msl"
    return "sp"


def download(jobs: list[tuple[str, str, int]], root: Path, upper: bool) -> None:
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(*j, root / MODELS[j[0]] / f"{j[1]}_{j[2]:03d}.grib2", upper), jobs))


def upper_pairs(init: str, work: Path, store) -> Path:
    """Radiosonde pairs of every model for one run, saved as Parquet; the run's files are deleted after."""
    from mverify.coarsen import coarsen
    from mverify.io.open import open_forecast
    from mverify.match import match

    out = work / "pairs" / f"{init}.parquet"
    if out.exists():
        return out
    root = work / "upper"
    download([(m, init, ld) for m in MODELS for ld in LEADS], root, upper=True)
    pairs = pd.concat(
        [
            match(
                coarsen(open_forecast(sorted((root / d).glob(f"{init}_*.grib2")), name=m), GRID),
                store,
                networks=["RAOB"],
            )
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
    ap.add_argument("--clean", action="store_true", help="delete the surface files once scored (small disks)")
    a = ap.parse_args()
    inits = [
        t.strftime("%Y%m%d%H")
        for t in pd.date_range(
            pd.to_datetime(a.inits[0], format="%Y%m%d%H"), pd.to_datetime(a.inits[1], format="%Y%m%d%H"), freq="1D"
        )
    ]
    pressure = pressure_variable(inits[0])
    ecmwf, gfs, gdps = PRESSURE[pressure]
    ECMWF_FIELDS.add(ecmwf)
    GFS_FIELDS.append(gfs)
    GDPS_FIELDS.append(gdps)
    MAP_VARS.insert(2, pressure)
    download([(m, i, ld) for m in MODELS for i in inits for ld in LEADS], a.work, upper=False)

    from mverify import report
    from mverify.coarsen import coarsen
    from mverify.db.store import ObsStore
    from mverify.io.open import open_forecast
    from mverify.precip.maps import cell_scores
    from mverify.verify import Result, score_pair_files, verify

    with ObsStore.open(a.db, read_only=True) as store:
        surface_files = {
            m: sorted(f for i in inits for f in (a.work / d).glob(f"{i}_*.grib2")) for m, d in MODELS.items()
        }
        surface = verify(surface_files, store, networks=["GHCNH"], grid=GRID)
        # rain maps from the same files, on the same grid, before --clean removes them
        maps = {m: cell_scores(coarsen(open_forecast(fs, name=m), GRID), store) for m, fs in surface_files.items()}
        maps = {m: c for m, c in maps.items() if c is not None}
        if a.clean:
            for f in (f for fs in surface_files.values() for f in fs):
                f.unlink()
        files = []
        for init in inits:
            files.append(upper_pairs(init, a.work, store))
            print(f"{init}: radiosonde pairs saved", flush=True)
    res = Result.concat([surface, score_pair_files(files)])
    report.write_json(res.scorecard(), OUT)
    report.write_json(report.station_map(res.stations, MAP_LEADS, MAP_VARS), MAP_OUT)
    if maps:
        report.write_json(report.precip_maps(maps), RAIN_OUT)
    print(f"{len(surface.pairs):,} surface pairs, {OUT.relative_to(REPO)}")
    stations = surface.pairs.groupby("variable")["station_id"].nunique()
    within = " (model terrain within 5 m)" if pressure == "sp" else ""
    print(
        f"stations scored: 2 m temperature {stations.get('t2m', 0):,}, {pressure} {stations.get(pressure, 0):,}{within}",
        flush=True,
    )


if __name__ == "__main__":
    main()
