#!/usr/bin/env python3
"""GFS, ECMWF IFS, AIFS and ECCC GDPS scored with mverify for verification.html.

Downloads only the fields that are scored (byte ranges from each file's index), every 12 h to day 10:
  * surface: 2 m temperature and dew point, mean sea-level pressure (only where the model terrain at the station is
    at most 500 m: mverify.match.MAX_TERRAIN_MSL), 10 m wind and precipitation, plus orography and the land-sea mask
    at step 0;
  * upper air: height, temperature and wind at 1000-100 hPa and specific humidity at 1000-300 hPa
    (mverify.variables.LEVELS, HUMIDITY_LEVELS), scored against radiosondes. These are some 50 fields a
    step, so each run is downloaded, matched to the soundings and deleted before the next; its pairs are
    kept as Parquet in <work>/pairs and scored level by level at the end (mverify.verify.score_pair_files).
    A run whose pairs are already there is not downloaded again (the nightly workflow carries the directory
    over); the two newest runs are matched afresh, since soundings of their last days can still arrive, and
    pairs of runs outside the window are deleted.
ECMWF comes from the Google Cloud mirror only (as everywhere on the site, see scripts/ecmwf/rangefetch.py);
GFS from NOAA on AWS; GDPS from the ECCC Datamart, which keeps about 30 days and has one file per field.
Every model is first averaged onto one 0.5 deg grid (mverify.coarsen, by area overlap, with its terrain and land-sea
mask), so the 0.15 deg GDPS and the 0.25 deg models are interpolated to the stations from cells of the same size. All
models are scored on one sample with paired significance tests, and the scorecard goes to
assets/verification/scores.json; rain against Stage IV in every cell of that grid, the bias tested cell by cell
(mverify.precip.maps), goes to assets/verification/precip_maps.json, and 10 m wind speed as a distribution (QQ
quantiles, a hybrid Weibull fit, exceedance; mverify.distribution) to assets/verification/wind_dist.json.

The database: once a run is --mature-days old (14: its day 10 is observed and the last soundings and station reports
are in), its pairs, surface (sfc_<init>.parquet) and radiosonde (raob_<init>.parquet), all models, are written to
--archive-out and the workflow adds them to the monthly release verify-db-YYYY-MM, where they are kept for good and
can be scored again. The newest 30 runs of the database are scored as assets/verification/scores_30.json and
stations_30.json. --archive-only adds a range of runs without scoring (.github/workflows/verify-backfill.yml): a
model a source no longer holds for a run (GDPS after ~30 days) is left out of that run.

    python scripts/verify/mverify_scores.py --db obs.duckdb --inits 2026092600 2026100500 --work /tmp/fc
"""

from __future__ import annotations

import argparse
import http.client
import json
import shutil
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
WIND_OUT = REPO / "assets" / "verification" / "wind_dist.json"
OUT30 = REPO / "assets" / "verification" / "scores_30.json"
MAP30_OUT = REPO / "assets" / "verification" / "stations_30.json"
MAP_LEADS = [24, 72, 120, 168, 240]
MAP_VARS = ["t2m", "td2m", "msl", "wind10"] + [f"{b}{lv}" for lv in (850, 500, 250) for b in ("z", "t", "wind")]
MAP_VARS += ["q850", "q700"]
LEADS = range(0, 241, 12)
GRID = 0.5  # degrees: the common grid, coarser than every model's own resolution (AIFS, the coarsest, is ~31 km)
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
DATAMART = "https://hpfx.collab.science.gc.ca"
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
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            pass  # HTTPException covers a body cut off mid-transfer (IncompleteRead)
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


def download(jobs: list[tuple[str, str, int]], root: Path, upper: bool) -> None:
    """Every (model, init, lead) into ``root``. A run a source does not hold (404: GDPS beyond the Datamart's ~30 days,
    AIFS before it was published) leaves that model out of that run: its files of the run are removed, so the run is
    scored with the models that have it."""
    missing: set[tuple[str, str]] = set()

    def one(job: tuple[str, str, int]) -> None:
        try:
            fetch(*job, root / MODELS[job[0]] / f"{job[1]}_{job[2]:03d}.grib2", upper)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):  # a bucket answers 403 for a key it does not hold
                raise
            missing.add(job[:2])

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    for m, init in sorted(missing):
        print(f"::warning::{m} {init}: not on the source, left out of this run", flush=True)
        for f in (root / MODELS[m]).glob(f"{init}_*.grib2"):
            f.unlink()


def opened(files: list[Path], model: str):
    """A model's files, opened and averaged onto the common grid."""
    from mverify.coarsen import coarsen
    from mverify.io.open import open_forecast

    return coarsen(open_forecast(files, name=model), GRID)


def upper_pairs(init: str, work: Path, store) -> Path:
    """Radiosonde pairs of every model for one run, saved as Parquet; the run's files are deleted after."""
    from mverify.match import match

    out = work / "pairs" / f"{init}.parquet"
    if out.exists():
        return out
    root = work / "upper"
    print(f"{init}: downloading the pressure levels", flush=True)
    download([(m, init, ld) for m in MODELS for ld in LEADS], root, upper=True)
    pairs = pd.concat(
        [
            match(opened(fs, m), store, networks=["RAOB"])
            for m, d in MODELS.items()
            if (fs := sorted((root / d).glob(f"{init}_*.grib2")))
        ],
        ignore_index=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out.with_suffix(".part"), compression="zstd")
    out.with_suffix(".part").rename(out)
    for f in root.glob(f"*/{init}_*.grib2"):
        f.unlink()
    return out


def surface_pairs(inits: list[str], work: Path, store, rain: bool) -> tuple[dict[str, Path], list, dict]:
    """Surface-station pairs of every model, one Parquet per run in <work>/sfc, from the downloaded surface files;
    with ``rain``, from the same files the precipitation scores against Stage IV and the per-cell rain maps."""
    from mverify.match import match
    from mverify.precip.maps import cell_scores
    from mverify.precip.verify import verify_precip

    per: dict[str, list[pd.DataFrame]] = {}
    tables, maps = [], {}
    for m, d in MODELS.items():
        files = sorted(f for i in inits for f in (work / d).glob(f"{i}_*.grib2"))
        if not files:
            continue
        fc = opened(files, m)
        if rain:
            tables.append(verify_precip(fc, store))
            if (c := cell_scores(fc, store)) is not None:
                maps[m] = c
        for init, g in match(fc, store, networks=["GHCNH"]).groupby("init"):
            per.setdefault(pd.Timestamp(init).strftime("%Y%m%d%H"), []).append(g)
    out = work / "sfc"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for init, frames in per.items():
        paths[init] = out / f"{init}.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(paths[init], compression="zstd")
    return paths, [t for t in tables if not t.empty], maps


def runs(first: str, last: str) -> list[str]:
    """The 00Z runs from ``first`` to ``last`` (YYYYMMDDHH)."""
    a, b = (pd.to_datetime(x, format="%Y%m%d%H") for x in (first, last))
    return [t.strftime("%Y%m%d%H") for t in pd.date_range(a, b, freq="1D")]


def archive(inits: list[str], work: Path, sfc: dict[str, Path], dest: Path, have: set[str]) -> list[str]:
    """Copy each run's pairs into ``dest`` as the database's files (sfc_<init>.parquet, raob_<init>.parquet) unless the
    database ``have``s them; the workflow uploads them to the monthly release verify-db-YYYY-MM."""
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for init in inits:
        files = {"sfc": sfc.get(init), "raob": work / "pairs" / f"{init}.parquet"}
        if not all(f and f.exists() for f in files.values()):
            print(f"::warning::{init}: not archived, its pairs are incomplete", flush=True)
            continue
        if {f"sfc_{init}.parquet", f"raob_{init}.parquet"} <= have:
            continue
        for kind, f in files.items():
            shutil.copyfile(f, dest / f"{kind}_{init}.parquet")
        done.append(init)
    return done


def window_scores(db: Path, n: int, out: Path, map_out: Path) -> None:
    """The scorecard and station maps of the newest ``n`` runs in the database (files in ``db``), every model on the
    observations every model has: a run without one of the models (GDPS before the database began) adds nothing."""
    from mverify import report
    from mverify.verify import score_pair_files

    inits = sorted({f.stem[4:] for f in db.glob("sfc_*.parquet")} & {f.stem[5:] for f in db.glob("raob_*.parquet")})[
        -n:
    ]
    if not inits:
        print(f"no runs in the database for the {n}-run window", flush=True)
        return
    res = score_pair_files([db / f"{k}_{i}.parquet" for i in inits for k in ("sfc", "raob")])
    card = res.scorecard()
    card["window"] = {"runs": len(inits), "first": inits[0], "last": inits[-1]}
    report.write_json(card, out)
    report.write_json(report.station_map(res.stations, MAP_LEADS, MAP_VARS), map_out)
    print(f"{n}-run window: {len(inits)} runs, {inits[0]} to {inits[-1]}, {out.relative_to(REPO)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="mverify observation database")
    ap.add_argument("--inits", nargs=2, required=True, metavar=("FIRST", "LAST"), help="00Z runs, YYYYMMDDHH")
    ap.add_argument("--work", type=Path, required=True, help="download directory")
    ap.add_argument("--clean", action="store_true", help="delete the surface files once scored (small disks)")
    ap.add_argument("--pairs-db", type=Path, help="the database's pair files already downloaded (sfc_*, raob_*)")
    ap.add_argument("--have", type=Path, help="names of the files the database holds, one per line")
    ap.add_argument("--archive-out", type=Path, help="where the runs to add to the database are written")
    ap.add_argument(
        "--mature-days", type=int, default=14, help="a run joins the database this many days after it starts"
    )
    ap.add_argument("--archive-only", action="store_true", help="only add the runs to the database (the backfill)")
    a = ap.parse_args()
    inits = runs(*a.inits)
    have = set(a.have.read_text().split()) if a.have and a.have.exists() else set()

    from mverify.db.store import ObsStore

    if a.archive_only:  # one run at a time, so a runner's disk holds any length of backfill
        with ObsStore.open(a.db, read_only=True) as store:
            for init in inits:
                if {f"sfc_{init}.parquet", f"raob_{init}.parquet"} <= have:
                    continue
                download([(m, init, ld) for m in MODELS for ld in LEADS], a.work, upper=False)
                sfc, _, _ = surface_pairs([init], a.work, store, rain=False)
                for f in (f for d in MODELS.values() for f in (a.work / d).glob(f"{init}_*.grib2")):
                    f.unlink()
                upper_pairs(init, a.work, store)
                print(f"{init}: archived {archive([init], a.work, sfc, a.archive_out, have)}", flush=True)
        return

    reuse = set(inits[:-2])  # the two newest are matched again: soundings of their last days can still arrive
    for f in (a.work / "pairs").glob("*.parquet"):
        if f.stem not in reuse:
            f.unlink()
    download([(m, i, ld) for m in MODELS for i in inits for ld in LEADS], a.work, upper=False)

    from mverify import report
    from mverify.distribution import compare
    from mverify.verify import common_sample, score_pair_files

    with ObsStore.open(a.db, read_only=True) as store:
        sfc, rain, maps = surface_pairs(inits, a.work, store, rain=True)
        if a.clean:
            for f in (f for d in MODELS.values() for f in (a.work / d).glob("*.grib2")):
                f.unlink()
        upper = []
        for init in inits:
            upper.append(upper_pairs(init, a.work, store))
            print(f"{init}: radiosonde pairs saved", flush=True)
    res = score_pair_files(list(sfc.values()) + upper)
    if rain:  # Stage IV rows beside the station rows, as mverify.verify.verify joins them
        res.table = pd.concat([res.table, *rain], ignore_index=True)
    report.write_json(res.scorecard(), OUT)
    report.write_json(report.station_map(res.stations, MAP_LEADS, MAP_VARS), MAP_OUT)
    if maps:
        report.write_json(report.precip_maps(maps), RAIN_OUT)
    ws = common_sample(pd.read_parquet(list(sfc.values()), filters=[("variable", "==", "ws10")]))
    report.write_json(compare(ws, "ws10"), WIND_OUT)
    print(f"{OUT.relative_to(REPO)}: {len(inits)} runs", flush=True)

    if a.archive_out:  # the runs old enough to have all their observations join the database
        cutoff = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=a.mature_days)).strftime("%Y%m%d%H")
        added = archive([i for i in inits if i <= cutoff], a.work, sfc, a.archive_out, have)
        print(f"database: {len(added)} runs added ({', '.join(added) or 'none'})", flush=True)
        if a.pairs_db:
            for f in a.archive_out.glob("*.parquet"):
                shutil.copyfile(f, a.pairs_db / f.name)
    if a.pairs_db:
        window_scores(a.pairs_db, 30, OUT30, MAP30_OUT)


if __name__ == "__main__":
    main()
