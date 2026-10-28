#!/usr/bin/env python3
"""GFS, ECMWF IFS, AIFS and ECCC GDPS scored with mverify for verification.html, and the verification database.

Downloads only the fields that are scored (byte ranges from each file's index), every 12 h to day 10:
  * surface: 2 m temperature and dew point, mean sea-level pressure, 10 m wind and precipitation, plus orography and
    the land-sea mask at step 0, against every GHCNh station and against Stage IV over the US;
  * upper air: height, temperature and wind at 1000-100 hPa and specific humidity at 1000-300 hPa
    (mverify.variables.LEVELS, HUMIDITY_LEVELS), scored against radiosondes. These are some 50 fields a
    step, so each run is downloaded, matched to the soundings and deleted before the next; its pairs are
    kept as Parquet in <work>/pairs. A run whose pairs are already there is not downloaded again (the
    nightly workflow carries the directory over); the two newest runs are matched afresh, since soundings
    of their last days can still arrive, and pairs of runs outside the window are deleted.
ECMWF comes from the Google Cloud mirror only (as everywhere on the site, see scripts/ecmwf/rangefetch.py);
GFS from NOAA on AWS; GDPS from ECCC's HPFX server, which keeps about 50 days and has one file per field.
Every model is first averaged onto one 0.5 deg grid (mverify.coarsen, by area overlap, with its terrain and land-sea
mask), so the 0.15 deg GDPS and the 0.25 deg models are interpolated to the stations from cells of the same size.

Every pair is kept (``matched``): mverify.match's cuts (model terrain more than 100 m from the station's, mean
sea-level pressure where the model terrain is over 500 m, thermometers outside 1.25-2.5 m) only mark a pair as not
``kept``, and the scores use the kept ones. All models are scored on one sample with paired significance tests over
the newest ten runs whose day 10 is observed: the scorecard to assets/verification/scores.json, the station maps to
stations.json, and 10 m wind speed as a distribution (QQ quantiles, a hybrid Weibull fit, exceedance;
mverify.distribution) to wind_dist.json.

The database: once a run is --mature-days old (14: its day 10 is observed and the last soundings and station reports
are in), it is written to --archive-out and the workflow adds it to the monthly release verify-db-YYYY-MM, where it is
kept for good: every model's pairs against the surface stations (sfc_<init>.parquet), the radiosondes (raob_) and
Stage IV (rain_), and their sums (stats_, stn_; ``run_stats``), from which any window of runs is scored without its
pairs. The newest 30, 90 and 365 runs and the whole database are scored from those (scores_30.json ...
scores_all.json, stations_30.json, stations_90.json; ``windows``), and its newest year of rain against Stage IV is
mapped on 2 deg boxes, the bias tested box by box (precip_maps.json, ``rain_map``). --archive-only adds runs without scoring
(.github/workflows/verify-backfill.yml): a model a source no longer holds for a run (GDPS after ~50 days) is left out
of that run.

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

import numpy as np
import pandas as pd
from mverify import variables as V

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "assets" / "verification"
MAP_LEADS = [24, 72, 120, 168, 240]
MAP_VARS = ["t2m", "td2m", "msl", "wind10"] + [f"{b}{lv}" for lv in (850, 500, 250) for b in ("z", "t", "wind")]
MAP_VARS += ["q850", "q700"]
WINDOWS = {30: True, 90: True, 365: False, None: False}  # runs of a window (None: all) -> with station maps
RAIN_BOX = 2.0  # degrees: the rain maps' boxes, 16 cells of the common grid
RAIN_RUNS, RAIN_MIN_RUNS = 365, 30  # the rain maps' window: the newest year of the database, once it has a month
WINDOW_SHARE = 0.95  # a model is scored in a window that it has at least this share of the runs of
PAIR_KEY = ["model", "init", "lead_h", "valid_time", "station_id", "variable"]
STAT_BY = ["model", "network", "variable", "region", "lead_h"]
SUMS = ["n", "se", "sse", "sae", "sf", "so", "sff", "soo", "sfo", "sov"]
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
HPFX_DAYS = 52  # HPFX keeps about 50 days: a GDPS run older than this is not asked for
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
    err = None
    for attempt in range(8):
        try:
            headers = {"User-Agent": UA, **({"Range": f"bytes={rng}"} if rng else {})}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            err = e  # HTTPException covers a body cut off mid-transfer (IncompleteRead)
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"gave up on {url}: {err!r}")


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
    """Every (model, init, lead) into ``root``. A run a source does not hold (GDPS beyond HPFX's ~50 days, not asked
    for; a 404, as for AIFS before it was published) leaves that model out of that run: its files of the run are
    removed, so the run is scored with the models that have it."""
    oldest = f"{pd.Timestamp.now('UTC') - pd.Timedelta(days=HPFX_DAYS):%Y%m%d}"
    missing = {(m, init) for m, init, _ in jobs if m == "GDPS" and init[:8] < oldest}

    def one(job: tuple[str, str, int]) -> None:
        try:
            fetch(*job, root / MODELS[job[0]] / f"{job[1]}_{job[2]:03d}.grib2", upper)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):  # a bucket answers 403 for a key it does not hold
                raise
            missing.add(job[:2])

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, [j for j in jobs if j[:2] not in missing]))
    for m, init in sorted(missing):
        print(f"::warning::{m} {init}: not on the source, left out of this run", flush=True)
        for f in (root / MODELS[m]).glob(f"{init}_*.grib2"):
            f.unlink()


def opened(files: list[Path], model: str):
    """A model's files, opened and averaged onto the common grid."""
    from mverify.coarsen import coarsen
    from mverify.io.open import open_forecast

    return coarsen(open_forecast(files, name=model), GRID)


def by_init(fc) -> dict:
    """A forecast of several runs as one per run (YYYYMMDDHH), so each run's pairs are written as it is matched."""
    from mverify.io.open import Forecast

    runs: dict = {}
    for f in fc.fields:
        runs.setdefault(pd.Timestamp(f.init).strftime("%Y%m%d%H"), []).append(f)
    return {init: Forecast(fc.name, fields) for init, fields in sorted(runs.items())}


def matched(fc, store, networks: list[str]) -> pd.DataFrame:
    """Every pair of ``fc`` with the observations of ``networks``: mverify.match without its cuts (terrain, thermometer
    height, mean sea-level pressure over high ground), so that the database holds them all, and the column ``kept``
    marking the pairs its default cuts keep, the ones that are scored."""
    from mverify.match import match

    every = match(fc, store, networks=networks, max_dz=None, t_height_range=None)
    kept = match(fc, store, networks=networks)[PAIR_KEY].drop_duplicates()
    hit = every[PAIR_KEY].merge(kept, on=PAIR_KEY, how="left", indicator=True)["_merge"].eq("both")
    return every.assign(kept=hit.to_numpy())


def rain_pairs(fc, store, product: str = "stageiv") -> pd.DataFrame:
    """The 24 h rain of ``fc`` against ``product`` cell by cell on the common grid, every window the analysis has
    inside the forecast's leads: the analysis averaged onto the grid where it covers at least half a cell
    (mverify.precip.maps.MIN_COVER), one row per model, run, window and cell (and member, for an ensemble) with the
    window's end as lead_h and valid_time, fcst and obs in mm and the share of the cell the analysis covers."""
    from mverify.precip.accum import by_run
    from mverify.precip.analyses import PRODUCTS
    from mverify.precip.maps import MIN_COVER
    from mverify.precip.regrid import Regridder, analysis_grid

    runs = by_run([f for f in fc.fields if f.var == "tp"])
    if not runs:
        return pd.DataFrame()
    spec = PRODUCTS[product]
    span = pd.Timedelta(hours=spec.accum_hours)
    inits = sorted({init for init, _, _ in runs})
    grid = next(iter(runs.values())).grid
    lat, lon = np.meshgrid(grid.lat, ((grid.lon + 180) % 360) - 180, indexing="ij")
    max_lead = max(f.lead for f in fc.fields if f.var == "tp")
    regridders: dict[tuple, Regridder] = {}
    rows = []
    for end in store.grid_times(product, spec.accum_hours, inits[0] + span, inits[-1] + max_lead):
        truth = store.get_grid(product, spec.accum_hours, end)
        tgrid = analysis_grid(truth)
        key = (tgrid.lat.shape, float(tgrid.lat.flat[0]), float(tgrid.lon.flat[0]))
        rg = regridders.setdefault(key, Regridder(tgrid, grid))
        cover = rg.coverage(truth.values)
        obs = np.where(cover >= MIN_COVER, rg(truth.values), np.nan)
        for (init, member, _grid), acc in runs.items():
            a, b = end - span - init, end - init
            if a < pd.Timedelta(0) or (total := acc.total(a, b)) is None:
                continue
            ok = np.isfinite(obs) & np.isfinite(total)
            rows.append(
                pd.DataFrame(
                    {
                        "model": fc.name,
                        "init": init,
                        **({} if member is None else {"member": member}),
                        "lead_h": b / pd.Timedelta("1h"),
                        "valid_time": end,
                        "lat": lat[ok],
                        "lon": lon[ok],
                        "fcst": total[ok],
                        "obs": obs[ok],
                        "cover": cover[ok],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def combine(parts: dict[str, list[Path]], out: Path) -> dict[str, Path]:
    """Each run's files of one model each as one Parquet file per run in ``out``; the parts are deleted."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for init, files in parts.items():
        paths[init] = out / f"{init}.parquet"
        table = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")
        pq.write_table(table, paths[init], compression="zstd")
        for f in files:
            f.unlink()
    return paths


def upper_pairs(init: str, work: Path, store) -> Path:
    """Radiosonde pairs of every model for one run, saved as Parquet; the run's files are deleted after."""
    out = work / "pairs" / f"{init}.parquet"
    if out.exists():
        return out
    root = work / "upper"
    print(f"{init}: downloading the pressure levels", flush=True)
    download([(m, init, ld) for m in MODELS for ld in LEADS], root, upper=True)
    pairs = pd.concat(
        [
            matched(opened(fs, m), store, ["RAOB"])
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


def surface_pairs(inits: list[str], work: Path, store, scores: bool) -> tuple[dict, dict, list]:
    """From the downloaded surface files, each run's surface-station pairs of every model (<work>/sfc/<init>.parquet)
    and its rain against Stage IV on the common grid (<work>/rain/<init>.parquet, ``rain_pairs``); with ``scores``,
    the Stage IV scores of all the runs too (mverify.precip.verify)."""
    from mverify.precip.verify import verify_precip

    parts: dict[str, dict[str, list[Path]]] = {"sfc": {}, "rain": {}}
    (work / "parts").mkdir(parents=True, exist_ok=True)
    tables = []
    for m, d in MODELS.items():
        files = sorted(f for i in inits for f in (work / d).glob(f"{i}_*.grib2"))
        if not files:
            continue
        fc = opened(files, m)
        if scores:
            tables.append(verify_precip(fc, store))
        for init, run in by_init(fc).items():
            for kind, frame in (("sfc", matched(run, store, ["GHCNH"])), ("rain", rain_pairs(run, store))):
                if not frame.empty:
                    parts[kind].setdefault(init, []).append(work / "parts" / f"{kind}_{init}_{d}.parquet")
                    frame.to_parquet(parts[kind][init][-1], compression="zstd")
    sfc, rain = (combine(parts[k], work / k) for k in ("sfc", "rain"))
    return sfc, rain, [t for t in tables if not t.empty]


def kept_only(paths: list[Path], dest: Path) -> list[Path]:
    """Copies of pair files holding only the ``kept`` pairs (``matched``), the ones scored."""
    import pyarrow.parquet as pq

    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        table = pq.read_table(p)
        out.append(dest / f"{p.parent.name}_{p.name}")
        pq.write_table(table.filter(table["kept"]), out[-1], compression="zstd")
    return out


def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sums(rp: pd.DataFrame) -> pd.DataFrame:
    f, o = rp["fcst"], rp["obs"]
    e = f - o
    d = rp.assign(
        n=1, se=e, sse=e * e, sae=e.abs(), sf=f, so=o, sff=f * f, soo=o * o, sfo=f * o, sov=rp["sigma_o"] ** 2
    )
    return d.groupby(STAT_BY, sort=False)[SUMS].sum(min_count=1).reset_index()


def _wind_sums(rp: pd.DataFrame) -> pd.DataFrame:
    """The vector wind of the u/v pairs as in mverify.scores.vector_wind: ``se`` sums the speed error, ``sse`` the
    squared vector error, ``sov`` the error variance of the observed vector."""
    key = ["model", "network", "region", "init", "lead_h", "valid_time", "station_id"]
    out = []
    for var_u in sorted(v for v in rp["variable"].unique() if v.startswith("u")):
        u, v = rp[rp["variable"] == var_u], rp[rp["variable"] == "v" + var_u[1:]]
        j = u.merge(v[[*key, "fcst", "obs", "sigma_o"]], on=key, suffixes=("_u", "_v"))
        if j.empty:
            continue
        j = j.assign(
            variable="wind" + var_u[1:],
            n=1,
            se=np.hypot(j["fcst_u"], j["fcst_v"]) - np.hypot(j["obs_u"], j["obs_v"]),
            sse=(j["fcst_u"] - j["obs_u"]) ** 2 + (j["fcst_v"] - j["obs_v"]) ** 2,
            sov=j["sigma_o_u"] ** 2 + j["sigma_o_v"] ** 2,
        )
        out.append(j.groupby(STAT_BY, sort=False)[["n", "se", "sse", "sov"]].sum(min_count=1).reset_index())
    return _cat(out)


def run_stats(paths: list[Path], init: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One run's scores as sums that add over runs, so that any window of the database is scored from these alone
    (``window_card``, ``window_map``). On the kept pairs (``matched``) that every model of the run has
    (mverify.verify.common_sample, one variable group at a time as in mverify.verify.score_pair_files): per model,
    network, variable, region and lead, the count and the sums of the error, its square and absolute value, the
    forecast, the observation, their squares and product and the observation-error variance (``SUMS``; u/v pairs
    also as the vector wind, ``_wind_sums``); and per station at the map leads, the count, median position and the
    sums of the error and its square (mverify.scores.by_station)."""
    import pyarrow.dataset as ds
    from mverify import scores
    from mverify.verify import common_sample, variable_group

    data = ds.dataset([str(p) for p in paths], format="parquet")
    groups: dict[str, list[str]] = {}
    for name in sorted(data.to_table(columns=["variable"]).column("variable").unique().to_pylist()):
        groups.setdefault(variable_group(name), []).append(name)
    stats, stations = [], []
    for members in groups.values():
        p = common_sample(data.to_table(filter=ds.field("variable").isin(members) & ds.field("kept")).to_pandas())
        if p.empty:
            continue
        for region in scores.REGIONS:
            if not (rp := scores.assign_regions(p, [region])).empty:
                stats += [_sums(rp), _wind_sums(rp)]
        st = scores.by_station(p[p["lead_h"].isin(MAP_LEADS)])
        st = st[st["variable"].isin(MAP_VARS)]
        stations.append(
            st.assign(se=st["n"] * st["bias"], sse=st["n"] * st["rmse"] ** 2).drop(columns=["rmse", "bias"])
        )
    when = pd.to_datetime(init, format="%Y%m%d%H")
    return _cat(stats).assign(init=when), _cat(stations).assign(init=when)


def runs(first: str, last: str) -> list[str]:
    """The 00Z runs from ``first`` to ``last`` (YYYYMMDDHH)."""
    a, b = (pd.to_datetime(x, format="%Y%m%d%H") for x in (first, last))
    return [t.strftime("%Y%m%d%H") for t in pd.date_range(a, b, freq="1D")]


def held(init: str, have: set[str]) -> bool:
    """Whether the database has the run: its pairs and their sums (rain pairs only where Stage IV covers it)."""
    return {f"{k}_{init}.parquet" for k in ("sfc", "raob", "stats", "stn")} <= have


def archive(inits: list[str], work: Path, sfc: dict, rain: dict, dest: Path, have: set[str]) -> list[str]:
    """Each run's files of the database, sfc_, raob_ and rain_<init>.parquet (the pairs) and stats_ and stn_ (their
    sums, ``run_stats``), into ``dest`` unless the database ``have``s them; the workflow uploads them to the monthly
    release verify-db-YYYY-MM."""
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for init in inits:
        pairs = {"sfc": sfc.get(init), "raob": work / "pairs" / f"{init}.parquet"}
        if not all(f and f.exists() for f in pairs.values()):
            print(f"::warning::{init}: not archived, its pairs are incomplete", flush=True)
            continue
        if held(init, have):
            continue
        if init not in rain:
            print(f"::warning::{init}: no Stage IV windows, archived without rain pairs", flush=True)
        for kind, f in {**pairs, "rain": rain.get(init)}.items():
            if f:
                shutil.copyfile(f, dest / f"{kind}_{init}.parquet")
        for kind, frame in zip(("stats", "stn"), run_stats(list(pairs.values()), init), strict=True):
            keys = [c for c in (*STAT_BY[:4], "station_id") if c in frame]  # categories: a year of runs fits in memory
            frame.astype(dict.fromkeys(keys, "category")).to_parquet(
                dest / f"{kind}_{init}.parquet", compression="zstd"
            )
        done.append(init)
    return done


def window_runs(stats: pd.DataFrame, share: float = WINDOW_SHARE) -> tuple[list[str], list, dict[str, int]]:
    """The models a window scores, those with at least ``share`` of its runs, and the runs that all of them have, so
    that every model is judged on the same runs; with the runs of each model left out (GDPS before the database
    began, say)."""
    runs = stats["init"].nunique()
    have = stats.groupby("model", observed=True)["init"].nunique()
    models = sorted(m for m, k in have.items() if k >= share * runs)
    per_run = stats[stats["model"].isin(models)].groupby("init")["model"].nunique()
    return (
        models,
        sorted(per_run[per_run == len(models)].index),
        {m: int(k) for m, k in have.items() if m not in models},
    )


def run_span(runs: list) -> dict:
    """How many ``runs`` (sorted) and the first and last, YYYYMMDDHH."""
    first, last = (pd.Timestamp(i).strftime("%Y%m%d%H") for i in (runs[0], runs[-1]))
    return {"runs": len(runs), "first": first, "last": last}


def window_card(stats: pd.DataFrame) -> dict | None:
    """The scorecard (mverify.report.scorecard) of the runs in ``stats`` (``run_stats`` rows): the scores from the
    summed sums, and for every pair of models the paired test of mverify.significance on the per-run mean squared
    errors, with its allowance for errors that persist from run to run."""
    from mverify import report
    from mverify.significance import BY, effective_runs, paired_test

    models, used, left_out = window_runs(stats)
    if not used:  # no run has every model the window would score
        return None
    s = stats[stats["model"].isin(models) & stats["init"].isin(used)]
    t = s.groupby(STAT_BY, observed=True)[SUMS].sum(min_count=1)
    n = t["n"]
    mse = t["sse"] / n
    with np.errstate(invalid="ignore", divide="ignore"):
        table = pd.DataFrame(
            {
                "n": n,
                "bias": t["se"] / n,
                "mae": t["sae"] / n,
                "rmse": np.sqrt(mse),
                "rmse_oc": np.sqrt((mse - t["sov"] / n).clip(lower=0)),
                "sde": np.sqrt(((t["sse"] - t["se"] ** 2 / n) / (n - 1)).where(n > 1)),
                "corr": (
                    (n * t["sfo"] - t["sf"] * t["so"])
                    / np.sqrt((n * t["sff"] - t["sf"] ** 2) * (n * t["soo"] - t["so"] ** 2))
                ).where(n > 2),
            }
        ).reset_index()
    table = table.astype(dict.fromkeys(STAT_BY[:4], str))  # categories in the files; mverify.report groups by name
    runs = s.assign(mse=s["sse"] / s["n"]).set_index([*BY, "init", "model"])["mse"].unstack("model").sort_index()
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for key, g in (runs[a] - runs[b]).dropna().groupby(level=BY, observed=True):
                d = g.to_numpy()
                found = (len(d), effective_runs(d), paired_test(d))
                rows += [(*key, a, b, *found), (*key, b, a, *found)]
    tests = pd.DataFrame(rows, columns=[*BY, "model", "reference", "runs", "runs_eff", "p"])
    inits = {m: [pd.Timestamp(i).strftime("%Y-%m-%dT%H") for i in used] for m in models}
    card = report.scorecard(table, inits, tests)
    card["window"] = run_span(used) | {"left_out": left_out}
    return card


def window_map(stn: pd.DataFrame, models: list[str], used: list) -> dict:
    """Per-station maps (mverify.report.station_map) of the runs ``used`` from their ``run_stats`` station sums."""
    from mverify import report
    from mverify.scores import STATION_KEY

    s = stn[stn["model"].isin(models) & stn["init"].isin(used)]
    g = s.groupby(STATION_KEY, observed=True)
    st = g[["n", "se", "sse"]].sum().join(g[["lat", "lon"]].median()).reset_index()
    st = st.astype(dict.fromkeys(STATION_KEY[:4], str))
    st = st.assign(rmse=np.sqrt(st["sse"] / st["n"]), bias=st["se"] / st["n"])
    return report.station_map(st, MAP_LEADS, MAP_VARS)


def rain_map(paths: list[Path]) -> dict | None:
    """Rain against Stage IV over the runs of ``paths`` (``rain_pairs`` files) on boxes of RAIN_BOX degrees: the
    common grid's cells pooled into boxes, which are tested where single cells never could be (a cell's daily rain
    error is mostly noise). Per model, lead group (mverify.precip.maps.LEAD_GROUPS) and box: the bias, the RMSE of
    the cells and the mean analysed rain, mm per window; and whether the bias is more than chance, tested over the
    valid days as mverify.precip.maps does for cells (the day's error averaged over the box and the runs that verify
    it, through the paired test with its allowance for errors that persist from day to day) and the boxes of a map
    through the Benjamini-Hochberg procedure. The models with WINDOW_SHARE of the runs are mapped on the runs they all
    have; one with fewer but at least RAIN_MIN_RUNS (GDPS before the database began, say) on its own, since a box's test
    needs only that model's days. As JSON shaped like mverify.report.precip_maps, plus the window and the runs of each
    model mapped on its own."""
    from mverify.precip.maps import FDR_Q, LEAD_GROUPS, MIN_DAYS
    from mverify.significance import fdr, paired_test

    cols = ["model", "init", "lead_h", "valid_time", "lat", "lon", "fcst", "obs"]
    per_run = []  # one run at a time: a year of cells would not fit in memory, its box sums do
    for f in paths:
        r = pd.read_parquet(f, columns=cols)
        r["group"] = np.select(
            [r["lead_h"].between(a, b) for _, a, b in LEAD_GROUPS], list(range(len(LEAD_GROUPS))), -1
        )
        r = r[r["group"] >= 0]
        e = r["fcst"] - r["obs"]
        r = r.assign(
            blat=np.floor(r["lat"] / RAIN_BOX) * RAIN_BOX + RAIN_BOX / 2,
            blon=np.floor(r["lon"] / RAIN_BOX) * RAIN_BOX + RAIN_BOX / 2,
            n=1,
            err=e,
            sq=e * e,
        )
        keys = ["model", "group", "valid_time", "blat", "blon"]
        per_run.append(r.groupby(keys)[["n", "err", "sq", "obs"]].sum().reset_index().assign(init=r["init"].iloc[0]))
    boxes = pd.concat(per_run, ignore_index=True)
    models, used, left_out = window_runs(boxes)
    if not used:
        return None
    own = {
        m: sorted(boxes.loc[boxes["model"] == m, "init"].unique()) for m, k in left_out.items() if k >= RAIN_MIN_RUNS
    }
    runs_of = dict.fromkeys(models, used) | own
    boxes = pd.concat([boxes[(boxes["model"] == m) & boxes["init"].isin(r)] for m, r in runs_of.items()])
    daily = boxes.groupby(["model", "group", "blat", "blon", "valid_time"])[["n", "err", "sq", "obs"]].sum()
    lats, lons = (np.sort(boxes[c].unique()) for c in ("blat", "blon"))
    out = {
        "product": "stageiv",
        "truth": "Stage IV analysis",
        "accum_hours": 24,
        "fdr_q": FDR_Q,
        "min_days": MIN_DAYS,
        "box": RAIN_BOX,
        "groups": [{"label": label, "ends_h": [a, b]} for label, a, b in LEAD_GROUPS],
        "models": {},
    }
    for m in sorted(runs_of):
        maps = {k: np.full((len(LEAD_GROUPS), lats.size, lons.size), np.nan) for k in ("bias", "rmse", "obs", "p")}
        days = np.zeros((len(LEAD_GROUPS), lats.size, lons.size), int)
        for (g, la, lo), d in daily.loc[m].groupby(level=["group", "blat", "blon"]):
            j, i = np.searchsorted(lats, la), np.searchsorted(lons, lo)
            n = d["n"].sum()
            maps["bias"][g, j, i] = d["err"].sum() / n
            maps["rmse"][g, j, i] = np.sqrt(d["sq"].sum() / n)
            maps["obs"][g, j, i] = d["obs"].sum() / n
            days[g, j, i] = len(d)
            if len(d) >= MIN_DAYS:
                maps["p"][g, j, i] = paired_test((d["err"] / d["n"]).to_numpy())
        out["models"][m] = {
            "lat": lats.round(4).tolist(),
            "lon": lons.round(4).tolist(),
            **{
                k: [[None if not np.isfinite(v) else int(np.rint(10 * v)) for v in g.ravel()] for g in maps[k]]
                for k in ("bias", "rmse", "obs")
            },
            "days": [g.ravel().tolist() for g in days],
            "significant": [fdr(g.ravel(), FDR_Q).astype(int).tolist() for g in maps["p"]],
        }
    out["window"] = run_span(used) | {
        "left_out": {m: k for m, k in left_out.items() if m not in own},
        "own": {m: run_span(r) for m, r in own.items()},
    }
    return out


def windows(derived: Path) -> None:
    """The scorecards of the newest 30, 90 and 365 runs of the database and of all of it once it is longer
    (scores_<runs>.json, scores_all.json), and station maps for the shorter ones (stations_<runs>.json), each
    once the database holds that many runs, from the runs' sums (``run_stats``) in ``derived``; and the rain maps
    of its newest year (precip_maps.json, ``rain_map``) once it holds RAIN_MIN_RUNS runs."""
    from mverify import report

    inits = sorted(f.stem[6:] for f in derived.glob("stats_*.parquet"))
    for runs, with_map in WINDOWS.items():
        if len(inits) < (runs or 366):
            continue
        use = inits[-runs:] if runs else inits
        stats = pd.read_parquet([derived / f"stats_{i}.parquet" for i in use])
        if (card := window_card(stats)) is None:
            continue
        name = runs or "all"
        report.write_json(card, OUT_DIR / f"scores_{name}.json")
        if with_map:
            models, used, _ = window_runs(stats)
            stn = pd.read_parquet([derived / f"stn_{i}.parquet" for i in use])
            report.write_json(window_map(stn, models, used), OUT_DIR / f"stations_{name}.json")
        w = card["window"]
        print(f"window {name}: {w['runs']} runs {w['first']} to {w['last']}, left out {w['left_out']}", flush=True)
    wet = sorted(derived.glob("rain_*.parquet"))[-RAIN_RUNS:]
    if len(wet) >= RAIN_MIN_RUNS and (rm := rain_map(wet)) is not None:
        report.write_json(rm, OUT_DIR / "precip_maps.json")
        print(f"rain maps: {len(wet)} runs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="mverify observation database")
    ap.add_argument("--inits", nargs=2, required=True, metavar=("FIRST", "LAST"), help="00Z runs, YYYYMMDDHH")
    ap.add_argument("--work", type=Path, required=True, help="download directory")
    ap.add_argument("--clean", action="store_true", help="delete the surface files once scored (small disks)")
    ap.add_argument("--derived", type=Path, help="the database's stats_, stn_ and rain_ files, downloaded")
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
                if held(init, have):
                    continue
                download([(m, init, ld) for m in MODELS for ld in LEADS], a.work, upper=False)
                sfc, rain, _ = surface_pairs([init], a.work, store, scores=False)
                for f in (f for d in MODELS.values() for f in (a.work / d).glob(f"{init}_*.grib2")):
                    f.unlink()
                upper_pairs(init, a.work, store)
                print(f"{init}: archived {archive([init], a.work, sfc, rain, a.archive_out, have)}", flush=True)
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
        sfc, rain, tables = surface_pairs(inits, a.work, store, scores=True)
        if a.clean:
            for f in (f for d in MODELS.values() for f in (a.work / d).glob("*.grib2")):
                f.unlink()
        upper = []
        for init in inits:
            upper.append(upper_pairs(init, a.work, store))
            print(f"{init}: radiosonde pairs saved", flush=True)
    kept_sfc = kept_only(list(sfc.values()), a.work / "kept")
    res = score_pair_files(kept_sfc + kept_only(upper, a.work / "kept"))
    if tables:  # Stage IV rows beside the station rows, as mverify.verify.verify joins them
        res.table = pd.concat([res.table, *tables], ignore_index=True)
    report.write_json(res.scorecard(), OUT_DIR / "scores.json")
    report.write_json(report.station_map(res.stations, MAP_LEADS, MAP_VARS), OUT_DIR / "stations.json")
    ws = common_sample(pd.read_parquet(kept_sfc, filters=[("variable", "==", "ws10")]))
    report.write_json(compare(ws, "ws10"), OUT_DIR / "wind_dist.json")
    print(f"scores.json: {len(inits)} runs", flush=True)

    if a.archive_out:  # the runs old enough to have all their observations join the database
        cutoff = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=a.mature_days)).strftime("%Y%m%d%H")
        added = archive([i for i in inits if i <= cutoff], a.work, sfc, rain, a.archive_out, have)
        print(f"database: {len(added)} runs added ({', '.join(added) or 'none'})", flush=True)
        if a.derived:
            for f in a.archive_out.glob("*.parquet"):
                if f.name.startswith(("stats_", "stn_", "rain_")):
                    shutil.copyfile(f, a.derived / f.name)
    if a.derived:
        windows(a.derived)


if __name__ == "__main__":
    main()
