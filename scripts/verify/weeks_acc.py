#!/usr/bin/env python3
"""Weeks 1-5 of the two days 1-35 ensemble pages, GEFS (gefs.html) and GEPS (subseasonal.html), verified against ERA5
on the weekly anomaly maps each page draws (verification.html#weeks).

  what     the ensemble-mean anomaly against 1991-2020 of 2 m temperature (scored over land), 500 hPa height and
           sea-level pressure, averaged over days 1-7, 8-14, 15-21, 22-28 and 29-35 (gefs_reforecast.WEEKS), on the
           GEFS page's 1.5 deg grid (WMO's 121 x 240, latlon.G15).
  gefs     the Monday and Thursday 00Z runs (GEPS's start days) from NOAA's ensemble-mean files (geavg, pgrb2a
           0.5 deg, each message byte-ranged by its .idx) at 12 h steps, reduced as the page reduces its members
           (gefs_daily.Acc: day d the mean of the 12Z and next 00Z fields, centred 3 x 3 boxes to 1.5 deg), and taken
           as the page's anomaly (gefs_live.anomalies): [F - M(init, lead)] - the ERA5 shift, less the operational
           drift Delta(lead, latitude). Delta is gefs_drift's estimate made against ERA5 rather than GEFS's analyses
           (the day-1 differencing removes a steady analysis offset either way), from this series' own runs in the
           page's window, those started 36 to 80 days before. wk-gefs_<init>.npz keeps a run's weekly anomalies before
           the drift and the daily zonal means of F - M that Delta is estimated from; weekly means are linear, so
           Delta is taken off when scoring.
  geps     after each GEPS run (geps.yml) its weekly anomalies as the page's weekly maps make them (forecast.anomaly
           on its own 1 deg nodes, render_maps.reref, 500 hPa's global mean removed), remapped conservatively:
           wk-geps_<init>.npz.
  era5     ERA5 at 00Z and 12Z of every day (ARCO on Google Cloud, 0.25 deg; ERA5T for the newest months), remapped
           conservatively (latlon.remap). Day d of a run is the mean of the 12Z and next 00Z samples, as GEFS's; GEPS
           averages four six-hour steps, which two samples come nearest. A complete month is wk-era5_<YYYYMM>.npz; the
           month in progress stays in the daily job's cache, a day a file.
  clim     the normal of that day mean: ERA5 1991-2020 from WeatherBench 2's 6-hourly conservative 1.5 deg ERA5, each
           slot of the 366-day calendar the mean of the days within 7 slots of it in all 30 years (z500_acc.clim's
           construction), and ERA5's land fraction on the grid. Release verify-clim-weeks-v1, built once.
  score    each run's centred anomaly correlation (each map less its area mean, cosine-of-latitude weights; WMO-No.
           485) per field, week and region, and that of persistence, the observed anomaly of the 7 days before the
           start. The mean over runs has a 90% interval from a moving-block bootstrap over weeks of starts (successive
           starts share their weather), and is published only where its test passes the Benjamini-Hochberg procedure
           at a 10% false-discovery rate over every test of the table: each mean against zero (climatology), each
           model against persistence, and GEPS against GEFS on their common starts. Fewer than MIN_WEEKS weeks of
           starts are not tested. -> assets/verification/weeks.json

Every file but the normal joins the verification database (scripts/verify/verifydb.sh: releases verify-weeks-YYYY),
and none is ever replaced.

    python weeks_acc.py clim --out clim_weeks_1p5.npz
    python weeks_acc.py plan --have have.txt --months 3               # the backfill's jobs, for the workflow
    python weeks_acc.py era5 --store DIR --have have.txt --archive-out db-new [--first YYYYMMDD --last YYYYMMDD]
    python weeks_acc.py gefs --cache DIR --have have.txt --archive-out db-new [--first YYYYMMDD --last YYYYMMDD]
    python weeks_acc.py geps --cycle YYYYMMDD --archive-out db-new    # in geps.yml, from the run's files
    python weeks_acc.py missing --store DIR --have have.txt           # the database's files the store lacks
    python weeks_acc.py score --store DIR --clim clim_weeks_1p5.npz --out assets/verification/weeks.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "gefs"))
import gefs_daily as G  # (the daily reduction, int16 packing)
import gefs_reforecast as R  # (weeks, .idx reads, GRIB decoding)
import webget
from latlon import G15, remap, slot366

TAGS = ("t2m", "zg500", "mslp")  # units here: K, m, hPa
NAMES = {"t2m": "2 m temperature (land)", "zg500": "500 hPa height", "mslp": "Sea-level pressure"}
GRAV = 9.80665
ARCO = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2 = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ERA5_VARS = {"t2m": ("2m_temperature", None), "zg500": ("geopotential", 500), "mslp": ("mean_sea_level_pressure", None)}
CLIM_YEARS = (1991, 2020)
CLIM_HALF = 7  # slots either side of each slot of the normal (z500_acc.CLIM_HALF)
STARTS = (0, 3)  # Monday and Thursday, GEPS's extended runs
FIRST_INIT = dt.date(2025, 8, 18)  # the first run in the drift window of the first scored one
SCORE_FROM = dt.date(2025, 11, 1)
GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
MIN_DRIFT = 5  # runs a drift estimate needs: a third of the window's, as gefs_drift asks 15 of its 45
MIN_WEEKS = 16  # weeks of starts a mean needs to be tested
BLOCK = 4  # weeks of starts per bootstrap block
NBOOT = 10000
LEVEL, FDR = 0.90, 0.10


def _have(path: Path | None) -> set[str]:
    return set(path.read_text().split()) if path and path.exists() else set()


def _date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def _today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def months(first: dt.date, last: dt.date) -> list[dt.date]:
    out, m = [], first.replace(day=1)
    while m <= last:
        out.append(m)
        m = (m + dt.timedelta(days=32)).replace(day=1)
    return out


def month_days(m: dt.date) -> list[dt.date]:
    nxt = (m + dt.timedelta(days=32)).replace(day=1)
    return [m + dt.timedelta(days=k) for k in range((nxt - m).days)]


def starts(first: dt.date, last: dt.date) -> list[dt.date]:
    return [d for k in range((last - first).days + 1) if (d := first + dt.timedelta(days=k)).weekday() in STARTS]


def era5_file(m: dt.date) -> str:
    return f"wk-era5_{m:%Y%m}.npz"


def run_file(model: str, init: dt.date) -> str:
    return f"wk-{model.lower()}_{init:%Y%m%d}00.npz"


# --------------------------------------------------------------------------- ERA5
class Zarr:
    """A public zarr (v2) store on Google Cloud read over HTTPS a chunk at a time, side by side (xarray reads a few at a
    time), from each array's own .zarray layout."""

    def __init__(self, path: str):
        self.url = f"https://storage.googleapis.com/{path}"
        self._meta: dict[str, tuple] = {}

    def attrs(self, name: str = "") -> dict:
        return json.loads(webget.get(f"{self.url}/{name + '/' if name else ''}.zattrs"))

    def meta(self, name: str) -> tuple:
        if name not in self._meta:
            import numcodecs

            m = json.loads(webget.get(f"{self.url}/{name}/.zarray"))
            if m.get("filters") or m.get("order", "C") != "C" or m.get("dimension_separator", ".") != ".":
                raise RuntimeError(f"{name}: a zarr layout this reader does not know")
            self._meta[name] = (m, numcodecs.get_codec(m["compressor"]))
        return self._meta[name]

    def chunk(self, name: str, *index: int) -> np.ndarray | None:
        """The chunk at ``index`` along the leading dimensions (whole along the rest), or None where the store has
        none (a step not written yet)."""
        m, codec = self.meta(name)
        key = ".".join(str(i) for i in [*index, *[0] * (len(m["shape"]) - len(index))])
        try:
            raw = webget.get(f"{self.url}/{name}/{key}", timeout=180)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise
        return np.frombuffer(codec.decode(raw), np.dtype(m["dtype"])).reshape(m["chunks"])

    def coord(self, name: str) -> np.ndarray:
        m, _ = self.meta(name)
        n, c = m["shape"][0], m["chunks"][0]
        return np.concatenate([self.chunk(name, i) for i in range(-(-n // c))])[:n]

    def times(self) -> pd.DatetimeIndex:
        step, _, origin = self.attrs("time")["units"].partition(" since ")
        return pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(self.coord("time"), unit=step))

    def dims(self, name: str, *want: str) -> None:
        if tuple(d := self.attrs(name)["_ARRAY_DIMENSIONS"]) not in want:
            raise RuntimeError(f"{self.url}: {name} is {d}, not as read here")


def era5_stop() -> dt.date:
    """The newest day ARCO holds (its ERA5T, about a week behind)."""
    return pd.Timestamp(Zarr(ARCO).attrs()["valid_time_stop_era5t"]).date()


def era5(a) -> None:
    """ERA5's 00Z and 12Z samples of each day of the months asked that ARCO holds and the database lacks, a day a file
    in the store's days/, and each month they complete as wk-era5_<YYYYMM>.npz (--complete: complete months only)."""
    have = _have(a.have)
    z = Zarr(ARCO)
    stop = pd.Timestamp(z.attrs()["valid_time_stop_era5t"]).date()
    for name, lev in ERA5_VARS.values():
        z.dims(name, ("time", "level", "latitude", "longitude") if lev else ("time", "latitude", "longitude"))
    t = z.times()
    lat, lon = z.coord("latitude"), z.coord("longitude")
    level = int(np.flatnonzero(z.coord("level") == 500)[0])
    days_dir, db = a.store / "days", a.store / "db"
    days_dir.mkdir(parents=True, exist_ok=True)
    a.archive_out.mkdir(parents=True, exist_ok=True)

    def read(job):
        d, hour, tag = job
        name, lev = ERA5_VARS[tag]
        v = z.chunk(name, t.get_loc(pd.Timestamp(d) + pd.Timedelta(hours=hour)))
        return None if v is None else (v[0] if lev is None else v[0, level]).copy()  # not a view of the 37 levels

    for m in months(a.first or FIRST_INIT, min(a.last or stop, stop)):
        mdays = month_days(m)
        if era5_file(m) in have or (db / era5_file(m)).exists():
            for d in mdays:  # the month's days are in the database: the cache drops them
                (days_dir / f"era5_{d:%Y%m%d}.npz").unlink(missing_ok=True)
            continue
        if a.complete and mdays[-1] > stop:
            continue
        todo = [d for d in mdays if d <= stop and not (days_dir / f"era5_{d:%Y%m%d}.npz").exists()]
        with ThreadPoolExecutor(8) as ex:
            for i in range(0, len(todo), 4):  # a few days at a time, so the samples waiting in order stay few
                jobs = [(d, h, tag) for d in todo[i : i + 4] for h in (0, 12) for tag in TAGS]
                got = dict(zip(jobs, ex.map(read, jobs)))
                for d in todo[i : i + 4]:
                    v = {(h, tag): got[(d, h, tag)] for h in (0, 12) for tag in TAGS}
                    if any(x is None or not np.isfinite(x).all() for x in v.values()):
                        print(f"::warning::ERA5 {d}: incomplete in ARCO, left for the next pass", flush=True)
                        continue
                    # remapped here, not in the threads: a matrix product from threads is unsafe with some BLAS
                    f = {tag: np.stack([remap(v[(h, tag)], lat, lon, G15) for h in (0, 12)]) for tag in TAGS}
                    f["zg500"] /= GRAV
                    np.savez_compressed(days_dir / f"era5_{d:%Y%m%d}.npz", **{k: G.pack(k, x) for k, x in f.items()})
                    print(f"ERA5 {d}: read", flush=True)
        files = [days_dir / f"era5_{d:%Y%m%d}.npz" for d in mdays]
        if all(p.exists() for p in files):
            month = {tag: [] for tag in TAGS}
            for p in files:
                with np.load(p) as x:
                    for tag in TAGS:
                        month[tag].append(x[tag])
            np.savez_compressed(
                a.archive_out / era5_file(m),
                days=np.array([f"{d:%Y%m%d}" for d in mdays]),
                **{tag: np.stack(v) for tag, v in month.items()},
            )
            print(f"ERA5 {m:%Y-%m}: {era5_file(m)}", flush=True)


def clim(a) -> None:
    """The ERA5 1991-2020 normal of the day mean (12Z and the next 00Z), for every slot of the 366-day calendar, from
    WeatherBench 2's 6-hourly 1.5 deg ERA5: every chunk holding a wanted step is read (8 steps a chunk, all 13 levels
    for geopotential), side by side over HTTPS as z500_acc.clim does. Each sample counts half to the day it belongs to:
    a 12Z sample to its own date, a 00Z sample to the day before."""
    z = Zarr(WB2)
    lat, lon = z.coord("latitude"), z.coord("longitude")
    if not (np.allclose(np.sort(lat)[::-1], G15.lat) and np.allclose(lon, G15.lon)):
        raise RuntimeError("WeatherBench 2's grid is not the verification grid")
    flip = lat[0] < lat[-1]
    t = z.times()
    day = t.normalize() - pd.to_timedelta((t.hour == 0).astype(int), unit="D")
    want = np.asarray(((t.hour == 0) | (t.hour == 12)) & (day.year >= CLIM_YEARS[0]) & (day.year <= CLIM_YEARS[1]))
    slots = slot366(day)
    rows = np.arange(366)
    near = (np.abs((rows[:, None] - rows[None, :] + 183) % 366 - 183) <= CLIM_HALF).astype(float)  # around the year
    out = {}
    for tag in TAGS:
        name, lev = ERA5_VARS[tag]
        z.dims(name, ("time", "level", "longitude", "latitude") if lev else ("time", "longitude", "latitude"))
        meta, _ = z.meta(name)
        if lev and meta["chunks"][1] != meta["shape"][1]:
            raise RuntimeError(f"{name}: its levels are chunked, not as read here")
        k_lev = int(np.flatnonzero(z.coord("level") == lev)[0]) if lev else None
        nt, unit = meta["chunks"][0], GRAV if tag == "zg500" else 1.0

        def block(b, name=name, k_lev=k_lev):
            v = z.chunk(name, b)
            return v if k_lev is None else v[:, k_lev]

        blocks = np.unique(np.flatnonzero(want) // nt)
        total, count = np.zeros((366, *G15.shape)), np.zeros(366)
        with ThreadPoolExecutor(32) as ex:
            for i in range(0, blocks.size, 256):  # a batch at a time, so the chunks waiting in order stay few
                batch = blocks[i : i + 256]
                for b, v in zip(batch, ex.map(block, batch), strict=True):
                    k = np.flatnonzero(want[b * nt : (b + 1) * nt])
                    f = v[k].transpose(0, 2, 1).astype(np.float64) / unit  # (steps, lat, lon)
                    f = f[:, ::-1] if flip else f
                    if not np.isfinite(f).all():
                        raise RuntimeError(f"{name}: missing values in chunk {b}")
                    np.add.at(total, slots[b * nt + k], 0.5 * f)
                    np.add.at(count, slots[b * nt + k], 0.5)
                print(f"normal {tag}: read to {t[min(batch[-1] * nt, t.size - 1)]:%Y-%m-%d}", flush=True)
        if count.sum() != want.sum() / 2:
            raise RuntimeError(f"normal {tag}: {count.sum()} of {want.sum() / 2} days read")
        out[tag] = G.pack(tag, (near @ total.reshape(366, -1)).reshape(total.shape) / (near @ count)[:, None, None])
    land = z.chunk("land_sea_mask").T
    np.savez_compressed(
        a.out,
        **out,
        land=(land[::-1] if flip else land).astype(np.float32),
        lat=G15.lat,
        lon=G15.lon,
        days=(near @ count).astype(np.int32),
    )
    print(f"normal: {a.out} ({int(count.sum())} days)", flush=True)
    # the truth is ARCO's ERA5 remapped here, the normal WeatherBench 2's own 1.5 deg: at a time both hold the two
    # must agree, or every anomaly would carry the difference of the two remappings
    when = pd.Timestamp("2022-07-01 00:00")
    za = Zarr(ARCO)
    alat, alon, ia = za.coord("latitude"), za.coord("longitude"), za.times().get_loc(when)
    for tag in TAGS:
        name, lev = ERA5_VARS[tag]
        v = za.chunk(name, ia)[0]
        v = v if lev is None else v[int(np.flatnonzero(za.coord("level") == lev)[0])]
        nt, i = z.meta(name)[0]["chunks"][0], t.get_loc(when)
        w = z.chunk(name, i // nt)[i % nt]
        w = (w if lev is None else w[int(np.flatnonzero(z.coord("level") == lev)[0])]).T
        d = (remap(v, alat, alon, G15) - (w[::-1] if flip else w)) / (GRAV if tag == "zg500" else 1.0)
        print(
            f"check {tag} {when:%Y-%m-%d %HZ}: ARCO remapped here less WeatherBench 2, rms {np.sqrt((d**2).mean()):.3g}, "
            f"largest {np.abs(d).max():.3g}",
            flush=True,
        )


class Truth:
    """ERA5 from the store, the database's month files (db/) and the days of the month in progress (days/): the day
    mean of a run's day (the 12Z sample and the next 00Z) and its anomaly from the normal."""

    def __init__(self, store: Path, clim_path: Path):
        with np.load(clim_path) as c:
            self.normal = np.stack([self._units(t, c[t]) for t in TAGS]).astype(np.float32)  # (3, 366, lat, lon)
            self.land = c["land"] >= 0.5
        self.samples: dict[dt.date, np.ndarray] = {}  # (3, 2, lat, lon): 00Z, 12Z
        for p in sorted((store / "db").glob("wk-era5_*.npz")):
            with np.load(p) as z:
                for i, d in enumerate(z["days"]):
                    self.samples[_date(str(d))] = np.stack([self._units(t, z[t][i]) for t in TAGS])
        for p in sorted((store / "days").glob("era5_*.npz")):
            with np.load(p) as z:
                self.samples.setdefault(_date(p.stem[5:]), np.stack([self._units(t, z[t]) for t in TAGS]))
        self._anom: dict[dt.date, np.ndarray | None] = {}

    @staticmethod
    def _units(tag, q):
        return (G.unpack(tag, q) * (0.01 if tag == "mslp" else 1.0)).astype(np.float32)

    def last(self) -> dt.date | None:
        """The newest day with an anomaly (its next day's 00Z is in)."""
        ok = [d for d in self.samples if d + dt.timedelta(days=1) in self.samples]
        return max(ok) if ok else None

    def anom(self, v: dt.date) -> np.ndarray | None:
        """(3, lat, lon) anomaly of day v, or None until ERA5 covers it."""
        if v not in self._anom:
            a, b = self.samples.get(v), self.samples.get(v + dt.timedelta(days=1))
            s = slot366(pd.DatetimeIndex([v]))[0]
            self._anom[v] = None if a is None or b is None else 0.5 * (a[:, 1] + b[:, 0]) - self.normal[:, s]
        return self._anom[v]

    def mean(self, first: dt.date, n: int) -> np.ndarray | None:
        x = [self.anom(first + dt.timedelta(days=k)) for k in range(n)]
        return None if any(v is None for v in x) else np.mean(x, axis=0)


# --------------------------------------------------------------------------- forecasts
def _geavg(init: dt.date, h: int) -> str:
    return f"{GEFS_S3}/gefs.{init:%Y%m%d}/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f{h:03d}"


def _index(url: str):
    try:
        return R.index(url)
    except Exception:  # noqa: BLE001  (not on S3: the run is not complete)
        return None


def gefs_run(init: dt.date, cache: Path) -> dict | None:
    """One run's weekly anomalies before the drift, a (3, 5, lat, lon), and the daily zonal means of F - M, fm (3, 35,
    lat); None while the run is not complete on S3."""
    import gefs_live as L  # (pyproj first: gefs_live loads it before eccodes)

    hours = range(12, 841, 12)
    if _index(_geavg(init, 840)) is None:
        return None
    urls = [_geavg(init, h) for h in hours]
    with ThreadPoolExecutor(24) as ex:
        idxs = list(ex.map(_index, urls))
    if any(i is None for i in idxs):
        return None
    jobs = []
    for h, url, idx in zip(hours, urls, idxs):
        for t in TAGS:
            name, lev = G.D2[t][4], G.D2[t][2]
            hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == f"{h} hour fcst"]
            if not hit:
                raise LookupError(f"{name} {lev} +{h} h not in {url}")
            jobs.append((url, hit[0][0], hit[0][1], t, (h + 23) // 24))
    daily = G.Acc(TAGS)
    with ThreadPoolExecutor(24) as ex:
        for (_, _, _, t, d), b in zip(jobs, ex.map(lambda j: R.get(j[0], (j[1], j[2])), jobs)):
            daily.add(t, d, R.decode(b))
    r = daily.finish()
    if r is None:
        raise LookupError(f"GEFS {init}: a day is missing")
    C2 = L.clim2_at(init, cache)
    if C2 is None:
        raise RuntimeError(f"the daily reforecast climatology (release {R.CLIM_TAG}) is required")
    sh = np.load(L.ref("era5_shift.npz", cache))
    days = L.valid_days(init)
    a, fm = [], []
    for t in TAGS:
        k = 0.01 if t == "mslp" else 1.0
        FM = r[f"d_{t}"] - C2[f"d_{t}"]
        a.append(L.weekly_of(FM - L.shift_series(sh, f"s15_{t}", days)) * k)
        fm.append(FM.mean(2) * k)
    return {"a": np.stack(a).astype(np.float32), "fm": np.stack(fm).astype(np.float32), "fields": np.array(TAGS)}


def gefs(a) -> None:
    """The Monday and Thursday runs the database lacks that are complete on S3 (--list: print them, nothing fetched)."""
    have = _have(a.have)
    due = [d for d in starts(a.first or FIRST_INIT, a.last or _today()) if run_file("GEFS", d) not in have]
    if a.list:
        print(*(f"{d:%Y%m%d}" for d in due))
        return
    a.archive_out.mkdir(parents=True, exist_ok=True)
    failed = []
    for d in due:
        try:
            z = gefs_run(d, a.cache)
        except Exception as e:  # noqa: BLE001  (the other runs go on; this one is tried again next pass)
            print(f"::warning::GEFS {d}: {str(e)[:200]}", flush=True)
            failed.append(d)
            continue
        if z is None:
            print(f"GEFS {d}: not complete on S3", flush=True)
            continue
        np.savez_compressed(a.archive_out / run_file("GEFS", d), **z)
        print(f"GEFS {d}: {run_file('GEFS', d)}", flush=True)
    if failed:
        raise SystemExit(f"GEFS runs not archived: {' '.join(f'{d:%Y%m%d}' for d in failed)}")


def geps(a) -> None:
    """A GEPS run's weekly anomalies as the page's weekly maps make them, from the run's files (geps.yml)."""
    import xarray as xr

    sys.path.insert(0, str(HERE.parent / "geps"))
    import paths
    import render_maps as M

    base = pd.Timestamp(a.cycle)
    lat1, lon1 = np.arange(-90.0, 90.0 + 1e-6, 1.0), np.arange(0.0, 360.0, 1.0)
    out = np.full((len(TAGS), len(M.WEEKS), *G15.shape), np.nan, np.float32)
    for i, tag in enumerate(TAGS):
        with xr.open_dataset(paths.LIVE / f"geps_live_{tag}_{a.cycle}.nc") as d:
            an = d[f"{tag}_anom"].sel(latitude=lat1, longitude=lon1).load()  # its own 1 deg nodes (archive.py)
        gm = an.global_mean.values
        r = M.reref(tag, an, base)
        if r is None:
            raise SystemExit(f"{tag}: no ERA5 shift in the reference set")
        an = an + r.values
        if M.FIELDS[tag][3]:  # 500 hPa: the page removes the per-lead global mean
            an = an - gm[:, None, None]
        an = an * M.FIELDS[tag][4]
        for k, (w0, w1) in enumerate(M.WEEKS):
            wk = an.sel(L=slice(w0, w1))
            if wk.sizes["L"] == w1 - w0 + 1:
                out[i, k] = remap(wk.mean("L").values, lat1, lon1, G15)
    a.archive_out.mkdir(parents=True, exist_ok=True)
    p = a.archive_out / run_file("GEPS", base.date())
    np.savez_compressed(p, a=out, fields=np.array(TAGS))
    print(f"GEPS {a.cycle}: {p.name}, {int(np.isfinite(out[:, :, 0, 0]).sum())} of {out[:, :, 0, 0].size} weeks")


# --------------------------------------------------------------------------- scores
def regions(land: np.ndarray) -> dict[str, tuple[str, dict[str, np.ndarray]]]:
    """{region: (label, {field: weights})}: cos(latitude) in the region, over land only for 2 m temperature. North and
    South America are the pages' map extents (gefs_live.DOMAINS)."""
    import gefs_live as L

    lat, lon = G15.lat[:, None], G15.lon[None, :]
    lon180 = np.where(lon > 180, lon - 360, lon)
    box = {"nh": ("Northern extratropics, 20–90°N", lat >= 20), "tropics": ("Tropics, 20°S–20°N", np.abs(lat) <= 20)}
    for key in ("na", "sa"):
        w, e, s, n = L.DOMAINS[key]["extent"]
        box[key] = (L.DOMAINS[key]["label"], (lat >= s) & (lat <= n) & (lon180 >= w) & (lon180 <= e))
    cos = np.cos(np.radians(lat))
    return {
        r: (label, {t: np.where(m & (land if t == "t2m" else True), cos, 0.0) * np.ones(G15.shape) for t in TAGS})
        for r, (label, m) in box.items()
    }


def acc(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> float:
    """The centred anomaly correlation of two maps over the cells of weight w > 0."""
    m = w > 0
    f, o, w = f[m], o[m], w[m]
    f, o = f - (w * f).sum() / w.sum(), o - (w * o).sum() / w.sum()
    return float((w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum()))


def scored(model: str, init: dt.date, A: np.ndarray, truth: Truth, W: dict) -> list[dict]:
    """One run's rows: its anomaly correlation and persistence's, per field, region and week ERA5 covers."""
    pers = truth.mean(init - dt.timedelta(days=7), 7)
    out = []
    for k, (w0, _) in enumerate(R.WEEKS):
        obs = truth.mean(init + dt.timedelta(days=w0 - 1), 7)
        if obs is None or not np.isfinite(A[:, k]).all():
            continue
        for r, (_, ws) in W.items():
            for i, t in enumerate(TAGS):
                out.append(
                    {
                        "model": model,
                        "init": pd.Timestamp(init),
                        "field": t,
                        "region": r,
                        "week": k + 1,
                        "acc": acc(A[i, k], obs[i], ws[t]),
                        "pers": np.nan if pers is None else acc(pers[i], obs[i], ws[t]),
                    }
                )
    return out


def summary(s: pd.Series, rng: np.random.Generator) -> dict:
    """Mean of a run series with its 90% interval and two-sided p against zero from a circular moving-block bootstrap
    over weeks of starts (BLOCK weeks a block); fewer than MIN_WEEKS weeks are not tested."""
    s = s.dropna()
    monday = s.index.min() - pd.Timedelta(days=s.index.min().weekday())
    wk = ((s.index - monday).days // 7).to_numpy()
    nw = int(np.unique(wk).size)
    out = {"n": int(s.size), "w": nw}
    if nw < MIN_WEEKS:
        return {**out, "few": True}
    W = int(wk.max()) + 1
    S, C = np.bincount(wk, s.to_numpy(), W), np.bincount(wk, minlength=W).astype(float)
    pick = (rng.integers(0, W, (NBOOT, -(-W // BLOCK)))[..., None] + np.arange(BLOCK)) % W
    c = C[pick].sum((1, 2))
    boot = S[pick].sum((1, 2))[c > 0] / c[c > 0]
    lo, hi = np.percentile(boot, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {**out, "mean": float(s.mean()), "lo": float(lo), "hi": float(hi), "p": p}


def bh(p: np.ndarray, q: float = FDR) -> np.ndarray:
    """Benjamini-Hochberg: which of the p values pass at false-discovery rate q."""
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, p.size + 1) / p.size
    keep = np.zeros(p.size, bool)
    if ok.any():
        keep[o[: np.flatnonzero(ok)[-1] + 1]] = True
    return keep


def score(a) -> None:
    import gefs_drift as GD

    truth = Truth(a.store, a.clim)
    W = regions(truth.land)
    files = {
        m: {_date(p.stem.split("_")[1][:8]): p for p in sorted((a.store / "db").glob(f"wk-{m.lower()}_*.npz"))}
        for m in ("GEFS", "GEPS")
    }
    D: dict[dt.date, np.ndarray | None] = {}

    def drift_input(i: dt.date) -> np.ndarray | None:
        """D_i(L) of gefs_drift.verify against ERA5: [zm(F - M) - zm(ERA5 anomaly)] less its day-1 value; with ERA5's
        normal for gefs_drift's O, the same as its (E - E(1)) - R."""
        if i not in D:
            x = [truth.anom(i + dt.timedelta(days=k)) for k in range(35)]
            if any(v is None for v in x):
                D[i] = None
            else:
                with np.load(files["GEFS"][i]) as z:
                    e = z["fm"].astype(np.float64) - np.stack([v.mean(-1) for v in x], axis=1)
                D[i] = e - e[:, :1]
        return D[i]

    rows, drift_n = [], []
    for i, p in files["GEFS"].items():
        if i < SCORE_FROM:
            continue
        last = i - dt.timedelta(days=GD.LAG)
        win = [j for j in files["GEFS"] if last - dt.timedelta(days=GD.WINDOW - 1) <= j <= last]
        Ds = [x for j in win if (x := drift_input(j)) is not None]
        if len(Ds) < MIN_DRIFT:
            print(f"GEFS {i}: {len(Ds)} runs for the drift, not scored", flush=True)
            continue
        delta = GD.combine(np.stack(Ds))[0].astype(np.float64)  # (3, 35, lat)
        with np.load(p) as z:
            A = z["a"].astype(np.float64)
        A -= np.stack([delta[:, w0 - 1 : w1].mean(1) for w0, w1 in R.WEEKS], axis=1)[..., None]
        # (the page also removes 500 hPa's daily global mean: a uniform offset leaves a centred correlation as it is)
        rows += scored("GEFS", i, A, truth, W)
        drift_n.append(len(Ds))
    for i, p in files["GEPS"].items():
        with np.load(p) as z:
            rows += scored("GEPS", i, z["a"].astype(np.float64), truth, W)
    df = pd.DataFrame(rows, columns=["model", "init", "field", "region", "week", "acc", "pers"])
    rng = np.random.default_rng(20261101)
    cells, tests = {}, []
    for t in TAGS:
        for r in W:
            cell = cells.setdefault(f"{t}/{r}", {})
            for k in range(1, len(R.WEEKS) + 1):
                sub = df[(df["field"] == t) & (df["region"] == r) & (df["week"] == k)]
                g, e = (sub[sub["model"] == m].set_index("init") for m in ("GEFS", "GEPS"))
                series = {
                    "GEFS": g["acc"],
                    "GEPS": e["acc"],
                    "persistence": g["pers"],
                    "GEFS-persistence": g["acc"] - g["pers"],
                    "GEPS-persistence": e["acc"] - e["pers"],
                    "GEPS-GEFS": (e["acc"] - g["acc"]).dropna(),
                }
                for key, s in series.items():
                    res = summary(s.sort_index(), rng) if len(s.dropna()) else {"n": 0, "w": 0, "few": True}
                    cell.setdefault(key, []).append(res)
                    if "p" in res:
                        tests.append(res)
    for res, ok in zip(tests, bh(np.array([x["p"] for x in tests])) if tests else []):
        p = res.pop("p")
        if ok:
            res.update({k: round(res[k], 3) for k in ("mean", "lo", "hi")}, p=float(f"{p:.2g}"))
        else:  # not published: only that it was tested
            for k in ("mean", "lo", "hi"):
                res.pop(k)
            res["ns"] = True
    runs = {}
    for m in ("GEFS", "GEPS"):
        d = df.loc[df["model"] == m, "init"]
        runs[m] = {"n": int(d.nunique())} | (
            {"first": f"{d.min():%Y-%m-%d}", "last": f"{d.max():%Y-%m-%d}"} if len(d) else {}
        )
    era = truth.last()
    out = {
        "era5_through": f"{era:%Y-%m-%d}" if era else None,
        "score_from": f"{SCORE_FROM:%Y-%m-%d}",
        "fields": [[t, NAMES[t]] for t in TAGS],
        "regions": [[r, label] for r, (label, _) in W.items()],
        "weeks": [f"{w0}–{w1}" for w0, w1 in R.WEEKS],
        "runs": runs,
        "drift_runs": [int(min(drift_n)), int(max(drift_n))] if drift_n else None,
        "test": {"level": LEVEL, "fdr": FDR, "block_weeks": BLOCK, "min_weeks": MIN_WEEKS, "tests": len(tests)},
        "cells": cells,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    sig = sum(1 for x in tests if not x.get("ns"))
    print(
        f"{a.out}: GEFS {runs['GEFS']['n']} runs, GEPS {runs['GEPS']['n']}, {sig} of {len(tests)} tests pass",
        flush=True,
    )


# --------------------------------------------------------------------------- the workflow's helpers
def plan(a) -> None:
    """The backfill's jobs: the months whose ERA5 file or Monday/Thursday GEFS runs the database lacks, a.months months
    a job and each kind apart, as the matrix of verify-weeks.yml."""
    have, stop, today = _have(a.have), era5_stop(), _today()
    chunks = {"era5": [], "gefs": []}
    for m in months(FIRST_INIT, today):
        end = month_days(m)[-1]
        if end <= stop and era5_file(m) not in have:
            chunks["era5"].append(m)
        if any(run_file("GEFS", d) not in have for d in starts(max(m, FIRST_INIT), min(end, today))):
            chunks["gefs"].append(m)
    jobs = [
        {
            "kind": kind,
            "first": f"{ms[i]:%Y%m%d}",
            "last": f"{month_days(ms[min(i + a.months, len(ms)) - 1])[-1]:%Y%m%d}",
        }
        for kind, ms in chunks.items()
        for i in range(0, len(ms), a.months)
    ]
    print("chunks=" + json.dumps(jobs))


def missing(a) -> None:
    print(*sorted(n for n in _have(a.have) if n.startswith("wk-") and not (a.store / "db" / n).exists()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clim", help="the ERA5 1991-2020 normal (built once)")
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("plan", help="print the backfill's jobs")
    p.add_argument("--months", type=int, default=3)
    e = sub.add_parser("era5", help="ERA5's samples of the days ARCO has added")
    e.add_argument("--store", type=Path, required=True)
    e.add_argument("--complete", action="store_true", help="complete months only (a backfill job keeps no days)")
    f = sub.add_parser("gefs", help="the Monday and Thursday GEFS runs")
    f.add_argument("--cache", type=Path, default=Path("gefs-clim"), help="the GEFS reforecast climatology's files")
    f.add_argument("--list", action="store_true", help="print the runs the database lacks")
    g = sub.add_parser("geps", help="a GEPS run's weekly anomalies, in geps.yml")
    g.add_argument("--cycle", required=True)
    m = sub.add_parser("missing", help="print the database's files the store lacks")
    m.add_argument("--store", type=Path, required=True)
    s = sub.add_parser("score", help="every run's scores -> the page's JSON")
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--clim", type=Path, required=True)
    s.add_argument("--out", type=Path, default=HERE.parents[1] / "assets" / "verification" / "weeks.json")
    for q in (e, f):
        q.add_argument("--first", type=_date, help="YYYYMMDD")
        q.add_argument("--last", type=_date, help="YYYYMMDD")
    for q in (p, e, f, m):
        q.add_argument("--have", type=Path, help="names of the files the database holds, one per line")
    for q in (e, f, g):
        q.add_argument("--archive-out", type=Path, default=Path("db-new"), help="where the files to add are written")
    a = ap.parse_args()
    {"clim": clim, "plan": plan, "era5": era5, "gefs": gefs, "geps": geps, "missing": missing, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
