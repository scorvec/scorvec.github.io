#!/usr/bin/env python3
"""Hadley outflow and subtropical descent: the analysed weeks as maps, a basin tracker and back-trajectories, against
ERA5 1991-2020 (build_branch_clim.py).

The source is the IFS HRES analysis (step 0) at 00, 06, 12 and 18 UTC, byte-ranged from the Google mirror and put on the
1.5 deg grid (hadley_branches). ARCH holds, in Actions as the release hbranch-hres:
  w3_YYYY-MM-DD.npz  the day's four times of u, v, omega and temperature on the 13 levels (half precision, temperature
                     less T_OFF) with the 500 hPa height and sea-level pressure: the last STORE_DAYS days, what the
                     back-trajectories need
  hb_YYYY-MM.npz     the daily fields (hadley_branches.fields of the four times' mean) since FIRST
  traj.npz           each day's 00 UTC release: the transit-time histograms (hadley_traj)
  offset_era5.npz    the analysis less ERA5 on the overturn lattice days from OFFSET_FIRST that ARCO has published: the
                     four times' mean difference of the 3-D fields and of the daily fields, as harmonic sums (mean,
                     annual, semiannual), so that parts add
Every frame and tracker value is a trailing 7-day window ending on its day, against the ERA5 normal at the window's
midpoint: the maps against the harmonic normal, the tracker against the distribution of ERA5's own 7-day windows. The
analysis is first corrected by its mean offset from ERA5 at the same season - the daily fields by theirs, the
trajectories' winds and temperatures by theirs - and nothing is drawn while fewer than MIN_PAIRS lattice days are paired.
The outflow, descent, ridge and high trackers also carry the three ensembles' latest forecast windows (FC, from
hadley_branch_fc.py) while it is fresh.

    python src/hadley_branch_maps.py truth [--part k/n --out f]  # pair lattice days with ERA5 (into OFFSETS, or f)
    python src/hadley_branch_maps.py merge f1 f2 ...             # add part files into OFFSETS
    python src/hadley_branch_maps.py update [--fresh]            # archive the new days, release their trajectories
    python src/hadley_branch_maps.py render [--rebuild]          # frames, manifest, still and trackers
    python src/hadley_branch_maps.py forecast                    # the interactive map's paths on the newest HRES run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import lru_cache, partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import build_branch_clim as C
import hadley_branches as HB
import hadley_traj as T
import overturn as O
from doycal import noleap_doy
from hadley_timeseries import MODEL_C

REPO = Path(__file__).resolve().parents[3]
ANIM = REPO / "assets" / "sst" / "anim"
OUT = REPO / "assets" / "sst"
ARCH = REPO / "scripts" / "mjo" / "data" / "hbranch"
WRITTEN, REMOVED = ARCH / "written.txt", ARCH / "removed.txt"  # the release's files to upload / to delete
OFFSETS, TRAJ = ARCH / "offset_era5.npz", ARCH / "traj.npz"
FC = OUT / "data" / "hbranch_fc.json"  # the ensembles' tracker numbers (hadley_branch_fc.py)
DIR, DT_DIR = "hbranch_hist", "hbranch_dt"  # the weekly descent and ridge loop; the daily outflow loop
FIRST = pd.Timestamp("2026-05-01")  # the daily fields' archive begins here (the maps reach back DAYS + WIN days)
OFFSET_FIRST = pd.Timestamp("2025-01-01")  # HRES carries omega and all 13 levels from late 2024
WIN, DAYS = 7, 120
STORE_DAYS = T.DAYS_BACK + WIN + 3
MIN_PAIRS = 60
VARS3 = ("u", "v", "w", "t")
T_OFF = 250.0  # K, taken off the stored temperatures so that half precision keeps 0.03 K
HOURS = (0, 6, 12, 18)
PC = ccrs.PlateCarree()
EXTENT = (-50.0, 50.0)
ROWS = ("nh", "npac", "natl", "sh", "spac", "satl", "sind")
DT_KM = np.arange(8.0, 17.51, 0.5)  # the daily map's colours: the dynamic tropopause's height (km)
W_EFOLD = 12.0  # spherical-harmonic degree damped by 1/e in the mapped vertical motion (a strong Gaussian filter)
# the interactive map's forward paths: seeds over the deep tropics at DROP_P, released DROP_BACK days before the last
# analysis and followed for up to DAYS_BACK days (or to the last analysis), 6-hourly, as int16 files per release and level
DROP_DIR = "hbranch_drop"
DROP_LAT = np.arange(-20.0, 20.1, 2.5)
DROP_LON = np.arange(0.0, 360.0, 3.0)
DROP_P = (150.0, 250.0, 400.0)
DROP_BACK = (35, 28, 21, 14)
FC_DIR, FC_HOURS = "hbranch_dropfc", 240  # the forecast release: the 00 and 12 UTC HRES runs reach day 10
BASE_EXTENT, BASE_PX = (-60.0, 60.0), 4  # the interactive map's basemap: latitudes, pixels per degree
POOL_LON = 2  # release points pooled along longitude in the transit-time map
SECTOR_NAME = {
    "nh": "Northern Hemisphere",
    "sh": "Southern Hemisphere",
    "npac": "North Pacific",
    "natl": "North Atlantic",
    "spac": "South Pacific",
    "satl": "South Atlantic",
    "sind": "South Indian",
}
LEV = {
    "desc": [-45, -30, -20, -15, -10, -5, 5, 10, 15, 20, 30, 45],
    "msl": [-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12],
}


# ── sources ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _messages(blob: bytes):
    i = 0
    while (j := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[j + 8 : j + 16], "big")  # GRIB2 section 0: the message's total length
        yield blob[j : j + n]
        i = j + n


def hres(t: pd.Timestamp, step: int = 0, params=VARS3, extras: bool = True) -> dict:
    """IFS HRES from the run at t, `step` hours on (0 = the analysis), on the 1.5 deg grid: `params` on LEVELS (level,
    lat, lon) and, with extras, z500 (m) and msl (Pa). The 06 and 18 UTC runs were stream scda until ECMWF moved them
    to oper."""
    import eccodes
    import rangefetch as rf

    date, hh = f"{t:%Y%m%d}", f"{t:%H}"
    for stream in ("oper", "scda"):
        try:
            idx = rf.fetch_index(date, hh, "ifs", step, "fc", stream=stream)
            break
        except rf.IndexUnavailable as e:
            if not e.not_found:
                raise
    else:
        raise RuntimeError(f"HRES {date} {hh}Z: not on the Google mirror")
    want = [e for e in rf.select(idx, param=list(params), levelist=list(HB.LEVELS)) if e.get("levtype") == "pl"]
    if extras:
        want += [e for e in rf.select(idx, param="gh", levelist=[500]) if e.get("levtype") == "pl"]
        want += [e for e in rf.select(idx, param="msl") if e.get("levtype") == "sfc"]
    if len(want) != len(params) * len(HB.LEVELS) + 2 * extras:
        raise RuntimeError(f"HRES {date} {hh}Z +{step}h: {len(want)} of the messages in the index")
    blob = rf.fetch_ranges(rf.path_for(date, hh, "ifs", step, "fc", stream=stream) + ".grib2", rf.coalesce(want))
    out = {k: np.full((len(HB.LEVELS), HB.LAT.size, HB.LON.size), np.nan, np.float32) for k in params}
    for m in _messages(blob):
        h = eccodes.codes_new_from_message(m)
        try:
            g = {k: eccodes.codes_get(h, k) for k in ("shortName", "level", "Ni", "Nj", "jScansPositively")}
            la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dj = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
            di = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
            vals = eccodes.codes_get_values(h).reshape(g["Nj"], g["Ni"])
        finally:
            eccodes.codes_release(h)
        lat = la0 + (dj if g["jScansPositively"] else -dj) * np.arange(g["Nj"])
        f = HB.coarse15(vals, lat, lo0 + di * np.arange(g["Ni"])).astype(np.float32)
        if g["shortName"] in params:
            out[g["shortName"]][list(HB.LEVELS).index(int(g["level"]))] = f
        elif g["shortName"] == "gh":
            out["z500"] = f
        else:
            out["msl"] = f
    if any(np.isnan(out[k]).any() for k in params) or (extras and ("z500" not in out or "msl" not in out)):
        raise RuntimeError(f"HRES {date} {hh}Z +{step}h: a field was missing from the messages")
    return out


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _arco() -> xr.Dataset:
    import era5_store

    return xr.open_zarr(era5_store.ARCO, chunks=None, storage_options={"token": "anon"})


def era5_time(t: pd.Timestamp) -> dict | None:
    """ERA5 at t from ARCO, as hres; None while ARCO has not published it (it pads with NaN)."""
    ds = _arco()
    names = {"u": "u_component_of_wind", "v": "v_component_of_wind", "w": "vertical_velocity", "t": "temperature"}

    def grid(da):
        return HB.coarse15(da.values, da.latitude.values, da.longitude.values).astype(np.float32)

    v = ds[names["v"]].sel(time=t, level=list(HB.LEVELS)).load()
    if bool(np.isnan(v.values).all()):
        return None
    out = {"v": grid(v)}
    for k in ("u", "w", "t"):
        out[k] = grid(ds[names[k]].sel(time=t, level=list(HB.LEVELS)).load())
    out["z500"] = grid(ds["geopotential"].sel(time=t, level=500).load()) / C.G
    out["msl"] = grid(ds["mean_sea_level_pressure"].sel(time=t).load())
    return out


def day_mean(times: list[dict]) -> dict:
    return {k: np.mean([x[k] for x in times], axis=0) for k in times[0]}


def derived(m: dict) -> dict:
    return HB.fields(m["u"], m["v"], m["w"], m["t"], m["z500"], m["msl"])


def _doy(day: pd.Timestamp) -> float:
    return float(noleap_doy(pd.DatetimeIndex([day]))[0])


def _note(path: Path, removed: bool = False):
    """List a file of ARCH for the release: to upload, or (removed) to delete."""
    ARCH.mkdir(parents=True, exist_ok=True)
    with (REMOVED if removed else WRITTEN).open("a") as fh:
        fh.write(path.name + "\n")


# ── the analysis offset from ERA5 ─────────────────────────────────────────────────────────────────────────────────────
def _harm(doy: float) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    return np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def _empty_sums() -> dict:
    return {
        "date": np.array([], "datetime64[D]"),
        "XtX": np.zeros((5, 5)),
        "XtY3": np.zeros((5, len(VARS3), len(HB.LEVELS), HB.LAT.size, HB.LON.size)),
        "XtY2": np.zeros((5, len(HB.FIELDS), HB.LAT.size, HB.LON.size)),
    }


def _save_sums(path: Path, S: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **S)
    if path == OFFSETS:
        _note(path)


def truth(until: pd.Timestamp, part=(0, 1), out: Path = OFFSETS):
    """Pair the lattice days from OFFSET_FIRST to `until` not yet in OFFSETS (every n-th from the k-th for a part, into
    its own file): HRES less ERA5, the four times' mean, added to the harmonic sums and saved after each day. ARCO
    publishes in order, so the first day it has not published ends the pass."""
    base = dict(np.load(OFFSETS)) if OFFSETS.exists() else _empty_sums()
    done = set(pd.DatetimeIndex(base["date"]))
    S = base if out == OFFSETS else _empty_sums()  # a part's file holds its own days only
    days = [d for d in O.lattice(OFFSET_FIRST, until, part) if d not in done]
    print(f"truth: {len(days)} lattice days to pair", flush=True)
    for day in days:
        try:  # HRES first: a day the mirror lacks costs a few index requests, not sixteen ERA5 reads
            H = [hres(day + pd.Timedelta(hours=h)) for h in HOURS]
        except Exception as e:  # noqa: BLE001
            print(f"  {day:%Y-%m-%d}: HRES {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        E = [era5_time(day + pd.Timedelta(hours=h)) for h in HOURS]
        if any(e is None for e in E):
            print(f"  ERA5 not published for {day:%Y-%m-%d}: stopping", flush=True)
            break
        mh, me = day_mean(H), day_mean(E)
        dh, de = derived(mh), derived(me)
        x = _harm(_doy(day))
        S["XtX"] += np.outer(x, x)
        S["XtY3"] += x[:, None, None, None, None] * np.stack([mh[k] - me[k] for k in VARS3])[None]
        S["XtY2"] += x[:, None, None, None] * np.stack([dh[f] - de[f] for f in HB.FIELDS])[None]
        S["date"] = np.r_[S["date"], np.datetime64(day.date())].astype("datetime64[D]")
        _save_sums(out, S)
        print(f"  {day:%Y-%m-%d} paired ({S['date'].size} in the file)", flush=True)
    if out != OFFSETS and not out.exists():  # a part with nothing to pair still hands the merge a file
        _save_sums(out, S)


def merge(paths):
    S = dict(np.load(OFFSETS)) if OFFSETS.exists() else _empty_sums()
    for p in paths:
        P = np.load(p)
        S["date"] = np.r_[S["date"], P["date"]].astype("datetime64[D]")
        for k in ("XtX", "XtY3", "XtY2"):
            S[k] = S[k] + P[k]
    _save_sums(OFFSETS, S)
    print(f"merge: {S['date'].size} lattice days paired", flush=True)


class Offsets:
    """The analysis's mean offset from ERA5 at a common-year day: off3 (VARS3, LEVELS, lat, lon), off2 {field: map}."""

    def __init__(self):
        S = np.load(OFFSETS) if OFFSETS.exists() else _empty_sums()
        self.n = S["date"].size
        if self.n < MIN_PAIRS:
            raise SystemExit(f"{self.n} lattice days paired with ERA5, {MIN_PAIRS} needed: run truth first")
        if S["XtY2"].shape[1] != len(HB.FIELDS):
            raise SystemExit("the offsets were paired for another set of fields: run hbranch-offsets.yml with fresh")
        inv = np.linalg.inv(S["XtX"])
        self.c3 = np.tensordot(inv, S["XtY3"], axes=(1, 0)).astype(np.float32)
        self.c2 = np.tensordot(inv, S["XtY2"], axes=(1, 0)).astype(np.float32)

    def off3(self, doy: float) -> np.ndarray:
        return np.tensordot(_harm(doy).astype(np.float32), self.c3, axes=(0, 0))

    def off2(self, doy: float) -> dict:
        return dict(zip(HB.FIELDS, np.tensordot(_harm(doy).astype(np.float32), self.c2, axes=(0, 0)), strict=True))


# ── archive ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _w3(day: pd.Timestamp) -> Path:
    return ARCH / f"w3_{day:%Y-%m-%d}.npz"


def save_w3(day: pd.Timestamp, times: list[dict]):
    S = {k: np.stack([x[k] for x in times]) for k in times[0]}
    S["t"] = S["t"] - T_OFF
    np.savez_compressed(_w3(day), **{k: v.astype(np.float16 if k in VARS3 else np.float32) for k, v in S.items()})
    _note(_w3(day))


def load_w3(day: pd.Timestamp) -> dict:
    S = {k: v.astype(np.float32) for k, v in np.load(_w3(day)).items()}
    S["t"] = S["t"] + T_OFF
    return S


def _merge_rows(path: Path, new: dict, keys) -> int:
    """Add `new` ({day: {key: array}}) to the npz at `path` in date order (a day already there is replaced) and list
    the file for upload; the days held."""
    old = dict(np.load(path)) if path.exists() else None
    stamp = np.array([np.datetime64(d.date()) for d in new], "datetime64[D]")
    keep = ~np.isin(old["date"], stamp) if old is not None else None
    date = pd.DatetimeIndex(old["date"][keep] if old is not None else []).append(pd.DatetimeIndex(list(new)))
    o = np.argsort(date.values)
    cols = {
        k: np.concatenate(([old[k][keep]] if old is not None else []) + [np.stack([v[k] for v in new.values()])])
        for k in keys
    }
    np.savez_compressed(
        path, date=date.values[o].astype("datetime64[D]"), **{k: v[o].astype(np.float32) for k, v in cols.items()}
    )
    _note(path)
    return date.size


def _files(since: pd.Timestamp | None = None) -> list[Path]:
    return [p for p in sorted(ARCH.glob("hb_????-??.npz")) if since is None or p.stem[3:] >= f"{since:%Y-%m}"]


def held() -> pd.DatetimeIndex:
    """The days in the daily fields' archive (an npz loads its arrays one by one, so this reads no fields)."""
    return pd.DatetimeIndex(np.concatenate([np.load(p)["date"] for p in _files()] or [np.array([], "datetime64[D]")]))


def load(since: pd.Timestamp | None = None) -> tuple[pd.DatetimeIndex, dict]:
    parts = [np.load(p) for p in _files(since)]
    if not parts:
        return pd.DatetimeIndex([]), {f: np.zeros((0, HB.LAT.size, HB.LON.size), np.float32) for f in HB.FIELDS}
    return pd.DatetimeIndex(np.concatenate([p["date"] for p in parts])), {
        f: np.concatenate([p[f] for p in parts]) for f in HB.FIELDS
    }


def _flush(new: dict):
    for m in sorted({d.strftime("%Y-%m") for d in new}):
        sub = {d: f for d, f in new.items() if d.strftime("%Y-%m") == m}
        print(f"  {m}: +{len(sub)} days, {_merge_rows(ARCH / f'hb_{m}.npz', sub, HB.FIELDS)} held", flush=True)


def winds(days, off: Offsets) -> T.Winds:
    """The trajectories' winds from the 3-D store over consecutive `days`, each day less its offset."""
    u, v, w, pt = [], [], [], []
    asc = [list(HB.LEVELS).index(int(p)) for p in T.P_ASC]
    for d in days:
        S, o = load_w3(d), off.off3(_doy(d))
        for i in range(len(HOURS)):
            c = {k: S[k][i] - o[j] for j, k in enumerate(VARS3)}
            u.append(c["u"][asc])
            v.append(c["v"][asc])
            w.append(c["w"][asc])
            pt.append(HB.tropopause_surface(c["u"], c["v"], c["t"])["p"].astype(np.float32))
    return T.Winds(days[0].timestamp(), *(np.stack(x) for x in (u, v, w, pt)))


def _covered(store: set, d: pd.Timestamp) -> bool:
    return all(x in store for x in pd.date_range(d - pd.Timedelta(days=T.DAYS_BACK), d))


def _released() -> set:
    """The releases in traj.npz; a table of another release layout (levels, bins) is dropped, to be traced again."""
    if not TRAJ.exists():
        return set()
    t = np.load(TRAJ)
    if t["hist"].shape[1:] != (len(HB.SUBS), len(T.RELEASE_P), T.NBIN + 1):
        TRAJ.unlink()
        return set()
    return set(pd.DatetimeIndex(t["date"]))


def releases(off: Offsets, upto: pd.Timestamp):
    """Trace every 00 UTC release to `upto` that the 3-D store covers and traj.npz lacks, twenty at a time."""
    done = _released()
    store = {pd.Timestamp(p.stem[3:]) for p in ARCH.glob("w3_*.npz")}
    todo = [d for d in sorted(store) if d <= upto and d not in done and _covered(store, d)]
    print(f"trajectories: {len(todo)} releases", flush=True)
    for b in range(0, len(todo), 20):
        batch = todo[b : b + 20]
        W = winds(pd.date_range(batch[0] - pd.Timedelta(days=T.DAYS_BACK), batch[-1]), off)
        hist = T.trace_many(W, [d.timestamp() for d in batch])
        _merge_rows(TRAJ, {d: {"hist": h} for d, h in zip(batch, hist, strict=True)}, ("hist",))
        print(f"  released {batch[0]:%Y-%m-%d} .. {batch[-1]:%Y-%m-%d}", flush=True)


def update(until: pd.Timestamp, fresh: bool = False):
    """Archive the days from FIRST to `until` not yet held whose four analyses are on the mirror, keeping the 3-D fields
    of every day a frame's release still to be traced reaches back to; release the trajectories the store now covers;
    prune the store to STORE_DAYS. fresh (after a change to what a day holds) archives and releases every day again,
    from the store where it still holds the day."""
    if fresh:
        for f in [*_files(), TRAJ]:
            f.unlink(missing_ok=True)
    have, done = set(held()), _released()
    todo = [d for d in pd.date_range(until - pd.Timedelta(days=DAYS + WIN), until) if d not in done]
    oldest3 = (todo[0] if todo else until) - pd.Timedelta(days=T.DAYS_BACK)  # the store reaches back for these only
    want = [d for d in pd.date_range(FIRST, until) if d not in have or (d >= oldest3 and not _w3(d).exists())]
    print(f"update: {len(want)} days to archive", flush=True)
    ARCH.mkdir(parents=True, exist_ok=True)
    new = {}
    for day in want:
        if _w3(day).exists():
            S = load_w3(day)
            times = [{k: S[k][i] for k in S} for i in range(len(HOURS))]
        else:
            try:
                times = [hres(day + pd.Timedelta(hours=h)) for h in HOURS]
            except Exception as e:  # noqa: BLE001
                print(f"  {day:%Y-%m-%d}: {type(e).__name__} {str(e)[:80]}", flush=True)
                continue
            if day >= oldest3:
                save_w3(day, times)
        if day not in have:
            new[day] = derived(day_mean(times))
        if len(new) == 10:
            _flush(new)
            new = {}
    if new:
        _flush(new)
    releases(Offsets(), until)
    for f in ARCH.glob("w3_*.npz"):
        if pd.Timestamp(f.stem[3:]) < until - pd.Timedelta(days=STORE_DAYS):
            _note(f, removed=True)
            f.unlink()


# ── windows and the climatology ───────────────────────────────────────────────────────────────────────────────────────
def week(date: pd.DatetimeIndex, D: dict, end: pd.Timestamp, off: Offsets):
    """(fields less the analysis offset, midpoint common-year day) of the 7-day window ending `end`; None if a day is
    missing."""
    want = pd.date_range(end - pd.Timedelta(days=WIN - 1), end)
    pos = {d: i for i, d in enumerate(date)}
    if any(d not in pos for d in want):
        return None
    mid = _doy(end - pd.Timedelta(days=(WIN - 1) / 2))
    o = off.off2(mid)
    rows = [pos[d] for d in want]
    return {f: D[f][rows].mean(0) - o[f] for f in HB.FIELDS}, mid


def traj_week(tdate: pd.DatetimeIndex, hist: np.ndarray, end: pd.Timestamp) -> dict:
    """The trajectory numbers of the 7 releases ending `end`; {} if one is missing."""
    want = pd.date_range(end - pd.Timedelta(days=WIN - 1), end)
    pos = {d: i for i, d in enumerate(tdate)}
    if any(d not in pos for d in want):
        return {}
    return C.traj_row(hist[[pos[d] for d in want]].sum(0))


def clim_map(cl: xr.Dataset, field: str, doy: float) -> np.ndarray:
    c = cl["coef"].sel(field=field).values
    return np.tensordot(_harm(doy), c, axes=(0, 0))


def clim_index(cl: xr.Dataset, key: str, doy, stat: str) -> np.ndarray:
    """A climatological stat of an index at common-year days (29 Feb = 59.5 takes its two neighbours' mean)."""
    doy = np.atleast_1d(np.asarray(doy, float))
    s = cl["index"].sel(key=key, stat=stat)
    return 0.5 * (s.sel(doy=np.floor(doy).astype(int)).values + s.sel(doy=np.ceil(doy).astype(int)).values)


# ── maps ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The maps are drawn in their own (Pacific-centred) coordinates, so cartopy has no polygon to reproject: x = longitude
# less CENTRE, on the grid, the seam at 21E (in Africa, every basin whole) closed by a repeated column.
CENTRE = 201.0
PROJ = ccrs.PlateCarree(central_longitude=CENTRE)
_ORDER = np.argsort((HB.LON - CENTRE + 180.0) % 360.0)
XN = np.r_[(HB.LON[_ORDER] - CENTRE + 180.0) % 360.0 - 180.0, 180.0]


def native(f: np.ndarray) -> np.ndarray:
    g = f[:, _ORDER]
    return np.concatenate([g, g[:, :1]], axis=1)


def _xn(lon: np.ndarray) -> np.ndarray:
    return (np.asarray(lon) - CENTRE + 180.0) % 360.0 - 180.0


def geo(ax, title: str, extent: tuple = EXTENT):
    ax.coastlines(linewidth=1.0, color="#111", zorder=5)
    ax.set_extent([-180.0, 180.0, *extent], crs=PROJ)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5, xlocs=range(-180, 181, 60), ylocs=[-30, 0, 30]
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7.5}
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")


def _panel(fig, rect, cax, field, levels, title, label):
    ax = fig.add_axes(rect, projection=PROJ)
    cf = ax.contourf(XN, HB.LAT, native(field), levels=levels, cmap="RdBu_r", extend="both")
    geo(ax, title)
    cb = fig.colorbar(cf, cax=fig.add_axes(cax), orientation="vertical")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    return ax


def layout(n: int, W: float = 13.0, extent: tuple = EXTENT):
    """Figure size and (axes, colorbar) rects in figure fractions for n map rows of the extent's aspect, in inches: a
    left margin for the latitude labels, a title and longitude labels around each row, a header and a footer."""
    left, mw, top, foot, block = 0.6, 11.3, 0.5, 0.5, 0.34
    mh = mw * (extent[1] - extent[0]) / 360.0
    H = top + n * (mh + 2 * block) + foot
    rects = []
    for i in range(n):
        y = H - top - i * (mh + 2 * block) - block - mh
        rects.append(
            ((left / W, y / H, mw / W, mh / H), ((left + mw + 0.25) / W, (y + 0.1) / H, 0.16 / W, (mh - 0.2) / H))
        )
    return (W, H), rects, left


def height_km(p: np.ndarray) -> np.ndarray:
    """The standard atmosphere's height (km) of pressure p (hPa): its lapse rate below 226.32 hPa, isothermal above."""
    p = np.asarray(p, float)
    return np.where(p >= 226.32, 44.3308 * (1.0 - (p / 1013.25) ** 0.190263), 11.0 + 6.34162 * np.log(226.32 / p))


def draw_dt(F: dict, day: pd.Timestamp, out: Path):
    """The day's outflow on the dynamic tropopause: the surface's height, and the divergent wind on it as arrows."""
    size, rects, left = layout(1)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(rects[0][0], projection=PROJ)
    cf = ax.contourf(XN, HB.LAT, native(height_km(F["pdt"])), levels=DT_KM, cmap="viridis", extend="both")
    s = 5
    ax.quiver(
        XN[:-1:s],
        HB.LAT[::s],
        native(F["uout"])[::s, :-1:s],
        native(F["vout"])[::s, :-1:s],
        scale=200,
        width=0.0012,
        color="#222",
    )
    geo(ax, "Divergent wind on the dynamic tropopause · the surface's height")
    cb = fig.colorbar(cf, cax=fig.add_axes(rects[0][1]), orientation="vertical")
    cb.set_label("height of the dynamic tropopause (km)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    W, H = size
    fig.text(
        left / W,
        1 - 0.3 / H,
        f"Hadley outflow on the dynamic tropopause — {day:%Y-%m-%d}, daily mean (IFS HRES analyses)",
        fontsize=13,
        fontweight="bold",
        va="center",
    )
    fig.text(
        left / W,
        0.12 / H,
        f"dynamic tropopause = the {HB.DT_PVU:.0f}-PVU surface, or the {HB.DT_THETA:.0f} K isentrope where that lies lower "
        "(the deep tropics); height from its pressure in the standard atmosphere · arrows = the divergent wind on it\n"
        "the analysis less its mean offset from ERA5",
        fontsize=8.2,
        color="#555",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def draw_map(F: dict, cl: xr.Dataset, mid: float, idx: dict, end: pd.Timestamp, out: Path):
    """The week's descent and subtropical ridges against ERA5."""
    size, rects, left = layout(2)
    fig = plt.figure(figsize=size)
    w = HB.PA_S_TO_HPA_DAY * F["wmid"]
    ax = _panel(
        fig,
        *rects[0],
        HB.large_scale(w - HB.PA_S_TO_HPA_DAY * clim_map(cl, "wmid", mid), efold=W_EFOLD),
        LEV["desc"],
        f"Descent · {HB.DESC_LAYER[0]:.0f}–{HB.DESC_LAYER[1]:.0f} hPa vertical motion",
        "anomaly (hPa day⁻¹, + = more descent)",
    )
    ax.contour(XN, HB.LAT, native(HB.large_scale(w, efold=W_EFOLD)), levels=[20.0], colors="#111", linewidths=0.8)
    ax = _panel(
        fig,
        *rects[1],
        F["msl"] - clim_map(cl, "msl", mid),
        LEV["msl"],
        "Subtropical ridges · sea-level pressure and the 588-dam line",
        "SLP anomaly (hPa)",
    )
    for zf, ls, lw in ((F["z500"], "-", 1.6), (clim_map(cl, "z500", mid), "--", 1.0)):
        ax.contour(XN, HB.LAT, native(zf), levels=[HB.Z588], colors="#111", linewidths=lw, linestyles=ls)
    for k in HB.HIGHS:
        ax.text(
            idx[f"hlon_{k}"],
            idx[f"hlat_{k}"],
            f"H\n{idx[f'high_{k}']:.0f}",
            transform=PC,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111",
        )
    W, H = size
    fig.text(
        left / W,
        1 - 0.3 / H,
        f"Subtropical descent and ridges — 7-day mean ending {end:%Y-%m-%d} (IFS HRES analyses)",
        fontsize=13,
        fontweight="bold",
        va="center",
    )
    fig.text(
        left / W,
        0.12 / H,
        "colour = anomaly against ERA5 1991–2020 at the week's midpoint, the analysis less its mean offset from ERA5 · "
        f"top: vertical motion strongly smoothed (degree {W_EFOLD:.0f} damped by 1/e), black line = descent of 20 hPa/day"
        "\nbottom: solid = this week's 588-dam z500 line, dashed = the normal's (at the "
        f"{int(C.EPOCH)} level of its trend), H = the highs' central pressure (hPa)",
        fontsize=8.2,
        color="#555",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def draw_transit(off: Offsets, latest: pd.Timestamp, out: Path):
    """The week's releases ending `latest`: in cells of POOL_LON release points, the median transit time of the cell's
    parcels over the seven days (where at least four were traced to the outflow layer, else grey), and a sample of the
    latest release's parcels traced back to where they left it."""
    days = pd.date_range(latest - pd.Timedelta(days=WIN - 1), latest)
    W = winds(pd.date_range(days[0] - pd.Timedelta(days=T.DAYS_BACK), latest), off)
    taus = []
    for d in days:
        tau, slat, slon, tracks = T.trace(W, d.timestamp(), keep_tracks=d == latest)
        taus.append(tau)
    taus = np.stack(taus)
    lat, lon, _, _ = T.release_points()
    nla, nlo, k = 2 * T.RELEASE_LAT.size, T.RELEASE_LON.size, POOL_LON
    pool = taus.reshape(len(days), nla, nlo // k, k).transpose(1, 2, 0, 3).reshape(nla, nlo // k, -1)
    med = np.full(pool.shape[:2], np.nan)
    enough = np.isfinite(pool).sum(-1) >= 4
    med[enough] = np.nanmedian(pool[enough], axis=-1)
    xc = _xn(T.RELEASE_LON[::k] + T.RES * (k - 1) / 2)
    order = np.argsort(xc)
    xe = np.r_[xc[order] - k * T.RES / 2, xc[order][-1] + k * T.RES / 2]
    grid = med[:, order]
    size, rects, left = layout(1)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(rects[0][0], projection=PROJ)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.BoundaryNorm(np.arange(6.0, 25.0, 2.0), cmap.N, extend="both")
    rows = lat.reshape(nla, nlo)[:, 0]
    for h in (rows > 0, rows < 0):  # each hemisphere's band of cells, edges half a cell either side of the rows
        la = np.sort(rows[h])
        ye = np.r_[la - T.RES / 2, la[-1] + T.RES / 2]
        g = grid[h][np.argsort(rows[h])]
        ax.axhspan(ye[0], ye[-1], color="#d9d9d9", lw=0, zorder=0)  # grey shows through the cells left empty
        pm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(g), cmap=cmap, norm=norm, shading="flat")
    for n in np.flatnonzero(np.isfinite(taus[-1]))[::12]:
        ok = np.isfinite(tracks[:, 0, n])
        x, y = np.r_[_xn(lon[n]), _xn(tracks[ok, 1, n])], np.r_[lat[n], tracks[ok, 0, n]]
        brk = np.flatnonzero(np.abs(np.diff(x)) > 180.0) + 1  # across the seam: a break, not a line round the world
        for xs, ys in zip(np.split(x, brk), np.split(y, brk), strict=True):
            ax.plot(xs, ys, color="#333", lw=0.5, alpha=0.7)
        ax.plot(_xn(slon[n]), slat[n], "o", color="#c0392b", ms=2.2)
    geo(ax, f"Transit time to 500 hPa · median of the week's releases ending {latest:%Y-%m-%d} (days)")
    cb = fig.colorbar(pm, cax=fig.add_axes(rects[0][1]), orientation="vertical")
    cb.set_label("days since it left the outflow layer", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    Wi, H = size
    fig.text(
        left / Wi,
        1 - 0.3 / H,
        "How long ago the subtropical air sinking through 500 hPa left the tropical outflow layer (IFS HRES winds)",
        fontsize=13,
        fontweight="bold",
        va="center",
    )
    fig.text(
        left / Wi,
        0.12 / H,
        f"each {POOL_LON * T.RES:.0f}° × {T.RES:.0f}° cell: the median of its {POOL_LON * WIN} parcels of the week "
        f"({POOL_LON} release points, {WIN} days) traced back {T.DAYS_BACK} days, grey where fewer than four were traced "
        "to the outflow layer (equatorward of {T.OUTFLOW_LAT:.0f}°, within "
        f"{HB.OUT_DEPTH:.0f} hPa below the dynamic tropopause)\nlines: one in twelve of the latest release's traced parcels, back "
        "to where it left the layer (dot)",
        fontsize=8.2,
        color="#555",
    )
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def _write_paths(d: Path, stem: str, W: T.Winds, t_start: float, hours: float) -> tuple[dict, int]:
    """Forward paths of every seed (DROP_LAT x DROP_LON) at each DROP_P from t_start (s) for `hours`, one int16 file a
    level in d of (seed, record, [lat x 100, (lon - 180) x 100, p x 10]); the files by level and the records a path."""
    la, lo = (x.ravel() for x in np.meshgrid(DROP_LAT, DROP_LON, indexing="ij"))
    files = {}
    for p in DROP_P:
        P = T.paths(W, t_start, la, lo, np.full(la.size, p), hours)
        q = np.stack([P[:, 0] * 100.0, (P[:, 1] - 180.0) * 100.0, P[:, 2] * 10.0], -1).transpose(1, 0, 2)
        files[str(int(p))] = name = f"{stem}_{int(p)}.bin"
        (d / name).write_bytes(np.round(q).astype("<i2").tobytes())
    return files, int(P.shape[0])


def _drop_dir(name: str) -> Path:
    d = ANIM / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.bin"):  # the directory is swapped onto the frames branch whole
        old.unlink()
    return d


def _drop_manifest(name: str, releases: list):
    doc = {
        "ver": int(pd.Timestamp.now("UTC").timestamp()),
        "dir": name,
        "step_h": int(T.DT_DATA // 3600),
        "seeds": {"lat": DROP_LAT.tolist(), "lon": DROP_LON.tolist()},
        "levels": [int(p) for p in DROP_P],
        "releases": releases,
        "basemap": {"src": "assets/sst/hbranch_basemap.webp", "centre": CENTRE, "lat": list(BASE_EXTENT)},
    }
    (ANIM / f"{name}_manifest.json").write_text(json.dumps(doc))


def drop(off: Offsets, last: pd.Timestamp):
    """The interactive map's analysed releases: forward paths released at 00 UTC DROP_BACK days before `last` and
    followed to DAYS_BACK days or to the last analysis, on the analyses less their offset from ERA5."""
    W = winds(pd.date_range(last - pd.Timedelta(days=max(DROP_BACK)), last), off)
    t_end = W.t0 + (W.u.shape[0] - 1) * T.DT_DATA
    d, rel = _drop_dir(DROP_DIR), []
    for back in DROP_BACK:
        r = last - pd.Timedelta(days=back)
        hours = min(T.DAYS_BACK * 24.0, (t_end - r.timestamp()) / 3600.0)
        files, n = _write_paths(d, f"R{r:%Y%m%d}", W, r.timestamp(), hours)
        rel.append({"date": f"{r:%Y-%m-%d}", "records": n, "files": files})
    _drop_manifest(DROP_DIR, rel)
    print(f"  drop: {len(rel)} releases x {len(DROP_P)} levels x {DROP_LAT.size * DROP_LON.size} seeds", flush=True)


def latest_run(now: pd.Timestamp) -> pd.Timestamp:
    """The newest 00 or 12 UTC HRES run whose last step (FC_HOURS) is on the mirror."""
    import rangefetch as rf

    t = now.floor("12h")
    for _ in range(4):
        try:
            rf.fetch_index(f"{t:%Y%m%d}", f"{t:%H}", "ifs", FC_HOURS, "fc", stream="oper")
            return t
        except rf.IndexUnavailable as e:
            if not e.not_found:
                raise
        t -= pd.Timedelta(hours=12)
    raise SystemExit(f"no HRES run with step {FC_HOURS} on the mirror in the last two days")


def forecast(now: pd.Timestamp):
    """The interactive map's forecast release: forward paths from the newest HRES run reaching FC_HOURS, on its u, v,
    omega every 6 hours, as forecast (the model's drift is not removed: it is measured for the ensembles only)."""
    init = latest_run(now)
    mp = ANIM / f"{FC_DIR}_manifest.json"
    if mp.exists() and json.loads(mp.read_text())["releases"][0]["init"] == f"{init:%Y-%m-%dT%H}":
        print(f"forecast: the {init:%Y-%m-%d %H}Z run is drawn already", flush=True)
        return
    asc = [list(HB.LEVELS).index(int(p)) for p in T.P_ASC]
    u, v, w = [], [], []
    for step in range(0, FC_HOURS + 1, int(T.DT_DATA // 3600)):
        F = hres(init, step, params=("u", "v", "w"), extras=False)
        u.append(F["u"][asc])
        v.append(F["v"][asc])
        w.append(F["w"][asc])
    W = T.Winds(init.timestamp(), np.stack(u), np.stack(v), np.stack(w))
    files, n = _write_paths(_drop_dir(FC_DIR), f"F{init:%Y%m%d%H}", W, init.timestamp(), FC_HOURS)
    _drop_manifest(FC_DIR, [{"date": f"{init:%Y-%m-%d}", "init": f"{init:%Y-%m-%dT%H}", "records": n, "files": files}])
    print(f"forecast: the {init:%Y-%m-%d %H}Z run, {n} records a path", flush=True)


def basemap(out: Path):
    """The interactive map's background: Pacific-centred, BASE_EXTENT, BASE_PX pixels a degree, nothing outside the
    map (the page puts a point at x = (lon - CENTRE + 180 mod 360) / 360, y = (lat_max - lat) / span)."""
    import cartopy.feature as cfeature

    span = BASE_EXTENT[1] - BASE_EXTENT[0]
    fig = plt.figure(figsize=(360 * BASE_PX / 100, span * BASE_PX / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1), projection=PROJ)
    ax.set_extent([-180.0, 180.0, *BASE_EXTENT], crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="#ece9e2", edgecolor="none")
    ax.coastlines(linewidth=0.9, color="#222")
    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 20), linewidth=0.4, color="#999", alpha=0.6)
    ax.axhspan(DROP_LAT[0], DROP_LAT[-1], color="#2b6cb0", alpha=0.06, lw=0)  # where a particle can be dropped
    ax.axhline(0.0, color="#777", lw=0.6)
    ax.spines["geo"].set_visible(False)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


# ── tracker ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _bands(ax, cl, key, x, doy):
    q = {s: clim_index(cl, key, doy, s) for s in ("p10", "p25", "p50", "p75", "p90")}
    ax.fill_between(x, q["p10"], q["p90"], color="#c9d8e6", lw=0)
    ax.fill_between(x, q["p25"], q["p75"], color="#9dbad3", lw=0)
    ax.plot(x, q["p50"], color="#8a5a00", lw=1.2, ls=(0, (5, 2)))


TRACKERS = (
    (
        "hbranch_flow.webp",
        "Hadley outflow and subtropical descent",
        (("out_{}", "Outflow", "m s⁻¹"), ("desc_{}", "Descent", "hPa day⁻¹")),
        ROWS,
        "outflow = poleward divergent wind on the dynamic tropopause over 10–30°, descent = "
        f"{HB.DESC_LAYER[0]:.0f}–{HB.DESC_LAYER[1]:.0f} hPa ω over 15–35°, in each basin's eastern subsidence region "
        "(hemispheres: all longitudes)",
    ),
    (
        "hbranch_ridges.webp",
        "Subtropical ridges and highs",
        (("r588_{}", "588-dam ridge area", "10⁶ km²"), ("high_{}", "Subtropical high", "hPa")),
        tuple(HB.HIGHS),
        f"ridge area = where z500 ≥ 588 dam in the basin's 10–45° band, ERA5 at the {int(C.EPOCH)} level of its "
        "1991–2020 trend · high = the subtropical high's central sea-level pressure",
    ),
    (
        "hbranch_turnover.webp",
        "How fast the Hadley cells turn over",
        (("transit_{}_500", "Transit to 500 hPa", "days"), ("share_{}_500", "Share from the outflow", "%")),
        ROWS,
        f"parcels at 500 hPa over 15–33°, released daily at 00 UTC on a {T.RES:.0f}° grid and traced back {T.DAYS_BACK} "
        f"days · tropical outflow layer = equatorward of {T.OUTFLOW_LAT:.0f}°, within {HB.OUT_DEPTH:.0f} hPa below the "
        "dynamic tropopause\n% = the area-weighted share of the parcels that were in that layer at some time "
        f"in the {T.DAYS_BACK} days · transit = their median time since they last left it; shorter = a faster-turning "
        "cell",
    ),
)


FIELD_TRACKERS = ("hbranch_flow.webp", "hbranch_ridges.webp")  # the trackers the ensembles forecast


def _forecast(last: pd.Timestamp) -> dict | None:
    """FC, unless it is stale (its run more than two days before the last analysed window: it would sit behind it)."""
    fc = json.loads(FC.read_text()) if FC.exists() else None
    return fc if fc and pd.Timestamp(fc["init"]) >= last - pd.Timedelta(days=2) else None


def draw_trackers(series: pd.DataFrame, cl: xr.Dataset, only: tuple | None = None):
    """The trackers (all of TRACKERS, or the files in `only`): the analysed windows against ERA5's, with each
    ensemble's forecast windows (FC) where they passed their skill test."""
    fc = _forecast(series.index[-1])
    valid = pd.DatetimeIndex(fc["valid"]) if fc else pd.DatetimeIndex([])
    x = series.index.union(valid)
    doy = noleap_doy(x - pd.Timedelta(days=(WIN - 1) / 2))
    last = noleap_doy(series.index[-1:] - pd.Timedelta(days=(WIN - 1) / 2))
    for name, title, cols, rows, what in TRACKERS:
        if only and name not in only:
            continue
        drawn = []  # the ensembles drawn, in FC's order
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(12.4, 1.8 * len(rows) + 1.0), sharex=True)
        for i, s in enumerate(rows):
            for j, (pat, label, unit) in enumerate(cols):
                ax, key = axes[i, j], pat.format(s)
                _bands(ax, cl, key, x, doy)
                ax.plot(series.index, series[key], color="#0b3d6b", lw=1.8)
                plumes = {m: d["keys"][key] for m, d in (fc or {"models": {}})["models"].items() if key in d["keys"]}
                for m, q in plumes.items():
                    v = valid[: len(q["p50"])]
                    lo, mid, hi = (np.array(q[k], float) for k in ("p10", "p50", "p90"))  # null: a gap
                    ax.fill_between(v, lo, hi, color=MODEL_C[m], alpha=0.13, lw=0)
                    ax.plot(v, mid, color=MODEL_C[m], lw=1.6)
                    if m not in drawn:
                        drawn.append(m)
                if plumes:
                    ax.axvline(valid[0], color="0.55", lw=0.8, ls=":")
                now = series[key].iloc[-1]
                mu, sd = (clim_index(cl, key, last, st)[0] for st in ("mean", "sd"))
                ax.set_title(
                    f"{SECTOR_NAME[s]} · {label} · latest {now:.1f} {unit} ({(now - mu) / sd:+.1f}σ)",
                    fontsize=9,
                    loc="left",
                    fontweight="bold",
                )
                ax.set_ylabel(unit, fontsize=8)
                ax.grid(alpha=0.25, lw=0.5)
                ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        fig.suptitle(
            f"{title} — analysed {'and forecast ' if drawn else ''}7-day windows against ERA5 1991–2020",
            fontsize=12.5,
            fontweight="bold",
            x=0.01,
            ha="left",
        )
        foot = (
            "blue = IFS HRES analyses (00, 06, 12, 18 UTC), 7-day windows ending on the day, less the analysis's offset "
            "from ERA5 · bands = ERA5's own 7-day windows 1991–2020 (10–90 %, 25–75 %), median dashed; σ = ERA5's spread"
        )
        if drawn:
            foot += (
                f"\nforecast, the {fc['init']} 00Z runs: each ensemble's median and 10–90 % ("
                + ", ".join(f"{fc['models'][m]['label']} {fc['models'][m]['members']}" for m in drawn)
                + " members), each member less its model's drift against ERA5, to the last window with significant "
                "skill on archived runs"
            )
            handles = [plt.Line2D([], [], color=MODEL_C[m], lw=1.6) for m in drawn]
            axes[0, -1].legend(handles, [fc["models"][m]["label"] for m in drawn], fontsize=7, frameon=False)
        foot += f"\n{what}"
        fig.text(0.01, 0.004, foot, fontsize=7.6, color="#555")
        h = fig.get_figheight()
        fig.tight_layout(rect=(0, (0.14 * (foot.count("\n") + 1) + 0.06) / h, 1, 0.975))
        fig.savefig(OUT / name, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
        plt.close(fig)
        print(f"wrote {OUT / name}", flush=True)


def loop(name: str, label: str, frames: dict, rebuild: bool) -> list:
    """The loop `name` made exactly the frames in `frames` ({date key: (file, label, draw)}): the ones the runner does
    not hold are drawn (all of them with rebuild), the rest of the directory is deleted (it is swapped onto the frames
    branch whole), and the manifest written; its frames."""
    d, mp = ANIM / name, ANIM / f"{name}_manifest.json"
    d.mkdir(parents=True, exist_ok=True)
    drawn = set()
    if not rebuild and mp.exists():  # a frame counts as drawn only if the runner holds it (fetched from the branch)
        drawn = {f["date"] for f in json.loads(mp.read_text())["regions"][name]["frames"] if (d / f["file"]).exists()}
    for key, (file, _, draw) in frames.items():
        if key not in drawn:
            draw(d / file)
    for old in {q.name for q in d.glob("*.webp")} - {f for f, _, _ in frames.values()}:
        (d / old).unlink()
    out = [{"file": f, "date": k, "label": lab, "idx": i} for i, (k, (f, lab, _)) in enumerate(sorted(frames.items()))]
    regions = {name: {"label": label, "n_frames": len(out), "frames": out}}
    mp.write_text(json.dumps({"ver": int(pd.Timestamp.now("UTC").timestamp()), "days": len(out), "regions": regions}))
    print(f"  {name}: {len(out)} frames, through {out[-1]['date']}", flush=True)
    return out


def render(rebuild: bool = False):
    """The weekly loop (a frame per day of the last DAYS with a full window), the daily loop, the stills, the trackers,
    the latest week's trajectories and the interactive map's paths."""
    off = Offsets()
    last = held().max()
    date, D = load(last - pd.Timedelta(days=DAYS + WIN))
    T_ = np.load(TRAJ)
    tdate, hist = pd.DatetimeIndex(T_["date"]), T_["hist"]
    cl = xr.open_dataset(C.OUT)
    weekly, rows, ends = {}, [], []
    for end in date[date > last - pd.Timedelta(days=DAYS)]:
        w = week(date, D, end, off)
        tr = traj_week(tdate, hist, end)
        if w is None or not tr:
            print(f"  {end:%Y-%m-%d}: not drawn (a day or a release missing)", flush=True)
            continue
        F, mid = w
        idx = HB.indices(F)
        rows.append(idx | tr)
        ends.append(end)
        weekly[f"{end:%Y-%m-%d}"] = (
            f"D{end:%Y%m%d}.webp",
            f"week ending {end:%a %b %d, %Y}",
            partial(draw_map, F, cl, mid, idx, end),
        )
    if not ends:
        raise SystemExit("no complete week to draw")
    o = off.off2(mid)  # what the correction does to the latest week (corrected less raw)
    raw = HB.indices({f: F[f] + o[f] for f in HB.FIELDS})
    keys = ("out_nh", "out_sh", "desc_nh", "desc_sh", "r588_npac", "r588_spac", "high_npac", "high_spac")
    print(
        f"  offset ({off.n} pairs), latest week: " + ", ".join(f"{k} {idx[k] - raw[k]:+.2f}" for k in keys), flush=True
    )
    frames = loop(DIR, "Subtropical descent and ridges, analysed weeks", weekly, rebuild)
    shutil.copyfile(ANIM / DIR / frames[-1]["file"], OUT / "hbranch_map.webp")
    daily = {}
    for i, day in enumerate(date):
        if day > last - pd.Timedelta(days=DAYS):
            o = off.off2(_doy(day))
            Fd = {f: D[f][i] - o[f] for f in HB.FIELDS}
            daily[f"{day:%Y-%m-%d}"] = (f"D{day:%Y%m%d}.webp", f"{day:%a %b %d, %Y}", partial(draw_dt, Fd, day))
    frames = loop(DT_DIR, "Hadley outflow on the dynamic tropopause, daily", daily, rebuild)
    shutil.copyfile(ANIM / DT_DIR / frames[-1]["file"], OUT / "hbranch_dt.webp")
    draw_trackers(pd.DataFrame(rows, index=pd.DatetimeIndex(ends)), cl)
    draw_transit(off, tdate.max(), OUT / "hbranch_traj.webp")
    drop(off, last)
    basemap(OUT / "hbranch_basemap.webp")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--part", default="0/1", help="k/n: every n-th lattice day from the k-th")
    t.add_argument("--out", default=None, help="a part's own file (default: OFFSETS)")
    m = sub.add_parser("merge")
    m.add_argument("parts", nargs="+")
    u = sub.add_parser("update")
    u.add_argument("--fresh", action="store_true", help="archive and release every day again")
    sub.add_parser("forecast")
    r = sub.add_parser("render")
    r.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    until = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)  # the last complete day
    if a.cmd == "truth":
        k, n = (int(x) for x in a.part.split("/"))
        truth(until, (k, n), Path(a.out) if a.out else OFFSETS)
    elif a.cmd == "merge":
        merge(a.parts)
    elif a.cmd == "update":
        update(until, a.fresh)
    elif a.cmd == "forecast":
        forecast(pd.Timestamp.now("UTC").tz_localize(None))
    else:
        render(a.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
