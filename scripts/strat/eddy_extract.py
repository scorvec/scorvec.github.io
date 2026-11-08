#!/usr/bin/env python3
"""Daily zonal-mean wind and eddy momentum flux by latitude, ERA5 and CMIP6 reduced alike, for testing whether the
models' tropospheric eddy feedback is weak and whether that weakens the surface response to sudden stratospheric
warmings (eddy_feedback.py). Runs in Actions (.github/workflows/eddy-feedback.yml), one CMIP6 run or a span of ERA5
years per job, reading the public Pangeo CMIP6 and WeatherBench 2 Google Cloud mirrors anonymously; nothing raw is kept.

From daily-mean u and v (CMIP6's day table; ERA5's four 6-hourly fields averaged first, so both see the same eddies),
on 20-87.5N every 2.5 deg, October to May:
  u      zonal-mean u, m/s, at the LEVELS the source has (CMIP6 plev8: 10, 50, 100, 250, 500, 700, 850 hPa; ERA5 from
         WeatherBench 2: 50-850 hPa, no 10 hPa)
  uv     the zonal mean of u*v* (departures from the zonal mean), m2/s2, at the same levels
  slp    zonal-mean sea-level pressure, hPa
  slp_p  sea-level pressure projected on the CMIP6 post-SSW pattern (ssw_imprint), hPa
  uv6    ERA5 only: u*v* from each 6-hourly field, then averaged (what daily means miss)
Each with year, month and day of the source's own calendar, which is stored with it.

    python eddy_extract.py plan --catalog pangeo-cmip6.csv --members 3 --era5-span 4
    python eddy_extract.py cmip6 --catalog pangeo-cmip6.csv --model CanESM5 --member r1i1p1f1 --out out/
    python eddy_extract.py era5 --years 1959-1962 --out out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_coupling_extract as CE

LEVELS = (10, 50, 100, 200, 250, 300, 400, 500, 600, 700, 850)
LAT = np.arange(20.0, 88.0, 2.5)
MONTHS = (10, 11, 12, 1, 2, 3, 4, 5)
CMIP6_YEARS = (1950, 2014)
ERA5_YEARS = (1959, 2023)
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
READ_BYTES = 1.2e9  # the most of one variable read at once


# ----------------------------------------------------------------------------------------------------- reductions ---
def to_lat(x: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The last axis, on ascending latitudes `lat`, linearly onto LAT."""
    j = np.clip(np.searchsorted(lat, LAT), 1, len(lat) - 1)
    f = (LAT - lat[j - 1]) / (lat[j] - lat[j - 1])
    return x[..., j - 1] * (1 - f) + x[..., j] * f


def zonal(u: np.ndarray, v: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., lat, lon) u and v -> zonal-mean u and [u*v*] on LAT; points below ground (NaN) are left out."""
    u, v = u.astype("float64"), v.astype("float64")
    um, vm = np.nanmean(u, -1), np.nanmean(v, -1)
    uv = np.nanmean((u - um[..., None]) * (v - vm[..., None]), -1)
    return to_lat(um, lat).astype("float32"), to_lat(uv, lat).astype("float32")


class Pressure:
    """Zonal mean and post-SSW pattern projection of daily sea-level pressure (hPa) on one grid."""

    def __init__(self, lat, lon, model):
        self.lat = np.asarray(lat, float)
        self.proj = CE.Fields(lat, lon, model).proj[0]
        self.boxes = CE.SI.Boxes(lat, lon)

    def __call__(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return to_lat(np.nanmean(p, -1), self.lat).astype("float32"), (self.boxes(p) * self.proj).sum((-2, -1))


def save(dest: Path, ymd: np.ndarray, calendar: str, levels, res: dict) -> None:
    part = dest.with_suffix(".part.npz")
    np.savez_compressed(
        part, year=ymd[:, 0], month=ymd[:, 1], day=ymd[:, 2], calendar=calendar, levels=np.array(levels, "int16"),
        lat=LAT.astype("float32"), **res,
    )  # fmt: skip
    part.replace(dest)


# ---------------------------------------------------------------------------------------------------------- CMIP6 ---
TABLE = {"ua": "day", "va": "day", "psl": "day"}


def catalog(path: str) -> pd.DataFrame:
    c = pd.read_csv(path)
    return c[(c.experiment_id == "historical") & (c.table_id == "day") & c.variable_id.isin(TABLE)]


def stores(cat: pd.DataFrame, model: str, member: str) -> dict:
    """{var: zstore} for one historical run, the newest version of each (a native grid first); {} if one is missing."""
    c = cat[(cat.source_id == model) & (cat.member_id == member)]
    out = {}
    for v in TABLE:
        g = c[c.variable_id == v]
        if not len(g):
            return {}
        g = g.assign(native=g.grid_label.str.startswith("gn")).sort_values(["native", "version"])
        out[v] = g.zstore.iloc[-1]
    return out


def plan(cat: pd.DataFrame, members: int, era5_span: int) -> list[dict]:
    """ERA5 in spans of years, then every model's first `members` historical members (r1, r2, ...)."""
    out = [
        {"source": "era5", "years": f"{y}-{min(y + era5_span - 1, ERA5_YEARS[1])}"}
        for y in range(ERA5_YEARS[0], ERA5_YEARS[1] + 1, era5_span)
    ]
    for m in sorted(cat.source_id.unique()):
        mems = sorted(cat[cat.source_id == m].member_id.unique(), key=lambda x: [int(p) for p in re.findall(r"\d+", x)])
        got = [mem for mem in mems if stores(cat, m, mem)][:members]
        out += [{"source": "cmip6", "model": m, "member": mem} for mem in got]
    return out


def pieces(da, ix: np.ndarray) -> list[tuple[int, int]]:
    """Runs of the wanted days whose store steps are consecutive and inside one store chunk, cut to READ_BYTES, so
    each chunk is fetched once (unless one chunk is bigger than READ_BYTES)."""
    ct = int((da.encoding.get("chunks") or (365,))[0])
    step = max(1, min(ct, int(READ_BYTES // (4 * np.prod(da.shape[1:])))))
    brk = np.flatnonzero((np.diff(ix) != 1) | (np.diff(ix // ct) != 0)) + 1
    return [(s, min(s + step, b)) for a, b in zip(np.r_[0, brk], np.r_[brk, len(ix)]) for s in range(a, b, step)]


def run_cmip6(cat, model, member, out: Path, workers: int) -> dict:
    dest = out / f"{model}_{member}.npz"
    st = stores(cat, model, member)
    if not st:
        return {"run": dest.stem, "status": "missing day ua, va or psl"}
    t0 = time.time()
    das = {v: CE.open_var(z, v) for v, z in st.items()}
    if das["ua"].shape[1:] != das["va"].shape[1:]:
        return {"run": dest.stem, "status": "ua and va on different grids"}
    keys = {v: {(t.year, t.month, t.day): i for i, t in enumerate(da.time.values)} for v, da in das.items()}
    days = sorted(
        d for d in set.intersection(*(set(k) for k in keys.values()))
        if CMIP6_YEARS[0] <= d[0] <= CMIP6_YEARS[1] and d[1] in MONTHS
    )  # fmt: skip
    if len(days) < 30 * 200:
        return {"run": dest.stem, "status": f"only {len(days)} common days"}
    p = das["ua"].plev.values
    lev = [L for L in LEVELS if CE._plev_index(p, L) is not None]
    kl = [CE._plev_index(p, L) for L in lev]
    lat = das["ua"].lat.values
    P = Pressure(das["psl"].lat.values, das["psl"].lon.values, model)
    n = len(days)
    res = {
        "u": np.full((n, len(lev), len(LAT)), np.nan, "float32"),
        "uv": np.full((n, len(lev), len(LAT)), np.nan, "float32"),
        "slp": np.full((n, len(LAT)), np.nan, "float32"),
        "slp_p": np.full(n, np.nan, "float32"),
    }
    # A store can hold a day twice or skip one (CESM2-WACCM), so days are indexed, never sliced by date.
    ix = {v: np.array([keys[v][d] for d in days]) for v in das}

    def winds(a, b):
        u = CE._read(das["ua"], ix["ua"][a:b])[:, kl]
        v = CE._read(das["va"], ix["va"][a:b])[:, kl]
        return dict(zip(("u", "uv"), zonal(u, v, lat)))

    def pressure(a, b):
        return dict(zip(("slp", "slp_p"), P(CE._read(das["psl"], ix["psl"][a:b]) / 100.0)))

    for lead, job in (("ua", winds), ("psl", pressure)):

        def one(piece, job=job):
            for attempt in range(5):
                try:
                    return piece[0], job(*piece)
                except Exception as err_:  # noqa: BLE001  (a GCS hiccup: retry the piece)
                    err = err_
                    time.sleep(5 * (attempt + 1))
            raise err

        with ThreadPoolExecutor(workers) as ex:
            for k, (a, r) in enumerate(ex.map(one, pieces(das[lead], ix[lead]))):
                b = a + len(next(iter(r.values())))
                for key, x in r.items():
                    res[key][a:b] = x
                if k % 25 == 0:
                    print(f"    {dest.stem}: {lead} {b}/{n} days, {time.time() - t0:.0f} s", flush=True)
    cal = str(das["ua"].time.values[0].calendar)
    save(dest, np.array(days, "int16"), cal, lev, res)
    return {
        "run": dest.stem, "status": "ok", "days": n, "calendar": cal, "levels": lev,
        "first": "-".join(map(str, days[0])), "last": "-".join(map(str, days[-1])), "seconds": round(time.time() - t0),
    }  # fmt: skip


# ----------------------------------------------------------------------------------------------------------- ERA5 ---
def run_era5(y0: int, y1: int, out: Path) -> dict:
    import xarray as xr

    dest = out / f"ERA5_{y0}-{y1}.npz"
    t0 = time.time()
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    ds = ds.sel(latitude=ds.latitude[ds.latitude >= LAT[0] - 5.0])
    lev = [L for L in LEVELS if L in ds.level.values]
    lat, lon = ds.latitude.values.astype(float), ds.longitude.values
    if lat[0] > lat[-1]:
        raise ValueError("WeatherBench 2 latitudes are expected ascending")
    P = Pressure(lat, lon, "ERA5")
    keep = {k: [] for k in ("ymd", "u", "uv", "uv6", "slp", "slp_p")}
    for y in range(y0, y1 + 1):
        for m in MONTHS:
            t = ds.time.sel(time=f"{y}-{m:02d}")
            if t.size < 4:
                continue
            for attempt in range(5):
                try:
                    sub = ds.sel(time=t.values, level=lev)
                    u = sub.u_component_of_wind.transpose("time", "level", "latitude", "longitude").values
                    v = sub.v_component_of_wind.transpose("time", "level", "latitude", "longitude").values
                    p = sub.mean_sea_level_pressure.transpose("time", "latitude", "longitude").values / 100.0
                    break
                except Exception as err_:  # noqa: BLE001
                    err = err_
                    time.sleep(10 * (attempt + 1))
            else:
                raise err
            nd = t.size // 4  # whole days of four 6-hourly fields (the store ends at 2023-01-10 00Z)
            day = pd.DatetimeIndex(t.values[: 4 * nd : 4])
            if not (pd.DatetimeIndex(t.values[: 4 * nd]).hour.values.reshape(nd, 4) == (0, 6, 12, 18)).all():
                raise ValueError(f"{y}-{m:02d}: not 00/06/12/18Z steps")
            sh = (nd, 4, *u.shape[1:])
            ud, vd = u[: 4 * nd].reshape(sh).mean(1), v[: 4 * nd].reshape(sh).mean(1)
            zu, zuv = zonal(ud, vd, lat)
            _, zuv6 = zonal(u[: 4 * nd], v[: 4 * nd], lat)
            ps, pp = P(p[: 4 * nd].reshape((nd, 4, *p.shape[1:])).mean(1))
            keep["ymd"].append(np.stack([day.year, day.month, day.day], 1))
            keep["u"].append(zu)
            keep["uv"].append(zuv)
            keep["uv6"].append(zuv6.reshape((nd, 4, *zuv6.shape[1:])).mean(1))
            keep["slp"].append(ps)
            keep["slp_p"].append(pp)
        print(f"    {dest.stem}: {y} done, {time.time() - t0:.0f} s", flush=True)
    ymd = np.concatenate(keep.pop("ymd")).astype("int16")
    res = {k: np.concatenate(x).astype("float32") for k, x in keep.items()}
    for k in ("u", "uv", "uv6"):
        if not np.isfinite(res[k]).all():
            raise ValueError(f"ERA5 {k} has missing values")
    if not (950 < np.nanmean(res["slp"]) < 1050 and 5 < np.nanmax(res["u"][:, lev.index(250)].mean(0)) < 60):
        raise ValueError("ERA5 values out of range")
    save(dest, ymd, "proleptic_gregorian", lev, res)
    return {"run": dest.stem, "status": "ok", "days": len(ymd), "levels": lev, "seconds": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the jobs, as a JSON list (for the workflow's matrix)")
    p.add_argument("--catalog", required=True)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--models", default="", help="comma-separated source_ids (blank = every model with the data)")
    p.add_argument("--era5-span", type=int, default=4)
    c = sub.add_parser("cmip6", help="reduce one CMIP6 historical run")
    c.add_argument("--catalog", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--member", required=True)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--out", required=True)
    e = sub.add_parser("era5", help="reduce a span of ERA5 years")
    e.add_argument("--years", required=True, help="first-last, e.g. 1959-1962")
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        cat = catalog(a.catalog)
        if a.models.strip():
            cat = cat[cat.source_id.isin([m.strip() for m in a.models.split(",")])]
        print(json.dumps(plan(cat, a.members, a.era5_span)))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.model}_{a.member}" if a.cmd == "cmip6" else f"ERA5_{a.years}"
    try:
        if a.cmd == "cmip6":
            res = run_cmip6(catalog(a.catalog), a.model, a.member, out, a.workers)
        else:
            y0, y1 = (int(x) for x in a.years.split("-"))
            res = run_era5(y0, y1, out)
    except Exception as err:  # noqa: BLE001
        res = {"run": name, "status": f"failed: {type(err).__name__}: {str(err)[:300]}"}
    print(res, flush=True)
    (out / f"{res['run']}_log.json").write_text(json.dumps(res))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
