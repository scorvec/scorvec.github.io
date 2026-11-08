#!/usr/bin/env python3
"""Daily zonal-mean wind and eddy momentum flux by latitude, ERA5 and CMIP6 reduced alike, for testing whether the
models' tropospheric eddy feedback is weak and whether that weakens the surface response to sudden stratospheric
warmings (eddy_feedback.py). Runs in Actions (.github/workflows/eddy-feedback.yml), one CMIP6 run or a span of ERA5
winters per job, reading the public Pangeo CMIP6 and WeatherBench 2 Google Cloud mirrors anonymously; nothing raw is
kept.

From daily-mean u and v (CMIP6's day table; ERA5's four 6-hourly fields averaged first, so both see the same eddies),
October to May:
  u      zonal-mean u, m/s, on 20-87.5N every 2.5 deg, at the LEVELS the source has (CMIP6 plev8: 10, 50, 100, 250,
         500, 700, 850 hPa; ERA5 from WeatherBench 2: 50-850 hPa, no 10 hPa)
  uv     the zonal mean of u*v* (departures from the zonal mean), m2/s2, the same way
  slp    zonal-mean sea-level pressure, hPa, the same latitudes
  slp_p  sea-level pressure projected on the CMIP6 post-SSW pattern (ssw_imprint), hPa
  nao    Stephenson et al. (2006) NAO: sea-level pressure over 20-55N less 55-90N, both 90W-60E, hPa
  g_na   the barotropic energy generation rate at 250 hPa over the North Atlantic (Atlantic below), m2 s-3
  uv6    ERA5 only: u*v* from each 6-hourly field, then averaged (what daily means miss)
Each with year, month and day of the source's own calendar, which is stored with it.

    python eddy_extract.py plan --catalog pangeo-cmip6.csv --members 3 --era5-span 4
    python eddy_extract.py cmip6 --catalog pangeo-cmip6.csv --model CanESM5 --member r1i1p1f1 --out out/
    python eddy_extract.py era5 --winters 1958-1961 --out out/
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
ERA5_WINTERS = (1958, 2022)  # the store runs from 1959-01-01 to 2023-01-10
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
READ_BYTES = 1.2e9  # the most of one variable read at once
NA_BOX = (30.0, 45.0, 300.0, 335.0)  # 30-45N, 60-25W: where the eddies feed the large-scale flow (Saffin et al. 2024)
NA_PAD = 5.0  # read around the box for the large-scale flow's derivatives
EARTH_RADIUS = 6.371e6


def winter(year, month):
    """The winter a day belongs to, named by the year of its October."""
    return np.where(np.asarray(month) >= 10, year, np.asarray(year) - 1)


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
    """Zonal mean, post-SSW pattern projection and NAO of daily sea-level pressure (hPa) on one grid."""

    def __init__(self, lat, lon, model):
        self.lat = np.asarray(lat, float)
        self.proj = CE.Fields(lat, lon, model).proj[0]
        self.boxes = CE.SI.Boxes(lat, lon)
        cl, co = CE.SI.LAT_EDGES[:-1] + 1.25, CE.SI.LON_EDGES[:-1] + 1.25
        w = np.cos(np.deg2rad(cl))[:, None] * ((co >= 270.0) | (co <= 60.0))[None]
        south = (cl < 55.0)[:, None]
        self.nao = np.where(south, w / (w * south).sum(), -w / (w * ~south).sum())

    def __call__(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.boxes(p)
        zm = to_lat(np.nanmean(p, -1), self.lat).astype("float32")
        return zm, (b * self.proj).sum((-2, -1)), (b * self.nao).sum((-2, -1))


def lanczos(fc: float, n: int = 31) -> np.ndarray:
    """Lanczos low-pass weights (Duchon 1979) for a cut-off of fc cycles a day: 2n - 1 of them (61 for n = 31)."""
    k = np.arange(1 - n, n)
    return 2 * fc * np.sinc(2 * fc * k) * np.sinc(k / n)


class Atlantic:
    """250 hPa winds around NA_BOX on one native grid, and the barotropic energy generation rate there, G = E . D
    (Mak & Cai 1989, in Fukutomi & Yasunari's 2002 spherical form, as Saffin et al. 2024 use it): the 2-6-day eddies'
    E = cos(lat) (0.5 (v'2 - u'2), -u'v') against the deformation D of the > 10-day flow, both from daily means with
    61-weight Lanczos filters, averaged over the box. Positive = energy from the large-scale flow to the eddies."""

    def __init__(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float) % 360.0
        a, b, c, d = NA_BOX
        self.j = np.flatnonzero((lat >= a - NA_PAD) & (lat <= b + NA_PAD))
        self.i = np.flatnonzero((lon >= c - NA_PAD) & (lon <= d + NA_PAD))
        if len(self.j) < 4 or len(self.i) < 4 or (np.diff(self.i) != 1).any():
            raise ValueError("the grid does not cover the North Atlantic box")
        self.lat, self.lon = lat[self.j], lon[self.i]
        inside = ((self.lat >= a) & (self.lat <= b))[:, None] & ((self.lon >= c) & (self.lon <= d))[None]
        self.w = np.where(inside, np.cos(np.deg2rad(self.lat))[:, None], 0.0)
        self.w /= self.w.sum()

    def cut(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x[..., self.j[:, None], self.i[None]], "float32")

    def g(self, u: np.ndarray, v: np.ndarray, winters: np.ndarray) -> np.ndarray:
        """Daily box-mean G from (day, lat, lon) winds, each winter filtered on its own (NaN 30 days from its ends)."""
        from scipy.ndimage import correlate1d

        band, slow = lanczos(1 / 2) - lanczos(1 / 6), lanczos(1 / 10)
        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        cos = np.cos(phi)[:, None]
        out = np.full(len(u), np.nan)
        for s in np.unique(winters):
            k = np.flatnonzero(winters == s)
            if len(k) <= len(slow):
                continue
            uu, vv = u[k].astype("float64"), v[k].astype("float64")
            ue, ve = correlate1d(uu, band, axis=0), correlate1d(vv, band, axis=0)
            uu, vv = correlate1d(uu, slow, axis=0), correlate1d(vv, slow, axis=0)
            d1 = (np.gradient(uu, lam, axis=-1) - np.gradient(vv * cos, phi, axis=-2)) / (EARTH_RADIUS * cos)
            d2 = (np.gradient(vv, lam, axis=-1) + np.gradient(uu * cos, phi, axis=-2)) / (EARTH_RADIUS * cos)
            g = (cos * (0.5 * (ve**2 - ue**2) * d1 - ue * ve * d2) * self.w).sum((-2, -1))
            m = len(slow) // 2
            g[:m] = g[-m:] = np.nan
            out[k] = g
        return out.astype("float32")


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
    """ERA5 in spans of winters, then every model's first `members` historical members (r1, r2, ...)."""
    w0, w1 = ERA5_WINTERS
    out = [{"source": "era5", "winters": f"{w}-{min(w + era5_span - 1, w1)}"} for w in range(w0, w1 + 1, era5_span)]
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
    if 250 not in lev:
        return {"run": dest.stem, "status": "no 250 hPa winds"}
    kl, k250 = [CE._plev_index(p, L) for L in lev], CE._plev_index(p, 250)
    lat = das["ua"].lat.values
    P = Pressure(das["psl"].lat.values, das["psl"].lon.values, model)
    NA = Atlantic(lat, das["ua"].lon.values)
    n, nl, nj = len(days), len(lev), len(LAT)
    res = {k: np.full((n, nl, nj), np.nan, "float32") for k in ("u", "uv")}
    res |= {"slp": np.full((n, nj), np.nan, "float32")} | {k: np.full(n, np.nan, "float32") for k in ("slp_p", "nao")}
    res |= {k: np.full((n, len(NA.j), len(NA.i)), np.nan, "float32") for k in ("na_u", "na_v")}
    # A store can hold a day twice or skip one (CESM2-WACCM), so days are indexed, never sliced by date.
    ix = {v: np.array([keys[v][d] for d in days]) for v in das}

    def winds(a, b):
        u, v = CE._read(das["ua"], ix["ua"][a:b]), CE._read(das["va"], ix["va"][a:b])
        zu, zuv = zonal(u[:, kl], v[:, kl], lat)
        return {"u": zu, "uv": zuv, "na_u": NA.cut(u[:, k250]), "na_v": NA.cut(v[:, k250])}

    def pressure(a, b):
        return dict(zip(("slp", "slp_p", "nao"), P(CE._read(das["psl"], ix["psl"][a:b]) / 100.0)))

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
    ymd = np.array(days, "int16")
    res["g_na"] = NA.g(res.pop("na_u"), res.pop("na_v"), winter(ymd[:, 0], ymd[:, 1]))
    cal = str(das["ua"].time.values[0].calendar)
    save(dest, ymd, cal, lev, res)
    return {
        "run": dest.stem, "status": "ok", "days": n, "calendar": cal, "levels": lev,
        "first": "-".join(map(str, days[0])), "last": "-".join(map(str, days[-1])), "seconds": round(time.time() - t0),
    }  # fmt: skip


# ----------------------------------------------------------------------------------------------------------- ERA5 ---
def run_era5(w0: int, w1: int, out: Path) -> dict:
    """Winters w0 to w1, each October to May."""
    import xarray as xr

    dest = out / f"ERA5_{w0}-{w1}.npz"
    t0 = time.time()
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    ds = ds.sel(latitude=ds.latitude[ds.latitude >= LAT[0] - 5.0])
    lev = [L for L in LEVELS if L in ds.level.values]
    lat, lon = ds.latitude.values.astype(float), ds.longitude.values
    if lat[0] > lat[-1]:
        raise ValueError("WeatherBench 2 latitudes are expected ascending")
    P, NA = Pressure(lat, lon, "ERA5"), Atlantic(lat, lon)
    tix = pd.DatetimeIndex(ds.time.values)
    keep = {k: [] for k in ("ymd", "u", "uv", "uv6", "slp", "slp_p", "nao", "na_u", "na_v")}
    for w in range(w0, w1 + 1):
        for m in MONTHS:
            sel = np.flatnonzero((tix.year == (w if m >= 10 else w + 1)) & (tix.month == m))
            nd = len(sel) // 4  # whole days of four 6-hourly fields (the store ends at 2023-01-10 00Z)
            if nd == 0:
                continue
            sel = sel[: 4 * nd]
            if not (tix[sel].hour.values.reshape(nd, 4) == (0, 6, 12, 18)).all():
                raise ValueError(f"{tix[sel[0]]:%Y-%m}: not 00/06/12/18Z steps")
            for attempt in range(5):
                try:
                    sub = ds.isel(time=sel).sel(level=lev)
                    u = sub.u_component_of_wind.transpose("time", "level", "latitude", "longitude").values
                    v = sub.v_component_of_wind.transpose("time", "level", "latitude", "longitude").values
                    p = sub.mean_sea_level_pressure.transpose("time", "latitude", "longitude").values / 100.0
                    break
                except Exception as err_:  # noqa: BLE001  (a GCS hiccup: retry the month)
                    err = err_
                    time.sleep(10 * (attempt + 1))
            else:
                raise err
            day = tix[sel[::4]]
            ud, vd = u.reshape(nd, 4, *u.shape[1:]).mean(1), v.reshape(nd, 4, *v.shape[1:]).mean(1)
            zu, zuv = zonal(ud, vd, lat)
            zuv6 = zonal(u, v, lat)[1]
            ps, pp, nao = P(p.reshape(nd, 4, *p.shape[1:]).mean(1))
            keep["ymd"].append(np.stack([day.year, day.month, day.day], 1))
            for k, x in (("u", zu), ("uv", zuv), ("slp", ps), ("slp_p", pp), ("nao", nao)):
                keep[k].append(x)
            keep["uv6"].append(zuv6.reshape(nd, 4, *zuv6.shape[1:]).mean(1))
            keep["na_u"].append(NA.cut(ud[:, lev.index(250)]))
            keep["na_v"].append(NA.cut(vd[:, lev.index(250)]))
        print(f"    {dest.stem}: winter {w} done, {time.time() - t0:.0f} s", flush=True)
    ymd = np.concatenate(keep.pop("ymd")).astype("int16")
    res = {k: np.concatenate(x).astype("float32") for k, x in keep.items()}
    res["g_na"] = NA.g(res.pop("na_u"), res.pop("na_v"), winter(ymd[:, 0], ymd[:, 1]))
    for k in ("u", "uv", "uv6", "slp", "nao"):
        if not np.isfinite(res[k]).all():
            raise ValueError(f"ERA5 {k} has missing values")
    if not (950 < res["slp"].mean() < 1050 and 5 < res["u"][:, lev.index(250)].mean(0).max() < 60):
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
    p.add_argument("--era5-span", type=int, default=4, help="ERA5 winters per job")
    c = sub.add_parser("cmip6", help="reduce one CMIP6 historical run")
    c.add_argument("--catalog", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--member", required=True)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--out", required=True)
    e = sub.add_parser("era5", help="reduce a span of ERA5 winters")
    e.add_argument("--winters", required=True, help="first-last, by the year of their October, e.g. 1958-1961")
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
    name = f"{a.model}_{a.member}" if a.cmd == "cmip6" else f"ERA5_{a.winters}"
    try:
        if a.cmd == "cmip6":
            res = run_cmip6(catalog(a.catalog), a.model, a.member, out, a.workers)
        else:
            res = run_era5(*(int(x) for x in a.winters.split("-")), out)
    except Exception as err:  # noqa: BLE001
        res = {"run": name, "status": f"failed: {type(err).__name__}: {str(err)[:300]}"}
    print(res, flush=True)
    (out / f"{res['run']}_log.json").write_text(json.dumps(res))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
