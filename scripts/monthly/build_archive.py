#!/usr/bin/env python3
"""Build or extend the ERA5 half-month archive of the month-ahead outlook (layout in mo_grid.py).

Sources: WeatherBench 2 ERA5 1.5° 6-hourly (1959 to 2022) and ARCO-ERA5 0.25° (ERA5 then ERA5T, 2023 onward), both
anonymous Google Cloud reads; ARCO is sampled at 00/06/12/18 UTC like the WB2 store, so both halves of the archive
are means of the same four synoptic hours. Both are box-averaged onto the same 3° cells (mo_grid.Coarsener).

    python scripts/monthly/build_archive.py --start 1959 --end 2022            # WB2, the static part
    python scripts/monthly/build_archive.py --source arco --start 2023          # ARCO tail up to the newest ERA5T
    python scripts/monthly/build_archive.py --tail                              # monthly run: refresh incomplete blocks

A block already complete (`n` equal to its expected 6-hourly count) is never re-read.
"""

from __future__ import annotations

import argparse
import calendar
import io
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mo_grid as G

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GRAV = 9.80665
HOURS = (0, 6, 12, 18)


def block_of(t: pd.Timestamp) -> tuple[int, int]:
    return t.month - 1, int(t.day >= G.SPLIT_DAY)


def block_days(y: int, m: int, h: int) -> tuple[date, date]:
    nd = calendar.monthrange(y, m)[1]
    return (date(y, m, 1), date(y, m, G.SPLIT_DAY - 1)) if h == 0 else (date(y, m, G.SPLIT_DAY), date(y, m, nd))


# ---------------------------------------------------------------- WB2 (1959-2022): whole years with dask
def wb2_year(y: int, only: list[str] | None = None) -> dict:
    import dask

    dask.config.set(scheduler="threads", num_workers=32)
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    sl = slice(f"{y}-01-01", f"{y}-12-31T18")
    lat = ds.latitude.values
    co = G.Coarsener(lat, ds.longitude.values)
    raw = {
        "z500": ds["geopotential"].sel(level=500, time=sl) / GRAV,
        "t2m": ds["2m_temperature"].sel(time=sl) - 273.15,
        "sst": ds["sea_surface_temperature"].sel(time=sl) - 273.15,
        "sic": ds["sea_ice_cover"].sel(time=sl),
        "u50": ds["u_component_of_wind"].sel(level=50, time=sl),
        "snow": ds["snow_depth"].sel(time=sl),
        "swvl": ds["volumetric_soil_water_layer_1"].sel(time=sl),
    }
    if only:
        raw = {k: v for k, v in raw.items() if k in only}
    times = pd.DatetimeIndex(next(iter(raw.values())).time.values)
    keys = np.array([b[0] * 2 + b[1] for b in map(block_of, times)])
    out = G.load_year(y, raw=True) if only else G.empty_year()
    for k, da in raw.items():
        t0 = time.time()
        a = da.transpose("time", "latitude", "longitude").values.astype(np.float32)
        blocks = np.full((24, *a.shape[1:]), np.nan, np.float32)
        for b in range(24):
            sel = keys == b
            if sel.any():
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    blocks[b] = np.nanmean(a[sel], axis=0) if k in ("sst", "sic") else a[sel].mean(0)
        if k == "u50":
            out["u50eq"] = G.band_mean(blocks, lat, -5, 5).reshape(12, 2)
            out["u50v"] = G.band_mean(blocks, lat, 57, 63).reshape(12, 2)
        else:
            grid = "sst" if k == "sst" else "nh"
            c = co(blocks, grid, min_frac=0.5 if k in ("sst", "sic") else 0.0)
            out[k] = c.reshape(12, 2, *c.shape[1:]).astype(np.float32)
        print(f"    {y} {k}: {time.time() - t0:.0f}s", flush=True)
    if not only:
        for b in range(24):
            out["n"][b // 2, b % 2] = int((keys == b).sum())
    return out


# ---------------------------------------------------------------- ARCO (2023 onward): one 6-hourly step at a time
RAW = "gcp-public-data-arco-era5/raw/date-variable-pressure_level"


def _open_raw(b: bytes) -> xr.Dataset:
    """ARCO raw daily files are netCDF4 (recent) or 64-bit netCDF3 (older)."""
    return xr.open_dataset(io.BytesIO(b), engine="h5netcdf" if b[:4] == b"\x89HDF" else "scipy")


class Arco:
    """ARCO-ERA5 reads, one day at a time. The two pressure-level fields come from the raw daily per-level files
    (~30-50 MB for 24 hours; the zarr chunk of one hour holds all 37 levels, ~80 MB), the surface fields from the
    zarr store (one ~2 MB chunk per hour)."""

    def __init__(self):
        import gcsfs

        self.fs = gcsfs.GCSFileSystem(token="anon")
        self.ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        lat, lon = self.ds.latitude.values, self.ds.longitude.values
        self.isst = np.where(lat >= -32.0)[0]
        self.inh = np.where(lat >= -2.0)[0]
        self.co_nh = G.Coarsener(lat[self.inh], lon)
        self.co_sst = G.Coarsener(lat[self.isst], lon)
        self.lat = lat
        self.ieq = np.where(np.abs(lat) <= 5.0)[0]
        self.iv = np.where((lat >= 57.0) & (lat <= 63.0))[0]
        stop = self.ds.attrs.get("valid_time_stop_era5t") or self.ds.attrs.get("valid_time_stop")
        self.last = pd.Timestamp(stop) + pd.Timedelta(hours=18)  # the attribute names the last complete day

    def _plev(self, d: date, var: str, lev: int) -> np.ndarray:
        """(4, lat, lon) at HOURS from the raw daily file, latitude 90..-90 like the zarr store."""
        with self.fs.open(f"{RAW}/{d:%Y/%m/%d}/{var}/{lev}.nc", "rb") as f:
            ds = _open_raw(f.read())
        da = ds[next(iter(ds.data_vars))].squeeze(drop=True)
        tdim = next(x for x in da.dims if "time" in x)
        hrs = pd.DatetimeIndex(da[tdim].values).hour
        da = da.isel({tdim: [list(hrs).index(h) for h in HOURS]}).transpose(tdim, "latitude", "longitude")
        if da.latitude.values[0] < da.latitude.values[-1]:
            da = da.isel(latitude=slice(None, None, -1))
        return da.values.astype(np.float32)

    def day(self, d: date, only: list[str] | None = None) -> dict | None:
        if only:
            return self.day_surface(d, only)
        for attempt in range(4):
            try:
                z = self._plev(d, "geopotential", 500)[:, self.inh] / GRAV
                u = self._plev(d, "u_component_of_wind", 50)
                ts = [pd.Timestamp(datetime(d.year, d.month, d.day, h)) for h in HOURS]
                sub = self.ds[["2m_temperature", "sea_surface_temperature", "sea_ice_cover"]].sel(time=ts)
                t2 = sub["2m_temperature"].values[:, self.inh] - 273.15
                ss = sub["sea_surface_temperature"].values[:, self.isst] - 273.15
                si = sub["sea_ice_cover"].values[:, self.inh]
                sn = self.ds["snow_depth"].sel(time=ts).values[:, self.inh]
                sw = self.ds["volumetric_soil_water_layer_1"].sel(time=ts).values[:, self.inh]
                break
            except Exception as e:  # noqa: BLE001  transient GCS errors; retried, then raised
                if attempt == 3:
                    raise
                print(f"    retry {d}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        if not np.isfinite(t2).any(axis=(1, 2)).all():
            return None  # NaN padding: the day is not fully published yet
        wq = np.cos(np.deg2rad(self.lat))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN land cells of the ocean fields
            out = {
                "z500": self.co_nh(z, "nh").sum(0),
                "t2m": self.co_nh(t2, "nh").sum(0),
                "sic": np.nanmean(self.co_nh(si, "nh", 0.5), 0) * len(HOURS),
                "sst": np.nanmean(self.co_sst(ss, "sst", 0.5), 0) * len(HOURS),
                "snow": self.co_nh(sn, "nh").sum(0),
                "swvl": self.co_nh(sw, "nh").sum(0),
            }
        out["u50eq"] = float((u[:, self.ieq].mean((0, 2)) * wq[self.ieq]).sum() / wq[self.ieq].sum())
        out["u50v"] = float((u[:, self.iv].mean((0, 2)) * wq[self.iv]).sum() / wq[self.iv].sum())
        return out


SURF = {"snow": "snow_depth", "swvl": "volumetric_soil_water_layer_1"}


def _day_surface(self: Arco, d: date, only: list[str]) -> dict | None:
    """Only the listed land-surface fields of one day (a field added to the archive later)."""
    ts = [pd.Timestamp(datetime(d.year, d.month, d.day, h)) for h in HOURS]
    for attempt in range(4):
        try:
            sub = self.ds[[SURF[k] for k in only]].sel(time=ts)
            vals = {k: sub[SURF[k]].values[:, self.inh] for k in only}
            break
        except Exception as e:  # noqa: BLE001  transient GCS errors; retried, then raised
            if attempt == 3:
                raise
            print(f"    retry {d}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    if not all(np.isfinite(v).any(axis=(1, 2)).all() for v in vals.values()):
        return None
    return {k: self.co_nh(v, "nh").sum(0) for k, v in vals.items()}


Arco.day_surface = _day_surface


def arco_blocks(
    arco: Arco, d: dict, y: int, todo: list[tuple[int, int]], workers: int, path=None, only: list[str] | None = None
) -> dict:
    """Recompute the listed (month index, half) blocks of year y from ARCO, using the complete published days up to
    the first that is not; the year file is saved after every block, so an interrupted run resumes."""
    for mi, h in todo:
        d0, d1 = block_days(y, mi + 1, h)
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        days = [x for x in days if pd.Timestamp(x) + pd.Timedelta(hours=18) <= arco.last]
        if not days:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda x: arco.day(x, only), days))
        if None in res:
            res = res[: res.index(None)]
        nd = len(res)
        if not nd:
            continue
        for k in only or G.FIELDS:  # each day holds a sum over HOURS
            with np.errstate(invalid="ignore"):
                d[k][mi, h] = np.sum([r[k] for r in res], 0) / (nd * len(HOURS))
        if not only:
            for k in G.INDICES:  # each day holds its daily mean
                d[k][mi, h] = np.mean([r[k] for r in res])
            d["n"][mi, h] = nd * len(HOURS)
        print(f"    {y}-{mi + 1:02d} block {h}: {nd} days in {time.time() - t0:.0f}s", flush=True)
        G.save_year(y, d, path)
    return d


def incomplete(d: dict, y: int, last: pd.Timestamp, only: list[str] | None = None) -> list[tuple[int, int]]:
    """Blocks to (re)compute: those short of their expected samples, or with `only`, those that have samples but
    no values yet for the listed fields."""
    todo = []
    for mi in range(12):
        for h in (0, 1):
            d0, _ = block_days(y, mi + 1, h)
            if pd.Timestamp(d0) > last:
                continue
            if only:
                if d["n"][mi, h] > 0 and any(np.isnan(d[k][mi, h]).all() for k in only):
                    todo.append((mi, h))
            elif d["n"][mi, h] < G.expected_samples(y, mi + 1, h) or any(np.isnan(d[k][mi, h]).all() for k in G.FIELDS):
                todo.append((mi, h))
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wb2", "arco"], default="wb2")
    ap.add_argument("--start", type=int, default=1959)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--tail", action="store_true", help="refresh the incomplete blocks of the last two years (ARCO)")
    ap.add_argument("--overlap", action="store_true", help="build 2022 from ARCO too (the source offset)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", help="comma-separated fields to add to existing year files (e.g. snow,swvl)")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None

    if a.overlap:
        arco = Arco()
        d = G.load_year(2022, G.OVERLAP)
        arco_blocks(arco, d, 2022, incomplete(d, 2022, arco.last, only), a.workers, G.OVERLAP, only)
        return 0

    if a.tail or a.source == "arco":
        arco = Arco()
        print(f"ARCO ERA5T through {arco.last:%Y-%m-%d}", flush=True)
        y1 = a.end or arco.last.year
        y0 = max(arco.last.year - 1, 2023) if a.tail else a.start
        for y in range(y0, y1 + 1):
            d = G.load_year(y, raw=True)
            todo = incomplete(d, y, arco.last, only)
            if not todo:
                print(f"  {y}: complete", flush=True)
                continue
            print(f"  {y}: {len(todo)} blocks", flush=True)
            arco_blocks(arco, d, y, todo, a.workers, only=only)
        return 0

    for y in range(a.start, (a.end or 2022) + 1):
        d = G.load_year(y, raw=True)
        if (d["n"] > 0).all() and not incomplete(d, y, pd.Timestamp("2022-12-31T18"), only):
            print(f"  {y}: complete", flush=True)
            continue
        t0 = time.time()
        G.save_year(y, wb2_year(y, only))
        print(f"  {y}: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
