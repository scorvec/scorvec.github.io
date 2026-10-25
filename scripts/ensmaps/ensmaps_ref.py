#!/usr/bin/env python3
"""Daily normals of the global ensemble maps (ensmaps.py), 1991-2020, as a smooth annual curve per grid point:
2 m temperature on the IFS 0.1-deg grid, 500 hPa height and sea-level pressure on a 1.5-deg grid.

2 m temperature: ERA5-Land over land, ERA5 over the sea, on the 3600 x 1801 points of the IFS-ENS 0.1-deg open data (90N
first, columns from 180W). ERA5-Land is ECMWF's land reanalysis on the ~9 km grid of the IFS (TCo1279) with its
orography, so its 0.1-deg 2 m temperature matches the IFS-ENS 0.1-deg fields point for point without a height
correction (the coarser models' normals are moved to their own orography in ensmaps.py). It has no value over the sea;
there ERA5 (0.25 deg) is interpolated bilinearly from its SEA nodes, each weighted by its sea fraction (ERA5's land-sea
mask), so a coastal mountain node does not chill the water beside it; a point with no sea node around it takes the
plain bilinear value. Both are read from the CDS as monthly means BY HOUR OF DAY at 00, 06, 12 and 18 UTC and averaged
over the four hours: the synoptic times a forecast day is the mean of. Actions only (ensmaps-ref.yml): one request per
reanalysis and decade, one at a time, each reduced at once to its per-month means.

500 hPa height and sea-level pressure: the local ERA5 store's global 1.5-deg layer ($ERA5_STORE/wb2_1p5_daily_global,
WeatherBench 2's conservative 1.5-deg ERA5, daily means of the four synoptic hours; z500 holds metres and slp hPa
whatever the files' attributes say): the 12 monthly means of 1991-2020 from the daily fields. Smooth fields - ensmaps.py
interpolates their curves bilinearly to each model's grid. Built offline from the store, nothing downloaded, and kept
beside this script (reference/ensmaps_upper_normals.npz).

Their spread, for the standardized anomalies (reference/ensmaps_upper_sd.npz): the same store's daily anomalies from
that curve, a linear trend removed per cell over 1991-2020, then the n-day running mean for each averaging period the
maps use (SPREAD_DAYS: a day, and 3, 4 and 5-day means - a 5-day mean varies far less than a day), squared, and the
variance fitted per cell as a mean + three harmonics of the day of a 366-day year (29 Feb its own slot): the variance
of an n-day mean anomaly through the year, whose square root divides the anomaly (Spread.sd).

Interim 2 m temperature (INTERIM_OUT, until the CDS build lands; Actions only, ensmaps-ref.yml's interim job): ERA5 at
1.5 deg from WeatherBench 2's public bucket (the store's source), the 12 monthly normals of the four synoptic hours,
fitted as below and interpolated bilinearly to the 0.1-deg grid, moved by 6.5 K/km from ERA5's own 1.5-deg surface
height to the IFS's 0.1-deg height (ifs_height10); inside the North America maps' box (77-9N, 180-26W, release
natemp-ref-v1, built as the final file will be: ERA5-Land and ERA5's sea) their normals as they are, blended over
NA_TAPER inside the box's edge. Same layout as the final file; ensmaps.yml takes it only while the final one is absent.

The daily curve: mean + three annual harmonics on the common-year calendar, fitted by least squares so that the curve's
MONTH AVERAGES - its exact integrals over each month - match the 12 monthly normals (scripts/lib/harmonic_clim: one
12 x 7 design matrix serves every point). The 2 m file keeps the mean in float32 and the harmonics in float16 (the
largest annual amplitude, ~30 K, to 0.02 K). Each fit prints its month residuals and a few places' curves.

    python scripts/ensmaps/ensmaps_ref.py fetch --out-dir ref        # Actions only: CDS, one request at a time
    python scripts/ensmaps/ensmaps_ref.py fit --in-dir ref --out ref/ensmaps_t2m_normals.npz   # exit 3: units missing
    python scripts/ensmaps/ensmaps_ref.py upper                      # from $ERA5_STORE -> reference/
    python scripts/ensmaps/ensmaps_ref.py spread                     # from $ERA5_STORE -> reference/
    python scripts/ensmaps/ensmaps_ref.py interim --natemp natemp_normals.npz --out ref/ensmaps_t2m_normals_interim.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[0] / "lib"), str(HERE.parents[0] / "sst")]
import harmonic_clim as HC

# the grids, 90N first, columns from 180W
LAT10 = np.round(np.arange(900, -901, -1) / 10.0, 1)  # the IFS 0.1-deg open data (and ERA5-Land): 1801 rows
LON10 = np.round(np.arange(-1800, 1800) / 10.0, 1)  # 3600 columns
LAT25 = np.round(np.arange(9000, -9001, -25) / 100.0, 2)  # ERA5 and the 0.25-deg open data: 721
LON25 = np.round(np.arange(-18000, 18000, 25) / 100.0, 2)  # 1440
LATU = np.round(np.arange(900, -901, -15) / 10.0, 1)  # the store's 1.5-deg grid: 121
LONU = np.round(np.arange(-1800, 1800, 15) / 10.0, 1)  # 240
YEARS = (1991, 2020)
HOURS = ("00:00", "06:00", "12:00", "18:00")
DECADES = ((1991, 2000), (2001, 2010), (2011, 2020))  # one CDS request each: 480 fields
T2M_OUT = "ensmaps_t2m_normals.npz"  # release ensmaps-ref-v1
UPPER_OUT = HERE / "reference" / "ensmaps_upper_normals.npz"
SPREAD_OUT = HERE / "reference" / "ensmaps_upper_sd.npz"
SPREAD_DAYS = (1, 3, 4, 5)  # the averaging periods standardized: a day, and the 3, 4 and 5-day means
INTERIM_OUT = "ensmaps_t2m_normals_interim.npz"  # release ensmaps-ref-v1, read only while T2M_OUT is not there
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
NA_TAPER = 3.0  # deg inside the North America normals' box over which the interim blends into them
GAMMA, RD = 6.5e-3, 287.05  # K/m, J/(kg K): the standard column of ifs_height10
T2M_META = {
    "source": "ERA5-Land monthly means by hour of day over land; ERA5 (0.25 deg, sea nodes) over the sea",
    "label": "ERA5-Land 1991–2020 over land, ERA5 over the sea, as a smooth daily curve",
}
INTERIM_META = {
    "source": "ERA5 1.5 deg (WeatherBench 2, conservative), monthly means of the four synoptic hours, interpolated "
    "bilinearly and moved by 6.5 K/km from ERA5's 1.5-deg surface height to the IFS's 0.1-deg one; inside 77-9N, "
    "180-26W the North America maps' normals (ERA5-Land 0.1 deg, ERA5 0.25 deg over the sea), blended over 3 deg "
    "inside that box's edge",
    "label": "interim normal: ERA5 1.5°, height-adjusted; ERA5-Land 0.1° to follow",
    "interim": True,
    "regions": {
        "na": "North America (9–77°N, 180–26°W): ERA5-Land 0.1° over land and ERA5 0.25° over the sea, the North "
        "America maps' normals",
        "rest": "elsewhere: interim normal: ERA5 1.5°, height-adjusted; ERA5-Land 0.1° to follow",
    },
}
UPPER_META = {
    "source": "ERA5 1.5 deg (WeatherBench 2, conservative), daily means of the four synoptic hours",
    "label": "ERA5 1991–2020 at 1.5°, as a smooth daily curve",
}
# the store's variable, its unit and the factor to it from what the store holds (z500 metres, slp hPa)
UPPER = {"z500": ("z500", "m", 1.0), "msl": ("slp", "hPa", 1.0)}
PLACES = {  # the fits' printed check
    "Chicago": (41.9, -87.6),
    "Yakutsk": (62.0, 129.7),
    "Delhi": (28.6, 77.2),
    "Sydney": (-33.9, 151.2),
    "Vostok": (-78.5, 106.8),
    "North Atlantic (sea)": (45.0, -30.0),
}


def cell(lat: float, lon: float, la: np.ndarray, lo: np.ndarray) -> tuple[int, int]:
    """(row, column) of the grid point nearest to lat, lon."""
    return int(np.abs(la - lat).argmin()), int(np.abs((lo - lon + 180) % 360 - 180).argmin())


def roll_to(v: np.ndarray, lon0: float, inc: float, lon: np.ndarray) -> np.ndarray:
    """A global field whose first column is at lon0 (deg E, any convention) -> its columns at `lon`."""
    cols = np.rint(((lon - lon0) % 360) / inc).astype(int) % v.shape[-1]
    return v[..., cols]


class Normals:
    """A normal curve on a regular grid: coef (1 + 2 NHARM, lat, lon), the field's unit and the build's metadata."""

    def __init__(self, coef: np.ndarray, lat: np.ndarray, lon: np.ndarray, meta: dict):
        self.coef, self.lat, self.lon, self.meta = coef.astype(np.float32), lat, lon, meta

    @classmethod
    def t2m(cls, path: Path) -> Normals:
        """The 2 m temperature normals (K) of the release file."""
        with np.load(path) as z:
            coef = np.concatenate([z["mean"][None].astype(np.float32), z["harm"].astype(np.float32)])
            if not (np.allclose(z["lat"], LAT10) and np.allclose(z["lon"], LON10)):
                raise SystemExit(f"{path}: the normals are not on the 0.1-deg grid")
            return cls(coef, LAT10, LON10, json.loads(str(z["meta"])))

    @classmethod
    def upper(cls, field: str, path: Path = UPPER_OUT) -> Normals:
        """The 500 hPa height (m) or sea-level pressure (hPa) normals."""
        with np.load(path) as z:
            return cls(z[field], z["lat"], z["lon"], json.loads(str(z["meta"])))

    def at(self, times) -> np.ndarray:
        """(len(times), lat, lon) float32: the curve at those times."""
        return HC.evaluate(self.coef, HC.times_of(times)).astype(np.float32)


def doy366(times) -> np.ndarray:
    """Day of a 366-day year, 1..366: 29 Feb is day 60 and 1 Mar day 61 in every year."""
    t = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(times)))
    d = t.dayofyear.to_numpy()
    return np.where(~t.is_leap_year & (d >= 60), d + 1, d)


def basis366(d: np.ndarray, nh: int = HC.NHARM) -> np.ndarray:
    """(len(d), 1 + 2 nh): the mean and nh harmonics of the 366-day year."""
    w = 2 * np.pi * np.arange(1, nh + 1) / 366.0
    x = np.asarray(d, float)[:, None] - 1.0
    return np.hstack([np.ones((x.size, 1)), np.cos(x * w), np.sin(x * w)])


class Spread:
    """The standard deviation of n-day mean anomalies through the year (SPREAD_OUT), on the store's grid or a
    model's: coef[n] (1 + 2 NHARM, lat, lon) fit the VARIANCE on 366 slots."""

    def __init__(self, coef: dict, lat: np.ndarray, lon: np.ndarray):
        self.coef, self.lat, self.lon = coef, lat, lon

    @classmethod
    def upper(cls, field: str, path: Path = SPREAD_OUT) -> Spread:
        with np.load(path) as z:
            coef = {n: z[f"{field}_{n}"].astype(np.float32) for n in SPREAD_DAYS}
            return cls(coef, z["lat"], z["lon"])

    def sd(self, days, n: int) -> np.ndarray:
        """(lat, lon): the sd of an n-day mean over `days` (its variance averaged over the days); a period the table
        does not hold takes the nearest it does."""
        n = min(self.coef, key=lambda d: (abs(d - n), d))
        v = np.tensordot(basis366(doy366(days)), self.coef[n], axes=(1, 0)).mean(0)
        return np.sqrt(np.clip(v, 1e-6, None)).astype(np.float32)


# ── 2 m temperature: fetch (GitHub Actions only) ──────────────────────────────────────────────────────────────────────
def units() -> list[tuple[str, str, dict]]:
    """(name, CDS dataset, request) of every request, in fetch order: global, no area (the CDS writes 0..360 E)."""
    common = {"variable": ["2m_temperature"], "month": [f"{m:02d}" for m in range(1, 13)], "time": list(HOURS)}
    common |= {"data_format": "grib", "download_format": "unarchived"}
    out = []
    for src, dataset in (
        ("land", "reanalysis-era5-land-monthly-means"),
        ("era5", "reanalysis-era5-single-levels-monthly-means"),
    ):
        for y0, y1 in DECADES:
            req = {"product_type": ["monthly_averaged_reanalysis_by_hour_of_day"], **common}
            out.append((f"{src}_{y0}", dataset, req | {"year": [str(y) for y in range(y0, y1 + 1)]}))
    lsm = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["land_sea_mask"], "year": ["2020"]}
    lsm |= {"month": ["01"], "time": ["00:00"], "data_format": "grib", "download_format": "unarchived"}
    out.append(("era5_lsm", "reanalysis-era5-single-levels-monthly-means", lsm))
    return out


def grid_of(src: str):
    return (LAT10, LON10) if src == "land" else (LAT25, LON25)


def read_field(gid, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """One global GRIB message on the regular grid (lat, lon), NaN where the message has no value; its columns rolled
    to start at 180W. Refuses any other grid."""
    import eccodes

    g = {k: eccodes.codes_get(gid, k) for k in ("Ni", "Nj", "jScansPositively")}
    g |= {
        k: eccodes.codes_get_double(gid, k)
        for k in (
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "iDirectionIncrementInDegrees",
            "jDirectionIncrementInDegrees",
        )
    }
    inc, step = g["iDirectionIncrementInDegrees"], g["jDirectionIncrementInDegrees"]
    la = g["latitudeOfFirstGridPointInDegrees"] + (1 if g["jScansPositively"] else -1) * step * np.arange(g["Nj"])
    v = eccodes.codes_get_values(gid).reshape(g["Nj"], g["Ni"])
    if eccodes.codes_get(gid, "bitmapPresent"):
        v = np.where(v == eccodes.codes_get_double(gid, "missingValue"), np.nan, v)
    if g["jScansPositively"]:
        la, v = la[::-1], v[::-1]
    if (
        la.size != lat.size
        or g["Ni"] != lon.size
        or not np.allclose(la, lat, atol=1e-4)
        or abs(g["Ni"] * inc - 360) > 1e-3
    ):
        raise SystemExit(f"GRIB grid {g} is not the global {lat.size} x {lon.size} grid")
    return roll_to(v, g["longitudeOfFirstGridPointInDegrees"], inc, lon)


def reduce_unit(name: str, path: Path, out: Path) -> None:
    """A unit's GRIB -> its per-month mean over years and hours (12, lat, lon) and the number of fields behind each
    value; checks that every month of every year holds four fields of distinct times (the four hours asked for) and
    prints the first month's times and global means."""
    import eccodes

    src = name.split("_")[0]
    lat, lon = grid_of(src)
    s = np.zeros((12, lat.size, lon.size))
    n = np.zeros((12, lat.size, lon.size), np.int16)
    times: dict = {}  # YYYYMM -> [(time keys, mean)]
    with open(path, "rb") as fh:
        while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                date = eccodes.codes_get(gid, "dataDate")
                when = tuple(eccodes.codes_get_string(gid, k) for k in ("dataTime", "validityTime", "stepRange"))
                f = read_field(gid, lat, lon)
            finally:
                eccodes.codes_release(gid)
            m = date // 100 % 100 - 1
            times.setdefault(date // 100, []).append((when, float(np.nanmean(f))))
            ok = np.isfinite(f)
            s[m][ok] += f[ok]
            n[m] += ok
    first = min(times)
    print(f"  {name} {first}: (dataTime, validityTime, stepRange) and mean: {times[first]}", flush=True)
    if name != "era5_lsm":
        y0, y1 = next(d for d in DECADES if name.endswith(str(d[0])))
        want = {y * 100 + m for y in range(y0, y1 + 1) for m in range(1, 13)}
        rows = {k: times.get(k, []) for k in want}
        bad = sorted(k for k, r in rows.items() if len(r) != len(HOURS) or len({w for w, _ in r}) != len(HOURS))
        extra = sorted(set(times) - want)
        if bad or extra:
            raise SystemExit(
                f"{name}: months without {len(HOURS)} distinct times: {bad[:6]}; out of range: {extra[:6]}"
            )
    with np.errstate(invalid="ignore"):
        np.savez_compressed(out, mean=(s / n).astype(np.float32), n=n, lat=lat, lon=lon)
    print(f"  {name}: {sum(map(len, times.values()))} fields -> {out.name}", flush=True)


def fetch(out_dir: Path) -> int:
    """Each unit not already reduced in out_dir: one CDS request (cds_seq: one at a time on the shared key), reduced at
    once and its GRIB deleted. Returns the number of units still missing."""
    from cds_seq import retrieve

    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("the CDS is read in GitHub Actions only (ensmaps-ref.yml)")
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name, dataset, req in units():
        npz = out_dir / f"ensmaps_{name}.npz"
        if npz.exists():
            print(f"  {name}: held", flush=True)
            continue
        grib = out_dir / f"{name}.grib"
        if not retrieve(dataset, req, grib, name):
            missing += 1
            continue
        reduce_unit(name, grib, npz)
        grib.unlink()
    return missing


# ── 2 m temperature: fit ──────────────────────────────────────────────────────────────────────────────────────────────
def merged(in_dir: Path, src: str):
    """The 1991-2020 monthly normals (12, lat, lon) of one reanalysis from its three decades, and the field counts."""
    s = n = 0
    for y0, _ in DECADES:
        with np.load(in_dir / f"ensmaps_{src}_{y0}.npz") as z:
            m, k = z["mean"].astype(np.float64), z["n"].astype(np.int64)
        s, n = s + np.where(k > 0, m * k, 0.0), n + k
    with np.errstate(invalid="ignore"):
        return s / n, n


def sea_interp(t25: np.ndarray, sea: np.ndarray, todo: np.ndarray) -> np.ndarray:
    """(12, LAT25, LON25) ERA5 monthly normals -> (12, LAT10, LON10) at the points `todo` (LAT10, LON10) (NaN
    elsewhere): bilinear from the sea nodes, weighted by sea fraction; plain bilinear where no sea node is near.
    Periodic in longitude."""
    from scipy.ndimage import map_coordinates

    jj, ii = np.nonzero(todo)
    rows = (LAT25[0] - LAT10[jj]) / 0.25  # fractional node indices, 90N first
    cols = (LON10[ii] - LON25[0]) / 0.25

    def interp(a):  # one column repeated past 180E so the last cell wraps to the first
        return map_coordinates(np.concatenate([a, a[:, :1]], axis=1), [rows, cols], order=1, mode="nearest")

    w = interp(sea)
    out = np.full((12, LAT10.size, LON10.size), np.nan)
    for m in range(12):
        plain, tw = interp(t25[m]), interp(t25[m] * sea)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m, jj, ii] = np.where(w > 1e-3, tw / w, plain)
    return out


def fit(in_dir: Path, out: Path) -> None:
    land_t, land_n = merged(in_dir, "land")
    era_t, era_n = merged(in_dir, "era5")
    full = 10 * len(DECADES) * len(HOURS)  # fields per month behind a complete normal
    if not (era_n == full).all():
        raise SystemExit(f"ERA5: {int((era_n != full).sum())} values without all {full} fields")
    land = (land_n == full).all(0)
    part = ((land_n > 0) & (land_n < full)).any(0)
    print(f"  ERA5-Land: {land.mean():.1%} of the grid is land with all {full} fields", flush=True)
    print(f"  {int(part.sum())} points with only some ERA5-Land fields take ERA5", flush=True)
    del land_n
    with np.load(in_dir / "ensmaps_era5_lsm.npz") as z:
        sea = 1.0 - np.clip(z["mean"][0].astype(np.float64), 0.0, 1.0)
    months = sea_interp(era_t, sea, ~land)
    months[:, land] = land_t[:, land]
    del land_t
    if not np.isfinite(months).all():
        raise SystemExit("normals: values missing after the ERA5 fill")
    coef = save_check(months, LAT10, LON10, land)
    meta = T2M_META | fit_meta(months, coef)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mean=coef[0].astype(np.float32),
        harm=coef[1:].astype(np.float16),
        lat=LAT10,
        lon=LON10,
        meta=json.dumps(meta),
    )
    print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def fit_meta(months: np.ndarray, coef: np.ndarray) -> dict:
    """The fit's month-average residuals and description."""
    resid = np.tensordot(HC.design(), coef, axes=(1, 0)) - months
    rms = np.sqrt((resid**2).mean(0))
    stats = {"median": np.median(rms), "p99": np.percentile(rms, 99), "max": rms.max()}
    stats = {k: round(float(v), 3) for k, v in stats.items()}
    print(f"  month-average residual RMS: {stats}", flush=True)
    return {
        "years": f"{YEARS[0]}-{YEARS[1]}",
        "hours": "00, 06, 12, 18 UTC",
        "curve": f"mean + {HC.NHARM} annual harmonics, month averages fitted to the monthly normals, common year",
        "resid_rms": stats,
        "built": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
    }


def save_check(months: np.ndarray, lat: np.ndarray, lon: np.ndarray, land: np.ndarray | None, k: float = 273.15):
    """The daily curves' coefficients through the monthly normals months (12, lat, lon); prints the PLACES' curves
    (less k: K -> deg C for the temperature)."""
    coef = HC.fit(months)
    days = pd.date_range("2021-01-01 12:00", "2021-12-31 12:00")
    t = HC.times_of(days)
    for place, (la, lo) in PLACES.items():
        j, i = cell(la, lo, lat, lon)
        curve = HC.evaluate(coef[:, j, i], t) - k
        mm = np.array([curve[days.month == m + 1].mean() for m in range(12)])
        kind = "" if land is None else ("land " if land[j, i] else "sea  ")
        print(
            f"  {place:22s} {kind}months {np.round(months[:, j, i] - k, 1)}; curve - month "
            f"{np.round(mm - (months[:, j, i] - k), 2)}; lowest {days[curve.argmin()]:%b %d} {curve.min():.1f}, "
            f"highest {days[curve.argmax()]:%b %d} {curve.max():.1f}",
            flush=True,
        )
    return coef


# ── 500 hPa height and sea-level pressure from the ERA5 store (offline) ───────────────────────────────────────────────
def store_years(var: str, scale: float):
    """(daily values (days, LATU, LONU) float64, their dates) of each year 1991-2020 of the store's global layer,
    checked: the 1.5-deg grid, every day, no missing value."""
    import xarray as xr
    from localdirs import env_dir

    root = env_dir("ERA5_STORE", "wb2_1p5_daily_global")
    for y in range(YEARS[0], YEARS[1] + 1):
        with xr.open_dataset(root / var / f"{var}_{y}.nc") as ds:
            da = ds[var].transpose("time", "latitude", "longitude")  # dimension orders differ between files
            lat, lon = da["latitude"].values, da["longitude"].values
            a = da.values.astype(np.float64) * scale
            t = pd.DatetimeIndex(da["time"].values)
        if lat[0] < lat[-1]:
            lat, a = lat[::-1], a[:, ::-1]
        if not np.allclose(lat, LATU) or lon.size != LONU.size:
            raise SystemExit(f"{var} {y}: not the 1.5-deg global grid")
        if np.isnan(a).any() or a.shape[0] != pd.Timestamp(f"{y}-12-31").dayofyear:
            raise SystemExit(f"{var} {y}: {a.shape[0]} days, {int(np.isnan(a).sum())} missing values")
        yield roll_to(a, float(lon[0]), 1.5, LONU), t


def upper(out: Path) -> None:
    res, meta = {}, {}
    for field, (var, unit, scale) in UPPER.items():
        s = np.zeros((12, LATU.size, LONU.size))
        n = np.zeros(12)
        for a, t in store_years(var, scale):
            mon = t.month.to_numpy()
            for m in range(12):
                s[m] += a[mon == m + 1].sum(0)
                n[m] += (mon == m + 1).sum()
        months = s / n[:, None, None]
        print(f"  {field} ({unit}): {int(n.sum())} days; global-mean month normals {np.round(months.mean((1, 2)), 1)}")
        res[field] = save_check(months, LATU, LONU, None, k=0.0).astype(np.float32)
        meta[field] = {"unit": unit} | fit_meta(months, res[field].astype(np.float64))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lat=LATU, lon=LONU, meta=json.dumps(UPPER_META | {"fields": meta}), **res)
    print(f"  -> {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)


def spread(out: Path) -> None:
    """SPREAD_OUT: per field and averaging period, the variance curve of the detrended n-day mean anomaly."""
    from scipy.ndimage import uniform_filter1d

    res = {}
    for field, (var, unit, scale) in UPPER.items():
        N = Normals.upper(field)
        parts, days = [], []
        for a, t in store_years(var, scale):
            parts.append((a - N.at(t + pd.Timedelta(hours=12))).astype(np.float32))
            days.append(t)
        A = np.concatenate(parts)
        del parts
        t = pd.DatetimeIndex(np.concatenate([d.values for d in days]))
        x = np.arange(A.shape[0], dtype=np.float64) - (A.shape[0] - 1) / 2  # the linear trend per cell, removed
        A -= A.mean(0)
        A -= (np.tensordot(x, A, axes=(0, 0)) / (x * x).sum()).astype(np.float32)[None] * x[:, None, None].astype(
            np.float32
        )
        B = basis366(doy366(t))
        G = B.T @ B
        for n in SPREAD_DAYS:
            An = uniform_filter1d(A, n, axis=0, mode="nearest") if n > 1 else A
            rhs = B.T @ (An * An).reshape(A.shape[0], -1).astype(np.float64)
            coef = np.linalg.solve(G, rhs).reshape(B.shape[1], LATU.size, LONU.size)
            res[f"{field}_{n}"] = coef.astype(np.float16)
            sd = np.sqrt(np.clip(np.tensordot(basis366(np.array([15, 196])), coef, axes=(1, 0)), 0, None))
            print(f"  {field} ({unit}) {n}-day: sd mid-Jan {sd[0].min():.1f}..{sd[0].max():.1f}, mid-Jul "
                  f"{sd[1].min():.1f}..{sd[1].max():.1f}", flush=True)  # fmt: skip
        del A
    meta = {"source": UPPER_META["source"], "years": f"{YEARS[0]}-{YEARS[1]}", "detrend": "linear per cell",
            "curve": f"variance: mean + {HC.NHARM} harmonics of the 366-day year", "days": list(SPREAD_DAYS)}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lat=LATU, lon=LONU, meta=json.dumps(meta), **res)
    print(f"  -> {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)


# ── interim 2 m normals, until the CDS build lands (GitHub Actions) ───────────────────────────────────────────────────
def wb2_t2m() -> tuple[np.ndarray, np.ndarray]:
    """(the 12 monthly normals (12, LATU, LONU) K, ERA5's surface height on the same grid (m)) from WeatherBench 2's
    ERA5 1.5-deg 6-hourly fields, 1991-2020, all four synoptic hours, read anonymously from its public bucket; checked:
    the grid, every step, no missing value."""
    import xarray as xr

    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})

    def grid(da) -> np.ndarray:
        da = da.transpose(..., "latitude", "longitude")  # the store keeps (time, longitude, latitude)
        lat, lon = da["latitude"].values, da["longitude"].values
        a = da.values.astype(np.float64)
        if lat[0] < lat[-1]:
            lat, a = lat[::-1], a[..., ::-1, :]
        if not np.allclose(lat, LATU) or lon.size != LONU.size:
            raise SystemExit("WeatherBench 2: not the 1.5-deg global grid")
        return roll_to(a, float(lon[0]), 1.5, LONU)

    zs = grid(ds["geopotential_at_surface"]) / 9.80665
    s, n = np.zeros((12, LATU.size, LONU.size)), np.zeros(12)
    for y in range(YEARS[0], YEARS[1] + 1):
        da = ds["2m_temperature"].sel(time=str(y))
        t = pd.DatetimeIndex(da["time"].values)
        a = grid(da)
        if a.shape[0] != 4 * pd.Timestamp(f"{y}-12-31").dayofyear or np.isnan(a).any():
            raise SystemExit(
                f"WeatherBench 2 2 m temperature {y}: {a.shape[0]} steps, {int(np.isnan(a).sum())} missing"
            )
        for m in range(12):
            sel = t.month == m + 1
            s[m] += a[sel].sum(0)
            n[m] += sel.sum()
        print(f"  ERA5 1.5-deg 2 m temperature {y}: {a.shape[0]} steps, mean {a.mean() - 273.15:.2f} °C", flush=True)
    return s / n[:, None, None], zs


def column_height(msl: np.ndarray, sp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """m: the depth of a column with the standard lapse rate GAMMA from the surface (pressure sp, temperature t) down to
    sea level (pressure msl), the inverse of the standard reduction to sea level."""
    return t / GAMMA * ((msl / sp) ** (RD * GAMMA / 9.80665) - 1.0)


def ifs_height10(date: str) -> np.ndarray:
    """(LAT10, LON10) m: the IFS's surface height on its 0.1-deg grid. The open data carry it at 0.25 deg only (the
    control's step-0 surface geopotential); the 0.1-deg detail comes from the hypsometric height of one member's
    step-0 sea-level and surface pressure and 2 m temperature on both grids: z = z(0.25) + h(0.1) - h(0.25), the
    0.25-deg fields interpolated bilinearly, so the approximations of the reduction to sea level cancel."""
    import ensmaps as E
    import rangefetch as rf

    def fields(resol: str, levtype: str | None) -> dict:
        idx = rf.fetch_index(date, "00", "ifs", 0, "ef", resol=resol, levtype=levtype)
        want = sorted(
            (e for e in idx if e.get("param") in ("msl", "sp", "2t") and str(e.get("number")) == "1"),
            key=lambda e: e["_offset"],
        )
        path = rf.path_for(date, "00", "ifs", 0, "ef", resol=resol, levtype=levtype) + ".grib2"
        if len(want) != 3:
            raise SystemExit(f"IFS-ENS {date} 00Z {resol}: {len(want)} of the 3 fields of member 1 in the index")
        blob = rf.fetch_ranges(path, rf.coalesce(want))
        ends = np.cumsum([e["_length"] for e in want])  # the wanted messages back to back (as ensmaps.ecmwf_messages)
        return {
            e["param"]: E.decode(blob[x - e["_length"] : x], "number")[1].astype(np.float64)
            for e, x in zip(want, ends, strict=True)
        }

    f10, f25 = fields("0p1", "sfc"), fields("0p25", None)
    z25 = E.ifs_orography(date, "00").astype(np.float64)
    h10, h25 = (column_height(f["msl"], f["sp"], f["2t"]) for f in (f10, f25))
    up = E.bilinear(np.stack([z25, h25]), LAT25, LON25, E.G10)
    z = up[0] + h10 - up[1]
    rows = np.rint((90.0 - LAT25[::2]) * 10).astype(int)  # the 0.5-deg nodes, which both grids hold
    cols = np.rint((LON25[::2] + 180.0) * 10).astype(int)
    diff = z[np.ix_(rows, cols)] - z25[::2, ::2]
    print(f"  IFS 0.1-deg height: {z.min():.0f}..{z.max():.0f} m; at the 0.5-deg nodes less the 0.25-deg height: "
          f"mean {diff.mean():.1f} m, |p99| {np.percentile(np.abs(diff), 99):.0f} m", flush=True)  # fmt: skip
    return z


def interim(natemp: Path, out: Path, date: str) -> None:
    """INTERIM_OUT: the ERA5 1.5-deg curve on the 0.1-deg grid at the IFS's height (6.5 K/km from ERA5's 1.5-deg
    height), the North America maps' normals (release natemp-ref-v1) inside their box, blended over NA_TAPER inside its
    edge; the final file's layout, so ensmaps.py reads either."""
    import ensmaps as E

    months, zs = wb2_t2m()
    coef = save_check(months, LATU, LONU, None)
    meta = INTERIM_META | fit_meta(months, coef)
    c10 = E.bilinear(np.concatenate([coef, zs[None]]), LATU, LONU, E.G10).astype(np.float64)
    z10 = ifs_height10(date)
    c10[0] -= E.LAPSE * (z10 - c10[-1])
    c10 = c10[:-1]
    with np.load(natemp) as zn:
        cn, la, lo = zn["coef"].astype(np.float64), zn["lat"], zn["lon"]
    r0, k0 = int(np.rint((90.0 - la[0]) * 10)), int(np.rint((lo[0] + 180.0) * 10))
    if not (np.allclose(LAT10[r0 : r0 + la.size], la) and np.allclose(LON10[k0 : k0 + lo.size], lo)):
        raise SystemExit(f"{natemp}: not on the 0.1-deg grid")
    inner = np.minimum.outer(np.minimum(la - la[-1], la[0] - la), np.minimum(lo - lo[0], lo[-1] - lo))
    w = 0.5 - 0.5 * np.cos(np.pi * np.clip(inner / NA_TAPER, 0.0, 1.0))
    box = c10[:, r0 : r0 + la.size, k0 : k0 + lo.size]
    edge = w < 1
    step = cn[0][edge] - box[0][edge]
    print(f"  North America normals less the interim inside the {NA_TAPER:.0f}-deg blend: mean {step.mean():.2f} K, "
          f"|p99| {np.percentile(np.abs(step), 99):.2f} K", flush=True)  # fmt: skip
    c10[:, r0 : r0 + la.size, k0 : k0 + lo.size] = w * cn + (1 - w) * box
    for place, (pla, plo) in PLACES.items():
        j, i = cell(pla, plo, LAT10, LON10)
        print(f"  {place:22s} interim annual mean {c10[0, j, i] - 273.15:.1f} °C at {z10[j, i]:.0f} m", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=c10[0].astype(np.float32), harm=c10[1:].astype(np.float16), lat=LAT10, lon=LON10,
                        meta=json.dumps(meta))  # fmt: skip
    print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out-dir", type=Path, required=True)
    b = sub.add_parser("fit")
    b.add_argument("--in-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("upper")
    u.add_argument("--out", type=Path, default=UPPER_OUT)
    v = sub.add_parser("spread")
    v.add_argument("--out", type=Path, default=SPREAD_OUT)
    t = sub.add_parser("interim")
    t.add_argument("--natemp", type=Path, required=True, help="natemp_normals.npz (release natemp-ref-v1)")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--date", default=(dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d"),
                   help="the IFS-ENS 00Z run whose step 0 gives the 0.1-deg height (default yesterday's)")  # fmt: skip
    a = ap.parse_args()
    if a.cmd == "interim":
        interim(a.natemp, a.out, a.date)
        return 0
    if a.cmd == "spread":
        spread(a.out)
        return 0
    if a.cmd == "fetch":
        missing = fetch(a.out_dir)
        print(f"  {missing} unit(s) not fetched", flush=True)
        return 1 if missing else 0
    if a.cmd == "upper":
        upper(a.out)
        return 0
    missing = [name for name, _, _ in units() if not (a.in_dir / f"ensmaps_{name}.npz").exists()]
    if missing:
        print(f"  units not fetched yet: {', '.join(missing)}", flush=True)
        return 3
    fit(a.in_dir, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
