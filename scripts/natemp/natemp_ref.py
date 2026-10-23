#!/usr/bin/env python3
"""Daily normals of the 2 m temperature for the North America maps (natemp.py), on the IFS-ENS 0.1-deg grid's own
points over the box (LAT, LON): ERA5-Land 1991-2020 over land, ERA5 over the sea, as a smooth annual curve per point.

ERA5-Land is ECMWF's land reanalysis on the ~9 km grid of the IFS (TCo1279) with its orography, so its 0.1-deg 2 m
temperature matches the IFS-ENS 0.1-deg fields point for point without a height correction. It has no value over the
sea; there ERA5 (0.25 deg) is interpolated bilinearly from its SEA nodes (each node weighted by its sea fraction, ERA5's
land-sea mask), so a coastal mountain node does not chill the water beside it; a point with no sea node around it
takes the plain bilinear value.

Both reanalyses are read as monthly means BY HOUR OF DAY at 00, 06, 12 and 18 UTC and averaged over the four hours: the
same four synoptic times a forecast day is the mean of (natemp.py), so normal and forecast sample the daily cycle alike.

The daily curve: mean + three annual harmonics on the common-year calendar, fitted by least squares so that the
curve's MONTH AVERAGES - its exact integrals over each month - match the 12 monthly normals (scripts/lib/harmonic_clim:
one 12 x 7 design matrix serves every point). Normals.at reads the curve at any times; natemp.py averages it over a
day's four forecast times. The fit prints the month residuals and a few cities' curves.

    python scripts/natemp/natemp_ref.py fetch --out-dir ref      # Actions only: CDS, one request at a time
    python scripts/natemp/natemp_ref.py fit --in-dir ref --out ref/natemp_normals.npz   # exit 3: units missing

Until then, interim normals (see "interim" below): the ERA5 store's 1.5-deg cells downscaled to the 0.1-deg points.

    python scripts/natemp/natemp_ref.py coarse --out ref/natemp_era5_1p5.npz       # reads $ERA5_STORE
    python scripts/natemp/natemp_ref.py interim --coarse ref/natemp_era5_1p5.npz --date 20261023 --time 00 \
        --out ref/natemp_normals_interim.npz                                         # natemp.yml
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
sys.path[:0] = [str(HERE.parents[0] / "lib"), str(HERE.parents[0] / "sst"), str(HERE.parents[0] / "ecmwf")]
import harmonic_clim as HC

# The box: rows 77N..9N and columns 180W..26W of the IFS 0.1-deg grid (3600 x 1801, 90N first), which hold the whole
# of the maps' Lambert rectangle (natemp_maps.EXTENT).
LAT = np.round(np.arange(770, 89, -1) / 10.0, 1)
LON = np.round(np.arange(-1800, -259) / 10.0, 1)
ROW0 = 130  # the box's first row in the global grid
AREA = [77.0, -180.0, 9.0, -26.0]  # CDS area: north, west, south, east
LAT25 = np.round(np.arange(7700, 899, -25) / 100.0, 2)  # ERA5's 0.25-deg nodes over the same box
LON25 = np.round(np.arange(-18000, -2599, 25) / 100.0, 2)
YEARS = (1991, 2020)
HOURS = ("00:00", "06:00", "12:00", "18:00")
DECADES = ((1991, 2000), (2001, 2010), (2011, 2020))  # one CDS request each: 480 fields
OUT = HERE / "data" / "natemp_normals.npz"
FINAL_META = {
    "source": "ERA5-Land monthly means by hour of day over land; ERA5 (0.25 deg, sea nodes) over the sea",
    "label": "ERA5-Land 1991–2020 over land, ERA5 over the sea, as a smooth daily curve",
}
INTERIM_META = {
    "source": "ERA5 daily means (WeatherBench 2, 1.5 deg) downscaled to the IFS 0.1-deg land fraction and height by "
    "local regression; interim, until the ERA5-Land normals",
    "label": "ERA5 1991–2020 (1.5°) downscaled to the 0.1° land and terrain, interim until ERA5-Land",
}
CITIES = {  # the fit's printed check
    "Chicago": (41.9, -87.6),
    "Denver": (39.7, -105.0),
    "Miami": (25.8, -80.2),
    "Fairbanks": (64.8, -147.7),
    "Mexico City": (19.4, -99.1),
    "Gulf of Alaska (sea)": (56.0, -145.0),
}


def cell(lat: float, lon: float) -> tuple[int, int]:
    """(row, column) of the box's grid point nearest to lat, lon."""
    return round((LAT[0] - lat) * 10), round((lon - LON[0]) * 10)


class Normals:
    """The 2 m temperature normal curve (K) on (LAT, LON), from the fitted coefficients (release natemp-ref-v1)."""

    def __init__(self, path: Path = OUT):
        with np.load(path) as z:
            self.coef = z["coef"].astype(np.float64)
            self.meta = json.loads(str(z["meta"]))
            if not (np.allclose(z["lat"], LAT) and np.allclose(z["lon"], LON)):
                raise SystemExit(f"{path}: the normals are not on the box's grid")

    def at(self, times) -> np.ndarray:
        """(len(times), LAT, LON): the curve at those times."""
        return HC.evaluate(self.coef, HC.times_of(times))


# ── fetch (GitHub Actions only) ───────────────────────────────────────────────────────────────────────────────────────
def units() -> list[tuple[str, str, dict]]:
    """(name, CDS dataset, request) of every request, in fetch order."""
    common = {"variable": ["2m_temperature"], "month": [f"{m:02d}" for m in range(1, 13)], "time": list(HOURS)}
    common |= {"area": AREA, "data_format": "grib", "download_format": "unarchived"}
    out = []
    for src, dataset in (
        ("land", "reanalysis-era5-land-monthly-means"),
        ("era5", "reanalysis-era5-single-levels-monthly-means"),
    ):
        for y0, y1 in DECADES:
            req = {"product_type": ["monthly_averaged_reanalysis_by_hour_of_day"], **common}
            out.append((f"{src}_{y0}", dataset, req | {"year": [str(y) for y in range(y0, y1 + 1)]}))
    lsm = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["land_sea_mask"], "year": ["2020"]}
    lsm |= {"month": ["01"], "time": ["00:00"], "area": AREA, "data_format": "grib", "download_format": "unarchived"}
    out.append(("era5_lsm", "reanalysis-era5-single-levels-monthly-means", lsm))
    return out


def grid_of(src: str):
    return (LAT, LON) if src == "land" else (LAT25, LON25)


def read_field(gid, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """One GRIB message on the regular lat/lon grid (lat, lon), NaN where the message has no value; refuses any other
    grid. Longitudes are compared modulo 360 (the CDS may write the box as 180..334 E)."""
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
    la = g["latitudeOfFirstGridPointInDegrees"] + (1 if g["jScansPositively"] else -1) * g[
        "jDirectionIncrementInDegrees"
    ] * np.arange(g["Nj"])
    lo = g["longitudeOfFirstGridPointInDegrees"] + g["iDirectionIncrementInDegrees"] * np.arange(g["Ni"])
    v = eccodes.codes_get_values(gid).reshape(g["Nj"], g["Ni"])
    if eccodes.codes_get(gid, "bitmapPresent"):
        v = np.where(v == eccodes.codes_get_double(gid, "missingValue"), np.nan, v)
    if g["jScansPositively"]:
        la, v = la[::-1], v[::-1]
    if (
        la.size != lat.size
        or lo.size != lon.size
        or not np.allclose(la, lat, atol=1e-4)
        or not np.allclose(lo % 360, lon % 360, atol=1e-4)
    ):
        raise SystemExit(f"GRIB grid {g} is not the expected {lat[0]}..{lat[-1]} x {lon[0]}..{lon[-1]}")
    return v


def reduce_unit(name: str, path: Path, out: Path) -> None:
    """A unit's GRIB -> its per-month mean over years and hours (12, lat, lon) and the number of fields behind each
    value; checks that every month of every year holds four fields of distinct times (the four hours asked for) and
    prints the first month's times and box means."""
    import eccodes

    src = name.split("_")[0]
    lat, lon = grid_of(src)
    s = np.zeros((12, lat.size, lon.size))
    n = np.zeros((12, lat.size, lon.size), np.int16)
    times: dict = {}  # YYYYMM -> [(time keys, box mean)]
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
    print(f"  {name} {first}: (dataTime, validityTime, stepRange) and box mean: {times[first]}", flush=True)
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
        raise SystemExit("the CDS is read in GitHub Actions only (natemp-ref.yml)")
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name, dataset, req in units():
        npz = out_dir / f"natemp_{name}.npz"
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


# ── fit ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def merged(in_dir: Path, src: str):
    """The 1991-2020 monthly normals (12, lat, lon) of one reanalysis from its three decades, and the field counts."""
    s = n = 0
    for y0, _ in DECADES:
        with np.load(in_dir / f"natemp_{src}_{y0}.npz") as z:
            m, k = z["mean"].astype(np.float64), z["n"].astype(np.int64)
        s, n = s + np.where(k > 0, m * k, 0.0), n + k
    with np.errstate(invalid="ignore"):
        return s / n, n


def sea_interp(t25: np.ndarray, sea: np.ndarray) -> np.ndarray:
    """(12, LAT25, LON25) ERA5 monthly normals -> (12, LAT, LON): bilinear from the sea nodes, weighted by sea fraction;
    plain bilinear where no sea node is near."""
    from scipy.interpolate import RegularGridInterpolator

    pts = np.stack(np.meshgrid(LAT[::-1], LON, indexing="ij"), -1).reshape(-1, 2)

    def interp(a):
        f = RegularGridInterpolator((LAT25[::-1], LON25), a[::-1], method="linear")
        return f(pts).reshape(LAT.size, LON.size)[::-1]

    w = interp(sea)
    out = np.empty((12, LAT.size, LON.size))
    for m in range(12):
        plain, tw = interp(t25[m]), interp(t25[m] * sea)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = np.where(w > 1e-3, tw / w, plain)
    return out


def fit(in_dir: Path, out: Path) -> None:
    land_t, land_n = merged(in_dir, "land")
    era_t, era_n = merged(in_dir, "era5")
    full = 10 * len(DECADES) * len(HOURS)  # fields per month behind a complete normal
    if not (era_n == full).all():
        raise SystemExit(f"ERA5: {int((era_n != full).sum())} values without all {full} fields")
    land = (land_n == full).all(0)
    part = ((land_n > 0) & (land_n < full)).any(0)
    print(f"  ERA5-Land: {land.mean():.1%} of the box is land with all {full} fields", flush=True)
    print(f"  {int(part.sum())} points with only some ERA5-Land fields take ERA5", flush=True)
    with np.load(in_dir / "natemp_era5_lsm.npz") as z:
        sea = 1.0 - np.clip(z["mean"][0].astype(np.float64), 0.0, 1.0)
    months = np.where(land[None], land_t, sea_interp(era_t, sea))
    if not np.isfinite(months).all():
        raise SystemExit("normals: values missing after the ERA5 fill")
    save_fit(months, land, out, FINAL_META)


def save_fit(months: np.ndarray, land: np.ndarray, out: Path, meta: dict) -> None:
    """The daily curves through the monthly normals months (12, LAT, LON) -> out, with meta; prints the month residuals
    and the CITIES' curves (land: the points called land there)."""
    coef = HC.fit(months)
    resid = np.tensordot(HC.design(), coef, axes=(1, 0)) - months
    rms = np.sqrt((resid**2).mean(0))
    stats = {"median": np.median(rms), "p99": np.percentile(rms, 99), "max": rms.max()}
    stats = {k: round(float(v), 3) for k, v in stats.items()}
    print(f"  month-average residual RMS (K): {stats}", flush=True)
    meta = meta | {
        "years": f"{YEARS[0]}-{YEARS[1]}",
        "hours": "00, 06, 12, 18 UTC",
        "curve": f"mean + {HC.NHARM} annual harmonics, month averages fitted to the monthly normals, common year",
        "resid_rms_K": stats,
        "built": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, coef=coef.astype(np.float32), lat=LAT, lon=LON, meta=json.dumps(meta))
    print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    days = pd.date_range("2021-01-01 12:00", "2021-12-31 12:00")
    t = HC.times_of(days)
    for city, (la, lo) in CITIES.items():
        j, i = cell(la, lo)
        curve = HC.evaluate(coef[:, j, i], t) - 273.15
        mm = np.array([curve[days.month == m + 1].mean() for m in range(12)])
        print(
            f"  {city:22s} {'land' if land[j, i] else 'sea '} months {np.round(months[:, j, i] - 273.15, 1)} °C; "
            f"curve - month {np.round(mm - (months[:, j, i] - 273.15), 2)}; coldest {days[curve.argmin()]:%b %d} "
            f"{curve.min():.1f}, warmest {days[curve.argmax()]:%b %d} {curve.max():.1f}",
            flush=True,
        )


# ── interim: ERA5 1.5 deg downscaled to the IFS 0.1-deg land and terrain (until the ERA5-Land normals exist) ──────────
# The ERA5 store's daily 2 m temperature (WeatherBench 2: the daily mean of 00, 06, 12 and 18 UTC, conservatively
# averaged onto 1.5-deg cells) gives the 1991-2020 monthly normals of each cell (coarse, an offline build from the
# store; natemp_era5_1p5.npz in the release). Their structure below 1.5 deg comes from the land and the height of the
# IFS 0.1-deg grid (its step-0 lsm and z, Google Cloud mirror): L is the land fraction smoothed over COAST_KM, so the
# sea's influence fades inland over a few tens of km rather than stopping at the coastline. Per cell and month,
# T = a + p dy + q dx + b L + c z by weighted least squares over the 5 x 5 cells around it (Gaussian weights, SIGMA
# cells): b is the local land-sea contrast and c the local lapse rate (an inversion gives c > 0), both drawn weakly
# towards PRIOR where the cells do not vary. A point's normal is the bilinear cell normal plus b and c times its L's
# and height's departures from the bilinear cell means.
CELL = 1.5
CLAT = np.round(np.arange(3, 55) * CELL, 2)  # 4.5..81 N: the box and three cells beyond it
CLON = np.round(np.arange(117, 227) * CELL, 2)  # 175.5..339 E
RADIUS, SIGMA = 2, 1.0  # cells
COAST_KM = 25.0
# (prior, weight): K per unit land fraction and K/km, weighted as one cell 0.05 of land or 0.05 km of height away
PRIOR = {"land": (0.0, 0.05), "height": (-6.5, 0.05)}
COARSE = HERE / "data" / "natemp_era5_1p5.npz"


def coarse(out: Path) -> None:
    """The ERA5 store's daily 2 m temperature 1991-2020 -> its monthly normals (12, CLAT, CLON), every day of a month
    weighted alike."""
    import xarray as xr
    from localdirs import env_dir

    s = np.zeros((12, CLAT.size, CLON.size))
    n = np.zeros(12)
    for y in range(YEARS[0], YEARS[1] + 1):
        with xr.open_dataset(env_dir("ERA5_STORE", "wb2_1p5_daily", "t2m", f"t2m_{y}.nc")) as ds:
            t = ds["t2m"].transpose("time", "latitude", "longitude")
            t = t.sel(latitude=CLAT, longitude=CLON, method="nearest", tolerance=1e-3).load()
        if t.latitude.size != CLAT.size or t.longitude.size != CLON.size or not 180 < float(t.mean()) < 320:
            raise SystemExit(f"t2m_{y}.nc: not the store's 1.5-deg grid or not kelvin")
        month = t.time.dt.month.values
        for m in range(12):
            s[m] += t.values[month == m + 1].sum(0)
            n[m] += (month == m + 1).sum()
    want = [sum(pd.Period(f"{y}-{m:02d}").days_in_month for y in range(YEARS[0], YEARS[1] + 1)) for m in range(1, 13)]
    if n.tolist() != want:
        raise SystemExit(f"days per month {n.tolist()}, expected {want}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=(s / n[:, None, None]).astype(np.float32), lat=CLAT, lon=CLON, days=n)
    print(f"  ERA5 1.5-deg monthly normals {YEARS[0]}-{YEARS[1]} ({int(n.sum())} days) -> {out}", flush=True)


def cell_weights(cells: np.ndarray, pts: np.ndarray, h: float) -> np.ndarray:
    """(cells, pts): the overlap of each point's cell (width h) with each CELL-wide cell, normalised per cell."""
    lo = np.maximum(cells[:, None] - CELL / 2, pts[None] - h / 2)
    hi = np.minimum(cells[:, None] + CELL / 2, pts[None] + h / 2)
    w = np.clip(hi - lo, 0.0, None)
    return w / w.sum(1, keepdims=True)


def coarsen(f: np.ndarray, lat: np.ndarray, lon: np.ndarray, h: float, clat=CLAT, clon=CLON) -> np.ndarray:
    """The area means over the cells (clat, clon) of a field on the h-deg points (lat, lon); longitudes east, unwrapped."""
    wy = cell_weights(clat, lat, h) * np.cos(np.deg2rad(lat))[None]
    wx = cell_weights(clon, lon, h)
    return (wy @ f @ wx.T) / (wy.sum(1)[:, None] * wx.sum(1)[None])


def exposure(lf: np.ndarray, lat: np.ndarray, h: float) -> np.ndarray:
    """The land fraction lf on h-deg points smoothed by a Gaussian of COAST_KM (its east-west width in degrees grows
    with latitude)."""
    from scipy.ndimage import gaussian_filter1d

    km = 111.2 * h
    out = gaussian_filter1d(lf.astype(np.float64), COAST_KM / km, axis=0, mode="nearest")
    for j, la in enumerate(lat):
        out[j] = gaussian_filter1d(out[j], COAST_KM / (km * np.cos(np.deg2rad(la))), mode="nearest")
    return out


def local_slopes(tc: np.ndarray, lc: np.ndarray, zc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(b, c), each (12, cells): the land-fraction (K) and height (K/km) slopes of T = a + p dy + q dx + b L + c z over
    the (2 RADIUS + 1)^2 cells around each cell, from its monthly normals tc (12, cells), land fraction lc and height
    zc (m); cells beyond the edge are left out."""
    from numpy.lib.stride_tricks import sliding_window_view

    r, k = RADIUS, 2 * RADIUS + 1

    def win(a):
        pad = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)], constant_values=np.nan)
        return sliding_window_view(pad, (k, k), axis=(-2, -1))

    dy, dx = np.mgrid[-r : r + 1, -r : r + 1].astype(float)
    L, Z = win(lc), win(zc / 1000.0)
    ok = np.isfinite(L)
    w = np.where(ok, np.exp(-(dy**2 + dx**2) / (2 * SIGMA**2)), 0.0)
    X = np.stack(np.broadcast_arrays(1.0, dy, dx, np.where(ok, L, 0.0), np.where(ok, Z, 0.0)), -1)
    A = np.einsum("jiab,jiabp,jiabq->jipq", w, X, X)
    rhs0 = np.zeros(A.shape[:-1])
    wsum = w.sum((-2, -1))
    for p, (prior, spread) in zip((3, 4), PRIOR.values()):
        A[..., p, p] += wsum * spread**2
        rhs0[..., p] += wsum * spread**2 * prior
    b, c = np.empty_like(tc), np.empty_like(tc)
    for m in range(tc.shape[0]):
        y = np.where(ok, win(tc[m]), 0.0)
        beta = np.linalg.solve(A, (rhs0 + np.einsum("jiab,jiabp,jiab->jip", w, X, y))[..., None])[..., 0]
        b[m], c[m] = beta[..., 3], beta[..., 4]
    return b, c


def downscale(tc, lc, zc, b, c, lf, zf, lat, lon, clat=CLAT, clon=CLON) -> np.ndarray:
    """(12, lat, lon): the cells' normals tc at the points (lat, lon) of land fraction lf and height zf (m) - the
    bilinear cell normal plus the local slopes b, c times the point's departures from the bilinear cell land fraction
    and height."""
    from scipy.interpolate import RegularGridInterpolator

    pts = np.stack(np.meshgrid(lat, lon, indexing="ij"), -1).reshape(-1, 2)
    up = clat[0] > clat[-1]

    def bil(a):
        a = a[::-1] if up else a
        f = RegularGridInterpolator((clat[::-1] if up else clat, clon), a, method="linear")
        return f(pts).reshape(lat.size, lon.size)

    dl, dz = lf - bil(lc), (zf - bil(zc)) / 1000.0
    return np.stack([bil(tc[m]) + bil(b[m]) * dl + bil(c[m]) * dz for m in range(tc.shape[0])])


def ifs_static(date: str, hh: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(land fraction, height m, lat, lon east) of the IFS 0.1-deg grid over the cells CLAT x CLON, from the step-0
    lsm and z of the deterministic run date hh (Google Cloud mirror)."""
    import eccodes
    import rangefetch as rf

    lat = np.round(np.arange(round((CLAT[-1] + 0.7) * 10), round((CLAT[0] - 0.7) * 10) - 1, -1) / 10.0, 1)
    lon = np.round(np.arange(round((CLON[0] - 0.7) * 10), round((CLON[-1] + 0.7) * 10) + 1) / 10.0, 1)
    ((path, idx),) = rf.oper_files(date, hh, "ifs", 0, resol="0p1", levtypes=("sfc",))
    want = sorted(rf.select(idx, param=["lsm", "z"]), key=lambda e: e["_offset"])
    if sorted(e["param"] for e in want) != ["lsm", "z"]:
        raise SystemExit(f"{path}: no single lsm and z at step 0")
    blob = rf.fetch_ranges(path, rf.coalesce(want))
    out = {}
    for e, end in zip(want, np.cumsum([e["_length"] for e in want])):
        gid = eccodes.codes_new_from_message(blob[end - e["_length"] : end])
        try:
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            lat0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
            lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
            if (ni, nj, lat0, eccodes.codes_get(gid, "jScansPositively")) != (3600, 1801, 90.0, 0):
                raise SystemExit(f"{e['param']}: not the global 0.1-deg grid")
            v = eccodes.codes_get_values(gid).reshape(nj, ni)
        finally:
            eccodes.codes_release(gid)
        rows = np.round((90.0 - lat) * 10).astype(int)
        cols = np.round((lon - lon0) % 360 * 10).astype(int) % 3600
        out[e["param"]] = v[rows][:, cols]
    return out["lsm"], out["z"] / 9.80665, lat, lon


def interim(coarse_path: Path, date: str, hh: str, out: Path) -> None:
    """The interim normals from the cells' monthly normals (coarse) and the IFS run date hh's land and height."""
    with np.load(coarse_path) as z:
        tc, clat, clon = z["mean"].astype(np.float64), z["lat"], z["lon"]
    if not (np.array_equal(clat, CLAT) and np.array_equal(clon, CLON)):
        raise SystemExit(f"{coarse_path}: not on the cells CLAT x CLON")
    lsm, zf, lat, lon = ifs_static(date, hh)
    lf = exposure(lsm, lat, 0.1)
    lc, zc = coarsen(lf, lat, lon, 0.1), coarsen(zf, lat, lon, 0.1)
    b, c = local_slopes(tc, lc, zc)
    rows = np.searchsorted(-lat, -LAT)
    cols = np.searchsorted(lon, LON % 360)
    if not (np.allclose(lat[rows], LAT) and np.allclose(lon[cols], LON % 360)):
        raise SystemExit("the box is not on the IFS points")
    box = np.ix_(rows, cols)
    months = downscale(tc, lc, zc, b, c, lf[box], zf[box], LAT, LON % 360)
    land = lc > 0.5
    for name, x in (("land-fraction slope b, K", b), ("height slope c, K/km", c)):
        q = np.percentile(x[:, land], [5, 50, 95], axis=1).round(1)
        print(f"  {name} over land cells, Jan / Jul (5, 50, 95 %): {q[:, 0].tolist()} / {q[:, 6].tolist()}", flush=True)
    save_fit(months, lsm[box] >= 0.5, out, INTERIM_META)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out-dir", type=Path, required=True)
    b = sub.add_parser("fit")
    b.add_argument("--in-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, default=OUT)
    c = sub.add_parser("coarse")
    c.add_argument("--out", type=Path, default=COARSE)
    i = sub.add_parser("interim")
    i.add_argument("--coarse", type=Path, default=COARSE)
    i.add_argument("--date", required=True, help="YYYYMMDD of an IFS run on the mirror (its step-0 land and height)")
    i.add_argument("--time", required=True, choices=("00", "06", "12", "18"))
    i.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "coarse":
        coarse(a.out)
        return 0
    if a.cmd == "interim":
        interim(a.coarse, a.date, a.time, a.out)
        return 0
    if a.cmd == "fetch":
        missing = fetch(a.out_dir)
        print(f"  {missing} unit(s) not fetched", flush=True)
        return 1 if missing else 0
    missing = [name for name, _, _ in units() if not (a.in_dir / f"natemp_{name}.npz").exists()]
    if missing:
        print(f"  units not fetched yet: {', '.join(missing)}", flush=True)
        return 3
    fit(a.in_dir, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
