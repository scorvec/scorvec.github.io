#!/usr/bin/env python3
"""The observed RMM: the Wheeler & Hendon (2004) index computed here from ERA5, 1979 to ARCO-ERA5's latest day.

The truth every MJO forecast on the site is verified against (the GEFS and GEPS hindcast skill, the observed MJO
composites of mjo.html, the checks of the ENSO correction), written to data/reference/rmm_era5.nc. It is the
arithmetic of the site's forecast RMM on the same reference (climatology.nc and eofs.nc: NOAA interpolated OLR and
NCEP winds 1979-2001, three annual harmonics), so a forecast and its truth share one definition:

  bands      daily 15S-15N cos-weighted band means on the reference's 144 longitudes. u850 and u200: the mean of the
             00, 06, 12 and 18 UTC fields, from WeatherBench2's 1.5 deg ERA5 to 2022 (interpolated to the
             longitudes) and from ARCO-ERA5 0.25 deg after it (the 2.5 deg cell centred on each longitude). OLR: ERA5
             top-of-atmosphere longwave, the RMM band of scripts/sst/build_olr_baseline.py fetch (the mean of the
             hourly means ending 00, 06, 12 and 18 UTC), the reduction of the live OLR channel and of its frame.
  anomalies  against the reference climatology on the common-year day of year (doycal.sel_doy); the OLR also minus
             the fixed ERA5-to-NOAA frame offset (olr_channel.frame_offset), as in the forecasts' OLR channel.
  filter     minus the mean of the 120 days ending on the day, the convention the EOFs were trained with
             (setup_reference.remove_prev120). ENSO is NOT removed; the products that need it subtract enso_rmm's
             correction, from the series on every date.
  index      each field divided by its reference standard deviation and projected onto eofs.nc: rmm1/rmm2 the
             three-field index (pc_std), rmm1_wind/rmm2_wind the wind-only index of the MJO page's diagram
             (pc_wind_std).

The file holds the daily series and the band means of its last TAIL days, from which `update` (mjo.yml, every run)
appends the days ARCO-ERA5 has added since; `build` (rmm-era5.yml) remakes the whole record from the parts `fetch`
writes, and the band means of the whole record go to the release asset BANDS (rmm-era5-v1). ARCO-ERA5 has the odd
hole near real time (2026-10-10 is missing while the days after it are there): a hole of at most MAX_GAP days inside
the record is interpolated linearly in time, band mean by band mean, and listed in the file's `interpolated`
attribute; a longer one stops the build, and ends an update before it.

    python src/rmm_era5.py plan                                         # the build's parts, as JSON
    python src/rmm_era5.py fetch --source wb2 --start 1979-01-01 --end 1983-12-31 --out part_1979.npz
    python src/rmm_era5.py build --parts part_*.npz [--seam seam.npz] [--bands era5_rmm_bands.npz]
    python src/rmm_era5.py update                                       # the new ARCO-ERA5 days
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olr_channel  # frame_offset
from doycal import noleap_doy, sel_doy

REF = HERE.parent / "data" / "reference"
OUT = REF / "rmm_era5.nc"
BANDS = "era5_rmm_bands.npz"  # the whole record's band means, release asset of rmm-era5.yml
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FIRST = pd.Timestamp("1979-01-01")
WB2_LAST = pd.Timestamp("2022-12-31")  # WeatherBench2's ERA5 ends on 10 January 2023; ARCO-ERA5 from 2023
HOURS = (0, 6, 12, 18)
LEVELS = (850, 200)
HALF = 15.0  # the band, 15S-15N
LON = np.arange(0, 360, 2.5)
TAGS = ("olr", "u850", "u200")
WINDOW = 120  # WH04's filter: the mean of the 120 days ending on the day
TAIL = 150  # days of band means kept for the next update's 120-day means
UPDATE_MAX = 40  # days appended by one update at most (ARCO runs about a week behind)
MAX_GAP = 3  # days ERA5 lacks inside the record that are interpolated; a longer hole stops the build or update
CELL = (10 * np.arange(len(LON))[:, None] + np.arange(-5, 6)) % 1440  # 0.25 deg columns of each centred 2.5 deg cell
CELL_W = np.r_[0.5, np.ones(9), 0.5] / 10.0  # the two edge columns are shared with the neighbouring cells


def olr_baseline():
    """scripts/sst/build_olr_baseline.py, the OLR band reduction (appended to the path where it is used, so its
    folder never shadows an importer's modules)."""
    sst = str(HERE.parents[1] / "sst")
    if sst not in sys.path:
        sys.path.append(sst)
    import build_olr_baseline

    return build_olr_baseline


# ── reference ─────────────────────────────────────────────────────────────────────────────────────────────────
def reference() -> tuple[xr.Dataset, xr.Dataset, xr.DataArray]:
    """climatology.nc, eofs.nc and the OLR frame offset (dayofyear, longitude), loaded."""
    clim = xr.open_dataset(REF / "climatology.nc").load()
    eofs = xr.open_dataset(REF / "eofs.nc").load()
    if not (np.allclose(clim.longitude.values, LON) and np.allclose(eofs.longitude.values, LON)):
        raise SystemExit("the WH04 reference is not on the 2.5 deg longitudes")
    return clim, eofs, olr_channel.frame_offset(clim)


def anomalies(t: pd.DatetimeIndex, bands: dict[str, np.ndarray], clim, frame) -> dict[str, np.ndarray]:
    """Band means (time, lon) -> anomalies against the reference climatology; the OLR's in ERA5's own frame."""
    out = {tag: bands[tag] - sel_doy(clim[f"clim_{tag}"], t) for tag in TAGS}
    out["olr"] = out["olr"] - sel_doy(frame, t)
    return out


def filtered(t: pd.DatetimeIndex, a: np.ndarray) -> np.ndarray:
    """(time, lon) anomaly minus its mean over the WINDOW days ending on each day; NaN until a full window. t is
    daily without gaps (checked by the callers)."""
    return a - pd.DataFrame(a, index=t).rolling(WINDOW, min_periods=WINDOW).mean().values


def project(f: dict[str, np.ndarray], clim, eofs) -> dict[str, np.ndarray]:
    """Filtered anomalies -> the three-field and the wind-only RMM1/RMM2 (time,)."""
    z = {tag: f[tag] / clim.attrs[f"std_{tag}"] for tag in TAGS}
    out = {}
    for m in (1, 2):
        wind = sum(z[tag] @ eofs[f"eof_{tag}"].sel(mode=m).values for tag in ("u850", "u200"))
        olr = z["olr"] @ eofs["eof_olr"].sel(mode=m).values
        out[f"rmm{m}"] = (olr + wind) / float(eofs["pc_std"].sel(mode=m))
        out[f"rmm{m}_wind"] = wind / float(eofs["pc_wind_std"].sel(mode=m))
    return out


def index(t: pd.DatetimeIndex, bands: dict[str, np.ndarray], ref=None) -> dict[str, np.ndarray]:
    """Band means of consecutive days -> RMM (NaN for the first WINDOW - 1 days)."""
    if len(t) > 1 and not (np.diff(t.values) == np.timedelta64(1, "D")).all():
        raise SystemExit("the band means are not on consecutive days")
    clim, eofs, frame = ref or reference()
    a = anomalies(t, bands, clim, frame)
    return project({tag: filtered(t, a[tag]) for tag in TAGS}, clim, eofs)


# ── reading the series ────────────────────────────────────────────────────────────────────────────────────────
def load(path: Path = OUT) -> pd.DataFrame:
    """The daily series: rmm1, rmm2 (three-field), rmm1_wind, rmm2_wind, indexed by date."""
    with xr.open_dataset(path) as d:
        return pd.DataFrame(
            {k: d[k].values.astype(float) for k in ("rmm1", "rmm2", "rmm1_wind", "rmm2_wind")},
            index=pd.DatetimeIndex(d.time.values).normalize(),
        )


def on_dates(dates, path: Path = OUT, wind: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """RMM1 and RMM2 on `dates` (any shape of datetime64 / Timestamps), NaN where the series has no value."""
    s = load(path)
    d = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[D]").ravel())
    k = ("rmm1_wind", "rmm2_wind") if wind else ("rmm1", "rmm2")
    shape = np.shape(np.asarray(dates, dtype="datetime64[D]"))
    return tuple(s[c].reindex(d).values.reshape(shape) for c in k)


# ── fetching band means ───────────────────────────────────────────────────────────────────────────────────────
def _band_rows(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.abs(lat) <= HALF + 1e-6
    return rows, np.cos(np.deg2rad(lat[rows]))


def fetch_wb2(t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """u850/u200 band means from WeatherBench2 (1.5 deg, 6-hourly), a month at a time, on LON."""
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=list(LEVELS))
    rows, w = _band_rows(u.latitude.values)
    u = u.isel(latitude=np.flatnonzero(rows))
    lon = u.longitude.values % 360
    days, out = [], {lev: [] for lev in LEVELS}
    for m0 in pd.date_range(t0.replace(day=1), t1, freq="MS"):
        end = min(m0 + pd.offsets.MonthEnd(0), t1) + pd.Timedelta(hours=23)
        part = u.sel(time=slice(max(m0, t0), end))
        tt = pd.DatetimeIndex(part.time.values)
        part = part.isel(time=np.flatnonzero(np.isin(tt.hour, HOURS)))
        v = part.transpose("time", "level", "latitude", "longitude").load()  # by name: the store is lon-major
        b = np.einsum("tlyx,y->tlx", v.values.astype(float), w / w.sum())  # (time, level, lon)
        day = pd.DatetimeIndex(v.time.values).normalize()
        for d in day.unique():
            k = day == d
            if k.sum() != len(HOURS):
                raise SystemExit(f"WeatherBench2 has {int(k.sum())} of the {len(HOURS)} hours on {d:%Y-%m-%d}")
            days.append(d)
            for i, lev in enumerate(v.level.values):
                out[int(lev)].append(_to_lon(b[k, i].mean(0), lon))
        print(f"  {m0:%Y-%m}: {len(day.unique())} days", flush=True)
    return pd.DatetimeIndex(days), {f"u{lev}": np.array(out[lev], "float32") for lev in LEVELS}


def _to_lon(v: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A band mean on the source longitudes -> LON, linear and periodic."""
    o = np.argsort(lon)
    x, y = lon[o], v[o]
    return np.interp(LON, np.r_[x, x[0] + 360.0], np.r_[y, y[0]])


def open_arco_u() -> xr.DataArray:
    a = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})["u_component_of_wind"]
    if not np.allclose(a.longitude.values, np.arange(1440) * 0.25):
        raise SystemExit("unexpected ARCO-ERA5 longitudes")
    rows, _ = _band_rows(a.latitude.values)
    return a.isel(latitude=np.flatnonzero(rows)).sel(level=list(LEVELS))


def arco_hour(u: xr.DataArray, t: pd.Timestamp) -> np.ndarray | None:
    """(level, lon) band means of one ARCO-ERA5 hour on LON (each a centred 2.5 deg cell), None where ARCO holds no
    value yet. One read is one chunk of all 37 levels; transient storage errors are retried."""
    if t > pd.Timestamp(u["time"].values[-1]):
        return None
    for attempt in range(4):
        try:
            v = u.sel(time=t).transpose("level", "latitude", "longitude").values.astype(float)
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    if np.isnan(v).all():
        return None  # ARCO pads the hours it has not published
    if not np.isfinite(v).all() or np.abs(v).max() > 150:
        raise ValueError(f"{t:%Y-%m-%d %H}Z: missing or implausible ARCO-ERA5 u")
    _, w = _band_rows(u.latitude.values)
    b = np.einsum("lyx,y->lx", v, w / w.sum())
    return (b[:, CELL] * CELL_W).sum(-1)


def fetch_arco(days: pd.DatetimeIndex, workers: int = 8) -> dict[pd.Timestamp, np.ndarray]:
    """u850/u200 band means (level, lon) from ARCO-ERA5 for each of `days` it holds all four hours of."""
    u = open_arco_u()

    def one(day):
        hrs = [arco_hour(u, day + pd.Timedelta(hours=h)) for h in HOURS]
        return None if any(x is None for x in hrs) else np.mean(hrs, axis=0)

    return _pool(one, days, workers, "u")


def fetch_olr(days: pd.DatetimeIndex, workers: int = 16) -> dict[pd.Timestamp, np.ndarray]:
    """The OLR band mean (lon,) of the RMM channel (build_olr_baseline.day_bands, `rmm`) for each of `days` ARCO-ERA5
    holds all four hours of."""
    B = olr_baseline()
    flux = B.open_flux()

    def one(day):
        try:
            b = B.day_bands(flux, day)
        except ValueError as e:  # an hour of the day missing or not published yet
            print(f"  {e}", flush=True)
            return None
        return None if b is None else b["rmm"]

    return _pool(one, days, workers, "OLR")


def _pool(one, days, workers, what) -> dict:
    t0 = time.time()
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (day, r) in enumerate(zip(days, ex.map(one, days)), 1):
            if r is not None:
                out[day] = r
            if i % 250 == 0:
                print(f"  {what}: {i}/{len(days)} days, {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def cmd_fetch(a) -> int:
    """One part's band means -> npz (time, lon, u850, u200 and, unless --no-olr, olr). A day a source lacks is left out
    and listed; the build interpolates short holes and stops at long ones."""
    t0 = pd.Timestamp(a.start)
    t1 = pd.Timestamp(a.end) if a.end else pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    days = pd.date_range(t0, t1)
    if a.source == "wb2":
        if t1 > WB2_LAST:
            raise SystemExit(f"WeatherBench2 parts end by {WB2_LAST:%Y-%m-%d}")
        tw, bw = fetch_wb2(t0, t1)
        u = {d: np.stack([bw["u850"][i], bw["u200"][i]]) for i, d in enumerate(tw)}
    else:
        u = fetch_arco(days, a.workers)
    o = fetch_olr(days) if a.olr else None
    have = [d for d in days if d in u and (o is None or d in o)]
    if not have:
        raise SystemExit(f"no day of {t0:%Y-%m-%d}..{t1:%Y-%m-%d} has every field")
    gaps = [d for d in days if have[0] <= d <= have[-1] and d not in have]
    if gaps:
        print(f"  left out (a field missing): {', '.join(f'{d:%Y-%m-%d}' for d in gaps)}", flush=True)
    out = {"u850": np.array([u[d][0] for d in have], "float32"), "u200": np.array([u[d][1] for d in have], "float32")}
    if o is not None:
        out["olr"] = np.array([o[d] for d in have], "float32")
    np.savez_compressed(a.out, time=np.array(have, dtype="datetime64[D]"), lon=LON, **out)
    print(f"wrote {a.out}: {len(have)} days, {have[0]:%Y-%m-%d} -> {have[-1]:%Y-%m-%d}", flush=True)
    return 0


def plan() -> list[dict]:
    """The build's parts: five-year WeatherBench2 spans to 2022, a year at a time from ARCO-ERA5 after it (the last to
    ARCO's newest day), and the seam check: ARCO-ERA5 winds over the last two months WeatherBench2 covers."""
    parts = [
        {"name": f"{y}", "start": f"{y}-01-01", "end": f"{min(y + 4, WB2_LAST.year)}-12-31", "source": "wb2", "olr": 1}
        for y in range(FIRST.year, WB2_LAST.year + 1, 5)
    ]
    this = pd.Timestamp.now("UTC").year
    parts += [
        {"name": f"{y}", "start": f"{y}-01-01", "end": f"{y}-12-31" if y < this else "", "source": "arco", "olr": 1}
        for y in range(WB2_LAST.year + 1, this + 1)
    ]
    parts.append({"name": "seam", "start": "2022-11-01", "end": f"{WB2_LAST:%Y-%m-%d}", "source": "arco", "olr": 0})
    return parts


# ── building and extending the series ─────────────────────────────────────────────────────────────────────────
def _load(paths, keys) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    zs = [np.load(p) for p in paths]
    if not all(np.allclose(z["lon"], LON) for z in zs):
        raise SystemExit("a part is not on the reference longitudes")
    t = pd.DatetimeIndex(np.concatenate([z["time"] for z in zs]).astype("datetime64[ns]"))
    o = np.argsort(t.values, kind="stable")
    if t[o].has_duplicates:
        raise SystemExit("the parts overlap")
    return t[o], {k: np.concatenate([z[k] for z in zs])[o].astype(float) for k in keys}


def fill(t: pd.DatetimeIndex, bands: dict[str, np.ndarray], days: pd.DatetimeIndex):
    """The band means on every day of `days` (t covers its first and last): a hole of at most MAX_GAP days is
    interpolated linearly in time, a longer one stops. -> bands on days, the interpolated days"""
    i = t.get_indexer(days)
    holes = days[i < 0]
    if not len(holes):
        return {k: v[i] for k, v in bands.items()}, holes
    runs = np.split(holes, np.flatnonzero(np.diff(holes.values) != np.timedelta64(1, "D")) + 1)
    if max(len(r) for r in runs) > MAX_GAP:
        long = [f"{r[0]:%Y-%m-%d}..{r[-1]:%Y-%m-%d}" for r in runs if len(r) > MAX_GAP]
        raise SystemExit(f"ERA5 holes longer than {MAX_GAP} days: {', '.join(long)}")
    x, xi = t.values.astype("datetime64[D]").astype(float), days.values.astype("datetime64[D]").astype(float)
    out = {k: np.stack([np.interp(xi, x, v[:, j]) for j in range(v.shape[1])], axis=1) for k, v in bands.items()}
    print(f"  interpolated across {len(holes)} day(s) ERA5 lacks: {', '.join(f'{d:%Y-%m-%d}' for d in holes)}")
    return out, holes


def write(t: pd.DatetimeIndex, r: dict[str, np.ndarray], tail_t, tail: dict[str, np.ndarray], attrs: dict) -> None:
    """The series from its first full filter window, and the band means of the last TAIL days."""
    ok = np.isfinite(r["rmm1"])
    first = int(np.flatnonzero(ok)[0])
    enc = {"zlib": True, "complevel": 5}
    ds = xr.Dataset(
        {k: ("time", r[k][first:].astype("float32")) for k in ("rmm1", "rmm2", "rmm1_wind", "rmm2_wind")}
        | {f"{tag}_tail": (("tail_time", "longitude"), tail[tag].astype("float32")) for tag in TAGS},
        coords={"time": t[first:], "tail_time": tail_t, "longitude": LON},
        attrs=attrs,
    )
    for k in ("rmm1", "rmm2"):
        ds[k].attrs["long_name"] = f"{k.upper()}, three-field (OLR + U850 + U200), pc_std-normalised"
        ds[f"{k}_wind"].attrs["long_name"] = f"{k.upper()}, wind-only (U850 + U200), pc_wind_std-normalised"
    for tag in TAGS:
        ds[f"{tag}_tail"].attrs["long_name"] = f"{tag} 15S-15N band mean of the last {TAIL} days (for update)"
    tmp = OUT.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=dict.fromkeys(ds.data_vars, enc))
    tmp.replace(OUT)
    print(f"wrote {OUT.name}: {ds.time.size} days, {t[first]:%Y-%m-%d} -> {t[-1]:%Y-%m-%d}", flush=True)


ATTRS = {
    "title": "Observed RMM (Wheeler & Hendon 2004) from ERA5",
    "method": "15S-15N band means of OLR, u850, u200 (daily means of 00/06/12/18 UTC) on the 2.5 deg reference "
    "longitudes; anomalies against climatology.nc (OLR also minus the ERA5-to-NOAA frame, olr_channel.frame_offset); "
    "minus the mean of the 120 days ending on the day; divided by the reference standard deviations and projected onto "
    "eofs.nc (rmm1/rmm2: three-field, pc_std; rmm1_wind/rmm2_wind: wind-only, pc_wind_std). ENSO not removed.",
    "source": "ERA5 (Hersbach et al. 2020): u from WeatherBench2 (1.5 deg) to 2022 and ARCO-ERA5 (0.25 deg) after; "
    "OLR (-mean_top_net_long_wave_radiation_flux) from ARCO-ERA5",
    "licence": "contains modified Copernicus Climate Change Service information; ERA5 CC BY 4.0",
    "builder": "scripts/mjo/src/rmm_era5.py",
}


def _span(t: pd.DatetimeIndex, y0: int, y1: int) -> np.ndarray:
    return (t.year >= y0) & (t.year <= y1)


def build_checks(t, bands, r, ref, seam) -> list[str]:
    """What the step summary reports about the record: its normalisation, the WeatherBench2-to-ARCO seam, and how
    much a wind frame (ERA5's own 1979-2001 wind climatology in place of the NCEP one) would change the index."""
    design, fit = olr_baseline().design, olr_baseline().fit

    clim = ref[0]
    lines = []
    for y0, y1 in ((1979, 2001), (2003, 2024), (1979, t.year[-1])):
        k = _span(t, y0, y1) & np.isfinite(r["rmm1"])
        lines.append(
            f"{y0}-{y1}: sd RMM1/RMM2 {r['rmm1'][k].std():.3f}/{r['rmm2'][k].std():.3f}, wind-only "
            f"{r['rmm1_wind'][k].std():.3f}/{r['rmm2_wind'][k].std():.3f}; r(three-field, wind-only) "
            f"{np.corrcoef(r['rmm1'][k], r['rmm1_wind'][k])[0, 1]:.3f}/{np.corrcoef(r['rmm2'][k], r['rmm2_wind'][k])[0, 1]:.3f}; "
            f"amplitude >= 1 on {100 * (np.hypot(r['rmm1'][k], r['rmm2'][k]) >= 1).mean():.0f} % of days"
        )
    if seam is not None:
        ts, us = seam
        i, j = t.get_indexer(ts), np.arange(len(ts))
        j, i = j[i >= 0], i[i >= 0]
        for lev in LEVELS:
            d = us[f"u{lev}"][j] - bands[f"u{lev}"][i]
            lines.append(
                f"seam, ARCO minus WeatherBench2 u{lev} on {len(i)} days to {ts[j[-1]]:%Y-%m-%d}: mean {d.mean():+.3f}, "
                f"rms {np.sqrt((d**2).mean()):.3f} m/s (std_u{lev} {clim.attrs[f'std_u{lev}']:.2f})"
            )
        alt = {k: v.copy() for k, v in bands.items()}
        for lev in LEVELS:
            alt[f"u{lev}"][i] = us[f"u{lev}"][j]
        ra = index(t, alt, ref)
        d = np.hypot(ra["rmm1"][i] - r["rmm1"][i], ra["rmm2"][i] - r["rmm2"][i])
        lines.append(
            f"seam: with ARCO's winds on those days the index moves by {d.mean():.3f} on average (at most {d.max():.3f})"
        )
    base = _span(t, 1979, 2001)
    alt = {k: v.copy() for k, v in bands.items()}
    for lev in LEVELS:
        tag = f"u{lev}"
        days = np.arange(1, 367)
        own = design(days - 1, int(clim.attrs["n_harmonics"]), 365.0) @ fit(
            noleap_doy(t[base].values) - 1, bands[tag][base], int(clim.attrs["n_harmonics"]), 365.0
        )
        off = own - clim[f"clim_{tag}"].transpose("dayofyear", "longitude").values
        alt[tag] = bands[tag] - sel_doy(clim[f"clim_{tag}"].copy(data=off), t)
        lines.append(
            f"wind frame, ERA5 minus NCEP {tag} 1979-2001 climatology: mean {off.mean():+.2f}, by longitude "
            f"{off.mean(0).min():+.2f}..{off.mean(0).max():+.2f}, seasonal range up to {np.ptp(off, axis=0).max():.2f} m/s"
        )
    ra = index(t, alt, ref)
    k = np.isfinite(ra["rmm1"]) & np.isfinite(r["rmm1"])
    d = np.hypot(ra["rmm1"][k] - r["rmm1"][k], ra["rmm2"][k] - r["rmm2"][k])
    lines.append(
        f"wind frame: the index with ERA5's own wind climatology differs by {d.mean():.3f} on average "
        f"(95th percentile {np.percentile(d, 95):.3f}); the forecasts' definition (NCEP climatology) is kept"
    )
    return lines


def cmd_build(a) -> int:
    t0, b0 = _load(a.parts, TAGS)
    if t0[0] != FIRST:
        raise SystemExit(f"the parts start {t0[0]:%Y-%m-%d}, not {FIRST:%Y-%m-%d}")
    t = pd.date_range(FIRST, t0[-1])
    bands, holes = fill(t0, b0, t)
    ref = reference()
    r = index(t, bands, ref)
    attrs = ATTRS | {
        "built": f"{datetime.now(UTC):%Y-%m-%d %H:%MZ}",
        "last_day": f"{t[-1]:%Y-%m-%d}",
        "interpolated": ", ".join(f"{d:%Y-%m-%d}" for d in holes),
    }
    write(t, r, t[-TAIL:], {tag: bands[tag][-TAIL:] for tag in TAGS}, attrs)
    seam = _load([a.seam], ("u850", "u200")) if a.seam else None
    lines = build_checks(t, bands, r, ref, seam)
    if len(holes):
        lines.append(f"interpolated across {len(holes)} day(s) ERA5 lacks: {attrs['interpolated']}")
    print("\n".join(lines), flush=True)
    if a.summary:
        Path(a.summary).write_text("\n".join(lines) + "\n")
    if a.bands:
        np.savez_compressed(
            a.bands,
            time=t.values.astype("datetime64[D]"),
            lon=LON,
            **{k: v.astype("float32") for k, v in bands.items()},
        )
        print(f"wrote {a.bands} ({Path(a.bands).stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


def cmd_update(a) -> int:
    """Append the days ARCO-ERA5 has added since the file's last day, at most UPDATE_MAX: up to the newest day it holds
    whole, a hole of at most MAX_GAP days before that interpolated (and recorded), a longer one ending the update."""
    if not OUT.exists():
        raise SystemExit(f"no {OUT.name} yet: build it with rmm-era5.yml")
    with xr.open_dataset(OUT) as f:
        d = f.load()
    t_tail = pd.DatetimeIndex(d.tail_time.values)
    last = pd.DatetimeIndex(d.time.values)[-1]
    if t_tail[-1] != last:
        raise SystemExit(f"{OUT.name}: the tail ends {t_tail[-1]:%Y-%m-%d}, the series {last:%Y-%m-%d}")
    t1 = min(
        pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1),
        last + pd.Timedelta(days=UPDATE_MAX),
    )
    days = pd.date_range(last + pd.Timedelta(days=1), t1)
    u, o = fetch_arco(days, 4), fetch_olr(days, 8)
    have = [x for x in days if x in u and x in o]
    for k, x in enumerate(have):  # stop before a hole longer than MAX_GAP
        prev = have[k - 1] if k else last
        if (x - prev).days - 1 > MAX_GAP:
            print(f"  ERA5 lacks {(x - prev).days - 1} days after {prev:%Y-%m-%d}: appending up to it only")
            have = have[:k]
            break
    if not have:
        print(f"{OUT.name} reaches {last:%Y-%m-%d}; ARCO-ERA5 holds no whole day after it yet")
        return 0
    add = {"olr": [o[x] for x in have], "u850": [u[x][0] for x in have], "u200": [u[x][1] for x in have]}
    tb = t_tail.append(pd.DatetimeIndex(have))
    bb = {tag: np.vstack([d[f"{tag}_tail"].values.astype(float), np.array(add[tag])]) for tag in TAGS}
    t = pd.date_range(t_tail[0], have[-1])
    bands, holes = fill(tb, bb, t)
    rn = index(t, bands)
    n = len(t) - len(t_tail)
    new = {k: rn[k][-n:] for k in rn}
    if not np.isfinite(new["rmm1"]).all():
        raise SystemExit("an appended day has no full 120-day window")
    tt = pd.DatetimeIndex(d.time.values).append(t[-n:])
    r = {k: np.r_[d[k].values.astype(float), new[k]] for k in new}
    filled = [s for s in str(d.attrs.get("interpolated", "")).split(", ") if s] + [f"{x:%Y-%m-%d}" for x in holes]
    attrs = dict(d.attrs) | {
        "updated": f"{datetime.now(UTC):%Y-%m-%d %H:%MZ}",
        "last_day": f"{tt[-1]:%Y-%m-%d}",
        "interpolated": ", ".join(filled),
    }
    write(tt, r, t[-TAIL:], {tag: bands[tag][-TAIL:] for tag in TAGS}, attrs)
    amp = np.hypot(new["rmm1"][-1], new["rmm2"][-1])
    print(f"appended {n} day(s) to {tt[-1]:%Y-%m-%d}; amplitude {amp:.2f} on the last", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="the build's parts, as JSON")
    f = sub.add_parser("fetch", help="one part's band means")
    f.add_argument("--start", required=True)
    f.add_argument("--end", default="", help="last day (empty: yesterday, as far as ARCO-ERA5 reaches)")
    f.add_argument("--source", choices=("wb2", "arco"), required=True, help="of the winds (the OLR is ARCO-ERA5's)")
    f.add_argument("--no-olr", dest="olr", action="store_false", help="winds only (the seam check)")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build", help=f"the whole record -> {OUT.name}")
    b.add_argument("--parts", type=Path, nargs="+", required=True, help="fetch outputs covering 1979 to the end")
    b.add_argument("--seam", type=Path, help="a --no-olr fetch output from ARCO-ERA5 over days WeatherBench2 covers")
    b.add_argument("--bands", type=Path, help=f"also write the record's band means here ({BANDS})")
    b.add_argument("--summary", type=Path, help="also write the checks here")
    sub.add_parser("update", help="append the new ARCO-ERA5 days")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan()))
        return 0
    return {"fetch": cmd_fetch, "build": cmd_build, "update": cmd_update}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
