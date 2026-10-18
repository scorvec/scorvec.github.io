#!/usr/bin/env python3
"""Day-of-year climatologies of the Hadley cells from ERA5 1991-2020 at 00 UTC: the metrics of the 7-DAY-MEAN
circulation (hadley_clim.nc) and the harmonic Psi field the anomaly maps subtract (mmsf_clim_coeffs.nc).

Every series this is scored against reads its cells off a 7-day-mean field: the intensity and edges of the cells in
the zonal-mean v averaged over the week (overturn_hist.week, overturn_fc.windows). The climatology must be the same
statistic of ERA5, or the comparison is biased. Its first version averaged the metrics of three single 12Z fields
within +-7 days; measured against the 7-day-mean field on the AIFS analyses (2026-10-16, 43 days) that statistic read
the southern cell 1.3 x 10^10 kg/s stronger (t = -3.2) and the northern edge 8 deg further equatorward (t = -11) -
the peak of an average is not the average of the peaks. So here every day of 1991-2020 gets the centred 7-day mean of
ERA5's zonal-mean v and the metrics of that field; a day of year pools the 30 years and the days within +-POOL of it.

ONE TIME OF DAY, 00 UTC, everywhere. The zonal-mean circulation has a diurnal cycle: on WeatherBench2, 7-day means of
the 00 and 12 UTC fields differ from those of all four synoptic times by up to 2.9 deg in the northern edge (SON) and
0.7 x 10^10 kg/s in the southern cell (JJA), well outside their standard errors. The forecasts are 00 UTC fields (daily
steps of the 00Z runs), the drift is measured at 00 UTC, so the analyses (overturn_hist), the climatologies and the
El Nino events (hadley_enso) are 00 UTC samples too.

The harmonic map climatology is fitted on the same daily samples (mean + annual + semiannual per level and latitude).
Its metrics are not a substitute for hadley_clim.nc: that field is the 30-YEAR MEAN circulation, whose cells are smeared
because their edges wander ~10 deg between years.

Both are on the 1.5 deg latitude bands of WeatherBench2 (6-hourly, conservative), and every map and metric compared with
them is read on the same bands (band_v; at 0.25 deg the cells read ~0.08 x 10^10 kg/s stronger).

    python src/build_hadley_clim.py fetch --years 1991 1995 --out part.npz   # 00 UTC zonal-mean v from WeatherBench2
    python src/build_hadley_clim.py fetch --start 2023-06-28 --end 2024-04-03 --source arco --out p.npz   # after 2022
    python src/build_hadley_clim.py build part*.npz        # -> data/reference/hadley_clim.nc + mmsf_clim_coeffs.nc
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from mmsf import LEVELS, hadley_cells, streamfunction
from doycal import noleap_doy

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "hadley_clim.nc"
COEFFS = REF / "mmsf_clim_coeffs.nc"  # mmsf.CLIM
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"  # 0.25 deg, to ~5 days ago
LAT15 = np.linspace(90.0, -90.0, 121)  # WeatherBench2's rows, north to south as the analysis grid
P_HPA = np.asarray(LEVELS, float)
WIN = 7  # days in the window, centred
POOL = 3  # +- days of year pooled with each day (every year)

# nh/sh edge lists come back south->north, so the poleward edge is the far one
METRICS = ("nh_psi", "nh_core", "nh_edge_eq", "nh_edge_pole", "sh_psi", "sh_core", "sh_edge_eq", "sh_edge_pole")
PCTS = (10, 25, 50, 75, 90)
# Below this the cell is not really there (the summer cell collapses for whole
# weeks). Intensity stays reportable — it is a real measurement of a weak cell —
# but an "edge latitude" of a cell that barely exists is noise, and unmasked it
# both spikes the observed series and blows the climatological band open. Gated
# identically here and in the plotted series so the two stay comparable.
MIN_PSI = 1.0


def sample_metrics(psi, p_hpa, lat) -> dict:
    """One Ψ field → the eight scalars, NaN where a cell/edge is unresolved."""
    c = hadley_cells(psi, p_hpa, lat)
    out = dict.fromkeys(METRICS, np.nan)
    nh, sh = c.get("nh"), c.get("sh")
    if nh:
        out["nh_psi"], out["nh_core"] = nh["psi"], nh["core_lat"]
        if abs(nh["psi"]) >= MIN_PSI:
            lo, hi = nh["edges"]  # south, north
            out["nh_edge_eq"] = np.nan if lo is None else lo
            out["nh_edge_pole"] = np.nan if hi is None else hi
    if sh:
        out["sh_psi"], out["sh_core"] = sh["psi"], sh["core_lat"]
        if abs(sh["psi"]) >= MIN_PSI:
            lo, hi = sh["edges"]  # south = poleward for the SH cell
            out["sh_edge_pole"] = np.nan if lo is None else lo
            out["sh_edge_eq"] = np.nan if hi is None else hi
    return out


@cache
def _band_matrix(lat_src: tuple) -> np.ndarray:
    """(121, n) weights: each 1.5 deg band's area-weighted mean of the source rows (cells of half the source spacing)."""
    lat = np.asarray(lat_src, float)
    half = 0.5 * abs(lat[1] - lat[0])
    edges = np.clip(np.r_[90.0, 0.5 * (LAT15[:-1] + LAT15[1:]), -90.0], -90.0, 90.0)
    W = np.zeros((LAT15.size, lat.size))
    for j in range(LAT15.size):
        hi, lo = np.minimum(edges[j], lat + half), np.maximum(edges[j + 1], lat - half)
        W[j] = np.where(hi > lo, np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)), 0.0)
    return W / W.sum(1, keepdims=True)


def band_v(vbar: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Zonal-mean v (..., lat) on any regular grid -> the climatology's 1.5 deg bands (..., 121)."""
    if lat.size == LAT15.size and np.allclose(lat, LAT15):
        return vbar
    return vbar @ _band_matrix(tuple(np.round(lat, 6))).T


def metrics(vbar: np.ndarray, lat: np.ndarray) -> dict:
    """The cell metrics of one (lev, lat) zonal-mean v field, read on the climatology's grid."""
    psi = streamfunction(band_v(vbar, lat), P_HPA * 100.0, LAT15)
    return sample_metrics(psi, P_HPA, LAT15)


def fetch_wb2(t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """ERA5's zonal-mean v at 00 UTC on LAT15 for the days t0..t1, from WeatherBench2."""
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    v = ds["v_component_of_wind"].sel(time=slice(t0, t1), level=LEVELS)
    v = v.isel(time=np.flatnonzero(pd.DatetimeIndex(v.time.values).hour == 0))
    days, rows = [], []
    for m0 in pd.date_range(t0, t1, freq="MS").union([t0]):
        part = v.sel(
            time=slice(m0, min(m0 + pd.offsets.MonthBegin(1), t1 + pd.Timedelta(days=1)) - pd.Timedelta(hours=1))
        )
        if not part.time.size:
            continue
        zm = part.mean("longitude").transpose("time", "level", "latitude").load()  # by NAME: the store is lon-major
        zm = zm.sortby("latitude", ascending=False)  # north to south, as LAT15
        assert np.allclose(zm.latitude.values, LAT15, atol=1e-3), "WeatherBench2 rows are not the 1.5 deg grid"
        days += list(pd.DatetimeIndex(zm.time.values).normalize())
        rows.append(zm.values.astype(np.float32))
        print(f"  {m0:%Y-%m}: {zm.time.size} days", flush=True)
    return {"date": np.array(days, "datetime64[D]"), "vbar": np.concatenate(rows)}


def fetch_arco(t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """The same from ARCO-ERA5 (0.25 deg, after WeatherBench2 ends in January 2023): each day's 00 UTC zonal mean,
    averaged into LAT15's bands, read straight from the store (one chunk holds all 37 levels, ~120 MB) and kept
    nowhere (era5_store would cache ~30 MB per time on disk)."""
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    days, rows = [], []
    for d in pd.date_range(t0, t1):
        v = ds["v_component_of_wind"].sel(time=d, level=LEVELS).load().sortby("level")
        if bool(np.isnan(v.values).all()):  # ARCO pads days it has not published
            continue
        days.append(d)
        rows.append(
            band_v(v.transpose("level", "latitude", "longitude").values.mean(-1), v.latitude.values).astype(np.float32)
        )
        if d.day == 1:
            print(f"  {d:%Y-%m}", flush=True)
    return {"date": np.array(days, "datetime64[D]"), "vbar": np.asarray(rows)}


def fetch(y0: int, y1: int) -> dict:
    """The climatology's years, with the margins its centred windows reach into."""
    half = pd.Timedelta(days=WIN // 2)
    return fetch_wb2(pd.Timestamp(f"{y0}-01-01") - half, pd.Timestamp(f"{y1}-12-31") + half)


def build(paths) -> xr.Dataset:
    acc_d, acc_v = [], []
    for p in paths:
        with np.load(p) as f:
            acc_d.append(f["date"])
            acc_v.append(f["vbar"])
    date = np.concatenate(acc_d)
    vbar = np.concatenate(acc_v)
    date, first = np.unique(date, return_index=True)
    vbar = vbar[first]
    days = pd.DatetimeIndex(date)
    in_base = (days.year >= 1991) & (days.year <= 2020)
    psi = np.stack([streamfunction(vbar[i], P_HPA * 100.0, LAT15) for i in np.flatnonzero(in_base)])
    w = 2 * np.pi * noleap_doy(days[in_base]) / 365.25  # common-year day: 1 March = 60 every year
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coef = np.linalg.lstsq(X, psi.reshape(len(psi), -1), rcond=None)[0].reshape(5, len(LEVELS), LAT15.size)
    harmonic = xr.DataArray(
        coef,
        dims=("coef", "level", "latitude"),
        coords={"coef": np.arange(5), "level": LEVELS, "latitude": LAT15},
        attrs={
            "units": "10^10 kg/s",
            "note": "ERA5 1991-2020 00 UTC Psi harmonic clim (WeatherBench2 1.5 deg, every day)",
        },
    )
    pos = {d: i for i, d in enumerate(days)}
    rows, idx = [], []
    for d in days[(days.year >= 1991) & (days.year <= 2020)]:
        w = [pos.get(d + pd.Timedelta(days=k)) for k in range(-(WIN // 2), WIN // 2 + 1)]
        if any(i is None for i in w):
            continue
        rows.append(metrics(vbar[w].mean(0), LAT15))
        idx.append(d)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    print(f"{len(df)} centred 7-day means, {df.index[0]:%Y-%m-%d} … {df.index[-1]:%Y-%m-%d}", flush=True)
    doy = noleap_doy(df.index)  # common-year day; 29 Feb = 59.5
    stats = {m: np.full((len(PCTS) + 2, 366), np.nan) for m in METRICS}
    counts = np.zeros(366, int)
    for d in range(1, 366):  # slots 1..365 are common-year days; slot 366 repeats 31 Dec
        off = np.abs(((doy - d + 182.5) % 365) - 182.5)  # circular distance on the 365-day cycle
        band = off <= POOL
        for k in METRICS:
            v = df[k].values[band]
            v = v[np.isfinite(v)]
            if k == METRICS[0]:
                counts[d - 1] = v.size
            if v.size < 15 * (2 * POOL + 1):  # fewer than ~15 years' worth: too thin to trust
                continue
            stats[k][0, d - 1] = v.mean()
            stats[k][1, d - 1] = v.std(ddof=1)
            for j, q in enumerate(PCTS):
                stats[k][2 + j, d - 1] = np.percentile(v, q)
    for k in METRICS:
        stats[k][:, 365] = stats[k][:, 364]
    counts[365] = counts[364]
    out = xr.Dataset(
        {k: (("stat", "doy"), stats[k]) for k in METRICS},
        coords={"stat": ["mean", "sd"] + [f"p{q}" for q in PCTS], "doy": np.arange(1, 367)},
        attrs={
            "source": "ERA5 1991-2020 at 00 UTC, WeatherBench2 1.5 deg conservative",
            "basis": "metrics of each day's CENTRED 7-day-mean zonal-mean v (the statistic every scored series uses), "
            f"pooled over the 30 years and +-{POOL} days of year",
            "grid": "1.5 deg latitude bands (build_hadley_clim.band_v); intensity 10^10 kg/s, latitudes degrees north",
            "window_days": WIN,
            "pool_days": POOL,
        },
    )
    out["n_pooled"] = ("doy", counts)
    return out, harmonic


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs=2, help="whole years (the climatology)")
    f.add_argument("--start", help="or a date range, YYYY-MM-DD (an event)")
    f.add_argument("--end")
    f.add_argument("--source", choices=("wb2", "arco"), default="wb2")
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fetch":
        if a.years:
            doc = fetch(*a.years)
        else:
            doc = (fetch_arco if a.source == "arco" else fetch_wb2)(pd.Timestamp(a.start), pd.Timestamp(a.end))
        np.savez_compressed(a.out, **doc)
        return 0
    out, harmonic = build(a.parts)
    REF.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    harmonic.to_netcdf(COEFFS)
    print(f"wrote {COEFFS}")
    print(f"wrote {OUT}  (pooled n/doy: {int(out.n_pooled.min())}–{int(out.n_pooled.max())})")
    for lab, d in (("1 Feb", 32), ("1 May", 121), ("1 Aug", 213), ("1 Nov", 305)):
        r = out.sel(doy=d)
        print(
            f"  {lab}: NH Ψmax {float(r.nh_psi.sel(stat='p50')):+6.2f} "
            f"[{float(r.nh_psi.sel(stat='p10')):+5.2f},{float(r.nh_psi.sel(stat='p90')):+5.2f}]"
            f"  edge {float(r.nh_edge_pole.sel(stat='p50')):+5.1f}"
            f"  |  SH Ψmax {float(r.sh_psi.sel(stat='p50')):+6.2f} "
            f"edge {float(r.sh_edge_pole.sel(stat='p50')):+6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
