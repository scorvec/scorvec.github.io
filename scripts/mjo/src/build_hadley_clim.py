#!/usr/bin/env python3
"""Day-of-year climatology of the Hadley cell metrics: the metrics of ERA5's 7-DAY-MEAN circulation, 1991-2020.

Every series this is scored against reads its cells off a 7-day-mean field: the intensity and edges of the cells in
the zonal-mean v averaged over the week (overturn_hist.week, overturn_fc.windows). The climatology must be the same
statistic of ERA5, or the comparison is biased. Its first version averaged the metrics of three single 12Z fields
within +-7 days; measured against the 7-day-mean field on the AIFS analyses (2026-10-16, 43 days) that statistic read
the southern cell 1.3 x 10^10 kg/s stronger (t = -3.2) and the northern edge 8 deg further equatorward (t = -11) -
the peak of an average is not the average of the peaks. So here every day of 1991-2020 gets the centred 7-day mean of
ERA5's daily-mean zonal-mean v (WeatherBench2: 6-hourly, 1.5 deg, conservative) and the metrics of that field; a day of
year pools the 30 years and the days within +-POOL of it.

The metrics of the harmonic Psi climatology (mmsf_clim_coeffs.nc) are not a substitute: that field is the 30-YEAR MEAN
circulation, whose cells are smeared because their edges wander ~10 deg between years.

Every comparison is made on this grid: a series or forecast averages its zonal-mean v into the same 1.5 deg latitude
bands (band_v) before its cells are read (metrics), so resolution does not enter either (at 0.25 deg the cells read
~0.08 x 10^10 kg/s stronger).

    python src/build_hadley_clim.py fetch --years 1991 1995 --out part.npz   # daily-mean vbar from WeatherBench2
    python src/build_hadley_clim.py build part*.npz                          # -> data/reference/hadley_clim.nc
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
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
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


def fetch(y0: int, y1: int) -> dict:
    """Daily means of ERA5's zonal-mean v for [y0-01-01 - 3 d, y1-12-31 + 3 d] (the centred windows reach that far)."""
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    t0, t1 = (
        pd.Timestamp(f"{y0}-01-01") - pd.Timedelta(days=WIN // 2),
        pd.Timestamp(f"{y1}-12-31 18:00") + pd.Timedelta(days=WIN // 2),
    )
    v = ds["v_component_of_wind"].sel(time=slice(t0, t1), level=LEVELS)
    days, rows = [], []
    for m0 in pd.date_range(t0.normalize(), t1.normalize(), freq="MS").union([t0.normalize()]):
        m1 = min(m0 + pd.offsets.MonthBegin(1), t1 + pd.Timedelta(hours=6))
        part = v.sel(time=slice(m0, m1 - pd.Timedelta(hours=1)))
        if not part.time.size:
            continue
        zm = (
            part.mean("longitude").transpose("time", "level", "latitude").load()
        )  # transpose by NAME (dim order differs)
        daily = zm.resample(time="1D").mean()
        daily = daily.sortby("latitude", ascending=False)  # north to south, as LAT15
        assert np.allclose(daily.latitude.values, LAT15, atol=1e-3), "WeatherBench2 rows are not the 1.5 deg grid"
        days += list(pd.DatetimeIndex(daily.time.values))
        rows.append(daily.values.astype(np.float32))
        print(f"  {m0:%Y-%m}: {daily.time.size} days", flush=True)
    return {"date": np.array(days, "datetime64[D]"), "vbar": np.concatenate(rows)}


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
            "source": "ERA5 1991-2020, WeatherBench2 6-hourly 1.5 deg conservative, daily means",
            "basis": "metrics of each day's CENTRED 7-day-mean zonal-mean v (the statistic every scored series uses), "
            f"pooled over the 30 years and +-{POOL} days of year",
            "grid": "1.5 deg latitude bands (build_hadley_clim.band_v); intensity 10^10 kg/s, latitudes degrees north",
            "window_days": WIN,
            "pool_days": POOL,
        },
    )
    out["n_pooled"] = ("doy", counts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs=2, required=True)
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fetch":
        np.savez_compressed(a.out, **fetch(*a.years))
        return 0
    out = build(a.parts)
    REF.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
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
