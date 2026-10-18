#!/usr/bin/env python3
"""ERA5 100 hPa eddy heat flux [v'T'], 45-75 deg, both hemispheres: the daily series and its 1991-2020 climatology.

The flux is heatflux100.py's own (heatflux100.fluxes and band_mean): the zonal-mean covariance of v and T at 100 hPa
over zonal wavenumbers 1..K_MAX (72), cos(lat)-weighted over 45-75 deg, the total and its wave-1 and wave-2 parts, the
SH sign-flipped so POSITIVE IS POLEWARD in both hemispheres. It is taken from single fields at 00 and 12 UTC, as the
forecasts are (one field per valid time) and as the analysed record is (the AIFS control's 00 and 12 UTC 0-h fields).

  to 2022  WeatherBench2 ERA5, 6-hourly, 1.5 deg (conservative from 0.25 deg): 240 longitudes carry k = 72
  2023 ->  ARCO-ERA5, hourly, 0.25 deg, to about five days ago (ERA5T). One zarr chunk holds all 37 levels of a
           variable at one hour (~150 MB), so every field costs a whole chunk

  fetch  one part of the record -> npz: the 00/12 UTC fluxes and, for `check`, WeatherBench2's flux of the
         daily-mean fields (00-18 UTC) and ARCO's flux on WeatherBench2's 1.5 deg cells
  pack   parts + the committed series -> reference/vt100_era5.nc (merged, so a later run over new days tops it up)
  clim   reference/vt100_era5.nc -> reference/vt100_clim.nc: by day of year (366-slot calendar, +/-15 days, 1991-2020)
         the mean, sd, p10 and p90 of the 00 and 12 UTC values, and the mean and sd of the trailing 40-day mean of the
         daily (00/12 UTC averaged) series, the quantity the vortex responds to
  check  parts -> printed: WeatherBench2 against ARCO on the days both hold, 0.25 deg against the 1.5 deg cells over
         the ARCO years, single fields against daily-mean fields over 1991-2020

Built in Actions (heatflux100-ref.yml).

    python build_vt100_clim.py fetch --since 1990-11-01 --until 2026-10-10 --part 0/8 --out era5_0.npz
    python build_vt100_clim.py pack parts/era5_*.npz
    python build_vt100_clim.py clim
    python build_vt100_clim.py check parts/era5_*.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heatflux100 as HF

SERIES = HERE / "reference" / "vt100_era5.nc"
OUT = HF.CLIM
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2_LAST = pd.Timestamp("2023-01-10")  # WeatherBench2's last day
SWITCH = pd.Timestamp("2023-01-01")  # the series is WeatherBench2 before this day and ARCO from it
HOURS = (0, 12)
KEYS = tuple(f"{h}_{k}" for h in ("nh", "sh") for k in ("tot", "k1", "k2"))
W15 = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])  # a 1.5 deg cell over 0.25 deg cells: five whole, two halves


def band_fluxes(v: np.ndarray, t: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(time, lat, lon) v and T -> (time, 6) in KEYS order, through heatflux100's flux and band mean."""
    da = {
        k: xr.DataArray(x, dims=("time", "latitude", "longitude"), coords={"latitude": lat})
        for k, x in (("v", v), ("t", t))
    }
    _, f = HF.fluxes(da["v"], da["t"])
    return np.stack([HF.band_mean(lat, f[k], north) for north in (True, False) for k in ("tot", "k1", "k2")], axis=-1)


def screen(v: np.ndarray, t: np.ndarray, what: str) -> None:
    """Refuse a field that is not a 100 hPa wind and temperature: non-finite values, a zero field, out-of-range values."""
    bad = []
    if not (np.isfinite(v).all() and np.isfinite(t).all()):
        bad.append("non-finite values")
    elif (np.abs(v).max(axis=(-2, -1)) == 0).any():
        bad.append("an all-zero wind field")
    elif np.abs(v).max() > 150 or t.min() < 170 or t.max() > 260:
        bad.append(f"v up to {np.abs(v).max():.0f} m/s, T {t.min():.0f}..{t.max():.0f} K")
    if bad:
        raise RuntimeError(f"{what}: {bad[0]}")


def cells15(f: np.ndarray, lat: np.ndarray):
    """0.25 deg fields (..., lat, 1440 from 0 E) -> WeatherBench2's 1.5 deg cells inside 45-75 deg (first-order
    conservative: each cell spans +-0.75 deg, five whole 0.25 deg cells and two halves a side, cos-lat weighted
    in latitude). Returns (..., rows, 240) and the rows' latitudes."""
    g = sum(w * np.roll(f, -k, axis=-1) for w, k in zip(W15, range(-3, 4)))[..., ::6] / W15.sum()
    on = np.isclose(np.mod(lat + 0.75, 1.5), 0.75) & (np.abs(lat) >= HF.BAND[0]) & (np.abs(lat) <= HF.BAND[1])
    rows = []
    for j in np.flatnonzero(on):
        w = W15 * np.cos(np.deg2rad(lat[j - 3 : j + 4]))
        rows.append(np.tensordot(w / w.sum(), g[..., j - 3 : j + 4, :], axes=([0], [-2])))
    return np.stack(rows, axis=-2), lat[on]


# ── fetch ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def fetch_wb2(days: pd.DatetimeIndex) -> dict:
    """WeatherBench2, month by month: the 00/12 UTC fluxes and the flux of each day's mean field (00-18 UTC)."""
    import dask

    dask.config.set(scheduler="threads", num_workers=16)  # the chunks (8 steps x 13 levels) stream in parallel
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    v_all, t_all = ds["v_component_of_wind"].sel(level=100), ds["temperature"].sel(level=100)
    it, fl, dd, fd = [], [], [], []
    for m in days.to_period("M").unique():
        sub = days[days.to_period("M") == m]
        sl = slice(sub[0], sub[-1] + pd.Timedelta(hours=18))
        v, t = dask.compute(v_all.sel(time=sl), t_all.sel(time=sl))
        v, t = (x.transpose("time", "latitude", "longitude") for x in (v, t))  # by name: the store is lon-major
        a, b, lat = v.values.astype(np.float64), t.values.astype(np.float64), v.latitude.values
        screen(a, b, f"WeatherBench2 {m}")
        tt = pd.DatetimeIndex(v.time.values)
        f = band_fluxes(a, b, lat)
        keep = tt.hour.isin(HOURS)
        it.append(tt[keep])
        fl.append(f[keep])
        day = tt.normalize()
        full = [d for d in day.unique() if (day == d).sum() == 4]  # the daily mean needs all four synoptic hours
        if full:
            dm = np.stack([np.stack([a[day == d].mean(0), b[day == d].mean(0)]) for d in full])
            dd.append(pd.DatetimeIndex(full))
            fd.append(band_fluxes(dm[:, 0], dm[:, 1], lat))
        print(f"  WeatherBench2 {m}: {int(keep.sum())} fields", flush=True)
    return {
        "wb2_time": np.array(np.concatenate(it), "datetime64[s]"),
        "wb2_flux": np.concatenate(fl).astype(np.float32),
        "wb2dm_day": np.array(np.concatenate(dd), "datetime64[s]"),
        "wb2dm_flux": np.concatenate(fd).astype(np.float32),
    }


def fetch_arco(days: pd.DatetimeIndex, workers: int = 8) -> dict:
    """ARCO-ERA5 at 00 and 12 UTC: the flux at 0.25 deg and on the 1.5 deg cells. Hours ARCO has not published (NaN
    padding, or beyond its time axis) are skipped."""
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    last = pd.Timestamp(ds.time.values[-1])
    times = [d + pd.Timedelta(hours=h) for d in days for h in HOURS if d + pd.Timedelta(hours=h) <= last]
    rows = slice(76.0, -76.0)  # both bands and the 1.5 deg cells' reach; ARCO runs north to south

    def read(ts):
        for attempt in range(4):
            try:
                v = ds["v_component_of_wind"].sel(time=ts, level=100, latitude=rows)
                return ts, v.values, ds["temperature"].sel(time=ts, level=100, latitude=rows).values, v.latitude.values
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"  ARCO {ts:%Y-%m-%d %H}: {type(e).__name__}, retrying", flush=True)
                time.sleep(10 * (attempt + 1))

    it, fl, f15, skipped = [], [], [], 0
    with ThreadPoolExecutor(workers) as ex:
        for ts, v, t, lat in ex.map(read, times):
            if np.isnan(v).all() or np.isnan(t).all():  # ARCO pads hours it has not published
                skipped += 1
                continue
            a, b = v.astype(np.float64)[None], t.astype(np.float64)[None]
            screen(a, b, f"ARCO {ts:%Y-%m-%d %H}")
            c_v, lat15 = cells15(a, lat)
            c_t, _ = cells15(b, lat)
            it.append(ts)
            fl.append(band_fluxes(a, b, lat)[0])
            f15.append(band_fluxes(c_v, c_t, lat15)[0])
            if ts.day == 1 and ts.hour == 0:
                print(f"  ARCO {ts:%Y-%m}", flush=True)
    if skipped:
        print(f"  ARCO: {skipped} unpublished hours skipped", flush=True)
    return {
        "arco_time": np.array(it, "datetime64[s]"),
        "arco_flux": np.array(fl, np.float32).reshape(len(it), len(KEYS)),
        "arco15_flux": np.array(f15, np.float32).reshape(len(it), len(KEYS)),
    }


def part_days(since, until, part: tuple[int, int]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """This part's contiguous share of the WeatherBench2 days and of the ARCO days (the ten days both hold are read
    from both, for `check`)."""
    k, n = part
    wb2 = pd.date_range(pd.Timestamp(since), min(pd.Timestamp(until), WB2_LAST))
    arco = pd.date_range(max(pd.Timestamp(since), SWITCH), pd.Timestamp(until))
    return (
        pd.DatetimeIndex(np.array_split(wb2.values, n)[k]),
        pd.DatetimeIndex(np.array_split(arco.values, n)[k]),
    )


# ── pack and climatology ────────────────────────────────────────────────────────────────────────────────────────────
def load_parts(paths) -> dict:
    acc: dict[str, list] = {}
    for p in paths:
        with np.load(p) as f:
            for k in f.files:
                acc.setdefault(k, []).append(f[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def pack(paths, out: Path = SERIES) -> xr.Dataset:
    """The 00/12 UTC series: WeatherBench2 before SWITCH, ARCO from it, over whatever the committed file holds."""
    rows: dict[pd.Timestamp, np.ndarray] = {}
    if out.exists():
        with xr.open_dataset(out) as old:
            for i, d in enumerate(pd.DatetimeIndex(old.time.values)):
                for j, h in enumerate(old.hour.values):
                    x = np.array([float(old[k].values[i, j]) for k in KEYS])
                    if np.isfinite(x).all():
                        rows[d + pd.Timedelta(hours=int(h))] = x
    p = load_parts(paths)
    for src, keep in (("wb2", lambda t: t < SWITCH), ("arco", lambda t: t >= SWITCH)):
        for t, x in zip(pd.DatetimeIndex(p.get(f"{src}_time", [])), p.get(f"{src}_flux", [])):
            if keep(t):
                rows[t] = x.astype(np.float64)
    days = pd.date_range(min(rows).normalize(), max(rows).normalize())
    arr = np.full((len(days), len(HOURS), len(KEYS)), np.nan, np.float32)
    pos = {d: i for i, d in enumerate(days)}
    for t, x in rows.items():
        arr[pos[t.normalize()], HOURS.index(t.hour)] = x
    ds = xr.Dataset(
        {k: (("time", "hour"), arr[..., i], {"units": "K m s-1"}) for i, k in enumerate(KEYS)},
        coords={"time": days, "hour": list(HOURS)},
        attrs={
            "title": "ERA5 100 hPa zonal-mean eddy heat flux [v'T'], 45-75 deg",
            "source": f"ERA5: WeatherBench2 6-hourly 1.5 deg (conservative) before {SWITCH:%Y-%m-%d}, ARCO-ERA5 "
            "hourly 0.25 deg (ERA5T at the end) from it",
            "method": f"heatflux100.fluxes / band_mean: zonal wavenumbers 1-{HF.K_MAX}, cos(lat)-weighted "
            f"{HF.BAND[0]:.0f}-{HF.BAND[1]:.0f} deg; single fields at 00 and 12 UTC",
            "sign": "positive = poleward in both hemispheres",
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding={k: {"zlib": True, "complevel": 6} for k in KEYS})
    n = int(np.isfinite(arr[..., 0]).sum())
    print(f"wrote {out.name}: {n} fields, {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}, {len(days) * 2 - n} missing")
    return ds


def clim(series: Path = SERIES, out: Path = OUT, y0: int = 1991, y1: int = 2020) -> None:
    ds = xr.open_dataset(series).load()
    days = pd.DatetimeIndex(ds.time.values)
    base = (days.year >= y0) & (days.year <= y1)
    slot = doy366(days.values) + 1  # 1..366, the same calendar slot in leap and common years
    res = {}
    for name in KEYS:
        x = ds[name].values.astype(np.float64)  # (day, hour)
        if not np.isfinite(x[base]).all():
            raise SystemExit(f"{name}: {int((~np.isfinite(x[base])).sum())} fields missing in {y0}-{y1}")
        run40 = pd.Series(x.mean(1)).rolling(40, min_periods=40).mean().values
        st = {k: np.full(366, np.nan) for k in ("mean", "sd", "p10", "p90", "m40", "sd40")}
        for d in range(1, 367):
            m = base & (np.abs(((slot - d) + 183) % 366 - 183) <= 15)
            inst = x[m].ravel()
            st["mean"][d - 1], st["sd"][d - 1] = inst.mean(), inst.std()
            st["p10"][d - 1], st["p90"][d - 1] = np.percentile(inst, [10, 90])
            r = run40[m & np.isfinite(run40)]
            st["m40"][d - 1], st["sd40"][d - 1] = r.mean(), r.std()
        for k, v in st.items():
            res[f"{name}_{k}"] = ("doy", v.astype("float32"))
    out_ds = xr.Dataset(res, coords={"doy": np.arange(1, 367)})
    out_ds.attrs.update(
        source=f"ERA5 {y0}-{y1}, 00 and 12 UTC (reference/vt100_era5.nc, build_vt100_clim.py)",
        band=f"{HF.BAND[0]:.0f}-{HF.BAND[1]:.0f} deg, cos(lat) weighted, zonal wavenumbers 1-{HF.K_MAX}",
        units="K m s-1",
        sign="positive = poleward in both hemispheres",
        window="+/-15 days; mean/sd/p10/p90 of the 00 and 12 UTC values, m40/sd40 of the trailing 40-day mean of "
        "the daily (00/12 UTC averaged) series",
        doy="1..366 on a 366-slot calendar: 29 Feb = 60, 1 March = 61 in every year",
    )
    out_ds.to_netcdf(out)
    print(
        f"wrote {out.name}: NH mean {res['nh_tot_mean'][1].min():.1f}..{res['nh_tot_mean'][1].max():.1f}, "
        f"SH {res['sh_tot_mean'][1].min():.1f}..{res['sh_tot_mean'][1].max():.1f} K m/s"
    )


# ── check ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def block_ci(d: np.ndarray, block: int = 30, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the mean of a daily series by a moving-block bootstrap (blocks of `block` days)."""
    rng = np.random.default_rng(seed)
    nb = max(1, len(d) // block)
    starts = rng.integers(0, max(1, len(d) - block + 1), size=(n, nb))
    means = np.array([np.concatenate([d[s : s + block] for s in row]).mean() for row in starts])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def check(paths) -> None:
    p = load_parts(paths)

    def frame(flux: str, when: str) -> pd.DataFrame | None:
        return pd.DataFrame(p[flux], index=pd.DatetimeIndex(p[when]), columns=KEYS).sort_index() if flux in p else None

    wb2, arco, a15 = frame("wb2_flux", "wb2_time"), frame("arco_flux", "arco_time"), frame("arco15_flux", "arco_time")
    if wb2 is not None and arco is not None:
        both = wb2.index.intersection(arco.index)
        print(f"WeatherBench2 against ARCO, {len(both)} fields both hold:")
        for h in ("nh", "sh"):
            k = f"{h}_tot"
            w, a, c = wb2.loc[both, k], arco.loc[both, k], a15.loc[both, k]
            print(
                f"  {h}: means WB2 {w.mean():+.2f}, ARCO 0.25 {a.mean():+.2f}, ARCO on 1.5 cells {c.mean():+.2f} K m/s; "
                f"WB2 - cells rms {np.sqrt(((w - c) ** 2).mean()):.3f}, WB2 - 0.25 rms {np.sqrt(((w - a) ** 2).mean()):.3f}"
            )
    if arco is not None:
        print("ARCO 0.25 deg against the 1.5 deg cells (daily means of 00/12 UTC):")
        for h in ("nh", "sh"):
            k = f"{h}_tot"
            a, c = (x[k].groupby(x.index.normalize()).mean() for x in (arco, a15))
            d = (c - a).values
            lo, hi = block_ci(d)
            by = (c - a).groupby(a.index.month).mean() / a.groupby(a.index.month).mean()
            print(
                f"  {h}: cells - 0.25 = {d.mean():+.3f} K m/s (95 % {lo:+.3f}..{hi:+.3f}), ratio of means "
                f"{c.mean() / a.mean():.4f}; by month (cells/0.25 - 1): "
                + " ".join(f"{m}:{x:+.3f}" for m, x in by.items())
            )
    dm = frame("wb2dm_flux", "wb2dm_day")
    if wb2 is None or dm is None:
        return
    inst = wb2.groupby(wb2.index.normalize()).mean()
    base = dm.index[(dm.index.year >= 1991) & (dm.index.year <= 2020)].intersection(inst.index)
    print(f"Flux of the daily-mean fields against the 00/12 UTC fields, WeatherBench2, {len(base)} days 1991-2020:")
    for h in ("nh", "sh"):
        k = f"{h}_tot"
        a, c = inst.loc[base, k], dm.loc[base, k]
        by = c.groupby(c.index.month).mean() / a.groupby(a.index.month).mean() - 1
        print(
            f"  {h}: means 00/12 {a.mean():+.2f}, daily-mean fields {c.mean():+.2f} K m/s (ratio {c.mean() / a.mean():.4f}); "
            "by month: " + " ".join(f"{m}:{x:+.3f}" for m, x in by.items())
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch")
    s.add_argument("--since", default="1990-11-01")  # the first 40-day means of 1991 reach back into 1990
    s.add_argument("--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=6)).date()))
    s.add_argument("--part", default="0/1")
    s.add_argument("--out", required=True)
    for name in ("pack", "check"):
        sub.add_parser(name).add_argument("parts", nargs="+")
    sub.add_parser("clim")
    a = ap.parse_args()
    if a.cmd == "fetch":
        k, n = (int(x) for x in a.part.split("/"))
        wb2, arco = part_days(a.since, a.until, (k, n))
        print(f"part {k}/{n}: WeatherBench2 {len(wb2)} days, ARCO {len(arco)} days", flush=True)
        doc = {}
        if len(wb2):
            doc.update(fetch_wb2(wb2))
        if len(arco):
            doc.update(fetch_arco(arco))
        np.savez_compressed(a.out, **doc)
    elif a.cmd == "pack":
        pack(a.parts)
    elif a.cmd == "clim":
        clim()
    else:
        check(a.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
