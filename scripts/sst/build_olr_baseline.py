#!/usr/bin/env python3
"""ERA5 baselines of equatorial OLR: the GMGSI proxy's 1991-2020 climatology and offset, and the RMM OLR channel's frame.

The equatorial wave products take their OLR from the GMGSI longwave-IR proxy (build_olr_realtime.py) and its anomaly
as proxy − offset − climatology, both looked up on the 366-slot calendar (quantmet doy366: 29 February is slot 59 and
1 March slot 60 in every year):

  climatology  ERA5 daily-mean OLR 1991-2020 in each of the proxy's bands (5°S-5°N, 15°S-15°N) and 2.5° longitude
               cells: a mean and three annual harmonics, least-squares fitted to every day of the 30 years.
  offset       proxy − ERA5 on the days both cover, fitted per longitude as a mean and annual harmonics, how many set
               by blocked cross-validation. The weather is common to both and cancels in the difference, so the fit
               is the proxy's own bias (its brightness-temperature calibration), not the anomaly. Taking out each
               longitude's mean over the proxy record instead also takes out much of a long-lasting anomaly such as
               an El Niño's.

The RMM's OLR channel (scripts/mjo/src/olr_channel.py) is ERA5 OLR put into the frame of the Wheeler & Hendon
reference climatology, NOAA interpolated OLR over 1979-2001 (scripts/mjo/data/reference/climatology.nc), and that
frame is built here too:

  frame        ERA5's own climatology over the reference's base period, in the RMM's band and fitted exactly as the
               reference was (a mean and three annual harmonics in the common-year day of doycal.noleap_doy, every
               day of the 23 years). ERA5 minus NOAA over the same years is then a fixed offset by longitude and day
               of year. Taking ERA5 − NOAA as its mean over the recent record instead also takes out that record's
               mean anomaly.

ERA5 OLR is −mean_top_net_long_wave_radiation_flux (net top-of-atmosphere longwave, negative upward) from the ARCO
store on Google Cloud; a day is the mean of the hourly means ending 00, 06, 12 and 18 UTC, four phases of the diurnal
cycle like the proxy's four images. A proxy band mean is unweighted over the 0.25° rows with |lat| at most the band's
half-width, averaged over 2.5° cells centred on the proxy's longitudes: the proxy's own footprint. The RMM band is the
reduction of the live ERA5 days (scripts/mjo/src/fetch_arco_olr_band.py): cos-weighted over the rows within 15° of the
equator, then the mean of the ten 0.25° columns from each 2.5° longitude eastward.

The offset's uncertainty is a moving-block bootstrap of the fit (blocks of BLOCK days); a forward test fits the first
365 days of the overlap and scores the rest, the way the static offset is used on days after it was estimated.

    python scripts/sst/build_olr_baseline.py fetch --start 1991-01-01 --end 1995-12-31 --out era5_olr_1991.npz
    python scripts/sst/build_olr_baseline.py fetch --out era5_olr_recent.npz   # the proxy store's days, to ERA5's last
    python scripts/sst/build_olr_baseline.py build --recent era5_olr_recent.npz --clim era5_olr_1991.npz ...
    python scripts/sst/build_olr_baseline.py build --recent era5_olr_recent.npz   # re-fit the offset, keep the clim
    python scripts/sst/build_olr_baseline.py frame --clim era5_olr_1979.npz ...   # fetch outputs covering 1979-2001
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

from build_olr_realtime import BANDS, GRID, SAMPLING, STORE

HERE = Path(__file__).resolve().parent
MJO_REF = HERE.parent / "mjo" / "data" / "reference"
sys.path.insert(0, str(HERE.parent / "mjo" / "src"))
from doycal import noleap_doy  # the calendar of the RMM reference climatology

OUT = HERE / "metar" / "olr_baseline.nc"
WH04_CLIM = MJO_REF / "climatology.nc"  # the RMM reference: clim_olr, NOAA interpolated OLR over its base_period
FRAME_OUT = MJO_REF / "era5_olr_clim.nc"
RMM_HALF = 15.0  # the RMM band: 15°S-15°N
KEYS = (*BANDS, "rmm")  # the band means each fetched day holds
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "mean_top_net_long_wave_radiation_flux"
HOURS = (0, 6, 12, 18)  # each day is the mean of the hourly means ending at these UTC hours
CLIM_YEARS = (1991, 2020)
CLIM_HARMONICS = 3
OFFSET_HARMONICS = (0, 1, 2, 3)  # candidate annual cycles of the offset (0 = a constant)
BLOCK = 30  # days per block, in the cross-validation and the bootstrap
NBOOT = 1000
NINO4 = (160.0, 210.0)  # 160°E-150°W, for the summary
SLOTS = np.arange(366)
CELL = (10 * np.arange(len(GRID))[:, None] + np.arange(-5, 6)) % 1440  # 0.25° columns of each centred 2.5° cell
CELL_W = np.r_[0.5, np.ones(9), 0.5] / 10.0  # the two edge columns are shared with the neighbouring cells


def open_flux() -> xr.DataArray:
    """ERA5 net top-of-atmosphere longwave flux over the widest band (lazy; ARCO holds one hour per chunk)."""
    a = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)[VAR]
    units = a.attrs.get("units", "")
    print(f"{VAR}: units {units!r}, chunks {a.encoding.get('chunks')}", flush=True)
    if units and not (units.startswith("W") and units.endswith("-2")):  # each day is range-checked as well
        raise SystemExit(f"{VAR} is not a flux in W m-2")
    if not np.allclose(a.longitude.values, np.arange(1440) * 0.25):
        raise SystemExit("unexpected ERA5 longitudes")
    half = max(*BANDS.values(), RMM_HALF)
    lat = a.latitude.values
    return a.sel(latitude=slice(half, -half) if lat[0] > lat[-1] else slice(-half, half))


def day_bands(flux: xr.DataArray, day: pd.Timestamp) -> dict[str, np.ndarray] | None:
    """Daily-mean OLR of each proxy band on the 2.5° cells and of the RMM band (KEYS), or None for a day ERA5 does not
    reach yet."""
    times = [day + pd.Timedelta(hours=h) for h in HOURS]
    if times[-1] > flux["time"].values[-1]:  # the store's time axis runs ahead of its data, but not without end
        return None
    for attempt in range(4):
        try:
            v = flux.sel(time=times).values
            break
        except Exception:  # noqa: BLE001  (transient storage errors)
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    if np.isnan(v).all():
        return None
    if not np.isfinite(v).all() or (v >= 0).any():
        raise ValueError(f"{day:%Y-%m-%d}: missing or non-negative net longwave flux")
    olr = -v.mean(0)  # net longwave is −OLR
    lat = flux.latitude.values
    out = {b: (olr[np.abs(lat) <= half + 1e-6].mean(0)[CELL] * CELL_W).sum(-1) for b, half in BANDS.items()}
    rows = np.abs(lat) <= RMM_HALF + 1e-6
    w = np.cos(np.deg2rad(lat[rows]))
    out["rmm"] = (w @ olr[rows] / w.sum()).reshape(len(GRID), 10).mean(-1)
    if not all(((x > 90) & (x < 340)).all() for x in out.values()):
        raise ValueError(f"{day:%Y-%m-%d}: band OLR outside 90-340 W m-2")
    return out


def fetch(start: str | None, end: str | None, out: Path, workers: int) -> None:
    """ERA5 daily band means from start to end (default: the proxy store's first day to ERA5's last) -> npz."""
    flux = open_flux()
    if start is None:
        with xr.open_dataset(STORE) as rt:
            start = str(rt["time"].values[0])[:10]
    last = pd.Timestamp(end) if end else pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    days = pd.date_range(start, last, freq="D")
    print(f"ERA5 {days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d}: {len(days)} days x {len(HOURS)} hours", flush=True)
    t0 = time.time()
    got = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(lambda d: day_bands(flux, d), days), 1):
            got.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(days)} days, {(time.time() - t0) / 60:.1f} min", flush=True)
    have = np.array([r is not None for r in got])
    n = int(np.flatnonzero(have)[-1]) + 1 if have.any() else 0
    if n == 0 or not have[:n].all() or (end and n < len(days)):
        gaps = [f"{d:%Y-%m-%d}" for d, h in zip(days, have) if not h]
        raise SystemExit(f"ERA5 missing on {len(gaps)} day(s): {', '.join(gaps[:10])}")
    np.savez_compressed(
        out,
        time=days[:n].values.astype("datetime64[D]"),
        lon=GRID,
        **{f"olr_{b}": np.stack([r[b] for r in got[:n]]).astype("float32") for b in KEYS},
    )
    print(f"wrote {out}: {n} days to {days[n - 1]:%Y-%m-%d} in {(time.time() - t0) / 60:.1f} min", flush=True)


def load_parts(paths, keys=tuple(BANDS)) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """fetch outputs joined in time order: the band means `keys`."""
    parts = [np.load(p) for p in paths]
    if not all(np.array_equal(z["lon"], GRID) for z in parts):
        raise SystemExit("a part is not on the proxy's longitudes")
    t = pd.DatetimeIndex(np.concatenate([z["time"] for z in parts]).astype("datetime64[ns]"))
    order = np.argsort(t.values, kind="stable")
    t = t[order]
    if t.has_duplicates:
        raise SystemExit("the parts overlap")
    return t, {b: np.concatenate([z[f"olr_{b}"] for z in parts])[order].astype(float) for b in keys}


def design(slot: np.ndarray, nh: int, year: float = 366.0) -> np.ndarray:
    """Least-squares columns of a mean and nh annual harmonics of a 0-based day on a calendar of `year` days: the
    366-slot calendar, or (year=365) the common-year day of the RMM reference, doycal.noleap_doy - 1."""
    ang = 2 * np.pi * np.outer(slot, np.arange(1, nh + 1)) / year
    return np.hstack([np.ones((len(slot), 1)), np.cos(ang), np.sin(ang)])


def fit(slot: np.ndarray, y: np.ndarray, nh: int, year: float = 366.0) -> np.ndarray:
    return np.linalg.lstsq(design(slot, nh, year), y, rcond=None)[0]


def fit_offset(days: pd.DatetimeIndex, d: np.ndarray, n4: np.ndarray, band: str) -> tuple[np.ndarray, dict]:
    """(366, lon) offset fitted to the daily proxy − ERA5 differences d, and the numbers that describe it."""
    slot = doy366(days.values)
    blk = np.asarray((days - days[0]).days) // BLOCK
    cv = {}
    for nh in OFFSET_HARMONICS:  # leave one block out at a time
        ss = 0.0
        for k in np.unique(blk):
            out = blk == k
            ss += ((d[out] - design(slot[out], nh) @ fit(slot[~out], d[~out], nh)) ** 2).sum()
        cv[nh] = float(np.sqrt(ss / d.size))
    nh = min(cv, key=cv.get)
    X = design(slot, nh)
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    off = design(SLOTS, nh) @ coef

    # moving-block bootstrap of the residuals (whole days, every longitude together)
    rng = np.random.default_rng(0)
    fitted = X @ coef
    res = d - fitted
    proj = design(SLOTS, nh) @ np.linalg.pinv(X)
    n = len(d)
    s1 = np.zeros_like(off)
    s2 = np.zeros_like(off)
    n4mean = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = ((rng.integers(0, n, -(-n // BLOCK))[:, None] + np.arange(BLOCK)) % n).ravel()[:n]
        ob = proj @ (fitted + res[idx])
        s1 += ob
        s2 += ob**2
        n4mean[i] = ob[:, n4].mean()
    se = np.sqrt(np.maximum(s2 / NBOOT - (s1 / NBOOT) ** 2, 0.0))

    info = {
        f"offset_harmonics_{band}": nh,
        f"offset_mean_{band}": round(float(off.mean()), 2),
        f"offset_nino4_{band}": round(float(off[:, n4].mean()), 2),
        f"offset_se_{band}": round(float(np.median(se)), 2),
        f"offset_se_nino4_{band}": round(float(n4mean.std()), 2),
    }
    print(
        f"  {band}: cross-validated RMS by harmonics "
        + ", ".join(f"{k}: {v:.3f}" for k, v in cv.items())
        + f" -> {nh}; offset mean {off.mean():+.1f} W m-2 (lon {off.mean(0).min():+.1f}..{off.mean(0).max():+.1f},"
        f" seasonal range up to {np.ptp(off, axis=0).max():.1f}); 1-sigma {np.median(se):.2f} median,"
        f" {np.percentile(se, 95):.2f} at the 95th percentile; Nino-4 {off[:, n4].mean():+.1f} +- {n4mean.std():.2f}",
        flush=True,
    )
    split = days[0] + pd.Timedelta(days=365)
    tr = np.asarray(days < split)
    if (~tr).sum() >= BLOCK:  # forward test: a year fitted, the rest scored
        err = (d[~tr] - design(slot[~tr], nh) @ fit(slot[tr], d[tr], nh)).mean(0)
        info[f"offset_forward_nino4_{band}"] = round(float(err[n4].mean()), 2)
        print(
            f"  {band}: forward test, fitted to {days[0]:%Y-%m-%d}..{days[tr][-1]:%Y-%m-%d}, scored on"
            f" {(~tr).sum()} days to {days[-1]:%Y-%m-%d}: mean error Nino-4 {err[n4].mean():+.2f},"
            f" RMS over longitudes {np.sqrt((err**2).mean()):.2f}, largest {np.abs(err).max():.2f} W m-2",
            flush=True,
        )
    return off, info


def build(clim_paths: list[Path], recent_path: Path) -> None:
    with xr.open_dataset(STORE) as ds:
        rt = ds.load()
    if rt.attrs.get("sampling") != SAMPLING:
        raise SystemExit(
            f"{STORE.name} is sampled at {rt.attrs.get('sampling')!r}, not {SAMPLING}: run build_olr_realtime.py"
        )
    if not np.array_equal(rt["lon"].values, GRID):
        raise SystemExit(f"{STORE.name} is not on the 2.5° grid")
    tg = pd.DatetimeIndex(rt["time"].values)
    te, era = load_parts([recent_path])

    if clim_paths:
        tc, c = load_parts(clim_paths)
        span = pd.date_range(f"{CLIM_YEARS[0]}-01-01", f"{CLIM_YEARS[1]}-12-31")
        if not tc.equals(span):
            raise SystemExit(f"the climatology parts hold {len(tc)} of the {len(span)} days {CLIM_YEARS}")
        clim = {b: design(SLOTS, CLIM_HARMONICS) @ fit(doy366(tc.values), c[b], CLIM_HARMONICS) for b in BANDS}
        clim_attrs = {
            "clim_source": f"ERA5 (ARCO) -{VAR}, daily mean of the hourly means ending "
            + "/".join(f"{h:02d}" for h in HOURS)
            + " UTC",
            "clim_period": f"{CLIM_YEARS[0]}-{CLIM_YEARS[1]}",
            "clim_harmonics": CLIM_HARMONICS,
        }
        for b in BANDS:
            print(
                f"  clim {b}: annual mean {clim[b].mean():.1f} W m-2, cells {clim[b].mean(0).min():.0f}"
                f"..{clim[b].mean(0).max():.0f}, seasonal range up to {np.ptp(clim[b], axis=0).max():.0f}",
                flush=True,
            )
    else:
        with xr.open_dataset(OUT) as old:
            clim = {b: old[f"clim_{b}"].values.astype(float) for b in BANDS}
            clim_attrs = {k: v for k, v in old.attrs.items() if k.startswith("clim_")}

    both = tg.intersection(te)
    G = {b: rt[f"olr_{b}"].sel(time=both).values.astype(float) for b in BANDS}
    D = {b: G[b] - era[b][te.get_indexer(both)] for b in BANDS}
    ok = np.all([np.isfinite(D[b]).all(1) for b in BANDS], axis=0)
    # screen out days the mosaic got wrong (a missing or duplicated pass): their scatter about a provisional fit is an
    # outlier by median + 5 MAD, so the outliers cannot widen their own threshold
    slot = doy366(both.values)
    rms = np.max(
        [np.sqrt(np.mean((D[b] - design(slot, 2) @ fit(slot[ok], D[b][ok], 2)) ** 2, axis=1)) for b in BANDS], axis=0
    )
    med = np.median(rms[ok])
    bad = ok & (rms > med + 5 * 1.4826 * np.median(np.abs(rms[ok] - med)))
    ok &= ~bad
    days = both[ok]
    print(
        f"offset: proxy ({SAMPLING} UTC) - ERA5 over {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}, {ok.sum()} days"
        f" ({(~ok).sum()} left out: " + (", ".join(f"{t:%Y-%m-%d}" for t in both[~ok]) or "none") + ")",
        flush=True,
    )
    lon = rt["lon"].values
    n4 = (lon >= NINO4[0]) & (lon <= NINO4[1])
    off, attrs = {}, {}
    for b in BANDS:
        off[b], info = fit_offset(days, D[b][ok], n4, b)
        attrs.update(info)

    # what the baseline changes, Nino-4: ERA5's anomaly, the proxy's against this baseline, and the proxy's once its
    # own record mean is taken out (the baseline it replaces); and how closely the proxy's anomalies follow ERA5's
    sl_all = doy366(tg.values)
    ig = tg.get_indexer(both)
    n30 = ok.sum() // 30 * 30
    for b in BANDS:
        a_e = era[b][te.get_indexer(both)] - clim[b][slot]
        a_g = rt[f"olr_{b}"].values - off[b][sl_all] - clim[b][sl_all]
        a_r = a_g - np.nanmean(a_g, axis=0)
        pair = (a_e[ok], a_g[ig[ok]])
        for key, x, y in (("daily", *pair), ("30d", *(z[:n30].reshape(-1, 30, z.shape[1]).mean(1) for z in pair))):
            dx, dy = x - x.mean(0), y - y.mean(0)  # about each longitude's mean
            slope = (dx * dy).sum() / (dx**2).sum()
            r = (dx * dy).sum() / np.sqrt((dx**2).sum() * (dy**2).sum())
            attrs[f"proxy_slope_{key}_{b}"] = round(float(slope), 2)
            print(f"  {b}: proxy anomaly regressed on ERA5's, {key}: slope {slope:.2f}, r {r:.2f}", flush=True)
        for label, w in (("overlap", slice(None)), ("last 90 days", slice(-90, None))):
            sel = np.flatnonzero(ok)[w]
            print(
                f"  {b} Nino-4, {label} ({both[sel[0]]:%Y-%m-%d}..{both[sel[-1]]:%Y-%m-%d}): ERA5 anomaly"
                f" {a_e[sel][:, n4].mean():+.1f}, proxy {a_g[ig[sel]][:, n4].mean():+.1f}, proxy less its record"
                f" mean {a_r[ig[sel]][:, n4].mean():+.1f} W m-2",
                flush=True,
            )

    ds = xr.Dataset(
        {
            **{f"clim_{b}": (("slot", "lon"), clim[b].astype("float32")) for b in BANDS},
            **{f"offset_{b}": (("slot", "lon"), off[b].astype("float32")) for b in BANDS},
        },
        coords={"slot": SLOTS, "lon": GRID},
        attrs={
            "title": "ERA5 baseline of the GMGSI OLR proxy: anomaly = proxy - offset - clim",
            "slot": "quantmet.degdays.doy366 (29 February = 59, 1 March = 60 in every year)",
            "bands": "05: |lat| <= 5, 15: |lat| <= 15; unweighted rows, 2.5 deg cells centred on lon",
            "units": "W m-2",
            **clim_attrs,
            "proxy_sampling": SAMPLING,
            "offset_overlap": f"{days[0]:%Y-%m-%d}/{days[-1]:%Y-%m-%d}",
            "offset_days": int(ok.sum()),
            "offset_uncertainty": f"1-sigma, moving-block bootstrap ({BLOCK}-day blocks, {NBOOT} draws)",
            **attrs,
        },
    )
    ds.to_netcdf(OUT, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e3:.0f} kB)", flush=True)


def frame(clim_paths: list[Path]) -> None:
    """ERA5's climatology of the RMM band over the reference climatology's base period -> FRAME_OUT, and its offset
    from the reference: the fixed ERA5-to-NOAA frame shift olr_channel.py takes out."""
    with xr.open_dataset(WH04_CLIM) as ref:
        noaa = ref["clim_olr"].transpose("dayofyear", "longitude").values
        days, lon = ref["dayofyear"].values, ref["longitude"].values
        period, nh = str(ref.attrs["base_period"]), int(ref.attrs["n_harmonics"])
    if not (np.array_equal(days, np.arange(1, 367)) and np.allclose(lon, GRID)):
        raise SystemExit(f"{WH04_CLIM.name} is not on the 366-day x 2.5° grid")
    y0, y1 = (int(y) for y in period.split("–"))
    t, c = load_parts(clim_paths, ("rmm",))
    span = pd.date_range(f"{y0}-01-01", f"{y1}-12-31")
    if not t.equals(span):
        raise SystemExit(f"the parts hold {len(t)} of the {len(span)} days of the reference period {period}")
    # the reference's fit (setup_reference.smooth_annual_cycle): every day, harmonics of the common-year day, 366
    # output days of which the last repeats the first
    clim = design(days - 1, nh, 365.0) @ fit(noleap_doy(t.values) - 1, c["rmm"], nh, 365.0)
    off = clim - noaa
    ann = off.mean(0)
    n4 = (lon >= NINO4[0]) & (lon <= NINO4[1])
    stats = {
        "frame_mean": round(float(off.mean()), 2),
        "frame_nino4": round(float(off[:, n4].mean()), 2),
        "frame_lon_min": round(float(ann.min()), 2),
        "frame_lon_max": round(float(ann.max()), 2),
        "frame_seasonal_range_max": round(float(np.ptp(off, axis=0).max()), 2),
    }
    print(
        f"frame: ERA5 {period} ({len(t)} days, annual mean {clim.mean():.1f} W m-2) minus the NOAA reference:"
        f" mean {off.mean():+.2f}, by longitude {ann.min():+.1f}..{ann.max():+.1f}, Nino-4 {off[:, n4].mean():+.2f},"
        f" seasonal range up to {np.ptp(off, axis=0).max():.1f} W m-2",
        flush=True,
    )
    ds = xr.Dataset(
        {"clim_olr": (("dayofyear", "longitude"), clim.astype("float32"))},
        coords={"dayofyear": days, "longitude": GRID},
        attrs={
            "title": "ERA5 15S-15N OLR climatology in the RMM band over the RMM reference period",
            "source": f"ERA5 (ARCO) -{VAR}, daily mean of the hourly means ending "
            + "/".join(f"{h:02d}" for h in HOURS)
            + " UTC",
            "band": "cos-weighted 0.25 deg rows with |lat| <= 15, then the mean of the ten 0.25 deg columns from each"
            " longitude eastward (the reduction of fetch_arco_olr_band.py)",
            "base_period": period,
            "n_harmonics": nh,
            "calendar": "dayofyear = common-year day (doycal.noleap_doy); 366 repeats 1, as in climatology.nc",
            "frame": "clim_olr minus climatology.nc clim_olr = ERA5 minus NOAA interpolated OLR over the same years",
            **stats,
            "units": "W m-2",
        },
    )
    ds.to_netcdf(FRAME_OUT, encoding={"clim_olr": {"zlib": True, "complevel": 5}})
    print(f"wrote {FRAME_OUT.name} ({FRAME_OUT.stat().st_size / 1e3:.0f} kB)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ERA5 climatology and offset for the GMGSI OLR proxy, and the RMM frame")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="ERA5 daily band means for a span of days")
    f.add_argument("--start", help="first day, YYYY-MM-DD (default: the proxy store's first day)")
    f.add_argument("--end", help="last day (default: ERA5's latest day, at most yesterday)")
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build", help=f"climatology and offset -> {OUT.name}")
    b.add_argument("--recent", type=Path, required=True, help="fetch output covering the proxy store's days")
    b.add_argument(
        "--clim",
        type=Path,
        nargs="*",
        default=[],
        help=f"fetch outputs covering {CLIM_YEARS[0]}-{CLIM_YEARS[1]} (without them the existing climatology is kept)",
    )
    r = sub.add_parser(
        "frame", help=f"ERA5 climatology of the RMM band over the RMM reference period -> {FRAME_OUT.name}"
    )
    r.add_argument(
        "--clim", type=Path, nargs="+", required=True, help=f"fetch outputs covering {WH04_CLIM.name}'s period"
    )
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.start, args.end, args.out, args.workers)
    elif args.cmd == "frame":
        frame(args.clim)
    else:
        build(args.clim, args.recent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
