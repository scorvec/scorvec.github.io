#!/usr/bin/env python3
"""NH polar vortex FORECAST monitor -- AIFS-ENS, IFS-ENS, the IFS control, GEFS, GEPS, GDPS and GEOS FP.

Three panels, forecast-first (a short analysis tail for context, then every
model that publishes the stratosphere):

  1. u(60N) at 10 hPa  -- the WMO sudden-stratospheric-warming diagnostic.
     Easterly here IS an SSW, so the zero line is heavy and the cross-model
     member fraction below it is called out.
  2. u(60N) at 100 hPa -- the coupling level. A 10 hPa event that never shows
     at 100 hPa rarely reaches the surface.
  3. 100 hPa polar-cap (65-90N) height anomaly -- positive = weak/displaced
     vortex; the field that leads the AO/NAO response.

Models (probed 2026-08-28, GEFS 2026-10-20; every one publishes u AND height at 10 and 100 hPa):

  aifs  AIFS-ENS  25 pf + cf, 0.25 deg, day 15.  FREE -- the AAM task already
                  caches pf_u_10-50-100-... every cycle. Height: 25 pf z at 100 hPa
                  pulled through the shared store (~0.35 GB) plus the control.
  gefs  GEFS      30 perturbed + control, 0.5 deg, day 16: u at 10/100 hPa and the 100 hPa height
                  from the shared GEFS fetch (strat_ens.py, part "vortex"; NOAA's S3, ~0.6 GB/cycle).
  geps  GEPS      20 members, 0.5 deg, day 16. One `allmbrs` GRIB per
                  variable/level/step (~2.4 MB UGRD, ~1.3 MB HGT) -> ~98 MB/cycle, through the
                  shared Datamart client (scripts/geps/datamart.py).
  gdps  GDPS      deterministic, 0.15 deg, day 10 -> ~31 MB/cycle.
  geosfp GEOS FP   NASA GMAO, deterministic, newest 00Z run (day 10; 06/12/18Z stop at day 5), NCCS OPeNDAP
                  MERRA-2's operational sibling.
  ifs   IFS-ENS   all 50 perturbed members, 0.25 deg, day 15. The ensemble files carry
                  no control at these levels - open data publishes only type=pf there.
                  Read from the shared IFS-ENS files (ifs_ens.py: u at 10/100 hPa and gh
                  at 100 hPa, fetched once a cycle for every stratosphere product, Google
                  Cloud mirror only). IFS day 15 reaches the mirror ~08:50Z / 20:55Z, so
                  this figure renders in strat-ifs.yml (09:05 / 21:05), not strat.yml; a
                  cycle the mirror does not have yet drops IFS with a message.
  hres  IFS control  the unperturbed run (called HRES until May 2026), 0.25 deg, day 15,
                  from its own open-data stream (oper): u and gh at 10/100 hPa from the
                  shared control file (ifs_ens.py ctrl_wind, ~0.1 GB). Drawn dashed, as
                  the other single runs.

DETRENDING (the reason the reference is built the way it is)
------------------------------------------------------------
Polar-cap 100 hPa height has a strong secular trend -- tropospheric warming
lifts the 100 hPa surface. Measured on MERRA-2 1980-2026 with the seasonal cycle
removed first (load_clim prints these every run):

    z100 cap    +18.1 m/decade    (residual sd 100 m)
    u60 @10hPa   -0.23 m/s/decade (residual sd 10.3 m/s)
    u60 @100hPa  -0.03 m/s/decade (residual sd  3.9 m/s)

At today's epoch the z100 trend alone is about +43 m against the record-mean level, so
an undetrended 1980-2026 band would place every recent day high by construction
and manufacture a "weak vortex" that is really just climate drift. The height
climatology is therefore referenced to the CURRENT epoch: the trend is removed
and every historical day is adjusted to today's level before the day-of-year
percentiles are taken. The winds are left alone -- their trends are under 2.5% of
their own variability, and pretending to correct that would be false precision.

Reference: MERRA-2 daily from 1980, day-of-year 10th-90th percentiles, from the committed series
reference/m2_strat_daily.nc (build_m2_reference.py). Its polar cap is the cos-weighted 65-90N mean of zbar,
the definition used on every forecast. The trends the figure's footer quotes are measured on it every run.

ANALYSIS TAIL: GEOS FP, MERRA-2's operational sibling -- the daily mean of its 00/06/12/18Z analyses, read from
the files the upper-stratosphere and BDC jobs commit every day (u60 at 10 and 100 hPa in upperstrat_tail.json, the
65-90N cap height in geosfp_cap_history.json). Its heights, and those of the GEOS FP forecast, are put on the
MERRA-2 scale by the measured GEOS FP - MERRA-2 offset (geosfp_cap.offset_at, -14 m at 100 hPa); the winds need
none. _consistency() checks the tail against MERRA-2 wherever the two overlap.

    python scripts/strat/nh_vortex.py                     # every model, IFS-ENS at 50 members
    python scripts/strat/nh_vortex.py --models aifs,geps  # no downloads beyond GEPS
    python scripts/strat/nh_vortex.py --cycle 2026092700  # a named cycle (no ens_cycle cache needed)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from quantmet.degdays import doy366

import geosfp_cap
import vortex_runs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "geps"))
import datamart  # every ECCC Datamart request: the site's User-Agent, retries, the fetch ledger

DATA = HERE / "data"
M2_CACHE = HERE / "reference" / "m2_strat_daily.nc"  # MERRA-2 daily u60 at 10/100 hPa and 100 hPa cap height
# GEOS FP analyses, daily: u60 by level (reference seed + the rolling file) and the 65-90N cap height
FP_U_TAIL = (
    REPO / "scripts" / "strat" / "reference" / "upperstrat_tail_seed.json",
    REPO / "assets" / "sst" / "data" / "upperstrat_tail.json",
)
FP_CAP_TAIL = REPO / "assets" / "sst" / "data" / "geosfp_cap_history.json"
CACHE = REPO / "scripts" / "ecmwf" / "cache"
DL = DATA / "vortex_dl"  # downloaded GRIB, reused within a cycle
OUT = REPO / "assets" / "sst" / "nh_vortex.webp"

PHI = 60.0  # WMO 60N for the SSW criterion
CAP = (65.0, 90.0)  # polar cap for the 100 hPa height anomaly
TAIL_DAYS = 21  # short analysis tail: this is a forecast product
G = 9.80665  # ECMWF `z` is geopotential; `gh` / HGT are already gpm
AIFS_Z_MEMBERS = 25  # perturbed members for the AIFS polar-cap height (as for the winds)
DETREND = {"zcap"}  # see the module docstring for why only this one

STYLE = {
    "aifs": ("#b4541f", "AIFS-ENS"),
    "ifs": ("#1f4b6e", "IFS-ENS"),
    "hres": ("#1f4b6e", "IFS control"),  # dashed, so the ensemble's navy is free for it
    "gefs": ("#3b88c4", "GEFS"),  # mid blue: apart from GEPS green for every reader, lighter than IFS-ENS navy
    "geps": ("#2f7d4f", "GEPS"),
    "gdps": ("#6b4c9a", "GDPS"),
    "geosfp": ("#b0306a", "GEOS FP"),
}


# ── shared reductions ───────────────────────────────────────────────────────
def zonal_at(da, phi=PHI):
    return da.sel(latitude=phi, method="nearest").mean("longitude")


def cap_mean(da, lo=CAP[0], hi=CAP[1]):
    lat = da.latitude
    sel = da.where((lat >= lo) & (lat <= hi), drop=True)
    w = np.cos(np.deg2rad(sel.latitude))
    return (sel.mean("longitude") * w).sum("latitude") / w.sum()


def _grib(path, **keys):
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(filter_by_keys=keys, indexpath=""))


def _first(ds):
    return ds[next(iter(ds.data_vars))]


def _frame(da, reduce_fn, base, scale=1.0):
    r = reduce_fn(da) * scale
    t = base + pd.to_timedelta(r.step.values)
    if "number" in r.dims:
        return pd.DataFrame(r.transpose("step", "number").values, index=t)
    return pd.DataFrame({0: np.asarray(r.values)}, index=t)


def _fetch(url, dest):
    got = datamart.download(url, dest, timeout=240)  # a no-op when this cycle's file is already cached
    if got is None:
        print(f"    miss {url.rsplit('/', 1)[-1][:52]}", flush=True)
    return got


# ── model loaders: each returns {"u10","u100","zcap"} of DataFrames ─────────
def _store():
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
    import store as ecmwf

    return ecmwf


def load_aifs(cdir, base):
    pf = glob.glob(f"{cdir}/aifs-ens/pf_u_10-*.grib2")
    cf = glob.glob(f"{cdir}/aifs-ens/cf_u_10-*.grib2")
    if not (pf and cf):
        # No ens_cycle cache (strat-ifs.yml runs without it): pull just u at 10/100 hPa, the 25 perturbed members
        # the cache would have held plus the control, through the shared store (~0.6 GB, Google mirror).
        try:
            ecmwf = _store()
            cyc = ecmwf.Cycle(f"{base:%Y%m%d}", f"{base:%H}")
            pf = [
                ecmwf.ensure(
                    cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", (10, 100), tuple(ecmwf.STEPS), AIFS_Z_MEMBERS)
                )
            ]
            cf = [ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", (10, 100), tuple(ecmwf.STEPS)))]
        except Exception as e:  # noqa: BLE001
            print(f"  AIFS-ENS u unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
            return None
    out = {}
    for lev, key in ((10, "u10"), (100, "u100")):
        d = _frame(_grib(pf[0], shortName="u", level=lev)["u"], zonal_at, base)
        d["cf"] = _frame(_grib(cf[0], shortName="u", level=lev)["u"], zonal_at, base)[0]
        out[key] = d
    zf = glob.glob(f"{cdir}/aifs-ens/cf_z_10-*.grib2")
    if not zf:
        try:
            ecmwf = _store()
            zf = [
                ecmwf.ensure(
                    ecmwf.Cycle(f"{base:%Y%m%d}", f"{base:%H}"),
                    ecmwf.Spec("aifs-ens", "cf", "z", "pl", (100,), tuple(ecmwf.STEPS)),
                )
            ]
        except Exception as e:  # noqa: BLE001
            print(f"  AIFS-ENS control z100 unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
    zc = _frame(_grib(zf[0], shortName="z", level=100)["z"], cap_mean, base, 1.0 / G) if zf else None
    # Open data does carry perturbed z at 100 hPa (index probed 2026-09-26: pf z at
    # 10/50/100, all 50 members, ~0.85 MB a field), so the panel gets members like the winds.
    # ~0.35 GB through the shared store; a failed pull leaves the control alone.
    zp = None
    try:
        import sys

        sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
        import store as ecmwf

        zpath = ecmwf.ensure(
            ecmwf.Cycle(f"{base:%Y%m%d}", f"{base:%H}"),
            ecmwf.Spec("aifs-ens", "pf", "z", "pl", (100,), tuple(ecmwf.STEPS), AIFS_Z_MEMBERS),
        )
        zp = _frame(_grib(zpath, shortName="z", level=100)["z"], cap_mean, base, 1.0 / G)
    except Exception as e:  # noqa: BLE001
        print(
            f"  AIFS-ENS perturbed z100 unavailable ({type(e).__name__}: {str(e)[:80]}); "
            "polar-cap height stays control-only",
            flush=True,
        )
    if zp is not None and zc is not None:
        zp["cf"] = zc[0].reindex(zp.index)
    out["zcap"] = zp if zp is not None else zc
    return out


GEPS_URL = (
    "https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/"
    "CMC_geps-raw_{v}_ISBL_{lev:04d}_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2"
)
GDPS_URL = (
    "https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{c}/{L:03d}/"
    "{d}T{c}Z_MSC_GDPS_{v}_IsbL-{lev:04d}_LatLon0.15_PT{L:03d}H.grib2"
)


def _eccc(kind, url_tpl, date, cyc, leads, spec):
    base = pd.Timestamp(f"{date} {cyc}:00")
    out = {}
    for key, var, lev, red in spec:
        rows = {}
        for L in leads:
            f = _fetch(
                url_tpl.format(d=date, c=cyc, v=var, lev=lev, L=L), DL / f"{kind}_{date}{cyc}_{var}{lev}_{L:03d}.grib2"
            )
            if f is None:
                continue
            # a GEPS `allmbrs` GRIB carries the 20 perturbed members AND the
            # control; opening it unfiltered raises "multiple values for unique
            # key". Take pf for the spread and append cf as one more column.
            vals = []
            for dt in ("pf", "cf") if kind == "geps" else (None,):
                try:
                    ds = _grib(f, dataType=dt) if dt else _grib(f)
                    vals.append(np.atleast_1d(np.asarray(red(_first(ds)).values)).ravel())
                except Exception as e:  # noqa: BLE001
                    if dt == "pf" or dt is None:
                        print(f"    {kind} {var}{lev} +{L}h unreadable: {str(e)[:50]}", flush=True)
            if not vals:
                continue
            rows[base + pd.Timedelta(hours=L)] = np.concatenate(vals)
        if not rows:
            return None
        out[key] = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return out


def load_geps(date, cyc, leads):
    return _eccc(
        "geps",
        GEPS_URL,
        date,
        cyc,
        leads,
        [("u10", "UGRD", 10, zonal_at), ("u100", "UGRD", 100, zonal_at), ("zcap", "HGT", 100, cap_mean)],
    )


def load_gdps(date, cyc, leads):
    return _eccc(
        "gdps",
        GDPS_URL,
        date,
        cyc,
        leads,
        [
            ("u10", "WindU", 10, zonal_at),
            ("u100", "WindU", 100, zonal_at),
            ("zcap", "GeopotentialHeight", 100, cap_mean),
        ],
    )


def load_gefs(date, cyc):
    """NOAA GEFS, control + 30 members, to day 16, from the shared GEFS fetch's "vortex" part (strat_ens.py: u at 10
    and 100 hPa and the 100 hPa height, daily, byte-ranged from NOAA's S3, ~0.6 GB). A cycle not complete on S3 falls
    back to the newest complete one within a day; with none the figure goes without GEFS."""
    import strat_ens as SE

    try:
        d0, h0 = SE.resolve("gefs", date, cyc)
        d = SE.load("gefs", d0, h0, ("vortex",))
    except SystemExit as e:
        print(f"::warning::GEFS unavailable ({e}); the figure goes without it", flush=True)
        return None
    keep = SE.complete_members(d, "zcap")
    idx = pd.Timestamp(f"{d0[:4]}-{d0[4:6]}-{d0[6:]} {h0}:00") + pd.to_timedelta(list(SE.LEADS_VORTEX), unit="h")
    return {
        "u10": pd.DataFrame(d["u60"][keep, :, 0].T, index=idx),
        "u100": pd.DataFrame(d["u60"][keep, :, 1].T, index=idx),
        "zcap": pd.DataFrame(d["zcap"][keep].T, index=idx),
    }


def load_ifs(date, cyc, leads, members, light=False):
    """IFS-ENS, all 50 perturbed members (or the first `members`), from the shared IFS-ENS files.

    ifs_ens.py stocks u/v at 10/100 hPa (12-hourly) and gh at the 11 drip levels (daily) ONCE a cycle for every
    stratosphere product; this reads u at 10/100 and gh at 100 hPa out of them at the daily steps. `light` (a past run
    recorded on its own, --record-only) pulls just those fields, ~2 GB instead of the bundle's ~7.5. Google Cloud
    mirror only: a cycle not yet on Google is dropped rather than pulled slowly from another mirror.
    """
    import sys

    sys.path.insert(0, str(HERE))
    try:
        import ifs_ens as IE
    except Exception as e:  # noqa: BLE001
        print(f"    ifs_ens unavailable ({e}); skipping IFS", flush=True)
        return None
    if not IE.published(date, cyc):
        print(
            f"::warning::IFS-ENS {date} {cyc}Z is not on the Google mirror yet; the figure goes without it", flush=True
        )
        return None
    cycle = IE.E.Cycle(date, cyc)
    try:
        if light:
            uv = IE.ensure(cycle, IE.E.Spec("ifs", "pf", "u", "pl", (10, 100), IE.STEPS))
            gh = IE.ensure(cycle, IE.E.Spec("ifs", "pf", "gh", "pl", (100,), IE.STEPS))
        else:
            uv = IE.ensure(cycle, IE.spec_uv())
            gh = IE.ensure(cycle, IE.spec_gh())
    except Exception as e:  # noqa: BLE001
        print(
            f"::warning::IFS-ENS shared files unavailable ({type(e).__name__}: {str(e)[:100]}); skipping IFS",
            flush=True,
        )
        return None
    base = pd.Timestamp(f"{date} {cyc}:00")
    out = {}
    for key, path, short, lev, red in (
        ("u10", uv, "u", 10, zonal_at),
        ("u100", uv, "u", 100, zonal_at),
        ("zcap", gh, "gh", 100, cap_mean),
    ):
        da = IE.open_field(path, short, lev)
        da = da.isel(step=np.flatnonzero(np.isin(IE.step_hours(da), leads)))
        if members and "number" in da.dims and members < da.sizes["number"]:
            da = da.isel(number=slice(0, members))
        out[key] = _frame(da, red, base)
    print(f"    IFS-ENS: {out['u10'].shape[1]} members from the shared files", flush=True)
    return out


def load_hres(date, cyc, leads):
    """The IFS control at the daily steps: u at 10/100 hPa and gh at 100 hPa from the shared control file
    (ifs_ens.py ctrl_wind, 12-hourly). Google Cloud mirror only; a cycle not there yet drops it with a message."""
    import sys

    sys.path.insert(0, str(HERE))
    try:
        import ifs_ens as IE
    except Exception as e:  # noqa: BLE001
        print(f"    ifs_ens unavailable ({e}); skipping the IFS control", flush=True)
        return None
    if not IE.ctrl_published(date, cyc):
        print(f"    IFS control {date} {cyc}Z is not on the Google mirror yet; the figure goes without it", flush=True)
        return None
    try:
        path = IE.ensure(IE.E.Cycle(date, cyc), IE.spec_ctrl_wind())
    except Exception as e:  # noqa: BLE001
        print(f"::warning::IFS control file unavailable ({type(e).__name__}: {str(e)[:100]}); skipping it", flush=True)
        return None
    base = pd.Timestamp(f"{date} {cyc}:00")
    out = {}
    for key, short, lev, red in (
        ("u10", "u", 10, zonal_at),
        ("u100", "u", 100, zonal_at),
        ("zcap", "gh", 100, cap_mean),
    ):
        da = IE.open_field(path, short, lev)
        out[key] = _frame(da.isel(step=np.flatnonzero(np.isin(IE.step_hours(da), leads))), red, base)
    return out


GEOSFP_FC = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/inst3_3d_asm_Np/inst3_3d_asm_Np.{c}"


def _fp_offset100(days) -> np.ndarray:
    """GEOS FP minus MERRA-2 65-90N cap height at 100 hPa (m) on these days: subtract it from a GEOS FP cap series."""
    return geosfp_cap.offset_at(pd.DatetimeIndex(days))[:, geosfp_cap.LEVELS.index(100.0)]


def load_geosfp(base):
    """NASA GMAO GEOS FP, deterministic, from the NCCS OPeNDAP forecast collection. Only the 00Z run
    goes to day 10 (the 06/12/18Z runs stop at day 5), so this takes the newest 00Z run at or before the figure's cycle,
    reading the daily 00Z steps of u at 10/100 hPa along 60N and h (geopotential height, m) at 100 hPa over 65-90N.
    GEOS FP is the operational sibling of MERRA-2, the reference the bands come from. ~1 min over OPeNDAP."""
    import time as _t

    day0 = base.normalize()
    for back in (0, 1):
        init = day0 - pd.Timedelta(days=back)
        url = GEOSFP_FC.format(c=f"{init:%Y%m%d}_00")
        d = None
        for k in range(3):
            try:
                d = xr.open_dataset(url)
                if not np.issubdtype(d.time.dtype, np.datetime64):
                    raise RuntimeError("time axis not decoded")
                break
            except Exception as e:  # noqa: BLE001
                d = None
                if "not found" in str(e).lower() or "404" in str(e):
                    break
                _t.sleep(15 * (k + 1))
        if d is None:
            continue
        tt = pd.DatetimeIndex(d.time.values)
        if len(tt) < 70:  # run still being written
            continue
        ti = np.flatnonzero(tt.hour == 0)
        lev = d.lev.values
        l10, l100 = int(np.argmin(np.abs(lev - 10))), int(np.argmin(np.abs(lev - 100)))
        try:
            u = d["u"].isel(time=ti, lev=[l10, l100]).sel(lat=PHI, method="nearest").load().mean("lon")
            z = d["h"].isel(time=ti, lev=l100).sel(lat=slice(CAP[0], CAP[1])).load()
        except Exception as e:  # noqa: BLE001
            print(
                f"::warning::GEOS FP {init:%Y%m%d} 00Z read failed ({str(e)[:90]}); figure goes without it", flush=True
            )
            return None
        w = np.cos(np.deg2rad(z.lat))
        idx = tt[ti]
        zc = (z.mean("lon") * w).sum("lat") / w.sum() - _fp_offset100(idx)  # on the MERRA-2 scale, like the tail
        print(
            f"    GEOS FP {init:%Y-%m-%d} 00Z: {len(idx)} days, u10(0) {float(u.isel(time=0, lev=0)):.1f} m/s, "
            f"cap z100(0) {float(zc.isel(time=0)):.0f} m",
            flush=True,
        )
        return {
            "u10": pd.DataFrame({0: u.isel(lev=0).values}, index=idx),
            "u100": pd.DataFrame({0: u.isel(lev=1).values}, index=idx),
            "zcap": pd.DataFrame({0: zc.values}, index=idx),
        }
    print(
        "::warning::no complete GEOS FP 00Z forecast for this cycle or the day before; figure goes without it",
        flush=True,
    )
    return None


# ── reference: MERRA-2, epoch-referenced where a trend matters ──────────────
def _cday(ix):
    """Calendar day 1..366 on the 366-slot calendar: 29 Feb is 60 and 1 March 61 in every year."""
    return doy366(pd.DatetimeIndex(ix)) + 1


def _detrend_to_epoch(s, epoch):
    """Adjust every historical day to `epoch`'s climate level.

    Seasonal cycle out first, then a linear fit in decimal years; the fitted
    slope is used to shift history forward.
    """
    ix = pd.DatetimeIndex(s.index)
    yr = np.asarray(ix.year + (ix.dayofyear - 1) / 365.25, dtype=float)
    r = (s - s.groupby(_cday(ix)).transform("mean")).values
    b, _ = np.polyfit(yr, r, 1)
    return pd.Series(s.values + b * (epoch - yr), index=ix)


def clim_doy(series):
    """Daily 10/50/90 % by calendar day (_cday; 29 Feb from the leap years), indexed 1..366."""
    g = series.groupby(_cday(series.index))

    def q(p):
        return g.quantile(p).reindex(range(1, 367)).interpolate().bfill().ffill()

    return q(0.10), q(0.50), q(0.90)


def _clim_series():
    """The three MERRA-2 daily series of reference/m2_strat_daily.nc (build_m2_reference.py)."""
    with xr.open_dataset(M2_CACHE) as c:
        return pd.DatetimeIndex(c.time.values), {k: np.asarray(c[k].values) for k in ("u10", "u100", "zcap")}


# Physically impossible values, used to screen the reference before it becomes a
# percentile band. MERRA-2 carries one corrupt day in 46 years — 1991-06-30, where
# the polar-cap height collapses to 0.80 m at 100 hPa and 3.19 m at 10 hPa (zbar
# goes negative) — and a single garbage day inside a day-of-year bin of ~46
# samples visibly drags that day's 10th percentile. An archive-wide check finds the
# rest of the archive clean; the NaNs at 1000/975 hPa near the pole are just the
# pressure surface being below ground.
PHYSICAL = {"u10": (-150.0, 200.0), "u100": (-150.0, 200.0), "zcap": (10000.0, 20000.0)}


def _screen(out):
    for k, (lo, hi) in PHYSICAL.items():
        if k not in out:
            continue
        s = out[k]
        bad = (s < lo) | (s > hi)
        if bad.any():
            print(
                f"    {k}: dropped {int(bad.sum())} non-physical day(s) "
                f"({', '.join(f'{d:%Y-%m-%d}' for d in s.index[bad][:3])})",
                flush=True,
            )
            out[k] = s[~bad]
    return out


def load_clim(epoch):
    """({key: screened daily series, the DETREND ones referenced to `epoch`}, {key: trend per decade})."""
    idx, raw = _clim_series()
    out = _screen({k: pd.Series(v, index=idx).dropna() for k, v in raw.items()})
    print(f"  MERRA-2 reference {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d} ({len(idx)} days)", flush=True)
    trend = {}
    for k in list(out):
        ix = pd.DatetimeIndex(out[k].index)
        yr = np.asarray(ix.year + (ix.dayofyear - 1) / 365.25, dtype=float)
        r = (out[k] - out[k].groupby(_cday(ix)).transform("mean")).values
        b = np.polyfit(yr, r, 1)[0]
        trend[k] = 10 * b
        if k in DETREND:
            out[k] = _detrend_to_epoch(out[k], epoch)
            print(
                f"    {k}: trend {trend[k]:+.2f} /decade — climatology referenced to "
                f"epoch {epoch:.1f} (shifts the band {b * (epoch - yr.mean()):+.1f})",
                flush=True,
            )
        else:
            print(
                f"    {k}: trend {trend[k]:+.2f} /decade vs resid sd "
                f"{np.std(r - np.polyval(np.polyfit(yr, r, 1), yr)):.2f} — left as-is",
                flush=True,
            )
    return out, trend


def analysis():
    """The observed tail (see the module docstring): GEOS FP analyses (daily mean of 00/06/12/18Z), u60 at 10 and
    100 hPa and the 100 hPa 65-90N cap height moved to the MERRA-2 scale. Empty when neither committed file is
    present."""
    days = {}
    for p in FP_U_TAIL:
        if p.exists():
            d = json.loads(p.read_text())
            lev = [float(v) for v in d["levels"]]
            for k, v in d["days"].items():
                days[k] = dict(zip(lev, v["nh"]["u60"]))
    out = {}
    if days:
        ks = sorted(days)
        idx = pd.DatetimeIndex(ks)
        for key, lev in (("u10", 10.0), ("u100", 100.0)):
            out[key] = pd.Series([days[k].get(lev) for k in ks], index=idx, dtype=float).dropna()
    cap = geosfp_cap.load_history(FP_CAP_TAIL)
    if len(cap):
        out["zcap"] = (cap[100.0].astype(float) - _fp_offset100(cap.index)).dropna()
    out = {k: s for k, s in out.items() if len(s)}
    if not out:
        print("  no observed tail available (no GEOS FP analysis file)", flush=True)
        return out
    for k, s in out.items():
        print(f"  observed tail {k}: GEOS FP {s.index[0]:%Y-%m-%d}..{s.index[-1]:%Y-%m-%d}", flush=True)
    last = max(s.index[-1] for s in out.values())
    if (pd.Timestamp.now("UTC").tz_localize(None) - last).days > 5:
        print(f"::warning::the GEOS FP analysis tail ends {last:%Y-%m-%d}", flush=True)
    return out


def _consistency(obs, clim, name, unit, tol):
    """Print the GEOS FP tail's mean difference from MERRA-2 over the days both have."""
    if obs is None or clim is None:
        return
    ix = obs.index.intersection(clim.index)
    if len(ix) < 20:
        print(f"  {name}: <20 overlapping days, consistency unchecked", flush=True)
        return
    dd = (obs.loc[ix] - clim.loc[ix]).dropna()
    flag = "  <-- EXCEEDS TOLERANCE" if abs(dd.mean()) > tol else ""
    print(
        f"  {name}: GEOS FP - MERRA-2 {dd.mean():+.2f} {unit} (sd {dd.std():.2f}, "
        f"r {obs.loc[ix].corr(clim.loc[ix]):.3f}, n={len(dd)}){flag}",
        flush=True,
    )


# ── plot ────────────────────────────────────────────────────────────────────
def draw_model(ax, df, key, sub=None):
    col, lab = STYLE[key]
    v = df.values if sub is None else df.values - sub[:, None]
    n = df.shape[1]
    if n >= 5:
        ax.fill_between(
            df.index,
            np.nanpercentile(v, 10, axis=1),
            np.nanpercentile(v, 90, axis=1),
            color=col,
            alpha=0.16,
            lw=0,
            zorder=2,
        )
        ax.plot(df.index, np.nanmean(v, axis=1), color=col, lw=2.4, zorder=5, label=f"{lab} ({n}m)")
    else:
        ax.plot(
            df.index,
            np.nanmean(v, axis=1),
            color=col,
            lw=2.0,
            ls="--",
            zorder=5,
            label=lab if key == "hres" else f"{lab} (det)",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="aifs,gefs,geps,gdps,ifs,hres,geosfp")
    ap.add_argument(
        "--ifs-members", type=int, default=50, help="IFS-ENS perturbed members (all 50 by default; the control is hres)"
    )
    ap.add_argument("--cycle")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--record-only",
        action="store_true",
        help="add this run to the run-to-run record (vortex_runs.py) and draw nothing: a past run, IFS pulled light",
    )
    a = ap.parse_args()
    want = [m.strip() for m in a.models.split(",") if m.strip()]

    cdirs = sorted(glob.glob(str(CACHE / "*z")))
    if a.cycle:
        # Cache dirs are named "20260829 12" + "z" ("2026082912z"). Accept the
        # cycle with or without the trailing z: strat.yml passes date+time with
        # no suffix, and an exact-match filter silently emptied the list, which
        # surfaced as the misleading "no AIFS-ENS cycle in the cache".
        want_c = a.cycle.lower().rstrip("z") + "z"
        cdirs = [d for d in cdirs if Path(d).name.lower() == want_c]
    cdir = base = None
    for d in reversed(cdirs):
        if glob.glob(f"{d}/aifs-ens/pf_u_10-*.grib2"):
            t = Path(d).name
            cdir, base = Path(d), pd.Timestamp(f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00")
            break
    if base is None and a.cycle:
        # No ens_cycle cache for this cycle (strat-ifs.yml): load_aifs pulls what it needs through the store.
        t = a.cycle.lower().rstrip("z")
        cdir, base = CACHE / f"{t}z", pd.Timestamp(f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00")
        print(f"no ens_cycle cache for {t}; AIFS-ENS winds come through the shared store", flush=True)
    if base is None:
        print("no AIFS-ENS cycle in the cache to anchor on")
        return 1
    date, cyc = f"{base:%Y%m%d}", f"{base:%H}"
    epoch = base.year + (base.dayofyear - 1) / 365.25
    print(f"cycle {base:%Y-%m-%d %HZ}  models={want}", flush=True)

    # the GRIB cache is ~250 MB a cycle — keep only this cycle's files
    tag = f"_{date}{cyc}_"
    for f in DL.glob("*.grib2"):
        if tag not in f.name:
            f.unlink(missing_ok=True)

    clim, trend = load_clim(epoch)
    y0 = min(s.index[0] for s in clim.values()).year  # the reference's span, for the legend and the footer
    y1 = max(s.index[-1] for s in clim.values()).year
    obs = analysis()
    _consistency(obs.get("u10"), clim["u10"], "u60 @10 hPa", "m/s", 1.0)
    _consistency(obs.get("zcap"), clim["zcap"], "z100 cap", "m", 60.0)

    fc = {}
    if "aifs" in want:
        fc["aifs"] = load_aifs(cdir, base)
    if "gefs" in want:
        fc["gefs"] = load_gefs(date, cyc)
    if "geps" in want:
        fc["geps"] = load_geps(date, cyc, list(range(0, 385, 24)))
    if "gdps" in want:
        fc["gdps"] = load_gdps(date, cyc, list(range(0, 241, 24)))
    if "ifs" in want:
        fc["ifs"] = load_ifs(date, cyc, list(range(0, 361, 24)), a.ifs_members, light=a.record_only)
    if "hres" in want:
        fc["hres"] = load_hres(date, cyc, list(range(0, 361, 24)))
    if "geosfp" in want:
        fc["geosfp"] = load_geosfp(base)
    fc = {k: v for k, v in fc.items() if v}
    if not fc:
        print("no model data available")
        return 1
    for k, v in fc.items():
        print(
            f"  {STYLE[k][1]}: " + ", ".join(f"{p}={v[p].shape[1]}m x{len(v[p])}" for p in v if v[p] is not None),
            flush=True,
        )

    zmed = clim_doy(clim["zcap"])[1]
    vortex_runs.record(
        base,
        {
            k: {
                "u10": d.get("u10"),
                "zcap": None
                if d.get("zcap") is None
                else d["zcap"].sub(zmed.reindex(_cday(d["zcap"].index)).values, axis=0),
            }
            for k, d in fc.items()
        },
    )
    if a.record_only:
        return 0

    tmax = max(d[p].index[-1] for d in fc.values() for p in d if d[p] is not None)
    t0 = base - pd.Timedelta(days=TAIL_DAYS)
    span = pd.date_range(t0, tmax, freq="D")
    doy = _cday(span)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11.4), sharex=True)
    panels = [
        ("u10", "u at 10 hPa, 60°N  (m s$^{-1}$)", False),
        ("u100", "u at 100 hPa, 60°N  (m s$^{-1}$)", False),
        ("zcap", f"100 hPa {CAP[0]:.0f}–{CAP[1]:.0f}°N height anomaly (m)", True),
    ]
    for ax, (key, ylab, anom) in zip(axes, panels):
        q10, q50, q90 = clim_doy(clim[key])
        b = q50.reindex(doy).values if anom else np.zeros(len(span))
        lab = (
            f"MERRA-2 climatology 10–90% ({y0}–, detrended to {base:%Y})"
            if key in DETREND
            else f"MERRA-2 climatology 10–90% ({y0}–)"
        )
        ax.fill_between(
            span, q10.reindex(doy).values - b, q90.reindex(doy).values - b, color="#e6e3dc", zorder=0, label=lab
        )
        if not anom:
            ax.plot(span, q50.reindex(doy).values, color="#9a958c", ls=":", lw=1.1, zorder=1)
        o = obs.get(key)
        if o is not None:
            o = o[(o.index >= t0) & (o.index <= base)]
            if len(o):
                ob = q50.reindex(_cday(o.index)).values if anom else 0
                ax.plot(o.index, o.values - ob, color="#1a1a1a", lw=2.4, zorder=7, label="GEOS FP analysis")
        for k, d in fc.items():
            df = d.get(key)
            if df is None or not len(df):
                continue
            sub = q50.reindex(_cday(df.index)).values if anom else None
            draw_model(ax, df, k, sub)
        ax.axvline(base, color="#8a8680", lw=1, zorder=3)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=7.4, ncol=3, framealpha=0.9)
        if key == "u10":
            ax.axhline(0, color="#a33", lw=1.5, zorder=4)
            # per ensemble: the share of MEMBERS easterly on at least one day to day 15 (vortex_runs.ssw_chance)
            msg = "easterly at 10 hPa / 60°N = SSW (WMO)"
            hits = {
                k: vortex_runs.ssw_chance(d["u10"], base)
                for k, d in fc.items()
                if d.get("u10") is not None and d["u10"].shape[1] >= 5
            }
            hits = {k: h for k, h in hits.items() if h is not None}
            if hits and max(hits.values()) > 0:
                worst = max(hits, key=hits.get)
                msg = (
                    f"{100 * hits[worst]:.0f}% of {STYLE[worst][1]} members easterly on some day to day "
                    f"{vortex_runs.SSW_DAYS} — easterly here = SSW (WMO)"
                )
            ax.text(0.005, 0.05, msg, transform=ax.transAxes, fontsize=8, color="#a33", style="italic")
        if key == "zcap":
            ax.axhline(0, color="#8a8680", lw=1, ls=":", zorder=4)
            ax.text(
                0.005,
                0.04,
                "positive = weaker / displaced vortex",  # bottom: at the top the legend hid it
                transform=ax.transAxes,
                fontsize=8,
                color="#6f6b64",
                style="italic",
            )
        if key == "u100" and obs.get("u100") is None:
            ax.text(
                0.5,
                0.05,
                "no analysis tail at this level",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
                style="italic",
                color="#6f6b64",
            )

    fig.suptitle(f"Northern polar vortex forecast — {base:%Y-%m-%d %HZ} cycle", fontsize=14, fontweight="bold", y=0.996)
    fig.text(
        0.5,
        0.004,
        "Zonal-mean zonal wind at 60°N and the 65–90°N polar-cap height. Shading is each ensemble's 10th–90th member "
        "percentile, the solid line its mean; dashed = deterministic or control-only.\n"
        "AIFS-ENS (25 members + control), IFS-ENS (50 perturbed members) and the IFS control to day 15, GEFS (30 + "
        "control) and GEPS (20 + control) to day 16, GDPS and GEOS FP (00Z run, NASA GMAO) to day 10.\n"
        f"Reference: MERRA-2 {y0}–{y1} day-of-year percentiles; the height climatology is detrended to the current "
        f"year ({trend['zcap']:+.1f} m/decade), the winds are not "
        f"({max(abs(trend['u10']), abs(trend['u100'])):.2f} m/s/decade or less).\n"
        "Vertical line = analysis time. Black: GEOS FP analyses (daily mean of 00/06/12/18Z). GEOS FP heights are on "
        "the MERRA-2 scale (measured offset removed). IFS control: ECMWF's unperturbed 9 km run (HRES until May 2026).",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.5,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.064, 1, 0.985))  # four footer lines under rotated dates
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=112, facecolor="white", bbox_inches="tight", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {a.out}")
    vortex_runs.render(out=Path(a.out).with_name(vortex_runs.OUT.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
