#!/usr/bin/env python3
"""NH polar vortex FORECAST monitor -- AIFS-ENS, IFS-ENS, the IFS control, AIFS single, GEFS, GEPS, GDPS and GEOS FP.

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

  aifs  AIFS-ENS  50 pf + cf, 0.25 deg, day 15.  u FREE -- ens_cycle --only strat
                  caches pf_u_10-50-100-... every cycle. Height: the 90-65N rows of z at
                  100 hPa, 50 pf + cf (~0.1 GB, store.ensure_rows).
  gefs  GEFS      30 perturbed + control, 0.5 deg, day 16: u at 10/100 hPa and the 100 hPa height
                  from the shared GEFS fetch (strat_ens.py, part "vortex"; NOAA's S3, ~0.6 GB/cycle).
  geps  GEPS      20 members + control, 0.5 deg, day 16, from the same shared fetch's "vortex" part (strat_ens.py;
                  the MSC Datamart), made by the job that renders GEPS's other products.
  gdps  GDPS      deterministic, 0.15 deg, day 10 -> ~31 MB/cycle.
  geosfp GEOS FP   NASA GMAO, deterministic, newest 00Z run (day 10; 06/12/18Z stop at day 5), NCCS OPeNDAP
                  MERRA-2's operational sibling.
  ifs   IFS-ENS   all 50 perturbed members, 0.25 deg, day 15. The ensemble files carry
                  no control at these levels - open data publishes only type=pf there.
                  Read from the shared IFS-ENS files (ifs_ens.py: u at 10/100 hPa and gh
                  at 100 hPa, fetched once a cycle for every stratosphere product, Google
                  Cloud mirror only). IFS-ENS lands on the mirror after AIFS-ENS (day 15
                  ~08:31Z / 20:31Z), so it is recorded in strat-ifs.yml, not strat.yml; a
                  cycle the mirror does not have yet drops IFS with a message.
  hres  IFS control  the unperturbed run (called HRES until May 2026), 0.25 deg, day 15,
                  from its own open-data stream (oper): u and gh at 10/100 hPa from the
                  shared control file (ifs_ens.py ctrl_wind, ~0.1 GB). Drawn dashed, as
                  the other single runs.
  aifsc AIFS-ENS control  member 0 of AIFS-ENS on its own: the "cf" column of the AIFS-ENS frames
                  (no download of its own), recorded and drawn dashed like the IFS control.
  aifs1 AIFS single  ECMWF's deterministic AIFS run, 0.25 deg, day 15: u and gh at 10/100 hPa from the shared
                  file its vortex-wind maps read (aifs_single.py, 12-hourly; nothing fetched when they have run),
                  recorded by its own workflow (strat-aifs1.yml) and drawn dashed.

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

RUN RECORD: every model's run goes into its own record (run_record.py, assets/sst/data/runs/vortex_<model>.json) from
the job that renders that model, as soon as it lands; the figure is drawn from each model's newest record (--from-records,
nothing fetched), so it moves model by model through the cycle.

    python scripts/strat/nh_vortex.py --models ifs,hres --cycle 2026102312 --record-only   # record two models' runs
    python scripts/strat/nh_vortex.py --from-records                                       # the figure

MEMBERS (--members, with --record-only): the job that records an ensemble also draws its run member by member,
vortex_members_{aifs,ifs,gefs,geps}.webp: u60 at 10 and 100 hPa of every member, the members' mean and the control,
the members that turn easterly at 10 hPa by day SSW_DAYS picked out in both panels. The backfill records past runs
without it, so the figures stay on each ensemble's newest run.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes: without it the process aborts at exit)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from quantmet.degdays import doy366

import geosfp_cap
import run_record as RR

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
MEMBERS_OUT = REPO / "assets" / "sst" / "vortex_members_{m}.webp"

PHI = 60.0  # WMO 60N for the SSW criterion
CAP = (65.0, 90.0)  # polar cap for the 100 hPa height anomaly
TAIL_DAYS = 21  # short analysis tail: this is a forecast product
G = 9.80665  # ECMWF `z` is geopotential; `gh` / HGT are already gpm
U_ROW = round((90.0 - PHI) / 0.25)  # 60N as a row of ECMWF's 0.25 deg grid, counted from 90N
CAP_ROWS = tuple(range(round((90.0 - CAP[0]) / 0.25) + 1))  # the cap's rows, 90-65N
DETREND = {"zcap"}  # see the module docstring for why only this one
CONTROLS = ("hres", "aifsc")  # an ensemble's control run, drawn and recorded on its own

STYLE = {
    "aifs": ("#b4541f", "AIFS-ENS"),
    "ifs": ("#1f4b6e", "IFS-ENS"),
    "hres": ("#1f4b6e", "IFS control"),  # dashed, so the ensemble's navy is free for it
    "aifsc": ("#b4541f", "AIFS-ENS control"),  # member 0 of AIFS-ENS on its own, dashed like the IFS control
    "aifs1": ("#af9000", "AIFS single"),  # dark gold: apart from AIFS-ENS orange, GFS olive and GEPS green
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


def _aifs_rows(base, kind: str, short: str, levels: tuple, rows: tuple, reduce, tag: str) -> dict:
    """{level: (time, member) DataFrame} of a per-message reduction of the leading rows of AIFS-ENS fields
    (store.ensure_rows), every daily step."""
    ecmwf = _store()
    spec = ecmwf.Spec("aifs-ens", kind, short, "pl", levels, tuple(ecmwf.STEPS))
    a, _, steps, levs = ecmwf.load_rows(
        ecmwf.ensure_rows(ecmwf.Cycle(f"{base:%Y%m%d}", f"{base:%H}"), spec, rows, reduce=reduce, tag=tag)
    )
    t = base + pd.to_timedelta(steps, unit="h")
    return {L: pd.DataFrame(a[:, :, i].T.astype(float), index=t) for i, L in enumerate(levs)}


def _u60(v, lat, lon):
    return float(v[0].astype(float).mean())


def _zcap(v, lat, lon):
    w = np.cos(np.deg2rad(lat))
    return float((v.astype(float).mean(-1) * w).sum() / w.sum()) / G


def load_aifs(cdir, base):
    pf = glob.glob(f"{cdir}/aifs-ens/pf_u_10-*.grib2")
    cf = glob.glob(f"{cdir}/aifs-ens/cf_u_10-*.grib2")
    out = {}
    try:
        if pf and cf:  # the 12-level u pull ens_cycle caches every cycle
            for lev, key in ((10, "u10"), (100, "u100")):
                d = _frame(_grib(pf[0], shortName="u", level=lev)["u"], zonal_at, base)
                d["cf"] = _frame(_grib(cf[0], shortName="u", level=lev)["u"], zonal_at, base)[0]
                out[key] = d
        else:  # no ens_cycle cache (a backfill): the 60N row alone, ~20 % of each message
            u = {k: _aifs_rows(base, k, "u", (10, 100), (U_ROW,), _u60, "u60") for k in ("pf", "cf")}
            for lev, key in ((10, "u10"), (100, "u100")):
                out[key] = u["pf"][lev].assign(cf=u["cf"][lev][0])
    except Exception as e:  # noqa: BLE001
        print(f"  AIFS-ENS u unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
        return None
    # The 100 hPa cap height of the control and the 50 perturbed members from the 90-65N rows (~12 % of each message);
    # a failed perturbed pull leaves the control alone.
    zc = zp = None
    try:
        zc = _aifs_rows(base, "cf", "z", (100,), CAP_ROWS, _zcap, "cap")[100]
        zp = _aifs_rows(base, "pf", "z", (100,), CAP_ROWS, _zcap, "cap")[100]
    except Exception as e:  # noqa: BLE001
        print(f"  AIFS-ENS z100 unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
    if zp is not None and zc is not None:
        zp["cf"] = zc[0].reindex(zp.index)
    out["zcap"] = zp if zp is not None else zc
    return out


def load_aifs1(base):
    """AIFS single at the daily steps: u at 10/100 hPa and gh at 100 hPa from the shared file its vortex-wind maps read
    (aifs_single.py, 12-hourly). Google Cloud mirror only; a run not there yet is not recorded."""
    import aifs_single as A1

    ecmwf = _store()
    try:
        path = ecmwf.ensure(ecmwf.Cycle(f"{base:%Y%m%d}", f"{base:%H}"), A1.spec_wind())
    except Exception as e:  # noqa: BLE001
        print(f"::warning::AIFS single unavailable ({type(e).__name__}: {str(e)[:100]}); not recorded", flush=True)
        return None
    out = {}
    for key, short, lev, red in (
        ("u10", "u", 10, zonal_at),
        ("u100", "u", 100, zonal_at),
        ("zcap", "gh", 100, cap_mean),
    ):
        da = _grib(path, shortName=short, level=lev)[short]
        h = (da.step.values / np.timedelta64(1, "h")).astype(int)
        out[key] = _frame(da.isel(step=np.flatnonzero(h % 24 == 0)), red, base)
    return out


GDPS_URL = (
    datamart.BASE + "/{d}/WXO-DD/model_gdps/15km/{c}/{L:03d}/"
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
            try:
                rows[base + pd.Timedelta(hours=L)] = np.atleast_1d(np.asarray(red(_first(_grib(f))).values)).ravel()
            except Exception as e:  # noqa: BLE001
                print(f"    {kind} {var}{lev} +{L}h unreadable: {str(e)[:50]}", flush=True)
        if not rows:
            return None
        out[key] = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return out


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


def load_ens(model, date, cyc):
    """GEFS (control + 30 members) or GEPS (control + 20), to day 16, from the shared fetch's "vortex" part
    (strat_ens.py: u at 10 and 100 hPa and the 100 hPa height, daily; NOAA's S3, the MSC Datamart), which the job that
    renders the model's other products has already made. A cycle not complete falls back to the newest complete one
    within a day; with none the model is not recorded this run."""
    import strat_ens as SE

    try:
        d0, h0 = SE.resolve(model, date, cyc)
        d = SE.load(model, d0, h0, ("vortex",))
    except SystemExit as e:
        print(f"::warning::{STYLE[model][1]} unavailable ({e}); not recorded this run", flush=True)
        return None
    keep = SE.complete_members(d, "zcap")
    idx = pd.Timestamp(f"{d0[:4]}-{d0[4:6]}-{d0[6:]} {h0}:00") + pd.to_timedelta(list(SE.LEADS_VORTEX), unit="h")
    return {  # columns = member numbers: 0 is the control
        "u10": pd.DataFrame(d["u60"][keep, :, 0].T, index=idx, columns=keep),
        "u100": pd.DataFrame(d["u60"][keep, :, 1].T, index=idx, columns=keep),
        "zcap": pd.DataFrame(d["zcap"][keep].T, index=idx, columns=keep),
    }


def load_ifs(date, cyc, leads, members, light=False):
    """IFS-ENS, all 50 perturbed members (or the first `members`), from the shared IFS-ENS files.

    ifs_ens.py stocks u/v at 10/100 hPa (12-hourly) and gh at 10/100 hPa (daily) ONCE a cycle for every
    stratosphere product; this reads u at 10/100 and gh at 100 hPa out of them at the daily steps. `light` (a past run
    recorded on its own, --record-only) pulls just those fields, ~1.3 GB instead of the bundle's ~7.5, unless the
    shared files are already on disk (the job that renders IFS-ENS stocks them first). Google Cloud mirror only: a
    cycle not yet on Google is dropped rather than pulled slowly from another mirror.
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
        if light and not all(IE.E.path(cycle, spec()).exists() for spec in (IE.spec_uv, IE.spec_gh_wind)):
            uv = IE.ensure(cycle, IE.E.Spec("ifs", "pf", "u", "pl", (10, 100), IE.STEPS))
            gh = IE.ensure(cycle, IE.E.Spec("ifs", "pf", "gh", "pl", (100,), IE.STEPS))
        else:
            uv = IE.ensure(cycle, IE.spec_uv())
            gh = IE.ensure(cycle, IE.spec_gh_wind())
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
        if tt[0] != init or not (tt.is_monotonic_increasing and tt.is_unique):
            print(f"::warning::GEOS FP {init:%Y%m%d} 00Z: time axis corrupt (starts {tt[0]}); skipped", flush=True)
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


# ── the run record ──────────────────────────────────────────────────────────
SSW_DAYS = 15
MAX_AGE_DAYS = 2  # a model's newest recorded run older than this against the newest of all is left off the figure


def ssw_chance(u10: pd.DataFrame, init: pd.Timestamp) -> float | None:
    """Share of members easterly at 10 hPa / 60N on at least one day to day SSW_DAYS, over the members with a value in
    that window (None when none has)."""
    lead = u10[u10.index <= init + pd.Timedelta(days=SSW_DAYS)].dropna(axis=1, how="all")
    return float((lead.to_numpy() < 0).any(axis=0).mean()) if lead.shape[1] else None


def record_runs(fc: dict, zmed: pd.Series) -> None:
    """Each model's run into its own record (run_record.py): u60 at 10 and 100 hPa and the cap-height anomaly against
    the detrended MERRA-2 median, the members' mean and 10-90 %, and an ensemble's sudden-warming chance. A run is
    filed under its own start (its first valid time: GEOS FP's newest 00Z run, a GEFS run that fell back a cycle)."""
    for m, d in fc.items():
        u = d.get("u10")
        if u is None or not len(u):
            continue
        init = u.index[0]
        z = d.get("zcap")
        frames = {
            "u10": u,
            "u100": d.get("u100"),
            "zcap": None if z is None else z.sub(zmed.reindex(_cday(z.index)).values, axis=0),
        }
        ssw = ssw_chance(u, init) if u.shape[1] >= 5 else None
        RR.record("vortex", m, init, RR.entry("vortex", frames, {"ssw": None if ssw is None else round(ssw, 3)}))


def newest_runs() -> dict:
    """{model: run}: every model's newest recorded run, those within MAX_AGE_DAYS of the newest of all."""
    out = {m: RR.runs("vortex", m)[-1] for m in STYLE if RR.runs("vortex", m)}
    if not out:
        return {}
    t1 = max(pd.Timestamp(r["init"]) for r in out.values())
    return {m: r for m, r in out.items() if pd.Timestamp(r["init"]) >= t1 - pd.Timedelta(days=MAX_AGE_DAYS)}


# ── plot ────────────────────────────────────────────────────────────────────
def draw_model(ax, run, key, m):
    """One model's newest run: an ensemble's members' 10-90 % and mean, a single run dashed; labelled with its start."""
    if key not in run:
        return
    col, lab = STYLE[m]
    init = pd.Timestamp(run["init"])
    mean = RR.series(run, key)
    if run["members"] >= 5:
        ax.fill_between(
            mean.index, RR.series(run, key, "p10"), RR.series(run, key, "p90"), color=col, alpha=0.16, lw=0, zorder=2
        )
        ax.plot(mean.index, mean.values, color=col, lw=2.4, zorder=5, label=f"{lab} {init:%-d/%HZ} ({run['members']}m)")
    else:
        ax.plot(mean.index, mean.values, color=col, lw=2.0, ls="--", zorder=5,
                label=f"{lab} {init:%-d/%HZ}" + ("" if m in CONTROLS else " (det)"))  # fmt: skip


MEMBER_MODELS = ("aifs", "ifs", "gefs", "geps")  # the ensembles drawn member by member (--members)
CONTROL_COLUMN = {"aifs": "cf", "gefs": 0, "geps": 0}  # the control's column in the ensemble's frames; IFS's is hres
MEMBER_TAIL_DAYS = 7
EAST = "#b8322a"  # members easterly at 10 hPa by day SSW_DAYS
MEMBER = "#8f8a80"  # the other members: neutral, so the highlight reads against every model's colour


def draw_members(m: str, fc: dict, clim: dict, obs: dict, out: Path) -> bool:
    """One ensemble's run member by member: u60 at 10 and 100 hPa of every member (thin), the members' mean (bold), the
    control (dashed), against the MERRA-2 day-of-year 10-90 % and the GEOS FP analysis tail. The members easterly at
    10 hPa on some day to day SSW_DAYS are drawn red in both panels."""
    d = fc[m]
    u10 = d.get("u10")
    if u10 is None or u10.shape[1] < 5:
        return False
    col, lab = STYLE[m]
    init = u10.index[0]
    ccol = CONTROL_COLUMN.get(m)
    ctrl = {k: v[ccol] for k, v in d.items() if v is not None and ccol in v} if ccol is not None else {}
    if m == "ifs" and fc.get("hres"):
        ctrl = {k: v[0] for k, v in fc["hres"].items() if v is not None}
    cut = init + pd.Timedelta(days=SSW_DAYS)
    east = (u10[u10.index <= cut] < 0).any(axis=0)
    n, k = len(east), int(east.sum())
    inc = ccol is not None and ccol in u10  # the control is one of the n
    c10 = ctrl.get("u10")
    ceast = c10 is not None and bool((c10[c10.index <= cut] < 0).any())
    t0 = init - pd.Timedelta(days=MEMBER_TAIL_DAYS)
    t1 = max(f.index[-1] for f in (d.get("u10"), d.get("u100")) if f is not None)
    span = pd.date_range(t0.normalize(), t1.ceil("D"), freq="D")
    y0 = min(s.index[0] for s in clim.values()).year
    y1 = max(s.index[-1] for s in clim.values()).year

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True)
    for ax, key, lev in zip(axes, ("u10", "u100"), (10, 100)):
        q10, q50, q90 = (q.reindex(_cday(span)).values for q in clim_doy(clim[key]))
        ax.fill_between(span, q10, q90, color="#e6e3dc", lw=0, zorder=0)
        ax.plot(span, q50, color="#9a958c", ls=":", lw=1.1, zorder=1)
        f = d.get(key)
        if f is None:
            ax.text(0.5, 0.5, f"no {lev} hPa wind in this run", transform=ax.transAxes, ha="center", color="#6f6b64")
            continue
        for c in f.columns:
            if c == ccol:
                continue
            s = f[c].dropna()
            e = bool(east.get(c, False))
            ax.plot(s.index, s.values, color=EAST if e else MEMBER, lw=0.9 if e else 0.7, alpha=0.8 if e else 0.5,
                    zorder=3 if e else 2)  # fmt: skip
        if key in ctrl:
            s = ctrl[key].dropna()
            ax.plot(s.index, s.values, color=EAST if ceast else "#3d3a35", lw=1.6, ls="--", zorder=5)
        mean = f.mean(axis=1)
        ax.plot(mean.index, mean.values, color=col, lw=2.8, zorder=6,
                path_effects=[patheffects.Stroke(linewidth=4.8, foreground="white"), patheffects.Normal()])  # fmt: skip
        o = obs.get(key)
        if o is not None:
            o = o[(o.index >= t0) & (o.index <= init)]
            ax.plot(o.index, o.values, color="#1a1a1a", lw=2.4, zorder=7)
        ax.axvline(init, color="#8a8680", lw=1, zorder=4)
        if key == "u10":
            ax.axhline(0, color="#a33", lw=1.5, zorder=4)
            ax.text(0.005, 0.04, f"{k} of {n} members{' (control included)' * inc} easterly on some day to day {SSW_DAYS} (red) — easterly here "
                    "= SSW (WMO)", transform=ax.transAxes, fontsize=8.5, color="#a33", style="italic", zorder=8,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})  # fmt: skip
        elif ax.get_ylim()[0] < 0:
            ax.axhline(0, color="#8a8680", lw=1, ls=":", zorder=4)
        ax.set_ylabel(f"u at {lev} hPa, 60°N  (m s$^{{-1}}$)")
        ax.grid(alpha=0.22)
    handles = [
        Line2D([], [], color=MEMBER, lw=0.9, alpha=0.7, label=f"members ({n}{', control included' * inc})"),
        Line2D([], [], color=EAST, lw=1.1, label=f"members easterly at 10 hPa by day {SSW_DAYS} ({k})"),
        Line2D([], [], color=col, lw=2.8, label="members' mean"),
    ]
    if ctrl:
        handles.append(
            Line2D([], [], color="#3d3a35", lw=1.6, ls="--", label="IFS control" if m == "ifs" else "control")
        )
    handles += [
        Line2D([], [], color="#1a1a1a", lw=2.4, label="GEOS FP analysis"),
        Line2D([], [], color="#e6e3dc", lw=7, label=f"MERRA-2 {y0}–{y1} 10–90% (dotted: median)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7.6, ncol=3, framealpha=0.9)
    axes[0].set_xlim(t0, t1)
    fig.suptitle(f"{lab} {init:%Y-%m-%d %HZ}: every member's polar-vortex wind (zonal mean at 60°N)", fontsize=14,
                 fontweight="bold", y=0.995)  # fmt: skip
    fig.text(0.5, 0.004, "Vertical line = the start; to its left the GEOS FP analyses (daily mean of 00/06/12/18Z). "
             "Raw model output.", ha="center", va="bottom", fontsize=8, color="#6f6b64")  # fmt: skip
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, facecolor="white", bbox_inches="tight", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out} ({k} of {n} members easterly by day {SSW_DAYS})", flush=True)
    return True


def render(out: Path) -> int:
    """The figure from every model's newest recorded run (no download): each model on its own start."""
    runs = newest_runs()
    if not runs:
        print("no recorded runs to draw")
        return 1
    base = max(pd.Timestamp(r["init"]) for r in runs.values())
    epoch = base.year + (base.dayofyear - 1) / 365.25
    clim, trend = load_clim(epoch)
    y0 = min(s.index[0] for s in clim.values()).year  # the reference's span, for the legend and the footer
    y1 = max(s.index[-1] for s in clim.values()).year
    obs = analysis()
    _consistency(obs.get("u10"), clim["u10"], "u60 @10 hPa", "m/s", 1.0)
    _consistency(obs.get("zcap"), clim["zcap"], "z100 cap", "m", 60.0)
    tmax = max(pd.Timestamp(r["valid"][-1]) for r in runs.values())
    t0 = base - pd.Timedelta(days=TAIL_DAYS)
    span = pd.date_range(t0, tmax, freq="D")
    doy = _cday(span)
    print("  runs: " + ", ".join(f"{STYLE[m][1]} {r['init']}Z" for m, r in runs.items()), flush=True)

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
        for m, run in runs.items():
            draw_model(ax, run, key, m)  # the record's cap height is already the anomaly
        ax.axvline(base, color="#8a8680", lw=1, zorder=3)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=7.4, ncol=3, framealpha=0.9)
        if key == "u10":
            ax.axhline(0, color="#a33", lw=1.5, zorder=4)
            msg = "easterly at 10 hPa / 60°N = SSW (WMO)"
            hits = {m: r["ssw"] for m, r in runs.items() if r.get("ssw") is not None}
            if hits and max(hits.values()) > 0:
                worst = max(hits, key=hits.get)
                msg = (
                    f"{100 * hits[worst]:.0f}% of {STYLE[worst][1]} members easterly on some day to day "
                    f"{SSW_DAYS} — easterly here = SSW (WMO)"
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

    fig.suptitle(
        f"Northern polar vortex forecast — the newest run of each model (to {base:%Y-%m-%d %HZ})",
        fontsize=14,
        fontweight="bold",
        y=0.996,
    )
    fig.text(
        0.5,
        0.004,
        "Zonal-mean zonal wind at 60°N and the 65–90°N polar-cap height. Shading is each ensemble's 10th–90th member "
        "percentile, the solid line its mean; dashed = deterministic or control-only. Each model is its newest run "
        "(start in the legend, day/hour UTC).\n"
        "AIFS-ENS (50 members + control), IFS-ENS (50 perturbed members), the IFS control and AIFS single to day 15, "
        "GEFS (30 + control) and GEPS (20 + control) to day 16, GDPS and GEOS FP (00Z run, NASA GMAO) to day 10.\n"
        f"Reference: MERRA-2 {y0}–{y1} day-of-year percentiles; the height climatology is detrended to the current "
        f"year ({trend['zcap']:+.1f} m/decade), the winds are not "
        f"({max(abs(trend['u10']), abs(trend['u100'])):.2f} m/s/decade or less).\n"
        "Vertical line = the newest start. Black: GEOS FP analyses (daily mean of 00/06/12/18Z). GEOS FP heights are on "
        "the MERRA-2 scale (measured offset removed). IFS control: ECMWF's unperturbed 9 km run (HRES until May 2026).",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.5,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.064, 1, 0.985))  # four footer lines under rotated dates
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, facecolor="white", bbox_inches="tight", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="aifs,aifsc,aifs1,gefs,geps,gdps,ifs,hres,geosfp")
    ap.add_argument(
        "--ifs-members", type=int, default=50, help="IFS-ENS perturbed members (all 50 by default; the control is hres)"
    )
    ap.add_argument("--cycle")
    ap.add_argument("--out", default=str(OUT))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--record-only",
        action="store_true",
        help="add these models' runs to their records (run_record.py) and draw nothing",
    )
    mode.add_argument(
        "--from-records", action="store_true", help="draw the figure from every model's newest record, nothing fetched"
    )
    ap.add_argument(
        "--members",
        action="store_true",
        help="also draw each ensemble's run member by member (vortex_members_{m}.webp)",
    )
    a = ap.parse_args()
    if a.from_records:
        return render(Path(a.out))
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
        # No ens_cycle cache for this cycle: load_aifs pulls what it needs through the store.
        t = a.cycle.lower().rstrip("z")
        cdir, base = CACHE / f"{t}z", pd.Timestamp(f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00")
        if {"aifs", "aifsc"} & set(want):
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

    clim, _ = load_clim(epoch)
    fc = {}
    if "aifs" in want or "aifsc" in want:
        fc["aifs"] = load_aifs(cdir, base)
        if "aifsc" in want and fc["aifs"]:  # the control's own column of the AIFS-ENS frames
            fc["aifsc"] = {k: v[["cf"]] for k, v in fc["aifs"].items() if v is not None and "cf" in v}
        if "aifs" not in want:
            del fc["aifs"]
    if "aifs1" in want:
        fc["aifs1"] = load_aifs1(base)
    for m in ("gefs", "geps"):
        if m in want:
            fc[m] = load_ens(m, date, cyc)
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
    record_runs(fc, clim_doy(clim["zcap"])[1])
    if a.members:
        obs = analysis()
        for m in MEMBER_MODELS:
            if m in fc:
                draw_members(m, fc, clim, obs, Path(str(MEMBERS_OUT).format(m=m)))
    return 0 if a.record_only else render(Path(a.out))


if __name__ == "__main__":
    raise SystemExit(main())
