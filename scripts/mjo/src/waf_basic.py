#!/usr/bin/env python3
"""Low-passed ANALYSIS basic state for the wave-activity flux (waf.py), added 2026-09-07.

TN01 linearises about a slowly varying flow. The ERA5 day-of-year climatology is the textbook
choice, but in a year like 2026 the jet is displaced from the climatological waveguide, so the
packets were being steered by a jet that was not there. This module keeps a rolling history of
the AIFS-ENS 0-h member-0 u, v and ψ at the WAF level on a 3° grid (assets/sst/anim/waf/
waf_basic_history.nc on the frames branch, merged with a committed seed) and returns the
30-day mean anomaly against the climatology. waf.py uses climatology + that anomaly as the
basic state (U, V) and as the reference for ψ′, so perturbation and basic state are consistent.

    python src/waf_basic.py --backfill 45        (laptop: seed from the Google mirror)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))

LAT3 = np.arange(-88.5, 88.6, 3.0)
LON3 = np.arange(0.0, 360.0, 3.0)
WINDOW_DAYS = 30
MIN_ANALYSES = 20
MAXN = 120                                                   # ~60 days of two analyses a day
HIST_NAME = "waf_basic_history.nc"
SEED = Path(__file__).resolve().parent.parent / "data" / "reference" / "waf_basic_history_seed.nc"


def to3(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    da = xr.DataArray(field, coords={"lat": lat, "lon": np.asarray(lon) % 360}, dims=("lat", "lon")).sortby("lon")
    da = da.isel(lon=~pd.Index(da.lon.values).duplicated())          # −180/180 grids fold onto one value
    if da.lat[0] > da.lat[-1]:
        da = da.sortby("lat")
    da = xr.concat([da, da.isel(lon=0).assign_coords(lon=da.lon[0] + 360)], dim="lon")   # cyclic
    return da.interp(lat=LAT3, lon=LON3, kwargs={"fill_value": None}).values.astype("float32")


def from3(field3: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    da = xr.DataArray(field3, coords={"lat": LAT3, "lon": LON3}, dims=("lat", "lon"))
    da = xr.concat([da, da.isel(lon=0).assign_coords(lon=360.0)], dim="lon")
    out = da.interp(lat=np.clip(lat, LAT3[0], LAT3[-1]), lon=lon % 360, kwargs={"fill_value": None}).values
    return out


def record(u3, v3, psi3, valid, level: int) -> xr.Dataset:
    t = [pd.Timestamp(valid)]
    return xr.Dataset({"u": (("time", "lat", "lon"), u3[None]), "v": (("time", "lat", "lon"), v3[None]), "psi": (("time", "lat", "lon"), psi3[None])},
                      coords={"time": t, "lat": LAT3, "lon": LON3},
                      attrs={"level_hPa": level, "note": "AIFS-ENS 0-h member-0 u, v (m/s) and streamfunction psi (m2/s) on a 3 deg grid"})


def load_history(path: Path) -> xr.Dataset | None:
    parts = []
    for p in (path, SEED):
        if p.exists():
            try:
                parts.append(xr.open_dataset(p).load())
            except Exception as ex:                                  # noqa: BLE001
                print(f"  waf_basic: cannot read {p.name} ({str(ex)[:60]})", flush=True)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    a, b = parts
    b = b.sel(time=~b.time.isin(a.time))
    return xr.concat([a, b], dim="time").sortby("time") if b.time.size else a


def update_history(path: Path, cur: xr.Dataset) -> xr.Dataset:
    old = load_history(path)
    if old is not None:
        old = old.sel(time=old.time != cur.time.values[0])
        ds = xr.concat([old, cur], dim="time").sortby("time") if old.time.size else cur
    else:
        ds = cur
    ds = ds.isel(time=slice(-MAXN, None))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("u", "v", "psi")}); tmp.replace(path)
    return ds


def lowpass_anomaly(hist: xr.Dataset, eval_clim, clim, lat, lon, end: pd.Timestamp, days: int = WINDOW_DAYS):
    """Mean over the trailing `days` of (analysis − climatology(doy)) for u, v, ψ, returned on the
    (lat, lon) DH2 grid, plus the number of analyses used. None if too few analyses."""
    t = pd.DatetimeIndex(hist.time.values)
    sel = (t > end - pd.Timedelta(days=days)) & (t <= end)
    if sel.sum() < MIN_ANALYSES:
        return None, int(sel.sum())
    h = hist.isel(time=np.where(sel)[0])
    out = {}
    for k in ("u", "v", "psi"):
        ck = {"u": "U", "v": "V", "psi": "psi"}[k]
        acc = np.zeros((lat.size, lon.size)); n = 0
        for i in range(h.time.size):
            doy = float(pd.Timestamp(h.time.values[i]).dayofyear)
            f = from3(h[k].values[i], lat, lon)
            acc += f - eval_clim(clim[ck].values, doy); n += 1
        out[k] = acc / n
    return out, int(sel.sum())


# ── seed backfill (laptop) ────────────────────────────────────────────────────
def backfill(days: int, level: int) -> int:
    import store as ecmwf
    from waf import streamfunction_psi
    t0 = time.time()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    done = 0
    for d in range(days, -1, -1):
        day = now - timedelta(days=d)
        for hh in ("00", "12"):
            valid = datetime(day.year, day.month, day.day, int(hh))
            if valid > now - timedelta(hours=8):
                continue
            cyc = ecmwf.Cycle(valid.strftime("%Y%m%d"), hh)
            try:
                up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", (level,), (0,)))
                vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", (level,), (0,)))
            except Exception as ex:                                  # noqa: BLE001
                print(f"  {cyc.tag}: not available ({str(ex)[:60]})", flush=True); continue
            u = xr.open_dataset(up, **ku)["u"].squeeze(drop=True); v = xr.open_dataset(vp, **ku)["v"].squeeze(drop=True)
            psi, plat, plon = streamfunction_psi(u, v)
            u3 = to3(u.values, u.latitude.values, u.longitude.values)
            v3 = to3(v.values, v.latitude.values, v.longitude.values)
            update_history(SEED, record(u3, v3, to3(psi, plat, plon), valid, level))
            done += 1
            print(f"  {cyc.tag} ok ({done}, {time.time() - t0:.0f}s)", flush=True)
    h = load_history(SEED)
    print(f"seed: {h.time.size} analyses {str(h.time.values[0])[:13]} … {str(h.time.values[-1])[:13]}, {SEED.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backfill", type=int, default=0); ap.add_argument("--level", type=int, default=250)
    a = ap.parse_args()
    raise SystemExit(backfill(a.backfill, a.level) if a.backfill else 0)
