#!/usr/bin/env python3
"""Stream ERA5 once and store everything the AAM torque budget needs to be REFIT
without ever streaming again.

Why this exists
---------------
Two defects were measured in the existing torque products (2026-08-28):

  1. The friction term uses a bulk stress with constant rho and constant
     C_d = 1.3e-3. Against ERA5's own turbulent surface stress over 1991-2020 that
     approximation is +14.7 Hadley too POSITIVE in the global annual mean (+8.6 vs
     -6.1) and carries only 70% of the true variance (sd 9.2 vs 13.2, r = 0.78).
     One ocean-like drag coefficient under-weights the mid-latitude braking that
     happens over rough land.
  2. build_torque_clim.py samples 12Z only, which aliases the semidiurnal pressure
     tide into the mountain term: the global mountain torque is +4.85 Hadley at 12Z
     but -0.73 at 00Z. That is most of the -4.5 Hadley annual non-closure.

Both are fixed by one pass: sample 00Z AND 12Z, and store the friction density as a
LINEAR BASIS in the unknown drag coefficients so they can be fitted (and refitted)
afterwards with no further download.

    f = -a cos(phi) * rho * |V| * u * C_d,      C_d = Cl*L + Cs*(1-L) + Cs2*|V|*(1-L)
      = Cl*Xl + Cs*Xs + Cs2*Xs2

with L the ERA5 land-sea mask and rho = p_s/(R*T). The three X fields are stored;
any coefficient vector reconstructs the density exactly by a dot product, so the
fit, its validation, and the rebuilt climatology all come from this one file.

rho uses T = 288 K, NOT the ERA5 2 m temperature, because the AIFS-ENS open-data
forecast publishes sp/10u/10v/msl but no 2t at forecast steps -- the reference must
be computable the same way the forecast is. `fric_rhoT` stores the per-latitude
truth with the real 2 m temperature so the cost of that approximation is measurable
rather than assumed.

Outputs ~/mjo/era5_cache/torque_basis_{y0}-{y1}_s{stride}.nc:
  per-latitude (the closure truth):  fric_era5, gwd_era5, mtn_lat, fric_rhoT,
                                     Xl_lat, Xs_lat, Xs2_lat
  5-degree density (for the maps):   mtn5, Xl5, Xs5, Xs25

    python src/build_torque_basis.py --stride 5 --workers 10
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
CACHE = Path.home() / "mjo" / "era5_cache"
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6
HU = 1e18
DEG = np.pi / 180.0
RD = 287.05
TREF = 288.0                      # rho = p_s/(R*TREF): what the forecast side can compute
COARSEN = 20                      # ~5 degrees, matches the torque maps

_ds = None
_lock_msg = []


def _open():
    global _ds
    if _ds is None:
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    return _ds


def _coarse(field2d, lat, lon):
    da = xr.DataArray(field2d, dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    return da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()


def _one(t, lat, lon, orog, lsm, cos, dlam):
    """All per-time diagnostics for one ERA5 timestamp, or None if it is missing."""
    ds = _open()

    # ARCO's grid is the cached orography's grid with latitude reversed and longitude
    # identical (verified exactly), so a numpy flip replaces the per-field reindex.
    # That is not just faster: xarray's index alignment is not thread-safe here and
    # threw sporadic "index has duplicates" errors on a 14-worker pool.
    def g(name):
        return np.asarray(ds[name].sel(time=t).values)[::-1, :]
    try:
        u = g("10m_u_component_of_wind"); v = g("10m_v_component_of_wind")
        sp = g("surface_pressure"); t2 = g("2m_temperature")
        ew = g("mean_eastward_turbulent_surface_stress")
        gw = g("mean_eastward_gravity_wave_surface_stress")
    except Exception as e:                                        # noqa: BLE001
        return t, None, f"{type(e).__name__}: {e}"[:160]

    cos2 = (cos ** 2)[:, None]
    cosc = cos[:, None]
    spd = np.hypot(u, v)
    rho = sp / (RD * TREF)                                        # forecast-computable
    rhoT = sp / (RD * t2)                                         # true-ish, for the penalty

    # friction basis: density = Cl*Xl + Cs*Xs + Cs2*Xs2   [N m per unit area]
    base = -(A * cosc) * rho * spd * u
    Xl = base * lsm
    Xs = base * (1.0 - lsm)
    Xs2 = base * spd * (1.0 - lsm)

    # mountain form drag, +h dp_s/dlam (periodic in longitude)
    dpd = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * dlam)
    mtn = orog * dpd

    dA_lon = dlam                                                 # per-lat integral weight
    out = {
        # per-latitude, Hadley per degree latitude
        "fric_era5": -(A ** 3) * cos2[:, 0] * (ew * dlam).sum(1) * DEG / HU,
        "gwd_era5": -(A ** 3) * cos2[:, 0] * (gw * dlam).sum(1) * DEG / HU,
        "fric_rhoT": (-(A * cosc) * rhoT * spd * u * 1.3e-3 * (A ** 2) * cosc
                      * dA_lon).sum(1) * DEG / HU,
        "mtn_lat": (A ** 2) * cos * mtn.sum(1) * dlam * DEG / HU,
        "Xl_lat": (Xl * (A ** 2) * cosc * dlam).sum(1) * DEG / HU,
        "Xs_lat": (Xs * (A ** 2) * cosc * dlam).sum(1) * DEG / HU,
        "Xs2_lat": (Xs2 * (A ** 2) * cosc * dlam).sum(1) * DEG / HU,
        # 5-degree densities for rebuilding the map climatology
        "mtn5": _coarse(mtn, lat, lon).values,
        "Xl5": _coarse(Xl, lat, lon).values,
        "Xs5": _coarse(Xs, lat, lon).values,
        "Xs25": _coarse(Xs2, lat, lon).values,
    }
    return t, out, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--hours", default="0,12")
    a = ap.parse_args()

    o = xr.open_dataarray(OROG)
    lat = o.latitude.values; lon = o.longitude.values
    orog = o.values
    cos = np.cos(np.deg2rad(lat))
    dlam = np.deg2rad(abs(float(lon[1] - lon[0])))

    ds = _open()
    # ARCO broadcasts the static mask over the full time axis but leaves index 0
    # (pre-record) all-NaN, and stores latitude DESCENDING while the cached orography
    # is ascending -- so isel(time=0) + a plain reindex silently yields an all-NaN mask.
    lsm = ds["land_sea_mask"]
    if "time" in lsm.dims:
        lsm = lsm.sel(time="2000-06-15T12:00", method="nearest")
    lsm = np.asarray(lsm.values)[::-1, :]
    assert np.array_equal(ds.latitude.values[::-1], lat) and np.array_equal(ds.longitude.values, lon), \
        "ARCO grid is no longer a pure latitude flip of the cached orography grid"
    if not np.isfinite(lsm).all() or not (0.2 < lsm.mean() < 0.45):
        raise SystemExit(f"land-sea mask looks wrong: mean {lsm.mean()}, "
                         f"nan {np.isnan(lsm).mean():.3f} -- refusing to build on it")
    print(f"land-sea mask: mean {lsm.mean():.3f}, finite everywhere", flush=True)

    hours = [int(h) for h in a.hours.split(",")]
    days = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D")
    want = pd.DatetimeIndex(sorted(set(
        d + pd.Timedelta(hours=h) for d in days for h in hours)))
    have = pd.to_datetime(ds.time.values)
    times = want[want.isin(have)]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-day stride, hours {hours}, "
          f"{a.y0}-{a.y1}) with {a.workers} workers", flush=True)

    clat = _coarse(orog, lat, lon).latitude.values
    clon = _coarse(orog, lat, lon).longitude.values
    res, bad, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t, lat, lon, orog, lsm, cos, dlam) for t in times]
        for n, f in enumerate(futs):
            t, out, err = f.result()
            if out is None:
                bad += 1
                if bad < 10:
                    print(f"  skip {t}: {err}", flush=True)
                continue
            res[t] = out
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(times)}  {t:%Y-%m-%d %HZ}  "
                      f"{el / 60:.1f} min elapsed, ~{el / max(n, 1) * (len(times) - n) / 60:.0f} min left",
                      flush=True)

    ts = pd.DatetimeIndex(sorted(res))
    latvars = ("fric_era5", "gwd_era5", "fric_rhoT", "mtn_lat", "Xl_lat", "Xs_lat", "Xs2_lat")
    mapvars = ("mtn5", "Xl5", "Xs5", "Xs25")
    dsout = xr.Dataset(
        {k: (("time", "latitude"), np.stack([res[t][k] for t in ts])) for k in latvars}
        | {k: (("time", "clat", "clon"), np.stack([res[t][k] for t in ts])) for k in mapvars},
        coords={"time": ts, "latitude": lat, "clat": clat, "clon": clon},
        attrs={"note": "ERA5 AAM torque basis. friction density = Cl*Xl + Cs*Xs + Cs2*Xs2; "
                       f"rho = p_s/(R*{TREF}K) so the forecast side can reproduce it; "
                       "fric_rhoT is the legacy C_d=1.3e-3 bulk with the real 2m temperature.",
               "hours": str(hours), "stride": a.stride},
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"torque_basis_{a.y0}-{a.y1}_s{a.stride}.nc"
    dsout.to_netcdf(out)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.0f} MB, {len(ts)} times, {bad} skipped)")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
