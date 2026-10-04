#!/usr/bin/env python3
"""Derive an effective drag-coefficient FIELD from ERA5's own surface layer.

Replaces both the old constant C_d = 1.3e-3 and the 10-degree-band fit that
succeeded it. Neither was the right object: the drag coefficient is a property
of the surface and the boundary layer, so it belongs on the map, not in a
latitude band.

ERA5 publishes what the IFS boundary-layer scheme actually did --
`mean_eastward_turbulent_surface_stress` -- so rather than guessing C_d we solve
for the value that best reproduces that stress from the bulk kernel the forecast
side is able to evaluate:

    tau_lambda ~= -C_d * rho * |V10| * u10

Per grid cell, a least-squares fit through the origin over all sampled times in
a calendar month:

    C_d(month, y, x) = -sum(tau * k) / sum(k^2),   k = rho |V10| u10

which is well conditioned where the bulk kernel has any amplitude, and is
skipped (filled from the annual mean) where it does not. This absorbs surface
roughness, mean stability and gustiness together -- everything the single
coefficient was missing -- and it stays applicable to the forecast, which only
ever has 10 m winds and surface pressure.

rho = p_s/(R*288K), matching build_torque_basis.py, because the AIFS-ENS open
data has no 2 m temperature at forecast steps.

Output: data/reference/drag_cd_field.nc -- C_d(month, latitude, longitude) on
the ~5 degree torque-map grid, plus the sample count and fit quality per cell.

    python src/build_drag_field.py --stride 10 --workers 14
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
OUT = REF / "drag_cd_field.nc"
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
RD = 287.05
TREF = 288.0
COARSEN = 20                      # ~5 deg, the torque-map grid

_ds = None


def _open():
    global _ds
    if _ds is None:
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    return _ds


def _coarse(f, lat, lon):
    return xr.DataArray(f, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon}
                        ).coarsen(latitude=COARSEN, longitude=COARSEN,
                                  boundary="trim").mean().values


def _one(t, lat, lon):
    """Coarsened (tau*k, k^2, tau^2) for one time — the pieces of the fit."""
    ds = _open()

    def g(name):
        return np.asarray(ds[name].sel(time=t).values)[::-1, :]
    try:
        u = g("10m_u_component_of_wind"); v = g("10m_v_component_of_wind")
        sp = g("surface_pressure"); tau = g("mean_eastward_turbulent_surface_stress")
    except Exception as e:                                    # noqa: BLE001
        return t, None, f"{type(e).__name__}: {e}"[:120]
    k = (sp / (RD * TREF)) * np.hypot(u, v) * u               # the bulk kernel per unit C_d
    return t, (_coarse(tau * k, lat, lon), _coarse(k * k, lat, lon),
               _coarse(tau * tau, lat, lon)), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--hours", default="0,12")
    a = ap.parse_args()

    o = xr.open_dataarray(OROG)
    lat, lon = o.latitude.values, o.longitude.values
    ds = _open()
    assert np.array_equal(ds.latitude.values[::-1], lat), "ARCO grid changed"

    hours = [int(h) for h in a.hours.split(",")]
    days = pd.date_range(f"{a.y0}-01-01", f"{a.y1}-12-31", freq=f"{a.stride}D")
    want = pd.DatetimeIndex(sorted(set(d + pd.Timedelta(hours=h)
                                       for d in days for h in hours)))
    times = want[want.isin(pd.to_datetime(ds.time.values))]
    print(f"sampling {len(times)} ERA5 times ({a.stride}-d stride, hours {hours}, "
          f"{a.y0}-{a.y1}) with {a.workers} workers", flush=True)

    shape = _coarse(o.values, lat, lon).shape
    num = np.zeros((12,) + shape)      # sum(tau*k)
    den = np.zeros((12,) + shape)      # sum(k^2)
    tt = np.zeros((12,) + shape)       # sum(tau^2), for the fit-quality diagnostic
    cnt = np.zeros(12, dtype=int)
    bad, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t, lat, lon) for t in times]
        for n, f in enumerate(futs):
            t, out, err = f.result()
            if out is None:
                bad += 1
                if bad < 6:
                    print(f"  skip {t}: {err}", flush=True)
                continue
            m = t.month - 1
            num[m] += out[0]; den[m] += out[1]; tt[m] += out[2]; cnt[m] += 1
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(times)}  {t:%Y-%m-%d %HZ}  {el/60:.1f} min, "
                      f"~{el/max(n,1)*(len(times)-n)/60:.0f} min left", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        # ERA5 metss is the stress ON THE SURFACE, so it carries the same sign as
        # the wind: tau = +C_d * k. The minus that makes it a torque on the
        # ATMOSPHERE belongs in the torque conversion, not here.
        cd = num / den                                   # LS through the origin
        # fraction of the stress variance the fitted kernel explains, per cell
        r2 = 1.0 - (tt - num**2 / np.where(den > 0, den, np.nan)) / np.where(tt > 0, tt, np.nan)
    good = np.isfinite(cd) & (cd > 1e-5) & (cd < 0.05)   # physical range for a 10 m C_d
    ann = np.nanmean(np.where(good, cd, np.nan), axis=0)
    fill = np.where(np.isfinite(ann), ann, 1.3e-3)       # last resort: the old constant
    cd = np.where(good, cd, fill[None])

    clat = np.linspace(-90, 90, shape[0] * COARSEN + 1)[COARSEN // 2::COARSEN][:shape[0]]
    cgrid = _coarse(o.values, lat, lon)
    ref = xr.DataArray(cgrid, dims=("latitude", "longitude"))
    lat5 = xr.DataArray(o.values, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon}
                        ).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    xr.Dataset(
        {"cd": (("month", "latitude", "longitude"), cd.astype("float32")),
         "r2": (("month", "latitude", "longitude"), r2.astype("float32")),
         "n": ("month", cnt)},
        coords={"month": np.arange(1, 13), "latitude": lat5.latitude.values,
                "longitude": lat5.longitude.values},
        attrs={"note": ("Effective surface drag coefficient solved per grid cell and "
                        "calendar month so the bulk kernel -C_d rho |V10| u10 best "
                        "reproduces ERA5 mean_eastward_turbulent_surface_stress. "
                        "Absorbs roughness, mean stability and gustiness together."),
               "rho": f"p_s/(R*{TREF}K) — matches the forecast side, which has no 2m T",
               "years": f"{a.y0}-{a.y1}", "stride_days": a.stride,
               "hours": str(hours), "skipped": bad},
    ).to_netcdf(OUT)
    v = cd[np.isfinite(cd)]
    print(f"\nwrote {OUT}")
    print(f"  C_d range {v.min():.2e}..{v.max():.2e}, median {np.median(v):.2e}")
    print(f"  {len(times)} times, {bad} skipped, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
