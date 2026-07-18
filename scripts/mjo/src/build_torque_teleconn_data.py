#!/usr/bin/env python3
"""Build the ERA5 cache for the AAM-torque → jet/teleconnection lag-regression.

Two artifacts in ~/mjo/era5_cache/, both on a common daily-12Z calendar:

  torque_indices_<y0>_<y1>.nc   PREDICTORS (heavy ARCO 0.25° pull)
      Daily scalar surface-torque indices ON the atmosphere, same physics + sign
      convention as src/build_torque_clim.py (eastward = adds westerly AAM):
        mountain (form drag)  density = +(a²) cosφ · h·∂p_s/∂λ · dλ · DEG / HU
        friction (turbulent)  density = -(a³) cos²φ · τ_turb · dλ · DEG / HU
      reduced to indices: regional mountain over the orographic-barrier boxes
      (Himalaya/Tibet, Rockies, Andes), global mountain, global/NH/SH friction.
      Absolute units are irrelevant downstream (each index is standardized before
      regression) — only the sign/relative structure matters, hence we copy the
      verified build_torque_clim formulas verbatim.

  uz_fields_<y0>_<y1>.nc        RESPONSES (cheap WB2 1.5° pull)
      Daily-12Z u250 (jet) and z500 (=geopotential/g; teleconnection), 1.5°,
      latitude 20°S–87.5°N (tropics + NH jet/PNA), float32.

Heavy step is the ARCO predictor loop (~16k daily reads for 1979–2023). It is a
big GCS read → run OFF-PEAK only (never during an MJO cycle; honours the ARCO
bandwidth constraint). Both artifacts are write-once + cached, so the analysis
script never re-pulls.

    # quick code/units smoke (near-zero bandwidth):
    python src/build_torque_teleconn_data.py --y0 2020 --y1 2020 --stride 12 --which both
    # off-peak prototype, then the full overnight build:
    python src/build_torque_teleconn_data.py --y0 2010 --y1 2023 --which torque
    python src/build_torque_teleconn_data.py --y0 1979 --y1 2023 --which both
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
CACHE = Path.home() / "mjo" / "era5_cache"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

A = 6.371e6
HU = 1e18
DEG = np.pi / 180.0
G = 9.80665

# Orographic-barrier boxes (lon0, lon1, lat0, lat1) on 0..360°E — copied from
# src/torque_map_anim.py:RANGES (the regional mountain-torque predictors).
RANGES = {
    "himalaya": (70, 105, 25, 45),
    "rockies": (232, 258, 30, 62),
    "andes": (282, 296, -56, 12),
}

# Response field latitude window (tropics + NH jet / PNA)
LAT_LO, LAT_HI = -20.0, 87.5


def _times(ds, y0: int, y1: int, stride: int) -> pd.DatetimeIndex:
    want = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq=f"{stride}D") + pd.Timedelta(hours=12)
    return want[want.isin(pd.to_datetime(ds.time.values))]


# ── resumable per-year checkpointing ───────────────────────────────────────────
# Each year is written atomically (tmp → rename) to a per-year file, so the build
# can be killed at any time (e.g. to free bandwidth for an MJO cycle) and re-run —
# completed years are skipped and it picks up where it left off.

def _atomic_save(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)                                    # atomic: never leaves a partial final


def _year_ok(path: Path, min_days: int = 360) -> bool:
    """A completed daily year (skip it). Partial/corrupt/smoke files fail → rebuilt."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as d:
            return d.sizes.get("time", 0) >= min_days
    except Exception:                                        # noqa: BLE001 — corrupt → rebuild
        return False


def _assemble(ydir: Path, y0: int, y1: int, out: Path, attrs: dict) -> Path:
    paths = [ydir / f"{y}.nc" for y in range(y0, y1 + 1) if (ydir / f"{y}.nc").exists()]
    ds = xr.concat([xr.open_dataset(p) for p in paths], dim="time")
    ds.attrs.update(attrs)
    _atomic_save(ds, out)
    print(f"assembled {out.name}  ({ds.sizes['time']} days from {len(paths)} years)", flush=True)
    return out


def build_torque_indices(y0: int, y1: int, stride: int) -> Path:
    """Vectorized year-by-year so dask batches/parallelizes the ARCO chunk reads
    (serial .sel(time=t) per day was ~14 h for 1979-2023; this is ~1-2 h)."""
    import gcsfs

    o = xr.open_dataarray(OROG)
    lat = o.latitude.values
    lon = o.longitude.values
    dlam = np.deg2rad(abs(float(lon[1] - lon[0])))
    dphi = np.deg2rad(abs(float(lat[1] - lat[0])))
    # orography height + cosφ as DataArrays on the target grid (broadcast over time)
    h = xr.DataArray(o.values, dims=["latitude", "longitude"],
                     coords={"latitude": lat, "longitude": lon})
    cosx = xr.DataArray(np.cos(np.deg2rad(lat)), dims=["latitude"], coords={"latitude": lat})

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(ARCO), chunks={"time": 48})

    def reduce_year(times: pd.DatetimeIndex) -> xr.Dataset:
        ew = ds["mean_eastward_turbulent_surface_stress"].sel(time=times) \
            .reindex(latitude=lat, longitude=lon, method="nearest")
        sp = ds["surface_pressure"].sel(time=times) \
            .reindex(latitude=lat, longitude=lon, method="nearest")
        # periodic ∂p_s/∂λ (matches np.roll(sp,∓1,axis=lon)); roll_coords=False keeps the coord
        dpdlam = (sp.roll(longitude=-1, roll_coords=False)
                  - sp.roll(longitude=1, roll_coords=False)) / (2 * dlam)
        m2d = (A ** 2) * cosx * (h * dpdlam) * dlam * DEG / HU      # (time,lat,lon) density
        f2d = -(A ** 3) * (cosx ** 2) * ew * dlam * DEG / HU
        D = ["latitude", "longitude"]
        idx = {
            "mtn_global": m2d.sum(D) * dphi,
            "fric_global": f2d.sum(D) * dphi,
            "fric_nh": f2d.sel(latitude=slice(0, 90)).sum(D) * dphi,
            "fric_sh": f2d.sel(latitude=slice(-90, 0)).sum(D) * dphi,
        }
        for k, (lo0, lo1, la0, la1) in RANGES.items():
            idx[f"mtn_{k}"] = (m2d.sel(latitude=slice(la0, la1),
                                       longitude=slice(lo0, lo1)).sum(D) * dphi)
        return xr.Dataset(idx).compute()

    ydir = CACHE / "torque_years"
    print(f"torque: {y1 - y0 + 1} years {y0}-{y1} (stride {stride}) → {ydir}", flush=True)
    for y in range(y0, y1 + 1):
        yp = ydir / f"{y}.nc"
        if _year_ok(yp):
            print(f"  {y}: cached ✓ (skip)", flush=True)
            continue
        times = _times(ds, y, y, stride)
        if len(times) == 0:
            continue
        t0 = time.time()
        print(f"  {y}: reducing {len(times)} days …", flush=True)
        _atomic_save(reduce_year(times), yp)
        print(f"  {y}: saved → {yp.name}  ({time.time() - t0:.0f}s)", flush=True)

    return _assemble(ydir, y0, y1, CACHE / f"torque_indices_{y0}_{y1}.nc",
                     attrs=dict(note="AAM surface-torque indices ON atmosphere; signs per "
                                     "build_torque_clim; mountain=+h ∂p_s/∂λ over barrier boxes. "
                                     "ARBITRARY absolute scale (a constant pi/180 vs N m is "
                                     "inherited from build_torque_clim) — downstream analysis "
                                     "standardizes, so only relative variations matter",
                                source=f"ARCO-ERA5 {y0}-{y1} 12Z stride {stride}"))


def build_uz_fields(y0: int, y1: int, stride: int) -> Path:
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(level=250).drop_vars("level")
    z = (ds["geopotential"].sel(level=500) / G).drop_vars("level")
    u = u.sortby("latitude").sel(latitude=slice(LAT_LO, LAT_HI))
    z = z.sortby("latitude").sel(latitude=slice(LAT_LO, LAT_HI))
    times = _times(ds, y0, y1, stride)
    print(f"fields: {len(times)} WB2 daily-12Z steps, {y0}-{y1} (stride {stride})", flush=True)

    # Pull year-by-year (bounds memory; WB2 1.5° is fast) with per-year checkpoints.
    # WB2 stores dims lon-before-lat → force conventional (time, lat, lon) so the
    # downstream reshape(nlat, nlon) is correct.
    ydir = CACHE / "uz_years"
    print(f"fields: WB2 u250/z500 {y0}-{y1} (stride {stride}) → {ydir}", flush=True)
    for y in range(y0, y1 + 1):
        yp = ydir / f"{y}.nc"
        if _year_ok(yp):
            print(f"  {y}: cached ✓ (skip)", flush=True)
            continue
        ty = times[(times.year == y)]
        if len(ty) == 0:
            continue
        t0 = time.time()
        print(f"  {y}: fetching {len(ty)} days …", flush=True)
        uy = u.sel(time=ty).astype("float32").load().transpose("time", "latitude", "longitude")
        zy = z.sel(time=ty).astype("float32").load().transpose("time", "latitude", "longitude")
        _atomic_save(xr.Dataset({"u250": uy, "z500": zy}), yp)
        print(f"  {y}: saved → {yp.name}  ({time.time() - t0:.0f}s)", flush=True)

    return _assemble(ydir, y0, y1, CACHE / f"uz_fields_{y0}_{y1}.nc",
                     attrs=dict(source=f"WeatherBench2 ERA5 1.5° {y0}-{y1} 12Z stride {stride}",
                                note="u250=jet, z500=geopotential/g"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1979)
    ap.add_argument("--y1", type=int, default=2023)
    ap.add_argument("--stride", type=int, default=1, help="sample every N days (1 = daily)")
    ap.add_argument("--which", choices=["torque", "fields", "both"], default="both")
    args = ap.parse_args(argv)

    if args.which in ("torque", "both"):
        build_torque_indices(args.y0, args.y1, args.stride)
    if args.which in ("fields", "both"):
        build_uz_fields(args.y0, args.y1, args.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
