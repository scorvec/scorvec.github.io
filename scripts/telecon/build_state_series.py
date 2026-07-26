#!/usr/bin/env python3
"""Daily NH state series for the teleconnection engine: AAM + mountain torque.

One 12Z snapshot per day. For each day compute:
  - relative AAM (global / NH / SH), the aam.py integral on the 1.5deg grid
  - mountain torque per orographic range (Himalaya/Tibet, Rockies, Andes) and
    NH total:  T_m = -a^2 cosphi * p_s * dh/dlambda, integrated over the box
    (same formulation and sign convention as scripts/mjo build_torque_clim.py:
    eastward torque ON the atmosphere is positive)

Sources (same split as build_u850_bandseries.py):
  --source wb2   1959-2023, WeatherBench2 ERA5 1.5deg 6-hourly (13 levels — no
                 10 hPa; fine for a phase index, noted in attrs)
  --source arco  2023->present tail, ARCO 0.25deg subsampled ::6 to 1.5deg

Laptop-only (streams from GCS; output is committed). Resumable: appends by
year, skipping years already complete in the output file.

    python build_state_series.py --source wb2  --start 1959 --end 2023
    python build_state_series.py --source arco --start 2023 --end 2026
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "mjo" / "data" / "reference"
OROG = REF / "era5_orography.nc"                 # geopotential height of the surface (m)
OUT = HERE / "data" / "state_series.nc"

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

A = 6.371e6
G = 9.80665
HU = 1e18                                        # Hadley unit (kg m^2 s^-2 = N m); torques reported in HU
AAM_SCALE = 1e25                                 # AAM reported in 10^25 kg m^2 s^-1
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# same boxes as torque_map_anim.RANGES (lon 0..360)
RANGES = {
    "himalaya": (70, 105, 25, 45),
    "rockies": (232, 258, 30, 62),
    "andes": (282, 296, -56, 12),
}


def _vert_weights(p_pa, sp):
    """aam.py's mass-layer thickness, clipped to surface pressure."""
    edges = np.empty(len(p_pa) + 1)
    edges[1:-1] = 0.5 * (p_pa[1:] + p_pa[:-1])
    edges[0] = max(p_pa[0] - 0.5 * (p_pa[1] - p_pa[0]), 0.0)
    edges[-1] = p_pa[-1] + 0.5 * (p_pa[-1] - p_pa[-2])
    lo = np.minimum(edges[:-1][:, None, None], sp)
    hi = np.minimum(edges[1:][:, None, None], sp)
    return np.clip(hi - lo, 0.0, None)


def aam_of(u, p_pa, sp, lat, dlon, dlat):
    """(global, NH, SH) relative AAM — identical math to scripts/mjo aam.py
    (equator row split half-and-half so NH + SH = global exactly)."""
    Uint = np.sum(u * _vert_weights(p_pa, sp), axis=0)
    cos2 = np.cos(np.deg2rad(lat)) ** 2
    dens = (A ** 3 / G) * Uint * cos2[:, None] * dlon * dlat
    eq = 0.5 * dens[np.isclose(lat, 0.0)].sum()
    return dens.sum(), dens[lat > 0].sum() + eq, dens[lat < 0].sum() + eq


def torque_fields(sp, lat, lon, dhdx):
    """Mountain-torque density  -a^2 cosphi p_s dh/dlambda  -> per-box + NH sums.
    dhdx is dh/dlambda (m per radian of longitude) on the same grid."""
    dlon = np.deg2rad(abs(lon[1] - lon[0]))
    dlat = np.deg2rad(abs(lat[1] - lat[0]))
    cos = np.cos(np.deg2rad(lat))[:, None]
    dens = -(A ** 2) * cos * sp * dhdx * dlon * dlat          # N m per cell
    out = {}
    for name, (lo0, lo1, la0, la1) in RANGES.items():
        m = ((lon[None, :] >= lo0) & (lon[None, :] <= lo1)
             & (lat[:, None] >= la0) & (lat[:, None] <= la1))
        out[f"tq_{name}"] = float(dens[m].sum() / HU)
    out["tq_nh"] = float(dens[lat > 0].sum() / HU)
    out["tq_global"] = float(dens.sum() / HU)
    return out


def _prep_orog(lat, lon):
    """Surface height on the working grid + its zonal derivative dh/dlambda."""
    o = xr.open_dataarray(OROG)
    o = o.assign_coords(longitude=o.longitude % 360).sortby("longitude")
    h = o.interp(latitude=lat, longitude=lon, method="linear").values
    if np.nanmax(np.abs(h)) > 1e4:                 # stored as geopotential, not height
        h = h / G
    dlon_rad = np.deg2rad(abs(lon[1] - lon[0]))
    dhdx = np.gradient(h, axis=1) / dlon_rad       # m per radian (periodic edge error negligible)
    return dhdx


def open_source(source):
    if source == "wb2":
        ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
        u = ds["u_component_of_wind"].sel(level=list(LEVELS))
        sp = ds["surface_pressure"]
    else:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        u = ds["u_component_of_wind"].sel(level=list(LEVELS)).isel(
            latitude=slice(None, None, 6), longitude=slice(None, None, 6))
        sp = ds["surface_pressure"].isel(
            latitude=slice(None, None, 6), longitude=slice(None, None, 6))
    return u, sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wb2", "arco"], default="wb2")
    ap.add_argument("--start", type=int, default=1959)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)

    have = None
    done_years = set()
    if out.exists():
        have = xr.open_dataset(out).load(); have.close()
        yc = pd.DatetimeIndex(have.time.values).year
        # a year counts as done if it has ~all its days (tail year always re-pulled)
        for y in np.unique(yc):
            if (yc == y).sum() >= 360 and y < pd.Timestamp.utcnow().year:
                done_years.add(int(y))

    u_all, sp_all = open_source(a.source)
    u_all = u_all.sortby("latitude")
    sp_all = sp_all.sortby("latitude")
    lat = u_all.latitude.values
    lon = (u_all.longitude.values % 360)
    order = np.argsort(lon)
    dhdx = _prep_orog(lat, lon[order])
    p_pa = np.array(LEVELS, float) * 100.0
    dlon = np.deg2rad(abs(float(lon[1] - lon[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))

    pieces = [have] if have is not None else []
    for y in range(a.start, a.end + 1):
        if y in done_years:
            print(f"  {y}: already complete — skipped", flush=True)
            continue
        t0 = time.time()
        sel = dict(time=slice(f"{y}-01-01", f"{y}-12-31"))
        uy = u_all.sel(**sel); spy = sp_all.sel(**sel)
        uy = uy.sel(time=uy.time.dt.hour == 12)
        spy = spy.sel(time=spy.time.dt.hour == 12)
        uy = uy.compute(); spy = spy.compute()
        rows = []
        for i, t in enumerate(pd.to_datetime(uy.time.values)):
            uu = uy.isel(time=i).transpose("level", "latitude", "longitude").values[:, :, order]
            ss = spy.isel(time=i).transpose("latitude", "longitude").values[:, order]
            g, nh, sh = aam_of(uu, p_pa, ss, lat, dlon, dlat)
            row = {"aam_global": g / AAM_SCALE, "aam_nh": nh / AAM_SCALE, "aam_sh": sh / AAM_SCALE}
            row.update(torque_fields(ss, lat, lon[order], dhdx))
            row["time"] = t
            rows.append(row)
        df = pd.DataFrame(rows).set_index("time")
        pieces.append(xr.Dataset({k: ("time", df[k].values) for k in df.columns},
                                 coords={"time": df.index}))
        merged = xr.concat([p for p in pieces if p is not None], dim="time").sortby("time")
        merged = merged.isel(time=~pd.Index(merged.time.values).duplicated(keep="last"))
        merged.attrs.update(levels=str(LEVELS), note="12Z daily; AAM in 1e25 kg m2 s-1; "
                            "torque in Hadley units (1e18 N m); no 10 hPa (WB2 has none)")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(out)
        pieces = [merged]
        print(f"  {y}: {len(df)} days in {time.time()-t0:.0f}s "
              f"(NH AAM mean {df.aam_nh.mean():+.2f})", flush=True)
    print(f"done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
