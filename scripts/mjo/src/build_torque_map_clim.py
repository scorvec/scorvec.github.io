#!/usr/bin/env python3
"""Rebuild the torque-density map climatology from the stored ERA5 basis.

An earlier builder streamed ERA5 and hard-coded a single constant drag
coefficient. This one reads the pre-computed basis
(build_torque_basis.py -> torque_basis_1991-2020_s5.nc in $MJO_CACHE/era5_cache, 00Z+12Z,
5-day stride, 1991-2020) and applies the ERA5-derived drag FIELD, so no ERA5 is re-streamed.
The live maps (torque_map_anim.py) take their friction from the model's own surface stress
and their anomalies against this climatology.

Why the drag changed
--------------------
The old friction term used rho = 1.225 and C_d = 1.3e-3 everywhere. Against
ERA5's own turbulent surface stress on held-out years that carried only 72% of
the true variability and had the wrong sign in the global mean (+8.8 vs -7.8
Hadley, rms 18.4). A per-10-degree-band fit was an improvement (rms 9.0), but a
drag coefficient is a property of the SURFACE, not of a latitude.

build_drag_field.py therefore solves C_d per grid cell and calendar month so the
bulk kernel best reproduces ERA5's own boundary-layer stress. Median fit r2 is
0.95, and C_d runs 1.3e-3 over ocean to 1.8e-2 over rough land -- the physics the
single constant was missing. This script applies that field to the stored basis, so
the climatology's friction is ERA5's own stress as closely as the bulk kernel allows.
(The time-series bands' spread, torque_hemi_sd.nc, is build_torque_hemi_sd.py's: daily
means, ERA5's own stress.) Offline, run by hand; no workflow runs it.

    python src/build_torque_map_clim.py      # -> data/reference/torque_map_clim_coeffs.nc (committed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from doycal import noleap_doy

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from localdirs import env_dir

BASIS = env_dir("MJO_CACHE", "era5_cache", "torque_basis_1991-2020_s5.nc")
DRAG = REF / "drag_cd_field.nc"  # per-cell, per-month C_d (build_drag_field.py)
OUT = REF / "torque_map_clim_coeffs.nc"


def main() -> int:
    if not BASIS.exists() or not DRAG.exists():
        raise SystemExit(f"need {BASIS.name} and {DRAG.name}")
    b = xr.open_dataset(BASIS)
    cdf = xr.open_dataset(DRAG)["cd"]
    clat, clon = b.clat.values, b.clon.values
    assert np.array_equal(cdf.latitude.values, clat) and np.array_equal(cdf.longitude.values, clon), (
        "drag field and basis are on different grids"
    )
    t = pd.DatetimeIndex(b.time.values)
    hours = np.array(sorted(set(t.hour)))
    print(f"basis {len(t)} times, hours {hours.tolist()}", flush=True)

    # Friction density = C_d(month, y, x) * (bulk kernel). Xl5 + Xs5 IS that kernel
    # (the land/sea split exists only so a fitted split could be tried); the fitted
    # drag field makes it ERA5's own stress, which the forecast's stress is compared with.
    kern = b["Xl5"].values + b["Xs5"].values
    cdm = cdf.values[t.month.values - 1]  # (time, lat, lon)
    fric = kern * cdm
    mtn = b["mtn5"].values
    print(
        f"  friction density rebuilt with the drag field (C_d {np.nanmin(cdm):.2e}..{np.nanmax(cdm):.2e})", flush=True
    )

    nlat, nlon = len(clat), len(clon)
    cf = np.empty((len(hours), 5, nlat, nlon))
    cm = np.empty_like(cf)
    for j, h in enumerate(hours):
        sel = t.hour == h
        w = 2 * np.pi * noleap_doy(t[sel]) / 365.25  # common-year day: 1 March = 60 every year
        X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        for arr, out in ((fric, cf), (mtn, cm)):
            y = arr[sel].reshape(sel.sum(), -1)
            out[j] = np.linalg.lstsq(X, y, rcond=None)[0].reshape(5, nlat, nlon)
        print(f"  fitted hour {h:02d}Z on {int(sel.sum())} samples", flush=True)

    xr.Dataset(
        {
            "friction": (("hour", "coef", "latitude", "longitude"), cf),
            "mountain": (("hour", "coef", "latitude", "longitude"), cm),
        },
        coords={"hour": hours, "coef": np.arange(5), "latitude": clat, "longitude": b.clon.values},
        attrs={
            "note": (
                "ERA5 1991-2020 torque-density harmonic climatology, 5 deg. "
                "Friction uses the per-cell per-month drag field solved against "
                "ERA5's own boundary-layer stress (median fit r2 0.95). Built from "
                "the stored basis, not re-streamed."
            ),
            "source_basis": BASIS.name,
            "drag_field": DRAG.name,
        },
    ).to_netcdf(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
