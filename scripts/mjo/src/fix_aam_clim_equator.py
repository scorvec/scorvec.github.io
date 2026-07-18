#!/usr/bin/env python3
"""One-off: fold the equator row into the hemispheric AAM reference files.

aam_of() used to exclude the φ=0 row from BOTH hemispheres (NH+SH ≠ Global);
it now splits it half-and-half. This patches the committed reference files the
same way — WITHOUT re-streaming ~127 GB of ARCO — using the per-band AAM
density climatology (aam_density_clim.nc, 1.5° WB2, per day-of-year):

    eq_band_025(doy) = Σ_lev m(doy, lev, φ=0) / 1.5° × 0.25°   (the 0.25° row)
    correction(doy)  = 0.5 × eq_band_025(doy)                  (added to NH & SH)

Patched: aam_clim.nc (mean/min/max curves + harmonic coeffs; σ unchanged —
the correction is a near-constant per-doy offset), aam_history.nc, and
aam_forecast_archive.nc (nh/sh series by valid doy). Global is untouched.
Idempotent: a marker attr prevents double application.

    python src/fix_aam_clim_equator.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from aam import SCALE                                    # 10^25 plot units

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
MARK = "equator_row_split_2026_07_18"


def eq_correction_by_doy() -> np.ndarray:
    """0.5 × 0.25°-equivalent equator-band AAM (SCALE units) per doy 1..366."""
    dens = xr.open_dataarray(REF / "aam_density_clim.nc")          # (doy,lev,lat), per 1.5° band
    lat = dens.latitude.values
    dlat = abs(float(lat[1] - lat[0]))
    eq = dens.sel(latitude=0.0, method="nearest").sum("level")     # (doy,) per 1.5° band
    return 0.5 * (eq.values / dlat) * 0.25 / SCALE


def _harm_of(doy: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * np.asarray(doy, float) / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _save(ds, path: Path):
    tmp = path.with_suffix(".tmp.nc"); ds.to_netcdf(tmp); tmp.replace(path)


def main() -> int:
    corr = eq_correction_by_doy()                                  # (366,) SCALE units
    print(f"equator half-row correction: {corr.min():.4f}…{corr.max():.4f} ×10²⁵ "
          f"(mean {corr.mean():.4f})")

    # ── aam_clim.nc ──
    p = REF / "aam_clim.nc"
    c = xr.open_dataset(p).load()
    if c.attrs.get(MARK):
        print(f"  {p.name}: already patched — skipping")
    else:
        doys = c.doy.values.astype(int)
        add = corr[doys - 1]
        for reg in ("nh", "sh"):
            i = list(c.region.values).index(reg)
            for var in ("mean", "min", "max"):
                c[var].values[i] += add
            c["coeffs"].values[i] += _harm_of(doys, add)
        c.attrs[MARK] = ("NH/SH curves += half the equator-row AAM (from "
                         "aam_density_clim); sigma unchanged")
        _save(c, p); print(f"  {p.name}: patched")

    # ── per-time series files ──
    for name, tdim in (("aam_history.nc", "time"), ("aam_forecast_archive.nc", None)):
        p = REF / name
        if not p.exists():
            print(f"  {name}: absent — skipping"); continue
        ds = xr.open_dataset(p).load()
        if ds.attrs.get(MARK):
            print(f"  {name}: already patched — skipping"); continue
        tcoord = tdim or next((d for d in ("time", "valid", "init") if d in ds.coords), None)
        if tcoord is None:
            print(f"  {name}: no time coord found — SKIPPED, patch manually"); continue
        doys = pd.to_datetime(ds[tcoord].values).dayofyear.values
        add = corr[doys - 1]
        patched = []
        for v in ds.data_vars:
            lv = v.lower()
            if "nh" in lv or "sh" in lv:
                axis = list(ds[v].dims).index(tcoord)
                shape = [1] * ds[v].ndim; shape[axis] = len(add)
                ds[v].values += add.reshape(shape)
                patched.append(v)
        if patched:
            ds.attrs[MARK] = "nh/sh += half equator-row AAM by day-of-year"
            _save(ds, p); print(f"  {name}: patched vars {patched}")
        else:
            # aam_forecast_archive.nc has a (init, region, lead) layout instead;
            # it was patched by hand on 2026-07-18 with the correction evaluated
            # at each VALID day-of-year (init + lead). See the MARK attr there.
            print(f"  {name}: no nh/sh variables — inspect layout manually")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
