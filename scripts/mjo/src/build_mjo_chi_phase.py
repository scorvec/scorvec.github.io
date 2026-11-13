#!/usr/bin/env python3
"""Build the reference that reads an RMM phase off the MJO chart's velocity potential (mjo_chi.py): ERA5 1991-2020,
offline, from the daily ERA5 store.

  field      daily 200 hPa velocity potential, 15°S-15°N, by the chart's solver on the chart's grid, less the chart's
             normal, filtered to the MJO's band over the whole record (mjo_chi.mjo_mask)
  patterns   the field regressed on the observed RMM1 and RMM2 (rmm_era5.nc), one pattern each by longitude
  projection a day's field fitted by the two patterns (least squares over longitude); each component divided by its
             standard deviation over the record, as the RMM's are, so an amplitude of 1 means what it means there

Printed, and kept in the file's attributes: how often the phase read this way is the observed RMM's on the days both
are active - with the patterns from other years (five-year blocks left out), by half-year, and as the chart sees it:
RECORD analysis days and a 15-day future, filtered alone, read at the last analysis day and at the future's last day.
Writes scripts/mjo/data/reference/mjo_chi_phase.nc (committed). Run by hand; no workflow runs it.

    python src/build_mjo_chi_phase.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import enso_rmm
import kelvin_tracker as K
import mjo_chi
import walker_chi as wc
from localdirs import env_dir

Y0, Y1 = 1991, 2020
BLOCK = 5  # years left out together in the cross-validation
EDGE = 60  # days dropped at the record's end, where the filter has no future
FUTURE = 15  # days of future the chart's filter has past the last analysis
EVERY = 7  # days between the emulated charts


def daily_wind(var: str, year: int) -> xr.DataArray:
    ds = xr.open_dataset(env_dir("ERA5_STORE", "wb2_1p5_daily_global", var) / f"{var}_{year}.nc")
    da = ds[next(iter(ds.data_vars))]
    da = da.assign_coords(
        latitude=np.round(da.latitude.values.astype("float64"), 3),
        longitude=np.round(da.longitude.values.astype("float64"), 3),
    )
    return da.transpose("time", "latitude", "longitude").sortby("latitude").load()


def band_record():
    """(days, band-mean χ200 (day, lon), lat, lon) for Y0-Y1."""
    t0 = time.time()
    days, rows = [], []
    for year in range(Y0, Y1 + 1):
        u, v = daily_wind("u200", year), daily_wind("v200", year)
        for k in range(u.sizes["time"]):
            chi, lat, lon = wc.chi_strip(u.isel(time=k), v.isel(time=k))
            rows.append(mjo_chi.band_mean(chi[:, :-1].astype(float), lat))
        days.append(u.time.values)
        print(f"  {year}: {len(rows)} days ({time.time() - t0:.0f}s)", flush=True)
    return pd.DatetimeIndex(np.concatenate(days)).normalize(), np.array(rows), lat, lon[:-1]


def projection(f: np.ndarray, rmm: np.ndarray):
    """f (day, lon), rmm (day, 2) -> (patterns (2, lon), projection (2, lon)): components = f @ projection.T, each of
    unit standard deviation over these days."""
    pat, *_ = np.linalg.lstsq(rmm, f, rcond=None)
    proj = np.linalg.pinv(pat).T
    return pat, proj / (f @ proj.T).std(0)[:, None]


def agreement(est: np.ndarray, rmm: np.ndarray, sel: np.ndarray) -> dict:
    """Over the days of `sel` on which both indices are active: the share in the same phase and within one phase."""
    act = sel & (np.hypot(*est.T) >= 1) & (np.hypot(*rmm.T) >= 1)
    d = (enso_rmm.phase_of(*est[act].T) - enso_rmm.phase_of(*rmm[act].T) + 4) % 8 - 4
    return {
        "days": int(act.sum()),
        "same": round(float((d == 0).mean()), 3),
        "within_one": round(float((np.abs(d) <= 1).mean()), 3),
    }


def main() -> int:
    days, band, lat, lon = band_record()
    clim = xr.open_dataset(K.CLIM)
    anom = band - mjo_chi.band_mean(K.chi_normal(days, clim, lat, lon), lat)
    f = K.wave_filter(anom[:, None], mjo_chi.mjo_mask, mjo_chi.PAD, mjo_chi.RAMP)[:, 0]
    r = xr.open_dataset(mjo_chi.REF / "rmm_era5.nc")
    rmm = pd.DataFrame({"a": r.rmm1.values, "b": r.rmm2.values}, index=pd.DatetimeIndex(r.time.values))
    rmm = rmm.reindex(days).values
    # the filter tapers the record's first tenth and has no future at its end: neither goes into the fit
    n = len(days)
    inner = (np.arange(n) >= int(0.1 * n)) & (np.arange(n) < n - EDGE) & np.isfinite(rmm).all(1)
    pat, proj = projection(f[inner], rmm[inner])

    est = np.full((n, 2), np.nan)
    for y in range(Y0, Y1 + 1, BLOCK):
        out = (days.year >= y) & (days.year < y + BLOCK)
        est[out] = f[out] @ projection(f[inner & ~out], rmm[inner & ~out])[1].T
    winter = np.isin(days.month, (11, 12, 1, 2, 3, 4))
    skill = {
        "all": agreement(est, rmm, inner),
        "Nov-Apr": agreement(est, rmm, inner & winter),
        "May-Oct": agreement(est, rmm, inner & ~winter),
        "r": [round(float(np.corrcoef(est[inner, k], rmm[inner, k])[0, 1]), 3) for k in (0, 1)],
        "active_share": round(float((np.hypot(*est[inner].T) >= 1).mean()), 3),
        "rmm_active_share": round(float((np.hypot(*rmm[inner].T) >= 1).mean()), 3),
    }
    print(f"  cross-validated against the RMM: {skill}", flush=True)

    # as the chart sees it: RECORD days and FUTURE more, its own mean out, filtered alone
    span = mjo_chi.RECORD + FUTURE
    ends = np.arange(int(0.1 * n) + span, n - EDGE, EVERY)
    now, last = np.empty((ends.size, 2)), np.empty((ends.size, 2))
    for i, e in enumerate(ends):
        w = anom[e - span : e]
        g = K.wave_filter((w - w.mean(0))[:, None], mjo_chi.mjo_mask, mjo_chi.PAD, mjo_chi.RAMP)[:, 0] @ proj.T
        now[i], last[i] = g[-FUTURE - 1], g[-1]
    full = f @ proj.T
    for name, got, at in (("last analysis day", now, ends - FUTURE - 1), ("day 15", last, ends - 1)):
        ok = np.isfinite(rmm[at]).all(1)
        s = agreement(got[ok], rmm[at][ok], np.ones(ok.sum(), bool))
        amp, ref = np.hypot(*got.T), np.hypot(*full[at].T)
        s["r_full_record"] = round(float(np.corrcoef(got.ravel(), full[at].ravel())[0, 1]), 3)
        s["amplitude_ratio"] = round(float(amp.mean() / ref.mean()), 3)
        s["active_share"] = round(float((amp >= 1).mean()), 3)
        skill[name] = s
        print(f"  the chart's view, {name}: {s}", flush=True)

    xr.Dataset(
        {
            "proj": (("component", "longitude"), proj),
            "pattern": (("component", "longitude"), pat),
        },
        coords={"component": ["rmm1", "rmm2"], "longitude": lon},
        attrs={
            "title": "RMM-like components of the MJO-band 200 hPa velocity potential, 15S-15N",
            "use": "components = field (1e6 m2 s-1, by longitude) @ proj.T; phase and amplitude as for the RMM",
            "pattern": "the field per unit RMM1 and RMM2 (1e6 m2 s-1)",
            "period": f"ERA5 {Y0}-{Y1}, observed RMM from rmm_era5.nc",
            "skill": json.dumps(skill),
            "builder": "scripts/mjo/src/build_mjo_chi_phase.py",
        },
    ).to_netcdf(mjo_chi.PHASE_REF)
    print(f"  wrote {mjo_chi.PHASE_REF.name} ({mjo_chi.PHASE_REF.stat().st_size / 1e3:.0f} kB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
