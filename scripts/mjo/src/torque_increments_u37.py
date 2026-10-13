#!/usr/bin/env python3
"""Vertical structure of ERA5's angular momentum jumps at the 4D-Var window starts (the zonal-mean increment study).

For every `--step`-th day, reads the 37-level zonal wind and surface pressure at 08, 09, 10, 11 and 20, 21, 22, 23 UTC
from ARCO-ERA5 and reduces each time to the relative-AAM density per pressure layer and 5-degree latitude band:

    D(k, b) = a^3/g * sum_{rows in b} cos^2(phi) dlam dphi * sum_lon u_k * dp_k,   dp_k clipped to p_s (layer edges at the
              midpoints between levels, the top edge at 0 hPa)

and writes, per day, the jump at each window start minus the mean of the neighbouring hourly changes

    J10 = [D(10) - D(09)] - 1/2 { [D(09) - D(08)] + [D(11) - D(10)] }     (likewise J22 from 20..23 UTC)

in 1e25 kg m^2 s^-1, plus the sum of D over the sampled times (for the mean section). ERA5 has no increment fields in
its open archive; the jump is the change across the boundary between two analysis trajectories with the local trend
removed. Runs in Actions (torque-history.yml, mode u37).

    python scripts/mjo/src/torque_increments_u37.py --start 2025-01-01 --end 2025-02-28 --step 5 --out j.nc
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6
G0 = 9.80665
MU = 1e25
HOURS = (8, 9, 10, 11, 20, 21, 22, 23)
_ds = None


def _open():
    global _ds
    if _ds is None:
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None, decode_timedelta=False)
    return _ds


def band5_weights(lat):
    edges = np.arange(-90, 91, 5)
    W = np.zeros((36, len(lat)))
    for i in range(36):
        s, n = edges[i], edges[i + 1]
        w = ((lat > s) & (lat < n)).astype(float)
        w[np.isclose(lat, s)] = 0.5 if s > -90 else 1.0
        w[np.isclose(lat, n)] = 0.5 if n < 90 else 1.0
        W[i] = w
    return W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step", type=int, default=5, help="every N-th day (counted from 2025-01-01)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = _open()
    t0 = pd.Timestamp(ds.time.values[0])
    lat = ds.latitude.values[::-1].astype(np.float64)
    lev = ds.level.values.astype(np.float64)                    # hPa, ascending 1..1000
    assert np.all(np.diff(lev) > 0)
    p = lev * 100.0
    edges = np.empty(len(p) + 1)
    edges[1:-1] = 0.5 * (p[1:] + p[:-1]); edges[0] = 0.0; edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    dl = np.deg2rad(0.25)
    w_lat = (A ** 3 / G0) * np.cos(np.deg2rad(lat)) ** 2 * dl * dl / MU   # per row
    W5 = band5_weights(lat) * w_lat[None, :]                             # (36, 721)
    tl = pd.Timestamp(ds.attrs.get("valid_time_stop_era5t") or ds.attrs["valid_time_stop"])
    anchor = pd.Timestamp("2025-01-01")
    days = [d for d in pd.date_range(a.start, min(pd.Timestamp(a.end), tl - pd.Timedelta(days=1)), freq="D")
            if (d - anchor).days % a.step == 0]
    print(f"{len(days)} days x {len(HOURS)} hours, 37 levels", flush=True)

    def dens(t):
        i = int((t - t0) / pd.Timedelta(hours=1))
        try:
            u = np.asarray(ds["u_component_of_wind"].isel(time=i).values, dtype=np.float32)[:, ::-1, :]
            sp = np.asarray(ds["surface_pressure"].isel(time=i).values, dtype=np.float32)[::-1, :]
        except Exception as e:                                         # noqa: BLE001
            return t, None, repr(e)[:160]
        if not (np.isfinite(sp).all() and np.isfinite(u).mean() > 0.999):
            return t, None, "NaN"
        u = np.nan_to_num(u)
        Z = np.empty((len(p), len(lat)))
        for k in range(len(p)):
            dp = np.clip(np.minimum(edges[k + 1], sp) - np.minimum(edges[k], sp), 0, None)
            Z[k] = (u[k] * dp).sum(1, dtype=np.float64)
        return t, Z @ W5.T, None                                        # (37, 36)

    J10, J22, kept, Dsum, nD, bad = [], [], [], np.zeros((len(p), 36)), 0, 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, d in enumerate(days):
            res = {t: (D, e) for t, D, e in ex.map(dens, [d + pd.Timedelta(hours=h) for h in HOURS])}
            Ds = {t.hour: D for t, (D, e) in res.items() if D is not None}
            if len(Ds) < len(HOURS):
                bad += 1
                print(f"  skip {d:%Y-%m-%d}: {[e for D, e in res.values() if D is None][:1]}", flush=True)
                continue
            J10.append((Ds[10] - Ds[9]) - 0.5 * ((Ds[9] - Ds[8]) + (Ds[11] - Ds[10])))
            J22.append((Ds[22] - Ds[21]) - 0.5 * ((Ds[21] - Ds[20]) + (Ds[23] - Ds[22])))
            kept.append(d)
            for D in Ds.values():
                Dsum += D; nD += 1
            if n % 5 == 0:
                print(f"  {n + 1}/{len(days)} {d:%Y-%m-%d}  {(time.time() - t_start) / 60:.1f} min", flush=True)
    out = xr.Dataset({"J10": (("day", "level", "band"), np.array(J10)), "J22": (("day", "level", "band"), np.array(J22)),
                      "Dsum": (("level", "band"), Dsum), "nD": ((), nD)},
                     coords={"day": pd.DatetimeIndex(kept), "level": lev, "band": np.arange(36),
                             "band_lat": ("band", -87.5 + 5 * np.arange(36))},
                     attrs={"units": "1e25 kg m2 s-1 per layer and 5-degree band",
                            "estimator": "J10 = [D10-D09] - 0.5*([D09-D08]+[D11-D10]); J22 likewise at 20..23 UTC"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.out)
    print(f"wrote {a.out}: {len(kept)} days, {bad} skipped, {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0 if bad <= max(1, 0.1 * len(days)) else 1


if __name__ == "__main__":
    sys.exit(main())
