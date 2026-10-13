#!/usr/bin/env python3
"""Maps of ERA5's low-level analysis increments of zonal wind and angular momentum (the jump method), to tell
orographic low-level drag from boundary-layer drag over the oceans.

For every `--step`-th day (counted from 2025-01-01) reads, from ARCO-ERA5 at 08, 09, 10, 11 and 20, 21, 22, 23 UTC, the
37-level u and v, surface pressure and the 10 m wind, and forms the jump at each 4D-Var window start

    J10 = [X(10) - X(09)] - 1/2 { [X(09) - X(08)] + [X(11) - X(10)] }      (J22 likewise from 20..23 UTC)

for X = u at 1000/925/850/700 hPa and v at 925/850 hPa (cells below the ground excluded), the 10 m wind, and the
relative angular momentum per cell of the layers at and below 850 hPa (amL) and of the whole column (amC),
a cos(phi) * sum_k u_k dp_k / g * cell area, dp_k clipped to p_s. Maps are block means (winds) or sums (AM) on a
2.5-degree grid. Per day it also writes the AM increment summed over objective regions defined at 0.25 degree:
  ORO  terrain above 1000 m OR subgrid-orography std (ERA5 sdor) above 100 m, plus the 500 km downstream (east)
  OCX  ocean (land-sea mask < 0.5) at 30-65 degrees latitude, outside ORO (the storm tracks)
  OCO  other ocean outside ORO;  LND  other land outside ORO
and, as sensitivities, HGT (terrain > 1000 m + 500 km downstream only) and SDO (sdor > 100 m + 500 km downstream only).

    python torque_increments_maps.py --start 2025-01-01 --end 2025-02-09 --step 3 --out part.nc
    python torque_increments_maps.py --combine parts/ --out final.nc --table final.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
A = 6.371e6
G0 = 9.80665
HOURS = (8, 9, 10, 11, 20, 21, 22, 23)
MAPV = ("u1000", "u925", "u850", "u700", "v925", "v850", "u10", "v10", "amL", "amC")
KEEP = ("u1000", "u925", "u850", "u700", "u10", "amL", "amC")         # committed summary
REGIONS = ("ORO", "OCX", "OCO", "LND", "HGT", "SDO", "GLB")
K = 10                                                                # 0.25 -> 2.5 degrees
_ds = None


def _open():
    global _ds
    if _ds is None:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None, decode_timedelta=False)
    return _ds


def coarse_mean(x):
    b = x[:720].reshape(72, K, 144, K)
    n = np.isfinite(b).sum((1, 3))
    m = np.nanmean(b, axis=(1, 3)) if np.isnan(b).any() else b.mean((1, 3))
    return np.where(n >= K * K // 2, m, np.nan).astype(np.float32)


def coarse_sum(x):
    return x[:720].reshape(72, K, 144, K).sum((1, 3)).astype(np.float64)


def masks(ds, lat, lon):
    """Region masks on the 0.25-degree grid (lat ascending)."""
    def static(name):
        v = ds[name]
        if "time" in v.dims:
            v = v.sel(time="2000-06-15T12:00", method="nearest")
        return np.asarray(v.values, dtype=np.float64)[::-1, :]
    lsm = static("land_sea_mask")
    sdor = static("standard_deviation_of_orography")
    h = xr.open_dataarray(REF / "era5_orography.nc").values.astype(np.float64)       # metres
    assert np.isfinite(lsm).all() and 0.2 < lsm.mean() < 0.45, "land-sea mask looks wrong"

    def downstream(core):
        out = core.copy()
        for i, la in enumerate(lat):
            if not core[i].any():
                continue
            nshift = int(np.ceil(500e3 / (A * np.deg2rad(0.25) * max(np.cos(np.deg2rad(la)), 0.05))))
            row = core[i]
            acc = row.copy()
            for s in range(1, min(nshift, 1440) + 1):
                acc |= np.roll(row, s)                                 # east = increasing longitude
            out[i] = acc
        return out
    hgt = downstream(h > 1000.0)
    sdo = downstream(sdor > 100.0)
    oro = hgt | sdo
    ocean = lsm < 0.5
    ext = (np.abs(lat) >= 30) & (np.abs(lat) <= 65)
    ocx = ocean & ext[:, None] & ~oro
    oco = ocean & ~ext[:, None] & ~oro
    lnd = ~ocean & ~oro
    return {"ORO": oro, "OCX": ocx, "OCO": oco, "LND": lnd, "HGT": hgt, "SDO": sdo,
            "GLB": np.ones_like(oro, bool)}, h, lsm, sdor


def run(a):
    ds = _open()
    t0 = pd.Timestamp(ds.time.values[0])
    lat = ds.latitude.values[::-1].astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)
    lev = ds.level.values.astype(np.float64)
    p = lev * 100.0
    edges = np.empty(len(p) + 1)
    edges[1:-1] = 0.5 * (p[1:] + p[:-1]); edges[0] = 0.0; edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    dl = np.deg2rad(0.25)
    cos = np.cos(np.deg2rad(lat))
    fac = (A * cos * (A ** 2) * cos * dl * dl / G0)[:, None]              # kg m2/s per (m/s * Pa) per cell
    low = lev >= 850
    ki = {L: int(np.where(lev == L)[0][0]) for L in (1000, 925, 850, 700)}
    M, h, lsm, sdor = masks(ds, lat, lon)
    tl = pd.Timestamp(ds.attrs.get("valid_time_stop_era5t") or ds.attrs["valid_time_stop"])
    anchor = pd.Timestamp("2025-01-01")
    days = [d for d in pd.date_range(a.start, min(pd.Timestamp(a.end), tl - pd.Timedelta(days=1)), freq="D")
            if (d - anchor).days % a.step == 0]
    print(f"{len(days)} days; region area fractions: " + ", ".join(
        f"{r} {float((M[r] * cos[:, None]).sum() / (cos.sum() * 1440)):.3f}" for r in REGIONS), flush=True)

    def one(t):
        i = int((t - t0) / pd.Timedelta(hours=1))
        try:
            u = np.asarray(ds["u_component_of_wind"].isel(time=i).values, dtype=np.float32)[:, ::-1, :]
            v = np.asarray(ds["v_component_of_wind"].isel(time=i).values, dtype=np.float32)[:, ::-1, :]
            sp = np.asarray(ds["surface_pressure"].isel(time=i).values, dtype=np.float32)[::-1, :]
            u10 = np.asarray(ds["10m_u_component_of_wind"].isel(time=i).values, dtype=np.float32)[::-1, :]
            v10 = np.asarray(ds["10m_v_component_of_wind"].isel(time=i).values, dtype=np.float32)[::-1, :]
        except Exception as e:                                         # noqa: BLE001
            return t, None, repr(e)[:160]
        if not np.isfinite(sp).all():
            return t, None, "NaN sp"
        out = {}
        for L in (1000, 925, 850, 700):
            x = u[ki[L]].copy(); x[sp < L * 100.0] = np.nan
            out[f"u{L}"] = x
        for L in (925, 850):
            x = v[ki[L]].copy(); x[sp < L * 100.0] = np.nan
            out[f"v{L}"] = x
        out["u10"] = u10; out["v10"] = v10
        uu = np.nan_to_num(u)
        amL = np.zeros(sp.shape, np.float64); amC = np.zeros(sp.shape, np.float64)
        for k in range(len(p)):
            dp = np.clip(np.minimum(edges[k + 1], sp) - np.minimum(edges[k], sp), 0, None)
            c = uu[k] * dp
            amC += c
            if low[k]:
                amL += c
        out["amL"] = amL * fac; out["amC"] = amC * fac
        return t, out, None

    J = {f"{v}_{w}": [] for v in MAPV for w in ("10", "22")}
    rows, kept = [], []
    tstart = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, d in enumerate(days):
            res = {tt.hour: o for tt, o, e in ex.map(one, [d + pd.Timedelta(hours=hh) for hh in HOURS]) if o is not None}
            if len(res) < len(HOURS):
                print(f"  skip {d:%Y-%m-%d}", flush=True); continue
            rec = {"day": d.strftime("%Y-%m-%d")}
            for var in MAPV:
                for w, (b0, b1, b2, b3) in {"10": (8, 9, 10, 11), "22": (20, 21, 22, 23)}.items():
                    X = {hh: res[hh][var] for hh in (b0, b1, b2, b3)}
                    jf = (X[b2] - X[b1]) - 0.5 * ((X[b1] - X[b0]) + (X[b3] - X[b2]))
                    if var.startswith("am"):
                        J[f"{var}_{w}"].append((coarse_sum(jf) / 1e20).astype(np.float32))      # 1e20 kg m2/s per cell
                        for r in REGIONS:
                            rec[f"{var}_{w}_{r}"] = float(jf[M[r]].sum() / 1e25)
                    else:
                        J[f"{var}_{w}"].append(coarse_mean(jf))
            rows.append(rec); kept.append(d)
            if n % 3 == 0:
                print(f"  {n + 1}/{len(days)} {d:%Y-%m-%d}  {(time.time() - tstart) / 60:.1f} min", flush=True)
    clat = lat[:720].reshape(72, K).mean(1); clon = lon.reshape(144, K).mean(1)
    dsout = xr.Dataset({k: (("day", "lat", "lon"), np.array(v)) for k, v in J.items()},
                       coords={"day": pd.DatetimeIndex(kept), "lat": clat, "lon": clon})
    tab = pd.DataFrame(rows)
    dsout.to_netcdf(a.out, encoding={k: {"zlib": True, "complevel": 4} for k in J})
    tab.to_csv(str(a.out).replace(".nc", ".csv"), index=False, float_format="%.6g")
    # static fields for the figures (2.5-degree), once per chunk (identical)
    st = xr.Dataset({"h_mean": (("lat", "lon"), coarse_mean(h.astype(np.float32))),
                     "h_max": (("lat", "lon"), h[:720].reshape(72, K, 144, K).max((1, 3)).astype(np.float32)),
                     "lsm": (("lat", "lon"), coarse_mean(lsm.astype(np.float32))),
                     "sdor": (("lat", "lon"), coarse_mean(sdor.astype(np.float32)))}
                    | {f"mask_{r}": (("lat", "lon"), coarse_mean(M[r].astype(np.float32))) for r in REGIONS},
                    coords={"lat": clat, "lon": clon},
                    attrs={f"area_{r}": float((M[r] * cos[:, None]).sum() / (cos.sum() * 1440)) for r in REGIONS})
    st.to_netcdf(str(a.out).replace(".nc", "_static.nc"))
    print(f"wrote {a.out}: {len(kept)} days, {(time.time() - tstart) / 60:.1f} min", flush=True)
    return 0


def combine(a):
    fs = sorted(f for f in glob.glob(str(Path(a.combine) / "*.nc")) if not f.endswith("_static.nc"))
    parts = [xr.open_dataset(f).load() for f in fs]
    J = xr.concat(parts, "day").sortby("day")
    days = pd.DatetimeIndex(J.day.values)
    m = days.month.values
    sets = {"ALL": np.ones(len(days), bool), "DJF": np.isin(m, (12, 1, 2)), "JJA": np.isin(m, (6, 7, 8))}
    out = {}
    for var in KEEP:
        for w in ("10", "22", "both"):
            X = J[f"{var}_10"].values + J[f"{var}_22"].values if w == "both" else J[f"{var}_{w}"].values
            for sname, sel in sets.items():
                if sname != "ALL" and w != "both":
                    continue
                Y = X[sel].astype(np.float64)
                n = np.isfinite(Y).sum(0)
                mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0, ddof=1)
                a0 = Y[:-1] - mu; a1 = Y[1:] - mu                       # lag-1 between consecutive sampled days
                r1 = np.nanmean(a0 * a1, 0) / np.nanvar(Y, 0)
                r1 = np.clip(np.nan_to_num(r1), 0, 0.9)
                neff = n * (1 - r1) / (1 + r1)
                t = mu / (sd / np.sqrt(np.maximum(neff, 2)))
                out[f"{var}_{w}_{sname}_mean"] = (("lat", "lon"), mu.astype(np.float32))
                out[f"{var}_{w}_{sname}_t"] = (("lat", "lon"), t.astype(np.float32))
                out[f"{var}_{w}_{sname}_neff"] = (("lat", "lon"), neff.astype(np.float32))
    res = xr.Dataset(out, coords={"lat": J.lat, "lon": J.lon},
                     attrs={"days": f"{days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d} ({len(days)} sampled days)",
                            "units": "u, v: m/s per window; amL/amC: 1e20 kg m2 s-1 per 2.5-degree cell per window",
                            "estimator": "J10 = [X10-X09] - 0.5*([X09-X08]+[X11-X10]); both = J10 + J22"})
    stf = sorted(glob.glob(str(Path(a.combine) / "*_static.nc")))
    if stf:
        st = xr.open_dataset(stf[0]).load()
        res = res.merge(st)
        res.attrs.update({k: v for k, v in st.attrs.items()})
    res.to_netcdf(a.out, encoding={k: {"zlib": True, "complevel": 5} for k in res.data_vars})
    tabs = [pd.read_csv(f.replace(".nc", ".csv")) for f in fs]
    pd.concat(tabs).sort_values("day").to_csv(a.table, index=False, float_format="%.6g")
    print(f"combined {len(fs)} parts, {len(days)} days -> {a.out}, {a.table}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default=None, help="directory of part files to combine")
    ap.add_argument("--table", default=None)
    a = ap.parse_args()
    return combine(a) if a.combine else run(a)


if __name__ == "__main__":
    sys.exit(main())
