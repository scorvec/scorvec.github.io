#!/usr/bin/env python3
"""CMIP6 Indian Ocean dipole + global 200/500 hPa heights for the ENSO-impact members (2026-10-05; user: IOD + strong El
Nino impact on the NH wave train and South American rain -- "might need to use CMIP6 for this"). ACTIONS-ONLY job
(.github/workflows/cmip6-iod.yml, temporary); streams the public Pangeo CMIP6 zarr stores on Google Cloud anonymously and
writes small derived npz files that leave the runner as one workflow artifact. Nothing here is committed as data.

Per member of the local ENSO-impact store (16 models, 429 historical members, 1950-2014):
  ts   monthly 1949-12..2014-12, 2.5-degree area-overlap mean over 20S-20N x 40-120E, int16 centi-degC  (DMI boxes)
  pr   monthly 1949-12..2014-12, same Indian Ocean grid, float16 mm/day  (rainfall-dipole index)
Per member of the z500 subset (<= 10 per model, 147 members), Sep/Oct/Nov months 1950-2014 only:
  zg   200 and 500 hPa, global 2.5 degree, int16 metres offset (z200 - 11500, z500 - 5500)
  pr   30S-30N, global 2.5 degree, float16 mm/day

    python scripts/sst/cmip6_iod_extract.py --model MRI-ESM2-0 --out out/ [--catalog pangeo-cmip6.csv]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SO = {"token": "anon"}
T0, T1 = "1949-12-01", "2014-12-31"
NMON = 781
IO_LAT = np.arange(-20.0, 20.01, 2.5); IO_LON = np.arange(40.0, 120.01, 2.5)
G_LAT = np.arange(-90.0, 90.01, 2.5); G_LON = np.arange(0.0, 357.51, 2.5)
T_LAT = np.arange(-30.0, 30.01, 2.5)
_OPS = {}

MEMBERS = {
    'ACCESS-ESM1-5': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1 r20i1p1f1 r21i1p1f1 r22i1p1f1 r23i1p1f1 r24i1p1f1 r25i1p1f1 r26i1p1f1 r27i1p1f1 r28i1p1f1 r29i1p1f1 r2i1p1f1 r30i1p1f1 r31i1p1f1 r32i1p1f1 r33i1p1f1 r34i1p1f1 r35i1p1f1 r36i1p1f1 r37i1p1f1 r38i1p1f1 r39i1p1f1 r3i1p1f1 r40i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'CESM2': 'r10i1p1f1 r11i1p1f1 r1i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'CNRM-CM6-1': 'r10i1p1f2 r11i1p1f2 r12i1p1f2 r13i1p1f2 r14i1p1f2 r15i1p1f2 r16i1p1f2 r17i1p1f2 r18i1p1f2 r19i1p1f2 r1i1p1f2 r20i1p1f2 r21i1p1f2 r22i1p1f2 r24i1p1f2 r25i1p1f2 r26i1p1f2 r27i1p1f2 r28i1p1f2 r29i1p1f2 r2i1p1f2 r30i1p1f2 r3i1p1f2 r4i1p1f2 r5i1p1f2 r6i1p1f2 r7i1p1f2 r8i1p1f2 r9i1p1f2',
    'CNRM-ESM2-1': 'r10i1p1f2 r11i1p1f2 r1i1p1f2 r2i1p1f2 r3i1p1f2 r4i1p1f2 r5i1p1f2 r7i1p1f2 r8i1p1f2 r9i1p1f2',
    'CanESM5': 'r10i1p1f1 r10i1p2f1 r11i1p1f1 r11i1p2f1 r12i1p1f1 r12i1p2f1 r13i1p1f1 r13i1p2f1 r14i1p1f1 r14i1p2f1 r15i1p1f1 r15i1p2f1 r16i1p1f1 r16i1p2f1 r17i1p1f1 r17i1p2f1 r18i1p1f1 r18i1p2f1 r19i1p1f1 r19i1p2f1 r1i1p1f1 r1i1p2f1 r20i1p1f1 r20i1p2f1 r21i1p1f1 r21i1p2f1 r22i1p1f1 r22i1p2f1 r23i1p1f1 r23i1p2f1 r24i1p1f1 r24i1p2f1 r25i1p1f1 r25i1p2f1 r26i1p2f1 r27i1p2f1 r28i1p2f1 r29i1p2f1 r2i1p1f1 r2i1p2f1 r30i1p2f1 r31i1p2f1 r32i1p2f1 r33i1p2f1 r34i1p2f1 r35i1p2f1 r36i1p2f1 r37i1p2f1 r38i1p2f1 r39i1p2f1 r3i1p1f1 r3i1p2f1 r40i1p2f1 r4i1p1f1 r4i1p2f1 r5i1p1f1 r5i1p2f1 r6i1p1f1 r6i1p2f1 r7i1p1f1 r7i1p2f1 r8i1p1f1 r8i1p2f1 r9i1p1f1 r9i1p2f1',
    'EC-Earth3': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1 r21i1p1f1 r22i1p1f1 r23i1p1f1 r24i1p1f1 r25i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r6i1p1f1 r7i1p1f1 r9i1p1f1',
    'GISS-E2-1-G': 'r101i1p1f1 r102i1p1f1 r10i1p1f1 r10i1p1f2 r10i1p3f1 r10i1p5f1 r11i1p1f2 r1i1p1f1 r1i1p1f2 r1i1p1f3 r1i1p3f1 r1i1p5f1 r2i1p1f1 r2i1p1f2 r2i1p1f3 r2i1p3f1 r2i1p5f1 r3i1p1f1 r3i1p1f2 r3i1p1f3 r3i1p3f1 r3i1p5f1 r4i1p1f1 r4i1p1f2 r4i1p1f3 r4i1p3f1 r4i1p5f1 r5i1p1f1 r5i1p1f2 r5i1p1f3 r5i1p3f1 r6i1p1f1 r6i1p1f2 r6i1p3f1 r6i1p5f1 r7i1p1f1 r7i1p1f2 r7i1p5f1 r8i1p1f1 r8i1p1f2 r8i1p3f1 r8i1p5f1 r9i1p1f1 r9i1p1f2 r9i1p3f1 r9i1p5f1',
    'GISS-E2-1-H': 'r10i1p1f1 r1i1p1f1 r1i1p1f2 r1i1p3f1 r1i1p5f1 r2i1p1f1 r2i1p1f2 r2i1p3f1 r2i1p5f1 r3i1p1f1 r3i1p1f2 r3i1p3f1 r3i1p5f1 r4i1p1f1 r4i1p1f2 r4i1p3f1 r4i1p5f1 r5i1p1f1 r5i1p1f2 r5i1p3f1 r5i1p5f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'INM-CM5-0': 'r10i1p1f1 r1i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'IPSL-CM6A-LR': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1 r20i1p1f1 r21i1p1f1 r22i1p1f1 r23i1p1f1 r24i1p1f1 r25i1p1f1 r26i1p1f1 r27i1p1f1 r28i1p1f1 r29i1p1f1 r2i1p1f1 r30i1p1f1 r31i1p1f1 r32i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MIROC-ES2L': 'r10i1p1f2 r11i1p1f2 r12i1p1f2 r13i1p1f2 r14i1p1f2 r15i1p1f2 r16i1p1f2 r17i1p1f2 r18i1p1f2 r19i1p1f2 r1i1000p1f2 r1i1p1f2 r20i1p1f2 r21i1p1f2 r22i1p1f2 r23i1p1f2 r24i1p1f2 r25i1p1f2 r26i1p1f2 r27i1p1f2 r28i1p1f2 r29i1p1f2 r2i1p1f2 r30i1p1f2 r3i1p1f2 r4i1p1f2 r5i1p1f2 r6i1p1f2 r7i1p1f2 r8i1p1f2 r9i1p1f2',
    'MIROC6': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1 r20i1p1f1 r21i1p1f1 r22i1p1f1 r23i1p1f1 r24i1p1f1 r25i1p1f1 r26i1p1f1 r27i1p1f1 r28i1p1f1 r29i1p1f1 r2i1p1f1 r30i1p1f1 r31i1p1f1 r32i1p1f1 r33i1p1f1 r34i1p1f1 r35i1p1f1 r36i1p1f1 r37i1p1f1 r38i1p1f1 r39i1p1f1 r3i1p1f1 r40i1p1f1 r41i1p1f1 r42i1p1f1 r43i1p1f1 r44i1p1f1 r45i1p1f1 r46i1p1f1 r47i1p1f1 r48i1p1f1 r49i1p1f1 r4i1p1f1 r50i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MPI-ESM1-2-HR': 'r10i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MPI-ESM1-2-LR': 'r10i1p1f1 r1i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MRI-ESM2-0': 'r10i1p1f1 r1i1p1f1 r1i2p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'NorCPM1': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1 r20i1p1f1 r21i1p1f1 r22i1p1f1 r23i1p1f1 r25i1p1f1 r26i1p1f1 r27i1p1f1 r28i1p1f1 r29i1p1f1 r2i1p1f1 r30i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
}
ZG_MEMBERS = {
    'ACCESS-ESM1-5': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1',
    'CESM2': 'r10i1p1f1 r11i1p1f1 r1i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1',
    'CNRM-CM6-1': 'r10i1p1f2 r11i1p1f2 r12i1p1f2 r13i1p1f2 r14i1p1f2 r15i1p1f2 r16i1p1f2 r17i1p1f2 r18i1p1f2 r19i1p1f2',
    'CNRM-ESM2-1': 'r10i1p1f2 r1i1p1f2 r2i1p1f2 r3i1p1f2 r4i1p1f2 r5i1p1f2 r7i1p1f2 r8i1p1f2 r9i1p1f2',
    'CanESM5': 'r10i1p1f1 r10i1p2f1 r11i1p1f1 r11i1p2f1 r12i1p1f1 r12i1p2f1 r13i1p1f1 r13i1p2f1 r14i1p1f1 r14i1p2f1',
    'EC-Earth3': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r18i1p1f1 r19i1p1f1 r1i1p1f1',
    'GISS-E2-1-G': 'r101i1p1f1 r102i1p1f1 r10i1p1f1 r10i1p1f2 r10i1p3f1 r10i1p5f1 r11i1p1f2 r1i1p1f1 r1i1p1f2 r1i1p1f3',
    'GISS-E2-1-H': 'r10i1p1f1 r1i1p1f1 r1i1p1f2 r1i1p3f1 r1i1p5f1 r2i1p1f1 r2i1p1f2 r2i1p3f1 r2i1p5f1 r3i1p1f1',
    'INM-CM5-0': 'r1i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1',
    'IPSL-CM6A-LR': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1',
    'MIROC-ES2L': 'r10i1p1f2 r11i1p1f2 r15i1p1f2 r17i1p1f2 r18i1p1f2 r19i1p1f2 r1i1000p1f2 r1i1p1f2 r21i1p1f2 r22i1p1f2',
    'MIROC6': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r16i1p1f1 r17i1p1f1 r18i1p1f1 r19i1p1f1',
    'MPI-ESM1-2-HR': 'r10i1p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MPI-ESM1-2-LR': 'r10i1p1f1 r1i1p1f1 r2i1p1f1 r3i1p1f1 r5i1p1f1 r6i1p1f1 r7i1p1f1 r8i1p1f1 r9i1p1f1',
    'MRI-ESM2-0': 'r1i1p1f1 r1i2p1f1 r2i1p1f1 r3i1p1f1 r4i1p1f1 r5i1p1f1',
    'NorCPM1': 'r10i1p1f1 r11i1p1f1 r12i1p1f1 r13i1p1f1 r14i1p1f1 r15i1p1f1 r19i1p1f1 r1i1p1f1 r20i1p1f1 r21i1p1f1',
}

def _bounds(c, periodic=False):
    m = 0.5 * (c[1:] + c[:-1])
    if periodic:
        lo = c[0] - 0.5 * ((c[0] - c[-1]) % 360); hi = c[-1] + 0.5 * ((c[0] - c[-1]) % 360)
    else:
        lo = c[0] - (m[0] - c[0]); hi = c[-1] + (c[-1] - m[-1])
    return np.r_[lo, m, hi]


def _overlap(src_edges, dst_edges, periodic=False):
    O = np.zeros((len(dst_edges) - 1, len(src_edges) - 1))
    for sh in ((-360.0, 0.0, 360.0) if periodic else (0.0,)):
        s0, s1 = src_edges[:-1] + sh, src_edges[1:] + sh
        O += np.clip(np.minimum(dst_edges[1:, None], s1[None]) - np.maximum(dst_edges[:-1, None], s0[None]), 0, None)
    return O


def ops(lat, lon, LAT, LON, periodic=True):
    """Separable area-overlap operators native -> target 2.5-degree cells (centres LAT x LON), row-normalised."""
    key = (len(lat), len(lon), float(lat[0]), float(lon[0]), len(LAT), float(LAT[0]), len(LON), float(LON[0]), periodic)
    if key not in _OPS:
        la_e = np.clip(_bounds(lat), -90, 90); lo_e = _bounds(lon, periodic=periodic)
        dst_la = np.clip(np.r_[LAT - 1.25, LAT[-1] + 1.25], -90, 90)
        A = _overlap(np.sin(np.deg2rad(la_e)), np.sin(np.deg2rad(dst_la)))
        B = _overlap(lo_e, np.r_[LON - 1.25, LON[-1] + 1.25], periodic=periodic)
        _OPS[key] = (A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True))
    return _OPS[key]


def regrid(x, lat, lon, LAT, LON, periodic=True):
    A, B = ops(lat, lon, LAT, LON, periodic)
    return np.einsum("il,tlm,jm->tij", A, x, B, optimize=True)


def open_da(store, var):
    ds = xr.open_zarr(store, storage_options=SO, consolidated=True)
    v = ds[var].sel(time=slice(T0, T1))
    if "lon" in v.coords:
        v = v.assign_coords(lon=v.lon % 360).sortby("lon")
    v = v.sortby("lat")
    months = [f"{t.year:04d}-{t.month:02d}" for t in pd.to_datetime([str(t)[:10] for t in v.time.values])]
    return v, np.array(months), ds[var].attrs.get("units", "")


def do_member(src, member, stores, want_zg, out):
    dest = out / f"{src}_{member}.npz"
    if dest.exists():
        return f"{src} {member}: have"
    t0 = time.time(); err = ""
    for attempt in range(3):
        try:
            rec = {"source_id": src, "member_id": member}
            v, months, u = open_da(stores["ts"], "ts")
            if len(months) != NMON:
                raise ValueError(f"ts {len(months)} months")
            x = v.sel(lat=slice(-30, 30), lon=slice(25, 135)).transpose("time", "lat", "lon")
            box = regrid(x.values.astype("float64"), x.lat.values.astype(float), x.lon.values.astype(float), IO_LAT, IO_LON, False)
            rec.update(ts_io=np.round((box - 273.15) * 100).astype("int16"), months=months, io_lat=IO_LAT, io_lon=IO_LON)
            if "pr" in stores:                       # Indian Ocean rainfall, every month, every member (rainfall-index analogues)
                q, qm, _ = open_da(stores["pr"], "pr")
                if len(qm) != NMON:
                    raise ValueError(f"pr {len(qm)} months")
                q = q.sel(lat=slice(-30, 30), lon=slice(25, 135)).transpose("time", "lat", "lon")
                pio = regrid(q.values.astype("float64") * 86400.0, q.lat.values.astype(float), q.lon.values.astype(float), IO_LAT, IO_LON, False)
                rec.update(pr_io=pio.astype("float16"))
            if want_zg and "zg" in stores and "pr" in stores:
                z, zm, _ = open_da(stores["zg"], "zg")
                son = np.array([m[5:] in ("09", "10", "11") and "1950" <= m[:4] <= "2014" for m in zm])
                plev = z.plev.values; scale = 1.0 if plev.max() > 2000 else 100.0
                ks = [int(np.argmin(np.abs(plev * scale - p))) for p in (20000.0, 50000.0)]
                for k, p in zip(ks, (20000.0, 50000.0)):
                    if abs(plev[k] * scale - p) > 1:
                        raise ValueError(f"no {p} Pa level: {plev}")
                zz = z.isel(plev=ks, time=np.where(son)[0]).transpose("time", "plev", "lat", "lon")
                arr = zz.values.astype("float64")
                la, lo = zz.lat.values.astype(float), zz.lon.values.astype(float)
                z200 = regrid(arr[:, 0], la, lo, G_LAT, G_LON); z500 = regrid(arr[:, 1], la, lo, G_LAT, G_LON)
                p_, pm, _ = open_da(stores["pr"], "pr")
                sonp = np.array([m[5:] in ("09", "10", "11") and "1950" <= m[:4] <= "2014" for m in pm])
                pp = p_.sel(lat=slice(-35, 35)).isel(time=np.where(sonp)[0]).transpose("time", "lat", "lon")
                prt = regrid(pp.values.astype("float64") * 86400.0, pp.lat.values.astype(float), pp.lon.values.astype(float), T_LAT, G_LON)
                if son.sum() != 195 or sonp.sum() != 195:
                    raise ValueError(f"SON months zg {son.sum()} pr {sonp.sum()}")
                rec.update(z200=np.round(z200 - 11500).clip(-32767, 32767).astype("int16"),
                           z500=np.round(z500 - 5500).clip(-32767, 32767).astype("int16"),
                           pr_trop=prt.astype("float16"), son_months=zm[son], g_lat=G_LAT, g_lon=G_LON, t_lat=T_LAT)
            tmp = out / f"{src}_{member}.part.npz"
            np.savez_compressed(tmp, **rec)
            tmp.replace(dest)
            return f"{src} {member}: {time.time() - t0:.0f}s{' +zg' if 'z200' in rec else ''}"
        except Exception as e:                                             # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:300]}"; time.sleep(10)
    return f"{src} {member}: FAILED {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--catalog", default="pangeo-cmip6.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(a.catalog)
    c = cat[(cat.table_id == "Amon") & (cat.experiment_id == "historical") & (cat.source_id == a.model)
            & (cat.variable_id.isin(["ts", "zg", "pr"]))]
    c = c.sort_values("version").groupby(["member_id", "variable_id"]).zstore.last().unstack("variable_id")
    mem = MEMBERS[a.model].split()
    mem = mem[: a.limit] if a.limit else mem
    zmem = set(ZG_MEMBERS.get(a.model, "").split())
    print(f"{a.model}: {len(mem)} members, {len(zmem)} with zg", flush=True)
    t0 = time.time()
    for i, m in enumerate(mem):
        if m not in c.index or pd.isna(c.loc[m].get("ts")):
            print(f"{m}: no ts store", flush=True); continue
        stores = {k: c.loc[m][k] for k in c.columns if isinstance(c.loc[m][k], str)}
        r = do_member(a.model, m, stores, m in zmem, out)
        print(f"[{i + 1}/{len(mem)}] {r} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
