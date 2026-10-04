#!/usr/bin/env python3
"""Derived climate indices from the SFS beta ensemble, with member spread.

Box indices (SST anomalies vs the model's own 1991-2020 reforecast field
climatology, init-month & lead specific — the cached clim_map_SST npy):
  * DMI / IOD   — west (10°S-10°N, 50-70°E) minus east (10°S-0°, 90-110°E),
                  Saji et al. (1999)
  * SASD        — SW pole (30-40°S, 30-10°W) minus NE pole (15-25°S, 20°W-0°),
                  Morioka et al. (2011)
  * Niño-3 / Niño-4 / Niño-1+2 — the standard boxes
PDO — projection of North Pacific (20-70°N, 110°E-100°W) SST anomalies,
global-mean SSTA removed, onto the model's OWN EOF1 computed from the same
init month's reforecast (30 yrs × 11 members × 10 leads pooled), normalized
by the reforecast PC σ. Sign fixed so positive PDO = cool central North
Pacific. Pattern + σ cached per init month (~900 MB one-time stream).

Feed: assets/sfs/data/sfs_indices.json — per-index member arrays (31 × 12).

    python scripts/sfs/sfs_indices.py [--issue 202608]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "assets" / "sfs" / "data" / "sfs_indices.json"
CLIMDIR = HERE / "data"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
CLIM_Y0, CLIM_Y1 = 1991, 2020
LEADS = list(range(0, 10))

# name -> (label, [(latmin, latmax, lonmin, lonmax, sign), ...])  lon in 0-360
BOXES = {
    "dmi":    ("IOD (DMI)", [(-10, 10, 50, 70, +1), (-10, 0, 90, 110, -1)]),
    "sasd":   ("South Atlantic Subtropical Dipole",
               [(-40, -30, 330, 350, +1), (-25, -15, 340, 360, -1)]),
    "nino3":  ("Niño-3", [(-5, 5, 210, 270, +1)]),
    "nino4":  ("Niño-4", [(-5, 5, 160, 210, +1)]),
    "nino12": ("Niño-1+2", [(-10, 0, 270, 280, +1)]),
}
PDO_DOM = (20, 70, 110, 260)                     # 20-70N, 110E-100W


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def _boxmean(field, lat, lon, b):
    """cos-weighted box mean over possibly-multiple leading dims."""
    la = (lat >= b[0]) & (lat <= b[1])
    lo = (lon >= b[2]) & (lon < b[3])
    sub = field[..., la, :][..., lo]
    w = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones((la.sum(), lo.sum()))
    m = np.isfinite(sub)
    ws = np.where(m, w, 0.0)
    return np.nansum(sub * ws, axis=(-2, -1)) / ws.sum(axis=(-2, -1))


def _global_ssta(anom, lat):
    """area-weighted 60S-60N mean SSTA per sample (for PDO detrending)."""
    la = (lat >= -60) & (lat <= 60)
    sub = anom[..., la, :]
    w = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones((la.sum(), anom.shape[-1]))
    m = np.isfinite(sub)
    ws = np.where(m, w, 0.0)
    return np.nansum(sub * ws, axis=(-2, -1)) / ws.sum(axis=(-2, -1))


def pdo_pattern(month: int, clim: np.ndarray, lat, lon):
    """(pattern_on_domain, ocean_mask_idx, sigma) — cached per init month."""
    pat_f = CLIMDIR / f"pdo_pattern_{month:02d}.npz"
    if pat_f.exists():
        z = np.load(pat_f)
        return z["pattern"], z["sigma"].item()
    ds = _open(f"{BASE}/reforecast/{month:02d}/ocn_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1))).isel(lead=LEADS)
    la = (lat >= PDO_DOM[0]) & (lat <= PDO_DOM[1])
    lo = (lon >= PDO_DOM[2]) & (lon < PDO_DOM[3])
    wgt = np.sqrt(np.cos(np.deg2rad(lat[la])))[:, None]
    samples = []
    n_init = ds.sizes["init"]
    for k in range(n_init):
        f = ds.SST.isel(init=k).values                # (member, lead, lat, lon)
        a = f - clim[None, :, :, :]
        a = a - _global_ssta(a, lat)[..., None, None]
        dom = a[..., la, :][..., lo] * wgt            # (member, lead, nla, nlo)
        samples.append(dom.reshape(-1, la.sum() * lo.sum()))
        if k % 10 == 0:
            print(f"pdo pattern: init {k + 1}/{n_init}", flush=True)
    X = np.concatenate(samples, axis=0).astype(np.float32)
    ocean = np.isfinite(X).all(axis=0)
    Xo = X[:, ocean]
    Xo = Xo - Xo.mean(axis=0)
    _u, _s, vt = np.linalg.svd(Xo, full_matrices=False)
    e1 = vt[0]
    # sign: positive PDO = COOL central North Pacific (35-45N, 170E-145W)
    full = np.full(X.shape[1], np.nan, np.float32)
    full[ocean] = e1
    grid = full.reshape(la.sum(), lo.sum())
    cla = (lat[la] >= 35) & (lat[la] <= 45)
    clo = (lon[lo] >= 170) & (lon[lo] <= 215)
    if np.nanmean(grid[np.ix_(cla, clo)]) > 0:
        e1, full = -e1, -full
    sigma = float(np.std(Xo @ e1, ddof=1))
    np.savez_compressed(pat_f, pattern=full.astype(np.float32),
                        sigma=np.float32(sigma))
    print(f"pdo pattern {month:02d}: cached (σ={sigma:.2f}, "
          f"{int(ocean.sum())} ocean pts)", flush=True)
    return full, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue, month = args.issue, int(args.issue[4:6])
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    clim_f = CLIMDIR / f"clim_map_SST_{month:02d}.npy"
    if not clim_f.exists():
        print("run sfs_maps.py first (needs the cached SST field clim)",
              file=sys.stderr)
        return 1
    clim = np.load(clim_f)                            # (lead, 181, 360)

    ds = _open(f"{BASE}/forecast/{issue}/ocn_monthly.zarr")
    lat, lon = ds.latitude.values, ds.longitude.values
    sst = ds.SST.isel(lead=LEADS).values              # (31, nlead, 181, 360)
    anom = sst - clim[None, :, :, :]
    print("NRT member SST loaded", flush=True)

    out = {}
    for key, (label, boxes) in BOXES.items():
        v = np.zeros(anom.shape[:2])
        for b in boxes:
            v = v + b[4] * _boxmean(anom, lat, lon, b)
        out[key] = {"label": label, "units": "°C",
                    "members": np.round(v, 3).tolist()}

    pattern, sigma = pdo_pattern(month, clim, lat, lon)
    la = (lat >= PDO_DOM[0]) & (lat <= PDO_DOM[1])
    lo = (lon >= PDO_DOM[2]) & (lon < PDO_DOM[3])
    wgt = np.sqrt(np.cos(np.deg2rad(lat[la])))[:, None]
    a = anom - _global_ssta(anom, lat)[..., None, None]
    dom = (a[..., la, :][..., lo] * wgt).reshape(anom.shape[0], anom.shape[1], -1)
    ok = np.isfinite(pattern)
    proj = np.nansum(dom[..., ok] * pattern[ok], axis=-1) / sigma
    out["pdo"] = {"label": "PDO (own-model EOF1, σ units)", "units": "σ",
                  "members": np.round(proj, 3).tolist()}

    valid = [(t0 + pd.DateOffset(months=k)).strftime("%Y-%m") for k in LEADS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "clim": f"own reforecast {CLIM_Y0}-{CLIM_Y1} per init month & lead",
        "valid_months": valid,
        "indices": out,
    }, separators=(",", ":")))
    for k, d in out.items():
        mn = np.nanmean(np.array(d["members"]), axis=0)
        print(f"{k}: ens-mean {np.round(mn[:6], 2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
