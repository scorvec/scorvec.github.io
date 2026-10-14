#!/usr/bin/env python3
"""CMIP6 daily relative AAM by hemisphere and latitude band, from EdayZ zonal-mean ua (2026-10-14: input for the GWO
tests in model space - user: "What about anything related to the gwo in cmip6?" ... "Or a hemispheric version").

Runs in Actions (.github/workflows/cmip6-aam.yml), one job per model member, reading the public Pangeo CMIP6 Google Cloud
mirror anonymously. Historical run, --y0..--y1 (default 1979-2014, the span of the MJO-impacts reductions).

  M_band = 2 pi a^3 / g  sum_j cos(phi_j) w_j(band) (sin phi_n - sin phi_s)_j  sum_k [u]_jk dp_k

[u] is the model's own daily zonal-mean u on its EdayZ pressure levels (Pa); dp_k are the layers between level midpoints,
from the lowest level to zero pressure. Levels below the ground (NaN, or an undecoded 1e20 fill as in CESM2) count as zero, so the absolute value is low near
high terrain - the GWO uses anomalies only. w_j(band) is the fraction of row j's sin-latitude interval inside the band,
so the bands add exactly to the global total. Units 1e25 kg m^2 s^-1, the ERA5 history's `aam_*` columns.

    python scripts/mjo/src/cmip6_aam_daily.py --catalog pangeo-cmip6.csv --model CESM2 --member r1i1p1f1 --out out/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

A = 6.371e6
G = 9.80665
MU = 1e25
FS = gcsfs.GCSFileSystem(token="anon")
BANDS = {"glob": (-90, 90), "nh": (0, 90), "sh": (-90, 0), "b9060s": (-90, -60), "b6030s": (-60, -30), "b3000s": (-30, 0),
         "b0030n": (0, 30), "b3060n": (30, 60), "b6090n": (60, 90)}


def zstore(cat, model, member):
    s = cat[(cat.source_id == model) & (cat.member_id == member) & (cat.experiment_id == "historical")
            & (cat.table_id == "EdayZ") & (cat.variable_id == "ua")]
    if s.empty:
        raise SystemExit(f"no EdayZ ua for {model} {member}")
    return s.sort_values("version").zstore.iloc[-1]


def band_weights(lat):
    """lat ascending (n,) -> {band: (n,) weight = cos(phi) * (overlap of the row's sin-interval with the band)}."""
    e = np.empty(len(lat) + 1)
    e[1:-1] = 0.5 * (lat[1:] + lat[:-1]); e[0] = -90.0; e[-1] = 90.0
    s0, s1 = np.sin(np.deg2rad(e[:-1])), np.sin(np.deg2rad(e[1:]))
    out = {}
    for b, (a, z) in BANDS.items():
        lo, hi = np.sin(np.deg2rad(a)), np.sin(np.deg2rad(z))
        out[b] = np.cos(np.deg2rad(lat)) * np.clip(np.minimum(s1, hi) - np.maximum(s0, lo), 0, None)
    return out


def layer_dp(p):
    """Pressure levels (Pa, any order) -> layer thickness per level (Pa), lowest level to zero."""
    o = np.argsort(p)[::-1]; q = p[o]
    edges = np.r_[q[0], 0.5 * (q[1:] + q[:-1]), 0.0]
    dp = np.empty_like(q); dp[:] = edges[:-1] - edges[1:]
    out = np.empty_like(p); out[o] = dp
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True); ap.add_argument("--model", required=True); ap.add_argument("--member", required=True)
    ap.add_argument("--y0", type=int, default=1979); ap.add_argument("--y1", type=int, default=2014); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    t0 = time.time()
    cat = pd.read_csv(a.catalog)
    zs = zstore(cat, a.model, a.member)
    ds = xr.open_zarr(FS.get_mapper(zs), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                      decode_timedelta=False)
    u = ds["ua"]
    for d in list(u.dims):
        if d in ("lon", "longitude", "x") and u.sizes[d] == 1:
            u = u.isel({d: 0}, drop=True)
    ren = {d: "lat" for d in u.dims if d in ("latitude", "y")}
    u = u.rename(ren).sortby("lat")
    pdim = [d for d in u.dims if d not in ("time", "lat")]
    assert len(pdim) == 1, f"unexpected dims {u.dims}"
    u = u.rename({pdim[0]: "plev"}).transpose("time", "plev", "lat")
    p = u.plev.values.astype(float)
    if p.max() < 2000:                                  # hPa -> Pa
        p = p * 100.0
    yrs = np.array([x.year for x in u.time.values])
    keep = np.where((yrs >= a.y0) & (yrs <= a.y1))[0]
    lat = u.lat.values.astype(float)
    W = band_weights(lat); dp = layer_dp(p)
    rows = []
    for i0 in range(0, len(keep), 1000):
        idx = keep[i0:i0 + 1000]
        X = u.isel(time=idx).values.astype(float)                          # (t, plev, lat)
        X[~np.isfinite(X) | (np.abs(X) > 1e3)] = 0.0                       # NaN or an undecoded fill (CESM2: 1e20) below ground
        col = np.einsum("tkj,k->tj", X, dp)                                # integral of [u] dp per row
        tt = u.time.values[idx]
        d = {"date": [f"{x.year:04d}-{x.month:02d}-{x.day:02d}" for x in tt]}
        for b, w in W.items():
            d[f"aam_{b}"] = 2 * np.pi * A ** 3 / G * (col * w[None, :]).sum(1) / MU
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True).groupby("date", as_index=False).mean()
    df.insert(0, "member", a.member); df.insert(0, "model", a.model)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.model}_{a.member}_aam.csv", index=False, float_format="%.7g")
    log = dict(model=a.model, member=a.member, zstore=zs, nlev=len(p), plev_hPa=[round(x / 100, 2) for x in sorted(p, reverse=True)],
               nlat=len(lat), calendar=str(getattr(u.time.values[0], "calendar", "")), days=len(df),
               first=df.date.iloc[0], last=df.date.iloc[-1], mean_glob=float(df.aam_glob.mean()), seconds=round(time.time() - t0))
    (out / f"{a.model}_{a.member}_aam.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
