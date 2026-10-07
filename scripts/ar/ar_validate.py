#!/usr/bin/env python3
"""IVT validation (runs in Actions): our 1000-300 hPa pressure-level IVT from each model's ANALYSIS against ERA5.

ERA5 reference (ARCO ERA5 on Google Cloud, anonymous): the model-level vertical integrals of eastward/northward
water-vapour flux (surface to top, ECMWF's own diagnostic) and, to separate the effect of the level set from model
differences, ERA5's own pressure-level q/u/v integrated with exactly our routine on (a) the ECMWF open-data 8 levels,
(b) every ERA5 level 1000-300 (20 levels).

    python ar_validate.py --times 2024-11-19T12 2024-11-20T00 ... --models ifs gfs --out out/val_nov2024.json
Statistics over the North Pacific box 20-60N, 150E-125W (ocean-dominated) and along the coastal transect.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arlib as A                                                      # noqa: E402

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BOX = dict(lat=(20.0, 60.0), lon=(150.0, 235.0))
ERA_LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300)


def era5(ds, t):
    """ERA5 at time t on the CROP box: dict(vi (magnitude of the vertical integrals), pl8, pl20, lat, lon)."""
    sel = ds.sel(time=t)
    lat = sel.latitude.values; lon = sel.longitude.values
    jj = np.where((lat >= A.CROP["lat"][0]) & (lat <= A.CROP["lat"][1]))[0]
    ii = np.where((lon >= A.CROP["lon"][0]) & (lon <= A.CROP["lon"][1]))[0]
    sub = sel.isel(latitude=jj, longitude=ii)
    vie = sub["vertical_integral_of_eastward_water_vapour_flux"].values
    vin = sub["vertical_integral_of_northward_water_vapour_flux"].values
    ps = sub["surface_pressure"].values
    pl = sub[["specific_humidity", "u_component_of_wind", "v_component_of_wind"]].sel(level=list(ERA_LEVELS)).load()
    q = pl["specific_humidity"].values; u = pl["u_component_of_wind"].values; v = pl["v_component_of_wind"].values
    o = np.argsort(lat[jj])
    la = lat[jj][o]
    f = lambda a: a[..., o, :]                                          # noqa: E731
    q, u, v, ps, vie, vin = f(q), f(u), f(v), f(ps), f(vie), f(vin)
    lv = list(ERA_LEVELS)
    k8 = [lv.index(L) for L in A.ECMWF_LEVELS]
    e8, n8 = A.integrate(A.ECMWF_LEVELS, q[k8], u[k8], v[k8], ps)
    e20, n20 = A.integrate(lv, q, u, v, ps)
    return {"vi": np.hypot(vie, vin).astype("float32"), "pl8": np.hypot(e8, n8), "pl20": np.hypot(e20, n20),
            "pl8_nomask": np.hypot(*A.integrate(A.ECMWF_LEVELS, q[k8], u[k8], v[k8], None)), "lat": la, "lon": lon[ii]}


def regrid(src_lat, src_lon, a, lat, lon):
    i = A.nearest_idx(src_lat, lat); j = A.nearest_idx(src_lon, lon)
    return a[np.ix_(i, j)]


def stats(x, ref, mask):
    d = x[mask] - ref[mask]
    return {"n": int(mask.sum()), "bias": round(float(d.mean()), 1), "rmse": round(float(np.sqrt((d ** 2).mean())), 1),
            "ratio_median": round(float(np.median(x[mask] / ref[mask])), 3),
            "r": round(float(np.corrcoef(x[mask], ref[mask])[0, 1]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["aifs", "ifs", "gfs", "gdps"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import xarray as xr
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    tl, tlon = A.transect()
    out = {"times": [], "box": BOX, "note": __doc__.strip().splitlines()[0]}
    for ts in a.times:
        t = pd.Timestamp(ts)
        t0 = time.time()
        E = era5(ds, t)
        lat, lon = E["lat"], E["lon"]
        boxm = ((lat[:, None] >= BOX["lat"][0]) & (lat[:, None] <= BOX["lat"][1]) &
                (lon[None, :] >= BOX["lon"][0]) & (lon[None, :] <= BOX["lon"][1]))
        ar = boxm & (E["vi"] >= 250)
        ci = A.nearest_idx(lat, tl); cj = A.nearest_idx(lon, tlon)
        rec = {"time": t.strftime("%Y-%m-%dT%HZ"), "era5_box_max": round(float(E["vi"][boxm].max())), "n_ar_points": int(ar.sum()),
               "era5_pl8_vs_vi": stats(E["pl8"], E["vi"], ar), "era5_pl20_vs_vi": stats(E["pl20"], E["vi"], ar),
               "era5_pl8_nomask_vs_vi": stats(E["pl8_nomask"], E["vi"], ar),
               "coast": {"lat": tl.tolist(), "era5_vi": np.round(E["vi"][ci, cj]).astype(int).tolist()}, "models": {}}
        pts = {}
        for la_, lo_, nm in A.coast_points():
            i = int(A.nearest_idx(lat, la_)[0]); j = int(A.nearest_idx(lon, lo_)[0])
            pts[nm] = {"era5_vi": round(float(E["vi"][i, j]))}
        print(f"{rec['time']}: ERA5 box max {rec['era5_box_max']}, AR points {rec['n_ar_points']}; ERA5 8-lev/VI "
              f"{rec['era5_pl8_vs_vi']}; 20-lev/VI {rec['era5_pl20_vs_vi']}; 8-lev unmasked {rec['era5_pl8_nomask_vs_vi']}",
              flush=True)
        for m in a.models:
            try:
                d = A.read_step(m, t.strftime("%Y%m%d"), t.strftime("%H"), 0)
            except Exception as e:                                       # noqa: BLE001
                d = None; print(f"  {m}: {str(e)[:120]}", flush=True)
            if d is None:
                rec["models"][m] = None; print(f"  {m}: analysis not available", flush=True); continue
            mag = A.ivt_of(d)[0]
            x = regrid(d["lat"], d["lon"], mag, lat, lon)
            s = {"vs_era5_vi": stats(x, E["vi"], ar), "vs_era5_same_levels": None}
            if m in ("gfs", "gdps"):
                x8 = regrid(d["lat"], d["lon"], A.ivt_of(d, A.ECMWF_LEVELS)[0], lat, lon)
                s["8lev_vs_era5_vi"] = stats(x8, E["vi"], ar)
            s["vs_era5_same_levels"] = stats(x, E["pl8"] if m in ("ifs", "aifs") else E["pl20"], ar)
            s["box_max"] = round(float(x[boxm].max()))
            s["coast"] = np.round(x[ci, cj]).astype(int).tolist()
            for la_, lo_, nm in A.coast_points():
                i = int(A.nearest_idx(lat, la_)[0]); j = int(A.nearest_idx(lon, lo_)[0])
                pts[nm][m] = round(float(x[i, j]))
            rec["models"][m] = s
            print(f"  {m}: vs ERA5 VI {s['vs_era5_vi']}; vs ERA5 same levels {s['vs_era5_same_levels']}; box max {s['box_max']}",
                  flush=True)
        rec["points"] = pts
        print("  points:", json.dumps(pts), flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        out["times"].append(rec)
    out["download"] = A.LEDGER.report()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, A.LEDGER.report())


if __name__ == "__main__":
    main()
