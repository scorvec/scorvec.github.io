#!/usr/bin/env python3
"""Reference file of the GEFS pattern products (laptop, once): gefs_patterns_ref.npz for release gefs-ref-v1.

Everything the daily job needs that does not depend on the GEFS reforecast (2026-09-26):

  proj_<id>, sigma_doy_<id>, std_month_z500, meta.patterns
        the GEPS page's teleconnection projectors, VERBATIM from geps_subx/telecon/patterns.nc (CPC-style patterns on
        NCEP/NCAR R1 1991-2020, build_telecon_patterns.py) and its validation.json - the patterns are model-independent,
        so the GEFS indices are the same indices and the loading-pattern figure is shared
  meta.defined_months
        the calendar months in which CPC itself publishes each NH pattern (>= 15 of the 30 base years in CPC's
        tele_index.nh), telecon_geps.Patterns' rule, precomputed so the daily job needs no CPC table
  reg_<sector>_<season>, reg_lat_<sector>, reg_lon_<sector>, meta.regime_names, meta.regime_meta
        the weather-regime centroids (build_regimes.py: k-means, k = 4, 5-day-mean NCEP R1 500 hPa 1991-2020)
  s25_zg500, s25_mslp, shift_doy
        the ERA5 1991-2020 minus 2001-2020 normal at the 73 5-day centres, 2.5 deg (gefs_ref/era5_shift.npz)
  s25_zg100
        the same for 100 hPa height, built HERE by the identical method (daily ERA5 from the local store, mean over
        the days within +-7 of each centre, 1991-2020 minus 2001-2020, bilinear to 2.5 deg - the method reproduces
        era5_shift's s25_zg500 to r 0.99999998 / 0.002 m rms). The store is 0-90N only, which is all NAM100 needs.
  Height shifts are stored for rows 90N -> 0 only (every z500/z100 product is northern) and all shifts as float16.

    python build_gefs_patterns_ref.py [--out ~/data_archive/gefs_ref/patterns]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_patterns as GP                                           # noqa: E402

TC = Path.home() / "data_archive" / "geps_subx" / "telecon"
SHIFT = Path.home() / "data_archive" / "gefs_ref" / "era5_shift.npz"
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily" / "zplev"


def defined_months(meta):
    """telecon_geps.Patterns.defined, from CPC's tele_index.nh (1991-2020)."""
    lines = (TC / "cpc" / "tele_index.nh").read_text().splitlines()
    cols, rows = None, []
    for ln in lines:
        if cols is None and "NAO" in ln and "PNA" in ln:
            cols = [c for c in re.split(r"\s+", ln.strip()) if c][2:]
            continue
        m = re.match(r"^\s*(\d{4})\s+(\d{1,2})\s+(.*)$", ln)
        if not m or cols is None:
            continue
        vals = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(3))]
        rows.append((int(m.group(1)), int(m.group(2)), vals[:len(cols)]))
    out = {}
    for sid, m in meta.items():
        name = m.get("cpc")
        if m["field"] == "slp" or name not in cols:
            out[sid] = list(range(1, 13)); continue
        k = cols.index(name)
        cnt = {mo: 0 for mo in range(1, 13)}
        for y, mo, v in rows:
            if 1991 <= y <= 2020 and k < len(v) and v[k] > -99:
                cnt[mo] += 1
        out[sid] = [mo for mo in range(1, 13) if cnt[mo] >= 15]
    return out


def shift_z100():
    """ERA5 100 hPa height, 1991-2020 minus 2001-2020 normal at the 73 centres (+-7 d), bilinear to 2.5 deg."""
    A, D, Y = [], [], []
    for y in range(1991, 2021):
        d = xr.open_dataset(ERA5 / f"zplev_{y}.nc")
        a = d["zplev"].sel(level=100).transpose("time", "latitude", "longitude").load()
        A.append(a.values.astype("float32")); D.append(a.time.dt.dayofyear.values); Y.append(np.full(a.sizes["time"], y))
        lat, lon = a.latitude.values, a.longitude.values
        d.close()
    A, D, Y = np.concatenate(A), np.concatenate(D), np.concatenate(Y)
    assert 15000 < float(np.nanmean(A)) < 17500, "zplev at 100 hPa is not in metres"
    if lat[0] > lat[-1]:
        lat, A = lat[::-1], A[:, ::-1]
    nh = GP.LAT25[GP.LAT25 >= 0]
    g = np.stack(np.meshgrid(nh, GP.LON25, indexing="ij"), -1)
    out = np.zeros((len(GP.CENTRES), GP.LAT25.size, GP.LON25.size), "float32")
    for i, c in enumerate(GP.CENTRES):
        dd = np.abs(D - c); dd = np.minimum(dd, 365 - dd); w = dd <= 7
        s = A[w].mean(0) - A[w & (Y >= 2001)].mean(0)
        f = np.concatenate([s, s[:, :1]], axis=1)
        out[i, :nh.size] = RegularGridInterpolator((lat, np.r_[lon, 360.0]), f)(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "data_archive" / "gefs_ref" / "patterns"))
    a = ap.parse_args()
    ds = xr.open_dataset(TC / "patterns.nc")
    assert np.allclose(ds.lat.values, GP.LAT25) and np.allclose(ds.lon.values, GP.LON25), "pattern grid differs"
    meta_p = json.loads((TC / "validation.json").read_text())
    save = {}
    for sid in ds.index.values:
        sid = str(sid)
        save[f"proj_{sid}"] = ds.projector.sel(index=sid).values.astype("float32")
        save[f"sigma_doy_{sid}"] = ds.sigma_doy.sel(index=sid).values.astype("float32")
    save["std_month_z500"] = ds.std_month_z500.values.astype("float32")
    rg = xr.open_dataset(TC / "regimes.nc")
    for sk in GP.SECTORS:
        save[f"reg_lat_{sk}"] = rg[f"lat_{sk}"].values.astype("float32")
        save[f"reg_lon_{sk}"] = rg[f"lon_{sk}"].values.astype("float32")
        for season in GP.SEASONS:
            save[f"reg_{sk}_{season}"] = rg[f"{sk}_{season}"].values.astype("float32")
    sh = np.load(SHIFT)
    assert list(sh["doy"]) == GP.CENTRES
    save["shift_doy"] = sh["doy"].astype("int16")
    # float16 (steps of 0.016 m / 0.125 Pa at the largest values); the height shifts only north of the equator (rows
    # 90 -> 0), where every z500 and z100 product lives - the loader pads the southern rows with zeros
    nh = int((GP.LAT25 >= 0).sum())
    save["s25_zg500"] = sh["s25_zg500"][:, :nh].astype("float16")
    save["s25_mslp"] = sh["s25_mslp"].astype("float16")
    save["s25_zg100"] = shift_z100()[:, :nh].astype("float16")
    names = json.loads((TC / "regime_names.json").read_text())
    meta = {"patterns": meta_p, "defined_months": defined_months(meta_p),
            "regime_names": {k: v for k, v in names.items() if not k.startswith("_")},
            "regime_meta": json.loads(rg.attrs["meta"]),
            "regime_constants": {"k": GP.K, "r_min": GP.R_MIN, "smooth": GP.SMOOTH,
                                 "sectors": GP.SECTORS, "seasons": GP.SEASONS},
            "sources": {"patterns": "geps_subx/telecon/patterns.nc + validation.json (NCEP/NCAR R1 1991-2020, CPC)",
                        "regimes": "geps_subx/telecon/regimes.nc + regime_names.json (NCEP/NCAR R1 1991-2020)",
                        "shift": "gefs_ref/era5_shift.npz s25_zg500/s25_mslp; s25_zg100 ERA5 +-7 d, NH only",
                        "units": "s25 in GEFS units (m, Pa); projectors in pattern units (z m, slp hPa)"}}
    save["meta_json"] = np.array(json.dumps(meta))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / GP.REF_NAME, **save)
    p = out / GP.REF_NAME
    print(f"{p}  {p.stat().st_size / 1e6:.2f} MB")
    print("defined months:", {k: v for k, v in meta["defined_months"].items() if len(v) < 12})
    s = save["s25_zg100"].astype("float32")
    print(f"s25_zg100 NH: mean {s.mean():+.1f} m, range {s.min():+.1f} .. {s.max():+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
