#!/usr/bin/env python3
"""Per-mode pc_wind_std from the 2003-2024 ARCO/ERA5 equatorial-u sweep.

Replicates the operational wind-only RMM path (rmm.py) over the arco_eq_u
archive: 15S-15N cos-weighted band means -> day-of-year climatology removal
(climatology.nc) -> trailing 120-day calendar mean removal (recent_analysis
convention: rolling window=120, min_periods=120) -> divide by std_u850/std_u200
-> project onto the wind portions of the reference EOFs. Sets pc_wind_std per
mode to the ddof=1 std of THAT series — the WH04 standardization measured on
the same reanalysis stream operations consume, replacing the x2.195 stopgap
calibrated on a 26-day OMI overlap (2026-08-10).

    python src/recalibrate_pc_wind_std.py            # compute + validate only
    python src/recalibrate_pc_wind_std.py --apply    # also rewrite eofs.nc
"""
from pathlib import Path
import argparse
import glob

import numpy as np
import pandas as pd
import xarray as xr

REF = Path(__file__).resolve().parents[1] / "data" / "reference"


YEARS = range(2003, 2025)


def band_series():
    files = [p for y in YEARS
             if (p := str(REF / "arco_eq_u" / f"arco_eq_u_{y}.nc"))
             and (REF / "arco_eq_u" / f"arco_eq_u_{y}.nc").exists()]
    ds = xr.open_mfdataset(files, combine="by_coords")["uband"].load()
    u850 = ds.sel(lev=850, drop=True)
    u200 = ds.sel(lev=200, drop=True)
    return u850, u200


def trailing120(a: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    full = pd.DataFrame(a, index=times).asfreq("D")
    roll = full.rolling(window=120, min_periods=120).mean()
    return roll.loc[times].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    clim = xr.open_dataset(REF / "climatology.nc")
    eofs = xr.open_dataset(REF / "eofs.nc")
    lon = eofs.longitude.values

    u850, u200 = band_series()
    # 1 deg -> the 2.5 deg EOF/climatology longitude grid (cyclic-safe: 0..359 covers 0..357.5)
    u850 = u850.interp(lon=lon).values
    u200 = u200.interp(lon=lon).values
    times = pd.DatetimeIndex(
        xr.open_mfdataset([str(REF / "arco_eq_u" / f"arco_eq_u_{y}.nc")
                           for y in YEARS], combine="by_coords").time.values)
    doy = times.dayofyear.to_numpy()

    a850 = u850 - clim["clim_u850"].sel(dayofyear=doy).values
    a200 = u200 - clim["clim_u200"].sel(dayofyear=doy).values
    a850 = a850 - trailing120(a850, times)
    a200 = a200 - trailing120(a200, times)
    a850 /= clim.attrs["std_u850"]
    a200 /= clim.attrs["std_u200"]

    e1 = np.concatenate([eofs["eof_u850"].sel(mode=1).values,
                         eofs["eof_u200"].sel(mode=1).values])
    e2 = np.concatenate([eofs["eof_u850"].sel(mode=2).values,
                         eofs["eof_u200"].sel(mode=2).values])
    combined = np.concatenate([a850, a200], axis=1)
    ok = np.isfinite(combined).all(axis=1)
    pc1, pc2 = combined[ok] @ e1, combined[ok] @ e2
    t_ok = times[ok]

    new = np.array([np.std(pc1, ddof=1), np.std(pc2, ddof=1)])
    cur = eofs["pc_wind_std"].values.astype(float)
    print(f"days used: {ok.sum()} ({t_ok[0].date()} .. {t_ok[-1].date()})")
    print(f"pc_wind_std  current (x2.195 stopgap): {cur[0]:.3f}, {cur[1]:.3f}")
    print(f"pc_wind_std  recalibrated 2003-2024:   {new[0]:.3f}, {new[1]:.3f}")
    print(f"amplitude change factor (cur/new):     {cur[0]/new[0]:.4f}, {cur[1]/new[1]:.4f}")

    # Validation vs BOM official RMM (frozen telecon copy; 1974-2024-02).
    bom = pd.read_csv(
        Path(__file__).resolve().parents[2] / "telecon" / "data" /
        "predictor_store" / "rmm_history.txt",
        skiprows=2, sep=r"\s+", usecols=range(7), na_values=[999, 1e36],
        names=["y", "m", "d", "rmm1", "rmm2", "phase", "amp"])
    bom = bom[bom.rmm1.abs() < 100]
    ot = pd.to_datetime(bom[["y", "m", "d"]].rename(
        columns={"y": "year", "m": "month", "d": "day"}))
    bom.index = pd.DatetimeIndex(ot)
    common = t_ok.intersection(bom.index)
    io = bom.index.get_indexer(common); im = t_ok.get_indexer(common)
    r1o = bom["rmm1"].values[io]; r2o = bom["rmm2"].values[io]
    m = np.isfinite(r1o) & np.isfinite(r2o)
    r1n, r2n = pc1[im][m] / new[0], pc2[im][m] / new[1]
    r1o, r2o = r1o[m], r2o[m]
    amp_n = np.hypot(r1n, r2n); amp_o = np.hypot(r1o, r2o)
    print(f"\nvalidation vs BOM RMM, {m.sum()} common days "
          f"({common[0].date()} .. {common[-1].date()}):")
    print(f"  corr RMM1 {np.corrcoef(r1n, r1o)[0,1]:.3f}   "
          f"RMM2 {np.corrcoef(r2n, r2o)[0,1]:.3f}")
    print(f"  median amplitude ratio ours/BOM: {np.median(amp_n/amp_o):.3f}")
    print(f"  std ratio RMM1 {np.std(r1n)/np.std(r1o):.3f}  "
          f"RMM2 {np.std(r2n)/np.std(r2o):.3f}")

    if args.apply:
        eofs = eofs.load(); eofs.close()
        eofs["pc_wind_std"].values[:] = new
        eofs.attrs["pc_wind_std_note"] = (
            "per-mode std of operational wind-only projection over ARCO/ERA5 "
            "2003-2024 (recalibrate_pc_wind_std.py, 2026-08-12); replaces "
            "x2.195 OMI-overlap stopgap")
        tmp = REF / "eofs.nc.tmp"
        eofs.to_netcdf(tmp); tmp.replace(REF / "eofs.nc")
        print("\neofs.nc updated.")


if __name__ == "__main__":
    main()
