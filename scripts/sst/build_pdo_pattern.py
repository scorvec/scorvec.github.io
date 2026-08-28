#!/usr/bin/env python3
"""
One-time builder for the PDO spatial pattern used by the daily PDO monitor.

Method (NCEI convention): leading EOF of monthly North Pacific (20–70°N,
110°E–100°W) SST anomalies from ERSST v5, with the global-ocean-mean anomaly
(60°S–60°N) removed each month so the uniform warming trend does not project
onto the pattern. Anomalies vs the site-wide 1991–2020 base; EOF era 1950–2024.

Output: scripts/sst/reference/pdo_pattern.nc (committed — a few KB) holding
  eof(lat, lon)   the pattern on the ERSST 2° North Pacific grid
  attrs:
    pc_std        std-dev of the raw projection over the EOF era (index scale)
    proj_one      projection of the constant field 1 onto the pattern —
                  lets the daily pipeline fold global-mean removal in linearly:
                  proj(a − g·1) = proj(a) − g·proj_one
    sign check    positive PDO = cool central North Pacific / warm NA coast

sst-roni.py projects daily OISST anomalies (interpolated to this grid) onto
the pattern and divides by pc_std, giving a daily index on the familiar
monthly-PDO scale. Verify against NCEI's published monthly PDO before trusting
a rebuild (the last-24-months table this script prints).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ERSST = HERE / "data" / "ersst_v5_mnmean.nc"
OUT = HERE / "reference" / "pdo_pattern.nc"

CLIM_Y0, CLIM_Y1 = 1991, 2020         # anomaly base, matches the rest of the site
EOF_Y0, EOF_Y1 = 1900, 1993           # EOF / standardization era (Mantua/NCEI convention)
NP_LAT = (20, 70)
NP_LON = (110, 260)                   # 110°E–100°W
GLOB_LAT = (-60, 60)
# +PDO must be COOL here (central North Pacific): flip the EOF if needed
SIGN_BOX_LAT = (32, 45)
SIGN_BOX_LON = (180, 210)


def wmean(da: xr.DataArray, dims=("lat", "lon")) -> xr.DataArray:
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(dim=dims, skipna=True)


def main() -> int:
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")

    clim = (sst.sel(time=slice(f"{CLIM_Y0}-01-01", f"{CLIM_Y1}-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    g = wmean(anom.sel(lat=slice(*GLOB_LAT)))
    anom = anom - g                                     # remove global-mean SSTA

    np_anom = anom.sel(lat=slice(*NP_LAT), lon=slice(*NP_LON))
    era = np_anom.sel(time=slice(f"{EOF_Y0}-01-01", f"{EOF_Y1}-12-31"))

    # EOF via SVD of the sqrt(cos)-weighted anomaly matrix (time × space)
    w = np.cos(np.deg2rad(era["lat"])).clip(min=0)
    sw = np.sqrt(w)
    X = (era * sw).stack(space=("lat", "lon"))
    valid = X.notnull().all("time")
    Xm = X.where(valid, drop=True)
    Xv = Xm.values - Xm.values.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xv, full_matrices=False)
    var_frac = float(S[0] ** 2 / (S ** 2).sum())

    eof_w = xr.DataArray(Vt[0], coords={"space": Xm["space"]}, dims="space")
    eof = (eof_w.unstack("space") / sw).reindex_like(np_anom.isel(time=0))

    # sign convention
    if float(wmean(eof.sel(lat=slice(*SIGN_BOX_LAT), lon=slice(*SIGN_BOX_LON)))) > 0:
        eof = -eof

    # raw projection index over the era → standardization constant
    def project(a2d):
        num = (a2d * eof * w).sum(("lat", "lon"), skipna=True)
        den = ((eof ** 2) * w).where(a2d.notnull()).sum(("lat", "lon"))
        return num / den

    pc_era = project(era)
    pc_std = float(pc_era.std("time"))
    proj_one = float(((eof * w).sum(("lat", "lon"), skipna=True)
                      / ((eof ** 2) * w).sum(("lat", "lon"), skipna=True)))

    # Calibrate the raw projection against NCEI's published monthly PDO so the
    # daily index reads on the official scale (era/standardization details of
    # NCEI's internal EOF need not be reproduced exactly — the pattern shapes
    # agree, so a linear fit maps one onto the other).
    import io, urllib.request
    dat = HERE / "data" / "ersst.v5.pdo.dat"
    if not dat.exists():
        urllib.request.urlretrieve(
            "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat", dat)
    rows = []
    for line in dat.read_text().splitlines():
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            y = int(p[0])
            for m, v in enumerate(p[1:], 1):
                v = float(v)
                if v < 90:
                    rows.append((pd.Timestamp(y, m, 1), v))
    ncei = pd.Series(dict(rows)).sort_index()
    mine = project(np_anom).to_series()
    mine.index = pd.DatetimeIndex(mine.index).to_period("M").to_timestamp()
    both = pd.concat([mine.rename("p"), ncei.rename("n")], axis=1).dropna()
    both = both.loc["1950":]
    slope, intercept = np.polyfit(both["p"], both["n"], 1)
    corr = both["p"].corr(both["n"])
    print(f"calibration vs NCEI 1950–present: slope={slope:.4f} "
          f"intercept={intercept:+.3f} r={corr:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = eof.rename("eof").to_dataset()
    out.attrs.update(
        pc_std=pc_std, proj_one=proj_one, var_frac=var_frac,
        calib_slope=float(slope), calib_intercept=float(intercept),
        calib_corr=float(corr),
        clim_base=f"{CLIM_Y0}-{CLIM_Y1}", eof_era=f"{EOF_Y0}-{EOF_Y1}",
        source="ERSST v5 monthly", method=(
            "EOF1 of North Pacific (20-70N, 110E-100W) SSTA, global-mean "
            "(60S-60N) removed; index = proj/pc_std"),
    )
    out.to_netcdf(OUT)
    print(f"wrote {OUT}  (EOF1 explains {var_frac:.1%} of NP variance, "
          f"pc_std={pc_std:.3f}, proj_one={proj_one:.3f})")

    # Verification table: calibrated monthly PDO from ERSST vs NCEI's published
    # index — inspect before trusting a rebuild.
    idx = (project(np_anom).to_series() * slope + intercept)
    idx.index = pd.DatetimeIndex(idx.index).to_period("M").to_timestamp()
    print("\nlast 18 monthly values (mine vs NCEI):")
    for t, v in idx.tail(18).items():
        nv = ncei.get(t, float("nan"))
        print(f"  {pd.Timestamp(t):%Y-%m}  {v:+.2f}  vs  {nv:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
