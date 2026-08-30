#!/usr/bin/env python3
"""Reduce the raw MERRA-2 stratosphere archive to the three series nh_vortex needs.

WHY THIS EXISTS: nh_vortex.py referenced scripts/telecon/data/m2_strat, which is
690 MB across 47 yearly files and gitignored - laptop-only. When the product
moved to GitHub Actions on 2026-08-29 it failed on every run with "no MERRA-2
files under .../m2_strat" while still exiting green, because the step is
wrapped in `|| echo ::warning::`.

Committing 690 MB is not an option (the repo was collapsed from 11 GB in Aug
2026). But load_clim() only ever reduces that archive to three DAILY SCALAR
series - u60 at 10 hPa, u60 at 100 hPa, and the cos-weighted polar-cap
mean of zbar at 100 hPa. At ~17k days that is a few hundred KB, small enough to
track.

What is cached is the RAW daily series, deliberately not the day-of-year
percentiles. The percentile band is epoch-referenced: load_clim detrends to the
current epoch at runtime so that 45 years of stratospheric height rise does not
manufacture a "weak vortex" out of climate drift. Caching post-detrend numbers
would freeze that correction at build time and slowly go wrong.

    python scripts/strat/build_m2_reference.py          # rebuild the cache
    python scripts/strat/build_m2_reference.py --check   # verify against raw
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
M2 = REPO / "scripts" / "telecon" / "data" / "m2_strat"
OUT = HERE / "reference" / "m2_strat_daily.nc"
OUT_OBS = HERE / "reference" / "strat_obs_daily.nc"

# CAP is IMPORTED, never restated. A first version of this builder hardcoded
# (60, 90) while nh_vortex uses (65, 90); the cache then held a different
# quantity to the one the raw path computes, which would have silently shifted
# the published 100 hPa height-anomaly reference band (trend +18.69 vs +17.77
# per decade). The equivalence check below caught it - keep both.
sys.path.insert(0, str(HERE))
from nh_vortex import CAP  # noqa: E402


def reduce_raw() -> pd.DataFrame:
    """The exact reduction load_clim() used to do against the raw archive."""
    fs = sorted(glob.glob(str(M2 / "m2_strat_*.nc")))
    if not fs:
        raise SystemExit(f"no MERRA-2 files under {M2} - this builder needs the "
                         f"raw archive and only runs on the laptop")
    d = xr.open_mfdataset(fs, combine="by_coords", chunks=None)
    idx = pd.DatetimeIndex(d.time.values)
    zb = d["zbar"].sel(lev=100.0)
    sel = zb.where((zb.lat >= CAP[0]) & (zb.lat <= CAP[1]), drop=True)
    w = np.cos(np.deg2rad(sel.lat))
    df = pd.DataFrame(
        {
            "u10": np.asarray(d["u60"].sel(lev=10.0).values),
            "u100": np.asarray(d["u60"].sel(lev=100.0).values),
            "zcap": np.asarray(((sel * w).sum("lat") / w.sum()).values),
        },
        index=idx,
    )
    print(f"  raw: {len(fs)} files -> {len(df)} days "
          f"{df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}")
    return df


def write(df: pd.DataFrame) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({c: ("time", df[c].values.astype("float32")) for c in df.columns},
                    coords={"time": df.index.values})
    ds.attrs["source"] = ("MERRA-2 daily, reduced from scripts/telecon/data/m2_strat "
                          "by scripts/strat/build_m2_reference.py")
    ds.attrs["cap"] = f"{CAP[0]}-{CAP[1]}N cos-weighted, zbar at 100 hPa"
    ds.attrs["note"] = ("pre-detrend dailies; nh_vortex applies the epoch "
                        "referencing at runtime")
    enc = {c: {"zlib": True, "complevel": 6} for c in df.columns}
    ds.to_netcdf(OUT, encoding=enc)
    print(f"  wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")


def check(df: pd.DataFrame) -> int:
    """Cached series must reproduce the raw reduction to float32 precision."""
    if not OUT.exists():
        print("  no cache to check"); return 1
    c = xr.open_dataset(OUT)
    ci = pd.DatetimeIndex(c.time.values)
    bad = 0
    for k in ("u10", "u100", "zcap"):
        a = df[k].reindex(ci)
        b = pd.Series(np.asarray(c[k].values), index=ci)
        both = a.notna() & b.notna()
        dmax = float(np.nanmax(np.abs((a - b)[both]))) if both.any() else np.nan
        ok = np.isfinite(dmax) and dmax < 1e-2
        print(f"  {k:5s} n={int(both.sum()):6d}  max|cache-raw| = {dmax:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")
        bad += 0 if ok else 1
    return bad


def build_obs() -> int:
    """Cache the observed ERA5 tail nh_vortex.analysis() derives.

    Reuses nh_vortex._analysis_raw() rather than restating the reduction, for
    the same reason CAP is imported: a restated copy drifts. strat_obs.nc is a
    ROLLING 150-day window, so unlike the MERRA-2 cache this one goes stale -
    rerun it whenever the local pipeline refreshes strat_obs.nc, and nh_vortex
    prints the cache age (flagging it past 10 days).
    """
    import nh_vortex as NV
    obs = NV._analysis_raw()
    if not obs:
        print(f"  no strat_obs.nc under {NV.DATA} - skipping the observed cache")
        return 0
    df = pd.DataFrame({k: obs[k] for k in ("u10", "zcap")}).dropna(how="all")
    ds = xr.Dataset({c: ("time", df[c].values.astype("float32")) for c in df.columns},
                    coords={"time": df.index.values})
    ds.attrs["source"] = "ERA5 via scripts/strat/data/strat_obs.nc (rolling window)"
    ds.attrs["note"] = "reduced by nh_vortex._analysis_raw(); refresh when strat_obs.nc updates"
    OUT_OBS.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT_OBS, encoding={c: {"zlib": True, "complevel": 6} for c in df.columns})
    print(f"  wrote {OUT_OBS.relative_to(REPO)}  ({OUT_OBS.stat().st_size / 1024:.0f} KB, "
          f"{df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}, {len(df)} days)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the cache against the raw archive")
    a = ap.parse_args()
    df = reduce_raw()
    if a.check:
        return check(df)
    write(df)
    rc = check(df)
    build_obs()
    return rc


if __name__ == "__main__":
    sys.exit(main())
