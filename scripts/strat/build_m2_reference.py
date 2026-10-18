#!/usr/bin/env python3
"""Reduce the yearly MERRA-2 stratosphere files to the three daily series nh_vortex.py's reference band is built from.

nh_vortex.load_clim needs only three DAILY SCALAR series from MERRA-2 (1980 on): u60 at 10 hPa, u60 at 100 hPa and
the cos-weighted 65-90N polar-cap mean of zbar at 100 hPa. At ~17k days that is a few hundred KB, so the series is
committed as reference/m2_strat_daily.nc and nothing else of MERRA-2 is needed on a runner.

What is kept is the RAW daily series, deliberately not the day-of-year percentiles: the band is epoch-referenced
(load_clim detrends the cap height to the current year at runtime, so decades of stratospheric height rise do not
manufacture a "weak vortex" out of climate drift). Caching post-detrend numbers would freeze that correction.

Input (offline, not in the repository): $DATA_ARCHIVE/m2_strat/m2_strat_YYYY.nc, one file a year (~15 MB) holding
u60(time, lev), the zonal-mean u at 60N, and zbar(time, lev, lat), the zonal-mean geopotential height, as daily means
of the 00/06/12/18 UTC MERRA-2 M2I3NPASM fields. Run by hand on the machine holding that archive whenever it is
extended:

    python scripts/strat/build_m2_reference.py          # rebuild the series, then check it against the archive
    python scripts/strat/build_m2_reference.py --check  # check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "reference" / "m2_strat_daily.nc"

# CAP is IMPORTED, never restated: a copy that drifted from nh_vortex's (65, 90) would change the published
# reference band, and the equivalence check below would no longer test the quantity the figure uses.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
from nh_vortex import CAP
from localdirs import env_dir


def reduce_raw() -> pd.DataFrame:
    """The three daily series from the yearly files."""
    src = env_dir("DATA_ARCHIVE", "m2_strat")
    fs = sorted(src.glob("m2_strat_*.nc"))
    if not fs:
        raise SystemExit(f"no m2_strat_YYYY.nc files under {src}")
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
    print(f"  raw: {len(fs)} files -> {len(df)} days {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}")
    return df


def write(df: pd.DataFrame) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({c: ("time", df[c].values.astype("float32")) for c in df.columns}, coords={"time": df.index.values})
    ds.attrs["source"] = "MERRA-2 daily, reduced from the yearly m2_strat files by scripts/strat/build_m2_reference.py"
    ds.attrs["cap"] = f"{CAP[0]}-{CAP[1]}N cos-weighted, zbar at 100 hPa"
    ds.attrs["note"] = "pre-detrend dailies; nh_vortex applies the epoch referencing at runtime"
    enc = {c: {"zlib": True, "complevel": 6} for c in df.columns}
    ds.to_netcdf(OUT, encoding=enc)
    print(f"  wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")


def check(df: pd.DataFrame) -> int:
    """Cached series must reproduce the raw reduction to float32 precision."""
    if not OUT.exists():
        print("  no cache to check")
        return 1
    c = xr.open_dataset(OUT)
    ci = pd.DatetimeIndex(c.time.values)
    bad = 0
    for k in ("u10", "u100", "zcap"):
        a = df[k].reindex(ci)
        b = pd.Series(np.asarray(c[k].values), index=ci)
        both = a.notna() & b.notna()
        dmax = float(np.nanmax(np.abs((a - b)[both]))) if both.any() else np.nan
        ok = np.isfinite(dmax) and dmax < 1e-2
        print(f"  {k:5s} n={int(both.sum()):6d}  max|cache-raw| = {dmax:.2e}  {'OK' if ok else 'MISMATCH'}")
        bad += 0 if ok else 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the cache against the raw archive")
    a = ap.parse_args()
    df = reduce_raw()
    if a.check:
        return check(df)
    write(df)
    return check(df)


if __name__ == "__main__":
    sys.exit(main())
