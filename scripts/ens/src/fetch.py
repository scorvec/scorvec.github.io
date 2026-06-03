#!/usr/bin/env python3
"""Per-ensemble fetchers → ensemble MEDIAN of z500 / t2m on the common North-America
grid, for each forecast lead. Returns an array (lead, lat, lon) in clim units
(z500 in dam, t2m in K). Phase 1 implements GEFS; IFS/AIFS/GEPS land in later phases.

    python src/fetch.py --ensemble gefs --var z500 --date 20260603 --run 00   # smoke test
"""
from __future__ import annotations
import argparse, sys, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from common import grid, LEADS, VARS

warnings.filterwarnings("ignore")


def _to_common(da: xr.DataArray, scale: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Squeeze a single-field DataArray to (lat, lon), scale to clim units, and
    bilinearly interpolate onto the variable's target grid (lon on 0..360)."""
    da = da.squeeze(drop=True)
    # normalise coord names
    ren = {}
    for c in da.coords:
        cl = c.lower()
        if cl in ("lat", "latitude"): ren[c] = "latitude"
        elif cl in ("lon", "longitude"): ren[c] = "longitude"
    da = da.rename(ren)
    if float(da.longitude.min()) < 0:                      # -180..180 → 0..360
        da = da.assign_coords(longitude=(da.longitude % 360)).sortby("longitude")
    da = da.sortby("latitude")
    out = da.interp(latitude=lat, longitude=lon, kwargs={"fill_value": None})
    return (out.values * scale).astype("float32")


# ─────────────────────────── GEFS (Herbie, 31 members) ───────────────────────────
GEFS_MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]


def fetch_gefs(date: str, run: str, var: str, workers: int = 12) -> np.ndarray:
    from herbie import Herbie
    v = VARS[var]
    lat, lon = grid(var)
    cycle = f"{date[:4]}-{date[4:6]}-{date[6:8]} {run}:00"
    stacks: dict[int, list] = {ld: [] for ld in LEADS}

    def one(args):
        ld, m = args
        try:
            H = Herbie(cycle, model="gefs", member=m, product="atmos.5", fxx=ld, verbose=False)
            if H.grib is None:
                return ld, None
            ds = H.xarray(v["gefs_search"], remove_grib=True)
            da = ds[list(ds.data_vars)[0]] if isinstance(ds, xr.Dataset) else ds
            return ld, _to_common(da, v["model_scale"], lat, lon)
        except Exception as e:                              # noqa: BLE001
            return ld, None

    tasks = [(ld, m) for ld in LEADS for m in GEFS_MEMBERS]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ld, fld in ex.map(one, tasks):
            if fld is not None:
                stacks[ld].append(fld)
    med = np.full((len(LEADS), len(lat), len(lon)), np.nan, "float32")
    for i, ld in enumerate(LEADS):
        if stacks[ld]:
            med[i] = np.median(np.stack(stacks[ld]), axis=0)
        print(f"  GEFS {var} f{ld:03d}: {len(stacks[ld])} members", flush=True)
    return med


FETCHERS = {"gefs": fetch_gefs}


def fetch(ensemble: str, date: str, run: str, var: str) -> np.ndarray:
    return FETCHERS[ensemble](date, run, var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True, choices=list(FETCHERS))
    ap.add_argument("--var", required=True, choices=list(VARS))
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    args = ap.parse_args()
    med = fetch(args.ensemble, args.date, args.run, args.var)
    valid = ~np.isnan(med)
    print(f"{args.ensemble} {args.var}: shape {med.shape}, "
          f"{valid.any(axis=(1,2)).sum()}/{len(LEADS)} leads, "
          f"range {np.nanmin(med):.1f}…{np.nanmax(med):.1f} {VARS[args.var]['units']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
