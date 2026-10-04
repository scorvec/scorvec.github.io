#!/usr/bin/env python3
"""The previous cycle, kept compactly between runs (STATE/runs, carried on the frames branch).

Three products compare this run with the previous extended cycle: the change maps (render_maps.change), the
previous-run mean on the teleconnection plumes (telecon_geps) and the previous-run outline on the regime chart
(regimes_geps). On the laptop the previous cycle's full live files simply stayed on disk (~1.6 GB a cycle); a runner
starts empty, so after each build this keeps exactly what those comparisons read, and nothing else:

  geps_anom1_<tag>_<cycle>.nc     the ensemble-mean anomaly on its OWN 1 deg grid (forecast.anomaly computes it there
                                  and only interpolates to 0.5 deg for output, so the 1 deg nodes ARE the field), plus
                                  the per-lead global mean; load_anom() rebuilds the 0.5 deg field the same way
  geps_members_<tag>_<cycle>.nc   member anomalies of zg500 / mslp / z100, 2.5 deg (teleconnections, regimes)

int16 with a range-derived scale (resolution ~ range / 65000). About 50 MB a cycle; the newest KEEP cycles are kept.

    python archive.py --cycle 20261001
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                                        # noqa: E402

TAGS = ("t2m", "pr", "zg500", "olr", "u850", "u200", "v850", "v200", "mslp")
MEMBER_TAGS = ("zg500", "mslp", "z100")
KEEP = 2


def _enc(da) -> dict:
    v = da.values
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    s = max((hi - lo) / 65000.0, 1e-12)
    return {"dtype": "int16", "scale_factor": s, "add_offset": 0.5 * (lo + hi), "_FillValue": np.int16(-32768),
            "zlib": True, "complevel": 5, "shuffle": True}


def write(cycle: str) -> list[str]:
    paths.ARCHIVE.mkdir(parents=True, exist_ok=True)
    out = []
    lat1, lon1 = np.arange(-90.0, 90.0 + 1e-6, 1.0), np.arange(0.0, 360.0, 1.0)
    for tag in TAGS:
        p = paths.LIVE / f"geps_live_{tag}_{cycle}.nc"
        if not p.exists():
            continue
        d = xr.open_dataset(p)
        a = d[f"{tag}_anom"]
        # the 0.5 deg field is a bilinear interpolation of the 1 deg anomaly, so at the 1 deg nodes it is that anomaly
        a1 = a.sel(latitude=lat1, longitude=lon1).astype("float32")
        ds = xr.Dataset({"anom": a1.drop_vars("global_mean")},
                        coords={"global_mean": ("L", a.global_mean.values),
                                "lat05": d.latitude.values, "lon05": d.longitude.values})
        ds.attrs = dict(d.attrs, archive="1 deg nodes of forecast.anomaly(); archive.load_anom rebuilds 0.5 deg")
        q = paths.ARCHIVE / f"geps_anom1_{tag}_{cycle}.nc"
        ds.to_netcdf(q, encoding={"anom": _enc(ds["anom"])})
        d.close()
        out.append(q.name)
    for tag in MEMBER_TAGS:
        p = paths.LIVE / f"geps_members_{tag}_{cycle}.nc"
        if not p.exists():
            continue
        d = xr.open_dataset(p)
        v = f"{tag}_anom"
        ds = d[[v]].load()
        ds.attrs = d.attrs
        q = paths.ARCHIVE / f"geps_members_{tag}_{cycle}.nc"
        ds.to_netcdf(q, encoding={v: _enc(ds[v])})
        d.close()
        out.append(q.name)
    prune()
    return out


def cycles(prefix: str) -> list[str]:
    rx = re.compile(re.escape(prefix) + r"_(\d{8})\.nc$")
    return sorted({m.group(1) for q in paths.ARCHIVE.glob(f"{prefix}_*.nc") if (m := rx.search(q.name))})


def prune(keep: int = KEEP) -> None:
    every = sorted({m.group(1) for q in paths.ARCHIVE.glob("*.nc") if (m := re.search(r"_(\d{8})\.nc$", q.name))})
    for c in every[:-keep]:
        for q in paths.ARCHIVE.glob(f"*_{c}.nc"):
            q.unlink()


def load_anom(tag: str, cycle: str):
    """The archived cycle's anomaly on the 0.5 deg output grid, as forecast.anomaly returned it."""
    q = paths.ARCHIVE / f"geps_anom1_{tag}_{cycle}.nc"
    if not q.exists():
        return None
    ds = xr.open_dataset(q)
    a = ds["anom"].load().astype("float64")
    a = xr.concat([a, a.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
    a = a.interp(latitude=ds.lat05.values, longitude=np.sort(ds.lon05.values % 360))
    return a.assign_coords(global_mean=("L", ds.global_mean.values))


def members_path(tag: str, cycle: str):
    """The live member file of a cycle, else its archived anomalies; None if neither."""
    for p in (paths.LIVE / f"geps_members_{tag}_{cycle}.nc", paths.ARCHIVE / f"geps_members_{tag}_{cycle}.nc"):
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    a = ap.parse_args()
    names = write(a.cycle)
    tot = sum((paths.ARCHIVE / n).stat().st_size for n in names)
    print(f"  archived {len(names)} files, {tot / 1e6:.1f} MB -> {paths.ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
