#!/usr/bin/env python3
"""GDPS precipitation over South America, verified against IMERG and MERGE.

The Colombia test was six small basins in one wet regime. This is the
continent, and it separates two things that were confounded there:

  * **IMERG** is satellite retrieval, available everywhere.
  * **MERGE** is that same GPM field merged with Brazil's surface gauge
    network by CPTEC. Over Brazil it is the better truth, and it sits on
    the identical 0.1 deg grid, so no regridding is involved.

Where GDPS scores differently against the two, the gap is a statement
about the *satellite*, not the model - which is worth knowing before
trusting any single-reference verdict.

GDPS (0.15 deg) is area-averaged onto the 0.1 deg reference grid's cells
by nearest-neighbour lookup, which for a coarse-to-fine mapping is exact
rather than interpolated.

    python scripts/sst/gdps_sa_verify.py --leads 1 3 5

Output: ~/brazil_hydro/out/gdps_sa_verify.json
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402

MERGE_DIR = Path.home() / "brazil_hydro" / "raw" / "merge"
OUT = Path.home() / "brazil_hydro" / "out" / "gdps_sa_verify.json"
BASE = ("https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/00/{lead:03d}/"
        "{date}T00Z_MSC_GDPS_Precip-Accum24h_Sfc_LatLon0.15_PT{lead:03d}H.grib2")

# regions: name -> (lon_min, lon_max, lat_min, lat_max) in -180..180
REGIONS = {
    "N Brazil / Amazon":   (-70, -50,  -8,   2),
    "NE Brazil":           (-45, -35, -12,  -3),
    "SE Brazil":           (-50, -40, -24, -15),
    "S Brazil":            (-56, -49, -32, -24),
    "C Brazil / Cerrado":  (-55, -45, -18,  -9),
    "Colombia":            (-78, -72,   2,  10),
    "N Argentina":         (-66, -58, -34, -26),
    "Andes / Peru-Bol":    (-72, -64, -18,  -8),
}


def gdps_grid(date: str, lead_h: int):
    import xarray as xr
    u = BASE.format(date=date, lead=lead_h)
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "scorvec-hydro/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    tmp = Path(tempfile.mkstemp(suffix=".grib2")[1])
    tmp.write_bytes(data)
    try:
        d = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
        v = list(d.data_vars)[0]
        arr = d[v].values
        lat = d.latitude.values
        lon = d.longitude.values
    except Exception:                                    # noqa: BLE001
        return None
    finally:
        tmp.unlink(missing_ok=True)
    return arr, lat, lon


def merge_grid(day: str):
    import xarray as xr
    f = MERGE_DIR / f"MERGE_CPTEC_{day}.grib2"
    if not f.exists():
        return None
    try:
        d = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception:                                    # noqa: BLE001
        return None
    v = list(d.data_vars)[0]
    return d[v].values, d.latitude.values, d.longitude.values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--days", type=int, default=25)
    a = ap.parse_args(argv)

    ml, mt = IP._grid_axes()
    ilon, ilat = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    today = datetime.now(timezone.utc).replace(tzinfo=None)

    acc = {}                       # (region, lead, ref) -> [f, o] pairs
    for k in range(a.days, 0, -1):
        valid = today - timedelta(days=k)
        vs = valid.strftime("%Y%m%d")
        npy = IP.DAILY_CACHE / f"{vs}.npy"
        img = np.load(npy) if npy.exists() else None
        mg = merge_grid(vs)
        for lead in a.leads:
            init = (valid - timedelta(days=lead - 1)).strftime("%Y%m%d")
            g = gdps_grid(init, lead * 24)
            if g is None:
                continue
            garr, glat, glon = g
            gl = np.where(glon > 180, glon - 360, glon)
            for rn, (lo0, lo1, la0, la1) in REGIONS.items():
                gi = (glat >= la0) & (glat <= la1)
                gj = (gl >= lo0) & (gl <= lo1)
                if gi.sum() == 0 or gj.sum() == 0:
                    continue
                gm = float(np.nanmean(garr[np.ix_(gi, gj)]))
                if img is not None:
                    ii = (ilat >= la0) & (ilat <= la1)
                    jj = (ilon >= lo0) & (ilon <= lo1)
                    om = float(np.nanmean(img[np.ix_(ii, jj)]))
                    acc.setdefault((rn, lead, "IMERG"), []).append((gm, om))
                if mg is not None:
                    marr, mlat, mlon = mg
                    mlo = np.where(mlon > 180, mlon - 360, mlon)
                    mi = (mlat >= la0) & (mlat <= la1)
                    mj = (mlo >= lo0) & (mlo <= lo1)
                    if mi.sum() and mj.sum():
                        om = float(np.nanmean(marr[np.ix_(mi, mj)]))
                        if np.isfinite(om):
                            acc.setdefault((rn, lead, "MERGE"), []).append((gm, om))
        print(f"  {vs} done", flush=True)

    import json
    res = {}
    print(f"\n{'region':22}{'lead':>5}{'ref':>7}{'n':>5}{'GDPS':>8}{'obs':>8}"
          f"{'bias':>7}{'MAE':>7}{'r':>7}")
    print("-" * 78)
    for (rn, lead, ref), v in sorted(acc.items()):
        if len(v) < 8:
            continue
        f = np.array([x[0] for x in v]); o = np.array([x[1] for x in v])
        m = np.isfinite(f) & np.isfinite(o)
        f, o = f[m], o[m]
        bias = f.mean() / o.mean() if o.mean() > 0.05 else np.nan
        mae = np.mean(np.abs(f - o))
        rr = np.corrcoef(f, o)[0, 1] if len(f) > 5 else np.nan
        res[f"{rn}|{lead}|{ref}"] = dict(n=len(f), gdps=f.mean(), obs=o.mean(),
                                         bias=bias, mae=mae, r=rr)
        print(f"{rn:22}{lead:5}{ref:>7}{len(f):5}{f.mean():8.2f}{o.mean():8.2f}"
              f"{bias:7.2f}{mae:7.2f}{rr:7.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
