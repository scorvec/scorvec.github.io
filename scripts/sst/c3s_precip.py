#!/usr/bin/env python3
"""C3S seasonal RAINFALL over the Colombia basins — forecast + hindcast.

The monthly inflow model currently infers months 2-6 from ENSO, storage
and antecedent wetness.  A dynamical seasonal model forecasts the rain
directly, so this pulls C3S monthly precipitation (ECMWF SEAS5, 51
members, 6 lead months) over the Colombia box and reduces it to a
basin-mean rain anomaly per member.

BIAS CORRECTION.  A seasonal model's absolute precipitation is
unusable raw - its climatology differs from the observed one by tens of
percent and drifts with lead time.  Each member is therefore expressed
as a RATIO to that system's own hindcast climatology for the same
(target month, lead), then that ratio is applied to the IMERG observed
climatology.  Model drift cancels; only the anomaly signal is carried
across.  This is why the 1993-2016 hindcast is retrieved rather than the
forecast alone - it is also what makes the addition validatable, since
the same hindcasts give ~24 years of out-of-sample monthly cases.

    python scripts/sst/c3s_precip.py --hindcast     # 1993-2016, all init months
    python scripts/sst/c3s_precip.py                # latest forecast only

Outputs: ~/colombia_hydro/raw/c3s_precip/*.grib  (cache, kept)
         ~/colombia_hydro/out/c3s_basin_precip.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
CACHE = PRIV / "raw" / "c3s_precip"
OUT = PRIV / "out" / "c3s_basin_precip.json"
DATASET = "seasonal-monthly-single-levels"
CENTRE, SYSTEM = "ecmwf", "51"
AREA = [13, -80, -5, -66]              # N, W, S, E — the Colombia basins
GRID = "0.5/0.5"
LEADS = ["1", "2", "3", "4", "5", "6"]
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
HIND_YEARS = [str(y) for y in range(1993, 2017)]


def retrieve(years, month, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    import cdsapi
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {"originating_centre": CENTRE, "system": SYSTEM,
           "variable": "total_precipitation", "product_type": "monthly_mean",
           "year": list(years), "month": [month], "leadtime_month": LEADS,
           "area": AREA, "grid": GRID, "data_format": "grib"}
    try:
        cdsapi.Client(timeout=1800, quiet=True, progress=False,
                      wait_until_complete=True, retry_max=1
                      ).retrieve(DATASET, req, str(dest))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:                              # noqa: BLE001
        print(f"  {month} {years[0]}..{years[-1]}: {repr(e)[:110]}", flush=True)
        return False


def coarse_weights(lon, lat):
    """Energy weights on a grid too coarse for polygon containment.

    On the C3S 0.5 deg grid the smaller catchments (CARIBE is one river)
    can enclose no cell centre at all, so the polygon-containment path
    returns nothing.  Here each river is assigned to the nearest cell to
    its catchment centroid and weighted by its generation energy, which
    is the honest reading of a coarse field: the basin value is the model
    grid box that contains the catchment, not a sub-grid average.
    """
    import json as _json
    from hydro_region_rain import (CATCH_GJ, _river_energy, _regulated_rivers)
    egy, reg = _river_energy(), _regulated_rivers()
    gj = _json.loads(Path(CATCH_GJ).read_text())
    lon = np.asarray(lon); lat = np.asarray(lat)
    acc = {r: np.zeros((len(lat), len(lon))) for r in ORDER}
    for ft in gj["features"]:
        pr = ft["properties"]
        rg, riv = pr.get("region"), pr.get("river")
        if rg not in acc or riv in reg:
            continue
        e = float(egy.get(riv, 0.0))
        if e <= 0:
            continue
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        pts = np.vstack([np.asarray(pl[0]) for pl in polys
                         if np.asarray(pl[0]).ndim == 2])
        clon, clat = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        j = int(np.argmin(np.abs(((lon - clon + 180) % 360) - 180)))
        i = int(np.argmin(np.abs(lat - clat)))
        acc[rg][i, j] += e
    out = {}
    for r in ORDER:
        t = acc[r].sum()
        if t <= 0:
            return None
        out[r] = acc[r] / t
    return out


def basin_series(path: Path):
    """[(init, lead, member, basin, mm/day)] from one seasonal GRIB.

    Dimension order is NOT assumed.  cfgrib returns these files as
    (number, time, step, lat, lon) - member first, then init year - and
    `step` is a RAGGED UNION of day-offsets, because a given lead month
    is a different number of days after a leap-year init than a normal
    one.  Positional indexing therefore silently reads members as years
    and invents leads that were never requested.  Lead is instead derived
    from valid_time minus init, in whole months, and empty cells dropped.
    """
    import xarray as xr
    from hydro_region_rain import region_weights_energy
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds["tprate"] * 86400.0 * 1000.0                # m/s -> mm/day
    lon = da.longitude.values
    if lon.max() > 180:
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
    da = da.sortby("longitude").sortby("latitude")
    W = region_weights_energy(da.longitude.values, da.latitude.values, ORDER)
    if W is None:
        W = coarse_weights(da.longitude.values, da.latitude.values)
    if W is None:
        raise RuntimeError("no usable basin weights on the C3S grid")

    for d_ in ("number", "time", "step"):
        if d_ not in da.dims:
            da = da.expand_dims(d_)
    da = da.transpose("number", "time", "step", "latitude", "longitude")
    arr = da.values
    nums = np.atleast_1d(da["number"].values)
    times = np.atleast_1d(np.asarray(da["time"].values, dtype="datetime64[D]"))
    vt = ds["valid_time"].values                        # (time, step) or (step,)
    vt = np.atleast_2d(np.asarray(vt, dtype="datetime64[D]"))
    if vt.shape[0] != len(times) and vt.shape[1] == len(times):
        vt = vt.T
    if vt.shape[0] == 1 and len(times) > 1:
        vt = np.repeat(vt, len(times), axis=0)

    Wv = {b: W[b].ravel() for b in ORDER}
    out = []
    for ti in range(arr.shape[1]):
        init = times[ti]
        iy, im = int(str(init)[:4]), int(str(init)[5:7])
        for si in range(arr.shape[2]):
            v = vt[ti, si] if si < vt.shape[1] else None
            if v is None or not np.isfinite(np.datetime64(v).astype("int64")):
                continue
            vy, vm = int(str(v)[:4]), int(str(v)[5:7])
            lead = (vy - iy) * 12 + (vm - im)
            if not 1 <= lead <= len(LEADS):
                continue
            for mi in range(arr.shape[0]):
                g = arr[mi, ti, si]
                if not np.isfinite(g).any():
                    continue
                flat = g.ravel()
                for b in ORDER:
                    out.append((str(init), lead, int(nums[mi]), b,
                                float(np.dot(np.nan_to_num(flat), Wv[b]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", action="store_true")
    ap.add_argument("--month", default=None)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.hindcast:
        for m in [f"{i:02d}" for i in range(1, 13)]:
            jobs.append((HIND_YEARS, m, CACHE / f"hind_{m}.grib"))
    now = datetime.now(timezone.utc)
    m = a.month or f"{now.month:02d}"
    jobs.append(([str(now.year)], m, CACHE / f"fcst_{now.year}{m}.grib"))
    rows = []
    for years, mth, dest in jobs:
        ok = retrieve(years, mth, dest)
        print(f"{dest.name}: {'ok' if ok else 'FAILED'} "
              f"({dest.stat().st_size/1024:.0f} KB)" if ok else f"{dest.name}: FAILED",
              flush=True)
        if ok:
            try:
                rows.extend(basin_series(dest))
            except Exception as e:                      # noqa: BLE001
                print(f"  parse failed: {repr(e)[:110]}", flush=True)
    if not rows:
        return 1
    data = {}
    for init, lead, mem, b, v in rows:
        data.setdefault(init, {}).setdefault(str(lead), {}).setdefault(b, []).append(
            round(v, 3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": f"C3S {DATASET} {CENTRE} system {SYSTEM}, total_precipitation",
        "units": "mm/day, basin-mean on energy weights",
        "inits": len(data), "data": data}, separators=(",", ":")))
    print(f"wrote {OUT}  ({len(data)} inits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
