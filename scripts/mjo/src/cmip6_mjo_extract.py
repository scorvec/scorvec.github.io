#!/usr/bin/env python3
"""CMIP6 daily fields for the MJO impact composites (2026-09-27; user: "cmip6 mjo composites for the different phases
by month"). LAPTOP job: streams the public Pangeo CMIP6 zarr stores (anonymous GCS), keeps only small reductions.

Per (model, member, variable), 1979-2014 of the historical run, regridded bilinearly (cyclic in longitude) to:
  band   15S-15N x 144 at 2.5 deg      rlut, pr, u850, u250  (the W&H RMM grid; spectra and propagation)
  na     15-75N, 190-310E at 2 deg     tas, pr               (North America)
  sa     57S-15N, 270-330E at 2 deg    tas, pr               (South America)
  nh     0-90N x 144 at 2.5 deg        z500                  (flat NH map)
stored as float16 in ~/data_archive/cmip6_mjo/red/<model>_<member>_<var>.npz with the model's own dates (cftime
calendars kept: 360-day and noleap models carry their own month/day). Units: pr mm/day, tas K, rlut W m-2, ua m/s, zg m.

The daily `day` table carries ua and zg on plev8 (1000, 850, 700, 500, 250, 100, 50, 10 hPa) in chunks that hold all 8
levels, so one level costs the full 3-D read (~11-19 GB a member for 1979-2014); 250 hPa stands in for RMM's 200 hPa.

    python cmip6_mjo_extract.py screen            # one member per candidate model: rlut + pr
    python cmip6_mjo_extract.py full M1 M2 ...    # screened models: tas/ua/zg for member 1, rlut/pr/tas/ua for member 2
    python cmip6_mjo_extract.py list
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home() / "data_archive" / "cmip6_mjo" / "red"
CAT = Path(__file__).resolve().parents[2] / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"
Y0, Y1 = 1979, 2014
GRIDS = {
    "band": (np.arange(-15.0, 15.01, 2.5), np.arange(0.0, 360.0, 2.5)),
    "na": (np.arange(15.0, 75.01, 2.0), np.arange(190.0, 310.01, 2.0)),
    "sa": (np.arange(-57.0, 15.01, 2.0), np.arange(270.0, 330.01, 2.0)),
    "nh": (np.arange(0.0, 90.01, 2.5), np.arange(0.0, 360.0, 2.5)),
}
# which grids each stored variable goes to
PLAN = {"rlut": ["band"], "pr": ["band", "na", "sa"], "tas": ["na", "sa"], "ua": ["band"], "z500": ["nh"]}
# job variable -> (store variable, {output name: pressure level or None}); ua is read ONCE for both levels
SRC = {"rlut": ("rlut", {"rlut": None}), "pr": ("pr", {"pr": None}), "tas": ("tas", {"tas": None}),
       "ua": ("ua", {"u850": 85000.0, "u250": 25000.0}), "z500": ("zg", {"z500": 50000.0})}


def catalog():
    c = pd.read_csv(CAT)
    return c[(c.table_id == "day") & (c.experiment_id == "historical")]


def candidates():
    d = catalog()
    piv = d[d.variable_id.isin(["rlut", "ua", "tas", "pr", "zg"])].groupby(["source_id", "variable_id"]).member_id.nunique().unstack(fill_value=0)
    ok = piv[(piv.get("rlut", 0) > 0) & (piv.get("ua", 0) > 0) & (piv.get("tas", 0) > 0) & (piv.get("pr", 0) > 0)]
    return ok


def members(model, need):
    """Members that have every variable in `need`, r1 first."""
    d = catalog()
    s = d[(d.source_id == model) & d.variable_id.isin(need)]
    have = s.groupby("member_id").variable_id.nunique()
    m = sorted(have[have == len(set(need))].index, key=lambda x: (not x.startswith("r1i1p1"), int(x[1:x.index("i")]), x))
    return m


def zstore(model, member, var):
    d = catalog()
    s = d[(d.source_id == model) & (d.member_id == member) & (d.variable_id == var)]
    if not len(s):
        return None
    return s.sort_values("version").zstore.iloc[-1]


def _interp(block, lat, lon, glat, glon):
    """(t, lat, lon) -> (t, glat, glon), bilinear, cyclic in longitude, latitude clamped at the edges."""
    order = np.argsort(lat)
    lat = lat[order]; block = block[:, order, :]
    lonc = np.concatenate([lon[-1:] - 360.0, lon, lon[:1] + 360.0])
    blk = np.concatenate([block[:, :, -1:], block, block[:, :, :1]], axis=2)
    ix = np.interp(glon % 360.0, lonc, np.arange(len(lonc)))
    iy = np.interp(glat, lat, np.arange(len(lat)))
    x0 = np.floor(ix).astype(int); x1 = np.minimum(x0 + 1, len(lonc) - 1); fx = (ix - x0)[None, None, :]
    y0 = np.floor(iy).astype(int); y1 = np.minimum(y0 + 1, len(lat) - 1); fy = (iy - y0)[None, :, None]
    a = blk[:, y0][:, :, x0]; b = blk[:, y0][:, :, x1]; c = blk[:, y1][:, :, x0]; d = blk[:, y1][:, :, x1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy).astype(np.float32)


def extract(model, member, var):
    """Stream one (model, member, variable) and write its reductions. Resumable: a finished file is skipped."""
    import xarray as xr
    dest = OUT / f"{model}_{member}_{var}.npz"
    if dest.exists():
        return f"{dest.name} exists"
    src, outv = SRC[var]
    z = zstore(model, member, src)
    if z is None:
        return f"{model} {member} {var}: no store"
    t0 = time.time()
    ds = xr.open_zarr(z, storage_options={"token": "anon"}, consolidated=True, use_cftime=True)
    da = ds[src]
    latn = "lat" if "lat" in da.dims else "latitude"; lonn = "lon" if "lon" in da.dims else "longitude"
    kidx = {}
    if "plev" in da.dims:
        p = da["plev"].values.astype(float)
        if p.max() < 2000:                                   # some stores label hPa
            p = p * 100.0
        for name, lev in outv.items():
            k = int(np.argmin(np.abs(p - lev)))
            if abs(p[k] - lev) > 1.0:
                return f"{model} {member} {var}: level {lev} missing (have {p})"
            kidx[name] = k
        da = da.isel(plev=sorted(set(kidx.values())))
        kpos = {name: sorted(set(kidx.values())).index(k) for name, k in kidx.items()}
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= Y0) & (yrs <= Y1)
    ii = np.flatnonzero(sel)
    da = da.isel(time=slice(int(ii[0]), int(ii[-1]) + 1))          # a slice keeps the zarr chunk layout
    lat = da[latn].values.astype(float); lon = da[lonn].values.astype(float) % 360.0
    so = np.argsort(lon); lon = lon[so]
    times = da.time.values
    dates = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in times])
    scale = 86400.0 if var == "pr" else 1.0
    outs = {(n, g): [] for n in outv for g in PLAN[var]}
    nt = len(times)
    # Blocks ALIGNED to the zarr time chunks (a 180-day block over a 1,135-day chunk re-read the chunk six times),
    # several chunks per block so dask fetches them concurrently, capped near 1.5 GB decoded.
    csz = da.chunks[0] if da.chunks else (nt,)
    per_day = np.prod([s for d_, s in zip(da.dims, da.shape) if d_ != "time"]) * 4
    cap = max(1, int(1.5e9 // per_day))
    bounds, cur, start = [], 0, 0
    for c in csz:
        if cur and cur + c > cap:
            bounds.append((start, start + cur)); start += cur; cur = 0
        cur += c
    if cur:
        bounds.append((start, start + cur))
    nbytes = 0
    for i0, i1 in bounds:
        raw = da.isel(time=slice(i0, i1)).values.astype(np.float32)
        nbytes += raw.nbytes
        for n in outv:
            blk = (raw[:, kpos[n]] if kidx else raw)[..., so] * scale
            for g in PLAN[var]:
                glat, glon = GRIDS[g]
                outs[(n, g)].append(_interp(blk, lat, lon, glat, glon).astype(np.float16))
    arr = {f"{n}_{g}": np.concatenate(v, axis=0) for (n, g), v in outs.items()}
    cal = str(getattr(times[0], "calendar", "standard"))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dates=dates, calendar=cal, **arr, **{f"lat_{g}": GRIDS[g][0] for g in PLAN[var]},
                        **{f"lon_{g}": GRIDS[g][1] for g in PLAN[var]})
    tmp.rename(dest)
    dt = time.time() - t0
    return f"{model} {member} {var}: {nt} days, {nbytes / 1e9:.1f} GB decoded in {dt:.0f} s ({cal})"


def run(jobs, workers=2):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import dask
    dask.config.set(scheduler="threads", num_workers=6)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_safe, *j): j for j in jobs}
        for f in futs:
            print(f.result(), flush=True)


def _safe(model, member, var):
    import dask
    dask.config.set(scheduler="threads", num_workers=8)
    for attempt in range(3):
        try:
            return extract(model, member, var)
        except Exception as e:                                # noqa: BLE001
            err = f"{model} {member} {var}: FAILED ({type(e).__name__}: {str(e)[:160]})"
            time.sleep(20 * (attempt + 1))
    return err


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    ok = candidates()
    if cmd == "list":
        print(ok.to_string()); return 0
    if cmd == "screen":
        jobs = []
        for m in ok.index:
            mem = members(m, ["rlut", "pr", "ua", "tas"])
            if not mem:
                print(m, "no member with rlut/pr/ua/tas"); continue
            jobs += [(m, mem[0], "rlut"), (m, mem[0], "pr")]
        run(jobs, workers=int(os.environ.get("W", "2")))
        return 0
    if cmd == "full":
        jobs = []
        for m in sys.argv[2:]:
            mem = members(m, ["rlut", "pr", "ua", "tas"])
            jobs += [(m, mem[0], "tas"), (m, mem[0], "ua")]
            if zstore(m, mem[0], "zg"):
                jobs.append((m, mem[0], "z500"))
            if len(mem) > 1:
                jobs += [(m, mem[1], v) for v in ("rlut", "pr", "tas", "ua")]
        run(jobs, workers=int(os.environ.get("W", "2")))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())


def n34(model, member):
    """Monthly Nino-3.4 box mean (5S-5N, 170-120W) of the run's own Amon ts, 1979-2014 -> <model>_<member>_n34.npz.
    Used to remove the linear ENSO signal from the run's RMM, as BoM/W&H do with SST1 (2026-09-27)."""
    import xarray as xr
    dest = OUT / f"{model}_{member}_n34.npz"
    if dest.exists():
        return f"{dest.name} exists"
    c = pd.read_csv(CAT)
    s = c[(c.table_id == "Amon") & (c.experiment_id == "historical") & (c.variable_id == "ts") & (c.source_id == model)
          & (c.member_id == member)]
    if not len(s):
        return f"{model} {member}: no Amon ts"
    ds = xr.open_zarr(s.sort_values("version").zstore.iloc[-1], storage_options={"token": "anon"}, consolidated=True,
                      use_cftime=True)
    da = ds["ts"]
    latn = "lat" if "lat" in da.dims else "latitude"; lonn = "lon" if "lon" in da.dims else "longitude"
    lat = da[latn].values; lon = da[lonn].values % 360
    box = da.isel({latn: np.flatnonzero((lat >= -5) & (lat <= 5)), lonn: np.flatnonzero((lon >= 190) & (lon <= 240))})
    w = np.cos(np.deg2rad(box[latn].values))
    v = box.values                                                    # (t, lat, lon)
    ts_ = (v * w[None, :, None]).sum(1).mean(1) / w.sum()
    months = np.array([f"{t.year:04d}-{t.month:02d}" for t in da.time.values])
    keep = (months >= f"{Y0}-01") & (months <= f"{Y1}-12")
    np.savez_compressed(dest, months=months[keep], n34=ts_[keep].astype(np.float32))
    return f"{model} {member} n34: {keep.sum()} months"
