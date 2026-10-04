#!/usr/bin/env python3
"""Build the compact DERIVED reference files the GEPS subseasonal page needs, ONCE, on the laptop, from local stores.

User rules (2026-10-04): every fetch/render/publish runs in Actions; the local hindcast and reanalysis stores are kept
forever and are never downloaded again. So Actions gets only what the live build actually reads, reduced to what it
reads, from the stores already on disk (computing locally is fine, downloading is not):

  geps8_clim_<tag>.npz        lead-dependent GEPS8 model climatology (refpack.py: quantised, delta + lzma), 9 fields,
                              leads 0.5-34.5. From climatology/geps8_clim_<tag>.nc (build/build_clim.py).
  era5_shift.nc               ERA5 1991-2020 minus 2001-2020 day-of-year normals, the re-basing term (render_maps.reref)
                              - only the DIFFERENCE is ever used, so the four 60-76 MB normals become one file. 1 deg
                              for t2m/mslp/pr/olr, 1.5 deg (_u suffix dims) for zg500 and the winds, as before.
  telecon/patterns.nc         teleconnection projectors, monthly sd, sigma_doy, the z100 day-of-year normal (the NCEP
                              z500/slp normals are dropped: only the hindcast build used them)
  telecon/validation.json, hindcast_indices.nc, mjo_hindcast.nc, regimes.nc, regime_names.json
  telecon/map_skill_<tag>.nc  hindcast skill r and forecast sd only (the fields the maps read)
  telecon/recent_<tag>.nc     2016-2025 weekly mean/terciles + weekly normal for the tercile maps
  telecon/cpc/tele_index.nh   CPC monthly teleconnection table (read for 1991-2020 only)
  strat/strat_clim_1991-2020.nc, strat/strat_trend.nc   ERA5 10/100 hPa normals for the vortex products
  strat/geps_s2s/*.json       GEPS reforecast drift tables (bias_nh, mapbias_{nh,sh}, capdrift_nh)

Packed fields (int16 + scale, zlib) are exact to half their stated step. The archive is published as the release asset
geps-ref-v1 (.github/workflows/geps.yml downloads it); bump the tag when anything here is rebuilt.

    python scripts/geps/make_reference.py --out /tmp/geps_ref          # ~30 min (lzma), 4 processes
    python scripts/geps/make_reference.py --out /tmp/geps_ref --check  # max decode error per climatology field
    python scripts/geps/make_reference.py --out /tmp/geps_ref --tar    # + geps_ref_v1.tar beside it
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                                        # noqa: E402
import refpack                                                      # noqa: E402

SRC = paths.LOCAL_STORE
import os                                                           # noqa: E402
# scripts/strat/data is gitignored: the laptop's copy lives in the main checkout
STRAT_SRC = Path(os.environ.get("STRAT_LOCAL_DATA", Path.home() / "scorvec.github.io" / "scripts" / "strat" / "data"))
TAGS = ["zg500", "t2m", "olr", "u850", "u200", "v850", "v200", "pr", "mslp"]
SFC_TAGS = ("t2m", "mslp", "pr", "olr")
# int16 resolution of the packed netCDF fields (error <= step / 2), in each field's units
SHIFT_STEP = {"t2m": 0.005, "mslp": 0.5, "pr": 5e-9, "olr": 0.01, "zg500": 0.05,            # K, Pa, kg m-2 s-1, W m-2, m
              "u850": 0.005, "u200": 0.005, "v850": 0.005, "v200": 0.005}                   # m s-1
RECENT_STEP = {"t2m": {"mean_recent": 0.002, "q33_recent": 0.002, "q67_recent": 0.002, "clim_weekly": 0.005},  # K
               "pr": {"mean_recent": 0.002, "q33_recent": 0.002, "q67_recent": 0.002, "clim_weekly": 0.002}}  # mm/day


def packed(da: xr.DataArray, step: float | None = None) -> dict:
    """int16 + scale/offset netCDF encoding; step=None takes the finest step that spans the field's range."""
    v = da.values
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    off = 0.5 * (lo + hi)
    s = step if step else max((hi - lo) / 65000.0, 1e-12)
    if (hi - off) / s > 32766:
        raise ValueError(f"{da.name}: step {s} too fine for range {lo}..{hi}")
    return {"dtype": "int16", "scale_factor": s, "add_offset": off, "_FillValue": np.int16(-32768),
            "zlib": True, "complevel": 6, "shuffle": True}


def one_clim(job):
    tag, out = job
    t0 = time.time()
    c = xr.open_dataset(SRC / "climatology" / f"geps8_clim_{tag}.nc")
    v = [k for k in c.data_vars if k != "n_starts"][0]
    da = c[v].load()
    z = refpack.encode(da.rename(v), tag)
    np.savez(out / f"geps8_clim_{tag}.npz", **z)
    return tag, (out / f"geps8_clim_{tag}.npz").stat().st_size, time.time() - t0


def check_clim(out: Path):
    refpack.paths.CLIM = out                                      # decode from the new packs
    for tag in TAGS:
        c = xr.open_dataset(SRC / "climatology" / f"geps8_clim_{tag}.nc")
        v = [k for k in c.data_vars if k != "n_starts"][0]
        o = c[v].sel(L=c.L[c.L < refpack.LMAX]).transpose("doy", "L", "Y", "X").values
        with np.load(out / f"geps8_clim_{tag}.npz") as z:
            d = refpack.decode(z).values
        e = np.abs(d.astype("float64") - o)
        print(f"  {tag:6s} max |decoded - original| {e.max():.4g}  (step {refpack.STEP[tag]:g}, rms {np.sqrt((e**2).mean()):.3g})",
              flush=True)


def era5_shift(out: Path):
    ds = {}
    enc = {}
    for kind, fmt, tags in (("sfc", "era5_clim_sfc_{}.nc", SFC_TAGS),
                            ("upper", "era5_clim_{}.nc", ("zg500", "u850", "u200", "v850", "v200"))):
        a = xr.open_dataset(SRC / "climatology" / fmt.format("2001-2020"))
        b = xr.open_dataset(SRC / "climatology" / fmt.format("1991-2020"))
        for t in tags:
            d = (b[t] - a[t]).astype("float32")
            if kind == "upper":
                d = d.rename({"lat": "lat_u", "lon": "lon_u"})
            d.name = t
            ds[t] = d
            enc[t] = packed(d, SHIFT_STEP[t])
    x = xr.Dataset(ds)
    x.attrs = {"title": "ERA5 day-of-year normal 1991-2020 minus 2001-2020 (the GEPS anomaly re-basing term)",
               "source": "derived from climatology/era5_clim{,_sfc}_{1991-2020,2001-2020}.nc (ERA5, laptop store)",
               "grids": "t2m/mslp/pr/olr on lat/lon (1 deg); zg500/u850/u200/v850/v200 on lat_u/lon_u (1.5 deg)",
               "units": "t2m K, mslp Pa, pr kg m-2 s-1, olr W m-2, zg500 m, winds m s-1"}
    x.to_netcdf(out / "era5_shift.nc", encoding=enc)


def telecon(out: Path):
    T = SRC / "telecon"
    o = out / "telecon"
    (o / "cpc").mkdir(parents=True, exist_ok=True)
    p = xr.open_dataset(T / "patterns.nc")
    p = p.drop_vars([v for v in ("clim_doy_z500", "clim_doy_slp") if v in p])
    p.to_netcdf(o / "patterns.nc", encoding={v: {"zlib": True, "complevel": 6, "shuffle": True}
                                              for v in p.data_vars if p[v].dtype.kind == "f"})
    for f in ("validation.json", "regimes.nc", "regime_names.json", "mjo_hindcast.nc"):
        shutil.copy2(T / f, o / f)
    shutil.copy2(T / "cpc" / "tele_index.nh", o / "cpc" / "tele_index.nh")
    h = xr.open_dataset(T / "hindcast_indices.nc")
    h.to_netcdf(o / "hindcast_indices.nc", encoding={v: {"zlib": True, "complevel": 6, "shuffle": True}
                                                     for v in ("fcst", "obs")})
    for tag in ("t2m", "pr", "zg500", "mslp"):
        m = xr.open_dataset(T / f"map_skill_{tag}.nc")[["r", "fsd"]]
        m.attrs = xr.open_dataset(T / f"map_skill_{tag}.nc").attrs
        m.to_netcdf(o / f"map_skill_{tag}.nc", encoding={v: {"zlib": True, "complevel": 6, "shuffle": True}
                                                         for v in m.data_vars})
    for tag in ("t2m", "pr"):
        r = xr.open_dataset(T / f"recent_median_{tag}.nc")
        keep = ["mean_recent", "q33_recent", "q67_recent", "clim_weekly"]
        x = r[keep].load()
        x.attrs = r.attrs
        x.to_netcdf(o / f"recent_median_{tag}.nc", encoding={v: packed(x[v], RECENT_STEP[tag][v]) for v in keep})


def strat(out: Path):
    o = out / "strat"
    (o / "geps_s2s").mkdir(parents=True, exist_ok=True)
    c = xr.open_dataset(STRAT_SRC / "strat_clim_1991-2020.nc").load()
    # T K, U m/s, Z m
    c.to_netcdf(o / "strat_clim_1991-2020.nc", encoding={v: packed(c[v], {"T": 0.005, "U": 0.005, "Z": 0.5}[v])
                                                         for v in c.data_vars})
    t = xr.open_dataset(STRAT_SRC / "strat_trend.nc").load()
    t.to_netcdf(o / "strat_trend.nc", encoding={v: {"zlib": True, "complevel": 6, "shuffle": True} for v in t.data_vars})
    for f in ("bias_nh.json", "mapbias_nh.json", "mapbias_sh.json", "capdrift_nh.json"):
        shutil.copy2(STRAT_SRC / "geps_s2s" / f, o / "geps_s2s" / f)


def state_seed(dest: Path, cycle: str) -> None:
    """The first Actions run's STATE (release asset geps_state_seed.tar), from what the laptop pipeline already holds:
    the GEPS day-0 analysis caches (the observed tails - the Datamart keeps only ~25 days, so these cannot be fetched
    again), the CPC/PSL/Long Paddock files, the vortex series and the compact archive of the last cycle built here (the
    first run's change maps and previous-run overlays). Nothing is downloaded."""
    import datetime as dt
    T = SRC / "telecon"
    (dest / "telecon" / "analysis").mkdir(parents=True, exist_ok=True)
    cut = (dt.date.today() - dt.timedelta(days=70)).strftime("%Y%m%d")
    n = 0
    for q in sorted((T / "analysis").glob("*.npz")):
        if q.stem.rsplit("_", 1)[-1] >= cut:
            shutil.copy2(q, dest / "telecon" / "analysis" / q.name); n += 1
    for sub, names in (("cpc", ("ao.daily.csv", "nao.daily.csv", "pna.daily.csv", "aao.daily.csv", "soi.txt",
                                "reqsoi.for")),
                       ("psl", ("epo.daily.txt", "wpo.daily.txt")),
                       ("longpaddock", ("DailySOI1887-1989Base.txt",))):
        (dest / "telecon" / sub).mkdir(parents=True, exist_ok=True)
        for f in names:
            shutil.copy2(T / sub / f, dest / "telecon" / sub / f)
    ser = STRAT_SRC / "series"
    (dest / "strat_series").mkdir(parents=True, exist_ok=True)
    for q in sorted(ser.glob(f"strat_*_{cycle}00.json")):
        shutil.copy2(q, dest / "strat_series" / q.name)
    import archive
    paths.LIVE, paths.ARCHIVE = SRC / "live", dest / "runs"
    archive.paths.LIVE, archive.paths.ARCHIVE = paths.LIVE, paths.ARCHIVE
    names = archive.write(cycle)
    (dest / "last_cycle.json").write_text(json.dumps({"cycle": cycle, "built": "laptop seed"}))
    print(f"  state seed: {n} analysis npz, {len(names)} archived files of {cycle}, CPC/PSL/Long Paddock, vortex series")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-seed", metavar="CYCLE", help="also build the first run's state from the laptop's files "
                    "(the newest cycle built here, e.g. 20261001) -> <out>/../geps_state_seed.tar")
    ap.add_argument("--out", default=str(paths.REF))
    ap.add_argument("--only", default="clim,shift,telecon,strat")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    if "clim" in only and not a.check:
        with ProcessPoolExecutor(a.procs) as ex:
            for tag, size, dt in ex.map(one_clim, [(t, out) for t in TAGS]):
                print(f"  geps8_clim_{tag}.npz  {size / 1e6:.1f} MB  ({dt:.0f} s)", flush=True)
    if a.check:
        check_clim(out)
    if "shift" in only and not a.check:
        era5_shift(out); print("  era5_shift.nc", flush=True)
    if "telecon" in only and not a.check:
        telecon(out); print("  telecon/", flush=True)
    if "strat" in only and not a.check:
        strat(out); print("  strat/", flush=True)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    tot = sum(p.stat().st_size for p in files)
    man = {str(p.relative_to(out)): p.stat().st_size for p in files}
    (out / "MANIFEST.json").write_text(json.dumps({"total_bytes": tot, "files": man}, indent=1))
    print(f"{len(files)} files, {tot / 1e6:.1f} MB in {out}")
    if a.state_seed:
        sd = out.parent / "geps_state_seed"
        shutil.rmtree(sd, ignore_errors=True)
        state_seed(sd, a.state_seed)
        tp = out.parent / "geps_state_seed.tar"
        with tarfile.open(tp, "w") as tf:
            for p in sorted(q for q in sd.rglob("*") if q.is_file()):
                tf.add(p, arcname=str(p.relative_to(sd)))
        print(f"  {tp}  {tp.stat().st_size / 1e6:.1f} MB")
    if a.tar:
        tp = out.parent / "geps_ref_v1.tar"
        with tarfile.open(tp, "w") as tf:
            for p in files + [out / "MANIFEST.json"]:
                tf.add(p, arcname=str(p.relative_to(out)))
        print(f"  {tp}  {tp.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
