#!/usr/bin/env python3
"""Per-ensemble fetchers → ensemble MEDIAN of z500 / t2m on the common North-America
grid, for each forecast lead. Returns an array (lead, lat, lon) in clim units
(z500 in dam, t2m in K). Phase 1 implements GEFS; IFS/AIFS/GEPS land in later phases.

    python src/fetch.py --ensemble gefs --var z500 --date 20260603 --run 00   # smoke test
"""
from __future__ import annotations
import argparse, os, sys, tempfile, time, warnings
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
            med[i] = np.nanmean(np.stack(stacks[ld]), axis=0)    # ens MEAN (smoother than median)
        print(f"  GEFS {var} f{ld:03d}: {len(stacks[ld])} members (mean)", flush=True)
    return med


def _open_grib(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _members_da(path: str) -> xr.DataArray:
    """Open an ensemble GRIB that mixes control (cf) + perturbed (pf) members (which
    cfgrib can't merge in one Dataset) and return the DataArray that carries the
    'number' member dimension."""
    import cfgrib
    dss = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    cand = [ds for ds in dss if "number" in ds.dims] or dss
    ds = cand[0]
    return ds[list(ds.data_vars)[0]]


def _ecmwf_retrieve(client, retries: int = 5, **kw) -> None:
    """retrieve with backoff on S3 503/SlowDown (the ECMWF open-data rate limit)."""
    for k in range(retries):
        try:
            client.retrieve(**kw); return
        except Exception as e:                                # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"    ecmwf retry {k+1}/{retries} ({repr(e)[:50]})", flush=True)
            time.sleep(6 * (k + 1))


def _ecmwf_kw(date: str, run: str, v: dict, step) -> dict:
    kw = dict(date=f"{date[:4]}-{date[4:6]}-{date[6:8]}", time=int(run),
              stream="enfo", param=v["ecmwf_param"], step=step)
    if v["ecmwf_levtype"] == "pl":
        kw["levelist"] = [str(x) for x in v["ecmwf_levelist"]]
    return kw


# ─────────────────── IFS-ENS (ECMWF open data, ensemble MEAN) ───────────────────
def fetch_ifs(date: str, run: str, var: str) -> np.ndarray:
    """IFS-ENS via the native ensemble-mean product (type=em): one small field per
    step, all steps in a single retrieve — sidesteps the 51-member S3 rate limit."""
    from ecmwf.opendata import Client
    v = VARS[var]; lat, lon = grid(var)
    c = Client(source="aws", model="ifs")
    out = np.full((len(LEADS), len(lat), len(lon)), np.nan, "float32")
    tgt = tempfile.mktemp(suffix=".grib2")
    try:
        _ecmwf_retrieve(c, **_ecmwf_kw(date, run, v, list(LEADS)), type="em", target=tgt)
        da = _open_grib(tgt); da = da[list(da.data_vars)[0]]
        steps_h = [int(s / np.timedelta64(1, "h")) for s in np.atleast_1d(da.step.values)]
        for i, ld in enumerate(LEADS):
            if ld in steps_h:
                fld = da.isel(step=steps_h.index(ld)) if "step" in da.dims else da
                out[i] = _to_common(fld, v["model_scale"], lat, lon)
            print(f"  IFS-ENS {var} f{ld:03d}: {'em' if ld in steps_h else 'MISSING'}", flush=True)
    finally:
        if os.path.exists(tgt):
            os.remove(tgt)
    return out


# ─────────────────── AIFS-ENS (ECMWF open data, ensemble MEAN) ──────────────────
def fetch_aifs(date: str, run: str, var: str) -> np.ndarray:
    """AIFS-ENS has no ensemble-mean product and its per-step perturbed file is huge, so
    byte-ranging the 50 members gets S3-throttled. We reuse the mjo pipeline's hardened
    AIFS downloader (parallel member streams + throughput watchdog + mirror-rotating
    retry + the GRIB completeness check) to pull just z@500 / 2t for all members, then
    take the mean. AIFS keys 500 mb as 'z' (geopotential, m²/s²) → its own param/scale."""
    v = VARS[var]; lat, lon = grid(var)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mjo" / "src"))
    from download_aifs import retrieve_parallel
    req = dict(model="aifs-ens", date=f"{date[:4]}-{date[4:6]}-{date[6:8]}", time=int(run),
               stream="enfo", levtype=v["ecmwf_levtype"], param=v["aifs_param"],
               step=list(LEADS), type="pf")
    if v["ecmwf_levtype"] == "pl":
        req["levelist"] = [int(x) for x in v["ecmwf_levelist"]]
    out = np.full((len(LEADS), len(lat), len(lon)), np.nan, "float32")
    tgt = tempfile.mktemp(suffix=".grib2")
    try:
        retrieve_parallel(req, tgt)                          # robust parallel member pull
        da = _members_da(tgt)                                # (number, step, lat, lon)
        nm = da.sizes.get("number", 1)
        mean = da.mean(dim="number") if "number" in da.dims else da   # ens MEAN
        steps_h = [int(s / np.timedelta64(1, "h")) for s in np.atleast_1d(mean.step.values)]
        for i, ld in enumerate(LEADS):
            if ld in steps_h:
                fld = mean.isel(step=steps_h.index(ld)) if "step" in mean.dims else mean
                out[i] = _to_common(fld, v["aifs_scale"], lat, lon)
            print(f"  AIFS-ENS {var} f{ld:03d}: {nm if ld in steps_h else 'MISSING'} members (mean)", flush=True)
    finally:
        if os.path.exists(tgt):
            os.remove(tgt)
    return out


# ─────────────────────── GEPS / CMC (MSC datamart, members) ─────────────────────
_GEPS_BASE = "https://dd.weather.gc.ca"


def fetch_geps(date: str, run: str, var: str) -> np.ndarray:
    """GEPS via MSC datamart: one all-members GRIB per (var, step) → member median.
    A single download per step (no byte-range fan-out), so no rate-limit issues."""
    import requests
    v = VARS[var]; lat, lon = grid(var); param = v["geps"]; init = f"{date}{run}"
    out = np.full((len(LEADS), len(lat), len(lon)), np.nan, "float32")
    for i, ld in enumerate(LEADS):
        url = (f"{_GEPS_BASE}/{date}/WXO-DD/ensemble/geps/grib2/raw/{run}/{ld:03d}/"
               f"CMC_geps-raw_{param}_latlon0p5x0p5_{init}_P{ld:03d}_allmbrs.grib2")
        tgt = tempfile.mktemp(suffix=".grib2")
        try:
            r = requests.get(url, timeout=180); r.raise_for_status()
            with open(tgt, "wb") as f:
                f.write(r.content)
            da = _members_da(tgt)
            n = da.sizes.get("number", 1)
            fld = da.mean(dim="number") if "number" in da.dims else da   # ens MEAN
            out[i] = _to_common(fld, v["model_scale"], lat, lon)
            print(f"  GEPS {var} f{ld:03d}: {n} members (mean)", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"  GEPS {var} f{ld:03d}: FAILED ({repr(e)[:50]})", flush=True)
        finally:
            if os.path.exists(tgt):
                os.remove(tgt)
    return out


FETCHERS = {"gefs": fetch_gefs, "ifs": fetch_ifs, "aifs": fetch_aifs, "geps": fetch_geps}


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
