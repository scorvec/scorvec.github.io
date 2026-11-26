#!/usr/bin/env python3
"""Every single run's upper air on the verification grid, and ERA5's, for the pressure-level error maps of
verification.html (scripts/verify/upper_maps.py). Files of the verification database (scripts/verify/verifydb.sh),
in releases of their own, verify-upper-YYYY-MM:

  upa-<m>_<init>.npz   GFS, the IFS control, AIFS and GDPS at 00Z: height (z, m), temperature (t, K), wind (u, v,
                       m/s) on LEVELS and specific humidity (q, kg/kg) on Q_LEVELS, at LEADS (lead 0: the model's own
                       analysis), each as (lead, level, lat, lon) on the WMO 1.5 deg grid (latlon.G15, conservative)
  era5-upa_<day>.npz   ERA5 at 00Z of the day from ARCO, the same fields as (level, lat, lon)

Each field is kept as 16-bit integers at a fixed step (PACK; ``pack``/``unpack``), the missing value the type's
extreme. The units are read from the magnitudes and checked against the standard atmosphere (``to_units``): ERA5
holds geopotential, ECCC files have carried decametres.

    upper_fields.py plan --first YYYYMMDD --last YYYYMMDD --have have.txt [--chunk 8]   # the runs to fetch, as JSON
    upper_fields.py fields --model GFS --init 2026112500 --have have.txt --archive-out db-new
    upper_fields.py truth --first YYYYMMDD --last YYYYMMDD --have have.txt --archive-out db-new
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ens_verify as E
import mverify_scores as D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from latlon import G15, remap

G = 9.80665
MODELS = {"GFS": "gfs", "IFS": "ifs", "AIFS": "aifs", "GDPS": "gdps"}  # the single runs, as z500_acc.DET
LEVELS = (925, 850, 700, 500, 300, 250, 200, 100, 50)  # hPa, every model's open data and ERA5 alike
Q_LEVELS = (925, 850, 700, 500, 300)  # humidity above 300 hPa is too small to map
LEADS = (0, 24, 72, 120, 168, 240)  # hours: the analysis, then days 1, 3, 5, 7 and 10
VARS = ("z", "t", "u", "v", "q")
# the standard atmosphere's height of each level (m): a field's unit is read against it
Z_STD = {925: 762, 850: 1457, 700: 3012, 500: 5574, 300: 9164, 250: 10363, 200: 11784, 100: 16180, 50: 20576}
# variable: (step, offset, type) of its 16-bit integers; the type's extreme marks a missing value. Each file stores the
# offsets it was packed with (off_<var>); files without them were packed with LEGACY_OFF (z from 0 m, which could not
# hold a 925 hPa height below sea level under a deep low: 2026-11-11 00Z IFS)
PACK = {
    "z": (0.5, -1000.0, np.uint16),
    "t": (0.01, 200.0, np.int16),
    "u": (0.01, 0.0, np.int16),
    "v": (0.01, 0.0, np.int16),
    "q": (1e-6, 0.0, np.uint16),
}
LEGACY_OFF = {"z": 0.0, "t": 200.0, "u": 0.0, "v": 0.0, "q": 0.0}
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_VARS = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}
GFS_NAMES = {"z": "HGT", "t": "TMP", "u": "UGRD", "v": "VGRD", "q": "SPFH"}
GDPS_NAMES = {"z": "GeopotentialHeight", "t": "AirTemp", "u": "WindU", "v": "WindV", "q": "SpecificHumidity"}
SHORT = {"gh": "z", "z": "z", "t": "t", "u": "u", "v": "v", "q": "q"}  # GRIB shortName: variable
DAYS = 47  # the daily catch-up's oldest run (HPFX keeps a GDPS run about 52 days)


def levels_of(var: str) -> tuple[int, ...]:
    return Q_LEVELS if var == "q" else LEVELS


def pack(x: np.ndarray, var: str) -> np.ndarray:
    step, off, typ = PACK[var]
    info = np.iinfo(typ)
    signed = info.min < 0
    miss = info.min if signed else info.max
    q = np.round((x - off) / step)
    ok = np.isfinite(q)
    lo, hi = (info.min + 1, info.max) if signed else (0, info.max - 1)  # the missing value is never a datum
    if (q[ok] < lo).any() or (q[ok] > hi).any():
        raise ValueError(f"{var}: values outside what {typ.__name__} holds at step {step}")
    return np.where(ok, q, miss).astype(typ)


def offsets(z) -> dict[str, float]:
    """The offsets an opened npz file was packed with."""
    return {v: float(z[f"off_{v}"]) if f"off_{v}" in z.files else LEGACY_OFF[v] for v in VARS}


def unpack(a: np.ndarray, var: str, off: float) -> np.ndarray:
    step, _, typ = PACK[var]
    info = np.iinfo(typ)
    miss = info.min if info.min < 0 else info.max
    return np.where(a == miss, np.nan, a.astype(np.float32) * step + off)


def to_units(var: str, level: int, v: np.ndarray, what: str) -> np.ndarray:
    """``v`` in the variable's unit (z m, t K, u/v m/s, q kg/kg), read from its magnitude; RuntimeError when it is not
    the field it should be."""
    med = float(np.nanmedian(v[::5, ::5]))
    if var == "z":
        v = v / G if med > 3 * Z_STD[level] else v * 10.0 if med < Z_STD[level] / 3 else v
        ok = abs(np.nanmedian(v) / Z_STD[level] - 1) < 0.15
    elif var == "t":
        v = v + 273.15 if med < 100 else v
        ok = 170 < np.nanmedian(v) < 320
    elif var == "q":
        v = v / 1000.0 if med > 0.1 else v
        # the models' numerics leave specks of slightly negative humidity (ERA5 500 hPa, 2026-10-08): set to zero
        ok = np.nanmin(v) > -1e-4 and np.nanmax(v) < 0.05
        v = np.clip(v, 0, None)
    else:
        ok = abs(med) < 60 and np.nanmax(np.abs(v)) < 200
    if not ok:
        raise RuntimeError(f"{what} {var}{level}: median {np.nanmedian(v):.4g}, not that field")
    return v


def decode(blob: bytes) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]:
    """(variable, level, latitudes, longitudes, values) of every GRIB message of a download."""
    import eccodes

    out, i = [], 0
    while (i := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[i + 8 : i + 16], "big")
        gid = eccodes.codes_new_from_message(blob[i : i + n])
        i += n
        try:
            short, level = eccodes.codes_get(gid, "shortName"), int(eccodes.codes_get(gid, "level"))
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            lat = np.linspace(
                eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees"),
                eccodes.codes_get_double(gid, "latitudeOfLastGridPointInDegrees"),
                nj,
            )
            lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
            lon = lon0 + eccodes.codes_get_double(gid, "iDirectionIncrementInDegrees") * np.arange(ni)
            v = eccodes.codes_get_values(gid).reshape(nj, ni)
        finally:
            eccodes.codes_release(gid)
        if short not in SHORT:
            raise RuntimeError(f"unexpected GRIB field {short} at {level} hPa")
        out.append((SHORT[short], level, lat, lon, v))
    return out


def wanted() -> list[tuple[str, int]]:
    return [(var, p) for var in VARS for p in levels_of(var)]


def step_blobs(model: str, init: str, lead: int) -> list[bytes]:
    """The downloads holding one step's fields: one file's byte ranges (GFS, the ECMWF models), or a file a field
    (GDPS)."""
    day, hh = init[:8], init[8:]
    if model == "GFS":
        url = f"{D.GFS}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"
        return [E.gefs_member(url, [f"{GFS_NAMES[v]}:{p} mb" for v, p in wanted()])]
    if model == "GDPS":
        base = f"{D.DATAMART}/{day}/WXO-DD/model_gdps/15km/{hh}/{lead:03d}/{day}T{hh}Z_MSC_GDPS"
        urls = [f"{base}_{GDPS_NAMES[v]}_IsbL-{p:04d}_LatLon0.15_PT{lead:03d}H.grib2" for v, p in wanted()]
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(D.get, urls))
    url = f"{D.GOOGLE}/{day}/{hh}z/{D.MODELS[model]}/0p25/oper/{day}{hh}0000-{lead}h-oper-fc"
    return [E.ecmwf_bytes(url, {("gh" if v == "z" else v, "pl", str(p)) for v, p in wanted()}, ("fc",))]


def file_of(model: str, init: str) -> str:
    return f"upa-{MODELS[model]}_{init}.npz"


def truth_file(day: pd.Timestamp) -> str:
    return f"era5-upa_{day:%Y%m%d}00.npz"


def run_fields(model: str, init: str, out: Path) -> Path:
    """One run's fields on the verification grid, as upa-<m>_<init>.npz in ``out``."""
    arr = {v: np.full((len(LEADS), len(levels_of(v)), *G15.shape), np.nan, np.float32) for v in VARS}
    for k, lead in enumerate(LEADS):
        got = set()
        for blob in step_blobs(model, init, lead):
            for var, level, lat, lon, raw in decode(blob):
                if level not in levels_of(var) or (var, level) in got:
                    continue
                v = to_units(var, level, raw, f"{model} {init} +{lead} h")
                arr[var][k, levels_of(var).index(level)] = remap(v, lat, lon, G15)
                got.add((var, level))
        if missing := set(wanted()) - got:
            raise RuntimeError(f"{model} {init} +{lead} h: no {sorted(missing)[:4]}")
        print(f"{model} {init} +{lead} h: {len(got)} fields", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    path = out / file_of(model, init)
    np.savez_compressed(
        path,
        lead_h=np.array(LEADS),
        levels=np.array(LEVELS),
        q_levels=np.array(Q_LEVELS),
        **{v: pack(arr[v], v) for v in VARS},
        **{f"off_{v}": np.float64(PACK[v][1]) for v in VARS},
    )
    return path


def load(path: Path) -> dict[str, np.ndarray]:
    """A upa- or era5-upa file's fields in their units (float32, NaN where missing)."""
    with np.load(path) as z:
        off = offsets(z)
        return {v: unpack(z[v], v, off[v]) for v in VARS}


def fields(a) -> None:
    have = _have(a.have)
    if file_of(a.model, a.init) in have:
        print(f"{a.model} {a.init}: held", flush=True)
        return
    run_fields(a.model, a.init, a.archive_out)


def truth(a) -> None:
    """ERA5 at 00Z of each day of the range the database lacks, once ARCO holds it."""
    import xarray as xr

    have = _have(a.have)
    a.archive_out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    stop = pd.Timestamp(ds.attrs["valid_time_stop_era5t"])
    print(f"ARCO holds ERA5 to {stop:%Y-%m-%d %H}Z", flush=True)
    for day in pd.date_range(pd.Timestamp(a.first), min(pd.Timestamp(a.last), stop), freq="D"):
        if truth_file(day) in have:
            continue
        out, ok = {}, True
        for var, name in ARCO_VARS.items():
            x = ds[name].sel(time=day, level=list(levels_of(var))).transpose("level", "latitude", "longitude")
            vals = x.values
            if not np.isfinite(vals).all():
                print(f"::warning::ERA5 {var} {day:%Y-%m-%d} 00Z: incomplete in ARCO, for the next pass", flush=True)
                ok = False
                break
            lat, lon = x["latitude"].values, x["longitude"].values
            out[var] = pack(
                np.stack(
                    [
                        remap(to_units(var, p, vals[i], f"ERA5 {day:%Y-%m-%d}"), lat, lon, G15)
                        for i, p in enumerate(levels_of(var))
                    ]
                ),
                var,
            )
        if ok:
            np.savez_compressed(
                a.archive_out / truth_file(day),
                levels=np.array(LEVELS),
                q_levels=np.array(Q_LEVELS),
                **out,
                **{f"off_{v}": np.float64(PACK[v][1]) for v in VARS},
            )
            print(f"ERA5 {day:%Y-%m-%d} 00Z: archived", flush=True)


def plan(a) -> None:
    """The runs each model lacks in the range, newest first (GDPS first of all: HPFX deletes its runs), as JSON chunks
    of at most ``chunk`` runs, and the range of days ERA5 is wanted for (each run's day 1 to its day 10)."""
    have = _have(a.have)
    first, last = (pd.Timestamp(x) for x in (a.first, a.last))
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    chunks = []
    for m in sorted(MODELS, key=lambda m: m != "GDPS"):
        oldest = max(first, today - pd.Timedelta(days=D.HPFX_DAYS)) if m == "GDPS" else first
        todo = [
            f"{d:%Y%m%d}00"
            for d in pd.date_range(oldest, last, freq="D")[::-1]
            if file_of(m, f"{d:%Y%m%d}00") not in have
        ]
        chunks += [{"model": m, "runs": " ".join(todo[i : i + a.chunk])} for i in range(0, len(todo), a.chunk)]
        print(f"{m}: {len(todo)} runs to add", file=sys.stderr)
    print("chunks=" + json.dumps(chunks))
    print(f"first={first + pd.Timedelta(days=1):%Y%m%d}")
    print(f"last={min(last + pd.Timedelta(days=10), today):%Y%m%d}")


def _have(path: Path | None) -> set[str]:
    return set(path.read_text().split()) if path and path.exists() else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the runs to fetch, as the workflow's matrix")
    p.add_argument("--first", default=f"{pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=DAYS):%Y%m%d}")
    p.add_argument("--last", default=f"{pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1):%Y%m%d}")
    p.add_argument("--chunk", type=int, default=8, help="runs per job")
    f = sub.add_parser("fields", help="one run of one model")
    f.add_argument("--model", required=True, choices=list(MODELS))
    f.add_argument("--init", required=True, help="00Z run, YYYYMMDDHH")
    t = sub.add_parser("truth", help="ERA5 at 00Z of each day ARCO holds")
    t.add_argument("--first", required=True, help="YYYYMMDD")
    t.add_argument("--last", required=True, help="YYYYMMDD")
    for s in (p, f, t):
        s.add_argument("--have", type=Path, help="names of the files the database holds, one per line")
    for s in (f, t):
        s.add_argument("--archive-out", type=Path, default=Path("db-new"), help="where the files to add are written")
    a = ap.parse_args()
    {"plan": plan, "fields": fields, "truth": truth}[a.cmd](a)


if __name__ == "__main__":
    main()
