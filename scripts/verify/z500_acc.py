#!/usr/bin/env python3
"""500 hPa height of every model in the verification database, and its anomaly correlation against ERA5, the WMO lead
centres' headline score (verification.html draws it as NCEP EMC does, a time series for each lead); and the ensembles'
mean sea-level pressure, which the days 1-15 Arctic Oscillation is calibrated with (scripts/ensmaps/telecon15.py).

  fields  For each 00Z run, the 500 hPa height at 00Z of every day of the forecast: GFS, the IFS control, AIFS and
          GDPS to day 10 (z500-<m>_<init>.npz), and IFS-ENS, AIFS-ENS, GEFS and GEPS to day 15, as the ensemble mean
          (ens-z500-<m>_) and every member (ens-z500m-<m>_), from each model's SINCE (the ensembles' archived runs
          back to 2024 and 2025, which calibrate the teleconnection forecasts), downloaded with the database's own fetchers
          (mverify_scores, ens_verify). Every field is remapped conservatively (``remap``: the overlap of the cells'
          areas) onto the WMO 1.5 deg grid with poles (121 x 240, WeatherBench 2's), the members also onto the 2.5 deg
          grid of the teleconnection patterns north of 20N, in whole metres (int16).
          The ensembles' sea-level pressure at the same steps from MSL_SINCE, as the ensemble mean on the patterns'
          grid in hPa (ens-msl-<m>_<init>.npz, ``run_msl``): IFS-ENS's and GEFS's published means where they publish
          one (IFS-ENS: em, the control and 50 members; GEFS: geavg, the control and 30), else every member's mean.
  truth   ERA5 at 00Z of each day (ARCO, 0.25 deg) on the same grid, era5-z500_<YYYYMMDD00>.npz, once ARCO holds it
          (its ERA5T, about a week behind real time); its sea-level pressure on the patterns' grid, era5-msl_.
  clim    The anomalies' normal: ERA5 1991-2020 at 00Z, the hour of the fields, from WeatherBench 2's 6-hourly
          conservative 1.5 deg ERA5 (the same grid), each slot of the 366-day calendar (29 February = 59, 1 March = 60
          in every year) the mean of the days within 7 slots of it in all 30 years. Built once (verify-clim.yml),
          release verify-clim-z500-v1. --field msl: sea-level pressure the same way at 00Z and 12Z, in hPa
          (telecon15.yml, with the teleconnections' reference).
  acc     Per run, each model and lead over 20-80N and 20-80S (EMC's areas), cosine-of-latitude weighted: the anomaly
          correlation, centred (each anomaly less its area mean; WMO-No. 485) and uncentred, the RMSE, the mean error
          and the standard deviation of each anomaly field. acc_<init>.parquet, once ERA5 covers the run's day 15 and
          every model that can still come is in (GDPS and GEPS only while HPFX keeps them, about 50 days).

A run whose files the database holds is not fetched again, and no file is ever replaced (scripts/verify/verifydb.sh).

    python z500_acc.py fields --model GEPS --inits 2026091600 2026092000 --have have.txt --archive-out db-new
    python z500_acc.py truth --first 20260514 --last 20261020 --have have.txt --archive-out db-new
    python z500_acc.py clim --out clim_z500_1p5.npz
    python z500_acc.py clim --field msl --out clim_msl_1p5.npz
    python z500_acc.py acc --list --have have.txt
    python z500_acc.py acc --init 2026051300 2026051400 --derived db --clim clim_z500_1p5.npz --have have.txt --archive-out db-new
    python z500_acc.py recent --have have.txt    # the files verify_figs.py scores the newest runs from
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

import ens_verify as E
import mverify_scores as D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from latlon import G15, G25N, remap, slot366

G = 9.80665
DET = {"GFS": "gfs", "IFS": "ifs", "AIFS": "aifs", "GDPS": "gdps"}  # the deterministic models (mverify_scores.MODELS)
DET_LEADS = range(24, 241, 24)  # 00Z of every day to day 10, the deterministic runs' length in the database
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2 = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
CLIM_YEARS = (1991, 2020)
CLIM_HALF = 7  # slots either side of each slot of the normal
REGIONS = {"NH": (20.0, 80.0), "SH": (-80.0, -20.0)}  # EMC's extratropical belts
HPFX = {"GDPS": D.HPFX_DAYS, "GEPS": E.HPFX_DAYS}  # days ECCC's HPFX keeps a run; the other sources keep them all
START = "2026051300"  # the verification database's first run (IFS 50r1, AIFS v2): the deterministic models from here
# each model's first run here: the ensembles go back as far as their archives (IFS-ENS on its 0.25 deg grid, AIFS-ENS
# from its first open run) to give the days 1-15 teleconnection forecasts winters to be calibrated and tested on
SINCE = {
    "IFS-ENS": "2024030100",
    "GEFS": "2024030100",
    "AIFS-ENS": "2025070200",
    "GEPS": START,
    **dict.fromkeys(DET, START),
}
# each ensemble's first run with its sea-level pressure: IFS-ENS and GEFS as their 500 hPa height, from their published
# means; AIFS-ENS and GEPS from every member, from the oldest run GEPS's HPFX still held when they were added
MSL_SINCE = {"IFS-ENS": "2024030100", "GEFS": "2024030100", "AIFS-ENS": "2026092100", "GEPS": "2026092100"}
GIVE_UP_DAYS = 365  # a run this old is scored with the models it has (a backfill can still fetch any other source)
RECENT_DAYS = 30  # runs this new without their scores yet are scored at the leads ERA5 covers (``table``)


# --------------------------------------------------------------------------- fields
def messages(blob: bytes, field: str = "z500") -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """(member, latitudes, longitudes, values) of every GRIB message of a download: 500 hPa height in metres, or with
    field msl sea-level pressure in hPa. The unit is read from the magnitude, as ensmaps.to_units does: AIFS publishes
    geopotential, ECCC files have carried decametres, and pressure comes in Pa."""
    import eccodes

    out, i = [], 0
    while (i := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[i + 8 : i + 16], "big")
        gid = eccodes.codes_new_from_message(blob[i : i + n])
        i += n
        try:
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            lat = np.linspace(
                eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees"),
                eccodes.codes_get_double(gid, "latitudeOfLastGridPointInDegrees"),
                nj,
            )
            lon0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
            lon = lon0 + eccodes.codes_get_double(gid, "iDirectionIncrementInDegrees") * np.arange(ni)
            try:
                num = int(eccodes.codes_get(gid, "number"))
            except eccodes.KeyValueNotFoundError:
                num = 0  # a deterministic run, or the GEPS control, which carries no perturbation number
            v = eccodes.codes_get_values(gid).reshape(nj, ni)
        finally:
            eccodes.codes_release(gid)
        med = float(np.median(v[::7, ::7]))
        if field == "msl":
            v = v / 100.0 if med > 2000 else v
            if not 950 < np.median(v) < 1050:
                raise RuntimeError(f"sea-level pressure of member {num}: median {np.median(v):.0f} hPa")
        else:
            v = v / G if med > 20000 else v * 10.0 if med < 1000 else v
            if not 4500 < np.median(v) < 6200:
                raise RuntimeError(f"500 hPa height of member {num}: median {np.median(v):.0f} m, not a 500 hPa field")
        out.append((num, lat, lon, v))
    return out


def det_bytes(model: str, init: str, lead: int) -> bytes:
    """The 500 hPa height message of one deterministic step, by its file's index (GDPS: its own file)."""
    day, hh = init[:8], init[8:]
    if model == "GDPS":
        base = f"{D.DATAMART}/{day}/WXO-DD/model_gdps/15km/{hh}/{lead:03d}/{day}T{hh}Z_MSC_GDPS"
        return D.get(f"{base}_GeopotentialHeight_IsbL-0500_LatLon0.15_PT{lead:03d}H.grib2")
    if model == "GFS":
        return E.gefs_member(f"{D.GFS}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}", ["HGT:500 mb"])
    url = f"{D.GOOGLE}/{day}/{hh}z/{D.MODELS[model]}/0p25/oper/{day}{hh}0000-{lead}h-oper-fc"
    return E.ecmwf_bytes(url, {("gh", "pl", "500")}, ("fc",))


def file_of(model: str, init: str, members: bool = False) -> str:
    if model in DET:
        return f"z500-{DET[model]}_{init}.npz"
    return f"ens-z500{'m' if members else ''}-{E.ENSEMBLES[model]}_{init}.npz"


def run_fields(model: str, init: str, work: Path, out: Path) -> None:
    """One run's 500 hPa heights on the verification grid (the ensemble mean, and every member on the patterns' grid)."""
    if model in DET:
        z = []
        for lead in DET_LEADS:
            (_, lat, lon, v), *more = messages(det_bytes(model, init, lead))
            if more:
                raise RuntimeError(f"{model} {init} +{lead} h: {len(more) + 1} 500 hPa height messages")
            z.append(remap(v, lat, lon, G15))
        np.savez_compressed(out / file_of(model, init), lead_h=np.array(DET_LEADS), z=np.stack(z))
        return
    mean, mem, numbers = [], [], None
    for lead in E.LEADS:
        f = E.fetch(model, init, lead, "z500", work / f"{E.ENSEMBLES[model]}_{init}_{lead:03d}.grib2")
        msgs = sorted(messages(f.read_bytes()), key=lambda m: m[0])
        f.unlink()
        nums = [m[0] for m in msgs]
        if numbers is not None and nums != numbers:
            raise RuntimeError(f"{model} {init} +{lead} h: members {nums}, not those of day 1")
        numbers = nums
        mean.append(np.mean([remap(v, lat, lon, G15) for _, lat, lon, v in msgs], axis=0))
        mem.append([np.round(remap(v, lat, lon, G25N)).astype(np.int16) for _, lat, lon, v in msgs])
        print(f"{model} {init} +{lead} h: {len(msgs)} members", flush=True)
    leads = np.array(E.LEADS)
    np.savez_compressed(out / file_of(model, init), lead_h=leads, z=np.stack(mean), n_members=len(numbers))
    np.savez_compressed(
        out / file_of(model, init, members=True),
        lead_h=leads,
        member=np.array(numbers, np.int16),
        z=np.array(mem),
        lat=G25N.lat,
        lon=G25N.lon,
    )


def msl_file(model: str, init: str) -> str:
    return f"ens-msl-{E.ENSEMBLES[model]}_{init}.npz"


def msl_bytes(model: str, init: str, lead: int, work: Path, published: dict) -> tuple[bytes, str]:
    """One step's sea-level pressure: the mean the centre publishes (IFS-ENS's em, in the members' file, while the run
    has it: ``published``; GEFS's own mean file), else every member's field."""
    day, hh = init[:8], init[8:]
    if model == "GEFS":
        url = E.GEFS_URL.format(d=day, h=hh, m=E.GEFS_MEAN, L=lead)
        return E.gefs_member(url, E.GEFS["msl"]), E.GEFS_MEAN
    if model == "IFS-ENS" and published.get(model, True):
        url = f"{D.GOOGLE}/{day}/{hh}z/ifs/0p25/enfo/{day}{hh}0000-{lead}h-enfo-ef"
        try:
            return E.ecmwf_bytes(url, set(E.ECMWF["msl"]), ("em",)), "em"
        except RuntimeError as e:  # no mean in the file's index
            print(f"::warning::IFS-ENS {init} +{lead} h: {e}; the members' mean from here", flush=True)
            published[model] = False
    f = E.fetch(model, init, lead, "msl", work / f"msl-{E.ENSEMBLES[model]}_{init}_{lead:03d}.grib2")
    blob = f.read_bytes()
    f.unlink()
    return blob, "members"


def run_msl(model: str, init: str, work: Path, out: Path) -> None:
    """One run's ensemble-mean sea-level pressure at 00Z of each day to day 15, on the patterns' grid, in hPa."""
    mean, source, count, published = [], [], [], {}
    for lead in E.LEADS:
        blob, kind = msl_bytes(model, init, lead, work, published)
        msgs = messages(blob, "msl")
        if kind != "members" and len(msgs) != 1:
            raise RuntimeError(f"{model} {init} +{lead} h: {len(msgs)} messages of the published mean")
        if count and kind == "members" == source[-1] and len(msgs) != count[-1]:
            raise RuntimeError(f"{model} {init} +{lead} h: {len(msgs)} members, not those of the step before")
        _, lat, lon, _ = msgs[0]
        mean.append(remap(np.mean([v for *_, v in msgs], axis=0), lat, lon, G25N))  # the members share a grid
        source.append(kind)
        count.append(len(msgs))
    np.savez_compressed(
        out / msl_file(model, init),
        lead_h=np.array(E.LEADS),
        msl=np.stack(mean).astype(np.float32),
        source=np.array(source),
        n_fields=np.array(count, np.int16),
        lat=G25N.lat,
        lon=G25N.lon,
    )
    print(f"{model} {init}: sea-level pressure from {', '.join(sorted(set(source)))} ({max(count)} fields)", flush=True)


def fields(a) -> None:
    """The runs of one model the database lacks, and the ensembles' sea-level pressure of runs from MSL_SINCE that
    lack it. A run its source no longer holds (HPFX's GDPS and GEPS past their days, a 404; a bucket answers 403) is
    left out with a warning, as mverify_scores.download does: no later pass can fetch it, so it is not a failure."""
    have = _have(a.have)
    a.archive_out.mkdir(parents=True, exist_ok=True)
    a.work.mkdir(parents=True, exist_ok=True)
    for init in D.runs(*a.inits):
        z500 = file_of(a.model, init) not in have
        msl = a.model in MSL_SINCE and init >= MSL_SINCE[a.model] and msl_file(a.model, init) not in have
        if not (z500 or msl):
            continue
        try:
            if z500:
                run_fields(a.model, init, a.work, a.archive_out)
            if msl:
                run_msl(a.model, init, a.work, a.archive_out)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
            print(f"::warning::{a.model} {init}: not on the source ({e.code}), left out", flush=True)
            continue
        print(f"{a.model} {init}: archived", flush=True)


# --------------------------------------------------------------------------- truth and normal
def truth_file(day: pd.Timestamp, field: str = "z500") -> str:
    return f"era5-{field}_{day:%Y%m%d}00.npz"


def truth(a) -> None:
    """ERA5's 500 hPa height on the verification grid (z) and its sea-level pressure on the patterns' grid (msl, hPa)
    at 00Z of each day the database lacks, each archived once ARCO holds it."""
    import xarray as xr

    have = _have(a.have)
    a.archive_out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    stop = pd.Timestamp(ds.attrs["valid_time_stop_era5t"])
    read = {
        "z500": (ds["geopotential"].sel(level=500), 1.0 / G, G15, "z"),
        "msl": (ds["mean_sea_level_pressure"], 0.01, G25N, "msl"),
    }
    for day in pd.date_range(pd.Timestamp(a.first), min(pd.Timestamp(a.last), stop), freq="D"):
        for field, (x, scale, grid, key) in read.items():
            if truth_file(day, field) in have:
                continue
            v = x.sel(time=day).transpose("latitude", "longitude").values * scale
            if not np.isfinite(v).all():
                print(f"::warning::ERA5 {field} {day:%Y-%m-%d} 00Z: incomplete in ARCO, for the next pass", flush=True)
                continue
            lat, lon = x["latitude"].values, x["longitude"].values
            np.savez_compressed(a.archive_out / truth_file(day, field), **{key: remap(v, lat, lon, grid)})
            print(f"ERA5 {field} {day:%Y-%m-%d} 00Z: archived", flush=True)


def wb2_normal(var: str, level: int | None, hours: tuple[int, ...], scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The CLIM_YEARS normal of one variable of WeatherBench 2's zarr (at one level) at each of ``hours``: (366, hours,
    lat, lon) times ``scale``, 90N first, and the days behind each slot (366, hours). A chunk holds several six-hourly
    steps (geopotential: 8, of all 13 levels), so every chunk is read whatever is wanted of it: they are fetched side by
    side over HTTPS (xarray reads them a few at a time, over two hours) and only the wanted slices kept."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import numcodecs
    import xarray as xr

    url = f"https://storage.googleapis.com/{WB2}/{var}"
    meta = json.loads(D.get(f"{url}/.zarray"))  # the chunks' own layout, whatever the zarr library makes of it
    z = xr.open_zarr(f"gs://{WB2}", storage_options={"token": "anon"})[var]  # the coordinates
    dims = ("time", *(("level",) if level is not None else ()), "longitude", "latitude")
    if z.dims != dims or meta.get("dimension_separator", ".") != "." or meta["chunks"][1:] != meta["shape"][1:]:
        raise RuntimeError(f"WeatherBench 2's {var} is {z.dims} in chunks {meta['chunks']}, not as read here")
    lat = z["latitude"].values
    if not (np.allclose(np.sort(lat)[::-1], G15.lat) and np.allclose(z["longitude"].values, G15.lon)):
        raise RuntimeError("WeatherBench 2's grid is not the verification grid")
    shape, dtype, codec = meta["chunks"], np.dtype(meta["dtype"]), numcodecs.get_codec(meta["compressor"])
    nt = shape[0]
    pick = (slice(None), int(np.flatnonzero(z["level"].values == level)[0])) if level is not None else (slice(None),)
    t = pd.DatetimeIndex(z["time"].values)
    want = (t.year >= CLIM_YEARS[0]) & (t.year <= CLIM_YEARS[1]) & np.isin(t.hour, hours)

    def block(b: int) -> np.ndarray:
        """The wanted level of time chunk ``b``: (steps, longitude, latitude)."""
        raw = codec.decode(D.get(f"{url}/{b}.{'.'.join('0' * (len(shape) - 1))}"))
        return np.frombuffer(raw, dtype).reshape(shape)[pick]

    blocks = np.unique(np.flatnonzero(want) // nt)
    total, count = np.zeros((366, len(hours), *G15.shape)), np.zeros((366, len(hours)))
    with ThreadPoolExecutor(32) as ex:
        for i in range(0, blocks.size, 256):  # a batch at a time, so the chunks waiting in order stay few
            batch = blocks[i : i + 256]
            for b, v in zip(batch, ex.map(block, batch), strict=True):
                k = np.flatnonzero(want[b * nt : (b + 1) * nt])
                f = v[k].transpose(0, 2, 1).astype(np.float64) * scale  # (steps, lat, lon)
                f = f[:, ::-1] if lat[0] < lat[-1] else f
                tk = t[b * nt + k]
                s, h = slot366(tk), np.searchsorted(hours, tk.hour)
                np.add.at(total, (s, h), f)
                np.add.at(count, (s, h), 1)
            print(f"normal: read to {t[batch[-1] * nt]:%Y-%m-%d}", flush=True)
    if count.sum() != want.sum():
        raise RuntimeError(f"normal: {int(count.sum())} of {int(want.sum())} fields read")
    rows = np.arange(366)
    near = (np.abs((rows[:, None] - rows[None, :] + 183) % 366 - 183) <= CLIM_HALF).astype(float)  # around the year
    days = near @ count
    normal = np.tensordot(near, total, axes=1) / days[:, :, None, None]
    return normal, days


def clim(a) -> None:
    """The normal on the verification grid: 500 hPa height (m) at 00Z, the hour of the fields, or with --field msl
    sea-level pressure (hPa) at 00Z and 12Z, the teleconnections' hours."""
    if a.field == "msl":
        hours = (0, 12)
        normal, days = wb2_normal("mean_sea_level_pressure", None, hours, 0.01)
        np.savez_compressed(a.out, msl=normal.astype(np.float32), hours=np.array(hours), lat=G15.lat, lon=G15.lon,
                            days=days.astype(np.int32))  # fmt: skip
    else:
        normal, days = wb2_normal("geopotential", 500, (0,), 1.0 / G)
        np.savez_compressed(
            a.out, z500=normal[:, 0].astype(np.float32), lat=G15.lat, lon=G15.lon, days=days[:, 0].astype(np.int32)
        )
    print(f"normal: {a.out} ({int(days[0].max())} days a slot)", flush=True)


# --------------------------------------------------------------------------- scores
def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((w * x).sum() / w.sum())


def scores(z: np.ndarray, era: np.ndarray, normal: np.ndarray) -> dict[str, dict[str, float]]:
    """{region: {acc, acc_u, rmse, me, sd_f, sd_a}} of one forecast field against its analysis."""
    out = {}
    for region, (lo, hi) in REGIONS.items():
        rows = (G15.lat >= lo) & (G15.lat <= hi)
        w = np.broadcast_to(np.cos(np.radians(G15.lat[rows]))[:, None], (rows.sum(), G15.lon.size))
        f, o, e = z[rows] - normal[rows], era[rows] - normal[rows], z[rows] - era[rows]
        fc, oc = f - _wmean(f, w), o - _wmean(o, w)
        out[region] = {
            "acc": _wmean(fc * oc, w) / np.sqrt(_wmean(fc**2, w) * _wmean(oc**2, w)),
            "acc_u": _wmean(f * o, w) / np.sqrt(_wmean(f**2, w) * _wmean(o**2, w)),
            "rmse": np.sqrt(_wmean(e**2, w)),
            "me": _wmean(e, w),
            "sd_f": np.sqrt(_wmean(fc**2, w)),
            "sd_a": np.sqrt(_wmean(oc**2, w)),
        }
    return out


def run_scores(init: str, files: dict[str, Path], truth_dir: Path, normal: np.ndarray) -> pd.DataFrame:
    """Every lead of every model of one run, for the regions, against the ERA5 files in ``truth_dir`` (a lead ERA5
    does not cover yet is left out: ``due`` waits for all of them, ``table`` scores what there is)."""
    t0, rows = pd.to_datetime(init, format="%Y%m%d%H"), []
    for model, path in files.items():
        d = np.load(path)
        for lead, z in zip(d["lead_h"], d["z"], strict=True):
            valid = t0 + pd.Timedelta(hours=int(lead))
            if not (truth_dir / truth_file(valid)).exists():
                continue
            era = np.load(truth_dir / truth_file(valid))["z"]
            for region, s in scores(z, era, normal[slot366(pd.DatetimeIndex([valid]))[0]]).items():
                rows.append(
                    {"model": model, "init": t0, "lead_h": int(lead), "valid_time": valid, "region": region, **s}
                )
    return pd.DataFrame(rows)


def models_of(init: str, have: set[str]) -> list[str]:
    return [m for m in [*DET, *E.ENSEMBLES] if file_of(m, init) in have]


def due(have: set[str], today: pd.Timestamp) -> dict[str, list[str]]:
    """The runs whose scores are due, with their models: ERA5 covers day 15, and every model is in that can still
    come (from its SINCE; GDPS and GEPS while HPFX keeps the run, the others for GIVE_UP_DAYS)."""
    out = {}
    for init in sorted({n.split("_")[1][:10] for n in have if n.startswith(("z500-", "ens-z500-"))}):
        t0 = pd.to_datetime(init, format="%Y%m%d%H")
        days = pd.date_range(t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=15), freq="D")
        models = models_of(init, have)
        coming = [
            m
            for m in [*DET, *E.ENSEMBLES]
            if init >= SINCE[m] and t0 >= today - pd.Timedelta(days=HPFX.get(m, GIVE_UP_DAYS))
        ]
        if (
            f"acc_{init}.parquet" not in have
            and set(coming) <= set(models)
            and all(truth_file(d) in have for d in days)
        ):
            out[init] = models
    return out


def recent(have: set[str], today: pd.Timestamp) -> list[str]:
    """The files ``table`` scores the runs of the newest RECENT_DAYS without their scores from: their fields (the
    ensembles' means) and the ERA5 days the database has of their forecasts."""
    out = set()
    for init in sorted({n.split("_")[1][:10] for n in have if n.startswith(("z500-", "ens-z500-"))}):
        t0 = pd.to_datetime(init, format="%Y%m%d%H")
        if f"acc_{init}.parquet" in have or t0 < today - pd.Timedelta(days=RECENT_DAYS):
            continue
        out |= {file_of(m, init) for m in models_of(init, have)}
        out |= {n for d in range(1, 16) if (n := truth_file(t0 + pd.Timedelta(days=d))) in have}
    return sorted(out)


def table(derived: Path, normal: np.ndarray) -> pd.DataFrame:
    """Every run's scores: its acc_ file in ``derived``, or for a recent run without one, its fields there scored at
    the leads ERA5 covers."""
    done = {f.stem.split("_")[1]: f for f in derived.glob("acc_*.parquet")}
    frames = [pd.read_parquet(f) for f in done.values()]
    for init in sorted({f.stem.split("_")[1] for f in derived.glob("*z500-*_*.npz")} - set(done)):
        files = {m: f for m in [*DET, *E.ENSEMBLES] if (f := derived / file_of(m, init)).exists()}
        frames.append(run_scores(init, files, derived, normal))
    t = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return t.astype({"model": str, "region": str})


def acc(a) -> None:
    have = _have(a.have)
    runs = due(have, pd.Timestamp.now(tz="UTC").tz_localize(None).normalize())
    if a.list:  # the files each due run's scores are made from, for the workflow to download
        for init, models in runs.items():
            t0 = pd.to_datetime(init, format="%Y%m%d%H")
            days = pd.date_range(t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=15), freq="D")
            print(init, *(file_of(m, init) for m in models), *(truth_file(d) for d in days))
        return
    normal = np.load(a.clim)["z500"]
    a.archive_out.mkdir(parents=True, exist_ok=True)
    for init, models in runs.items():
        if a.init and init not in a.init:
            continue
        frame = run_scores(init, {m: a.derived / file_of(m, init) for m in models}, a.derived, normal)
        frame.astype({"model": "category", "region": "category"}).to_parquet(
            a.archive_out / f"acc_{init}.parquet", compression="zstd"
        )
        print(f"{init}: scores of {', '.join(models)}", flush=True)


def _have(path: Path | None) -> set[str]:
    return set(path.read_text().split()) if path and path.exists() else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fields", help="the 500 hPa heights of runs of one model")
    f.add_argument("--model", required=True, choices=[*DET, *E.ENSEMBLES])
    f.add_argument("--inits", nargs=2, required=True, metavar=("FIRST", "LAST"), help="00Z runs, YYYYMMDDHH")
    f.add_argument("--work", type=Path, default=Path("z500-work"), help="download directory")
    t = sub.add_parser("truth", help="ERA5's 500 hPa height at 00Z of each day ARCO holds")
    t.add_argument("--first", required=True, help="YYYYMMDD")
    t.add_argument("--last", required=True, help="YYYYMMDD")
    c = sub.add_parser("clim", help="the ERA5 1991-2020 normal on the verification grid (built once, offline)")
    c.add_argument("--field", choices=("z500", "msl"), default="z500")
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("acc", help="each due run's scores against ERA5")
    s.add_argument("--list", action="store_true", help="print each due run and the files its scores are made from")
    s.add_argument("--init", nargs="+", help="only these runs (YYYYMMDDHH)")
    s.add_argument("--derived", type=Path, default=Path("."), help="the due runs' fields and ERA5 files, downloaded")
    s.add_argument("--clim", type=Path, help="the normal (release verify-clim-z500-v1)")
    r = sub.add_parser("recent", help="print the files the nightly scores the newest runs without their scores from")
    for p in (f, t, s, r):
        p.add_argument("--have", type=Path, help="names of the files the database holds, one per line")
        p.add_argument("--archive-out", type=Path, default=Path("db-new"), help="where the files to add are written")
    a = ap.parse_args()
    if a.cmd == "recent":
        print(*recent(_have(a.have), pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()))
        return
    {"fields": fields, "truth": truth, "clim": clim, "acc": acc}[a.cmd](a)


if __name__ == "__main__":
    main()
