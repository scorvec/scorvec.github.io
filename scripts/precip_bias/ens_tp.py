#!/usr/bin/env python3
"""Rain from the four ensembles of the tropical rainfall forecast, as 1° box means on the global band of grid.py.

One reader per system; each returns rain accumulated from the 00Z start (mm) at the requested steps, member by member:

  ifs, aifs-ens  ECMWF open data from the Google Cloud mirror only (scripts/ecmwf/rangefetch.py), tp accumulated from
                 the start (ecmwf_tp.decode reads the GRIB units: IFS in m, AIFS in kg m-2). Perturbed members 1-50.
  gefs           NOAA GEFS v12 0.5° (pgrb2a) from the AWS Open Data bucket noaa-gefs-pds, one byte range per record
                 through each file's .idx (never NOMADS). APCP comes in 6-hour buckets ("{h-6}-{h} hour acc fcst" in
                 every 6-hourly file), so the total from the start is the sum of its buckets. Control + 30 members.
  geps           ECCC GEPS 0.5° from the MSC Datamart (dd.weather.gc.ca, whose dated directories keep about 30 days):
                 one file per step holding every member, APCP_SFC_0 accumulated from the start in kg m-2. Control + 20.

The readers fetch in threads and decode in the calling thread: eccodes inside a thread pool crashed the interpreter on
the runner (exit 139, scripts/mjo/src/overturn.py).

Two products:
  archive  the bias reference of global_bias.py and blend.py: members 1-6 of each 00Z run at 168 h and 336 h (weeks 1
           and 2), one file a month in ecmwf_tp.py's format, {model}_{YYYYMM}.npz. GEFS from 2024-10, the start of
           the IMERG archive the bias is measured against; GEPS only as far back as the Datamart still reaches. A month
           is merged with the file already in the release (--have), so only new runs are fetched and the GEPS runs that
           have left the Datamart are kept.
  live     one run for the forecast maps (tropical_rain.py): every member, four windows, rainfc_{model}_{YYYYMMDD}.npz
           (win: member x window x lat x lon, float16 mm): this run's weeks 1 and 2 (0-168 h, 168-336 h) and the same
           calendar weeks as the next day's run will see them (24-192 h, 192-360 h), for that run's change map.

    python scripts/precip_bias/ens_tp.py month --model gefs --month 2025-01 --out DIR [--have FILE] [--last YYYY-MM-DD]
    python scripts/precip_bias/ens_tp.py live --model geps --date 20261016 --out DIR
-> exit 0 (month: also when no run was new, nothing written), 1 when nothing could be built, 3 (live) when the run is
   not complete on its source yet.
"""

from __future__ import annotations

import pyproj  # noqa: F401  side effect: load before eccodes/xarray, which preload eckit's own libproj

import argparse
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecmwf"))
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import grid as G
import webget

MODELS = ("ifs", "aifs-ens", "gefs", "geps")
LIVE_MEMBERS = {
    "ifs": tuple(range(1, 51)),  # the open data carry no IFS-ENS control
    "aifs-ens": tuple(range(1, 51)),
    "gefs": tuple(range(31)),  # 0 = the control gec00
    "geps": tuple(range(21)),  # 0 = the control
}
# Archive: six perturbed members of every system - the bias is a mean, and its noise is set by the weather sampled across
# runs, not by the member count (as ecmwf_tp.MEMBERS).
ARCH_MEMBERS, ARCH_STEPS = (1, 2, 3, 4, 5, 6), (168, 336)
LIVE_STEPS = (24, 168, 192, 336, 360)
WINDOWS = {"wk1": (0, 168), "wk2": (168, 336), "next1": (24, 192), "next2": (192, 360)}  # hours from the start
FIRST = {"gefs": date(2024, 10, 1)}  # archive start (IMERG archive start; GEFS v12 itself runs since 2020-09-24)
DATAMART_DAYS = 30  # how far back the Datamart's dated directories reach

GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
GEPS_URL = (
    "https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/00/{h:03d}/"
    "CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_{d}00_P{h:03d}_allmbrs.grib2"
)
GEFS_WORKERS, GEPS_WORKERS = 32, 3  # S3 takes many parallel range requests; the Datamart is a shared public service

_LOCK = threading.Lock()
FETCHED = {"bytes": 0, "requests": 0}  # S3 + Datamart traffic of this process (ECMWF is accounted by rangefetch)


class NotPublished(RuntimeError):
    """The run, or a step of it, is not on its source (yet, or any more)."""


# ── transfer and decoding ────────────────────────────────────────────────────────────────────────────────────────────
def _get(url: str, rng: tuple[int, int | None] | None = None, timeout: int = 120) -> bytes | None:
    """Body of `url` (or of a byte range of it) with the site's honest User-Agent and retries; None on a 404."""
    hdr = {"Range": f"bytes={rng[0]}-{'' if rng[1] is None else rng[1]}"} if rng else None
    try:
        b = webget.get(url, headers=hdr, timeout=timeout, tries=4, backoff=4.0)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with _LOCK:
        FETCHED["bytes"] += len(b)
        FETCHED["requests"] += 1
    return b


def _exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": webget.DEFAULT_UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001  any failure = not there yet
        return False


def _messages(blob: bytes) -> list[bytes]:
    """Concatenated GRIB2 messages -> single messages."""
    out, o = [], 0
    while (k := blob.find(b"GRIB", o)) >= 0:
        if blob[k + 7] != 2:
            raise ValueError("not GRIB2")
        n = int.from_bytes(blob[k + 8 : k + 16], "big")
        out.append(blob[k : k + n])
        o = k + n
    return out


def decode(msg: bytes) -> tuple[dict, np.ndarray, float]:
    """One global regular lat-lon rain message -> (meta: number, end step), field in mm (rows 90°N -> 90°S, columns from
    180°W), grid spacing in degrees."""
    import eccodes

    g = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
        lat0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
        di = eccodes.codes_get(g, "iDirectionIncrementInDegrees")
        units = eccodes.codes_get(g, "units")
        code = tuple(eccodes.codes_get(g, k) for k in ("discipline", "parameterCategory", "parameterNumber"))
        meta = {
            k: (eccodes.codes_get(g, k) if eccodes.codes_is_defined(g, k) else None)
            for k in ("perturbationNumber", "endStep")
        }
        v = eccodes.codes_get_values(g).reshape(nj, ni)
        if eccodes.codes_get(g, "bitmapPresent"):
            v = np.where(v == eccodes.codes_get(g, "missingValue"), np.nan, v)
    finally:
        eccodes.codes_release(g)
    if abs(ni * di - 360) > 1e-6 or abs((nj - 1) * di - 180) > 1e-6 or abs(abs(lat0) - 90) > 1e-6:
        raise ValueError(f"not a global regular grid: Ni {ni} Nj {nj} step {di} first {lat0},{lon0}")
    scale = {"m": 1000.0, "kg m**-2": 1.0, "kg m-2": 1.0, "mm": 1.0}.get(units)
    # WMO GRIB2 table 4.2 0/1/8 "total precipitation" is kg m-2 by definition; eccodes may leave its units unnamed under a
    # centre's own tables (ECCC's)
    if scale is None and code == (0, 1, 8):
        scale = 1.0
    if scale is None:
        raise ValueError(f"unknown rain units {units!r} (parameter {code})")
    meta = {"number": int(meta["perturbationNumber"] or 0), "end": meta["endStep"]}
    return meta, G.canonical(v * scale, lat0, lon0), float(di)


# ── the readers: acc (step, member, NY, NX), mm from the start ───────────────────────────────────────────────────────
def ecmwf_run(model: str, d: date, steps, members) -> np.ndarray:
    import ecmwf_tp as E
    import rangefetch as R

    out = np.full((len(steps), len(members), G.NY, G.NX), np.nan, "float32")
    for si, step in enumerate(steps):
        for k in range(4):
            try:
                f = E.decode(R.fetch(f"{d:%Y%m%d}", "00", model, step, param="tp", numbers=list(members), kind="pf"))
                miss = [m for m in members if m not in f]
                if len(miss) > 5:
                    raise ValueError(f"{len(miss)} members missing")
                for mi, m in enumerate(members):
                    if m in f:
                        out[si, mi] = G.ecmwf_to_1deg(f.pop(m))
                break
            except R.IndexUnavailable as ex:
                if ex.not_found:
                    raise NotPublished(f"{model} {d} +{step} h not on the mirror") from ex
                if k == 3:
                    raise
            except Exception as ex:  # noqa: BLE001
                if k == 3:
                    raise
                print(f"  {model} +{step} h: {repr(ex)[:90]}; retrying", flush=True)
            time.sleep(10 * (k + 1))
    return out


def gefs_url(d: date, member: int, h: int) -> str:
    mem = "gec00" if member == 0 else f"gep{member:02d}"
    return f"{GEFS_S3}/gefs.{d:%Y%m%d}/00/atmos/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{h:03d}"


def gefs_bucket(d: date, member: int, h: int) -> bytes | None:
    """The APCP record of the 6 hours ending at h, byte-ranged through the file's .idx; None if the file is not there."""
    url = gefs_url(d, member, h)
    idx = _get(url + ".idx")
    if idx is None:
        return None
    lines = [ln.split(":") for ln in idx.decode().splitlines() if ln.strip()]
    want = f"{h - 6}-{h} hour acc fcst"
    for i, p in enumerate(lines):
        if p[3] == "APCP" and p[4] == "surface" and p[5] == want:
            b = _get(url, (int(p[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None))
            if b is None or not b.startswith(b"GRIB"):
                raise ValueError(f"bad range for APCP {want} in {url.rsplit('/', 1)[-1]}")
            return b
    raise ValueError(f"no APCP {want} in {url.rsplit('/', 1)[-1]}.idx")


def gefs_run(d: date, steps, members, workers: int = GEFS_WORKERS) -> np.ndarray:
    """Buckets fetched in threads, at most 2 x workers ahead of the summation, which decodes in this thread."""
    out = np.full((len(steps), len(members), G.NY, G.NX), np.nan, "float32")
    jobs = deque((mi, h) for mi in range(len(members)) for h in range(6, max(steps) + 1, 6))
    total: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(workers) as ex:
        live = deque()
        while jobs or live:
            while jobs and len(live) < 2 * workers:
                mi, h = jobs.popleft()
                live.append((mi, h, ex.submit(gefs_bucket, d, members[mi], h)))
            mi, h, fut = live.popleft()
            msg = fut.result()
            if msg is None:
                raise NotPublished(f"GEFS {d} member {members[mi]} +{h} h not on S3")
            _, f, res = decode(msg)
            total[mi] = total.get(mi, 0.0) + np.maximum(f, 0.0)  # a missing point stays NaN: run() drops the member
            if h in steps:
                out[steps.index(h), mi] = G.to_1deg(total[mi], res)
    return out


def geps_run(d: date, steps, members, workers: int = GEPS_WORKERS) -> np.ndarray:
    """One all-member file per step, fetched in threads, decoded here in step order."""
    out = np.full((len(steps), len(members), G.NY, G.NX), np.nan, "float32")
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_get, GEPS_URL.format(d=f"{d:%Y%m%d}", h=h), None, 300) for h in steps]
        for si, (h, fut) in enumerate(zip(steps, futs)):
            blob = fut.result()
            if blob is None:
                raise NotPublished(f"GEPS {d} P{h:03d} not on the Datamart")
            got = {}
            for msg in _messages(blob):
                meta, f, res = decode(msg)
                if meta["end"] is not None and int(meta["end"]) != h:
                    raise ValueError(f"GEPS P{h:03d} message ends at {meta['end']} h")
                got[meta["number"]] = (f, res)
            miss = [m for m in members if m not in got]
            if miss:
                raise ValueError(f"GEPS {d} P{h:03d}: members {miss} missing")
            for mi, m in enumerate(members):
                out[si, mi] = G.to_1deg(*got[m])
    return out


def plausible(acc: np.ndarray, steps) -> tuple[np.ndarray | None, str | None]:
    """(acc made non-decreasing in step, None) or (None, why). The band mean of the first accumulation must be 0.3-15 mm a
    day; an accumulation may fall by up to 0.5 mm (packing), which is clipped."""
    w = G.area_weights()
    d1 = float(np.nansum(acc[0].mean(0) * w) / w.sum()) / (steps[0] / 24)
    if not (0.3 < d1 < 15.0):
        return None, f"implausible first-step domain mean {d1:.2f} mm/day"
    if len(steps) > 1 and np.nanmin(np.diff(acc, axis=0)) < -0.5:
        return None, f"accumulation falls by {-np.nanmin(np.diff(acc, axis=0)):.2f} mm"
    return np.maximum.accumulate(acc, axis=0), None


def run(model: str, d: date, steps, members) -> tuple[np.ndarray, tuple]:
    """(acc (step, member, NY, NX) mm from the start, the members it holds): members missing from the source are dropped
    (ECMWF: up to 5). Raises NotPublished when the run is not on its source, ValueError when it fails the checks."""
    if model in ("ifs", "aifs-ens"):
        acc = ecmwf_run(model, d, steps, members)
    elif model == "gefs":
        acc = gefs_run(d, steps, members)
    elif model == "geps":
        acc = geps_run(d, steps, members)
    else:
        raise ValueError(f"unknown model {model!r}")
    ok = np.isfinite(acc).all(axis=(0, 2, 3))
    acc, why = plausible(acc[:, ok], steps)
    if acc is None:
        raise ValueError(f"{model} {d}: {why}")
    return acc, tuple(m for m, k in zip(members, ok) if k)


def ready(model: str, d: date) -> bool:
    """Is the 00Z run of day d on its source to 360 h? (ECMWF: the mirror's completeness sentinel; GEFS: the last member's
    last file; GEPS: the last step's file.)"""
    if model in ("ifs", "aifs-ens"):
        import rangefetch as R

        return R.cycle_complete(f"{d:%Y%m%d}", "00", model, last_step=LIVE_STEPS[-1])
    if model == "gefs":
        return _exists(gefs_url(d, LIVE_MEMBERS["gefs"][-1], LIVE_STEPS[-1]) + ".idx")
    return _exists(GEPS_URL.format(d=f"{d:%Y%m%d}", h=LIVE_STEPS[-1]))


# ── live files ───────────────────────────────────────────────────────────────────────────────────────────────────────
def live_path(out: Path, model: str, d: date) -> Path:
    return Path(out) / f"rainfc_{model}_{d:%Y%m%d}.npz"


def save_live(out: Path, model: str, d: date, acc: np.ndarray, members) -> Path:
    """acc (LIVE_STEPS, member, NY, NX) -> the four windows of every member (float16 mm: 0.05 % precision, to 65,504 mm)."""
    win = []
    for a, b in WINDOWS.values():
        lo = acc[LIVE_STEPS.index(a)] if a else 0.0
        win.append(np.maximum(acc[LIVE_STEPS.index(b)] - lo, 0.0))
    p = live_path(out, model, d)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        init=f"{d:%Y%m%d}",
        model=model,
        members=np.array(members),
        windows=np.array(list(WINDOWS)),
        win=np.stack(win, axis=1).astype("float16"),
        lat=G.LAT,
        lon=G.LON,
        made=datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
    )
    return p


def load_live(p: Path) -> tuple[dict, int]:
    """{window: (member, NY, NX) mm} and the member count of a live file."""
    z = np.load(p)
    if tuple(z["windows"]) != tuple(WINDOWS) or z["win"].shape[-2:] != (G.NY, G.NX):
        raise ValueError(f"{p.name}: windows {tuple(z['windows'])}, shape {z['win'].shape}")
    win = z["win"].astype("float32")
    return {k: win[:, i] for i, k in enumerate(WINDOWS)}, win.shape[0]


def live(model: str, d: date, out: Path) -> int:
    p = live_path(out, model, d)
    if p.exists():
        print(f"{p.name} already there", flush=True)
        return 0
    t0 = time.time()
    try:
        acc, members = run(model, d, LIVE_STEPS, LIVE_MEMBERS[model])
    except NotPublished as ex:
        print(f"{model} {d}: {ex}", flush=True)
        return 3
    save_live(out, model, d, acc, members)
    print(
        f"{p.name}: {len(members)} members, {p.stat().st_size / 1e6:.1f} MB, {(time.time() - t0) / 60:.1f} min; "
        f"{FETCHED['requests']} requests, {FETCHED['bytes'] / 1e6:.0f} MB from S3/Datamart",
        flush=True,
    )
    return 0


# ── archive months ───────────────────────────────────────────────────────────────────────────────────────────────────
def month(model: str, ym: str, out: Path, last: date | None = None, have: Path | None = None) -> int:
    y, m = map(int, ym.split("-"))
    d0 = date(y, m, 1)
    d1 = (d0 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    reach = FIRST[model] if model == "gefs" else datetime.now(UTC).date() - timedelta(days=DATAMART_DAYS)
    d0, d1 = max(d0, reach), min(d1, last) if last else d1
    held = {}
    if have and Path(have).exists():
        z = np.load(have)
        if tuple(z["steps"]) != ARCH_STEPS or tuple(z["members"]) != ARCH_MEMBERS or z["acc"].shape[-1] != G.NX:
            raise SystemExit(
                f"{Path(have).name}: steps {tuple(z['steps'])}, members {tuple(z['members'])} - not this archive"
            )
        held = {datetime.strptime(str(s), "%Y%m%d").date(): a for s, a in zip(z["inits"], z["acc"])}
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    t0, new, missing = time.time(), {}, []
    for d in (d for d in days if d not in held):
        try:
            acc, members = run(model, d, ARCH_STEPS, ARCH_MEMBERS)
            if members != ARCH_MEMBERS:
                raise ValueError(f"members {sorted(set(ARCH_MEMBERS) - set(members))} missing")
            new[d] = acc
        except Exception as ex:  # noqa: BLE001  one bad run must not stop the month
            missing.append(f"{d:%Y%m%d}: {repr(ex)[:100]}")
            print(f"  {model} {missing[-1]}", flush=True)
    if not new:
        print(f"{model} {ym}: no new runs ({len(held)} held)", flush=True)
        return 0 if held else 1
    runs = dict(sorted({**held, **new}.items()))
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{model}_{y:04d}{m:02d}.npz"
    np.savez_compressed(
        f,
        inits=np.array([f"{d:%Y%m%d}" for d in runs]),
        acc=np.stack(list(runs.values())).astype("float32"),
        steps=np.array(ARCH_STEPS),
        members=np.array(ARCH_MEMBERS),
        lat=G.LAT,
        lon=G.LON,
        missing=np.array(missing, dtype=str),
    )
    print(
        f"{model} {ym}: {len(new)} new + {len(held)} held runs, {f.stat().st_size / 1e6:.0f} MB, "
        f"{(time.time() - t0) / 60:.1f} min; {FETCHED['requests']} requests, {FETCHED['bytes'] / 1e9:.2f} GB",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("month", help="archive month (members 1-6, weeks 1 and 2)")
    a1.add_argument("--model", required=True, choices=["gefs", "geps"])
    a1.add_argument("--month", required=True, help="YYYY-MM")
    a1.add_argument("--out", required=True)
    a1.add_argument("--last", help="last init to include (YYYY-MM-DD); default: end of month")
    a1.add_argument("--have", help="this month's file as the release holds it, to merge with")
    a2 = sub.add_parser("live", help="one run, every member, the four windows of the forecast maps")
    a2.add_argument("--model", required=True, choices=MODELS)
    a2.add_argument("--date", required=True, help="init YYYYMMDD (00Z)")
    a2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "month":
        last = date.fromisoformat(a.last) if a.last else None
        return month(a.model, a.month, Path(a.out), last, Path(a.have) if a.have else None)
    return live(a.model, datetime.strptime(a.date, "%Y%m%d").date(), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
