#!/usr/bin/env python3
"""ECMWF ensemble rainfall archive over the Americas - the forecast half of the model-rain bias field.

ECMWF publishes no reforecast in its open data, so the IFS and AIFS ensembles' own rain climate has to be built from
the operational archive on the Google Cloud mirror (the only ECMWF host this site may use): IFS-ENS 0.25° from
2024-03-01, AIFS-ENS from 2025-07-02. Per 00Z run we keep six perturbed members (1-6: the bias is a mean, and its
noise is set by the weather sampled across runs, not by the member count) and only the accumulated total
precipitation at the six steps that bound the lead bands below, so a run costs ~31 MB instead of ~140 MB for every
day of all members:

    day 1 (0-24 h), days 2-3, days 4-7, days 8-10, days 11-14 (with 4-7: weeks 1 and 2), day 15

Each global field is reduced at once to the Americas 1° grid (grid.py) and stored as accumulations in mm. UNIT TRAP:
IFS tp arrives in m, AIFS tp in kg m-2 (= mm); the GRIB `units` key decides, nothing is assumed.

    python scripts/precip_bias/ecmwf_tp.py --model ifs --month 2024-03 --out DIR
    python scripts/precip_bias/ecmwf_tp.py --model ifs --month 2026-05 --out DIR --domain global

--domain global (the tropical rainfall forecast on circulation.html): the 60°S-60°N band at all longitudes and
only the week-1 / week-2 bounds (168 h, 336 h) - ~10 MB a run.
-> DIR/{model}_{YYYYMM}.npz: inits (YYYYMMDD), acc (run, step, member, lat, lon) float32 mm, steps, members, missing
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pyproj  # noqa: F401  side effect: load before eccodes/xarray, which preload eckit's own libproj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rangefetch as R
import grid as G

STEPS = (24, 72, 168, 240, 336, 360)
STEPS_GLOBAL = (168, 336)
BANDS = (
    (0, 24, "day 1"),
    (24, 72, "days 2-3"),
    (72, 168, "days 4-7"),
    (168, 240, "days 8-10"),
    (240, 336, "days 11-14"),
    (336, 360, "day 15"),
)
MEMBERS = (1, 2, 3, 4, 5, 6)
FIRST = {"ifs": date(2024, 3, 1), "aifs-ens": date(2025, 7, 2)}


def decode(blob: bytes) -> dict:
    """{member number: (721, 1440) field in mm} from a concatenated GRIB2 blob."""
    import eccodes

    out = {}
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tf:
        tf.write(blob)
        tf.flush()
        with open(tf.name, "rb") as f:
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    if eccodes.codes_get(h, "shortName") != "tp":
                        continue
                    ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                    lat0, lon0 = (
                        eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                        eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                    )
                    if (ni, nj) != (1440, 721) or lat0 != 90.0 or lon0 not in (180.0, -180.0):
                        raise ValueError(f"unexpected grid Ni {ni} Nj {nj} first {lat0},{lon0}")
                    units = eccodes.codes_get(h, "units")
                    scale = {"m": 1000.0, "kg m**-2": 1.0, "kg m-2": 1.0, "mm": 1.0}.get(units)
                    if scale is None:
                        raise ValueError(f"unknown tp units {units!r}")
                    out[int(eccodes.codes_get(h, "number"))] = eccodes.codes_get_values(h).reshape(721, 1440) * scale
                finally:
                    eccodes.codes_release(h)
    return out


def one_run(model: str, d: date, tries: int = 4):
    """(acc (step, member, NY, NX) mm, None) or (None, reason)."""
    ds = d.strftime("%Y%m%d")
    acc = np.full((len(STEPS), len(MEMBERS), G.NY, G.NX), np.nan, dtype="float32")
    for si, step in enumerate(STEPS):
        for k in range(tries):
            try:
                f = decode(R.fetch(ds, "00", model, step, param="tp", numbers=list(MEMBERS), kind="pf"))
                miss = [m for m in MEMBERS if m not in f]
                if miss:
                    raise ValueError(f"members {miss} missing")
                for mi, m in enumerate(MEMBERS):
                    acc[si, mi] = G.ecmwf_to_1deg(f[m])
                break
            except R.IndexUnavailable as ex:
                if ex.not_found:
                    return None, f"step {step} not published"
                if k == tries - 1:
                    return None, f"step {step}: {repr(ex)[:80]}"
            except Exception as ex:  # noqa: BLE001
                if k == tries - 1:
                    return None, f"step {step}: {repr(ex)[:80]}"
            time.sleep(3 * (k + 1))
    # plausibility: domain mean of the first accumulation 0.3-15 mm per day, accumulations never fall (a packing wobble
    # of 0.05 mm is clipped)
    w = G.area_weights()
    d1 = float(np.nansum(acc[0].mean(0) * w) / w.sum()) / (STEPS[0] / 24)
    if not (0.3 < d1 < 15.0):
        return None, f"implausible first-step domain mean {d1:.2f} mm/day"
    drop = np.diff(acc, axis=0)
    if np.nanmin(drop) < -0.5:
        return None, f"accumulation falls by {-np.nanmin(drop):.2f} mm"
    acc = np.maximum.accumulate(acc, axis=0)
    return acc, None


def month(model: str, ym: str, out: Path, threads: int = 2, last: date | None = None) -> Path | None:
    y, m = map(int, ym.split("-"))
    d0 = max(date(y, m, 1), FIRST[model])
    d1 = (date(y, m, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    if last:
        d1 = min(d1, last)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    if not days:
        return None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(lambda d: one_run(model, d), days))
    ok = [(d, a) for d, (a, why) in zip(days, res) if a is not None]
    missing = [f"{d:%Y%m%d}: {why}" for d, (a, why) in zip(days, res) if a is None]
    for s in missing:
        print(f"  {model} {s}", flush=True)
    if not ok:
        print(f"{model} {ym}: no complete runs", flush=True)
        return None
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{model}_{y:04d}{m:02d}.npz"
    np.savez_compressed(
        f,
        inits=np.array([f"{d:%Y%m%d}" for d, _ in ok]),
        acc=np.stack([a for _, a in ok]),
        steps=np.array(STEPS),
        members=np.array(MEMBERS),
        lat=G.LAT,
        lon=G.LON,
        missing=np.array(missing, dtype=str),
    )
    print(
        f"{model} {ym}: {len(ok)}/{len(days)} runs, {f.stat().st_size / 1e6:.0f} MB, {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(FIRST))
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--last", help="last init date to include (YYYY-MM-DD); default: end of month")
    ap.add_argument("--domain", default="americas", choices=["americas", "global"])
    a = ap.parse_args()
    if a.domain == "global":
        global STEPS  # noqa: PLW0603  set once from --domain before any use
        G.use_domain("global")
        STEPS = STEPS_GLOBAL
    f = month(a.model, a.month, Path(a.out), a.threads, date.fromisoformat(a.last) if a.last else None)
    return 0 if f else 1


if __name__ == "__main__":
    sys.exit(main())
