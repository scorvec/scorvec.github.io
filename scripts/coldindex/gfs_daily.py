#!/usr/bin/env python3
"""Daily raw inputs of the US cold-risk index from the GFS analysis (2026-10-15).

GFS 0.5-degree analyses (f000 of the 00/06/12/18 UTC cycles) from NOAA's S3 open-data mirror, byte-ranged through the
.idx files (the site rule for NOAA models). Each field is reduced exactly as the ERA5 training data were built
(~/data_archive/coldspell/code/build_indices.py + predictors.py):
  - to the 1.5-degree grid by a 3x3 block mean of the 0.5-degree points, the same cells as WeatherBench2's conservative
    1.5-degree ERA5;
  - t2m_us: US population-weighted 2 m temperature (GeoNames places >= 15,000, reference/pop_us_1p5.npz), K;
  - pch1000 / pch100: polar-cap (65-90N) cos-weighted mean height, m;
  - gb_raw: z500 box 60-80N 290-340E on the 3-degree subgrid, m (the Greenland block);
  - nao_raw: sea-level pressure box 33-45N 330-357E minus box 57-70N 320-345E on the 3-degree subgrid, hPa;
  - u10_60n: zonal-mean 10 hPa zonal wind at 60N, m/s (the SSW test, Charlton & Polvani 2007).
A day is the mean of its four analyses; a day with fewer than four is written with n_cycles < 4 and not used.

    python scripts/coldindex/gfs_daily.py --start 2025-11-01 --end 2026-03-31 --out gfs_daily.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p50.f000"
WANT = {"TMP:2 m above ground": "t2m", "HGT:1000 mb": "z1000", "HGT:500 mb": "z500", "HGT:100 mb": "z100",
        "PRMSL:mean sea level": "msl", "UGRD:10 mb": "u10"}
COLS = ["t2m_us", "pch1000", "pch100", "gb_raw", "nao_raw", "u10_60n"]
_POP = np.load(HERE / "reference" / "pop_us_1p5.npz")
LAT15, LON15, POP = _POP["lat"].astype(float), _POP["lon"].astype(float), _POP["pop"].astype(float)   # 0..90N, 0..358.5E
LAT3, LON3 = LAT15[::2], LON15[::2]


def _get(url: str, rng: tuple[int, int | str] | None = None, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scorvec.com cold-risk index (research)"})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(5 * (k + 1))
        except Exception:                                                # noqa: BLE001 - network hiccups: retry
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def cycle_fields(cycle: dt.datetime) -> dict | None:
    """The six fields of one analysis as (361, 720) arrays, north->south, 0E first; None if the cycle isn't published."""
    import eccodes
    url = S3.format(d=f"{cycle:%Y%m%d}", h=f"{cycle:%H}")
    try:
        idx = _get(url + ".idx").decode().splitlines()
    except urllib.error.HTTPError:
        return None
    ent = [l.split(":") for l in idx]
    out = {}
    for i, e in enumerate(ent):
        key = f"{e[3]}:{e[4]}"
        if key in WANT and WANT[key] not in out:
            end = int(ent[i + 1][1]) - 1 if i + 1 < len(ent) else ""
            msg = _get(url, (int(e[1]), end))
            h = eccodes.codes_new_from_message(msg)
            try:
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                v = eccodes.codes_get_values(h).reshape(nj, ni)
            finally:
                eccodes.codes_release(h)
            assert (nj, ni) == (361, 720), (nj, ni)
            out[WANT[key]] = v if lat0 > 0 else v[::-1]
    return out if len(out) == len(WANT) else None


def to_1p5(f: np.ndarray) -> np.ndarray:
    """(361, 720) 0.5-degree, 90N->90S -> (61, 240) 1.5-degree cells centred 0..90N, 0..358.5E (3x3 block means)."""
    lat05 = 90.0 - 0.5 * np.arange(361)
    rows = []
    for la in LAT15:
        j = [int(round((90.0 - x) / 0.5)) for x in (la - 0.5, la, la + 0.5) if x <= 90.0]
        rows.append(f[j].mean(0))
    r = np.array(rows)                                                  # (61, 720)
    r = np.concatenate([r[:, -1:], r, r[:, :1]], axis=1)                # cyclic: centre k*1.5 deg = column 3k (+1 pad)
    c = 3 * np.arange(240) + 1
    return (r[:, c - 1] + r[:, c] + r[:, c + 1]) / 3.0


def box(f3: np.ndarray, la0, la1, lo0, lo1) -> float:
    r = (LAT3 >= la0) & (LAT3 <= la1); c = (LON3 >= lo0) & (LON3 <= lo1)
    w = np.cos(np.deg2rad(LAT3[r]))[:, None] * np.ones((1, c.sum()))
    return float((f3[r][:, c] * w).sum() / w.sum())


def indices(F: dict) -> dict:
    t2 = to_1p5(F["t2m"]); z1 = to_1p5(F["z1000"]); z5 = to_1p5(F["z500"]); zh = to_1p5(F["z100"]); sl = to_1p5(F["msl"]) / 100.0
    cap = LAT15 >= 65; wc = np.cos(np.deg2rad(LAT15[cap]))[:, None]
    pc = lambda z: float((z[cap] * wc).sum() / (wc.sum() * z.shape[1]))
    z5_3, sl_3 = z5[::2, ::2], sl[::2, ::2]
    j60 = int(round((90.0 - 60.0) / 0.5))
    return {"t2m_us": float((t2 * POP).sum() / POP.sum()), "pch1000": pc(z1), "pch100": pc(zh),
            "gb_raw": box(z5_3, 60, 80, 290, 340), "nao_raw": box(sl_3, 33, 45, 330, 357) - box(sl_3, 57, 70, 320, 345),
            "u10_60n": float(F["u10"][j60].mean())}


def day(d: dt.date) -> dict:
    cyc = [dt.datetime(d.year, d.month, d.day, h) for h in (0, 6, 12, 18)]
    with ThreadPoolExecutor(4) as ex:
        got = [g for g in ex.map(cycle_fields, cyc) if g is not None]
    rows = [indices(g) for g in got]
    out = {"date": d.isoformat(), "n_cycles": len(rows)}
    for c in COLS:
        out[c] = round(float(np.mean([r[c] for r in rows])), 4) if rows else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d0, d1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(day, days))
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "n_cycles"] + COLS); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} days -> {a.out}; complete: {sum(r['n_cycles'] == 4 for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
