#!/usr/bin/env python3
"""Daily OISST v2.1 over 25S-25N on ERSSTv6's 2° cells, 1982 on: the record of the Niño-3.4 model's fresh state.

On the 15th ERSSTv6 ends with the month before; OISST already holds the first two weeks of the issue month.
enso_fc.py adds the OISST anomaly over the 30 days to the 14th to its state for the forecasts to six months
(oisst_state), trained on this record.

Each year: PSL's sst.day.mean.<YYYY>.nc (scripts/sst/oisst9120.ensure_mean: PSL with the NCEI fallback and top-up),
box-averaged (cos-weighted, at least half the 0.25° cells ocean) onto the 2° cells centred on ERSSTv6's points
(24S..24N, 0..358E: each cell 8 x 8 OISST cells), absolute SST packed as int16 hundredths of a degree:

    oisst2_<YYYY>.npz   dates (str, one per day), sst (days, 25, 180) int16 (FILL where not ocean), lat, lon

The yearly raw file is deleted once reduced (OISST_KEEP_RAW=1 keeps it). Downloads run in Actions: the record is
kept in the release `oisst-daily-v1` (built whole by oisst-daily.yml), and each outlook issue (monthly-outlook.yml)
rebuilds this year's table, and last year's through February (its last days are preliminary for about two weeks) or
while it stops short of 31 December, and uploads them.

    python scripts/monthly/oisst_daily.py --years 1982-2025 --out DIR
    python scripts/monthly/oisst_daily.py --refresh --out DIR          # this year (and last year, as above)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "sst"))

LAT = np.arange(-24.0, 25.0, 2.0)  # ERSSTv6 points 24S..24N
LON = np.arange(0.0, 360.0, 2.0)
FILL = np.int16(-32768)
SCALE = 0.01


def cells(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For OISST's 0.25° centres: the 2° cell row and column of each (-1 outside 25S-25N) and the cos weight."""
    i = np.floor((lat + 25.0) / 2.0).astype(int)
    i[(lat < -25.0) | (lat > 25.0)] = -1
    j = np.floor(((lon + 1.0) % 360.0) / 2.0).astype(int)  # cell c spans c-1 .. c+1 degrees east
    return i, j, np.cos(np.deg2rad(lat))


def reduce_year(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(dates, (days, 25, 180) float32 °C) box means of one yearly file, NaN where under half the cell is ocean."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        da = ds["sst"].sel(lat=slice(-25.0, 25.0))
        lat, lon = da["lat"].values.astype(float), da["lon"].values.astype(float)
        dates = np.array([str(t)[:10] for t in da["time"].values])
        x = da.values.astype(np.float32)  # (days, lat, lon), NaN over land
    i, j, w = cells(lat, lon)
    keep = i >= 0
    x, i, w = x[:, keep], i[keep], w[keep]
    nlon = len(LON)
    # sum over the 8 longitudes of each cell, then over its rows: weights cos(lat) x ocean
    ok = np.isfinite(x)
    xw = np.where(ok, x, 0.0) * w[None, :, None]
    ww = ok * w[None, :, None]
    sx = np.zeros((len(dates), len(LAT), nlon), np.float64)
    sw = np.zeros_like(sx)
    full = np.zeros((len(LAT), nlon))
    for jj in range(nlon):
        cols = j == jj
        cx, cw = xw[:, :, cols].sum(2), ww[:, :, cols].sum(2)  # (days, rows)
        for ii in range(len(LAT)):
            rows = i == ii
            sx[:, ii, jj] = cx[:, rows].sum(1)
            sw[:, ii, jj] = cw[:, rows].sum(1)
            full[ii, jj] = w[rows].sum() * cols.sum()
    out = np.where(sw >= 0.5 * full[None], sx / np.maximum(sw, 1e-12), np.nan)
    return dates, out.astype(np.float32)


def save(year: int, dates: np.ndarray, sst: np.ndarray, out: Path) -> Path:
    q = np.where(np.isfinite(sst), np.round(sst / SCALE), FILL).astype(np.int16)
    p = out / f"oisst2_{year}.npz"
    np.savez_compressed(p, dates=dates, sst=q, lat=LAT, lon=LON)
    return p


def load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """(dates, (days, 25, 180) float32 °C) from oisst2_<YYYY>.npz files, in date order."""
    d, x = [], []
    for p in sorted(paths):
        with np.load(p) as z:
            d.append(z["dates"])
            q = z["sst"]
            x.append(np.where(q == FILL, np.nan, q.astype(np.float32) * SCALE))
    return np.concatenate(d), np.concatenate(x)


def complete(path: Path, year: int) -> bool:
    """Whether the table at `path` holds `year` to 31 December."""
    if not path.exists():
        return False
    with np.load(path) as z:
        return str(z["dates"][-1]) == f"{year}-12-31"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--years", help="YYYY or YYYY-YYYY")
    g.add_argument("--refresh", action="store_true", help="this year, and last year through February or until whole")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.refresh:
        today = dt.datetime.now(dt.UTC).date()
        last = today.year - 1
        years = [*([last] if today.month <= 2 or not complete(a.out / f"oisst2_{last}.npz", last) else []), today.year]
    else:
        y0, _, y1 = a.years.partition("-")
        years = range(int(y0), int(y1 or y0) + 1)
    import oisst9120

    for y in years:
        raw = oisst9120.ensure_mean(y)
        dates, sst = reduce_year(raw)
        p = save(y, dates, sst, a.out)
        if not os.environ.get("OISST_KEEP_RAW"):
            raw.unlink()
        n_ocean = int(np.isfinite(sst[0]).sum())
        print(f"{p.name}: {dates[0]}..{dates[-1]} ({len(dates)} days), {n_ocean} ocean cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
