#!/usr/bin/env python3
"""Real-time GMGSI OLR tail for the wave decomposition — global, daily, rolling window.

The Wheeler-Kiladis filter (olr_waves.py) needs a continuous equatorial OLR record up to
today on the FULL global longitude circle. interp_OLR (the historical backbone) stops in
2022, so this maintains the recent tail from the GMGSI longwave-IR mosaic: per day it
averages a few hours' McIDAS-Tb → Ohring-Gruber OLR, meridional-means the 15°S-15°N and
5°S-5°N bands, bins to interp_OLR's 2.5° global grid, and appends to a rolling store. A
constant proxy-vs-true-OLR offset is irrelevant to the filtered waves (the bands exclude the
f=0 time-mean), so no explicit bias-correction is needed — only deseasonalising, which
olr_waves does with the interp_OLR climatology.

    python scripts/sst/build_olr_realtime.py --bootstrap 330   # backfill N days, then daily
    python scripts/sst/build_olr_realtime.py                   # append today/yesterday
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "scripts") / "lib"))
from webget import get  # noqa: E402

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
STORE = HERE / "metar" / "olr_rt.nc"            # committed (rolling state for the daily Action)
S3 = "https://noaa-gmgsi-pds.s3.amazonaws.com"
SIGMA = 5.67e-8
HOURS = (0, 12)                          # day+night sample averaged into each daily mean
KEEP_DAYS = 550                          # ~1.5 yr — margin for the 120-day low-pass + longer display
BANDS = {"15": 15.0, "05": 5.0}
GRID = np.arange(0, 360, 2.5)            # interp_OLR longitude centres (144 @ 2.5°)


def _gmgsi_day(dd) -> dict | None:
    """Daily-mean synthetic OLR(lon) on the 2.5° global grid for each band, or None."""
    per = {b: [] for b in BANDS}
    for h in HOURS:
        dt = datetime(dd.year, dd.month, dd.day, h, tzinfo=timezone.utc)
        pre = f"GMGSI_LW/{dt:%Y/%m/%d/%H}/"
        try:
            xml = get(f"{S3}/?list-type=2&prefix={pre}&max-keys=5", timeout=30).decode()
            keys = re.findall(r"<Key>([^<]+)</Key>", xml)
            if not keys:
                continue
            with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
                tf.write(get(f"{S3}/{keys[0]}", timeout=90)); tf.flush()
                d = xr.open_dataset(tf.name)
                b = d["data"].isel(time=0).astype("float32")
                lat = d["lat"]; lon = d["lon"]
                Tb = xr.where(b > 176, 418.0 - b, (660.0 - b) / 2.0)
                lon1d = (lon.isel(yc=lon.sizes["yc"] // 2).values % 360)
                idx = (np.round(lon1d / 2.5).astype(int)) % len(GRID)
                for bd, half in BANDS.items():
                    band = Tb.where((lat >= -half) & (lat <= half)).mean("yc")
                    Tf = band * (1.228 - 1.106e-3 * band)
                    olr = (SIGMA * Tf ** 4).values
                    good = np.isfinite(olr)
                    sums = np.bincount(idx[good], weights=olr[good], minlength=len(GRID))
                    cnts = np.bincount(idx[good], minlength=len(GRID))
                    per[bd].append(np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan))
        except Exception:                                       # noqa: BLE001
            continue
    if not per["15"]:
        return None
    return {bd: np.nanmean(per[bd], axis=0) for bd in BANDS}


def load() -> xr.Dataset | None:
    if STORE.exists():
        return xr.open_dataset(STORE)
    return None


def _write(rows: dict, today) -> xr.Dataset:
    """Atomically (re)write the rolling store from the accumulated rows."""
    cutoff = today - timedelta(days=KEEP_DAYS)
    times = sorted(t for t in rows["15"] if t > cutoff)
    out = xr.Dataset(
        {f"olr_{b}": (("time", "lon"), np.array([rows[b][t] for t in times], dtype="float32"))
         for b in BANDS},
        coords={"time": [np.datetime64(t, "D") for t in times], "lon": GRID},
    )
    tmp = STORE.with_suffix(".nc.tmp")
    out.to_netcdf(tmp); tmp.replace(STORE)                      # atomic: never lose the prior store
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0, help="backfill N days before today")
    args = ap.parse_args(argv)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    ndays = max(args.bootstrap, 2)
    dates = [today - timedelta(days=k) for k in range(ndays - 1, -1, -1)]

    ds = load()
    have = set(pd.to_datetime(ds["time"].values).date) if ds is not None else set()
    rows = {b: {} for b in BANDS}
    if ds is not None:                                          # carry existing store rows
        for b in BANDS:
            for t, v in zip(pd.to_datetime(ds["time"].values).date, ds[f"olr_{b}"].values):
                rows[b][t] = v
    n_new = 0
    for dd in dates:
        if dd in have:                                         # resume: skip days already stored
            continue
        r = _gmgsi_day(dd)
        if r is None:
            print(f"  {dd}: no GMGSI", flush=True); continue
        for b in BANDS:
            rows[b][dd] = r[b]
        n_new += 1
        print(f"  {dd}: OLR15 min {np.nanmin(r['15']):.0f} W/m²", flush=True)
        if n_new % 30 == 0:                                    # checkpoint so a crash can't undo progress
            _write(rows, today); print(f"  …checkpoint ({n_new} new days)", flush=True)

    out = _write(rows, today)
    print(f"wrote {STORE.name}: {out.sizes['time']} days × {out.sizes['lon']} lon "
          f"({str(out['time'].values[0])[:10]}→{str(out['time'].values[-1])[:10]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
