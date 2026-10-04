#!/usr/bin/env python3
"""Build the lead-dependent MODEL climatology from the ingested GEPS8 hindcast.

This is the thing the whole subseasonal page depends on. A subseasonal anomaly
must be taken against the model's *own* behaviour at that lead, not against a
reanalysis: GEPS drifts with lead, and differencing against ERA5/MERRA-2 would
fold that drift straight into what looks like signal. (Exactly the trap the
polar-cap z100 trend turned out to be -- see scorvec.github.io
scripts/strat/nh_vortex.py.)

For each (variable, lead, day-of-year, lat, lon) it averages the hindcast starts
whose day-of-year falls within +/- HALFWIN days, across all 20 years. Starts run
4x/week, so a +/-15 day window pools roughly 20 yr * 4/wk * 30 d/7 ~ 340 starts
per cell -- enough to be stable without smearing the seasonal cycle.

The output carries `n_starts` alongside the mean so downstream code can refuse
to use thin cells rather than silently trusting them.

    python build_clim.py --tag zg500
    python build_clim.py --all                 # every complete variable
    python build_clim.py --tag olr --halfwin 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LOCAL_STORE as ARCHIVE                      # noqa: E402  laptop only
STORE = ARCHIVE / "geps8_hindcast"
CLIM = ARCHIVE / "climatology"
MANIFEST = ARCHIVE / "manifest.csv"
HALFWIN = 15                 # +/- days of year pooled into each climatological day
DOY_STEP = 5                 # climatology evaluated every 5 days of year, interpolated later
EXPECTED_MONTHS = 240        # 2001-2020


def complete_tags():
    if not MANIFEST.exists():
        return []
    m = pd.read_csv(MANIFEST)
    out = []
    for tag, g in m.groupby("tag"):
        if len(g) >= EXPECTED_MONTHS:
            out.append(tag)
        else:
            print(f"  {tag}: {len(g)}/{EXPECTED_MONTHS} months — not complete, skipping")
    return out


def build(tag: str, halfwin: int) -> Path | None:
    """One pass over the monthly files, accumulating into day-of-year bins.

    The first version built one lazy mean per day-of-year window and let
    to_netcdf compute them all, which re-read the entire ~13 GB variable once
    per window — 74 passes. This walks the files once and adds each start into
    every bin whose +/-halfwin window covers it, so the cost is one read of the
    archive and a bounded accumulator (74 x 39 x 181 x 360 float32 ~ 750 MB).
    """
    fs = sorted(STORE.glob(f"geps8_{tag}_*.nc"))
    if not fs:
        print(f"{tag}: nothing ingested"); return None
    centres = np.arange(1, 367, DOY_STEP)
    acc = cnt = None
    name = None
    nstart = np.zeros(len(centres), dtype=int)
    t0 = time.time()
    for k, f in enumerate(fs):
        d = xr.open_dataset(f, decode_timedelta=False)
        if name is None:
            name = [v for v in d.data_vars][0]
        a = d[name].astype("float32")
        # zg/ua/va keep a singleton P dimension from the level selection, so the
        # first axis is NOT always S. Drop singletons and order the dims explicitly.
        a = a.squeeze(drop=True)
        a = a.transpose("S", *[x for x in a.dims if x != "S"])
        v = a.values                                    # (S, L, Y, X)
        doy = pd.DatetimeIndex(pd.to_datetime(d.S.values)).dayofyear.values
        if acc is None:
            acc = np.zeros((len(centres),) + v.shape[1:], dtype="float64")
            cnt = np.zeros(len(centres), dtype="int64")
        for bi, c in enumerate(centres):
            dist = np.minimum(np.abs(doy - c), 365 - np.abs(doy - c))
            sel = np.where(dist <= halfwin)[0]
            if sel.size:
                acc[bi] += v[sel].sum(0)
                cnt[bi] += sel.size
                nstart[bi] += sel.size
        d.close()
        if k % 40 == 0:
            print(f"   {k}/{len(fs)} files, {time.time()-t0:.0f} s", flush=True)
    mean = (acc / np.maximum(cnt, 1)[:, None, None, None]).astype("float32")
    mean[cnt == 0] = np.nan

    ref = xr.open_dataset(fs[0], decode_timedelta=False)
    a0 = ref[name].squeeze(drop=True)
    a0 = a0.transpose("S", *[x for x in a0.dims if x != "S"])
    out = xr.DataArray(mean, dims=("doy",) + a0.dims[1:],
                       coords={"doy": centres, **{k: a0[k].values for k in a0.dims[1:]}})
    ds = out.to_dataset(name=name)
    ds["n_starts"] = ("doy", nstart)
    ds[name].attrs = dict(a0.attrs)
    ds.attrs = {
        "title": f"GEPS8 hindcast model climatology — {tag}",
        "summary": ("Lead-dependent model climatology: mean over hindcast starts whose "
                    f"day-of-year lies within +/-{halfwin} days of each centre, 2001-2020. "
                    "Subtract from a live GEPS forecast at the SAME lead for an anomaly "
                    "free of the model's own drift."),
        "source": "IRI Data Library, SOURCES/.Models/.SubC/.ECCC/.GEPS8/.hindcast",
        "halfwin_days": halfwin, "doy_step": DOY_STEP, "n_months_used": len(fs),
        "caution": ("Anomalies must use the same lead L and the same member treatment "
                    "(ensemble mean) as this climatology."),
    }
    CLIM.mkdir(parents=True, exist_ok=True)
    dst = CLIM / f"geps8_clim_{tag}.nc"
    lo, hi = float(np.nanmin(mean)), float(np.nanmax(mean))
    enc = {name: dict(dtype="int16", zlib=True, complevel=6, shuffle=True,
                      scale_factor=max((hi - lo) / 60000.0, 1e-12),
                      add_offset=0.5 * (lo + hi), _FillValue=-32767)}
    ds.to_netcdf(dst, encoding=enc)
    print(f"  wrote {dst.name} ({dst.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f} s; "
          f"{nstart.min()}-{nstart.max()} starts per doy bin")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--halfwin", type=int, default=HALFWIN)
    a = ap.parse_args()
    tags = complete_tags() if a.all else ([a.tag] if a.tag else [])
    if not tags:
        print("nothing to build (use --tag or --all)"); return 1
    for t in tags:
        build(t, a.halfwin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
