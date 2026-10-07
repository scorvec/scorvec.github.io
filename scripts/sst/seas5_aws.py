#!/usr/bin/env python3
"""SEAS5 lead month 7 hindcast from Planette's anonymous AWS copy of the C3S SEAS5 daily archive.

WHY. The CDS monthly product stops at leadtime_month 6, so the SEAS5 page's 7th month (April for an October
start) needs the daily data. For the forecast that is the CDS's own sub-daily product (seas5_lead7.py); for
the 1993–2016 hindcast reference the CDS would cost tens of GB of 6-hourly fields per start month, so the
instantaneous variables come from s3://planette-c3s-seasonal-forecasts/seas5_ic/hindcast (us-east-2,
anonymous, Copernicus licence; written by icechunk 2.2.2, which needs Python >= 3.12). It is a third-party
copy of C3S data, not ECMWF open data, so the Google-only ECMWF rule does not apply (user, 2026-09-26).

    python seas5_aws.py probe                         # layout, variables, coordinates, throughput (Actions)

Runs in GitHub Actions only (seas5-lead7.yml); the laptop never downloads (seas5_ref.guard_fetch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUCKET, REGION = "planette-c3s-seasonal-forecasts", "us-east-2"


def _guard(what: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true" and os.environ.get("SEAS5_ALLOW_FETCH") != "1":
        raise RuntimeError(f"download refused ({what}): SEAS5 data is fetched in GitHub Actions only")


def open_root(period: str = "hindcast"):
    _guard(f"AWS SEAS5 {period}")
    import icechunk as ic
    import zarr
    st = ic.s3_storage(bucket=BUCKET, prefix=f"seas5_ic/{period}", region=REGION, anonymous=True)
    sess = ic.Repository.open(st).readonly_session("main")
    return zarr.open_group(sess.store, mode="r"), sess


def _coord_summary(g, name):
    try:
        a = g[name]
    except KeyError:
        return None
    v = a[:]
    out = dict(n=int(v.size), first=v.ravel()[0].item() if v.size else None, last=v.ravel()[-1].item() if v.size else None,
               attrs={k: (str(x)[:200]) for k, x in a.attrs.items()})
    if v.size <= 40:
        out["values"] = [x.item() for x in v.ravel()]
    return out


def probe() -> None:
    import pandas as pd
    for period in ("hindcast", "forecast"):
        root, sess = open_root(period)
        print(f"===== {period}: groups {list(root.group_keys())}", flush=True)
        for gname in ("single/1platx1plon", "pressure/1platx1plon"):
            try:
                g = root[gname]
            except KeyError:
                print(f"  {gname}: absent", flush=True); continue
            print(f"--- {period}/{gname}: attrs {json.dumps({k: str(v)[:300] for k, v in g.attrs.items()})}", flush=True)
            for name, a in g.arrays():
                print(f"  {name:28s} shape={a.shape} chunks={a.chunks} dtype={a.dtype} dims={a.attrs.get('_ARRAY_DIMENSIONS', a.metadata.dimension_names if hasattr(a.metadata, 'dimension_names') else '?')} "
                      f"attrs={json.dumps({k: str(v)[:160] for k, v in a.attrs.items() if k != '_ARRAY_DIMENSIONS'})}", flush=True)
            for c in ("init_time", "lead", "member", "number", "lat", "lon", "latitude", "longitude", "pressure_level", "valid_time"):
                s = _coord_summary(g, c)
                if s is not None:
                    print(f"  coord {c}: {json.dumps(s, default=str)[:900]}", flush=True)
            try:
                it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
                octs = [str(t.date()) for t in it if t.month == 10]
                print(f"  init_time {it[0].date()} .. {it[-1].date()} ({len(it)}); October starts: {octs[:3]} .. {octs[-3:]}", flush=True)
            except Exception as e:                                          # noqa: BLE001
                print(f"  init_time decode failed: {e}", flush=True)
        # throughput + definition check on one chunk: t2m (or the first 4-D+ variable) of the 2000-10 start
        try:
            g = root["single/1platx1plon"]
            it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
            k = [i for i, t in enumerate(it) if t.year == (2000 if period == "hindcast" else 2024) and t.month == 10][0]
            for var in [v for v in ("t2m", "2t", "tas", "sst", "pr", "tp", "msl") if v in [n for n, _ in g.arrays()]][:3]:
                a = g[var]
                t0 = time.time()
                blk = a[0:10, k, 0:46] if a.ndim == 5 else a[0:10, k, 0:46, 0]
                dt = time.time() - t0
                print(f"  read {period} {var} [10 members, init {it[k].date()}, 46 days] {blk.shape} {blk.dtype} in {dt:.1f} s "
                      f"({blk.nbytes / 1e6:.0f} MB raw) min={np.nanmin(blk):.4g} max={np.nanmax(blk):.4g} mean={np.nanmean(blk):.4g} "
                      f"nan={np.isnan(blk).mean():.3f}", flush=True)
                lat = g["lat"][:]; lon = g["lon"][:]
                i = int(np.argmin(np.abs(lat - 40.5))); j = int(np.argmin(np.abs((lon % 360) - 255.5)))
                print(f"    member 0 at ({lat[i]}, {lon[j]}) days 0..6: {np.round(blk[0, :7, i, j], 4).tolist()}", flush=True)
        except Exception as e:                                              # noqa: BLE001
            print(f"  read test failed: {e}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe"])
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        probe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
