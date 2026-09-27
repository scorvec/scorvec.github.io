#!/usr/bin/env python3
"""100 hPa eddy heat flux [v'T'] from the EXTENDED-range ensembles, GEPS and GEFS, days 1-35 (2026-09-27; user: "On the
stratosphere page, can we add the geps and gefs wave-1/wave-2 forcing and eddy heat flux?").

The same product as heatflux100.py (AIFS-ENS, days 1-15) - the same flux (per member, then averaged; waves 1..72; wave-1
and wave-2 kept apart; positive = poleward), the same 45-75 deg band, the same NCEP R1 1991-2020 climatology and the same
analysed tail (the AIFS control's step-0 history, assets/sst/data/heatflux100_history.json) - so the three models read
against one reference. One instant per day, the 00Z field at every 24 h step, as the AIFS product uses.

  geps  ECCC GEPS 00Z extended run (Mon/Thu), 20 perturbed + control, 0.5 deg, the `allmbrs` VGRD/TMP 100 hPa files
        the GEPS vortex products already cache (strat_maps.fetch). Laptop, in run_geps.sh.
  gefs  NOAA GEFS 00Z run, gec00 + gep01-30, pgrb2a 0.5 deg, VGRD/TMP at 100 mb byte-ranged from NOAA's S3 bucket via
        the .idx files (the site rule for NOAA models). Actions, in gefs.yml.

    python heatflux100_ext.py --model geps --date 20260924 --out-dir assets/geps
    python heatflux100_ext.py --model gefs --date 20260926 --out-dir assets/gefs
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import heatflux100 as HF                                              # noqa: E402

LEADS = list(range(24, 841, 24))
LAT_H = np.arange(90.0, -90.0 - 1e-6, -0.5)
LON_H = np.arange(0.0, 360.0 - 1e-6, 0.5)


def as_da(arr):
    """(member, step, lat, lon) numpy -> the DataArray shape heatflux100.fluxes expects."""
    return xr.DataArray(arr, dims=("number", "step", "latitude", "longitude"),
                        coords={"latitude": LAT_H, "longitude": LON_H})


def geps(date: str):
    """(v, t, control index) from the cached GEPS allmbrs files; control = the cf member."""
    import strat_maps as sm
    V, T = [], []
    for L in LEADS:
        per = {}
        for var in ("VGRD", "TMP"):
            f = sm.fetch(date, "00", var, 100, L)
            if f is None:
                raise SystemExit(f"GEPS {var} 100 hPa step {L} unavailable")
            parts = []
            for dt in ("cf", "pf"):                                   # control first, so it is member 0
                ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs=dict(filter_by_keys={"dataType": dt}, indexpath=""))
                a = ds[list(ds.data_vars)[0]]
                a = a.expand_dims("number") if "number" not in a.dims else a
                a = a.transpose("number", "latitude", "longitude")
                if a.latitude.values[0] < a.latitude.values[-1]:
                    a = a.isel(latitude=slice(None, None, -1))
                a = a.assign_coords(longitude=a.longitude % 360).sortby("longitude")
                parts.append(a.values)
            per[var] = np.concatenate(parts, axis=0)
        V.append(per["VGRD"]); T.append(per["TMP"])
    return np.stack(V, axis=1), np.stack(T, axis=1)                   # (member, step, lat, lon)


def gefs(date: str, workers: int = 32):
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_reforecast as R
    S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    jobs = []
    for mi, m in enumerate(mems):
        for li, L in enumerate(LEADS):
            jobs.append((mi, li, f"{S3}/gefs.{date}/00/atmos/pgrb2ap5/{m}.t00z.pgrb2a.0p50.f{L:03d}", L))

    def one(j):
        mi, li, url, L = j
        idx = R.index(url)
        out = {}
        for name in ("VGRD", "TMP"):
            hit = [e for e in idx if e[4] == name and e[2] == "100 mb" and e[3] == f"{L} hour fcst"]
            if not hit:
                return mi, li, None
            out[name] = R.decode(R.get(url, (hit[0][0], hit[0][1])))
        return mi, li, out
    V = np.full((len(mems), len(LEADS), len(LAT_H), len(LON_H)), np.nan, "float32"); T = V.copy()
    with ThreadPoolExecutor(workers) as ex:
        for mi, li, o in ex.map(one, jobs):
            if o is not None:
                V[mi, li], T[mi, li] = o["VGRD"], o["TMP"]
    ok = np.isfinite(V).all(axis=(1, 2, 3)) & np.isfinite(T).all(axis=(1, 2, 3))
    if ok.sum() < 16 or not ok[0]:
        raise SystemExit(f"only {int(ok.sum())} complete GEFS members (control {'ok' if ok[0] else 'missing'})")
    return V[ok], T[ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("geps", "gefs")); ap.add_argument("--date", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--history", default=str(REPO / "assets" / "sst" / "data" / "heatflux100_history.json"))
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]} 00:00")
    v, t = geps(a.date) if a.model == "geps" else gefs(a.date)
    name = {"geps": "GEPS", "gefs": "GEFS"}[a.model]
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=L) for L in LEADS])
    lat, f = HF.fluxes(as_da(v), as_da(t))                            # {tot,k1,k2}: (member, step, lat)
    fm = {k: x[1:] for k, x in f.items()}                             # perturbed members
    fc = {k: x[:1] for k, x in f.items()}                             # control = member 0
    fall = {k: x for k, x in f.items()}                               # the ensemble = control + perturbed, as it is issued
    clim = xr.open_dataset(HF.CLIM).load() if HF.CLIM.exists() else None
    hist = HF.load_history(Path(a.history))
    out = Path(a.out_dir); out = out if out.is_absolute() else REPO / out
    (out / "data").mkdir(parents=True, exist_ok=True)
    n = int(v.shape[0])
    doc = {"model": name, "init": init.isoformat(), "members": n, "band": list(HF.BAND), "k_max": HF.K_MAX,
           "valid": [d.strftime("%Y-%m-%d %H:%M") for d in valid], "units": "K m s-1", "sign": "positive = poleward",
           "hemispheres": {}}
    for hemi, north in (("nh", True), ("sh", False)):
        mem = {k: HF.band_mean(lat, fall[k], north) for k in fall}
        ctrl = {k: HF.band_mean(lat, fc[k], north)[0] for k in fc}
        obs = HF.history_frame(hist, hemi)
        if obs is not None:
            obs = obs[obs.index <= init.normalize()]
            gaps = np.flatnonzero(np.diff(obs.index.values) > np.timedelta64(2, "D"))
            if len(gaps):
                obs = obs.iloc[gaps[-1] + 1:]
            obs = obs.asfreq("D").interpolate(limit=1).iloc[-(HF.OBS_DAYS + 40):]
            if len(obs) < 5:
                obs = None
        z_last = HF.render(hemi, north, valid, mem, ctrl, lat, fall["tot"].mean(0), obs, clim, init, n,
                           out / f"{a.model}_heatflux100_{hemi}.webp", model=f"{name} extended")
        h = {k: {"mean": np.round(mem[k].mean(0), 2).tolist(), "p10": np.round(np.percentile(mem[k], 10, 0), 2).tolist(),
                 "p90": np.round(np.percentile(mem[k], 90, 0), 2).tolist(), "control": np.round(ctrl[k], 2).tolist()} for k in mem}
        if clim is not None:
            h["clim_mean"] = np.round(HF.clim_at(clim, hemi, "tot", "mean", valid), 2).tolist()
            h["k1_share_d6_35"] = round(float(mem["k1"].mean(0)[5:].mean() / max(1e-6, mem["tot"].mean(0)[5:].mean())), 2)
        h["z40_end_of_forecast"] = None if z_last is None else round(z_last, 2)
        doc["hemispheres"][hemi] = h
        print(f"  {name} {hemi}: mean flux d1-7 {mem['tot'].mean(0)[:7].mean():+.1f}, d8-35 {mem['tot'].mean(0)[7:].mean():+.1f} K m/s; "
              f"wave-2 d8-35 {mem['k2'].mean(0)[7:].mean():+.2f}" + (f"; 40-d standardised at day 35 {z_last:+.2f}" if z_last is not None else ""),
              flush=True)
    (out / "data" / f"{a.model}_heatflux100.json").write_text(json.dumps(doc, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
