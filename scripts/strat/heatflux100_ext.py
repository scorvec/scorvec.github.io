#!/usr/bin/env python3
"""100 hPa eddy heat flux [v'T'] from the other ensembles: GEPS and GEFS to day 35 (2026-09-27; user: "On the
stratosphere page, can we add the geps and gefs wave-1/wave-2 forcing and eddy heat flux?"), and IFS-ENS to day 15
(2026-09-27; user: "We also add in the ifs-ens for all strat products").

The same product as heatflux100.py (AIFS-ENS, days 1-15) - the same flux (per member, then averaged; waves 1..72; wave-1
and wave-2 kept apart; positive = poleward), the same 45-75 deg band, the same NCEP R1 1991-2020 climatology and the same
analysed tail (the AIFS control's step-0 history, assets/sst/data/heatflux100_history.json) - so the three models read
against one reference. One instant per day, the 00Z field at every 24 h step, as the AIFS product uses.

  geps  ECCC GEPS 00Z extended run (Mon/Thu), 20 perturbed + control, 0.5 deg, the `allmbrs` VGRD/TMP 100 hPa files
        the GEPS vortex products already cache (strat_maps.fetch). Laptop, in run_geps.sh.
  gefs  NOAA GEFS 00Z run, gec00 + gep01-30, pgrb2a 0.5 deg, VGRD/TMP at 100 mb byte-ranged from NOAA's S3 bucket via
        the .idx files (the site rule for NOAA models). Actions, in gefs.yml.
  ifs   ECMWF IFS-ENS 00/12Z, ALL 50 perturbed members (open data has no IFS control at pressure levels), days 0-15,
        v at 100 hPa from the shared IFS-ENS u/v file and t at 100 hPa (ifs_ens.py, Google Cloud mirror only), taken to
        0.5 deg like GEPS and GEFS - wavenumbers 1-72 need no finer grid. Actions, in strat-ifs.yml.

    python heatflux100_ext.py --model geps --date 20260924 --out-dir assets/geps
    python heatflux100_ext.py --model gefs --date 20260926 --out-dir assets/gefs
    python heatflux100_ext.py --model ifs --date 20260927 --time 00 --out-dir assets/sst
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
IFS_LEADS = list(range(0, 361, 24))                                  # as the AIFS product: analysis + days 1..15
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


def ifs(date: str, time: str):
    """(v, t) for all 50 IFS-ENS perturbed members, daily 0..360 h, at 0.5 deg on LAT_H x LON_H (every other 0.25 deg
    point; the open-data grid starts at -180, so it is rolled to start at 0)."""
    import ifs_ens as IE
    if not IE.published(date, time):
        raise SystemExit(f"IFS-ENS {date} {time}Z is not on the Google mirror yet")
    cyc = IE.E.Cycle(date, time)
    out = []
    for spec, short in ((IE.spec_uv(), "v"), (IE.spec_t100(), "t")):
        da = IE.open_field(IE.ensure(cyc, spec), short, 100)
        hrs = IE.step_hours(da)
        keep = [int(np.flatnonzero(hrs == L)[0]) for L in IFS_LEADS]
        lon = da.longitude.values[::2]
        roll = int(np.flatnonzero(np.isclose(lon % 360.0, 0.0))[0])
        arr = np.empty((da.sizes["number"], len(keep), len(LAT_H), len(LON_H)), "float32")
        for j, i in enumerate(keep):                                  # one step in memory at a time
            arr[:, j] = np.roll(da.isel(step=i).values[:, ::2, ::2], -roll, axis=-1)
        out.append(arr)
    return out[0], out[1]


def fluxes_chunked(v, t, n: int = 10):
    """HF.fluxes over member chunks: the same numbers, a tenth of the peak memory (50 members x 16 steps of 0.5 deg
    fields go through a float64 FFT)."""
    parts = [HF.fluxes(as_da(v[i:i + n]), as_da(t[i:i + n])) for i in range(0, len(v), n)]
    return parts[0][0], {k: np.concatenate([p[1][k] for p in parts], axis=0) for k in parts[0][1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("geps", "gefs", "ifs")); ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00", help="cycle hour (IFS-ENS: 00 or 12; GEPS/GEFS use their 00Z run)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--history", default=str(REPO / "assets" / "sst" / "data" / "heatflux100_history.json"))
    a = ap.parse_args()
    hour = int(a.time) if a.model == "ifs" else 0
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]} {hour:02d}:00")
    leads = IFS_LEADS if a.model == "ifs" else LEADS
    v, t = {"geps": lambda: geps(a.date), "gefs": lambda: gefs(a.date), "ifs": lambda: ifs(a.date, a.time)}[a.model]()
    name = {"geps": "GEPS", "gefs": "GEFS", "ifs": "IFS-ENS"}[a.model]
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=L) for L in leads])
    lat, f = fluxes_chunked(v, t)                                     # {tot,k1,k2}: (member, step, lat)
    has_cf = a.model != "ifs"                                         # IFS-ENS: 50 perturbed members, no control
    fc = {k: x[:1] for k, x in f.items()} if has_cf else None         # control = member 0
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
        ctrl = {k: HF.band_mean(lat, fc[k], north)[0] for k in fc} if fc is not None else None
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
                           out / f"{a.model}_heatflux100_{hemi}.webp", model=name if a.model == "ifs" else f"{name} extended")
        h = {k: {"mean": np.round(mem[k].mean(0), 2).tolist(), "p10": np.round(np.percentile(mem[k], 10, 0), 2).tolist(),
                 "p90": np.round(np.percentile(mem[k], 90, 0), 2).tolist(),
                 "control": None if ctrl is None else np.round(ctrl[k], 2).tolist()} for k in mem}
        d8 = 8 if a.model == "ifs" else 7                             # IFS leads start at the analysis (index 0)
        last = leads[-1] // 24
        if clim is not None:
            h["clim_mean"] = np.round(HF.clim_at(clim, hemi, "tot", "mean", valid), 2).tolist()
            h[f"k1_share_d6_{last}"] = round(float(mem["k1"].mean(0)[d8 - 2:].mean() / max(1e-6, mem["tot"].mean(0)[d8 - 2:].mean())), 2)
        h["z40_end_of_forecast"] = None if z_last is None else round(z_last, 2)
        doc["hemispheres"][hemi] = h
        print(f"  {name} {hemi}: mean flux d1-7 {mem['tot'].mean(0)[d8 - 7:d8].mean():+.1f}, d8-{last} {mem['tot'].mean(0)[d8:].mean():+.1f} K m/s; "
              f"wave-2 d8-{last} {mem['k2'].mean(0)[d8:].mean():+.2f}" + (f"; 40-d standardised at day {last} {z_last:+.2f}" if z_last is not None else ""),
              flush=True)
    (out / "data" / f"{a.model}_heatflux100.json").write_text(json.dumps(doc, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
