#!/usr/bin/env python3
"""FORECAST dripping paint: standardised polar-cap (65-90N) geopotential height from 1000 to 10 hPa against time - the
observed record of the last ~60 days, then the ensemble-mean forecast - for GEPS (day 35), GEFS (day 35) and
AIFS-ENS (day 15) (2026-09-27; user: "It might be nice to get zonal mean 'dripping paint' plots from the geps/gefs as
well. The geps is showing a legitimate pv displacement in october").

Same scale as the MERRA-2 history drips (strat_history.py): every value is (cap height - the MERRA-2 harmonics+trend
mean) / the MERRA-2 seasonal sd, from reference/drip_std.json (build_drip_ref.py, which reproduces the history
standardisation exactly).

Observed tail: GEOS FP analyses (geosfp_cap.py; same system as MERRA-2, daily mean of 00/06/12/18Z, the measured
GEOS FP - MERRA-2 offset removed per level, reference/geosfp_cap_offset.json).

Model drift - never shown as signal:
  GEPS     measured per level and lead on the ECMWF S2S GEPS reforecasts (geps_s2s_hindcast.py capdrift: 2001-2020,
           control + 3 members, start dates within 8 days of the init's day of year), as (reforecast cap - MERRA-2
           reference) - so it includes GEPS's analysis offset. Levels whose reforecast part has not arrived are drawn
           but HATCHED as uncorrected.
  GEFS     from the GEFS v2 reforecast daily climatology (gefs-clim-v2, 2000-2019): (model climate cap at that lead -
           MERRA-2 reference at the reforecast epoch). It holds 10, 100 and 500 hPa; the other levels are hatched.
  AIFS-ENS no reforecast exists, so the control's cap forecasts from recent 00Z cycles are verified against the GEOS FP
           analyses (plus the GEOS FP - MERRA-2 offset) and the mean error per level and lead is removed
           (--measure-aifs-drift -> reference/aifs_capdrift.json).
  IFS-ENS  no open reforecast either (ECMWF's are not open data), and no control at pressure levels on open data, so
           the same verification is made with the mean of all 50 perturbed members over the same cycles
           (--measure-drift ifs -> reference/ifs_capdrift.json). 50 members, 0.25 deg, day 15, from the shared
           IFS-ENS file (ifs_ens.py, Google Cloud mirror only).
  At any level without a measured drift, only the model's offset from GEOS FP at the analysis and day 1 is removed and
  the level is HATCHED. The 1000 hPa cap needs that offset most: models extrapolate heights below ground where GEOS FP
  and MERRA-2 leave those points out (in summer whole rows near the pole are underground and the observed 1000 hPa cap
  is undefined), so a model's 1000 hPa cap reads ~30-40 m (about 1 sd) low at analysis time.
Ensemble mean of the members' standardised anomalies; the lower panels are the 1000 hPa cap (an AO proxy: high cap =
negative AO) and the share of members with the 100 hPa cap >= +1 sd.

    python forecast_drip.py --model geps --date 20260924 --out-dir assets/geps
    python forecast_drip.py --model gefs --date 20260926 --out-dir assets/gefs
    python forecast_drip.py --model aifs --date 20260927 --time 00 --out-dir assets/sst
    python forecast_drip.py --model ifs --date 20260927 --time 00 --out-dir assets/sst
    python forecast_drip.py --measure-drift ifs          # laptop, once: -> reference/ifs_capdrift.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_drip_ref import Ref                                          # noqa: E402
import geosfp_cap as FP                                                 # noqa: E402

LEVELS = FP.LEVELS                                                       # 1000 ... 10 hPa (11)
G = 9.80665
TAIL = 60
INK, MUTED = "#1f2328", "#6f6b64"


def cap_of(field: np.ndarray, lat: np.ndarray) -> float:
    m = lat >= 65.0
    w = np.cos(np.deg2rad(lat[m]))
    zm = np.nanmean(field[m], axis=-1)
    return float((zm * w).sum() / w.sum())


# ----------------------------------------------------------------------------------------------------------- fetch
def geps(date: str):
    """(member, lead, level) cap heights, m; leads 0..840 h by 24. 10/100 hPa reuse strat_maps' cache."""
    import eccodes
    import strat_maps as sm
    leads = list(range(0, 841, 24))
    tmp = Path(tempfile.mkdtemp(prefix="geps_drip_"))

    def one(job):
        li, ki, L, lev = job
        keep = lev in (10.0, 100.0)
        if keep:
            f = sm.fetch(date, "00", "HGT", int(lev), L)
        else:
            f = tmp / f"HGT{int(lev)}_{L:03d}.grib2"
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        sm.GEPS.format(d=date, c="00", v="HGT", lev=int(lev), L=L), headers=sm.UA), timeout=300) as r:
                    f.write_bytes(r.read())
            except Exception as e:                                               # noqa: BLE001
                print(f"    miss HGT{int(lev)} +{L}h: {str(e)[:60]}", flush=True)
                return li, ki, None
        if f is None:
            return li, ki, None
        caps = {}
        with open(f, "rb") as fh:
            while True:
                gid = eccodes.codes_grib_new_from_file(fh)
                if gid is None:
                    break
                try:
                    nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
                    la0, la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), \
                        eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                    v = eccodes.codes_get_values(gid).reshape(nj, ni)
                    try:
                        num = int(eccodes.codes_get(gid, "perturbationNumber"))
                    except Exception:                                            # noqa: BLE001
                        num = 0
                    caps[num] = cap_of(v, np.linspace(la0, la1, nj))
                finally:
                    eccodes.codes_release(gid)
        if not keep:
            f.unlink(missing_ok=True)
        return li, ki, caps
    jobs = [(li, ki, L, lev) for li, L in enumerate(leads) for ki, lev in enumerate(LEVELS)]
    C = None
    with ThreadPoolExecutor(8) as ex:
        for li, ki, caps in ex.map(one, jobs):
            if not caps:
                continue
            if C is None:
                C = np.full((max(caps) + 1, len(leads), len(LEVELS)), np.nan)
            for n, v in caps.items():
                if n < C.shape[0]:
                    C[n, li, ki] = v
    tmp.rmdir() if not any(tmp.iterdir()) else None
    return C, leads


def gefs(date: str, workers: int = 32):
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_reforecast as R
    S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    leads = list(range(0, 841, 24))
    lat = np.arange(90.0, -90.0 - 1e-6, -0.5)

    def one(job):
        mi, li, L = job
        url = f"{S3}/gefs.{date}/00/atmos/pgrb2ap5/{mems[mi]}.t00z.pgrb2a.0p50.f{L:03d}"
        try:
            idx = R.index(url)
        except Exception:                                                        # noqa: BLE001
            return mi, li, None
        out = {}
        tag = "anl" if L == 0 else f"{L} hour fcst"
        for lev in LEVELS:
            hit = [e for e in idx if e[4] == "HGT" and e[2] == f"{int(lev)} mb" and e[3] == tag]
            if not hit:
                return mi, li, None
            out[lev] = cap_of(R.decode(R.get(url, (hit[0][0], hit[0][1]))), lat)
        return mi, li, out
    C = np.full((len(mems), len(leads), len(LEVELS)), np.nan)
    with ThreadPoolExecutor(workers) as ex:
        for mi, li, o in ex.map(one, [(mi, li, L) for mi in range(len(mems)) for li, L in enumerate(leads)]):
            if o:
                C[mi, li] = [o[lev] for lev in LEVELS]
    ok = np.isfinite(C[:, 1:]).all(axis=(1, 2))
    if ok.sum() < 16:
        raise SystemExit(f"only {int(ok.sum())} complete GEFS members")
    return C[ok], leads


def aifs(date: str, time: str, members: int = 10):
    sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
    import store as E
    import xarray as xr
    cyc = E.Cycle(date, time)
    steps = list(range(0, 361, 24))
    levs = tuple(int(x) for x in LEVELS)
    parts = []
    for kind, n in (("cf", None), ("pf", members)):
        spec = E.Spec("aifs-ens", kind, "z", "pl", levs, tuple(steps), n) if n else E.Spec("aifs-ens", kind, "z", "pl", levs, tuple(steps))
        path = E.ensure(cyc, spec)
        arr = np.full(((n or 1), len(steps), len(LEVELS)), np.nan)
        for ki, lev in enumerate(levs):
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
                filter_by_keys={"shortName": "z", "level": lev}, indexpath=""))
            z = ds["z"]
            z = z.expand_dims("number") if "number" not in z.dims else z
            z = z.sel(latitude=z.latitude[z.latitude >= 65]) / G
            lat = z.latitude.values
            w = np.cos(np.deg2rad(lat))
            cap = (z.mean("longitude") * xr.DataArray(w, dims="latitude", coords={"latitude": lat})).sum("latitude") / w.sum()
            cap = cap.transpose("number", "step").values
            arr[:, :, ki] = cap[:, :len(steps)]
        parts.append(arr)
    return np.concatenate(parts, axis=0), steps


def _caps_members(path, short: str, levs, scale: float = 1.0):
    """(member, step, level) 65-90N cap heights from one multi-member, multi-level GRIB, one level at a time."""
    import xarray as xr
    arr = None
    steps = None
    for ki, lev in enumerate(levs):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
            filter_by_keys={"shortName": short, "level": int(lev)}, indexpath=""))
        z = ds[short] if short in ds else ds[list(ds.data_vars)[0]]
        z = z.expand_dims("number") if "number" not in z.dims else z
        z = z.expand_dims("step") if "step" not in z.dims else z
        z = z.sel(latitude=z.latitude[z.latitude >= 65])
        lat = z.latitude.values
        w = np.cos(np.deg2rad(lat))
        cap = ((z.mean("longitude") * xr.DataArray(w, dims="latitude", coords={"latitude": lat})).sum("latitude")
               / w.sum() * scale).transpose("number", "step").values
        if arr is None:
            steps = (z.step.values / np.timedelta64(1, "h")).astype(int)
            arr = np.full((cap.shape[0], cap.shape[1], len(levs)), np.nan)
        arr[:, :, ki] = cap
        ds.close()
    return arr, steps


def ifs(date: str, time: str):
    """(member, lead, level) cap heights for all 50 IFS-ENS perturbed members (no control exists at pressure levels
    on open data), daily 0..360 h, from the shared IFS-ENS gh file (ifs_ens.py; gh is already geopotential metres)."""
    import ifs_ens as IE
    if not IE.published(date, time):
        raise SystemExit(f"IFS-ENS {date} {time}Z is not on the Google mirror yet")
    path = IE.ensure(IE.E.Cycle(date, time), IE.spec_gh())
    C, steps = _caps_members(path, "gh", [int(x) for x in LEVELS])
    return C, [int(s) for s in steps]


# ----------------------------------------------------------------------------------------------------------- drift
def drift_geps(init: pd.Timestamp, leads, halfwin=8):
    """{level: drift array over leads} and the list of levels corrected; lead 0 takes the day-1 value."""
    f = HERE / "data" / "geps_s2s" / "capdrift_nh.json"
    if not f.exists():
        return {}, "no GEPS reforecast drift table"
    st = json.loads(f.read_text())["starts"]
    doy = init.dayofyear
    out, used = {}, set()
    for L in LEVELS:
        acc, ws, steps = 0.0, 0.0, None
        for d, e in st.items():
            gap = abs(pd.Timestamp(d).dayofyear - doy); gap = min(gap, 365 - gap)
            if gap > halfwin or str(L) not in e["levels"]:
                continue
            w = 1.0 - gap / (halfwin + 1.0)
            acc = acc + w * np.asarray(e["levels"][str(L)]["drift"], float); ws += w
            steps = np.asarray(e["step_h"], float); used.add(d)
        if ws:
            b = acc / ws
            pad = np.r_[b[:1], b, b[-1:]]; b = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
            out[L] = np.array([np.interp(max(h, steps[0]), steps, b) for h in leads])
    return out, f"S2S GEPS reforecasts, start dates {', '.join(sorted(used))}" if used else "no reforecast start date within 8 days"


def drift_gefs(init: pd.Timestamp, leads, ref: Ref):
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_live as GL
    import gefs_reforecast as R
    import os
    cache = Path(os.environ.get("GEFS_CLIM_CACHE", str(Path(tempfile.gettempdir()) / "gefs_clim2")))   # gefs.yml: /tmp/gefs_work/clim
    C2 = GL.clim2_at(init.date(), cache)
    if C2 is None:
        return {}, "GEFS reforecast climatology unavailable"
    lat = R.LAT if hasattr(R, "LAT") else np.linspace(90, -90, 121)
    out = {}
    for L, tag in ((10.0, "d_zg10"), (100.0, "d_zg100"), (500.0, "d_zg500")):
        if tag not in C2:
            continue
        f = np.asarray(C2[tag], float)                                            # (35, 121, 240): day 1..35 means
        la = np.linspace(90, -90, f.shape[1])
        caps = np.array([cap_of(f[d], la) for d in range(f.shape[0])])
        dr = []
        for h in leads:
            d = int(h // 24)                                                     # valid UTC day init+d -> clim day d+1
            k = min(max(d, 0), len(caps) - 1)
            vd = (init + pd.Timedelta(days=d))
            vref = vd - pd.DateOffset(years=vd.year - 2009)                       # the 2000-2019 reforecast epoch
            dr.append(caps[k] - float(ref.mean(pd.DatetimeIndex([vref]), L)[0]))
        out[L] = np.array(dr)
    return out, "GEFS v2 reforecast daily climatology (2000-2019)"


AIFS_DRIFT = HERE / "reference" / "aifs_capdrift.json"
IFS_DRIFT = HERE / "reference" / "ifs_capdrift.json"
DRIFT_FILE = {"aifs": AIFS_DRIFT, "ifs": IFS_DRIFT}


def measure_aifs_drift(history: Path, cycles: int = 25, every: int = 3, steps=(0, 72, 168, 240, 360)) -> int:
    return measure_drift("aifs", history, cycles, every, steps)


def measure_drift(model: str, history: Path, cycles: int = 25, every: int = 3, steps=(0, 72, 168, 240, 360)) -> int:
    """No reforecast exists for AIFS-ENS, nor an open one for IFS-ENS, so measure the drift on recent cycles: the cap
    forecast at each step minus the GEOS FP analysis cap at the valid date (both raw), averaged over past 00Z cycles
    whose day 15 has verified. Adding the GEOS FP - MERRA-2 offset puts it on the MERRA-2 scale like the other models'
    drifts.
      aifs  the CONTROL                                  -> reference/aifs_capdrift.json
      ifs   the mean of all 50 perturbed members (open data has no IFS control at pressure levels; the member mean's
            error has the same expectation as any member's)  -> reference/ifs_capdrift.json
    The IFS cycle files (~1.1 GB each) are deleted as soon as they are reduced."""
    sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
    import store as E
    H = FP.load_history(history)
    last = H.index.max() - pd.Timedelta(days=15)
    inits = [d for d in pd.date_range(H.index.min(), last, freq=f"{every}D")][-cycles:]
    rows = {s: [] for s in steps}
    levs = tuple(int(x) for x in LEVELS)
    for d in inits:
        cyc = E.Cycle(f"{d:%Y%m%d}", "00")
        try:
            if model == "ifs":
                import ifs_ens as IE
                spec = E.Spec("ifs", "pf", "gh", "pl", levs, tuple(steps))
                path = IE.ensure(cyc, spec)
                C, _ = _caps_members(path, "gh", levs)
                cap_ls = np.nanmean(C, axis=0).T                                  # (level, step): member mean
                path.unlink(missing_ok=True); path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
            else:
                path = E.ensure(cyc, E.Spec("aifs-ens", "cf", "z", "pl", levs, tuple(steps)))
                C, _ = _caps_members(path, "z", levs, 1.0 / G)
                cap_ls = C[0].T
        except Exception as e:                                                   # noqa: BLE001
            print(f"  {d:%Y-%m-%d}: {str(e)[:60]}"); continue
        for ki, lev in enumerate(LEVELS):
            for si, s_ in enumerate(steps):
                v = d + pd.Timedelta(hours=s_)
                if v in H.index and np.isfinite(H.loc[v, lev]):
                    rows[s_].append((lev, float(cap_ls[ki, si] - H.loc[v, lev] + FP.offset_at(pd.DatetimeIndex([v]))[0, LEVELS.index(lev)])))
        print(f"  {d:%Y-%m-%d} done", flush=True)
    what = ("AIFS-ENS control" if model == "aifs"
            else "IFS-ENS mean of the 50 perturbed members (open data has no IFS control at pressure levels)")
    out = {"method": f"{what}: cap height minus GEOS FP analysis cap at the valid date, plus the GEOS FP - MERRA-2 "
                     "offset (so on the MERRA-2 scale); mean over past 00Z cycles", "cycles": [f"{d:%Y-%m-%d}" for d in inits],
           "step_h": list(steps), "levels": {}}
    for lev in LEVELS:
        m, se = [], []
        for s_ in steps:
            v = np.array([x for L, x in rows[s_] if L == lev])
            m.append(round(float(v.mean()), 1) if len(v) else None)
            se.append(round(float(v.std(ddof=1) / np.sqrt(len(v))), 1) if len(v) > 2 else None)
        out["levels"][str(lev)] = {"drift": m, "se": se, "n": int(len([x for L, x in rows[steps[-1]] if L == lev]))}
        print(f"  {lev:6.0f} hPa: " + "  ".join(f"+{s_//24}d {mm:+.0f}±{ss:.0f}" for s_, mm, ss in zip(steps, m, se) if mm is not None and ss is not None))
    DRIFT_FILE[model].write_text(json.dumps(out, indent=1))
    return 0


def drift_aifs(init, leads, ref, tail_z_raw, model: str = "aifs"):
    """Measured drift (measure_drift): applied at a level only where it is significant at some lead (|mean| >
    2 se); interpolated linearly in lead between the measured steps. Levels where it is not significant are left
    uncorrected and hatched."""
    f = DRIFT_FILE[model]
    name = {"aifs": "AIFS", "ifs": "IFS-ENS"}[model]
    if not f.exists():
        return {}, f"no {name} drift measurement (uncorrected)"
    j = json.loads(f.read_text())
    st = np.array(j["step_h"], float)
    out = {}
    for lev in LEVELS:
        e = j["levels"].get(str(lev))
        if not e or any(v is None for v in e["drift"]):
            continue
        m, se = np.array(e["drift"], float), np.array([v or np.inf for v in e["se"]], float)
        if not (np.abs(m) > 2 * se).any():
            continue
        out[lev] = np.interp(np.array(leads, float), st, m)
    who = "AIFS control" if model == "aifs" else "IFS-ENS 50-member mean"
    return out, f"{who} vs GEOS FP analyses, {len(j['cycles'])} cycles {j['cycles'][0]}..{j['cycles'][-1]}"


# ---------------------------------------------------------------------------------------------------------- figure
def render(model, init, leads, Zf, corrected, tail_days, Zt, frac100, out: Path, note: str, nmem: int, hour: str = "00"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    days_f = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in leads])
    t_all = tail_days.append(days_f)
    X = np.r_[(tail_days - init).days.values, np.array(leads) / 24.0]
    Z = np.vstack([Zt, np.nanmean(Zf, 0)])                                     # (time, level)
    p = np.array(LEVELS)
    fig = plt.figure(figsize=(13.4, 8.9), dpi=120)
    name = {"geps": "GEPS", "gefs": "GEFS", "aifs": "AIFS-ENS", "ifs": "IFS-ENS"}[model]
    fig.text(0.07, 0.975, f"{name} · dripping paint: polar-cap height, observed and forecast, init {init:%d %b %Y} {hour}Z",
             ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.935, f"Standardised 65–90°N geopotential height (MERRA-2 1980–2026 scale, as the history drips). Left of "
             f"the line: GEOS FP analyses (offset to MERRA-2 removed). Right: ensemble mean of {nmem} members, model drift "
             f"removed where measured ({note}); hatched = drift not measured at that level (only the analysis-time offset "
             f"from GEOS FP removed), read with caution.",
             ha="left", va="top", fontsize=9.4, color=MUTED, wrap=True)
    ax = fig.add_axes([0.07, 0.40, 0.84, 0.47])
    lv = np.arange(-2.4, 2.41, 0.3)
    cf = ax.contourf(X, p, Z.T, levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(X, p, Z.T, levels=[x for x in lv if abs(x) > 1e-6], colors="#333", linewidths=0.3)
    unc = [k for k, L in enumerate(LEVELS) if L not in corrected]
    if unc:
        mask = np.zeros_like(Z.T); mask[unc, len(tail_days):] = 1
        ax.contourf(X, p, mask, levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.axvline(0, color=INK, lw=1.4)
    ax.text(-0.5, 12, "observed", ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.5, 12, "forecast", ha="left", va="top", fontsize=9, color=INK)
    ax.set_yscale("log"); ax.set_ylim(1000, 10)
    ax.set_yticks([1000, 700, 500, 300, 200, 100, 50, 30, 10]); ax.set_yticklabels(["1000", "700", "500", "300", "200", "100", "50", "30", "10"])
    ax.minorticks_off(); ax.set_ylabel("pressure, hPa", fontsize=10, color=INK)
    ax.set_xlim(X[0], X[-1])
    ticks = [x for x in range(int(np.ceil(X[0] / 7)) * 7, int(X[-1]) + 1, 7)]       # inside the data: no blank margin
    ax.set_xticks(ticks); ax.set_xticklabels([f"{init + pd.Timedelta(days=t):%d %b}" for t in ticks], fontsize=8.5)
    ax.set_xlim(X[0], X[-1])
    cax = fig.add_axes([0.925, 0.40, 0.012, 0.47])
    cb = fig.colorbar(cf, cax=cax); cb.set_label("standard deviations (red = high cap = weak vortex)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    # 1000 hPa strip and the member share at 100 hPa
    ax2 = fig.add_axes([0.07, 0.07, 0.84, 0.26])
    k1000 = LEVELS.index(1000.0); k100 = LEVELS.index(100.0)
    xf = np.array(leads) / 24.0
    ax2.plot((tail_days - init).days.values, Zt[:, k1000], color=INK, lw=1.4, label="1000 hPa cap, observed (GEOS FP)")
    q10, q90 = np.nanpercentile(Zf[:, :, k1000], [10, 90], axis=0)
    ax2.fill_between(xf, q10, q90, color="#b3122b", alpha=0.15, lw=0, label="members 10–90 %")
    ax2.plot(xf, np.nanmean(Zf[:, :, k1000], 0), color="#b3122b", lw=1.6,
             label="ensemble mean" + ("" if 1000.0 in corrected else " (analysis offset removed; drift not measured)"))
    ax2.axhline(0, color="#555", lw=0.8); ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xticks(ticks); ax2.set_xticklabels([f"{init + pd.Timedelta(days=t):%d %b}" for t in ticks], fontsize=8.5)
    ax2.set_xlim(X[0], X[-1])
    ax2.set_ylabel("sd (high = −AO)", fontsize=10, color=INK)
    ax3 = ax2.twinx()
    ax3.bar(xf, frac100 * 100, width=0.8, color="#8c6d31", alpha=0.35, label="members with 100 hPa cap ≥ +1 sd (%)")
    ax3.set_ylim(0, 100); ax3.set_ylabel("% of members, 100 hPa ≥ +1 sd", fontsize=9, color="#8c6d31")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.3, frameon=False, ncol=4, loc="upper left")
    ax2.set_title("Near the surface: the 1000 hPa polar cap (an Arctic Oscillation proxy) and the lower stratosphere's weak-vortex share",
                  loc="left", fontsize=10.2, fontweight="bold", color=INK)
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.text(0.07, 0.012, "Sources: " + {"geps": "ECCC GEPS (MSC Datamart); drift: ECMWF S2S GEPS reforecasts (non-commercial research licence)",
                                         "gefs": "NOAA GEFS (AWS Open Data); drift: GEFS v2 reforecast climatology",
                                         "aifs": "ECMWF AIFS-ENS open data (CC BY 4.0)",
                                         "ifs": "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25°, Google Cloud mirror"}[model]
             + " · GEOS FP (NASA GMAO) · MERRA-2 scale (NASA GMAO)", fontsize=8.2, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def _js(a, nd):
    """Rounded nested list with NaN -> null (the observed 1000 hPa cap is undefined on days the surface is underground
    across a whole latitude row, as in MERRA-2)."""
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), np.round(a, nd), None).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("geps", "gefs", "aifs", "ifs"))
    ap.add_argument("--date"); ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir")
    ap.add_argument("--history", default=str(REPO / "assets" / "sst" / "data" / "geosfp_cap_history.json"))
    ap.add_argument("--cache", help="npz of the reduced member caps (reused if present)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--measure-aifs-drift", action="store_true")
    ap.add_argument("--measure-drift", choices=("aifs", "ifs"))
    a = ap.parse_args()
    if a.measure_aifs_drift or a.measure_drift:
        return measure_drift(a.measure_drift or "aifs", Path(a.history))
    if not a.model or not a.date or not a.out_dir:
        ap.error("--model, --date and --out-dir are required")
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]}")
    ref = Ref()
    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        z = np.load(cache); C, leads = z["C"], [int(x) for x in z["leads"]]
    else:
        C, leads = {"geps": lambda: geps(a.date), "gefs": lambda: gefs(a.date),
                    "aifs": lambda: aifs(a.date, a.time), "ifs": lambda: ifs(a.date, a.time)}[a.model]()
        if cache:
            np.savez_compressed(cache, C=C, leads=np.array(leads))
    if a.fetch_only:
        print(f"  fetched {C.shape} -> {cache}")
        return 0
    dr, note = {"geps": lambda: drift_geps(init, leads), "gefs": lambda: drift_gefs(init, leads, ref),
                "aifs": lambda: drift_aifs(init, leads, ref, None),
                "ifs": lambda: drift_aifs(init, leads, ref, None, "ifs")}[a.model]()
    C = C[np.isfinite(C[:, 1:, :]).all(axis=(1, 2))]
    # observed tail first: it also measures each model's ANALYSIS offset from GEOS FP (hence from MERRA-2), which is
    # removed at the levels whose drift has not been measured. The 1000 hPa cap is the case that needs it: models
    # extrapolate heights below ground (Greenland) where GEOS FP and MERRA-2 leave the points out of the zonal mean, so
    # a model's 1000 hPa cap reads ~40 m (about 1 sd) low at analysis time - a definition, not a forecast.
    H = FP.load_history(Path(a.history))
    Hraw = H.copy()
    offsets = {}
    for L in LEVELS:
        if L in dr:
            continue
        diffs = []
        for li, h in enumerate(leads[:2]):                                   # the analysis and day 1
            v = init + pd.Timedelta(hours=h)
            if v in Hraw.index and np.isfinite(Hraw.loc[v, L]):
                o = FP.offset_at(pd.DatetimeIndex([v]))[0, LEVELS.index(L)]
                diffs.append(np.nanmean(C[:, li, LEVELS.index(L)]) - (Hraw.loc[v, L] - o))
        if diffs:
            offsets[L] = float(np.mean(diffs))
            dr[L] = np.full(len(leads), offsets[L])
    days_f = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in leads]).normalize()
    # forecast: drop lead 0 (the analysis is the observed tail), standardise per member after removing drift
    keep = np.array(leads) > 0
    leads_f = [h for h, k in zip(leads, keep) if k]; days_f = days_f[keep]
    Zf = np.full((C.shape[0], len(leads_f), len(LEVELS)), np.nan)
    for k, L in enumerate(LEVELS):
        d = dr.get(L)
        dd = d[keep] if d is not None else 0.0
        Zf[:, :, k] = (C[:, keep, k] - dd - ref.mean(days_f, L)) / ref.sd(days_f, L)
    # observed tail
    H = H[(H.index < init) & (H.index >= init - pd.Timedelta(days=TAIL))]
    if len(H) < 20:
        raise SystemExit(f"GEOS FP tail too short ({len(H)} days)")
    H = H.asfreq("D").interpolate(limit=3)
    raw = H.values - FP.offset_at(H.index)
    Zt = np.column_stack([(raw[:, k] - ref.mean(H.index, L)) / ref.sd(H.index, L) for k, L in enumerate(LEVELS)])
    frac100 = np.nanmean(Zf[:, :, LEVELS.index(100.0)] >= 1.0, 0)
    corrected = set(dr) - set(offsets)
    name = {"geps": "GEPS", "gefs": "GEFS", "aifs": "AIFS-ENS", "ifs": "IFS-ENS"}[a.model]
    out = Path(a.out_dir); out = out if out.is_absolute() else REPO / out
    render(a.model, init, leads_f, Zf, corrected, H.index, Zt, frac100, out / f"{a.model}_drip.webp", note, C.shape[0],
           hour=f"{int(a.time):02d}")
    mean = np.nanmean(Zf, 0)
    doc = {"model": name, "init": f"{init:%Y-%m-%d} {a.time}Z", "members": int(C.shape[0]), "levels": LEVELS,
           "corrected_levels": sorted(corrected), "drift_source": note,
           "analysis_offset_only_m": {str(L): round(v, 1) for L, v in offsets.items()},
           "drift_m": {str(L): np.round(dr[L][keep], 1).tolist() for L in dr}, "lead_h": leads_f,
           "forecast_mean_sd": _js(mean, 2), "frac100_ge1": _js(frac100, 3),
           "tail_days": [f"{d:%Y-%m-%d}" for d in H.index], "tail_sd": _js(Zt, 2)}
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / f"{a.model}_drip.json").write_text(json.dumps(doc, separators=(",", ":")))
    k10, k100, k1000 = LEVELS.index(10.0), LEVELS.index(100.0), LEVELS.index(1000.0)
    for d0, d1 in ((1, 7), (8, 14), (15, 21), (22, 35)):
        m = (np.array(leads_f) / 24 >= d0) & (np.array(leads_f) / 24 <= d1)
        if m.any():
            print(f"  {name} days {d0}-{d1}: cap 10/100/1000 hPa {np.nanmean(mean[m, k10]):+.2f}/{np.nanmean(mean[m, k100]):+.2f}/"
                  f"{np.nanmean(mean[m, k1000]):+.2f} sd; 100 hPa >= +1 sd in {100 * np.nanmean(frac100[m]):.0f} % of members")
    print(f"  drift ({note}): " + ", ".join(f"{L:.0f} hPa d{leads[-1] // 24} {dr[L][-1]:+.0f} m" for L in sorted(corrected)))
    if offsets:
        print("  analysis offset only: " + ", ".join(f"{L:.0f} hPa {v:+.0f} m" for L, v in sorted(offsets.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
