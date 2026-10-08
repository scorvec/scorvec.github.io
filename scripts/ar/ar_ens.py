#!/usr/bin/env python3
"""Ensemble AR probabilities, days 1-15, 6-hourly: ECMWF AIFS-ENS (51), ECMWF IFS-ENS (50 perturbed), NCEP GEFS (31),
ECCC GEPS (21), and their equal-weight combination.

Exact IVT for every member needs q, u, v through the column: ~6-50 GB per system per cycle. Each system therefore
uses the PROXY  IVT ≈ k · W · |V850|  (W = total column water / precipitable water, V850 the 850 hPa wind, the vector
along V850), with k fitted EVERY CYCLE through the origin against that system's own exact IVT for one member
(points with exact IVT > 100; the fit r, rmse and the proxy's hit/false-alarm rates at 250 are published):

    AIFS-ENS  control, q/u/v on the 8 open-data levels 1000-300 (Google mirror)
    IFS-ENS   member 1 (the open data carry no IFS-ENS control at these steps), same 8 levels (Google mirror)
    GEFS      control gec00, RH/T/u/v on every level 1000-300 in pgrb2a + pgrb2b, q from RH (AWS S3)
    GEPS      control, RH/T at 1000/925/850/700/500/250 and u/v at 1000-250 (Datamart); q at 600/400/300 by
              log-linear interpolation in p between 700/500 and 500/250, u/v at 600 linearly - calibrated every 48 h

Steps: 6-hourly from the analysis to +360 h for all four (every system publishes 6-hourly or finer to 360 h; the
finer early steps are not used, so every system and the combination share one time axis).

    python ar_ens.py system --sys gefs --work /tmp/are [--date YYYYMMDD --time HH]   -> /tmp/are/ens_gefs.npz
    python ar_ens.py combine --work /tmp/are                                          -> frames, figures, JSON
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arlib as A                                                      # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "ar"
ANIM = ASSETS / "anim"
STEPS = tuple(range(0, 361, 6))
LAT_E = np.arange(12.0, 66.001, 0.5)
LON_E = np.arange(130.0, 300.001, 0.5)
ECROP = dict(lat=(11.9, 66.1), lon=(129.9, 300.1))
SYSTEMS = {
    "aifs": dict(label="ECMWF AIFS-ENS", short="AIFS-ENS", cal="control"),
    "ifs": dict(label="ECMWF IFS-ENS", short="IFS-ENS", cal="member 1"),
    "gefs": dict(label="NCEP GEFS", short="GEFS", cal="control"),
    "geps": dict(label="ECCC GEPS", short="GEPS", cal="control"),
}
ORDER = ("aifs", "ifs", "gefs", "geps")


def on_grid(lat, lon, v):
    """Pick the 0.5-deg ENS points out of a native field (exact for the 0.25 and 0.5 deg grids)."""
    i = A.nearest_idx(lat, LAT_E); j = A.nearest_idx(lon, LON_E)
    if np.max(np.abs(lat[i] - LAT_E)) > 0.01 or np.max(np.abs(lon[j] - LON_E)) > 0.01:
        raise RuntimeError("native grid does not hold the 0.5-deg points")
    return v[..., i[:, None], j[None, :]]


def fit_proxy(x, y):
    """k through the origin over y > 100, with r, rmse and the proxy's skill at the 250 threshold."""
    ok = np.isfinite(x) & np.isfinite(y) & (y > 100)
    k = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())
    px = k * x
    hit = (px >= 250) & (y >= 250); miss = (px < 250) & (y >= 250); fa = (px >= 250) & (y < 250)
    return {"k": round(k, 4), "r": round(float(np.corrcoef(px[ok], y[ok])[0, 1]), 3),
            "rmse": round(float(np.sqrt(np.mean((px[ok] - y[ok]) ** 2))), 1), "n": int(ok.sum()),
            "pod250": round(float(hit.sum() / max(1, hit.sum() + miss.sum())), 3),
            "far250": round(float(fa.sum() / max(1, hit.sum() + fa.sum())), 3),
            "bias250": round(float((hit.sum() + fa.sum()) / max(1, hit.sum() + miss.sum())), 3)}


# ── readers: each returns dict(W (mem, step, lat, lon), U, V, exact (cal steps, lat, lon), exact_steps, init, members)
def _ecmwf_grib(path, nmem, steps, member_offset=0, numbers=None):
    """Decode an ECMWF store file into (member, step, lat, lon) on the ENS grid."""
    import eccodes as ec
    out = np.full((nmem, len(steps), len(LAT_E), len(LON_E)), np.nan, "float32")
    si = {s: k for k, s in enumerate(steps)}
    with open(path, "rb") as f:
        while True:
            g = ec.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                msg = ec.codes_get_message(g)
                num = ec.codes_get(g, "perturbationNumber") if ec.codes_is_defined(g, "perturbationNumber") else 0
                st = int(ec.codes_get(g, "endStep"))
            finally:
                ec.codes_release(g)
            if st not in si:
                continue
            m = (numbers.index(num) if numbers else num - member_offset)
            if not (0 <= m < nmem):
                continue
            _, v, la, lo = A.decode(msg, ECROP)
            out[m, si[st]] = on_grid(la, lo, v)
    return out


def read_ecmwf(model, date, hh):
    import store as ecmwf
    cyc = ecmwf.Cycle(date, hh)
    S = tuple(STEPS)
    if model == "aifs-ens":
        parts = {}
        for p, lt, lv in (("tcw", "sfc", ()), ("u", "pl", (850,)), ("v", "pl", (850,))):
            cf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", p, lt, lv, S))
            pf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", p, lt, lv, S))
            a = np.concatenate([_ecmwf_grib(cf, 1, S), _ecmwf_grib(pf, 50, S, member_offset=1)])
            A.LEDGER.add("ecmwf-google", cf.stat().st_size + pf.stat().st_size)
            parts[p] = a; os.remove(pf)
            print(f"  {p}: {a.shape}, NaN {np.isnan(a).mean():.4f}", flush=True)
        cal_type, cal_numbers = "cf", None
        members = 51
    else:                                                                # IFS-ENS: 50 perturbed members
        parts = {}
        for p, lt, lv in (("tcw", "sfc", ()), ("u", "pl", (850,)), ("v", "pl", (850,))):
            pf = ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", p, lt, lv, S))
            parts[p] = _ecmwf_grib(pf, 50, S, member_offset=1)
            A.LEDGER.add("ecmwf-google", pf.stat().st_size); os.remove(pf)
            print(f"  {p}: {parts[p].shape}, NaN {np.isnan(parts[p]).mean():.4f}", flush=True)
        cal_type, cal_numbers = "pf", [1]
        members = 50
    # exact IVT for the calibration member, 8 levels, plus surface pressure
    lv = A.ECMWF_LEVELS
    fld = {}
    for p in ("q", "u", "v"):
        sp = ecmwf.Spec(model if model == "aifs-ens" else "ifs", cal_type, p, "pl", lv, S, nmembers=(1 if cal_type == "pf" else 0))
        fp = ecmwf.ensure(cyc, sp); A.LEDGER.add("ecmwf-google", fp.stat().st_size)
        fld[p] = _levels(fp, lv, S)
    sps = ecmwf.Spec(model if model == "aifs-ens" else "ifs", cal_type, "sp", "sfc", (), S, nmembers=(1 if cal_type == "pf" else 0))
    ps = _ecmwf_grib(ecmwf.ensure(cyc, sps), 1, S, numbers=[0] if cal_type == "cf" else [1])[0]
    exact = np.stack([np.hypot(*A.integrate(lv, fld["q"][:, k], fld["u"][:, k], fld["v"][:, k], ps[k])) for k in range(len(S))])
    return dict(W=parts["tcw"], U=parts["u"], V=parts["v"], exact=exact, exact_steps=list(S), members=members,
                cal_index=(0 if model == "aifs-ens" else 0), cal_levels=list(lv))


def _levels(path, levels, steps):
    """(lev, step, lat, lon) of a one-member pressure-level store file."""
    import eccodes as ec
    out = np.full((len(levels), len(steps), len(LAT_E), len(LON_E)), np.nan, "float32")
    si = {s: k for k, s in enumerate(steps)}; li = {L: k for k, L in enumerate(levels)}
    with open(path, "rb") as f:
        while True:
            g = ec.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                msg = ec.codes_get_message(g); st = int(ec.codes_get(g, "endStep")); L = int(ec.codes_get(g, "level"))
            finally:
                ec.codes_release(g)
            if st in si and L in li:
                _, v, la, lo = A.decode(msg, ECROP)
                out[li[L], si[st]] = on_grid(la, lo, v)
    return out


GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"


def gefs_url(date, hh, mem, h, part="a"):
    sub, nm = (("pgrb2ap5", "pgrb2a.0p50") if part == "a" else ("pgrb2bp5", "pgrb2b.0p50"))
    return f"{GEFS_S3}/gefs.{date}/{hh}/atmos/{sub}/{mem}.t{hh}z.{nm}.f{h:03d}"


def _range(url, s, e):
    b = A.http_get(url, (s, e), src="aws-s3")
    if b is None or not b.startswith(b"GRIB"):
        raise RuntimeError(f"bad range {url}")
    _, v, la, lo = A.decode(b, ECROP)
    return on_grid(la, lo, v)


def read_gefs(date, hh, nmem=31):
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, nmem)]
    W = np.full((nmem, len(STEPS), len(LAT_E), len(LON_E)), np.nan, "float32"); U = W.copy(); V = W.copy()
    jobs = []

    def plan(args):
        m, mem, k, h = args
        recs = A.idx_records(gefs_url(date, hh, mem, h))
        if recs is None:
            return []
        got = []
        for s, e, var, lev, st in recs:
            if var == "PWAT" and lev.startswith("entire atmosphere"):
                got.append((m, k, "W", s, e))
            elif var in ("UGRD", "VGRD") and lev == "850 mb":
                got.append((m, k, var[0], s, e))
        return [(gefs_url(date, hh, mem, h),) + g for g in got]

    with ThreadPoolExecutor(24) as ex:
        for r in ex.map(plan, [(m, mem, k, h) for m, mem in enumerate(mems) for k, h in enumerate(STEPS)]):
            jobs += r
    print(f"  GEFS: {len(jobs)} records planned of {nmem * len(STEPS) * 3}", flush=True)
    tgt = {"W": W, "U": U, "V": V}

    def get(j):
        url, m, k, which, s, e = j
        tgt[which][m, k] = _range(url, s, e)

    with ThreadPoolExecutor(24) as ex:
        list(ex.map(get, jobs))
    # exact IVT for the control: every RH/T/u/v level 1000-300 in pgrb2a + pgrb2b, surface pressure
    ex_list = []
    lv_used = None

    def ctrl(h):
        fld = {}
        for part in ("a", "b"):
            url = gefs_url(date, hh, "gec00", h, part)
            recs = A.idx_records(url)
            if recs is None:
                return None
            for s, e, var, lev, st in recs:
                m_ = re.match(r"^(\d+) mb$", lev)
                if var in ("RH", "TMP", "UGRD", "VGRD") and m_ and 300 <= int(m_.group(1)) <= 1000:
                    fld.setdefault((var, int(m_.group(1))), (url, s, e))
                elif var == "PRES" and lev == "surface" and part == "a":
                    fld[("PRES", 0)] = (url, s, e)
        levs = sorted({L for (v, L) in fld if v == "RH"} & {L for (v, L) in fld if v == "TMP"} &
                      {L for (v, L) in fld if v == "UGRD"} & {L for (v, L) in fld if v == "VGRD"}, reverse=True)
        vals = {key: _range(*fld[key]) for key in [(v, L) for v in ("RH", "TMP", "UGRD", "VGRD") for L in levs] + [("PRES", 0)]}
        q = np.stack([A.q_from_rh(vals[("RH", L)], vals[("TMP", L)], L) for L in levs])
        u = np.stack([vals[("UGRD", L)] for L in levs]); v = np.stack([vals[("VGRD", L)] for L in levs])
        return levs, np.hypot(*A.integrate(levs, q, u, v, vals[("PRES", 0)]))

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(ctrl, STEPS))
    exact = np.stack([r[1] for r in res]); lv_used = res[0][0]
    print(f"  GEFS control exact IVT on {len(lv_used)} levels: {lv_used}", flush=True)
    return dict(W=W, U=U, V=V, exact=exact, exact_steps=list(STEPS), members=nmem, cal_index=0, cal_levels=lv_used)


GEPS_URL = (A.DATAMART + "/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/CMC_geps-raw_{v}_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2")
GEPS_CAL_STEPS = tuple(range(0, 361, 48)) + (360,)


def _geps_file(date, hh, var, h):
    b = A.http_get(GEPS_URL.format(d=date, c=hh, v=var, L=h), src="datamart", timeout=180)
    if b is None:
        return None
    out = {}
    for msg in A.split_grib(b):
        meta, v, la, lo = A.decode(msg, ECROP)
        out[int(meta.get("perturbationNumber") or 0)] = on_grid(la, lo, v)
    return out


def read_geps(date, hh):
    jobs = [(var, k, h) for var in ("PWAT_EATM_0", "UGRD_ISBL_0850", "VGRD_ISBL_0850") for k, h in enumerate(STEPS)]
    res = {}

    def one(j):
        var, k, h = j
        res[(var, k)] = _geps_file(date, hh, var, h)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, jobs))
    miss = [j for j in jobs if res.get((j[0], j[1])) is None]
    if miss:
        raise SystemExit(f"GEPS {date} {hh}Z: {len(miss)} files missing (e.g. {miss[:3]}); cycle left partial")
    nmem = max(len(v) for v in res.values())
    shp = (nmem, len(STEPS), len(LAT_E), len(LON_E))
    W = np.full(shp, np.nan, "float32"); U = W.copy(); V = W.copy()
    for (var, k), d in res.items():
        tgt = {"PWAT_EATM_0": W, "UGRD_ISBL_0850": U, "VGRD_ISBL_0850": V}[var]
        for m, v in d.items():
            tgt[m, k] = v
    # control exact IVT every 48 h
    lq = (1000, 925, 850, 700, 500, 250); lw = (1000, 925, 850, 700, 500, 400, 300, 250)
    exact = []

    def ctrl(h):
        f = {}
        for L in lq:
            f[("RH", L)] = _geps_file(date, hh, f"RH_ISBL_{L:04d}", h)[0]
            f[("TMP", L)] = _geps_file(date, hh, f"TMP_ISBL_{L:04d}", h)[0]
        for L in lw:
            f[("U", L)] = _geps_file(date, hh, f"UGRD_ISBL_{L:04d}", h)[0]
            f[("V", L)] = _geps_file(date, hh, f"VGRD_ISBL_{L:04d}", h)[0]
        ps = _geps_file(date, hh, "PRES_SFC_0", h)[0]
        if np.nanmedian(ps) < 2000:
            ps = ps * 100.0
        return geps_column(f, ps)

    with ThreadPoolExecutor(2) as ex:
        exact = list(ex.map(ctrl, GEPS_CAL_STEPS))
    return dict(W=W, U=U, V=V, exact=np.stack(exact), exact_steps=list(GEPS_CAL_STEPS), members=nmem, cal_index=0,
                cal_levels="q: 1000/925/850/700/500/250 (600, 400, 300 log-interpolated); u/v: 1000-250 (600 interpolated)")


def geps_column(f, ps):
    """IVT from the GEPS level set: q at 1000/925/850/700/500/250, u/v at 1000/925/850/700/500/400/300/250. q at 600,
    400 and 300 hPa is interpolated linearly in ln q against ln p (700-500 and 500-250), u/v at 600 linearly in ln p."""
    q = {L: A.q_from_rh(f[("RH", L)], f[("TMP", L)], L) for L in (1000, 925, 850, 700, 500, 250)}

    def logq(L, a, b):
        wa = (np.log(b) - np.log(L)) / (np.log(b) - np.log(a))
        return np.exp(wa * np.log(np.maximum(q[a], 1e-7)) + (1 - wa) * np.log(np.maximum(q[b], 1e-7)))

    q[600] = logq(600, 700, 500); q[400] = logq(400, 500, 250); q[300] = logq(300, 500, 250)
    w6 = (np.log(500) - np.log(600)) / (np.log(500) - np.log(700))
    u = {L: f[("U", L)] for L in (1000, 925, 850, 700, 500, 400, 300)}; v = {L: f[("V", L)] for L in u}
    u[600] = w6 * u[700] + (1 - w6) * u[500]; v[600] = w6 * v[700] + (1 - w6) * v[500]
    lv = A.ECMWF_LEVELS
    return np.hypot(*A.integrate(lv, np.stack([q[L] for L in lv]), np.stack([u[L] for L in lv]), np.stack([v[L] for L in lv]), ps))


# ── cycle resolution ─────────────────────────────────────────────────────────────────────────────────────────────
def _ready(system, d, hh):
    if system in ("aifs", "ifs"):
        import cycles
        prof = cycles.AIFS_ENS if system == "aifs" else (cycles.Need("ifs", "ef", step=360, every=cycles.IFS_STEPS),)
        return cycles.is_ready(d, hh, prof)[0]
    if system == "gefs":
        return A.http_get(gefs_url(d, hh, "gep30", 360) + ".idx", src="aws-s3") is not None
    import requests
    r = requests.head(GEPS_URL.format(d=d, c=hh, v="VGRD_ISBL_0850", L=360), headers={"User-Agent": A.UA}, timeout=30)
    return r.status_code == 200


def latest(system):
    """Newest complete 00/12Z cycle. The cycle due now (init >= 7 h ago) is waited for up to AR_WAIT_MIN minutes,
    then older ones are taken (the page states each system's init)."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    t_end = time.time() + 60 * float(os.environ.get("AR_WAIT_MIN", "0"))
    for back in range(0, 6):
        t = (now - pd.Timedelta(hours=7) - pd.Timedelta(hours=12 * back)).floor("12h")
        d, hh = t.strftime("%Y%m%d"), t.strftime("%H")
        while True:
            if _ready(system, d, hh):
                return d, hh
            if back > 0 or time.time() > t_end:
                break
            print(f"  {system} {d} {hh}Z not complete yet; waiting", flush=True); time.sleep(120)
        print(f"  {system} {d} {hh}Z not complete", flush=True)
    return None


def run_system(system, date, hh, work: Path):
    t0 = time.time()
    if date is None:
        r = latest(system)
        if r is None:
            raise SystemExit(f"{system}: no complete cycle in the last 3 days")
        date, hh = r
    init = pd.Timestamp(f"{date}T{hh}:00")
    print(f"== {SYSTEMS[system]['label']} {date} {hh}Z", flush=True)
    if system == "aifs":
        D = read_ecmwf("aifs-ens", date, hh)
    elif system == "ifs":
        D = read_ecmwf("ifs", date, hh)
    elif system == "gefs":
        D = read_gefs(date, hh)
    else:
        D = read_geps(date, hh)
    t_fetch = time.time() - t0
    W, U, V = D["W"], D["U"], D["V"]
    nan = float(np.isnan(W).mean())
    if nan > 0.001:
        raise SystemExit(f"{system}: {nan:.3%} of the member fields missing")
    # screens
    if not (20 < float(np.nanmax(W)) < 170) or not (10 < float(np.nanmax(np.abs(U))) < 120):
        raise SystemExit(f"{system}: range screen failed (max W {np.nanmax(W):.1f}, max |u850| {np.nanmax(np.abs(U)):.1f})")
    si = [STEPS.index(s) for s in D["exact_steps"]]
    m0 = D["cal_index"]
    spd = np.hypot(U, V)
    x = (W[m0][si] * spd[m0][si]).ravel(); y = D["exact"].ravel()
    cal = fit_proxy(x, y)
    print(f"  proxy on {SYSTEMS[system]['cal']}: {cal}", flush=True)
    k = cal["k"]
    ivt = (k * W * spd).astype("float32")                              # (member, step, lat, lon)
    p250 = (ivt >= 250).mean(0).astype("float16"); p500 = (ivt >= 500).mean(0).astype("float16")
    mean = ivt.mean(0).astype("float16")
    ue = (k * W * U).mean(0).astype("float16"); un = (k * W * V).mean(0).astype("float16")
    tl, tlon = A.transect()
    ci = A.nearest_idx(LAT_E, tl); cj = A.nearest_idx(LON_E, tlon)
    coast = np.clip(np.round(ivt[:, :, ci, cj]), 0, 32000).astype("int16")
    pts = A.coast_points()
    pi = A.nearest_idx(LAT_E, [p[0] for p in pts]); pj = A.nearest_idx(LON_E, [p[1] for p in pts])
    points = np.clip(np.round(ivt[:, :, pi, pj]), 0, 32000).astype("int16")
    valid = np.array([(init + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H") for h in STEPS])
    import resource
    meta = {"system": system, "label": SYSTEMS[system]["label"], "init": init.strftime("%Y-%m-%dT%HZ"),
            "members": int(D["members"]), "calibration": dict(cal, member=SYSTEMS[system]["cal"], levels=D["cal_levels"],
                                                              steps=len(D["exact_steps"])),
            "download": A.LEDGER.report(), "runtime_s": {"fetch": round(t_fetch), "total": round(time.time() - t0)},
            "peak_rss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
    np.savez_compressed(work / f"ens_{system}.npz", p250=p250, p500=p500, mean=mean, ue=ue, un=un,
                        wmean=W.mean(0).astype("float16"), coast=coast, points=points, valid=valid,
                        meta=np.array(json.dumps(meta)))
    print(f"== {system}: {json.dumps(meta)}", flush=True)


# ── combine: maps, coastal figures, JSON ─────────────────────────────────────────────────────────────────────────
_R: dict = {}


def _lab(init, t):
    h = int((t - init) / pd.Timedelta(hours=1))
    return f"+{h} h · {t:%a %d %b %HZ}"


def _render(job):
    key, k = job
    R = _R[key]
    out = R["dir"] / f"F{k:02d}.webp"
    t = R["valid"][k]
    if R["kind"] == "prob":
        A.prob_frame(out, LAT_E, LON_E, R["p"][k].astype("float32"), A.ENS_DOMAIN, R["title"] + f" · valid {t:%a %d %b %HZ}",
                     R["sub"], contour=(None if R["c"] is None else R["c"][k].astype("float32")), contour_label=R["clab"])
    else:
        pwat_frame(out, R, k)
    return key, k


def pwat_frame(out, R, k):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    fig, ax, H = A.map_axes(A.ENS_DOMAIN)
    t = R["valid"][k]
    if R["kind"] == "pwat":
        lev = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        cm = ListedColormap(["#f1e3c4", "#d9c08c", "#b7d7a8", "#7fc4a0", "#45a8a0", "#2c88b5", "#2a62a8", "#3b3f9a", "#5e2d8f",
                             "#8b2a8a", "#b8327a", "#dc4f61", "#ef7b4a", "#f6aa3c"])
        pm = ax.pcolormesh(LON_E, LAT_E, R["w"][k], cmap=cm, norm=BoundaryNorm(lev, cm.N, extend="both"), transform=pc,
                           shading="nearest", rasterized=True)
        ax.contour(LON_E, LAT_E, R["a"][k], levels=[10, 20], colors=["#111", "#000"], linewidths=[0.7, 1.2], transform=pc)
        cl = "total column water (mm) · black contours: anomaly +10 and +20 mm vs ERA5 1991–2020"
        ttl = "Precipitable water, ensemble mean"
    else:
        alev = [-30, -25, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 25, 30]
        acm = LinearSegmentedColormap.from_list("pwa", ["#5c3305", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#f5f5f5",
                                                        "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30"], N=len(alev) + 1)
        pm = ax.pcolormesh(LON_E, LAT_E, R["a"][k], cmap=acm, norm=BoundaryNorm(alev, acm.N, extend="both"), transform=pc,
                           shading="nearest", rasterized=True)
        cs = ax.contour(LON_E, LAT_E, R["w"][k], levels=[25, 40], colors=["#333", "#000"], linewidths=[0.6, 1.0], transform=pc)
        ax.clabel(cs, fmt="%d mm", fontsize=7)
        cl = f"anomaly (mm) vs ERA5 1991–2020 at {t:%HZ} for the date · contours: precipitable water 25 and 40 mm"
        ttl = "Precipitable-water anomaly, ensemble mean"
    A.map_features(ax)
    A.heading(fig, H, f"{ttl} — {R['label']} · init {R['init']:%d %b %HZ} · {_lab(R['init'], t)}", R["sub"])
    cax = fig.add_axes([0.2, 0.36 / H, 0.6, 0.13 / H]); cb = fig.colorbar(pm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.5); cb.set_label(cl, fontsize=8)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 65, "method": 6}); plt.close(fig)


def pwat_clim(valid):
    import xarray as xr
    d = xr.open_dataset(HERE / "data" / "pwat_clim.nc")
    out = []
    for t in valid:
        c = d["coef"].sel(hour=12 if 6 <= t.hour < 18 else 0).values     # the climatology holds 00Z and 12Z
        w = 2 * np.pi * t.dayofyear / 365.25
        b = [1.0] + [f(k * w) for k in range(1, (c.shape[0] - 1) // 2 + 1) for f in (np.cos, np.sin)]
        f = xr.DataArray(np.tensordot(np.array(b), c, axes=(0, 0)), dims=("latitude", "longitude"),
                         coords={"latitude": d.latitude.values, "longitude": d.longitude.values})
        out.append(f.interp(latitude=LAT_E, longitude=LON_E).values)
    return np.stack(out).astype("float32")


def coast_figure(out, label, init, valid, coast_mem, title_note):
    """Coast latitude x time: P(IVT >= 250) shaded, ensemble-mean IVT 250/500 contoured, P(>= 500) = 50 % dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm, ListedColormap
    tl, _ = A.transect()
    p250 = (coast_mem >= 250).mean(0); p500 = (coast_mem >= 500).mean(0); mean = coast_mem.mean(0)
    t = pd.DatetimeIndex(valid)
    fig, ax = plt.subplots(figsize=(12, 6.4))
    lev = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cols = ["#eef4fa", "#d4e4f2", "#a9cbe6", "#7fb0d8", "#5793c4", "#3a76ad", "#f2c14e", "#ec8f2f", "#d9542b", "#a51c30"]
    cm = ListedColormap(cols)
    pm = ax.pcolormesh(t, tl, np.where(p250 >= 0.05, p250, np.nan).T, cmap=cm, norm=BoundaryNorm(lev, cm.N), shading="nearest")
    cs = ax.contour(t, tl, mean.T, levels=[250, 500], colors=["#1b365d", "#08102a"], linewidths=[0.9, 1.4])
    ax.clabel(cs, fmt="%d", fontsize=7)
    if (p500 >= 0.5).any():
        ax.contour(t, tl, p500.T, levels=[0.5], colors=["#111"], linewidths=1.0, linestyles="--")
    named = [p for p in A.coast_points()]
    ax.set_yticks([p[0] for p in named]); ax.set_yticklabels([f"{p[2].split(' (')[0]}  {p[0]:g}°N" for p in named], fontsize=7.5)
    ax.set_ylim(tl.min(), tl.max())
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    ax.tick_params(axis="x", labelsize=7.5); ax.grid(True, axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.21)
    fig.text(0.02, 0.97, f"West Coast landfall — {label}, init {init:%d %b %Y %HZ}, 6-hourly to day 15", fontsize=12.5,
             fontweight="bold", va="top")
    fig.text(0.02, 0.925, title_note, fontsize=8.5, color="#444", va="top")
    cax = fig.add_axes([0.3, 0.085, 0.5, 0.025])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=lev[:-1] + [1.0])
    cb.ax.set_xticklabels(["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]); cb.ax.tick_params(labelsize=7.5)
    cb.set_label("P(IVT ≥ 250 kg m⁻¹ s⁻¹), % of members · navy: ensemble-mean IVT 250 / 500 · dashed: P(IVT ≥ 500) = 50 %", fontsize=8)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6}); plt.close(fig)


def combine(work: Path, procs: int):
    t0 = time.time()
    loaded = {}
    for s in ORDER:
        p = work / f"ens_{s}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            loaded[s] = {k: z[k] for k in z.files}
            loaded[s]["meta"] = json.loads(str(z["meta"]))
    if not loaded:
        raise SystemExit("no ensemble systems to combine")
    # common valid times (each system's own newest cycle; 6-hourly at 00/06/12/18Z)
    vt = {s: pd.DatetimeIndex(pd.to_datetime(d["valid"], format="%Y-%m-%dT%H")) for s, d in loaded.items()}
    newest = max(v[0] for v in vt.values())
    stale = [s for s in loaded if (newest - vt[s][0]) > pd.Timedelta(hours=24)]
    for s in stale:
        print(f"  {s}: init {vt[s][0]} is more than 24 h older than the newest system; left out of the combination", flush=True)
    common = sorted(set.intersection(*[set(vt[s]) for s in loaded if s not in stale]))
    common = pd.DatetimeIndex([t for t in common if t >= newest])
    mm_sys = [s for s in ORDER if s in loaded and s not in stale]
    print(f"  combination: {mm_sys}, {len(common)} common 6-hourly times {common[0]} .. {common[-1]}", flush=True)

    def at(s, key):
        idx = [list(vt[s]).index(t) for t in common]
        return loaded[s][key][:, idx] if key in ("coast", "points") else loaded[s][key][idx]

    mm250 = np.mean([at(s, "p250").astype("float32") for s in mm_sys], axis=0)
    mm500 = np.mean([at(s, "p500").astype("float32") for s in mm_sys], axis=0)

    # ── loops ──
    loops = {"ens_mm_p250": dict(kind="prob", p=mm250, c=mm500, valid=list(common), clab="dashed: P(IVT ≥ 500) = 50 %",
                                 title=f"P(IVT ≥ 250), {len(mm_sys)} ensembles equally weighted",
                                 sub=f"{' + '.join(SYSTEMS[s]['short'] for s in mm_sys)}: the mean of each system's member fraction "
                                     "(each system counts once, whatever its size) · proxy IVT k·W·|V850|, calibrated per system"),
             "ens_mm_p500": dict(kind="prob", p=mm500, c=mm250, valid=list(common), clab="dashed: P(IVT ≥ 250) = 50 %",
                                 title=f"P(IVT ≥ 500), {len(mm_sys)} ensembles equally weighted",
                                 sub=f"{' + '.join(SYSTEMS[s]['short'] for s in mm_sys)}: moderate (AR2) transport or stronger"
                                     " · proxy IVT k·W·|V850|, calibrated per system")}
    for s, d in loaded.items():
        m = d["meta"]; init = vt[s][0]
        loops[f"ens_{s}_p250"] = dict(kind="prob", p=d["p250"], c=d["p500"], valid=list(vt[s]), clab="dashed: P(IVT ≥ 500) = 50 %",
                                      title=f"P(IVT ≥ 250) — {m['label']} {m['members']} members, init {init:%d %b %HZ}",
                                      sub=f"proxy IVT k·W·|V850| with k = {m['calibration']['k']:.3f} fitted this cycle on the "
                                          f"{m['calibration']['member']}'s exact IVT (r = {m['calibration']['r']:.2f})")
    for s in ("aifs", "ifs"):
        if s in loaded:
            d = loaded[s]; v = list(vt[s]); w = d["wmean"].astype("float32"); a = w - pwat_clim(v)
            for kind in ("pwat", "pwatanom"):
                loops[f"{kind}_{s}"] = dict(kind=kind, w=w, a=a, valid=v, label=f"{d['meta']['label']} {d['meta']['members']} members",
                                            init=v[0], sub="ECMWF total column water (vapour + cloud), ensemble mean, 0.5° · "
                                                           "anomaly vs ERA5 1991–2020 (3 harmonics, 1.5°) at the same hour")
    dirs = {}
    for key, L in loops.items():
        dd = ANIM / key; dd.mkdir(parents=True, exist_ok=True)
        for f in dd.glob("F*.webp"):
            f.unlink()
        L["dir"] = dd; dirs[key] = dd
        _R[key] = L
    jobs = [(key, k) for key, L in loops.items() for k in range(len(L["valid"]))]
    import multiprocessing as mp
    A.warm_features()
    t1 = time.time()
    with mp.get_context("fork").Pool(procs) as pool:
        pool.map(_render, jobs, chunksize=4)
    t_render = time.time() - t1
    regions = {}
    lab = {"ens_mm_p250": "All ensembles · P(IVT ≥ 250)", "ens_mm_p500": "All ensembles · P(IVT ≥ 500)"}
    for key, L in loops.items():
        init = L.get("init", L["valid"][0])
        s_ = key.split("_")[1] if key.startswith("ens_") else key.split("_")[1]
        name = lab.get(key) or (f"{SYSTEMS[s_]['short']} · P(IVT ≥ 250)" if key.startswith("ens_") else
                                ("Precipitable water" if key.startswith("pwat_") else "PWAT anomaly") + f", {SYSTEMS[s_]['short']}")
        regions[key] = {"label": name, "frames": [{"idx": k, "file": f"F{k:02d}.webp", "date": t.strftime("%Y-%m-%d"),
                                                   "label": f"{t:%a %d %b %HZ} · " + (f"+{int((t - vt[s_][0]) / pd.Timedelta(hours=1))} h {SYSTEMS[s_]['short']}" if key.startswith(("ens_a", "ens_i", "ens_g", "pwat")) else f"+{int((t - newest) / pd.Timedelta(hours=1))} h from {newest:%d %HZ}")}
                                                  for k, t in enumerate(L["valid"])]}
    mani = {"ver": int(time.time()), "default": "ens_mm_p250", "regions": regions, "data_dir": "ens_data"}
    (ANIM / "ar_ens_manifest.json").write_text(json.dumps(mani, separators=(",", ":")))

    # ── coastal figures + JSON ──
    figs = {}
    for s, d in loaded.items():
        m = d["meta"]
        out = ASSETS / f"ens_coast_{s}.webp"
        coast_figure(out, f"{m['label']} {m['members']} members", vt[s][0], vt[s], d["coast"].astype("float32"),
                     f"Read at the first ocean point west of the coast on each 0.25° latitude · proxy IVT (k = {m['calibration']['k']:.3f}, "
                     f"r = {m['calibration']['r']:.2f} against the {m['calibration']['member']}'s exact IVT)")
        figs[s] = out.name
    pts = A.coast_points()
    plume = {"points": [{"lat": p[0], "lon": round(p[1] - 360, 2), "name": p[2]} for p in pts], "systems": {}}
    days = sorted({t.strftime("%Y-%m-%d") for t in common})
    scale = {"days": days, "points": [p[2] for p in pts], "systems": {}}
    for s, d in loaded.items():
        v = vt[s]
        plume["systems"][s] = {"label": d["meta"]["label"], "init": d["meta"]["init"], "valid": [t.strftime("%Y-%m-%dT%HZ") for t in v],
                               "members": d["points"].transpose(2, 0, 1).tolist()}           # [point][member][step]
        # daily Ralph-scale probabilities per point
        P1, P3 = [], []
        for ip in range(len(pts)):
            rk = [A.daily_ranks(d["points"][m, :, ip], list(v)) for m in range(d["points"].shape[0])]
            P1.append([round(float(np.mean([r.get(dy, 0) >= 1 for r in rk])), 2) if dy in rk[0] else None for dy in days])
            P3.append([round(float(np.mean([r.get(dy, 0) >= 3 for r in rk])), 2) if dy in rk[0] else None for dy in days])
        scale["systems"][s] = {"ge1": P1, "ge3": P3, "label": d["meta"]["label"]}
    for key in ("ge1", "ge3"):
        mmv = []
        for ip in range(len(pts)):
            row = []
            for j in range(len(days)):
                vals = [scale["systems"][s][key][ip][j] for s in mm_sys if scale["systems"][s][key][ip][j] is not None]
                row.append(round(float(np.mean(vals)), 2) if len(vals) == len(mm_sys) else None)
            mmv.append(row)
        scale.setdefault("mm", {})[key] = mmv
    (ANIM / "ens_data").mkdir(parents=True, exist_ok=True)
    (ANIM / "ens_data" / "plume.json").write_text(json.dumps(plume, separators=(",", ":")))
    nb = {k: A.dir_bytes(v) for k, v in dirs.items()}
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "combination": mm_sys, "stale": stale,
           "common": [common[0].strftime("%Y-%m-%dT%HZ"), common[-1].strftime("%Y-%m-%dT%HZ")],
           "systems": {s: d["meta"] for s, d in loaded.items()}, "coast_figs": figs, "scale": scale,
           "bytes_frames": nb, "bytes_frames_total_MB": round(sum(nb.values()) / 1e6, 1),
           "render_s": round(t_render), "combine_s": round(time.time() - t0)}
    (ASSETS / "ar_ens.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {len(loops)} loops ({len(jobs)} frames, {sum(nb.values()) / 1e6:.1f} MB) in {t_render:.0f}s; ar_ens.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("system")
    a1.add_argument("--sys", required=True, choices=ORDER)
    a1.add_argument("--date"); a1.add_argument("--time")
    a1.add_argument("--work", default="/tmp/are")
    a2 = sp.add_parser("combine")
    a2.add_argument("--work", default="/tmp/are"); a2.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "system":
        run_system(a.sys, a.date or None, (f"{int(a.time):02d}" if a.time else None), work)
    else:
        combine(work, a.procs)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc or 0)
