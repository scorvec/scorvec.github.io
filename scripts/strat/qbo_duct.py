#!/usr/bin/env python3
"""E-P flux and planetary-wave driving in the stratosphere, from AIFS-ENS.

Per cycle:

assets/sst/anim/epflux/F##.webp (+ epflux_manifest.json)
    16-frame day-0..15 loop of the Eliassen-Palm flux and its divergence -
    the force planetary waves exert on the zonal-mean flow, i.e. what
    actually decelerates the polar vortex. Ensemble mean over the AIFS-ENS
    control + perturbed members; v and t are fetched through the shared
    ecmwf store, u rides the RMM/AAM pulls. Julia (CairoMakie) renders the
    frames, matplotlib re-renders any frame Julia drops.
    assets/strat/epflux.webp is a copy of the analysis frame.

assets/sst/anim/epflux_ifs/F##.webp (+ epflux_ifs_manifest.json), --model ifs
    The same loop from ALL 50 IFS-ENS perturbed members (2026-09-27, user: "We also add in the ifs-ens for all strat
    products"). u, v and t at the 14 levels are ~1.4 GB a step for 50 members, ~23 GB a cycle, so they are NOT
    cached: each step is byte-ranged from the Google Cloud mirror (rangefetch, under the shared request budget),
    decoded in memory, reduced to the per-member quadratics and dropped, a few steps in parallel. Open data has no
    IFS control at pressure levels, so step 0 is the members' initial states. Runs in strat-ifs.yml.

Fixed 2026-09-27: _band13 filtered along axis 1, which for the (member, lat, lon) blocks of the ensemble version is
LATITUDE, not longitude - so the "k = 1-3 eddies" were meridional modes 1-3 of the whole field, zonal mean included,
and a flow with no eddies at all produced a large u'v' and v'theta'. It now filters the last (longitude) axis.

Removed 2026-08-30: the QBO strip loop (assets/sst/anim/qbo_strip/) and the
wave_channel section. The strip rendered 16 frames and a manifest every cycle
but no page ever referenced it, and wave_channel had already lost its code -
only the docstring still advertised it.

    python scripts/strat/qbo_duct.py
    python scripts/strat/qbo_duct.py --model ifs --date 20260927 --time 00
"""
from __future__ import annotations
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE = REPO / "scripts" / "ecmwf" / "cache"
OUT_CHAN = REPO / "assets" / "strat" / "epflux.webp"
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf

LEVS_STRIP = [10, 50, 100]
UC1 = 28.0                                   # wave-1 Charney-Drazin ceiling (m/s)
KMAX = 3                                     # planetary wavenumbers for u'v'


def latest_cycle():
    for d in sorted(glob.glob(str(CACHE / "*z")), reverse=True):
        u_full = glob.glob(f"{d}/aifs-ens/cf_u_10-*x16.grib2")        # 12 lev, 0..360 h
        u_rmm = glob.glob(f"{d}/aifs-ens/cf_u_200-850_s0-*x16.grib2") # 200/850, 0..360 h
        # Gate on the u files ONLY. This used to also require the 14-level
        # analysis v (cf_v_10-*_s0-0x1), which the E-P flux never reads - it
        # fetches its own v and t through ecmwf.ensure. On a fresh runner the
        # store writes the 13-level analysis v (no 10 hPa), so that vestigial
        # condition matched nothing and the whole script bailed with
        # "no AIFS-ENS cycle with the needed files" while the data it actually
        # needed was sitting right there.
        if u_full:
            tag = Path(d).name
            return u_full[0], (sorted(u_rmm)[-1] if u_rmm else None), tag
    raise SystemExit("no AIFS-ENS cycle with the u files in the cache")












def _band13(f):
    """Zonal wavenumbers 1..KMAX of f(..., lat, lon): the LAST axis is longitude, whatever leads it (a member axis in
    the ensemble blocks). It was axis=1 until 2026-09-27, which on (member, lat, lon) is latitude."""
    F = np.fft.rfft(f, axis=-1)
    F[..., 0] = 0.0
    F[..., KMAX + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[-1], axis=-1)


def _quadratics(u, v, t, lev):
    """Per-member eddy quadratics for one (level, step), members averaged: [n, [u'v'], [v'th'], [th], [u]] by lat,
    plus, at the ZMU_LEVELS only, the PER-MEMBER zonal-mean u (member, lat) for the QBO zero-line tracker
    (qbo_zeroline.py; the same fields, so no extra download)."""
    th = t * (1000.0 / lev) ** 0.2854
    up, vp, thp = _band13(u), _band13(v), _band13(th)
    return [len(u),
            (up * vp).mean(axis=-1).mean(axis=0),
            (vp * thp).mean(axis=-1).mean(axis=0),
            th.mean(axis=-1).mean(axis=0),
            u.mean(axis=-1).mean(axis=0),
            u.mean(axis=-1).astype(np.float32) if lev in ZMU_LEVELS else None]


ZMU_LEVELS = (10, 50, 100)
LAST_ACC = {}                                         # (step, lev) -> quadratics of the last compute, for save_zmu


def save_zmu(acc, lat, steps, model, base):
    """Per-member zonal-mean u at ZMU_LEVELS -> data/zmu_members_<model>.npz for qbo_zeroline.py (best effort)."""
    try:
        arrs = {}
        for lev in ZMU_LEVELS:
            rows = [acc[(sh, lev)][5] if (sh, lev) in acc and len(acc[(sh, lev)]) > 5 else None for sh in steps]
            if any(r is None for r in rows):
                continue
            n = min(len(r) for r in rows)
            arrs[f"u{lev}"] = np.stack([r[:n] for r in rows])          # (step, member, lat)
        if not arrs:
            print("  zmu: nothing to save", flush=True); return
        out = HERE / "data" / f"zmu_members_{model}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, lat=np.asarray(lat, float), steps=np.asarray(steps, int), init=str(base)[:16], **arrs)
        print(f"  zmu: {out.name} {sorted(arrs)} {next(iter(arrs.values())).shape}", flush=True)
    except Exception as e:                                                   # noqa: BLE001
        print(f"  zmu: not saved ({str(e)[:80]})", flush=True)


def _open_lev(path, short, lev):
    """Lazy DataArray for one level of a (possibly multi-member) GRIB."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": short, "level": lev}, indexpath=""))
    return ds[short]


def compute_epflux_ensemble(paths, lat, steps):
    """Ensemble QG E-P flux: per-member quadratics accumulated per (level,
    step) — the flux is quadratic in the eddies, so ensemble products must
    average per-member fluxes, never take the flux of the ensemble mean
    (averaging damps the waves with lead time). Memory stays at one
    (member, lat, lon) block. Returns {step: (levs, p_pa, U, force)}."""
    acc = {}          # (step, lev) -> [n, uv, vth, th, ub]
    levs_seen = set()
    for lev in sorted(ecmwf.LEVELS_AAM):
        try:
            das = {k: _open_lev(*spec) for k, spec in paths[lev].items()}
        except Exception as e:                            # noqa: BLE001
            print(f"  epflux: level {lev} unavailable ({str(e)[:50]})", flush=True)
            continue
        levs_seen.add(lev)
        for si, sh in enumerate(steps):
            def at_step(da):
                d = da
                if "step" in d.dims:
                    d = d.sel(step=pd.Timedelta(hours=sh))
                a = d.values
                return a if a.ndim == 3 else a[None]      # (member, lat, lon)
            try:
                u = np.concatenate([at_step(das["u_cf"]), at_step(das["u_pf"])])
                v = np.concatenate([at_step(das["v_cf"]), at_step(das["v_pf"])])
                t = np.concatenate([at_step(das["t_cf"]), at_step(das["t_pf"])])
            except Exception as e:                        # noqa: BLE001
                print(f"  epflux: lev {lev} step {sh} skipped ({str(e)[:50]})", flush=True)
                continue
            n = min(len(u), len(v), len(t))
            acc[(sh, lev)] = _quadratics(u[:n], v[:n], t[:n], lev)
        for da in das.values():
            da.close()
    LAST_ACC.clear(); LAST_ACC.update(acc)
    return _finish(acc, levs_seen, lat, steps)


def _finish(acc, levs_seen, lat, steps):
    """Accumulated quadratics -> {step: (levs, p_pa, U, force, Fphi, Fp, th_prof)}, member count."""
    A = 6.371e6
    OMEGA = 7.292e-5
    latr = np.deg2rad(lat)
    cosp = np.cos(latr)[None, :]
    f_cor = 2 * OMEGA * np.sin(latr)[None, :]
    w = np.cos(latr)
    out = {}
    nmem = 0
    for sh in steps:
        levs = np.array(sorted(l for l in levs_seen if (sh, l) in acc), float)
        if len(levs) < 8:
            continue
        UV = np.stack([acc[(sh, l)][1] for l in levs])
        VTH = np.stack([acc[(sh, l)][2] for l in levs])
        TH = np.stack([acc[(sh, l)][3] for l in levs])
        U = np.stack([acc[(sh, l)][4] for l in levs])
        nmem = acc[(sh, levs[0])][0]
        p_pa = (levs * 100.0)[:, None]
        th_prof = (TH * w[None, :]).sum(axis=1) / w.sum()
        dthdp = np.gradient(th_prof, p_pa[:, 0])[:, None]
        dthdp = np.where(dthdp > -5e-5, -5e-5, dthdp)
        Fphi = -A * cosp * UV
        Fp = A * cosp * f_cor * VTH / dthdp
        cosp_safe = np.clip(cosp, np.cos(np.deg2rad(85.0)), None)
        dFphi = np.gradient(Fphi * cosp, latr, axis=1) / (A * cosp_safe)
        dFp = np.gradient(Fp, axis=0) / np.gradient(p_pa, axis=0)
        force = (dFphi + dFp) / (A * cosp_safe) * 86400.0
        force = ndi.gaussian_filter(force, sigma=(0.6, 10))
        force[:, np.abs(lat) > 82] = np.nan
        out[sh] = (levs, p_pa, U, force, Fphi, Fp, th_prof)
    return out, nmem


def cd_ceiling_excess(U, lat, levs, th_prof, k=1, Hs=7000.0):
    """u - U_c for stationary zonal wavenumber k, plane-wave Charney-Drazin
    form: U_c = beta / [ (k/(a cos))^2 + l^2 + f^2/(4 N^2 H^2) ] with the
    standard meridional scale l = 2/a (which puts U_c(60 deg) ~ 28 m/s).
    The zero contour is the propagation ceiling; with the u = 0 line it
    brackets the corridor 0 < u < U_c where wave-k can propagate vertically.
    Chosen over the full Matsuno n^2 = 0 after validation: a strong jet
    sharpens its own PV gradient, so the full index stays positive over the
    jet core and its zero line marks flank reflecting pockets, not the lid.
    NaN in the tropics (QG invalid) and below 400 hPa (off-story)."""
    A = 6.371e6
    OMEGA = 7.292e-5
    G = 9.80665
    latr = np.deg2rad(lat)
    cosp = np.clip(np.cos(latr), 5e-3, None)[None, :]
    f = 2 * OMEGA * np.sin(latr)[None, :]
    beta = 2 * OMEGA * cosp / A
    z = -Hs * np.log(levs / 1000.0)
    N2 = G / th_prof * np.gradient(th_prof, z)
    N2 = np.clip(N2, 5e-6, None)[:, None]
    Uc = beta / ((k / (A * cosp)) ** 2 + (2.0 / A) ** 2
                 + f ** 2 / (4.0 * N2 * Hs ** 2))
    Us = ndi.gaussian_filter1d(U, sigma=8, axis=1, mode="nearest")
    out = Us - Uc
    out[:, np.abs(lat) < 20] = np.nan
    out[:, np.abs(lat) > 82] = np.nan   # beta->0: Uc collapses at the poles
    out[levs > 400] = np.nan
    return out


IFS_LEVELS = tuple(ecmwf.LEVELS_AAM)                  # 10..1000 hPa, all 14 on IFS open data
IFS_LON_STRIDE = 4    # 0.25 -> 1 deg in longitude before the FFT: k = 1-3 alias only from k ~ 357-363, which is nil


def _ifs_step(job):
    """One IFS-ENS step: byte-range u, v, t at the 14 levels for the 50 perturbed members from the Google mirror
    (rangefetch, under the shared request budget), decode in memory, return {lev: quadratics} and the latitudes.
    Nothing is written to disk; the ~1.4 GB of messages is dropped when the step is reduced."""
    import eccodes
    import rangefetch as rf
    date, time_, sh = job
    t0 = time.time()
    idx = rf.fetch_index(date, time_, "ifs", sh, "ef", sources=["google"])
    want = sorted(rf.select(idx, param=["u", "v", "t"], levelist=list(IFS_LEVELS), type="pf"),
                  key=lambda e: e["_offset"])
    blob = rf.fetch_ranges(rf.path_for(date, time_, "ifs", sh, "ef") + ".grib2", rf.coalesce(want),
                           sources=["google"])
    t1 = time.time()
    where, pos = {}, 0                                    # fetch_ranges returns the messages in offset order
    for e in want:
        where.setdefault(int(e["levelist"]), {}).setdefault(e["param"], []).append((int(e["number"]), pos, e["_length"]))
        pos += e["_length"]
    if pos != len(blob):
        raise RuntimeError(f"step {sh}: {len(blob)} bytes for {pos} indexed")
    # The ef files are in ARRIVAL order and the member order differs between parameters (measured 2026-09-28: u and v
    # at 250 hPa part at the 12th message, t follows a different order altogether), so rows are keyed by member number
    # and stacked in number order. Until 2026-09-28 they were stacked in file order, pairing u, v and t of DIFFERENT
    # members in the quadratics (u'v', v'T') of some members.
    jet = os.environ.get("PACJET_IFS_DIR")                # the North Pacific jet product reads 250 hPa u/v from this stream
    lat = lon = sl = None
    out = {}
    for lev in IFS_LEVELS:
        f = {}
        keep = {}
        for par in ("u", "v", "t"):
            rows = {}
            for num, o, n in where.get(lev, {}).get(par, []):
                gid = eccodes.codes_new_from_message(bytes(blob[o:o + n]))
                try:
                    nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
                    if lat is None:
                        la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                        la1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                        lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                        lat = np.linspace(la0, la1, nj)
                        lon = (lo0 + (360.0 / ni) * np.arange(ni)) % 360
                    full = eccodes.codes_get_values(gid).reshape(nj, ni)
                    if jet and lev == 250 and par in ("u", "v"):
                        if sl is None:                    # crop to the jet sector at once (50 global fields are 0.4 GB)
                            sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
                            import pacjet_core as PJ
                            sl = PJ.native_slices(lat, lon)
                        keep.setdefault(par, {})[num] = full[np.ix_(sl[0], sl[1])].astype("float32")
                    rows[num] = full[:, ::IFS_LON_STRIDE]
                finally:
                    eccodes.codes_release(gid)
            f[par] = rows
        common = sorted(set(f["u"]) & set(f["v"]) & set(f["t"]))
        if not common:
            continue
        out[lev] = _quadratics(np.stack([f["u"][k] for k in common]), np.stack([f["v"][k] for k in common]),
                               np.stack([f["t"][k] for k in common]), lev)
        if keep and set(keep) == {"u", "v"}:
            both = sorted(set(keep["u"]) & set(keep["v"]))
            try:
                ii, jj = np.arange(sl[0].size), np.arange(sl[1].size)
                Path(jet).mkdir(parents=True, exist_ok=True)
                np.savez(Path(jet) / f"step_{sh:03d}.npz", members=np.array(both),
                         u=PJ.to_work(np.stack([keep["u"][k] for k in both]), ii, jj).astype("float16"),
                         v=PJ.to_work(np.stack([keep["v"][k] for k in both]), ii, jj).astype("float16"))
            except Exception as e:                        # noqa: BLE001 - the jet dump must never cost the E-P flux
                print(f"  pacjet dump step {sh} failed: {str(e)[:100]}", flush=True)
    del blob
    print(f"  IFS-ENS epflux step {sh:3d}: {pos / 1e6:,.0f} MB in {t1 - t0:.0f} s, reduced in {time.time() - t1:.0f} s "
          f"({len(out)} levels, {out[next(iter(out))][0] if out else 0} members)", flush=True)
    return sh, lat, out


def compute_epflux_ifs(date, time_, steps, workers=None):
    """The IFS-ENS counterpart of compute_epflux_ensemble, streamed (see _ifs_step)."""
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or int(os.environ.get("EPFLUX_IFS_WORKERS", "3"))
    acc, levs_seen, lat = {}, set(), None
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sh, la, per in ex.map(_ifs_step, [(date, time_, s) for s in steps]):
            if la is not None and lat is None:
                lat = la
            for lev, q in per.items():
                acc[(sh, lev)] = q
                levs_seen.add(lev)
    if lat is None:
        raise SystemExit("IFS-ENS epflux: no step decoded")
    LAST_ACC.clear(); LAST_ACC.update(acc)
    by_step, nmem = _finish(acc, levs_seen, lat, steps)
    return by_step, nmem, lat


EP_STAGING = HERE / "data" / ".epflux_staging"
JULIA_SCRIPT = REPO / "scripts" / "julia" / "mjo_render.jl"


def _cmap_hex(name, n):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap(name)
    return [mcolors.to_hex(cmap((k + 0.5) / n)) for k in range(n)]


def epflux_loop(u_full, u_rmm, tag, base):
    """16-frame day-0..15 E-P flux loop. Julia (CairoMakie, the shared
    mjo_render.jl rasterizer) renders the frames; matplotlib re-renders any
    frame Julia drops. Figure text is a title + one credit line — the reading
    guide lives in the page captions."""
    import subprocess
    from PIL import Image
    cyc = ecmwf.Cycle(tag[:8], tag[8:10])
    S = tuple(ecmwf.STEPS)
    L14 = tuple(ecmwf.LEVELS_AAM)
    L12 = tuple(ecmwf.LEVELS_AAM_REST)
    NM = ecmwf.AAM_PF_MEMBERS
    # control + perturbed members; u rides the RMM/AAM pulls, v & t are ours
    fetch = dict(
        v_cf=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", L14, S)),
        t_cf=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "t", "pl", L14, S)),
        v_pf=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "v", "pl", L14, S, NM)),
        t_pf=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "t", "pl", L14, S, NM)),
        u_cf12=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", L12, S)),
        u_pf12=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", L12, S, NM)),
        u_cf2=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", ecmwf.LEVELS_RMM, S)),
        u_pf2=ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", ecmwf.LEVELS_RMM, S)),
    )
    paths = {lev: dict(
        u_cf=(fetch["u_cf2" if lev in (200, 850) else "u_cf12"], "u", lev),
        u_pf=(fetch["u_pf2" if lev in (200, 850) else "u_pf12"], "u", lev),
        v_cf=(fetch["v_cf"], "v", lev), v_pf=(fetch["v_pf"], "v", lev),
        t_cf=(fetch["t_cf"], "t", lev), t_pf=(fetch["t_pf"], "t", lev),
    ) for lev in L14}
    with xr.open_dataset(fetch["v_cf"], engine="cfgrib", backend_kwargs=dict(
            filter_by_keys={"shortName": "v", "level": 500}, indexpath="")) as ds:
        lat = ds.latitude.values

    by_step, nmem = compute_epflux_ensemble(paths, lat, S)
    save_zmu(LAST_ACC, lat, S, "aifs", base)
    render_epflux(by_step, nmem, lat, S, base, "aifs")


MODEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
SOURCE = {"aifs": "ECMWF AIFS-ENS open data (CC BY 4.0)",
          "ifs": "ECMWF IFS-ENS open data (CC BY 4.0), 50 members"}      # longer overflows the frame footer


def render_epflux(by_step, nmem, lat, S, base, model="aifs"):
    """Frames + manifest for one model's loop: assets/sst/anim/epflux[_ifs]/ and epflux[_ifs]_manifest.json."""
    import subprocess
    from PIL import Image
    region = "epflux" if model == "aifs" else f"epflux_{model}"
    staging = EP_STAGING if model == "aifs" else HERE / "data" / f".{region}_staging"
    outdir = REPO / "assets" / "sst" / "anim" / region
    outdir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    for f in staging.glob("*"):
        f.unlink()
    per_step = [(s, by_step[s]) for s in S if s in by_step]
    levs0, p0, _, force0, _, _, _ = per_step[0][1]
    strat = levs0 <= 300
    cap = max(2.0, float(np.ceil(np.nanpercentile(np.abs(
        np.stack([r[1][3][strat] for r in per_step])), 98))))
    ncols = 20
    fill_cols = _cmap_hex("PuOr_r", ncols)
    fill_levels = list(np.linspace(-cap, cap, ncols + 1))

    yticks = [1000, 700, 500, 300, 200, 100, 50, 10]
    frames_meta, frames = [], []
    sk = 24
    for idx, (s, (levs, p_pa, U, force, Fphi, Fp, th_prof)) in enumerate(per_step):
        valid = base + pd.Timedelta(hours=s)
        # EP arrows: display-normalized; vertical displacement scaled by the
        # local pressure so arrow length is uniform on the log-p axis
        Lq = lat[::sk]
        Fx = Fphi[:, ::sk]
        Fyp = (-Fp / p_pa)[:, ::sk]
        fx_ref = np.nanpercentile(np.abs(Fx), 95) or 1.0
        fy_ref = np.nanpercentile(np.abs(Fyp), 95) or 1.0
        fxn = np.clip(Fx / fx_ref, -1.6, 1.6)
        fyn = np.clip(Fyp / fy_ref, -1.6, 1.6)
        keep = np.hypot(fxn, fyn) > 0.15
        Xq = np.broadcast_to(Lq[None, :], fxn.shape)[keep]
        Yq = np.broadcast_to(levs[:, None], fxn.shape)[keep]
        du = fxn[keep] * 5.5                              # degrees latitude
        dv = -Yq * (np.expm1(fyn[keep] * 0.28))           # log-p displacement, up = -dp
        fid = f"F{idx:02d}"
        arrays = dict(lat=lat, lev=levs, force=np.nan_to_num(force, nan=0.0),
                      u=U, cdlid=cd_ceiling_excess(U, lat, levs, th_prof),
                      qx=Xq, qy=Yq, qu=du, qv=dv)
        meta = dict(
            out_png=fid + ".png", figsize=[12.0, 7.0],
            title=(f"E\u2013P flux & wave driving (wavenumbers 1\u20133) \u2014 "
                   f"{MODEL[model]} ensemble ({nmem} members) \u00b7 day {s // 24} \u00b7 valid {valid:%a %b %d %HZ}"),
            footer=f"{SOURCE[model]} \u00b7 QG E\u2013P flux, k=1\u20133 \u00b7 shading: \u2207\u00b7F as zonal force (m/s/day) \u00b7 dashed magenta: \u016b = U_c, the wave-1 Charney\u2013Drazin ceiling (l = 2/a) \u00b7 poleward of 82\u00b0 masked",
            xlabel="latitude", ylabel="pressure (hPa)",
            ylog=True, yreversed=True,
            xlim=[float(lat.min()), float(lat.max())], ylim=[1000.0, 10.0],
            xticks=list(range(-80, 81, 20)),
            xticklabels=[f"{abs(t)}\u00b0{'S' if t < 0 else 'N' if t > 0 else ''}"
                         for t in range(-80, 81, 20)],
            yticks=yticks,
            fill=dict(npz="force", x="lat", y="lev", levels=fill_levels,
                      colors=fill_cols,
                      cbar_label="E\u2013P flux divergence (m s\u207b\u00b9 day\u207b\u00b9)"),
            contours=[dict(npz="u", levels=[float(l) for l in range(-80, 81, 10) if l], 
                           color="#404040", width=0.7, dash=False),
                      dict(npz="u", levels=[0.0], color="#000000", width=2.0, dash=False),
                      dict(npz="cdlid", levels=[0.0], color="#c2185b", width=1.7, dash=True)],
            arrows=dict(x="qx", y="qy", u="qu", v="qv", scale=1.0),
            texts=[], frame_id=fid)
        np.savez_compressed(staging / f"{fid}.npz",
                            **{k: np.asarray(v, np.float64) for k, v in arrays.items()})
        frames_meta.append(meta)
        frames.append({"idx": idx, "file": f"{fid}.webp",
                       "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s // 24} \u00b7 {valid:%b %d}"})

    spec = dict(staging=str(staging), frames=frames_meta)
    (staging / "spec.json").write_text(json.dumps(spec))
    ok = False
    if shutil.which("julia"):
        try:
            r = subprocess.run(
                ["julia", "--project=" + str(JULIA_SCRIPT.parent),
                 str(JULIA_SCRIPT), str(staging / "spec.json")],
                capture_output=True, text=True, timeout=1800)
            ok = r.returncode == 0 and "JULIA RENDER DONE" in r.stdout
            if not ok:
                print(f"  epflux julia failed:\n{(r.stderr or r.stdout)[-400:]}", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"  epflux julia failed ({str(e)[:80]})", flush=True)

    n_jl = 0
    for meta, (s, fields) in zip(frames_meta, per_step):
        fid = meta["frame_id"]
        png = staging / f"{fid}.png"
        out = outdir / f"{fid}.webp"
        if ok and png.exists():
            Image.open(png).convert("RGB").save(out, quality=84, method=6)
            png.unlink()
            n_jl += 1
        else:
            _epflux_frame_mpl(meta, fields, lat, out, staging)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {region: {
               "label": f"E\u2013P flux & wave driving (k=1\u20133) \u2014 {MODEL[model]} ensemble",
               "n_frames": len(frames), "frames": frames}}}
    (REPO / "assets" / "sst" / "anim" / f"{region}_manifest.json").write_text(json.dumps(man))
    if model != "aifs":
        print(f"wrote {len(frames)} {region} frames ({n_jl} via julia) + manifest")
        return
    # assets/strat/ is gitignored (the local page), so on a fresh runner the
    # directory does not exist and copy2 raised FileNotFoundError AFTER all 16
    # frames had already rendered - the loop was fine, the still was not.
    OUT_CHAN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outdir / "F00.webp", OUT_CHAN)
    print(f"wrote {len(frames)} epflux frames ({n_jl} via julia) + manifest")


def _epflux_frame_mpl(meta, fields, lat, out, staging=EP_STAGING):
    """Matplotlib fallback for a single staged frame (same spec/arrays)."""
    levs, p_pa, U, force, Fphi, Fp, th_prof = fields
    z = np.load(staging / f"{meta['frame_id']}.npz")
    fig, ax = plt.subplots(figsize=meta["figsize"])
    f = meta["fill"]
    cf = ax.contourf(z["lat"], z["lev"], z["force"], levels=f["levels"],
                     colors=f["colors"], extend="both")
    for c in meta["contours"]:
        ax.contour(z["lat"], z["lev"], z[c["npz"]], levels=c["levels"],
                   colors=c["color"], linewidths=c["width"],
                   linestyles="dashed" if c.get("dash") else "solid")
    ax.quiver(z["qx"], z["qy"], z["qu"], z["qv"], angles="xy",
              scale_units="xy", scale=1.0, width=0.0022,
              facecolor="#111111", edgecolor="white", linewidth=0.3,
              pivot="mid", alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1000, 10)
    ax.set_yticks(meta["yticks"])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(meta["xticks"]); ax.set_xticklabels(meta["xticklabels"], fontsize=9)
    ax.set_xlim(*meta["xlim"])
    ax.set_xlabel(meta["xlabel"]); ax.set_ylabel(meta["ylabel"])
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(f["cbar_label"], fontsize=9); cb.ax.tick_params(labelsize=8)
    ax.set_title(meta["title"], fontsize=12.5, fontweight="bold", loc="left")
    fig.text(0.5, 0.005, meta["footer"], fontsize=8, ha="center", va="bottom",
             color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=115)
    plt.close(fig)


def main():
    import argparse
    # --epflux-only/--strip-only are accepted and ignored: the QBO strip was
    # removed on 2026-08-30 (nothing on the site ever displayed it), so the
    # E-P flux loop is the only product left. Kept as no-ops so an old caller
    # does not hard-fail on an unknown flag.
    ap = argparse.ArgumentParser()
    ap.add_argument("--epflux-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--strip-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=("aifs", "ifs"), default="aifs")
    ap.add_argument("--date"); ap.add_argument("--time")
    args = ap.parse_args()
    if args.strip_only:
        print("--strip-only: the QBO strip was removed; nothing to do")
        return
    if args.model == "ifs":
        sys.path.insert(0, str(HERE))
        import ifs_ens as IE
        if not (args.date and args.time):
            ap.error("--model ifs needs --date and --time")
        if not IE.published(args.date, args.time):
            raise SystemExit(f"IFS-ENS {args.date} {args.time}Z is not on the Google mirror yet")
        base = pd.Timestamp(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]} {args.time}:00")
        S = tuple(ecmwf.STEPS)
        t0 = time.time()
        by_step, nmem, lat = compute_epflux_ifs(args.date, args.time, S)
        print(f"  IFS-ENS epflux: {len(by_step)} steps, {nmem} members, {time.time() - t0:.0f} s", flush=True)
        save_zmu(LAST_ACC, lat, S, "ifs", base)
        render_epflux(by_step, nmem, lat, S, base, "ifs")
        return
    u_full, u_rmm, tag = latest_cycle()
    base = pd.Timestamp(f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00")
    epflux_loop(u_full, u_rmm, tag, base)


if __name__ == "__main__":
    main()
