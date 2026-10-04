#!/usr/bin/env python3
"""E-P flux and planetary-wave driving in the stratosphere, from the NOAA GEFS ensemble; plus the per-member zonal-mean
u that the QBO zero-line tracker reads from AIFS-ENS and IFS-ENS.

--model gefs  (the E-P flux product, since 2026-09-30)
assets/sst/anim/epflux_gefs/F##.webp (+ epflux_gefs_manifest.json)
    16-frame day-0..15 loop of the Eliassen-Palm flux and its divergence - the force planetary waves exert on the
    zonal-mean flow, i.e. what decelerates the polar vortex - from the GEFS control + 30 perturbed members.
    User decision 2026-09-30: "switch models to one with more levels in the stratosphere". ECMWF open data has no
    levels between 10 and 50 hPa, so the old AIFS-ENS / IFS-ENS loops took the stratospheric divergence as a centred
    difference over 10-100 hPa and a one-sided one at 10 hPa. GEFS (noaa-gefs-pds on S3, pgrb2a + pgrb2b 0.5 deg)
    carries u, v and T at 31 levels: 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150 ... 1000 hPa. Every message is
    byte-ranged through the .idx from the S3 mirror (the site rule for NOAA models: never NOMADS).
    Measured 2026-09-30: 93 messages, ~17 MB per member-step, ~8.4 GB per cycle for 31 members x 16 daily steps.

THE MATHS (quasi-geostrophic, pressure coordinates; Edmon, Hoskins and McIntyre 1980; Andrews, Holton and Leovy 1987)
    F_phi = -a cos(phi) [u'v']             F_p = a cos(phi) f [v'theta'] / [theta]_p
    div F = 1/(a cos phi) d(F_phi cos phi)/d phi + dF_p/dp,      zonal force = div F / (a cos phi)  (m/s/day)
  * [theta]_p is the ZONAL-MEAN static stability (phi, p) since 2026-10-01 (EPFLUX_THETA_P=global restores the global
    mean, a profile in p only). Reason: above the polar tropopause (250-300 hPa poleward of ~70 deg) the global mean
    carries TROPOSPHERIC stability, 3-4x too small, which inflated F_p and its p-derivative there; and against the GEOS FP
    analysis budget (momentum_budget.py, primitive-equation EP divergence, 2026-09-26 daily mean, 100-2 hPa, 40-80 deg)
    the pattern correlation rose SH 0.83 -> 0.87, NH 0.69 -> 0.81.
  * Below ground: GEFS fills levels under the surface by extrapolation. PRES:surface (pgrb2a, ~0.25 MB per member-step)
    gives each member's share of below-ground longitudes per (lat, level); where the member mean exceeds BG_FRAC = 0.10
    the eddy fluxes are NaN before any derivative (so the level just above, whose centred difference would reach into
    extrapolated data, goes too) and the region is drawn grey: Antarctica (plateau 600-750 hPa), Greenland, the Andes,
    Tibet.
  * Poles: no extra smoothing. The divergence is not singular (u'v' -> 0 at the pole and the merid. term reduces to
    -(1/(a cos^2)) d(cos^2 [u'v'])/dphi); against GEOS FP the 75-82 deg band means agree in the stratosphere (e.g. SH
    1-2 hPa 75-82S +5..+10 GEFS vs +12..+17 GEOS FP m/s/day). Large day-0 values at the polar tropopause are a single
    near-deterministic snapshot of a high-latitude trough (k1-3 [u'v'] ~ -150 m2/s2 at 75-78N 250 hPa on 2026-10-01
    12Z); by day 5 the member average brings them to a few m/s/day. Mask stays at 82 deg; smoothing is NaN-aware and the
    polar rows are blanked BEFORE it, so nothing leaks in from the pole.
  * Eddies are zonal wavenumbers 1-3 (FFT along longitude), or all wavenumbers with --waves all.
  * Ensemble: the flux is quadratic in the eddies, so the per-member quadratics are averaged; the flux of the
    ensemble mean would fade with lead as the members decorrelate.
  * Vertical derivatives are taken in ln p with numpy's non-uniform second-order differences on the ACTUAL
    coordinate: dF_p/dp = (1/p) dF_p/dln p and theta_p = (1/p) dtheta/dln p. (Until 2026-09-30 dF_p/dp was a
    ratio of index gradients and theta_p a gradient in p.)
  * Smoothing: latitude only, a Gaussian of SMOOTH_DEG degrees in physical units; none in the vertical. (Was a
    sigma of 0.6 LEVEL INDICES, which on the uneven grid meant 6 hPa near the ground and 30+ hPa aloft.)
  * The top level (1 hPa) has only a one-sided derivative: the 1-2 hPa layer is hatched on the frames.
  * Poleward of 82 deg is masked (the 1/cos phi factors).
  * Arrows, Edmon et al. (1980) display scaling written for a log-p axis (Jucker 2021): in axis-fraction units the
    components are ( p F_phi / (a * dphi_axis),  -F_p / ln(p_bottom/p_top) ), dphi_axis the plotted latitude span in
    radians, so that their visual divergence is p div F; both are then multiplied by 1000 hPa / p (exp(z/H), so the
    stratosphere is legible) and by ONE fixed factor ARROW_K for every frame and every cycle, so lengths compare from
    day to day and run to run. Directions stay exact. Arrows longer than ARROW_CAP are drawn at ARROW_CAP in red.
    ARROW_K was set on 2026-09-30 00Z (SH spring, no arrow capped) and 2026-01-15 00Z (NH winter, k=1: none capped;
    k=1-3: 21 capped, all at 1-10 hPa over 45-75N).
  * The dashed magenta line is u = U_c, the wave-1 Charney-Drazin ceiling (cd_ceiling_excess) on the same levels.

--model aifs / --model ifs  (zonal-mean u only, since 2026-09-30)
    The AIFS-ENS and IFS-ENS E-P flux loops are RETIRED (user decision 2026-09-30). What survives is what other
    products read from their stream: data/zmu_members_<model>.npz, the per-member zonal-mean u at 10/50(/100) hPa
    for qbo_zeroline.py. aifs reads the u files ens_cycle --only strat already cached and ensures v at 250 hPa
    (control + 25) for pacjet.py; ifs streams only u at 10/50/250 and v at 250 hPa for the 50 members (~2 GB a cycle
    instead of ~23 GB for u/v/t at 14 levels) and leaves the 250 hPa sector in PACJET_IFS_DIR for pacjet.py.

Fixed 2026-09-27: _band13 filtered along axis 1, which for (member, lat, lon) blocks is LATITUDE.

    python scripts/strat/qbo_duct.py --model gefs --date 20260930 --time 00 [--days 15] [--waves 1-3|all]
    python scripts/strat/qbo_duct.py --model aifs --date 20260930 --time 00
    python scripts/strat/qbo_duct.py --model ifs  --date 20260930 --time 00
"""
from __future__ import annotations
import datetime as dt
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE = REPO / "scripts" / "ecmwf" / "cache"
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))

A = 6.371e6
OMEGA = 7.292e-5
KAPPA = 0.2854
KMAX = 3                                     # planetary wavenumbers 1..KMAX (0 = every wavenumber)
SMOOTH_DEG = 2.5                             # latitude smoothing of the divergence (Gaussian sigma, degrees)
POLE_MASK = 82.0
ZMU_LEVELS = (10, 50, 100)


# ------------------------------------------------------------------------------------------------------ eddies ---
def _band13(f):
    """Zonal wavenumbers 1..KMAX of f(..., lat, lon) (all non-zero wavenumbers when KMAX == 0): the LAST axis is
    longitude, whatever leads it. It was axis=1 until 2026-09-27, which on (member, lat, lon) is latitude."""
    F = np.fft.rfft(f, axis=-1)
    F[..., 0] = 0.0
    if KMAX:
        F[..., KMAX + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[-1], axis=-1)


def _quadratics(u, v, t, lev):
    """Eddy quadratics for one (level, step) of a (member, lat, lon) block, members averaged:
    [n, [u'v'], [v'th'], [th], [u]] by latitude."""
    th = t * (1000.0 / lev) ** KAPPA
    up, vp, thp = _band13(u), _band13(v), _band13(th)
    return [len(u),
            (up * vp).mean(axis=-1).mean(axis=0),
            (vp * thp).mean(axis=-1).mean(axis=0),
            th.mean(axis=-1).mean(axis=0),
            u.mean(axis=-1).mean(axis=0)]


def _nan_smooth_lat(f, sigma_pts):
    """Gaussian along latitude that ignores NaN (normalised convolution); NaN stays NaN."""
    ok = np.isfinite(f)
    num = ndi.gaussian_filter1d(np.where(ok, f, 0.0), sigma_pts, axis=1, mode="nearest")
    den = ndi.gaussian_filter1d(ok.astype(float), sigma_pts, axis=1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


BG_FRAC = 0.10           # a (lat, level) is below ground when more than this share of its longitudes is
THETA_P = os.environ.get("EPFLUX_THETA_P", "zonal")       # "zonal" (local [theta]_p, default since 2026-10-01) | "global"


def _finish(acc, levs_seen, lat, steps):
    """Accumulated quadratics -> {step: (levs, p_pa, U, force, Fphi, Fp, th_prof, below)}, member count.

    Levels ascend in pressure (top first). Vertical derivatives in ln p on the actual coordinate; latitude-only
    smoothing in degrees (NaN-aware); the top level is one-sided (hatched by the renderer). Where the ensemble-mean
    share of below-ground longitudes exceeds BG_FRAC the eddy fluxes are set to NaN before any derivative, so the
    mask also blanks the level just above (its centred difference would reach into extrapolated data)."""
    latr = np.deg2rad(lat)
    cosp = np.cos(latr)[None, :]
    f_cor = 2 * OMEGA * np.sin(latr)[None, :]
    w = np.cos(latr)
    dlat = float(np.abs(np.diff(lat)).mean())
    out, nmem = {}, 0
    for sh in steps:
        levs = np.array(sorted(l for l in levs_seen if (sh, l) in acc), float)
        if len(levs) < 8:
            continue
        UV = np.stack([acc[(sh, l)][1] for l in levs])
        VTH = np.stack([acc[(sh, l)][2] for l in levs])
        TH = np.stack([acc[(sh, l)][3] for l in levs])
        U = np.stack([acc[(sh, l)][4] for l in levs])
        below = (np.stack([acc[(sh, l)][5] for l in levs]) if len(acc[(sh, levs[0])]) > 5
                 else np.zeros_like(U))
        nmem = acc[(sh, levs[0])][0]
        p = levs * 100.0
        lnp = np.log(p)
        th_prof = (TH * w[None, :]).sum(axis=1) / w.sum()
        if THETA_P == "zonal":
            dthdp = np.gradient(TH, lnp, axis=0) / p[:, None]
        else:
            dthdp = (np.gradient(th_prof, lnp) / p)[:, None]       # theta_p = (1/p) dtheta/dln p
        dthdp = np.where(dthdp > -5e-5, -5e-5, dthdp)              # never neutral or unstable
        bg = below > BG_FRAC
        UV = np.where(bg, np.nan, UV)
        VTH = np.where(bg, np.nan, VTH)
        Fphi = -A * cosp * UV
        Fp = A * cosp * f_cor * VTH / dthdp
        cosp_safe = np.clip(cosp, np.cos(np.deg2rad(85.0)), None)
        dFphi = np.gradient(Fphi * cosp, latr, axis=1) / (A * cosp_safe)
        dFp = np.gradient(Fp, lnp, axis=0) / p[:, None]           # dF_p/dp = (1/p) dF_p/dln p
        force = (dFphi + dFp) / (A * cosp_safe) * 86400.0
        force[:, np.abs(lat) > POLE_MASK] = np.nan                 # before smoothing: nothing leaks in from the pole
        force = _nan_smooth_lat(force, SMOOTH_DEG / dlat)
        out[sh] = (levs, p[:, None], U, force, Fphi, Fp, th_prof, below)
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
    NaN in the tropics (QG invalid) and below 400 hPa (off-story).
    Latitude smoothing in degrees (2 deg sigma), so it reads the same on any grid."""
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
    dlat = float(np.abs(np.diff(lat)).mean())
    Us = ndi.gaussian_filter1d(U, sigma=2.0 / dlat, axis=1, mode="nearest")
    out = Us - Uc
    out[:, np.abs(lat) < 20] = np.nan
    out[:, np.abs(lat) > POLE_MASK] = np.nan   # beta->0: Uc collapses at the poles
    out[levs > 400] = np.nan
    return out


# -------------------------------------------------------------------------------------------------------- GEFS ---
GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
_MB = re.compile(r"^(\d+(?:\.\d+)?) mb$")


def _get(url, rng=None, tries=5):
    import urllib.request
    for k in range(tries):
        try:
            h = {"User-Agent": "scorvec-epflux"}
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except Exception:                                             # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def gefs_url(date, hh, mem, kind, h):
    return f"{GEFS_S3}/gefs.{date}/{hh}/atmos/pgrb2{kind}p5/{mem}.t{hh}z.pgrb2{kind}.0p50.f{h:03d}"


def _gefs_index(url):
    """[(start, end, var, level hPa)] for u/v/T on isobaric levels, plus (.., "PRES", "sfc") for surface pressure;
    end -1 = to EOF."""
    lines = [l.split(":") for l in _get(url + ".idx").decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        m = _MB.match(p[4])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
        if p[3] in ("UGRD", "VGRD", "TMP") and m:
            out.append((int(p[1]), end, p[3], float(m.group(1))))
        elif p[3] == "PRES" and p[4] == "surface":          # pgrb2a: for the below-ground mask
            out.append((int(p[1]), end, "PRES", "sfc"))
    return out


def _gefs_job(job):
    """One member at one step: byte-range u, v, T at every isobaric level of pgrb2a + pgrb2b, decode, reduce to that
    member's eddy quadratics per level. -> (step, member, lat, {lev: [1, uv, vth, th, u]}, bytes)"""
    import eccodes
    date, hh, mem, h, kmax = job
    global KMAX
    KMAX = kmax
    fields, nbytes, lat = {}, 0, None
    for kind in ("a", "b"):
        url = gefs_url(date, hh, mem, kind, h)
        idx = _gefs_index(url)
        runs, cur = [], []                                # coalesce exactly-contiguous messages into one request
        for e in idx:
            if cur and cur[-1][1] + 1 == e[0]:
                cur.append(e)
            else:
                if cur:
                    runs.append(cur)
                cur = [e]
        if cur:
            runs.append(cur)

        def pull(run):
            return run, _get(url, (run[0][0], run[-1][1]))
        with ThreadPoolExecutor(12) as ex:
            for run, blob in ex.map(pull, runs):
                nbytes += len(blob)
                pos = 0
                for s, e, var, lev in run:
                    n = (e - s + 1) if e >= 0 else len(blob) - pos
                    gid = eccodes.codes_new_from_message(bytes(blob[pos:pos + n]))
                    pos += n
                    try:
                        nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
                        v = eccodes.codes_get_values(gid).reshape(nj, ni)
                        j0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                    finally:
                        eccodes.codes_release(gid)
                    if j0 < 0:
                        v = v[::-1]
                    if lat is None:
                        lat = np.linspace(90.0, -90.0, nj)
                    fields.setdefault(lev, {})[var] = v
    out = {}
    ps = fields.get("sfc", {}).get("PRES")
    for lev, d in fields.items():
        if lev != "sfc" and len(d) == 3:
            q = _quadratics(d["UGRD"][None], d["VGRD"][None], d["TMP"][None], lev)
            # share of longitudes where this level is below this member's ground (GEFS extrapolates there)
            q.append((ps < lev * 100.0).mean(axis=-1) if ps is not None else np.zeros(d["UGRD"].shape[0]))
            out[lev] = q
    return h, mem, lat, out, nbytes


def gefs_ready(date, hh, last_h):
    try:
        _get(gefs_url(date, hh, "gep30", "b", last_h) + ".idx")
        _get(gefs_url(date, hh, "gep30", "a", last_h) + ".idx")
        return True
    except Exception:                                                 # noqa: BLE001
        return False


def gefs_resolve(date, hh, last_h, back=4):
    """The asked-for cycle if GEFS has reached last_h there, else the newest of the previous `back` 6-hourly cycles
    that has (strat.yml passes the AIFS-ENS cycle; GEFS is normally complete well before it)."""
    t = dt.datetime.strptime(date + hh, "%Y%m%d%H")
    for k in range(back + 1):
        c = t - dt.timedelta(hours=6 * k)
        d, h = c.strftime("%Y%m%d"), c.strftime("%H")
        if gefs_ready(d, h, last_h):
            if k:
                print(f"  GEFS {date} {hh}Z not complete to {last_h} h; using {d} {h}Z", flush=True)
            return d, h
    raise SystemExit(f"no GEFS cycle complete to {last_h} h within {6 * back} h of {date} {hh}Z")


def compute_epflux_gefs(date, hh, steps, procs=None, members=None):
    """Ensemble QG E-P flux from GEFS: per-member quadratics, member-averaged per (step, level)."""
    procs = procs or int(os.environ.get("EPFLUX_GEFS_PROCS", str(min(8, os.cpu_count() or 2))))
    mems = members or GEFS_MEMBERS
    jobs = [(date, hh, m, h, KMAX) for h in steps for m in mems]
    sums, levs_seen, lat, nbytes, t0 = {}, set(), None, 0, time.time()
    done_by_step = {}
    import multiprocessing as mp
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("spawn")) as ex:   # fork + threads dies on macOS
        futs = {ex.submit(_gefs_job, j): j for j in jobs}
        from concurrent.futures import as_completed
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                h, mem, la, per, nb = fu.result()
            except Exception as e:                                    # noqa: BLE001
                print(f"  GEFS {j[2]} f{j[3]:03d}: skipped ({str(e)[:80]})", flush=True)
                continue
            if la is not None and lat is None:
                lat = la
            nbytes += nb
            for lev, q in per.items():
                s = sums.setdefault((h, lev), [0, 0.0, 0.0, 0.0, 0.0, 0.0])
                s[0] += 1
                for i in range(1, 6):
                    s[i] = s[i] + q[i]
                levs_seen.add(lev)
            done_by_step[h] = done_by_step.get(h, 0) + 1
            if done_by_step[h] == len(mems):
                print(f"  GEFS f{h:03d}: {len(mems)} members ({nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s so far)",
                      flush=True)
    if lat is None:
        raise SystemExit("GEFS epflux: nothing decoded")
    # a level must carry every member that the step has, or it would mix ensembles
    acc = {}
    for (h, lev), s in sums.items():
        nmax = max(v[0] for (hh_, l), v in sums.items() if hh_ == h)
        if s[0] == nmax:
            acc[(h, lev)] = [s[0]] + [x / s[0] for x in s[1:]]
    by_step, nmem = _finish(acc, levs_seen, lat, steps)
    stats = dict(bytes=nbytes, seconds=time.time() - t0, jobs=len(jobs))
    return by_step, nmem, lat, stats


# ------------------------------------------------------------------------------------------------------ render ---
ARROW_K = 7.0e-9                   # axis-fraction length per unit of the Edmon-scaled flux (fixed; see the docstring)
ARROW_CAP = 0.09                  # longest arrow drawn, axis fraction
ARROW_LEVS = (1000, 850, 700, 500, 300, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2)
ARROW_LATS = np.arange(-80, 80.1, 5.0)
P_TOP, P_BOT = 1.0, 1000.0
FILL_LEVELS = [-40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
STAGING = HERE / "data" / ".epflux_gefs_staging"


def edmon_arrows(Fphi, Fp, levs, lat, xspan_deg=180.0):
    """Arrow components in AXIS-FRACTION units. Edmon et al. (1980) scaling written for a log-p axis (Jucker 2021):
    x = p F_phi / (a dphi_axis), y_up = -F_p / ln(p_bot/p_top), so that the arrows' visual divergence on this plot is
    p div F; then both components times 1000 hPa / p (= exp(z/H), i.e. per unit mass) so the stratosphere is legible,
    and the one fixed ARROW_K. The common factor leaves every direction exact."""
    p = levs[:, None] * 100.0
    g = 1.0e5 / p
    ax = p * Fphi / (A * np.deg2rad(xspan_deg)) * g
    ay = -Fp / np.log(P_BOT / P_TOP) * g
    return ARROW_K * ax, ARROW_K * ay


def render_frame(fields, lat, out, title, footer, xlim=(-90.0, 90.0)):
    from matplotlib.colors import BoundaryNorm
    levs, p_pa, U, force, Fphi, Fp, th_prof = fields[:7]
    below = fields[7] if len(fields) > 7 else np.zeros_like(U)
    fig = plt.figure(figsize=(12.0, 7.0))
    ax = fig.add_axes([0.065, 0.115, 0.80, 0.79])
    cax = fig.add_axes([0.885, 0.115, 0.017, 0.79])
    cmap = plt.get_cmap("PuOr_r")
    norm = BoundaryNorm(FILL_LEVELS, cmap.N, extend="both")
    cf = ax.contourf(lat, levs, force, levels=FILL_LEVELS, cmap=cmap, norm=norm, extend="both")
    ul = [l for l in range(-150, 151, 10) if l]
    # grey: below ground, plus the levels just above it that the mask blanked through the centred differences
    bgm = (below > BG_FRAC) | (np.isnan(force) & (levs >= 500)[:, None] & (np.abs(lat) <= POLE_MASK)[None, :])
    U = np.where(np.abs(lat)[None, :] > 86.0, np.nan, U)           # no contour squiggles at the pole rows
    U = np.where(bgm, np.nan, U)
    if bgm.any():                                                    # below ground (GEFS surface pressure)
        ax.contourf(lat, levs, bgm.astype(float), levels=[0.5, 1.5], colors=["#c9c9c9"], zorder=2)
    ax.contour(lat, levs, U, levels=ul, colors="#404040", linewidths=0.6)
    ax.contour(lat, levs, U, levels=[0.0], colors="#000000", linewidths=1.8)
    cd = cd_ceiling_excess(U, lat, levs, th_prof)
    if np.isfinite(cd).any():
        ax.contour(lat, levs, cd, levels=[0.0], colors="#c2185b", linewidths=1.6, linestyles="dashed")
    ax.set_yscale("log")
    ax.set_ylim(P_BOT, P_TOP)
    ax.set_xlim(*xlim)
    yt = [1000, 700, 500, 300, 200, 100, 50, 30, 20, 10, 5, 3, 2, 1]
    ax.set_yticks(yt); ax.set_yticklabels([str(y) for y in yt], fontsize=9)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    xt = list(range(-80, 81, 20))
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{abs(t)}°{'S' if t < 0 else 'N' if t > 0 else ''}" for t in xt], fontsize=9)
    ax.set_xlabel("latitude", fontsize=10); ax.set_ylabel("pressure (hPa)", fontsize=10)
    # the top layer: the derivative at 1 hPa is one-sided
    ax.fill_between(list(xlim), levs.min(), levs[1], facecolor="none", edgecolor="#555555", hatch="////",
                    linewidth=0.0, zorder=3)
    ax.text(xlim[0] + 1.5, np.sqrt(levs.min() * levs[1]), " one-sided ∂/∂p", fontsize=7.5, va="center",
            color="#333333", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))
    # arrows on an overlay axis in axis-fraction units (aspect-correct, independent of the log scale)
    ox = fig.add_axes(ax.get_position(), frameon=False)
    ox.set_xlim(0, 1); ox.set_ylim(0, 1); ox.set_xticks([]); ox.set_yticks([])
    ix = [int(np.argmin(np.abs(lat - la))) for la in ARROW_LATS]
    il = [int(np.flatnonzero(levs == l)[0]) for l in ARROW_LEVS if l in levs]
    ux, uy = edmon_arrows(Fphi, Fp, levs, lat, xlim[1] - xlim[0])
    X, Y, DX, DY, capped = [], [], [], [], []
    lp0, lp1 = np.log(P_BOT), np.log(P_TOP)
    for i in il:
        for j in ix:
            if abs(lat[j]) > POLE_MASK:
                continue
            dx, dy = ux[i, j], uy[i, j]
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            L = np.hypot(dx, dy)
            if L < 0.004:
                continue
            c = L > ARROW_CAP
            if c:
                dx, dy = dx * ARROW_CAP / L, dy * ARROW_CAP / L
            X.append((lat[j] - xlim[0]) / (xlim[1] - xlim[0]))
            Y.append((np.log(levs[i]) - lp0) / (lp1 - lp0))
            DX.append(dx); DY.append(dy); capped.append(c)
    X, Y, DX, DY, capped = map(np.asarray, (X, Y, DX, DY, capped))
    for sel, col in ((~capped, "#111111"), (capped, "#b71c1c")):
        if sel.any():
            ox.quiver(X[sel], Y[sel], DX[sel], DY[sel], angles="xy", scale_units="xy", scale=1.0, width=0.0021,
                      headwidth=3.6, headlength=4.0, facecolor=col, edgecolor="white", linewidth=0.3, pivot="tail",
                      alpha=0.92, zorder=5)
    cb = fig.colorbar(cf, cax=cax, ticks=[l for l in FILL_LEVELS if abs(l) in (0.5, 1, 2, 5, 10, 20, 40)], format="%g")
    cb.set_label("E–P flux divergence as zonal force (m s⁻¹ day⁻¹)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.065, 0.955, title, fontsize=12.5, fontweight="bold", ha="left", va="center")
    fig.text(0.5, 0.008, footer, fontsize=7.6, ha="center", va="bottom", color="#6f6b64", linespacing=1.35)
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return int(capped.sum())


def render_epflux(by_step, nmem, lat, S, base, model="gefs", waves="1-3"):
    """Frames + manifest: assets/sst/anim/epflux_gefs/ and epflux_gefs_manifest.json."""
    from PIL import Image
    region = f"epflux_{model}"
    outdir = REPO / "assets" / "sst" / "anim" / region
    outdir.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    for f in list(outdir.glob("F*.webp")) + list(STAGING.glob("*")):
        f.unlink()
    wtxt = "wavenumbers 1–3" if waves == "1-3" else "all wavenumbers"
    frames = []
    per_step = [(s, by_step[s]) for s in S if s in by_step]
    ncap = 0
    for idx, (s, fields) in enumerate(per_step):
        valid = base + pd.Timedelta(hours=s)
        fid = f"F{idx:02d}"
        title = (f"GEFS ensemble ({nmem} members) · E–P flux & wave driving, {wtxt} · "
                 f"day {s // 24} · valid {valid:%a %b %d %HZ}")
        footer = (f"NOAA GEFS 0.5\u00b0 (public domain), init {base:%Y-%m-%d %HZ}, 31 levels 1000\u20131 hPa, per-member "
                  "fluxes averaged \u00b7 QG E\u2013P flux, zonal-mean static stability, \u2202/\u2202p in ln p, grey: below ground\n"
                  "arrows: Edmon et al. (1980) log-p scaling \u00d7 1000 hPa/p, one fixed scale (red = capped) \u00b7 "
                  "dashed magenta: \u016b = U_c (wave-1 Charney\u2013Drazin) \u00b7 hatched: one-sided at 1 hPa "
                  "\u00b7 >82\u00b0 masked")
        png = STAGING / f"{fid}.png"
        ncap += render_frame(fields, lat, png, title, footer)
        Image.open(png).convert("RGB").save(outdir / f"{fid}.webp", quality=84, method=6)
        png.unlink()
        frames.append({"idx": idx, "file": f"{fid}.webp", "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s // 24} · {valid:%b %d}"})
    man = {"ver": int(time.time()), "days": len(frames), "init": f"{base:%Y-%m-%d %HZ}",
           "regions": {region: {"label": f"E–P flux & wave driving ({wtxt}) — GEFS ensemble",
                                "n_frames": len(frames), "frames": frames}}}
    (REPO / "assets" / "sst" / "anim" / f"{region}_manifest.json").write_text(json.dumps(man))
    print(f"wrote {len(frames)} {region} frames + manifest ({ncap} capped arrows in the loop)", flush=True)


# --------------------------------------------------------------------------------- zonal-mean u for the zero line ---
def save_zmu(zmu, lat, steps, model, base):
    """{(step, lev): (member, lat)} -> data/zmu_members_<model>.npz for qbo_zeroline.py (best effort)."""
    try:
        arrs = {}
        for lev in ZMU_LEVELS:
            rows = [zmu.get((sh, lev)) for sh in steps]
            if any(r is None for r in rows):
                continue
            n = min(len(r) for r in rows)
            arrs[f"u{lev}"] = np.stack([r[:n] for r in rows]).astype(np.float32)     # (step, member, lat)
        if not arrs:
            print("  zmu: nothing to save", flush=True); return
        out = HERE / "data" / f"zmu_members_{model}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, lat=np.asarray(lat, float), steps=np.asarray(steps, int), init=str(base)[:16], **arrs)
        print(f"  zmu: {out.name} {sorted(arrs)} {next(iter(arrs.values())).shape}", flush=True)
    except Exception as e:                                                   # noqa: BLE001
        print(f"  zmu: not saved ({str(e)[:80]})", flush=True)


def _read_u_members(path, lev, steps):
    """{step: (member, lat)} zonal-mean u at one level from a (cf or pf) AIFS-ENS GRIB in the store cache."""
    import eccodes as ec
    got, lat = {}, None
    with open(path, "rb") as fh:
        while True:
            g = ec.codes_grib_new_from_file(fh)
            if g is None:
                break
            try:
                if int(ec.codes_get(g, "level")) != lev:
                    continue
                st = int(ec.codes_get(g, "endStep"))
                if st not in steps:
                    continue
                nj, ni = ec.codes_get(g, "Nj"), ec.codes_get(g, "Ni")
                if lat is None:
                    la0 = ec.codes_get(g, "latitudeOfFirstGridPointInDegrees")
                    la1 = ec.codes_get(g, "latitudeOfLastGridPointInDegrees")
                    lat = np.linspace(la0, la1, nj)
                num = int(ec.codes_get(g, "number")) if ec.codes_is_defined(g, "number") else 0
                got.setdefault(st, {})[num] = ec.codes_get_values(g).reshape(nj, ni).mean(axis=1)
            finally:
                ec.codes_release(g)
    return got, lat


def aifs_zmu(date, time_):
    """AIFS-ENS: per-member zonal-mean u at ZMU_LEVELS from the u files ens_cycle --only strat cached (control + 25),
    and v at 250 hPa (control + 25) ensured for pacjet.py. No E-P flux since 2026-09-30."""
    import store as ecmwf
    cyc = ecmwf.Cycle(date, time_)
    S = tuple(ecmwf.STEPS)
    NM = ecmwf.AAM_PF_MEMBERS
    L12 = tuple(ecmwf.LEVELS_AAM_REST)
    u_cf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", L12, S))
    u_pf = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", L12, S, NM))
    for typ, nm in (("cf", 0), ("pf", NM)):                       # pacjet.py reads these from the cache
        try:
            ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "v", "pl", (250,), S, nm))
        except Exception as e:                                        # noqa: BLE001
            print(f"  v250 {typ} for pacjet not fetched ({str(e)[:80]})", flush=True)
    zmu, lat = {}, None
    for lev in ZMU_LEVELS:
        cf, la = _read_u_members(u_cf, lev, S)
        pf, _ = _read_u_members(u_pf, lev, S)
        lat = la if la is not None else lat
        for sh in S:
            rows = {**cf.get(sh, {}), **pf.get(sh, {})}
            if rows:
                zmu[(sh, lev)] = np.stack([rows[k] for k in sorted(rows)])
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {time_}:00")
    save_zmu(zmu, lat, S, "aifs", base)


IFS_U_LEVELS = (10, 50, 250)          # 10/50 for the zero line, 250 (with v) for the North Pacific jet


def _ifs_step(job):
    """One IFS-ENS step: byte-range u at 10/50/250 hPa and v at 250 hPa for the 50 perturbed members from the Google
    mirror, decode in memory. -> (step, lat, {lev: (member, lat) zonal-mean u}, bytes). The 250 hPa sector goes to
    PACJET_IFS_DIR for the North Pacific jet."""
    import eccodes
    import rangefetch as rf
    date, time_, sh = job
    t0 = time.time()
    idx = rf.fetch_index(date, time_, "ifs", sh, "ef", sources=["google"])
    want = [e for e in rf.select(idx, param=["u", "v"], levelist=list(IFS_U_LEVELS), type="pf")
            if e["param"] == "u" or int(e["levelist"]) == 250]
    want = sorted(want, key=lambda e: e["_offset"])
    blob = rf.fetch_ranges(rf.path_for(date, time_, "ifs", sh, "ef") + ".grib2", rf.coalesce(want),
                           sources=["google"])
    where, pos = {}, 0
    for e in want:
        where.setdefault(int(e["levelist"]), {}).setdefault(e["param"], []).append((int(e["number"]), pos, e["_length"]))
        pos += e["_length"]
    if pos != len(blob):
        raise RuntimeError(f"step {sh}: {len(blob)} bytes for {pos} indexed")
    # ef files are in ARRIVAL order and the member order differs between parameters: key rows by member number.
    jet = os.environ.get("PACJET_IFS_DIR")
    lat = lon = sl = PJ = None
    zmu, keep = {}, {}
    for lev in IFS_U_LEVELS:
        for par in ("u", "v"):
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
                    if jet and lev == 250:
                        if sl is None:
                            sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
                            import pacjet_core as PJ
                            sl = PJ.native_slices(lat, lon)
                        keep.setdefault(par, {})[num] = full[np.ix_(sl[0], sl[1])].astype("float32")
                    if par == "u":
                        rows[num] = full.mean(axis=1)
                finally:
                    eccodes.codes_release(gid)
            if par == "u" and rows and lev in ZMU_LEVELS:
                zmu[lev] = np.stack([rows[k] for k in sorted(rows)])
    if keep and set(keep) == {"u", "v"}:
        both = sorted(set(keep["u"]) & set(keep["v"]))
        try:
            ii, jj = np.arange(sl[0].size), np.arange(sl[1].size)
            Path(jet).mkdir(parents=True, exist_ok=True)
            np.savez(Path(jet) / f"step_{sh:03d}.npz", members=np.array(both),
                     u=PJ.to_work(np.stack([keep["u"][k] for k in both]), ii, jj).astype("float16"),
                     v=PJ.to_work(np.stack([keep["v"][k] for k in both]), ii, jj).astype("float16"))
        except Exception as e:                        # noqa: BLE001
            print(f"  pacjet dump step {sh} failed: {str(e)[:100]}", flush=True)
    del blob
    print(f"  IFS-ENS step {sh:3d}: {pos / 1e6:,.0f} MB in {time.time() - t0:.0f} s", flush=True)
    return sh, lat, zmu, pos


def ifs_zmu(date, time_, workers=None):
    sys.path.insert(0, str(HERE))
    import ifs_ens as IE
    import store as ecmwf
    if not IE.published(date, time_):
        raise SystemExit(f"IFS-ENS {date} {time_}Z is not on the Google mirror yet")
    workers = workers or int(os.environ.get("EPFLUX_IFS_WORKERS", "3"))
    S = tuple(ecmwf.STEPS)
    zmu, lat, nb, t0 = {}, None, 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sh, la, per, b in ex.map(_ifs_step, [(date, time_, s) for s in S]):
            lat = la if lat is None else lat
            nb += b
            for lev, r in per.items():
                zmu[(sh, lev)] = r
    print(f"  IFS-ENS zonal-mean u / jet stream: {nb / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {time_}:00")
    save_zmu(zmu, lat, S, "ifs", base)


def latest_cycle():
    """Newest AIFS-ENS cycle in the store cache with the u files (when strat.yml passes none)."""
    for d in sorted(glob.glob(str(CACHE / "*z")), reverse=True):
        if glob.glob(f"{d}/aifs-ens/cf_u_10-*x16.grib2"):
            tag = Path(d).name
            return tag[:8], tag[8:10]
    raise SystemExit("no AIFS-ENS cycle with the u files in the cache")


def main():
    import argparse
    global KMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--epflux-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--strip-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=("gefs", "aifs", "ifs"), default="gefs")
    ap.add_argument("--date"); ap.add_argument("--time")
    ap.add_argument("--days", type=int, default=15, help="GEFS: last day (15; 35 only on 00Z runs)")
    ap.add_argument("--waves", choices=("1-3", "all"), default="1-3")
    ap.add_argument("--procs", type=int)
    ap.add_argument("--members", type=int, default=31, help="GEFS: control + N-1 perturbed")
    args = ap.parse_args()
    if args.strip_only:
        print("--strip-only: the QBO strip was removed; nothing to do")
        return
    if args.model == "aifs":
        d, t = (args.date, args.time) if args.date and args.time else latest_cycle()
        aifs_zmu(d, t)
        return
    if args.model == "ifs":
        if not (args.date and args.time):
            ap.error("--model ifs needs --date and --time")
        ifs_zmu(args.date, args.time)
        return
    KMAX = 3 if args.waves == "1-3" else 0
    if not args.date:
        now = dt.datetime.utcnow() - dt.timedelta(hours=6)
        args.date, args.time = now.strftime("%Y%m%d"), f"{(now.hour // 6) * 6:02d}"
    hh = f"{int(args.time or 0):02d}"
    if args.days > 16 and hh != "00":
        ap.error("GEFS runs past day 16 only from 00Z")
    last_h = 24 * args.days
    date, hh = gefs_resolve(args.date, hh, last_h)
    S = tuple(range(0, last_h + 1, 24))
    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    by_step, nmem, lat, st = compute_epflux_gefs(date, hh, S, procs=args.procs,
                                                 members=GEFS_MEMBERS[:args.members])
    print(f"  GEFS epflux {date} {hh}Z: {len(by_step)} steps, {nmem} members, {st['bytes'] / 1e9:.2f} GB "
          f"in {st['seconds']:.0f} s ({st['bytes'] / 1e6 / max(st['seconds'], 1):.0f} MB/s)", flush=True)
    render_epflux(by_step, nmem, lat, S, base, "gefs", args.waves)


if __name__ == "__main__":
    main()
