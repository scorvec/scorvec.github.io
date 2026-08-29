#!/usr/bin/env python3
"""Tropical stratospheric winds + the planetary-wave channel, from AIFS-ENS.

Per cycle:

assets/sst/anim/qbo_strip/F##.webp (+ qbo_strip_manifest.json)
    The QBO in map view, as a 16-frame day-0..15 forecast loop (site anim
    player): wind SPEED (shaded) + direction vectors over the 30S-30N strip
    at 10 / 50 / 100 hPa, from the AIFS-ENS control. Solid black: u = 0
    critical line. Per-level colour range fixed across the loop at the 99.5th
    percentile of speed over ALL steps (rounded up to 5), so nothing
    saturates and frames are comparable. assets/strat/qbo_strip.webp is a
    copy of the analysis frame (static fallback).
    v at 10/50/100 for every step is fetched through the shared ecmwf store
    (~35 MB/cycle, control only); u rides the AAM pull.

assets/strat/wave_channel.webp
    Where planetary waves CAN go, and where they ARE going: latitude x height
    section (1000 -> 10 hPa, all 14 AIFS levels) of zonal-mean zonal wind,
    with the u = 0 critical line, the ~28 m/s wave-1 Charney-Drazin ceiling,
    hatching over the open wave-1 channel (0 < u < 28), and horizontal
    arrows of the meridional flux of wave activity by planetary waves
    (zonal wavenumbers 1-3): F_phi = -cos(phi) [u'v'], the horizontal EP flux
    component - arrows point the way wave packets are propagating.

    python scripts/strat/qbo_duct.py
"""
from __future__ import annotations
import glob
import json
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE = REPO / "scripts" / "ecmwf" / "cache"
OUT_STRIP = REPO / "assets" / "strat" / "qbo_strip.webp"
OUT_CHAN = REPO / "assets" / "strat" / "epflux.webp"
ANIM_DIR = REPO / "assets" / "sst" / "anim" / "qbo_strip"
MANIFEST = REPO / "assets" / "sst" / "anim" / "qbo_strip_manifest.json"
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf

LEVS_STRIP = [10, 50, 100]
UC1 = 28.0                                   # wave-1 Charney-Drazin ceiling (m/s)
KMAX = 3                                     # planetary wavenumbers for u'v'


def latest_cycle():
    for d in sorted(glob.glob(str(CACHE / "*z")), reverse=True):
        u_full = glob.glob(f"{d}/aifs-ens/cf_u_10-*x16.grib2")        # 12 lev, 0..360 h
        u_rmm = glob.glob(f"{d}/aifs-ens/cf_u_200-850_s0-*x16.grib2") # 200/850, 0..360 h
        v_a14 = glob.glob(f"{d}/aifs-ens/cf_v_10-*_s0-0x1.grib2")     # 14 lev, analysis
        if u_full and v_a14:
            tag = Path(d).name
            return u_full[0], (sorted(u_rmm)[-1] if u_rmm else None), v_a14[0], tag
    raise SystemExit("no AIFS-ENS cycle with the needed files in the cache")


def open_field(path, short, lev=None, step=None):
    kw = {"shortName": short}
    if lev is not None:
        kw["level"] = lev
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs=dict(filter_by_keys=kw, indexpath=""))
    da = ds[short]
    if step is not None and "step" in da.dims:
        da = da.sel(step=pd.Timedelta(hours=step))
    return da.load()


def smooth_lonwrap(f, sig):
    return ndi.gaussian_filter(f, sigma=sig, mode=("nearest", "wrap"))


def wave_uv_flux(u2d, v2d, lat):
    """Zonal-mean meridional wave-activity flux by planetary waves:
    F_phi = -cos(phi) [u'v'] with u', v' bandpassed to zonal wavenumbers
    1..KMAX. Positive = wave packets propagating poleward... sign convention:
    [u'v'] > 0 means poleward westerly-momentum flux = EQUATORWARD wave
    propagation in the NH; the EP component -cos(phi)[u'v'] points WITH the
    wave group velocity. Returns (nlat,) array."""
    def band(f):
        F = np.fft.rfft(f, axis=1)
        F[:, 0] = 0.0
        F[:, KMAX + 1:] = 0.0
        return np.fft.irfft(F, n=f.shape[1], axis=1)
    up, vp = band(u2d), band(v2d)
    return -np.cos(np.deg2rad(lat)) * (up * vp).mean(axis=1)


def load_strip(u_full, tag):
    """(steps_h, latS, lon, {lev: (u, v)}) — full-forecast u and v on the
    30S-30N strip, all steps. u rides the cached AAM pull; v is ensured
    through the shared store (small, control only)."""
    cyc = ecmwf.Cycle(tag[:8], tag[8:10])
    vpath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl",
                                         tuple(LEVS_STRIP), tuple(ecmwf.STEPS)))
    out = {}
    steps = None
    latS = lon = None
    for lev in LEVS_STRIP:
        u = open_field(u_full, "u", lev)
        v = open_field(vpath, "v", lev)
        lat = u.latitude.values
        m = (lat >= -30) & (lat <= 30)
        latS, lon = lat[m], u.longitude.values
        su = (u.step.values / np.timedelta64(1, "h")).astype(int)
        sv = (v.step.values / np.timedelta64(1, "h")).astype(int)
        steps = sorted(set(su) & set(sv)) if steps is None else \
            [s for s in steps if s in set(su) & set(sv)]
        out[lev] = (u.sel(latitude=u.latitude[m]),
                    v.sel(latitude=v.latitude[m]))
    return steps, latS, lon, out


def strip_frames(u_full, tag, base):
    steps, latS, lon, F = load_strip(u_full, tag)
    pc = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180.0)
    titles = {10: "10 hPa", 50: "50 hPa", 100: "100 hPa"}
    # colour range fixed across the whole loop, per level
    caps = {}
    for lev in LEVS_STRIP:
        u, v = F[lev]
        spd = np.hypot(u.values, v.values)
        caps[lev] = max(15, float(np.ceil(np.percentile(spd, 99.5) / 5) * 5))

    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    for old in ANIM_DIR.glob("F*.webp"):
        old.unlink()
    frames = []
    qs_lat, qs_lon = 14, 30                     # vector subsample: ~3.5° × 7.5°
    for i, s in enumerate(steps):
        valid = base + pd.Timedelta(hours=s)
        fig, axes = plt.subplots(3, 1, figsize=(13.0, 8.2),
                                 subplot_kw={"projection": proj})
        for ax, lev in zip(axes, LEVS_STRIP):
            u, v = F[lev]
            ug = u.sel(step=pd.Timedelta(hours=s)).values
            vg = v.sel(step=pd.Timedelta(hours=s)).values
            spd = np.hypot(ug, vg)
            spdc, lonc = add_cyclic_point(spd, coord=lon)
            lstep = 5 if caps[lev] >= 40 else 2.5
            # below 10 m/s stays unfilled (transparent) — only real flow shows
            cf = ax.contourf(lonc, latS, spdc, transform=pc,
                             levels=np.arange(10, caps[lev] + 0.1, lstep),
                             cmap="YlGnBu", extend="max")
            uc, _ = add_cyclic_point(smooth_lonwrap(ug, (6, 6)), coord=lon)
            ax.contour(lonc, latS, uc, transform=pc, levels=[0],
                       colors="k", linewidths=1.8)
            # unit direction vectors (ASCAT convention: shading carries speed)
            with np.errstate(invalid="ignore"):
                m = spd[::qs_lat, ::qs_lon] > 0.5
                uq = np.where(m, (ug / np.maximum(spd, 0.1))[::qs_lat, ::qs_lon], np.nan)
                vq = np.where(m, (vg / np.maximum(spd, 0.1))[::qs_lat, ::qs_lon], np.nan)
            ax.quiver(lon[::qs_lon], latS[::qs_lat], uq, vq, transform=pc,
                      scale=85, width=0.0011, headwidth=3.2,
                      color="#222222", alpha=0.85, pivot="mid")
            ax.set_xlim(-180, 180); ax.set_ylim(-30, 30)
            # land tint under the (partly transparent) shading + bold coasts
            ax.add_feature(cfeature.LAND, facecolor="0.90", zorder=0)
            ax.coastlines(lw=1.0, color="0.15")
            gl = ax.gridlines(draw_labels=True, lw=0.3, color="0.85",
                              xlocs=range(-180, 181, 60),
                              ylocs=[-30, -15, 0, 15, 30])
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
            ax.set_title(f"{titles[lev]}   (shading to {caps[lev]:.0f} m s⁻¹)",
                         fontsize=11.5, fontweight="bold", loc="left")
            cb = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.05)
            cb.set_label("wind speed (m s⁻¹)", fontsize=8.5)
            cb.ax.tick_params(labelsize=7.5)
        fig.suptitle(f"Tropical stratospheric wind (30°S–30°N) — AIFS-ENS "
                     f"control · day {s // 24} · valid {valid:%a %b %d %HZ}",
                     fontsize=13.5, fontweight="bold")
        fig.text(0.09, 0.006,
                 "Shading: wind speed (fixed per-level range across the loop) · vectors: direction · "
                 "solid black: u = 0 critical line (stationary Rossby waves absorbed).",
                 fontsize=8.6, va="bottom")
        fig.tight_layout(rect=(0, 0.025, 1, 0.965))
        out = ANIM_DIR / f"F{i:02d}.webp"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        frames.append({"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s // 24} · {valid:%b %d}"})
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"qbo_strip": {
               "label": "Tropical stratospheric wind (30°S–30°N) — AIFS-ENS, 10/50/100 hPa",
               "n_frames": len(frames), "frames": frames}}}
    MANIFEST.write_text(json.dumps(man))
    OUT_STRIP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANIM_DIR / "F00.webp", OUT_STRIP)
    print(f"wrote {len(frames)} qbo_strip frames + manifest")


def _band13(f):
    """Zonal wavenumbers 1..KMAX of f(lat, lon)."""
    F = np.fft.rfft(f, axis=1)
    F[:, 0] = 0.0
    F[:, KMAX + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[1], axis=1)


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
    KAPPA = 0.2854
    acc = {}          # (step, lev) -> [n, uv, vth, th, ub]
    levs_seen = set()
    for lev in sorted(ecmwf.LEVELS_AAM):
        try:
            das = {k: _open_lev(*spec) for k, spec in paths[lev].items()}
        except Exception as e:                            # noqa: BLE001
            print(f"  epflux: level {lev} unavailable ({str(e)[:50]})", flush=True)
            continue
        levs_seen.add(lev)
        fac = (1000.0 / lev) ** KAPPA
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
            u, v, t = u[:n], v[:n], t[:n]
            th = t * fac
            up, vp, thp = _band13(u), _band13(v), _band13(th)
            acc[(sh, lev)] = [n,
                              (up * vp).mean(axis=-1).mean(axis=0),
                              (vp * thp).mean(axis=-1).mean(axis=0),
                              th.mean(axis=-1).mean(axis=0),
                              u.mean(axis=-1).mean(axis=0)]
        for da in das.values():
            da.close()

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

    outdir = REPO / "assets" / "sst" / "anim" / "epflux"
    outdir.mkdir(parents=True, exist_ok=True)
    EP_STAGING.mkdir(parents=True, exist_ok=True)
    for f in EP_STAGING.glob("*"):
        f.unlink()

    by_step, nmem = compute_epflux_ensemble(paths, lat, S)
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
                   f"AIFS-ENS ensemble ({nmem} members) \u00b7 day {s // 24} \u00b7 valid {valid:%a %b %d %HZ}"),
            footer="ECMWF AIFS-ENS open data (CC BY 4.0) \u00b7 QG E\u2013P flux, k=1\u20133 \u00b7 shading: \u2207\u00b7F as zonal force (m/s/day) \u00b7 dashed magenta: \u016b = U_c, the wave-1 Charney\u2013Drazin ceiling (l = 2/a) \u00b7 poleward of 82\u00b0 masked",
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
        np.savez_compressed(EP_STAGING / f"{fid}.npz",
                            **{k: np.asarray(v, np.float64) for k, v in arrays.items()})
        frames_meta.append(meta)
        frames.append({"idx": idx, "file": f"{fid}.webp",
                       "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s // 24} \u00b7 {valid:%b %d}"})

    spec = dict(staging=str(EP_STAGING), frames=frames_meta)
    (EP_STAGING / "spec.json").write_text(json.dumps(spec))
    ok = False
    if shutil.which("julia"):
        try:
            r = subprocess.run(
                ["julia", "--project=" + str(JULIA_SCRIPT.parent),
                 str(JULIA_SCRIPT), str(EP_STAGING / "spec.json")],
                capture_output=True, text=True, timeout=1800)
            ok = r.returncode == 0 and "JULIA RENDER DONE" in r.stdout
            if not ok:
                print(f"  epflux julia failed:\n{(r.stderr or r.stdout)[-400:]}", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"  epflux julia failed ({str(e)[:80]})", flush=True)

    n_jl = 0
    for meta, (s, fields) in zip(frames_meta, per_step):
        fid = meta["frame_id"]
        png = EP_STAGING / f"{fid}.png"
        out = outdir / f"{fid}.webp"
        if ok and png.exists():
            Image.open(png).convert("RGB").save(out, quality=84, method=6)
            png.unlink()
            n_jl += 1
        else:
            _epflux_frame_mpl(meta, fields, lat, out)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {"epflux": {
               "label": "E\u2013P flux & wave driving (k=1\u20133) \u2014 AIFS-ENS ensemble",
               "n_frames": len(frames), "frames": frames}}}
    (REPO / "assets" / "sst" / "anim" / "epflux_manifest.json").write_text(json.dumps(man))
    shutil.copy2(outdir / "F00.webp", OUT_CHAN)
    print(f"wrote {len(frames)} epflux frames ({n_jl} via julia) + manifest")


def _epflux_frame_mpl(meta, fields, lat, out):
    """Matplotlib fallback for a single staged frame (same spec/arrays)."""
    levs, p_pa, U, force, Fphi, Fp, th_prof = fields
    z = np.load(EP_STAGING / f"{meta['frame_id']}.npz")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip-only", action="store_true")
    ap.add_argument("--epflux-only", action="store_true")
    args = ap.parse_args()
    u_full, u_rmm, v_a14, tag = latest_cycle()
    base = pd.Timestamp(f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00")
    if not args.epflux_only:
        strip_frames(u_full, tag, base)
    if not args.strip_only:
        epflux_loop(u_full, u_rmm, tag, base)


if __name__ == "__main__":
    main()
