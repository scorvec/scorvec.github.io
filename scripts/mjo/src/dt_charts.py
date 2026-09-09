#!/usr/bin/env python3
"""Dynamic-tropopause charts from the AIFS-ENS CONTROL member (PV does not survive averaging).

From temperature and wind on nine pressure levels (700–100 hPa), 12-hourly to day 10:
  Ertel PV on the pressure grid    PV = −g [ (ζ + f) ∂θ/∂p − ∂v/∂p ∂θ/∂x + ∂u/∂p ∂θ/∂y ]   (PVU = 1e−6 K m² kg⁻¹ s⁻¹)
  the dynamic tropopause           first crossing of 2 PVU searching upward from 700 hPa; θ, p and the wind
                                   interpolated linearly in PV between the bracketing levels
  isentropic PV                    PV and wind interpolated in θ onto 330 K and 350 K
Rendered on a North-Pacific-to-Atlantic polar stereographic view (20–90°N) as three loops
(assets/sst/anim/dt, dt_pv330, dt_pv350 + dt_manifest.json). The static four-panel figure was dropped 2026-09-07 (user).
    python src/dt_charts.py --date 20260906 --time 00 --anim-dir ../../assets/sst/anim \
        --manifest ../../assets/sst/anim/dt_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                                   # noqa: E402

LEVS = (700, 600, 500, 400, 300, 250, 200, 150, 100)
STEPS = tuple(range(0, 241, 12))
A_EARTH, OMEGA, G0, KAPPA = 6.371e6, 7.2921e-5, 9.80665, 0.2857
LAT0 = 15.0                                                              # southern edge of the computation
THETAS = (330.0, 350.0)
CENTRAL_LON = -140.0


def load(cyc):
    out = {}
    for par in ("t", "u", "v"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", par, "pl", LEVS, STEPS))
        da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})[par]
        if "number" in da.dims:
            da = da.squeeze("number", drop=True)
        da = da.sortby("latitude").sel(latitude=slice(LAT0, 90.0)).sel(isobaricInhPa=list(LEVS))
        out[par] = da.transpose("step", "isobaricInhPa", "latitude", "longitude").astype("float32")
    return out


def pv_step(T, U, V, p_pa, lat, lon):
    """Ertel PV (PVU) on the pressure grid for one time: arrays (lev, lat, lon)."""
    th = T * (1e5 / p_pa[:, None, None]) ** KAPPA
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[None, :, None]
    dlam = lam[1] - lam[0]; dphi = phi[1] - phi[0]
    def ddx(a):
        return (np.roll(a, -1, -1) - np.roll(a, 1, -1)) / (2 * dlam) / (A_EARTH * cosphi)
    def ddy(a):
        return np.gradient(a, dphi, axis=1) / A_EARTH
    zeta = ddx(V) - ddy(U * cosphi) / cosphi
    f = (2 * OMEGA * np.sin(phi))[None, :, None]
    th_p = np.gradient(th, p_pa, axis=0); u_p = np.gradient(U, p_pa, axis=0); v_p = np.gradient(V, p_pa, axis=0)
    pv = -G0 * ((zeta + f) * th_p - v_p * ddx(th) + u_p * ddy(th)) * 1e6
    # light 3-point smoothing in both horizontal directions (the 0.25° grid carries some noise in ζ)
    pv = (pv + np.roll(pv, 1, -1) + np.roll(pv, -1, -1)) / 3.0
    pv[:, 1:-1] = (pv[:, :-2] + pv[:, 1:-1] + pv[:, 2:]) / 3.0
    return pv, th


def dynamic_tropopause(pv, th, U, V, p_pa, thr=2.0):
    """θ, p (hPa), u, v on the thr-PVU surface, searching upward from the lowest level; NaN where
    the column never reaches thr (deep tropics) — those points are masked in the maps."""
    above = pv >= thr
    nl = pv.shape[0]
    idx = np.argmax(above, axis=0)                                       # first level at/above thr
    never = ~above.any(axis=0)
    k1 = np.clip(idx, 1, nl - 1); k0 = k1 - 1
    jj, ii = np.meshgrid(np.arange(pv.shape[1]), np.arange(pv.shape[2]), indexing="ij")
    pv0, pv1 = pv[k0, jj, ii], pv[k1, jj, ii]
    w = np.clip((thr - pv0) / np.where(np.abs(pv1 - pv0) < 1e-6, 1e-6, pv1 - pv0), 0, 1)
    w = np.where(idx == 0, 0.0, w)                                       # already above thr at 700 hPa: take the level itself
    def interp(a):
        return a[k0, jj, ii] * (1 - w) + a[k1, jj, ii] * w
    lnp = np.log(p_pa)[:, None, None] * np.ones_like(pv)
    out = {"theta": interp(th), "p": np.exp(interp(lnp)) / 100.0, "u": interp(U), "v": interp(V)}
    for k in out:
        out[k] = np.where(never, np.nan, out[k])
    return out


def on_isentrope(pv, th, U, V, theta):
    """PV and wind interpolated linearly in θ onto one isentrope (θ increases with height here)."""
    nl = pv.shape[0]
    below = th <= theta
    idx = np.clip(np.sum(below, axis=0) - 1, 0, nl - 2)
    jj, ii = np.meshgrid(np.arange(pv.shape[1]), np.arange(pv.shape[2]), indexing="ij")
    t0, t1 = th[idx, jj, ii], th[idx + 1, jj, ii]
    w = np.clip((theta - t0) / np.where(np.abs(t1 - t0) < 1e-3, 1e-3, t1 - t0), 0, 1)
    valid = (th[0] <= theta) & (th[-1] >= theta)
    def interp(a):
        return np.where(valid, a[idx, jj, ii] * (1 - w) + a[idx + 1, jj, ii] * w, np.nan)
    return {"pv": interp(pv), "u": interp(U), "v": interp(V)}


# ── rendering ────────────────────────────────────────────────────────────────
# Canvas geometry (inches). The map is a CIRCLE inscribed in a square, and a GeoAxes keeps an equal
# aspect, so any rect that is not itself square leaves the leftover as white margin: the old
# 9.6x9.9 figure with a 0.96x0.87 rect drew the circle 851 px wide inside 960 px and put 61 px of
# white above the title. The rect below is square by construction (MAP_S on both sides), which is
# the whole trick — the title and colour bar get the only bands that are not map.
FIG_W, MAP_S, PAD_TOP, PAD_BOT = 9.6, 9.40, 0.26, 0.52
FIG_H = MAP_S + PAD_TOP + PAD_BOT


def _frame():
    """A figure with the polar map filling it, and the colour-bar axes under it."""
    import cartopy.crs as ccrs
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    proj = ccrs.NorthPolarStereo(central_longitude=CENTRAL_LON)
    ax = fig.add_axes([(FIG_W - MAP_S) / 2 / FIG_W, PAD_BOT / FIG_H, MAP_S / FIG_W, MAP_S / FIG_H], projection=proj)
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 200); circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    cax = fig.add_axes([0.22, 0.36 / FIG_H, 0.56, 0.115 / FIG_H])
    return fig, ax, cax


def _bar(fig, cf, cax, label):
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.6, pad=1.5)
    cb.set_label(label, fontsize=7.6, labelpad=2)
    return cb


def _coarse(a, n=2):
    """Every n-th point; the pole row is dropped (degenerate cells) and a cyclic column appended."""
    if a.ndim == 1:
        if a.size > 400:                                                  # longitude
            b = a[::n]; return np.concatenate([b, [b[0] + 360]])
        return a[:-1:n]                                                   # latitude, pole row off
    b = a[:-1:n, ::n]
    return np.concatenate([b, b[:, :1]], axis=1)


def _projected(ax, lo, la):
    """2-D map coordinates of a lon/lat grid in the axes' projection — drawing in native
    coordinates avoids cartopy's wrap handling, which streaked these fields on the polar view."""
    import cartopy.crs as ccrs
    LO, LA = np.meshgrid(lo, la)
    xyz = ax.projection.transform_points(ccrs.PlateCarree(), LO, LA)
    return xyz[..., 0], xyz[..., 1]


def draw_dt(ax, lat, lon, dt, title):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    import matplotlib.pyplot as plt
    pc = ccrs.PlateCarree()
    la, lo = _coarse(lat), _coarse(lon)
    levels = np.arange(280, 381, 6)
    X, Y = _projected(ax, lo, la)
    cmap = plt.get_cmap("turbo", len(levels) + 1)
    cf = ax.pcolormesh(X, Y, _coarse(dt["theta"]), cmap=cmap, norm=BoundaryNorm(levels, cmap.N, extend="both"), shading="nearest", rasterized=True)
    ax.contour(X, Y, _coarse(dt["p"]), levels=[200, 300, 400, 500], colors="k", linewidths=[0.5, 0.7, 0.9, 1.1], alpha=0.55)
    ax.quiver(_coarse(lon, 20)[:-1], _coarse(lat, 20), _coarse(dt["u"], 20)[:, :-1], _coarse(dt["v"], 20)[:, :-1], transform=pc, scale=1100, width=0.0022, color="#111", alpha=0.85)
    ax.coastlines(resolution="50m", lw=1.0, color="#000", zorder=6); ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor="#222", zorder=6)
    ax.gridlines(lw=0.3, color="#888", alpha=0.6, ylocs=range(20, 90, 20), xlocs=range(-180, 181, 30))
    ax.set_title(title, fontsize=9.6, loc="left", fontweight="bold")
    return cf


def draw_pv(ax, lat, lon, iso, theta, title):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    import matplotlib.pyplot as plt
    pc = ccrs.PlateCarree()
    la, lo = _coarse(lat), _coarse(lon)
    levels = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12]
    X, Y = _projected(ax, lo, la)
    cmap = plt.get_cmap("PuBuGn", len(levels))
    cf = ax.pcolormesh(X, Y, _coarse(iso["pv"]), cmap=cmap, norm=BoundaryNorm(levels, cmap.N, extend="max"), shading="nearest", rasterized=True)
    ax.contour(X, Y, _coarse(iso["pv"]), levels=[2], colors="#7a0c0c", linewidths=1.2)
    ax.quiver(_coarse(lon, 20)[:-1], _coarse(lat, 20), _coarse(iso["u"], 20)[:, :-1], _coarse(iso["v"], 20)[:, :-1], transform=pc, scale=1100, width=0.0022, color="#111", alpha=0.85)
    ax.coastlines(resolution="50m", lw=1.0, color="#000", zorder=6); ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor="#222", zorder=6)
    ax.gridlines(lw=0.3, color="#888", alpha=0.6, ylocs=range(20, 90, 20), xlocs=range(-180, 181, 30))
    ax.set_title(title, fontsize=9.6, loc="left", fontweight="bold")
    return cf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim"); ap.add_argument("--manifest", default="assets/sst/anim/dt_manifest.json")
    ap.add_argument("--max-steps", type=int, default=0, help="render only the first N steps (timing tests)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    cyc = ecmwf.Cycle(a.date, a.time); init = pd.Timestamp(f"{a.date}T{a.time}:00")
    d = load(cyc); print(f"  fields loaded {dict(d['t'].sizes)} ({time.time() - t0:.0f}s)", flush=True)
    lat, lon = d["t"].latitude.values, d["t"].longitude.values
    p_pa = np.array(LEVS, float) * 100.0
    steps = (d["t"].step / np.timedelta64(1, "h")).values.astype(int)
    anim = Path(a.anim_dir)
    dirs = {"dt": anim / "dt", "dt_pv330": anim / "dt_pv330", "dt_pv350": anim / "dt_pv350"}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
        for old in p.glob("F*.webp"):
            old.unlink()
    entries = {k: [] for k in dirs}
    keep = {}
    for k, h in enumerate(steps):
        if a.max_steps and k >= a.max_steps:
            break
        T, U, V = d["t"].isel(step=k).values, d["u"].isel(step=k).values, d["v"].isel(step=k).values
        pv, th = pv_step(T, U, V, p_pa, lat, lon)
        dt = dynamic_tropopause(pv, th, U, V, p_pa)
        valid = init + pd.Timedelta(hours=int(h)); lab = ("analysis" if h == 0 else f"+{h} h") + f" · {valid:%a %d %b %HZ}"
        if h in (0, 48, 96, 144):
            keep[h] = dt
        fig, ax, cax = _frame()
        cf = draw_dt(ax, lat, lon, dt, f"Dynamic tropopause θ (2 PVU) — AIFS-ENS control, init {init:%d %b %HZ} · {lab}")
        _bar(fig, cf, cax, "θ on the 2-PVU surface (K) · black contours: DT pressure 200/300/400/500 hPa · arrows: wind on the DT")
        fp = dirs["dt"] / f"F{k:02d}.webp"; fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
        entries["dt"].append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"), "label": lab})
        for theta, key in zip(THETAS, ("dt_pv330", "dt_pv350")):
            iso = on_isentrope(pv, th, U, V, theta)
            fig, ax, cax = _frame()
            cf = draw_pv(ax, lat, lon, iso, theta, f"PV on {theta:.0f} K — AIFS-ENS control, init {init:%d %b %HZ} · {lab}")
            _bar(fig, cf, cax, f"PV on {theta:.0f} K (PVU) · dark red: 2 PVU (the dynamic tropopause on this surface) · arrows: wind on {theta:.0f} K")
            fp = dirs[key] / f"F{k:02d}.webp"; fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
            entries[key].append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"), "label": lab})
        if k % 5 == 0:
            print(f"    step {h:3d} h done ({time.time() - t0:.0f}s)", flush=True)
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "default": "dt",
            "regions": {"dt": {"label": "Dynamic tropopause θ (2 PVU)", "frames": entries["dt"]},
                        "dt_pv330": {"label": "PV on 330 K", "frames": entries["dt_pv330"]},
                        "dt_pv350": {"label": "PV on 350 K", "frames": entries["dt_pv350"]}}}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True); Path(a.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries['dt'])} frames × 3 loops, {a.manifest} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
