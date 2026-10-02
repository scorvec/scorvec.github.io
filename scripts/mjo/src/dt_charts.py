#!/usr/bin/env python3
"""Dynamic-tropopause charts from the AIFS-ENS member 0 (PV does not survive averaging).

From temperature and wind on nine pressure levels (700–100 hPa), 12-hourly to day 10:
  Ertel PV on the pressure grid    PV = −g [ (ζ + f) ∂θ/∂p − ∂v/∂p ∂θ/∂x + ∂u/∂p ∂θ/∂y ]   (PVU = 1e−6 K m² kg⁻¹ s⁻¹)
  the dynamic tropopause           first crossing of 2 PVU searching upward from 700 hPa; θ, p and the wind
                                   interpolated linearly in PV between the bracketing levels
  isentropic PV                    PV and wind interpolated in θ onto 330 K and 350 K
Rendered on the hemispheric polar view (loops dt, dt_pv330, dt_pv350) and, since 2026-10-02, four regional Lambert views
(suffix _npac, _na, _atl, _asia): 15 loops in dt_manifest.json. The static four-panel figure was dropped 2026-09-07 (user).
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
LAT0 = 0.0                                                               # southern edge of the computation (the regional
                                                                         # Lambert maps reach ~10N at their corners; tropics: no 2-PVU crossing below 100 hPa = 100 hPa values)
THETAS = (330.0, 350.0)
CENTRAL_LON = -140.0
# 2026-10-02 (user: "split it into regions, like npac, north america, etc and maybe change the color scale"): the
# hemispheric polar view plus four regional Lambert views, every field in each. lon in 0..360, a range may wrap.
REGIONS = {
    "nh":   dict(label="Northern Hemisphere"),
    "npac": dict(label="North Pacific", lon=(115, 250), lat=(17, 72), clon=-178, clat=45),
    "na":   dict(label="North America", lon=(200, 315), lat=(15, 70), clon=-100, clat=45),
    "atl":  dict(label="North Atlantic & Europe", lon=(282, 45), lat=(22, 75), clon=-20, clat=50),
    "asia": dict(label="East Asia", lon=(75, 175), lat=(15, 68), clon=125, clat=42),
}
DT_LEV = np.arange(260, 401, 5)
# θ on the dynamic tropopause: purple/blue = low θ (troughs, cut-offs, stratospheric air folded down), greens through
# the middle, yellow-orange-red = high θ (ridges, tropical air). Replaces 'turbo' (garish, no light-dark order).
DT_ANCH = ["#f2c6e6", "#c77fc9", "#8e3fa8", "#5d2a8a", "#3b3f9e", "#3f6fcf", "#3fa9df", "#4fc6cd", "#6fd6a8",
           "#a0e07f", "#d6e76a", "#f6d54c", "#f8ae3d", "#ef8231", "#dc552a", "#bd3029", "#93192b", "#5a0d24", "#3a0a14"]
PV_LEV = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12]
# PV: tropospheric air in warm yellow-orange, a hard switch to blue at 2 PVU, stratospheric reservoir in navy-purple
PV_COLS = ["#fbe9b0", "#f8d27e", "#f2ad57", "#e5803d", "#b9d8ee", "#86b9e3", "#5a96d3", "#3e74c0", "#3654a6",
           "#3c3a8c", "#4a2675", "#5b1560"]


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
    # no 2 PVU below the data top (deep tropics): the tropopause is at or above 100 hPa, so use the 100 hPa values
    # (a lower bound on its θ) instead of a hole in the map
    top = {"theta": th[-1], "p": np.full(th.shape[1:], p_pa[-1] / 100.0), "u": U[-1], "v": V[-1]}
    for k in out:
        out[k] = np.where(never, top[k], out[k])
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


RG_W, RG_H = 11.8, 7.9


def _frame_region(rg):
    """A wide figure with one regional Lambert map and the colour-bar axes under it."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    R = REGIONS[rg]
    fig = plt.figure(figsize=(RG_W, RG_H))
    proj = ccrs.LambertConformal(central_longitude=R["clon"], central_latitude=R["clat"], standard_parallels=(30, 60))
    ax = fig.add_axes([0.012, 0.105, 0.976, 0.845], projection=proj)
    lo0, lo1 = R["lon"]; span = (lo1 - lo0) % 360
    ax.set_extent([lo0 - 360 if lo0 > 180 else lo0, (lo0 - 360 if lo0 > 180 else lo0) + span, R["lat"][0], R["lat"][1]], crs=ccrs.PlateCarree())
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.022])
    return fig, ax, cax


def _subset(rg, lat, lon, *fields):
    """Region slice: lat ascending, lon in a contiguous (unwrapped) order; returns lat, lon, fields."""
    # The whole hemisphere around the map's centre longitude: a Lambert map's top corners reach across the pole to the
    # far side, and a box-shaped subset left white wedges there. The cone's cut sits at clon+180, so longitudes run
    # clon-179 .. clon+179 and no cell straddles it.
    clon = REGIONS[rg]["clon"]
    jl = np.where((lat >= -10) & (lat < 90))[0]                            # pole row off
    off = (lon - clon + 180) % 360
    il = np.where((off >= 1) & (off <= 359))[0]; il = il[np.argsort(off[il])]
    lon_u = clon - 180 + off[il]
    return lat[jl], lon_u, [f[np.ix_(jl, il)] for f in fields]


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


def _cmaps():
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    n = len(DT_LEV) + 1
    dtc = LinearSegmentedColormap.from_list("dt", DT_ANCH)
    dcm = ListedColormap([dtc(x) for x in np.linspace(0, 1, n)])
    pcm = ListedColormap(PV_COLS)                                           # 11 bins + the >12 PVU extension
    return (dcm, BoundaryNorm(DT_LEV, n, extend="both")), (pcm, BoundaryNorm(PV_LEV, len(PV_COLS), extend="max"))


def _decorate(ax, polar, title):
    import cartopy.feature as cfeature
    ax.coastlines(resolution="50m", lw=1.0, color="#000", zorder=6); ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor="#222", zorder=6)
    if not polar: ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25, edgecolor="#444", zorder=6)
    ax.gridlines(lw=0.3, color="#6f6b64", alpha=0.6, ylocs=range(10, 90, 10 if not polar else 20), xlocs=range(-180, 181, 20 if not polar else 30))
    ax.set_title(title, fontsize=10.5 if not polar else 9.6, loc="left", fontweight="bold")


def _draw(ax, lat, lon, field, pcont, u, v, kind, polar, title):
    """kind 'dt': θ shading + DT pressure contours; 'pv': PV shading + 2-PVU line. Arrows: wind on that surface."""
    import cartopy.crs as ccrs
    (dcm, dnorm), (pcm, pnorm) = _cmaps()
    if polar:
        la, lo = _coarse(lat), _coarse(lon); F, C, U_, V_ = _coarse(field), (_coarse(pcont) if pcont is not None else None), u, v
        qlo, qla, qu, qv = _coarse(lon, 20)[:-1], _coarse(lat, 20), _coarse(u, 20)[:, :-1], _coarse(v, 20)[:, :-1]
    else:
        la, lo, F, C = lat, lon, field, pcont
        st = max(1, int(round(4.5 / abs(lon[1] - lon[0]))))                      # arrows every ~4.5 deg
        qlo, qla, qu, qv = lon[::st], lat[::st], u[::st, ::st], v[::st, ::st]
    X, Y = _projected(ax, lo, la)
    if kind == "dt":
        cf = ax.pcolormesh(X, Y, F, cmap=dcm, norm=dnorm, shading="nearest", rasterized=True)
        ax.contour(X, Y, C, levels=[200, 300, 400, 500], colors="k", linewidths=[0.5, 0.7, 0.9, 1.1], alpha=0.55)
    else:
        cf = ax.pcolormesh(X, Y, F, cmap=pcm, norm=pnorm, shading="nearest", rasterized=True)
        ax.contour(X, Y, F, levels=[2], colors="k", linewidths=1.5)
    qla_ = np.asarray(qla)
    qu = np.where(qla_[:, None] > 80, np.nan, qu)                       # no arrows near the pole (they pile up there)
    if polar:                                                           # equal-area-ish arrows: thin longitudes by 1/cos(lat)
        thin = np.maximum(1, np.round(1 / np.cos(np.deg2rad(np.clip(qla_, 0, 80))))).astype(int)
        qu = np.where((np.arange(qu.shape[1])[None, :] % thin[:, None]) == 0, qu, np.nan)
    ax.quiver(qlo, qla, qu, qv, transform=ccrs.PlateCarree(), scale=1100 if polar else 1500, width=0.0022 if polar else 0.0015, color="#111", alpha=0.8)
    _decorate(ax, polar, title)
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
    # loop keys: the hemispheric ones keep their old names (dt, dt_pv330, dt_pv350); regions add a suffix
    FIELDS = (("dt", "θ on 2 PVU"), ("dt_pv330", "PV on 330 K"), ("dt_pv350", "PV on 350 K"))
    key = lambda f, rg: f if rg == "nh" else f"{f}_{rg}"
    dirs = {key(f, rg): anim / key(f, rg) for f, _ in FIELDS for rg in REGIONS}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
        for old in p.glob("F*.webp"):
            old.unlink()
    entries = {k: [] for k in dirs}
    for k, h in enumerate(steps):
        if a.max_steps and k >= a.max_steps:
            break
        T, U, V = d["t"].isel(step=k).values, d["u"].isel(step=k).values, d["v"].isel(step=k).values
        pv, th = pv_step(T, U, V, p_pa, lat, lon)
        dt = dynamic_tropopause(pv, th, U, V, p_pa)
        isos = {th_: on_isentrope(pv, th, U, V, th_) for th_ in THETAS}
        valid = init + pd.Timedelta(hours=int(h)); lab = ("analysis" if h == 0 else f"+{h} h") + f" · {valid:%a %d %b %HZ}"
        for rg, R in REGIONS.items():
            polar = rg == "nh"
            for f, flab in FIELDS:
                if f == "dt":
                    fld, pc, uu, vv, kind = dt["theta"], dt["p"], dt["u"], dt["v"], "dt"
                    blab = ("θ on 2 PVU (K) · black: DT pressure 200–500 hPa · arrows: DT wind · DT above 100 hPa: θ at 100 hPa" if polar else
                            "θ on the 2-PVU surface (K) · blue/purple = low θ (troughs) · red = high θ (ridges) · black: DT pressure 200–500 hPa · arrows: DT wind · DT above 100 hPa: θ at 100 hPa")
                    ttl = f"Dynamic tropopause θ (2 PVU) · {R['label']} — AIFS-ENS member 0, init {init:%d %b %HZ} · {lab}"
                else:
                    iso = isos[330.0 if f == "dt_pv330" else 350.0]; th_ = 330 if f == "dt_pv330" else 350
                    fld, pc, uu, vv, kind = iso["pv"], None, iso["u"], iso["v"], "pv"
                    blab = f"PV on {th_} K (PVU) · black line: 2 PVU (the dynamic tropopause on this surface) · arrows: wind on {th_} K"
                    ttl = f"PV on {th_} K · {R['label']} — AIFS-ENS member 0, init {init:%d %b %HZ} · {lab}"
                if polar:
                    fig, ax, cax = _frame(); la_, lo_ = lat, lon; arrs = (fld, pc, uu, vv)
                else:
                    fig, ax, cax = _frame_region(rg)
                    la_, lo_, arrs = _subset(rg, lat, lon, fld, pc if pc is not None else fld, uu, vv)
                    if pc is None: arrs[1] = None
                cf = _draw(ax, la_, lo_, arrs[0], arrs[1], arrs[2], arrs[3], kind, polar, ttl)
                _bar(fig, cf, cax, blab)
                kk = key(f, rg); fp = dirs[kk] / f"F{k:02d}.webp"
                fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
                entries[kk].append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"), "label": lab})
        if k % 5 == 0:
            print(f"    step {h:3d} h done ({time.time() - t0:.0f}s)", flush=True)
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "default": "dt",
            "regions": {key(f, rg): {"label": f"{flab} · {R['label']}", "frames": entries[key(f, rg)]}
                        for f, flab in FIELDS for rg, R in REGIONS.items()}}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True); Path(a.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries['dt'])} frames × {len(entries)} loops, {a.manifest} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
