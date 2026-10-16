#!/usr/bin/env python3
"""Dynamic-tropopause charts from the AIFS-ENS member 0 (PV does not survive averaging).

From temperature and wind on nine pressure levels (700–100 hPa), 6-hourly to day 10:
  Ertel PV on the pressure grid    PV = −g [ (ζ + f) ∂θ/∂p − ∂v/∂p ∂θ/∂x + ∂u/∂p ∂θ/∂y ]   (PVU = 1e−6 K m² kg⁻¹ s⁻¹)
  the dynamic tropopause           first crossing of 2 PVU searching upward from 700 hPa; θ, p and the wind
                                   interpolated linearly in PV between the bracketing levels
  isentropic PV                    PV and wind interpolated in θ onto 330 K and 350 K
Rendered on the hemispheric polar view (loops dt, dt_pv330, dt_pv350) and five regional views (suffix _npac, _na, _atl,
_asia: Lambert; _sam: plate carree): 18 loops in dt_manifest.json.
    python src/dt_charts.py --date 20260906 --time 00 --anim-dir ../../assets/sst/anim \
        --manifest ../../assets/sst/anim/dt_manifest.json
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.pv import dynamic_tropopause, on_isentrope

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf
import tc_jet as TC

LEVS = (700, 600, 500, 400, 300, 250, 200, 150, 100)
STEPS = tuple(range(0, 241, 6))  # 6-hourly to day 10
A_EARTH, OMEGA, G0, KAPPA = 6.371e6, 7.2921e-5, 9.80665, 0.2857
LAT0 = -80.0  # southern edge of the computation (South America;
# the NH regional Lambert maps reach ~10N at their
# corners; tropics: no 2-PVU crossing below 100 hPa = 100 hPa values)
THETAS = (330.0, 350.0)
CENTRAL_LON = -140.0
# the hemispheric polar view plus five regional views, every field in each. lon in 0..360, a range may wrap.
REGIONS = {
    "nh": dict(label="Northern Hemisphere"),
    "npac": dict(label="North Pacific", lon=(115, 250), lat=(17, 72), clon=-178, clat=45),
    "na": dict(label="North America", lon=(200, 315), lat=(15, 70), clon=-100, clat=45),
    "atl": dict(label="North Atlantic & Europe", lon=(282, 45), lat=(22, 75), clon=-20, clat=50),
    "asia": dict(label="East Asia", lon=(75, 175), lat=(15, 68), clon=125, clat=42),
    # South America: Southern Hemisphere, so the dynamic tropopause is the -2 PVU surface and the isentropic maps show -PV
    # (same colours and thresholds as the NH). Plate carree: a southern Lambert cone in a frame this wide tilts the
    # continent and spends half the map on the tropics. Wide enough to show troughs coming in from the SE Pacific and
    # leaving into the South Atlantic, and the whole continent north to 12N.
    "sam": dict(label="South America", lon=(230, 5), lat=(-68, 12), clon=-65, clat=-28, sh=True, proj="pc"),
}
DT_LEV = np.arange(260, 401, 5)
# θ on the dynamic tropopause: purple/blue = low θ (troughs, cut-offs, stratospheric air folded down), greens through
# the middle, yellow-orange-red = high θ (ridges, tropical air). Unlike 'turbo', it has a light-dark order.
DT_ANCH = [
    "#f2c6e6",
    "#c77fc9",
    "#8e3fa8",
    "#5d2a8a",
    "#3b3f9e",
    "#3f6fcf",
    "#3fa9df",
    "#4fc6cd",
    "#6fd6a8",
    "#a0e07f",
    "#d6e76a",
    "#f6d54c",
    "#f8ae3d",
    "#ef8231",
    "#dc552a",
    "#bd3029",
    "#93192b",
    "#5a0d24",
    "#3a0a14",
]
PV_LEV = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12]
# PV: tropospheric air in warm yellow-orange, a hard switch to blue at 2 PVU, stratospheric reservoir in navy-purple
PV_COLS = [
    "#fbe9b0",
    "#f8d27e",
    "#f2ad57",
    "#e5803d",
    "#b9d8ee",
    "#86b9e3",
    "#5a96d3",
    "#3e74c0",
    "#3654a6",
    "#3c3a8c",
    "#4a2675",
    "#5b1560",
]


def load(cyc):
    out = {}
    for par in ("t", "u", "v"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", par, "pl", LEVS, STEPS))
        da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})[par]
        if "number" in da.dims:
            da = da.squeeze("number", drop=True)
        if par in ("u", "v"):  # GLOBAL 300-200 hPa layer mean: the
            out[par + "_layer"] = (
                da.sel(isobaricInhPa=[300, 250, 200])
                .mean("isobaricInhPa")  # divergence inversion
                .transpose("step", "latitude", "longitude")
                .astype("float32")
            )  # needs the sphere
        da = da.sortby("latitude").sel(latitude=slice(LAT0, 90.0)).sel(isobaricInhPa=list(LEVS))
        out[par] = da.transpose("step", "isobaricInhPa", "latitude", "longitude").astype("float32")
    return out


def pv_step(T, U, V, p_pa, lat, lon):
    """Ertel PV (PVU) on the pressure grid for one time: arrays (lev, lat, lon)."""
    th = T * (1e5 / p_pa[:, None, None]) ** KAPPA
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[None, :, None]
    dlam = lam[1] - lam[0]
    dphi = phi[1] - phi[0]

    def ddx(a):
        return (np.roll(a, -1, -1) - np.roll(a, 1, -1)) / (2 * dlam) / (A_EARTH * cosphi)

    def ddy(a):
        return np.gradient(a, dphi, axis=1) / A_EARTH

    zeta = ddx(V) - ddy(U * cosphi) / cosphi
    f = (2 * OMEGA * np.sin(phi))[None, :, None]
    th_p = np.gradient(th, p_pa, axis=0)
    u_p = np.gradient(U, p_pa, axis=0)
    v_p = np.gradient(V, p_pa, axis=0)
    pv = -G0 * ((zeta + f) * th_p - v_p * ddx(th) + u_p * ddy(th)) * 1e6
    # light 3-point smoothing in both horizontal directions (the 0.25° grid carries some noise in ζ)
    pv = (pv + np.roll(pv, 1, -1) + np.roll(pv, -1, -1)) / 3.0
    pv[:, 1:-1] = (pv[:, :-2] + pv[:, 1:-1] + pv[:, 2:]) / 3.0
    return pv, th


# ── rendering ────────────────────────────────────────────────────────────────
# Canvas geometry (inches). The map is a CIRCLE inscribed in a square, and a GeoAxes keeps an equal
# aspect, so any rect that is not itself square leaves the leftover as white margin. The rect below
# is square by construction (MAP_S on both sides), so the title and colour bar get the only bands
# that are not map.
FIG_W, MAP_S, PAD_TOP, PAD_BOT = 9.6, 9.40, 0.26, 0.74  # PAD_BOT: the colour bar and its label
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
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    cax = fig.add_axes([0.22, 0.36 / FIG_H, 0.56, 0.115 / FIG_H])
    return fig, ax, cax


RG_W, RG_H = 11.8, 8.6  # initial size; _frame_region sets the height from the map's aspect


RG_TOP, RG_BOT = 0.40, 0.80  # inches: title band; colour-bar band under the map


def _frame_region(rg):
    """A wide figure with one regional map (Lambert, or plate carree for South America) and the colour-bar axes under it.
    The figure height follows the map's own aspect, so there is no white band above the title or under the map. The
    size depends only on the region, so every frame of one loop is the same."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    R = REGIONS[rg]
    fig = plt.figure(figsize=(RG_W, RG_H))
    sh = R.get("sh", False)
    if R.get("proj") == "pc":  # cut at clon+180, the same as _subset's
        proj = ccrs.PlateCarree(central_longitude=R["clon"])
    else:
        proj = ccrs.LambertConformal(
            central_longitude=R["clon"],
            central_latitude=R["clat"],
            standard_parallels=(-30, -60) if sh else (30, 60),
            cutoff=30 if sh else -30,
        )
    ax = fig.add_axes([0.012, 0.172, 0.976, 0.777], projection=proj)
    lo0, lo1 = R["lon"]
    span = (lo1 - lo0) % 360
    ax.set_extent(
        [lo0 - 360 if lo0 > 180 else lo0, (lo0 - 360 if lo0 > 180 else lo0) + span, R["lat"][0], R["lat"][1]],
        crs=ccrs.PlateCarree(),
    )
    x0, x1, y0, y1 = ax.get_extent()
    map_w = 0.976 * RG_W
    map_h = map_w * abs(y1 - y0) / abs(x1 - x0)
    H = map_h + RG_BOT + RG_TOP
    fig.set_size_inches(RG_W, H)
    ax.set_position([0.012, RG_BOT / H, 0.976, map_h / H])
    cax = fig.add_axes([0.2, 0.36 / H, 0.6, 0.172 / H])
    return fig, ax, cax


def _subset(rg, lat, lon, *fields):
    """Region slice: lat ascending, lon in a contiguous (unwrapped) order; returns lat, lon, fields."""
    # The whole hemisphere around the map's centre longitude: a Lambert map's top corners reach across the pole to the
    # far side, and a box-shaped subset left white wedges there. The cone's cut sits at clon+180, so longitudes run
    # clon-179 .. clon+179 and no cell straddles it.
    clon = REGIONS[rg]["clon"]
    jl = (
        np.where((lat > -89.9) & (lat <= 30))[0] if REGIONS[rg].get("sh") else np.where((lat >= -10) & (lat < 90))[0]
    )  # pole row off
    off = (lon - clon + 180) % 360
    il = np.where((off >= 1) & (off <= 359))[0]
    il = il[np.argsort(off[il])]
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
        if a.size > 400:  # longitude
            b = a[::n]
            return np.concatenate([b, [b[0] + 360]])
        return a[:-1:n]  # latitude, pole row off
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
    pcm = ListedColormap(PV_COLS)  # 11 bins + the >12 PVU extension
    return (dcm, BoundaryNorm(DT_LEV, n, extend="both")), (pcm, BoundaryNorm(PV_LEV, len(PV_COLS), extend="max"))


def _decorate(ax, polar, title):
    import cartopy.feature as cfeature

    ax.coastlines(resolution="50m", lw=1.0, color="#000", zorder=6)
    ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor="#222", zorder=6)
    if not polar:
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25, edgecolor="#444", zorder=6)
    ax.gridlines(
        lw=0.3,
        color="#6f6b64",
        alpha=0.6,
        ylocs=range(-80, 90, 10) if not polar else range(20, 90, 20),
        xlocs=range(-180, 181, 20 if not polar else 30),
    )
    ax.set_title(title, fontsize=10.5 if not polar else 9.6, loc="left", fontweight="bold")


def _draw(ax, lat, lon, field, pcont, u, v, kind, polar, title):
    """kind 'dt': θ shading + DT pressure contours; 'pv': PV shading + 2-PVU line. Arrows: wind on that surface."""
    import cartopy.crs as ccrs

    (dcm, dnorm), (pcm, pnorm) = _cmaps()
    if polar:
        la, lo = _coarse(lat), _coarse(lon)
        F, C = _coarse(field), (_coarse(pcont) if pcont is not None else None)
        qlo, qla, qu, qv = _coarse(lon, 20)[:-1], _coarse(lat, 20), _coarse(u, 20)[:, :-1], _coarse(v, 20)[:, :-1]
    else:
        la, lo, F, C = lat, lon, field, pcont
        st = max(1, int(np.rint(4.5 / abs(lon[1] - lon[0]))))  # arrows every ~4.5 deg
        qlo, qla, qu, qv = lon[::st], lat[::st], u[::st, ::st], v[::st, ::st]
    X, Y = _projected(ax, lo, la)
    if kind == "dt":
        cf = ax.pcolormesh(X, Y, F, cmap=dcm, norm=dnorm, shading="nearest", rasterized=True)
        ax.contour(X, Y, C, levels=[200, 300, 400, 500], colors="k", linewidths=[0.5, 0.7, 0.9, 1.1], alpha=0.55)
    else:
        cf = ax.pcolormesh(X, Y, F, cmap=pcm, norm=pnorm, shading="nearest", rasterized=True)
        ax.contour(X, Y, F, levels=[2], colors="k", linewidths=1.5)
    qla_ = np.asarray(qla)
    qu = np.where(qla_[:, None] > 80, np.nan, qu)  # no arrows near the pole (they pile up there)
    if polar:  # equal-area-ish arrows: thin longitudes by 1/cos(lat)
        thin = np.maximum(1, np.round(1 / np.cos(np.deg2rad(np.clip(qla_, 0, 80))))).astype(int)
        qu = np.where((np.arange(qu.shape[1])[None, :] % thin[:, None]) == 0, qu, np.nan)
    ax.quiver(
        qlo,
        qla,
        qu,
        qv,
        transform=ccrs.PlateCarree(),
        scale=1100 if polar else 1500,
        width=0.0022 if polar else 0.0015,
        color="#111",
        alpha=0.8,
    )
    _decorate(ax, polar, title)
    return cf


def load_msl_control(cyc):
    """The control's mean sea-level pressure every 12 h to day 15 (hPa; lat ascending -5..85, lon 0..360) - the same request
    as tc_phase.load_msl, so whichever runs second finds the file cached."""
    p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "msl", "sfc", (), tuple(range(0, 361, 12))))
    da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["msl"]
    if "number" in da.dims:
        da = da.squeeze("number", drop=True)
    da = (
        da.assign_coords(longitude=da.longitude % 360)
        .sortby("longitude")
        .sortby("latitude")
        .sel(latitude=slice(-5.0, 85.0))
    )
    return (
        (da / 100.0).values.astype("float32"),
        da.latitude.values,
        da.longitude.values,
        (da.step / np.timedelta64(1, "h")).values.astype(int),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim")
    ap.add_argument("--manifest", default="assets/sst/anim/dt_manifest.json")
    ap.add_argument("--max-steps", type=int, default=0, help="render only the first N steps (timing tests)")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    cyc = ecmwf.Cycle(a.date, a.time)
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    d = load(cyc)
    print(f"  fields loaded {dict(d['t'].sizes)} ({time.time() - t0:.0f}s)", flush=True)
    lat, lon = d["t"].latitude.values, d["t"].longitude.values
    p_pa = np.array(LEVS, float) * 100.0
    steps = (d["t"].step / np.timedelta64(1, "h")).values.astype(int)
    anim = Path(a.anim_dir)
    # loop keys: the hemispheric ones keep their old names (dt, dt_pv330, dt_pv350); regions add a suffix
    FIELDS = (("dt", "θ on 2 PVU"), ("dt_pv330", "PV on 330 K"), ("dt_pv350", "PV on 350 K"))

    def key(f, rg):
        return f if rg == "nh" else f"{f}_{rg}"

    dirs = {key(f, rg): anim / key(f, rg) for f, _ in FIELDS for rg in REGIONS}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
        for old in p.glob("F*.webp"):
            old.unlink()
    entries = {k: [] for k in dirs}
    # tropical cyclones in the same control run (ECMWF's tracker on AIFS-ENS) + the outflow-jet metric, for the TC-jet
    # card (tcjet.webp / data/tcjet.json) only: the maps draw no storms. Only the storms the ensembles support enter
    # (observed at t=0, or carried by enough AIFS-ENS or IFS-ENS members: tc_ens.supported); a system the control alone
    # develops is left out and logged
    storms = []
    trk = "missing"
    dropped = []
    support = None
    try:
        fp = TC.fetch(a.date, int(a.time), Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc")
        if fp:
            storms = TC.nh_storms(TC.decode(fp))
            trk = "ok"
    except Exception as e:  # never let the tracks take the maps down
        TC.report(f"AIFS-ENS tracker file unreadable ({e!r}); TC-jet card drawn without storms", "warning")
        storms = []
    if storms:
        import tc_ens as TE

        try:
            storms, dropped, files = TE.supported(storms, a.date, int(a.time))
            support = dict(min_frac=TE.SUPPORT_FRAC, files=files)
        except Exception as e:  # no support test: only the observed storms
            TC.report(f"ensemble support unavailable ({e!r}); only storms observed at t=0 enter the card", "warning")
            dropped = [x for x in storms if not x.get("observed", x["named"])]
            storms = [x for x in storms if x.get("observed", x["named"])]
    # ECMWF's tracker stops at or near extratropical transition, which can be just before the storm deepens explosively
    # (Choi-Wan deepened 979 -> 948 hPa in the day after the tracker stopped). Each storm is followed on as the control's
    # surface low (tc_jet.follow_low, the same follower as the depth and phase-space charts; merge=True also follows it
    # into a low that already existed and absorbs it, as Choi-Wan did in the 2 Oct 12Z run) - dashed and labelled
    # post-tropical on the card, and the outflow index continues along it. Recurvature stays a property of the tracker's
    # track.
    for x in storms:
        x["rec_h"] = TC.recurvature(x)
    if storms:
        try:
            msl, mlat, mlon, msteps = load_msl_control(cyc)
            land = TC.load_land(cyc)  # the follower's land rule (heat / lee lows)
            ext = []
            for x in storms:
                cont = TC.follow_low(
                    msl, mlat, mlon, msteps, x, h_max=min(int(STEPS[-1]), TC.FOLLOW_H_MAX), merge=True, land=land
                )
                if len(cont["steps"]):
                    mg = (
                        f", merged into another low at +{int(cont['steps'][cont['merged']][0])} h"
                        if cont["merged"].any()
                        else ""
                    )
                    print(
                        f"  {TC.label(x)}: followed past the tracker's last fix (+{int(x['steps'][-1])} h) to "
                        f"+{int(cont['steps'][-1])} h, deepest {np.nanmin(cont['pmsl']):.0f} hPa{mg}",
                        flush=True,
                    )
                    ext.append(dict(x, **TC.extended(x, cont)))
                else:
                    ext.append(x)
            storms = ext
        except Exception as e:
            print(f"  control msl unavailable ({e}); storms end at the tracker's last fix", flush=True)
    print(
        f"  {len(storms)} NH tropical cyclones in the control: {', '.join(TC.label(x) for x in storms) or 'none'}",
        flush=True,
    )
    idx = {x["id"]: {} for x in storms}
    v250 = []
    for k, h in enumerate(steps):
        if a.max_steps and k >= a.max_steps:
            break
        T, U, V = d["t"].isel(step=k).values, d["u"].isel(step=k).values, d["v"].isel(step=k).values
        pv, th = pv_step(T, U, V, p_pa, lat, lon)
        v250.append(V[LEVS.index(250)])
        try:  # the TC-jet card's outflow index, at the tracker's exact fixes
            fixes = [(x, TC.at(x, int(h))) for x in storms]
            fixes = [(x, pos) for x, pos in fixes if pos is not None and pos[0] > 0]
            if fixes:
                uc, vc = TC.irrotational(d["u_layer"].isel(step=k), d["v_layer"].isel(step=k), lat, lon)
                pvl = pv[[LEVS.index(300), LEVS.index(250), LEVS.index(200)]].mean(0)
                adv = TC.pv_advection(uc, vc, pvl, lat, lon)
                adv[lat > 85] = np.nan  # the pole row's PV is singular
                for x, pos in fixes:
                    idx[x["id"]][int(h)] = TC.disc_index(adv, lat, lon, *pos)
        except Exception as e:
            print(f"  outflow PV advection failed at +{h} h ({e})", flush=True)
        dt = dynamic_tropopause(pv, th, U, V, p_pa)
        isos = {th_: on_isentrope(pv, th, U, V, th_) for th_ in THETAS}
        if any(R.get("sh") for R in REGIONS.values()):  # Southern Hemisphere: on -PV
            dt_s = dynamic_tropopause(-pv, th, U, V, p_pa)
            isos_s = {th_: on_isentrope(-pv, th, U, V, th_) for th_ in THETAS}
        valid = init + pd.Timedelta(hours=int(h))
        lab = ("analysis" if h == 0 else f"+{h} h") + f" · {valid:%a %d %b %HZ}"
        for rg, R in REGIONS.items():
            polar = rg == "nh"
            sh = R.get("sh", False)
            dtr, isr, pvt, sg = (dt_s, isos_s, "−2 PVU", "−PV") if sh else (dt, isos, "2 PVU", "PV")
            for f, _flab in FIELDS:
                if f == "dt":
                    fld, pc, uu, vv, kind = dtr["theta"], dtr["p"], dtr["u"], dtr["v"], "dt"
                    blab = f"θ on {pvt} (K) · black: DT pressure 200–500 hPa · arrows: DT wind · tropics: θ at 100 hPa"
                    ttl = f"Dynamic tropopause θ ({pvt}) · {R['label']} — AIFS-ENS member 0, init {init:%d %b %HZ} · {lab}"
                else:
                    iso = isr[330.0 if f == "dt_pv330" else 350.0]
                    th_ = 330 if f == "dt_pv330" else 350
                    fld, pc, uu, vv, kind = iso["pv"], None, iso["u"], iso["v"], "pv"
                    blab = (
                        f"{sg} on {th_} K (PVU"
                        + ("; Southern Hemisphere, sign flipped" if sh else "")
                        + f") · black line: {pvt} (the dynamic tropopause on this surface) · arrows: wind on {th_} K"
                    )
                    ttl = f"{sg} on {th_} K · {R['label']} — AIFS-ENS member 0, init {init:%d %b %HZ} · {lab}"
                if polar:
                    fig, ax, cax = _frame()
                    jn = lat >= 0.0  # the polar stereographic map: NH rows only (SH
                    la_, lo_ = lat[jn], lon  # points project to non-finite coordinates)
                    arrs = tuple(None if a_ is None else a_[jn] for a_ in (fld, pc, uu, vv))
                else:
                    fig, ax, cax = _frame_region(rg)
                    la_, lo_, arrs = _subset(rg, lat, lon, fld, pc if pc is not None else fld, uu, vv)
                    if pc is None:
                        arrs[1] = None
                cf = _draw(ax, la_, lo_, arrs[0], arrs[1], arrs[2], arrs[3], kind, polar, ttl)
                _bar(fig, cf, cax, blab)
                kk = key(f, rg)
                fp = dirs[kk] / f"F{k:02d}.webp"
                # loop frames: WebP q65 method 6 (~30-35 % smaller than q82; text and contours unchanged at 1x)
                fig.savefig(fp, dpi=100 if polar else 125, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
                plt.close(fig)
                entries[kk].append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"), "label": lab})
        if k % 5 == 0:
            print(f"    step {h:3d} h done ({time.time() - t0:.0f}s)", flush=True)
    mani = {
        "ver": int(pd.Timestamp.now().timestamp()),
        "default": "dt",
        "regions": {
            key(f, rg): {"label": f"{flab} · {R['label']}", "frames": entries[key(f, rg)]}
            for f, flab in FIELDS
            for rg, R in REGIONS.items()
        },
    }
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(mani))
    try:
        TC.render_card(
            storms,
            idx,
            np.array(v250),
            lat,
            lon,
            steps[: len(v250)],
            init,
            anim.parent / "tcjet.webp",
            anim.parent / "data" / "tcjet.json",
            status="missing" if trk == "missing" else ("ok" if storms else "none"),
            dropped=dropped,
            support=support,
        )
    except Exception as e:
        TC.report(f"TC-jet card failed ({e!r}); tcjet.webp and tcjet.json not updated", "error")
    print(
        f"wrote {len(entries['dt'])} frames × {len(entries)} loops, {a.manifest} in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
