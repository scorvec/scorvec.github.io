#!/usr/bin/env python3
"""Isentropic potential vorticity on 475 K and 850 K from GEOS FP, analysis + 10-day forecast (2026-09-27; user: "can we
add some gfs or geos-fp potential vorticity plots (showing a level in the lower and upper stratosphere)").

Source: NASA GMAO GEOS FP, `fcast/inst3_3d_asm_Np.<YYYYMMDD_00>` on NCCS OPeNDAP - the 00Z forecast, whose 00Z steps are
the initial state (day 0) and days 1-10. The collection carries Ertel PV itself (`epv`, SI: K m2 kg-1 s-1; 70 hPa cap
~2.7e-5 = 27 PVU on 2026-09-27, printed when this was built), so nothing is differentiated here. Only 200-3 hPa and
20-90 deg of each hemisphere are read, every second point (0.5 x 0.625 deg): ~17 MB and ~15 s per hemisphere per day.

Levels. 475 K is the lower stratosphere (~60-70 hPa over the winter pole), where the vortex couples to the troposphere;
850 K is ~10 hPa, the level a vortex split or displacement is judged at. PV, u and v are interpolated to each surface
linearly in ln(theta), theta = T (1000/p)^0.2857, column by column; the pressure of the surface is kept for the note.
Southern PV is sign-flipped so the vortex is positive in both hemispheres.

Vortex edge: Nash et al. (1996, JGR 101, 9471): grid points are ranked by PV and given an equivalent latitude (the
latitude of a polar cap with the same area as the region of higher PV); the edge is the equivalent latitude where the
PV gradient in equivalent latitude times the mean wind speed along the contour peaks (searched 45-85 deg, 3-degree
smoothed). It is drawn only when that wind is at least 15.2 m/s, their threshold for a vortex to exist - so a summer
hemisphere or a vortex that has broken down shows no edge rather than a meaningless one.

    python geos_pv.py --out assets/sst                       # newest complete cycle
    python geos_pv.py --out /tmp/pv --cycle 20260927_00
Writes <out>/anim/geos_pv_{nh,sh}/F00..F10.webp, <out>/anim/geos_pv_manifest.json, <out>/data/geos_pv.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Ambiguous reference date.*")

FC = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/inst3_3d_asm_Np"
LEVELS = (475.0, 850.0)
# fixed ladders, so a loop and a season read on one scale (SH late-September vortex: 475 K p99 66, 850 K p99 880 PVU)
SCALE = {475.0: np.arange(0, 85, 5), 850.0: np.arange(0, 1275, 75)}
NAME = {475.0: "lower stratosphere", 850.0: "middle stratosphere"}
KAPPA = 0.2857
R_E = 6371.0                                     # km
NASH_JET = 15.2                                  # m/s


def log(*a):
    print(*a, flush=True)


def retry(fn, what, n=5, base=20):
    for i in range(n):
        try:
            return fn()
        except Exception as e:                                           # noqa: BLE001
            if i == n - 1:
                raise
            log(f"  {what}: {str(e)[:120]} - retry {i + 1}")
            time.sleep(base * (i + 1))


def cycles():
    lst = retry(lambda: urllib.request.urlopen(FC, timeout=120).read().decode(), "listing")
    return sorted(set(re.findall(r"inst3_3d_asm_Np\.(\d{8}_00)", lst)))


def fetch(cyc: str):
    """{hemi: dict(epv, t, u, v: (step, lev, lat, lon), lat, lon, lev)} and the valid times, for one cycle."""
    import xarray as xr
    ds = retry(lambda: xr.open_dataset(f"{FC}/inst3_3d_asm_Np.{cyc}"), f"open {cyc}")
    steps = [t for t in pd.DatetimeIndex(ds.time.values) if t.hour == 0][:11]
    if len(steps) < 11:
        raise RuntimeError(f"{cyc}: only {len(steps)} 00Z steps")
    lev = ds.lev.values
    i0, i1 = int(np.argmin(np.abs(lev - 200))), int(np.argmin(np.abs(lev - 3)))
    lat = ds.lat.values
    sel = {"nh": slice(int(np.argmin(np.abs(lat - 20))), None, 2), "sh": slice(0, int(np.argmin(np.abs(lat + 20))) + 1, 2)}
    out = {}
    for h, sl in sel.items():
        arrs = {k: [] for k in ("epv", "t", "u", "v")}
        for t in steps:
            t0 = time.time()
            sub = retry(lambda: ds[["epv", "t", "u", "v"]].sel(time=t).isel(lev=slice(i0, i1 + 1), lat=sl,
                                                                            lon=slice(None, None, 2)).load(), f"{h} {t:%m-%d}")
            for k in arrs:
                x = sub[k].values.astype("float64")
                x[np.abs(x) > 1e10] = np.nan
                arrs[k].append(x)
            if not np.isfinite(arrs["epv"][-1]).any():
                raise RuntimeError(f"{cyc}: {h} {t:%m-%d} is empty (not posted yet?)")
            log(f"  {h} {t:%Y-%m-%d} {time.time() - t0:.0f}s")
        out[h] = {k: np.stack(v) for k, v in arrs.items()}
        out[h].update(lat=sub.lat.values, lon=sub.lon.values, lev=sub.lev.values)
    return out, steps


def to_theta(f, tgt):
    """PV (PVU), u, v and p on the theta surface tgt, for every step: dict of (step, lat, lon)."""
    p = f["lev"][None, :, None, None]
    th = f["t"] * (1000.0 / p) ** KAPPA
    lt = np.log(th)
    k = np.clip((th < tgt).sum(1) - 1, 0, th.shape[1] - 2)[:, None]
    a = np.take_along_axis(lt, k, 1)[:, 0]; b = np.take_along_axis(lt, k + 1, 1)[:, 0]
    w = (np.log(tgt) - a) / (b - a)
    bad = ~((w >= 0) & (w <= 1))

    def at(x):
        x0 = np.take_along_axis(x, k, 1)[:, 0]; x1 = np.take_along_axis(x, k + 1, 1)[:, 0]
        y = x0 + w * (x1 - x0)
        y[bad] = np.nan
        return y
    lnp = np.broadcast_to(np.log(p), th.shape)
    return {"pv": at(f["epv"]) * 1e6, "u": at(f["u"]), "v": at(f["v"]), "p": np.exp(at(lnp))}


def smooth(x, lat, km=80.0):
    """Gaussian filter of ~km on the (lat, lon) grid, isotropic on the sphere: GEOS FP's analysed EPV carries grid-scale
    noise at 0.5 deg that speckles both the shading and any contour. Display filter only; the edge is found on the same
    smoothed field it is drawn on."""
    from scipy.ndimage import gaussian_filter1d
    dlat = abs(float(lat[1] - lat[0])) * 111.2
    y = gaussian_filter1d(x, km / dlat, axis=0, mode="nearest")
    dlon = 360.0 / x.shape[1] * 111.2
    for j, la in enumerate(lat):
        sig = min(km / (dlon * max(np.cos(np.deg2rad(la)), 0.03)), x.shape[1] / 6)
        y[j] = gaussian_filter1d(y[j], sig, mode="wrap")
    return y


def nash_edge(pv, spd, lat):
    """Equivalent-latitude edge of the vortex (Nash et al. 1996) on one hemisphere's (lat, lon) field, PV positive in
    the vortex and lat as |lat|. -> dict(eqlat, pv, jet, area) or None when there is no vortex."""
    alat = np.abs(lat)
    w = np.broadcast_to(np.cos(np.deg2rad(alat))[:, None], pv.shape)
    ok = np.isfinite(pv) & np.isfinite(spd)
    q, s, a = pv[ok], spd[ok], w[ok]
    o = np.argsort(-q)
    q, s, a = q[o], s[o], a[o]
    frac = np.cumsum(a) / a.sum() * (1 - np.sin(np.deg2rad(alat.min())))     # cap area / 2 pi R^2, from the pole
    eql = np.rad2deg(np.arcsin(np.clip(1 - frac, -1, 1)))
    edges = np.arange(20, 91, 1.0)
    idx = np.digitize(eql, edges) - 1
    n = len(edges) - 1
    cnt = np.bincount(idx, minlength=n)[:n]
    Q = np.bincount(idx, weights=q * a, minlength=n)[:n] / np.maximum(np.bincount(idx, weights=a, minlength=n)[:n], 1e-12)
    U = np.bincount(idx, weights=s * a, minlength=n)[:n] / np.maximum(np.bincount(idx, weights=a, minlength=n)[:n], 1e-12)
    c = 0.5 * (edges[:-1] + edges[1:])
    good = cnt > 0
    Q, U, c = Q[good], U[good], c[good]
    dQ = np.gradient(Q, c)
    P = np.convolve(dQ * U, np.ones(3) / 3, mode="same")
    band = (c >= 45) & (c <= 85)
    if not band.any():
        return None
    i = np.flatnonzero(band)[int(np.argmax(P[band]))]
    if U[i] < NASH_JET or dQ[i] <= 0:
        return None
    area = 2 * np.pi * R_E ** 2 * (1 - np.sin(np.deg2rad(c[i]))) / 1e6         # million km2
    return {"eqlat": float(c[i]), "pv": float(Q[i]), "jet": float(U[i]), "area": float(area)}


class Native:
    """A regular grid in the polar-stereographic plane (the map's own coordinates) covering 20-90 deg, and the
    bilinear weights from the lat/lon grid onto it. Shading and contours are drawn on it WITHOUT a transform: no seam
    at the dateline, no singular row at the pole, and contour lengths come out in metres (a contour drawn with
    transform=PlateCarree split at 180 and ringed the pole)."""

    def __init__(self, north, lat, lon, n=420):
        import cartopy.crs as ccrs
        from scipy.interpolate import RegularGridInterpolator
        self.proj = ccrs.NorthPolarStereo() if north else ccrs.SouthPolarStereo()
        r = self.proj.transform_point(0.0, 20.0 if north else -20.0, ccrs.PlateCarree())
        R = float(np.hypot(*r))
        self.x = np.linspace(-R, R, n); self.y = np.linspace(-R, R, n)
        X, Y = np.meshgrid(self.x, self.y)
        ll = ccrs.PlateCarree().transform_points(self.proj, X, Y)
        self.lon, self.lat = ll[..., 0] % 360, ll[..., 1]
        self.out = np.hypot(X, Y) > R
        o = np.argsort(lat)
        self._lat, self._o = lat[o], o
        ol = np.argsort(lon % 360)
        self._ol = ol
        self._lon = np.append((lon % 360)[ol], (lon % 360)[ol][0] + 360)
        self._RGI = RegularGridInterpolator
        self._pts = np.column_stack([np.clip(self.lat.ravel(), self._lat[0], self._lat[-1]), self.lon.ravel()])

    def __call__(self, f):
        g = f[self._o][:, self._ol]
        g = np.concatenate([g, g[:, :1]], axis=1)
        v = self._RGI((self._lat, self._lon), g, bounds_error=False, fill_value=np.nan)(self._pts).reshape(self.lat.shape)
        v[self.out] = np.nan
        return v


def edge_lines(nat, F, level, min_km=2500.0):
    """Contour lines of F at level on the native grid, dropping pieces shorter than min_km (small filaments and noise
    islands inside or outside the vortex; the edge itself is thousands of km long)."""
    import contourpy
    lines = contourpy.contour_generator(nat.x, nat.y, np.ma.masked_invalid(F)).lines(level)
    keep = [L for L in lines if np.hypot(*np.diff(L, axis=0).T).sum() / 1e3 >= min_km]
    return keep


def render(hemi, north, lat, lon, fields, edges, pmean, valid, day, path, nat=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    nat = nat or Native(north, lat, lon)
    proj = nat.proj
    fig = plt.figure(figsize=(8.4, 5.2))
    span = 0.5
    for col, L in enumerate(LEVELS):
        ax = fig.add_axes([0.012 + col * span, 0.155, span * 0.955, 0.735], projection=proj)
        ax.set_xlim(nat.x[0], nat.x[-1]); ax.set_ylim(nat.y[0], nat.y[-1])
        lv = SCALE[L]
        cmap = plt.get_cmap("YlGnBu")
        nm = BoundaryNorm(lv, cmap.N, extend="max")
        F = nat(fields[L])
        m = ax.contourf(nat.x, nat.y, F, levels=lv, cmap=cmap, norm=nm, extend="max")
        e = edges[L]
        if e is not None:
            for ln in edge_lines(nat, F, e["pv"]):
                ax.plot(ln[:, 0], ln[:, 1], color="#c2410c", lw=1.7, solid_capstyle="round")
            note = (f"vortex edge {e['pv']:.0f} PVU at {e['eqlat']:.0f}°{'N' if north else 'S'} eq. lat · "
                    f"jet {e['jet']:.0f} m/s · {e['area']:.0f} M km²")
        else:
            note = f"no vortex edge (wind along the PV gradient < {NASH_JET} m/s)"
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.45, edgecolor="#555")
        ax.gridlines(crs=ccrs.PlateCarree(), lw=0.3, color="#bbb", ylocs=[60 if north else -60], draw_labels=False)
        ax.set_title(f"{L:.0f} K · {NAME[L]} · ~{pmean[L]:.0f} hPa over the cap", fontsize=10, fontweight="bold", pad=4)
        ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#2c4a72",
                fontweight="bold")
        cax = fig.add_axes([0.012 + col * span + span * 0.16, 0.058, span * 0.63, 0.021])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=lv[::4])
        cb.set_label("PVU" + ("" if north else "  (sign flipped: vortex positive)"), fontsize=8, labelpad=1)
        cb.ax.tick_params(labelsize=7, pad=1)
    when = "analysis" if day == 0 else f"day {day}"
    fig.suptitle(f"GEOS FP · {'Northern' if north else 'Southern'} Hemisphere potential vorticity — {when}, "
                 f"valid {valid:%a %d %b} 00Z", fontsize=11.5, fontweight="bold", y=0.985, va="top")
    fig.savefig(path, dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="site assets root (assets/sst): anim/ and data/ go under it")
    ap.add_argument("--cycle", help="YYYYMMDD_00 (default: the newest complete 00Z forecast)")
    a = ap.parse_args()
    out = Path(a.out)
    cands = [a.cycle] if a.cycle else cycles()[::-1][:3]
    for cyc in cands:
        try:
            log(f"GEOS FP forecast {cyc}")
            F, steps = fetch(cyc)
            break
        except RuntimeError as e:
            log(f"  {e}; trying the previous cycle")
    else:
        raise SystemExit("no complete GEOS FP forecast in the last three cycles")
    summary = {"source": "NASA GMAO GEOS FP, fcast inst3_3d_asm_Np (NCCS OPeNDAP)", "cycle": cyc,
               "valid": [f"{t:%Y-%m-%d}" for t in steps], "units": "PVU (SH sign flipped)",
               "edge_method": "Nash et al. 1996, max dPV/d(eq lat) x wind speed in 45-85 deg, jet >= 15.2 m/s",
               "hemispheres": {}}
    manifest = {"ver": cyc.replace("_", ""), "days": len(steps), "selectorLabel": "Hemisphere",
                "default": "geos_pv_nh", "regions": {}}
    for hemi in ("nh", "sh"):
        north = hemi == "nh"
        f = F[hemi]
        lat, lon = f["lat"], f["lon"]
        sgn = 1.0 if north else -1.0
        th = {L: to_theta(f, L) for L in LEVELS}
        capm = (np.abs(lat) >= 60)
        wc = np.cos(np.deg2rad(lat[capm]))
        nat = Native(north, lat, lon)
        d = out / "anim" / f"geos_pv_{hemi}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        frames, hs = [], {f"{L:.0f}": {"edge_eqlat": [], "edge_pv": [], "edge_jet": [], "area_Mkm2": [], "cap_pv": [],
                                          "cap_hpa": []} for L in LEVELS}
        for i, t in enumerate(steps):
            fields, edges, pmean = {}, {}, {}
            for L in LEVELS:
                pv = smooth(sgn * th[L]["pv"][i], lat)
                spd = np.hypot(th[L]["u"][i], th[L]["v"][i])
                fields[L] = pv
                edges[L] = nash_edge(pv, spd, lat)
                pc = th[L]["p"][i][capm]
                pmean[L] = float(np.exp((np.nanmean(np.log(pc), 1) * wc).sum() / wc.sum()))
                s = hs[f"{L:.0f}"]; e = edges[L]
                s["edge_eqlat"].append(None if e is None else round(e["eqlat"], 1))
                s["edge_pv"].append(None if e is None else round(e["pv"], 1))
                s["edge_jet"].append(None if e is None else round(e["jet"], 1))
                s["area_Mkm2"].append(None if e is None else round(e["area"], 1))
                s["cap_pv"].append(round(float((np.nanmean(pv[capm], 1) * wc).sum() / wc.sum()), 1))
                s["cap_hpa"].append(round(pmean[L], 1))
            fp = d / f"F{i:02d}.webp"
            render(hemi, north, lat, lon, fields, edges, pmean, t, i, fp, nat)
            frames.append({"idx": i, "file": fp.name, "date": f"{t:%Y-%m-%d}",
                           "label": ("analysis" if i == 0 else f"day {i}") + f" · {t:%a %d %b}"})
        log(f"  {hemi}: {len(frames)} frames; 850 K edge " + ", ".join("-" if x is None else f"{x:.0f}" for x in hs["850"]["edge_eqlat"]))
        manifest["regions"][f"geos_pv_{hemi}"] = {"label": f"{'Northern' if north else 'Southern'} hemisphere, 475 K and 850 K",
                                                  "n_frames": len(frames), "frames": frames}
        summary["hemispheres"][hemi] = hs
    (out / "anim" / "geos_pv_manifest.json").write_text(json.dumps(manifest))
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "geos_pv.json").write_text(json.dumps(summary, separators=(",", ":")))
    log(f"wrote {sum(len(r['frames']) for r in manifest['regions'].values())} frames for {cyc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
