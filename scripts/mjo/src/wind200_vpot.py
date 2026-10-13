#!/usr/bin/env python3
"""AIFS-ENS 200 hPa velocity-potential anomaly + irrotational (divergent) wind.

The large-scale divergent circulation aloft, the cleanest view of where convection is firing
(upper-level divergence, χ minimum) and subsiding (convergence, χ maximum). Tropical-Tidbits
style: velocity-potential anomaly shaded (green = divergence/convection, orange = convergence),
with the irrotational-wind anomaly as vectors pointing out of the divergence centres.

Ensemble mean: AIFS-ENS publishes no `em` product for these fields, so we average the 50 `pf`
members of u, v @ 200 hPa. Velocity potential is LINEAR in the wind (divergence is linear and
the Poisson inversion is linear), so χ(mean wind) = mean(χ) exactly — averaging the wind first
is both correct and cheap. χ is obtained by inverting ∇²χ = δ on the sphere with spherical
harmonics (pyshtools), on a pole-free Gauss-Legendre grid so the 1/cosφ metric never blows up.
Anomaly = χ − the ERA5 1991-2020 day-of-year harmonic climatology (build_vp200_clim.py).

    python src/wind200_vpot.py --date 20260624 --time 12 \
        --anim-dir assets/sst/anim/wind200 --manifest assets/sst/anim/wind200_manifest.json \
        --out assets/sst/wind200.webp
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import pyshtools as pysh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf

A = 6.371e6                                              # Earth radius (m)
LMAX = 106                                              # spherical-harmonic truncation (~1.7°, smooth χ)
REF = Path(__file__).resolve().parent.parent / "data" / "reference"   # scripts/mjo/data/reference (committed clims)
CLIM = REF / "vp200_clim_coeffs.nc"                     # ERA5 1991-2020 χ harmonic coeffs (committed)
# Which of the 16 rmm_steps to animate. u@200 is reused from the cached RMM pf download (every
# cycle pulls pf u @ 200/850 at all rmm_steps), so we only ever fetch v@200 at these frame steps.
# The rmm_steps are 00Z-anchored (one per day: analysis + day 1..15), so all 16 = daily frames.
FRAME_IDX = tuple(range(16))

# velocity-potential ANOMALY shading: green = divergence (χ′<0, convection), orange = convergence
_VP_STOPS = ["#1b5e20", "#43a047", "#86c98a", "#cfe8cf", "#ffffff",
             "#fbe2bd", "#f0a64b", "#df6a1e", "#a8330f"]
VP_CMAP = LinearSegmentedColormap.from_list("vpot", _VP_STOPS)
VP_LEVELS = [-16, -12, -8, -5, -3, -1.5, 1.5, 3, 5, 8, 12, 16]   # ×1e6 m² s⁻¹
WIND_LEVELS = [15, 30, 40, 50, 60, 70]                           # 200 hPa wind-speed jet contours (m/s)


def _to_0360(da: xr.DataArray) -> xr.DataArray:
    """Longitudes mapped to [0, 360) and sorted; a repeated seam column (0 and 360) is dropped."""
    da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    keep = np.r_[True, np.diff(da.longitude.values) > 1e-9]
    return da.isel(longitude=np.flatnonzero(keep))


def _pole_row(row: np.ndarray) -> np.ndarray:
    """A wind component at a pole keeps only zonal wavenumber 1 of the nearest row (cross-polar flow in local
    east/north components)."""
    c = np.fft.rfft(row)
    keep = np.zeros_like(c); keep[1:2] = c[1:2]
    return np.fft.irfft(keep, n=row.size)


def _padded(da: xr.DataArray) -> xr.DataArray:
    """2-D (latitude, longitude) field in [0, 360) padded with a column at lon - 360 and lon + 360 and, where the grid
    stops short of the poles, pole rows, so interpolation to ANY Gauss-Legendre grid stays inside the data.
    2026-10-12 (quantmet 0.3.1 fix): without the wrap, the GLQ columns past the input's last longitude came out NaN
    (chi 100 % NaN for a 1.5 or 2.5 deg input at lmax >= 120)."""
    da = da.sortby("latitude").transpose("latitude", "longitude")
    lon = da.longitude.values
    da = xr.concat([da.isel(longitude=[-1]).assign_coords(longitude=[lon[-1] - 360.0]), da,
                    da.isel(longitude=[0]).assign_coords(longitude=[lon[0] + 360.0])], dim="longitude")
    lat = da.latitude.values
    rows, lats = [da.values], [lat]
    if lat[0] > -90.0 + 1e-9:
        rows.insert(0, _pole_row(da.values[0])[None]); lats.insert(0, [-90.0])
    if lat[-1] < 90.0 - 1e-9:
        rows.append(_pole_row(da.values[-1])[None]); lats.append([90.0])
    return xr.DataArray(np.concatenate(rows, 0), dims=("latitude", "longitude"),
                        coords={"latitude": np.concatenate(lats), "longitude": da.longitude.values})


def _ddlon(f: np.ndarray, lon_deg) -> np.ndarray:
    """d f / d lambda (per radian) on the last axis: periodic centred differences on a global uniform grid (the GLQ
    grid, or the extended DH2 grid whose 360 column repeats 0), np.gradient on a regional one. 2026-10-12: replaces
    np.gradient everywhere, whose one-sided differences at the 0/360 seam put a first-order error (and a spurious
    line of divergence) along the prime meridian."""
    lon = np.asarray(lon_deg, float)
    d = np.diff(lon); dl = d.mean()
    if np.allclose(d, dl, rtol=1e-6, atol=1e-9):
        dlr = np.deg2rad(dl)
        if np.isclose(lon[-1] - lon[0], 360.0, atol=1e-6):            # closed grid: last column = first
            core = f[..., :-1]
            g = (np.roll(core, -1, axis=-1) - np.roll(core, 1, axis=-1)) / (2 * dlr)
            return np.concatenate([g, g[..., :1]], axis=-1)
        if np.isclose(lon[-1] - lon[0] + dl, 360.0, atol=1e-6):       # full circle, open
            return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dlr)
    return np.gradient(f, np.deg2rad(lon), axis=-1)


def velocity_potential(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = LMAX):
    """χ (m² s⁻¹) on a regular DH2 grid from 2-D u, v, by inverting ∇²χ = δ with spherical
    harmonics on a pole-free Gauss-Legendre grid. Returns (chi, dlat, dlon)."""
    u2d = _padded(_to_0360(u2d)); v2d = _padded(_to_0360(v2d))
    glat, glon = pysh.expand.GLQGridCoord(lmax)         # Gauss-Legendre latitudes exclude the poles
    ug = u2d.interp(latitude=glat, longitude=glon).values
    vg = v2d.interp(latitude=glat, longitude=glon).values
    latr = np.deg2rad(glat); cosp = np.cos(latr)[:, None]
    div = (_ddlon(ug, glon) + np.gradient(vg * cosp, latr, axis=0)) / (A * cosp)
    clm = pysh.SHGrid.from_array(div, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l); fac[1:] = -(A ** 2) / (l[1:] * (l[1:] + 1))   # χ_lm = -a²/(l(l+1)) δ_lm
    chi_clm = clm.copy(); chi_clm.coeffs *= fac[None, :, None]
    g = chi_clm.expand(grid="DH2")
    return g.data, np.array(g.lats()), np.array(g.lons())


def irrotational_wind(chi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """Irrotational (divergent) wind ∇χ on the χ grid: (1/(a cosφ) ∂χ/∂λ, 1/a ∂χ/∂φ); ∂/∂λ periodic on a global grid."""
    latr = np.deg2rad(dlat)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]   # clip the poles so the vectors stay finite
    uchi = _ddlon(chi, dlon) / (A * cosp)
    vchi = np.gradient(chi, latr, axis=0) / A
    return uchi, vchi


def eval_vp_clim(coef: np.ndarray, doy: int) -> np.ndarray:
    """χ climatology (m² s⁻¹) for a day-of-year from mean+annual+semiannual harmonic coeffs."""
    w = 2 * np.pi * doy / 365.25
    return (coef[0] + coef[1] * np.cos(w) + coef[2] * np.sin(w)
            + coef[3] * np.cos(2 * w) + coef[4] * np.sin(2 * w))


def render(anom, uchi, vchi, ufull, vfull, dlat, dlon, init, valid, tag: str, out: Path):
    # fixed canvas and FIXED axes positions (2026-10-04: constrained_layout let the fixed-aspect map grow until the title
    # ran off the top of the image, user: "The title of this plot is cut off")
    fig = plt.figure(figsize=(13.6, 6.6))
    ax = fig.add_axes([0.01, 0.17, 0.98, 0.775], projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([-180, 180, -75, 75], crs=ccrs.PlateCarree())
    PC = ccrs.PlateCarree()
    cf = ax.contourf(dlon, dlat, anom / 1e6, levels=VP_LEVELS, cmap=VP_CMAP,
                     extend="both", transform=PC)
    # full 200 hPa wind speed (jet streams) as line contours
    cj = ax.contour(dlon, dlat, np.hypot(ufull, vfull), levels=WIND_LEVELS, colors="#111111",
                    linewidths=1.1, alpha=0.9, transform=PC)
    ax.clabel(cj, fmt="%d", fontsize=6.5, inline=True, inline_spacing=2)
    s = max(1, len(dlat) // 26)                         # subsample the irrotational-wind vectors (sparser)
    ax.quiver(dlon[::s], dlat[::s], uchi[::s, ::s], vchi[::s, ::s], transform=PC,
              scale=420, width=0.0014, color="#222")
    ax.coastlines(linewidth=0.5, color="0.35")
    ax.axhline(0, color="0.55", lw=0.4, ls=":")
    # run / valid-time label on the map (persists through the animator embed)
    ax.text(0.006, 0.97, f"Init: {init:%HZ %a %d %b %Y}\nValid: {valid:%HZ %a %d %b %Y}  ({tag})",
            transform=ax.transAxes, fontsize=9, va="top", ha="left", family="monospace", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.6", linewidth=0.5))
    cb = fig.colorbar(cf, cax=fig.add_axes([0.12, 0.095, 0.76, 0.03]), orientation="horizontal")
    cb.set_label("200 hPa velocity-potential anomaly (m² s⁻¹ × 10⁻⁶); green = divergence / convection, "
                 "orange = convergence  ·  vectors = irrotational wind  ·  contours = 200 hPa wind speed (m s⁻¹)",
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.canvas.draw()                                   # the map's real (aspect-fitted) box, so the title sits on its left edge
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.012, "AIFS-ENS 200 hPa Velocity-Potential Anomaly & Irrotational Wind (ensemble mean)",
             fontsize=10.5, ha="left", va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)          # fixed size (no tight bbox) so every frame matches


def _ens_mean(cyc, all_steps, frame_steps) -> xr.Dataset:
    """Ensemble-mean (50 pf members) u, v @ 200 hPa at the frame steps. u@200 is reused from the
    cached RMM pf download (pf u @ 200/850 at every rmm_step) — no re-download; only v@200 is
    fetched, and only at the frame steps."""
    up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", ecmwf.LEVELS_RMM, tuple(all_steps)))
    u = xr.open_dataset(up, engine="cfgrib", backend_kwargs={"indexpath": ""})["u"]
    if "isobaricInhPa" in u.dims:
        u = u.sel(isobaricInhPa=200)
    vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "v", "pl", (200,), tuple(frame_steps)))
    v = xr.open_dataset(vp, engine="cfgrib", backend_kwargs={"indexpath": ""})["v"]
    u = u.sel(step=v.step)                              # the frame steps only
    return xr.Dataset({"u": u.mean("number"), "v": v.mean("number")}).squeeze(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/wind200")
    ap.add_argument("--manifest", default="assets/sst/anim/wind200_manifest.json")
    ap.add_argument("--out", default="assets/sst/wind200.webp")
    args = ap.parse_args()

    if not CLIM.exists():
        print(f"  clim {CLIM} missing — run build_vp200_clim.py first; skipping.", file=sys.stderr)
        return 0
    coef = xr.open_dataset(CLIM)["coef"].values

    import download_aifs
    all_steps = list(download_aifs.rmm_steps(args.time))            # 00Z-anchored leads (init-time dependent)
    frame_steps = [all_steps[i] for i in FRAME_IDX if i < len(all_steps)]
    cyc = ecmwf.Cycle(args.date, args.time)
    ds = _ens_mean(cyc, all_steps, frame_steps)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    steps_h = (ds.step / np.timedelta64(1, "h")).round().astype(int).values

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, sh in enumerate(steps_h):
        valid = init + pd.Timedelta(hours=int(sh))
        lead = int(round(sh / 24))
        u2d, v2d = ds["u"].isel(step=i), ds["v"].isel(step=i)
        chi, dlat, dlon = velocity_potential(u2d, v2d)
        anom = chi - eval_vp_clim(coef, int(valid.dayofyear))
        uchi, vchi = irrotational_wind(anom, dlat, dlon)
        uf = _to_0360(u2d).interp(latitude=dlat, longitude=dlon).transpose("latitude", "longitude").values
        vf = _to_0360(v2d).interp(latitude=dlat, longitude=dlon).transpose("latitude", "longitude").values
        tag = "analysis" if sh == 0 else f"forecast +{lead} d"
        fp = anim / f"F{i:02d}.webp"
        render(anom, uchi, vchi, uf, vf, dlat, dlon, init, valid, tag, fp)
        if sh == 0:
            render(anom, uchi, vchi, uf, vf, dlat, dlon, init, valid, tag, Path(args.out))
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": "analysis" if sh == 0 else f"+{lead} d  ({valid:%a %b %d})"})
        print(f"  rendered {fp.name}  ({tag}, χ′ range {anom.min()/1e6:.0f}..{anom.max()/1e6:.0f})", flush=True)
    mani = {"ver": int(_time.time()), "regions": {"wind200": {
        "label": "200 hPa velocity potential & irrotational wind (AIFS-ENS)",
        "n_frames": len(frames), "frames": frames}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"  wrote {len(frames)} frames + manifest + static {Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
