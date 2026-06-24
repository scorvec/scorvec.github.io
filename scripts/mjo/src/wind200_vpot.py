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
# cycle pulls pf u @ 200/850 at all rmm_steps), so we only ever fetch v@200, and only at these
# frame steps — a ~daily-spread 6-frame subset that keeps the 50-member v pull light.
FRAME_IDX = (0, 3, 6, 9, 12, 15)

# velocity-potential ANOMALY shading: green = divergence (χ′<0, convection), orange = convergence
_VP_STOPS = ["#1b5e20", "#43a047", "#86c98a", "#cfe8cf", "#ffffff",
             "#fbe2bd", "#f0a64b", "#df6a1e", "#a8330f"]
VP_CMAP = LinearSegmentedColormap.from_list("vpot", _VP_STOPS)
VP_LEVELS = [-16, -12, -8, -5, -3, -1.5, 1.5, 3, 5, 8, 12, 16]   # ×1e6 m² s⁻¹


def _to_0360(da: xr.DataArray) -> xr.DataArray:
    return da.assign_coords(longitude=da.longitude % 360).sortby("longitude")


def velocity_potential(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = LMAX):
    """χ (m² s⁻¹) on a regular DH2 grid from 2-D u, v, by inverting ∇²χ = δ with spherical
    harmonics on a pole-free Gauss-Legendre grid. Returns (chi, dlat, dlon)."""
    u2d = _to_0360(u2d); v2d = _to_0360(v2d)
    glat, glon = pysh.expand.GLQGridCoord(lmax)         # Gauss-Legendre latitudes exclude the poles
    ug = u2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    vg = v2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    latr = np.deg2rad(glat); lonr = np.deg2rad(glon); cosp = np.cos(latr)[:, None]
    div = (np.gradient(ug, lonr, axis=1) + np.gradient(vg * cosp, latr, axis=0)) / (A * cosp)
    clm = pysh.SHGrid.from_array(div, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l); fac[1:] = -(A ** 2) / (l[1:] * (l[1:] + 1))   # χ_lm = -a²/(l(l+1)) δ_lm
    chi_clm = clm.copy(); chi_clm.coeffs *= fac[None, :, None]
    g = chi_clm.expand(grid="DH2")
    return g.data, np.array(g.lats()), np.array(g.lons())


def irrotational_wind(chi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """Irrotational (divergent) wind ∇χ on the χ grid: (1/(a cosφ) ∂χ/∂λ, 1/a ∂χ/∂φ)."""
    latr = np.deg2rad(dlat); lonr = np.deg2rad(dlon)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]   # clip the poles so the vectors stay finite
    uchi = np.gradient(chi, lonr, axis=1) / (A * cosp)
    vchi = np.gradient(chi, latr, axis=0) / A
    return uchi, vchi


def eval_vp_clim(coef: np.ndarray, doy: int) -> np.ndarray:
    """χ climatology (m² s⁻¹) for a day-of-year from mean+annual+semiannual harmonic coeffs."""
    w = 2 * np.pi * doy / 365.25
    return (coef[0] + coef[1] * np.cos(w) + coef[2] * np.sin(w)
            + coef[3] * np.cos(2 * w) + coef[4] * np.sin(2 * w))


def render(anom, uchi, vchi, dlat, dlon, init, valid, tag: str, out: Path):
    fig = plt.figure(figsize=(13.6, 6.6), constrained_layout=True)   # fixed canvas → frames don't jitter
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([-180, 180, -75, 75], crs=ccrs.PlateCarree())
    PC = ccrs.PlateCarree()
    cf = ax.contourf(dlon, dlat, anom / 1e6, levels=VP_LEVELS, cmap=VP_CMAP,
                     extend="both", transform=PC)
    s = max(1, len(dlat) // 26)                         # subsample the irrotational-wind vectors (sparser)
    ax.quiver(dlon[::s], dlat[::s], uchi[::s, ::s], vchi[::s, ::s], transform=PC,
              scale=420, width=0.0014, color="#222")
    ax.coastlines(linewidth=0.5, color="0.35")
    ax.axhline(0, color="0.55", lw=0.4, ls=":")
    # run / valid-time label on the map (persists through the animator embed)
    ax.text(0.006, 0.97, f"Init: {init:%HZ %a %d %b %Y}\nValid: {valid:%HZ %a %d %b %Y}  ({tag})",
            transform=ax.transAxes, fontsize=9, va="top", ha="left", family="monospace", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.6", linewidth=0.5))
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=55, shrink=0.78)
    cb.set_label("200 hPa velocity-potential anomaly (10⁶ m² s⁻¹)  ·  green = divergence / convection, "
                 "orange = convergence  ·  vectors = irrotational wind", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("AIFS-ENS 200 hPa Velocity-Potential Anomaly & Irrotational Wind (ensemble mean)",
                 fontsize=10.5, loc="left")
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
        chi, dlat, dlon = velocity_potential(ds["u"].isel(step=i), ds["v"].isel(step=i))
        anom = chi - eval_vp_clim(coef, int(valid.dayofyear))
        uchi, vchi = irrotational_wind(anom, dlat, dlon)
        tag = "analysis" if sh == 0 else f"forecast +{lead} d"
        fp = anim / f"F{i:02d}.webp"
        render(anom, uchi, vchi, dlat, dlon, init, valid, tag, fp)
        if sh == 0:
            render(anom, uchi, vchi, dlat, dlon, init, valid, tag, Path(args.out))
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
