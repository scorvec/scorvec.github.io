#!/usr/bin/env python3
"""Rossby wave-activity flux (Takaya–Nakamura 2001) at 200 hPa through the
AIFS-ENS forecast — a live "wave-packet radar".

W is the phase-independent flux of quasi-stationary Rossby wave activity: its
vectors point along the group velocity (where packet energy is HEADING, ducted
along the jet waveguides), and its convergence marks where the downstream flow
will amplify days later — the mechanism behind downstream development and
blocking onset, and the conduit that turns tropical (El Niño / MJO) forcing
into the PNA and other teleconnection arcs.

    ψ′ = ψ − ψ_clim(doy)   (streamfunction anomaly; ∇²ψ = ζ inverted in
                            spherical harmonics — same solver family as χ)
    W  = p̂ cosφ / (2|U|) ·
         [ U(ψ′ₓ² − ψ′ψ′ₓₓ) + V(ψ′ₓψ′ᵧ − ψ′ψ′ₓᵧ) ,
           U(ψ′ₓψ′ᵧ − ψ′ψ′ₓᵧ) + V(ψ′ᵧ² − ψ′ψ′ᵧᵧ) ]        (TN01 eq. 38, horizontal)

with p̂ = 200/1000 and (U, V) the ERA5 1991–2020 day-of-year basic state
(build_waf_clim.py). Masked where |lat| < 20° or the basic-state wind < 3 m/s
(the quasi-stationary linear theory needs a westerly waveguide).

Data cost: ZERO new downloads — ens-mean u@200 (RMM/AAM pull) and v@200
(velocity-potential pull) at every daily lead are already cached per cycle.

    python src/waf.py --date 20260718 --time 00 \
        --anim-dir assets/sst/anim/waf --manifest assets/sst/anim/waf_manifest.json \
        --out assets/sst/waf.webp
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyshtools as pysh
import cartopy.crs as ccrs

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf
from wind200_vpot import _ens_mean, _to_0360

A = 6.371e6
LMAX = 63                                    # ψ truncation (~2.8°): synoptic + planetary
LFILT = 15                                   # ∇·W shown at planetary scale (T15, ≳2500 km)
PHAT = 200.0 / 1000.0                        # p/p0 factor at 200 hPa
UMIN, LATMIN = 3.0, 20.0                     # basic-state westerly / tropics mask
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / "waf_clim_coeffs.nc"            # U, V, ψ harmonic clims on the DH2 grid


def streamfunction_psi(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = LMAX):
    """ψ (m² s⁻¹) on a regular DH2 grid from 2-D u, v: invert ∇²ψ = ζ in spherical
    harmonics on a pole-free Gauss–Legendre grid (mirror of velocity_potential)."""
    u2d = _to_0360(u2d); v2d = _to_0360(v2d)
    glat, glon = pysh.expand.GLQGridCoord(lmax)
    ug = u2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    vg = v2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    latr = np.deg2rad(glat); lonr = np.deg2rad(glon); cosp = np.cos(latr)[:, None]
    vort = (np.gradient(vg, lonr, axis=1) - np.gradient(ug * cosp, latr, axis=0)) / (A * cosp)
    clm = pysh.SHGrid.from_array(vort, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l); fac[1:] = -(A ** 2) / (l[1:] * (l[1:] + 1))
    psi_clm = clm.copy(); psi_clm.coeffs *= fac[None, :, None]
    g = psi_clm.expand(grid="DH2")
    return g.data, np.array(g.lats()), np.array(g.lons())


def tn01_flux(psi_a: np.ndarray, U: np.ndarray, V: np.ndarray,
              lat: np.ndarray, lon: np.ndarray):
    """Horizontal TN01 W = (Wx, Wy) from ψ′ and the basic state, all on one grid.
    Spherical derivatives: ∂x = (a cosφ)⁻¹∂λ, ∂y = a⁻¹∂φ. Masked outside the
    westerly waveguide."""
    latr = np.deg2rad(lat); lonr = np.deg2rad(lon)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    px = np.gradient(psi_a, lonr, axis=1) / (A * cosp)
    py = np.gradient(psi_a, latr, axis=0) / A
    pxx = np.gradient(px, lonr, axis=1) / (A * cosp)
    pxy = np.gradient(px, latr, axis=0) / A
    pyy = np.gradient(py, latr, axis=0) / A
    spd = np.hypot(U, V)
    pref = PHAT * cosp / (2.0 * np.maximum(spd, 1e-6))
    wx = pref * (U * (px ** 2 - psi_a * pxx) + V * (px * py - psi_a * pxy))
    wy = pref * (U * (px * py - psi_a * pxy) + V * (py ** 2 - psi_a * pyy))
    bad = (spd < UMIN) | (np.abs(lat)[:, None] < LATMIN)
    wx[bad] = np.nan; wy[bad] = np.nan
    # flux divergence ∇·W: NEGATIVE (convergence) marks where wave activity
    # piles up — the downstream-amplification precursor. Spectrally truncated
    # to planetary/synoptic scales (l ≤ LFILT): second derivatives of the flux
    # carry gridscale ripple that a light gaussian cannot tame.
    wx0 = np.nan_to_num(wx); wy0 = np.nan_to_num(wy)
    divw = (np.gradient(wx0, lonr, axis=1) / (A * cosp)
            + np.gradient(wy0 * cosp, latr, axis=0) / (A * cosp))
    g = pysh.SHGrid.from_array(divw, grid="DH")
    clm = g.expand()
    clm.coeffs[:, LFILT + 1:, :] = 0.0
    divw = clm.expand(grid="DH2").data
    divw[bad] = np.nan
    return wx, wy, divw


def eval_clim(coefs: np.ndarray, doy: float) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    b = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.tensordot(b, coefs, axes=(0, 0))


def render(psi_a, wx, wy, divw, Uc, Vc, lat, lon, title: str, sub: str, out: Path, vlim: float,
           spd_fc=None):
    fig = plt.figure(figsize=(12.8, 6.4))
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    ax.set_global()
    # shade CONVERGENCE (−∇·W, 10⁻⁶ m s⁻²): red = wave activity piling up →
    # the downstream flow amplifies over the following days; blue = emission.
    cf = ax.contourf(lon, lat, -divw * 1e6, levels=np.linspace(-vlim, vlim, 21),
                     cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
    if spd_fc is not None:
        # the REAL waveguide: the forecast's own 200 hPa jet at this lead
        ax.contour(spd_fc[2], spd_fc[1], spd_fc[0], levels=[25, 35, 45],
                   colors="#1b5e20", linewidths=[0.8, 1.1, 1.5], alpha=0.85,
                   transform=ccrs.PlateCarree())
    s = max(1, lat.size // 36)
    Wm = np.hypot(wx, wy)
    show = Wm > np.nanpercentile(Wm, 65)                  # hide the weak background flux
    qx = np.where(show, wx, np.nan)[::s, ::s]
    qy = np.where(show, wy, np.nan)[::s, ::s]
    q = ax.quiver(lon[::s], lat[::s], qx, qy, transform=ccrs.PlateCarree(),
                  color="#111", width=0.0016, scale=2200, headwidth=3.6, alpha=0.85,
                  pivot="tail", zorder=6)
    ax.quiverkey(q, 0.90, -0.045, 100, "W = 100 m²/s²", labelpos="E", fontproperties={"size": 7.5})
    ax.coastlines(lw=0.45, color="0.62")
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    cb = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.032)
    cb.set_label("−∇·W  (10⁻⁶ m s⁻²; red = convergence → amplification)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    # anchor to the AXES, not the figure: the map shrinks to its forced 2:1 aspect
    # inside the figure box, and figure-coord text pins the tight bbox to the full
    # (mostly empty) figure height — axes-coord text collapses with the map instead.
    ax.text(0.5, -0.085, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/waf")
    ap.add_argument("--manifest", default="assets/sst/anim/waf_manifest.json")
    ap.add_argument("--out", default="assets/sst/waf.webp")
    args = ap.parse_args()
    if not CLIM.exists():
        print(f"  clim {CLIM} missing — run build_waf_clim.py first; skipping.", file=sys.stderr)
        return 0
    c = xr.open_dataset(CLIM)
    clat, clon = c.latitude.values, c.longitude.values

    import download_aifs
    all_steps = list(download_aifs.rmm_steps(args.time))
    frame_steps = all_steps[:16]
    cyc = ecmwf.Cycle(args.date, args.time)
    ds = _ens_mean(cyc, all_steps, frame_steps)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    steps_h = (ds.step / np.timedelta64(1, "h")).round().astype(int).values

    # pass 1: ψ′ per frame for a common colour scale
    fields = []
    for i, sh in enumerate(steps_h):
        valid = init + pd.Timedelta(hours=int(sh))
        psi, plat, plon = streamfunction_psi(ds["u"].isel(step=i), ds["v"].isel(step=i))
        assert np.allclose(plat, clat) and np.allclose(plon, clon), \
            "clim grid != live DH2 grid — rebuild waf_clim with the same LMAX"
        doy = float(valid.dayofyear)
        psi_a = psi - eval_clim(c["psi"].values, doy)
        Uc, Vc = eval_clim(c["U"].values, doy), eval_clim(c["V"].values, doy)
        wx, wy, divw = tn01_flux(psi_a, Uc, Vc, clat, clon)
        uf = ds["u"].isel(step=i); vf = ds["v"].isel(step=i)
        spd_fc = (np.hypot(uf.values, vf.values),
                  uf.latitude.values, uf.longitude.values)
        fields.append((valid, sh, psi_a, wx, wy, divw, Uc, Vc, spd_fc))
    vlim = float(np.nanpercentile(np.abs(np.stack([f[5] for f in fields])) * 1e6, 99.0)) or 5.0

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    sub = ("arrows = Takaya–Nakamura (2001) wave-activity flux, computed on the CLIMATOLOGICAL basic state "
           "(ψ′ = forecast − clim) · shading = −∇·W at planetary scale (T15; red ⇒ downstream amplification)\n"
           "green = the forecast's own 200 hPa jet at this lead (25/35/45 m/s) — the waveguide the packets follow · "
           "masked equatorward of 20° / basic-state wind < 3 m/s")
    frames = []
    for i, (valid, sh, psi_a, wx, wy, divw, Uc, Vc, spd_fc) in enumerate(fields):
        fp = anim / f"F{i:02d}.webp"
        lead = int(round(sh / 24))
        render(psi_a, wx, wy, divw, Uc, Vc, clat, clon,
               f"Rossby wave-activity flux (TN01) 200 hPa — AIFS-ENS mean · "
               f"init {init:%Y-%m-%d %HZ} · day {lead} (valid {valid:%a %b %d})",
               sub, fp, vlim, spd_fc=spd_fc)
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {lead} · {valid:%b %d}"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"waf": {"label": "Wave-activity flux (TN01, 200 hPa)",
                                "n_frames": len(frames), "frames": frames}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    # static latest = the analysis frame
    valid, sh, psi_a, wx, wy, divw, Uc, Vc, spd_fc = fields[0]
    render(psi_a, wx, wy, divw, Uc, Vc, clat, clon,
           f"Rossby wave-activity flux (TN01) 200 hPa — analysis {init:%Y-%m-%d %HZ}",
           sub, Path(args.out), vlim, spd_fc=spd_fc)
    print(f"  wrote {len(frames)} frames + manifest; conv vlim ±{vlim:.0f}×10⁻⁶ m/s²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
