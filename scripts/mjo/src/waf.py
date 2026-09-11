#!/usr/bin/env python3
"""Rossby wave-activity flux (Takaya–Nakamura 2001) at 250 hPa through the
AIFS-ENS forecast — a live "wave-packet radar".

Since 2026-09-07 (user review): (1) the flux is computed PER MEMBER and averaged — W is
quadratic in ψ′, so the flux of the ensemble mean fades with lead as members decorrelate,
which read as "waves dying" when they were merely uncertain; (2) 250 hPa instead of 200 —
the mid-latitude jet core and the stationary-wave ψ maximum sit at 250–300 hPa, and 200 hPa
is lowermost stratosphere poleward of ~50°N in winter; (3) ψ′ is low-passed with a 5-day
running mean along the lead before the flux (TN01 is a quasi-stationary theory; fast synoptic
packets enter the phase-independent form with error); (4) the shading is hatched where
members disagree on the sign; (5) the basic state is the ERA5 day-of-year climatology PLUS the
30-day mean anomaly of the AIFS 0-h member-0 analyses (waf_basic.py), and ψ′ is taken against the
same low-passed flow, so perturbation and basic state are consistent — in a year with a displaced
jet the packets are steered by the waveguide that is actually there.

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

with p̂ = 250/1000 and (U, V) the ERA5 1991–2020 day-of-year basic state
(build_waf_clim.py). Masked where |lat| < 20° or the basic-state wind < 3 m/s
(the quasi-stationary linear theory needs a westerly waveguide).

Data cost: u and v at 250 hPa for NMEMBERS perturbed members at the 16 daily frame steps
(~2 × 200 MB per cycle); the store caches them for the cycle.

    python src/waf.py --date 20260718 --time 00 \
        --anim-dir assets/sst/anim/waf --manifest assets/sst/anim/waf_manifest.json \
        --out assets/sst/waf.webp
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
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
import waf_basic as wb

A = 6.371e6
LMAX = 63                                    # ψ truncation (~2.8°): synoptic + planetary
LFILT = 15                                   # ∇·W shown at planetary scale (T15, ≳2500 km)
LEVEL = int(os.environ.get("WAF_LEVEL", "250"))
PHAT = LEVEL / 1000.0                        # p/p0 factor
UMIN, LATMIN = 3.0, 20.0                     # basic-state westerly / tropics mask
NMEMBERS = int(os.environ.get("WAF_MEMBERS", "25"))   # perturbed members for the flux average
TSMOOTH = 5                                  # days: running mean of ψ′ along the lead (quasi-stationary)
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / ("waf_clim_coeffs.nc" if LEVEL == 200 else f"waf_clim_coeffs_{LEVEL}.nc")   # U, V, ψ harmonic clims on the DH2 grid


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


def render(psi_a, wx, wy, divw, Uc, Vc, lat, lon, title: str, sub: str, out: Path, vlim: float, agree=None,
           spd_fc=None):
    """One row per hemisphere, 20-80 deg, sharing a colour bar.

    A single global map spent a fifth of its height on the tropical band the
    diagnostic masks by construction (no Rossby waveguide there) and more on
    the polar caps the flux never reaches, so the waveguides themselves came
    out small (user, 2026-09-11: "too small and has too much whitespace").
    Cropping to the two mid-latitude belts is the same data at roughly 1.6x the
    scale, and stacking them keeps one figure rather than the multi-panel decks
    the pages moved away from.
    """
    from matplotlib.gridspec import GridSpec
    proj = ccrs.PlateCarree(central_longitude=180)
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.6, 5.0))
    gs = GridSpec(2, 1, figure=fig, hspace=0.08, left=0.012, right=0.93, top=0.90, bottom=0.10)
    axes = []
    for row, (la0, la1) in enumerate(((20, 80), (-80, -20))):
        ax = fig.add_subplot(gs[row], projection=proj)
        ax.set_extent([-180, 180, la0, la1], crs=pc)
        cf = ax.contourf(lon, lat, -divw * 1e6, levels=np.linspace(-vlim, vlim, 21),
                         cmap="RdBu_r", extend="both", transform=pc)
        if agree is not None:                               # hatch where members DISAGREE on the sign of ∇·W
            ax.contourf(lon, lat, np.where(np.isfinite(agree), agree, 1.0), levels=[-0.01, 0.6], colors="none",
                        hatches=["//"], transform=pc, zorder=2)
        if spd_fc is not None:
            # the REAL waveguide: the forecast's own 200 hPa jet at this lead
            ax.contour(spd_fc[2], spd_fc[1], spd_fc[0], levels=[25, 35, 45],
                       colors="#1b5e20", linewidths=[0.8, 1.1, 1.5], alpha=0.85, transform=pc)
        # Arrows: fewer and stronger than before. At 65% of the flux magnitude the
        # field read as noise; the packets are the top quarter and they are what the
        # chart is for.
        s = max(1, lat.size // 30)
        Wm = np.hypot(wx, wy)
        show = Wm > np.nanpercentile(Wm, 75)
        qx = np.where(show, wx, np.nan)[::s, ::s]
        qy = np.where(show, wy, np.nan)[::s, ::s]
        q = ax.quiver(lon[::s], lat[::s], qx, qy, transform=pc,
                      color="#111", width=0.0016, scale=2200, headwidth=3.6, alpha=0.85,
                      pivot="tail", zorder=6)
        ax.coastlines(lw=0.45, color="0.62")
        axes.append((ax, cf, q))
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=11.5, fontweight="bold")
    cb = fig.colorbar(axes[0][1], ax=[a for a, _, _ in axes], pad=0.008, fraction=0.030, aspect=34)
    cb.set_label("−∇·W  (10⁻⁶ m s⁻²; red = convergence → amplification)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    axes[1][0].quiverkey(axes[1][2], 0.88, -0.16, 100, "W = 100 m²/s²", labelpos="E",
                         fontproperties={"size": 7.5})
    fig.text(0.012, 0.012, sub, ha="left", va="bottom", fontsize=8, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=118, bbox_inches="tight", facecolor="white")
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
    frame_steps = tuple(all_steps[:16])
    cyc = ecmwf.Cycle(args.date, args.time)
    up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", (LEVEL,), frame_steps, NMEMBERS))
    vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "v", "pl", (LEVEL,), frame_steps, NMEMBERS))
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    U = xr.open_dataset(up, **ku)["u"]; Vv = xr.open_dataset(vp, **ku)["v"]
    if "isobaricInhPa" in U.dims:
        U = U.sel(isobaricInhPa=LEVEL); Vv = Vv.sel(isobaricInhPa=LEVEL)
    U = U.squeeze(drop=True); Vv = Vv.squeeze(drop=True)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    steps_h = (U.step / np.timedelta64(1, "h")).round().astype(int).values
    members = list(U.number.values)
    nstep = len(steps_h)

    # low-passed analysis basic state: archive this cycle's 0-h control, then the 30-day mean anomaly
    anom, n_an = None, 0
    try:
        ucp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", (LEVEL,), (0,)))
        vcp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", (LEVEL,), (0,)))
        uc0 = xr.open_dataset(ucp, **ku)["u"].squeeze(drop=True); vc0 = xr.open_dataset(vcp, **ku)["v"].squeeze(drop=True)
        psi0, p0lat, p0lon = streamfunction_psi(uc0, vc0)
        hist = wb.update_history(Path(args.anim_dir) / wb.HIST_NAME,
                                 wb.record(wb.to3(uc0.values, uc0.latitude.values, uc0.longitude.values),
                                           wb.to3(vc0.values, vc0.latitude.values, vc0.longitude.values),
                                           wb.to3(psi0, p0lat, p0lon), init, LEVEL))
        anom, n_an = wb.lowpass_anomaly(hist, eval_clim, c, clat, clon, init)
        print(f"  basic state: {'clim + 30-d analysis anomaly' if anom else 'climatology only'} ({n_an} analyses in the window)", flush=True)
    except Exception as ex:                                       # noqa: BLE001
        print(f"  basic-state history unavailable ({str(ex)[:80]}); climatology only", flush=True)
    psi_ref = (lambda doy: eval_clim(c["psi"].values, doy) + anom["psi"]) if anom else (lambda doy: eval_clim(c["psi"].values, doy))
    U_ref = (lambda doy: eval_clim(c["U"].values, doy) + anom["u"]) if anom else (lambda doy: eval_clim(c["U"].values, doy))
    V_ref = (lambda doy: eval_clim(c["V"].values, doy) + anom["v"]) if anom else (lambda doy: eval_clim(c["V"].values, doy))

    # ψ′ per member and step (the expensive part: nmember × nstep inversions, ~0.1 s each)
    psi_a = np.zeros((len(members), nstep, len(clat), len(clon)), dtype="float32")
    for m, num in enumerate(members):
        for i in range(nstep):
            valid = init + pd.Timedelta(hours=int(steps_h[i]))
            psi, plat, plon = streamfunction_psi(U.sel(number=num).isel(step=i), Vv.sel(number=num).isel(step=i))
            if m == 0 and i == 0:
                assert np.allclose(plat, clat) and np.allclose(plon, clon), \
                    "clim grid != live DH2 grid — rebuild waf_clim with the same LMAX"
            psi_a[m, i] = psi - psi_ref(float(valid.dayofyear))
    # quasi-stationary: running mean of ψ′ along the lead (TSMOOTH days, shrinking at the ends)
    half = TSMOOTH // 2
    psi_s = np.empty_like(psi_a)
    for i in range(nstep):
        psi_s[:, i] = psi_a[:, max(0, i - half):min(nstep, i + half + 1)].mean(axis=1)
    fields = []
    for i, sh in enumerate(steps_h):
        valid = init + pd.Timedelta(hours=int(sh))
        doy = float(valid.dayofyear)
        Uc, Vc = U_ref(doy), V_ref(doy)
        WX, WY, DV = [], [], []
        for m in range(len(members)):
            wx, wy, divw = tn01_flux(psi_s[m, i].astype("float64"), Uc, Vc, clat, clon)
            WX.append(wx); WY.append(wy); DV.append(divw)
        WX, WY, DV = np.stack(WX), np.stack(WY), np.stack(DV)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN masked cells
            wx, wy, divw = np.nanmean(WX, 0), np.nanmean(WY, 0), np.nanmean(DV, 0)
        sign = np.sign(DV); agree = np.nanmax(np.stack([(sign > 0).mean(0), (sign < 0).mean(0)]), 0)
        agree = np.where(np.isfinite(divw), agree, np.nan)
        um = U.isel(step=i).mean("number"); vm = Vv.isel(step=i).mean("number")
        spd_fc = (np.hypot(um.values, vm.values), um.latitude.values, um.longitude.values)
        fields.append((valid, sh, psi_s[:, i].mean(0), wx, wy, divw, Uc, Vc, spd_fc, agree))
    vlim = float(np.nanpercentile(np.abs(np.stack([f[5] for f in fields])) * 1e6, 99.0)) or 5.0

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    basic = (f"ERA5 clim + 30-day mean anomaly of the AIFS 0-h analyses ({n_an} analyses)" if anom else "ERA5 climatology (analysis history too short)")
    sub = (f"arrows = Takaya–Nakamura (2001) wave-activity flux, mean of {len(members)} members' fluxes · basic state and ψ′ reference = {basic} "
           f"({TSMOOTH}-day running mean of ψ′ along the lead) · shading = −∇·W at planetary scale (T15; red ⇒ downstream amplification), "
           "hatched where fewer than 60% of members agree on the sign\n"
           f"green = the ensemble-mean {LEVEL} hPa jet at this lead (25/35/45 m/s) — the waveguide the packets follow · "
           "masked equatorward of 20° / basic-state wind < 3 m/s · each row is one hemisphere's 20–80° belt")
    frames = []
    for i, (valid, sh, pa, wx, wy, divw, Uc, Vc, spd_fc, agree) in enumerate(fields):
        fp = anim / f"F{i:02d}.webp"
        lead = int(round(sh / 24))
        render(pa, wx, wy, divw, Uc, Vc, clat, clon,
               f"Rossby wave-activity flux (TN01) {LEVEL} hPa — AIFS-ENS, member-mean flux · "
               f"init {init:%Y-%m-%d %HZ} · day {lead} (valid {valid:%a %b %d})",
               sub, fp, vlim, agree=agree, spd_fc=spd_fc)
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {lead} · {valid:%b %d}"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"waf": {"label": f"Wave-activity flux (TN01, {LEVEL} hPa)",
                                "n_frames": len(frames), "frames": frames}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    # static latest = the analysis frame
    valid, sh, pa, wx, wy, divw, Uc, Vc, spd_fc, agree = fields[0]
    render(pa, wx, wy, divw, Uc, Vc, clat, clon,
           f"Rossby wave-activity flux (TN01) {LEVEL} hPa — analysis {init:%Y-%m-%d %HZ}",
           sub, Path(args.out), vlim, agree=agree, spd_fc=spd_fc)
    print(f"  wrote {len(frames)} frames + manifest; conv vlim ±{vlim:.0f}×10⁻⁶ m/s²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
