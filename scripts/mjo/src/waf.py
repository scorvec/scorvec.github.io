#!/usr/bin/env python3
"""Rossby wave-activity flux (Takaya–Nakamura 2001) at 250 hPa through the
AIFS-ENS forecast — a live "wave-packet radar".

Design: (1) the flux is computed PER MEMBER and averaged — W is quadratic in ψ′, so the
flux of the ensemble mean fades with lead as members decorrelate, which reads as "waves
dying" when they are merely uncertain; (2) 250 hPa instead of 200 —
the mid-latitude jet core and the stationary-wave ψ maximum sit at 250–300 hPa, and 200 hPa
is lowermost stratosphere poleward of ~50°N in winter; (3) ψ′ is low-passed with a 5-day
running mean along the lead before the flux (TN01 is a quasi-stationary theory; fast synoptic
packets enter the phase-independent form with error); (4) the shading is hatched where
members disagree on the sign; (5) the basic state is the ERA5 day-of-year climatology PLUS the
30-day mean anomaly of the AIFS 0-h member-0 analyses (waf_basic.py), and ψ′ is taken against the
same low-passed flow, so perturbation and basic state are consistent — in a year with a displaced
jet the packets are steered by the waveguide that is actually there.

W and its T15 divergence come from quantmet.waf.tn01_flux (0.4.1, div_method="zero"): the
masked cells' divergence is set to zero before the truncation, so the mask-edge step does not
leak into the filter. Against truncating the divergence as computed (the method used for frames
before 2026-10-12), synthetic and live tests (scorvec/quantmet-tools#1) give a known-divergence
error of 0.16 instead of 0.19 and a spurious divergence of a divergence-free flux of 0.06 instead
of 0.12 (live 250 hPa flux, real mask); the shading differs by ~8 % rms. The arrows (W itself)
and the mask are the same under both methods.

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
import argparse
import json
import os
import sys
import warnings
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
from wind200_vpot import _to_0360, _padded, _ddlon
import waf_basic as wb
from doycal import noleap_doy  # common-year day of year (29 Feb = 59.5)

# The flux and its T15 divergence come from quantmet 0.4.1. div_method="zero" (sets the masked cells' divergence to
# zero before the truncation) has a smaller error than "as_computed" in quantmet's synthetic tests
# (tests/test_waf_divergence.py, CHANGELOG 0.4.1, scorvec/quantmet-tools#1).
from quantmet.waf import tn01_flux

A = 6.371e6
LMAX = 63  # ψ truncation (~2.8°): synoptic + planetary
LFILT = 15  # ∇·W shown at planetary scale (T15, ≳2500 km)
LEVEL = int(os.environ.get("WAF_LEVEL", "250"))
UMIN, LATMIN = 3.0, 20.0  # basic-state westerly / tropics mask
NMEMBERS = int(os.environ.get("WAF_MEMBERS", "25"))  # perturbed members for the flux average
TSMOOTH = 5  # days: running mean of ψ′ along the lead (quasi-stationary)
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / (
    "waf_clim_coeffs.nc" if LEVEL == 200 else f"waf_clim_coeffs_{LEVEL}.nc"
)  # U, V, ψ harmonic clims on the DH2 grid


def streamfunction_psi(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = LMAX):
    """ψ (m² s⁻¹) on a regular DH2 grid from 2-D u, v: invert ∇²ψ = ζ in spherical
    harmonics on a pole-free Gauss–Legendre grid (mirror of velocity_potential)."""
    u2d = _padded(_to_0360(u2d))
    v2d = _padded(_to_0360(v2d))  # longitude wrap + pole rows
    glat, glon = pysh.expand.GLQGridCoord(lmax)
    ug = u2d.interp(latitude=glat, longitude=glon).values
    vg = v2d.interp(latitude=glat, longitude=glon).values
    latr = np.deg2rad(glat)
    cosp = np.cos(latr)[:, None]
    vort = (_ddlon(vg, glon) - np.gradient(ug * cosp, latr, axis=0)) / (A * cosp)  # periodic at the seam
    clm = pysh.SHGrid.from_array(vort, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l)
    fac[1:] = -(A**2) / (l[1:] * (l[1:] + 1))
    psi_clm = clm.copy()
    psi_clm.coeffs *= fac[None, :, None]
    g = psi_clm.expand(grid="DH2")
    return g.data, np.array(g.lats()), np.array(g.lons())


def _seam_filled(c: xr.Dataset) -> xr.Dataset:
    """waf_clim_coeffs*.nc built before 2026-10-12 hold NaN U, V coefficients in the two DH2 columns past the 1.5 deg
    ERA5 input's last longitude (358.6 and 360 E): build_waf_clim.py interpolated without a longitude wrap, so the
    basic state -- and with it the flux and its divergence -- had a NaN strip along the prime meridian. Fill them
    periodically: linear in longitude across the seam on the open circle (as the builder's own interpolation), the
    360 column then copying the 0 column it duplicates. A no-op for a clim rebuilt with the wrapped builder."""
    c = c.load()
    lon = c.longitude.values
    closed = np.isclose(lon[-1] - lon[0], 360.0)
    lc = lon[:-1] if closed else lon
    for v in ("U", "V"):
        a = c[v].values
        core = (a[..., :-1] if closed else a).copy()
        flat = core.reshape(-1, lc.size)
        for r in np.flatnonzero(np.isnan(flat).any(1) & np.isfinite(flat).any(1)):
            ok = np.isfinite(flat[r])
            flat[r, ~ok] = np.interp(lc[~ok], lc[ok], flat[r, ok], period=360.0)
        core = flat.reshape(core.shape)
        c[v] = (c[v].dims, np.concatenate([core, core[..., :1]], axis=-1) if closed else core)
    return c


def eval_clim(coefs: np.ndarray, doy: float) -> np.ndarray:
    """Harmonic climatology at common-year day `doy` (doycal.noleap_doy, as in build_waf_clim.py)."""
    w = 2 * np.pi * doy / 365.25
    b = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.tensordot(b, coefs, axes=(0, 0))


def render(
    psi_a, wx, wy, divw, Uc, Vc, lat, lon, title: str, sub: str, out: Path, vlim: float, agree=None, spd_fc=None
):
    """One row per hemisphere, 20-80 deg, sharing a colour bar.

    A single global map spends a fifth of its height on the tropical band the
    diagnostic masks by construction (no Rossby waveguide there) and more on
    the polar caps the flux never reaches, which leaves the waveguides small.
    Cropping to the two mid-latitude belts is the same data at roughly 1.6x the
    scale, and stacking them keeps a single figure.
    """
    import textwrap

    proj = ccrs.PlateCarree(central_longitude=180)
    pc = ccrs.PlateCarree()
    # Size the figure to the maps. Each strip is 360 x 60 degrees (6:1); a fixed
    # figure size plus a long one-line footnote under bbox_inches="tight" leaves
    # the maps in part of a wide image that the page scales down until they are
    # unreadable.
    W = 14.0
    left, right, cbw = 0.045, 0.905, 0.012
    map_w = (right - left) * W
    strip_h = map_w / 6.0
    top_in, gap_in, bot_in = 0.62, 0.30, 0.95
    H = top_in + 2 * strip_h + gap_in + bot_in
    fig = plt.figure(figsize=(W, H))
    bottoms = [bot_in + strip_h + gap_in, bot_in]
    # whole-number colour steps: vlim rounded up to a multiple of 10, ticks every 10
    vtop = float(np.ceil(max(vlim, 10.0) / 10.0) * 10.0)
    levels = np.linspace(-vtop, vtop, 21)
    axes = []
    for row, ((la0, la1), name) in enumerate(
        (((20, 80), "Northern Hemisphere, 20–80°N"), ((-80, -20), "Southern Hemisphere, 20–80°S"))
    ):
        ax = fig.add_axes([left, bottoms[row] / H, right - left, strip_h / H], projection=proj)
        # a full 360 on a dateline-centred map is set in the projection's own frame:
        # cartopy 0.26 collapses the PlateCarree(-180, 180) form to a 0.2-degree strip
        ax.set_extent([-180, 180, la0, la1], crs=proj)
        cf = ax.contourf(lon, lat, -divw * 1e6, levels=levels, cmap="RdBu_r", extend="both", transform=pc)
        if agree is not None:  # hatch where members DISAGREE on the sign of ∇·W
            ax.contourf(
                lon,
                lat,
                np.where(np.isfinite(agree), agree, 1.0),
                levels=[-0.01, 0.6],
                colors="none",
                hatches=["//"],
                transform=pc,
                zorder=2,
            )
        if spd_fc is not None:
            # the REAL waveguide: the forecast's own 200 hPa jet at this lead
            ax.contour(
                spd_fc[2],
                spd_fc[1],
                spd_fc[0],
                levels=[25, 35, 45],
                colors="#1b5e20",
                linewidths=[0.8, 1.1, 1.5],
                alpha=0.85,
                transform=pc,
            )
        # Arrows: the packets are the top quarter of the flux magnitude
        s = max(1, lat.size // 30)
        Wm = np.hypot(wx, wy)
        show = Wm > np.nanpercentile(Wm, 75)
        qx = np.where(show, wx, np.nan)[::s, ::s]
        qy = np.where(show, wy, np.nan)[::s, ::s]
        q = ax.quiver(
            lon[::s],
            lat[::s],
            qx,
            qy,
            transform=pc,
            color="#111",
            width=0.0016,
            scale=2200,
            headwidth=3.6,
            alpha=0.85,
            pivot="tail",
            zorder=6,
        )
        ax.coastlines(resolution="50m", lw=0.8, color="#1a1a1a", zorder=5)
        gl = ax.gridlines(
            crs=pc,
            draw_labels=True,
            linewidth=0.3,
            color="#8a8a8a",
            alpha=0.5,
            xlocs=range(-180, 181, 60),
            ylocs=[20, 40, 60, 80] if la0 > 0 else [-80, -60, -40, -20],
        )
        gl.top_labels = gl.right_labels = False
        gl.bottom_labels = row == 1
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": "#333"}
        ax.text(
            0.006,
            0.965,
            name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#111",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )
        axes.append((ax, cf, q))
    fig.text(left, 1 - 0.20 / H, title, ha="left", va="top", fontsize=12.5, fontweight="bold")
    cax = fig.add_axes([right + 0.012, bot_in / H, cbw, (2 * strip_h + gap_in) / H])
    cb = fig.colorbar(axes[0][1], cax=cax, ticks=np.arange(-vtop, vtop + 0.1, 10))
    cb.set_label("−∇·W  (10⁻⁶ m s⁻²; red = convergence → amplification)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    axes[1][0].quiverkey(axes[1][2], 0.93, -0.20, 100, "W = 100 m²/s²", labelpos="E", fontproperties={"size": 8})
    # the footnote wraps to the figure width, so it can never widen the image
    fig.text(
        left,
        0.10 / H,
        textwrap.fill(sub, width=205),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#6f6b64",
        linespacing=1.35,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=118, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/waf")
    ap.add_argument("--manifest", default="assets/sst/anim/waf_manifest.json")
    ap.add_argument("--out", default="assets/sst/waf.webp")
    args = ap.parse_args()
    if not CLIM.exists():
        print(f"  clim {CLIM} missing — run build_waf_clim.py first; skipping.", file=sys.stderr)
        return 0
    c = _seam_filled(xr.open_dataset(CLIM))
    clat, clon = c.latitude.values, c.longitude.values

    import download_aifs

    all_steps = list(download_aifs.rmm_steps(args.time))
    frame_steps = tuple(all_steps[:16])
    cyc = ecmwf.Cycle(args.date, args.time)
    up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", (LEVEL,), frame_steps, NMEMBERS))
    vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "v", "pl", (LEVEL,), frame_steps, NMEMBERS))
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    U = xr.open_dataset(up, **ku)["u"]
    Vv = xr.open_dataset(vp, **ku)["v"]
    if "isobaricInhPa" in U.dims:
        U = U.sel(isobaricInhPa=LEVEL)
        Vv = Vv.sel(isobaricInhPa=LEVEL)
    U = U.squeeze(drop=True)
    Vv = Vv.squeeze(drop=True)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    steps_h = (U.step / np.timedelta64(1, "h")).round().astype(int).values
    members = list(U.number.values)
    nstep = len(steps_h)

    # low-passed analysis basic state: archive this cycle's 0-h control, then the 30-day mean anomaly
    anom, n_an = None, 0
    try:
        ucp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", (LEVEL,), (0,)))
        vcp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", (LEVEL,), (0,)))
        uc0 = xr.open_dataset(ucp, **ku)["u"].squeeze(drop=True)
        vc0 = xr.open_dataset(vcp, **ku)["v"].squeeze(drop=True)
        psi0, p0lat, p0lon = streamfunction_psi(uc0, vc0)
        hist = wb.update_history(
            Path(args.anim_dir) / wb.HIST_NAME,
            wb.record(
                wb.to3(uc0.values, uc0.latitude.values, uc0.longitude.values),
                wb.to3(vc0.values, vc0.latitude.values, vc0.longitude.values),
                wb.to3(psi0, p0lat, p0lon),
                init,
                LEVEL,
            ),
        )
        anom, n_an = wb.lowpass_anomaly(hist, eval_clim, c, clat, clon, init)
        print(
            f"  basic state: {'clim + 30-d analysis anomaly' if anom else 'climatology only'} ({n_an} analyses in the window)",
            flush=True,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"  basic-state history unavailable ({str(ex)[:80]}); climatology only", flush=True)
    psi_ref = (
        (lambda doy: eval_clim(c["psi"].values, doy) + anom["psi"])
        if anom
        else (lambda doy: eval_clim(c["psi"].values, doy))
    )
    U_ref = (
        (lambda doy: eval_clim(c["U"].values, doy) + anom["u"]) if anom else (lambda doy: eval_clim(c["U"].values, doy))
    )
    V_ref = (
        (lambda doy: eval_clim(c["V"].values, doy) + anom["v"]) if anom else (lambda doy: eval_clim(c["V"].values, doy))
    )

    # ψ′ per member and step (the expensive part: nmember × nstep inversions, ~0.1 s each)
    psi_a = np.zeros((len(members), nstep, len(clat), len(clon)), dtype="float32")
    for m, num in enumerate(members):
        for i in range(nstep):
            valid = init + pd.Timedelta(hours=int(steps_h[i]))
            psi, plat, plon = streamfunction_psi(U.sel(number=num).isel(step=i), Vv.sel(number=num).isel(step=i))
            if m == 0 and i == 0:
                assert np.allclose(plat, clat) and np.allclose(plon, clon), (
                    "clim grid != live DH2 grid — rebuild waf_clim with the same LMAX"
                )
            psi_a[m, i] = psi - psi_ref(float(noleap_doy(valid)))
    # quasi-stationary: running mean of ψ′ along the lead (TSMOOTH days, shrinking at the ends)
    half = TSMOOTH // 2
    psi_s = np.empty_like(psi_a)
    for i in range(nstep):
        psi_s[:, i] = psi_a[:, max(0, i - half) : min(nstep, i + half + 1)].mean(axis=1)
    fields = []
    for i, sh in enumerate(steps_h):
        valid = init + pd.Timedelta(hours=int(sh))
        doy = float(noleap_doy(valid))
        Uc, Vc = U_ref(doy), V_ref(doy)
        WX, WY, DV = [], [], []
        for m in range(len(members)):
            r = tn01_flux(
                psi_s[m, i].astype("float64"),
                Uc,
                Vc,
                clat,
                clon,
                p_hpa=LEVEL,
                umin=UMIN,
                lat_min=LATMIN,
                div_lmax=LFILT,
                div_method="zero",
            )
            WX.append(r.wx)
            WY.append(r.wy)
            DV.append(r.div)
        WX, WY, DV = np.stack(WX), np.stack(WY), np.stack(DV)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN masked cells
            wx, wy, divw = np.nanmean(WX, 0), np.nanmean(WY, 0), np.nanmean(DV, 0)
        sign = np.sign(DV)
        agree = np.nanmax(np.stack([(sign > 0).mean(0), (sign < 0).mean(0)]), 0)
        agree = np.where(np.isfinite(divw), agree, np.nan)
        um = U.isel(step=i).mean("number")
        vm = Vv.isel(step=i).mean("number")
        spd_fc = (np.hypot(um.values, vm.values), um.latitude.values, um.longitude.values)
        fields.append((valid, sh, psi_s[:, i].mean(0), wx, wy, divw, Uc, Vc, spd_fc, agree))
    vlim = float(np.nanpercentile(np.abs(np.stack([f[5] for f in fields])) * 1e6, 99.0)) or 5.0

    anim = Path(args.anim_dir)
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    basic = (
        f"ERA5 clim + 30-day mean anomaly of the AIFS 0-h analyses ({n_an} analyses)"
        if anom
        else "ERA5 climatology (analysis history too short)"
    )
    sub = (
        f"arrows = Takaya–Nakamura (2001) wave-activity flux, mean of {len(members)} members' fluxes · basic state and ψ′ reference = {basic} "
        f"({TSMOOTH}-day running mean of ψ′ along the lead) · shading = −∇·W at planetary scale (T15; red ⇒ downstream amplification), "
        "hatched where fewer than 60% of members agree on the sign · "  # (no "\n" here: the wrap folds it and the text would read "the sign green = …")
        f"green = the ensemble-mean {LEVEL} hPa jet at this lead (25/35/45 m/s) — the waveguide the packets follow · "
        "masked equatorward of 20° / basic-state wind < 3 m/s · each row is one hemisphere's 20–80° belt"
    )
    frames = []
    for i, (valid, sh, pa, wx, wy, divw, Uc, Vc, spd_fc, agree) in enumerate(fields):
        fp = anim / f"F{i:02d}.webp"
        lead = int(np.rint(sh / 24))
        render(
            pa,
            wx,
            wy,
            divw,
            Uc,
            Vc,
            clat,
            clon,
            f"Rossby wave-activity flux (TN01) {LEVEL} hPa — AIFS-ENS, member-mean flux · "
            f"init {init:%Y-%m-%d %HZ} · day {lead} (valid {valid:%a %b %d})",
            sub,
            fp,
            vlim,
            agree=agree,
            spd_fc=spd_fc,
        )
        frames.append({"idx": i, "file": fp.name, "date": f"{valid:%Y-%m-%d}", "label": f"day {lead} · {valid:%b %d}"})
    mani = {
        "ver": int(pd.Timestamp.now().timestamp()),
        "days": len(frames),
        "regions": {
            "waf": {"label": f"Wave-activity flux (TN01, {LEVEL} hPa)", "n_frames": len(frames), "frames": frames}
        },
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    # static latest = the analysis frame
    valid, sh, pa, wx, wy, divw, Uc, Vc, spd_fc, agree = fields[0]
    render(
        pa,
        wx,
        wy,
        divw,
        Uc,
        Vc,
        clat,
        clon,
        f"Rossby wave-activity flux (TN01) {LEVEL} hPa — analysis {init:%Y-%m-%d %HZ}",
        sub,
        Path(args.out),
        vlim,
        agree=agree,
        spd_fc=spd_fc,
    )
    print(f"  wrote {len(frames)} frames + manifest; conv vlim ±{vlim:.0f}×10⁻⁶ m/s²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
