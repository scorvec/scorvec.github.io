#!/usr/bin/env python3
"""Walker circulation: equatorial zonal overturning streamfunction Ψ_W(λ,p) from
the AIFS-ENS 0-h analysis — the zonal-cell analog of the Hadley-cell MMSF plot.

    Ψ_W(λ,p) = (Δy/g) ∫_0^p u_D dp′       u_D = 5°S–5°N mean DIVERGENT zonal wind
    Δy = a·Δφ_band (the meridional width of the band, m); units 10^10 kg s^-1.
    Band continuity gives [ω] ∝ −∂Ψ_W/∂λ: ascent where Ψ_W increases eastward.
    The climatological Pacific cell (warm-pool ascent ~130°E, east-Pacific
    descent) is therefore a POSITIVE Ψ_W hump over the Pacific — eastward
    divergent flow aloft, westward at the surface, clockwise as drawn (pressure
    increasing downward). El Niño weakens the hump toward zero.

The divergent wind comes from the velocity potential (∇²χ = δ solved in spherical
harmonics per pressure level — the same solver as the 200 hPa χ product), so the
rotational flow (which dwarfs u_D near the equator) is cleanly removed.

Each cycle's u_D(λ, p) is appended to a rolling history; the last N cycles render
as Ψ_W anomaly frames (vs the ERA5 1991–2020 harmonic climatology from
build_walker_clim.py) with the absolute cells as contours. A Pacific Walker cell
index (mean Ψ_W over 140°E–160°W, 300–700 hPa) headlines each frame: positive =
normal circulation; near zero or negative = collapsed/reversed (strong El Niño).

    python src/walker.py --date 20260718 --time 00 \
        --anim-dir assets/sst/anim/walker --manifest assets/sst/anim/walker_manifest.json \
        --out assets/sst/walker_anom.webp
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf
from wind200_vpot import velocity_potential, irrotational_wind

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
A = 6.371e6; G = 9.80665; SCALE = 1e10
BAND = 5.0                                     # ± latitude of the equatorial band (°)
DY = A * np.deg2rad(2 * BAND)                  # band width (m)
LMAX = 42                                      # χ truncation — u_D is planetary-scale
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / "walker_clim_coeffs.nc"           # ERA5 1991-2020 Ψ_W harmonic coeffs
HIST = REF / "walker_ud_history.nc"            # rolling per-cycle equatorial u_D(λ,p)
MAXN = 120
# Pacific Walker cell index: MEAN Ψ_W over the west-central Pacific mid-troposphere.
# Validated on ERA5 Decembers: La Niñas (1988/98/2007/2010) give +0.28…+0.41,
# super El Niños (1982/97/2015) give −0.08…−0.13 — the box mean cleanly separates
# strength from the eastward SHIFT of the cell (a max-based index conflated both).
PAC = dict(lon=(140.0, 200.0), p=(300.0, 700.0))


def download_uv(date: str, time: str):
    """0-h analysis u and v on the 13 levels (control member), via the shared store.
    The v file is the SAME spec mmsf.py pulls, so it is always a cache hit when the
    heavy-atmos cycle fetch ran; u@13lev step0 is a small one-off (~8 MB)."""
    cyc = ecmwf.Cycle(date, time)
    up = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", tuple(LEVELS), (0,)))
    vp = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", tuple(LEVELS), (0,)))
    return up, vp


def equatorial_ud(upath: Path, vpath: Path):
    """5°S–5°N mean divergent zonal wind u_D(lev, lon) from the analysis u, v.
    χ is solved per level at LMAX on a pole-free grid; the returned longitude
    axis is the χ solver's own (DH2) grid."""
    ku = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
    u = xr.open_dataset(upath, **ku)["u"].sortby("isobaricInhPa")
    v = xr.open_dataset(vpath, **ku)["v"].sortby("isobaricInhPa")
    rows, lon_out = [], None
    for lev in u.isobaricInhPa.values:
        chi, dlat, dlon = velocity_potential(u.sel(isobaricInhPa=lev),
                                             v.sel(isobaricInhPa=lev), lmax=LMAX)
        uchi, _ = irrotational_wind(chi, dlat, dlon)
        band = np.abs(dlat) <= BAND
        # cosφ-weighted band mean (≈ unweighted at ±5°, but exact costs nothing)
        w = np.cos(np.deg2rad(dlat[band]))
        rows.append((uchi[band] * w[:, None]).sum(0) / w.sum())
        lon_out = dlon
    p_pa = u.isobaricInhPa.values * 100.0
    return np.stack(rows), p_pa, lon_out       # (lev, lon), Pa, °E


def streamfunction(ud: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Ψ_W(p,lon) in 10^10 kg/s from u_D (lev,lon), p ascending (Pa). Same
    top-down trapezoid as the MMSF (Ψ=0 at the model top)."""
    psi = np.zeros_like(ud)
    for k in range(1, len(p_pa)):
        dp = p_pa[k] - p_pa[k - 1]
        psi[k] = psi[k - 1] + 0.5 * (ud[k] + ud[k - 1]) * dp
    return psi * (DY / G) / SCALE


def pac_index(psi: np.ndarray, p_hpa: np.ndarray, lon: np.ndarray) -> float:
    """Pacific Walker cell index: mean Ψ_W in the PAC box (10^10 kg/s). Positive
    in climatology; collapses toward zero — and goes NEGATIVE (reversed cell)
    during strong El Niños."""
    mp = (p_hpa >= PAC["p"][0]) & (p_hpa <= PAC["p"][1])
    ml = (lon >= PAC["lon"][0]) & (lon <= PAC["lon"][1])
    return float(np.nanmean(psi[np.ix_(mp, ml)]))


def update_history(ud, p_pa, lon, valid: datetime) -> xr.DataArray:
    cur = xr.DataArray(
        ud[None], dims=("time", "level", "longitude"),
        coords={"time": [pd.Timestamp(valid)],
                "level": (np.asarray(p_pa) / 100).astype(int), "longitude": lon},
        name="ud")
    if HIST.exists():
        old = xr.open_dataarray(HIST)
        old = old.sel(time=old.time != np.datetime64(valid))
        da = xr.concat([old, cur], dim="time").sortby("time") if old.time.size else cur
    else:
        da = cur
    da = da.isel(time=slice(-MAXN, None))
    HIST.parent.mkdir(parents=True, exist_ok=True)
    tmp = HIST.with_suffix(".tmp.nc"); da.to_netcdf(tmp); tmp.replace(HIST)
    return da


def clim_psi(coeffs: xr.DataArray, doy: float) -> np.ndarray:
    """Evaluate the mean+annual+semiannual Ψ_W climatology for a day-of-year."""
    w = 2 * np.pi * doy / 365.25
    basis = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.tensordot(basis, coeffs.values, axes=(0, 0))


def plot_psi(psi, p_hpa, lon, out: Path, title: str, vlim, psi_abs, pidx, pclim):
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    levs = np.linspace(-vlim, vlim, 21)
    cf = ax.contourf(lon, p_hpa, psi, levels=levs, cmap="RdBu_r", extend="both")
    clev = np.array([x for x in range(-30, 31, 2) if x != 0])
    cs = ax.contour(lon, p_hpa, psi_abs, levels=clev, colors="k", linewidths=0.8)
    ax.clabel(cs, levels=[x for x in clev if x % 8 == 0], fmt="%d", fontsize=6, inline=True)
    ax.contour(lon, p_hpa, psi_abs, levels=[0], colors="k", linewidths=1.0)
    # vertical-motion arrows: band continuity ⇒ [ω] ∝ −∂Ψ_W/∂λ, so ASCENT (up
    # arrow, +V in quiver) ∝ +∂Ψ_W/∂λ; normalized like the MMSF plot, weak
    # columns hidden.
    dpsidl = np.gradient(psi_abs, np.deg2rad(lon), axis=1)
    wn = dpsidl / (np.nanpercentile(np.abs(dpsidl), 96) or 1.0)
    wn[np.abs(wn) < 0.12] = np.nan
    sj = max(1, lon.size // 30)
    LONm, Pm = np.meshgrid(lon, p_hpa)
    ax.quiver(LONm[::2, ::sj], Pm[::2, ::sj], np.zeros_like(wn)[::2, ::sj], wn[::2, ::sj],
              angles="uv", scale_units="height", scale=9, width=0.0026,
              headwidth=4, headlength=5, color="0.15", alpha=0.8, pivot="mid", zorder=6)
    ax.set_ylim(1000, 100); ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100])
    ax.minorticks_off()
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60))
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_xlabel("longitude"); ax.set_ylabel("pressure (hPa)")
    for x, lab in ((120, "Maritime\nContinent"), (255, "E Pacific"), (300, "S America")):
        ax.text(x, 118, lab, ha="center", va="top", fontsize=6.5, color="0.45", style="italic")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, pad=0.015)
    cb.set_label("Ψ_W anomaly  (10¹⁰ kg s⁻¹)", fontsize=9)
    fig.text(0.5, 0.005,
             f"5°S–5°N divergent wind (χ, spherical harmonics) · colour = Ψ_W′ vs ERA5 1991–2020 · "
             f"black contours = absolute Ψ_W · arrows = vertical motion · "
             f"Pacific cell {pidx:+.1f} (clim {pclim:+.1f}) ×10¹⁰ kg/s — lower = weaker Walker",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def build_anim(hist: xr.DataArray, anim_dir: Path, manifest: Path, static_out: Path) -> int:
    if not CLIM.exists():
        print(f"  clim coeffs {CLIM} missing — run build_walker_clim.py first; skipping anim.",
              file=sys.stderr)
        return 1
    coeffs = xr.open_dataarray(CLIM)
    lon = hist.longitude.values
    coeffs = coeffs.interp(longitude=lon, kwargs={"fill_value": None})
    assert list(hist.level.values) == LEVELS, \
        f"history level axis {list(hist.level.values)} != LEVELS — stale walker_ud_history.nc?"
    p_pa = np.asarray(LEVELS, float) * 100.0
    st_all = pd.to_datetime(hist.time.values)
    ud_all = hist.values

    WIN, MINSPAN = pd.Timedelta(days=7), pd.Timedelta(days=6)
    st, ud_wk = [], []
    for i in range(len(st_all)):
        w = (st_all > st_all[i] - WIN) & (st_all <= st_all[i])
        if st_all[i] - st_all[w][0] < MINSPAN:
            continue
        st.append(st_all[i]); ud_wk.append(ud_all[w].mean(axis=0))
    if not st:
        st, ud_wk = list(st_all), [ud_all[i] for i in range(len(st_all))]
    st = pd.DatetimeIndex(st)

    psia, psip, pidx, pcl = [], [], [], []
    for i in range(len(st)):
        psi = streamfunction(ud_wk[i], p_pa)
        cl = clim_psi(coeffs, (st[i] - pd.Timedelta(days=3.5)).dayofyear)  # window midpoint
        psia.append(psi); psip.append(psi - cl)
        pidx.append(pac_index(psi, p_pa / 100, lon))
        pcl.append(pac_index(cl, p_pa / 100, lon))
    vlim = float(np.nanpercentile(np.abs(np.stack(psip)), 99.0)) or 1.0

    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for i in range(len(st)):
        fp = anim_dir / f"F{i:02d}.webp"
        plot_psi(psip[i], p_pa / 100, lon, fp,
                 f"Walker circulation anomaly Ψ_W′ (5°S–5°N) — 7-day mean ending {st[i]:%Y-%m-%d}",
                 vlim, psia[i], pidx[i], pcl[i])
        frames.append({"idx": i, "file": fp.name, "date": f"{st[i]:%Y-%m-%d}",
                       "label": f"week ending {st[i]:%a %b %d} · Pacific {pidx[i]:+.1f}"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"walker": {
                "label": "Walker circulation anomaly (7-day mean, Ψ_W′ vs ERA5 1991-2020)",
                "n_frames": len(frames), "frames": frames}}}
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(mani))
    plot_psi(psip[-1], p_pa / 100, lon, Path(static_out),
             f"Walker circulation anomaly Ψ_W′ (5°S–5°N) — 7-day mean ending {st[-1]:%Y-%m-%d}",
             vlim, psia[-1], pidx[-1], pcl[-1])
    print(f"  Ψ_W′ range {np.nanmin(psip[-1]):.1f} … {np.nanmax(psip[-1]):.1f} ×10¹⁰ kg/s; "
          f"Pacific cell {pidx[-1]:+.1f} (clim {pcl[-1]:+.1f}); wrote {len(frames)} frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/walker")
    ap.add_argument("--manifest", default="assets/sst/anim/walker_manifest.json")
    ap.add_argument("--out", default="assets/sst/walker_anom.webp")
    args = ap.parse_args()
    up, vp = download_uv(args.date, args.time)
    ud, p_pa, lon = equatorial_ud(up, vp)
    valid = datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H")
    hist = update_history(ud, p_pa, lon, valid)
    print(f"  history: {hist.time.size} cycles "
          f"({pd.to_datetime(hist.time.values[0]):%Y-%m-%d} … {valid:%Y-%m-%d %HZ})")
    return build_anim(hist, Path(args.anim_dir), Path(args.manifest), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
