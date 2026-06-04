#!/usr/bin/env python3
"""Anatomy of the AIFS-ENS angular-momentum forecast: WHERE (latitude) and at WHAT
LEVEL the relative AAM is changing over the forecast.

The global relative AAM is  M = (a³/g) ∫∫∫ u·cos²φ dλ dφ dp.  Keep the integrand
zonally-summed but resolved in (latitude, pressure) and you get the AAM *density*
m(φ,p) whose full integral is the global AAM. The forecast CHANGE Δm(φ,p) = m(lead) −
m(analysis) shows exactly where/at what level the wind field is adding or removing
angular momentum — the spatial breakdown of the global-AAM tendency the timeseries
shows. Ensemble mean of AIFS-ENS (the 13-level u already pulled by aam.py).

Cross-section + marginals: the main panel is Δm(φ,p); the top marginal is its vertical
integral (ΔAAM per latitude), the right marginal its meridional integral (ΔAAM per
level); the corner prints the global ΔAAM (which matches the AAM timeseries).

    python src/aam_zonal.py --date 20260603 --time 00 --data-dir data/aam \
        --out ../../assets/sst/aam_zonal.webp
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from aam import A, G, _vert_weights

DENS = 1e20          # plot density unit: 10²⁰ kg m² s⁻¹ per (° lat · hPa)
TOT = 1e24           # global/marginal totals unit: 10²⁴ kg m² s⁻¹


def zonal_aam_density(up: Path, sp_path: Path):
    """Ensemble-mean AAM per (step, lev, lat) zonal band [kg m² s⁻¹], computed exactly
    as aam.py so Σ_lev,lat = the global AAM. Also returns zonal-mean u for the jet
    contour. (a³/g)·cos²φ·dλ·dφ · Σ_lon(u·Δp), dλ/dφ in radians, Δp surface-clipped."""
    du = xr.open_dataset(up, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
    dsp = xr.open_dataset(sp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
    u = du["u"].sortby("isobaricInhPa")
    p_hpa = u.isobaricInhPa.values.astype(float)
    p_pa = p_hpa * 100.0
    lat = u.latitude.values
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))             # RADIANS (the integral needs it)
    steps_h = (u.step / np.timedelta64(1, "h")).values.astype(int)
    cos2 = np.cos(np.deg2rad(lat)) ** 2

    ubar = u.mean("number").transpose("step", "isobaricInhPa", "latitude", "longitude").values
    spv = dsp[[v for v in dsp.data_vars][0]]
    spbar = spv.mean("number").transpose("step", "latitude", "longitude").values
    ubar_zm = ubar.mean(axis=3)                                  # (step,lev,lat) jet context

    nstep, nlev, nlat = ubar.shape[0], len(p_hpa), len(lat)
    aam = np.empty((nstep, nlev, nlat))                         # AAM per (lev,lat) band [kg m² s⁻¹]
    pre = (A ** 3 / G) * dlon * dlat
    for k in range(nstep):
        dp = _vert_weights(p_pa, spbar[k])                     # (lev,lat,lon) Pa, surface-clipped
        aam[k] = pre * cos2[None, :] * (ubar[k] * dp).sum(axis=2)   # Σ_lon u·Δp → per band
    return aam, p_hpa, lat, steps_h, ubar_zm


def _xsec(ax, lat, p_hpa, dens, ubar, lim, label):
    """One latitude–pressure cross-section: filled density + zonal-mean-u contours."""
    levels = np.linspace(-lim, lim, 21)
    cf = ax.contourf(lat, p_hpa, dens, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(lat, p_hpa, ubar, levels=np.arange(-60, 61, 10), colors="k", linewidths=0.45)
    ax.clabel(cs, levels=cs.levels[::2], fmt="%d", fontsize=5.5)
    ax.contour(lat, p_hpa, ubar, levels=[0], colors="k", linewidths=1.0)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50])
    ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100, 50], fontsize=7)
    ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_ylabel("Pressure (hPa)", fontsize=8)
    ax.text(0.012, 0.91, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=2))
    return cf


def _frame(aam, p_hpa, lat, ubar_zm, k, init, steps_h, lims, out_fp):
    """Render one lead's frame (fixed figsize + colour limits, no tight bbox so every
    frame is identical-size for the slider)."""
    dlat_deg = abs(float(lat[1] - lat[0]))
    layer = np.abs(np.gradient(p_hpa))
    cell = dlat_deg * layer[:, None]
    lim_a, lim_c, lim_pl = lims
    absd = aam[k] / cell / DENS
    dA = aam[k] - aam[0]
    chgd = dA / cell / DENS
    perlat = dA.sum(axis=0) / TOT
    total = dA.sum() / TOT
    glob = aam[k].sum() / TOT

    fig = plt.figure(figsize=(9.4, 9.0))
    gs = GridSpec(3, 1, height_ratios=[0.8, 3, 3], hspace=0.17,
                  left=0.10, right=0.87, top=0.905, bottom=0.065)
    axt = fig.add_subplot(gs[0]); axa = fig.add_subplot(gs[1], sharex=axt)
    axc = fig.add_subplot(gs[2], sharex=axt)

    axt.bar(lat, perlat, width=dlat_deg, color=np.where(perlat >= 0, "#c0392b", "#2c5aa0"))
    axt.axhline(0, color="0.5", lw=0.6); axt.tick_params(labelbottom=False, labelsize=7)
    axt.set_ylim(-lim_pl, lim_pl); axt.margins(x=0)
    axt.set_ylabel("ΔAAM per °lat\n(10²⁴)", fontsize=7)
    axt.set_title("vertical integral of the change below — where the global ΔAAM comes from",
                  fontsize=7.5, color="0.4", pad=3)

    cfa = _xsec(axa, lat, p_hpa, absd, ubar_zm[k], lim_a, "ABSOLUTE  (forecast)")
    cfc = _xsec(axc, lat, p_hpa, chgd, ubar_zm[k], lim_c, "CHANGE  vs analysis (tendency)")
    axc.set_xlabel("Latitude", fontsize=8); axc.tick_params(labelsize=7); axa.tick_params(labelbottom=False)
    fig.colorbar(cfa, cax=fig.add_axes([0.89, 0.42, 0.016, 0.20]), extend="both"
                 ).set_label("abs (10²⁰/°·hPa)", fontsize=7)
    fig.colorbar(cfc, cax=fig.add_axes([0.89, 0.10, 0.016, 0.20]), extend="both"
                 ).set_label("Δ (10²⁰/°·hPa)", fontsize=7)

    valid = init + pd.Timedelta(hours=int(steps_h[k]))
    sgn = "+" if total >= 0 else "−"
    fig.suptitle(f"AIFS-ENS relative angular momentum — absolute vs forecast change\n"
                 f"init {init:%Y-%m-%d %HZ}  ·  Day {steps_h[k]//24} (valid {valid:%a %d %b})  ·  "
                 f"global AAM {glob:.0f} ·  ΔAAM = {sgn}{abs(total):.1f}  (10²⁴ kg m² s⁻¹)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_fp, dpi=120); plt.close(fig)
    return total


def animate(aam, p_hpa, lat, steps_h, ubar_zm, init, anim_dir: Path, manifest: Path):
    """Render every lead as a frame (fixed colour scale) + a viewer manifest."""
    dlat_deg = abs(float(lat[1] - lat[0]))
    layer = np.abs(np.gradient(p_hpa)); cell = dlat_deg * layer[:, None]
    print(f"  abs global AAM (analysis) = {aam[0].sum()/TOT:.0f}×10²⁴  (expect ~150)")
    # FIXED limits across all leads so the change visibly grows frame to frame
    absd_all = aam / cell[None] / DENS
    chgd_all = (aam - aam[0]) / cell[None] / DENS
    perlat_all = (aam - aam[0]).sum(axis=1) / TOT
    lims = (float(np.nanpercentile(np.abs(absd_all), 99.5)) or 1.0,
            float(np.nanpercentile(np.abs(chgd_all), 99.5)) or 1.0,
            float(np.nanpercentile(np.abs(perlat_all), 100)) * 1.05 or 1.0)

    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for k in range(len(steps_h)):
        fp = anim_dir / f"F{k:02d}.webp"
        total = _frame(aam, p_hpa, lat, ubar_zm, k, init, steps_h, lims, fp)
        valid = init + pd.Timedelta(hours=int(steps_h[k]))
        frames.append({"idx": k, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"Day {steps_h[k]//24} · ΔAAM {total:+.1f}"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"aam_zonal": {"label": "AAM anatomy — absolute & forecast change",
                                      "n_frames": len(frames), "frames": frames}}}
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(mani))
    print(f"wrote {len(frames)} frames + {Path(manifest).name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/aam")
    ap.add_argument("--anim-dir", default="../../assets/sst/anim/aam_zonal")
    ap.add_argument("--manifest", default="../../assets/sst/anim/aam_zonal_manifest.json")
    a = ap.parse_args()
    dd = Path(a.data_dir)
    up = dd / f"u_{a.date}_{a.time}z_pf.grib2"; sp = dd / f"sp_{a.date}_{a.time}z_pf.grib2"
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    print("== AAM zonal anatomy (AIFS-ENS ens-mean) ==", flush=True)
    m, p_hpa, lat, steps_h, ubar_zm = zonal_aam_density(up, sp)
    animate(m, p_hpa, lat, steps_h, ubar_zm, init, Path(a.anim_dir), Path(a.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
