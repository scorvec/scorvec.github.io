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
import argparse, sys
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


def plot(aam, p_hpa, lat, steps_h, ubar_zm, lead_h: int, init: pd.Timestamp, out: Path):
    k = int(np.argmin(np.abs(steps_h - lead_h)))
    dA = aam[k] - aam[0]                                        # (lev,lat) AAM change [kg m² s⁻¹]
    dlat_deg = abs(float(lat[1] - lat[0]))
    layer = np.abs(np.gradient(p_hpa))                         # ~layer thickness per level [hPa]

    # exact marginals & global total (these integrate the per-band AAM, not a density)
    perlat = dA.sum(axis=0) / TOT                              # ΔAAM in each lat band
    perlev = dA.sum(axis=1) / TOT                              # ΔAAM at each level
    total = dA.sum() / TOT
    print(f"  abs global AAM (analysis) = {aam[0].sum()/TOT:.0f}×10²⁴  (expect ~150)")

    # display density per (°lat·hPa) so colours are resolution/level-thickness independent
    dens = dA / (dlat_deg * layer[:, None]) / DENS
    lim = float(np.nanpercentile(np.abs(dens), 99)) or 1.0
    levels = np.linspace(-lim, lim, 21)

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  wspace=0.04, hspace=0.04, left=0.09, right=0.97, top=0.88, bottom=0.12)
    axm = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)

    cf = axm.contourf(lat, p_hpa, dens, levels=levels, cmap="RdBu_r", extend="both")
    cs = axm.contour(lat, p_hpa, ubar_zm[k], levels=np.arange(-60, 61, 10),
                     colors="k", linewidths=0.5)
    axm.clabel(cs, levels=cs.levels[::2], fmt="%d", fontsize=6)
    axm.contour(lat, p_hpa, ubar_zm[k], levels=[0], colors="k", linewidths=1.1)
    axm.set_yscale("log"); axm.invert_yaxis()
    axm.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50])
    axm.set_yticklabels([1000, 850, 700, 500, 300, 200, 100, 50])
    axm.set_xlim(-90, 90); axm.set_xticks(np.arange(-90, 91, 30))
    axm.set_xlabel("Latitude"); axm.set_ylabel("Pressure (hPa)")

    axt.bar(lat, perlat, width=dlat_deg, color=np.where(perlat >= 0, "#c0392b", "#2c5aa0"))
    axt.axhline(0, color="0.5", lw=0.6); axt.tick_params(labelbottom=False, labelsize=7)
    axt.set_ylabel("ΔAAM per lat\n(10²⁴ kg m² s⁻¹)", fontsize=7)
    axt.margins(x=0)
    axr.barh(p_hpa, perlev, height=layer, color=np.where(perlev >= 0, "#c0392b", "#2c5aa0"))
    axr.axvline(0, color="0.5", lw=0.6); axr.tick_params(labelleft=False, labelsize=7)
    axr.set_xlabel("ΔAAM per level\n(10²⁴)", fontsize=7)

    cax = fig.add_axes([0.09, 0.045, 0.4, 0.013])
    fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both",
                 label="Δ AAM density  (10²⁰ kg m² s⁻¹ per °lat · hPa)").ax.tick_params(labelsize=6)
    valid = init + pd.Timedelta(hours=int(steps_h[k]))
    sign = "+" if total >= 0 else "−"
    fig.suptitle(f"AIFS-ENS relative AAM — where the forecast change happens\n"
                 f"init {init:%Y-%m-%d %HZ}  ·  Day {steps_h[k]//24} (valid {valid:%a %d %b})  ·  "
                 f"global ΔAAM = {sign}{abs(total):.1f}×10²⁴ kg m² s⁻¹ vs analysis",
                 fontsize=11, fontweight="bold")
    axt.text(0.01, 0.80, "red = AAM gained · blue = lost   (black = zonal-mean u, m s⁻¹)",
             transform=axt.transAxes, fontsize=6.5, color="0.4")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"wrote {out}  (Day {steps_h[k]//24}, global ΔAAM {total:+.1f}×10²⁴)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/aam")
    ap.add_argument("--lead", type=int, default=-1, help="forecast hour (default: max |ΔAAM|)")
    ap.add_argument("--out", default="../../assets/sst/aam_zonal.webp")
    a = ap.parse_args()
    dd = Path(a.data_dir)
    up = dd / f"u_{a.date}_{a.time}z_pf.grib2"; sp = dd / f"sp_{a.date}_{a.time}z_pf.grib2"
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    print("== AAM zonal anatomy (AIFS-ENS ens-mean) ==", flush=True)
    m, p_hpa, lat, steps_h, ubar_zm = zonal_aam_density(up, sp)
    if a.lead < 0:                                              # pick the biggest swing
        dlat = abs(float(lat[1] - lat[0]))
        tot = [np.nansum((m[k] - m[0]) * np.abs(np.gradient(p_hpa))[:, None] * dlat)
               for k in range(len(steps_h))]
        a.lead = int(steps_h[int(np.argmax(np.abs(tot)))])
    plot(m, p_hpa, lat, steps_h, ubar_zm, a.lead, init, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
