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
CLIM = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_density_clim.nc"


def _interp_lat(field, lat_src, lat_dst):
    """Interp (lev, lat_src) → (lev, lat_dst) along latitude (np.interp needs ascending)."""
    o = np.argsort(lat_src); ls = lat_src[o]
    return np.stack([np.interp(lat_dst, ls, field[i][o]) for i in range(field.shape[0])])


def zonal_aam_density(up_rest: Path, up_rmm: Path, sp_path: Path):
    """Ensemble-mean AAM per (step, lev, lat) zonal band [kg m² s⁻¹], computed exactly
    as aam.py so Σ_lev,lat = the global AAM. Also returns zonal-mean u for the jet
    contour. (a³/g)·cos²φ·dλ·dφ · Σ_lon(u·Δp), dλ/dφ in radians, Δp surface-clipped."""
    kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
    du_rest = xr.open_dataset(up_rest, **kw)             # 11 AAM-only levels
    du_rmm = xr.open_dataset(up_rmm, **kw)               # 200/850 (shared with the RMM)
    dsp = xr.open_dataset(sp_path, engine="cfgrib",
                          backend_kwargs={"filter_by_keys": {"shortName": "sp"}, "indexpath": ""},
                          chunks={"number": 1})          # sp out of the batched surface file
    u = xr.concat([du_rest["u"], du_rmm["u"]], dim="isobaricInhPa").sortby("isobaricInhPa")
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


def _frame(fields, lat, p_hpa, ubar, lims, head, out_fp):
    """2×2 frame: absolute & tendency (left) + their anomalies vs climatology (right).
    `fields` = [absolute, abs_anomaly, tendency, tendency_anomaly] density arrays.
    Fixed figsize + colour limits (no tight bbox) so every frame is identical-size."""
    lim_a, lim_d = lims
    labels = ["ABSOLUTE  (forecast)", "ANOMALY  vs 1991–2020",
              "CHANGE  vs analysis (tendency)", "TENDENCY  ANOMALY  vs clim"]
    lim4 = [lim_a, lim_d, lim_d, lim_d]
    fig = plt.figure(figsize=(12.8, 8.6))
    gs = GridSpec(2, 2, left=0.065, right=0.9, top=0.90, bottom=0.075, wspace=0.10, hspace=0.16)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    cfs = []
    for ax, fld, lab, lm in zip(axes, fields, labels, lim4):
        cfs.append(_xsec(ax, lat, p_hpa, fld, ubar, lm, lab))
    for ax in (axes[0], axes[1]):                            # top row: no x label
        ax.tick_params(labelbottom=False)
    for ax in (axes[1], axes[3]):                            # right col: no y label
        ax.set_ylabel("")
    for ax in (axes[2], axes[3]):
        ax.set_xlabel("Latitude", fontsize=8)
    fig.colorbar(cfs[0], cax=fig.add_axes([0.915, 0.55, 0.013, 0.33]), extend="both"
                 ).set_label("abs (10²⁰/°·hPa)", fontsize=7)
    fig.colorbar(cfs[1], cax=fig.add_axes([0.915, 0.10, 0.013, 0.33]), extend="both"
                 ).set_label("departure (10²⁰/°·hPa)", fontsize=7)
    fig.suptitle(head, fontsize=11, fontweight="bold")
    fig.savefig(out_fp, dpi=110); plt.close(fig)


def animate(aam, p_hpa, lat, steps_h, ubar_zm, init, anim_dir: Path, manifest: Path):
    """Render every lead as a 2×2 frame (fixed colour scale) + viewer manifest."""
    dlat_deg = abs(float(lat[1] - lat[0]))
    layer = np.abs(np.gradient(p_hpa)); cell = dlat_deg * layer[:, None]
    print(f"  abs global AAM (analysis) = {aam[0].sum()/TOT:.0f}×10²⁴  (expect ~150)")
    densf = aam / cell[None] / DENS                          # forecast density (step,lev,lat)

    # climatological density at the init & each valid day-of-year, interp to forecast grid
    clim = xr.open_dataarray(CLIM)                           # (doy,lev,latc) per-band 1.5°
    latc = clim.latitude.values; dlatc = abs(float(latc[1] - latc[0]))
    cellc = dlatc * layer[:, None]
    doy0 = int(init.dayofyear)
    doys = [int((init + pd.Timedelta(hours=int(h))).dayofyear) for h in steps_h]

    def cden(doy):                                           # clim density on forecast lat
        cb = clim.sel(dayofyear=doy).values / cellc / DENS
        return _interp_lat(cb, latc, lat)
    c0 = cden(doy0)
    clead = np.stack([cden(d) for d in doys])                # (step,lev,lat)

    absd = densf
    anom = densf - clead                                     # absolute anomaly
    tend = densf - densf[0]                                  # change vs analysis
    tanom = tend - (clead - c0)                              # tendency anomaly (de-seasonalised)

    lim_a = float(np.nanpercentile(np.abs(absd), 99.5)) or 1.0
    lim_d = float(np.nanpercentile(np.abs(np.concatenate([anom, tend, tanom])), 99.5)) or 1.0

    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for k in range(len(steps_h)):
        fp = anim_dir / f"F{k:02d}.webp"
        glob = aam[k].sum() / TOT
        dAAM = (aam[k] - aam[0]).sum() / TOT
        aAAM = float(((densf[k] - clead[k]) * cell * DENS).sum()) / TOT   # global AAM anomaly
        valid = init + pd.Timedelta(hours=int(steps_h[k]))
        sgn = lambda x: f"{x:+.1f}"
        head = (f"AIFS-ENS relative angular momentum — absolute, tendency & anomalies\n"
                f"init {init:%Y-%m-%d %HZ} · Day {steps_h[k]//24} (valid {valid:%a %d %b}) · "
                f"AAM {glob:.0f} · anomaly {sgn(aAAM)} · ΔAAM {sgn(dAAM)}  (10²⁴ kg m² s⁻¹)")
        _frame([absd[k], anom[k], tend[k], tanom[k]], lat, p_hpa, ubar_zm[k],
               (lim_a, lim_d), head, fp)
        frames.append({"idx": k, "file": fp.name, "date": f"{valid:%Y-%m-%d}",
                       "label": f"Day {steps_h[k]//24} · anom {sgn(aAAM)} · ΔAAM {sgn(dAAM)}"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"aam_zonal": {"label": "AAM anatomy — absolute, tendency & anomalies",
                                      "n_frames": len(frames), "frames": frames}}}
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(mani))
    print(f"wrote {len(frames)} frames + {Path(manifest).name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="../../assets/sst/anim/aam_zonal")
    ap.add_argument("--manifest", default="../../assets/sst/anim/aam_zonal_manifest.json")
    a = ap.parse_args()
    # shared store: the AAM-only 11 u-levels + the RMM 200/850 file (concatenated back to
    # 13 in zonal_aam_density) + sp — fetch-once, no duplicate of 200/850.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
    import store as ecmwf
    from aam import DAILY_STEPS
    cyc = ecmwf.Cycle(a.date, a.time); steps = tuple(DAILY_STEPS)
    up_rest = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", ecmwf.LEVELS_AAM_REST, steps))
    up_rmm = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "u", "pl", ecmwf.LEVELS_RMM, steps))
    sp = ecmwf.sfc_path(cyc, "aifs-ens", "pf", "sp")   # sp out of the batched surface file
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    print("== AAM zonal anatomy (AIFS-ENS ens-mean) ==", flush=True)
    m, p_hpa, lat, steps_h, ubar_zm = zonal_aam_density(up_rest, up_rmm, sp)
    animate(m, p_hpa, lat, steps_h, ubar_zm, init, Path(a.anim_dir), Path(a.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
