#!/usr/bin/env python3
"""AAM surface-torque budget products (forecast days), AIFS-ENS ENSEMBLE MEAN.

Two outputs:
  1. Frame-based applet: per-day 2-panel maps (friction + mountain torque-density
     ANOMALY vs the forecast-period mean, ~5°, ±60° cosφ-weighted zonal inset,
     colourbars, smoothed ensemble-mean MSLP contours + H/L centres).
  2. Budget time-series: Global | NH | SH, absolute (top) + anomaly (bottom) rows;
     friction + mountain + implied GWD residual + observed d(AAM)/dt.

Ensemble mean: friction τ_λ=ρC_d|V|u is NONLINEAR, so we average the per-member
friction (dask streams member-by-member); mountain (∝p_s) and MSLP are linear, so
their ensemble mean is just the member mean. We REUSE the perturbed members the
other products already download — sp (AAM, --sp-dir), 10u (Hovmöller, --u10-dir),
msl (SOI, --msl-dir) — and download only 10v ourselves. The observed d(AAM)/dt is
the ensemble-mean AAM tendency read from aam.py's forecast archive (AAM ∝ u, linear).

    python src/torque_map_anim.py --date 20260602 --time 00 \
        --anim-dir assets/sst/anim/torque --manifest assets/sst/anim/torque_manifest.json \
        --ts-out assets/sst/torque_timeseries.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve, retrieve_parallel

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
AAM_ARCHIVE = REF / "aam_forecast_archive.nc"
DAILY_STEPS = list(range(24, 361, 24))          # days 1..15 (matches reused 10u/msl)
COARSEN = 20                                    # ~5°
A = 6.371e6; RHO = 1.225; CD = 1.3e-3; HU = 1e18; AAM_SCALE = 1e25   # Hadley unit; archive units


def _open_cf_pf(cf_path: Path, pf_path: Path) -> xr.DataArray:
    """cf+pf GRIB -> (number, step, lat, lon) on a 0..360 ascending-lat grid,
    members renumbered 0..N, steps restricted to DAILY_STEPS. Lazy (chunked)."""
    parts = []
    for p in (cf_path, pf_path):
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
        da = ds[[v for v in ds.data_vars][0]]
        if "number" not in da.dims:
            da = da.expand_dims("number")
        parts.append(da)
    da = xr.concat(parts, dim="number")
    da = da.assign_coords(number=np.arange(da.sizes["number"])).sortby("latitude")
    if float(da.longitude.min()) < 0:
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    da = da.isel(step=np.isin(hrs, DAILY_STEPS))
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return da.assign_coords(day=("step", hrs // 24))


def _dl(dd: Path, date: str, time: str, param: str, typ: str) -> Path:
    p = dd / f"{param}_aifs_{date}_{time}z_{typ}.grib2"
    if not p.exists():
        print(f"  downloading {param} ({typ}, daily steps) …", flush=True)
        req = dict(model="aifs-ens", date=date, time=int(time), stream="enfo",
                   type=typ, levtype="sfc", param=param, step=DAILY_STEPS)
        (retrieve_parallel if typ == "pf" else _retrieve)(req, str(p))
    return p


def _aam_dmdt(date: str, time: str):
    """Ensemble-mean d(AAM)/dt (Hadley) per lead day from aam.py's forecast archive
    (fc_mean, ×10²⁵ kg m²/s). Returns {G,NH,SH: {lead: dmdt}} or None."""
    if not AAM_ARCHIVE.exists():
        return None
    ds = xr.open_dataset(AAM_ARCHIVE)
    init = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time}:00")
    if init not in ds.init.values:
        print(f"  (init {init} not in AAM archive; skipping observed dM/dt overlay)")
        return None
    fc = ds["fc_mean"].sel(init=init)
    lead = ds.lead.values.astype(int)
    out = {}
    for reg, key in (("global", "G"), ("nh", "NH"), ("sh", "SH")):
        M = fc.sel(region=reg).values * AAM_SCALE                  # kg m²/s
        dmdt = np.gradient(M, lead * 86400.0) / HU                 # Hadley
        out[key] = dict(zip(lead.tolist(), dmdt.tolist()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/torque")        # where we put 10v
    ap.add_argument("--sp-dir", default="data/aam")             # AAM's sp (cf+pf)
    ap.add_argument("--u10-dir", default="data/u10")            # Hovmöller's 10u (cf+pf)
    ap.add_argument("--msl-dir", default="data/msl")            # SOI's msl (cf+pf)
    ap.add_argument("--anim-dir", default="assets/sst/anim/torque")
    ap.add_argument("--manifest", default="assets/sst/anim/torque_manifest.json")
    ap.add_argument("--ts-out", default="assets/sst/torque_timeseries.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    dd = Path(args.data_dir); dd.mkdir(parents=True, exist_ok=True)
    D, T = args.date, args.time

    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    cosphi = np.cos(np.deg2rad(lat))[:, None]
    dlam = np.deg2rad(abs(lon[1] - lon[0])); dphi = np.deg2rad(abs(lat[1] - lat[0]))

    def grid(da):
        return da.reindex(latitude=lat, longitude=lon, method="nearest")

    # --- reuse cf+pf members the other products already downloaded; fetch only 10v ---
    sp = _open_cf_pf(Path(args.sp_dir) / f"sp_{D}_{T}z_cf.grib2", Path(args.sp_dir) / f"sp_{D}_{T}z_pf.grib2")
    u10 = _open_cf_pf(Path(args.u10_dir) / f"u10_aifs_{D}_{T}z_cf.grib2", Path(args.u10_dir) / f"u10_aifs_{D}_{T}z_pf.grib2")
    msl = _open_cf_pf(Path(args.msl_dir) / f"msl_aifs-ens_{D}_{T}z_cf.grib2", Path(args.msl_dir) / f"msl_aifs-ens_{D}_{T}z_pf.grib2")
    v10 = _open_cf_pf(_dl(dd, D, T, "10v", "cf"), _dl(dd, D, T, "10v", "pf"))
    print(f"  members: sp={sp.sizes['number']} 10u={u10.sizes['number']} 10v={v10.sizes['number']} msl={msl.sizes['number']}", flush=True)

    days = [int(d) for d in np.unique(sp.day.values) if 1 <= d <= 14]
    cosn = np.cos(np.deg2rad(u10.latitude.values))[:, None]
    # friction is nonlinear → per-member then mean over members (dask streams these)
    fricN = (-RHO * CD * np.hypot(u10, v10) * u10 * (A * cosn)).mean("number")
    fric = grid(fricN).sel(step=fricN.step).load()                       # (step,lat,lon) ens-mean
    spm = grid(sp.mean("number")).load(); mslm = grid(msl.mean("number")).load()
    # mountain (form-drag) torque density as h·∂p_s/∂λ — identical net to -p_s ∂h/∂λ
    # (integration by parts in λ) but NO dipole: single-signed over each range, reading
    # directly with the cross-mountain pressure gradient (high-west/low-east ⇒ blue/braking).
    lonax = spm.get_axis_num("longitude")
    dpdlam = (np.roll(spm.values, -1, axis=lonax) - np.roll(spm.values, 1, axis=lonax)) / (2 * dlam)  # periodic
    mtn = spm.copy(data=o.values[None] * dpdlam)

    def by_day(arr):
        a = arr.assign_coords(day=("step", (arr.step / np.timedelta64(1, "h")).values.astype(int) // 24))
        return a.swap_dims({"step": "day"}).sel(day=days)
    fricD = by_day(fric); mtnD = by_day(mtn); mslD = by_day(mslm)
    # ens-mean 10-m ZONAL wind (the AAM-relevant component) for the friction-panel vectors
    uwc = by_day(grid(u10.mean("number"))).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean().load()

    # --- torque totals (Hadley) over Global / each hemi ---
    dA = (A**2) * cosphi * dlam * dphi
    def total(dens, mask): return (dens.values * dA[None])[:, mask, :].sum((1, 2)) / HU
    masks = {"G": np.ones_like(lat, bool), "NH": lat > 0, "SH": lat < 0}
    rows = {nm: dict(fric=total(fricD, m), mtn=total(mtnD, m)) for nm, m in masks.items()}
    for nm in rows:
        rows[nm]["sum"] = rows[nm]["fric"] + rows[nm]["mtn"]
    obs = _aam_dmdt(D, T)

    # --- map frames (anomaly vs period mean, coarsened) ---
    fa = (fricD - fricD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    ma = (mtnD - mtnD.mean("day")).coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    clon = fa.longitude.values; clat = fa.latitude.values
    fl = float(np.nanpercentile(np.abs(fa.values), 99)) * 1.6
    ml = float(np.nanpercentile(np.abs(ma.values), 99)) * 1.6
    mm = (clat >= -60) & (clat <= 60); cosw = np.cos(np.deg2rad(clat[mm]))
    zf = float(np.nanpercentile(np.abs(np.nanmean(fa.values, axis=2)[:, mm] * cosw), 98)) or fl
    zm = float(np.nanpercentile(np.abs(np.nanmean(ma.values, axis=2)[:, mm] * cosw), 98)) or ml
    import scipy.ndimage as ndi
    mslS = np.stack([ndi.gaussian_filter(s, sigma=6, mode="wrap") for s in (mslD.values / 100.0)])
    mlevs = np.arange(940, 1052, 8)

    def inset(zi, prof, zlim):
        zi.plot(prof, clat[mm], color="#444", lw=1.3)
        zi.axvline(0, color="0.6", lw=0.7); zi.set_ylim(-60, 60)
        zi.set_yticks([-60, -30, 0, 30, 60]); zi.yaxis.tick_right(); zi.tick_params(labelsize=7)
        zi.set_xlim(-zlim, zlim)
        e = int(np.floor(np.log10(zlim))) if zlim > 0 else 0; s = 10.0 ** e
        zi.set_xticks([-zlim, 0, zlim])
        zi.set_xticklabels([f"{-zlim / s:.0f}", "0", f"{zlim / s:.0f}"], fontsize=6.5)
        zi.set_title(f"zonal-mean anom\n·cosφ (×10$^{{{e}}}$)", fontsize=7.3); zi.grid(True, alpha=0.25)

    def hl_centers(F, mode, size=64, n=9, latcap=72):
        flt = (ndi.maximum_filter if mode == "H" else ndi.minimum_filter)(F, size=size, mode="wrap")
        ys, xs = np.where(F == flt)
        order = np.argsort(F[ys, xs])[::-1] if mode == "H" else np.argsort(F[ys, xs])
        picks = []
        for i in order:
            y, x = ys[i], xs[i]
            if abs(lat[y]) > latcap:
                continue
            if all(abs(lat[y] - py) > 12 or abs(((lon[x] - px + 180) % 360) - 180) > 18 for py, px, _ in picks):
                picks.append((lat[y], lon[x], F[y, x]))
            if len(picks) >= n:
                break
        return picks

    anim = Path(args.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, d in enumerate(days):
        hl = {"H": hl_centers(mslS[k], "H"), "L": hl_centers(mslS[k], "L")}
        fig = plt.figure(figsize=(11, 9.4))
        gs = GridSpec(2, 2, width_ratios=[7, 1], wspace=0.11, hspace=0.09,
                      left=0.012, right=0.965, top=0.935, bottom=0.05)
        for row, (arr, lim, zlim, ttl) in enumerate(
                [(fa.sel(day=d).values, fl, zf, "Friction-torque density anomaly  (−ρC$_d$|V|u·a cosφ)"),
                 (ma.sel(day=d).values, ml, zm, "Mountain-torque density anomaly  (h ∂p$_s$/∂λ)")]):
            ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=180))
            pm = ax.pcolormesh(clon, clat, arr, cmap="RdBu_r", vmin=-lim, vmax=lim,
                               transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
            cax = ax.inset_axes([1.015, 0.0, 0.02, 1.0])
            ec = int(np.floor(np.log10(lim))) if lim > 0 else 0; sc = 10.0 ** ec
            cb = fig.colorbar(pm, cax=cax, ticks=[-lim, 0, lim])
            cb.ax.set_yticklabels([f"{-lim / sc:.0f}", "0", f"{lim / sc:.0f}"], fontsize=6.5)
            cb.ax.set_title(f"×10$^{{{ec}}}$", fontsize=6.5); cb.outline.set_linewidth(0.4)
            cyc, cl = add_cyclic_point(mslS[k], coord=lon)
            ax.contour(cl, lat, cyc, levels=mlevs, colors="0.1", linewidths=0.55,
                       alpha=0.75, transform=ccrs.PlateCarree())
            for sym, col in (("H", "#b2182b"), ("L", "#2166ac")):
                for plat, plon, pval in hl[sym]:
                    ax.text(plon, plat, sym, color=col, fontsize=11, fontweight="bold",
                            ha="center", va="center", transform=ccrs.PlateCarree(), clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
                    ax.text(plon, plat - 3.5, f"{pval:.0f}", color=col, fontsize=5.5,
                            ha="center", va="top", transform=ccrs.PlateCarree(), clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
            qs = 3; qln = clon[::qs]; qlt = clat[::qs]
            if row == 0:                                           # friction: zonal (u) 10-m wind only
                uq = uwc.sel(day=d).values[::qs, ::qs]
                ax.quiver(qln, qlt, uq, np.zeros_like(uq),
                          transform=ccrs.PlateCarree(), color="0.2", scale=420, width=0.0014,
                          headwidth=4, headlength=4.5, alpha=0.7, zorder=5, pivot="middle")
            else:                                                  # mountain: east/west, scaled by |torque|
                Uq = (arr / lim)[::qs, ::qs]
                ax.quiver(qln, qlt, Uq, np.zeros_like(Uq), transform=ccrs.PlateCarree(),
                          color="0.2", scale=18, width=0.0014, headwidth=4, headlength=4.5,
                          alpha=0.7, zorder=5, pivot="middle")
            ax.coastlines(resolution="110m", lw=0.4, color="0.35"); ax.set_global()
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            inset(fig.add_subplot(gs[row, 1]), np.nanmean(arr, axis=1)[mm] * cosw, zlim)
        valid = init + pd.Timedelta(days=d)
        fig.suptitle(f"AIFS-ENS (ensemble mean) · AAM surface-torque anomalies — {valid:%Y-%m-%d} (forecast day {d})",
                     fontsize=12, fontweight="bold", y=0.99)
        fig.text(0.5, 0.013,
                 "red = anomalous torque adding westerly AAM (spin-up) · blue = removing it (spin-down), vs the forecast-period mean"
                 "   ·   grey = ensemble-mean MSLP, H / L = centres (hPa)",
                 ha="center", va="bottom", fontsize=8.2, color="0.4")
        fp = anim / f"F{k:02d}.webp"
        fig.savefig(fp, dpi=104, bbox_inches="tight"); plt.close(fig)
        entries.append({"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"),
                        "label": f"day {d} · {valid:%b %d}"})
    mani = {"regions": {"torque": {"label": "Surface torque (friction + mountain)", "frames": entries}}}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries)} frames + {args.manifest}")

    # --- budget time-series: 2 rows (absolute | anomaly) × 3 cols (Global|NH|SH) ---
    valid = [init + pd.Timedelta(days=d) for d in days]
    cols = ("G", "NH", "SH")
    titles = {"G": "Global budget", "NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}
    obs_by_day = {h: {d: obs[h].get(d, np.nan) for d in days} for h in cols} if obs else {}

    def da_(v, anom): return v - np.nanmean(v) if anom else v

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), sharex=True, sharey="row")
    for r, anom in enumerate((False, True)):
        for c, hemi in enumerate(cols):
            ax = axes[r, c]
            ax.plot(valid, da_(rows[hemi]["fric"], anom), color="#2166ac", lw=1.7, label="friction torque")
            ax.plot(valid, da_(rows[hemi]["mtn"], anom), color="#b2182b", lw=1.7, label="mountain torque")
            ax.plot(valid, da_(rows[hemi]["sum"], anom), color="k", lw=2.4, label="friction + mountain")
            if obs_by_day:
                odm = np.array([obs_by_day[hemi][d] for d in days])
                if hemi == "G":
                    ax.plot(valid, da_(odm - rows["G"]["sum"], anom), color="#1b7837", lw=1.7, label="implied GWD (residual)")
                ax.plot(valid, da_(odm, anom), color="0.35", lw=2.2, ls="--", label="observed d(AAM)/dt")
            ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(titles[hemi], fontsize=11, fontweight="bold")
            if r == 0 and c == 0:
                ax.legend(fontsize=7.6, loc="best")
            for lb in ax.get_xticklabels():
                lb.set_rotation(45); lb.set_ha("right"); lb.set_fontsize(7.5)
    axes[0, 0].set_ylabel("absolute torque\n(Hadley = 10¹⁸ N m)", fontsize=9)
    axes[1, 0].set_ylabel("anomaly vs period mean\n(Hadley)", fontsize=9)
    fig.suptitle(f"AAM torque budget (ensemble mean) — AIFS-ENS init {init:%Y-%m-%d %HZ}   ·   top: absolute   ·   bottom: anomaly",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.005,
             "Ensemble mean of all 51 members (friction averaged per-member; mountain, MSLP & d(AAM)/dt are linear).   Top: absolute torques; bottom: anomalies vs the forecast-period mean.\n"
             "Only the GLOBAL budget closes: friction + mountain + implied gravity-wave/form drag (residual) = d(AAM)/dt.\n"
             "Each hemisphere's gap between net surface torque (black) and observed d(AAM)/dt (dashed) is cross-equatorial momentum transport (+ GWD) — an internal flux, not a torque.",
             ha="center", va="top", fontsize=8.5, color="0.35")
    fig.tight_layout()
    Path(args.ts_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.ts_out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.ts_out} (Global net {rows['G']['sum'][0]:+.0f}→{rows['G']['sum'][-1]:+.0f} Hadley; "
          f"obs dM/dt {'overlaid' if obs else 'n/a (run aam.py first)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
