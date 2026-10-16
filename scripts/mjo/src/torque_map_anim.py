#!/usr/bin/env python3
"""AAM surface-torque budget products (forecast days), IFS-ENS ENSEMBLE MEAN.

Two outputs:
  1. Frame-based applet: per-day 2-panel maps (friction + mountain torque-density
     ANOMALY vs the forecast-period mean, ~5°, ±60° cosφ-weighted zonal inset,
     colourbars, smoothed ensemble-mean MSLP contours + H/L centres).
  2. Torque ANOMALY time-series: Global | NH | SH, one row. Absolute values are
     deliberately not shown -- see the note above the figure block.

IFS-ENS, a physics model, so the torques are its own and its budget closes (aifs-aam-budget.html): the 50 perturbed
members of the 00/12 UTC run (open data, Google Cloud mirror). Every quantity is a mean over a forecast DAY, the
window from day d-1 to day d:
  friction  -a cosφ τ, τ = the window difference of the accumulated eastward turbulent surface stress ewss / 86400 s
            (the model's own stress, turbulent orographic form drag included; linear, so the member mean is exact);
  mountain  h ∂p_s/∂λ on the ERA5 orography (the climatology's) from the members' mean surface pressure, the
            trapezoid of its 12-hourly values over the window - the 00/12 UTC sampling of the ERA5 climatology, so its
            aliasing of the semidiurnal pressure tide cancels in the anomaly;
  d(AAM)/dt the window difference of the ensemble-mean AAM in aam.py's archive of the same runs.
The parameterised gravity-wave and blocking drag stresses are not in open data, so their torque sits in the gap.

    python src/torque_map_anim.py --date 20260602 --time 00 \
        --anim-dir assets/sst/anim/torque --manifest assets/sst/anim/torque_manifest.json \
        --ts-out assets/sst/torque_timeseries.webp --ranges-out assets/sst/torque_ranges.webp
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
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf  # shared ECMWF download manager
from aam import ARCHIVE_PATH as AAM_ARCHIVE, DAILY_STEPS, SP_STEPS, member_mean
from doycal import noleap_doy  # common-year day of year (29 Feb = 59.5)

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
AAM_CLIM = REF / "aam_clim.nc"  # ERA5 AAM seasonal cycle -> climatological dM/dt
MAP_CLIM = REF / "torque_map_clim_coeffs.nc"
RANGE_SD = REF / "torque_range_sd.nc"  # per-barrier 1-sigma, for scale context
HEMI_SD = REF / "torque_hemi_sd.nc"  # net-torque anomaly 1-sigma, G/NH/SH    # ERA5 per-gridcell torque-density clim
DAYS = list(range(1, 16))  # forecast days: the windows ending 24, 48 ... 360 h
COARSEN = 20  # ~5°


def _map_clim(coef_da, clat, clon, doys):
    """Evaluate the ERA5 harmonic torque-density clim on the map grid for each
    forecast day's common-year day (doycal.noleap_doy) → (day, clat, clon). The coeffs are per hour (00Z/12Z; the
    friction density has a strong land diurnal cycle); a forecast DAY's mean takes the mean of the two."""
    if "hour" in coef_da.dims:
        coef_da = coef_da.mean("hour")
    c = coef_da.reindex(latitude=clat, longitude=clon, method="nearest").values
    out = np.empty((len(doys), len(clat), len(clon)))
    for i, doy in enumerate(doys):
        w = 2 * np.pi * doy / 365.25
        b = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        out[i] = np.tensordot(b, c, axes=(0, 0))
    return out


# major orographic barriers (lon0, lon1, lat0, lat1) on 0..360°E — for the by-range breakdown
RANGES = {
    "Himalaya/Tibet": (70, 105, 25, 45),
    "Rockies/W. N.America": (232, 258, 30, 62),
    "Andes": (282, 296, -56, 12),
    "Greenland": (300, 345, 58, 84),
    "Antarctica": (0, 360, -90, -66),
    "Alps/Europe": (0, 45, 36, 50),
}
A = 6.371e6
HU = 1e18
AAM_SCALE = 1e25  # Hadley unit; archive units
DAY_S = 86400.0


def _open_ifs(path: Path, short: str | None = None) -> xr.DataArray:
    """An IFS-ENS GRIB -> (number, step, lat, lon) on a 0..360, ascending-latitude grid. Lazy, one member a chunk.
    `short` filters one field out of a batched multi-param surface file."""
    bk = {"indexpath": ""}
    if short:
        bk["filter_by_keys"] = {"shortName": short}
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=bk, chunks={"number": 1})
    da = ds[next(iter(ds.data_vars))].sortby("latitude")
    if float(da.longitude.min()) < 0:
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    return da


def _at(da: xr.DataArray, hours) -> xr.DataArray:
    """The steps at `hours` (forecast hours)."""
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return da.isel(step=[int(np.flatnonzero(hrs == h)[0]) for h in hours])


def _aam_dmdt(date: str, time: str):
    """Ensemble-mean d(AAM)/dt (Hadley) over each forecast day d (the window d-1 to d) from aam.py's archive of the
    same runs (fc_mean, ×10²⁵ kg m²/s). Returns {G,NH,SH: {d: dmdt}} or None."""
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
        M = fc.sel(region=reg).values * AAM_SCALE  # kg m²/s
        dmdt = np.diff(M) / (np.diff(lead) * DAY_S) / HU  # Hadley, the window ending at lead[1:]
        out[key] = dict(zip(lead[1:].tolist(), dmdt.tolist()))
    return out


def _clim_dmdt(valids):
    """Climatological d(AAM)/dt (Hadley) per region for each valid date, from the
    ERA5 AAM climatology. NOT negligible: the global term runs from -14 Hadley in
    June to +9 in September, so the anomaly panel has to subtract it — plotting the
    absolute tendency against ERA5-referenced torque anomalies mis-states the
    residual by that much."""
    if not AAM_CLIM.exists():
        return None
    c = xr.open_dataset(AAM_CLIM)
    doy = c.doy.values.astype(float)
    out = {}
    for reg, key in (("global", "G"), ("nh", "NH"), ("sh", "SH")):
        M = c["mean"].sel(region=reg).values * AAM_SCALE  # kg m2/s
        dmdt = np.gradient(M, doy * 86400.0) / HU  # Hadley, on the clim doy grid
        out[key] = np.interp(noleap_doy(valids), doy, dmdt, period=365.0)  # common-year days, as the clim
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/torque")
    ap.add_argument("--manifest", default="assets/sst/anim/torque_manifest.json")
    ap.add_argument("--ts-out", default="assets/sst/torque_timeseries.webp")
    ap.add_argument("--ranges-out", default="assets/sst/torque_ranges.webp")
    ap.add_argument(
        "--series-out",
        default="assets/sst/data/torque_ranges.json",
        help="ensemble-mean torque anomaly by range per forecast day (read by pacjet.py)",
    )
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    D, T = args.date, args.time

    o = xr.open_dataarray(OROG)
    lat = o.latitude.values
    lon = o.longitude.values
    dlam = np.deg2rad(abs(lon[1] - lon[0]))

    def grid(da):
        return da.reindex(latitude=lat, longitude=lon, method="nearest")

    # --- shared store: sp (12-hourly, aam.py's file), ewss (daily, accumulated) and msl (the shared IFS surface batch,
    # days 1-15) of the 50 IFS-ENS members ---
    cyc = ecmwf.Cycle(D, T)
    sp = _open_ifs(ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "sp", "sfc", (), SP_STEPS)))
    ewss = _open_ifs(ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "ewss", "sfc", (), tuple(DAILY_STEPS))))
    msl = _open_ifs(ecmwf.sfc_path(cyc, "ifs", "pf", "msl"), "msl")
    print(
        f"  IFS-ENS members: sp={sp.sizes['number']} ewss={ewss.sizes['number']} msl={msl.sizes['number']}", flush=True
    )
    days = DAYS
    end_h = [24 * d for d in days]  # each day's window ends here

    # Linear in every member field, so the means are taken first, one member at a time (aam.member_mean).
    def mean(da):
        return grid(da.isel(number=0, drop=True).copy(data=member_mean(da)))

    spm = mean(sp)
    ewm = mean(ewss)
    mslm = mean(_at(msl, end_h))

    # friction: the model's own surface stress, the window mean from the accumulated ewss (N m^-2 s)
    cosn = np.cos(np.deg2rad(lat))[:, None]
    tau = (_at(ewm, end_h).values - _at(ewm, [h - 24 for h in end_h]).values) / DAY_S
    fricD = xr.DataArray(
        -(A * cosn)[None] * tau,
        dims=("day", "latitude", "longitude"),
        coords={"day": days, "latitude": lat, "longitude": lon},
    )
    # mountain (form-drag) torque density as h·∂p_s/∂λ — identical net to -p_s ∂h/∂λ
    # (integration by parts in λ) but NO dipole: single-signed over each range, reading
    # directly with the cross-mountain pressure gradient (high-west/low-east ⇒ blue/braking).
    # Each day: the trapezoid of the 12-hourly values over its window (weights 1/4, 1/2, 1/4).
    lonax = spm.get_axis_num("longitude")
    dpdlam = (np.roll(spm.values, -1, axis=lonax) - np.roll(spm.values, 1, axis=lonax)) / (2 * dlam)  # periodic
    mtn12 = spm.copy(data=o.values[None] * dpdlam)
    mtnD = xr.concat(
        [
            0.25 * _at(mtn12, [h - 24]).squeeze("step", drop=True)
            + 0.5 * _at(mtn12, [h - 12]).squeeze("step", drop=True)
            + 0.25 * _at(mtn12, [h]).squeeze("step", drop=True)
            for h in end_h
        ],
        dim=xr.DataArray(days, dims="day", name="day"),
    )
    # the contours: the day's end
    mslD = mslm.assign_coords(day=("step", days)).swap_dims({"step": "day"})
    spmD = _at(spm, end_h).assign_coords(day=("step", days)).swap_dims({"step": "day"})

    obs = _aam_dmdt(D, T)

    # --- map frames (torque ANOMALY vs the ERA5 1991-2020 per-gridcell climatology,
    #     coarsened) — subtracting clim(day-of-year) leaves the TRUE departure from
    #     climatology (the standing El Niño signal + synoptic), not just the day-to-day
    #     wiggle within the forecast window. Falls back to the forecast-period mean if the
    #     clim coeffs are missing. Contoured against the surface-pressure anomaly (sp'). ---
    fricC = fricD.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    mtnC = mtnD.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
    clon = fricC.longitude.values
    clat = fricC.latitude.values
    if MAP_CLIM.exists():
        cc = xr.open_dataset(MAP_CLIM)
        doys = noleap_doy([init + pd.Timedelta(hours=24 * int(d) - 12) for d in days])  # each window's midpoint
        fa = fricC.copy(data=fricC.values - _map_clim(cc["friction"], clat, clon, doys))
        ma = mtnC.copy(data=mtnC.values - _map_clim(cc["mountain"], clat, clon, doys))
        base_lbl = "ERA5 1991–2020 climatology"
    else:
        print(f"  WARN: {MAP_CLIM.name} missing — anomaly vs forecast-period mean", flush=True)
        fa = fricC - fricC.mean("day")
        ma = mtnC - mtnC.mean("day")
        base_lbl = "forecast-period mean"
    fl = float(np.nanpercentile(np.abs(fa.values), 99)) * 1.2
    # The mountain torque is zero wherever there is no terrain, which is most of the
    # planet, so a percentile over the whole field is set by ocean zeros and the
    # ranges come out nearly blank. Scale on the cells that actually have relief.
    orog5 = (
        xr.DataArray(o.values, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        .coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim")
        .mean()
        .values
    )
    hasrelief = orog5 > 150.0
    _mv = np.abs(ma.values)[:, hasrelief]
    ml = float(np.nanpercentile(_mv, 93)) * 1.10 if _mv.size else float(np.nanpercentile(np.abs(ma.values), 99)) * 1.2
    mm = (clat >= -60) & (clat <= 60)
    cosw = np.cos(np.deg2rad(clat[mm]))
    zf = float(np.nanpercentile(np.abs(np.nanmean(fa.values, axis=2)[:, mm] * cosw), 98)) or fl
    zm = float(np.nanpercentile(np.abs(np.nanmean(ma.values, axis=2)[:, mm] * cosw), 98)) or ml
    import scipy.ndimage as ndi

    # STANDARDIZED pressure anomaly z = (p − clim_mean_month)/clim_std_month — equalises
    # the huge latitude dependence of pressure variability so tropical signals show, not
    # just storm-track ones (σ floored at 0.5 hPa). Prefer MSLP (continuous → no terrain
    # holes); else standardized surface pressure (masked over high orography); else raw hPa.
    MSL_CLIM = REF / "msl_clim_monthly.nc"
    SP_CLIM = REF / "sp_clim_monthly.nc"
    months = [pd.Timestamp(init + pd.Timedelta(days=int(d))).month for d in days]

    def _stdz(fieldD, clim_path):
        c = xr.open_dataset(clim_path)
        # fill_value=None → linear extrapolation at the clim grid's edges. The clim
        # lives on an offset ~1° grid, so plain interp left NaN columns at lon
        # 0/359.75 + the pole rows, which the σ=6 gaussian smear turned into a
        # ~13°-wide blank meridian band through the UK/Greenwich sector.
        mg = c["mean"].interp(latitude=lat, longitude=lon, kwargs={"fill_value": None})
        sg = c["std"].interp(latitude=lat, longitude=lon, kwargs={"fill_value": None})
        return np.stack(
            [
                (fieldD.sel(day=d).values / 100.0 - mg.sel(month=mo).values) / np.maximum(sg.sel(month=mo).values, 0.5)
                for d, mo in zip(days, months)
            ]
        )

    mask_orog = False
    if MSL_CLIM.exists():
        spaD = _stdz(mslD, MSL_CLIM)
        plevs = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        sp_unit = "σ"
        sp_lbl = "standardized MSLP anomaly"
    elif SP_CLIM.exists():
        spaD = _stdz(spmD, SP_CLIM)
        plevs = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        sp_unit = "σ"
        sp_lbl = "standardized surface-pressure anomaly"
        mask_orog = True
    else:
        spaD = (spmD - spmD.mean("day")).values / 100.0
        plevs = np.array([2, 4, 6, 8, 12, 16, 20, 24])
        sp_unit = "hPa"
        sp_lbl = "surface-pressure anomaly"
    hlfmt = "%+.1f" if sp_unit == "σ" else "%+.0f"
    spaS = np.stack([ndi.gaussian_filter(s, sigma=6, mode="wrap") for s in spaD])
    if mask_orog:  # surface pressure only
        spaS[:, ndi.binary_dilation(o.values > 1000.0, iterations=4)] = np.nan
    mask_note = "; masked over high terrain" if mask_orog else ""

    def inset(zi, prof, zlim):
        zi.plot(prof, clat[mm], color="#444", lw=1.3)
        zi.axvline(0, color="0.6", lw=0.7)
        zi.set_ylim(-60, 60)
        zi.set_yticks([-60, -30, 0, 30, 60])
        zi.yaxis.tick_right()
        zi.tick_params(labelsize=7)
        zi.set_xlim(-zlim, zlim)
        e = int(np.floor(np.log10(zlim))) if zlim > 0 else 0
        s = 10.0**e
        zi.set_xticks([-zlim, 0, zlim])
        zi.set_xticklabels([f"{-zlim / s:.0f}", "0", f"{zlim / s:.0f}"], fontsize=6.5)
        zi.set_title(f"zonal-mean anom\n·cosφ (×10$^{{{e}}}$)", fontsize=7.3)
        zi.grid(True, alpha=0.25)

    def hl_centers(F, mode, size=90, n=5, latcap=72):
        flt = (ndi.maximum_filter if mode == "H" else ndi.minimum_filter)(F, size=size, mode="wrap")
        ys, xs = np.where(F == flt)  # noqa: SIM300  variable F (maths notation), not a constant
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

    anim = Path(args.anim_dir)
    anim.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, d in enumerate(days):
        hl = {"H": hl_centers(spaS[k], "H"), "L": hl_centers(spaS[k], "L")}
        fig = plt.figure(figsize=(9.6, 9.4))
        # hspace 0.09 -> 0.2 and bottom 0.05 -> 0.075 (2026-10-08 sweep): the upper profile's tick labels ran into the
        # lower profile's title, and the footnote into the lower profile's tick labels
        gs = GridSpec(
            2, 2, width_ratios=[7, 1], wspace=0.11, hspace=0.2, left=0.012, right=0.965, top=0.935, bottom=0.075
        )
        for row, (arr, lim, zlim, ttl) in enumerate(
            [
                (
                    fa.sel(day=d).values,
                    fl,
                    zf,
                    "Friction torque density anomaly  (−τ a cosφ, the model's surface stress)",
                ),
                (
                    ma.sel(day=d).values,
                    ml,
                    zm,
                    "Mountain form-drag torque anomaly\n(h ∂p$_s$/∂λ — terrain height × east–west pressure gradient)",
                ),
            ]
        ):  # one line overran the colour bar
            ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree(central_longitude=180))
            # smooth the coarse (~5°) field and draw it as filled contours instead of
            # raw grid boxes: gaussian in longitude (periodic) + latitude, then contourf.
            sm = ndi.gaussian_filter1d(arr, 1.5, axis=1, mode="wrap")  # longitude (wrap)
            sm = ndi.gaussian_filter1d(sm, 1.1, axis=0, mode="nearest")  # latitude
            cyc, cl = add_cyclic_point(sm, coord=clon)
            pm = ax.contourf(
                cl,
                clat,
                cyc,
                levels=np.linspace(-lim, lim, 25),
                cmap="RdBu_r",
                extend="both",
                transform=ccrs.PlateCarree(),
            )
            cax = ax.inset_axes([1.015, 0.0, 0.02, 1.0])
            ec = int(np.floor(np.log10(lim))) if lim > 0 else 0
            sc = 10.0**ec
            cb = fig.colorbar(pm, cax=cax, ticks=[-lim, 0, lim])
            cb.ax.set_yticklabels([f"{-lim / sc:.0f}", "0", f"{lim / sc:.0f}"], fontsize=6.5)
            cb.ax.set_title(f"×10$^{{{ec}}}$", fontsize=6.5)
            cb.outline.set_linewidth(0.4)
            if row == 1:  # p_s contours only on the mountain panel
                cyc, cl = add_cyclic_point(spaS[k], coord=lon)
                ax.contour(
                    cl,
                    lat,
                    cyc,
                    levels=plevs,
                    colors="0.12",
                    linewidths=0.5,
                    linestyles="solid",
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )  # + anomaly
                ax.contour(
                    cl,
                    lat,
                    cyc,
                    levels=-plevs[::-1],
                    colors="0.12",
                    linewidths=0.5,
                    linestyles="dashed",
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )  # − anomaly
                # terrain outline: without it the reader cannot see which barrier a
                # blob belongs to, which is the whole point of a form-drag map
                cyc_o, cl_o = add_cyclic_point(orog5, coord=clon)
                ax.contour(
                    cl_o,
                    clat,
                    cyc_o,
                    levels=[1500],
                    colors="#4a4744",
                    linewidths=0.7,
                    alpha=0.7,
                    transform=ccrs.PlateCarree(),
                )
                for nm, (lo0, lo1, la0, la1) in RANGES.items():
                    if nm == "Antarctica":
                        continue  # a full zonal band, not a box
                    ax.plot(
                        [lo0, lo1, lo1, lo0, lo0],
                        [la0, la0, la1, la1, la0],
                        color="#8a4b2a",
                        lw=1.0,
                        alpha=0.9,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )
                    ax.text(
                        0.5 * (lo0 + lo1),
                        la1 + 2.5,
                        nm.split("/")[0],
                        color="#8a4b2a",
                        fontsize=6.2,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                        transform=ccrs.PlateCarree(),
                        zorder=7,
                        clip_on=True,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
                    )
                for sym, col in (("H", "#b2182b"), ("L", "#2166ac")):
                    for plat, plon, pval in hl[sym]:
                        ax.text(
                            plon,
                            plat,
                            sym + "′",
                            color=col,
                            fontsize=8.5,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            transform=ccrs.PlateCarree(),
                            clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
                        )
                        ax.text(
                            plon,
                            plat - 3.2,
                            hlfmt % pval,
                            color=col,
                            fontsize=6.0,
                            ha="center",
                            va="top",
                            transform=ccrs.PlateCarree(),
                            clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
                        )
            ax.coastlines(resolution="110m", lw=0.55, color="#4a4744")
            ax.set_global()
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            inset(fig.add_subplot(gs[row, 1]), np.nanmean(arr, axis=1)[mm] * cosw, zlim)
        valid = init + pd.Timedelta(days=d)
        fig.suptitle(
            f"IFS-ENS (ensemble mean) · AAM surface-torque anomalies — "
            f"init {init:%Y-%m-%d %HZ} · forecast day {d}, the 24 h to {valid:%Y-%m-%d %HZ}",
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )
        fig.text(
            0.5,
            0.013,
            f"red = anomalous torque adding westerly AAM (spin-up) · blue = removing it (spin-down), vs the {base_lbl}\n"
            f"mountain panel: grey = {sp_lbl} (solid +, dashed −{mask_note}), H′/L′ centres ({sp_unit}), "
            f"thin dark = 1500 m terrain contour, brown boxes = the barriers broken out in the by-range figure",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="0.4",
            linespacing=1.5,
        )
        fp = anim / f"F{k:02d}.webp"
        fig.savefig(fp, dpi=104, bbox_inches="tight")
        plt.close(fig)
        entries.append(
            {"idx": k, "file": fp.name, "date": valid.strftime("%Y-%m-%d"), "label": f"day {d} · {valid:%b %d}"}
        )
    mani = {
        "ver": int(pd.Timestamp.now().timestamp()),
        "regions": {"torque": {"label": "Surface torque (friction + mountain)", "frames": entries}},
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(entries)} frames + {args.manifest}")

    # --- torque ANOMALY time-series: Global | NH | SH ---
    valid = [init + pd.Timedelta(hours=24 * d - 12) for d in days]  # each day's mean, at its window's midpoint
    cols = ("G", "NH", "SH")
    titles = {"G": "Global budget", "NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}
    obs_by_day = {h: {d: obs[h].get(d, np.nan) for d in days} for h in cols} if obs else {}
    cdm = _clim_dmdt(valid)  # climatological tendency, for the anomaly row
    if cdm is None:
        print("  WARN: aam_clim.nc missing - anomaly row will use the absolute tendency")

    # ERA5-referenced anomaly totals = the spatial integral of the anomaly maps (fa, ma),
    # so the anomaly time-series matches the maps exactly (vs the ERA5 clim, not the
    # forecast-period mean). The observed tendency gets its OWN climatology subtracted on
    # that row (_clim_dmdt): the old code assumed clim dM/dt ~ 0, but it is +-15 Hadley.
    dlam5 = np.deg2rad(abs(clon[1] - clon[0]))
    dphi5 = np.deg2rad(abs(clat[1] - clat[0]))
    dA5 = (A**2) * np.cos(np.deg2rad(clat)) * dlam5 * dphi5
    cmasks = {"G": np.ones_like(clat, bool), "NH": clat > 0, "SH": clat < 0}

    def total5(dens, m):
        return (dens.values * dA5[None, :, None])[:, m, :].sum((1, 2)) / HU

    arows = {nm: {"fric": total5(fa, m), "mtn": total5(ma, m)} for nm, m in cmasks.items()}
    for nm in arows:
        arows[nm]["sum"] = arows[nm]["fric"] + arows[nm]["mtn"]

    # ANOMALY ONLY (2026-08-29). The absolute row was dropped: the absolute budget
    # does not close and cannot be made to. ERA5's own three terms sum to -8.8
    # Hadley in the annual mean against a required ~0, the resolved mountain torque
    # is not resolution-converged (+3.1 Hadley at 0.25 deg, -7.7 at 0.5, -23.7 at
    # 1.0), and 00Z/12Z sampling is 12 h apart so it cannot resolve the SEMIDIURNAL
    # pressure tide at all -- both samples sit at the same S2 phase. Showing
    # absolute numbers invited reading a level that none of that supports. The
    # anomalies are robust, because every one of those errors sits in the mean and
    # cancels in the difference.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharex=True, sharey=True)
    hsd = {}
    if HEMI_SD.exists():
        _h = xr.open_dataset(HEMI_SD)
        hsd = {str(k): float(v) for k, v in zip(_h["region"].values, _h["sd"].values)}
    for c, hemi in enumerate(cols):
        ax = axes[c]
        # climatological spread of the NET anomaly, built from the same drag field —
        # without it there is no way to tell a routine swing from a real event
        if hemi in hsd:
            for k, al in ((2, 0.09), (1, 0.16)):
                ax.fill_between(
                    valid,
                    -k * hsd[hemi],
                    k * hsd[hemi],
                    color="#8a4b2a",
                    alpha=al,
                    lw=0,
                    zorder=0,
                    label=f"±{k}σ climatology" if k == 1 else None,
                )
        ax.plot(valid, arows[hemi]["fric"], color="#2166ac", lw=1.8, label="friction")
        ax.plot(valid, arows[hemi]["mtn"], color="#b2182b", lw=1.8, label="mountain")
        ax.plot(valid, arows[hemi]["sum"], color="k", lw=2.6, label="friction + mountain")
        if obs_by_day:
            odm = np.array([obs_by_day[hemi][d] for d in days])
            if cdm is not None:
                odm = odm - cdm[hemi]
            ax.plot(valid, odm, color="0.35", lw=2.2, ls="--", label="d(AAM)/dt")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(valid[0], valid[-1])
        ax.set_title(titles[hemi], fontsize=11, fontweight="bold")
        if c == 0:  # under the panels; out of the layout, which would otherwise widen this panel's slot by its width
            leg = ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=5, frameon=False)
            leg.set_in_layout(False)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        ax.tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("torque anomaly vs ERA5 1991–2020\n(Hadley = 10¹⁸ N m)", fontsize=9.5)
    fig.suptitle(
        f"AAM torque anomalies (ensemble mean) — IFS-ENS init {init:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5,
        -0.07,
        "Anomalies against the ERA5 1991–2020 climatology, everything on the same footing: the forecast tendency has its own "
        "climatological tendency removed too. Shaded = ±1σ and ±2σ of the net anomaly over 1991–2020.\n"
        "Each day is its 24 h mean: friction from the model's own surface stress, mountain torque from its surface "
        "pressure, d(AAM)/dt from its winds.\nEach hemisphere's gap between net surface torque (black) and d(AAM)/dt "
        "(dashed) is the gravity-wave-drag torque (not in ECMWF open data), cross-equatorial momentum transport (an "
        "internal flux, not a torque) and any non-closure of the forecast's own budget.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="0.35",
        linespacing=1.5,
    )
    fig.tight_layout()
    Path(args.ts_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.ts_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved {args.ts_out} (Global anomaly {arows['G']['sum'][0]:+.0f}→{arows['G']['sum'][-1]:+.0f} Hadley; "
        f"dM/dt {'overlaid' if obs else 'n/a (run aam.py first)'})"
    )

    # --- mountain-torque anomaly by range ---------------------------------
    # Was seven lines on one axis, which answered "what is the total" but not the
    # question the panel exists for: is THIS barrier doing something unusual? Each
    # range now gets its own small panel scaled to its own climatological spread
    # (torque_range_sd.nc, 1-sigma of the seasonally-detrended anomaly over ERA5
    # 1991-2020), so a 2-sigma Andes event reads as one at a glance instead of
    # being lost next to Antarctica.
    def box_net5(field, b):
        lo0, lo1, la0, la1 = b
        ym = (clat >= la0) & (clat <= la1)
        xm = (clon >= lo0) & (clon <= lo1)
        return (field.values * dA5[None, :, None])[:, ym][:, :, xm].sum((1, 2)) / HU

    sd = {}
    if RANGE_SD.exists():
        _s = xr.open_dataset(RANGE_SD)
        sd = {str(k): float(v) for k, v in zip(_s["range"].values, _s["sd"].values)}
    series = {nm: box_net5(ma, bx) for nm, bx in RANGES.items()}
    gtot = box_net5(ma, (0, 360, -90, 90))
    order = sorted(series, key=lambda k: -np.abs(series[k]).max())
    if args.series_out:
        import json as _json

        Path(args.series_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.series_out).write_text(
            _json.dumps(
                {
                    "model": "IFS-ENS",
                    "init": init.strftime("%Y-%m-%dT%HZ"),
                    "valid": [v.strftime("%Y-%m-%dT%H:00") for v in valid],  # each day's window midpoint
                    "units": "Hadley (1e18 N m), ensemble-mean mountain-torque anomaly vs ERA5 1991-2020",
                    "ranges": {nm: [round(float(x), 2) for x in series[nm]] for nm in series},
                    "global": [round(float(x), 2) for x in gtot],
                    "sd": {k: round(v, 2) for k, v in sd.items()},
                },
                separators=(",", ":"),
            )
        )
        print(f"saved {args.series_out}")

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = GridSpec(
        3, 3, height_ratios=[1.45, 1, 1], hspace=0.72, wspace=0.24, left=0.07, right=0.985, top=0.9, bottom=0.16
    )
    ax = fig.add_subplot(gs[0, :])
    gsd = sd.get("__global__")
    if gsd:
        for k, al in ((2, 0.10), (1, 0.18)):
            ax.fill_between(
                valid,
                -k * gsd,
                k * gsd,
                color="#8a4b2a",
                alpha=al,
                lw=0,
                label=f"±{k}σ climatology" if k == 1 else None,
            )
    ax.plot(valid, gtot, color="#1a1a1a", lw=2.8)
    ax.set_xlim(valid[0], valid[-1])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.grid(True, alpha=0.22)
    ax.set_ylabel("Hadley (10¹⁸ N m)")
    ax.set_title("Global mountain-torque anomaly", fontsize=11, fontweight="bold", loc="left")
    if gsd:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for lb in ax.get_xticklabels():
        lb.set_rotation(30)
        lb.set_ha("right")
        lb.set_fontsize(7.5)

    for k, nm in enumerate(order):
        a = fig.add_subplot(gs[1 + k // 3, k % 3])
        v = series[nm]
        s1 = sd.get(nm)
        if s1:
            a.fill_between(valid, -2 * s1, 2 * s1, color="#8a4b2a", alpha=0.09, lw=0)
            a.fill_between(valid, -s1, s1, color="#8a4b2a", alpha=0.17, lw=0)
        a.plot(valid, v, color="#b4541f", lw=2.2)
        a.set_xlim(valid[0], valid[-1])
        a.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        a.axhline(0, color="0.5", lw=0.7)
        a.grid(True, alpha=0.2)
        pk = v[np.argmax(np.abs(v))]
        tag = f"  peak {pk:+.0f}" + (f" ({pk / s1:+.1f}σ)" if s1 else "")
        a.set_title(nm + tag, fontsize=9, fontweight="bold", loc="left")
        a.tick_params(labelsize=7)
        if s1:
            a.set_ylim(min(-2.4 * s1, 1.15 * v.min()), max(2.4 * s1, 1.15 * v.max()))
        for lb in a.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha("right")
            lb.set_fontsize(6.5)
        if k % 3 == 0:
            a.set_ylabel("Hadley", fontsize=8)
    fig.suptitle(
        f"Which barriers are driving the mountain torque — IFS-ENS ensemble mean, init {init:%Y-%m-%d %HZ}",
        fontsize=12.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.005,
        "Form-drag torque anomaly vs the ERA5 1991–2020 climatology, integrated over each barrier and sorted by peak. "
        "Shaded bands are that barrier's own ±1σ and ±2σ of day-to-day anomaly variability over 1991–2020,\n"
        "so the panels are comparable even though the Andes routinely swing five times harder than the Alps.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.5,
    )
    Path(args.ranges_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.ranges_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.ranges_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
