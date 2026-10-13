#!/usr/bin/env python3
"""ERA5 seasonal climatology of the three AAM surface-torque terms, by latitude.

Torque ON the atmosphere (eastward = adds westerly AAM):
  friction (turbulent):  T_f(φ) = -a³ cos²φ ∮ τ^turb_λ dλ      [τ from ERA5 mean_eastward_turbulent_surface_stress]
  gravity-wave drag:     T_g(φ) = -a³ cos²φ ∮ τ^gwd_λ  dλ      [ERA5 mean_eastward_gravity_wave_surface_stress]
  mountain (form drag):  T_m(φ) = -a²  cosφ ∮ p_s ∂h/∂λ dλ     [ERA5 surface_pressure + cached orography]
Sign verified: ERA5 eastward stress has the same sign as the surface wind, so the
torque on the atmosphere is MINUS the stress (the surface/mountain brakes the flow).

Samples ARCO-ERA5 every --stride days over 1991–2020 at 12Z, accumulates the
zonal-integrated torque per latitude into monthly bins → data/reference/torque_seasonal.nc,
and renders a latitude×month seasonal-cycle figure (friction | mountain | GWD | sum).

    python src/build_torque_clim.py --stride 5 --out assets/sst/torque_seasonal.webp
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
NC_OUT = REF / "torque_seasonal.nc"
CACHE = Path.home() / "mjo" / "era5_cache"          # reduced ERA5 samples, kept for re-use
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6; HU = 1e18; DEG = np.pi / 180.0
TERMS = ("friction", "mountain", "gwd")
# Measured non-closure, stated on both figures rather than left for the reader to find.
# CORRECTED 2026-10-12 (user approved, from the torque/ENSO study, torque_daily_era5.csv.gz): this script's
# 12Z-only, 5-day-stride sample gave -4.48 +/- 0.65 Hadley for the annual mean of friction+mountain+GWD (which
# must be ~zero: AAM has no trend). That figure is a SAMPLING ARTEFACT. Sampled 4x daily (00/06/12/18Z) over
# 1991-2020 the same sum is -11.3 +/- 0.6 Hadley (friction -9.6, mountain +1.1, GWD -2.8; se from the 30 annual
# means), against 12Z-only friction -6.1, mountain +4.8, GWD -3.2: the semidiurnal pressure tide aliases ~+4 Hadley
# into the 12Z mountain torque and the land diurnal cycle ~+3.5 into the 12Z friction. Hourly ERA5 for 2026 gives a
# near-constant residual dM/dt - torques of +15.6 Hadley (motion + mass AAM). The mean is also not resolution-
# converged (global mountain torque +3.1 Hadley at 0.25 deg, -7.7 at 0.5, -23.7 at 1.0, while the day-to-day sd
# barely moves), and ERA5's analysis increments add AAM at every 4D-Var window. The SHAPE of the seasonal cycle
# still matches the tendency implied by the ERA5 AAM climatology (r = 0.90). All of this sits in the MEAN, so the
# anomaly products are unaffected. Verified NOT to be a coding error -- the mountain formula reproduces an
# analytic ridge+pressure-wave case to -0.000%, both null tests are exact, and +h dp/dlam and -p dh/dlam agree to
# machine precision on 120 real ERA5 fields.
NONCLOSURE = ("Annual-mean friction + mountain + GWD should be ~zero (AAM has no trend). Sampled 4x daily, ERA5 1991-2020 gives -11.3 +/- 0.6 Hadley; "
              "this figure's 12Z-only sample shows -4.5 because the semidiurnal pressure tide aliases into the mountain term\n"
              "(+4.8 Hadley at 12Z vs +1.1 for the 4x-daily mean) and the land diurnal cycle into friction. The seasonal SHAPE matches the tendency implied "
              "by the ERA5 AAM climatology (r = 0.90), but the absolute level is not determined:\n"
              "the global-mean mountain torque is not resolution-converged (+3.1 Hadley at 0.25 deg, -7.7 at 0.5, -23.7 at 1.0) and ERA5's analysis "
              "increments add AAM every 12 h. Read the seasonal structure, not the absolute level.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="sample every N days")
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--out", default="assets/sst/torque_seasonal.webp")
    args = ap.parse_args()

    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    cos = np.cos(np.deg2rad(lat)); dlam = np.deg2rad(abs(float(lon[1] - lon[0])))
    cpath = CACHE / f"torque_seasonal_terms_{args.y0}-{args.y1}_s{args.stride}.nc"
    if cpath.exists():
        print(f"reusing cached ERA5 samples: {cpath}", flush=True)
        c = xr.open_dataset(cpath)
        fr, gw, mt = c["friction"].values, c["gwd"].values, c["mountain"].values   # (time, lat)
        months_of = pd.to_datetime(c.time.values).month.values
    else:
        fs = gcsfs.GCSFileSystem(token="anon")
        ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
        def on_grid(da): return da.reindex(latitude=lat, longitude=lon, method="nearest").values
        want = pd.date_range(f"{args.y0}-01-01", f"{args.y1}-12-31", freq=f"{args.stride}D") + pd.Timedelta(hours=12)
        times = want[want.isin(pd.to_datetime(ds.time.values))]
        print(f"sampling {len(times)} ERA5 times ({args.stride}-day stride, {args.y0}-{args.y1})", flush=True)
        frl, gwl, mtl, tl = [], [], [], []
        for n, t in enumerate(times):
            try:
                ew = on_grid(ds["mean_eastward_turbulent_surface_stress"].sel(time=t))
                gwf = on_grid(ds["mean_eastward_gravity_wave_surface_stress"].sel(time=t))
                sp = on_grid(ds["surface_pressure"].sel(time=t))
            except Exception as e:                                    # noqa: BLE001
                print(f"  skip {t:%Y-%m-%d}: {e}"); continue
            dpdlam = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * dlam)   # periodic ∂p_s/∂λ
            frl.append(-(A**3) * cos**2 * (ew * dlam).sum(1) * DEG / HU)
            gwl.append(-(A**3) * cos**2 * (gwf * dlam).sum(1) * DEG / HU)
            mtl.append((A**2) * cos * (o.values * dpdlam).sum(1) * dlam * DEG / HU)   # +h·∂p_s/∂λ (periodic)
            tl.append(t)
            if n % 200 == 0:
                print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)
        fr, gw, mt = np.stack(frl), np.stack(gwl), np.stack(mtl)
        CACHE.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"friction": (("time", "latitude"), fr), "gwd": (("time", "latitude"), gw),
                    "mountain": (("time", "latitude"), mt)},
                   coords={"time": pd.DatetimeIndex(tl), "latitude": lat}).to_netcdf(cpath)
        print(f"saved ERA5 samples → {cpath}  ({(fr.nbytes+gw.nbytes+mt.nbytes)/1e6:.0f} MB)", flush=True)
        months_of = pd.DatetimeIndex(tl).month.values

    acc = {k: np.zeros((12, len(lat))) for k in TERMS}; cnt = np.zeros(12)
    for i, mo in enumerate(months_of):
        acc["friction"][mo - 1] += fr[i]; acc["gwd"][mo - 1] += gw[i]; acc["mountain"][mo - 1] += mt[i]; cnt[mo - 1] += 1
    for k in TERMS:
        acc[k] /= cnt[:, None]
    clim = xr.Dataset({k: (("month", "latitude"), acc[k]) for k in TERMS},
                      coords={"month": np.arange(1, 13), "latitude": lat})
    clim.attrs["note"] = f"ERA5 {args.y0}-{args.y1}; torque ON atmosphere; mountain = +h ∂p_s/∂λ (periodic)"
    clim.to_netcdf(NC_OUT)
    print(f"wrote {NC_OUT}  (counts/month: {cnt.astype(int)})")

    # --- seasonal-cycle figure: latitude × month, one panel per term + the sum ---
    mm = (lat >= -80) & (lat <= 80)
    months = np.arange(1, 13)
    panels = [("friction", "Friction (turbulent) torque"), ("mountain", "Mountain (form-drag) torque"),
              ("gwd", "Gravity-wave-drag torque"), ("sum", "Sum (friction + mountain + GWD)")]
    import scipy.ndimage as ndi
    data = {k: ndi.gaussian_filter1d(acc[k], 1.6, axis=1) for k in TERMS}   # smooth in latitude for clean contours
    data["sum"] = sum(data[k] for k in TERMS)
    dphi = abs(lat[1] - lat[0])
    # PER-PANEL scale: GWD's density is ~15% of friction's, so a shared scale washes
    # it out — but its broad integral (∫) is a real budget term. Each panel its own
    # colourbar; the ∫ annotation gives the global tendency so magnitudes stay honest.
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6), sharey=True)
    for ax, (k, ttl) in zip(axes, panels):
        lim = float(np.nanpercentile(np.abs(data[k][:, mm]), 99.5)) or 1e-9
        gint = float(np.nanmean(np.nansum(data[k], axis=1)) * dphi)         # annual-mean global tendency, Hadley
        pc = ax.contourf(months, lat[mm], data[k][:, mm].T, levels=np.linspace(-lim, lim, 21),
                         cmap="RdBu_r", extend="both")
        ax.contour(months, lat[mm], data[k][:, mm].T, levels=[0], colors="0.3", linewidths=0.6)
        ax.set_title(f"{ttl}\n∫ ≈ {gint:+.0f} Hadley (annual)", fontsize=10, fontweight="bold")
        ax.set_xticks(months); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8)
        ax.axhline(0, color="0.5", lw=0.7); ax.set_yticks([-60, -30, 0, 30, 60])
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13)
        cb.set_label("Hadley/°lat", fontsize=7); cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel("latitude")
    fig.suptitle(f"Seasonal cycle of the AAM surface-torque terms by latitude — ERA5 {args.y0}–{args.y1}"
                 "   (each panel own scale · red = adds westerly AAM · ∫ = global tendency)",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.04, NONCLOSURE, ha="center", va="top", fontsize=8, color="0.35", linespacing=1.45)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}")

    # --- collapsed: each term integrated over Global / NH / SH, vs month ---
    figts, axts = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    eqw = 0.5 * np.isclose(lat, 0.0)                 # φ=0 row half to each hemisphere
    doms = [("Global", np.ones_like(lat, float)),
            ("Northern Hemisphere", (lat > 0).astype(float) + eqw),
            ("Southern Hemisphere", (lat < 0).astype(float) + eqw)]
    series = [("friction", "friction", "#2166ac"), ("mountain", "mountain", "#b2182b"),
              ("gwd", "GWD", "#1b7837"), ("sum", "sum", "k")]
    for ax, (dn, dm) in zip(axts, doms):
        for key, lbl, col in series:
            ts = (data[key] * dm[None, :] * dphi).sum(1)        # integrate over the domain → Hadley
            ax.plot(months, ts, color=col, lw=2.6 if key == "sum" else 1.9, label=lbl)
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.25)
        ax.set_title(dn, fontsize=11, fontweight="bold")
        ax.set_xticks(months); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8)
    axts[0].set_ylabel("torque (Hadley = 10¹⁸ N m)"); axts[0].legend(fontsize=8.5, loc="best")
    figts.suptitle(f"Seasonal cycle of the integrated AAM surface-torque terms — ERA5 {args.y0}–{args.y1}",
                   fontsize=12.5, fontweight="bold")
    figts.text(0.5, -0.10, NONCLOSURE, ha="center", va="top", fontsize=8, color="0.35", linespacing=1.45)
    figts.tight_layout()
    tsout = str(Path(args.out).with_name("torque_seasonal_ts.webp"))
    figts.savefig(tsout, dpi=120, bbox_inches="tight"); plt.close(figts)
    print(f"saved {tsout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
