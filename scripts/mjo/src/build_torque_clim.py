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
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6; HU = 1e18; DEG = np.pi / 180.0
TERMS = ("friction", "mountain", "gwd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="sample every N days")
    ap.add_argument("--y0", type=int, default=1991); ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--out", default="assets/sst/torque_seasonal.webp")
    args = ap.parse_args()

    o = xr.open_dataarray(OROG); lat = o.latitude.values; lon = o.longitude.values
    cos = np.cos(np.deg2rad(lat))
    dlam = np.deg2rad(abs(float(lon[1] - lon[0])))
    dhdlam = np.gradient(o.values, np.deg2rad(lon), axis=1)            # (lat,lon)

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks={"time": 1})
    # align ERA5 grid to the orography grid (both 0.25° ARCO; reindex guards any offset)
    def on_grid(da):
        da = da.reindex(latitude=lat, longitude=lon, method="nearest")
        return da.values

    want = pd.date_range(f"{args.y0}-01-01", f"{args.y1}-12-31", freq=f"{args.stride}D") + pd.Timedelta(hours=12)
    have = pd.to_datetime(ds.time.values)
    times = want[want.isin(have)]
    print(f"sampling {len(times)} ERA5 times ({args.stride}-day stride, {args.y0}-{args.y1})", flush=True)

    acc = {k: np.zeros((12, len(lat))) for k in TERMS}
    cnt = np.zeros(12)
    for n, t in enumerate(times):
        try:
            ew = on_grid(ds["mean_eastward_turbulent_surface_stress"].sel(time=t))
            gw = on_grid(ds["mean_eastward_gravity_wave_surface_stress"].sel(time=t))
            sp = on_grid(ds["surface_pressure"].sel(time=t))
        except Exception as e:                                        # noqa: BLE001
            print(f"  skip {t:%Y-%m-%d}: {e}"); continue
        m = t.month - 1
        # zonal-integrated torque per latitude, Hadley per degree latitude
        acc["friction"][m] += -(A**3) * cos**2 * (ew * dlam).sum(1) * DEG / HU
        acc["gwd"][m]      += -(A**3) * cos**2 * (gw * dlam).sum(1) * DEG / HU
        acc["mountain"][m] += -(A**2) * cos * (sp * dhdlam).sum(1) * dlam * DEG / HU
        cnt[m] += 1
        if n % 200 == 0:
            print(f"  {n}/{len(times)} … {t:%Y-%m-%d}", flush=True)

    for k in TERMS:
        acc[k] /= cnt[:, None]
    clim = xr.Dataset({k: (("month", "latitude"), acc[k]) for k in TERMS},
                      coords={"month": np.arange(1, 13), "latitude": lat})
    clim.attrs["units"] = "Hadley (1e18 N m) per degree latitude"
    clim.attrs["note"] = f"ERA5 {args.y0}-{args.y1}, {args.stride}-day stride; torque ON the atmosphere"
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}")

    # --- collapsed: each term integrated over Global / NH / SH, vs month ---
    figts, axts = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    doms = [("Global", np.ones_like(lat, bool)), ("Northern Hemisphere", lat > 0), ("Southern Hemisphere", lat < 0)]
    series = [("friction", "friction", "#2166ac"), ("mountain", "mountain", "#b2182b"),
              ("gwd", "GWD", "#1b7837"), ("sum", "sum", "k")]
    for ax, (dn, dm) in zip(axts, doms):
        for key, lbl, col in series:
            ts = (data[key][:, dm] * dphi).sum(1)               # integrate over the domain → Hadley
            ax.plot(months, ts, color=col, lw=2.6 if key == "sum" else 1.9, label=lbl)
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.25)
        ax.set_title(dn, fontsize=11, fontweight="bold")
        ax.set_xticks(months); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8)
    axts[0].set_ylabel("torque (Hadley = 10¹⁸ N m)"); axts[0].legend(fontsize=8.5, loc="best")
    figts.suptitle(f"Seasonal cycle of the integrated AAM surface-torque terms — ERA5 {args.y0}–{args.y1}",
                   fontsize=12.5, fontweight="bold")
    figts.tight_layout()
    tsout = str(Path(args.out).with_name("torque_seasonal_ts.webp"))
    figts.savefig(tsout, dpi=120, bbox_inches="tight"); plt.close(figts)
    print(f"saved {tsout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
