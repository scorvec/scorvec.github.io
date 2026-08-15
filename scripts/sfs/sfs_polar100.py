#!/usr/bin/env python3
"""SFS beta — month-1 mean 100-hPa polar-stereographic view.

Ensemble-mean month-1 (init month, valid days 0-30) 100-hPa height anomaly
vs the model's own 1991-2020 x 11-member reforecast month-1 mean, with the
absolute ensemble-mean height contoured and the absolute ensemble-mean
100-hPa wind as vectors. Heights and the anomaly baseline both come from
the daily stream (the monthly store stops at 200 hPa), and the hindcast
baseline means any lead-dependent model bias common to forecast and
reforecast cancels in the anomaly.

Output: assets/sfs/polar100.webp (+ climo cache polar100_climo_{MM}.npz).

    python scripts/sfs/sfs_polar100.py [--issue 202608]
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CLIMDIR = HERE / "data"
OUTPNG = REPO / "assets" / "sfs" / "polar100.webp"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
LAT0 = 20          # plot + load domain: 20N poleward
MAXDAY = 30        # month-1 window: valid days 0-30 of the init month


def _open(url):
    import fsspec
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True,
                        decode_timedelta=True)


def _nh(ds):
    return ds.where(ds.lat >= LAT0, drop=True)


def _msel(ds):
    """Even-lead (instantaneous-field) indices with valid day <= MAXDAY."""
    days = pd.to_timedelta(ds.lead.values).days.values
    probe = ds.HGT_100mb
    idx = {"member": 0, "lat": 0, "lon": 0}
    if "init" in probe.dims:
        idx["init"] = 0
    col = probe.isel(**idx).values
    return np.where(np.isfinite(col) & (days <= MAXDAY))[0]


def polar100_climo(month: int) -> dict:
    """Reforecast month-1 mean z100 (dam): 30-year mean, interannual σ, and
    the per-gridpoint linear trend (dam/yr) so the baseline can be evaluated
    at the forecast year — same convention as the daily anomaly maps; a plain
    1991-2020 mean leaves a near-uniform warming offset that saturates the
    color scale before any pattern shows."""
    f = CLIMDIR / f"polar100_climo_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _nh(_open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
             .sel(init=slice("1991", "2020")))
    sel = _msel(ds)
    years = ds.init.dt.year.values.astype(np.float64)
    n_init = ds.sizes["init"]
    yearly = np.zeros((n_init, ds.sizes["lat"], ds.sizes["lon"]), np.float32)
    for yi in range(n_init):
        z = ds.HGT_100mb.isel(init=yi).values[:, sel]     # (11, n, lat, lon)
        yearly[yi] = z.mean(axis=(0, 1)) / 10.0           # m -> dam
        print(f"polar100 climo: init {yi + 1}/{n_init}", flush=True)
    yc = years - years.mean()
    slope = np.tensordot(yc, yearly - yearly.mean(axis=0), axes=(0, 0)) / (yc ** 2).sum()
    out = {"mu": yearly.mean(axis=0), "sd": yearly.std(axis=0, ddof=1),
           "slope": slope.astype(np.float32), "y_mid": years.mean(),
           "lat": ds.lat.values, "lon": ds.lon.values}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"polar100 climo {month:02d}: cached "
          f"(trend {slope.min():+.3f}..{slope.max():+.3f} dam/yr)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    C = polar100_climo(t0.month)
    ds = _nh(_open(f"{BASE}/forecast/{issue}/atm_daily.zarr"))
    sel = _msel(ds)
    lat, lon = ds.lat.values, ds.lon.values
    z = ds.HGT_100mb.values[:, sel].mean(axis=(0, 1)) / 10.0   # (lat, lon) dam
    u = ds.UGRD_100mb.values[:, sel].mean(axis=(0, 1))
    v = ds.VGRD_100mb.values[:, sel].mean(axis=(0, 1))
    anom = z - (C["mu"] + C["slope"] * (t0.year - C["y_mid"]))
    print(f"z100 anom (trend-adjusted): min {anom.min():+.1f} max {anom.max():+.1f} dam "
          f"({len(sel)} even-lead days)", flush=True)

    # cyclic point so filled contours close at the dateline
    lonc = np.concatenate([lon, [lon[0] + 360]])
    zc, ac = [np.concatenate([x, x[:, :1]], axis=1) for x in (z, anom)]
    uc, vc = [np.concatenate([x, x[:, :1]], axis=1) for x in (u, v)]

    proj = ccrs.NorthPolarStereo(central_longitude=-90)
    fig = plt.figure(figsize=(9.4, 9.9))
    ax = fig.add_subplot(projection=proj)
    ax.set_extent([-180, 180, LAT0, 90], ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 200)
    circ = mpath.Path(np.column_stack([np.sin(th), np.cos(th)]) * 0.5 + 0.5)
    ax.set_boundary(circ, transform=ax.transAxes)

    lv = np.arange(-18, 18.1, 2.0)
    cf = ax.contourf(lonc, lat, ac, levels=lv, cmap="RdBu_r", extend="both",
                     transform=ccrs.PlateCarree())
    cs = ax.contour(lonc, lat, zc, levels=np.arange(1590, 1720, 6),
                    colors="0.25", linewidths=0.8, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt="%d", fontsize=7)
    st = 12
    ax.quiver(lonc[::st], lat[::st], uc[::st, ::st], vc[::st, ::st],
              transform=ccrs.PlateCarree(), regrid_shape=28, color="0.1",
              width=0.0022, scale=700, alpha=0.75)
    ax.coastlines(lw=0.6, color="0.45")
    ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 45, 60, 75])

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.03, aspect=45)
    cb.set_label("100-hPa height anomaly (dam) vs reforecast 1991–2020 mean + trend "
                 f"evaluated at {t0.year}", fontsize=9)
    ax.set_title(f"SFS beta — 100 hPa, month-1 mean ({t0:%b %Y}, valid days 0–{MAXDAY})\n"
                 "31-member ensemble mean · contours: mean height (dam) · "
                 "vectors: mean wind", fontsize=11, fontweight="bold", loc="left")
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.02, right=0.98)
    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=140, pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {OUTPNG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
