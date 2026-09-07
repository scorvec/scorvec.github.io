#!/usr/bin/env python3
"""SFS beta — 100-hPa polar-stereographic products from the daily members.

One product (the monthly store stops at 200 hPa, so this comes from the
daily stream, instantaneous fields on even leads): the sfs_z100d loop —
assets/sfs/anim/sfs_z100d/F*.webp + manifest, one frame per even lead
out to day 46.

Baseline is the model's own 1991-2020 x 11-member reforecast per-lead-day
mean + its linear trend evaluated at the forecast year — a plain 30-year
mean leaves a near-uniform warming offset that saturates the color scale.
Absolute ensemble-mean height is contoured and the absolute mean wind
drawn as vectors on every panel.

    python scripts/sfs/sfs_polar100.py [--issue 202608]
"""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
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
ANIM = REPO / "assets" / "sfs" / "anim"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
LAT0 = 20          # plot + load domain: 20N poleward



def lead_days_of(lead) -> np.ndarray:
    """Lead in whole days whether the store's `lead` decodes as timedelta or as integer days
    (xarray 2026.7 leaves the int64 'days' coordinate undecoded; pd.to_timedelta on ints reads
    nanoseconds and every lead became 0, 2026-09-07)."""
    v = np.asarray(lead.values if hasattr(lead, "values") else lead)
    if np.issubdtype(v.dtype, np.timedelta64):
        return (v / np.timedelta64(1, "D")).astype(int)
    units = str(getattr(lead, "attrs", {}).get("units", "days")).lower()
    v = v.astype(float)
    if units.startswith("hour"): v = v / 24.0
    elif units.startswith("second"): v = v / 86400.0
    return np.round(v).astype(int)

def _open(url):
    import fsspec
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True,
                        decode_timedelta=True)


def _nh(ds):
    return ds.where(ds.lat >= LAT0, drop=True)


def _msel(ds, maxday=None):
    """Even-lead (instantaneous-field) indices, optionally capped by valid day."""
    days = lead_days_of(ds.lead)
    probe = ds.HGT_100mb
    idx = {"member": 0, "lat": 0, "lon": 0}
    if "init" in probe.dims:
        idx["init"] = 0
    col = probe.isel(**idx).values
    ok = np.isfinite(col)
    if maxday is not None:
        ok &= days <= maxday
    return np.where(ok)[0]


def _trend_fit(yearly, years):
    """Per-gridpoint OLS slope of yearly (n_init, ...) against years."""
    yc = years - years.mean()
    return (np.tensordot(yc, yearly - yearly.mean(axis=0), axes=(0, 0))
            / (yc ** 2).sum()).astype(np.float32)


def polar100_daily_climo(month: int) -> dict:
    """Per-lead-day reforecast z100 climatology (dam): 30-year mean and
    linear trend at every even lead — baseline for the daily loop frames."""
    f = CLIMDIR / f"polar100_daily_climo_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _nh(_open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
             .sel(init=slice("1991", "2020")))
    sel = _msel(ds)
    years = ds.init.dt.year.values.astype(np.float64)
    n_init = ds.sizes["init"]
    yearly = np.zeros((n_init, len(sel), ds.sizes["lat"], ds.sizes["lon"]),
                      np.float32)
    for yi in range(n_init):
        z = ds.HGT_100mb.isel(init=yi).values[:, sel]     # (11, n, lat, lon)
        yearly[yi] = z.mean(axis=0) / 10.0                # m -> dam, member mean
        print(f"polar100 daily climo: init {yi + 1}/{n_init}", flush=True)
    out = {"mu": yearly.mean(axis=0), "slope": _trend_fit(yearly, years),
           "y_mid": years.mean()}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"polar100 daily climo {month:02d}: cached ({len(sel)} leads)",
          flush=True)
    return out


def render_polar(lat, lon, z, anom, u, v, title, cbar_label, outpath,
                 levels):
    """One polar-stereo panel: filled anomaly, absolute-height contours,
    absolute wind vectors."""
    lonc = np.concatenate([lon, [lon[0] + 360]])
    zc = np.concatenate([z, z[:, :1]], axis=1)
    ac = np.concatenate([anom, anom[:, :1]], axis=1)
    uc = np.concatenate([u, u[:, :1]], axis=1)
    vc = np.concatenate([v, v[:, :1]], axis=1)

    import sys as _sys; _sys.path.insert(0, str(REPO / "scripts" / "sst")); import mapstyle as MS   # shared outlook map look
    proj = ccrs.NorthPolarStereo(central_longitude=-90)
    W, top, bot = 8.0, 1.05, 0.85; H = W + top + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.02, bot / H, 0.96, W / H], projection=proj)
    ax.set_extent([-180, 180, LAT0, 90], ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 200)
    circ = mpath.Path(np.column_stack([np.sin(th), np.cos(th)]) * 0.5 + 0.5)
    ax.set_boundary(circ, transform=ax.transAxes)

    cf = ax.contourf(lonc, lat, ac, levels=levels, cmap="RdBu_r", extend="both",
                     transform=ccrs.PlateCarree())
    cs = ax.contour(lonc, lat, zc, levels=np.arange(1590, 1720, 6),
                    colors="0.25", linewidths=0.8, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt="%d", fontsize=7)
    st = 12
    ax.quiver(lonc[::st], lat[::st], uc[::st, ::st], vc[::st, ::st],
              transform=ccrs.PlateCarree(), regrid_shape=28, color="0.1",
              width=0.0022, scale=700, alpha=0.75)
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    ax.gridlines(linewidth=0.3, color="#888", alpha=0.5, ylocs=[30, 45, 60, 75], zorder=4)
    head, _, sub = title.partition("\n")
    MS.heading(fig, H, head, sub or "Ensemble mean of 31 members; anomaly against the model's own 1991–2020 reforecast mean plus trend at the forecast year; contours the mean 100 hPa height (dam), arrows the mean wind.",
               title_size=12, sub_size=8, wrap=118)
    MS.colorbar(fig, H, cf, cbar_label)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    MS.save(fig, outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    ds = _nh(_open(f"{BASE}/forecast/{issue}/atm_daily.zarr"))
    lat, lon = ds.lat.values, ds.lon.values
    lead_days = lead_days_of(ds.lead)
    sel = _msel(ds)                                   # all even leads
    z = ds.HGT_100mb.values[:, sel].mean(axis=0) / 10.0   # (n, lat, lon) dam
    u = ds.UGRD_100mb.values[:, sel].mean(axis=0)
    v = ds.VGRD_100mb.values[:, sel].mean(axis=0)

    # ── daily loop ──────────────────────────────────────────────────────────
    D = polar100_daily_climo(t0.month)
    base = D["mu"] + D["slope"] * (t0.year - D["y_mid"])   # (n, lat, lon)
    name = "sfs_z100d"
    outdir = ANIM / name
    outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, k in enumerate(sel):
        valid = t0 + pd.Timedelta(days=int(lead_days[k]))
        anom = z[i] - base[i]
        render_polar(
            lat, lon, z[i], anom, u[i], v[i],
            f"SFS beta 100 hPa height anomaly · {valid:%b %d %Y} (day {int(lead_days[k])}) · {t0:%b %Y} issue\n"
            f"Ensemble mean of 31 members against the model's own 1991–2020 reforecast per-lead mean plus trend at {t0.year}; contours the mean 100 hPa height (dam), arrows the mean wind.",
            "anomaly (dam)",
            outdir / f"F{i:02d}.webp", np.arange(-24, 24.1, 3.0))
        frames.append({"idx": i, "file": f"F{i:02d}.webp",
                       "date": f"{valid:%Y-%m-%d}",
                       "label": f"{valid:%b %d} · day {int(lead_days[k])}"})
        print(f"frame {i + 1}/{len(sel)}", flush=True)
    (ANIM / f"{name}_manifest.json").write_text(json.dumps(
        {"ver": int(_time.time()), "days": len(frames),
         "regions": {name: {"label": "SFS 100-hPa height anomaly (polar)",
                            "n_frames": len(frames), "frames": frames}}}))
    print(f"loop {name}: {len(frames)} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
