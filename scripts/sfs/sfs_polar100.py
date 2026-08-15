#!/usr/bin/env python3
"""SFS beta — 100-hPa polar-stereographic products from the daily members.

Two products (the monthly store stops at 200 hPa, so both come from the
daily stream, instantaneous fields on even leads):

1. assets/sfs/polar100.webp — month-1 mean (valid days 0-30) ensemble-mean
   height anomaly.
2. assets/sfs/anim/sfs_z100d/F*.webp + manifest — one frame per even lead
   out to day 46 for the loop player.

Baselines are the model's own 1991-2020 x 11-member reforecast (month-1
mean for the static, per-lead-day for the loop), each evaluated as
mean + own linear trend at the forecast year — a plain 30-year mean
leaves a near-uniform warming offset that saturates the color scale.
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
OUTPNG = REPO / "assets" / "sfs" / "polar100.webp"
ANIM = REPO / "assets" / "sfs" / "anim"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
LAT0 = 20          # plot + load domain: 20N poleward
MAXDAY = 30        # month-1 window: valid days 0-30 of the init month


def _open(url):
    import fsspec
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True,
                        decode_timedelta=True)


def _nh(ds):
    return ds.where(ds.lat >= LAT0, drop=True)


def _msel(ds, maxday=None):
    """Even-lead (instantaneous-field) indices, optionally capped by valid day."""
    days = pd.to_timedelta(ds.lead.values).days.values
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


def polar100_climo(month: int) -> dict:
    """Reforecast month-1 mean z100 (dam): 30-year mean, interannual σ, and
    linear trend (dam/yr) so the baseline can be evaluated at the forecast
    year — same convention as the daily anomaly maps."""
    f = CLIMDIR / f"polar100_climo_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _nh(_open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
             .sel(init=slice("1991", "2020")))
    sel = _msel(ds, MAXDAY)
    years = ds.init.dt.year.values.astype(np.float64)
    n_init = ds.sizes["init"]
    yearly = np.zeros((n_init, ds.sizes["lat"], ds.sizes["lon"]), np.float32)
    for yi in range(n_init):
        z = ds.HGT_100mb.isel(init=yi).values[:, sel]     # (11, n, lat, lon)
        yearly[yi] = z.mean(axis=(0, 1)) / 10.0           # m -> dam
        print(f"polar100 climo: init {yi + 1}/{n_init}", flush=True)
    out = {"mu": yearly.mean(axis=0), "sd": yearly.std(axis=0, ddof=1),
           "slope": _trend_fit(yearly, years), "y_mid": years.mean(),
           "lat": ds.lat.values, "lon": ds.lon.values}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"polar100 climo {month:02d}: cached "
          f"(trend {out['slope'].min():+.3f}..{out['slope'].max():+.3f} dam/yr)",
          flush=True)
    return out


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

    proj = ccrs.NorthPolarStereo(central_longitude=-90)
    fig = plt.figure(figsize=(9.4, 9.9))
    ax = fig.add_subplot(projection=proj)
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
    ax.coastlines(lw=0.6, color="0.45")
    ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 45, 60, 75])

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.03, aspect=45)
    cb.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.02, right=0.98)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=130, pad_inches=0.1)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    ds = _nh(_open(f"{BASE}/forecast/{issue}/atm_daily.zarr"))
    lat, lon = ds.lat.values, ds.lon.values
    lead_days = pd.to_timedelta(ds.lead.values).days.values
    sel = _msel(ds)                                   # all even leads (loop)
    sel1 = sel[lead_days[sel] <= MAXDAY]              # month-1 subset (static)
    z = ds.HGT_100mb.values[:, sel].mean(axis=0) / 10.0   # (n, lat, lon) dam
    u = ds.UGRD_100mb.values[:, sel].mean(axis=0)
    v = ds.VGRD_100mb.values[:, sel].mean(axis=0)
    m1 = np.isin(sel, sel1)

    # ── month-1 static ──────────────────────────────────────────────────────
    C = polar100_climo(t0.month)
    anom1 = z[m1].mean(axis=0) - (C["mu"] + C["slope"] * (t0.year - C["y_mid"]))
    print(f"z100 month-1 anom: min {anom1.min():+.1f} max {anom1.max():+.1f} dam",
          flush=True)
    render_polar(
        lat, lon, z[m1].mean(axis=0), anom1, u[m1].mean(axis=0), v[m1].mean(axis=0),
        f"SFS beta — 100 hPa, month-1 mean ({t0:%b %Y}, valid days 0–{MAXDAY})\n"
        "31-member ensemble mean · contours: mean height (dam) · vectors: mean wind",
        f"100-hPa height anomaly (dam) vs reforecast 1991–2020 mean + trend "
        f"evaluated at {t0.year}",
        OUTPNG, np.arange(-18, 18.1, 2.0))
    print(f"wrote {OUTPNG.relative_to(REPO)}", flush=True)

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
            f"SFS beta — 100 hPa · {valid:%a %b %d %Y} (day {int(lead_days[k])}) "
            f"· issue {t0:%b %Y}\n31-member ensemble mean · contours: mean height "
            "(dam) · vectors: mean wind",
            f"100-hPa height anomaly (dam) vs reforecast 1991–2020 per-lead mean "
            f"+ trend evaluated at {t0.year}",
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
