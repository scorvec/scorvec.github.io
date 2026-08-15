#!/usr/bin/env python3
"""SFS beta global anomaly maps — ensemble mean vs own reforecast clim.

Fields (atm_monthly.zarr, 0.5°): 2 m temperature, precipitation rate,
MSLP, 500 hPa height, and 850/200 hPa vector wind. Anomalies are the
31-member NRT ensemble mean minus the model's OWN reforecast climatology
(1991-2020, 11 members × 30 years, same init month, per lead) — drift
and bias cancel per-lead, the same convention as the Niño-3.4 feed.

Panels per figure: the init month itself, then the three rolling
3-month seasons (leads 1-3, 4-6, 7-9).

Clim cache: scripts/sfs/data/clim_map_{var}_{MM}.npy (~10 MB each,
gitignored — the first build per init month streams ~14 GB of reforecast
chunks; afterwards it's free). Output: assets/sfs/*_maps.webp.

    python scripts/sfs/sfs_maps.py [--issue 202608] [--clim-only]
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ASSETS = REPO / "assets" / "sfs"
CLIMDIR = HERE / "data"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
CLIM_Y0, CLIM_Y1 = 1991, 2020
LEADS = list(range(0, 10))                       # month 1 + three seasons

VARS = ("tmp2m", "pratesfc", "prmsl", "z500", "u850", "v850", "u200", "v200")

# figure key -> (variables consumed, colormap, scale factor, units, title)
FIGS = {
    "t2m":    (("tmp2m",),        "RdBu_r",   1.0,     "°C",     "2 m temperature anomaly"),
    "precip": (("pratesfc",),     "BrBG",     86400.0, "mm/day", "Precipitation anomaly"),
    "mslp":   (("prmsl",),        "RdBu_r",   0.01,    "hPa",    "Mean sea-level pressure anomaly"),
    "z500":   (("z500",),         "RdBu_r",   0.1,     "dam",    "500 hPa height anomaly"),
    "wind850": (("u850", "v850"), "RdBu_r",   1.0,     "m/s",    "850 hPa wind anomaly (shading: zonal)"),
    "wind200": (("u200", "v200"), "RdBu_r",   1.0,     "m/s",    "200 hPa wind anomaly (shading: zonal)"),
}
VLIM = {"t2m": 3.0, "precip": 4.0, "mslp": 4.0, "z500": 6.0,
        "wind850": 4.0, "wind200": 8.0}


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def clim_maps(month: int) -> dict:
    """{var: (lead, lat, lon)} reforecast climatology, cached per var."""
    out, missing = {}, []
    for v in VARS:
        f = CLIMDIR / f"clim_map_{v}_{month:02d}.npy"
        if f.exists():
            out[v] = np.load(f)
        else:
            missing.append(v)
    if not missing:
        return out
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1))).isel(lead=LEADS)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    for v in missing:
        print(f"clim {v} {month:02d}: streaming "
              f"{ds.sizes['init']}y x {ds.sizes['member']}m ...", flush=True)
        c = ds[v].mean(("init", "member")).values.astype(np.float32)
        np.save(CLIMDIR / f"clim_map_{v}_{month:02d}.npy", c)
        out[v] = c
        print(f"clim {v} {month:02d}: cached", flush=True)
    return out


def season_label(t0: pd.Timestamp, leads: list[int]) -> str:
    if len(leads) == 1:
        return (t0 + pd.DateOffset(months=leads[0])).strftime("%b %Y")
    mons = [(t0 + pd.DateOffset(months=k)).strftime("%b")[0] for k in leads]
    yr = (t0 + pd.DateOffset(months=leads[-1])).year
    return "".join(mons) + f" {yr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    ap.add_argument("--clim-only", action="store_true")
    args = ap.parse_args()
    issue, month = args.issue, int(args.issue[4:6])
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    clim = clim_maps(month)
    if args.clim_only:
        return 0

    ds = _open(f"{BASE}/forecast/{issue}/atm_monthly.zarr")
    lat, lon = ds.lat.values, ds.lon.values
    ens = {}
    for v in VARS:
        ens[v] = ds[v].isel(lead=LEADS).mean("member").values  # (lead, lat, lon)
        print(f"NRT {v}: loaded", flush=True)

    PANELS = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ASSETS.mkdir(parents=True, exist_ok=True)
    LON, LAT = np.meshgrid(lon, lat)
    for key, (vs, cmap, scale, units, title) in FIGS.items():
        vmax = VLIM[key]
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.6),
                                 subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
        for ax, leads in zip(axes.flat, PANELS):
            a = {v: (ens[v][leads].mean(0) - clim[v][leads].mean(0)) * scale
                 for v in vs}
            shade = a[vs[0]]
            pm = ax.pcolormesh(LON, LAT, shade, cmap=cmap, vmin=-vmax, vmax=vmax,
                               transform=ccrs.PlateCarree(), rasterized=True)
            if len(vs) == 2:                       # wind vectors, subsampled
                st = 18
                ax.quiver(LON[::st, ::st], LAT[::st, ::st],
                          a[vs[0]][::st, ::st], a[vs[1]][::st, ::st],
                          transform=ccrs.PlateCarree(), color="k",
                          scale=vmax * 30, width=0.0016, alpha=0.75)
            ax.coastlines(lw=0.5, color="0.25")
            ax.add_feature(cfeature.BORDERS, lw=0.25, edgecolor="0.45")
            ax.set_global()
            ax.set_title(season_label(t0, leads), fontsize=11, loc="left",
                         fontweight="bold")
        cb = fig.colorbar(pm, ax=axes, orientation="horizontal",
                          fraction=0.045, pad=0.04, aspect=45, extend="both")
        cb.set_label(f"{title} ({units})", fontsize=10)
        fig.suptitle(f"SFS beta — {title} · issue {t0:%b %Y} · 31-member mean "
                     f"vs own reforecast {CLIM_Y0}–{CLIM_Y1}",
                     fontsize=13, fontweight="bold", y=0.99)
        out = ASSETS / f"{key}_maps.webp"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
