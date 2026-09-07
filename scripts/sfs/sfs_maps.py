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
import json
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

VARS = ("tmp2m", "pratesfc", "prmsl", "z500", "u850", "v850", "u200", "v200", "vpot200")

# figure key -> (variables consumed, colormap, scale factor, units, title)
FIGS = {
    "sst":    (("SST",),          "RdBu_r",   1.0,     "°C",     "Sea-surface temperature anomaly"),
    "t2m":    (("tmp2m",),        "RdBu_r",   1.0,     "°C",     "2 m temperature anomaly (land)"),
    "precip": (("pratesfc",),     "BrBG",     86400.0, "mm/day", "Precipitation anomaly"),
    "mslp":   (("prmsl",),        "RdBu_r",   0.01,    "hPa",    "Mean sea-level pressure anomaly"),
    "z500":   (("z500",),         "RdBu_r",   0.1,     "dam",    "500 hPa height anomaly"),
    "wind850": (("u850", "v850"), "RdBu_r",   1.0,     "m/s",    "850 hPa wind anomaly (shading: zonal)"),
    "wind200": (("u200", "v200"), "RdBu_r",   1.0,     "m/s",    "200 hPa wind anomaly (shading: zonal)"),
    # SFS stores vpot200 with the OPPOSITE sign convention to the site's
    # pyshtools chi (chi_lm = -a²/(l(l+1)) delta_lm): flip so negative =
    # divergent outflow, matching the daily loop and the AIFS product
    "chi200": (("vpot200",),      "BrBG_r",   -1e-6,   "10⁶ m²/s", "200 hPa velocity potential anomaly (neg = outflow)"),
}
# limits picked from the actual anomaly distributions so strong-El Niño fields
# stay inside the scale instead of saturating whole regions
VLIM = {"sst": 4.0, "t2m": 5.0, "precip": 6.0, "mslp": 6.0, "z500": 9.0,
        "wind850": 6.0, "wind200": 12.0, "chi200": 6.0}
# discrete colour steps, the same construction as the SEAS5 maps (SST identical to seas5_build.MAP_SPEC)
LEVELS = {
    "sst": [-5, -4.5, -4, -3.5] + [round(x, 2) for x in np.arange(-3.0, 3.01, 0.2)] + [3.5, 4, 4.5, 5],
    "t2m": [-5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5],
    "precip": [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
    "mslp": [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
    "z500": [-9, -6, -4.5, -3, -1.5, -0.5, 0.5, 1.5, 3, 4.5, 6, 9],
    "wind850": [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
    "wind200": [-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12],
    "chi200": [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
}


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


def sst_clim(month: int) -> np.ndarray:
    f = CLIMDIR / f"clim_map_SST_{month:02d}.npy"
    if f.exists():
        return np.load(f)
    ds = _open(f"{BASE}/reforecast/{month:02d}/ocn_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1))).isel(lead=LEADS)
    print(f"clim SST {month:02d}: streaming ...", flush=True)
    c = ds["SST"].mean(("init", "member")).values.astype(np.float32)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.save(f, c)
    print(f"clim SST {month:02d}: cached", flush=True)
    return c


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

    # SST (ocean store, 1°) + its clim; also gives the ocean mask for t2m
    dso = _open(f"{BASE}/forecast/{issue}/ocn_monthly.zarr")
    sst = dso["SST"].isel(lead=LEADS).mean("member").values     # (lead, 181, 360)
    olat, olon = dso.latitude.values, dso.longitude.values
    clim["SST"] = sst_clim(month)
    ens["SST"] = sst
    print("NRT SST: loaded", flush=True)

    # land mask for t2m: nearest ocean cell finite -> ocean -> mask out
    ila = np.clip(np.round((lat[:, None] - olat[0]) / (olat[1] - olat[0])
                           ).astype(int), 0, len(olat) - 1)
    ilo = np.clip(np.round((lon[None, :] - olon[0]) / (olon[1] - olon[0])
                           ).astype(int), 0, len(olon) - 1)
    ocean = np.isfinite(sst[0])[ila, ilo]                       # (361, 720) bool
    ens["tmp2m"] = np.where(ocean[None, :, :], np.nan, ens["tmp2m"])

    SEASONS = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    MONTHLY = [[k] for k in LEADS]
    ASSETS.mkdir(parents=True, exist_ok=True)
    grids = {True: np.meshgrid(olon, olat), False: np.meshgrid(lon, lat)}

    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts" / "sst")); import mapstyle as MS       # the site's one map style
    from matplotlib.colors import BoundaryNorm

    def render(key, vs, cmap, scale, units, title, leads, fname, L):
        """One map per lead in the SEAS5 style (user 2026-09-07: consistent maps across the outlook pages)."""
        vmax = VLIM[key]
        is_sst = vs[0] == "SST"
        LONg, LATg = grids[is_sst]
        out = ASSETS / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        levels = LEVELS[key]
        fig, ax, H, pc = MS.open_map(kind="sst" if is_sst else "atm")
        a = {v: (ens[v][leads].mean(0) - clim[v][leads].mean(0)) * scale for v in vs}
        if is_sst:                                                  # smooth filled contours with a cyclic column, as on the SEAS5 SST map
            from cartopy.util import add_cyclic_point
            lon1d = LONg[0, :]; order = np.argsort(lon1d); fld = a[vs[0]][:, order]
            fld_c, lon_c = add_cyclic_point(fld, coord=lon1d[order])
            pm = ax.contourf(lon_c, LATg[:, 0], np.ma.masked_invalid(fld_c), levels=levels, cmap=plt.get_cmap(cmap, len(levels) - 1),
                             norm=BoundaryNorm(levels, len(levels) - 1), extend="both", transform=pc, zorder=1)
            cl = [x for x in np.arange(-6, 6.01, 0.5) if abs(x) > 1e-9]
            cs = ax.contour(lon_c, LATg[:, 0], np.ma.masked_invalid(fld_c), levels=cl, colors="#333", linewidths=0.35, transform=pc, zorder=2)
            ax.clabel(cs, fontsize=5.5, fmt=lambda v: f"{v:+.2g}", inline=True, inline_spacing=2)
        else:
            pm = ax.pcolormesh(LONg, LATg, a[vs[0]], cmap=plt.get_cmap(cmap, len(levels) - 1), norm=BoundaryNorm(levels, len(levels) - 1),
                               transform=pc, shading="auto", rasterized=True, zorder=1)
        if key == "chi200":                    # chi contours + divergent wind
            _sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
            from wind200_vpot import irrotational_wind
            ax.contour(LONg, LATg, a[vs[0]], levels=[l for l in np.arange(-12, 12.1, 1.5) if abs(l) > .1],
                       colors="k", linewidths=0.45, alpha=0.6, transform=pc, zorder=2)
            lat1d = LATg[:, 0]; lon1d = LONg[0, :]
            uchi, vchi = irrotational_wind(a[vs[0]] * 1e6, lat1d, lon1d)
            st = 22
            ax.quiver(LONg[::st, ::st], LATg[::st, ::st], uchi[::st, ::st], vchi[::st, ::st],
                      transform=pc, color="k", scale=90, width=0.0015, alpha=0.8, zorder=3)
        if len(vs) == 2:                       # wind vectors, subsampled
            st = 18
            ax.quiver(LONg[::st, ::st], LATg[::st, ::st], a[vs[0]][::st, ::st], a[vs[1]][::st, ::st],
                      transform=pc, color="k", scale=vmax * 30, width=0.0016, alpha=0.75, zorder=3)
        MS.features(ax, land_only=(key == "t2m"), states=not is_sst)
        mon = t0 + pd.DateOffset(months=L)
        MS.heading(fig, H, f"SFS beta {title} · {mon:%b %Y} (lead {L}) · {t0:%B %Y} issue",
                   f"Ensemble mean of 31 members minus the model's own {CLIM_Y0}–{CLIM_Y1} reforecast mean (11 members × 30 years) for the same start month and lead, "
                   + ("1° grid." if is_sst else "0.5° grid."))
        MS.colorbar(fig, H, pm, f"anomaly ({units})", levels)
        MS.save(fig, out)
        print(f"wrote {out.relative_to(REPO)}", flush=True)

    import time as _time
    for key, (vs, cmap, scale, units, title) in FIGS.items():
        # monthly animator loops only (the seasonal 2x2 multipanels are retired)
        name = f"sfs_{key}"
        frames = []
        for L in LEADS:
            fn = f"F{L:02d}.webp"
            render(key, vs, cmap, scale, units, title, [L], f"anim/{name}/{fn}", L)
            mon = (t0 + pd.DateOffset(months=L))
            frames.append({"idx": L, "file": fn, "date": f"{mon:%Y-%m}",
                           "label": f"{mon:%b %Y} · lead {L}"})
        (ASSETS / "anim" / f"{name}_manifest.json").write_text(json.dumps(
            {"ver": int(_time.time()), "days": len(frames),
             "regions": {name: {"label": f"SFS beta · {title} · monthly",
                                "n_frames": len(frames), "frames": frames}}}))
        print(f"loop {name}: {len(frames)} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
