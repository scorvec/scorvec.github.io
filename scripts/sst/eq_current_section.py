#!/usr/bin/env python3
"""Equatorial Pacific zonal-current & thermocline section animator — Copernicus Marine 1/12° model.

A fixed-start loop (every day since 1 Mar 2026) of the depth × longitude slice along the equator
(1.5°S-1.5°N, 160°E-90°W): daily zonal current (shaded; eastward = red) with the 20 °C isotherm
(the thermocline) overlaid. It resolves the Equatorial Undercurrent (the eastward subsurface jet
at ~50-200 m), the surface South Equatorial Current, and the thermocline tilt — which flatten/weaken
as El Niño matures. The window grows (it is an event archive, NOT a rolling window) so the 2026
downwelling-Kelvin-wave progression never scrolls off.

The latitude-mean section (uo, thetao on time × depth × longitude) is kept in a persistent local
NetCDF store, so a daily run downloads only the new day(s) and a full re-render (e.g. a plot-style
change) costs zero download — it rebuilds the frames straight from the cache. Daily frames named by
date; later runs append the new day(s). Feeds sst_anim.html (region "eq_cur_section"). Runs in
run_local_sst.

    python scripts/sst/eq_current_section.py                     # append new day(s) + refresh the loop
    python scripts/sst/eq_current_section.py --start 2026-03-01  # (re)build from a given start date
    python scripts/sst/eq_current_section.py --rerender          # rebuild every frame from the cache (no download)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cmems"
STORE = CACHE / "section_store.nc"   # persistent latitude-mean (time,depth,lon) cache — gitignored
ANIM = HERE.parent.parent / "assets" / "sst" / "anim" / "eq_cur_section"
MANIFEST = HERE.parent.parent / "assets" / "sst" / "anim" / "eq_cur_section_manifest.json"
CUR = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
TEM = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
LON0, LON1, LATB, DMAX = 160, 270, 1.5, 400
START = date(2026, 3, 1)         # fixed window start — keep every day since 1 Mar 2026 so the 2026
                                 # downwelling-Kelvin-wave progression never scrolls off. This is an
                                 # event archive (the window grows), not a rolling window.


def _pull_reduced(d0: date, d1: date):
    """Subset uo + thetao over [d0, d1] in ≤31-day chunks, 1.5°S-1.5°N mean → (uo, thetao) as (time,depth,lon).

    The latitude average is taken before anything is kept, so only the small reduced section is held
    in memory / written to the cache (≈0.23 MB/day vs ≈12 MB/day for the full lat-resolved subset).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for ds, var in [(CUR, "uo"), (TEM, "thetao")]:
        pieces, c0 = [], d0
        while c0 <= d1:
            c1 = min(c0 + timedelta(days=30), d1)
            fn = f"sec_{var}_chunk.nc"
            cm.subset(dataset_id=ds, variables=[var], minimum_longitude=LON0, maximum_longitude=LON1,
                      minimum_latitude=-LATB, maximum_latitude=LATB, minimum_depth=0, maximum_depth=DMAX,
                      start_datetime=str(c0), end_datetime=str(c1),
                      output_filename=fn, output_directory=str(CACHE), overwrite=True)
            pieces.append(xr.open_dataset(CACHE / fn)[var].mean("latitude").load())
            c0 = c1 + timedelta(days=1)
        out[var] = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
    return out["uo"], out["thetao"]


def update_store(start: date, today: date) -> xr.Dataset:
    """Append any days in [start, today] not already in the persistent cache; trim to ≥ start, save.

    Only the missing tail is downloaded, so steady-state daily cost is one day (~0.5 MB). Returns the
    full cached section (uo, thetao on time × depth × longitude) for rendering — no download needed
    when every wanted day is already cached (e.g. a plot-style re-render).
    """
    ds = xr.open_dataset(STORE) if STORE.exists() else None
    have = set(pd.to_datetime(ds["time"].values).date) if ds is not None else set()
    want = [start + timedelta(days=k) for k in range((today - start).days + 1)]
    need = [d for d in want if d not in have]
    if need:
        u, t = _pull_reduced(min(need), max(need))
        new = xr.Dataset({"uo": u, "thetao": t})
        ds = new if ds is None else xr.concat([ds, new], dim="time")
        ds = ds.sortby("time")
        ds = ds.isel(time=~pd.Index(ds["time"].values).duplicated(keep="last"))     # dedup re-pulled days
        ds = ds.sel(time=ds["time"] >= np.datetime64(start))                         # honour the fixed start
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".nc.tmp"); ds.to_netcdf(tmp); tmp.replace(STORE)    # atomic write
        print(f"  cache: +{len(need)} day(s), {ds.sizes['time']} total "
              f"({str(ds['time'].values[0])[:10]}→{str(ds['time'].values[-1])[:10]})", flush=True)
    else:
        print(f"  cache: hit, {ds.sizes['time']} days (no download)", flush=True)
    return ds


def render_frame(u: xr.DataArray, t: xr.DataArray, dt: datetime, out: Path) -> None:
    """u, t are the latitude-mean section for one day → (depth, longitude)."""
    lon = u["longitude"].values; dep = u["depth"].values
    U = u.values; T = t.values
    fig, ax = plt.subplots(figsize=(12, 5.6))
    pm = ax.contourf(lon, dep, U, levels=np.arange(-1.2, 1.21, 0.15), cmap="RdBu_r", extend="both")
    cs20 = ax.contour(lon, dep, T, levels=[20], colors="black", linewidths=2.0)
    ax.clabel(cs20, fmt="20°C", fontsize=8)
    csw = ax.contour(lon, dep, T, levels=[26, 28], colors="black", linewidths=1.1, linestyles="--")
    ax.clabel(csw, fmt="%d°C", fontsize=7)                       # warm-pool base / upper thermocline
    ax.contour(lon, dep, T, levels=[12, 14, 16, 18, 22, 24], colors="0.35", linewidths=0.4, alpha=0.6)
    ax.set_ylim(DMAX, 0); ax.set_xlim(LON0, LON1)
    ax.set_xticks([160, 180, 200, 220, 240, 260])
    ax.set_xticklabels(["160E", "180", "160W", "140W", "120W", "100W"])
    ax.set_ylabel("depth (m)"); ax.set_xlabel("longitude")
    ax.axvline(190, color="0.5", lw=0.4, ls=":"); ax.axvline(240, color="0.5", lw=0.4, ls=":")
    ax.set_title(f"Equatorial Pacific zonal current & thermocline (1.5°S–1.5°N)  ·  {dt:%Y-%m-%d}\n"
                 "red = eastward (incl. the Equatorial Undercurrent); black = 20 / 26 / 28 °C isotherms", fontsize=10, pad=12)
    cb = fig.colorbar(pm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cb.set_label("zonal current (m s⁻¹)   ·   eastward +")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(START), help="window start date (YYYY-MM-DD); keep every day since")
    ap.add_argument("--rerender", action="store_true", help="rebuild every frame from the cache (no download)")
    args = ap.parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    ANIM.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    store = update_store(start, today)                  # download only the missing tail; cache the rest
    sdates = pd.to_datetime(store["time"].values).date
    want = [start + timedelta(days=k) for k in range((today - start).days + 1)]
    for d in want:
        fp = ANIM / f"{d:%Y%m%d}.webp"
        if fp.exists() and not args.rerender:
            continue
        i = int(np.argmin([abs((sd - d).days) for sd in sdates]))
        if abs((sdates[i] - d).days) > 1:
            continue                                    # day not in the model output
        render_frame(store["uo"].isel(time=i), store["thetao"].isel(time=i),
                     datetime(d.year, d.month, d.day), fp)
        print(f"  rendered {d}", flush=True)
    # drop only frames before the fixed start (e.g. the old rolling-window tail); keep everything since
    entries = []
    for fp in sorted(ANIM.glob("*.webp")):
        try:
            dt = datetime.strptime(fp.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if dt < start:
            fp.unlink(); continue
        entries.append({"idx": len(entries), "file": fp.name,
                        "date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%-d %b %Y")})
    MANIFEST.write_text(json.dumps({"ver": today.strftime("%Y%m%d"),
                                    "regions": {"eq_cur_section": {"label": "Equatorial zonal current & thermocline",
                                                                   "frames": entries}}}))
    print(f"wrote {len(entries)} frames + {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
