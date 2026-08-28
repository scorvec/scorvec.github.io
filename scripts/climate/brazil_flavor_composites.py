#!/usr/bin/env python3
"""
El Niño flavor composite differences over Brazil — 20th Century Reanalysis v3.

NDJFM composites of detrended 2-m temperature and precipitation anomalies,
summers 1870/71–2014/15, for events classified from ERSST relative indices
(east-lean = rel. Niño-1+2 − rel. Niño-4, NDJFM mean, RONI ≥ 0.5):
  EP  east-based  (east-lean ≥ +0.75):  1876, 1877, 1896, 1982, 1997
  CP  Modoki      (east-lean ≤ −0.25):  16 events
  REG regular     (in between):         18 events

Four maps: (EP−CP) and (EP−REG) for temperature (°C) and precip (mm/day),
Welch-t stippling at 90 %. Detrending is a per-gridcell linear fit over the
full record, so century drift and 20CR's early-era biases don't masquerade
as flavor signal.

Output: ~/brazil_flavor_composites.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVENTS = json.loads((Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
                          "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad/"
                          "flavor_events.json")).read_text())
EXTENT = (-85, -30, -40, 12)          # South America, Brazil-centred
Y0, Y1 = 1870, 2014


def ndjfm_stack(path: Path, var: str, to=None,
                months=(11, 12, 1, 2, 3)) -> xr.DataArray:
    """(summer, lat, lon) mean over `months` in the SA window, detrended per
    cell. months=(m,) gives a single-month cube."""
    ds = xr.open_dataset(path)
    da = ds[var]
    lon = da["lon"]
    if float(lon.max()) > 180:
        sel_lon = (lon >= EXTENT[0] % 360) & (lon <= EXTENT[1] % 360)
        da = da.sel(lon=sel_lon)
        da = da.assign_coords(lon=(da["lon"] + 180) % 360 - 180)
    da = da.sortby("lat").sel(lat=slice(EXTENT[2] - 2, EXTENT[3] + 2)).load()
    if to == "C" and float(da.max()) > 200:
        da = da - 273.15
    if to == "mmday":
        u = ds[var].attrs.get("units", "")
        cm = ds[var].attrs.get("cell_methods", "")
        if "3-hourly" in cm:
            da = da * 8.0                         # mean mm per 3 h → mm/day
        elif "/s" in u.lower():
            da = da * 86400.0                     # kg m-2 s-1 → mm/day
    t = pd.DatetimeIndex(da["time"].values)
    stacks = []
    yrs = []
    for y in range(Y0, Y1 + 1):
        stamps = [pd.Timestamp(y if m >= 11 else y + 1, m, 1) for m in months]
        idx = t.get_indexer(stamps)
        if (idx < 0).any():
            continue
        stacks.append(da.isel(time=idx).mean("time"))
        yrs.append(y)
    cube = xr.concat(stacks, dim=pd.Index(yrs, name="summer"))
    # detrend per gridcell
    yr = np.asarray(yrs, float)
    yc = yr - yr.mean()
    v = cube.values
    slope = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
    fit = v.mean(0)[None] + yc[:, None, None] * slope[None]
    return cube.copy(data=v - fit)


def welch_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return stats.ttest_ind(a, b, axis=0, equal_var=False).pvalue


def main() -> int:
    print("building NDJFM cubes (temp, precip)…", flush=True)
    T = ndjfm_stack(DATA / "air.2m.mon.mean.nc", "air", to="C")
    P = ndjfm_stack(DATA / "apcp.mon.mean.nc", "apcp", to="mmday")
    print(f"  {T.sizes['summer']} summers, grid {T.sizes['lat']}×{T.sizes['lon']}; "
          f"precip units now mm/day (max {float(P.max()):.1f})")

    groups = {k: [y for y in EVENTS[k] if y in T["summer"].values]
              for k in ("EP", "CP", "REG")}
    print("  events used:", {k: len(v) for k, v in groups.items()})

    panels = []
    for var, cube, unit, vmax, cmap in (
            ("Temperature", T, "°C", 1.2, "RdBu_r"),
            ("Precipitation", P, "mm/day", 2.0, "BrBG")):
        A = cube.sel(summer=groups["EP"]).values
        for other, tag in (("CP", "EP − Modoki"), ("REG", "EP − regular")):
            B = cube.sel(summer=groups[other]).values
            d = A.mean(0) - B.mean(0)
            p = welch_p(A, B)
            panels.append((f"{var}: {tag}", d, p, unit, vmax, cmap, cube))

    fig = plt.figure(figsize=(13.5, 12.5), dpi=150)
    for i, (title, d, p, unit, vmax, cmap, cube) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        lv = np.linspace(-vmax, vmax, 17)
        cf = ax.contourf(cube["lon"], cube["lat"], d, levels=lv, cmap=cmap,
                         extend="both", transform=ccrs.PlateCarree())
        yy, xx = np.meshgrid(cube["lat"], cube["lon"], indexing="ij")
        m = p < 0.10
        ax.plot(xx[m], yy[m], ".", color="#222", ms=1.1, alpha=0.6,
                transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                       edgecolor="#333")
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25,
                       edgecolor="#888")
        ax.coastlines("50m", lw=0.6, color="#333")
        cb = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.035,
                          pad=0.02)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11, loc="left")
    fig.suptitle("NDJFM El Niño flavor composites over South America — "
                 "20CRv3, detrended, 1870/71–2014/15\n"
                 f"EP (east-based, n={len(groups['EP'])}: 1876, 1877, 1896, 1982, "
                 f"1997) vs Modoki (n={len(groups['CP'])}) and regular "
                 f"(n={len(groups['REG'])}) — stippled where Welch p<0.10",
                 fontsize=12, x=0.03, ha="left")
    fig.text(0.012, 0.008,
             "Flavor from ERSST relative (RONI-style) indices; detrended per "
             "gridcell so century drift is not flavor. The 2026/27 event is "
             "running east-lean ≈ +3 (30-day OISST) — firmly in the EP class.",
             fontsize=8, color="#555")
    out = Path.home() / "brazil_flavor_composites.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


def monthly_main() -> int:
    """Per-month composite differences: one figure per variable, rows = month,
    cols = (EP − Modoki, EP − regular). Saved to the home folder."""
    MNAME = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
    specs = [("Temperature", DATA / "air.2m.mon.mean.nc", "air", "C",
              "°C", 1.5, "RdBu_r", "temp"),
             ("Precipitation", DATA / "apcp.mon.mean.nc", "apcp", "mmday",
              "mm/day", 3.0, "BrBG", "precip")]
    for label, path, var, conv, unit, vmax, cmap, tag in specs:
        fig = plt.figure(figsize=(11.5, 24), dpi=140)
        k = 0
        groups = None
        for m in (11, 12, 1, 2, 3):
            cube = ndjfm_stack(path, var, to=conv, months=(m,))
            if groups is None:
                groups = {g: [y for y in EVENTS[g] if y in cube["summer"].values]
                          for g in ("EP", "CP", "REG")}
            A = cube.sel(summer=groups["EP"]).values
            for other, ttl in (("CP", "EP − Modoki"), ("REG", "EP − regular")):
                B = cube.sel(summer=groups[other]).values
                d = A.mean(0) - B.mean(0)
                p = welch_p(A, B)
                k += 1
                ax = fig.add_subplot(5, 2, k, projection=ccrs.PlateCarree())
                ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
                lv = np.linspace(-vmax, vmax, 17)
                cf = ax.contourf(cube["lon"], cube["lat"], d, levels=lv,
                                 cmap=cmap, extend="both",
                                 transform=ccrs.PlateCarree())
                yy, xx = np.meshgrid(cube["lat"], cube["lon"], indexing="ij")
                mk = p < 0.10
                ax.plot(xx[mk], yy[mk], ".", color="#222", ms=0.9, alpha=0.6,
                        transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5,
                               edgecolor="#333")
                ax.coastlines("50m", lw=0.5, color="#333")
                cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
                cb.set_label(unit, fontsize=8)
                cb.ax.tick_params(labelsize=7)
                ax.set_title(f"{MNAME[m]} — {ttl}", fontsize=10, loc="left")
        fig.suptitle(f"{label}: El Niño flavor differences by month — 20CRv3 "
                     f"detrended, 1870/71–2014/15\nEP n={len(groups['EP'])} "
                     f"(1876, 1877, 1896, 1982, 1997) · Modoki "
                     f"n={len(groups['CP'])} · regular n={len(groups['REG'])} "
                     "· stippled Welch p<0.10", fontsize=12, x=0.03, ha="left")
        out = Path.home() / f"brazil_flavor_monthly_{tag}.png"
        fig.savefig(out, facecolor="white", bbox_inches="tight",
                    pad_inches=0.15)
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    if "--monthly" in sys.argv:
        sys.exit(monthly_main())
    sys.exit(main())
