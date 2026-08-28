#!/usr/bin/env python3
"""
Extended flavor composites: 20CRv3 reaches 1806, so the event sample can
more than double. Pre-1870 summers are classified from 20CR's own 2-m air
temperature over the index boxes (tropical-ocean T2m tracks the prescribed
SST), validated against the ERSST-based indices on the 1870–2014 overlap;
1870+ keeps the ERSST classification already used.

Caveat printed with the figure: 1806–1850s 20CR rests on very sparse
observations — early-era composites lean on the model + its SST boundary.

Output: ~/brazil_flavor_composites_1806.png + printed event lists.
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
sys.path.insert(0, str(HERE))
import brazil_flavor_composites as bfc                       # noqa: E402

Y0X, Y1X = 1806, 2014


def cr_indices() -> pd.DataFrame:
    """Monthly relative Niño indices from 20CR 2-m air, 1806–2015."""
    ds = xr.open_dataset(bfc.DATA / "air.2m.mon.mean.nc")
    air = ds["air"].sortby("lat")

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    clim = (air.sel(time=slice("1951-01-01", "1980-12-31"))
            .groupby("time.month").mean("time"))
    anom = air.groupby("time.month") - clim
    trop = wm(anom.sel(lat=slice(-20, 20)))
    def rel(la, lo):
        s = wm(anom.sel(lat=slice(*la), lon=slice(*lo))) - trop
        return s.rolling(time=3, center=True, min_periods=2).mean().to_series()
    return pd.DataFrame({
        "n34": rel((-5, 5), (190, 240)),
        "n12": rel((-10, 0), (270, 280)),
        "n4": rel((-5, 5), (160, 210)),
    })


def ndjfm(s: pd.Series, y: int) -> float:
    m = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
        [pd.Timestamp(y + 1, mm, 1) for mm in (1, 2, 3)]
    v = [s.get(x, np.nan) for x in m]
    return float(np.nanmean(v)) if np.isfinite(v).sum() >= 4 else np.nan


def main() -> int:
    cr = cr_indices()
    tab = pd.DataFrame({y: dict(roni=ndjfm(cr["n34"], y),
                                elean=ndjfm(cr["n12"], y) - ndjfm(cr["n4"], y))
                        for y in range(Y0X, Y1X + 1)}).T.dropna()

    # validate vs the ERSST classification table on the overlap
    ers = json.loads(Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
                          "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad/"
                          "flavor_events.json").read_text())
    # rebuild ERSST NDJFM series for overlap correlation
    ds = xr.open_dataset(HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc")
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)
    trop = wm(anom.sel(lat=slice(-20, 20)))
    def rel(la, lo):
        s = wm(anom.sel(lat=slice(*la), lon=slice(*lo))) - trop
        return s.rolling(time=3, center=True, min_periods=2).mean().to_series()
    e34, e12, e4 = rel((-5, 5), (190, 240)), rel((-10, 0), (270, 280)), \
        rel((-5, 5), (160, 210))
    etab = pd.DataFrame({y: dict(roni=ndjfm(e34, y),
                                 elean=ndjfm(e12, y) - ndjfm(e4, y))
                         for y in range(1870, Y1X + 1)}).T.dropna()
    ov = tab.index.intersection(etab.index)
    print(f"validation 1870–2014: r(RONI)={tab.loc[ov,'roni'].corr(etab.loc[ov,'roni']):.2f}  "
          f"r(east-lean)={tab.loc[ov,'elean'].corr(etab.loc[ov,'elean']):.2f}")

    # calibrate 20CR indices onto ERSST scale (regression on overlap), then
    # classify pre-1870 with the SAME thresholds
    def calib(col):
        b = np.polyfit(tab.loc[ov, col], etab.loc[ov, col], 1)
        return np.polyval(b, tab[col])
    tabc = pd.DataFrame({"roni": calib("roni"), "elean": calib("elean")},
                        index=tab.index)
    early = tabc[tabc.index < 1870]
    ep_e = sorted(early[(early.roni >= 0.5) & (early.elean >= 0.75)].index.astype(int))
    cp_e = sorted(early[(early.roni >= 0.5) & (early.elean <= -0.25)].index.astype(int))
    rg_e = sorted(early[(early.roni >= 0.5) & (early.elean > -0.25)
                        & (early.elean < 0.75)].index.astype(int))
    print(f"new pre-1870 events — EP: {ep_e}  CP: {cp_e}  REG: {rg_e}")

    events = {"EP": ep_e + ers["EP"], "CP": cp_e + ers["CP"],
              "REG": rg_e + ers["REG"]}
    print("totals:", {k: len(v) for k, v in events.items()})

    # composites on the full 1806–2014 span
    bfc.Y0 = Y0X                                  # widen the stack window
    print("building cubes 1806–2014…", flush=True)
    T = bfc.ndjfm_stack(bfc.DATA / "air.2m.mon.mean.nc", "air", to="C")
    P = bfc.ndjfm_stack(bfc.DATA / "apcp.mon.mean.nc", "apcp", to="mmday")
    groups = {k: [y for y in v if y in T["summer"].values]
              for k, v in events.items()}

    fig = plt.figure(figsize=(13.5, 12.5), dpi=150)
    panels = []
    for var, cube, unit, vmax, cmap in (
            ("Temperature", T, "°C", 1.2, "RdBu_r"),
            ("Precipitation", P, "mm/day", 2.0, "BrBG")):
        A = cube.sel(summer=groups["EP"]).values
        for other, tag in (("CP", "EP − Modoki"), ("REG", "EP − regular")):
            B = cube.sel(summer=groups[other]).values
            d = A.mean(0) - B.mean(0)
            p = stats.ttest_ind(A, B, axis=0, equal_var=False).pvalue
            panels.append((f"{var}: {tag}", d, p, unit, vmax, cmap, cube))
    order = [0, 2, 1, 3]
    for slot, i in enumerate(order):
        title, dd, p, unit, vmax, cmap, cube = panels[i]
        ax = fig.add_subplot(2, 2, slot + 1, projection=ccrs.PlateCarree())
        ax.set_extent(bfc.EXTENT, crs=ccrs.PlateCarree())
        lv = np.linspace(-vmax, vmax, 17)
        cf = ax.contourf(cube["lon"], cube["lat"], dd, levels=lv, cmap=cmap,
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
        cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(unit, fontsize=9)
        ax.set_title(f"NDJFM {title}", fontsize=11, loc="left")
    fig.suptitle("El Niño flavor composites, EXTENDED 1806/07–2014/15 — 20CRv3, "
                 "detrended\n"
                 f"EP n={len(groups['EP'])} · Modoki n={len(groups['CP'])} · "
                 f"regular n={len(groups['REG'])} — pre-1870 events classified "
                 "from calibrated 20CR-air indices · stippled Welch p<0.10",
                 fontsize=12, x=0.03, ha="left")
    fig.text(0.012, 0.008,
             "Caveat: 1806–1850s 20CR rests on very sparse observations; early "
             "fields lean on the model and its SST boundary conditions.",
             fontsize=8, color="#555")
    out = Path.home() / "brazil_flavor_composites_1806.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
