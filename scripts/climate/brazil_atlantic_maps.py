#!/usr/bin/env python3
"""
Atlantic teleconnections to South American summer — ENSO-removed regression
maps, 20CRv3, NDJFM 1870/71–2014/15.

Complements the c3s project's skill/bridge work (~/c3s: TNA forecast ACC
0.6–0.8, TSA moderate; basin-rain bridge weak) with the spatial view: partial
regressions of detrended NDJFM 2-m temperature and precipitation on

  TSA  Tropical South Atlantic (0–20°S, 30°W–10°E)   — S. Atl ITCZ / interior
  TNA  Tropical North Atlantic (5.5–23.5°N, 57.5–15°W) — Atlantic ITCZ dipole

with the ENSO signal (relative NDJFM Niño-3.4) regressed out of BOTH the
index and the field first, so these are Atlantic impacts beyond what ENSO
already explains. Stippled where the regression t-test gives p<0.10.

Output: ~/brazil_atlantic_teleconnections.png
"""
from __future__ import annotations

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
from brazil_flavor_composites import ndjfm_stack, DATA, EXTENT, Y0, Y1  # noqa: E402

ERSST = HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc"


def ersst_index(lat, lon, relative=False) -> pd.Series:
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    s = wm(anom.sel(lat=slice(*lat), lon=slice(*lon)))
    if relative:
        s = s - wm(anom.sel(lat=slice(-20, 20)))
    ser = s.to_series()
    out = {}
    for y in range(Y0, Y1 + 1):
        m = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1),
             pd.Timestamp(y + 1, 1, 1), pd.Timestamp(y + 1, 2, 1),
             pd.Timestamp(y + 1, 3, 1)]
        v = [ser.get(x, np.nan) for x in m]
        if np.isfinite(v).sum() >= 4:
            out[y] = float(np.nanmean(v))
    return pd.Series(out)


def detrend(s: pd.Series) -> pd.Series:
    x = s.index.to_numpy(float)
    b = np.polyfit(x, s.values, 1)
    return s - np.polyval(b, x)


def partial_regression(cube: xr.DataArray, idx: pd.Series, enso: pd.Series):
    """Regress cube on idx with enso removed from both. Returns (beta, p)."""
    yrs = [y for y in cube["summer"].values if y in idx.index and y in enso.index]
    C = cube.sel(summer=yrs).values                     # (n, lat, lon), detrended
    xi = idx.loc[yrs].to_numpy()
    ei = enso.loc[yrs].to_numpy()
    # residualize index on ENSO
    xi_r = xi - np.polyval(np.polyfit(ei, xi, 1), ei)
    # residualize field on ENSO (per cell)
    ec = ei - ei.mean()
    slope_e = np.tensordot(ec, C - C.mean(0), axes=(0, 0)) / (ec @ ec)
    C_r = C - C.mean(0) - ec[:, None, None] * slope_e[None]
    xc = xi_r - xi_r.mean()
    beta = np.tensordot(xc, C_r, axes=(0, 0)) / (xc @ xc)
    resid = C_r - xc[:, None, None] * beta[None]
    dof = len(yrs) - 3
    se = np.sqrt((resid ** 2).sum(0) / dof / (xc @ xc))
    p = 2 * stats.t.sf(np.abs(beta / np.maximum(se, 1e-12)), dof)
    return beta, p, len(yrs), float(np.std(xi_r))


def main() -> int:
    print("indices…", flush=True)
    # TSA crosses 0°E: build from two chunks
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    a1 = anom.sel(lat=slice(-20, 0), lon=slice(330, 360))
    a2 = anom.sel(lat=slice(-20, 0), lon=slice(0, 10))
    tsa_m = wm(xr.concat([a1, a2], dim="lon")).to_series()

    def season(ser):
        out = {}
        for y in range(Y0, Y1 + 1):
            m = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1),
                 pd.Timestamp(y + 1, 1, 1), pd.Timestamp(y + 1, 2, 1),
                 pd.Timestamp(y + 1, 3, 1)]
            v = [ser.get(x, np.nan) for x in m]
            if np.isfinite(v).sum() >= 4:
                out[y] = float(np.nanmean(v))
        return pd.Series(out)

    tsa = detrend(season(tsa_m))
    tna = detrend(ersst_index((5.5, 23.5), (302.5, 345)))
    enso = ersst_index((-5, 5), (190, 240), relative=True)
    print(f"  TSA sd {tsa.std():.2f} °C, TNA sd {tna.std():.2f} °C; "
          f"latest NDJFM ({tsa.index.max()}): TSA {tsa.iloc[-1]:+.2f}, "
          f"TNA {tna.iloc[-1]:+.2f}")

    print("cubes…", flush=True)
    T = ndjfm_stack(DATA / "air.2m.mon.mean.nc", "air", to="C")
    P = ndjfm_stack(DATA / "apcp.mon.mean.nc", "apcp", to="mmday")

    panels = []
    for name, idx in (("TSA", tsa), ("TNA", tna)):
        for var, cube, unit, vmax, cmap in (
                ("temperature", T, "°C per °C", 0.6, "RdBu_r"),
                ("precipitation", P, "mm/day per °C", 1.2, "BrBG")):
            beta, p, n, sd = partial_regression(cube, idx, enso)
            panels.append((f"{var} on {name} (ENSO removed)", beta, p,
                           unit, vmax, cmap, cube, n, sd, name))

    fig = plt.figure(figsize=(13.5, 12.5), dpi=150)
    order = [0, 2, 1, 3]                       # temp row then precip row
    for slot, i in enumerate(order):
        title, beta, p, unit, vmax, cmap, cube, n, sd, name = panels[i]
        ax = fig.add_subplot(2, 2, slot + 1, projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        lv = np.linspace(-vmax, vmax, 17)
        cf = ax.contourf(cube["lon"], cube["lat"], beta, levels=lv, cmap=cmap,
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
        cb.ax.tick_params(labelsize=8)
        ax.set_title(f"NDJFM {title}", fontsize=11, loc="left")
    fig.suptitle("Atlantic teleconnections to South American summer — "
                 "20CRv3 partial regressions, 1870/71–2014/15 "
                 "(trend and ENSO removed)\n"
                 "TSA = tropical South Atlantic (0–20°S, 30°W–10°E) · "
                 "TNA = tropical North Atlantic (5.5–23.5°N) · "
                 "stippled p<0.10", fontsize=12, x=0.03, ha="left")
    fig.text(0.012, 0.008,
             "Per +1 °C of the (detrended, ENSO-residualised) index. "
             "c3s context: models forecast TNA with ACC 0.6–0.8 and TSA "
             "moderately; TSA is forecast to reach ≈+1 °C by Jan 2027.",
             fontsize=8, color="#555")
    out = Path.home() / "brazil_atlantic_teleconnections.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
