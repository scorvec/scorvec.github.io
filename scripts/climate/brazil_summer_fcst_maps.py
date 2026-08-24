#!/usr/bin/env python3
"""
Brazil summer (Nov 2026 – Mar 2027) monthly temperature forecast maps,
expressed against 30-, 10- and 5-year trailing normals.

Construction, per calendar month and grid cell:
  forecast = ERA5 hinge-1970 trend projected to 2026/27
           + EP fingerprint · amplitude scaling
  map value = forecast − trailing-normal mean (last 30/10/5 occurrences)

EP fingerprint = 50/50 blend of
  · observed: 20CRv3 monthly composite of the 5 historical EP events
    (1876, 1877, 1896, 1982, 1997), per-cell detrended
  · model: pooled monthly EP composites from the CMIP6 exam passers
    (EC-Earth3, MPI-ESM1-2-LR, MIROC6; EP-count-weighted)
each scaled linearly from its own mean event amplitude to the assumed
current-event NDJFM RONI of +2.75 °C.

Outputs: ~/brazil_summer_fcst_vs{30,10,5}yr.png
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brazil_flavor_composites as bfc                       # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
               "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad")
ERA5 = HERE.parent / "sst" / "data" / "era5_t2m_mon.nc"
ERSST = HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc"
PASSERS = ["EC-Earth3", "MPI-ESM1-2-LR", "MIROC6"]
EP_OBS = [1876, 1877, 1896, 1982, 1997]
RONI_FC = 2.75
MONTHS = [11, 12, 1, 2, 3]
MNAME = {11: "Nov 2026", 12: "Dec 2026", 1: "Jan 2027", 2: "Feb 2027",
         3: "Mar 2027"}
EXT = (-76, -32, -35, 7)                       # Brazil-focused window
TARGET_YEAR = 2026                             # event year (NDJFM 2026/27)


def wm(da):
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"), skipna=True)


def obs_amplitude() -> float:
    """Mean NDJFM relative-Niño-3.4 of the 5 observed EP events (ERSST)."""
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim
    trop = wm(anom.sel(lat=slice(-20, 20))).to_series()
    s = wm(anom.sel(lat=slice(-5, 5), lon=slice(190, 240))).to_series() - trop
    amps = []
    for y in EP_OBS:
        stamps = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                 [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
        amps.append(np.nanmean([s.get(x, np.nan) for x in stamps]))
    print("obs EP amplitudes:", {y: round(a, 2) for y, a in zip(EP_OBS, amps)})
    return float(np.nanmean(amps))


def obs_monthly_fingerprint():
    """20CR monthly EP composite (detrended °C), dict month → DataArray."""
    ds = xr.open_dataset(bfc.DATA / "air.2m.mon.mean.nc")
    air = ds["air"].sortby("lat")
    sa = air.sel(lat=slice(EXT[2] - 4, EXT[3] + 4),
                 lon=slice((EXT[0] - 4) % 360, (EXT[1] + 4) % 360)).load()
    clim = sa.groupby("time.month").mean("time")
    an = sa.groupby("time.month") - clim
    t = pd.DatetimeIndex(an["time"].values)
    out = {}
    for m in MONTHS:
        sel = an.isel(time=(t.month == m))
        yrs = (pd.DatetimeIndex(sel["time"].values).year
               - (1 if m <= 3 else 0)).to_numpy(float)
        keep = (yrs >= 1870) & (yrs <= 2014)
        sel, yrs = sel.isel(time=keep), yrs[keep]
        yc = yrs - yrs.mean()
        v = sel.values
        sl = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
        dv = v - v.mean(0)[None] - yc[:, None, None] * sl[None]
        idx = [int(np.where(yrs == y)[0][0]) for y in EP_OBS]
        comp = dv[idx].mean(0)
        lo = sel["lon"].values
        out[m] = xr.DataArray(
            comp, coords=dict(lat=sel["lat"].values,
                              lon=np.where(lo > 180, lo - 360, lo)),
            dims=("lat", "lon"))
    return out


def model_monthly_fingerprint(lat_t, lon_t):
    """Pooled, per-model-amplitude-scaled CMIP6 EP composite per month,
    each model interpolated to the target grid before weighting."""
    fps, ns = {m: [] for m in MONTHS}, []
    for mdl in PASSERS:
        z = np.load(SCRATCH / f"cmip6_ep_monthly_{mdl}.npz")
        n, amp = int(z["n_ep"]), float(z["amp_sum"]) / int(z["n_ep"])
        print(f"{mdl}: EP n={n}, mean amp {amp:+.2f} °C, "
              f"scale ×{RONI_FC/amp:.2f}")
        ns.append(n)
        for m in MONTHS:
            da = xr.DataArray(z[f"m{m}"] / n * (RONI_FC / amp),
                              coords=dict(lat=z["lat"], lon=z["lon"]),
                              dims=("lat", "lon")).sortby("lon")
            fps[m].append(da.interp(lat=lat_t, lon=lon_t).values)
    w = np.array(ns, float) / sum(ns)
    return {m: sum(fp * wi for fp, wi in zip(fps[m], w)) for m in MONTHS}


def main() -> int:
    # ── ERA5 base: per-month hinge trend projection + trailing normals ──
    ds = xr.open_dataset(ERA5)
    t2 = ds["t2m"].sortby("lat")
    t2 = t2.assign_coords(lon=(((t2["lon"] + 180) % 360) - 180)).sortby("lon")
    t2 = t2.sel(lat=slice(EXT[2] - 4, EXT[3] + 4),
                lon=slice(EXT[0] - 4, EXT[1] + 4)).load()
    t = pd.DatetimeIndex(t2["time"].values)

    proj, normals = {}, {30: {}, 10: {}, 5: {}}
    for m in MONTHS:
        sel = t2.isel(time=(t.month == m))
        yrs = (pd.DatetimeIndex(sel["time"].values).year
               - (1 if m <= 3 else 0)).to_numpy(float)
        x = yrs
        A = np.column_stack([np.ones_like(x), x - 2000,
                             np.clip(x - 1970, 0, None)])
        v = sel.values.reshape(len(x), -1)
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        xt = np.array([1.0, TARGET_YEAR - 2000, TARGET_YEAR - 1970])
        proj[m] = (xt @ c).reshape(sel.shape[1:])
        for nb in (30, 10, 5):
            normals[nb][m] = sel.isel(time=slice(-nb, None)).mean("time").values
            lasty = int(yrs[-1])
            assert lasty == 2025, f"month {m} last event-year {lasty}"

    # ── fingerprints, blended on the ERA5 grid ──
    amp_obs = obs_amplitude()
    print(f"obs mean amp {amp_obs:+.2f} °C, scale ×{RONI_FC/amp_obs:.2f}")
    obs_fp = obs_monthly_fingerprint()
    lat_e, lon_e = t2["lat"].values, t2["lon"].values
    mdl_fp = model_monthly_fingerprint(lat_e, lon_e)
    fp = {}
    for m in MONTHS:
        o = obs_fp[m].interp(lat=lat_e, lon=lon_e).values * (RONI_FC / amp_obs)
        fp[m] = 0.5 * o + 0.5 * mdl_fp[m]

    # ── figures ──
    for nb in (30, 10, 5):
        fig = plt.figure(figsize=(15.5, 11.5), dpi=150)
        fields = {}
        for i, m in enumerate(MONTHS):
            fields[m] = proj[m] + fp[m] - normals[nb][m]
        fields["NDJFM"] = np.mean([fields[m] for m in MONTHS], axis=0)
        for i, key in enumerate(MONTHS + ["NDJFM"]):
            ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            lv = np.linspace(-3.0, 3.0, 25)
            cf = ax.contourf(lon_e, lat_e, fields[key], levels=lv,
                             cmap="RdBu_r", extend="both",
                             transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                           edgecolor="#333")
            ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25,
                           edgecolor="#888")
            ax.coastlines("50m", lw=0.6, color="#333")
            cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("°C", fontsize=9)
            ttl = MNAME.get(key, "NDJFM mean")
            ax.set_title(ttl, fontsize=12, loc="left")
        fig.suptitle(
            f"Brazil 2026/27 summer temperature forecast — anomaly vs "
            f"trailing {nb}-year normal\n"
            "ERA5 hinge-1970 trend + EP-El Niño fingerprint scaled to "
            f"RONI +{RONI_FC:.2f} (50% 20CR obs n=5 · 50% CMIP6 passers "
            "n≈200)", fontsize=13, x=0.03, ha="left")
        fig.text(0.012, 0.008,
                 "Fingerprint amplitude-scaled linearly from each source's "
                 "mean event; normals = last N occurrences of each month in "
                 "ERA5 (through 2025/26). Assumes RONI +2.75 NDJFM.",
                 fontsize=8, color="#555")
        out = Path.home() / f"brazil_summer_fcst_vs{nb}yr.png"
        fig.savefig(out, facecolor="white", bbox_inches="tight",
                    pad_inches=0.15)
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
