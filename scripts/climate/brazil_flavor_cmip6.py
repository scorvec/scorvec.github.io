#!/usr/bin/env python3
"""
EP-flavor El Niño composites from a CMIP6 large ensemble (CanESM5, 50
historical members, 1850–2014) — the sample reality can't provide.

Per member: monthly tas-based relative Niño indices (same construction as the
20CR extension: box T minus 20S–20N tropical mean, 3-mo smoothed), NDJFM
means, linear+hinge detrended, STANDARDIZED WITHIN MEMBER, then classified
with z-thresholds matched to the observed event frequencies:
  event  z(RONI)  ≥ +1.1     (obs RONI ≥ 0.5 ≈ 1.1σ)
  EP     z(elean) ≥ +1.6     (obs +0.75 ≈ 1.6σ)
  CP     z(elean) ≤ −0.53    (obs −0.25 ≈ −0.53σ)
Composites of detrended SA tas/pr for EP−CP and EP−REG, Welch stippling, and
pattern correlation against the 20CR (observed-era) EP−regular maps.

Output: ~/brazil_flavor_cmip6.png + printed counts.
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

BASE = "cmip6/CMIP6/CMIP/CCCma/CanESM5/historical"
N_MEMBERS = 50
Y0M, Y1M = 1850, 2013
EXT = bfc.EXTENT                                             # SA window
ZEV, ZEP, ZCP = 1.1, 1.6, -0.53


def wmean(da, la="lat", lo="lon"):
    w = np.cos(np.deg2rad(da[la]))
    return da.weighted(w).mean((la, lo), skipna=True)


def rel_index(anom, la0, la1, lo0, lo1, trop):
    s = wmean(anom.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))) - trop
    return s.rolling(time=3, center=True, min_periods=2).mean().to_series()


def ndjfm_series(s: pd.Series):
    t = pd.DatetimeIndex(s.index)
    out = {}
    for y in range(Y0M, Y1M + 1):
        stamps = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                 [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
        v = [s.get(x, np.nan) for x in stamps]
        if np.isfinite(v).sum() >= 4:
            out[y] = float(np.nanmean(v))
    return pd.Series(out)


def detrend_z(s: pd.Series) -> pd.Series:
    x = s.index.to_numpy(float)
    A = np.column_stack([np.ones_like(x), x - 1950, np.clip(x - 1970, 0, None)])
    c, *_ = np.linalg.lstsq(A, s.values, rcond=None)
    d = s.values - A @ c
    return pd.Series(d / d.std(), index=s.index)


def member_fields(fs, mem, var):
    import gcsfs
    vdirs = fs.ls(f"{BASE}/{mem}/Amon/{var}/gn")
    ds = xr.open_zarr(gcsfs.GCSMap(vdirs[-1], gcs=fs), consolidated=True)
    da = ds[var]
    if "time" in da.dims:
        t = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, 1)
                              for x in da["time"].values])
        da = da.assign_coords(time=t)
    return da.sortby("lat")


def main() -> int:
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    mems = sorted({p.rsplit("/", 1)[-1] for p in fs.ls(BASE)})
    mems = [m for m in mems if m.endswith(("p1f1", "p2f1"))][:N_MEMBERS]
    print(f"{len(mems)} members")

    acc = {g: dict(n=0, s_t=None, q_t=None, s_p=None, q_p=None)
           for g in ("EP", "CP", "REG")}
    counts = {g: 0 for g in acc}
    n_done = 0
    for mem in mems:
        try:
            tas = member_fields(fs, mem, "tas")
            pr = member_fields(fs, mem, "pr")
            trop_band = tas.sel(lat=slice(-22, 22)).load()
            sa_t = tas.sel(lat=slice(EXT[2] - 3, EXT[3] + 3),
                           lon=slice(EXT[0] % 360, EXT[1] % 360)).load()
            sa_p = (pr.sel(lat=slice(EXT[2] - 3, EXT[3] + 3),
                           lon=slice(EXT[0] % 360, EXT[1] % 360)) * 86400).load()
        except Exception as e:                               # noqa: BLE001
            print(f"  {mem}: load failed ({repr(e)[:50]})")
            continue

        clim = trop_band.groupby("time.month").mean("time")
        anom = trop_band.groupby("time.month") - clim
        trop = wmean(anom.sel(lat=slice(-20, 20)))
        n34 = detrend_z(ndjfm_series(rel_index(anom, -5, 5, 190, 240, trop)))
        n12 = detrend_z(ndjfm_series(rel_index(anom, -10, 0, 270, 280, trop)))
        n4 = detrend_z(ndjfm_series(rel_index(anom, -5, 5, 160, 210, trop)))
        el = detrend_z(n12 - n4)

        # NDJFM SA cubes, detrended per cell
        def cube(da):
            c = da.groupby("time.month").mean("time")
            a = da.groupby("time.month") - c
            t = pd.DatetimeIndex(a["time"].values)
            stacks, yrs = [], []
            for y in range(Y0M, Y1M + 1):
                stamps = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                         [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
                idx = t.get_indexer(stamps)
                if (idx < 0).any():
                    continue
                stacks.append(a.isel(time=idx).mean("time"))
                yrs.append(y)
            cb = xr.concat(stacks, dim=pd.Index(yrs, name="summer"))
            yr = np.asarray(yrs, float)
            yc = yr - yr.mean()
            v = cb.values
            slope = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
            return cb.copy(data=v - v.mean(0)[None] - yc[:, None, None] * slope[None])

        ct, cp_ = cube(sa_t), cube(sa_p)
        ev = n34[n34 >= ZEV].index
        for y in ev:
            if y not in el.index or y not in ct["summer"].values:
                continue
            g = "EP" if el[y] >= ZEP else ("CP" if el[y] <= ZCP else "REG")
            counts[g] += 1
            ft = ct.sel(summer=y).values
            fp = cp_.sel(summer=y).values
            a = acc[g]
            if a["s_t"] is None:
                a["s_t"], a["q_t"] = ft.copy(), ft ** 2
                a["s_p"], a["q_p"] = fp.copy(), fp ** 2
            else:
                a["s_t"] += ft; a["q_t"] += ft ** 2
                a["s_p"] += fp; a["q_p"] += fp ** 2
            a["n"] += 1
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(mems)} members; counts {counts}", flush=True)
        grid_t, grid_p = ct, cp_                              # keep coords

    print(f"final counts: {counts}")

    def group_stats(g, which):
        a = acc[g]
        n = a["n"]
        m = a[f"s_{which}"] / n
        var = a[f"q_{which}"] / n - m ** 2
        return m, var * n / (n - 1), n

    def welch(m1, v1, n1, m2, v2, n2):
        from scipy import stats as st
        se = np.sqrt(v1 / n1 + v2 / n2)
        tstat = (m1 - m2) / np.maximum(se, 1e-12)
        dof = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        return 2 * st.t.sf(np.abs(tstat), dof)

    # observed (20CR) EP−regular for pattern correlation
    To = bfc.ndjfm_stack(bfc.DATA / "air.2m.mon.mean.nc", "air", to="C")
    Po = bfc.ndjfm_stack(bfc.DATA / "apcp.mon.mean.nc", "apcp", to="mmday")
    import json
    ers = json.loads(Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
                          "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad/"
                          "flavor_events.json").read_text())
    obs_d = {}
    for name, cube_o in (("t", To), ("p", Po)):
        A = cube_o.sel(summer=[y for y in ers["EP"] if y in cube_o["summer"].values])
        B = cube_o.sel(summer=[y for y in ers["REG"] if y in cube_o["summer"].values])
        obs_d[name] = A.mean("summer") - B.mean("summer")

    fig = plt.figure(figsize=(13.5, 12.5), dpi=150)
    slot = 0
    patcorr = {}
    for which, unit, vmax, cmap, grid, ob in (
            ("t", "°C", 1.2, "RdBu_r", grid_t, obs_d["t"]),
            ("p", "mm/day", 2.0, "BrBG", grid_p, obs_d["p"])):
        mE, vE, nE = group_stats("EP", which)
        for other, tag in (("CP", "EP − Modoki"), ("REG", "EP − regular")):
            mO, vO, nO = group_stats(other, which)
            d = mE - mO
            p = welch(mE, vE, nE, mO, vO, nO)
            slot += 1
            ax = fig.add_subplot(2, 2, slot if which == "t" else slot,
                                 projection=ccrs.PlateCarree())
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            lons = grid["lon"].values
            lons180 = np.where(lons > 180, lons - 360, lons)
            lv = np.linspace(-vmax, vmax, 17)
            cf = ax.contourf(lons180, grid["lat"], d, levels=lv, cmap=cmap,
                             extend="both", transform=ccrs.PlateCarree())
            yy, xx = np.meshgrid(grid["lat"], lons180, indexing="ij")
            mm = p < 0.10
            ax.plot(xx[mm], yy[mm], ".", color="#222", ms=1.4, alpha=0.6,
                    transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                           edgecolor="#333")
            ax.coastlines("50m", lw=0.6, color="#333")
            cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(unit, fontsize=9)
            nm = {"t": "Temperature", "p": "Precipitation"}[which]
            ax.set_title(f"{nm}: {tag} (n={nE} vs {nO})", fontsize=11,
                         loc="left")
            if other == "REG":
                # pattern correlation vs 20CR on the model grid
                obi = ob.interp(lat=grid["lat"].values, lon=lons180)
                w = np.cos(np.deg2rad(grid["lat"].values))[:, None]
                a1, a2 = d.ravel(), np.asarray(obi.values).ravel()
                ww = np.broadcast_to(w, d.shape).ravel()
                mfin = np.isfinite(a1) & np.isfinite(a2)
                r = np.corrcoef(a1[mfin] * np.sqrt(ww[mfin]),
                                a2[mfin] * np.sqrt(ww[mfin]))[0, 1]
                patcorr[which] = r
                ax.text(0.02, 0.03, f"pattern r vs 20CR: {r:+.2f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="#999"))
    fig.suptitle(f"EP-flavor composites from the CanESM5 large ensemble — "
                 f"{len(mems)} members × 164 summers\n"
                 f"EP n={acc['EP']['n']} · Modoki n={acc['CP']['n']} · "
                 f"regular n={acc['REG']['n']} — member-standardized indices, "
                 "thresholds matched to observed frequencies · stippled p<0.10",
                 fontsize=12, x=0.03, ha="left")
    out = Path.home() / "brazil_flavor_cmip6.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"pattern correlations vs 20CR EP−regular: {patcorr}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
