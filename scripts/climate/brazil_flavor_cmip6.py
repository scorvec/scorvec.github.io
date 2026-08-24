#!/usr/bin/env python3
"""
EP-flavor El Niño mining across CMIP6 large ensembles — with an entry exam.

For each model: every historical member 1850–2014 is searched for El Niño
events using member-standardized relative indices (thresholds matched to
observed frequencies: event z(RONI) ≥ 1.1, EP z(east-lean) ≥ 1.6, CP ≤ −0.53).
The model's EP−regular South America composite is then validated against the
20CR observed one by area-weighted pattern correlation. Models that fail the
exam (CanESM5: r −0.11/+0.01, EP:CP ratio 1:20) must not be pooled.

    python brazil_flavor_cmip6.py --model MIROC6
    python brazil_flavor_cmip6.py --screen          # all models in MODELS

Outputs: ~/brazil_flavor_cmip6_<model>.png per model +
         scratchpad/cmip6_flavor_screen.json accumulating the scoreboard.
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brazil_flavor_composites as bfc                       # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
               "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad")
SCORE = SCRATCH / "cmip6_flavor_screen.json"

MODELS = {
    "CanESM5": ("CCCma", "gn", 50),
    "MIROC6": ("MIROC", "gn", 50),
    "ACCESS-ESM1-5": ("CSIRO", "gn", 40),
    "MPI-ESM1-2-LR": ("MPI-M", "gn", 10),
    "EC-Earth3": ("EC-Earth-Consortium", "gr", 20),
}
Y0M, Y1M = 1850, 2013
EXT = bfc.EXTENT
ZEV, ZEP, ZCP = 1.1, 1.6, -0.53


def wmean(da):
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"), skipna=True)


def rel_index(anom, la0, la1, lo0, lo1, trop):
    s = wmean(anom.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))) - trop
    return s.rolling(time=3, center=True, min_periods=2).mean().to_series()


def ndjfm_series(s: pd.Series):
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


def run_model(model: str) -> dict:
    inst, grid, nmax = MODELS[model]
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    base = f"cmip6/CMIP6/CMIP/{inst}/{model}/historical"
    mems = sorted({p.rsplit("/", 1)[-1] for p in fs.ls(base)})
    mems = [m for m in mems if m.endswith(("i1p1f1", "i1p2f1"))][:nmax]
    print(f"{model}: {len(mems)} members", flush=True)

    def fields(mem, var):
        import gcsfs as _g
        vdirs = fs.ls(f"{base}/{mem}/Amon/{var}/{grid}")
        ds = xr.open_zarr(_g.GCSMap(vdirs[-1], gcs=fs), consolidated=True)
        da = ds[var]
        tv = da["time"].values
        if np.issubdtype(np.asarray(tv).dtype, np.datetime64):
            t = pd.DatetimeIndex(tv).to_period("M").to_timestamp()
        else:                                    # cftime objects
            t = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, 1)
                                  for x in tv])
        return da.assign_coords(time=t).sortby("lat")

    acc = {g: dict(n=0, s_t=None, q_t=None, s_p=None, q_p=None)
           for g in ("EP", "CP", "REG")}
    grid_t = None
    for k, mem in enumerate(mems):
        try:
            tas = fields(mem, "tas")
            pr = fields(mem, "pr")
            trop_band = tas.sel(lat=slice(-22, 22)).load()
            sa_t = tas.sel(lat=slice(EXT[2] - 3, EXT[3] + 3),
                           lon=slice(EXT[0] % 360, EXT[1] % 360)).load()
            sa_p = (pr.sel(lat=slice(EXT[2] - 3, EXT[3] + 3),
                           lon=slice(EXT[0] % 360, EXT[1] % 360)) * 86400).load()
        except Exception as e:                               # noqa: BLE001
            print(f"  {mem}: load failed ({repr(e)[:50]})", flush=True)
            continue
        clim = trop_band.groupby("time.month").mean("time")
        anom = trop_band.groupby("time.month") - clim
        trop = wmean(anom.sel(lat=slice(-20, 20)))
        n34 = detrend_z(ndjfm_series(rel_index(anom, -5, 5, 190, 240, trop)))
        n12 = detrend_z(ndjfm_series(rel_index(anom, -10, 0, 270, 280, trop)))
        n4 = detrend_z(ndjfm_series(rel_index(anom, -5, 5, 160, 210, trop)))
        el = detrend_z(n12 - n4)

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
            sl = np.tensordot(yc, v - v.mean(0), axes=(0, 0)) / (yc @ yc)
            return cb.copy(data=v - v.mean(0)[None] - yc[:, None, None] * sl[None])

        ct, cp_ = cube(sa_t), cube(sa_p)
        grid_t = ct
        for y in n34[n34 >= ZEV].index:
            if y not in el.index or y not in ct["summer"].values:
                continue
            g = "EP" if el[y] >= ZEP else ("CP" if el[y] <= ZCP else "REG")
            ft, fp = ct.sel(summer=y).values, cp_.sel(summer=y).values
            a = acc[g]
            if a["s_t"] is None:
                a["s_t"], a["q_t"] = ft.copy(), ft ** 2
                a["s_p"], a["q_p"] = fp.copy(), fp ** 2
            else:
                a["s_t"] += ft; a["q_t"] += ft ** 2
                a["s_p"] += fp; a["q_p"] += fp ** 2
            a["n"] += 1
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(mems)}; counts "
                  f"{ {g: acc[g]['n'] for g in acc} }", flush=True)

    counts = {g: acc[g]["n"] for g in acc}
    print(f"{model} final: {counts}", flush=True)
    lons_ = grid_t["lon"].values if grid_t is not None else None
    if grid_t is not None:
        np.savez(SCRATCH / f"cmip6_acc_{model}.npz",
                 lat=grid_t["lat"].values,
                 lon=np.where(lons_ > 180, lons_ - 360, lons_),
                 **{f"{g}_{k}": np.asarray(acc[g][k]) for g in acc
                    for k in ("s_t", "q_t", "s_p", "q_p")
                    if acc[g][k] is not None},
                 **{f"{g}_n": acc[g]["n"] for g in acc})
    if min(counts["EP"], counts["REG"]) < 5:
        print(f"{model}: too few events to composite")
        return dict(model=model, counts=counts, r_t=None, r_p=None)

    def gstats(g, which):
        a = acc[g]; n = a["n"]
        m = a[f"s_{which}"] / n
        v = (a[f"q_{which}"] / n - m ** 2) * n / (n - 1)
        return m, v, n

    def welch(m1, v1, n1, m2, v2, n2):
        from scipy import stats as st
        se = np.sqrt(v1 / n1 + v2 / n2)
        tt = (m1 - m2) / np.maximum(se, 1e-12)
        dof = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        return 2 * st.t.sf(np.abs(tt), dof)

    # observed comparator
    To = bfc.ndjfm_stack(bfc.DATA / "air.2m.mon.mean.nc", "air", to="C")
    Po = bfc.ndjfm_stack(bfc.DATA / "apcp.mon.mean.nc", "apcp", to="mmday")
    ers = json.loads((SCRATCH / "flavor_events.json").read_text())
    obs = {}
    for nm, cu in (("t", To), ("p", Po)):
        A = cu.sel(summer=[y for y in ers["EP"] if y in cu["summer"].values])
        B = cu.sel(summer=[y for y in ers["REG"] if y in cu["summer"].values])
        obs[nm] = A.mean("summer") - B.mean("summer")

    lons = grid_t["lon"].values
    lons180 = np.where(lons > 180, lons - 360, lons)
    rvals = {}
    fig = plt.figure(figsize=(13.5, 12.5), dpi=150)
    slot = 0
    for which, unit, vmax, cmap in (("t", "°C", 1.2, "RdBu_r"),
                                    ("p", "mm/day", 2.0, "BrBG")):
        mE, vE, nE = gstats("EP", which)
        for other, tag in (("CP", "EP − Modoki"), ("REG", "EP − regular")):
            mO, vO, nO = gstats(other, which)
            d = mE - mO
            p = welch(mE, vE, nE, mO, vO, nO)
            slot += 1
            ax = fig.add_subplot(2, 2, slot, projection=ccrs.PlateCarree())
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            lv = np.linspace(-vmax, vmax, 17)
            cf = ax.contourf(lons180, grid_t["lat"], d, levels=lv, cmap=cmap,
                             extend="both", transform=ccrs.PlateCarree())
            yy, xx = np.meshgrid(grid_t["lat"], lons180, indexing="ij")
            mm = p < 0.10
            ax.plot(xx[mm], yy[mm], ".", color="#222", ms=1.4, alpha=0.6,
                    transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                           edgecolor="#333")
            ax.coastlines("50m", lw=0.6, color="#333")
            cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(unit, fontsize=9)
            nmv = {"t": "Temperature", "p": "Precipitation"}[which]
            ax.set_title(f"{nmv}: {tag} (n={nE} vs {nO})", fontsize=11,
                         loc="left")
            if other == "REG":
                obi = obs[which].interp(lat=grid_t["lat"].values, lon=lons180)
                w = np.cos(np.deg2rad(grid_t["lat"].values))[:, None]
                a1, a2 = d.ravel(), np.asarray(obi.values).ravel()
                ww = np.broadcast_to(w, d.shape).ravel()
                mf = np.isfinite(a1) & np.isfinite(a2)
                r = float(np.corrcoef(a1[mf] * np.sqrt(ww[mf]),
                                      a2[mf] * np.sqrt(ww[mf]))[0, 1])
                rvals[which] = r
                ax.text(0.02, 0.03, f"pattern r vs 20CR: {r:+.2f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="#999"))
    fig.suptitle(f"EP-flavor composites — {model} ({len(mems)} members)\n"
                 f"EP n={counts['EP']} · Modoki n={counts['CP']} · regular "
                 f"n={counts['REG']} · stippled p<0.10", fontsize=12,
                 x=0.03, ha="left")
    out = Path.home() / f"brazil_flavor_cmip6_{model}.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"{model}: pattern r_t={rvals.get('t'):+.2f} "
          f"r_p={rvals.get('p'):+.2f} → {out}", flush=True)
    return dict(model=model, counts=counts,
                r_t=round(rvals.get("t", np.nan), 3),
                r_p=round(rvals.get("p", np.nan), 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    todo = list(MODELS) if args.screen else [args.model or "CanESM5"]
    board = json.loads(SCORE.read_text()) if SCORE.exists() else {}
    for m in todo:
        if not args.force and m in board and board[m].get("r_p") is not None:
            print(f"{m}: already scored {board[m]}")
            continue
        board[m] = run_model(m)
        SCORE.write_text(json.dumps(board, indent=1))
    print("\nSCOREBOARD:")
    for m, s in board.items():
        print(f"  {m:15s} {s['counts']}  r_t={s.get('r_t')}  r_p={s.get('r_p')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
