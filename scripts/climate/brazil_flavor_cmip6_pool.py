#!/usr/bin/env python3
"""
Pooled EP-flavor composite from the CMIP6 large ensembles that PASSED the
20CR entry exam (pattern correlation of their EP−regular South America
composite vs observed): EC-Earth3, MPI-ESM1-2-LR, MIROC6. CanESM5 and
ACCESS-ESM1-5 failed and are excluded.

Reads the per-model accumulators exported by brazil_flavor_cmip6.py
(cmip6_acc_<model>.npz: per-group sum, sum-of-squares, n on the model grid),
bilinearly interpolates each to a common 1° grid, then pools groups across
models by summing the interpolated sums — event-weighted, so MIROC6's 146 EP
events count accordingly. Welch t on the pooled group stats for stippling.

Output: ~/brazil_flavor_cmip6_pooled.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brazil_flavor_composites as bfc                       # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-shawn-scorvec-github-io/"
               "8564d971-7757-4a74-8f76-403c1520d16f/scratchpad")
PASSERS = ["EC-Earth3", "MPI-ESM1-2-LR", "MIROC6"]
EXT = bfc.EXTENT
LAT = np.arange(EXT[2] - 2, EXT[3] + 2.01, 1.0)
LON = np.arange(EXT[0] - 2, EXT[1] + 2.01, 1.0)


def to_common(lat, lon, arr):
    da = xr.DataArray(arr, coords=dict(lat=lat, lon=lon), dims=("lat", "lon"))
    return da.interp(lat=LAT, lon=LON).values


def main() -> int:
    pool = {g: {k: np.zeros((LAT.size, LON.size))
                for k in ("s_t", "q_t", "s_p", "q_p")} | {"n": 0}
            for g in ("EP", "CP", "REG")}
    per_model_n = {}
    for m in PASSERS:
        z = np.load(SCRATCH / f"cmip6_acc_{m}.npz")
        per_model_n[m] = {g: int(z[f"{g}_n"]) for g in ("EP", "CP", "REG")}
        for g in ("EP", "CP", "REG"):
            pool[g]["n"] += int(z[f"{g}_n"])
            for k in ("s_t", "q_t", "s_p", "q_p"):
                pool[g][k] += to_common(z["lat"], z["lon"], z[f"{g}_{k}"])
    print("pooled counts:", {g: pool[g]["n"] for g in pool})
    print("per-model:", per_model_n)

    def gstats(g, which):
        n = pool[g]["n"]
        mn = pool[g][f"s_{which}"] / n
        v = (pool[g][f"q_{which}"] / n - mn ** 2) * n / (n - 1)
        return mn, v, n

    def welch(m1, v1, n1, m2, v2, n2):
        se = np.sqrt(v1 / n1 + v2 / n2)
        tt = (m1 - m2) / np.maximum(se, 1e-12)
        dof = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        return 2 * st.t.sf(np.abs(tt), dof)

    # observed comparator for the pattern-r footer
    To = bfc.ndjfm_stack(bfc.DATA / "air.2m.mon.mean.nc", "air", to="C")
    Po = bfc.ndjfm_stack(bfc.DATA / "apcp.mon.mean.nc", "apcp", to="mmday")
    ers = json.loads((SCRATCH / "flavor_events.json").read_text())
    obs = {}
    for nm, cu in (("t", To), ("p", Po)):
        A = cu.sel(summer=[y for y in ers["EP"] if y in cu["summer"].values])
        B = cu.sel(summer=[y for y in ers["REG"] if y in cu["summer"].values])
        lo = cu["lon"].values
        d = (A.mean("summer") - B.mean("summer")).assign_coords(
            lon=np.where(lo > 180, lo - 360, lo)).sortby("lon")
        obs[nm] = d.interp(lat=LAT, lon=LON).values

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
            cf = ax.contourf(LON, LAT, d, levels=lv, cmap=cmap,
                             extend="both", transform=ccrs.PlateCarree())
            yy, xx = np.meshgrid(LAT, LON, indexing="ij")
            mm = p < 0.10
            ax.plot(xx[mm], yy[mm], ".", color="#222", ms=1.6, alpha=0.6,
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
                w = np.cos(np.deg2rad(LAT))[:, None]
                a1, a2 = d.ravel(), obs[which].ravel()
                ww = np.broadcast_to(w, d.shape).ravel()
                mf = np.isfinite(a1) & np.isfinite(a2)
                r = float(np.corrcoef(a1[mf] * np.sqrt(ww[mf]),
                                      a2[mf] * np.sqrt(ww[mf]))[0, 1])
                ax.text(0.02, 0.03, f"pattern r vs 20CR: {r:+.2f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="#999"))
                print(f"pooled pattern r ({which}) = {r:+.3f}")
    cnt = {g: pool[g]["n"] for g in pool}
    fig.suptitle("EP-flavor composites — POOLED CMIP6 large ensembles "
                 f"(exam passers: {', '.join(PASSERS)})\n"
                 f"EP n={cnt['EP']} · Modoki n={cnt['CP']} · regular "
                 f"n={cnt['REG']} · NDJFM, member-detrended · stippled Welch "
                 "p<0.10 · CanESM5 & ACCESS-ESM1-5 excluded (failed 20CR "
                 "pattern exam)", fontsize=12, x=0.03, ha="left")
    out = Path.home() / "brazil_flavor_cmip6_pooled.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
