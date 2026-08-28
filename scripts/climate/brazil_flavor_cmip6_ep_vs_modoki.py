#!/usr/bin/env python3
"""
EP vs Modoki, side by side — pooled CMIP6 exam passers (EC-Earth3,
MPI-ESM1-2-LR, MIROC6). For temperature and precipitation: the EP-event
composite anomaly, the Modoki-event composite anomaly, and the EP − Modoki
delta (Welch-stippled). Reads the cmip6_acc_<model>.npz accumulators.

Output: ~/brazil_flavor_cmip6_ep_vs_modoki.png
"""
from __future__ import annotations

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
            for g in ("EP", "CP")}
    for m in PASSERS:
        z = np.load(SCRATCH / f"cmip6_acc_{m}.npz")
        for g in ("EP", "CP"):
            pool[g]["n"] += int(z[f"{g}_n"])
            for k in ("s_t", "q_t", "s_p", "q_p"):
                pool[g][k] += to_common(z["lat"], z["lon"], z[f"{g}_{k}"])
    nE, nC = pool["EP"]["n"], pool["CP"]["n"]
    print(f"pooled EP n={nE}, Modoki n={nC}")

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

    fig = plt.figure(figsize=(17.5, 11.5), dpi=150)
    for row, (which, unit, vmax, cmap) in enumerate(
            (("t", "°C", 1.2, "RdBu_r"), ("p", "mm/day", 2.0, "BrBG"))):
        mE, vE, _ = gstats("EP", which)
        mC, vC, _ = gstats("CP", which)
        p = welch(mE, vE, nE, mC, vC, nC)
        panels = [(f"EP composite (n={nE})", mE, None),
                  (f"Modoki composite (n={nC})", mC, None),
                  ("EP − Modoki", mE - mC, p)]
        for col, (title, fld, pp) in enumerate(panels):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1,
                                 projection=ccrs.PlateCarree())
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            lv = np.linspace(-vmax, vmax, 17)
            cf = ax.contourf(LON, LAT, fld, levels=lv, cmap=cmap,
                             extend="both", transform=ccrs.PlateCarree())
            if pp is not None:
                yy, xx = np.meshgrid(LAT, LON, indexing="ij")
                mm = pp < 0.10
                ax.plot(xx[mm], yy[mm], ".", color="#222", ms=1.6, alpha=0.6,
                        transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                           edgecolor="#333")
            ax.coastlines("50m", lw=0.6, color="#333")
            cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(unit, fontsize=9)
            nmv = {"t": "Temperature", "p": "Precipitation"}[which]
            ax.set_title(f"{nmv}: {title}", fontsize=11, loc="left")
    fig.suptitle("EP vs Modoki El Niño, NDJFM anomalies — pooled CMIP6 large "
                 f"ensembles ({', '.join(PASSERS)})\n"
                 "member-detrended anomalies · delta stippled Welch p<0.10",
                 fontsize=12.5, x=0.03, ha="left")
    out = Path.home() / "brazil_flavor_cmip6_ep_vs_modoki.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
