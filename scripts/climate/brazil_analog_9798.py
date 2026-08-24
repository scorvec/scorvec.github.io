#!/usr/bin/env python3
"""
The 1997/98 analog, month by month — detrended 20CRv3 anomalies over South
America, Nov 1997 → Apr 1998 (April included: the event's spring transition —
TNA surge, Nordeste wet-season failure — is part of the analog's lesson).

Anomalies are departures from each gridcell's own linear trend for that
calendar month (1870–2014), i.e. the same detrending as the flavor composites,
so "warm/dry" means relative to the era, not the century mean.

Output: ~/analog_9798_monthly.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_flavor_composites import ndjfm_stack, DATA, EXTENT   # noqa: E402

MONTHS = [11, 12, 1, 2, 3, 4]
MNAME = {11: "Nov 1997", 12: "Dec 1997", 1: "Jan 1998", 2: "Feb 1998",
         3: "Mar 1998", 4: "Apr 1998"}


def main() -> int:
    fig = plt.figure(figsize=(11.5, 27), dpi=140)
    k = 0
    for m in MONTHS:
        T = ndjfm_stack(DATA / "air.2m.mon.mean.nc", "air", to="C",
                        months=(m,)).sel(summer=1997)
        P = ndjfm_stack(DATA / "apcp.mon.mean.nc", "apcp", to="mmday",
                        months=(m,)).sel(summer=1997)
        for field, unit, vmax, cmap, lab in (
                (T, "°C", 2.5, "RdBu_r", "temp"),
                (P, "mm/day", 5.0, "BrBG", "precip")):
            k += 1
            ax = fig.add_subplot(len(MONTHS), 2, k,
                                 projection=ccrs.PlateCarree())
            ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
            lv = np.linspace(-vmax, vmax, 17)
            cf = ax.contourf(field["lon"], field["lat"], field.values,
                             levels=lv, cmap=cmap, extend="both",
                             transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5,
                           edgecolor="#333")
            ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2,
                           edgecolor="#999")
            ax.coastlines("50m", lw=0.5, color="#333")
            cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(unit, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(f"{MNAME[m]} — {lab}", fontsize=10, loc="left")
    fig.suptitle("1997/98 — the joint analog, month by month\n"
                 "20CRv3 detrended anomalies (departure from each month's "
                 "1870–2014 gridcell trend)", fontsize=13, x=0.03, ha="left")
    out = Path.home() / "analog_9798_monthly.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
