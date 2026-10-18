#!/usr/bin/env python3
"""Tropical sea-level pressure against ERA5 1991-2020, daily: the Hadley cells' surface highs and troughs as anomalies in
units of the day's ERA5 standard deviation (normalised) and in hPa, over 40S-40N.

The analysis is the day's mean of the IFS HRES analyses at 00, 06, 12 and 18 UTC that the Hadley outflow pipeline
archives on the 1.5 deg grid (hadley_branch_maps.py, release hbranch-hres: the field msl of hb_YYYY-MM.npz), less the
analysis's mean offset from ERA5 at that time of year (offset_era5.npz), so that it stands on the footing of the ERA5
normals (build_slp_clim.py):

    normalised anomaly = (HRES - offset - normal) / sd,    anomaly = HRES - offset - normal

Every archived day is a frame of both loops (UNITS); a frame the runner holds (fetched from the frames branch) is not
drawn again. draw() is shared with the forecast loops (hadley_slp_fc.py).

    python src/hadley_slp.py render [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build_slp_clim as C
import hadley_branch_maps as M
from doycal import noleap_doy

EXTENT = (-40.0, 40.0)
CONTOUR = 4.0  # hPa between the sea-level pressure contours
# the two loops: colour levels fixed so that every frame (analysis and forecast) reads on one scale
UNITS = {
    "sigma": {
        "dir": "hslp_ana",
        "levels": [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3],
        "label": "anomaly in σ of ERA5's daily means",
        "name": "normalised anomaly",
    },
    "hpa": {
        "dir": "hslp_ana_hpa",
        "levels": [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 4],
        "label": "anomaly (hPa)",
        "name": "anomaly",
    },
}
ROWS = C.ROWS


def anomalies(msl: np.ndarray, N: C.Normals, day) -> dict:
    """{units: field} of a daily-mean sea-level pressure (hPa) on the domain, NaN over high ground."""
    a = np.where(N.high, np.nan, msl - N.normal(day))
    return {"sigma": a / N.sd(day), "hpa": a}


def draw(anom: np.ndarray, msl: np.ndarray, N: C.Normals, units: str, title: str, panel: str, foot: str, out: Path):
    """One strip map: colour = the anomaly on the units' fixed scale (blank where NaN), contours = the sea-level
    pressure every CONTOUR hPa, high ground grey."""
    u = UNITS[units]
    size, rects, left = M.layout(1, extent=EXTENT)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(rects[0][0], projection=M.PROJ)
    lat = C.LAT
    grey = matplotlib.colors.ListedColormap(["#d9d9d9"])
    ax.pcolormesh(M.XN, lat, M.native(np.where(N.high, 1.0, np.nan)), cmap=grey, shading="nearest")  # whole cells
    cf = ax.contourf(M.XN, lat, M.native(anom), levels=u["levels"], cmap="RdBu_r", extend="both")
    p = M.native(np.where(N.high, np.nan, msl))
    lev = np.arange(CONTOUR * np.floor(np.nanmin(p) / CONTOUR), np.nanmax(p) + CONTOUR, CONTOUR)
    cs = ax.contour(M.XN, lat, p, levels=lev, colors="#333", linewidths=0.55)
    ax.clabel(cs, levels=[x for x in lev if x % (2 * CONTOUR) == 0], fontsize=6.5, fmt="%d", inline_spacing=2)
    M.geo(ax, panel, extent=EXTENT)
    cb = fig.colorbar(cf, cax=fig.add_axes(rects[0][1]), orientation="vertical", ticks=u["levels"])
    cb.set_label(u["label"], fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    W, H = size
    fig.text(left / W, 1 - 0.3 / H, title, fontsize=13, fontweight="bold", va="center")
    fig.text(left / W, 0.12 / H, foot, fontsize=8.2, color="#555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def panel_title(units: str) -> str:
    return f"Sea-level pressure {UNITS[units]['name']} against ERA5 1991–2020 · contours every {CONTOUR:.0f} hPa"


def normals_note() -> str:
    return (
        f"normal and σ by day of year from ERA5's daily means 1991–2020 (harmonics, the normal with its trend at "
        f"{int(C.EPOCH)}) · grey = ground above {C.TERRAIN_M:.0f} m, where the pressure is a reduction"
    )


def render(rebuild: bool = False):
    """Both loops over every archived day."""
    off = M.Offsets()
    date, D = M.load()
    if not date.size:
        raise SystemExit("no archived days")
    N = C.Normals()
    loops = {k: {} for k in UNITS}
    for i, day in enumerate(date):
        doy = float(noleap_doy(pd.DatetimeIndex([day]))[0])
        msl = (D["msl"][i] - off.off2(doy)["msl"])[ROWS]
        A = anomalies(msl, N, day)
        for units, loop in loops.items():
            foot = (
                "the day's mean of the 00, 06, 12 and 18 UTC analyses (the tides cancel), less the analysis's mean "
                f"offset from ERA5\n{normals_note()}"
            )
            loop[f"{day:%Y-%m-%d}"] = (
                f"D{day:%Y%m%d}.webp",
                f"{day:%a %b %d, %Y}",
                partial(
                    draw,
                    A[units],
                    msl,
                    N,
                    units,
                    f"Tropical sea-level pressure — {day:%Y-%m-%d}, daily mean (IFS control analyses)",
                    panel_title(units),
                    foot,
                ),
            )
    for units, loop in loops.items():
        M.loop(UNITS[units]["dir"], f"Sea-level pressure {UNITS[units]['name']}, daily analyses", loop, rebuild)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--rebuild", action="store_true", help="redraw every frame (after a change to the normals)")
    a = ap.parse_args()
    render(a.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
