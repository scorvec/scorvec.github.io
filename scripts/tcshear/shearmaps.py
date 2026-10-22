#!/usr/bin/env python3
"""Where the shear comes from: 330 K potential vorticity with the 200-850 hPa shear vectors around a storm, IFS control
(HRES until May 2026), 6-hourly to day 5: from its 0.25-degree run, then redrawn from its 0.1-degree run once that is
out (tcshear.redraw_maps); the smoothing and the arrow spacing are in degrees, so both grids draw alike. PV from the models page's code (scripts/models/upper.py via shearlib.pv330) and the same PV
colours as the dynamic-tropopause charts (scripts/mjo/src/dt_charts.py), so the three products read alike.

Each frame: 330 K Ertel PV (tropospheric air warm, the 2-PVU line black, stratospheric air blue: the upstream trough),
the 200-850 hPa shear of the TOTAL wind smoothed over ~300 km (grey arrows: the large-scale shear field), the storm's
environmental (vortex-removed) shear vector at its centre (red arrow), the run's track, and a dashed line to the
nearest 2-PVU air on 330 K outside 400 km (the trough's distance).
Southern Hemisphere storms: PV is negative there, so the map shows -PV (stratospheric air is PV <= -2 PVU) and says so.
Longitudes are unwrapped around the map's centre so a track across the dateline is drawn the short way.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np

import shearlib as S

PV_LEV = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12]
PV_COLS = [
    "#fbe9b0",
    "#f8d27e",
    "#f2ad57",
    "#e5803d",
    "#b9d8ee",
    "#86b9e3",
    "#5a96d3",
    "#3e74c0",
    "#3654a6",
    "#3c3a8c",
    "#4a2675",
    "#5b1560",
]
TEXT = "#1b1d22"
MUTED = "#6f6b64"


def _shear_field(box, sigma_deg=1.25):
    """200-850 hPa shear of the total wind, Gaussian-smoothed by sigma_deg (~300 km across) whatever the grid."""
    from scipy.ndimage import gaussian_filter

    li = list(S.LEVELS)
    du = np.asarray(box["u"][li.index(200)], float) - np.asarray(box["u"][li.index(850)], float)
    dv = np.asarray(box["v"][li.index(200)], float) - np.asarray(box["v"][li.index(850)], float)
    n = sigma_deg / abs(float(box["lat"][1] - box["lat"][0]))
    return gaussian_filter(np.nan_to_num(du), n), gaussian_filter(np.nan_to_num(dv), n)


def render(storm, boxes, recs, init, out_dir: Path, obs=None, grid="0.25°"):
    """boxes: {step: map box (IFS control)}; recs: that run's records (shearlib.analyze + 'h'); grid: the boxes' grid,
    for the title. -> [file names]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, ListedColormap

    import tclib as TC

    sid = storm["sid"]
    for f in out_dir.glob(f"{sid}_F*.webp"):
        f.unlink()
    a0, a1, b0, b1 = storm["mapbox"]
    asp = TC.sector_aspect(b0, b1, a0, a1)
    W = 10.0
    mh = W * 0.98 * asp
    H = mh + 1.55
    cmap = ListedColormap(PV_COLS)
    norm = BoundaryNorm(PV_LEV, len(PV_COLS), extend="max")
    sgn = S.hemi(recs[0]["lat"] if recs else (a0 + a1) / 2.0)  # PV x sign(lat): stratospheric air >= 2 either side
    mid = (b0 + b1) / 2.0

    def ulon(x):  # unwrap around the map centre (dateline)
        return float(mid + ((x - mid + 180.0) % 360.0 - 180.0))

    byh = {r["h"]: r for r in recs}
    trk = [(r["h"], r["lat"], ulon(r["lon"])) for r in recs]
    files = []
    for step in sorted(boxes):
        b = boxes[step]
        try:
            pv, lat, lon = S.pv330(b)
        except Exception as e:  # noqa: BLE001
            print(f"  map +{step}h: {e}")
            continue
        pv = sgn * pv
        du, dv = _shear_field(b)
        fig = plt.figure(figsize=(W, H))
        ax = TC.sector_axes(fig, [0.01, 1.0 / H, 0.98, mh / H], b0, b1, a0, a1, land=False)
        LO, LA = np.meshgrid(lon, lat)
        m = ax.pcolormesh(
            LO,
            LA,
            np.ma.masked_invalid(pv),
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="auto",
            rasterized=True,
        )
        ax.contour(LO, LA, np.nan_to_num(pv), levels=[2.0], colors="#000", linewidths=1.3, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#555")
        k = max(1, round(2.0 / abs(float(lat[1] - lat[0]))))  # an arrow every 2 degrees
        ax.quiver(
            LO[::k, ::k],
            LA[::k, ::k],
            du[::k, ::k],
            dv[::k, ::k],
            transform=ccrs.PlateCarree(),
            color="#3a3a3a",
            scale=900,
            width=0.0022,
            headwidth=3.5,
            alpha=0.85,
            regrid_shape=None,
        )
        past = [(la, lo) for h_, la, lo in trk if h_ <= step]
        fut = [(la, lo) for h_, la, lo in trk if h_ >= step]
        if len(past) > 1:
            ax.plot([p[1] for p in past], [p[0] for p in past], color="#111", lw=1.6, transform=ccrs.PlateCarree())
        if len(fut) > 1:
            ax.plot(
                [p[1] for p in fut],
                [p[0] for p in fut],
                color="#111",
                lw=1.2,
                ls=(0, (3, 2)),
                transform=ccrs.PlateCarree(),
            )
        r = byh.get(step)
        note = ""
        if r:
            rlon = ulon(r["lon"])
            ax.plot(
                rlon,
                r["lat"],
                marker="o",
                ms=9,
                mfc="#d1352b",
                mec="#fff",
                mew=1.5,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
            if r.get("deep"):
                mag, frm = r["deep"]
                to = math.radians((frm + 180) % 360)
                L = mag / 10.0 * 2.2  # degrees of arrow per 10 m/s
                ax.annotate(
                    "",
                    xy=(rlon + L * math.sin(to) / math.cos(math.radians(r["lat"])), r["lat"] + L * math.cos(to)),
                    xytext=(rlon, r["lat"]),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="-|>", color="#d1352b", lw=2.6, mutation_scale=18),
                    zorder=9,
                )
                note = f"Run {init:%d %b %H}Z.  Environmental shear {mag:.0f} m/s ({mag * S.KT:.0f} kt) from {frm:.0f}°"
            if r.get("d2pvu") is not None and r.get("b2pvu") is not None:
                la2, lo2 = S.step_km(
                    r["lat"],
                    r["lon"],
                    r["d2pvu"] * math.sin(math.radians(r["b2pvu"])),
                    r["d2pvu"] * math.cos(math.radians(r["b2pvu"])),
                )
                ax.plot(
                    [rlon, ulon(lo2)],
                    [r["lat"], la2],
                    color="#1f3f8f",
                    lw=1.4,
                    ls=(0, (4, 2)),
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                )
                note += f";  2-PVU air on 330 K {r['d2pvu']:,} km to the {_compass(r['b2pvu'])}"
        valid = init + dt.timedelta(hours=step)
        top = 1 - 0.12 / H
        title = fig.text(
            0.012,
            top,
            f"{storm['name']} — 330 K PV and 200–850 hPa shear, IFS control {grid}, "
            f"+{step} h (valid {valid:%a %d %b %H}Z)",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT,
            va="top",
        )
        TC.fit_text(fig, title, W * (1 - 2 * 0.012))
        fig.text(0.012, top - 0.3 / H, note or " ", fontsize=10, color=TEXT, va="top")
        cax = fig.add_axes([0.04, 0.42 / H, 0.44, 0.17 / H])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=PV_LEV)
        cb.set_label(
            "PV on 330 K (PVU); black line = 2 PVU"
            if sgn > 0
            else "−PV on 330 K (PVU; SH: PV < 0); black line = −2 PVU",
            fontsize=9,
            color=TEXT,
        )
        cb.ax.tick_params(labelsize=8)
        fig.text(
            0.52,
            0.55 / H,
            "grey arrows: 200–850 hPa shear of the total wind, smoothed ~300 km\n"
            "red arrow: the storm's vortex-removed shear (10 m/s per 2.2°)\n"
            "dashed blue: nearest 2-PVU air on 330 K beyond 400 km (smoothed PV)",
            fontsize=8.6,
            color=MUTED,
            va="center",
        )
        name = f"{sid}_F{step:03d}.webp"
        fig.savefig(out_dir / name, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
        plt.close(fig)
        files.append(name)
    return files


def _compass(b):
    return [
        "north",
        "north-northeast",
        "northeast",
        "east-northeast",
        "east",
        "east-southeast",
        "southeast",
        "south-southeast",
        "south",
        "south-southwest",
        "southwest",
        "west-southwest",
        "west",
        "west-northwest",
        "northwest",
        "north-northwest",
    ][int(((b % 360) + 11.25) // 22.5) % 16]
