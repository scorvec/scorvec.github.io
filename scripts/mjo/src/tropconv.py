#!/usr/bin/env python3
"""Low-level convergence and monsoon troughs of the tropics: the 00Z IFS-ENS, a day at a time to day 14.

Every member's 925 and 850 hPa wind as 1-degree box means over 40S-40N, each day the mean of its 00 and 12 UTC states
(half a day apart, they cancel the daily cycle of the low-level wind). Per member: the 925 hPa convergence and the
850 hPa relative vorticity on the sphere, smoothed with a Gaussian of SIGMA degrees. Kept per lead day (day): the
members' mean wind and convergence, and how many members converge and how many turn cyclonically at each box.

The maps (render) colour the mean convergence only where the members' signs differ from a coin toss (two-sided sign
test, false discovery rate FDR over the map), draw the mean 925 hPa wind as streamlines and a trough axis where the
mean 850 hPa zonal wind turns from westerly on the equator's side to easterly on the pole's, kept where it falls by
AXIS_SHEAR towards the pole and the members agree that the 850 hPa flow turns cyclonically. A box is left grey where
more than BELOW of its smoothing weight lies under the ground (below_ground, from the surface pressure of the run's
first state).

    python src/hadley_members.py --model ifs --date 20261111 --out runs --products conv   # -> runs/tropconv_ifs.npz
    python src/tropconv.py runs/tropconv_ifs.npz        # the four sectors' loops and their manifest
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import overturn as O

MODEL = "ifs"  # the ensemble drawn (overturn.MODELS)
LEVELS = (850, 925)  # hPa: the trough axis, the convergence
LEADS = tuple(range(15))  # lead days, 00 and 12 UTC of each: the run ends at 00 UTC of day 15
EDGE = 40.0  # the reduction's last box centres; the maps stop at BAND, clear of the one-sided derivative there
BAND = 35.0
LAT = np.arange(-EDGE, EDGE + 0.5, 1.0)
LON = np.arange(0.0, 360.0, 1.0)
COS = np.cos(np.deg2rad(LAT))[:, None]
A = 6.371e6  # m
SIGMA = 1.5  # degrees: the Gaussian's standard deviation (about 3.5 degrees across at half height)
MARGIN = 10.0  # hPa: a point counts as under the ground when its surface pressure is within this of the level
BELOW = 0.25  # a box is not drawn with more than this share of its smoothing weight under the ground
FDR = 0.05
AXIS_LAT = (3.0, 30.0)  # the trough axis is drawn between these latitudes of either hemisphere
AXIS_SHEAR = 0.5  # m/s a degree of latitude: the least fall of the zonal wind towards the pole across the axis
AXIS_MIN = 8.0  # degrees: a shorter piece of axis is not drawn
FIELDS = ("n", "u9", "v9", "conv", "pos", "u8", "cyc")
PREFIX = "tropconv"
ANIM = Path(__file__).resolve().parents[3] / "assets" / "sst" / "anim"
# sector: its name and its western and eastern edges (degrees east)
SECTORS = {
    "atl": ("Atlantic and South America", -100.0, 20.0),
    "afr": ("Africa and the Indian Ocean", -20.0, 100.0),
    "wpac": ("West Pacific and Australia", 90.0, 210.0),
    "epac": ("East Pacific and the Americas", 180.0, 300.0),
}
LEV = np.array([-16, -12, -8, -6, -4, -2, 2, 4, 6, 8, 12, 16], float)  # 1e-6 s-1; nothing drawn between -2 and 2


# ── reduction ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _five(x: np.ndarray, axis: int) -> np.ndarray:
    """Means of five neighbours, the end ones at half weight, every fourth point along axis: 1-degree boxes centred
    on whole degrees of a 0.25-degree grid whose 4n + 1 points start half a degree before the first centre."""
    x = np.moveaxis(x, axis, 0)
    out = (0.5 * x[0:-1:4] + x[1::4] + x[2::4] + x[3::4] + 0.5 * x[4::4]) / 4.0
    return np.moveaxis(out, 0, axis)


def box(f: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A global 0.25-degree field (north to south, 0..360 east) as 1-degree box means on LAT x LON."""
    if f.shape != (721, 1440) or lat[0] != 90.0:
        raise ValueError(f"not the 0.25-degree grid from the north pole: {f.shape}, first row {lat[0]}")
    r0 = round((90.0 - EDGE - 0.5) / 0.25)
    g = np.roll(f[r0 : r0 + 4 * LAT.size + 1], 2, axis=1)  # half a degree west of 0 E first
    g = _five(_five(np.concatenate([g, g[:, :1]], axis=1), 1), 0)
    return g[::-1].astype(np.float32)


def _ddx(f: np.ndarray) -> np.ndarray:
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * np.deg2rad(1.0))


def _ddy(f: np.ndarray) -> np.ndarray:
    return np.gradient(f, np.deg2rad(LAT), axis=-2)


def convergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Horizontal convergence (s-1) of a wind on LAT x LON."""
    return -(_ddx(u) + _ddy(v * COS)) / (A * COS)


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Relative vorticity (s-1) of a wind on LAT x LON."""
    return (_ddx(v) - _ddy(u * COS)) / (A * COS)


def smooth(f: np.ndarray) -> np.ndarray:
    """The Gaussian over the last two axes (LAT x LON), around the globe in longitude."""
    sigma = (0.0,) * (f.ndim - 2) + (SIGMA, SIGMA)
    return gaussian_filter(f, sigma, mode=("nearest",) * (f.ndim - 1) + ("wrap",))


def day(files, steps: tuple, members) -> dict | None:
    """One lead day from the GRIB files that between them hold every member's u and v on LEVELS at the day's two
    steps: FIELDS (n the members complete, u9 v9 conv u8 their means, pos and cyc how many converge at 925 hPa and
    turn cyclonically at 850 hPa). None when no member is complete."""
    held = [(p, O.index_grib(p)) for p in files]

    def read(short: str, m: int, lev: int, s: int):
        for p, idx in held:
            if (short, m, lev, s) in idx:
                f, lat, _ = O.read_at(p, idx[(short, m, lev, s)])
                return box(f, lat)
        return None

    u, v = [], []
    for m in members:
        parts = [read(c, m, lev, s) for c in "uv" for lev in LEVELS for s in steps]
        if any(x is None for x in parts):
            continue
        x = np.stack(parts).reshape(2, len(LEVELS), len(steps), LAT.size, LON.size).mean(axis=2)
        u.append(x[0])
        v.append(x[1])
    if not u:
        return None
    u, v = np.stack(u), np.stack(v)
    k8, k9 = LEVELS.index(850), LEVELS.index(925)
    conv = smooth(convergence(u[:, k9], v[:, k9]))
    cyclonic = smooth(vorticity(u[:, k8], v[:, k8])) * np.sign(LAT)[:, None]
    return {
        "n": np.int16(len(u)),
        "u9": u[:, k9].mean(axis=0),
        "v9": v[:, k9].mean(axis=0),
        "conv": conv.mean(axis=0),
        "pos": (conv > 0).sum(axis=0).astype(np.int16),
        "u8": smooth(u[:, k8]).mean(axis=0),
        "cyc": (cyclonic > 0).sum(axis=0).astype(np.int16),
    }


def below_ground(path: Path) -> np.ndarray:
    """(level, LAT, LON): the share of each box's smoothing weight that lies under the ground at each of LEVELS, from
    a GRIB file holding one surface-pressure field."""
    (offset,) = (o for key, o in O.index_grib(path).items() if key[0] == "sp")
    sp, lat, _ = O.read_at(path, offset)
    return np.stack([smooth(box((sp < 100.0 * (lev + MARGIN)).astype(float), lat)) for lev in LEVELS])


def save_run(init: pd.Timestamp, days: list, ground: np.ndarray, out: Path) -> bool:
    """The run's lead days (day()'s dicts, None for a day not read) and below_ground as one file; False with none."""
    keep = [j for j, d in enumerate(days) if d is not None]
    if not keep:
        return False
    np.savez_compressed(
        out,
        init=f"{init:%Y-%m-%d}",
        leads=np.array([LEADS[j] for j in keep]),
        below=ground.astype(np.float32),
        **{k: np.stack([days[j][k] for j in keep]) for k in FIELDS},
    )
    return True


# ── maps ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def agreed(count: np.ndarray, n: int, where: np.ndarray) -> np.ndarray:
    """Boxes of `where` in which `count` of n members on one side differs from a coin toss: two-sided sign test, the
    false discovery rate held at FDR over the boxes tested."""
    from quantmet.stats import benjamini_hochberg

    p = np.minimum(1.0, 2.0 * binom.sf(np.maximum(count, n - count) - 1, n, 0.5))
    return benjamini_hochberg(np.where(where, p, np.nan), FDR)


def trough_axis(u8: np.ndarray, cyc: np.ndarray, n: int, open_air: np.ndarray) -> np.ndarray:
    """The mean 850 hPa zonal wind where its zero line is a trough axis, NaN elsewhere: westerlies on the equator's
    side and easterlies on the pole's (the wind falling by AXIS_SHEAR or more towards the pole), between AXIS_LAT,
    the members agreeing that the flow turns cyclonically."""
    poleward = np.deg2rad(_ddy(u8)) * np.sign(LAT)[:, None]  # the zonal wind's change a degree towards the pole
    band = (np.abs(LAT) >= AXIS_LAT[0]) & (np.abs(LAT) <= AXIS_LAT[1])
    where = open_air & band[:, None]
    ok = agreed(cyc, n, where) & (2 * cyc > n) & (poleward <= -AXIS_SHEAR)
    return np.where(ok, u8, np.nan)


def draw(d: dict, sector: str, title: str, foot: str, out: Path):
    """One day of one sector. d: conv (1e-6 s-1, NaN where not drawn), u9 v9 (NaN under the ground), axis
    (trough_axis), ground (true where 925 hPa is under the ground), all on LAT x LON."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import house
    import matplotlib.pyplot as plt
    from contourpy import contour_generator
    from matplotlib import patheffects
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import from_levels_and_colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _, west, east = SECTORS[sector]
    mid = (west + east) / 2
    x = (LON - mid + 180.0) % 360.0 - 180.0  # degrees east of the sector's central meridian: the axes' own
    order = np.argsort(x)
    order = order[np.abs(x[order]) <= (east - west) / 2 + 4]  # coordinate, a few boxes past the frame
    x = x[order]
    f = {k: v[:, order] for k, v in d.items()}

    width = 12.0
    w_map = width - 0.95
    h_map = w_map * 2 * BAND / (east - west)
    height = h_map + 1.75
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        [0.6 / width, 1.2 / height, w_map / width, h_map / height], projection=ccrs.PlateCarree(central_longitude=mid)
    )
    ax.set_xlim(west - mid, east - mid)
    ax.set_ylim(-BAND, BAND)
    colours = plt.get_cmap("BrBG")(np.linspace(0.04, 0.96, LEV.size + 1))
    colours[LEV.size // 2] = (1, 1, 1, 1)
    cmap, norm = from_levels_and_colors(LEV, colours, extend="both")
    if f["ground"].any():
        ax.contourf(x, LAT, f["ground"].astype(float), levels=[0.5, 1.5], colors=[house.EDGE], zorder=1)
    if np.isfinite(f["conv"]).any():
        ax.contourf(x, LAT, f["conv"], levels=LEV, cmap=cmap, norm=norm, extend="both", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor=house.INK, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor=house.MUTED, zorder=4)
    speed = np.hypot(f["u9"], f["v9"])
    # matplotlib's own streamplot: the grid is already the axes' coordinates (cartopy's would regrid it)
    Axes.streamplot(
        ax,
        x,
        LAT,
        np.ma.masked_invalid(f["u9"]),
        np.ma.masked_invalid(f["v9"]),
        density=(2.6, 1.5),
        color=house.INK2,
        linewidth=0.45 + 0.9 * np.clip(np.nan_to_num(speed) / 12.0, 0, 1),
        arrowsize=0.75,
        zorder=5,
    )
    halo = [patheffects.withStroke(linewidth=4.6, foreground="white")]
    for line in contour_generator(x, LAT, np.ma.masked_invalid(f["axis"])).lines(0.0):
        if np.hypot(*np.diff(line, axis=0).T).sum() >= AXIS_MIN:
            ax.plot(*line.T, color=house.INK, linewidth=2.6, solid_capstyle="round", path_effects=halo, zorder=6)
    xt = (np.arange(np.ceil(west / 20) * 20, east + 1, 20) + 180) % 360 - 180
    ax.set_xticks((xt - mid + 180) % 360 - 180)
    ax.set_xticklabels([f"{abs(t):.0f}°{'W' if t < 0 else 'E'}" if t % 180 else f"{abs(t):.0f}°" for t in xt])
    yt = np.arange(-30, 31, 10)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{abs(t)}°{'S' if t < 0 else 'N'}" if t else "0°" for t in yt])
    ax.tick_params(length=3)
    ax.axhline(0, color=house.MUTED, linewidth=0.5, linestyle=(0, (4, 4)), zorder=3)

    cax = fig.add_axes([0.6 / width, 0.72 / height, 0.47 * w_map / width, 0.15 / height])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=LEV)
    cb.set_label("925 hPa convergence (10⁻⁶ s⁻¹): green where the wind converges, brown where it spreads", fontsize=9)
    cb.ax.tick_params(labelsize=8.5, length=2)
    fig.legend(
        handles=[
            Line2D([], [], color=house.INK, linewidth=2.6, label="850 hPa trough axis"),
            Line2D([], [], color=house.INK2, linewidth=1.0, marker=">", markersize=4, label="925 hPa wind"),
            Patch(facecolor=house.EDGE, label="925 hPa under the ground"),
        ],
        loc="lower right",
        bbox_to_anchor=((0.6 + w_map) / width, 0.56 / height),
        ncols=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.text(0.6 / width, 1 - 0.3 / height, title, fontsize=13.5, fontweight="semibold", va="center")
    house.source(fig, foot, y=0.06 / height)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def render(run: Path):
    """Every sector's loop of the run (FNN.webp a lead day) and the viewer's manifest."""
    import house
    import matplotlib

    matplotlib.use("Agg")
    house.apply()
    z = np.load(run)
    init = pd.Timestamp(str(z["init"]))
    k8, k9 = LEVELS.index(850), LEVELS.index(925)
    ground = z["below"] > BELOW
    band = (np.abs(LAT) <= BAND)[:, None]
    frames = []
    for i, lead in enumerate(z["leads"]):
        n = int(z["n"][i])
        sure = agreed(z["pos"][i], n, ~ground[k9] & band)
        fields = {
            "conv": np.where(sure, 1e6 * z["conv"][i], np.nan),
            "u9": np.where(ground[k9], np.nan, z["u9"][i]),
            "v9": np.where(ground[k9], np.nan, z["v9"][i]),
            "axis": trough_axis(z["u8"][i], z["cyc"][i], n, ~ground[k8]),
            "ground": ground[k9],
        }
        least = int(np.maximum(z["pos"][i], n - z["pos"][i])[sure].min()) if sure.any() else n
        date = init + pd.Timedelta(days=int(lead))
        frames.append((int(lead), date, n, least, fields))
        print(f"  day {lead}: {n} members, colour from {least} of them on one side", flush=True)
    doc = {
        "ver": f"{init:%Y%m%d}00",
        "init": f"{init:%Y-%m-%d}T00:00Z",
        "default": f"{PREFIX}_{next(iter(SECTORS))}",
        "selectorLabel": "Sector",
        "regions": {},
    }
    label = O.MODELS[MODEL]["label"]
    for sector, (name, _, _) in SECTORS.items():
        d = ANIM / f"{PREFIX}_{sector}"  # the viewer reads <anim>/<region id>/<file>
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        rows = []
        for lead, date, n, least, fields in frames:
            fp = d / f"F{lead:02d}.webp"
            draw(
                fields,
                sector,
                f"Low-level convergence and monsoon troughs, {name} — {label} {init:%Y-%m-%d} 00Z, "
                f"{date:%a %-d %b} (+{lead} d)",
                f"ECMWF open data · the members' mean, each day from its 00 and 12 UTC states · colour where at least "
                f"{least} of the {n} members agree on the sign",
                fp,
            )
            rows.append({"file": fp.name, "date": f"{date:%Y-%m-%d}", "label": f"{date:%a %b %-d} · +{lead} d"})
        rows = [dict(r, idx=i) for i, r in enumerate(rows)]
        doc["regions"][f"{PREFIX}_{sector}"] = {"label": name, "n_frames": len(rows), "frames": rows}
        print(f"  {PREFIX}_{sector}: {len(rows)} days", flush=True)
    (ANIM / f"{PREFIX}_manifest.json").write_text(json.dumps(doc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="hadley_members.py's tropconv_<model>.npz")
    render(ap.parse_args().run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
