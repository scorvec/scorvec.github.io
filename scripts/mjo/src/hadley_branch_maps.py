#!/usr/bin/env python3
"""Hadley outflow and subtropical descent: the analysed weeks as maps and as a basin tracker, against ERA5 1991-2020.

The archive holds one 00 UTC AIFS-ENS control analysis a day (step 0 of the 00Z run) since FIRST, reduced to
hadley_branches' five fields on the 1.5 deg grid, as monthly files hb_YYYY-MM.npz in ARCH (in Actions, the release
hbranch-archive, as the Hadley loop's drift reference is a release: only the month in progress changes). Every frame and
tracker value is a trailing 7-day mean ending on its day (the Hadley product's windows), against the ERA5 1991-2020 normal
at the window's midpoint (build_branch_clim.py): the maps against the harmonic normal, the tracker against the
distribution of ERA5's own 7-day means. The analysis is first corrected by its mean offset from ERA5 at the same season,
as the Hadley loop's is (overturn.py): on the overturn lattice days the archive holds and ARCO has published, the analysis
less ERA5 (read at full precision, through the same reductions), pooled over +-overturn.WINDOW days of the year with
triangular weights; a season with fewer than MIN_PAIRS such days is not drawn.

    python src/hadley_branch_maps.py update             # archive the days not yet held (monthly files in ARCH)
    python src/hadley_branch_maps.py truth              # ERA5 on the lattice days not yet paired (OFFSETS in ARCH)
    python src/hadley_branch_maps.py render [--rebuild] # frames for days without one, the manifest, stills and tracker
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import build_branch_clim as C
import hadley_branches as HB
import overturn as O
from doycal import noleap_doy

REPO = Path(__file__).resolve().parents[3]
ANIM = REPO / "assets" / "sst" / "anim"
ARCH = REPO / "scripts" / "mjo" / "data" / "hbranch"
WRITTEN = ARCH / "written.txt"
OFFSETS = ARCH / "offset_era5.npz"  # per lattice day: the analysis less ERA5, each field
MIN_PAIRS = 6
DIR = "hbranch_hist"
FIRST = pd.Timestamp("2025-07-01")  # the AIFS-ENS open data begins here
WIN, DAYS = 7, 120
PC = ccrs.PlateCarree()
EXTENT = (-50.0, 50.0)
SECTOR_NAME = {
    "nh": "Northern Hemisphere",
    "sh": "Southern Hemisphere",
    "npac": "North Pacific",
    "natl": "North Atlantic",
    "spac": "South Pacific",
    "satl": "South Atlantic",
    "sind": "South Indian",
}
LEV = {
    "out": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4],
    "desc": [-60, -45, -30, -20, -10, -5, 5, 10, 20, 30, 45, 60],
    "msl": [-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12],
}


# ── archive ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _files(since: pd.Timestamp | None = None) -> list[Path]:
    return [p for p in sorted(ARCH.glob("hb_*.npz")) if since is None or p.stem[3:] >= f"{since:%Y-%m}"]


def held() -> pd.DatetimeIndex:
    """The archived days (an npz loads its arrays one by one, so this reads no fields)."""
    return pd.DatetimeIndex(np.concatenate([np.load(p)["date"] for p in _files()] or [np.array([], "datetime64[D]")]))


def load(since: pd.Timestamp | None = None) -> tuple[pd.DatetimeIndex, dict]:
    """The archived days from the month of `since` on, and their fields."""
    parts = [np.load(p) for p in _files(since)]
    if not parts:
        return pd.DatetimeIndex([]), {f: np.zeros((0, HB.LAT.size, HB.LON.size), np.float32) for f in HB.FIELDS}
    return pd.DatetimeIndex(np.concatenate([p["date"] for p in parts])), {
        f: np.concatenate([p[f] for p in parts]) for f in HB.FIELDS
    }


def analysis_day(day: pd.Timestamp) -> dict:
    """The five fields of the AIFS-ENS control's 00 UTC analysis of `day`, through the shared store."""
    import store as ecmwf

    cyc = ecmwf.Cycle(f"{day:%Y%m%d}", "00")
    paths = {
        "uv": ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", ("u", "v"), "pl", (200,), (0,))),
        "wz": ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", ("w", "z"), "pl", (500,), (0,))),
        "msl": ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "msl", "sfc", (), (0,))),
    }

    def get(key, short):
        kw = {"indexpath": "", "filter_by_keys": {"shortName": short}}
        return xr.open_dataset(paths[key], engine="cfgrib", backend_kwargs=kw)[short].squeeze(drop=True).load()

    F = HB.fields(get("uv", "u"), get("uv", "v"), get("wz", "w"), get("wz", "z") / C.G, get("msl", "msl"))
    for p in paths.values():
        p.unlink(missing_ok=True)
    return F


def _merge(path: Path, new: dict) -> int:
    """Add `new` ({day: fields}) to the npz at `path` in date order, list the file in WRITTEN for upload; the days held."""
    old = np.load(path) if path.exists() else None
    date = pd.DatetimeIndex(old["date"] if old is not None else []).append(pd.DatetimeIndex(list(new)))
    o = np.argsort(date.values)
    cols = {
        f: np.concatenate(([old[f]] if old is not None else []) + [np.stack([v[f] for v in new.values()])])
        for f in HB.FIELDS
    }
    np.savez_compressed(
        path, date=date.values[o].astype("datetime64[D]"), **{f: v[o].astype(np.float32) for f, v in cols.items()}
    )
    with WRITTEN.open("a") as fh:
        fh.write(path.name + "\n")
    return date.size


def update(until: pd.Timestamp):
    """Archive every day from FIRST to `until` not yet held, a month at a time."""
    have = set(held())
    want = pd.DatetimeIndex([d for d in pd.date_range(FIRST, until) if d not in have])
    print(f"update: {len(want)} days to archive", flush=True)
    ARCH.mkdir(parents=True, exist_ok=True)
    for m in sorted(set(want.strftime("%Y-%m"))):
        new = {}
        for day in want[want.strftime("%Y-%m") == m]:
            try:
                new[day] = analysis_day(day)
            except Exception as e:  # noqa: BLE001
                print(f"  {day:%Y-%m-%d}: {type(e).__name__} {str(e)[:80]}", flush=True)
        if new:
            print(f"  {m}: +{len(new)} days, {_merge(ARCH / f'hb_{m}.npz', new)} held", flush=True)


# ── the analysis offset from ERA5 ─────────────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _arco() -> xr.Dataset:
    import era5_store

    return xr.open_zarr(era5_store.ARCO, chunks=None, storage_options={"token": "anon"})


def era5_day(day: pd.Timestamp) -> dict | None:
    """ERA5's five fields at 00 UTC on `day`, from ARCO; None while ARCO has not published the day (it pads with NaN)."""

    def get(name, **level):
        return _arco()[name].sel(time=day, **level).load()

    v = get("v_component_of_wind", level=200)
    if bool(np.isnan(v.values).all()):
        return None
    return HB.fields(
        get("u_component_of_wind", level=200),
        v,
        get("vertical_velocity", level=500),
        get("geopotential", level=500) / C.G,
        get("mean_sea_level_pressure"),
    )


def truth(until: pd.Timestamp):
    """The analysis less ERA5 on every lattice day to `until` the archive holds and OFFSETS does not, saved every ten;
    ARCO publishes in order, so the first day it has not published ends the pass."""
    done = set(pd.DatetimeIndex(np.load(OFFSETS)["date"])) if OFFSETS.exists() else set()
    have = set(held())
    days = [d for d in O.lattice(FIRST, until) if d in have and d not in done]
    print(f"truth: {len(days)} lattice days to pair", flush=True)
    new = {}
    for day in days:
        E = era5_day(day)
        if E is None:
            print(f"  ERA5 not published for {day:%Y-%m-%d}: stopping", flush=True)
            break
        m = np.load(ARCH / f"hb_{day:%Y-%m}.npz")
        i = int(np.flatnonzero(pd.DatetimeIndex(m["date"]) == day)[0])
        new[day] = {f: m[f][i] - E[f] for f in HB.FIELDS}
        print(f"  {day:%Y-%m-%d} paired", flush=True)
        if len(new) == 10:
            print(f"truth: {_merge(OFFSETS, new)} lattice days paired", flush=True)
            new = {}
    if new:
        print(f"truth: {_merge(OFFSETS, new)} lattice days paired", flush=True)


def offset(pairs, doy: float) -> dict | None:
    """The analysis's mean offset from ERA5 at common-year day `doy`; None with fewer than MIN_PAIRS days in reach."""
    dd = np.abs(noleap_doy(pd.DatetimeIndex(pairs["date"])) - doy)
    w = np.clip(1.0 - np.minimum(dd, 365.0 - dd) / O.WINDOW, 0.0, None)
    if (w > 0).sum() < MIN_PAIRS:
        return None
    return {f: np.tensordot(w, pairs[f], axes=(0, 0)) / w.sum() for f in HB.FIELDS}


# ── windows and the climatology ───────────────────────────────────────────────────────────────────────────────────────
def week(date: pd.DatetimeIndex, D: dict, end: pd.Timestamp, pairs):
    """(fields less the analysis offset, midpoint common-year day) of the 7-day window ending `end`; None if a day is
    missing or the season's offset is unknown."""
    want = pd.date_range(end - pd.Timedelta(days=WIN - 1), end)
    pos = {d: i for i, d in enumerate(date)}
    if any(d not in pos for d in want):
        return None
    mid = float(noleap_doy(pd.DatetimeIndex([end - pd.Timedelta(days=(WIN - 1) / 2)]))[0])
    off = offset(pairs, mid)
    if off is None:
        return None
    rows = [pos[d] for d in want]
    return {f: D[f][rows].mean(0) - off[f] for f in HB.FIELDS}, mid


def clim_map(cl: xr.Dataset, field: str, doy: float) -> np.ndarray:
    c = cl["coef"].sel(field=field).values
    w = 2 * np.pi * doy / 365.25
    return c[0] + c[1] * np.cos(w) + c[2] * np.sin(w) + c[3] * np.cos(2 * w) + c[4] * np.sin(2 * w)


def clim_index(cl: xr.Dataset, key: str, doy, stat: str) -> np.ndarray:
    """A climatological stat of an index at common-year days (29 Feb = 59.5 takes its two neighbours' mean)."""
    doy = np.atleast_1d(np.asarray(doy, float))
    s = cl["index"].sel(key=key, stat=stat)
    return 0.5 * (s.sel(doy=np.floor(doy).astype(int)).values + s.sel(doy=np.ceil(doy).astype(int)).values)


# ── maps ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The maps are drawn in their own (Pacific-centred) coordinates, so cartopy has no polygon to reproject: x = longitude
# less CENTRE, on the grid, the seam at 21E (in Africa, every basin whole) closed by a repeated column.
CENTRE = 201.0
PROJ = ccrs.PlateCarree(central_longitude=CENTRE)
_ORDER = np.argsort((HB.LON - CENTRE + 180.0) % 360.0)
XN = np.r_[(HB.LON[_ORDER] - CENTRE + 180.0) % 360.0 - 180.0, 180.0]


def _native(f: np.ndarray) -> np.ndarray:
    g = f[:, _ORDER]
    return np.concatenate([g, g[:, :1]], axis=1)


def _poleward(v: np.ndarray) -> np.ndarray:
    return v * np.sign(HB.LAT)[:, None]


def _panel(fig, rect, cax, field, levels, title, label):
    ax = fig.add_axes(rect, projection=PROJ)
    cf = ax.contourf(XN, HB.LAT, _native(field), levels=levels, cmap="RdBu_r", extend="both")
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.set_extent([-180.0, 180.0, *EXTENT], crs=PROJ)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.6", alpha=0.6, xlocs=range(-180, 181, 60), ylocs=[-30, 0, 30]
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7.5}
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    cb = fig.colorbar(cf, cax=fig.add_axes(cax), orientation="vertical")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    return ax


def draw_map(F: dict, cl: xr.Dataset, mid: float, idx: dict, end: pd.Timestamp, out: Path):
    # inches: three map rows of the extent's aspect, each with room for its title and longitude labels
    W, left, mw, top, foot, block = 13.0, 0.6, 11.3, 0.5, 0.5, 0.34
    mh = mw * (EXTENT[1] - EXTENT[0]) / 360.0
    H = top + 3 * (mh + 2 * block) + foot
    rects = []
    for i in range(3):
        y = H - top - i * (mh + 2 * block) - block - mh
        rects.append(
            ((left / W, y / H, mw / W, mh / H), ((left + mw + 0.25) / W, (y + 0.1) / H, 0.16 / W, (mh - 0.2) / H))
        )
    fig = plt.figure(figsize=(W, H))
    ax = _panel(
        fig,
        *rects[0],
        _poleward(F["vchi200"]) - _poleward(clim_map(cl, "vchi200", mid)),
        LEV["out"],
        "Outflow · 200 hPa poleward divergent wind",
        "anomaly (m s⁻¹, + = more poleward)",
    )
    s = 4
    ax.quiver(
        XN[:-1:s],
        HB.LAT[::s],
        _native(F["uchi200"])[::s, :-1:s],
        _native(F["vchi200"])[::s, :-1:s],
        scale=90,
        width=0.0012,
        color="#222",
    )
    w = HB.PA_S_TO_HPA_DAY * F["w500"]
    ax = _panel(
        fig,
        *rects[1],
        w - HB.PA_S_TO_HPA_DAY * clim_map(cl, "w500", mid),
        LEV["desc"],
        "Descent · 500 hPa vertical motion",
        "anomaly (hPa day⁻¹, + = more descent)",
    )
    ax.contour(XN, HB.LAT, _native(w), levels=[20.0], colors="#111", linewidths=0.6)
    ax = _panel(
        fig,
        *rects[2],
        F["msl"] - clim_map(cl, "msl", mid),
        LEV["msl"],
        "Subtropical ridges · sea-level pressure and the 588-dam line",
        "SLP anomaly (hPa)",
    )
    for zf, ls, lw in ((F["z500"], "-", 1.6), (clim_map(cl, "z500", mid), "--", 1.0)):
        ax.contour(XN, HB.LAT, _native(zf), levels=[HB.Z588], colors="#111", linewidths=lw, linestyles=ls)
    for k in HB.HIGHS:
        ax.text(
            idx[f"hlon_{k}"],
            idx[f"hlat_{k}"],
            f"H\n{idx[f'high_{k}']:.0f}",
            transform=PC,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111",
        )
    fig.text(
        left / W,
        1 - 0.3 / H,
        f"Hadley outflow and subtropical descent — 7-day mean ending {end:%Y-%m-%d} (AIFS-ENS analyses, 00 UTC)",
        fontsize=13,
        fontweight="bold",
        va="center",
    )
    fig.text(
        left / W,
        0.12 / H,
        "colour = anomaly against ERA5 1991–2020 at the week's midpoint · arrows = 200 hPa divergent wind · middle: black "
        "line = descent of 20 hPa/day\nbottom: solid = this week's 588-dam z500 line, dashed = the normal's (at the "
        f"{int(C.EPOCH)} level of the 1991–2020 trend) · H = the subtropical highs' central pressure (hPa)",
        fontsize=8.2,
        color="#555",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


# ── tracker ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _bands(ax, cl, key, x, doy):
    q = {s: clim_index(cl, key, doy, s) for s in ("p10", "p25", "p50", "p75", "p90")}
    ax.fill_between(x, q["p10"], q["p90"], color="#c9d8e6", lw=0)
    ax.fill_between(x, q["p25"], q["p75"], color="#9dbad3", lw=0)
    ax.plot(x, q["p50"], color="#8a5a00", lw=1.2, ls=(0, (5, 2)))


TRACKERS = (
    (
        "hbranch_flow.webp",
        "Hadley outflow and subtropical descent",
        (("out", "Outflow", "m s⁻¹"), ("desc", "Descent", "hPa day⁻¹")),
        ("nh", "npac", "natl", "sh", "spac", "satl", "sind"),
        "outflow = poleward 200 hPa divergent wind over 10–30°, descent = 500 hPa ω over 15–35°, in each basin's "
        "eastern subsidence region (hemispheres: all longitudes)",
    ),
    (
        "hbranch_ridges.webp",
        "Subtropical ridges and highs",
        (("r588", "588-dam ridge area", "10⁶ km²"), ("high", "Subtropical high", "hPa")),
        tuple(HB.HIGHS),
        f"ridge area = where z500 ≥ 588 dam in the basin's 10–45° band, ERA5 at the {int(C.EPOCH)} level of its "
        "1991–2020 trend · high = the subtropical high's central sea-level pressure",
    ),
)


def draw_tracker(series: pd.DataFrame, cl: xr.Dataset, out_dir: Path):
    x = series.index
    doy = noleap_doy(x - pd.Timedelta(days=(WIN - 1) / 2))
    for name, title, cols, rows, what in TRACKERS:
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(12.4, 1.8 * len(rows) + 1.0), sharex=True)
        for i, s in enumerate(rows):
            for j, (pre, label, unit) in enumerate(cols):
                ax, key = axes[i, j], f"{pre}_{s}"
                _bands(ax, cl, key, x, doy)
                ax.plot(x, series[key], color="#0b3d6b", lw=1.8)
                now = series[key].iloc[-1]
                mu, sd = (clim_index(cl, key, doy[-1:], st)[0] for st in ("mean", "sd"))
                ax.set_title(
                    f"{SECTOR_NAME[s]} · {label} · latest {now:.1f} ({(now - mu) / sd:+.1f}σ)",
                    fontsize=9,
                    loc="left",
                    fontweight="bold",
                )
                ax.set_ylabel(unit, fontsize=8)
                ax.grid(alpha=0.25, lw=0.5)
                ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        fig.suptitle(
            f"{title} — analysed 7-day means against ERA5 1991–2020",
            fontsize=12.5,
            fontweight="bold",
            x=0.01,
            ha="left",
        )
        fig.text(
            0.01,
            0.004,
            "blue = AIFS-ENS analyses (00 UTC), 7-day means ending on the day · bands = ERA5's own 7-day means 1991–2020 "
            f"(10–90 %, 25–75 %), median dashed; σ = ERA5's spread\n{what}",
            fontsize=7.6,
            color="#555",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.975))
        fig.savefig(out_dir / name, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
        plt.close(fig)
        print(f"wrote {out_dir / name}", flush=True)


def render(rebuild: bool = False):
    """A frame per day of the last DAYS with a full window (only those not yet drawn, unless rebuild; older frames are
    deleted, and the directory is swapped onto the frames branch whole), the manifest, the still and the tracker."""
    since = held().max() - pd.Timedelta(days=DAYS + WIN)
    date, D = load(since)
    if not OFFSETS.exists():
        raise SystemExit("no analysis offset from ERA5 yet: run truth first")
    pairs = dict(np.load(OFFSETS))
    cl = xr.open_dataset(C.OUT)
    frame_dir, mp = ANIM / DIR, ANIM / f"{DIR}_manifest.json"
    frame_dir.mkdir(parents=True, exist_ok=True)
    drawn = {}
    if not rebuild and mp.exists():  # a frame counts as drawn only if the runner holds it (fetched from the branch)
        drawn = {f["date"]: f for f in json.loads(mp.read_text())["regions"][DIR]["frames"]}
        drawn = {k: f for k, f in drawn.items() if (frame_dir / f["file"]).exists()}
    rows, ends = [], []
    for end in date[date > date.max() - pd.Timedelta(days=DAYS)]:
        w = week(date, D, end, pairs)
        if w is None:
            print(f"  {end:%Y-%m-%d}: not drawn (a day missing, or fewer than {MIN_PAIRS} offset pairs)", flush=True)
            continue
        F, mid = w
        idx = HB.indices(F)
        rows.append(idx)
        ends.append(end)
        key = f"{end:%Y-%m-%d}"
        if key not in drawn:
            name = f"D{end:%Y%m%d}.webp"
            draw_map(F, cl, mid, idx, end, frame_dir / name)
            drawn[key] = {"file": name, "date": key, "label": f"week ending {end:%a %b %d, %Y}"}
    keep = {k: drawn[k] for k in (f"{e:%Y-%m-%d}" for e in ends)}
    for old in {p.name for p in frame_dir.glob("*.webp")} - {f["file"] for f in keep.values()}:
        (frame_dir / old).unlink()
    frames = [dict(f, idx=i) for i, (_, f) in enumerate(sorted(keep.items()))]
    regions = {
        DIR: {
            "label": "Hadley outflow and subtropical descent, analysed weeks",
            "n_frames": len(frames),
            "frames": frames,
        }
    }
    mp.write_text(
        json.dumps({"ver": int(pd.Timestamp.now("UTC").timestamp()), "days": len(frames), "regions": regions})
    )
    shutil.copyfile(frame_dir / frames[-1]["file"], REPO / "assets" / "sst" / "hbranch_map.webp")
    print(f"  {len(frames)} frames, through {frames[-1]['date']}", flush=True)
    draw_tracker(pd.DataFrame(rows, index=pd.DatetimeIndex(ends)), cl, REPO / "assets" / "sst")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update")
    sub.add_parser("truth")
    r = sub.add_parser("render")
    r.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    until = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    if a.cmd == "update":
        update(until)
    elif a.cmd == "truth":
        truth(until)
    else:
        render(a.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
