#!/usr/bin/env python3
"""AIFS single vs AIFS-ENS member 0 — side-by-side forecast animators.

Three loops per 00Z/12Z cycle, one frame every 6 h through hour 360, each frame
showing the two models side by side:
  * compare (North America, Lambert conformal): MSLP, 1000–500 hPa thickness
    (540 dam bold) and 6-h precipitation
  * z500 (Northern Hemisphere, polar stereographic): 500 hPa height anomaly vs
    the ERA5 1991–2020 day-of-year climatology, with height contours (height
    alone when the climatology file is absent)
  * t2m (North America): 2 m temperature anomaly vs the 1991–2020 normal for
    the valid hour (a 24-h mean against the daily normal when the hour-matched
    file is absent), with 2 m temperature contours; no frames without a
    climatology

Frames are drawn with matplotlib + cartopy in a pool of forked workers (fields
loaded once in the parent) and written as WebP into assets/sst/anim/aifs_compare,
aifs_z500 and aifs_t2m, each with its manifest.

Data flows through the shared ecmwf store (scripts/ecmwf): Google Cloud mirror,
per-file locks, GRIB message-count verification. A non-blocking flock makes
concurrent invocations impossible.

    python aifs_compare_anim.py [--date YYYYMMDD --time 00]
"""

from __future__ import annotations
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf

ANIM = REPO / "assets" / "sst" / "anim" / "aifs_compare"
MANIFEST = REPO / "assets" / "sst" / "anim" / "aifs_compare_manifest.json"
ANIM_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500"
MANIFEST_Z = REPO / "assets" / "sst" / "anim" / "aifs_z500_manifest.json"
ANIM_T = REPO / "assets" / "sst" / "anim" / "aifs_t2m"
MANIFEST_T = REPO / "assets" / "sst" / "anim" / "aifs_t2m_manifest.json"

STEPS = list(range(0, 361, 6))
G = 9.80665
EXTENT = [-128, -63, 22, 55]  # North America zoom
P_LEVELS = [0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 50]
P_COLORS = ["#d7ecd7", "#a8d8a8", "#6cbf6c", "#2d8f2d", "#ffe57d", "#ffbb4d", "#ff8c33", "#e85c29", "#c62828"]


def fetch(date: str, hh: str) -> dict:
    """All four files through the shared store; returns {(model,kind): path}."""
    cyc = ecmwf.Cycle(date, hh)
    S = tuple(STEPS)
    out = {}
    for mkey, model, typ in (("single", "aifs-single", "fc"), ("ens", "aifs-ens", "cf")):
        out[(mkey, "sfc")] = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, ("msl", "tp"), "sfc", (), S))
        out[(mkey, "2t")] = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "2t", "sfc", (), S))
        out[(mkey, "z")] = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "z", "pl", (1000, 500), S))
    return out


def load_all(paths):
    out = {}
    for mkey in ("single", "ens"):
        kw = dict(engine="cfgrib")
        msl = xr.open_dataset(
            paths[(mkey, "sfc")], backend_kwargs=dict(filter_by_keys={"shortName": "msl"}, indexpath=""), **kw
        )
        tp = xr.open_dataset(
            paths[(mkey, "sfc")], backend_kwargs=dict(filter_by_keys={"shortName": "tp"}, indexpath=""), **kw
        )
        z = xr.open_dataset(paths[(mkey, "z")], backend_kwargs={"indexpath": ""}, **kw)
        t2 = xr.open_dataset(paths[(mkey, "2t")], backend_kwargs={"indexpath": ""}, **kw)
        out[mkey] = dict(msl=msl["msl"], tp=tp["tp"], z=z["z"], t2m=t2["t2m"])
    return out


# ---------------------------------------------------------------- matplotlib
# Three loops, every frame drawn by one function; frames are rendered in
# parallel across the machine's cores (fork: the loaded fields are inherited,
# nothing is pickled). Serial on a 4-core runner the three loops took the
# best part of an hour (2026-09-02) - past the step timeout, so nothing was
# ever published.
Z500_CLIM = REPO / "scripts" / "verify" / "data" / "clim" / "clim_1p5.npz"
LOOPS = {
    "compare": dict(
        anim=ANIM,
        manifest=MANIFEST,
        region="aifs_compare",
        label="AIFS single vs AIFS-ENS member 0 — MSLP/thickness/precip",
    ),
    "z500": dict(
        anim=ANIM_Z,
        manifest=MANIFEST_Z,
        region="aifs_z500",
        label="AIFS single vs AIFS-ENS member 0 — 500 hPa height anomaly + height (NH)",
    ),
    "t2m": dict(
        anim=ANIM_T,
        manifest=MANIFEST_T,
        region="aifs_t2m",
        label="AIFS single vs AIFS-ENS member 0 — 2 m temperature anomaly (North America)",
    ),
}
_G = {}  # fields shared with forked workers


T2M_CLIM_6H = REPO / "scripts" / "verify" / "data" / "clim" / "clim_1p5_t2m6h.npz"


def clim_row(t, d) -> int:
    """Row of a (366, ...) day-of-year climatology file `d` for time t: doy366 for files built on the 366-slot
    calendar (key `slot366`; 1 March = row 60 in every year). Older files were grouped by the raw day of year, so
    their rows after 29 Feb are mostly common-year days: read the common-year row (1 March = row 59 in every year)."""
    s = int(doy366(t))
    return s if "slot366" in d.files or s <= 59 else s - 1


def t2m_clim_6h(valid, hour):
    """Hour-of-day (00/06/12/18Z) 2 m climatology → (120, 240) °C on the
    verifier grid (NH rows filled, SH NaN), or None if the file is absent.
    Built by build_t2m_clim_6h.py; 18Z temperatures get an 18Z normal."""
    if not T2M_CLIM_6H.exists():
        return None
    d = np.load(T2M_CLIM_6H)
    key = f"t2m_h{int(np.rint(hour / 6.0)) * 6 % 24:02d}"
    if key not in d.files:
        return None
    nh = d[key][clim_row(valid, d)].astype(float) - 273.15  # (60, 240)
    full = np.full((120, 240), np.nan)
    full[: nh.shape[0]] = nh
    return full


def z500_clim_on(lat, lon, valid, var="z500", hour=None):
    """ERA5 1991-2020 ±7 d day-of-year climatology on the model grid: z500 in
    dam, t2m in °C (hour-matched when the 6-hourly file exists). The
    climatology lives on a 1.5° NH grid (see aifs_station_verify.py); NaN
    south of the equator, which neither loop shows."""
    if not Z500_CLIM.exists():
        return None
    d = np.load(Z500_CLIM)
    if var not in d.files:
        return None
    c = None
    if var == "t2m" and hour is not None:
        c = t2m_clim_6h(valid, hour)
    if c is None:
        c = d[var][clim_row(valid, d)].astype(float)  # (120, 240)
        if var == "t2m" and np.nanmean(c) > 100:
            c = c - 273.15
    # The WB2 source grid ends at 358.5E, so the climatology's last 1.5° column
    # (centre 359.125E) was never filled: NaN there spread into a wedge from
    # the pole to the Greenwich meridian on every anomaly map. Fill it by
    # wrapping between its neighbours.
    if np.isnan(c[:, -1]).all() and not np.isnan(c[:, -2]).all():
        c[:, -1] = 0.5 * (c[:, -2] + c[:, 0])
    clat = 90.0 - 0.25 * (np.arange(720).reshape(120, 6).mean(axis=1))
    clon = 0.25 * (np.arange(1440).reshape(240, 6).mean(axis=1))
    da = xr.DataArray(c, coords=dict(latitude=clat, longitude=clon), dims=("latitude", "longitude"))
    # pad BOTH ends: the 1.5° cell centres run 0.75..359.25, so a 0.25° point
    # between 359.25 and 360 or between 0 and 0.75 fell outside the range and
    # came back NaN - a white wedge from the pole to the Greenwich meridian
    da = xr.concat(
        [
            da.isel(longitude=-1).assign_coords(longitude=clon[-1] - 360.0),
            da,
            da.isel(longitude=0).assign_coords(longitude=clon[0] + 360.0),
        ],
        dim="longitude",
    )
    lon360 = np.mod(lon, 360.0)
    out = da.interp(latitude=("y", lat), longitude=("x", lon360), method="linear").values
    return out / 10.0 if var == "z500" else out


def _subset(lat, lon, arr, lat0, lat1, lon0, lon1):
    """Cut a global (lat, lon) field to a lon/lat box (lon in -180..180) and
    return 2-D meshgrids: contouring 1.3 M global points onto a projection was
    what made a frame cost 24 s on a runner; the box is ~5% of that, and
    transform_first=True lets cartopy project the points before contouring."""
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180)
    lon180 = lon180[order]
    arr = arr[:, order]
    mi = (lat >= lat0) & (lat <= lat1)
    mj = (lon180 >= lon0) & (lon180 <= lon1)
    X, Y = np.meshgrid(lon180[mj], lat[mi])
    return X, Y, arr[np.ix_(mi, mj)]


NA_BOX = (12.0, 66.0, -150.0, -40.0)  # generous around EXTENT for the LCC corners
TF = dict(transform_first=True)


def _na_axes(ax, cfeature, ccrs):
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.coastlines(lw=1.1, color="0.05")
    ax.add_feature(cfeature.BORDERS, lw=0.8, edgecolor="0.15", facecolor="none")
    ax.add_feature(cfeature.STATES, lw=0.55, edgecolor="0.3", facecolor="none")
    ax.add_feature(cfeature.LAKES, lw=0.55, edgecolor="0.3", facecolor="none")


def _prewarm_features():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    for proj, ext in (
        (ccrs.LambertConformal(central_longitude=-97, central_latitude=39), EXTENT),
        (ccrs.NorthPolarStereo(central_longitude=-100), [-180, 180, 20, 90]),
    ):
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(ext, ccrs.PlateCarree())
        ax.coastlines(lw=0.5)
        for f in (cfeature.BORDERS, cfeature.STATES, cfeature.LAKES):
            ax.add_feature(f, lw=0.3, facecolor="none")
        fig.canvas.draw()
        plt.close(fig)


def _draw_frame(job):
    """Render one frame of one loop; returns the manifest entry. One retry:
    a transient failure must not take the whole loop down."""
    for attempt in (1, 2):
        try:
            return _draw_frame_once(job)
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  frame {job} failed twice: {str(e)[:90]}", flush=True)
                return None
            time.sleep(2)


def _draw_frame_once(job):
    kind, i, s = job
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point

    F, base = _G["F"], _G["base"]
    valid = base + pd.Timedelta(hours=s)
    sd = pd.Timedelta(hours=s)
    lat, lon = _G["lat"], _G["lon"]
    if kind == "z500":
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6), constrained_layout=True, subplot_kw={"projection": proj})
        levels = np.arange(486, 601, 6)
        alev = np.arange(-27, 27.1, 3)
        clim = z500_clim_on(lat, lon, valid)
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS member 0")):
            z5 = F[mkey]["z"].sel(step=sd, isobaricInhPa=500).values / G / 10.0
            # polar view: subset to the hemisphere (the saving) but let cartopy
            # transform the contours (transform_first folds the grid at the
            # pole and produced a solid black frame, 2026-09-02)
            nh = lat >= 12.0
            latn = lat[nh]
            # Start the longitude axis at the projection's antimeridian (80E for
            # a -100 central meridian) so the data wrap and the map boundary
            # coincide: otherwise contourf leaves a white seam from the pole to
            # the boundary where its polygons are clipped.
            lon360 = np.mod(lon, 360.0)
            order = np.argsort(lon360)  # -180..180 input → 0..360 monotonic
            lon360 = lon360[order]
            k = int(np.argmin(np.abs(lon360 - 80.0)))
            lonr = np.concatenate([lon360[k:], lon360[:k] + 360.0])
            zs = z5[nh][:, order]
            z5c, lonc = add_cyclic_point(np.roll(zs, -k, axis=1), coord=lonr)
            if clim is not None:
                a_s = (z5 - clim)[nh][:, order]
                ac, _ = add_cyclic_point(np.roll(a_s, -k, axis=1), coord=lonr)
                cf = ax.contourf(
                    lonc, latn, ac, levels=alev, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree()
                )
            else:
                cf = ax.contourf(
                    lonc, latn, z5c, levels=levels, cmap="turbo", extend="both", transform=ccrs.PlateCarree()
                )
            cl = ax.contour(lonc, latn, z5c, levels=levels, colors="k", linewidths=0.55, transform=ccrs.PlateCarree())
            ax.clabel(cl, levels=levels[::2], fmt="%d", fontsize=6.5, inline_spacing=2)
            ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
            ax.coastlines(lw=0.8, color="0.15")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.3", facecolor="none")
            ax.gridlines(lw=0.3, color="0.75", ylocs=[30, 50, 70], xlocs=range(-180, 181, 60))
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label(
            "500 hPa height anomaly vs 1991–2020 (dam) · contours: height (dam, every 6)"
            if clim is not None
            else "500 hPa height (dam)",
            fontsize=8.5,
        )
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"500 hPa height anomaly + height · NH — hour {s} · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
            fontsize=12,
            fontweight="bold",
        )
    elif kind == "t2m":
        proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), constrained_layout=True, subplot_kw={"projection": proj})
        alev = np.arange(-16, 16.1, 2)
        tlev = np.arange(-40, 46, 5)
        clim = z500_clim_on(lat, lon, valid, var="t2m", hour=valid.hour)
        hourly = T2M_CLIM_6H.exists()
        if clim is None:
            plt.close(fig)
            return None
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS member 0")):
            t2 = F[mkey]["t2m"].sel(step=sd).values - 273.15
            if hourly:
                # hour-matched normal: the anomaly of THIS hour's temperature
                anom_src = t2
            else:
                # daily-mean normal only: an instantaneous field against it
                # reads cold every morning and warm every evening, so fall
                # back to the 24-h mean centred on the valid time
                steps = F[mkey]["t2m"].step.values
                win = [st for st in steps if abs((st - sd) / np.timedelta64(1, "h")) <= 12]
                anom_src = F[mkey]["t2m"].sel(step=win).mean("step").values - 273.15
            X, Y, T = _subset(lat, lon, t2, *NA_BOX)
            _, _, A = _subset(lat, lon, anom_src - clim, *NA_BOX)
            cf = ax.contourf(X, Y, A, levels=alev, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree(), **TF)
            ct = ax.contour(X, Y, T, levels=tlev, colors="0.25", linewidths=0.5, transform=ccrs.PlateCarree(), **TF)
            ax.clabel(ct, levels=tlev[::2], fmt="%d", fontsize=6, inline_spacing=2)
            c0 = ax.contour(X, Y, T, levels=[0], colors="#1565c0", linewidths=1.5, transform=ccrs.PlateCarree(), **TF)
            ax.clabel(c0, fmt="%d°C", fontsize=6.5, inline_spacing=2)
            _na_axes(ax, cfeature, ccrs)
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label(
            (
                "2 m temperature anomaly vs the 1991–2020 normal for this hour (°C)"
                if hourly
                else "24-h mean 2 m temperature anomaly vs 1991–2020 (°C)"
            )
            + " · contours: 2 m temperature now (°C, every 5; 0 °C blue)",
            fontsize=8.5,
        )
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"2 m temperature anomaly + temperature — hour {s} · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
            fontsize=12,
            fontweight="bold",
        )
    else:  # compare: MSLP / thickness / precip
        proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=39)
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), constrained_layout=True, subplot_kw={"projection": proj})
        for ax, mkey, name in zip(axes, ("single", "ens"), ("AIFS single", "AIFS-ENS member 0")):
            d = F[mkey]
            msl = d["msl"].sel(step=sd).values / 100.0
            thk = (d["z"].sel(step=sd, isobaricInhPa=500) - d["z"].sel(step=sd, isobaricInhPa=1000)).values / G / 10.0
            if s == 0:
                pr = np.zeros_like(msl)
            else:
                pr = d["tp"].sel(step=sd).values - d["tp"].sel(step=sd - pd.Timedelta(hours=6)).values
            X, Y, PR = _subset(lat, lon, np.clip(pr, 0, None), *NA_BOX)
            _, _, TH = _subset(lat, lon, thk, *NA_BOX)
            _, _, MS = _subset(lat, lon, msl, *NA_BOX)
            cf = ax.contourf(
                X, Y, PR, levels=P_LEVELS, colors=P_COLORS, extend="max", transform=ccrs.PlateCarree(), **TF
            )
            for rng, col in (((410, 540), "#1565c0"), ((546, 620), "#c62828")):
                ct = ax.contour(
                    X,
                    Y,
                    TH,
                    levels=np.arange(rng[0], rng[1], 6),
                    colors=col,
                    linewidths=0.6,
                    linestyles="--",
                    transform=ccrs.PlateCarree(),
                    **TF,
                )
                if len(ct.levels):
                    ax.clabel(ct, levels=list(ct.levels)[::2], fmt="%d", fontsize=5.5, inline_spacing=2)
            c540 = ax.contour(
                X,
                Y,
                TH,
                levels=[540],
                colors="#1565c0",
                linewidths=1.6,
                linestyles="--",
                transform=ccrs.PlateCarree(),
                **TF,
            )
            ax.clabel(c540, fmt="%d", fontsize=6, inline_spacing=2)
            cs = ax.contour(
                X,
                Y,
                MS,
                levels=np.arange(940, 1061, 4),
                colors="k",
                linewidths=0.75,
                transform=ccrs.PlateCarree(),
                **TF,
            )
            ax.clabel(cs, levels=np.arange(940, 1061, 8), fmt="%d", fontsize=6)
            _na_axes(ax, cfeature, ccrs)
            ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
        cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", pad=0.02, fraction=0.05, aspect=48)
        cb.set_label("6-h precipitation (mm)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"MSLP (hPa) · 1000–500 thickness (dam, 540 bold) · 6-h precip — hour {s}"
            f" · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}",
            fontsize=12,
            fontweight="bold",
        )
    out = LOOPS[kind]["anim"] / f"F{i:02d}.webp"
    fig.savefig(
        out, dpi=150, pil_kwargs={"quality": 65, "method": 6}
    )  # loop frames: WebP q65 m6 (2026-10-05 encode study)
    plt.close(fig)
    return (kind, {"idx": i, "file": out.name, "date": f"{valid:%Y-%m-%d}", "label": f"h{s:03d} · {valid:%b %d %HZ}"})


def render_mpl(date: str, hh: str, paths, kinds=("compare", "z500", "t2m"), workers=None):
    import multiprocessing as mp
    import os

    base = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}:00")
    F = load_all(paths)
    # cfgrib is lazy: read everything once in the parent so forked workers
    # never touch the GRIB files concurrently
    for mkey in F:
        for k in list(F[mkey]):
            F[mkey][k] = F[mkey][k].load()
    _G.update(F=F, base=base, lat=F["single"]["msl"].latitude.values, lon=F["single"]["msl"].longitude.values)
    for kind in kinds:
        LOOPS[kind]["anim"].mkdir(parents=True, exist_ok=True)
        for old in LOOPS[kind]["anim"].glob("F*.webp"):
            old.unlink()
    # Pre-warm cartopy's Natural Earth cache in the PARENT: on a fresh runner
    # every forked worker otherwise downloads the same shapefiles at once and
    # one of them reads a half-written file (struct.error, 2026-09-02).
    _prewarm_features()
    jobs = [(kind, i, s) for kind in kinds for i, s in enumerate(STEPS)]
    workers = workers or max(1, min(os.cpu_count() or 2, 8))
    t0 = time.time()
    frames = {k: [] for k in kinds}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_draw_frame, jobs, chunksize=2), 1):
            if res:
                frames[res[0]].append(res[1])
            if n % 30 == 0:
                print(f"  {n}/{len(jobs)} frames in {time.time() - t0:.0f}s ({workers} workers)", flush=True)
    for kind in kinds:
        fr = sorted(frames[kind], key=lambda f: f["idx"])
        if not fr:
            print(f"  {kind}: no frames rendered — manifest untouched", flush=True)
            continue
        man = {
            "ver": int(time.time()),
            "days": len(fr),
            "regions": {LOOPS[kind]["region"]: {"label": LOOPS[kind]["label"], "n_frames": len(fr), "frames": fr}},
        }
        LOOPS[kind]["manifest"].write_text(json.dumps(man))
        print(f"wrote {len(fr)} {kind} frames + manifest", flush=True)
    print(f"rendered {len(jobs)} frames in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default="00")
    args = ap.parse_args()

    lockdir = REPO / "assets" / "sst" / "anim"
    lockdir.mkdir(parents=True, exist_ok=True)
    lockf = (lockdir / ".aifs_anim.lock").open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another aifs animator instance is running — aborting")

    paths = fetch(args.date, args.time)
    render_mpl(args.date, args.time, paths)


if __name__ == "__main__":
    main()
