#!/usr/bin/env python3
"""Dynamic-tropopause maps around each storm on the TC board, from the IFS control (HRES until May 2026; AIFS single
as the labelled fallback).

For every storm in data/tc_board.json (up to MAX_STORMS), a loop 6-hourly to day 10 on a Lambert fan around its members'
tracks (tclib.track_sector, tclib.sector_axes):
  shading   theta on the 2-PVU surface: Ertel PV on the nine levels 700-100 hPa (dt_charts.pv_step), the surface by
            quantmet.pv.dynamic_tropopause, as on the hemispheric dynamic-tropopause maps (where 2 PVU lies above
            100 hPa, the 100 hPa values); black contours: its pressure, 200-500 hPa
  magenta   outflow against the PV gradient: -v_chi . grad(PV) in the 300-200 hPa layer (tclib.irrotational,
            tclib.pv_advection) at -1 and -2 PVU/day, within OUTFLOW_KM of the deterministic run's storm (else of the
            members' median position)
  storm     the deterministic run's own storm (ECMWF's tracker on that run, from the board): its track as a thin black
            line and a dot at the frame's time; small dots: where the AIFS-ENS (blue) and IFS-ENS (red) members have the
            storm at that time (tc_ens.py --members), open once followed past the tracker (post-tropical)
The maps show where the storms, the troughs and the outflow are; nothing about what one does to another.
Frames: assets/sst/anim/tcdt/<storm>/F##.webp (frames branch); manifest: assets/sst/anim/tcdt/tcdt_manifest.json (main).
    python src/tc_maps.py --date 20261018 --time 00 --board ../../assets/sst/data/tc_board.json \
        --members /tmp/tc_members.json --anim-dir ../../assets/sst/anim/tcdt
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.pv import dynamic_tropopause

sys.path.insert(0, str(Path(__file__).parent))
import dt_charts as DT
import tclib as TC

STEPS = tuple(range(0, TC.FOLLOW_H_MAX + 1, 6))
MAX_STORMS = 4
OUTFLOW_KM = 1500.0
SOURCES = (("ifs", TC.DET["ifs"]), ("aifs-single", TC.DET["aifs"]))  # the run the maps use, in order of preference
MAP_W_MAX, MAP_H_MAX, TOP_IN, BOT_IN = 10.8, 6.6, 0.42, 1.12  # frame layout (inches): never taller than a screen


def source(date: str, hh: str) -> tuple[str, str] | None:
    """(model, label) of the first deterministic run with every map step on the Google mirror."""
    import cycles

    for model, lab in SOURCES:
        names = set(cycles._list_names(f"{date}/{hh}z/{model}/0p25/oper/") or [])
        need = [f"{date}/{hh}z/{model}/0p25/oper/{date}{hh}0000-{s}h-oper-fc" for s in STEPS]
        if all(n + ".index" in names and n + ".grib2" in names for n in need):
            return model, lab
        print(f"  {lab} {date} {hh}Z: not every step to +{STEPS[-1]} h is on the Google mirror", flush=True)
    return None


def outflow(U, V, pv, lat, lon):
    """-v_chi . grad(PV) (PVU/day), 300-200 hPa layer, on the field grid."""
    k = [DT.LEVS.index(L) for L in (300, 250, 200)]
    co = dict(latitude=lat, longitude=lon)
    u2 = xr.DataArray(U[k].mean(0), dims=("latitude", "longitude"), coords=co)
    v2 = xr.DataArray(V[k].mean(0), dims=("latitude", "longitude"), coords=co)
    uc, vc = TC.irrotational(u2, v2, lat, lon)
    adv = TC.pv_advection(uc, vc, pv[k].mean(0), lat, lon)
    adv[np.abs(lat) > 85] = np.nan  # the pole rows' PV is singular
    return adv


def window(lat, lon, lon0, lon1, lat0, lat1, pad=10.0):
    """Row and column indices of the grid covering the sector (with a margin), and the column longitudes unwrapped into
    the sector's continuous frame."""
    j = np.where((lat >= lat0 - pad) & (lat <= lat1 + pad) & (np.abs(lat) < 89.9))[0]
    off = (lon - (lon0 - pad)) % 360
    i = np.where(off <= (lon1 - lon0) + 2 * pad)[0]
    i = i[np.argsort(off[i])]
    return j, i, lon0 - pad + off[i]


def position(tr: dict, h: int):
    """(lat, lon, followed) of a member/run track at lead h, or None."""
    if h not in tr["h"]:
        return None
    k = tr["h"].index(h)
    return tr["lat"][k], tr["lon"][k], bool(tr.get("followed", [False] * len(tr["h"]))[k])


def frame(storm, members, det, sector, fields, init, h, lab, out: Path):
    """One map: theta on 2 PVU, DT pressure, outflow near the storm, the storm's positions."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    lon0, lon1, lat0, lat1, asp = sector
    lat, lon, th, p, adv = fields
    mw = min(MAP_W_MAX, MAP_H_MAX / asp)
    mh = mw * asp
    W, H = max(mw + 0.3, 8.6), TOP_IN + mh + BOT_IN  # the figure is as wide as its map (room for the key below)
    fig = plt.figure(figsize=(W, H))
    ax = TC.sector_axes(fig, [(W - mw) / 2 / W, BOT_IN / H, mw / W, mh / H], lon0, lon1, lat0, lat1, land=False)
    j, i, lon_u = window(lat, lon, lon0, lon1, lat0, lat1)
    LO, LA = np.meshgrid(lon_u, lat[j])
    xyz = ax.projection.transform_points(ccrs.PlateCarree(), LO, LA)  # native coordinates: no wrap streaks
    X, Y = xyz[..., 0], xyz[..., 1]
    (dcm, dnorm), _ = DT._cmaps()
    sub = np.ix_(j, i)
    cf = ax.pcolormesh(X, Y, th[sub], cmap=dcm, norm=dnorm, shading="nearest", rasterized=True, zorder=0)
    ax.contour(X, Y, p[sub], levels=[200, 300, 400, 500], colors="k", linewidths=[0.5, 0.7, 0.9, 1.1], alpha=0.55)
    ax.coastlines("50m", lw=0.9, color="#000", zorder=5)  # over the shading (sector_axes draws them under it)
    here = position(det, h) if det else None
    pts = [q for q in (position(t, h) for m in members.values() for t in m) if q is not None]
    if here is None and pts:  # no deterministic storm at this time: the members' median position
        here = (float(np.median([q[0] for q in pts])), TC.lon_median([q[1] for q in pts]), False)
    if here is not None and adv is not None:
        a = np.where(TC.within(lat[j], lon_u, here[0], here[1], OUTFLOW_KM), adv[sub], np.nan)
        if np.isfinite(a).any() and np.nanmin(a) <= -1:
            ax.contour(X, Y, a, levels=[-2, -1], colors="#d4148c", linewidths=[1.8, 1.1], linestyles="solid", zorder=4)
    for m, col in (("aifs", "#1f5fa8"), ("ifs", "#c0392b")):
        q = [x for x in (position(t, h) for t in members.get(m, [])) if x is not None]
        if q:
            qa = np.array([[x[0], x[1]] for x in q])
            fol = np.array([x[2] for x in q])
            for sel, face in ((~fol, col), (fol, "#fff")):
                if sel.any():
                    ax.scatter(
                        qa[sel, 1],
                        qa[sel, 0],
                        s=11,
                        c=face,
                        edgecolors=col,
                        linewidths=0.8,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )
    if det:
        ax.plot(det["lon"], det["lat"], color="#000", lw=1.0, transform=ccrs.Geodetic(), zorder=7)
        d0 = position(det, h)
        if d0 is not None:
            ax.plot(
                d0[1],
                d0[0],
                marker="o",
                ms=8,
                color="#000",
                mec="#fff",
                mew=1.2,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
    valid = init + pd.Timedelta(hours=int(h))
    fig.text(
        0.15 / W,
        1 - 0.08 / H,
        f"{storm['name']} on the dynamic tropopause — {lab}, init {init:%d %b %HZ} · "
        + ("analysis" if h == 0 else f"+{h} h")
        + f" · {valid:%a %d %b %HZ}",
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    cax = fig.add_axes([0.15 / W, 0.62 / H, 3.9 / W, 0.13 / H])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.6, pad=1.5)
    cb.set_label("θ on 2 PVU (K) · black: its pressure, 200–500 hPa\ntropics: θ at 100 hPa", fontsize=7.6, labelpad=2)
    fig.text(
        4.35 / W,
        0.86 / H,
        f"magenta: outflow −v_χ·∇PV at −1 and −2 PVU/day (300–200 hPa)\nwithin {OUTFLOW_KM:,.0f} km of the storm · "
        f"black line and dot: the storm in {lab}\n(ECMWF tracker) · dots: where the AIFS-ENS (blue) and IFS-ENS (red)\n"
        "members have it now, open once followed past the tracker",
        fontsize=7.4,
        color="#333",
        va="top",
    )
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 65, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--board", default="assets/sst/data/tc_board.json")
    ap.add_argument("--members", required=True, help="tc_ens.py --members work file")
    ap.add_argument("--anim-dir", default="assets/sst/anim/tcdt")
    ap.add_argument("--max-steps", type=int, default=0, help="render only the first N steps (timing tests)")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    t0 = time.time()
    hh = f"{int(a.time):02d}"
    init = pd.Timestamp(f"{a.date}T{hh}:00")
    board = json.loads(Path(a.board).read_text())
    mem = json.loads(Path(a.members).read_text())
    want = f"{init:%Y-%m-%dT%H:%MZ}"
    if board.get("init") != want or mem.get("init") != want:
        raise SystemExit(f"board ({board.get('init')}) or member file ({mem.get('init')}) is not the {want} cycle")
    storms = [s for s in board["storms"] if s["slug"] in mem["storms"]][:MAX_STORMS]
    anim = Path(a.anim_dir)
    if anim.exists():
        shutil.rmtree(anim)  # departed storms' loops go with the old frames
    anim.mkdir(parents=True)
    # start: the embed opens on the analysis (sst_anim.html ?start=manifest), not on its default last frame
    mani = {"ver": int(time.time()), "init": want, "model": None, "default": None, "start": 0, "regions": {}}
    src = source(a.date, hh) if storms else None
    if storms and src is None:
        TC.report(
            f"TC maps {a.date} {hh}Z: neither {SOURCES[0][1]} nor {SOURCES[1][1]} is complete on the Google mirror",
            "error",
        )
    if src is not None:
        model, lab = src
        mani["model"] = lab
        sect = {}
        for s in storms:
            la = [v for m in mem["storms"][s["slug"]].values() for t in m for v in t["lat"]]
            lo = [v for m in mem["storms"][s["slug"]].values() for t in m for v in t["lon"]]
            det = s["deterministic"].get(lab)
            if det:
                la, lo = la + det["lat"], lo + det["lon"]
            b = TC.track_sector(la, lo)
            sect[s["slug"]] = (*b, TC.sector_aspect(*b))
            (anim / s["slug"]).mkdir()
            mani["regions"][s["slug"]] = {"label": s["name"], "frames": []}
        mani["default"] = storms[0]["slug"]
        p_pa = np.array(DT.LEVS, float) * 100.0
        for k, h in enumerate(STEPS):
            if a.max_steps and k >= a.max_steps:
                break
            T, U, V, lat, lon = DT.step_fields(model, a.date, hh, h)
            pv, th = DT.pv_step(T, U, V, p_pa, lat, lon)
            dt_ = dynamic_tropopause(pv, th, U, V, p_pa)
            try:
                adv = outflow(U, V, pv, lat, lon)
            except Exception as e:  # noqa: BLE001  the maps stand without the outflow contours
                print(f"  outflow at +{h} h failed ({e!r})", flush=True)
                adv = None
            valid = init + pd.Timedelta(hours=h)
            for s in storms:
                fp = anim / s["slug"] / f"F{k:02d}.webp"
                frame(
                    s,
                    mem["storms"][s["slug"]],
                    s["deterministic"].get(lab),
                    sect[s["slug"]],
                    (lat, lon, dt_["theta"], dt_["p"], adv),
                    init,
                    h,
                    lab,
                    fp,
                )
                mani["regions"][s["slug"]]["frames"].append(
                    {
                        "idx": k,
                        "file": fp.name,
                        "date": valid.strftime("%Y-%m-%d"),
                        "label": ("analysis" if h == 0 else f"+{h} h") + f" · {valid:%a %d %b %HZ}",
                    }
                )
            if k % 8 == 0:
                print(f"    +{h:3d} h done ({time.time() - t0:.0f} s)", flush=True)
    (anim / "tcdt_manifest.json").write_text(json.dumps(mani, separators=(",", ":")))
    n = sum(len(r["frames"]) for r in mani["regions"].values())
    print(
        f"wrote {n} frames for {len(mani['regions'])} storms ({mani['model']}) in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
