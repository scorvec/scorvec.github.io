#!/usr/bin/env python3
"""Draw the MJO impact composites (mjo.html, "MJO impacts by phase and month") from the committed reference
scripts/mjo/data/reference/mjo_impacts_site.npz (built on the laptop by mjo_impacts.py). Runs in Actions
(.github/workflows/mjo-impacts.yml); frames go to the frames branch, the manifest to main.

One animator region per (field, data, month, lag): 8 frames = RMM phases 1-8; plus a one-frame "all eight phases"
strip for the same selection. Only significant cells are shaded (the reference carries NaN elsewhere).

    python scripts/mjo/src/mjo_impacts_render.py [--procs 4] [--only tas_na]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SITE_ROOT", Path(__file__).resolve().parents[3]))
REF = ROOT / "scripts" / "mjo" / "data" / "reference"
ANIM = ROOT / "assets" / "mjo" / "impacts" / "anim"      # one directory for publish_frames_ci.sh; base must end in /anim
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
PHASE_WHERE = {1: "W. Hemisphere & Africa", 2: "Indian Ocean", 3: "Indian Ocean", 4: "Maritime Continent",
               5: "Maritime Continent", 6: "Western Pacific", 7: "Western Pacific", 8: "W. Hemisphere & Africa"}
FIELD = {
    "tas_na": dict(var="2 m temperature", unit="K", grid="na", cmap="RdBu_r",
                   lv=[-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3], ext=[-170, -50, 15, 75], cl=180 - 360),
    "tas_sa": dict(var="2 m temperature", unit="K", grid="sa", cmap="RdBu_r",
                   lv=[-2, -1.5, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 1.5, 2], ext=[-90, -30, -57, 15], cl=0),
    "pr_na": dict(var="precipitation", unit="mm/day", grid="na", cmap="BrBG",
                  lv=[-4, -3, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 3, 4], ext=[-170, -50, 15, 75], cl=180 - 360),
    "pr_sa": dict(var="precipitation", unit="mm/day", grid="sa", cmap="BrBG",
                  lv=[-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6], ext=[-90, -30, -57, 15], cl=0),
    "prpct_na": dict(var="precipitation, % of normal", unit="%", grid="na", cmap="BrBG",
                     lv=[-60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60], ext=[-170, -50, 15, 75], cl=180 - 360),
    "prpct_sa": dict(var="precipitation, % of normal", unit="%", grid="sa", cmap="BrBG",
                     lv=[-60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60], ext=[-90, -30, -57, 15], cl=0),
    "z500_nh": dict(var="500 hPa height", unit="gpm", grid="nh", cmap="RdBu_r",
                    lv=[-80, -60, -40, -30, -20, -10, 10, 20, 30, 40, 60, 80], ext=None, cl=180),
}
DATA = {"obs": "Observed", "cmip6": "CMIP6"}
MUTED = "#6f6b64"


def _ref():
    z = np.load(REF / "mjo_impacts_site.npz")
    meta = json.loads((REF / "mjo_impacts_site.json").read_text())
    return z, meta


def rid(name, data, m, lag, strip=False):
    return f"mi_{name}_{data}_m{m:02d}_l{lag:02d}" + ("_s" if strip else "")


def _note(z, meta, name, data, li, p, m):
    if data == "obs":
        ev = int(z[f"obs_{name}_events"][p, m])
        src = ("ERA5" if not name.startswith("pr") else "GPCP 1DD") + (", 1979–2024" if not name.startswith("pr") else ", 1997–2024")
        tested = ev >= 8
        return tested, (f"{src} · {ev} MJO events (BoM RMM, amplitude ≥ 1) · shaded only where significant: "
                        f"event-block bootstrap, false-discovery rate 10 % over the map")
    nm = int(z[f"cmip6_{name}_nmod"][li, p, m])
    tested = nm >= 5
    return tested, (f"mean of {nm} CMIP6 models, 1979–2014 · shaded only where robust: across-model t-test, "
                    f"false-discovery rate 10 % over the map, and ≥ 80 % of the models agree on the sign")


def _draw(ax, fig, lat, lon, F, spec, crs):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import cartopy.feature as cf
    from matplotlib.colors import ListedColormap
    lv = spec["lv"]; base = plt.get_cmap(spec["cmap"], len(lv) + 1)
    cols = [base(i) for i in range(len(lv) + 1)]
    cols[len(lv) // 2] = (1.0, 1.0, 1.0, 1.0)                           # the bin around zero is white, not a tint
    cmap = ListedColormap(cols)
    norm = BoundaryNorm([-1e9] + lv + [1e9], cmap.N)
    lo = np.where(lon > 180, lon - 360, lon) if spec["grid"] != "nh" else lon
    order = np.argsort(lo); lo = lo[order]; F = F[:, order]
    m = ax.pcolormesh(lo, lat, np.ma.masked_invalid(F), cmap=cmap, norm=norm, transform=crs, shading="nearest")
    ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
    if spec["grid"] != "nh":
        ax.add_feature(cf.BORDERS.with_scale("110m"), lw=0.35, edgecolor="#555")
    if spec["grid"] == "na":
        ax.add_feature(cf.STATES.with_scale("110m"), lw=0.2, edgecolor="#777")
    return m


def render_one(job):
    """One selection: 8 phase frames + the strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    name, data, m, li, lag = job
    z, meta = _ref()
    spec = FIELD[name]; g = spec["grid"]
    lat, lon = z[f"lat_{g}"], z[f"lon_{g}"]
    A = z[f"{data}_{name}"][li].astype(np.float32)                        # (phase, month, cells)
    proj = ccrs.PlateCarree(central_longitude=180 if g == "nh" else 0)
    crs = ccrs.PlateCarree()
    out = ANIM / rid(name, data, m + 1, lag); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("F*.webp"):
        f.unlink()
    frames = []
    asp = {"na": 2.0, "sa": 60 / 72, "nh": 4.0}[g]                    # map width / height (plate carree)
    W = {"na": 9.0, "sa": 6.4, "nh": 11.0}[g]
    mw = W - 0.3; mh = mw / asp; top, bot = 0.95, 0.72                     # inches: header, colour bar
    H = mh + top + bot
    for p in range(8):
        F = A[p, m].reshape(len(lat), len(lon))
        tested, note = _note(z, meta, name, data, li, p, m)
        fig = plt.figure(figsize=(W, H), dpi=100)
        ax = fig.add_axes([0.15 / W, bot / H, mw / W, mh / H], projection=proj)
        ax.set_extent(spec["ext"] or [-180, 180, 0, 90], crs=crs)
        mesh = _draw(ax, fig, lat, lon, F, spec, crs)
        if not tested:
            ax.text(0.5, 0.5, "Not tested (too few events)" if data == "obs" else "Not tested (fewer than 5 models)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12, color=MUTED,
                    bbox=dict(fc="white", ec="none", alpha=0.85))
        elif not np.isfinite(F).any():
            ax.text(0.5, 0.5, "No significant signal", transform=ax.transAxes, ha="center", va="center", fontsize=12,
                    color=MUTED, bbox=dict(fc="white", ec="none", alpha=0.85))
        fig.text(0.15 / W, 1 - 0.1 / H, f"{DATA[data]} · MJO phase {p + 1} ({PHASE_WHERE[p + 1]}) · {MONTHS[m]} · day +{lag}",
                 fontsize=12.5, fontweight="bold", va="top")
        fig.text(0.15 / W, 1 - 0.38 / H, f"{spec['var'].capitalize()} anomaly · " + note, fontsize=8.2, color=MUTED, va="top",
                 wrap=True)
        cax = fig.add_axes([0.2, 0.3 / H, 0.6, 0.14 / H])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=spec["lv"])
        cb.set_label(f"{spec['unit']}" if spec["unit"] != "%" else "% of the months' normal", fontsize=8.5, labelpad=1)
        cb.ax.tick_params(labelsize=7.5, pad=1)
        fp = out / f"F{p + 1:02d}.webp"
        fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
        plt.close(fig)
        frames.append({"idx": p, "file": fp.name, "date": f"p{p + 1}", "label": f"Phase {p + 1} · {PHASE_WHERE[p + 1]}"})
    # strip: all eight phases, grid shaped by the map aspect
    so = ANIM / rid(name, data, m + 1, lag, strip=True); so.mkdir(parents=True, exist_ok=True)
    cols, rows = (2, 4) if g == "nh" else (4, 2)
    pw = {"na": 3.6, "sa": 2.6, "nh": 5.6}[g]; ph_ = pw / asp; gap = 0.32
    SW = cols * pw + (cols - 1) * 0.1 + 0.2; SH = rows * (ph_ + gap) + 0.72 + 0.85
    fig = plt.figure(figsize=(SW, SH), dpi=90)
    for p in range(8):
        r, c = divmod(p, cols)
        x0 = 0.1 + c * (pw + 0.1); y0 = SH - 0.72 - (r + 1) * (ph_ + gap) + 0.02
        ax = fig.add_axes([x0 / SW, y0 / SH, pw / SW, ph_ / SH], projection=proj)
        ax.set_extent(spec["ext"] or [-180, 180, 0, 90], crs=crs)
        F = A[p, m].reshape(len(lat), len(lon))
        mesh = _draw(ax, fig, lat, lon, F, spec, crs)
        tested, _ = _note(z, meta, name, data, li, p, m)
        if not tested or not np.isfinite(F).any():
            ax.text(0.5, 0.5, "not tested" if not tested else "no significant signal", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, color=MUTED, bbox=dict(fc="white", ec="none", alpha=0.85))
        ax.set_title(f"Phase {p + 1} · {PHASE_WHERE[p + 1]}", fontsize=9, pad=2)
    _, note = _note(z, meta, name, data, li, 0, m)
    if data == "obs":
        note = note.split(" · ", 1)[0] + " · MJO events (BoM RMM, amplitude ≥ 1) · " + note.split(" · ")[-1]
    fig.text(0.1 / SW, 1 - 0.08 / SH, f"{DATA[data]} · {MONTHS[m]} · day +{lag}: {spec['var']} anomaly by MJO phase",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.1 / SW, 1 - 0.4 / SH, note, fontsize=8.2, color=MUTED, va="top")
    cax = fig.add_axes([0.3, 0.42 / SH, 0.4, 0.13 / SH])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=spec["lv"])
    cb.set_label(spec["unit"] if spec["unit"] != "%" else "% of the months' normal", fontsize=8.5, labelpad=1)
    cb.ax.tick_params(labelsize=7.5, pad=1)
    for f in so.glob("F*.webp"):
        f.unlink()
    fig.savefig(so / "F01.webp", dpi=90, facecolor="white", pil_kwargs={"quality": 78, "method": 6})
    plt.close(fig)
    return name, data, m, lag, frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    z, meta = _ref()
    lags = meta["lags"]
    names = [n for n in FIELD if f"obs_{n}" in z.files or f"cmip6_{n}" in z.files]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    jobs = [(n, d, m, li, lag) for n in names for d in DATA if f"{d}_{n}" in z.files
            for m in range(12) for li, lag in enumerate(lags)]
    t0 = time.time()
    # Natural Earth is downloaded on first use: fetch it HERE, once, before the pool, or the workers race to write
    # the same shapefiles and read each other's half-written copies (struct.error on the first Actions run).
    import cartopy.feature as cf
    for feat in (cf.COASTLINE, cf.BORDERS, cf.STATES):
        list(feat.with_scale("110m").geometries())
    regions = {}
    with ProcessPoolExecutor(a.procs) as ex:
        for name, data, m, lag, frames in ex.map(render_one, jobs, chunksize=4):
            lab = f"{DATA[data]} · {FIELD[name]['var']} ({FIELD[name]['grid'].upper()}) · {MONTHS[m]} · day +{lag}"
            regions[rid(name, data, m + 1, lag)] = {"label": lab, "n_frames": 8, "frames": frames}
            regions[rid(name, data, m + 1, lag, True)] = {"label": lab + " · all phases", "n_frames": 1,
                                                          "frames": [{"idx": 0, "file": "F01.webp", "date": "strip",
                                                                      "label": "All eight phases"}]}
    ANIM.mkdir(parents=True, exist_ok=True)
    man = {"ver": int(time.time()), "selectorLabel": "Selection", "regions": regions,
           "default": rid("tas_na", "cmip6", 1, lags[-1], True)}
    (ANIM / "mjo_impacts_manifest.json").write_text(json.dumps(man))
    print(f"{len(jobs)} selections, {len(regions)} regions in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
