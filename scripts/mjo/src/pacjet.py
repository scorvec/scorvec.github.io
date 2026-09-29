#!/usr/bin/env python3
"""North Pacific jet: where the 250 hPa jet will be over the next two weeks, member by member (redesigned 2026-09-28).

Per cycle and model (AIFS-ENS in strat.yml, IFS-ENS in strat-ifs.yml; NO NEW DOWNLOADS - both read 250 hPa winds the
stratosphere jobs already pull from the Google Cloud mirror):

  assets/sst/pacjet_{model}_wk1.webp, _wk2.webp   week-1 / week-2 maps, 100E-60W: ensemble-mean wind speed with
                                                   isotachs, the mean jet axis against the ERA5 normal axis, and the
                                                   share of member-days with a >= 50 m/s core against its normal
  assets/sst/anim/pacjet_{model}/F00..F15.webp    the same two panels day by day (+ pacjet_{model}_manifest.json)
  assets/sst/pacjet_{model}_axis.webp             each member's jet axis on days 3, 7, 10 and 14 against the ERA5 axis
  assets/sst/pacjet_{model}_phase.webp            the Winters et al. (2019) jet-phase diagram (Oct-Apr only), phase
                                                   odds by day and the measured skill of the phase indices by lead
  assets/sst/data/pacjet_{model}.json             the numbers behind them
  assets/sst/pacjet_torque.webp                   (--torque, AIFS only) the research view: ERA5 lagged regression of
                                                   the jet indices on the Himalayan mountain torque, and what it implies
                                                   for this cycle only where it is significant

Members: AIFS-ENS control + perturbed 1-25 (u from the 12-level AAM pull, v from the E-P flux pull - the 250 hPa level
is only fetched for 25 perturbed members); IFS-ENS perturbed 1-50 (qbo_duct.py --model ifs streams u/v at every level
and leaves the 250 hPa sector here via PACJET_IFS_DIR). Every field is reduced to the same 0.5 deg grid by conservative
averaging (pacjet_core). Each model's lead-dependent drift of the 250 hPa zonal wind was measured on its own archived
forecasts (build_pacjet_hindcast.py; data/reference/pacjet_drift.nc, smoothed and noise-shrunk) and is subtracted only
where that lowered the cross-validated error of the phase indices (IFS-ENS yes, AIFS-ENS no, as of 2026-09-28).

    python src/pacjet.py --model aifs --date 20260928 --time 00
    python src/pacjet.py --model ifs  --date 20260928 --time 00 --ifs-dir /tmp/pacjet_ifs
    python src/pacjet.py --model aifs --date 20260928 --time 00 --torque-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import pacjet_core as PC                                                    # noqa: E402

REF = HERE.parent / "data" / "reference"
F_REF, F_DRIFT, F_SKILL, F_TORQ = (REF / "pacjet_ref.nc", REF / "pacjet_drift.nc", REF / "pacjet_skill.json",
                                   REF / "pacjet_torque.json")
SITE = REPO / "assets" / "sst"
HIST = SITE / "data" / "pacjet_analysis.json"
LABEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
MCOL = {"aifs": "#1b365d", "ifs": "#1e7b52"}
INK, MUTED = "#1a1a1a", "#6f6b64"
NDAY = 16                                                                   # steps 0, 24, ..., 360 h
TAIL = 20                                                                   # analysis days drawn before day 0
SPD_LEV = np.arange(20, 91, 5)
SPD_COL = ["#e8f1fa", "#c6dbef", "#9ecae1", "#6baed6", "#3a8fc9", "#2a9d8f", "#57b86a", "#a6d854", "#f2e34c",
           "#f9b233", "#f37b2d", "#e0442f", "#b8235a", "#7d1d6f"]
P_LEV = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
P_COL = ["#fff3c4", "#fde28a", "#fdc458", "#fca03f", "#f7792f", "#e8512a", "#cc2f2f", "#a41d3f", "#7a1450", "#4d0f52"]
PH_COL = {"extension": "#c0392b", "poleward": "#2e86c1", "retraction": "#7d5ba6", "equatorward": "#d68910",
          "neutral": "#bdbdbd"}


# ── members ──────────────────────────────────────────────────────────────────
def _read_grib(path, short, want_level=PC.LEVEL):
    """{(number, step_h): (nj, ni) values} for one parameter at one level, plus (lat, lon)."""
    import eccodes as ec
    out, lat, lon = {}, None, None
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if ec.codes_get(h, "shortName") != short or int(ec.codes_get(h, "level")) != want_level:
                    continue
                nj, ni = ec.codes_get(h, "Nj"), ec.codes_get(h, "Ni")
                if lat is None:
                    la0 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                    la1 = ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
                    lo0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                    lat = np.linspace(la0, la1, nj); lon = (lo0 + (360.0 / ni) * np.arange(ni)) % 360
                num = int(ec.codes_get(h, "number")) if ec.codes_get(h, "dataType") == "pf" else 0
                step = int(ec.codes_get(h, "endStep"))
                out[(num, step)] = ec.codes_get_values(h).reshape(nj, ni)
            finally:
                ec.codes_release(h)
    return out, lat, lon


def load_aifs(date, time_):
    """(members, u, v) with u, v (member, day 0..15, 141, 401) on the work grid, from the shared store cache ONLY."""
    import store as E
    cyc = E.Cycle(date, time_); S = tuple(E.STEPS); L14 = tuple(E.LEVELS_AAM)
    files = {"u": [E.path(cyc, E.Spec("aifs-ens", "cf", "u", "pl", E.LEVELS_AAM_REST, S)),
                   E.path(cyc, E.Spec("aifs-ens", "pf", "u", "pl", E.LEVELS_AAM_REST, S, E.AAM_PF_MEMBERS))],
             "v": [E.path(cyc, E.Spec("aifs-ens", "cf", "v", "pl", L14, S)),
                   E.path(cyc, E.Spec("aifs-ens", "pf", "v", "pl", L14, S, E.AAM_PF_MEMBERS))]}
    fields, li = {}, None
    for par, paths in files.items():
        got = {}
        for p in paths:
            if not p.exists():
                raise SystemExit(f"AIFS-ENS {par}{PC.LEVEL} not in the store cache ({p.name}); nothing is fetched here")
            d, lat, lon = _read_grib(p, par)
            if li is None:
                li, lj = PC.native_slices(lat, lon)
            got.update(d)
        fields[par] = got
    keys = sorted(set(fields["u"]) & set(fields["v"]))
    mem = sorted({k[0] for k in keys})
    steps = [24 * d for d in range(NDAY)]
    mem = [m for m in mem if all((m, s) in fields["u"] and (m, s) in fields["v"] for s in steps)]
    U = np.stack([np.stack([PC.to_work(fields["u"][(m, s)], li, lj) for s in steps]) for m in mem])
    V = np.stack([np.stack([PC.to_work(fields["v"][(m, s)], li, lj) for s in steps]) for m in mem])
    return np.array(mem), U, V


def load_ifs(ifs_dir):
    """The 250 hPa sector qbo_duct.py --model ifs left in ifs_dir (one npz per step, members in number order)."""
    parts = []
    for d in range(NDAY):
        p = Path(ifs_dir) / f"step_{24 * d:03d}.npz"
        if not p.exists():
            raise SystemExit(f"IFS-ENS step {24 * d} h missing in {ifs_dir} (did the E-P flux stream run?)")
        z = np.load(p)
        parts.append((z["members"], z["u"].astype("float32"), z["v"].astype("float32")))
    mem = parts[0][0]
    common = sorted(set(mem).intersection(*[set(p[0]) for p in parts]))
    pick = lambda p: [int(np.flatnonzero(p[0] == m)[0]) for m in common]
    U = np.stack([p[1][pick(p)] for p in parts], axis=1)
    V = np.stack([p[2][pick(p)] for p in parts], axis=1)
    return np.array(common), U, V


# ── drift ────────────────────────────────────────────────────────────────────
def drift_correction(model, init):
    """(15, 141, 401) m/s to SUBTRACT from u at days 1-15, a label for the figures, and the details for the JSON.
    The correction is the model's all-season 250 hPa u drift (pacjet_drift.nc), applied only when build_pacjet_hindcast
    found that, cross-validated, it lowers the error of the phase indices in the season and over the year; otherwise
    the drift is reported as measured and within the noise, and nothing is subtracted."""
    zero = np.zeros((NDAY - 1, PC.WORK_LAT.size, PC.WORK_LON.size), "float32")
    if not F_DRIFT.exists():
        return zero, "no drift correction (not yet measured)", {}
    ds = xr.open_dataset(F_DRIFT)
    key = f"corr_all_{model}"
    if key not in ds:
        return zero, "no drift correction (accruing)", {}
    n = int(ds[key].attrs.get("cases", 0))
    sk = json.loads(F_SKILL.read_text()).get("models", {}).get(model, {}) if F_SKILL.exists() else {}
    info = {"field": key, "cases": n, "applied": bool(ds[key].attrs.get("apply", 0)), "cv_mse_change_pct": sk.get("cv_mse_change_pct")}
    if not info["applied"]:
        return zero, f"drift measured on {n} archived runs and within the noise: no correction", info
    c = ds[key].interp(latitude=PC.WORK_LAT, longitude=PC.WORK_LON, method="linear",
                       kwargs={"fill_value": None}).fillna(0.0).values.astype("float32")
    info["mean_abs_d15"] = round(float(np.abs(c[-1]).mean()), 2)
    return c, f"drift-corrected (lead-dependent 250 hPa u drift from {n} archived runs)", info


# ── indices ──────────────────────────────────────────────────────────────────
def basis_month(init):
    """The EOF basis of the whole forecast: the month of its midpoint (init + 7 d), so a late-September run is read on
    the October patterns and one basis serves every lead (a trajectory must not change coordinates half-way)."""
    return int((init + pd.Timedelta(days=7)).month)


def pcs_of(u_work, valid, ref, month):
    anom = PC.to_eof(u_work) - PC.harm_eval(ref.u_coef.values, valid.dayofyear.values)
    return PC.project(anom, ref.proj.sel(month=month).values)


HIST_SEED = REF / "pacjet_analysis_seed.json"                               # 90 analyses to 2026-09-28 from the Google mirror


def update_history(an_pcs_all, day, keep=90):
    """Append the analysis (day 0) PCs on all twelve monthly bases to the committed tail file (started from the seed)."""
    src = HIST if HIST.exists() else HIST_SEED
    h = json.loads(src.read_text()) if src.exists() else {"days": {}}
    h["days"][day] = np.round(an_pcs_all, 3).tolist()
    ks = sorted(h["days"])[-keep:]
    h["days"] = {k: h["days"][k] for k in ks}
    h["note"] = ("ECMWF operational analysis (AIFS-ENS control, 00Z step 0) projected on each month's 250 hPa jet-phase "
                 "basis: days -> 12 x [PC1, PC2] (Jan..Dec)")
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(h, separators=(",", ":")))
    return h


def tail_pcs(month, before, n=TAIL):
    src = HIST if HIST.exists() else HIST_SEED
    if not src.exists():
        return pd.DatetimeIndex([]), np.zeros((0, 2))
    h = json.loads(src.read_text())["days"]
    ks = [k for k in sorted(h) if pd.Timestamp(k) < before][-n:]
    return pd.DatetimeIndex(ks), np.array([h[k][month - 1] for k in ks]).reshape(-1, 2)


# ── drawing helpers ──────────────────────────────────────────────────────────
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    return plt


def _map(fig, rect, title=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(PC.MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#333", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.35, edgecolor="#555", zorder=4)
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#777", zorder=4)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False,
                      xlocs=range(-180, 181, 20), ylocs=range(10, 81, 10))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9, "color": INK}
    if title:
        ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", color=INK)
    return ax


def _geom(n_maps, map_w=11.4, lat=(PC.MAP_EXTENT[2], PC.MAP_EXTENT[3]), top=1.25, gap=0.62, bot=0.45, left=0.55):
    """Figure geometry from the map aspect (no letterbox): -> (W, H, [rects])."""
    mh = map_w * (lat[1] - lat[0]) / (PC.MAP_EXTENT[1] - PC.MAP_EXTENT[0])
    W = left + map_w + 1.05
    H = top + n_maps * mh + (n_maps - 1) * gap + bot
    rects = [[left / W, (bot + (n_maps - 1 - k) * (mh + gap)) / H, map_w / W, mh / H] for k in range(n_maps)]
    return W, H, rects


def _cbar(fig, cf, rect, W, H, label, ticks=None):
    x0, y0, w, h = rect
    cax = fig.add_axes([x0 + w + 0.12 / W, y0 + 0.05 * h, 0.16 / W, 0.9 * h])
    cb = fig.colorbar(cf, cax=cax, ticks=ticks)
    cb.set_label(label, fontsize=10); cb.ax.tick_params(labelsize=9)


def _head(fig, W, H, title, sub):
    fig.text(0.55 / W, 1 - 0.14 / H, title, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.55 / W, 1 - 0.47 / H, textwrap.fill(sub, int(W * 12.6)), fontsize=9.6, va="top", color=MUTED, linespacing=1.3)


def _foot(fig, W, H, text):
    fig.text(0.55 / W, 0.1 / H, text, fontsize=8.6, va="bottom", color=MUTED)


def _axis_line(ax, lon, ax_lat, smooth=True, **kw):
    import cartopy.crs as ccrs
    step = float(lon[1] - lon[0])
    ax_lat = PC.smooth_axis(ax_lat, n=(7 if step < 1 else 3)) if smooth else ax_lat
    for seg in PC.axis_segments(lon, ax_lat):
        ax.plot(seg[:, 0], seg[:, 1], transform=ccrs.PlateCarree(), **kw)


def two_panel(model, nmem, spd, pcore, axis_mean, axis_norm, freq_norm, title, sub, foot, out, dpi=100):
    """Mean speed (fill + isotachs + axes) above the >= 50 m/s share (fill + the normal-frequency contour)."""
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe
    from matplotlib.colors import BoundaryNorm, ListedColormap
    plt = _mpl()
    W, H, rects = _geom(2, top=1.42)
    fig = plt.figure(figsize=(W, H))
    lon, lat = PC.WORK_LON, PC.WORK_LAT
    cmap = ListedColormap(SPD_COL); cmap.set_over("#4a1060")
    ax = _map(fig, rects[0], "Ensemble-mean 250 hPa wind speed (m/s) · isotachs every 10 m/s")
    cf = ax.contourf(lon, lat, spd, levels=SPD_LEV, cmap=cmap, norm=BoundaryNorm(SPD_LEV, cmap.N), extend="max",
                     transform=ccrs.PlateCarree(), zorder=1)
    cs = ax.contour(lon, lat, spd, levels=[30, 40, 50, 60, 70, 80], colors="#222", linewidths=[0.5, 0.6, 0.9, 0.9, 1.0, 1.1],
                    transform=ccrs.PlateCarree(), zorder=3)
    ax.clabel(cs, fmt="%d", fontsize=8, inline_spacing=2)
    halo = [pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()]
    _axis_line(ax, PC.EOF_LON, axis_norm, color="k", lw=1.8, ls=(0, (5, 3)), zorder=6, path_effects=halo)
    _axis_line(ax, lon, axis_mean, color="#c2185b", lw=2.4, zorder=7, path_effects=halo)
    ax.plot([], [], color="#c2185b", lw=2.4, label="jet axis, ensemble mean")
    ax.plot([], [], color="k", lw=1.8, ls=(0, (5, 3)), label="ERA5 1991–2020 normal axis")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="none")
    _cbar(fig, cf, rects[0], W, H, "m/s", ticks=SPD_LEV[::2])
    pmap = ListedColormap(P_COL)
    ax2 = _map(fig, rects[1], f"Chance of a jet core ≥ {PC.CORE:.0f} m/s (share of the {nmem} members)")
    cf2 = ax2.contourf(lon, lat, 100 * pcore, levels=P_LEV, cmap=pmap, norm=BoundaryNorm(P_LEV, pmap.N),
                       transform=ccrs.PlateCarree(), zorder=1)
    if freq_norm is not None:
        ax2.contour(PC.EOF_LON, PC.EOF_LAT, 100 * freq_norm, levels=[20], colors="k", linewidths=1.5, linestyles=[(0, (5, 3))],
                    transform=ccrs.PlateCarree(), zorder=5)
        ax2.plot([], [], color="k", lw=1.5, ls=(0, (5, 3)), label=f"normal: ≥ {PC.CORE:.0f} m/s on 20% of days (ERA5)")
        ax2.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="none")
    _cbar(fig, cf2, rects[1], W, H, "% of members")
    _head(fig, W, H, title, sub)
    _foot(fig, W, H, foot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


# ── products ─────────────────────────────────────────────────────────────────
def render_weeks(model, init, valid, U, V, ref, dlabel, anim_dir, manifest, out_dir):
    spd = np.hypot(U, V)                                                   # (m, 16, lat, lon)
    nmem = spd.shape[0]
    core = spd >= PC.CORE
    foot = (f"ECMWF {LABEL[model]} open data (CC BY 4.0), 250 hPa u and v, {nmem} members, 0.5° · normals: ERA5 1991–2020 "
            f"(WeatherBench 2) · scorvec.com")
    summ = {}
    for wk, (d0, d1) in (("wk1", (1, 7)), ("wk2", (8, 14))):
        sl = slice(d0, d1 + 1)
        ms = spd[:, sl].mean((0, 1)); pc = core[:, sl].mean((0, 1))
        mid = valid[(d0 + d1) // 2]
        axis_norm = ref.axis_clim.sel(doy=min(mid.dayofyear, 366)).values
        fn = ref.freq50.sel(month=mid.month).values
        title = (f"North Pacific jet, week {wk[-1]} (days {d0}–{d1}, {valid[d0]:%-d %b}–{valid[d1]:%-d %b}) — {LABEL[model]} "
                 f"{nmem} members, init {init:%Y-%m-%d %HZ}")
        sub = (f"250 hPa, {dlabel}. Top: the week's ensemble-mean wind speed and isotachs; magenta the jet axis (latitude of "
               f"the fastest wind at each longitude, where it reaches {PC.AXIS_MIN:.0f} m/s) against the ERA5 normal axis for "
               f"{mid:%-d %b} (dashed). Bottom: the share of member-days with wind ≥ {PC.CORE:.0f} m/s at each point, "
               "against the normal 20% frequency line.")
        two_panel(model, nmem, ms, pc, PC.jet_axis(ms), axis_norm, fn, title, sub, foot, out_dir / f"pacjet_{model}_{wk}.webp")
        summ[wk] = {"max_mean_speed": round(float(ms.max()), 1),
                    "core_area_frac": round(float((pc >= 0.5).mean()), 3),
                    "axis_mean_lat_150E_150W": round(float(np.nanmean(PC.jet_axis(ms)[(PC.WORK_LON >= 150) & (PC.WORK_LON <= 210)])), 1)}
    if anim_dir:
        anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
        for old in anim_dir.glob("F*.webp"):
            old.unlink()
        frames = []
        for d in range(NDAY):
            ms = spd[:, d].mean(0); pc = core[:, d].mean(0)
            axis_norm = ref.axis_clim.sel(doy=min(valid[d].dayofyear, 366)).values
            title = f"North Pacific jet, day {d} (valid {valid[d]:%a %-d %b %HZ}) — {LABEL[model]} {nmem} members, init {init:%Y-%m-%d %HZ}"
            sub = (f"250 hPa, {dlabel if d else 'day 0 (the initial state)'}. Top: ensemble-mean wind speed, isotachs, the mean "
                   "jet axis (magenta) and the ERA5 normal axis (dashed). Bottom: share of members with a ≥ 50 m/s core.")
            fp = anim_dir / f"F{d:02d}.webp"
            two_panel(model, nmem, ms, pc, PC.jet_axis(ms), axis_norm, ref.freq50.sel(month=valid[d].month).values,
                      title, sub, foot, fp, dpi=90)
            frames.append({"idx": d, "file": fp.name, "date": valid[d].strftime("%Y-%m-%d"),
                           "label": f"day {d} · valid {valid[d]:%a %b %d} · max {ms.max():.0f} m/s"})
        mani = {"ver": int(time.time()), "days": NDAY,
                "regions": {f"pacjet_{model}": {"label": f"North Pacific jet, 250 hPa ({LABEL[model]})", "n_frames": len(frames),
                                                "frames": frames}}}
        Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest).write_text(json.dumps(mani))
        print(f"  loop: {len(frames)} frames -> {anim_dir}", flush=True)
    return summ


def render_axis(model, init, valid, U, V, ref, dlabel, out):
    import cartopy.crs as ccrs
    plt = _mpl()
    days = (3, 7, 10, 14)
    lat_rng = (15.0, 72.0)
    W, H, rects = _geom(len(days), lat=lat_rng, top=1.45, gap=0.5)
    fig = plt.figure(figsize=(W, H))
    spd = np.hypot(U, V)
    nmem = spd.shape[0]
    lon = PC.WORK_LON
    summ = {}
    for k, d in enumerate(days):
        ax = _map(fig, rects[k])
        ax.set_extent((PC.MAP_EXTENT[0], PC.MAP_EXTENT[1], lat_rng[0], lat_rng[1]), crs=ccrs.PlateCarree())
        axes = PC.jet_axis(spd[:, d])                                        # (m, lon)
        m = valid[d].month
        lo, hi = ref.axis_p10.sel(month=m).values, ref.axis_p90.sel(month=m).values
        okb = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(PC.EOF_LON[okb], lo[okb], hi[okb], color="#9e9e9e", alpha=0.28, lw=0, transform=ccrs.PlateCarree(), zorder=2)
        for i in range(nmem):
            _axis_line(ax, lon, axes[i], color=MCOL[model], lw=0.8, alpha=0.45, zorder=5)
        defined = np.isfinite(axes).mean(0)
        with np.errstate(all="ignore"):
            med = np.where(defined >= 0.5, np.nanmedian(axes, 0), np.nan)
        _axis_line(ax, PC.EOF_LON, ref.axis_clim.sel(doy=min(valid[d].dayofyear, 366)).values, color="k", lw=2.0,
                   ls=(0, (5, 3)), zorder=7)
        _axis_line(ax, lon, med, color="#f39c12", lw=2.6, zorder=8)
        ax.set_title(f"Day {d} · {valid[d]:%a %-d %b}", fontsize=11.5, loc="left", fontweight="bold", color=INK)
        if k == 0:
            ax.plot([], [], color=MCOL[model], lw=1.2, label="each member's axis")
            ax.plot([], [], color="#f39c12", lw=2.6, label="member median (where ≥ half have a core)")
            ax.plot([], [], color="k", lw=2.0, ls=(0, (5, 3)), label="ERA5 normal axis")
            ax.fill_between([], [], [], color="#9e9e9e", alpha=0.35, label="ERA5 10–90% range for the month")
            ax.legend(loc="lower left", fontsize=8.8, ncol=2, framealpha=0.92, edgecolor="none")
        east = (lon >= 150) & (lon <= 240)
        with np.errstate(all="ignore"):
            summ[f"d{d}"] = {"median_lat_150E_120W": round(float(np.nanmean(med[east])), 1),
                             "members_with_core_east_of_180": round(float(np.mean(np.isfinite(axes[:, (lon >= 180) & (lon <= 230)]).mean(1) > 0.5)), 2)}
    nday_def = f"{PC.AXIS_MIN:.0f} m/s"
    _head(fig, W, H, f"North Pacific jet axis, members on days 3, 7, 10 and 14 — {LABEL[model]} {nmem} members, init {init:%Y-%m-%d %HZ}",
          f"250 hPa, {dlabel}. The axis is the latitude of the fastest wind at each longitude between 15 and 70°N, drawn only "
          f"where it reaches {nday_def}; a line breaks where the fastest wind jumps between branches of a split flow. "
          "Spread of the lines = forecast uncertainty in where the jet (and the storm track under it) will be.")
    _foot(fig, W, H, f"ECMWF {LABEL[model]} open data (CC BY 4.0) · normal axis and range: ERA5 1991–2020 daily 250 hPa wind (WeatherBench 2) · scorvec.com")
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    return summ


def season_label(ref):
    """'Oct–May' for a season that wraps the year end."""
    ok = [bool(ref.in_season.sel(month=m)) for m in range(1, 13)]
    start = next(m for m in range(1, 13) if ok[m - 1] and not ok[(m - 2) % 12])
    end = next(m for m in range(1, 13) if ok[m - 1] and not ok[m % 12])
    return f"{pd.Timestamp(2001, start, 1):%b}–{pd.Timestamp(2001, end, 1):%b}"


def render_phase(model, init, valid, pcs, ref, month, dlabel, out, other=None):
    """Phase diagram + phase odds by day + measured skill; a season note when the basis month is out of season."""
    plt = _mpl()
    in_season = bool(ref.in_season.sel(month=month))
    nmem = pcs.shape[0]
    mname = pd.Timestamp(2001, month, 1).strftime("%B")
    if not in_season:
        W, H = 13.0, 4.4
        fig = plt.figure(figsize=(W, H))
        cap = float(ref.capture.sel(month=month)); cs = ref.plane_cos.sel(month=month).values
        fig.text(0.04, 0.9, f"North Pacific jet phases — not shown in {mname} ({LABEL[model]} init {init:%Y-%m-%d %HZ})",
                 fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(0.04, 0.74, textwrap.fill(
            f"The jet phases (extension / retraction, poleward / equatorward shift; Winters et al. 2019) are the two leading "
            f"patterns of the cold-season 250 hPa wind. In {mname} they no longer describe the jet's variability: in ERA5 "
            f"1991–2020 they capture only {cap:.0%} of what {mname}'s own two leading patterns capture (in season means ≥ 80%, "
            f"with the two planes' cosines ≥ 0.75; here {cs.min():.2f}), so a reading in σ would not mean the same thing. "
            f"The diagram is shown for forecasts centred in {season_label(ref)}; the maps and the jet-axis view carry the "
            "forecast in the meantime.", 150), fontsize=11.5, color=INK, va="top",
            linespacing=1.4)
        axm = fig.add_axes([0.04, 0.12, 0.92, 0.18])
        for m in range(1, 13):
            ok = bool(ref.in_season.sel(month=m))
            axm.bar(m, 1, color=("#2e86c1" if ok else "#e0e0e0"), edgecolor="white", width=0.95)
            axm.text(m, 0.5, pd.Timestamp(2001, m, 1).strftime("%b"), ha="center", va="center", fontsize=10.5,
                     color=("white" if ok else MUTED), fontweight=("bold" if m == month else "normal"))
        axm.set_xlim(0.5, 12.5); axm.axis("off")
        fig.text(0.04, 0.04, "blue: months whose sliding three-month EOFs are the cold-season extension/shift pair (ERA5 1991–2020)",
                 fontsize=9, color=MUTED)
        fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
        return {"in_season": False, "basis_month": month}
    W, H = 13.4, 7.6
    fig = plt.figure(figsize=(W, H))
    W, H = 13.4, 8.2
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0.05, 0.19, 0.64 * H / W, 0.64])
    lim = float(np.clip(np.nanpercentile(np.abs(pcs), 99) + 0.4, 2.5, 4.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(th), np.sin(th), color="#f2f2f2", zorder=0)
    ax.plot(np.cos(th), np.sin(th), color="#9e9e9e", lw=1)
    for a in (45, 135):
        x = np.cos(np.radians(a)) * np.array([1, lim * 1.5]); y = np.sin(np.radians(a)) * np.array([1, lim * 1.5])
        ax.plot(x, y, color="#9e9e9e", lw=0.8); ax.plot(-x, -y, color="#9e9e9e", lw=0.8)
    for name, (x, y, ha, va) in {"extension": (lim * 0.97, 0, "right", "center"), "poleward": (0, lim * 0.97, "center", "top"),
                                 "retraction": (-lim * 0.97, 0, "left", "center"), "equatorward": (0, -lim * 0.97, "center", "bottom")}.items():
        ax.text(x, y, PC.PHASE_LABEL[name], ha=ha, va=va, fontsize=11, fontweight="bold", color=PH_COL[name],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.text(0, 0, "neutral", ha="center", va="center", fontsize=9.5, color=MUTED)
    for i in range(nmem):
        ax.plot(pcs[i, :, 0], pcs[i, :, 1], color=MCOL[model], lw=0.5, alpha=0.12, zorder=2)
    for d, col in ((5, "#9ecae1"), (10, "#4292c6"), (15, "#08306b")):
        ax.scatter(pcs[:, d, 0], pcs[:, d, 1], s=16, color=col, edgecolor="white", lw=0.4, zorder=3, label=f"members, day {d}")
    mean = pcs.mean(0)
    tdays, tpc = tail_pcs(month, init.normalize())
    if len(tpc):
        ax.plot(np.r_[tpc[:, 0], mean[0, 0]], np.r_[tpc[:, 1], mean[0, 1]], color="#222", lw=1.6, marker="o", ms=3, zorder=4,
                label=f"analyses, last {len(tpc)} days")
    if other is not None:
        ax.plot(other["mean"][:, 0], other["mean"][:, 1], color=MCOL[other["model"]], lw=2.0, ls=(0, (4, 2)), zorder=5,
                label=f"{LABEL[other['model']]} mean ({other['n']} members)")
    ax.plot(mean[:, 0], mean[:, 1], color=MCOL[model], lw=2.8, marker="o", ms=4, zorder=6, label=f"{LABEL[model]} ensemble mean")
    for d in (0, 5, 10, 15):
        ax.annotate(f"d{d}", (mean[d, 0], mean[d, 1]), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold",
                    color=MCOL[model], zorder=7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#ccc", lw=0.6, zorder=0); ax.axvline(0, color="#ccc", lw=0.6, zorder=0)
    ax.set_xlabel("PC1 (σ): + extended, − retracted", fontsize=10.5); ax.set_ylabel("PC2 (σ): + poleward, − equatorward", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.4, frameon=False, columnspacing=1.2)
    ax.tick_params(labelsize=9.5)
    # phase odds by day
    ph = PC.phase_of(pcs[..., 0], pcs[..., 1])                            # (m, day)
    axp = fig.add_axes([0.57, 0.56, 0.41, 0.26])
    bottom = np.zeros(NDAY)
    odds = {}
    for k, name in [(-1, "neutral")] + list(enumerate(PC.PHASES)):
        f = (ph == k).mean(0) * 100
        odds[name] = np.round(f, 1).tolist()
        axp.bar(np.arange(NDAY), f, bottom=bottom, color=PH_COL[name], width=0.85, edgecolor="white", lw=0.5,
                label=PC.PHASE_LABEL[name])
        bottom += f
    axp.set_xlim(-0.6, NDAY - 0.4); axp.set_ylim(0, 100)
    axp.set_xticks(range(0, NDAY)); axp.set_xticklabels([str(d) if d % 2 == 0 else "" for d in range(NDAY)], fontsize=9)
    axp.set_ylabel("% of members", fontsize=10); axp.tick_params(labelsize=9)
    axp.set_title("Jet phase by forecast day 0–15 (share of members)", fontsize=11, loc="left", fontweight="bold", color=INK)
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8.6, frameon=False, handlelength=1.2)
    for s_ in ("top", "right"):
        axp.spines[s_].set_visible(False)
    # skill strip
    axs = fig.add_axes([0.57, 0.1, 0.41, 0.3])
    sk = json.loads(F_SKILL.read_text()) if F_SKILL.exists() else None
    mm = (sk or {}).get("models", {}).get(model, {})
    blk = mm.get("season") or mm.get("all")
    if blk:
        L = np.array(sk["leads"])
        for name, col, lab in (("pc1", "#c0392b", "PC1 extension"), ("pc2", "#2e86c1", "PC2 shift")):
            q = blk[name]
            axs.fill_between(L, q["r_lo"], q["r_hi"], color=col, alpha=0.18, lw=0)
            axs.plot(L, q["r"], color=col, lw=2, marker="o", ms=3, label=f"{lab}, {LABEL[model]}")
            axs.plot(L, q["pers_r"], color=col, lw=1.1, ls=(0, (3, 2)), alpha=0.9)
        axs.plot([], [], color="#555", lw=1.1, ls=(0, (3, 2)), label="persistence of the day-0 analysis")
        axs.axhline(0.6, color="#999", lw=0.7)
        axs.set_ylim(min(0, np.nanmin([min(blk[k]["pers_r"]) for k in ("pc1", "pc2")]) - 0.05), 1)
        axs.set_xlim(0.6, 15.4); axs.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        axs.set_xlabel("lead (days)", fontsize=10); axs.set_ylabel("correlation with the analysis", fontsize=10); axs.tick_params(labelsize=9)
        axs.legend(loc="lower left", fontsize=8.4, frameon=False)
        axs.set_title(f"Measured skill: {blk['n']} {'in-season ' if mm.get('season') else ''}runs, "
                      f"{pd.Timestamp(mm['first']):%b %Y}–{pd.Timestamp(mm['last']):%b %Y} (90% CI)", fontsize=10.2, loc="left",
                      fontweight="bold", color=INK)
    else:
        axs.axis("off"); axs.text(0.5, 0.5, "skill not yet measured for this model", ha="center", va="center", color=MUTED,
                                  transform=axs.transAxes)
    for s_ in ("top", "right"):
        axs.spines[s_].set_visible(False)
    fig.text(0.045, 0.975, f"North Pacific jet phase diagram — {LABEL[model]} {nmem} members, init {init:%Y-%m-%d %HZ}",
             fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.045, 0.935, textwrap.fill(
        f"250 hPa, {dlabel}. Axes: the two leading cold-season patterns of the 250 hPa wind over 10–80°N 100°E–120°W "
        f"(ERA5 1991–2020; Winters et al. 2019), in σ of the {mname} ± 1 month spread. Outside the grey circle a phase "
        f"is named by the nearest axis. Faint lines: each member, days 0–15; dots: the members on days 5, 10 and 15; black: "
        f"the analyses of the last {TAIL} days leading into day 0.", 165),
        fontsize=9.5, va="top", color=MUTED, linespacing=1.3)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    return {"in_season": True, "basis_month": month, "odds": odds}


# ── torque (research) ────────────────────────────────────────────────────────
def render_torque(init, pcs_mean, valid, out, torque_json):
    """ERA5 signed lagged regression of the jet indices on the Himalayan mountain torque (season window of the forecast's
    basis month), and the forecast implication only where the regression is significant after FDR."""
    plt = _mpl()
    import matplotlib.dates as mdates
    tr = json.loads(F_TORQ.read_text())
    month = basis_month(init)
    win = tr["windows"].get(str(month))
    W, H = 13.4, 7.4
    fig = plt.figure(figsize=(W, H))
    axr = fig.add_axes([0.06, 0.2, 0.40, 0.58])
    lags = np.array(tr["lags"])
    summ = {"month": month}
    if win is None:
        axr.axis("off"); axr.text(0.5, 0.5, f"no regression for {pd.Timestamp(2001, month, 1):%B}", transform=axr.transAxes, ha="center")
    else:
        for name, col, lab in (("pc1", "#c0392b", "PC1 extension"), ("pc2", "#2e86c1", "PC2 poleward shift")):
            q = win[name]
            b = np.array(q["beta"]); lo = np.array(q["lo"]); hi = np.array(q["hi"]); sig = np.array(q["sig"], bool)
            axr.fill_between(lags, lo, hi, color=col, alpha=0.15, lw=0)
            axr.plot(lags, b, color=col, lw=1.8, label=lab)
            axr.plot(lags[sig], b[sig], "o", color=col, ms=5.5, zorder=5)
        axr.axhline(0, color="#777", lw=0.8); axr.axvline(0, color="#999", lw=0.8, ls=":")
        axr.set_xlabel("lag (days; + = jet index after the torque)", fontsize=10.5)
        axr.set_ylabel("σ of jet index per σ of torque (= r)", fontsize=10.5)
        axr.legend(loc="upper left", fontsize=9.5, frameon=False)
        axr.set_title(f"ERA5 1991–2020, {win['label']}", fontsize=11.5, loc="left", fontweight="bold", color=INK)
        fig.text(0.06, 0.035, textwrap.fill(
            f"Shading: 90% moving-block bootstrap interval ({tr['block_days']}-day blocks, {tr['n_boot']} draws). Dots: significant with "
            f"the false discovery rate held at 10% over all lags and both indices. n = {win['n_days']} days (effective ≈ "
            f"{min(win['n_eff_lag1'])} from the lag-1 autocorrelations).", 95), fontsize=8.8, color=MUTED, va="bottom")
        axr.tick_params(labelsize=9.5)
        for s_ in ("top", "right"):
            axr.spines[s_].set_visible(False)
    # forecast side
    axt = fig.add_axes([0.55, 0.6, 0.42, 0.18]); axf = fig.add_axes([0.55, 0.2, 0.42, 0.3], sharex=axt)
    tq = json.loads(Path(torque_json).read_text()) if torque_json and Path(torque_json).exists() else None
    tz = None
    if tq and "Himalaya/Tibet" in tq.get("ranges", {}):
        tv = pd.to_datetime(tq["valid"]); s1 = tq.get("sd", {}).get("Himalaya/Tibet") or np.nan
        tz = np.array(tq["ranges"]["Himalaya/Tibet"], float) / s1
        axt.axhspan(-1, 1, color="#000", alpha=0.05); axt.axhline(0, color="#777", lw=0.7)
        axt.plot(tv, tz, color="#8d5524", lw=2.2, marker="o", ms=3)
        axt.set_ylabel("σ", fontsize=10); axt.tick_params(labelsize=9, labelbottom=False)
        axt.set_title(f"Himalaya/Tibet torque forecast, AIFS-ENS mean (init {pd.Timestamp(tq['init'][:10]):%-d %b} {tq['init'][-3:]})",
                      fontsize=11, loc="left", fontweight="bold", color=INK)
        summ["torque_peak_sigma"] = round(float(tz[np.nanargmax(np.abs(tz))]), 2)
    else:
        axt.axis("off"); axt.text(0.5, 0.5, "torque forecast not available", ha="center", transform=axt.transAxes, color=MUTED)
    axf.axhline(0, color="#777", lw=0.7)
    notes = []
    summ["implied"] = {}
    for k, (name, col, lab) in enumerate((("pc1", "#c0392b", "PC1"), ("pc2", "#2e86c1", "PC2"))):
        axf.plot(valid, pcs_mean[:, k], color=col, lw=2.2, marker="o", ms=3, label=f"{lab}, AIFS-ENS mean")
        if win is None or tz is None:
            continue
        q = win[name]; sig = np.array(q["sig"], bool); b = np.array(q["beta"])
        pos = [(i, L) for i, L in enumerate(lags) if L >= 1 and sig[i]]
        if not pos:
            notes.append(f"{lab}: not significant at any positive lag in {win['label']}, no implication drawn.")
            summ["implied"][name] = None
            continue
        i, L = max(pos, key=lambda z: abs(b[z[0]]))
        imp_t = tv + pd.Timedelta(days=int(L)); imp = b[i] * tz
        keep = imp_t <= valid[-1]
        axf.plot(imp_t[keep], imp[keep], color=col, lw=1.6, ls=(0, (4, 2)), label=f"{lab} implied: {b[i]:+.2f} × torque {L} d earlier")
        summ["implied"][name] = {"lag": int(L), "beta": round(float(b[i]), 3), "r2": round(float(b[i] ** 2), 3)}
        notes.append(f"{lab}: β(+{L} d) = {b[i]:+.2f} [{q['lo'][i]:+.2f}, {q['hi'][i]:+.2f}], {b[i] ** 2:.0%} of the variance.")
    axf.legend(loc="upper left", fontsize=8.6, frameon=False, ncol=2)
    axf.set_ylabel("σ", fontsize=10); axf.tick_params(labelsize=9)
    axf.xaxis.set_major_locator(mdates.DayLocator(interval=3)); axf.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    axf.set_title("Jet phase indices and the part the torque alone implies", fontsize=11, loc="left", fontweight="bold", color=INK)
    if notes:
        fig.text(0.55, 0.035, "\n".join(notes), fontsize=8.8, color=INK, va="bottom")
    for a_ in (axt, axf):
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.text(0.06, 0.975, f"Research view: does the Himalayan mountain torque steer the North Pacific jet? — init {init:%Y-%m-%d %HZ}",
             fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.06, 0.935, textwrap.fill(
        "A burst of mountain torque over the Himalaya and Tibet changes the atmosphere's angular momentum and has been linked to "
        "the East Asian jet downstream. Left: the ERA5 answer, as a signed regression of each jet index on the torque at every lag "
        "(both standardised, so the slope is also the correlation). Right: this cycle's torque forecast and, only where the regression "
        "is significant, the jet anomaly it would imply on its own.", 170), fontsize=9.5, va="top", color=MUTED, linespacing=1.3)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    return summ


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["aifs", "ifs"], required=True)
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--ifs-dir", default=os.environ.get("PACJET_IFS_DIR", "/tmp/pacjet_ifs"))
    ap.add_argument("--out-dir", default=str(SITE))
    ap.add_argument("--anim-dir", default=None, help="default: <out-dir>/anim/pacjet_<model>")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--torque", default=str(SITE / "data" / "torque_ranges.json"), help="AIFS only: the torque forecast")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = Path(a.out_dir)
    global HIST
    HIST = out_dir / "data" / "pacjet_analysis.json"
    ref = xr.open_dataset(F_REF).load()
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    valid = pd.DatetimeIndex([init + pd.Timedelta(days=d) for d in range(NDAY)])
    mem, U, V = load_aifs(a.date, a.time) if a.model == "aifs" else load_ifs(a.ifs_dir)
    print(f"  {LABEL[a.model]}: {len(mem)} members, u/v {U.shape} ({time.time() - t0:.0f}s)", flush=True)
    corr, dlabel, dinfo = drift_correction(a.model, init)
    U[:, 1:] -= corr[None]
    month = basis_month(init)
    pcs = pcs_of(U, valid, ref, month)                                     # (m, day, 2)
    if a.model == "aifs" and 0 in mem and a.time == "00":                  # the 00Z control at step 0 is the analysis
        u0 = U[list(mem).index(0), 0]
        an = np.stack([pcs_of(u0[None], valid[:1], ref, m)[0] for m in range(1, 13)])
        update_history(an, init.strftime("%Y-%m-%d"))
    anim = None if a.no_loop else Path(a.anim_dir or out_dir / "anim" / f"pacjet_{a.model}")
    doc = {"model": a.model, "init": init.strftime("%Y-%m-%dT%HZ"), "members": int(len(mem)), "drift": dinfo, "drift_label": dlabel,
           "basis_month": month, "valid": [v.strftime("%Y-%m-%d") for v in valid]}
    doc["weeks"] = render_weeks(a.model, init, valid, U, V, ref, dlabel, anim,
                                out_dir / "anim" / f"pacjet_{a.model}_manifest.json", out_dir)
    doc["axis"] = render_axis(a.model, init, valid, U, V, ref, dlabel, out_dir / f"pacjet_{a.model}_axis.webp")
    other = None
    om = "ifs" if a.model == "aifs" else "aifs"
    op = out_dir / "data" / f"pacjet_{om}.json"
    if op.exists():
        o = json.loads(op.read_text())
        if o.get("init") == doc["init"] and o.get("pc_mean") and o.get("basis_month") == month:
            other = {"model": om, "mean": np.array(o["pc_mean"]), "n": o.get("members")}
    doc["phase"] = render_phase(a.model, init, valid, pcs, ref, month, dlabel, out_dir / f"pacjet_{a.model}_phase.webp", other)
    doc["pc_mean"] = np.round(pcs.mean(0), 3).tolist()
    doc["pc_p10"] = np.round(np.percentile(pcs, 10, axis=0), 3).tolist()
    doc["pc_p90"] = np.round(np.percentile(pcs, 90, axis=0), 3).tolist()
    if a.model == "aifs":                                                  # the static impact figures, from the committed composites
        try:
            import build_pacjet_impacts as BI
            if BI.NPZ.exists():
                BI.render(out_dir)
        except Exception as e:                                             # noqa: BLE001
            print(f"  impact figures failed: {e}", flush=True)
    if a.model == "aifs" and F_TORQ.exists():
        try:
            doc["torque"] = render_torque(init, pcs.mean(0), valid, out_dir / "pacjet_torque.webp", a.torque)
        except Exception as e:                                             # noqa: BLE001
            print(f"  torque view failed: {e}", flush=True)
    doc["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / f"pacjet_{a.model}.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
