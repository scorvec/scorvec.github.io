#!/usr/bin/env python3
"""Cyclone phase space (Hart 2003, MWR 131, 585-616) for the storms on the TC board, IFS control (HRES until May
2026).

Along IFS HRES's own tracker position for each storm on data/tc_board.json (tc_ens.py: the storms observed at t=0 or
carried by enough members of either ensemble; the HRES track from ECMWF's tracker), every 12 h to day 10:
  B      thermal asymmetry = mean (Z600 - Z900) in the semicircle RIGHT of the storm motion minus the LEFT one, within
         500 km (m; NH sign). ~0 = thermally symmetric (tropical); > 10 m = frontal.
  -VT_L  lower-troposphere thermal wind = slope of dZ = Zmax - Zmin (within 500 km) against ln p, 900-600 hPa.
  -VT_U  the same, 600-300 hPa.   > 0 = warm core, < 0 = cold core.
Grid and levels: Hart computed the parameters from ~1-degree analyses; the 0.25-degree HRES geopotential height is
therefore block-averaged to BLOCK_DEG first (the 500-km semicircle means and the max - min height perturbation are
otherwise set by grid-scale structure the threshold was not calibrated on). The open data carry 1000/925/850/700/600/500/
400/300 hPa but not 900 hPa, so Z900 is interpolated linearly in ln p between 925 and 850 hPa (the hypsometric equation
makes Z close to linear in ln p over that 75-hPa layer). B uses Z600 - Z900, -VT_L the levels 900 (interpolated)/850/
700/600 hPa and -VT_U 600/500/400/300 hPa: Hart's two layers, on the levels the open data have.
Parameters are smoothed with a 24-h running mean (Hart 2003).
Extratropical transition (Evans & Hart 2003, MWR 131, 909-925): onset when B exceeds ET_B, complete when -VT_L turns
negative, at or after onset. Here each condition must hold at consecutive 12-hourly points covering at least ET_HOLD_H
(et_times): one point over the threshold, such as a lopsided analysis at t=0, does not start a transition.
One deterministic run, one path through phase space per storm (the ensembles' spread is on the board). The path
continues past the tracker's last fix: ECMWF's tracker stops at (or near) extratropical transition, exactly when the
phase diagram gets interesting, so HRES's low is followed on as the surface-pressure minimum (tclib.follow_low, 12-hourly,
from HRES's own msl) and those points are drawn dashed. tcphase.json records a status: "ok", "none" (no storm on the
board, or none HRES tracks long enough for a path) or "missing" (the board is not this cycle's, or HRES's fields could
not be had).
    python src/tc_phase.py --date 20261002 --time 00 --board ../../assets/sst/data/tc_board.json \
        --out-dir ../../assets/sst --json ../../assets/sst/data/tcphase.json
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tclib as TC

LEVS = (925, 850, 700, 600, 500, 400, 300)  # loaded from the open data
P_LOW = 900  # Hart's lower bound; interpolated (z_at)
LOW, UPP = (P_LOW, 850, 700, 600), (600, 500, 400, 300)
STEPS = tuple(range(0, TC.FOLLOW_H_MAX + 1, 12))
STEP_H = 12  # spacing of the phase-space points (h)
BLOCK_DEG = 1.0  # geopotential height block-averaged to about this grid before the parameters are computed
R_KM = 500.0
MAX_STORMS = 4
ET_B = 10.0  # m: Evans & Hart (2003) onset threshold on B
ET_HOLD_H = 24  # h: onset and completion need their condition at consecutive points covering this long
RUN = TC.DET["ifs"]  # the board's key for the run, and the figures' label


def load_fields(date: str, hh: str):
    """HRES geopotential height (m) on LEVS, 12-hourly, -5..80N block-averaged to ~BLOCK_DEG, as (z (step, lev, lat,
    lon), lat, lon), and its mean sea-level pressure (hPa, -5..85N, 0.25 deg) as (msl (step, lat, lon), lat, lon)."""
    zs, ms = [], []
    for h in STEPS:
        f = TC.oper_fields("ifs", date, hh, h, pl=("gh",), levels=LEVS, sfc=("msl",))
        lat, lon = f["lat"], f["lon"]
        jz, jm = (lat >= -5.0) & (lat <= 80.0), (lat >= -5.0) & (lat <= 85.0)
        n = max(1, round(BLOCK_DEG / abs(float(lat[1] - lat[0]))))
        z = np.stack([f[("gh", L)][jz] for L in LEVS])
        ny, nx = (z.shape[1] // n) * n, (z.shape[2] // n) * n
        zs.append(z[:, :ny, :nx].reshape(len(LEVS), ny // n, n, nx // n, n).mean(axis=(2, 4)))
        ms.append(f["msl"][jm] / 100.0)
    zlat = lat[jz][:ny].reshape(-1, n).mean(1)
    zlon = lon[:nx].reshape(-1, n).mean(1)
    return (np.array(zs, np.float32), zlat, zlon), (np.array(ms, np.float32), lat[jm], lon)


def z_at(Z, lev, p):
    """Height on pressure level p from Z (lev, ...) on the levels `lev`: the level itself, or linear in ln p between the
    two levels that bracket it."""
    if p in lev:
        return Z[lev.index(p)]
    lo = max(q for q in lev if q < p)
    hi = min(q for q in lev if q > p)
    w = np.log(hi / p) / np.log(hi / lo)
    return Z[lev.index(hi)] + w * (Z[lev.index(lo)] - Z[lev.index(hi)])


def _disc(lat, lon, clat, clon):
    """Indices (rows, cols window) and distances / bearings of grid points within R_KM of the centre."""
    dl = R_KM / 111.0 + 0.5
    j = np.where(np.abs(lat - clat) <= dl)[0]
    dlon = ((lon - clon + 180) % 360) - 180
    i = np.where(np.abs(dlon) <= dl / max(np.cos(np.deg2rad(clat + dl)), 0.2))[0]
    la = np.deg2rad(lat[j])[:, None]
    lo = np.deg2rad(lon[i])[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    d = TC.A_EARTH / 1000 * np.arccos(np.clip(cosd, -1, 1))
    brg = np.arctan2(np.sin(lo - l0) * np.cos(la), np.cos(c0) * np.sin(la) - np.sin(c0) * np.cos(la) * np.cos(lo - l0))
    return j, i, d <= R_KM, brg


def _slope(dz, levs):
    x = np.log(np.asarray(levs, float))
    return float(np.polyfit(x, np.asarray(dz, float), 1)[0])


def phase_series(z, lat, lon, steps, s):
    """B, -VT_L, -VT_U at each 12-h step where the (extended) track has a position (z (step, lev, lat, lon) on LEVS from
    load_fields; Z900 by z_at), smoothed with a 24-h running mean."""
    lev = list(LEVS)
    fol = s.get("followed", np.zeros(len(s["steps"]), bool))
    mrg = s.get("merged", np.zeros(len(s["steps"]), bool))
    out = {
        "h": [],
        "lat": [],
        "lon": [],
        "B": [],
        "VTL": [],
        "VTU": [],
        "followed": [],
        "merged": [],
        "pmsl": [],
        "wind": [],
    }
    pm = np.asarray(s.get("pmsl", np.full(len(s["steps"]), np.nan)), float)
    wd = np.asarray(s.get("wind", np.full(len(s["steps"]), np.nan)), float)
    for k, h in enumerate(steps):
        pos = TC.at(s, int(h))
        if pos is None or pos[0] <= 0:
            continue
        # storm motion from the track 6 h either side (or the nearest available)
        kk = np.where(s["steps"] == h)[0][0]
        a, b = max(kk - 1, 0), min(kk + 1, len(s["steps"]) - 1)
        if a == b:
            continue
        dlon = ((s["lon"][b] - s["lon"][a] + 180) % 360) - 180
        mot = np.arctan2(np.deg2rad(dlon) * np.cos(np.deg2rad(pos[0])), np.deg2rad(s["lat"][b] - s["lat"][a]))
        j, i, m, brg = _disc(lat, lon, *pos)
        Z = z[k][:, j][:, :, i]  # (lev, y, x)
        rel = ((brg - mot + np.pi) % (2 * np.pi)) - np.pi  # bearing relative to motion
        right, left = m & (rel > 0) & (rel < np.pi), m & (rel < 0) & (rel > -np.pi)
        if not right.any() or not left.any():
            continue
        zl = {p: z_at(Z, lev, p) for p in sorted(set(LOW) | set(UPP))}
        thick = zl[600] - zl[P_LOW]
        B = float(thick[right].mean() - thick[left].mean())
        dzd = {p: float(f[m].max() - f[m].min()) for p, f in zl.items()}
        out["followed"].append(bool(fol[kk]))
        out["merged"].append(bool(mrg[kk]))
        out["pmsl"].append(round(float(pm[kk]), 1) if kk < len(pm) and np.isfinite(pm[kk]) else None)
        out["wind"].append(round(float(wd[kk]), 1) if kk < len(wd) and np.isfinite(wd[kk]) else None)
        out["h"].append(int(h))
        out["lat"].append(pos[0])
        out["lon"].append(pos[1])
        out["B"].append(B)
        out["VTL"].append(_slope([dzd[p] for p in LOW], LOW))
        out["VTU"].append(_slope([dzd[p] for p in UPP], UPP))
    for key in ("B", "VTL", "VTU"):  # Hart: 24-h running mean
        v = np.array(out[key], float)
        h = np.array(out["h"])
        sm = [np.nanmean(v[np.abs(h - hh) <= 12]) for hh in h]
        out[key] = [round(float(x), 1) for x in sm]
    return out


def held_from(h, ok, hold=ET_HOLD_H, gap=STEP_H):
    """The first lead h[i] that starts a run of points where `ok` is true, each within `gap` hours of the one before (a
    missing point breaks the run), spanning at least `hold` hours; None if there is none."""
    k0 = None
    for k in range(len(h)):
        if not ok[k]:
            k0 = None
            continue
        if k0 is None or h[k] - h[k - 1] > gap:
            k0 = k
        if h[k] - h[k0] >= hold:
            return int(h[k0])
    return None


def et_times(ser):
    """Extratropical transition after Evans & Hart (2003): onset where B exceeds ET_B, completion where -VT_L turns
    negative at or after onset (Hart 2003's cold-core lower troposphere). Each condition must hold at consecutive points
    covering ET_HOLD_H (held_from), so a single point over the threshold, such as a lopsided analysis at t=0, does not
    start a transition. Both times are the first point of the run. No onset, no completion (a developing disturbance
    with -VT_L slightly negative at t=0 and B never above ET_B is not "ET complete" at +0 h)."""
    h = list(ser["h"])
    onset = held_from(h, [b > ET_B for b in ser["B"]])
    if onset is None:
        return None, None
    k0 = h.index(onset)
    return onset, held_from(h[k0:], [v < 0 for v in ser["VTL"][k0:]])


def place_labels(fig, ax, labels, caps):
    """Put each label at the first spot around its point that stays inside the panel and clear of the corner captions
    and the labels already placed (fixed offsets run off the frame and into each other)."""
    ren = fig.canvas.get_renderer()
    frame = ax.get_window_extent(ren)
    taken = [c.get_window_extent(ren).expanded(1.02, 1.1) for c in caps]
    # a looping track needs more than eight spots, so: three rings plus straight left/right/up/down. A DATED label (the
    # every-second-day marks) that finds no clear spot is left out rather than drawn over another; the dots keep their
    # date through the colour bar. start / end / tracker's last fix / merges are always drawn.
    spots = [
        (12, 12, "left", "bottom"),
        (-12, 12, "right", "bottom"),
        (12, -12, "left", "top"),
        (-12, -12, "right", "top"),
        (26, 28, "left", "bottom"),
        (-26, 28, "right", "bottom"),
        (26, -28, "left", "top"),
        (-26, -28, "right", "top"),
        (16, 0, "left", "center"),
        (-16, 0, "right", "center"),
        (0, 22, "center", "bottom"),
        (0, -22, "center", "top"),
        (44, 46, "left", "bottom"),
        (-44, 46, "right", "bottom"),
        (44, -46, "left", "top"),
        (-44, -46, "right", "top"),
        (40, 0, "left", "center"),
        (-40, 0, "right", "center"),
        (0, 50, "center", "bottom"),
        (0, -50, "center", "top"),
    ]
    dated = __import__("re").compile(r"^\d{1,2} [A-Z][a-z]{2}")
    box = dict(boxstyle="round,pad=0.15", fc="#fff", ec="none", alpha=0.85)
    for text, xy, leader in labels:
        kw = dict(
            textcoords="offset points",
            fontsize=8.3 if leader else 8.5,
            color="#444" if leader else "#000",
            bbox=box,
            zorder=6,
        )
        if leader:
            kw["arrowprops"] = dict(arrowstyle="-", color="#888", lw=0.8)
        chosen, fallback = None, None
        for spot in spots:
            dx, dy, ha, va = spot
            a = ax.annotate(text, xy, xytext=(dx, dy), ha=ha, va=va, **kw)
            bb = a.get_window_extent(ren)
            inside = bb.x0 >= frame.x0 + 2 and bb.x1 <= frame.x1 - 2 and bb.y0 >= frame.y0 + 2 and bb.y1 <= frame.y1 - 2
            if inside and not any(bb.overlaps(o) for o in taken):
                chosen = bb
                break
            if inside and fallback is None:
                fallback = spot  # inside the frame, though touching something
            a.remove()
        if chosen is None:
            if dated.match(text):
                continue  # crowded: skip this date rather than overlap
            dx, dy, ha, va = fallback or spots[0]
            chosen = ax.annotate(text, xy, xytext=(dx, dy), ha=ha, va=va, **kw).get_window_extent(ren)
        taken.append(chosen)


def render(rows, init, out_png):
    """The phase-space diagrams (and intensity) of one or more storms, one row each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(rows)
    # each storm's axes are fitted to its own track (always keeping the zero lines and the B = 10 m threshold): Hart's
    # fixed -600..300 frame squeezes every track into a corner
    H = 4.25 * n + 1.25
    # intensity: point size = central pressure (bigger = deeper), pressure in the date labels, and a third panel with the
    # central pressure against time, ET onset / completion marked. The tracker's 10-m wind is not shown: a global model's
    # 10-m wind is far too weak for its central pressure (AIFS-ENS: 947 hPa with 59 kt), so it would misstate the storm's
    # category
    fig, axs = plt.subplots(n, 3, figsize=(16.4, H), squeeze=False, gridspec_kw={"width_ratios": [1, 1, 0.66]})
    fig.subplots_adjust(left=0.06, right=0.905, top=1 - 0.95 / H, bottom=1.05 / H, hspace=0.36, wspace=0.25)
    fig.suptitle(
        f"Cyclone phase space (Hart 2003) — {RUN}, init {init:%d %b %Y %HZ}",
        fontsize=14,
        fontweight="bold",
        x=0.075,
        ha="left",
        y=1 - 0.22 / H,
    )

    def span(v, need, minspan, pad=0.12):
        lo, hi = min(np.nanmin(v), need[0]), max(np.nanmax(v), need[1])
        if hi - lo < minspan:
            m = (hi + lo) / 2
            lo, hi = m - minspan / 2, m + minspan / 2
        w = hi - lo
        return lo - pad * w, hi + pad * w

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, STEPS[-1] / 24)

    def psize(p):  # marker area from central pressure: 1010 hPa -> 10, 950 -> ~150
        p = np.array([np.nan if v is None else v for v in p], float)
        return np.where(np.isfinite(p), np.clip(10 + 2.35 * (1010 - p), 10, 190), 18)

    for r, (s, ser, onset, comp) in enumerate(rows):
        d = np.array(ser["h"]) / 24.0
        rB = span(np.array(ser["B"]), (-5, 15), 30)
        rL = span(np.array(ser["VTL"]), (-40, 40), 160)
        rU = span(np.array(ser["VTU"]), (-40, 40), 160)
        for c, (xk, yk, xl, yl, xr, yr) in enumerate(
            (
                (
                    "B",
                    "VTL",
                    "B: thermal asymmetry, 900–600 hPa thickness right minus left of motion (m)",
                    "−VT_L: 900–600 hPa thermal wind",
                    rB,
                    rL,
                ),
                ("VTL", "VTU", "−VT_L: 900–600 hPa thermal wind", "−VT_U: 600–300 hPa thermal wind", rL, rU),
            )
        ):
            ax = axs[r][c]
            caps, labels = [], []
            x, y = np.array(ser[xk]), np.array(ser[yk])
            if c == 0:
                ax.axvline(ET_B, color="#888", lw=1, ls="--")
                ax.axhline(0, color="#888", lw=1, ls="--")
                for tx, ty, t in (
                    (0.02, 0.97, "symmetric warm core\n(tropical)"),
                    (0.98, 0.97, "asymmetric warm core"),
                    (0.02, 0.03, "symmetric cold core"),
                    (0.98, 0.03, "asymmetric cold core\n(extratropical)"),
                ):
                    caps.append(
                        ax.text(
                            tx,
                            ty,
                            t,
                            transform=ax.transAxes,
                            fontsize=8.5,
                            color="#6f6b64",
                            ha="left" if tx < 0.5 else "right",
                            va="top" if ty > 0.5 else "bottom",
                            zorder=1,
                            bbox=dict(boxstyle="round,pad=0.1", fc="#fff", ec="none", alpha=0.7),
                        )
                    )
            else:
                ax.axvline(0, color="#888", lw=1, ls="--")
                ax.axhline(0, color="#888", lw=1, ls="--")
                for tx, ty, t in (
                    (0.98, 0.97, "deep warm core\n(tropical)"),
                    (0.02, 0.03, "deep cold core\n(extratropical)"),
                    (0.98, 0.03, "shallow warm core\n(hybrid / ET)"),
                    (0.02, 0.97, "warm aloft, cold below"),
                ):
                    caps.append(
                        ax.text(
                            tx,
                            ty,
                            t,
                            transform=ax.transAxes,
                            fontsize=8.5,
                            color="#6f6b64",
                            ha="left" if tx < 0.5 else "right",
                            va="top" if ty > 0.5 else "bottom",
                            zorder=1,
                            bbox=dict(boxstyle="round,pad=0.1", fc="#fff", ec="none", alpha=0.7),
                        )
                    )
            pts = np.c_[x, y].reshape(-1, 1, 2)
            fol = np.array(ser.get("followed", [False] * len(x)), bool)
            if len(x) > 1:
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                for mask, ls in ((~fol[1:], "solid"), (fol[1:], (0, (2.5, 1.6)))):  # tracker solid, followed dashed
                    if mask.any():
                        lc = LineCollection(segs[mask], cmap=cmap, norm=norm, lw=3, linestyles=ls)
                        lc.set_array(d[:-1][mask])
                        ax.add_collection(lc)
            sz = psize(ser.get("pmsl", [None] * len(x)))
            ax.scatter(
                x[~fol],
                y[~fol],
                c=d[~fol],
                cmap=cmap,
                norm=norm,
                s=sz[~fol],
                zorder=3,
                edgecolors="#fff",
                linewidths=0.5,
            )
            if fol.any():
                ax.scatter(
                    x[fol],
                    y[fol],
                    s=sz[fol],
                    zorder=3,
                    facecolors="#fff",
                    edgecolors=[cmap(norm(v)) for v in d[fol]],
                    linewidths=1.4,
                )
                j = np.where(fol)[0][0] - 1
                if j >= 0:
                    labels.append(("tracker's last fix", (x[j], y[j]), True))
                mg = np.array(ser.get("merged", [False] * len(x)), bool)
                if mg.any():  # absorbed by a low that already existed (tclib.follow_low)
                    jm = int(np.where(mg)[0][0])
                    labels.append(("merges into another low", (x[jm], y[jm]), True))
            ax.scatter(x[:1], y[:1], marker="o", s=110, facecolors="none", edgecolors="#000", lw=1.6, zorder=4)
            ax.scatter(x[-1:], y[-1:], marker="X", s=90, color="#000", zorder=4)
            labels[:0] = [
                (f"start, {TC.date_lab(init, ser['h'][0] / 24, 'short')}", (x[0], y[0]), False),
                (f"end, {TC.date_lab(init, ser['h'][-1] / 24, 'short')}", (x[-1], y[-1]), False),
            ]
            hh = np.asarray(ser["h"])  # a date at every second day's point (+ its pressure)
            pms = ser.get("pmsl", [None] * len(x))
            for kd in range(2, int(hh[-1] // 24), 2):
                jj = np.where(hh == 24 * kd)[0]
                if len(jj):
                    pv = pms[jj[0]]
                    labels.append(
                        (
                            TC.date_lab(init, kd, "short") + (f" · {pv:.0f} hPa" if pv is not None else ""),
                            (x[jj[0]], y[jj[0]]),
                            True,
                        )
                    )
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            place_labels(fig, ax, labels, caps)
            ax.set_xlabel(xl, fontsize=9.5)
            ax.set_ylabel(yl, fontsize=9.5)
            ax.grid(alpha=0.3)
        ax = axs[r][2]  # intensity against time
        pv = np.array([np.nan if v is None else v for v in ser.get("pmsl", [])], float)
        if len(pv) and np.isfinite(pv).any():
            for mask, ls in ((~fol, "-"), (fol, (0, (2.5, 1.6)))):
                if mask.any():
                    m2 = mask.copy()
                    idx = np.where(mask)[0]
                    if ls != "-" and idx[0] > 0:
                        m2[idx[0] - 1] = True  # join the dashed part to the last tracker fix
                    ax.plot(d[m2], pv[m2], color="#1b3a5c", lw=2.0, ls=ls, zorder=2)
            ax.scatter(
                d, pv, c=d, cmap=cmap, norm=norm, s=psize(ser["pmsl"]), zorder=3, edgecolors="#fff", linewidths=0.5
            )
            ax.invert_yaxis()
            ax.set_ylabel("central pressure (hPa)", fontsize=9.5, color="#1b3a5c")
            jmin = int(np.nanargmin(pv))
            ax.annotate(
                f"deepest {pv[jmin]:.0f} hPa\n{TC.date_lab(init, d[jmin], 'short')}",
                (d[jmin], pv[jmin]),
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                fontsize=8.6,
                fontweight="bold",
                color="#1b3a5c",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#fff", ec="none", alpha=0.85),
            )
        marks = (
            ((onset, "ET onset and complete"),)
            if comp is not None and comp == onset
            else ((onset, "ET onset"), (comp, "ET complete"))
        )  # one line when both fall on one point
        for hv, lab_ in marks:
            if hv is not None:
                ax.axvline(hv / 24, color="#6f6b64", lw=1, ls="--")
                ax.text(
                    hv / 24,
                    0.02,
                    lab_,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    color="#6f6b64",
                    rotation=90,
                    va="bottom",
                    ha="center",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#fff", ec="none", alpha=0.85),
                )
        ax.set_xlabel("forecast day", fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8.5)
        ax.set_title("Intensity", loc="left", fontsize=10.5, fontweight="bold")
        et = []
        if onset is not None:
            et.append(f"ET onset {TC.date_lab(init, onset / 24, 'short')} (B > {ET_B:.0f} m for {ET_HOLD_H} h)")
        if comp is not None:
            et.append(f"ET complete {TC.date_lab(init, comp / 24, 'short')} (−VT_L < 0 for {ET_HOLD_H} h)")
        axs[r][0].set_title(
            f"{s['label']}: " + ("; ".join(et) if et else f"no extratropical transition in {RUN}'s track"),
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
    cax = fig.add_axes([0.952, 0.25, 0.010, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("valid date (UTC) and forecast day")
    cb.set_ticks(range(0, STEPS[-1] // 24 + 1, 2))
    cb.set_ticklabels(
        [TC.date_lab(init, d, "one").replace(" (", "\n(") for d in range(0, STEPS[-1] // 24 + 1, 2)], fontsize=7.6
    )
    fig.text(
        0.075,
        0.004,
        "Axes fitted to each storm's track. Thermal wind = slope of the height perturbation (max − min within 500 km) against ln p; "
        "positive = warm core; 24-h running mean. Point size = central pressure (bigger = deeper).\n"
        f"Geopotential height block-averaged to {BLOCK_DEG:g}°; levels 900/850/700/600 and 600/500/400/300 hPa (900 hPa interpolated in ln p "
        f"between 925 and 850). ET onset / complete: the condition holds for {ET_HOLD_H} h (Evans & Hart 2003).\n"
        "Solid: ECMWF's tracker. Dashed, open dots: the low followed on as the surface-pressure minimum after the tracker's last fix,\n"
        "or the low it merges into (the tracker usually stops at extratropical transition), for up to 72 h and to day 10; a low of "
        "1000 hPa or more only over the sea north of 30°N.",
        fontsize=8,
        color="#6f6b64",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=125,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
        pil_kwargs={"quality": 85, "method": 6},
    )
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--board", default="assets/sst/data/tc_board.json")
    ap.add_argument("--out-dir", default="assets/sst", help="per-storm figures go to <out-dir>/tc/phase_<storm>.webp")
    ap.add_argument("--json", default="assets/sst/data/tcphase.json")
    a = ap.parse_args()
    t0 = time.time()
    hh = f"{int(a.time):02d}"
    init = pd.Timestamp(f"{a.date}T{hh}:00")
    board = json.loads(Path(a.board).read_text()) if Path(a.board).exists() else {}
    fresh = board.get("init") == f"{init:%Y-%m-%dT%H:%MZ}" and board.get("status") != "missing"
    # the board's storms (support-filtered there), named ones first, with HRES's track of each
    storms = []
    for s in board.get("storms", []) if fresh else []:
        d = s.get("deterministic", {}).get(RUN)
        if d and len(d["h"]) >= 2:
            storms.append(
                dict(
                    id=s["slug"],
                    name=s["name"],
                    label=s["name"] if s["named"] else f"Possible new storm {s['id']}",  # a row title that fits
                    named=s["named"],
                    steps=np.array(d["h"], int),
                    lat=np.array(d["lat"], float),
                    lon=np.array(d["lon"], float),
                    pmsl=np.array([np.nan if v is None else v for v in d["pmsl"]], float),
                )
            )
    storms.sort(key=lambda s: not s["named"])
    rows, z = [], None
    if storms:
        try:
            (z, lat, lon), (msl, mlat, mlon) = load_fields(a.date, hh)
            print(f"  {RUN} gh and msl loaded in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001  this cycle's (empty) JSON and figure are still written
            print(f"  {RUN} fields unavailable ({e!r}); no phase-space diagrams this cycle", flush=True)
    if z is not None:
        land = TC.load_land(a.date, hh)  # the follower's land rule (heat / lee lows)
        for s in storms:
            if len(rows) >= MAX_STORMS:
                break
            # day 10 at most, the same follower as the board's members and the maps
            cont = TC.follow_low(msl, mlat, mlon, np.array(STEPS), s, h_max=TC.FOLLOW_H_MAX, merge=True, land=land)
            if len(cont["steps"]):
                mg = (
                    f", merged into another low at +{int(cont['steps'][cont['merged']][0])} h"
                    if cont["merged"].any()
                    else ""
                )
                print(
                    f"  {s['name']}: followed past the tracker's last fix (+{int(s['steps'][-1])} h) to "
                    f"+{int(cont['steps'][-1])} h, deepest {np.nanmin(cont['pmsl']):.0f} hPa{mg}",
                    flush=True,
                )
            st = dict(s, **TC.extended(s, cont))
            ser = phase_series(z, lat, lon, STEPS, st)
            if len(ser["h"]) >= 2:
                on, co = et_times(ser)
                rows.append((st, ser, on, co))
                print(
                    f"  {s['name']}: start B {ser['B'][0]:+.0f} m, -VTL {ser['VTL'][0]:+.0f}, -VTU {ser['VTU'][0]:+.0f}; "
                    f"end +{ser['h'][-1]} h B {ser['B'][-1]:+.0f}, -VTL {ser['VTL'][-1]:+.0f}, -VTU {ser['VTU'][-1]:+.0f}; "
                    f"ET onset {on}, complete {co}",
                    flush=True,
                )
    if not fresh:
        status, note = "missing", f"Phase space unavailable this cycle (no storm board for {init:%d %b %HZ})"
    elif storms and z is None:
        status, note = "missing", f"Phase space unavailable this cycle ({RUN}'s fields could not be loaded)"
    elif rows:
        status, note = "ok", None
    elif storms:
        status, note = "none", f"No storm on the board is tracked long enough in {RUN} for a phase-space path"
    else:
        status, note = "none", f"No storm on the board has a track in {RUN} this cycle"
    if status == "missing":
        TC.report(f"phase space {a.date} {hh}Z: {note[0].lower() + note[1:]}", "error")
    # one diagram per storm for the board; stale storms' files removed first
    tcd = Path(a.out_dir) / "tc"
    tcd.mkdir(parents=True, exist_ok=True)
    for f in tcd.glob("phase_*.webp"):
        f.unlink()
    files = {}
    for row in rows:
        try:
            render([row], init, tcd / f"phase_{row[0]['id']}.webp")
            files[row[0]["id"]] = f"tc/phase_{row[0]['id']}.webp"
        except Exception as e:  # noqa: BLE001  one storm's figure must not stop the others
            print(f"  per-storm phase {row[0]['id']} failed ({e!r})", flush=True)
    js = {
        "init": f"{init:%Y-%m-%dT%H:%MZ}",
        "status": status,
        "note": note,
        "run": RUN,
        "radius_km": R_KM,
        "grid_deg": BLOCK_DEG,
        "levels": {"B": [P_LOW, 600], "VTL": list(LOW), "VTU": list(UPP), "interpolated": [P_LOW]},
        "et_rule": {"onset": f"B > {ET_B:g} m", "complete": "-VT_L < 0 at or after onset", "hold_h": ET_HOLD_H},
        "storms": [
            {
                "slug": s["id"],
                "name": s["name"],
                "et_onset_h": on,
                "et_complete_h": co,
                "series": ser,
                "file": files.get(s["id"]),
            }
            for s, ser, on, co in rows
        ],
    }
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(js))
    print(
        f"wrote {len(files)} figures and {a.json} ({status}, {len(rows)} storms) in {time.time() - t0:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
