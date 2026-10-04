#!/usr/bin/env python3
"""Cyclone phase space (Hart 2003, MWR 131, 585-616) for live Northern-Hemisphere tropical cyclones, AIFS-ENS control.

Along the control's own tracker position (tc_jet.decode, the run the dynamic-tropopause charts draw), every 12 h to day 15:
  B      thermal asymmetry = mean (Z600 - Z900) in the semicircle RIGHT of the storm motion minus the LEFT one, within
         500 km (m; NH sign). ~0 = thermally symmetric (tropical); > 10 m = frontal (Evans & Hart 2003: ET onset).
  -VT_L  lower-troposphere thermal wind = slope of dZ = Zmax - Zmin (within 500 km) against ln p, 900-600 hPa.
  -VT_U  the same, 600-300 hPa.   > 0 = warm core, < 0 = cold core.  ET complete when -VT_L turns negative.
Approximation: AIFS-ENS open data has z on 1000/925/850/700/600/500/400/300 hPa only, so the lower layer is 925-600 hPa
(925, 850, 700, 600) instead of Hart's 900-600 every 25 hPa, B uses Z600 - Z925, and the upper layer is 600/500/400/300.
Parameters are smoothed with a 24-h running mean (Hart). Control member only: one deterministic path through phase space.
Since 2026-10-03 the path continues past the tracker's last fix: ECMWF's tracker stops at (or near) extratropical transition,
exactly when the phase diagram gets interesting, so the control's low is followed on as the surface-pressure minimum
(tc_jet.follow_low, 12-hourly, from the control's own msl) and those points are drawn dashed.
    python src/tc_phase.py --date 20261002 --time 00 --out ../../assets/sst/tcphase.webp --json ../../assets/sst/data/tcphase.json
"""
from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray load eckit's own libproj (exit abort 'double free', see CLAUDE.md pacjet)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                                   # noqa: E402
import tc_jet as TC                                                     # noqa: E402

LEVS = (925, 850, 700, 600, 500, 400, 300)
LOW, UPP = (925, 850, 700, 600), (600, 500, 400, 300)
STEPS = tuple(range(0, 361, 12))
R_KM = 500.0
G0 = 9.80665
MAX_STORMS = 4


def load_z(cyc):
    import xarray as xr
    p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "z", "pl", LEVS, STEPS))
    da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["z"]
    if "number" in da.dims:
        da = da.squeeze("number", drop=True)
    da = da.sortby("latitude").sel(latitude=slice(-5.0, 80.0))
    return (da / G0).transpose("step", "isobaricInhPa", "latitude", "longitude").astype("float32"), p


def _disc(lat, lon, clat, clon):
    """Indices (rows, cols window) and distances / bearings of grid points within R_KM of the centre."""
    dl = R_KM / 111.0 + 0.5
    j = np.where(np.abs(lat - clat) <= dl)[0]
    dlon = ((lon - clon + 180) % 360) - 180
    i = np.where(np.abs(dlon) <= dl / max(np.cos(np.deg2rad(clat + dl)), 0.2))[0]
    la = np.deg2rad(lat[j])[:, None]; lo = np.deg2rad(lon[i])[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    d = TC.A_EARTH / 1000 * np.arccos(np.clip(cosd, -1, 1))
    brg = np.arctan2(np.sin(lo - l0) * np.cos(la), np.cos(c0) * np.sin(la) - np.sin(c0) * np.cos(la) * np.cos(lo - l0))
    return j, i, d <= R_KM, brg


def _slope(dz, levs):
    x = np.log(np.asarray(levs, float))
    return float(np.polyfit(x, np.asarray(dz, float), 1)[0])


def load_msl(cyc):
    import xarray as xr
    p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "msl", "sfc", (), STEPS))
    da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["msl"]
    if "number" in da.dims:
        da = da.squeeze("number", drop=True)
    da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude").sortby("latitude").sel(latitude=slice(-5.0, 85.0))
    return (da / 100.0).values.astype("float32"), da.latitude.values, da.longitude.values, (da.step / np.timedelta64(1, "h")).values.astype(int)


def phase_series(z, lat, lon, steps, s):
    """B, -VT_L, -VT_U at each 12-h step where the (extended) control track has a position."""
    lev = list(z.isobaricInhPa.values.astype(int))
    fol = s.get("followed", np.zeros(len(s["steps"]), bool))
    mrg = s.get("merged", np.zeros(len(s["steps"]), bool))
    out = {"h": [], "lat": [], "lon": [], "B": [], "VTL": [], "VTU": [], "followed": [], "merged": []}
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
        Z = z.isel(step=k).values[:, j][:, :, i]                          # (lev, y, x)
        rel = ((brg - mot + np.pi) % (2 * np.pi)) - np.pi                 # bearing relative to motion
        right, left = m & (rel > 0) & (rel < np.pi), m & (rel < 0) & (rel > -np.pi)
        thick = Z[lev.index(600)] - Z[lev.index(925)]
        if not right.any() or not left.any():
            continue
        B = float(thick[right].mean() - thick[left].mean())
        dz = [float(Z[lev.index(p)][m].max() - Z[lev.index(p)][m].min()) for p in LEVS]
        dzd = dict(zip(LEVS, dz))
        out["followed"].append(bool(fol[kk])); out["merged"].append(bool(mrg[kk]))
        out["h"].append(int(h)); out["lat"].append(pos[0]); out["lon"].append(pos[1]); out["B"].append(B)
        out["VTL"].append(_slope([dzd[p] for p in LOW], LOW)); out["VTU"].append(_slope([dzd[p] for p in UPP], UPP))
    for key in ("B", "VTL", "VTU"):                                       # Hart: 24-h running mean
        v = np.array(out[key], float); h = np.array(out["h"])
        sm = [np.nanmean(v[np.abs(h - hh) <= 12]) for hh in h]
        out[key] = [round(float(x), 1) for x in sm]
    return out


def et_times(ser):
    """Evans & Hart (2003): onset = first B > 10 m; completion = first -VT_L < 0 at or after onset."""
    onset = next((h for h, b in zip(ser["h"], ser["B"]) if b > 10), None)
    comp = next((h for h, v in zip(ser["h"], ser["VTL"]) if v < 0 and (onset is None or h >= onset)), None)
    return onset, comp


def place_labels(fig, ax, labels, caps):
    """Put each label at the first of eight spots around its point that stays inside the panel and clear of the corner captions
    and the labels already placed (2026-10-03: fixed offsets ran off the frame and into each other)."""
    ren = fig.canvas.get_renderer(); frame = ax.get_window_extent(ren)
    taken = [c.get_window_extent(ren).expanded(1.02, 1.1) for c in caps]
    spots = [(12, 12, "left", "bottom"), (-12, 12, "right", "bottom"), (12, -12, "left", "top"), (-12, -12, "right", "top"),
             (26, 28, "left", "bottom"), (-26, 28, "right", "bottom"), (26, -28, "left", "top"), (-26, -28, "right", "top")]
    box = dict(boxstyle="round,pad=0.15", fc="#fff", ec="none", alpha=0.85)
    for text, xy, leader in labels:
        kw = dict(textcoords="offset points", fontsize=8.3 if leader else 8.5, color="#444" if leader else "#000", bbox=box, zorder=6)
        if leader: kw["arrowprops"] = dict(arrowstyle="-", color="#888", lw=0.8)
        chosen, fallback = None, None
        for spot in spots:
            dx, dy, ha, va = spot
            a = ax.annotate(text, xy, xytext=(dx, dy), ha=ha, va=va, **kw); bb = a.get_window_extent(ren)
            inside = bb.x0 >= frame.x0 + 2 and bb.x1 <= frame.x1 - 2 and bb.y0 >= frame.y0 + 2 and bb.y1 <= frame.y1 - 2
            if inside and not any(bb.overlaps(o) for o in taken): chosen = bb; break
            if inside and fallback is None: fallback = spot          # inside the frame, though touching something
            a.remove()
        if chosen is None:
            dx, dy, ha, va = fallback or spots[0]
            chosen = ax.annotate(text, xy, xytext=(dx, dy), ha=ha, va=va, **kw).get_window_extent(ren)
        taken.append(chosen)


def render(rows, init, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    n = max(1, len(rows))
    # 2026-10-03 (user: "way too small with way too much whitespace"): each storm's axes are fitted to its own track (always
    # keeping the zero lines and the B = 10 m threshold), the rows sit closer and the figure is drawn at a higher resolution;
    # Hart's fixed -600..300 frame had squeezed every track into a corner.
    H = 4.25 * n + 1.25
    fig, axs = plt.subplots(n, 2, figsize=(12.6, H), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.915, top=1 - 0.95 / H, bottom=1.05 / H, hspace=0.36, wspace=0.22)
    fig.suptitle(f"Cyclone phase space (Hart 2003) — AIFS-ENS control, init {init:%d %b %Y %HZ}", fontsize=14,
                 fontweight="bold", x=0.075, ha="left", y=1 - 0.22 / H)

    def span(v, need, minspan, pad=0.12):
        lo, hi = min(np.nanmin(v), need[0]), max(np.nanmax(v), need[1])
        if hi - lo < minspan: m = (hi + lo) / 2; lo, hi = m - minspan / 2, m + minspan / 2
        w = hi - lo; return lo - pad * w, hi + pad * w
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(0, 15)
    if not rows:
        for ax in axs[0]:
            ax.axis("off")
        axs[0][0].text(0.5, 0.5, "No Northern-Hemisphere tropical storm in the control run", transform=fig.transFigure,
                       ha="center", fontsize=13, color="#555")
    for r, (s, ser, onset, comp) in enumerate(rows):
        d = np.array(ser["h"]) / 24.0
        rB = span(np.array(ser["B"]), (-5, 15), 30); rL = span(np.array(ser["VTL"]), (-40, 40), 160)
        rU = span(np.array(ser["VTU"]), (-40, 40), 160)
        for c, (xk, yk, xl, yl, xr, yr) in enumerate((
                ("B", "VTL", "B: thermal asymmetry, 925–600 hPa thickness right minus left of motion (m)",
                 "−VT_L: 925–600 hPa thermal wind", rB, rL),
                ("VTL", "VTU", "−VT_L: 925–600 hPa thermal wind", "−VT_U: 600–300 hPa thermal wind", rL, rU))):
            ax = axs[r][c]; caps, labels = [], []
            x, y = np.array(ser[xk]), np.array(ser[yk])
            if c == 0:
                ax.axvline(10, color="#888", lw=1, ls="--"); ax.axhline(0, color="#888", lw=1, ls="--")
                for tx, ty, t in ((0.02, 0.97, "symmetric warm core\n(tropical)"), (0.98, 0.97, "asymmetric warm core"),
                                  (0.02, 0.03, "symmetric cold core"), (0.98, 0.03, "asymmetric cold core\n(extratropical)")):
                    caps.append(ax.text(tx, ty, t, transform=ax.transAxes, fontsize=8.5, color="#6f6b64", ha="left" if tx < .5 else "right",
                            va="top" if ty > .5 else "bottom", zorder=1, bbox=dict(boxstyle="round,pad=0.1", fc="#fff", ec="none", alpha=0.7)))
            else:
                ax.axvline(0, color="#888", lw=1, ls="--"); ax.axhline(0, color="#888", lw=1, ls="--")
                for tx, ty, t in ((0.98, 0.97, "deep warm core\n(tropical)"), (0.02, 0.03, "deep cold core\n(extratropical)"),
                                  (0.98, 0.03, "shallow warm core\n(hybrid / ET)"), (0.02, 0.97, "warm aloft, cold below")):
                    caps.append(ax.text(tx, ty, t, transform=ax.transAxes, fontsize=8.5, color="#6f6b64", ha="left" if tx < .5 else "right",
                            va="top" if ty > .5 else "bottom", zorder=1, bbox=dict(boxstyle="round,pad=0.1", fc="#fff", ec="none", alpha=0.7)))
            pts = np.c_[x, y].reshape(-1, 1, 2)
            fol = np.array(ser.get("followed", [False] * len(x)), bool)
            if len(x) > 1:
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                for mask, ls in ((~fol[1:], "solid"), (fol[1:], (0, (2.5, 1.6)))):     # tracker solid, followed dashed
                    if mask.any():
                        lc = LineCollection(segs[mask], cmap=cmap, norm=norm, lw=3, linestyles=ls)
                        lc.set_array(d[:-1][mask]); ax.add_collection(lc)
            sc = ax.scatter(x[~fol], y[~fol], c=d[~fol], cmap=cmap, norm=norm, s=22, zorder=3, edgecolors="none")
            if fol.any():
                ax.scatter(x[fol], y[fol], s=26, zorder=3, facecolors="#fff", edgecolors=[cmap(norm(v)) for v in d[fol]], linewidths=1.4)
                j = np.where(fol)[0][0] - 1
                if j >= 0:
                    labels.append(("tracker's last fix", (x[j], y[j]), True))
                mg = np.array(ser.get("merged", [False] * len(x)), bool)
                if mg.any():                                   # absorbed by a low that already existed (tc_jet.follow_low)
                    jm = int(np.where(mg)[0][0])
                    labels.append(("merges into another low", (x[jm], y[jm]), True))
            ax.scatter(x[:1], y[:1], marker="o", s=110, facecolors="none", edgecolors="#000", lw=1.6, zorder=4)
            ax.scatter(x[-1:], y[-1:], marker="X", s=90, color="#000", zorder=4)
            labels[:0] = [(f"start, {TC.date_lab(init, ser['h'][0] / 24, 'short')}", (x[0], y[0]), False),
                          (f"end, {TC.date_lab(init, ser['h'][-1] / 24, 'short')}", (x[-1], y[-1]), False)]
            hh = np.asarray(ser["h"])                          # 2026-10-04: a date at every second day's point
            for kd in range(2, int(hh[-1] // 24), 2):
                jj = np.where(hh == 24 * kd)[0]
                if len(jj):
                    labels.append((TC.date_lab(init, kd, "short"), (x[jj[0]], y[jj[0]]), True))
            ax.set_xlim(*xr); ax.set_ylim(*yr)
            place_labels(fig, ax, labels, caps)
            ax.set_xlabel(xl, fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5); ax.grid(alpha=0.3)
        et = []
        if onset is not None: et.append(f"ET onset +{onset} h (B > 10 m)")
        if comp is not None: et.append(f"ET complete +{comp} h (−VT_L < 0)")
        if onset is not None: et[0] = f"ET onset {TC.date_lab(init, onset / 24, 'short')} (B > 10 m)"
        if comp is not None: et[-1] = f"ET complete {TC.date_lab(init, comp / 24, 'short')} (−VT_L < 0)"
        axs[r][0].set_title(f"{TC.label(s)}: " + ("; ".join(et) if et else "no extratropical transition in the control's track"),
                            loc="left", fontsize=11, fontweight="bold")
    cax = fig.add_axes([0.932, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("valid date (UTC) and forecast day")
    cb.set_ticks(range(0, 16, 2)); cb.set_ticklabels([TC.date_lab(init, d, "one").replace(" (", "\n(") for d in range(0, 16, 2)], fontsize=7.6)
    fig.text(0.075, 0.004, "Axes fitted to each storm's track. Thermal wind = slope of the height perturbation (max − min within 500 km) against ln p; "
             "positive = warm core; 24-h running mean.\nLevels 925/850/700/600 and 600/500/400/300 hPa (open data; Hart used 900–600 every 25 hPa).\n"
             "Solid: ECMWF's tracker. Dashed, open dots: the low followed on as the surface-pressure minimum after the tracker's last fix,\n"
             "or the low it merges into (the tracker usually stops at extratropical transition).", fontsize=8, color="#6f6b64")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125, facecolor="white", pil_kwargs={"quality": 85, "method": 6}); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out", default="assets/sst/tcphase.webp"); ap.add_argument("--json", default="assets/sst/data/tcphase.json")
    a = ap.parse_args()
    t0 = time.time()
    cyc = ecmwf.Cycle(a.date, a.time); init = pd.Timestamp(f"{a.date}T{a.time}:00")
    fp = TC.fetch(a.date, int(a.time), Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc")
    storms = TC.nh_storms(TC.decode(fp)) if fp else []
    # named storms first, then the strongest
    storms = sorted(storms, key=lambda s: (not s["named"], -np.nanmax(np.r_[s["wind"], 0])))[:MAX_STORMS]
    rows = []
    if storms:
        z, zp = load_z(cyc)
        print(f"  z {zp.name}: {zp.stat().st_size / 1e6:.0f} MB, loaded in {time.time() - t0:.0f} s", flush=True)
        lat, lon = z.latitude.values, z.longitude.values
        steps = (z.step / np.timedelta64(1, "h")).values.astype(int)
        try:
            msl, mlat, mlon, msteps = load_msl(cyc)
        except Exception as e:                                           # the extension is a bonus: never lose the chart for it
            print(f"  control msl unavailable ({e}); phase paths end at the tracker's last fix", flush=True); msl = None
        for s in storms:
            if msl is not None:
                cont = TC.follow_low(msl, mlat, mlon, msteps, s, merge=True)
                if len(cont["steps"]):
                    mg = f", merged into another low at +{int(cont['steps'][cont['merged']][0])} h" if cont["merged"].any() else ""
                    print(f"  {TC.label(s)}: followed past the tracker's last fix (+{int(s['steps'][-1])} h) to +{int(cont['steps'][-1])} h, "
                          f"deepest {np.nanmin(cont['pmsl']):.0f} hPa{mg}", flush=True)
                s = dict(s, **TC.extended(s, cont))
            ser = phase_series(z, lat, lon, steps, s)
            if len(ser["h"]) >= 2:
                on, co = et_times(ser)
                rows.append((s, ser, on, co))
                print(f"  {TC.label(s)}: start B {ser['B'][0]:+.0f} m, -VTL {ser['VTL'][0]:+.0f}, -VTU {ser['VTU'][0]:+.0f}; "
                      f"end +{ser['h'][-1]} h B {ser['B'][-1]:+.0f}, -VTL {ser['VTL'][-1]:+.0f}, -VTU {ser['VTU'][-1]:+.0f}; "
                      f"ET onset {on}, complete {co}", flush=True)
    render(rows, init, Path(a.out))
    # one diagram per storm for the page's storm picker (2026-10-03); stale storms' files removed first
    tcd = Path(a.out).parent / "tc"; tcd.mkdir(parents=True, exist_ok=True)
    for f in tcd.glob("phase_*.webp"): f.unlink()
    files = {}
    for row in rows:
        sid = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(row[0]["id"]))
        try:
            render([row], init, tcd / f"phase_{sid}.webp"); files[row[0]["id"]] = f"tc/phase_{sid}.webp"
        except Exception as e:
            print(f"  per-storm phase {row[0]['id']} failed ({e!r})", flush=True)
    js = {"init": f"{init:%Y-%m-%dT%H:%MZ}", "member": "control", "radius_km": R_KM,
          "levels": {"B": [925, 600], "VTL": list(LOW), "VTU": list(UPP)},
          "storms": [{"id": s["id"], "name": TC.label(s), "et_onset_h": on, "et_complete_h": co, "series": ser, "file": files.get(s["id"])}
                     for s, ser, on, co in rows]}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(js))
    print(f"wrote {a.out} and {a.json} ({len(rows)} storms) in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
