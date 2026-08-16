#!/usr/bin/env python3
"""IMERG rainfall maps over Colombia with IDEAM rain gauges overlaid.

The visual ground-truth: satellite estimate as the field, every reporting
IDEAM gauge as a dot on the SAME color scale — where dots vanish into the
background the satellite is right; where they pop, it isn't. Four panels:

  1. yesterday's IMERG daily total + gauges
  2. 7-day accumulation + gauges (stations reporting >=6 of 7 days)
  3. 30-day accumulation + gauges (>=27 of 30 days)
  4. IMERG-at-gauge vs gauge scatter for the 30-day totals, log-log,
     colored by hydro region, annotated with each region's median
     satellite/gauge ratio — the multiplicative bias factors the
     rain->inflow model will use.

XM region polygons (ideam variant) outlined on every map. Bias factors
also written to colombia_hydro/data/gauge_bias.json.

    python scripts/sst/colombia_rain_map.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                      # noqa: E402
from ideam_gauges import fetch_range           # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
OUT_PNG = REPO / "colombia_hydro" / "rain_vs_gauges.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "gauge_bias.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
RCOL = {"ANTIOQUIA": "#1b7837", "CALDAS": "#762a83", "CARIBE": "#2166ac",
        "CENTRO": "#b2182b", "ORIENTE": "#e08214", "VALLE": "#35978f"}
# Colombia crop (deg E 0-360, deg N)
# Crop = bounding box of the region polygons + padding (set in main once
# the geojson is loaded; these are fallbacks)
LON0, LON1, LAT0, LAT1 = 282.0, 287.8, 1.5, 8.7

CMAP = ListedColormap([
    "#f7f7f7", "#c7e9c0", "#74c476", "#238b45", "#41b6c4", "#225ea8",
    "#253494", "#54278f", "#7a0177", "#ae017e"])


def region_paths():
    gj = json.loads(REGIONS_GJ.read_text())
    out = {}
    for ft in gj["features"]:
        name = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [np.array(r[0]) for r in polys]
        out[name] = rings
    return out


def imerg_stack(days: list[datetime]):
    """(nday, lat, lon) cropped daily fields + axes; missing days -> None rows."""
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml] % 360)
    lats = np.sort(IP._LAT[mt])
    li = (lons >= LON0) & (lons <= LON1)
    la = (lats >= LAT0) & (lats <= LAT1)
    fields = []
    for d in days:
        g = IP._load(IP.DAILY_CACHE, f"{d:%Y%m%d}")
        fields.append(None if g is None else g[np.ix_(la, li)])
    return fields, lons[li], lats[la]


CORR_NPZ = HERE / "data" / "imerg_gauge_corr.npz"
CORR_MAX_AGE_DAYS = 7


def build_correction(days_all, fields_all, rdates, lons, lats):
    """Cell-wise gauge-correction field for IMERG: per-station ratio of
    summed gauge vs summed IMERG-at-station over ALL paired archive days
    (>=120 days, >=50 mm gauge total), log-IDW interpolated to the grid
    with a unity-pulling background so the field relaxes to 1.0 away from
    stations. Multiplicative: corrected = IMERG * F. Cached, rebuilt
    weekly as the archive grows."""
    import time as _t
    if CORR_NPZ.exists() and (_t.time() - CORR_NPZ.stat().st_mtime) < 86400 * CORR_MAX_AGE_DAYS:
        z = np.load(CORR_NPZ)
        return z["F"]
    from ideam_gauges import CACHE as GCACHE
    gsum, isum, ndays, meta = {}, {}, {}, {}
    for i, d in enumerate(rdates):
        f = GCACHE / f"{str(d).replace('-', '')}.json"
        if not f.exists() or fields_all[i] is None:
            continue
        day = json.loads(f.read_text())
        for code, g in day.items():
            la, lo = g["la"], g["lo"] % 360
            if not (LON0 <= lo <= LON1 and LAT0 <= la <= LAT1):
                continue
            ii = int(np.argmin(np.abs(lats - la)))
            jj = int(np.argmin(np.abs(lons - lo)))
            sat = float(fields_all[i][ii, jj])
            if not np.isfinite(sat):
                continue
            gsum[code] = gsum.get(code, 0.0) + g["mm"]
            isum[code] = isum.get(code, 0.0) + sat
            ndays[code] = ndays.get(code, 0) + 1
            meta[code] = (la, lo)
    pts, logf = [], []
    for code in gsum:
        if ndays[code] >= 120 and gsum[code] >= 50 and isum[code] >= 50:
            fac = np.clip(gsum[code] / isum[code], 1 / 3.0, 3.0)  # gauge/IMERG
            pts.append(meta[code])
            logf.append(np.log(fac))
    print(f"correction field: {len(pts)} stations qualify", flush=True)
    pts = np.array(pts) if pts else np.zeros((0, 2))
    logf = np.array(logf)
    LO, LA = np.meshgrid(lons, lats)
    F = np.ones(LO.shape)
    if len(pts):
        L = 0.45                                       # deg, IDW length scale
        for ii in range(LO.shape[0]):
            # distances station->cells of this row (vectorized)
            dy = pts[:, 0][:, None] - LA[ii, :][None, :] * 0 - LA[ii, 0]
            dx = pts[:, 1][:, None] - LO[ii, :][None, :]
            w = np.exp(-((dx ** 2 + dy ** 2)) / (2 * L * L))
            w0 = 0.15                                  # unity-pulling background
            F[ii, :] = np.exp((w * logf[:, None]).sum(0) / (w.sum(0) + w0))
    CORR_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CORR_NPZ, F=F, lons=lons, lats=lats)   # lons 0..360
    print(f"correction field: {F.min():.2f}..{F.max():.2f} (x)", flush=True)
    return F


def main() -> int:
    end = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    # full archive span for the correction field; windows use the tail
    nall = len(sorted(IP.DAILY_CACHE.glob("*.npy")))
    days = [end - timedelta(days=k) for k in range(min(nall + 5, 800))][::-1]

    # crop FIRST (basin bbox + padding, squared) — the IMERG field subset
    # below uses these bounds, so they must be final before imerg_stack
    rp = region_paths()
    global LON0, LON1, LAT0, LAT1
    xs = np.concatenate([r[:, 0] % 360 for reg_, rr in rp.items() if reg_ in ORDER for r in rr])
    ys = np.concatenate([r[:, 1] for reg_, rr in rp.items() if reg_ in ORDER for r in rr])
    LAT0, LAT1 = ys.min() - 0.5, ys.max() + 0.5
    lon_pad = max(0.6, ((LAT1 - LAT0) - (xs.max() - xs.min())) / 2)
    LON0, LON1 = xs.min() - lon_pad, xs.max() + lon_pad

    IP.ensure_daily({d for d in days[-35:]})
    fields, lons, lats = imerg_stack(days)
    have = [i for i, f in enumerate(fields) if f is not None]
    if not have:
        print("no IMERG dailies cached")
        return 1
    latest_i = have[-1]
    rdates = np.array([f"{d:%Y-%m-%d}" for d in days])

    F = build_correction(days, fields, rdates, lons, lats)

    print("fetching gauges …", flush=True)
    gauges = fetch_range(end, 100)

    feed_days = [d for d in days if gauges.get(d)]
    imerg_ok = {days[i] for i in have}
    paired = [d for d in feed_days if d in imerg_ok]

    def paired_sum(window):
        need = max(1, int(0.85 * len(window)))
        acc, cnt, meta = {}, {}, {}
        for d in window:
            for code, g in gauges[d].items():
                acc[code] = acc.get(code, 0.0) + g["mm"]
                cnt[code] = cnt.get(code, 0) + 1
                meta[code] = (g["la"], g["lo"])
        gg = {c: {"la": meta[c][0], "lo": meta[c][1], "mm": acc[c]}
              for c in acc if cnt[c] >= need}
        sat = np.nansum([fields[days.index(d)] for d in window], axis=0)
        return gg, sat

    paths = {r: [MplPath(np.column_stack([(ring[:, 0] % 360), ring[:, 1]]))
                 for ring in rings] for r, rings in rp.items() if r in ORDER}

    def assign_region(la, lo):
        for r, ps in paths.items():
            if any(p.contains_point((lo % 360, la)) for p in ps):
                return r
        return None

    import matplotlib.patheffects as pe

    def draw_panel(ax, field, gg, lev, label):
        norm = BoundaryNorm(lev, CMAP.N)
        pm = ax.pcolormesh(lons, lats, field, cmap=CMAP, norm=norm,
                           shading="nearest", transform=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", lw=1.0, color="#333333", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.8,
                       edgecolor="#555555", zorder=4)
        for r, rings in rp.items():
            if r not in ORDER:
                continue
            col = RCOL.get(r, "#222222")
            big = max(rings, key=len)
            for ring in rings:
                ax.plot(ring[:, 0] % 360, ring[:, 1], color=col, lw=1.5,
                        transform=ccrs.PlateCarree(), zorder=4)
            cx, cy = (big[:, 0] % 360).mean(), big[:, 1].mean()
            ax.annotate(r, (cx - 360, cy), ha="center", fontsize=8.5,
                        fontweight="bold", color=col, zorder=7,
                        xycoords=ax.transData,
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground="white")])
        n_in = 0
        for g in (gg or {}).values():
            lo, la, mm = g["lo"] % 360, g["la"], g["mm"]
            if not (LON0 <= lo <= LON1 and LAT0 <= la <= LAT1):
                continue
            n_in += 1
            ax.scatter([lo], [la], c=[mm], cmap=CMAP, norm=norm, s=30,
                       edgecolors="black", linewidths=0.45, zorder=5,
                       transform=ccrs.PlateCarree())
            ax.annotate(f"{mm:.0f}", (lo % 360 - 360, la), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=4.6,
                        color="#111111", zorder=6, xycoords=ax.transData,
                        path_effects=[pe.withStroke(linewidth=1.2,
                                                    foreground="white")])
        ax.set_title(f"{label} · {n_in} gauges", fontsize=11,
                     fontweight="bold", loc="left")
        ax.set_extent([LON0 - 360, LON1 - 360, LAT0, LAT1], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, lw=0.2, color="0.75", alpha=0.5,
                          xlocs=[-79, -77, -75, -73], ylocs=[2, 4, 6, 8])
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}
        return pm

    LEVS = {1: [0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150],
            7: [0, 5, 10, 25, 50, 100, 150, 200, 300, 400, 600],
            14: [0, 10, 25, 50, 100, 150, 250, 350, 500, 700, 1000],
            30: [0, 20, 50, 100, 200, 300, 450, 600, 800, 1000, 1500],
            90: [0, 50, 150, 300, 500, 750, 1000, 1400, 1900, 2500, 3500]}
    for N in [1, 7, 90]:
        win = paired[-N:]
        if not win:
            continue
        gg, raw = paired_sum(win)
        corr = raw * F
        fig = plt.figure(figsize=(16.6, 10.2))
        for k, (fld, ttl) in enumerate([(raw, "raw IMERG"),
                                        (corr, "gauge-corrected IMERG")]):
            ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
            pm = draw_panel(ax, fld, gg, LEVS[N],
                            f"{ttl} — {len(win)}-day total")
            cb = fig.colorbar(pm, ax=ax, orientation="horizontal",
                              fraction=0.04, pad=0.04, aspect=40)
            cb.set_label("mm", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.90, bottom=0.03, left=0.04, right=0.98, wspace=0.08)
        fig.suptitle(f"IMERG vs gauge-corrected IMERG — last {len(win)} paired "
                     f"feed days (through {win[-1]:%b %d}) · gauge totals overlaid "
                     "on the same scale", fontsize=13, fontweight="bold", y=0.97)
        out = OUT_PNG.parent / f"rain_cmp_{N}d.webp"
        fig.savefig(out, dpi=115, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
