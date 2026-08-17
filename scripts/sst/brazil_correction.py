#!/usr/bin/env python3
"""Brazil gauge-corrected IMERG: correction field + bias analysis.

Mirrors the Colombia methodology on the INMET network: per-station
ratio of summed gauge vs summed IMERG-at-station over all paired
archive days (>=120 days, >=50 mm both sides, ratio clipped 1/3..3),
log-IDW interpolated (L=0.7 deg — sparser network than Colombia) with
a unity-pulling background. corrected = IMERG * F.

Bias analysis outputs (the deliverable):
  - per-SIN-basin median station factors + spread (bias geography)
  - leave-one-station-out CV: raw vs corrected error, % improved
  - year-1 vs year-2 factor stability
  - brazil_hydro/bias_map.webp   (F field + station ratios + basins)
  - brazil_hydro/rain_cmp_30d.webp (raw vs corrected, gauges overlaid)
  - cache ~/brazil_hydro/raw/imerg_gauge_corr.npz (F + coords, weekly)

    python scripts/sst/brazil_correction.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                   # noqa: E402
from brazil_model import basin_weights, MAJORS              # noqa: E402
from matplotlib.path import Path as MplPath                 # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "brazil_hydro"
GCACHE = PRIV / "raw" / "gauges"
BASINS_GJ = PRIV / "out" / "brazil_basins.geojson"
CORR_NPZ = PRIV / "raw" / "imerg_gauge_corr.npz"
OUT_STATS = PRIV / "out" / "gauge_bias_analysis.json"
SITE = REPO / "brazil_hydro"
CORR_MAX_AGE_DAYS = 7
L_IDW = 0.7
W0 = 0.15

RAIN_COLS = ["#f6f7f5", "#d9edcf", "#a5d99b", "#57b86b", "#1f9e89",
             "#2380b9", "#20539c", "#5b3f9e", "#93357f", "#c2185b"]
RAIN_CMAP = ListedColormap(RAIN_COLS)
RAIN_CMAP.set_over("#7a1240")
LEV_30D = [0, 10, 25, 50, 100, 150, 200, 300, 400, 600, 900]
FAC_COLS = ["#7a4a12", "#a8702a", "#cd9d57", "#e8cf9e", "#f6f5f0",
            "#c9e7c2", "#7cc87c", "#2e9e4f", "#1d6fb8", "#6a3d9a"]
FAC_CMAP = ListedColormap(FAC_COLS)
LEV_FAC = [0.33, 0.5, 0.67, 0.8, 0.95, 1.05, 1.25, 1.5, 2.0, 2.5, 3.0]


def station_sums():
    """Per-station paired sums + optional split halves over the archive."""
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml])
    lats = np.sort(IP._LAT[mt])
    gsum, isum, nd, meta = {}, {}, {}, {}
    halves = {}                                 # code -> [g1,i1,g2,i2]
    paired_files = [f for f in files if (GCACHE / f"{f.stem}.json").exists()]
    half = paired_files[len(paired_files) // 2].stem
    for f in paired_files:
        day = json.loads((GCACHE / f"{f.stem}.json").read_text())
        if not day:
            continue
        g = np.load(f)
        second = f.stem >= half
        for code, st in day.items():
            la, lo = st["la"], st["lo"]
            if not (lons.min() <= lo <= lons.max()
                    and lats.min() <= la <= lats.max()):
                continue
            i = int(np.argmin(np.abs(lats - la)))
            j = int(np.argmin(np.abs(lons - lo)))
            sat = float(g[i, j])
            if not np.isfinite(sat):
                continue
            gsum[code] = gsum.get(code, 0.0) + st["mm"]
            isum[code] = isum.get(code, 0.0) + sat
            nd[code] = nd.get(code, 0) + 1
            meta[code] = (la, lo)
            h = halves.setdefault(code, [0.0, 0.0, 0.0, 0.0])
            if second:
                h[2] += st["mm"]
                h[3] += sat
            else:
                h[0] += st["mm"]
                h[1] += sat
    return gsum, isum, nd, meta, halves, lons, lats, len(paired_files), half


def main() -> int:
    (gsum, isum, nd, meta, halves, lons, lats,
     npaired, half) = station_sums()
    codes = [c for c in gsum if nd[c] >= 120 and gsum[c] >= 50
             and isum[c] >= 50]
    print(f"qualifying stations: {len(codes)} of {len(gsum)} "
          f"({npaired} paired days)", flush=True)
    pts = np.array([meta[c] for c in codes])
    ratio = np.array([gsum[c] / isum[c] for c in codes])
    logf = np.log(np.clip(ratio, 1 / 3, 3.0))

    # ── bias geography: factors by SIN basin ────────────────────────────────
    gj = json.loads(BASINS_GJ.read_text())
    paths = {}
    for ft in gj["features"]:
        if ft["properties"]["basin"] in MAJORS:
            paths[ft["properties"]["basin"]] = [
                MplPath(np.array(p[0])) for p in ft["geometry"]["coordinates"]]
    by_basin = {}
    for k, c in enumerate(codes):
        la, lo = pts[k]
        for b, ps in paths.items():
            if any(p.contains_point((lo, la)) for p in ps):
                by_basin.setdefault(b, []).append(ratio[k])
                break
    basin_stats = {b: {"n": len(v),
                       "median_factor": round(float(np.median(v)), 2),
                       "iqr": [round(float(np.percentile(v, 25)), 2),
                               round(float(np.percentile(v, 75)), 2)]}
                   for b, v in sorted(by_basin.items()) if len(v) >= 5}

    # ── LOSO cross-validation ───────────────────────────────────────────────
    err_raw, err_cor = [], []
    for k in range(len(codes)):
        dx = pts[:, 1] - pts[k, 1]
        dy = pts[:, 0] - pts[k, 0]
        w = np.exp(-(dx ** 2 + dy ** 2) / (2 * L_IDW * L_IDW))
        w[k] = 0.0
        F_pred = np.exp((w * logf).sum() / (w.sum() + W0))
        err_raw.append(abs(np.log(ratio[k])))
        err_cor.append(abs(np.log(ratio[k] / F_pred)))
    err_raw, err_cor = np.array(err_raw), np.array(err_cor)
    loso = {"raw_median_factor_error": round(float(np.exp(np.median(err_raw))), 2),
            "corrected_median_factor_error": round(float(np.exp(np.median(err_cor))), 2),
            "pct_stations_improved": round(100 * float((err_cor < err_raw).mean()), 0),
            "median_error_reduction_pct": round(
                100 * (1 - float(np.median(err_cor) / np.median(err_raw))), 0)}

    # ── temporal stability ──────────────────────────────────────────────────
    both = [c for c in codes if halves[c][0] > 25 and halves[c][1] > 25
            and halves[c][2] > 25 and halves[c][3] > 25]
    f1 = np.log(np.clip([halves[c][0] / halves[c][1] for c in both], 1/3, 3))
    f2 = np.log(np.clip([halves[c][2] / halves[c][3] for c in both], 1/3, 3))
    stability = round(float(np.corrcoef(f1, f2)[0, 1]), 2) if len(both) > 30 else None

    # ── the field ───────────────────────────────────────────────────────────
    LO, LA = np.meshgrid(lons, lats)
    F = np.ones(LO.shape)
    for i in range(LO.shape[0]):
        dy = pts[:, 0][:, None] - LA[i, 0]
        dx = pts[:, 1][:, None] - LO[i, :][None, :]
        w = np.exp(-((dx ** 2 + dy ** 2)) / (2 * L_IDW * L_IDW))
        F[i, :] = np.exp((w * logf[:, None]).sum(0) / (w.sum(0) + W0))
    np.savez_compressed(CORR_NPZ, F=F, lons=lons, lats=lats)
    print(f"field: {F.min():.2f}..{F.max():.2f}", flush=True)

    stats = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "n_stations": len(codes), "paired_days": npaired,
             "split_date": half,
             "overall_median_factor": round(float(np.median(ratio)), 2),
             "basin_factors": basin_stats, "loso_cv": loso,
             "year1_vs_year2_r": stability,
             "idw": {"L_deg": L_IDW, "w0": W0, "clip": [0.33, 3.0]}}
    OUT_STATS.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1)[:800], flush=True)

    # ── figures ─────────────────────────────────────────────────────────────
    ext = [-75.5, -33.5, -34.5, 6.0]
    rings = {ft["properties"]["basin"]: ft["geometry"]["coordinates"]
             for ft in gj["features"] if ft["properties"]["basin"] in MAJORS}

    def decorate(ax):
        ax.coastlines(resolution="50m", lw=0.8, color="#2b2b2b", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                       edgecolor="#4a4a4a", zorder=4)
        for b, polys in rings.items():
            for p in polys:
                arr = np.array(p[0])
                ax.plot(arr[:, 0], arr[:, 1], color="#13273d", lw=1.0,
                        transform=ccrs.PlateCarree(), zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground="white")])
        ax.set_extent(ext, ccrs.PlateCarree())

    # 1) bias map: F field + station dots by ratio
    fig = plt.figure(figsize=(10.5, 9.0))
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.86], projection=ccrs.PlateCarree())
    norm = BoundaryNorm(LEV_FAC, FAC_CMAP.N)
    pm = ax.pcolormesh(lons, lats, F, cmap=FAC_CMAP, norm=norm,
                       shading="nearest", transform=ccrs.PlateCarree(),
                       rasterized=True)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6ebf0",
                   zorder=3)
    decorate(ax)
    ax.scatter(pts[:, 1], pts[:, 0], c=ratio, cmap=FAC_CMAP, norm=norm,
               s=14, edgecolors="black", linewidths=0.4, zorder=6,
               transform=ccrs.PlateCarree())
    ax.set_title(f"IMERG bias over Brazil — gauge/satellite factor, "
                 f"{len(codes)} INMET stations ({npaired} paired days)\n"
                 "field = log-IDW of station ratios (relaxes to ×1 away from "
                 "stations) · brown = satellite too wet · green/blue = too dry",
                 fontsize=11, fontweight="bold", loc="left")
    cax = fig.add_axes([0.15, 0.045, 0.7, 0.018])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cax.set_title("correction factor (corrected = IMERG × F)", fontsize=8)
    fig.savefig(SITE / "bias_map.webp", dpi=115)
    plt.close(fig)

    # 2) raw vs corrected 30-day totals with gauges
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    paired = [f for f in files if (GCACHE / f"{f.stem}.json").exists()][-30:]
    tot = np.nansum([np.load(f) for f in paired], axis=0)
    acc, cnt, metag = {}, {}, {}
    for f in paired:
        for code, st in json.loads((GCACHE / f"{f.stem}.json").read_text()).items():
            acc[code] = acc.get(code, 0.0) + st["mm"]
            cnt[code] = cnt.get(code, 0) + 1
            metag[code] = (st["la"], st["lo"])
    gg = {c: v for c, v in acc.items() if cnt[c] >= 26}
    fig = plt.figure(figsize=(14.5, 8.6))
    for k, (field, ttl) in enumerate([(tot, "Raw IMERG"),
                                      (tot * F, "Gauge-corrected")]):
        ax = fig.add_axes([0.035 + k * 0.485, 0.09, 0.45, 0.82],
                          projection=ccrs.PlateCarree())
        n2 = BoundaryNorm(LEV_30D, RAIN_CMAP.N)
        pm = ax.pcolormesh(lons, lats, field, cmap=RAIN_CMAP, norm=n2,
                           shading="nearest", transform=ccrs.PlateCarree(),
                           rasterized=True)
        decorate(ax)
        seen = set()
        for c, mm in sorted(gg.items(), key=lambda kv: -kv[1]):
            la, lo = metag[c]
            key = (round(lo / 1.2), round(la / 1.2))
            if key in seen:
                continue
            seen.add(key)
            ax.scatter([lo], [la], c=[mm], cmap=RAIN_CMAP, norm=n2, s=22,
                       edgecolors="black", linewidths=0.5, zorder=6,
                       transform=ccrs.PlateCarree())
            ax.annotate(f"{mm:.0f}", (lo, la), xytext=(0, 3.5),
                        textcoords="offset points", ha="center",
                        fontsize=5.6, fontweight="bold", color="white",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground="black")])
        ax.set_title(f"{ttl} — last 30 paired days (mm)", fontsize=11,
                     fontweight="bold", loc="left")
        cax = fig.add_axes([0.06 + k * 0.485, 0.045, 0.4, 0.016])
        cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="max")
        cb.set_ticks(LEV_30D[:-1])
        cb.ax.tick_params(labelsize=7.5)
    fig.savefig(SITE / "rain_cmp_30d.webp", dpi=115)
    plt.close(fig)
    print("wrote brazil_hydro/bias_map.webp + rain_cmp_30d.webp", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
