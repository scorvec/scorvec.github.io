#!/usr/bin/env python3
"""Area-mean IMERG rainfall over XM's hydrological regions (Colombia).

Consumes the polygons from build_hydro_regions.py (IDEAM subzonas variant
drives the headline series; the HydroBASINS contributing-area variant is
computed alongside for comparison) and the imerg_precip daily 0.1° cache.
Per region: daily mm/day, the harmonic IMERG climatology evaluated on the
same cells, and trailing 30-day percent-of-normal.

Outputs:
    assets/sst/data/colombia_region_rain.json
    assets/sst/colombia_region_rain.webp   (6 panels, 120-day daily series)

    python scripts/sst/hydro_region_rain.py
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
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg_precip as IP
from build_imerg_clim import OUT as CLIM_NC, eval_clim

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_JSON = REPO / "assets" / "sst" / "data" / "colombia_region_rain.json"
OUT_PNG = REPO / "assets" / "sst" / "colombia_region_rain.webp"
DAYS = 120                    # default; --days N extends the JSON series (chart
PLOT_DAYS = 120               # always shows the trailing PLOT_DAYS)
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
COLORS = {"ANTIOQUIA": "#68B79F", "CALDAS": "#4F5BE3", "CARIBE": "#F0A169",
          "CENTRO": "#F5D76E", "ORIENTE": "#C0608D", "VALLE": "#43128F"}


def _axes():
    import xarray as xr
    ds = xr.open_dataset(CLIM_NC)
    return ds["lon"].values, ds["lat"].values, ds["coef"].values


def region_weights(path: Path, lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """cos(lat)-weighted inside-mask per region on the IMERG grid."""
    import geopandas as gpd
    from shapely import contains_xy, prepare
    xx, yy = np.meshgrid(lon, lat)
    coslat = np.cos(np.radians(yy))
    out = {}
    for _, row in gpd.read_file(path).iterrows():
        g = row.geometry
        prepare(g)
        inside = contains_xy(g, xx, yy)
        w = np.where(inside, coslat, 0.0)
        if w.sum() == 0:
            raise SystemExit(f"region {row['name']} matched no grid cells")
        out[row["name"]] = w / w.sum()
    return out


def gauge_correction(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Gauge-correction field F (corrected = IMERG * F) embedded onto an
    arbitrary lat/lon grid: 1.0 outside the field's footprint. Built by
    colombia_rain_map.build_correction (549-station log-IDW); returns all
    ones if the cached NPZ is missing or lacks coordinates."""
    npz = HERE / "data" / "imerg_gauge_corr.npz"
    F = np.ones((len(lats), len(lons)))
    if not npz.exists():
        return F
    z = np.load(npz)
    if "lons" not in z:
        return F
    flon, flat = z["lons"], z["lats"]          # saved in 0..360 convention
    tl = np.asarray(lons) % 360
    ji = {round(v, 3): j for j, v in enumerate(np.round(flon % 360, 3))}
    ii = {round(v, 3): i for i, v in enumerate(np.round(flat, 3))}
    for i, la in enumerate(np.round(np.asarray(lats), 3)):
        si = ii.get(round(float(la), 3))
        if si is None:
            continue
        for j, lo in enumerate(np.round(tl, 3)):
            sj = ji.get(round(float(lo), 3))
            if sj is not None:
                F[i, j] = z["F"][si, sj]
    return F


def gauge_correction_mtime() -> float:
    npz = HERE / "data" / "imerg_gauge_corr.npz"
    return npz.stat().st_mtime if npz.exists() else 0.0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--cached-only", action="store_true",
                    help="use every already-cached daily grid; no downloads")
    args = ap.parse_args(argv)
    lon, lat, coef = _axes()
    w_ideam = region_weights(HERE / "colombia_hydro_regions.geojson", lon, lat)
    w_hybas = region_weights(HERE / "colombia_hydro_regions_hydrobasins.geojson", lon, lat)

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if not args.cached_only:
        # --days only bounds what gets DOWNLOADED; the series always uses the
        # whole cache, so the nightly run never truncates the backfilled record
        IP.ensure_daily({today - timedelta(days=k) for k in range(args.days, 0, -1)})
    days = [datetime.strptime(f.stem, "%Y%m%d").replace(tzinfo=timezone.utc)
            for f in sorted(IP.DAILY_CACHE.glob("*.npy"))]

    dates, grids = [], []
    for d in days:
        g = IP._load(IP.DAILY_CACHE, f"{d:%Y%m%d}")
        if g is not None:
            dates.append(d); grids.append(g)
    if len(dates) < 30:
        print(f"only {len(dates)} daily grids cached — aborting"); return 1
    stack = np.stack(grids)                                   # (t, lat, lon)

    series = {}
    for name in ORDER:
        mm = stack.reshape(len(dates), -1) @ w_ideam[name].ravel()
        mmh = stack.reshape(len(dates), -1) @ w_hybas[name].ravel()
        cl = np.array([float((eval_clim(coef, d.timetuple().tm_yday)
                              * w_ideam[name]).sum()) for d in dates])
        n30 = min(30, len(dates))
        pct30 = 100.0 * mm[-n30:].sum() / max(cl[-n30:].sum(), 0.1)
        series[name] = dict(
            dates=[d.strftime("%Y-%m-%d") for d in dates],
            mm=[round(float(v), 2) for v in mm],
            mm_hydrobasins=[round(float(v), 2) for v in mmh],
            clim=[round(float(v), 2) for v in cl],
            pct_normal_30d=round(float(pct30), 1))
        print(f"  {name}: 30-day {mm[-n30:].sum():.0f} mm "
              f"({pct30:.0f}% of normal)", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "updated": today.strftime("%Y-%m-%d"),
        "source": "GPM IMERG (early/late daily) · regions: XM ListadoRios -> "
                  "IDEAM Zonificación Hidrográfica SZH (headline) + HydroSHEDS "
                  "HydroBASINS lev12 traces (mm_hydrobasins)",
        "regions": series}))

    fig, axs = plt.subplots(3, 2, figsize=(11.6, 9.2), sharex=True)
    dts_all = [datetime.strptime(d, "%Y-%m-%d") for d in series[ORDER[0]]["dates"]]
    p0 = max(0, len(dts_all) - PLOT_DAYS)
    dts = dts_all[p0:]
    for ax, name in zip(axs.ravel(), ORDER):
        s = series[name]
        ax.bar(dts, s["mm"][p0:], width=1.0, color=COLORS[name], alpha=0.75, lw=0)
        ax.plot(dts, s["clim"][p0:], color="0.25", lw=1.4, ls="--", label="climatology")
        ax.plot(dts, s["mm_hydrobasins"][p0:], color="0.35", lw=0.7, alpha=0.65,
                label="HydroBASINS variant")
        ax.set_title(f"{name} — 30-day: {s['pct_normal_30d']:.0f}% of normal",
                     fontsize=10, fontweight="bold", loc="left",
                     color="#b02020" if s["pct_normal_30d"] < 80 else
                           ("#1a7a1a" if s["pct_normal_30d"] > 120 else "k"))
        ax.grid(alpha=0.25); ax.set_ylabel("mm/day", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Basin rainfall over XM hydro regions — GPM IMERG, area-weighted",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "regions = XM river list (servapibi.xm.com.co) mapped to IDEAM Zonificación "
             "Hidrográfica subzonas · thin gray = HydroBASINS lev-12 contributing-area variant",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(OUT_PNG, dpi=115, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG.name} + {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
