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
import os
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
from matplotlib.path import Path as MplPath

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


def region_weights_area(path: Path, lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """cos(lat)-weighted inside-mask per region on the IMERG grid (area basis)."""
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


def region_weights(path: Path, lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Per-region rainfall weights. Default ENERGY basis (each river's catchment
    weighted by its trailing-year AporEner, regulated rivers excluded); falls
    back to the AREA basis when catchments/energy are unavailable. Override with
    CO_REGION_WEIGHTS=area."""
    area = region_weights_area(path, lon, lat)
    if os.environ.get("CO_REGION_WEIGHTS", "energy").lower() == "area":
        return area
    try:
        ew = region_weights_energy(lon, lat, list(area))
    except Exception as e:                               # noqa: BLE001
        print(f"  energy weights failed ({repr(e)[:60]}) — area basis", flush=True)
        ew = None
    return ew if ew else area


# ── energy-weighted region masks ──────────────────────────────────────────────
# The XM regions are unions of river catchments, but their AREA is a poor proxy
# for where inflow ENERGY comes from: CENTRO's largest polygon (61% of area)
# holds both Sogamoso (the best rain responder) and the heavily REGULATED
# Bogotá N.R., whose inflows are operator decisions; VALLE's Calima piece is 39%
# of area for 3% of energy. Weighting each river's catchment by its trailing-year
# AporEner energy — and dropping regulated rivers — lifts the rain→inflow fit in
# both split-halves (CENTRO r .375→.432, VALLE .348→.446; audit 2026-08-18).
# Switch with CO_REGION_WEIGHTS=area|energy (default energy); falls back to area
# whenever catchments or energy are unavailable.
CATCH_GJ = REPO / "colombia_hydro" / "data" / "xm_river_catchments.geojson"
APOR_CACHE = Path.home() / "colombia_hydro" / "raw" / "aporener_daily.json.gz"
DAM_MODELS = REPO / "colombia_hydro" / "data" / "dam_models.json"
EW_CACHE = HERE / "data" / "region_weights_energy.npz"
REGULATED_FALLBACK = {"BOGOTA N.R."}


def _regulated_rivers() -> set:
    """Rivers whose inflow is operator-driven (rain-insensitive): from the
    per-dam models' REGULATED flag, plus a static fallback."""
    out = set(REGULATED_FALLBACK)
    try:
        dm = json.loads(DAM_MODELS.read_text())["params"]
        out |= {r for r, p in dm.items() if p.get("regulated")}
    except Exception:                                   # noqa: BLE001
        pass
    return out


def _river_energy() -> dict:
    """Trailing-365-day mean AporEner per river, GWh/day."""
    import gzip
    with gzip.open(APOR_CACHE, "rt") as f:
        apor = json.load(f)
    days = sorted(apor)[-365:]
    e = {}
    for d in days:
        for r, v in apor[d].items():
            e[r] = e.get(r, 0.0) + v / len(days) / 1e6
    return e


def region_weights_energy(lon, lat, regions):
    """{region: weight grid} — cos-lat catchment masks combined in proportion
    to each river's energy, regulated rivers excluded. None if unavailable."""
    import hashlib
    key = hashlib.md5(f"{lon[0]}_{lon[-1]}_{len(lon)}_{lat[0]}_{lat[-1]}_{len(lat)}"
                      .encode()).hexdigest()[:12]
    if EW_CACHE.exists():
        try:
            z = np.load(EW_CACHE, allow_pickle=True)
            if str(z["key"]) == key and float(z["apor_mtime"]) == APOR_CACHE.stat().st_mtime:
                d = z["W"].item()
                if all(r in d for r in regions):
                    return {r: d[r] for r in regions}
        except Exception:                               # noqa: BLE001
            pass
    if not CATCH_GJ.exists() or not APOR_CACHE.exists():
        return None
    egy = _river_energy()
    reg = _regulated_rivers()
    gj = json.loads(CATCH_GJ.read_text())
    LO, LA = np.meshgrid(np.asarray(lon), np.asarray(lat))
    pts = np.column_stack([LO.ravel() % 360, LA.ravel()])
    coslat = np.cos(np.deg2rad(LA))
    acc = {r: np.zeros(LO.shape) for r in regions}
    used = {r: 0.0 for r in regions}
    for ft in gj["features"]:
        pr = ft["properties"]
        rg, riv = pr.get("region"), pr.get("river")
        if rg not in acc or riv in reg:
            continue
        e = egy.get(riv, 0.0)
        if e <= 0:
            continue
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        m = np.zeros(LO.shape, bool)
        for pl in polys:
            ring = np.array(pl[0])
            if ring.ndim != 2 or len(ring) < 4:
                continue
            m |= MplPath(np.column_stack([ring[:, 0] % 360, ring[:, 1]])
                         ).contains_points(pts).reshape(LO.shape)
        w = np.where(m, coslat, 0.0)
        if w.sum() == 0:
            continue
        acc[rg] += e * (w / w.sum())
        used[rg] += e
    out = {}
    for r in regions:
        if used[r] <= 0 or acc[r].sum() == 0:
            return None                                  # incomplete -> caller falls back
        out[r] = acc[r] / acc[r].sum()
    try:
        EW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EW_CACHE, W=np.array(out, dtype=object), key=key,
                            apor_mtime=APOR_CACHE.stat().st_mtime)
    except Exception:                                    # noqa: BLE001
        pass
    return out


def gauge_blend_field(sat_corr: np.ndarray, day: str,
                      lons: np.ndarray, lats: np.ndarray,
                      k: float = 0.5) -> np.ndarray:
    """Blend direct gauge measurements into the corrected-satellite field,
    weighted by local gauge density: cells with n co-located stations get
    w = n/(n+k) gauge weight (k=0.5 -> 0.67 for one station, 0.8 for two),
    pure corrected satellite where the network is absent. `day` is
    YYYYMMDD; missing gauge day -> field returned unchanged. Outlier guard
    drops station-days wildly above the satellite (stuck counters)."""
    from ideam_gauges import CACHE as GCACHE
    f = GCACHE / f"{day}.json"
    if not f.exists():
        return sat_corr
    try:
        g = json.loads(f.read_text())
    except ValueError:
        return sat_corr
    if not g:
        return sat_corr
    tl = np.asarray(lons) % 360
    acc = np.zeros_like(sat_corr)
    cnt = np.zeros_like(sat_corr)
    for st in g.values():
        la, lo, mm = st["la"], st["lo"] % 360, st["mm"]
        if not (tl.min() <= lo <= tl.max() and lats.min() <= la <= lats.max()):
            continue
        i = int(np.argmin(np.abs(lats - la)))
        j = int(np.argmin(np.abs(tl - lo)))
        sat = sat_corr[i, j]
        if np.isfinite(sat) and mm > max(100.0, 6.0 * (sat + 10.0)):
            continue                       # stuck/cumulative counter
        acc[i, j] += mm
        cnt[i, j] += 1
    w = cnt / (cnt + k)
    with np.errstate(invalid="ignore"):
        gv = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return np.where(cnt > 0, w * gv + (1 - w) * sat_corr, sat_corr)


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
    fig.suptitle("Basin rainfall over XM hydro regions — GPM IMERG, energy-weighted catchments",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "regions = XM river list (servapibi.xm.com.co) mapped to IDEAM Zonificación "
             "Hidrográfica subzonas · cells weighted by each river's trailing-365d "
             "generation energy (regulated rivers excluded) · thin gray = "
             "HydroBASINS lev-12 contributing-area variant",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(OUT_PNG, dpi=115, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG.name} + {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
