#!/usr/bin/env python3
"""IMERG rainfall tracker for the Lake Gatún watershed (Panama Canal).

Rides entirely on data we already pull: imerg_precip's binned region caches
(the Colombia+Brazil framing contains Panama) and build_imerg_clim's harmonic
2001–2025 climatology. No extra granule types — just a deeper daily cache
(imerg_precip.DAILY_KEEP_H covers the 90-day window) and a tighter frame.

Products (assets/sst/):
  anim/gatun_precip_{1h,24h,5d,7d,14d,30d,mtd,90d}/  accumulation loops
  anim/gatun_anom_{1d,5d,7d,14d,30d,mtd,90d}/        anomaly loops (vs clim)
  anim/gatun_precip_manifest.json, gatun_anom_manifest.json   (sst_anim.html)
  gatun_rain_level.webp    watershed rainfall anomaly vs lake level (8 years)
  gatun_rain_daily.csv     committed daily basin-mean rainfall record

    python scripts/sst/imerg_gatun.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg_precip as IP                       # shared data layer (caches, grid, cmap)
import imerg_precip_anom as IPA                 # anomaly cmap + clim accumulation
from build_imerg_clim import OUT as CLIM_NC

ANIM_ROOT = IP.ANIM_ROOT
ASSETS = ANIM_ROOT.parent                       # assets/sst
PC = ccrs.PlateCarree()

# Gatún framing: the isthmus + both approaches (81°W–77.4°W, 7.4–10.6°N)
GCFG = dict(extent=(279.0, 282.6, 7.4, 10.6), clon=280.8, figsize=(7.8, 7.4),
            dlon=1, dlat=1, scale="10m")
# Chagres/Gatún watershed averaging box for the daily rain series (°E 0–360)
BASIN = dict(lat=(9.0, 9.6), lon=(280.0, 280.7))          # 80.0°W–79.3°W

# accumulation windows: id, source kind, trailing units, frame step h, loop span h, levels mm
WINDOWS = [
    dict(id="gatun_precip_1h",  kind="hourly", units=1,  step_h=1,  span_h=48,
         label="1 hour",   levels=[0.5, 1, 2, 4, 7, 10, 15, 20, 30, 45, 65]),
    dict(id="gatun_precip_24h", kind="hourly", units=24, step_h=3,  span_h=72,
         label="24 hours", levels=[1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200]),
    dict(id="gatun_precip_5d",  kind="daily",  units=5,  step_h=24, span_h=240,
         label="5 days",   levels=[2, 5, 10, 20, 40, 70, 100, 150, 225, 325, 450]),
    dict(id="gatun_precip_7d",  kind="daily",  units=7,  step_h=24, span_h=240,
         label="7 days",   levels=[2, 5, 10, 20, 40, 70, 100, 150, 250, 350, 500]),
    dict(id="gatun_precip_14d", kind="daily",  units=14, step_h=24, span_h=240,
         label="14 days",  levels=[5, 10, 20, 40, 70, 100, 175, 275, 400, 550, 750]),
    dict(id="gatun_precip_30d", kind="daily",  units=30, step_h=24, span_h=240,
         label="30 days",  levels=[10, 25, 50, 100, 175, 275, 400, 550, 750, 1000, 1300]),
    dict(id="gatun_precip_mtd", kind="mtd",    units=31, step_h=24, span_h=240,
         label="month-to-date", levels=[2, 5, 10, 25, 50, 100, 175, 275, 400, 550, 750]),
    dict(id="gatun_precip_90d", kind="daily",  units=90, step_h=24, span_h=168,
         label="90 days",  levels=[50, 100, 200, 350, 550, 800, 1100, 1450, 1850, 2300, 2800]),
]
DEFAULT_WINDOW = "gatun_precip_24h"

# anomaly windows (daily-based): id, N days, loop span (frames), symmetric mm levels
ANOM_WINDOWS = [
    dict(id="gatun_anom_1d",  days=1,  span=10, label="1-day",
         levels=[3, 7, 12, 20, 35, 60]),
    dict(id="gatun_anom_5d",  days=5,  span=10, label="5-day",
         levels=[5, 15, 30, 50, 85, 140]),
    dict(id="gatun_anom_7d",  days=7,  span=10, label="7-day",
         levels=[5, 15, 30, 60, 100, 160]),
    dict(id="gatun_anom_14d", days=14, span=10, label="14-day",
         levels=[10, 20, 40, 70, 120, 200]),
    dict(id="gatun_anom_30d", days=30, span=10, label="30-day",
         levels=[15, 30, 60, 110, 180, 300]),
    dict(id="gatun_anom_mtd", days=31, mtd=True, span=10, label="month-to-date",
         levels=[15, 30, 60, 110, 180, 300]),
    dict(id="gatun_anom_90d", days=90, span=7,  label="90-day",
         levels=[30, 60, 120, 220, 350, 550]),
]
DEFAULT_ANOM = "gatun_anom_30d"


# ── sub-grid extraction from the shared region grid ───────────────────────────
def _sub_axes():
    ml, mt = IP._grid_axes()
    LONb, LATb = IP._LON[ml], IP._LAT[mt]
    glo0, glo1, gla0, gla1 = GCFG["extent"]
    sl = (LONb % 360 >= glo0) & (LONb % 360 <= glo1)
    st = (LATb >= gla0) & (LATb <= gla1)
    return sl, st, LONb[sl], LATb[st]


def _sub(field: np.ndarray) -> np.ndarray:
    sl, st, _, _ = _sub_axes()
    return field[np.ix_(st, sl)]


def _basin_mask():
    ml, mt = IP._grid_axes()
    LONb, LATb = IP._LON[ml], IP._LAT[mt]
    sl = (LONb % 360 >= BASIN["lon"][0]) & (LONb % 360 <= BASIN["lon"][1])
    st = (LATb >= BASIN["lat"][0]) & (LATb <= BASIN["lat"][1])
    return sl, st


def basin_mean(field: np.ndarray) -> float:
    sl, st = _basin_mask()
    return float(field[np.ix_(st, sl)].mean())


# ── rendering ─────────────────────────────────────────────────────────────────
def _geo(ax, dark: bool):
    """Gatún-frame geography: 10m coasts/borders + Natural Earth lakes (Gatún,
    Alajuela) outlined, with a halo'd label so it reads on either background."""
    coast, border, lake = (("#b6b6b6", "#c2c2c2", "#7fd8e8") if dark
                           else ("#555", "#666", "#0a6d8f"))
    ax.coastlines(linewidth=0.8, color=coast, resolution="10m")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor=border)
    lakes = cfeature.NaturalEarthFeature("physical", "lakes", "10m", facecolor="none")
    ax.add_feature(lakes, edgecolor=lake, linewidth=1.2, zorder=4)
    ax.text(-79.88, 9.05, "Lake Gatun", transform=PC, fontsize=7.5, color=lake,
            ha="center", va="top", zorder=5, fontweight="bold",
            path_effects=[mpe.withStroke(linewidth=1.8,
                                         foreground="#0d0d16" if dark else "white")])


def _frame_axes(dark: bool):
    lo0, lo1, la0, la1 = GCFG["extent"]; clon = GCFG["clon"]
    proj = ccrs.PlateCarree(central_longitude=clon)
    fig = plt.figure(figsize=GCFG["figsize"])
    ax = plt.axes(projection=proj)
    ax.set_extent([IP._offset(lo0, clon), IP._offset(lo1, clon), la0, la1], crs=proj)
    ax.set_facecolor("#0d0d16" if dark else "#f3f1ec")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.35,
                      linestyle=(0, (3, 3)))
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([((t + 180) % 360) - 180
                  for t in range(int(lo0), int(lo1) + 1) if t % GCFG["dlon"] == 0])
    gl.ylocator = mticker.FixedLocator(range(int(la0), int(la1) + 1, GCFG["dlat"]))
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return fig, ax


def render_total(field: np.ndarray, vlabel: str, out: Path, win: dict):
    sl, st, LON, LAT = _sub_axes()
    off = IP._offset(LON, GCFG["clon"])
    levels = win["levels"]
    norm = BoundaryNorm(levels, IP.PRECIP_CMAP.N, extend="max")
    fig, ax = _frame_axes(dark=True)
    wet = np.ma.masked_less(field, levels[0])
    cf = ax.contourf(off, LAT, wet, levels=levels, cmap=IP.PRECIP_CMAP, norm=norm,
                     extend="max", transform=ccrs.PlateCarree(central_longitude=GCFG["clon"]))
    _geo(ax, dark=True)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=40, extend="max")
    cb.set_label("accumulated precip (mm)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Lake Gatun watershed — IMERG {win['label']} precipitation  ·  {vlabel}",
                 fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def render_anom(field: np.ndarray, vlabel: str, out: Path, win: dict):
    sl, st, LON, LAT = _sub_axes()
    off = IP._offset(LON, GCFG["clon"])
    lev = [-x for x in reversed(win["levels"])] + list(win["levels"])
    norm = BoundaryNorm(lev, IPA.ANOM_CMAP.N, extend="both")
    fig, ax = _frame_axes(dark=False)
    cf = ax.contourf(off, LAT, field, levels=lev, cmap=IPA.ANOM_CMAP, norm=norm,
                     extend="both", transform=ccrs.PlateCarree(central_longitude=GCFG["clon"]))
    _geo(ax, dark=False)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=40, extend="both")
    cb.set_label("precip anomaly vs 2001–2025 climatology (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Lake Gatun watershed — IMERG {win['label']} precip anomaly  ·  {vlabel}",
                 fontsize=9.5, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


# ── loops ─────────────────────────────────────────────────────────────────────
def _units(w: dict, ft: datetime, step: timedelta) -> int:
    return (ft - step).day if w["kind"] == "mtd" else w["units"]


def build_totals(anchor_h: datetime, anchor_day: datetime) -> dict:
    regions = {}
    for w in WINDOWS:
        cache = IP.HOURLY_CACHE if w["kind"] == "hourly" else IP.DAILY_CACHE
        step = timedelta(hours=1) if w["kind"] == "hourly" else timedelta(hours=24)
        fmt = "%Y%m%d%H" if w["kind"] == "hourly" else "%Y%m%d"
        base = anchor_h if w["kind"] == "hourly" else anchor_day
        n = w["span_h"] // w["step_h"]
        ftimes = [base - k * timedelta(hours=w["step_h"]) for k in range(n)][::-1]
        need = set()
        for ft in ftimes:
            for k in range(1, _units(w, ft, step) + 1):
                need.add((ft - k * step).replace(minute=0, second=0, microsecond=0))
        (IP.ensure_hourly if w["kind"] == "hourly" else IP.ensure_daily)(need)

        anim = ANIM_ROOT / w["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            u = _units(w, ft, step)
            if not fp.exists():
                field = IP.trailing_sum(cache, ft, u, step, fmt)
                if field is None:
                    continue
                if w["kind"] == "mtd":
                    e = ft - step
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                elif w["kind"] == "daily":
                    vlabel = f"{(ft - timedelta(days=u)):%b %d} – {(ft - step):%b %d, %Y}"
                else:
                    vlabel = f"ending {ft:%Y-%m-%d %HZ}"
                render_total(_sub(field), vlabel, fp, w)
                print(f"  {w['id']}: rendered {tag}", flush=True)
            d = ft
            label = (f"MTD → {(d - step):%b %-d}" if w["kind"] == "mtd"
                     else f"{(d - timedelta(days=u)):%b %d} – {(d - step):%b %d}"
                     if w["kind"] == "daily" else f"{d:%-d %b %HZ}")
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": f"{d:%Y-%m-%d}", "label": label})
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[w["id"]] = {"label": f"{w['label']} accumulation", "frames": entries}
    return regions


def build_anoms(coef: np.ndarray, anchor_day: datetime) -> dict:
    regions = {}
    step = timedelta(days=1)
    for w in ANOM_WINDOWS:
        ftimes = [anchor_day - k * step for k in range(w["span"])][::-1]
        need = set()
        for ft in ftimes:
            nd = (ft - step).day if w.get("mtd") else w["days"]
            for k in range(1, nd + 1):
                need.add((ft - k * step).replace(hour=0, minute=0, second=0, microsecond=0))
        IP.ensure_daily(need)
        anim = ANIM_ROOT / w["id"]; anim.mkdir(parents=True, exist_ok=True)
        entries = []
        for ft in ftimes:
            tag = ft.strftime("%Y%m%d%H")
            fp = anim / f"{tag}.webp"
            nd = (ft - step).day if w.get("mtd") else w["days"]
            if not fp.exists():
                recent = IP.trailing_sum(IP.DAILY_CACHE, ft, nd, step, "%Y%m%d")
                if recent is None:
                    continue
                anom = recent - IPA.clim_accum(coef, ft, nd)
                if w.get("mtd"):
                    e = ft - step
                    vlabel = f"{e.replace(day=1):%b %-d} – {e:%b %-d, %Y}"
                else:
                    vlabel = f"{(ft - timedelta(days=nd)):%b %d} – {(ft - step):%b %d, %Y}"
                render_anom(_sub(anom), vlabel, fp, w)
                print(f"  {w['id']}: rendered {tag}", flush=True)
            label = (f"MTD → {(ft - step):%b %-d}" if w.get("mtd")
                     else f"{(ft - timedelta(days=nd)):%b %d} – {(ft - step):%b %d}")
            entries.append({"idx": len(entries), "file": fp.name,
                            "date": f"{ft:%Y-%m-%d}", "label": label})
        keep = {ft.strftime("%Y%m%d%H") for ft in ftimes}
        for old in anim.glob("*.webp"):
            if old.stem not in keep:
                old.unlink()
        regions[w["id"]] = {"label": f"{w['label']} anomaly", "frames": entries}
    return regions


# ── rainfall vs lake level time series ────────────────────────────────────────
RAIN_CSV = ASSETS / "gatun_rain_daily.csv"
ACP_HISTORY = "https://evtms-rpts.pancanal.com/eng/h2o/Download_Gatun_Lake_Water_Level_History.csv"
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point"
             "?parameters=PRECTOTCORR&community=RE&format=JSON"
             "&latitude={lat}&longitude={lon}&start={start}&end={end}")
POWER_PTS = [(9.1, -79.9), (9.4, -79.7), (9.2, -79.5), (9.5, -79.9)]  # across the watershed
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def _load_rain_csv() -> dict:
    rows = {}
    if RAIN_CSV.exists():
        for r in csv.DictReader(io.StringIO(RAIN_CSV.read_text())):
            rows[r["date"]] = (float(r["mm"]), r["src"])
    return rows


def update_rain_series() -> pd.DataFrame:
    """Maintain the committed daily basin-rain CSV: NASA POWER (GPM-based) for
    history, overwritten by IMERG Early basin means wherever our cache has the
    day. Returns the merged series as a DataFrame indexed by date."""
    rows = _load_rain_csv()

    # 1) POWER backfill, once (2001 → present, four points averaged)
    if not any(src == "power" for _, src in rows.values()):
        print("  POWER backfill (2001→present, 4 basin points)…", flush=True)
        end = datetime.now(timezone.utc).strftime("%Y%m%d")
        pts = []
        for lat, lon in POWER_PTS:
            with urllib.request.urlopen(
                    POWER_URL.format(lat=lat, lon=lon, start="20010101", end=end),
                    timeout=180) as r:
                d = json.load(r)["properties"]["parameter"]["PRECTOTCORR"]
            pts.append({k: v for k, v in d.items() if v is not None and v >= 0})
        common = set.intersection(*[set(p) for p in pts])
        for k in sorted(common):
            iso = f"{k[:4]}-{k[4:6]}-{k[6:]}"
            if iso not in rows or rows[iso][1] == "power":
                rows[iso] = (round(float(np.mean([p[k] for p in pts])), 2), "power")

    # 2) IMERG basin means from the daily cache (authoritative where present)
    for f in sorted(IP.DAILY_CACHE.glob("*.npy")):
        iso = f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:]}"
        rows[iso] = (round(basin_mean(np.load(f)), 2), "imerg")

    with open(RAIN_CSV, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["date", "mm", "src"])
        for k in sorted(rows):
            wtr.writerow([k, rows[k][0], rows[k][1]])
    ser = pd.Series({pd.Timestamp(k): v[0] for k, v in rows.items()}).sort_index()
    print(f"  rain series: {len(ser)} days ({ser.index[0]:%Y-%m-%d} → {ser.index[-1]:%Y-%m-%d})")
    return ser


def _doy_clim(ser: pd.Series) -> np.ndarray:
    """Smooth day-of-year normal (mean + 2 harmonics) from the series itself."""
    doy = ser.index.dayofyear.values.clip(max=365) - 1
    means = np.zeros(365)
    for d in range(365):
        v = ser.values[doy == d]
        means[d] = np.nanmean(v) if len(v) else np.nan
    x = 2 * np.pi * np.arange(365) / 365
    A = np.column_stack([np.ones(365), np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)])
    ok = ~np.isnan(means)
    beta, *_ = np.linalg.lstsq(A[ok], means[ok], rcond=None)
    return A @ beta


def _oni_episodes():
    seas_month = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                  "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    try:
        with urllib.request.urlopen(ONI_URL, timeout=60) as r:
            lines = r.read().decode().splitlines()[1:]
    except Exception:                                        # noqa: BLE001
        return []
    pts = []
    for line in lines:
        p = line.split()
        if len(p) == 4 and p[0] in seas_month:
            pts.append((pd.Timestamp(int(p[1]), seas_month[p[0]], 15), float(p[3])))
    spans, run = [], None
    for t, a in pts:
        if a >= 0.5:
            run = [run[0], t] if run else [t, t]
        else:
            if run:
                spans.append(tuple(run)); run = None
    if run:
        spans.append(tuple(run))
    return spans


def _lake_levels() -> pd.Series:
    import ssl
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(ACP_HISTORY, timeout=120, context=ctx) as r:
            text = r.read().decode()
    except Exception:                                        # noqa: BLE001
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(ACP_HISTORY, timeout=120, context=ctx) as r:
            text = r.read().decode()
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            v = float(row["GATUN_LAKE_LEVEL(FEET)"])
        except (ValueError, KeyError):
            continue
        if 75 <= v <= 90:                                   # ACP missing-data sentinels
            out[pd.Timestamp(row["DATE_LOG"])] = v
    return pd.Series(out).sort_index()


def rain_level_chart(out: Path, years: int = 8):
    rain = update_rain_series()
    clim = _doy_clim(rain)
    doy = rain.index.dayofyear.values.clip(max=365) - 1
    anom = rain - pd.Series(clim[doy], index=rain.index)
    a90 = anom.rolling(90, min_periods=60).sum()
    a30 = anom.rolling(30, min_periods=20).sum()
    lake = _lake_levels()

    t0 = pd.Timestamp(datetime.now(timezone.utc).year - years, 1, 1)
    a90, a30, lake = a90[a90.index >= t0], a30[a30.index >= t0], lake[lake.index >= t0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.4), dpi=100, sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    for x0, x1 in _oni_episodes():
        if x1 < t0:
            continue
        for ax in (ax1, ax2):
            ax.axvspan(max(x0, t0), x1, color="#e8833a", alpha=0.10, lw=0)
    ax1.fill_between(a90.index, a90.values, 0, where=a90.values >= 0,
                     color="#1f8f8f", alpha=0.75, lw=0)
    ax1.fill_between(a90.index, a90.values, 0, where=a90.values < 0,
                     color="#9c6b1e", alpha=0.8, lw=0)
    ax1.plot(a30.index, a30.values, color="#444", lw=0.7, alpha=0.7)
    ax1.axhline(0, color="#333", lw=0.8)
    ax1.set_ylabel("rain anomaly (mm)", fontsize=10)
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_title("Lake Gatun watershed rainfall vs lake level — 90-day (fill) and 30-day (line) "
                  "rainfall anomaly; El Niño episodes shaded",
                  fontsize=11.5, loc="left", pad=8)

    ax2.plot(lake.index, lake.values, color="#2a7fb8", lw=1.3)
    ax2.set_ylabel("lake level (ft PLD)", fontsize=10)
    ax2.grid(axis="y", alpha=0.2)
    rec = lake.min()
    ax2.axhline(rec, color="#9c6b1e", lw=0.8, ls="--", alpha=0.7)
    ax2.text(lake.index[0], rec, f" period low {rec:.2f} ft", fontsize=7.5,
             color="#9c6b1e", va="bottom")
    ax1.margins(x=0.01); ax2.margins(x=0.01)
    fig.autofmt_xdate()
    fig.text(0.005, 0.005,
             "Basin box 9.0–9.6°N, 80.0–79.3°W. Rain: NASA POWER (GPM-based) daily history + "
             "IMERG Early basin means (recent), anomaly vs the series' own harmonic day-of-year normal. "
             "Lake: ACP Gatún history (ft PLD). El Niño shading: CPC ONI ≥ +0.5.",
             fontsize=7, color="#888")
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.1,
                pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)
    print(f"  wrote {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    anchor_h, anchor_day = IP._latest_anchors()
    print(f"  anchors: hourly→{anchor_h:%Y-%m-%d %HZ}, daily→{anchor_day:%Y-%m-%d}", flush=True)

    regions = build_totals(anchor_h, anchor_day)
    (ANIM_ROOT / "gatun_precip_manifest.json").write_text(json.dumps(
        {"ver": anchor_h.strftime("%Y%m%d%H"), "selectorLabel": "Accumulation",
         "default": DEFAULT_WINDOW, "regions": regions}))
    print(f"wrote {len(regions)} total windows + gatun_precip_manifest.json", flush=True)

    if CLIM_NC.exists():
        coef = xr.open_dataset(CLIM_NC)["coef"].values
        aregions = build_anoms(coef, anchor_day)
        (ANIM_ROOT / "gatun_anom_manifest.json").write_text(json.dumps(
            {"ver": anchor_day.strftime("%Y%m%d"), "selectorLabel": "Window",
             "default": DEFAULT_ANOM, "regions": aregions}))
        print(f"wrote {len(aregions)} anomaly windows + gatun_anom_manifest.json", flush=True)
    else:
        print(f"climatology {CLIM_NC.name} missing — anomaly loops skipped", flush=True)

    try:
        rain_level_chart(ASSETS / "gatun_rain_level.webp")
    except Exception as e:                                   # noqa: BLE001
        print(f"  rain/level chart failed ({repr(e)[:90]}); loops unaffected", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
