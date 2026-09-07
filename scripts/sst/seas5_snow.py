#!/usr/bin/env python3
"""SEAS5 snowfall products for North America (user 2026-09-07: "prob of 1 day snowfall above a certain
threshold and total snowfall products").

Data: the 6-hourly SEAS5 dataset's accumulated `snowfall` (m of water equivalent since the start) at
DAILY leadtimes for 6 months over 75N–25N, 170W–50W on the 1° grid, 51 members — one CDS request per
issue (data/seas5/sixh/na_{ym}_sf.grib, ~120 MB). Differencing consecutive days gives every member's
daily snowfall; a 10:1 snow-to-liquid ratio turns mm w.e. into cm of snow (stated on the figures).
Products per forecast month:
  * P(at least one day with ≥ T cm)      T in SNOW1D   → seas5_snow1d_{T}_{YYYY_MM}.webp
  * ensemble-mean monthly total (cm)                   → seas5_snowtot_mean_{YYYY_MM}.webp
  * P(monthly total ≥ T cm)              T in SNOWTOT  → seas5_snowtot_{T}_{YYYY_MM}.webp
  * monthly total as % of the SEAS5 1993–2016 hindcast mean (the `na_snow` monthly means the maps
    viewer already fetches; a model-consistent normal, since no gridded snowfall observation normal
    is on hand)                                          → seas5_snowtot_pct_{YYYY_MM}.webp
plus assets/sst/data/seas5_snow.json with the population-weighted table and, for the page's clickable
cities, every member's monthly total and largest 1-day fall at each city's cell. Members are NOT bias
corrected (no daily snowfall hindcast fetched); the % of normal map is the calibrated read.

    python seas5_snow.py fetch [--issue 202609]
    python seas5_snow.py build [--issue 202609]
"""
from __future__ import annotations
import argparse, calendar, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, DATA, CENTRE, SYSTEM, _client, fc_path, hc_path   # noqa: E402
from seas5_build import load_field, valid_months                                       # noqa: E402
from seas5_popT import SIXH, month_hours                                                          # noqa: E402
from seas5_extremes_build import CITIES                                                          # noqa: E402

AREA = [75, -170, 25, -50]                       # N, W, S, E — the na_snow box
SNOW1D = [5, 10, 20, 30]                         # cm in one day
SNOWTOT = [10, 25, 50, 100]                      # cm in the month
RATIO = 10.0                                     # cm of snow per cm of water equivalent
OUT_JSON = ASSETS / "data" / "seas5_snow.json"
MAP_W = 11.0


def sf_path(ym: str) -> Path:
    return SIXH / f"na_{ym}_sf.grib"


def fetch(ym: str) -> bool:
    dest = sf_path(ym)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    hours = sorted({int(h) for k in range(1, 7) for h in month_hours(ym, k) if int(h) % 24 == 0 and int(h) <= 5160})
    req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": ["snowfall"],
           "year": [ym[:4]], "month": [ym[4:]], "day": ["01"], "leadtime_hour": [str(h) for h in hours],
           "area": AREA, "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS snowfall NA {ym} ({len(hours)} daily steps) …", flush=True)
            _client().retrieve("seasonal-original-single-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest); print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                                                   # noqa: BLE001
            print(f"    attempt {attempt + 1} failed ({str(e)[:120]})", flush=True); time.sleep(30)
    return False


def load_daily(ym: str):
    """(member, day, lat, lon) daily snowfall in cm of snow, plus the valid date of each day."""
    import xarray as xr, pandas as pd
    ds = xr.open_dataset(sf_path(ym), engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = ds[[k for k in ds.data_vars][0]]                                 # sf, m w.e. accumulated
    v = v.sortby("step")
    acc = v.values                                                       # (number, step, lat, lon)
    if acc.ndim == 3: acc = acc[None]
    daily = np.diff(np.concatenate([np.zeros_like(acc[:, :1]), acc], axis=1), axis=1)
    daily = np.clip(daily, 0, None) * 1000.0 / 10.0 * RATIO              # m w.e. → mm w.e. → cm w.e. → cm snow
    init = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")
    dates = pd.DatetimeIndex([init + pd.Timedelta(hours=int(h)) - pd.Timedelta(hours=1) for h in (v.step.values / np.timedelta64(1, "h")).astype(int)])
    return daily.astype(np.float32), dates, v.latitude.values, v.longitude.values


def map_box():
    W = MAP_W; H = W * (AREA[0] - AREA[2]) / (AREA[3] - AREA[1]) + 2.1
    return W, H, [0.03, 0.95 / H, 0.94, (H - 2.1) / H]


_PX = {}


def draw_map(field, lat, lon, levels, colors, label, title, sub, out: Path, extend="max", hatch=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, textwrap
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    pc = ccrs.PlateCarree(); W, H, box = map_box()
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes(box, projection=pc)
    ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    m = ax.pcolormesh(lon, lat, field, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)), transform=pc, shading="auto", zorder=1)
    if hatch is not None:
        ax.contourf(lon, lat, np.ma.masked_invalid(hatch), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.5, color="#333", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#666", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#999", zorder=3)
    ax.apply_aspect()
    for city in CITIES["us"]:                        # no labels or dots on these maps (user: "not necessary here");
        name, la, lo, side = city[:4]                # the page overlays its own clickable markers at these fractions
        if name not in _PX:
            x, y = ax.transData.transform((lo, la)); _PX[name] = (round(float(x / fig.bbox.width), 4), round(float(1 - y / fig.bbox.height), 4))
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, "\n".join(textwrap.wrap(sub, int(W * 13))), fontsize=8.4, color="#444", va="top", linespacing=1.3)
    cax = fig.add_axes([0.25, 0.40 / H, 0.50, 0.14 / H]); cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=8.5); cb.ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)


PROB_LEVELS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
PROB_COLORS = ["#f4f4f1", "#e3ecf5", "#c9dcee", "#a9c8e4", "#86b1d8", "#6197c8", "#437bb3", "#2f629c", "#214b82", "#163665", "#0c224a"]
TOT_LEVELS = [0, 1, 2.5, 5, 10, 15, 25, 40, 60, 90, 130, 200, 300]
TOT_COLORS = ["#f4f4f1", "#e8f0f7", "#d3e2f1", "#b6d0e8", "#94bcdd", "#6fa4cf", "#4f8bc0", "#396fa8", "#2b578c", "#213f6d", "#1a2e52", "#12203a"]
PCT_LEVELS = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400]
PCT_COLORS = ["#8f2a0d", "#c8451c", "#e8703c", "#fbc39c", "#f4f4f1", "#dbe9f6", "#b7d2ec", "#8ab6df", "#5c95cd", "#3672b6"]


def build(ym: str) -> dict:
    t0 = time.time()
    daily, dates, lat, lon = load_daily(ym)
    issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    ci = [(c[0], int(np.abs(lat - c[1]).argmin()), int(np.abs(lon - c[2]).argmin())) for c in CITIES["us"]]
    # hindcast normal: monthly-mean snowfall rate (m w.e. s-1) × seconds → cm snow, per forecast month
    normal = None
    if fc_path("na_snow", ym).exists() and hc_path("na_snow", ym[4:]).exists():
        hc, hlat, hlon = load_field(hc_path("na_snow", ym[4:]), "mtsfr")           # (samples, lead, lat, lon)
        if hlat.shape == lat.shape and hlon.shape == lon.shape and np.allclose(hlat, lat) and np.allclose(hlon, lon):
            normal = np.nanmean(hc, axis=0)                                          # (lead, lat, lon)
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "ratio": RATIO, "thr1d": SNOW1D, "thrtot": SNOWTOT,
           "months": {}, "cities": {}}
    for city in CITIES["us"]:
        name, la, lo, side = city[:4]
        doc["cities"][name] = {"lat": la, "lon": lo, "side": side, "months": {}}
    vms = valid_months(ym)
    for k, vm in enumerate(vms[:6], start=1):
        sel = (dates.year == int(vm[:4])) & (dates.month == int(vm[5:]))
        if sel.sum() < 20: continue
        d = daily[:, sel]                                                            # (member, day, lat, lon)
        mx = d.max(axis=1); tot = d.sum(axis=1)                                      # (member, lat, lon)
        mo = int(vm[5:]); plabel = f"{calendar.month_abbr[mo]} {vm[:4]}"; key = vm.replace("-", "_")
        rec = {"days": int(sel.sum()), "snow1d": {}, "snowtot": {}, "mean_total": None, "pct": None}
        sub = f"SEAS5 51 members, daily snowfall from the accumulated field, {RATIO:.0f}:1 snow-to-liquid ratio, 1° cells, no bias correction. {issue_lbl}."
        for T in SNOW1D:
            p = 100.0 * (mx >= T).mean(axis=0); out = ASSETS / f"seas5_snow1d_{T}_{key}.webp"
            draw_map(p, lat, lon, PROB_LEVELS, PROB_COLORS, "probability (%)", f"SEAS5 · chance of a day with ≥ {T} cm of snow · {plabel}", sub, out, extend="neither")
            rec["snow1d"][str(T)] = out.name
        mean_tot = tot.mean(axis=0); out = ASSETS / f"seas5_snowtot_mean_{key}.webp"
        draw_map(mean_tot, lat, lon, TOT_LEVELS, TOT_COLORS, "cm of snow", f"SEAS5 · ensemble-mean monthly snowfall · {plabel}", sub, out, extend="max")
        rec["mean_total"] = out.name
        for T in SNOWTOT:
            p = 100.0 * (tot >= T).mean(axis=0); out = ASSETS / f"seas5_snowtot_{T}_{key}.webp"
            draw_map(p, lat, lon, PROB_LEVELS, PROB_COLORS, "probability (%)", f"SEAS5 · chance the month's snowfall reaches {T} cm · {plabel}", sub, out, extend="neither")
            rec["snowtot"][str(T)] = out.name
        if normal is not None and k - 1 < normal.shape[0]:
            nrm = normal[k - 1] * 86400.0 * calendar.monthrange(int(vm[:4]), mo)[1] * 100.0 * RATIO    # m w.e./s → cm snow per month
            pct = np.where(nrm >= 1.0, np.clip(100.0 * mean_tot / np.maximum(nrm, 1e-6), 0, 9999), np.nan)
            out = ASSETS / f"seas5_snowtot_pct_{key}.webp"
            draw_map(pct, lat, lon, PCT_LEVELS, PCT_COLORS, "% of the hindcast normal", f"SEAS5 · monthly snowfall as % of the model's 1993–2016 normal · {plabel}",
                     f"Ensemble-mean monthly snowfall against the mean of SEAS5's own 1993–2016 hindcast for the same start month and lead (25 members × 24 years), so the model's snowfall bias cancels. Hatched: normal under 1 cm a month. {issue_lbl}.",
                     out, extend="max", hatch=np.where(nrm < 1.0, 1.0, np.nan))
            rec["pct"] = out.name
        for name, i, j in ci:
            doc["cities"][name]["months"][vm] = {"tot": [round(float(v), 1) for v in tot[:, i, j]], "max1d": [round(float(v), 1) for v in mx[:, i, j]],
                                                 "normal": (round(float(normal[k - 1][i, j] * 86400.0 * calendar.monthrange(int(vm[:4]), mo)[1] * 100.0 * RATIO), 1) if normal is not None and k - 1 < normal.shape[0] else None)}
        doc["months"][vm] = rec
        print(f"  {vm}: {sel.sum()} days, mean total at Chicago {tot[:, ci[12][1], ci[12][2]].mean():.1f} cm", flush=True)
    for name, (fx, fy) in _PX.items():
        doc["cities"][name]["x"], doc["cities"][name]["y"] = fx, fy
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fetch", "build"]); ap.add_argument("--issue", default=None)
    a = ap.parse_args()
    import datetime as _dt
    ym = a.issue or _dt.datetime.utcnow().strftime("%Y%m")
    if a.cmd == "fetch":
        sys.exit(0 if fetch(ym) else 1)
    build(ym)
