#!/usr/bin/env python3
"""SEAS5 10 m wind-event products over the United States (user 2026-09-08: "a higher frequency of
strong southerly flow events along the US east coast this winter ... some deviation from normal
metric for 10m winds above some threshold").

Data: the 6-hourly SEAS5 dataset's 10 m u and v over the US box (REGIONS["us"], 1°), every step, one
chunk per forecast month, forecast (51 members) and hindcast (25 members × 1993–2016). Per member and
month: days whose daily-MEAN southerly component v10 ≥ T (T in SOUTHERLY, m/s), days whose daily-MAX
6-hourly wind speed ≥ T (T in SPEED). Maps per month: ensemble-mean count as % of the hindcast's
count (ratio of normal frequency, hatched where the normal is under 0.5 d/month), the raw member
probability of ≥ 1 such day, and the monthly-mean v10 anomaly against the hindcast (m/s). Cities
carry the member counts for the page's clickable overlay.

    python seas5_wind.py fetch    [--issue 202609]     # forecast chunks
    python seas5_wind.py hindcast [--issue 202609]     # hindcast chunks (~6 × 300 MB)
    python seas5_wind.py build    [--issue 202609]
"""
from __future__ import annotations
import argparse, calendar, json, os, sys, time
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, CENTRE, SYSTEM, CLIM_YEARS, _client                     # noqa: E402
from seas5_popT import REGIONS, SIXH, month_hours                                          # noqa: E402
from seas5_extremes_build import CITIES, map_geometry, _CITY_PX                            # noqa: E402
from seas5_build import valid_months                                                       # noqa: E402

REGION = "us"; AREA = REGIONS["us"][2]
SOUTHERLY = [3, 4, 6]            # daily-mean v10 (m/s), positive = southerly; 10 m wind over land is weak, so start low
SPEED = [6, 8, 10]               # daily-max 6-hourly 10 m speed (m/s)
OUT_JSON = ASSETS / "data" / "seas5_wind.json"
PCT_LEVELS = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400]
PCT_COLORS = ["#8f2a0d", "#c8451c", "#e8703c", "#fbc39c", "#f4f4f1", "#dbe9f6", "#b7d2ec", "#8ab6df", "#5c95cd", "#3672b6"]
PROB_LEVELS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
PROB_COLORS = ["#f4f4f1", "#e3ecf5", "#c9dcee", "#a9c8e4", "#86b1d8", "#6197c8", "#437bb3", "#2f629c", "#214b82", "#163665", "#0c224a"]
V_LEVELS = [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3]
V_COLORS = ["#40004b", "#762a83", "#9970ab", "#c2a5cf", "#e7d4e8", "#f4f4f1", "#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#00441b"]


def fc_path(ym, k): return SIXH / f"{REGION}_{ym}_m{k}_w.grib"
def hc_path(ym, k): return SIXH / f"hc_{REGION}_{ym[4:]}_m{k}_w.grib"


def _retrieve(dest: Path, years, ym, k, label):
    if dest.exists() and dest.stat().st_size > 0: return True
    hours = [h for h in month_hours(ym, k) if int(h) <= 5160]
    req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
           "year": [str(y) for y in years], "month": [ym[4:]], "day": ["01"], "leadtime_hour": hours, "area": AREA, "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS 10 m wind {label} {ym} month {k} ({len(hours)} steps × {len(years)} yr) …", flush=True)
            _client().retrieve("seasonal-original-single-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest); print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True); return True
        except Exception as e:                                                   # noqa: BLE001
            print(f"    attempt {attempt + 1} failed ({str(e)[:120]})", flush=True); time.sleep(30)
    return False


def load(path: Path):
    """→ (u[sample, step, lat, lon], v[...], lat, lon, hours) for one chunk; samples = member (× year)."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    dims = [d for d in ds["u10"].dims if d not in ("latitude", "longitude", "step")]
    u = ds["u10"].stack(sample=dims).transpose("sample", "step", "latitude", "longitude").sortby("step") if len(dims) > 1 else ds["u10"].transpose(dims[0], "step", "latitude", "longitude").sortby("step")
    v = ds["v10"].stack(sample=dims).transpose("sample", "step", "latitude", "longitude").sortby("step") if len(dims) > 1 else ds["v10"].transpose(dims[0], "step", "latitude", "longitude").sortby("step")
    hours = (u.step.values / np.timedelta64(1, "h")).astype(int)
    return u.values.astype(np.float32), v.values.astype(np.float32), ds.latitude.values, ds.longitude.values, hours


def daily_stats(u, v, hours):
    """Group the 6-hourly steps into days (day d = the 06, 12, 18 and next-00 steps): daily-mean v10,
    daily-max speed → (vmean[sample, day, lat, lon], smax[...])."""
    day = ((hours - 1) // 24).astype(int); day -= day.min()
    nd = day.max() + 1; spd = np.hypot(u, v)
    vm = np.stack([v[:, day == d].mean(1) for d in range(nd)], 1); sm = np.stack([spd[:, day == d].max(1) for d in range(nd)], 1)
    return vm, sm


def draw_map(field, lat, lon, levels, colors, label, title, sub, out: Path, extend="max", hatch=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.patheffects as pe, textwrap
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    pc = ccrs.PlateCarree(); area, W, H, box = map_geometry(REGION)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes(box, projection=pc)
    ax.set_extent([area[1], area[3], area[2], area[0]], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    m = ax.pcolormesh(lon, lat, field, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)), transform=pc, shading="auto", zorder=1)
    if hatch is not None:
        ax.contourf(lon, lat, np.ma.masked_invalid(hatch), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc, zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2); ax.coastlines(resolution="50m", linewidth=0.5, color="#333", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#666", zorder=3); ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#999", zorder=3)
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]; ax.apply_aspect()
    for c in CITIES[REGION]:
        name, la, lo, side = c[:4]; dy = c[4] if len(c) > 4 else 0.0
        if name not in _CITY_PX.setdefault(REGION, {}):
            x, y = ax.transData.transform((lo, la)); _CITY_PX[REGION][name] = (round(float(x / fig.bbox.width), 4), round(float(1 - y / fig.bbox.height), 4))
        ax.plot(lo, la, "o", ms=4.2, mfc="#111", mec="white", mew=0.9, transform=pc, zorder=6)
        ax.text(lo + (0.55 if side == "r" else -0.55), la + dy, name, fontsize=8.8, fontweight="semibold", color="#111", ha="left" if side == "r" else "right", va="center", transform=pc, zorder=6, path_effects=halo)
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, "\n".join(textwrap.wrap(sub, int(W * 13))), fontsize=8.4, color="#444", va="top", linespacing=1.3)
    cax = fig.add_axes([0.25, 0.40 / H, 0.50, 0.14 / H]); cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=8.5); cb.ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)


def build(ym: str) -> dict:
    t0 = time.time(); vms = valid_months(ym); issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "thr_south": SOUTHERLY, "thr_speed": SPEED, "months": {}, "cities": {}}
    for c in CITIES[REGION]:
        doc["cities"][c[0]] = {"lat": c[1], "lon": c[2], "side": c[3], "months": {}}
    for k, vm in enumerate(vms, start=1):
        if not fc_path(ym, k).exists(): continue
        u, v, lat, lon, hours = load(fc_path(ym, k)); vm_f, sm_f = daily_stats(u, v, hours); del u, v
        vmean_f = vm_f.mean(1)                                                       # (member, lat, lon) monthly-mean v10
        hc = None
        if hc_path(ym, k).exists():
            try:
                uh, vh, hlat, hlon, hh = load(hc_path(ym, k)); vm_h, sm_h = daily_stats(uh, vh, hh); del uh, vh
                if hlat.shape == lat.shape and hlon.shape == lon.shape: hc = (vm_h, sm_h)
                print(f"    hindcast month {k}: {vm_h.shape[0]} samples", flush=True)
            except Exception as e:                                                   # noqa: BLE001
                print(f"    hindcast month {k} unreadable ({str(e)[:80]})", flush=True)
        mo = int(vm[5:]); plabel = f"{calendar.month_abbr[mo]} {vm[:4]}"; key = vm.replace("-", "_")
        rec = {"days": int(vm_f.shape[1]), "south": {}, "speed": {}, "vanom": None}
        ci = [(c[0], int(np.abs(lat - c[1]).argmin()), int(np.abs(lon - c[2]).argmin())) for c in CITIES[REGION]]
        sub = f"SEAS5 51 members, 6-hourly 10 m wind, 1° cells. {issue_lbl}."
        subr = f"Ensemble-mean count of days divided by the mean count in SEAS5's own 1993–2016 hindcast (25 members × 24 years, same start month and lead), so the model's wind bias cancels; hatched where the normal is under half a day a month. {issue_lbl}."
        for kind, thrs, stat_f, stat_h, what in (("south", SOUTHERLY, vm_f, hc[0] if hc else None, "daily-mean southerly 10 m wind ≥"),
                                                 ("speed", SPEED, sm_f, hc[1] if hc else None, "daily-max 10 m wind speed ≥")):
            for T in thrs:
                cnt = (stat_f >= T).sum(1).astype(np.float32); meanc = cnt.mean(0); pany = 100 * (cnt >= 1).mean(0)
                e = {}
                f1 = ASSETS / f"seas5_wind_{kind}_{T}_prob_{key}.webp"
                draw_map(pany, lat, lon, PROB_LEVELS, PROB_COLORS, "probability (%)", f"SEAS5 · chance of a day with {what} {T} m/s · {plabel}", sub, f1, extend="neither"); e["prob"] = f1.name
                if stat_h is not None:
                    hcnt = (stat_h >= T).sum(1).astype(np.float32).mean(0)
                    r = np.where(hcnt >= 0.5, np.clip(100 * meanc / np.maximum(hcnt, 1e-6), 0, 9999), np.nan)
                    f2 = ASSETS / f"seas5_wind_{kind}_{T}_pct_{key}.webp"
                    draw_map(r, lat, lon, PCT_LEVELS, PCT_COLORS, "% of the normal count", f"SEAS5 · days with {what} {T} m/s, % of normal · {plabel}", subr, f2, extend="max", hatch=np.where(hcnt < 0.5, 1.0, np.nan)); e["pct"] = f2.name
                    for name, i, j in ci:
                        doc["cities"][name]["months"].setdefault(vm, {}).setdefault(kind, {})[str(T)] = {"m": [int(x) for x in cnt[:, i, j]], "n": round(float(hcnt[i, j]), 2)}
                else:
                    for name, i, j in ci:
                        doc["cities"][name]["months"].setdefault(vm, {}).setdefault(kind, {})[str(T)] = {"m": [int(x) for x in cnt[:, i, j]], "n": None}
                rec[kind][str(T)] = e
        if hc is not None:
            va = vmean_f.mean(0) - hc[0].mean(1).mean(0); f3 = ASSETS / f"seas5_wind_vanom_{key}.webp"
            draw_map(va, lat, lon, V_LEVELS, V_COLORS, "m/s, southerly positive", f"SEAS5 · monthly-mean 10 m meridional wind anomaly · {plabel}",
                     f"Ensemble mean of 51 members minus the 1993–2016 hindcast mean for the same start month and lead; green = more southerly, purple = more northerly. {issue_lbl}.", f3, extend="both"); rec["vanom"] = f3.name
            for name, i, j in ci:
                doc["cities"][name]["months"][vm]["vanom"] = round(float(va[i, j]), 2)
        doc["months"][vm] = rec
        print(f"  {vm}: {vm_f.shape[1]} days · mean v10 anomaly {np.nanmean(vmean_f.mean(0) - (hc[0].mean(1).mean(0) if hc else 0)):+.2f} m/s", flush=True)
    for name, (fx, fy) in _CITY_PX.get(REGION, {}).items():
        doc["cities"][name]["x"], doc["cities"][name]["y"] = fx, fy
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fetch", "hindcast", "build"]); ap.add_argument("--issue", default=None)
    a = ap.parse_args()
    import datetime as _dt
    ym = a.issue or _dt.datetime.utcnow().strftime("%Y%m")
    if a.cmd == "fetch":
        ok = [_retrieve(fc_path(ym, k), [ym[:4]], ym, k, "forecast") for k in range(1, 7)]; sys.exit(0 if all(ok) else 1)
    if a.cmd == "hindcast":
        ok = [_retrieve(hc_path(ym, k), CLIM_YEARS, ym, k, "hindcast") for k in range(1, 7)]; sys.exit(0 if all(ok) else 1)
    build(ym)
